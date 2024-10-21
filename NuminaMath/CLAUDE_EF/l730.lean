import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l730_73054

theorem smallest_number_divisible (n : ℕ) : n = 82 ↔ 
  (∀ d ∈ ({2, 6, 12, 24, 36} : Set ℕ), (n - 10) % d = 0) ∧
  (∀ m : ℕ, m < n → ∃ d ∈ ({2, 6, 12, 24, 36} : Set ℕ), (m - 10) % d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l730_73054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l730_73019

theorem price_change_theorem (original_price : ℝ) (original_price_pos : 0 < original_price) :
  original_price * (1 + 0.15) * (1 - 0.15) = original_price * 0.9775 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l730_73019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_chip_count_l730_73024

theorem blue_chip_count (n m : ℕ) : 
  (n > m) →                           -- More blue chips than green chips
  (2 < n + m) →                       -- Total chips greater than 2
  (n + m < 50) →                      -- Total chips less than 50
  ((n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1)) = 
   (2 * n * m) / ((n + m) * (n + m - 1))) →  -- Equal probabilities condition
  (n ∈ ({3, 6, 10, 15, 21, 28} : Set ℕ)) :=
by
  sorry

#check blue_chip_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_chip_count_l730_73024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l730_73088

-- Define the points
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (20/3, 0)
noncomputable def B : ℝ × ℝ := (0, 20)
noncomputable def C : ℝ × ℝ := (10, 0)
noncomputable def E : ℝ × ℝ := (5, 5)

-- Define the slope of the first line
def m : ℝ := -3

-- Define the area function for a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem area_of_quadrilateral_OBEC :
  triangle_area O B E + triangle_area O E C = 275/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l730_73088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_l730_73079

/-- Represents a square dartboard with side length 2 units -/
structure Dartboard where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- Represents a triangular section of the dartboard -/
structure TriangularSection where
  area : ℝ

/-- A triangular section touching a vertex of the square -/
noncomputable def vertex_triangle (d : Dartboard) : TriangularSection :=
  { area := 1 / 4 }

/-- A triangular section not touching a vertex of the square -/
noncomputable def midpoint_triangle (d : Dartboard) : TriangularSection :=
  { area := 1 / 4 }

/-- The theorem stating that the ratio of areas is 1 -/
theorem area_ratio_is_one (d : Dartboard) :
  (midpoint_triangle d).area / (vertex_triangle d).area = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_l730_73079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_B_multiple_l730_73091

-- Define the production rates of machines A and B
noncomputable def rate_A (x : ℝ) : ℝ := x / 10
noncomputable def rate_B (x : ℝ) (m : ℝ) : ℝ := m * x / 5

-- Define the combined production rate
noncomputable def combined_rate (x : ℝ) (m : ℝ) : ℝ := rate_A x + rate_B x m

-- Theorem statement
theorem machine_B_multiple (x : ℝ) (m : ℝ) (h1 : x > 0) :
  combined_rate x m = x / 2 → m = 2 := by
  intro h
  unfold combined_rate rate_A rate_B at h
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_B_multiple_l730_73091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_floor_equation_l730_73059

-- Define the fractional part function as noncomputable
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem largest_solution_floor_equation :
  ∃ (x : ℝ), ⌊x⌋ = 5 + 100 * (frac x) ∧
  ∀ (y : ℝ), ⌊y⌋ = 5 + 100 * (frac y) → y ≤ x ∧
  x = 104.99 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_floor_equation_l730_73059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_is_one_to_e_l730_73037

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a)

-- State the theorem
theorem range_of_a_is_one_to_e :
  ∃ (a : ℝ), (∀ x ∈ Set.Icc 1 (Real.exp 1), ∃ (x₀ y₀ : ℝ),
    y₀ = Real.sin x₀ ∧ f a (f a y₀) = y₀) ∧
  (∀ a' : ℝ, a' < 1 ∨ a' > Real.exp 1 →
    ¬∃ (x₀ y₀ : ℝ), y₀ = Real.sin x₀ ∧ f a' (f a' y₀) = y₀) :=
by sorry

-- Note: Set.Icc 1 (Real.exp 1) represents the closed interval [1, e]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_is_one_to_e_l730_73037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_L_expression_l730_73044

/-- The side length of the cube in meters -/
noncomputable def cube_side_length : ℝ := 3

/-- The surface area of the cube in square meters -/
noncomputable def cube_surface_area : ℝ := 6 * cube_side_length ^ 2

/-- The radius of the sphere in meters -/
noncomputable def sphere_radius : ℝ := (cube_surface_area / (4 * Real.pi)) ^ (1/2)

/-- The volume of the sphere in cubic meters -/
noncomputable def sphere_volume : ℝ := (4/3) * Real.pi * sphere_radius ^ 3

/-- The value L in the expression of the sphere's volume -/
noncomputable def L : ℝ := (sphere_volume * Real.pi.sqrt) / Real.sqrt 15

theorem sphere_volume_equals_L_expression : L = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_equals_L_expression_l730_73044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_cannot_reach_opposite_corner_l730_73004

/-- Represents a point in the 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the state of the light beam --/
structure BeamState where
  position : Point
  direction : Bool  -- True for upward diagonal, False for downward diagonal

/-- The room configuration --/
def Room (m : ℕ) : Type :=
  { p : Point // p.x < 2*m ∧ p.y < 2*m }

/-- The next position of the beam after one step --/
def nextPosition (state : BeamState) : BeamState :=
  if state.direction then
    { position := { x := state.position.x + 1, y := state.position.y + 1 }, direction := state.direction }
  else
    { position := { x := state.position.x + 1, y := state.position.y - 1 }, direction := state.direction }

/-- Checks if a point is on an edge of the room --/
def isEdge (m : ℕ) (p : Point) : Bool :=
  p.x = 0 ∨ p.y = 0 ∨ p.x = 2*m ∨ p.y = 2*m

/-- The main theorem --/
theorem light_cannot_reach_opposite_corner (m : ℕ) (h : m > 0) :
  ∀ n : ℕ, (Nat.iterate nextPosition n { position := { x := 0, y := 0 }, direction := true }).position ≠ { x := 2*m, y := 2*m } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_cannot_reach_opposite_corner_l730_73004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_sum_arithmetic_sequence_l730_73038

/-- An arithmetic sequence with first term a₁ and common difference d -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ

/-- The nth term of an arithmetic sequence -/
noncomputable def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + seq.d * (n - 1)

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a₁ + (n - 1) * seq.d) / 2

/-- The condition that the quadratic inequality has solution set [0, 22] -/
def hasSpecificSolutionSet (seq : ArithmeticSequence) (c : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 22 ↔ (seq.d / 2) * x^2 + (seq.a₁ - seq.d / 2) * x + c ≥ 0

theorem maximize_sum_arithmetic_sequence (seq : ArithmeticSequence) (c : ℝ)
    (h : hasSpecificSolutionSet seq c) :
    ∀ n : ℕ, n > 0 → seq.sumFirstN n ≤ seq.sumFirstN 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_sum_arithmetic_sequence_l730_73038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_volume_ratio_l730_73081

theorem sphere_triangle_volume_ratio 
  (a b c r : ℝ) 
  (h_triangle : a^2 + b^2 = c^2) 
  (h_a : a = 8) 
  (h_b : b = 15) 
  (h_c : c = 17) 
  (h_r : r = 5) : 
  (let s := (a + b + c) / 2
   let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
   let inscribed_radius := (2 * area) / (a + b + c)
   let d := Real.sqrt (r^2 - inscribed_radius^2)
   let h := r - d
   let segment_volume := (h * (3 * inscribed_radius^2 + h^2) * Real.pi) / 6
   let sphere_volume := (4 * Real.pi * r^3) / 3
   let remaining_volume := sphere_volume - segment_volume
   segment_volume / remaining_volume) = 7 / 243 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_volume_ratio_l730_73081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_range_l730_73049

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x + a + 1 else a*x + 4

theorem function_equality_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a ∈ Set.Ioi 4 ∪ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_range_l730_73049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l730_73076

/-- The distance from a point (x, y) to a line y = k -/
def distToLine (x y k : ℝ) : ℝ := |y - k|

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distBetweenPoints (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem: If a point M(x, y) satisfies the given condition, then it lies on the parabola x^2 = 8y -/
theorem point_on_parabola (x y : ℝ) :
  distBetweenPoints x y 0 2 = distToLine x y (-4) - 2 →
  x^2 = 8 * y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l730_73076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_changes_l730_73083

-- Define the type for our dataset
def Dataset := List ℝ

-- Define functions for median, mean, and variance
noncomputable def median (data : Dataset) : ℝ := sorry
def mean (data : Dataset) : ℝ := sorry
def variance (data : Dataset) : ℝ := sorry

-- Define a property for a value being significantly larger than all elements in a dataset
def significantlyLarger (x : ℝ) (data : Dataset) : Prop := sorry

-- Main theorem
theorem dataset_changes 
  (n : ℕ) 
  (hn : n ≥ 3) 
  (data : Dataset) 
  (hdata : data.length = n) 
  (x_new : ℝ) 
  (h_larger : significantlyLarger x_new data) :
  let new_data := x_new :: data
  (mean new_data > mean data) ∧ 
  (median new_data = median data ∨ median new_data > median data) ∧
  (variance new_data > variance data) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_changes_l730_73083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_p_and_q_range_when_p_or_q_l730_73053

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, (0 < a ∧ a < 1 ∨ a > 1) → (a^x > 1 ↔ x < 0)

def q (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a*x^2 - x + a)

-- Theorem 1
theorem range_when_p_and_q (a : ℝ) : p a ∧ q a → 1/2 < a ∧ a < 1 :=
by sorry

-- Theorem 2
theorem range_when_p_or_q (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → (0 < a ∧ a ≤ 1/2) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_p_and_q_range_when_p_or_q_l730_73053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l730_73090

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A line in 3D space -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance from a point to a plane -/
noncomputable def distancePointToPlane (plane : Plane) (point : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := point
  |plane.a * x + plane.b * y + plane.c * z + plane.d| /
    Real.sqrt (plane.a^2 + plane.b^2 + plane.c^2)

/-- Check if a plane contains a line -/
def planeContainsLine (plane : Plane) (line : Line) : Prop :=
  let (x, y, z) := line.point
  let (dx, dy, dz) := line.direction
  plane.a * x + plane.b * y + plane.c * z + plane.d = 0 ∧
  plane.a * dx + plane.b * dy + plane.c * dz = 0

theorem plane_equation (p1 p2 p : Plane) (l : Line) :
  p1.a = 1 ∧ p1.b = 1 ∧ p1.c = 2 ∧ p1.d = -4 ∧
  p2.a = 2 ∧ p2.b = -1 ∧ p2.c = 1 ∧ p2.d = -1 ∧
  planeContainsLine p1 l ∧
  planeContainsLine p2 l ∧
  planeContainsLine p l ∧
  p ≠ p1 ∧ p ≠ p2 ∧
  distancePointToPlane p (2, 2, -2) = 5 / Real.sqrt 6 ∧
  p.a > 0 ∧
  Nat.gcd (Int.natAbs (Int.floor p.a)) (Int.natAbs (Int.floor p.b)) = 1 ∧
  Nat.gcd (Int.natAbs (Int.floor p.a)) (Int.natAbs (Int.floor p.c)) = 1 ∧
  Nat.gcd (Int.natAbs (Int.floor p.a)) (Int.natAbs (Int.floor p.d)) = 1 →
  p.a = 1 ∧ p.b = 1 ∧ p.c = 2 ∧ p.d = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l730_73090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opal_savings_bet_ratio_l730_73075

noncomputable def initial_winnings : ℝ := 100

-- S represents the amount put into savings after the first win
-- B represents the amount bet after the first win
def savings_bet_sum (S B : ℝ) : Prop := S + B = initial_winnings

noncomputable def profit_rate : ℝ := 0.6

noncomputable def total_savings (S B : ℝ) : ℝ := S + (1/2) * (1 + profit_rate) * B

theorem opal_savings_bet_ratio (S B : ℝ) 
  (h1 : savings_bet_sum S B) 
  (h2 : total_savings S B = 90) : 
  S = B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opal_savings_bet_ratio_l730_73075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l730_73027

theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ (3 * Real.sqrt 3) / 2 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = (3 * Real.sqrt 3) / 2 ↔ 
   A = π / 9 ∧ B = π / 9 ∧ C = 7 * π / 9) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l730_73027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l730_73025

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l730_73025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_l730_73042

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + b < 0}
def C : Set ℝ := {x : ℝ | x^3 + x^2 + x = 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (Set.univ \ (A ∪ B a b) = C) ∧
    (A ∩ B a b = {x : ℝ | 2 < x ∧ x < 4}) ∧
    a = 4 ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_l730_73042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l730_73014

/-- The area enclosed by a figure formed from 8 congruent circular arcs,
    each centered on a vertex of a regular pentagon with side length 2
    and arc length 3π/4 -/
noncomputable def enclosed_area (num_arcs : ℕ) (pentagon_side : ℝ) (arc_length : ℝ) : ℝ :=
  sorry

/-- The side length of the regular pentagon -/
def pentagon_side : ℝ := 2

/-- The number of congruent circular arcs -/
def num_arcs : ℕ := 8

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := 3 * Real.pi / 4

/-- Theorem stating that the enclosed area is equal to 4.8284 + 3π -/
theorem enclosed_area_theorem :
  enclosed_area num_arcs pentagon_side arc_length = 4.8284 + 3 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l730_73014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l730_73095

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def has_four_consecutive_terms (b : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, (b n ∈ ({-53, -23, 19, 37, 82} : Set ℝ)) ∧ 
            (b (n + 1) ∈ ({-53, -23, 19, 37, 82} : Set ℝ)) ∧ 
            (b (n + 2) ∈ ({-53, -23, 19, 37, 82} : Set ℝ)) ∧ 
            (b (n + 3) ∈ ({-53, -23, 19, 37, 82} : Set ℝ))

theorem geometric_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (∀ n : ℕ, b n = a n + 1) →
  abs q > 1 →
  has_four_consecutive_terms b →
  6 * q = -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l730_73095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solutions_l730_73056

noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1
  else if a < 0 then -1
  else 0

def satisfies_equations (x y z : ℝ) : Prop :=
  x = 3021 - 3022 * sign (y*z) ∧
  y = 3021 - 3022 * sign (x*z) ∧
  z = 3021 - 3022 * sign (x*y)

theorem unique_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    solutions.card = 5 ∧
    ∀ (triple : ℝ × ℝ × ℝ), triple ∈ solutions ↔ satisfies_equations triple.1 triple.2.1 triple.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solutions_l730_73056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunlight_creates_parallel_projection_l730_73043

/-- A light source that emits parallel rays of light. -/
structure ParallelLightSource where
  emitsParallelRays : Bool

/-- A light source that emits light from a single point. -/
structure PointLightSource where
  emitsFromPoint : Bool

/-- A projection created by a light source. -/
inductive Projection
  | Parallel
  | Central

/-- Creates a projection based on the type of light source. -/
def createProjection (light : ParallelLightSource ⊕ PointLightSource) : Projection :=
  match light with
  | Sum.inl parallelSource => 
      if parallelSource.emitsParallelRays then Projection.Parallel else Projection.Central
  | Sum.inr pointSource => Projection.Central

/-- Sunlight is considered to travel in parallel lines when it reaches Earth. -/
def sunlight_parallel : ParallelLightSource := {
  emitsParallelRays := true
}

theorem sunlight_creates_parallel_projection :
  createProjection (Sum.inl sunlight_parallel) = Projection.Parallel := by
  simp [createProjection, sunlight_parallel]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunlight_creates_parallel_projection_l730_73043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_coefficient_sum_l730_73065

theorem factorization_coefficient_sum : 
  ∃ (a b c d e f g h j k : ℤ),
    (125 : ℤ) * X^6 - 216 * Z^6 = 
    (a * X + b * Z) * (c * X^2 + d * X * Z + e * Z^2) * 
    (f * X + g * Z) * (h * X^2 + j * X * Z + k * Z^2) ∧
    a + b + c + d + e + f + g + h + j + k = 90 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_coefficient_sum_l730_73065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_cans_required_l730_73034

def drink_quantities : List ℕ := [127, 251, 163, 193, 107]

theorem least_cans_required (quantities : List ℕ) : 
  (quantities.foldl Nat.gcd 0 = 1) → quantities.sum = quantities.sum
    := by
  intro h
  rfl

#eval drink_quantities.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_cans_required_l730_73034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_implies_cos_value_l730_73030

theorem sin_double_angle_implies_cos_value (α : ℝ) 
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : 0 < α)
  (h3 : α < π / 2) :
  Real.sqrt 2 * Real.cos (π / 4 - α) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_implies_cos_value_l730_73030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_6_l730_73005

/-- Calculates the overall average speed of a journey with cycling, stopping, and walking segments -/
noncomputable def overall_average_speed (
  cycling_time : ℝ) 
  (cycling_speed : ℝ) 
  (stopping_time : ℝ) 
  (walking_time : ℝ) 
  (walking_speed : ℝ) : ℝ :=
  let total_distance := cycling_time / 60 * cycling_speed + walking_time / 60 * walking_speed
  let total_time := (cycling_time + stopping_time + walking_time) / 60
  total_distance / total_time

/-- Proves that the overall average speed is approximately 6 mph given the specified conditions -/
theorem average_speed_approx_6 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |overall_average_speed 45 12 15 75 3 - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_6_l730_73005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l730_73031

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.log x - x^2
noncomputable def g (x m : ℝ) := (x - 2)^2 - 1 / (2 * x - 4) - m

-- Define the symmetry condition
def symmetric_about_point (f g : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, f (a + x) = g (a - x) + b

-- Theorem statement
theorem range_of_m :
  ∃ m_min : ℝ, m_min = 1 - Real.log 2 ∧
  (∀ m : ℝ, (∃ x : ℝ, symmetric_about_point f (g · m) 1 0) ↔ m ≥ m_min) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l730_73031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delegates_without_badges_l730_73020

theorem delegates_without_badges (total : ℕ) (preprinted : ℕ) : 
  total = 45 → preprinted = 16 → 
  (∃ (break_count handwritten without : ℕ), 
    break_count = total / 3 ∧ 
    handwritten = (total - break_count) / 4 ∧ 
    without = (total - break_count) - handwritten ∧ 
    without + (preprinted - break_count) = 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delegates_without_badges_l730_73020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_inequality_l730_73017

/-- Represents the number of paths between two ports in a network -/
def path_count (i j : Fin 6) : ℕ := sorry

/-- Theorem stating the inequality for path counts in a 6-port network -/
theorem path_count_inequality :
  path_count 1 6 * path_count 2 5 * path_count 3 4 +
  path_count 1 5 * path_count 2 4 * path_count 3 6 +
  path_count 1 4 * path_count 2 6 * path_count 3 5 ≥
  path_count 1 6 * path_count 2 4 * path_count 3 5 +
  path_count 1 5 * path_count 2 6 * path_count 3 4 +
  path_count 1 4 * path_count 2 5 * path_count 3 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_inequality_l730_73017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_100_terms_l730_73084

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + (n + 1)

def sequence_b (n : ℕ) : ℚ := 1 / sequence_a (n + 1)

theorem sum_of_first_100_terms :
  (Finset.range 100).sum sequence_b = 200 / 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_100_terms_l730_73084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l730_73041

def sequenceA (n : ℕ+) : ℚ :=
  (-1 : ℚ)^(n.val + 1) * (1 : ℚ) / ((n.val : ℚ)^2 + 1)

theorem sequence_properties :
  (sequenceA 8 = -1/65) ∧
  (sequenceA 1 = 1/2) ∧
  (sequenceA 2 = -1/5) ∧
  (sequenceA 3 = 1/10) ∧
  (sequenceA 4 = -1/17) ∧
  (sequenceA 5 = 1/26) ∧
  (sequenceA 6 = -1/37) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l730_73041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l730_73010

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l₁ l₂ : Line) : ℝ :=
  abs (l₁.c - l₂.c) / Real.sqrt (l₁.a^2 + l₁.b^2)

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a ∧ l₁.a * l₂.c ≠ l₁.c * l₂.a

theorem parallel_lines_distance :
  ∀ a : ℝ,
  let l₁ : Line := ⟨1, a, 6⟩
  let l₂ : Line := ⟨a - 2, 3, 2 * a⟩
  parallel l₁ l₂ → distance_between_parallel_lines l₁ l₂ = 8 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l730_73010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_for_h_l730_73029

/-- The displacement function h(t) = t^2 + 1 -/
noncomputable def h (t : ℝ) : ℝ := t^2 + 1

/-- The average velocity function for a given displacement function and time interval -/
noncomputable def averageVelocity (h : ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ :=
  (h t₂ - h t₁) / (t₂ - t₁)

theorem average_velocity_for_h (Δt : ℝ) (h_pos : Δt > 0) :
  averageVelocity h 1 (1 + Δt) = 2 + Δt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_for_h_l730_73029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l730_73013

-- Define the job as a unit of work
noncomputable def job : ℝ := 1

-- Define B's work rate
noncomputable def B_rate : ℝ := 1 / 19.5

-- Define A's work rate as half of B's
noncomputable def A_rate : ℝ := B_rate / 2

-- Define the combined work rate of A and B
noncomputable def combined_rate : ℝ := A_rate + B_rate

-- Theorem to prove
theorem job_completion_time :
  (job / combined_rate) = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l730_73013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distances_product_constant_hyperbola_min_distance_to_point_l730_73055

/-- The hyperbola C: x²/5 - y² = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 5 - y^2 = 1

/-- The first asymptote of the hyperbola -/
def asymptote1 (x y : ℝ) : Prop := x - Real.sqrt 5 * y = 0

/-- The second asymptote of the hyperbola -/
def asymptote2 (x y : ℝ) : Prop := x + Real.sqrt 5 * y = 0

/-- The distance from a point (x, y) to a line ax + by + c = 0 -/
noncomputable def distance_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The distance between two points (x₁, y₁) and (x₂, y₂) -/
noncomputable def distance_between_points (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem hyperbola_distances_product_constant (x₀ y₀ : ℝ) :
  hyperbola x₀ y₀ →
  distance_to_line x₀ y₀ 1 (-Real.sqrt 5) 0 *
  distance_to_line x₀ y₀ 1 (Real.sqrt 5) 0 = 5/6 := by
  sorry

theorem hyperbola_min_distance_to_point (x₀ y₀ : ℝ) :
  hyperbola x₀ y₀ →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 15 / 3 ∧
  ∀ (x y : ℝ), hyperbola x y →
  distance_between_points x y 4 0 ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distances_product_constant_hyperbola_min_distance_to_point_l730_73055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l730_73036

/-- The lower bound of the interval -/
def lower_bound : ℝ := -30

/-- The upper bound of the interval -/
def upper_bound : ℝ := 15

/-- The probability of selecting a number from a subinterval of [lower_bound, upper_bound] -/
noncomputable def prob_subinterval (a b : ℝ) : ℝ :=
  (b - a) / (upper_bound - lower_bound)

/-- The probability that the product of two independently and uniformly selected
    real numbers from [lower_bound, upper_bound] is greater than zero -/
noncomputable def prob_product_positive : ℝ :=
  prob_subinterval 0 upper_bound ^ 2 + prob_subinterval lower_bound 0 ^ 2

theorem product_positive_probability :
  prob_product_positive = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l730_73036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_third_quadrant_l730_73003

theorem tan_double_angle_third_quadrant (α : Real) :
  (α ∈ Set.Icc π (3 * π / 2)) →  -- α is in the third quadrant
  (Real.cos (α + π) = 4 / 5) →   -- given condition
  (Real.tan (2 * α) = 24 / 7) := -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_third_quadrant_l730_73003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l730_73063

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The distance between foci of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := 2 * Real.sqrt (e.a^2 - e.b^2)

/-- A line represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
noncomputable def point_line_distance (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

theorem ellipse_properties (e : Ellipse) 
  (h1 : e.a^2 + e.b^2 = 2 * (e.a^2 - e.b^2))
  (h2 : point_line_distance 0 0 ⟨e.b, e.a, -e.a*e.b⟩ = 3*Real.sqrt 3/2)
  (h3 : ∃ (m n : ℝ × ℝ), 
    (m.1^2 / e.a^2 + m.2^2 / e.b^2 = 1) ∧
    (n.1^2 / e.a^2 + n.2^2 / e.b^2 = 1) ∧
    (m.1 + n.1 = -4) ∧ (m.2 + n.2 = 2)) :
  (eccentricity e = Real.sqrt 6 / 3) ∧
  (e.a = 3 * Real.sqrt 3 ∧ e.b = 3) ∧
  (∃ (l : Line), l = ⟨2, -3, 7⟩) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l730_73063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_zero_l730_73040

theorem sum_remainder_zero (x y z w : ℕ) 
  (hx : x % 8 = 3)
  (hy : y % 8 = 5)
  (hz : z % 8 = 7)
  (hw : w % 8 = 1) :
  (x + y + z + w) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_zero_l730_73040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_fb_perp_asymptote_l730_73011

/-- A hyperbola with focus F and imaginary axis endpoint B -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The slope of the asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

/-- The slope of the line FB -/
noncomputable def fb_slope (h : Hyperbola) : ℝ := -h.c / h.b

/-- Theorem: If FB is perpendicular to an asymptote, the eccentricity is (√5 + 1) / 2 -/
theorem hyperbola_eccentricity_when_fb_perp_asymptote (h : Hyperbola) 
  (h_perp : fb_slope h * asymptote_slope h = -1) : 
  eccentricity h = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_fb_perp_asymptote_l730_73011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_sum_l730_73048

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the properties of g based on the graph
axiom g_property : g (-1) = 5 ∧ g 3 = 5

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 5)

-- Theorem statement
theorem intersection_and_sum :
  (g (intersection_point.fst) = intersection_point.snd) ∧
  (g (intersection_point.fst - 4) = intersection_point.snd) ∧
  (intersection_point.fst + intersection_point.snd = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_sum_l730_73048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_social_group_theorem_l730_73058

noncomputable def knows : ℕ → ℕ → Prop := sorry

theorem social_group_theorem (n : ℕ) (h : n > 0) :
  ∃ (S : Finset (Finset ℕ)), 
    Finset.card S = Nat.choose (2*n) n ∧ 
    ∃ (T : Finset ℕ), 
      T ⊆ S.sup id ∧ 
      Finset.card T = n + 1 ∧
      (∀ i j, i ∈ T → j ∈ T → i ≠ j → (knows i j ∨ (¬ knows i j ∧ ¬ knows j i))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_social_group_theorem_l730_73058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_fish_percentage_rounded_l730_73082

/-- Represents a grocery item with its name, quantity, and price per unit -/
structure GroceryItem where
  name : String
  quantity : ℚ
  price_per_unit : ℚ

/-- Calculates the total cost of a grocery item -/
def item_cost (item : GroceryItem) : ℚ := item.quantity * item.price_per_unit

/-- Determines if a grocery item is meat or fish -/
def is_meat_or_fish (item : GroceryItem) : Bool :=
  item.name = "Bacon" || item.name = "Chicken" || item.name = "Tilapia" || item.name = "Steak"

/-- Calculates the total cost of all items in a list -/
def total_cost (items : List GroceryItem) : ℚ :=
  items.foldl (fun acc item => acc + item_cost item) 0

/-- Calculates the total cost of meat and fish items in a list -/
def meat_fish_cost (items : List GroceryItem) : ℚ :=
  items.filter is_meat_or_fish |> total_cost

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  if x - ⌊x⌋ ≥ 1/2 then ⌈x⌉ else ⌊x⌋

theorem meat_fish_percentage_rounded (grocery_list : List GroceryItem) : 
  round_to_nearest ((meat_fish_cost grocery_list / total_cost grocery_list) * 100) = 39 :=
by sorry

#check meat_fish_percentage_rounded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_fish_percentage_rounded_l730_73082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_inequality_l730_73066

open Real Set

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- The dot product of two 2D vectors -/
def dot (v w : Vec2D) : ℝ := v.x * w.x + v.y * w.y

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : Vec2D) : ℝ := Real.sqrt (dot v v)

/-- The distance between two points represented by 2D vectors -/
noncomputable def distance (v w : Vec2D) : ℝ := magnitude (Vec2D.mk (w.x - v.x) (w.y - v.y))

/-- The curve C -/
def C (a b : Vec2D) : Set Vec2D :=
  {p | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * π ∧ p = Vec2D.mk (a.x * cos θ + b.x * sin θ) (a.y * cos θ + b.y * sin θ)}

/-- The region Ω -/
def Ω (Q : Vec2D) (r R : ℝ) : Set Vec2D :=
  {p | 0 < r ∧ r ≤ distance p Q ∧ distance p Q ≤ R ∧ r < R}

theorem intersection_implies_inequality (a b Q : Vec2D) (r R : ℝ) :
  magnitude a = 1 →
  magnitude b = 1 →
  dot a b = 0 →
  Q = Vec2D.mk (Real.sqrt 2 * (a.x + b.x)) (Real.sqrt 2 * (a.y + b.y)) →
  (C a b ∩ Ω Q r R).Nonempty →
  (C a b ∩ Ω Q r R).Finite →
  1 < r ∧ r < R ∧ R < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_inequality_l730_73066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_profit_share_l730_73062

noncomputable def total_investment : ℝ := 120000
noncomputable def profit_ratio_A : ℝ := 4
noncomputable def profit_ratio_B : ℝ := 3
noncomputable def profit_ratio_C : ℝ := 2
noncomputable def total_profit : ℝ := 50000

noncomputable def investment_difference_AB : ℝ := 6000
noncomputable def investment_difference_BC : ℝ := 8000

noncomputable def investment_C : ℝ := (total_investment - investment_difference_AB - 2 * investment_difference_BC) / 3
noncomputable def investment_B : ℝ := investment_C + investment_difference_BC
noncomputable def investment_A : ℝ := investment_B + investment_difference_AB

noncomputable def total_ratio_parts : ℝ := profit_ratio_A + profit_ratio_B + profit_ratio_C
noncomputable def profit_per_part : ℝ := total_profit / total_ratio_parts

theorem C_profit_share (ε : ℝ) (hε : ε > 0) : 
  ∃ (share : ℝ), abs (share - profit_ratio_C * profit_per_part) < ε ∧ 
  abs (share - 11111.11) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_profit_share_l730_73062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fₖ_monotone_increasing_on_neg_infinity_to_neg_one_l730_73068

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2^(-abs x)

/-- The modified function fₖ(x) with K = 1/2 -/
noncomputable def fₖ (x : ℝ) : ℝ := min (f x) (1/2)

/-- Theorem stating that fₖ(x) is monotonically increasing on (-∞, -1) -/
theorem fₖ_monotone_increasing_on_neg_infinity_to_neg_one :
  MonotoneOn fₖ (Set.Iio (-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fₖ_monotone_increasing_on_neg_infinity_to_neg_one_l730_73068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_max_sum_of_sides_l730_73094

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a = 2 * Real.sin t.A ∧
  t.b = 2 * Real.sin t.B ∧
  t.c = 2 * Real.sin t.C

-- Define the dot product condition
def dot_product_condition (t : Triangle) : Prop :=
  Real.cos (-t.A) * (2 * Real.cos (t.A + 1)) + 1 * (-1) = -Real.cos t.A

-- Define the theorem for part 1
theorem area_of_triangle (t : Triangle) :
  triangle_conditions t →
  dot_product_condition t →
  t.a = 2 →
  t.b = 2 →
  1/2 * t.a * t.b * Real.sin t.C = 2 := by sorry

-- Define the theorem for part 2
theorem max_sum_of_sides (t : Triangle) :
  triangle_conditions t →
  dot_product_condition t →
  t.a = 2 →
  t.b + t.c ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_max_sum_of_sides_l730_73094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bar_chart_characteristic_l730_73002

/-- Represents the characteristics of a bar chart -/
structure BarChartCharacteristics where
  /-- The main characteristic of a bar chart -/
  main_characteristic : String

/-- Creates an instance of BarChartCharacteristics -/
def barChartInfo : BarChartCharacteristics :=
  { main_characteristic := "it can easily show the quantity" }

/-- A theorem stating the main characteristic of a bar chart -/
theorem bar_chart_characteristic :
  barChartInfo.main_characteristic = "it can easily show the quantity" := by
  rfl

#check bar_chart_characteristic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bar_chart_characteristic_l730_73002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l730_73032

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)
  (right_angle_F : (F.1 - E.1) * (G.1 - F.1) + (F.2 - E.2) * (G.2 - F.2) = 0)
  (right_angle_H : (H.1 - G.1) * (E.1 - H.1) + (H.2 - G.2) * (E.2 - H.2) = 0)
  (diagonal_EG : ((E.1 - G.1)^2 + (E.2 - G.2)^2) = 25)
  (distinct_integer_sides : ∃ (a b : ℕ), a ≠ b ∧ 
    (((E.1 - F.1)^2 + (E.2 - F.2)^2 = a^2 ∧ (F.1 - G.1)^2 + (F.2 - G.2)^2 = b^2) ∨
     ((G.1 - H.1)^2 + (G.2 - H.2)^2 = a^2 ∧ (H.1 - E.1)^2 + (H.2 - E.2)^2 = b^2)))

/-- The area of a quadrilateral with the given properties is 12 -/
theorem quadrilateral_area (q : Quadrilateral) : 
  abs ((q.E.1 - q.G.1) * (q.F.2 - q.H.2) - (q.F.1 - q.H.1) * (q.E.2 - q.G.2)) / 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l730_73032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cut_length_l730_73098

/-- Represents a triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * t.base * t.height

/-- Represents the problem setup -/
structure IsoscelesTriangleProblem where
  triangle : Triangle
  trapezoidArea : ℝ

/-- The main theorem statement -/
theorem isosceles_triangle_cut_length 
  (problem : IsoscelesTriangleProblem)
  (h1 : problem.triangle.area = 200)
  (h2 : problem.triangle.height = 40)
  (h3 : problem.trapezoidArea = 150) :
  let smallerTriangleArea := problem.triangle.area - problem.trapezoidArea
  let areaRatio := smallerTriangleArea / problem.triangle.area
  let lengthRatio := Real.sqrt areaRatio
  lengthRatio * problem.triangle.base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_cut_length_l730_73098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABC_l730_73089

noncomputable section

-- Define the ellipse C₁
def C₁ (x y : ℝ) (a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1 ∧ a > b ∧ b > 0

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y = x^2

-- Define point A
def A : ℝ × ℝ := (0, 1/16)

-- Define the tangent line to C₂ at point N(t, t²)
def tangentLine (t x y : ℝ) : Prop := y = 2*t*x - t^2

-- Define the area of triangle ABC
noncomputable def areaABC (t : ℝ) : ℝ := 
  Real.sqrt (-(t^2 - 8)^2 + 65) / 8

theorem max_area_ABC : 
  ∀ a b : ℝ, 
  C₁ 2 0 a 1 → -- Using the fact that b = 1 from the solution
  eccentricity a 1 = Real.sqrt 3 / 2 →
  (∀ t : ℝ, areaABC t ≤ Real.sqrt 65 / 8) ∧ 
  (∃ t : ℝ, areaABC t = Real.sqrt 65 / 8) := by
  sorry

#check max_area_ABC

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABC_l730_73089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_complex_absolute_value_l730_73067

/-- Two complex numbers are conjugates -/
def is_conjugate (z w : ℂ) : Prop :=
  z.re = w.re ∧ z.im = -w.im

theorem conjugate_complex_absolute_value (α β : ℂ) :
  is_conjugate α β →
  (∃ (r : ℝ), α / β^2 = r) →
  (Complex.abs (α - β) = 4) →
  Complex.abs α = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_complex_absolute_value_l730_73067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_l730_73093

noncomputable def csc (y : ℝ) : ℝ := 1 / Real.sin y
noncomputable def cot (y : ℝ) : ℝ := Real.cos y / Real.sin y
noncomputable def sec (y : ℝ) : ℝ := 1 / Real.cos y
noncomputable def tan (y : ℝ) : ℝ := Real.sin y / Real.cos y

theorem sum_of_fractions (y : ℝ) (p q : ℕ) 
  (h1 : csc y + cot y = 25 / 7)
  (h2 : sec y + tan y = p / q)
  (h3 : Nat.Coprime p q) :
  p + q = 29517 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_l730_73093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_blue_numbers_l730_73051

/-- Represents a card with a blue number and a red number -/
structure Card where
  blue : Nat
  red : Nat
deriving Inhabited

/-- The function to calculate the red number for a given card -/
def f (cards : List Card) (index : Nat) : Nat :=
  let nextFiveHundred := (List.take 500 (List.drop (index + 1) (cards ++ cards)))
  (nextFiveHundred.filter (λ c => c.blue > (cards.get! index).blue)).length

/-- The main theorem stating that the blue numbers can be determined from the red numbers -/
theorem determine_blue_numbers
  (cards : List Card)
  (h1 : cards.length = 1001)
  (h2 : ∀ i, i < 1001 → (cards.get! i).blue ∈ Finset.range 1001)
  (h3 : ∀ i j, i ≠ j → (cards.get! i).blue ≠ (cards.get! j).blue)
  (h4 : ∀ i, i < 1001 → (cards.get! i).red = f cards i) :
  ∃ (g : List Nat → List Nat), g (cards.map Card.red) = cards.map Card.blue := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_blue_numbers_l730_73051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_sequence_closed_form_l730_73078

noncomputable def our_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | k + 1 => our_sequence k + (k + 1)^3

theorem our_sequence_closed_form (n : ℕ) :
  our_sequence n = (n^2 * (n + 1)^2 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_sequence_closed_form_l730_73078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_l730_73015

def x : ℕ → ℕ
  | 0 => 1  -- Adding a case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 14 * x (n + 2) - x (n + 1) - 4

theorem x_is_perfect_square :
  ∃ y : ℕ → ℕ, ∀ n : ℕ, x n = (y n)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_l730_73015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l730_73092

-- Define the function f(x) = 2^(x+1)
noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = y) ↔ 1 ≤ y ∧ y ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l730_73092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2016_power_divisibility_l730_73001

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Added this case to cover Nat.zero
  | 1 => 1
  | 2 => 3
  | n+3 => 3 * sequence_a (n+2) - sequence_a (n+1)

def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem smallest_n_for_2016_power_divisibility :
  ∃ n : ℕ, 
    (divides (2^2016) (sequence_a n)) ∧ 
    ¬(divides (2^2017) (sequence_a n)) ∧
    (∀ m : ℕ, m < n → ¬(divides (2^2016) (sequence_a m))) ∧
    n = 3 * 2^2013 := by
  sorry

#eval sequence_a 10  -- Added this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2016_power_divisibility_l730_73001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_truck_loads_approx_l730_73007

/-- The total number of truck-loads of material needed for a renovation project -/
noncomputable def total_truck_loads : ℝ :=
  (0.16666666666666666 * Real.pi) +  -- Sand
  (0.3333333333333333 * Real.exp 1) +  -- Dirt (e = exp 1)
  (0.16666666666666666 * Real.sqrt 2)  -- Cement

/-- The total number of truck-loads is approximately 1.6653937026604812 -/
theorem total_truck_loads_approx :
  abs (total_truck_loads - 1.6653937026604812) < 1e-10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_truck_loads_approx_l730_73007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l730_73097

noncomputable section

open Real

-- Define the curves C₁ and C₂
def C₁ (φ : ℝ) : ℝ × ℝ := (Real.cos φ, Real.sin φ)
def C₂ (a b φ : ℝ) : ℝ × ℝ := (a * Real.cos φ, b * Real.sin φ)

-- Define the conditions
def condition_positive (a b : ℝ) : Prop := a > b ∧ b > 0

def condition_distance (a : ℝ) : Prop :=
  (a - 1)^2 = 4  -- Distance between points when α = 0 is 2

def condition_coincide (b : ℝ) : Prop :=
  b = 1  -- Points coincide when α = π/2

-- Define the result
def result (a b : ℝ) : Prop :=
  a = 3 ∧ b = 1

-- Define the polar equations
def polar_eq_A₁A₂ (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = Real.sqrt 2 / 2

def polar_eq_B₁B₂ (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 3 * Real.sqrt 10 / 10

-- Main theorem
theorem curve_properties :
  ∀ a b : ℝ,
  condition_positive a b →
  condition_distance a →
  condition_coincide b →
  result a b ∧
  (∀ ρ θ : ℝ, polar_eq_A₁A₂ ρ θ) ∧
  (∀ ρ θ : ℝ, polar_eq_B₁B₂ ρ θ) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l730_73097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_surface_area_is_134_l730_73008

noncomputable section

def bottom_base : ℝ := 10
def top_base : ℝ := 7
def trapezoid_height : ℝ := 4
def depth : ℝ := 3

def trapezoid_area (b1 b2 h : ℝ) : ℝ := (b1 + b2) * h / 2
def rectangle_area (l w : ℝ) : ℝ := l * w

def brick_surface_area : ℝ :=
  2 * trapezoid_area bottom_base top_base trapezoid_height +
  2 * rectangle_area trapezoid_height depth +
  2 * rectangle_area top_base depth

theorem brick_surface_area_is_134 : brick_surface_area = 134 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_surface_area_is_134_l730_73008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_real_root_equations_eq_38_l730_73023

def b_set : Finset Int := {-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6}
def c_set : Finset Int := {1, 2, 3, 4, 5, 6}

def has_real_roots (b : Int) (c : Int) : Bool :=
  b * b ≥ 4 * c

def count_real_root_equations : Nat :=
  (b_set.filter (fun b => (c_set.filter (fun c => has_real_roots b c)).card > 0)).card

theorem count_real_root_equations_eq_38 : count_real_root_equations = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_real_root_equations_eq_38_l730_73023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_tenth_greater_than_ten_l730_73050

noncomputable def sequenceA (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (sequenceA a n)^2 + 1/2

theorem sequence_tenth_greater_than_ten (a : ℝ) : sequenceA a 9 > 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_tenth_greater_than_ten_l730_73050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_in_fourth_quadrant_l730_73035

noncomputable def z : ℂ := (1 - Complex.I) / (2 * Complex.I)

noncomputable def Z : ℂ := z

def symmetric_point (p : ℂ) : ℂ := Complex.mk (-p.re) p.im

theorem symmetric_point_in_fourth_quadrant :
  let sym_Z := symmetric_point Z
  0 < sym_Z.re ∧ sym_Z.im < 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_in_fourth_quadrant_l730_73035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_pentominoes_l730_73071

/-- A pentomino is a plane geometric figure formed by joining 5 equal squares edge to edge. -/
structure Pentomino where
  -- We'll represent a pentomino as a unit for now, but in a real implementation,
  -- this would contain the actual representation of the shape.
  mk :: 

/-- Two pentominoes are considered equivalent if they can be transformed into one another via rotation or reflection. -/
def are_equivalent : Pentomino → Pentomino → Prop := sorry

/-- The set of all distinct pentominoes -/
def distinct_pentominoes : Finset Pentomino := sorry

/-- The number of distinct pentominoes is 18 -/
theorem count_distinct_pentominoes : Finset.card distinct_pentominoes = 18 := by
  sorry

#check count_distinct_pentominoes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_pentominoes_l730_73071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_inequality_l730_73077

/-- The function f(x) = x / e^x -/
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

/-- Theorem: If x₁ ≠ x₂ and f(x₁) = f(x₂), then x₁ > 2 - x₂ -/
theorem function_equality_implies_inequality (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) (h2 : f x₁ = f x₂) : x₁ > 2 - x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_inequality_l730_73077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angles_l730_73057

structure IsoscelesTriangle where
  -- Define an isosceles triangle
  base : ℝ
  leg : ℝ
  height : ℝ
  isIsosceles : base > 0 ∧ leg > 0
  heightCondition : height = leg / 2

def IsoscelesTriangle.baseAngles (t : IsoscelesTriangle) : Set ℝ :=
  {75, 15}

theorem isosceles_triangle_base_angles 
  (t : IsoscelesTriangle) : 
  ∃ angle, angle ∈ t.baseAngles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_angles_l730_73057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_average_scores_l730_73072

noncomputable def male_scores : List ℚ := [92, 89, 93, 90]
noncomputable def female_scores : List ℚ := [92, 88, 93, 91]

noncomputable def average (scores : List ℚ) : ℚ := (scores.sum) / scores.length

theorem equal_average_scores : average male_scores = average female_scores := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_average_scores_l730_73072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentysixth_card_is_red_l730_73052

-- Define the color type
inductive Color where
  | Red
  | Black

-- Define the card placement function
def cardPlacement : ℕ → Color := sorry

-- Define the conditions
axiom different_colors_on_top : ∀ n : ℕ, cardPlacement n ≠ cardPlacement (n + 1)
axiom tenth_card_red : cardPlacement 10 = Color.Red
axiom eleventh_card_red : cardPlacement 11 = Color.Red
axiom twentyfifth_card_black : cardPlacement 25 = Color.Black

-- The theorem to prove
theorem twentysixth_card_is_red : cardPlacement 26 = Color.Red := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentysixth_card_is_red_l730_73052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l730_73087

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Adding case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | (n + 4) => (3 + a (n + 3) * a (n + 2)) / (a (n + 3) - 2)

theorem a_formula (n : ℕ) (h : n ≥ 1) :
  a n = ((5 + 2 * Real.sqrt 5) / 10) * ((3 + Real.sqrt 5) / 2) ^ n +
        ((5 - 2 * Real.sqrt 5) / 10) * ((3 - Real.sqrt 5) / 2) ^ n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l730_73087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l730_73070

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d₁ d₂ : ℝ) : ℝ :=
  |d₁ - d₂| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance between the planes 3x - y + 2z = 6 and 6x - 2y + 4z = -4 is 4√14 / 7 -/
theorem distance_specific_planes :
  distance_between_planes 3 (-1) 2 6 (-2) = 4 * Real.sqrt 14 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l730_73070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_eel_length_value_l730_73016

/-- The length of Jenna's eel in inches -/
def jenna_eel_length : ℚ := sorry

/-- The length of Bill's eel in inches -/
def bill_eel_length : ℚ := sorry

/-- The length of Lucy's eel in inches -/
def lucy_eel_length : ℚ := sorry

/-- Jenna's eel is 2/5 as long as Bill's eel -/
axiom jenna_bill_ratio : jenna_eel_length = (2 / 5 : ℚ) * bill_eel_length

/-- Jenna's eel is 3/7 as long as Lucy's eel -/
axiom jenna_lucy_ratio : jenna_eel_length = (3 / 7 : ℚ) * lucy_eel_length

/-- The combined length of their eels is 124 inches -/
axiom total_length : jenna_eel_length + bill_eel_length + lucy_eel_length = 124

/-- Theorem: Given the conditions, Jenna's eel length is 744/35 inches -/
theorem jenna_eel_length_value : jenna_eel_length = 744 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_eel_length_value_l730_73016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_to_N_l730_73039

/-- The ellipse C1 defined by x²/16 + y²/8 = 1 -/
def C1 (x y : ℝ) : Prop := x^2/16 + y^2/8 = 1

/-- The point N, which is the center of C2 -/
def N : ℝ × ℝ := (-1, 0)

/-- The distance between a point (x, y) and N -/
noncomputable def distance_to_N (x y : ℝ) : ℝ := 
  Real.sqrt ((x - N.1)^2 + (y - N.2)^2)

/-- The minimum distance between any point on C1 and N is √7 -/
theorem min_distance_C1_to_N : 
  ∃ (d : ℝ), d = Real.sqrt 7 ∧ 
  ∀ (x y : ℝ), C1 x y → distance_to_N x y ≥ d ∧ 
  ∃ (x₀ y₀ : ℝ), C1 x₀ y₀ ∧ distance_to_N x₀ y₀ = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_to_N_l730_73039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_positive_range_l730_73064

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = 4 → ∃ m b : ℝ, ∀ x y : ℝ,
    y = f a x → (x = 1 ∧ y = f a 1) → (y = m * (x - 1) + f a 1 ↔ 2 * x + y - 2 = 0) := by
  sorry

-- Theorem for the range of a
theorem f_positive_range (a : ℝ) :
  (∀ x : ℝ, x > 1 → f a x > 0) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_positive_range_l730_73064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_square_l730_73000

-- Define a square
def Square (a : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a}

-- Define a point on the perimeter of the square
def OnPerimeter (p : ℝ × ℝ) (a : ℝ) : Prop :=
  p ∈ Square a ∧ (p.1 = 0 ∨ p.1 = a ∨ p.2 = 0 ∨ p.2 = a)

-- Define an equilateral triangle
def Equilateral (p q r : ℝ × ℝ) : Prop :=
  let d := λ x y : ℝ × ℝ => Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)
  d p q = d q r ∧ d q r = d r p

-- Theorem statement
theorem equilateral_triangle_in_square (a : ℝ) (h : a > 0) :
  ∀ p : ℝ × ℝ, OnPerimeter p a →
    ∃ q r : ℝ × ℝ, OnPerimeter q a ∧ OnPerimeter r a ∧
      Equilateral p q r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_square_l730_73000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l730_73073

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.cos (x - Real.pi/3)

-- State the theorem
theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  -- The range of f in [0, π/2] is [0, 3]
  (∀ y, (∃ x, x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = y) ↔ y ∈ Set.Icc 0 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l730_73073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_6_l730_73096

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  seq_prop : ∀ n, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
noncomputable def S (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a 1
  else g.a 1 * (1 - g.q ^ n) / (1 - g.q)

theorem geometric_sequence_sum_6 (g : GeometricSequence) :
  g.a 3 = 4 ∧ S g 3 = 7 → S g 6 = 63 ∨ S g 6 = 133 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_6_l730_73096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_relation_l730_73061

-- Define the triangle ABC and points M and O
variable (A B C M O : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : (O - A) + (O - B) + (O - C) = 0)
variable (h2 : (M - A) + (M - B) + 2 • (M - C) = 0)
variable (h3 : ∃ (x y : ℝ), M - O = x • (B - A) + y • (C - A))

-- State the theorem
theorem triangle_point_relation :
  ∃ (x y : ℝ), M - O = x • (B - A) + y • (C - A) ∧ x + y = -1/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_relation_l730_73061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_area_relation_l730_73028

/-- The areas of the two base faces of a frustum are S₁ and S₂, and the area of the midsection
    (the section passing through the midpoint of each edge) is S₀. Then 2√S₀ = √S₁ + √S₂ --/
theorem frustum_area_relation (S₀ S₁ S₂ : ℝ) (h₀ : S₀ > 0) (h₁ : S₁ > 0) (h₂ : S₂ > 0) :
  2 * Real.sqrt S₀ = Real.sqrt S₁ + Real.sqrt S₂ := by
  sorry

#check frustum_area_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_area_relation_l730_73028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l730_73086

theorem trigonometric_expression_value (α : ℝ) :
  (α > π) ∧ (α < 3*π/2) →  -- α is in the third quadrant
  (3 * (Real.cos α)^2 - Real.cos α - 2 = 0) →  -- cos α is a root of 3x^2 - x - 2 = 0
  (Real.sin (-α + 3*π/2) * Real.cos (3*π/2 + α) * Real.tan (π - α)^2) / 
  (Real.cos (π/2 + α) * Real.sin (π/2 - α)) = 5/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l730_73086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l730_73085

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n + 1 => 1 / (2 - mySequence n)

theorem mySequence_formula (n : ℕ) (h : n > 0) : mySequence n = (n - 1) / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l730_73085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l730_73074

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 - t, -1 + 2*t)

def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

def point_P : ℝ × ℝ := (1, -1)

def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), 
    line_l t₁ = A ∧ 
    line_l t₂ = B ∧ 
    curve_C A.1 A.2 ∧ 
    curve_C B.1 B.2

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersection_reciprocal_sum (A B : ℝ × ℝ) :
  intersection_points A B →
  1 / distance point_P A + 1 / distance point_P B = 12 * Real.sqrt 30 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l730_73074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_circular_arrangement_l730_73021

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 100 ∧
  (∀ n, n ∈ arr → 1 ≤ n ∧ n ≤ 100) ∧
  (∀ i, i < arr.length → 
    30 ≤ Int.natAbs (arr[i]! - arr[(i + 1) % arr.length]!) ∧ 
    Int.natAbs (arr[i]! - arr[(i + 1) % arr.length]!) ≤ 50)

theorem no_valid_circular_arrangement : 
  ¬ ∃ (arr : List Nat), is_valid_arrangement arr := by
  sorry

#check no_valid_circular_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_circular_arrangement_l730_73021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l730_73069

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def isUnitVector (v : V) : Prop := ‖v‖ = 1

def isParallel (a b : V) : Prop := ∃ (k : ℝ), b = k • a

theorem all_propositions_false :
  (∀ (a b : V), isUnitVector a → isUnitVector b → isParallel a b → a = b) = False ∧
  (∀ (k : ℝ), k • (0 : V) = (0 : V)) = False ∧
  (∀ (a b : V), isParallel a b → ‖b‖ = ‖a‖) = False ∧
  (∀ (k : ℝ) (a : V), k • a = 0 → k = 0) = False ∧
  (∀ (a : V), ‖a‖ = 0 → a = 0) = False :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l730_73069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l730_73026

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 20 cm, height 12 cm, and area 228 cm², the other side is 18 cm -/
theorem trapezium_other_side_length (t : Trapezium) (h1 : t.side1 = 20) (h2 : t.height = 12) (h3 : area t = 228) : t.side2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l730_73026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_change_l730_73022

def vip_price : ℚ := 120
def regular_price : ℚ := 60
def discount_price : ℚ := 30
def wheelchair_price : ℚ := 40

def vip_quantity : ℕ := 5
def regular_quantity : ℕ := 7
def discount_quantity : ℕ := 4
def wheelchair_quantity : ℕ := 2

def tax_rate : ℚ := 5 / 100
def processing_fee : ℚ := 25
def payment : ℚ := 2000

theorem concert_ticket_change :
  let total_cost := vip_price * vip_quantity + 
                    regular_price * regular_quantity + 
                    discount_price * discount_quantity + 
                    wheelchair_price * wheelchair_quantity
  let tax := tax_rate * total_cost
  let final_cost := total_cost + tax + processing_fee
  payment - final_cost = 694 := by
  -- Proof goes here
  sorry

#eval let total_cost := vip_price * vip_quantity + 
                        regular_price * regular_quantity + 
                        discount_price * discount_quantity + 
                        wheelchair_price * wheelchair_quantity
      let tax := tax_rate * total_cost
      let final_cost := total_cost + tax + processing_fee
      payment - final_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_change_l730_73022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radians_to_degrees_1_radians_to_degrees_2_radians_to_degrees_3_degrees_to_radians_1_degrees_to_radians_2_l730_73006

-- Define the conversion factor
noncomputable def π_degrees : ℝ := 180

-- Define the conversion functions
noncomputable def radians_to_degrees (x : ℝ) : ℝ := x * π_degrees / Real.pi
noncomputable def degrees_to_radians (x : ℝ) : ℝ := x * Real.pi / π_degrees

-- State the theorems
theorem radians_to_degrees_1 : radians_to_degrees (Real.pi / 12) = 15 := by sorry

theorem radians_to_degrees_2 : radians_to_degrees (13 * Real.pi / 6) = 390 := by sorry

theorem radians_to_degrees_3 : radians_to_degrees (-5 * Real.pi / 12) = -75 := by sorry

theorem degrees_to_radians_1 : degrees_to_radians 36 = Real.pi / 5 := by sorry

theorem degrees_to_radians_2 : degrees_to_radians (-105) = -7 * Real.pi / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radians_to_degrees_1_radians_to_degrees_2_radians_to_degrees_3_degrees_to_radians_1_degrees_to_radians_2_l730_73006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_red_two_blue_probability_l730_73047

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def marbles_drawn : ℕ := 4

theorem two_red_two_blue_probability :
  (red_marbles.choose 2 * blue_marbles.choose 2 * Nat.factorial 4) / total_marbles.choose marbles_drawn = 168 / 323 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_red_two_blue_probability_l730_73047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_exists_l730_73060

/-- A right triangle in the coordinate plane with legs parallel to the x and y axes. -/
structure RightTriangle where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ

/-- The slope of a median in the right triangle. -/
def median_slope (t : RightTriangle) : ℚ := t.c / t.d

/-- The condition that one median lies on the line y = 2x + 1. -/
def median_on_line_1 (t : RightTriangle) : Prop := median_slope t = 2

/-- The condition that another median lies on the line y = mx + 2. -/
def median_on_line_2 (t : RightTriangle) (m : ℚ) : Prop := median_slope t = m

/-- The theorem stating that there exists exactly one value of m for which
    a right triangle can be constructed satisfying the given conditions. -/
theorem unique_m_exists : ∃! m : ℚ, ∃ t : RightTriangle, median_on_line_1 t ∧ median_on_line_2 t m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_exists_l730_73060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_system_problem_l730_73099

/-- Given a rectangular coordinate system and a polar coordinate system on a plane,
    with two points M and N, and a circle C, prove the following:
    1. The equation of line OP
    2. The equation of line l passing through M and N
    3. The relationship between line l and circle C -/
theorem coordinate_system_problem 
  (M N : ℝ × ℝ) (C : Set (ℝ × ℝ)) :
  (M = (2, 0)) →
  (N = (0, 2 * Real.sqrt 3 / 3)) →
  (∃ center : ℝ × ℝ, center = (2, Real.sqrt 3) ∧ 
    C = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 4}) →
  let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let l := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 - 2 = 0}
  (∃ k, {p : ℝ × ℝ | p.2 = k * p.1} = {p : ℝ × ℝ | p.2 = Real.sqrt 3 / 3 * p.1}) ∧
  (l = {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 - 2 = 0}) ∧
  (∃ p : ℝ × ℝ, p ∈ l ∧ p ∈ C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_system_problem_l730_73099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_is_one_l730_73046

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A parabola passes through a given point (x, y) -/
def Parabola.passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.y_at x = y

/-- The x-coordinate of the vertex of a parabola -/
noncomputable def Parabola.vertex_x (p : Parabola) : ℝ :=
  -p.b / (2 * p.a)

theorem parabola_vertex_x_is_one (p : Parabola) 
  (h1 : p.passes_through (-2) 9)
  (h2 : p.passes_through 4 9)
  (h3 : p.passes_through 7 14) :
  p.vertex_x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_is_one_l730_73046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_routeB_is_fastest_l730_73080

/-- Represents a route with its characteristics -/
structure Route where
  distance : ℝ
  speed : ℝ
  trafficDelay : ℝ
  restStops : ℕ
  restStopDuration : ℝ
  mealBreaks : ℕ
  mealBreakDuration : ℝ
  fuelStops : ℕ
  fuelStopDuration : ℝ
  additionalDelay : ℝ

/-- Calculates the total time for a given route -/
noncomputable def totalTime (r : Route) : ℝ :=
  r.distance / r.speed +
  r.trafficDelay +
  r.restStops * r.restStopDuration +
  r.mealBreaks * r.mealBreakDuration +
  r.fuelStops * r.fuelStopDuration +
  r.additionalDelay

/-- The four routes given in the problem -/
noncomputable def routeA : Route := {
  distance := 1500,
  speed := 75,
  trafficDelay := 2,
  restStops := 3,
  restStopDuration := 0.5,
  mealBreaks := 1,
  mealBreakDuration := 1,
  fuelStops := 2,
  fuelStopDuration := 1/6,
  additionalDelay := 1.5
}

noncomputable def routeB : Route := {
  distance := 1300,
  speed := 70,
  trafficDelay := 0,
  restStops := 2,
  restStopDuration := 0.75,
  mealBreaks := 1,
  mealBreakDuration := 0.75,
  fuelStops := 3,
  fuelStopDuration := 0.25,
  additionalDelay := 1
}

noncomputable def routeC : Route := {
  distance := 1800,
  speed := 80,
  trafficDelay := 2.5,
  restStops := 4,
  restStopDuration := 1/3,
  mealBreaks := 2,
  mealBreakDuration := 1,
  fuelStops := 2,
  fuelStopDuration := 1/6,
  additionalDelay := 2
}

noncomputable def routeD : Route := {
  distance := 750,
  speed := 25,
  trafficDelay := 0,
  restStops := 1,
  restStopDuration := 1,
  mealBreaks := 1,
  mealBreakDuration := 0.5,
  fuelStops := 1,
  fuelStopDuration := 1/3,
  additionalDelay := 3
}

theorem routeB_is_fastest :
  totalTime routeB < totalTime routeA ∧
  totalTime routeB < totalTime routeC ∧
  totalTime routeB < totalTime routeD :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_routeB_is_fastest_l730_73080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_67432_to_base10_l730_73045

/-- Converts a list of digits in base 7 to its equivalent in base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  (List.enum digits).foldl
    (fun acc (i, d) => acc + d * (7 ^ i))
    0

/-- The base 10 equivalent of 67432 in base 7 -/
theorem base7_67432_to_base10 :
  base7ToBase10 [2, 3, 4, 7, 6] = 17026 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_67432_to_base10_l730_73045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_acute_triangle_inequality_sin_2A_eq_sin_2B_not_always_isosceles_circumradius_not_sqrt3_over_3_l730_73012

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h_angles : A + B + C = Real.pi
  h_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define an acute triangle
def isAcute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

theorem sine_inequality (t : Triangle) (h : t.A > t.B) : 
  Real.sin t.A > Real.sin t.B := by sorry

theorem acute_triangle_inequality (t : Triangle) (h : isAcute t) : 
  t.b^2 + t.c^2 - t.a^2 > 0 := by sorry

theorem sin_2A_eq_sin_2B_not_always_isosceles : 
  ∃ t : Triangle, Real.sin (2 * t.A) = Real.sin (2 * t.B) ∧ t.a ≠ t.b := by sorry

theorem circumradius_not_sqrt3_over_3 (t : Triangle) 
  (h1 : t.b = 3) (h2 : t.A = Real.pi/3) (h3 : t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3) : 
  t.a / (2 * Real.sin t.A) ≠ Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_acute_triangle_inequality_sin_2A_eq_sin_2B_not_always_isosceles_circumradius_not_sqrt3_over_3_l730_73012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_union_area_l730_73018

/-- The area of the union of a square and a circle, where the square has side length 8,
    the circle has radius 12, and the circle is centered at one of the square's vertices. -/
noncomputable def unionArea : ℝ := 64 + 108 * Real.pi

/-- Theorem stating that the area of the union of a square with side length 8 and a circle
    with radius 12 centered at one of the square's vertices is equal to 64 + 108π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = unionArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_union_area_l730_73018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_runtime_is_one_point_five_l730_73009

/-- Represents the costs and runtime of a computer program. -/
structure ProgramCosts where
  overhead : ℝ
  time_cost_per_ms : ℝ
  tape_cost : ℝ
  total_cost : ℝ

/-- Calculates the runtime of a program in seconds given its costs. -/
noncomputable def calculate_runtime (costs : ProgramCosts) : ℝ :=
  ((costs.total_cost - costs.overhead - costs.tape_cost) / costs.time_cost_per_ms) / 1000

/-- Theorem stating that given the specific costs, the program runtime is 1.5 seconds. -/
theorem program_runtime_is_one_point_five :
  let costs : ProgramCosts := {
    overhead := 1.07,
    time_cost_per_ms := 0.023,
    tape_cost := 5.35,
    total_cost := 40.92
  }
  calculate_runtime costs = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_runtime_is_one_point_five_l730_73009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l730_73033

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem solution_exists : ∃ x : ℝ, (3 * x + 5 * (floor x) - 49 = 0) ∧ x = 19/3 := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l730_73033
