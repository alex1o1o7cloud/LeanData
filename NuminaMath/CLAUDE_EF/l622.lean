import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l622_62235

theorem triangle_side_b (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Ensure angles are valid
  A + B + C = π →  -- Sum of angles in a triangle
  Real.cos A = 5/13 →
  Real.sin B = 3/5 →
  a = 20 →
  b = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l622_62235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l622_62254

theorem remainder_theorem : 
  ∃ q : Polynomial ℝ, 3 * X^8 - 2 * X^5 + 5 * X^3 - 9 = (X^2 - 2*X + 1) * q + (29*X - 32) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l622_62254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l622_62297

/-- The line y = 2x - 7 -/
def line (x : ℝ) : ℝ := 2 * x - 7

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (3, 4)

/-- The proposed closest point on the line -/
def closest_point : ℝ × ℝ := (5, 3)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem closest_point_on_line :
  (∀ x : ℝ, distance (x, line x) point ≥ distance closest_point point) ∧
  (closest_point.2 = line closest_point.1) := by
  sorry

#check closest_point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l622_62297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_twelve_equals_product_l622_62226

theorem factorial_twelve_equals_product (n : ℕ) : 2^7 * 3^3 * 5 * 27720 = Nat.factorial 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_twelve_equals_product_l622_62226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l622_62234

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| - |x + 1|

-- Define the set of x that satisfies f(x) > 2 when a = 1
def solution_set : Set ℝ := {x | f 1 x > 2}

-- Define the function g used in part II
def g (a : ℝ) (x : ℝ) : ℝ := f a x + |x + 1| + x

-- Theorem for part I
theorem part_i : solution_set = {x | x < -2/3 ∨ x > 4} := by sorry

-- Theorem for part II
theorem part_ii : {a : ℝ | ∀ x, g a x > a^2 - 1/2} = {a | -1/2 < a ∧ a < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_l622_62234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_end_time_l622_62255

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  hk : hours < 24
  minK : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry, by sorry⟩

theorem movie_end_time (startTime : Time) (duration : Nat) :
  startTime.hours = 18 ∧ startTime.minutes = 30 ∧ duration = 120 →
  (addMinutes startTime duration).hours = 20 ∧ (addMinutes startTime duration).minutes = 30 := by
  sorry

#check movie_end_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_end_time_l622_62255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l622_62288

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with S₃ = 6 and S₆ = 54, the first term a₁ = 6/7 -/
theorem geometric_sequence_first_term :
  ∀ (a q : ℝ),
    geometric_sum a q 3 = 6 →
    geometric_sum a q 6 = 54 →
    a = 6/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l622_62288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logans_father_cartons_less_correct_l622_62238

/-- Proves the number of cartons less received given the conditions of Logan's father's milk delivery --/
def logans_father_cartons_less (
  normal_cartons : ℕ := 50
) (jars_per_carton : ℕ := 20
) (damaged_jars_per_carton : ℕ := 3
) (damaged_cartons : ℕ := 5
) (totally_damaged_cartons : ℕ := 1
) (good_jars_for_sale : ℕ := 565
) : ℕ := by
  let total_jars := normal_cartons * jars_per_carton
  let jars_after_total_damage := total_jars - (totally_damaged_cartons * jars_per_carton)
  let damaged_jars := damaged_jars_per_carton * damaged_cartons
  let jars_after_all_damage := jars_after_total_damage - damaged_jars
  let missing_jars := jars_after_all_damage - good_jars_for_sale
  let cartons_less : ℕ := missing_jars / jars_per_carton
  exact cartons_less

theorem logans_father_cartons_less_correct :
  logans_father_cartons_less = 20 := by
  rfl

#eval logans_father_cartons_less -- Should evaluate to 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logans_father_cartons_less_correct_l622_62238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_small_triangle_l622_62246

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

/-- A set of n points in the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- Check if a point is inside a triangle -/
noncomputable def pointInTriangle (p p1 p2 p3 : Point) : Prop :=
  let a1 := triangleArea p p1 p2
  let a2 := triangleArea p p2 p3
  let a3 := triangleArea p p3 p1
  let totalArea := triangleArea p1 p2 p3
  a1 + a2 + a3 ≤ totalArea

theorem points_in_small_triangle (n : ℕ) (points : PointSet n)
  (h : ∀ (i j k : Fin n), triangleArea (points i) (points j) (points k) ≤ 1) :
  ∃ (p1 p2 p3 : Point), (∀ (i : Fin n), triangleArea p1 p2 p3 ≤ 4 ∧
    pointInTriangle (points i) p1 p2 p3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_small_triangle_l622_62246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_is_one_hour_l622_62267

/-- The time taken to row to a place and back given the rowing speed, current speed, and distance. -/
noncomputable def rowingTime (rowingSpeed currentSpeed distance : ℝ) : ℝ :=
  let upstreamSpeed := rowingSpeed - currentSpeed
  let downstreamSpeed := rowingSpeed + currentSpeed
  distance / upstreamSpeed + distance / downstreamSpeed

/-- Theorem stating that under the given conditions, the total rowing time is 1 hour. -/
theorem rowing_time_is_one_hour :
  let rowingSpeed : ℝ := 5
  let currentSpeed : ℝ := 1
  let distance : ℝ := 2.4
  rowingTime rowingSpeed currentSpeed distance = 1 := by
  -- Unfold the definition of rowingTime
  unfold rowingTime
  -- Simplify the expression
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_is_one_hour_l622_62267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l622_62249

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Definition of the unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

/-- Chord length of the perpendicular line through the left focus -/
noncomputable def perpendicular_chord_length : ℝ := Real.sqrt 2

/-- Theorem about the ellipse C and its properties -/
theorem ellipse_properties :
  ∀ (l : Set (ℝ × ℝ)) (E F G H : ℝ × ℝ),
  (∀ x y, (x, y) ∈ l → (unit_circle x y ↔ (x, y) = E ∨ (x, y) = F) ∧ 
                       (ellipse_C x y ↔ (x, y) = G ∨ (x, y) = H)) →
  (∃ (chord_length : ℝ), 
    Real.sqrt 3 ≤ chord_length ∧ 
    chord_length ≤ 2 ∧
    (∀ (area : ℝ), area ≤ 1/2 → 
      ∃ (l' : Set (ℝ × ℝ)) (E' F' G' H' : ℝ × ℝ),
        (∀ x y, (x, y) ∈ l' → (unit_circle x y ↔ (x, y) = E' ∨ (x, y) = F') ∧ 
                               (ellipse_C x y ↔ (x, y) = G' ∨ (x, y) = H')) ∧
        area = abs (E'.1 * F'.2 - E'.2 * F'.1) / 2 ∧
        chord_length = Real.sqrt ((G'.1 - H'.1)^2 + (G'.2 - H'.2)^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l622_62249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_from_circumradius_and_ratio_l622_62213

/-- Given a triangle with circumradius r, side a, and ratio l of the other two sides,
    this theorem states the formulas for the lengths of sides b and c. -/
theorem triangle_sides_from_circumradius_and_ratio 
  (r a l : ℝ) 
  (hr : r > 0) 
  (ha : a > 0) 
  (hl : l > 0) 
  (ha_lt_2r : a < 2*r) :
  ∃ (b c : ℝ),
    b = (2*l*a*r) / Real.sqrt (4*r^2*(l^2 + 1) - 4*r*l*Real.sqrt (4*r^2 - a^2)) ∧
    c = (2*a*r) / Real.sqrt (4*r^2*(l^2 + 1) - 4*r*l*Real.sqrt (4*r^2 - a^2)) ∧
    b/c = l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_from_circumradius_and_ratio_l622_62213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l622_62264

noncomputable def h (f g : ℤ → ℝ) : ℤ → ℝ :=
  fun n => ⨆ k : ℤ, max (f (n - k) * g k) 0

theorem inequality_proof
  (f g : ℤ → ℝ)
  (hf : ∃ S : Finset ℤ, ∀ n ∉ S, f n = 0)
  (hg : ∃ S : Finset ℤ, ∀ n ∉ S, g n = 0)
  (hf_nonneg : ∀ n, f n ≥ 0)
  (hg_nonneg : ∀ n, g n ≥ 0)
  (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq : 1 / p + 1 / q = 1) :
  ∑' n, h f g n ≥ (∑' n, (f n) ^ p) ^ (1 / p) * (∑' n, (g n) ^ q) ^ (1 / q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l622_62264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_specific_lines_l622_62215

/-- The minimum distance between two parallel lines -/
noncomputable def min_distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Theorem: The minimum distance between the lines 3x + 4y - 10 = 0 and 6x + 8y + 5 = 0 is 5/2 -/
theorem min_distance_specific_lines :
  min_distance_parallel_lines 3 4 (-10) (5/2) = 5/2 := by
  sorry

#check min_distance_specific_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_specific_lines_l622_62215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_floor_value_l622_62242

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | n + 1 => (8/5) * a n + (6/5) * Real.sqrt (4^n - (a n)^2)

theorem a_10_floor_value : ⌊a 10⌋ = 983 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_floor_value_l622_62242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l622_62299

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 3/2

theorem max_value_of_f (a : ℝ) :
  (∃! (z₁ z₂ : ℝ), 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < π ∧ f a z₁ = 0 ∧ f a z₂ = 0) →
  (∀ x ∈ Set.Icc 0 (π/2), f a x ≤ a - 3/2) ∧
  (∃ x ∈ Set.Icc 0 (π/2), f a x = a - 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l622_62299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dexter_student_count_l622_62233

/-- The number of students in Sam's high school -/
def S : ℚ := sorry

/-- The number of students in Dexter's high school -/
def D : ℚ := sorry

/-- The number of students in the new school -/
def N : ℚ := sorry

/-- Dexter's high school has 4 times as many students as Sam's high school -/
axiom dexter_four_times_sam : D = 4 * S

/-- The total number of students in all three schools is 3600 -/
axiom total_students : D + S + N = 3600

/-- The new school has 400 fewer students than Sam's high school -/
axiom new_school_fewer : N = S - 400

/-- Theorem stating that Dexter's high school has 8000/3 students -/
theorem dexter_student_count : D = 8000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dexter_student_count_l622_62233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l622_62229

theorem exponential_inequality : 
  (1/2 : ℝ) ^ (1/2 : ℝ) < (1/2 : ℝ) ^ (1/3 : ℝ) ∧ 
  (1/2 : ℝ) ^ (1/3 : ℝ) < (1/2 : ℝ) ^ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l622_62229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l622_62294

theorem function_inequality (f : ℝ → ℝ) :
  (∀ x, f (Real.sin x - 1) ≥ Real.cos x ^ 2 + 2) →
  (∀ x, -2 ≤ x ∧ x ≤ 0 → f x ≥ -x^2 - 2*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l622_62294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_perpendicular_transitive_l622_62276

-- Define the types for lines and planes
structure Line : Type where

structure Plane : Type where

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular 
  (m n : Line) (a : Plane) 
  (h1 : perpendicular m a) 
  (h2 : parallel_line_plane n a) : 
  perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_perpendicular_transitive 
  (m : Line) (a b γ : Plane)
  (h1 : parallel_planes a b)
  (h2 : parallel_planes b γ)
  (h3 : perpendicular m a) :
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_perpendicular_transitive_l622_62276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l622_62225

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15

-- State the theorem
theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 240/841 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l622_62225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solutions_l622_62202

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the number of solutions
inductive NumSolutions
  | none
  | one
  | two

-- Define the function to determine the number of solutions
def determineSolutions (t : Triangle) : NumSolutions :=
  sorry

-- Scenario A
noncomputable def scenarioA : Triangle where
  a := 0  -- Unknown
  b := 19
  c := 0  -- Unknown
  A := 45 * Real.pi / 180
  B := 0  -- Unknown
  C := 30 * Real.pi / 180

-- Scenario B
noncomputable def scenarioB : Triangle where
  a := Real.sqrt 3
  b := 2 * Real.sqrt 2
  c := 0  -- Unknown
  A := 45 * Real.pi / 180
  B := 0  -- Unknown
  C := 0  -- Unknown

-- Scenario C
noncomputable def scenarioC : Triangle where
  a := 3
  b := 2 * Real.sqrt 2
  c := 0  -- Unknown
  A := 45 * Real.pi / 180
  B := 0  -- Unknown
  C := 0  -- Unknown

-- Scenario D
noncomputable def scenarioD : Triangle where
  a := 7
  b := 7
  c := 0  -- Unknown
  A := 75 * Real.pi / 180
  B := 0  -- Unknown
  C := 0  -- Unknown

-- Theorem statement
theorem triangle_solutions :
  (determineSolutions scenarioA = NumSolutions.one) ∧
  (determineSolutions scenarioB = NumSolutions.none) ∧
  (determineSolutions scenarioC = NumSolutions.one) ∧
  (determineSolutions scenarioD = NumSolutions.one) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solutions_l622_62202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l622_62271

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Dilates a point from the origin by a given factor -/
def dilate (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Theorem: The farthest vertex of the dilated square -/
theorem farthest_vertex_of_dilated_square (s : Square)
    (h1 : s.center = { x := 10, y := -5 })
    (h2 : s.area = 16)
    (h3 : (dilate { x := 36, y := -21 } (1/3)).y > (dilate { x := 36, y := -21 } (1/3)).x - 10) -- top side is horizontal
    : ∃ (v : Point), 
      v ∈ Set.range (λ (p : Point) ↦ dilate p 3) ∧ 
      (∀ (p : Point), p ∈ Set.range (λ (p : Point) ↦ dilate p 3) → 
        distance v { x := 0, y := 0 } ≥ distance p { x := 0, y := 0 }) ∧
      v = { x := 36, y := -21 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l622_62271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_container_emptying_l622_62211

/-- Represents a state of water distribution among three containers -/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a water transfer operation -/
inductive TransferOp
  | AB : TransferOp  -- Transfer from A to B
  | AC : TransferOp  -- Transfer from A to C
  | BA : TransferOp  -- Transfer from B to A
  | BC : TransferOp  -- Transfer from B to C
  | CA : TransferOp  -- Transfer from C to A
  | CB : TransferOp  -- Transfer from C to B

/-- Applies a transfer operation to a water state -/
def applyTransfer (state : WaterState) (op : TransferOp) : WaterState :=
  match op with
  | TransferOp.AB => ⟨state.a - min state.a state.b, state.b + min state.a state.b, state.c⟩
  | TransferOp.AC => ⟨state.a - min state.a state.c, state.b, state.c + min state.a state.c⟩
  | TransferOp.BA => ⟨state.a + min state.a state.b, state.b - min state.a state.b, state.c⟩
  | TransferOp.BC => ⟨state.a, state.b - min state.b state.c, state.c + min state.b state.c⟩
  | TransferOp.CA => ⟨state.a + min state.a state.c, state.b, state.c - min state.a state.c⟩
  | TransferOp.CB => ⟨state.a, state.b + min state.b state.c, state.c - min state.b state.c⟩

/-- Theorem: For any initial water state, there exists a sequence of transfers
    that results in at least one container being empty -/
theorem water_container_emptying (initial : WaterState) :
  ∃ (ops : List TransferOp), 
    let final := ops.foldl applyTransfer initial
    (final.a = 0) ∨ (final.b = 0) ∨ (final.c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_container_emptying_l622_62211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l622_62232

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x / (x^2 + x + 1)
  else Real.exp x - 3/4

theorem f_range : Set.range f = Set.Ioc (-3/4) (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l622_62232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_same_color_probability_l622_62204

theorem bob_same_color_probability :
  let total_marbles : ℕ := 9
  let colors : ℕ := 3
  let marbles_per_color : ℕ := 3
  let marbles_taken_each : ℕ := 3

  let total_ways : ℕ := (total_marbles.choose marbles_taken_each) *
                    ((total_marbles - marbles_taken_each).choose marbles_taken_each) *
                    ((total_marbles - 2 * marbles_taken_each).choose marbles_taken_each)

  let favorable_outcomes : ℕ := colors * ((total_marbles - 2 * marbles_taken_each).choose marbles_taken_each)

  (favorable_outcomes : ℚ) / (total_ways : ℚ) = 1 / 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_same_color_probability_l622_62204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_marks_percentage_l622_62250

theorem exam_marks_percentage :
  ∀ T P X : ℝ,
  (0.30 * T = P - 30) →
  (P = 120) →
  ((X / 100) * T = P + 15) →
  X = 45 :=
by
  intro T P X h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_marks_percentage_l622_62250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_60_equals_fahrenheit_140_l622_62244

-- Define the conversion function from Celsius to Fahrenheit
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := (c * 9 / 5) + 32

-- Theorem statement
theorem celsius_60_equals_fahrenheit_140 : 
  celsius_to_fahrenheit 60 = 140 := by
  -- Unfold the definition of celsius_to_fahrenheit
  unfold celsius_to_fahrenheit
  -- Simplify the arithmetic
  simp [mul_div_assoc, add_left_inj]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_60_equals_fahrenheit_140_l622_62244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l622_62296

/-- Represents a route with its total distance and speed information -/
structure Route where
  totalDistance : ℝ
  mainSpeed : ℝ
  schoolZoneDistance : ℝ
  schoolZoneSpeed : ℝ

/-- Calculates the travel time for a given route in hours -/
noncomputable def travelTime (r : Route) : ℝ :=
  (r.totalDistance - r.schoolZoneDistance) / r.mainSpeed + r.schoolZoneDistance / r.schoolZoneSpeed

/-- Converts hours to minutes -/
noncomputable def hoursToMinutes (hours : ℝ) : ℝ :=
  hours * 60

theorem route_time_difference : 
  let routeA : Route := { 
    totalDistance := 6,
    mainSpeed := 30,
    schoolZoneDistance := 0,
    schoolZoneSpeed := 30
  }
  let routeB : Route := {
    totalDistance := 5,
    mainSpeed := 40,
    schoolZoneDistance := 0.5,
    schoolZoneSpeed := 20
  }
  hoursToMinutes (travelTime routeA - travelTime routeB) = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l622_62296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_desktop_revenue_is_102000_l622_62281

def total_computers_per_week : ℕ := 72
def desktop_price : ℕ := 1000

def week1_laptop_ratio : ℚ := 1/2
def week1_netbook_ratio : ℚ := 1/3

def week2_laptop_ratio : ℚ := 2/5
def week2_netbook_ratio : ℚ := 1/5

def week3_laptop_ratio : ℚ := 3/10
def week3_netbook_ratio : ℚ := 1/2

def week4_laptop_ratio : ℚ := 1/10
def week4_netbook_ratio : ℚ := 1/4

def desktop_sales (laptop_ratio netbook_ratio : ℚ) : ℕ :=
  total_computers_per_week - (Int.toNat ⌊(total_computers_per_week : ℚ) * (laptop_ratio + netbook_ratio)⌋)

def total_desktop_revenue : ℕ :=
  (desktop_sales week1_laptop_ratio week1_netbook_ratio +
   desktop_sales week2_laptop_ratio week2_netbook_ratio +
   desktop_sales week3_laptop_ratio week3_netbook_ratio +
   desktop_sales week4_laptop_ratio week4_netbook_ratio) * desktop_price

theorem total_desktop_revenue_is_102000 : total_desktop_revenue = 102000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_desktop_revenue_is_102000_l622_62281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_transfers_suffice_l622_62279

/-- A type representing settlements around a circular lake. -/
structure Settlement where
  id : ℕ
deriving BEq, Repr

/-- A type representing ferry routes between settlements. -/
def FerryRoute := Settlement → Settlement → Prop

/-- A property that defines the ferry route connection rule. -/
def ValidFerryRoutes (route : FerryRoute) (settlements : List Settlement) : Prop :=
  ∀ i j k l,
    settlements.indexOf i < settlements.indexOf j ∧
    settlements.indexOf j < settlements.indexOf k ∧
    settlements.indexOf k < settlements.indexOf l →
    (route i j ↔ ¬ route k l)

/-- The main theorem stating that any two settlements can be connected with at most two transfers. -/
theorem two_transfers_suffice
  (settlements : List Settlement)
  (route : FerryRoute)
  (valid_routes : ValidFerryRoutes route settlements)
  (circular : settlements.length > 0 ∧ settlements.head? = settlements.getLast?)
  : ∀ start finish : Settlement,
    start ∈ settlements →
    finish ∈ settlements →
    ∃ path : List Settlement,
      path.length ≤ 4 ∧
      path.head? = some start ∧
      path.getLast? = some finish ∧
      ∀ (i j : Settlement), (i :: j :: []) ⊆ path → route i j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_transfers_suffice_l622_62279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_sum_ge_weight_of_first_l622_62273

/-- The weight of a polynomial is the number of its odd coefficients -/
def weight (P : Polynomial ℤ) : ℕ :=
  (P.support.filter (fun i => P.coeff i % 2 ≠ 0)).card

/-- Given a list of natural numbers, create a sum of polynomials (1+x)^i -/
noncomputable def sumOfPowers (l : List ℕ) : Polynomial ℤ :=
  l.foldl (fun acc i => acc + (1 + Polynomial.X : Polynomial ℤ) ^ i) 0

theorem weight_of_sum_ge_weight_of_first (l : List ℕ) (h : l.Sorted (· < ·)) :
  weight (sumOfPowers l) ≥ weight ((1 + Polynomial.X : Polynomial ℤ) ^ l.head!) := by
  sorry

#check weight_of_sum_ge_weight_of_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_sum_ge_weight_of_first_l622_62273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_divisible_by_four_l622_62258

theorem only_one_divisible_by_four : 
  ∃! x, (x = 777^2021 * 999^2021 - 1 ∨ 
         x = 999^2021 * 555^2021 - 1 ∨ 
         x = 555^2021 * 777^2021 - 1) ∧ 
        4 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_divisible_by_four_l622_62258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_theorem_l622_62201

noncomputable def ellipse_sum (h k a b : ℝ) : ℝ := a + k

theorem ellipse_sum_theorem :
  ∀ (h k a b : ℝ),
    a > 0 → b > 0 →
    (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
      ((x - 2)^2 + y^2).sqrt + ((x + 2)^2 + y^2).sqrt = (3 * Real.sqrt 2 + Real.sqrt 58)) →
    (5 - h)^2 / a^2 + (3 - k)^2 / b^2 = 1 →
    ellipse_sum h k a b = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_theorem_l622_62201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_two_element_set_l622_62207

theorem subset_count_two_element_set :
  ∀ (S : Finset α), Finset.card S = 2 → Finset.card (Finset.powerset S) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_two_element_set_l622_62207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l622_62298

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the function g as a composition of f and a translation
noncomputable def g (x : ℝ) : ℝ := f (x + 4)

-- State the theorem
theorem inverse_function_point (h : f 0 = -1) : 
  g⁻¹ (-1) = -4 ∧ g (g⁻¹ (-1)) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l622_62298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miller_rabin_complexity_l622_62228

/-- Miller-Rabin algorithm for primality testing -/
def miller_rabin (n : ℕ) : Bool := sorry

/-- Time complexity function for an algorithm -/
def time_complexity (f : ℕ → Bool) (n : ℕ) : ℝ := sorry

/-- Big O notation -/
def big_o (f g : ℕ → ℝ) : Prop :=
  ∃ c k, ∀ n ≥ k, f n ≤ c * g n

/-- Logarithm function -/
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem miller_rabin_complexity :
  big_o (time_complexity miller_rabin) (λ n => (log 2 (n : ℝ))^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miller_rabin_complexity_l622_62228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_one_year_l622_62274

/-- Pete's current age -/
def p : ℤ := sorry

/-- Claire's current age -/
def c : ℤ := sorry

/-- The number of years until the ratio of Pete's age to Claire's age is 3:2 -/
def x : ℤ := sorry

/-- Pete's age three years ago was twice Claire's age three years ago -/
axiom condition1 : p - 3 = 2 * (c - 3)

/-- Pete's age five years ago was three times Claire's age five years ago -/
axiom condition2 : p - 5 = 3 * (c - 5)

/-- The ratio of their ages after x years will be 3:2 -/
axiom future_ratio : (p + x) * 2 = (c + x) * 3

theorem age_ratio_in_one_year : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_one_year_l622_62274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l622_62263

-- Define the line
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | line p.1 p.2 ∧ parabola p.1 p.2}

-- Theorem statement
theorem intersection_distance : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l622_62263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_students_enrolled_l622_62282

/-- The percentage of students enrolled in biology classes -/
noncomputable def percentage_enrolled (total : ℝ) (not_enrolled : ℝ) : ℝ :=
  (total - not_enrolled) / total * 100

/-- Proof that 50% of students are enrolled in biology classes -/
theorem half_students_enrolled (total : ℝ) (not_enrolled : ℝ)
    (h1 : total = 880)
    (h2 : not_enrolled = 440.00000000000006) :
    percentage_enrolled total not_enrolled = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_students_enrolled_l622_62282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l622_62260

/-- The time it takes for two workers to complete a job together, given their individual work rates -/
noncomputable def time_to_complete_together (rate_a rate_b : ℝ) : ℝ :=
  1 / (rate_a + rate_b)

theorem job_completion_time 
  (rate_a rate_b : ℝ) 
  (h1 : rate_a = 2 * rate_b) 
  (h2 : rate_b = 1 / 12) : 
  time_to_complete_together rate_a rate_b = 4 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l622_62260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l622_62256

theorem circle_intersection (a : ℝ) (h₁ : a > 0) : 
  (∀ x y : ℝ, x^2 + (y - 1)^2 = a^2 ∧ (x - 2)^2 + y^2 = 4 → y = 2*x) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_l622_62256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_are_omega_l622_62224

noncomputable def exp (x : ℝ) : ℝ := Real.exp x

structure Point where
  x : ℝ
  y : ℝ

structure MyVector where
  x : ℝ
  y : ℝ

def vector_eq (v1 v2 : MyVector) : Prop :=
  v1.x = v2.x ∧ v1.y = v2.y

def is_omega_point (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ),
    let Q : Point := ⟨a, 0⟩
    let A : Point := ⟨x₁, exp x₁⟩
    let B : Point := ⟨x₂, exp x₂⟩
    let QA : MyVector := ⟨x₁ - a, exp x₁⟩
    let AB : MyVector := ⟨x₂ - x₁, exp x₂ - exp x₁⟩
    vector_eq QA AB

theorem all_points_are_omega : ∀ (a : ℝ), is_omega_point a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_are_omega_l622_62224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l622_62293

/-- Given a circle with equation x^2 + y^2 - 4x + 2y - 4 = 0, prove that its center is at (2, -1) and its radius is 3 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := fun x y ↦ x^2 + y^2 - 4*x + 2*y - 4 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.fst)^2 + (y - center.snd)^2 = radius^2 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l622_62293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daniel_drives_60_miles_l622_62275

/-- The distance Daniel drives back from work every day -/
noncomputable def D : ℝ := sorry

/-- The speed at which Daniel drives on Sunday -/
noncomputable def x : ℝ := sorry

/-- The time it takes Daniel to drive back from work on Sunday -/
noncomputable def T_sunday : ℝ := D / x

/-- The time it takes Daniel to drive the first 32 miles on Monday -/
noncomputable def T_32 : ℝ := 32 / (2 * x)

/-- The time it takes Daniel to drive the rest of the way on Monday -/
noncomputable def T_rest : ℝ := (D - 32) / (x / 2)

/-- The total time it takes Daniel to drive back from work on Monday -/
noncomputable def T_monday : ℝ := T_32 + T_rest

/-- Theorem stating that Daniel drives 60 miles back from work every day -/
theorem daniel_drives_60_miles :
  (T_monday = T_sunday * 1.20) → D = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daniel_drives_60_miles_l622_62275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_female_contestants_l622_62277

theorem probability_all_female_contestants (total : ℕ) (females : ℕ) (chosen : ℕ) 
  (h_total : total = 8) 
  (h_females : females = 5) 
  (h_chosen : chosen = 3) :
  (Nat.choose females chosen : ℚ) / (Nat.choose total chosen : ℚ) = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_female_contestants_l622_62277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brad_drank_five_glasses_l622_62227

/-- Represents the lemonade stand problem --/
def lemonade_stand_problem (glasses_per_gallon : ℕ) (cost_per_gallon : ℚ) 
  (gallons_made : ℕ) (price_per_glass : ℚ) (glasses_unsold : ℕ) (net_profit : ℚ) : Prop :=
  ∃ (glasses_drunk : ℕ),
    (let total_glasses := glasses_per_gallon * gallons_made
     let glasses_sold := total_glasses - glasses_unsold - glasses_drunk
     let revenue := glasses_sold * price_per_glass
     let total_cost := gallons_made * cost_per_gallon
     net_profit = revenue - total_cost) ∧ glasses_drunk = 5

/-- The main theorem to prove --/
theorem brad_drank_five_glasses :
  lemonade_stand_problem 16 (35/10) 2 1 6 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brad_drank_five_glasses_l622_62227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_triangle_l622_62219

noncomputable section

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def P : point := (5, 1)
def A : point := (0, 0)
def B : point := (8, 0)
def C : point := (4, 6)

theorem sum_distances_triangle :
  distance P A + distance P B + distance P C = 2 * Real.sqrt 26 + Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_triangle_l622_62219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_piles_l622_62247

theorem impossible_three_similar_piles :
  ∀ (x : ℝ), x > 0 →
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_piles_l622_62247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_pi_sixth_to_pi_half_l622_62289

theorem sin_range_pi_sixth_to_pi_half :
  ∃ (lower upper : ℝ), lower = 1/2 ∧ upper = 1 ∧
  (∀ x : ℝ, π/6 ≤ x ∧ x ≤ π/2 → lower ≤ Real.sin x ∧ Real.sin x ≤ upper) ∧
  (∃ x1 x2 : ℝ, π/6 ≤ x1 ∧ x1 ≤ π/2 ∧ π/6 ≤ x2 ∧ x2 ≤ π/2 ∧ 
   Real.sin x1 = lower ∧ Real.sin x2 = upper) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_pi_sixth_to_pi_half_l622_62289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_c_l622_62295

/-- Given a quadratic equation x^2 - 3x + c = 0 with roots in the form x = (3 ± √c) / 2, prove that c = 9/5 -/
theorem quadratic_roots_imply_c (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ ∃ ε : Bool, x = (3 + if ε then Real.sqrt c else -Real.sqrt c) / 2) →
  c = 9/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_c_l622_62295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_relation_exists_l622_62222

/-- Two rectangles with positive dimensions -/
structure TwoRectangles where
  a1 : ℝ
  b1 : ℝ
  a2 : ℝ
  b2 : ℝ
  ha1 : a1 > 0
  hb1 : b1 > 0
  ha2 : a2 > 0
  hb2 : b2 > 0

/-- The perimeter of the first rectangle is greater than the perimeter of the second -/
def perimeterRelation (r : TwoRectangles) : Prop :=
  2 * (r.a1 + r.b1) > 2 * (r.a2 + r.b2)

/-- Possible area relations between the two rectangles -/
inductive AreaRelation where
  | greater : AreaRelation
  | less : AreaRelation
  | equal : AreaRelation

/-- For any area relation, there exist two rectangles satisfying the perimeter condition
    and having that area relation -/
theorem area_relation_exists (ar : AreaRelation) :
    ∃ r : TwoRectangles, perimeterRelation r ∧
      match ar with
      | AreaRelation.greater => r.a1 * r.b1 > r.a2 * r.b2
      | AreaRelation.less => r.a1 * r.b1 < r.a2 * r.b2
      | AreaRelation.equal => r.a1 * r.b1 = r.a2 * r.b2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_relation_exists_l622_62222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_in_S_l622_62206

/-- Region S in the xy-plane -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 + 1/2 * p.2| ≤ 10 ∧ |p.1| ≤ 10 ∧ |p.2| ≤ 10}

/-- The radius of the largest circle centered at (0,0) that fits in S -/
noncomputable def r : ℝ := 4 * Real.sqrt 5

/-- Theorem: The area of the largest circle centered at (0,0) that can be fitted in region S is 80π -/
theorem largest_circle_area_in_S : 
  (π * r^2 : ℝ) = 80 * π := by
  -- Proof steps go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_in_S_l622_62206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_15_8_l622_62214

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem floor_expression_equals_15_8 :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 15.8 := by
  -- Convert integer literals to reals explicitly
  have h1 : (floor 6.5 : ℝ) = 6 := by sorry
  have h2 : (floor (2/3 : ℝ) : ℝ) = 0 := by sorry
  have h3 : (floor 2 : ℝ) = 2 := by sorry
  have h4 : (floor 8.3 : ℝ) = 8 := by sorry

  -- Rewrite the expression using the above equalities
  rw [h1, h2, h3, h4]

  -- Perform the calculation
  norm_num
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_15_8_l622_62214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_two_g_roots_inequality_l622_62209

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x - (x + 1) * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 - 1

-- Statement 1
theorem f_less_than_two : ∀ x : ℝ, x > 1 → f x < 2 := by
  sorry

-- Statement 2
theorem g_roots_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  g a x₁ = 0 → g a x₂ = 0 → x₁ ≠ x₂ →
  (Real.log x₁ + Real.log x₂) / 2 > 1 + 2 / Real.sqrt (x₁ * x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_two_g_roots_inequality_l622_62209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_triangle_area_ratio_l622_62237

/-- An equilateral triangle inscribed in a circle -/
structure InscribedEquilateralTriangle where
  circle : Real → Real → Real → Prop
  triangle : Real → Real → Real → Prop
  inscribed : Prop

/-- A smaller equilateral triangle with specific vertex positions relative to a larger one -/
structure SmallerTriangle (large : InscribedEquilateralTriangle) where
  small : Real → Real → Real → Prop
  coincident_vertex : Prop
  midpoint_vertex : Prop

/-- The theorem stating the area ratio of the two triangles -/
theorem smaller_triangle_area_ratio 
  (large : InscribedEquilateralTriangle) 
  (small : SmallerTriangle large) : 
  ∃ (area_small area_large : Real), 
    area_small / area_large = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_triangle_area_ratio_l622_62237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l622_62278

noncomputable section

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 3
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 3*x + 3*y - 3*m = 0

-- Define the common chord
def common_chord (x y m : ℝ) : Prop := x - y + m = 1

-- Define the triangle area
noncomputable def triangle_area (m : ℝ) : ℝ := (1/2) * |1 - m| * |m - 1|

theorem circle_intersection_theorem (m : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ∧ 
  (∃ x y : ℝ, common_chord x y m) ∧
  (triangle_area m = 2) →
  m = 3 ∨ m = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l622_62278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l622_62200

-- Define the equation
def equation (a : ℝ) (x : ℝ) : Prop :=
  a * abs (x + 1) + (x^2 - 5*x + 6) / (2 - x) = 0

-- Define the set of values for a
def a_values : Set ℝ :=
  Set.Icc (-1) (-1/3) ∪ Set.Ioo (-1/3) 1

-- Theorem statement
theorem unique_solution (a : ℝ) :
  (∃! x, equation a x) ↔ a ∈ a_values :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l622_62200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAM_l622_62265

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the points
def M : ℝ × ℝ := (0, 1)
def F : ℝ × ℝ := (1, 0)
def O : ℝ × ℝ := (0, 0)

-- Define point A as the intersection of MF and the parabola
noncomputable def A : ℝ × ℝ :=
  let x := 3 - 2 * Real.sqrt 2
  (x, 2 * Real.sqrt x)

-- Theorem statement
theorem area_of_triangle_OAM :
  let xA := A.1
  (1/2) * M.2 * xA = 3/2 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAM_l622_62265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l622_62261

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 12 then (1 - 2*a)*x + 5 else a^(x - 13)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ := f a n

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ m n : ℕ, m ≠ n → (m - n : ℝ) * (a_n a m - a_n a n) < 0) →
  (1/2 : ℝ) < a ∧ a < (2/3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l622_62261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l622_62236

-- Define the function g
noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

-- State the theorem
theorem unique_number_not_in_range_of_g 
  (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : g p q r s 13 = 13) 
  (h2 : g p q r s 61 = 61)
  (h3 : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l622_62236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_surface_area_l622_62210

/-- Represents a sphere inscribed in a cube. -/
structure Sphere where
  /-- The sphere is tangent to all faces of the cube. -/
  isTangentToCubeFaces : Prop

/-- Given a cube with volume 729 and an inscribed sphere touching the center of each face,
    the combined surface area of the cube and the sphere is 486 + 81π. -/
theorem cube_sphere_surface_area (cube_volume : ℝ) (sphere : Sphere) :
  cube_volume = 729 →
  sphere.isTangentToCubeFaces →
  let cube_side := (cube_volume ^ (1/3 : ℝ))
  let cube_surface_area := 6 * cube_side^2
  let sphere_surface_area := 4 * Real.pi * (cube_side/2)^2
  cube_surface_area + sphere_surface_area = 486 + 81 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_surface_area_l622_62210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sequence_formula_l622_62243

noncomputable def a_sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 10
  | k + 1 => 10 * Real.sqrt (a_sequence k)

theorem a_sequence_formula (n : ℕ) : 
  a_sequence n = 10 ^ (2 - (1/2 : ℝ) ^ (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sequence_formula_l622_62243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coincidences_value_l622_62221

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- The number of questions Vasya guessed correctly -/
def vasya_correct : ℕ := 6

/-- The number of questions Misha guessed correctly -/
def misha_correct : ℕ := 8

/-- The probability of Vasya guessing correctly -/
noncomputable def p_vasya : ℝ := vasya_correct / num_questions

/-- The probability of Misha guessing correctly -/
noncomputable def p_misha : ℝ := misha_correct / num_questions

/-- The probability of both Vasya and Misha guessing correctly or both guessing incorrectly -/
noncomputable def p_coincidence : ℝ := p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)

/-- The expected number of coincidences -/
noncomputable def expected_coincidences : ℝ := num_questions * p_coincidence

theorem expected_coincidences_value :
  expected_coincidences = 10.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coincidences_value_l622_62221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_travel_time_l622_62240

/-- Calculates the total travel time for a train journey with two segments and a waiting period. -/
theorem train_travel_time (x y : ℝ) : 
  (x / 50 + y / 70 + 0.5 : ℝ) = (7 * x + 5 * y) / 350 + 0.5 := by
  sorry

#check train_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_travel_time_l622_62240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_fourth_term_l622_62262

/-- A geometric series with fifth term 120 (5!) and sixth term 720 (6!) has fourth term 20 -/
theorem geometric_series_fourth_term : 
  ∀ (a r : ℕ),
  a * r^4 = 120 ∧ 
  a * r^5 = 720 →
  a * r^3 = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_fourth_term_l622_62262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_minus_one_minus_i_l622_62285

noncomputable def theta : ℝ := (5 * Real.pi) / 4

def z : ℂ := -1 - Complex.I

theorem complex_argument_minus_one_minus_i :
  Complex.arg z = theta := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_minus_one_minus_i_l622_62285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eden_carried_nine_buckets_l622_62203

-- Define the number of buckets each person carried
def eden_buckets : ℕ := 9
def mary_buckets : ℕ := eden_buckets + 3
def iris_buckets : ℕ := mary_buckets - 1

-- Define the total number of buckets
def total_buckets : ℕ := 34

-- State the theorem
theorem eden_carried_nine_buckets :
  eden_buckets + mary_buckets + iris_buckets = total_buckets ∧
  eden_buckets = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eden_carried_nine_buckets_l622_62203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_rent_percentage_l622_62269

/-- Represents Elaine's financial situation over two years --/
structure ElaineFinances where
  last_year_earnings : ℚ
  last_year_rent_percentage : ℚ
  earnings_increase_percentage : ℚ
  rent_increase_percentage : ℚ

/-- Calculates the percentage of earnings spent on rent this year --/
def rent_percentage_this_year (e : ElaineFinances) : ℚ :=
  (e.rent_increase_percentage / 100) * e.last_year_rent_percentage /
  (1 + e.earnings_increase_percentage / 100)

/-- Theorem stating that Elaine spent 30% of her earnings on rent this year --/
theorem elaine_rent_percentage
  (e : ElaineFinances)
  (h1 : e.last_year_rent_percentage = 20)
  (h2 : e.earnings_increase_percentage = 20)
  (h3 : e.rent_increase_percentage = 180) :
  rent_percentage_this_year e = 30 := by
  sorry

#eval rent_percentage_this_year { last_year_earnings := 1000, last_year_rent_percentage := 20, earnings_increase_percentage := 20, rent_increase_percentage := 180 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_rent_percentage_l622_62269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l622_62266

/-- A power function passing through the point (2, √2) has exponent 1/2 -/
theorem power_function_through_point (α : ℝ) : 
  (∀ x : ℝ, x > 0 → (fun x ↦ x^α) x = x^α) → 
  2^α = Real.sqrt 2 → 
  α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l622_62266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_line_l622_62272

/-- The hyperbola equation xy = 18 -/
def hyperbola_eq (x y : ℝ) : Prop := x * y = 18

/-- The circle equation x^2 + y^2 = 36 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 36

/-- A point (x, y) is an intersection point if it satisfies both the hyperbola and circle equations -/
def intersection_point (x y : ℝ) : Prop := hyperbola_eq x y ∧ circle_eq x y

/-- The set of all intersection points -/
def intersection_set : Set (ℝ × ℝ) :=
  {p | intersection_point p.1 p.2}

/-- The theorem stating that the intersection points form a straight line -/
theorem intersection_forms_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ), (x, y) ∈ intersection_set → a*x + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_line_l622_62272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l622_62284

/-- A parabola with equation y² = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ y^2 = 4*x

/-- A line with equation x - y - 1 = 0 -/
structure Line where
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ x - y - 1 = 0

/-- Intersection points of a parabola and a line -/
def IntersectionPoints (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | p.eq point.1 point.2 ∧ l.eq point.1 point.2}

/-- The length of a line segment between two points -/
noncomputable def LineSegmentLength (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

/-- Main theorem: The length of the line segment between intersection points is 8 -/
theorem intersection_segment_length (p : Parabola) (l : Line) :
  ∃ a b : ℝ × ℝ, a ∈ IntersectionPoints p l ∧ b ∈ IntersectionPoints p l ∧ a ≠ b ∧ LineSegmentLength a b = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l622_62284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_gcd_ratio_exists_for_six_not_five_l622_62218

/-- Arithmetic mean of a list of natural numbers -/
def arithmeticMean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Greatest common divisor of a list of natural numbers -/
def listGcd (list : List ℕ) : ℕ :=
  list.foldl Nat.gcd 0

/-- Existence of ten distinct natural numbers with specific arithmetic mean to GCD ratio -/
def distinct_numbers_mean_gcd_ratio (k : ℕ) : Prop :=
  ∃ (list : List ℕ),
    list.length = 10 ∧
    list.Nodup ∧
    (arithmeticMean list = k * (listGcd list : ℚ))

theorem mean_gcd_ratio_exists_for_six_not_five :
    distinct_numbers_mean_gcd_ratio 6 ∧ ¬(distinct_numbers_mean_gcd_ratio 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_gcd_ratio_exists_for_six_not_five_l622_62218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_AD_is_385_l622_62286

/-- The number of people in Factory A -/
def factory_A : ℕ := sorry

/-- The number of people in Factory B -/
def factory_B : ℕ := sorry

/-- The number of people in Factory C -/
def factory_C : ℕ := sorry

/-- The number of people in Factory D -/
def factory_D : ℕ := sorry

/-- The sum of people in Factory A and Factory B is 283 -/
axiom sum_AB : factory_A + factory_B = 283

/-- The sum of people in Factory B and Factory C is 386 -/
axiom sum_BC : factory_B + factory_C = 386

/-- The sum of people in Factory C and Factory D is 488 -/
axiom sum_CD : factory_C + factory_D = 488

/-- The theorem stating that the sum of people in Factory A and Factory D is 385 -/
theorem sum_AD_is_385 : factory_A + factory_D = 385 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_AD_is_385_l622_62286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_z_axis_with_distance_7_l622_62223

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A point is on the z-axis if its x and y coordinates are zero -/
def onZAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0

theorem point_on_z_axis_with_distance_7 
  (A : Point3D)
  (h_A : A = ⟨-2, 3, 4⟩)
  (B : Point3D)
  (h_B_z : onZAxis B)
  (h_dist : distance A B = 7) :
  B = ⟨0, 0, -2⟩ ∨ B = ⟨0, 0, 10⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_z_axis_with_distance_7_l622_62223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_30_60_90_triangle_l622_62205

theorem area_30_60_90_triangle (DF : ℝ) (h : DF = 8) :
  let DE := DF / (2 : ℝ)
  let EF := DF * Real.sqrt 3 / 2
  (1 / 2 : ℝ) * DE * DF = 32 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_30_60_90_triangle_l622_62205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_top_left_value_l622_62259

def grid_sum (n : ℕ) (t : ℕ) : ℕ → ℕ → ℕ 
| 0, _ => t
| _, 0 => t
| i+1, j+1 => grid_sum n t i (j+1) + grid_sum n t (i+1) j

theorem unique_top_left_value : 
  ∃! t : ℕ, grid_sum 5 t 5 5 = 2016 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_top_left_value_l622_62259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_enrollment_l622_62252

/-- Represents the number of players in a set -/
def PlayerCount := Nat

instance : OfNat PlayerCount n := ⟨n⟩
instance : Add PlayerCount := ⟨Nat.add⟩
instance : Sub PlayerCount := ⟨Nat.sub⟩
instance : LE PlayerCount := ⟨Nat.le⟩

theorem soccer_team_enrollment (total : PlayerCount) (physics : PlayerCount) (both : PlayerCount) 
  (h1 : total = 15)
  (h2 : physics = 9)
  (h3 : both = 4)
  (h4 : physics ≥ both)
  (h5 : total ≥ physics) :
  ∃ (math : PlayerCount), math = total - (physics - both) ∧ math = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_enrollment_l622_62252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l622_62253

/-- The area of the triangle formed by point M and vertices of the equilateral triangle -/
noncomputable def area_triangle_from_center (a d : ℝ) : ℝ :=
  sorry

/-- Given an equilateral triangle ABC with side length a and a point M at distance d
    from its center, the area S of triangle MAB is (√3 / 12) |a^2 - 3d^2| -/
theorem equilateral_triangle_area (a d : ℝ) (h_a : a > 0) :
  let S := area_triangle_from_center a d
  S = (Real.sqrt 3 / 12) * abs (a^2 - 3 * d^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l622_62253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_g_2x_l622_62280

noncomputable def g (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem g_composition_equals_g_2x (x : ℝ) 
  (h : -1 < x ∧ x < 1) : 
  g ((4 * x + x^2) / (1 + 4 * x + x^2)) = g (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_g_2x_l622_62280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_100th_group_l622_62248

-- Define the sequence
def our_sequence (n : ℕ) : ℕ := 2^(n-1)

-- Define the grouping function
def group_size (n : ℕ) : ℕ := n

-- Define the sum of group sizes up to n
def sum_group_sizes (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement
theorem first_number_100th_group :
  our_sequence (sum_group_sizes 99 + 1) = 2^4950 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_100th_group_l622_62248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_plus_pi_third_l622_62290

theorem cos_two_theta_plus_pi_third (θ : Real) 
  (h : Real.sin θ + Real.sin (θ + Real.pi/3) = 1) : 
  Real.cos (2*θ + Real.pi/3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_plus_pi_third_l622_62290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_greater_than_half_alpha_l622_62208

theorem beta_greater_than_half_alpha (α β : Real) 
  (h1 : Real.sin β = (3/4) * Real.sin α) 
  (h2 : α ≤ 90 * π / 180) : β > α / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_greater_than_half_alpha_l622_62208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l622_62231

/-- The area of a triangle with vertices (x₁, y₁), (x₂, y₂), and (x₃, y₃) -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|

/-- Theorem: The area of a triangle with vertices at (2, 3), (9, 3), and (5, 12) is 31.5 square units -/
theorem triangle_area_example : triangleArea 2 3 9 3 5 12 = 31.5 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l622_62231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_product_l622_62239

-- Define the ellipse
def is_ellipse (x y m n : ℝ) : Prop := x^2 / m + y^2 / n = 1

-- Define the hyperbola
def is_hyperbola (x y a b : ℝ) : Prop := x^2 / a - y^2 / b = 1

-- Define the property of having the same foci
def same_foci (e h : ℝ × ℝ → Prop) : Prop := 
  ∃ f₁ f₂ : ℝ × ℝ, ∀ p, (e p ∨ h p) → 
    (p.1 - f₁.1)^2 + (p.2 - f₁.2)^2 = (p.1 - f₂.1)^2 + (p.2 - f₂.2)^2

theorem ellipse_hyperbola_intersection_product (m n a b : ℝ) 
    (hm : m > n) (hn : n > 0) (ha : a > b) (hb : b > 0) :
  same_foci (λ p ↦ is_ellipse p.1 p.2 m n) (λ p ↦ is_hyperbola p.1 p.2 a b) →
  ∃ P F₁ F₂ : ℝ × ℝ, 
    is_ellipse P.1 P.2 m n ∧ 
    is_hyperbola P.1 P.2 a b ∧
    ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = (m - a)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_product_l622_62239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_zero_l622_62287

/-- The circle representing the locus of point P -/
def circle_locus (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

/-- The coordinates of point Q -/
def Q (a : ℝ) : ℝ × ℝ := (3*a, 4*a + 5)

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_is_zero :
  ∃ a : ℝ, ∀ x y : ℝ, circle_locus x y → distance x y (Q a).1 (Q a).2 ≥ 0 ∧ 
  (∃ x₀ y₀ : ℝ, circle_locus x₀ y₀ ∧ distance x₀ y₀ (Q a).1 (Q a).2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_zero_l622_62287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_range_in_interval_l622_62216

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + Real.sin x * Real.cos x + 2 * Real.sin (Real.pi/4 - x) * Real.cos (Real.pi/4 - x)

theorem f_value_at_pi_third :
  f (Real.pi/3) = (Real.sqrt 3 + 1) / 4 := by sorry

theorem f_range_in_interval :
  ∀ x ∈ Set.Icc (Real.pi/12) (Real.pi/4), 1 ≤ f x ∧ f x ≤ (Real.sqrt 2 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_range_in_interval_l622_62216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_4pi_circle_equation_has_solution_l622_62245

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 9 * x + 6 * y + 27 = 0

-- Define the area of the circle
noncomputable def circle_area : ℝ := 4 * Real.pi

-- Theorem statement
theorem circle_area_is_4pi :
  ∃ (x y : ℝ), circle_equation x y ∧ circle_area = 4 * Real.pi := by
  -- Proof goes here
  sorry

-- Additional theorem to show the existence of a solution
theorem circle_equation_has_solution :
  ∃ (x y : ℝ), circle_equation x y := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_4pi_circle_equation_has_solution_l622_62245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_triple_characterization_l622_62291

theorem gcd_triple_characterization (a b c : ℕ) :
  a > 0 → b > 0 → c > 0 →
  Nat.gcd a 20 = b ∧ Nat.gcd b 15 = c ∧ Nat.gcd a c = 5 →
  (∃ (k : ℕ), k > 0 ∧ k % 2 = 1 ∧ a = 20 * k ∧ b = 20 ∧ c = 5) ∨
  (∃ (t : ℕ), t > 0 ∧ t % 2 = 1 ∧ a = 10 * t ∧ b = 10 ∧ c = 5) ∨
  (∃ (m : ℕ), m > 0 ∧ a = 5 * m ∧ b = 5 ∧ c = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_triple_characterization_l622_62291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intercepts_sum_l622_62270

/-- The parabola function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

/-- The y-intercept -/
def d : ℝ := f 0

/-- The x-intercepts -/
noncomputable def e : ℝ := (9 + Real.sqrt 33) / 6
noncomputable def g : ℝ := (9 - Real.sqrt 33) / 6

theorem parabola_intercepts_sum : d + e + g = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intercepts_sum_l622_62270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_probability_l622_62212

/-- Represents a right circular cone -/
structure RightCircularCone where
  base : ℝ  -- Area of the base
  height : ℝ  -- Height of the cone
  base_positive : base > 0
  height_positive : height > 0

/-- Volume of a right circular cone -/
noncomputable def volume (cone : RightCircularCone) : ℝ := (1/3) * cone.base * cone.height

/-- The probability that a point P is taken within a right circular cone (M-ABC) 
    such that the volume of cone (P-ABC) is not greater than one-third of the 
    volume of cone (M-ABC) -/
theorem cone_volume_probability (cone : RightCircularCone) : 
  (19 : ℝ) / 27 = 19 / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_probability_l622_62212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_distance_theorem_l622_62292

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_fixed_distance (p a b : V) : Prop :=
  ∃ (c : ℝ), ∀ (x : V), ‖p - x‖ = c → x = (9/8 : ℝ) • a - (1/8 : ℝ) • b

theorem fixed_distance_theorem (a b : V) :
  ∀ p : V, ‖p - b‖ = 3 * ‖p - a‖ → is_fixed_distance p a b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_distance_theorem_l622_62292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_ziba_numbers_l622_62257

/-- A natural number is ziba if every natural number up to and including it
    can be expressed as the sum of some of its positive and distinct divisors. -/
def IsZiba (m : ℕ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ m → ∃ s : Finset ℕ, 
    (∀ d, d ∈ s → d ∣ m ∧ d > 0) ∧ 
    (∀ d₁ d₂, d₁ ∈ s → d₂ ∈ s → d₁ ≠ d₂ → d₁ ≠ d₂) ∧
    (s.sum id = n)

/-- There exist infinitely many natural numbers k such that k^2 + k + 2022 is a ziba number. -/
theorem infinitely_many_ziba_numbers : 
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ IsZiba (k^2 + k + 2022) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_ziba_numbers_l622_62257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l622_62283

noncomputable def distinct_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun p : ℕ × ℕ => 
    let (x, y) := p
    0 < x ∧ x < y ∧ (Real.sqrt 2916 = Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ)))
  (Finset.product (Finset.range 1000) (Finset.range 1000))

theorem distinct_pairs_count : Finset.card distinct_pairs = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l622_62283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_path_theorem_l622_62217

/-- Represents the number of distinct paths of exactly n jumps ending at E on an octagon -/
noncomputable def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then 0
  else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2

/-- Theorem stating the properties of the frog's path on the octagon -/
theorem frog_path_theorem :
  (∀ n : ℕ, n ≥ 1 → a (2*n - 1) = 0) ∧
  (∀ n : ℕ, n ≥ 1 → a (2*n) = ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1)) / Real.sqrt 2) := by
  sorry

#check frog_path_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_path_theorem_l622_62217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l622_62268

/-- A function f defined on ℝ with a parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2^(|x - m|) + 1

/-- The theorem stating the inequality for the given function -/
theorem function_inequality (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m 0 < f m (Real.log 2 / Real.log (1/2)) ∧
  f m (Real.log 2 / Real.log (1/2)) < f m 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l622_62268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fourth_term_value_l622_62241

/-- Given a sequence {aₙ} satisfying certain conditions, prove that a₄ = 3 -/
theorem sequence_fourth_term_value (a : ℕ → ℕ) (lambda : ℕ) :
  a 1 = 1 →
  a 2 = 2 →
  (∀ n : ℕ, n > 0 → a (n + 1) * a n = n * lambda) →
  a 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fourth_term_value_l622_62241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l622_62251

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x - 1) / (Real.sqrt x + 1)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ -1 ≤ y ∧ y < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l622_62251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_growth_perimeter_after_two_growths_perimeter_after_four_growths_l622_62220

/-- Perimeter after n growth operations -/
def perimeter_after_growth (n : ℕ) : ℚ :=
  27 * (4/3)^n

/-- The initial equilateral triangle has side length 9 -/
def initial_side_length : ℚ := 9

/-- The theorem to be proved -/
theorem perimeter_growth (n : ℕ) :
  perimeter_after_growth n = initial_side_length * 3 * (4/3)^n := by
  sorry

/-- After two growth operations, the perimeter is 48 -/
theorem perimeter_after_two_growths :
  perimeter_after_growth 2 = 48 := by
  sorry

/-- After four growth operations, the perimeter is 256/3 -/
theorem perimeter_after_four_growths :
  perimeter_after_growth 4 = 256/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_growth_perimeter_after_two_growths_perimeter_after_four_growths_l622_62220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_approx_48_minutes_l622_62230

/-- Represents the tank filling problem with given parameters -/
structure TankProblem where
  capacity : ℝ
  initialFill : ℝ
  inflowRate : ℝ
  outflowRate1 : ℝ
  outflowRate2 : ℝ

/-- Calculates the time to fill the tank completely -/
noncomputable def timeToFill (p : TankProblem) : ℝ :=
  let remainingVolume := p.capacity - p.initialFill
  let netInflowRate := p.inflowRate - (p.outflowRate1 + p.outflowRate2)
  remainingVolume / netInflowRate

/-- Theorem stating the time to fill the tank is approximately 48 minutes -/
theorem tank_fill_time_approx_48_minutes :
  let p : TankProblem := {
    capacity := 8000,
    initialFill := 4000,
    inflowRate := 0.5,
    outflowRate1 := 0.25,
    outflowRate2 := 1/6
  }
  abs (timeToFill p - 48) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_approx_48_minutes_l622_62230
