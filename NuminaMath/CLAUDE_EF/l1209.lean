import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_120_degrees_l1209_120910

/-- Represents a time on an analog clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  inv_hours : hours < 24
  inv_minutes : minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position --/
noncomputable def hourHandAngle (t : ClockTime) : ℝ :=
  (t.hours % 12 + t.minutes / 60 : ℝ) * 360 / 12

/-- Calculates the angle of the minute hand from 12 o'clock position --/
noncomputable def minuteHandAngle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 360 / 60

/-- Calculates the angle between the hour and minute hands --/
noncomputable def angleBetweenHands (t : ClockTime) : ℝ :=
  abs (hourHandAngle t - minuteHandAngle t)

/-- Rounds a real number to the nearest integer --/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Theorem: The clock hands form a 120° angle at 7:16 and 8:00 --/
theorem clock_hands_120_degrees : 
  ∃ (t1 t2 : ClockTime),
    t1.hours = 7 ∧ 
    t2.hours = 8 ∧
    t2.minutes = 0 ∧
    roundToNearest (angleBetweenHands t1) = 120 ∧
    roundToNearest (angleBetweenHands t2) = 120 ∧
    t1.minutes = 16 ∧
    ∀ (t : ClockTime), 
      (t.hours = 7 ∨ t.hours = 8) → 
      roundToNearest (angleBetweenHands t) = 120 → 
      (t = t1 ∨ t = t2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_120_degrees_l1209_120910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_in_first_round_l1209_120967

/-- Represents the number of fish caught in the first round -/
def first_round : ℝ := 8

/-- Represents the number of fish caught in the second round -/
def second_round : ℝ := first_round + 12

/-- Represents the number of fish caught in the third round -/
def third_round : ℝ := second_round + 0.6 * second_round

/-- The total number of fish caught in all three rounds -/
def total_fish : ℝ := 60

theorem fish_in_first_round :
  first_round + second_round + third_round = total_fish :=
by
  -- Expand the definitions
  unfold first_round second_round third_round total_fish
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_in_first_round_l1209_120967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1209_120988

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 40 ∧ 
  time = 0.9999200063994881 ∧ 
  speed = (length / 1000) / (time / 3600) →
  ∃ ε > 0, |speed - 144| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1209_120988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1209_120902

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculates the slope between two points -/
noncomputable def slopeBetween (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

theorem line_equation_proof (A : Point) (m : ℝ) (l : Line) :
  A.x = 3 →
  A.y = -2 →
  m = -4/3 →
  l.a = 4 →
  l.b = 3 →
  l.c = -6 →
  pointOnLine A l ∧ 
  ∀ (B : Point), B ≠ A → pointOnLine B l → slopeBetween A B = m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1209_120902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_condition_l1209_120963

-- Define the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (m, 3*m - 4)
def b : ℝ × ℝ := (1, 2)

-- Define the condition for vectors to be a basis
def is_basis (m : ℝ) : Prop :=
  ∀ c : ℝ × ℝ, ∃! (lambda mu : ℝ), c = lambda • (a m) + mu • b

-- Theorem statement
theorem basis_condition (m : ℝ) : 
  is_basis m ↔ m ∈ Set.univ \ {4} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_condition_l1209_120963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l1209_120931

theorem sine_cosine_identity (α : Real) : 
  (Real.sin (Real.pi + α))^2 - Real.cos (Real.pi + α) * Real.cos (-α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l1209_120931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_alex_sam_sum_l1209_120930

/-- Rounds a number to the nearest multiple of 5, rounding 2.5s up -/
def roundToNearestFive (n : ℤ) : ℤ :=
  5 * ((n + 2) / 5)

/-- Sums up integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums up integers from 1 to n after rounding each to nearest multiple of 5 -/
def sumRoundedIntegers (n : ℕ) : ℤ :=
  List.sum (List.map (roundToNearestFive ∘ Int.ofNat) (List.range n))

theorem difference_alex_sam_sum :
  (sumIntegers 150 : ℤ) - sumRoundedIntegers 150 = 10875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_alex_sam_sum_l1209_120930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1209_120929

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part 1: Maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (M : ℝ), M = -1 ∧ ∀ x, x > 0 → f 0 x ≤ M :=
sorry

-- Part 2: Range of a for exactly one zero
theorem one_zero_iff_a_positive (a : ℝ) :
  (∃! x, x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1209_120929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1209_120903

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 - x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (let (x₀, y₀) := point
     y₀ = f x₀ ∧
     m = (deriv f) x₀ ∧
     y₀ = m * x₀ + b) ∧
    (3 : ℝ) * point.1 - point.2 - 2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1209_120903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_motion_l1209_120993

/-- Point on a unit circle -/
structure UnitCirclePoint where
  x : ℝ
  y : ℝ
  on_circle : x^2 + y^2 = 1

/-- Motion of a point on a unit circle -/
structure CircleMotion where
  p : ℝ → UnitCirclePoint
  ω : ℝ
  counterclockwise : ∀ t, p t = ⟨Real.cos (ω * t), Real.sin (ω * t), by sorry⟩

/-- Derived point Q based on point P -/
def Q (p : UnitCirclePoint) : UnitCirclePoint where
  x := -2 * p.x * p.y
  y := p.y^2 - p.x^2
  on_circle := by sorry

/-- Theorem stating the motion of point Q -/
theorem q_motion (m : CircleMotion) :
  ∃ ω' : ℝ, ∀ t, Q (m.p t) = ⟨Real.cos (-2 * ω' * t), Real.sin (-2 * ω' * t), by sorry⟩ ∧ ω' = m.ω := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_motion_l1209_120993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1209_120974

noncomputable def arithmetic_sequence (a : ℝ) (b : ℕ → ℝ) :=
  ∀ n, Real.log (b (n + 1)) / Real.log a - Real.log (b n) / Real.log a = 1

theorem range_of_a (a : ℝ) (b : ℕ → ℝ) (a_seq : ℕ → ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  arithmetic_sequence a b ∧
  Real.log (b 1) / Real.log a = 2 ∧
  (∀ n, a_seq (n + 1) > a_seq n) ∧
  (∀ n, a_seq n = b n * (Real.log a / Real.log (b n))) →
  (a ∈ Set.Ioo 0 (2/3) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1209_120974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l1209_120952

theorem inequality_solution_range :
  ∀ a : ℝ, (∀ x : ℝ, a^2 + 2*a - Real.sin x^2 - 2*a*(Real.cos x) > 2) ↔ 
  (a < -2 - Real.sqrt 6 ∨ a > Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l1209_120952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l1209_120901

/-- The area of a triangle given its three vertices in 3D space -/
noncomputable def triangleArea (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, _) := A
  let (x₂, y₂, _) := B
  let (x₃, y₃, _) := C
  (1/2) * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

/-- Theorem: The area of a triangle with vertices at (0, 3, 6), (-2, 2, 2), and (-5, 5, 2) is 4.5 -/
theorem triangle_area_specific : 
  triangleArea (0, 3, 6) (-2, 2, 2) (-5, 5, 2) = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l1209_120901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increase_for_16x_l1209_120934

/-- Represents the rate of change between x and y -/
noncomputable def rate_of_change (Δx Δy : ℝ) : ℝ := Δy / Δx

/-- Theorem: Given a linear relationship where y increases by 9 units for every 4 unit increase in x,
    when x increases by 16 units, y will increase by 36 units. -/
theorem y_increase_for_16x (Δy : ℝ) :
  rate_of_change 4 9 = rate_of_change 16 Δy → Δy = 36 := by
  intro h
  -- The proof steps would go here
  sorry

#check y_increase_for_16x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increase_for_16x_l1209_120934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_approx_l1209_120965

noncomputable section

/-- The circumference of the smaller circle in meters -/
def c1 : ℝ := 268

/-- The circumference of the larger circle in meters -/
def c2 : ℝ := 380

/-- The mathematical constant pi -/
def π : ℝ := Real.pi

/-- The radius of the smaller circle -/
noncomputable def r1 : ℝ := c1 / (2 * π)

/-- The radius of the larger circle -/
noncomputable def r2 : ℝ := c2 / (2 * π)

/-- The area of the smaller circle -/
noncomputable def a1 : ℝ := π * r1^2

/-- The area of the larger circle -/
noncomputable def a2 : ℝ := π * r2^2

/-- The difference between the areas of the larger and smaller circles -/
noncomputable def area_difference : ℝ := a2 - a1

theorem circle_area_difference_approx :
  |area_difference - 5778.76| < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_approx_l1209_120965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_trajectory_l1209_120940

/-- Given two points A and B, and two lines intersecting at points C and D,
    prove that the trajectory of the intersection point M of AC and BD
    satisfies a specific equation. -/
theorem intersection_trajectory
  (A B C D M : ℝ × ℝ)
  (circle : Set (ℝ × ℝ))
  (line_AC line_BD : ℝ → ℝ)
  (h_A : A = (-2, 0))
  (h_B : B = (2, 0))
  (h_C : C.1 = 2)
  (h_D : D.1 = -2)
  (h_circle : circle = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4})
  (h_tangent : ∃ p ∈ circle, (D.2 - C.2) * p.1 + (C.1 - D.1) * p.2 + (D.1 * C.2 - C.1 * D.2) = 0)
  (h_AC : ∀ x, line_AC x = (x - A.1) * (C.2 - A.2) / (C.1 - A.1) + A.2)
  (h_BD : ∀ x, line_BD x = (x - B.1) * (D.2 - B.2) / (D.1 - B.1) + B.2)
  (h_M : M.2 = line_AC M.1 ∧ M.2 = line_BD M.1)
  (h_M_nonzero : M.2 ≠ 0) :
  M.1^2 / 4 + M.2^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_trajectory_l1209_120940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_over_2_l1209_120971

noncomputable def A : ℝ × ℝ × ℝ := (0, 5, 8)
noncomputable def B : ℝ × ℝ × ℝ := (-2, 4, 4)
noncomputable def C : ℝ × ℝ × ℝ := (-3, 7, 4)

noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let (x3, y3, z3) := p3
  (1/2) * abs (
    (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) +
    (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1) +
    (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
  )

theorem triangle_area_is_15_over_2 : triangle_area A B C = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_over_2_l1209_120971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1209_120999

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → (a ≥ Real.sqrt 3 ∨ a ≤ -Real.sqrt 3) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1209_120999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1209_120905

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem max_value_of_f :
  ∃ (max : ℝ), max = 20/21 ∧
  (∀ x y : ℝ, 1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/2 ≤ y ∧ y ≤ 5/8 → f x y ≤ max) ∧
  (∃ x y : ℝ, 1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/2 ≤ y ∧ y ≤ 5/8 ∧ f x y = max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1209_120905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1209_120983

/-- The standard equation of a hyperbola passing through (2,1) with asymptotes tangent to x^2 + (y-1)^2 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (x^2 / (11/3) - y^2 / 11 = 1) ↔ 
  ((x = 2 ∧ y = 1) ∨ 
   (∃ (t : ℝ), x^2 + (y-1)^2 = 1 ∧ 
    (y = Real.sqrt (11/3) * x + t ∨ y = -Real.sqrt (11/3) * x + t) ∧
    (x^2 + (y-t-1)^2 = 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1209_120983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1209_120942

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →
  distance A focus = distance B focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1209_120942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrent_lines_l1209_120982

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def point_on_segment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if two lines intersect -/
def lines_intersect (l1 l2 : Line) : Prop := sorry

/-- Checks if a quadrilateral has an incircle -/
def has_incircle (q : Quadrilateral) : Prop := sorry

/-- Checks if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Checks if a point is on a line -/
def point_on_line (P : Point) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem concurrent_lines 
  (ABCD : Quadrilateral) 
  (P Q R S : Point) 
  (PQ SR AC : Line) 
  (O : Point)
  (h_convex : is_convex ABCD)
  (h_P : point_on_segment P ABCD.A ABCD.B)
  (h_Q : point_on_segment Q ABCD.B ABCD.C)
  (h_R : point_on_segment R ABCD.C ABCD.D)
  (h_S : point_on_segment S ABCD.D ABCD.A)
  (h_intersect : lines_intersect PQ SR)
  (h_O_PQ : point_on_line O PQ)
  (h_O_SR : point_on_line O SR)
  (h_APOS : has_incircle ⟨ABCD.A, P, O, S⟩)
  (h_BQOP : has_incircle ⟨ABCD.B, Q, O, P⟩)
  (h_CROQ : has_incircle ⟨ABCD.C, R, O, Q⟩)
  (h_DSOR : has_incircle ⟨ABCD.D, S, O, R⟩)
  : are_concurrent AC PQ SR :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrent_lines_l1209_120982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1209_120928

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.B ∧
  t.b = 2 * Real.sqrt 3 ∧
  (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = π/3 ∧ 
  t.a + t.b + t.c = 6 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1209_120928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_and_arithmetic_sequence_l1209_120923

-- Define what it means for three numbers to be in arithmetic sequence
def arithmetic_sequence (α β γ : ℝ) : Prop := β - α = γ - β

-- State the theorem
theorem sin_equation_and_arithmetic_sequence :
  (∀ α β γ : ℝ, arithmetic_sequence α β γ → Real.sin (α + γ) = Real.sin (2 * β)) ∧
  (∃ α β γ : ℝ, Real.sin (α + γ) = Real.sin (2 * β) ∧ ¬ arithmetic_sequence α β γ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_and_arithmetic_sequence_l1209_120923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l1209_120998

/-- A circle with center (0,k) where k > 8 is tangent to the lines y = 2x, y = -2x, and y = 8.
    This theorem proves that the radius of such a circle is 8√5 + 8. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 8) :
  let center := (0, k)
  let r := Real.sqrt 5 * 8 + 8
  let circle := Metric.sphere center r
  let line1 := {p : ℝ × ℝ | p.2 = 2 * p.1}
  let line2 := {p : ℝ × ℝ | p.2 = -2 * p.1}
  let line3 := {p : ℝ × ℝ | p.2 = 8}
  (∃ (p : ℝ × ℝ), p ∈ circle ∩ line1) ∧
  (∃ (p : ℝ × ℝ), p ∈ circle ∩ line2) ∧
  (∃ (p : ℝ × ℝ), p ∈ circle ∩ line3) →
  r = Real.sqrt 5 * 8 + 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_tangent_to_lines_l1209_120998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_function_value_l1209_120924

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the rotated function g
noncomputable def g (x : ℝ) : ℝ := 2^(-x)

-- Theorem statement
theorem rotated_function_value : g (-2) = 4 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- This should evaluate to 4, but we'll use sorry to skip the detailed proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_function_value_l1209_120924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_factorial_l1209_120907

theorem factorial_of_factorial (n : ℕ) : (n.factorial.factorial) / n.factorial = (n.factorial - 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_of_factorial_l1209_120907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_for_f_l1209_120921

def horner_polynomial (a b c d : ℝ) (x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

def f (x : ℝ) : ℝ := 2 * x^3 + x - 3

theorem horner_method_for_f :
  ∃ (a b c d : ℝ), (∀ x, f x = horner_polynomial a b c d x) ∧ f 3 = 54 := by
  -- We use 2, 0, 1, -3 as the coefficients for Horner's method
  use 2, 0, 1, -3
  apply And.intro
  · -- Prove that the Horner form is equivalent to f for all x
    intro x
    simp [f, horner_polynomial]
    ring
  · -- Prove that f(3) = 54
    simp [f]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_for_f_l1209_120921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_angles_congruent_l1209_120959

/-- Two angles are vertical if they are opposite angles formed by two intersecting lines. -/
def VerticalAngles (α β : Real) : Prop := sorry

/-- Two angles are congruent if they have the same measure. -/
def Congruent (α β : Real) : Prop := α = β

/-- Theorem: Vertical angles are congruent. -/
theorem vertical_angles_congruent (α β : Real) (h : VerticalAngles α β) : Congruent α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_angles_congruent_l1209_120959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_four_l1209_120932

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.sin (x - 1) + x + 1

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 3

-- State the theorem
theorem max_min_sum_equals_four :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
               (∃ x ∈ I, f x = M) ∧
               (∀ x ∈ I, m ≤ f x) ∧ 
               (∃ x ∈ I, f x = m) ∧
               (M + m = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_four_l1209_120932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l1209_120951

-- Define the sets S and T
def S : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def T : Set ℝ := {z | ∃ x : ℝ, z = -2*x}

-- State the theorem
theorem intersection_S_T : S ∩ T = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l1209_120951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_less_than_zero_l1209_120992

open Set

theorem solution_set_of_f_less_than_zero
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (h1 : ∀ x, deriv f x + 2 * f x > 0)
  (h2 : f (-1) = 0) :
  {x : ℝ | f x < 0} = Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_less_than_zero_l1209_120992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_prime_minus_one_l1209_120917

/-- Recurrence sequence a_n -/
def a : ℕ → ℤ
  | 0 => 2
  | n + 1 => 2 * (a n)^2 - 1

theorem divisibility_of_prime_minus_one 
  (N : ℕ) 
  (hN : N ≥ 1) 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (hdiv : (p : ℤ) ∣ a N) 
  (x : ℤ) 
  (hx : x^2 ≡ 3 [ZMOD p]) :
  (2^(N+2) : ℕ) ∣ (p - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_prime_minus_one_l1209_120917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trash_to_initial_ratio_l1209_120953

def initial_emails : ℕ := 400
def work_folder_percentage : ℚ := 40 / 100
def remaining_emails : ℕ := 120

theorem trash_to_initial_ratio : 
  ∃ (trash_emails : ℕ), 
    (initial_emails - trash_emails - (work_folder_percentage * (initial_emails - trash_emails)).floor = remaining_emails) ∧
    (2 * trash_emails = initial_emails) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trash_to_initial_ratio_l1209_120953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubes_in_box_is_126_l1209_120944

/-- The number of cubes with 25 cm edges that can fit in a 1.5 m × 1.2 m × 1.1 m box -/
def cubes_in_box : ℕ :=
  let box_length : ℚ := 150  -- in cm
  let box_width : ℚ := 120   -- in cm
  let box_height : ℚ := 110  -- in cm
  let cube_edge : ℚ := 25    -- in cm
  let box_volume : ℚ := box_length * box_width * box_height
  let cube_volume : ℚ := cube_edge * cube_edge * cube_edge
  (box_volume / cube_volume).floor.toNat

/-- Proof that the number of cubes is 126 -/
theorem cubes_in_box_is_126 : cubes_in_box = 126 := by
  -- Unfold the definition of cubes_in_box
  unfold cubes_in_box
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

#eval cubes_in_box

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubes_in_box_is_126_l1209_120944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l1209_120919

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 6) :
  let circumference := 2 * Real.pi * r
  let sector_arc := circumference / 3
  let base_radius := sector_arc / (2 * Real.pi)
  let slant_height := r
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2) = 4 * Real.sqrt 2 :=
by
  -- Introduce the given values
  have h1 : r = 6 := h
  
  -- Define intermediate calculations
  let circumference := 2 * Real.pi * r
  let sector_arc := circumference / 3
  let base_radius := sector_arc / (2 * Real.pi)
  let slant_height := r
  
  -- The main proof steps would go here
  -- For now, we'll use sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l1209_120919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1209_120926

/-- Theorem: For an ellipse C with equation x²/2 + y² = 1, if a line y = x + m intersects C at points A and B such that OA · OB = -1 (where O is the origin), then m = ±√3/3. -/
theorem ellipse_intersection_theorem (m : ℝ) : 
  let C := {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.2 = p.1 + m}
  let intersection := C ∩ line
  (∃ A B : ℝ × ℝ, A ∈ intersection ∧ B ∈ intersection ∧ 
    A.1 * B.1 + A.2 * B.2 = -1) → m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1209_120926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l1209_120977

noncomputable def nabla (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

theorem nabla_calculation :
  ∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 →
  nabla (nabla x y) (nabla z w) = 49 / 56 :=
by
  intros x y z w hx hy hz hw
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l1209_120977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_coefficient_terms_count_l1209_120943

open Nat

/-- The number of terms with rational coefficients in the expansion of (x⁴√2 + y√5)^1200 -/
def rational_coefficient_terms : ℕ :=
  (List.range 1201).filter (fun k => 4 ∣ k ∧ 2 ∣ (1200 - k)) |>.length

/-- Theorem stating that the number of terms with rational coefficients is 301 -/
theorem rational_coefficient_terms_count : rational_coefficient_terms = 301 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_coefficient_terms_count_l1209_120943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_equality_l1209_120984

def isStrictlyIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem sequences_equality
  (a b : ℕ → ℝ)
  (ha : isStrictlyIncreasing a)
  (hb : isStrictlyIncreasing b)
  (hb0 : b 0 = 0)
  (ha_pos : ∀ n, 0 < a n)
  (h_perm : ∀ i j k r s t,
    a i + a j + a k = a r + a s + a t →
    ∃ σ : Equiv.Perm (Fin 3), (a i, a j, a k) = (a (σ 0), a (σ 1), a (σ 2)))
  (h_repr : ∀ x > 0,
    (∃ i j, x = a j - a i) ↔ (∃ m n, x = b m - b n)) :
  ∀ k, a k = b k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_equality_l1209_120984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_from_clicks_l1209_120972

/-- Represents the length of a rail in feet -/
noncomputable def rail_length : ℝ := 25

/-- Represents the time period in seconds -/
noncomputable def time_period : ℝ := 30

/-- Calculates the number of clicks in the given time period based on the train's speed -/
noncomputable def clicks_in_period (speed : ℝ) : ℝ :=
  (speed * 5280 / 60 / rail_length) * (time_period / 60)

/-- The theorem stating that 136 clicks in 30 seconds corresponds to a specific train speed -/
theorem train_speed_from_clicks :
  ∃ (speed : ℝ), clicks_in_period speed = 136 ∧ speed > 0 := by
  -- The proof goes here
  sorry

#check train_speed_from_clicks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_from_clicks_l1209_120972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l1209_120911

-- Define the function f(x) = (1/3)x^3 - ax^2 + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 1

-- Theorem statement
theorem unique_root_in_interval (a : ℝ) (h : a > 2) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ f a x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l1209_120911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1209_120987

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 --/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The equation of a circle with center (h, k) and radius r --/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The equation of a line ax + by + c = 0 --/
def line_equation (x y a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem circle_tangent_to_line :
  ∀ (x y : ℝ),
    circle_equation x y 2 (-1) 1 ↔
    (∃ (x₀ y₀ : ℝ),
      circle_equation x₀ y₀ 2 (-1) 1 ∧
      line_equation x₀ y₀ 3 4 (-7) ∧
      distance_point_to_line 2 (-1) 3 4 (-7) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1209_120987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_allocation_l1209_120915

theorem stratified_sampling_allocation (total_students : ℕ) 
  (class1_size class2_size sample_size : ℕ) 
  (h1 : class1_size = 54)
  (h2 : class2_size = 42)
  (h3 : sample_size = 16)
  (h4 : total_students = class1_size + class2_size) :
  (class1_size * sample_size) / total_students = 9 ∧
  sample_size - (class1_size * sample_size) / total_students = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_allocation_l1209_120915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ticket_exceed_percent_is_30_l1209_120966

/-- The percentage of motorists who receive speeding tickets -/
noncomputable def speeding_ticket_percent : ℝ := 10

/-- The percentage of motorists who exceed the speed limit -/
noncomputable def speed_limit_exceed_percent : ℝ := 14.285714285714285

/-- The percentage of motorists who exceed the speed limit but do not receive speeding tickets -/
noncomputable def no_ticket_exceed_percent : ℝ := 
  (speed_limit_exceed_percent - speeding_ticket_percent) / speed_limit_exceed_percent * 100

theorem no_ticket_exceed_percent_is_30 : 
  no_ticket_exceed_percent = 30 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ticket_exceed_percent_is_30_l1209_120966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l1209_120912

noncomputable section

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The line x - y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Distance from a point (x, y) to the line x - y - 1 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 1| / Real.sqrt 2

theorem min_distance_parabola_to_line :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 / 8 ∧
  ∀ (x : ℝ), 
    distance_to_line x (parabola x) ≥ d ∧
    (∃ (x₀ : ℝ), distance_to_line x₀ (parabola x₀) = d) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l1209_120912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_finished_first_l1209_120945

-- Define the participants
inductive Participant : Type
  | Allen : Participant
  | Bart : Participant
  | Clay : Participant
  | Dick : Participant

-- Define the finishing order relation
def finished_before : Participant → Participant → Prop := sorry

-- Define finishing positions
def finished_first (p : Participant) : Prop := ∀ x : Participant, x ≠ p → finished_before p x
def finished_second (p : Participant) : Prop := ∃ x : Participant, finished_before x p ∧ ∀ y : Participant, y ≠ x ∧ y ≠ p → finished_before p y
def finished_third (p : Participant) : Prop := ∃ x y : Participant, finished_before x y ∧ finished_before y p ∧ ∀ z : Participant, z ≠ x ∧ z ≠ y ∧ z ≠ p → finished_before p z
def finished_last (p : Participant) : Prop := ∀ x : Participant, x ≠ p → finished_before x p

-- Define the statements made by each participant
def statement (p : Participant) : Fin 2 → Prop
  | ⟨0, _⟩ => match p with
    | Participant.Allen => ∃ x, x = Participant.Bart ∧ finished_before p x
    | Participant.Bart => ∃ x, x = Participant.Clay ∧ finished_before p x
    | Participant.Clay => ∃ x, x = Participant.Dick ∧ finished_before p x
    | Participant.Dick => ∃ x, x = Participant.Allen ∧ finished_before p x
  | ⟨1, _⟩ => match p with
    | Participant.Allen => ¬ (finished_first p)
    | Participant.Bart => ¬ (finished_second p)
    | Participant.Clay => ¬ (finished_third p)
    | Participant.Dick => ¬ (finished_last p)

-- Define the main theorem
theorem clay_finished_first :
  (∃! (p1 p2 : Participant × Fin 2), statement p1.1 p1.2 ∧ statement p2.1 p2.2) →
  (∀ p : Participant, finished_first p → (statement p 0 ∨ statement p 1)) →
  finished_first Participant.Clay := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_finished_first_l1209_120945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_l1209_120962

noncomputable section

open Real

theorem triangle_angle_sum (A B C : ℝ) :
  tan A + tan B + tan C = tan A * tan B * tan C →
  A + B + C = π :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_l1209_120962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l1209_120939

-- Define the curve
noncomputable def x (t : ℝ) : ℝ := (1 + Real.log t) / t^2
noncomputable def y (t : ℝ) : ℝ := (3 + 2 * Real.log t) / t

-- Define the parameter value
def t₀ : ℝ := 1

-- State the theorem
theorem tangent_and_normal_lines :
  let x₀ : ℝ := x t₀
  let y₀ : ℝ := y t₀
  let slope : ℝ := (deriv y t₀) / (deriv x t₀)
  (∀ x' y' : ℝ, y' - y₀ = slope * (x' - x₀) ↔ y' = x' + 2) ∧
  (∀ x' y' : ℝ, y' - y₀ = (-1 / slope) * (x' - x₀) ↔ y' = -x' + 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l1209_120939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1209_120961

open Real

theorem triangle_angle_proof (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →  -- Angles are positive
  A + B + C = π →  -- Sum of angles in a triangle
  (sin (2*B) + sin (2*C)) / sin (2*A) = 1 →  -- Given condition
  B = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1209_120961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_approx_83_33_l1209_120997

/-- Represents a spherical triangle with angles in the ratio 3:4:5 and sum 200° -/
structure SphericalTriangle where
  /-- The base unit for the angle ratio -/
  x : ℝ
  /-- The sum of angles is 200° -/
  angle_sum : 3 * x + 4 * x + 5 * x = 200
  /-- The angles are positive -/
  x_pos : x > 0

/-- The largest angle in the spherical triangle -/
def largest_angle (t : SphericalTriangle) : ℝ := 5 * t.x

/-- Theorem: The largest angle in the specified spherical triangle is approximately 83.33° -/
theorem largest_angle_is_approx_83_33 (t : SphericalTriangle) : 
  ∃ ε > 0, abs (largest_angle t - 83.33) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_approx_83_33_l1209_120997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinting_competition_races_l1209_120913

theorem sprinting_competition_races (total_sprinters initial_lanes qualified_sprinters main_event_lanes : Nat)
  (h1 : total_sprinters = 300)
  (h2 : initial_lanes = 8)
  (h3 : qualified_sprinters = 192)
  (h4 : main_event_lanes = 6)
  (only_winner_advances : Bool) :
  (total_sprinters + initial_lanes - 1) / initial_lanes +
  (qualified_sprinters - 1 + (main_event_lanes - 1) - 1) / (main_event_lanes - 1) = 77 := by
  
  -- Define preliminary_races
  let preliminary_races := (total_sprinters + initial_lanes - 1) / initial_lanes
  
  -- Define eliminated_in_main_event
  let eliminated_in_main_event := qualified_sprinters - 1
  
  -- Define main_event_races
  let main_event_races := (eliminated_in_main_event + (main_event_lanes - 1) - 1) / (main_event_lanes - 1)
  
  -- Define total_races
  let total_races := preliminary_races + main_event_races
  
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinting_competition_races_l1209_120913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_equilateral_division_equilateral_no_unequal_division_l1209_120941

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- Define an equilateral triangle
def IsEquilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

-- Define similarity between triangles
def AreSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

-- Define a division of a triangle into smaller triangles
def IsDivision (t : Triangle) (ts : List Triangle) : Prop :=
  ∀ t', t' ∈ ts → AreSimilar t t' ∧ t'.a < t.a ∧ t'.b < t.b ∧ t'.c < t.c

-- Statement 1: Any non-equilateral triangle can be divided into unequal triangles similar to the original
theorem non_equilateral_division (t : Triangle) (h : ¬IsEquilateral t) :
  ∃ ts : List Triangle, IsDivision t ts ∧ ts.length > 1 ∧ ∀ t1 t2, t1 ∈ ts → t2 ∈ ts → t1 ≠ t2 → ¬(AreSimilar t1 t2) :=
sorry

-- Statement 2: An equilateral triangle cannot be divided into unequal equilateral triangles
theorem equilateral_no_unequal_division (t : Triangle) (h : IsEquilateral t) :
  ¬∃ ts : List Triangle, IsDivision t ts ∧ ts.length > 1 ∧ (∀ t', t' ∈ ts → IsEquilateral t') ∧ ∀ t1 t2, t1 ∈ ts → t2 ∈ ts → t1 ≠ t2 → ¬(AreSimilar t1 t2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_equilateral_division_equilateral_no_unequal_division_l1209_120941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1209_120960

-- Define the functions
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ p.2 ≤ max (f p.1) (g p.1)}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1209_120960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabellas_hair_length_l1209_120937

/-- Isabella's hair length in cubes -/
def current_length : ℕ := sorry

/-- The length of Isabella's hair after growing 4 inches -/
def future_length : ℕ := 22

/-- The growth in inches -/
def growth : ℕ := 4

theorem isabellas_hair_length : current_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabellas_hair_length_l1209_120937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l1209_120957

-- Define a regular octagon inscribed in a circle
structure RegularOctagon :=
  (side_length : ℝ)
  (radius : ℝ)
  (is_regular : side_length = radius * Real.sqrt (2 - Real.sqrt 2))

-- Define the area of a circular sector
noncomputable def sector_area (r : ℝ) (angle : ℝ) : ℝ := (angle / (2 * Real.pi)) * Real.pi * r^2

-- Define the area of an isosceles triangle
def isosceles_triangle_area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

-- Theorem statement
theorem octagon_reflected_arcs_area (o : RegularOctagon) (h : o.side_length = 2) :
  ∃ (bounded_area : ℝ),
    bounded_area = Real.pi * o.radius^2 - 
                   8 * sector_area o.radius (Real.pi / 4) + 
                   8 * isosceles_triangle_area o.side_length (Real.sqrt (o.radius^2 - (o.side_length/2)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l1209_120957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_average_price_l1209_120933

/-- Represents the properties of a type of rice -/
structure Rice where
  weight : ℝ
  price : ℝ

/-- Calculates the average price per kg of a mixture of rice -/
noncomputable def averagePrice (rices : List Rice) : ℝ :=
  let totalCost := (rices.map (λ r => r.weight * r.price)).sum
  let totalWeight := (rices.map (λ r => r.weight)).sum
  totalCost / totalWeight

/-- Theorem: The average price of the rice mixture is approximately 20.29 -/
theorem rice_mixture_average_price :
  let basmati := Rice.mk 10 20
  let longGrain := Rice.mk 5 25
  let shortGrain := Rice.mk 15 18
  let mediumGrain := Rice.mk 8 22
  let mixture := [basmati, longGrain, shortGrain, mediumGrain]
  abs (averagePrice mixture - 20.29) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_average_price_l1209_120933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1209_120918

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  Real.arccos ((distance p1 p2)^2 + (distance p2 p3)^2 - (distance p1 p3)^2) / (2 * distance p1 p2 * distance p2 p3)

/-- Theorem statement -/
theorem ellipse_triangle_area
  (e : Ellipse)
  (f1 f2 p : Point)
  (h1 : f1 = ⟨-2, 0⟩)
  (h2 : f2 = ⟨2, 0⟩)
  (h3 : isOnEllipse e ⟨2, 5/3⟩)
  (h4 : isOnEllipse e p)
  (h5 : angle f1 p f2 = π/3)
  : (1/2) * distance f1 p * distance p f2 * Real.sin (π/3) = 5 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1209_120918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_half_and_four_l1209_120970

noncomputable def P (k : ℕ) (a : ℝ) : ℝ := a / (k * (k + 1))

noncomputable def sum_probabilities (a : ℝ) : ℝ :=
  (Finset.range 6).sum (λ k => P (k + 1) a)

theorem probability_between_half_and_four (a : ℝ) :
  sum_probabilities a = 1 →
  (Finset.range 3).sum (λ k => P (k + 1) a) = 7/8 := by
  sorry

#check probability_between_half_and_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_half_and_four_l1209_120970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1209_120909

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- A line with slope k and y-intercept m -/
structure Line where
  k : ℝ
  m : ℝ

/-- Theorem: For an ellipse with focal length 2√3 passing through (1, √3/2),
    intersected by a line at two points satisfying y₁y₂ = k²x₁x₂,
    the slope of the line is either 1/2 or -1/2 -/
theorem ellipse_line_intersection
  (C : Ellipse)
  (L : Line)
  (h_focal : C.a ^ 2 - C.b ^ 2 = 3)
  (h_point : C.a ^ 2 * (3 / 4) + C.b ^ 2 = 1)
  (h_distinct : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = L.k * x₁ + L.m ∧
    y₂ = L.k * x₂ + L.m ∧
    x₁ ^ 2 / C.a ^ 2 + y₁ ^ 2 / C.b ^ 2 = 1 ∧
    x₂ ^ 2 / C.a ^ 2 + y₂ ^ 2 / C.b ^ 2 = 1)
  (h_product : ∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁ = L.k * x₁ + L.m →
    y₂ = L.k * x₂ + L.m →
    x₁ ^ 2 / C.a ^ 2 + y₁ ^ 2 / C.b ^ 2 = 1 →
    x₂ ^ 2 / C.a ^ 2 + y₂ ^ 2 / C.b ^ 2 = 1 →
    y₁ * y₂ = L.k ^ 2 * x₁ * x₂) :
  L.k = 1/2 ∨ L.k = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1209_120909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1209_120920

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ :=
  (12, Real.pi / 6, 2 * Real.sqrt 3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ :=
  (6 * Real.sqrt 3, 6, 2 * Real.sqrt 3)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1209_120920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2beta_plus_pi_over_6_l1209_120991

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (3*Real.pi/2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem sin_2beta_plus_pi_over_6 (β : Real) :
  (∀ k : ℤ, Real.pi/2 + β ≠ k * Real.pi/2) →
  f (Real.pi/2 + β) = -Real.sqrt 3 / 3 →
  0 < Real.cos β →
  Real.sin β > 0 →
  Real.cos β > 0 →
  Real.sin (2*β + Real.pi/6) = (1 + 2*Real.sqrt 6) / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2beta_plus_pi_over_6_l1209_120991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_movement_notation_moving_right_3m_l1209_120964

/-- Represents the direction of movement --/
inductive Direction
| Left
| Right

/-- Represents a movement with distance and direction --/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Defines the notation for a movement --/
def moveNotation (m : Movement) : ℝ :=
  match m.direction with
  | Direction.Left => m.distance
  | Direction.Right => -m.distance

/-- Theorem stating the correct notation for moving right --/
theorem right_movement_notation (d : ℝ) :
  moveNotation { distance := d, direction := Direction.Right } = -d :=
by
  simp [moveNotation]

/-- Given condition: moving 2m to the left is denoted as +2m --/
axiom left_movement_notation : moveNotation { distance := 2, direction := Direction.Left } = 2

/-- Theorem to prove: moving 3m to the right is denoted as -3m --/
theorem moving_right_3m :
  moveNotation { distance := 3, direction := Direction.Right } = -3 :=
by
  simp [moveNotation]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_movement_notation_moving_right_3m_l1209_120964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1209_120947

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  Real.cos C = 1/5 →
  a * b * Real.cos C = 1 →
  a + b = Real.sqrt 37 →
  Real.sin (C + π/4) = (4 * Real.sqrt 3 + Real.sqrt 2) / 10 ∧
  c = 5 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1209_120947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_eight_relation_l1209_120922

theorem power_eight_relation (x : ℝ) : (8 : ℝ)^(3*x) = 512 → (8 : ℝ)^(3*x - 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_eight_relation_l1209_120922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_EFGH_l1209_120975

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
noncomputable def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let d₁ := Real.sqrt ((q.E.1 - q.F.1)^2 + (q.E.2 - q.F.2)^2)
  let d₂ := Real.sqrt ((q.F.1 - q.G.1)^2 + (q.F.2 - q.G.2)^2)
  let d₃ := Real.sqrt ((q.G.1 - q.H.1)^2 + (q.G.2 - q.H.2)^2)
  let d₄ := Real.sqrt ((q.H.1 - q.E.1)^2 + (q.H.2 - q.E.2)^2)
  d₁ = 5 ∧ d₂ = 6 ∧ d₃ = 8 ∧ d₄ = 10

-- Define the right angle at G
def has_right_angle_at_G (q : Quadrilateral) : Prop :=
  let vec_FG := (q.G.1 - q.F.1, q.G.2 - q.F.2)
  let vec_HG := (q.G.1 - q.H.1, q.G.2 - q.H.2)
  vec_FG.1 * vec_HG.1 + vec_FG.2 * vec_HG.2 = 0

-- Define the area of the quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ :=
  let area_FGH := 0.5 * 6 * 8
  let s := (5 + 10 + 10) / 2
  let area_EFH := Real.sqrt (s * (s - 5) * (s - 10) * (s - 10))
  area_FGH + area_EFH

-- Theorem statement
theorem area_of_quadrilateral_EFGH (q : Quadrilateral) 
  (h₁ : is_valid_quadrilateral q) (h₂ : has_right_angle_at_G q) : 
  area q = 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_EFGH_l1209_120975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_share_is_704_l1209_120996

/-- Represents the investment and profit distribution of a partnership --/
structure Partnership where
  invest_B : ℚ
  invest_A : ℚ := 3 * invest_B
  invest_C : ℚ := (3/2) * invest_B
  invest_D : ℚ := (1/2) * invest_B
  time_A : ℕ := 9
  time_B : ℕ := 6
  time_C : ℕ := 4
  time_D : ℕ := 12
  total_profit : ℚ := 5280

/-- Calculates B's share of the profit in the partnership --/
noncomputable def calculate_B_share (p : Partnership) : ℚ :=
  let total_capital_months := p.invest_A * p.time_A + p.invest_B * p.time_B + 
                              p.invest_C * p.time_C + p.invest_D * p.time_D
  let B_capital_months := p.invest_B * p.time_B
  (B_capital_months / total_capital_months) * p.total_profit

/-- Theorem stating that B's share of the profit is 704 --/
theorem B_share_is_704 (p : Partnership) : calculate_B_share p = 704 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_share_is_704_l1209_120996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_price_change_l1209_120908

theorem painting_price_change (P : ℝ) (hP : P > 0) : 
  let first_year_price := 1.30 * P
  let second_year_price := 1.04 * P
  ∃ (decrease_rate : ℝ), 
    first_year_price * (1 - decrease_rate) = second_year_price ∧ 
    decrease_rate = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_price_change_l1209_120908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1209_120955

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then -(x-a)^2 else -x^2-2*x-3+a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a 0) → -2 ≤ a ∧ a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1209_120955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l1209_120995

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ  -- Semi-major axis
  c : ℝ  -- Focal distance
  h_pos_a : 0 < a
  h_pos_c : 0 < c
  h_c_gt_a : c > a

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- Represents a point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : (x^2 / h.a^2) - (y^2 / (h.c^2 - h.a^2)) = 1

/-- Predicate for a point being on the right branch of a hyperbola -/
def is_on_right_branch (h : Hyperbola) (p : PointOnHyperbola h) : Prop :=
  p.x > 0

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Predicate for an isosceles triangle being acute-angled -/
def is_acute_isosceles (a b c : ℝ) : Prop :=
  a = b ∧ c^2 < 2 * a^2

theorem hyperbola_eccentricity_bound (h : Hyperbola) :
  (∀ (p : PointOnHyperbola h), is_on_right_branch h p →
    is_acute_isosceles (distance p.x p.y (-h.c) 0) (distance p.x p.y h.c 0) (2 * h.c)) →
  eccentricity h > Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l1209_120995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_width_for_load_l1209_120938

/-- Represents the relationship between beam width and maximum load -/
structure BeamLoad where
  width : ℝ
  load : ℝ

/-- The constant of proportionality between beam width and maximum load -/
noncomputable def proportionalityConstant (b : BeamLoad) : ℝ := b.load / b.width

theorem beam_width_for_load 
  (b1 b2 : BeamLoad) 
  (h1 : b1.width = 1.5) 
  (h2 : b1.load = 250) 
  (h3 : b2.load = 583.3333) 
  (h4 : proportionalityConstant b1 = proportionalityConstant b2) : 
  b2.width = 3.5 := by
  sorry

#check beam_width_for_load

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_width_for_load_l1209_120938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_time_l1209_120900

theorem road_trip_time (total_distance heavy_traffic_speed adverse_weather_speed 
  heavy_traffic_time adverse_weather_time : ℝ)
  (h1 : total_distance = 190)
  (h2 : heavy_traffic_speed = 40)
  (h3 : adverse_weather_speed = 30)
  (h4 : adverse_weather_time = heavy_traffic_time + 2)
  (h5 : total_distance = heavy_traffic_speed * heavy_traffic_time + adverse_weather_speed * adverse_weather_time)
  : heavy_traffic_time = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_time_l1209_120900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_upper_bound_l1209_120948

/-- A set of points in the plane with specific properties -/
structure PointSet (n k : ℕ) where
  S : Set (ℝ × ℝ)
  card_S : Nat.card S = n
  non_collinear : ∀ (p q r : ℝ × ℝ), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r →
    ¬(∃ (a b c : ℝ), a * p.1 + b * p.2 + c = 0 ∧ a * q.1 + b * q.2 + c = 0 ∧ a * r.1 + b * r.2 + c = 0)
  equidistant_points : ∀ (p : ℝ × ℝ), p ∈ S →
    ∃ (r : ℝ), r > 0 ∧ Nat.card {q : ℝ × ℝ | q ∈ S ∧ Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2) = r} ≥ k

/-- The main theorem stating the upper bound for k -/
theorem k_upper_bound (n k : ℕ) (h : PointSet n k) : k < 1/2 + Real.sqrt (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_upper_bound_l1209_120948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_l1209_120976

theorem cube_edge_length (volume : ℝ) (edge : ℝ) (h1 : volume = 4) (h2 : volume = edge ^ 3) :
  edge = (4 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_l1209_120976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1209_120935

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1209_120935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hit_ground_time_ball_hit_ground_proof_l1209_120954

/-- The time it takes for a ball to hit the ground when thrown upward -/
theorem ball_hit_ground_time (initial_velocity : ℝ) (initial_height : ℝ) 
  (gravity : ℝ) (air_resistance : ℝ) :
  let height := λ t : ℝ ↦ -gravity * t^2 + (initial_velocity - air_resistance * t) * t + initial_height
  ∃ t : ℝ, t = 223 / 110 ∧ height t = 0 :=
by
  sorry

/-- Proof that the ball hits the ground after 223/110 seconds -/
theorem ball_hit_ground_proof :
  ∃ t : ℝ, t = 223 / 110 ∧ 
    (-5.5 * t^2 + 7.2 * t + 8 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hit_ground_time_ball_hit_ground_proof_l1209_120954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_eight_fifths_l1209_120925

noncomputable def a (α : Real) : Real × Real := (Real.sin (2 * α), Real.cos α)
noncomputable def b (α : Real) : Real × Real := (1, Real.cos α)

theorem dot_product_equals_eight_fifths (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (a α).1 * (b α).1 + (a α).2 * (b α).2 = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_eight_fifths_l1209_120925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_female_selection_count_l1209_120969

/-- Number of male students in Group A -/
def male_a : ℕ := 5

/-- Number of female students in Group A -/
def female_a : ℕ := 3

/-- Number of male students in Group B -/
def male_b : ℕ := 6

/-- Number of female students in Group B -/
def female_b : ℕ := 2

/-- Number of students to be selected from each group -/
def selected_per_group : ℕ := 2

/-- The total number of ways to select exactly 1 female student among 4 chosen students -/
theorem one_female_selection_count : 
  (Nat.choose female_a 1 * Nat.choose male_a 1 * Nat.choose male_b 2) +
  (Nat.choose female_b 1 * Nat.choose male_a 2 * Nat.choose male_b 1) = 345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_female_selection_count_l1209_120969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l1209_120936

open Real

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Define the shift amount
noncomputable def shift : ℝ := π / 4

-- Define the resulting function after the shift
noncomputable def resulting_function (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

-- Theorem statement
theorem shift_sine_graph :
  ∀ x : ℝ, original_function (x - shift) = resulting_function x :=
by
  intro x
  simp [original_function, resulting_function, shift]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l1209_120936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_abs_tan_l1209_120973

/-- The axis of symmetry for the function y = |tan x| is x = (π/2)k, where k ∈ ℤ -/
theorem axis_of_symmetry_abs_tan (x : ℝ) :
  ∃ k : ℤ, x = (π / 2) * k ↔ ∀ y, |Real.tan (x - y)| = |Real.tan (x + y)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_abs_tan_l1209_120973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_of_springfield_interest_l1209_120979

/-- Calculates the interest earned on a savings account with annual compounding -/
noncomputable def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem bank_of_springfield_interest :
  round_to_nearest (interest_earned 1000 0.01 5) = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_of_springfield_interest_l1209_120979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l1209_120916

theorem quadratic_root_difference (m n : ℕ) : 
  (∃ (r₁ r₂ : ℝ), 2 * r₁^2 - 5 * r₁ - 3 = 0 ∧ 
                   2 * r₂^2 - 5 * r₂ - 3 = 0 ∧ 
                   |r₁ - r₂| = Real.sqrt (m : ℝ) / (n : ℝ)) →
  (n > 0) →
  (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ m)) →
  m + n = 51 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l1209_120916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_distinct_terms_l1209_120950

/-- The type of natural numbers starting from 1 -/
def PositiveNat := {n : ℕ // n > 0}

/-- Sum of p-th powers of decimal digits of n -/
def f (p : PositiveNat) (n : ℕ) : ℕ :=
  sorry

/-- Sequence defined by b_{k+1} = f(b_k) -/
def sequenceF (p : PositiveNat) (b₀ : ℕ) : ℕ → ℕ
  | 0 => b₀
  | k + 1 => f p (sequenceF p b₀ k)

theorem finite_distinct_terms (p : PositiveNat) (b₀ : ℕ) :
  ∃ (S : Finset ℕ), ∀ k, (sequenceF p b₀ k) ∈ S :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_distinct_terms_l1209_120950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_webinar_active_minutes_l1209_120927

theorem webinar_active_minutes : ℕ := by
  let total_hours : ℕ := 13
  let total_extra_minutes : ℕ := 17
  let break_minutes : ℕ := 22
  let minutes_per_hour : ℕ := 60
  
  let total_minutes : ℕ := total_hours * minutes_per_hour + total_extra_minutes
  let active_minutes : ℕ := total_minutes - break_minutes
  
  have h : active_minutes = 775 := by sorry
  exact active_minutes


end NUMINAMATH_CALUDE_ERRORFEEDBACK_webinar_active_minutes_l1209_120927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_y_is_negative_five_l1209_120994

theorem x_value_when_y_is_negative_five (x y : ℝ) :
  16 * (3 : ℝ)^x = (7 : ℝ)^(y + 5) →
  y = -5 →
  x = -4 * Real.log 2 / Real.log 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_y_is_negative_five_l1209_120994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_company_performance_l1209_120978

/-- Represents the data for a bus company --/
structure BusCompanyData where
  onTime : Nat
  notOnTime : Nat

/-- Calculates the probability of on-time buses for a company --/
def onTimeProbability (data : BusCompanyData) : Rat :=
  data.onTime / (data.onTime + data.notOnTime)

/-- Calculates the K² statistic --/
def calculateKSquared (a b c d : Nat) : Rat :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the probabilities and relationship between on-time performance and bus company --/
theorem bus_company_performance 
  (companyA : BusCompanyData)
  (companyB : BusCompanyData)
  (h1 : companyA = ⟨240, 20⟩)
  (h2 : companyB = ⟨210, 30⟩) :
  (onTimeProbability companyA = 12 / 13) ∧ 
  (onTimeProbability companyB = 7 / 8) ∧
  (calculateKSquared 240 20 210 30 > 2706 / 1000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_company_performance_l1209_120978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangents_iff_range_l1209_120958

/-- The curve function y = (x + 1)e^x -/
noncomputable def curve (x : ℝ) : ℝ := (x + 1) * Real.exp x

/-- Condition for exactly two tangent lines -/
def has_two_tangents (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ →
    (curve x - 0) / (x - a) ≠ (curve x₁ - 0) / (x₁ - a))

/-- Theorem: The curve y = (x + 1)e^x has exactly two tangent lines
    passing through P(a, 0) iff a ∈ (-∞, -5) ∪ (-1, +∞) -/
theorem two_tangents_iff_range (a : ℝ) :
  has_two_tangents a ↔ a < -5 ∨ a > -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangents_iff_range_l1209_120958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_924_l1209_120949

theorem sum_distinct_prime_factors_924 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.factors 924).toFinset) id) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_924_l1209_120949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_properties_l1209_120968

noncomputable section

def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

structure Point where
  x : ℝ
  y : ℝ

def is_on_ellipse (P : Point) : Prop := ellipse P.x P.y

noncomputable def distance (P Q : Point) : ℝ := Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

noncomputable def cos_angle (A B C : Point) : ℝ :=
  let a := distance B C
  let b := distance A C
  let c := distance A B
  (b^2 + c^2 - a^2) / (2 * b * c)

noncomputable def perimeter (A B C : Point) : ℝ := distance A B + distance B C + distance C A

noncomputable def area (A B C : Point) : ℝ :=
  let a := distance B C
  let b := distance A C
  let c := distance A B
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem ellipse_triangle_properties (P F₁ F₂ : Point) 
  (h_on_ellipse : is_on_ellipse P)
  (h_foci : F₁.x < F₂.x ∧ F₁.y = 0 ∧ F₂.y = 0)
  (h_cos : cos_angle F₁ P F₂ = 1/2) :
  perimeter F₁ P F₂ = 16 ∧
  area F₁ P F₂ = 16 * Real.sqrt 3 / 3 ∧
  |P.y| = 16 * Real.sqrt 3 / 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_properties_l1209_120968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_penny_excess_purchase_l1209_120980

/-- Represents the honey purchase scenario -/
structure HoneyPurchase where
  bulk_price : ℚ  -- Bulk price of honey per pound
  min_spend : ℚ   -- Minimum spend before tax
  tax_rate : ℚ    -- Tax per pound
  total_paid : ℚ  -- Total amount Penny paid

/-- Calculates the excess pounds of honey purchased above the minimum spend -/
def excess_pounds (purchase : HoneyPurchase) : ℚ :=
  let min_pounds := purchase.min_spend / purchase.bulk_price
  let total_pounds := purchase.total_paid / (purchase.bulk_price + purchase.tax_rate)
  total_pounds - min_pounds

/-- Theorem stating that Penny's purchase exceeded the minimum spend by 32 pounds -/
theorem penny_excess_purchase (penny_purchase : HoneyPurchase) 
  (h1 : penny_purchase.bulk_price = 5)
  (h2 : penny_purchase.min_spend = 40)
  (h3 : penny_purchase.tax_rate = 1)
  (h4 : penny_purchase.total_paid = 240) :
  excess_pounds penny_purchase = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_penny_excess_purchase_l1209_120980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_ton_equals_2600_pounds_l1209_120986

-- Define the constants from the problem
def ounces_per_pound : ℕ := 16
def num_packets : ℕ := 2080
def packet_weight_pounds : ℕ := 16
def packet_weight_ounces : ℕ := 4
def bag_capacity_tons : ℕ := 13

-- Define the theorem
theorem one_ton_equals_2600_pounds :
  ∃ (pounds_per_ton : ℕ),
    pounds_per_ton = 2600 ∧
    (num_packets : ℚ) * ((packet_weight_pounds : ℚ) + (packet_weight_ounces : ℚ) / (ounces_per_pound : ℚ)) = (bag_capacity_tons : ℚ) * (pounds_per_ton : ℚ) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_ton_equals_2600_pounds_l1209_120986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_f_l1209_120906

/-- The function f(x) = ae^x - ln x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

/-- The derivative of f(x) with respect to x --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 1 / x

/-- The theorem stating the minimum value of a for which f is monotonically increasing on (1, 2) --/
theorem min_a_for_monotone_f (a : ℝ) :
  (∀ x y, x ∈ Set.Ioo 1 2 → y ∈ Set.Ioo 1 2 → x < y → f a x < f a y) ↔ a ≥ Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_f_l1209_120906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_ℓ_range_of_ℓ_l1209_120946

-- Define the function ℓ as noncomputable
noncomputable def ℓ (y : ℝ) : ℝ := 1 / ((y - 2) + (y - 8))

-- State the theorem about the domain of ℓ
theorem domain_of_ℓ :
  {y : ℝ | y ≠ 5} = {y : ℝ | ∃ z : ℝ, z = ℓ y} :=
by sorry

-- Alternative formulation using Set.range
theorem range_of_ℓ :
  Set.range ℓ = {z : ℝ | ∃ y : ℝ, y ≠ 5 ∧ z = ℓ y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_ℓ_range_of_ℓ_l1209_120946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1209_120956

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = -2 is √2 -/
theorem hyperbola_eccentricity : 
  ∃ (e : ℝ), e = Real.sqrt 2 ∧ 
    ∀ (a b c : ℝ), (∀ (x y : ℝ), x^2 - y^2 = -2 ↔ y^2 / a^2 - x^2 / b^2 = 1) →
    c^2 = a^2 + b^2 → e = c / a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1209_120956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_meeting_sum_l1209_120990

/-- The duration of the time window in minutes -/
def time_window : ℕ := 120

/-- The probability of the two friends meeting -/
def prob_meeting : ℚ := 1/2

/-- The function representing the duration of play -/
noncomputable def m (a b c : ℕ) : ℝ := a - b * Real.sqrt (c : ℝ)

/-- The theorem stating the sum of a, b, and c -/
theorem friend_meeting_sum (a b c : ℕ) :
  (∀ p : ℕ, Nat.Prime p → ¬ p^2 ∣ c) →
  (prob_meeting = 1 - ((time_window : ℝ) - m a b c)^2 / (time_window^2 : ℝ)) →
  a + b + c = 206 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_meeting_sum_l1209_120990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l1209_120981

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola C (trajectory of the center of the moving circle)
def C (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the right focus F2
def F2 : ℝ × ℝ := (1, 0)

-- Define the condition for collinearity
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the area of a quadrilateral given its four vertices
noncomputable def quadrilateral_area (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let a := p2.1 - p1.1
  let b := p2.2 - p1.2
  let c := p4.1 - p3.1
  let d := p4.2 - p3.2
  abs (a * d - b * c) / 2

-- State the theorem
theorem min_quadrilateral_area :
  ∃ (M N : ℝ × ℝ) (P Q : ℝ × ℝ),
    C M.1 M.2 ∧ C N.1 N.2 ∧ C1 P.1 P.2 ∧ C1 Q.1 Q.2 ∧
    collinear M F2 N ∧ collinear P F2 Q ∧
    dot_product (P.1 - F2.1, P.2 - F2.2) (M.1 - F2.1, M.2 - F2.2) = 0 ∧
    (∀ (M' N' : ℝ × ℝ) (P' Q' : ℝ × ℝ),
      C M'.1 M'.2 → C N'.1 N'.2 → C1 P'.1 P'.2 → C1 Q'.1 Q'.2 →
      collinear M' F2 N' → collinear P' F2 Q' →
      dot_product (P'.1 - F2.1, P'.2 - F2.2) (M'.1 - F2.1, M'.2 - F2.2) = 0 →
      quadrilateral_area P M Q N ≤ quadrilateral_area P' M' Q' N') ∧
    quadrilateral_area P M Q N = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l1209_120981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l1209_120989

def set_A : Set ℝ := {x : ℝ | |x - 1| > 2}
def set_B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a + 1) * x + a < 0}

theorem intersection_equals_open_interval :
  ∃! a : ℝ, set_A ∩ set_B a = Set.Ioo 3 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l1209_120989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_release_angle_l1209_120985

noncomputable def g : ℝ := 32  -- Acceleration due to gravity in feet/s²

-- Define the turntable properties
noncomputable def radius : ℝ := 1  -- in feet
noncomputable def period : ℝ := 0.5  -- in seconds

-- Define the velocity of the ball at release
noncomputable def velocity : ℝ := 2 * Real.pi * radius / period

-- Define the minimum release angle
noncomputable def θ_m : ℝ := Real.arcsin (g / (16 * Real.pi^2))

-- Theorem statement
theorem minimum_release_angle :
  ∀ θ : ℝ, θ > θ_m → 
    (2 * radius * Real.cos θ < (velocity^2 * Real.sin (2*θ)) / g) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_release_angle_l1209_120985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_iff_c_zero_or_a_eq_b_l1209_120914

/-- Represents a line in 2D space with equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  ab_nonzero : a * b ≠ 0

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.b

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.a

/-- Condition for equal intercepts -/
def equal_intercepts (l : Line) : Prop :=
  x_intercept l = y_intercept l

theorem equal_intercepts_iff_c_zero_or_a_eq_b (l : Line) :
  equal_intercepts l ↔ l.c = 0 ∨ l.a = l.b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_iff_c_zero_or_a_eq_b_l1209_120914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l1209_120904

theorem two_integers_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧
  x * y + x + y = 159 ∧
  Nat.gcd x y = 1 ∧
  x < 30 ∧ y < 30 →
  x + y = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l1209_120904
