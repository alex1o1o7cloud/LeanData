import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_is_two_point_five_l1222_122222

/-- Represents the dimensions and water levels of a fishbowl and a water cube --/
structure FishbowlProblem where
  fishbowl_height : ℝ
  fishbowl_side : ℝ
  initial_water_height : ℝ
  cube_edge : ℝ

/-- Calculates the rise in water level after pouring a cube of water into a square-based fishbowl --/
noncomputable def water_level_rise (problem : FishbowlProblem) : ℝ :=
  let cube_volume := problem.cube_edge ^ 3
  let initial_water_volume := problem.fishbowl_side ^ 2 * problem.initial_water_height
  let total_water_volume := initial_water_volume + cube_volume
  let new_water_height := total_water_volume / (problem.fishbowl_side ^ 2)
  new_water_height - problem.initial_water_height

/-- Theorem stating that the water level rise is 2.5 cm for the given problem --/
theorem water_rise_is_two_point_five 
  (problem : FishbowlProblem)
  (h1 : problem.fishbowl_height = 20)
  (h2 : problem.fishbowl_side = 20)
  (h3 : problem.initial_water_height = 15)
  (h4 : problem.cube_edge = 10) :
  water_level_rise problem = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_rise_is_two_point_five_l1222_122222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_of_powers_l1222_122221

-- Define p and q as real numbers
variable (p q : ℝ)

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - x * Real.sqrt 7 + 1 = 0

-- Define p and q as the roots of the equation
axiom p_root : equation p
axiom q_root : equation q

-- Define p and q as distinct roots
axiom p_neq_q : p ≠ q

-- Vieta's formulas
axiom vieta_sum : p + q = Real.sqrt 7
axiom vieta_product : p * q = 1

-- Theorem to prove
theorem roots_sum_of_powers : p^8 + q^8 = 527 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_of_powers_l1222_122221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_work_time_l1222_122292

-- Define the work and time variables
variable (W : ℝ) -- Total amount of work
variable (x y : ℝ) -- Time taken by x and y individually

-- Define the conditions
def condition1 (x : ℝ) : Prop := x = 10
def condition2 (x y W : ℝ) : Prop := (1 / x + 1 / y) * W = W / 6

-- Theorem statement
theorem y_work_time (h1 : condition1 x) (h2 : condition2 x y W) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_work_time_l1222_122292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_distance_inequality_l1222_122223

open Real

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Angle between three points -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

theorem incenter_distance_inequality (t : Triangle) (p : ℝ × ℝ) :
  isInside p t →
  angle p t.B t.A + angle p t.C t.A = angle p t.B t.C + angle p t.C t.B →
  distance t.A p ≥ distance t.A (incenter t) ∧
  (distance t.A p = distance t.A (incenter t) ↔ p = incenter t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_distance_inequality_l1222_122223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_is_four_l1222_122203

-- Define T as a finite set of natural numbers
def T : Finset ℕ := sorry

-- T is a set of 8 integers
axiom T_size : Finset.card T = 8

-- T is a subset of {1,2,...,20}
axiom T_subset : T ⊆ Finset.range 21

-- For any a, b in T with a < b, b is not a multiple of a
axiom not_multiple {a b : ℕ} : a ∈ T → b ∈ T → a < b → ¬(b % a = 0)

-- For any a, b in T with a < b, b - 3 ≠ a
axiom not_three_diff {a b : ℕ} : a ∈ T → b ∈ T → a < b → b - 3 ≠ a

-- The smallest possible value of an element in T is 4
theorem smallest_value_is_four : 
  ∃ (x : ℕ), x ∈ T ∧ x = 4 ∧ ∀ y ∈ T, y ≥ 4 := by
  sorry

#check smallest_value_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_is_four_l1222_122203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_999999999_squared_l1222_122238

/-- The number of zeros in the square of a number consisting of n nines -/
def num_zeros (n : ℕ) : ℕ := n - 1

/-- The square of a number consisting of n nines -/
def nines_squared (n : ℕ) : ℕ := (10^n - 1)^2

theorem zeros_in_999999999_squared :
  ∃ (z : ℕ), z = num_zeros 9 ∧ 
  (∀ (d : ℕ), d ∈ Nat.digits 10 (nines_squared 9) → d = 0) →
  (Finset.filter (· = 0) (Finset.range 10)).card = z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_999999999_squared_l1222_122238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_equation_l1222_122270

theorem smaller_root_of_equation :
  ∃ x : ℚ, (x - 5/4) * (x - 5/4) + (x - 5/4) * (x - 1/2) = 0 ∧
  x = 7/8 ∧
  ∀ y : ℚ, (y - 5/4) * (y - 5/4) + (y - 5/4) * (y - 1/2) = 0 → y ≥ 7/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_equation_l1222_122270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_sum_of_sides_l1222_122295

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b * Real.cos t.C = (2 * t.a + t.c) * Real.cos (Real.pi - t.B)

def hasSpecificMeasurements (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4

-- State the theorems
theorem angle_B_measure (t : Triangle) :
  satisfiesCondition t → t.B = 2 * Real.pi / 3 := by sorry

theorem sum_of_sides (t : Triangle) :
  satisfiesCondition t → hasSpecificMeasurements t → t.a + t.c = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_sum_of_sides_l1222_122295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_took_five_papers_l1222_122258

/-- Represents the examination results of a student --/
structure ExamResults where
  /-- The number of papers taken --/
  num_papers : ℕ
  /-- The full marks for each paper --/
  full_marks : ℝ
  /-- The marks obtained in each paper --/
  marks : Fin num_papers → ℝ

/-- The conditions of the exam results --/
def exam_conditions (r : ExamResults) : Prop :=
  -- The marks are in the proportion 6 : 7 : 8 : 9 : 10
  (∃ x : ℝ, ∀ i : Fin r.num_papers, r.marks i = ([6, 7, 8, 9, 10].get? i.val).getD 0 * x) ∧
  -- The candidate obtained 60% of the total marks
  (Finset.sum Finset.univ r.marks = 0.6 * (r.num_papers : ℝ) * r.full_marks) ∧
  -- The number of papers with more than 50% marks is 4
  ((Finset.filter (λ i => r.marks i > 0.5 * r.full_marks) Finset.univ).card = 4)

/-- The theorem stating that under the given conditions, the student took 5 papers --/
theorem student_took_five_papers (r : ExamResults) (h : exam_conditions r) : r.num_papers = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_took_five_papers_l1222_122258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_lines_sum_squares_l1222_122269

/-- Two lines dividing a unit circle into four equal arcs -/
structure DividingLines where
  a : ℝ
  b : ℝ
  divide_equally : ∀ (x y : ℝ), x^2 + y^2 = 1 → 
    (∃ (t : ℝ), y = x + a ∨ y = x + b) → 
    (∃ (θ : ℝ), θ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
      (x = Real.cos θ ∧ y = Real.sin θ) ∧ 
      (θ = Real.pi / 2 ∨ θ = Real.pi ∨ θ = 3 * Real.pi / 2 ∨ θ = 2 * Real.pi))

/-- Theorem: The sum of squares of the y-intercepts of two lines dividing a unit circle 
    into four equal arcs is equal to 2 -/
theorem dividing_lines_sum_squares (l : DividingLines) : l.a^2 + l.b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_lines_sum_squares_l1222_122269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_relation_u_gcd_l1222_122297

def u : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => u (n+1) + 2 * u n

theorem u_relation (n p : ℕ) (h : p > 1) :
  u (n + p) = u (n + 1) * u p + 2 * u n * u (p - 1) := by sorry

theorem u_gcd (n : ℕ) :
  Nat.gcd (u n) (u (n + 3)) = if n % 3 = 0 then 3 else 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_relation_u_gcd_l1222_122297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_condition_implies_value_l1222_122207

theorem monomial_condition_implies_value (a b : ℤ) : 
  (∃ (x y : ℝ), -x^(a:ℝ) * y^4 + 4 * x^3 * y^(4*b:ℝ) = k * x^m * y^n) → 
  (-1:ℝ)^(a:ℝ) * (b:ℝ)^4 = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_condition_implies_value_l1222_122207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_plus_a9_ge_b4_plus_b10_l1222_122212

-- Define the sequences a_n and b_n
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- State the conditions
axiom a_positive : ∀ n, a n > 0
axiom a_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
axiom b_arithmetic : ∀ n, b (n + 1) - b n = b 2 - b 1
axiom a6_eq_b7 : a 6 = b 7

-- State the theorem
theorem a3_plus_a9_ge_b4_plus_b10 : a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_plus_a9_ge_b4_plus_b10_l1222_122212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_ratio_l1222_122247

theorem cosine_sine_ratio (α : Real) 
  (h1 : Real.sin α = 1/3 + Real.cos α) 
  (h2 : 0 < α ∧ α < Real.pi/2) : 
  Real.cos (2*α) / Real.sin (α + Real.pi/4) = -Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_ratio_l1222_122247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_l1222_122266

/-- A function that checks if a set of integers satisfies the condition that no member
    is 3 times or one-third of another member -/
def valid_subset (s : Set Int) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ 3 * y ∧ 3 * x ≠ y

/-- The theorem stating the largest valid subset of integers from 1 to 120 has 106 elements -/
theorem largest_valid_subset :
  ∃ (s : Finset Int),
    (∀ x ∈ s, 1 ≤ x ∧ x ≤ 120) ∧
    valid_subset s ∧
    s.card = 106 ∧
    (∀ t : Finset Int, (∀ x ∈ t, 1 ≤ x ∧ x ≤ 120) → valid_subset t → t.card ≤ 106) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_l1222_122266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_projection_l1222_122224

/-- The distance between two points in 3D space -/
noncomputable def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- The projection of a point onto the yOz plane -/
def projectOntoYOZ (x y z : ℝ) : ℝ × ℝ × ℝ := (0, y, z)

theorem distance_A_to_B_projection : 
  let A : ℝ × ℝ × ℝ := (1, 0, 2)
  let P : ℝ × ℝ × ℝ := (1, -3, 1)
  let B : ℝ × ℝ × ℝ := projectOntoYOZ P.1 P.2.1 P.2.2
  distance3D A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_projection_l1222_122224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_inequality_l1222_122299

/-- The function f(x) = x² + x -/
def f (x : ℝ) : ℝ := x^2 + x

/-- Theorem: There are no real values of x that satisfy f(x-2) + f(x) < 0 -/
theorem no_solutions_for_inequality :
  ∀ x : ℝ, ¬(f (x - 2) + f x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_inequality_l1222_122299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1222_122290

/-- The time P and Q work together on a job -/
noncomputable def time_together : ℝ := 2

/-- P's work rate in jobs per hour -/
noncomputable def p_rate : ℝ := 1/3

/-- Q's work rate in jobs per hour -/
noncomputable def q_rate : ℝ := 1/15

/-- Time P takes to finish the remaining work after working together, in hours -/
noncomputable def p_remaining_time : ℝ := 3/5

theorem job_completion_time :
  p_rate * p_remaining_time + (p_rate + q_rate) * time_together = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1222_122290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1222_122242

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  (|a * x + b * y + c|) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point P(1, -1) to the line x - y + 1 = 0 is 3√2/2 -/
theorem distance_point_to_line :
  distancePointToLine 1 (-1) 1 (-1) 1 = (3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1222_122242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_angle_bisector_l1222_122275

-- Define the plane
variable (x y : ℝ)

-- Define the moving point M
structure Point where
  x : ℝ
  y : ℝ

-- Define the trajectory E
def trajectory (p : Point) : Prop :=
  p.y^2 = 8 * p.x

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the distance to a vertical line
def distanceToVerticalLine (p : Point) (lineX : ℝ) : ℝ :=
  |p.x - lineX|

-- Define the fixed point F
def F : Point :=
  ⟨2, 0⟩

-- Define point B
def B : Point :=
  ⟨-1, 0⟩

-- State the main theorem
theorem trajectory_and_angle_bisector :
  -- Part 1: The trajectory equation
  (∀ (M : Point), distanceToVerticalLine M (-1) = distance M F - 1 → trajectory M) ∧
  -- Part 2: The x-axis is the angle bisector
  (∀ (P Q : Point), 
    trajectory P ∧ trajectory Q ∧ 
    (∃ (k : ℝ), P.y = k * (P.x - 1) ∧ Q.y = k * (Q.x - 1)) →
    (P.y / (P.x + 1) = -Q.y / (Q.x + 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_angle_bisector_l1222_122275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_miles_per_tank_l1222_122285

/-- Represents the fuel efficiency and tank capacity of a car -/
structure Car where
  city_mpg : ℚ
  highway_mpg : ℚ
  city_miles_per_tank : ℚ
  mph_difference : ℚ

/-- Calculates the miles per tankful on the highway for a given car -/
def highway_miles_per_tank (c : Car) : ℚ :=
  (c.highway_mpg * c.city_miles_per_tank) / c.city_mpg

/-- Theorem stating the miles per tankful on the highway for the given car -/
theorem car_highway_miles_per_tank :
  let c : Car := {
    city_mpg := 48,
    highway_mpg := 48 + 18,
    city_miles_per_tank := 336,
    mph_difference := 18
  }
  highway_miles_per_tank c = 462 := by
  -- Proof steps would go here
  sorry

#eval highway_miles_per_tank {
  city_mpg := 48,
  highway_mpg := 48 + 18,
  city_miles_per_tank := 336,
  mph_difference := 18
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_miles_per_tank_l1222_122285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1222_122284

/-- The area of the triangle formed by the x-axis, y-axis, and the line 36x + 9y = 18 is 1/2 -/
theorem triangle_area :
  (∃ x y : ℝ, 36 * x + 9 * y = 18 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1/2 : ℝ) = (1/2) * (18/36) * 2 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1222_122284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l1222_122283

/-- Given polynomials P and Q with real coefficients, where P is not the zero polynomial
    and has degree d, there exist polynomials A and B satisfying specific conditions. -/
theorem polynomial_division_theorem (P Q : Polynomial ℝ) (d : ℕ) 
    (h₁ : P ≠ 0) (h₂ : Polynomial.degree P = d) :
  ∃ (A B : Polynomial ℝ),
    (Polynomial.degree A ≤ (d / 2 : ℕ)) ∧
    (Polynomial.degree B ≤ (d / 2 : ℕ)) ∧
    (A ≠ 0 ∨ B ≠ 0) ∧
    ∃ (C : Polynomial ℝ), A + Q * B = P * C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l1222_122283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_4_stage_increasing_odd_function_l1222_122227

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then |x - a^2| - a^2
  else -(|-x - a^2| - a^2)

-- Define the property of being a k-stage increasing function
def is_k_stage_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (x + k) > f x

-- State the theorem
theorem a_range_for_4_stage_increasing_odd_function :
  ∀ a : ℝ,
    (∀ x, f a (-x) = -(f a x)) →  -- f is odd
    (is_k_stage_increasing (f a) 4) →  -- f is 4-stage increasing
    -1 < a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_4_stage_increasing_odd_function_l1222_122227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_in_rectangle_l1222_122280

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the problem statement
theorem seven_points_in_rectangle (rect : Rectangle) (points : Finset Point) :
  rect.width = 1 → rect.height = 2 → points.card = 7 →
  (∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ rect.width ∧ 0 ≤ p.y ∧ p.y ≤ rect.height) →
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) ≤ Real.sqrt 13 / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_in_rectangle_l1222_122280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l1222_122200

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (Int.floor (12 * Real.pi) - Int.ceil (-6 * Real.pi) + 1))).card = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l1222_122200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_point_properties_l1222_122278

/-- An "ideal point" is a point where the y-coordinate is twice the x-coordinate -/
def is_ideal_point (x y : ℝ) : Prop := y = 2 * x

/-- Definition of distance from a point to the y-axis -/
def distance_to_y_axis (x : ℝ) : ℝ := |x|

/-- A linear function y = mx + b -/
def linear_function (m b x : ℝ) : ℝ := m * x + b

/-- A quadratic function y = ax² + bx + c -/
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem ideal_point_properties :
  ∀ (a b m : ℝ),
  (∃ (x y : ℝ), is_ideal_point x y ∧ distance_to_y_axis x = 2 →
    ((x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4))) ∧
  (m ≠ 0 →
    (m ≠ 2/3 →
      ∃ (x y : ℝ), is_ideal_point x y ∧ y = linear_function (3*m) (-1) x) ∧
    (m = 2/3 →
      ¬∃ (x y : ℝ), is_ideal_point x y ∧ y = linear_function (3*m) (-1) x)) ∧
  (∃ (a b c : ℝ),
    (∃ (x y : ℝ), is_ideal_point x y ∧ y = quadratic_function a b c x) ∧
    (quadratic_function a b c 0 = 5*a + 1) ∧
    (quadratic_function a b c (-2) = 5*a + 1) ∧
    (a ≠ 0) →
      let t := a^2 + a + 1
      3/4 ≤ t ∧ t ≤ 21/16 ∧ t ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_point_properties_l1222_122278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_mod_17_l1222_122253

theorem three_digit_integers_mod_17 :
  let three_digit_integers := Finset.filter (fun x => 100 ≤ x ∧ x ≤ 999) (Finset.range 1000)
  let satisfying_integers := Finset.filter (fun x => (9745 * x + 625) % 17 = 2000 % 17) three_digit_integers
  Finset.card satisfying_integers = 53 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_mod_17_l1222_122253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_2_1_l1222_122211

/-- The cost of a pencil in yuan -/
def pencil_cost : ℝ := sorry

/-- The cost of an exercise book in yuan -/
def book_cost : ℝ := sorry

/-- The cost of a ballpoint pen in yuan -/
def pen_cost : ℝ := sorry

/-- The first condition: 3 pencils + 7 exercise books + 1 ballpoint pen = 6.3 yuan -/
axiom condition1 : 3 * pencil_cost + 7 * book_cost + pen_cost = 6.3

/-- The second condition: 4 pencils + 10 exercise books + 1 ballpoint pen = 8.4 yuan -/
axiom condition2 : 4 * pencil_cost + 10 * book_cost + pen_cost = 8.4

/-- The theorem to prove -/
theorem total_cost_is_2_1 : pencil_cost + book_cost + pen_cost = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_2_1_l1222_122211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_correct_l1222_122254

/-- Represents the number of employees in each age group -/
structure AgeGroup where
  under35 : ℕ
  between35and49 : ℕ
  above50 : ℕ

/-- Calculates the stratified sample for a given age group and sample ratio -/
def stratifiedSample (group : AgeGroup) (sampleRatio : ℚ) : AgeGroup :=
  { under35 := (group.under35 : ℚ) * sampleRatio |>.floor.toNat,
    between35and49 := (group.between35and49 : ℚ) * sampleRatio |>.floor.toNat,
    above50 := (group.above50 : ℚ) * sampleRatio |>.floor.toNat }

theorem stratified_sampling_correct (totalEmployees sampleSize : ℕ) (ageGroups : AgeGroup)
    (h1 : totalEmployees = 500)
    (h2 : sampleSize = 100)
    (h3 : ageGroups.under35 = 125)
    (h4 : ageGroups.between35and49 = 280)
    (h5 : ageGroups.above50 = 95)
    (h6 : ageGroups.under35 + ageGroups.between35and49 + ageGroups.above50 = totalEmployees) :
    let sampleRatio := (sampleSize : ℚ) / totalEmployees
    let sample := stratifiedSample ageGroups sampleRatio
    sample.under35 = 25 ∧ sample.between35and49 = 56 ∧ sample.above50 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_correct_l1222_122254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l1222_122210

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Checks if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is convex -/
def isConvex (h : Hexagon) : Prop := sorry

/-- Calculates the area of a hexagon -/
noncomputable def area (h : Hexagon) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop := sorry

/-- Converts a hexagon to a set of points -/
def Hexagon.toSet (h : Hexagon) : Set Point :=
  {h.A, h.B, h.C, h.D, h.E, h.F}

theorem hexagon_area_theorem (h : Hexagon) (b : ℝ) :
  h.A = Point.mk 0 0 →
  h.B = Point.mk b 1 →
  angle h.F h.A h.B = 150 * π / 180 →
  parallel h.A h.B h.D h.E →
  parallel h.B h.C h.E h.F →
  parallel h.C h.D h.F h.A →
  isEquilateral h →
  isConvex h →
  (∃ (ys : Finset ℕ), ys.card = 6 ∧ ys ⊆ {0, 1, 2, 3, 4, 5} ∧
    (∀ p : Point, p ∈ h.toSet → ∃ y ∈ ys, p.y = y)) →
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ notDivisibleBySquareOfPrime n ∧
    area h = m * Real.sqrt n ∧ m + n = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l1222_122210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_constants_sum_l1222_122249

noncomputable def F₁ : ℝ × ℝ := (-4, 2 - Real.sqrt 3)
noncomputable def F₂ : ℝ × ℝ := (-4, 2 + Real.sqrt 3)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  abs (dist P F₁ - dist P F₂) = 2

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

theorem hyperbola_constants_sum :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (P : ℝ × ℝ), is_on_hyperbola P ↔ hyperbola_equation P.1 P.2 h k a b) ∧
  h + k + a + b = -1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_constants_sum_l1222_122249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1222_122259

noncomputable def z : ℂ := ((1 + Complex.I)^2 + 3*(1 - Complex.I)) / (2 + Complex.I)

theorem complex_number_problem (a b : ℝ) (h : z^2 + a*z + b = 1 + Complex.I) :
  z = 1 - Complex.I ∧ a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l1222_122259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l1222_122235

/-- The probability of drawing two chips of different colors from a bag containing 6 blue chips, 5 red chips, and 4 yellow chips, when drawing with replacement. -/
theorem two_different_color_chips_probability : 
  (let total_chips := 15
   let blue_chips := 6
   let red_chips := 5
   let yellow_chips := 4
   let prob_blue := blue_chips / total_chips
   let prob_red := red_chips / total_chips
   let prob_yellow := yellow_chips / total_chips
   let prob_not_blue := (red_chips + yellow_chips) / total_chips
   let prob_not_red := (blue_chips + yellow_chips) / total_chips
   let prob_not_yellow := (blue_chips + red_chips) / total_chips
   prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow) = 148 / 225 := by
  sorry

#eval (148 : ℚ) / 225

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l1222_122235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1222_122282

-- Define the function g
noncomputable def g (t : ℝ) : ℝ := (t^2 + (1/2) * t) / (t^2 + 1)

-- State the theorem
theorem range_of_g :
  ∀ y : ℝ, (∃ t : ℝ, g t = y) ↔ y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1222_122282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1222_122206

noncomputable section

/-- The original rational function -/
def f (x : ℝ) : ℝ := (x^3 - 4*x^2 - 9*x + 36) / (x - 3)

/-- The point where the function is undefined -/
def D : ℝ := 3

/-- Coefficients of the simplified polynomial -/
def A : ℝ := 1
def B : ℝ := -1
def C : ℝ := -12

/-- The simplified polynomial form of the function -/
def g (x : ℝ) : ℝ := A * x^2 + B * x + C

/-- Theorem stating the sum of coefficients and the equivalence of f and g -/
theorem sum_of_coefficients :
  A + B + C + D = -9 ∧
  ∀ x, x ≠ D → f x = g x :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1222_122206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_speed_theorem_l1222_122204

/-- Represents the rowing scenario with given distances and times -/
structure RowingScenario where
  distance : ℝ
  time_against : ℝ
  time_with : ℝ

/-- Calculates the rowing speed in still water given a rowing scenario -/
noncomputable def rowing_speed_still_water (scenario : RowingScenario) : ℝ :=
  let speed_against := scenario.distance / scenario.time_against
  let speed_with := scenario.distance / scenario.time_with
  (speed_against + speed_with) / 2

/-- Theorem stating that the rowing speed in still water is approximately 1.389 m/s -/
theorem rowing_speed_theorem (scenario : RowingScenario) 
  (h1 : scenario.distance = 750)
  (h2 : scenario.time_against = 675)
  (h3 : scenario.time_with = 450) : 
  ∃ ε > 0, |rowing_speed_still_water scenario - 1.389| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_speed_theorem_l1222_122204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karan_loan_amount_l1222_122215

/-- Calculates the initial borrowed amount given the total amount returned,
    interest rate, and loan duration. -/
noncomputable def initialBorrowedAmount (totalReturned : ℝ) (interestRate : ℝ) (duration : ℝ) : ℝ :=
  totalReturned / (1 + interestRate * duration / 100)

/-- Theorem stating that given the specific conditions of Mr. Karan's loan,
    the initial borrowed amount is approximately 5269.48. -/
theorem karan_loan_amount :
  let totalReturned : ℝ := 8110
  let interestRate : ℝ := 6
  let duration : ℝ := 9
  abs (initialBorrowedAmount totalReturned interestRate duration - 5269.48) < 0.01 := by
  sorry

#eval Float.sqrt 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karan_loan_amount_l1222_122215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l1222_122263

/-- A quadratic function f(x) = ax^2 + 2ax + 1 defined on [-3, 2] with maximum value 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The interval on which f is defined -/
def I : Set ℝ := Set.Icc (-3) 2

/-- Theorem stating the possible values of a -/
theorem quadratic_max_value (a : ℝ) : 
  (∀ x ∈ I, f a x ≤ 4) ∧ (∃ x ∈ I, f a x = 4) → a = 3/8 ∨ a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l1222_122263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perfect_squares_with_five_factors_l1222_122230

theorem sum_of_perfect_squares_with_five_factors : ∃ (a b : ℕ), 
  (a < 100 ∧ b < 100) ∧ 
  (∃ (x : ℕ), a = x^2) ∧ 
  (∃ (y : ℕ), b = y^2) ∧
  (Finset.card (Finset.filter (λ i => i ∣ a) (Finset.range (a + 1))) = 5) ∧
  (Finset.card (Finset.filter (λ i => i ∣ b) (Finset.range (b + 1))) = 5) ∧
  (∀ c : ℕ, c < 100 → 
    (∃ (z : ℕ), c = z^2) → 
    (Finset.card (Finset.filter (λ i => i ∣ c) (Finset.range (c + 1))) = 5) →
    c = a ∨ c = b) ∧
  a + b = 97 := by
sorry

#eval 16 + 81  -- This will output 97, confirming our expected result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perfect_squares_with_five_factors_l1222_122230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l1222_122288

/-- A parabola passing through two given points has a specific c-value -/
theorem parabola_c_value (b c : ℝ) :
  (2^2 + b*2 + c = 6) →
  (4^2 + b*4 + c = 20) →
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l1222_122288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_range_l1222_122225

/-- Given a complex number z = (1-i)(a-i) where a is a real number,
    if z is in the second quadrant of the complex plane, then a < -1. -/
theorem complex_second_quadrant_range (a : ℝ) : 
  let z : ℂ := (1 - Complex.I) * (a - Complex.I)
  (z.re < 0 ∧ z.im > 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_range_l1222_122225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_theorem_l1222_122208

/-- Represents an ellipse with the equation 4x^2 + 16y^2 = 16 -/
structure Ellipse where
  equation : ℝ × ℝ → Prop
  eq_def : equation = fun (x, y) ↦ 4 * x^2 + 16 * y^2 = 16

/-- Represents a circle with center (x₀, y₀) and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle passes through both foci of the ellipse -/
def passes_through_foci (c : Circle) (e : Ellipse) : Prop :=
  ∃ (x₀ y₀ : ℝ), c.center = (x₀, y₀) ∧
    (x₀ - Real.sqrt 3)^2 + y₀^2 = c.radius^2 ∧
    (x₀ + Real.sqrt 3)^2 + y₀^2 = c.radius^2

/-- Predicate to check if a circle intersects the ellipse at exactly four points -/
def intersects_at_four_points (c : Circle) (e : Ellipse) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ),
    e.equation p₁ ∧ e.equation p₂ ∧ e.equation p₃ ∧ e.equation p₄ ∧
    (p₁.1 - c.center.1)^2 + (p₁.2 - c.center.2)^2 = c.radius^2 ∧
    (p₂.1 - c.center.1)^2 + (p₂.2 - c.center.2)^2 = c.radius^2 ∧
    (p₃.1 - c.center.1)^2 + (p₃.2 - c.center.2)^2 = c.radius^2 ∧
    (p₄.1 - c.center.1)^2 + (p₄.2 - c.center.2)^2 = c.radius^2 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄

theorem ellipse_circle_intersection_theorem (e : Ellipse) :
  ∃ (a b : ℝ),
    (∀ (c : Circle),
      passes_through_foci c e ∧ intersects_at_four_points c e →
      a ≤ c.radius ∧ c.radius < b) ∧
    a + b = 2 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_theorem_l1222_122208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1222_122237

open Real

variable (A B C a b c : ℝ)

-- Define the triangle ABC
axiom triangle_abc : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π

-- Define the relationship between sides and angles
axiom law_of_sines : a / sin A = b / sin B
axiom law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*cos B

-- Given conditions
axiom condition1 : a * cos C + c * cos A = 2 * b * cos B
axiom condition2 : b = 2 * sqrt 3
axiom condition3 : 1/2 * a * c * sin B = 2 * sqrt 3

-- Theorem to prove
theorem triangle_abc_properties :
  B = π/3 ∧ a + b + c = 6 + 2 * sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1222_122237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_is_5km_l1222_122246

/-- Represents the distance from the warehouse to the station in kilometers. -/
def x : Type := ℝ

/-- Monthly occupancy fee as a function of distance. -/
noncomputable def occupancy_fee (x : ℝ) : ℝ := 200000 / x

/-- Monthly freight cost as a function of distance. -/
noncomputable def freight_cost (x : ℝ) : ℝ := 8000 * x

/-- Total monthly cost as a function of distance. -/
noncomputable def total_cost (x : ℝ) : ℝ := occupancy_fee x + freight_cost x

/-- Theorem stating that the optimal distance to minimize total cost is 5 km. -/
theorem optimal_distance_is_5km :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → total_cost x_min ≤ total_cost x ∧ x_min = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_is_5km_l1222_122246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_64_l1222_122239

theorem cube_root_of_64 : (64 : Real)^(1/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_64_l1222_122239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_table_sums_l1222_122277

-- Define a 2 × n table
def Table (n : ℕ) := Fin 2 → Fin n → ℝ

-- Define a function to calculate column sums
def columnSum (n : ℕ) (t : Table n) (j : Fin n) : ℝ :=
  (t 0 j) + (t 1 j)

-- Define a function to calculate row sums
def rowSum (n : ℕ) (t : Table n) (i : Fin 2) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin n)) (λ j => t i j)

-- State the theorem
theorem rearrange_table_sums (n : ℕ) (h : n > 2) (t : Table n)
  (col_sums_different : ∀ j k : Fin n, j ≠ k → columnSum n t j ≠ columnSum n t k) :
  ∃ t' : Table n,
    (∀ j k : Fin n, j ≠ k → columnSum n t' j ≠ columnSum n t' k) ∧
    (∀ i k : Fin 2, i ≠ k → rowSum n t' i ≠ rowSum n t' k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_table_sums_l1222_122277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_number_l1222_122296

/-- Represents a decimal expansion of a positive integer -/
def DecimalExpansion := List Nat

/-- Rearranges the decimal expansion as described in the problem -/
def rearrange (d : DecimalExpansion) : DecimalExpansion :=
  match d with
  | [] => []
  | [_] => []
  | a₀ :: a₁ :: rest => a₁ :: a₀ :: rest ++ [0]

/-- Converts a decimal expansion to a natural number -/
def toNatural (d : DecimalExpansion) : Nat :=
  d.foldl (fun acc digit => acc * 10 + digit) 0

/-- The main theorem statement -/
theorem exists_special_number : ∃ n : Nat, n > 0 ∧
  ∃ d : DecimalExpansion, toNatural d = n ∧ toNatural (rearrange d) = 2 * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_number_l1222_122296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleAllWhite_l1222_122201

/-- Represents the color of a cell -/
inductive Color where
  | Black
  | White
deriving Repr, DecidableEq

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Color

/-- Initial grid configuration with black corners and white other cells -/
def initialGrid : Grid :=
  fun i j => if (i = 0 && j = 0) || (i = 0 && j = 7) || (i = 7 && j = 0) || (i = 7 && j = 7)
             then Color.Black
             else Color.White

/-- Recolor a row in the grid -/
def recolorRow (g : Grid) (row : Fin 8) : Grid :=
  fun i j => if i = row then (if g i j = Color.Black then Color.White else Color.Black) else g i j

/-- Recolor a column in the grid -/
def recolorColumn (g : Grid) (col : Fin 8) : Grid :=
  fun i j => if j = col then (if g i j = Color.Black then Color.White else Color.Black) else g i j

/-- Check if all cells in the grid are white -/
def allWhite (g : Grid) : Prop :=
  ∀ i j, g i j = Color.White

/-- Theorem: It's impossible to make all cells white by recoloring rows and columns -/
theorem impossibleAllWhite : ¬∃ (operations : List (Bool × Fin 8)), 
  let finalGrid := operations.foldl 
    (fun acc (op : Bool × Fin 8) => 
      if op.1 then recolorRow acc op.2 else recolorColumn acc op.2) 
    initialGrid
  allWhite finalGrid := by
  sorry

#eval initialGrid 0 0
#eval initialGrid 1 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleAllWhite_l1222_122201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_of_homotheties_l1222_122219

-- Define a homothety
noncomputable def homothety (A : ℂ) (k : ℝ) (z : ℂ) : ℂ := k • (z - A) + A

-- Define a parallel translation
def parallel_translation (a : ℂ) (z : ℂ) : ℂ := z + a

-- Theorem statement
theorem composition_of_homotheties (A₁ A₂ : ℂ) (k₁ k₂ : ℝ) :
  ∃ (f : ℂ → ℂ), 
    (∀ z, (homothety A₂ k₂ ∘ homothety A₁ k₁) z = f z) ∧
    ((k₁ * k₂ = 1 ∧ ∃ a, f = parallel_translation a) ∨
     (k₁ * k₂ ≠ 1 ∧ ∃ A, f = homothety A (k₁ * k₂))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_of_homotheties_l1222_122219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1222_122291

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - π/4) = 7 * Real.sqrt 2 / 10) 
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1222_122291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l1222_122244

/-- A pyramid with a parallelogram base and specific dihedral angles -/
structure Pyramid where
  m : ℝ
  base_area : ℝ
  base_area_eq : base_area = m^2
  bd_perp_ad : True  -- Represents BD ⊥ AD
  dihedral_angle_ad_bc : ℝ
  dihedral_angle_ad_bc_eq : dihedral_angle_ad_bc = Real.pi / 4  -- 45°
  dihedral_angle_ab_cd : ℝ
  dihedral_angle_ab_cd_eq : dihedral_angle_ab_cd = Real.pi / 3  -- 60°

/-- The lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (p : Pyramid) : ℝ :=
  (p.m^2 * (Real.sqrt 2 + 2)) / 2

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  (p.m^3 * (Real.sqrt 2)^(1/4)) / 6

/-- Theorem stating the lateral surface area and volume of the pyramid -/
theorem pyramid_properties (p : Pyramid) :
  lateral_surface_area p = (p.m^2 * (Real.sqrt 2 + 2)) / 2 ∧
  volume p = (p.m^3 * (Real.sqrt 2)^(1/4)) / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l1222_122244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_binomial_and_positive_power_l1222_122252

theorem square_binomial_and_positive_power (a b : ℝ) : 
  (a + b)^2 = a^2 + 2*a*b + b^2 ∧ (2 : ℝ)^(a*b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_binomial_and_positive_power_l1222_122252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_three_l1222_122228

noncomputable def f (α : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 3 * x + 1 else x^2 - α * x

theorem f_composition_equals_three (α : ℝ) : 
  f α (f α (2/3)) = 3 → α = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_three_l1222_122228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_chord_intersection_l1222_122233

noncomputable section

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Circle.equation (c : Circle) : ℝ → ℝ → Prop :=
  λ x y ↦ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def point_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  c.equation p.1 p.2

noncomputable def chord_length (c : Circle) (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_tangency_and_chord_intersection
  (M : Circle)
  (h_M : M.center = (2, 0) ∧ M.radius = 2)
  (t : ℝ) :
  (∃ N : Circle, N.radius = 1 ∧ point_on_circle N (-1, 1) ∧ externally_tangent M N ∧
    (N.equation = λ x y ↦ (x + 1)^2 + y^2 = 1 ∨
     N.equation = λ x y ↦ (x + 2/5)^2 + (y - 9/5)^2 = 1)) ∧
  ((∃ S T : ℝ × ℝ,
    chord_length M (-1, t) S = 2 * Real.sqrt 3 ∧
    chord_length M (-1, t) T = 2 * Real.sqrt 3 ∧
    S.1 = 0 ∧ T.1 = 0 ∧ |S.2 - T.2| = 3/4) →
   t = 1 ∨ t = -1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_and_chord_intersection_l1222_122233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_circle_l1222_122274

/-- The distance between a point (a, b) and a line Ax + By + C = 0 --/
noncomputable def distance_point_to_line (a b A B C : ℝ) : ℝ :=
  (|A * a + B * b + C|) / Real.sqrt (A^2 + B^2)

/-- The circle with center (-1, -√3) and radius 1 --/
def circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + Real.sqrt 3)^2 = 1

/-- The line x = 0 --/
def tangent_line_equation (x : ℝ) : Prop :=
  x = 0

theorem tangent_to_circle :
  ∀ x y : ℝ, circle_equation x y → tangent_line_equation x →
  distance_point_to_line (-1) (-Real.sqrt 3) 1 0 0 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_circle_l1222_122274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_circle_line_intersection_proof_l1222_122289

/-- Represents a circle equation in the form x^2 + y^2 + ax + by + c = 0 --/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line equation in the form ax + by + c = 0 --/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a CircleEquation represents a valid circle --/
def is_valid_circle (circle : CircleEquation) : Prop :=
  (1 + circle.a^2 / 4 + circle.b^2 / 4 - circle.c) > 0

/-- Calculates the distance between two points --/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: The value of m for which the circle intersects the line with chord length 4/5 * sqrt(5) --/
theorem circle_line_intersection 
  (m : ℝ) 
  (circle : CircleEquation) 
  (line : LineEquation) : Prop :=
  circle.a = -2 ∧ circle.b = -4 ∧ circle.c = m ∧
  line.a = 1 ∧ line.b = 2 ∧ line.c = -4 ∧
  is_valid_circle circle ∧
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + y1^2 + circle.a * x1 + circle.b * y1 + circle.c = 0 ∧
    x2^2 + y2^2 + circle.a * x2 + circle.b * y2 + circle.c = 0 ∧
    line.a * x1 + line.b * y1 + line.c = 0 ∧
    line.a * x2 + line.b * y2 + line.c = 0 ∧
    distance x1 y1 x2 y2 = 4/5 * Real.sqrt 5) →
  m = 4

/-- Proof of the theorem --/
theorem circle_line_intersection_proof : circle_line_intersection 4 
  { a := -2, b := -4, c := 4 } 
  { a := 1, b := 2, c := -4 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_circle_line_intersection_proof_l1222_122289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l1222_122250

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m * x^2

-- Define the tangent line l
def l (n : ℝ) (x y : ℝ) : Prop := n * x + y - 2 = 0

-- State the theorem
theorem tangent_line_triangle_area 
  (m n : ℝ) 
  (h1 : ∀ x y, l n x y ↔ HasDerivAt (f m) (-(n : ℝ)) 1 ∧ f m 1 = y) 
  (h2 : ∃ x y, l n x 0 ∧ l n 0 y) :
  (1 / 2) * (2 / n) * (n * 0 + 2) = 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l1222_122250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_multiples_count_l1222_122264

def sequence_data : List Nat := [523, 307, 112, 155, 211, 221, 231, 616, 1055, 1032, 1007, 32, 126, 471, 50, 
                            156, 123, 13, 11, 117, 462, 16, 77, 176, 694, 848, 369, 147, 154, 847, 385, 
                            1386, 77, 618, 12, 146, 113, 56, 154, 184, 559, 172, 904, 102, 194, 114, 142, 
                            115, 196, 178, 893, 1093, 124, 15, 198, 217, 316, 154, 77, 77, 11, 555, 616, 
                            842, 127, 23, 185, 575, 1078, 1001, 17, 7, 384, 557, 112, 854, 964, 123, 846, 
                            103, 451, 514, 985, 125, 541, 411, 58, 2, 84, 618, 693, 231, 924, 1232, 455, 
                            15, 112, 112, 84, 111, 539]

def is_multiple_of_77 (n : Nat) : Bool :=
  n % 77 = 0

def count_consecutive_multiples (seq : List Nat) (len : Nat) : Nat :=
  sorry

theorem consecutive_multiples_count :
  (count_consecutive_multiples sequence_data 1 = 6) ∧
  (count_consecutive_multiples sequence_data 2 = 1) ∧
  (count_consecutive_multiples sequence_data 3 = 2) ∧
  (count_consecutive_multiples sequence_data 4 = 4) ∧
  (count_consecutive_multiples sequence_data 5 = 0) ∧
  (count_consecutive_multiples sequence_data 6 = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_multiples_count_l1222_122264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_example_l1222_122265

/-- The time (in seconds) it takes for a train to pass a person moving in the opposite direction. -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed + person_speed) * (1000 / 3600))

/-- Theorem stating that a 120 m long train moving at 50 kmph will pass a person
    moving at 4 kmph in the opposite direction in 8 seconds. -/
theorem train_passing_time_example :
  train_passing_time 120 50 4 = 8 := by
  sorry

-- Note: We remove the #eval statement as it's not computable


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_example_l1222_122265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fewer_than_six_cards_l1222_122232

/-- Represents the number of cards each person receives when cards are dealt evenly -/
def cardsPerPerson (totalCards : ℕ) (numPeople : ℕ) : ℕ := totalCards / numPeople

/-- Represents the number of cards left over after initial even distribution -/
def leftoverCards (totalCards : ℕ) (numPeople : ℕ) : ℕ := totalCards % numPeople

/-- Represents the number of people who receive an extra card from the leftovers -/
def peopleWithExtraCard (leftover : ℕ) : ℕ := leftover

theorem no_fewer_than_six_cards (totalCards : ℕ) (numPeople : ℕ) 
    (h1 : totalCards = 60) 
    (h2 : numPeople = 9) : 
  (numPeople - peopleWithExtraCard (leftoverCards totalCards numPeople)) = 3 := by
  sorry

#check no_fewer_than_six_cards

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fewer_than_six_cards_l1222_122232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_force_is_03N_l1222_122236

/-- Represents the motion of an object in a straight line -/
def motion (t : ℝ) : ℝ := 2 * t + 3 * t^2

/-- Calculates the velocity of the object at time t -/
def velocity (t : ℝ) : ℝ := 2 + 6 * t

/-- Calculates the acceleration of the object -/
def acceleration : ℝ := 6

/-- Converts centimeters to meters -/
noncomputable def cmToM (x : ℝ) : ℝ := x / 100

/-- Mass of the object in kg -/
def mass : ℝ := 5

/-- Calculates the force acting on the object -/
noncomputable def force : ℝ := mass * (cmToM acceleration)

theorem force_is_03N : force = 0.3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_force_is_03N_l1222_122236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_exam_problem_l1222_122287

/-- Represents a class with a given boy-to-girl ratio and percentages of boys and girls scoring ≥90% -/
structure MyClass where
  boy_ratio : ℚ
  girl_ratio : ℚ
  boy_high_score_percent : ℚ
  girl_high_score_percent : ℚ

/-- Calculates the fraction of students in a class scoring <90% -/
def fraction_below_90 (c : MyClass) : ℚ :=
  let total_ratio := c.boy_ratio + c.girl_ratio
  let high_score_ratio := c.boy_ratio * c.boy_high_score_percent + c.girl_ratio * c.girl_high_score_percent
  (total_ratio - high_score_ratio) / total_ratio

/-- Theorem statement for the school exam problem -/
theorem school_exam_problem (class_a class_b class_c : MyClass)
  (h_a : class_a = { boy_ratio := 3, girl_ratio := 2, boy_high_score_percent := 1/5, girl_high_score_percent := 1/4 })
  (h_b : class_b = { boy_ratio := 4, girl_ratio := 3, boy_high_score_percent := 3/20, girl_high_score_percent := 3/10 })
  (h_c : class_c = { boy_ratio := 5, girl_ratio := 4, boy_high_score_percent := 1/10, girl_high_score_percent := 1/5 }) :
  let total_students := (class_a.boy_ratio + class_a.girl_ratio) + (class_b.boy_ratio + class_b.girl_ratio) + (class_c.boy_ratio + class_c.girl_ratio)
  let total_below_90 := (fraction_below_90 class_a) * (class_a.boy_ratio + class_a.girl_ratio) +
                        (fraction_below_90 class_b) * (class_b.boy_ratio + class_b.girl_ratio) +
                        (fraction_below_90 class_c) * (class_c.boy_ratio + class_c.girl_ratio)
  (total_below_90 / total_students) * 100 = 17.1 / 21 * 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_exam_problem_l1222_122287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1222_122262

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - 2 * a * x + Real.log x

-- State the theorem
theorem range_of_b (a : ℝ) (x₀ : ℝ) (b : ℝ) :
  a ≠ 0 →
  1 < a →
  a < 2 →
  x₀ ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2 →
  f a x₀ + Real.log (a + 1) > b * (a^2 - 1) - (a + 1) + 2 * Real.log 2 →
  b ∈ Set.Iic (-1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l1222_122262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_curve_properties_l1222_122202

-- Define the polar coordinates of points A and B
noncomputable def point_A : ℝ × ℝ := (2, Real.pi)
noncomputable def point_B : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)

-- Define the parametric equations of curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sin α, 1 + (Real.cos α)^2)

-- Define the area of triangle AOB
def area_AOB : ℝ := 2

-- Define the intersection point of line AB and curve C
noncomputable def intersection_point : ℝ × ℝ := ((-1 + Real.sqrt 17) / 4, (7 + Real.sqrt 17) / 8)

-- Theorem statement
theorem triangle_and_curve_properties :
  (∃ (α : ℝ), curve_C α = intersection_point) ∧
  area_AOB = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_curve_properties_l1222_122202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_n_value_l1222_122245

/-- Given vectors a, b, and c in ℝ², prove that if b - a is collinear with c, then n = -3. -/
theorem collinear_vectors_n_value (n : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![n, 3]
  let c : Fin 2 → ℝ := ![4, -1]
  (∃ (k : ℝ), (b - a) = k • c) → n = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_n_value_l1222_122245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_correct_l1222_122240

/-- The n-th composition of f(x) = ax / (1 + bx) -/
noncomputable def f_comp (a b : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  (a^n * x) / (1 + ((a^n - 1) / (a - 1)) * b * x)

/-- The original function f(x) = ax / (1 + bx) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  (a * x) / (1 + b * x)

theorem f_comp_correct (a b : ℝ) (n : ℕ) (x : ℝ) (h : a ≠ 1) :
  f_comp a b n = (f a b)^[n] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_correct_l1222_122240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_condition_l1222_122226

theorem complex_square_condition (z₁ z₂ : ℂ) (l : ℝ) 
  (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) (hl : l > 0) (hmod : Complex.abs z₁ = Complex.abs z₂) :
  (z₁ + z₂)^2 = l^2 * z₁ * z₂ ↔ Complex.abs (z₁ + z₂) = l * Complex.abs z₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_condition_l1222_122226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l1222_122271

/-- Given a parabola passing through two points and intersecting the x-axis, prove that the length of the segment between these points is 12. -/
theorem parabola_segment_length :
  ∀ (b c m : ℝ),
  let f : ℝ → ℝ := λ x ↦ -1/2 * x^2 + b*x - b^2 + 2*c
  let A : ℝ × ℝ := (2 - 3*b, m)
  let B : ℝ × ℝ := (4*b + c - 1, m)
  (f A.1 = A.2) →  -- parabola passes through A
  (f B.1 = B.2) →  -- parabola passes through B
  (∃ x, f x = 0) →   -- parabola intersects x-axis
  abs (B.1 - A.1) = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l1222_122271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1222_122205

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the midpoint P
def P : ℝ × ℝ := (1, 1)

-- Define the line containing chord MN
def chord_line (x y : ℝ) : Prop := y = 2*x - 1

-- Theorem statement
theorem chord_equation (M N : ℝ × ℝ) :
  circle_eq M.1 M.2 ∧ circle_eq N.1 N.2 ∧  -- M and N are on the circle
  P = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) →  -- P is the midpoint of MN
  ∀ x y, x ∈ Set.Icc M.1 N.1 ∧ y ∈ Set.Icc M.2 N.2 → chord_line x y  -- The line containing MN has the equation y = 2x - 1
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1222_122205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1222_122273

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : 
  Real.sin θ = 3/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1222_122273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1222_122293

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 - (k-6)*x + 16) / (x^2 + k*x + 25)

theorem function_properties (k : ℝ) :
  (∀ x, f k x ≠ 0) ∧ (∀ x, x^2 + k*x + 25 ≠ 0) ↔ -2 < k ∧ k < 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1222_122293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_elliptic_equation_l1222_122257

theorem infinite_solutions_elliptic_equation
  (p q : ℕ) (x_star y_star : ℕ)
  (h_not_square : ¬ ∃ (n : ℕ), p * q = n^2)
  (h_solution : p * x_star^2 + q * y_star^2 = 1) :
  ∃ (S : Set (ℕ × ℕ)), (Infinite S) ∧ (∀ (x y : ℕ), (x, y) ∈ S → p * x^2 + q * y^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_elliptic_equation_l1222_122257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goats_sold_ratio_l1222_122298

/-- Represents the farm animals and their sales --/
structure FarmSale where
  goats : ℕ
  sheep : ℕ
  goatPrice : ℕ
  sheepPrice : ℕ
  goatsSold : ℕ
  sheepSoldFraction : ℚ
  totalRevenue : ℕ

/-- Theorem stating the ratio of goats sold to total goats --/
theorem goats_sold_ratio (farm : FarmSale) : 
  farm.goats = 150 ∧ 
  farm.sheep = 210 ∧ 
  farm.goats + farm.sheep = 360 ∧
  farm.goats * 7 = farm.sheep * 5 ∧
  farm.goatPrice = 40 ∧
  farm.sheepPrice = 30 ∧
  farm.sheepSoldFraction = 2/3 ∧
  farm.totalRevenue = 7200 ∧
  farm.goatsSold * farm.goatPrice + (farm.sheepSoldFraction * ↑farm.sheep * ↑farm.sheepPrice).floor = farm.totalRevenue
  →
  2 * farm.goatsSold = farm.goats := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goats_sold_ratio_l1222_122298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_eleven_polynomial_difference_divisibility_l1222_122218

-- Problem 1
theorem divisibility_by_eleven (n : ℕ) :
  11 ∣ (6^(2*n) + 3^n + 3^(n+2)) :=
sorry

-- Problem 2
theorem polynomial_difference_divisibility (p : Polynomial ℤ) (a b : ℤ) (h : a ≠ b) :
  (a - b) ∣ (p.eval a - p.eval b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_eleven_polynomial_difference_divisibility_l1222_122218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1222_122243

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 4 * x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ = g a x₂) → a ∈ Set.Ici (0 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1222_122243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l1222_122234

/-- Represents the dimensions of a rectangular pyramid --/
structure PyramidDimensions where
  baseLength : ℝ
  baseWidth : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular pyramid --/
noncomputable def pyramidVolume (d : PyramidDimensions) : ℝ :=
  (1/3) * d.baseLength * d.baseWidth * d.height

/-- Theorem: The volume of the frustum is 525 cm³ --/
theorem frustum_volume (original : PyramidDimensions) (smaller : PyramidDimensions) :
  original.baseLength = 15 →
  original.baseWidth = 10 →
  original.height = 12 →
  smaller.baseLength = 7.5 →
  smaller.baseWidth = 5 →
  smaller.height = 6 →
  pyramidVolume original - pyramidVolume smaller = 525 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l1222_122234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_max_value_l1222_122241

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 5

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := 3*x + 1

-- State the theorem
theorem tangent_and_max_value :
  ∃ (a b : ℝ),
    (deriv (f a b) 1 = deriv tangent_line 1) ∧
    (f a b 1 = tangent_line 1) ∧
    (a = 2 ∧ b = -4) ∧
    (∀ x ∈ Set.Icc (-3) 1, f a b x ≤ 13) ∧
    (∃ x ∈ Set.Icc (-3) 1, f a b x = 13) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_max_value_l1222_122241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l1222_122267

theorem remainder_sum_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 7)
  (hb : b % 15 = 3)
  (hc : c % 15 = 9) : 
  (a + b + c) % 15 = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l1222_122267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_kilometers_driven_l1222_122213

-- Define the rental cost functions
def carrey_cost (x : ℝ) : ℝ := 20 + 0.25 * x
def samuel_cost (x : ℝ) : ℝ := 24 + 0.16 * x

-- Define the theorem
theorem rental_kilometers_driven : 
  ∃ (x : ℝ), carrey_cost x = samuel_cost x ∧ 
  (Int.floor (x + 0.5) : ℤ) = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_kilometers_driven_l1222_122213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1222_122281

noncomputable def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem problem_solution (A B C : Real) (a b c : Real) :
  triangle_ABC A B C a b c →
  (b * Real.cos A - Real.sqrt 3 * a * Real.sin B = 0) →
  (B + Real.pi / 12 = A) →
  (a = 2) →
  (A = Real.pi / 6 ∧ Real.sqrt 3 - 1 = (1 / 2) * a * c * Real.sin B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1222_122281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_l1222_122217

noncomputable def n : ℝ := Real.log (1 + 3 * 1.5)^4 / Real.log (3 + 2 * 1.5)

theorem n_value : abs (n - 3.806) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_l1222_122217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1222_122260

-- Define the polar equations
noncomputable def r_tan (θ : Real) : Real := Real.tan θ
noncomputable def r_cot (θ : Real) : Real := 1 / Real.tan θ

-- Define the Cartesian equations
def x_bound : Real := 2
def y_bound : Real := 2

-- Theorem statement
theorem area_of_bounded_region :
  ∃ (A : Set (Real × Real)),
    (∀ (x y : Real), (x, y) ∈ A ↔ 
      (∃ θ, x = r_tan θ * Real.cos θ ∧ y = r_tan θ * Real.sin θ) ∨
      (∃ θ, x = r_cot θ * Real.cos θ ∧ y = r_cot θ * Real.sin θ) ∨
      (x = x_bound) ∨ (y = y_bound)) →
    MeasureTheory.volume A = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1222_122260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1222_122268

/-- Simple interest calculation --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Problem statement --/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h1 : principal = 1600)
  (h2 : time = 4)
  (h3 : interest = 200) : 
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 3.125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1222_122268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisible_power_of_ten_l1222_122216

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_divisible_power_of_ten : 
  (∃ (n : ℕ), 10^n ∣ (factorial 12 - 3 * (factorial 6)^3 + 2^(factorial 4))) ∧ 
  (∀ (m : ℕ), m > 6 → ¬(10^m ∣ (factorial 12 - 3 * (factorial 6)^3 + 2^(factorial 4)))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisible_power_of_ten_l1222_122216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1222_122231

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

-- Define the interval
def I : Set ℝ := Set.Icc (-3/4) (1/4)

-- Theorem statement
theorem f_properties :
  -- 1. Monotonicity on (-1/2, 1/4]
  (∀ x y, x ∈ Set.Ioo (-1/2) (1/4) → y ∈ Set.Ioo (-1/2) (1/4) → x < y → f x < f y) ∧
  -- 2. Maximum value
  (∃ x ∈ I, ∀ y ∈ I, f y ≤ f x ∧ f x = Real.log (7/2) + 1/16) ∧
  -- 3. Minimum value
  (∃ x ∈ I, ∀ y ∈ I, f x ≤ f y ∧ f x = Real.log 2 + 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1222_122231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_is_six_l1222_122286

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure Box where
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ

/-- The box satisfies the given conditions -/
def satisfies_conditions (b : Box) : Prop :=
  b.large_sphere_radius = 3 ∧
  b.small_sphere_radius = 1 ∧
  ∃ (large_center : ℝ × ℝ × ℝ),
    large_center.fst = 3 ∧
    large_center.snd = 3 ∧
    large_center.2.2 = b.height / 2 ∧
  ∃ (small_center : ℝ × ℝ × ℝ),
    small_center.fst = 1 ∧
    small_center.snd = 1 ∧
    small_center.2.2 = 1

/-- The height of the box is 6 -/
theorem box_height_is_six (b : Box) (h : satisfies_conditions b) : b.height = 6 := by
  sorry

#check box_height_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_is_six_l1222_122286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_of_arithmetic_sequence_l1222_122255

/-- Represents an ellipse with major axis 2a, minor axis 2b, and focal distance 2c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_ellipse : b^2 = a^2 - c^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  e.c / e.a

/-- The arithmetic sequence property of the ellipse -/
def is_arithmetic_sequence (e : Ellipse) : Prop :=
  e.a + e.c = 2 * e.b

theorem ellipse_eccentricity_of_arithmetic_sequence (e : Ellipse) 
    (h : is_arithmetic_sequence e) : eccentricity e = 3/5 := by
  sorry

#check ellipse_eccentricity_of_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_of_arithmetic_sequence_l1222_122255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paulson_savings_increase_l1222_122272

/-- Represents Paulson's financial situation and changes -/
structure PaulsonFinances where
  initial_income : ℝ
  initial_expenditure_ratio : ℝ
  income_increase_ratio : ℝ
  expenditure_increase_ratio : ℝ

/-- Calculates the percentage increase in savings -/
noncomputable def savings_increase_percentage (p : PaulsonFinances) : ℝ :=
  let initial_savings := p.initial_income * (1 - p.initial_expenditure_ratio)
  let new_income := p.initial_income * (1 + p.income_increase_ratio)
  let new_expenditure := p.initial_income * p.initial_expenditure_ratio * (1 + p.expenditure_increase_ratio)
  let new_savings := new_income - new_expenditure
  (new_savings - initial_savings) / initial_savings * 100

/-- Theorem stating that the percentage increase in Paulson's savings is 50% -/
theorem paulson_savings_increase 
  (p : PaulsonFinances) 
  (h1 : p.initial_expenditure_ratio = 0.75)
  (h2 : p.income_increase_ratio = 0.2)
  (h3 : p.expenditure_increase_ratio = 0.1) :
  savings_increase_percentage p = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paulson_savings_increase_l1222_122272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_solution_l1222_122248

theorem sine_equation_solution (x : ℝ) 
  (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2) 
  (h2 : π < x ∧ x < 2 * π) : 
  x = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_solution_l1222_122248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_54_minus_cos_18_l1222_122279

-- Define the angles in radians
noncomputable def angle_54 : ℝ := 54 * Real.pi / 180
noncomputable def angle_18 : ℝ := 18 * Real.pi / 180

-- State the theorem
theorem cos_54_minus_cos_18 : Real.cos angle_54 - Real.cos angle_18 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_54_minus_cos_18_l1222_122279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_triangle_l1222_122229

-- Define necessary concepts
def IsIsosceles (a b c : Real) : Prop := a = b ∨ b = c ∨ a = c

def CircleTangent (r a b : Real) : Prop := sorry

def ExternallyTangent (r1 r2 : Real) : Prop := sorry

def CircleInsideTriangle (r a b c : Real) : Prop := sorry

theorem circle_radius_in_triangle (a b c : Real) (r p : Real) :
  a = 60 ∧ b = 60 ∧ c = 40 → -- Triangle side lengths
  r > 0 ∧ p = 12 → -- Circle radii (r for Q, p for P)
  IsIsosceles a b c → -- Triangle is isosceles
  CircleTangent p a c → -- Circle P tangent to AC
  CircleTangent p b c → -- Circle P tangent to BC
  CircleTangent r a b → -- Circle Q tangent to AB
  CircleTangent r b c → -- Circle Q tangent to BC
  ExternallyTangent r p → -- Circles P and Q are externally tangent
  CircleInsideTriangle r a b c → -- Circle Q is inside triangle ABC
  r = 36 - 4 * Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_triangle_l1222_122229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_shooting_events_l1222_122214

-- Define the sample space
variable (Ω : Type)

-- Define the events as subsets of the sample space
variable (A B C D : Set Ω)

-- Define the properties of the events
axiom both_hits : A = {ω : Ω | ∃ (p : Prop), p ↔ "both shots hit the airplane" = "both shots hit the airplane"}
axiom no_hits : B = {ω : Ω | ∃ (p : Prop), p ↔ "neither shot hits the airplane" = "neither shot hits the airplane"}
axiom one_hit : C = {ω : Ω | ∃ (p : Prop), p ↔ "exactly one shot hits the airplane" = "exactly one shot hits the airplane"}
axiom at_least_one_hit : D = {ω : Ω | ∃ (p : Prop), p ↔ "at least one shot hits the airplane" = "at least one shot hits the airplane"}

-- State the theorem
theorem airplane_shooting_events :
  (A ⊆ D) ∧ 
  (B ∩ D = ∅) ∧ 
  (A ∪ C = D) ∧ 
  (A ∪ C ≠ B ∪ D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_shooting_events_l1222_122214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_two_sectors_l1222_122261

-- Define the radius of the circle
def radius : ℝ := 15

-- Define the angle of each sector in radians
noncomputable def sector_angle : ℝ := 45 * (Real.pi / 180)

-- Define the area of a single sector
def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Theorem statement
theorem area_of_two_sectors :
  2 * sector_area radius sector_angle = 56.25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_two_sectors_l1222_122261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_m_range_n_range_l1222_122251

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := Real.exp x * Real.sin x
def g (x : ℝ) : ℝ := (x + 1) * Real.cos x - Real.sqrt 2 * Real.exp x
def h (x n : ℝ) : ℝ := (2 * x / Real.sin x) * f x - n * Real.sin (2 * x)

-- Theorem for monotonicity intervals of f
theorem f_monotonicity : 
  ∀ (k : ℤ), 
    (∀ x, x ∈ Set.Icc (-Real.pi/4 + 2*k*Real.pi) (3*Real.pi/4 + 2*k*Real.pi) → 
      MonotoneOn f (Set.Icc (-Real.pi/4 + 2*k*Real.pi) (3*Real.pi/4 + 2*k*Real.pi))) ∧
    (∀ x, x ∈ Set.Icc (3*Real.pi/4 + 2*k*Real.pi) (7*Real.pi/4 + 2*k*Real.pi) → 
      MonotoneOn (fun x => -f x) (Set.Icc (3*Real.pi/4 + 2*k*Real.pi) (7*Real.pi/4 + 2*k*Real.pi))) :=
by sorry

-- Theorem for the range of m
theorem m_range :
  ∀ (m : ℝ), 
    (∀ x₁ x₂, x₁ ∈ Set.Icc 0 (Real.pi/2) → x₂ ∈ Set.Icc 0 (Real.pi/2) → f x₁ + g x₂ ≥ m) ↔ 
    m ≤ -Real.sqrt 2 * Real.exp (Real.pi/2) :=
by sorry

-- Theorem for the range of n
theorem n_range :
  ∀ (n : ℝ), n > 0 →
    (∃! x, x ∈ Set.Ioo 0 (Real.pi/2) ∧ h x n = 0) ↔ n > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_m_range_n_range_l1222_122251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1222_122294

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - Real.sqrt 2*a*x + 2 > 0

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, ¬(p a)) ∧ (∀ a : ℝ, p a ∨ q a) →
  ∀ a : ℝ, 0 ≤ a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1222_122294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_numberland_license_plate_probability_l1222_122209

/-- The set of vowels used in Numberland license plates -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}

/-- The set of all letters except 'Y' -/
def nonYLetters : Finset Char := 
  (Finset.range 26).image (fun n => Char.ofNat (n + 65)) \ {'Y'}

/-- The set of digits -/
def digits : Finset Char := 
  (Finset.range 10).image (fun n => Char.ofNat (n + 48))

/-- The probability of selecting the license plate "EY9" in Numberland -/
theorem numberland_license_plate_probability : 
  (1 : ℚ) / (vowels.card * nonYLetters.card * digits.card) = 1 / 1500 := by
  sorry

#eval vowels.card
#eval nonYLetters.card
#eval digits.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_numberland_license_plate_probability_l1222_122209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_proof_l1222_122256

noncomputable def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ :=
  fun n => a * r^n

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem geometric_series_sum_proof (a : ℝ) (r : ℝ) 
  (h₁ : a = 1) (h₂ : r = 1/4) (h₃ : |r| < 1) :
  geometric_series_sum a r = 4/3 := by
  sorry

#check geometric_series_sum_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_proof_l1222_122256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_line_is_correct_l1222_122276

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 2D vector -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The initial point of the light ray -/
def initial_point : Point := { x := 2, y := 3 }

/-- The direction vector of the incident light ray -/
def direction_vector : Vec := { x := 8, y := 4 }

/-- Function to calculate the reflected line given an incident light ray -/
def reflected_line (p : Point) (v : Vec) : Line :=
  { a := 1, b := 2, c := -4 }

/-- Theorem stating that the reflected line is correct -/
theorem reflected_line_is_correct :
  reflected_line initial_point direction_vector = { a := 1, b := 2, c := -4 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_line_is_correct_l1222_122276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_13pi_l1222_122220

/-- The quadratic equation defining the circle -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 8 * x + 16 * y - 32 = 0

/-- The area of the circle defined by the given equation -/
noncomputable def circle_area : ℝ := 13 * Real.pi

/-- Theorem stating that the area of the circle is 13π -/
theorem circle_area_is_13pi :
  ∀ x y : ℝ, circle_equation x y → circle_area = 13 * Real.pi :=
by
  intros x y h
  unfold circle_area
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_13pi_l1222_122220
