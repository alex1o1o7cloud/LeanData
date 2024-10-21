import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l543_54344

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the distance function from a point to F
noncomputable def distToF (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - F.1)^2 + P.2^2)

-- Define the distance function from a point to y-axis
def distToYAxis (P : ℝ × ℝ) : ℝ := |P.1|

-- Define the trajectory condition
def satisfiesTrajectory (P : ℝ × ℝ) : Prop :=
  distToF P = distToYAxis P + 1

-- Theorem statement
theorem trajectory_equation (P : ℝ × ℝ) :
  satisfiesTrajectory P ↔ 
    (P.1 ≥ 0 ∧ P.2^2 = 4 * P.1) ∨ (P.1 ≤ 0 ∧ P.2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l543_54344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l543_54306

theorem smallest_number_divisible (n : ℕ) : n = 1010 ↔ 
  (∀ m : ℕ, m < n → ¬(∀ d : ℕ, d ∈ ({12, 16, 18, 21, 28} : Finset ℕ) → (m - 2) % d = 0)) ∧ 
  (∀ d : ℕ, d ∈ ({12, 16, 18, 21, 28} : Finset ℕ) → (n - 2) % d = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l543_54306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_point_y_value_l543_54365

theorem sine_point_y_value (α : ℝ) (y : ℝ) :
  Real.sin α = -1/2 →
  (∃ P : ℝ × ℝ, P.1 = 2 ∧ P.2 = y ∧ P.1 > 0 ∧ (P.2 / P.1 = Real.tan α)) →
  y = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_point_y_value_l543_54365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_proofs_l543_54336

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (3, 1)

noncomputable def point_A : ℝ × ℝ := (5, -1)
noncomputable def point_B : ℝ × ℝ := (-1, 7)
noncomputable def point_C : ℝ × ℝ := (1, 2)

noncomputable def vector_a2 : ℝ × ℝ := (1, 1)
noncomputable def length_b : ℝ := 4
noncomputable def dot_product_ab : ℝ := Real.pi/4

theorem vector_proofs :
  (∃ (x y : ℝ), x • vector_a + y • vector_b = (1, 0) ∧ x • vector_a + y • vector_b = (0, 1)) ∧
  (point_A.1 - point_B.1 + point_C.1, point_A.2 - point_B.2 + point_C.2) = (7, -6) ∧
  (2 : ℝ) • vector_a2 = (length_b * Real.cos dot_product_ab / Real.sqrt 2) • vector_a2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_proofs_l543_54336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l543_54317

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^5 + Real.log (x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ 0) ∧
  ¬(∀ a b : ℝ, f a + f b ≥ 0 → a + b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l543_54317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_theorem_l543_54362

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 - y^2 = 1
def C₂ (p x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the origin O and point P
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (7/2, 0)

-- State the theorem
theorem intersection_circle_theorem (p : ℝ) (M N : ℝ × ℝ) :
  C₁ M.1 M.2 ∧ C₁ N.1 N.2 ∧  -- M and N lie on C₁
  C₂ p M.1 M.2 ∧ C₂ p N.1 N.2 ∧  -- M and N lie on C₂
  (∃ (center : ℝ × ℝ) (radius : ℝ),  -- Circumcircle condition
    (center.1 - O.1)^2 + (center.2 - O.2)^2 = radius^2 ∧
    (center.1 - M.1)^2 + (center.2 - M.2)^2 = radius^2 ∧
    (center.1 - N.1)^2 + (center.2 - N.2)^2 = radius^2 ∧
    (center.1 - P.1)^2 + (center.2 - P.2)^2 = radius^2) →
  p = 3/4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_theorem_l543_54362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_circle_intersection_l543_54358

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | distance center p = radius}

theorem triangle_with_circle_intersection
  (ABC : Triangle)
  (h1 : distance ABC.A ABC.B = 86)
  (h2 : distance ABC.A ABC.C = 97)
  (X : ℝ × ℝ)
  (h3 : X ∈ Circle ABC.A 86)
  (h4 : ∃ (bx cx : ℤ), distance ABC.B X = bx ∧ distance X ABC.C = cx) :
  distance ABC.B ABC.C = 94 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_circle_intersection_l543_54358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_minus_2i_l543_54367

theorem min_abs_z_minus_2i :
  ∃ (min : ℝ), min = 1 ∧ ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z - 2*Complex.I) ≥ min :=
by
  use 1
  constructor
  · rfl
  · intro z hz
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_minus_2i_l543_54367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l543_54371

-- Define the circles
def circle_C : ℝ × ℝ → Prop := λ p => let (x, y) := p; x^2 + y^2 = 1
def circle_D : ℝ × ℝ → Prop := λ p => let (x, y) := p; x^2 + y^2 - 4*x + 2*y - 4 = 0

-- Define the number of common tangent lines
noncomputable def num_common_tangents (C D : (ℝ × ℝ) → Prop) : ℕ := 
  sorry

-- Theorem statement
theorem circles_have_two_common_tangents : 
  num_common_tangents circle_C circle_D = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l543_54371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l543_54322

/-- Two lines in the plane are perpendicular if and only if the product of their slopes is -1 -/
def two_lines_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of a line Ax + By + C = 0 is -A/B -/
noncomputable def line_slope (A B : ℝ) : ℝ := -A / B

theorem perpendicular_lines_a_equals_one : 
  ∀ a : ℝ, 
  two_lines_perpendicular (line_slope 1 a) (line_slope (a + 1) (-2)) → 
  a = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l543_54322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cosine_sine_equation_l543_54320

theorem unique_solution_cosine_sine_equation :
  ∃! (n : ℕ), n > 0 ∧ Real.cos (π / (2 * n)) - Real.sin (π / (2 * n)) = Real.sqrt n / 3 :=
by
  -- The unique solution is n = 6
  use 6
  constructor
  · constructor
    · norm_num
    · sorry  -- Proof of the equation for n = 6
  · intro m ⟨m_pos, hm⟩
    sorry  -- Proof of uniqueness


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cosine_sine_equation_l543_54320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_possible_l543_54337

-- Define a Step type
inductive Step
  | operational : Step

-- Define the characteristics of an algorithm
structure Algorithm where
  steps : List Step
  is_finite : List.length steps < ω
  is_clear : ∀ s ∈ steps, s = Step.operational
  result_is_clear : Bool

-- Define a problem
structure Problem where
  description : String

-- Define a function that checks if a given algorithm solves a problem
def solves_problem (a : Algorithm) (p : Problem) : Bool := sorry

-- Theorem: There can be multiple algorithms for a problem
theorem multiple_algorithms_possible (p : Problem) : 
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ solves_problem a1 p ∧ solves_problem a2 p := by
  sorry

#check multiple_algorithms_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_possible_l543_54337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_moment_of_inertia_circle_moment_of_inertia_l543_54324

-- Define the rectangle
structure Rectangle where
  a : ℝ
  b : ℝ

-- Define the circle
structure Circle where
  R : ℝ

-- Define the moment of inertia calculations
noncomputable def momentOfInertiaRectangleVertex (rec : Rectangle) (ρ : ℝ) : ℝ :=
  (ρ * rec.a * rec.b * (rec.a^2 + rec.b^2)) / 3

noncomputable def momentOfInertiaRectangleCentroid (rec : Rectangle) (ρ : ℝ) : ℝ :=
  (ρ * rec.a * rec.b * (rec.a^2 + rec.b^2)) / 12

noncomputable def momentOfInertiaCircleTangent (circ : Circle) (ρ : ℝ) : ℝ :=
  (5 * Real.pi * ρ * circ.R^4) / 4

noncomputable def momentOfInertiaCircleCenter (circ : Circle) (ρ : ℝ) : ℝ :=
  (3 * Real.pi * ρ * circ.R^4) / 2

-- Theorem statements
theorem rectangle_moment_of_inertia (rec : Rectangle) (ρ : ℝ) :
  momentOfInertiaRectangleVertex rec ρ = (ρ * rec.a * rec.b * (rec.a^2 + rec.b^2)) / 3 ∧
  momentOfInertiaRectangleCentroid rec ρ = (ρ * rec.a * rec.b * (rec.a^2 + rec.b^2)) / 12 := by
  sorry

theorem circle_moment_of_inertia (circ : Circle) (ρ : ℝ) :
  momentOfInertiaCircleTangent circ ρ = (5 * Real.pi * ρ * circ.R^4) / 4 ∧
  momentOfInertiaCircleCenter circ ρ = (3 * Real.pi * ρ * circ.R^4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_moment_of_inertia_circle_moment_of_inertia_l543_54324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_inverse_product_l543_54305

theorem fraction_sum_inverse_product : 12 * (1/3 + 1/4 + 1/6 : ℚ)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_inverse_product_l543_54305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_d_value_l543_54395

def a : Fin 2 → ℝ := ![1, -3]
def b : Fin 2 → ℝ := ![-2, 4]
def c : Fin 2 → ℝ := ![-1, -2]

theorem vector_d_value :
  let d : Fin 2 → ℝ := ![-2, -6]
  4 • a + (4 • b - 2 • c) + 2 • (a - c) + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_d_value_l543_54395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_circle_l543_54319

theorem six_points_in_circle (points : Finset (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) :
  radius = 1 →
  Finset.card points = 6 →
  (∀ p ∈ points, dist p center ≤ radius) →
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ dist p q ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_circle_l543_54319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l543_54303

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem function_properties 
  (ω : ℝ) (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < Real.pi / 2) 
  (h_period : ∀ x, f ω φ (x + Real.pi / 2) = f ω φ x) 
  (h_symmetry : ∀ x, f ω φ (-Real.pi / 3 + x) = f ω φ (-Real.pi / 3 - x)) :
  (∀ x, f ω φ x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (∀ α β a, α ≠ β → α ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3) → β ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3) → 
    g α = a → g β = a → α + β = 7 * Real.pi / 6) ∧
  (∀ a, (∃ x, x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3) ∧ g x = a) ↔ a ∈ Set.Ioo (-2) (-Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l543_54303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_ratio_l543_54397

/-- Circle with diameter AB and center C, points D and E on the diameter, and random point F --/
structure CircleWithPoints where
  center : ℝ × ℝ
  radius : ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The probability that triangle DEF has perimeter less than diameter --/
noncomputable def perimeter_probability (c : CircleWithPoints) : ℝ := 17 / 128

/-- The ratio of DE to AB --/
noncomputable def segment_ratio (c : CircleWithPoints) : ℝ := 
  Real.sqrt ((Real.sqrt 16095) / 128)

theorem circle_segment_ratio (c : CircleWithPoints) :
  perimeter_probability c = 17 / 128 → segment_ratio c = 127 / 128 := by
  sorry

#eval 127 + 128

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_ratio_l543_54397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_48_26459_to_hundredth_l543_54321

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The statement that rounding 48.26459 to the nearest hundredth equals 48.26 -/
theorem round_48_26459_to_hundredth :
  roundToHundredth 48.26459 = 48.26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_48_26459_to_hundredth_l543_54321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_highways_for_ten_cities_l543_54338

/-- A type representing a city --/
def City : Type := Fin 10

/-- A function representing the existence of a highway between two cities --/
def highway : City → City → Prop := sorry

/-- The condition that for any three cities, either all pairs are connected or exactly two pairs are not connected --/
def three_city_condition : Prop :=
  ∀ (a b c : City), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (highway a b ∧ highway b c ∧ highway a c) ∨
    (¬highway a b ∧ ¬highway b c ∧ highway a c) ∨
    (¬highway a b ∧ highway b c ∧ ¬highway a c) ∨
    (highway a b ∧ ¬highway b c ∧ ¬highway a c)

/-- The theorem stating the minimum number of highways required for 10 cities --/
theorem min_highways_for_ten_cities :
  three_city_condition →
  (∀ (a b : City), a ≠ b → (highway a b ↔ highway b a)) →
  (∃ (count : ℕ), count = 40 ∧
    (∀ (c : ℕ), c < count →
      ∃ (a b : City), a ≠ b ∧ highway a b) ∧
    (∀ (c : ℕ), c ≥ count →
      ¬∃ (a b : City), a ≠ b ∧ highway a b ∧
        (∀ (i j : City), i ≠ j ∧ highway i j →
          ∃ (k : ℕ), k < c ∧ (i = a ∧ j = b ∨ i = b ∧ j = a)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_highways_for_ten_cities_l543_54338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_four_plus_four_sqrt_five_l543_54312

/-- A square with side length 4 -/
structure Square :=
  (side_length : ℝ)
  (is_four : side_length = 4)

/-- The midpoint of a side -/
structure Midpoint :=
  (x : ℝ)
  (y : ℝ)

/-- Distance from a point to a midpoint -/
noncomputable def distance_to_midpoint (x y : ℝ) (m : Midpoint) : ℝ :=
  Real.sqrt ((x - m.x)^2 + (y - m.y)^2)

/-- Sum of distances from a vertex to all midpoints -/
noncomputable def sum_distances (s : Square) (x y : ℝ) (midpoints : List Midpoint) : ℝ :=
  (midpoints.map (distance_to_midpoint x y)).sum

/-- Theorem: The sum of distances from a vertex to midpoints equals 4 + 4√5 -/
theorem sum_distances_equals_four_plus_four_sqrt_five (s : Square) :
  ∃ (x y : ℝ) (midpoints : List Midpoint),
    sum_distances s x y midpoints = 4 + 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_four_plus_four_sqrt_five_l543_54312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l543_54363

noncomputable def angle : ℝ := 600 * (Real.pi / 180)

def point (a : ℝ) : ℝ × ℝ := (-4, a)

theorem point_on_terminal_side (a : ℝ) :
  point a = (-4, a) ∧ 
  (point a).1 = -4 * Real.cos angle ∧ 
  (point a).2 = -4 * Real.sin angle →
  a = -4 * Real.sqrt 3 := by
  sorry

#check point_on_terminal_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l543_54363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_lines_intersection_impossibility_l543_54381

theorem seven_lines_intersection_impossibility :
  ¬ ∃ (n m : ℕ), n ≥ 6 ∧ m ≥ 4 ∧ (Nat.choose 7 2 < 3 * n + m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_lines_intersection_impossibility_l543_54381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_on_parabola_l543_54386

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Distance between points A and B on a parabola -/
theorem distance_AB_on_parabola (C : Parabola) (F A B : Point) :
  C.equation = fun x y ↦ y^2 = 4*x →  -- Parabola equation
  F.x = 1 ∧ F.y = 0 →  -- Focus position
  C.equation A.x A.y →  -- A is on the parabola
  B.x = 3 ∧ B.y = 0 →  -- B position
  distance A F = distance B F →  -- |AF| = |BF|
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_on_parabola_l543_54386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_m_n_l543_54389

theorem least_sum_m_n : ∃ (m n : ℕ), 
  (m > 0) ∧ (n > 0) ∧
  (Nat.gcd (m + n) 231 = 1) ∧ 
  (∃ k : ℕ, m^m = k * n^n) ∧ 
  (∀ k : ℕ, m ≠ k * n) ∧
  (m + n = 24) ∧
  (∀ m' n' : ℕ, 
    m' > 0 → n' > 0 →
    (Nat.gcd (m' + n') 231 = 1) → 
    (∃ k : ℕ, m'^m' = k * n'^n') → 
    (∀ k : ℕ, m' ≠ k * n') → 
    (m' + n' ≥ 24)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_m_n_l543_54389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_degree_term_linear_term_coefficient_l543_54377

-- Define the polynomial
noncomputable def p (x y : ℝ) : ℝ := -4/5 * x^2 * y + 2/3 * x^4 * y^2 - x + 1

-- Theorem for the highest degree term
theorem highest_degree_term :
  ∃ (c : ℝ), c ≠ 0 ∧ 
  ∀ (x y : ℝ), (p x y - c * x^4 * y^2) < x^4 * y^2 :=
sorry

-- Theorem for the coefficient of the linear term
theorem linear_term_coefficient :
  ∃ (t : ℝ → ℝ → ℝ), (∀ x y, |t x y| < |x|) ∧
  ∀ (x y : ℝ), p x y = -x + t x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_degree_term_linear_term_coefficient_l543_54377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_A_worked_approx_five_l543_54354

/-- The number of days A worked before leaving the job -/
noncomputable def days_A_worked (total_days_A : ℝ) (total_days_B : ℝ) (days_B_completed : ℝ) : ℝ :=
  (total_days_A * total_days_B - days_B_completed * total_days_A) / 
  (total_days_A + total_days_B - days_B_completed)

/-- Theorem stating that A worked approximately 5 days before leaving the job -/
theorem days_A_worked_approx_five :
  let total_days_A : ℝ := 15
  let total_days_B : ℝ := 14.999999999999996
  let days_B_completed : ℝ := 10
  abs (days_A_worked total_days_A total_days_B days_B_completed - 5) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_A_worked_approx_five_l543_54354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_length_l543_54382

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, 1 - (Real.sqrt 3 / 2) * t)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Theorem statement
theorem intersection_and_length :
  (∃ (A B : ℝ × ℝ), A ≠ B ∧
    (∃ (t₁ t₂ : ℝ), line_l t₁ = A ∧ line_l t₂ = B) ∧
    (∃ (θ₁ θ₂ : ℝ), 
      Real.sqrt ((A.1)^2 + (A.2)^2) = circle_C θ₁ ∧
      Real.sqrt ((B.1)^2 + (B.2)^2) = circle_C θ₂) ∧
    (∀ (P : ℝ × ℝ), (∃ (t : ℝ), line_l t = P) ∧ 
      (∃ (θ : ℝ), Real.sqrt ((P.1)^2 + (P.2)^2) = circle_C θ) → 
      P = A ∨ P = B)) ∧
  (∀ (A B : ℝ × ℝ), 
    (∃ (t₁ t₂ : ℝ), line_l t₁ = A ∧ line_l t₂ = B) ∧
    (∃ (θ₁ θ₂ : ℝ), 
      Real.sqrt ((A.1)^2 + (A.2)^2) = circle_C θ₁ ∧
      Real.sqrt ((B.1)^2 + (B.2)^2) = circle_C θ₂) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_length_l543_54382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l543_54311

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 2)

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 2 * Real.pi / 3) ∧
  (∀ x, f (x + 2 * Real.pi / 3) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l543_54311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_decomposition_cube_decomposition_five_squared_find_m_l543_54345

def sum_first_n_odd (n : ℕ) : ℕ := n^2

def sum_m_consecutive_odd_from_mth (m : ℕ) : ℕ := m^3

def decompose_square (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2*i + 1)

def decompose_cube (m : ℕ) : List ℕ :=
  List.range m |>.map (fun i => 2*(m + i) - 1)

theorem square_decomposition (n : ℕ) (h : n ≥ 2) :
  sum_first_n_odd n = (decompose_square n).sum := by sorry

theorem cube_decomposition (m : ℕ) (h : m ≥ 2) :
  sum_m_consecutive_odd_from_mth m = (decompose_cube m).sum := by sorry

theorem five_squared :
  5^2 = [1, 3, 5, 7, 9].sum := by sorry

theorem find_m (m : ℕ) :
  (decompose_cube m).minimum? = some 21 → m = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_decomposition_cube_decomposition_five_squared_find_m_l543_54345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_exists_l543_54328

noncomputable def numbers (x : ℝ) : List ℝ := [3, 7, 12, 21, x]

noncomputable def mean (x : ℝ) : ℝ := (3 + 7 + 12 + 21 + x) / 5

def median (x : ℝ) : ℝ := 12

theorem unique_x_exists : ∃! x : ℝ, 
  (x > 12) ∧ 
  (median x = 9 + (mean x) ^ (1/4)) ∧
  (x = 362) := by
  sorry

#check unique_x_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_exists_l543_54328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_proof_l543_54349

/-- The capacity of the pool in cubic meters -/
noncomputable def pool_capacity : ℝ := 12000

/-- The time it takes to fill the pool with both valves open, in minutes -/
noncomputable def both_valves_time : ℝ := 48

/-- The time it takes to fill the pool with the first valve alone, in minutes -/
noncomputable def first_valve_time : ℝ := 120

/-- The difference in water emission between the second and first valve, in cubic meters per minute -/
noncomputable def valve_difference : ℝ := 50

/-- The rate at which the first valve fills the pool, in cubic meters per minute -/
noncomputable def first_valve_rate : ℝ := pool_capacity / first_valve_time

/-- The rate at which the second valve fills the pool, in cubic meters per minute -/
noncomputable def second_valve_rate : ℝ := first_valve_rate + valve_difference

theorem pool_capacity_proof :
  pool_capacity = 12000 ∧
  first_valve_rate + second_valve_rate = pool_capacity / both_valves_time :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_proof_l543_54349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_and_increasing_l543_54361

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(-m^2 + 2*m + 3)

theorem power_function_even_and_increasing (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  (∀ x y : ℝ, 0 < x → x < y → f m x < f m y) →
  m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_and_increasing_l543_54361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_blue_quadrilateral_l543_54333

/-- A color can be either red or blue -/
inductive Color
| Red
| Blue

/-- A segment in the nonagon -/
structure Segment where
  start : Fin 9
  finish : Fin 9
  color : Color

/-- A regular nonagon with colored segments -/
structure ColoredNonagon where
  segments : List Segment

/-- Check if three vertices form a red triangle -/
def hasRedTriangle (n : ColoredNonagon) : Prop :=
  ∃ (a b c : Fin 9), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∃ (s1 s2 s3 : Segment), 
      s1 ∈ n.segments ∧ s2 ∈ n.segments ∧ s3 ∈ n.segments ∧
      ((s1.start = a ∧ s1.finish = b) ∨ (s1.start = b ∧ s1.finish = a)) ∧
      ((s2.start = b ∧ s2.finish = c) ∨ (s2.start = c ∧ s2.finish = b)) ∧
      ((s3.start = c ∧ s3.finish = a) ∨ (s3.start = a ∧ s3.finish = c)) ∧
      s1.color = Color.Red ∧ s2.color = Color.Red ∧ s3.color = Color.Red)

/-- Check if four vertices form a blue quadrilateral -/
def hasBlueQuadrilateral (n : ColoredNonagon) : Prop :=
  ∃ (a b c d : Fin 9),
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
    (∀ (s : Segment), 
      s ∈ n.segments ∧
      ((s.start = a ∧ s.finish = b) ∨ (s.start = b ∧ s.finish = a) ∨
       (s.start = b ∧ s.finish = c) ∨ (s.start = c ∧ s.finish = b) ∨
       (s.start = c ∧ s.finish = d) ∨ (s.start = d ∧ s.finish = c) ∨
       (s.start = d ∧ s.finish = a) ∨ (s.start = a ∧ s.finish = d) ∨
       (s.start = a ∧ s.finish = c) ∨ (s.start = c ∧ s.finish = a) ∨
       (s.start = b ∧ s.finish = d) ∨ (s.start = d ∧ s.finish = b)) →
      s.color = Color.Blue)

theorem nonagon_blue_quadrilateral (n : ColoredNonagon) :
  (∀ (s : Segment), s ∈ n.segments → s.color = Color.Red ∨ s.color = Color.Blue) →
  ¬hasRedTriangle n →
  hasBlueQuadrilateral n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_blue_quadrilateral_l543_54333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_fixed_point_l543_54307

-- Define the exponential function as noncomputable
noncomputable def exp_function (a : ℝ) (x : ℝ) : ℝ := a^x

-- Theorem statement
theorem exp_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  exp_function a 0 = 1 := by
  -- Unfold the definition of exp_function
  unfold exp_function
  -- Use the property of exponents: a^0 = 1 for any a ≠ 0
  exact Real.rpow_zero a


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_fixed_point_l543_54307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l543_54347

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 - 4*x + y^2 = 0

-- Define the distance between two points
noncomputable def my_distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the theorem
theorem point_B_coordinates :
  ∃ a : ℝ, 
    (∀ x y : ℝ, my_circle x y → my_distance x y a 0 = 2 * my_distance x y 1 0) →
    a = -2 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l543_54347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_three_l543_54327

/-- Sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
noncomputable def arithmetic_sum (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

/-- The theorem stating that if the ratio of T_{4n} to T_n is constant for an arithmetic sequence
    with common difference 5, then the first term of the sequence is 3 -/
theorem first_term_is_three (a : ℝ) :
  (∀ n : ℕ+, ∃ k : ℝ, arithmetic_sum a 5 (4 * n) / arithmetic_sum a 5 n = k) →
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_three_l543_54327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_project_theorem_l543_54399

/-- Represents the construction project -/
structure ConstructionProject where
  initialDays : ℕ  -- Initial planned days
  workedDays : ℕ   -- Days worked before change
  earlierDays : ℕ  -- Days to complete earlier

/-- Represents a construction team -/
structure ConstructionTeam where
  daysToComplete : ℚ  -- Days to complete the project alone

/-- Calculate the days needed for both teams to complete the project -/
noncomputable def daysForBothTeams (team1 team2 : ConstructionTeam) : ℚ :=
  1 / (1 / team1.daysToComplete + 1 / team2.daysToComplete)

theorem construction_project_theorem (project : ConstructionProject) 
  (team1 team2 : ConstructionTeam) : 
  project.initialDays = 30 ∧ 
  project.workedDays = 10 ∧ 
  project.earlierDays = 8 ∧
  team1.daysToComplete = 30 →
  team2.daysToComplete = 45 ∧ 
  daysForBothTeams team1 team2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_project_theorem_l543_54399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l543_54394

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3*a) / Real.log (1/2)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x y, 2 ≤ x ∧ x < y → f a y < f a x) →
  -4 < a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l543_54394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l543_54343

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem function_properties (a b : ℝ) :
  (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → f x + f y = f ((x + y) / (1 + x * y))) →
  (∀ x, x ∈ Set.Ioo (-1) 0 → f x > 0) →
  (f ((a + b) / (1 + a * b)) = 1) →
  (f ((a - b) / (1 - a * b)) = 2) →
  (abs a < 1) →
  (abs b < 1) →
  (f (-1/2) = 1) →
  (f a = 3/2 ∧ f b = -1/2 ∧ f (2 - Real.sqrt 3) = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l543_54343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_double_implies_k_sqrt_three_l543_54355

theorem slope_angle_double_implies_k_sqrt_three (k : ℝ) (α : ℝ) : 
  Real.tan α = Real.sqrt 3 / 3 → 
  Real.tan (2 * α) = k → 
  k = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_double_implies_k_sqrt_three_l543_54355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l543_54376

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x - (a + 1) * x - 1

theorem tangent_line_and_range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x > 0) ↔ a ≥ 0 ∧
  (∀ x : ℝ, x = 0 → f a x = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l543_54376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equation_l543_54375

theorem square_root_equation (a b : ℝ) : 
  (Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b)) → (a + b = 41) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equation_l543_54375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l543_54314

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := (x - 3) / 2
noncomputable def g_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem composition_result : 
  f (g_inv (f_inv (f (g (f_inv (f (g (f 11)))))))) = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l543_54314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l543_54318

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the lines 3x + 4y - 3 = 0 and 6x + 8y + 14 = 0 is 11/5 -/
theorem distance_between_specific_lines :
  distance_between_parallel_lines 3 4 (-3) (-14) = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l543_54318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christine_bought_four_more_l543_54364

/-- Represents the price of a marker in cents -/
def marker_price : ℕ := sorry

/-- Represents the number of markers Lucas bought -/
def lucas_markers : ℕ := sorry

/-- Represents the number of markers Christine bought -/
def christine_markers : ℕ := sorry

/-- The price of a marker is more than 1 cent -/
axiom price_gt_penny : marker_price > 1

/-- Lucas paid $2.25 for his markers -/
axiom lucas_paid : marker_price * lucas_markers = 225

/-- Christine paid $3.25 for her markers -/
axiom christine_paid : marker_price * christine_markers = 325

/-- Theorem: Christine bought exactly 4 more markers than Lucas -/
theorem christine_bought_four_more : christine_markers = lucas_markers + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christine_bought_four_more_l543_54364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_zero_g_range_l543_54351

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - a + 1) * x^(a + 1)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x + Real.sqrt (1 - 2*x)

-- Theorem 1: Prove that a = 0
theorem a_equals_zero (h1 : ∀ x, f 0 x = x) 
                      (h2 : ∀ x, f 0 (-x) = -(f 0 x)) : 0 = 0 := by
  sorry

-- Theorem 2: Prove the range of g(x)
theorem g_range : Set.Icc (1/2 : ℝ) 1 = Set.range (g ∘ (Set.Icc 0 (1/2)).restrict g) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_zero_g_range_l543_54351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_major_axis_length_l543_54334

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The ellipse is tangent to a horizontal line at y = 1 -/
  tangent_horizontal_at_one : Bool
  /-- The x-coordinate of the first focus -/
  focus1_x : ℝ
  /-- The y-coordinate of both foci -/
  foci_y : ℝ
  /-- The x-coordinate of the second focus -/
  focus2_x : ℝ
  /-- The foci are symmetric about the y-axis -/
  foci_symmetric : focus1_x = -focus2_x

/-- The length of the major axis of the special ellipse -/
def major_axis_length (e : SpecialEllipse) : ℝ := 2

/-- The length of the major axis of the special ellipse is 2 -/
theorem special_ellipse_major_axis_length (e : SpecialEllipse)
    (h1 : e.tangent_y_axis = true)
    (h2 : e.tangent_horizontal_at_one = true)
    (h3 : e.focus1_x = -Real.sqrt 5)
    (h4 : e.focus2_x = Real.sqrt 5)
    (h5 : e.foci_y = 2) :
    2 = major_axis_length e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_major_axis_length_l543_54334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_satisfying_conditions_l543_54300

/-- The equation of a plane that satisfies specific conditions --/
theorem plane_equation_satisfying_conditions :
  ∃ (A B C D : ℤ),
    A > 0 ∧
    Int.gcd A (Int.gcd B (Int.gcd C D)) = 1 ∧
    (∀ (x y z : ℝ),
      (x + y + 2*z = 4 ∧ 2*x - 2*y + z = 1) →
      (A*x + B*y + C*z + D = 0)) ∧
    (A ≠ 1 ∨ B ≠ 1 ∨ C ≠ 2 ∨ D ≠ -4) ∧
    (A ≠ 2 ∨ B ≠ -2 ∨ C ≠ 1 ∨ D ≠ -1) ∧
    (Real.sqrt ((A*1 + B*2 + C*3 + D)^2 / (A^2 + B^2 + C^2 : ℝ)) = 1) ∧
    A = 6 ∧ B = 0 ∧ C = 5 ∧ D = -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_satisfying_conditions_l543_54300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_of_specific_sequence_l543_54380

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  length : ℕ
  first : ℚ
  last : ℚ

/-- Returns the nth term of an arithmetic sequence -/
noncomputable def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first + (n - 1 : ℚ) * ((seq.last - seq.first) / (seq.length - 1 : ℚ))

theorem eighth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk 25 3 78
  nthTerm seq 8 = 199/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_of_specific_sequence_l543_54380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l543_54331

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 --/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The equation of a circle with center (h, k) and radius r --/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line :
  let center_x : ℝ := 3
  let center_y : ℝ := 0
  let line_A : ℝ := 1
  let line_B : ℝ := Real.sqrt 2
  let line_C : ℝ := 0
  let radius := distance_point_to_line center_x center_y line_A line_B line_C
  ∀ x y : ℝ,
    circle_equation x y center_x center_y radius ↔
    (x - 3)^2 + y^2 = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l543_54331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_initial_weight_l543_54385

/-- Represents Jack's weight loss scenario -/
structure WeightLossScenario where
  current_weight : ℝ
  future_weight : ℝ
  months_to_future : ℝ
  months_on_diet : ℝ

/-- Calculates the initial weight given a WeightLossScenario -/
noncomputable def calculate_initial_weight (scenario : WeightLossScenario) : ℝ :=
  let weight_loss_rate := (scenario.current_weight - scenario.future_weight) / scenario.months_to_future
  scenario.current_weight + weight_loss_rate * scenario.months_on_diet

/-- Theorem stating that Jack's initial weight was 200.4 pounds -/
theorem jack_initial_weight :
  let scenario := WeightLossScenario.mk 198 180 45 6
  calculate_initial_weight scenario = 200.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_initial_weight_l543_54385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l543_54308

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 0  -- Add a case for 0 to avoid missing case error
  | 1 => 3
  | 2 => 5
  | n + 3 => a (n + 2) + 2  -- Changed to n + 3 to ensure termination

-- Define S_n as the sum of the first n terms of a_n
def S : ℕ → ℚ
  | 0 => 0
  | n + 1 => S n + a (n + 1)

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a n / 3^n

-- Define T_n as the sum of the first n terms of b_n
def T : ℕ → ℚ
  | 0 => 0
  | n + 1 => T n + b (n + 1)

-- State the main theorem
theorem sequence_properties :
  (∀ n ≥ 3, S n + S (n - 2) = 2 * S (n - 1) + 2) →
  (∀ n ≥ 2, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n : ℕ, T n = 2 - (n + 2) / 3^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l543_54308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_calculation_l543_54315

/-- The marked price calculation problem -/
theorem marked_price_calculation
  (initial_price : ℝ)
  (initial_discount_rate : ℝ)
  (profit_margin : ℝ)
  (final_discount_rate : ℝ)
  (h1 : initial_price = 30)
  (h2 : initial_discount_rate = 0.15)
  (h3 : profit_margin = 0.20)
  (h4 : final_discount_rate = 0.25) :
  (initial_price * (1 - initial_discount_rate) * (1 + profit_margin)) / (1 - final_discount_rate) = 40.80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_calculation_l543_54315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l543_54304

/-- Line represented by a point and a direction vector -/
structure Line (α : Type*) [Field α] where
  point : α × α
  direction : α × α

/-- The intersection point of two lines -/
noncomputable def intersection (α : Type*) [Field α] (l1 l2 : Line α) : α × α :=
  sorry

/-- The first line -/
noncomputable def line1 : Line ℝ :=
  { point := (1, 1), direction := (-3, 4) }

/-- The second line -/
noncomputable def line2 : Line ℝ :=
  { point := (2, -7), direction := (6, -1) }

/-- Theorem stating that the intersection of the two lines is (5.5, -5) -/
theorem intersection_point : intersection ℝ line1 line2 = (5.5, -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l543_54304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l543_54335

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - Real.sin x + 6
def g (x k : ℝ) : ℝ := x^2 - k

-- State the theorem
theorem find_k : ∃ k : ℝ, f 3 - g 3 k = 5 ∧ k = -55 + Real.sin 3 := by
  -- Introduce k
  let k := -55 + Real.sin 3
  
  -- Use 'exists.intro' to provide the value of k
  use k
  
  -- Split the goal into two parts
  constructor
  
  -- Prove f 3 - g 3 k = 5
  · simp [f, g]
    -- The rest of the proof would go here
    sorry
  
  -- Prove k = -55 + Real.sin 3
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l543_54335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_quarters_mowing_l543_54378

/-- The number of quarters Sam got for mowing lawns -/
def quarters_from_mowing (total_amount : ℚ) (pennies : ℕ) : ℕ :=
  (((total_amount - (pennies : ℚ) / 100) / (1 : ℚ) / 4).floor : ℤ).natAbs

/-- Theorem stating that Sam got 7 quarters for mowing lawns -/
theorem sam_quarters_mowing :
  quarters_from_mowing (184 / 100) 9 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_quarters_mowing_l543_54378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_square_park_l543_54368

/-- The shortest path from one corner to the opposite corner of a square park with a circular flowerbed -/
theorem shortest_path_square_park (a d : ℝ) (h₁ : a > 0) (h₂ : d > 0) (h₃ : d < a) (h₄ : d/a < 0.746) :
  let diagonal_path := a * Real.sqrt 2 - d + d * Real.pi / 2
  let around_path := 2 * a - d + d * Real.pi / 4
  diagonal_path < around_path ∧ diagonal_path < a * Real.sqrt 2 := by
  sorry

#check shortest_path_square_park

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_square_park_l543_54368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_division_l543_54359

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A point on a line segment -/
def PointOnSegment (A B : ℝ × ℝ) := { P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B }

/-- The area of a polygon -/
noncomputable def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

/-- The ratio in which a point divides a line segment -/
noncomputable def divisionRatio (A P B : ℝ × ℝ) : ℝ × ℝ := sorry

theorem hexagon_area_division (ABCDEF : RegularHexagon) 
  (K : PointOnSegment (ABCDEF.vertices 4) (ABCDEF.vertices 5)) :
  area [ABCDEF.vertices 0, K.val, ABCDEF.vertices 4, ABCDEF.vertices 5, ABCDEF.vertices 0] = 
    (3/4) * area [ABCDEF.vertices 0, ABCDEF.vertices 1, ABCDEF.vertices 2, 
                  ABCDEF.vertices 3, ABCDEF.vertices 4, ABCDEF.vertices 5] →
  divisionRatio (ABCDEF.vertices 4) K.val (ABCDEF.vertices 5) = (3, 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_division_l543_54359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_rationals_l543_54360

def numbers : List ℚ := [22/7, -2, 0, 4/3, 32/100]

theorem count_positive_rationals :
  (numbers.filter (λ x => x > 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_rationals_l543_54360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_exists_l543_54350

structure GeometricSetup where
  -- The plane
  Plane : Type
  -- Point type
  Point : Type
  -- Line type
  Line : Type
  -- Circle type
  Circle : Type
  -- Distance function
  distance : Point → Point → ℝ
  -- Incidence relation for a point on a line
  on_line : Point → Line → Prop
  -- Incidence relation for a point on a circle
  on_circle : Point → Circle → Prop
  -- Function to get the center of a circle
  center : Circle → Point
  -- Function to get the radius of a circle
  radius : Circle → ℝ
  -- Perpendicular relation between lines
  perpendicular : Line → Line → Prop
  -- Midpoint of a line segment
  midpoint : Point → Point → Point
  -- Line through two points
  line_through : Point → Point → Line

variable (S : GeometricSetup)

def secant_construction (c : S.Circle) (l : S.Line) : Prop :=
  ∃ (O X Y A B A' B' C : S.Point) (aux_circle : S.Circle),
    -- O is the center of the given circle
    O = S.center c
    -- Auxiliary circle has radius 3 times the given circle
    ∧ S.radius aux_circle = 3 * S.radius c
    -- X and Y are intersections of auxiliary circle with given line
    ∧ S.on_line X l ∧ S.on_circle X aux_circle
    ∧ S.on_line Y l ∧ S.on_circle Y aux_circle
    -- A, B, A', B' are intersections of OX and OY with given circle
    ∧ S.on_circle A c ∧ S.on_circle B c
    ∧ S.on_circle A' c ∧ S.on_circle B' c
    -- C is on the line AB'
    ∧ S.on_line C (S.line_through A B')
    -- AB'C is perpendicular to l
    ∧ S.perpendicular (S.line_through A B') l
    -- C is the midpoint of AB'
    ∧ C = S.midpoint A B'

theorem secant_exists (c : S.Circle) (l : S.Line) : secant_construction S c l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_exists_l543_54350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l543_54341

open Real

-- Define the given spherical coordinates
noncomputable def ρ : ℝ := 15
noncomputable def θ : ℝ := 3 * π / 4
noncomputable def φ : ℝ := π / 4

-- Define the conversion functions from spherical to rectangular coordinates
noncomputable def x (ρ θ φ : ℝ) : ℝ := ρ * sin φ * cos θ
noncomputable def y (ρ θ φ : ℝ) : ℝ := ρ * sin φ * sin θ
noncomputable def z (ρ θ φ : ℝ) : ℝ := ρ * cos φ

-- State the theorem
theorem spherical_to_rectangular_conversion :
  (x ρ θ φ, y ρ θ φ, z ρ θ φ) = (-15/2, 15/2, 15*sqrt 2/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l543_54341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shuttlecock_weight_probability_l543_54372

theorem shuttlecock_weight_probability (P : Set ℝ → ℝ) 
  (h1 : P {x | x < 4.8} = 0.3) 
  (h2 : P {x | x < 4.85} = 0.32) :
  P {x | 4.8 ≤ x ∧ x < 4.85} = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shuttlecock_weight_probability_l543_54372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_75_l543_54316

/-- Represents the boat's journey with given parameters. -/
structure BoatJourney where
  downstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the upstream distance given a boat journey. -/
noncomputable def upstream_distance (journey : BoatJourney) : ℝ :=
  let boat_speed := journey.downstream_distance / journey.downstream_time - journey.stream_speed
  (boat_speed - journey.stream_speed) * journey.upstream_time

/-- Theorem stating that for the given journey parameters, the upstream distance is 75 km. -/
theorem upstream_distance_is_75 :
  let journey := BoatJourney.mk 100 4 15 10
  upstream_distance journey = 75 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_75_l543_54316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l543_54310

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 4 + a 8 = 12) :
  S a 11 = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l543_54310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_female_given_female_participation_score_female_relationship_expected_sum_of_scores_l543_54370

-- Define the number of male and female teachers
def num_male : ℕ := 5
def num_female : ℕ := 3

-- Define the probability distribution for female teachers
noncomputable def female_prob (n : ℕ) : ℝ :=
  if n = 1 ∨ n = 2 then 1/2 else 0

-- Define the probability distribution for male teachers
noncomputable def male_prob (n : ℕ) : ℝ :=
  if n = 2 ∨ n = 3 then 1/2 else 0

-- Define the score per activity
def score_per_activity : ℝ := 10

-- Define the total number of teachers
def total_teachers : ℕ := num_male + num_female

-- Define the number of teachers selected
def selected_teachers : ℕ := 2

-- Theorem 1: Probability of exactly one female given at least one female
theorem prob_one_female_given_female_participation : ℝ := 5/6

-- Theorem 2: Relationship between sum of scores and number of female participants
theorem score_female_relationship (X Y : ℝ) : Prop := X = 50 - 10 * Y

-- Theorem 3: Expected value of the sum of scores
theorem expected_sum_of_scores : ℝ := 42.5

-- Proofs
lemma prob_one_female_given_female_participation_proof : 
  prob_one_female_given_female_participation = 5/6 := by sorry

lemma score_female_relationship_proof (X Y : ℝ) : 
  score_female_relationship X Y ↔ X = 50 - 10 * Y := by sorry

lemma expected_sum_of_scores_proof : 
  expected_sum_of_scores = 42.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_female_given_female_participation_score_female_relationship_expected_sum_of_scores_l543_54370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_YPQ_l543_54366

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angle XYZ
noncomputable def angle_XYZ (t : Triangle) : ℝ := 50 * Real.pi / 180

-- Define the lengths of XY and XZ
def XY_length : ℝ := 8
def XZ_length : ℝ := 5

-- Define points P and Q
noncomputable def P (t : Triangle) : ℝ × ℝ := sorry
noncomputable def Q (t : Triangle) : ℝ × ℝ := sorry

-- Define the sum YP + PQ + QZ
noncomputable def sum_YPQ (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem min_sum_YPQ (t : Triangle) :
  ∃ (min_sum : ℝ), 
    (∀ (p q : ℝ × ℝ), sum_YPQ t ≥ min_sum) ∧ 
    (min_sum = XZ_length / Real.sin (angle_XYZ t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_YPQ_l543_54366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l543_54309

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (x + 1) / 2 + 7

-- State the theorem
theorem h_fixed_point :
  ∃! x : ℝ, h x = x ∧ ∀ y : ℝ, h (4 * y - 1) = 2 * y + 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_fixed_point_l543_54309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_scalar_multiple_l543_54329

theorem vector_scalar_multiple (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![2, -4]
  (∃ lambda : ℝ, a = lambda • b) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_scalar_multiple_l543_54329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l543_54357

theorem calculation_proof :
  ((Real.sqrt 12 + Real.sqrt (4/3)) * Real.sqrt 3 = 8) ∧
  (Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6) :=
by
  constructor
  · sorry  -- Proof for the first part
  · sorry  -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l543_54357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l543_54301

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a, b > 0),
    if a line with slope 1 passing through the left vertex A(-a, 0)
    intersects the right branch at point B, and the projection of B
    on the x-axis is the right focus F(c, 0), then the eccentricity
    of the hyperbola is 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x ↦ b * Real.sqrt ((x^2 / a^2) - 1)
  let A : ℝ × ℝ := (-a, 0)
  let F : ℝ × ℝ := (c, 0)
  let B : ℝ × ℝ := (c, f c)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → (y - A.2) = (x - A.1)) →
  B.1 = F.1 →
  c / a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l543_54301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l543_54340

noncomputable def f (k x : ℝ) : ℝ := Real.log ((k^2 - 1) * x^2 - (k + 1) * x + 1)

theorem range_of_k (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f k x = y) ↔ k ∈ Set.Icc 1 (5/3) := by
  sorry

#check range_of_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l543_54340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l543_54339

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, x ∈ Set.Icc 0 1 → f x ≥ 0)
  (h2 : f 1 = 1)
  (h3 : ∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂) :
  ∀ x, x ∈ Set.Icc 0 1 → f x ≤ 2 * x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l543_54339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l543_54325

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given polar coordinates -/
noncomputable def given_polar : ℝ × ℝ := (12, 5 * Real.pi / 4)

/-- The expected rectangular coordinates -/
noncomputable def expected_rectangular : ℝ × ℝ := (-6 * Real.sqrt 2, -6 * Real.sqrt 2)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular given_polar.1 given_polar.2 = expected_rectangular := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l543_54325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_about_direct_proof_l543_54302

-- Define the statements
def foundational_assumptions_no_proof : Prop :=
  ∃ x, x = "Foundational assumptions in mathematics do not necessitate proof"

def proofs_various_sequences : Prop :=
  ∃ x, x = "Different mathematical proofs can be structured in various valid sequences"

def variables_must_be_defined : Prop :=
  ∃ x, x = "All variables and expressions in a mathematical proof must be explicitly defined beforehand"

def false_premises_invalid_conclusion : Prop :=
  ∃ x, x = "A mathematical proof cannot have a logically valid conclusion if its premises include falsehoods"

def direct_proof_conflicting_premises : Prop :=
  ∃ x, x = "A direct proof method is applicable every time there are conflicting premises"

-- Define the theorem
theorem incorrect_statement_about_direct_proof : 
  foundational_assumptions_no_proof ∧
  proofs_various_sequences ∧
  variables_must_be_defined ∧
  false_premises_invalid_conclusion →
  ¬direct_proof_conflicting_premises :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_about_direct_proof_l543_54302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_bound_l543_54390

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x^2) * Real.exp x

-- State the theorem
theorem smallest_a_bound (a : ℝ) : 
  (∀ x ≥ 0, f x ≤ a * x + 1) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_bound_l543_54390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l543_54396

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) ≤ 0
def solution_set_f_leq_0 : Set ℝ := {x | f x ≤ 0}

-- Define the solution set of f(log x) > 0
def solution_set_f_log_gt_0 : Set ℝ := {x | f (Real.log x) > 0}

-- State the theorem
theorem solution_set_equivalence :
  solution_set_f_leq_0 = Set.Icc (-1) 2 →
  solution_set_f_log_gt_0 = (Set.Ioo 0 (1/10) ∪ Set.Ioi 100) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l543_54396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_subset_condition_l543_54353

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 + x) + Real.log (4 - x)

-- Define the domain A
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- Define the set B
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Theorem statement
theorem domain_and_subset_condition :
  (∀ x, x ∈ A ↔ -2 ≤ x ∧ x < 4) ∧
  (∀ m, B m ⊂ A ↔ m < 5/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_subset_condition_l543_54353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l543_54388

/-- The magnitude of the sum of two vectors (2,2) and (-2,3) is 5 -/
theorem vector_sum_magnitude : 
  let F₁ : Fin 2 → ℝ := ![2, 2]
  let F₂ : Fin 2 → ℝ := ![-2, 3]
  ‖F₁ + F₂‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l543_54388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l543_54352

noncomputable def sequence_a (c : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => sequence_a c n + c

noncomputable def sequence_b (c : ℝ) (n : ℕ) : ℝ :=
  1 / (sequence_a c n * sequence_a c (n + 1))

noncomputable def sum_b (c : ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i => sequence_b c (i + 1))

theorem sequence_properties (c : ℝ) :
  (∃ r : ℝ, r ≠ 1 ∧ 
    sequence_a c 2 = sequence_a c 1 * r ∧ 
    sequence_a c 5 = sequence_a c 2 * r) →
  c = 2 ∧ ∀ n : ℕ, n > 0 → sum_b c n = n / (2 * n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l543_54352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l543_54356

/-- Curve C is defined by x = 2pt^2 and y = 2pt, where t is the parameter -/
def curve_C (p : ℝ) (t : ℝ) : ℝ × ℝ := (2 * p * t^2, 2 * p * t)

/-- The length of a segment between two points -/
noncomputable def segment_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem length_of_segment_AB (p : ℝ) (t₁ t₂ : ℝ) (h : t₁ + t₂ ≠ 0) :
  segment_length (curve_C p t₁) (curve_C p t₂) = |2 * p * (t₁ - t₂)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l543_54356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l543_54393

-- Define the cone's properties
noncomputable def sector_radius : ℝ := 6
noncomputable def central_angle_degrees : ℝ := 120

-- Theorem statement
theorem cone_surface_area :
  let central_angle_radians : ℝ := central_angle_degrees * Real.pi / 180
  let lateral_area : ℝ := (1 / 2) * sector_radius^2 * central_angle_radians
  let base_radius : ℝ := sector_radius * central_angle_radians / (2 * Real.pi)
  let base_area : ℝ := Real.pi * base_radius^2
  let total_area : ℝ := lateral_area + base_area
  total_area = 16 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l543_54393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_theorem_l543_54346

theorem merchant_discount_theorem (cost markup_percent profit_percent : ℝ)
  (h1 : markup_percent = 75)
  (h2 : profit_percent = 5)
  (h3 : cost > 0) :
  let marked_price := cost * (1 + markup_percent / 100)
  let selling_price := cost * (1 + profit_percent / 100)
  let discount_percent := (marked_price - selling_price) / marked_price * 100
  discount_percent = 40 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_theorem_l543_54346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_excircle_distance_l543_54323

noncomputable def center_incircle (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

noncomputable def center_excircle (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

theorem incircle_excircle_distance (P Q R : EuclideanSpace ℝ (Fin 2)) 
  (h_PQ : dist P Q = 17)
  (h_PR : dist P R = 15)
  (h_QR : dist Q R = 8) :
  let s := (dist P Q + dist P R + dist Q R) / 2
  let K := Real.sqrt (s * (s - dist Q R) * (s - dist P R) * (s - dist P Q))
  let r := K / s
  let r' := K / (s - dist Q R)
  let I := center_incircle P Q R
  let E := center_excircle P Q R
  dist I E = 5 * Real.sqrt 17 - 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_excircle_distance_l543_54323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sequence_formula_l543_54392

def sequenceTerms (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | k + 1 => sequenceTerms k + 3 * (k + 1)

def sumSequence (n : ℕ) : ℕ :=
  (List.range n).map sequenceTerms |>.sum

theorem sum_sequence_formula (n : ℕ) :
  sumSequence n = n * (n + 1) * (n + 1) / 4 + 2 * n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sequence_formula_l543_54392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_blue_ball_after_removal_l543_54374

/-- The probability of pulling a blue ball after removing some balls from a jar. -/
theorem probability_blue_ball_after_removal 
  (initial_total : ℕ) 
  (initial_blue : ℕ) 
  (removed : ℕ) 
  (h1 : initial_total = 15)
  (h2 : initial_blue = 7)
  (h3 : removed = 3)
  (h4 : removed ≤ initial_blue)
  (h5 : removed < initial_total) :
  (initial_blue - removed : ℚ) / (initial_total - removed : ℚ) = 1/3 := by
  sorry

#check probability_blue_ball_after_removal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_blue_ball_after_removal_l543_54374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l543_54373

theorem circle_radius (A C : ℝ) (h : A + (1/2) * C = 56 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l543_54373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l543_54348

/-- 
Proves that the point (-3, π/4) in polar coordinates is equivalent to (3, 5π/4) 
in standard polar coordinate representation.
-/
theorem polar_coordinate_equivalence :
  let r₁ : ℝ := -3
  let θ₁ : ℝ := π / 4
  let r₂ : ℝ := 3
  let θ₂ : ℝ := 5 * π / 4
  (r₁ * Real.cos θ₁ = r₂ * Real.cos θ₂ ∧ r₁ * Real.sin θ₁ = r₂ * Real.sin θ₂) ∧
  r₂ > 0 ∧ 0 ≤ θ₂ ∧ θ₂ < 2 * π :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l543_54348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_income_l543_54384

noncomputable def tax_rate (p : ℝ) (income : ℝ) : ℝ :=
  if income ≤ 35000 then
    0.01 * p * income
  else if income ≤ 55000 then
    0.01 * p * 35000 + 0.01 * (p + 3) * (income - 35000)
  else
    0.01 * p * 35000 + 0.01 * (p + 3) * 20000 + 0.01 * (p + 5) * (income - 55000)

theorem laura_income (p : ℝ) :
  ∃ income : ℝ, income > 0 ∧ tax_rate p income = (0.01 * p + 0.0045) * income ∧ income = 75000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_income_l543_54384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_one_l543_54342

theorem x_equals_one (x y : ℕ) 
  (hx : x > 0) (hy : y > 0)
  (h : ∀ n : ℕ, n > 0 → (x ^ (2 ^ n) - 1) % (2 ^ n * y + 1) = 0) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_one_l543_54342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l543_54391

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) + (Real.cos x) ^ 2 - 1 / 2

theorem f_properties :
  -- Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ π) ∧
  -- Maximum value in [0, π/2] is 1
  (∀ x, x ∈ Set.Icc 0 (π / 2) → f x ≤ 1) ∧
  (∃ x, x ∈ Set.Icc 0 (π / 2) ∧ f x = 1) ∧
  -- If f(x₀) = 1/3 and x₀ ∈ [π/6, 5π/12], then sin(2x₀) = (√3 - 2√2) / 6
  (∀ x₀, x₀ ∈ Set.Icc (π / 6) (5 * π / 12) → f x₀ = 1 / 3 →
    Real.sin (2 * x₀) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l543_54391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_to_subset_inclusion_l543_54383

universe u

variable {E : Type u} [PartialOrder E]

theorem order_to_subset_inclusion :
  ∃ (F : Type u) (X : E → Set F), Finite F ∧
    ∀ (e₁ e₂ : E), e₁ ≤ e₂ ↔ X e₂ ⊆ X e₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_to_subset_inclusion_l543_54383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_sqrt3_cos_range_l543_54332

theorem sin_plus_sqrt3_cos_range (θ : ℝ) (h : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
  1 < Real.sin θ + Real.sqrt 3 * Real.cos θ ∧ Real.sin θ + Real.sqrt 3 * Real.cos θ ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_sqrt3_cos_range_l543_54332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_range_l543_54369

/-- Given two points A(2, 2) and B(m, 0) in a Cartesian coordinate system,
    if there are exactly two lines that are at a distance of 1 from A
    and at a distance of 3 from B, then m is in the range (2-2√3, 2) ∪ (2, 2+2√3) -/
theorem line_distance_range (m : ℝ) :
  (∃! (l₁ l₂ : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ l₁ → ((p.1 - 2)^2 + (p.2 - 2)^2).sqrt = 1) ∧
    (∀ (p : ℝ × ℝ), p ∈ l₁ → ((p.1 - m)^2 + p.2^2).sqrt = 3) ∧
    (∀ (p : ℝ × ℝ), p ∈ l₂ → ((p.1 - 2)^2 + (p.2 - 2)^2).sqrt = 1) ∧
    (∀ (p : ℝ × ℝ), p ∈ l₂ → ((p.1 - m)^2 + p.2^2).sqrt = 3) ∧
    l₁ ≠ l₂) →
  m ∈ Set.Ioo (2 - 2 * Real.sqrt 3) 2 ∪ Set.Ioo 2 (2 + 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_range_l543_54369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l543_54330

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ := (abs (f x) - f x) / 2

theorem three_intersection_points (a b : ℝ) :
  (a > 0) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    g x₁ = a * x₁ + b ∧
    g x₂ = a * x₂ + b ∧
    g x₃ = a * x₃ + b) ↔
  (2 * a < b ∧ b < (1/4) * (a + 1)^2 + 2 ∧ 0 < a ∧ a < 3) :=
by
  sorry

#check three_intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l543_54330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coterminal_angle_negative_950_l543_54313

/-- The angle (in degrees) that is coterminal with the given angle -/
noncomputable def coterminalAngle (angle : ℝ) : ℝ :=
  angle - 360 * ⌊angle / 360⌋

/-- 
Theorem: The angle in the range [0°, 180°] with the same terminal side as -950° is 130°.
-/
theorem coterminal_angle_negative_950 :
  coterminalAngle (-950) ∈ Set.Icc 0 180 ∧ coterminalAngle (-950) = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coterminal_angle_negative_950_l543_54313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l543_54387

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ x₂ ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ 
    f x₁ = 1/4 ∧ f x₂ = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l543_54387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_properties_l543_54398

/-- Represents the properties of a ball bounce scenario -/
structure BallBounce where
  initial_height : ℝ
  slope_angle : ℝ
  gravity : ℝ

/-- Calculates the final velocity before impact -/
noncomputable def final_velocity (b : BallBounce) : ℝ :=
  Real.sqrt (2 * b.gravity * b.initial_height)

/-- Calculates the vertical component of the rebound velocity -/
noncomputable def vertical_velocity (b : BallBounce) (v : ℝ) : ℝ :=
  v * Real.sin b.slope_angle

/-- Calculates the maximum height reached after bounce -/
noncomputable def max_height (b : BallBounce) (vy : ℝ) : ℝ :=
  vy^2 / (2 * b.gravity)

/-- Calculates the horizontal component of the rebound velocity -/
noncomputable def horizontal_velocity (b : BallBounce) (v : ℝ) : ℝ :=
  v * Real.cos b.slope_angle

/-- Calculates the time the ball is in the air -/
noncomputable def air_time (b : BallBounce) (vy : ℝ) : ℝ :=
  2 * vy / b.gravity

/-- Calculates the horizontal distance traveled -/
def horizontal_distance (vx t : ℝ) : ℝ :=
  vx * t

/-- Theorem stating the properties of the ball bounce -/
theorem ball_bounce_properties (b : BallBounce)
    (h1 : b.initial_height = 10.9)
    (h2 : b.slope_angle = 30 * Real.pi / 180)
    (h3 : b.gravity = 9.8) :
    ∃ (h d : ℝ), abs (h - 2.725) < 0.001 ∧ abs (d - 18.879) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_properties_l543_54398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_cube_sum_of_20_consecutive_integers_l543_54326

def isPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem smallest_perfect_cube_sum_of_20_consecutive_integers : 
  (∃ (n : ℕ), 1000 = (20 * n + 190)) ∧ 
  (∀ (m : ℕ), m < 1000 → isPerfectCube m → ¬∃ (k : ℕ), m = (20 * k + 190)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_cube_sum_of_20_consecutive_integers_l543_54326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equality_l543_54379

theorem cubic_root_equality (m : ℝ) : (243 : ℝ) ^ (1/3 : ℝ) = 3 ^ m → m = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equality_l543_54379
