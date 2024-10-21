import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l58_5886

/-- Represents the time it takes for a tap to fill the cistern -/
structure FillTime where
  minutes : ℚ
  positive : minutes > 0

/-- Represents the portion of the cistern filled -/
def portion_filled (time : FillTime) (duration : ℚ) : ℚ :=
  duration / time.minutes

theorem cistern_fill_time 
  (tap_a tap_b : FillTime) 
  (ha : tap_a.minutes = 45) 
  (hb : portion_filled tap_a 9 + portion_filled tap_b 23 = 1) : 
  tap_b.minutes = 115 / 4 := by
  sorry

#eval (115 : ℚ) / 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l58_5886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_percentage_l58_5847

-- Define the radio types
structure Radio where
  costPrice : ℕ
  salePrice : ℕ

-- Define the problem parameters
def radioA : Radio := { costPrice := 2400, salePrice := 2100 }
def radioB : Radio := { costPrice := 3500, salePrice := 3300 }
def radioC : Radio := { costPrice := 5000, salePrice := 4800 }

-- Calculate total cost price
def totalCostPrice : ℕ := radioA.costPrice + radioB.costPrice + radioC.costPrice

-- Calculate total sale price
def totalSalePrice : ℕ := radioA.salePrice + radioB.salePrice + radioC.salePrice

-- Calculate total loss
def totalLoss : ℕ := totalCostPrice - totalSalePrice

-- Define the loss percentage calculation
noncomputable def lossPercentage : ℝ := (totalLoss : ℝ) / (totalCostPrice : ℝ) * 100

-- Theorem to prove
theorem overall_loss_percentage :
  abs (lossPercentage - 6.42) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_percentage_l58_5847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_30_minus_tan_45_l58_5852

theorem sin_30_minus_tan_45 : 
  Real.sin (π / 6) - Real.tan (π / 4) = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_30_minus_tan_45_l58_5852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_jumps_l58_5875

/-- Represents the number of jump rope repetitions on a given day -/
def jumpRope : ℕ → ℕ
  | 0 => 15  -- Define the base case for day 0 (or day 1 in the problem)
  | n + 1 => 2 * jumpRope n

theorem fourth_day_jumps : jumpRope 3 = 120 := by
  -- Expand the definition of jumpRope for days 0, 1, 2, and 3
  have h0 : jumpRope 0 = 15 := rfl
  have h1 : jumpRope 1 = 30 := by simp [jumpRope]
  have h2 : jumpRope 2 = 60 := by simp [jumpRope, h1]
  have h3 : jumpRope 3 = 120 := by simp [jumpRope, h2]
  -- The final result
  exact h3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_jumps_l58_5875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_linear_combination_l58_5814

/-- Given two vectors e₁ and e₂ in ℝ², and a vector a in ℝ², 
    prove that there exists a unique pair of real numbers (λ₁, λ₂) 
    such that a = λ₁ * e₁ + λ₂ * e₂, and this pair is (-1, 1). -/
theorem vector_linear_combination (e₁ e₂ a : ℝ × ℝ) 
  (h₁ : e₁ = (2, 1)) 
  (h₂ : e₂ = (1, 3)) 
  (h₃ : a = (-1, 2)) : 
  ∃! p : ℝ × ℝ, a = (p.1 * e₁.1 + p.2 * e₂.1, p.1 * e₁.2 + p.2 * e₂.2) ∧ 
                   p.1 = -1 ∧ p.2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_linear_combination_l58_5814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_journey_time_l58_5843

/-- Represents the journey of a cyclist with a speed increase halfway through. -/
structure CyclistJourney where
  S : ℝ  -- Total distance
  v : ℝ  -- Initial speed
  planned_time : ℝ  -- Planned time for the journey
  speed_increase : ℝ  -- Factor by which speed increases

/-- The actual time taken for the journey given the cyclist's plan and speed increase. -/
noncomputable def actual_time (j : CyclistJourney) : ℝ :=
  (j.S / (2 * j.v)) + (j.S / (2 * j.v * (1 + j.speed_increase)))

/-- Theorem stating that under the given conditions, the actual time taken is 4.5 hours. -/
theorem cyclist_journey_time (j : CyclistJourney) 
    (h1 : j.planned_time = 5)
    (h2 : j.speed_increase = 0.25)
    (h3 : j.S = j.v * j.planned_time) : 
  actual_time j = 4.5 := by
  sorry

#check cyclist_journey_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_journey_time_l58_5843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_max_angle_line_trajectory_equation_l58_5864

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define a point P inside the circle
def point_inside_circle (x₀ y₀ : ℝ) : Prop := 
  circle_C x₀ y₀ ∧ (x₀ - 2)^2 + (y₀ - 2)^2 < 8

-- Define the chord AB
def chord (x₀ y₀ x y : ℝ) : Prop :=
  point_inside_circle x₀ y₀ ∧ 
  ∃ t : ℝ, x = t * x₀ + (1 - t) * x ∧ y = t * y₀ + (1 - t) * y ∧
  circle_C x y

-- Theorem 1: Equation of chord AB when P(1, 1) and |AB| = 2√7
theorem chord_equation (x y : ℝ) :
  chord 1 1 x y ∧ (x - 1)^2 + (y - 1)^2 = 7 → x = 1 ∨ y = 1 := 
by sorry

-- Theorem 2: Equation of line AP when angle PAC is maximized and P(1, 1)
theorem max_angle_line (x y : ℝ) :
  point_inside_circle 1 1 →
  (∀ x' y', point_inside_circle x' y' → 
    (x' - 1) * (x - 1) + (y' - 1) * (y - 1) ≤ (x - 1)^2 + (y - 1)^2) →
  y = -x + 2 :=
by sorry

-- Theorem 3: Trajectory equation for point M
theorem trajectory_equation (x₀ y₀ x' y' : ℝ) :
  point_inside_circle x₀ y₀ →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    chord x₀ y₀ x₁ y₁ ∧ chord x₀ y₀ x₂ y₂ ∧
    (x₁ - 2) * (x' - 2) + (y₁ - 2) * (y' - 2) = 8 ∧
    (x₂ - 2) * (x' - 2) + (y₂ - 2) * (y' - 2) = 8) →
  (x₀ - 2) * (x' - 2) + (y₀ - 2) * (y' - 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_max_angle_line_trajectory_equation_l58_5864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l58_5869

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x - 2 * Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem range_of_f :
  ∀ x ∈ Set.Icc (π / 4) (π / 2), -1 ≤ f x ∧ f x ≤ Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l58_5869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l58_5876

/-- The parabola y^2 = -4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -4 * p.1}

/-- The focus of the parabola y^2 = -4x -/
noncomputable def Focus : ℝ × ℝ := (1/4, 0)

/-- Point A -/
def A : ℝ × ℝ := (-2, 1)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The sum of distances from a point to A and to the focus -/
noncomputable def distanceSum (p : ℝ × ℝ) : ℝ :=
  distance p A + distance p Focus

theorem parabola_min_distance :
  ∀ p ∈ Parabola, distanceSum p ≥ distanceSum (-1/4, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l58_5876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_l58_5880

noncomputable section

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) : Prop := ellipse a b (Real.sqrt 3) (1/2)

-- Define the line l
def line_l (x y : ℝ) : Prop := y - 1/2 = -1/4 * (x - 1/2)

-- Define the midpoint of AB
def midpoint_AB : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  ellipse 2 1 x₁ y₁ ∧ ellipse 2 1 x₂ y₂ ∧ 
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  (x₁ + x₂) / 2 = 1/2 ∧ (y₁ + y₂) / 2 = 1/2

-- Main theorem
theorem ellipse_and_triangle : 
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  point_on_ellipse a b →
  eccentricity a b = Real.sqrt 3 / 2 →
  midpoint_AB →
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ (S : ℝ), S = 25 / 32 ∧ 
    S = (1/2) * (5/8) * (5/2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_l58_5880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_implies_expression_half_l58_5810

theorem tan_two_implies_expression_half (θ : Real) (h : Real.tan θ = 2) :
  (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_implies_expression_half_l58_5810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_displacement_l58_5838

/-- A segment on a plane -/
structure Segment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- The trace of a point's movement -/
structure Trace where
  path : ℝ → ℝ × ℝ

/-- Movement of a segment parallel to a line -/
def ParallelMovement (s : Segment) (t : ℝ → Segment) : Prop :=
  ∀ x, (t x).start.2 - (t x).endpoint.2 = s.start.2 - s.endpoint.2

/-- Non-intersecting traces -/
def NonIntersectingTraces (traceA traceB : Trace) : Prop :=
  ∀ x y, traceA.path x ≠ traceB.path y

/-- Theorem: Unbounded displacement of point A -/
theorem unbounded_displacement
  (s : Segment)
  (t : ℝ → Segment)
  (traceA traceB : Trace)
  (h_unit : (s.endpoint.1 - s.start.1)^2 + (s.endpoint.2 - s.start.2)^2 = 1)
  (h_parallel : ParallelMovement s t)
  (h_nonintersect : NonIntersectingTraces traceA traceB)
  (h_return : ∃ x, (t x).start.2 = s.start.2 ∧ (t x).endpoint.2 = s.endpoint.2) :
  ∀ d, ∃ x, |((t x).start.1 - s.start.1)| > d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_displacement_l58_5838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l58_5892

-- Part 1
theorem sin_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by sorry

-- Part 2
noncomputable def f (a x : ℝ) := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, x ≠ 0 → f a x < f a 0) ↔ 
  a < -Real.sqrt 2 ∨ a > Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l58_5892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l58_5898

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 7
noncomputable def b : ℝ := Real.log 7 / Real.log (1/3)
noncomputable def c : ℝ := 3 ^ (7/10)

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l58_5898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l58_5879

def A : Set ℝ := {x | (3 - x) / (x + 1) ≥ 2}

theorem complement_of_A : Set.compl A = Set.Iic (-1) ∪ Set.Ioi (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l58_5879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l58_5801

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  4 * (Real.cos (t.A / 2))^2 - Real.cos (2 * (t.B + t.C)) = 7/2 ∧
  t.a = 2

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem max_triangle_area (t : Triangle) :
  triangle_conditions t →
  ∃ (max_area : ℝ), 
    (∀ (t' : Triangle), triangle_conditions t' → triangle_area t' ≤ max_area) ∧
    max_area = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l58_5801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_wrap_theorem_l58_5893

/-- The number of times a rope can wrap around a cylinder -/
noncomputable def wrap_count (rope_length : ℝ) (cylinder_radius : ℝ) : ℝ :=
  rope_length / (2 * Real.pi * cylinder_radius)

/-- Theorem: A rope that wraps 49 times around a cylinder with radius 20 cm
    will wrap 70 times around a cylinder with radius 14 cm -/
theorem rope_wrap_theorem :
  let rope_length := 49 * (2 * Real.pi * 20)
  wrap_count rope_length 14 = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_wrap_theorem_l58_5893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_set_2_3_4_sum_equals_262_l58_5873

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped -/
noncomputable def volume_set (l w h : ℝ) : ℝ :=
  let parallelepiped := l * w * h
  let external_parallelepipeds := 2 * (l * w + l * h + w * h)
  let spheres := 8 * (1 / 8 * (4 / 3) * Real.pi)
  let cylinders := Real.pi * (l + w + h)
  parallelepiped + external_parallelepipeds + spheres + cylinders

/-- Theorem stating the volume of the set for a 2x3x4 parallelepiped -/
theorem volume_set_2_3_4 :
  volume_set 2 3 4 = (228 + 31 * Real.pi) / 3 := by sorry

/-- The sum of m, n, and p in the problem -/
def m_plus_n_plus_p : ℕ := 262

/-- Theorem stating that m + n + p = 262 -/
theorem sum_equals_262 : m_plus_n_plus_p = 262 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_set_2_3_4_sum_equals_262_l58_5873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l58_5858

-- Define the spinner with four regions
structure Spinner where
  A : Set ℝ
  B : Set ℝ
  C : Set ℝ
  D : Set ℝ

-- Define the probability measure
variable (P : Set ℝ → ℝ)

-- State the theorem
theorem spinner_probability (s : Spinner) 
  (hP : P (s.A ∪ s.B ∪ s.C ∪ s.D) = 1) 
  (hA : P s.A = 1/4) (hB : P s.B = 1/3) 
  (hDisjoint : Disjoint s.A s.B ∧ Disjoint s.A s.C ∧ Disjoint s.A s.D ∧ 
               Disjoint s.B s.C ∧ Disjoint s.B s.D ∧ Disjoint s.C s.D) :
  P (s.C ∪ s.D) = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l58_5858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_square_to_right_angle_l58_5848

/-- Represents a right triangle as two sets of points in ℝ² -/
def IsRightTriangle : (Set ℝ × Set ℝ) → Prop := sorry

/-- Returns the lengths of the legs of a right triangle -/
def LegLengths : (Set ℝ × Set ℝ) → ℝ × ℝ := sorry

/-- Represents a square constructed externally on the hypotenuse of a triangle -/
def IsSquareOnHypotenuse : (Set ℝ × Set ℝ) → (Set ℝ × Set ℝ) → Prop := sorry

/-- Calculates the distance from the center of the square to the right angle vertex of the triangle -/
noncomputable def DistanceCenterToRightAngle : (Set ℝ × Set ℝ) → (Set ℝ × Set ℝ) → ℝ := sorry

/-- Given a right triangle with legs 3 and 5, and a square constructed externally on its hypotenuse,
    the distance from the center of the square to the vertex of the right angle is 4√2. -/
theorem distance_center_square_to_right_angle (triangle : Set ℝ × Set ℝ) 
  (h1 : IsRightTriangle triangle)
  (h2 : LegLengths triangle = (3, 5))
  (square : Set ℝ × Set ℝ)
  (h3 : IsSquareOnHypotenuse triangle square) :
  DistanceCenterToRightAngle triangle square = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_square_to_right_angle_l58_5848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_part_value_l58_5821

noncomputable def total_amount : ℝ := 8154

noncomputable def ratios : List ℝ := [5/11, 7/15, 11/19, 2/9, 17/23, 13/29, 19/31]

noncomputable def sum_of_ratios : ℝ := ratios.sum

noncomputable def proportion_of_fifth_part : ℝ := (17/23) / sum_of_ratios

noncomputable def fifth_part : ℝ := total_amount * proportion_of_fifth_part

theorem fifth_part_value :
  ∃ ε > 0, |fifth_part - 1710.05| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_part_value_l58_5821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l58_5870

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- State the theorem
theorem root_in_interval :
  ∃! x, x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l58_5870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l58_5802

theorem polynomial_factorization (p : ℕ) (hp : Prime p) (n : ℕ) :
  (∃ (P₁ P₂ : Polynomial ℤ), (X^4 - 2*(n+p : ℤ)*X^2 + (n-p : ℤ)^2 : Polynomial ℤ) = P₁ * P₂ ∧ 
    Polynomial.degree P₁ = 2 ∧ Polynomial.degree P₂ = 2) ↔ 
  (∃ (k : ℕ), n = k^2 ∨ n = p * k^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l58_5802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_length_calculation_l58_5816

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Represents the dimensions of a wall in meters -/
structure WallDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the length of a brick given wall dimensions, number of bricks, and brick width and height -/
def calculateBrickLength (wall : WallDimensions) (numBricks : ℕ) (brickWidth : ℚ) (brickHeight : ℚ) : ℚ :=
  (wall.length * wall.width * wall.height * 1000000) / (numBricks * brickWidth * brickHeight)

theorem brick_length_calculation (wall : WallDimensions) (numBricks : ℕ) (brickWidth brickHeight : ℚ) :
  wall.length = 10 ∧ 
  wall.width = 8 ∧ 
  wall.height = 24.5 ∧
  numBricks = 12250 ∧
  brickWidth = 10 ∧
  brickHeight = 8 →
  calculateBrickLength wall numBricks brickWidth brickHeight = 2000 := by
  sorry

#eval calculateBrickLength ⟨10, 8, 24.5⟩ 12250 10 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_length_calculation_l58_5816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l58_5809

-- Define the function f using a piecewise definition
noncomputable def f (x : ℝ) : ℝ :=
  if x = -2 then 9
  else if x = -1 then 2
  else if x = 0 then 0
  else if x = 1 then 2
  else if x = 2 then 9
  else 0  -- Default case for other values

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem statement
theorem f_increasing_interval :
  IncreasingOn f 0 2 ∧ 
  ¬IncreasingOn f (-2) 0 ∧
  ¬IncreasingOn f (-2) (-1) ∧
  ¬IncreasingOn f (-1) 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l58_5809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_generating_sequence_l58_5885

theorem existence_of_prime_generating_sequence : ∃ (n : ℕ) (c : Fin n → ℕ),
  (∃ (a₁ a₂ : ℕ), a₁ ≠ a₂ ∧
    (∀ i : Fin n, Nat.Prime (a₁ + c i)) ∧
    (∀ i : Fin n, Nat.Prime (a₂ + c i))) ∧
  (∃ (m : ℕ), ∀ (a : ℕ),
    (∀ i : Fin n, Nat.Prime (a + c i)) →
    (a ∈ Finset.range m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_generating_sequence_l58_5885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_filling_time_l58_5859

/-- Represents the time it takes for Pipe A to fill the pool alone -/
noncomputable def t : ℝ := sorry

/-- Pipe A's filling rate (portion of pool filled per hour) -/
noncomputable def rate_A : ℝ := 1 / t

/-- Pipe B's filling rate (portion of pool filled per hour) -/
noncomputable def rate_B : ℝ := rate_A / 3

/-- The portion of the pool filled by Pipe A in 8 hours -/
noncomputable def portion_filled_A : ℝ := 8 * rate_A

/-- The portion of the pool left to be filled by Pipe B -/
noncomputable def portion_left_B : ℝ := 1 - portion_filled_A

/-- Time it takes for Pipe B to fill its portion -/
def time_B : ℝ := 12

theorem pipe_filling_time :
  portion_left_B = time_B * rate_B ∧ t = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_filling_time_l58_5859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l58_5884

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x ≤ 0 then 3 * x^2 - x + 2 * a + 1
  else 3 * (-x)^2 - (-x) + 2 * a + 1

-- State the theorem
theorem find_a :
  ∃ a : ℝ, (∀ x : ℝ, f a x = f a (-x)) ∧ f a 2 = 13 ∧ a = -1 := by
  -- Proof goes here
  sorry

#check find_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l58_5884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_product_l58_5832

/-- Given an ellipse and a hyperbola with common foci and a common point,
    prove that the minimum product of their eccentricities is √3/2 when the angle
    formed by the foci and the common point is 60°. -/
theorem min_eccentricity_product (F₁ F₂ P : ℝ × ℝ) 
  (e : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ) -- eccentricity function for ellipse
  (h : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ) -- eccentricity function for hyperbola
  (angle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ) -- angle function
  (is_common_foci : F₁ ≠ F₂) -- F₁ and F₂ are distinct points
  (is_common_point : ∃ (c : ℝ), e F₁ F₂ P = c ∧ h F₁ F₂ P = c) -- P is on both ellipse and hyperbola
  (angle_condition : angle F₁ P F₂ = π / 3) -- ∠F₁PF₂ = 60°
  : ∀ (e₁ e₂ : ℝ), e₁ = e F₁ F₂ P → e₂ = h F₁ F₂ P → e₁ * e₂ ≥ Real.sqrt 3 / 2 :=
by
  sorry

#check min_eccentricity_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_product_l58_5832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_is_002_l58_5830

noncomputable def data : Fin 5 → ℝ
  | 0 => 10.1
  | 1 => 9.8
  | 2 => 10
  | 3 => 10.2
  | 4 => 9.8  -- This is 'x', calculated from the average condition

def average : ℝ := 10

noncomputable def variance (d : Fin 5 → ℝ) (avg : ℝ) : ℝ :=
  (1 : ℝ) / 5 * (Finset.univ.sum (fun i => (d i - avg) ^ 2))

theorem variance_of_data_is_002 :
  variance data average = 0.02 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_is_002_l58_5830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l58_5857

noncomputable def larger_circle_radius : ℝ := 10
noncomputable def smaller_circle_radius : ℝ := 5

noncomputable def larger_circle_area : ℝ := Real.pi * larger_circle_radius^2
noncomputable def smaller_circle_area : ℝ := Real.pi * smaller_circle_radius^2

noncomputable def shaded_area : ℝ := larger_circle_area - 2 * smaller_circle_area

theorem shaded_area_calculation : shaded_area = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l58_5857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_price_theorem_l58_5853

/-- The original price of an item before discounts -/
def original_price (p : ℝ) : Prop := p > 0

/-- The final price after applying two successive discounts -/
def final_price (p : ℝ) : ℝ := p * (1 - 0.25) * (1 - 0.20)

theorem saree_price_theorem (p : ℝ) :
  original_price p → final_price p = 240 → p = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_price_theorem_l58_5853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AP_l58_5813

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Check if a point is in a plane defined by three other points -/
def isInPlane (p a b c : Point3D) : Prop := sorry

/-- Main theorem: Minimum length of AP in the given rectangular prism -/
theorem min_length_AP (prism : RectangularPrism) 
  (h1 : distance prism.A prism.D = 1)
  (h2 : distance prism.A prism.B = 2)
  (h3 : distance prism.A prism.A₁ = 3)
  (P : Point3D)
  (h4 : isInPlane P prism.A₁ prism.B prism.D) :
  ∃ (min_length : ℝ), min_length = 6/7 ∧ 
  ∀ (Q : Point3D), isInPlane Q prism.A₁ prism.B prism.D → 
  distance prism.A Q ≥ min_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AP_l58_5813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_worked_three_shifts_l58_5866

/-- Calculates the number of 8-hour shifts worked by a waitress given her hourly wage, tip rate, average orders per hour, and total earnings for the week. -/
def calculate_shifts (hourly_wage : ℚ) (tip_rate : ℚ) (avg_orders_per_hour : ℚ) (total_earnings : ℚ) : ℕ :=
  let tips_per_hour := avg_orders_per_hour * tip_rate
  let total_per_hour := hourly_wage + tips_per_hour
  let total_hours := total_earnings / total_per_hour
  (total_hours / 8).floor.toNat

/-- Proves that given the specific conditions for Jill, she worked 3 eight-hour shifts. -/
theorem jill_worked_three_shifts :
  calculate_shifts 4 (15/100) 40 240 = 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_worked_three_shifts_l58_5866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_older_female_students_l58_5807

theorem percentage_of_older_female_students
  (total_students : ℝ)
  (h1 : total_students > 0)
  (male_ratio : ℝ)
  (h2 : male_ratio = 0.4)
  (older_male_ratio : ℝ)
  (h3 : older_male_ratio = 0.5)
  (younger_prob : ℝ)
  (h4 : younger_prob = 0.62) :
  let female_ratio := 1 - male_ratio
  let younger_male := male_ratio * (1 - older_male_ratio)
  let older_female_ratio := 1 - (younger_prob - younger_male) / female_ratio
  older_female_ratio = 0.3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_older_female_students_l58_5807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_value_l58_5849

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x + 1

theorem minimum_m_value :
  ∃ m : ℝ, m = -20/3 ∧
  (∀ a : ℝ, 1 ≤ a ∧ a ≤ 2 →
    ∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ 3 ≤ x₂ ∧ x₁ ≠ x₂ →
      (f a x₁ - f a x₂) / (x₂ - x₁) < m) ∧
  (∀ m' : ℝ, m' < m →
    ∃ a : ℝ, 1 ≤ a ∧ a ≤ 2 ∧
      ∃ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ 3 ≤ x₂ ∧ x₁ ≠ x₂ ∧
        (f a x₁ - f a x₂) / (x₂ - x₁) ≥ m') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_m_value_l58_5849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_ratio_l58_5897

/-- A right triangle with sides in ratio 5:4:3 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ratio_condition : a / b = 5 / 4 ∧ b / c = 4 / 3
  right_angle : a^2 = b^2 + c^2

/-- The incircle of a right triangle -/
noncomputable def incircle (t : RightTriangle) : ℝ := (t.b + t.c - t.a) / 2

/-- The segments created by the points of tangency -/
noncomputable def segments (t : RightTriangle) : ℝ × ℝ := (t.b - incircle t, t.c - incircle t)

/-- The theorem stating the ratio of segments -/
theorem segments_ratio (t : RightTriangle) : 
  let (seg1, seg2) := segments t
  seg2 / seg1 = 3 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segments_ratio_l58_5897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l58_5842

/-- Given complex numbers z₁ and z₂ defined in terms of a real parameter m -/
noncomputable def z₁ (m : ℝ) : ℂ := 2 * m^2 / (1 - Complex.I)
noncomputable def z₂ (m : ℝ) : ℂ := (2 + Complex.I) * m - 3 * (1 + 2 * Complex.I)

/-- Part I: If z₁ + z₂ is purely imaginary, then m = 1 -/
theorem part_one (m : ℝ) : Complex.im (z₁ m + z₂ m) ≠ 0 ∧ Complex.re (z₁ m + z₂ m) = 0 → m = 1 := by
  sorry

/-- Part II: If the real part of z₁ + z₂ is positive, then z₁ * z₂ = 20 - 12i -/
theorem part_two (m : ℝ) : Complex.re (z₁ m + z₂ m) > 0 → z₁ m * z₂ m = 20 - 12 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l58_5842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_to_neg_third_equals_two_l58_5871

theorem eighth_to_neg_third_equals_two : (1 / 8 : ℝ) ^ (-(1 / 3 : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_to_neg_third_equals_two_l58_5871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l58_5817

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x)

-- State the theorem about the domain of f
theorem f_domain : 
  ∀ x : ℝ, x < 1 ↔ ∃ y : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l58_5817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_is_two_l58_5820

theorem sin_double_angle_when_tan_is_two (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_is_two_l58_5820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_ninth_power_is_identity_l58_5889

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![Real.cos (π/4), 0, -Real.sin (π/4);
     0, 1, 0;
     Real.sin (π/4), 0, Real.cos (π/4)]

theorem B_ninth_power_is_identity :
  B^9 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_ninth_power_is_identity_l58_5889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotonicity_condition_l58_5863

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log (x + 1)

-- Part 1: Tangent line when a = 2
theorem tangent_line_at_zero :
  ∃ (m : ℝ), m = 2 ∧ (∀ x, m * x = (f 2 x - f 2 0) / (x - 0)) :=
sorry

-- Part 2: Monotonicity condition
theorem monotonicity_condition :
  ∀ a, (∀ x > 0, Monotone (λ y ↦ f a y - y)) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotonicity_condition_l58_5863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_order_l58_5899

theorem y_order : let y₁ : ℝ := (4 : ℝ)^(0.9 : ℝ)
                  let y₂ : ℝ := (8 : ℝ)^(0.44 : ℝ)
                  let y₃ : ℝ := ((1/2) : ℝ)^(-(1.5 : ℝ))
                  y₁ > y₃ ∧ y₃ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_order_l58_5899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_l58_5824

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = 1.96 ∧
    (∀ (y : ℝ), (⌊y⌋ : ℝ) = 2 + 50 * (y - ⌊y⌋) → x ≤ y) ∧
    (⌊x⌋ : ℝ) = 2 + 50 * (x - ⌊x⌋) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_l58_5824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l58_5823

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ (∃ x : ℝ, |x| > 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l58_5823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_x_v_u_l58_5881

noncomputable section

-- Define the interval (0.9, 1.0)
def I : Set ℝ := { x | 0.9 < x ∧ x < 1.0 }

-- Define the functions
noncomputable def u (x : ℝ) : ℝ := x^(x^x)
noncomputable def v (x : ℝ) : ℝ := x^(2*x)

-- State the theorem
theorem order_of_x_v_u : ∀ x ∈ I, x < v x ∧ v x < u x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_x_v_u_l58_5881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l58_5837

/-- The trajectory of point C in a triangle ABC, where A(-5,0) and B(5,0) are fixed,
    and the product of slopes of AC and BC is -1/2. -/
theorem trajectory_of_C (x y : ℝ) (h : x ≠ 5 ∧ x ≠ -5) : 
  (y / (x + 5)) * (y / (x - 5)) = -1/2 ↔ x^2/25 + y^2/(25/2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l58_5837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l58_5868

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given conditions and functions
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

def angleBisector (t : Triangle) (v : Point) : Prop := sorry

def planeAngle (t : Triangle) (plane : Point → Prop) : ℝ := sorry

def lineAngle (p1 p2 : Point) (plane : Point → Prop) : ℝ := sorry

-- State the theorem
theorem triangle_construction_theorem 
  (A D : Point) (p q : ℝ) : 
  ∃ (B C : Point), 
    let t := Triangle.mk A B C
    angleBisector t A ∧ 
    distance A B - distance B D = p ∧
    distance A C - distance C D = q ∧
    planeAngle t (λ _ => True) = lineAngle A D (λ _ => True) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l58_5868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_products_min_sum_of_products_equality_l58_5831

/-- A type representing a table with 24 rows and 8 columns -/
def Table := Fin 24 → Fin 8 → Fin 8

/-- Predicate to check if a row is a valid permutation of 1 to 8 -/
def IsValidPermutation (row : Fin 8 → Fin 8) : Prop :=
  ∀ i j : Fin 8, i ≠ j → row i ≠ row j

/-- Predicate to check if the entire table is valid -/
def IsValidTable (t : Table) : Prop :=
  ∀ row : Fin 24, IsValidPermutation (t row)

/-- Product of a column in the table -/
def ColumnProduct (t : Table) (col : Fin 8) : ℕ :=
  (Finset.univ.prod fun row => (t row col).val.succ)

/-- Sum of all column products -/
def SumOfProducts (t : Table) : ℕ :=
  (Finset.univ.sum fun col => ColumnProduct t col)

/-- The main theorem -/
theorem min_sum_of_products (t : Table) (h : IsValidTable t) :
    SumOfProducts t ≥ 8 * (Nat.factorial 8)^3 := by
  sorry

/-- The equality case -/
theorem min_sum_of_products_equality :
    ∃ t : Table, IsValidTable t ∧ SumOfProducts t = 8 * (Nat.factorial 8)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_products_min_sum_of_products_equality_l58_5831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l58_5800

-- Define the function
noncomputable def f (x : ℝ) : ℝ := -3 * Real.sin x + 1

-- State the theorem
theorem f_range :
  Set.range f = Set.Icc (-4) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l58_5800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l58_5867

/-- A cubic function f(x) = ax³ + bx² + cx where a ≠ 0 -/
def cubic_function (a b c : ℝ) (ha : a ≠ 0) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x

theorem cubic_function_properties (a b c : ℝ) (ha : a ≠ 0) :
  let f := cubic_function a b c ha
  (∀ x, f (-x) = -f x) →  -- f is odd
  f (-1) = 1 →  -- f takes extreme value 1 at x = -1
  (∃ x, f x = 1 ∧ ∀ y, f y ≤ f x) →  -- 1 is indeed an extreme value
  a = 1/2 ∧ b = 0 ∧ c = -3/2 ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ 2) ∧
  (∀ s, (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ s) → s ≥ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l58_5867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_percentage_l58_5894

/-- Calculates the percentage reduction between two numbers -/
noncomputable def percentageReduction (original : ℝ) (reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

/-- The original number of faculty members -/
def originalFaculty : ℝ := 226.74

/-- The reduced number of faculty members -/
def reducedFaculty : ℝ := 195

/-- Theorem stating that the percentage reduction in faculty members is approximately 13.99% -/
theorem faculty_reduction_percentage :
  abs (percentageReduction originalFaculty reducedFaculty - 13.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_percentage_l58_5894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l58_5846

theorem solution_set_of_inequality (x : ℝ) : 
  (1/2 : ℝ)^(x-5) ≤ 2^x ↔ x ≥ 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l58_5846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_squared_interpretation_l58_5804

-- Define the coefficient of determination
noncomputable def R_squared : ℝ := sorry

-- Axiom: R² is between 0 and 1
axiom R_squared_bounds : 0 ≤ R_squared ∧ R_squared ≤ 1

-- Define a measure of linear relationship strength
noncomputable def linear_relationship_strength : ℝ := sorry

-- Define a measure of regression model fit
noncomputable def regression_model_fit : ℝ := sorry

-- Theorem to prove
theorem R_squared_interpretation :
  ∀ ε > 0, ∃ δ > 0, 
    1 - R_squared < δ → 
    (linear_relationship_strength > 1 - ε ∧ 
     regression_model_fit > 1 - ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_squared_interpretation_l58_5804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_size_is_1920000_acres_l58_5828

/-- Represents the dimensions of a trapezoid on a map -/
structure MapDimensions where
  longerBase : ℚ
  shorterBase : ℚ
  height : ℚ

/-- Calculates the actual size of a plot of land in acres -/
noncomputable def actualPlotSize (mapDim : MapDimensions) (mapScale : ℚ) (sqMileToAcres : ℚ) : ℚ :=
  let mapArea := (mapDim.longerBase + mapDim.shorterBase) / 2 * mapDim.height
  let actualAreaSqMiles := mapArea * mapScale^2
  actualAreaSqMiles * sqMileToAcres

/-- Theorem stating the actual size of the plot -/
theorem plot_size_is_1920000_acres :
  let mapDim : MapDimensions := { longerBase := 18, shorterBase := 12, height := 8 }
  let mapScale : ℚ := 5  -- 1 cm on map = 5 miles in reality
  let sqMileToAcres : ℚ := 640
  actualPlotSize mapDim mapScale sqMileToAcres = 1920000 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_size_is_1920000_acres_l58_5828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_is_28_percent_l58_5878

/-- Calculates the profit percentage without discount given the discount percentage and profit percentage with discount -/
noncomputable def profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) : ℝ :=
  let cost_price := 100
  let selling_price_with_discount := cost_price * (1 + profit_with_discount_percent / 100)
  let selling_price_without_discount := selling_price_with_discount / (1 - discount_percent / 100)
  (selling_price_without_discount - cost_price) / cost_price * 100

/-- Theorem stating that given a 5% discount and 21.6% profit with discount, the profit without discount would be 28% -/
theorem profit_without_discount_is_28_percent :
  profit_without_discount 5 21.6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_is_28_percent_l58_5878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l58_5850

-- Define the function f
def f (x y z : ℝ) : ℝ := |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| + |x|

-- Define the region
def region : Set (ℝ × ℝ × ℝ) := {p | f p.1 p.2.1 p.2.2 ≤ 6}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume region = 288 / 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l58_5850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payments_l58_5872

/-- Represents the weekly payments to employees X, Y, and Z -/
structure EmployeePayments where
  x : ℝ  -- Payment to X in USD
  y : ℝ  -- Payment to Y in INR
  z : ℝ  -- Payment to Z in EUR

/-- Conversion rates and payment relationships -/
structure PaymentConditions where
  total : ℝ        -- Total payment in USD
  usd_to_inr : ℝ   -- Conversion rate from USD to INR
  usd_to_eur : ℝ   -- Conversion rate from USD to EUR
  x_to_y_ratio : ℝ -- Ratio of X's payment to Y's payment
  z_to_x_ratio : ℝ -- Ratio of Z's payment to X's payment

/-- Check if the given payments satisfy all conditions -/
def satisfiesConditions (p : EmployeePayments) (c : PaymentConditions) : Prop :=
  -- Total payment condition
  p.x + p.y / c.usd_to_inr + p.z / c.usd_to_eur = c.total ∧
  -- Payment relationship conditions
  p.x = c.x_to_y_ratio * p.y / c.usd_to_inr ∧
  p.z = c.z_to_x_ratio * p.x / c.usd_to_eur

/-- The main theorem to prove -/
theorem correct_payments : 
  ∃ (p : EmployeePayments),
    satisfiesConditions p 
      { total := 1200
      , usd_to_inr := 74
      , usd_to_eur := 0.85
      , x_to_y_ratio := 1.2
      , z_to_x_ratio := 1.5 } ∧ 
    (abs (p.x - 333.60) < 0.01 ∧ 
     abs (p.y - 20564.65) < 0.01 ∧ 
     abs (p.z - 589.53) < 0.01) := by
  sorry

#check correct_payments

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payments_l58_5872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_specific_parabola_l58_5854

/-- The vertex of a parabola defined by a quadratic equation -/
noncomputable def parabola_vertex (a b c d : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  let y := -(a * x^2 + b * x + d) / c
  (x, y)

/-- Theorem: The vertex of the parabola x^2 - 4x + 3y + 10 = 0 is (2, -2) -/
theorem vertex_of_specific_parabola :
  parabola_vertex 1 (-4) 3 10 = (2, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_of_specific_parabola_l58_5854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_triangle_conditions_l58_5861

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Condition for a geometric sequence to form a triangle -/
def is_triangle_forming (q : ℝ) : Prop :=
  1 ≤ q ∧ q < φ

/-- Condition for a geometric sequence to form a right triangle -/
def is_right_triangle_forming (q : ℝ) : Prop :=
  q = Real.sqrt φ

theorem geometric_sequence_triangle_conditions (a q : ℝ) (hq : q ≥ 1) :
  (is_triangle_forming q ↔ a > 0 ∧ a + a*q > a*q^2) ∧
  (is_right_triangle_forming q ↔ a^2 + (a*q)^2 = (a*q^2)^2) :=
by
  sorry

#check geometric_sequence_triangle_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_triangle_conditions_l58_5861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l58_5856

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  2 * k * x^2 + k * x - 1 < 0

def necessary_but_not_sufficient (k : ℝ) : Prop :=
  (∀ x, quadratic_inequality k x) ∧ 
  ∃ k', k' ≠ k ∧ (∀ x, quadratic_inequality k' x)

theorem quadratic_inequality_range :
  {k : ℝ | necessary_but_not_sufficient k} = Set.Icc (-8 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l58_5856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_cut_perimeter_difference_l58_5865

/-- Given a 6-foot by 9-foot rectangular piece of plywood cut into 3 congruent rectangles
    with no waste, the positive difference between the maximum and minimum possible
    perimeter of a single piece is 4 feet. -/
theorem plywood_cut_perimeter_difference :
  ∀ (cut_length cut_width : ℝ),
  cut_length > 0 ∧ cut_width > 0 →
  cut_length * cut_width = 18 →
  3 * cut_length = 9 ∨ 3 * cut_width = 6 →
  (max (2 * cut_length + 2 * cut_width) (2 * 2 + 2 * 9)) -
  (min (2 * cut_length + 2 * cut_width) (2 * 6 + 2 * 3)) = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_cut_perimeter_difference_l58_5865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_one_l58_5822

-- Define the bounds of the region
noncomputable def lower_bound : ℝ := 0
noncomputable def upper_bound : ℝ := 1

-- Define the functions that bound the region
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g : ℝ := Real.exp 1

-- Theorem statement
theorem area_of_region_is_one :
  (∫ x in lower_bound..upper_bound, g - f x) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_one_l58_5822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l58_5815

-- Define the circle C
def myCircle (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y + 2*a - 1)^2 = 2

-- Define the line l
def myLine (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

-- Define the condition that the circle is below the line
def circle_below_line (a b : ℝ) : Prop :=
  ∀ x y : ℝ, myCircle a x y → y < x + b

-- Define the condition that the circle and line have at most one intersection
def at_most_one_intersection (a b : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, myCircle a p.1 p.2 ∧ myLine b p.1 p.2

-- Main theorem
theorem min_b_value :
  ∀ a b : ℝ,
    -1 ≤ a → a ≤ 1 →
    circle_below_line a b →
    at_most_one_intersection a b →
    b ≥ 6 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l58_5815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_ellipse_with_foci_on_x_axis_l58_5895

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_ellipse_with_foci_on_x_axis (m n : ℕ) : Prop := m > n

def count_favorable_pairs : ℕ := 
  Finset.sum (Finset.range 5) (λ i => i + 1)

def total_pairs : ℕ := S.card * S.card

theorem probability_of_ellipse_with_foci_on_x_axis :
  (count_favorable_pairs : ℚ) / total_pairs = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_ellipse_with_foci_on_x_axis_l58_5895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_angle_l58_5812

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

-- Define the line on which P moves
def line_P (x y : ℝ) : Prop := x + y + 2 = 0

-- Define a point on the circle
def point_on_circle (A : ℝ × ℝ) : Prop := circle_C A.1 A.2

-- Define a point on the line
def point_on_line (P : ℝ × ℝ) : Prop := line_P P.1 P.2

-- Define a tangent point
def is_tangent_point (P A : ℝ × ℝ) : Prop := 
  point_on_line P ∧ point_on_circle A ∧ 
  ∃ (t : ℝ), A.1 = P.1 + t * (P.2 + 1) ∧ A.2 = P.2 - t * (P.1 + 1)

-- Define area of quadrilateral PACB (placeholder)
noncomputable def area_PACB (P A C B : ℝ × ℝ) : ℝ := sorry

-- Define angle between three points (placeholder)
noncomputable def angle (A C B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_area_angle : 
  ∀ (P A B C : ℝ × ℝ),
  point_on_circle C →
  is_tangent_point P A →
  is_tangent_point P B →
  (∀ (P' A' B' : ℝ × ℝ), 
    is_tangent_point P' A' → 
    is_tangent_point P' B' → 
    area_PACB P A C B ≤ area_PACB P' A' C B') →
  angle A C B = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_angle_l58_5812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l58_5803

/-- The time taken for two trains to cross each other -/
noncomputable def train_crossing_time (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

/-- Conversion factor from km/h to m/s -/
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

theorem train_crossing_theorem (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 140)
  (h2 : length2 = 160)
  (h3 : speed1 = 60 * kmh_to_ms)
  (h4 : speed2 = 48 * kmh_to_ms) :
  train_crossing_time length1 length2 speed1 speed2 = 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 140 160 (60 * kmh_to_ms) (48 * kmh_to_ms)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l58_5803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_pens_count_l58_5862

/-- The number of pens in the box -/
def total_pens : ℕ := 10

/-- The probability of selecting two non-defective pens -/
noncomputable def prob_non_defective : ℚ := 7/15

/-- Calculates the probability of selecting two non-defective pens -/
noncomputable def calculate_probability (non_defective : ℕ) : ℚ :=
  (non_defective : ℚ) / total_pens * ((non_defective - 1) : ℚ) / (total_pens - 1)

theorem defective_pens_count :
  ∃ (defective : ℕ), 
    defective + (total_pens - defective) = total_pens ∧
    calculate_probability (total_pens - defective) = prob_non_defective ∧
    defective = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_pens_count_l58_5862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_F5_l58_5877

def diamonds : ℕ → ℕ
  | 0 => 3  -- We'll use 0 to represent F₁
  | 1 => 19 -- This is F₂
  | n + 2 => diamonds (n + 1) + 4 * (diamonds (n + 1) - diamonds n + 2)

theorem diamonds_F5 : diamonds 4 = 91 := by
  -- The proof goes here
  sorry

#eval diamonds 4  -- This will evaluate F₅ (which is the 5th figure, but index 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_F5_l58_5877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_l58_5883

/-- The function φ that satisfies the integral equation -/
noncomputable def φ (t : ℝ) : ℝ := (2 / Real.pi) * (t / (1 + t^2))

/-- The integral equation -/
theorem integral_equation (x : ℝ) (hx : x > 0) :
  ∫ (t : ℝ) in Set.Ici 0, φ t * Real.sin (x * t) = Real.exp (-x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_l58_5883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_five_in_25_factorial_l58_5844

theorem greatest_power_of_five_in_25_factorial : 
  ∃ k : ℕ, k = 6 ∧ 
  (∀ m : ℕ, 5^m ∣ Nat.factorial 25 → m ≤ k) ∧
  5^k ∣ Nat.factorial 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_five_in_25_factorial_l58_5844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_equivalent_to_power_of_three_l58_5888

def satisfies_condition (k : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ¬(3^((k-1)*n+1) ∣ (Nat.factorial (k*n) / Nat.factorial n)^2)

def is_power_of_three (k : ℕ) : Prop :=
  ∃ m : ℕ, k = 3^m

theorem condition_equivalent_to_power_of_three (k : ℕ) (h : k ≤ 2020) :
  satisfies_condition k ↔ is_power_of_three k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_equivalent_to_power_of_three_l58_5888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_travel_time_l58_5855

/- Define the given constants and variables -/
noncomputable def distance_AB : ℝ := 100
noncomputable def initial_speed_bus2 : ℝ := 50
def max_capacity : ℕ := 42
def passengers_bus1 : ℕ := 20
def initial_passengers_bus2 : ℕ := 22
noncomputable def speed_decrease_rate : ℝ := 0.5  -- 1 km/h per 2 passengers

noncomputable def breakdown_distance : ℝ := 30
noncomputable def remaining_distance : ℝ := distance_AB - breakdown_distance

noncomputable def time_to_breakdown : ℝ := breakdown_distance / initial_speed_bus2

def total_passengers : ℕ := passengers_bus1 + initial_passengers_bus2
noncomputable def speed_decrease : ℝ := (total_passengers - initial_passengers_bus2 : ℝ) * speed_decrease_rate
noncomputable def final_speed : ℝ := initial_speed_bus2 - speed_decrease

noncomputable def time_after_pickup : ℝ := remaining_distance / final_speed

theorem bus_travel_time :
  time_to_breakdown + time_after_pickup = 2 + 21 / 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_travel_time_l58_5855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_divides_altitude_l58_5806

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define an altitude of a triangle
noncomputable def altitude (t : Triangle) (vertex : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define tangent function
noncomputable def tan (angle : ℝ) : ℝ := sorry

theorem orthocenter_divides_altitude (t : Triangle) :
  let H := orthocenter t
  let D := altitude t t.B
  distance H D = 8 ∧ distance H t.B = 20 →
  tan (Real.arctan (sorry)) * tan (Real.arctan (sorry)) = 7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_divides_altitude_l58_5806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_equals_present_value_l58_5833

/-- The principal required for infinite withdrawals -/
noncomputable def required_principal (i : ℝ) : ℝ := ((1 + i) * (2 + i)) / i^3

/-- The present value of withdrawals -/
noncomputable def present_value_of_withdrawals (i : ℝ) : ℝ := 
  ∑' n, (n^2 : ℝ) * (1 + i)^(-n)

/-- Theorem stating that the required principal equals the present value of withdrawals -/
theorem principal_equals_present_value (i : ℝ) (h : i > 0) :
  required_principal i = present_value_of_withdrawals i := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_equals_present_value_l58_5833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bolt_catch_ace_l58_5808

/-- The distance Bolt must run to catch Ace -/
noncomputable def bolt_distance (z : ℝ) (m : ℝ) : ℝ :=
  (100 + z) * m / (100 - z)

/-- Theorem stating the distance Bolt must run to catch Ace -/
theorem bolt_catch_ace (z m v : ℝ) (hz : z > 0) (hm : m > 0) (hv : v > 0) :
  let ace_speed := v
  let bolt_speed := (1 + z / 100) * v
  let relative_speed := bolt_speed - ace_speed
  let catch_up_time := m / relative_speed
  bolt_speed * catch_up_time = bolt_distance z m :=
by
  sorry

#check bolt_catch_ace

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bolt_catch_ace_l58_5808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_from_radius_relation_l58_5825

/-- Given a triangle with sides a, b, c, circumradius R, inradius r, and exradius r₁,
    if abc = 4Rrr₁, then the triangle is right-angled. -/
theorem triangle_right_angled_from_radius_relation
  (a b c R r r₁ : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0 ∧ r₁ > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_relation : a * b * c = 4 * R * r * r₁) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ Real.cos θ = a / (2 * R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_from_radius_relation_l58_5825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_square_half_area_l58_5834

/-- A structure representing two squares and a circle with specific properties -/
structure SquaresAndCircle where
  largeSquare : Set (ℝ × ℝ)
  circle : Set (ℝ × ℝ)
  smallSquare : Set (ℝ × ℝ)
  inscribedCircle : circle ⊆ largeSquare
  inscribedSmallSquare : smallSquare ⊆ circle
  sideCoincidence : ∃ (side : Set (ℝ × ℝ)), side ⊆ smallSquare ∧ side ⊆ largeSquare
  verticesOnCircle : ∃ (v1 v2 : ℝ × ℝ), v1 ∈ smallSquare ∧ v2 ∈ smallSquare ∧ v1 ∈ circle ∧ v2 ∈ circle

/-- The area of a set in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

/-- The theorem stating that the area of the small square is half the area of the large square -/
theorem small_square_half_area (sc : SquaresAndCircle) :
  area sc.smallSquare = (1/2) * area sc.largeSquare := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_square_half_area_l58_5834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_ratio_l58_5839

/-- Given a quadrilateral ABCD and a line intersecting its sides or their extensions at points P, Q, R, S,
    the product of the ratios of the segments is equal to 1. -/
theorem quadrilateral_intersection_ratio (A B C D P Q R S : ℝ × ℝ) : 
  let dist := λ (X Y : ℝ × ℝ) ↦ Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  (dist A P / dist P B) * (dist B Q / dist Q C) * (dist C R / dist R D) * (dist D S / dist S A) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_ratio_l58_5839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l58_5818

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Define the proposed tangent line
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

theorem tangent_line_at_point :
  (∃ (m : ℝ), HasDerivAt f m point.fst) →
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - point.fst| < δ → |f x - (tangent_line x)| ≤ ε * |x - point.fst|) :=
by
  sorry

#check tangent_line_at_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l58_5818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_powers_l58_5896

def next_triple (t : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (a, b, c) := t
  (a * b, b * c, c * a)

def sequence_triple : ℕ → ℕ × ℕ × ℕ
  | 0 => (2, 3, 5)
  | n + 1 => next_triple (sequence_triple n)

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (base k : ℕ), k > 1 ∧ n = base ^ k

theorem no_perfect_powers (n : ℕ) :
  let (x, y, z) := sequence_triple n
  ¬(is_perfect_power x) ∧ ¬(is_perfect_power y) ∧ ¬(is_perfect_power z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_powers_l58_5896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_sequence_unique_solution_l58_5882

/-- A sequence with a linear general term -/
def LinearSequence (k b : ℚ) : ℕ → ℚ := fun n ↦ k * n + b

theorem linear_sequence_unique_solution (a : ℕ → ℚ) (h_linear : ∃ k b : ℚ, k ≠ 0 ∧ ∀ n, a n = k * n + b)
    (h_a1 : a 1 = 2) (h_a17 : a 17 = 66) :
  (∀ n, a n = 4 * n - 2) ∧ a 2015 = 8058 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_sequence_unique_solution_l58_5882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_repeated_digit_integers_l58_5835

def repeated_digit_integer (n : ℕ) : ℕ := n * 1000000000 + n * 1000000 + n * 1000 + n

theorem gcd_repeated_digit_integers :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → d ∣ repeated_digit_integer n) ∧
  (∀ (m : ℕ), m > 0 ∧ 
    (∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → m ∣ repeated_digit_integer n) → 
    m ≤ d) ∧
  d = 100001001001 := by
  sorry

#check gcd_repeated_digit_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_repeated_digit_integers_l58_5835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_rv_generation_formula_l58_5840

/-- A continuous random variable uniformly distributed in the interval (a, b) -/
structure UniformRV (a b : ℝ) where
  lt_ab : a < b

/-- The distribution function for a uniform random variable -/
noncomputable def distributionFunction (a b : ℝ) (x : ℝ) : ℝ :=
  (x - a) / (b - a)

/-- A random number uniformly distributed over (0, 1) -/
structure UniformRandom where
  value : ℝ
  in_01 : 0 < value ∧ value < 1

/-- The theorem stating the explicit formula for generating a uniform random variable -/
theorem uniform_rv_generation_formula (a b : ℝ) (X : UniformRV a b) (r : UniformRandom) :
  ∃ (x : ℝ), a < x ∧ x < b ∧ distributionFunction a b x = r.value ∧ x = (b - a) * r.value + a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_rv_generation_formula_l58_5840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l58_5891

/-- The first line equation: 2x + 3y - 6 = 0 -/
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0

/-- The second line equation: 4x - 3y - 6 = 0 -/
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

/-- The x-coordinate of the potential intersection point -/
noncomputable def x_coord : ℝ := 2

/-- The y-coordinate of the potential intersection point -/
noncomputable def y_coord : ℝ := 2/3

/-- Theorem stating that (2, 2/3) is the unique intersection point -/
theorem unique_intersection :
  (∀ x y : ℝ, line1 x y ∧ line2 x y → x = x_coord ∧ y = y_coord) ∧
  line1 x_coord y_coord ∧
  line2 x_coord y_coord := by
  sorry

#check unique_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l58_5891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_minimal_perimeter_l58_5836

/-- A quadrilateral with a given area -/
structure Quadrilateral :=
  (area : ℝ)
  (perimeter : ℝ)

/-- The perimeter of a square with a given area -/
noncomputable def square_perimeter (area : ℝ) : ℝ := 4 * Real.sqrt area

/-- Theorem: Among all quadrilaterals with a given area, the square has the smallest perimeter -/
theorem square_minimal_perimeter (q : Quadrilateral) :
  q.perimeter ≥ square_perimeter q.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_minimal_perimeter_l58_5836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_to_read_l58_5819

def total_books : ℕ := 210
def percentage_read : ℚ := 60 / 100

theorem books_to_read :
  total_books - (percentage_read * ↑total_books).floor = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_to_read_l58_5819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_along_stream_l58_5845

/-- Represents the speed of a boat in km/hr -/
noncomputable def BoatSpeed : ℝ := 7

/-- Represents the distance traveled against the stream in km -/
noncomputable def DistanceAgainst : ℝ := 5

/-- Represents the time of travel in hours -/
noncomputable def TravelTime : ℝ := 1

/-- Calculates the speed of the stream based on the boat's speed and distance traveled against the stream -/
noncomputable def StreamSpeed : ℝ := BoatSpeed - (DistanceAgainst / TravelTime)

/-- Calculates the distance traveled along the stream in one hour -/
noncomputable def DistanceAlong : ℝ := (BoatSpeed + StreamSpeed) * TravelTime

theorem boat_distance_along_stream :
  DistanceAlong = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_along_stream_l58_5845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_part_III_l58_5890

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define point P
def P : ℝ × ℝ := (4, 5)

-- Define point M
def M : ℝ × ℝ → Prop := λ p => C p.1 p.2

-- Theorem for part I
theorem part_I :
  C P.1 P.2 →
  (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 10) ∧
  ((P.2 - Q.2) / (P.1 - Q.1) = 1/3) :=
by sorry

-- Theorem for part II
theorem part_II :
  ∀ m : ℝ × ℝ, M m →
  (∃ m₀, M m₀ ∧ Real.sqrt ((m₀.1 - Q.1)^2 + (m₀.2 - Q.2)^2) = 6 * Real.sqrt 2) ∧
  (∃ m₁, M m₁ ∧ Real.sqrt ((m₁.1 - Q.1)^2 + (m₁.2 - Q.2)^2) = 2 * Real.sqrt 2) ∧
  (∀ m', M m' → 2 * Real.sqrt 2 ≤ Real.sqrt ((m'.1 - Q.1)^2 + (m'.2 - Q.2)^2) ∧
                Real.sqrt ((m'.1 - Q.1)^2 + (m'.2 - Q.2)^2) ≤ 6 * Real.sqrt 2) :=
by sorry

-- Theorem for part III
theorem part_III :
  ∀ m n : ℝ, M (m, n) →
  (∃ m₀ n₀, M (m₀, n₀) ∧ (n₀ - 3) / (m₀ + 2) = 2 + Real.sqrt 3) ∧
  (∃ m₁ n₁, M (m₁, n₁) ∧ (n₁ - 3) / (m₁ + 2) = 2 - Real.sqrt 3) ∧
  (∀ m' n', M (m', n') → 2 - Real.sqrt 3 ≤ (n' - 3) / (m' + 2) ∧ (n' - 3) / (m' + 2) ≤ 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_part_III_l58_5890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l58_5827

noncomputable def largest_circle_area : ℝ := 100 * Real.pi

noncomputable def largest_shaded_area : ℝ := largest_circle_area / 2
noncomputable def medium_shaded_area : ℝ := (largest_circle_area / 4) / 2
noncomputable def smallest_shaded_area : ℝ := (largest_circle_area / 16) / 2

theorem total_shaded_area :
  largest_shaded_area + medium_shaded_area + smallest_shaded_area = 65.625 * Real.pi :=
by
  -- Expand definitions
  unfold largest_shaded_area medium_shaded_area smallest_shaded_area largest_circle_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l58_5827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_sum_l58_5874

-- Define the number of straight and slanted segments
def num_straight_segments : ℕ := 12
def num_slanted_segments : ℕ := 2

-- Define the length of a straight segment
def straight_segment_length : ℝ := 1

-- Define the length of a slanted segment
noncomputable def slanted_segment_length : ℝ := Real.sqrt 2

-- Theorem statement
theorem xyz_length_sum : 
  (num_straight_segments : ℝ) * straight_segment_length + 
  (num_slanted_segments : ℝ) * slanted_segment_length = 12 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_sum_l58_5874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alis_stone_is_green_l58_5826

-- Define the color of stones
inductive Color
| Red
| Green

-- Define the people
inductive Person
| Ali
| Bev
| Chaz

-- Define the stone ownership
def stone_color : Person → Color := sorry

-- Define the statement made by each person
def ali_statement : Prop := stone_color Person.Ali = stone_color Person.Bev
def bev_statement : Prop := stone_color Person.Bev = stone_color Person.Chaz
def chaz_statement : Prop := 
  (stone_color Person.Ali = Color.Red) ∧ 
  (stone_color Person.Bev = Color.Red) ∨
  (stone_color Person.Ali = Color.Red) ∧ 
  (stone_color Person.Chaz = Color.Red) ∨
  (stone_color Person.Bev = Color.Red) ∧ 
  (stone_color Person.Chaz = Color.Red)

-- Define that all statements are false
axiom all_statements_false : 
  ¬ali_statement ∧ ¬bev_statement ∧ ¬chaz_statement

-- Theorem to prove
theorem alis_stone_is_green : stone_color Person.Ali = Color.Green := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alis_stone_is_green_l58_5826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l58_5860

/-- The equation of the hyperbola -2x^2 + 3y^2 + 20x - 9y + 4 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 + 20 * x - 9 * y + 4 = 0

/-- A point is a focus of the hyperbola if it satisfies the focus condition -/
def is_focus (h : ℝ → ℝ → Prop) (fx fy : ℝ) : Prop :=
  ∃ (cx cy a b : ℝ),
    (∀ x y, h x y ↔ ((y - cy)^2 / a^2) - ((x - cx)^2 / b^2) = 1) ∧
    fx = cx ∧ fy = cy + Real.sqrt (a^2 + b^2)

theorem hyperbola_focus :
  is_focus hyperbola_equation 5 (1.5 + Real.sqrt 39.375) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l58_5860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_mean_eq_weighted_sum_div_total_l58_5805

variable (m n p : ℕ) (x₁ x₂ x₃ : ℝ)

/-- The mean of a sample with m instances of x₁, n instances of x₂, and p instances of x₃ -/
noncomputable def sampleMean (m n p : ℕ) (x₁ x₂ x₃ : ℝ) : ℝ :=
  (m * x₁ + n * x₂ + p * x₃) / (m + n + p : ℝ)

/-- Theorem stating that the sample mean is equal to the weighted sum divided by total instances -/
theorem sample_mean_eq_weighted_sum_div_total :
  sampleMean m n p x₁ x₂ x₃ = (m * x₁ + n * x₂ + p * x₃) / (m + n + p : ℝ) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_mean_eq_weighted_sum_div_total_l58_5805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_heads_in_eight_tosses_l58_5811

theorem probability_three_heads_in_eight_tosses :
  let n : ℕ := 8  -- number of tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of ways to choose k heads from n tosses
  (favorable_outcomes : ℚ) / total_outcomes = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_heads_in_eight_tosses_l58_5811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mustard_oil_cost_is_13_l58_5887

/-- Represents the cost of groceries and change left after shopping --/
structure GroceryShopping where
  initialAmount : ℚ
  pennePerPound : ℚ
  pennePounds : ℚ
  sauceCost : ℚ
  mustardLiters : ℚ
  remainingAmount : ℚ

/-- Calculates the cost of mustard oil per liter --/
noncomputable def mustardOilCost (g : GroceryShopping) : ℚ :=
  (g.initialAmount - g.remainingAmount - (g.pennePerPound * g.pennePounds + g.sauceCost)) / g.mustardLiters

/-- Theorem stating the cost of mustard oil per liter --/
theorem mustard_oil_cost_is_13 (g : GroceryShopping)
  (h1 : g.initialAmount = 50)
  (h2 : g.pennePerPound = 4)
  (h3 : g.pennePounds = 3)
  (h4 : g.sauceCost = 5)
  (h5 : g.mustardLiters = 2)
  (h6 : g.remainingAmount = 7) :
  mustardOilCost g = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mustard_oil_cost_is_13_l58_5887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l58_5841

/-- Represents the possible operators in the game -/
inductive Operator
| Add
| Sub
| Mul

/-- Represents a game state -/
structure GameState where
  numbers : List Nat
  operators : List Operator

/-- Represents a player's strategy -/
def Strategy := GameState → Operator

/-- Calculates the result of applying operators to numbers without parentheses -/
def calculateResult (state : GameState) : Nat :=
  sorry

/-- Checks if a number is even -/
def isEven (n : Nat) : Prop :=
  ∃ k, n = 2 * k

/-- Theorem: The first player has a winning strategy -/
theorem first_player_winning_strategy :
  ∃ (s : Strategy),
    ∀ (opponent_moves : List Operator),
      isEven (calculateResult {
        numbers := List.range 101,  -- Changed to 101 to get numbers from 1 to 100
        operators := opponent_moves
      }) :=
by
  sorry

#check first_player_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l58_5841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tokens_to_convex_polygon_l58_5851

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function to reflect a point about another point -/
def reflect (p : Point) (center : Point) : Point :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- A predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- A predicate to check if a list of points forms a convex polygon -/
def is_convex_polygon (points : List Point) : Prop :=
  sorry  -- Definition of convex polygon

theorem tokens_to_convex_polygon 
  (tokens : List Point) 
  (h1 : tokens.length ≥ 3)
  (h2 : ∃ p1 p2 p3, p1 ∈ tokens ∧ p2 ∈ tokens ∧ p3 ∈ tokens ∧ ¬collinear p1 p2 p3) : 
  ∃ (new_tokens : List Point), 
    (∀ p ∈ new_tokens, ∃ q ∈ tokens, ∃ center ∈ tokens, p = reflect q center) ∧ 
    is_convex_polygon new_tokens :=
by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tokens_to_convex_polygon_l58_5851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_correspondence_l58_5829

-- Define the angle in degrees
noncomputable def angle : ℝ := 72

-- Define the line equation
noncomputable def line_equation (x : ℝ) : ℝ := 1 - x * Real.tan (angle * Real.pi / 180)

-- State the theorem
theorem slope_angle_correspondence :
  ∃ (m : ℝ), (∀ x y : ℝ, y = line_equation x → y - line_equation 0 = m * (x - 0)) ∧
             m = Real.tan ((180 - angle) * Real.pi / 180) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_correspondence_l58_5829
