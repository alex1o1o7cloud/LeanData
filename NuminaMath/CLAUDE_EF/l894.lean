import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l894_89488

-- Define the original function f
def f (x : ℝ) : ℝ := 3 + 7 * x - x^2

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := (7 + Real.sqrt (37 + 4 * x)) / 2

-- Theorem statement
theorem inverse_function_proof :
  ∀ x : ℝ, f (g x) = x ∧ g (f x) = x := by
  sorry

#check inverse_function_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l894_89488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l894_89442

/-- Given a line with direction vector (2, -1, z) perpendicular to a plane with normal vector (4, -2, -2), prove that z = -1 -/
theorem perpendicular_line_plane (z : ℝ) : 
  let m : Fin 3 → ℝ := ![2, -1, z]
  let n : Fin 3 → ℝ := ![4, -2, -2]
  (∃ (k : ℝ), m = fun i => k * n i) → z = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l894_89442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_correct_original_and_inverse_not_equivalent_l894_89409

-- Define a structure for Line
structure Line where
  -- You might want to add more properties here, but for now we'll keep it simple
  mk :: -- empty constructor

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the equal_corresponding_angles relation
def equal_corresponding_angles (l1 l2 : Line) : Prop := sorry

-- Define the original proposition
def original_proposition : Prop := ∀ (l1 l2 : Line), parallel l1 l2 → equal_corresponding_angles l1 l2

-- Define the inverse proposition
def inverse_proposition : Prop := ∀ (l1 l2 : Line), equal_corresponding_angles l1 l2 → parallel l1 l2

-- Theorem stating that the inverse proposition is correct
theorem inverse_proposition_correct :
  (∀ (l1 l2 : Line), equal_corresponding_angles l1 l2 → parallel l1 l2) ↔ inverse_proposition :=
by
  -- The proof is trivial since the left and right sides are identical
  rfl

-- Additional theorem to show the relationship between the original and inverse propositions
theorem original_and_inverse_not_equivalent :
  ¬(original_proposition ↔ inverse_proposition) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_correct_original_and_inverse_not_equivalent_l894_89409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l894_89427

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (max min : ℝ),
    (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ -π/12 → f x ≤ max) ∧
    (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ -π/12 ∧ f x = max) ∧
    (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ -π/12 → min ≤ f x) ∧
    (∃ (x : ℝ), -π/6 ≤ x ∧ x ≤ -π/12 ∧ f x = min) ∧
    max + min = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l894_89427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_term_properties_l894_89471

-- Define the term
noncomputable def term (a b : ℝ) : ℝ := -((2 * Real.pi * a * b) / 3)

-- Theorem statement
theorem term_properties (a b : ℝ) :
  (∃ c, term a b = c * a * b ∧ c = -(2 * Real.pi / 3)) ∧
  (∀ x y, term x y ≠ 0 → (∃ n : ℕ, n = 2 ∧ ∀ t, t > 0 → term (t*x) (t*y) = t^n * term x y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_term_properties_l894_89471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_is_22_l894_89406

/-- A structure representing a regular polygon --/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- The interior angle of a regular polygon --/
noncomputable def interiorAngle (p : RegularPolygon) : ℝ :=
  180 * (p.sides - 2 : ℝ) / p.sides

/-- A configuration of four regular polygons meeting at a point --/
structure PolygonConfiguration where
  p1 : RegularPolygon
  p2 : RegularPolygon
  p3 : RegularPolygon
  p4 : RegularPolygon
  congruentPair : p1 = p4 ∨ p1 = p3 ∨ p1 = p2 ∨ p2 = p3 ∨ p2 = p4 ∨ p3 = p4
  unitSides : p1.sideLength = 1 ∧ p2.sideLength = 1 ∧ p3.sideLength = 1 ∧ p4.sideLength = 1
  angleSum : interiorAngle p1 + interiorAngle p2 + interiorAngle p3 + interiorAngle p4 = 360

/-- The perimeter of a polygon configuration --/
def perimeter (c : PolygonConfiguration) : ℝ :=
  c.p1.sides + c.p2.sides + c.p3.sides + c.p4.sides - 8

/-- The theorem stating the maximum perimeter --/
theorem max_perimeter_is_22 :
  ∃ (c : PolygonConfiguration), ∀ (d : PolygonConfiguration), perimeter c ≥ perimeter d ∧ perimeter c = 22 := by
  sorry

#check max_perimeter_is_22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_is_22_l894_89406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l894_89484

/-- Calculates the speed in km/h given a distance in meters and time in seconds -/
noncomputable def calculate_speed (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

/-- Theorem stating that traveling 475.038 meters in 30 seconds results in a speed of approximately 57.006 km/h -/
theorem speed_calculation :
  let distance_meters : ℝ := 475.038
  let time_seconds : ℝ := 30
  let calculated_speed := calculate_speed distance_meters time_seconds
  ∃ ε > 0, |calculated_speed - 57.006| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l894_89484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_crossing_time_approx_45_seconds_l894_89437

/-- Represents the properties of a train and its journey through a tunnel and platform -/
structure TrainJourney where
  tunnel_length : ℝ
  platform_length : ℝ
  train_length : ℝ
  platform_crossing_time : ℝ

/-- Calculates the time taken for the train to cross the tunnel -/
noncomputable def tunnel_crossing_time (tj : TrainJourney) : ℝ :=
  let total_platform_distance := tj.train_length + tj.platform_length
  let train_speed := total_platform_distance / tj.platform_crossing_time
  let total_tunnel_distance := tj.train_length + tj.tunnel_length
  total_tunnel_distance / train_speed

/-- Theorem stating that the time taken to cross the tunnel is approximately 45 seconds -/
theorem tunnel_crossing_time_approx_45_seconds (tj : TrainJourney) 
  (h1 : tj.tunnel_length = 1200)
  (h2 : tj.platform_length = 180)
  (h3 : tj.train_length = 330)
  (h4 : tj.platform_crossing_time = 15) :
  ∃ ε > 0, |tunnel_crossing_time tj - 45| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_crossing_time_approx_45_seconds_l894_89437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_paths_count_l894_89458

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move on the grid -/
inductive Move
  | east1  : Move
  | east2  : Move
  | north1 : Move
  | north2 : Move

/-- A path is a list of moves -/
def RobotPath := List Move

/-- Function to check if a path is valid (reaches from A to B) -/
def isValidPath (start finish : Point) (path : RobotPath) : Prop := sorry

/-- Function to count the number of valid paths -/
def countValidPaths (start finish : Point) : ℕ := sorry

/-- The main theorem stating that there are 556 valid paths -/
theorem robot_paths_count (A B : Point) :
  countValidPaths A B = 556 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_paths_count_l894_89458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_l894_89413

-- Define the function y = (2c-1)^x
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (2*c - 1)^x

-- Define the function for the inequality
def g (c : ℝ) (x : ℝ) : ℝ := x + |x - 2*c|

-- Main theorem
theorem range_of_c (c : ℝ) 
  (h1 : c > 0) 
  (h2 : ¬(∀ x y : ℝ, x < y → f c x > f c y)) 
  (h3 : ∀ x : ℝ, g c x > 1) : 
  c ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_l894_89413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_equals_one_equilateral_triangle_l894_89448

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def angles_form_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

def sides_form_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

-- State the theorems
theorem sin_C_equals_one (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : angles_form_arithmetic_sequence t)
  (h3 : t.a = 1)
  (h4 : t.b = Real.sqrt 3) :
  Real.sin t.C = 1 := by sorry

theorem equilateral_triangle (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : angles_form_arithmetic_sequence t)
  (h3 : sides_form_arithmetic_sequence t) :
  t.A = t.B ∧ t.B = t.C ∧ t.A = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_equals_one_equilateral_triangle_l894_89448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepted_segment_length_l894_89420

/-- The line in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin θ - ρ * Real.cos θ = 1

/-- The curve in polar coordinates -/
def polar_curve (ρ : ℝ) : Prop := ρ = 1

/-- The length of the intercepted line segment -/
noncomputable def intercepted_length : ℝ := Real.sqrt 2

/-- Theorem stating that the length of the intercepted line segment is √2 -/
theorem intercepted_segment_length :
  ∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
    polar_line ρ₁ θ₁ ∧
    polar_line ρ₂ θ₂ ∧
    polar_curve ρ₁ ∧
    polar_curve ρ₂ ∧
    (ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2 = intercepted_length^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepted_segment_length_l894_89420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glaze_experiment_result_l894_89490

/-- Represents the experimental setup for the glaze formula optimization --/
structure GlazeExperiment where
  initial_range : Set ℝ
  test_points_per_batch : ℕ
  num_batches : ℕ

/-- Calculates the length of the remaining optimal range interval after a batch of experiments --/
def remaining_interval_length (experiment : GlazeExperiment) (batch : ℕ) : ℝ :=
  sorry

/-- The main theorem stating that the remaining interval length after the second batch is 0.8 --/
theorem glaze_experiment_result (experiment : GlazeExperiment) :
  experiment.initial_range = Set.Icc 10 28 →
  experiment.test_points_per_batch = 8 →
  experiment.num_batches = 2 →
  remaining_interval_length experiment 2 = 0.8 := by
  sorry

#check glaze_experiment_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glaze_experiment_result_l894_89490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_converges_l894_89411

noncomputable def mySequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => 2 + 1 / mySequence n

theorem mySequence_converges :
  ∃ (L : ℝ), L = 1 + Real.sqrt 2 ∧ Filter.Tendsto mySequence Filter.atTop (nhds L) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_converges_l894_89411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_fahrenheit_l894_89477

/-- The boiling point of water in Celsius -/
noncomputable def water_boiling_point_celsius : ℝ := 100

/-- Conversion function from Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := (c * 9 / 5) + 32

/-- Theorem: The boiling point of water in Fahrenheit is 212 °F -/
theorem water_boiling_point_fahrenheit :
  celsius_to_fahrenheit water_boiling_point_celsius = 212 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_fahrenheit_l894_89477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l894_89495

/-- A function satisfying the given conditions -/
def f_condition (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x, DifferentiableAt ℝ f x ∧ deriv f x < (1/3 : ℝ)

/-- The theorem statement -/
theorem solution_set (f : ℝ → ℝ) (h : f_condition f) :
  {x : ℝ | f (x^2) > x^2/3 + 2/3} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l894_89495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_integral_l894_89421

/-- The probability that a randomly chosen term in the expansion of (3√x - 2∛x)^11 is rational -/
noncomputable def p : ℝ := 1/6

/-- The definite integral of x^p from 0 to 1 -/
noncomputable def integral : ℝ := ∫ x in (0:ℝ)..1, x^p

theorem expansion_integral : integral = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_integral_l894_89421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_at_tc_l894_89412

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the problem setup
structure GeometricSetup where
  outer_circle : Circle
  inner_circle : Circle
  A : Point
  C : Point
  Q : Point
  P : Point
  R : Point
  S : Point
  T : Point

-- Define the conditions
def ConcentricCircles (setup : GeometricSetup) : Prop :=
  setup.outer_circle.center = setup.inner_circle.center

def ChordTouchesInnerCircle (setup : GeometricSetup) : Prop :=
  sorry -- Placeholder for the condition

def PMidpointAQ (setup : GeometricSetup) : Prop :=
  sorry -- Placeholder for the midpoint condition

def LineIntersectsInnerCircle (setup : GeometricSetup) : Prop :=
  sorry -- Placeholder for the intersection condition

def PerpendicularBisectorsMeet (setup : GeometricSetup) : Prop :=
  sorry -- Placeholder for the perpendicular bisectors condition

-- Define distance function
def Distance (p1 p2 : Point) : ℝ :=
  sorry -- Placeholder for distance calculation

-- Define the theorem
theorem ratio_at_tc (setup : GeometricSetup) :
  ConcentricCircles setup →
  ChordTouchesInnerCircle setup →
  PMidpointAQ setup →
  LineIntersectsInnerCircle setup →
  PerpendicularBisectorsMeet setup →
  (Distance setup.A setup.T) / (Distance setup.T setup.C) = 5 / 3 :=
by
  sorry -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_at_tc_l894_89412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_raise_percentage_l894_89444

/-- Calculates the percentage increase between two amounts -/
noncomputable def percentageIncrease (originalAmount newAmount : ℝ) : ℝ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem johns_raise_percentage (originalAmount newAmount : ℝ) 
  (h1 : originalAmount = 40)
  (h2 : newAmount = 55) :
  percentageIncrease originalAmount newAmount = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_raise_percentage_l894_89444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_polar_curve_l894_89447

/-- The polar curve defined by ρ = 4sin(θ - π/3) -/
noncomputable def polar_curve (θ : Real) : Real := 4 * Real.sin (θ - Real.pi / 3)

/-- The line of symmetry for the polar curve -/
noncomputable def line_of_symmetry : Real := 5 * Real.pi / 6

/-- Theorem stating that the line θ = 5π/6 is the line of symmetry for the given polar curve -/
theorem symmetry_of_polar_curve :
  ∀ θ : Real, polar_curve (2 * line_of_symmetry - θ) = polar_curve θ := by
  sorry

#check symmetry_of_polar_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_polar_curve_l894_89447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_result_l894_89424

/-- Represents a summer camp with a range of student numbers. -/
structure Camp where
  start : Nat
  endNum : Nat

/-- Represents the systematic sampling parameters. -/
structure SamplingParams where
  totalStudents : Nat
  sampleSize : Nat
  firstSelected : Nat

/-- Calculates the number of students selected from a camp given the sampling parameters. -/
def studentsSelectedFromCamp (camp : Camp) (params : SamplingParams) : Nat :=
  sorry

/-- Theorem stating the correct number of students selected from each camp. -/
theorem systematic_sampling_result (params : SamplingParams) 
  (camp1 camp2 camp3 : Camp) : 
  params.totalStudents = 600 ∧ 
  params.sampleSize = 50 ∧ 
  params.firstSelected = 3 ∧
  camp1.start = 1 ∧ camp1.endNum = 300 ∧
  camp2.start = 301 ∧ camp2.endNum = 495 ∧
  camp3.start = 496 ∧ camp3.endNum = 600 →
  studentsSelectedFromCamp camp1 params = 24 ∧
  studentsSelectedFromCamp camp2 params = 17 ∧
  studentsSelectedFromCamp camp3 params = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_result_l894_89424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l894_89455

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 5*x + 6)

-- State the theorem
theorem f_max_min_on_interval :
  let a : ℝ := 1
  let b : ℝ := 5
  let domain : Set ℝ := Set.Icc (-1 : ℝ) 6
  let interval : Set ℝ := Set.Icc a b
  (∀ x ∈ domain, f x ≥ 0) →
  (∀ x ∈ Set.Icc (-1 : ℝ) (5/2), ∀ y ∈ Set.Icc (-1 : ℝ) (5/2), x ≤ y → f x ≤ f y) →
  (∀ x ∈ Set.Icc (5/2 : ℝ) 6, ∀ y ∈ Set.Icc (5/2 : ℝ) 6, x ≤ y → f x ≥ f y) →
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x) ∧
  (∀ x ∈ interval, f x ≤ 7/2) ∧
  (∀ x ∈ interval, f x ≥ Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l894_89455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l894_89483

/-- The distance from the origin to a line in 2D space -/
noncomputable def distanceFromOriginToLine (a b c : ℝ) : ℝ :=
  |c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the origin to the line x-2y+5=0 is √5 -/
theorem distance_to_line : distanceFromOriginToLine 1 (-2) 5 = Real.sqrt 5 := by
  -- Unfold the definition of distanceFromOriginToLine
  unfold distanceFromOriginToLine
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l894_89483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l894_89462

-- Define the circles and points
def origin : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (5, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem circle_problem (k : ℝ) :
  (distance origin P = distance origin (S k) + 5) →
  k = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l894_89462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_consecutive_l894_89451

theorem smallest_divisible_consecutive (M : ℕ) : M = 484 ↔ 
  (M > 0) ∧ 
  (∃ i ∈ ({M, M+1, M+2} : Set ℕ), i % 8 = 0) ∧
  (∃ i ∈ ({M, M+1, M+2} : Set ℕ), i % 9 = 0) ∧
  (∃ i ∈ ({M, M+1, M+2} : Set ℕ), i % 121 = 0) ∧
  (∃ i ∈ ({M, M+1, M+2} : Set ℕ), i % 49 = 0) ∧
  (∀ N < M, ¬(
    (N > 0) ∧
    (∃ i ∈ ({N, N+1, N+2} : Set ℕ), i % 8 = 0) ∧
    (∃ i ∈ ({N, N+1, N+2} : Set ℕ), i % 9 = 0) ∧
    (∃ i ∈ ({N, N+1, N+2} : Set ℕ), i % 121 = 0) ∧
    (∃ i ∈ ({N, N+1, N+2} : Set ℕ), i % 49 = 0)
  )) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_consecutive_l894_89451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l894_89422

-- Define the function f as noncomputable due to the use of real exponentiation
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then Real.exp (Real.log 2 * |x - a|) else x + 1

-- State the theorem
theorem min_value_implies_a_range (a : ℝ) :
  (∀ x, f a 1 ≤ f a x) → 1 ≤ a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l894_89422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l894_89499

theorem problem_statement :
  (∀ a : ℝ, a^2017 > -1 → a > -1) ∨ (∀ x : ℝ, x^2 * Real.tan x^2 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l894_89499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l894_89404

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l894_89404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l894_89439

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for |r| < 1 -/
noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

/-- 
For -1 < b < 1, if T(b)T(-b) = 3600, then T(b) + T(-b) = 480,
where T(r) is the sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ...
-/
theorem geometric_series_sum (b : ℝ) (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 3600) : T b + T (-b) = 480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l894_89439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l894_89492

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  sides : a > 0 ∧ b > 0 ∧ c > 0
  law_of_sines : a / Real.sin A = b / Real.sin B
  opposite_sides : a / Real.sin A = c / Real.sin C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : (Real.sqrt 3 * t.a, t.c) = 3 * (Real.sin t.A, Real.cos t.C)) :
  t.C = π / 3 ∧ 
  (3 * Real.sqrt 3 + 3) / 2 < t.a + t.b + t.c ∧ 
  t.a + t.b + t.c ≤ 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l894_89492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l894_89479

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem symmetry_axis 
  (t ω φ : ℝ) 
  (h1 : Real.cos φ = t / 2)
  (h2 : ω > 0)
  (h3 : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ x₂ - x₁ = π) :
  ∃ (k : ℤ), (π / 12 : ℝ) + k * π / 2 ∈ {x | ∀ y, f ω φ (2 * (π / 12 + k * π / 2) - y) = f ω φ y} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l894_89479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_at_difference_l894_89419

/-- Custom operation @ -/
def custom_at (x y : ℤ) : ℤ := x * y - 3 * x + y

/-- Theorem stating the result of (6@5) - (5@6) -/
theorem custom_at_difference : (custom_at 6 5) - (custom_at 5 6) = -4 := by
  -- Unfold the definition of custom_at
  unfold custom_at
  -- Perform arithmetic
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_at_difference_l894_89419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_matches_given_terms_l894_89453

def sequenceTerm (n : ℕ) : ℕ := n^2 + n

theorem sequence_matches_given_terms :
  sequenceTerm 1 = 1 * 2 ∧
  sequenceTerm 2 = 2 * 3 ∧
  sequenceTerm 3 = 3 * 4 ∧
  sequenceTerm 4 = 4 * 5 := by
  have h1 : sequenceTerm 1 = 1 * 2 := by rfl
  have h2 : sequenceTerm 2 = 2 * 3 := by rfl
  have h3 : sequenceTerm 3 = 3 * 4 := by rfl
  have h4 : sequenceTerm 4 = 4 * 5 := by rfl
  exact ⟨h1, h2, h3, h4⟩

#eval sequenceTerm 1
#eval sequenceTerm 2
#eval sequenceTerm 3
#eval sequenceTerm 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_matches_given_terms_l894_89453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l894_89429

-- Define the arithmetic sequence
noncomputable def a (n : ℕ) : ℝ := 2 * n + 1

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 1 / (a (n - 1) * a n)

-- Define the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := n / (2 * n + 1)

-- Theorem statement
theorem arithmetic_sequence_properties :
  ∃ (d : ℝ), d ≠ 0 ∧
  a 3 + S 5 = 42 ∧
  (∃ (r : ℝ), a 4 = a 1 * r ∧ a 13 = a 4 * r) ∧
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, T n = n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l894_89429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_lambda_l894_89430

/-- A geometric sequence with sum of first n terms S_n = 2^(n+1) + lambda has lambda = -2 --/
theorem geometric_sequence_lambda (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda : ℝ) :
  (∀ n, S n = 2^(n+1) + lambda) →
  (∀ n, a (n+1) = a n * (a 2 / a 1)) →
  (∀ n, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →
  lambda = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_lambda_l894_89430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b3_value_l894_89487

theorem b3_value :
  ∀ b1 q : ℤ,
  b1^2 * (1 + q^2 + q^4) = 2275 →
  q^2 = 9 →
  ∃ b3 : ℤ, b3 = 45 ∨ b3 = -45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b3_value_l894_89487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_prime_power_solutions_l894_89441

def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = (p : ℤ) ^ k

theorem quadratic_prime_power_solutions :
  {x : ℤ | is_prime_power (f x)} = {-3, 2, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_prime_power_solutions_l894_89441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bandit_with_eight_fights_l894_89460

/-- Represents a group of bandits and their fights -/
structure BanditGroup where
  n : ℕ  -- number of bandits
  fights : Fin n → Fin n → Bool  -- fights between bandits

/-- Definition of a valid bandit group -/
def validBanditGroup (g : BanditGroup) : Prop :=
  g.n = 50 ∧
  ∀ i j : Fin g.n, i ≠ j → g.fights i j = true ∧
  ∀ i : Fin g.n, g.fights i i = false

/-- Count of fights for a given bandit -/
def fightCount (g : BanditGroup) (i : Fin g.n) : ℕ :=
  (Finset.univ.filter (λ j => g.fights i j)).card

/-- Theorem: In a valid bandit group, there exists a bandit with at least 8 fights -/
theorem bandit_with_eight_fights (g : BanditGroup) (h : validBanditGroup g) :
  ∃ i : Fin g.n, fightCount g i ≥ 8 := by
  sorry

#check bandit_with_eight_fights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bandit_with_eight_fights_l894_89460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l894_89450

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the domain
def domain : Set ℝ := { x | -1.5 ≤ x ∧ x ≤ 1.5 }

-- Define the solution set
def solution_set : Set ℝ := { x ∈ domain | g (g (g x)) = g x }

-- Theorem statement
theorem solution_count : ∃ (n : ℕ), n = 36 ∧ ∃ (f : solution_set → Fin n), Function.Bijective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l894_89450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mpg_approximation_l894_89486

/-- Calculates the average miles per gallon for a round trip with different fuel efficiencies for each leg -/
noncomputable def average_mpg (total_distance : ℝ) (first_leg_mpg second_leg_mpg : ℝ) : ℝ :=
  total_distance / ((total_distance / 2 / first_leg_mpg) + (total_distance / 2 / second_leg_mpg))

theorem round_trip_mpg_approximation :
  ∀ ε > 0, |average_mpg 300 25 40 - 31| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mpg_approximation_l894_89486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_y_axis_l894_89467

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Theorem: If P(0,y,0) is on the y-axis and its distance from (2,5,-6) is 7, then y = 8 or y = 2 -/
theorem point_on_y_axis (y : ℝ) : 
  distance 0 y 0 2 5 (-6) = 7 → y = 8 ∨ y = 2 := by
  sorry

#check point_on_y_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_y_axis_l894_89467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l894_89475

theorem tan_double_angle_special_case (α : ℝ) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) π) :
  Real.tan (2 * α) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l894_89475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_problem_difference_l894_89423

theorem homework_problem_difference (total_problems : ℕ) 
  (martha_problems : ℕ) (angela_problems : ℕ) (jenna_problems : ℕ) :
  total_problems = 20 →
  martha_problems = 2 →
  angela_problems = 9 →
  jenna_problems + martha_problems + (jenna_problems / 2) + angela_problems = total_problems →
  Int.natAbs (jenna_problems - 4 * martha_problems) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_problem_difference_l894_89423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operating_days_rounded_capacity_l894_89452

-- Define the start and end dates
def start_date : Nat := 24  -- November 24
def end_date : Nat := 18    -- December 18
def days_in_november : Nat := 30

-- Define the number to be rounded
def transport_capacity : Nat := 287000

-- Theorem for the number of operating days
theorem operating_days :
  (days_in_november - start_date) + end_date = 25 := by sorry

-- Theorem for rounding to nearest ten thousand
theorem rounded_capacity :
  (transport_capacity / 10000 + 1) * 10000 = 290000 := by sorry

#check operating_days
#check rounded_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operating_days_rounded_capacity_l894_89452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_hexagons_l894_89434

def is_valid_hexagon (sides : List ℕ) : Prop :=
  sides.length = 6 ∧
  sides.sum = 20 ∧
  ∀ i j k, i < sides.length ∧ j < sides.length ∧ k < sides.length →
    i ≠ j ∧ j ≠ k ∧ i ≠ k →
    sides[i]! + sides[j]! > sides[k]!

theorem infinitely_many_valid_hexagons :
  ∃ f : ℕ → List ℕ, ∀ n : ℕ,
    is_valid_hexagon (f n) ∧
    ∀ m : ℕ, m ≠ n → f m ≠ f n := by
  sorry

#check infinitely_many_valid_hexagons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_hexagons_l894_89434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cookie_purchase_l894_89425

/-- The maximum number of cookies that can be bought given a total amount of money and the cost per cookie -/
def max_cookies (total_money : ℚ) (cost_per_cookie : ℚ) : ℕ :=
  (total_money / cost_per_cookie).floor.toNat

/-- Theorem stating that given $24.75 in total money and $2.25 per cookie, the maximum number of cookies that can be bought is 11 -/
theorem john_cookie_purchase :
  max_cookies (24 + 75/100) (2 + 25/100) = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cookie_purchase_l894_89425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_max_min_equality_l894_89494

/-- Given distinct real numbers a, b, c, d, e with a < b < c < d < e,
    prove that M(m(b, M(a,d)), M(c, m(e, d))) = d -/
theorem nested_max_min_equality
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e) :
  max (min b (max a d)) (max c (min e d)) = d :=
by sorry

/-- Maximum of two real numbers -/
noncomputable def M (x y : ℝ) : ℝ := max x y

/-- Minimum of two real numbers -/
noncomputable def m (x y : ℝ) : ℝ := min x y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_max_min_equality_l894_89494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l894_89491

/-- Calculates the speed of a train in km/hr given its length and the time it takes to cross a platform of equal length -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := 2 * train_length
  let speed_m_per_min := total_distance / crossing_time
  speed_m_per_min * 60 / 1000

theorem train_speed_calculation :
  train_speed 1050 1 = 35 := by
  unfold train_speed
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l894_89491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lina_cycling_difference_l894_89410

/-- Represents Lina's cycling ability at different ages --/
structure CyclingAbility where
  miles : ℚ
  hours : ℚ

/-- Calculates the minutes per mile for a given cycling ability --/
def minutesPerMile (ability : CyclingAbility) : ℚ :=
  (ability.hours * 60) / ability.miles

/-- Proves the difference in Lina's cycling speed between youth and older adult --/
theorem lina_cycling_difference (youth adult : CyclingAbility)
  (youth_miles : youth.miles = 20)
  (youth_hours : youth.hours = 2)
  (adult_miles : adult.miles = 12)
  (adult_hours : adult.hours = 3) :
  minutesPerMile adult - minutesPerMile youth = 9 := by
  sorry

#check lina_cycling_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lina_cycling_difference_l894_89410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_curve_a_value_l894_89476

-- Define the function f as noncomputable due to its dependence on Real.sqrt
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => Real.sqrt (-x - a)

-- Define the theorem
theorem symmetric_curve_a_value :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → f a x = Real.sqrt (-x - a)) →
  f a (-2) = 2 * f a (-1) →
  a = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_curve_a_value_l894_89476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_lower_bound_l894_89469

theorem polynomial_degree_lower_bound
  (N : ℕ)
  (h_prime : Nat.Prime (N + 1))
  (a : Fin (N + 1) → Fin 2)
  (h_not_all_same : ∃ i j : Fin (N + 1), a i ≠ a j)
  (f : Polynomial ℝ)
  (h_interpolation : ∀ i : Fin (N + 1), f.eval (i : ℝ) = (a i : ℝ)) :
  Polynomial.degree f ≥ N := by
  sorry

#check polynomial_degree_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_lower_bound_l894_89469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_has_two_elements_l894_89480

theorem set_has_two_elements (A : Set ℕ) : 
  (∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ a ≠ b) →
  (∀ a b : ℕ, a ∈ A → b ∈ A → a > b → (Nat.lcm a b) / (a - b) ∈ A) →
  ∃! a b : ℕ, A = {a, b} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_has_two_elements_l894_89480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsets_count_l894_89403

-- Define the set S
def S : Finset Nat := Finset.range 15

-- Define the properties of the subsets
def valid_subsets (A : Finset (Finset Nat)) : Prop :=
  ∀ X ∈ A,
    (X.card = 7) ∧
    (∀ Y ∈ A, Y ≠ X → (X ∩ Y).card ≤ 3) ∧
    (∀ M : Finset Nat, M ⊆ S → M.card = 3 → ∃ X ∈ A, M ⊆ X)

-- Theorem statement
theorem min_subsets_count :
  ∃ n : Nat, n = 15 ∧
    (∃ A : Finset (Finset Nat), A.card = n ∧ valid_subsets A) ∧
    (∀ m : Nat, m < 15 → ¬∃ A : Finset (Finset Nat), A.card = m ∧ valid_subsets A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsets_count_l894_89403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l894_89407

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_of_sequence : ℕ := 
  (List.range 10).map (fun i => sequence_term (i + 1)) |> List.sum

theorem units_digit_of_sum :
  sum_of_sequence % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l894_89407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_seven_l894_89414

-- Define the function g (we don't know its exact form, so we leave it as a parameter)
noncomputable def g : ℝ → ℝ := sorry

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 - 1 else g x

-- State that f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- State the theorem
theorem f_sum_equals_seven : f (-1) + g 2 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_seven_l894_89414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_imply_omega_range_l894_89498

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (2 * ω * x + Real.pi / 3)

theorem zeros_imply_omega_range (ω : ℝ) :
  ω > 0 →
  (∃ (s : Finset ℝ), s.card = 10 ∧ 
    (∀ x ∈ s, 0 < x ∧ x < 2*Real.pi ∧ f ω x = 0) ∧
    (∀ y, 0 < y ∧ y < 2*Real.pi ∧ f ω y = 0 → y ∈ s)) →
  55/24 < ω ∧ ω ≤ 61/24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_imply_omega_range_l894_89498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_composition_equals_one_l894_89401

noncomputable def s (θ : ℝ) : ℝ := 1 / (2 - θ)

theorem s_composition_equals_one :
  s (s (s (s (s (s 5))))) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_composition_equals_one_l894_89401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_l894_89431

/-- The price of 100 apples in cents -/
def P : ℕ := sorry

/-- The condition that if the price of 100 apples increases by 4 cents,
    the number of apples bought with 120 cents decreases by 5 -/
axiom price_condition : (120 * 100) / P - (120 * 100) / (P + 4) = 5

/-- Theorem stating that the price of 100 apples is 96 cents -/
theorem apple_price : P = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_l894_89431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_value_l894_89468

/-- A line passing through given points -/
structure Line where
  points : List (ℝ × ℝ)
  is_line : ∀ p q r, p ∈ points → q ∈ points → r ∈ points → 
    (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

/-- Theorem: For a line passing through (2, 8), (5, 17), (8, 26), and (34, t), t = 104 -/
theorem line_point_value (l : Line) 
    (h1 : (2, 8) ∈ l.points) 
    (h2 : (5, 17) ∈ l.points)
    (h3 : (8, 26) ∈ l.points) 
    (h4 : ∃ t, (34, t) ∈ l.points) : 
    ∃ t, (34, t) ∈ l.points ∧ t = 104 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_value_l894_89468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_diligence_l894_89496

/-- The number of students in section Diligence before the transfer -/
def D : ℕ := sorry

/-- The number of students in section Industry before the transfer -/
def I : ℕ := sorry

/-- The total number of students in both sections is 50 -/
axiom total_students : D + I = 50

/-- After transferring 2 students from Industry to Diligence, both sections have an equal number of students -/
axiom equal_after_transfer : D + 2 = I - 2

theorem students_in_diligence : D = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_diligence_l894_89496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gold_stars_l894_89405

def gold_stars_problem (yesterday_stars : ℕ) (today_multiplier : ℚ) : ℚ :=
  (yesterday_stars : ℚ) + (yesterday_stars : ℚ) * today_multiplier

theorem total_gold_stars :
  ⌊gold_stars_problem 4 (7/2)⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gold_stars_l894_89405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l894_89478

theorem equation_solution : ∃ x : ℚ, (3 : ℝ) ^ ((4 : ℝ) * x^2 - (9 : ℝ) * x + 5) = (3 : ℝ) ^ ((4 : ℝ) * x^2 + (3 : ℝ) * x - 7) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l894_89478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l894_89463

theorem logarithm_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) :
  (Real.log (x^2) / Real.log (y^10)) * (Real.log (y^3) / Real.log (x^7)) * 
  (Real.log (x^4) / Real.log (y^8)) * (Real.log (y^6) / Real.log (x^9)) * 
  (Real.log (x^11) / Real.log (y^5)) = (1 / 15) * (Real.log x / Real.log y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l894_89463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l894_89464

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (4*x - 5)

theorem f_at_3 : f 3 = 16/7 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the numerator and denominator
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l894_89464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_equivalence_l894_89408

-- Define the domain of x
def X : Set ℝ := Set.Icc (-2) 2

-- Define the function f
noncomputable def f : X → Set.Icc 0 4 := sorry

-- Define the function g
def g (a : ℝ) : X → ℝ := fun x ↦ a * (x : ℝ) - 1

-- State the theorem
theorem function_range_equivalence (a : ℝ) :
  (∀ x₁ : X, ∃ x₀ : X, g a x₀ = f x₁) ↔ a ∈ Set.Iic (-5/2) ∪ Set.Ici (5/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_equivalence_l894_89408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_l894_89440

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := 2*x + y - 6 = 0

/-- Point P on curve C₁ -/
noncomputable def point_P : ℝ × ℝ := (2 * Real.sqrt 5 / 5, Real.sqrt 5 / 5)

/-- Distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2*x + y - 6) / Real.sqrt 5

theorem minimum_distance_point :
  C₁ point_P.1 point_P.2 ∧
  (∀ x y, C₁ x y → distance_to_line x y ≥ distance_to_line point_P.1 point_P.2) ∧
  distance_to_line point_P.1 point_P.2 = 6 * Real.sqrt 5 / 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_l894_89440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_gray_field_bounds_l894_89445

/-- Represents a triangle with numbers on its sides -/
structure Triangle where
  left_top : ℕ
  right_top : ℕ
  bottom : ℕ
  left_bottom : ℕ
  right_bottom : ℕ
  gray_field : ℕ

/-- The product of numbers on each side of the triangle is equal -/
def valid_triangle (t : Triangle) : Prop :=
  t.left_top * t.left_bottom * t.gray_field = 
    t.right_top * t.right_bottom * t.gray_field ∧
  t.left_top * t.left_bottom * t.gray_field = 
    t.left_bottom * t.bottom * t.right_bottom

/-- The theorem stating the smallest and largest possible values for the gray field -/
theorem triangle_gray_field_bounds :
  ∀ t : Triangle, 
    t.left_top = 16 ∧ 
    t.right_top = 72 ∧ 
    t.bottom = 42 ∧ 
    valid_triangle t → 
    (1 ≤ t.gray_field ∧ t.gray_field ≤ 336 ∧ 
     ∃ t1 t2 : Triangle, 
       t1.gray_field = 1 ∧ 
       t2.gray_field = 336 ∧ 
       valid_triangle t1 ∧ 
       valid_triangle t2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_gray_field_bounds_l894_89445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l894_89472

-- Define the circle M
noncomputable def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define a point P on line l
noncomputable def point_P (x y : ℝ) : Prop := line_l x y

-- Define the fixed point of intersection
noncomputable def fixed_point : ℝ × ℝ := (1/2, 15/4)

-- Theorem statement
theorem fixed_intersection_point :
  ∀ (x y : ℝ), point_P x y →
  ∃ (a b : ℝ), circle_M a b ∧
  (∀ (c d : ℝ), circle_M c d →
    ∃ (k : ℝ), (c - x)^2 + (d - y)^2 = k * ((c - 0)^2 + (d - 4)^2)) →
  circle_M (fixed_point.1) (fixed_point.2) := by
  sorry

#check fixed_intersection_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l894_89472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_walk_distance_l894_89415

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points --/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Sophia's walk from A to B --/
noncomputable def sophia_walk (start : Point) : Point :=
  let p1 := Point.mk start.x (start.y - 50)  -- 50 yards south
  let p2 := Point.mk (p1.x - 80) p1.y        -- 80 yards west
  let p3 := Point.mk p2.x (p2.y + 20)        -- 20 yards north
  let p4 := Point.mk (p3.x + 30) p3.y        -- 30 yards east
  Point.mk (p4.x - 10 * Real.sqrt 2 / 2) (p4.y - 10 * Real.sqrt 2 / 2)  -- 10 yards southwest

theorem sophia_walk_distance (A : Point) :
  let B := sophia_walk A
  abs (distance A B - 68.06) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_walk_distance_l894_89415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_contrapositive_equivalence_l894_89481

theorem inverse_contrapositive_equivalence (α : Real) :
  (α = π / 4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_contrapositive_equivalence_l894_89481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l894_89493

noncomputable def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

noncomputable def leftFocus (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 + b^2), 0)
noncomputable def rightFocus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

theorem hyperbola_eccentricity (a b c : ℝ) (P : ℝ × ℝ) (A E M : ℝ × ℝ) 
  (h1 : a > 0) (h2 : b > 0)
  (h3 : P ∈ Hyperbola a b)
  (h4 : P.1 > 0 ∧ P.2 > 0)
  (h5 : (P.2 - (rightFocus a b).2) / (P.1 - (rightFocus a b).1) = 
        -((rightFocus a b).1 - (leftFocus a b).1) / ((rightFocus a b).2 - (leftFocus a b).2))
  (h6 : A.2 = 0)
  (h7 : (P.2 - A.2) / (P.1 - A.1) = -(P.1 - (leftFocus a b).1) / (P.2 - (leftFocus a b).2))
  (h8 : E = ((P.1 + A.1) / 2, (P.2 + A.2) / 2))
  (h9 : ∃ t s, M = (1-t) • E + t • (leftFocus a b) ∧ 
              M = (1-s) • P + s • (rightFocus a b) ∧ 0 < t ∧ t < 1 ∧ 0 < s ∧ s < 1)
  (h10 : (P.1 - M.1)^2 + (P.2 - M.2)^2 = 4 * ((rightFocus a b).1 - M.1)^2 + 4 * ((rightFocus a b).2 - M.2)^2)
  : eccentricity a b = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l894_89493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_constraint_l894_89438

/-- A line in the xy-plane -/
noncomputable def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sqrt 3 * p.1 + p.2 - Real.sqrt 3 * m = 0}

/-- A circle in the xy-plane centered at (1, 0) with radius √3 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 3}

/-- The distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the relationship between m and the chord length -/
theorem chord_length_constraint (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Line m ∩ Circle ∧ B ∈ Line m ∩ Circle ∧ distance A B ≥ 3) →
  0 ≤ m ∧ m ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_constraint_l894_89438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_of_tangency_triangle_is_incenter_l894_89497

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define an incircle
structure Incircle where
  center : Point
  radius : ℝ

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : Point :=
  { x := 0, y := 0 } -- Placeholder implementation

-- Define the incenter of a triangle
noncomputable def incenter (t : Triangle) : Point :=
  { x := 0, y := 0 } -- Placeholder implementation

-- Define the points of tangency of an incircle with a triangle
noncomputable def tangency_points (t : Triangle) (i : Incircle) : Triangle :=
  { A := { x := 0, y := 0 },
    B := { x := 0, y := 0 },
    C := { x := 0, y := 0 } } -- Placeholder implementation

-- Theorem statement
theorem orthocenter_of_tangency_triangle_is_incenter (t : Triangle) (i : Incircle) :
  orthocenter (tangency_points t i) = incenter t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_of_tangency_triangle_is_incenter_l894_89497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l894_89446

noncomputable def f (x y : ℝ) : ℝ := Real.sqrt ((1 + x * y) / (1 + x^2)) + Real.sqrt ((1 - x * y) / (1 + y^2))

theorem f_range (x y : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) : 
  f x y ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l894_89446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_time_l894_89485

/-- The time (in hours) for a steamboat to travel from Upper to Lower Vasyuki -/
noncomputable def steamboat_time : ℝ := 1

/-- The time (in hours) for a motorboat to travel from Upper to Lower Vasyuki -/
noncomputable def motorboat_time : ℝ := 3/4

/-- The ratio of motorboat speed to steamboat speed in still water -/
noncomputable def speed_ratio : ℝ := 2

/-- Theorem: The time for a raft to float downstream from Upper to Lower Vasyuki is 90 minutes -/
theorem raft_time :
  ∃ (distance : ℝ) (steamboat_speed motorboat_speed current_speed : ℝ),
    steamboat_speed + current_speed = distance / steamboat_time ∧
    motorboat_speed + current_speed = distance / motorboat_time ∧
    motorboat_speed = speed_ratio * steamboat_speed ∧
    distance / current_speed = 3/2 := by
  sorry

#check raft_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_time_l894_89485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_light_intersection_l894_89459

/-- Particle's x-coordinate as a function of time -/
def x (t : ℝ) : ℝ := 3

/-- Particle's y-coordinate as a function of time -/
noncomputable def y (t : ℝ) : ℝ := 3 + Real.sin t * Real.cos t - Real.sin t - Real.cos t

/-- Light ray equation -/
def light_ray (c : ℝ) (x : ℝ) : ℝ := c * x

/-- The set of c values for which the particle never intersects the light ray -/
def no_intersection_set : Set ℝ :=
  {c | c > 0 ∧ (c < 3/2 ∨ c > (7 + 2 * Real.sqrt 2) / 6)}

theorem particle_light_intersection (c : ℝ) :
  (∀ t : ℝ, y t ≠ light_ray c (x t)) ↔ c ∈ no_intersection_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_light_intersection_l894_89459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_approx_l894_89465

/-- Represents the number of students in each grade --/
structure StudentCounts where
  kindergarten : Nat
  first : Nat
  second : Nat
  third : Nat
  fourth : Nat
  fifth : Nat

/-- Represents the time in minutes for each type of test --/
structure TestTimes where
  lice : Nat
  vision : Nat
  hearing : Nat

/-- Calculates the total time in hours for all checks and tests --/
noncomputable def totalTimeInHours (students : StudentCounts) (times : TestTimes) : Real :=
  let totalStudents := students.kindergarten + students.first + students.second +
                       students.third + students.fourth + students.fifth
  let timePerStudent := times.lice + times.vision + times.hearing
  let totalMinutes := totalStudents * timePerStudent
  (totalMinutes : Real) / 60

/-- Theorem stating that the total time for all checks and tests is approximately 17.27 hours --/
theorem total_time_approx (students : StudentCounts) (times : TestTimes) :
  students.kindergarten = 26 →
  students.first = 19 →
  students.second = 20 →
  students.third = 25 →
  students.fourth = 30 →
  students.fifth = 28 →
  times.lice = 2 →
  times.vision = 2 →
  times.hearing = 3 →
  abs (totalTimeInHours students times - 17.27) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_approx_l894_89465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_negative_three_fourths_l894_89432

/-- The function f(x) = (2 - 3x) / (4x + 5) -/
noncomputable def f (x : ℝ) : ℝ := (2 - 3*x) / (4*x + 5)

/-- The theorem stating that there is no x such that f(x) = -3/4 -/
theorem no_solution_for_negative_three_fourths :
  ∀ x : ℝ, x ≠ -5/4 → f x ≠ -3/4 := by
  sorry

#check no_solution_for_negative_three_fourths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_negative_three_fourths_l894_89432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowing_rate_calculation_l894_89474

/-- Simple interest calculation function -/
def simpleInterest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  (principal * rate * time) / 100

theorem borrowing_rate_calculation
  (principal : ℕ) (time : ℕ) (lending_rate : ℚ) (gain_per_year : ℕ) :
  principal = 5000 →
  time = 2 →
  lending_rate = 5 →
  gain_per_year = 50 →
  ∃ (borrowing_rate : ℚ),
    borrowing_rate = 9 ∧
    simpleInterest principal lending_rate time -
    simpleInterest principal borrowing_rate time =
    (gain_per_year : ℚ) * time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowing_rate_calculation_l894_89474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l894_89456

def A : Set ℤ := {x : ℤ | |x| < 5}
def B : Set ℤ := {x : ℤ | x ≥ 2}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l894_89456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_term_is_five_l894_89443

def next_term (n : Int) : Int :=
  if n < 7 then n * 8
  else if n % 2 = 0 then n / 3
  else n - 4

def sequence_term (n : Nat) : Int :=
  match n with
  | 0 => 53
  | n + 1 => next_term (sequence_term n)

theorem fifty_third_term_is_five :
  sequence_term 52 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_term_is_five_l894_89443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_tape_coloring_l894_89433

theorem min_colors_for_tape_coloring (n : ℕ) (x y : ℕ) 
  (h_distinct : x ≠ y) (h_x_pos : x > 0) (h_y_pos : y > 0) (h_x_le_n : x ≤ n) (h_y_le_n : y ≤ n) :
  ∃ (f : ℕ → Fin 3), ∀ (i j : ℕ), i ≤ n^2 ∧ j ≤ n^2 → 
    (i - j = x ∨ j - i = x ∨ i - j = y ∨ j - i = y) → f i ≠ f j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_tape_coloring_l894_89433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribe_sphere_inscribe_sphere_l894_89435

-- Define a truncated circular cone
structure TruncatedCone where
  r1 : ℝ  -- radius of the bottom base
  r2 : ℝ  -- radius of the top base
  h : ℝ   -- height of the truncated cone
  l : ℝ   -- slant height

-- Define what it means to be a sphere
def IsSphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop :=
  sorry

-- Define what it means for a sphere to circumscribe a cone
def CircumscribesCone (center : ℝ × ℝ × ℝ) (radius : ℝ) (cone : TruncatedCone) : Prop :=
  sorry

-- Define what it means for a sphere to inscribe a cone
def InscribesCone (center : ℝ × ℝ × ℝ) (radius : ℝ) (cone : TruncatedCone) : Prop :=
  sorry

-- Theorem for circumscribing a sphere
theorem circumscribe_sphere (cone : TruncatedCone) : 
  ∃ (center : ℝ × ℝ × ℝ) (radius : ℝ), 
    IsSphere center radius ∧ CircumscribesCone center radius cone :=
by
  sorry

-- Theorem for inscribing a sphere
theorem inscribe_sphere (cone : TruncatedCone) : 
  (∃ (center : ℝ × ℝ × ℝ) (radius : ℝ), 
    IsSphere center radius ∧ InscribesCone center radius cone) ↔ 
  2 * cone.r1 + 2 * cone.r2 = 2 * cone.l :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribe_sphere_inscribe_sphere_l894_89435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_x_value_l894_89489

/-- Given two vectors PA and PB in ℝ², prove that if PA = (-1, 2), PB = (2, x),
    and the points P, A, and B are collinear, then x = -4 -/
theorem collinear_vectors_x_value (PA PB : ℝ × ℝ) (x : ℝ) :
  PA = (-1, 2) →
  PB = (2, x) →
  ∃ (k : ℝ), PA = k • PB →
  x = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_x_value_l894_89489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_comparison_l894_89470

/-- Given a surface area S, calculates the volume of a cube with that surface area -/
noncomputable def cube_volume (S : ℝ) : ℝ := (S^(3/2)) / (6 * Real.sqrt 6)

/-- Given a surface area S, calculates the volume of a right circular cylinder 
    with equal height and diameter and that surface area -/
noncomputable def cylinder_volume (S : ℝ) : ℝ := Real.sqrt (S^3 / (54 * Real.pi))

/-- Given a surface area S, calculates the volume of a sphere with that surface area -/
noncomputable def sphere_volume (S : ℝ) : ℝ := Real.sqrt (S^3 / (36 * Real.pi))

/-- Theorem stating that for any positive real surface area S, 
    the volumes of a cube, cylinder, and sphere with that surface area 
    satisfy the inequality V₁ < V₂ < V₃ -/
theorem volume_comparison (S : ℝ) (h : S > 0) : 
  cube_volume S < cylinder_volume S ∧ cylinder_volume S < sphere_volume S :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_comparison_l894_89470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l894_89428

def m : ℕ := 2^5 * 3^7 * 5^4

theorem number_of_factors_of_m : 
  (Finset.filter (fun x => x ∣ m) (Finset.range (m + 1))).card = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l894_89428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_iff_x_eq_neg_three_l894_89461

def a : ℝ × ℝ × ℝ := (6, -2, 6)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-3, 1, x)

theorem vectors_parallel_iff_x_eq_neg_three :
  (∃ (k : ℝ), b (-3) = (k * a.1, k * a.2.1, k * a.2.2)) ↔
  ∀ (x : ℝ), (∃ (k : ℝ), b x = (k * a.1, k * a.2.1, k * a.2.2)) → x = -3 := by
  sorry

#check vectors_parallel_iff_x_eq_neg_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_iff_x_eq_neg_three_l894_89461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l894_89426

theorem cone_lateral_surface_area (m r : ℝ) (hm : m > 0) (hr : r > 0) :
  let lateral_area := π * r * m
  lateral_area = π * r * m := by
  -- Define the lateral area
  let lateral_area := π * r * m
  
  -- The proof is straightforward as we're proving an equality to itself
  rfl

  -- Note: This proof doesn't include the geometric reasoning, which would require more steps
  -- The actual geometric proof would involve showing that the lateral surface unfolds to a sector
  -- and then calculating its area. For a complete proof, we'd need to define more concepts and prove more lemmas.

-- Example usage:
#check cone_lateral_surface_area

-- You can add more theorems or calculations here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l894_89426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_light_most_probable_l894_89402

/-- Proves that the green light has the highest probability of being encountered at an intersection --/
theorem green_light_most_probable (red_duration yellow_duration green_duration : ℕ)
  (h_red : red_duration = 30)
  (h_yellow : yellow_duration = 5)
  (h_green : green_duration = 40) :
  let total_duration := red_duration + yellow_duration + green_duration
  let p_red := (red_duration : ℚ) / total_duration
  let p_yellow := (yellow_duration : ℚ) / total_duration
  let p_green := (green_duration : ℚ) / total_duration
  p_green > p_red ∧ p_green > p_yellow :=
by
  -- Convert natural numbers to rationals for division
  have total_duration : ℚ := (red_duration + yellow_duration + green_duration : ℚ)
  
  -- Calculate probabilities
  have p_red : ℚ := red_duration / total_duration
  have p_yellow : ℚ := yellow_duration / total_duration
  have p_green : ℚ := green_duration / total_duration
  
  -- Prove that green light has the highest probability
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_light_most_probable_l894_89402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l894_89457

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 8) / Real.log 3

theorem monotonic_decreasing_interval_of_f :
  ∃ (a : ℝ), a = -2 ∧ 
  (∀ x y, x < y ∧ y < a → f x > f y) ∧
  (∀ ε > 0, ∃ x y, a < x ∧ x < y ∧ y < a + ε ∧ f x < f y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l894_89457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_vector_perpendicular_iff_negation_equivalence_zero_existence_inverse_zero_existence_inverse_false_l894_89418

-- Define the necessary types and functions
def N_star := {n : ℕ | n > 0}

-- Statement A
theorem ln_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by sorry

-- Statement B
def vector_perpendicular (m : ℝ) : Prop :=
  let a := (1, m)
  let b := (m, 2*m - 1)
  a.1 * b.1 + a.2 * b.2 = 0

theorem vector_perpendicular_iff (m : ℝ) : vector_perpendicular m ↔ m = 0 := by sorry

-- Statement C
def inequality_prop (n : ℕ) : Prop := 3^n > (n + 2) * 2^(n - 1)

theorem negation_equivalence :
  (∀ n ∈ N_star, inequality_prop n) ↔ ¬(∃ n ∈ N_star, ¬(inequality_prop n)) := by sorry

-- Statement D
theorem zero_existence_inverse (f : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn f (Set.Icc a b)) :
  (∃ x ∈ Set.Ioo a b, f x = 0) → f a * f b < 0 := by sorry

theorem zero_existence_inverse_false : ∃ f : ℝ → ℝ, ∃ a b : ℝ,
  ContinuousOn f (Set.Icc a b) ∧ (∃ x ∈ Set.Ioo a b, f x = 0) ∧ f a * f b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_vector_perpendicular_iff_negation_equivalence_zero_existence_inverse_zero_existence_inverse_false_l894_89418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_rectangle_l894_89416

theorem equilateral_triangle_from_rectangle :
  ∀ (rectangle_length rectangle_width : ℝ) (triangle_side : ℝ),
    rectangle_length = 20 →
    rectangle_width = 10 →
    (rectangle_length * rectangle_width) = (Real.sqrt 3 / 4) * triangle_side^2 →
    triangle_side = 20 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_rectangle_l894_89416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_at_x_50_l894_89436

/-- A straight line passing through three given points -/
structure StraightLine where
  m : ℝ  -- Slope of the line
  b : ℝ  -- y-intercept of the line
  /-- The line passes through (2, 11) -/
  point1 : 11 = m * 2 + b
  /-- The line passes through (6, 23) -/
  point2 : 23 = m * 6 + b
  /-- The line passes through (10, 35) -/
  point3 : 35 = m * 10 + b

/-- Theorem stating that for the given straight line, y = 155 when x = 50 -/
theorem y_value_at_x_50 (line : StraightLine) : 
  155 = line.m * 50 + line.b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_at_x_50_l894_89436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radii_l894_89473

/-- Three circles inscribed in an angle -/
structure InscribedCircles where
  /-- Radius of the small circle -/
  r : ℝ
  /-- Distance from the center of the small circle to the vertex of the angle -/
  a : ℝ
  /-- The large circle passes through the center of the medium circle -/
  large_through_medium : Bool
  /-- The medium circle passes through the center of the small circle -/
  medium_through_small : Bool

/-- Radii of the medium and large circles -/
noncomputable def circle_radii (c : InscribedCircles) : ℝ × ℝ :=
  (c.a * c.r / (c.a - c.r), c.a^2 * c.r / (c.a - c.r)^2)

theorem inscribed_circles_radii (c : InscribedCircles) 
  (h1 : c.large_through_medium = true) 
  (h2 : c.medium_through_small = true) 
  (h3 : c.r > 0) 
  (h4 : c.a > c.r) : 
  circle_radii c = (c.a * c.r / (c.a - c.r), c.a^2 * c.r / (c.a - c.r)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radii_l894_89473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l894_89466

/-- Piecewise function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -2 * x^2 + a * x - 2 else x - 1

/-- The theorem stating the relationship between the range of f and the range of a -/
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  (a ≤ -4 ∨ a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l894_89466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_bounds_l894_89417

noncomputable def f (x : ℝ) : ℝ := 3^(abs x)

theorem interval_length_bounds (a b : ℝ) (h1 : a ≤ b) (h2 : Set.Icc a b = f ⁻¹' Set.Icc 1 9) :
  (∃ (c d : ℝ), c ≤ d ∧ Set.Icc c d = f ⁻¹' Set.Icc 1 9 ∧ d - c = 4) ∧
  (∃ (e g : ℝ), e ≤ g ∧ Set.Icc e g = f ⁻¹' Set.Icc 1 9 ∧ g - e = 2) ∧
  (∀ (x y : ℝ), x ≤ y → Set.Icc x y = f ⁻¹' Set.Icc 1 9 → y - x ≤ 4) ∧
  (∀ (x y : ℝ), x ≤ y → Set.Icc x y = f ⁻¹' Set.Icc 1 9 → y - x ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_bounds_l894_89417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceva_implies_mass_distribution_l894_89400

/-- Ceva's theorem and mass distribution in a triangle -/
theorem ceva_implies_mass_distribution (A B C : EuclideanSpace ℝ (Fin 2)) 
  (D : EuclideanSpace ℝ (Fin 2)) -- Point on BC
  (E : EuclideanSpace ℝ (Fin 2)) -- Point on CA
  (F : EuclideanSpace ℝ (Fin 2)) -- Point on AB
  (h_ceva : (dist A F / dist F B) * (dist B D / dist D C) * (dist C E / dist E A) = 1) :
  ∃ (m₁ m₂ m₃ : ℝ), m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0 ∧
    (dist A D / dist D B = m₂ / m₁) ∧
    (dist B E / dist E C = m₃ / m₂) ∧
    (dist C F / dist F A = m₁ / m₃) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceva_implies_mass_distribution_l894_89400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_swap_time_l894_89482

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ
  h_valid : hours < 24 ∧ minutes < 60 ∧ seconds < 60

/-- Converts time to angle in radians for the hour hand -/
noncomputable def hourAngle (t : Time) : ℝ :=
  2 * Real.pi * (t.hours / 12 + t.minutes / 720 + t.seconds / 43200 : ℝ)

/-- Converts time to angle in radians for the minute hand -/
noncomputable def minuteAngle (t : Time) : ℝ :=
  2 * Real.pi * (t.minutes / 60 + t.seconds / 3600 : ℝ)

/-- Checks if a time is between 3:20 and 3:25 -/
def isBetween320And325 (t : Time) : Prop :=
  (t.hours = 3 ∧ t.minutes ≥ 20 ∧ t.minutes < 25) ∨
  (t.hours = 3 ∧ t.minutes = 25 ∧ t.seconds = 0)

/-- Checks if swapping hour and minute hands results in a valid time -/
def isValidSwap (t : Time) : Prop :=
  ∃ (t' : Time), hourAngle t = minuteAngle t' ∧ minuteAngle t = hourAngle t'

theorem unique_swap_time :
  ∃! (t : Time), isBetween320And325 t ∧ isValidSwap t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_swap_time_l894_89482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_picking_result_l894_89449

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  serezha_pattern : ℕ → ℕ  -- Function mapping picked berries to basket berries
  dima_pattern : ℕ → ℕ     -- Function mapping picked berries to basket berries
  serezha_speed : ℕ
  dima_speed : ℕ

/-- The specific berry picking scenario from the problem -/
def problem_scenario : BerryPicking :=
  { total_berries := 450
  , serezha_pattern := λ n => n / 2
  , dima_pattern := λ n => 2 * n / 3
  , serezha_speed := 2
  , dima_speed := 1 }

/-- Theorem stating the result of the berry picking problem -/
theorem berry_picking_result (bp : BerryPicking) (h : bp = problem_scenario) :
  let serezha_picked := bp.total_berries * bp.serezha_speed / (bp.serezha_speed + bp.dima_speed)
  let dima_picked := bp.total_berries * bp.dima_speed / (bp.serezha_speed + bp.dima_speed)
  let serezha_basket := bp.serezha_pattern serezha_picked
  let dima_basket := bp.dima_pattern dima_picked
  serezha_basket = dima_basket + 50 := by
  sorry

#check berry_picking_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_picking_result_l894_89449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_students_l894_89454

theorem first_class_students (avg_first : ℝ) (avg_second : ℝ) (students_second : ℕ) (avg_combined : ℝ) :
  avg_first = 50 →
  avg_second = 65 →
  students_second = 40 →
  avg_combined = 59.23076923076923 →
  ∃ students_first : ℕ,
    students_first = 25 ∧
    (avg_first * (students_first : ℝ) + avg_second * (students_second : ℝ)) / ((students_first : ℝ) + (students_second : ℝ)) = avg_combined :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_students_l894_89454
