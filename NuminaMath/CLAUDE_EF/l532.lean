import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l532_53292

/-- IsInscribed T C means that triangle T is inscribed in circle C -/
def IsInscribed (T : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop := sorry

/-- ArcLengths T C returns the set of arc lengths created by 
    the vertices of triangle T on circle C -/
def ArcLengths (T : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Set ℝ := sorry

/-- Area T returns the area of triangle T -/
def Area (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given a triangle inscribed in a circle where the vertices divide the circle
    into arcs of lengths 5, 5, and 6, the area of the triangle is 32(2√2 + 2) / π² -/
theorem inscribed_triangle_area (T : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) 
    (h_inscribed : IsInscribed T C)
    (h_arcs : ∃ (a b c : ℝ), ArcLengths T C = {a, b, c} ∧ a = 5 ∧ b = 5 ∧ c = 6) :
    Area T = 32 * (2 * Real.sqrt 2 + 2) / Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l532_53292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_l532_53278

/-- The atomic mass of Aluminum in g/mol -/
noncomputable def atomic_mass_Al : ℝ := 26.98

/-- The atomic mass of Carbon in g/mol -/
noncomputable def atomic_mass_C : ℝ := 12.01

/-- The atomic mass of Oxygen in g/mol -/
noncomputable def atomic_mass_O : ℝ := 16.00

/-- The number of Aluminum atoms in Aluminum carbonate -/
def num_Al : ℕ := 2

/-- The number of Carbon atoms in Aluminum carbonate -/
def num_C : ℕ := 3

/-- The number of Oxygen atoms in Aluminum carbonate -/
def num_O : ℕ := 9

/-- The molar mass of Aluminum carbonate in g/mol -/
noncomputable def molar_mass_Al2CO3 : ℝ :=
  num_Al * atomic_mass_Al + num_C * atomic_mass_C + num_O * atomic_mass_O

/-- The mass of Oxygen in one mole of Aluminum carbonate in g -/
noncomputable def mass_O_in_Al2CO3 : ℝ := num_O * atomic_mass_O

/-- The mass percentage of Oxygen in Aluminum carbonate -/
noncomputable def mass_percentage_O : ℝ := (mass_O_in_Al2CO3 / molar_mass_Al2CO3) * 100

/-- Theorem stating that the mass percentage of Oxygen in Aluminum carbonate is approximately 61.54% -/
theorem mass_percentage_O_approx :
  ∃ ε > 0, |mass_percentage_O - 61.54| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_l532_53278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_trajectory_equations_l532_53265

/-- Given an ellipse C with the following properties:
    - General equation: x²/a² + y²/b² = 1 where a > b > 0
    - Passes through point (√2, 1)
    - Eccentricity is √2/2
    - Points M and N are on the ellipse
    - Product of slopes of OM and ON is -1/2
    - Point P satisfies OP = OM + 2ON
    Prove that:
    1. The equation of ellipse C is x²/4 + y²/2 = 1
    2. The trajectory equation of point P is x²/20 + y²/10 = 1 -/
theorem ellipse_and_trajectory_equations 
  (C : Set (ℝ × ℝ))
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_C : ∀ x y, (x, y) ∈ C ↔ x^2/a^2 + y^2/b^2 = 1)
  (h_point : (Real.sqrt 2, 1) ∈ C)
  (h_eccentricity : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2)
  (M N : ℝ × ℝ)
  (h_M_on_C : M ∈ C)
  (h_N_on_C : N ∈ C)
  (h_slopes : (M.2 / M.1) * (N.2 / N.1) = -1/2)
  (P : ℝ × ℝ)
  (h_P : P.1 = M.1 + 2*N.1 ∧ P.2 = M.2 + 2*N.2) :
  (∀ x y, (x, y) ∈ C ↔ x^2/4 + y^2/2 = 1) ∧
  (∀ x y, (x, y) = P ↔ x^2/20 + y^2/10 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_trajectory_equations_l532_53265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_sum_l532_53223

open Real

theorem sin_angle_sum (α : ℝ) (f : ℝ → ℝ) :
  (f = λ x ↦ sin (x + π/6)) →
  (cos α = 3/5) →
  (0 < α) →
  (α < π/2) →
  f (α + π/12) = 7 * sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_sum_l532_53223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_train_observation_time_l532_53212

/-- The length of the high-speed train "EMU" in meters -/
noncomputable def emu_length : ℝ := 80

/-- The length of the regular train in meters -/
noncomputable def regular_length : ℝ := 100

/-- The time in seconds it takes for a passenger on the high-speed train to observe the regular train passing by -/
noncomputable def emu_observation_time : ℝ := 5

/-- The relative speed of the trains (meters per second) -/
noncomputable def relative_speed : ℝ := regular_length / emu_observation_time

theorem regular_train_observation_time :
  (emu_length / relative_speed) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_train_observation_time_l532_53212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l532_53237

/-- The distance between the intersection points of y = 2x - 1 and (x-1)² + (y+2)² = 5 is 2√2 -/
theorem intersection_distance :
  let line := λ x : ℝ => 2 * x - 1
  let circle := λ x y : ℝ => (x - 1)^2 + (y + 2)^2 = 5
  let intersection_points := { p : ℝ × ℝ | line p.1 = p.2 ∧ circle p.1 p.2 }
  ∃ p₁ p₂, p₁ ∈ intersection_points ∧ p₂ ∈ intersection_points ∧
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l532_53237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_estimate_l532_53279

/-- Represents the result of a germination experiment -/
structure GerminationExperiment where
  n : ℕ  -- number of grains
  m : ℕ  -- number of germinations
  deriving Repr

/-- Calculates the germination rate for an experiment -/
noncomputable def germinationRate (exp : GerminationExperiment) : ℚ :=
  exp.m / exp.n

/-- A sequence of germination experiments with increasing sample sizes -/
def ExperimentSequence := ℕ → GerminationExperiment

/-- The condition that the sequence has increasing sample sizes -/
def hasIncreasingSamples (seq : ExperimentSequence) : Prop :=
  ∀ i j : ℕ, i < j → (seq i).n < (seq j).n

/-- The germination rates of the sequence converge to a limit -/
def ratesConverge (seq : ExperimentSequence) (limit : ℚ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |germinationRate (seq n) - limit| < ε

/-- Estimated germination probability based on the experiment sequence -/
noncomputable def estimatedGerminationProbability (seq : ExperimentSequence) : ℚ :=
  -- This is a placeholder implementation
  -- In practice, this would involve some statistical analysis
  germinationRate (seq 1000)  -- Using the 1000th experiment as an estimate

/-- The main theorem: if rates converge, the limit is the estimated probability -/
theorem germination_probability_estimate 
    (seq : ExperimentSequence) 
    (increasing : hasIncreasingSamples seq) 
    (limit : ℚ) 
    (convergence : ratesConverge seq limit) : 
  limit = estimatedGerminationProbability seq :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_estimate_l532_53279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_of_polynomial_l532_53285

theorem leading_coefficient_of_polynomial (x : ℝ) : 
  let p : Polynomial ℝ := -5 * (X^5 - X^4 + 2*X^3) + 8 * (X^5 + 3) - 3 * (3*X^5 + X^3 + 2)
  p.leadingCoeff = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_of_polynomial_l532_53285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l532_53274

def factorial_minus_n (n : ℕ) : ℕ := n.factorial - n

def sum_sequence : ℕ := (Finset.range 12).sum (λ i => factorial_minus_n (i + 1))

theorem units_digit_of_sum_sequence :
  sum_sequence % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l532_53274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coord_2005_is_3_l532_53230

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ
deriving Inhabited

/-- Represents the spiral maze -/
def SpiralMaze := List Point

/-- Generates the spiral maze up to a given number of stops -/
def generateMaze (stops : ℕ) : SpiralMaze :=
  sorry

/-- The maze starts at (0,0) -/
axiom start_point : (generateMaze 1).head! = Point.mk 0 0

/-- The first five stops are as described -/
axiom first_five_stops :
  (generateMaze 5).take 5 = [
    Point.mk 0 0,
    Point.mk 1 0,
    Point.mk 1 1,
    Point.mk 0 1,
    Point.mk (-1) 1
  ]

/-- The ninth stop is at (2,-1) -/
axiom ninth_stop : (generateMaze 9).getLast! = Point.mk 2 (-1)

/-- The x-coordinate of the 2005th stop -/
def x_coord_2005 : ℤ :=
  (generateMaze 2005).getLast!.x

/-- The main theorem to prove -/
theorem x_coord_2005_is_3 : x_coord_2005 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coord_2005_is_3_l532_53230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_range_of_k_l532_53240

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - 2*x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^2 + x - 2

-- Theorem for monotonic intervals
theorem monotonic_intervals :
  (∀ x < -2, f' x > 0) ∧
  (∀ x ∈ Set.Ioo (-2) 1, f' x < 0) ∧
  (∀ x > 1, f' x > 0) := by
  sorry

-- Theorem for range of k
theorem range_of_k :
  ∀ k : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 2*k ∧ f y = 2*k ∧ f z = 2*k) ↔
  k ∈ Set.Ioo (-1/12) (13/6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_range_of_k_l532_53240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocentric_tetrahedron_properties_l532_53207

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Vector3D

/-- Defines an orthocentric tetrahedron -/
def IsOrthocentric (t : Tetrahedron) : Prop := sorry

/-- The orthocenter of a tetrahedron -/
noncomputable def Orthocenter (t : Tetrahedron) : Point3D := sorry

/-- Normal transversal of two edges -/
noncomputable def NormalTransversal (e1 e2 : Line3D) : Line3D := sorry

/-- Checks if a point lies on a line -/
def PointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Foot of altitude from a point to a plane -/
noncomputable def FootOfAltitude (p : Point3D) (plane : Plane3D) : Point3D := sorry

/-- Checks if two edges are opposite in a tetrahedron -/
def IsOppositeEdges (t : Tetrahedron) (e1 e2 : Line3D) : Prop := sorry

/-- Main theorem about orthocentric tetrahedra -/
theorem orthocentric_tetrahedron_properties (t : Tetrahedron) 
  (h : IsOrthocentric t) : 
  let o := Orthocenter t
  let ab := Line3D.mk t.A (Vector3D.mk (t.B.x - t.A.x) (t.B.y - t.A.y) (t.B.z - t.A.z))
  let cd := Line3D.mk t.C (Vector3D.mk (t.D.x - t.C.x) (t.D.y - t.C.y) (t.D.z - t.C.z))
  let nt := NormalTransversal ab cd
  let bcd_plane := Plane3D.mk 1 1 1 1  -- placeholder values
  ∀ (e1 e2 : Line3D), 
    (IsOppositeEdges t e1 e2) → 
    (PointOnLine o (NormalTransversal e1 e2)) ∧ 
    (PointOnLine (FootOfAltitude t.A bcd_plane) nt) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocentric_tetrahedron_properties_l532_53207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l532_53298

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^(abs x) else abs (x^2 - 2*x)

theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a x ≤ 3 ↔ x ≤ 3) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l532_53298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_park_time_l532_53262

/-- The time taken for John to reach the park -/
noncomputable def time_to_park (speed : ℝ) (distance : ℝ) : ℝ :=
  (distance / 1000) / speed * 60

/-- Theorem stating that John will reach the park in approximately 2 minutes -/
theorem john_park_time : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |time_to_park 9 300 - 2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_park_time_l532_53262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l532_53288

theorem absolute_value_equation_solutions :
  ∃ (S : Finset ℝ), (∀ x : ℝ, x ∈ S ↔ |x + 1| = |x - 3| + |x - 4|) ∧ S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l532_53288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_conditions_l532_53224

/-- A function f(x) with parameters a, b, and c. -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- Theorem stating the conditions for f(x) to have both a maximum and a minimum. -/
theorem f_max_min_conditions (a b c : ℝ) (ha : a ≠ 0) 
  (hf : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    (∀ x > 0, f a b c x ≤ f a b c x₁) ∧
    (∀ x > 0, f a b c x ≥ f a b c x₂)) :
  a * b > 0 ∧ b^2 + 8*a*c > 0 ∧ a * c < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_conditions_l532_53224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_13_l532_53281

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like (n + 1) + fibonacci_like n

theorem seventh_term_is_13 : fibonacci_like 6 = 13 := by
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rw [fibonacci_like]
  rfl

#eval fibonacci_like 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_13_l532_53281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_count_l532_53268

/-- A configuration of X's on a 5x5 grid --/
def Configuration := Fin 5 → Fin 5 → Bool

/-- Check if three points are in a straight line --/
def inLine (p1 p2 p3 : Fin 5 × Fin 5) : Prop :=
  ∃ (a b : ℚ), (p3.1 : ℚ) = a * (p2.1 : ℚ) + b * (p1.1 : ℚ) ∧
                (p3.2 : ℚ) = a * (p2.2 : ℚ) + b * (p1.2 : ℚ) ∧
                a + b = 1

/-- A valid configuration satisfies the problem constraints --/
def isValid (c : Configuration) : Prop :=
  ∀ p1 p2 p3 : Fin 5 × Fin 5,
    c p1.1 p1.2 ∧ c p2.1 p2.2 ∧ c p3.1 p3.2 → ¬inLine p1 p2 p3

/-- Count the number of X's in a configuration --/
def countX (c : Configuration) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 5)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 5)) fun j =>
      if c i j then 1 else 0)

/-- The main theorem stating that 7 is the maximum number of X's --/
theorem max_x_count :
  (∃ c : Configuration, isValid c ∧ countX c = 7) ∧
  (∀ c : Configuration, isValid c → countX c ≤ 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_count_l532_53268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l532_53241

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_properties (a b : ℝ) 
  (h_odd : ∀ x, f a b (-x) = -(f a b x)) :
  (a = 2 ∧ b = 1) ∧ 
  (∀ x y, x < y → f a b x > f a b y) ∧
  (∀ k, (∀ θ, -π/2 < θ ∧ θ < π/2 → 
    f a b k + f a b (Real.cos θ^2 - 2 * Real.sin θ) ≤ 0) → 
    k > -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l532_53241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_is_centroid_l532_53295

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the center of mass
noncomputable def centerOfMass (p1 p2 p3 : Point) : Point := sorry

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point := sorry

-- Define a property that a point is on a line segment
def isOnSegment (p : Point) (a b : Point) : Prop := sorry

-- Define a property that a point has traversed the perimeter
def hasTraversedPerimeter (p : Point) (t : Triangle) : Prop := sorry

-- Main theorem
theorem center_of_mass_is_centroid (t : Triangle) (p1 p2 p3 : ℝ → Point) :
  (∀ (t' : ℝ), isOnSegment (p1 t') t.A t.B ∨ isOnSegment (p1 t') t.B t.C ∨ isOnSegment (p1 t') t.C t.A) →
  (∀ (t' : ℝ), isOnSegment (p2 t') t.A t.B ∨ isOnSegment (p2 t') t.B t.C ∨ isOnSegment (p2 t') t.C t.A) →
  (∀ (t' : ℝ), isOnSegment (p3 t') t.A t.B ∨ isOnSegment (p3 t') t.B t.C ∨ isOnSegment (p3 t') t.C t.A) →
  (∃ i, hasTraversedPerimeter (p1 i) t ∨ hasTraversedPerimeter (p2 i) t ∨ hasTraversedPerimeter (p3 i) t) →
  (∀ (t' : ℝ), centerOfMass (p1 t') (p2 t') (p3 t') = centerOfMass (p1 0) (p2 0) (p3 0)) →
  centerOfMass (p1 0) (p2 0) (p3 0) = centroid t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_is_centroid_l532_53295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_year_l532_53290

/-- Represents the outcome of a single die roll --/
inductive DieOutcome
  | Divisible : DieOutcome  -- Divisible by 2 or 3
  | Prime : DieOutcome      -- Prime number
  | RollAgain : DieOutcome  -- Roll a 1

/-- The probability of stopping on a single roll --/
noncomputable def stop_probability : ℝ := 8/10

/-- The probability of continuing (rolling again) on a single roll --/
noncomputable def continue_probability : ℝ := 1/10

/-- The number of days in a non-leap year --/
def days_in_year : ℕ := 365

/-- The expected number of rolls on a single day --/
noncomputable def expected_rolls_per_day : ℝ := 10/9

/-- Theorem stating the expected number of die rolls in a non-leap year --/
theorem expected_rolls_in_year :
  (expected_rolls_per_day * days_in_year : ℝ) = 3650/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_in_year_l532_53290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_root_combined_with_sqrt8_l532_53222

theorem simplest_quadratic_root_combined_with_sqrt8 (a : ℝ) : 
  (∃ (b : ℝ), b > 0 ∧ Real.sqrt (a + 1) = b * Real.sqrt 2) →
  (Real.sqrt (a + 1) = Real.sqrt 8) →
  a = 1 := by
  intro h1 h2
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_root_combined_with_sqrt8_l532_53222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l532_53244

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let F := (a * e, 0)
  let m := Real.sqrt 3
  let tangent_line := λ x => m * (x - F.1)
  (∃! x, x > F.1 ∧ (x^2 / a^2 - (tangent_line x)^2 / b^2 = 1)) →
  e ∈ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l532_53244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_functions_l532_53250

noncomputable section

-- Function 1
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (Real.cos x))

-- Function 2
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x^2)

-- Domain of function 1
def domain_f : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi}

-- Domain of function 2
def domain_g : Set ℝ := Set.Icc (-3) (-Real.pi/2) ∪ Set.Ioo 0 (Real.pi/2)

theorem domain_of_functions :
  (∀ x : ℝ, f x ∈ Set.univ ↔ x ∈ domain_f) ∧
  (∀ x : ℝ, g x ∈ Set.univ ↔ x ∈ domain_g) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_functions_l532_53250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_equation_l532_53217

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Triangle ABC -/
def triangle_ABC : Point × Point × Point :=
  (⟨2, 3⟩, ⟨7, 8⟩, ⟨-4, 6⟩)

/-- Reflected triangle A'B'C' -/
def triangle_ABC_reflected : Point × Point × Point :=
  (⟨2, -5⟩, ⟨7, -10⟩, ⟨-4, -8⟩)

/-- Function to check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Function to check if a line is horizontal -/
def is_horizontal (l : Line) : Prop :=
  l.m = 0

/-- Function to check if a point is the midpoint of two other points -/
def is_midpoint (m : Point) (p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Theorem: The line of reflection for the given triangle is y = -1 -/
theorem reflection_line_equation :
  ∃ (l : Line),
    (is_horizontal l) ∧
    (l.b = -1) ∧
    (∀ (p p_reflected : Point),
      (p, p_reflected) ∈ [
        (triangle_ABC.1, triangle_ABC_reflected.1),
        (triangle_ABC.2.1, triangle_ABC_reflected.2.1),
        (triangle_ABC.2.2, triangle_ABC_reflected.2.2)
      ] →
      ∃ (m : Point),
        (is_midpoint m p p_reflected) ∧
        (point_on_line m l)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_equation_l532_53217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_squared_product_of_two_primes_squared_l532_53266

theorem divisors_of_squared_product_of_two_primes_squared (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  let n := p^2 * q^2
  (Finset.card (Finset.filter (· ∣ n^2) (Finset.range (n^2 + 1)))) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_squared_product_of_two_primes_squared_l532_53266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finish_on_day_seven_l532_53220

/-- Represents the number of problems solved on a given day -/
def problems_solved (day : ℕ) (start : ℕ) : ℕ := sorry

/-- The total number of problems in the textbook -/
def total_problems : ℕ := 91

/-- The number of problems left on September 8 (day 3) -/
def problems_left_day3 : ℕ := 46

/-- The day on which Yura finishes solving all problems -/
def finish_day : ℕ := 7

/-- Axiom: Each day from day 2 onwards, Yura solves one less problem than the previous day -/
axiom solve_pattern (n : ℕ) (start : ℕ) : 
  n ≥ 2 → problems_solved n start = problems_solved (n-1) start - 1

/-- Axiom: The sum of problems solved over the first three days equals the total minus problems left on day 3 -/
axiom sum_first_three_days (start : ℕ) : 
  problems_solved 1 start + problems_solved 2 start + problems_solved 3 start = total_problems - problems_left_day3

/-- Theorem: Yura finishes solving all problems on the 7th day -/
theorem finish_on_day_seven : finish_day = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finish_on_day_seven_l532_53220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_lines_l532_53249

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The point A through which all lines must pass -/
def point_A : ℝ × ℝ := (1, 4)

/-- A function to check if a line passes through point A -/
def passes_through_A (l : Line) : Prop :=
  4 = l.slope * 1 + l.intercept

/-- A function to check if a line has equal absolute values of x and y intercepts -/
def equal_abs_intercepts (l : Line) : Prop :=
  abs (l.intercept / l.slope) = abs l.intercept

/-- The set of all lines passing through A with equal absolute intercepts -/
def valid_lines : Set Line :=
  {l : Line | passes_through_A l ∧ equal_abs_intercepts l}

/-- Theorem stating that there are exactly three valid lines -/
theorem three_valid_lines : ∃ (s : Finset Line), s.card = 3 ∧ ∀ l, l ∈ s ↔ l ∈ valid_lines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_lines_l532_53249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_log_half_l532_53252

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := a / x^3 - b / x + 2

-- State the theorem
theorem function_value_at_log_half 
  (a b : ℝ) 
  (h : f a b (Real.log 2) = 3) : 
  f a b (Real.log (1/2)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_log_half_l532_53252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_w_range_l532_53208

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

theorem three_zeros_implies_w_range (w : ℝ) (h1 : w > 0) 
  (h2 : ∃ (x1 x2 x3 : ℝ), 0 < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < π ∧ 
    f w x1 = 0 ∧ f w x2 = 0 ∧ f w x3 = 0 ∧
    ∀ (x : ℝ), 0 < x ∧ x < π ∧ f w x = 0 → x = x1 ∨ x = x2 ∨ x = x3) :
  7/3 < w ∧ w ≤ 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_w_range_l532_53208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l532_53284

noncomputable def a : ℝ × ℝ := (-1, 3)
noncomputable def b : ℝ × ℝ := (3, -4)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_a_on_b :
  projection a b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l532_53284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_a_pow_25_minus_a_l532_53239

theorem largest_divisor_of_a_pow_25_minus_a (a : ℤ) : 
  ∃ (n : ℕ), n = 2730 ∧ (n : ℤ) ∣ (a^25 - a) ∧ ∀ m : ℕ, m > n → ¬(∀ b : ℤ, (m : ℤ) ∣ (b^25 - b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_a_pow_25_minus_a_l532_53239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_gh_products_l532_53255

def is_valid_number (g h : ℕ) : Prop :=
  g < 10 ∧ h < 10 ∧ 
  (5 * 10^9 + 3 * 10^8 + 8 * 10^7 + g * 10^6 + 5 * 10^5 + 0 * 10^4 + 7 * 10^3 + 3 * 10^2 + h * 10 + 6) % 72 = 0

def distinct_gh_products (g h : ℕ) : Finset ℕ :=
  if g < 10 ∧ h < 10 ∧ (5 * 10^9 + 3 * 10^8 + 8 * 10^7 + g * 10^6 + 5 * 10^5 + 0 * 10^4 + 7 * 10^3 + 3 * 10^2 + h * 10 + 6) % 72 = 0
  then {g * h}
  else ∅

theorem sum_of_distinct_gh_products :
  (Finset.biUnion (Finset.range 10) (λ g => Finset.biUnion (Finset.range 10) (λ h => distinct_gh_products g h))).sum id = 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_gh_products_l532_53255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_bn_sn_l532_53258

theorem min_value_bn_sn :
  ∀ n : ℕ+,
  let a : ℕ+ → ℝ := λ k => (k : ℝ)^2 + k
  let S : ℕ+ → ℝ := λ k => k / (k + 1)
  let b : ℕ+ → ℝ := λ k => k - 35
  ∀ k : ℕ+, b k * S k ≥ -25 ∧ ∃ m : ℕ+, b m * S m = -25 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_bn_sn_l532_53258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l532_53282

/-- The length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) : ℝ :=
  2 * Real.sqrt (side_length ^ 2 - (shorter_diagonal / 2) ^ 2)

/-- Theorem: In a rhombus with side length 53 units and shorter diagonal 50 units, 
    the longer diagonal is 94 units -/
theorem rhombus_longer_diagonal :
  longer_diagonal 53 50 = 94 := by
  -- Unfold the definition of longer_diagonal
  unfold longer_diagonal
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l532_53282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_overlapped_squares_l532_53203

/-- Represents a square on a checkerboard -/
structure CheckerboardSquare where
  side : ℝ
  assumption : side = 2

/-- Represents a square card -/
structure Card where
  side : ℝ
  assumption : side = 2.5

/-- Calculates the diagonal of a square -/
noncomputable def diagonal (s : ℝ) : ℝ := Real.sqrt (2 * s^2)

/-- Predicate to determine if a card overlaps a given number of squares -/
def card_overlaps (n : ℕ) (board : CheckerboardSquare) (card : Card) (placement : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem stating the maximum number of checkerboard squares a card can overlap -/
theorem max_overlapped_squares (board : CheckerboardSquare) (card : Card) :
  ∃ (n : ℕ), n ≤ 9 ∧ 
  ∀ (m : ℕ), (∃ (placement : ℝ × ℝ), card_overlaps m board card placement) → m ≤ n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_overlapped_squares_l532_53203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_incorrect_l532_53289

-- Define the propositions
def proposition1 : Prop := ∀ p q : Prop, (p ∧ q = False) → (p = False ∧ q = False)

def proposition2 : Prop := 
  (∃ a b : ℝ, a > b ∧ (2 : ℝ)^a ≤ (2 : ℝ)^b - 1) ↔ (∃ a b : ℝ, a ≤ b ∧ (2 : ℝ)^a ≤ (2 : ℝ)^b - 1)

def proposition3 : Prop := 
  (∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (¬∃ x : ℝ, x^2 + 1 < 0)

noncomputable def proposition4 : Prop := 
  ∀ A B C : ℝ, (A > B ↔ Real.sin A > Real.sin B)

-- Theorem statement
theorem exactly_one_incorrect : 
  (¬proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4) ∨
  (proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ proposition4) ∨
  (proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4) ∨
  (proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬proposition4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_incorrect_l532_53289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turkish_olympiad_2003_l532_53218

theorem turkish_olympiad_2003 (x m n : ℕ) : 
  x^m = 2^(2*n + 1) + 2^n + 1 ↔ 
    ((x = 2^(2*n + 1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turkish_olympiad_2003_l532_53218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l532_53232

theorem triangle_area_proof (a b c : ℝ) (h_perimeter : a + b + c = 2 * Real.sqrt 2 + Real.sqrt 5)
  (h_sine_ratio : (Real.sqrt 2 - 1) * Real.sin b = Real.sqrt 5 * Real.sin a ∧
                  Real.sqrt 5 * Real.sin c = (Real.sqrt 2 + 1) * Real.sin b) :
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2)) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l532_53232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_sine_ratio_l532_53227

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_to_pi : A + B + C = Real.pi

-- Theorem 1
theorem cosine_squared_sum (t : Triangle) :
  Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 = 1 - 2 * Real.cos t.A * Real.cos t.B * Real.cos t.C :=
sorry

-- Theorem 2
theorem sine_ratio (t : Triangle) 
  (h : ∃ k : Real, Real.cos t.A / 39 = Real.cos t.B / 33 ∧ Real.cos t.B / 33 = Real.cos t.C / 25 ∧ Real.cos t.C / 25 = k) :
  ∃ m : Real, Real.sin t.A / 13 = Real.sin t.B / 14 ∧ Real.sin t.B / 14 = Real.sin t.C / 15 ∧ Real.sin t.C / 15 = m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_sine_ratio_l532_53227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_boys_count_l532_53287

theorem absent_boys_count (total_students girls_present : ℕ) : 
  total_students = 250 →
  girls_present = 140 →
  girls_present = 2 * (total_students - girls_present) →
  total_students - (girls_present + (total_students - girls_present)) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_boys_count_l532_53287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_approx_l532_53236

/-- The length of a bridge crossed by a man walking at a given speed for a given time -/
noncomputable def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 1000 / 60

/-- Theorem stating that a man walking at 5 km/hr crossing a bridge in 15 minutes implies the bridge length is approximately 1250 meters -/
theorem bridge_length_approx :
  let speed := 5 -- km/hr
  let time := 15 -- minutes
  let length := bridge_length speed time
  ∃ ε > 0, |length - 1250| < ε :=
by
  sorry

-- Use #eval only for computable functions
def bridge_length_nat (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time * 1000 / 60

#eval bridge_length_nat 5 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_approx_l532_53236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l532_53273

theorem circle_equation (x y : ℝ) : 
  (∃ (b : ℝ), x^2 + (y - b)^2 = 1 ∧ 
              (0 : ℝ) = 0 ∧ -- Center on y-axis
              1^2 + (2 - b)^2 = 1) →
  x^2 + (y - 2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l532_53273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_intersection_ratio_l532_53209

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid where
  M : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if four points form a parallelogram -/
def isParallelogram (A B C D : Point3D) : Prop := sorry

/-- Checks if two line segments are equal -/
def segmentsEqual (P Q R S : Point3D) : Prop := sorry

/-- Checks if a point is on a line -/
def pointOnLine (P Q R : Point3D) : Prop := sorry

/-- Checks if a point is on a plane -/
def pointOnPlane (P Q R S : Point3D) : Prop := sorry

/-- Calculates the ratio of two line segments -/
noncomputable def segmentRatio (P Q R : Point3D) : ℝ := sorry

theorem pyramid_intersection_ratio 
  (pyramid : QuadrangularPyramid) 
  (K P X : Point3D) :
  isParallelogram pyramid.A pyramid.B pyramid.C pyramid.D →
  segmentsEqual pyramid.D K K pyramid.M →
  segmentRatio pyramid.B P pyramid.M = 1/4 →
  pointOnLine X pyramid.M pyramid.C →
  pointOnPlane X pyramid.A K P →
  segmentRatio pyramid.M X pyramid.C = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_intersection_ratio_l532_53209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_rainfall_l532_53226

/-- Represents the probability and amount of rainfall for a weather condition -/
structure RainCondition where
  probability : ℝ
  amount : ℝ

/-- Calculates the expected value of rainfall for a single day -/
def dailyExpectedRainfall (conditions : List RainCondition) : ℝ :=
  (conditions.map fun c => c.probability * c.amount).sum

/-- The number of days in the forecast -/
def forecastDays : ℕ := 5

/-- The list of possible rain conditions for each day -/
def rainConditions : List RainCondition := [
  { probability := 0.5, amount := 0 },
  { probability := 0.2, amount := 3 },
  { probability := 0.3, amount := 8 }
]

/-- Statement: The expected value of total rainfall for the forecast period is 15.0 inches -/
theorem expected_total_rainfall :
  forecastDays * (dailyExpectedRainfall rainConditions) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_rainfall_l532_53226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l532_53205

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center to a focus -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The distance from a focus to a vertex -/
noncomputable def focus_to_vertex (e : Ellipse) : ℝ :=
  e.a + focal_distance e

/-- The distance from a focus to a point P where PF is perpendicular to the major axis -/
noncomputable def focus_to_perpendicular (e : Ellipse) : ℝ :=
  e.b^2 / e.a

theorem ellipse_eccentricity_theorem (e : Ellipse) :
  focus_to_perpendicular e = (3/4) * focus_to_vertex e →
  eccentricity e = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l532_53205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_k_range_l532_53243

/-- The function f(x) = 4x^2 - kx - 8 is monotonically increasing on [5, 20] -/
def is_monotone_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 5 20 → y ∈ Set.Icc 5 20 → x ≤ y → f x ≤ f y

/-- The function f(x) = 4x^2 - kx - 8 -/
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

theorem monotone_increasing_k_range :
  ∀ k, is_monotone_increasing (f k) k ↔ k ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_k_range_l532_53243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l532_53206

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (1 - x) else x * (x + 1)

-- State the theorem
theorem f_is_odd_and_correct : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, x ≥ 0 → f x = x * (1 - x)) ∧ 
  (∀ x, x ≤ 0 → f x = x * (x + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l532_53206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vote_count_fraction_l532_53238

theorem vote_count_fraction (V C : ℝ) (x : ℝ) (h1 : V > 0) (h2 : C > 0) (h3 : C ≤ V)
  (h4 : x = 0.7857142857142856)
  (h5 : (1/4 : ℝ) * C + x * (V - C) = 2 * ((3/4 : ℝ) * C)) :
  abs (C / V - 0.3857) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vote_count_fraction_l532_53238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_top_face_l532_53242

/-- Represents a standard 6-faced die -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (valid : ∀ i, faces i ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ))

/-- The probability of selecting a dot from a face with n dots -/
def prob_select_dot (n : ℕ) : ℚ := n / 21

/-- The probability that a face with n dots loses exactly one dot when two dots are removed -/
def prob_lose_one_dot (n : ℕ) : ℚ :=
  prob_select_dot n * (1 - (n - 1) / 20) + (1 - prob_select_dot n) * (n / 20)

/-- The probability that the top face has an even number of dots after two dots are removed -/
def prob_even_after_removal : ℚ :=
  (1 / 6) * (prob_lose_one_dot 1 + prob_lose_one_dot 3 + prob_lose_one_dot 5)

theorem prob_even_top_face : prob_even_after_removal = 23 / 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_top_face_l532_53242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egg_composition_theorem_l532_53291

/-- Represents the components of an egg -/
inductive EggComponent
  | Yolk
  | White
  | Shell

/-- Structure representing an egg -/
structure Egg where
  total_weight : ℝ
  yolk_percentage : ℝ
  white_percentage : ℝ

/-- Calculate the weight of a component given its percentage -/
noncomputable def component_weight (egg : Egg) (percentage : ℝ) : ℝ :=
  egg.total_weight * percentage / 100

/-- Find the component closest to a given weight -/
noncomputable def closest_component (egg : Egg) (target : ℝ) : EggComponent :=
  let yolk_weight := component_weight egg egg.yolk_percentage
  let white_weight := component_weight egg egg.white_percentage
  let shell_weight := component_weight egg (100 - egg.yolk_percentage - egg.white_percentage)
  let yolk_diff := abs (yolk_weight - target)
  let white_diff := abs (white_weight - target)
  let shell_diff := abs (shell_weight - target)
  if yolk_diff ≤ white_diff ∧ yolk_diff ≤ shell_diff then
    EggComponent.Yolk
  else if white_diff ≤ shell_diff then
    EggComponent.White
  else
    EggComponent.Shell

theorem egg_composition_theorem (egg : Egg)
  (h1 : egg.total_weight = 60)
  (h2 : egg.yolk_percentage = 32)
  (h3 : egg.white_percentage = 53) :
  (100 - egg.yolk_percentage - egg.white_percentage = 15) ∧
  (closest_component egg 32 = EggComponent.White) := by
  sorry

#eval 100 - 32 - 53  -- This should output 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_egg_composition_theorem_l532_53291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l532_53245

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ

/-- Function to check if a point is on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Function to check if a point is on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  point.y = line.m * point.x + line.b

/-- Function to calculate distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Function to check if a triangle is equilateral -/
def isEquilateral (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

/-- Main theorem -/
theorem parabola_theorem (C : Parabola) (F A B : Point) (l : Line) :
  onParabola A C →
  onLine B l →
  isEquilateral A B F →
  distance A B = 4 →
  ∃ (N : Point),
    N.y = 0 ∧
    N.x = 2 ∧
    C.p = 2 ∧
    ∀ (Q R : Point) (l' : Line),
      onLine N l' →
      onParabola Q C →
      onParabola R C →
      onLine Q l' →
      onLine R l' →
      1 / (distance N Q)^2 + 1 / (distance N R)^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l532_53245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l532_53211

theorem tan_theta_minus_pi_fourth (θ : ℝ) 
  (h1 : θ > π / 2) 
  (h2 : θ < π) 
  (h3 : Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5) : 
  Real.tan (θ - π / 4) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l532_53211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l532_53214

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + f a x

theorem problem_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (m : ℝ), m = 0 ∧ ∀ x, f a x ≥ m) :
  a = 1 ∧ 
  ∃ M, (∀ x, g a x ≤ M ∨ g a x ≥ M) ∧ -2.5 < M ∧ M < -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l532_53214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l532_53256

/-- Represents the time (in days) it takes for A and B to complete the work together -/
noncomputable def time_AB : ℝ → ℝ → ℝ → ℝ := λ t_A t_AB_part t_A_part ↦ 
  (t_A * t_AB_part) / (t_A - t_AB_part - t_A_part)

/-- The theorem statement based on the given problem -/
theorem work_completion_time 
  (t_A : ℝ) 
  (t_AB_part : ℝ) 
  (t_A_part : ℝ) 
  (h_t_A : t_A = 28) 
  (h_t_AB_part : t_AB_part = 10) 
  (h_t_A_part : t_A_part = 21) :
  time_AB t_A t_AB_part t_A_part = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l532_53256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l532_53235

noncomputable def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m * (180 / Real.pi)

theorem line_properties :
  ∃ m b : ℝ,
    (∀ x y, line_equation x y ↔ y = m * x + b) ∧
    slope_angle m = 135 ∧
    b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l532_53235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_extended_point_circle_l532_53260

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse (F₁ F₂ : ℝ × ℝ) where
  a : ℝ
  h : a > 0

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse F₁ F₂) (P : ℝ × ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * e.a

/-- The point Q obtained by extending F₁P -/
noncomputable def ExtendedPoint (F₁ : ℝ × ℝ) (P : ℝ × ℝ) (F₂ : ℝ × ℝ) : ℝ × ℝ :=
  let t : ℝ := (dist P F₂) / (dist P F₁)
  (F₁.1 + t * (P.1 - F₁.1), F₁.2 + t * (P.2 - F₁.2))

theorem ellipse_extended_point_circle 
  (F₁ F₂ : ℝ × ℝ) (e : Ellipse F₁ F₂) (P : ℝ × ℝ) 
  (h : PointOnEllipse e P) :
  let Q := ExtendedPoint F₁ P F₂
  dist Q F₁ = 2 * e.a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_extended_point_circle_l532_53260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_product_l532_53271

theorem inverse_sum_product : (12 : ℝ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_product_l532_53271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_work_hours_l532_53215

theorem suresh_work_hours (suresh_rate ashutosh_rate : ℚ) 
  (ashutosh_remaining_time : ℚ) : 
  suresh_rate = 1 / 15 →
  ashutosh_rate = 1 / 10 →
  ashutosh_remaining_time = 4 →
  ∃ (suresh_time : ℚ), 
    suresh_time * suresh_rate + ashutosh_remaining_time * ashutosh_rate = 1 ∧
    suresh_time = 9 := by
  intro h1 h2 h3
  use 9
  constructor
  · -- Prove that the equation holds
    rw [h1, h2, h3]
    norm_num
  · -- Prove that suresh_time = 9
    rfl

#check suresh_work_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_work_hours_l532_53215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l532_53270

theorem greatest_x_value (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 (Nat.lcm 18 21)) = 630) : 
  x ≤ 630 ∧ ∃ (y : ℕ), y > 630 → Nat.lcm y (Nat.lcm 15 (Nat.lcm 18 21)) > 630 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l532_53270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l532_53231

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point P on the hyperbola
variable (P : ℝ × ℝ)

-- Define vectors PF₁ and PF₂
noncomputable def PF₁ (P : ℝ × ℝ) : ℝ × ℝ := (F₁.1 - P.1, F₁.2 - P.2)
noncomputable def PF₂ (P : ℝ × ℝ) : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem hyperbola_equation (P : ℝ × ℝ) 
  (h1 : dot_product (PF₁ P) (PF₂ P) = 0) 
  (h2 : magnitude (PF₁ P) * magnitude (PF₂ P) = 2) :
  ∃ (x y : ℝ), P = (x, y) ∧ x^2/4 - y^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l532_53231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_is_CD_l532_53202

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the angles
def angle_ABD : ℝ := 30
def angle_ADB : ℝ := 65
def angle_BDC : ℝ := 65
def angle_CBD : ℝ := 85

-- Define the segments
noncomputable def segment_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem longest_segment_is_CD (ABCD : Quadrilateral) :
  segment_length ABCD.C ABCD.D > segment_length ABCD.A ABCD.B ∧
  segment_length ABCD.C ABCD.D > segment_length ABCD.B ABCD.C ∧
  segment_length ABCD.C ABCD.D > segment_length ABCD.A ABCD.D ∧
  segment_length ABCD.C ABCD.D > segment_length ABCD.B ABCD.D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_is_CD_l532_53202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l532_53269

def IsOnTerminalSideOf (P : ℝ × ℝ) (α : ℝ) : Prop :=
  P.1 = Real.cos α * Real.sqrt (P.1^2 + P.2^2) ∧
  P.2 = Real.sin α * Real.sqrt (P.1^2 + P.2^2)

theorem angle_terminal_side_point (α : ℝ) (a : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (-4, a) ∧ IsOnTerminalSideOf P α) →
  (Real.sin α * Real.cos α = Real.sqrt 3 / 4) →
  (a = -4 * Real.sqrt 3 ∨ a = -4 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l532_53269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_l532_53229

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

-- State the theorem
theorem sum_of_min_max_f : 
  ∃ (min_f max_f : ℝ), 
    (∀ x, f x ≥ min_f) ∧ 
    (∃ x₁, f x₁ = min_f) ∧ 
    (∀ x, f x ≤ max_f) ∧ 
    (∃ x₂, f x₂ = max_f) ∧ 
    (min_f + max_f = -3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_l532_53229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l532_53210

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The radius of the circle in feet -/
def r : ℝ := 40

/-- The minimum distance traveled by all points -/
noncomputable def min_distance : ℝ := 640 + 960 * Real.sqrt 2

/-- Theorem stating the minimum distance traveled by all points -/
theorem min_distance_proof :
  let points := n
  let radius := r
  let non_adjacent_visits := points - 3  -- Each point visits all but 3 others (itself and 2 adjacent)
  let total_visits := points * non_adjacent_visits
  total_visits * radius * (2 + 3 * Real.sqrt 2) = min_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l532_53210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_bounds_vector_sum_magnitude_l532_53213

namespace TriangleProof

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

variable (t : Triangle)

/-- First theorem: bounds on angle B -/
theorem angle_B_bounds (h1 : Real.sin t.B = Real.sqrt 7 / 4) 
    (h2 : (Real.cos t.A / Real.sin t.A) + (Real.cos t.C / Real.sin t.C) = 4 * Real.sqrt 7 / 7) : 
    0 < t.B ∧ t.B ≤ π / 3 := by
  sorry

/-- Second theorem: magnitude of vector sum -/
theorem vector_sum_magnitude (h1 : Real.sin t.B = Real.sqrt 7 / 4) 
    (h2 : t.a * t.c * Real.cos t.B = 3 / 2) : 
    t.a^2 + t.c^2 + 2 * t.a * t.c * Real.cos t.B = 8 := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_bounds_vector_sum_magnitude_l532_53213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_6_or_8_l532_53225

def total_marbles : ℕ := 60

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 || n % 8 = 0

def count_multiples : ℕ :=
  (Finset.range total_marbles).filter (λ n => is_multiple_of_6_or_8 (n + 1)) |>.card

theorem probability_multiple_of_6_or_8 :
  (count_multiples : ℚ) / total_marbles = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_6_or_8_l532_53225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l532_53263

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Slope of the first line ax+(2a-1)y+1=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a / (2*a - 1)

/-- Slope of the second line 3x+ay+3=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -3 / a

/-- Theorem stating that a=-1 is a sufficient but not necessary condition for perpendicularity -/
theorem perpendicular_lines (a : ℝ) :
  (perpendicular (slope1 (-1)) (slope2 (-1))) ∧
  (∃ b : ℝ, b ≠ -1 ∧ perpendicular (slope1 b) (slope2 b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l532_53263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_grid_tiling_with_crosses_l532_53233

/-- Represents a cell in the infinite grid --/
structure Cell where
  x : ℤ
  y : ℤ

/-- Represents a cross in the tiling --/
structure Cross where
  center : Cell
  up : Cell
  down : Cell
  left : Cell
  right : Cell

/-- Predicate to check if a cell is adjacent to the center of a cross --/
def isAdjacent (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

/-- Predicate to check if a cross is valid --/
def isValidCross (cr : Cross) : Prop :=
  isAdjacent cr.center cr.up ∧
  isAdjacent cr.center cr.down ∧
  isAdjacent cr.center cr.left ∧
  isAdjacent cr.center cr.right

/-- The tiling function that maps each cell to a cross --/
noncomputable def tiling : Cell → Cross :=
  sorry

/-- Theorem stating that the tiling is possible --/
theorem infinite_grid_tiling_with_crosses :
  (∀ c : Cell, isValidCross (tiling c)) ∧
  (∀ c : Cell, ∃! cr : Cross, (cr = tiling c ∨ c = cr.center ∨ c = cr.up ∨ c = cr.down ∨ c = cr.left ∨ c = cr.right)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_grid_tiling_with_crosses_l532_53233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_meeting_arrangements_l532_53251

/-- Represents the number of countries --/
def num_countries : ℕ := 5

/-- Represents the number of pairs that don't meet --/
def num_non_meeting_pairs : ℕ := 2

/-- Represents the number of days for meetings --/
def num_days : ℕ := 2

/-- Represents the number of sessions per day --/
def sessions_per_day : ℕ := 2

/-- Calculates the total number of possible pairs --/
def total_possible_pairs : ℕ := (num_countries * (num_countries - 1)) / 2

/-- Calculates the number of actual meeting pairs --/
def actual_meeting_pairs : ℕ := total_possible_pairs - num_non_meeting_pairs

/-- Represents the number of pairs that can meet simultaneously --/
def simultaneous_meetings : ℕ := 2

/-- Theorem stating the number of ways to arrange meetings --/
theorem number_of_meeting_arrangements :
  (Nat.factorial (actual_meeting_pairs / simultaneous_meetings)) * 2 = 48 := by
  sorry

#eval (Nat.factorial (actual_meeting_pairs / simultaneous_meetings)) * 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_meeting_arrangements_l532_53251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_unique_solution_l532_53277

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the condition for a unique solution
def hasUniqueSolution (t : Triangle) : Prop :=
  (t.a = 3 ∧ t.b = 4 ∧ Real.cos t.B = 3/5) ∨
  (t.a = 3 ∧ t.b = 4 ∧ t.C = Real.pi/6) ∨
  (t.a = 3 ∧ t.b = 4 ∧ t.B = Real.pi/6)

-- Theorem statement
theorem triangle_unique_solution (t : Triangle) :
  hasUniqueSolution t → ∃! (t' : Triangle), t' = t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_unique_solution_l532_53277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_implies_m_range_l532_53257

theorem tangent_condition_implies_m_range (m : ℝ) : 
  (∃ (T₁ T₂ : ℝ × ℝ), 
    T₁.1^2 + T₁.2^2 + m*T₁.1 + m*T₁.2 + 2 = 0 ∧
    T₂.1^2 + T₂.2^2 + m*T₂.1 + m*T₂.2 + 2 = 0 ∧
    T₁ ≠ T₂ ∧ 
    (∀ (Q : ℝ × ℝ), Q.1^2 + Q.2^2 + m*Q.1 + m*Q.2 + 2 = 0 → 
      (Q = T₁ ∨ Q = T₂ ∨ (Q.1 - 1)^2 + (Q.2 - 1)^2 > (T₁.1 - 1)^2 + (T₁.2 - 1)^2))) →
  m > 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_implies_m_range_l532_53257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fourth_root_integer_l532_53254

theorem smallest_fourth_root_integer (m n : ℕ) (s : ℝ) : 
  (0 < n) →
  (0 < s) →
  (s < 1/2000) →
  ((n : ℝ) + s)^4 = m →
  (∀ k < n, ∀ t > 0, t < 1/2000 → ((k : ℝ) + t)^4 ∉ Set.range (Nat.cast : ℕ → ℝ)) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fourth_root_integer_l532_53254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_point_l532_53228

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem symmetric_about_point (ω φ : ℝ) :
  ω > 0 →
  |φ| < π / 2 →
  (∀ x : ℝ, f ω φ (x + π) = f ω φ x) →
  (∀ x : ℝ, g ω x = f ω φ (x + π / 3)) →
  ∀ x : ℝ, f ω φ (π / 6 + x) = -f ω φ (π / 6 - x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_about_point_l532_53228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l532_53264

/-- A parabola symmetric about the x-axis with vertex at the origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- The focus of a parabola -/
noncomputable def Parabola.focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_theorem (para : Parabola) (y₀ : ℝ) :
  para.eq 2 y₀ →
  distance (2, y₀) para.focus = 3 →
  distance (0, 0) (2, y₀) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l532_53264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2xy_value_l532_53293

theorem cos_2xy_value (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = Real.log 2)
  (h2 : Real.cos x + Real.cos y = Real.log 4 / Real.log 7) : 
  Real.cos (2 * (x + y)) = -7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2xy_value_l532_53293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l532_53253

noncomputable def ellipse_equation (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity :
  eccentricity 4 (Real.sqrt 8) = Real.sqrt 2 / 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l532_53253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_symmetry_cos_minus_sin_l532_53200

open Real

/-- The line of symmetry for the function y = cos x - sin x is x = -π/4 -/
theorem line_of_symmetry_cos_minus_sin :
  ∃ (x : ℝ), x = -π/4 ∧ 
  ∀ (t : ℝ), (Real.cos (x + t) - Real.sin (x + t)) = (Real.cos (x - t) - Real.sin (x - t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_symmetry_cos_minus_sin_l532_53200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l532_53296

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_positive : ∀ x > 0, f x = Real.sqrt x + 1

-- Theorem to prove
theorem f_negative (x : ℝ) (h : x < 0) : f x = -Real.sqrt (-x) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l532_53296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l532_53261

noncomputable def pyramid_volume (r : ℝ) : ℝ :=
  (8 / 3) * r^3 * Real.sqrt 3

theorem pyramid_volume_proof (r : ℝ) (h : r > 0) :
  let base_acute_angle : ℝ := 30 * π / 180
  let lateral_inclination : ℝ := 60 * π / 180
  let base_area : ℝ := 8 * r^2
  let height : ℝ := r * Real.sqrt 3
  pyramid_volume r = (1 / 3) * base_area * height :=
by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l532_53261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_kids_played_l532_53297

structure DayData where
  kids : ℕ
  hours : ℝ

def weekData : List DayData := [
  ⟨17, 1.5⟩,
  ⟨15, 2.25⟩,
  ⟨2, 1.75⟩,
  ⟨12, 2.5⟩,
  ⟨7, 3⟩
]

def kidHours (data : DayData) : ℝ :=
  (data.kids : ℝ) * data.hours

def totalKidHours (week : List DayData) : ℝ :=
  week.map kidHours |>.sum

theorem total_kids_played (week : List DayData) :
  week = weekData → totalKidHours week = 114.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_kids_played_l532_53297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_505_l532_53286

def my_sequence (n : ℕ) : ℕ := 
  if n = 1 then 1
  else
    let start := (n * (n - 1)) / 2 + 1
    (start + start + n - 1) * n / 2

theorem a_10_equals_505 : my_sequence 10 = 505 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_505_l532_53286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_approximation_l532_53275

noncomputable section

/-- The side length of each square sheet -/
def side_length : ℝ := 8

/-- The number of sides in the resulting polygon -/
def num_sides : ℕ := 24

/-- The rotation angle of the middle sheet in radians -/
noncomputable def middle_rotation : ℝ := Real.pi / 4

/-- The rotation angle of the top sheet in radians -/
noncomputable def top_rotation : ℝ := Real.pi / 2

/-- The radius of the circumscribed circle of each square -/
noncomputable def circumscribed_radius : ℝ := side_length * Real.sqrt 2 / 2

/-- The area of the resulting 24-sided polygon -/
noncomputable def polygon_area : ℝ := num_sides * (circumscribed_radius^2 * Real.sin (2 * Real.pi / num_sides)) / 2

/-- Theorem stating that the polygon area is approximately 99.38 -/
theorem polygon_area_approximation :
  ∃ ε > 0, abs (polygon_area - 99.38) < ε :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_approximation_l532_53275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_count_l532_53216

/-- The number of oranges in the first three baskets -/
def oranges : ℕ := 15

/-- The number of apples in each of the first three baskets -/
def apples : ℕ := 9

/-- The number of bananas in each of the first three baskets -/
def bananas : ℕ := 14

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

/-- The number of baskets -/
def num_baskets : ℕ := 4

/-- The number of fruits less in the fourth basket compared to the others -/
def difference : ℕ := 2

theorem orange_count : 
  (apples + oranges + bananas) * 3 + 
  ((apples - difference) + (oranges - difference) + (bananas - difference)) = total_fruits := by
  -- Proof goes here
  sorry

#check orange_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_count_l532_53216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_positive_implies_obtuse_l532_53234

-- Define a triangle ABC in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define vector subtraction for 2D vectors
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  ∃ (θ : ℝ), θ > Real.pi / 2 ∧ 
    Real.cos θ = dot_product (vector_sub t.B t.A) (vector_sub t.C t.B) / 
            (Real.sqrt (dot_product (vector_sub t.B t.A) (vector_sub t.B t.A)) * 
             Real.sqrt (dot_product (vector_sub t.C t.B) (vector_sub t.C t.B)))

-- The theorem statement
theorem dot_product_positive_implies_obtuse (t : Triangle) :
  dot_product (vector_sub t.B t.A) (vector_sub t.C t.B) > 0 → is_obtuse t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_positive_implies_obtuse_l532_53234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l532_53283

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 = 2*b*c →
  (a = b → Real.cos A = 1/4) ∧
  (A = π/2 → b = Real.sqrt 6 → (1/2) * b * c = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l532_53283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_length_l532_53259

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := x + m*y + 1 = 0

-- Define the point M
def point_M (m : ℝ) : ℝ × ℝ := (m, m)

-- Define the length of PQ
noncomputable def length_PQ (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem circle_intersection_length :
  ∀ (m : ℝ) (P Q : ℝ × ℝ),
  (∃ (A B : ℝ × ℝ), circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
    line_l ((A.1 + B.1)/2) ((A.2 + B.2)/2) m) →
  circle_C P.1 P.2 →
  circle_C Q.1 Q.2 →
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    (center.1 - (point_M m).1)^2 + (center.2 - (point_M m).2)^2 = r^2 ∧
    (center.1 - P.1)^2 + (center.2 - P.2)^2 = r^2 ∧
    (center.1 - Q.1)^2 + (center.2 - Q.2)^2 = r^2) →
  length_PQ P Q = 12 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_length_l532_53259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_15020_l532_53247

/-- Calculates the total cost of fencing a field given the specifications -/
noncomputable def total_fencing_cost (total_length : ℝ) (ratio_a ratio_b ratio_c : ℕ) 
  (cost_a cost_b cost_c : ℝ) : ℝ :=
  let segment_unit := total_length / (ratio_a + ratio_b + ratio_c : ℝ)
  let length_a := segment_unit * ratio_a
  let length_b := segment_unit * ratio_b
  let length_c := segment_unit * ratio_c
  length_a * cost_a + length_b * cost_b + length_c * cost_c

/-- Theorem stating that the total fencing cost for the given field is 15020 -/
theorem fencing_cost_is_15020 : 
  total_fencing_cost 2400 3 4 5 6.25 4.90 7.35 = 15020 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_fencing_cost 2400 3 4 5 6.25 4.90 7.35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_15020_l532_53247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l532_53299

theorem sin_cos_product (x : ℝ) (h1 : -3*π/2 < x) (h2 : x < -π) (h3 : Real.tan x = -3) :
  Real.sin x * Real.cos x = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l532_53299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l532_53276

-- Define the line
def line (m : ℝ) (x : ℝ) : ℝ := m * x

-- Define the circle
def circleEq (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y - 1)^2 = m^2 - 1

-- Define the intersection of line and circle
def intersection (m : ℝ) (x y : ℝ) : Prop :=
  line m x = y ∧ circleEq m x y

-- Define the angle ACB
def angle_ACB : ℝ := 60

-- Theorem statement
theorem circle_area (m : ℝ) :
  (∃ A B : ℝ × ℝ, intersection m A.1 A.2 ∧ intersection m B.1 B.2) →
  angle_ACB = 60 →
  (π * (m^2 - 1) : ℝ) = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l532_53276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_85_over_8_l532_53248

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- Represents the dimensions of a square -/
structure Square where
  side : ℚ

/-- Calculates the area of the shaded region -/
noncomputable def shaded_area (sq : Square) (rect : Rectangle) : ℚ :=
  rect.width * rect.height - (rect.height * (sq.side * rect.height) / (sq.side + rect.width)) / 2

/-- The theorem stating the area of the shaded region -/
theorem shaded_area_is_85_over_8 :
  let sq : Square := { side := 12 }
  let rect : Rectangle := { width := 4, height := 5 }
  shaded_area sq rect = 85 / 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_85_over_8_l532_53248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_primes_equation_l532_53246

theorem quadruple_primes_equation (p q r : Nat) (n : Nat) :
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p^2 = q^2 + r^n →
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_primes_equation_l532_53246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l532_53294

theorem binomial_expansion_constant_term :
  ∀ n : ℕ, 
    4^n + 2^n = 72 →
    ∃ (a : ℕ → ℚ), 
      (∀ k, 0 ≤ k ∧ k ≤ n → a k = (Nat.choose n k : ℚ) * 3^k) ∧ 
      a 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l532_53294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_l532_53267

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 2) / (x^2 - 4*x + 3)

-- State the theorem
theorem limit_of_f_at_one :
  Filter.Tendsto f (nhds 1) (nhds (1/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_l532_53267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_l532_53221

/-- Inverse proportion function passing through (2, -3) -/
noncomputable def inverse_proportion (x : ℝ) : ℝ := -6 / x

/-- The point (-2, 3) -/
def point : ℝ × ℝ := (-2, 3)

theorem point_on_graph :
  inverse_proportion point.fst = point.snd := by
  unfold inverse_proportion point
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_graph_l532_53221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l532_53219

-- Define the constants as noncomputable
noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

-- State the theorem
theorem ordering_of_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l532_53219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l532_53272

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of a polygon given its vertices -/
noncomputable def perimeter (vertices : List Point) : ℝ :=
  let pairs := List.zip vertices (vertices.rotateRight 1)
  pairs.foldl (fun acc (p1, p2) => acc + distance p1 p2) 0

/-- Theorem: The perimeter of polygon ABCDE is 33 units -/
theorem polygon_perimeter : 
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨4, 8⟩
  let C : Point := ⟨4, 4⟩
  let D : Point := ⟨0, -2⟩
  let E : Point := ⟨9, -2⟩
  perimeter [A, B, C, D, E] = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l532_53272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_sample_id_is_18_l532_53280

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  firstSampleId : Nat

/-- Calculates the interval between samples -/
def SystematicSampling.interval (s : SystematicSampling) : Nat :=
  s.totalStudents / s.sampleSize

/-- Calculates the ID of the nth sample -/
def SystematicSampling.nthSampleId (s : SystematicSampling) (n : Nat) : Nat :=
  s.firstSampleId + (n - 1) * s.interval

/-- Theorem: In the given systematic sampling scenario, the fourth sample ID is 18 -/
theorem fourth_sample_id_is_18 (s : SystematicSampling) 
    (h1 : s.totalStudents = 48)
    (h2 : s.sampleSize = 4)
    (h3 : s.firstSampleId = 6) :
    s.nthSampleId 4 = 18 := by
  sorry

#check fourth_sample_id_is_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_sample_id_is_18_l532_53280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l532_53204

noncomputable def f (x : ℝ) := x / (x^2 + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  f (1/2) = 2/5 ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) ∧  -- f is increasing on (0, 1)
  (∀ x y, 1 < x ∧ x < y → f x > f y)  -- f is decreasing on (1, +∞)
  := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l532_53204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_correct_l532_53201

/-- Two lines in a coordinate plane -/
structure TwoLines where
  m : ℝ
  n : ℝ
  p : ℝ
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  h1 : line1 n = m
  h2 : line1 (n + 15) = m + p
  h3 : ∀ y, line1 y = y / 5 - 2 / 5
  h4 : ∀ y, line2 y = 3 * y / 7 + 1 / 7

/-- The value of p and the intersection point of the two lines -/
noncomputable def solution (tl : TwoLines) : ℝ × (ℝ × ℝ) :=
  (3, (-7/8, -19/8))

/-- Theorem stating that the solution is correct -/
theorem solution_correct (tl : TwoLines) : 
  solution tl = (tl.p, (tl.line1 (-19/8), -19/8)) ∧ 
  tl.line1 (-19/8) = tl.line2 (-19/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_correct_l532_53201
