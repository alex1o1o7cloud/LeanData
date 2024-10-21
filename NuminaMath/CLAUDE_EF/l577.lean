import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l577_57717

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  c = 2 →
  Real.sin A = 2 * Real.sin C →
  Real.cos B = 1 / 4 →
  let S := (1 / 2) * a * c * Real.sin B
  S = Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l577_57717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roy_trip_distance_l577_57738

/-- Represents the details of Roy's car trip -/
structure CarTrip where
  batteryDistance : ℝ
  firstGasolineDistance : ℝ
  firstGasolineRate : ℝ
  secondGasolineRate : ℝ
  averageEfficiency : ℝ

/-- Calculates the total trip distance given the car trip details -/
noncomputable def totalTripDistance (trip : CarTrip) : ℝ :=
  trip.batteryDistance + trip.firstGasolineDistance + 
    (trip.averageEfficiency * (trip.firstGasolineDistance * trip.firstGasolineRate + 
    (trip.averageEfficiency * (trip.firstGasolineDistance * trip.firstGasolineRate + 
    trip.batteryDistance / trip.averageEfficiency) - trip.batteryDistance - trip.firstGasolineDistance) * 
    trip.secondGasolineRate) - trip.batteryDistance)

/-- Theorem stating that Roy's trip distance is 360 miles -/
theorem roy_trip_distance : 
  let trip := CarTrip.mk 60 100 0.03 0.015 75
  totalTripDistance trip = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roy_trip_distance_l577_57738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_regular_tetrahedron_with_equal_altitude_segments_l577_57721

/-- A tetrahedron in 3D space --/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The inscribed sphere of a tetrahedron --/
structure InscribedSphere (t : Tetrahedron) where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The altitude of a tetrahedron from a vertex to the opposite face --/
def altitude (t : Tetrahedron) (v : Fin 4) : ℝ × ℝ × ℝ :=
  sorry

/-- The segment of an altitude that falls inside the inscribed sphere --/
def altitudeSegmentInSphere (t : Tetrahedron) (s : InscribedSphere t) (v : Fin 4) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

/-- The length of a line segment --/
def segmentLength (p : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- A tetrahedron is regular if all its edges have the same length --/
def isRegular (t : Tetrahedron) : Prop :=
  sorry

/-- Theorem: There exists a non-regular tetrahedron with equal altitude segments in its inscribed sphere --/
theorem non_regular_tetrahedron_with_equal_altitude_segments :
  ∃ (t : Tetrahedron) (s : InscribedSphere t),
    (∀ (v w : Fin 4), segmentLength (altitudeSegmentInSphere t s v) = segmentLength (altitudeSegmentInSphere t s w)) ∧
    ¬(isRegular t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_regular_tetrahedron_with_equal_altitude_segments_l577_57721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_a_in_zero_one_l577_57758

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

-- State the theorem
theorem f_has_minimum_iff_a_in_zero_one (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_a_in_zero_one_l577_57758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_P_l577_57744

/-- The set of all prime numbers -/
def P : Set ℕ := {n : ℕ | Nat.Prime n}

/-- A subset M of P satisfying the given conditions -/
def M : Set ℕ := sorry

/-- M is a subset of P -/
axiom M_subset_P : M ⊆ P

/-- M has at least 3 elements -/
axiom M_size : ∃ (a b c : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- For any finite subset A of M, all prime factors of (product of A) - 1 are in M -/
axiom M_property (A : Finset ℕ) (hA : ↑A ⊆ M) :
  ∀ p, p ∈ P → p ∣ ((A.prod id) - 1) → p ∈ M

/-- The main theorem: M equals P -/
theorem M_equals_P : M = P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_P_l577_57744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_is_60_degrees_l577_57782

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the foci
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 7, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem angle_measure_is_60_degrees (P : ℝ × ℝ) 
  (h1 : is_on_ellipse P.1 P.2) 
  (h2 : distance P left_focus * distance P right_focus = 12) : 
  Real.arccos ((distance P left_focus)^2 + (distance P right_focus)^2 - 28) / 
    (2 * distance P left_focus * distance P right_focus) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_is_60_degrees_l577_57782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_product_values_l577_57774

theorem sine_product_values (α β γ : ℝ) :
  (Real.sin α = Real.sin (α + β + γ) + 1) →
  (Real.sin β = 3 * Real.sin (α + β + γ) + 2) →
  (Real.sin γ = 5 * Real.sin (α + β + γ) + 3) →
  (Real.sin α * Real.sin β * Real.sin γ = 3/64 ∨ Real.sin α * Real.sin β * Real.sin γ = 1/8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_product_values_l577_57774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l577_57716

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² - c² = 2b and sin B = 6 cos A * sin C, then b = 3 -/
theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensuring positive side lengths
  0 < A ∧ A < π →  -- Ensuring valid angle measures
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Sine rule
  a * Real.sin C = c * Real.sin A →  -- Sine rule
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine rule
  a^2 - c^2 = 2*b →  -- Given condition
  Real.sin B = 6 * Real.cos A * Real.sin C →  -- Given condition
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l577_57716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_inequality_system_solution_l577_57777

theorem calculation_proof :
  (3.14 - Real.pi) ^ (0 : ℝ) - (1/2)^(-2 : ℝ) + 2 * Real.cos (60 * π / 180) - |1 - Real.sqrt 3| + Real.sqrt 12 = Real.sqrt 3 - 1 := by
  sorry

theorem inequality_system_solution :
  Set.Icc (-3 : ℝ) 3 ∩ Set.Iio 3 = {x : ℝ | 2*x - 6 < 0 ∧ (1 - 3*x) / 2 ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_inequality_system_solution_l577_57777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kira_time_away_correct_l577_57772

/-- Represents the time Kira was away from home -/
noncomputable def time_away : ℝ := 7.2

/-- Cat A's eating rate in pounds per hour -/
noncomputable def cat_a_rate : ℝ := 1 / 4

/-- Cat B's eating rate in pounds per hour -/
noncomputable def cat_b_rate : ℝ := 1 / 6

/-- Initial amount of kibble in pounds -/
noncomputable def initial_kibble : ℝ := 4

/-- Final amount of kibble in pounds -/
noncomputable def final_kibble : ℝ := 1

/-- Theorem stating that the time Kira was away is correct given the conditions -/
theorem kira_time_away_correct :
  time_away * (cat_a_rate + cat_b_rate) = initial_kibble - final_kibble :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kira_time_away_correct_l577_57772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inverse_evaluation_l577_57746

theorem complex_inverse_evaluation (i : ℂ) (h : i^2 = -1) :
  (2*i + 3*i⁻¹)⁻¹ = i := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inverse_evaluation_l577_57746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_my_geometric_series_l577_57770

noncomputable def geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

noncomputable def series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_convergence (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∃ (L : ℝ), L = series_sum a r ∧
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |geometric_series a r n - L| < ε :=
sorry

theorem my_geometric_series :
  let a : ℝ := 3
  let r : ℝ := 1/4
  let L : ℝ := series_sum a r
  (L = 4) ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |geometric_series a r n - L| < ε) ∧
  (∃ (limit : ℝ), ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |geometric_series a r n - limit| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_my_geometric_series_l577_57770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l577_57702

theorem sin_plus_cos_value (x : ℝ) 
  (h1 : Real.sin x * Real.cos x = -1/4)
  (h2 : 3*Real.pi/4 < x ∧ x < Real.pi) : 
  Real.sin x + Real.cos x = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l577_57702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_is_kite_l577_57724

/-- A quadrilateral with perpendicular diagonals, one of which is twice the length of the other. -/
structure SpecialQuadrilateral where
  /-- The quadrilateral -/
  Q : Type
  /-- The diagonals are perpendicular -/
  perpendicular_diagonals : Q → Prop
  /-- One diagonal is twice the length of the other -/
  diagonal_length_ratio : Q → Prop

/-- A kite is a quadrilateral with two pairs of adjacent congruent sides -/
def is_kite (Q : Type) : Prop := sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, where one diagonal is twice the length of the other, is a kite -/
theorem special_quadrilateral_is_kite (SQ : SpecialQuadrilateral) : is_kite SQ.Q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_is_kite_l577_57724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_inequality_l577_57700

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating that for an arithmetic sequence where S₅ > S₆,
    the inequality a₃ + a₆ + a₁₂ < 2a₇ does not necessarily hold -/
theorem arithmetic_seq_inequality (seq : ArithmeticSequence) 
  (h : S seq 5 > S seq 6) :
  ¬ (∀ seq : ArithmeticSequence, seq.a 3 + seq.a 6 + seq.a 12 < 2 * seq.a 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_inequality_l577_57700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l577_57730

/-- Given vectors a and b in ℝ², prove that their properties lead to the specified projection. -/
theorem vector_projection (a b : ℝ × ℝ) : 
  ‖a‖ = 2 →
  ‖b‖ = 3 →
  (∃ (l : ℝ), ∀ (μ : ℝ), ‖b - μ • a‖ ≥ ‖b - l • a‖ ∧ ‖b - l • a‖ = 2 * Real.sqrt 2) →
  let proj := (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) / ‖a‖
  proj = 3 ∨ proj = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l577_57730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_l577_57718

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

theorem unique_alpha : ∃! α : ℝ, 
  (∀ x : ℝ, ∃ y : ℝ, f α x = y) ∧ 
  (∀ x : ℝ, f α (-x) = -(f α x)) ∧
  α = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_l577_57718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l577_57780

theorem problem_solution (n a b : ℕ) 
  (h1 : (3 * a + 5 * b) % (n + 1) = 19)
  (h2 : (4 * a + 2 * b) % (n + 1) = 25) :
  2 * a + 6 * b = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l577_57780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_on_sphere_l577_57709

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Check if a point is on a sphere with radius 1 centered at origin -/
def isOnUnitSphere (p : Point3D) : Prop :=
  p.x^2 + p.y^2 + p.z^2 = 1

/-- Product of distances between four points -/
noncomputable def distanceProduct (a b c d : Point3D) : ℝ :=
  (distance a b) * (distance a c) * (distance a d) *
  (distance b c) * (distance b d) * (distance c d)

/-- Check if a tetrahedron is regular -/
def isRegularTetrahedron (a b c d : Point3D) : Prop :=
  distance a b = distance a c ∧ distance a b = distance a d ∧
  distance a b = distance b c ∧ distance a b = distance b d ∧
  distance a b = distance c d

theorem regular_tetrahedron_on_sphere (a b c d : Point3D) 
  (ha : isOnUnitSphere a) (hb : isOnUnitSphere b) (hc : isOnUnitSphere c) (hd : isOnUnitSphere d)
  (h_product : distanceProduct a b c d = 512/27) :
  isRegularTetrahedron a b c d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_on_sphere_l577_57709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l577_57735

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∃! a b : ℝ, 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ a ≠ b ∧ f ω a + f ω b = 4) →
  5 ≤ ω ∧ ω < 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l577_57735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_theorem_l577_57720

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians) -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Convert degrees to radians -/
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * Real.pi / 180

/-- Theorem about a specific triangle -/
theorem specific_triangle_theorem :
  ∃ (t : Triangle),
    t.a = 7.012 ∧
    t.c - t.b = 1.753 ∧
    t.B = deg_to_rad (38 + 12/60 + 48/3600) ∧
    abs (t.A - deg_to_rad (81 + 47/60 + 12.5/3600)) < 0.0001 ∧
    t.C = deg_to_rad 60 ∧
    abs (t.b - 4.3825) < 0.0001 ∧
    abs (t.c - 6.1355) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_theorem_l577_57720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l577_57757

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

-- State the theorem
theorem arithmetic_sequence_problem (m : ℕ) (d : ℝ) :
  m ≥ 2 →
  sum_arithmetic_sequence 1 d (m - 1) = 16 →
  sum_arithmetic_sequence 1 d m = 25 →
  m = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l577_57757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_slice_theorem_l577_57715

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  base_side : ℝ
  height : ℝ

/-- Represents a frustum formed by slicing a pyramid -/
structure Frustum where
  pyramid : SquarePyramid
  slice_height : ℝ

/-- The volume of a pyramid -/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

/-- The distance from the apex of the original pyramid to the center of the frustum's circumsphere -/
noncomputable def apex_to_circumcenter (f : Frustum) : ℝ :=
  (3/2) * f.slice_height

theorem pyramid_slice_theorem (p : SquarePyramid) (f : Frustum) 
  (h_base : p.base_side = 10)
  (h_height : p.height = 20)
  (h_volume_ratio : pyramid_volume p = 15 * pyramid_volume { base_side := p.base_side * (f.slice_height / p.height), height := f.slice_height })
  : apex_to_circumcenter f = 30 / (1 + Real.rpow 15 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_slice_theorem_l577_57715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_300_and_600_l577_57705

theorem count_even_numbers_between_300_and_600 : 
  Finset.card (Finset.filter (λ x => 300 < x ∧ x < 600 ∧ Even x) (Finset.range 600)) = 149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_between_300_and_600_l577_57705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_for_specific_table_l577_57707

/-- The length of a placemat on a circular table -/
noncomputable def placemat_length (r : ℝ) (n : ℕ) : ℝ :=
  2 * r * Real.sin (Real.pi / (2 * n))

/-- Theorem: The length of each placemat on a circular table with radius 5 and 8 placemats -/
theorem placemat_length_for_specific_table :
  placemat_length 5 8 = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

#check placemat_length_for_specific_table

end NUMINAMATH_CALUDE_ERRORFEEDBACK_placemat_length_for_specific_table_l577_57707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l577_57723

theorem evaluate_expression : (12 : ℝ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  -- Convert fractions to real numbers
  have h1 : (1/3 : ℝ) + (1/4 : ℝ) + (1/6 : ℝ) = 3/4 := by
    -- Proof steps here
    sorry
  
  -- Calculate the inverse
  have h2 : ((3/4 : ℝ)⁻¹) = 4/3 := by
    -- Proof steps here
    sorry

  -- Multiply by 12
  have h3 : (12 : ℝ) * (4/3 : ℝ) = 16 := by
    -- Proof steps here
    sorry

  -- Combine the steps
  calc
    (12 : ℝ) * (1/3 + 1/4 + 1/6)⁻¹ = (12 : ℝ) * ((3/4 : ℝ)⁻¹) := by rw [h1]
    _ = (12 : ℝ) * (4/3 : ℝ) := by rw [h2]
    _ = 16 := by rw [h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l577_57723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l577_57797

noncomputable section

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (Real.sqrt 6, 0)

/-- A point on the asymptote of the hyperbola -/
def asymptote_point : ℝ × ℝ := (Real.sqrt 6 / 2, Real.sqrt 3 / 2)

/-- The origin point -/
def origin : ℝ × ℝ := (0, 0)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

theorem hyperbola_triangle_area :
  hyperbola asymptote_point.1 asymptote_point.2 →
  distance asymptote_point origin = distance asymptote_point right_focus →
  triangle_area asymptote_point right_focus origin = 3 * Real.sqrt 2 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l577_57797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_2x_cos_2x_l577_57754

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.cos (2 * x)

theorem min_positive_period_sin_2x_cos_2x :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_2x_cos_2x_l577_57754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l577_57793

/-- Computes the final amount for an investment with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Computes the final amount for an investment with monthly compound interest -/
noncomputable def monthly_compound_interest (principal : ℝ) (annual_rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + annual_rate / 12) ^ (years * 12)

theorem investment_difference (principal : ℝ) (annual_rate : ℝ) (years : ℕ) :
  let yearly_compounded := compound_interest principal annual_rate years
  let monthly_compounded := monthly_compound_interest principal annual_rate years
  ∃ ε > 0, |monthly_compounded - yearly_compounded - 168.21| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l577_57793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l577_57722

open Real

/-- The solution to the Cauchy problem -/
noncomputable def solution (x : ℝ) : ℝ := 1 / x

theorem cauchy_problem_solution (x : ℝ) (hx : x ≠ 0) :
  let y := solution
  (∀ x, x ≠ 0 → (deriv y x - (1 / x) * y x = -2 / x^2)) ∧
  y 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l577_57722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABC_l577_57703

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point D as the midpoint of BC
noncomputable def D (A B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define point E on AC
noncomputable def E (A C : ℝ × ℝ) : ℝ × ℝ := ((2 * A.1 + 3 * C.1) / 5, (2 * A.2 + 3 * C.2) / 5)

-- Define point F on AD
noncomputable def F (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let D := D A B C
  ((2 * A.1 + D.1) / 3, (2 * A.2 + D.2) / 3)

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := 
  abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)

-- Theorem statement
theorem area_ABC (A B C : ℝ × ℝ) 
  (h : triangleArea (D A B C) (E A C) (F A B C) = 20) : 
  triangleArea A B C = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABC_l577_57703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l577_57741

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l577_57741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l577_57712

theorem distinct_remainders (n : ℕ) (hn : Odd n) (hn_pos : 0 < n) :
  let a : ℕ → ℕ := λ i => 3 * i
  let b : ℕ → ℕ := λ i => 3 * i - 1
  let sums : Fin n → Fin 3 → ℕ := λ i j =>
    match j with
    | 0 => a i.val + a ((i.val % n + 1) % n + 1)
    | 1 => a i.val + b i.val
    | 2 => b i.val + b ((i.val + k - 1) % n + 1)
  ∀ k : ℕ, 0 < k → k < n →
    Function.Injective (λ (p : Fin n × Fin 3) => (sums p.1 p.2) % (3 * n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l577_57712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_x_implies_cos_tan_2x_l577_57734

theorem tan_pi_4_plus_x_implies_cos_tan_2x (x : ℝ) :
  Real.tan (π / 4 + x) = 2014 → 1 / Real.cos (2 * x) + Real.tan (2 * x) = 2014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_x_implies_cos_tan_2x_l577_57734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_right_angle_range_l577_57708

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.fst - 3)^2 + (p.snd - 4)^2 = 1}

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for point P
def is_right_angle (P A B : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Theorem statement
theorem circle_right_angle_range (m : ℝ) :
  m > 0 →
  (∃ P : ℝ × ℝ, P ∈ C ∧ is_right_angle P (A m) (B m)) →
  4 ≤ m ∧ m ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_right_angle_range_l577_57708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_reaching_R_l577_57711

def coin_toss_paths (n : ℕ) : ℕ := 2^n

def paths_to_point (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_reaching_R (total_tosses : ℕ) (up_moves : ℕ) (right_moves : ℕ) :
  total_tosses = 4 →
  up_moves = 2 →
  right_moves = 2 →
  (paths_to_point total_tosses up_moves : ℚ) / (coin_toss_paths total_tosses) = 3/8 := by
  sorry

#check probability_of_reaching_R

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_reaching_R_l577_57711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expression_l577_57799

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  (∀ y, 0 < y ∧ y < 8 → ∃ x, x = (a - b) * c^2 ∧ y = x) ∧
  (∀ x, x = (a - b) * c^2 → 0 < x ∧ x < 8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_expression_l577_57799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l577_57778

/-- The distance between points A(1, 1, 2) and B(2, 3, 4) in three-dimensional Cartesian space is 3. -/
theorem distance_between_points :
  let A : Fin 3 → ℝ := ![1, 1, 2]
  let B : Fin 3 → ℝ := ![2, 3, 4]
  Real.sqrt ((B 0 - A 0)^2 + (B 1 - A 1)^2 + (B 2 - A 2)^2) = 3 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l577_57778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l577_57704

/-- The area of a triangle with vertices at (1,1), (1,6), and (8,9) is 17.5 square units. -/
theorem triangle_area : ∃ (v1 v2 v3 : ℝ × ℝ) (area : ℝ),
  v1 = (1, 1) ∧ v2 = (1, 6) ∧ v3 = (8, 9) ∧
  area = (1/2) * (v2.2 - v1.2) * (v3.1 - v1.1) ∧
  area = 17.5 := by
  -- Introduce the vertices and area
  let v1 : ℝ × ℝ := (1, 1)
  let v2 : ℝ × ℝ := (1, 6)
  let v3 : ℝ × ℝ := (8, 9)
  let area : ℝ := (1/2) * (v2.2 - v1.2) * (v3.1 - v1.1)
  
  -- Prove the existence of these points and the area calculation
  use v1, v2, v3, area
  
  -- Prove the equalities
  have h1 : v1 = (1, 1) := rfl
  have h2 : v2 = (1, 6) := rfl
  have h3 : v3 = (8, 9) := rfl
  have h4 : area = (1/2) * (v2.2 - v1.2) * (v3.1 - v1.1) := rfl
  
  -- Calculate the area
  have h5 : area = 17.5 := by
    rw [h4]
    norm_num
  
  -- Combine all the facts
  exact ⟨h1, h2, h3, h4, h5⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l577_57704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_volume_calculation_l577_57726

/-- Calculates the volume of water flowing per minute in a river -/
noncomputable def river_flow_volume (depth : ℝ) (width : ℝ) (flow_rate_kmph : ℝ) : ℝ :=
  let cross_sectional_area := depth * width
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  cross_sectional_area * flow_rate_mpm

/-- Theorem: The volume of water flowing per minute in the given river is 26000 cubic meters -/
theorem river_flow_volume_calculation :
  river_flow_volume 4 65 6 = 26000 := by
  -- Unfold the definition of river_flow_volume
  unfold river_flow_volume
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_volume_calculation_l577_57726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_properties_l577_57781

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | ∃ y, y = x^2 - 4}
def B : Set ℝ := {y : ℝ | ∃ x, y = x^2 - 4}
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 - 4}

-- State the theorem
theorem sets_properties : (A ∪ B = Set.univ) ∧ (A.prod Set.univ ∩ C = ∅) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_properties_l577_57781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l577_57731

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Define the propositions
def proposition1 (Line Plane : Type)
  (parallel : Line → Line → Prop)
  (parallel_line_plane : Line → Plane → Prop)
  (parallel_plane : Plane → Plane → Prop) : Prop :=
  ∀ (m n : Line) (α β : Plane),
    parallel_line_plane m α → parallel_line_plane n β → parallel_plane α β →
    parallel m n

def proposition2 (Line Plane : Type)
  (perpendicular_line_plane : Line → Plane → Prop)
  (perpendicular_plane : Plane → Plane → Prop)
  (intersection : Plane → Plane → Line → Prop) : Prop :=
  ∀ (m : Line) (α β γ : Plane),
    intersection α β m → perpendicular_plane α γ → perpendicular_plane β γ →
    perpendicular_line_plane m γ

def proposition3 (Line Plane : Type)
  (perpendicular_line_plane : Line → Plane → Prop)
  (perpendicular_plane : Plane → Plane → Prop)
  (perpendicular_line : Line → Line → Prop) : Prop :=
  ∀ (m n : Line) (α β : Plane),
    perpendicular_line_plane m α → perpendicular_line_plane n β → perpendicular_plane α β →
    perpendicular_line m n

-- The theorem to prove
theorem exactly_two_true :
  ∃ (Line Plane : Type)
    (parallel : Line → Line → Prop)
    (parallel_line_plane : Line → Plane → Prop)
    (parallel_plane : Plane → Plane → Prop)
    (perpendicular_line_plane : Line → Plane → Prop)
    (perpendicular_plane : Plane → Plane → Prop)
    (perpendicular_line : Line → Line → Prop)
    (intersection : Plane → Plane → Line → Prop),
  (¬ proposition1 Line Plane parallel parallel_line_plane parallel_plane ∧
   proposition2 Line Plane perpendicular_line_plane perpendicular_plane intersection ∧
   proposition3 Line Plane perpendicular_line_plane perpendicular_plane perpendicular_line) ∨
  (proposition1 Line Plane parallel parallel_line_plane parallel_plane ∧
   proposition2 Line Plane perpendicular_line_plane perpendicular_plane intersection ∧
   ¬ proposition3 Line Plane perpendicular_line_plane perpendicular_plane perpendicular_line) ∨
  (proposition1 Line Plane parallel parallel_line_plane parallel_plane ∧
   ¬ proposition2 Line Plane perpendicular_line_plane perpendicular_plane intersection ∧
   proposition3 Line Plane perpendicular_line_plane perpendicular_plane perpendicular_line) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l577_57731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_seq_convergence_l577_57706

noncomputable def x_seq (x₁ : ℝ) : ℕ → ℝ
  | 0 => x₁
  | n + 1 => (x_seq x₁ n ^ 3 + 3 * x_seq x₁ n) / (3 * x_seq x₁ n ^ 2 + 1)

theorem x_seq_convergence (x₁ : ℝ) :
  (x₁ < -1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |x_seq x₁ n + 1| < ε) ∧
  (-1 < x₁ ∧ x₁ < 0 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |x_seq x₁ n + 1| < ε) ∧
  (0 < x₁ ∧ x₁ < 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |x_seq x₁ n - 1| < ε) ∧
  (x₁ > 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |x_seq x₁ n - 1| < ε) ∧
  (x₁ = -1 ∨ x₁ = 0 ∨ x₁ = 1 → ∀ n : ℕ, x_seq x₁ n = x₁) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_seq_convergence_l577_57706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_power_sum_condition_l577_57742

theorem integer_power_sum_condition (n : ℕ) :
  (∃ x : ℚ, ¬ (∃ m : ℤ, x = ↑m) ∧ ∃ k : ℤ, (x : ℝ)^n + (x+1 : ℝ)^n = k) ↔ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_power_sum_condition_l577_57742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_l577_57752

/-- Given an equilateral triangle DEF with a point Q inside, prove that the area of DEF is approximately 26 -/
theorem area_of_equilateral_triangle (D E F Q : ℝ × ℝ) : 
  let dist := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- DEF is equilateral
  (dist D E = dist E F) ∧ (dist E F = dist F D) ∧ (dist F D = dist D E) →
  -- Q is inside DEF
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
    Q.1 = a * D.1 + b * E.1 + c * F.1 ∧
    Q.2 = a * D.2 + b * E.2 + c * F.2) →
  -- Given distances
  dist D Q = 5 →
  dist E Q = 7 →
  dist F Q = 9 →
  -- Area of DEF is approximately 26
  ∃ (area : ℝ), area > 25.5 ∧ area < 26.5 ∧
  area = (Real.sqrt 3 / 4) * (dist D E)^2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_l577_57752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l577_57760

-- Define the ellipse
def Ellipse (a b : ℝ) := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}

-- Define the foci
noncomputable def Foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let c := Real.sqrt (a^2 - b^2)
  (-c, 0, c, 0)

-- Define the incenter of a triangle
noncomputable def Incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the area of a triangle
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_eccentricity_half 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (P : ℝ × ℝ) 
  (h3 : P ∈ Ellipse a b) 
  (F1 F2 : ℝ × ℝ) 
  (h4 : (F1.1, F1.2, F2.1, F2.2) = Foci a b) 
  (I : ℝ × ℝ) 
  (h5 : I = Incenter P F1 F2) 
  (h6 : TriangleArea I P F1 + TriangleArea I P F2 = 2 * TriangleArea I F1 F2) : 
  Real.sqrt (a^2 - b^2) / a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l577_57760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_count_is_three_l577_57745

/-- Represents the fruit basket contents and prices -/
structure FruitBasket where
  banana_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grape_price : ℚ
  total_price : ℚ

/-- The number of apples in the fruit basket -/
def apple_count (fb : FruitBasket) : ℚ :=
  (fb.total_price
    - (fb.banana_count * fb.banana_price
    + fb.strawberry_count * fb.strawberry_price / 12
    + fb.avocado_count * fb.avocado_price
    + 2 * fb.grape_price)) / fb.apple_price

/-- Theorem stating that the number of apples in the fruit basket is 3 -/
theorem apple_count_is_three (fb : FruitBasket)
  (h1 : fb.banana_count = 4)
  (h2 : fb.strawberry_count = 24)
  (h3 : fb.avocado_count = 2)
  (h4 : fb.banana_price = 1)
  (h5 : fb.apple_price = 2)
  (h6 : fb.strawberry_price = 4)
  (h7 : fb.avocado_price = 3)
  (h8 : fb.grape_price = 2)
  (h9 : fb.total_price = 28) :
  apple_count fb = 3 := by
  sorry

#eval apple_count {
  banana_count := 4,
  strawberry_count := 24,
  avocado_count := 2,
  banana_price := 1,
  apple_price := 2,
  strawberry_price := 4,
  avocado_price := 3,
  grape_price := 2,
  total_price := 28
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_count_is_three_l577_57745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_attendees_l577_57794

/-- Proves that given the concert conditions, the total number of attendees is 1500 --/
theorem concert_attendees :
  ∀ (adults children : ℕ) (adult_price child_price total_revenue : ℚ),
  children = 3 * adults →
  adult_price = 7 →
  child_price = 3 →
  total_revenue = 6000 →
  adult_price * adults + child_price * children = total_revenue →
  adults + children = 1500 := by
  intro adults children adult_price child_price total_revenue
  intro h_children h_adult_price h_child_price h_total_revenue h_revenue_equation
  sorry

#check concert_attendees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_attendees_l577_57794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l577_57714

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def triangle : Triangle := {
  A := 0,  -- placeholder, will be determined
  B := Real.arccos (1/7),
  C := 0,  -- placeholder, will be determined
  a := 0,  -- placeholder, will be determined
  b := 0,  -- placeholder, will be determined
  c := 5
}

-- Theorem statement
theorem triangle_problem (k : ℤ) : 
  (∀ x ∈ Set.Icc (-(π/6) + k * π) ((π/3) + k * π), Monotone f) ∧ 
  triangle.A = π/3 ∧ 
  triangle.a = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l577_57714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_is_worst_player_l577_57740

-- Define the set of players
inductive Player : Type
  | Man : Player
  | Sister : Player
  | Daughter : Player
  | Son : Player

-- Define the sex of a player
inductive Sex : Type
  | Male : Sex
  | Female : Sex

-- Define the age of a player
def Age : Type := ℕ

-- Define the cousin relationship
def is_cousin : Player → Player → Prop := sorry

-- Define the sex of a player
def sex : Player → Sex := sorry

-- Define the age of a player
def age : Player → Age := sorry

-- Define the best and worst players
def best_player : Player := sorry
def worst_player : Player := sorry

-- State the theorem
theorem daughter_is_worst_player :
  (∀ p : Player, p = Player.Man ∨ p = Player.Sister ∨ p = Player.Daughter ∨ p = Player.Son) →
  (∃ p : Player, is_cousin best_player p) →
  (∀ p : Player, is_cousin best_player p → sex p ≠ sex worst_player) →
  (age best_player = age worst_player) →
  worst_player = Player.Daughter :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_is_worst_player_l577_57740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_l577_57791

/-- The radius of a cylinder satisfying specific volume conditions -/
theorem cylinder_radius : ∃ r : ℝ, r > 0 ∧ 
  (let h : ℝ := 3;
   π * (r + 4)^2 * h = π * r^2 * (h + 12) ∧
   r = 1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_l577_57791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_theorem_l577_57750

theorem class_size_theorem (total_classes total_students : ℕ) :
  total_classes = 30 →
  total_students = 1000 →
  ∃ (class_size : ℕ), class_size ≤ total_classes ∧ 34 ≤ class_size :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_theorem_l577_57750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l577_57768

/-- The function f(x) = e^x + e^(2-x) -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

/-- The proposition that the inequality [f(x)]^2 - af(x) ≤ 0 has exactly 3 integer solutions -/
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), (f (w : ℝ))^2 - a * f (w : ℝ) ≤ 0 ↔ (w = x ∨ w = y ∨ w = z))

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a :
  ∀ a : ℝ, has_three_integer_solutions a →
    (1 + Real.exp 2 ≤ a ∧ a < Real.exp (-1) + Real.exp 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l577_57768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parcels_variance_l577_57753

def parcels : List ℕ := [1, 3, 2, 2, 2]

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs) ^ 2)).sum / xs.length

theorem parcels_variance :
  variance (parcels.map (λ x => (x : ℝ))) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parcels_variance_l577_57753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_digit_number_divisible_by_2_pow_n_l577_57767

def digits (x : ℕ) : List ℕ :=
  if x < 10 then [x]
  else (x % 10) :: digits (x / 10)

theorem exists_n_digit_number_divisible_by_2_pow_n (n : ℕ) :
  ∃ x : ℕ,
    (10^(n-1) ≤ x ∧ x < 10^n) ∧  -- x is an n-digit number
    (x % 2^n = 0) ∧              -- x is divisible by 2^n
    (∀ d : ℕ, d ∈ digits x → d = 1 ∨ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_digit_number_divisible_by_2_pow_n_l577_57767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_zero_when_cos_product_negative_one_l577_57784

theorem sin_sum_zero_when_cos_product_negative_one (α β : ℝ) :
  Real.cos α * Real.cos β = -1 → Real.sin (α + β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_zero_when_cos_product_negative_one_l577_57784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_range_of_a_l577_57756

/-- The function f(x) defined on [1, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log x + 2 / x

/-- f is monotonic on [1, +∞) -/
def is_monotonic (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ y → (f a x ≤ f a y ∨ f a y ≤ f a x)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_range_of_a :
  ∀ a, is_monotonic a ↔ a ∈ Set.Ici (0 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_range_of_a_l577_57756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l577_57759

/-- The number of classes --/
def num_classes : ℕ := 4

/-- The maximum number of students per class --/
def max_students_per_class : ℕ := 30

/-- The percentage of students who received a grade of 5 --/
def grade_5_percentage : ℚ := 28 / 100

/-- The percentage of students who received a grade of 4 --/
def grade_4_percentage : ℚ := 35 / 100

/-- The percentage of students who received a grade of 3 --/
def grade_3_percentage : ℚ := 25 / 100

/-- The percentage of students who received a grade of 2 --/
def grade_2_percentage : ℚ := 12 / 100

/-- The total number of students who took the exam --/
def total_students : ℕ := 100

theorem exam_students_count :
  ∃ (n : ℕ), n = total_students ∧
    n ≤ num_classes * max_students_per_class ∧
    (n * grade_5_percentage).num % (n * grade_5_percentage).den = 0 ∧
    (n * grade_4_percentage).num % (n * grade_4_percentage).den = 0 ∧
    (n * grade_3_percentage).num % (n * grade_3_percentage).den = 0 ∧
    (n * grade_2_percentage).num % (n * grade_2_percentage).den = 0 ∧
    grade_5_percentage + grade_4_percentage + grade_3_percentage + grade_2_percentage = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l577_57759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_on_log_curve_l577_57713

/-- The function representing the curve on which the quadrilateral vertices lie -/
noncomputable def f (x : ℝ) : ℝ := Real.log x

/-- The area of the quadrilateral -/
noncomputable def S : ℝ := Real.log (91/90)

/-- The x-coordinate of the leftmost vertex -/
def n : ℕ := 12

/-- The theorem stating the properties of the quadrilateral and its leftmost vertex -/
theorem quadrilateral_on_log_curve :
  (∀ i : Fin 4, f (n + i) = Real.log (n + i)) → 
  (∀ i : Fin 4, (n + i : ℕ) > 0) →
  (f (n + 1) + f (n + 2) + f (n + 3) - 3 * f n) / 2 = S →
  n = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_on_log_curve_l577_57713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l577_57771

structure Geometry where
  ABCD : Set (ℝ × ℝ)
  EFGH : Set (ℝ × ℝ)
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  is_square : Set (ℝ × ℝ) → ℝ → Prop
  side_length : Set (ℝ × ℝ) → ℝ
  on_side : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop
  intersection : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop
  distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ
  collinear : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop

def geometry_problem (g : Geometry) : Prop :=
  g.is_square g.ABCD (g.side_length g.ABCD) ∧
  g.is_square g.EFGH (g.side_length g.EFGH) ∧
  g.side_length g.ABCD = 33 ∧
  g.side_length g.EFGH = 12 ∧
  g.on_side g.E g.F g.D g.C ∧
  g.intersection g.H g.B g.D g.C g.X ∧
  g.distance g.D g.E = 18 →
  (g.distance g.E g.X = 4 ∧ g.collinear g.A g.X g.G)

theorem geometry_theorem (g : Geometry) : geometry_problem g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l577_57771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l577_57719

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - Real.sqrt (3 - Real.sqrt (4 - x)))

-- Define the domain of f
def domain (x : ℝ) : Prop :=
  2 - Real.sqrt (3 - Real.sqrt (4 - x)) ≥ 0 ∧
  3 - Real.sqrt (4 - x) ≥ 0 ∧
  4 - x ≥ 0

-- Theorem stating that the domain of f is [0, 4]
theorem domain_of_f :
  ∀ x : ℝ, domain x ↔ 0 ≤ x ∧ x ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l577_57719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_initial_investment_l577_57787

/-- Represents the initial investment of partner A in rupees -/
def A_investment : ℚ := 3500

/-- Represents the investment of partner B in rupees -/
def B_investment : ℚ := 9000

/-- Represents the number of months A invested before B joined -/
def months_before_B : ℚ := 5

/-- Represents the total number of months in a year -/
def total_months : ℚ := 12

/-- Represents the profit ratio of A to B -/
def profit_ratio : ℚ := 2 / 3

theorem A_initial_investment :
  A_investment * total_months * profit_ratio.den = 
  B_investment * (total_months - months_before_B) * profit_ratio.num := by
  norm_num
  rfl

#eval A_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_initial_investment_l577_57787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_is_rhombus_l577_57749

/-- A convex quadrilateral with perimeter 10^100 and natural number side lengths -/
structure ConvexQuadrilateral where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  perimeter_eq : a + b + c + d = 10^100
  sum_divisible_by_fourth : 
    (a + b + c) % d = 0 ∧
    (a + b + d) % c = 0 ∧
    (a + c + d) % b = 0 ∧
    (b + c + d) % a = 0

/-- The quadrilateral is a rhombus if all sides are equal -/
def is_rhombus (q : ConvexQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

/-- Theorem: A convex quadrilateral with the given properties is a rhombus -/
theorem convex_quadrilateral_is_rhombus (q : ConvexQuadrilateral) : is_rhombus q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_is_rhombus_l577_57749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l577_57736

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.cos α = 1/3)
  (h2 : Real.cos (α + β) = -1/3)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) :
  Real.cos (α - β) = 23/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l577_57736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_properties_l577_57725

noncomputable section

-- Define the circles O₁ and O₂
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - y = 0

-- Define the perpendicular bisector of AB
def perp_bisector_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the maximum distance from a point on O₁ to line AB
noncomputable def max_distance : ℝ := Real.sqrt 2 / 2 + 1

theorem circles_intersection_properties :
  ∃ (A B : ℝ × ℝ),
    (circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2) ∧
    (circle_O₁ B.1 B.2 ∧ circle_O₂ B.1 B.2) ∧
    (∀ (x y : ℝ), line_AB x y ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) ∧
    (∀ (x y : ℝ), perp_bisector_AB x y ↔ (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) ∧
    (∀ (P : ℝ × ℝ), circle_O₁ P.1 P.2 → 
      ∀ (Q : ℝ × ℝ), line_AB Q.1 Q.2 → 
        (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≤ max_distance^2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_properties_l577_57725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_histogram_area_sum_k2_statistic_relationship_l577_57727

-- Define a frequency distribution histogram
structure FrequencyHistogram where
  rectangles : List (ℝ × ℝ) -- List of (height, width) pairs for each rectangle

-- Define the K^2 statistic for categorical variables
structure K2Statistic where
  observed_value : ℝ
  x : Type
  y : Type

-- Helper function to represent relationship strength (not part of the original problem)
noncomputable def relationship_strength (k : K2Statistic) : ℝ := sorry

-- Theorem 1: The sum of areas of all rectangles in a frequency histogram is 1
theorem frequency_histogram_area_sum (h : FrequencyHistogram) : 
  (h.rectangles.map (λ r => r.1 * r.2)).sum = 1 := by sorry

-- Theorem 2: Larger K^2 statistic indicates stronger relationship between variables
theorem k2_statistic_relationship (k1 k2 : K2Statistic) (h : k1.x = k2.x ∧ k1.y = k2.y) :
  k1.observed_value > k2.observed_value → 
  (∃ r : ℝ, r > 0 ∧ relationship_strength k1 = relationship_strength k2 + r) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_histogram_area_sum_k2_statistic_relationship_l577_57727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_power_seven_l577_57776

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]
variable (N : Matrix n n ℝ)

theorem det_N_power_seven (h : det N = 3) : det (N^7) = 2187 := by
  have h1 : det (N^7) = (det N)^7 := by
    simp [Matrix.det_pow]
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_power_seven_l577_57776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_equality_l577_57775

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line type -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Point type -/
abbrev Point := ℝ × ℝ

/-- Intersection of a line and a circle -/
noncomputable def intersect (l : Line) (c : Circle) : Set Point := sorry

/-- Tangent point of two circles -/
noncomputable def tangentPointCircles (c1 c2 : Circle) : Point := sorry

/-- Tangent point of a circle and a line -/
noncomputable def tangentPointCircleLine (c : Circle) (l : Line) : Point := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem circle_line_intersection_equality 
  (k1 k2 : Circle) (l : Line) (A B C D T : Point) : 
  k2.center.1 > k1.center.1 → -- k2 is outside k1
  A ∈ intersect l k1 →
  B ∈ intersect l k1 →
  A ≠ B →
  C = tangentPointCircles k1 k2 →
  D = tangentPointCircleLine k2 l →
  T ∈ intersect (Line.mk C D) k1 →
  T ≠ C →
  distance A T = distance T B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_equality_l577_57775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_winning_strategy_l577_57789

/-- Represents a card in the game -/
structure Card where
  id : Nat
  deriving Repr

/-- Represents the state of the game -/
structure GameState where
  player1_cards : List Card
  player2_cards : List Card
  deriving Repr

/-- Represents the strength relationship between cards -/
def beats (card1 card2 : Card) : Prop := sorry

/-- Represents a valid game state -/
def valid_game_state (state : GameState) (n : Nat) : Prop :=
  state.player1_cards.length + state.player2_cards.length = n ∧
  state.player1_cards.length > 0 ∧ state.player2_cards.length > 0

/-- Represents a final game state where one player has all cards -/
def final_state (state : GameState) : Prop :=
  state.player1_cards.length = 0 ∨ state.player2_cards.length = 0

/-- Represents a strategy for playing the game -/
def Strategy := GameState → GameState

/-- Theorem: There exists a strategy that leads to a final state -/
theorem exists_winning_strategy (n : Nat) (initial_state : GameState) 
  (h : valid_game_state initial_state n) :
  ∃ (strategy : Strategy), ∃ (k : Nat), 
    final_state (Nat.iterate strategy k initial_state) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_winning_strategy_l577_57789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l577_57762

/-- Represents a statement that can be true or false -/
inductive Statement
| irrational_infinite_decimal
| rational_one_to_one
| abs_equal_self
| pi_half_fraction
| approx_range

/-- Determines if a given statement is correct -/
def is_correct (s : Statement) : Bool :=
  match s with
  | Statement.irrational_infinite_decimal => true
  | Statement.rational_one_to_one => false
  | Statement.abs_equal_self => false
  | Statement.pi_half_fraction => false
  | Statement.approx_range => true

/-- The list of all statements -/
def all_statements : List Statement :=
  [Statement.irrational_infinite_decimal,
   Statement.rational_one_to_one,
   Statement.abs_equal_self,
   Statement.pi_half_fraction,
   Statement.approx_range]

/-- Counts the number of correct statements -/
def count_correct_statements : Nat :=
  (all_statements.filter is_correct).length

theorem correct_statements_count :
  count_correct_statements = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l577_57762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l577_57779

/-- The parabola defined by y^2 = 16x -/
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

/-- The focus of the parabola y^2 = 16x -/
def focus : ℝ × ℝ := (4, 0)

/-- Point P on the parabola -/
def P : ℝ × ℝ := (9, 12)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_to_focus :
  parabola P.1 P.2 → distance P focus = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l577_57779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l577_57710

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sqrt 3 * Real.sin α)

def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 2 / Real.sin (θ + Real.pi/4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_C₁_C₂ : 
  ∃ (α θ : ℝ), ∀ (β γ : ℝ), distance (C₁ α) (C₂ θ) ≤ distance (C₁ β) (C₂ γ) ∧ 
  distance (C₁ α) (C₂ θ) = Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l577_57710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_formula_l577_57763

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Prime factorization of a natural number -/
structure PrimeFactorization (n : ℕ) where
  primes : List ℕ
  exponents : List ℕ
  is_factorization : n = (primes.zip exponents).foldl (fun acc (p, e) => acc * p ^ e) 1
  is_prime_list : ∀ p ∈ primes, Nat.Prime p
  is_sorted : primes.Sorted (· < ·)

/-- Theorem: Euler's totient function formula for prime factorization -/
theorem euler_totient_formula {n : ℕ} (f : PrimeFactorization n) :
  phi n = (f.primes.zip f.exponents).foldl
    (fun acc (p, e) => acc * (p - 1) * p ^ (e - 1))
    1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_formula_l577_57763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_after_two_years_l577_57733

noncomputable def tree_height (years : ℕ) : ℝ :=
  243 / (3 ^ (5 - years))

theorem height_after_two_years :
  tree_height 2 = 9 := by
  -- Unfold the definition of tree_height
  unfold tree_height
  -- Simplify the expression
  simp [pow_sub]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_after_two_years_l577_57733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_negative_identity_l577_57786

open Matrix

theorem matrix_power_negative_identity {A : Matrix (Fin 2) (Fin 2) ℚ} 
  (h : ∃ (n : ℕ), n ≠ 0 ∧ A ^ n = -(1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_negative_identity_l577_57786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equivalence_l577_57743

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := (x^2 - 1) / (x - 1)
def g1 (x : ℝ) : ℝ := x + 1

def f2 (x : ℝ) : ℝ := abs x
noncomputable def g2 (x : ℝ) : ℝ := Real.sqrt (x^2)

def f3 (x : ℝ) : ℝ := x^2 - 2*x - 1
def g3 (t : ℝ) : ℝ := t^2 - 2*t - 1

-- Theorem statements
theorem functions_equivalence :
  (∀ x : ℝ, f2 x = g2 x) ∧ 
  (∀ x t : ℝ, f3 x = g3 t) ∧ 
  ¬(∀ x : ℝ, f1 x = g1 x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equivalence_l577_57743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_is_eight_thirteenths_l577_57739

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The initial rectangle -/
noncomputable def initial_rectangle : Rectangle := { length := 10, width := 6 }

/-- The smaller rectangle after folding and cutting -/
noncomputable def smaller_rectangle : Rectangle := { length := initial_rectangle.length / 2, width := initial_rectangle.width / 2 }

/-- The larger rectangle after folding and cutting -/
noncomputable def larger_rectangle : Rectangle := { length := initial_rectangle.length, width := initial_rectangle.width / 2 }

theorem perimeter_ratio_is_eight_thirteenths :
  perimeter smaller_rectangle / perimeter larger_rectangle = 8 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_is_eight_thirteenths_l577_57739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_27_negative_l577_57701

theorem cube_root_27_negative : -(27 : ℝ)^(1/3 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_27_negative_l577_57701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_theorem_l577_57788

-- Define the bus route
structure BusRoute where
  stops : ℕ
  max_passengers : ℕ

-- Define a passenger's journey
structure Journey where
  start : ℕ
  stop : ℕ

-- Define the theorem
theorem bus_route_theorem (route : BusRoute) 
  (h1 : route.stops = 14) 
  (h2 : route.max_passengers = 25) : 
  (∃ (A₁ B₁ A₂ B₂ A₃ B₃ A₄ B₄ : ℕ), 
    A₁ < B₁ ∧ A₂ < B₂ ∧ A₃ < B₃ ∧ A₄ < B₄ ∧
    A₁ < A₂ ∧ A₂ < A₃ ∧ A₃ < A₄ ∧
    B₁ < B₂ ∧ B₂ < B₃ ∧ B₃ < B₄ ∧
    (∀ (journey : Journey), 
      (journey.start = A₁ → journey.stop ≠ B₁) ∧
      (journey.start = A₂ → journey.stop ≠ B₂) ∧
      (journey.start = A₃ → journey.stop ≠ B₃) ∧
      (journey.start = A₄ → journey.stop ≠ B₄))) ∧
  (∃ (passenger_distribution : List Journey), 
    ¬∃ (A₁ B₁ A₂ B₂ A₃ B₃ A₄ B₄ A₅ B₅ : ℕ), 
      A₁ < B₁ ∧ A₂ < B₂ ∧ A₃ < B₃ ∧ A₄ < B₄ ∧ A₅ < B₅ ∧
      A₁ < A₂ ∧ A₂ < A₃ ∧ A₃ < A₄ ∧ A₄ < A₅ ∧
      B₁ < B₂ ∧ B₂ < B₃ ∧ B₃ < B₄ ∧ B₄ < B₅ ∧
      (∀ (journey : Journey), 
        (journey.start = A₁ → journey.stop ≠ B₁) ∧
        (journey.start = A₂ → journey.stop ≠ B₂) ∧
        (journey.start = A₃ → journey.stop ≠ B₃) ∧
        (journey.start = A₄ → journey.stop ≠ B₄) ∧
        (journey.start = A₅ → journey.stop ≠ B₅))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_theorem_l577_57788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l577_57748

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def line_through (p q : Point) (x y : ℝ) : Prop :=
  (y - p.y) * (q.x - p.x) = (x - p.x) * (q.y - p.y)

-- Main theorem
theorem hyperbola_property (a b : ℝ) (F P Q A B M : Point) :
  a > 0 → b > 0 →
  hyperbola a b F.x F.y →
  F.x = 2 → F.y = 0 →
  asymptote A.x A.y →
  asymptote B.x B.y →
  hyperbola a b P.x P.y →
  hyperbola a b Q.x Q.y →
  P.x > Q.x → Q.x > 0 → P.y > 0 →
  line_through F A M.x M.y →
  line_through F B M.x M.y →
  line_through P M (M.x + 1) (M.y - Real.sqrt 3) →
  line_through Q M (M.x + 1) (M.y + Real.sqrt 3) →
  line_through A B M.x M.y →
  (∃ (k : ℝ), line_through P Q (A.x + k) (A.y + k * (B.y - A.y) / (B.x - A.x))) →
  (M.x - A.x)^2 + (M.y - A.y)^2 = (M.x - B.x)^2 + (M.y - B.y)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l577_57748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_evaluation_at_zero_l577_57798

/-- The simplified form of the given expression -/
noncomputable def simplified_expression (x : ℝ) : ℝ := -x^2 - x

/-- The original expression -/
noncomputable def original_expression (x : ℝ) : ℝ :=
  ((2*x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))

theorem expression_simplification (x : ℝ) (h : x ≠ -1 ∧ x ≠ 2) :
  original_expression x = simplified_expression x := by
  sorry

theorem evaluation_at_zero :
  original_expression 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_evaluation_at_zero_l577_57798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_side_ratio_l577_57751

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧ 
  A + B + C = Real.pi

theorem triangle_angle_and_side_ratio 
  (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C) 
  (h_eq1 : b + a * Real.cos C = 0) 
  (h_eq2 : Real.sin A = 2 * Real.sin (A + C)) : 
  C = 2 * Real.pi / 3 ∧ c / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_side_ratio_l577_57751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_solution_l577_57764

/-- The volume of a parallelepiped generated by three vectors -/
def parallelepipedVolume (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := v1
  let (b1, b2, b3) := v2
  let (c1, c2, c3) := v3
  |a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)|

theorem parallelepiped_volume_solution :
  ∃ k : ℝ, k > 0 ∧ parallelepipedVolume (2,3,4) (1,k,2) (1,2,k) = 15 ∧ k = 9/2 := by
  use 9/2
  apply And.intro
  · norm_num
  apply And.intro
  · sorry -- Proof of volume calculation
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_solution_l577_57764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_sqrt_two_l577_57729

noncomputable def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

noncomputable def circle_points (θ : ℝ) : ℝ × ℝ × ℝ :=
  spherical_to_cartesian 2 θ (Real.pi / 4)

theorem circle_radius_sqrt_two :
  ∀ θ : ℝ, 
    let (x, y, z) := circle_points θ
    (x^2 + y^2 = 2) ∧ (z = Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_sqrt_two_l577_57729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_45_simplification_l577_57732

theorem tan_cot_45_simplification :
  let tan_45 := Real.tan (45 * Real.pi / 180)
  let cot_45 := 1 / tan_45
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_45_simplification_l577_57732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_count_l577_57765

/-- The number of positive divisors that 9240 and 8820 have in common -/
def common_divisors : ℕ := 24

/-- 9240 as a natural number -/
def a : ℕ := 9240

/-- 8820 as a natural number -/
def b : ℕ := 8820

/-- Theorem stating that the number of positive divisors that 9240 and 8820 have in common is 24 -/
theorem common_divisors_count : (Finset.filter (fun x => x∣a ∧ x∣b) (Finset.range (min a b + 1))).card = common_divisors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_count_l577_57765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l577_57761

/-- A rhombus inscribed in a circle with given diagonal lengths. -/
structure InscribedRhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  inscribed : Bool

/-- Calculate the perimeter of the rhombus. -/
noncomputable def perimeter (r : InscribedRhombus) : ℝ :=
  4 * Real.sqrt ((r.diagonal1 / 2) ^ 2 + (r.diagonal2 / 2) ^ 2)

/-- Calculate the radius of the circumscribed circle. -/
noncomputable def radius (r : InscribedRhombus) : ℝ :=
  max r.diagonal1 r.diagonal2 / 2

theorem rhombus_properties (r : InscribedRhombus) 
    (h1 : r.diagonal1 = 20)
    (h2 : r.diagonal2 = 16)
    (h3 : r.inscribed = true) :
    perimeter r = 16 * Real.sqrt 41 ∧ radius r = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l577_57761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l577_57796

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x|

-- Define the sequence of functions fₙ
def f_n : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (f_n n)

-- State the theorem
theorem solutions_count (n : ℕ) :
  ∃ (S : Finset ℝ), S.card = 2^n ∧
  (∀ x ∈ S, x ∈ Set.Icc (0 : ℝ) 1 ∧ f_n n x = 1/2 * x) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) 1, f_n n x = 1/2 * x → x ∈ S) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l577_57796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l577_57795

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Theorem: For an obtuse triangle with vertices at (9,3), (0,0), and (x,0) where x < 0,
    if the area of the triangle is 45 square units, then x = -30 -/
theorem third_vertex_coordinate (x : ℝ) :
  x < 0 →
  triangleArea (Point.mk 9 3) (Point.mk 0 0) (Point.mk x 0) = 45 →
  x = -30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l577_57795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_condition_l577_57747

/-- A linear equation in two variables x and y is of the form ax + by = c, where a and b are not both zero -/
def IsLinearEquation (a b : ℝ) : Prop := a ≠ 0 ∨ b ≠ 0

/-- Given equation ax + y = -1 -/
def GivenEquation (a x y : ℝ) : Prop := a * x + y = -1

theorem linear_equation_condition (a : ℝ) :
  (∀ x y : ℝ, IsLinearEquation a 1) ↔ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_condition_l577_57747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l577_57766

theorem test_maximum_marks (passing_percentage : ℝ) (student_score : ℕ) (marks_to_pass : ℕ) : 
  passing_percentage = 0.7 →
  student_score = 120 →
  marks_to_pass = 150 →
  ∃ (max_marks : ℕ), max_marks = 386 ∧ 
    (passing_percentage * (max_marks : ℝ) = (student_score + marks_to_pass : ℝ)) :=
by
  intro h_passing h_score h_marks
  use 386
  constructor
  · rfl
  · norm_num
    rw [h_passing, h_score, h_marks]
    norm_num
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l577_57766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_theorem_l577_57792

noncomputable section

-- Define the circle
def our_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points A, B, C
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the perpendicularity condition
def perpendicular (x y : ℝ) : Prop :=
  (x + 2) * (x - 1) + y * (-y) = 0

-- Main theorem
theorem circle_points_theorem :
  ∃ (x y : ℝ),
    (our_circle x y ∧ perpendicular x y) ∧
    ((x = 1.5 ∧ y^2 = 1.75) ∨ (x = -2 ∧ y = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_theorem_l577_57792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_property_l577_57755

/-- A reflection in ℝ² is a function that maps points to their mirror images across a line. -/
def Reflection : Type := 
  Fin 2 → ℝ → Fin 2 → ℝ

/-- Given a reflection in ℝ² that maps (3, -2) to (-2, 6), 
    it will map (5, 1) to (-67/17, 55/17). -/
theorem reflection_property (r : Reflection) :
  (∀ i, r i 3 0 = -2 ∧ r i (-2) 1 = 6) →
  (∀ i, r i 5 0 = -67/17 ∧ r i 1 1 = 55/17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_property_l577_57755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l577_57728

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.exp 1 - 2
noncomputable def b : ℝ := 1 - Real.log 2
noncomputable def c : ℝ := Real.exp (Real.exp 1) - Real.exp 2

-- State the theorem
theorem ordering_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l577_57728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l577_57783

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1 / 2) :
  a^2 + 8*a*b + 32*b^2 + 16*b*c + 8*c^2 ≥ 18 ∧
  (a^2 + 8*a*b + 32*b^2 + 16*b*c + 8*c^2 = 18 ↔ a = 4 ∧ b = 1/4 ∧ c = 2) := by
  sorry

#check min_value_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l577_57783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_sum_not_2004_exists_corner_sum_2005_l577_57773

/-- Represents a square with numbers at its vertices -/
structure Square where
  vertices : Fin 4 → Nat
  valid : ∀ i, vertices i ∈ ({1, 2, 3, 4} : Set Nat)
  sum_10 : (vertices 0) + (vertices 1) + (vertices 2) + (vertices 3) = 10

/-- Represents a stack of squares -/
def SquareStack := List Square

/-- Calculates the sum of numbers in a corner of the stack -/
def corner_sum (stack : SquareStack) (corner : Fin 4) : Nat :=
  stack.foldl (fun acc square ↦ acc + square.vertices corner) 0

/-- Theorem: The sum in each corner of the stack cannot be 2004 -/
theorem corner_sum_not_2004 (stack : SquareStack) :
  ¬(∀ corner : Fin 4, corner_sum stack corner = 2004) := by
  sorry

/-- Theorem: There exists a stack where the sum in each corner is 2005 -/
theorem exists_corner_sum_2005 :
  ∃ stack : SquareStack, ∀ corner : Fin 4, corner_sum stack corner = 2005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_sum_not_2004_exists_corner_sum_2005_l577_57773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l577_57769

-- Define the hyperbolas
def C₁ (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1
def C₂ (x y : ℝ) : Prop := y^2/9 - x^2/16 = 1

-- Define asymptotes
def asymptote (a b : ℝ) (x y : ℝ) : Prop := y = a * x + b ∨ y = a * x - b

-- Define the focal length of a hyperbola
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem hyperbola_properties :
  (∃ a b : ℝ, ∀ x y : ℝ, (C₁ x y ∨ C₂ x y) → asymptote a b x y) ∧
  (∀ x y : ℝ, ¬(C₁ x y ∧ C₂ x y)) ∧
  (focal_length 4 3 = focal_length 3 4) := by
  sorry

#check hyperbola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l577_57769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_AP_l577_57790

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Define a point P on the circle
def point_P (x y : ℝ) : Prop := circle_C x y

-- Define the slope of line AP
noncomputable def slope_AP (x y : ℝ) : ℝ := (y - point_A.2) / (x - point_A.1)

-- Theorem statement
theorem slope_range_AP :
  ∀ x y : ℝ, point_P x y →
  0 ≤ slope_AP x y ∧ slope_AP x y ≤ 4/3 := by
  sorry

#check slope_range_AP

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_AP_l577_57790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l577_57785

theorem min_value_expression (a b : ℕ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) :
  (∀ x y : ℕ, 0 < x ∧ x < 10 → 0 < y ∧ y < 10 → (2 : ℤ) * x - x * y ≥ (2 : ℤ) * a - a * b) →
  (2 : ℤ) * a - a * b = -63 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l577_57785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_above_paths_count_l577_57737

/-- The number of paths from (0,0) to (2n, 0) that stay above the x-axis without intersecting it -/
def abovePaths (n : ℕ) : ℚ :=
  (Nat.choose (2*n - 2) (n - 1)) / n

/-- Theorem stating the number of paths for n ≥ 3 -/
theorem above_paths_count (n : ℕ) (h : n ≥ 3) :
  abovePaths n = (Nat.choose (2*n - 2) (n - 1)) / n := by
  -- The proof is omitted for now
  sorry

#check above_paths_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_above_paths_count_l577_57737
