import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_properties_l139_13908

-- Define the basic structures
structure Plane where

structure Line where

-- Define perpendicularity relations
axiom perpendicular_planes (p1 p2 : Plane) : Prop

axiom perpendicular_line_plane (l : Line) (p : Plane) : Prop

axiom perpendicular_lines (l1 l2 : Line) : Prop

-- Define a line being in a plane
axiom line_in_plane (l : Line) (p : Plane) : Prop

-- Define the intersection line of two planes
axiom intersection_line (p1 p2 : Plane) : Line

-- Theorem statement
theorem perpendicular_planes_properties 
  (p1 p2 : Plane) 
  (h : perpendicular_planes p1 p2) :
  (∃ (l : Line), line_in_plane l p1 ∧ 
    (∃ (infinitely_many : Set Line), ∀ (l2 : Line), l2 ∈ infinitely_many → 
      line_in_plane l2 p2 ∧ perpendicular_lines l l2)) ∧
  (∀ (point : Line), line_in_plane point p1 → 
    let perp := intersection_line p1 p2
    perpendicular_lines point perp → perpendicular_line_plane point p2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_properties_l139_13908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l139_13917

-- Define the power function f
noncomputable def f (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

-- Main theorem
theorem power_function_properties :
  (∀ x : ℝ, x > 0 → f x = x^(-(1/2 : ℝ))) ∧
  (f 9 = 1/3) ∧
  (f 25 = 1/5) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → f a = b → a = 1/b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l139_13917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l139_13929

/-- The length of a platform crossed by a train -/
noncomputable def platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  train_speed * (5/18) * crossing_time - train_length

/-- Theorem: Given a train with speed 72 kmph and length 230.0416 meters,
    crossing a platform in 26 seconds, the platform length is 289.9584 meters -/
theorem platform_length_calculation :
  platform_length 72 230.0416 26 = 289.9584 := by
  -- Unfold the definition of platform_length
  unfold platform_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l139_13929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_when_no_zeros_l139_13987

/-- The function f(x) = sin(πx - π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x - Real.pi / 3)

/-- The theorem states that if y = f(a*sin(x) + 1) has no zeros for all real x, 
    then a must be in the open interval (-1/3, 1/3) -/
theorem a_range_when_no_zeros (a : ℝ) :
  (∀ x : ℝ, f (a * Real.sin x + 1) ≠ 0) → a ∈ Set.Ioo (-1/3) (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_when_no_zeros_l139_13987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_F_maximum_l139_13998

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def F (m x : ℝ) : ℝ := m * Real.sqrt (1 - x^2) + f x

-- Define the piecewise function g
noncomputable def g (m : ℝ) : ℝ :=
  if m ≤ -Real.sqrt 2 / 2 then Real.sqrt 2
  else if m ≤ -1/2 then -1/(2*m) - m
  else m + 2

-- State the theorem
theorem f_properties_and_F_maximum :
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-1) 1) ∧ 
  (∀ y, y ∈ Set.Icc (Real.sqrt 2) 2 → ∃ x, f x = y) ∧
  (∀ m, ∀ x, F m x ≤ g m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_F_maximum_l139_13998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_zero_points_l139_13950

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem f_monotonicity_and_zero_points :
  (∀ x ∈ Set.Ioo 0 2, StrictMonoOn (f 1) (Set.Ioo 0 2)) ∧
  (∀ x ∈ Set.Ioi 2, StrictMonoOn (f 1) (Set.Ioi 2)) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 0 (1/3), f a x ≠ 0) ↔ a ∈ Set.Ici (2 - 3 * Real.log 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_zero_points_l139_13950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_most_appropriate_l139_13932

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Cluster
  | Systematic

/-- Represents a breeding room with a number of mice -/
structure BreedingRoom where
  mice : ℕ
deriving Inhabited

/-- Represents the research institute with breeding rooms -/
structure ResearchInstitute where
  rooms : List BreedingRoom
  sample_size : ℕ

/-- Determines if a sampling method is appropriate for a given research institute -/
def is_appropriate_sampling_method (institute : ResearchInstitute) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.Stratified => 
      institute.rooms.length > 1 ∧ 
      ∃ (i j : ℕ), i < institute.rooms.length ∧ j < institute.rooms.length ∧ i ≠ j ∧ 
        (institute.rooms.get! i).mice ≠ (institute.rooms.get! j).mice
  | _ => False

/-- The theorem stating that Stratified Sampling is the most appropriate method -/
theorem stratified_sampling_most_appropriate (institute : ResearchInstitute) : 
  institute.rooms = [⟨18⟩, ⟨24⟩, ⟨54⟩, ⟨48⟩] ∧ 
  institute.sample_size = 24 → 
  is_appropriate_sampling_method institute SamplingMethod.Stratified := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_most_appropriate_l139_13932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_round_trip_time_l139_13920

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
noncomputable def roundTripTime (boatSpeed streamSpeed distance : ℝ) : ℝ :=
  let upstreamSpeed := boatSpeed - streamSpeed
  let downstreamSpeed := boatSpeed + streamSpeed
  (distance / upstreamSpeed) + (distance / downstreamSpeed)

/-- Theorem stating that for a boat with speed 14 kmph in standing water, 
    a stream with speed 1.2 kmph, and a destination 4864 km away, 
    the total time for a round trip is 700 hours. -/
theorem boat_round_trip_time :
  roundTripTime 14 1.2 4864 = 700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_round_trip_time_l139_13920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l139_13985

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The left vertex of the ellipse -/
def A : ℝ × ℝ := (-2, 0)

/-- The upper vertex of the ellipse -/
def B : ℝ × ℝ := (0, 1)

/-- The first focus of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)

/-- The second focus of the ellipse -/
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

/-- A point on the line segment AB -/
def P : ℝ × ℝ → Prop
  | (x, y) => ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 1 * t

/-- The dot product of vectors PF₁ and PF₂ -/
noncomputable def dotProduct (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  (F₁.1 - x) * (F₂.1 - x) + (F₁.2 - y) * (F₂.2 - y)

/-- The theorem stating the minimum value of the dot product -/
theorem min_dot_product : 
  ∃ (m : ℝ), (∀ p, P p → dotProduct p ≥ m) ∧ (∃ p, P p ∧ dotProduct p = m) ∧ m = -11/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l139_13985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_ellipse_equation_l139_13907

/-- Ellipse C with equation (x²/a²) + (y²/b²) = 1 -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The focus of an ellipse -/
noncomputable def Focus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - b^2), 0)

theorem ellipse_foci_coordinates (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2*a = 4) :
  let C := Ellipse a b
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = b^2}
  (∃ (x : ℝ), (x, x + 2) ∈ circle) →
  Focus a b = (Real.sqrt 2, 0) ∧ Focus a b = (-Real.sqrt 2, 0) := by
  sorry

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := Ellipse a b
  (∀ (P M N : ℝ × ℝ), P ∈ C → M ∈ C → N ∈ C →
    (∃ (t : ℝ), M = (t * M.1, t * M.2) ∧ N = (-t * N.1, -t * N.2)) →
    let kPM := (P.2 - M.2) / (P.1 - M.1)
    let kPN := (P.2 - N.2) / (P.1 - N.1)
    kPM * kPN = -1/4) →
  C = Ellipse 2 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_ellipse_equation_l139_13907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_eq_1728_l139_13944

/-- An equilateral triangle with vertices on the hyperbola xy = 4 and centroid at a vertex of the hyperbola --/
structure EquilateralTriangleOnHyperbola where
  /-- The hyperbola equation xy = 4 --/
  hyperbola : ℝ → ℝ
  hyperbola_def : ∀ x y, hyperbola x = y ↔ x * y = 4
  /-- The vertices of the triangle --/
  vertices : Fin 3 → ℝ × ℝ
  /-- The vertices lie on the hyperbola --/
  vertices_on_hyperbola : ∀ i, hyperbola (vertices i).1 = (vertices i).2
  /-- The triangle is equilateral --/
  equilateral : ∀ i j, i ≠ j → ‖vertices i - vertices j‖ = ‖vertices 0 - vertices 1‖
  /-- The centroid of the triangle --/
  centroid : ℝ × ℝ
  /-- The centroid is a vertex of the hyperbola --/
  centroid_on_hyperbola : hyperbola centroid.1 = centroid.2
  /-- The centroid is the average of the vertices --/
  centroid_def : centroid = (vertices 0 + vertices 1 + vertices 2) / 3

/-- Calculate the area of a triangle given its vertices --/
noncomputable def triangleArea (v : Fin 3 → ℝ × ℝ) : ℝ :=
  let a := v 0
  let b := v 1
  let c := v 2
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

/-- The theorem stating that the square of the area of the triangle is 1728 --/
theorem area_squared_eq_1728 (t : EquilateralTriangleOnHyperbola) : 
  (triangleArea t.vertices) ^ 2 = 1728 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_eq_1728_l139_13944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l139_13921

-- Define the constants a and b as noncomputable
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := a^x + x - b

-- State the theorem
theorem root_interval (k : ℤ) : 
  (∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → k = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l139_13921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_function_range_l139_13978

noncomputable section

/-- Definition of a "mean value function" on an interval [a,b] -/
def is_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f x₁ = (f b - f a) / (b - a)) ∧
    (deriv^[2] f x₂ = (f b - f a) / (b - a))

/-- The function f(x) = (1/3)x³ - (1/2)x² + π -/
def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + Real.pi

theorem mean_value_function_range (m : ℝ) :
  is_mean_value_function f 0 m → 3/4 < m ∧ m < 3/2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_function_range_l139_13978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l139_13906

def set_A : Set ℝ := {x | x ≤ 2}
def set_B : Set ℝ := {x | x ∈ Set.Icc 1 3 ∧ x^2 - x - 6 ≤ 0}

theorem intersection_of_A_and_B :
  (set_A ∩ set_B) = {1, 2} := by
  sorry

#check intersection_of_A_and_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l139_13906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_ratio_l139_13965

theorem boat_speed_ratio (v_b v_c : ℝ) (h1 : v_b = 15) (h2 : v_c = 5) :
  (2 * v_b / (v_b / (v_b + v_c) + v_b / (v_b - v_c))) / v_b = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_ratio_l139_13965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l139_13935

theorem complex_fraction_equality : 
  (1 + Complex.I) * (2 - Complex.I) / Complex.I = 1 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l139_13935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_number_l139_13990

theorem smallest_special_number : ∃ (n : ℕ), n > 0 ∧ n % 2 ≠ 0 ∧ n % 3 ≠ 0 ∧
  (∀ a b : ℕ, (2^a : ℤ) - (3^b : ℤ) ≠ n ∧ (3^b : ℤ) - (2^a : ℤ) ≠ n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → m % 2 = 0 ∨ m % 3 = 0 ∨ (∃ a b : ℕ, (2^a : ℤ) - (3^b : ℤ) = m ∨ (3^b : ℤ) - (2^a : ℤ) = m)) ∧
  n = 35 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_special_number_l139_13990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_5x5_table_l139_13979

/-- Represents a 5x5 table of natural numbers -/
def Table := Fin 5 → Fin 5 → ℕ

/-- Sum of a row in the table -/
def rowSum (t : Table) (i : Fin 5) : ℕ := Finset.sum (Finset.univ : Finset (Fin 5)) (λ j => t i j)

/-- Sum of a column in the table -/
def colSum (t : Table) (j : Fin 5) : ℕ := Finset.sum (Finset.univ : Finset (Fin 5)) (λ i => t i j)

/-- Total sum of all numbers in the table -/
def totalSum (t : Table) : ℕ := Finset.sum (Finset.univ : Finset (Fin 5)) (λ i => Finset.sum (Finset.univ : Finset (Fin 5)) (λ j => t i j))

/-- Predicate to check if all row and column sums are distinct -/
def distinctSums (t : Table) : Prop :=
  ∀ i j, i ≠ j → (rowSum t i ≠ rowSum t j ∧ colSum t i ≠ colSum t j)

theorem min_sum_5x5_table :
  ∀ t : Table, distinctSums t → totalSum t ≥ 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_5x5_table_l139_13979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l139_13930

noncomputable def f (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x < 3 then -x^2 + 4*x
  else if 3 ≤ x ∧ x < 4 then (x^2 + 2) / x
  else 0  -- Define f outside [2,4) to be 0 for completeness

def g (a x : ℝ) : ℝ := a * x + 1

theorem range_of_a :
  ∀ (a : ℝ),
    (∀ (x : ℝ), f (x + 2) = 2 * f x) →
    (∀ (x₁ : ℝ), -2 ≤ x₁ ∧ x₁ < 0 →
      ∃ (x₂ : ℝ), -2 ≤ x₂ ∧ x₂ ≤ 1 ∧ g a x₂ = f x₁) →
    (a ≤ -1/4 ∨ a ≥ 1/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l139_13930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_appears_l139_13991

/-- Represents the result of rolling a die --/
def DieRoll := Fin 6

/-- Calculates the average of a list of die rolls --/
def average (rolls : List DieRoll) : ℚ :=
  (rolls.map (fun r => (r.val + 1 : ℚ))).sum / rolls.length

/-- Calculates the variance of a list of die rolls --/
def variance (rolls : List DieRoll) : ℚ :=
  let avg := average rolls
  (rolls.map (fun r => ((r.val + 1 : ℚ) - avg) ^ 2)).sum / rolls.length

theorem no_six_appears (rolls : List DieRoll) :
  rolls.length = 5 →
  average rolls = 2 →
  variance rolls = (31 : ℚ) / 10 →
  ∀ r ∈ rolls, r.val ≠ 5 := by
  sorry

#check no_six_appears

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_appears_l139_13991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_increasing_interval_in_domain_l139_13999

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 
    x₁ > (1/2 : ℝ) → 
    x₂ > (1/2 : ℝ) → 
    x₁ < x₂ → 
    f x₁ < f x₂ :=
by
  sorry

-- Define the domain of f(x)
def f_domain : Set ℝ := {x : ℝ | x > 0}

-- State that (1/2, +∞) is a subset of the domain
theorem increasing_interval_in_domain : 
  {x : ℝ | x > (1/2 : ℝ)} ⊆ f_domain :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_increasing_interval_in_domain_l139_13999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_per_metre_l139_13970

/-- The cost price per metre of cloth -/
noncomputable def cost_per_metre (total_cost : ℝ) (total_metres : ℝ) : ℝ :=
  total_cost / total_metres

/-- Theorem stating the cost per metre of cloth -/
theorem cloth_cost_per_metre :
  let total_cost : ℝ := 444
  let total_metres : ℝ := 9.25
  cost_per_metre total_cost total_metres = 48 := by
  -- Unfold the definition of cost_per_metre
  unfold cost_per_metre
  -- Perform the division
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_per_metre_l139_13970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l139_13910

/-- Represents a quadrilateral DFSR -/
structure Quadrilateral :=
  (D F R S : ℝ × ℝ)

/-- Calculates the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ :=
  Real.arccos ((distance p q)^2 + (distance q r)^2 - (distance p r)^2) / (2 * distance p q * distance q r)

/-- Theorem: In quadrilateral DFSR, if ∠RFS = ∠FDR, FD = 4, DR = 6, FR = 5, and FS = 7.5, then RS = 6.25 -/
theorem quadrilateral_theorem (DFSR : Quadrilateral) : 
  angle DFSR.R DFSR.F DFSR.S = angle DFSR.F DFSR.D DFSR.R →
  distance DFSR.F DFSR.D = 4 →
  distance DFSR.D DFSR.R = 6 →
  distance DFSR.F DFSR.R = 5 →
  distance DFSR.F DFSR.S = 7.5 →
  distance DFSR.R DFSR.S = 6.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l139_13910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_square_exists_l139_13956

/-- Represents a domino on the grid -/
structure Domino where
  x1 : Nat
  y1 : Nat
  x2 : Nat
  y2 : Nat

/-- Represents the 8x8 grid -/
def Grid := Finset (Nat × Nat)

/-- Checks if two dominoes form a 2x2 square -/
def form_square (d1 d2 : Domino) : Prop := sorry

/-- The main theorem statement -/
theorem domino_square_exists (grid : Grid) (dominoes : Finset Domino) : 
  (grid.card = 64) → 
  (dominoes.card = 32) → 
  (∀ d ∈ dominoes, d.x1 ≤ 8 ∧ d.y1 ≤ 8 ∧ d.x2 ≤ 8 ∧ d.y2 ≤ 8) →
  (∀ d ∈ dominoes, (d.x1 = d.x2 ∧ d.y2 = d.y1 + 1) ∨ (d.y1 = d.y2 ∧ d.x2 = d.x1 + 1)) →
  (∀ (x y : Nat), x ≤ 8 → y ≤ 8 → ∃! d, d ∈ dominoes ∧ ((d.x1 = x ∧ d.y1 = y) ∨ (d.x2 = x ∧ d.y2 = y))) →
  ∃ d1 d2, d1 ∈ dominoes ∧ d2 ∈ dominoes ∧ d1 ≠ d2 ∧ form_square d1 d2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_square_exists_l139_13956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l139_13940

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x^2 + 4)

-- State the theorem
theorem function_properties :
  ∀ a b : ℝ,
  (∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f a b x = (a * x + b) / (x^2 + 4)) →
  f a b 0 = 0 →
  f a b (1/2) = 2/17 →
  (a = 1 ∧ b = 0) ∧
  (∀ x y, x ∈ Set.Ioo (-2 : ℝ) 2 → y ∈ Set.Ioo (-2 : ℝ) 2 → x < y → f a b x < f a b y) ∧
  (∀ a' : ℝ, f 1 0 (a' + 1) - f 1 0 (2 * a' - 1) > 0 → a' ∈ Set.Ioo (-1/2 : ℝ) 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l139_13940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_15000_l139_13969

-- Define the fixed charter cost
def charter_cost : ℚ := 10000

-- Define the function for ticket price based on group size
noncomputable def ticket_price (x : ℚ) : ℚ :=
  if x ≤ 20 then 800
  else 800 - 10 * (x - 20)

-- Define the profit function
noncomputable def profit (x : ℚ) : ℚ :=
  x * ticket_price x - charter_cost

-- Theorem statement
theorem max_profit_is_15000 :
  ∃ x : ℚ, 0 ≤ x ∧ x ≤ 75 ∧ profit x = 15000 ∧
  ∀ y : ℚ, 0 ≤ y ∧ y ≤ 75 → profit y ≤ 15000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_15000_l139_13969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_length_k_l139_13964

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := -3 * x + 2
def h : ℝ → ℝ := Function.const ℝ 1

-- Define k as the minimum of f, g, and h
noncomputable def k (x : ℝ) : ℝ := min (min (f x) (g x)) (h x)

-- Define the length of the graph of k from -4 to 4
noncomputable def length_k : ℝ := sorry

-- Theorem statement
theorem square_length_k : length_k ^ 2 = (2 * Real.sqrt ((35/3)^2 + 24^2) + 2/3) ^ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_length_k_l139_13964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_per_square_inch_l139_13980

/-- Represents a TV with its dimensions and cost -/
structure TV where
  width : ℚ
  height : ℚ
  cost : ℚ

/-- Calculates the area of a TV -/
def TV.area (tv : TV) : ℚ := tv.width * tv.height

/-- Calculates the cost per square inch of a TV -/
def TV.costPerSquareInch (tv : TV) : ℚ := tv.cost / tv.area

/-- The first TV -/
def firstTV : TV := {
  width := 24,
  height := 16,
  cost := 672
}

/-- The new TV -/
def newTV : TV := {
  width := 48,
  height := 32,
  cost := 1152
}

/-- Theorem stating the difference in cost per square inch between the first TV and the new TV -/
theorem cost_difference_per_square_inch :
  firstTV.costPerSquareInch - newTV.costPerSquareInch = 1 := by
  -- Unfold definitions and simplify
  unfold TV.costPerSquareInch TV.area
  simp [firstTV, newTV]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_per_square_inch_l139_13980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_negative_three_l139_13951

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) + 1 / (x + 2)

-- State the theorem
theorem f_at_negative_three : f (-3) = -1 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.sqrt_zero]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_negative_three_l139_13951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_inequality_l139_13972

/-- A triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius

/-- Predicate to determine if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  t.R = 2 * t.r

/-- Theorem stating the relationship between circumradius and inradius of a triangle -/
theorem triangle_radius_inequality (t : Triangle) : 
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ is_equilateral t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_inequality_l139_13972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_l139_13976

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) - x^2 - 4 * x

-- State the theorem
theorem tangent_line_and_max_value (a b : ℝ) :
  (f a b 0 = 4 ∧ (deriv (f a b)) 0 = 4) →
  (a = 4 ∧ b = 4 ∧
   ∀ x : ℝ, f 4 4 x ≤ f 4 4 (-2) ∧
   f 4 4 (-2) = 4 * (1 - Real.exp (-2))) := by
  sorry

#check tangent_line_and_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_l139_13976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_equation_solution_l139_13915

-- Define complex numbers V and Z
def V : ℂ := 2 + 2*Complex.I
def Z : ℂ := 3 - 4*Complex.I

-- Define the expected result for I
noncomputable def expected_I : ℂ := -2/25 + 14/25*Complex.I

-- Theorem statement
theorem circuit_equation_solution :
  V = expected_I * Z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_equation_solution_l139_13915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_abc_together_l139_13989

-- Define the number of people
def n : ℕ := 8

-- Define the number of people in the group we're interested in
def k : ℕ := 3

-- Define the probability function
noncomputable def probability_together (n k : ℕ) : ℚ :=
  (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n

-- State the theorem
theorem probability_abc_together :
  probability_together n k = 1 / (40320 / 4320) :=
by
  -- Expand the definition of probability_together
  unfold probability_together
  -- Simplify the expression
  simp [n, k, Nat.factorial]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_abc_together_l139_13989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_angle_B_perimeter_range_l139_13958

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.b * Real.cos t.C = t.a - (1/2) * t.c

-- Theorem 1: Measure of angle B
theorem measure_of_angle_B (t : Triangle) (h : triangle_condition t) : 
  t.B = Real.pi / 3 := by sorry

-- Theorem 2: Range of perimeter when b = 1
theorem perimeter_range (t : Triangle) (h1 : triangle_condition t) (h2 : t.b = 1) :
  ∃ l : ℝ, 2 < l ∧ l ≤ 3 ∧ l = t.a + t.b + t.c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_angle_B_perimeter_range_l139_13958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_length_l139_13913

-- Define necessary structures and functions
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2

def IsMedianToHypotenuse (median : ℝ) (triangle : RightTriangle) : Prop :=
  median^2 = (triangle.a^2 + triangle.b^2) / 4

def HypotenuseLength (triangle : RightTriangle) : ℝ :=
  triangle.c

theorem right_triangle_hypotenuse_length 
  (triangle : RightTriangle) 
  (median1 : ℝ) (median2 : ℝ)
  (h1 : median1 = Real.sqrt 40)
  (h2 : median2 = 5)
  (h3 : IsMedianToHypotenuse median1 triangle)
  (h4 : IsMedianToHypotenuse median2 triangle) :
  HypotenuseLength triangle = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_length_l139_13913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_formula_l139_13963

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 2/3  -- Add a case for 0 to cover all natural numbers
  | 1 => 2/3
  | (n+2) => (n+1)/(n+2) * a (n+1)

-- Theorem statement
theorem a_n_formula (n : ℕ) : a n = 2 / (3 * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_formula_l139_13963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_years_is_one_l139_13942

/-- Represents the financial transaction described in the problem -/
structure FinancialTransaction where
  principal : ℚ
  borrowing_rate : ℚ
  lending_rate : ℚ
  gain_per_year : ℚ

/-- Calculates the number of years for the financial transaction -/
def calculate_years (t : FinancialTransaction) : ℚ :=
  (t.lending_rate - t.borrowing_rate) * t.principal / 100

/-- Theorem stating that under the given conditions, the number of years is 1 -/
theorem transaction_years_is_one (t : FinancialTransaction) 
  (h1 : t.principal = 5000)
  (h2 : t.borrowing_rate = 4)
  (h3 : t.lending_rate = 7)
  (h4 : t.gain_per_year = 150) : 
  calculate_years t = 1 := by
  sorry

#check transaction_years_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_years_is_one_l139_13942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_inequality_l139_13968

/-- A parabola passing through (1,1) and symmetric about x = 1 -/
structure Parabola where
  a : ℝ
  b : ℝ
  point_condition : a + b = 1
  symmetry_condition : -b / (2 * a) = 1

/-- The x-coordinate of the intersection of the parabola and y = -2/x -/
noncomputable def d (p : Parabola) : ℝ :=
  Real.sqrt (2 / (p.a * 2 + p.b))

/-- The main theorem to prove -/
theorem parabola_intersection_inequality (p : Parabola) : (3 / d p) > (1 / d p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_inequality_l139_13968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eta_expectation_and_variance_l139_13947

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  rv : ℝ → ℝ

/-- The expectation of a binomial random variable -/
def expectation (n : ℕ) (p : ℝ) (X : BinomialRV n p) : ℝ := n * p

/-- The variance of a binomial random variable -/
def variance (n : ℕ) (p : ℝ) (X : BinomialRV n p) : ℝ := n * p * (1 - p)

/-- A random variable defined as η = 8 - ξ, where ξ follows B(10, 0.6) -/
def eta (ξ : BinomialRV 10 0.6) : BinomialRV 10 0.6 :=
  { rv := fun x ↦ 8 - ξ.rv x }

theorem eta_expectation_and_variance (ξ : BinomialRV 10 0.6) :
  expectation 10 0.6 ξ = 6 ∧ 
  variance 10 0.6 ξ = 2.4 ∧ 
  expectation 10 0.6 (eta ξ) = 2 ∧ 
  variance 10 0.6 (eta ξ) = 2.4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eta_expectation_and_variance_l139_13947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l139_13993

-- Define the sets M and N
def M : Set ℝ := {x | 2 < x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5/2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 2 (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l139_13993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l139_13911

theorem equation_solution : ∃ x : ℝ, 32 = 2 * (16 : ℝ) ^ (x - 2) ∧ x = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l139_13911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lexi_laps_l139_13938

/-- Represents the length of a lap on the track in miles -/
def lap_length : ℚ := 1/4

/-- Represents the total distance Lexi wants to run in miles -/
def total_distance : ℚ := 13/4

/-- Calculates the number of complete laps needed to cover a given distance -/
def complete_laps (distance : ℚ) : ℕ := (distance / lap_length).floor.toNat

theorem lexi_laps : complete_laps total_distance = 13 := by
  -- Proof goes here
  sorry

#eval complete_laps total_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lexi_laps_l139_13938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_equivalence_l139_13986

/-- 
Given a real number x, the expression "5x - 3" represents 
"A number that is 5 times larger than x and then decreased by 3".
-/
theorem algebraic_expression_equivalence (x : ℝ) : 
  (5 * x - 3 : ℝ) = 5 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_equivalence_l139_13986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pics_in_one_album_l139_13937

def total_pictures : Nat := 45
def num_small_albums : Nat := 9
def pics_per_small_album : Nat := 2

def pics_in_small_albums : Nat := num_small_albums * pics_per_small_album

theorem pics_in_one_album : total_pictures - pics_in_small_albums = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pics_in_one_album_l139_13937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l139_13955

-- Define the sale price and savings percentage
def sale_price : ℚ := 20
def savings_percentage : ℚ := 12087912087912088 / 1000000000000000

-- Define the function to calculate the original price
noncomputable def original_price : ℚ := sale_price / (1 - savings_percentage / 100)

-- Define the function to calculate the amount saved
noncomputable def amount_saved : ℚ := original_price - sale_price

-- Theorem to prove
theorem savings_calculation :
  ∃ (ε : ℚ), ε > 0 ∧ |amount_saved - 275/100| < ε :=
by
  -- We'll use 1/1000 as our ε
  use 1/1000
  constructor
  · -- Prove ε > 0
    norm_num
  · -- Prove |amount_saved - 2.75| < ε
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l139_13955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_to_circle_l139_13995

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def Circle (a b r : ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}

theorem tangent_length_to_circle (P : ℝ × ℝ) :
  P ∈ Circle 2 1 1 →  -- P is on the circle (x-2)^2 + (y-1)^2 = 1
  (∀ Q : ℝ × ℝ, Q ∈ Circle 2 1 1 → distance 0 0 P.1 P.2 ≤ distance 0 0 Q.1 Q.2) →  -- OP is tangent to the circle
  distance 0 0 P.1 P.2 = 2  -- |OP| = 2
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_to_circle_l139_13995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l139_13939

-- Define the line
def line (a b : ℝ) (x y : ℝ) : Prop := a * x + 2 * b * y - 1 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Define the constraint on a and b based on the chord length
def constraint (a b : ℝ) : Prop := a^2 + 4 * b^2 = 1

-- State the theorem
theorem max_value_of_expression (a b : ℝ) :
  constraint a b →
  ∃ (x y : ℝ), line a b x y ∧ circle_eq x y →
  (∀ (a' b' : ℝ), constraint a' b' → 3 * a' + 2 * b' ≤ 3 * a + 2 * b) →
  3 * a + 2 * b = Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l139_13939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_numbers_l139_13962

theorem max_distinct_numbers (S : Finset ℕ) : 
  S.card = 11 → 
  (∀ a b, a ∈ S → b ∈ S → Nat.gcd a b ∈ S) → 
  (∀ x, x ∈ S → ∃ a b, a ∈ S ∧ b ∈ S ∧ x = Nat.gcd a b) → 
  Finset.card (Finset.image id S) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_numbers_l139_13962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l139_13997

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (10 - x))

-- State the theorem
theorem g_max_value :
  ∃ (N : ℝ), N = 30 * Real.sqrt 2 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 10 → g x ≤ N) ∧
  g 10 = N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l139_13997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_rhombus_l139_13961

-- Define the curve equation
def curve_equation (x y a b : ℝ) : Prop :=
  (|x + y| / (2 * a)) + (|x - y| / (2 * b)) = 1

-- Define a rhombus
def is_rhombus (vertices : List (ℝ × ℝ)) : Prop :=
  vertices.length = 4 ∧
  ∃ (d₁ d₂ : ℝ), d₁ ≠ d₂ ∧
    (List.get? vertices 0).isSome ∧ (List.get? vertices 1).isSome ∧
    (List.get? vertices 2).isSome ∧ (List.get? vertices 3).isSome ∧
    (((List.get? vertices 0).get!.1 - (List.get? vertices 2).get!.1)^2 +
     ((List.get? vertices 0).get!.2 - (List.get? vertices 2).get!.2)^2 = d₁^2) ∧
    (((List.get? vertices 1).get!.1 - (List.get? vertices 3).get!.1)^2 +
     ((List.get? vertices 1).get!.2 - (List.get? vertices 3).get!.2)^2 = d₂^2)

-- Theorem statement
theorem curve_is_rhombus (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) :
  ∃ (vertices : List (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ vertices → curve_equation x y a b) ∧
    is_rhombus vertices := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_rhombus_l139_13961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_in_closed_unit_interval_l139_13948

open Real Set

/-- A function f is monotonically increasing on ℝ -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x + a*sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * sin x

theorem monotone_increasing_f_implies_a_in_closed_unit_interval :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) → a ∈ Icc (-1 : ℝ) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_in_closed_unit_interval_l139_13948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_periodicity_example_function_satisfies_equation_example_function_periodic_l139_13900

/-- A function f is periodic with period 2a if it satisfies the given functional equation. -/
theorem function_periodicity (f : ℝ → ℝ) (a : ℝ) (h : a > 0) :
  (∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) →
  (∀ x, f (x + 2*a) = f x) :=
by sorry

/-- An example of a non-constant periodic function satisfying the given equation for a = 1. -/
noncomputable def example_function : ℝ → ℝ :=
  fun x => if x % 2 < 1 then 1 else 1/2

/-- The example function satisfies the given functional equation. -/
theorem example_function_satisfies_equation :
  ∀ x, example_function (x + 1) = 1/2 + Real.sqrt (example_function x - (example_function x)^2) :=
by sorry

/-- The example function is periodic with period 2. -/
theorem example_function_periodic :
  ∀ x, example_function (x + 2) = example_function x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_periodicity_example_function_satisfies_equation_example_function_periodic_l139_13900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l139_13953

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * (-1)^(Int.floor k) * Real.log x

theorem f_properties (a : ℝ) (k : ℕ) (h_a : a > 0) :
  let f := f a k
  (∀ x, x > 0 → Odd k → (deriv f) x > 0) ∧
  (∀ x, x > 0 → Even k → (x < Real.sqrt a → (deriv f) x < 0) ∧ (x > Real.sqrt a → (deriv f) x > 0)) ∧
  (k = 2018 → (∃! x, x > 0 ∧ f x = 2 * a * x) → a = 1/2) ∧
  (k = 2019 → ∀ x, x > 0 → x * Real.log x > x / Real.exp x - 2 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l139_13953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_competition_selection_l139_13926

theorem science_competition_selection (female_students male_students : ℕ) 
  (h1 : female_students = 2)
  (h2 : male_students = 4) :
  (Finset.sum (Finset.range (female_students + 1)) 
    (λ k => Nat.choose female_students k * Nat.choose male_students (3 - k))) - 
  Nat.choose male_students 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_competition_selection_l139_13926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_after_transformation_l139_13971

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x - Real.pi / 3) + 1

/-- The transformed function after stretching and shifting -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 3) + 1

/-- Theorem stating that the axis of symmetry of the transformed function is x = -π/6 -/
theorem axis_of_symmetry_after_transformation :
  ∀ x : ℝ, g (x - Real.pi / 6) = g (-x - Real.pi / 6) := by
  intro x
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_after_transformation_l139_13971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_flame_duration_l139_13931

/-- The duration of each flame in Jason's game --/
noncomputable def flame_duration (fire_interval : ℝ) (flame_time_per_minute : ℝ) : ℝ :=
  (flame_time_per_minute * fire_interval) / 60

theorem jason_flame_duration :
  flame_duration 15 20 = 5 := by
  -- Unfold the definition of flame_duration
  unfold flame_duration
  -- Simplify the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_flame_duration_l139_13931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_six_l139_13949

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first four terms
  sum_first_four : ℚ
  -- The fifth term
  fifth_term : ℚ
  -- The property that the sum of the first four terms is 10
  sum_property : sum_first_four = 10
  -- The property that the fifth term is 5
  fifth_term_property : fifth_term = 5

/-- The sixth term of the arithmetic sequence -/
def sixth_term (seq : ArithmeticSequence) : ℚ :=
  seq.fifth_term + (seq.fifth_term - (seq.sum_first_four / 4 - seq.fifth_term / 4))

/-- Theorem stating that the sixth term is 6 -/
theorem sixth_term_is_six (seq : ArithmeticSequence) : sixth_term seq = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_six_l139_13949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_35_hours_l139_13981

/-- Represents the journey described in the problem -/
structure Journey where
  totalDistance : ℕ
  carSpeed : ℕ
  walkingSpeed : ℕ
  initialCarDistance : ℕ
  dickWalkingDistance : ℕ

/-- Calculates the total time for the journey -/
def journeyTime (j : Journey) : ℚ :=
  (j.initialCarDistance : ℚ) / j.carSpeed +
  ((j.totalDistance - j.initialCarDistance : ℚ) / j.walkingSpeed)

/-- The theorem stating that the journey time is 35 hours -/
theorem journey_time_is_35_hours (j : Journey)
  (h1 : j.totalDistance = 150)
  (h2 : j.carSpeed = 30)
  (h3 : j.walkingSpeed = 3)
  (h4 : j.initialCarDistance = 50)
  (h5 : j.dickWalkingDistance = 15)
  : journeyTime j = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_35_hours_l139_13981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l139_13977

theorem angle_in_second_quadrant (α : Real) :
  (Real.tan α < 0 ∧ Real.cos α < 0) → 
  (π / 2 < α ∧ α < π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l139_13977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l139_13936

/-- The equation of a hyperbola sharing foci with an ellipse and having a given asymptote -/
theorem hyperbola_equation (e a h : ℝ → ℝ → Prop) :
  (∀ x y, e x y ↔ x^2/5 + y^2 = 1) →
  (∀ x y, a x y ↔ Real.sqrt 3*x - y = 0) →
  (∀ x y, h x y ↔ x^2 - y^2/3 = 1) →
  (∀ x y, h x y ↔ 
    (∃ c, c^2 = 4 ∧ 
      (∀ x' y', e x' y' → (x' - c)^2 + y'^2 = (x' + c)^2 + y'^2) ∧
      (∀ x' y', h x' y' → (x' - c)^2 - y'^2 = (x' + c)^2 - y'^2)) ∧
    (∃ k, k = Real.sqrt 3 ∧ 
      (∀ x' y', a x' y' → y' = k*x'))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l139_13936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_length_l139_13975

/-- A regular hexagon with a point inside it -/
structure RegularHexagonWithPoint where
  /-- The side length of the regular hexagon -/
  side_length : ℝ
  /-- The point inside the hexagon -/
  point : ℝ × ℝ
  /-- The vertices of the hexagon -/
  vertices : Fin 6 → ℝ × ℝ
  /-- The hexagon is regular -/
  regular : ∀ i : Fin 6, dist (vertices i) (vertices ((i + 1) % 6)) = side_length
  /-- The point is inside the hexagon -/
  inside : ∀ i : Fin 6, dist point (vertices i) ≤ side_length

/-- Theorem: If there's a point inside a regular hexagon with distances 1, 1, and 2 to three consecutive vertices, the side length is √7 -/
theorem hexagon_side_length (h : RegularHexagonWithPoint) 
  (dist_1 : dist h.point (h.vertices 0) = 1)
  (dist_2 : dist h.point (h.vertices 1) = 1)
  (dist_3 : dist h.point (h.vertices 2) = 2) :
  h.side_length = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_length_l139_13975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_trig_l139_13916

/-- Given two circles C₁ and C₂ symmetric about a line, prove sin(θ)cos(θ) = -2/5 --/
theorem circle_symmetry_trig (a θ : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a*x = 0 → (2*x - y - 1 = 0 ↔ 2*(-x) - (-y) - 1 = 0)) →
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + y*Real.tan θ = 0 → (2*x - y - 1 = 0 ↔ 2*(-x) - (-y) - 1 = 0)) →
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_trig_l139_13916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_theorem_l139_13946

/-- The dimensions of the larger box -/
def larger_box_dimensions (x : ℚ) : Fin 3 → ℚ := 
  ![12, x, 16]

/-- The dimensions of the smaller box -/
def smaller_box_dimensions : Fin 3 → ℚ := 
  ![3, 7, 2]

/-- The volume of a box given its dimensions -/
def box_volume (dimensions : Fin 3 → ℚ) : ℚ :=
  (dimensions 0) * (dimensions 1) * (dimensions 2)

/-- The maximum number of smaller boxes that fit in the larger box -/
def max_boxes_fit (x : ℚ) : ℚ :=
  box_volume (larger_box_dimensions x) / box_volume smaller_box_dimensions

/-- The theorem statement -/
theorem box_dimension_theorem (x : ℚ) :
  max_boxes_fit x = 64 → x = 14 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_theorem_l139_13946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_l139_13992

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 5 * x - 7 else -3 * x + 12

-- State the theorem
theorem f_zeros : {x : ℝ | f x = 0} = {7/5, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_l139_13992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_neg_three_l139_13901

/-- A function that is defined as f(x) = a*sin(x + π/4) + 3*sin(x - π/4) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (x + Real.pi/4) + 3 * Real.sin (x - Real.pi/4)

/-- A predicate that checks if a function is even --/
def isEven (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

/-- Theorem stating that if f is even, then a = -3 --/
theorem f_even_implies_a_eq_neg_three (a : ℝ) :
  isEven (f a) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_neg_three_l139_13901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l139_13982

noncomputable def z : ℂ := -Complex.I / (1 + 2 * Complex.I)

theorem z_in_third_quadrant : 
  z.re < 0 ∧ z.im < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l139_13982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l139_13994

/-- The distance between two planes in 3D space -/
noncomputable def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  let n₁ := (a₁, b₁, c₁)
  let n₂ := (a₂, b₂, c₂)
  |d₂ - d₁| / Real.sqrt (((a₂ - a₁)^2 + (b₂ - b₁)^2 + (c₂ - c₁)^2) : ℝ)

/-- The distance between the planes x + 2y - 2z + 3 = 0 and 2x + 5y - 4z + 7 = 0 is √5/15 -/
theorem distance_specific_planes :
  distance_between_planes 1 2 (-2) (-3) 2 5 (-4) (-7) = Real.sqrt 5 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l139_13994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_transformation_l139_13928

/-- Scaling transformation on the plane -/
noncomputable def scaling (x y : ℝ) : ℝ × ℝ :=
  (1/2 * x, 3 * y)

/-- Original sine curve -/
noncomputable def original_curve (x : ℝ) : ℝ :=
  Real.sin x

/-- Transformed curve -/
noncomputable def transformed_curve (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x)

theorem sine_curve_transformation :
  ∀ x y : ℝ, y = original_curve x → 
  let (x', y') := scaling x y
  y' = transformed_curve x' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_transformation_l139_13928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_value_max_tan_C_minus_B_l139_13959

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  m : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = t.m * t.b * Real.cos t.C

-- Part 1
theorem cosA_value (t : Triangle) 
  (h : triangle_conditions t) 
  (h_m : t.m = 2) 
  (h_cosC : Real.cos t.C = Real.sqrt 10 / 10) : 
  Real.cos t.A = 4/5 := by sorry

-- Part 2
theorem max_tan_C_minus_B (t : Triangle) 
  (h : triangle_conditions t) 
  (h_m : t.m = 4) : 
  (∀ t' : Triangle, triangle_conditions t' → t'.m = 4 → 
    Real.tan (t'.C - t'.B) ≤ Real.sqrt 3 / 3) ∧
  (∃ t' : Triangle, triangle_conditions t' ∧ t'.m = 4 ∧ 
    Real.tan (t'.C - t'.B) = Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_value_max_tan_C_minus_B_l139_13959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l139_13973

noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Theorem statement
theorem inverse_proportion_properties :
  -- 1. The graph is distributed in the first and third quadrants
  (∀ x, x ≠ 0 → (x > 0 ∧ f x > 0) ∨ (x < 0 ∧ f x < 0)) ∧
  -- 2. The graph is both axisymmetric and centrally symmetric
  (∀ x, x ≠ 0 → f x = f (-x) ∧ f x = -f (-x)) ∧
  -- 3. When x > 0, the value of f(x) decreases as x increases
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₂ < f x₁) ∧
  -- 4. When x < 0, the value of f(x) decreases as x increases
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f x₂ < f x₁) := by
  sorry

#check inverse_proportion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l139_13973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l139_13927

-- Define the line
def line (t : ℝ) : ℝ × ℝ := (t - 1, 2 - t)

-- Define the curve
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

-- Define the number of intersection points
def num_intersection_points : ℕ := 2

-- Theorem statement
theorem intersection_points_count :
  ∃ (t₁ t₂ θ₁ θ₂ : ℝ), 
    t₁ ≠ t₂ ∧ 
    line t₁ = curve θ₁ ∧ 
    line t₂ = curve θ₂ ∧
    (∀ (t θ : ℝ), line t = curve θ → t = t₁ ∨ t = t₂) ∧
    num_intersection_points = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l139_13927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_swimming_speed_l139_13919

-- Define the swimming speed function
noncomputable def V (Q : ℝ) : ℝ := (1/2) * Real.log (Q / 100) / Real.log 3

-- Theorem statement
theorem salmon_swimming_speed :
  -- Condition: V is directly proportional to log₃(Q/100)
  (∃ k : ℝ, ∀ Q : ℝ, V Q = k * Real.log (Q / 100) / Real.log 3) ∧
  -- Condition: When Q = 900, V = 1
  V 900 = 1 ∧
  -- Answer to question 1: The functional notation of V in terms of Q
  (∀ Q : ℝ, V Q = (1/2) * Real.log (Q / 100) / Real.log 3) ∧
  -- Answer to question 2: When V = 1.5, Q = 2700
  (V 2700 = 1.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_swimming_speed_l139_13919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l139_13943

/-- Circle M -/
def circle_M (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*y = 0 ∧ a > 0

/-- Line -/
def line (x y : ℝ) : Prop :=
  x + y = 0

/-- Circle N -/
def circle_N (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 1

/-- Length of intersection segment -/
noncomputable def intersection_length (a : ℝ) : ℝ := 2 * Real.sqrt 2

theorem circles_intersect (a : ℝ) :
  (∃ x y : ℝ, circle_M a x y ∧ line x y) →
  (∀ x y : ℝ, circle_M a x y ∧ line x y → 
    ∃ x' y' : ℝ, circle_M a x' y' ∧ line x' y' ∧ 
    ((x - x')^2 + (y - y')^2)^(1/2 : ℝ) = intersection_length a) →
  (∃ x y : ℝ, circle_M a x y ∧ circle_N x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l139_13943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_implies_m_range_l139_13918

-- Define the operation G
noncomputable def G (x y : ℝ) : ℝ :=
  if x ≥ y then x - y else y - x

-- State the theorem
theorem exactly_three_solutions_implies_m_range :
  ∀ m : ℝ,
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    ∀ x ∈ s, (x : ℝ) > 0 ∧ G x 1 > 4 ∧ G (-1) x ≤ m) →
  9 ≤ m ∧ m < 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_implies_m_range_l139_13918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l139_13912

theorem triangle_angle_measure (a b c : ℝ) (h1 : c * (a - b)^2 + 6 = c^2) 
  (h2 : (1/2) * a * b * Real.sin (Real.arccos (1/2)) = (3 * Real.sqrt 3) / 2) : 
  Real.arccos (1/2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l139_13912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l139_13903

/-- The function f(x, a) represents the left side of the inequality. -/
noncomputable def f (x a : ℝ) : ℝ := 
  Real.exp (2 * x) - Real.exp (-2 * x) - 4 * x - a * Real.exp x + a * Real.exp (-x) + 2 * a * x

/-- The theorem states that 8 is the maximum value of a for which f(x, a) ≥ 0 holds for all positive real x. -/
theorem max_a_value : 
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f x a ≥ 0) → a ≤ 8) ∧ 
  (∀ x : ℝ, x > 0 → f x 8 ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l139_13903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l139_13945

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse with a left focus at (-2, 0) and a vertex at (0, 1) -/
structure Ellipse where
  left_focus : Point := ⟨-2, 0⟩
  vertex : Point := ⟨0, 1⟩

/-- The line passing through the left focus and vertex of the ellipse -/
def line_equation (p : Point) : Prop :=
  p.x - 2 * p.y + 2 = 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  2 * Real.sqrt 5 / 5

theorem ellipse_eccentricity (e : Ellipse) :
  line_equation e.left_focus ∧ line_equation e.vertex →
  eccentricity e = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l139_13945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l139_13902

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) + 3

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 2 ∧ f a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l139_13902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_extreme_points_count_a_range_for_non_negative_f_l139_13988

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3*x + 2)

-- Theorem for part (1)
theorem tangent_line_at_x_1 (a : ℝ) :
  a = 0 → ∃ m b : ℝ, ∀ x : ℝ, m * x + b = x - 1 ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
    |f 0 (1 + h) - (f 0 1 + m * h)| ≤ ε * |h|) :=
by sorry

-- Theorem for part (2)
theorem extreme_points_count (a : ℝ) :
  (a < 0 → (∃! x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f a y ≤ f a x))) ∧
  (0 ≤ a ∧ a ≤ 8/9 → (∀ x y : ℝ, 0 < x ∧ x < y → f a x < f a y)) ∧
  (a > 8/9 → (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (∀ y : ℝ, y > 0 → f a y ≤ f a x₁ ∨ f a y ≤ f a x₂) ∧
    (∀ z : ℝ, x₁ < z ∧ z < x₂ → f a z < f a x₁ ∧ f a z < f a x₂))) :=
by sorry

-- Theorem for part (3)
theorem a_range_for_non_negative_f :
  {a : ℝ | ∀ x : ℝ, x ≥ 1 → f a x ≥ 0} = {a : ℝ | 0 ≤ a ∧ a ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_extreme_points_count_a_range_for_non_negative_f_l139_13988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l139_13984

/-- Given a hyperbola and a line intersecting it, prove the eccentricity --/
theorem hyperbola_eccentricity (a b : ℝ) (A B : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (A.1^2 / a^2 - A.2^2 / b^2 = 1) →
  (B.1^2 / a^2 - B.2^2 / b^2 = 1) →
  (B.2 - A.2) / (B.1 - A.1) = 3 →
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (6, 2) →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l139_13984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_sum_l139_13952

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the hexagon -/
def hexagonVertices : List Point := [
  ⟨0, 0⟩, ⟨1, 2⟩, ⟨3, 3⟩, ⟨5, 3⟩, ⟨6, 1⟩, ⟨4, -1⟩
]

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the perimeter of the hexagon -/
noncomputable def hexagonPerimeter : ℝ :=
  let pairs := List.zip hexagonVertices (hexagonVertices.rotateLeft 1)
  (pairs.map (fun (p1, p2) => distance p1 p2)).sum

/-- Represents the perimeter in the form a + b√2 + c√5 -/
structure PerimeterForm where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The theorem to be proved -/
theorem hexagon_perimeter_sum :
  ∃ (form : PerimeterForm),
    (form.a : ℝ) + form.b * Real.sqrt 2 + form.c * Real.sqrt 5 = hexagonPerimeter ∧
    form.a + form.b + form.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_sum_l139_13952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_less_than_one_l139_13954

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) * Real.exp x - 1

-- Define the function g
noncomputable def g (x : ℝ) (h : x > -1 ∧ x ≠ 0) : ℝ := f x / x

-- Theorem for the maximum value of f
theorem f_max_value :
  ∃ (M : ℝ), M = 0 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

-- Theorem for g(x) < 1
theorem g_less_than_one (x : ℝ) (h : x > -1 ∧ x ≠ 0) :
  g x h < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_less_than_one_l139_13954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l139_13966

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ Real.sqrt (x - x^2)

-- State the theorem
theorem f_monotonic_increasing_interval :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 1 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∧
  (∀ c d, (∀ x y, c ≤ x ∧ x < y ∧ y ≤ d → f x ≤ f y) → 
    a ≤ c ∧ d ≤ b) :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l139_13966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_shortest_l139_13974

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a path as a continuous function from [0, 1] to Point2D
def ContinuousPath := Set.Icc 0 1 → Point2D

-- Define the length of a path
noncomputable def pathLength (p : ContinuousPath) : ℝ := sorry

-- Define a straight line segment between two points
def straightLine (a b : Point2D) : ContinuousPath := sorry

-- Theorem: The straight line segment is the shortest path between two points
theorem straight_line_shortest (a b : Point2D) (p : ContinuousPath) 
  (h1 : p 0 = a) (h2 : p 1 = b) : 
  pathLength (straightLine a b) ≤ pathLength p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_shortest_l139_13974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_is_50_l139_13960

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Represents the financial transaction -/
structure Transaction where
  borrowedAmount : ℝ
  borrowingPeriod : ℝ
  borrowingRate : ℝ
  lendingAmount : ℝ
  lendingPeriod : ℝ
  lendingRate : ℝ

/-- Calculates the gain per year for a given transaction -/
noncomputable def gainPerYear (t : Transaction) : ℝ :=
  let interestEarned := simpleInterest t.lendingAmount t.lendingRate t.lendingPeriod
  let interestPaid := simpleInterest t.borrowedAmount t.borrowingRate t.borrowingPeriod
  (interestEarned - interestPaid) / t.lendingPeriod

/-- The main theorem stating that the gain per year is 50 for the given transaction -/
theorem transaction_gain_is_50 :
  let t : Transaction := {
    borrowedAmount := 5000,
    borrowingPeriod := 2,
    borrowingRate := 4,
    lendingAmount := 5000,
    lendingPeriod := 2,
    lendingRate := 5
  }
  gainPerYear t = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_is_50_l139_13960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l139_13941

/-- A function satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- y = f(x+1) is an even function -/
axiom f_even : ∀ x : ℝ, f (x + 1) = f (-x + 1)

/-- For x ≥ 1, f(x) = (1/2)^x - 1 -/
axiom f_def : ∀ x : ℝ, x ≥ 1 → f x = (1/2)^x - 1

/-- The main theorem to prove -/
theorem f_order : f (2/3) > f (3/2) ∧ f (3/2) > f (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l139_13941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_given_radii_l139_13909

/-- The radius of an inscribed circle given three mutually externally tangent circles -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c - Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem stating the radius of the inscribed circle for given radii -/
theorem inscribed_circle_radius_for_given_radii :
  let r := inscribed_circle_radius 5 10 25
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |r - 6.2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_for_given_radii_l139_13909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_one_iff_a_sqrt_two_l139_13933

theorem modulus_one_iff_a_sqrt_two (a : ℝ) : 
  Complex.abs ((1 - 2*a*Complex.I) / (3*Complex.I)) = 1 ↔ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_one_iff_a_sqrt_two_l139_13933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_lt_x_l139_13934

theorem negation_of_forall_sin_lt_x :
  (¬ (∀ x : ℝ, x > 0 → Real.sin x < x)) ↔ (∃ x : ℝ, x > 0 ∧ Real.sin x ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_lt_x_l139_13934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_and_area_l139_13923

/-- Given a circle with radius 5 and a chord PQ where the distance from the center to the chord is 4,
    prove that the length of PQ is 6 and the area of the circle is 25π. -/
theorem circle_chord_and_area (r : ℝ) (d : ℝ) (h1 : r = 5) (h2 : d = 4) :
  2 * Real.sqrt (r ^ 2 - d ^ 2) = 6 ∧ π * r ^ 2 = 25 * π := by
  -- Substitute the given values
  have r_eq : r = 5 := h1
  have d_eq : d = 4 := h2

  -- Calculate the chord length
  have chord_length : 2 * Real.sqrt (r ^ 2 - d ^ 2) = 6 := by
    rw [r_eq, d_eq]
    -- Here we would prove that 2 * √(5² - 4²) = 6
    sorry

  -- Calculate the area
  have area : π * r ^ 2 = 25 * π := by
    rw [r_eq]
    -- Here we would prove that π * 5² = 25π
    sorry

  -- Combine the results
  exact ⟨chord_length, area⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_and_area_l139_13923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_f_at_5_l139_13925

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 25 / (4 + 2 * x)

-- State the theorem
theorem inverse_of_inverse_f_at_5 : (Function.invFun f 5)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_f_at_5_l139_13925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_123_l139_13914

def S : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | 2 => 4
  | n + 3 => S (n + 1) + S (n + 2)

theorem tenth_term_is_123 : S 9 = 123 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_123_l139_13914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_repayment_period_is_five_years_l139_13983

/-- Given a loan amount, new repayment period, and additional monthly payment,
    calculate the original repayment period in years. -/
noncomputable def calculate_original_repayment_period (loan_amount : ℝ) (new_repayment_period : ℝ) (additional_payment : ℝ) : ℝ :=
  let new_monthly_payment := loan_amount / (new_repayment_period * 12)
  let original_monthly_payment := new_monthly_payment - additional_payment
  loan_amount / (original_monthly_payment * 12)

/-- Theorem stating that for a $6,000 loan with a 2-year new repayment plan
    and $150 additional monthly payment, the original repayment period was 5 years. -/
theorem original_repayment_period_is_five_years :
  calculate_original_repayment_period 6000 2 150 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_repayment_period_is_five_years_l139_13983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l139_13922

-- Define the hyperbola parameters
noncomputable def center : ℝ × ℝ := (3, -1)
noncomputable def focus : ℝ × ℝ := (3 + 5 * Real.sqrt 2, -1)
noncomputable def vertex : ℝ × ℝ := (1, -1)

-- Define h and k from the center coordinates
noncomputable def h : ℝ := center.1
noncomputable def k : ℝ := center.2

-- Define a as the distance between center and vertex
noncomputable def a : ℝ := |center.1 - vertex.1|

-- Define c as the distance between center and focus
noncomputable def c : ℝ := |focus.1 - center.1|

-- Define b using the relation b^2 = c^2 - a^2
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 4 + Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l139_13922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_c_measure_l139_13957

-- Define a custom type for degree-minute angles
structure DegreeMinute where
  degrees : ℕ
  minutes : ℕ
  h_valid : minutes < 60

-- Define addition for DegreeMinute
def DegreeMinute.add (a b : DegreeMinute) : DegreeMinute :=
  let total_minutes := a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes
  ⟨total_minutes / 60, total_minutes % 60, by sorry⟩

-- Define subtraction of DegreeMinute from 180°
def DegreeMinute.sub_from_180 (a : DegreeMinute) : DegreeMinute :=
  ⟨179 - a.degrees - (if a.minutes > 0 then 1 else 0), if a.minutes > 0 then 60 - a.minutes else 0, by sorry⟩

-- Define a structure for triangles
structure Triangle where
  angle_A : DegreeMinute
  angle_B : DegreeMinute
  angle_C : DegreeMinute

-- Define congruence for triangles
def Triangle.congruent (t1 t2 : Triangle) : Prop :=
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

-- Use notation ≅ for triangle congruence
notation:50 t1 " ≅ " t2 => Triangle.congruent t1 t2

theorem angle_c_measure 
  (ABC A'B'C' : Triangle) 
  (h_congruent : ABC ≅ A'B'C')
  (h_angle_A : ABC.angle_A = ⟨35, 25, by sorry⟩)
  (h_angle_B' : A'B'C'.angle_B = ⟨49, 45, by sorry⟩) :
  ABC.angle_C = ⟨94, 50, by sorry⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_c_measure_l139_13957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_line_equation_at_max_distance_reflected_ray_equation_l139_13967

-- Define the line l: x + ay + a - 1 = 0
def line_l (a : ℝ) (x y : ℝ) : Prop := x + a * y + a - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, 1)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
  let (x₀, y₀) := p
  abs (x₀ + a * y₀ + a - 1) / Real.sqrt (1 + a^2)

-- Theorem 1: Maximum distance from P to line l
theorem max_distance_to_line :
  ∃ (a : ℝ), distance_to_line point_P a = Real.sqrt 13 ∧
  ∀ (b : ℝ), distance_to_line point_P b ≤ Real.sqrt 13 :=
sorry

-- Theorem 2: Equation of line l when distance is maximum
theorem line_equation_at_max_distance :
  ∃ (a : ℝ), distance_to_line point_P a = Real.sqrt 13 ∧
  (∀ x y, line_l a x y ↔ 3 * x - 2 * y - 5 = 0) :=
sorry

-- Define the reflected ray
def reflected_ray (x y : ℝ) : Prop := 12 * x + y = 0

-- Theorem 3: Equation of reflected ray when a = 2
theorem reflected_ray_equation :
  ∃ (x₁ y₁ : ℝ),
    line_l 2 x₁ y₁ ∧  -- Point on line l
    (x₁ - (-2))^2 + (y₁ - 1)^2 = (x₁ - 0)^2 + (y₁ - 0)^2 ∧  -- Equidistant from P and origin
    reflected_ray x₁ y₁  -- Point on reflected ray
:=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_line_equation_at_max_distance_reflected_ray_equation_l139_13967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l139_13924

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem hyperbola_focus_distance 
  (x y xF1 yF1 xF2 yF2 : ℝ) 
  (h1 : hyperbola x y) 
  (h2 : distance x y xF1 yF1 = 9) :
  distance x y xF2 yF2 = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l139_13924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l139_13905

def a : Fin 3 → ℝ := ![-2, -3, 1]
def b : Fin 3 → ℝ := ![2, 0, 4]
def c : Fin 3 → ℝ := ![-4, -6, 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

def is_parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * (w i)

def is_perpendicular (v w : Fin 3 → ℝ) : Prop :=
  dot_product v w = 0

theorem vector_relationships : is_parallel a c ∧ is_perpendicular a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l139_13905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l139_13996

noncomputable def quadrilateral_problem (A B C D : ℂ) : Prop :=
  let O : ℂ := 0
  ∃ (B' C' H₁ H₂ M O' : ℂ),
    -- ABCD is inscribed in a unit circle
    Complex.abs A = 1 ∧ Complex.abs B = 1 ∧ Complex.abs C = 1 ∧ Complex.abs D = 1 ∧
    -- Angle conditions
    (B - O) / (A - O) = Complex.exp (Complex.I * (135 * Real.pi / 180)) ∧
    (D - O) / (C - O) = Complex.exp (Complex.I * (135 * Real.pi / 180)) ∧
    -- BC = 1
    Complex.abs (C - B) = 1 ∧
    -- B' is reflection of A across BO
    B' - O = (B - O) * (B - O) / (A - O) ∧
    -- C' is reflection of A across CO
    C' - O = (C - O) * (C - O) / (A - O) ∧
    -- H₁ is orthocenter of AB'C'
    H₁ = A + B' + C' ∧
    -- H₂ is orthocenter of BCD
    H₂ = B + C + D ∧
    -- M is midpoint of OH₁
    M = (O + H₁) / 2 ∧
    -- O' is reflection of O about midpoint of MH₂
    O' - ((M + H₂) / 2) = -O + ((M + H₂) / 2) ∧
    -- Distance OO' equals the specified value
    Complex.abs (O' - O) = Real.sqrt ((8 - Real.sqrt 6 - 3 * Real.sqrt 2) / 4)

theorem quadrilateral_theorem :
  ∀ A B C D : ℂ, quadrilateral_problem A B C D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l139_13996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l139_13904

/-- Given a hyperbola and a parabola that share a focus, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), x^2 = 12 * y) →  -- Parabola equation
  (∃ (x₀ y₀ : ℝ), x₀^2 = 12 * y₀ ∧ y₀^2 / 5 - x₀^2 / m = 1) →  -- Shared focus condition
  (∃ (k : ℝ), ∀ (x y : ℝ), (y = k * x ∨ y = -k * x) ↔ (y^2 / 5 - x^2 / m = 1 ∧ (x ≠ 0 ∨ y ≠ 0))) ∧
  (∃ k : ℝ, k = Real.sqrt 5 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l139_13904
