import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l191_19116

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi) ((3 * Real.pi) / 2)) (h2 : Real.tan α = 2) : 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l191_19116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_composite_l191_19105

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_consecutive_composite : 
  ∀ (a : ℕ), 
    (a > 9 ∧ a < 33) → 
    (∀ i : Fin 7, ¬(is_prime (a + i))) →
    (∀ i : Fin 7, (a + i < 40)) →
    (a + 6 = 30) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_composite_l191_19105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_grid_paths_l191_19158

/-- Represents the number of paths in a grid with given dimensions and forbidden segments -/
def grid_paths (columns rows : ℕ) (forbidden_columns : List ℕ) : ℕ :=
  let total_paths := Nat.choose (columns + rows) rows
  let forbidden_paths := (forbidden_columns.map fun col =>
    Nat.choose (col + 1) 2 * Nat.choose (columns - col + 1) 2).sum
  total_paths - forbidden_paths

/-- The specific grid configuration from the problem -/
def problem_grid : ℕ := grid_paths 11 4 [5, 6, 7]

theorem problem_grid_paths : problem_grid = 237 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_grid_paths_l191_19158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_l191_19175

/-- Given triangle XYZ with coordinates X(6,0), Y(8,4), Z(10,0), 
    and that the area of triangle XYZ is 0.1111111111111111 times 
    the area of triangle ABC, prove that the area of triangle ABC is 72. -/
theorem area_of_triangle_abc (A B C : ℝ × ℝ) : 
  let X : ℝ × ℝ := (6, 0)
  let Y : ℝ × ℝ := (8, 4)
  let Z : ℝ × ℝ := (10, 0)
  let area_xyz := abs ((X.1 * (Y.2 - Z.2) + Y.1 * (Z.2 - X.2) + Z.1 * (X.2 - Y.2)) / 2)
  let area_abc := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area_xyz = 0.1111111111111111 * area_abc →
  area_abc = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_l191_19175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alberto_bjorn_difference_alberto_carlos_difference_l191_19147

/-- Represents a biker's journey --/
structure BikerJourney where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- Calculate the distance traveled by a biker --/
def distanceTraveled (journey : BikerJourney) : ℝ :=
  journey.finish.2 - journey.start.2

/-- The time period of the journey in hours --/
def timePeriod : ℝ := 6

/-- Alberto's journey --/
def albertoJourney : BikerJourney := ⟨(0, 0), (6, 90)⟩

/-- Bjorn's journey --/
def bjornJourney : BikerJourney := ⟨(0, 0), (6, 72)⟩

/-- Carlos' journey --/
def carlosJourney : BikerJourney := ⟨(0, 0), (6, 60)⟩

/-- Theorem stating the difference in distance traveled between Alberto and Bjorn --/
theorem alberto_bjorn_difference :
  distanceTraveled albertoJourney - distanceTraveled bjornJourney = 18 := by
  simp [distanceTraveled, albertoJourney, bjornJourney]
  norm_num

/-- Theorem stating the difference in distance traveled between Alberto and Carlos --/
theorem alberto_carlos_difference :
  distanceTraveled albertoJourney - distanceTraveled carlosJourney = 30 := by
  simp [distanceTraveled, albertoJourney, carlosJourney]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alberto_bjorn_difference_alberto_carlos_difference_l191_19147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_condition_for_Q_angle_AQ_PD_unique_Q_dihedral_angle_A_PD_Q_l191_19176

-- Define the rectangular parallelepiped
structure RectParallelepiped where
  a : ℝ
  ab : ℝ
  pa : ℝ
  ab_eq : ab = Real.sqrt 3
  pa_eq : pa = 4

-- Define the point Q on BC
def Q (r : RectParallelepiped) : Type := ℝ

-- Define the condition for PQ ⊥ QD
def PQ_perp_QD (r : RectParallelepiped) (q : Q r) : Prop := sorry

-- Part 1: Existence condition for Q
theorem existence_condition_for_Q (r : RectParallelepiped) :
  (∃ q : Q r, PQ_perp_QD r q) → r.a ≥ 2 * Real.sqrt 3 := by sorry

-- Part 2: Angle between AQ and PD when Q is unique
noncomputable def angle_between_lines : ℝ → ℝ → ℝ := sorry

theorem angle_AQ_PD_unique_Q (r : RectParallelepiped) :
  (∃! q : Q r, PQ_perp_QD r q) →
  ∃ θ : ℝ, θ = Real.arccos (-Real.sqrt 42 / 14) ∧
            angle_between_lines 0 0 = θ := by sorry

-- Part 3: Dihedral angle A-PD-Q when a = 4
noncomputable def dihedral_angle : ℝ → ℝ → ℝ → ℝ := sorry

theorem dihedral_angle_A_PD_Q (r : RectParallelepiped) (q : Q r) :
  r.a = 4 → PQ_perp_QD r q →
  ∃ θ : ℝ, (θ = Real.arccos (Real.sqrt 15 / 5) ∨
            θ = Real.arccos (Real.sqrt 7 / 7)) ∧
           dihedral_angle 0 0 0 = θ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_condition_for_Q_angle_AQ_PD_unique_Q_dihedral_angle_A_PD_Q_l191_19176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_repayment_correct_l191_19125

/-- Calculate the annual repayment amount for a loan -/
noncomputable def annual_repayment (a : ℝ) (r : ℝ) : ℝ :=
  (a * r * (1 + r)^5) / ((1 + r)^5 - 1)

/-- Theorem: The annual repayment amount for a 5-year loan is correct -/
theorem annual_repayment_correct (a : ℝ) (r : ℝ) (x : ℝ) 
    (h_a : a > 0) (h_r : r > 0) (h_x : x = annual_repayment a r) :
  a * (1 + r)^5 = x * ((1 + r)^5 - 1) / r := by
  sorry

#check annual_repayment_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_repayment_correct_l191_19125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l191_19180

/-- The distance between the foci of a hyperbola defined by xy = 4 -/
noncomputable def distance_between_foci (x y : ℝ) : ℝ :=
  4 * Real.sqrt 2

/-- Theorem: The distance between the foci of the hyperbola xy = 4 is 4√2 -/
theorem hyperbola_foci_distance :
  ∀ x y : ℝ, x * y = 4 → distance_between_foci x y = 4 * Real.sqrt 2 := by
  intros x y h
  unfold distance_between_foci
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l191_19180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_range_l191_19136

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)*x^2 - 2*x + 5

-- State the theorem
theorem f_upper_bound_range :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (-1) 2, f x < m) ↔ m ∈ Set.Ioi 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_range_l191_19136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_volume_l191_19145

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  /-- The three lateral edges are mutually perpendicular -/
  edges_perpendicular : Bool
  /-- The lengths of the three lateral edges -/
  edge_lengths : Fin 3 → ℝ
  /-- All four vertices lie on the same spherical surface -/
  vertices_on_sphere : Bool

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Theorem stating the volume of the sphere containing the triangular pyramid -/
theorem pyramid_sphere_volume (p : TriangularPyramid) 
  (h1 : p.edges_perpendicular = true)
  (h2 : p.edge_lengths 0 = 1)
  (h3 : p.edge_lengths 1 = Real.sqrt 3)
  (h4 : p.edge_lengths 2 = 2)
  (h5 : p.vertices_on_sphere = true) :
  ∃ (r : ℝ), sphere_volume r = (8 * Real.sqrt 2 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_volume_l191_19145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l191_19107

/-- Given a complex number z satisfying |z-3+4i| = |z+3-4i|, 
    the set of all such z forms a straight line on the complex plane. -/
theorem trajectory_is_straight_line : 
  ∃ (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0), 
    {z : ℂ | Complex.abs (z - (3 - 4*I)) = Complex.abs (z + (3 - 4*I))} = 
    {z : ℂ | a * z.re + b * z.im = c} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l191_19107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_total_distance_l191_19168

/-- Represents Nina's running drill with various segments --/
structure RunningDrill where
  warmUp : Float
  firstHillUphill : Float
  firstHillDownhill : Float
  firstHillRecovery : Float
  tempoRun : Float
  secondHillUphill : Float
  secondHillDownhill : Float
  secondHillRecovery : Float
  fartlekTraining : Float
  intervalSprintYards : Float
  intervalJoggingBetweenSprints : Float
  coolDown : Float

/-- Calculates the total distance of Nina's running drill --/
def totalDistance (drill : RunningDrill) : Float :=
  drill.warmUp +
  drill.firstHillUphill + drill.firstHillDownhill + drill.firstHillRecovery +
  drill.tempoRun +
  drill.secondHillUphill + drill.secondHillDownhill + drill.secondHillRecovery +
  drill.fartlekTraining +
  (drill.intervalSprintYards / 1760) + drill.intervalJoggingBetweenSprints +
  drill.coolDown

/-- Theorem stating that Nina's total running distance is approximately 5.877 miles --/
theorem nina_total_distance :
  let drill := RunningDrill.mk 0.25 0.15 0.25 0.15 1.5 0.2 0.35 0.1 1.8 400 1.6 0.3
  Float.abs (totalDistance drill - 5.877) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_total_distance_l191_19168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l191_19188

/-- Calculates the selling price of an item given its cost price and gain percentage. -/
noncomputable def selling_price (cost_price : ℝ) (gain_percentage : ℝ) : ℝ :=
  cost_price * (1 + gain_percentage / 100)

/-- Theorem stating that the selling price of a cycle with a cost price of 900 and a gain of 25% is 1125. -/
theorem cycle_selling_price :
  selling_price 900 25 = 1125 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that 900 * (1 + 25 / 100) = 1125
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l191_19188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_score_relation_derek_score_percentage_difference_l191_19124

/-- Represents the scores of five friends in two aptitude tests -/
structure TestScores where
  first_test : Fin 5 → ℚ
  second_test : Fin 5 → ℚ

/-- The average score of the first test -/
def average_first_test (scores : TestScores) : ℚ :=
  (Finset.sum Finset.univ scores.first_test) / 5

/-- Derek's index (assuming it's 3, but it doesn't matter for the proof) -/
def derek_index : Fin 5 := 3

/-- Theorem stating the relationship between Derek's scores and the average -/
theorem derek_score_relation (scores : TestScores) 
    (h1 : scores.first_test derek_index = 1/2 * average_first_test scores)
    (h2 : scores.second_test derek_index = 3/2 * scores.first_test derek_index)
    (h3 : ∀ i : Fin 5, i ≠ derek_index → scores.second_test i = scores.first_test i) :
    scores.second_test derek_index = 3/4 * average_first_test scores := by
  sorry

/-- Main theorem proving Derek's score in the second test is 25% less than the average -/
theorem derek_score_percentage_difference (scores : TestScores) 
    (h1 : scores.first_test derek_index = 1/2 * average_first_test scores)
    (h2 : scores.second_test derek_index = 3/2 * scores.first_test derek_index)
    (h3 : ∀ i : Fin 5, i ≠ derek_index → scores.second_test i = scores.first_test i) :
    (average_first_test scores - scores.second_test derek_index) / average_first_test scores = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_score_relation_derek_score_percentage_difference_l191_19124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l191_19197

noncomputable section

open Real

-- Define necessary structures and instances
structure Point where
  x : ℝ
  y : ℝ

instance : Dist Point where
  dist a b := Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

def onCircle (p : Point) (center : Point) (radius : ℝ) : Prop :=
  dist p center = radius

def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)

def angle (a b c : Point) : ℝ :=
  Real.arccos (((b.x - a.x) * (c.x - a.x) + (b.y - a.y) * (c.y - a.y)) /
    (dist a b * dist a c))

theorem circle_chord_length (O A M B C : Point) (α : ℝ) :
  -- Circle with center O and radius 15
  dist O A = 15 ∧
  dist O B = 15 ∧
  dist O C = 15 ∧
  -- M is on radius OA
  collinear O A M ∧
  -- B and C are on the circle
  onCircle B O 15 ∧
  onCircle C O 15 ∧
  -- Angle conditions
  angle A M B = α ∧
  angle O M C = α ∧
  -- Given sin(α)
  sin α = Real.sqrt 21 / 5 →
  -- Conclusion: length of BC is 12
  dist B C = 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l191_19197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l191_19142

/-- A regular polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : Nat
  vertices : Finset (Fin 2 → ℝ)

/-- The set of intersection points between sides of different polygons -/
def intersectionPoints (polygons : List InscribedPolygon) : Finset (Fin 2 → ℝ) := sorry

/-- The number of intersection points between sides of different polygons -/
def numIntersectionPoints (polygons : List InscribedPolygon) : Nat :=
  (intersectionPoints polygons).card

theorem intersection_points_count
  (p6 p7 p8 p9 : InscribedPolygon)
  (h6 : p6.sides = 6)
  (h7 : p7.sides = 7)
  (h8 : p8.sides = 8)
  (h9 : p9.sides = 9)
  (shared_vertices : (p6.vertices ∩ p9.vertices).card = 2)
  (no_other_shared : ∀ (p q : InscribedPolygon),
    p ≠ q → (p, q) ≠ (p6, p9) → (p, q) ≠ (p9, p6) →
    (p.vertices ∩ q.vertices).card = 0)
  (no_triple_intersections : ∀ (p q r : InscribedPolygon),
    p ≠ q → q ≠ r → p ≠ r →
    (intersectionPoints [p, q, r]).card = 0) :
  numIntersectionPoints [p6, p7, p8, p9] = 76 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l191_19142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l191_19190

theorem trig_identity (α : ℝ) 
  (h : (Real.sin α + 3 * Real.cos α) / (3 * Real.cos α - Real.sin α) = 5) : 
  Real.sin α ^ 2 - Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l191_19190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_31_l191_19178

def mySequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * mySequence n + 1

theorem fifth_term_is_31 : mySequence 4 = 31 := by
  rfl

#eval mySequence 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_31_l191_19178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_individual_l191_19102

def RandomNumberTable : List (List Nat) :=
  [[7816, 6572, 0802, 6314, 0702, 4311],
   [3204, 9234, 4935, 8200, 3623, 4869]]

def PopulationSize : Nat := 20

def ValidNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ PopulationSize

def SelectIndividuals (table : List (List Nat)) : List Nat :=
  let flattened := table.join
  let pairs := flattened.map (λ n => n % 100)
  pairs.filter ValidNumber

theorem fifth_selected_individual :
  (SelectIndividuals RandomNumberTable).get? 4 = some 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_individual_l191_19102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_configuration_l191_19152

/-- A configuration of points in a square --/
structure DiscConfiguration where
  n : ℕ
  points : Finset (ℝ × ℝ)

/-- Predicate to check if a configuration is valid --/
def ValidConfiguration (c : DiscConfiguration) : Prop :=
  c.n > 1 ∧
  c.points.card = c.n^2 ∧
  (∀ p ∈ c.points, 0 ≤ p.1 ∧ p.1 ≤ c.n ∧ 0 ≤ p.2 ∧ p.2 ≤ c.n) ∧
  (∀ p q, p ∈ c.points → q ∈ c.points → p ≠ q → (p.1 - q.1)^2 + (p.2 - q.2)^2 > 1)

/-- Theorem stating that there exists a valid configuration --/
theorem exists_valid_configuration : ∃ c : DiscConfiguration, ValidConfiguration c := by
  -- We know that n = 10 works, so we'll use that
  let n := 10
  -- We'll construct the points later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_configuration_l191_19152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_theorem_l191_19173

/-- The circle equation: x^2 + y^2 - 4x - 4y - 10 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 10 = 0

/-- The line equation: x + y - 14 = 0 -/
def line_eq (x y : ℝ) : Prop := x + y - 14 = 0

/-- A point (x, y) is on the circle if it satisfies the circle equation -/
def point_on_circle (x y : ℝ) : Prop := circle_eq x y

/-- Distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 14| / Real.sqrt 2

/-- The theorem stating the difference between max and min distance -/
theorem distance_difference_theorem :
  ∃ (max_dist min_dist : ℝ),
    (∀ (x y : ℝ), point_on_circle x y → 
      min_dist ≤ distance_to_line x y ∧ distance_to_line x y ≤ max_dist) ∧
    max_dist - min_dist = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_theorem_l191_19173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l191_19171

theorem min_distance_to_line (x y : ℤ) : 
  (|25 * x - 15 * y + 12| : ℝ) / Real.sqrt (25^2 + 15^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l191_19171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_difference_l191_19156

/-- The price of a single notebook in cents -/
def notebook_price : ℚ := sorry

/-- The number of notebooks Carl bought -/
def carl_notebooks : ℕ := sorry

/-- The number of notebooks Mina bought -/
def mina_notebooks : ℕ := sorry

theorem notebook_difference :
  notebook_price > (10 : ℚ) →
  notebook_price * carl_notebooks = 234 →
  notebook_price * mina_notebooks = 312 →
  mina_notebooks = carl_notebooks + 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_difference_l191_19156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_ellipse_through_points_l191_19100

noncomputable section

-- Define the ellipse E
def ellipse_E (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_equation :
  ∃ a b : ℝ, 
    ellipse_E a b 0 (-1) ∧ 
    eccentricity a b = Real.sqrt 2 / 2 ∧
    (∀ x y : ℝ, ellipse_E a b x y ↔ x^2 / 2 + y^2 = 1) := by
  sorry

-- Define a general ellipse
def general_ellipse (m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / m + y^2 / n = 1 ∧ m > 0 ∧ n > 0 ∧ m ≠ n

theorem ellipse_through_points :
  ∃ m n : ℝ,
    general_ellipse m n 2 (Real.sqrt 2) ∧
    general_ellipse m n (Real.sqrt 6) 1 ∧
    (∀ x y : ℝ, general_ellipse m n x y ↔ x^2 / 8 + y^2 / 4 = 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_ellipse_through_points_l191_19100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constructible_under_4300000000_l191_19129

/-- A prime number p is a Fermat prime if it is of the form 2^(2^k) + 1 for some non-negative integer k. -/
def is_fermat_prime (p : ℕ) : Prop :=
  ∃ k : ℕ, p = 2^(2^k) + 1 ∧ Nat.Prime p

/-- A natural number n is constructible if it is a product of a power of 2 and distinct Fermat primes. -/
def is_constructible (n : ℕ) : Prop :=
  ∃ k : ℕ, ∃ primes : List ℕ,
    (∀ p, p ∈ primes → is_fermat_prime p) ∧
    (∀ p q, p ∈ primes → q ∈ primes → p ≠ q → p ≠ q) ∧
    n = 2^k * (primes.prod)

/-- The theorem states that 2^32 is the largest constructible number less than 4,300,000,000. -/
theorem largest_constructible_under_4300000000 :
  (∀ n : ℕ, n < 4300000000 → is_constructible n → n ≤ 2^32) ∧
  is_constructible (2^32) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constructible_under_4300000000_l191_19129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l191_19115

theorem cos_plus_sin_value (α : Real) 
  (h1 : Real.tan α = -2) 
  (h2 : π/2 < α) 
  (h3 : α < π) : 
  Real.cos α + Real.sin α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l191_19115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_min_value_l191_19110

theorem geometric_sequence_min_value (a : ℕ → ℝ) (q : ℝ) (m n : ℕ+) :
  (∀ k, a k = q ^ k) →
  a 1 = q →
  a m * (a n)^2 = (a 4)^2 →
  (2 : ℝ) / m.val + 1 / n.val ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_min_value_l191_19110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_is_3_sqrt_5_l191_19189

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (-2, 1)
def b (k : ℝ) : ℝ × ℝ := (k, -3)
def c : ℝ × ℝ := (1, 2)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Main theorem -/
theorem magnitude_of_b_is_3_sqrt_5 (k : ℝ) :
  dot_product (a - (2 • b k)) c = 0 →
  magnitude (b k) = 3 * Real.sqrt 5 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_is_3_sqrt_5_l191_19189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_sum_max_l191_19164

theorem triangle_cos_sum_max (A B C : ℝ) : 
  A = π/3 → -- 60° in radians
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  Real.cos A + Real.cos B * Real.cos C ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_sum_max_l191_19164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_proof_l191_19192

noncomputable section

/-- The distance between two observers in miles -/
def distance_between_observers : ℝ := 12

/-- The angle of elevation from the first observer (Alice) in radians -/
noncomputable def angle_elevation_1 : ℝ := 30 * Real.pi / 180

/-- The angle of elevation from the second observer (Bob) in radians -/
noncomputable def angle_elevation_2 : ℝ := 45 * Real.pi / 180

/-- The altitude of the airplane in miles -/
noncomputable def airplane_altitude : ℝ := 6 * Real.sqrt 2

theorem airplane_altitude_proof :
  ∀ (d : ℝ) (θ₁ θ₂ : ℝ),
    d = distance_between_observers →
    θ₁ = angle_elevation_1 →
    θ₂ = angle_elevation_2 →
    ∃ (h : ℝ),
      h = airplane_altitude ∧
      h = d * (Real.tan θ₁ * Real.tan θ₂) / (Real.tan θ₁ + Real.tan θ₂) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_proof_l191_19192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_magnitude_max_a2_plus_c2_l191_19101

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.B

-- Theorem 1: Magnitude of angle B
theorem angle_B_magnitude (t : Triangle) (h : problem_conditions t) :
  t.B = π / 3 := by
  sorry

-- Theorem 2: Maximum value of a^2 + c^2 when b = √3
theorem max_a2_plus_c2 (t : Triangle) (h : problem_conditions t) (hb : t.b = Real.sqrt 3) :
  (∀ t' : Triangle, problem_conditions t' → t'.b = Real.sqrt 3 → t'.a^2 + t'.c^2 ≤ t.a^2 + t.c^2) →
  t.a^2 + t.c^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_magnitude_max_a2_plus_c2_l191_19101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_parametric_line_l191_19143

/-- The angle of inclination of a straight line given by parametric equations -/
theorem angle_of_inclination_parametric_line :
  ∀ (t : ℝ),
  let x : ℝ → ℝ := λ t => 5 - 3 * t
  let y : ℝ → ℝ := λ t => 3 + Real.sqrt 3 * t
  Real.arctan ((y 1 - y 0) / (x 1 - x 0)) * (180 / Real.pi) = 150 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_parametric_line_l191_19143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l191_19128

/-- Definition of an isosceles right triangle -/
def IsIsoscelesRight (a b c A B C : ℝ) : Prop :=
  a = b ∧ C = Real.pi / 2

/-- Given a triangle ABC where log a - log c = log sin B = -log √2 and B is an acute angle,
    prove that ABC is an isosceles right triangle -/
theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < Real.pi / 2 →
  Real.log a - Real.log c = Real.log (Real.sin B) →
  Real.log (Real.sin B) = -Real.log (Real.sqrt 2) →
  IsIsoscelesRight a b c A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l191_19128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_equation_solution_l191_19183

/-- Function representing exponentiation --/
def exp (m : ℕ) (n : ℕ) : ℕ := m ^ n

/-- Theorem stating the existence and uniqueness of n given m and k --/
theorem exp_equation_solution (m : ℕ) (k : ℕ) :
  ∃! n : ℕ, exp 10 m = n * exp 22 k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_equation_solution_l191_19183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_in_interval_l191_19104

-- Define the function f(x) = ln x - 3/e
noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 / Real.exp 1

-- Theorem statement
theorem zero_point_exists_in_interval :
  ∃ c ∈ Set.Ioo (Real.exp 1) ((Real.exp 1) ^ 2), f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_exists_in_interval_l191_19104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_rectangle_exists_l191_19194

/-- A color type representing red, green, and blue --/
inductive Color
  | Red
  | Green
  | Blue

/-- A type representing a 4 × 82 grid colored with three colors --/
def Grid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color --/
def is_monochromatic_rectangle (g : Grid) (x1 y1 x2 y2 : Fin 4) : Prop :=
  x1 < x2 ∧ y1 < y2 ∧
  g x1 y1 = g x1 y2 ∧
  g x1 y1 = g x2 y1 ∧
  g x1 y1 = g x2 y2

/-- Theorem: In any 3-coloring of a 4 × 82 grid, there exists a rectangle whose vertices are all the same color --/
theorem monochromatic_rectangle_exists (g : Grid) : 
  ∃ (x1 y1 x2 y2 : Fin 4), is_monochromatic_rectangle g x1 y1 x2 y2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_rectangle_exists_l191_19194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l191_19161

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_focal_length : Real.sqrt (h.a^2 + h.b^2) = 2)
  (h_focus_asymptote : h.b = Real.sqrt 3) : 
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l191_19161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_problem_l191_19193

/-- The edge length of the cube in centimeters -/
def cube_edge : ℝ := 2

/-- The area of the shadow cast by the cube, excluding the area beneath the cube, in square centimeters -/
def shadow_area : ℝ := 200

/-- The height of the light source above an upper vertex of the cube in centimeters -/
noncomputable def light_height : ℝ := Real.sqrt (shadow_area + cube_edge ^ 2)

/-- The greatest integer not exceeding 1000 times the light height -/
noncomputable def result : ℤ := ⌊1000 * light_height⌋

theorem cube_shadow_problem :
  result = 14282 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_problem_l191_19193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l191_19170

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) Real :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

noncomputable def angle : Real := 150 * Real.pi / 180

theorem rotation_150_degrees :
  rotation_matrix angle = ![![-Real.sqrt 3 / 2, -1 / 2],
                            ![1 / 2, -Real.sqrt 3 / 2]] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l191_19170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_satisfies_equations_tangency_point_is_unique_l191_19157

/-- The point of tangency for two parabolas -/
noncomputable def point_of_tangency : ℝ × ℝ := (-11/2, -43/2)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 12*x + 40

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 44*y + 400

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations : 
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y :=
by sorry

/-- Theorem stating that the point_of_tangency is unique -/
theorem tangency_point_is_unique :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_satisfies_equations_tangency_point_is_unique_l191_19157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_shift_equivalence_l191_19140

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def g (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem sine_shift_equivalence :
  ∀ x : ℝ, f (x - Real.pi / 6) = g x :=
by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_shift_equivalence_l191_19140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_product_l191_19112

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + t, 1 + Real.sqrt 2 * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 1 / (1 - Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point M
def M : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_point_product : 
  ∃ (t₁ t₂ : ℝ), 
    C₁ t₁ = A ∧ 
    C₁ t₂ = B ∧ 
    (A.1 - M.1)^2 + (A.2 - M.2)^2 * 
    ((B.1 - M.1)^2 + (B.2 - M.2)^2) = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_product_l191_19112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_squares_triangle_area_is_70_l191_19159

theorem triangle_area_from_squares (square1_area square2_area : ℝ) 
  (h1 : square1_area = 196)
  (h2 : square2_area = 100) : ℝ :=
  let side1 := Real.sqrt square1_area
  let side2 := Real.sqrt square2_area
  (1 / 2) * side1 * side2

theorem triangle_area_is_70 (square1_area square2_area : ℝ) 
  (h1 : square1_area = 196)
  (h2 : square2_area = 100) : 
  triangle_area_from_squares square1_area square2_area h1 h2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_squares_triangle_area_is_70_l191_19159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l191_19139

theorem cosine_inequality (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (hy : 0 ≤ y ∧ y ≤ π) :
  Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l191_19139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_45_degrees_l191_19108

theorem sec_negative_45_degrees : 1 / Real.cos (-45 * π / 180) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_45_degrees_l191_19108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sin_geq_half_l191_19119

open MeasureTheory Real Set

-- Define the interval [-3, 3]
def interval : Set ℝ := Icc (-3) 3

-- Define the event where sin(πx/6) ≥ 1/2
def event (x : ℝ) : Prop := Real.sin (π * x / 6) ≥ 1/2

-- Define a probability measure on the interval
noncomputable def probMeasure : Measure ℝ := 
  volume.restrict interval

-- State the theorem
theorem probability_sin_geq_half :
  probMeasure {x ∈ interval | event x} / probMeasure interval = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sin_geq_half_l191_19119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_7x6_l191_19123

/-- The number of paths on a grid from (0,0) to (m,n) moving only right or up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The width of the grid -/
def gridWidth : ℕ := 7

/-- The height of the grid -/
def gridHeight : ℕ := 6

theorem grid_paths_7x6 : gridPaths gridWidth gridHeight = 1716 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_7x6_l191_19123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l191_19199

noncomputable section

/-- The curve function --/
def f (x : ℝ) : ℝ := (1/3) * x^3 + x

/-- The point of interest on the curve --/
def point : ℝ × ℝ := (1, 4/3)

/-- The derivative of the curve function --/
def f' (x : ℝ) : ℝ := x^2 + 1

/-- The slope of the tangent line at the point of interest --/
def tangent_slope : ℝ := f' point.1

/-- The y-intercept of the tangent line --/
def y_intercept : ℝ := point.2 - tangent_slope * point.1

/-- The x-intercept of the tangent line --/
def x_intercept : ℝ := -y_intercept / tangent_slope

/-- The area of the triangle --/
def triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)

theorem tangent_triangle_area :
  triangle_area = 1/9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l191_19199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_and_half_radius_l191_19135

-- Define the circles and square
def circle_x (r : Real) : Prop := 2 * Real.pi * r = 20 * Real.pi
def circle_y (r : Real) : Prop := ∃ s, s = 2 * r ∧ s * s = 4 * r * r

-- State the theorem
theorem square_side_and_half_radius :
  ∀ rx ry : Real,
  circle_x rx →
  circle_y ry →
  rx * rx * Real.pi = ry * ry * Real.pi →
  ∃ s hr,
  s = 2 * ry ∧
  hr = ry / 2 ∧
  s = 20 ∧
  hr = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_and_half_radius_l191_19135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_discriminant_equality_l191_19148

theorem quadratic_root_discriminant_equality 
  (a b c x₀ : ℝ) 
  (h_root : a * x₀^2 + b * x₀ + c = 0) 
  (h_a_nonzero : a ≠ 0) : 
  (b^2 - 4*a*c) = (2*a*x₀ + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_discriminant_equality_l191_19148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_community_b_selection_l191_19106

/-- Represents the number of low-income families in a community -/
def LowIncomeFamilies := ℕ

/-- Represents the number of affordable housing units -/
def AffordableHousingUnits := ℕ

/-- Calculates the number of families selected from a community using stratified sampling -/
def stratifiedSample (communityFamilies totalFamilies : ℕ) (totalUnits : ℕ) : ℕ :=
  (communityFamilies * totalUnits) / totalFamilies

theorem community_b_selection 
  (community_a : ℕ)
  (community_b : ℕ)
  (community_c : ℕ)
  (total_units : ℕ)
  (h1 : community_a = 360)
  (h2 : community_b = 270)
  (h3 : community_c = 180)
  (h4 : total_units = 90) :
  stratifiedSample community_b (community_a + community_b + community_c) total_units = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_community_b_selection_l191_19106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cut_ratio_l191_19150

/-- The perimeter of a square -/
def perimeter_square (side : ℝ) : ℝ := 4 * side

/-- The perimeter of a regular pentagon -/
def perimeter_pentagon (side : ℝ) : ℝ := 5 * side

/-- Given a wire cut into two pieces, where one piece forms a square with perimeter x
    and the other forms a regular pentagon with perimeter y, prove that x/y = 1 when
    the perimeters are equal. -/
theorem wire_cut_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x = perimeter_square (x/4) → y = perimeter_pentagon (y/5) → x = y → x / y = 1 :=
by
  intros hx hy hx_square hy_pentagon hxy
  rw [hxy]
  exact div_self (ne_of_gt hy)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cut_ratio_l191_19150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l191_19141

-- Define the real number α
variable (α : ℝ)

-- Define the function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (x - α * x^2) / ((x + 1) * (1 - α^2))

-- State the theorem
theorem function_satisfies_equation (α : ℝ) (x : ℝ) (h : x > 0) :
  α * x^2 * f α (1/x) + f α x = x / (x + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l191_19141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_is_friday_l191_19177

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day of the week after a certain number of days
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

theorem first_day_is_friday (day26 : DayOfWeek) :
  day26 = DayOfWeek.Tuesday → dayAfter DayOfWeek.Friday 25 = day26 :=
by
  intro h
  sorry

#check first_day_is_friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_is_friday_l191_19177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_sigma_28_l191_19153

def σ (n : ℕ) : ℕ := (Nat.divisors n).sum id

theorem largest_n_with_sigma_28 : 
  ∀ n : ℕ, n > 0 → σ n = 28 → n ≤ 12 :=
by
  sorry

#eval σ 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_sigma_28_l191_19153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l191_19137

/-- Proves that a circular garden with the same perimeter as a 60x20 rectangular garden has a larger area --/
theorem garden_area_increase : 
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_area := rect_length * rect_width
  let perimeter := 2 * (rect_length + rect_width)
  let circle_radius := perimeter / (2 * Real.pi)
  let circle_area := Real.pi * circle_radius^2
  ∃ ε > 0, |circle_area - rect_area - 837.62| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l191_19137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l191_19191

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x, f (5 * Real.pi / 12 + x) = -f (5 * Real.pi / 12 - x)) ∧
  (∀ x, f x = Real.sin (2 * x + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l191_19191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_of_n_l191_19155

def n : ℕ := 2^3 * 3^1 * 7^2 * 5^1

def is_even_factor (k : ℕ) : Prop :=
  k ∣ n ∧ Even k

theorem count_even_factors_of_n :
  (Finset.filter (fun k => k ∣ n ∧ Even k) (Finset.range (n + 1))).card = 36 := by
  sorry

#eval (Finset.filter (fun k => k ∣ n ∧ Even k) (Finset.range (n + 1))).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_of_n_l191_19155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l191_19103

/-- The efficiency ratio between p and q -/
noncomputable def efficiency_ratio : ℝ := 1.6

/-- The time taken by p to complete the work alone -/
noncomputable def p_time : ℝ := 26

/-- The time taken by p and q together to complete the work -/
noncomputable def combined_time : ℝ := 1690 / 91

/-- Theorem stating the relationship between p's efficiency, p's time, and the combined time of p and q -/
theorem work_completion_time :
  combined_time = 1 / ((1 / p_time) + (1 / (efficiency_ratio * p_time))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l191_19103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_theorem_l191_19179

noncomputable def triangle_abc (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem abc_theorem (a b c : ℝ) (A B C : ℝ) :
  triangle_abc a b c A B C →
  let vec_a := (2 * Real.sin C, Real.sqrt 3 * Real.cos C)
  let vec_b := (-Real.sin C, 2 * Real.sin C)
  let f := vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2
  f = 1 →
  c = 1 →
  a * b = 2 * Real.sqrt 3 →
  a > b →
  a = 2 ∧ b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_theorem_l191_19179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_salary_l191_19134

/-- Represents John's salary and bonus information --/
structure SalaryInfo where
  lastYearSalary : ℚ
  lastYearBonus : ℚ
  thisYearTotal : ℚ

/-- Calculates John's salary before bonus for this year --/
def calculateSalary (info : SalaryInfo) : ℚ :=
  let bonusRate := info.lastYearBonus / info.lastYearSalary
  info.thisYearTotal / (1 + bonusRate)

/-- Theorem stating that John's salary before bonus this year is $200,000 --/
theorem johns_salary (info : SalaryInfo) 
    (h1 : info.lastYearSalary = 100000)
    (h2 : info.lastYearBonus = 10000)
    (h3 : info.thisYearTotal = 220000) : 
  calculateSalary info = 200000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_salary_l191_19134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_cut_pentagon_area_l191_19196

/-- A pentagon formed by cutting a triangular corner from a rectangle -/
structure CornerCutPentagon where
  sides : Fin 5 → ℕ
  is_valid_sides : ∃ (p : Equiv.Perm (Fin 5)), (λ i => sides (p i)) = ![17, 23, 24, 30, 37]

/-- The area of a CornerCutPentagon -/
def area (p : CornerCutPentagon) : ℕ := sorry

/-- Theorem stating that a CornerCutPentagon with the given side lengths has an area of 900 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : area p = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_cut_pentagon_area_l191_19196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_equality_l191_19146

/-- The work rate of one man -/
noncomputable def man_rate : ℝ := 1

/-- The work rate of one woman -/
noncomputable def woman_rate : ℝ := 1/2

/-- The number of men in the second group -/
def x : ℕ := 6

theorem work_rate_equality :
  (3 * man_rate + 8 * woman_rate = x * man_rate + 2 * woman_rate) ∧
  (2 * man_rate + 2 * woman_rate = (3/7) * (3 * man_rate + 8 * woman_rate)) →
  x = 6 :=
by
  intro h
  sorry

#check work_rate_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_equality_l191_19146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_plus_pi_fourth_l191_19127

theorem cos_theta_plus_pi_fourth (θ : ℝ) 
  (h1 : Real.cos θ = -12/13) 
  (h2 : θ ∈ Set.Ioo π (3*π/2)) : 
  Real.cos (θ + π/4) = -7*Real.sqrt 2/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_plus_pi_fourth_l191_19127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_quadrilateral_area_l191_19186

/-- Represents a parabola with equation y = 2px^2 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def Parabola.focus (par : Parabola) : Point :=
  { x := 0, y := 1 / (4 * par.p) }

noncomputable def Parabola.directrix_y (par : Parabola) : ℝ :=
  -1 / (4 * par.p)

def Parabola.contains (par : Parabola) (pt : Point) : Prop :=
  pt.y = 2 * par.p * pt.x^2

noncomputable def perpendicular_foot (par : Parabola) (pt : Point) : Point :=
  { x := pt.x, y := par.directrix_y }

noncomputable def axis_directrix_intersection (par : Parabola) : Point :=
  { x := 0, y := par.directrix_y }

noncomputable def quadrilateral_area (A B C D : Point) : ℝ :=
  sorry  -- Definition of area calculation

theorem parabola_quadrilateral_area 
  (par : Parabola) 
  (P : Point) 
  (h_P_on_parabola : par.contains P) 
  (h_P_coords : P.x = 1 ∧ P.y = 1/4) :
  let F := par.focus
  let Q := perpendicular_foot par P
  let M := axis_directrix_intersection par
  quadrilateral_area P Q M F = 13/8 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_quadrilateral_area_l191_19186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_perpendicular_l191_19138

-- Define the equations
def equation1 (x y : ℝ) : Prop := 4 * y - 3 * x = 16
def equation4 (x y : ℝ) : Prop := 3 * y + 4 * x = 15

-- Define the slope of a line given its equation
noncomputable def slopeOf (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

-- Define perpendicularity of two lines
def perpendicular (eq1 eq2 : (ℝ → ℝ → Prop)) : Prop :=
  slopeOf eq1 * slopeOf eq2 = -1

-- Theorem statement
theorem equations_perpendicular :
  perpendicular equation1 equation4 := by
  sorry

#check equations_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_perpendicular_l191_19138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_penalty_kick_game_theorem_l191_19181

/-- Football penalty kick game between A and B -/
structure PenaltyKickGame where
  prob_score : ℝ  -- Probability of A and B scoring a goal
  prob_a_block : ℝ  -- Probability of A blocking B's shot
  prob_b_block : ℝ  -- Probability of B blocking A's shot

/-- Score of player A after one round -/
inductive Score
  | Negative : Score  -- -1
  | Zero : Score      -- 0
  | Positive : Score  -- 1

/-- Probability distribution of A's score after one round -/
def score_distribution (game : PenaltyKickGame) : Score → ℝ
  | Score.Negative => (1 - game.prob_score * (1 - game.prob_b_block)) * game.prob_score * (1 - game.prob_a_block)
  | Score.Zero => game.prob_score * (1 - game.prob_b_block) * game.prob_score * (1 - game.prob_a_block) + 
                  (1 - game.prob_score * (1 - game.prob_b_block)) * (1 - game.prob_score * (1 - game.prob_a_block))
  | Score.Positive => game.prob_score * (1 - game.prob_b_block) * (1 - game.prob_score * (1 - game.prob_a_block))

/-- Expected value of A's score after one round -/
def expected_score (game : PenaltyKickGame) : ℝ :=
  -1 * score_distribution game Score.Negative +
   0 * score_distribution game Score.Zero +
   1 * score_distribution game Score.Positive

/-- Probability that A's cumulative score is higher than B's after 3 rounds -/
def prob_a_wins_after_3_rounds (game : PenaltyKickGame) : ℝ :=
  let p_pos := score_distribution game Score.Positive
  let p_zero := score_distribution game Score.Zero
  let p_neg := score_distribution game Score.Negative
  p_pos^3 + 3 * p_pos^2 * p_zero + 3 * p_pos^2 * p_neg + 3 * p_pos * p_zero^2

theorem penalty_kick_game_theorem (game : PenaltyKickGame) 
  (h1 : game.prob_score = 1/2) 
  (h2 : game.prob_a_block = 1/2) 
  (h3 : game.prob_b_block = 1/3) : 
  score_distribution game Score.Negative = 1/6 ∧ 
  score_distribution game Score.Zero = 7/12 ∧ 
  score_distribution game Score.Positive = 1/4 ∧
  expected_score game = 1/12 ∧
  prob_a_wins_after_3_rounds game = 79/192 := by
  sorry

#eval "This is a placeholder for evaluation. The actual proof is omitted with 'sorry'."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_penalty_kick_game_theorem_l191_19181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_water_cells_l191_19167

/-- Represents a cell in the grid -/
inductive Cell
| Water
| Sandbag
| Empty

/-- Represents the grid -/
def Grid := List (List Cell)

/-- Represents a move in the game -/
def Move := List (Nat × Nat)

/-- The game state -/
structure GameState where
  grid : Grid
  moves : Nat

/-- Initializes the grid with water in the top row -/
def initGrid : Grid :=
  let waterRow := List.replicate 14 Cell.Water
  let emptyRow := List.replicate 14 Cell.Empty
  waterRow :: List.replicate 13 emptyRow

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Simulates water flow after a move -/
def simulateWaterFlow (state : GameState) : GameState := sorry

/-- Counts the number of water cells in the grid -/
def countWaterCells (grid : Grid) : Nat := sorry

/-- Theorem: The minimum number of water-filled cells is 37 -/
theorem min_water_cells : 
  ∀ (strategy : List Move), 
    let finalState := (strategy.foldl (λ s m => simulateWaterFlow (applyMove s m)) 
                      { grid := initGrid, moves := 0 })
    countWaterCells finalState.grid ≥ 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_water_cells_l191_19167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l191_19182

/-- Given a journey where 80% of the distance is traveled at 80 mph and the remaining 20% at 20 mph, 
    the average speed for the entire journey is 50 mph. -/
theorem average_speed_theorem (d : ℝ) (h : d > 0) : 
  (d / ((0.8 * d / 80) + (0.2 * d / 20))) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l191_19182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_normal_hours_l191_19163

/-- Represents Lloyd's work scenario -/
structure WorkScenario where
  regularRate : ℚ  -- Regular hourly rate
  overtimeMultiplier : ℚ  -- Overtime pay multiplier
  totalHoursWorked : ℚ  -- Total hours worked on the given day
  totalEarnings : ℚ  -- Total earnings for the given day

/-- Calculates the normal work hours given a work scenario -/
noncomputable def calculateNormalHours (scenario : WorkScenario) : ℚ :=
  let regularEarnings := scenario.regularRate * scenario.totalHoursWorked
  let overtimeEarnings := (scenario.totalEarnings - regularEarnings) / (scenario.regularRate * scenario.overtimeMultiplier)
  scenario.totalHoursWorked - overtimeEarnings

/-- Theorem stating that Lloyd's normal work hours are 7.5 -/
theorem lloyds_normal_hours (scenario : WorkScenario) 
  (h1 : scenario.regularRate = 4)
  (h2 : scenario.overtimeMultiplier = 3/2)
  (h3 : scenario.totalHoursWorked = 21/2)
  (h4 : scenario.totalEarnings = 48) :
  calculateNormalHours scenario = 15/2 := by
  sorry

#eval (15 : ℚ) / 2  -- To verify that 15/2 is indeed 7.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_normal_hours_l191_19163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l191_19121

noncomputable def f (a x : ℝ) : ℝ := 1/x + Real.log ((1 + a*x) / (1 - x))

theorem odd_function_implies_a_equals_one :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l191_19121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l191_19187

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the squared distance between two points
noncomputable def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Theorem statement
theorem triangle_centroid_property (t : Triangle) :
  let G := centroid t
  (dist_squared G t.A + dist_squared G t.B + dist_squared G t.C = 75) →
  (dist_squared t.A t.B + dist_squared t.A t.C + dist_squared t.B t.C = 225) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l191_19187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l191_19122

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2^x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l191_19122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l191_19198

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x - (2 : ℝ)^(-x)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f (2*x + 1) + f 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l191_19198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l191_19166

theorem log_inequality_range (a : ℝ) : 
  (0 < a ∧ a ≠ 1 ∧ Real.log (2/5) / Real.log a < 1) ↔ (0 < a ∧ a < 2/5) ∨ (a > 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l191_19166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_problem_l191_19133

theorem quadratic_function_problem (f g : ℝ → ℝ) (a : ℝ) :
  (∃ k : ℝ, ∀ x, f x = k * (x - 1)^2 + 16) →  -- vertex at (1, 16)
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 8) →  -- roots 8 units apart
  (∀ x, g x = (2 - 2*a)*x - f x) →  -- definition of g(x)
  (∀ x ∈ Set.Icc 0 2, g x ≤ 5) →  -- maximum value of g(x) in [0, 2] is 5
  (∃ x ∈ Set.Icc 0 2, g x = 5) →  -- g(x) reaches the maximum value 5 in [0, 2]
  a = -4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_problem_l191_19133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_q_l191_19144

theorem existence_of_prime_q (p : ℕ) (h_p : Nat.Prime p) : 
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬(q ∣ (n^p : ℕ) - p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_q_l191_19144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l191_19109

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), (∀ x, tangent_line x (m * x + b)) ∧
               (f 0 = m * 0 + b) ∧
               (deriv f 0 = m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l191_19109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l191_19174

theorem sin_double_angle (α : Real) 
  (h1 : Real.sin α = -4/5) 
  (h2 : α > -Real.pi/2 ∧ α < Real.pi/2) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l191_19174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_leq_x_pow_5_eq_one_sixth_l191_19149

/-- The probability that y ≤ x^5 when x and y are uniformly distributed over [0,1] -/
noncomputable def probability_y_leq_x_pow_5 : ℝ :=
  ∫ x in Set.Icc 0 1, x^5

theorem probability_y_leq_x_pow_5_eq_one_sixth :
  probability_y_leq_x_pow_5 = 1/6 := by
  sorry

-- #eval probability_y_leq_x_pow_5  -- Removed as it's noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_leq_x_pow_5_eq_one_sixth_l191_19149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l191_19160

noncomputable def f (x : ℝ) := (2 : ℝ)^(x^2 - 2*x + 4)

theorem f_monotonicity :
  (∀ x y : ℝ, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ y < 1 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l191_19160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_a_b_l191_19165

theorem unique_solution_a_b (a b : ℕ+) 
  (h1 : (2 * (a : ℝ) ^ (b : ℝ) + 16 + 3 * (a : ℝ) ^ (b : ℝ) - 8) / 2 = 84) : 
  a = 2 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_a_b_l191_19165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_coordinates_l191_19111

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (9, 0)

-- Define the vertical line intersecting AC at R and BC at S
def R : ℝ × ℝ → Prop := λ r => True
def S : ℝ × ℝ → Prop := λ s => True

-- Define that R is on AC and S is on BC
axiom R_on_AC : ∀ r, R r → (r.2 = -2/3 * r.1 + 6)
axiom S_on_BC : ∀ s, S s → s.2 = 0

-- Define that R and S have the same x-coordinate (vertical line)
axiom R_S_vertical : ∀ r s, R r → S s → r.1 = s.1

-- Define the area of triangle RSC
noncomputable def area_RSC (r : ℝ × ℝ) : ℝ := (1/2) * |9 - r.1| * |r.2|

-- Theorem statement
theorem difference_of_coordinates :
  ∀ r, R r → area_RSC r = 15 → |r.1 - r.2| = 17/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_coordinates_l191_19111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_when_a_greater_one_root_for_g_l191_19131

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

-- Theorem for part B
theorem two_roots_when_a_greater (a : ℝ) (h : a > 1 + Real.log 2) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f x₁ = a ∧ f x₂ = a ∧
  ∀ x : ℝ, x > 0 → f x = a → (x = x₁ ∨ x = x₂) :=
by sorry

-- Theorem for part C
theorem one_root_for_g :
  ∃! x : ℝ, x > 0 ∧ f x = x :=
by sorry

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := f x - x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_when_a_greater_one_root_for_g_l191_19131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_root_greater_than_1_999_l191_19184

-- Define the polynomial
noncomputable def P (n : ℕ) (x : ℝ) : ℝ :=
  x^n - (x^n - 1) / (x - 1)

-- State the theorem
theorem smallest_n_for_root_greater_than_1_999 :
  (∀ k < 10, ¬ ∃ x > 1.999, P k x = 0) ∧
  (∃ x > 1.999, P 10 x = 0) := by
  sorry

#check smallest_n_for_root_greater_than_1_999

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_root_greater_than_1_999_l191_19184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l191_19130

noncomputable def line1 : ℝ → ℝ := λ x => -3 * (x - 5) + 5

noncomputable def line2 : ℝ → ℝ := λ x => -x + 10

noncomputable def x_intercept : ℝ := 10/3

noncomputable def y_intercept : ℝ := line1 0

noncomputable def area : ℝ := 175/3

theorem quadrilateral_area :
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, y_intercept)
  let E : ℝ × ℝ := (5, 5)
  let C : ℝ × ℝ := (10, 0)
  (area = (1/2 * x_intercept * y_intercept) + (1/2 * 10 * 5)) ∧
  (line1 5 = 5) ∧ (line2 5 = 5) ∧ (line2 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l191_19130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_expression_equal_100_l191_19169

/-- An arithmetic expression using numbers 1 to 9 --/
inductive ArithExpr
  | num : Fin 9 → ArithExpr
  | add : ArithExpr → ArithExpr → ArithExpr
  | sub : ArithExpr → ArithExpr → ArithExpr
  | mul : ArithExpr → ArithExpr → ArithExpr
  | div : ArithExpr → ArithExpr → ArithExpr

/-- Evaluate an arithmetic expression --/
def eval : ArithExpr → ℚ
  | ArithExpr.num n => (n.val + 1 : ℚ)
  | ArithExpr.add a b => eval a + eval b
  | ArithExpr.sub a b => eval a - eval b
  | ArithExpr.mul a b => eval a * eval b
  | ArithExpr.div a b => eval a / eval b

/-- Check if an expression uses each number from 1 to 9 exactly once --/
def usesAllNumbers : ArithExpr → Bool := sorry

/-- There exists an arithmetic expression using numbers 1 to 9 that evaluates to 100 --/
theorem exists_expression_equal_100 : ∃ e : ArithExpr, usesAllNumbers e ∧ eval e = 100 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_expression_equal_100_l191_19169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l191_19118

-- Define the parabola
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2)^2 = 4 * P.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the distance to y-axis
def distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs P.1

-- Theorem statement
theorem parabola_point_distance (P : ℝ × ℝ) :
  is_on_parabola P → distance P focus = 9 → distance_to_y_axis P = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l191_19118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_56_l191_19120

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The price of 1 table and 1 chair is $64 -/
axiom combined_price : chair_price + table_price = 64

/-- Theorem: The price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_56_l191_19120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_entry_exit_time_l191_19117

/-- Represents the position of an object in 2D space -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the state of the car and storm system at time t -/
structure SystemState (t : ℝ) where
  car_pos : Position
  storm_center : Position
  storm_radius : ℝ

/-- Calculates the position of the car at time t -/
noncomputable def car_position (t : ℝ) : Position :=
  { x := t, y := 0 }

/-- Calculates the position of the storm center at time t -/
noncomputable def storm_center_position (t : ℝ) : Position :=
  { x := (3 / 2) * t, y := 150 - (3 / 2) * t }

/-- Defines the state of the system at time t -/
noncomputable def system_state (t : ℝ) : SystemState t :=
  { car_pos := car_position t
  , storm_center := storm_center_position t
  , storm_radius := 60 }

/-- Determines if the car is inside the storm at time t -/
noncomputable def is_car_in_storm (t : ℝ) : Prop :=
  let state := system_state t
  let dx := state.car_pos.x - state.storm_center.x
  let dy := state.car_pos.y - state.storm_center.y
  dx^2 + dy^2 ≤ state.storm_radius^2

/-- The entry and exit times of the car into and out of the storm -/
noncomputable def entry_exit_times : ℝ × ℝ :=
  sorry  -- Placeholder for the actual calculation of t₁ and t₂

/-- Theorem stating that the average of entry and exit times is 212.4 minutes -/
theorem average_entry_exit_time : 
  let (t₁, t₂) := entry_exit_times
  (t₁ + t₂) / 2 = 212.4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_entry_exit_time_l191_19117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_seven_l191_19132

def divisors_of_245 : List Nat := [5, 7, 35, 49, 245]

def has_common_factor_greater_than_one (a b : Nat) : Prop :=
  ∃ (f : Nat), f > 1 ∧ f ∣ a ∧ f ∣ b

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i, i < arr.length → has_common_factor_greater_than_one (arr[i]!) (arr[(i + 1) % arr.length]!)

theorem sum_of_adjacent_to_seven (arr : List Nat) :
  arr = divisors_of_245 →
  is_valid_arrangement arr →
  7 ∈ arr →
  (∃ i j, i < arr.length ∧ j < arr.length ∧
          arr[i]! = 7 ∧ 
          arr[j]! = 7 ∧ 
          i ≠ j ∧
          (arr[(i + 1) % arr.length]! + arr[(j - 1 + arr.length) % arr.length]! = 294 ∨
           arr[(i - 1 + arr.length) % arr.length]! + arr[(j + 1) % arr.length]! = 294)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_seven_l191_19132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l191_19154

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the point on the ellipse
noncomputable def point_on_ellipse : ℝ × ℝ := (2, 3 * Real.sqrt 3 / 2)

-- Define the standard form of an ellipse
noncomputable def standard_ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation : 
  ∃ (a b : ℝ), 
    (∀ x y, hyperbola x y → ∃ c, c^2 = 7 ∧ c > 0) ∧ 
    standard_ellipse a b point_on_ellipse.1 point_on_ellipse.2 ∧ 
    a^2 = 16 ∧ b^2 = 9 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l191_19154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_approx_l191_19195

/-- Calculate the cost price given the selling price and profit percentage -/
noncomputable def costPrice (sellingPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  sellingPrice / (1 + profitPercentage / 100)

/-- The selling price of Item A -/
def sellingPriceA : ℝ := 150

/-- The profit percentage of Item A -/
def profitPercentageA : ℝ := 25

/-- The selling price of Item B -/
def sellingPriceB : ℝ := 200

/-- The profit percentage of Item B -/
def profitPercentageB : ℝ := 20

/-- The selling price of Item C -/
def sellingPriceC : ℝ := 250

/-- The profit percentage of Item C -/
def profitPercentageC : ℝ := 15

/-- The total cost price of all three items -/
noncomputable def totalCostPrice : ℝ :=
  costPrice sellingPriceA profitPercentageA +
  costPrice sellingPriceB profitPercentageB +
  costPrice sellingPriceC profitPercentageC

theorem total_cost_price_approx :
  abs (totalCostPrice - 504.06) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_approx_l191_19195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_canoes_production_l191_19172

/-- The sum of a geometric sequence with first term a, common ratio r, and n terms -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The number of canoes built in the first month -/
def initial_canoes : ℕ := 5

/-- The ratio of increase in canoe production each month -/
def production_ratio : ℕ := 3

/-- The number of months in the production period -/
def production_months : ℕ := 6

/-- Theorem: The total number of canoes built over 6 months is 1820 -/
theorem total_canoes_production :
  geometric_sum (initial_canoes : ℝ) (production_ratio : ℝ) production_months = 1820 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_canoes_production_l191_19172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l191_19114

def M : Set ℝ := {x | x^2 - 5*x - 6 > 0}

theorem complement_of_M : Set.compl M = Set.Icc (-1) 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l191_19114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_onto_CD_l191_19113

noncomputable section

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (-2, 2)
def D : ℝ × ℝ := (-3, 5)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_CD : ℝ × ℝ := (D.1 - C.1, D.2 - C.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / magnitude w

theorem projection_AB_onto_CD :
  projection vector_AB vector_CD = 2 * Real.sqrt 10 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_onto_CD_l191_19113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shaded_squares_l191_19151

/-- Represents a 3×3 grid of squares --/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if two positions in the grid are adjacent --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if a shading is valid (no adjacent shaded squares) --/
def validShading (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, g p1.1 p1.2 ∧ g p2.1 p2.2 → ¬adjacent p1 p2

/-- Counts the number of shaded squares in a grid --/
def shadedCount (g : Grid) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 3)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) fun j =>
      if g i j then 1 else 0

/-- The main theorem: The maximum number of shaded squares in a valid 3×3 grid is 6 --/
theorem max_shaded_squares :
  (∃ g : Grid, validShading g ∧ shadedCount g = 6) ∧
  (∀ g : Grid, validShading g → shadedCount g ≤ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shaded_squares_l191_19151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l191_19162

/-- The vertex of a quadratic function f(x) = ax^2 + bx + c -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the distance between the vertices of the given quadratic functions is 5√2 -/
theorem distance_between_vertices : 
  let C := vertex 1 6 15
  let D := vertex 1 (-4) 5
  distance C D = 5 * Real.sqrt 2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l191_19162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_y_values_l191_19185

noncomputable def list : List ℝ := [8, 3, 6, 3, 7, 3]

noncomputable def mean (y : ℝ) : ℝ := (list.sum + y) / 7

def mode : ℝ := 3

noncomputable def median (y : ℝ) : ℝ :=
  if y ≤ 3 then 3
  else if y < 6 then y
  else 6

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a ≠ c

theorem sum_of_possible_y_values :
  ∃ y₁ y₂ : ℝ,
    (∀ y : ℝ,
      is_arithmetic_progression mode (median y) (mean y) →
      y = y₁ ∨ y = y₂) ∧
    y₁ + y₂ = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_y_values_l191_19185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l191_19126

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  B : ℝ

-- Define the conditions
def geometric_progression (t : Triangle) : Prop :=
  t.b ^ 2 = t.a * t.c

def side_b_is_2 (t : Triangle) : Prop :=
  t.b = 2

def angle_B_is_pi_div_3 (t : Triangle) : Prop :=
  t.B = Real.pi / 3

-- Define the area function
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_area_is_sqrt_3 (t : Triangle) 
  (h1 : geometric_progression t) 
  (h2 : side_b_is_2 t) 
  (h3 : angle_B_is_pi_div_3 t) : 
  area t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l191_19126
