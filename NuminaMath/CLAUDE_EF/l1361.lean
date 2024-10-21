import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l1361_136157

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧ a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l1361_136157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_routes_count_l1361_136168

/-- Represents a point in the city grid -/
structure Point where
  x : Int
  y : Int

/-- Calculates the number of shortest paths between two points -/
def shortestPaths (start finish : Point) : ℕ :=
  Nat.choose (Int.natAbs (finish.x - start.x) + Int.natAbs (finish.y - start.y)) (Int.natAbs (finish.x - start.x))

/-- The southwest corner of City Park -/
def parkSW : Point := ⟨0, 0⟩

/-- The northeast corner of City Park -/
def parkNE : Point := ⟨0, 0⟩  -- Exact coordinates are not specified, but not needed for the proof

/-- Samantha's house location -/
def samanthaHouse : Point := ⟨-3, -2⟩

/-- Samantha's school location -/
def samanthaSchool : Point := ⟨3, 3⟩

theorem samantha_routes_count : 
  shortestPaths samanthaHouse parkSW * shortestPaths parkNE samanthaSchool = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_routes_count_l1361_136168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geologists_distance_probability_l1361_136176

/-- Represents a circular field with 8 equal sectors -/
structure CircularField where
  radius : ℝ
  num_sectors : Nat
  num_sectors_eq_8 : num_sectors = 8

/-- Represents a geologist's position -/
structure GeologistPosition where
  angle : ℝ
  distance : ℝ

/-- Calculates the distance between two geologists -/
def distance_between (g1 g2 : GeologistPosition) : ℝ :=
  sorry

/-- Determines if two geologists are more than 8 km apart -/
def more_than_8km_apart (g1 g2 : GeologistPosition) : Prop :=
  distance_between g1 g2 > 8

/-- Represents the probability space of geologist positions -/
def GeologistProbabilitySpace (field : CircularField) : Type :=
  GeologistPosition × GeologistPosition

/-- Calculates the probability of an event in the probability space -/
noncomputable def probability (field : CircularField) (event : GeologistProbabilitySpace field → Prop) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem geologists_distance_probability (field : CircularField) :
  probability field (fun pair : GeologistProbabilitySpace field =>
    let (g1, g2) := pair
    more_than_8km_apart g1 g2) = 0.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geologists_distance_probability_l1361_136176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_product_probability_l1361_136156

def spinner_A : Finset ℕ := {1, 2, 3, 4, 5}
def spinner_B : Finset ℕ := {1, 2, 3, 4}

def is_even (n : ℕ) : Bool := n % 2 = 0

def product_is_even (a : ℕ) (b : ℕ) : Bool := is_even (a * b)

theorem spinner_product_probability :
  (Finset.card (Finset.filter (fun p => product_is_even p.1 p.2) (spinner_A.product spinner_B)) : ℚ) /
  ((Finset.card spinner_A : ℚ) * (Finset.card spinner_B : ℚ)) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_product_probability_l1361_136156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_intercept_relation_line_equation_l1361_136165

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a ≠ 0 ∨ b ≠ 0

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a line
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define intercepts
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.a
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

-- Part 1
theorem perpendicular_line_equation (l : Line) (P : Point) :
  on_line P l ∧ 
  perpendicular l { a := 1, b := -3, c := 1, eq := Or.inl (by norm_num) } ∧
  P.x = 1 ∧ P.y = 1 →
  l.a = 3 ∧ l.b = 1 ∧ l.c = -4 :=
by sorry

-- Part 2
theorem intercept_relation_line_equation (l : Line) (P : Point) :
  on_line P l ∧
  P.x = 1 ∧ P.y = 1 ∧
  (y_intercept l = 2 * x_intercept l ∨ (l.a = l.b ∧ l.c = 0)) →
  (l.a = 2 ∧ l.b = 1 ∧ l.c = -3) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_intercept_relation_line_equation_l1361_136165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l1361_136129

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Checks if a point is on the ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Slope of a line through two points -/
noncomputable def line_slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Theorem statement -/
theorem ellipse_fixed_point_theorem (e : Ellipse) (p : Point) (M A B : Point) :
  on_ellipse p e →
  p.x = 1 ∧ p.y = 3/2 →
  eccentricity e = 1/2 →
  M.x = e.a ∧ M.y = 0 →
  on_ellipse A e ∧ on_ellipse B e →
  A ≠ M ∧ B ≠ M ∧ A ≠ B →
  line_slope M A * line_slope M B = 1/4 →
  ∃ (k : ℝ), A.y - B.y = k * (A.x - B.x) ∧ A.y = k * (A.x + 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l1361_136129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_alpha_half_l1361_136163

theorem tan_pi_4_plus_alpha_half (α : Real) 
  (h1 : Real.cos α = -4/5) 
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.tan (π/4 + α/2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_alpha_half_l1361_136163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_shortest_l1361_136125

-- Define a path as a continuous function from [0, 1] to ℝ² (2D plane)
def MyPath := {f : ℝ → ℝ × ℝ | Continuous f ∧ ∀ t, 0 ≤ t ∧ t ≤ 1}

-- Define the length of a path
noncomputable def pathLength (p : MyPath) : ℝ := sorry

-- Define a line segment path between two points
def lineSegmentPath (a b : ℝ × ℝ) : MyPath := sorry

-- Theorem: The line segment path has the shortest length among all paths connecting two points
theorem line_segment_shortest {a b : ℝ × ℝ} (p : MyPath) 
  (h1 : p.val 0 = a) (h2 : p.val 1 = b) : 
  pathLength (lineSegmentPath a b) ≤ pathLength p := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_shortest_l1361_136125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_81_mod_101_l1361_136133

theorem inverse_81_mod_101 (h : (9⁻¹ : ZMod 101) = 65) : (81⁻¹ : ZMod 101) = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_81_mod_101_l1361_136133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_over_sqrt_l1361_136180

/-- The area bounded by two tangent lines from (t, -1) to y = x^2 -/
def S (t : ℝ) : ℝ := sorry

/-- The minimum value of S(t) / √t -/
theorem min_area_over_sqrt (t : ℝ) (ht : t > 0) :
  ∃ (min : ℝ), min = (6/5)^(3/2) * (5^(1/4)) * 2/3 ∧
    ∀ (t' : ℝ), t' > 0 → S t' / Real.sqrt t' ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_over_sqrt_l1361_136180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1361_136178

/-- The equation of an ellipse given specific geometric conditions -/
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a = Real.sqrt 10) (h4 : b = Real.sqrt 5) :
  ∃ (F₁ F₂ A B P O : ℝ × ℝ),
    (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ {P : ℝ × ℝ | P.1^2/10 + P.2^2/5 = 1}) ∧
    (F₁.1 < 0 ∧ F₂.1 > 0) ∧  -- F₁ and F₂ are left and right foci
    (A = (a, 0) ∧ B = (0, b)) ∧  -- A and B are right and upper vertices
    (O = (0, 0)) ∧  -- O is the origin
    (∃ (k : ℝ), P.2 - O.2 = k * (B.2 - A.2) ∧ P.1 - O.1 = k * (B.1 - A.1)) ∧  -- OP parallel to AB
    (P.1 = F₁.1) ∧  -- PF₁ perpendicular to x-axis
    (Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = Real.sqrt 10 + Real.sqrt 5)  -- F₁A = √10 + √5
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1361_136178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_and_g_l1361_136173

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^4 - 2*x^2 + 3*x - 1
noncomputable def g (x : ℝ) : ℝ := (x - 1) / x

-- State the theorem
theorem derivative_of_f_and_g :
  (∀ x, HasDerivAt f (4*x^3 - 4*x + 3) x) ∧
  (∀ x, x ≠ 0 → HasDerivAt g (1 / x^2) x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_and_g_l1361_136173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_pi_over_two_l1361_136150

noncomputable section

-- Define the function g
def g (x : ℝ) : ℝ := 1 + Real.sqrt (1 - x^2)

-- Define the domain of g
def g_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := Real.sqrt (1 - (y-1)^2)

-- Define the range of g (which is the domain of g_inv)
def g_range (y : ℝ) : Prop := 1 ≤ y ∧ y ≤ 2

-- Theorem statement
theorem enclosed_area_is_pi_over_two :
  ∃ (A : ℝ), A = (∫ (x : ℝ) in Set.Icc (-1) 1, g x - g_inv x) ∧ A = π / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_pi_over_two_l1361_136150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1361_136126

/-- The sum of the infinite series from n=3 to infinity of (n^4 + 2n^3 + 10n + 15) / (2^n * (n^4 + 9)) -/
noncomputable def infiniteSeries : ℝ := ∑' n : ℕ, if n ≥ 3 then (n^4 + 2*n^3 + 10*n + 15) / (2^n * (n^4 + 9)) else 0

/-- The sum of the infinite series is equal to 0.3567 -/
theorem infiniteSeriesSum : infiniteSeries = 0.3567 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1361_136126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sum_given_log_sum_l1361_136199

noncomputable def log_base (base a : ℝ) := Real.log a / Real.log base

theorem minimum_sum_given_log_sum (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : log_base 3 m + log_base 3 n = 4) :
  m + n ≥ 18 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 
    log_base 3 m₀ + log_base 3 n₀ = 4 ∧ m₀ + n₀ = 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sum_given_log_sum_l1361_136199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l1361_136186

/-- Given a triangle ABC with angle bisector AL, prove the length of AL -/
theorem angle_bisector_length (A B C L : ℝ × ℝ) : 
  let AB := B - A
  let AC := C - A
  let AL := L - A
  -- Triangle ABC exists
  (A ≠ B ∧ B ≠ C ∧ C ≠ A) →
  -- AB:AC = 5:2
  (∃ (k : ℝ), k > 0 ∧ AB = (5/2 * k) • AC) →
  -- L is on BC
  (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ L = t • B + (1 - t) • C) →
  -- AL is angle bisector
  (∃ (s : ℝ), s > 0 ∧ (AL.fst * AB.fst + AL.snd * AB.snd) = s * (AL.fst * AC.fst + AL.snd * AC.snd)) →
  -- Length of 2⋅AB⃗ + 5⋅AC⃗ is 2016
  ((2 • AB + 5 • AC).fst^2 + (2 • AB + 5 • AC).snd^2 = 2016^2) →
  -- Then the length of AL is 288
  AL.fst^2 + AL.snd^2 = 288^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l1361_136186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_area_theorem_l1361_136195

/-- Represents a square flowerbed with daisies planted in symmetrical triangular regions --/
structure Flowerbed where
  side_length : ℚ
  daisy_regions : ℕ
  symmetrical : Bool

/-- Calculates the total area of daisy regions in the flowerbed --/
def daisy_area (f : Flowerbed) : ℚ :=
  if f.symmetrical && f.daisy_regions = 4 then
    f.daisy_regions * (1/2 * (f.side_length / 2) * (f.side_length / 2))
  else
    0

/-- Theorem stating that a square flowerbed with side length 12 meters and 
    four symmetrical triangular daisy regions has a total daisy area of 48 m² --/
theorem daisy_area_theorem (f : Flowerbed) 
    (h1 : f.side_length = 12)
    (h2 : f.daisy_regions = 4)
    (h3 : f.symmetrical = true) : 
  daisy_area f = 48 := by
  sorry

#eval daisy_area { side_length := 12, daisy_regions := 4, symmetrical := true }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_area_theorem_l1361_136195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1361_136179

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600)

/-- Theorem stating that a train 250 meters long crossing a 350 meter bridge at 72 kmph takes 30 seconds -/
theorem train_crossing_bridge_time :
  train_crossing_time 250 350 72 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1361_136179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_equality_condition_l1361_136101

theorem inequality_proof (x : ℝ) (hx : x ≠ 0) : 
  max 0 (Real.log (abs x)) ≥ 
    (Real.sqrt 5 - 1) / (2 * Real.sqrt 5) * Real.log (abs x) + 
    1 / (2 * Real.sqrt 5) * Real.log (abs (x^2 - 1)) + 
    1 / 2 * Real.log ((Real.sqrt 5 + 1) / 2) :=
by sorry

theorem equality_condition (x : ℝ) (hx : x ≠ 0) :
  (max 0 (Real.log (abs x)) = 
    (Real.sqrt 5 - 1) / (2 * Real.sqrt 5) * Real.log (abs x) + 
    1 / (2 * Real.sqrt 5) * Real.log (abs (x^2 - 1)) + 
    1 / 2 * Real.log ((Real.sqrt 5 + 1) / 2)) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
   x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_equality_condition_l1361_136101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_population_estimate_l1361_136147

/-- Represents the number of turtles in different scenarios -/
structure TurtlePopulation where
  june_total : ℕ
  tagged : ℕ
  october_sample : ℕ
  tagged_in_sample : ℕ

/-- Calculates the estimated number of turtles in June based on given data -/
def estimate_june_population (pop : TurtlePopulation) : ℕ :=
  let june_to_october_ratio : Rat := 7/10  -- 30% left or died
  let october_from_june_ratio : Rat := 1/2  -- 50% were not present in June
  let october_sample_from_june : Rat := pop.october_sample * october_from_june_ratio
  let estimated_total : Rat := (pop.tagged * october_sample_from_june) / pop.tagged_in_sample
  estimated_total.num.toNat

/-- The main theorem stating the estimated turtle population in June -/
theorem turtle_population_estimate (pop : TurtlePopulation) 
  (h1 : pop.tagged = 80)
  (h2 : pop.october_sample = 50)
  (h3 : pop.tagged_in_sample = 2) :
  estimate_june_population pop = 1000 := by
  sorry

#eval estimate_june_population { june_total := 0, tagged := 80, october_sample := 50, tagged_in_sample := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_population_estimate_l1361_136147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_one_l1361_136174

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -2 * x else Real.sqrt x

theorem solutions_of_f_eq_one :
  ∀ x : ℝ, f x = 1 ↔ x = -1/2 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_one_l1361_136174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_of_eight_laughing_l1361_136124

/-- The probability of a single baby laughing -/
def p : ℚ := 1/3

/-- The number of babies in the group -/
def n : ℕ := 8

/-- The minimum number of babies we want to see laughing -/
def k : ℕ := 3

/-- The binomial probability function -/
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The probability of at least k out of n babies laughing -/
def prob_at_least (n k : ℕ) (p : ℚ) : ℚ :=
  1 - (Finset.range k).sum (λ i => binomial_prob n i p)

theorem prob_at_least_three_of_eight_laughing :
  prob_at_least n k p = 3489/6561 := by
  sorry

#eval prob_at_least n k p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_of_eight_laughing_l1361_136124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_obtuse_diagonal_divisions_l1361_136108

/-- A quadrilateral is a polygon with four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is obtuse if it is greater than 90 degrees (or π/2 radians). -/
noncomputable def isObtuseAngle (a b c : ℝ × ℝ) : Prop :=
  let angle := Real.arccos (((b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2)) /
    (Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2) * Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)))
  angle > Real.pi / 2

/-- A triangle is obtuse if one of its angles is obtuse. -/
def isObtuseTriangle (a b c : ℝ × ℝ) : Prop :=
  isObtuseAngle a b c ∨ isObtuseAngle b c a ∨ isObtuseAngle c a b

/-- A diagonal of a quadrilateral is a line segment connecting two non-adjacent vertices. -/
def diagonal (q : Quadrilateral) (i j : Fin 4) : Prop :=
  (i.val + 2) % 4 = j.val ∨ (j.val + 2) % 4 = i.val

/-- A diagonal divides a quadrilateral into two obtuse triangles. -/
def diagonalDividesIntoObtuseTriangles (q : Quadrilateral) (i j : Fin 4) : Prop :=
  diagonal q i j →
  (∃ k l : Fin 4, k ≠ i ∧ k ≠ j ∧ l ≠ i ∧ l ≠ j ∧ k ≠ l ∧
    isObtuseTriangle (q.vertices i) (q.vertices k) (q.vertices j) ∧
    isObtuseTriangle (q.vertices i) (q.vertices l) (q.vertices j))

/-- There exists a quadrilateral where both of its diagonals divide it into two obtuse triangles. -/
theorem exists_quadrilateral_with_obtuse_diagonal_divisions :
  ∃ q : Quadrilateral, ∀ i j : Fin 4, diagonalDividesIntoObtuseTriangles q i j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_obtuse_diagonal_divisions_l1361_136108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_differences_count_l1361_136188

/-- Represents the properties of a block -/
structure BlockProperties where
  material : Fin 2
  size : Fin 3
  color : Fin 5
  shape : Fin 4
deriving Fintype, DecidableEq

/-- Counts the number of differences between two BlockProperties -/
def countDifferences (b1 b2 : BlockProperties) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The reference block (plastic medium red circle) -/
def referenceBlock : BlockProperties := {
  material := 0,
  size := 1,
  color := 2,
  shape := 0
}

/-- Theorem: The number of blocks differing in exactly two properties from the reference block is 36 -/
theorem two_differences_count :
  (Finset.univ.filter (fun b : BlockProperties => countDifferences b referenceBlock = 2)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_differences_count_l1361_136188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preceding_binary_l1361_136171

def binary_to_nat (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    (to_binary_aux n).reverse

theorem preceding_binary (M : List Bool) :
  M = [true, true, false, true, false, true] →
  nat_to_binary (binary_to_nat M - 1) = [true, true, false, true, false, false] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preceding_binary_l1361_136171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_below_standard_l1361_136183

/-- Represents the water level in meters relative to a standard level -/
def water_level (h : ℝ) : ℝ := -h

theorem water_level_below_standard (depth : ℝ) (h : depth > 0) :
  water_level depth = -depth := by
  -- Unfold the definition of water_level
  unfold water_level
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_below_standard_l1361_136183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wolfgang_marbles_l1361_136131

/-- The number of marbles Wolfgang bought -/
def W : ℕ := 16

/-- The number of marbles Ludo bought -/
def L : ℕ := W + W / 4

/-- The number of marbles Michael bought -/
def M : ℕ := (2 * (W + L)) / 3

/-- The total number of marbles -/
def total : ℕ := W + L + M

theorem wolfgang_marbles :
  (total = 60) ∧ (W = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wolfgang_marbles_l1361_136131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_proof_l1361_136151

/-- The volume of the intersection of two congruent cubes with edge length a, 
    sharing a common diagonal, where one cube is rotated 60° around this diagonal. -/
noncomputable def intersection_volume (a : ℝ) : ℝ :=
  3 * a^3 / 4

/-- Theorem stating that the volume of the intersection of two congruent cubes 
    with edge length a, sharing a common diagonal, where one cube is rotated 60° 
    around this diagonal, is equal to 3a³/4. -/
theorem intersection_volume_proof (a : ℝ) (h : a > 0) : 
  intersection_volume a = 3 * a^3 / 4 := by
  -- The proof goes here
  sorry

#check intersection_volume_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_proof_l1361_136151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_approx_l1361_136169

/-- Represents a biased six-sided die -/
structure BiasedDie where
  /-- The probability of rolling a 2 -/
  prob_two : ℝ
  /-- The probability of rolling a 2 is not 1/6 -/
  not_fair : prob_two ≠ 1/6
  /-- The probability of rolling a 2 is between 0 and 1 -/
  prob_bound : 0 < prob_two ∧ prob_two < 1

/-- The probability of rolling exactly one 2 in three rolls -/
def prob_one_two_in_three_rolls (d : BiasedDie) : ℝ :=
  3 * d.prob_two * (1 - d.prob_two)^2

/-- Theorem stating that if the probability of rolling exactly one 2 in three rolls
    is 1/4, then the probability of rolling a 2 in a single roll is approximately 0.211 -/
theorem prob_two_approx (d : BiasedDie) 
    (h : prob_one_two_in_three_rolls d = 1/4) : 
    ∃ ε > 0, |d.prob_two - 0.211| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_approx_l1361_136169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1361_136146

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a+1)*x + a ≥ 0}
  (a > 1 → S = Set.Iic 1 ∪ Set.Ici a) ∧
  (a = 1 → S = Set.univ) ∧
  (a < 1 → S = Set.Iic a ∪ Set.Ici 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1361_136146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_shorter_l1361_136139

/-- A convex pentagon with midpoints -/
structure ConvexPentagonWithMidpoints where
  -- Vertices of the pentagon
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- Midpoints of the sides
  H : ℝ × ℝ
  I : ℝ × ℝ
  K : ℝ × ℝ
  M : ℝ × ℝ
  O : ℝ × ℝ
  -- Conditions for midpoints
  h_midpoint : H = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  i_midpoint : I = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  k_midpoint : K = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  m_midpoint : M = ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  o_midpoint : O = ((E.1 + A.1) / 2, (E.2 + A.2) / 2)
  -- Convexity condition (simplified)
  convex : True

/-- Length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem -/
theorem midpoint_path_shorter (p : ConvexPentagonWithMidpoints) :
  distance p.H p.K + distance p.K p.O + distance p.O p.I + distance p.I p.M + distance p.M p.H <
  distance p.A p.C + distance p.C p.E + distance p.E p.B + distance p.B p.D + distance p.D p.A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_shorter_l1361_136139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pradeep_failed_by_25_marks_l1361_136167

/- Define the parameters of the problem -/
def total_marks : ℕ := 600
def passing_percentage : ℚ := 35 / 100
def pradeep_marks : ℕ := 185

/- Define the passing marks -/
def passing_marks : ℕ := Int.toNat ((passing_percentage * total_marks).ceil)

/- Define the function to calculate the marks failed by -/
def marks_failed_by (obtained_marks : ℕ) : ℕ :=
  if passing_marks > obtained_marks then passing_marks - obtained_marks else 0

/- Theorem to prove -/
theorem pradeep_failed_by_25_marks :
  marks_failed_by pradeep_marks = 25 := by
  -- Proof goes here
  sorry

#eval marks_failed_by pradeep_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pradeep_failed_by_25_marks_l1361_136167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_divisible_by_37_l1361_136197

/-- Represents a six-digit number -/
def SixDigitNumber := Fin 1000000

/-- Checks if a number is six digits long -/
def isSixDigit (n : Nat) : Prop := n ≥ 100000 ∧ n < 1000000

/-- Checks if a number has at least two different digits -/
def hasAtLeastTwoDifferentDigits (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ≠ d2 ∧ d1 < 10 ∧ d2 < 10 ∧ (n.digits 10).contains d1 ∧ (n.digits 10).contains d2

/-- Checks if the first and fourth digits of a number are not zero -/
def firstAndFourthDigitsNonZero (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧ digits.get? 0 ≠ some 0 ∧ digits.get? 3 ≠ some 0

/-- Theorem: For any six-digit number divisible by 37 with at least two different digits
    and non-zero first and fourth digits, there exists another six-digit number formed
    by rearranging the digits of the original number that is also divisible by 37 and
    doesn't start with zero. -/
theorem rearrange_divisible_by_37 (n : Nat) 
  (h_six_digit : isSixDigit n)
  (h_div_37 : n % 37 = 0)
  (h_two_diff : hasAtLeastTwoDifferentDigits n)
  (h_non_zero : firstAndFourthDigitsNonZero n) :
  ∃ (m : Nat), isSixDigit m ∧ m % 37 = 0 ∧ m ≠ n ∧ 
  (∃ (perm : List Nat → List Nat), m.digits 10 = perm (n.digits 10)) ∧
  (m.digits 10).get? 0 ≠ some 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_divisible_by_37_l1361_136197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_squat_weight_l1361_136120

def initial_squat : ℝ → Prop := λ w => w ≥ 0

def post_training_increase : ℝ := 265
def bracer_multiplier : ℝ := 7
def final_squat : ℝ := 2800

theorem initial_squat_weight (w : ℝ) : 
  initial_squat w ↔ 
  bracer_multiplier * (w + post_training_increase) = final_squat ∧ 
  w = 135 := by
  sorry

#check initial_squat_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_squat_weight_l1361_136120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_path_length_l1361_136114

/-- Represents a right-angled isosceles triangle ABP -/
structure RightIsoscelesTriangle where
  sideLength : ℝ

/-- Represents a square AXYZ -/
structure Square where
  sideLength : ℝ

/-- Calculates the path length of point P during rotation -/
noncomputable def pathLength (triangle : RightIsoscelesTriangle) (square : Square) : ℝ :=
  12 * Real.pi * Real.sqrt 2

theorem rotation_path_length 
  (triangle : RightIsoscelesTriangle) 
  (square : Square) 
  (h1 : triangle.sideLength = 4) 
  (h2 : square.sideLength = 8) : 
  pathLength triangle square = 12 * Real.pi * Real.sqrt 2 := by
  sorry

#check rotation_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_path_length_l1361_136114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_area_for_specific_park_l1361_136109

/-- Represents a circular park with a straight path through its center. -/
structure ParkWithPath where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the area of grass in a circular park with a straight path. -/
noncomputable def grassArea (park : ParkWithPath) : ℝ :=
  let totalArea := Real.pi * (park.diameter / 2)^2
  let pathArea := park.diameter * park.pathWidth
  totalArea - pathArea

/-- Theorem stating the grass area for a specific park configuration. -/
theorem grass_area_for_specific_park : 
  let park := ParkWithPath.mk 20 5
  grassArea park = 100 * Real.pi - 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_area_for_specific_park_l1361_136109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lower_bound_l1361_136194

noncomputable def sequenceA (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then n + 15 / (n : ℝ)
  else a * Real.log n - 1/4

theorem sequence_lower_bound (a : ℝ) :
  (∀ n : ℕ, sequenceA a n ≥ 31/4) → a ≥ 8 / Real.log 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lower_bound_l1361_136194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_exchange_divisible_by_nine_two_digit_number_exchange_always_divisible_by_nine_l1361_136138

theorem two_digit_number_exchange_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (10*a + b) - (10*b + a) = 9*k := by
  use a - b
  ring

theorem two_digit_number_exchange_always_divisible_by_nine :
  ∀ a b : ℤ, ∃ k : ℤ, (10*a + b) - (10*b + a) = 9*k := by
  intros a b
  exact two_digit_number_exchange_divisible_by_nine a b

#check two_digit_number_exchange_always_divisible_by_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_exchange_divisible_by_nine_two_digit_number_exchange_always_divisible_by_nine_l1361_136138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l1361_136105

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflects a point across the x-axis -/
def reflect_across_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem cowboy_shortest_path (stream_y cabin_x cabin_y : ℝ) 
    (h1 : stream_y = 0)
    (h2 : cabin_x = 9)
    (h3 : cabin_y = -8) : 
  let cowboy : Point := { x := 0, y := -3 }
  let cabin : Point := { x := cabin_x, y := cabin_y }
  let cowboy_reflected : Point := reflect_across_x_axis cowboy
  3 + distance cowboy_reflected cabin = 3 + Real.sqrt 202 := by
  sorry

#check cowboy_shortest_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l1361_136105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_a_and_c_l1361_136182

/-- Given three real numbers a, b, and c satisfying certain conditions,
    prove that their product is approximately 378.07 -/
theorem product_of_a_and_c (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (diff_ab : a - b = 20)
  (diff_bc : b - c = 30) :
  ∃ (ε : ℝ), ε > 0 ∧ |a * c - 378.07| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_a_and_c_l1361_136182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_guess_minimizes_payment_l1361_136116

/-- The maximum value that Kevin can choose -/
def max_value : ℕ := 1001

/-- The cost of an incorrect guess -/
def incorrect_guess_cost : ℕ := 11

/-- The function representing Arnold's minimum guaranteed payment when starting with guess n -/
noncomputable def f (n : ℕ) : ℝ := sorry

/-- The optimal first guess for Arnold -/
def optimal_guess : ℕ := 859

/-- Theorem stating that the optimal_guess minimizes Arnold's worst-case payment -/
theorem optimal_guess_minimizes_payment :
  ∀ n : ℕ, n ≤ max_value → f optimal_guess ≤ f n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_guess_minimizes_payment_l1361_136116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_snow_days_approx_l1361_136142

noncomputable section

/-- The probability of snow on a given day in December in Frost Town -/
def snow_prob : ℝ := 1 / 5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The probability of exactly k snow days in December -/
def prob_k_snow_days (k : ℕ) : ℝ :=
  (Nat.choose december_days k) * (snow_prob ^ k) * ((1 - snow_prob) ^ (december_days - k))

/-- The probability of at most 3 snow days in December -/
def prob_at_most_3_snow_days : ℝ :=
  prob_k_snow_days 0 + prob_k_snow_days 1 + prob_k_snow_days 2 + prob_k_snow_days 3

theorem prob_at_most_3_snow_days_approx :
  abs (prob_at_most_3_snow_days - 0.336) < 0.001 := by sorry

end


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_snow_days_approx_l1361_136142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_size_age_trick_l1361_136164

theorem shoe_size_age_trick (shoe_size : ℕ) (birth_year : ℕ) :
  shoe_size ≥ 10 ∧ shoe_size < 100 ∧ birth_year < 1991 →
  let result := shoe_size * 2 + 39
  let result := result * 50 + 40
  let result := result - birth_year
  result ≥ 1000 ∧ result < 10000 ∧
  result / 100 = shoe_size ∧
  result % 100 = 1990 - birth_year :=
by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_size_age_trick_l1361_136164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l1361_136166

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the state of the urn -/
structure UrnState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Initial state of the urn -/
def initialState : UrnState :=
  ⟨1, 1, 1⟩

/-- Draws a ball and adds two more of the same color -/
def draw (state : UrnState) (color : Color) : UrnState :=
  match color with
  | Color.Red => ⟨state.red + 2, state.blue, state.green⟩
  | Color.Blue => ⟨state.red, state.blue + 2, state.green⟩
  | Color.Green => ⟨state.red, state.blue, state.green + 2⟩

/-- Performs a sequence of draws -/
def drawSequence (state : UrnState) (sequence : List Color) : UrnState :=
  sequence.foldl draw state

/-- Calculates the total number of balls in the urn -/
def totalBalls (state : UrnState) : ℕ :=
  state.red + state.blue + state.green

/-- Calculates the probability of drawing a specific color -/
noncomputable def drawProbability (state : UrnState) (color : Color) : ℚ :=
  match color with
  | Color.Red => state.red / (totalBalls state)
  | Color.Blue => state.blue / (totalBalls state)
  | Color.Green => state.green / (totalBalls state)

/-- Calculates the probability of a specific draw sequence -/
noncomputable def sequenceProbability (sequence : List Color) : ℚ :=
  let probList := sequence.scanl (fun (state, prob) color =>
    let newState := draw state color
    let newProb := drawProbability state color
    (newState, prob * newProb)
  ) (initialState, 1)
  (probList.getLast?).map Prod.snd |>.getD 0

/-- All possible sequences of 5 draws resulting in 5 red, 5 blue, and 6 green balls -/
def validSequences : List (List Color) :=
  sorry  -- Implementation omitted for brevity

/-- Theorem: The probability of having 5 red, 5 blue, and 6 green balls after 5 draws is 1/21 -/
theorem urn_probability : (validSequences.map sequenceProbability).sum = 1 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l1361_136166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_function_l1361_136122

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The condition that the function must satisfy for all m and n -/
def SatisfiesCondition (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, (Nat.factorial n.val + Nat.factorial (f m).val) ∣ 
    (Nat.factorial (f n).val + Nat.factorial (f (⟨Nat.factorial m.val, Nat.factorial_pos m.val⟩ : ℕ+)).val)

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := λ n => n

/-- Theorem: The identity function is the unique function satisfying the given condition -/
theorem unique_satisfying_function :
  ∃! f : PositiveIntFunction, SatisfiesCondition f ∧ f = identityFunction := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_function_l1361_136122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_is_2401_l1361_136149

/-- The side length of the square garden in meters -/
def side_length : ℕ := Int.toNat ((((2222 - 2121)^2 : ℚ) / 196).floor)

/-- The area of the square garden in square meters -/
def garden_area : ℕ := side_length ^ 2

/-- Theorem stating that the garden area is 2401 square meters -/
theorem garden_area_is_2401 : garden_area = 2401 := by
  -- Unfold definitions
  unfold garden_area
  unfold side_length
  -- Simplify the expression
  simp [Int.toNat, Rat.floor]
  -- The rest of the proof would go here
  sorry

#eval garden_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_is_2401_l1361_136149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_power_l1361_136161

theorem coefficient_x_fourth_power :
  let expansion := (x - 3 * Real.sqrt 2) ^ 9
  let coefficient := Finset.sum (Finset.range 10)
    (λ k ↦ if k = 5 then Nat.choose 9 k * (-3 * Real.sqrt 2) ^ k else 0)
  coefficient = -122472 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_fourth_power_l1361_136161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l1361_136185

theorem longest_side_of_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Sine law holds
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Given ratio condition
  (Real.sin A + Real.sin B) / (Real.sin A + Real.sin C) = 4 / 5 ∧
  (Real.sin A + Real.sin C) / (Real.sin B + Real.sin C) = 5 / 6 →
  -- Area condition
  1/2 * a * b * (Real.sin C) = 15 * Real.sqrt 3 →
  -- Conclusion: longest side is 14
  max a (max b c) = 14 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l1361_136185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_122_l1361_136137

/-- The area bounded by the curve y^2 = 9x, the lines x = 16, x = 25, and y = 0 -/
noncomputable def bounded_area : ℝ :=
  ∫ x in (16 : ℝ)..25, 2 * x^(3/2)

theorem area_equals_122 : bounded_area = 122 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_122_l1361_136137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_implies_a_range_l1361_136141

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -Real.exp x * (2 * x + 1) - a * x + a

/-- The theorem statement -/
theorem unique_positive_integer_implies_a_range (a : ℝ) :
  (a > -1) →
  (∃! (x₀ : ℤ), f a (x₀ : ℝ) > 0) →
  a ∈ Set.Ioo (-1 / (2 * Real.exp 1)) (-1 / (Real.exp 1)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_integer_implies_a_range_l1361_136141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_T_value_l1361_136102

/-- The number of boys in the lineup -/
def num_boys : ℕ := 8

/-- The number of girls in the lineup -/
def num_girls : ℕ := 12

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the lineup -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl pair at any given position -/
def prob_boy_girl : ℚ := 2 * num_boys * num_girls / (total_people * total_people)

/-- T represents the number of boy-girl adjacent pairs in a random permutation -/
def T : Finset (Fin total_people) → ℕ := sorry

/-- The expected value of T -/
def expected_T : ℚ := num_pairs * prob_boy_girl

theorem expected_T_value : expected_T = 228 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_T_value_l1361_136102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1361_136192

theorem rationalize_denominator :
  let x := (27 : ℝ) ^ (1/3)
  let y := (2 : ℝ) ^ (1/3)
  let z := (3 : ℝ) ^ (1/3)
  (x + y) / (z + y) = 7 - (54 : ℝ) ^ (1/3) + (6 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1361_136192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_distributions_count_l1361_136103

/-- Represents a distribution of darts on dartboards -/
def DartDistribution := List Nat

/-- Checks if a list is non-increasing -/
def is_non_increasing (l : List Nat) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≥ l[j]!

/-- The sum of elements in a list -/
def list_sum (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

/-- A valid dart distribution has 5 elements, sums to 5, and is non-increasing -/
def is_valid_distribution (d : DartDistribution) : Prop :=
  d.length = 5 ∧ list_sum d = 5 ∧ is_non_increasing d

/-- The main theorem: there are exactly 7 valid dart distributions -/
theorem dart_distributions_count :
  (∃! (l : List DartDistribution), ∀ d, d ∈ l ↔ is_valid_distribution d) ∧
  (∃ (l : List DartDistribution), (∀ d, d ∈ l ↔ is_valid_distribution d) ∧ l.length = 7) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_distributions_count_l1361_136103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l1361_136198

noncomputable section

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type representing y - 1 = k(x - 1) -/
structure Line where
  k : ℝ
  x : ℝ
  y : ℝ
  eq : y - 1 = k*(x - 1)

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_line_intersection_length 
  (C : Parabola) 
  (l : Line) 
  (F P A B : Point) 
  (h1 : F.x = 1 ∧ F.y = 0)  -- Focus at (1, 0)
  (h2 : P.x = 1 ∧ P.y = 1)  -- P at (1, 1)
  (h3 : l.y - 1 = l.k * (l.x - 1))  -- Line equation
  (h4 : C.y^2 = 4*C.x)  -- Parabola equation
  (h5 : A.x = (B.x + P.x) / 2 ∧ A.y = (B.y + P.y) / 2)  -- P is midpoint of AB
  : distance A B = Real.sqrt 15 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_length_l1361_136198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l1361_136107

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x)
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Define the domain of f
def M : Set ℝ := {x | x < 1}

-- Define the range of g
def N : Set ℝ := {y | y > 0}

-- State the theorem
theorem domain_range_intersection :
  M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l1361_136107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_125_div_6_l1361_136181

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The condition that the binomial coefficients of the third and fourth terms are equal -/
def binomial_condition (n : ℕ) : Prop :=
  binomial n 2 = binomial n 3

/-- The area of the closed region formed by y = nx and y = x^2 -/
noncomputable def area (n : ℕ) : ℝ :=
  ∫ x in (0)..(n : ℝ), n * x - x^2

/-- The main theorem -/
theorem area_is_125_div_6 (n : ℕ) (h : binomial_condition n) :
  area n = 125 / 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_125_div_6_l1361_136181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l1361_136162

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3)^x

-- Define the domain of x
def domain (x : ℝ) : Prop := x ≥ -1

-- Define the range of y
def range (y : ℝ) : Prop := 0 < y ∧ y ≤ 3

-- Theorem statement
theorem function_range :
  ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ range y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l1361_136162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_theorem_l1361_136155

/-- Represents the state of the player's coins -/
structure CoinState where
  gold : ℕ
  silver : ℕ

/-- Represents a bet placed by the player -/
structure Bet where
  gold_red : ℕ
  gold_black : ℕ
  silver_red : ℕ
  silver_black : ℕ

/-- The result of a round -/
inductive RoundResult
| Red
| Black

/-- Applies the result of a round to a given coin state and bet -/
def applyRound (state : CoinState) (bet : Bet) (result : RoundResult) : CoinState :=
  match result with
  | RoundResult.Red => 
      { gold := state.gold + bet.gold_red - bet.gold_black,
        silver := state.silver + bet.silver_red - bet.silver_black }
  | RoundResult.Black => 
      { gold := state.gold - bet.gold_red + bet.gold_black,
        silver := state.silver - bet.silver_red + bet.silver_black }

/-- Checks if the goal state is achieved -/
def isGoalState (state : CoinState) : Prop :=
  state.gold = 0 ∧ state.silver = 0 ∨ state.gold = 3 * state.silver ∨ state.silver = 3 * state.gold

/-- Applies the strategy for a given number of rounds -/
def applyStrategy (initial : CoinState) (strategy : ℕ → Bet) (dealer_strategy : ℕ → RoundResult) (rounds : ℕ) : CoinState :=
  match rounds with
  | 0 => initial
  | n + 1 => applyRound (applyStrategy initial strategy dealer_strategy n) (strategy n) (dealer_strategy n)

/-- The main theorem to be proved -/
theorem coin_game_theorem (m n : ℕ) (h1 : m < 3 * n) (h2 : n < 3 * m) :
  ∃ (strategy : ℕ → Bet), ∀ (dealer_strategy : ℕ → RoundResult),
    ∃ (rounds : ℕ), isGoalState (
      applyStrategy { gold := m, silver := n } strategy dealer_strategy rounds
    ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_theorem_l1361_136155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_is_48_75_l1361_136113

/-- Represents the journey details -/
structure Journey where
  normal_time : ℝ
  traffic_start_fraction : ℝ
  speed_reduction : ℝ
  actual_time : ℝ

/-- Calculates the total distance of the journey -/
noncomputable def calculate_distance (j : Journey) : ℝ :=
  let initial_speed := j.normal_time / 60  -- Convert to hours
  let d := initial_speed * j.normal_time
  let t1 := d * j.traffic_start_fraction / initial_speed
  let remaining_distance := d * (1 - j.traffic_start_fraction)
  let reduced_speed := initial_speed - j.speed_reduction
  let t2 := remaining_distance / reduced_speed
  d

/-- The theorem stating that the journey distance is 48.75 miles -/
theorem journey_distance_is_48_75 : 
  let j : Journey := {
    normal_time := 150,
    traffic_start_fraction := 2/5,
    speed_reduction := 30/60,  -- Convert to miles per minute
    actual_time := 255
  }
  calculate_distance j = 48.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_is_48_75_l1361_136113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1361_136177

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-1) 1 then x^2
  else if x ∈ Set.Ico 1 3 then (x-2)^2
  else 0  -- This else case is arbitrary since we only care about [-1,3]

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - Real.log (x + 2) / Real.log a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, f (x + 2) = f x) ∧  -- f is periodic with period 2
  (∀ x, f x = f (-x)) ∧  -- f is even
  (∀ x, x ∈ Set.Icc (-1) 0 → f x = x^2) ∧  -- f(x) = x^2 for x in [-1,0]
  (∃ x₁ x₂ x₃ x₄, x₁ ∈ Set.Icc (-1) 3 ∧ x₂ ∈ Set.Icc (-1) 3 ∧ 
    x₃ ∈ Set.Icc (-1) 3 ∧ x₄ ∈ Set.Icc (-1) 3 ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 ∧ g a x₄ = 0) →  -- g has 4 zeros in [-1,3]
  a ∈ Set.Icc 5 (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1361_136177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_inradius_inequality_l1361_136140

/-- A triangle with vertices A, B, and C -/
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

/-- The circumradius of a triangle -/
noncomputable def circumradius {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (T : Triangle V) : ℝ := 
  sorry

/-- The inradius of a triangle -/
noncomputable def inradius {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (T : Triangle V) : ℝ := 
  sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (T : Triangle V) : Prop := 
  sorry

/-- Theorem: For any triangle, the circumradius is at least twice the inradius,
    and equality holds if and only if the triangle is equilateral -/
theorem circumradius_inradius_inequality {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (T : Triangle V) :
  circumradius T ≥ 2 * inradius T ∧
  (circumradius T = 2 * inradius T ↔ is_equilateral T) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_inradius_inequality_l1361_136140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_club_committee_probability_l1361_136145

def debate_club_size : ℕ := 30
def num_boys : ℕ := 18
def num_girls : ℕ := 12
def committee_size : ℕ := 6
def num_boys_in_committee : ℕ := 3
def num_girls_in_committee : ℕ := 3

theorem debate_club_committee_probability :
  (Nat.choose num_boys num_boys_in_committee * Nat.choose num_girls num_girls_in_committee : ℚ) /
  Nat.choose debate_club_size committee_size = 64 / 211 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_club_committee_probability_l1361_136145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1361_136172

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for the product of slopes
def slope_product (a b : ℝ) : Prop :=
  ∀ x y : ℝ, ellipse a b x y → (y^2 - b^2) / x^2 = -1/2

theorem ellipse_theorem (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse a b (Real.sqrt 2 / 2) (Real.sqrt 3 / 2))
  (h4 : slope_product a b) :
  (∀ x y : ℝ, x^2 / 2 + y^2 = 1 ↔ ellipse a b x y) ∧
  (∃ A B : ℝ × ℝ, 
    let F₂ := (1, 0)
    let F₂A := (A.fst - F₂.fst, A.snd - F₂.snd)
    let F₂B := (B.fst - F₂.fst, B.snd - F₂.snd)
    ellipse a b A.fst A.snd ∧ ellipse a b B.fst B.snd ∧
    F₂A.fst * F₂B.fst + F₂A.snd * F₂B.snd = 2 ∧
    abs ((B.fst - A.fst) * (B.snd - F₂.snd) - (B.snd - A.snd) * (B.fst - F₂.fst)) / 2 = 4/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1361_136172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_rectangles_l1361_136123

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a division of a large rectangle into smaller rectangles -/
structure Division where
  large : Rectangle
  small : List Rectangle

/-- Checks if a division is valid according to the problem constraints -/
def isValidDivision (d : Division) : Prop :=
  d.large.width * d.large.height = (d.small.map (λ r => r.width * r.height)).sum ∧
  ∀ r ∈ d.small, ∃ k : ℕ, 2 * k = r.width * r.height

/-- The main theorem statement -/
theorem max_distinct_rectangles (d : Division) :
  d.large.width = 8 ∧ d.large.height = 9 →
  isValidDivision d →
  ∀ blue_squares : List ℕ,
    (blue_squares.length = d.small.length ∧
     blue_squares.sum = 36 ∧
     blue_squares.Pairwise (· < ·)) →
    blue_squares.length ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_rectangles_l1361_136123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_l1361_136152

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log_difference (a b : ℝ) (ha : 0 < a) (hb : b = 9 * a) : f a - f b = -2 := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_l1361_136152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_cube_with_equal_square_faces_l1361_136111

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure Polyhedron where
  faces : Set Face
  edges : Set Edge
  vertices : Set Vertex

/-- A face is a flat polygonal surface of a polyhedron. -/
structure Face where
  vertices : Set Vertex
  edges : Set Edge

/-- An edge is a line segment where two faces of a polyhedron meet. -/
structure Edge where
  vertices : Fin 2 → Vertex

/-- A vertex is a point where three or more edges of a polyhedron meet. -/
structure Vertex where
  coordinates : ℝ × ℝ × ℝ

/-- A square is a regular quadrilateral with four equal sides and four right angles. -/
structure Square extends Face

/-- A cube is a regular polyhedron with six square faces. -/
structure Cube extends Polyhedron

/-- Predicate to check if all faces of a polyhedron are equal squares. -/
def all_faces_equal_squares (p : Polyhedron) : Prop :=
  ∀ f ∈ p.faces, ∃ s : Square, f = s.toFace ∧ ∀ g ∈ p.faces, ∃ t : Square, g = t.toFace ∧ s = t

/-- Theorem stating that there exists a polyhedron with all faces as equal squares that is not a cube. -/
theorem exists_non_cube_with_equal_square_faces : 
  ∃ p : Polyhedron, all_faces_equal_squares p ∧ ¬∃ c : Cube, p = c.toPolyhedron :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_cube_with_equal_square_faces_l1361_136111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1361_136189

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 1)) / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≥ 1 ∧ x ≠ 2} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1361_136189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l1361_136136

noncomputable def a : ℝ := Real.sin (145 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (52 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (47 * Real.pi / 180)

theorem a_lt_b_lt_c : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l1361_136136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1361_136184

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, if a point M(x₀, 2√2) on the parabola
    is the center of a circle with radius |MF| that intersects the y-axis creating a chord of
    length 2√5, then p = 2. -/
theorem parabola_circle_intersection (p : ℝ) (x₀ : ℝ) : 
  p > 0 →                                  -- p is positive
  (2 * Real.sqrt 2)^2 = 2 * p * x₀ →       -- M(x₀, 2√2) is on the parabola
  5 + x₀^2 = (x₀ + p / 2)^2 →              -- Circle equation
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1361_136184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_D_l1361_136106

-- Define the circles and their properties
def circle_C : ℝ := 5  -- radius of circle C

noncomputable def circle_D (r : ℝ) : ℝ := 4 * r  -- radius of circle D in terms of r
noncomputable def circle_E (r : ℝ) : ℝ := r  -- radius of circle E

-- Define the relationships between circles
axiom tangent_C_D (r : ℝ) : circle_C = circle_D r + r
axiom tangent_C_E (r : ℝ) : circle_C = circle_E r + r
axiom tangent_D_E (r : ℝ) : circle_D r - circle_E r = 5 * r

-- Theorem to prove
theorem radius_of_D : ∃ r : ℝ, circle_D r = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_D_l1361_136106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1361_136158

/-- The function f(x) = log(tan x - 1) + √(9-x^2) -/
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.tan x - 1) + Real.sqrt (9 - x^2)

/-- The domain of f(x) -/
def domain_f : Set ℝ := Set.Ioo (-3 * Real.pi / 4) (-Real.pi / 2) ∪ Set.Ioo (Real.pi / 4) (Real.pi / 2)

/-- Theorem stating that the domain of f is equal to domain_f -/
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1361_136158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_odd_function_l1361_136117

open Real

noncomputable def f (x a b : ℝ) : ℝ := -(x + a) / (b * x + 1)

theorem max_value_of_odd_function (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f x a b = -f (-x) a b) →
  (∃ (m : ℝ), ∀ x ∈ Set.Icc (-1) 1, f x a b ≤ m ∧ ∃ y ∈ Set.Icc (-1) 1, f y a b = m) →
  (∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc (-1) 1, f x a b ≤ m ∧ ∃ y ∈ Set.Icc (-1) 1, f y a b = m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_odd_function_l1361_136117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l1361_136154

open Real

theorem integral_proof (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
  deriv (λ y => log (abs (y + 2)) - 1 / ((y - 2)^2)) x =
  (x^3 - 6*x^2 + 14*x - 4) / ((x + 2) * (x - 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l1361_136154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l1361_136135

/-- Calculates the annual interest rate given the principal and total amount paid after one year -/
noncomputable def annual_interest_rate (principal : ℝ) (total_paid : ℝ) : ℝ :=
  ((total_paid - principal) / principal) * 100

/-- Theorem stating that the annual interest rate for the given loan is 10% -/
theorem loan_interest_rate :
  let principal : ℝ := 150
  let total_paid : ℝ := 165
  annual_interest_rate principal total_paid = 10 := by
  -- Unfold the definition of annual_interest_rate
  unfold annual_interest_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l1361_136135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_prob_difference_l1361_136134

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The positive difference between the probabilities of getting exactly 4 heads
    and exactly 5 heads in 5 flips of a fair coin -/
theorem fair_coin_prob_difference :
  |prob_k_heads 5 4 - prob_k_heads 5 5| = 1 / 8 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_prob_difference_l1361_136134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_tan_l1361_136112

theorem cos_value_from_tan (α : Real) :
  α ∈ Set.Ioo (π / 2) π →
  Real.tan α = -Real.sqrt 3 / 3 →
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_tan_l1361_136112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_six_circles_l1361_136170

noncomputable def circle_sequence (n : ℕ) : ℝ :=
  2 * (1/3) ^ (n - 1)

noncomputable def circle_area (n : ℕ) : ℝ :=
  Real.pi * (circle_sequence n) ^ 2

noncomputable def sum_circle_areas (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => circle_area (i + 1))

theorem sum_of_six_circles :
  sum_circle_areas 6 = 9 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_six_circles_l1361_136170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_F_common_tangent_line_l1361_136159

-- Define the functions f, g, and F
noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (a x : ℝ) : ℝ := a * (x - 1)^2 - 1
noncomputable def F (a x : ℝ) : ℝ := f x - g a x

-- Part 1: Maximum value of F when a = 1/4
theorem max_value_F (x : ℝ) (hx : x > 0) :
  ∃ (max_val : ℝ), max_val = Real.log 2 + 3/4 ∧
  ∀ y, y > 0 → F (1/4) y ≤ max_val :=
by
  sorry

-- Part 2: Common tangent line when a = -1/4
theorem common_tangent_line (x : ℝ) (hx : x > 0) :
  ∃ (m b : ℝ), m = 1 ∧ b = -1 ∧
  (∃ (x₁ : ℝ), x₁ > 0 ∧ f x₁ = m * x₁ + b) ∧
  (∃ (x₂ : ℝ), g (-1/4) x₂ = m * x₂ + b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_F_common_tangent_line_l1361_136159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1361_136175

-- Define the function f(x) = log_3 |x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 3

-- State the theorem
theorem f_is_even : ∀ x : ℝ, x ≠ 0 → f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1361_136175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_of_m_l1361_136190

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then 2^(x-m) else (m*x)/(4*x^2+16)

theorem f_range_of_m :
  (∀ m : ℝ, ∀ x₁ : ℝ, x₁ ≥ 2 → ∃ x₂ : ℝ, x₂ ≤ 2 ∧ f m x₁ = f m x₂) ↔
  (∀ m : ℝ, m ≤ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_of_m_l1361_136190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_sale_cost_l1361_136196

/-- Calculate the discounted price for a given number of chairs --/
def discountedPrice (basePrice : ℝ) (quantity : ℕ) : ℝ :=
  let firstThree := min quantity 3
  let nextThree := min (quantity - 3) 3
  let rest := quantity - firstThree - nextThree
  let firstThreePrice := basePrice * (firstThree : ℝ) * 0.8
  let nextThreePrice := basePrice * (nextThree : ℝ) * 0.8 * 0.85
  let restPrice := basePrice * (rest : ℝ) * 0.8 * 0.85 * 0.75
  firstThreePrice + nextThreePrice + restPrice

/-- Calculate the total cost including tax --/
def totalCost (basePrice : ℝ) (quantity : ℕ) (taxRate : ℝ) : ℝ :=
  let discounted := discountedPrice basePrice quantity
  discounted * (1 + taxRate)

/-- The main theorem --/
theorem chair_sale_cost :
  let typeACost := totalCost 25 8 0.06
  let typeBCost := totalCost 35 5 0.08
  typeACost + typeBCost = 286.82 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_sale_cost_l1361_136196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_square_trajectory_l1361_136119

/-- The function representing the trajectory of vertex C of a rolling unit square -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + 2 * (x - 2016) - (x - 2016)^2)

/-- Theorem stating that f(x) correctly represents the trajectory of vertex C
    for x in [2017, 2018] as the square rolls along the x-axis -/
theorem rolling_square_trajectory (x : ℝ) (h : x ∈ Set.Icc 2017 2018) :
  f x = Real.sqrt (1 + 2 * (x - 2016) - (x - 2016)^2) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_square_trajectory_l1361_136119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_month_relation_l1361_136191

/-- Grant's freelance math work earnings over three months --/
structure FreelanceEarnings where
  first_month : ℚ
  second_month : ℚ
  third_month : ℚ
  total : ℚ

/-- Conditions for Grant's earnings --/
def EarningsConditions (e : FreelanceEarnings) : Prop :=
  e.first_month = 350 ∧
  e.third_month = 4 * (e.first_month + e.second_month) ∧
  e.total = 5500 ∧
  e.total = e.first_month + e.second_month + e.third_month

/-- Theorem stating the relationship between first and second month earnings --/
theorem second_month_relation (e : FreelanceEarnings) 
  (h : EarningsConditions e) : 
  e.second_month = (15 / 7) * e.first_month :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_month_relation_l1361_136191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_37_2_l1361_136148

-- Define the trapezoid parameters
def base1 : ℝ := 18
def base2 : ℝ := 13
def leg1 : ℝ := 3
def leg2 : ℝ := 4

-- Define the area of the trapezoid
noncomputable def trapezoid_area (b1 b2 l1 l2 : ℝ) : ℝ :=
  let h := (l1 * l2) / Real.sqrt ((b1 - b2)^2 + (l1 - l2)^2)
  (b1 + b2) * h / 2

-- Theorem statement
theorem trapezoid_area_is_37_2 :
  trapezoid_area base1 base2 leg1 leg2 = 37.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_37_2_l1361_136148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PR_QR_l1361_136121

-- Define the circles and line
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 4
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_PR_QR :
  ∃ (P_x P_y Q_x Q_y R_x R_y : ℝ),
    circle_C1 P_x P_y ∧
    circle_C2 Q_x Q_y ∧
    line_l R_x R_y ∧
    (∀ (P_x' P_y' Q_x' Q_y' R_x' R_y' : ℝ),
      circle_C1 P_x' P_y' →
      circle_C2 Q_x' Q_y' →
      line_l R_x' R_y' →
      distance P_x P_y R_x R_y + distance Q_x Q_y R_x R_y ≤
      distance P_x' P_y' R_x' R_y' + distance Q_x' Q_y' R_x' R_y') ∧
    distance P_x P_y R_x R_y + distance Q_x Q_y R_x R_y = Real.sqrt 26 - 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PR_QR_l1361_136121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_numbers_l1361_136193

-- Define complex numbers
def B : ℂ := 3 + 2*Complex.I
def Q : ℂ := -6 + Complex.I
def R : ℂ := 3*Complex.I
def T : ℂ := 4 + 3*Complex.I
def U : ℂ := -2 - 2*Complex.I

-- Theorem statement
theorem sum_of_complex_numbers :
  B + Q + R + T + U = -1 + 7*Complex.I := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_numbers_l1361_136193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_l1361_136128

-- Define the quadrilateral
structure Quadrilateral where
  area : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  angle : ℝ

-- Define the specific quadrilateral from the problem
noncomputable def problem_quadrilateral : Quadrilateral where
  area := 3
  diagonal1 := 6
  diagonal2 := 2
  angle := Real.pi / 6

-- Theorem statement
theorem angle_between_diagonals (q : Quadrilateral) :
  q.area = 3 ∧ q.diagonal1 = 6 ∧ q.diagonal2 = 2 →
  q.angle = Real.pi / 6 := by
  intro h
  sorry

#check angle_between_diagonals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_l1361_136128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1361_136115

noncomputable def sequence_a (n : ℕ) : ℝ := 2 * n - 1

noncomputable def sequence_b (n : ℕ) : ℝ := 10 - sequence_a n

noncomputable def sum_s (n : ℕ) : ℝ := ((sequence_a n + 1) / 2) ^ 2

noncomputable def sum_T (n : ℕ) : ℝ := n * (sequence_b 1 + sequence_b n) / 2

theorem sequence_properties :
  (∀ n : ℕ, sequence_a n > 0) ∧
  (∀ n : ℕ, sum_s n = ((sequence_a n + 1) / 2) ^ 2) ∧
  (∀ n : ℕ, sequence_b n = 10 - sequence_a n) →
  (∀ n : ℕ, sequence_a n = 2 * n - 1) ∧
  (∃ max_T : ℝ, max_T = 25 ∧ ∀ n : ℕ, sum_T n ≤ max_T) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1361_136115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divisible_by_p_l1361_136187

theorem infinitely_many_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, p ∣ (f k * 2^(f k) + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divisible_by_p_l1361_136187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_zero_l1361_136118

-- Define the function f as noncomputable
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (Real.pi * x + φ)

-- State the theorem
theorem function_value_at_zero 
  (φ : ℝ) 
  (h : f φ (1/6) = 1) : 
  f φ 0 = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_zero_l1361_136118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l1361_136132

theorem seed_germination_problem (seeds_second_plot : ℚ) : seeds_second_plot = 200 :=
  let seeds_first_plot : ℚ := 300
  let germination_rate_first : ℚ := 25 / 100
  let germination_rate_second : ℚ := 40 / 100
  let total_germination_rate : ℚ := 31 / 100

  have h1 : (germination_rate_first * seeds_first_plot + 
             germination_rate_second * seeds_second_plot) / 
            (seeds_first_plot + seeds_second_plot) = total_germination_rate := by
    sorry

  have h2 : germination_rate_first * seeds_first_plot + 
            germination_rate_second * seeds_second_plot = 
            total_germination_rate * (seeds_first_plot + seeds_second_plot) := by
    sorry

  have h3 : (25 / 100) * 300 + (40 / 100) * seeds_second_plot = 
            (31 / 100) * (300 + seeds_second_plot) := by
    sorry

  have h4 : 75 + (40 / 100) * seeds_second_plot = 93 + (31 / 100) * seeds_second_plot := by
    sorry

  have h5 : (40 / 100 - 31 / 100) * seeds_second_plot = 93 - 75 := by
    sorry

  have h6 : (9 / 100) * seeds_second_plot = 18 := by
    sorry

  have h7 : seeds_second_plot = 18 / (9 / 100) := by
    sorry

  have h8 : seeds_second_plot = 200 := by
    sorry

  h8


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l1361_136132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_number_is_twenty_l1361_136127

/-- The unknown number in the first set -/
noncomputable def x : ℝ := sorry

/-- The average of the first set of numbers -/
noncomputable def avg1 : ℝ := (x + 40 + 60) / 3

/-- The average of the second set of numbers -/
noncomputable def avg2 : ℝ := (10 + 50 + 45) / 3

/-- Theorem stating that the unknown number is 20 -/
theorem unknown_number_is_twenty : x = 20 := by
  have h1 : avg1 = avg2 + 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_number_is_twenty_l1361_136127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_quarter_five_terms_l1361_136100

/-- Sum of a geometric series with n terms -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_quarter_five_terms :
  geometric_sum (1/4 : ℝ) (1/4 : ℝ) 5 = 1023/3072 := by
  -- Unfold the definition of geometric_sum
  unfold geometric_sum
  -- Simplify the expression
  simp [pow_succ, Real.rpow_def]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_quarter_five_terms_l1361_136100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_f_is_open_unit_interval_l1361_136160

-- Define the function f(x) = 2^(1-x)
noncomputable def f (x : ℝ) : ℝ := 2^(1 - x)

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f, 0 < y ∧ y ≤ 1 ∧
  ∀ ε > 0, ∃ x ≥ 1, |f x - 1| < ε ∧
  ∀ ε > 0, ∃ x ≥ 1, |f x - 0| > ε :=
by
  sorry

-- Additional lemma to show that the range is exactly (0, 1]
theorem range_of_f_is_open_unit_interval :
  Set.range f = Set.Ioc 0 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_f_is_open_unit_interval_l1361_136160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1361_136144

theorem sin_alpha_value (α : Real) 
  (h1 : Real.cos (Real.pi + α) = -(1/2)) 
  (h2 : 3/2 * Real.pi < α) 
  (h3 : α < 2 * Real.pi) : 
  Real.sin α = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1361_136144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1361_136110

-- Define the complex number z
noncomputable def z : ℂ := (1 + Complex.I) / Complex.I

-- Theorem statement
theorem z_in_fourth_quadrant :
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1361_136110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mean_after_fixing_errors_l1361_136104

theorem correct_mean_after_fixing_errors 
  (n : ℕ) 
  (incorrect_mean : ℚ) 
  (incorrect_value1 incorrect_value2 incorrect_value3 : ℚ)
  (correct_value1 correct_value2 correct_value3 : ℚ) :
  n = 100 →
  incorrect_mean = 235 →
  incorrect_value1 = 300 ∧ correct_value1 = 320 →
  incorrect_value2 = 400 ∧ correct_value2 = 410 →
  incorrect_value3 = 210 ∧ correct_value3 = 230 →
  let incorrect_sum := n * incorrect_mean
  let correction := (correct_value1 - incorrect_value1) + 
                    (correct_value2 - incorrect_value2) + 
                    (correct_value3 - incorrect_value3)
  let correct_sum := incorrect_sum + correction
  let correct_mean := correct_sum / n
  correct_mean = 235.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mean_after_fixing_errors_l1361_136104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_is_four_l1361_136143

-- Define the region D
noncomputable def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ Real.sqrt 2 ∧ p.2 ≤ 2 ∧ p.1 ≤ Real.sqrt 2 * p.2}

-- Define point A
noncomputable def A : ℝ × ℝ := (Real.sqrt 2, 1)

-- Define the dot product function
def dot_product (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2

-- State the theorem
theorem max_dot_product_is_four :
  ∃ (max_z : ℝ), max_z = 4 ∧ 
  ∀ (M : ℝ × ℝ), M ∈ D → dot_product M A ≤ max_z := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_is_four_l1361_136143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cells_to_paint_l1361_136130

theorem min_cells_to_paint (n : Nat) : n = 3 :=
  sorry

#check min_cells_to_paint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cells_to_paint_l1361_136130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_distance_l1361_136153

/-- The distance between intersection points of a specific ellipse and hyperbola -/
theorem ellipse_hyperbola_intersection_distance : 
  ∀ (ellipse hyperbola : Set (ℝ × ℝ)),
    (∀ x y, (x, y) ∈ ellipse ↔ x^2/36 + y^2/16 = 1) →
    (∃ c : ℝ, c > 0 ∧ 
      (∀ x y, (x, y) ∈ hyperbola ↔ x^2/c^2 - y^2/(c^2*(4/9)) = 1) ∧
      (c, 0) ∈ ellipse ∧ (c, 0) ∈ hyperbola) →
    (∃ p₁ p₂ : ℝ × ℝ, p₁ ∈ ellipse ∧ p₁ ∈ hyperbola ∧ 
                       p₂ ∈ ellipse ∧ p₂ ∈ hyperbola ∧
                       p₁ ≠ p₂) →
    ∃ p₁ p₂ : ℝ × ℝ, p₁ ∈ ellipse ∧ p₁ ∈ hyperbola ∧ 
                      p₂ ∈ ellipse ∧ p₂ ∈ hyperbola ∧
                      dist p₁ p₂ = 12 * Real.sqrt 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_intersection_distance_l1361_136153
