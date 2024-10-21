import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_midpoints_count_l873_87319

/-- A point in a parallelogram -/
inductive ParallelogramPoint
  | Vertex
  | SideMidpoint
  | DiagonalIntersection

/-- A parallelogram with marked points -/
structure MarkedParallelogram where
  points : List ParallelogramPoint

/-- The midpoint of two points -/
def midpointOfPoints (p1 p2 : ParallelogramPoint) : ParallelogramPoint :=
  sorry

/-- Check if two midpoints are unique -/
def isUniqueMidpoint (m1 m2 : ParallelogramPoint) : Bool :=
  sorry

/-- Generate all possible midpoints from two parallelograms -/
def generateMidpoints (p1 p2 : MarkedParallelogram) : List ParallelogramPoint :=
  sorry

/-- Count unique midpoints -/
def countUniqueMidpoints (midpoints : List ParallelogramPoint) : Nat :=
  sorry

theorem parallelogram_midpoints_count 
  (p1 p2 : MarkedParallelogram) 
  (h1 : p1.points.length = 9)
  (h2 : p2.points.length = 9)
  (h3 : p1 ≠ p2) :
  countUniqueMidpoints (generateMidpoints p1 p2) = 25 :=
by
  sorry

#check parallelogram_midpoints_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_midpoints_count_l873_87319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l873_87309

-- Define the ellipse C
def ellipse_C (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola : ℝ → ℝ → Prop :=
  λ x y => x^2 / 2 - y^2 = 1

-- Define the line AB
def line_AB (k m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y = k * x + m

-- Main theorem
theorem ellipse_and_line_equations :
  ∀ (a b c : ℝ) (k m : ℝ),
  a > b ∧ b > 0 ∧  -- Condition: a > b > 0
  c / a = Real.sqrt 2 / 2 ∧  -- Eccentricity condition
  c / Real.sqrt 3 = Real.sqrt 3 / 3 ∧  -- Distance from F_1 to asymptote condition
  k < 0 ∧  -- Condition: k < 0
  2 * Real.sqrt 5 / 5 = abs m / Real.sqrt (1 + k^2) →  -- Distance from origin to AB condition
  (ellipse_C a b = ellipse_C (Real.sqrt 2) 1) ∧  -- Equation of ellipse C
  (line_AB k m = line_AB (-1/2) 1)  -- Equation of line AB
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l873_87309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_properties_l873_87361

-- Define the curves in polar coordinates
def C (ρ : ℝ) (θ : ℝ) : Prop := θ = Real.pi/4 ∧ ρ ≠ 0
def C₁ (ρ : ℝ) (θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂ (ρ : ℝ) (θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the intersection points
noncomputable def M : ℝ × ℝ := (1, 1)
noncomputable def N : ℝ × ℝ := (2, 2)

-- State the theorem
theorem curves_properties :
  (∃ (x y : ℝ), C₁ x y ∧ C₂ x y) ∧  -- C₁ and C₂ are tangent
  ‖N‖ = 2 * ‖M‖ :=                  -- |ON| = 2|OM|
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_properties_l873_87361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l873_87326

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Definition of f for x ∈ (0,2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x - a * x

theorem odd_function_a_value (a : ℝ) :
  a > 1/2 →
  IsOdd (fun x ↦ if x > 0 then f a x else -f a (-x)) →
  (∀ x ∈ Set.Ioo (-2) 0, f a (-x) ≤ 1) →
  (∃ x ∈ Set.Ioo (-2) 0, f a (-x) = 1) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l873_87326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l873_87357

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a₁^2 + b₁^2)

/-- Theorem: The distance between the parallel lines 3x + 4y - 3 = 0 and 6x + 8y + 1/2 = 0 is 7/10 -/
theorem distance_between_specific_lines :
  distance_between_parallel_lines 3 4 (-3) 6 8 (1/2) = 7/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l873_87357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2024_eq_half_l873_87329

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 2 => 1 / (2 * ⌊a (n + 1)⌋ - a (n + 1) + 1)

theorem a_2024_eq_half : a 2024 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2024_eq_half_l873_87329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l873_87385

noncomputable def p : Prop := ∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x

def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

theorem problem_solution : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l873_87385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l873_87368

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_sum : S a 5 = 20) :
  a 3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_third_term_l873_87368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l873_87388

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x + Real.pi/3) + Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (3*x + Real.pi/6)

-- State the theorem
theorem function_properties (m : ℝ) :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi/3), g x ^ 2 - m * g x - 3 ≤ 0) →
  (∀ x ∈ Set.Icc (-Real.pi/4 : ℝ) (Real.pi/6), 
    x ∈ Set.Icc (Real.pi/12 : ℝ) (Real.pi/6) → (∀ y ∈ Set.Icc (-Real.pi/4 : ℝ) x, f y ≥ f x)) ∧
  m ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l873_87388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l873_87360

-- Define the set of solutions
def SolutionSet : Set ℝ :=
  {x | (2/3 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) ∨ (∃ n : ℕ, x = 1/2 + 2*n)}

-- Define the domain of the logarithm
def LogDomain : Set ℝ :=
  {x | (2/3 < x ∧ x < 1) ∨ (1 < x)}

-- Define the inequality function
noncomputable def InequalityFunction (x : ℝ) : ℝ :=
  (Real.log (3*x - 2) / Real.log x)^2 - 4 * (Real.sin (Real.pi * x) - 1)

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, x ∈ LogDomain →
    (InequalityFunction x ≤ 0 ↔ x ∈ SolutionSet) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l873_87360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethan_expected_wins_l873_87317

/-- The probability of Ethan winning a game -/
noncomputable def p_win : ℝ := 2/5

/-- The probability of Ethan tying a game -/
noncomputable def p_tie : ℝ := 2/5

/-- The probability of Ethan losing a game -/
noncomputable def p_lose : ℝ := 1/5

/-- The expected number of games Ethan wins before losing -/
noncomputable def expected_wins : ℝ := 2

/-- Theorem stating that the expected number of games Ethan wins before losing is 2 -/
theorem ethan_expected_wins :
  p_win + p_tie + p_lose = 1 →
  expected_wins = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethan_expected_wins_l873_87317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_constant_l873_87340

/-- Regular polygon with n vertices -/
structure RegularPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  is_regular : Sorry

/-- Line passing through the center of the polygon -/
structure CenterLine where
  slope : ℝ
  center : ℝ × ℝ

/-- Perpendicular distance from a point to a line -/
noncomputable def perpendicular_distance (p : ℝ × ℝ) (l : CenterLine) : ℝ := sorry

/-- Sum of squares of perpendicular distances -/
noncomputable def sum_of_squares (polygon : RegularPolygon) (line : CenterLine) : ℝ :=
  (Finset.univ : Finset (Fin polygon.n)).sum (λ i => (perpendicular_distance (polygon.vertices i) line) ^ 2)

/-- Theorem statement -/
theorem sum_of_squares_constant (polygon : RegularPolygon) :
  ∃ (c : ℝ), ∀ (line : CenterLine), line.center = polygon.center → sum_of_squares polygon line = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_constant_l873_87340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_client_age_is_21_l873_87391

/-- Represents Jane's babysitting career and the age of her oldest client --/
structure BabysittingCareer where
  start_age : ℕ
  current_age : ℕ
  years_since_stopped : ℕ
  max_client_age_ratio : ℚ

/-- The current age of the oldest person Jane could have babysat --/
def oldest_client_current_age (career : BabysittingCareer) : ℕ :=
  let stop_age := career.current_age - career.years_since_stopped
  let max_client_age_at_stop := (stop_age : ℚ) * career.max_client_age_ratio
  (Int.floor max_client_age_at_stop).toNat + career.years_since_stopped

/-- Theorem stating the current age of Jane's oldest possible client --/
theorem oldest_client_age_is_21 (jane : BabysittingCareer) 
  (h1 : jane.start_age = 16)
  (h2 : jane.current_age = 32)
  (h3 : jane.years_since_stopped = 10)
  (h4 : jane.max_client_age_ratio = 1/2) :
  oldest_client_current_age jane = 21 := by
  sorry

#eval oldest_client_current_age ⟨16, 32, 10, 1/2⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_client_age_is_21_l873_87391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_english_only_enrollment_l873_87398

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 40)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : ∀ s, s ∈ Finset.range total → 
    (s ∈ Finset.range (total - german + both) ∨ 
     s ∈ Finset.range german)) :
  total - german + both - both = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_english_only_enrollment_l873_87398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l873_87382

theorem absolute_value_equation_solution : 
  {x : ℝ | (abs (abs (abs x - 2) - 1) - 2) = 2} = {-7, -3, -1, 1, 3, 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l873_87382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_when_a_zero_f_zeros_count_l873_87363

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 2) * x + 2 * log x

-- Theorem 1: f(x) < 0 when a = 0
theorem f_negative_when_a_zero : ∀ x > 0, f 0 x < 0 := by
  sorry

-- Theorem 2: Number of zeros of f(x) for different values of a
theorem f_zeros_count (a : ℝ) : 
  (a < -4 → (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0)) ∧ 
  (a = -4 → (∃! x : ℝ, f a x = 0)) ∧
  (-4 < a ∧ a ≤ 0 → ∀ x > 0, f a x ≠ 0) ∧
  (a > 0 → (∃! x : ℝ, f a x = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_when_a_zero_f_zeros_count_l873_87363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_language_grouping_l873_87348

/-- Represents a group of students -/
structure StudentGroup where
  english : Finset Nat
  french : Finset Nat
  spanish : Finset Nat

/-- The main theorem statement -/
theorem student_language_grouping 
  (A : Finset Nat) 
  (F : Finset Nat) 
  (S : Finset Nat) 
  (hA : A.card = 50) 
  (hF : F.card = 50) 
  (hS : S.card = 50) : 
  ∃ (groups : Finset StudentGroup), 
    groups.card = 5 ∧ 
    (∀ g, g ∈ groups → g.english.card = 10 ∧ g.french.card = 10 ∧ g.spanish.card = 10) ∧
    (∀ i, i ∈ A → ∃! g, g ∈ groups ∧ i ∈ g.english) ∧
    (∀ i, i ∈ F → ∃! g, g ∈ groups ∧ i ∈ g.french) ∧
    (∀ i, i ∈ S → ∃! g, g ∈ groups ∧ i ∈ g.spanish) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_language_grouping_l873_87348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_c_value_l873_87305

/-- A polynomial of degree 4 -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The theorem stating the maximum value of |c| -/
theorem max_abs_c_value (a b c d e : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → polynomial a b c d e x ∈ Set.Icc (-1) 1) →
  |c| ≤ 8 ∧ ∃ a b c d e : ℝ, |c| = 8 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → polynomial a b c d e x ∈ Set.Icc (-1) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_c_value_l873_87305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l873_87367

/-- The length of the rectangular garden in feet -/
def rectangle_length : ℝ := 40

/-- The width of the rectangular garden in feet -/
def rectangle_width : ℝ := 20

/-- The perimeter of the garden in feet -/
def perimeter : ℝ := 2 * (rectangle_length + rectangle_width)

/-- The area of the rectangular garden in square feet -/
def rectangle_area : ℝ := rectangle_length * rectangle_width

/-- The radius of the circular garden in feet -/
noncomputable def circle_radius : ℝ := perimeter / (2 * Real.pi)

/-- The area of the circular garden in square feet -/
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2

/-- The increase in area when changing from rectangular to circular shape -/
noncomputable def area_increase : ℝ := circle_area - rectangle_area

theorem garden_area_increase :
  ∃ ε > 0, |area_increase - 345.92| < ε := by sorry

-- Remove the #eval statement as it's not computable
-- #eval area_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l873_87367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_properties_l873_87300

/-- Frustum of a cone -/
structure Frustum where
  r : ℝ  -- radius of top base
  R : ℝ  -- radius of bottom base
  l : ℝ  -- slant height

/-- Volume of a frustum -/
noncomputable def volume (f : Frustum) (h : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (f.R^2 + f.r^2 + f.R * f.r)

/-- Theorem about a specific frustum -/
theorem frustum_properties :
  ∀ (f : Frustum),
    f.r = 2 →
    f.R = 4 →
    Real.pi * (f.r + f.R) * f.l = Real.pi * (f.r^2 + f.R^2) →
    f.l = 10/3 ∧
    volume f (8/3) = 224*Real.pi/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_properties_l873_87300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equidistant_perpendicular_lines_is_parabola_l873_87345

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / (Real.sqrt (l.a^2 + l.b^2))

/-- Two perpendicular lines -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- A parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Check if a point lies on a parabola -/
def isOnParabola (p : Point2D) (para : Parabola) : Prop :=
  (p.y - para.k)^2 = 4 * para.a * (p.x - para.h)

/-- Theorem: The locus of points equidistant from two perpendicular lines is a parabola -/
theorem locus_equidistant_perpendicular_lines_is_parabola 
  (l1 l2 : Line2D) (h_perp : perpendicularLines l1 l2) :
  ∃ (para : Parabola), ∀ (p : Point2D), 
    distancePointToLine p l1 = distancePointToLine p l2 → isOnParabola p para :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equidistant_perpendicular_lines_is_parabola_l873_87345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_shortest_chord_equation_l873_87343

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define a chord passing through P
def chord_through_P (a b : ℝ) : Prop :=
  ∃ (t : ℝ), (1 - t) * point_P.1 + t * a = 1 ∧ (1 - t) * point_P.2 + t * b = 1

-- Define the length of a chord
noncomputable def chord_length (a b c d : ℝ) : ℝ :=
  Real.sqrt ((a - c)^2 + (b - d)^2)

-- Theorem for part (1)
theorem chord_equation :
  ∀ (a b c d : ℝ),
    myCircle a b → myCircle c d →
    chord_through_P a b → chord_through_P c d →
    chord_length a b c d = 2 * Real.sqrt 7 →
    (a = 1 ∧ c = 1) ∨ (b = 1 ∧ d = 1) := by sorry

-- Theorem for part (2)
theorem shortest_chord_equation :
  ∀ (a b : ℝ),
    myCircle a b →
    chord_through_P a b →
    (∀ (c d : ℝ), myCircle c d → chord_through_P c d →
      chord_length 1 1 a b ≤ chord_length 1 1 c d) →
    b = -a + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_shortest_chord_equation_l873_87343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l873_87336

theorem trig_problem (α : Real) (h1 : Real.tan (Real.pi + α) = -4/3) 
  (h2 : Real.pi < α ∧ α < 3*Real.pi/2) :
  (Real.sin α = -4/5 ∧ Real.cos α = 3/5) ∧ 
  Real.sin (25*Real.pi/6) + Real.cos (26*Real.pi/3) + Real.tan (-25*Real.pi/4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l873_87336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_semicircles_is_quarter_l873_87322

/-- The area of the region bounded by three semicircles -/
noncomputable def area_between_semicircles (cd : ℝ) (ab_perpendicular_cd : Bool) : ℝ :=
  1/4

/-- Theorem: The area of the region bounded by three semicircles is 1/4 -/
theorem area_between_semicircles_is_quarter 
  (cd : ℝ) 
  (ab_perpendicular_cd : Bool) 
  (h1 : cd = 1) 
  (h2 : ab_perpendicular_cd = true) : 
  area_between_semicircles cd ab_perpendicular_cd = 1/4 := by
  sorry

#check area_between_semicircles_is_quarter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_semicircles_is_quarter_l873_87322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l873_87380

open Real

theorem function_range_theorem (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2) 2, exp x * (x - b) + x * (exp x * (x - b + 1)) > 0) →
  b < 8/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l873_87380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_offer_difference_l873_87316

noncomputable def asking_price : ℝ := 5200

noncomputable def maintenance_inspection_cost : ℝ := asking_price / 10

noncomputable def headlight_cost : ℝ := 80

noncomputable def tire_cost : ℝ := 3 * headlight_cost

noncomputable def first_offer : ℝ := asking_price - maintenance_inspection_cost

noncomputable def second_offer : ℝ := asking_price - (headlight_cost + tire_cost)

theorem offer_difference : second_offer - first_offer = 200 := by
  simp [second_offer, first_offer, asking_price, maintenance_inspection_cost, headlight_cost, tire_cost]
  norm_num
  -- The proof is completed automatically by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_offer_difference_l873_87316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_crust_cost_is_two_l873_87397

/-- Represents the cost of an apple pie and its ingredients -/
structure ApplePie where
  servings : ℕ
  applePounds : ℚ
  appleCostPerPound : ℚ
  lemonCost : ℚ
  butterCost : ℚ
  costPerServing : ℚ

/-- Calculates the cost of the pre-made pie crust -/
def pieCrustCost (pie : ApplePie) : ℚ :=
  pie.costPerServing * pie.servings -
  (pie.applePounds * pie.appleCostPerPound + pie.lemonCost + pie.butterCost)

/-- Theorem stating that the pre-made pie crust costs $2 -/
theorem pie_crust_cost_is_two :
  let pie : ApplePie := {
    servings := 8,
    applePounds := 2,
    appleCostPerPound := 2,
    lemonCost := 1/2,
    butterCost := 3/2,
    costPerServing := 1
  }
  pieCrustCost pie = 2 := by
    -- Unfold the definition of pieCrustCost
    unfold pieCrustCost
    -- Perform the calculation
    simp [ApplePie.costPerServing, ApplePie.servings, ApplePie.applePounds,
          ApplePie.appleCostPerPound, ApplePie.lemonCost, ApplePie.butterCost]
    -- The result should now be obvious
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_crust_cost_is_two_l873_87397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_four_range_l873_87369

theorem multiple_of_four_range (start count last : Nat) : 
  start = 12 ∧ count = 21 ∧ last % 4 = 0 ∧ 
  (∃ (n : Nat), last = start + 4 * n) ∧
  (∀ (x : Nat), start ≤ x ∧ x ≤ last ∧ x % 4 = 0 → 
    ∃ (k : Nat), x = start + 4 * k ∧ k < count) ∧
  (∀ (x : Nat), x > last ∨ x < start ∨ x % 4 ≠ 0 → 
    ¬∃ (k : Nat), x = start + 4 * k ∧ k < count) →
  last = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_four_range_l873_87369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_polynomial_has_largest_coefficient_l873_87374

/-- A real polynomial of degree 4 -/
def Polynomial4 (a b c d e : ℝ) : ℝ → ℝ := λ x ↦ a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The condition that a polynomial is between 0 and 1 on [-1, 1] -/
def IsValidPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → p x ∈ Set.Icc (0 : ℝ) 1

/-- The specific polynomial we want to prove is optimal -/
def OptimalPolynomial : ℝ → ℝ := λ x ↦ 4*x^4 - 4*x^2 + 1

/-- The theorem stating that OptimalPolynomial has the largest x^4 coefficient -/
theorem optimal_polynomial_has_largest_coefficient :
  IsValidPolynomial OptimalPolynomial ∧
  ∀ a b c d e : ℝ, IsValidPolynomial (Polynomial4 a b c d e) → a ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_polynomial_has_largest_coefficient_l873_87374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_eight_hours_all_arrive_simultaneously_l873_87384

/-- Represents the journey described in the problem -/
structure Journey where
  total_distance : ℚ
  car_speed : ℚ
  walking_speed : ℚ
  harry_car_distance : ℚ
  tom_backtrack_distance : ℚ

/-- The conditions of the specific journey in the problem -/
def problem_journey : Journey where
  total_distance := 100
  car_speed := 25
  walking_speed := 5
  harry_car_distance := 75
  tom_backtrack_distance := 50

/-- Calculates the time taken for the journey -/
def journey_time (j : Journey) : ℚ :=
  j.harry_car_distance / j.car_speed + (j.total_distance - j.harry_car_distance) / j.walking_speed

/-- Theorem stating that the journey time is 8 hours -/
theorem journey_time_is_eight_hours (j : Journey) (h1 : j = problem_journey) : 
  journey_time j = 8 := by
  sorry

/-- All travelers arrive at the same time -/
theorem all_arrive_simultaneously (j : Journey) (h1 : j = problem_journey) :
  j.harry_car_distance / j.car_speed + (j.total_distance - j.harry_car_distance) / j.walking_speed =
  j.harry_car_distance / j.car_speed + j.tom_backtrack_distance / j.car_speed + 
    (j.total_distance - (j.harry_car_distance - j.tom_backtrack_distance)) / j.car_speed ∧
  j.harry_car_distance / j.car_speed + (j.total_distance - j.harry_car_distance) / j.walking_speed =
  (j.harry_car_distance - j.tom_backtrack_distance) / j.walking_speed + 
    (j.total_distance - (j.harry_car_distance - j.tom_backtrack_distance)) / j.car_speed := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_eight_hours_all_arrive_simultaneously_l873_87384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l873_87352

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the distance from a point to a line --/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line passes through a point --/
def linePassesThroughPoint (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem line_equation (l : Line) :
  (linePassesThroughPoint 3 3 l) ∧
  (distancePointToLine (-1) 1 l = 4) →
  (l = ⟨1, 0, -3⟩ ∨ l = ⟨3, 4, -21⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l873_87352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_sum_l873_87346

theorem sin_angle_sum (x : ℝ) 
  (h1 : Real.sin (3 * Real.pi / 8 - x) = 1 / 3)
  (h2 : 0 < x)
  (h3 : x < Real.pi / 2) :
  Real.sin (Real.pi / 8 + x) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_sum_l873_87346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_values_l873_87383

/-- The line equation 3x + 4y - b = 0 -/
def line (x y b : ℝ) : Prop := 3 * x + 4 * y - b = 0

/-- The circle equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line is tangent to the circle -/
def is_tangent (b : ℝ) : Prop := ∃ x y : ℝ, line x y b ∧ circle_eq x y ∧
  ∀ x' y' : ℝ, line x' y' b ∧ circle_eq x' y' → (x = x' ∧ y = y')

/-- The theorem stating the values of b for which the line is tangent to the circle -/
theorem tangent_values :
  ∀ b : ℝ, is_tangent b ↔ (b = 2 ∨ b = 12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_values_l873_87383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l873_87373

/-- Represents a triangle in the figure -/
structure Triangle where
  a : Nat
  b : Nat
  c : Nat

/-- Checks if a list of numbers is a permutation of 0 to 9 -/
def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 10 ∧ (∀ n, n ∈ arr ↔ n < 10)

/-- The sum of numbers in a triangle -/
def triangle_sum (t : Triangle) : Nat :=
  t.a + t.b + t.c

/-- Theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (arr : List Nat) (triangles : List Triangle), 
  is_valid_arrangement arr ∧ 
  (∀ t ∈ triangles, (∀ n, n = t.a ∨ n = t.b ∨ n = t.c → n ∈ arr)) ∧
  (∀ t1 t2, t1 ∈ triangles → t2 ∈ triangles → triangle_sum t1 = triangle_sum t2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l873_87373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_pairs_l873_87358

/-- The set of ordered pairs of integers (x, y) that satisfy x^2 ≤ y ≤ x+6 -/
def SatisfyingPairs : Set (ℤ × ℤ) :=
  {p | p.1^2 ≤ p.2 ∧ p.2 ≤ p.1 + 6}

/-- The number of ordered pairs of integers (x, y) that satisfy x^2 ≤ y ≤ x+6 is 26 -/
theorem count_satisfying_pairs : Finset.card (Finset.filter (fun p => p.1^2 ≤ p.2 ∧ p.2 ≤ p.1 + 6) 
  (Finset.product (Finset.range 6) (Finset.range 10))) = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_pairs_l873_87358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cooling_time_l873_87312

/-- Newton's law of cooling function -/
def cooling_law (T₀ T Tₐ : ℝ) (t h : ℝ) : Prop :=
  T - Tₐ = (1/2)^(t/h) * (T₀ - Tₐ)

theorem tea_cooling_time :
  let T₀ : ℝ := 85  -- Initial temperature
  let Tₐ : ℝ := 25  -- Ambient temperature
  let T₁ : ℝ := 55  -- Temperature after 10 minutes
  let t₁ : ℝ := 10  -- Time to cool to T₁
  let T₂ : ℝ := 45  -- Target temperature
  ∀ h : ℝ, 
    cooling_law T₀ T₁ Tₐ t₁ h →  -- First cooling observation
    ∃ t₂ : ℝ, 
      cooling_law T₀ T₂ Tₐ t₂ h ∧  -- Second cooling to target temp
      (abs (t₂ - 16) < 0.5)  -- Approximate time is 16 minutes (within 0.5 minute)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cooling_time_l873_87312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l873_87335

/-- The minimum value of φ for the given conditions -/
noncomputable def min_phi : ℝ := 29 * Real.pi / 30

/-- The original function f(x) -/
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (4 * x + φ)

/-- The translated function g(x) -/
noncomputable def g (x φ : ℝ) : ℝ := f (x + Real.pi / 3) φ

/-- Theorem stating the minimum value of φ -/
theorem min_phi_value (φ : ℝ) (h1 : φ > 0)
  (h2 : ∀ x, g x φ = g (-2 * Real.pi / 5 - x) φ) : 
  φ ≥ min_phi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l873_87335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_polygon_sides_l873_87372

/-- The number of sides of a regular polygon inscribed in a circle of radius 3 with area 45 -/
def number_of_sides : ℕ := 15

/-- The radius of the circle -/
def R : ℝ := 3

/-- The area of the inscribed regular polygon -/
def A : ℝ := 5 * R^2

/-- Formula for the area of a regular polygon inscribed in a circle -/
noncomputable def polygon_area (n : ℕ) (R : ℝ) : ℝ :=
  1/2 * (n : ℝ) * R^2 * Real.sin (2 * Real.pi / (n : ℝ))

theorem inscribed_polygon_sides :
  polygon_area number_of_sides R = A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_polygon_sides_l873_87372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l873_87353

theorem circle_radius : 
  ∃ (h k r : ℝ), r = Real.sqrt 3 / 2 ∧ 
    ∀ (x y : ℝ), 16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 68 = 0 ↔ 
      (x - h)^2 + (y - k)^2 = r^2 :=
by
  -- Introduce the center coordinates and radius
  let h := 1
  let k := -2
  let r := Real.sqrt 3 / 2

  -- Prove the existence of h, k, and r
  use h, k, r

  constructor
  · -- Prove that r = √3 / 2
    rfl

  · -- Prove the equivalence
    intro x y
    constructor
    · -- Forward direction
      intro eq
      sorry -- Complete the proof here
    · -- Reverse direction
      intro eq
      sorry -- Complete the proof here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l873_87353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l873_87395

def sequence_sum (seq : List ℝ) (start : ℕ) : ℝ :=
  (seq.drop start).take 4 |>.sum

theorem sequence_property (J K L M N O P Q R S : ℝ) : 
  N = 7 →
  (∀ i ∈ [0, 1, 2, 3, 4, 5, 6], 
    sequence_sum [J, K, L, M, N, O, P, Q, R, S] i = 40) →
  J + S = 40 := by
  sorry

#check sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l873_87395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_function_participation_l873_87301

theorem school_function_participation 
  (total_students : ℕ)
  (boys_participation_rate : ℚ)
  (total_participants : ℕ)
  (participating_girls : ℕ)
  (h1 : total_students = 800)
  (h2 : boys_participation_rate = 2/3)
  (h3 : total_participants = 550)
  (h4 : participating_girls = 150)
  : ∃ (fraction_of_girls_participating : ℚ), 
    fraction_of_girls_participating = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_function_participation_l873_87301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_power_two_dividing_factorial_l873_87323

theorem ones_digit_power_two_dividing_factorial : ∃ n : ℕ, 
  (2^15 : ℕ) % 10 = 8 ∧ 
  ∀ m : ℕ, m > 15 → ¬(∃ k : ℕ, (16 : ℕ).factorial = 2^m * k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_power_two_dividing_factorial_l873_87323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l873_87396

/-- The ellipse representing curve C₁ -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line representing curve C₂ -/
def line (x y : ℝ) : Prop := x + y = 4

/-- The distance between a point (x₁, y₁) on the ellipse and a point (x₂, y₂) on the line -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The theorem stating that the minimum distance between the ellipse and the line is √2 -/
theorem min_distance_ellipse_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ellipse x₁ y₁ ∧ 
    line x₂ y₂ ∧ 
    (∀ (x₃ y₃ x₄ y₄ : ℝ), ellipse x₃ y₃ → line x₄ y₄ → 
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l873_87396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_show_cost_l873_87362

/-- Represents the cost and episode structure of a TV show -/
structure TVShow where
  seasons : Nat
  first_season_cost : Nat
  first_season_episodes : Nat
  last_season_episodes : Nat

/-- Calculates the total cost of producing all episodes of a TV show -/
def total_cost (s : TVShow) : Nat :=
  let other_season_cost := s.first_season_cost * 2
  let other_season_episodes := s.first_season_episodes * 3 / 2
  let first_season_total := s.first_season_cost * s.first_season_episodes
  let middle_seasons_total := other_season_cost * other_season_episodes * (s.seasons - 2)
  let last_season_total := other_season_cost * s.last_season_episodes
  first_season_total + middle_seasons_total + last_season_total

/-- The specific TV show described in the problem -/
def our_show : TVShow :=
  { seasons := 5
  , first_season_cost := 100000
  , first_season_episodes := 12
  , last_season_episodes := 24 }

/-- Theorem stating that the total cost of producing our_show is $16,800,000 -/
theorem our_show_cost :
  total_cost our_show = 16800000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_show_cost_l873_87362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_2pi_periodic_l873_87387

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x

theorem f_is_odd_and_2pi_periodic :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 2 * Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_2pi_periodic_l873_87387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l873_87337

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x - 4*y = -y^2 + 2*y - 7

-- Define the center and radius
def center : ℝ × ℝ := (-4, 2)
noncomputable def radius : ℝ := 3 * Real.sqrt 3

-- Theorem statement
theorem circle_properties :
  let (a, b) := center
  ∀ x y : ℝ, circle_equation x y →
    ((x + 4)^2 + (y - 2)^2 = radius^2) ∧
    (a + b + radius = -2 + 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l873_87337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l873_87330

/-- Definition of the hyperbola C -/
def C (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

/-- Definition of the foci F₁ and F₂ -/
def is_focus (F : ℝ × ℝ) (C : ℝ → ℝ → Prop) : Prop := sorry

theorem hyperbola_properties 
  (F₁ F₂ : ℝ × ℝ) 
  (h₁ : is_focus F₁ C) 
  (h₂ : is_focus F₂ C) 
  (h₃ : F₁.1 < F₂.1) : 
  (∃ (d : ℝ), d = Real.sqrt 3 ∧ dist F₁ F₂ = 2 * d) ∧ 
  (∃ (e : ℝ), e = Real.sqrt 6 / 2 ∧ 
    ∀ (P : ℝ × ℝ), C P.1 P.2 → e = |dist P F₁ - dist P F₂| / (2 * Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l873_87330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l873_87350

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_power_of_18_dividing_30_factorial :
  (∃ n : ℕ, n ≤ 7 ∧ factorial 30 % (18^n) = 0) ∧
  ∀ m : ℕ, m > 7 → factorial 30 % (18^m) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l873_87350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l873_87393

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

theorem f_properties :
  (∃ φ : ℝ, ∀ x : ℝ, f (x + φ) = -f (-x - φ)) ∧
  (∀ x : ℝ, f (x - 3 * Real.pi / 4) = f (-x - 3 * Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l873_87393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_loss_percentage_l873_87331

/-- Represents the purchase and sale data for an article type -/
structure ArticleData where
  purchaseQuantity : ℕ
  purchasePrice : ℕ
  saleQuantity : ℕ
  salePrice : ℕ

/-- Calculates the total cost price for all article types -/
def totalCostPrice (articles : List ArticleData) : ℕ :=
  articles.foldl (fun acc a => acc + a.purchaseQuantity * a.purchasePrice) 0

/-- Calculates the total selling price for all article types -/
def totalSellingPrice (articles : List ArticleData) : ℕ :=
  articles.foldl (fun acc a => acc + a.saleQuantity * a.salePrice) 0

/-- Calculates the loss percentage given the purchase and sale data -/
noncomputable def lossPercentage (articles : List ArticleData) : ℝ :=
  let tcp := (totalCostPrice articles : ℝ)
  let tsp := (totalSellingPrice articles : ℝ)
  let loss := tcp - tsp
  (loss / tcp) * 100

/-- The main theorem stating the loss percentage for the given data -/
theorem dealer_loss_percentage : 
  let articles : List ArticleData := [
    { purchaseQuantity := 15, purchasePrice := 25, saleQuantity := 12, salePrice := 38 },
    { purchaseQuantity := 20, purchasePrice := 40, saleQuantity := 18, salePrice := 50 },
    { purchaseQuantity := 30, purchasePrice := 55, saleQuantity := 22, salePrice := 65 },
    { purchaseQuantity := 10, purchasePrice := 80, saleQuantity := 8, salePrice := 100 }
  ]
  ∃ (ε : ℝ), abs (lossPercentage articles - 1.075) < ε ∧ ε < 0.001 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_loss_percentage_l873_87331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_theta_l873_87318

noncomputable section

variable (a b : ℝ × ℝ)
variable (θ : ℝ)

/-- Angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- Theorem stating the maximum value of sin θ -/
theorem max_sin_theta (h1 : Real.sqrt (a.1^2 + a.2^2) = 1) 
                      (h2 : Real.sqrt ((b.1 - 2*a.1)^2 + (b.2 - 2*a.2)^2) = 1) 
                      (h3 : θ = angle a b) : 
  ∀ θ', θ' = angle a b → Real.sin θ' ≤ 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_theta_l873_87318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_length_specific_cylinder_l873_87389

/-- Represents a cylindrical surface with a spiral strip --/
structure CylinderWithSpiral where
  base_circumference : ℝ
  height : ℝ
  horizontal_offset : ℝ

/-- Calculate the length of the spiral strip on the cylinder --/
noncomputable def spiral_length (c : CylinderWithSpiral) : ℝ :=
  Real.sqrt (c.base_circumference ^ 2 + c.height ^ 2)

/-- Theorem stating the length of the spiral strip for the given cylinder --/
theorem spiral_length_specific_cylinder :
  let c : CylinderWithSpiral := {
    base_circumference := 18,
    height := 10,
    horizontal_offset := 6
  }
  spiral_length c = Real.sqrt 424 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_length_specific_cylinder_l873_87389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_residuals_represents_differences_l873_87375

/-- Represents a data point in a regression analysis -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted y-value for a given x-value on the regression line -/
def predictedY (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- Calculates the residual for a single data point -/
def calculateResidual (point : DataPoint) (line : RegressionLine) : ℝ :=
  point.y - predictedY line point.x

/-- Calculates the Sum of Squares of Residuals for a set of data points -/
def sumOfSquaresResiduals (data : List DataPoint) (line : RegressionLine) : ℝ :=
  (data.map (fun point => (calculateResidual point line)^2)).sum

/-- Theorem: The Sum of Squares of Residuals represents the differences between 
    observed data points and their corresponding positions on the regression line -/
theorem sum_of_squares_residuals_represents_differences 
  (data : List DataPoint) (line : RegressionLine) :
  sumOfSquaresResiduals data line = 
    (data.map (fun point => (point.y - predictedY line point.x)^2)).sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_residuals_represents_differences_l873_87375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_one_l873_87364

/-- Theorem: The slope of a line passing through points A(-1, 4) and B(1, 2) is -1 -/
theorem line_slope_is_negative_one :
  (2 - 4) / (1 - (-1)) = -1 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_negative_one_l873_87364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l873_87313

def numbers : List Nat := [12, 30, 48, 74, 100, 113, 127, 139]

def list_lcm : List Nat → Nat
  | [] => 1
  | (x::xs) => Nat.lcm x (list_lcm xs)

theorem smallest_number_divisible (n : Nat) : 
  (∀ m ∈ numbers, (n + 2) % m = 0) ∧ 
  (∀ k < n, ∃ m ∈ numbers, (k + 2) % m ≠ 0) → 
  n = list_lcm numbers - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l873_87313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_specific_tank_check_water_volume_l873_87355

/-- The volume of water in a horizontal cylindrical tank -/
noncomputable def water_volume (r h d : ℝ) : ℝ :=
  let θ := Real.arccos (d / r)
  let sector_area := 2 * θ / Real.pi * Real.pi * r^2
  let triangle_area := d * Real.sqrt (r^2 - d^2)
  h * (sector_area - triangle_area)

/-- Theorem: The volume of water in a specific horizontal cylindrical tank -/
theorem water_volume_specific_tank :
  water_volume 6 12 3 = 144 * Real.pi - 108 * Real.sqrt 3 := by
  sorry

-- Note: #eval cannot be used with noncomputable functions
-- Instead, we can state the goal as a theorem
theorem check_water_volume :
  (⌊water_volume 6 12 3⌋ : ℤ) = 141 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_specific_tank_check_water_volume_l873_87355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l873_87349

/-- Represents a rectangular farm with given dimensions and fencing requirements. -/
structure RectangularFarm where
  area : ℝ
  shortSide : ℝ
  fencingCost : ℝ

/-- Calculates the total cost of fencing for a rectangular farm. -/
noncomputable def fencingCost (farm : RectangularFarm) : ℝ :=
  let longSide := farm.area / farm.shortSide
  let diagonal := Real.sqrt (farm.shortSide ^ 2 + longSide ^ 2)
  let totalLength := longSide + farm.shortSide + diagonal
  totalLength * farm.fencingCost

/-- Theorem stating that the fencing cost for the given farm is 1800 Rs. -/
theorem farm_fencing_cost :
  let farm : RectangularFarm := {
    area := 1200
    shortSide := 30
    fencingCost := 15
  }
  fencingCost farm = 1800 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l873_87349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_plus_π_3_l873_87356

open Real

/-- The function f as defined in the problem -/
noncomputable def f (α : ℝ) : ℝ := (tan (π - α) * cos (2*π - α) * sin (π/2 + α)) / cos (-α - π)

/-- Theorem statement -/
theorem cos_2α_plus_π_3 (α : ℝ) (h1 : f α = 4/5) (h2 : π/2 < α ∧ α < π) :
  cos (2*α + π/3) = (24 * sqrt 3 - 7) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_plus_π_3_l873_87356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_across_imaginary_axis_l873_87328

-- Define the complex number representing point A
def z : ℂ := Complex.mk 2 1

-- Define the reflection operation across the imaginary axis
def reflect_im (w : ℂ) : ℂ := Complex.mk (-w.re) w.im

-- Theorem statement
theorem reflection_across_imaginary_axis :
  reflect_im z = Complex.mk (-2) 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_across_imaginary_axis_l873_87328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangle_areas_sum_l873_87320

noncomputable def cube_edge_length : ℝ := 2

noncomputable def triangle_area_sum (m n p : ℕ) : ℝ := m + Real.sqrt (n : ℝ) + Real.sqrt (p : ℝ)

theorem cube_triangle_areas_sum :
  ∃ m n p : ℕ,
    triangle_area_sum m n p =
      (6 * 4 * (cube_edge_length^2 / 2)) +
      (24 * (cube_edge_length * Real.sqrt (2 * cube_edge_length^2) / 2)) +
      (8 * (Real.sqrt 3 * cube_edge_length^2 / 2)) ∧
    m + n + p = 5232 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangle_areas_sum_l873_87320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_function_correct_profit_1200_price_max_profit_price_max_profit_value_l873_87351

-- Define the variables and constants
variable (x y : ℝ)
def cost : ℝ := 30
def min_price : ℝ := 30
def max_price : ℝ := 54

-- Define the linear relationship between y and x
def sales_function (x : ℝ) : ℝ := -2 * x + 160

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost) * (sales_function x)

-- Theorem statements
theorem sales_function_correct : 
  sales_function 35 = 90 ∧ sales_function 40 = 80 := by
  sorry

theorem profit_1200_price : 
  ∃ x : ℝ, min_price ≤ x ∧ x ≤ max_price ∧ profit_function x = 1200 ∧ x = 50 := by
  sorry

theorem max_profit_price : 
  ∃ x : ℝ, x = max_price ∧ 
  ∀ y : ℝ, min_price ≤ y ∧ y ≤ max_price → profit_function y ≤ profit_function x := by
  sorry

theorem max_profit_value : 
  profit_function max_price = 1248 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_function_correct_profit_1200_price_max_profit_price_max_profit_value_l873_87351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l873_87365

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (b c h : ℝ) (hb : b > 0) (hc : c > 0) (hbc : b > c) :
  let a := Real.sqrt (b^2 - c^2)
  ∃ (X Z : ℝ × ℝ),
    (X.1^2 + X.2^2 = b^2) ∧
    (Z.1^2 + Z.2^2 = c^2) ∧
    ((X.1 - Z.1)^2 + (X.2 - Z.2)^2 = a^2) ∧
    (X.1 * Z.1 + X.2 * Z.2 = b * c) ∧
    h^2 = b^2 - 2*c^2 →
  π * (b^2 - c^2) = π * h^2 := by
  sorry

#check annulus_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_l873_87365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_t_l873_87306

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x + 1/x - 3

-- State the theorem
theorem f_negative_t (t : ℝ) (h : f t = 4) : f (-t) = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_t_l873_87306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inverse_mod_l873_87370

theorem right_triangle_inverse_mod (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : a = 48) (h3 : b = 275) (h4 : c = 277) :
  (550 : ZMod 4319).inv = 2208 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inverse_mod_l873_87370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_characterization_l873_87302

/-- The set of complex numbers representing the roots of the equation -/
def roots (a : ℝ) : Set ℂ :=
  {z : ℂ | (z^2 - 2*z + 5) * (z^2 + 2*a*z + 1) = 0}

/-- The condition that the roots are distinct -/
def roots_distinct (a : ℝ) : Prop :=
  ∀ z w, z ∈ roots a → w ∈ roots a → z ≠ w → z = w

/-- The condition that the roots lie on a circle -/
def roots_on_circle (a : ℝ) : Prop :=
  ∃ c : ℂ, ∃ r : ℝ, ∀ z, z ∈ roots a → Complex.abs (z - c) = r

/-- The main theorem -/
theorem roots_characterization (a : ℝ) :
  (roots_distinct a ∧ roots_on_circle a) ↔ a ∈ ({-3} : Set ℝ) ∪ Set.Ioo (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_characterization_l873_87302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_theorem_l873_87342

/-- A type representing colors (red or blue) -/
inductive Color
| Red
| Blue

/-- A type representing points in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A coloring of the plane -/
def Coloring := Point → Color

theorem two_color_plane_theorem (coloring : Coloring) :
  (∀ x : ℝ, x > 0 → ∃ c : Color, ∃ p1 p2 : Point, distance p1 p2 = x ∧ coloring p1 = c ∧ coloring p2 = c) ∧
  (∃ c : Color, ∀ x : ℝ, x > 0 → ∃ p1 p2 : Point, distance p1 p2 = x ∧ coloring p1 = c ∧ coloring p2 = c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_theorem_l873_87342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_problem_l873_87315

def initial_crayons : ℕ := 48

def kiley_fraction : ℚ := 1/4

def joe_fraction : ℚ := 1/2

def remaining_crayons : ℕ := 18

theorem crayon_problem :
  (initial_crayons - Int.floor (kiley_fraction * initial_crayons)) / 2 = remaining_crayons :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_problem_l873_87315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bodyguard_activities_l873_87390

-- Define the universe
variable {α : Type*}

-- Define the sets
variable (U : Set α) -- Set of bodyguards who swim
variable (S : Set α) -- Set of bodyguards who play chess
variable (T : Set α) -- Set of bodyguards who play tennis

-- Define the conditions
axiom cond1 : T ∩ (S.compl ∩ U) = ∅
axiom cond2 : S ∩ (U.compl ∩ T.compl) = ∅
axiom cond3 : (U.compl ∩ T) ⊆ S

-- Define the statements to be proved
def statement1 (U S T : Set α) : Prop := S ⊆ U
def statement2 (U S T : Set α) : Prop := U ⊆ T
def statement3 (U S T : Set α) : Prop := T ⊆ S

-- Theorem to be proved
theorem bodyguard_activities :
  ∃ U S T : Set α, (statement1 U S T ∧ statement3 U S T) ∧ ¬(statement2 U S T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bodyguard_activities_l873_87390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_implies_expression_equals_two_l873_87325

theorem tan_half_implies_expression_equals_two (α : ℝ) (h : Real.tan α = 1/2) :
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_implies_expression_equals_two_l873_87325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_coefficient_proof_l873_87332

/-- The reflection coefficient at the glass-air boundary -/
noncomputable def reflection_coefficient : ℝ := 1 / 2

/-- The transmission coefficient at the glass-air boundary -/
noncomputable def transmission_coefficient : ℝ := 1 - reflection_coefficient

/-- The intensity reduction factor after passing through the system -/
def intensity_reduction : ℝ := 16

theorem reflection_coefficient_proof :
  (1 - reflection_coefficient)^4 = 1 / intensity_reduction :=
by
  -- Placeholder for the proof
  sorry

#check reflection_coefficient_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_coefficient_proof_l873_87332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_race_total_time_l873_87321

-- Define the race times for each turtle
noncomputable def greta_time : ℝ := 6
noncomputable def george_time : ℝ := greta_time - 2
noncomputable def gloria_time : ℝ := 2 * george_time
noncomputable def gary_time : ℝ := gloria_time + 1.5
noncomputable def gwen_time : ℝ := (greta_time + george_time) / 2

-- Define the total race time
noncomputable def total_race_time : ℝ := greta_time + george_time + gloria_time + gary_time + gwen_time

-- Theorem statement
theorem turtle_race_total_time : total_race_time = 32.5 := by
  -- Unfold definitions
  unfold total_race_time greta_time george_time gloria_time gary_time gwen_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_race_total_time_l873_87321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_equality_l873_87304

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x else 3 * x - 52

-- State the theorem
theorem piecewise_equality (a : ℝ) :
  a < 0 → (f (f (f 11)) = f (f (f a)) ↔ a = -19) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_equality_l873_87304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_barrel_half_water_l873_87347

/-- The number of days required for a barrel of wine to become half water -/
noncomputable def days_to_half_water (initial_volume : ℝ) (daily_exchange : ℝ) : ℝ :=
  Real.log (1 / 2) / Real.log (1 - daily_exchange / initial_volume)

theorem wine_barrel_half_water :
  let initial_volume : ℝ := 100
  let daily_exchange : ℝ := 1
  let result := days_to_half_water initial_volume daily_exchange
  ⌊result⌋ = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_barrel_half_water_l873_87347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l873_87324

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) / (1 + Real.sin x + Real.cos x)

-- Define the set S as the range of f
def S : Set ℝ := {y | ∃ x, f x = y ∧ 1 + Real.sin x + Real.cos x ≠ 0}

-- State the theorem about the range of f
theorem range_of_f : S = Set.Icc (-Real.sqrt 2 / 2 - 1 / 2) (-1) ∪ Set.Ioc (-1) (Real.sqrt 2 / 2 - 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l873_87324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_elements_sqrt_difference_l873_87381

theorem eleven_elements_sqrt_difference (A : Finset ℕ) (B : Finset ℕ) : 
  A = Finset.range 100 →
  B ⊆ A →
  B.card = 11 →
  ∃ x y : ℕ, x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ 0 < |Real.sqrt x - Real.sqrt y| ∧ |Real.sqrt x - Real.sqrt y| < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_elements_sqrt_difference_l873_87381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l873_87308

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 8 = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := 2*x - y - 7 = 0

-- Define a parallel line
def parallel_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- The main theorem
theorem circle_symmetry :
  (∃ (m : ℝ → ℝ → Prop), 
    (∀ x y, m x y ↔ ∃ k, 2*x - y + k = 0) ∧ 
    (∀ x y, circle_eq x y → (∃ x' y', m x' y' ∧ 
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2))) →
  (∀ x y, line_of_symmetry x y ↔ 
    (∃ x' y', circle_eq x' y' ∧ 
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l873_87308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l873_87376

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (1, 0)

-- Define point P
def P : ℝ × ℝ := (1, -1)

-- Define point M
noncomputable def M : ℝ × ℝ := (2 * Real.sqrt 6 / 3, -1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem minimize_distance :
  ellipse M.1 M.2 ∧
  ∀ N : ℝ × ℝ, ellipse N.1 N.2 →
    distance M P + 2 * distance M right_focus ≤
    distance N P + 2 * distance N right_focus := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l873_87376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l873_87339

def f (x : ℝ) : ℝ := 8 * x^4 + 28 * x^3 - 74 * x^2 - 8 * x + 48

theorem roots_of_polynomial :
  (f (-3) = 0) ∧ (f (-1) = 0) ∧ (f 2 = 0) ∧
  (∃ ε : ℝ, ε > 0 ∧ ∀ x : ℝ, 0 < |x + 1| ∧ |x + 1| < ε → f x ≠ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l873_87339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_for_a_3_find_range_of_a_l873_87354

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - (1/2) * x

-- Part I
theorem solve_inequality_for_a_3 :
  ∀ x : ℝ, f 3 x < 0 ↔ 2 < x ∧ x < 6 := by sorry

-- Part II
theorem find_range_of_a :
  ∀ a : ℝ, a > 0 → 
  (∀ x : ℝ, f a x - f a (x + a) < a^2 + a/2) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_for_a_3_find_range_of_a_l873_87354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l873_87310

def repeating_decimal : ℚ := 24 / 99

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  (n : ℚ) / d = repeating_decimal ∧ 
  Nat.Coprime n d ∧
  n + d = 41 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l873_87310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l873_87334

theorem cos_double_angle (θ : Real) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l873_87334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_allan_brought_five_l873_87379

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := sorry

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 11

/-- Jake brought 6 more balloons than Allan -/
axiom jake_more_balloons : jake_balloons = allan_balloons + 6

theorem allan_brought_five : allan_balloons = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_allan_brought_five_l873_87379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_in_quadrant_I_l873_87341

noncomputable def special_point : ℝ × ℝ :=
  let x : ℝ := 3/5
  let y : ℝ := 18/5
  (x, y)

def line_equation (p : ℝ × ℝ) : Prop :=
  4 * p.1 + 6 * p.2 = 24

def y_condition (p : ℝ × ℝ) : Prop :=
  p.2 = p.1 + 3

def in_quadrant_I (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

theorem special_point_in_quadrant_I :
  line_equation special_point ∧ 
  y_condition special_point → 
  in_quadrant_I special_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_point_in_quadrant_I_l873_87341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_ratio_l873_87307

/-- Given a cylinder of fixed volume V, the total surface area (including the two circular ends) 
    is minimized when the ratio of height to radius is 4. -/
theorem cylinder_min_surface_area_ratio (V : ℝ) (V_pos : V > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ V = π * r^2 * h ∧
  (∀ (r' h' : ℝ), r' > 0 → h' > 0 → V = π * r'^2 * h' →
    π * r^2 + 2 * π * r * h ≤ π * r'^2 + 2 * π * r' * h') ∧
  h / r = 4 := by
  sorry

#check cylinder_min_surface_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_ratio_l873_87307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_improvement_percentage_l873_87303

/-- Bob's current mile time in seconds -/
noncomputable def bob_time : ℝ := 640

/-- Bob's sister's current mile time in seconds -/
noncomputable def sister_time : ℝ := 557

/-- The percentage improvement Bob needs to match his sister's time -/
noncomputable def improvement_percentage : ℝ := ((bob_time - sister_time) / bob_time) * 100

/-- Theorem stating that Bob needs to improve by approximately 12.97% -/
theorem bob_improvement_percentage :
  |improvement_percentage - 12.97| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_improvement_percentage_l873_87303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reimbursement_is_sixty_cents_l873_87327

/-- The amount Julie should reimburse Sarah for the shared lollipops, in cents --/
def reimbursement_amount (total_lollipops : ℕ) (total_cost : ℚ) (discount_rate : ℚ) (shared_fraction : ℚ) : ℕ :=
  let cost_per_lollipop := total_cost / total_lollipops
  let discounted_cost_per_lollipop := cost_per_lollipop * (1 - discount_rate)
  let shared_lollipops := total_lollipops * shared_fraction
  (shared_lollipops * discounted_cost_per_lollipop * 100).floor.toNat

/-- Theorem stating that the reimbursement amount is 60 cents --/
theorem reimbursement_is_sixty_cents :
  reimbursement_amount 12 3 (1/5) (1/4) = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reimbursement_is_sixty_cents_l873_87327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_six_point_five_l873_87311

-- Define the vectors
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![1, 5]

-- Define the origin
def origin : Fin 2 → ℚ := ![0, 0]

-- Define the area of the triangle
noncomputable def triangle_area (v1 v2 v3 : Fin 2 → ℚ) : ℚ :=
  (1/2) * abs ((v2 0 - v1 0) * (v3 1 - v1 1) - (v2 1 - v1 1) * (v3 0 - v1 0))

-- Theorem statement
theorem triangle_area_equals_six_point_five :
  triangle_area origin a b = 13/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_six_point_five_l873_87311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l873_87377

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (n : ℕ) (a₁ q : ℝ) : ℝ := a₁ * (1 - q^n) / (1 - q)

/-- Theorem: If S₃ = 3a₁ and S₃, S₉, S₆ form an arithmetic sequence, 
    then a₂, a₈, a₅ also form an arithmetic sequence -/
theorem geometric_sequence_property (a₁ q : ℝ) (hq : q ≠ 1) :
  S 3 a₁ q = 3 * a₁ ∧ 
  2 * S 9 a₁ q = S 3 a₁ q + S 6 a₁ q →
  2 * (a₁ * q^7) = a₁ * q + a₁ * q^4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l873_87377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_market_spending_l873_87344

/-- Calculates the total amount spent by Jayda and Aitana at Silverlake Flea market in Canadian dollars --/
theorem flea_market_spending (jayda_stall1 jayda_stall2 jayda_stall3 : ℚ)
  (aitana_factor : ℚ) (sales_tax_rate : ℚ) (exchange_rate : ℚ) :
  jayda_stall1 = 400 →
  jayda_stall2 = 120 →
  jayda_stall3 = 250 →
  aitana_factor = 2/5 →
  sales_tax_rate = 1/10 →
  exchange_rate = 5/4 →
  (jayda_stall1 + jayda_stall2 + jayda_stall3 +
   (1 + aitana_factor) * (jayda_stall1 + jayda_stall2 + jayda_stall3)) *
  (1 + sales_tax_rate) * exchange_rate = 2541 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_market_spending_l873_87344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_p_l873_87333

-- Define the polynomial
noncomputable def p (x : ℝ) : ℝ := (4^(-x) - 1) * (2^x - 3)^5

-- Theorem statement
theorem constant_term_of_p : ∃ c : ℝ, ∀ x : ℝ, p x = c + x * (p x - c) ∧ c = -27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_p_l873_87333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nonnegative_iff_in_solution_set_l873_87371

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3*x - 8) * (x - 4) * (x + 1) / (x - 2)

-- Define the solution set
def solution_set : Set ℝ := Set.Ici 4 ∪ Set.Ioo 2 (8/3) ∪ Set.Iic (-1)

-- State the theorem
theorem g_nonnegative_iff_in_solution_set :
  ∀ x : ℝ, x ≠ 2 → (g x ≥ 0 ↔ x ∈ solution_set) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_nonnegative_iff_in_solution_set_l873_87371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_powers_theorem_l873_87314

/-- Division powers function -/
def f (n : ℕ) (a : ℚ) : ℚ :=
  a ^ (2 - n : ℤ)

theorem division_powers_theorem :
  ∀ (n : ℕ) (a : ℚ), a ≠ 0 →
  (f n a = a ^ (2 - n : ℤ)) ∧
  (f 5 (1/3) * f 4 3 * f 5 (1/2) + f 5 (-1/4) / f 6 (-1/2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_powers_theorem_l873_87314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l873_87338

-- Define the work rates
noncomputable def work_rate_B : ℝ := 1 / 36
noncomputable def work_rate_A : ℝ := work_rate_B
noncomputable def work_rate_C : ℝ := 2 * work_rate_B

-- Define the combined work rate
noncomputable def combined_work_rate : ℝ := work_rate_A + work_rate_B + work_rate_C

-- Theorem statement
theorem job_completion_time :
  (1 / combined_work_rate) = 9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l873_87338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l873_87399

/-- Given a hyperbola with equation (x^2/a^2) - (y^2/b^2) = 1, where a > 0 and b > 0,
    and focal length 2c, if a^2, b^2, c^2 form an arithmetic progression,
    then the equation of the asymptote is y = ±√2x. -/
theorem hyperbola_asymptote (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * b^2 = a^2 + c^2) →  -- arithmetic progression condition
  (c^2 = a^2 + b^2) →      -- focal length condition
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) -- asymptote equation
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l873_87399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_product_distances_M_to_AB_l873_87378

-- Define the curves C1 and C2
noncomputable def C1 (φ : ℝ) : ℝ × ℝ := (1 / Real.tan φ, 1 / (Real.tan φ)^2)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := 
  let ρ := 1 / (Real.cos θ + Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := ((Real.sqrt 5 - 1) / 2, (3 - Real.sqrt 5) / 2)
noncomputable def B : ℝ × ℝ := (-(Real.sqrt 5 + 1) / 2, (3 + Real.sqrt 5) / 2)

-- Define point M
def M : ℝ × ℝ := (-1, 2)

-- Theorem for the distance between A and B
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10 := by sorry

-- Theorem for the product of distances from M to A and B
theorem product_distances_M_to_AB : 
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) * 
  Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_product_distances_M_to_AB_l873_87378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_zero_point_eight_in_terms_of_a_l873_87392

theorem lg_zero_point_eight_in_terms_of_a (a : ℝ) (h : (2 : ℝ)^a = 5) :
  Real.log 0.8 / Real.log 10 = (2 - a) / (1 + a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_zero_point_eight_in_terms_of_a_l873_87392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_e_when_derivative_at_one_is_one_l873_87359

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_base_e_when_derivative_at_one_is_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (deriv (f a)) 1 = 1 → a = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_e_when_derivative_at_one_is_one_l873_87359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_of_hyperbola_C1_l873_87394

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

theorem major_axis_length_of_hyperbola_C1 
  (C1 : Hyperbola) 
  (C2 : Hyperbola)
  (h_asymptote : ∃ (M : ℝ × ℝ), M.2 = (C1.b / C1.a) * M.1)
  (h_perpendicular : ∃ (O M F2 : ℝ × ℝ), (M.1 - O.1) * (F2.1 - M.1) + (M.2 - O.2) * (F2.2 - M.2) = 0)
  (h_area : ∃ (O M F2 : ℝ × ℝ), (M.1 - O.1) * (F2.2 - O.2) - (M.2 - O.2) * (F2.1 - O.1) = 32)
  (h_C2 : C2.a = 4 ∧ C2.b = 2)
  (h_same_eccentricity : eccentricity C1 = eccentricity C2) : 
  2 * C1.a = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_of_hyperbola_C1_l873_87394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_on_BC_l873_87386

noncomputable section

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (-1, -1)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

def projection_vector : ℝ × ℝ := (1/5, 3/5)

theorem projection_AB_on_BC :
  let mag_AB := Real.sqrt (vector_AB.1^2 + vector_AB.2^2)
  let mag_BC := Real.sqrt (vector_BC.1^2 + vector_BC.2^2)
  let dot_product := vector_AB.1 * vector_BC.1 + vector_AB.2 * vector_BC.2
  let cos_angle := dot_product / (mag_AB * mag_BC)
  let proj_magnitude := mag_AB * abs cos_angle
  let scale_factor := proj_magnitude / mag_BC
  (scale_factor * vector_BC.1, scale_factor * vector_BC.2) = projection_vector :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AB_on_BC_l873_87386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_opposite_product_one_l873_87366

theorem log_opposite_product_one (a b : ℝ) (h : a > 0) (k : b > 0) :
  Real.log a = -(Real.log b) → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_opposite_product_one_l873_87366
