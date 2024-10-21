import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l882_88284

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem statement
theorem tangent_slope_angle_at_one :
  Real.arctan (f' 1) = 3 * π / 4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l882_88284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l882_88285

-- Define the number of male students as a natural number
variable (m : ℕ)

-- Define the contingency table
def a (m : ℕ) : ℕ := 2 * m  -- Male students who like sweets
def b (m : ℕ) : ℕ := 4 * m  -- Female students who like sweets
def c (m : ℕ) : ℕ := 3 * m  -- Male students who dislike sweets
def d (m : ℕ) : ℕ := m      -- Female students who dislike sweets

-- Define the total number of students
def n (m : ℕ) : ℕ := a m + b m + c m + d m

-- Define the K² formula
noncomputable def K_squared (m : ℕ) : ℚ := 
  (n m : ℚ) * ((a m * d m - b m * c m : ℕ) ^ 2 : ℚ) / 
  ((a m + b m : ℕ) * (c m + d m) * (a m + c m) * (b m + d m) : ℚ)

-- Define the 95% certainty threshold
def lower_bound : ℚ := 3841 / 1000
def upper_bound : ℚ := 6635 / 1000

-- Theorem statement
theorem survey_result : 
  (K_squared 3 ≥ lower_bound ∧ K_squared 3 < upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_l882_88285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l882_88232

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l882_88232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l882_88269

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 36 * x^2 - 48 * x + 21

-- Define the proposed inverse function h_inv
noncomputable def h_inv (x : ℝ) : ℝ := (4 + Real.sqrt (x - 5)) / 6

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, x ≥ 5 → h (h_inv x) = x ∧ h_inv (h x) = x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l882_88269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l882_88215

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the theorem
theorem triangle_cosine_theorem (t : Triangle) 
  (h1 : Real.sin t.A = 12/13) 
  (h2 : Real.cos t.B = 3/5) : 
  Real.cos t.C = 33/65 ∨ Real.cos t.C = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l882_88215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l882_88276

noncomputable def h (x : ℝ) : ℝ := (x^3 - 2*x^2 - x + 1) / (x^2 - 9)

theorem domain_of_h :
  {x : ℝ | ∃ y, h x = y} = Set.Iio (-3) ∪ Set.Ioo (-3) 3 ∪ Set.Ioi 3 := by
  sorry

#check domain_of_h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l882_88276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_coefficient_terms_count_correct_l882_88230

noncomputable def rational_coefficient_terms_count : ℕ :=
  let x : ℝ := Real.sqrt (Real.sqrt 2)
  let y : ℝ := Real.sqrt 5
  let expansion := (x : ℝ) + (y : ℝ)
  let total_terms : ℕ := 989
  let power : ℕ := 988
  -- Number of terms with rational coefficients
  248

theorem rational_coefficient_terms_count_correct :
  rational_coefficient_terms_count = 248 := by
  -- Proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_coefficient_terms_count_correct_l882_88230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_existence_l882_88249

theorem matrix_inverse_existence :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℝ),
    (∃ (a b e f : ℝ),
      M = !![a, b, 0, 0;
             e, f, 0, 0;
             0, 0, 1, 0;
             0, 0, 0, 1]) ∧
    M * !![(-4 : ℝ), 3, 0, 0;
           6, (-8), 0, 0;
           0, 0, 2, 1;
           0, 0, 1, 2] = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_existence_l882_88249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_experiments_with_m_failures_l882_88252

/-- The expected number of experiments with exactly m failures -/
def expectedExperimentsWithMFailures (n m N : ℕ) (p : ℝ) : ℝ :=
  N * (n.choose m) * p^m * (1-p)^(n-m)

/-- Theorem: The expected number of experiments with exactly m failures
    is N * (n choose m) * p^m * (1-p)^(n-m) -/
theorem expected_experiments_with_m_failures
  (n m N : ℕ) (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let X := expectedExperimentsWithMFailures n m N p
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K,
    |((k.choose m : ℝ) / k - X / N)| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_experiments_with_m_failures_l882_88252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_one_l882_88260

noncomputable def f (x : ℝ) : ℝ := |⌊x⌋| - |⌊2 - x⌋|

theorem symmetry_about_one (x : ℝ) : f (1 + x) = f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_one_l882_88260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_vectors_l882_88299

/-- The area of a triangle formed by two vectors -/
noncomputable def triangleArea (a b : ℝ × ℝ × ℝ) : ℝ :=
  (1/2) * Real.sqrt (
    (a.2.1*b.2.2 - a.2.2*b.2.1)^2 + 
    (a.2.2*b.1 - a.1*b.2.2)^2 + 
    (a.1*b.2.1 - a.2.1*b.1)^2
  )

theorem triangle_area_specific_vectors :
  let a : ℝ × ℝ × ℝ := (2, -3, 1)
  let b : ℝ × ℝ × ℝ := (4, -1, 5)
  triangleArea a b = Real.sqrt 332 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_vectors_l882_88299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l882_88246

/-- Represents the graph structure with 5 vertices --/
structure Graph :=
  (edges : List (Fin 5 × Fin 5))

/-- Checks if two vertices are connected in the graph --/
def isConnected (g : Graph) (v1 v2 : Fin 5) : Bool :=
  (v1, v2) ∈ g.edges ∨ (v2, v1) ∈ g.edges

/-- Represents an assignment of natural numbers to the vertices --/
def Assignment := Fin 5 → ℕ

/-- Checks if an assignment is valid for a given n --/
def isValidAssignment (g : Graph) (n : ℕ) (a : Assignment) : Prop :=
  ∀ v1 v2 : Fin 5, v1 ≠ v2 →
    if isConnected g v1 v2
    then ∃ d > 1, d ∣ (a v1 + a v2) ∧ d ∣ n
    else Nat.Coprime (a v1 + a v2) n

/-- The main theorem statement --/
theorem smallest_valid_n : 
  ∀ g : Graph, 
    (∃ n : ℕ, ∃ a : Assignment, isValidAssignment g n a ∧ (∀ i j : Fin 5, i ≠ j → a i ≠ a j)) →
    (∀ n < 15, ¬∃ a : Assignment, isValidAssignment g n a ∧ (∀ i j : Fin 5, i ≠ j → a i ≠ a j)) ∧
    (∃ a : Assignment, isValidAssignment g 15 a ∧ (∀ i j : Fin 5, i ≠ j → a i ≠ a j)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l882_88246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l882_88243

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_values (ω φ : ℝ) :
  ω > 0 ∧ 
  0 ≤ φ ∧ φ ≤ π ∧
  (∀ x, f ω φ x = f ω φ (-x)) ∧ 
  (∀ x, f ω φ x = -f ω φ (3 * π / 2 - x)) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  (ω = 2/3 ∨ ω = 2) ∧ φ = π/2 :=
by
  sorry

#check function_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l882_88243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_phi_values_make_g_odd_l882_88265

open Real

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := sin (2*x + π/3)

/-- The shifted function g(x) -/
noncomputable def g (x φ : ℝ) : ℝ := sin (2*(x - φ) + π/3)

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The theorem stating that there are exactly two values of φ in (0, π) that make g odd -/
theorem two_phi_values_make_g_odd :
  ∃! (s : Finset ℝ), s.card = 2 ∧ (∀ φ ∈ s, 0 < φ ∧ φ < π) ∧
    (∀ φ, (0 < φ ∧ φ < π ∧ is_odd (g · φ)) ↔ φ ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_phi_values_make_g_odd_l882_88265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l882_88275

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (1.2 * L) * (0.8 * B) = 0.96 * (L * B) := by
  calc
    (1.2 * L) * (0.8 * B) = 1.2 * 0.8 * (L * B) := by ring
    _ = 0.96 * (L * B) := by norm_num

#check rectangle_area_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l882_88275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l882_88292

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Theorem: The equation of the trajectory of point P -/
theorem trajectory_equation (x y : ℝ) :
  distance x y 0 0 0 4 = 5 → x^2 + y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l882_88292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_value_l882_88242

/-- Configuration of three circles with a common tangent chord -/
structure CircleConfiguration where
  /-- Radius of the largest circle -/
  R : ℝ
  /-- Radius of the first smaller circle -/
  r₁ : ℝ
  /-- Radius of the second smaller circle -/
  r₂ : ℝ
  /-- The largest circle contains the two smaller circles -/
  contains : R > r₁ ∧ R > r₂
  /-- The two smaller circles are externally tangent to each other -/
  externally_tangent : r₁ + r₂ = R - r₁ - r₂

/-- The square of the length of the common external tangent chord -/
noncomputable def chord_length_squared (c : CircleConfiguration) : ℝ :=
  4 * (c.R^2 - ((c.r₁ * c.r₂) / (c.r₁ + c.r₂))^2)

/-- Theorem stating the square of the chord length for the given configuration -/
theorem chord_length_squared_value (c : CircleConfiguration)
  (h₁ : c.R = 14) (h₂ : c.r₁ = 4) (h₃ : c.r₂ = 9) :
  chord_length_squared c = 528 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_value_l882_88242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_circle_properties_l882_88267

-- Define the lines and point C
def l₁ (x : ℝ) : ℝ := 2 * x + 3
def l₂ (x : ℝ) : ℝ := x + 2
def C : ℝ × ℝ := (-1, 1)

-- Define the tangent line and circle
def tangentLine (x y : ℝ) : Prop := 3 * x + 4 * y + 4 = 0
def circleEq (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define the intersecting line
def intersectingLine (x y : ℝ) (t : ℝ) : Prop := x + y + t = 0

-- Helper function to calculate the area of a triangle
noncomputable def areaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem intersection_and_circle_properties :
  -- 1. C is the intersection of l₁ and l₂
  (l₁ C.1 = C.2 ∧ l₂ C.1 = C.2) ∧
  -- 2. The circle is tangent to the given line
  (∃ (x y : ℝ), tangentLine x y ∧ circleEq x y) ∧
  -- 3. Maximum area of triangle ABC is 1/2
  (∀ (t : ℝ), 
    ∀ (A B : ℝ × ℝ),
    (intersectingLine A.1 A.2 t ∧ intersectingLine B.1 B.2 t ∧ circleEq A.1 A.2 ∧ circleEq B.1 B.2) →
    areaTriangle C A B ≤ 1/2) ∧
  -- 4. The value of t that maximizes the area is either 1 or -1
  (∃ (A B : ℝ × ℝ), 
    (intersectingLine A.1 A.2 1 ∧ intersectingLine B.1 B.2 1 ∧ circleEq A.1 A.2 ∧ circleEq B.1 B.2 ∧
     areaTriangle C A B = 1/2) ∨
    (intersectingLine A.1 A.2 (-1) ∧ intersectingLine B.1 B.2 (-1) ∧ circleEq A.1 A.2 ∧ circleEq B.1 B.2 ∧
     areaTriangle C A B = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_circle_properties_l882_88267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_min_distance_equals_8_l882_88213

/-- Line l in parametric form -/
noncomputable def line_l (t θ : ℝ) : ℝ × ℝ := (t * Real.sin θ, 2 + t * Real.cos θ)

/-- Curve C in polar form -/
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.cos θ)^2 = 8 * Real.sin θ

/-- Distance between two points on line l -/
noncomputable def distance (θ : ℝ) (t₁ t₂ : ℝ) : ℝ :=
  Real.sqrt ((t₁ - t₂)^2 * (Real.sin θ)^2 + (t₁ - t₂)^2 * (Real.cos θ)^2)

theorem min_distance_line_curve (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi) :
  ∃ (t₁ t₂ : ℝ),
    curve_C (Real.sqrt ((t₁ * Real.sin θ)^2 + (2 + t₁ * Real.cos θ)^2)) θ ∧
    curve_C (Real.sqrt ((t₂ * Real.sin θ)^2 + (2 + t₂ * Real.cos θ)^2)) θ ∧
    ∀ (s₁ s₂ : ℝ),
      curve_C (Real.sqrt ((s₁ * Real.sin θ)^2 + (2 + s₁ * Real.cos θ)^2)) θ →
      curve_C (Real.sqrt ((s₂ * Real.sin θ)^2 + (2 + s₂ * Real.cos θ)^2)) θ →
      distance θ t₁ t₂ ≤ distance θ s₁ s₂ :=
by sorry

theorem min_distance_equals_8 :
  ∃ (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi) (t₁ t₂ : ℝ),
    curve_C (Real.sqrt ((t₁ * Real.sin θ)^2 + (2 + t₁ * Real.cos θ)^2)) θ ∧
    curve_C (Real.sqrt ((t₂ * Real.sin θ)^2 + (2 + t₂ * Real.cos θ)^2)) θ ∧
    distance θ t₁ t₂ = 8 ∧
    ∀ (φ : ℝ) (hφ : 0 < φ ∧ φ < Real.pi) (s₁ s₂ : ℝ),
      curve_C (Real.sqrt ((s₁ * Real.sin φ)^2 + (2 + s₁ * Real.cos φ)^2)) φ →
      curve_C (Real.sqrt ((s₂ * Real.sin φ)^2 + (2 + s₂ * Real.cos φ)^2)) φ →
      distance φ s₁ s₂ ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_min_distance_equals_8_l882_88213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l882_88253

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function
noncomputable def area (abc : Triangle) : ℝ := 
  1/2 * abc.a * abc.c * Real.sin abc.B

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.B - abc.a * Real.sin abc.C = 0) 
  (h2 : abc.a = 1) 
  (h3 : abc.c = 2) : 
  (∃ r : ℝ, abc.a * r = abc.b ∧ abc.b * r = abc.c) ∧ 
  (area abc = Real.sqrt 7 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l882_88253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faith_work_hours_l882_88295

/-- Faith's work schedule and pay --/
structure WorkSchedule where
  hourlyRate : ℚ
  daysPerWeek : ℕ
  overtimeHoursPerDay : ℚ
  weeklyEarnings : ℚ
  overtimeRate : ℚ

/-- Calculates the number of hours worked per day without overtime --/
noncomputable def hoursPerDayWithoutOvertime (w : WorkSchedule) : ℚ :=
  let regularEarnings := w.weeklyEarnings - (w.daysPerWeek : ℚ) * w.overtimeHoursPerDay * w.hourlyRate * w.overtimeRate
  regularEarnings / (w.hourlyRate * (w.daysPerWeek : ℚ))

/-- Theorem stating that Faith works 7 hours per day without overtime --/
theorem faith_work_hours (w : WorkSchedule) 
  (h1 : w.hourlyRate = 27/2)
  (h2 : w.daysPerWeek = 5)
  (h3 : w.overtimeHoursPerDay = 2)
  (h4 : w.weeklyEarnings = 675)
  (h5 : w.overtimeRate = 3/2) :
  hoursPerDayWithoutOvertime w = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faith_work_hours_l882_88295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_distribution_l882_88270

theorem strawberry_distribution (total_strawberries : ℝ) (num_girls : ℝ) 
  (h1 : total_strawberries = 53.5) 
  (h2 : num_girls = 8.5) : 
  ∃ (strawberries_per_girl : ℝ), 
    (strawberries_per_girl ≥ 6.28 ∧ strawberries_per_girl ≤ 6.30) ∧ 
    strawberries_per_girl = total_strawberries / num_girls :=
by
  let strawberries_per_girl := total_strawberries / num_girls
  use strawberries_per_girl
  constructor
  · sorry -- Proof that 6.28 ≤ strawberries_per_girl ≤ 6.30
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_distribution_l882_88270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_f_9_l882_88200

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the condition that g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Define the given relationship between f⁻¹ and g
axiom f_inv_g_eq (x : ℝ) : f⁻¹ (g x) = x^4 - x^2 + 1

-- State the theorem to be proved
theorem g_inv_f_9 : g⁻¹ (f 9) = Real.sqrt ((1 + Real.sqrt 33) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_f_9_l882_88200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_florida_migration_rate_l882_88206

/-- The average number of people moving to Florida per hour over five days -/
def average_people_per_hour (total_people : ℕ) (days : ℕ) : ℕ :=
  (total_people + (days * 24) / 2) / (days * 24)

/-- Theorem: Given 3000 people moving to Florida over 5 days, 
    the average number of people moving per hour (rounded to the nearest whole number) is 25 -/
theorem florida_migration_rate : average_people_per_hour 3000 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_florida_migration_rate_l882_88206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_range_l882_88237

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + a * x + 1)

-- Define the property that the function's domain is all real numbers
def hasDomainR (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Theorem statement
theorem domain_implies_a_range :
  ∀ a : ℝ, hasDomainR a ↔ 0 ≤ a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_range_l882_88237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_l882_88201

/-- The convergence and sum of an infinite geometric series -/
theorem geometric_series_convergence (a : ℝ) :
  let series := (λ n : ℕ => a^n)
  (∃ s, HasSum series s) ↔ |a| < 1 ∧
  (∀ s, HasSum series s → s = 1 / (1 - a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_l882_88201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_tilde_20_l882_88250

def F_tilde : ℕ → ℕ
  | 0 => 13  -- Define base case for 0
  | 1 => 13  -- Define base case for 1
  | 2 => 13  -- Define base case for 2
  | 3 => 13  -- Original base case
  | n+4 => F_tilde (n+3) + 4 * (n + 5)  -- Recursive case for n ≥ 4

theorem F_tilde_20 : F_tilde 20 = 1661 := by
  -- Proof goes here
  sorry

#eval F_tilde 20  -- This will evaluate F_tilde 20 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_tilde_20_l882_88250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_trick_sum_theorem_l882_88258

/-- Represents a card value in the deck -/
def CardValue := Fin 13

/-- A stack of cards -/
structure Stack where
  bottomCard : CardValue
  size : Nat
  size_eq : size = 13 - bottomCard.val

/-- The card trick setup -/
structure CardTrick where
  stacks : List Stack
  setAsideCards : Nat
  total_cards : (stacks.map (λ s => s.size)).sum + setAsideCards = 52

/-- The main theorem stating the relationship between the sum of bottom cards,
    number of stacks, and set aside cards -/
theorem card_trick_sum_theorem (ct : CardTrick) :
  (ct.stacks.map (λ s => s.bottomCard.val)).sum =
    13 * (ct.stacks.length - 4) + ct.setAsideCards := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_trick_sum_theorem_l882_88258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_in_five_throws_l882_88234

/-- The number of girls in the circle -/
def n : ℕ := 15

/-- The number of girls skipped in each throw -/
def skip : ℕ := 5

/-- The function that determines the next position after a throw -/
def next (x : ℕ) : ℕ := (x + skip + 1) % n

/-- The sequence of positions starting from 1 -/
def ballSequence : ℕ → ℕ
  | 0 => 1
  | k + 1 => next (ballSequence k)

/-- The theorem stating that it takes 5 throws to return to the starting position -/
theorem ball_returns_in_five_throws :
  ballSequence 5 = 1 ∧ ∀ k < 5, ballSequence k ≠ 1 := by
  sorry

#eval [ballSequence 0, ballSequence 1, ballSequence 2, ballSequence 3, ballSequence 4, ballSequence 5]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_in_five_throws_l882_88234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l882_88256

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + x + 1/2

theorem min_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m)) ∧ 
  (let m := 1; 
   (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = m →
    2 * (a^3 + b^3 + c^3) ≥ a*b + b*c + c*a - 3*a*b*c)) :=
by
  sorry

#check min_value_and_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l882_88256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_mass_calculation_l882_88204

/-- The mass of a conical pile of sand -/
noncomputable def sand_pile_mass (base_area : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  density * (1/3 * base_area * height)

/-- Theorem: The mass of the specific conical sand pile is 94.2 tons -/
theorem sand_pile_mass_calculation :
  sand_pile_mass 62.8 3 1.5 = 94.2 := by
  -- Unfold the definition of sand_pile_mass
  unfold sand_pile_mass
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_mass_calculation_l882_88204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_properties_l882_88262

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_area_and_function_properties 
  (abc : Triangle) 
  (h1 : f (abc.A / 2 - Real.pi / 6) = Real.sqrt 3)
  (h2 : abc.a = 7)
  (h3 : Real.sin abc.B + Real.sin abc.C = 13 * Real.sqrt 3 / 14) :
  -- Conclusion 1: Smallest positive period
  (∃ T : ℝ, T > 0 ∧ T = Real.pi / 3 ∧ 
    ∀ x : ℝ, f (-3 * x) + 1 = f (-3 * (x + T)) + 1 ∧
    ∀ S : ℝ, S > 0 → (∀ x : ℝ, f (-3 * x) + 1 = f (-3 * (x + S)) + 1) → T ≤ S) ∧
  -- Conclusion 2: Intervals of monotonic decrease
  (∀ k : ℤ, ∀ x : ℝ, 
    k * Real.pi / 3 - Real.pi / 36 ≤ x ∧ x ≤ k * Real.pi / 3 + 5 * Real.pi / 36 →
    ∀ y : ℝ, x < y ∧ y ≤ k * Real.pi / 3 + 5 * Real.pi / 36 →
    f (-3 * y) + 1 < f (-3 * x) + 1) ∧
  -- Conclusion 3: Area of triangle ABC
  (1/2 * abc.b * abc.c * Real.sin abc.A = 10 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_properties_l882_88262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_l882_88229

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - Real.exp x

-- State the theorem
theorem f_increasing_on_neg_reals :
  ∀ x y : ℝ, x < y → y < 0 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_l882_88229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l882_88268

theorem number_of_factors (n : ℕ) (h : n = 4^5 * 5^5 * 7^3) :
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 264 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l882_88268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_distance_squared_l882_88248

/-- The square of the distance between two points on a parabola that are equidistant from the origin -/
theorem parabola_points_distance_squared : 
  ∀ (P Q : ℝ × ℝ),
  let f := λ (x : ℝ) => 3*x^2 - 5*x + 2
  f P.1 = P.2 →                              -- P is on the parabola
  f Q.1 = Q.2 →                              -- Q is on the parabola
  P.1^2 + P.2^2 = Q.1^2 + Q.2^2 →             -- P and Q are equidistant from origin
  P.1 + Q.1 = 0 ∧ P.2 + Q.2 = 0 →             -- Origin is midpoint of PQ
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 244/9       -- Square of distance between P and Q
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_distance_squared_l882_88248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_20_over_3_l882_88224

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x ≤ 2 then x + 1
  else 0  -- This else case is added to make f total

-- State the theorem
theorem integral_f_equals_20_over_3 :
  ∫ x in (-2 : ℝ)..2, f x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_20_over_3_l882_88224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_water_tank_l882_88280

/-- Represents the dimensions and costs of a rectangular water tank -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  baseCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank -/
noncomputable def totalCost (tank : WaterTank) (x : ℝ) : ℝ :=
  tank.wallCost * tank.depth * (2 * x + 2 * tank.volume / (x * tank.depth)) + tank.baseCost * tank.volume / tank.depth

/-- Theorem stating the minimum cost of constructing the water tank -/
theorem min_cost_water_tank (tank : WaterTank) (h1 : tank.volume = 0.18)
    (h2 : tank.depth = 0.5) (h3 : tank.baseCost = 400) (h4 : tank.wallCost = 100) :
    ∃ (x : ℝ), ∀ (y : ℝ), y > 0 → totalCost tank x ≤ totalCost tank y ∧ totalCost tank x = 264 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_water_tank_l882_88280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pure_imaginary_m_l882_88298

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 - 1)

/-- Theorem stating that m = -2 is the only real value that makes z a pure imaginary number -/
theorem unique_pure_imaginary_m :
  ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pure_imaginary_m_l882_88298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_travel_distance_is_360_l882_88219

/-- Represents a car in the problem -/
structure Car where
  fuel : ℕ  -- Amount of fuel in barrels
  distance : ℕ  -- Distance traveled in km

/-- The problem setup -/
structure ProblemSetup where
  maxFuel : ℕ := 24  -- Maximum fuel capacity in barrels
  fuelEfficiency : ℕ := 60  -- km per barrel of fuel

/-- Calculates the maximum distance one car can travel away from the starting point -/
def maxTravelDistance (setup : ProblemSetup) : ℕ :=
  setup.maxFuel * setup.fuelEfficiency / 4

/-- Theorem stating that the maximum travel distance is 360 km -/
theorem max_travel_distance_is_360 (setup : ProblemSetup) :
  maxTravelDistance setup = 360 := by
  sorry

#eval maxTravelDistance { maxFuel := 24, fuelEfficiency := 60 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_travel_distance_is_360_l882_88219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_ratio_approaches_one_l882_88214

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.side_frame_width,
   fp.painting_height + 6 * fp.side_frame_width)

/-- Calculates the area of the frame -/
def frameArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h - fp.painting_width * fp.painting_height

/-- Theorem: The ratio of smaller to larger dimension approaches 1 for the given conditions -/
theorem framed_painting_ratio_approaches_one :
  ∃ (fp : FramedPainting),
    fp.painting_width = 30 ∧
    fp.painting_height = 20 ∧
    frameArea fp = fp.painting_width * fp.painting_height →
    let (w, h) := framedDimensions fp
    abs ((min w h) / (max w h) - 1) < 0.1 := by
  sorry

#check framed_painting_ratio_approaches_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_ratio_approaches_one_l882_88214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_l882_88259

/-- The quadratic equation to be solved -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 4*x - 8 = 0

/-- The first solution to the quadratic equation -/
noncomputable def x₁ : ℝ := 2 + 2 * Real.sqrt 3

/-- The second solution to the quadratic equation -/
noncomputable def x₂ : ℝ := 2 - 2 * Real.sqrt 3

/-- Theorem stating that x₁ and x₂ are solutions to the quadratic equation -/
theorem quadratic_solutions : quadratic_equation x₁ ∧ quadratic_equation x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solutions_l882_88259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_9_l882_88244

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1989^1989  -- Added case for 0
  | 1 => 1989^1989
  | (n+1) => sum_of_digits (a n)

/-- Theorem stating that a_5 equals 9 -/
theorem a_5_equals_9 : a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_9_l882_88244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l882_88274

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / (x - 1)
noncomputable def g (k : ℕ+) (x : ℝ) : ℝ := k / x

-- State the theorem
theorem max_k_value :
  ∃ (k_max : ℕ+),
    (∀ (c : ℝ), c > 1 →
      ∃ (a b : ℝ), 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g k_max b) ∧
    (∀ (k : ℕ+), k > k_max →
      ∃ (c : ℝ), c > 1 ∧
        ∀ (a b : ℝ), 0 < a → a < b → b < c →
          ¬(f c = f a ∧ f a = g k b)) ∧
    k_max = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l882_88274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_playground_side_length_l882_88212

/-- Represents a square playground -/
structure Playground where
  perimeter : ℕ
  sideLength : ℕ

/-- Calculates the perimeter of a square playground given the total distance walked and number of laps -/
def calculatePerimeter (totalDistance : ℕ) (laps : ℕ) : ℕ :=
  totalDistance / laps

/-- Theorem: For a square playground with a perimeter of 200 meters, its side length is 50 meters,
    given that a person walks 400 meters in 2 laps around the playground -/
theorem square_playground_side_length 
  (totalDistance : ℕ) 
  (laps : ℕ) 
  (playground : Playground) 
  (h1 : totalDistance = 400)
  (h2 : laps = 2)
  (h3 : playground.perimeter = calculatePerimeter totalDistance laps)
  (h4 : playground.perimeter = 4 * playground.sideLength) : 
  playground.sideLength = 50 := by
  sorry

#eval calculatePerimeter 400 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_playground_side_length_l882_88212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l882_88289

theorem inequality_solution_set (x : ℝ) : 
  (2 : ℝ)^(2*x) ≤ 3 * (2 : ℝ)^(x + Real.sqrt x) + 4 * (2 : ℝ)^(2 * Real.sqrt x) ↔ (0 ≤ x ∧ x ≤ 4) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l882_88289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l882_88283

theorem factorial_simplification :
  (11 * Nat.factorial 10) / (Nat.factorial 9 + 2 * Nat.factorial 8) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l882_88283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l882_88211

/-- Curve C in parametric form -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)

/-- Point M in polar coordinates -/
noncomputable def point_M : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

/-- Line l passing through M -/
noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 1 + t * Real.sin α)

/-- Theorem stating the length of segment AB -/
theorem length_of_AB (θ₁ θ₂ α t₁ t₂ : ℝ) : 
  curve_C θ₁ = line_l α t₁ ∧ 
  curve_C θ₂ = line_l α t₂ ∧ 
  abs (t₁ - 1) = 2 * abs (t₂ - 1) → 
  abs (t₁ - t₂) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l882_88211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_problem_l882_88290

/-- The distance between points A and B in kilometers. -/
axiom distance : ℝ

/-- The speed of the first pedestrian in km/h. -/
axiom speed1 : ℝ

/-- The speed of the second pedestrian in km/h. -/
axiom speed2 : ℝ

/-- The remaining distance for the second pedestrian when the first finishes. -/
axiom remaining_distance : ℝ

/-- The theorem stating the conditions and the result to be proved. -/
theorem pedestrian_problem (h1 : distance / 2 * (speed2 / speed1) + 24 = distance)
                           (h2 : distance / 2 * (speed1 / speed2) + 15 = distance)
                           (h3 : distance * (speed2 / speed1) + remaining_distance = distance) :
  remaining_distance = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_problem_l882_88290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_crumbs_l882_88296

/-- The number of crumbs carried by three ants given specific conditions -/
theorem ant_crumbs (c : ℝ) : 
  let arthur_crumbs := c
  let amy_trips := 2
  let amy_crumbs_per_trip := 1.5
  let andy_trips := 3
  let andy_crumbs_per_trip := 2 * amy_crumbs_per_trip
  let amy_crumbs := amy_trips * amy_crumbs_per_trip * arthur_crumbs
  let andy_crumbs := andy_trips * andy_crumbs_per_trip * arthur_crumbs
  arthur_crumbs + amy_crumbs + andy_crumbs = 13 * c := by
  -- Proof goes here
  sorry

#check ant_crumbs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_crumbs_l882_88296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_54_81_l882_88217

def divisors (n : ℕ) : Finset ℕ := Finset.filter (λ d => n % d = 0) (Finset.range (n + 1))

theorem common_divisors_54_81 : Finset.card (divisors 54 ∩ divisors 81) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_54_81_l882_88217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l882_88254

noncomputable def line_x (t : ℝ) : ℝ := -1 + t * Real.sin (40 * Real.pi / 180)
noncomputable def line_y (t : ℝ) : ℝ := 3 + t * Real.cos (40 * Real.pi / 180)

theorem line_slope :
  (∀ t₁ t₂ : ℝ, t₁ ≠ t₂ →
    (line_y t₂ - line_y t₁) / (line_x t₂ - line_x t₁) = Real.tan (50 * Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l882_88254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_square_diagonal_l882_88261

-- Define the square ABCD
def square_side : ℝ := 10

-- Define the position of point E
def DE_length : ℝ := 3

-- Theorem statement
theorem fold_square_diagonal : 
  ∃ x : ℝ, x = 91 / 20 ∧ 
  (square_side = 10 → 
   DE_length = 3 → 
   (10 - x)^2 = x^2 + 3^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_square_diagonal_l882_88261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_covering_disks_for_circle_6_l882_88238

/-- The minimum diameter of five equal-sized disks that can completely cover a circle -/
noncomputable def min_covering_disk_diameter (circle_diameter : ℝ) : ℝ :=
  let R := circle_diameter / 2
  let r := (4 * R) / (5 + Real.sqrt 5)
  2 * (⌊(2 * r + 0.25) / 0.5⌋ * 0.5)

/-- Theorem stating the minimum diameter of five disks covering a circle of diameter 6 -/
theorem min_covering_disks_for_circle_6 :
  min_covering_disk_diameter 6 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_covering_disks_for_circle_6_l882_88238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sufficient_not_necessary_prove_parallel_sufficient_not_necessary_l882_88241

-- Define the basic structures
structure Line where
  mk :: -- Add a constructor

structure Plane where
  mk :: -- Add a constructor

-- Define the relationships
def parallel (l1 l2 : Line) : Prop := sorry

def inPlane (l : Line) (p : Plane) : Prop := sorry

def atLeastParallelToOneLineInPlane (l : Line) (p : Plane) : Prop :=
  ∃ (l' : Line), inPlane l' p ∧ parallel l l'

-- State the theorem
theorem parallel_sufficient_not_necessary 
  (a : Line) (β : Line) (plane_β : Plane) : Prop :=
  -- Sufficient condition
  (parallel a β → atLeastParallelToOneLineInPlane a plane_β) ∧
  -- Not necessary condition
  ¬(atLeastParallelToOneLineInPlane a plane_β → parallel a β)

-- The proof
theorem prove_parallel_sufficient_not_necessary :
  ∀ (a : Line) (β : Line) (plane_β : Plane), 
  parallel_sufficient_not_necessary a β plane_β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sufficient_not_necessary_prove_parallel_sufficient_not_necessary_l882_88241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pit_no_replant_probability_l882_88266

/-- The probability of a single seed germinating -/
noncomputable def seed_germination_prob : ℝ := 1/2

/-- The number of seeds in each pit -/
def seeds_per_pit : ℕ := 2

/-- The total number of pits -/
def total_pits : ℕ := 3

/-- The probability that a pit does not require replanting -/
noncomputable def pit_no_replant_prob : ℝ := 1 - (1 - seed_germination_prob) ^ seeds_per_pit

/-- The probability that exactly one pit out of three does not require replanting -/
theorem one_pit_no_replant_probability : 
  (total_pits.choose 1 : ℝ) * pit_no_replant_prob * (1 - pit_no_replant_prob) ^ (total_pits - 1) = 9/64 := by
  sorry

#eval (3 : ℕ).choose 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pit_no_replant_probability_l882_88266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_right_angle_l882_88286

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse (a b c : ℝ) : Type :=
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_relation : a^2 - c^2 = b^2)

/-- A point on the ellipse. -/
structure PointOnEllipse (E : Ellipse a b c) : Type :=
  (x y : ℝ)
  (h_on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)

/-- The focus of the ellipse. -/
noncomputable def focus (E : Ellipse a b c) : ℝ × ℝ := (c, 0)

/-- The tangent line at a point on the ellipse. -/
noncomputable def tangent_line (E : Ellipse a b c) (P : PointOnEllipse E) : ℝ → ℝ :=
  λ x ↦ P.y * b^2 / (a^2 * P.x) * (x - P.x) + P.y

/-- The intersection points of the tangent with the vertex tangents. -/
noncomputable def intersection_points (E : Ellipse a b c) (P : PointOnEllipse E) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-a, b^2 / P.y * (1 + P.x / a)), (a, b^2 / P.y * (1 - P.x / a)))

/-- The theorem to be proved. -/
theorem tangent_segment_right_angle (a b c : ℝ) (E : Ellipse a b c) 
  (P : PointOnEllipse E) : 
  let (M, N) := intersection_points E P
  let F := focus E
  let m := (M.2 - F.2) / (M.1 - F.1)
  let n := (N.2 - F.2) / (N.1 - F.1)
  m * n = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_right_angle_l882_88286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hill_ocean_depth_l882_88279

-- Define the hill's properties
noncomputable def total_height : ℝ := 5000
noncomputable def volume_ratio_above_water : ℝ := 1/5

-- Define the function to calculate the depth
noncomputable def ocean_depth (h : ℝ) (r : ℝ) : ℝ :=
  h * (1 - (1 - r)^(1/3))

-- Theorem statement
theorem hill_ocean_depth :
  ∃ ε > 0, |ocean_depth total_height volume_ratio_above_water - 347| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hill_ocean_depth_l882_88279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_eq_x_sgn_x_l882_88235

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

-- Theorem statement
theorem abs_eq_x_sgn_x (x : ℝ) : |x| = x * sgn x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_eq_x_sgn_x_l882_88235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_of_nested_sqrt_l882_88257

noncomputable def x : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_of_nested_sqrt :
  x^4 = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_of_nested_sqrt_l882_88257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l882_88277

/-- The time taken for a train to pass a pole, given its length and the time it takes to pass a platform of known length. -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_passing_time : ℝ) 
  (h1 : train_length = 500) 
  (h2 : platform_length = 500) 
  (h3 : platform_passing_time = 100) : 
  train_length / (train_length + platform_length) * platform_passing_time = 50 := by
  -- Replace the entire proof with 'sorry'
  sorry

#check train_passing_pole_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l882_88277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_always_form_polygon_l882_88225

/-- A sequence of stick lengths following powers of 2 -/
def stickLengths : Fin 100 → ℕ
  | ⟨n, _⟩ => 2^n

/-- Condition for forming a polygon: sum of any n-1 sides > length of remaining side -/
def canFormPolygon (lengths : List ℕ) : Prop :=
  ∀ i, i < lengths.length → (lengths.sum - lengths.getD i 0) > lengths.getD i 0

/-- Main theorem: It's impossible to always select sticks to form a polygon -/
theorem impossible_to_always_form_polygon :
  ¬ ∃ (subset : List (Fin 100)), subset.length ≥ 3 ∧ canFormPolygon (subset.map stickLengths) := by
  sorry

#check impossible_to_always_form_polygon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_always_form_polygon_l882_88225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_knights_at_table_l882_88282

/-- Represents the two types of statements made by islanders -/
inductive Statement
| bothLiars : Statement
| oneLiar : Statement

/-- Represents the type of islander (knight or liar) -/
inductive IslanderType
| knight : IslanderType
| liar : IslanderType

instance : DecidableEq Statement :=
  fun a b => match a, b with
  | Statement.bothLiars, Statement.bothLiars => isTrue rfl
  | Statement.oneLiar, Statement.oneLiar => isTrue rfl
  | Statement.bothLiars, Statement.oneLiar => isFalse (fun h => Statement.noConfusion h)
  | Statement.oneLiar, Statement.bothLiars => isFalse (fun h => Statement.noConfusion h)

instance : DecidableEq IslanderType :=
  fun a b => match a, b with
  | IslanderType.knight, IslanderType.knight => isTrue rfl
  | IslanderType.liar, IslanderType.liar => isTrue rfl
  | IslanderType.knight, IslanderType.liar => isFalse (fun h => IslanderType.noConfusion h)
  | IslanderType.liar, IslanderType.knight => isFalse (fun h => IslanderType.noConfusion h)

/-- A function that determines if a statement is consistent with the islander type and their neighbors -/
def isConsistent (s : Statement) (t : IslanderType) (left : IslanderType) (right : IslanderType) : Prop :=
  match s, t with
  | Statement.bothLiars, IslanderType.knight => left = IslanderType.liar ∧ right = IslanderType.liar
  | Statement.bothLiars, IslanderType.liar => left = IslanderType.knight ∨ right = IslanderType.knight
  | Statement.oneLiar, IslanderType.knight => (left = IslanderType.liar ∧ right = IslanderType.knight) ∨ 
                                              (left = IslanderType.knight ∧ right = IslanderType.liar)
  | Statement.oneLiar, IslanderType.liar => (left = IslanderType.liar ∧ right = IslanderType.liar) ∨ 
                                            (left = IslanderType.knight ∧ right = IslanderType.knight)

/-- The main theorem stating the maximum number of knights possible -/
theorem max_knights_at_table (islanders : Fin 100 → IslanderType) 
  (statements : Fin 100 → Statement)
  (h_half_both_liars : (Finset.filter (fun i => statements i = Statement.bothLiars) Finset.univ).card = 50)
  (h_half_one_liar : (Finset.filter (fun i => statements i = Statement.oneLiar) Finset.univ).card = 50)
  (h_consistent : ∀ i : Fin 100, isConsistent (statements i) (islanders i) 
                                  (islanders ((i + 99) % 100)) (islanders ((i + 1) % 100))) :
  (Finset.filter (fun i => islanders i = IslanderType.knight) Finset.univ).card ≤ 67 := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_knights_at_table_l882_88282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_imply_a_and_b_values_l882_88264

/-- The function f(x) with parameters a and b -/
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a b x : ℝ) : ℝ := a / x + 2 * b * x + 1

theorem extreme_values_imply_a_and_b_values :
  ∀ a b : ℝ,
  (f_deriv a b 1 = 0 ∧ f_deriv a b 2 = 0) →
  (a = -2/3 ∧ b = -1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_imply_a_and_b_values_l882_88264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l882_88208

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_positive : a 1 > 0
  sum_2005_2006_positive : a 2005 + a 2006 > 0
  product_2005_2006_negative : a 2005 * a 2006 < 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The theorem stating the maximum n for which the sum is positive -/
theorem max_positive_sum (seq : ArithmeticSequence) :
  (∀ n : ℕ, n ≤ 4010 → sum_n seq n > 0) ∧
  (sum_n seq 4011 ≤ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l882_88208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_intersection_segment_length_l882_88240

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 4 * Real.sin θ)

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 + (1/2) * t, 2 + (Real.sqrt 3 / 2) * t)

theorem curve_C_cartesian_equation :
  ∀ x y : ℝ, (∃ θ : ℝ, curve_C θ = (x, y)) ↔ x^2 + y^2 = 16 := by sorry

theorem intersection_segment_length :
  let intersections := {p : ℝ × ℝ | ∃ t : ℝ, line_l t = p ∧ p.1^2 + p.2^2 = 16}
  ∃ A B : ℝ × ℝ, A ∈ intersections ∧ B ∈ intersections ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_intersection_segment_length_l882_88240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l882_88226

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 5/4 * x - 5

-- Theorem statement
theorem zero_point_in_interval :
  ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = 0 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_continuous : Continuous f := by
  sorry

lemma f_at_3_negative : f 3 < 0 := by
  sorry

lemma f_at_4_positive : f 4 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l882_88226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_min_distance_to_point_l882_88273

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

theorem curve_min_distance_to_point (a : ℝ) :
  a > 0 →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → 
    let (x, y) := curve θ
    (x - a)^2 + y^2 ≥ (3/4)^2) →
  (∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    let (x, y) := curve θ
    (x - a)^2 + y^2 = (3/4)^2) →
  a = Real.sqrt 21 / 4 ∨ a = 11/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_min_distance_to_point_l882_88273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_odd_functions_l882_88203

-- Define the functions
noncomputable def f₁ : ℝ → ℝ := λ x => x^3
noncomputable def f₂ : ℝ → ℝ := λ x => Real.exp (x * Real.log 2)
noncomputable def f₃ : ℝ → ℝ := λ x => x^2 + 1
noncomputable def f₄ : ℝ → ℝ := λ x => 2 * Real.sin x

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem exactly_two_odd_functions :
  (IsOdd f₁ ∧ ¬IsOdd f₂ ∧ ¬IsOdd f₃ ∧ IsOdd f₄) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_odd_functions_l882_88203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_equation_l882_88294

open Vector

variable (a b c : ℝ × ℝ × ℝ)

def cross (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_equation (h1 : cross a b = (2, -3, 1)) (h2 : cross a c = (1, 1, -1)) :
  (2 : ℝ) • (cross a ((4 : ℝ) • b + (3 : ℝ) • c)) = (22, -18, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_equation_l882_88294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l882_88221

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- State the theorem
theorem power_function_value (α : ℝ) :
  power_function α 4 = 2 → power_function α (1/16) = 1/4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l882_88221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_3_l882_88291

/-- Triangle ABC with centroid G -/
structure Triangle :=
  (A B C G : ℝ × ℝ)
  (is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3))

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Length of a vector -/
noncomputable def length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Angle B in radians -/
noncomputable def angle_B (t : Triangle) : ℝ :=
  Real.arccos ((length (vector t.A t.C))^2 + (length (vector t.A t.B))^2 - (length (vector t.B t.C))^2) 
               / (2 * length (vector t.A t.C) * length (vector t.A t.B))

/-- Main theorem -/
theorem angle_B_is_pi_over_3 (t : Triangle) :
  (length (vector t.B t.C) / 5) • (vector t.G t.A) + 
  (length (vector t.C t.A) / 7) • (vector t.G t.B) + 
  (length (vector t.A t.B) / 8) • (vector t.G t.C) = (0, 0) →
  angle_B t = π / 3 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_3_l882_88291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_and_tangent_range_l882_88278

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/2 * x^2 - 2*x + 5

noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - x - 2

theorem monotonic_intervals_and_tangent_range :
  (∀ x < -2/3, (f' x > 0)) ∧
  (∀ x ∈ Set.Ioo (-2/3) 1, (f' x < 0)) ∧
  (∀ x > 1, (f' x > 0)) ∧
  (∃ a_min a_max : ℝ, a_min = 5 ∧ a_max = 5 + 1/216 ∧
    ∀ a ∈ Set.Ioo a_min a_max, ∃ x₁ x₂ x₃ : ℝ,
      x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
      a = f x₁ + f' x₁ * (-x₁) ∧
      a = f x₂ + f' x₂ * (-x₂) ∧
      a = f x₃ + f' x₃ * (-x₃)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_and_tangent_range_l882_88278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_range_hyperbola_asymptotes_l882_88223

/-- Represents a hyperbola of the form x² - y²/m = 1 -/
structure Hyperbola where
  m : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.m)

theorem hyperbola_m_range (h : Hyperbola) :
  eccentricity h < Real.sqrt 2 → 0 < h.m ∧ h.m < 1 := by sorry

theorem hyperbola_asymptotes (h : Hyperbola) (h_m : h.m = 2) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧
    (∀ (x y : ℝ), y = k * x ∨ y = -k * x → x^2 - y^2 / h.m = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_range_hyperbola_asymptotes_l882_88223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l882_88207

/-- The curve defined by xy - x - y = 1 -/
def curve (x y : ℝ) : Prop := x * y - x - y = 1

/-- The distance from the origin to a point (x, y) -/
noncomputable def distanceFromOrigin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The minimum distance from the origin to any point on the curve -/
theorem min_distance_to_curve : 
  ∃ (min : ℝ), min = 2 - Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), curve x y → distanceFromOrigin x y ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l882_88207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l882_88227

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 1) ∧
  (∃ t : ℝ, t = -2 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x + t ≤ 2 ∧
    ∀ s : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x + s ≤ 2) → s ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l882_88227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_equals_1_l882_88263

def b : ℕ → ℚ
  | 0 => 3  -- We define b(0) as 3 to match b_1 in the original problem
  | 1 => 4  -- This corresponds to b_2 in the original problem
  | n + 2 => (b (n + 1) + 1) / b n

theorem b_2023_equals_1 : b 2022 = 1 := by
  sorry

#eval b 2022  -- This will evaluate b 2022, which corresponds to b_2023 in the original problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_equals_1_l882_88263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l882_88209

theorem equation_solution : ∃! y : ℚ, (16 : ℝ) ^ (3 * (y : ℝ) - 4) = (1 / 4 : ℝ) ^ ((y : ℝ) + 3) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l882_88209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_difference_2025_l882_88239

theorem smallest_factor_difference_2025 :
  ∃ (a b : ℕ), (a * b = 2025) ∧
    (∀ (c d : ℕ), c * d = 2025 → |Int.ofNat c - Int.ofNat d| ≥ |Int.ofNat a - Int.ofNat b|) ∧
    |Int.ofNat a - Int.ofNat b| = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_difference_2025_l882_88239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_of_inertia_relation_l882_88236

/-- Given a system of points with center of mass O and total mass m,
    prove that the moment of inertia I_X with respect to an arbitrary point X
    is equal to the sum of the moment of inertia I_O with respect to the center of mass O
    and the product of the total mass m and the square of the distance XO between X and O. -/
theorem moment_of_inertia_relation
  (O X : EuclideanSpace ℝ (Fin 3)) -- Center of mass and arbitrary point in 3D space
  (m : ℝ) -- Total mass of the system
  (I_O : ℝ) -- Moment of inertia with respect to O
  (I_X : ℝ) -- Moment of inertia with respect to X
  (h1 : m > 0) -- Assumption: total mass is positive
  (h2 : I_O ≥ 0) -- Assumption: moment of inertia is non-negative
  (h3 : I_X ≥ 0) -- Assumption: moment of inertia is non-negative
  : I_X = I_O + m * ‖X - O‖^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moment_of_inertia_relation_l882_88236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l882_88287

-- Define the parallel lines
variable (l m : Line Plane)

-- Define the angles
variable (A B C : Angle Plane)

-- State the theorem
theorem angle_C_measure
  (h1 : l.parallel m)
  (h2 : A.measure = 120)
  (h3 : B.measure = 150)
  (h4 : C.isAngleBetween l m)
  : C.measure = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l882_88287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l882_88272

theorem cos_2theta_value (θ : ℝ) (h : ∑' n, Real.cos θ^(2*n) = 9) : Real.cos (2*θ) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2theta_value_l882_88272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_equivalence_l882_88202

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 1 ∧ p x₀) ↔ (∀ x : ℝ, x ≥ 1 → ¬ p x) :=
by sorry

-- Define the specific property for this problem
def property (x : ℝ) : Prop := (Real.log 3 / Real.log 2) ^ x ≥ 1

-- Theorem stating the equivalence of the negation
theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 1 ∧ property x₀) ↔ (∀ x : ℝ, x ≥ 1 → (Real.log 3 / Real.log 2) ^ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_equivalence_l882_88202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_outfit_combinations_l882_88297

/-- Represents the colors available for clothing items -/
inductive Color
| Tan | Black | Blue | Gray | Green | White | Yellow
deriving Repr, Fintype, DecidableEq

/-- Represents a clothing item -/
structure ClothingItem where
  color : Color
deriving Repr, Fintype, DecidableEq

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem
deriving Repr, Fintype, DecidableEq

def is_valid_outfit (o : Outfit) : Prop :=
  o.shirt.color ≠ o.hat.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 6

def pants_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Green]
def shirt_hat_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Green, Color.White, Color.Yellow]

instance : DecidablePred is_valid_outfit :=
  fun o => show Decidable (o.shirt.color ≠ o.hat.color) from inferInstance

theorem valid_outfit_combinations :
  (Finset.filter is_valid_outfit (Finset.univ : Finset Outfit)).card = 205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_outfit_combinations_l882_88297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_solution_system_l882_88233

theorem integral_solution_system (x y z : ℕ) : 
  (x ^ z = y ^ (z - 1)) ∧ 
  ((2 : ℕ) ^ z = 8 * (2 : ℕ) ^ x) ∧ 
  (x ^ 2 + y ^ 2 + z ^ 2 = 72) ↔ 
  (x = 3 ∧ y = 3 ∧ z = 6) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_solution_system_l882_88233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_not_necessarily_positive_l882_88293

def sequence_property (s : Fin 20 → ℝ) : Prop :=
  ∀ i : Fin 18, s i + s (i.succ) + s (i.succ.succ) > 0

theorem sequence_sum_not_necessarily_positive :
  ∃ s : Fin 20 → ℝ, sequence_property s ∧ ¬(Finset.sum Finset.univ s > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_not_necessarily_positive_l882_88293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l882_88271

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 6 * x^2 - 5 * x + 18 = 0 ↔ x = Complex.ofReal a + Complex.I * b ∨ x = Complex.ofReal a - Complex.I * b) →
  a + b^2 = 467 / 144 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l882_88271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_l882_88245

/-- Represents a pairing of left and right shoes -/
def ShoePairing := Fin 15 → Fin 15

/-- Checks if a pairing satisfies the given condition -/
def valid_pairing (p : ShoePairing) : Prop :=
  ∀ k : ℕ, k < 7 → ∀ s : Finset (Fin 15), s.card = k →
    (∃ i ∈ s, p i ∉ s) ∨ (∃ i ∉ s, p i ∈ s)

/-- The total number of possible pairings -/
def total_pairings : ℕ := Nat.factorial 15

/-- The number of valid pairings -/
noncomputable def valid_pairings : ℕ := 
  Nat.factorial 14

/-- The probability of a valid pairing -/
noncomputable def probability : ℚ := valid_pairings / total_pairings

theorem shoe_pairing_probability :
  probability = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_l882_88245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_pricing_and_profit_l882_88288

/-- Represents the factory's pricing and profit calculation -/
structure Factory where
  production_cost : ℚ := 40
  base_price : ℚ := 60
  price_decrease_rate : ℚ := 2/100
  min_price : ℚ := 51
  threshold : ℕ := 100

/-- Calculates the actual factory price based on the order quantity -/
def actual_price (f : Factory) (x : ℕ) : ℚ :=
  if x ≤ f.threshold then f.base_price
  else max (f.base_price - (x - f.threshold : ℚ) * f.price_decrease_rate) f.min_price

/-- Calculates the profit for a given order quantity -/
def profit (f : Factory) (x : ℕ) : ℚ :=
  x * (actual_price f x - f.production_cost)

/-- Theorem stating the key properties of the factory's pricing and profit -/
theorem factory_pricing_and_profit (f : Factory) :
  (∃ x : ℕ, x = 550 ∧ actual_price f x = f.min_price) ∧
  (actual_price f 500 = 52) ∧
  (profit f 500 = 6000) ∧
  (profit f 1000 = 11000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_pricing_and_profit_l882_88288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_4_l882_88251

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- Line l1 in polar coordinates -/
def line_l1 (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi/3) + 3 * Real.sqrt 3 = 0

/-- Line l2 in polar coordinates -/
def line_l2 (ρ θ : ℝ) : Prop := θ = Real.pi/3

/-- Point P is the intersection of curve C and line l2 -/
noncomputable def point_P : ℝ × ℝ := (1, Real.pi/3)

/-- Point Q is the intersection of line l1 and line l2 -/
noncomputable def point_Q : ℝ × ℝ := (-3, Real.pi/3)

/-- The length of segment PQ is the absolute difference of their ρ coordinates -/
noncomputable def length_PQ : ℝ := |point_P.1 - point_Q.1|

theorem length_PQ_is_4 :
  curve_C point_P.1 point_P.2 ∧
  line_l2 point_P.1 point_P.2 ∧
  line_l1 point_Q.1 point_Q.2 ∧
  line_l2 point_Q.1 point_Q.2 →
  length_PQ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_4_l882_88251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l882_88218

-- Define the function g(x)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x^2 + 3*x + 1 else 3*x + 6

-- State the theorem
theorem continuous_piecewise_function (m : ℝ) :
  Continuous (g m) ↔ m = Real.sqrt 5 ∨ m = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l882_88218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_sunday_l882_88255

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week n days before a given day
def daysBefore (d : DayOfWeek) : Nat → DayOfWeek
  | 0 => d
  | n + 1 => match daysBefore d n with
    | .Sunday => .Saturday
    | .Monday => .Sunday
    | .Tuesday => .Monday
    | .Wednesday => .Tuesday
    | .Thursday => .Wednesday
    | .Friday => .Thursday
    | .Saturday => .Friday

theorem february_first_is_sunday (h : DayOfWeek.Friday = daysBefore DayOfWeek.Friday 12) :
  DayOfWeek.Sunday = daysBefore DayOfWeek.Friday 12 := by
  sorry

#check february_first_is_sunday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_sunday_l882_88255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l882_88205

theorem article_price_after_discounts (original_price : ℝ) : 
  let first_discount : ℝ := 0.2
  let second_discount : ℝ := 0.24
  let final_price : ℝ := 320
  (1 - first_discount) * (1 - second_discount) * original_price = final_price →
  abs (original_price - 526.32) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_price_after_discounts_l882_88205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exampleTrainLength_l882_88231

/-- Calculates the length of a train compartment given its speed in km/h and time in seconds -/
noncomputable def compartmentLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (5/18) * time

/-- Represents a train with 5 compartments -/
structure Train :=
  (speeds : Fin 5 → ℝ)
  (times : Fin 5 → ℝ)

/-- Calculates the total length of the train -/
noncomputable def totalLength (train : Train) : ℝ :=
  (Finset.sum (Finset.range 5) fun i => compartmentLength (train.speeds i) (train.times i))

/-- The specific train from the problem -/
def exampleTrain : Train :=
  { speeds := ![40, 50, 60, 70, 80],
    times := ![9, 8, 7, 6, 5] }

/-- Theorem stating that the total length of the example train is approximately 555.56 meters -/
theorem exampleTrainLength : 
  ⌊totalLength exampleTrain⌋ = 555 ∧ |totalLength exampleTrain - 555.56| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exampleTrainLength_l882_88231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l882_88247

/-- Given a hyperbola with asymptotes y = ±√2x and foci at (-√6, 0) and (√6, 0),
    its equation is x²/2 - y²/4 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (k : ℝ), y = k * x ∧ k^2 = 2) ∧  -- Asymptotes condition
  (∃ (a : ℝ), a^2 = 6 ∧ ((-a, 0) ∈ ({(-a, 0), (a, 0)} : Set (ℝ × ℝ)) ∧ 
                         (a, 0) ∈ ({(-a, 0), (a, 0)} : Set (ℝ × ℝ)))) →  -- Foci condition
  (x^2 / 2 - y^2 / 4 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l882_88247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l882_88220

theorem relationship_abc : 
  let a := Real.log 2
  let b := (5 : ℝ) ^ (-(1/2 : ℝ))
  let c := ∫ x in (0 : ℝ)..(Real.pi/2), (1/2) * Real.cos x
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l882_88220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_parabola_l882_88210

/-- A line is tangent to a parabola if and only if their intersection has exactly one point -/
def is_tangent_line (a b c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, (a * p.1 + b * p.2 + c = 0) ∧ (p.2^2 = 32 * p.1)

theorem tangent_line_to_parabola (k : ℝ) :
  is_tangent_line 4 7 k ↔ k = 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_parabola_l882_88210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_next_polygon_l882_88281

/-- The number of diagonals in a convex polygon with n sides -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem: The number of diagonals in a convex polygon with n+1 sides 
    is equal to f(n) + n, where f(n) is the number of diagonals in a 
    polygon with n sides. -/
theorem diagonals_next_polygon (n : ℕ) : f (n + 1) = f n + n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_next_polygon_l882_88281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l882_88228

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem f_increasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l882_88228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_wrt_x_axis_l882_88222

/-- Given two points in ℝ³, this function checks if they are symmetrical with respect to the x-axis --/
def symmetrical_wrt_x_axis (p q : ℝ × ℝ × ℝ) : Prop :=
  p.fst = q.fst ∧ p.snd.fst = -q.snd.fst ∧ p.snd.snd = -q.snd.snd

/-- Theorem stating that (2, -3, -4) is symmetrical to (2, 3, 4) with respect to the x-axis --/
theorem symmetry_wrt_x_axis :
  symmetrical_wrt_x_axis (2, 3, 4) (2, -3, -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_wrt_x_axis_l882_88222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_affine_transformation_affine_transformation_triangles_affine_transformation_parallelograms_l882_88216

-- Define a vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define an affine transformation
def AffineTransformation (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  V → V

-- Define a basis (using a different name to avoid conflict)
def VectorBasis (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  Fin 2 → V

-- Theorem: Existence and uniqueness of affine transformation
theorem unique_affine_transformation
  (O O' : V) (e e' : VectorBasis V) :
  ∃! (L : AffineTransformation V),
    (L O = O') ∧
    (∀ i, L (O + e i) = O' + e' i) :=
sorry

-- Define a triangle
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

-- Theorem: Affine transformation between triangles
theorem affine_transformation_triangles
  (T1 T2 : Triangle V) :
  ∃! (L : AffineTransformation V),
    (L T1.A = T2.A) ∧ (L T1.B = T2.B) ∧ (L T1.C = T2.C) :=
sorry

-- Define a parallelogram
structure Parallelogram (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V
  D : V
  parallel_sides : (B - A) = (D - C) ∧ (C - B) = (A - D)

-- Theorem: Affine transformation between parallelograms
theorem affine_transformation_parallelograms
  (P1 P2 : Parallelogram V) :
  ∃! (L : AffineTransformation V),
    (L P1.A = P2.A) ∧ (L P1.B = P2.B) ∧ (L P1.C = P2.C) ∧ (L P1.D = P2.D) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_affine_transformation_affine_transformation_triangles_affine_transformation_parallelograms_l882_88216
