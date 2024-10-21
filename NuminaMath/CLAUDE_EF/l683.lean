import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l683_68379

noncomputable section

variable (f : ℝ → ℝ)

axiom f_derivative : Differentiable ℝ f

axiom f_symmetry (x : ℝ) : f x = 4 * x^2 - f (-x)

axiom f_derivative_bound (x : ℝ) (h : x < 0) : 
  deriv f x + 1/2 < 4 * x

theorem range_of_m : 
  {m : ℝ | f (m + 1) ≤ f (-m) + 3 * m + 3/2} = {m : ℝ | m ≥ -1/2} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l683_68379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_calculation_l683_68398

/-- Water pricing structure and usage calculation -/
theorem water_fee_calculation (a : ℝ) : 
  (17 * a) + (3 * (a + 1.2)) = 20 * a + 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_calculation_l683_68398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diabetes_related_to_drinking_l683_68349

/-- Represents the contingency table for the diabetes and drinking survey. -/
structure ContingencyTable :=
  (diabeticRegular : ℕ)
  (diabeticNonRegular : ℕ)
  (nonDiabeticRegular : ℕ)
  (nonDiabeticNonRegular : ℕ)

/-- Calculates the K^2 value for the given contingency table. -/
noncomputable def calculateK2 (table : ContingencyTable) : ℝ :=
  let n := table.diabeticRegular + table.diabeticNonRegular + table.nonDiabeticRegular + table.nonDiabeticNonRegular
  let a := table.diabeticRegular
  let b := table.diabeticNonRegular
  let c := table.nonDiabeticRegular
  let d := table.nonDiabeticNonRegular
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The reference K_0 value for 99.5% confidence. -/
def referenceK0 : ℝ := 7.879

/-- Theorem stating that diabetes is related to drinking with 99.5% confidence. -/
theorem diabetes_related_to_drinking (totalSurveyed : ℕ) (diabeticProbability : ℚ) 
  (table : ContingencyTable) :
  totalSurveyed = 30 →
  diabeticProbability = 1/3 →
  table.diabeticNonRegular = 2 →
  table.nonDiabeticRegular = 4 →
  table.diabeticRegular + table.diabeticNonRegular + table.nonDiabeticRegular + table.nonDiabeticNonRegular = totalSurveyed →
  calculateK2 table > referenceK0 := by
  sorry

#check diabetes_related_to_drinking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diabetes_related_to_drinking_l683_68349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_value_l683_68381

/-- The function f(x) = |x^2 + ax| -/
def f (a : ℝ) (x : ℝ) : ℝ := abs (x^2 + a*x)

/-- M(a) is the maximum value of f(x) on [0,1] -/
noncomputable def M (a : ℝ) : ℝ := sSup (Set.image (f a) (Set.Icc 0 1))

/-- The minimum value of M(a) is 3 - 2√2 -/
theorem min_M_value :
  ⨅ a, M a = 3 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_value_l683_68381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l683_68390

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (Real.sqrt (8 - x))

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Ici 3 ∩ Set.Iio 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l683_68390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_theorem_l683_68358

theorem orange_crates_theorem (total_crates min_oranges max_oranges : ℕ) :
  total_crates = 150 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ (n : ℕ), n ≥ 5 ∧ 
  ∃ (count : ℕ), count ≥ min_oranges ∧ count ≤ max_oranges ∧
  (Finset.filter (fun crate => crate = count) (Finset.range total_crates)).card ≥ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_theorem_l683_68358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_cubic_poly_l683_68385

noncomputable def munificence (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ⨆ (x : ℝ) (_ : a ≤ x ∧ x ≤ b), |f x|

def cubic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

theorem smallest_munificence_cubic_poly :
  ∀ a b c : ℝ, munificence (cubic_polynomial a b c) (-1) 1 ≥ 3 := by
  sorry

#check smallest_munificence_cubic_poly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_cubic_poly_l683_68385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l683_68331

/-- The speed of a train given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_calculation :
  let train_length : ℝ := 1500
  let crossing_time : ℝ := 50
  ∃ ε > 0, |train_speed train_length crossing_time - 108| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l683_68331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l683_68352

-- Define the equations of the two lines
def line1 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 10 = 0
def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ m b₁ b₂, ∀ (x y : ℝ), (f x y ↔ y = m * x + b₁) ∧ (g x y ↔ y = m * x + b₂) ∧ b₁ ≠ b₂

-- Theorem statement
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel (line1 a) (line2 a) → a = -1 ∨ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l683_68352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l683_68303

/-- A hyperbola passing through the point (6, √3) with asymptotes y = ±(1/3)x has the equation (x²/9) - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (x^2 / 9 - y^2 = k)) →
  ((6^2 / 9 - (Real.sqrt 3)^2 = 1) ∧
   (∀ t : ℝ, y = (1/3) * x ∨ y = -(1/3) * x)) →
  x^2 / 9 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l683_68303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l683_68370

/-- Circle C with center (4, -1) and radius 5 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 + 1)^2 = 25}

/-- Point M on the circle C -/
def M : ℝ × ℝ := (0, 2)

/-- Line l' parallel to the tangent line at M -/
def l' (a : ℝ) : Set (ℝ × ℝ) := {p | 4 * p.1 - a * p.2 + 2 = 0}

/-- Tangent line predicate -/
def IsTangentLine (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ C ∧ p ∈ l ∧ ∀ q ∈ C, q ≠ p → q ∉ l

/-- Parallel lines predicate -/
def IsParallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ k b1 b2, l1 = {p | p.2 = k * p.1 + b1} ∧ l2 = {p | p.2 = k * p.1 + b2}

/-- Distance between parallel lines -/
noncomputable def DistanceBetweenParallelLines (l1 l2 : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem distance_between_parallel_lines :
  M ∈ C →
  ∃ l : Set (ℝ × ℝ), ∃ a : ℝ,
    (IsTangentLine C l M) ∧
    (IsParallel l (l' a)) ∧
    (DistanceBetweenParallelLines l (l' a) = 4/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l683_68370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_non_negative_condition_l683_68308

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (2*a - 1)/x - 2*a * Real.log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 + (2*a - 1)/(x^2) - 2*a/x

theorem extreme_value_condition (a : ℝ) :
  f_deriv a 2 = 0 → a = 3/2 := by
  sorry

theorem non_negative_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_non_negative_condition_l683_68308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_eggs_sold_l683_68355

/-- Represents the number of eggs sold by the first vendor -/
def x : ℕ := sorry

/-- Represents the total revenue of each vendor -/
def y : ℚ := sorry

/-- The total number of eggs sold -/
def total_eggs : ℕ := 110

/-- Condition: Both vendors had equal revenues -/
axiom equal_revenue : y = y

/-- Condition: If vendor 1 sold all eggs at their price -/
axiom vendor1_all_eggs : (total_eggs : ℚ) * (y / x) = 5

/-- Condition: If vendor 2 sold all eggs at their price -/
axiom vendor2_all_eggs : (total_eggs : ℚ) * (y / (total_eggs - x)) = (36 : ℚ) / 5

/-- The theorem to be proved -/
theorem vendor_eggs_sold : x = 60 ∧ total_eggs - x = 50 := by
  sorry

#check vendor_eggs_sold

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_eggs_sold_l683_68355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_abs_S_5_equals_1_l683_68395

-- Define the sequence of Bernoulli trials
def bernoulli_sequence (n : ℕ) : List ℤ := sorry

-- Define the sum of the first n elements
def S (n : ℕ) : ℤ := (bernoulli_sequence n).sum

-- State the theorem
theorem probability_abs_S_5_equals_1 :
  (Finset.filter (λ s => |S 5| = 1) (Finset.range (2^5))).card / (2^5 : ℚ) = 5/8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_abs_S_5_equals_1_l683_68395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_calculation_l683_68315

/-- The positive difference in minutes between the time it takes Tom to cover twice Linda's distance and half of Linda's distance -/
noncomputable def time_difference (linda_speed : ℝ) (tom_speed : ℝ) (start_delay : ℝ) : ℝ :=
  let linda_distance := linda_speed * start_delay
  let half_distance_time := (linda_distance / 2) / tom_speed * 60
  let double_distance_time := (linda_distance * 2) / tom_speed * 60
  double_distance_time - half_distance_time

/-- Theorem stating the time difference for the given speeds and delay -/
theorem time_difference_calculation :
  time_difference 2 8 1 = 22.5 := by
  unfold time_difference
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_calculation_l683_68315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_percentage_correct_l683_68321

/-- Represents a test with a number of problems and a percentage of correct answers -/
structure Test where
  problems : ℕ
  percentage : ℚ

/-- Calculates the total number of correctly answered problems across all tests -/
def totalCorrect (tests : List Test) : ℚ :=
  tests.foldl (fun acc test => acc + test.problems * test.percentage) 0

/-- Calculates the total number of problems across all tests -/
def totalProblems (tests : List Test) : ℕ :=
  tests.foldl (fun acc test => acc + test.problems) 0

/-- The main theorem stating the overall percentage of correctly answered problems -/
theorem overall_percentage_correct (tests : List Test) 
    (h : tests = [
      ⟨30, 85/100⟩, 
      ⟨50, 75/100⟩, 
      ⟨20, 65/100⟩, 
      ⟨40, 95/100⟩
    ]) : 
    (totalCorrect tests) / (totalProblems tests : ℚ) = 8143/10000 := by
  sorry

#eval (8143 : ℚ) / 10000  -- To display the result as 0.8143

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_percentage_correct_l683_68321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l683_68367

noncomputable def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

noncomputable def vertex (b c : ℝ) : ℝ × ℝ := (b/2, quadratic_function b c (b/2))

noncomputable def range_on_interval (b c : ℝ) (a d : ℝ) : Set ℝ :=
  {y | ∃ x, a ≤ x ∧ x ≤ d ∧ y = quadratic_function b c x}

theorem quadratic_properties :
  (vertex 4 3 = (2, 7)) ∧
  (range_on_interval 4 3 (-1) 3 = {y | -2 ≤ y ∧ y ≤ 7}) ∧
  (∀ x ≤ 0, quadratic_function 2 2 x ≤ 2) ∧
  (∀ x > 0, quadratic_function 2 2 x ≤ 3) ∧
  (∃ x₁ ≤ 0, quadratic_function 2 2 x₁ = 2) ∧
  (∃ x₂ > 0, quadratic_function 2 2 x₂ = 3) := by
  sorry

#check quadratic_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l683_68367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l683_68354

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (A + B + C = Real.pi) →
  (Real.cos A = 2 * Real.sqrt 5 / 5) →
  (Real.sin B = Real.sqrt 10 / 10) →
  (a = Real.sqrt 10) →
  -- Conclusions to prove
  (C = 3 * Real.pi / 4) ∧
  (1/2 * a * b * Real.sin C = 5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l683_68354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_C_through_A_l683_68337

/-- The circle C with equation (x-2)^2 + (y-3)^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

/-- Point A with coordinates (-1, 4) -/
def A : ℝ × ℝ := (-1, 4)

/-- A line in the plane, represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  |l.a * p.1 + l.b * p.2 + l.c| / Real.sqrt (l.a^2 + l.b^2)

/-- A line is tangent to the circle C if the distance from the center of C to the line equals the radius -/
def isTangentToC (l : Line) : Prop :=
  distancePointToLine (2, 3) l = 1

/-- A line passes through point A -/
def passesThroughA (l : Line) : Prop :=
  l.a * A.1 + l.b * A.2 + l.c = 0

/-- The two possible tangent lines -/
def l₁ : Line := { a := 0, b := 1, c := -4 }
def l₂ : Line := { a := 3, b := 4, c := -13 }

theorem tangent_lines_to_C_through_A :
  ∀ l : Line, isTangentToC l ∧ passesThroughA l ↔ l = l₁ ∨ l = l₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_C_through_A_l683_68337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OBEC_l683_68384

-- Define the points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (6, 0)
def E : ℝ × ℝ := (3, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := -3 * x + 12
noncomputable def line2 (x : ℝ) : ℝ := (x - 6) * (3 / 3) + 0

-- Define the area function for a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_OBEC : 
  triangleArea O B E + triangleArea E C (3, 0) = 22.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OBEC_l683_68384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_function_property_l683_68341

/-- Given an invertible function f: ℝ → ℝ, if f(a) = 3 and f(3) = 6, then a - 3 = -2 -/
theorem invertible_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_inv : Function.Bijective f)
  (h_fa : f a = 3)
  (h_f3 : f 3 = 6) :
  a - 3 = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_function_property_l683_68341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_value_l683_68340

-- Problem 1
theorem trigonometric_simplification (α : ℝ) :
  (Real.cos (α - π/2)) / (Real.sin (5*π/2 + α)) * Real.sin (α - 2*π) * Real.cos (2*π - α) = Real.sin α ^ 2 := by
  sorry

-- Problem 2
theorem trigonometric_value (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (2*α) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_value_l683_68340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_function_a_range_l683_68383

open Real

theorem decreasing_log_function_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 →
    logb a (2 - a * x₂) < logb a (2 - a * x₁)) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_function_a_range_l683_68383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_circumcenter_orthocenter_implies_60_degrees_l683_68319

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle. -/
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The orthocenter of a triangle. -/
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between two points. -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The measure of an angle in degrees. -/
noncomputable def angle_measure (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- A triangle is acute if all its angles are less than 90°. -/
def is_acute (t : Triangle) : Prop :=
  angle_measure t t.A < 90 ∧ angle_measure t t.B < 90 ∧ angle_measure t t.C < 90

theorem equidistant_circumcenter_orthocenter_implies_60_degrees (t : Triangle) :
  is_acute t →
  distance t.A (circumcenter t) = distance t.A (orthocenter t) →
  angle_measure t t.A = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_circumcenter_orthocenter_implies_60_degrees_l683_68319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l683_68376

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

-- State the theorem
theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l683_68376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_for_given_conditions_l683_68394

/-- Calculates the speed in km/hr given the side length of a square field in meters and the time to run around it in seconds -/
noncomputable def calculate_speed (side_length : ℝ) (time_seconds : ℝ) : ℝ :=
  let perimeter := 4 * side_length
  let distance_km := perimeter / 1000
  let time_hours := time_seconds / 3600
  distance_km / time_hours

/-- Theorem stating that for a square field with side length 50 meters and a run time of 72 seconds, the speed is 10 km/hr -/
theorem speed_for_given_conditions :
  calculate_speed 50 72 = 10 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_for_given_conditions_l683_68394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_equality_l683_68310

theorem divisibility_implies_equality (a b : ℕ+) 
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a.val^(n+1) + b.val^(n+1)) % (a.val^n + b.val^n) = 0) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_equality_l683_68310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l683_68329

-- Define the geometric series sum function T
noncomputable def T (r : ℝ) : ℝ := 6 / (1 - r)

-- State the theorem
theorem geometric_series_sum_property (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 504) : 
  T b + T (-b) = 168 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l683_68329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_arithmetic_mean_on_interval_l683_68375

def has_arithmetic_mean (f : ℝ → ℝ) (D : Set ℝ) (c : ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ (f x₁ + f x₂) / 2 = c

theorem ln_arithmetic_mean_on_interval :
  has_arithmetic_mean Real.log (Set.Icc 2 8) (Real.log 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_arithmetic_mean_on_interval_l683_68375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_inscribed_and_circumscribed_squares_l683_68324

/-- The angle between the sides of an inscribed square and a circumscribed square of a circle -/
noncomputable def angle_between_squares (r : ℝ) : ℝ := Real.arcsin ((Real.sqrt 6 - Real.sqrt 2) / 4)

/-- The side length of the inscribed square -/
noncomputable def inscribed_square_side (r : ℝ) : ℝ := r * Real.sqrt 2

/-- The side length of the circumscribed square -/
def circumscribed_square_side (r : ℝ) : ℝ := 2 * r

theorem angle_between_inscribed_and_circumscribed_squares (r : ℝ) (hr : r > 0) :
  let x := angle_between_squares r
  let a := inscribed_square_side r
  let b := circumscribed_square_side r
  b^2 = (a + 2 * r * Real.sin x)^2 + (2 * r * Real.sin x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_inscribed_and_circumscribed_squares_l683_68324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_yield_growth_rate_l683_68316

/-- Represents the farm's pumpkin production data and growth rates -/
structure FarmData where
  initial_area : ℝ
  initial_yield : ℝ
  final_yield : ℝ
  area_growth_rate : ℝ → ℝ
  yield_growth_rate : ℝ

/-- The growth rate of the average yield per acre is 0.5 -/
theorem pumpkin_yield_growth_rate (farm : FarmData) :
  farm.initial_area = 10 ∧
  farm.initial_yield = 20000 ∧
  farm.final_yield = 60000 ∧
  farm.area_growth_rate = (λ x => 1 + 2 * x) ∧
  farm.yield_growth_rate = 0.5 →
  farm.initial_yield * (1 + 2 * farm.yield_growth_rate) * (1 + farm.yield_growth_rate) = farm.final_yield := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_yield_growth_rate_l683_68316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fraction_equality_l683_68304

theorem cos_fraction_equality (θ : Real) (h : Real.cos (Real.pi + θ) = 1/3) :
  (Real.cos (2*Real.pi - θ)) / (Real.sin (Real.pi/2 + θ) * Real.cos (Real.pi - θ) + Real.cos (-θ)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_fraction_equality_l683_68304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_integer_l683_68323

theorem fraction_is_integer (a : ℤ) : 
  a ≠ 0 → (∃ k : ℤ, (a^2 - 21*a + 17) = k * a ↔ a = 1 ∨ a = -1 ∨ a = 17 ∨ a = -17) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_integer_l683_68323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_route_time_l683_68336

noncomputable def route1_distance : ℝ := 1500
noncomputable def route1_speed : ℝ := 75
noncomputable def route2_distance : ℝ := 750
noncomputable def route2_speed : ℝ := 25

noncomputable def route_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem fastest_route_time :
  min (route_time route1_distance route1_speed) (route_time route2_distance route2_speed) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_route_time_l683_68336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_probability_l683_68391

/-- The probability of an acute angle between two vectors -/
theorem acute_angle_probability : 
  let dice_outcomes : Finset (ℕ × ℕ) := Finset.filter (λ (m, n) => 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) (Finset.product (Finset.range 6) (Finset.range 6))
  let acute_angle_outcomes : Finset (ℕ × ℕ) := Finset.filter (λ (m, n) => m - 2*n > 0) dice_outcomes
  (Finset.card acute_angle_outcomes : ℚ) / (Finset.card dice_outcomes : ℚ) = 7/36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_probability_l683_68391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_angle_measure_l683_68326

noncomputable def radius_outer : ℝ := 4
noncomputable def radius_middle : ℝ := 3
noncomputable def radius_inner : ℝ := 2

noncomputable def area_ratio : ℝ := 9 / 14

theorem circle_angle_measure (θ : ℝ) :
  (radius_outer ^ 2 * θ + radius_inner ^ 2 * θ = 
   area_ratio * (π * (radius_outer ^ 2 + radius_middle ^ 2 + radius_inner ^ 2) - 
   (radius_outer ^ 2 * θ + radius_inner ^ 2 * θ))) →
  θ = 1827 * π / 3220 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_angle_measure_l683_68326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_difference_theorem_l683_68386

/-- Calculates the difference between adjusted apples and peaches, and pears --/
def fruitDifference (apples : ℕ) (peachMultiplier : ℕ) (pearDivisor : ℕ) 
  (appleEatenPercent : ℚ) (peachEatenPercent : ℚ) : ℤ :=
  let peaches := apples * peachMultiplier
  let pears := apples / pearDivisor
  let adjustedApples := apples - (apples * appleEatenPercent).floor
  let adjustedPeaches := peaches - (peaches * peachEatenPercent).floor
  (adjustedApples + adjustedPeaches : ℤ) - pears

/-- Theorem stating the difference between adjusted fruits and pears --/
theorem fruit_difference_theorem : 
  fruitDifference 60 3 2 (1/10) (1/20) = 195 := by
  sorry

#eval fruitDifference 60 3 2 (1/10) (1/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_difference_theorem_l683_68386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l683_68365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem solution_set_of_inequality (a : ℝ) (h : is_even_function (f a)) :
  {x : ℝ | f a (x - 1) > (Real.exp 4 + 1) / Real.exp 2} = {x : ℝ | x < -1 ∨ x > 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l683_68365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l683_68374

-- Define the transformations
def rotate_x_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, -p.2.2)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2.1, p.2.2)

def rotate_z_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, -p.2.1, p.2.2)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (rotate_z_180 (reflect_yz (reflect_xz (rotate_x_180 p))))

-- Theorem statement
theorem point_transformation :
  transform (2, 2, 2) = (-2, -2, -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l683_68374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l683_68369

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  Real.sin (t.B / 2) = Real.sqrt 5 / 5 →
  t.a * t.c * Real.cos t.B = 6 →
  (1/2) * t.a * t.c * Real.sin t.B = 4 ∧
  (t.c + t.a = 8 → t.b = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l683_68369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_three_l683_68300

/-- Rectangular parallelepiped with dimensions AB, BC, CG -/
structure Parallelepiped where
  AB : ℝ
  BC : ℝ
  CG : ℝ

/-- Point on edge AB -/
noncomputable def point_on_AB (p : Parallelepiped) (x : ℝ) : ℝ := x

/-- Midpoint of edge FG -/
noncomputable def midpoint_FG (p : Parallelepiped) : ℝ := p.CG / 2

/-- Area of rectangle BCEF -/
noncomputable def area_BCEF (p : Parallelepiped) : ℝ := p.BC * p.CG

/-- Volume of pyramid with base BCEF and apex M -/
noncomputable def volume_pyramid (p : Parallelepiped) (h : ℝ) : ℝ :=
  (1 / 3) * area_BCEF p * h

/-- Theorem: Volume of the specific pyramid is 3 -/
theorem volume_is_three (p : Parallelepiped) (E : ℝ) :
  p.AB = 4 → p.BC = 2 → p.CG = 3 → E = point_on_AB p 1 →
  volume_pyramid p (midpoint_FG p) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_three_l683_68300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l683_68335

def M : Set ℝ := {-2, -1, 0, 1, 2}

def N : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l683_68335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smiths_children_ages_l683_68305

theorem smiths_children_ages (children_ages : Finset ℕ) : 
  children_ages.card = 9 ∧ 
  (∀ a, a ∈ children_ages → a ∈ Finset.range 11 ∧ a ≠ 0) ∧
  (∀ a, a ∈ children_ages → 5353 % a = 0) ∧
  11 ∉ children_ages ∧
  (∀ a b, a ∈ children_ages → b ∈ children_ages → a ≠ b → a ≠ b) →
  6 ∉ children_ages :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smiths_children_ages_l683_68305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_three_speak_l683_68382

def probability_of_speaking : ℚ := 1/3

def number_of_babies : ℕ := 6

def at_least_speaking : ℕ := 3

theorem probability_at_least_three_speak :
  let p := probability_of_speaking
  let n := number_of_babies
  let k := at_least_speaking
  (Finset.range (n + 1)).sum (λ i ↦ if i ≥ k then (n.choose i) * p^i * (1 - p)^(n - i) else 0) = 233/729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_three_speak_l683_68382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_price_increase_is_87_5_l683_68301

/-- The percentage increase in electricity price -/
noncomputable def electricity_price_increase (old_power : ℝ) (power_increase : ℝ) 
  (old_price : ℝ) (new_cost : ℝ) (run_time : ℝ) : ℝ :=
  let new_power := old_power * (1 + power_increase)
  let old_cost := old_power / 1000 * run_time * old_price
  let new_price := new_cost / (new_power / 1000 * run_time)
  (new_price - old_price) / old_price * 100

/-- Theorem stating that the electricity price increase is 87.5% -/
theorem electricity_price_increase_is_87_5 :
  electricity_price_increase 800 0.5 0.12 9 50 = 87.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval electricity_price_increase 800 0.5 0.12 9 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_price_increase_is_87_5_l683_68301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_y_iff_x_range_l683_68342

/-- For real x and y satisfying 9y^2 + 3xy + 2x + 9 = 0, y is real if and only if x ≤ -3 or x ≥ 12 -/
theorem real_y_iff_x_range (x y : ℝ) (h : 9 * y^2 + 3 * x * y + 2 * x + 9 = 0) :
  (x ≤ -3 ∨ x ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_y_iff_x_range_l683_68342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sin_sum_max_floor_10L_eq_25_l683_68306

open Real

/-- The maximum value of sin(3A) + sin(3B) + sin(3C) for angles A, B, C of a triangle -/
noncomputable def L : ℝ := (3 * Real.sqrt 3) / 2

theorem triangle_angle_sin_sum_max :
  ∀ A B C : ℝ,
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧
  A + B + C = π →
  sin (3 * A) + sin (3 * B) + sin (3 * C) ≤ L :=
by sorry

theorem floor_10L_eq_25 : 
  ⌊10 * L⌋ = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sin_sum_max_floor_10L_eq_25_l683_68306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sqrt_at_4_l683_68343

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem tangent_slope_sqrt_at_4 :
  let x₀ : ℝ := 4
  let y₀ : ℝ := f x₀
  let slope : ℝ := (deriv f) x₀
  slope = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sqrt_at_4_l683_68343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_solution_set_part_two_range_of_a_l683_68389

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- Part I
theorem part_one_solution_set :
  {x : ℝ | x < -3 ∨ x > 1} = {x : ℝ | f 1 x > 4} :=
sorry

-- Part II
theorem part_two_range_of_a (a : ℝ) (h : a ≠ 0) :
  (∃! x, x ∈ Set.Ioo 1 2 ∧ f a x = 0) → -1/3 < a ∧ a < -1/8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_solution_set_part_two_range_of_a_l683_68389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l683_68388

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log a
  else if -4 ≤ x ∧ x < 0 then |x + 3|
  else 0

def has_two_symmetric_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f a x = f a (-y) ∧ f a y = f a (-x)

theorem range_of_a (a : ℝ) :
  a > 0 ∧ a ≠ 1 ∧ has_two_symmetric_points a →
  (a > 0 ∧ a < 1) ∨ (a > 1 ∧ a < 4) :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l683_68388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_triangle_l683_68332

theorem angle_measure_in_triangle (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b^2 + c^2 = 3 + b*c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_triangle_l683_68332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_nonzero_z_magnitude_l683_68351

-- Define the primitive third root of unity
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

-- Define the complex numbers a, b, c
noncomputable def a : ℂ := 1
noncomputable def b : ℂ := -ω
noncomputable def c : ℂ := -ω^2

-- Theorem statement
theorem smallest_nonzero_z_magnitude :
  ∃ (z : ℂ), z ≠ 0 ∧ a * z^2 + b * z + c = 0 ∧ Complex.abs z = (1/2 : ℝ) ∧
  ∀ (w : ℂ), w ≠ 0 → a * w^2 + b * w + c = 0 → Complex.abs w ≥ (1/2 : ℝ) :=
sorry

-- Additional lemmas that might be useful for the proof
lemma abs_a : Complex.abs a = 1 := by sorry
lemma abs_b : Complex.abs b = 1 := by sorry
lemma abs_c : Complex.abs c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_nonzero_z_magnitude_l683_68351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l683_68399

theorem complex_absolute_value : ∀ i : ℂ, i^2 = -1 →
  Complex.abs (2 + i^2 + 2*i^3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l683_68399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l683_68344

/-- A circle with its center on the line y = 2x and tangent to the x-axis at (1, 0) -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  /-- The center of the circle is on the line y = 2x -/
  center_on_line : center.2 = 2 * center.1
  /-- The circle is tangent to the x-axis at the point (1, 0) -/
  tangent_point : center.1 = 1 ∧ radius = center.2

/-- The standard equation of the circle is (x - 1)² + (y - 2)² = 4 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ (x y : ℝ), ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ↔ (x - 1)^2 + (y - 2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l683_68344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_value_l683_68373

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (x + Real.pi/4) + Real.sqrt 3 * Real.sin (x - Real.pi/4)

theorem even_function_implies_a_value (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_value_l683_68373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_ratio_l683_68356

noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let square := { p : ℝ × ℝ | max (|p.1|) (|p.2|) = r * Real.sqrt 2 / 2 }
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = r^2 }
  let square_area := (2 * r^2 : ℝ)
  let circle_area := (π * r^2 : ℝ)
  (∀ (x y : ℝ), (x, y) ∈ square → (x^2 + y^2 ≤ r^2)) →
  (∀ (side : Set (ℝ × ℝ)), side ⊆ square → (∃ (p q : ℝ × ℝ), p ∈ side ∧ q ∈ side ∧ p ∈ circle ∧ q ∈ circle ∧ p ≠ q ∧ (distance p q)^2 = r^2)) →
  square_area / circle_area = 2 / π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_ratio_l683_68356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_trapezoids_l683_68314

/-- A right trapezoid ABCD with integer side lengths -/
structure RightTrapezoid where
  ab : ℕ
  bc : ℕ
  ad : ℕ
  perimeter : ℕ

/-- The set of valid right trapezoids satisfying the given conditions -/
def ValidTrapezoids : Set RightTrapezoid :=
  {t : RightTrapezoid |
    t.ad = 2 ∧
    t.bc = t.bc ∧
    t.perimeter < 100 ∧
    t.perimeter = t.ab + 2 * t.bc + 2}

/-- Theorem stating that there are exactly 5 valid right trapezoids -/
theorem count_valid_trapezoids :
  ∃ (s : Finset RightTrapezoid), s.card = 5 ∧ ∀ t ∈ s, t ∈ ValidTrapezoids := by
  sorry

#check count_valid_trapezoids

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_trapezoids_l683_68314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMCN_is_31_25_l683_68392

/-- Rectangle with midpoints on two adjacent sides -/
structure RectangleWithMidpoints where
  /-- Length of the rectangle -/
  length : ℝ
  /-- Width of the rectangle -/
  width : ℝ
  /-- Assertion that length and width are positive -/
  length_pos : 0 < length
  width_pos : 0 < width

/-- Calculate the area of the quadrilateral formed by connecting two vertices
    of a rectangle with the midpoints of two adjacent sides -/
noncomputable def areaAMCN (r : RectangleWithMidpoints) : ℝ :=
  r.length * r.width - (r.length * r.width / 4) - (r.length * r.width / 8)

/-- Theorem stating that for a 10 cm by 5 cm rectangle, 
    the area of quadrilateral AMCN is 31.25 cm² -/
theorem area_AMCN_is_31_25 :
  let r : RectangleWithMidpoints := {
    length := 10,
    width := 5,
    length_pos := by norm_num,
    width_pos := by norm_num
  }
  areaAMCN r = 31.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMCN_is_31_25_l683_68392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_uphill_time_l683_68348

/-- Represents Alex's bike trip segments --/
structure BikeTrip where
  flatSpeed : ℚ
  flatTime : ℚ
  uphillSpeed : ℚ
  downhillSpeed : ℚ
  downhillTime : ℚ
  walkingDistance : ℚ
  totalDistance : ℚ

/-- Calculates the uphill time given a BikeTrip --/
def uphillTime (trip : BikeTrip) : ℚ :=
  let flatDistance := trip.flatSpeed * trip.flatTime
  let downhillDistance := trip.downhillSpeed * trip.downhillTime
  let remainingDistance := trip.totalDistance - trip.walkingDistance - flatDistance - downhillDistance
  remainingDistance / trip.uphillSpeed

/-- Theorem stating that Alex's uphill time is 2.5 hours --/
theorem alex_uphill_time :
  let trip := BikeTrip.mk 20 (9/2) 12 24 (3/2) 8 164
  uphillTime trip = (5/2) := by
  -- The proof goes here
  sorry

#eval uphillTime (BikeTrip.mk 20 (9/2) 12 24 (3/2) 8 164)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_uphill_time_l683_68348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crowd_size_l683_68346

theorem max_crowd_size : 
  ∃ (x : ℕ), x = 37 ∧ 
  (∀ y : ℕ, y > 37 → 
    ¬(∃ (a b c : ℕ), 
      a + b + c = y ∧ 
      |((a : ℝ) - (y : ℝ)/2)| ≤ 1/2 ∧ 
      |((b : ℝ) - (y : ℝ)/3)| ≤ 1/2 ∧ 
      |((c : ℝ) - (y : ℝ)/5)| ≤ 1/2)) ∧
  (∃ (a b c : ℕ), 
    a + b + c = 37 ∧ 
    |((a : ℝ) - 37/2)| ≤ 1/2 ∧ 
    |((b : ℝ) - 37/3)| ≤ 1/2 ∧ 
    |((c : ℝ) - 37/5)| ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crowd_size_l683_68346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_theorem_l683_68330

-- Define the properties of the oil cans
structure OilCan where
  height : ℝ
  radius : ℝ

-- Define the relationship between cans X and Y
def can_Y (x : OilCan) : OilCan :=
  { height := 4 * x.height,
    radius := 4 * x.radius }

-- Define the volume of a cylinder
noncomputable def volume (can : OilCan) : ℝ :=
  Real.pi * can.radius ^ 2 * can.height

-- Define the price of oil in can X when filled to capacity
def price_X : ℝ := 2

-- Theorem statement
theorem oil_price_theorem (x : OilCan) :
  price_X * (volume (can_Y x) / volume x) / 2 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_theorem_l683_68330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_narrower_band_better_regression_larger_abs_correlation_stronger_linear_relation_l683_68311

-- Define the width of the band area with uniformly distributed residual points
def band_width : ℝ → ℝ := sorry

-- Define the regression effect of the model
def regression_effect : ℝ → ℝ := sorry

-- Define the sample correlation coefficient
def sample_correlation_coefficient : ℝ := sorry

-- Define the strength of linear correlation
def linear_correlation_strength : ℝ → ℝ := sorry

-- Statement B
theorem narrower_band_better_regression :
  ∀ w1 w2 : ℝ, w1 < w2 → regression_effect (band_width w1) > regression_effect (band_width w2) :=
by sorry

-- Statement C
theorem larger_abs_correlation_stronger_linear_relation :
  ∀ r1 r2 : ℝ, |r1| > |r2| → 
  linear_correlation_strength r1 > linear_correlation_strength r2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_narrower_band_better_regression_larger_abs_correlation_stronger_linear_relation_l683_68311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_and_min_reciprocal_sum_l683_68328

def is_valid_xy (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + 5 * y = 20

theorem max_log_sum_and_min_reciprocal_sum 
  (x y : ℝ) 
  (h : is_valid_xy x y) : 
  Real.log x + Real.log y ≤ Real.log 10 ∧ 
  1 / x + 1 / y ≥ (7 + 2 * Real.sqrt 10) / 20 := by
  sorry

#check max_log_sum_and_min_reciprocal_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_and_min_reciprocal_sum_l683_68328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_inequality_l683_68378

theorem triangle_sin_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B * Real.sin C ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_inequality_l683_68378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l683_68368

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - x - 2 * Real.log x

theorem min_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 1 (Real.exp 1) → f y ≥ f x) ∧
  f x = -2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l683_68368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_domain_interval_l683_68347

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := 1 / (x^(1/3))

-- Define the domain of v(x)
def domain_v : Set ℝ := {x : ℝ | x ≠ 0}

-- Theorem stating that the domain of v(x) is all real numbers except zero
theorem v_domain : 
  {x : ℝ | ∃ y, v x = y} = domain_v := by
  sorry

-- Theorem stating that the domain is equivalent to (-∞, 0) ∪ (0, ∞)
theorem domain_interval : 
  domain_v = Set.Iio 0 ∪ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_domain_interval_l683_68347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CO_l683_68371

/-- The molar mass of carbon in g/mol -/
noncomputable def molar_mass_C : ℝ := 12.01

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of carbon monoxide (CO) in g/mol -/
noncomputable def molar_mass_CO : ℝ := molar_mass_C + molar_mass_O

/-- The mass percentage of oxygen in carbon monoxide (CO) -/
noncomputable def mass_percentage_O : ℝ := (molar_mass_O / molar_mass_CO) * 100

/-- Theorem stating that the mass percentage of oxygen in CO is approximately 57.12% -/
theorem mass_percentage_O_in_CO : 
  57.11 < mass_percentage_O ∧ mass_percentage_O < 57.13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CO_l683_68371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_inequality_l683_68302

-- Define the open interval (1, 2)
def open_interval := Set.Ioo 1 2

-- State the theorem
theorem a_range_for_inequality (a : ℝ) :
  (∀ x ∈ open_interval, (x - 1)^2 < Real.log x / Real.log a) ↔ a ∈ Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_inequality_l683_68302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_solutions_l683_68396

theorem non_real_solutions (m : ℤ) : 
  (∀ n : ℚ, n^2 - 2*m^2 = m*(n + 3) - 3 → False) ↔ m ∈ ({-2, -1, 0} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_solutions_l683_68396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_perimeter_is_48_l683_68360

/-- The perimeter of the shaded region formed by three identical touching circles -/
noncomputable def shaded_perimeter (circle_circumference : ℝ) (arc_angle : ℝ) : ℝ :=
  3 * (arc_angle / 360) * circle_circumference

/-- Theorem: The perimeter of the shaded region is 48 -/
theorem shaded_perimeter_is_48 :
  shaded_perimeter 48 120 = 48 := by
  -- Unfold the definition of shaded_perimeter
  unfold shaded_perimeter
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_perimeter_is_48_l683_68360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_unshaded_cells_l683_68364

/-- Represents a cell in the 10x10 table -/
structure Cell where
  row : Fin 10
  col : Fin 10
  value : Nat

/-- The 10x10 table -/
def Table := Array (Array Cell)

/-- Checks if two cells are neighbors -/
def are_neighbors (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col.val + 1 = c2.col.val ∨ c2.col.val + 1 = c1.col.val)) ∨
  (c1.col = c2.col ∧ (c1.row.val + 1 = c2.row.val ∨ c2.row.val + 1 = c1.row.val))

/-- Checks if a cell should be shaded based on its neighbors -/
def should_be_shaded (table : Table) (cell : Cell) : Prop :=
  ∃ (n1 n2 : Cell), are_neighbors cell n1 ∧ are_neighbors cell n2 ∧
    cell.value < n1.value ∧ cell.value > n2.value

/-- Theorem: The minimal sum of unshaded cells is 20 -/
theorem min_sum_unshaded_cells (table : Table) : 
  (∀ c : Cell, c ∈ table.flatten → c.value > 0) →
  (∃! c1 c2 : Cell, c1 ∈ table.flatten ∧ c2 ∈ table.flatten ∧ 
    c1 ≠ c2 ∧ ¬(should_be_shaded table c1) ∧ ¬(should_be_shaded table c2) ∧
    (∀ c : Cell, c ∈ table.flatten → c ≠ c1 → c ≠ c2 → should_be_shaded table c)) →
  (∃ c1 c2 : Cell, c1 ∈ table.flatten ∧ c2 ∈ table.flatten ∧ 
    c1 ≠ c2 ∧ ¬(should_be_shaded table c1) ∧ ¬(should_be_shaded table c2) ∧
    c1.value + c2.value = 20 ∧
    (∀ d1 d2 : Cell, d1 ∈ table.flatten → d2 ∈ table.flatten → 
      d1 ≠ d2 → ¬(should_be_shaded table d1) → ¬(should_be_shaded table d2) →
      d1.value + d2.value ≥ 20)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_unshaded_cells_l683_68364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l683_68318

noncomputable section

/-- The original function f(x) -/
def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) - 1/2

/-- The shifted function g(x) -/
def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem smallest_positive_phi :
  ∃ φ₀ : ℝ, φ₀ > 0 ∧ is_odd (g φ₀) ∧ ∀ φ, φ > 0 → is_odd (g φ) → φ₀ ≤ φ :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l683_68318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l683_68325

/-- Represents a cubic polynomial of the form x^3 + ax^2 + bx + c -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The mean of the zeros of a cubic polynomial -/
noncomputable def mean_of_zeros (p : CubicPolynomial) : ℝ := -p.a / 3

/-- The product of the zeros of a cubic polynomial -/
def product_of_zeros (p : CubicPolynomial) : ℝ := -p.c

/-- The sum of the coefficients of a cubic polynomial -/
def sum_of_coefficients (p : CubicPolynomial) : ℝ := 1 + p.a + p.b + p.c

/-- The y-intercept of a cubic polynomial -/
def y_intercept (p : CubicPolynomial) : ℝ := p.c

/-- Theorem: If the mean of zeros, product of zeros, and sum of coefficients are equal,
    and the y-intercept is 2, then b = -11 -/
theorem cubic_polynomial_property (p : CubicPolynomial) 
  (h1 : mean_of_zeros p = product_of_zeros p)
  (h2 : mean_of_zeros p = sum_of_coefficients p)
  (h3 : y_intercept p = 2) :
  p.b = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l683_68325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_X_eq_two_l683_68380

/-- A discrete random variable with probability distribution P(X=i) = i/(2a) for i = 1, 2, 3 -/
noncomputable def X (a : ℝ) (i : ℕ) : ℝ := i / (2 * a)

/-- The sum of probabilities for all possible outcomes equals 1 -/
axiom sum_prob_eq_one (a : ℝ) : X a 1 + X a 2 + X a 3 = 1

/-- Theorem: For the given discrete random variable X, P(X=2) = 1/3 -/
theorem prob_X_eq_two :
  ∃ a : ℝ, X a 2 = 1/3 := by
  -- We'll use a = 3
  use 3
  -- Simplify the left-hand side
  simp [X]
  -- The result is true by arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_X_eq_two_l683_68380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_method_age_relation_l683_68397

-- Define the data from the problem
def total_sample : ℕ := 100
def under_60_mobile : ℕ := 40
def under_60_cash : ℕ := 10
def over_60_mobile : ℕ := 30
def over_60_cash : ℕ := 20

-- Define the K^2 formula
noncomputable def k_squared (a b c d : ℕ) : ℝ :=
  let n : ℝ := (a + b + c + d : ℝ)
  n * (a * d - b * c : ℝ)^2 / 
    ((a + b : ℝ) * (c + d : ℝ) * (a + c : ℝ) * (b + d : ℝ))

-- Define the binomial distribution
noncomputable def binomial_prob (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

-- Define the theorem
theorem payment_method_age_relation :
  k_squared under_60_mobile under_60_cash over_60_mobile over_60_cash < 6.635 ∧
  binomial_prob 3 (2/5) 0 = 27/125 ∧
  binomial_prob 3 (2/5) 1 = 54/125 ∧
  binomial_prob 3 (2/5) 2 = 36/125 ∧
  binomial_prob 3 (2/5) 3 = 8/125 ∧
  3 * (2/5 : ℝ) = 6/5 ∧
  3 * (2/5 : ℝ) * (1 - 2/5) = 18/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_method_age_relation_l683_68397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_14_power_14_equals_1_l683_68361

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 11^11  -- Add case for 0
  | 1 => 11^11
  | 2 => 12^12
  | 3 => 13^13
  | n+4 => Int.natAbs (a (n+3) - a (n+2)) + Int.natAbs (a (n+2) - a (n+1))

-- State the theorem
theorem a_14_power_14_equals_1 : a (14^14) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_14_power_14_equals_1_l683_68361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_sum_of_powers_l683_68312

theorem last_digit_sum_of_powers : ∃ n : ℕ, 
  (1023^3923 + n^3921) % 10 = 8 ∧ n % 10 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_sum_of_powers_l683_68312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_value_max_perimeter_l683_68320

open Real

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define vectors m and n
noncomputable def m (A : ℝ) : ℝ × ℝ := (2 * cos (A/2), sin (A/2))
noncomputable def n (A : ℝ) : ℝ × ℝ := (cos (A/2), -sin (A/2))

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem 1
theorem cos_A_value (tri : Triangle) : 
  dot_product (m tri.A) (n tri.A) = -1 → cos tri.A = -1/2 := by
  sorry

-- Theorem 2
theorem max_perimeter (tri : Triangle) : 
  tri.a = 2 * Real.sqrt 3 → 
  (∃ (p : ℝ), p = tri.a + tri.b + tri.c ∧ 
    ∀ (q : ℝ), q = tri.a + tri.b + tri.c → q ≤ p) → 
  tri.a + tri.b + tri.c = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_value_max_perimeter_l683_68320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dist_PQ_on_ellipse_l683_68377

noncomputable def ellipse_C (x y : ℝ) : Prop := y^2/16 + x^2/4 = 1

noncomputable def dist_PQ (x y t : ℝ) : ℝ := Real.sqrt (x^2 + (y - t)^2)

theorem min_dist_PQ_on_ellipse :
  ∀ t : ℝ,
  (∃ x y : ℝ, ellipse_C x y ∧ x^2 = 3 ∧ y = -2) →
  (∀ x y : ℝ, ellipse_C x y →
    dist_PQ x y t ≥ 
      (if t ≤ -3 then |t + 4|
       else if t < 3 then Real.sqrt (4 - t^2/3)
       else |t - 4|)) ∧
  (∃ x y : ℝ, ellipse_C x y ∧
    dist_PQ x y t = 
      (if t ≤ -3 then |t + 4|
       else if t < 3 then Real.sqrt (4 - t^2/3)
       else |t - 4|)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dist_PQ_on_ellipse_l683_68377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_distances_l683_68372

-- Define the parametric equation of line l
noncomputable def line_l (t θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 + t * Real.cos θ, t * Real.sin θ)

-- Define the polar equation of curve C
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 / (1 + 3 * Real.sin θ^2)

-- Define the standard equation of curve C
def curve_C_standard (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (θ : ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, A = line_l t₁ θ ∧ B = line_l t₂ θ ∧
  curve_C_standard A.1 A.2 ∧ curve_C_standard B.1 B.2

-- Define F₁ as the left focus of the ellipse
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- State the theorem
theorem min_product_of_distances :
  ∀ θ : ℝ, 0 ≤ θ → θ < π →
  ∀ A B : ℝ × ℝ, intersection_points A B θ →
  1 ≤ distance F₁ A * distance F₁ B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_distances_l683_68372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_quadrant_IV_l683_68338

-- Define the triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  A + B + C = Real.pi ∧ A > 0 ∧ B > 0 ∧ C > 0

-- Define an obtuse angle
def ObtuseAngle (angle : ℝ) : Prop := Real.pi / 2 < angle ∧ angle < Real.pi

-- Define the coordinates of point P
noncomputable def PointP (A B : ℝ) : ℝ × ℝ := (Real.tan B, Real.cos A)

-- Define Quadrant IV
def QuadrantIV (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_quadrant_IV (A B C : ℝ) :
  Triangle A B C → ObtuseAngle A → QuadrantIV (PointP A B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_quadrant_IV_l683_68338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_top_number_l683_68353

/-- Represents a layer in the pyramid --/
structure Layer where
  size : Nat
  numbers : List Nat

/-- Represents the entire pyramid --/
structure Pyramid where
  base : Layer
  layer2 : Layer
  layer3 : Layer
  top : Nat

/-- Calculates the average of four numbers, rounded down --/
def averageRoundDown (a b c d : Nat) : Nat :=
  (a + b + c + d) / 4

/-- Checks if a layer is valid according to the problem description --/
def isValidLayer (l : Layer) : Prop :=
  l.numbers.length = l.size * l.size

/-- Checks if the base layer is valid --/
def isValidBase (base : Layer) : Prop :=
  isValidLayer base ∧ base.size = 4 ∧ base.numbers.all (· ≥ 1) ∧ base.numbers.all (· ≤ 16)

/-- Checks if an upper layer is valid based on the layer below --/
def isValidUpperLayer (upper lower : Layer) : Prop :=
  isValidLayer upper ∧ 
  upper.size + 1 = lower.size ∧
  ∀ n ∈ upper.numbers, ∃ a b c d, a ∈ lower.numbers ∧ b ∈ lower.numbers ∧ 
                               c ∈ lower.numbers ∧ d ∈ lower.numbers ∧ 
                               n = averageRoundDown a b c d

/-- Checks if the entire pyramid is valid --/
def isValidPyramid (p : Pyramid) : Prop :=
  isValidBase p.base ∧
  isValidUpperLayer p.layer2 p.base ∧
  isValidUpperLayer p.layer3 p.layer2 ∧
  ∃ a b c d, a ∈ p.layer3.numbers ∧ b ∈ p.layer3.numbers ∧ 
             c ∈ p.layer3.numbers ∧ d ∈ p.layer3.numbers ∧ 
             p.top = averageRoundDown a b c d

theorem smallest_top_number (p : Pyramid) :
  isValidPyramid p → p.top ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_top_number_l683_68353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_multiplication_l683_68339

theorem floor_ceil_multiplication :
  3 * (Int.floor (-5.67 : ℝ) + Int.ceil (34.1 : ℝ)) = 87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_multiplication_l683_68339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_hyperbola_standard_equation_l683_68313

-- Parabola
def parabola_equation (x y : ℝ) : Prop := y^2 = 8*x

theorem parabola_standard_equation (directrix : ℝ) :
  directrix = -2 → ∃ f : ℝ → ℝ, ∀ x y : ℝ, parabola_equation x y ↔ y = f x :=
by sorry

-- Hyperbola
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2/4 = 1

theorem hyperbola_standard_equation (center : ℝ × ℝ) (focus_axis : ℝ → ℝ × ℝ) 
  (asymptote_slope : ℝ) (real_axis_length : ℝ) :
  center = (0, 0) →
  (∀ x : ℝ, focus_axis x = (x, 0)) →
  asymptote_slope = 2 →
  real_axis_length = 2 →
  ∃ f : ℝ → ℝ, ∀ x y : ℝ, hyperbola_equation x y ↔ y = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_hyperbola_standard_equation_l683_68313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ten_digit_max_digit_sum_l683_68350

def is_ten_digit (n : ℕ) : Prop := 1000000000 ≤ n ∧ n ≤ 9999999999

def digit_sum (n : ℕ) : ℕ :=
  let digits := String.toList (toString n)
  List.foldl (λ sum d => sum + (d.toNat - 48)) 0 digits

theorem smallest_ten_digit_max_digit_sum :
  ∀ n : ℕ, is_ten_digit n → n < 1899999999 → digit_sum n < digit_sum 1899999999 :=
by sorry

#eval digit_sum 1899999999  -- Should output 81

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ten_digit_max_digit_sum_l683_68350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_positive_and_negative_l683_68317

noncomputable def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => (k * sequence_a a₁ k + 1) / (k - sequence_a a₁ k)

theorem infinitely_many_positive_and_negative (a₁ : ℝ) :
  (∀ n : ℕ, ∃ k > n, sequence_a a₁ k > 0) ∧
  (∀ n : ℕ, ∃ k > n, sequence_a a₁ k < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_positive_and_negative_l683_68317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_xiaoming_triangle_problem_l683_68307

def is_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b

theorem triangle_inequality (a b c : ℝ) : 
  is_triangle a b c ↔ (0 < a ∧ 0 < b ∧ 0 < c) ∧ (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

theorem xiaoming_triangle_problem : 
  is_triangle 2 5 4 ∧ 
  ¬ is_triangle 2 5 1 ∧
  ¬ is_triangle 2 5 2 ∧
  ¬ is_triangle 2 5 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_xiaoming_triangle_problem_l683_68307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_is_825_l683_68334

/-- Given an initial amount P and an interest rate R,
    calculate the final amount after 3 years of simple interest -/
noncomputable def final_amount (P : ℝ) (R : ℝ) : ℝ :=
  P * (1 + 3 * R / 100)

/-- The problem statement -/
theorem initial_amount_is_825 :
  ∃ (P R : ℝ),
    final_amount P R = 956 ∧
    final_amount P (R + 4) = 1055 ∧
    P = 825 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_is_825_l683_68334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l683_68327

theorem sum_of_angles (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3)
  (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) : α + β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l683_68327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l683_68345

theorem simplify_fraction (y : ℝ) (h : y ≠ 0) :
  (5 / (4 * y^4)) * ((4 * y^3) / 3) = (5 * y^7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l683_68345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l683_68363

theorem inequality_proof (a b c : ℝ) : 
  a = (1 : ℝ) / 2023 → 
  b = Real.log (2024 / 2023) → 
  c = (Real.log (2024 / 2023)) / (Real.log 5) → 
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l683_68363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_in_terms_of_y_l683_68387

theorem cot_in_terms_of_y (θ : ℝ) (y : ℝ) 
  (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_y : y > 1) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((y + 1) / (2 * y))) : 
  Real.tan θ⁻¹ = 1 / Real.sqrt (y^2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_in_terms_of_y_l683_68387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l683_68357

-- Define the set M as a function of a
def M (a : ℝ) : Set ℝ := {x | (a*x - 5) / (x^2 - a) < 0}

-- Define the range of a
def a_range : Set ℝ := Set.Icc 1 (5/3) ∪ Set.Ioc 9 25

-- Theorem statement
theorem solution_range :
  ∀ a : ℝ, (3 ∈ M a ∧ 5 ∉ M a) ↔ a ∈ a_range :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l683_68357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l683_68322

noncomputable def ellipseEccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def hyperbolaEccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem ellipse_eccentricity_sqrt_two_over_two 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipseEccentricity a b * hyperbolaEccentricity a b = Real.sqrt 3 / 2) :
  ellipseEccentricity a b = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l683_68322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l683_68309

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def primes_up_to_29 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def Q : ℕ := primes_up_to_29.prod

def sequence_terms : List ℕ := (List.range 14).map (λ i => Q + primes_up_to_29[i+1]!)

theorem no_primes_in_sequence :
  ∀ n, n ∈ sequence_terms → ¬(is_prime n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l683_68309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l683_68359

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P
noncomputable def point_P : ℝ × ℝ := (Real.sqrt 3, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 4 = 0

-- Theorem statement
theorem tangent_line_proof :
  (my_circle point_P.1 point_P.2) ∧
  (∀ x y, my_circle x y → tangent_line x y → (x, y) = point_P) ∧
  (tangent_line point_P.1 point_P.2) := by
  sorry

#check tangent_line_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l683_68359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_calculation_l683_68362

/-- Represents the cost structure of a restaurant meal -/
structure MealCost where
  preTax : ℝ
  taxRate : ℝ
  tipRate : ℝ
  total : ℝ

/-- Calculates the total cost of a meal given pre-tax cost, tax rate, and tip rate -/
def totalCost (m : MealCost) : ℝ :=
  m.preTax * (1 + m.taxRate + m.tipRate)

/-- Theorem stating that given specific conditions, the pre-tax cost of a meal is approximately $36.04 -/
theorem meal_cost_calculation (m : MealCost) 
    (h1 : m.taxRate = 0.095)
    (h2 : m.tipRate = 0.18)
    (h3 : m.total = 45.95)
    (h4 : totalCost m = m.total) :
    ∃ ε > 0, |m.preTax - 36.04| < ε := by
  sorry

#check meal_cost_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_cost_calculation_l683_68362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_range_l683_68333

theorem complex_number_range (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ = Real.sqrt 2)
  (h₂ : Complex.abs z₂ = Real.sqrt 2)
  (h₃ : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h₄ : Complex.abs (z₁ + z₂ - z₃) = 1) :
  1 ≤ Complex.abs z₃ ∧ Complex.abs z₃ ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_range_l683_68333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_R_value_l683_68366

-- Define a right triangle PQR
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  right_angle : (P.1 - Q.1) * (P.1 - R.1) + (P.2 - Q.2) * (P.2 - R.2) = 0

-- Define the tangent of an angle R in a right triangle
noncomputable def tan_R (t : RightTriangle) : ℝ :=
  abs ((t.Q.2 - t.P.2) / (t.Q.1 - t.P.1))

-- Define the cosine of an angle R in a right triangle
noncomputable def cos_R (t : RightTriangle) : ℝ :=
  abs ((t.R.1 - t.P.1) / Real.sqrt ((t.R.1 - t.P.1)^2 + (t.R.2 - t.P.2)^2))

-- Theorem statement
theorem cos_R_value (t : RightTriangle) (h : tan_R t = 1/2) :
  cos_R t = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_R_value_l683_68366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l683_68393

noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 2*x - 5) / (x - 4)

theorem slant_asymptote_sum (m b : ℝ) :
  (∀ ε > 0, ∃ M, ∀ x > M, |f x - (m*x + b)| < ε) → m + b = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l683_68393
