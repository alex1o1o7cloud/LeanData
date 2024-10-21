import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_range_l841_84147

noncomputable def f (x : ℝ) : ℝ := x^3 + 2

noncomputable def curvature (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := 3 * x₁^2
  let k₂ := 3 * x₂^2
  |k₁ - k₂| / Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem curvature_range (x₁ x₂ : ℝ) (hx : x₁ + x₂ = 2) (hne : x₁ ≠ x₂) :
  0 < curvature x₁ x₂ ∧ curvature x₁ x₂ < 3 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_range_l841_84147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_proof_l841_84162

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def circle_radius : ℝ := Real.sqrt 5

def line_slope : ℚ := -3/4
def line_intercept : ℚ := 1

def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

theorem line_slope_proof :
  let d := Real.sqrt 3 -- Half of |AB|
  let c := circle_center
  let r := circle_radius
  let m := line_slope
  let b := line_intercept
  let A := point_A
  let B := point_B
  (A.1 - 3)^2 + A.2^2 = r^2 ∧  -- A is on the circle
  (B.1 - 3)^2 + B.2^2 = r^2 ∧  -- B is on the circle
  A.2 = m * (A.1 - 1) + b ∧    -- A is on the line
  B.2 = m * (B.1 - 1) + b ∧    -- B is on the line
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*d)^2  -- Distance between A and B
  →
  m = -3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_proof_l841_84162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_productivity_increase_l841_84149

/-- Represents the mowing productivity increase problem -/
theorem mowing_productivity_increase 
  (initial_productivity : ℝ) 
  (total_area : ℝ) 
  (final_days : ℕ) :
  initial_productivity = 15 →
  total_area = 20 →
  final_days = 1 →
  abs ((total_area / final_days - initial_productivity) / initial_productivity * 100 - 100/3) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_productivity_increase_l841_84149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawyer_fee_correct_l841_84129

/-- Approximate equality for rational numbers, with tolerance 0.01 --/
def approx_eq (x y : ℚ) : Prop := abs (x - y) < 1/100

infixl:50 " ≈ " => approx_eq

/-- Calculates the lawyer's fee and John's payment --/
def lawyer_fee (upfront_fee : ℚ) (court_hours : ℚ) (total_payment : ℚ) : ℚ × ℚ :=
  let prep_hours := 2 * court_hours
  let total_hours := court_hours + prep_hours
  let hourly_fee := (total_payment - upfront_fee) / total_hours
  let johns_payment := total_payment / 2 + upfront_fee
  (hourly_fee, johns_payment)

theorem lawyer_fee_correct :
  let result := lawyer_fee 1000 50 8000
  result.1 ≈ 46.67 ∧ result.2 = 5000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawyer_fee_correct_l841_84129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l841_84166

/-- The solution set of the inequality √(a² - x²) > 2x + a -/
def SolutionSet (a : ℝ) : Set ℝ :=
  if a > 0 then 
    Set.Icc (-a) 0 \ {0}
  else if a = 0 then 
    ∅ 
  else 
    Set.Ico a (-4/5 * a)

/-- Theorem stating that SolutionSet correctly describes the solutions of the inequality -/
theorem inequality_solution (a : ℝ) (x : ℝ) : 
  x ∈ SolutionSet a ↔ Real.sqrt (a^2 - x^2) > 2*x + a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l841_84166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l841_84112

-- Define the parametric functions
noncomputable def x (t : ℝ) : ℝ := t + Real.sin t
noncomputable def y (t : ℝ) : ℝ := 2 + Real.cos t

-- Define the second derivative
noncomputable def y_xx_double_prime (t : ℝ) : ℝ := -1 / (1 + Real.cos t)^2

-- Theorem statement
theorem second_derivative_parametric_function (t : ℝ) :
  (y_xx_double_prime t) = -1 / (1 + Real.cos t)^2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l841_84112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_calculation_l841_84144

/-- Represents the total house rent -/
def R : ℝ := sorry

/-- Represents Purity's share of the rent -/
def P : ℝ := sorry

/-- Sheila's share of the rent -/
def sheila_share : ℝ := 5 * P

/-- Rose's share of the rent -/
def rose_share : ℝ := 3 * P

/-- John's share of the rent -/
def john_share : ℝ := 4 * P

theorem total_rent_calculation (h1 : rose_share = 1800) :
  R = 7800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_calculation_l841_84144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_fourths_l841_84191

theorem cos_five_pi_fourths : Real.cos (5 * π / 4) = -1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_fourths_l841_84191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l841_84128

noncomputable def f (a b x : ℝ) : ℝ := (x^2 - (a + 3) * x + b) * Real.exp x

theorem tangent_line_and_monotonicity (a b : ℝ) :
  (∀ x, f (-3) 0 x = x^2 * Real.exp x) ∧
  (∀ x, deriv (f (-3) 0) x = (x^2 + 2*x) * Real.exp x) ∧
  (3 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 ↔ 
    y = deriv (f (-3) 0) 1 * (x - 1) + f (-3) 0 1) ∧
  (deriv (f a b) 1 = 0 → b = 2*a + 3) ∧
  (deriv (f a b) 1 = 0 → 
    (a = 1 → ∀ x, deriv (f a (2*a + 3)) x ≥ 0) ∧
    (a > 1 → ∀ x, (x > a ∨ x < 1) → deriv (f a (2*a + 3)) x > 0) ∧
    (a > 1 → ∀ x, (1 < x ∧ x < a) → deriv (f a (2*a + 3)) x < 0) ∧
    (a < 1 → ∀ x, (x > 1 ∨ x < a) → deriv (f a (2*a + 3)) x > 0) ∧
    (a < 1 → ∀ x, (a < x ∧ x < 1) → deriv (f a (2*a + 3)) x < 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l841_84128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_line_l841_84106

-- Define the set T
def T : Set ℂ := {w : ℂ | ∃ (r : ℝ), (2 + 5*Complex.I) * w = r}

-- Theorem statement
theorem T_is_line : ∃ (a b : ℝ), T = {w : ℂ | ∃ (t : ℝ), w = t * Complex.mk a b} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_line_l841_84106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_mexican_meals_l841_84155

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_meals_per_week : ℕ := sorry

/-- Represents the number of antacids Nancy takes per day for each type of meal -/
def antacids_per_day : ℕ → ℕ → ℕ → ℕ := sorry

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_meals_per_week : ℕ := 3

/-- Represents the total number of antacids Nancy takes per month -/
def total_antacids_per_month : ℕ := 60

/-- Represents the number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

theorem nancy_mexican_meals :
  mexican_meals_per_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_mexican_meals_l841_84155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_conditions_l841_84105

-- Define a structure for Tetrahedron
structure Tetrahedron (α : Type*) where
  vertices : Fin 4 → α
  edges : Fin 6 → α

-- Define a structure for Edge
structure Edge (α : Type*) where
  length : α

theorem tetrahedron_edge_conditions (a : ℝ) (h : a > 0) :
  (∃ (t : Tetrahedron ℝ), 
    (∃ (k : Fin 6), k.val + 1 = 1 ∧ 
      (∀ (e : Edge ℝ), e.length = a ∨ e.length = 1) ∧
      (∃ (edges : Finset (Edge ℝ)), edges.card = k.val + 1 ∧ 
        (∀ e ∈ edges, e.length = a) ∧
        (∀ e ∉ edges, e.length = 1))) →
    a < Real.sqrt 3) ∧
  (∃ (t : Tetrahedron ℝ), 
    (∃ (k : Fin 6), k.val + 1 = 2 ∧ 
      (∀ (e : Edge ℝ), e.length = a ∨ e.length = 1) ∧
      (∃ (edges : Finset (Edge ℝ)), edges.card = k.val + 1 ∧ 
        (∀ e ∈ edges, e.length = a) ∧
        (∀ e ∉ edges, e.length = 1))) →
    a < Real.sqrt (2 + Real.sqrt 3)) ∧
  (∃ (t : Tetrahedron ℝ), 
    (∃ (k : Fin 6), k.val + 1 = 3 ∧ 
      (∀ (e : Edge ℝ), e.length = a ∨ e.length = 1) ∧
      (∃ (edges : Finset (Edge ℝ)), edges.card = k.val + 1 ∧ 
        (∀ e ∈ edges, e.length = a) ∧
        (∀ e ∉ edges, e.length = 1))) →
    a > 0) ∧
  (∃ (t : Tetrahedron ℝ), 
    (∃ (k : Fin 6), k.val + 1 = 4 ∧ 
      (∀ (e : Edge ℝ), e.length = a ∨ e.length = 1) ∧
      (∃ (edges : Finset (Edge ℝ)), edges.card = k.val + 1 ∧ 
        (∀ e ∈ edges, e.length = a) ∧
        (∀ e ∉ edges, e.length = 1))) →
    a > Real.sqrt (2 - Real.sqrt 3)) ∧
  (∃ (t : Tetrahedron ℝ), 
    (∃ (k : Fin 6), k.val + 1 = 5 ∧ 
      (∀ (e : Edge ℝ), e.length = a ∨ e.length = 1) ∧
      (∃ (edges : Finset (Edge ℝ)), edges.card = k.val + 1 ∧ 
        (∀ e ∈ edges, e.length = a) ∧
        (∀ e ∉ edges, e.length = 1))) →
    a > 1 / Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_conditions_l841_84105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_in_grade_l841_84142

/-- 
Given a grade where a student is both the 75th best and 75th worst,
prove that the total number of students in the grade is 149.
-/
theorem student_count_in_grade {grade : Type} [Fintype grade] (misha : grade) 
  (h_best : ∃ (better : Finset grade), better.card = 74 ∧ ∀ s ∈ better, s ≠ misha)
  (h_worst : ∃ (worse : Finset grade), worse.card = 74 ∧ ∀ s ∈ worse, s ≠ misha) :
  Fintype.card grade = 149 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_in_grade_l841_84142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_die_expected_value_l841_84125

open BigOperators

def unfair_die_probability (n : ℕ) : ℚ :=
  if n = 8 then 1/3
  else if n ≤ 4 then 1/15
  else if n ≤ 7 then 2/15
  else 0

def expected_value : ℚ :=
  ∑ n in Finset.range 9, n * unfair_die_probability n

theorem unfair_die_expected_value :
  expected_value = 116/15 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_die_expected_value_l841_84125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_equals_interval_intersection_A_B_implies_a_equals_2_l841_84127

-- Define the domain A
noncomputable def A : Set ℝ := {x : ℝ | -x^2 + x + 2 > 0}

-- Define the exponential function
noncomputable def exp_func (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the range B
noncomputable def B (a : ℝ) : Set ℝ := {y : ℝ | ∃ x ∈ A, y = exp_func a x}

-- Theorem 1
theorem union_A_B_equals_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a = 2 → A ∪ B a = Set.Ioo (-1) 4 := by sorry

-- Theorem 2
theorem intersection_A_B_implies_a_equals_2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  A ∩ B a = Set.Ioo (1/2) 2 → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_A_B_equals_interval_intersection_A_B_implies_a_equals_2_l841_84127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_chords_sum_squares_constant_l841_84170

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A chord of a circle -/
structure Chord (c : Circle) where
  start : Point
  endpoint : Point
  on_circle : (start.1 - c.center.1)^2 + (start.2 - c.center.2)^2 = c.radius^2 ∧
              (endpoint.1 - c.center.1)^2 + (endpoint.2 - c.center.2)^2 = c.radius^2

/-- Two chords are perpendicular -/
def perpendicular (c : Circle) (c1 c2 : Chord c) : Prop :=
  let v1 := (c1.endpoint.1 - c1.start.1, c1.endpoint.2 - c1.start.2)
  let v2 := (c2.endpoint.1 - c2.start.1, c2.endpoint.2 - c2.start.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- The length of a chord -/
noncomputable def chord_length (c : Circle) (chord : Chord c) : ℝ :=
  Real.sqrt ((chord.endpoint.1 - chord.start.1)^2 + (chord.endpoint.2 - chord.start.2)^2)

/-- Point lies on chord -/
def point_on_chord (p : Point) (c : Circle) (chord : Chord c) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.1 = chord.start.1 + t * (chord.endpoint.1 - chord.start.1) ∧
    p.2 = chord.start.2 + t * (chord.endpoint.2 - chord.start.2)

/-- The theorem statement -/
theorem perpendicular_chords_sum_squares_constant 
  (c : Circle) (k : Point) (inside : (k.1 - c.center.1)^2 + (k.2 - c.center.2)^2 < c.radius^2) :
  ∃ (const : ℝ), ∀ (c1 c2 : Chord c), 
    perpendicular c c1 c2 → 
    point_on_chord k c c1 → 
    point_on_chord k c c2 → 
    (chord_length c c1)^2 + (chord_length c c2)^2 = const :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_chords_sum_squares_constant_l841_84170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_and_g_relationship_explicit_f_f_of_3_l841_84119

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 2*x) + (x - 1)^0 + 1 / (x + 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < -1 ∨ (-1 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)}

-- Define the alternative function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem for the domain of f
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ f x ∈ Set.univ :=
sorry

-- Theorem for the relationship between f and g
theorem f_and_g_relationship : 
  ∀ x : ℝ, f (x + 1) = g x :=
sorry

-- Theorem for the explicit form of f
theorem explicit_f : 
  ∀ x : ℝ, f x = x^2 - 4*x + 3 :=
sorry

-- Theorem for the value of f(3)
theorem f_of_3 : f 3 = 0 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_and_g_relationship_explicit_f_f_of_3_l841_84119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_negative_when_x_negative_l841_84145

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 5) + 4

-- Define the fixed point M(m, n)
def m : ℝ := -5
def n : ℝ := 5

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := m + n^x

-- Theorem statement
theorem g_negative_when_x_negative 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x : ℝ, f a x = f a (-5)) :
  ∀ x : ℝ, x < 0 → g x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_negative_when_x_negative_l841_84145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l841_84133

theorem negation_of_proposition :
  (∃ x : ℝ, x < 1) ↔ ¬(∀ x : ℝ, x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l841_84133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l841_84111

theorem trig_equation_solution (x : ℝ) : 
  (Real.cos x - Real.sin x)^2 + Real.cos x^4 - Real.sin x^4 = 0.5 * Real.sin (4*x) ↔ 
  (∃ n : ℤ, x = Real.pi/2 * (2*n + 1)) ∨ (∃ k : ℤ, x = Real.pi/4 * (4*k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l841_84111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_polar_line_l841_84168

-- Define the circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the polar line
noncomputable def polarLine (A : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the tangent line at a point on the circle
noncomputable def tangentLine (B : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define a line perpendicular to a given line passing through a point
noncomputable def perpendicularLine (l : Set (EuclideanSpace ℝ (Fin 2))) (O : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the boundary of a circle
def Circle.boundary (c : Circle) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {x | ‖x - c.center‖ = c.radius}

-- Define a line through two points
noncomputable def Line.throughPoints (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Main theorem
theorem locus_is_polar_line (c : Circle) (A : EuclideanSpace ℝ (Fin 2)) :
  ∀ B : EuclideanSpace ℝ (Fin 2), B ∈ c.boundary →
    let tangent := tangentLine B c
    let perpendicular := perpendicularLine (Line.throughPoints A B) c.center
    (Set.inter tangent perpendicular) ⊆ polarLine A c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_polar_line_l841_84168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_135_degrees_l841_84188

def vector_AB : Fin 3 → ℝ := ![2, 4, 0]
def vector_BC : Fin 3 → ℝ := ![-1, 3, 0]

theorem angle_ABC_is_135_degrees :
  let vector_BA : Fin 3 → ℝ := ![-vector_AB 0, -vector_AB 1, -vector_AB 2]
  Real.arccos (
    (vector_BA 0 * vector_BC 0 + vector_BA 1 * vector_BC 1 + vector_BA 2 * vector_BC 2) /
    (Real.sqrt ((vector_BA 0)^2 + (vector_BA 1)^2 + (vector_BA 2)^2) *
     Real.sqrt ((vector_BC 0)^2 + (vector_BC 1)^2 + (vector_BC 2)^2))
  ) = 135 * Real.pi / 180 := by
  sorry

#eval vector_AB
#eval vector_BC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_135_degrees_l841_84188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l841_84122

-- Define the time it takes for the train to pass the oak tree (in seconds)
noncomputable def passing_time : ℝ := 11.999040076793857

-- Define the speed of the train (in km/hr)
noncomputable def train_speed : ℝ := 54

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

-- Calculate the train's speed in m/s
noncomputable def train_speed_m_s : ℝ := train_speed * km_hr_to_m_s

-- Define the length of the train
noncomputable def train_length : ℝ := train_speed_m_s * passing_time

-- Theorem to prove
theorem train_length_approximation :
  ∃ ε > 0, |train_length - 179.99| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l841_84122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_y_fourth_quadrant_l841_84156

theorem tan_half_y_fourth_quadrant 
  (x y : Real) 
  (h1 : Real.cos (x + y) * Real.sin x - Real.sin (x + y) * Real.cos x = 12/13)
  (h2 : 3*Real.pi/2 < y) 
  (h3 : y < 2*Real.pi) : 
  Real.tan (y/2) = -2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_y_fourth_quadrant_l841_84156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_increasing_l841_84184

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

-- State the theorem
theorem min_a_for_monotonic_increasing :
  ∃ (a_min : ℝ), ∀ (a : ℝ), a ≥ a_min ↔
    (∀ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 → f a x ≤ f a y) ∧
    a_min = Real.exp (-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotonic_increasing_l841_84184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l841_84159

variable (p q r s : ℝ)

def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 1
def g (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s

theorem polynomial_division_theorem :
  (∃ h : ℝ → ℝ, ∀ x, g p q r s x = (f x) * (h x)) →
  (p + q + r) * s = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l841_84159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_properties_l841_84137

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

noncomputable def distance_focus_to_asymptote (a b : ℝ) : ℝ :=
  b * Real.sqrt (1 - b^2 / a^2) / Real.sqrt (a^2 + b^2)

def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  y = m * x - 2

theorem hyperbola_and_line_properties
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 3)
  (h4 : distance_focus_to_asymptote a b = Real.sqrt 2) :
  (∀ x y, hyperbola a b x y ↔ hyperbola 1 (Real.sqrt 2) x y) ∧
  (∃ m, m = 1 ∨ m = -1) ∧
  (∀ m, m = 1 ∨ m = -1 →
    ∀ x y, line_equation m x y → 
      (∃ A B : ℝ × ℝ, 
        hyperbola 1 (Real.sqrt 2) A.1 A.2 ∧
        hyperbola 1 (Real.sqrt 2) B.1 B.2 ∧
        line_equation m A.1 A.2 ∧
        line_equation m B.1 B.2 ∧
        (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * ((A.1 + B.1)/2)^2 + 4 * ((A.2 + B.2)/2)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_properties_l841_84137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_addition_l841_84130

-- Define the ※ operation
noncomputable def star (a b : ℝ) : ℝ := Real.log b / Real.log a

-- Theorem statement
theorem star_addition (x y : ℝ) :
  star 4 5 + star 4 6 = star 4 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_addition_l841_84130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_log_eq_solution_set_exp_eq_l841_84134

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Theorem for proposition (②)
theorem solution_set_log_eq (x : ℝ) : 
  log10 (Real.sqrt (x - 2)) = log10 (x - 2) ↔ x = 3 :=
sorry

-- Theorem for proposition (③)
theorem solution_set_exp_eq (x : ℝ) :
  (3 : ℝ)^(1 - x) - 2 = 0 ↔ x = 1 - Real.log 2 / Real.log 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_log_eq_solution_set_exp_eq_l841_84134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cindy_distance_proof_l841_84102

/-- Cindy's running speed in miles per hour -/
noncomputable def running_speed : ℝ := 3

/-- Cindy's walking speed in miles per hour -/
noncomputable def walking_speed : ℝ := 1

/-- Total time taken for the journey in hours -/
noncomputable def total_time : ℝ := 40 / 60

/-- The distance Cindy runs and walks -/
noncomputable def distance : ℝ := 1 / 2

/-- Proof that the given distance satisfies the problem conditions -/
theorem cindy_distance_proof :
  (distance / running_speed) + (distance / walking_speed) = total_time :=
by
  -- Expand the definitions
  unfold distance running_speed walking_speed total_time
  -- Perform algebraic simplification
  simp [add_div]
  -- The proof is completed with numerical computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cindy_distance_proof_l841_84102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_gain_calculation_l841_84169

/-- Compound interest calculation --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- The problem statement --/
theorem b_gain_calculation (principal : ℝ) (rate_ab : ℝ) (rate_bc : ℝ) 
  (time_ab : ℝ) (time_bc : ℝ) (compounds_ab : ℝ) (compounds_bc : ℝ) :
  principal = 3150 ∧ 
  rate_ab = 0.08 ∧ 
  rate_bc = 0.125 ∧
  time_ab = 3 ∧ 
  time_bc = 2.5 ∧
  compounds_ab = 1 ∧
  compounds_bc = 2 →
  let amount_ab := compound_interest principal rate_ab compounds_ab time_ab
  let amount_bc := compound_interest principal rate_bc compounds_bc time_bc
  let gain := amount_bc - amount_ab
  ∃ ε > 0, |gain - 282.32| < ε := by
  sorry

#eval 1 -- This line is added to ensure the file is not empty after the sorry tactic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_gain_calculation_l841_84169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l841_84189

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

-- Define the length of a line segment
noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the angle bisector
def is_angle_bisector (P Q R S : ℝ × ℝ) : Prop :=
  (length P S) / (length S Q) = (length P R) / (length Q R)

-- State the theorem
theorem angle_bisector_length 
  (P Q R S : ℝ × ℝ) 
  (h₁ : Triangle P Q R) 
  (h₂ : length P Q = 3) 
  (h₃ : length Q R = 3 * Real.sqrt 3) 
  (h₄ : is_angle_bisector P R Q S) 
  (h₅ : S.1 = P.1 + length P S * (Q.1 - P.1) / length P Q) 
  (h₆ : S.2 = P.2 + length P S * (Q.2 - P.2) / length P Q) :
  length P S = 6 * Real.sqrt 3 - 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l841_84189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_axis_l841_84179

theorem cosine_symmetry_axis (k : ℤ) : 
  (∀ x : ℝ, Real.cos x = Real.cos (-x)) → 
  (∀ x : ℝ, Real.cos (2*x + Real.pi/6) = Real.cos (-2*x - Real.pi/6)) → 
  (∀ x : ℝ, Real.cos x = Real.cos (2*(↑k*Real.pi - x))) →
  (∃ n : ℤ, -Real.pi/12 = ↑n*Real.pi/2 - Real.pi/12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_axis_l841_84179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_curve_symmetry_l841_84186

/-- A normal distribution with mean μ and standard deviation σ. -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The property of a normal curve being symmetric about the y-axis. -/
def symmetric_about_y_axis (nd : NormalDistribution) : Prop :=
  ∀ x : ℝ, nd.σ * Real.sqrt (2 * Real.pi) * Real.exp (-((x - nd.μ)^2 / (2 * nd.σ^2))) =
            nd.σ * Real.sqrt (2 * Real.pi) * Real.exp (-(((-x) - nd.μ)^2 / (2 * nd.σ^2)))

/-- 
If a normal curve is symmetric about the y-axis, 
then the mean of the corresponding normal distribution is 0.
-/
theorem normal_curve_symmetry (nd : NormalDistribution) :
  symmetric_about_y_axis nd → nd.μ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_curve_symmetry_l841_84186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_eq_z_l841_84176

/-- The function t(x) -/
def t (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 8

/-- The function s(t(x)) -/
noncomputable def s (y : ℝ) : ℝ := 
  let x := ((-1 + Real.sqrt 31) / 3 : ℝ)
  x^3 + 2 * x^2 + 3 * x - 2

/-- The value of z -/
noncomputable def z : ℝ := 
  let x := ((-1 + Real.sqrt 31) / 3 : ℝ)
  x^3 / 27 + 2 * x^2 / 9 + x - 2

/-- Theorem stating that s(2) = z -/
theorem s_of_2_eq_z : s 2 = z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_2_eq_z_l841_84176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_example_l841_84158

/-- Given a triangle with inradius r and area A, calculate its perimeter p -/
noncomputable def triangle_perimeter (r : ℝ) (A : ℝ) : ℝ :=
  (2 * A) / r

/-- Theorem: A triangle with inradius 4.5 cm and area 78.75 cm² has a perimeter of 35 cm -/
theorem triangle_perimeter_example : triangle_perimeter 4.5 78.75 = 35 := by
  -- Unfold the definition of triangle_perimeter
  unfold triangle_perimeter
  -- Simplify the expression
  simp
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_example_l841_84158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l841_84100

noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

def conic_section (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2/m = 1

noncomputable def eccentricity_ellipse (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

noncomputable def eccentricity_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + b^2/a^2)

theorem conic_section_eccentricity :
  ∀ m : ℝ, m = geometric_mean 2 8 →
  (∃ e : ℝ, (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧
   (∀ x y : ℝ, conic_section m x y →
    (m > 0 → e = eccentricity_ellipse 2 1) ∧
    (m < 0 → e = eccentricity_hyperbola 1 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l841_84100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_chest_value_change_l841_84107

/-- Represents the contents and value of a chest filled with golden and silver sand -/
structure SandChest where
  silver_volume : ℝ
  gold_volume : ℝ
  silver_value : ℝ
  gold_value : ℝ

/-- The initial state of the chest -/
def initial_chest : SandChest := {
  silver_volume := 1
  gold_volume := 2
  silver_value := 1
  gold_value := 7
}

/-- Calculates the total value of the chest -/
noncomputable def chest_value (chest : SandChest) : ℝ :=
  chest.silver_volume * chest.silver_value + chest.gold_volume * chest.gold_value

/-- Represents the chest after removing half of the silver sand and replacing it with gold -/
noncomputable def silver_to_gold (chest : SandChest) : SandChest := {
  silver_volume := chest.silver_volume / 2
  gold_volume := chest.gold_volume + chest.silver_volume / 2
  silver_value := chest.silver_value
  gold_value := chest.gold_value
}

/-- Represents the chest after removing half of the gold sand and replacing it with silver -/
noncomputable def gold_to_silver (chest : SandChest) : SandChest := {
  silver_volume := chest.silver_volume + chest.gold_volume / 2
  gold_volume := chest.gold_volume / 2
  silver_value := chest.silver_value
  gold_value := chest.gold_value
}

theorem sand_chest_value_change :
  chest_value (silver_to_gold initial_chest) = 1.2 * chest_value initial_chest →
  chest_value (gold_to_silver initial_chest) = 0.6 * chest_value initial_chest := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_chest_value_change_l841_84107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_line_plane_l841_84160

/-- The sine of the angle between a line and a plane -/
theorem sine_angle_line_plane :
  let d : Fin 3 → ℝ := ![3, 4, 5]  -- direction vector of the line
  let n : Fin 3 → ℝ := ![4, 5, -2]  -- normal vector of the plane
  let dot_product := d 0 * n 0 + d 1 * n 1 + d 2 * n 2
  let magnitude_d := Real.sqrt (d 0^2 + d 1^2 + d 2^2)
  let magnitude_n := Real.sqrt (n 0^2 + n 1^2 + n 2^2)
  let sin_theta := dot_product / (magnitude_d * magnitude_n)
  sin_theta = 11 * Real.sqrt 10 / 75
  := by sorry

#check sine_angle_line_plane

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_line_plane_l841_84160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l841_84120

noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

def line_through_focus (p : ℝ) (x y : ℝ) : Prop := y = x + p/2

def intersection_points (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2 ∧ parabola p N.1 N.2 ∧
  line_through_focus p M.1 M.2 ∧ line_through_focus p N.1 N.2

noncomputable def distance_MN (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

def tangent_line (x y : ℝ) : Prop := y = x - 1

def dot_product_PM_PN (P M N : ℝ × ℝ) : ℝ :=
  (M.1 - P.1) * (N.1 - P.1) + (M.2 - P.2) * (N.2 - P.2)

theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ) :
  parabola p M.1 M.2 ∧ parabola p N.1 N.2 ∧
  intersection_points p M N ∧
  distance_MN M N = 8 →
  (∃ (x y : ℝ), parabola 2 x y) ∧
  (∃ (P : ℝ × ℝ), tangent_line P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ), tangent_line Q.1 Q.2 →
      dot_product_PM_PN P M N ≤ dot_product_PM_PN Q M N ∧
      dot_product_PM_PN P M N = -32) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l841_84120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_range_max_traffic_flow_speed_max_traffic_flow_l841_84175

/-- The traffic flow function --/
noncomputable def y (v : ℝ) : ℝ := 240 * v / (v^2 + 20 * v + 1600)

/-- The range of v for which y > 2 --/
theorem traffic_flow_range (v : ℝ) (h : v > 0) :
  y v > 2 ↔ 20 < v ∧ v < 80 := by
  sorry

/-- The average speed at which the traffic flow is maximum --/
theorem max_traffic_flow_speed :
  ∃ (v : ℝ), v > 0 ∧ ∀ (u : ℝ), u > 0 → y v ≥ y u := by
  sorry

/-- The maximum traffic flow --/
theorem max_traffic_flow :
  ∃ (v : ℝ), v > 0 ∧ y v = 2.4 ∧ ∀ (u : ℝ), u > 0 → y u ≤ 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_range_max_traffic_flow_speed_max_traffic_flow_l841_84175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_weight_l841_84143

/-- The initial average weight of A, B, and C -/
def W : ℝ := sorry

/-- The weight of person D -/
def D : ℝ := sorry

/-- The weight of person A -/
def A : ℝ := 81

/-- The sum of weights of A, B, and C -/
def sum_ABC : ℝ := 3 * W

/-- The sum of weights of A, B, C, and D -/
def sum_ABCD : ℝ := 4 * W

/-- The sum of weights of B, C, D, and E -/
def sum_BCDE : ℝ := 4 * 68

theorem initial_average_weight :
  (sum_ABCD = sum_ABC + D) →
  (sum_BCDE = sum_ABC - A + D + (D + 3)) →
  (W = 70) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_weight_l841_84143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_line_k_l841_84173

/-- A line in the coordinate plane. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The x-intercept of a line. -/
noncomputable def x_intercept (l : Line) : ℝ := -l.y_intercept / l.slope

/-- A line k where the slope is 0.5 times its y-intercept. -/
noncomputable def line_k (b : ℝ) : Line :=
  { slope := 0.5 * b, y_intercept := b }

/-- The x-intercept of line k is -2. -/
theorem x_intercept_of_line_k (b : ℝ) :
  x_intercept (line_k b) = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_line_k_l841_84173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l841_84180

-- Define the roots of the equation
noncomputable def root1 (a : ℝ) : ℝ := 2*a + 2*(a^2 - 3*a + 2).sqrt
noncomputable def root2 (a : ℝ) : ℝ := 2*a - 2*(a^2 - 3*a + 2).sqrt
noncomputable def root3 (a : ℝ) : ℝ := (3*a - 2)/a

-- Theorem statement
theorem equation_roots (a : ℝ) :
  (a ≥ 2/3 ∧ a < 1 ∨ a > 2) →
    (root1 a ≥ 0 ∧ root2 a ≥ 0 ∧ root1 a ≠ root2 a) ∧
  (0 < a ∧ a < 2/3) →
    (root3 a < 0 ∧ root1 a ≥ 0 ∧ root3 a ≠ root1 a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l841_84180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l841_84197

/-- Given real numbers x, y, z satisfying the specified conditions, prove that m + n = 9 -/
theorem triangle_problem (x y z : ℝ) (m n : ℕ+) 
  (hx : x = Real.sqrt (y^2 - 1/16) + Real.sqrt (z^2 - 1/16))
  (hy : y = Real.sqrt (z^2 - 1/25) + Real.sqrt (x^2 - 1/25))
  (hz : z = Real.sqrt (x^2 - 1/36) + Real.sqrt (y^2 - 1/36))
  (hsum : x + y + z = (m : ℝ) / Real.sqrt (n : ℝ))
  (hn_squarefree : ∀ (p : ℕ) (hp : Nat.Prime p), ¬(p^2 ∣ (n : ℕ))) :
  m + n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l841_84197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l841_84181

theorem plant_arrangement (n m : ℕ) (hn : n = 5) (hm : m = 4) : 
  (Nat.factorial (n + 1)) * (Nat.factorial m) = 17280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_l841_84181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_example_l841_84187

/-- Calculates the profit percentage given cost price, selling price, and discount percentage. -/
noncomputable def profit_percentage (cost_price selling_price discount_percent : ℝ) : ℝ :=
  let marked_price := selling_price / (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem: Given a cost price of 50, a selling price of 69.44 after a 10% discount,
    the profit percentage is approximately 38.88%. -/
theorem profit_percentage_example :
  abs (profit_percentage 50 69.44 10 - 38.88) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_example_l841_84187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_case_l841_84146

/-- The length of a wire stretched between two vertical poles -/
noncomputable def wire_length (horizontal_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) (base_height_difference : ℝ) : ℝ :=
  Real.sqrt (horizontal_distance^2 + (tall_pole_height - short_pole_height + base_height_difference)^2)

/-- Theorem stating the length of the wire in the given scenario -/
theorem wire_length_specific_case :
  wire_length 20 8 18 2 = Real.sqrt 544 := by
  unfold wire_length
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_case_l841_84146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastrami_costs_five_l841_84157

/-- The price of a Reuben sandwich -/
def reuben_price : ℝ := sorry

/-- The price of a pastrami sandwich -/
def pastrami_price : ℝ := sorry

/-- Pastrami costs $2 more than Reuben -/
axiom price_difference : pastrami_price = reuben_price + 2

/-- Total earnings from selling 10 Reubens and 5 pastramis -/
axiom total_earnings : 10 * reuben_price + 5 * pastrami_price = 55

theorem pastrami_costs_five : pastrami_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastrami_costs_five_l841_84157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l841_84140

theorem functional_equation_solution (f g h : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y^3) + g (x^3 + y) = h (x*y)) :
  (∃ c : ℝ, ∀ x : ℝ, f x = c) ∧
  (∀ x : ℝ, g x = h 0 - f x) ∧
  (∃ e : ℝ, ∀ x : ℝ, h x = e) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l841_84140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_links_opened_l841_84198

theorem max_links_opened (N : ℕ) (h : N > 3) :
  ∃ (max_opened : ℕ), max_opened = (3 * N) / 4 ∧
  ∀ (rearrangement : List ℕ),
    rearrangement.length = N →
    rearrangement.toFinset = Finset.range N →
    (∃ (opened : ℕ), opened ≤ max_opened ∧
      (∃ (proof : Prop), proof ↔ rearrangement.isPerm (List.range N))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_links_opened_l841_84198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l841_84104

theorem triangle_formation (a b : ℝ) (sticks : List ℝ) : 
  a = 4 ∧ 
  b = 9 ∧ 
  sticks = [4, 5, 9, 13] →
  ∃ x ∈ sticks, (x > a - b ∧ x < a + b) ∧ x = 9 := by
  intro h
  use 9
  constructor
  · simp [h.2.2]
  constructor
  · simp [h.1, h.2.1]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l841_84104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_cost_l841_84193

/-- The cost of an animal in Rupees -/
structure Cost where
  value : ℕ

instance : HMul ℕ Cost Cost where
  hMul n c := ⟨n * c.value⟩

instance : OfNat Cost n where
  ofNat := ⟨n⟩

/-- The cost relationship between camels and horses -/
def camel_horse_ratio (camel_cost horse_cost : Cost) : Prop :=
  (10 : ℕ) * camel_cost = (24 : ℕ) * horse_cost

/-- The cost relationship between horses and oxen -/
def horse_ox_ratio (horse_cost ox_cost : Cost) : Prop :=
  (16 : ℕ) * horse_cost = (4 : ℕ) * ox_cost

/-- The cost relationship between oxen and elephants -/
def ox_elephant_ratio (ox_cost elephant_cost : Cost) : Prop :=
  (6 : ℕ) * ox_cost = (4 : ℕ) * elephant_cost

/-- The cost of 10 elephants -/
def elephant_cost : Cost := 150000

theorem camel_cost (camel_cost horse_cost ox_cost elephant_cost : Cost) :
  camel_horse_ratio camel_cost horse_cost →
  horse_ox_ratio horse_cost ox_cost →
  ox_elephant_ratio ox_cost elephant_cost →
  (10 : ℕ) * elephant_cost = elephant_cost →
  camel_cost = (6000 : Cost) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_cost_l841_84193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_drop_l841_84190

theorem temperature_drop (t : ℝ) : t - 2 = t - 2 := by
  rfl

#check temperature_drop

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_drop_l841_84190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l841_84199

/-- The probability of getting heads for an unfair coin -/
noncomputable def p : ℝ := 3/4

/-- The number of coin tosses -/
def n : ℕ := 30

/-- The probability of getting an even number of heads after k tosses -/
noncomputable def P (k : ℕ) : ℝ :=
  1/2 * (1 + (-1/2)^k)

/-- The main theorem: The probability of getting an even number of heads
    after n tosses for the given unfair coin -/
theorem even_heads_probability :
  P n = 1/2 * (1 + 1/2^n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l841_84199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sammy_eats_15_slices_l841_84195

/-- The number of pickle slices Sammy can eat -/
def sammy_slices : ℕ := 15

/-- The number of pickle slices Tammy can eat -/
def tammy_slices : ℕ := 2 * sammy_slices

/-- The number of pickle slices Ron can eat -/
def ron_slices : ℕ := 24

/-- Theorem stating that given the conditions, Sammy can eat 15 pickle slices -/
theorem sammy_eats_15_slices : 
  (tammy_slices = 2 * sammy_slices) → 
  (ron_slices = (8 * tammy_slices) / 10) → 
  (ron_slices = 24) → 
  (sammy_slices = 15) := by
  intro h1 h2 h3
  -- The proof goes here
  sorry

#check sammy_eats_15_slices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sammy_eats_15_slices_l841_84195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_odd_unit_digit_l841_84165

-- Define the set of digits
def Digit := Fin 10

-- Define the property of being an odd digit
def isOddDigit (d : Digit) : Prop := d.val % 2 ≠ 0

-- Define the property of a digit being in the units place of an odd number
def isUnitDigitOfOddNumber (d : Digit) : Prop := isOddDigit d

-- Statement: 0 is the smallest digit not found in the units place of an odd number
theorem smallest_non_odd_unit_digit :
  ∀ d : Digit, d.val < 0 → isUnitDigitOfOddNumber d := by
  sorry

#check smallest_non_odd_unit_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_odd_unit_digit_l841_84165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_from_area_and_offsets_l841_84174

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (1 / 2) * q.diagonal * (q.offset1 + q.offset2)

theorem diagonal_length_from_area_and_offsets
  (q : Quadrilateral)
  (h1 : q.offset1 = 9)
  (h2 : q.offset2 = 6)
  (h3 : area q = 225) :
  q.diagonal = 30 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_from_area_and_offsets_l841_84174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_four_enchiladas_five_tacos_l841_84177

/-- The price of an enchilada in dollars -/
def e : ℝ := sorry

/-- The price of a taco in dollars -/
def t : ℝ := sorry

/-- The cost of three enchiladas and four tacos is $3.50 -/
axiom price_combo1 : 3 * e + 4 * t = 3.50

/-- The cost of four enchiladas and three tacos is $3.90 -/
axiom price_combo2 : 4 * e + 3 * t = 3.90

/-- The cost of four enchiladas and five tacos is $4.56 -/
theorem cost_four_enchiladas_five_tacos : 4 * e + 5 * t = 4.56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_four_enchiladas_five_tacos_l841_84177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_100_factorial_l841_84109

open Nat

theorem last_two_nonzero_digits_100_factorial : 
  ∃ k : ℕ, factorial 100 = k * 10^24 + 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_100_factorial_l841_84109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l841_84154

/-- Calculates the length of a train given its speed and time to pass a point. -/
noncomputable def train_length (speed_kmph : ℝ) (time_seconds : ℝ) : ℝ :=
  (speed_kmph * 1000 / 3600) * time_seconds

/-- Theorem stating the length of the train given the specified conditions. -/
theorem train_length_calculation :
  let speed_kmph : ℝ := 36
  let time_seconds : ℝ := 8.999280057595392
  let calculated_length : ℝ := train_length speed_kmph time_seconds
  ∃ (ε : ℝ), ε > 0 ∧ |calculated_length - 89.99| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l841_84154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l841_84116

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -Real.log x
  else if x > 1 then Real.log x
  else 0

noncomputable def tangent_slope (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -1/x
  else if x > 1 then 1/x
  else 0

theorem triangle_area_range (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < 1) (h₃ : 1 < x₂) 
    (h₄ : tangent_slope x₁ * tangent_slope x₂ = -1) : 
  ∃ S : ℝ, 0 < S ∧ S < 1 ∧ 
    S = (2 * x₁ * x₂) / (x₁ + x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l841_84116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_is_30_l841_84139

/-- A trip with two parts -/
structure Trip where
  total_distance : ℝ
  first_distance : ℝ
  first_speed : ℝ
  average_speed : ℝ

/-- Calculate the speed of the second part of the trip -/
noncomputable def second_part_speed (t : Trip) : ℝ :=
  let total_time := t.total_distance / t.average_speed
  let first_time := t.first_distance / t.first_speed
  let second_distance := t.total_distance - t.first_distance
  let second_time := total_time - first_time
  second_distance / second_time

/-- Theorem stating the speed of the second part of the trip -/
theorem second_part_speed_is_30 (t : Trip) 
  (h1 : t.total_distance = 50)
  (h2 : t.first_distance = 25)
  (h3 : t.first_speed = 60)
  (h4 : t.average_speed = 40) :
  second_part_speed t = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_is_30_l841_84139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_volume_l841_84131

/-- The volume of a regular hexagonal pyramid with base side length a and height a -/
noncomputable def hexagonal_pyramid_volume (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt 3) / 2

/-- Theorem: The volume of a regular hexagonal pyramid with base side length a and height a
    is equal to (a^3 * √3) / 2 -/
theorem regular_hexagonal_pyramid_volume (a : ℝ) (h : a > 0) :
  hexagonal_pyramid_volume a = (a^3 * Real.sqrt 3) / 2 := by
  -- Unfold the definition of hexagonal_pyramid_volume
  unfold hexagonal_pyramid_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_volume_l841_84131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_removal_prove_impossible_removal_l841_84151

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem impossible_removal (seq : List ℕ) : Prop :=
  seq = arithmetic_sequence 4 10 11 →
  ∀ (r₁ r₂ r₃ r₄ : List ℕ),
    r₁.length = 1 ∧
    r₂.length = 2 ∧
    r₃.length = 3 ∧
    r₄.length = 4 ∧
    (r₁ ++ r₂ ++ r₃ ++ r₄).toFinset = seq.toFinset →
    ¬(is_divisible_by (sum_list (seq.filter (λ x => ¬(r₁.contains x)))) 11 ∧
      is_divisible_by (sum_list (seq.filter (λ x => ¬((r₁ ++ r₂).contains x)))) 11 ∧
      is_divisible_by (sum_list (seq.filter (λ x => ¬((r₁ ++ r₂ ++ r₃).contains x)))) 11 ∧
      is_divisible_by (sum_list (seq.filter (λ x => ¬((r₁ ++ r₂ ++ r₃ ++ r₄).contains x)))) 11)

theorem prove_impossible_removal : impossible_removal (arithmetic_sequence 4 10 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_removal_prove_impossible_removal_l841_84151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_theorem_l841_84167

-- Define the domain
def Domain : Set ℝ := { x : ℝ | x ≥ 1 }

-- Define the functions
noncomputable def f₁ : ℝ → ℝ := fun x ↦ 1 / x
noncomputable def f₃ : ℝ → ℝ := fun x ↦ Real.sqrt (x^2 - 1)
noncomputable def f₄ : ℝ → ℝ := fun x ↦ Real.log x / x

-- Define the channel width property
def has_channel_width_one (f : ℝ → ℝ) : Prop :=
  ∃ (k m : ℝ), ∀ x ∈ Domain, k * x + m ≤ f x ∧ f x ≤ k * x + (m + 1)

-- State the theorem
theorem channel_width_theorem :
  has_channel_width_one f₁ ∧
  has_channel_width_one f₃ ∧
  has_channel_width_one f₄ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_width_theorem_l841_84167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l841_84183

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 12 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + 2*y - 5 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- The main theorem
theorem symmetric_circle_equation :
  ∀ (C : Set (ℝ × ℝ)),
  (∀ (p : ℝ × ℝ), p ∈ C ↔ symmetric_circle p.1 p.2) →
  (∃ (f : ℝ × ℝ → ℝ × ℝ),
    (∀ (p : ℝ × ℝ), symmetry_line ((p.1 + (f p).1) / 2) ((p.2 + (f p).2) / 2)) ∧
    (∀ (p : ℝ × ℝ), given_circle p.1 p.2 ↔ given_circle (f p).1 (f p).2) ∧
    (∀ (p : ℝ × ℝ), p ∈ C ↔ f p ∈ C)) →
  ∀ (p : ℝ × ℝ), p ∈ C → symmetric_circle p.1 p.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l841_84183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T1_T2_T3_l841_84161

-- Define the basic types
variable (S : Type) (pib maa : S → Prop)

-- Define membership for S
instance : Membership S S where
  mem := λ x y => x = y

-- Axioms
axiom P1 : ∀ p, pib p → ∃ m, maa m ∧ (∀ x, x ∈ p ↔ maa x)
axiom P2 : ∀ p q, pib p → pib q → p ≠ q → ∃! m, maa m ∧ m ∈ p ∧ m ∈ q
axiom P3 : ∀ m, maa m → ∃! p q, pib p ∧ pib q ∧ p ≠ q ∧ m ∈ p ∧ m ∈ q
axiom P4 : ∃! (a b c d : S), pib a ∧ pib b ∧ pib c ∧ pib d ∧ 
           a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
           (∀ x, pib x → (x = a ∨ x = b ∨ x = c ∨ x = d))

-- Theorems to prove
theorem T1 : 
  ∃! (a b c d e f : S), maa a ∧ maa b ∧ maa c ∧ maa d ∧ maa e ∧ maa f ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  (∀ x, maa x → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f)) :=
sorry

theorem T2 : ∀ p, pib p → ∃! (a b c : S), maa a ∧ maa b ∧ maa c ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a ∈ p ∧ b ∈ p ∧ c ∈ p ∧
  (∀ x, maa x ∧ x ∈ p → (x = a ∨ x = b ∨ x = c)) :=
sorry

theorem T3 : ∀ m, maa m → ∃! n, maa n ∧ m ≠ n ∧ 
  (∀ p, pib p → (m ∈ p ↔ n ∉ p)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T1_T2_T3_l841_84161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_18_representation_l841_84172

/-- The factorial of 18 -/
def factorial_18 : ℕ := Nat.factorial 18

/-- The base-ten representation of 18! -/
def base_ten_rep (T M : ℕ) : ℕ := 64023*10^9 + T*10^8 + 752*10^6 + M*10^5 + 108800

/-- The theorem stating that T + M = 11 for the given representation of 18! -/
theorem factorial_18_representation (T M : ℕ) :
  base_ten_rep T M = factorial_18 → T + M = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_18_representation_l841_84172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l841_84185

noncomputable section

/-- The parabola y = x^2 - 6x + 15 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 15

/-- The line y = 2x - 7 -/
def line (x : ℝ) : ℝ := 2*x - 7

/-- The distance between a point (x, parabola x) and the line y = 2x - 7 -/
noncomputable def distance (x : ℝ) : ℝ := 
  abs (2*x - (parabola x) - 7) / Real.sqrt 5

/-- The shortest distance between a point on the parabola and a point on the line is 6√5/5 -/
theorem shortest_distance : 
  ∃ (x : ℝ), distance x = (6 * Real.sqrt 5) / 5 ∧ 
  ∀ (y : ℝ), distance y ≥ (6 * Real.sqrt 5) / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l841_84185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_empty_square_l841_84135

/-- Represents a peaceful arrangement of rooks on an n × n chessboard -/
def PeacefulArrangement (n : ℕ) := 
  { arrangement : Fin n → Fin n // ∀ i j : Fin n, i ≠ j → arrangement i ≠ arrangement j }

/-- 
Theorem: For any n × n chessboard (n ≥ 2) with a peaceful arrangement of n rooks, 
the largest k such that there always exists a k × k square with all k² unit squares 
unoccupied by rooks is ⌊√(n-1)⌋.
-/
theorem largest_empty_square (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, k = Nat.floor (Real.sqrt (n - 1 : ℝ)) ∧
    (∀ arrangement : PeacefulArrangement n, 
      ∃ i j : Fin n, ∀ x y : Fin k, 
        (arrangement.val (i + ⟨x.val, by sorry⟩)) ≠ (j + ⟨y.val, by sorry⟩)) ∧
    (∀ k' : ℕ, k' > k → 
      ∃ arrangement : PeacefulArrangement n,
        ∀ i j : Fin n, ∃ x y : Fin k', 
          (arrangement.val (i + ⟨x.val, by sorry⟩)) = (j + ⟨y.val, by sorry⟩)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_empty_square_l841_84135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_26_triple_prime_products_l841_84178

/-- A function that returns the number of positive integers that are products of three distinct primes summing to a given value. -/
def count_triple_prime_products (sum : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there are exactly 26 positive integers that are products of three distinct primes summing to 261, given that 4103 is the product of three distinct primes whose sum is 261. -/
theorem exactly_26_triple_prime_products :
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p * q * r = 4103 ∧ p + q + r = 261) →
  count_triple_prime_products 261 = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_26_triple_prime_products_l841_84178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_problem_l841_84118

theorem power_of_three_problem (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_problem_l841_84118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_y_equals_x_l841_84141

theorem angle_of_inclination_y_equals_x :
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ 
  (∀ x y : ℝ, y = x → Real.tan α = (y - 0) / (x - 0)) → 
  α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_y_equals_x_l841_84141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedral_angle_cosine_l841_84182

-- Define a type for 3D vectors
def Vector3D := Fin 3 → ℝ

-- Define a function to calculate the dot product of two 3D vectors
def dot_product (v1 v2 : Vector3D) : ℝ :=
  (Finset.univ.sum fun i => v1 i * v2 i)

-- Define a function to calculate the magnitude of a 3D vector
noncomputable def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (Finset.univ.sum fun i => v i ^ 2)

-- Theorem statement
theorem tetrahedral_angle_cosine :
  ∀ (r1 r2 r3 r4 : Vector3D),
    (∀ (i j : Fin 4), i ≠ j → 
      let rays := [r1, r2, r3, r4]
      dot_product (rays.get i) (rays.get j) / (magnitude (rays.get i) * magnitude (rays.get j)) = 
      dot_product r1 r2 / (magnitude r1 * magnitude r2)) →
    dot_product r1 r2 / (magnitude r1 * magnitude r2) = -1/3 :=
by
  sorry

#check tetrahedral_angle_cosine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedral_angle_cosine_l841_84182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_representation_l841_84124

theorem factorial_15_representation (X Y Z : ℕ) : 
  (Nat.factorial 15) % 1000000 = X * 100000 + Y * 1000 + Z * 30100 →
  Z = 0 →
  (X + Y) % 3 = 0 →
  (7 - X + 2 * Y) % 11 = 0 →
  X + Y + Z = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_representation_l841_84124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_roots_theorem_l841_84196

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^a
noncomputable def g (x : ℝ) : ℝ := |Real.log x|

-- State the theorem
theorem power_function_and_roots_theorem (a : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, f a x = k * x^a) →
  (∃ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧
    g (x₁ - 1) + f a 1 = 0 ∧
    g (x₂ - 1) + f a 1 = 0 ∧
    x₁ ≠ x₂) →
  (a = 2 ∧
   (∀ x < 0, ∀ y > x, f a y < f a x) ∧
   (2 - Real.log 2 < a + 1 / x₁ + 1 / x₂ ∧ a + 1 / x₁ + 1 / x₂ < 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_and_roots_theorem_l841_84196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_incircle_locus_l841_84152

-- Define the circle and line
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the tangent line condition
def IsTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the midpoint condition
def IsMidpoint (m : ℝ × ℝ) (q r : ℝ × ℝ) : Prop :=
  m.1 = (q.1 + r.1) / 2 ∧ m.2 = (q.2 + r.2) / 2

-- Define the incircle condition
def IsIncircle (c : Set (ℝ × ℝ)) (p q r : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem tangent_incircle_locus 
  (c : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) (m : ℝ × ℝ) 
  (h_tangent : IsTangent l c)
  (h_m_on_l : m ∈ l) :
  ∃ (a b : ℝ), 
    {p : ℝ × ℝ | ∃ (q r : ℝ × ℝ), 
      q ∈ l ∧ r ∈ l ∧ 
      IsMidpoint m q r ∧ 
      IsIncircle c p q r} = 
    {p : ℝ × ℝ | p.2 = (b/a) * p.1 + b ∧ p.2 > 2*b} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_incircle_locus_l841_84152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_is_false_l841_84192

theorem proposition_p_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 - 3*x + 12 < 0) :=
by
  push_neg
  use 2
  constructor
  · norm_num
  · norm_num
  done

#check proposition_p_is_false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_is_false_l841_84192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_condition_l841_84121

-- Define the sequence a_n
def a (n : ℕ+) (lambda : ℝ) : ℝ := n^2 + 2*lambda*n + 1

-- State the theorem
theorem sequence_increasing_condition (lambda : ℝ) :
  (∀ n : ℕ+, a n lambda < a (n + 1) lambda) ↔ lambda > -3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_condition_l841_84121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l841_84163

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = x + 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle_eq p.1 p.2 ∧ line_eq p.1 p.2}

-- State the theorem
theorem distance_between_intersection_points :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 5 * Real.sqrt 2 :=
sorry

#check distance_between_intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l841_84163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sunzi_books_l841_84171

/-- Represents the number of books of each type -/
structure BookQuantities where
  zhuangzi : ℕ
  kongzi : ℕ
  mengzi : ℕ
  laozi : ℕ
  sunzi : ℕ

/-- Represents the price of each book type -/
def book_prices : BookQuantities := {
  zhuangzi := 10,
  kongzi := 20,
  mengzi := 15,
  laozi := 30,
  sunzi := 12
}

/-- The total cost of all books -/
def total_cost : ℕ := 4500

/-- The total number of books -/
def total_books : ℕ := 300

/-- Calculates the total cost of books given their quantities -/
def calculate_total_cost (quantities : BookQuantities) : ℕ :=
  quantities.zhuangzi * book_prices.zhuangzi +
  quantities.kongzi * book_prices.kongzi +
  quantities.mengzi * book_prices.mengzi +
  quantities.laozi * book_prices.laozi +
  quantities.sunzi * book_prices.sunzi

/-- Calculates the total number of books given their quantities -/
def calculate_total_books (quantities : BookQuantities) : ℕ :=
  quantities.zhuangzi + quantities.kongzi + quantities.mengzi + quantities.laozi + quantities.sunzi

/-- The main theorem to prove -/
theorem number_of_sunzi_books (quantities : BookQuantities) :
  quantities.zhuangzi = quantities.kongzi →
  quantities.sunzi = 4 * quantities.laozi + 15 →
  calculate_total_cost quantities = total_cost →
  calculate_total_books quantities = total_books →
  quantities.sunzi = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_sunzi_books_l841_84171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luna_total_budget_l841_84115

/-- Luna's monthly budget calculation -/
def luna_budget (house_rental : ℝ) (food : ℝ) (phone_bill : ℝ) (transportation : ℝ) (entertainment : ℝ) (total : ℝ) : Prop :=
  food = 0.6 * house_rental ∧
  phone_bill = 0.1 * food ∧
  transportation = 0.25 * house_rental ∧
  entertainment = 0.15 * (food + transportation) ∧
  house_rental + food + transportation = 300 ∧
  total = house_rental + food + transportation + phone_bill + entertainment ∧
  abs (total - 330.41) < 0.01

/-- The main theorem stating Luna's total monthly budget -/
theorem luna_total_budget : ∃ (house_rental food phone_bill transportation entertainment total : ℝ),
  luna_budget house_rental food phone_bill transportation entertainment total :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luna_total_budget_l841_84115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_444_l841_84132

/-- Calculates the distance between two stations given the conditions of two trains meeting --/
noncomputable def distance_between_stations (speed1 speed2 : ℝ) (extra_distance : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let time := extra_distance / (speed2 - speed1)
  (speed1 * time) + (speed2 * time)

/-- Theorem stating the distance between the stations is 444 km --/
theorem distance_is_444 :
  distance_between_stations 16 21 60 = 444 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_between_stations 16 21 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_444_l841_84132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_pairs_l841_84117

/-- The number of pairs of positive integers (a, b) satisfying the given conditions -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    a > 0 ∧ b > 0 ∧
    (↑a + 3 / ↑b) / (1 / ↑a + 3 * ↑b) = 7 ∧
    a + b ≤ 150
  ) (Finset.range 151 ×ˢ Finset.range 151)).card

/-- Theorem stating that there are exactly 6 pairs satisfying the conditions -/
theorem six_pairs : count_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_pairs_l841_84117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l841_84136

/-- A complex number tracing a circle centered at the origin with radius 3 -/
noncomputable def CircleTracer : Type := { w : ℂ // Complex.abs w = 3 }

/-- The locus of w + 2/w for w tracing a circle -/
noncomputable def Locus (w : CircleTracer) : ℂ := w.val + 2 / w.val

theorem locus_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (w : CircleTracer), 
    let z := Locus w
    (z.re / a)^2 + (z.im / b)^2 = 1 :=
by
  sorry

#check locus_is_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l841_84136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l841_84110

/-- Definition of a valid triangle based on the triangle inequality -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given a triangle with two sides of lengths 3 and 5, prove that the third side y satisfies 2 < y < 8 -/
theorem triangle_side_range (y : ℝ) : 
  (∃ (a b : ℝ), a = 3 ∧ b = 5 ∧ is_triangle a b y) → 2 < y ∧ y < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l841_84110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l841_84150

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

def perpendicular (v1 v2 : Point3D) : Prop := sorry

def onPlane (p : Point3D) (a b c : Point3D) : Prop := sorry

def onSphere (p : Point3D) (s : Sphere) : Prop := sorry

noncomputable def distance (p1 p2 : Point3D) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

noncomputable def angle (p1 p2 p3 : Point3D) : ℝ := sorry

instance : HSub Point3D Point3D Point3D where
  hSub := λ p1 p2 => ⟨p1.x - p2.x, p1.y - p2.y, p1.z - p2.z⟩

theorem sphere_surface_area 
  (pyramid : TriangularPyramid) 
  (s : Sphere) :
  angle pyramid.B pyramid.O pyramid.C = Real.pi / 2 →
  perpendicular (pyramid.A - pyramid.O) (pyramid.B - pyramid.O) →
  perpendicular (pyramid.A - pyramid.O) (pyramid.C - pyramid.O) →
  distance pyramid.A pyramid.B = Real.sqrt 10 →
  distance pyramid.B pyramid.C = Real.sqrt 13 →
  distance pyramid.A pyramid.C = Real.sqrt 5 →
  onSphere pyramid.O s →
  onSphere pyramid.A s →
  onSphere pyramid.B s →
  onSphere pyramid.C s →
  4 * Real.pi * s.radius^2 = 14 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l841_84150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l841_84126

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 2)

theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc (-5 : ℝ) (-4 : ℝ) = Set.Icc (-3/2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l841_84126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l841_84108

/-- Proposition p: There exists a > 0 such that f(x) = ax^2 - 4x is monotonically decreasing on (-∞, 2] -/
def proposition_p (a : ℝ) : Prop :=
  a > 0 ∧ MonotoneOn (fun x => a * x^2 - 4 * x) (Set.Iic 2)

/-- Proposition q: There exists a ∈ R such that for all x ∈ R, 16x^2 - 16(a-1)x + 1 ≠ 0 -/
def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, 16 * x^2 - 16 * (a - 1) * x + 1 ≠ 0

/-- The range of real numbers for a given p ∧ q is true -/
theorem range_of_a (a : ℝ) (h : proposition_p a ∧ proposition_q a) :
  a ∈ Set.Ioo (1/2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l841_84108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_sum_l841_84123

-- Define the curve C
noncomputable def curve_C (φ : Real) : Real × Real :=
  (Real.sqrt 5 * Real.cos φ, Real.sqrt 15 * Real.sin φ)

-- Define the line l
noncomputable def line_l (t : Real) : Real × Real :=
  (-1/2 * t, Real.sqrt 3 + (Real.sqrt 3 / 2) * t)

-- Define point P
noncomputable def point_P : Real × Real :=
  (0, Real.sqrt 3)

-- Define the general equation of curve C
def curve_C_equation (x y : Real) : Prop :=
  x^2 / 5 + y^2 / 15 = 1

-- Theorem statement
theorem intersection_points_distance_sum :
  ∃ A B : Real × Real,
    (∃ φ : Real, curve_C φ = A) ∧
    (∃ t₁ : Real, line_l t₁ = A) ∧
    (∃ φ : Real, curve_C φ = B) ∧
    (∃ t₂ : Real, line_l t₂ = B) ∧
    Real.sqrt ((point_P.1 - A.1)^2 + (point_P.2 - A.2)^2) +
    Real.sqrt ((point_P.1 - B.1)^2 + (point_P.2 - B.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_sum_l841_84123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_5pi_l841_84194

open Real

theorem sum_of_solutions_is_5pi : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * π ∧ 1 / sin x + 1 / cos x = 4) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * π ∧ 1 / sin x + 1 / cos x = 4 → x ∈ S) ∧
    (S.sum id) = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_5pi_l841_84194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l841_84101

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l841_84101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_equals_circle_radius_l841_84148

theorem rectangle_length_equals_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 1296) 
  (h2 : rectangle_area = 360) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) = Real.sqrt square_area := by
  -- Define local variables
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := rectangle_area / rectangle_breadth
  
  -- Proof steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_equals_circle_radius_l841_84148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_probability_l841_84113

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_probability (a₁ d : ℝ) :
  sum_arithmetic_sequence a₁ d 4 = Real.pi ∧
  arithmetic_sequence a₁ d 4 = 2 * arithmetic_sequence a₁ d 2 →
  (Finset.filter (fun i => Real.cos (arithmetic_sequence a₁ d i) < 0) (Finset.range 30)).card / 30 = 7 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_probability_l841_84113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_divisor_last_digit_sum_33_l841_84164

def is_odd (n : ℕ) : Prop := n % 2 = 1

def in_range (n : ℕ) : Prop := 500 < n ∧ n < 1000

def last_digit (n : ℕ) : ℕ := n % 10

def divisors (n : ℕ) : Finset ℕ := Finset.filter (·∣n) (Finset.range (n+1))

def sum_last_digits_of_divisors (n : ℕ) : ℕ :=
  (divisors n).sum last_digit

theorem unique_number_with_divisor_last_digit_sum_33 :
  ∃! n : ℕ, is_odd n ∧ in_range n ∧ sum_last_digits_of_divisors n = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_divisor_last_digit_sum_33_l841_84164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_M_l841_84103

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {4, 5}
def M : Finset ℕ := Finset.biUnion A (fun a => Finset.image (· + a) B)

theorem number_of_proper_subsets_of_M : Finset.card (Finset.powerset M \ {M}) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_M_l841_84103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l841_84153

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.tan x - Real.sqrt 3)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} =
  {x : ℝ | f x ≠ 0 ∨ (∀ ε > 0, ∃ y, |y - x| < ε ∧ f y ≠ 0)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l841_84153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_sum_l841_84114

theorem exponential_sequence_sum (b : ℕ → ℝ) (r : ℝ) : 
  b 1 = 2 → 
  r = 1/3 → 
  (∀ n : ℕ, b (n + 1) = b n * r) → 
  (∑' n : ℕ, b n) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_sum_l841_84114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l841_84138

/-- Represents the number of fifth grade students -/
def f : ℕ := 1  -- We use a concrete value to avoid issues with functions

/-- Average minutes run per day by third grade students -/
def third_grade_avg : ℚ := 10

/-- Average minutes run per day by fourth grade students -/
def fourth_grade_avg : ℚ := 18

/-- Average minutes run per day by fifth grade students -/
def fifth_grade_avg : ℚ := 8

/-- Number of fourth grade students -/
def fourth_grade_count : ℕ := 2 * f

/-- Number of third grade students -/
def third_grade_count : ℕ := 3 * fourth_grade_count

/-- Total number of students -/
def total_students : ℕ := third_grade_count + fourth_grade_count + f

/-- Total minutes run by all students -/
noncomputable def total_minutes : ℚ := 
  third_grade_avg * (third_grade_count : ℚ) + 
  fourth_grade_avg * (fourth_grade_count : ℚ) + 
  fifth_grade_avg * (f : ℚ)

/-- Theorem stating the average minutes run per day by all students -/
theorem average_minutes_run : total_minutes / (total_students : ℚ) = 104 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_l841_84138
