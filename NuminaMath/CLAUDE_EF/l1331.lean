import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_payment_frequency_l1331_133116

-- Define the problem parameters
def initial_investment : ℚ := 10000
def annual_interest_rate : ℚ := 95 / 1000
def total_period_months : ℕ := 18
def interest_per_payment : ℚ := 475 / 2

-- Define the function to calculate total interest
def total_interest (principal : ℚ) (rate : ℚ) (time_years : ℚ) : ℚ :=
  principal * rate * time_years

-- Define the function to calculate number of payments
def number_of_payments (total_interest : ℚ) (interest_per_payment : ℚ) : ℚ :=
  total_interest / interest_per_payment

-- Define the function to calculate payment frequency in months
def payment_frequency_months (total_period_months : ℕ) (num_payments : ℚ) : ℚ :=
  (total_period_months : ℚ) / num_payments

-- Theorem statement
theorem interest_payment_frequency :
  let total_int := total_interest initial_investment annual_interest_rate ((total_period_months : ℚ) / 12)
  let num_payments := number_of_payments total_int interest_per_payment
  let frequency := payment_frequency_months total_period_months num_payments
  frequency = 3 := by
    -- Unfold definitions
    unfold payment_frequency_months number_of_payments total_interest
    -- Simplify expressions
    simp [initial_investment, annual_interest_rate, total_period_months, interest_per_payment]
    -- Prove equality
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_payment_frequency_l1331_133116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_values_l1331_133119

theorem sum_of_a_values : 
  ∃ (S : Finset ℤ), 
    (∀ a ∈ S, 
      a ≥ -2 ∧ 
      ∃ x : ℤ, x > 0 ∧ (10 - a : ℚ) / 3 = x ∧
      a ≠ -2) ∧
    (S.sum id) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_values_l1331_133119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_line_arrangements_l1331_133187

theorem library_line_arrangements (n k : ℕ) : 
  n = 8 → k = 2 → (Nat.factorial (n - k + 1)) * (Nat.factorial k) = 10080 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_line_arrangements_l1331_133187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_expectation_l1331_133197

/-- The probability of an animal dying in the first month -/
noncomputable def prob_die_first_month : ℝ := 1/8

/-- The probability of an animal dying in the second month (without disease) -/
noncomputable def prob_die_second_month : ℝ := 1/6

/-- The probability of an animal dying in the third month -/
noncomputable def prob_die_third_month : ℝ := 1/4

/-- The proportion of newborns affected by the disease in the second month -/
noncomputable def prop_affected_disease : ℝ := 0.05

/-- The reduction in survival rate for diseased animals in the second month -/
noncomputable def survival_rate_reduction : ℝ := 0.5

/-- The total number of newborn animals -/
def total_newborns : ℕ := 800

/-- The expected number of survivors after three months -/
def expected_survivors : ℕ := 435

theorem survival_expectation :
  let survival_first_month := 1 - prob_die_first_month
  let survival_second_month_healthy := 1 - prob_die_second_month
  let survival_second_month_diseased := 1 - (prob_die_second_month + prob_die_second_month * survival_rate_reduction)
  let overall_survival_second_month := (1 - prop_affected_disease) * survival_second_month_healthy + prop_affected_disease * survival_second_month_diseased
  let survival_third_month := 1 - prob_die_third_month
  let overall_survival_rate := survival_first_month * overall_survival_second_month * survival_third_month
  Int.floor (total_newborns * overall_survival_rate) = expected_survivors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_expectation_l1331_133197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock1_is_precise_l1331_133146

/- Define the type for clocks -/
structure Clock where
  id : ℕ

/- Define the function that represents the time shown by each clock -/
noncomputable def clock_time (c : Clock) : ℚ → ℚ :=
  sorry

/- Define the conditions -/
axiom five_clocks : ∃ c₁ c₂ c₃ c₄ c₅ : Clock, c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₁ ≠ c₅ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₂ ≠ c₅ ∧ c₃ ≠ c₄ ∧ c₃ ≠ c₅ ∧ c₄ ≠ c₅

axiom clock3_lags_clock2 (c₂ c₃ : Clock) (t : ℚ) : clock_time c₃ t = clock_time c₂ t - 3

axiom clock1_lags_clock2 (c₁ c₂ : Clock) (t : ℚ) : clock_time c₁ t = clock_time c₂ t - 1

axiom clock4_lags_clock3 (c₃ c₄ : Clock) (t : ℚ) : clock_time c₄ t < clock_time c₃ t

axiom clock5_lags_clock1 (c₁ c₅ : Clock) (t : ℚ) : clock_time c₅ t < clock_time c₁ t

axiom average_is_correct_time (c₁ c₂ c₃ c₄ c₅ : Clock) (t : ℚ) :
  (clock_time c₁ t + clock_time c₂ t + clock_time c₃ t + clock_time c₄ t + clock_time c₅ t) / 5 = t

axiom one_clock_precise : ∃ c : Clock, ∀ t : ℚ, clock_time c t = t

/- The theorem to prove -/
theorem clock1_is_precise :
  ∃ c₁ c₂ c₃ c₄ c₅ : Clock, (∀ t : ℚ, clock_time c₁ t = t) ∧
    c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₁ ≠ c₅ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₂ ≠ c₅ ∧ c₃ ≠ c₄ ∧ c₃ ≠ c₅ ∧ c₄ ≠ c₅ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock1_is_precise_l1331_133146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1331_133117

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (2 - x)) / (x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -3 ∨ (-3 < x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1331_133117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_exists_l1331_133137

/-- Represents a square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- The large square containing everything -/
def large_square : Square :=
  { center := (7.5, 7.5), side_length := 15 }

/-- The set of small squares within the large square -/
def small_squares : Finset Square :=
  sorry

/-- Checks if two squares are non-overlapping -/
def non_overlapping (s1 s2 : Square) : Prop :=
  sorry

/-- Checks if a point is inside a square -/
def point_in_square (p : ℝ × ℝ) (s : Square) : Prop :=
  sorry

/-- Checks if a point is at least 1 unit away from a square -/
def point_away_from_square (p : ℝ × ℝ) (s : Square) : Prop :=
  sorry

/-- Main theorem: There exists a point for the circle's center -/
theorem circle_placement_exists : ∃ (p : ℝ × ℝ),
  (∀ s, s ∈ small_squares → point_away_from_square p s) ∧
  (point_in_square p { center := (7.5, 7.5), side_length := 13 }) :=
by sorry

/-- All small squares are pairwise non-overlapping -/
axiom small_squares_non_overlapping :
  ∀ s1 s2, s1 ∈ small_squares → s2 ∈ small_squares → s1 ≠ s2 → non_overlapping s1 s2

/-- The number of small squares is exactly 20 -/
axiom small_squares_count : small_squares.card = 20

/-- All small squares have side length 1 -/
axiom small_squares_size :
  ∀ s, s ∈ small_squares → s.side_length = 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_exists_l1331_133137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_200_l1331_133140

noncomputable def series_term (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

noncomputable def series_sum : ℚ := ∑' n, series_term n

theorem series_sum_equals_one_over_200 : series_sum = 1 / 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_200_l1331_133140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1331_133189

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.cos (2*x) + (Real.sqrt 3 / 2) * Real.sin (2*x)

-- State the theorem
theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Maximum value is √3
  (∀ (x : ℝ), f x ≤ Real.sqrt 3) ∧
  (∃ (x : ℝ), f x = Real.sqrt 3) ∧
  -- Minimum value is -√3
  (∀ (x : ℝ), -Real.sqrt 3 ≤ f x) ∧
  (∃ (x : ℝ), f x = -Real.sqrt 3) ∧
  -- Increasing interval
  (∀ (k : ℤ) (x y : ℝ),
    -5*Real.pi/12 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi/12 + k*Real.pi →
    f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1331_133189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a6_b3_ratio_l1331_133139

/-- A_n is the sum of the first n terms of an arithmetic sequence {a_n} -/
def A : ℕ → ℝ := sorry

/-- B_n is the sum of the first n terms of an arithmetic sequence {b_n} -/
def B : ℕ → ℝ := sorry

/-- a_n is the nth term of the arithmetic sequence {a_n} -/
def a : ℕ → ℝ := sorry

/-- b_n is the nth term of the arithmetic sequence {b_n} -/
def b : ℕ → ℝ := sorry

/-- The ratio of A_n to B_n is (5n - 3) / (n + 9) for all n -/
axiom ratio_condition : ∀ n : ℕ, A n / B n = (5 * ↑n - 3) / (↑n + 9)

theorem a6_b3_ratio : a 6 / b 3 = 26 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a6_b3_ratio_l1331_133139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_of_coefficients_l1331_133101

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The complex number i such that i² = -1 -/
noncomputable def I : ℂ := Complex.I

axiom I_squared : I^2 = -1

/-- Definition of the roots of the polynomial -/
noncomputable def roots (w : ℂ) : Finset ℂ := {w + 3*I, w + 9*I, 2*w - 4}

/-- The theorem to be proven -/
theorem cubic_polynomial_sum_of_coefficients 
  (P : CubicPolynomial) 
  (w : ℂ) 
  (h : ∀ z : ℂ, z ∈ roots w ↔ z^3 + P.a * z^2 + P.b * z + P.c = 0) : 
  P.a + P.b + P.c = -136 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_of_coefficients_l1331_133101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_relation_l1331_133135

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus
def focus : ℝ × ℝ := (0, 1)

-- Define points A and B
variable (A B : ℝ × ℝ)

-- Define that A and B are on the parabola
variable (h_A : is_on_parabola A.1 A.2)
variable (h_B : is_on_parabola B.1 B.2)

-- Define the vector relationship
variable (lambda : ℝ)
variable (h_vector : A.1 - focus.1 = lambda * (B.1 - focus.1) ∧ 
                     A.2 - focus.2 = lambda * (B.2 - focus.2))

-- Define the magnitude of AF
variable (h_AF_magnitude : Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) = 3/2)

-- Theorem statement
theorem parabola_vector_relation :
  lambda = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_relation_l1331_133135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_probability_l1331_133162

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid defined by four points -/
structure Trapezoid where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a trapezoid given its bases and height -/
noncomputable def trapezoidArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The probability of a randomly selected point from the given trapezoid
    being not below the x-axis is (16√2 + 16) / (16√2 + 40) -/
theorem trapezoid_probability (PQRS : Trapezoid) 
    (h1 : PQRS.P = ⟨4, 4⟩)
    (h2 : PQRS.Q = ⟨-4, -4⟩)
    (h3 : PQRS.R = ⟨-10, -4⟩)
    (h4 : PQRS.S = ⟨-2, 4⟩) :
    (trapezoidArea (8 * Real.sqrt 2) 8 4) / 
    (trapezoidArea (8 * Real.sqrt 2) 8 4 + trapezoidArea 6 6 4) = 
    (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_probability_l1331_133162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_volume_theorem_l1331_133178

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  O : Point3D

/-- Represents a line segment -/
structure LineSegment where
  start : Point3D
  endpoint : Point3D  -- Changed 'end' to 'endpoint' to avoid reserved keyword

/-- The volume of the geometric body formed by the trajectory of the midpoint of a moving line segment and the surfaces of a tetrahedron -/
noncomputable def trajectoryVolume (t : Tetrahedron) (l : LineSegment) : Set ℝ :=
  sorry

/-- Given conditions of the problem -/
def problemConditions (t : Tetrahedron) (l : LineSegment) : Prop :=
  -- OA, OB, OC are perpendicular
  (t.O.x - t.A.x) * (t.O.x - t.B.x) = 0 ∧
  (t.O.y - t.A.y) * (t.O.y - t.B.y) = 0 ∧
  (t.O.z - t.A.z) * (t.O.z - t.B.z) = 0 ∧
  (t.O.x - t.A.x) * (t.O.x - t.C.x) = 0 ∧
  (t.O.y - t.A.y) * (t.O.y - t.C.y) = 0 ∧
  (t.O.z - t.A.z) * (t.O.z - t.C.z) = 0 ∧
  (t.O.x - t.B.x) * (t.O.x - t.C.x) = 0 ∧
  (t.O.y - t.B.y) * (t.O.y - t.C.y) = 0 ∧
  (t.O.z - t.B.z) * (t.O.z - t.C.z) = 0 ∧
  -- OA, OB, OC have length 6
  (t.O.x - t.A.x)^2 + (t.O.y - t.A.y)^2 + (t.O.z - t.A.z)^2 = 36 ∧
  (t.O.x - t.B.x)^2 + (t.O.y - t.B.y)^2 + (t.O.z - t.B.z)^2 = 36 ∧
  (t.O.x - t.C.x)^2 + (t.O.y - t.C.y)^2 + (t.O.z - t.C.z)^2 = 36 ∧
  -- MN has length 2
  (l.start.x - l.endpoint.x)^2 + (l.start.y - l.endpoint.y)^2 + (l.start.z - l.endpoint.z)^2 = 4

theorem trajectory_volume_theorem (t : Tetrahedron) (l : LineSegment) :
  problemConditions t l →
  trajectoryVolume t l = {π/6, 36 - π/6} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_volume_theorem_l1331_133178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_with_right_angle_tangents_l1331_133148

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the line
structure PointOnLine where
  x : ℝ
  y : ℝ
  on_line : line x y

-- Define the tangent condition and right angle
def has_right_angle_tangents (p : PointOnLine) : Prop :=
  ∃ (m n : ℝ × ℝ),
    circle_eq m.1 m.2 ∧
    circle_eq n.1 n.2 ∧
    (p.x - m.1)^2 + (p.y - m.2)^2 = 1 ∧
    (p.x - n.1)^2 + (p.y - n.2)^2 = 1 ∧
    (m.1 - p.x) * (n.1 - p.x) + (m.2 - p.y) * (n.2 - p.y) = 0

theorem unique_point_with_right_angle_tangents :
  ∃! (p : PointOnLine), has_right_angle_tangents p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_with_right_angle_tangents_l1331_133148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l1331_133106

/-- Calculates the total distance hiked east given hiking conditions -/
noncomputable def total_distance_east (rate : ℝ) (initial_distance : ℝ) (total_time : ℝ) : ℝ :=
  let time_spent := initial_distance * rate
  let remaining_time := total_time - time_spent
  let additional_distance := (remaining_time / 2) / rate
  initial_distance + additional_distance

/-- Theorem stating that Annika's total distance hiked east is 3.5 kilometers -/
theorem annika_hike_distance :
  let rate : ℝ := 12  -- 12 minutes per kilometer
  let initial_distance : ℝ := 2.75  -- 2.75 kilometers already hiked east
  let total_time : ℝ := 51  -- 51 minutes total time
  total_distance_east rate initial_distance total_time = 3.5 := by
  -- Unfold the definition of total_distance_east
  unfold total_distance_east
  -- Simplify the expressions
  simp
  -- The proof is completed using numerical computation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l1331_133106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1331_133103

/-- A circle C passes through points A(0, 2) and B(2, -2), and its center lies on the line x - y + 1 = 0.
    The standard equation of this circle is (x + 3)² + (y + 2)² = 25. -/
theorem circle_equation (C : Set (ℝ × ℝ)) (h1 : (0, 2) ∈ C) (h2 : (2, -2) ∈ C)
    (h3 : ∃ (x y : ℝ), (x, y) ∈ C ∧ x - y + 1 = 0 ∧ ∀ (p : ℝ × ℝ), p ∈ C → (p.1 - x)^2 + (p.2 - y)^2 = (x - 0)^2 + (y - 2)^2) :
  ∀ (x y : ℝ), (x, y) ∈ C ↔ (x + 3)^2 + (y + 2)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1331_133103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_40_and_41_l1331_133177

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem x_plus_y_between_40_and_41 
  (x y : ℝ) 
  (h1 : y = 3 * (floor x) + 4)
  (h2 : y = 4 * (floor (x - 3)) + 7)
  (h3 : x ≠ ↑(floor x)) :
  40 < x + y ∧ x + y < 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_40_and_41_l1331_133177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_max_value_non_monotonic_condition_l1331_133121

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

noncomputable def f' (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 1

theorem extremum_condition (a : ℝ) : 
  f' a 1 = 0 → a = 0 ∨ a = 2 := by sorry

theorem max_value (a b : ℝ) :
  f a b 1 = 2 ∧ f' a 1 = -1 → 
  ∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≤ 8 := by sorry

theorem non_monotonic_condition (a : ℝ) :
  a ≠ 0 → (∃ x ∈ Set.Ioo (-1 : ℝ) 1, f' a x = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_max_value_non_monotonic_condition_l1331_133121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_numbers_l1331_133160

def numbers : List Nat := [45, 65, 85, 119, 143]

/-- The largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_of_numbers :
  (numbers.map largestPrimeFactor).maximum? = some 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_numbers_l1331_133160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_compartments_l1331_133164

/-- The speed of a train as a function of the number of compartments attached. -/
noncomputable def train_speed (n : ℝ) : ℝ :=
  96 - 24 * Real.sqrt n

/-- The problem statement -/
theorem train_compartments : 
  ∃ (n : ℝ), 
    n > 0 ∧ 
    train_speed n = 24 ∧ 
    train_speed 16 = 0 ∧ 
    n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_compartments_l1331_133164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_satisfaction_correlation_l1331_133168

theorem job_satisfaction_correlation (x : ℕ) : 
  (x > 0 ∧ x < 20 ∧ (20 * x : ℚ) / 99 > (2706 : ℚ) / 1000) →
  (Finset.filter (λ y : ℕ => y > 0 ∧ y < 20 ∧ (20 * y : ℚ) / 99 > (2706 : ℚ) / 1000) (Finset.range 20)).card = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_satisfaction_correlation_l1331_133168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_l1331_133147

/-- Represents a parabola of the form y = ax^2 -/
structure Parabola where
  a : ℝ
  h_pos : a > 0

/-- The distance from the focus to the directrix of a parabola -/
noncomputable def focus_directrix_distance (p : Parabola) : ℝ := 1 / (2 * p.a)

theorem parabola_focus_directrix (p : Parabola) :
  focus_directrix_distance p = 2 → p.a = 1/4 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_l1331_133147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_costume_material_length_l1331_133183

/-- The length of the rectangle of material for each skirt -/
def skirt_length : ℝ := 12

/-- The width of each skirt -/
def skirt_width : ℝ := 4

/-- The number of skirts -/
def num_skirts : ℕ := 3

/-- The area of material required for the bodice -/
def bodice_area : ℝ := 12

/-- The cost of material per square foot -/
def cost_per_sqft : ℝ := 3

/-- The total cost of the material -/
def total_cost : ℝ := 468

theorem costume_material_length :
  skirt_length = 12 ∧
  skirt_width * skirt_length * (num_skirts : ℝ) + bodice_area = total_cost / cost_per_sqft :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_costume_material_length_l1331_133183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_forty_integers_from_five_l1331_133161

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sequence_sum (a₁ : ℝ) (aₙ : ℝ) (n : ℕ) : ℝ := n / 2 * (a₁ + aₙ)

theorem arithmetic_mean_of_forty_integers_from_five (a₁ : ℝ) (n : ℕ) :
  a₁ = 5 → n = 40 → 
  (arithmetic_sequence_sum a₁ (arithmetic_sequence a₁ 1 n) n) / n = 24.5 := by
  sorry

#check arithmetic_mean_of_forty_integers_from_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_forty_integers_from_five_l1331_133161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_ratio_approximation_l1331_133151

/-- The side length of the square -/
def side_length : ℕ := 2020

/-- The probability of a point lying within one of the four circular sectors -/
def probability : ℚ := 1/2

/-- The ratio of the radius to the side length of the square -/
def radius_ratio : ℝ := 0.4

/-- Theorem stating the relationship between the radius ratio and the given probability -/
theorem radius_ratio_approximation (d : ℝ) :
  (4 * Real.pi * d^2) / (side_length^2 : ℝ) = probability →
  |d / side_length - radius_ratio| < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_ratio_approximation_l1331_133151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_exponential_implies_limit_zero_l1331_133107

theorem limit_exponential_implies_limit_zero
  (a : ℝ) (h_a_pos : a > 0) (h_a_neq_one : a ≠ 1)
  (x : ℕ → ℝ) (h_lim : Filter.Tendsto (fun n => a^(x n)) Filter.atTop (nhds 1)) :
  Filter.Tendsto x Filter.atTop (nhds 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_exponential_implies_limit_zero_l1331_133107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bound_by_endpoints_l1331_133170

theorem f_bound_by_endpoints (a b x : ℝ) (ha : a > 0) (hx : x ∈ Set.Icc 0 1) :
  let f := λ t : ℝ => 3 * a * t^2 - 2 * (a + b) * t + b
  |f x| ≤ max (f 0) (f 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bound_by_endpoints_l1331_133170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1331_133172

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1331_133172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1331_133182

def A : Set ℤ := {x | |x| ≤ 2}
def B : Set ℤ := {y : ℤ | ∃ x : ℝ, y = ⌊1 - x^2⌋}

theorem intersection_A_B : A ∩ B = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1331_133182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_odd_l1331_133188

def α_set : Set ℝ := {1, 2, 3, (1/2), -1}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem power_function_domain_and_odd (α : ℝ) :
  α ∈ α_set →
  (Set.univ = {x : ℝ | ∃ y : ℝ, y = x^α}) ∧ is_odd_function (λ x : ℝ => Real.rpow x α) ↔ 
  α = 1 ∨ α = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_odd_l1331_133188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_area_theorem_l1331_133127

/-- Projectile motion model -/
structure ProjectileModel where
  h : ℝ  -- Launch height
  v : ℝ  -- Half of initial velocity
  g : ℝ  -- Acceleration due to gravity

/-- Parametric equations for projectile motion -/
noncomputable def trajectoryEquations (model : ProjectileModel) (θ : ℝ) (t : ℝ) : ℝ × ℝ :=
  (2 * model.v * t * Real.cos θ, 2 * model.v * t * Real.sin θ - (1/2) * model.g * t^2 + model.h)

/-- The curve traced by highest points of projectile trajectories -/
noncomputable def highestPointsCurve (model : ProjectileModel) (θ : ℝ) : ℝ × ℝ :=
  let t := 2 * model.v * Real.sin θ / model.g
  (2 * model.v^2 / model.g * Real.sin (2 * θ), model.h)

/-- Area enclosed by the highest points curve -/
noncomputable def enclosedArea (model : ProjectileModel) : ℝ :=
  2 * Real.pi * (4 * model.v^4 / model.g^2)

theorem projectile_area_theorem (model : ProjectileModel) :
  enclosedArea model = 2 * Real.pi * (4 * model.v^4 / model.g^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_area_theorem_l1331_133127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_solution_l1331_133156

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 * x^2 + 2 * x + 1) - Real.sqrt (4 * x^2 + 14 * x + 5) - (6 * x + 2)

theorem unique_real_solution :
  ∃! x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_solution_l1331_133156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locker_count_proof_l1331_133110

def cost_per_digit (n : ℕ) : ℚ :=
  if n ≤ 1500 then 2/100 else 3/100

def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

noncomputable def total_cost (num_lockers : ℕ) : ℚ :=
  (Finset.range num_lockers).sum (λ i => 
    (digit_count (i + 1) : ℚ) * cost_per_digit (i + 1))

theorem locker_count_proof : 
  total_cost 3009 = 27894/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locker_count_proof_l1331_133110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1331_133186

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt ((e.a^2 - e.b^2) / e.a^2)

/-- The equation of a line y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

theorem ellipse_property (C : Ellipse) (l : Line) (k k' : ℝ) :
  eccentricity C = Real.sqrt 2 / 2 →
  ∃ (circle_center : Point),
    (circle_center.y = l.k * circle_center.x + l.m) ∧
    (circle_center.x^2 + circle_center.y^2 = C.b^2) →
  k ≥ 0 →
  ∀ (P : Point),
    P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1 →
    k' = P.y / P.x →
    k * k' = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l1331_133186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l1331_133145

/-- The area of a right triangle with base 4 and height 2 is 4 -/
theorem park_area (base height : Real)
  (h1 : base = 4)
  (h2 : height = 2) :
  (1 / 2) * base * height = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l1331_133145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_symmetry_implies_values_l1331_133149

/-- The function f(x) = (bx - ab + 1) / (x - a) with center of symmetry (2, -1) -/
noncomputable def f (a b x : ℝ) : ℝ := (b * x - a * b + 1) / (x - a)

/-- The center of symmetry of f is (2, -1) -/
def center_of_symmetry (a b : ℝ) : Prop := ∀ x : ℝ, f a b (4 - x) = -2 - f a b x

theorem center_symmetry_implies_values (a b : ℝ) :
  center_of_symmetry a b → a = 2 ∧ b = -1 := by
  sorry

#check center_symmetry_implies_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_symmetry_implies_values_l1331_133149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l1331_133199

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: Light path length in a cube -/
theorem light_path_length_in_cube (cube_side : ℝ) (p : Point3D) :
  cube_side = 10 →
  p = ⟨10, 4, 3⟩ →
  ∃ (n : ℕ), distance ⟨0, 0, 0⟩ p * n = 50 * Real.sqrt 5 := by
  sorry

/-- Final answer calculation -/
def final_answer : ℕ := 55

#eval final_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_in_cube_l1331_133199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EFD_collinear_l1331_133114

-- Define the given points
variable (A B C D E F P : EuclideanSpace ℝ 2)

-- Define the conditions
variable (triangle_ABC : Triangle A B C)
variable (BC_extended : ∃ t : ℝ, D = B + t • (C - B) ∧ t > 1)
variable (CD_eq_AC : ‖D - C‖ = ‖A - C‖)
variable (P_on_circles : OnCircle P (circumcenter A C D) (circumradius A C D) ∧ 
                         OnCircle P ((B + C) / 2) (‖B - C‖ / 2))
variable (E_on_AC : ∃ t : ℝ, E = A + t • (C - A))
variable (F_on_AB : ∃ t : ℝ, F = A + t • (B - A))
variable (BP_through_E : ∃ t : ℝ, E = B + t • (P - B))
variable (CP_through_F : ∃ t : ℝ, F = C + t • (P - C))

-- State the theorem
theorem EFD_collinear :
  Collinear E F D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EFD_collinear_l1331_133114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_business_class_l1331_133179

def total_passengers : ℕ := 300
def women_percentage : ℚ := 70 / 100
def business_class_percentage : ℚ := 15 / 100

theorem women_in_business_class :
  ⌈(total_passengers : ℚ) * women_percentage * business_class_percentage⌉ = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_business_class_l1331_133179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_correct_l1331_133171

/-- The smallest number of distinct integers in [1, 1000] that guarantees
    the existence of two numbers satisfying the given inequality. -/
def smallest_n : ℕ := 11

/-- Checks if two real numbers satisfy the given inequality. -/
def satisfies_inequality (a b : ℝ) : Prop :=
  0 < |a - b| ∧ |a - b| < 1 + 3 * (a * b) ^ (1/3)

theorem smallest_n_correct :
  (∀ (S : Finset ℝ), S.card = smallest_n → (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 1000) →
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ satisfies_inequality a b) ∧
  (∃ (S : Finset ℝ), S.card = smallest_n - 1 ∧ (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 1000) ∧
    ∀ a b, a ∈ S → b ∈ S → a ≠ b → ¬satisfies_inequality a b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_correct_l1331_133171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_260_l1331_133113

/-- Calculates the length of a platform given the speed of a train, time to cross the platform, and length of the train. -/
noncomputable def platformLength (trainSpeed : ℝ) (crossingTime : ℝ) (trainLength : ℝ) : ℝ :=
  trainSpeed * (5/18) * crossingTime - trainLength

/-- Theorem: The length of the platform is 260 meters. -/
theorem platform_length_is_260 :
  platformLength 72 26 260 = 260 := by
  -- Unfold the definition of platformLength
  unfold platformLength
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_260_l1331_133113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coincidences_value_l1331_133123

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- The number of questions Vasya guessed correctly -/
def vasya_correct : ℕ := 6

/-- The number of questions Misha guessed correctly -/
def misha_correct : ℕ := 8

/-- The probability of Vasya guessing correctly -/
noncomputable def p_vasya : ℝ := vasya_correct / num_questions

/-- The probability of Misha guessing correctly -/
noncomputable def p_misha : ℝ := misha_correct / num_questions

/-- The probability of both Vasya and Misha guessing correctly or incorrectly -/
noncomputable def p_coincidence : ℝ := p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)

/-- The expected number of coincidences -/
noncomputable def expected_coincidences : ℝ := num_questions * p_coincidence

theorem expected_coincidences_value : expected_coincidences = 10.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coincidences_value_l1331_133123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_log2_on_interval_l1331_133198

-- Define the function f(x) = log₂x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem min_value_log2_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ 
    (∀ y ∈ Set.Icc 1 2, f x ≤ f y) ∧ 
    f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_log2_on_interval_l1331_133198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1331_133124

/-- Calculates the speed in km/hour given distance in km and time in minutes -/
noncomputable def calculate_speed (distance : ℝ) (time_minutes : ℝ) : ℝ :=
  distance / (time_minutes / 60)

/-- Theorem stating that for a distance of 1.8 km and a time of 3 minutes, the speed is 36 km/hour -/
theorem speed_calculation :
  calculate_speed 1.8 3 = 36 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the arithmetic
  simp [div_div]
  -- Check that the result is equal to 36
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1331_133124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiders_can_win_l1331_133176

-- Define the dodecahedron graph
structure Dodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  is_regular : Bool
  edge_length : ℝ

-- Define the players
inductive Player
| FastSpider
| SlowSpider1
| SlowSpider2
| Beetle

-- Define the game state
structure GameState where
  positions : Player → ℕ
  time : ℝ

-- Define the maximum speeds
noncomputable def max_speed (p : Player) : ℝ :=
  match p with
  | Player.FastSpider => 1
  | Player.SlowSpider1 => 1 / 2018
  | Player.SlowSpider2 => 1 / 2018
  | Player.Beetle => 1

-- Define the winning condition
def spiders_win (state : GameState) : Prop :=
  ∃ s : Player, s ≠ Player.Beetle ∧ state.positions s = state.positions Player.Beetle

-- Define the theorem
theorem spiders_can_win (d : Dodecahedron) (initial_state : GameState) :
  ∃ (strategy : GameState → Player → ℕ),
    ∀ (beetle_movement : GameState → ℕ),
      ∃ (final_state : GameState), spiders_win final_state :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiders_can_win_l1331_133176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1331_133141

/-- Given a function f and a real number a, with |x - a| < 1, 
    we want to prove or disprove |f(x)f(a)| < 2(|a| + 1) -/
theorem function_inequality (a : ℝ) : 
  ∃ (x : ℝ), |x - a| < 1 ∧ 
  (let f : ℝ → ℝ := fun t ↦ t^2 + t + 13;
   |f x * f a| < 2 * (|a| + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1331_133141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_circle_area_ratio_l1331_133196

/-- The ratio of the area of a regular pentagon to the area of a circle with equal perimeter --/
theorem pentagon_circle_area_ratio :
  ∀ (s r : ℝ), s > 0 → r > 0 →
  5 * s = 2 * Real.pi * r →
  (5 * s^2 * Real.tan (54 * Real.pi / 180) / 4) / (Real.pi * r^2) = Real.pi * Real.tan (54 * Real.pi / 180) / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_circle_area_ratio_l1331_133196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_value_l1331_133133

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 1
def C₂ (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ ∧ 0 ≤ φ ∧ φ < 2 * Real.pi

-- Define the polar coordinates
noncomputable def polar_to_cart (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define points A and B
noncomputable def point_A (α : ℝ) : ℝ × ℝ := polar_to_cart (1 / (Real.cos α + Real.sin α)) α
noncomputable def point_B (α : ℝ) : ℝ × ℝ := polar_to_cart (4 * Real.cos α) α

-- Define the ratio |OB|/|OA|
noncomputable def ratio (α : ℝ) : ℝ := 4 * Real.cos α * (Real.cos α + Real.sin α)

-- State the theorem
theorem max_ratio_value :
  ∃ (max_val : ℝ), max_val = 2 + 2 * Real.sqrt 2 ∧
  ∀ (α : ℝ), 0 ≤ α ∧ α ≤ Real.pi / 2 → ratio α ≤ max_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_value_l1331_133133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_l1331_133104

/-- The set of ordered triples (x,y,z) of nonnegative real numbers that lie in the plane x+y+z=1 -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

/-- The subset of T where exactly one of x ≥ 1/3, y ≥ 1/4, or z ≥ 1/5 is true -/
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | p ∈ T ∧
    ((p.1 ≥ 1/3 ∧ p.2.1 < 1/4 ∧ p.2.2 < 1/5) ∨
     (p.1 < 1/3 ∧ p.2.1 ≥ 1/4 ∧ p.2.2 < 1/5) ∨
     (p.1 < 1/3 ∧ p.2.1 < 1/4 ∧ p.2.2 ≥ 1/5))}

/-- The area of a set in ℝ³ -/
noncomputable def area (A : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of S divided by the area of T equals 1/4 -/
theorem area_ratio_S_T : area S / area T = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_l1331_133104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_sin_over_two_plus_sin_l1331_133105

open Real

theorem indefinite_integral_sin_over_two_plus_sin (x : ℝ) :
  let F := λ x : ℝ => x - (4 / Real.sqrt 3) * arctan ((2 * tan (x / 2) + 1) / Real.sqrt 3)
  deriv F x = sin x / (2 + sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_sin_over_two_plus_sin_l1331_133105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_15_l1331_133125

-- Define the sequence
def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | 4 => 15  -- We replace 'x' with the actual value we're proving
  | 5 => 21
  | 6 => 28
  | n + 7 => mySequence (n + 6) + (n + 7)

-- Define the property that each term is the sum of the previous term and a consecutive natural number
def sequenceProperty (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → s n = s (n - 1) + n

theorem x_equals_15 (h : sequenceProperty mySequence) : mySequence 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_15_l1331_133125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1331_133142

theorem trig_inequality (h : 1 < 20/19 ∧ 20/19 < Real.pi/2) :
  Real.cos (20/19) < Real.arctan (1 / Real.tan (20/19)) ∧ 
  Real.arctan (1 / Real.tan (20/19)) < Real.sin (20/19) ∧ 
  Real.sin (20/19) < Real.tan (20/19) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1331_133142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_period_l1331_133144

/-- The period of the function f(x) = sin x cos x is π -/
theorem sin_cos_period : ∃ T > 0, ∀ x : ℝ, Real.sin x * Real.cos x = Real.sin (x + T) * Real.cos (x + T) ∧ T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_period_l1331_133144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1331_133166

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x - Real.sqrt (1 - 2*x)

-- State the theorem
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, x ≤ (1/2) ∧ g x = y) ↔ y ≤ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1331_133166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_comparison_l1331_133120

theorem savings_comparison (S : ℝ) (h : S > 0) : 
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 0.15 * this_year_salary
  (this_year_savings / last_year_savings) * 100 = 165 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_comparison_l1331_133120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_four_l1331_133126

/-- An odd function f defined on ℝ with a specific form for non-negative inputs -/
noncomputable def f (b : ℝ) : ℝ → ℝ :=
  fun x => if x ≥ 0 then 2^(x+1) + 2*x + b else -(2^(-x+1) + 2*(-x) + b)

/-- The main theorem: f(-1) = -4 for the given function f -/
theorem f_neg_one_eq_neg_four (b : ℝ) :
  (∀ x, f b (-x) = -(f b x)) →  -- f is odd
  f b 0 = 0 →                   -- f(0) = 0
  f b (-1) = -4 := by
  intros h_odd h_zero
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_four_l1331_133126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_problem_solution_l1331_133138

/-- Calculates the downstream distance given the conditions of the swimming problem -/
noncomputable def downstream_distance (upstream_distance : ℝ) (swim_time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let river_speed := (still_water_speed - upstream_distance / swim_time) / 2
  (still_water_speed + river_speed) * swim_time

/-- Theorem stating that given the conditions of the swimming problem, the downstream distance is 42 km -/
theorem swimming_problem_solution :
  downstream_distance 18 3 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_problem_solution_l1331_133138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_one_fifth_l1331_133153

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  -- Line l₁: 3x - 4y + 1 = 0
  l₁ : ℝ → ℝ → Prop
  l₁_def : ∀ x y, l₁ x y ↔ 3 * x - 4 * y + 1 = 0
  -- Line l₂: 6x + my + 4 = 0
  l₂ : ℝ → ℝ → Prop
  m : ℝ
  l₂_def : ∀ x y, l₂ x y ↔ 6 * x + m * y + 4 = 0
  -- Parallel condition
  parallel : m = -8

/-- The minimum distance between two parallel lines -/
noncomputable def min_distance (pl : ParallelLines) : ℝ :=
  1 / 5

/-- Theorem stating that the minimum distance between any point on l₁ and any point on l₂ is 1/5 -/
theorem min_distance_is_one_fifth (pl : ParallelLines) :
  ∀ (p q : ℝ × ℝ), pl.l₁ p.1 p.2 → pl.l₂ q.1 q.2 →
  ∀ (r : ℝ × ℝ), pl.l₁ r.1 r.2 → ∀ (s : ℝ × ℝ), pl.l₂ s.1 s.2 →
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ min_distance pl :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_one_fifth_l1331_133153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_in_nickels_l1331_133184

/-- Given that Alice has 3p + 2 quarters and Bob has 2p + 8 quarters,
    prove that the difference in their money in nickels is 5p - 30. -/
theorem money_difference_in_nickels (p : ℤ) : 
  (5 : ℤ) * ((3 * p + 2) - (2 * p + 8)) = 5 * p - 30 := by
  -- Expand the expression
  calc (5 : ℤ) * ((3 * p + 2) - (2 * p + 8))
       = 5 * (3 * p + 2 - 2 * p - 8) := by ring
     _ = 5 * (p - 6) := by ring
     _ = 5 * p - 30 := by ring

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_in_nickels_l1331_133184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1331_133131

noncomputable def f (x : ℝ) := (1/2) * (Real.exp x - Real.exp (-x))

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :=
by
  sorry

#check f_odd_and_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1331_133131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_cosine_sine_sum_l1331_133174

/-- In an acute triangle, the sum of cosines of its angles is less than the sum of sines of its angles. -/
theorem acute_triangle_cosine_sine_sum (A B C : ℝ) : 
  0 < A → A < π/2 →
  0 < B → B < π/2 →
  0 < C → C < π/2 →
  A + B + C = π →
  Real.cos A + Real.cos B + Real.cos C < Real.sin A + Real.sin B + Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_cosine_sine_sum_l1331_133174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equations_equivalence_xy_product_bounds_xy_product_extrema_l1331_133122

/-- A circle in polar coordinates -/
def PolarCircle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi/4) + 6 = 0

/-- A circle in Cartesian coordinates -/
def CartesianCircle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

/-- Parametric representation of the circle -/
def ParametricCircle (θ x y : ℝ) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos θ ∧ y = 2 + Real.sqrt 2 * Real.sin θ

/-- The product xy for points on the circle -/
noncomputable def XYProduct (θ : ℝ) : ℝ :=
  (2 + Real.sqrt 2 * Real.cos θ) * (2 + Real.sqrt 2 * Real.sin θ)

theorem circle_equations_equivalence :
  ∀ ρ θ x y,
  PolarCircle ρ θ ↔ CartesianCircle x y ↔ ∃ θ', ParametricCircle θ' x y := by
  sorry

theorem xy_product_bounds :
  ∀ θ, 1 ≤ XYProduct θ ∧ XYProduct θ ≤ 9 := by
  sorry

theorem xy_product_extrema :
  (∃ θ, XYProduct θ = 1) ∧ (∃ θ, XYProduct θ = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equations_equivalence_xy_product_bounds_xy_product_extrema_l1331_133122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1331_133167

theorem coefficient_x_cubed_in_expansion : 
  let f : Polynomial ℤ := (1 - X)^5 * (1 + X)^3
  f.coeff 3 = -14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1331_133167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_two_tangent_through_one_one_l1331_133143

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem for the tangent line at (1, 2)
theorem tangent_at_one_two :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    (y = m * x + b) ↔ (y - f 1 = f' 1 * (x - 1) ∧ m = 2 ∧ b = 0) :=
sorry

-- Theorem for the tangent line through (1, 1)
theorem tangent_through_one_one :
  ∃ (x₀ m b : ℝ), 
    x₀ = 2 ∧
    (∀ x y : ℝ, y = m * x + b ↔ 
      (y - f x₀ = f' x₀ * (x - x₀) ∧ 
       1 = m * 1 + b ∧
       m = 4 ∧ b = -3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_two_tangent_through_one_one_l1331_133143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1331_133169

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance from a point to the focus
noncomputable def distance_to_focus (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 1)^2 + y^2)

theorem parabola_point_x_coordinate 
  (x y : ℝ) 
  (h1 : is_on_parabola x y) 
  (h2 : distance_to_focus x y = 5) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1331_133169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_m_range_l1331_133175

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Define the condition given in the problem
axiom condition (x : ℝ) :
  x > 1 → (x + x * Real.log x) * f' x > f x

-- Define the function g
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / (1 + Real.log x)

-- Theorem 1: g is increasing on (1, +∞)
theorem g_increasing (f : ℝ → ℝ) :
  ∀ x > 1, StrictMono (g f) := sorry

-- Define the specific function f(x) = e^x + mx
noncomputable def f_specific (m : ℝ) (x : ℝ) : ℝ := Real.exp x + m * x

-- Theorem 2: The range of m is [-2e, +∞)
theorem m_range :
  ∃ m₀ : ℝ, m₀ = -2 * Real.exp 1 ∧
  ∀ m : ℝ, (∀ x > 1, (x + x * Real.log x) * (Real.exp x + m) > Real.exp x + m * x) ↔ m ≥ m₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_m_range_l1331_133175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5_049_l1331_133195

noncomputable def round_to_one_decimal (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_5_049 :
  round_to_one_decimal 5.049 = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5_049_l1331_133195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_mean_score_l1331_133185

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℕ) 
  (non_senior_count : ℕ) 
  (senior_mean : ℝ) 
  (non_senior_mean : ℝ) :
  total_students = 120 →
  overall_mean = 100 →
  non_senior_count = 2 * senior_count →
  senior_mean = 1.6 * non_senior_mean →
  total_students * overall_mean = senior_count * senior_mean + non_senior_count * non_senior_mean →
  abs (senior_mean - 133.33) < 0.01 :=
by
  intros h1 h2 h3 h4 h5
  sorry

#check senior_mean_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_senior_mean_score_l1331_133185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_upper_bound_l1331_133108

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

-- Theorem 1: f is monotonically decreasing on [1, +∞)
theorem f_decreasing (x : ℝ) (h : x ≥ 1) : 
  HasDerivAt f (1/x - 2*x + 1) x ∧ (1/x - 2*x + 1 < 0) := by sorry

-- Theorem 2: f(x) ≤ x² + 2x - 1 for all x > 0, and 2 is the minimum integer satisfying this
theorem f_upper_bound (x : ℝ) (h : x > 0) : 
  f x ≤ x^2 + 2*x - 1 ∧ 
  ∀ a : ℕ, (∀ y : ℝ, y > 0 → f y ≤ ((a : ℝ)/2 - 1)*y^2 + a*y - 1) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_upper_bound_l1331_133108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_is_282_l1331_133109

/-- Represents the problem of calculating the ship's length given Emily's walking parameters. -/
structure ShipLengthProblem where
  emily_speed : ℚ  -- Emily's walking speed in units per minute
  ship_speed : ℚ   -- Ship's downstream speed in units per minute
  steps_downstream : ℕ  -- Number of steps Emily takes from back to front of the ship
  steps_upstream : ℕ    -- Number of steps Emily takes from front to back of the ship

/-- Calculates the length of the ship in terms of Emily's steps. -/
def ship_length (p : ShipLengthProblem) : ℚ :=
  ((p.steps_downstream : ℚ) * (p.emily_speed - p.ship_speed) + 
   (p.steps_upstream : ℚ) * (p.emily_speed + p.ship_speed)) / 
  (p.emily_speed - p.ship_speed + p.emily_speed + p.ship_speed)

/-- Theorem stating that for the given problem parameters, the ship's length is 282 steps. -/
theorem ship_length_is_282 : 
  let p : ShipLengthProblem := {
    emily_speed := 7,
    ship_speed := 4,
    steps_downstream := 308,
    steps_upstream := 56
  }
  ship_length p = 282 := by
  -- Proof goes here
  sorry

#eval ship_length {
  emily_speed := 7,
  ship_speed := 4,
  steps_downstream := 308,
  steps_upstream := 56
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_is_282_l1331_133109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_91_l1331_133191

def a (n : ℕ) : ℕ := 
  202000000 + 2000000 * ((10^n - 1) / 9) + 20

theorem divisible_by_91 (n : ℕ) : 91 ∣ a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_91_l1331_133191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_of_digits_l1331_133115

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ ({5, 6, 7, 8} : Set Nat) ∧ 
  b ∈ ({5, 6, 7, 8} : Set Nat) ∧ 
  c ∈ ({5, 6, 7, 8} : Set Nat) ∧ 
  d ∈ ({5, 6, 7, 8} : Set Nat) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product_of_arrangement (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product_of_digits :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
  product_of_arrangement a b c d ≥ 4368 := by
  sorry

#check smallest_product_of_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_of_digits_l1331_133115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1331_133154

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1

-- Define the ellipse
def ellipse (x y m : ℝ) : Prop := m * x^2 + m * y^2 = 1

-- Define the hyperbola
def hyperbola (x y m n : ℝ) : Prop := y^2 / m^2 - x^2 / n^2 = 1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (m n : ℝ) : ℝ := Real.sqrt (1 + n^2 / m^2)

-- State the theorem
theorem hyperbola_eccentricity (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (∃ x1 y1 x2 y2 : ℝ,
    line x1 y1 ∧ line x2 y2 ∧
    ellipse x1 y1 m ∧ ellipse x2 y2 m ∧
    (x1 + x2) / 2 = -1/3) →
  eccentricity m n = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1331_133154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l1331_133190

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the line
def lineEq (x y : ℝ) : Prop := y = x + 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ circleEq x y ∧ lineEq x y}

-- Theorem statement
theorem distance_between_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 6 * Real.sqrt 2 := by
  sorry

#check distance_between_intersection_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l1331_133190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_killing_two_birds_random_l1331_133136

/-- Represents an idiom --/
inductive Idiom
| CatchingTurtle
| KillingTwoBirds
| FishingMoon
| DrippingWater

/-- Defines the characteristics of an idiom --/
def characteristics (i : Idiom) : Prop :=
  match i with
  | Idiom.CatchingTurtle => ∃ p, p = "Can be planned and executed with certainty"
  | Idiom.KillingTwoBirds => ∃ p, p = "Achieving two outcomes with a single action, success varies"
  | Idiom.FishingMoon => ∃ p, p = "Attempting something impossible, guaranteed failure"
  | Idiom.DrippingWater => ∃ p, p = "Certain process given enough time"

/-- Defines a random event --/
def isRandomEvent (i : Idiom) : Prop :=
  ∃ p, p = "Achieving two outcomes with a single action, success varies" ∧ characteristics i = (∃ q, q = p)

/-- Theorem: Killing two birds with one stone is the only idiom describing a random event --/
theorem killing_two_birds_random : 
  (∀ i : Idiom, isRandomEvent i ↔ i = Idiom.KillingTwoBirds) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_killing_two_birds_random_l1331_133136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sqrt_l1331_133155

-- Define the original function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the proposed inverse function f_inv
def f_inv (x : ℝ) : ℝ := x^2

-- State the theorem
theorem inverse_of_sqrt : 
  (∀ x ≥ 0, f (f_inv x) = x) ∧ 
  (∀ x ≥ 0, f_inv (f x) = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sqrt_l1331_133155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1331_133112

-- Define the function f(x) = 2x - ln(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

-- State the theorem
theorem f_properties :
  -- Part 1: Monotonicity intervals
  (∀ x > (1/2 : ℝ), (deriv f x) > 0) ∧
  (∀ x, 0 < x → x < (1/2 : ℝ) → (deriv f x) < 0) ∧
  -- Part 2: Lower bound condition
  (∃ k : ℝ, k = 2 - 1 / Real.exp 1 ∧
    (∀ x ≥ 1, f x ≥ k * x) ∧
    (∀ k' > k, ∃ x ≥ 1, f x < k' * x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1331_133112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l1331_133157

/-- The circle equation x² + y² - 2y = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (0, 1)

/-- The given point that the line passes through -/
noncomputable def given_point : ℝ × ℝ := (Real.sqrt 3, -2)

/-- The slope of the line passing through the given point and the circle center -/
noncomputable def line_slope : ℝ := 
  (circle_center.2 - given_point.2) / (circle_center.1 - given_point.1)

/-- The acute angle between the line and the positive x-axis -/
noncomputable def acute_angle (θ : ℝ) : Prop :=
  θ = Real.arctan (abs line_slope) * (180 / Real.pi)

theorem line_angle_is_60_degrees :
  ∃ θ, acute_angle θ ∧ θ = 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l1331_133157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_16_l1331_133193

theorem square_root_of_16 : Real.sqrt 16 = 4 ∨ Real.sqrt 16 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_16_l1331_133193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_four_l1331_133102

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  hd : d ≠ 0
  ha1 : a 1 = 1
  hGeom : (a 3)^2 = (a 1) * (a 13)
  hArith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The expression to be minimized -/
noncomputable def F (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (2 * S seq n + 16) / (seq.a n + 3)

theorem min_value_is_four (seq : ArithmeticSequence) :
  ∃ n₀ : ℕ, ∀ n : ℕ, F seq n ≥ 4 ∧ F seq n₀ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_four_l1331_133102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_area_l1331_133130

/-- Function to calculate the area of an inscribed regular octagon given the circle's area -/
noncomputable def area_of_inscribed_regular_octagon (circle_area : ℝ) : ℝ :=
  let r := Real.sqrt (circle_area / Real.pi)
  (2 * r^2) * (1 + Real.sqrt 2)

/-- The area of a regular octagon inscribed in a circle with area 400π -/
theorem inscribed_octagon_area : 
  ∀ (circle_area : ℝ) (octagon_area : ℝ),
    circle_area = 400 * Real.pi →
    octagon_area = (20^2) * (1 + Real.sqrt 2) →
    octagon_area = area_of_inscribed_regular_octagon circle_area :=
by
  intros circle_area octagon_area h1 h2
  unfold area_of_inscribed_regular_octagon
  sorry  -- The proof is omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_area_l1331_133130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1331_133159

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 3)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' point.1

theorem tangent_line_equation :
  ∀ x y : ℝ, (y - point.2 = m * (x - point.1)) ↔ (2*x - y + 1 = 0) := by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1331_133159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1331_133181

-- Define the function representing the left side of the inequality
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x - x^a + Real.exp (-x)

-- State the theorem
theorem min_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, x > 1 → f a x ≥ 0) →
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x ≥ 0) → a ≥ -Real.exp 1) ∧
  (∃ a : ℝ, (∀ x : ℝ, x > 1 → f a x ≥ 0) ∧ a = -Real.exp 1) :=
by
  sorry

#check min_value_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1331_133181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_in_alley_l1331_133111

/-- The width of an alley where a ladder can rest against opposite walls -/
noncomputable def alley_width (ℓ : ℝ) : ℝ :=
  ℓ * (1/2 + Real.cos (70 * Real.pi / 180))

/-- Theorem stating the width of the alley given ladder positions -/
theorem ladder_in_alley (ℓ w : ℝ) (h_pos : ℓ > 0) :
  (∃ (m n : ℝ),
    0 < m ∧ 0 < n ∧
    m = ℓ * Real.sin (60 * Real.pi / 180) ∧
    n = ℓ * Real.sin (70 * Real.pi / 180) ∧
    w = ℓ * Real.cos (60 * Real.pi / 180) + ℓ * Real.cos (70 * Real.pi / 180)) →
  w = alley_width ℓ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_in_alley_l1331_133111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_problems_with_common_answers_l1331_133132

-- Define the student_answer function
def student_answer : ℕ → ℕ → ℕ := sorry

-- Define the valid_assignment predicate
def valid_assignment (n k m : ℕ) (assignment : ℕ → ℕ → ℕ) : Prop := sorry

theorem max_problems_with_common_answers (n k m : ℕ) : 
  n = 16 → 
  k = 4 → 
  (∀ i j, i ≠ j → ∃! q, student_answer i q = student_answer j q) → 
  m ≤ 5 ∧ ∃ assignment, valid_assignment n k m assignment :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_problems_with_common_answers_l1331_133132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_katies_wallet_l1331_133194

/-- Represents the number of bills of a specific denomination -/
abbrev NumBills : Type := Nat

/-- Represents the value of a bill in dollars -/
abbrev BillValue : Type := Nat

/-- Calculates the total value of bills given the number of bills and their value -/
def totalValue (num : NumBills) (value : BillValue) : Nat :=
  num * value

theorem katies_wallet 
  (total_amount : Nat) 
  (five_dollar_bills : NumBills) 
  (ten_dollar_bills : NumBills) 
  (h1 : total_amount = 80)
  (h2 : totalValue five_dollar_bills 5 + totalValue ten_dollar_bills 10 = total_amount)
  (h3 : five_dollar_bills + ten_dollar_bills = 12) :
  five_dollar_bills = 8 := by
  sorry

#eval totalValue 8 5 + totalValue 4 10  -- Should output 80
#eval 8 + 4  -- Should output 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_katies_wallet_l1331_133194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_length_l1331_133128

/-- Triangle ABC with sides AB = 500, BC = 550, AC = 600 -/
structure Triangle :=
  (AB : ℝ) (BC : ℝ) (AC : ℝ)

/-- Segments through P parallel to sides of the triangle -/
structure ParallelSegments :=
  (d : ℝ)

/-- The theorem stating the relationship between the triangle sides and the length of parallel segments -/
theorem parallel_segments_length 
  (t : Triangle) 
  (p : ParallelSegments) 
  (h1 : t.AB = 500) 
  (h2 : t.BC = 550) 
  (h3 : t.AC = 600) : 
  abs (p.d - 251.145) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_length_l1331_133128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l1331_133118

noncomputable def numbers : List ℝ := [0.4, -2/7, Real.sqrt 5, 2023, -Real.sqrt 0.01, -0.030030003]

def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q
def is_irrational (x : ℝ) : Prop := ¬(is_rational x)

noncomputable def positive_set : List ℝ := [0.4, Real.sqrt 5, 2023]
noncomputable def negative_set : List ℝ := [-2/7, -Real.sqrt 0.01, -0.030030003]
noncomputable def rational_set : List ℝ := [0.4, -2/7, 2023, -Real.sqrt 0.01]
noncomputable def irrational_set : List ℝ := [Real.sqrt 5, -0.030030003]

theorem number_classification :
  (∀ x ∈ positive_set, is_positive x) ∧
  (∀ x ∈ negative_set, is_negative x) ∧
  (∀ x ∈ rational_set, is_rational x) ∧
  (∀ x ∈ irrational_set, is_irrational x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_classification_l1331_133118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_is_five_sixths_l1331_133129

/-- Represents the cost price of a single chocolate -/
noncomputable def C : ℝ := sorry

/-- Represents the selling price of a single chocolate -/
noncomputable def S : ℝ := sorry

/-- The condition that the cost price of 44 chocolates equals the selling price of 24 chocolates -/
axiom price_equality : 44 * C = 24 * S

/-- The definition of gain percent -/
noncomputable def gain_percent : ℝ := (S - C) / C * 100

/-- Theorem stating that under the given condition, the gain percent is 5/6 * 100 -/
theorem gain_percent_is_five_sixths :
  gain_percent = 5/6 * 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_is_five_sixths_l1331_133129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_equals_zero_l1331_133173

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (3 : ℝ) ^ a = (81 : ℝ) ^ (b + 1))
  (h2 : (125 : ℝ) ^ b = (5 : ℝ) ^ (a - 4)) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_equals_zero_l1331_133173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_slope_product_perpendicular_lines_a_value_l1331_133158

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def two_lines_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Theorem stating that two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (m₁ m₂ : ℝ) :
  two_lines_perpendicular m₁ m₂ ↔ m₁ * m₂ = -1 := by sorry

/-- Given two lines ax + 2y + 2 = 0 and 3x + (a-1)y - a + 5 = 0 are perpendicular, prove a = 2/5 -/
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, two_lines_perpendicular (-a/2) (3/(a-1)) → a = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_slope_product_perpendicular_lines_a_value_l1331_133158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_l1331_133180

theorem cosine_value (α : ℝ) 
  (h1 : Real.cos (α - π/6) = 15/17) 
  (h2 : α ∈ Set.Ioo (π/6) (π/2)) : 
  Real.cos α = (15 * Real.sqrt 3 - 8) / 34 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_l1331_133180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l1331_133134

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = Real.sqrt 3 * t.b * t.c ∧
  t.a = 1

-- Theorem statement
theorem circumcircle_area (t : Triangle) 
  (h1 : triangle_properties t) 
  (h2 : given_conditions t) : 
  ∃ R : ℝ, R > 0 ∧ Real.pi * R^2 = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l1331_133134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_3_side_c_is_one_max_area_l1331_133192

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.b * t.c = t.b^2 + t.c^2

-- Helper function for area
noncomputable def area (t : Triangle) : ℝ := 
  (1 / 2) * t.b * t.c * Real.sin t.A

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfiesCondition t) : 
  t.A = π / 3 := by sorry

-- Theorem 2
theorem side_c_is_one (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.b = 2) (h3 : t.a = Real.sqrt 3) :
  t.c = 1 := by sorry

-- Theorem 3
theorem max_area (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : t.A = π / 3) :
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / 4 ∧ 
  ∀ (t' : Triangle), t'.a = t.a → t'.A = t.A → area t' ≤ max_area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_3_side_c_is_one_max_area_l1331_133192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1331_133165

theorem evaluate_expression : (64 : ℝ) ^ (-(2 : ℝ) ^ (-(3 : ℝ))) = 1 / (8 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1331_133165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1331_133163

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angles_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

-- State the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h : Real.sqrt ((1 - Real.cos (2 * t.C)) / 2) + Real.sin (t.B - t.A) = 2 * Real.sin (2 * t.A)) :
  (t.a / t.b = 1/2) ∧ 
  (t.c ≤ t.b → 0 < Real.cos t.C ∧ Real.cos t.C ≤ 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1331_133163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_ten_factorial_l1331_133152

theorem divisors_of_ten_factorial (n : ℕ) (h : n = 10) : 
  (Finset.filter (· ∣ n.factorial) (Finset.range (n.factorial + 1))).card = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_ten_factorial_l1331_133152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_from_A_l1331_133150

-- Define the hexagon
structure Hexagon :=
  (sides : Fin 6 → ℝ)
  (inscribed : Bool)

-- Define the properties of our specific hexagon
def special_hexagon : Hexagon :=
  { sides := λ i => if i = 0 then 36 else 90,
    inscribed := true }

-- Define the diagonals from vertex A
def diagonals (h : Hexagon) : Fin 3 → ℝ := sorry

-- Theorem statement
theorem sum_of_diagonals_from_A (h : Hexagon) :
  h = special_hexagon →
  (diagonals h 0) + (diagonals h 1) + (diagonals h 2) = 428.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_diagonals_from_A_l1331_133150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_travel_time_l1331_133100

/-- Represents the state of a traffic control device (railway crossing or traffic light) -/
inductive DeviceState
| Open
| Closed

/-- Represents a traffic control device on the highway -/
structure Device where
  location : ℚ  -- Location in km (using rational numbers instead of reals)
  openTime : ℚ  -- Time in minutes the device is open
  closedTime : ℚ  -- Time in minutes the device is closed

/-- Represents the highway with its devices -/
structure Highway where
  length : ℚ
  devices : List Device

/-- Calculates the state of a device at a given time -/
def deviceState (d : Device) (t : ℚ) : DeviceState :=
  if (t % (d.openTime + d.closedTime)) < d.closedTime then DeviceState.Closed else DeviceState.Open

/-- Checks if a given position and time is valid for travel -/
def isValidPosition (h : Highway) (position : ℚ) (time : ℚ) : Prop :=
  ∀ d ∈ h.devices, position ≥ d.location → deviceState d time = DeviceState.Open

/-- The main theorem to prove -/
theorem shortest_travel_time (h : Highway) : 
  h.length = 12 ∧ 
  h.devices = [
    ⟨2, 3, 3⟩,  -- Railway crossing
    ⟨4, 3, 2⟩,  -- First traffic light
    ⟨6, 3, 2⟩   -- Second traffic light
  ] →
  (∃ t : ℚ, 
    t = 24 ∧
    (∀ position : ℚ, 0 ≤ position ∧ position ≤ h.length → 
      isValidPosition h position (t * position / h.length)) ∧
    (∀ t' : ℚ, t' < t → 
      ¬(∀ position : ℚ, 0 ≤ position ∧ position ≤ h.length → 
        isValidPosition h position (t' * position / h.length)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_travel_time_l1331_133100
