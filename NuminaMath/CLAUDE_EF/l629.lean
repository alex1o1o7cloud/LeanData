import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_lateral_surface_l629_62945

/-- If the lateral surface development diagram of a cone is a semicircle with an area of 2π, 
    then the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (r l h : ℝ) : 
  r > 0 ∧ l > 0 ∧ h > 0 → 
  (1/2) * Real.pi * l^2 = 2 * Real.pi → 
  2 * Real.pi * r = 2 * Real.pi → 
  l^2 = r^2 + h^2 → 
  (1/3) * Real.pi * r^2 * h = (Real.sqrt 3 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_lateral_surface_l629_62945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_mowing_days_l629_62910

/-- The rate for mowing a lawn in dollars per hour -/
noncomputable def mowing_rate : ℚ := 14

/-- The number of hours David mowed per day -/
noncomputable def hours_per_day : ℚ := 2

/-- The amount of money David had left after spending on shoes and giving to his mom -/
noncomputable def money_left : ℚ := 49

/-- The number of days David mowed lawns -/
noncomputable def mowing_days : ℚ := (money_left * 4) / (mowing_rate * hours_per_day)

theorem david_mowing_days : mowing_days = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_mowing_days_l629_62910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l629_62989

-- Define the sum of the geometric series
noncomputable def S (r : ℝ) : ℝ := 12 / (1 - r)

-- Theorem statement
theorem geometric_series_sum_property 
  (a : ℝ) 
  (h1 : -1 < a) 
  (h2 : a < 1) 
  (h3 : S a * S (-a) = 2016) : 
  S a + S (-a) = 336 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_property_l629_62989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l629_62940

/-- The hyperbola C: x²/a² - y²/b² = 1 with a > 0 and b > 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : Point × Point := sorry

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := sorry

/-- Slope of a line between two points -/
noncomputable def line_slope (p q : Point) : ℝ := sorry

/-- Origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point lies on the hyperbola -/
def on_hyperbola (p : Point) (h : Hyperbola) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) 
  (h_on : on_hyperbola p h) 
  (h_dist : distance p origin = (1/2) * distance (foci h).1 (foci h).2)
  (h_slope : line_slope p origin = Real.sqrt 3) :
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l629_62940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l629_62924

theorem range_of_t (t : ℝ) : 
  (∃ (f : ℝ → ℝ) (x₀ : ℝ), 
    (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 
      f = λ x ↦ 2*a*x^2 + 2*b*x) ∧
    x₀ ∈ Set.Ioo 0 t ∧
    (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → f x₀ = a + b)) →
  t > 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l629_62924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l629_62954

noncomputable section

open Real

/-- The inequality function -/
def f (x y : ℝ) : ℝ :=
  (y^2 - (arccos (cos x))^2) *
  (y^2 - (arccos (cos (x + π/3)))^2) *
  (y^2 - (arccos (cos (x - π/3)))^2)

/-- The bounding curves -/
def g₁ (x : ℝ) : ℝ := arccos (cos x)
def g₂ (x : ℝ) : ℝ := arccos (cos (x + π/3))
def g₃ (x : ℝ) : ℝ := arccos (cos (x - π/3))

/-- The theorem statement -/
theorem inequality_solution_set (x y : ℝ) :
  f x y < 0 ↔ 
  (∃ i j k : Bool, 
    (i = true ↔ y > g₁ x ∨ y < -g₁ x) ∧
    (j = true ↔ y > g₂ x ∨ y < -g₂ x) ∧
    (k = true ↔ y > g₃ x ∨ y < -g₃ x) ∧
    (i = true) ≠ (j = true) ∨ (j = true) ≠ (k = true) ∨ (i = true) ≠ (k = true)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l629_62954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l629_62922

/-- The area of a triangle with side lengths 7, 3, and 8 is 6√3 -/
theorem triangle_area_specific : ∃ (A B C : ℝ) (S : ℝ),
  (A + B + C = π) →
  (Real.cos A = (3^2 + 8^2 - 7^2) / (2 * 3 * 8)) →
  S = 1/2 * 3 * 8 * Real.sin A →
  S = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l629_62922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_n_l629_62931

open Real BigOperators

-- Define the sum of fractions
noncomputable def sum_of_fractions : ℝ := ∑ k in Finset.range 59, 1 / (sin ((k + 30) * π / 180) * sin ((k + 31) * π / 180))

-- Define the theorem
theorem smallest_positive_integer_n :
  ∃ (n : ℕ), n > 0 ∧ sum_of_fractions = 1 / cos (n * π / 180) ∧
  ∀ (m : ℕ), m > 0 → m < n → sum_of_fractions ≠ 1 / cos (m * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_n_l629_62931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_l629_62916

theorem binomial_coefficient_divisibility (a : ℕ) (ha : a > 0) :
  ∃ (k₁ k₂ k₃ : ℕ), (2 * a).choose a = 2 * k₁ ∧
                    (2 * a).choose a = (a + 1) * k₂ ∧
                    (2 * a).choose a = (2 * a - 1) * k₃ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_l629_62916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_prime_divisors_50_factorial_l629_62917

/-- The number of prime divisors of 50 factorial -/
def num_prime_divisors_50_factorial : ℕ := 15

/-- The set of prime numbers less than or equal to 50 -/
def primes_le_50 : Finset ℕ := Finset.filter (fun p => p.Prime) (Finset.range 51)

theorem count_prime_divisors_50_factorial :
  primes_le_50.card = num_prime_divisors_50_factorial :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_prime_divisors_50_factorial_l629_62917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_lambda_l629_62994

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Given vectors a and b, if a is collinear with b, then lambda = -2 -/
theorem collinear_implies_lambda (lambda : ℝ) :
  let a : ℝ × ℝ := (lambda + 1, 2)
  let b : ℝ × ℝ := (1, -2)
  collinear a b → lambda = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_lambda_l629_62994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tetrahedron_volume_l629_62943

/-- The volume of the tetrahedron PACB formed by folding right triangle ABC along PC. -/
noncomputable def tetrahedron_volume (a b : ℝ) (P : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of tetrahedron volume based on a, b, and P's position

/-- Given a right triangle ABC with ∠ACB = 90°, BC = a > 0, CA = b > 0, and point P on AB,
    the maximum volume of tetrahedron PACB formed by folding ABC along PC is
    (1/6) * (a^2 * b^2) / (a^(2/3) + b^(2/3))^(3/2). -/
theorem max_tetrahedron_volume (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (V : ℝ), V = (1/6) * (a^2 * b^2) / (a^(2/3) + b^(2/3))^(3/2) ∧
  ∀ (P : ℝ × ℝ), P.1 ≥ 0 ∧ P.1 ≤ Real.sqrt (a^2 + b^2) →
    tetrahedron_volume a b P ≤ V :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tetrahedron_volume_l629_62943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_william_journey_time_l629_62934

/-- Calculates the total time spent on a journey with time zone difference, stops, and delays -/
noncomputable def journey_time (departure_time : Nat) (arrival_time : Nat) (time_zone_diff : Nat) 
  (stops : List Nat) (traffic_delay : Nat) : Real :=
  let journey_hours := (arrival_time - (departure_time + time_zone_diff) : Real)
  let stop_time := (stops.sum / 60 : Real)
  let delay_time := (traffic_delay / 60 : Real)
  journey_hours + stop_time + delay_time

/-- William's journey theorem -/
theorem william_journey_time :
  journey_time 7 20 2 [25, 10, 25] 45 = 12.75 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval journey_time 7 20 2 [25, 10, 25] 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_william_journey_time_l629_62934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_coefficient_equals_360_sqrt3i_l629_62966

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the coefficient of the 8th term in the expansion of (√3i - x)^10
noncomputable def eighth_term_coefficient : ℂ := 
  (binomial 10 7 : ℂ) * (Complex.I * Real.sqrt 3) ^ 3 * (-1) ^ 7

-- Theorem statement
theorem eighth_term_coefficient_equals_360_sqrt3i :
  eighth_term_coefficient = 360 * Complex.I * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_coefficient_equals_360_sqrt3i_l629_62966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_segment_volume_formula_l629_62980

/-- Represents a segment of a circle -/
structure CircleSegment where
  radius : ℝ
  chord_length : ℝ
  chord_projection : ℝ
  chord_not_intersect_diameter : chord_projection < radius

/-- Calculates the volume of the solid formed by rotating the circle segment -/
noncomputable def rotated_segment_volume (segment : CircleSegment) : ℝ :=
  (Real.pi * segment.chord_length^2 * segment.chord_projection) / 6

/-- Theorem stating that the volume of the rotated segment is as calculated -/
theorem rotated_segment_volume_formula (segment : CircleSegment) :
  rotated_segment_volume segment = (Real.pi * segment.chord_length^2 * segment.chord_projection) / 6 := by
  -- Unfold the definition of rotated_segment_volume
  unfold rotated_segment_volume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_segment_volume_formula_l629_62980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_6000_l629_62961

-- Define the given values
noncomputable def train_length : ℝ := 1200 -- meters
noncomputable def train_speed : ℝ := 96 -- km/hr
noncomputable def time_to_cross : ℝ := 1.5 -- minutes

-- Define the function to calculate the tunnel length
noncomputable def tunnel_length (train_length speed time : ℝ) : ℝ :=
  (speed * 1000 / 60) * time - train_length

-- Theorem statement
theorem tunnel_length_is_6000 :
  tunnel_length train_length train_speed time_to_cross = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_6000_l629_62961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_linear_function_k_range_l629_62958

-- Part 1
noncomputable def f₁ (x : ℝ) := x^2 + x
noncomputable def h₁ (x : ℝ) := -x^2 + x

theorem unique_linear_function (k b : ℝ) :
  (∀ x, f₁ x ≥ k * x + b) ∧ (∀ x, k * x + b ≥ h₁ x) →
  k = 1 ∧ b = 0 := by
  sorry

-- Part 2
noncomputable def f₂ (x : ℝ) := x^2 + x + 2
noncomputable def h₂ (x : ℝ) := x - 1/x

theorem k_range (k : ℝ) :
  (∀ x > 0, f₂ x ≥ k * x + 1) ∧ (∀ x > 0, k * x + 1 ≥ h₂ x) ↔
  1 ≤ k ∧ k ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_linear_function_k_range_l629_62958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l629_62919

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (c : ℝ), c > 0 ∧ 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ∧
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
    (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) →
    ({x, y, z} : Set ℝ) ⊆ {a, b, c} →
    (1/2 : ℝ) * x * y ≥ (1/2 : ℝ) * a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l629_62919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_problem_l629_62953

/-- Calculates the time (in minutes) for a train to pass through a tunnel -/
noncomputable def train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length_km : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let tunnel_length_m := tunnel_length_km * 1000
  let total_distance := train_length + tunnel_length_m
  let time_seconds := total_distance / train_speed_ms
  time_seconds / 60

/-- Theorem stating that a train of length 100 meters traveling at 72 km/hr through a tunnel of length 2.9 km takes 2.5 minutes to pass through -/
theorem train_tunnel_problem :
  train_tunnel_time 100 72 2.9 = 2.5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_problem_l629_62953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l629_62951

theorem sin_alpha_value (m : ℝ) (α : ℝ) :
  (m, 9) ∈ Set.range (λ t : ℝ × ℝ => (t.1 * Real.cos α, t.1 * Real.sin α)) →
  Real.tan α = 3/4 →
  Real.sin α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l629_62951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l629_62901

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the line
def my_line (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

-- Statement to prove
theorem max_distance_circle_to_line :
  ∃ (d : ℝ), d = 8 ∧
  ∀ (P : ℝ × ℝ), my_circle P.1 P.2 →
  ∀ (Q : ℝ × ℝ), my_line Q.1 Q.2 →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l629_62901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_two_l629_62936

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

noncomputable def f' (x : ℝ) : ℝ := 1 / ((x + 1)^2)

theorem tangent_line_at_negative_two :
  let x₀ : ℝ := -2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y + 4 = 0) :=
by
  -- Introduce the variables
  intro x y
  -- Define x₀, y₀, and m
  let x₀ := -2
  let y₀ := f x₀
  let m := f' x₀
  
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_two_l629_62936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajeev_profit_share_l629_62902

/-- Represents the profit share of a partner in a business partnership. -/
structure ProfitShare where
  amount : ℚ

/-- Represents a ratio between two partners' shares. -/
structure Ratio where
  first : ℕ
  second : ℕ

/-- Calculates the profit share of a partner given the total profit and the overall ratio. -/
def calculateShare (totalProfit : ℚ) (overallRatio : List ℕ) (partnerIndex : Fin (overallRatio.length)) : ProfitShare :=
  let totalParts := overallRatio.sum
  let partValue := totalProfit / totalParts
  { amount := partValue * overallRatio[partnerIndex] }

/-- Theorem stating that Rajeev's share of the profit is 12000 given the specified conditions. -/
theorem rajeev_profit_share 
  (totalProfit : ℚ)
  (rameshXYZRatio : Ratio)
  (xYZRajeevRatio : Ratio)
  (h1 : totalProfit = 36000)
  (h2 : rameshXYZRatio = { first := 5, second := 4 })
  (h3 : xYZRajeevRatio = { first := 8, second := 9 }) :
  (calculateShare totalProfit [10, 8, 9] ⟨2, by simp⟩).amount = 12000 :=
by sorry

#eval (calculateShare 36000 [10, 8, 9] ⟨2, by simp⟩).amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajeev_profit_share_l629_62902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_hyperbola_or_ellipse_l629_62926

/-- Two circles with unequal radii -/
structure TwoCircles where
  O₁ : EuclideanSpace ℝ (Fin 2)
  O₂ : EuclideanSpace ℝ (Fin 2)
  r₁ : ℝ
  r₂ : ℝ
  h_neq : r₁ ≠ r₂
  h_no_common : ∀ p : EuclideanSpace ℝ (Fin 2), ¬(dist O₁ p = r₁ ∧ dist O₂ p = r₂)
  h_diff_centers : O₁ ≠ O₂

/-- A moving circle internally tangent to two fixed circles -/
structure MovingCircle (tc : TwoCircles) where
  O : EuclideanSpace ℝ (Fin 2)
  R : ℝ
  h_tangent₁ : dist tc.O₁ O = tc.r₁ - R
  h_tangent₂ : dist tc.O₂ O = tc.r₂ - R

/-- The trajectory of the center of the moving circle -/
def trajectory (tc : TwoCircles) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | ∃ mc : MovingCircle tc, mc.O = p}

/-- Definition of a hyperbola branch -/
def is_hyperbola_branch (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

/-- Definition of an ellipse -/
def is_ellipse (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

/-- The main theorem -/
theorem trajectory_is_hyperbola_or_ellipse (tc : TwoCircles) :
  is_hyperbola_branch (trajectory tc) ∨ is_ellipse (trajectory tc) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_hyperbola_or_ellipse_l629_62926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrent_lines_l629_62983

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the rectangle and its properties
def Rectangle (A B C D : Point) : Prop := sorry

-- Define midpoint
def Midpoint (M P Q : Point) : Prop := sorry

-- Define intersection of lines
def Intersection (P : Point) (L1 L2 : Line) : Prop := sorry

-- Define perpendicular line through a point
noncomputable def PerpendicularLine (L : Line) (P : Point) : Line := sorry

-- Define a membership relation for Point and Line
def PointOnLine (P : Point) (L : Line) : Prop := sorry

-- Define the problem setup
theorem concurrent_lines 
  (A B C D P Q K M N X Y Z : Point)
  (ℓ1 ℓ2 ℓ3 : Line) :
  Rectangle A B C D →
  Midpoint P B C →
  Midpoint Q C D →
  Intersection K (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- PD and QB
  Intersection M (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- PD and QA
  Intersection N (Line.mk 0 0 0) (Line.mk 0 0 0) →  -- PA and QB
  Midpoint X A N →
  Midpoint Y K N →
  Midpoint Z A M →
  ℓ1 = PerpendicularLine (Line.mk 0 0 0) X →  -- MK
  ℓ2 = PerpendicularLine (Line.mk 0 0 0) Y →  -- AM
  ℓ3 = PerpendicularLine (Line.mk 0 0 0) Z →  -- KN
  ∃ (P : Point), PointOnLine P ℓ1 ∧ PointOnLine P ℓ2 ∧ PointOnLine P ℓ3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrent_lines_l629_62983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_value_l629_62965

theorem max_integer_value (a b c d : ℕ+) : 
  (a + b + c + d : ℚ) / 4 = 50 →
  (a + b : ℚ) / 2 = 35 →
  c ≤ 129 ∧ d ≤ 129 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_value_l629_62965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_natural_numbers_sum_properties_l629_62950

-- Define Prob as a noncomputable function
noncomputable def Prob (p : Prop) : ℚ := sorry

theorem random_natural_numbers_sum_properties :
  ∃ (Prob : Prop → ℚ),
  ∀ a b : ℕ,
  -- Part (a)
  (Prob (Even (a + b)) = 1/2 ∧ Prob (Odd (a + b)) = 1/2) ∧
  -- Part (b)
  (Prob ((a + b) % 3 = 0) = 1/3 ∧
   Prob ((a + b) % 3 = 1) = 1/3 ∧
   Prob ((a + b) % 3 = 2) = 1/3) ∧
  -- Part (c)
  (Prob ((a + b) % 4 = 0) = 1/4 ∧
   Prob ((a + b) % 4 = 1) = 1/4 ∧
   Prob ((a + b) % 4 = 2) = 1/4 ∧
   Prob ((a + b) % 4 = 3) = 1/4)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_natural_numbers_sum_properties_l629_62950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l629_62957

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1 / (Real.log (2 * x) / Real.log 4)

-- State the theorem
theorem min_value_of_f :
  ∀ x > (1/2 : ℝ), f x ≥ 2 * Real.sqrt 2 - 1 ∧
  ∃ x > (1/2 : ℝ), f x = 2 * Real.sqrt 2 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l629_62957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_productivity_increase_l629_62914

/-- Represents Jane's toy bear production --/
structure BearProduction where
  bears : ℚ  -- Number of bears produced per week
  hours : ℚ  -- Number of hours worked per week

/-- Calculates the productivity (bears per hour) --/
def productivity (p : BearProduction) : ℚ := p.bears / p.hours

/-- Jane's production with first assistant --/
def withFirstAssistant (p : BearProduction) : BearProduction :=
  { bears := 18/10 * p.bears, hours := 9/10 * p.hours }

/-- Jane's production with second assistant --/
def withSecondAssistant (p : BearProduction) : BearProduction :=
  { bears := 16/10 * p.bears, hours := 8/10 * p.hours }

/-- Jane's production with both assistants --/
def withBothAssistants (p : BearProduction) : BearProduction :=
  { bears := 24/10 * p.bears, hours := 72/100 * p.hours }

/-- Theorem: The overall percentage increase in Jane's output with both assistants is 233.33% --/
theorem jane_productivity_increase (p : BearProduction) :
  (productivity (withBothAssistants p) / productivity p - 1) * 100 = 2400/1029 := by
  sorry

#eval (2400 : ℚ) / 1029  -- This will output the approximate decimal value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_productivity_increase_l629_62914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_properties_l629_62908

structure Club where
  members : Finset Nat
  friend : Nat → Nat
  enemy : Nat → Nat
  friend_injective : Function.Injective friend
  enemy_injective : Function.Injective enemy
  friend_enemy_distinct : ∀ m, m ∈ members → friend m ≠ enemy m
  friend_symmetric : ∀ m n, m ∈ members → n ∈ members → (friend m = n ↔ friend n = m)
  enemy_symmetric : ∀ m n, m ∈ members → n ∈ members → (enemy m = n ↔ enemy n = m)

theorem club_properties (c : Club) :
  (Even c.members.card) ∧
  (∃ (g1 g2 : Finset Nat), 
    c.members = g1 ∪ g2 ∧ 
    g1 ∩ g2 = ∅ ∧
    (∀ m n, m ∈ g1 → n ∈ g1 → c.friend m ≠ n ∧ c.enemy m ≠ n) ∧
    (∀ m n, m ∈ g2 → n ∈ g2 → c.friend m ≠ n ∧ c.enemy m ≠ n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_properties_l629_62908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l629_62920

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties (ω φ : ℝ) :
  ω > 0 ∧
  -Real.pi / 2 ≤ φ ∧ φ < Real.pi / 2 ∧
  (∀ x : ℝ, f ω φ (x - Real.pi / 3) = f ω φ (Real.pi / 3 - x)) ∧
  (∀ x : ℝ, f ω φ (x + Real.pi) = f ω φ x) →
  ω = 2 ∧
  φ = -Real.pi / 6 ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f ω φ x ≤ Real.sqrt 3) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f ω φ x ≥ -Real.sqrt 3 / 2) ∧
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
              x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
              f ω φ x₁ = Real.sqrt 3 ∧ 
              f ω φ x₂ = -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l629_62920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_value_exponential_equation_value_l629_62968

-- Part 1
theorem logarithm_expression_value : 
  (Real.log 3 / Real.log 4) * (Real.log 8 / (Real.log 9 + Real.log 16 / Real.log 27)) = 17/12 := by
  sorry

-- Part 2
theorem exponential_equation_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (3 : ℝ)^x = (4 : ℝ)^y ∧ (4 : ℝ)^y = (6 : ℝ)^z) : 
  y/z - y/x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_value_exponential_equation_value_l629_62968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l629_62930

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)*x^2 - 2*x + 5

theorem f_upper_bound (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) ↔ m > 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l629_62930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_square_swap_l629_62952

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_different (n : ℕ) : Prop :=
  let digits := List.map (fun i => (n / (10^i)) % 10) [0, 1, 2, 3, 4]
  List.Nodup digits

def swap_first_last (n : ℕ) : ℕ :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  let d := (n / 10) % 10
  let e := n % 10
  e * 10000 + b * 1000 + c * 100 + d * 10 + a

theorem five_digit_square_swap :
  ∃ (x y : ℕ),
    x < y ∧
    is_five_digit (x^2) ∧
    is_five_digit (y^2) ∧
    all_digits_different (x^2) ∧
    y^2 = swap_first_last (x^2) ∧
    x^2 = 41209 ∧
    y^2 = 91204 := by
  sorry

#eval swap_first_last 41209

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_square_swap_l629_62952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_is_6000_l629_62959

/-- Represents the profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  a_period : ℚ
  b_period : ℚ
  total_profit : ℚ

/-- Calculates B's profit in the partnership --/
def b_profit (p : Partnership) : ℚ :=
  (p.b_investment * p.b_period) / (p.a_investment * p.a_period + p.b_investment * p.b_period) * p.total_profit

/-- Theorem stating B's profit is 6000 given the partnership conditions --/
theorem b_profit_is_6000 (p : Partnership)
  (h1 : p.a_investment = 3 * p.b_investment)
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.total_profit = 42000) :
  b_profit p = 6000 := by
  sorry

#eval b_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, total_profit := 42000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_is_6000_l629_62959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_and_dot_product_l629_62964

noncomputable def P : ℝ × ℝ := (1/2, Real.sqrt 3/2)

noncomputable def rotate (v : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (v.1 * Real.cos θ - v.2 * Real.sin θ, v.1 * Real.sin θ + v.2 * Real.cos θ)

noncomputable def Q (x : ℝ) : ℝ × ℝ := rotate P x

noncomputable def f (x : ℝ) : ℝ := P.1 * (Q x).1 + P.2 * (Q x).2

noncomputable def g (x : ℝ) : ℝ := f x * f (x + Real.pi/3)

theorem rotation_and_dot_product :
  (Q (Real.pi/4) = ((Real.sqrt 2 - Real.sqrt 6)/4, (Real.sqrt 2 + Real.sqrt 6)/4)) ∧
  (Set.Icc (-1/4) (3/4) = Set.range g) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_and_dot_product_l629_62964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_translation_for_symmetry_l629_62973

noncomputable def f (x : ℝ) := Real.cos (2 * x) - Real.sin (2 * x)

noncomputable def translated_f (m : ℝ) (x : ℝ) := f (x + m)

def is_symmetric_about_origin (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem smallest_translation_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ 
    is_symmetric_about_origin (translated_f m) ∧
    ∀ m' : ℝ, m' > 0 → is_symmetric_about_origin (translated_f m') → m ≤ m' :=
by
  -- The proof would go here
  sorry

#check smallest_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_translation_for_symmetry_l629_62973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_sin_A_over_b_l629_62955

-- Define the triangle and its properties
def Triangle (A B C a b c : Real) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define an acute triangle
def AcuteTriangle (A B C : Real) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2

-- Theorem statement
theorem range_of_a_sin_A_over_b 
  (A B C a b c : Real) 
  (h1 : Triangle A B C a b c) 
  (h2 : AcuteTriangle A B C) 
  (h3 : B = 2 * A) : 
  ∃ (x : Real), x = a * Real.sin A / b ∧ Real.sqrt 3 / 6 < x ∧ x < 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_sin_A_over_b_l629_62955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_bounds_l629_62977

-- Define a circle with diameter d
variable (d : ℝ) (h : d > 0)

-- Define a point X inside or on the circumference of the circle
variable (X : ℝ × ℝ)

-- Define two mutually perpendicular chords AC and BD passing through X
variable (A B C D : ℝ × ℝ)

-- Define the sum S
def S (A B C D : ℝ × ℝ) : ℝ := dist A C + dist B D

-- State the theorem
theorem chord_sum_bounds :
  (dist X (0, 0) ≤ d / 2) →
  ((A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0) →
  d ≤ S A B C D ∧ S A B C D ≤ 2 * d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_bounds_l629_62977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_even_function_l629_62960

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of f being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem integral_even_function 
  (h_even : is_even f) 
  (h_integral : ∫ x in (0 : ℝ)..(6 : ℝ), f x = 8) : 
  ∫ x in (-6 : ℝ)..(6 : ℝ), f x = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_even_function_l629_62960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_difference_l629_62921

theorem angle_terminal_side_difference (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (1, a) ∧ B = (2, b) ∧ 
    (∀ t : ℝ, (t * A.1 + (1 - t) * 0, t * A.2 + (1 - t) * 0) ∈ Set.range (λ t : ℝ ↦ (t, 0))) ∧
    (∀ t : ℝ, (t * B.1 + (1 - t) * 0, t * B.2 + (1 - t) * 0) ∈ Set.range (λ t : ℝ ↦ (t, 0)))) →
  Real.cos (2 * Real.arccos 1) = 2/3 →
  |a - b| = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_difference_l629_62921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_l629_62900

def expression : ℕ := 7^7 - 7^4 + 2^2

theorem sum_of_distinct_prime_factors : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (expression + 1)))
    (λ p ↦ if p ∣ expression then p else 0)) = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_l629_62900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_range_l629_62904

-- Define the quadratic function f
noncomputable def f : ℝ → ℝ := sorry

-- State the conditions
axiom f_0 : f 0 = 2
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x - 1

-- State that f is quadratic
axiom f_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Theorem for the analytical expression of f
theorem f_expression : ∀ x, f x = x^2 - 2*x + 2 := by sorry

-- Theorem for the range of f(2^t) when t ∈ [-1, 3]
theorem f_range : 
  ∀ y : ℝ, (∃ t : ℝ, t ∈ Set.Icc (-1) 3 ∧ y = f (2^t)) ↔ y ∈ Set.Icc 1 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_range_l629_62904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cap_distance_difference_l629_62905

-- Define the initial distances and bounce factors
noncomputable def jenny_initial_distance : ℝ := 18
noncomputable def jenny_bounce_factor : ℝ := 1/3
noncomputable def mark_initial_distance : ℝ := 15
noncomputable def mark_bounce_factor : ℝ := 2

-- Define the total distances
noncomputable def jenny_total_distance : ℝ := jenny_initial_distance + jenny_initial_distance * jenny_bounce_factor
noncomputable def mark_total_distance : ℝ := mark_initial_distance + mark_initial_distance * mark_bounce_factor

-- Theorem statement
theorem bottle_cap_distance_difference :
  mark_total_distance - jenny_total_distance = 21 := by
  -- Unfold definitions
  unfold mark_total_distance jenny_total_distance
  unfold mark_initial_distance jenny_initial_distance
  unfold mark_bounce_factor jenny_bounce_factor
  -- Simplify the expression
  simp
  -- The proof is completed by computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cap_distance_difference_l629_62905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_range_f_monotonic_increase_l629_62971

/-- The function f(x) = √3 * cos(x) + sin(x) + 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x + 1

/-- The smallest positive period of f(x) is 2π -/
theorem f_period : ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) := by
  sorry

/-- The range of f(x) is [-1, 3] -/
theorem f_range : Set.range f = Set.Icc (-1) 3 := by
  sorry

/-- The intervals of monotonic increase of f(x) are [2kπ - π/3, 2kπ + π/6] for k ∈ ℤ -/
theorem f_monotonic_increase (k : ℤ) :
  StrictMonoOn f (Set.Icc (2 * k * Real.pi - Real.pi / 3) (2 * k * Real.pi + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_range_f_monotonic_increase_l629_62971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_theft_percentage_l629_62942

/-- Represents the percentage of profit a shopkeeper takes on goods -/
noncomputable def profit_percent : ℚ := 10

/-- Represents the percentage of loss experienced by the shopkeeper -/
noncomputable def loss_percent : ℚ := 67

/-- Represents the cost price of goods (assumed to be 100 for simplicity) -/
noncomputable def cost_price : ℚ := 100

/-- Calculates the selling price based on the cost price and profit percentage -/
noncomputable def selling_price : ℚ := cost_price * (1 + profit_percent / 100)

/-- Represents the percentage of goods lost during theft -/
noncomputable def theft_percent : ℚ := 70

theorem shopkeeper_theft_percentage :
  selling_price * (1 - theft_percent / 100) = cost_price * (1 - loss_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_theft_percentage_l629_62942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_einstein_born_on_friday_l629_62997

/-- Represents a date with year, month, and day -/
structure Date where
  year : Int
  month : Int
  day : Int

/-- Represents a day of the week as an integer from 1 to 7 -/
def DayOfWeek := Fin 7

/-- Einstein's birth date -/
def einstein_birth : Date := { year := 1879, month := 3, day := 14 }

/-- Reference date (31 May 2006, known to be a Wednesday) -/
def reference_date : Date := { year := 2006, month := 5, day := 31 }

/-- Function to calculate the day of the week for a given date -/
noncomputable def calculate_day_of_week (date : Date) : DayOfWeek :=
  sorry -- Implementation details omitted

/-- Theorem stating that Einstein was born on a Friday (day 5) -/
theorem einstein_born_on_friday :
  calculate_day_of_week einstein_birth = ⟨5, by norm_num⟩ := by
  sorry

#check einstein_born_on_friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_einstein_born_on_friday_l629_62997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l629_62915

theorem rectangular_to_polar :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r * Real.cos θ = Real.sqrt 2 ∧ r * Real.sin θ = -Real.sqrt 2 ∧
  r = 2 ∧ θ = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l629_62915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FMN_MN_passes_through_fixed_point_l629_62932

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define a chord of the parabola
def chord (p k b : ℝ) (x y : ℝ) : Prop := y = k*x + b ∧ parabola p x y

-- Define the midpoint of a chord
def chord_midpoint (p k b : ℝ) : ℝ × ℝ := (k, k^2 + b)

-- Part 1
theorem area_of_triangle_FMN (p : ℝ) (h : p > 0) :
  let k₁ : ℝ := 1
  let k₂ : ℝ := -1
  let M := chord_midpoint p k₁ (1/2)
  let N := chord_midpoint p k₂ (1/2)
  let F : ℝ × ℝ := (0, p/2)
  ∃ (area : ℝ), area = 1 :=
sorry

-- Part 2
theorem MN_passes_through_fixed_point (p : ℝ) (h : p > 0) (k₁ k₂ : ℝ) 
  (h₁ : k₁ ≠ 0) (h₂ : k₂ ≠ 0) (h₃ : 1/k₁ + 1/k₂ = 1) :
  let M := chord_midpoint p k₁ (p/2)
  let N := chord_midpoint p k₂ (p/2)
  ∃ (line : Set (ℝ × ℝ)), (1, 1/2) ∈ line ∧ M ∈ line ∧ N ∈ line :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FMN_MN_passes_through_fixed_point_l629_62932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_weight_correlation_l629_62979

/-- The correlation index between height and weight -/
def correlation_index : ℝ := 0.64

/-- The percentage of weight variation explained by height -/
def height_explanation : ℝ := 0.64

/-- The percentage of weight variation contributed by random errors -/
def random_error_contribution : ℝ := 0.36

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) : Prop := abs (x - y) < 0.01

theorem height_weight_correlation :
  height_explanation + random_error_contribution = 1 ∧
  approx_equal correlation_index height_explanation :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_weight_correlation_l629_62979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l629_62972

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_focus_distance 
  (M : ℝ × ℝ) 
  (h1 : parabola M.1 M.2) 
  (h2 : (distance M origin)^2 = 3 * (distance M focus)) : 
  distance M focus = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l629_62972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_xy_l629_62918

theorem existence_of_xy (a b : ℝ) : ∃ x y : ℝ, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ |x * y - a * x - b * y| ≥ (1 : ℝ) / 3 := by
  sorry

#check existence_of_xy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_xy_l629_62918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l629_62995

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- Theorem: If a hyperbola has an asymptote with equation x + y = 0, its eccentricity is √2 -/
theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola) 
  (asymptote : ∀ (x y : ℝ), x + y = 0 → (x^2 / h.a^2 - y^2 / h.b^2 = 1)) :
  eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l629_62995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l629_62985

-- Define the slope of a line ax + by + c = 0
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

-- Define the condition for two lines to be parallel
def are_parallel (a1 b1 a2 b2 : ℝ) : Prop :=
  line_slope a1 b1 = line_slope a2 b2

-- Define the condition for two lines to be perpendicular
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  line_slope a1 b1 * line_slope a2 b2 = -1

theorem line_relations (a : ℝ) :
  (are_parallel (a - 2) 3 a (a - 2) ↔ a = (7 + Real.sqrt 33) / 2 ∨ a = (7 - Real.sqrt 33) / 2) ∧
  (are_perpendicular (a - 2) 3 a (a - 2) ↔ a = 2 ∨ a = -3) := by
  sorry

#check line_relations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l629_62985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_e_over_4_l629_62912

-- Define the function f(x) = x * e^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the point of tangency
noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

-- Define the tangent line at P
noncomputable def tangent_line (x : ℝ) : ℝ := 2 * Real.exp 1 * x - Real.exp 1

-- Define the x-intercept of the tangent line
noncomputable def x_intercept : ℝ := 1 / 2

-- Define the y-intercept of the tangent line
noncomputable def y_intercept : ℝ := -Real.exp 1

-- State the theorem
theorem triangle_area_is_e_over_4 :
  (1/2 * x_intercept * (-y_intercept)) = Real.exp 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_e_over_4_l629_62912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l629_62970

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 1

theorem min_positive_period_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l629_62970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l629_62987

theorem equation_solution : ∃ x : ℝ, (10 : ℝ) ^ x * (100 : ℝ) ^ (3 * x) = (1000 : ℝ) ^ 6 ∧ x = 18 / 7 := by
  use 18 / 7
  apply And.intro
  · -- Prove the equation
    sorry
  · -- Prove x = 18 / 7
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l629_62987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eat_six_jars_together_l629_62909

/-- Represents the time (in minutes) it takes to eat jars of jam and honey -/
structure EatingTime where
  jam : ℚ
  honey : ℚ

/-- Calculate the rate of eating jam in jars per minute -/
def jamRate (time : EatingTime) : ℚ :=
  1 / time.jam

/-- Karlsson's eating time for jam and honey -/
def karlsson : EatingTime :=
  { jam := 5, honey := 10 }

/-- Little Brother's eating time for jam and honey -/
def littleBrother : EatingTime :=
  { jam := 10, honey := 25 }

/-- The main theorem stating the time it takes to eat 6 jars of jam together -/
theorem eat_six_jars_together (karlsson littleBrother : EatingTime) 
  (h1 : 3 * karlsson.jam + karlsson.honey = 25)
  (h2 : 3 * littleBrother.jam + littleBrother.honey = 55)
  (h3 : karlsson.jam + 3 * karlsson.honey = 35)
  (h4 : littleBrother.jam + 3 * littleBrother.honey = 85) :
  (6 : ℚ) / (jamRate karlsson + jamRate littleBrother) = 20 := by
  sorry

-- Remove the #eval statement as it's not necessary for building
-- and might cause issues if the theorem is not fully proved

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eat_six_jars_together_l629_62909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_reflections_l629_62925

/-- The expected number of reflections for a billiard ball on a rectangular table -/
noncomputable def expected_reflections (table_length : ℝ) (table_width : ℝ) (travel_distance : ℝ) : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1/4) - Real.arcsin (3/4) + Real.arccos (3/4))

/-- Theorem stating the expected number of reflections for the given billiard problem -/
theorem billiard_reflections :
  let table_length : ℝ := 3
  let table_width : ℝ := 1
  let travel_distance : ℝ := 2
  let ball_start : ℝ × ℝ := (table_length / 2, table_width / 2)
  expected_reflections table_length table_width travel_distance =
    (2 / Real.pi) * (3 * Real.arccos (1/4) - Real.arcsin (3/4) + Real.arccos (3/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billiard_reflections_l629_62925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l629_62991

-- Define the points in polar coordinates
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (2, Real.pi / 2)
noncomputable def B : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)

-- Define the parametric equation of circle C2
noncomputable def C2 (a : ℝ) (θ : ℝ) : ℝ × ℝ := (-1 + a * Real.cos θ, -1 + a * Real.sin θ)

-- Define the polar equation of circle C1
def C1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

-- Define the external tangency condition
def externally_tangent (a : ℝ) : Prop := Real.sqrt 2 + abs a = 2 * Real.sqrt 2

-- Main theorem
theorem circle_tangency (a : ℝ) : 
  externally_tangent a → 
  (C1 (2 * Real.sqrt 2) (Real.pi / 4) ∧ (a = Real.sqrt 2 ∨ a = -Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l629_62991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l629_62996

-- Define the hyperbolas C1 and C2
noncomputable def C1 (k : ℝ) (x y : ℝ) : Prop := x^2 / 4 - y^2 / k = 1
noncomputable def C2 (k : ℝ) (x y : ℝ) : Prop := x^2 / k - y^2 / 9 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Define the condition that C1 and C2 have the same eccentricity
noncomputable def same_eccentricity (k : ℝ) : Prop :=
  eccentricity 2 (Real.sqrt k) = eccentricity (Real.sqrt k) 3

-- Define the asymptote equation
def asymptote (m : ℝ) (x y : ℝ) : Prop := y = m * x ∨ y = -m * x

-- Theorem statement
theorem hyperbola_asymptotes (k : ℝ) :
  same_eccentricity k →
  ∃ (x y : ℝ), C1 k x y ∧ asymptote (Real.sqrt 6 / 2) x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l629_62996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_alloy_force_l629_62975

/-- Represents an alloy of two metals -/
structure Alloy where
  mass : ℚ
  force : ℚ
  ratio : ℚ  -- ratio of first metal to second metal

/-- Calculates the force exerted by the combined alloy -/
def combinedForce (a b : Alloy) : ℚ :=
  a.force + b.force

theorem combined_alloy_force (a b : Alloy) :
  a.mass = 6 ∧ a.ratio = 2 ∧ a.force = 30 ∧
  b.mass = 3 ∧ b.ratio = 1/5 ∧ b.force = 10 →
  combinedForce a b = 40 := by
  intro h
  simp [combinedForce]
  rw [h.2.2.1, h.2.2.2.2.2]
  norm_num

#eval combinedForce { mass := 6, force := 30, ratio := 2 }
                     { mass := 3, force := 10, ratio := 1/5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_alloy_force_l629_62975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_frac_zeta_even_l629_62963

-- Define the Riemann zeta function
noncomputable def riemann_zeta (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / n^x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem sum_frac_zeta_even : ∑' k : ℕ, frac (riemann_zeta (2 * ↑k)) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_frac_zeta_even_l629_62963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zongzi_problem_l629_62948

-- Define the variables
variable (x y : ℚ)  -- Prices of zongzi A and B
variable (a : ℕ)    -- Number of zongzi A

-- Define the conditions
def condition1 (x y : ℚ) : Prop := 4 * x + 5 * y = 35
def condition2 (x y : ℚ) : Prop := 2 * x + 3 * y = 19
def condition3 (a : ℕ) : Prop := 5 * a + 3 * (1000 - a) ≤ 4000

-- State the theorem
theorem zongzi_problem :
  (∃ x y : ℚ, condition1 x y ∧ condition2 x y ∧ x = 5 ∧ y = 3) ∧ 
  (∀ a : ℕ, condition3 a → a ≤ 500) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zongzi_problem_l629_62948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_minus_2cos_squared_theta_l629_62956

theorem sin_2theta_minus_2cos_squared_theta (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_minus_2cos_squared_theta_l629_62956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_24_and_1080_l629_62949

theorem smallest_n_divisible_by_24_and_1080 :
  ∀ n : ℕ, n > 0 → n^2 % 24 = 0 ∧ n^3 % 1080 = 0 → n ≥ 120 :=
by
  intro n h_pos h_div
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_24_and_1080_l629_62949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l629_62988

theorem calculate_expressions : 
  ((Real.sqrt 8 - Real.sqrt 24) / Real.sqrt 2 + |1 - Real.sqrt 3| = 1 - Real.sqrt 3) ∧
  ((1/2)⁻¹ - 2 * Real.cos (π/6) + |2 - Real.sqrt 3| - (2 * Real.sqrt 2 + 1)^0 = 3 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l629_62988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l629_62929

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (2 * x + b)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) : 
  (∀ x, f b x ≠ 0 → f_inv (f b x) = x) → b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l629_62929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l629_62938

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 240 ∧ 
  platform_length = 240 ∧ 
  train_speed_kmh = 64 → 
  ∃ (time : ℝ), (time ≥ 26.9 ∧ time ≤ 27.1) ∧ 
    time = (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l629_62938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_two_planes_implies_parallel_perpendicular_to_same_plane_implies_parallel_not_always_parallel_to_two_planes_implies_parallel_not_always_parallel_to_same_plane_implies_parallel_l629_62911

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_to_two_planes_implies_parallel 
  (m : Line) (α β : Plane) : 
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

-- Theorem 2
theorem perpendicular_to_same_plane_implies_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel_lines m n :=
sorry

-- Theorem 3
theorem not_always_parallel_to_two_planes_implies_parallel :
  ¬ (∀ (m : Line) (α β : Plane), 
    (parallel_line_plane m α → parallel_line_plane m β → parallel_planes α β)) :=
sorry

-- Theorem 4
theorem not_always_parallel_to_same_plane_implies_parallel :
  ¬ (∀ (m n : Line) (α : Plane), 
    (parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_two_planes_implies_parallel_perpendicular_to_same_plane_implies_parallel_not_always_parallel_to_two_planes_implies_parallel_not_always_parallel_to_same_plane_implies_parallel_l629_62911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_right_angles_equal_diagonals_pentagon_right_angle_unequal_diagonals_possible_l629_62907

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a pentagon
structure Pentagon :=
  (vertices : Fin 5 → ℝ × ℝ)

-- Define right angle
def is_right_angle (a b c : ℝ × ℝ) : Prop :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let bc := (c.1 - b.1, c.2 - b.2)
  ab.1 * bc.1 + ab.2 * bc.2 = 0

-- Define diagonal length
noncomputable def diagonal_length (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Theorem for quadrilaterals
theorem quadrilateral_right_angles_equal_diagonals (q : Quadrilateral) :
  (∀ i : Fin 4, is_right_angle (q.vertices i) (q.vertices ((i + 1) % 4)) (q.vertices ((i + 2) % 4))) →
  diagonal_length (q.vertices 0) (q.vertices 2) = diagonal_length (q.vertices 1) (q.vertices 3) := by
  sorry

-- Theorem for pentagons
theorem pentagon_right_angle_unequal_diagonals_possible (p : Pentagon) :
  (∃ i : Fin 5, is_right_angle (p.vertices i) (p.vertices ((i + 1) % 5)) (p.vertices ((i + 2) % 5))) →
  ¬ (diagonal_length (p.vertices 0) (p.vertices 2) = diagonal_length (p.vertices 1) (p.vertices 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_right_angles_equal_diagonals_pentagon_right_angle_unequal_diagonals_possible_l629_62907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt2_approaches_1_l629_62984

/-- A sequence representing nested square roots of 2 -/
noncomputable def NestedSqrt2 : ℕ → ℝ
  | 0 => Real.sqrt 2
  | n + 1 => Real.sqrt (2 - NestedSqrt2 n)

/-- The set of all numbers representable as nested square roots of 2 -/
def NestedSqrt2Set : Set ℝ :=
  {x : ℝ | ∃ n : ℕ, x = NestedSqrt2 n}

/-- Theorem stating that numbers of the form √(2 ± √(2 ± ⋯ ± √2)) can be arbitrarily close to 1 -/
theorem nested_sqrt2_approaches_1 :
  ∀ ε > 0, ∃ x ∈ NestedSqrt2Set, |x - 1| < ε := by
  sorry

#check nested_sqrt2_approaches_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt2_approaches_1_l629_62984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_sum_l629_62944

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def sum_of_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_geometric_sequence_sum 
  (a b : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n, S n = sum_of_arithmetic a n) →
  b 1 = 2 →
  (∃ q : ℝ, q > 0 ∧ ∀ n, b (n + 1) = b n * q) →
  b 2 + b 3 = 12 →
  b 3 = a 4 - 2 * a 1 →
  S 11 = 11 * b 4 →
  (∀ n, a n = 3 * n - 2) ∧
  (∀ n, b n = 2^n) ∧
  (∀ n, sum_of_arithmetic (λ k ↦ a (2*k) * b k) n = (3*n - 4) * 2^(n+2) + 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_sum_l629_62944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_of_two_circles_l629_62976

/-- Two circles configuration -/
structure TwoCircles where
  largeArea : ℝ
  touchInternally : Prop
  smallerCenterOnLargeCircumference : Prop

/-- The total shaded area of two circles -/
noncomputable def totalShadedArea (c : TwoCircles) : ℝ :=
  (2 / 3) * c.largeArea + (2 / 3) * (c.largeArea / 4)

/-- Theorem stating the total shaded area for the given configuration -/
theorem total_shaded_area_of_two_circles (c : TwoCircles)
    (h1 : c.largeArea = 100 * Real.pi)
    (h2 : c.touchInternally)
    (h3 : c.smallerCenterOnLargeCircumference) :
    totalShadedArea c = 250 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_of_two_circles_l629_62976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_expected_l629_62927

noncomputable section

-- Define the dimensions of the parallelepiped
def box_length : ℝ := 2
def box_width : ℝ := 3
def box_height : ℝ := 6

-- Define the dimensions of the cylinder
def cylinder_radius : ℝ := 2
def cylinder_height : ℝ := 3

-- Define the extension distance
def extension_distance : ℝ := 2

-- Define the volume function
noncomputable def volume_of_shape : ℝ :=
  let box_volume := box_length * box_width * box_height
  let extended_box_volume := 
    box_volume + 
    2 * (box_length * box_width * extension_distance) +
    2 * (box_length * box_height * extension_distance) +
    2 * (box_width * box_height * extension_distance)
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  let sphere_corners_volume := 8 * ((4/3) * Real.pi * extension_distance^3 / 8)
  extended_box_volume + cylinder_volume + sphere_corners_volume

-- Theorem statement
theorem volume_equals_expected : volume_of_shape = (540 + 48 * Real.pi) / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_expected_l629_62927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_zeros_prob_product_four_l629_62993

/-- Represents the content of a bag of cards -/
structure Bag where
  zeros : Nat
  ones : Nat
  twos : Nat

/-- The probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  ↑favorable / ↑total

/-- The contents of bag A -/
def bagA : Bag := { zeros := 1, ones := 2, twos := 3 }

/-- The contents of bag B -/
def bagB : Bag := { zeros := 4, ones := 1, twos := 2 }

/-- The number of cards drawn from bag A -/
def drawA : Nat := 1

/-- The number of cards drawn from bag B -/
def drawB : Nat := 2

/-- Theorem stating the probability of drawing all 0 cards -/
theorem prob_all_zeros (bagA bagB : Bag) (drawA drawB : Nat) :
  probability 6 126 = (1 : ℚ) / 21 := by sorry

/-- Theorem stating the probability of drawing cards with a product of 4 -/
theorem prob_product_four (bagA bagB : Bag) (drawA drawB : Nat) :
  probability 27 126 = (4 : ℚ) / 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_zeros_prob_product_four_l629_62993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_repeating_decimal_l629_62967

def CircleDigits : List ℕ := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def IsValidDecimal (n : ℚ) : Prop :=
  ∃ (start : ℕ) (length : ℕ),
    length > 0 ∧
    length ≤ CircleDigits.length ∧
    (∀ i, (n * 10^(i + 1)).num % (10^length * (n * 10^(i + 1)).den) =
      ((CircleDigits.rotate start).take length).getD (i % length) 0 * 10^(length - 1 - (i % length)) * (n * 10^(i + 1)).den)

noncomputable def LargestDecimal : ℚ := 9 + 579139 / 1000000

theorem largest_repeating_decimal :
  IsValidDecimal LargestDecimal ∧
  ∀ n, IsValidDecimal n → n ≤ LargestDecimal := by
  sorry

#check largest_repeating_decimal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_repeating_decimal_l629_62967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l629_62998

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  BC : Real
  AC : Real

-- Define the area function for a triangle
noncomputable def Triangle.area (t : Triangle) : Real :=
  1 / 2 * t.BC * t.AC * Real.sin t.A

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : 2 * Real.sin (abc.B + abc.C) ^ 2 = Real.sqrt 3 * Real.sin (2 * abc.A))
  (h2 : abc.BC = 7)
  (h3 : abc.AC = 5) :
  abc.A = Real.pi / 3 ∧ 
  Triangle.area abc = 10 * Real.sqrt 3 := by
  sorry

-- Note: Real.pi / 3 is equivalent to 60 degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l629_62998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_isosceles_triangle_l629_62913

noncomputable def C₁ (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

def P : ℝ × ℝ := (-2, 1)

theorem ellipse_and_isosceles_triangle
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 3 / 2)
  (h4 : C₁ a b P.1 P.2) :
  (∀ x y : ℝ, C₁ a b x y ↔ x^2 / 8 + y^2 / 2 = 1) ∧
  (∀ A B Q C D E : ℝ × ℝ,
    A = (-2, -1) → B = (2, 1) → Q = (2, -1) →
    C₁ a b C.1 C.2 → C₁ a b D.1 D.2 →
    ∃ k t : ℝ, k = (B.2 - A.2) / (B.1 - A.1) ∧
              C.2 = k * C.1 + t ∧ D.2 = k * D.1 + t →
    E = (-C.1, -C.2) →
    (D.2 - P.2) / (D.1 - P.1) = -(E.2 - P.2) / (E.1 - P.1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_isosceles_triangle_l629_62913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l629_62923

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2)^3
def g (x : ℝ) : ℝ := 4*x - 8

-- Define the area function
noncomputable def area_between_curves (a b : ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

-- Theorem statement
theorem area_bounded_by_curves :
  area_between_curves 0 4 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l629_62923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_descending_digit_numbers_eq_84_l629_62999

/-- A function that counts the number of three-digit numbers where each digit is greater than the digit to its right -/
def count_descending_digit_numbers : ℕ :=
  Finset.sum (Finset.range 8) (fun k => Finset.sum (Finset.range (k + 1)) (fun j => j))

/-- Theorem stating that the count of three-digit numbers where each digit is greater than the digit to its right is 84 -/
theorem count_descending_digit_numbers_eq_84 :
  count_descending_digit_numbers = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_descending_digit_numbers_eq_84_l629_62999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_product_distances_M_to_AB_l629_62928

-- Define the curves C₁ and C₂
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (1 / Real.tan φ, 1 / (Real.tan φ)^2)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 1 / (Real.cos θ + Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define point M
def M : ℝ × ℝ := (-1, 2)

-- Statement for the distance between A and B
theorem distance_AB : Real.sqrt 10 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) := by sorry

-- Statement for the product of distances from M to A and B
theorem product_distances_M_to_AB : 
  2 = Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) * Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_product_distances_M_to_AB_l629_62928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_is_maximum_l629_62978

noncomputable def a (n : ℕ) : ℝ := 8/3 * (1/8)^n - 3 * (1/4)^n + (1/2)^n

theorem a2_is_maximum : ∀ (n : ℕ), a n ≤ a 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_is_maximum_l629_62978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l629_62981

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - 1

-- State the theorem
theorem a_value (a : ℝ) : 
  a ∈ Set.Ici (0 : ℝ) → -- a is in [0, +∞)
  f a = 3 →             -- f(a) = 3
  a = 2 := by           -- Then a = 2
  intro h1 h2
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l629_62981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_binomial_l629_62947

/-- A random variable representing the number of successes in 5 independent trials -/
def ξ : ℕ → ℝ := sorry

/-- The probability of success in each trial -/
noncomputable def p : ℝ := 1/3

/-- The number of trials -/
def n : ℕ := 5

/-- The expected value of ξ -/
noncomputable def E_ξ : ℝ := n * p

theorem expected_value_binomial :
  E_ξ = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_binomial_l629_62947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h2so4_moles_formed_l629_62982

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String

-- Define the reaction equation
structure ReactionEquation where
  reactant1 : ChemicalSpecies
  reactant2 : ChemicalSpecies
  product : ChemicalSpecies

-- Define the moles of each species
def moles_of (species : ChemicalSpecies) : ℚ := 0 -- Default value, to be replaced with actual implementation

-- Define the reaction equation for SO2 + H2O2 → H2SO4
def so2_h2o2_reaction : ReactionEquation :=
  { reactant1 := { name := "SO2" },
    reactant2 := { name := "H2O2" },
    product := { name := "H2SO4" } }

-- Theorem statement
theorem h2so4_moles_formed
  (h1 : moles_of so2_h2o2_reaction.reactant1 = 1)
  (h2 : moles_of so2_h2o2_reaction.reactant2 = 1) :
  moles_of so2_h2o2_reaction.product = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h2so4_moles_formed_l629_62982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_reals_l629_62946

/-- The function f(x) = log(ax^2 + ax + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + a * x + 1)

/-- The theorem stating the range of a for which f is defined on all real numbers -/
theorem f_defined_on_reals (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_reals_l629_62946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_possible_under_srs_l629_62962

/-- Represents the composition of a class -/
structure ClassComposition where
  total_students : ℕ
  boys : ℕ
  girls : ℕ

/-- Represents a sample drawn from the class -/
structure Sample where
  size : ℕ
  boys : ℕ
  girls : ℕ

/-- The probability of drawing a specific sample under simple random sampling -/
def sampleProbability (c : ClassComposition) (s : Sample) : ℚ :=
  (Nat.choose c.boys s.boys * Nat.choose c.girls s.girls) / Nat.choose c.total_students s.size

/-- Theorem stating that the given sample is possible under simple random sampling -/
theorem sample_possible_under_srs (c : ClassComposition) (s : Sample) 
    (h_class : c.total_students = 50 ∧ c.boys = 20 ∧ c.girls = 30)
    (h_sample : s.size = 10 ∧ s.boys = 4 ∧ s.girls = 6) :
    sampleProbability c s > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_possible_under_srs_l629_62962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_root_combined_with_sqrt_12_l629_62974

theorem simplest_quadratic_root_combined_with_sqrt_12 (a : ℝ) :
  (∃ (b : ℝ), Real.sqrt (a + 2) = b * Real.sqrt 3) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_root_combined_with_sqrt_12_l629_62974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_main_theorem_l629_62906

/-- The radius of each small circle -/
noncomputable def small_radius : ℝ := 4

/-- The number of small circles -/
def num_circles : ℕ := 8

/-- The diameter of the large circle -/
noncomputable def large_diameter : ℝ := 16 / Real.sin (Real.pi / num_circles)

/-- Theorem stating the relationship between the small circles and the large circle -/
theorem large_circle_diameter :
  ∀ (small_r : ℝ) (n : ℕ),
  small_r > 0 → n > 2 →
  let large_d := 2 * small_r / Real.sin (Real.pi / n)
  large_d = 2 * (n * small_r) / (2 * Real.sin (Real.pi / n)) := by
  sorry

/-- The main theorem proving the diameter of the large circle -/
theorem main_theorem : large_diameter = 16 / Real.sin (Real.pi / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_main_theorem_l629_62906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_lines_l629_62992

/-- The slope of the angle bisector of the acute angle formed by two lines -/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ + Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

/-- Theorem: The slope of the angle bisector of the acute angle formed by y = 2x and y = 4x is (√21 - 6) / 7 -/
theorem angle_bisector_slope_specific_lines :
  angle_bisector_slope 2 4 = (Real.sqrt 21 - 6) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_specific_lines_l629_62992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l629_62969

noncomputable section

-- Define the ellipse parameters
def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt 2

-- Define the eccentricity
noncomputable def e : ℝ := Real.sqrt 2 / 2

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line equation
def line_eq (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the area of triangle AMN
noncomputable def triangle_area (k : ℝ) : ℝ := 4 * Real.sqrt 2 / 5

theorem ellipse_properties :
  (a > b ∧ b > 0) →
  (a = 2) →
  (e = Real.sqrt 2 / 2) →
  (c^2 = a^2 - b^2) →
  (∀ x y, ellipse_eq x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∃ k, ∀ x y, line_eq k x y → ellipse_eq x y → triangle_area k = 4 * Real.sqrt 2 / 5 → k = Real.sqrt 2 ∨ k = -Real.sqrt 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l629_62969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_volume_is_18_l629_62933

/-- The function defining the inequality constraint -/
def g (x y z : ℝ) : ℝ :=
  2 * abs (x + y + z) + 2 * abs (x + y - z) + 2 * abs (x - y + z) + 2 * abs (-x + y + z)

/-- The region defined by the inequality constraint -/
def region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | g p.1 p.2.1 p.2.2 ≤ 12}

/-- The volume of the region -/
noncomputable def regionVolume : ℝ := sorry

/-- Theorem stating that the volume of the region is 18 cubic units -/
theorem region_volume_is_18 : regionVolume = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_volume_is_18_l629_62933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l629_62937

noncomputable def z (m : ℝ) : ℂ := m^2 * (1 / (m + 5) + Complex.I) + (8*m + 15) * Complex.I + (m - 6) / (m + 5)

theorem complex_number_properties (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 ∧
  ((z m).re = 0 ↔ m = 2) ∧
  ((z m).im = 0 ↔ m = -3) ∧
  (∀ w : ℂ, w ≠ z m ↔ m ≠ -3 ∧ m ≠ -5) ∧
  ((z m).re < 0 ∧ (z m).im > 0 ↔ m < -5 ∨ (-3 < m ∧ m < 2)) :=
by
  sorry

#check complex_number_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l629_62937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_and_range_b_l629_62941

noncomputable def f (x a : ℝ) := |x - 3/2| - a

def M (a : ℝ) := {x : ℝ | f x a < 0}

theorem max_a_and_range_b :
  (∃ (a : ℝ), (1/2 : ℝ) ∈ M a ∧ (-1/2 : ℝ) ∉ M a ∧
    ∀ (a' : ℝ), (1/2 : ℝ) ∈ M a' ∧ (-1/2 : ℝ) ∉ M a' → a' ≤ a) ∧
  (∀ (a : ℕ+) (b : ℝ), (∃ (x : ℝ), |x - (a : ℝ)| - |x - 3| > b) ↔ b < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_and_range_b_l629_62941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l629_62939

/-- Calculates the compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (compounds_per_year : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- The difference in compound interest between monthly and yearly compounding --/
theorem compound_interest_difference : 
  let principal := (30000 : ℝ)
  let rate := (0.05 : ℝ)
  let time := (3 : ℝ)
  let monthly_compounds := (12 : ℝ)
  let yearly_compounds := (1 : ℝ)
  ‖compound_interest principal rate time monthly_compounds - 
   compound_interest principal rate time yearly_compounds - 121.56‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_difference_l629_62939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_main_theorem_l629_62990

open Real

-- Define the equation
def equation (x : ℝ) : Prop :=
  tan (6 * x) = (cos x - sin x) / (cos x + sin x)

-- State the theorem
theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < π/2 ∧ equation x ∧ 
  ∀ (y : ℝ), y > 0 → y < π/2 → equation y → x ≤ y :=
by sorry

-- Define the result
noncomputable def result : ℝ := 45 * (π / 180) / 7

-- State the main theorem
theorem main_theorem :
  ∃ (x : ℝ), x > 0 ∧ x < π/2 ∧ equation x ∧ 
  ∀ (y : ℝ), y > 0 → y < π/2 → equation y → x ≤ y ∧
  x = result :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_main_theorem_l629_62990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l629_62986

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, f (f (f m) + 2 * f (f n)) = m^2 + 2 * n^2) :
  ∀ n : ℕ+, f n = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l629_62986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l629_62903

/-- Calculates the total distance traveled given the total time and average speed -/
noncomputable def total_distance (total_time_minutes : ℝ) (average_speed_mph : ℝ) : ℝ :=
  (total_time_minutes / 60) * average_speed_mph

/-- Proves that a round trip with given parameters results in a total distance of 2 miles -/
theorem round_trip_distance 
  (time_to_friend : ℝ) 
  (time_from_friend : ℝ) 
  (average_speed : ℝ) 
  (h1 : time_to_friend = 8)
  (h2 : time_from_friend = 22)
  (h3 : average_speed = 4) :
  total_distance (time_to_friend + time_from_friend) average_speed = 2 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l629_62903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l629_62935

/-- A function f(x) that depends on a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + (1/2) * x^2 - a * x

/-- The theorem stating the condition for f to be monotonically decreasing -/
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∀ y : ℝ, y > x → f a y ≤ f a x) ↔ a ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l629_62935
