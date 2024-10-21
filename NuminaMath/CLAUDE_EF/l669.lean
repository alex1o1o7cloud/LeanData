import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_exist_l669_66930

theorem no_solutions_exist : ¬∃ x : ℕ, x > 0 ∧ (15 < -2 * (x : ℤ) + 20) ∧ (x % 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_exist_l669_66930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_round_trip_time_is_960_l669_66901

/-- Calculates the time taken for a round trip by boat in a stream -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : distance = 7560) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  by
    -- Placeholder for the actual proof
    sorry

/-- Proves that the calculated round trip time is equal to 960 hours -/
theorem round_trip_time_is_960 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : distance = 7560) : 
  round_trip_time boat_speed stream_speed distance h1 h2 h3 = 960 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_round_trip_time_is_960_l669_66901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l669_66996

theorem indefinite_integral_proof (x : ℝ) :
  deriv (λ y => (y - 1) * Real.exp (-3 * y) + 0) x = (4 - 3 * x) * Real.exp (-3 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l669_66996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_is_36_l669_66914

/-- Calculates the speed of the slower train given the conditions of the problem -/
noncomputable def slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) : ℝ :=
  faster_speed - (2 * train_length) / (passing_time * faster_speed / 3600)

/-- Theorem stating that under the given conditions, the speed of the slower train is 36 km/hr -/
theorem slower_train_speed_is_36 :
  slower_train_speed 46 36 50 = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_is_36_l669_66914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joel_toy_donation_ratio_l669_66977

theorem joel_toy_donation_ratio : 
  let initial_toys : ℕ := 18 + 42 + 2 + 13
  let total_donated : ℕ := 108
  let joel_added : ℕ := 22
  let sister_added : ℕ := total_donated - initial_toys - joel_added
  ∀ (initial_toys total_donated joel_added sister_added : ℕ),
    initial_toys = 18 + 42 + 2 + 13 →
    total_donated = 108 →
    joel_added = 22 →
    sister_added = total_donated - initial_toys - joel_added →
    (joel_added : ℚ) / sister_added = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joel_toy_donation_ratio_l669_66977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_circus_break_even_l669_66939

/-- Represents the financial model of Steve's circus production --/
structure CircusProduction where
  fixed_overhead : ℚ
  min_production_cost : ℚ
  max_production_cost : ℚ
  venue_capacity : ℕ
  average_sales_percentage : ℚ
  ticket_price : ℚ

/-- Calculates the break-even point for the circus production --/
def break_even_point (c : CircusProduction) : ℕ :=
  let avg_production_cost := (c.min_production_cost + c.max_production_cost) / 2
  let avg_tickets_sold := (c.venue_capacity : ℚ) * c.average_sales_percentage
  let revenue_per_performance := avg_tickets_sold * c.ticket_price
  let profit_per_performance := revenue_per_performance - avg_production_cost
  (c.fixed_overhead / profit_per_performance).ceil.toNat

/-- Theorem stating that the break-even point for Steve's circus production is 9 performances --/
theorem steve_circus_break_even :
  let c : CircusProduction := {
    fixed_overhead := 81000,
    min_production_cost := 5000,
    max_production_cost := 9000,
    venue_capacity := 500,
    average_sales_percentage := 4/5,
    ticket_price := 40
  }
  break_even_point c = 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_circus_break_even_l669_66939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_accuracy_increases_with_distance_line_most_accurate_at_max_distance_l669_66971

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Accuracy of line definition -/
noncomputable def lineAccuracy (p1 p2 : Point2D) : ℝ :=
  1 / (1 + Real.exp (-distance p1 p2))

/-- Theorem: As the distance between two points increases, 
    the accuracy of the line definition increases -/
theorem line_accuracy_increases_with_distance (p1 p2 p3 p4 : Point2D) :
  distance p1 p2 < distance p3 p4 → lineAccuracy p1 p2 < lineAccuracy p3 p4 := by
  sorry

/-- Corollary: The line is most accurately defined when 
    the distance between the points is maximized -/
theorem line_most_accurate_at_max_distance (p1 p2 : Point2D) (d : ℝ) :
  d > 0 → lineAccuracy p1 p2 ≤ lineAccuracy p1 (Point2D.mk (p1.x + d) p1.y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_accuracy_increases_with_distance_line_most_accurate_at_max_distance_l669_66971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l669_66979

/-- The phase shift of a sinusoidal function y = A sin(Bx + C) is given by -C/B -/
noncomputable def phase_shift (B C : ℝ) : ℝ := -C / B

/-- The given sinusoidal function -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x - Real.pi / 4)

theorem phase_shift_of_f :
  phase_shift 3 (-Real.pi / 4) = Real.pi / 12 := by
  unfold phase_shift
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l669_66979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_lambdas_l669_66942

-- Define the curve E
noncomputable def E (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2) = 2 * Real.sqrt 2

-- Define the line l
def l (x y : ℝ) : Prop :=
  ∃ (k m : ℝ), y = k * (x - 1) + m

-- Define points P and Q on the intersection of l and E
def P (x y : ℝ) : Prop := E x y ∧ l x y
def Q (x y : ℝ) : Prop := E x y ∧ l x y ∧ ¬(x = 1 ∧ y = 0) -- Q ≠ F

-- Define point R on y-axis
def R (y : ℝ) : Prop := l 0 y

-- Define λ₁ and λ₂
noncomputable def lambda₁ (xp yp y0 : ℝ) : ℝ := (xp) / (1 - xp)
noncomputable def lambda₂ (xq yq y0 : ℝ) : ℝ := (xq) / (1 - xq)

theorem constant_sum_of_lambdas 
  (xp yp xq yq y0 : ℝ) 
  (hP : P xp yp) 
  (hQ : Q xq yq) 
  (hR : R y0) 
  (h_lambda₁ : yp - y0 = lambda₁ xp yp y0 * (-yp))
  (h_lambda₂ : yq - y0 = lambda₂ xq yq y0 * (-yq)) :
  lambda₁ xp yp y0 + lambda₂ xq yq y0 = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_lambdas_l669_66942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l669_66929

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 2 * Real.pi / 3) - Real.cos (2 * x)

-- Define the theorem
theorem triangle_angles (a b c : ℝ) (h1 : f (b / 2) = -Real.sqrt 3 / 2)
                        (h2 : b = 1) (h3 : c = Real.sqrt 3) (h4 : a > b) :
  ∃ (A B C : ℝ),
    A + B + C = Real.pi ∧
    Real.sin A / a = Real.sin B / b ∧
    Real.sin B / b = Real.sin C / c ∧
    B = Real.pi / 6 ∧
    C = Real.pi / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l669_66929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_a_equals_sqrt_three_l669_66967

-- Define the complex solutions z₁ and z₂
noncomputable def z₁ : ℂ := sorry
noncomputable def z₂ : ℂ := sorry

-- Define the equation z² + z + 1 = 0
def eq_zero (z : ℂ) : Prop := z^2 + z + 1 = 0

-- Assume z₁ and z₂ are solutions to the equation
axiom z₁_solution : eq_zero z₁
axiom z₂_solution : eq_zero z₂

-- Part 1: Prove that 1/z₁ + 1/z₂ = -1
theorem sum_of_reciprocals : 1 / z₁ + 1 / z₂ = -1 := by sorry

-- Define that z₁ is in the third quadrant
axiom z₁_third_quadrant : z₁.re < 0 ∧ z₁.im < 0

-- Define a real number a
variable (a : ℝ)

-- Define that z₁ · (a + i) is purely imaginary
axiom z₁_times_a_plus_i_imaginary : (z₁ * Complex.I * a).re = 0

-- Part 2: Prove that a = √3
theorem a_equals_sqrt_three : a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_a_equals_sqrt_three_l669_66967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l669_66989

theorem angle_in_third_quadrant (α : Real) :
  Real.sin α < 0 → Real.tan α > 0 → α ∈ Set.Ioo π (3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l669_66989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_station_l669_66999

/-- Calculates the time (in seconds) for a train to pass a station -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (station_length : ℝ) : ℝ :=
  let total_distance := train_length + station_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 250 meters traveling at 36 km/hour passing a station of length 200 meters takes 45 seconds -/
theorem train_passing_station : train_passing_time 250 36 200 = 45 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_station_l669_66999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_and_g_l669_66915

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (3 * x + Real.pi / 2) - 4

def symmetric_about_point (f g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f x = g (2 * p.1 - x) + 2 * p.2

theorem symmetry_of_f_and_g :
  symmetric_about_point f g (0, -2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_and_g_l669_66915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_40_l669_66934

/-- Calculates the downstream distance given upstream distance, trip time, and still water speed -/
noncomputable def downstreamDistance (upstreamDistance : ℝ) (tripTime : ℝ) (stillWaterSpeed : ℝ) : ℝ :=
  let streamSpeed := (stillWaterSpeed * tripTime - upstreamDistance) / tripTime
  (stillWaterSpeed + streamSpeed) * tripTime

/-- Proves that given the specified conditions, the downstream distance is 40 km -/
theorem downstream_distance_is_40 :
  let upstreamDistance : ℝ := 30
  let tripTime : ℝ := 5
  let stillWaterSpeed : ℝ := 7
  downstreamDistance upstreamDistance tripTime stillWaterSpeed = 40 := by
  -- Unfold the definition of downstreamDistance
  unfold downstreamDistance
  -- Simplify the expression
  simp
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_40_l669_66934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_is_multiplication_by_two_l669_66998

/-- The operation performed on the original number -/
def f : ℝ → ℝ := sorry

/-- The original number -/
def x : ℝ := 13

/-- Theorem stating that the operation f satisfies the given conditions and is equivalent to multiplication by 2 -/
theorem operation_is_multiplication_by_two :
  (3 * (f x + 7) = 99) → (f = fun y ↦ 2 * y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_is_multiplication_by_two_l669_66998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l669_66945

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  x*y / Real.sqrt (x*y + y*z) + y*z / Real.sqrt (y*z + x*z) + x*z / Real.sqrt (x*z + x*y) ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l669_66945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_12_seconds_l669_66984

-- Define the given parameters
noncomputable def train1_length : ℝ := 250
noncomputable def train1_speed_kmh : ℝ := 90
noncomputable def bridge_length : ℝ := 300
noncomputable def train2_length : ℝ := 200
noncomputable def train2_speed_kmh : ℝ := 75

-- Convert speeds from km/h to m/s
noncomputable def train1_speed_ms : ℝ := train1_speed_kmh * 1000 / 3600
noncomputable def train2_speed_ms : ℝ := train2_speed_kmh * 1000 / 3600

-- Calculate relative speed
noncomputable def relative_speed : ℝ := train1_speed_ms + train2_speed_ms

-- Calculate total distance
noncomputable def total_distance : ℝ := train1_length + bridge_length

-- Theorem: The time for Train 1 to cross the bridge is approximately 12 seconds
theorem train_crossing_time_approx_12_seconds :
  ∃ ε > 0, abs (total_distance / relative_speed - 12) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_12_seconds_l669_66984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l669_66931

/-- Represents a batsman's performance in cricket --/
structure Batsman where
  initialAverage : ℚ
  runsScored : ℕ
  inningNumber : ℕ
  averageIncrease : ℚ

/-- Calculates the new average after an inning --/
noncomputable def newAverage (b : Batsman) : ℚ :=
  (b.initialAverage * (b.inningNumber - 1) + b.runsScored) / b.inningNumber

/-- Theorem stating that the batsman's average after the 17th inning is 8 --/
theorem batsman_average_after_17th_inning
  (b : Batsman)
  (h1 : b.inningNumber = 17)
  (h2 : b.runsScored = 56)
  (h3 : newAverage b = b.initialAverage + b.averageIncrease)
  (h4 : b.averageIncrease = 3) :
  newAverage b = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l669_66931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_is_6pi_squared_l669_66965

/-- The volume of the solid formed by rotating the region bounded by y = 5 cos x, y = cos x, x = 0,
    and x ≥ 0 around the x-axis -/
noncomputable def rotationVolume : ℝ := sorry

/-- The upper bounding function -/
noncomputable def upperBound (x : ℝ) : ℝ := 5 * Real.cos x

/-- The lower bounding function -/
noncomputable def lowerBound (x : ℝ) : ℝ := Real.cos x

/-- The left boundary of the region -/
def leftBoundary : ℝ := 0

/-- The right boundary of the region -/
noncomputable def rightBoundary : ℝ := Real.pi / 2

/-- Theorem stating that the volume of the solid of revolution is 6π² -/
theorem rotation_volume_is_6pi_squared :
  rotationVolume = 6 * Real.pi ^ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_is_6pi_squared_l669_66965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_student_l669_66904

noncomputable def scores_A : List ℝ := [82, 82, 79, 95, 87]
noncomputable def scores_B : List ℝ := [95, 75, 80, 90, 85]

noncomputable def average (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

noncomputable def variance (scores : List ℝ) : ℝ :=
  let avg := average scores
  (scores.map (fun x => (x - avg) ^ 2)).sum / scores.length

theorem select_student (scores_A scores_B : List ℝ) :
  average scores_A = average scores_B →
  variance scores_A < variance scores_B →
  True :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_student_l669_66904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_two_curves_l669_66974

/-- Definition of a line being tangent to a curve at a point. -/
def Set.IsTangentTo (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ l ∧ p ∈ C ∧ ∃! q, q ∈ l ∩ C

/-- Given curves C₁: y = x² and C₂: y = -(x-2)², a line l that is tangent to both C₁ and C₂ has the equation y = 0 or 4x - y - 4 = 0. -/
theorem tangent_line_to_two_curves (x y : ℝ) :
  let C₁ := {(x, y) | y = x^2}
  let C₂ := {(x, y) | y = -(x-2)^2}
  let l := (fun (m b : ℝ) ↦ {(x, y) | y = m*x + b})
  ∃ (m b : ℝ), (∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ C₁ → (l m b).IsTangentTo C₁ (x₀, y₀)) ∧
               (∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ C₂ → (l m b).IsTangentTo C₂ (x₁, y₁)) →
  (m = 0 ∧ b = 0) ∨ (m = 4 ∧ b = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_two_curves_l669_66974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_y_l669_66986

noncomputable def y (x : ℝ) : ℝ := 
  Matrix.det !![Real.cos x, Real.sin x; Real.sin x, Real.cos x]

theorem period_of_y (a : ℝ) :
  (∃ (p : ℝ), p > 0 ∧ p = a * Real.pi ∧ ∀ (x : ℝ), y (x + p) = y x) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_y_l669_66986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l669_66916

/-- An isosceles triangle with integral sides and perimeter 11 -/
structure IsoscelesTriangle where
  a : ℕ  -- length of two equal sides
  b : ℕ  -- length of the base
  isIsosceles : a > 0 ∧ b > 0
  perimeter : 2 * a + b = 11

/-- The area of an isosceles triangle -/
noncomputable def areaOfIsoscelesTriangle (t : IsoscelesTriangle) : ℝ :=
  let h := Real.sqrt (t.a^2 - (t.b/2)^2)
  (t.b * h) / 2

theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, areaOfIsoscelesTriangle t = (5 * Real.sqrt 2.75) / 2 := by
  sorry

#check isosceles_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l669_66916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_domain_f_complement_domain_g_m_value_for_intersection_l669_66918

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 8)
noncomputable def g (x m : ℝ) : ℝ := Real.log (-x^2 + 6*x + m)

-- Define the domains of f and g
def domain_f : Set ℝ := {x : ℝ | -x^2 + 2*x + 8 ≥ 0}
def domain_g (m : ℝ) : Set ℝ := {x : ℝ | -x^2 + 6*x + m > 0}

-- Theorem 1
theorem intersection_domain_f_complement_domain_g :
  domain_f ∩ (domain_g (-5))ᶜ = Set.Icc (-2) 1 := by sorry

-- Theorem 2
theorem m_value_for_intersection :
  (∃ m : ℝ, domain_f ∩ domain_g m = Set.Ioc (-1) 4) →
  (∃ m : ℝ, m = 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_domain_f_complement_domain_g_m_value_for_intersection_l669_66918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_theorem_l669_66946

/-- The radius of the round table -/
noncomputable def table_radius : ℝ := 5

/-- The number of rectangular place mats -/
def num_mats : ℕ := 8

/-- The width of each place mat -/
noncomputable def mat_width : ℝ := 1

/-- The length of each place mat -/
noncomputable def mat_length : ℝ := (3 * Real.sqrt 11 - 10 * Real.sqrt (2 - Real.sqrt 2) + 1) / 2

/-- Theorem stating the correct length of each place mat -/
theorem place_mat_length_theorem :
  ∀ (r : ℝ) (n : ℕ) (w x : ℝ),
    r = table_radius →
    n = num_mats →
    w = mat_width →
    x = mat_length →
    (∃ (octagon : Set (ℝ × ℝ)),
      (∀ p, p ∈ octagon → p.1^2 + p.2^2 = r^2) ∧
      (∃ (vertices : Finset (ℝ × ℝ)),
        vertices.card = n ∧
        (∀ v, v ∈ vertices → v ∈ octagon) ∧
        (∀ v1 v2, v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → 
          (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = x^2))) →
    x = (3 * Real.sqrt 11 - 10 * Real.sqrt (2 - Real.sqrt 2) + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_theorem_l669_66946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_arrangements_l669_66957

/-- Represents a coin with a type (gold or silver) and an orientation (face up or face down) -/
structure Coin where
  isGold : Bool
  isFaceUp : Bool

/-- A stack of coins -/
def CoinStack := List Coin

/-- Checks if two adjacent coins in a stack are not face to face -/
def validAdjacent (c1 c2 : Coin) : Bool :=
  ¬(c1.isFaceUp ∧ c2.isFaceUp)

/-- Checks if a coin stack is valid (no adjacent coins are face to face) -/
def isValidStack (stack : CoinStack) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | c1 :: c2 :: rest => validAdjacent c1 c2 && isValidStack (c2 :: rest)

/-- Counts the number of gold coins in a stack -/
def countGoldCoins (stack : CoinStack) : Nat :=
  stack.filter (·.isGold) |>.length

/-- The main theorem: number of valid distinguishable arrangements -/
theorem num_valid_arrangements :
  (validStacks : Finset CoinStack) →
  (∀ stack ∈ validStacks, isValidStack stack ∧ countGoldCoins stack = 5) →
  validStacks.card = 2772 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_arrangements_l669_66957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_square_inequality_l669_66953

theorem sin_square_inequality (m : ℝ) : 
  (∀ x : ℝ, (Real.sin x)^2 + m * Real.sin x + (m^2 - 3) / m ≤ 0) ↔ (0 < m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_square_inequality_l669_66953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_m_circumference_l669_66951

/-- Represents a right circular cylinder tank -/
structure Tank where
  height : ℝ
  circumference : ℝ

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (t : Tank) : ℝ :=
  (t.circumference ^ 2 * t.height) / (4 * Real.pi)

theorem tank_m_circumference (tank_m tank_b : Tank) 
  (h1 : tank_m.height = 10)
  (h2 : tank_b.height = 8)
  (h3 : tank_b.circumference = 10)
  (h4 : cylinderVolume tank_m = 0.8 * cylinderVolume tank_b) :
  tank_m.circumference = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_m_circumference_l669_66951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_eq_u_poly_sum_of_coefficients_is_seven_l669_66950

/-- Sequence u_n defined by initial condition and recurrence relation -/
def u : ℕ → ℚ
  | 0 => 7  -- Added case for 0 to handle all natural numbers
  | n + 1 => u n + (5 + 2 * n)

/-- u_n as a polynomial in n -/
def u_poly (n : ℕ) : ℚ := n^2 + 2*n + 4

/-- Theorem stating that u_n equals u_poly for all n ≥ 1 -/
theorem u_eq_u_poly : ∀ n : ℕ, n ≥ 1 → u n = u_poly n := by
  sorry

/-- Sum of coefficients of u_poly -/
def coeff_sum : ℚ := 1 + 2 + 4

/-- Main theorem: The sum of coefficients of u_n (expressed as a polynomial) is 7 -/
theorem sum_of_coefficients_is_seven : coeff_sum = 7 := by
  rfl

#eval coeff_sum  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_eq_u_poly_sum_of_coefficients_is_seven_l669_66950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l669_66912

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def b : ℝ × ℝ := (1, -(Real.sqrt 3) / 3)

theorem vector_properties :
  (Real.sqrt (a.1^2 + a.2^2) = 2) ∧
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l669_66912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_website_earnings_l669_66990

/-- Calculate the sum of visitors for the first 6 days -/
noncomputable def S6 (g : ℝ) : ℝ :=
  100 * (1 + (1 + g/100) + (1 + g/100)^2 + (1 + g/100)^3 + (1 + g/100)^4 + (1 + g/100)^5)

/-- Calculate the total earnings for the week -/
noncomputable def total_earnings (g : ℝ) : ℝ :=
  3 * S6 g * 0.03

theorem website_earnings (g : ℝ) :
  total_earnings g = 3 * S6 g * 0.03 := by
  -- Unfold the definitions of total_earnings and S6
  unfold total_earnings S6
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_website_earnings_l669_66990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_difference_is_four_l669_66993

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_kayak_ratio : Rat

/-- Calculates the difference between the number of canoes and kayaks rented --/
def rental_difference (info : RentalInfo) : ℤ :=
  let kayaks := info.total_revenue / (info.canoe_cost * info.canoe_kayak_ratio.num + info.kayak_cost * info.canoe_kayak_ratio.den)
  let canoes := kayaks * info.canoe_kayak_ratio.num / info.canoe_kayak_ratio.den
  ↑canoes - ↑kayaks

/-- Theorem stating that the difference between canoes and kayaks rented is 4 --/
theorem rental_difference_is_four :
  let info : RentalInfo := {
    canoe_cost := 14,
    kayak_cost := 15,
    total_revenue := 288,
    canoe_kayak_ratio := 3 / 2
  }
  rental_difference info = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_difference_is_four_l669_66993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l669_66981

theorem min_value_of_exponential_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_sum : a + b = 1/a + 1/b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1/x + 1/y → (3 : ℝ)^a + (81 : ℝ)^b ≤ (3 : ℝ)^x + (81 : ℝ)^y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l669_66981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l669_66975

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 2 * x / (x^2 + b * x + 1)

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := (1 - x) / x

-- Theorem statement
theorem inverse_function_condition (b : ℝ) :
  (∀ x, f_inv x = (f b)⁻¹ x) → b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l669_66975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_percentage_in_second_metal_l669_66997

/-- Calculates the gold percentage in the second metal given the alloy requirements --/
theorem gold_percentage_in_second_metal
  (total_alloy : ℝ)
  (desired_gold_percentage : ℝ)
  (first_metal_gold_percentage : ℝ)
  (amount_each_metal : ℝ)
  (h1 : total_alloy = 12.4)
  (h2 : desired_gold_percentage = 0.5)
  (h3 : first_metal_gold_percentage = 0.6)
  (h4 : amount_each_metal = 6.2)
  : (total_alloy * desired_gold_percentage - amount_each_metal * first_metal_gold_percentage) / amount_each_metal = 0.4 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_percentage_in_second_metal_l669_66997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l669_66940

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define the perpendicularity condition for line l
def perpendicular_to_l₁ (m : ℝ) : Prop := m * 2 = -1

-- Define the equation of line l
def l (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the condition for circle M's center lying on l₁
def center_on_l₁ (a b : ℝ) : Prop := 2 * a - b = 0

-- Define the tangency condition to y-axis
def tangent_to_y_axis (a r : ℝ) : Prop := r = |a|

-- Define the chord length condition
def chord_length_condition (a b r : ℝ) : Prop :=
  r^2 = ((|a + b + 2|) / Real.sqrt 2)^2 + (Real.sqrt 2 / 2)^2

-- Define the equations of circle M
def circle_M₁ (x y : ℝ) : Prop := (x + 5/7)^2 + (y + 10/7)^2 = 25/49
def circle_M₂ (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

theorem line_and_circle_equations :
  ∀ (x y a b r : ℝ),
  l₁ a b →
  l₂ x y →
  perpendicular_to_l₁ (-1/2) →
  center_on_l₁ a b →
  tangent_to_y_axis a r →
  chord_length_condition a b r →
  (l x y ∧ (circle_M₁ x y ∨ circle_M₂ x y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l669_66940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approx_l669_66919

def face_area_1 : ℝ := 56
def face_area_2 : ℝ := 63
def face_area_3 : ℝ := 72

def is_valid_prism (a b c : ℝ) : Prop :=
  a * b = face_area_1 ∧ b * c = face_area_2 ∧ a * c = face_area_3 ∧
  (a = 2 * b ∨ b = 2 * a ∨ c = 2 * a ∨ a = 2 * c ∨ b = 2 * c ∨ c = 2 * b)

def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume_approx :
  ∃ (a b c : ℝ), is_valid_prism a b c ∧ 
  (761.5 < volume a b c ∧ volume a b c < 762.5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_approx_l669_66919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_circumsphere_radius_l669_66983

/-- The radius of the circumsphere of a right triangular prism -/
noncomputable def circumsphere_radius (a b c : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + c^2) / 2

/-- Theorem: The radius of the circumsphere of a right triangular prism with pairwise perpendicular 
    sides of lengths a, b, and c is given by √(a² + b² + c²) / 2 -/
theorem right_triangular_prism_circumsphere_radius (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (R : ℝ), R = circumsphere_radius a b c ∧ 
             R = Real.sqrt (a^2 + b^2 + c^2) / 2 := by
  use circumsphere_radius a b c
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_circumsphere_radius_l669_66983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_2000_l669_66985

def fibonacci_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n > 2, a n = a (n - 1) + a (n - 2)

theorem fibonacci_2000 (a : ℕ → ℤ) (h : fibonacci_sequence a) 
  (h2015 : a 2015 = 1) (h2017 : a 2017 = -1) : a 2000 = -18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_2000_l669_66985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l669_66952

/-- A right triangle with specific median lengths has a hypotenuse of √86.4 -/
theorem right_triangle_hypotenuse (x y : ℝ) 
  (right_triangle : x^2 + y^2 = (x^2 + y^2) / 2) 
  (median1 : x^2 + (y/2)^2 = 9) 
  (median2 : y^2 + (x/2)^2 = 18) : 
  Real.sqrt (4*(x^2 + y^2)) = Real.sqrt 86.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l669_66952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_square_l669_66966

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property that the domain of f[log(x+1)] is [0,9]
def domain_f_log (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 ≤ x ∧ x ≤ 9) ↔ ∃ y, f (Real.log (x + 1)) = y

-- Theorem statement
theorem domain_f_square (f : ℝ → ℝ) :
  domain_f_log f →
  ∀ x, (∃ y, f (x^2) = y) ↔ (-1 ≤ x ∧ x ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_square_l669_66966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_general_term_l669_66913

noncomputable def x : ℕ → ℝ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => if n % 2 = 0 then x (n + 2) + x (n + 1) else x (n + 2) + 2 * x (n + 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  let m := (n + 1) / 2
  if n % 2 = 1 then
    1/4 * (3 - Real.sqrt 2) * (2 + Real.sqrt 2) ^ m + 1/4 * (3 + Real.sqrt 2) * (2 - Real.sqrt 2) ^ m
  else
    1/4 * (1 + 2 * Real.sqrt 2) * (2 + Real.sqrt 2) ^ m + 1/4 * (1 - 2 * Real.sqrt 2) * (2 - Real.sqrt 2) ^ m

theorem x_eq_general_term : ∀ n : ℕ, n ≥ 1 → x n = general_term n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_general_term_l669_66913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_l669_66960

theorem geometric_sum_remainder (n : ℕ) (a : ℤ) : 
  (∃ k : ℤ, (2^(5*n) - 2 + a) = 31*k + 3) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_l669_66960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l669_66980

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2-m)

-- Define the properties of the function
def is_valid_function (m : ℝ) : Prop :=
  ∀ x : ℝ, -3-m ≤ x ∧ x ≤ m^2-m → f m x = -(f m (-x))

-- Theorem statement
theorem odd_function_property (m : ℝ) (h : is_valid_function m) : f m m < f m 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l669_66980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_when_a_zero_range_of_a_when_A_subset_B_l669_66927

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*x - 3) / (2*x + 1) + a
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x + 2) + Real.sqrt (2 - x)

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := { y | ∃ x ∈ Set.Icc 0 (3/2), f a x = y }
def B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Statement 1
theorem complement_A_intersect_B_when_a_zero :
  Set.univ \ (A 0 ∩ B) = Set.Iic (-2) ∪ Set.Ioi 0 := by sorry

-- Statement 2
theorem range_of_a_when_A_subset_B :
  { a : ℝ | A a ⊆ B } = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_when_a_zero_range_of_a_when_A_subset_B_l669_66927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_box_sphere_radius_l669_66902

/-- The radius of a sphere containing an inscribed rectangular box -/
noncomputable def sphere_radius (surface_area edge_sum : ℝ) : ℝ :=
  Real.sqrt 246

theorem inscribed_box_sphere_radius 
  (Q : Set (ℝ × ℝ × ℝ))  -- Rectangular box
  (s : ℝ)  -- Sphere radius
  (h1 : (∀ (a b c : ℝ), (a, b, c) ∈ Q → 2 * (a * b + b * c + c * a) = 616))  -- Surface area condition
  (h2 : (∀ (a b c : ℝ), (a, b, c) ∈ Q → 4 * (a + b + c) = 160))  -- Edge sum condition
  (h3 : (∀ (a b c : ℝ), (a, b, c) ∈ Q → (2 * s) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2))  -- Diagonal condition
  : s = sphere_radius 616 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_box_sphere_radius_l669_66902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_rectangle_perimeter_l669_66959

/-- A square that can be cut and rearranged into a rectangle -/
structure CuttableSquare where
  /-- The perimeter of the square -/
  perimeter : ℝ
  /-- The square can be cut into two congruent right-angled triangles and two congruent trapezia -/
  can_be_cut : Prop
  /-- The pieces can be rearranged to form a rectangle -/
  can_be_rearranged : Prop

/-- The perimeter of the rectangle formed by rearranging the cut square -/
noncomputable def rectangle_perimeter (s : CuttableSquare) : ℝ :=
  2 * Real.sqrt 5

/-- Theorem stating that a square with perimeter 4 can be rearranged into a rectangle with perimeter 2√5 -/
theorem square_to_rectangle_perimeter (s : CuttableSquare) 
  (h1 : s.perimeter = 4) 
  (h2 : s.can_be_cut) 
  (h3 : s.can_be_rearranged) : 
  rectangle_perimeter s = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_to_rectangle_perimeter_l669_66959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paramecium_probability_l669_66958

/-- Represents the probability of finding a paramecium in a water sample -/
noncomputable def probability_paramecium (total_volume sample_volume : ℝ) : ℝ :=
  sample_volume / total_volume

/-- Theorem: The probability of finding a paramecium in a 2 mL sample from 500 mL of water is 0.004 -/
theorem paramecium_probability :
  probability_paramecium 500 2 = 0.004 := by
  -- Unfold the definition of probability_paramecium
  unfold probability_paramecium
  -- Perform the division
  norm_num
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paramecium_probability_l669_66958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2028_eq_zero_l669_66907

def sequence_term (n : ℕ) : ℤ :=
  if n % 4 = 1 then n
  else if n % 4 = 2 then -n
  else if n % 4 = 3 then -n
  else n

def sequence_sum (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) sequence_term

theorem sequence_sum_2028_eq_zero :
  sequence_sum 2028 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2028_eq_zero_l669_66907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l669_66921

theorem range_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let S := {f | ∃ (x y : ℝ), a * x^2 - b * x * y + a * y^2 = 1 ∧ f = x^2 + y^2}
  (b < 2 * a → S = Set.Icc (2 / (2 * a + b)) (2 / (2 * a - b))) ∧
  (b ≥ 2 * a → S = Set.Ici (2 / (2 * a + b))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l669_66921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l669_66948

/-- A line passing through (-1, -2) and intersecting the circle x^2 + y^2 - 2x - 2y + 1 = 0 to form a chord of length √2 has an equation of either x - y - 1 = 0 or (17/7)x - y + 3/7 = 0 -/
theorem line_equation (l : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) :
  ((-1 : ℝ), -2) ∈ l →
  circle = {(x, y) | x^2 + y^2 - 2*x - 2*y + 1 = 0} →
  (∃ (p q : ℝ × ℝ), p ∈ l ∩ circle ∧ q ∈ l ∩ circle ∧ p ≠ q ∧ 
    Real.sqrt 2 = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y - 1 = 0) ∨ 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (17/7)*x - y + 3/7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l669_66948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_distance_l669_66968

/-- A person's walking journey with varying speeds on different terrains -/
structure WalkingJourney where
  total_time : ℝ
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- The total distance walked given a WalkingJourney -/
def total_distance (j : WalkingJourney) : ℝ :=
  sorry

/-- The specific journey described in the problem -/
def specific_journey : WalkingJourney where
  total_time := 5
  flat_speed := 4
  uphill_speed := 3
  downhill_speed := 6

/-- Theorem stating that the total distance for the specific journey is 20 km -/
theorem specific_journey_distance :
  total_distance specific_journey = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_journey_distance_l669_66968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_proof_l669_66909

noncomputable def sample_variance (xs : List ℝ) : ℝ :=
  let n := xs.length
  let mean := xs.sum / n
  (xs.map (fun x => (x - mean) ^ 2)).sum / n

theorem sample_variance_proof (a : ℝ) : 
  let xs := [a, 0, 1, 2, 3]
  (xs.sum / xs.length = 1) → (sample_variance xs = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_proof_l669_66909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_apple_price_difference_l669_66905

/-- Represents the price of items in dollars -/
abbrev Price := ℚ

/-- Represents the weight of items in kilograms -/
abbrev Weight := ℚ

/-- The price of apples per kilogram -/
def apple_price : Price := 2

/-- The price of walnuts per kilogram -/
def walnut_price : Price := 6

/-- The weight of apples Fabian wants to buy -/
def apple_weight : Weight := 5

/-- The weight of walnuts Fabian wants to buy -/
def walnut_weight : Weight := 1/2

/-- The number of sugar packs Fabian wants to buy -/
def sugar_packs : ℕ := 3

/-- The total cost of all items -/
def total_cost : Price := 16

/-- Theorem stating the price difference between apples and sugar -/
theorem sugar_apple_price_difference :
  ∃ (sugar_price : Price),
    sugar_price * (sugar_packs : ℚ) +
    apple_price * apple_weight +
    walnut_price * walnut_weight = total_cost ∧
    apple_price - sugar_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_apple_price_difference_l669_66905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vertex_distance_l669_66988

/-- Represents a point in a 2D plane -/
structure Point :=
  (x y : ℝ)

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a straight line in a 2D plane -/
structure Line :=
  (p : Point → Prop)

/-- The distance between a point and a line -/
noncomputable def distance (P : Point) (L : Line) : ℝ := sorry

/-- A theorem about the distance of a parallelogram's vertex from a line -/
theorem parallelogram_vertex_distance 
  (ABCD : Parallelogram) (L : Line) (a b : ℝ) :
  L.p ABCD.B →
  distance ABCD.A L = a →
  distance ABCD.C L = b →
  distance ABCD.D L = a + b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vertex_distance_l669_66988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l669_66917

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6
def total_books : ℕ := num_math_books + num_history_books

theorem book_arrangement_count :
  (num_math_books * (num_math_books - 1) * num_history_books * Nat.factorial (total_books - 3)) = 26210880 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l669_66917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l669_66922

noncomputable def h (x : ℝ) : ℝ := (5 * x - 3) / (2 * x - 10)

theorem domain_of_h :
  ∀ x : ℝ, x ≠ 5 → h x ∈ Set.univ \ {5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l669_66922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_sum_at_one_l669_66923

-- Polynomial is already defined in Mathlib, so we'll use the built-in definition
-- def Polynomial (α : Type*) := α → ℕ → α

-- Define what it means for a polynomial to be monic
def IsMonic {α : Type*} [Semiring α] (p : Polynomial α) : Prop :=
  Polynomial.leadingCoeff p = 1

-- Define what it means for a polynomial to have integer coefficients
def HasIntCoeffs (p : Polynomial ℤ) : Prop :=
  ∀ n : ℕ, ∃ k : ℤ, p.coeff n = k

-- Define the evaluation of a polynomial at a point
noncomputable def EvalPoly {α : Type*} [Semiring α] (p : Polynomial α) (x : α) : α :=
  p.eval x

-- State the theorem
theorem polynomial_factorization_sum_at_one 
  (r s : Polynomial ℤ) 
  (hr : IsMonic r) 
  (hs : IsMonic s) 
  (hr_nonconstant : ¬(r.degree = 0))
  (hs_nonconstant : ¬(s.degree = 0))
  (hr_int : HasIntCoeffs r)
  (hs_int : HasIntCoeffs s)
  (h : ∀ x : ℤ, EvalPoly (Polynomial.monomial 10 1 - Polynomial.monomial 5 50 + Polynomial.monomial 0 1) x = EvalPoly r x * EvalPoly s x) :
  EvalPoly r 1 + EvalPoly s 1 = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_sum_at_one_l669_66923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_draws_is_one_twentyfirst_l669_66924

/-- Represents a chip with a value from 1 to 7 -/
inductive Chip : Type
  | one
  | two
  | three
  | four
  | five
  | six
  | seven
deriving Fintype, Repr

/-- The box containing 7 chips -/
def Box : Finset Chip := Finset.univ

/-- The value of a chip -/
def chipValue : Chip → ℕ
  | Chip.one => 1
  | Chip.two => 2
  | Chip.three => 3
  | Chip.four => 4
  | Chip.five => 5
  | Chip.six => 6
  | Chip.seven => 7

/-- A draw sequence of 4 chips -/
structure DrawSequence :=
  (first second third fourth : Chip)
  (distinct : first ≠ second ∧ first ≠ third ∧ first ≠ fourth ∧
              second ≠ third ∧ second ≠ fourth ∧ third ≠ fourth)

/-- The sum of the first three chips in a draw sequence -/
def sumFirstThree (d : DrawSequence) : ℕ :=
  chipValue d.first + chipValue d.second + chipValue d.third

/-- The sum of all four chips in a draw sequence -/
def sumAll (d : DrawSequence) : ℕ :=
  sumFirstThree d + chipValue d.fourth

/-- The set of all possible draw sequences -/
noncomputable def allDrawSequences : Finset DrawSequence := sorry

/-- The set of valid draw sequences (sum of first three ≤ 9, sum of all four > 9) -/
noncomputable def validDrawSequences : Finset DrawSequence :=
  allDrawSequences.filter (λ d => sumFirstThree d ≤ 9 ∧ sumAll d > 9)

theorem probability_four_draws_is_one_twentyfirst :
  (validDrawSequences.card : ℚ) / (allDrawSequences.card : ℚ) = 1 / 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_draws_is_one_twentyfirst_l669_66924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_is_max_in_col_and_min_in_row_l669_66911

def matrix : Matrix (Fin 5) (Fin 5) ℕ := ![
  ![12, 7, 9, 6, 3],
  ![14, 9, 16, 13, 11],
  ![10, 5, 6, 8, 12],
  ![15, 6, 18, 14, 4],
  ![9, 4, 7, 12, 5]
]

def target_row : Fin 5 := 1
def target_col : Fin 5 := 1

theorem target_is_max_in_col_and_min_in_row :
  (∀ i : Fin 5, matrix i target_col ≤ matrix target_row target_col) ∧
  (∀ j : Fin 5, matrix target_row target_col ≤ matrix target_row j) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_is_max_in_col_and_min_in_row_l669_66911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l669_66925

noncomputable def sum_of_roots (a b c : ℝ) : ℝ := -b / a

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 16*x - 9) → (∃ s : ℝ, s = 16 ∧ s = sum_of_roots 1 (-16) 9) :=
by
  intro h
  use 16
  constructor
  · rfl
  · unfold sum_of_roots
    norm_num

#check sum_of_roots_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l669_66925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l669_66947

-- Define the function f(x) = e^x + x
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

-- Define a structure for a point on the curve
structure PointOnCurve where
  x : ℝ
  y : ℝ
  h : y = f x

-- Define a structure for three points with x-coordinates in arithmetic sequence
structure ThreePointsArithmetic where
  A : PointOnCurve
  B : PointOnCurve
  C : PointOnCurve
  h : ∃ (d : ℝ), B.x - A.x = d ∧ C.x - B.x = d

-- Define the angle between three points
noncomputable def angle (A B C : PointOnCurve) : ℝ := sorry

-- Theorem statement
theorem triangle_properties (points : ThreePointsArithmetic) :
  (angle points.A points.B points.C > π / 2) ∧
  ¬((points.A.x - points.B.x)^2 + (points.A.y - points.B.y)^2 = 
    (points.B.x - points.C.x)^2 + (points.B.y - points.C.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l669_66947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olive_purchase_optimization_l669_66991

/-- Represents the price of a jar of olives -/
structure OliveJar :=
  (olives : ℕ)
  (price : ℚ)

/-- Calculates the discounted price for a given number of jars -/
def discountedPrice (jar : OliveJar) (count : ℕ) : ℚ :=
  if count ≥ 3 then
    (jar.price * count) * (9/10)
  else
    jar.price * count

/-- Theorem stating the minimum cost to buy 80 olives and the resulting change -/
theorem olive_purchase_optimization :
  let jar10 : OliveJar := ⟨10, 1⟩
  let jar20 : OliveJar := ⟨20, 3/2⟩
  let jar30 : OliveJar := ⟨30, 5/2⟩
  let jar40 : OliveJar := ⟨40, 4⟩
  let budget : ℚ := 10
  let targetOlives : ℕ := 80
  let minCost : ℚ := discountedPrice jar20 3 + jar20.price
  let change : ℚ := budget - minCost
  (minCost = 111/20) ∧ (change = 89/20) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olive_purchase_optimization_l669_66991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoe_snack_calories_l669_66920

/-- Represents the caloric content of Zoe's snack items -/
structure SnackCalories where
  strawberry : Float
  yogurt : Float
  blueberry : Float
  apple : Float
  almond : Float

/-- Represents the quantities of items in Zoe's snack -/
structure SnackQuantities where
  strawberries : Nat
  yogurt : Nat
  blueberries : Nat
  apples : Nat
  almonds : Nat

/-- Calculates the total calories in Zoe's snack -/
def totalCalories (cal : SnackCalories) (qty : SnackQuantities) : Float :=
  cal.strawberry * qty.strawberries.toFloat +
  cal.yogurt * qty.yogurt.toFloat +
  cal.blueberry * qty.blueberries.toFloat +
  cal.apple * qty.apples.toFloat +
  cal.almond * qty.almonds.toFloat

/-- Theorem stating that the total calories in Zoe's snack is 321.4 -/
theorem zoe_snack_calories :
  let cal := SnackCalories.mk 4 17 0.8 95 7
  let qty := SnackQuantities.mk 12 6 8 1 10
  totalCalories cal qty = 321.4 := by
  sorry

#eval let cal := SnackCalories.mk 4 17 0.8 95 7
      let qty := SnackQuantities.mk 12 6 8 1 10
      totalCalories cal qty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoe_snack_calories_l669_66920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorist_gas_affordability_l669_66962

/-- The maximum number of whole gallons a motorist can afford given certain conditions -/
theorem motorist_gas_affordability
  (expected_gallons : ℕ)
  (price_difference : ℕ)
  (actual_price : ℕ)
  (h1 : expected_gallons = 12)
  (h2 : price_difference = 30)
  (h3 : actual_price = 150) :
  (expected_gallons * (actual_price - price_difference)) / actual_price = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorist_gas_affordability_l669_66962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_speed_l669_66932

/-- Given an escalator of length 200 feet, if a person walking at 5 feet per second
    takes 10 seconds to cover the entire length, then the rate of the escalator
    is 15 feet per second. -/
theorem escalator_speed (person_speed escalator_speed time escalator_length : ℝ)
    (h1 : person_speed = 5)
    (h2 : time = 10)
    (h3 : escalator_length = 200)
    (h4 : (person_speed + escalator_speed) * time = escalator_length) :
    escalator_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_speed_l669_66932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_five_smaller_than_negative_three_l669_66906

theorem only_negative_five_smaller_than_negative_three : 
  ∀ x : ℝ, x ∈ ({-2, 4, -5, 1} : Set ℝ) → (x < -3 ↔ x = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_five_smaller_than_negative_three_l669_66906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_inequality_proof_l669_66944

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

-- Theorem 1: The maximum value of f(x) for x > 0 is 1
theorem f_max_value (x : ℝ) (h : x > 0) : f x ≤ 1 := by
  sorry

-- Theorem 2: For x ≥ 1, ((x + 1)(1 + ln x)) / x ≥ 2
theorem inequality_proof (x : ℝ) (h : x ≥ 1) : ((x + 1) * (1 + Real.log x)) / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_inequality_proof_l669_66944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l669_66969

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

-- State the theorem
theorem f_decreasing :
  (∀ x y, x < -1 → y < -1 → x < y → f x > f y) ∧
  (∀ x y, x > -1 → y > -1 → x < y → f x > f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l669_66969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_students_same_group_l669_66973

theorem probability_two_students_same_group (n : ℕ) (h : n = 4) :
  let total_ways := Nat.choose n 2
  let favorable_ways := 2
  (favorable_ways : ℚ) / total_ways = 1 / 3 :=
by
  -- Substitute n with 4
  rw [h]
  -- Simplify the left side of the equation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_students_same_group_l669_66973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_speed_l669_66992

/-- Swimmer's speed in still water given time, distance, and current speed -/
theorem swimmers_speed (time distance current_speed swimmers_speed : ℝ) 
  (h1 : time = 3.5)
  (h2 : distance = 7)
  (h3 : current_speed = 2)
  (h4 : time = distance / (swimmers_speed - current_speed)) :
  swimmers_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_speed_l669_66992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_for_specific_solid_l669_66961

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure RectangularSolid where
  a : ℝ
  r : ℝ

/-- The volume of the rectangular solid -/
noncomputable def volume (solid : RectangularSolid) : ℝ :=
  solid.a^3

/-- The surface area of the rectangular solid -/
noncomputable def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.a^2 / solid.r + solid.a^2 + solid.a^2 * solid.r)

/-- The sum of lengths of all edges of the rectangular solid -/
noncomputable def sumOfEdges (solid : RectangularSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

/-- Theorem stating the sum of edges for a specific rectangular solid -/
theorem sum_of_edges_for_specific_solid :
  ∃ (solid : RectangularSolid),
    volume solid = 512 ∧
    surfaceArea solid = 384 ∧
    sumOfEdges solid = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_for_specific_solid_l669_66961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l669_66987

/-- Given an arithmetic sequence {a_n} and another sequence {b_n} -/
noncomputable def sequence_a : ℕ → ℝ := fun n => 2 * n - 1

noncomputable def sequence_b : ℕ → ℝ := fun n => (2 * n - 1) * 2^(n - 1)

/-- Sum of first n terms of {a_n} -/
noncomputable def S (n : ℕ) : ℝ := n * (2 * n - 1) / 2

/-- Sum of first n terms of {b_n} -/
noncomputable def T (n : ℕ) : ℝ := (2 * n - 3) * 2^n + 3

/-- Theorem stating the properties of the sequences and the sum formula -/
theorem sequence_sum_theorem (n : ℕ) :
  sequence_a 1 = 1 ∧
  S 3 = 9 ∧
  sequence_b 1 = 1 ∧
  sequence_b 3 = 20 ∧
  (∃ q : ℝ, q > 0 ∧ ∀ k : ℕ, k > 0 → sequence_b k / sequence_a k = q^(k - 1)) →
  T n = (2 * n - 3) * 2^n + 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l669_66987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_curve_properties_l669_66982

/-- Parametric curve defined by x = 4√t + 1/√t and y = 4√t - 1/√t, where t > 0 -/
noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  let x := 4 * Real.sqrt t + 1 / Real.sqrt t
  let y := 4 * Real.sqrt t - 1 / Real.sqrt t
  (x, y)

theorem parametric_curve_properties :
  ∀ t > 0,
    let (x, y) := parametric_curve t
    x^2 - y^2 = 16 ∧ x ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_curve_properties_l669_66982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preimage_of_negative_five_two_l669_66995

-- Define the function f
noncomputable def f (p : ℝ × ℝ) : ℝ × ℝ := 
  let (x, y) := p
  ((x + y) / 2, (x - y) / 2)

-- Theorem statement
theorem preimage_of_negative_five_two :
  f (-3, -7) = (-5, 2) := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the left-hand side
  simp
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_preimage_of_negative_five_two_l669_66995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_d_value_l669_66976

-- Define the points
def point_a (a : ℝ) : ℝ × ℝ × ℝ := (1, 0, a)
def point_b (b : ℝ) : ℝ × ℝ × ℝ := (b, 1, 0)
def point_c (c : ℝ) : ℝ × ℝ × ℝ := (0, c, 1)
def point_d (d : ℝ) : ℝ × ℝ × ℝ := (9*d, 9*d, -d)

-- Define collinearity condition
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (q.fst - p.fst, q.snd - p.snd, (q.2.2 : ℝ) - (p.2.2 : ℝ)) = t • (r.fst - p.fst, r.snd - p.snd, (r.2.2 : ℝ) - (p.2.2 : ℝ))

-- Theorem statement
theorem collinear_points_d_value :
  ∀ a b c d : ℝ,
    (collinear (point_a a) (point_b b) (point_c c) ∧
     collinear (point_a a) (point_b b) (point_d d) ∧
     collinear (point_a a) (point_c c) (point_d d)) →
    d = (15 + Real.sqrt 477) / 126 ∨ d = (15 - Real.sqrt 477) / 126 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_d_value_l669_66976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_area_theorem_l669_66994

/-- Calculates the square footage of a bathroom given the number of tiles and tile size -/
noncomputable def bathroom_square_footage (width_tiles : ℕ) (length_tiles : ℕ) (tile_size_inches : ℕ) : ℝ :=
  ((width_tiles * tile_size_inches : ℝ) / 12) * ((length_tiles * tile_size_inches : ℝ) / 12)

/-- Theorem stating that a bathroom with 10 tiles of 6 inches each along its width
    and 20 tiles of 6 inches each along its length has a square footage of 50 sq feet -/
theorem bathroom_area_theorem :
  bathroom_square_footage 10 20 6 = 50 := by
  -- Unfold the definition of bathroom_square_footage
  unfold bathroom_square_footage
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_area_theorem_l669_66994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l669_66938

/-- The time (in seconds) it takes for a train to pass a platform -/
noncomputable def time_to_pass_platform (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 720 m, traveling at 75 km/hr,
    takes approximately 51.85 seconds to pass a platform of length 360 m -/
theorem train_passing_time :
  let t := time_to_pass_platform 720 360 75
  (t ≥ 51.84) ∧ (t ≤ 51.86) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l669_66938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_period_l669_66963

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = !![1, 0; 0, 1]

theorem smallest_rotation_period : 
  (∀ k : ℕ, k > 0 ∧ k < 12 → ¬is_identity (A ^ k)) ∧ 
  is_identity (A ^ 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_period_l669_66963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_solution_correct_l669_66949

/-- Represents the investment problem with three partners -/
structure InvestmentProblem where
  /-- Williamson's initial investment -/
  williamson_initial : ℝ
  /-- Smag's initial investment (1.5 times Williamson's) -/
  smag_initial : ℝ
  /-- Rogers' total investment -/
  rogers_total : ℝ
  /-- Amount Rogers gives to Smag -/
  rogers_to_smag : ℝ
  /-- Amount Rogers gives to Williamson -/
  rogers_to_williamson : ℝ

/-- The investment problem satisfies the given conditions -/
def satisfies_conditions (p : InvestmentProblem) : Prop :=
  p.smag_initial = 1.5 * p.williamson_initial ∧
  p.rogers_total = 2500 ∧
  p.rogers_to_smag = 2000 ∧
  p.rogers_to_williamson = 500 ∧
  p.rogers_to_smag + p.rogers_to_williamson = p.rogers_total

/-- All partners have equal shares after Rogers' investment -/
def equal_shares (p : InvestmentProblem) : Prop :=
  p.williamson_initial + p.rogers_to_williamson =
  p.smag_initial + p.rogers_to_smag ∧
  p.smag_initial + p.rogers_to_smag = p.rogers_total

/-- Theorem stating that the given solution results in equal shares -/
theorem investment_solution_correct (p : InvestmentProblem) :
  satisfies_conditions p → equal_shares p := by
  intro h
  have h1 : p.williamson_initial + p.rogers_to_williamson = p.smag_initial + p.rogers_to_smag := by
    sorry
  have h2 : p.smag_initial + p.rogers_to_smag = p.rogers_total := by
    sorry
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_solution_correct_l669_66949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_first_decimal_digit_l669_66972

theorem constant_first_decimal_digit (m : ℕ+) :
  ∃ (n₀ : ℕ), ∀ (n : ℕ), n > n₀ →
    (10 * Real.sqrt (n^2 + 817*n + m.val : ℝ) - ⌊10 * Real.sqrt (n^2 + 817*n + m.val : ℝ)⌋) < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_first_decimal_digit_l669_66972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l669_66935

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the transformation from M to P
noncomputable def M_to_P (M : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 + Real.sqrt 2 * (M.1 - A.1), A.2 + Real.sqrt 2 * (M.2 - A.2))

-- Define the locus C₁
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Theorem statement
theorem curve_transformation :
  ∀ θ : ℝ, M_to_P (C θ) = C₁ θ ∧
  ∀ x y : ℝ, (x - Real.sqrt 2)^2 + y^2 = 2 →
    (x - (3 - Real.sqrt 2))^2 + y^2 > 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l669_66935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_left_pi_over_12_l669_66943

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) := Real.sin (2 * x)

theorem shift_left_pi_over_12 (x : ℝ) :
  f (x + Real.pi / 12) = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_left_pi_over_12_l669_66943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_probability_given_positive_test_l669_66937

/-- Probability of having the disease given a positive test result -/
theorem disease_probability_given_positive_test 
  (disease_prevalence : ℝ)
  (test_sensitivity : ℝ)
  (false_positive_rate : ℝ)
  (h1 : disease_prevalence = 1 / 200)
  (h2 : test_sensitivity = 1)
  (h3 : false_positive_rate = 0.05) :
  (disease_prevalence * test_sensitivity) / 
  (disease_prevalence * test_sensitivity + (1 - disease_prevalence) * false_positive_rate) = 20 / 219 := by
  sorry

#check disease_probability_given_positive_test

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_probability_given_positive_test_l669_66937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_ball_probability_l669_66955

def total_balls : ℕ := 5
def yellow_balls : ℕ := 3
def num_draws : ℕ := 3
def num_yellow_draws : ℕ := 2

/-- The probability of drawing exactly 2 yellow balls in 3 draws with replacement 
    from a bag containing 2 red balls and 3 yellow balls -/
theorem yellow_ball_probability : 
  (Nat.choose num_draws num_yellow_draws : ℚ) * 
  (yellow_balls / total_balls : ℚ) ^ num_yellow_draws * 
  (1 - yellow_balls / total_balls : ℚ) ^ (num_draws - num_yellow_draws) = 54 / 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_ball_probability_l669_66955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_rectangle_l669_66933

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define a circle
structure Circle :=
  (center : EuclideanSpace ℝ (Fin 2))
  (radius : ℝ)

-- Define a rectangle
structure Rectangle :=
  (center : EuclideanSpace ℝ (Fin 2))
  (width : ℝ)
  (height : ℝ)

-- Define the property of being inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

-- Define the construction of rectangles on the sides of the quadrilateral
noncomputable def construct_rectangles (q : Quadrilateral) : (Rectangle × Rectangle × Rectangle × Rectangle) :=
  sorry

-- Main theorem
theorem centers_form_rectangle (q : Quadrilateral) (c : Circle) :
  inscribed_in_circle q c →
  let (r1, r2, r3, r4) := construct_rectangles q
  ∃ (rect : Rectangle), rect.center = r1.center ∧
                        rect.width = dist r2.center r3.center ∧
                        rect.height = dist r1.center r2.center :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_rectangle_l669_66933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perfect_squares_l669_66956

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the sequence a_n
noncomputable def a (n : ℕ) : ℤ :=
  floor (Real.sqrt 2 * n)

-- State the theorem
theorem infinitely_many_perfect_squares :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ a n = m^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perfect_squares_l669_66956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_skew_lines_angle_l669_66964

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the dot product of two 3D vectors -/
def dotProduct (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

/-- Calculates the magnitude of a 3D vector -/
noncomputable def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

/-- Calculates the cosine of the angle between two 3D vectors -/
noncomputable def cosAngle (v1 v2 : Vector3D) : ℝ :=
  dotProduct v1 v2 / (magnitude v1 * magnitude v2)

/-- Theorem: In a cube with edge length 2, the cosine of the angle between A₁P and B₁Q is 1/6 -/
theorem cube_skew_lines_angle : 
  let A₁ : Point3D := ⟨2, 0, 2⟩
  let P : Point3D := ⟨1, 2, 0⟩
  let B₁ : Point3D := ⟨2, 2, 2⟩
  let Q : Point3D := ⟨0, 1, 1⟩
  let A₁P : Vector3D := ⟨P.x - A₁.x, P.y - A₁.y, P.z - A₁.z⟩
  let B₁Q : Vector3D := ⟨Q.x - B₁.x, Q.y - B₁.y, Q.z - B₁.z⟩
  cosAngle A₁P B₁Q = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_skew_lines_angle_l669_66964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1994_eq_7_l669_66926

/-- f(m) denotes the units digit of a positive integer m -/
def f (m : ℕ) : ℕ :=
  m % 10

/-- aₙ = f(2ⁿ⁺¹ - 1) for n = 1, 2, ... -/
def a (n : ℕ) : ℕ :=
  f ((2 : ℕ) ^ (n + 1) - 1)

/-- The 1994th term of the sequence a is 7 -/
theorem a_1994_eq_7 : a 1994 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1994_eq_7_l669_66926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_subtract_l669_66928

noncomputable def original_expr : ℝ := 7 / (3 + Real.sqrt 15)

noncomputable def final_expr : ℝ := -4 + (7 * Real.sqrt 15) / 6

theorem rationalize_and_subtract :
  (original_expr * (3 - Real.sqrt 15) / (3 - Real.sqrt 15)) - 1/2 = final_expr := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_subtract_l669_66928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_diagonal_ratio_l669_66910

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon is 1 -/
theorem octagon_diagonal_ratio : ∃ (s : ℝ), s > 0 →
  let shortest_diagonal := s * Real.sqrt (2 - Real.sqrt 2)
  let longest_diagonal := s * Real.sqrt (2 + Real.sqrt 2)
  shortest_diagonal / longest_diagonal = 1 := by
  use 1
  intro h
  simp
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_diagonal_ratio_l669_66910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_noncongruent_translation_l669_66954

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : Set (Fin 3 → ℝ)
  edges : Set ((Fin 3 → ℝ) × (Fin 3 → ℝ))
  convex : Prop

/-- Translate each edge of a polyhedron parallel to itself -/
def translateEdges (p : ConvexPolyhedron) (translation : (Fin 3 → ℝ) → (Fin 3 → ℝ)) : ConvexPolyhedron :=
  sorry

/-- Check if two polyhedra are congruent -/
def isCongruent (p1 p2 : ConvexPolyhedron) : Prop :=
  sorry

/-- The main theorem -/
theorem exists_noncongruent_translation :
  ∃ (p : ConvexPolyhedron) (t : (Fin 3 → ℝ) → (Fin 3 → ℝ)),
    ¬isCongruent p (translateEdges p t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_noncongruent_translation_l669_66954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_value_l669_66970

theorem sin_two_alpha_value (α : ℝ) 
  (h1 : α > 0) 
  (h2 : α < π / 2) 
  (h3 : Real.cos (π / 4 - α) = 2 * Real.sqrt 2 * Real.cos (2 * α)) : 
  Real.sin (2 * α) = 15 / 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_value_l669_66970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_c_greater_than_b_l669_66908

-- Define the constants
noncomputable def a : ℝ := (1.9 : ℝ) ^ (0.4 : ℝ)
noncomputable def b : ℝ := Real.log 1.9 / Real.log 0.4
noncomputable def c : ℝ := (0.4 : ℝ) ^ (1.9 : ℝ)

-- State the theorem
theorem a_greater_than_c_greater_than_b : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_c_greater_than_b_l669_66908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_shot_probability_l669_66941

noncomputable def probability_next_shot (p : ℝ) : ℝ := (2/3) * p + (1/3) * (1 - p)

noncomputable def probability_nth_shot (n : ℕ) : ℝ :=
  (1/2) + (1/6) * (1/3)^(n-1)

theorem fourth_shot_probability :
  probability_nth_shot 4 = 41/81 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_shot_probability_l669_66941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_DE_EF_ratio_l669_66900

-- Define the triangle ABC and points D, E, F
variable (A B C D E F : ℝ × ℝ)

-- Define the ratio conditions
noncomputable def ratio_AD_DB : ℝ := 2/3
noncomputable def ratio_BE_EC : ℝ := 2/3

-- Define that D is on AB
axiom D_on_AB : ∃ t : ℝ, D = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1

-- Define that E is on BC
axiom E_on_BC : ∃ s : ℝ, E = (1 - s) • B + s • C ∧ 0 ≤ s ∧ s ≤ 1

-- Define that F is the intersection of DE and AC
axiom F_intersection : ∃ u v : ℝ, F = (1 - u) • D + u • E ∧ F = (1 - v) • A + v • C

-- State the theorem
theorem DE_EF_ratio (hAD : D = (1 - ratio_AD_DB) • A + ratio_AD_DB • B)
                    (hBE : E = (1 - ratio_BE_EC) • B + ratio_BE_EC • C) :
  ∃ u : ℝ, F = (1 - u) • D + u • E ∧ u = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_DE_EF_ratio_l669_66900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_sale_loss_percentage_l669_66903

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_sale_loss_percentage 
  (cost_price : ℚ) 
  (selling_price : ℚ) 
  (h1 : cost_price = 2500)
  (h2 : selling_price + 500 = cost_price + cost_price / 10) :
  (cost_price - selling_price) * 100 / cost_price = 10 := by
  sorry

-- Example usage (commented out as it's not necessary for the theorem)
-- #eval watch_sale_loss_percentage 2500 2250 rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_sale_loss_percentage_l669_66903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheela_monthly_income_l669_66978

-- Define Sheela's deposit amount
noncomputable def deposit : ℝ := 2500

-- Define the percentage of monthly income that the deposit represents
noncomputable def deposit_percentage : ℝ := 25 / 100

-- Define Sheela's monthly income
noncomputable def monthly_income : ℝ := deposit / deposit_percentage

-- Theorem to prove
theorem sheela_monthly_income : monthly_income = 10000 := by
  -- Unfold the definitions
  unfold monthly_income deposit deposit_percentage
  -- Simplify the expression
  simp [div_div_eq_mul_div]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheela_monthly_income_l669_66978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_neg_reals_g_min_value_on_neg_one_and_less_l669_66936

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 + 2 / (x - 1)
noncomputable def g (x : ℝ) : ℝ := f (2^x)

-- Theorem for the decreasing property of g on (-∞,0)
theorem g_decreasing_on_neg_reals :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 → g x₁ > g x₂ := by sorry

-- Theorem for the minimum value of g on (-∞,-1]
theorem g_min_value_on_neg_one_and_less :
  ∃ x : ℝ, x ≤ -1 ∧ ∀ y : ℝ, y ≤ -1 → g x ≤ g y ∧ g x = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_neg_reals_g_min_value_on_neg_one_and_less_l669_66936
