import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_matrix_l1085_108543

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -6; 4, -1]

theorem eigenvalues_of_matrix (k : ℝ) :
  (∃ v : Fin 2 → ℝ, v ≠ 0 ∧ A.mulVec v = k • v) ↔ k = 6 ∨ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_matrix_l1085_108543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_above_g_implies_a_greater_than_two_l1085_108566

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := Real.exp (Real.log 3 * |x - 1|) - 2*x + a
def g (x : ℝ) : ℝ := 2 - x^2

-- State the theorem
theorem f_above_g_implies_a_greater_than_two (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 3, f a x > g x) → a > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_above_g_implies_a_greater_than_two_l1085_108566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l1085_108556

noncomputable section

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop :=
  y^2 / 9 - x^2 / 16 = 1

/-- The asymptotic line equation -/
def asymptotic_line (x y : ℝ) : Prop :=
  3 * x + 4 * y = 0

/-- The distance formula from a point to a line -/
def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The theorem statement -/
theorem distance_to_asymptote :
  distance_point_to_line 0 3 3 4 0 = 12 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l1085_108556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2x_plus_2cos_x_minus_3_l1085_108504

theorem max_value_sin_2x_plus_2cos_x_minus_3 :
  ∃ M : ℝ, M = -1 ∧ ∀ x : ℝ, Real.sin (2 * x) + 2 * Real.cos x - 3 ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2x_plus_2cos_x_minus_3_l1085_108504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_at_432_sum_of_digits_432_l1085_108511

-- Define the divisor counting function
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

-- Define the function f
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / (n.val : ℝ)^(1/4 : ℝ)

-- State the theorem
theorem max_f_at_432 : ∀ n : ℕ+, n ≠ 432 → f 432 > f n := by sorry

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of 432 is 9
theorem sum_of_digits_432 : sum_of_digits 432 = 9 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_at_432_sum_of_digits_432_l1085_108511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_positions_l1085_108532

theorem clock_hand_positions : ∀ θ : ℝ,
  (θ ≥ 0 ∧ θ < 360) →  -- Ensures angle is within valid range
  (∃ (final_θ : ℝ), final_θ = θ + 30) →  -- Hour hand moves 30° in 1 hour
  (∀ (t : ℝ), t ∈ Set.Icc 0 1 → ∃ (θ_t : ℝ), θ_t = θ + 30 * t) →  -- Continuous movement
  (∃ (bisector : ℝ), bisector = θ + 15 ∨ bisector = θ + 165) →  -- Minute hand bisects
  (θ = 15 ∨ θ = 165) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hand_positions_l1085_108532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_inequality_l1085_108575

/-- The smallest constant M that satisfies the inequality for all real a, b, c -/
noncomputable def M : ℝ := (9 / 32) * Real.sqrt 2

/-- The theorem stating that M is the smallest constant satisfying the inequality -/
theorem smallest_constant_inequality (a b c : ℝ) :
  |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 ∧
  ∀ M' : ℝ, (∀ x y z : ℝ, |x * y * (x^2 - y^2) + y * z * (y^2 - z^2) + z * x * (z^2 - x^2)| ≤ M' * (x^2 + y^2 + z^2)^2) →
  M ≤ M' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_inequality_l1085_108575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_l1085_108513

theorem cube_root_sum (p q r : ℕ+) 
  (h : 2 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = 
       Real.rpow p (1/3) + Real.rpow q (1/3) - Real.rpow r (1/3)) :
  p + q + r = 254 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_l1085_108513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trent_average_speed_l1085_108527

/-- Represents Trent's journey with given conditions -/
structure TrentJourney where
  block_length : ℚ
  walk_blocks : ℕ
  walk_speed : ℚ
  bus_blocks : ℕ
  bus_speed : ℚ
  bike_blocks : ℕ
  bike_speed : ℚ
  jog_blocks : ℕ
  jog_speed : ℚ
  total_time : ℚ

/-- Calculates the average speed of Trent's journey -/
def average_speed (journey : TrentJourney) : ℚ :=
  let total_distance := (journey.walk_blocks * 2 + journey.bus_blocks * 2 + journey.bike_blocks + journey.jog_blocks) * journey.block_length
  total_distance / (1000 * journey.total_time)

/-- Theorem stating that Trent's average speed is 0.8 km/h -/
theorem trent_average_speed :
  let journey := TrentJourney.mk 50 4 4 7 15 5 10 5 6 2
  average_speed journey = 4/5 := by
  sorry

#eval average_speed (TrentJourney.mk 50 4 4 7 15 5 10 5 6 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trent_average_speed_l1085_108527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_conference_center_l1085_108500

/-- Represents the problem of calculating the distance to the conference center -/
theorem distance_to_conference_center :
  ∃ (total_distance : ℝ),
    total_distance = 191.25 ∧
    ∃ (t initial_speed speed_increase time_late time_early first_hour_distance : ℝ),
      initial_speed = 45 ∧
      speed_increase = 20 ∧
      time_late = 0.75 ∧
      time_early = 0.25 ∧
      first_hour_distance = 45 ∧
      total_distance = initial_speed * (t + time_late) ∧
      total_distance - first_hour_distance = (initial_speed + speed_increase) * (t - 1 - time_early) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_conference_center_l1085_108500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_entry_exit_time_l1085_108528

/-- Represents the speed of the car in miles per minute -/
noncomputable def car_speed : ℝ := 2/3

/-- Represents the radius of the storm in miles -/
noncomputable def storm_radius : ℝ := 60

/-- Represents the speed of the storm in miles per minute -/
noncomputable def storm_speed : ℝ := 1

/-- Represents the initial north-south distance between the car and storm center in miles -/
noncomputable def initial_distance : ℝ := 130

/-- Represents the east-west position of the car at time t -/
noncomputable def car_position (t : ℝ) : ℝ := car_speed * t

/-- Represents the east-west position of the storm center at time t -/
noncomputable def storm_position_x (t : ℝ) : ℝ := storm_speed * t

/-- Represents the north-south position of the storm center at time t -/
noncomputable def storm_position_y (t : ℝ) : ℝ := initial_distance - storm_speed * t

/-- Theorem stating that the average time of entry and exit is 117 minutes -/
theorem average_entry_exit_time : 
  ∃ t₁ t₂ : ℝ, 
    (t₁ < t₂) ∧ 
    (∀ t, t₁ ≤ t ∧ t ≤ t₂ → 
      (car_position t - storm_position_x t)^2 + (storm_position_y t)^2 ≤ storm_radius^2) ∧
    (∀ t, t < t₁ ∨ t > t₂ → 
      (car_position t - storm_position_x t)^2 + (storm_position_y t)^2 > storm_radius^2) ∧
    ((t₁ + t₂) / 2 = 117) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_entry_exit_time_l1085_108528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_l1085_108541

/-- The area of a regular hexadecagon inscribed in a circle -/
theorem hexadecagon_area (r : ℝ) (r_pos : r > 0) : 
  ∃ (A : ℝ), A = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2) ∧ 
  A = 16 * (1/2 * r^2 * Real.sin (Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_l1085_108541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1085_108547

open Real

theorem trig_identities :
  (cos (70 * π / 180) * cos (80 * π / 180) - sin (70 * π / 180) * sin (80 * π / 180) = -sqrt 3 / 2) ∧
  (4 * tan (π / 8) / (1 - tan (π / 8) ^ 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1085_108547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1085_108563

def team_sizes : List Nat := [6, 8, 9, 10]

def probability_both_captains (sizes : List Nat) : ℚ :=
  let total_prob := (sizes.map (λ n => (1 : ℚ) / (n * (n - 1)))).sum
  (1 : ℚ) / sizes.length * total_prob

theorem probability_theorem :
  probability_both_captains team_sizes = 131 / 5040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1085_108563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_l1085_108554

-- Define the constants a, b, c
variable (a b c : ℝ)

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x - a) * (x - b) / (x - c)

-- State the theorem
theorem solution_value :
  (∀ x, f a b c x ≤ 0 ↔ (x < -3 ∨ |x - 30| ≤ 2)) →
  a < b →
  a + 2*b + 3*c = 83 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_l1085_108554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l1085_108564

/-- Given a circle with center O and radius r, a fixed point A on its circumference,
    and a constant c² (where 0 < c² < 8r²), the locus of points F such that
    for some chord BC with F as its midpoint, AB² + AC² = c²,
    is the interior of a chord GH of the circle (excluding the endpoints). -/
theorem midpoint_locus (O A : ℝ × ℝ) (r c : ℝ) 
  (h_circle : ‖A - O‖ = r) 
  (h_c_bounds : 0 < c^2 ∧ c^2 < 8*r^2) : 
  ∃ G H : ℝ × ℝ, G ≠ H ∧ 
  ‖G - O‖ = r ∧ ‖H - O‖ = r ∧
  ∀ F : ℝ × ℝ, 
    (∃ B C : ℝ × ℝ, ‖B - O‖ = r ∧ ‖C - O‖ = r ∧ 
     F = (B + C) / 2 ∧ 
     ‖B - A‖^2 + ‖C - A‖^2 = c^2) ↔ 
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ F = G + t • (H - G)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l1085_108564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_sum_l1085_108530

-- Define the sequence a_n
def a : ℕ → ℝ
  | n => 2^n

-- Define the sequence b_n
noncomputable def b : ℕ → ℝ
  | n => a n * Real.log (a n) / Real.log 2

-- Define the partial sum S_n
noncomputable def S : ℕ → ℝ
  | n => (Finset.range n).sum (λ i => b (i + 1))

theorem geometric_sequence_and_sum :
  (∀ n : ℕ, a (n + 1) > a n) ∧  -- monotonically increasing
  a 2 + a 3 + a 4 = 28 ∧  -- sum condition
  a 3 + 2 = (a 2 + a 4) / 2 ∧  -- arithmetic mean condition
  (∀ n : ℕ, a n = 2^n) ∧  -- general formula
  (∀ n : ℕ, n < 5 → S n + n * 2^(n+1) ≤ 50) ∧  -- S_n condition for n < 5
  S 5 + 5 * 2^6 > 50  -- S_n condition for n = 5
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_sum_l1085_108530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_period_l1085_108517

/-- Calculates the number of years for a compound interest investment -/
noncomputable def calculate_years (principal : ℝ) (rate : ℝ) (interest : ℝ) : ℝ :=
  let total := principal + interest
  (Real.log (total / principal)) / (Real.log (1 + rate))

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem compound_interest_period (principal rate interest : ℝ) 
  (h_principal : principal = 6000)
  (h_rate : rate = 0.15)
  (h_interest : interest = 2331.75) :
  round_to_nearest (calculate_years principal rate interest) = 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval round_to_nearest (calculate_years 6000 0.15 2331.75)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_period_l1085_108517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_six_l1085_108526

/-- Given a parabola and a hyperbola intersecting at two points, with one point being the focus of the parabola, 
    if these three points form a right triangle, then the eccentricity of the hyperbola is √6. -/
theorem hyperbola_eccentricity_sqrt_six 
  (x y a : ℝ) 
  (h_parabola : y^2 = 4*x)
  (h_hyperbola : x^2/a^2 - y^2 = 1)
  (h_a_pos : a > 0)
  (A B : ℝ × ℝ) 
  (h_intersect : (y^2 = 4*x ∧ x^2/a^2 - y^2 = 1) → (A ∈ Set.univ.prod Set.univ ∧ 
                                                    B ∈ Set.univ.prod Set.univ))
  (F : ℝ × ℝ)
  (h_focus : F = (1, 0))
  (h_right_triangle : (A.1 - F.1)^2 + (A.2 - F.2)^2 + (B.1 - F.1)^2 + (B.2 - F.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2) :
  Real.sqrt ((a^2 + 1)/a^2) = Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_six_l1085_108526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_relation_l1085_108514

/-- Predicate stating that three points form a triangle -/
def IsTriangle (A B C : EuclideanPlane) : Prop := sorry

/-- Predicate stating that DEF is the orthic triangle of ABC -/
def IsOrthicTriangle (D E F A B C : EuclideanPlane) : Prop := sorry

/-- Predicate stating that the circumradius of triangle ABC is equal to R -/
def CircumradiusEq (A B C : EuclideanPlane) (R : ℝ) : Prop := sorry

/-- Given a triangle ABC with circumradius R and its orthic triangle DEF with circumradius R',
    prove that R = 2R'. -/
theorem circumradius_relation (A B C D E F : EuclideanPlane) (R R' : ℝ) : 
  IsTriangle A B C → 
  IsOrthicTriangle D E F A B C → 
  CircumradiusEq A B C R → 
  CircumradiusEq D E F R' → 
  R = 2 * R' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_relation_l1085_108514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_time_l1085_108503

/-- Represents the time (in days) it takes for a worker to complete a job alone -/
structure WorkerTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the rate at which a worker completes a job (fraction per day) -/
noncomputable def workerRate (w : WorkerTime) : ℝ := 1 / w.days

/-- Represents the total earnings for a job -/
structure JobEarnings where
  total : ℝ
  total_pos : total > 0

/-- Represents a worker's share of the total earnings -/
structure WorkerShare where
  amount : ℝ
  amount_pos : amount > 0

/-- Theorem stating that worker c takes 12 days to complete the job alone -/
theorem worker_c_time
  (worker_a : WorkerTime)
  (worker_b : WorkerTime)
  (job : JobEarnings)
  (share_b : WorkerShare)
  (h1 : worker_a.days = 6)
  (h2 : worker_b.days = 8)
  (h3 : job.total = 1170)
  (h4 : share_b.amount = 390)
  (h5 : share_b.amount / job.total = workerRate worker_b / (workerRate worker_a + workerRate worker_b + 1 / 12)) :
  ∃ (worker_c : WorkerTime), worker_c.days = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_time_l1085_108503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pst_measure_l1085_108578

-- Define the triangle PQR
structure Triangle (P Q R : Point) : Prop where
  exists_triangle : ∃ (p q r : Point), p = P ∧ q = Q ∧ r = R

-- Define the angle measure
noncomputable def angle_measure (A B C : Point) : ℝ := sorry

-- Define the bisector of an angle
def is_angle_bisector (A B C X : Point) : Prop :=
  angle_measure A B X = angle_measure X B C

-- Define a point on a line segment
def point_on_segment (A B X : Point) : Prop := sorry

-- Define the statement
theorem angle_pst_measure 
  (P Q R S T : Point) 
  (triangle : Triangle P Q R) 
  (angle_pqr : angle_measure P Q R = 60) 
  (ps_bisects : is_angle_bisector Q P R S) 
  (sr_bisects : is_angle_bisector Q R P S) 
  (t_on_qr : point_on_segment Q R T) 
  (st_bisects : is_angle_bisector Q S R T) : 
  angle_measure P S T = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pst_measure_l1085_108578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1085_108512

theorem complex_modulus (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l1085_108512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_value_l1085_108536

theorem cos_two_alpha_value (α : Real) 
  (h1 : Real.sin (α + Real.pi / 2) = -Real.sqrt 5 / 5)
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_value_l1085_108536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_rate_l1085_108505

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the volume of a rectangular tank -/
noncomputable def tankVolume (d : TankDimensions) : ℝ :=
  d.length * d.width * d.depth

/-- Calculates the fill rate of a tank -/
noncomputable def fillRate (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

/-- Theorem: The fill rate of the given tank is 5 cubic feet per hour -/
theorem tank_fill_rate :
  let dimensions : TankDimensions := ⟨10, 6, 5⟩
  let volume : ℝ := tankVolume dimensions
  let time : ℝ := 60
  fillRate volume time = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_rate_l1085_108505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l1085_108568

-- Define the original function
def f : ℝ → ℝ := sorry

-- Define the transformed function
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Define horizontal compression
def horizontalCompress (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (k * x)

-- Define horizontal shift
def horizontalShift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x + h)

-- Theorem stating the equivalence of the transformations
theorem transform_equivalence :
  ∀ x : ℝ, g x = (horizontalShift (horizontalCompress f 2) (-1/2)) x :=
by
  intro x
  simp [g, horizontalShift, horizontalCompress]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l1085_108568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_with_given_conditions_l1085_108583

/-- Calculates the total distance that can be driven given fuel efficiency, gas cost, and amount of money. -/
noncomputable def total_distance (fuel_efficiency : ℝ) (gas_cost : ℝ) (money : ℝ) : ℝ :=
  (money / gas_cost) * fuel_efficiency

/-- Theorem: Given the specified conditions, the total distance that can be driven is 200 miles. -/
theorem distance_with_given_conditions :
  let fuel_efficiency : ℝ := 40
  let gas_cost : ℝ := 5
  let money : ℝ := 25
  total_distance fuel_efficiency gas_cost money = 200 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_with_given_conditions_l1085_108583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integers_product_sum_l1085_108584

theorem distinct_integers_product_sum (m n p q : ℕ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integers_product_sum_l1085_108584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1085_108570

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > 0
  k : b > 0
  l : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The focus of a parabola y = ax² + bx + c -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ := 
  (- b / (2 * a), (1 - b^2) / (4 * a) - c)

/-- The theorem stating the equation of the ellipse under given conditions -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.eccentricity = Real.sqrt 3 / 2)
  (h2 : ∃ (x y : ℝ), (x, y) = parabola_focus (-1/(4*Real.sqrt 3)) 0 0 ∧ 
                     (x^2 / e.a^2 + y^2 / e.b^2 = 1)) :
  e.a = 2 ∧ e.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1085_108570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_is_reals_except_one_l1085_108577

-- Define the function f(x) = 1 / (x - 1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 1} := by
  sorry

-- Additional theorem to show that the domain is equivalent to (-∞, 1) ∪ (1, +∞)
theorem domain_is_reals_except_one :
  {x : ℝ | x < 1 ∨ x > 1} = {x : ℝ | x ≠ 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_is_reals_except_one_l1085_108577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_is_2012_l1085_108591

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Query function that returns the sum of digits of |x - a| -/
def query (a x : ℕ) : ℕ :=
  sumOfDigits (Int.natAbs (x - a))

/-- The set of all natural numbers with sum of digits 2012 -/
def S : Set ℕ :=
  {a | sumOfDigits a = 2012}

/-- The minimal number of queries needed to determine any number in S -/
noncomputable def minQueries : ℕ :=
  sorry

theorem min_queries_is_2012 : minQueries = 2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_queries_is_2012_l1085_108591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_repetition_in_2_power_1970_l1085_108549

theorem digit_repetition_in_2_power_1970 :
  ∃ (d : ℕ) (i j : ℕ), i ≠ j ∧ d < 10 ∧
  (∃ (sequence : ℕ → ℕ),
    sequence 0 = 2^1970 ∧
    (∀ n : ℕ, sequence (n + 1) = (sequence n % 10^(Nat.digits 10 (sequence n)).length) + 
                                 (sequence n / 10^(Nat.digits 10 (sequence n)).length)) ∧
    (∃ (k : ℕ), sequence k < 10 ∧ sequence i = d ∧ sequence j = d)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_repetition_in_2_power_1970_l1085_108549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_range_l1085_108560

/-- A piecewise function f defined on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then a * x - 3 else -x^2 + 2*x - 7

/-- The property of a function being monotonic on ℝ. -/
def IsMonotonic (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → g x ≤ g y ∨ (∀ x y : ℝ, x ≤ y → g x ≥ g y)

/-- The main theorem stating the range of 'a' for which f is monotonic. -/
theorem f_monotonic_range :
  {a : ℝ | IsMonotonic (f a)} = Set.Ici (-2) ∩ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_range_l1085_108560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_with_triple_volume_l1085_108565

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem sphere_diameter_with_triple_volume (r : ℝ) (h : r = 6) :
  let v := sphere_volume r
  let new_r := (3 * v / ((4/3) * Real.pi))^(1/3)
  2 * new_r = 12 * Real.rpow 2 (1/3) :=
by
  -- Placeholder for the proof
  sorry

#eval (12 : ℕ) + (2 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_with_triple_volume_l1085_108565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_difference_l1085_108592

noncomputable section

-- Define the points A, B, C
def A : ℝ × ℝ := (-1, 7)
def B : ℝ × ℝ := (4, -2)
def C : ℝ × ℝ := (9, -2)

-- Define the line AC
def line_AC (x : ℝ) : ℝ := 
  (7 - (-2)) / ((-1) - 9) * (x - (-1)) + 7

-- Define the point R
def R : ℝ × ℝ := (3, 4)

-- Define the point S
def S : ℝ × ℝ := (3, -2)

-- State the theorem
theorem triangle_coordinates_difference : 
  -- R is on line AC
  line_AC R.1 = R.2 ∧
  -- S is on line BC (which is horizontal at y = -2)
  S.2 = -2 ∧
  -- RS is vertical (same x-coordinate)
  R.1 = S.1 ∧
  -- Area of triangle RSC is 18
  (1/2 : ℝ) * (R.2 - S.2) * (C.1 - S.1) = 18 →
  -- The positive difference between x and y coordinates of R is 1
  |R.2 - R.1| = 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_difference_l1085_108592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_headed_frogs_count_l1085_108569

/-- Represents the number of frogs with a specific characteristic --/
structure FrogCount where
  count : Nat

/-- Represents the total number of frogs caught --/
def TotalFrogs (extra_legs bright_red normal two_heads : FrogCount) : Nat :=
  extra_legs.count + bright_red.count + normal.count + two_heads.count

/-- Represents the number of mutated frogs --/
def MutatedFrogs (extra_legs bright_red two_heads : FrogCount) : Nat :=
  extra_legs.count + bright_red.count + two_heads.count

/-- Theorem stating that the number of frogs with two heads is 2 --/
theorem two_headed_frogs_count :
  ∀ (two_heads : FrogCount),
    let extra_legs : FrogCount := ⟨5⟩
    let bright_red : FrogCount := ⟨2⟩
    let normal : FrogCount := ⟨18⟩
    let total := TotalFrogs extra_legs bright_red normal two_heads
    let mutated := MutatedFrogs extra_legs bright_red two_heads
    (mutated : Rat) / total = 33 / 100 →
    two_heads = ⟨2⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_headed_frogs_count_l1085_108569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l1085_108501

theorem product_approximation : ∃ ε : ℝ, ε > 0 ∧ |3.05 * 7.95 * (6.05 + 3.95) - 240| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l1085_108501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_k_sum_correct_l1085_108520

/-- The sum of integer values of k for which the area of the triangle with vertices 
    (2,10), (14,19), and (6,k) is minimum -/
def min_area_k_sum : ℤ := 26

/-- Point type for 2D coordinates -/
structure Point where
  x : ℚ
  y : ℚ

/-- Define the three points of the triangle -/
def p1 : Point := ⟨2, 10⟩
def p2 : Point := ⟨14, 19⟩
def p3 (k : ℤ) : Point := ⟨6, k⟩

/-- Function to calculate the area of a triangle given three points -/
def triangle_area (a b c : Point) : ℚ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

/-- Theorem stating that the sum of k values for minimum area is correct -/
theorem min_area_k_sum_correct : 
  ∃ (k1 k2 : ℤ), k1 ≠ k2 ∧ 
  (∀ (k : ℤ), triangle_area p1 p2 (p3 k) ≥ triangle_area p1 p2 (p3 k1)) ∧
  (∀ (k : ℤ), triangle_area p1 p2 (p3 k) ≥ triangle_area p1 p2 (p3 k2)) ∧
  k1 + k2 = min_area_k_sum :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_k_sum_correct_l1085_108520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1085_108535

noncomputable def geometricSequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => a₁ * r ^ n

noncomputable def geometricSum (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (a₁ * (1 - r^n)) / (1 - r)

noncomputable def geometricProduct (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁^n * r^((n * (n - 1)) / 2)

theorem geometric_sequence_first_term
  (a₁ : ℝ) (r : ℝ) :
  (geometricProduct a₁ r 5 = 243) →
  (geometricSum a₁ r 3 = geometricSequence a₁ r 1 + 4 * a₁) →
  a₁ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1085_108535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_sum_bound_l1085_108506

/-- Triangle with sides a ≥ b ≥ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a ≥ b
  hb : b ≥ c
  hc : c > 0

/-- Point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point := sorry

/-- Intersection of line from centroid to vertex with opposite side -/
noncomputable def intersection (t : Triangle) (p : Point) (v : Point) : Point := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Sum of distances from centroid intersections to vertices -/
noncomputable def s (t : Triangle) : ℝ :=
  let p := centroid t
  let a := Point.mk 0 0
  let b := Point.mk t.c 0
  let c := Point.mk ((t.a * t.a + t.c * t.c - t.b * t.b) / (2 * t.c)) 
                    (Real.sqrt (t.a * t.a - ((t.a * t.a + t.c * t.c - t.b * t.b) / (2 * t.c))^2))
  let a' := intersection t p b
  let b' := intersection t p c
  let c' := intersection t p a
  distance a a' + distance b b' + distance c c'

theorem centroid_intersection_sum_bound (t : Triangle) :
  s t < 2/3 * (t.a + t.b + t.c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_sum_bound_l1085_108506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_twelfth_power_l1085_108548

theorem fourth_root_sixteen_twelfth_power : (16 : ℝ) ^ ((1/4 : ℝ) * 12) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_twelfth_power_l1085_108548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_T_l1085_108546

/-- Curve C in parametric form -/
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (4 * Real.cos α, 4 * Real.sin α)

/-- Curve T in polar form -/
def curve_T (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin θ + ρ * Real.cos θ = 20

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) : ℝ := 
  |4 * x + 2 * y - 20| / Real.sqrt 5

/-- Maximum distance from curve C to curve T -/
theorem max_distance_C_to_T : 
  ∀ α : ℝ, distance_point_to_line (curve_C α).1 (curve_C α).2 ≤ 2 + 2 * Real.sqrt 5 :=
by
  sorry

#check max_distance_C_to_T

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_T_l1085_108546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1085_108576

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- The main theorem to prove -/
theorem ellipse_equation (f1 f2 p : Point) (e : Ellipse) : 
  f1 = Point.mk 3 3 →
  f2 = Point.mk 3 7 →
  p = Point.mk 15 (-4) →
  e.a > 0 →
  e.b > 0 →
  isOnEllipse e p →
  distance f1 p + distance f2 p = 2 * e.a →
  e = Ellipse.mk (Real.sqrt 221) 15 3 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1085_108576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_from_48_numbers_l1085_108552

/-- Given a set of 48 natural numbers whose product has exactly 10 different prime factors,
    there exist 4 numbers in this set whose product is a perfect square. -/
theorem perfect_square_from_48_numbers (S : Finset ℕ) 
  (h1 : S.card = 48)
  (h2 : (S.prod id).factorization.support.card = 10) :
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a * b * c * d = (a * b * c * d).sqrt ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_from_48_numbers_l1085_108552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1085_108586

/-- The speed of a train given its length and time to pass a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A 200-meter long train that crosses a stationary point in 9 seconds
    has a speed of approximately 22.22 meters per second -/
theorem train_speed_calculation :
  let speed := train_speed 200 9
  ∃ ε > 0, abs (speed - 22.22) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1085_108586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_theorem_l1085_108523

/-- Represents a grid with circles on top -/
structure GridWithCircles where
  gridSize : Nat
  squareSize : ℝ
  numCircles : Nat

/-- Calculates the total area of the grid -/
def totalGridArea (g : GridWithCircles) : ℝ :=
  (g.gridSize * g.squareSize) ^ 2

/-- Calculates the area of a single circle -/
noncomputable def circleArea (g : GridWithCircles) : ℝ :=
  Real.pi * (g.squareSize * 2) ^ 2 / 4

/-- Calculates the total area of all circles -/
noncomputable def totalCircleArea (g : GridWithCircles) : ℝ :=
  g.numCircles * circleArea g

/-- Theorem: For a 4x4 grid with 3cm squares and 4 circles, A + B = 180 -/
theorem grid_circle_theorem (g : GridWithCircles) 
    (h1 : g.gridSize = 4)
    (h2 : g.squareSize = 3)
    (h3 : g.numCircles = 4)
    (A B : ℝ)
    (h4 : totalGridArea g - totalCircleArea g = A - B * Real.pi) :
    A + B = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_theorem_l1085_108523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_exists_l1085_108582

/-- A line in a plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A region in a plane --/
inductive Region where
  | Triangle : Region
  | Quadrilateral : Region
  | Other : Region

/-- The set of regions formed by intersecting lines --/
def regions_from_lines (lines : List Line) : Set Region :=
  sorry -- Implementation details omitted for simplicity

/-- The number of lines intersecting in the plane --/
def num_lines : Nat := 4

/-- Theorem: When 4 lines intersect in a plane, at least one of the resulting regions is a quadrilateral --/
theorem quadrilateral_exists :
  ∃ (lines : List Line), lines.length = num_lines ∧
  ∃ (r : Region), r ∈ regions_from_lines lines ∧ r = Region.Quadrilateral :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_exists_l1085_108582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1085_108550

noncomputable def f (x : ℝ) : ℝ := (1 - x + Real.sqrt (2 * x^2 - 2 * x + 1)) / (2 * x)

theorem max_value_of_f :
  ∃ (m : ℝ), m = (Real.sqrt 5 - 1) / 4 ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x ≤ m) ∧
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 3 ∧ f x = m) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1085_108550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_roots_l1085_108508

/-- The quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The product of roots of a quadratic equation -/
noncomputable def productOfRoots (eq : QuadraticEquation) : ℝ := eq.c / eq.a

/-- The discriminant of a quadratic equation -/
noncomputable def discriminant (eq : QuadraticEquation) : ℝ := eq.b^2 - 4*eq.a*eq.c

/-- Theorem: The value of k that maximizes the product of roots of 6x^2 - 5x + k = 0,
    given that the roots are real, is 25/24 -/
theorem max_product_of_roots :
  ∃ (k : ℝ), ∀ (j : ℝ),
    let eq : QuadraticEquation := ⟨6, -5, k⟩
    let eq_j : QuadraticEquation := ⟨6, -5, j⟩
    discriminant eq ≥ 0 → discriminant eq_j ≥ 0 →
    productOfRoots eq ≥ productOfRoots eq_j ∧
    k = 25/24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_roots_l1085_108508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_area_relation_l1085_108573

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- A region in the xy-plane -/
structure Region where
  area : ℝ

/-- The problem statement -/
theorem circle_regions_area_relation (c : Circle) (l1 l2 : Line) (R1 R2 R3 R4 : Region) :
  c.center = (0, 0) →
  c.radius = 6 →
  l1 = ⟨1, 0, -4⟩ →  -- x = 4
  l2 = ⟨0, 1, -3⟩ →  -- y = 3
  R1.area > R2.area →
  R2.area > R3.area →
  R3.area > R4.area →
  R1.area - R2.area - R3.area + R4.area = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_area_relation_l1085_108573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_connection_bound_l1085_108587

/-- A graph representing schools and their connections -/
structure SchoolGraph where
  n : ℕ  -- number of schools (vertices)
  d : ℕ  -- number of connections per school (degree of each vertex)
  adj : Fin n → Finset (Fin n)  -- adjacency relation
  deg_eq : ∀ v, (adj v).card = d  -- each vertex has exactly d edges
  sym : ∀ v w, w ∈ adj v ↔ v ∈ adj w  -- symmetry of adjacency

/-- The main theorem: in a SchoolGraph, d < 2 * (n^(1/3)) -/
theorem school_connection_bound (G : SchoolGraph) : 
  (G.d : ℝ) < 2 * (G.n : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_connection_bound_l1085_108587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1085_108531

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Part I
theorem part_one (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) : 
  a = 2 := by sorry

-- Part II
theorem part_two (a : ℝ) (h1 : a > 1)
  (h2 : ∀ x ≤ 2, StrictMonoOn (fun y => -(f a y)) (Set.Iic x))
  (h3 : ∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → 
    |f a x₁ - f a x₂| ≤ 4) :
  2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1085_108531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_eleven_equals_negative_two_l1085_108515

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 7

noncomputable def g (x : ℝ) : ℝ := 
  let y := (x + 7) / 2  -- This is f⁻¹(x)
  3 * y^2 + 4 * y - 6

-- State the theorem
theorem g_of_negative_eleven_equals_negative_two : g (-11) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_eleven_equals_negative_two_l1085_108515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincidence_l1085_108518

/-- The value of b for a hyperbola x² - y²/b² = 1 when its right focus coincides with
    the focus of the parabola y² = 8x -/
theorem hyperbola_parabola_focus_coincidence :
  ∀ b : ℝ,
  b > 0 →
  (∃ (x y : ℝ), y^2 = 8*x ∧ x^2 - y^2/b^2 = 1) →
  (∀ (x y : ℝ), y^2 = 8*x → x = 2) →
  (∀ (x y : ℝ), x^2 - y^2/b^2 = 1 → x = Real.sqrt (1 + b^2)) →
  b = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincidence_l1085_108518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1085_108572

/-- A function satisfying f(xy) = f(x) + f(y) for positive reals and f(2) = 1 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) ∧ f 2 = 1

/-- The main theorem: if f satisfies the functional equation, then f(1/64) = -6 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f (1 / 64) = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1085_108572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1085_108581

-- Define the hyperbola and parallel lines
noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def line1 (a x y : ℝ) : Prop := y = x + a
def line2 (a x y : ℝ) : Prop := y = x - a

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
    hyperbola a b x1 y1 ∧ hyperbola a b x2 y2 ∧ hyperbola a b x3 y3 ∧ hyperbola a b x4 y4 ∧
    line1 a x1 y1 ∧ line1 a x2 y2 ∧ line2 a x3 y3 ∧ line2 a x4 y4 ∧
    (x2 - x1) * (y4 - y3) = 8 * b^2) →
  eccentricity a b = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1085_108581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_ln_intersection_points_exp_and_quadratic_exp_inequality_l1085_108580

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (x : ℝ) := Real.log x

theorem tangent_slope_of_ln (k : ℝ) :
  (∃ x, x > 0 ∧ g x = k * x + 2 ∧ 1 / x = k) →
  k = Real.exp (-3) := by
  sorry

theorem intersection_points_exp_and_quadratic (m : ℝ) :
  m > 0 →
  (∀ x > 0, f x = m * x^2 ↔ 
    (m < Real.exp 2 / 4 ∧ False) ∨
    (m = Real.exp 2 / 4 ∧ x = 2) ∨
    (m > Real.exp 2 / 4 ∧ (∃ y, y ≠ x ∧ y > 0 ∧ f y = m * y^2))) := by
  sorry

theorem exp_inequality (a b : ℝ) :
  a < b →
  (f a + f b) / 2 > (f b - f a) / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_ln_intersection_points_exp_and_quadratic_exp_inequality_l1085_108580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_properties_l1085_108539

noncomputable section

open Real

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
axiom side_angle_relation : c * sin A = Real.sqrt 3 * a * cos C
axiom side_relation : (a - c) * (a + c) = b * (b - c)

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin x * cos (Real.pi/2 - x) - 
                     Real.sqrt 3 * sin (Real.pi + x) * cos x + 
                     sin (Real.pi/2 + x) * cos x

-- State the theorem
theorem triangle_function_properties : 
  (∀ x, f (x + Real.pi) = f x) ∧ f B = 5/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_properties_l1085_108539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_three_fourths_l1085_108529

/-- A rectangle with length twice its width and midpoints on adjacent sides -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  length_eq_twice_width : length = 2 * width
  R : ℝ × ℝ
  S : ℝ × ℝ
  R_is_midpoint : R = (0, width / 2)
  S_is_midpoint : S = (width, length)

/-- The fraction of the rectangle's area that is shaded -/
noncomputable def shadedFraction (rect : SpecialRectangle) : ℝ :=
  3 / 4

theorem shaded_fraction_is_three_fourths (rect : SpecialRectangle) :
  shadedFraction rect = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_three_fourths_l1085_108529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_value_implies_n_l1085_108561

def t : ℕ → ℚ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | n + 2 => if (n + 2) % 2 = 0 then 1 + t ((n + 2) / 2) else 1 / t (n + 1)

theorem t_value_implies_n (n : ℕ) :
  t n = 7 / 29 → n = 1905 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_value_implies_n_l1085_108561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_triangle_perimeter_sum_l1085_108593

/-- Represents the side length of a triangle in the sequence --/
noncomputable def side_length (n : ℕ) : ℝ := 40 / 2^n

/-- Represents the perimeter of a triangle in the sequence --/
noncomputable def perimeter (n : ℕ) : ℝ := 3 * side_length n

/-- The sum of the perimeters of all triangles in the infinite sequence --/
def perimeter_sum : ℝ := 240

/-- Theorem stating that the sum of all perimeters equals perimeter_sum --/
theorem infinite_triangle_perimeter_sum : 
  (∑' n, perimeter n) = perimeter_sum := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_triangle_perimeter_sum_l1085_108593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1085_108594

/-- A line in the form y = kx + 1 -/
structure Line where
  k : ℝ

/-- A circle in the form (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Two points on a plane -/
structure TwoPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p : TwoPoints) : ℝ :=
  Real.sqrt ((p.A.1 - p.B.1)^2 + (p.A.2 - p.B.2)^2)

/-- Check if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.k * p.1 + 1

/-- Check if a point is on a circle -/
def pointOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.h)^2 + (p.2 - c.k)^2 = c.r^2

/-- Theorem: If a line y = kx + 1 intersects a circle x^2 + y^2 - 2x - 2y + 1 = 0
    at two points A and B such that |AB| = √2, then k = ±1 -/
theorem line_circle_intersection
  (l : Line)
  (c : Circle)
  (p : TwoPoints)
  (h1 : c.h = 1 ∧ c.k = 1 ∧ c.r = 1)  -- Circle equation: (x-1)^2 + (y-1)^2 = 1
  (h2 : distance p = Real.sqrt 2)     -- |AB| = √2
  (h3 : pointOnLine p.A l ∧ pointOnCircle p.A c)  -- A is on both line and circle
  (h4 : pointOnLine p.B l ∧ pointOnCircle p.B c)  -- B is on both line and circle
  : l.k = 1 ∨ l.k = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1085_108594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_walked_other_days_l1085_108510

/-- Represents the number of dogs walked on a given day -/
def DogsWalked : Type := Nat

/-- Represents the earnings in dollars -/
def Earnings : Type := Nat

/-- The rate per dog in dollars -/
def rate : Nat := 5

/-- The total number of days Harry walks dogs -/
def total_days : Nat := 5

/-- The number of dogs walked on Tuesday -/
def tuesday_dogs : Nat := 12

/-- The number of dogs walked on Thursday -/
def thursday_dogs : Nat := 9

/-- The total earnings for the week -/
def total_earnings : Nat := 210

/-- The number of dogs walked on Monday, Wednesday, and Friday combined -/
def other_days_dogs : Nat := 21

theorem dogs_walked_other_days :
  (tuesday_dogs + thursday_dogs + other_days_dogs) * rate = total_earnings ∧
  other_days_dogs = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_walked_other_days_l1085_108510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_like_is_identity_l1085_108598

/-- A function f: ℕ → ℕ is identity-like if for every m, n ∈ ℕ and every prime p,
    f(m+n) is divisible by p if and only if f(m) + f(n) is divisible by p -/
def IdentityLike (f : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ) (p : ℕ), Nat.Prime p →
    (p ∣ f (m + n) ↔ p ∣ f m + f n)

theorem identity_like_is_identity (f : ℕ → ℕ) (hf : Function.Surjective f) (h : IdentityLike f) :
  ∀ n, f n = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_like_is_identity_l1085_108598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dali_prints_consecutive_probability_l1085_108519

/-- The number of art pieces --/
def total_pieces : ℕ := 12

/-- The number of Dali prints --/
def dali_prints : ℕ := 4

/-- The probability of Dali prints being consecutive --/
def probability : ℚ := 1 / 55

theorem dali_prints_consecutive_probability :
  (Nat.factorial (total_pieces - dali_prints + 1) * Nat.factorial dali_prints) / 
  Nat.factorial total_pieces = probability := by
  -- Convert natural numbers to rationals
  have h1 : (↑(Nat.factorial (total_pieces - dali_prints + 1)) * ↑(Nat.factorial dali_prints)) / 
            ↑(Nat.factorial total_pieces) = probability := by sorry
  -- Use the above hypothesis to prove the theorem
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dali_prints_consecutive_probability_l1085_108519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_trip_is_30_minutes_l1085_108537

/-- Represents the trip details of Jerry and Beth from Smallville to Crown City -/
structure TripDetails where
  jerry_speed : ℝ
  beth_extra_distance : ℝ
  beth_speed : ℝ
  beth_extra_time : ℝ

/-- Calculates Jerry's trip time in hours given the trip details -/
noncomputable def jerry_trip_time (t : TripDetails) : ℝ :=
  let jerry_distance := t.jerry_speed * (t.beth_extra_time / 60 + (t.beth_extra_distance + t.beth_speed * t.beth_extra_time / 60) / t.beth_speed)
  jerry_distance / t.jerry_speed

/-- Theorem stating that Jerry's trip time is 0.5 hours (30 minutes) -/
theorem jerry_trip_is_30_minutes (t : TripDetails) 
  (h1 : t.jerry_speed = 40)
  (h2 : t.beth_extra_distance = 5)
  (h3 : t.beth_speed = 30)
  (h4 : t.beth_extra_time = 20) :
  jerry_trip_time t = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_trip_is_30_minutes_l1085_108537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficient_l1085_108574

theorem parabola_coefficient (a b c : ℝ) : 
  (2 = a * 1^2 + b * 1 + c) →
  (6 = a * 2^2 + b * 2 + c) →
  (4 = c) →
  b = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_coefficient_l1085_108574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l1085_108588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 - a*x^2 + a
  else 2^((2-a)*x) + 1/2

def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_monotone_iff_a_in_range (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ 0 ≤ a ∧ a ≤ 3/2 := by
  sorry

#check f_monotone_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l1085_108588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coord_l1085_108524

/-- A parabola is defined by the equation y^2 = 12x. -/
def is_on_parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The focus of the parabola y^2 = 12x is at (3, 0). -/
def focus : ℝ × ℝ := (3, 0)

/-- The distance between two points in 2D space. -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem states that the x-coordinate of a point on the parabola
    that is at a distance of 7 from the focus is 4. -/
theorem parabola_point_x_coord :
  ∃ (y : ℝ), is_on_parabola 4 y ∧ distance (4, y) focus = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coord_l1085_108524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_october_rose_count_l1085_108521

/-- Represents the number of roses displayed in a given month -/
def RoseCount (n : ℕ) : ℕ := sorry

/-- The constant difference between consecutive months -/
def difference : ℕ := 12

theorem october_rose_count 
  (nov_count : RoseCount 11 = 120)
  (dec_count : RoseCount 12 = 132)
  (jan_count : RoseCount 1 = 144)
  (feb_count : RoseCount 2 = 156)
  (pattern : ∀ n : ℕ, RoseCount (n + 1) = RoseCount n + difference) :
  RoseCount 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_october_rose_count_l1085_108521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_zero_monotonically_increasing_iff_a_in_range_max_b_when_a_negative_one_b_zero_has_solution_l1085_108525

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 / 3 - x^2 - a*x + Real.log (a*x + 1)

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - a + a / (a*x + 1)

theorem extremum_point_implies_a_zero (a : ℝ) :
  (f' a 2 = 0) → a = 0 := by sorry

theorem monotonically_increasing_iff_a_in_range (a : ℝ) :
  (∀ x ≥ 3, f' a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ (3 + Real.sqrt 13) / 2) := by sorry

theorem max_b_when_a_negative_one (b : ℝ) :
  (∃ x, f (-1) x = x^3 / 3 + b / (1 - x)) → b ≤ 0 := by sorry

theorem b_zero_has_solution :
  ∃ x, f (-1) x = x^3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_zero_monotonically_increasing_iff_a_in_range_max_b_when_a_negative_one_b_zero_has_solution_l1085_108525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_five_halves_l1085_108507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x^2 - 3*x + 1
  else (1/2)^x + 1/2

-- State the theorem
theorem f_composition_equals_five_halves : f (f 2) = 5/2 := by
  -- Evaluate f(2)
  have h1 : f 2 = -1 := by
    -- Simplify f(2)
    simp [f]
    -- Calculate 2^2 - 3*2 + 1
    norm_num
  
  -- Evaluate f(-1)
  have h2 : f (-1) = 5/2 := by
    -- Simplify f(-1)
    simp [f]
    -- Calculate (1/2)^(-1) + 1/2
    norm_num
  
  -- Combine the results
  calc
    f (f 2) = f (-1) := by rw [h1]
    _       = 5/2    := by rw [h2]

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_five_halves_l1085_108507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_satisfaction_count_l1085_108522

theorem inequality_satisfaction_count : ∃ (S : Finset ℚ), 
  S = {-1, 0, 1, 1/2} ∧ (S.filter (λ x ↦ 2 * x - 1 < x)).card = 3 := by
  -- Define the set of numbers
  let S : Finset ℚ := {-1, 0, 1, 1/2}
  
  -- Define the predicate for the inequality
  let P : ℚ → Prop := λ x ↦ 2 * x - 1 < x
  
  -- Assert that exactly 3 elements in S satisfy P
  have h : (S.filter P).card = 3 := by
    -- This is where the actual proof would go
    sorry
  
  -- Provide the existence proof
  exact ⟨S, by simp [S], h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_satisfaction_count_l1085_108522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_fifteen_eighths_l1085_108571

/-- A right triangle with sides 3, 4, and 5 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 3
  side_b : b = 4
  side_c : c = 5

/-- The length of the crease when point A is folded onto point B -/
noncomputable def crease_length (t : RightTriangle) : ℝ := 15/8

/-- Theorem: The length of the crease in the folded right triangle is 15/8 inches -/
theorem crease_length_is_fifteen_eighths (t : RightTriangle) :
  crease_length t = 15/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_fifteen_eighths_l1085_108571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_linear_implies_m_value_l1085_108509

/-- G is defined as a function of y and m -/
noncomputable def G (y m : ℝ) : ℝ := (8 * y^2 + 20 * y + 5 * m) / 4

/-- G is the square of a linear expression in y -/
def is_square_of_linear (G : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ y, G y = (a * y + b)^2

/-- Theorem stating that if G is the square of a linear expression in y, then m = 2.5 -/
theorem square_of_linear_implies_m_value (m : ℝ) :
  is_square_of_linear (λ y => G y m) → m = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_linear_implies_m_value_l1085_108509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_equals_24_14_l1085_108597

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

/-- The given number to be rounded -/
def givenNumber : ℝ := 24.1397

/-- Theorem stating that rounding the given number to the nearest hundredth equals 24.14 -/
theorem round_to_hundredth_equals_24_14 : 
  roundToHundredth givenNumber = 24.14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_equals_24_14_l1085_108597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_problem_l1085_108590

theorem group_size_problem (total_collection : ℚ) (h1 : total_collection = 20.25) :
  ∃ n : ℕ, n * n = (total_collection * 100).floor ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_problem_l1085_108590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fon_last_remaining_l1085_108542

/-- Represents a student in the circle -/
inductive Student : Type
| Arn | Bob | Cyd | Dan | Eve | Fon | Gun | Hal

/-- Determines if a number contains 5 as a digit or is a multiple of 5 -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 5 = 0 || n.repr.contains '5'

/-- Simulates the elimination process and returns the last remaining student -/
def lastRemainingStudent (students : List Student) : Student :=
  sorry

/-- Theorem stating that Fon is the last remaining student -/
theorem fon_last_remaining :
  lastRemainingStudent [Student.Arn, Student.Bob, Student.Cyd, Student.Dan,
                        Student.Eve, Student.Fon, Student.Gun, Student.Hal] = Student.Fon :=
by sorry

#eval isEliminationNumber 15  -- Should return true
#eval isEliminationNumber 25  -- Should return true
#eval isEliminationNumber 7   -- Should return false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fon_last_remaining_l1085_108542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integer_solutions_implies_a_is_one_l1085_108557

-- Define the equation
def f (x a : ℤ) : Prop := abs (abs (x - 3) - 1) = a

-- Theorem statement
theorem three_integer_solutions_implies_a_is_one :
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, f x a) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integer_solutions_implies_a_is_one_l1085_108557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_translation_and_symmetry_l1085_108589

/-- The original curve C -/
noncomputable def C (x y : ℝ) : Prop := y = x^3 - x

/-- The translated curve C1 -/
noncomputable def C1 (x y t s : ℝ) : Prop := y = (x - t)^3 - (x - t) + s

/-- Point of symmetry -/
noncomputable def A (t s : ℝ) : ℝ × ℝ := (t/2, s/2)

/-- Symmetry relation between two points with respect to A -/
def symmetric_wrt_A (x1 y1 x2 y2 t s : ℝ) : Prop :=
  x2 = t - x1 ∧ y2 = s - y1

theorem curve_translation_and_symmetry (t s : ℝ) :
  (∀ x y, C x y ↔ C1 x y t s) ∧
  (∀ x1 y1 x2 y2, C x1 y1 ∧ C1 x2 y2 t s →
    symmetric_wrt_A x1 y1 x2 y2 t s) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_translation_and_symmetry_l1085_108589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_eq_one_max_sin_B_plus_sin_C_l1085_108558

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (Real.pi/4 + x) * Real.sin (Real.pi/4 - x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_pi_sixth_eq_one : f (Real.pi/6) = 1 := by sorry

theorem max_sin_B_plus_sin_C (A B C : ℝ) 
  (triangle_angles : A + B + C = Real.pi) 
  (f_A_half_eq_one : f (A/2) = 1) : 
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ 
    ∀ (B' C' : ℝ), B' + C' = Real.pi - A → Real.sin B' + Real.sin C' ≤ max := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_eq_one_max_sin_B_plus_sin_C_l1085_108558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1085_108585

noncomputable def f (x : ℝ) := Real.sqrt (x + 3) / Real.sqrt (7 - x)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 ≤ x ∧ x < 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1085_108585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_ten_terms_l1085_108502

/-- Euler's totient function -/
noncomputable def φ : ℕ → ℕ := sorry

/-- Sequence aₙ defined as φ(2ⁿ) -/
noncomputable def a : ℕ → ℕ := λ n => φ (2^n)

/-- Sum of the first n terms of sequence a -/
noncomputable def S : ℕ → ℕ := λ n => (List.range n).map (fun i => a (i+1)) |>.sum

theorem sum_first_ten_terms : S 10 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_ten_terms_l1085_108502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quincy_harold_difference_l1085_108516

/-- The number of telephone poles -/
def num_poles : ℕ := 51

/-- The number of Harold's strides between consecutive poles -/
def harold_strides : ℕ := 50

/-- The number of Quincy's jumps between consecutive poles -/
def quincy_jumps : ℕ := 15

/-- The total distance between the first and last pole in feet -/
def total_distance : ℝ := 6000

/-- Harold's stride length in feet -/
noncomputable def harold_stride_length : ℝ := total_distance / (harold_strides * (num_poles - 1))

/-- Quincy's jump length in feet -/
noncomputable def quincy_jump_length : ℝ := total_distance / (quincy_jumps * (num_poles - 1))

/-- The theorem stating the difference between Quincy's jump and Harold's stride -/
theorem quincy_harold_difference : quincy_jump_length - harold_stride_length = 5.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quincy_harold_difference_l1085_108516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_arithmetic_implies_double_angle_sine_arithmetic_l1085_108538

theorem tangent_arithmetic_implies_double_angle_sine_arithmetic 
  (α β γ : Real) (h_triangle : α + β + γ = Real.pi) 
  (h_arithmetic : Real.tan α + Real.tan γ = 2 * Real.tan β) : 
  Real.sin (2 * α) + Real.sin (2 * γ) = 2 * Real.sin (2 * β) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_arithmetic_implies_double_angle_sine_arithmetic_l1085_108538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_sum_divisible_by_24_l1085_108540

theorem divisors_sum_divisible_by_24 (n : ℕ) (h : 24 ∣ (n + 1)) :
  24 ∣ (Finset.sum (Nat.divisors n) id) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_sum_divisible_by_24_l1085_108540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hilary_tip_percentage_l1085_108534

/-- Represents the cost structure and tip calculation for a meal at Delicious Delhi restaurant -/
structure MealCost where
  samosa_price : ℚ
  samosa_quantity : ℕ
  pakora_price : ℚ
  pakora_quantity : ℕ
  lassi_price : ℚ
  total_with_tax : ℚ

/-- Calculates the tip percentage given the meal cost structure -/
noncomputable def tip_percentage (meal : MealCost) : ℚ :=
  let food_cost := meal.samosa_price * meal.samosa_quantity + 
                   meal.pakora_price * meal.pakora_quantity + 
                   meal.lassi_price
  let tip := meal.total_with_tax - food_cost
  (tip / food_cost) * 100

/-- Theorem stating that the tip percentage for Hilary's meal was 25% -/
theorem hilary_tip_percentage :
  let meal : MealCost := {
    samosa_price := 2,
    samosa_quantity := 3,
    pakora_price := 3,
    pakora_quantity := 4,
    lassi_price := 2,
    total_with_tax := 25
  }
  tip_percentage meal = 25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hilary_tip_percentage_l1085_108534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l1085_108555

open Real

noncomputable def f (x φ : ℝ) : ℝ := sin (2 * x + φ)

theorem increasing_interval_of_f 
  (φ : ℝ) 
  (h_φ : abs φ < π) 
  (h_bound : ∀ x : ℝ, f x φ ≤ abs (f (π/6) φ))
  (h_compare : f (π/2) φ > f π φ) :
  ∀ k : ℤ, StrictMonoOn (f · φ) (Set.Icc (k * π + π/6) (k * π + 2*π/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l1085_108555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_tuesday_l1085_108544

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the previous day of the week -/
def previousDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Monday => .Sunday
  | .Tuesday => .Monday
  | .Wednesday => .Tuesday
  | .Thursday => .Wednesday
  | .Friday => .Thursday
  | .Saturday => .Friday
  | .Sunday => .Saturday

/-- Returns the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) : Nat → DayOfWeek
  | 0 => d
  | n + 1 => daysBefore (previousDay d) n

theorem february_first_is_tuesday (h : DayOfWeek.Wednesday = daysBefore DayOfWeek.Wednesday 22) :
  daysBefore DayOfWeek.Wednesday 22 = DayOfWeek.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_tuesday_l1085_108544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_tangent_l1085_108595

/-- Definition of Circle 1 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

/-- Definition of Circle 2 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating that the two circles are tangent -/
theorem circles_are_tangent :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧
  distance 0 0 (-1) 0 = 1 ∧
  distance 0 0 x y = 2 ∧
  distance (-1) 0 x y = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_tangent_l1085_108595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l1085_108599

/-- The radius of a circle tangent to all semicircles in a square configuration -/
theorem tangent_circle_radius (square_side : ℝ) (num_semicircles : ℕ) : 
  square_side = 4 → num_semicircles = 10 → 
  let semicircle_radius := square_side / (2 * (num_semicircles / 4 : ℝ))
  let hypotenuse := Real.sqrt ((square_side / 2) ^ 2 + semicircle_radius ^ 2)
  (hypotenuse - semicircle_radius) / 2 = Real.sqrt 116 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l1085_108599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_of_regular_is_regular_l1085_108553

-- Define a regular polyhedron
structure RegularPolyhedron where
  vertices : Set Point
  edges : Set (Point × Point)
  faces : Set (Set Point)
  is_regular : Bool

-- Define a dual polyhedron
noncomputable def dual_polyhedron (T : RegularPolyhedron) : RegularPolyhedron :=
  { vertices := sorry,
    edges := sorry,
    faces := sorry,
    is_regular := sorry }

-- Theorem statement
theorem dual_of_regular_is_regular (T : RegularPolyhedron) :
  T.is_regular → (dual_polyhedron T).is_regular :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_of_regular_is_regular_l1085_108553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_imply_k_range_l1085_108545

/-- A circle in the xy-plane with center (-k, -1) and radius 5 -/
def Circle (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + k)^2 + (p.2 + 1)^2 = 25}

/-- A point is outside a circle if its distance from the center is greater than the radius -/
def IsOutside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, 
    c = {q : ℝ × ℝ | (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2} ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 > radius^2

/-- Definition of a tangent line to a circle -/
def IsTangentLine (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∧ p ∈ c ∧ ∀ q : ℝ × ℝ, q ∈ l ∧ q ∈ c → q = p

/-- Two tangent lines can be drawn from a point to a circle if and only if the point is outside the circle -/
axiom two_tangents_iff_outside {p : ℝ × ℝ} {c : Set (ℝ × ℝ)} :
  (∃ l1 l2 : Set (ℝ × ℝ), l1 ≠ l2 ∧ IsTangentLine l1 c ∧ IsTangentLine l2 c ∧ p ∈ l1 ∧ p ∈ l2) ↔
  IsOutside p c

/-- The main theorem: If two tangent lines can be drawn from (1, 3) to the circle,
    then k is either greater than 2 or less than -4 -/
theorem tangent_lines_imply_k_range (k : ℝ) :
  (∃ l1 l2 : Set (ℝ × ℝ), l1 ≠ l2 ∧ IsTangentLine l1 (Circle k) ∧ IsTangentLine l2 (Circle k) ∧ (1, 3) ∈ l1 ∧ (1, 3) ∈ l2) →
  k > 2 ∨ k < -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_imply_k_range_l1085_108545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_probability_l1085_108579

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  is_equilateral : (dist A B = dist B C) ∧ (dist B C = dist C A)

-- Define a random point inside the triangle
noncomputable def random_point (t : EquilateralTriangle) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the area of a triangle
noncomputable def triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the probability function
noncomputable def probability (t : EquilateralTriangle) : ℝ := sorry

-- The main theorem
theorem area_probability (t : EquilateralTriangle) :
  probability t = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_probability_l1085_108579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_spacing_l1085_108559

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 660)
  (h2 : num_trees = 42)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 660 / 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_spacing_l1085_108559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_approx_l1085_108567

/-- Represents a segment of a trip with distance and speed -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the average speed of a trip given its segments -/
noncomputable def averageSpeed (segments : List Segment) : ℝ :=
  let totalDistance := segments.map (·.distance) |>.sum
  let totalTime := segments.map (fun s => s.distance / s.speed) |>.sum
  totalDistance / totalTime

/-- The trip described in the problem -/
def tripSegments : List Segment := [
  { distance := 40, speed := 50 },
  { distance := 35, speed := 30 },
  { distance := 25, speed := 60 }
]

theorem trip_average_speed_approx :
  ∃ ε > 0, |averageSpeed tripSegments - 41.96| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_approx_l1085_108567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_l1085_108562

/-- Area function for geometric shapes -/
noncomputable def area (shape : Set ℝ × Set ℝ) : ℝ := sorry

/-- Rectangle ADEF -/
def ADEF : Set ℝ × Set ℝ := sorry

/-- Triangle ADB -/
def ADB : Set ℝ × Set ℝ := sorry

/-- Triangle ACF -/
def ACF : Set ℝ × Set ℝ := sorry

/-- Triangle ABC -/
def ABC : Set ℝ × Set ℝ := sorry

/-- Given a rectangle ADEF and triangles ADB and ACF within it, 
    we prove that the area of triangle ABC is 6.5 -/
theorem area_triangle_ABC 
  (h1 : area ADEF = 16)
  (h2 : area ADB = 3)
  (h3 : area ACF = 4) :
  area ABC = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_l1085_108562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_range_l1085_108533

/-- Circle with center (3,4) and radius 1 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

/-- Point A on x-axis -/
def A (m : ℝ) : ℝ × ℝ := (-m, 0)

/-- Point B on x-axis -/
def B (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Condition for right angle APB -/
def isRightAngle (P : ℝ × ℝ) (m : ℝ) : Prop :=
  (P.1 + m) * (P.1 - m) + P.2^2 = 0

theorem circle_points_range (m : ℝ) (h : m > 0) :
  (∀ P ∈ C, ¬isRightAngle P m) → m ∈ Set.Ioo 0 4 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_range_l1085_108533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_system_solution_l1085_108596

theorem equation_system_solution :
  ∃! (x y z : ℕ),
    (z : ℝ)^(x : ℝ) = (y : ℝ)^(2*(x : ℝ)) ∧
    (2 : ℝ)^(z : ℝ) = 4 * (8 : ℝ)^(x : ℝ) ∧
    x + y + z = 18 ∧
    x = 5 ∧ y = 4 ∧ z = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_system_solution_l1085_108596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_catches_mouse_l1085_108551

/-- Represents the path lengths and speeds in the cat-mouse chase problem -/
structure CatMouseProblem where
  ab : ℝ  -- Length of AB in meters
  bc : ℝ  -- Length of BC in meters
  cd : ℝ  -- Length of CD in meters
  mouse_speed : ℝ  -- Mouse speed in meters per minute
  cat_speed : ℝ  -- Cat speed in meters per minute

/-- Calculates the time taken by the mouse to reach point D -/
noncomputable def mouse_time (p : CatMouseProblem) : ℝ :=
  (p.ab + p.bc) / p.mouse_speed

/-- Calculates the time taken by the cat to reach point D -/
noncomputable def cat_time (p : CatMouseProblem) : ℝ :=
  (p.ab + p.bc + p.cd) / p.cat_speed

/-- Theorem stating that the cat catches the mouse -/
theorem cat_catches_mouse (p : CatMouseProblem) 
  (hab : p.ab = 200) 
  (hbc : p.bc = 140) 
  (hcd : p.cd = 20) 
  (hms : p.mouse_speed = 60) 
  (hcs : p.cat_speed = 80) : 
  cat_time p < mouse_time p := by
  sorry

#eval "Cat catches mouse theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_catches_mouse_l1085_108551
