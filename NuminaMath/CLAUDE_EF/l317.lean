import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_zero_l317_31736

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > -3 then x^2 - 9 else 2*x + 6

-- Theorem statement
theorem unique_double_zero :
  ∃! x : ℝ, f (f x) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_zero_l317_31736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l317_31718

theorem power_equality (a b : ℝ) (h1 : (30 : ℝ)^a = 2) (h2 : (30 : ℝ)^b = 3) :
  (6 : ℝ)^((1 - a - b) / (2 * (1 - b))) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l317_31718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_stage_discount_l317_31719

/-- Proves the actual discount and difference from claimed discount for a two-stage discount process -/
theorem two_stage_discount (initial_discount additional_discount claimed_discount : ℝ) 
  (h1 : initial_discount = 0.25)
  (h2 : additional_discount = 0.15)
  (h3 : claimed_discount = 0.40) : 
  (1 - (1 - initial_discount) * (1 - additional_discount) = 0.3625) ∧ 
  (|claimed_discount - (1 - (1 - initial_discount) * (1 - additional_discount))| = 0.0375) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_stage_discount_l317_31719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_not_sum_l317_31703

theorem power_difference_not_sum (k : ℕ) (h1 : k > 1) (h2 : k ≠ 3) :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
    ∀ n : ℕ, ∃ a b : ℕ, f n = a^k - b^k ∧
      ∀ u v : ℕ, f n ≠ u^k + v^k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_not_sum_l317_31703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l317_31778

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 3) / Real.log 2

theorem f_monotone_increasing : 
  MonotoneOn f (Set.Ioi 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l317_31778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l317_31737

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  let ρ : Real := 5
  let θ : Real := 7 * Real.pi / 4
  let φ : Real := Real.pi / 3
  spherical_to_rectangular ρ θ φ = (-5 * Real.sqrt 6 / 4, -5 * Real.sqrt 6 / 4, 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l317_31737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_conditions_l317_31723

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem count_integers_satisfying_conditions : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 4 ∣ n ∧ Nat.lcm (factorial 4) n = 4 * Nat.gcd (factorial 8) n) ∧ 
    Finset.card S = 72 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_conditions_l317_31723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_age_theorem_l317_31765

theorem baby_age_theorem (initial_members : Nat) (years_passed : Nat) (initial_average_age : Nat) (current_average_age : Nat) (baby_age : Nat) :
  initial_members = 5 →
  years_passed = 3 →
  initial_average_age = 17 →
  current_average_age = 17 →
  (initial_members * initial_average_age + initial_members * years_passed + baby_age) / (initial_members + 1) = current_average_age →
  baby_age = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baby_age_theorem_l317_31765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_egg_production_l317_31749

theorem chicken_egg_production 
  (x y z w v : ℚ) 
  (hx : x > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0) :
  let rate := y / (x * z)
  (rate * w * v) = (w * y * v) / (x * z) :=
by
  -- Define the rate of egg production per chicken per day
  let rate := y / (x * z)
  
  -- Show that (rate * w * v) = (w * y * v) / (x * z)
  calc
    rate * w * v = (y / (x * z)) * w * v := rfl
    _ = (y * w * v) / (x * z) := by ring
    _ = (w * y * v) / (x * z) := by ring

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_egg_production_l317_31749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_theorem_l317_31774

/-- The area of a parallelogram with sides a and b (a > b) and acute angle α between its diagonals --/
noncomputable def parallelogram_area (a b α : ℝ) : ℝ :=
  1/2 * (a^2 - b^2) * Real.tan α

/-- Theorem: Area of a parallelogram with given conditions --/
theorem parallelogram_area_theorem (a b α : ℝ) 
  (h1 : a > b) 
  (h2 : 0 < α ∧ α < π/2) : 
  parallelogram_area a b α = 1/2 * (a^2 - b^2) * Real.tan α :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_theorem_l317_31774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_roller_diameter_l317_31794

/-- Represents a garden roller as a cylinder -/
structure GardenRoller where
  length : ℝ
  diameter : ℝ

/-- Calculates the area covered by a garden roller in one revolution -/
noncomputable def areaCoveredPerRevolution (roller : GardenRoller) : ℝ :=
  roller.length * (Real.pi * roller.diameter)

/-- The theorem stating the diameter of the garden roller -/
theorem garden_roller_diameter :
  ∃ (roller : GardenRoller),
    roller.length = 2 ∧
    4 * areaCoveredPerRevolution roller = 35.2 ∧
    (Real.pi = 22 / 7) →
    roller.diameter = 1.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_roller_diameter_l317_31794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l317_31726

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → (2 : ℝ)^a > (2 : ℝ)^b) ↔ ((2 : ℝ)^a ≤ (2 : ℝ)^b → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l317_31726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_effective_speed_l317_31775

/-- Represents the speed and stopping time of a bus -/
structure Bus where
  speed_without_stops : ℝ  -- Speed in km/h without stops
  stop_time : ℝ           -- Stop time in minutes per hour

/-- Calculates the effective speed of a bus including stops -/
noncomputable def effective_speed (b : Bus) : ℝ :=
  let moving_time := 60 - b.stop_time
  let distance := (b.speed_without_stops * moving_time) / 60
  distance

theorem bus_effective_speed (b : Bus) 
  (h1 : b.speed_without_stops = 80)
  (h2 : b.stop_time = 15) :
  effective_speed b = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_effective_speed_l317_31775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_10_pow_100_minus_94_l317_31788

/-- The sum of the digits of 10^100 - 94 is 888 -/
theorem sum_of_digits_10_pow_100_minus_94 : 
  (Nat.digits 10 (10^100 - 94)).sum = 888 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_10_pow_100_minus_94_l317_31788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_theorem_l317_31722

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length a -/
structure Cube (a : ℝ) where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- The area of the section formed by the plane passing through K, L, and N in the cube -/
noncomputable def sectionArea (a : ℝ) (cube : Cube a) : ℝ :=
  (11 * a^2 * Real.sqrt 77) / 96

/-- Theorem statement for the section area in the cube -/
theorem section_area_theorem (a : ℝ) (cube : Cube a) 
  (hK : Point3D)  -- K is the midpoint of B₁C₁
  (hL : Point3D)  -- L is on C₁D₁ with D₁L = 2C₁L
  (hN : Point3D)  -- N is the midpoint of AA₁
  (hKmid : hK.x = a ∧ hK.y = a ∧ hK.z = 3*a/2)
  (hLpos : hL.x = 0 ∧ hL.y = a ∧ hL.z = 2*a/3)
  (hNmid : hN.x = 0 ∧ hN.y = 0 ∧ hN.z = a/2) :
  sectionArea a cube = (11 * a^2 * Real.sqrt 77) / 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_theorem_l317_31722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OPQ_l317_31769

-- Define the circle M
noncomputable def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l₁
noncomputable def line_l1 (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 4 = 0

-- Define the curve C (trajectory of point N)
noncomputable def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l (perpendicular to l₁)
noncomputable def line_l (x y m : ℝ) : Prop := Real.sqrt 3 * x + y + m = 0

-- Define the area of △OPQ
noncomputable def area_OPQ (m : ℝ) : ℝ := 2 * Real.sqrt (m^2 * (13 - m^2)) / 13

-- State the theorem
theorem max_area_OPQ :
  ∃ (m : ℝ), ∀ (m' : ℝ), area_OPQ m' ≤ area_OPQ m ∧ area_OPQ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OPQ_l317_31769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_travel_time_l317_31798

/-- Represents the speed of a boat in kilometers per hour -/
noncomputable def BoatSpeed : ℝ := 12

/-- Represents the distance traveled in kilometers -/
noncomputable def Distance : ℝ := 54

/-- Represents the time taken to travel upstream in hours -/
noncomputable def UpstreamTime : ℝ := 9

/-- Calculates the time taken to travel downstream in hours -/
noncomputable def DownstreamTime : ℝ :=
  let currentSpeed := BoatSpeed - (Distance / UpstreamTime)
  Distance / (BoatSpeed + currentSpeed)

theorem downstream_travel_time :
  DownstreamTime = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_travel_time_l317_31798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gamma_value_l317_31795

/-- Given unit vectors α and β in a plane with an angle of 60° between them,
    and (2α - γ) · (β - γ) = 0, the maximum value of |γ| is (√7 + √3) / 2. -/
theorem max_gamma_value (α β γ : Fin 2 → ℝ) :
  ‖α‖ = 1 →
  ‖β‖ = 1 →
  α • β = 1 / 2 →
  (2 • α - γ) • (β - γ) = 0 →
  ‖γ‖ ≤ (Real.sqrt 7 + Real.sqrt 3) / 2 ∧
  ∃ γ₀ : Fin 2 → ℝ, (2 • α - γ₀) • (β - γ₀) = 0 ∧ ‖γ₀‖ = (Real.sqrt 7 + Real.sqrt 3) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gamma_value_l317_31795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_equals_2_l317_31762

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / sequenceA n

theorem sequence_2023_equals_2 : sequenceA 2022 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_equals_2_l317_31762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_peak_line_l317_31767

/-- Represents the trajectory of a projectile --/
structure Trajectory where
  v : ℝ  -- Initial velocity
  g : ℝ  -- Acceleration due to gravity

/-- The x-coordinate of the projectile at time t --/
noncomputable def x_coord (traj : Trajectory) (t : ℝ) : ℝ :=
  traj.v * t * (Real.sqrt 2 / 2)

/-- The y-coordinate of the projectile at time t --/
noncomputable def y_coord (traj : Trajectory) (t : ℝ) : ℝ :=
  traj.v * t * (Real.sqrt 2 / 2) - (1/2) * traj.g * t^2

/-- The time at which the projectile reaches its highest point --/
noncomputable def peak_time (traj : Trajectory) : ℝ :=
  traj.v * (Real.sqrt 2 / 2) / traj.g

/-- The x-coordinate of the highest point of the trajectory --/
noncomputable def peak_x (traj : Trajectory) : ℝ :=
  x_coord traj (peak_time traj)

/-- The y-coordinate of the highest point of the trajectory --/
noncomputable def peak_y (traj : Trajectory) : ℝ :=
  y_coord traj (peak_time traj)

/-- Theorem stating that the highest points of all trajectories form a straight line y = x/2 --/
theorem trajectory_peak_line (traj : Trajectory) : peak_y traj = (1/2) * peak_x traj := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_peak_line_l317_31767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_vector_difference_l317_31753

open Real InnerProductSpace

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 4 and the projection of b on a equal to -2,
    prove that the minimum value of |a - 3b| is 10. -/
theorem min_value_of_vector_difference (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 4) (h2 : inner a b / ‖a‖ = -2) : 
  ∀ x : V, ‖a - 3 • b‖ ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_vector_difference_l317_31753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_not_filler_is_75_percent_l317_31761

/-- Given a burger with a total weight and filler weight, calculates the percentage that is not filler -/
noncomputable def burger_not_filler_percentage (total_weight filler_weight : ℝ) : ℝ :=
  (total_weight - filler_weight) / total_weight * 100

/-- Theorem: The percentage of a burger that is not filler is 75%, given specific weights -/
theorem burger_not_filler_is_75_percent :
  burger_not_filler_percentage 180 45 = 75 := by
  -- Unfold the definition of burger_not_filler_percentage
  unfold burger_not_filler_percentage
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_burger_not_filler_is_75_percent_l317_31761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exceeds_twenty_l317_31760

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => sequence_a n + 1 / sequence_a n

theorem sequence_exceeds_twenty (n : ℕ) : sequence_a n > 20 ↔ n > 191 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exceeds_twenty_l317_31760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_rhombus_revolution_proof_l317_31748

/-- The volume of a solid of revolution formed by rotating a rhombus -/
noncomputable def volume_rhombus_revolution (a : ℝ) (α : ℝ) : ℝ :=
  2 * Real.pi * a^3 * Real.sin α * Real.sin (α/2)

/-- Theorem stating the volume of the solid of revolution formed by a rhombus -/
theorem volume_rhombus_revolution_proof (a : ℝ) (α : ℝ) 
  (h1 : a > 0) 
  (h2 : 0 < α ∧ α < Real.pi/2) :
  volume_rhombus_revolution a α = 2 * Real.pi * a^3 * Real.sin α * Real.sin (α/2) := by
  -- Unfold the definition of volume_rhombus_revolution
  unfold volume_rhombus_revolution
  -- The definition and the right-hand side are identical, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_rhombus_revolution_proof_l317_31748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l317_31738

noncomputable section

/-- The area of the region inside a rectangle but outside three quarter circles --/
def area_outside_circles (ef fg r₁ r₂ r₃ : ℝ) : ℝ :=
  ef * fg - (Real.pi / 4) * (r₁^2 + r₂^2 + r₃^2)

/-- Theorem stating that the area of the region inside the rectangle EFGH but outside 
    the quarter circles centered at E, F, and G is approximately 8.8 --/
theorem area_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |area_outside_circles 4 6 2 3 2.5 - 8.8| < ε :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l317_31738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l317_31708

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  hyp_eq : c^2 = a^2 + b^2

/-- The distance from the focus to the asymptote of a hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ :=
  (h.b * h.c) / Real.sqrt (h.b^2 + h.a^2)

theorem hyperbola_focus_asymptote_distance (h : Hyperbola) 
  (dist_eq : focus_to_asymptote_distance h = (Real.sqrt 3 / 2) * h.c) :
  h.b / h.c = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l317_31708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_negative_terms_characterization_l317_31777

/-- An arithmetic sequence -/
def arithmetic_sequence (a d : ℝ) : ℕ → ℝ := λ n ↦ a + d * n

/-- Predicate for a sequence containing negative terms -/
def has_negative_terms (s : ℕ → ℝ) : Prop :=
  ∃ n, s n < 0

/-- Predicate for a sequence having only finitely many negative terms -/
def finitely_many_negative_terms (s : ℕ → ℝ) : Prop :=
  ∃ N, ∀ n ≥ N, s n ≥ 0

theorem arithmetic_sequence_negative_terms_characterization (a d : ℝ) :
  (has_negative_terms (arithmetic_sequence a d) ∧
   finitely_many_negative_terms (arithmetic_sequence a d)) ↔
  (a < 0 ∧ d > 0) := by
  sorry

#check arithmetic_sequence_negative_terms_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_negative_terms_characterization_l317_31777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_time_is_16_seconds_l317_31731

/-- Calculates the time for two trains to clear each other after meeting. -/
noncomputable def trainClearanceTime (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let totalLength := length1 + length2
  let speed1InMPS := speed1 * 1000 / 3600
  let speed2InMPS := speed2 * 1000 / 3600
  let relativeSpeed := speed1InMPS + speed2InMPS
  totalLength / relativeSpeed

/-- Theorem stating that the time for two trains to clear each other is approximately 16 seconds. -/
theorem train_clearance_time_is_16_seconds :
  ∃ ε > 0, |trainClearanceTime 100 220 42 30 - 16| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearance_time_is_16_seconds_l317_31731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_geometric_progression_l317_31782

/-- The sequence from which we want to extract a geometric progression -/
def originalSequence (n : ℕ) : ℚ := 1 / n

/-- A geometric progression extracted from the sequence -/
def geometricProgression (n : ℕ) : ℚ := 1 / (2^(n-1))

/-- Theorem stating that a geometric progression can be extracted from the sequence -/
theorem exists_geometric_progression (k : ℕ) (h : k > 2) :
  ∃ (f : ℕ → ℕ), (∀ i < k, originalSequence (f i) = geometricProgression i) ∧
                 (∀ i < k - 1, geometricProgression (i + 1) = (geometricProgression i) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_geometric_progression_l317_31782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_lines_through_point_l317_31758

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A straight line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The set of all lines passing through a given point -/
def linesThrough (p : Point) : Set Line :=
  {l : Line | p.y = l.slope * p.x + l.intercept}

/-- Theorem: There are infinitely many lines passing through any given point -/
theorem infinitely_many_lines_through_point (p : Point) :
  Set.Infinite (linesThrough p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_lines_through_point_l317_31758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_properties_M_remainder_l317_31742

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i < digits.length → j < digits.length → i ≠ j → digits[i]! ≠ digits[j]!

noncomputable def M : ℕ := sorry

theorem M_properties :
  is_multiple_of_9 M ∧
  has_unique_digits M ∧
  ∀ n : ℕ, is_multiple_of_9 n → has_unique_digits n → n ≤ M :=
by sorry

theorem M_remainder :
  M % 1000 = 963 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_properties_M_remainder_l317_31742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l317_31717

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => 2 * sequence_a (n + 1) + 1

theorem a_4_equals_15 : sequence_a 4 = 15 := by
  -- Proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l317_31717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_f_range_endpoints_l317_31747

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x + Real.pi/6) + 1

-- Statement 1: The smallest positive period of f is π
theorem smallest_positive_period (x : ℝ) : f (x + Real.pi) = f x := by sorry

-- Statement 2: The range of f when x ∈ [-7π/12, 0] is [-2, 1]
theorem range_of_f : ∀ x, x ∈ Set.Icc (-7*Real.pi/12) 0 → -2 ≤ f x ∧ f x ≤ 1 := by sorry

-- Additional theorem to show that the range includes the endpoints
theorem range_endpoints :
  ∃ x₁ x₂, x₁ ∈ Set.Icc (-7*Real.pi/12) 0 ∧ x₂ ∈ Set.Icc (-7*Real.pi/12) 0 ∧ f x₁ = -2 ∧ f x₂ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_f_range_endpoints_l317_31747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l317_31716

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - 2 * Real.pi / 3)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l317_31716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_tangent_line_through_origin_l317_31768

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_at_e (x : ℝ) :
  ∃ (m b : ℝ), m * x + b = f (Real.exp 1) + (deriv f) (Real.exp 1) * (x - Real.exp 1) :=
by sorry

theorem tangent_line_through_origin (x : ℝ) :
  ∃ (x₀ : ℝ), f x₀ + (deriv f) x₀ * (x - x₀) = Real.exp 1 * x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_tangent_line_through_origin_l317_31768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_correct_l317_31704

/-- The area of a square inscribed in the ellipse x²/5 + y²/10 = 1, 
    with its sides parallel to the coordinate axes -/
noncomputable def inscribed_square_area : ℝ := 40/3

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2/5 + y^2/10 = 1

/-- A point is on the square if its coordinates are equal in absolute value
    and satisfy the ellipse equation -/
def on_inscribed_square (x y : ℝ) : Prop :=
  abs x = abs y ∧ ellipse_equation x y

theorem inscribed_square_area_is_correct :
  ∃ (x y : ℝ), on_inscribed_square x y ∧
  (4 * x^2 = inscribed_square_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_correct_l317_31704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_problem_l317_31764

noncomputable section

/-- The area between two concentric circles -/
def area_between_circles (R r : ℝ) : ℝ := Real.pi * (R^2 - r^2)

/-- The radius of the inner circle given the radius of the outer circle and a tangent chord length -/
def inner_radius (R chord_length : ℝ) : ℝ := 
  Real.sqrt (R^2 - (chord_length/2)^2)

theorem area_between_circles_problem (R chord_length : ℝ) 
  (h1 : R = 13)
  (h2 : chord_length = 24) :
  area_between_circles R (inner_radius R chord_length) = 144 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_problem_l317_31764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_of_progression_l317_31725

noncomputable def arithmetic_progression (a b : ℝ) (n : ℕ) : ℝ := a + (n - 1) * b

noncomputable def sum_arithmetic_progression (a b : ℝ) (n : ℕ) : ℝ := 
  (n / 2 : ℝ) * (2 * a + (n - 1) * b)

theorem fifteenth_term_of_progression (a b : ℝ) :
  sum_arithmetic_progression a b 10 = 60 →
  sum_arithmetic_progression a b 20 = 320 →
  arithmetic_progression a b 15 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_of_progression_l317_31725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_prime_in_repeat_67_list_l317_31787

def repeat_67 (n : ℕ) : ℕ := 
  if n = 0 then 0 else (67 : ℕ) + 100 * (repeat_67 (n - 1))

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem only_one_prime_in_repeat_67_list : 
  (∃! k : ℕ, k ∈ (List.range 10).map repeat_67 ∧ is_prime k) ∧
  is_prime 67 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_prime_in_repeat_67_list_l317_31787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_explicit_formula_l317_31754

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | (n + 1) => (n : ℚ) / (n + 1 : ℚ) * sequence_a n

theorem sequence_a_explicit_formula :
  ∀ n : ℕ, n ≥ 1 → sequence_a n = 3 / n := by
  intro n hn
  sorry  -- Skip the proof for now

#eval sequence_a 5  -- Add this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_explicit_formula_l317_31754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l317_31759

/-- Given a point P and a midpoint M of line segment PQ, 
    if vector PQ is collinear with vector a = (λ, 1), then λ = -2/3 -/
theorem collinear_vectors (P M Q : ℝ × ℝ) (l : ℝ) :
  P = (-1, 2) →
  M = (1, -1) →
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  ∃ (k : ℝ), k ≠ 0 ∧ (Q.1 - P.1, Q.2 - P.2) = (k * l, k * 1) →
  l = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l317_31759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_triples_l317_31746

theorem divisibility_triples :
  ∀ x y z : ℕ+,
    (x ∣ y + 1) ∧ (y ∣ z + 1) ∧ (z ∣ x + 1) →
    ((x, y, z) ∈ ({(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1),
                   (1, 3, 2), (3, 2, 1), (2, 1, 3),
                   (3, 5, 4), (5, 4, 3), (4, 3, 5)} : Set (ℕ+ × ℕ+ × ℕ+))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_triples_l317_31746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_satisfies_conditions_l317_31714

-- Define the polar coordinate system
noncomputable def PolarCoord := ℝ × ℝ

-- Define the intersection point
noncomputable def intersection_point : PolarCoord := (2, 3 * Real.pi / 4)

-- Define the conditions
def curve1 (p : PolarCoord) : Prop := p.1 = 2
def curve2 (p : PolarCoord) : Prop := Real.cos p.2 + Real.sin p.2 = 0
def angle_range (p : PolarCoord) : Prop := 0 ≤ p.2 ∧ p.2 ≤ Real.pi

-- Theorem statement
theorem intersection_point_satisfies_conditions :
  curve1 intersection_point ∧
  curve2 intersection_point ∧
  angle_range intersection_point := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_satisfies_conditions_l317_31714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l317_31701

theorem triangle_shape (A B C : ℝ) (hsin : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  Real.cos C < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l317_31701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_at_80km_l317_31739

/-- The distance at which B catches up with A -/
noncomputable def catchup_distance (speed_a speed_b : ℝ) (delay : ℝ) : ℝ :=
  (speed_a * delay * speed_b) / (speed_b - speed_a)

/-- Theorem stating that B catches up with A at 80 km -/
theorem catchup_at_80km (speed_a speed_b delay : ℝ) 
  (h1 : speed_a = 10)
  (h2 : speed_b = 20)
  (h3 : delay = 4) :
  catchup_distance speed_a speed_b delay = 80 := by
  sorry

/-- Evaluation of the catchup distance for the given values -/
def result : ℚ := 80

#eval result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_at_80km_l317_31739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l317_31784

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hour_to_m_per_sec (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

/-- Calculates the time taken for an object to travel a given distance at a given speed -/
noncomputable def time_to_cross (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_crossing_time :
  let train_length : ℝ := 160
  let train_speed_km_h : ℝ := 72
  let train_speed_m_s : ℝ := km_per_hour_to_m_per_sec train_speed_km_h
  time_to_cross train_length train_speed_m_s = 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l317_31784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_min_value_inequality_l317_31700

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + 3 * |x|

-- Theorem for the solution set of f(x) ≥ 10
theorem solution_set (x : ℝ) : f x ≥ 10 ↔ x ∈ Set.Iic (-2) ∪ Set.Ici 3 := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value : ∃ (m : ℝ), m = 2 ∧ ∀ (x : ℝ), f x ≥ m := by sorry

-- Theorem for the inequality given a + b + c = m
theorem inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 2) :
  a^2 + b^2 + c^2 ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_min_value_inequality_l317_31700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_points_l317_31745

theorem quiz_points (n : ℕ) (total : ℕ) (increment : ℕ) : 
  n = 8 → 
  total = 360 → 
  increment = 4 → 
  ∃ (first : ℕ), 
    (List.range n).foldl (λ acc i => acc + first + i * increment) 0 = total ∧ 
    first + 2 * increment = 39 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_points_l317_31745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kn_value_l317_31720

theorem max_kn_value (k n : ℕ+) (x y : ℕ → ℤ) (P : Polynomial ℤ) :
  (∀ i j, i ≠ j → x i ≠ x j) →
  (∀ i j, i ≠ j → y i ≠ y j) →
  (∀ i j, x i ≠ y j) →
  (∀ i ≤ k, P.eval (x i) = 54) →
  (∀ i ≤ n, P.eval (y i) = 2013) →
  k * n ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kn_value_l317_31720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_eleven_l317_31781

theorem ceiling_sum_equals_eleven : 
  ⌈Real.sqrt (9/4 : ℝ)⌉ + ⌈(9/4 : ℝ)⌉ + ⌈((9/4 : ℝ)^2)⌉ = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_eleven_l317_31781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l317_31771

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | (n + 2) => (sequence_a (n + 1) - 1) / sequence_a (n + 1)

theorem a_2015_value : sequence_a 2015 = 2/3 := by
  sorry

#eval sequence_a 2015  -- This line is added for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l317_31771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_fixed_point_l317_31730

/-- For any base a > 0 and a ≠ 1, the logarithmic function y = log_a(x-2) + 3 passes through the point (3, 3) -/
theorem log_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => Real.logb a (x - 2) + 3
  f 3 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_fixed_point_l317_31730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_school_year_hours_l317_31791

/-- Represents Ben's work schedule and earnings --/
structure WorkSchedule where
  summer_hours_per_week : ℚ
  summer_weeks : ℚ
  summer_earnings : ℚ
  school_year_weeks : ℚ
  school_year_earnings : ℚ

/-- Calculates the required weekly hours for the school year --/
def required_school_year_hours (w : WorkSchedule) : ℚ :=
  let hourly_wage := w.summer_earnings / (w.summer_hours_per_week * w.summer_weeks)
  let total_hours := w.school_year_earnings / hourly_wage
  total_hours / w.school_year_weeks

/-- Theorem stating that Ben must work 20 hours per week during the school year --/
theorem ben_school_year_hours (w : WorkSchedule) 
  (h1 : w.summer_hours_per_week = 40)
  (h2 : w.summer_weeks = 8)
  (h3 : w.summer_earnings = 3200)
  (h4 : w.school_year_weeks = 24)
  (h5 : w.school_year_earnings = 4800) : 
  required_school_year_hours w = 20 := by
  sorry

#eval required_school_year_hours ⟨40, 8, 3200, 24, 4800⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_school_year_hours_l317_31791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_30_28_12_l317_31741

/-- The area of a triangle with sides a, b, and c -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 30, 28, and 12 is approximately 110.84 -/
theorem triangle_area_30_28_12 :
  ∃ ε > 0, |triangle_area 30 28 12 - 110.84| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_30_28_12_l317_31741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_curve_length_l317_31773

/-- The length of a parametric curve described by (x,y) = (3 cos t, 3 sin t) from t = 0 to t = 3π/2 -/
noncomputable def parametricCurveLength : ℝ := (9 * Real.pi) / 2

/-- The parametric equations of the curve -/
noncomputable def curveEquations (t : ℝ) : ℝ × ℝ := (3 * Real.cos t, 3 * Real.sin t)

/-- The start point of the curve -/
def startPoint : ℝ := 0

/-- The end point of the curve -/
noncomputable def endPoint : ℝ := (3 * Real.pi) / 2

theorem parametric_curve_length :
  ∃ (length : ℝ), length = parametricCurveLength ∧
  (∀ t ∈ Set.Icc startPoint endPoint, curveEquations t ∈ Set.range curveEquations) ∧
  length = (9 * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_curve_length_l317_31773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l317_31786

def b : ℕ → ℚ
  | 0 => 2  -- Add this case to cover Nat.zero
  | 1 => 2
  | 2 => 1
  | (n + 3) => b (n + 2) + b (n + 1)

noncomputable def series_sum : ℚ := ∑' n, b n / 3^(n + 1)

theorem series_sum_value : series_sum = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l317_31786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_wage_increase_l317_31706

/-- Given a worker's original daily wage and a percentage increase, 
    calculate the new daily wage after the increase. -/
noncomputable def new_wage (original_wage : ℝ) (percent_increase : ℝ) : ℝ :=
  original_wage * (1 + percent_increase / 100)

/-- Theorem: The worker's new daily wage after a 50% increase from $34 is $51. -/
theorem worker_wage_increase : new_wage 34 50 = 51 := by
  -- Unfold the definition of new_wage
  unfold new_wage
  -- Simplify the arithmetic expression
  simp [mul_add, mul_one]
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_wage_increase_l317_31706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_contact_lenses_sold_l317_31744

/-- Represents the number of pairs of hard contact lenses sold -/
def H : ℕ := sorry

/-- Represents the number of pairs of soft contact lenses sold -/
def S : ℕ := sorry

/-- The price of a pair of soft contact lenses -/
def soft_price : ℕ := 150

/-- The price of a pair of hard contact lenses -/
def hard_price : ℕ := 85

/-- The total sales amount -/
def total_sales : ℕ := 1455

/-- Theorem stating the total number of contact lenses sold -/
theorem total_contact_lenses_sold :
  (S = H + 5) →
  (soft_price * S + hard_price * H = total_sales) →
  (H + S = 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_contact_lenses_sold_l317_31744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_direction_angle_l317_31735

theorem cosine_direction_angle (Q : ℝ × ℝ × ℝ) 
  (h_positive : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0) 
  (α β γ : ℝ) 
  (h_angles : α = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
              β = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
              γ = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_cos_α : Real.cos α = 4/5)
  (h_cos_β : Real.cos β = 1/2) : 
  Real.cos γ = Real.sqrt 11 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_direction_angle_l317_31735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l317_31750

noncomputable def coneVolume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_volume_ratio :
  let r_C : ℝ := 15
  let h_C : ℝ := 30
  let r_D : ℝ := 30
  let h_D : ℝ := 15
  (coneVolume r_C h_C) / (coneVolume r_D h_D) = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l317_31750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l317_31797

theorem expression_equality : Real.sqrt 4 + |Real.tan (60 * π / 180) - 1| - 2023^0 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l317_31797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l317_31709

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- The area of the triangle formed by the origin and the intersection points of x = a with the asymptotes -/
def triangle_area (h : Hyperbola) : ℝ := h.a * h.b

theorem min_focal_length (h : Hyperbola) (h_area : triangle_area h = 8) :
  ∃ (h_min : Hyperbola), focal_length h_min ≤ focal_length h ∧ focal_length h_min = 8 := by
  sorry

#check min_focal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l317_31709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_N_l317_31732

def N : ℕ := 69^5 + 5*69^4 + 10*69^3 + 10*69^2 + 5*69 + 1

theorem number_of_factors_of_N : 
  (Finset.filter (λ x : ℕ => x > 0 ∧ N % x = 0) (Finset.range (N + 1))).card = 216 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_N_l317_31732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_relationship_l317_31710

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- The area of a circle -/
noncomputable def Circle.area (c : Circle) : ℝ := Real.pi * c.radius^2

theorem circle_area_relationship (A B : Circle) 
  (h1 : A.area = 9)
  (h2 : B.radius = 2 * A.radius) : 
  B.area = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_relationship_l317_31710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_coefficients_l317_31707

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2)) / 2

/-- Given a line ax + by = 1 intersecting the unit circle x² + y² = 1,
    if the triangle formed by the intersection points and the origin has maximum area,
    then a + b is maximized at 2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + b * A.2 = 1) ∧ 
    (a * B.1 + b * B.2 = 1) ∧ 
    (A.1^2 + A.2^2 = 1) ∧ 
    (B.1^2 + B.2^2 = 1) ∧ 
    (∀ C D : ℝ × ℝ, 
      (a * C.1 + b * C.2 = 1) → 
      (a * D.1 + b * D.2 = 1) → 
      (C.1^2 + C.2^2 = 1) → 
      (D.1^2 + D.2^2 = 1) → 
      area_triangle (0, 0) A B ≥ area_triangle (0, 0) C D)) →
  a + b ≤ 2 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_coefficients_l317_31707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_zeros_iff_omega_range_l317_31772

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x + |Real.log x| - 2
  else if -Real.pi ≤ x then Real.sin (ω * x + Real.pi / 4) - 1 / 2
  else 0

def has_seven_distinct_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ),
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
    x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
    x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
    x₆ ≠ x₇ ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 ∧ f x₆ = 0 ∧ f x₇ = 0

theorem seven_zeros_iff_omega_range :
  ∀ ω : ℝ, has_seven_distinct_zeros (f ω) ↔ 49/12 ≤ ω ∧ ω < 65/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_zeros_iff_omega_range_l317_31772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_recursive_formula_l317_31724

/-- The number of diagonals in a convex polygon with k sides -/
def f (k : ℕ) : ℕ := sorry

/-- Theorem: For k ≥ 3, the number of diagonals in a convex polygon with k+1 sides
    is equal to the number of diagonals in a polygon with k sides plus k-1 -/
theorem diagonals_recursive_formula (k : ℕ) (h : k ≥ 3) : f (k + 1) = f k + (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_recursive_formula_l317_31724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_fixed_point_on_line_l317_31728

noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + 15 * y^2 / 16 = 1

def chord_length (p : ℝ) : ℝ := 2 * p

noncomputable def angle_of_inclination (x y : ℝ) : ℝ := Real.arctan (y / x)

theorem parabola_ellipse_intersection (p : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    chord_length p = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ↔
  p = 1 :=
sorry

theorem fixed_point_on_line (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧
  angle_of_inclination x₁ y₁ + angle_of_inclination x₂ y₂ = Real.arctan 2 →
  ∃ t : ℝ, t * x₁ + (1 - t) * x₂ = -2 ∧ t * y₁ + (1 - t) * y₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_fixed_point_on_line_l317_31728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_parallel_chords_l317_31793

-- Define the two lines
def line1 (p : ℝ × ℝ) : Prop := p.1 - Real.sqrt 3 * p.2 + 2 = 0
def line2 (p : ℝ × ℝ) : Prop := p.1 - Real.sqrt 3 * p.2 - 6 = 0

-- Define the circle
def myCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the chord length
def chordLength (c : Set (ℝ × ℝ)) (l : (ℝ × ℝ) → Prop) : ℝ := 2

-- Theorem statement
theorem circle_area_with_parallel_chords
  (c : Set (ℝ × ℝ))
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h1 : c = myCircle center radius)
  (h2 : chordLength c line1 = 2)
  (h3 : chordLength c line2 = 2)
  : Real.pi * radius^2 = 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_parallel_chords_l317_31793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_cube_l317_31785

theorem sphere_volume_circumscribing_cube (cube_surface_area : Real) (h : cube_surface_area = 24) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := cube_edge * Real.sqrt 3 / 2
  (4 / 3) * Real.pi * sphere_radius^3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_cube_l317_31785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l317_31796

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 5)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let mag := magnitude v
  (v.1 / mag, v.2 / mag)

theorem unit_vector_AB_is_correct :
  unit_vector vector_AB = (-3/5, 4/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l317_31796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_is_two_l317_31711

/-- The minimum value of ω that satisfies the given conditions -/
noncomputable def min_omega : ℝ := 2

/-- The function representing the translated sine graph -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x - Real.pi / 4))

/-- Theorem stating that the minimum value of ω satisfying the conditions is 2 -/
theorem min_omega_is_two :
  ∀ ω : ℝ, ω > 0 →
  (∀ x : ℝ, f ω x = f ω (3 * Real.pi / 2 - x)) →
  ω ≥ min_omega := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_is_two_l317_31711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_l317_31783

/-- Represents the time lost due to stoppages for a train journey -/
noncomputable def time_lost (distance : ℝ) (speed_without_stop : ℝ) (speed_with_stop : ℝ) : ℝ :=
  distance / speed_with_stop - distance / speed_without_stop

/-- Theorem stating that the time lost due to stoppages is 15 minutes per hour -/
theorem train_stop_time 
  (distance : ℝ) 
  (speed_without_stop : ℝ) 
  (speed_with_stop : ℝ) 
  (h1 : speed_without_stop = 80) 
  (h2 : speed_with_stop = 60) 
  (h3 : distance > 0) 
  (h4 : speed_without_stop > speed_with_stop) :
  time_lost distance speed_without_stop speed_with_stop * 60 = 15 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stop_time_l317_31783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meteor_encounter_time_l317_31763

-- Define the time intervals for encountering meteors
def time_towards : ℝ := 7
def time_same_direction : ℝ := 13

-- Define the theorem
theorem meteor_encounter_time :
  let rate_towards := 1 / time_towards
  let rate_same_direction := 1 / time_same_direction
  let total_rate := rate_towards + rate_same_direction
  let harmonic_mean := 2 * rate_towards * rate_same_direction / total_rate
  let encounter_time := 1 / harmonic_mean
  ∃ (ε : ℝ), ε > 0 ∧ |encounter_time - 9.1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meteor_encounter_time_l317_31763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_removal_l317_31712

theorem parentheses_removal (a b c d : ℤ) :
  (-a) - (-b) + c - (-d) = -a + b + c + d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parentheses_removal_l317_31712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l317_31755

theorem quadratic_equation_solution :
  let f : ℂ → ℂ := λ y => y^2 - 5*y + 6 + (y + 1)*(y + 7)
  ∃ y₁ y₂ : ℂ, y₁ = (-3 + Complex.I * Real.sqrt 95) / 4 ∧
             y₂ = (-3 - Complex.I * Real.sqrt 95) / 4 ∧
             f y₁ = 0 ∧ f y₂ = 0 ∧
             ∀ y : ℂ, f y = 0 → y = y₁ ∨ y = y₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l317_31755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l317_31734

theorem ascending_order (a b c : ℝ) 
  (ha : a = Real.log (1/2)) 
  (hb : b = (1/3)^(4/5 : ℝ)) 
  (hc : c = 2^(1/3 : ℝ)) :
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l317_31734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l317_31715

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  S_5_eq_30 : S 5 = 30
  a_4_eq_8 : a 4 = 8

/-- The sum of the first n terms of the sequence b_n -/
noncomputable def T (n : ℕ) : ℚ := 2/9 * ((3*n - 1) * 4^(n+1) + 4)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2 * n) ∧
  (∀ n, seq.S n = n^2 + n) ∧
  (∀ n, T n = (2/9) * ((3*n - 1) * 4^(n+1) + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l317_31715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l317_31705

/-- An ellipse with the given properties has one of two standard equations -/
theorem ellipse_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 = b^2 + c^2) -- standard ellipse relation
  (h5 : b = Real.sqrt 3 * c) -- equilateral triangle condition
  (h6 : a - c = Real.sqrt 3) -- shortest distance condition
  : (∀ x y : ℝ, x^2/12 + y^2/9 = 1 ∨ x^2/9 + y^2/12 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l317_31705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l317_31766

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l317_31766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_sum_l317_31713

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0, 
    with foci at F₁(-c, 0) and F₂(c, 0), and points A and B on the ellipse 
    above the x-axis such that AF₁ ∥ BF₂, and P is the intersection of 
    lines AF₂ and BF₁, prove that PF₁ + PF₂ is constant. -/
theorem ellipse_constant_sum (a b c : ℝ) (A B P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧
  (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
  (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
  A.2 > 0 ∧ B.2 > 0 ∧
  (A.1 - F₁.1) * (B.2 - F₂.2) = (A.2 - F₁.2) * (B.1 - F₂.1) ∧
  (∃ t : ℝ, P = A + t • (F₂ - A)) ∧
  (∃ s : ℝ, P = B + s • (F₁ - B)) →
  dist P F₁ + dist P F₂ = (a^2 + c^2) / a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_sum_l317_31713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l317_31729

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2)

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l317_31729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l317_31757

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (5^3 + 7^3)^(1/3) → 
  ∃ (n : ℤ), ∀ (m : ℤ), |x - ↑n| ≤ |x - ↑m| ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l317_31757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_tan_A_value_l317_31751

noncomputable section

variable (a b c A B C : ℝ)

def triangle_abc (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem angle_B_measure
  (h1 : triangle_abc a b c)
  (h2 : a^2 + c^2 - b^2 = a * c)
  (h3 : B = Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) :
  B = π / 3 := by sorry

theorem tan_A_value
  (h1 : triangle_abc a b c)
  (h2 : a^2 + c^2 - b^2 = a * c)
  (h3 : c = 3 * a)
  (h4 : A = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) :
  Real.tan A = Real.sqrt 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_tan_A_value_l317_31751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_1061_base9_l317_31780

/-- Convert a base-9 number represented as a list of digits to a natural number. -/
def toNat9 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Convert a natural number to its base-9 representation. -/
def toBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- The main theorem stating that the sum of the given base-9 numbers equals 1061 in base 9. -/
theorem sum_equals_1061_base9 :
  toBase9 (toNat9 [1, 7, 5] + toNat9 [7, 1, 4] + toNat9 [6, 1]) = [1, 0, 6, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_1061_base9_l317_31780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_l317_31740

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 3)

-- State the theorem
theorem symmetry_point :
  ∀ (x : ℝ), f (Real.pi / 3 - x) = -f (Real.pi / 3 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_l317_31740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_colorings_l317_31743

/-- Represents a coloring of a cube with two sides of each color --/
structure CubeColoring where
  blue_sides : Fin 6 × Fin 6
  red_sides : Fin 6 × Fin 6
  green_sides : Fin 6 × Fin 6
  distinct_sides : blue_sides.1 ≠ blue_sides.2 ∧ 
                   red_sides.1 ≠ red_sides.2 ∧ 
                   green_sides.1 ≠ green_sides.2
  all_sides_colored : {blue_sides.1, blue_sides.2, red_sides.1, red_sides.2, green_sides.1, green_sides.2} = Finset.univ

/-- Two cube colorings are equivalent if they can be rotated to match each other --/
def coloring_equiv (c1 c2 : CubeColoring) : Prop := sorry

/-- The set of all distinct cube colorings up to rotation --/
noncomputable def distinct_colorings : Finset CubeColoring := sorry

/-- The number of distinct cube colorings up to rotation is 12 --/
theorem num_distinct_colorings : Finset.card distinct_colorings = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_colorings_l317_31743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_area_ratio_l317_31733

theorem window_area_ratio :
  ∀ (rect_height : ℝ) (rect_length : ℝ) (semicircle_radius : ℝ),
    rect_height = 45 →
    rect_length = (4/3) * rect_height →
    semicircle_radius = rect_height / 2 →
    (rect_length * rect_height) / (π * semicircle_radius^2) = 17 / π :=
by
  intros rect_height rect_length semicircle_radius h_height h_length h_radius
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_area_ratio_l317_31733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_ratio_sequences_l317_31790

def is_positive_divisor_of_24 (n : ℕ) : Prop := n > 0 ∧ 24 % n = 0

def valid_sequence (s : Fin 8 → ℕ) : Prop :=
  (∀ i : Fin 8, is_positive_divisor_of_24 (s i)) ∧
  (∀ i : Fin 4, s (2*i) / s (2*i+1) = s 0 / s 1) ∧
  (∃ k : ℕ, s 0 = k * s 1) ∧
  (∀ i j : Fin 8, i ≠ j → s i ≠ s j)

theorem equal_ratio_sequences :
  ∃! (s₁ s₂ : Fin 8 → ℕ),
    valid_sequence s₁ ∧ valid_sequence s₂ ∧
    s₁ 0 / s₁ 1 = 2 ∧ s₂ 0 / s₂ 1 = 3 ∧
    (∀ s : Fin 8 → ℕ, valid_sequence s → (s = s₁ ∨ s = s₂)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_ratio_sequences_l317_31790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l317_31721

/-- Given a line y = x + 2a intersecting a circle x^2 + y^2 - 2ay - 2 = 0 (where a > 0)
    at points A and B, if |AB| = 2√3, then a = √2. -/
theorem intersection_line_circle (a : ℝ) (A B : ℝ × ℝ) :
  a > 0 →
  (∀ x y, y = x + 2*a ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) →
  (∀ x y, x^2 + y^2 - 2*a*y - 2 = 0 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) →
  ‖A - B‖ = 2 * Real.sqrt 3 →
  a = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l317_31721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l317_31792

noncomputable section

def triangle : ℝ → ℝ → ℝ := sorry

axiom triangle_def (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  triangle a (triangle b c) = (a / b) * c

axiom triangle_self (a : ℝ) (ha : a ≠ 0) : triangle a a = 2

theorem triangle_solution :
  ∃ x : ℝ, x ≠ 0 ∧ triangle 8 (triangle 4 x) = 48 ∧ x = 1/6 ∧ triangle 2 (1/6) = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l317_31792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l317_31779

theorem remainder_sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l317_31779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l317_31752

-- Define the function f as noncomputable due to its dependence on Real.exp
noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x > 0 then Real.exp x - a * x + Real.exp 3 else -Real.exp (-x) - a * x - Real.exp 3

-- State the theorem
theorem min_value_of_a (a : ℝ) :
  (∀ x, x ≠ 0 → f a x = -f a (-x)) →  -- f is odd
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧  -- arithmetic sequence
    x₁ + x₄ = 0 ∧
    ∃ d : ℝ, x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d ∧  -- arithmetic sequence
    ∃ q : ℝ, q ≠ 0 ∧  -- geometric sequence
      f a x₂ = q * f a x₁ ∧
      f a x₃ = q * f a x₂ ∧
      f a x₄ = q * f a x₃) →
  a ≥ 3/4 * Real.exp 3 + 1/4 * Real.exp 1 :=
by
  sorry  -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l317_31752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_distance_l317_31776

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem ellipse_constant_distance (x₁ y₁ x₂ y₂ : ℝ) :
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ (a b c : ℝ), distance_point_to_line 0 0 a b c = 2 * Real.sqrt 21 / 7 :=
by
  sorry

#check ellipse_constant_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_distance_l317_31776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_abs_2_l317_31702

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

/-- The absolute value (magnitude) of a complex number -/
noncomputable def complex_abs (z : ℂ) : ℝ := Real.sqrt (z.re * z.re + z.im * z.im)

theorem complex_second_quadrant_abs_2 :
  ∃ z : ℂ, in_second_quadrant z ∧ complex_abs z = 2 ∧ z = -1 + Complex.I * Real.sqrt 3 := by
  sorry

#check complex_second_quadrant_abs_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_abs_2_l317_31702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_range_lower_bound_l317_31727

def is_median (s : Finset Int) (m : Int) : Prop :=
  s.card % 2 = 0 ∧ 
  (s.filter (λ x => x ≤ m)).card = s.card / 2 ∧
  (s.filter (λ x => x ≥ m)).card = s.card / 2

def range (s : Finset Int) : Int :=
  if h : s.Nonempty then s.max' h - s.min' h else 0

theorem set_range_lower_bound (x : Finset Int) (h1 : x.card = 10) 
  (h2 : is_median x 30) (h3 : ∃ m : Int, m ∈ x ∧ m = 50) : 
  range x ≥ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_range_lower_bound_l317_31727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_range_of_f_t_l317_31770

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1/2 - 2/(Real.exp x + 1)
  else 2/(Real.exp x + 1) - 3/2

-- Theorem for the zeros of f
theorem zeros_of_f :
  {x : ℝ | f x = 0} = {Real.log 3, -Real.log 3} := by
  sorry

-- Theorem for the range of f(t) given the condition
theorem range_of_f_t (t : ℝ) :
  (f (Real.log 2 * t) + f (Real.log 2 * (1/t)) < 2 * f 2) →
  (1/2 - 2/(Real.exp (1/4) + 1) < f t ∧ f t < 1/2 - 2/(Real.exp 4 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_range_of_f_t_l317_31770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_function_properties_l317_31789

-- Define the Gauss function
noncomputable def gauss (x : ℝ) : ℤ := Int.floor x

-- Define f(x)
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (1 + 2^x)

-- Define g(x)
noncomputable def g (x : ℝ) : ℤ := gauss (f x)

-- Theorem statement
theorem gauss_function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range g = {-1, 0}) := by
  sorry

#check gauss_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_function_properties_l317_31789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_total_payment_l317_31756

/-- Represents the payment structure for an appliance purchase -/
structure AppliancePayment where
  initialPrice : ℚ
  downPayment : ℚ
  monthlyPayment : ℚ
  monthlyInterestRate : ℚ

/-- Calculates the total amount paid for an appliance purchase -/
def totalAmountPaid (payment : AppliancePayment) : ℚ :=
  payment.initialPrice + (20 * (20 + 1)) / 2

/-- Theorem stating the total amount paid for a specific appliance purchase -/
theorem appliance_total_payment :
  let payment : AppliancePayment := {
    initialPrice := 2250,
    downPayment := 250,
    monthlyPayment := 100,
    monthlyInterestRate := 1/100
  }
  totalAmountPaid payment = 2460 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_total_payment_l317_31756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_add_two_can_map_odd_to_prime_l317_31799

-- Define the operations
def exp_cube (n : ℤ) : ℤ := n^3

noncomputable def log3_floor (n : ℤ) : ℤ := Int.floor (Real.log n / Real.log 3)

def add_two (n : ℤ) : ℤ := n + 2
def mult_three (n : ℤ) : ℤ := 3 * n
def sub_one (n : ℤ) : ℤ := n - 1

-- Define properties
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_prime (n : ℤ) : Prop := Nat.Prime n.natAbs

-- Theorem statement
theorem only_add_two_can_map_odd_to_prime :
  ∀ n : ℤ, is_odd n →
    (∃ m : ℤ, is_odd m ∧ is_prime (add_two m)) ∧
    (∀ m : ℤ, is_odd m → ¬is_prime (exp_cube m)) ∧
    (∀ m : ℤ, is_odd m → ¬is_prime (log3_floor m)) ∧
    (∀ m : ℤ, is_odd m → m ≠ 1 → ¬is_prime (mult_three m)) ∧
    (∀ m : ℤ, is_odd m → m ≠ 3 → ¬is_prime (sub_one m)) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_add_two_can_map_odd_to_prime_l317_31799
