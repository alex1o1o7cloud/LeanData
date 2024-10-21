import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_sum_max_l701_70195

theorem triangle_cos_sum_max (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- angles are positive
  A + B + C = Real.pi ∧ -- sum of angles in a triangle
  Real.sin A ^ 2 + Real.cos B ^ 2 = 1 → -- given condition
  (∃ (x y z : ℝ), x + y + z ≤ Real.cos A + Real.cos B + Real.cos C ∧
                     x + y + z ≤ 3/2 ∧
                     ∀ (a b c : ℝ), a + b + c ≤ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cos_sum_max_l701_70195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_john_finds_train_probability_l701_70152

/-- The time range in minutes for both John and the train to arrive -/
noncomputable def timeRange : ℝ := 120

/-- The waiting time of the train in minutes -/
noncomputable def waitTime : ℝ := 30

/-- The probability that John finds the train at the station -/
noncomputable def probabilityJohnFindsTrainAtStation : ℝ := 7 / 32

/-- 
Theorem stating that the probability of John finding the train at the station
is 7/32, given the conditions of the problem.
-/
theorem john_finds_train_probability : 
  probabilityJohnFindsTrainAtStation = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_john_finds_train_probability_l701_70152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_size_l701_70185

/-- A function that returns true if a set of natural numbers satisfies the condition
    that no member is 4 times another member -/
def validSet (s : Finset ℕ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ 4 * y

/-- The theorem stating that the largest subset of integers from 1 to 150,
    where no member is 4 times another member, has 120 elements -/
theorem largest_subset_size :
  ∃ (s : Finset ℕ), s ⊆ Finset.range 150 ∧ validSet s ∧ s.card = 120 ∧
  ∀ (t : Finset ℕ), t ⊆ Finset.range 150 → validSet t → t.card ≤ 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_size_l701_70185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_line_l701_70136

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line x - √3y = 0 -/
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

/-- The focus of the parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- The distance from a point (x₀, y₀) to the line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem distance_focus_to_line :
  distance_point_to_line focus.1 focus.2 1 (-Real.sqrt 3) 0 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_line_l701_70136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_distance_l701_70199

noncomputable section

/-- The distance from Andrey's home to the airport in kilometers -/
def total_distance : ℝ := 180

/-- The initial speed in km/h -/
def initial_speed : ℝ := 60

/-- The increased speed in km/h -/
def increased_speed : ℝ := 90

/-- The initial driving time in hours -/
def initial_drive_time : ℝ := 1

/-- The time difference in hours (20 minutes = 1/3 hour) -/
def time_difference : ℝ := 1/3

theorem airport_distance : 
  ∃ (planned_time : ℝ),
    -- Distance covered in the first hour
    initial_speed * initial_drive_time +
    -- Remaining distance at increased speed
    increased_speed * (planned_time - time_difference - initial_drive_time) = 
    total_distance ∧
    -- Remaining distance if continued at initial speed
    initial_speed * (planned_time + time_difference - initial_drive_time) =
    total_distance - initial_speed * initial_drive_time :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_distance_l701_70199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_interval_g_greater_than_f_condition_l701_70130

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -2 * Real.cos x - x
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.log x + k / x

-- Theorem for the interval of monotonic increase of f
theorem f_monotonic_increase_interval (k : ℤ) :
  ∃ (a b : ℝ), a = 2 * π * ↑k + π / 6 ∧ b = 2 * π * ↑k + 5 * π / 6 ∧
  ∀ x ∈ Set.Ioo a b, HasDerivAt f ((2 * Real.sin x - 1) : ℝ) x ∧ 2 * Real.sin x - 1 > 0 :=
sorry

-- Theorem for the range of k
theorem g_greater_than_f_condition (k : ℝ) :
  (∀ x₁ ∈ Set.Icc 0 (1/2), ∀ x₂ ∈ Set.Icc (1/2) 1, f x₁ < g k x₂) ↔
  k > -1 + 1/2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_interval_g_greater_than_f_condition_l701_70130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_A_is_9_l701_70197

/-- Represents a telephone number in the format ABC-DEF-GHIJ --/
structure TelephoneNumber where
  A : Nat
  B : Nat
  C : Nat
  D : Nat
  E : Nat
  F : Nat
  G : Nat
  H : Nat
  I : Nat
  J : Nat

/-- Checks if all digits in the telephone number are different --/
def all_digits_different (n : TelephoneNumber) : Prop :=
  n.A ≠ n.B ∧ n.A ≠ n.C ∧ n.A ≠ n.D ∧ n.A ≠ n.E ∧ n.A ≠ n.F ∧ n.A ≠ n.G ∧ n.A ≠ n.H ∧ n.A ≠ n.I ∧ n.A ≠ n.J ∧
  n.B ≠ n.C ∧ n.B ≠ n.D ∧ n.B ≠ n.E ∧ n.B ≠ n.F ∧ n.B ≠ n.G ∧ n.B ≠ n.H ∧ n.B ≠ n.I ∧ n.B ≠ n.J ∧
  n.C ≠ n.D ∧ n.C ≠ n.E ∧ n.C ≠ n.F ∧ n.C ≠ n.G ∧ n.C ≠ n.H ∧ n.C ≠ n.I ∧ n.C ≠ n.J ∧
  n.D ≠ n.E ∧ n.D ≠ n.F ∧ n.D ≠ n.G ∧ n.D ≠ n.H ∧ n.D ≠ n.I ∧ n.D ≠ n.J ∧
  n.E ≠ n.F ∧ n.E ≠ n.G ∧ n.E ≠ n.H ∧ n.E ≠ n.I ∧ n.E ≠ n.J ∧
  n.F ≠ n.G ∧ n.F ≠ n.H ∧ n.F ≠ n.I ∧ n.F ≠ n.J ∧
  n.G ≠ n.H ∧ n.G ≠ n.I ∧ n.G ≠ n.J ∧
  n.H ≠ n.I ∧ n.H ≠ n.J ∧
  n.I ≠ n.J

/-- Checks if the digits are in decreasing order in each segment --/
def digits_in_decreasing_order (n : TelephoneNumber) : Prop :=
  n.A > n.B ∧ n.B > n.C ∧ n.D > n.E ∧ n.E > n.F ∧ n.G > n.H ∧ n.H > n.I ∧ n.I > n.J

/-- Checks if D, E, and F are consecutive digits --/
def DEF_consecutive (n : TelephoneNumber) : Prop :=
  n.E = n.D - 1 ∧ n.F = n.E - 1

/-- Checks if G, H, I, and J are consecutive digits starting from an even number --/
def GHIJ_consecutive_even (n : TelephoneNumber) : Prop :=
  n.G % 2 = 0 ∧ n.H = n.G - 1 ∧ n.I = n.H - 1 ∧ n.J = n.I - 1

/-- The main theorem stating that A must be 9 given the conditions --/
theorem telephone_number_A_is_9 (n : TelephoneNumber) 
  (h1 : all_digits_different n)
  (h2 : digits_in_decreasing_order n)
  (h3 : DEF_consecutive n)
  (h4 : GHIJ_consecutive_even n)
  (h5 : n.A + n.B + n.C = 10) :
  n.A = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_A_is_9_l701_70197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_volume_l701_70124

/-- The ellipsoid equation -/
def ellipsoid (a b c x y z : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1

/-- The volume bounded by coordinate planes and a tangent plane -/
noncomputable def volume (a b c x y z : ℝ) : ℝ :=
  (a^2 * b^2 * c^2) / (6 * x * y * z)

/-- The theorem stating the smallest volume -/
theorem smallest_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), ellipsoid a b c x y z ∧
    (∀ (x' y' z' : ℝ), ellipsoid a b c x' y' z' →
      volume a b c x y z ≤ volume a b c x' y' z') ∧
    volume a b c x y z = Real.sqrt 3 / 2 * a * b * c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_volume_l701_70124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l701_70167

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  isosceles : dist A B = dist B C

-- Define a median
def is_median (A B C D : ℝ × ℝ) : Prop :=
  D.1 = (B.1 + C.1) / 2 ∧ D.2 = (B.2 + C.2) / 2

-- Define perpendicularity
def perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem statement
theorem isosceles_triangle_area 
  (A B C : ℝ × ℝ) 
  (D : ℝ × ℝ) -- Endpoint of median BD
  (E : ℝ × ℝ) -- Endpoint of median CE
  (h1 : Triangle A B C)
  (h2 : is_median A B C D)
  (h3 : is_median A C B E)
  (h4 : perpendicular B D C E)
  (h5 : dist B D = 15)
  (h6 : dist C E = 18) :
  triangle_area A B C = 540 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l701_70167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_removal_increases_probability_l701_70148

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def sumPairs (s : Finset Nat) : Finset (Nat × Nat) :=
  s.product s |>.filter (fun p => p.1 < p.2 ∧ p.1 + p.2 = 14)

noncomputable def probability (s : Finset Nat) : Rat :=
  (sumPairs s).card / Nat.choose s.card 2

theorem seven_removal_increases_probability :
  ∀ n ∈ S, n ≠ 7 → probability (S.erase 7) > probability (S.erase n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_removal_increases_probability_l701_70148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l701_70109

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem f_decreasing_on_interval :
  ∀ x y, π/6 ≤ x ∧ x < y ∧ y ≤ π → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l701_70109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_max_value_is_two_max_value_achieved_l701_70161

theorem max_value_x_plus_2y (x y : ℝ) (h : (2:ℝ)^x + (4:ℝ)^y = 4) : 
  ∀ a b : ℝ, (2:ℝ)^a + (4:ℝ)^b = 4 → x + 2*y ≥ a + 2*b :=
by sorry

theorem max_value_is_two (x y : ℝ) (h : (2:ℝ)^x + (4:ℝ)^y = 4) : 
  x + 2*y ≤ 2 :=
by sorry

theorem max_value_achieved (x y : ℝ) (h : (2:ℝ)^x + (4:ℝ)^y = 4) :
  ∃ a b : ℝ, (2:ℝ)^a + (4:ℝ)^b = 4 ∧ a + 2*b = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_max_value_is_two_max_value_achieved_l701_70161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_286_roots_l701_70184

/-- A function satisfying the given symmetry conditions -/
noncomputable def g : ℝ → ℝ := sorry

/-- The symmetry condition around 3 -/
axiom symmetry_3 : ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The symmetry condition around 8 -/
axiom symmetry_8 : ∀ x : ℝ, g (8 + x) = g (8 - x)

/-- The given root at x = 1 -/
axiom root_at_1 : g 1 = 0

/-- The set of roots of g in the interval [-1000, 1000] -/
def roots : Set ℝ :=
  {x : ℝ | -1000 ≤ x ∧ x ≤ 1000 ∧ g x = 0}

/-- The theorem stating that there are at least 286 roots -/
theorem at_least_286_roots : ∃ (s : Finset ℝ), s.card ≥ 286 ∧ ∀ x ∈ s, x ∈ roots := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_286_roots_l701_70184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_draining_rate_l701_70119

/-- Represents the filling rate of a pipe in liters per minute -/
def FillingRate := ℝ

/-- Represents the draining rate of a pipe in liters per minute -/
def DrainingRate := ℝ

/-- Represents the capacity of a tank in liters -/
def TankCapacity := ℝ

/-- Represents the duration of the filling process in minutes -/
def FillingDuration := ℝ

/-- Represents a cycle of filling and draining -/
structure FillDrainCycle where
  pipeA : FillingRate
  pipeB : FillingRate
  pipeC : DrainingRate

/-- The theorem stating the draining rate of pipe C -/
theorem determine_draining_rate 
  (capacity : TankCapacity)
  (cycle : FillDrainCycle)
  (duration : FillingDuration) :
  capacity = (750 : ℝ) ∧ 
  cycle.pipeA = (40 : ℝ) ∧ 
  cycle.pipeB = (30 : ℝ) ∧ 
  duration = (45 : ℝ) → 
  cycle.pipeC = (20 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_draining_rate_l701_70119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l701_70165

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem shortest_chord_length :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), circle_equation x y → 
      ∃ (t : ℝ), (t * x, t * y) = point_P ∧ 
        ∀ (other_length : ℝ), 
          (∃ (x' y' : ℝ), circle_equation x' y' ∧ 
            ∃ (t' : ℝ), (t' * x', t' * y') = point_P ∧ 
              other_length = 2 * Real.sqrt ((x - x')^2 + (y - y')^2)) →
          other_length ≥ chord_length) ∧
    chord_length = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l701_70165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_weeks_per_month_proof_l701_70116

/-- Proves that Josh and Carl work 4 weeks a month given their work schedules and pay rates. -/
def work_weeks_per_month : ℚ :=
  let josh_hours_per_day : ℕ := 8
  let josh_days_per_week : ℕ := 5
  let carl_hours_per_day : ℕ := josh_hours_per_day - 2
  let josh_hourly_rate : ℚ := 9
  let carl_hourly_rate : ℚ := josh_hourly_rate / 2
  let total_monthly_pay : ℚ := 1980

  let josh_weekly_hours : ℕ := josh_hours_per_day * josh_days_per_week
  let carl_weekly_hours : ℕ := carl_hours_per_day * josh_days_per_week
  let josh_weekly_pay : ℚ := josh_weekly_hours * josh_hourly_rate
  let carl_weekly_pay : ℚ := carl_weekly_hours * carl_hourly_rate
  let total_weekly_pay : ℚ := josh_weekly_pay + carl_weekly_pay

  total_monthly_pay / total_weekly_pay

/-- Proof of the theorem -/
theorem work_weeks_per_month_proof : work_weeks_per_month = 4 := by
  unfold work_weeks_per_month
  -- The proof steps would go here
  sorry

#eval work_weeks_per_month

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_weeks_per_month_proof_l701_70116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_station_location_l701_70163

/-- Represents the freight cost ratio between railway and road transportation -/
def freight_cost_ratio : ℚ := 3 / 5

/-- Calculates the total cost of transportation from C to A via D -/
noncomputable def total_cost (d : ℝ) : ℝ :=
  3 * (100 - d) + 5 * Real.sqrt (d^2 + 20^2)

/-- Theorem stating the optimal location of station D -/
theorem optimal_station_location :
  ∃ (d : ℝ), d = 15 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 100 → total_cost d ≤ total_cost x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_station_location_l701_70163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_a_theorem_l701_70138

/-- The triangle formed by the letter A -/
structure LetterA where
  left_side : ℝ → ℝ
  right_side : ℝ → ℝ
  top_vertex : ℝ × ℝ

/-- The properties of the letter A -/
def letter_a_properties (a : LetterA) : Prop :=
  a.left_side = (λ x => 3 * x + 6) ∧
  a.right_side = (λ x => -3 * x + 6) ∧
  a.top_vertex = (0, 6)

/-- The area above the line y = c -/
noncomputable def area_above_line (c : ℝ) (a : LetterA) : ℝ := sorry

/-- The total area of the triangle -/
noncomputable def total_area (a : LetterA) : ℝ := sorry

/-- The theorem to be proved -/
theorem letter_a_theorem (a : LetterA) (c : ℝ) 
  (h : letter_a_properties a) :
  (∃ (area_ratio : ℝ), 
    area_ratio = 4/9 ∧ 
    (area_above_line c a) = area_ratio * (total_area a)) → 
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_a_theorem_l701_70138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_positive_power_of_two_l701_70175

theorem negation_of_all_positive_power_of_two :
  (¬ ∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x₀ : ℝ, (2 : ℝ)^x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_positive_power_of_two_l701_70175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_chord_length_l701_70134

/-- Definition of the circle C -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- Definition of the line -/
def line_equation (x y : ℝ) : Prop := y = x

/-- The chord length formed by the intersection of the circle and the line -/
noncomputable def chord_length : ℝ := Real.sqrt 2

theorem circle_line_intersection_chord_length :
  ∀ x y : ℝ, circle_equation x y → line_equation x y →
  ∃ x1 y1 x2 y2 : ℝ,
    circle_equation x1 y1 ∧ line_equation x1 y1 ∧
    circle_equation x2 y2 ∧ line_equation x2 y2 ∧
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = chord_length :=
by
  sorry

#check circle_line_intersection_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_chord_length_l701_70134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_count_l701_70164

/-- Represents a team in the badminton tournament -/
structure Team where
  size : Nat
  h_size : size = 2 ∨ size = 3

/-- The tournament setup -/
structure Tournament where
  teams : Fin 4 → Team
  h_teams : (teams 0).size = 2 ∧ (teams 1).size = 2 ∧ (teams 2).size = 3 ∧ (teams 3).size = 3

/-- Calculate the total number of players in the tournament -/
def totalPlayers (t : Tournament) : Nat :=
  (t.teams 0).size + (t.teams 1).size + (t.teams 2).size + (t.teams 3).size

/-- Calculate the number of handshakes for a single player -/
def handshakesPerPlayer (t : Tournament) (teamSize : Nat) : Nat :=
  totalPlayers t - teamSize

/-- Calculate the total number of handshakes in the tournament -/
def totalHandshakes (t : Tournament) : Nat :=
  let twoPlayerHandshakes := 2 * (handshakesPerPlayer t 2) * 2 / 2
  let threePlayerHandshakes := 2 * (handshakesPerPlayer t 3) * 3 / 2
  twoPlayerHandshakes + threePlayerHandshakes

/-- The main theorem stating that the total number of handshakes is 37 -/
theorem handshake_count (t : Tournament) : totalHandshakes t = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_count_l701_70164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l701_70102

-- Define set M
def M : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Define set N
def N : Set ℝ := {x | (1/2 : ℝ)^x ≥ 4}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | x ≤ -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l701_70102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_inch_gold_cube_worth_l701_70170

/-- The worth of a cube of gold given its side length -/
noncomputable def gold_worth (side_length : ℝ) : ℝ :=
  800 * (side_length ^ 3 / 4 ^ 3)

/-- Theorem stating the worth of a 5-inch cube of gold -/
theorem five_inch_gold_cube_worth :
  ⌊gold_worth 5⌋ = 1563 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_inch_gold_cube_worth_l701_70170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_of_sequence_l701_70151

def a : ℕ → ℕ
| 0 => 5
| n + 1 => a n ^ 3 - 2 * (a n ^ 2) + 2

theorem prime_divisor_of_sequence (p : ℕ) (hp : Prime p) :
  p ∣ (a 2013 + 1) → p % 4 = 3 → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_of_sequence_l701_70151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_fib_l701_70159

/-- The function f(n) counts the number of ways to express a positive integer n
    as a sum of positive odd integers, where the order matters. -/
def f : ℕ+ → ℕ := sorry

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: f(n) is equal to the n-th Fibonacci number for all positive integers n -/
theorem f_eq_fib : ∀ n : ℕ+, f n = fib n.val :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_fib_l701_70159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_theta_l701_70179

open Real

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the relationship between f and g
axiom f_def : ∀ x, f x = 2 * g x + (x - 4) / (x^2 + 1)

-- Define the inequality condition
def inequality_condition (θ : ℝ) : Prop :=
  f (1 / sin θ) + f (cos (2 * θ)) < f π - f (1 / π)

-- Theorem to prove
theorem range_of_theta :
  ∀ θ : ℝ, inequality_condition θ ↔ 
    ∃ k : ℤ, θ ∈ Set.Ioo (2 * k * π - 5 * π / 6) (2 * k * π - π / 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_theta_l701_70179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lasso_theorem_l701_70198

noncomputable def prob_success_in_3_attempts : ℝ := 0.875

noncomputable def prob_single_success : ℝ := 1 - (1 - prob_success_in_3_attempts) ^ (1/3 : ℝ)

noncomputable def avg_throws : ℝ := 1 / prob_single_success

theorem lasso_theorem : avg_throws = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lasso_theorem_l701_70198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l701_70176

noncomputable section

-- Define the parabola
def parabola (x : ℝ) : ℝ := (1/4) * x^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the angle of the line passing through the focus
def angle : ℝ := Real.pi/6  -- 30 degrees in radians

-- Define the slope of the line
noncomputable def line_slope : ℝ := Real.tan angle

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    (A.2 = parabola A.1) ∧
    (B.2 = parabola B.1) ∧
    (A.2 - focus.2 = line_slope * (A.1 - focus.1)) ∧
    (B.2 - focus.2 = line_slope * (B.1 - focus.1)) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l701_70176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l701_70154

theorem coefficient_of_x_in_expansion : 
  let p : Polynomial ℤ := Polynomial.X^2 + 3 * Polynomial.X + 2
  let expanded : Polynomial ℤ := p^5
  expanded.coeff 1 = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l701_70154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_angles_equal_l701_70125

open Real

-- Define a circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define a point on the circle
def onCircle (c : Circle) (p : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖p - c.center‖ = c.radius

-- Define an arc on the circle
def Arc (c : Circle) (a b : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | onCircle c p ∧ (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b)}

-- Define an inscribed angle
def InscribedAngle (c : Circle) (a b v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  onCircle c a ∧ onCircle c b ∧ onCircle c v ∧ v ∉ Arc c a b

-- Define angle measure
noncomputable def angleMeasure (a b c : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem inscribed_angles_equal 
  (c : Circle)
  (a b : EuclideanSpace ℝ (Fin 2))
  (v₁ v₂ : EuclideanSpace ℝ (Fin 2))
  (h₁ : InscribedAngle c a b v₁)
  (h₂ : InscribedAngle c a b v₂) : 
  angleMeasure a v₁ b = angleMeasure a v₂ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_angles_equal_l701_70125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_fixed_point_no_extreme_values_l701_70101

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * Real.exp x

-- Theorem for the tangent line passing through a fixed point
theorem tangent_line_fixed_point (a : ℝ) :
  ∃ (m b : ℝ), ∀ x, m * x + b = f a x + (deriv (f a)) 0 * (x - 0) - f a 0 ∧ 
  m * (-1) + b = 0 := by sorry

-- Theorem for no extreme values on (0, +∞) for specified range of a
theorem no_extreme_values (a : ℝ) (h : a ≤ 0 ∨ a ≥ 24 / Real.exp 2) :
  ∀ x > 0, deriv (f a) x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_fixed_point_no_extreme_values_l701_70101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l701_70142

theorem angle_properties (α : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 ∧ (2*Real.sin α + 3*Real.cos α)/(Real.cos α - Real.sin α) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l701_70142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l701_70166

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 + Real.log x / Real.log 2

-- State the theorem
theorem zero_point_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo 1 2 ∧ f c = 0 := by
  sorry

-- Additional helper lemmas to support the main theorem
lemma f_continuous : Continuous f := by
  sorry

lemma f_increasing : StrictMono f := by
  sorry

lemma f_neg_at_one : f 1 < 0 := by
  sorry

lemma f_pos_at_two : f 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l701_70166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l701_70131

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem function_properties :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≥ (1/2) ∧ f x ≤ ((1/2) * (Real.exp 1)^2 + 1)) ∧
  (∀ x ≥ 1, f x < g x) ∧
  (∀ n : ℕ, n ≥ 1 → ∀ x > 0, (f' x)^n - f' (x^n) ≥ 2^n - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l701_70131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l701_70126

theorem complex_equality (a : ℝ) : 
  (Complex.re (-3 * Complex.I * (a + Complex.I)) = Complex.im (-3 * Complex.I * (a + Complex.I))) → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l701_70126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_l701_70186

/-- Given points A, B, and Q on a line where AQ:QB = 7:2, 
    prove that Q = -2/5 * A + 7/5 * B -/
theorem point_division (A B Q : ℝ × ℝ) : 
  (‖Q - A‖ / ‖B - Q‖ = 7 / 2) → 
  (Q = (-2/5 : ℝ) • A + (7/5 : ℝ) • B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_l701_70186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_three_fifths_l701_70156

/-- Given a point P(-4, 3) on the terminal side of angle α, prove that sin α = 3/5 -/
theorem sin_alpha_equals_three_fifths (α : ℝ) (P : ℝ × ℝ) :
  P = (-4, 3) →  -- P lies on the terminal side of angle α
  Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_three_fifths_l701_70156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_odd_iff_perfect_square_l701_70132

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := (Finset.filter (· ∣ n.val) (Finset.range n.val)).card + 1

/-- A positive integer is a perfect square -/
def is_perfect_square (n : ℕ+) : Prop := ∃ k : ℕ+, n = k * k

theorem divisors_odd_iff_perfect_square (n : ℕ+) :
  Odd (num_divisors n) ↔ is_perfect_square n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_odd_iff_perfect_square_l701_70132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l701_70145

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a line perpendicular to the x-axis passes through the right focus F₂
    and intersects the hyperbola at point P such that ∠PF₁F₂ = π/6,
    then the eccentricity of the hyperbola is √3. --/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  ∃ (c : ℝ), c > a ∧
  (∀ x y, f x y = 1 → ∃ P : ℝ × ℝ, P.1 = x ∧ P.2 = y) →
  (∃ F₁ F₂ P : ℝ × ℝ, F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0 ∧
    P.1 = c ∧ f P.1 P.2 = 1 ∧ 
    (P.2 - F₁.2) / (P.1 - F₁.1) = Real.tan (π/6)) →
  c / a = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l701_70145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_always_positive_quadratic_l701_70192

theorem range_of_a_for_always_positive_quadratic :
  let S : Set ℝ := {a : ℝ | ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0}
  S = Set.Icc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_always_positive_quadratic_l701_70192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_polygons_max_sides_cube_intersection_l701_70153

/-- A regular polygon that can be formed by intersecting a cube with a plane. -/
inductive CubePlaneIntersectionPolygon
  | EquilateralTriangle
  | Square
  | RegularPentagon

/-- The set of all possible regular polygons that can be formed by intersecting a cube with a plane. -/
def PossibleIntersectionPolygons : Set CubePlaneIntersectionPolygon :=
  {CubePlaneIntersectionPolygon.EquilateralTriangle,
   CubePlaneIntersectionPolygon.Square,
   CubePlaneIntersectionPolygon.RegularPentagon}

/-- The number of faces of a cube. -/
def cube_faces : Nat := 6

/-- Theorem stating that the only regular polygons that can be formed by intersecting a cube with a plane
    are equilateral triangles, squares, and regular pentagons. -/
theorem cube_plane_intersection_polygons :
  ∀ (p : CubePlaneIntersectionPolygon),
    p ∈ PossibleIntersectionPolygons ↔
      (p = CubePlaneIntersectionPolygon.EquilateralTriangle ∨
       p = CubePlaneIntersectionPolygon.Square ∨
       p = CubePlaneIntersectionPolygon.RegularPentagon) :=
by sorry

/-- The maximum number of sides a polygon formed by intersecting a cube with a plane can have
    is equal to the number of faces of the cube. -/
theorem max_sides_cube_intersection (p : CubePlaneIntersectionPolygon) :
  (∃ (n : Nat), n ≤ cube_faces ∧ n = match p with
    | CubePlaneIntersectionPolygon.EquilateralTriangle => 3
    | CubePlaneIntersectionPolygon.Square => 4
    | CubePlaneIntersectionPolygon.RegularPentagon => 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_polygons_max_sides_cube_intersection_l701_70153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l701_70183

def b : ℕ → ℚ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 1/3
  | n+3 => (2 - b (n+2)) / (3 * b (n+1) + 1)

theorem b_100_value : b 100 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_value_l701_70183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_less_than_1_1_is_9_10_l701_70106

def given_numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]

def is_less_than_1_1 (x : ℚ) : Bool := x < 1.1

def largest_less_than_1_1 (numbers : List ℚ) : ℚ :=
  (numbers.filter is_less_than_1_1).maximum?
    |>.getD 0  -- Default to 0 if the list is empty

theorem largest_less_than_1_1_is_9_10 :
  largest_less_than_1_1 given_numbers = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_less_than_1_1_is_9_10_l701_70106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_arctangent_identity_l701_70141

/-- The golden ratio -/
noncomputable def r : ℝ := (1 + Real.sqrt 5) / 2

/-- The main theorem -/
theorem golden_ratio_arctangent_identity :
  7 * (Real.arctan r)^2 + 2 * (Real.arctan (r^3))^2 - (Real.arctan (r^5))^2 = 7 * Real.pi^2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_arctangent_identity_l701_70141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l701_70113

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sqrt (1 + x)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Ici (-2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l701_70113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_translated_sine_l701_70112

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 4)

noncomputable def g (x : ℝ) := Real.sin (2 * x + 3 * Real.pi / 4) + 2

def is_symmetry_center (h : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, h (c.fst + x) = h (c.fst - x)

theorem symmetry_center_of_translated_sine :
  g = (λ x ↦ f (x + Real.pi / 4) + 2) →
  is_symmetry_center g (Real.pi / 8, 2) := by
  sorry

#check symmetry_center_of_translated_sine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_translated_sine_l701_70112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_l701_70120

/-- Given a circle and two fixed points A and B on it, prove that as a third point C moves
    along the circle, the locus of the centroid of triangle ABC is another circle -/
theorem centroid_locus (O : ℝ × ℝ) (r : ℝ) (A B : ℝ × ℝ) :
  let circle := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  A ∈ circle → B ∈ circle →
  ∃ (r' : ℝ), r' = r / 3 ∧
    {M : ℝ × ℝ | ∃ C ∈ circle, M = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)} =
    {P : ℝ × ℝ | (P.1 - midpoint.1)^2 + (P.2 - midpoint.2)^2 = r'^2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_locus_l701_70120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_fours_l701_70143

theorem four_fours (n : Nat) (h : n ≥ 1 ∧ n ≤ 22) :
  ∃ (a b c d e : Nat), a = 4 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4 ∧
  ∃ (f : Nat → Nat → Nat → Nat → Nat → Nat), f a b c d e = n :=
sorry

#check four_fours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_fours_l701_70143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cone_l701_70105

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Represents the water level in the cone -/
structure WaterLevel where
  height : ℝ

theorem water_height_in_cone (tank : Cone) (water : WaterLevel) :
  tank.radius = 12 →
  tank.height = 48 →
  coneVolume { radius := water.height * tank.radius / tank.height, height := water.height } = 0.4 * coneVolume tank →
  water.height = 24 * (384 : ℝ) ^ (1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cone_l701_70105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_and_inequality_l701_70189

theorem min_sum_squares_and_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (∃ M : ℝ, M = 3 ∧ 
    (∀ a' b' c' : ℝ, a' + b' + c' = 3 → a'^2 + b'^2 + c'^2 ≥ M) ∧ 
    a^2 + b^2 + c^2 = M) ∧
  (∀ x : ℝ, |x + 4| - |x - 1| ≥ 3 ↔ x ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_and_inequality_l701_70189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intercept_and_distance_l701_70174

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem line_equation_from_intercept_and_distance :
  ∀ l : Line,
    y_intercept l = 10 →
    distance_point_to_line 0 0 l = 8 →
    (l.a = 3 ∧ l.b = -4 ∧ l.c = -40) ∨ (l.a = 3 ∧ l.b = 4 ∧ l.c = -40) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intercept_and_distance_l701_70174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_five_l701_70146

def is_multiple_of_five (n : Nat) : Bool := n % 5 = 0

def count_multiples_of_five (upper_bound : Nat) : Nat :=
  (Finset.range upper_bound).filter (fun n => is_multiple_of_five n) |>.card

theorem probability_multiple_of_five :
  (count_multiples_of_five 30 : ℚ) / 30 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_five_l701_70146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_inequality_l701_70187

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- For any natural number, if the sum of its digits equals the sum of the digits of twice the number plus one, then the sum of the digits of thrice the number minus three cannot equal the sum of the digits of the number minus two. -/
theorem digit_sum_inequality (n : ℕ) :
  (digit_sum n = digit_sum (2*n + 1)) →
  (digit_sum (3*n - 3) ≠ digit_sum (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_inequality_l701_70187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l701_70139

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Ioc (1/2) 1 then (7*x - 3) / (2*x + 2)
  else if x ∈ Set.Icc 0 (1/2) then -1/3*x + 1/6
  else 0  -- undefined for other x

noncomputable def g (a x : ℝ) : ℝ := a * Real.sin (Real.pi/6 * x) - 2*a + 2

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f x₁ = g a x₂) →
  a ∈ Set.Icc (1/2) (4/3) :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l701_70139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_condition_l701_70103

/-- A function f(x) = 1 / (kx^2 + 2kx + 3) with domain R -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 1 / (k * x^2 + 2 * k * x + 3)

/-- The theorem stating the range of k for which f has domain R -/
theorem f_domain_condition (k : ℝ) : 
  (∀ x, ∃ y, f k x = y) ↔ 0 ≤ k ∧ k < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_condition_l701_70103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l701_70169

noncomputable section

-- Define the isosceles triangle
def isosceles_triangle (base leg : ℝ) : Prop :=
  base > 0 ∧ leg > 0

-- Define the area of the triangle
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1/2) * base * height

-- Define the semi-perimeter
noncomputable def semi_perimeter (base leg : ℝ) : ℝ :=
  (2 * leg + base) / 2

-- Define the radius of the inscribed circle
noncomputable def inscribed_circle_radius (base leg : ℝ) : ℝ :=
  let s := semi_perimeter base leg
  let area := triangle_area base (Real.sqrt (leg^2 - (base/2)^2))
  area / s

-- Theorem statement
theorem isosceles_triangle_properties (base leg : ℝ) 
  (h : isosceles_triangle base leg) 
  (hbase : base = 6) 
  (hleg : leg = 5) : 
  triangle_area base (Real.sqrt (leg^2 - (base/2)^2)) = 12 ∧ 
  inscribed_circle_radius base leg = 1.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l701_70169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_function_l701_70117

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 4)

-- State the theorem
theorem symmetry_axis_of_sine_function :
  ∃ (k : ℤ), ∀ (x : ℝ), f (π / 8 + x) = f (π / 8 - x) :=
by
  -- We'll use k = 0 to prove the existence
  use 0
  intro x
  -- Expand the definition of f
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_function_l701_70117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_and_eight_digit_number_properties_l701_70115

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

def is_valid_seven_digit_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧
  ∀ d, d ∈ (digits n) → d ≠ 0 ∧ n % d = 0 ∧
  (∀ i j, i ≠ j → (digits n).nthLe i (by sorry) ≠ (digits n).nthLe j (by sorry))

def no_valid_eight_digit_number : Prop :=
  ∀ n, 10000000 ≤ n ∧ n < 100000000 →
    ¬(∀ d, d ∈ (digits n) → d ≠ 0 ∧ n % d = 0 ∧
      (∀ i j, i ≠ j → (digits n).nthLe i (by sorry) ≠ (digits n).nthLe j (by sorry)))

theorem seven_and_eight_digit_number_properties :
  is_valid_seven_digit_number 7639128 ∧ no_valid_eight_digit_number :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_and_eight_digit_number_properties_l701_70115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_range_l701_70149

open Real

/-- Definition of a tangent line to a function at a point -/
def IsTangentLine (l : ℝ → ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  HasDerivAt f (l x₀ - f x₀) x₀ ∧ l x₀ = f x₀

/-- The range of a for which there exists a common tangent line to y = ln x and y = x^a -/
theorem common_tangent_range (x : ℝ) (a : ℝ) (h_x : x > 0) (h_a : a ≠ 0) :
  (∃ (l : ℝ → ℝ), IsTangentLine l (fun x => log x) x ∧ IsTangentLine l (fun x => x^a) x) →
  a ∈ Set.Ioo 0 (1/Real.exp 1) ∪ Set.Ioi 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_range_l701_70149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_theorem_l701_70190

-- Define the types for lines and planes in space
structure Line3D where
  -- Add necessary fields
  dummy : Unit

structure Plane3D where
  -- Add necessary fields
  dummy : Unit

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

-- Define the theorem
theorem spatial_relationships_theorem 
  (m n : Line3D) (α β : Plane3D) 
  (h_diff_lines : m ≠ n) (h_diff_planes : α ≠ β) :
  (∃ (m' : Line3D) (α' β' : Plane3D),
    parallel_line_plane m' α' ∧ parallel_line_plane m' β' ∧ ¬(parallel_planes α' β')) ∧
  (∃ (m' n' : Line3D) (α' : Plane3D),
    parallel_line_plane m' α' ∧ parallel_lines m' n' ∧ ¬(parallel_line_plane n' α')) ∧
  (∀ (m' : Line3D) (α' β' : Plane3D),
    perpendicular_line_plane m' α' → parallel_line_plane m' β' → perpendicular_planes α' β') ∧
  (∀ (m' : Line3D) (α' β' : Plane3D),
    perpendicular_line_plane m' α' → parallel_planes α' β' → perpendicular_line_plane m' β') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_theorem_l701_70190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_299_is_monday_l701_70110

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Function to advance a day by one -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by a number of days -/
def advanceDay (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem day_299_is_monday (day15 : DayOfWeek) (h : day15 = DayOfWeek.Wednesday) :
  advanceDay day15 284 = DayOfWeek.Monday :=
by sorry

#eval advanceDay DayOfWeek.Wednesday 284

end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_299_is_monday_l701_70110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l701_70107

-- Define the point A
noncomputable def A : ℝ × ℝ := (Real.sqrt 3, 1)

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a line passing through point A with slope k
def is_on_line (k : ℝ) (x y : ℝ) : Prop := y - A.2 = k * (x - A.1)

-- Define the condition that the line intersects the circle
def intersects (k : ℝ) : Prop := ∃ x y, is_on_circle x y ∧ is_on_line k x y

-- Define the slope angle in terms of k
noncomputable def slope_angle (k : ℝ) : ℝ := Real.arctan k

-- Theorem statement
theorem slope_angle_range :
  ∀ k, intersects k → 0 ≤ slope_angle k ∧ slope_angle k ≤ Real.pi / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l701_70107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_cost_l701_70173

/-- Calculates the cost of a rice mixture given the costs of two varieties and their mixing ratio -/
noncomputable def mixtureCost (cost1 cost2 ratio : ℝ) : ℝ :=
  (cost1 * ratio + cost2) / (1 + ratio)

/-- Proves that the mixture of rice varieties costing 5 and 8.75 per kg, mixed in a 0.5 ratio, costs 7.5 per kg -/
theorem rice_mixture_cost :
  mixtureCost 5 8.75 0.5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_cost_l701_70173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l701_70133

-- Define the radii of the two initial semicircles
variable (r R : ℝ)
-- Assume r and R are positive
variable (hr : r > 0)
variable (hR : R > 0)

-- Define the radius of the circle touching all three semicircles
noncomputable def x (r R : ℝ) : ℝ := (R * r * (R + r)) / (R^2 + R*r + r^2)

-- State the theorem
theorem circle_radius_theorem (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  ∃ (x : ℝ), x > 0 ∧ x = (R * r * (R + r)) / (R^2 + R*r + r^2) :=
by
  -- Use the previously defined x function
  use x r R
  constructor
  · sorry -- Proof that x > 0
  · -- Proof that x equals the given expression
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l701_70133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l701_70111

/-- A path in the Cartesian plane -/
inductive CartesianPath
  | right : CartesianPath → CartesianPath
  | up : CartesianPath → CartesianPath
  | diagonal : CartesianPath → CartesianPath
  | end : CartesianPath

/-- Checks if a path is valid (no right-angle turns) -/
def isValidPath : CartesianPath → Bool
  | CartesianPath.right (CartesianPath.right _) => false
  | CartesianPath.up (CartesianPath.up _) => false
  | _ => true

/-- Checks if a path ends at (4,4) -/
def endsAt44 : CartesianPath → Bool
  | CartesianPath.end => true
  | _ => false

/-- Counts the number of valid paths from (0,0) to (4,4) -/
def countValidPaths : ℕ := 37  -- Simplified for demonstration

/-- The main theorem stating that there are 37 valid paths -/
theorem valid_paths_count : countValidPaths = 37 := by
  rfl  -- reflexivity proves the equality


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l701_70111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_l701_70128

theorem smallest_d (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : (a + b + c) / 3 = ((a + b + c + d) / 4) / 2) :
  d ≥ 10 ∧ ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    a' < b' ∧ b' < c' ∧ c' < 10 ∧
    (a' + b' + c') / 3 = ((a' + b' + c' + 10) / 4) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_l701_70128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l701_70100

def sequence_sum (a : ℕ → ℕ) : Prop :=
  (a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2) ∧
  (∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1) ∧
  (∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)) ∧
  (Finset.sum (Finset.range 100) (fun i => a (i + 1)) = 200)

theorem sequence_sum_theorem :
  ∃ a : ℕ → ℕ, sequence_sum a := by
  sorry

#check sequence_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l701_70100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_properties_l701_70177

noncomputable section

-- Define the ellipse C₁
def C₁ (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the dot product of vectors
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define perpendicularity of vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_parabola_properties
  (a b : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : eccentricity a b = Real.sqrt 3 / 3)
  (h₄ : dot_product (-a) b a b = -1) :
  (∃ x y, C₁ x y 3 2) ∧
  (∃ x y, C₂ x y) ∧
  (∀ x₀ y₀ x₁ x₂ y₂,
    C₂ x₀ y₀ → C₂ x₁ 2 → C₂ x₂ y₂ →
    x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ →
    perpendicular (x₂ - x₁) (y₂ - 2) (x₀ - x₂) (y₀ - y₂) →
    (y₀ < -6 ∨ y₀ ≥ 10)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_properties_l701_70177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_76_l701_70191

/-- A square with an additional point forming a right triangle --/
structure SquareWithTriangle where
  -- The side length of the square
  side : ℝ
  -- The lengths of the legs of the right triangle
  leg1 : ℝ
  leg2 : ℝ
  -- Ensure the legs form a right triangle within the square
  leg1_le_side : leg1 ≤ side
  leg2_le_side : leg2 ≤ side

/-- The area of the pentagon formed by the square and the right triangle --/
noncomputable def pentagonArea (s : SquareWithTriangle) : ℝ :=
  s.side * s.side - (s.leg1 * s.leg2) / 2

/-- Theorem stating that for the given dimensions, the pentagon area is 76 --/
theorem pentagon_area_is_76 :
  ∃ s : SquareWithTriangle, s.leg1 = 8 ∧ s.leg2 = 6 ∧ pentagonArea s = 76 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_76_l701_70191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_three_nines_delta_n_nines_l701_70108

/-- Definition of the delta operation -/
def delta (a b : ℕ) : ℕ := a * b + a + b

/-- Theorem for the first part of the problem -/
theorem delta_three_nines :
  (delta (delta (delta 1 9) 9) 9) = 1999 := by sorry

/-- Theorem for the second part of the problem -/
theorem delta_n_nines (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, k = 1 * 10^n + 99 * (10^n - 1) / 9 ∧
  (List.foldl delta 1 (List.replicate n 9)) = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_three_nines_delta_n_nines_l701_70108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_probability_probability_is_64_729_l701_70155

/-- A regular cube with 8 vertices -/
structure Cube where
  vertices : Fin 8

/-- An ant on a vertex of the cube -/
structure Ant where
  position : Fin 8

/-- The probability of an ant moving to an adjacent vertex -/
def move_probability : ℚ := 1 / 3

/-- The number of ants -/
def num_ants : ℕ := 8

/-- The number of valid permutations for ants to move without collision -/
def valid_permutations : ℕ := 24 * 24  -- 4! * 4!

/-- Function to get adjacent vertices -/
def adjacent_vertices (v : Fin 8) : List (Fin 8) := sorry

theorem ant_movement_probability (cube : Cube) (ants : Fin num_ants → Ant) :
  (∀ i j, i ≠ j → (ants i).position ≠ (ants j).position) →
  (∀ i, ∃ j, (ants i).position = j ∧ j ∈ adjacent_vertices (ants i).position) →
  (probability_no_collision : ℚ) =
    (valid_permutations : ℚ) * move_probability ^ num_ants :=
by sorry

theorem probability_is_64_729 (cube : Cube) (ants : Fin num_ants → Ant) :
  (∀ i j, i ≠ j → (ants i).position ≠ (ants j).position) →
  (∀ i, ∃ j, (ants i).position = j ∧ j ∈ adjacent_vertices (ants i).position) →
  (probability_no_collision : ℚ) = 64 / 729 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_probability_probability_is_64_729_l701_70155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l701_70123

noncomputable def g (x : ℝ) : ℝ := x + x / (x^2 + 1) + x * (x + 3) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l701_70123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_values_l701_70182

theorem A_values (k : ℤ) (α : ℝ) (h1 : Real.sin α ≠ 0) (h2 : Real.cos α ≠ 0) :
  (Real.sin (k * Real.pi + α) / Real.sin α + Real.cos (k * Real.pi + α) / Real.cos α) = 2 ∨
  (Real.sin (k * Real.pi + α) / Real.sin α + Real.cos (k * Real.pi + α) / Real.cos α) = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_values_l701_70182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l701_70122

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 1) / Real.log 10

-- State the theorem about the range of f(x)
theorem range_of_f :
  Set.range f = Set.Iic 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l701_70122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l701_70144

theorem election_votes (total_votes : ℝ) : 
  (0.7 * total_votes - 0.3 * total_votes = 200) → 
  total_votes = 500 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l701_70144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_problem_l701_70172

theorem arithmetic_geometric_mean_problem (x y : ℤ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  ∃ (a b : ℤ), 
    (x + y) / 2 = 10 * a + b ∧
    10 * a + b ≥ 10 ∧
    10 * a + b < 100 ∧
    (10 * a + b) % 3 = 0 ∧
    Real.sqrt (x * y : ℝ) = 10 * b + a + 2 →
  |x - y| = 48 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_problem_l701_70172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solutions_l701_70129

theorem exponential_equation_solutions :
  ∀ a b : ℕ, a > 0 → b > 0 → (a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solutions_l701_70129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_theorem_l701_70121

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on an ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Checks if a line is tangent to a circle -/
def Line.tangentToCircle (l : Line) : Prop :=
  l.intercept^2 = l.slope^2 + 1

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  1/2 * base * height

theorem ellipse_and_triangle_area_theorem (e : Ellipse) (E : Point) (l : Line) :
  e.contains E ∧
  E.x = Real.sqrt 3 ∧
  E.y = 1/2 ∧
  e.eccentricity = Real.sqrt 3 / 2 ∧
  l.tangentToCircle →
  (∃ A B : Point,
    e.contains A ∧
    e.contains B ∧
    A ≠ B ∧
    (∀ p : Point, e.contains p → p.x^2 / 4 + p.y^2 = 1) ∧
    (∀ base height, triangleArea base height ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_theorem_l701_70121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consec_win_prob_highest_second_l701_70194

/-- Represents the probability of winning against a player -/
@[ext] structure WinProb where
  p : ℝ
  pos : 0 < p

/-- Represents the probability of winning two consecutive games -/
def consec_win_prob (p₁ p₂ p₃ : WinProb) : WinProb → WinProb → ℝ
| x, y => 2 * (x.p * (y.p + p₃.p) - 2 * x.p * y.p * p₃.p)

/-- The main theorem -/
theorem max_consec_win_prob_highest_second 
  (p₁ p₂ p₃ : WinProb) 
  (h₁ : p₁.p < p₂.p) (h₂ : p₂.p < p₃.p) :
  consec_win_prob p₁ p₂ p₃ p₁ p₃ > consec_win_prob p₁ p₂ p₃ p₁ p₂ ∧ 
  consec_win_prob p₁ p₂ p₃ p₁ p₃ > consec_win_prob p₁ p₂ p₃ p₂ p₃ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consec_win_prob_highest_second_l701_70194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_not_determined_by_A_l701_70180

-- Define a polynomial type
def MyPolynomial := ℕ → ℝ

-- Define a characteristic function for polynomials
def A : MyPolynomial → Set ℝ := sorry

-- Define a degree function for polynomials
def degree : MyPolynomial → ℕ := sorry

-- Theorem statement
theorem degree_not_determined_by_A : 
  ∃ (P1 P2 : MyPolynomial), A P1 = A P2 ∧ degree P1 ≠ degree P2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_not_determined_by_A_l701_70180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clicks_in_25_seconds_approx_speed_l701_70140

/-- Represents the length of a short rail segment in feet -/
noncomputable def short_segment : ℝ := 20

/-- Represents the length of a long rail segment in feet -/
noncomputable def long_segment : ℝ := 50

/-- Represents the total length of one full rail segment (short + long) in feet -/
noncomputable def full_segment : ℝ := short_segment + long_segment

/-- Converts miles per hour to feet per minute -/
noncomputable def mph_to_fpm (speed : ℝ) : ℝ := speed * 5280 / 60

/-- Calculates the number of clicks per minute for a given speed in mph -/
noncomputable def clicks_per_minute (speed : ℝ) : ℝ := mph_to_fpm speed / full_segment

/-- Theorem: The number of clicks heard in 25 seconds approximately equals the train's speed in mph -/
theorem clicks_in_25_seconds_approx_speed (speed : ℝ) (h : speed > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |clicks_per_minute speed * (25 / 60) - speed| < ε := by
  sorry

#check clicks_in_25_seconds_approx_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clicks_in_25_seconds_approx_speed_l701_70140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_difference_l701_70127

theorem circle_triangle_area_difference :
  let r : ℝ := 3
  let s : ℝ := 4
  let circle_area := π * r^2
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  let difference := circle_area - triangle_area
  difference = 9 * π - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_difference_l701_70127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l701_70178

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors m and n
noncomputable def m (B : Real) : Real × Real := (2 * Real.sin B, 2 - Real.cos (2 * B))
noncomputable def n (B : Real) : Real × Real := (2 * (Real.sin ((B / 2) + (Real.pi / 4)))^2, -1)

-- Define the perpendicularity condition
def perpendicular (v w : Real × Real) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem triangle_problem (abc : Triangle) :
  abc.a = Real.sqrt 3 →
  abc.b = 1 →
  perpendicular (m abc.B) (n abc.B) →
  (abc.B = Real.pi / 6 ∨ abc.B = 5 * Real.pi / 6) ∧
  (abc.c = 1 ∨ abc.c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l701_70178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_discounts_rounded_l701_70160

-- Define the final price after discounts
def final_price : ℝ := 174.99999999999997

-- Define the function to round to the nearest cent
noncomputable def round_to_cent (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

-- Theorem statement
theorem price_after_discounts_rounded : 
  round_to_cent final_price = 175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_discounts_rounded_l701_70160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_on_ellipse_l701_70193

def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 16 = 1

def is_real_fraction (z : ℂ) : Prop :=
  ∃ (r : ℝ), (z - 1 - Complex.I) / (z - Complex.I) = r

theorem complex_number_on_ellipse (z : ℂ) :
  let x := z.re
  let y := z.im
  is_on_ellipse x y ∧ is_real_fraction z →
  z = Complex.mk ((3 * Real.sqrt 15) / 4) 1 ∨
  z = Complex.mk (-(3 * Real.sqrt 15) / 4) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_on_ellipse_l701_70193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_rectangle_area_l701_70104

/-- Regular octagon with side length 8 cm -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 8)

/-- Rectangle formed by connecting midpoints of alternate sides of the octagon -/
def midpoint_rectangle (octagon : RegularOctagon) : ℝ → ℝ → Prop :=
  λ length width ↦ length = octagon.side_length * Real.sqrt 2 ∧ width = octagon.side_length * Real.sqrt 2

/-- The area of the rectangle formed by connecting midpoints of alternate sides
    of a regular octagon with side length 8 cm is 128 square cm -/
theorem midpoint_rectangle_area (octagon : RegularOctagon) :
  ∀ length width, midpoint_rectangle octagon length width →
  length * width = 128 := by
  sorry

#check midpoint_rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_rectangle_area_l701_70104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_theorem_l701_70162

-- Define the weight of the largest frog
def largest_frog_weight : ℚ := 120

-- Define the ratio of weights between the largest and smallest frog
def weight_ratio : ℚ := 10

-- Define the weight of the smallest frog
def smallest_frog_weight : ℚ := largest_frog_weight / weight_ratio

-- Theorem to prove
theorem weight_difference_theorem :
  largest_frog_weight - smallest_frog_weight = 108 := by
  -- Unfold the definitions
  unfold largest_frog_weight smallest_frog_weight weight_ratio
  -- Simplify the arithmetic
  simp [Rat.div_def]
  -- The proof is complete
  rfl

#eval largest_frog_weight - smallest_frog_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_theorem_l701_70162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_l701_70188

theorem square_difference (n : ℕ) : 
  (n + 1)^2 - n^2 = 101 → n^2 - (n - 1)^2 = 196 := by
  intro h
  have n_eq : n = 50 := by
    -- Proof that n = 50 would go here
    sorry
  rw [n_eq]
  -- Rest of the proof would go here
  sorry

#eval (51^2 - 50^2)
#eval (50^2 - 49^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_l701_70188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_razorback_shop_jersey_revenue_l701_70150

/-- Represents the revenue model for the Razorback shop -/
structure RevenueModel where
  revenue_per_tshirt : ℚ
  tshirts_sold : ℕ
  total_tshirt_revenue : ℚ
  jerseys_sold : ℕ

/-- Calculates the revenue per jersey given the revenue model -/
def revenue_per_jersey (model : RevenueModel) : ℚ :=
  (model.total_tshirt_revenue - model.revenue_per_tshirt * model.tshirts_sold) / model.jerseys_sold

/-- Theorem stating that the revenue per jersey is zero for the given scenario -/
theorem razorback_shop_jersey_revenue :
  let model : RevenueModel := {
    revenue_per_tshirt := 215,
    tshirts_sold := 20,
    total_tshirt_revenue := 4300,
    jerseys_sold := 64
  }
  revenue_per_jersey model = 0 := by
  sorry

#eval revenue_per_jersey {
  revenue_per_tshirt := 215,
  tshirts_sold := 20,
  total_tshirt_revenue := 4300,
  jerseys_sold := 64
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_razorback_shop_jersey_revenue_l701_70150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70_to_79_approx_24_percent_l701_70158

/-- Represents the frequency distribution of test scores -/
structure TestScoreDistribution where
  score_90_100 : Nat
  score_80_89 : Nat
  score_70_79 : Nat
  score_60_69 : Nat
  score_50_59 : Nat
  score_below_50 : Nat

/-- Calculates the total number of students -/
def totalStudents (dist : TestScoreDistribution) : Nat :=
  dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + 
  dist.score_60_69 + dist.score_50_59 + dist.score_below_50

/-- Calculates the percentage of students in a given range -/
noncomputable def percentageInRange (studentsInRange : Nat) (total : Nat) : Real :=
  (studentsInRange : Real) / (total : Real) * 100

/-- Rounds a real number to the nearest whole number -/
noncomputable def roundToNearest (x : Real) : Int :=
  Int.floor (x + 0.5)

/-- Theorem: The percentage of students scoring in the 70% - 79% range is approximately 24% -/
theorem percentage_70_to_79_approx_24_percent (dist : TestScoreDistribution) 
    (h : dist = { score_90_100 := 5, score_80_89 := 10, score_70_79 := 8, 
                  score_60_69 := 4, score_50_59 := 3, score_below_50 := 4 }) : 
    roundToNearest (percentageInRange dist.score_70_79 (totalStudents dist)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70_to_79_approx_24_percent_l701_70158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_variables_l701_70118

/-- The area of a circle -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_variables :
  ∃ (S r : ℝ), S = circle_area r ∧
  (∀ (x : ℝ), x ≠ S → x ≠ r → ¬∃ (y : ℝ), circle_area y = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_variables_l701_70118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_february_starts_friday_l701_70171

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Inhabited, BEq

/-- Represents a day in February -/
structure FebruaryDay where
  day : Nat
  weekday : Weekday
deriving Inhabited

/-- Definition of a February with the given conditions -/
structure SpecialFebruary where
  days : List FebruaryDay
  h1 : days.length ≥ 28
  h2 : days.length ≤ 29
  h3 : ∀ i j, i < j → (days.get! i).day < (days.get! j).day
  h4 : (days.filter (λ d => d.weekday == Weekday.Monday)).length = 3
  h5 : (days.filter (λ d => d.weekday == Weekday.Friday)).length = 5

/-- Theorem: In a SpecialFebruary, the first day is a Friday -/
theorem special_february_starts_friday (feb : SpecialFebruary) : 
  (feb.days.get! 0).weekday = Weekday.Friday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_february_starts_friday_l701_70171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_l701_70181

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem monotonicity_of_f :
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x ≤ y → f x ≥ f y) ∧
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y) ∧
  (∀ x y, x ∈ Set.Iic (-1) → y ∈ Set.Iic (-1) → x ≤ y → f x ≤ f y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_l701_70181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_distance_l701_70157

/-- The curve function -/
noncomputable def curve (k : ℝ) (x : ℝ) : ℝ := x^2 + k * Real.log x

/-- The derivative of the curve function -/
noncomputable def curve_derivative (k : ℝ) (x : ℝ) : ℝ := 2*x + k/x

/-- The tangent line function -/
def tangent_line (k : ℝ) (x : ℝ) : ℝ := (2 + k) * x - 1 - k

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (k : ℝ) : ℝ :=
  abs ((-1) * (2 + k) + 1 + k) / Real.sqrt ((2 + k)^2 + 1^2)

theorem curve_tangent_distance (k : ℝ) :
  curve k 1 = tangent_line k 1 ∧
  curve_derivative k 1 = 2 + k ∧
  distance_point_to_line k = 1 →
  k = -5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_distance_l701_70157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_f_is_even_g_is_even_h_is_even_k_not_even_l701_70196

-- Define the functions
def f (x : ℝ) : ℝ := -abs x + 2
def g (x : ℝ) : ℝ := x^2 - 3
noncomputable def h (x : ℝ) : ℝ := Real.sqrt (1 - x^2)
noncomputable def k (x : ℝ) : ℝ := Real.sqrt x

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem stating which functions are even
theorem even_functions :
  (is_even f) ∧ (is_even g) ∧ (is_even h) ∧ (¬ is_even k) := by
  sorry

-- Individual theorems for each function
theorem f_is_even : is_even f := by
  sorry

theorem g_is_even : is_even g := by
  sorry

theorem h_is_even : is_even h := by
  sorry

theorem k_not_even : ¬ is_even k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_f_is_even_g_is_even_h_is_even_k_not_even_l701_70196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_m_is_21_l701_70168

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the set S
def S : Set ℕ := {n : ℕ | sumOfDigits n = 15 ∧ n < 10^8}

-- Define m as the number of elements in S
noncomputable def m : ℕ := Finset.card (Finset.filter (λ n => sumOfDigits n = 15) (Finset.range (10^8)))

-- Theorem statement
theorem sum_of_digits_of_m_is_21 : sumOfDigits m = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_m_is_21_l701_70168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_in_circle_intersect_l701_70114

-- Define a circle with radius 1
def unit_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

-- Define a triangle as a set of three points in ℝ²
def Triangle : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Function to check if a triangle is inside the unit circle
def inside_circle (t : Triangle) : Prop :=
  ∀ p, p ∈ [t.1, t.2.1, t.2.2] → p ∈ unit_circle

-- Function to calculate the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let (a, b, c) := t
  abs ((a.1 - c.1) * (b.2 - a.2) - (a.1 - b.1) * (c.2 - a.2)) / 2

-- Function to check if two triangles intersect
def triangles_intersect (t1 t2 : Triangle) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ [t1.1, t1.2.1, t1.2.2] ∨ p ∈ [t2.1, t2.2.1, t2.2.2]

-- Theorem statement
theorem triangles_in_circle_intersect (t1 t2 : Triangle) 
  (h1 : inside_circle t1) (h2 : inside_circle t2)
  (h3 : triangle_area t1 > 1) (h4 : triangle_area t2 > 1) :
  triangles_intersect t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_in_circle_intersect_l701_70114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_age_ratio_l701_70135

-- Define Tom's current age and the number of years ago
variable (T N : ℝ)

-- Define the condition that Tom's age is the sum of his four children's ages
axiom children_sum : T = T

-- Define the condition that N years ago, Tom's age was three times the sum of his children's ages
axiom past_age_relation : T - N = 3 * (T - 4 * N)

-- Theorem to prove
theorem toms_age_ratio : T / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_age_ratio_l701_70135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_two_l701_70137

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 3*x - 1 else -(3*(-x) - 1)

-- State the theorem
theorem f_neg_one_eq_neg_two :
  (∀ x, f (-x) = -f x) →  -- f is odd
  f (-1) = -2 := by
  intro h
  have h1 : f 1 = 2 := by
    simp [f]
    norm_num
  have h2 : f (-1) = -f 1 := h 1
  rw [h1] at h2
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_two_l701_70137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l701_70147

theorem theta_range (θ : Real) (h1 : 0 < θ) (h2 : θ < π) : 
  (∀ x : Real, Real.cos θ * x^2 - (4 * Real.sin θ) * x + 6 > 0) → θ < π/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l701_70147
