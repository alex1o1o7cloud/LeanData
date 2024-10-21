import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l792_79297

-- Define the cone parameters
def cone_base_radius : ℝ := 10
def cone_height : ℝ := 15

-- Define the number of spheres
def num_spheres : ℕ := 4

-- Define the theorem
theorem sphere_radius_in_cone :
  ∃ (r : ℝ),
    r > 0 ∧
    r = 3 ∧
    (∀ (i j : Fin num_spheres), i ≠ j → ∃ (x y z : ℝ),
      x^2 + y^2 + z^2 = (2*r)^2) ∧
    (∀ (i : Fin num_spheres), ∃ (x y : ℝ),
      x^2 + y^2 = (cone_base_radius - r)^2 ∧
      y = r) ∧
    (∀ (i : Fin num_spheres), ∃ (x y z : ℝ),
      x^2 + y^2 = (cone_base_radius * z / cone_height)^2 ∧
      z + r = Real.sqrt (cone_height^2 + cone_base_radius^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l792_79297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l792_79299

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => seq.a i)

theorem geometric_sequence_sum_five (seq : GeometricSequence) 
  (h1 : seq.a 2 - seq.a 0 = 3)
  (h2 : seq.a 3 - seq.a 1 = 6) :
  sum_n seq 5 = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l792_79299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_six_equals_sixteen_l792_79223

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x ↦ (x^2 + 10*x + 32) / 8

-- State the theorem
theorem f_of_six_equals_sixteen :
  (∀ x : ℝ, f (4*x - 2) = 2*x^2 + 3*x + 2) → f 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_six_equals_sixteen_l792_79223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_specific_point_l792_79243

/-- The point A with coordinates (0, 1) -/
def A : ℝ × ℝ := (0, 1)

/-- The line on which point B moves -/
def line_B (x y : ℝ) : Prop := x + y = 0

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the distance between A and B is minimized when B has coordinates (-1/2, 1/2) -/
theorem min_distance_at_specific_point :
  ∀ (B : ℝ × ℝ), line_B B.1 B.2 →
    distance A B ≥ distance A (-1/2, 1/2) ∧
    line_B (-1/2) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_at_specific_point_l792_79243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l792_79283

/-- Represents the production rate and quantity for a batch of cogs -/
structure CogBatch where
  rate : ℚ  -- Production rate in cogs per hour
  quantity : ℚ  -- Number of cogs to produce

/-- Calculates the overall average output for two consecutive cog batches -/
def overallAverageOutput (batch1 batch2 : CogBatch) : ℚ :=
  let totalCogs := batch1.quantity + batch2.quantity
  let totalTime := batch1.quantity / batch1.rate + batch2.quantity / batch2.rate
  totalCogs / totalTime

/-- Theorem: The overall average output for the given production scenario is 72 cogs per hour -/
theorem assembly_line_output :
  let batch1 := CogBatch.mk 90 60
  let batch2 := CogBatch.mk 60 60
  overallAverageOutput batch1 batch2 = 72 := by
  sorry

#eval overallAverageOutput (CogBatch.mk 90 60) (CogBatch.mk 60 60)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l792_79283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l792_79225

open Set Real

/-- Given sets U and A, prove that the complement of A in U is as specified. -/
theorem complement_of_A_in_U :
  let U : Set ℝ := {x | x > -1}
  let A : Set ℝ := {x | |x - 2| < 1}
  (U \ A) = {x : ℝ | (-1 < x ∧ x ≤ 1) ∨ x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l792_79225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_limit_l792_79242

/-- Represents the probability of getting an odd number after n operations -/
noncomputable def probability_odd (n : ℕ) : ℝ :=
  1/3 * (1 - (1/4)^n) + 1/(2^(2*n + 1))

/-- The limit of the probability as n approaches infinity -/
noncomputable def limit_probability : ℝ := 1/3

/-- Theorem stating that the probability of getting an odd number
    approaches 1/3 as the number of operations increases -/
theorem probability_odd_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |probability_odd n - limit_probability| < ε :=
by
  sorry

#check probability_odd_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_limit_l792_79242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l792_79271

noncomputable section

-- Define the side length of the square
def side_length : ℝ := 4

-- Define the areas of the inscribed shapes
def circle_area : ℝ := Real.pi * (side_length / 2)^2
def semicircles_area : ℝ := Real.pi * (side_length / 2)^2
def triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2

-- Define the shaded areas
def shaded_area_A : ℝ := side_length^2 - circle_area
def shaded_area_B : ℝ := side_length^2 - semicircles_area
def shaded_area_C : ℝ := side_length^2 - triangle_area

-- Theorem statement
theorem largest_shaded_area :
  shaded_area_C > shaded_area_A ∧ shaded_area_C > shaded_area_B := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l792_79271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l792_79213

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = -29/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l792_79213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_deg_matrix_l792_79253

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) Real :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

noncomputable def angle_150_deg : Real := 5 * Real.pi / 6

theorem rotation_150_deg_matrix :
  rotation_matrix angle_150_deg = ![![-Real.sqrt 3 / 2, -1 / 2],
                                    ![1 / 2, -Real.sqrt 3 / 2]] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_deg_matrix_l792_79253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_alpha_l792_79231

/-- Given that the terminal side of angle α passes through the point P(-5, -12),
    prove that sin(3π/2 + α) = 5/13 -/
theorem sin_three_pi_half_plus_alpha (α : ℝ) :
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos α) = -5 ∧ r * (Real.sin α) = -12) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_alpha_l792_79231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_ann_age_difference_l792_79246

/-- Proves that Maria is 3 years younger than Ann given the problem conditions -/
theorem maria_ann_age_difference (ann_age : ℕ) :
  -- Maria's current age
  let maria_age : ℕ := 7
  -- Four years ago, Maria's age was one-half Ann's age
  (maria_age - 4 = (ann_age - 4) / 2) →
  -- The age difference is 3 years
  ann_age - maria_age = 3 := by
  sorry

#check maria_ann_age_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_ann_age_difference_l792_79246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l792_79263

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l792_79263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_vertices_eq_octahedron_faces_eq_octahedron_probability_l792_79275

/-- Probability of being on a middle vertex of an octahedron after n steps -/
def p : ℕ → ℚ
| 0 => 0
| n + 1 => 1 - p n / 2

/-- An octahedron has 6 vertices -/
axiom octahedron_vertices : ℕ

/-- An octahedron has 8 faces -/
axiom octahedron_faces : ℕ

theorem octahedron_vertices_eq : octahedron_vertices = 6 := by sorry

theorem octahedron_faces_eq : octahedron_faces = 8 := by sorry

/-- The probability of being on a middle vertex after 5 steps is 11/16 -/
theorem octahedron_probability : p 5 = 11 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_vertices_eq_octahedron_faces_eq_octahedron_probability_l792_79275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_range_l792_79210

-- Define the curve (marked as noncomputable due to dependency on Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 3/5

-- Define the derivative of the curve (marked as noncomputable due to dependency on Real.sqrt)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

-- Theorem statement
theorem tangent_slope_angle_range :
  ∀ x : ℝ, ∃ α : ℝ, 
    (α ∈ Set.Ioc 0 (Real.pi / 2) ∪ Set.Ico (2 * Real.pi / 3) Real.pi) ∧
    (Real.tan α = f' x) :=
by
  sorry

#check tangent_slope_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_range_l792_79210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_is_49_60_percent_l792_79248

/-- The percentage decrease in value of a baseball card over three years -/
noncomputable def card_value_decrease (initial_value : ℝ) : ℝ :=
  let year1_value := initial_value * (1 - 0.30)
  let year2_value := year1_value * (1 - 0.10)
  let year3_value := year2_value * (1 - 0.20)
  (initial_value - year3_value) / initial_value * 100

/-- Theorem stating that the total percentage decrease in value of the baseball card over three years is 49.60% -/
theorem card_value_decrease_is_49_60_percent (initial_value : ℝ) (h : initial_value > 0) :
  card_value_decrease initial_value = 49.60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_value_decrease_is_49_60_percent_l792_79248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_green_marbles_l792_79282

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def num_trials : ℕ := 7
def num_green_chosen : ℕ := 4

/-- The probability of choosing exactly four green marbles in seven trials -/
theorem probability_four_green_marbles : 
  (Nat.choose num_trials num_green_chosen : ℚ) * 
  (green_marbles / total_marbles : ℚ) ^ num_green_chosen * 
  (purple_marbles / total_marbles : ℚ) ^ (num_trials - num_green_chosen) = 
  49172480 / 170859375 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_green_marbles_l792_79282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l792_79265

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  ∃ (A ω φ : ℝ),
    A > 0 ∧
    ω > 0 ∧
    0 < φ ∧ φ < Real.pi / 2 ∧
    (∀ x : ℝ, f x = A * Real.sin (ω * x + φ)) ∧
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₂ - x₁| = Real.pi / 2) ∧
    (∃ M : ℝ, ∀ x : ℝ, f x ≥ f M) ∧
    (∀ x : ℝ, Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1 ≤ f x ∧ f x ≤ 2) ∧
    (∃ x₁ x₂ : ℝ, Real.pi / 12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f x₁ = -1 ∧ f x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l792_79265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l792_79234

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = 4/5 →
  B = π/3 →
  b = 5 * Real.sqrt 3 →
  a = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l792_79234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_trip_charge_l792_79260

/-- Represents a taxi trip with its parameters and calculates the additional charge per segment -/
noncomputable def TaxiTrip (initial_fee : ℚ) (trip_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let segment_length : ℚ := 2 / 5
  let num_segments : ℚ := trip_distance / segment_length
  let distance_charge : ℚ := total_charge - initial_fee
  distance_charge / num_segments

/-- Theorem stating that for the given taxi trip parameters, the additional charge per 2/5 of a mile is $0.35 -/
theorem taxi_trip_charge : TaxiTrip (235 / 100) (18 / 5) (11 / 2) = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_trip_charge_l792_79260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_arrival_time_l792_79215

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds two Times together -/
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- Calculates the time difference between two Times -/
def Time.diff (t1 t2 : Time) : Time :=
  let totalMinutes := (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem flight_arrival_time 
  (toronto_departure : Time)
  (flight_duration : Time)
  (toronto_gander_diff : Time)
  (h1 : toronto_departure = { hours := 15, minutes := 0 })
  (h2 : flight_duration = { hours := 2, minutes := 50 })
  (h3 : toronto_gander_diff = { hours := 1, minutes := 30 })
  : Time.add (Time.add toronto_departure flight_duration) toronto_gander_diff = { hours := 19, minutes := 20 } := by
  sorry

#eval Time.add (Time.add { hours := 15, minutes := 0 } { hours := 2, minutes := 50 }) { hours := 1, minutes := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_arrival_time_l792_79215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_sum_difference_no_1996_sum_l792_79262

/-- Represents a path sum in an n × n array -/
def PathSum (n : ℕ) : Type := ℕ

/-- The maximum path sum in an n × n array -/
def M (n : ℕ) : ℕ :=
  (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2

/-- The minimum path sum in an n × n array -/
def m (n : ℕ) : ℕ :=
  (n^3 + 2 * n^2 - n) / 2

theorem path_sum_difference (n : ℕ) : M n - m n = (n - 1)^3 := by
  sorry

theorem no_1996_sum : ∀ n : ℕ, ¬∃ (s : ℕ), m n ≤ s ∧ s ≤ M n ∧ s = 1996 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_sum_difference_no_1996_sum_l792_79262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_without_music_is_four_l792_79229

/-- Represents the running scenario with given conditions -/
structure RunningScenario where
  speed_with_music : ℚ
  music_duration : ℚ
  total_distance : ℚ
  total_time : ℚ

/-- Calculates the speed without music given a running scenario -/
noncomputable def speed_without_music (scenario : RunningScenario) : ℚ :=
  let distance_with_music := scenario.speed_with_music * (scenario.music_duration / 60)
  let remaining_distance := scenario.total_distance - distance_with_music
  let remaining_time := scenario.total_time - scenario.music_duration
  remaining_distance / (remaining_time / 60)

/-- Theorem stating that under the given conditions, the speed without music is 4 MPH -/
theorem speed_without_music_is_four
  (scenario : RunningScenario)
  (h1 : scenario.speed_with_music = 6)
  (h2 : scenario.music_duration = 40)
  (h3 : scenario.total_distance = 6)
  (h4 : scenario.total_time = 70) :
  speed_without_music scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_without_music_is_four_l792_79229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l792_79257

/-- The parabola function -/
noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The area of the triangle as a function of p -/
noncomputable def area (p : ℝ) : ℝ := (3/2) * (p - 1) * (3 - p)

theorem max_triangle_area :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 3 ∧
  (∀ (x : ℝ), 0 ≤ x → x ≤ 3 → area x ≤ area p) ∧
  area p = 3/2 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l792_79257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l792_79201

theorem diophantine_equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | 2^x + 3^y = z^2} =
    {(3, 0, 3), (0, 1, 2), (4, 2, 5)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l792_79201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_divisibility_by_41_l792_79255

/-- A function that performs a single right cyclic shift on a five-digit number -/
def cycleRight (n : ℕ) : ℕ :=
  (n / 10) + (n % 10) * 10000

/-- Proposition: For any five-digit number divisible by 41, its cyclic permutations are also divisible by 41 -/
theorem cyclic_divisibility_by_41 (n : ℕ) (h1 : 10000 ≤ n ∧ n < 100000) (h2 : n % 41 = 0) :
  ∀ k : ℕ, k < 5 → (Nat.iterate cycleRight k n) % 41 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_divisibility_by_41_l792_79255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_average_speed_l792_79207

structure Car where
  startTime : ℕ
  endTime : ℕ
  distance : ℕ

def averageSpeed (car : Car) : ℚ :=
  car.distance / (car.endTime - car.startTime)

theorem highest_average_speed (carA carB carC : Car)
  (hA : carA.startTime = 0 ∧ carA.endTime = 3 ∧ carA.distance = 150)
  (hB : carB.startTime = 3 ∧ carB.endTime = 7 ∧ carB.distance = 320)
  (hC : carC.startTime = 7 ∧ carC.endTime = 10 ∧ carC.distance = 210) :
  averageSpeed carB > averageSpeed carA ∧ averageSpeed carB > averageSpeed carC :=
by
  sorry

#eval averageSpeed { startTime := 0, endTime := 3, distance := 150 }
#eval averageSpeed { startTime := 3, endTime := 7, distance := 320 }
#eval averageSpeed { startTime := 7, endTime := 10, distance := 210 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_average_speed_l792_79207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l792_79204

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 5/3
  | (n+3) => 5/3 * sequence_a (n+2) - 2/3 * sequence_a (n+1)

def sequence_b (n : ℕ) : ℚ := sequence_a (n+1) - sequence_a n

theorem sequence_properties :
  (∀ n : ℕ, sequence_b n = (2/3)^n) ∧
  (∀ n : ℕ, sequence_a n = 3 - 2 * (2/3)^(n-1)) := by
  sorry

#eval sequence_a 5  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l792_79204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l792_79249

/-- Predicate indicating that a triangle with sides a, b, c is tangent to a sphere of radius r -/
def IsTangent (r a b c : ℝ) : Prop :=
  ∃ (x y z : ℝ), x^2 + y^2 + z^2 = r^2 ∧
    (x + r)^2 + y^2 + z^2 = a^2 ∧
    (x - r)^2 + y^2 + z^2 = b^2 ∧
    x^2 + (y + r)^2 + z^2 = c^2

/-- The distance from the center of a sphere to the plane of a triangle tangent to the sphere -/
noncomputable def DistanceSpherePlane (r a b c : ℝ) : ℝ :=
  Real.sqrt (r^2 - (a * b * c / (a + b + c))^2)

/-- The distance from the center of a sphere to the plane of a triangle tangent to the sphere -/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_radius : r = 8)
  (h_triangle : a = 13 ∧ b = 13 ∧ c = 10) (h_tangent : IsTangent r a b c) :
  DistanceSpherePlane r a b c = 2 * Real.sqrt 119 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l792_79249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_diff_identities_l792_79294

theorem trig_sum_diff_identities (A B : ℝ) (h : A + B < π / 2) :
  (Real.cos (A + B) = Real.cos A * Real.cos B - Real.sin A * Real.sin B) ∧
  (Real.sin (A + B) = Real.sin A * Real.cos B + Real.cos A * Real.sin B) ∧
  (Real.cos (A - B) = Real.cos A * Real.cos B + Real.sin A * Real.sin B) ∧
  (Real.sin (A - B) = Real.sin A * Real.cos B - Real.cos A * Real.sin B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_diff_identities_l792_79294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachability_symmetric_l792_79214

/-- Represents a string as a list of characters -/
def MyString := List Char

/-- Defines reachability between two strings -/
def is_reachable (A B : MyString) : Prop :=
  ∃ (inserted : List Nat), 
    let typed := List.foldl 
      (λ acc (i : Nat) => 
        if i < acc.length 
        then acc.take i ++ acc.drop (i + 1) ++ [acc.get! i]
        else acc) 
      A inserted
    typed = B

/-- The main theorem: reachability is symmetric -/
theorem reachability_symmetric (A B : MyString) : 
  is_reachable A B ↔ is_reachable B A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachability_symmetric_l792_79214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_trig_functions_l792_79285

/-- Given that y = sin x + a cos x is symmetric about x = 5π/3, 
    prove that y = a sin x + cos x is symmetric about x = 11π/6 --/
theorem symmetry_of_trig_functions (a : ℝ) :
  (∀ x : ℝ, Real.sin x + a * Real.cos x = Real.sin (10 * Real.pi / 3 - x) + a * Real.cos (10 * Real.pi / 3 - x)) →
  (∀ x : ℝ, a * Real.sin x + Real.cos x = a * Real.sin (11 * Real.pi / 3 - x) + Real.cos (11 * Real.pi / 3 - x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_trig_functions_l792_79285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l792_79286

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.S₁ - t.S₂ + t.S₃ = Real.sqrt 3 / 2)
  (h2 : Real.sin t.B = 1 / 3) :
  -- Part 1: Area of triangle ABC
  (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 2 / 8) ∧
  -- Part 2: If sin A * sin C = √2/3, then b = 1/2
  ((Real.sin t.A * Real.sin t.C = Real.sqrt 2 / 3) → t.b = 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l792_79286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_action_movies_rented_l792_79237

def comedies_rented : ℕ := 15
def comedy_to_action_ratio : ℚ := 3 / 1

theorem action_movies_rented : 
  comedies_rented / (comedy_to_action_ratio.num.toNat) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_action_movies_rented_l792_79237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_palindrome_div_by_11_probability_l792_79279

/-- A 5-digit palindrome is a number of the form abcba where a, b, c are digits and a ≠ 0 -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ∃ a b c, a ≠ 0 ∧ n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- The set of all 5-digit palindromes -/
def five_digit_palindromes : Set ℕ :=
  {n : ℕ | is_five_digit_palindrome n}

/-- The set of 5-digit palindromes divisible by 11 -/
def five_digit_palindromes_div_by_11 : Set ℕ :=
  {n : ℕ | is_five_digit_palindrome n ∧ n % 11 = 0}

/-- Count of 5-digit palindromes -/
def count_five_digit_palindromes : ℕ := 900

/-- Count of 5-digit palindromes divisible by 11 -/
def count_five_digit_palindromes_div_by_11 : ℕ := 77

theorem five_digit_palindrome_div_by_11_probability :
  (count_five_digit_palindromes_div_by_11 : ℚ) / (count_five_digit_palindromes : ℚ) = 77 / 900 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_palindrome_div_by_11_probability_l792_79279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangles_exist_l792_79261

/-- Triangle with given heights and median -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_a : ℝ
  m_a : ℝ
  h_b : ℝ

/-- Predicate to check if a triangle satisfies the given measurements -/
def satisfies_measurements (t : Triangle) : Prop :=
  let M := ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)
  let BC_length := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let AM_length := Real.sqrt ((t.A.1 - M.1)^2 + (t.A.2 - M.2)^2)
  let area := BC_length * t.h_a / 2
  area = AM_length * t.h_b / 2 ∧
  AM_length = t.m_a ∧
  area = BC_length * t.h_a / 2

theorem two_triangles_exist (h_a m_a h_b : ℝ) :
  ∃ t1 t2 : Triangle, t1 ≠ t2 ∧
    t1.h_a = h_a ∧ t1.m_a = m_a ∧ t1.h_b = h_b ∧
    t2.h_a = h_a ∧ t2.m_a = m_a ∧ t2.h_b = h_b ∧
    satisfies_measurements t1 ∧ satisfies_measurements t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangles_exist_l792_79261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_property_l792_79244

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_first_two : a 1 + a 3 = 8
  sum_middle_two : a 2 + a 4 = 12

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem arithmetic_geometric_property (seq : ArithmeticSequence) :
  ∃ k : ℕ, k > 0 ∧ (seq.a 1) * (seq.a k) = (seq.a 1) * (sum_n seq (k + 2)) ∧ k = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_property_l792_79244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_in_specific_triangle_l792_79268

/-- A right triangle with given side lengths -/
structure RightTriangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  is_right : de^2 + df^2 = ef^2

/-- The distance from the right angle to the midpoint of the hypotenuse -/
noncomputable def median_to_hypotenuse (t : RightTriangle) : ℝ :=
  t.ef / 2

theorem median_length_in_specific_triangle :
  let t : RightTriangle := {
    de := 15,
    df := 20,
    ef := 25,
    is_right := by norm_num
  }
  median_to_hypotenuse t = 12.5 := by
    unfold median_to_hypotenuse
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_in_specific_triangle_l792_79268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l792_79224

noncomputable def circled_plus (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

noncomputable def star (a b : ℝ) : ℝ := Real.sqrt ((a - b)^2)

noncomputable def f (x : ℝ) : ℝ := (circled_plus 2 x) / (2 - star x 2)

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l792_79224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ABCD_is_4252_l792_79226

-- Define the grid
def Grid := Fin 6 → Fin 6 → ℕ

-- Define a region
structure Region where
  cells : List (Fin 6 × Fin 6)

-- Define adjacency
def adjacent : (Fin 6 × Fin 6) → (Fin 6 × Fin 6) → Prop
| (i, j), (k, l) => (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
                    (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

-- Define the property of a valid grid filling
def valid_filling (g : Grid) (regions : List Region) : Prop :=
  -- Each region is filled with numbers 1 through N, where N is the number of cells in the region
  ∀ r ∈ regions, ∀ n ∈ Finset.range r.cells.length, ∃ c ∈ r.cells, g c.1 c.2 = n + 1
  ∧ -- Adjacent cells have different numbers
  ∀ i j k l : Fin 6, adjacent (i, j) (k, l) → g i j ≠ g k l

-- Define the specific grid layout for this problem
def problem_grid : List Region := sorry

-- Define the positions of A, B, C, D
def pos_A : Fin 6 × Fin 6 := sorry
def pos_B : Fin 6 × Fin 6 := sorry
def pos_C : Fin 6 × Fin 6 := sorry
def pos_D : Fin 6 × Fin 6 := sorry

-- The main theorem
theorem ABCD_is_4252 :
  ∀ g : Grid, valid_filling g problem_grid →
    g pos_A.1 pos_A.2 = 4 ∧
    g pos_B.1 pos_B.2 = 2 ∧
    g pos_C.1 pos_C.2 = 5 ∧
    g pos_D.1 pos_D.2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ABCD_is_4252_l792_79226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_codes_to_check_l792_79277

/-- Represents a five-digit code -/
def FiveDigitCode := Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10

/-- Check if a FiveDigitCode contains both 21 and 16 -/
def containsBoth21And16 (code : FiveDigitCode) : Prop :=
  (code.1 = 2 ∧ code.2.1 = 1) ∨
  (code.2.1 = 2 ∧ code.2.2.1 = 1) ∨
  (code.2.2.1 = 2 ∧ code.2.2.2.1 = 1) ∨
  (code.2.2.2.1 = 2 ∧ code.2.2.2.2 = 1) ∨
  (code.1 = 1 ∧ code.2.1 = 6) ∨
  (code.2.1 = 1 ∧ code.2.2.1 = 6) ∨
  (code.2.2.1 = 1 ∧ code.2.2.2.1 = 6) ∨
  (code.2.2.2.1 = 1 ∧ code.2.2.2.2 = 6)

/-- The set of all valid codes containing both 21 and 16 -/
def validCodes : Set FiveDigitCode :=
  {code | containsBoth21And16 code}

/-- Theorem: The minimum number of codes to check is 6 -/
theorem min_codes_to_check :
  ∃ (codes : Finset FiveDigitCode),
    codes.card = 6 ∧
    (∀ code, containsBoth21And16 code → code ∈ codes) ∧
    (∀ (codes' : Finset FiveDigitCode),
      (∀ code, containsBoth21And16 code → code ∈ codes') →
      codes'.card ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_codes_to_check_l792_79277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_settings_count_l792_79200

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of digits on each dial -/
def digits_per_dial : ℕ := 10

/-- The number of different settings possible for the lock -/
def num_settings : ℕ := 5040

/-- Calculates the factorial of a number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The number of different settings for a lock with 'num_dials' dials, 
    where each dial has 'digits_per_dial' digits and all digits must be different, 
    is equal to 'num_settings' -/
theorem lock_settings_count : 
  (factorial digits_per_dial) / (factorial (digits_per_dial - num_dials)) = num_settings :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_settings_count_l792_79200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_amount_theorem_l792_79272

/-- Calculates the amount of jasmine added to a solution -/
noncomputable def jasmine_added (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (added_water : ℝ) : ℝ :=
  let initial_jasmine := initial_volume * initial_concentration
  let final_volume := initial_volume + added_water + 
    (final_concentration * (initial_volume + added_water) - initial_jasmine) / (1 - final_concentration)
  (final_volume * final_concentration) - initial_jasmine

/-- Theorem stating the amount of jasmine added to the solution -/
theorem jasmine_amount_theorem :
  jasmine_added 80 0.1 0.16 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_amount_theorem_l792_79272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_calculation_l792_79254

/-- Calculates the markup percentage given cost price, selling price, and discount percentage. -/
noncomputable def calculate_markup_percentage (cost_price selling_price discount_percentage : ℝ) : ℝ :=
  let marked_price := selling_price / (1 - discount_percentage / 100)
  let markup := (marked_price - cost_price) / cost_price
  markup * 100

/-- Theorem stating that the markup percentage is approximately 14.944444444444445% 
    given the specific cost price, selling price, and discount percentage. -/
theorem markup_percentage_calculation :
  let cost_price : ℝ := 540
  let selling_price : ℝ := 462
  let discount_percentage : ℝ := 25.603864734299517
  let calculated_markup := calculate_markup_percentage cost_price selling_price discount_percentage
  abs (calculated_markup - 14.944444444444445) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_calculation_l792_79254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_is_one_l792_79291

-- Define the piecewise linear function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else -x + 2

-- Define the function g(x) = x * f(x)
noncomputable def g (x : ℝ) : ℝ := x * f x

-- Theorem statement
theorem area_under_curve_is_one :
  ∫ x in (0 : ℝ)..(2 : ℝ), g x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_is_one_l792_79291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ravenswood_remaining_gnomes_l792_79238

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken from Ravenswood forest -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_ravenswood_gnomes : ℕ := 48

theorem ravenswood_remaining_gnomes :
  remaining_ravenswood_gnomes = 
    (ravenswood_ratio * westerville_gnomes) - 
    (Int.toNat ((ravenswood_ratio * westerville_gnomes : ℚ) * taken_percentage).floor) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ravenswood_remaining_gnomes_l792_79238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_on_parabola_l792_79245

/-- Given points P and Q on the parabola y = -x², prove the length of PO in isosceles triangle POQ --/
theorem isosceles_triangle_on_parabola (p : ℝ) :
  let P : ℝ × ℝ := (p, -p^2)
  let Q : ℝ × ℝ := (-p, -p^2)
  let O : ℝ × ℝ := (0, 0)
  (‖P - O‖ = ‖Q - O‖) →  -- isosceles condition
  ‖P - O‖ = p * Real.sqrt 2 :=
by
  -- Introduce the let-bindings
  intro P Q O h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_on_parabola_l792_79245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l792_79278

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x / (exp x)

-- State the theorem
theorem f_decreasing_on_interval (a b : ℝ) (h1 : a < b) (h2 : b < 1) : f a > f b := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l792_79278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l792_79252

-- Define the initial and final radii
def initial_radius : ℝ := 8
def final_radius : ℝ := 10

-- Define the function to calculate the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the percent increase function
noncomputable def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

-- Theorem statement
theorem garden_area_increase :
  percent_increase (circle_area initial_radius) (circle_area final_radius) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l792_79252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_carnation_cost_l792_79208

/-- Represents the cost of carnations in cents -/
abbrev CarnationCost := Nat

/-- Represents the quantity of carnations -/
abbrev CarnationQuantity := Nat

/-- Cost of a single carnation in cents -/
def single_cost : CarnationCost := 50

/-- Cost of a dozen carnations in cents -/
def dozen_cost : CarnationCost := 400

/-- Cost of a bundle of 25 carnations in cents -/
def bundle_cost : CarnationCost := 800

/-- Calculate the cost of individual carnations -/
def individual_cost (quantity : CarnationQuantity) : CarnationCost :=
  quantity * single_cost

/-- Calculate the cost of dozens of carnations -/
def dozens_cost (quantity : CarnationQuantity) : CarnationCost :=
  quantity * dozen_cost

/-- Calculate the cost of bundles of carnations -/
def bundles_cost (quantity : CarnationQuantity) : CarnationCost :=
  quantity * bundle_cost

/-- Calculate the total cost for teachers -/
def teachers_cost : CarnationCost :=
  dozens_cost 1 +  -- Teacher 1
  (dozens_cost 1 + individual_cost 3) +  -- Teacher 2
  bundle_cost +  -- Teacher 3
  (dozens_cost 2 + individual_cost 9) +  -- Teacher 4
  dozens_cost 3  -- Teacher 5

/-- Calculate the total cost for friends -/
def friends_cost : CarnationCost :=
  individual_cost 3 +  -- Friends 1-3
  dozens_cost 1 +  -- Friend 4
  individual_cost 9 +  -- Friends 5-7
  individual_cost 7 +  -- Friend 8
  dozens_cost 2 +  -- Friend 9
  individual_cost 10 +  -- Friends 10-11
  individual_cost 10 +  -- Friend 12
  (dozens_cost 1 + individual_cost 5) +  -- Friend 13
  bundle_cost  -- Friend 14

/-- The main theorem stating the total cost -/
theorem total_carnation_cost : teachers_cost + friends_cost = 8800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_carnation_cost_l792_79208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l792_79273

open Real

-- Define the domain of the tangent function
def TanDomain (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 2 < x ∧ x < k * Real.pi + Real.pi / 2

-- Define the solution set
def SolutionSet (x : ℝ) : Prop :=
  ∃ k : ℤ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x < Real.pi / 2 + k * Real.pi

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, TanDomain x →
  (1 + Real.sqrt 3 * Real.tan x ≥ 0 ↔ SolutionSet x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l792_79273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l792_79296

-- Define the original function
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x else x^2

-- Define the proposed inverse function
noncomputable def g (y : ℝ) : ℝ :=
  if y < 0 then y else Real.sqrt y

-- State the theorem
theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l792_79296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l792_79232

open Set
open Function

theorem no_such_function_exists :
  ¬∃ (f : ℝ → ℝ), (∀ x, 0 < f x) ∧ (∀ x y, 0 < x → 0 < y → f (x + y) ≥ f x + y * f (f x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l792_79232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_heights_l792_79241

theorem basketball_team_heights 
  (tallest : ℝ) 
  (diff_1_2 : ℝ) 
  (diff_2_3 : ℝ) 
  (diff_3_4 : ℝ) 
  (diff_4_5 : ℝ) 
  (h_tallest : tallest = 80.5)
  (h_diff_1_2 : diff_1_2 = 6.25)
  (h_diff_2_3 : diff_2_3 = 3.75)
  (h_diff_3_4 : diff_3_4 = 5.5)
  (h_diff_4_5 : diff_4_5 = 4.8) :
  let second := tallest - diff_1_2
  let third := second - diff_2_3
  let fourth := third - diff_3_4
  let shortest := fourth - diff_4_5
  second = 74.25 ∧ third = 70.5 ∧ fourth = 65 ∧ shortest = 60.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_heights_l792_79241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l792_79220

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then |x^2 - 3*x| else |(-x)^2 - 3*(-x)|

theorem solution_set_equality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x))
  (h2 : ∀ x, x ≥ 0 → f x = |x^2 - 3*x|) :
  {x : ℝ | f (x - 2) ≤ 2} = 
  {x : ℝ | -3 ≤ x ∧ x ≤ 1 ∨ 
           0 ≤ x ∧ x ≤ (Real.sqrt 17 - 1) / 2 ∨ 
           -(7 + Real.sqrt 17) / 2 ≤ x ∧ x ≤ -4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l792_79220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l792_79218

-- Define the vector product
def vectorProduct (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

-- Define the given vectors
noncomputable def m : ℝ × ℝ := (2, 1/2)
noncomputable def n : ℝ × ℝ := (Real.pi/3, 0)

-- Define the function for point P
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, Real.sin x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let OQ := vectorProduct m (P x) + n
  OQ.2

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l792_79218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_common_points_condition_l792_79230

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a triangle ABC, returns its circumcenter -/
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Given a triangle ABC and a point D on BC, returns true if AD is an internal angle bisector -/
def isInternalBisector (t : Triangle) (D : ℝ × ℝ) : Prop := sorry

/-- Given two points, returns the ray starting from the first point and passing through the second -/
def ray (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Given three points A, B, C on a line, returns the ratio AB:BC -/
noncomputable def ratio (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a circle and two points, returns true if the circle passes through the first point and is tangent to the line OA at A, where O is the second point -/
def circlePassesThroughAndTangent (c : Circle) (L A : ℝ × ℝ) : Prop := sorry

/-- Given a set of circles, returns true if they have exactly two common points -/
def haveExactlyTwoCommonPoints (circles : List Circle) : Prop := sorry

/-- Returns true if the triangle is acute -/
def Triangle.isAcute (t : Triangle) : Prop := sorry

/-- Returns true if the triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem circle_common_points_condition (t : Triangle) (k : ℝ) 
  (h_acute : t.isAcute)
  (h_not_isosceles : ¬ t.isIsosceles)
  (O : ℝ × ℝ)
  (h_O : O = circumcenter t)
  (D E F : ℝ × ℝ)
  (h_D : isInternalBisector t D)
  (h_E : isInternalBisector t E)
  (h_F : isInternalBisector t F)
  (L : ℝ × ℝ)
  (h_L : L ∈ ray t.A D)
  (M : ℝ × ℝ)
  (h_M : M ∈ ray t.B E)
  (N : ℝ × ℝ)
  (h_N : N ∈ ray t.C F)
  (h_ratio : ratio t.A L D = ratio t.B M E ∧ ratio t.B M E = ratio t.C N F ∧ ratio t.C N F = k)
  (O₁ O₂ O₃ : Circle)
  (h_O₁ : circlePassesThroughAndTangent O₁ L t.A)
  (h_O₂ : circlePassesThroughAndTangent O₂ M t.B)
  (h_O₃ : circlePassesThroughAndTangent O₃ N t.C)
  (h_k_pos : k > 0) :
  haveExactlyTwoCommonPoints [O₁, O₂, O₃] ↔ (k = 1/2 ∨ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_common_points_condition_l792_79230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_distribution_l792_79269

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) (h1 : total_cards = 60) (h2 : num_people = 10) :
  let min_cards := total_cards / num_people
  (∀ p, p ≤ num_people → p > 0 → min_cards ≤ (total_cards + p - 1) / p) →
  (Finset.filter (fun i => i ≤ num_people ∧ i > 0 ∧ (total_cards + i - 1) / i < 6) (Finset.range (num_people + 1))).card = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_distribution_l792_79269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l792_79203

theorem calculation_proof : Real.sqrt 2 * Real.sqrt 6 + |2 - Real.sqrt 3| - (1/2)^(-2 : Int) = Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l792_79203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_perimeter_value_l792_79280

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  Real.cos t.A = 1/3 ∧
  t.a = 4 * Real.sqrt 2 ∧
  t.B = Real.pi/6

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : Real :=
  (1/2) * t.b * t.c * Real.sin t.A

-- Define the perimeter of the triangle
def triangle_perimeter (t : Triangle) : Real :=
  t.a + t.b + t.c

-- Theorem 1: If the conditions are met, then b = 3
theorem b_value (t : Triangle) (h : triangle_conditions t) : t.b = 3 := by
  sorry

-- Theorem 2: If the conditions are met and the area is 2√2, then the perimeter is 4√2 + 4√3
theorem perimeter_value (t : Triangle) (h1 : triangle_conditions t) (h2 : triangle_area t = 2 * Real.sqrt 2) :
  triangle_perimeter t = 4 * Real.sqrt 2 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_perimeter_value_l792_79280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_problem_binomial_sum_problem_l792_79292

-- Part 1
theorem binomial_coefficient_problem (a : ℝ) : 
  (∃ k : ℕ, k * a^4 * 2^2 = 960 ∧ Nat.choose 6 2 = k) → a = 2 := by sorry

-- Part 2
theorem binomial_sum_problem (a n : ℕ) :
  (a + 2)^n = 3^10 ∧ n + a = 12 → (Finset.range (n + 1)).sum (λ i ↦ Nat.choose n i) = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_problem_binomial_sum_problem_l792_79292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_main_theorem_l792_79222

-- Define the ∇ operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_calculation :
  ∀ (a b c d : ℝ), 0 < a → 0 < b → 0 < c → 0 < d →
  nabla (nabla a b) (nabla c d) = nabla (nabla 2 3) (nabla 4 5) :=
by
  sorry

-- Main theorem to prove
theorem main_theorem : nabla (nabla 2 3) (nabla 4 5) = 49 / 56 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_main_theorem_l792_79222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_times_fixed_point_l792_79202

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1/2 then 2*x else 2*x - 1

def solution_set : Set ℝ :=
  {x | x = 0 ∨ (∃ k : ℕ, k ≤ 30 ∧ x = k / 31) ∨ x = 1}

theorem f_five_times_fixed_point (x : ℝ) :
  f (f (f (f (f x)))) = x ↔ x ∈ solution_set := by
  sorry

#check f_five_times_fixed_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_times_fixed_point_l792_79202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_water_limit_l792_79251

-- Define constants
def max_liquid : ℝ := 1000 -- maximum liquid in milliliters
def milk_consumed : ℝ := 250 -- milk consumed in milliliters
def juice_consumed : ℝ := 500 -- juice consumed in milliliters
def ml_per_oz : ℝ := 29.57 -- milliliters per ounce

-- Theorem statement
theorem jamie_water_limit :
  let total_consumed := milk_consumed + juice_consumed
  let remaining_ml := max_liquid - total_consumed
  let remaining_oz := remaining_ml / ml_per_oz
  ∃ (ε : ℝ), ε > 0 ∧ |remaining_oz - 8.45| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_water_limit_l792_79251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_distributive_laws_l792_79298

noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

theorem avg_distributive_laws :
  (∃ x y z : ℝ, avg x (y - z) ≠ avg x y - avg x z) ∧
  (∀ x y z : ℝ, x - avg y z = avg (x - y) (x - z)) ∧
  (∀ x y z : ℝ, avg x (avg y z) = avg (avg x y) (avg x z)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_distributive_laws_l792_79298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_problem_l792_79236

theorem hcf_problem (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (lcm_eq : Nat.lcm x.val y.val = 120)
  (sum_reciprocals : 1 / (x.val : ℚ) + 1 / (y.val : ℚ) = 11 / 120) :
  Nat.gcd x.val y.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_problem_l792_79236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_outer_distance_not_equal_l792_79266

-- Define the wheel
structure Wheel where
  outer_diameter : ℝ
  inner_diameter : ℝ
  h_positive_outer : outer_diameter > 0
  h_positive_inner : inner_diameter > 0
  h_inner_smaller : inner_diameter < outer_diameter

-- Define the distance traveled by a point on a circle during one rotation
noncomputable def distance_traveled (diameter : ℝ) : ℝ := Real.pi * diameter

-- Theorem statement
theorem inner_outer_distance_not_equal (w : Wheel) :
  distance_traveled w.outer_diameter ≠ distance_traveled w.inner_diameter := by
  -- Proof steps would go here
  sorry

#check inner_outer_distance_not_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_outer_distance_not_equal_l792_79266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_difference_term_l792_79212

/-- In regression analysis, the term representing the difference between a data point
    and its corresponding position on the regression line is the sum of squares of residuals. -/
theorem regression_difference_term : 
  ∃ (regression_difference_term sum_of_squares_of_residuals : ℝ → ℝ → ℝ),
  ∀ (data_point regression_line_point : ℝ),
  regression_difference_term data_point regression_line_point = 
    sum_of_squares_of_residuals data_point regression_line_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_difference_term_l792_79212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l792_79290

theorem equilateral_triangle_on_parabola :
  ∀ p : ℝ,
  let P : ℝ × ℝ := (p, -p^2)
  let Q : ℝ × ℝ := (-p, -p^2)
  let O : ℝ × ℝ := (0, 0)
  (dist P O = dist Q O) →
  (dist P O = dist P Q) →
  (dist P O = 2 * Real.sqrt 3) :=
by
  intro p P Q O h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l792_79290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_and_intersection_l792_79239

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_and_intersection :
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  ((𝕌 \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_and_intersection_l792_79239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacob_has_more_money_l792_79211

/-- Represents the value of different coin types in cents -/
def coin_value : Fin 7 → ℕ
| 0 => 1    -- penny
| 1 => 5    -- nickel
| 2 => 10   -- dime
| 3 => 25   -- quarter
| 4 => 50   -- half-dollar
| 5 => 100  -- dollar coin
| 6 => 100  -- dollar bill

/-- Calculates the total value of coins in cents -/
def total_value (coins : Fin 7 → ℕ) : ℕ :=
  (Finset.univ.sum fun i => coins i * coin_value i)

/-- Mrs. Hilt's coin counts -/
def mrs_hilt_coins : Fin 7 → ℕ
| 0 => 3  -- pennies
| 1 => 2  -- nickels
| 2 => 2  -- dimes
| 3 => 5  -- quarters
| 4 => 0  -- half-dollars
| 5 => 1  -- dollar coins
| 6 => 0  -- dollar bills

/-- Jacob's coin counts -/
def jacob_coins : Fin 7 → ℕ
| 0 => 4  -- pennies
| 1 => 1  -- nickels
| 2 => 1  -- dimes
| 3 => 3  -- quarters
| 4 => 2  -- half-dollars
| 5 => 0  -- dollar coins
| 6 => 2  -- dollar bills

theorem jacob_has_more_money : 
  total_value jacob_coins - total_value mrs_hilt_coins = 136 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacob_has_more_money_l792_79211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equation_one_real_root_l792_79281

theorem determinant_equation_one_real_root
  (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  ∃! x : ℝ, Matrix.det !![x, p, -r; -p, x, q; r, -q, x] = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equation_one_real_root_l792_79281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_zero_l792_79217

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expressions
noncomputable def expr1 : ℂ := ((1 + i) / (1 - i)) ^ 2017
noncomputable def expr2 : ℂ := ((1 - i) / (1 + i)) ^ 2017

-- State the theorem
theorem complex_sum_zero : expr1 + expr2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_zero_l792_79217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_units_l792_79293

/-- Represents the number of modular units in a satellite -/
def U : ℕ := 24

/-- Represents the number of non-upgraded sensors per unit -/
def N : ℚ := 1  -- Placeholder value

/-- Represents the total number of upgraded sensors on the entire satellite -/
def S : ℚ := 1  -- Placeholder value

theorem satellite_units :
  (∀ (u₁ u₂ : ℕ), N = N) →  -- Each unit contains the same number of non-upgraded sensors
  N = (1 / 4 : ℚ) * S →  -- Non-upgraded sensors on one unit is 1/4 of total upgraded sensors
  S / (S + U * N) = (1 / 7 : ℚ) →  -- Fraction of upgraded sensors is 1/7 (≈ 0.14285714285714285)
  U = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_units_l792_79293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_to_sine_product_l792_79250

open Real

theorem cosine_sum_to_sine_product :
  ∀ x : ℝ,
  (cos x)^2 + (cos (2*x))^2 + (cos (3*x))^2 + (cos (4*x))^2 = 2 ↔
  sin x * sin (2*x) * sin (5*x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_to_sine_product_l792_79250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l792_79256

theorem functional_equation_solution (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, n ≥ 1 → f n + f (n + 1) = f (n + 2) * f (n + 3) - 11)
  (h2 : ∀ n : ℕ, n ≥ 1 → f n ≥ 2) :
  ∃ (a b : ℕ), 
    ((a = 13 ∧ b = 2) ∨ (a = 7 ∧ b = 3) ∨ (a = 5 ∧ b = 4) ∨
     (a = 2 ∧ b = 13) ∨ (a = 3 ∧ b = 7) ∨ (a = 4 ∧ b = 5)) ∧
    (∀ n : ℕ, n ≥ 1 → f n = if n % 2 = 1 then a else b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l792_79256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_l792_79219

theorem cos_double_angle_special (α : ℝ) :
  Real.sin (α + π / 5) = Real.sqrt 3 / 3 →
  Real.cos (2 * α + 2 * π / 5) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_l792_79219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_50_eq_49_l792_79274

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℕ := n^2 + n + 1

/-- The nth term of sequence a_n -/
def a : ℕ → ℕ
  | 0 => 3
  | n+1 => 2 * (n + 1)

/-- The nth term of sequence b_n -/
def b (n : ℕ) : ℤ := (-1)^n * (a n - 2)

/-- The sum of the first 50 terms of sequence b_n -/
def sum_b_50 : ℤ := (Finset.range 50).sum (fun i => b (i + 1))

theorem sum_b_50_eq_49 : sum_b_50 = 49 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_b_50_eq_49_l792_79274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l792_79276

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_of_angle_between_vectors (a b : V) 
  (ha : ‖a‖ = 5)
  (hb : ‖b‖ = 9)
  (hab : ‖a + b‖ = 10) :
  inner a b / (‖a‖ * ‖b‖) = -1/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l792_79276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_is_58_point_5_l792_79221

noncomputable section

def group1_percent : ℚ := 20
def group2_percent : ℚ := 50
def group3_percent : ℚ := 25
def group4_percent : ℚ := 100 - (group1_percent + group2_percent + group3_percent)

def group1_average : ℚ := 80
def group2_average : ℚ := 60
def group3_average : ℚ := 40
def group4_average : ℚ := 50

def overall_average : ℚ := (group1_percent * group1_average + 
                            group2_percent * group2_average + 
                            group3_percent * group3_average + 
                            group4_percent * group4_average) / 100

theorem class_average_is_58_point_5 :
  overall_average = 117/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_is_58_point_5_l792_79221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l792_79205

theorem cos_sin_equation (x : ℝ) :
  Real.cos x - 7 * Real.sin x = 5 →
  Real.sin x - 3 * Real.cos x = -25/7 ∨ Real.sin x - 3 * Real.cos x = 15/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l792_79205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l792_79295

noncomputable def f (a x : ℝ) := |x - a| + |x - 1/2|

theorem problem_solution :
  (∀ x : ℝ, f (5/2) x ≤ x + 10 ↔ -7/3 ≤ x ∧ x ≤ 13) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l792_79295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_Q_satisfies_conditions_l792_79206

-- Define the two given planes
def plane1 (x y z : ℝ) : Prop := 3*x - y + 2*z = 6
def plane2 (x y z : ℝ) : Prop := x + 3*y - z = 2

-- Define the intersection line L (implicitly)
def L (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the plane Q
def Q (x y z : ℝ) : Prop := x - 7*y + 4*z = 2

-- Define the distance function from a point to a plane
noncomputable def distance_to_plane (a b c d : ℝ) (x y z : ℝ) : ℝ :=
  abs (a*x + b*y + c*z + d) / Real.sqrt (a^2 + b^2 + c^2)

-- Theorem statement
theorem plane_Q_satisfies_conditions :
  (∀ x y z, L x y z → Q x y z) ∧
  (distance_to_plane 1 (-7) 4 (-2) 1 (-1) 2 = 3 / Real.sqrt 2) ∧
  (distance_to_plane 1 (-7) 4 (-2) 0 1 0 = 1 / Real.sqrt 6) := by
  sorry

#check plane_Q_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_Q_satisfies_conditions_l792_79206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_point_coordinates_l792_79289

/-- The point of reflection on a plane given incident and reflected ray points -/
def reflection_point (A B C : ℝ × ℝ × ℝ) (plane_normal : ℝ × ℝ × ℝ) (plane_constant : ℝ) : Prop :=
  let D := (2 * (A.1 + A.2.1 + A.2.2 - plane_constant) / (plane_normal.1 + plane_normal.2.1 + plane_normal.2.2)) • plane_normal
  ∃ t : ℝ, B = (C.1 + t * (D.1 - C.1), C.2.1 + t * (D.2.1 - C.2.1), C.2.2 + t * (D.2.2 - C.2.2)) ∧ 
            B.1 + B.2.1 + B.2.2 = plane_constant

theorem reflection_point_coordinates : 
  let A : ℝ × ℝ × ℝ := (2, 4, 5)
  let C : ℝ × ℝ × ℝ := (1, 1, 3)
  let plane_normal : ℝ × ℝ × ℝ := (1, 1, 1)
  let plane_constant : ℝ := 10
  let B : ℝ × ℝ × ℝ := (13/8, 23/8, 11/2)
  reflection_point A B C plane_normal plane_constant := by
  sorry

#check reflection_point_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_point_coordinates_l792_79289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_sightings_ratio_l792_79228

/-- Animal sightings in a national park over three months -/
theorem animal_sightings_ratio (january_sightings : ℕ) 
  (february_multiplier : ℕ) (total_sightings : ℕ) :
  january_sightings = 26 →
  february_multiplier = 3 →
  total_sightings = 143 →
  ∃ (march_ratio : ℚ),
    march_ratio = 1 / 2 ∧
    total_sightings = january_sightings + 
      february_multiplier * january_sightings + 
      (march_ratio * (february_multiplier * january_sightings : ℚ)).floor :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_sightings_ratio_l792_79228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_statements_l792_79235

/-- Represents the type of resident: Knight (truth-teller) or Liar --/
inductive ResidentType
  | Knight
  | Liar

/-- Represents the statement a resident can make about their partner --/
inductive Statement
  | IsKnight
  | IsLiar

/-- The total number of residents on the island --/
def totalResidents : Nat := 1234

/-- The number of pairs formed by the residents --/
def numPairs : Nat := 617

/-- A function that determines what statement a resident makes about their partner --/
def makeStatement (speaker : ResidentType) (partner : ResidentType) : Statement :=
  match speaker, partner with
  | ResidentType.Knight, ResidentType.Knight => Statement.IsKnight
  | ResidentType.Knight, ResidentType.Liar => Statement.IsLiar
  | ResidentType.Liar, ResidentType.Knight => Statement.IsKnight
  | ResidentType.Liar, ResidentType.Liar => Statement.IsKnight

/-- Helper function to make ResidentType decidable --/
instance : DecidableEq ResidentType :=
  fun a b => match a, b with
  | ResidentType.Knight, ResidentType.Knight => isTrue rfl
  | ResidentType.Liar, ResidentType.Liar => isTrue rfl
  | ResidentType.Knight, ResidentType.Liar => isFalse (fun h => ResidentType.noConfusion h)
  | ResidentType.Liar, ResidentType.Knight => isFalse (fun h => ResidentType.noConfusion h)

/-- Helper function to make Statement decidable --/
instance : DecidableEq Statement :=
  fun a b => match a, b with
  | Statement.IsKnight, Statement.IsKnight => isTrue rfl
  | Statement.IsLiar, Statement.IsLiar => isTrue rfl
  | Statement.IsKnight, Statement.IsLiar => isFalse (fun h => Statement.noConfusion h)
  | Statement.IsLiar, Statement.IsKnight => isFalse (fun h => Statement.noConfusion h)

theorem unequal_statements :
  ¬∃ (residents : Fin totalResidents → ResidentType),
    ∃ (pairing : Fin numPairs → Fin totalResidents × Fin totalResidents),
      (∀ i : Fin numPairs, (pairing i).1 ≠ (pairing i).2) ∧
      (∀ r : Fin totalResidents, ∃! i : Fin numPairs, r = (pairing i).1 ∨ r = (pairing i).2) ∧
      (2 * (Finset.univ.filter (fun i => makeStatement (residents ((pairing i).1)) (residents ((pairing i).2)) = Statement.IsKnight)).card
        = totalResidents) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_statements_l792_79235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_problem_l792_79259

/-- The height of the building given the conditions -/
noncomputable def building_height (tree_height : ℝ) (tree_shadow : ℝ) (building_shadow : ℝ) : ℝ :=
  (building_shadow * tree_height) / tree_shadow

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem building_height_problem (tree_height : ℝ) (tree_shadow : ℝ) (building_shadow : ℝ)
  (h1 : tree_height = 30)
  (h2 : tree_shadow = 36)
  (h3 : building_shadow = 72) :
  round_to_nearest (building_height tree_height tree_shadow building_shadow) = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_problem_l792_79259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l792_79258

theorem cubic_equation_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ > 0 ∧ r₂ < 0 ∧ r₃ < 0 ∧
  ∀ x : ℝ, x^3 + 9*x^2 + 26*x + 24 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l792_79258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l792_79284

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p2 = p1 + t • (p3 - p1) ∨ p3 = p1 + t • (p2 - p1)

/-- Given three collinear points with specific coordinates, prove that a + b = 4. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, a, b) (a, 2, b) (a, b, 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l792_79284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_bounds_l792_79267

/-- Definition of M as a function of x, y, and z -/
noncomputable def M (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + x*y + y^2) * Real.sqrt (y^2 + y*z + z^2) +
  Real.sqrt (y^2 + y*z + z^2) * Real.sqrt (z^2 + z*x + x^2) +
  Real.sqrt (z^2 + z*x + x^2) * Real.sqrt (x^2 + x*y + y^2)

/-- The theorem stating the maximum value of α and minimum value of β -/
theorem alpha_beta_bounds :
  (∃ α β : ℝ, ∀ x y z : ℝ,
    α * (x*y + y*z + z*x) ≤ M x y z ∧
    M x y z ≤ β * (x^2 + y^2 + z^2)) →
  (∃ α_max β_min : ℝ,
    (∀ α' : ℝ, (∀ x y z : ℝ, α' * (x*y + y*z + z*x) ≤ M x y z) → α' ≤ α_max) ∧
    (∀ β' : ℝ, (∀ x y z : ℝ, M x y z ≤ β' * (x^2 + y^2 + z^2)) → β_min ≤ β') ∧
    α_max = 3 ∧ β_min = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_bounds_l792_79267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_jump_theorem_l792_79270

/-- Represents a jump of the flea -/
structure Jump where
  start : ℤ
  finish : ℤ

/-- Checks if a jump is valid according to the rules -/
def is_valid_jump (j : Jump) : Prop :=
  j.finish = j.start + (j.start - j.start) - 1 ∨
  j.finish = j.start + (j.start - j.start) ∨
  j.finish = j.start + (j.start - j.start) + 1

/-- A sequence of jumps starting from 0 to 1, then following the rules -/
def valid_jump_sequence : List Jump → Prop
  | [] => False
  | [j] => j.start = 0 ∧ j.finish = 1
  | j₁ :: j₂ :: js => j₁.start = 0 ∧ j₁.finish = 1 ∧ 
                      is_valid_jump j₂ ∧ 
                      valid_jump_sequence (j₂ :: js)

/-- The sequence contains point n at least twice -/
def contains_twice (n : ℤ) (js : List Jump) : Prop :=
  (js.filter (fun j => j.finish = n)).length ≥ 2

/-- Main theorem: If a valid jump sequence contains n twice, its length is at least ⌈2√n⌉ -/
theorem flea_jump_theorem (n : ℕ) (js : List Jump) 
  (h₁ : valid_jump_sequence js) 
  (h₂ : contains_twice (n : ℤ) js) : 
  js.length ≥ Int.ceil (2 * Real.sqrt n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_jump_theorem_l792_79270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_range_of_b_l792_79216

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

-- Theorem for the maximum value of f in [0, π/4]
theorem max_value_f : 
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 4) ∧ 
  f x = Real.sqrt 2 ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 4) → f y ≤ Real.sqrt 2 :=
sorry

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = Real.pi)
  (pos_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (pos_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (sine_law : a / Real.sin A = b / Real.sin B)

-- Theorem for the range of side b
theorem range_of_b (t : Triangle) 
  (h1 : f (3 * t.B / 4) = 1) 
  (h2 : t.a + t.c = 2) : 
  t.b ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_range_of_b_l792_79216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_non_intersecting_lines_l792_79287

/-- Two planes in 3D space -/
structure Plane3D where
  -- Define a plane (implementation details omitted)
  dummy : Unit

/-- A line in 3D space -/
structure Line3D where
  -- Define a line (implementation details omitted)
  dummy : Unit

/-- Two planes are parallel -/
def parallel_planes (α β : Plane3D) : Prop :=
  -- Definition of parallel planes
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line being contained in a plane
  sorry

/-- Two lines are non-intersecting -/
def non_intersecting (a b : Line3D) : Prop :=
  -- Definition of non-intersecting lines
  sorry

/-- Theorem: If two planes are parallel and each contains a line, 
    then these lines are non-intersecting -/
theorem parallel_planes_non_intersecting_lines 
  (α β : Plane3D) (a b : Line3D) 
  (h1 : parallel_planes α β) 
  (h2 : line_in_plane a α) 
  (h3 : line_in_plane b β) : 
  non_intersecting a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_non_intersecting_lines_l792_79287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_dog_weight_proof_l792_79288

/-- The weight of Evan's dog in pounds -/
noncomputable def evans_dog_weight : ℝ := 63

/-- The weight of Ivan's dog in pounds -/
noncomputable def ivans_dog_weight : ℝ := evans_dog_weight / 7

/-- The total weight of both dogs in pounds -/
noncomputable def total_weight : ℝ := 72

theorem evans_dog_weight_proof :
  (evans_dog_weight = 7 * ivans_dog_weight) ∧
  (evans_dog_weight + ivans_dog_weight = total_weight) →
  evans_dog_weight = 63 := by
  sorry

#check evans_dog_weight_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_dog_weight_proof_l792_79288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_possible_range_l792_79227

/-- Represents the score range for a person -/
structure ScoreRange where
  lower : ℕ
  upper : ℕ
  valid : lower ≤ upper

/-- The problem setup -/
def problem_setup : List ScoreRange :=
  [⟨17, 31, by norm_num⟩, ⟨28, 47, by norm_num⟩, ⟨35, 60, by norm_num⟩,
   ⟨45, 75, by norm_num⟩, ⟨52, 89, by norm_num⟩]

/-- The minimum score from all ranges -/
def min_score (ranges : List ScoreRange) : ℕ :=
  ranges.map (fun x => x.lower) |>.minimum?.getD 0

/-- The maximum score from all ranges -/
def max_score (ranges : List ScoreRange) : ℕ :=
  ranges.map (fun x => x.upper) |>.maximum?.getD 0

/-- The theorem stating the minimum possible range -/
theorem min_possible_range :
  max_score problem_setup - min_score problem_setup = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_possible_range_l792_79227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l792_79240

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x else -x^2

-- Define g as a variable function
variable (g : ℝ → ℝ)

-- Define the properties of function g
axiom g_odd : ∀ x : ℝ, g (-x) = -(g x)
axiom g_neg : ∀ x : ℝ, x < 0 → g x = x^2 - 2*x - 5

-- Define the main theorem
theorem range_of_a (a : ℝ) : 
  f (g a) ≤ 2 ↔ a ∈ Set.Iic (-1) ∪ Set.Icc 0 (2 * Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l792_79240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_three_necessary_not_sufficient_l792_79264

theorem x_squared_three_necessary_not_sufficient :
  (∃ x : ℝ, x^2 = 3 ∧ x ≠ Real.sqrt 3) ∧
  (∀ x : ℝ, x = Real.sqrt 3 → x^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_three_necessary_not_sufficient_l792_79264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_set_characterization_l792_79233

/-- A set of integers with specific closure properties -/
def IntegerSet (S : Set Int) : Prop :=
  (∀ x, x ∈ S → (2 * x) ∈ S) ∧
  (∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S) ∧
  (∃ p n, p ∈ S ∧ n ∈ S ∧ p > 0 ∧ n < 0)

/-- The main theorem -/
theorem integer_set_characterization (S : Set Int) (hS : IntegerSet S) :
  ∃ a : Int, a > 0 ∧ S = {k : Int | ∃ m : Int, k = m * a} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_set_characterization_l792_79233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_square_l792_79247

/-- The side length of the square -/
noncomputable def square_side : ℝ := 40

/-- The area of the square -/
noncomputable def square_area : ℝ := square_side ^ 2

/-- The side length of each equilateral triangle -/
noncomputable def triangle_side : ℝ := (2 * square_side / 2) * (2 / Real.sqrt 3)

/-- The area of one equilateral triangle -/
noncomputable def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2

/-- The total area of all four triangles -/
noncomputable def total_triangle_area : ℝ := 4 * triangle_area

/-- The theorem stating the shaded area in the square -/
theorem shaded_area_in_square :
  square_area - total_triangle_area = 1600 - (1600 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_square_l792_79247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_less_than_20_degrees_l792_79209

/-- A type representing a line in a plane --/
def Line : Type := Unit

/-- A type representing a point in a plane --/
def Point : Type := Unit

/-- A function that checks if a line passes through a point --/
def passes_through (l : Line) (p : Point) : Prop := sorry

/-- A function that calculates the angle between two lines --/
noncomputable def angle_between (l1 l2 : Line) : ℝ := sorry

/-- Theorem: Given 10 lines passing through a single point on a plane,
    the minimum angle formed between any two of these lines is less than 20 degrees --/
theorem min_angle_less_than_20_degrees 
  (lines : Finset Line) 
  (p : Point) 
  (h1 : lines.card = 10) 
  (h2 : ∀ l, l ∈ lines → passes_through l p) : 
  ∃ l1 l2, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ angle_between l1 l2 < 20 * π / 180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_less_than_20_degrees_l792_79209
