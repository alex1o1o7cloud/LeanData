import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_approx_l1339_133989

/-- Represents the increase in wheel radius given initial conditions --/
noncomputable def wheel_radius_increase (original_radius : ℝ) (outward_distance : ℝ) (return_distance : ℝ) : ℝ :=
  let new_radius := original_radius * (outward_distance / return_distance)
  new_radius - original_radius

/-- Theorem stating the increase in wheel radius given specific conditions --/
theorem wheel_radius_increase_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |wheel_radius_increase 15 450 440 - 0.34| < ε := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_approx_l1339_133989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_with_stoppages_l1339_133958

/-- Given a bus with speed excluding stoppages and stopping time per hour,
    calculate its speed including stoppages -/
theorem bus_speed_with_stoppages
  (speed_without_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_without_stoppages = 64)
  (h2 : stopping_time = 15)
  (h3 : stopping_time ≥ 0)
  (h4 : stopping_time < 60) :
  let running_time := 60 - stopping_time
  let distance := speed_without_stoppages * running_time / 60
  distance = 48 := by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_with_stoppages_l1339_133958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_altitudes_relation_l1339_133962

noncomputable def Triangle := ℝ → ℝ → ℝ → Type

noncomputable def Triangle.altitude (t : Triangle) (v : ℝ) : ℝ := sorry
noncomputable def Triangle.angleBisector (t : Triangle) (v : ℝ) : ℝ := sorry
noncomputable def Triangle.angle (t : Triangle) (v : ℝ) : ℝ := sorry

theorem angle_bisector_altitudes_relation (ABC : Triangle) 
  (h_a h_b l : ℝ) (h_a_pos : h_a > 0) (h_b_pos : h_b > 0) (l_pos : l > 0) :
  (h_a = ABC.altitude 0) → 
  (h_b = ABC.altitude 1) → 
  (l = ABC.angleBisector 2) →
  Real.sin (ABC.angle 2 / 2) = (h_a * h_b) / (l * (h_a + h_b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_altitudes_relation_l1339_133962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_2theta_l1339_133924

theorem cos_pi_half_minus_2theta (θ : ℝ) 
  (h : Real.cos θ + Real.sin θ = -Real.sqrt 5 / 3) : 
  Real.cos (π / 2 - 2 * θ) = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_minus_2theta_l1339_133924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_field_time_l1339_133984

/-- The time taken to run around a square field -/
noncomputable def time_to_run_around_field (side_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let perimeter := 4 * side_length
  let speed_ms := speed_kmh * 1000 / 3600
  perimeter / speed_ms

/-- Theorem stating that running around a square field of side 50 meters at 10 km/h takes approximately 72 seconds -/
theorem run_around_field_time :
  ∃ ε > 0, |time_to_run_around_field 50 10 - 72| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_field_time_l1339_133984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_I_equals_2H_l1339_133965

-- Define H as a noncomputable function of x
noncomputable def H (x : ℝ) : ℝ := Real.log ((2 + x) / (2 - x))

-- Define I as a noncomputable function of x
noncomputable def I (x : ℝ) : ℝ := Real.log ((2 + (4 * x) / (1 + x^2)) / (2 - (4 * x) / (1 + x^2)))

-- Theorem statement
theorem I_equals_2H (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 1 ∧ x ≠ -1) : I x = 2 * H x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_I_equals_2H_l1339_133965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_result_is_correct_l1339_133917

def result : Real := 4.4

theorem result_is_correct : result = 4.4 := by
  -- The proof is trivial since we defined result as 4.4
  rfl

#eval result -- This will output 4.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_result_is_correct_l1339_133917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l1339_133955

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define the base case for 0
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) / (1 + 2 * sequence_a (n + 1))

theorem tenth_term_value : sequence_a 10 = 1 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l1339_133955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1339_133994

/-- Given a train of length 1200 m that crosses a tree in 120 sec and passes a platform in 240 sec,
    the length of the platform is 1200 m. -/
theorem platform_length
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_passing_time : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_passing_time = 240) :
  let train_speed := train_length / tree_crossing_time
  let platform_length := train_speed * platform_passing_time - train_length
  platform_length = 1200 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1339_133994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_focus_l1339_133970

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1

-- Define the left focus F1
def F1 : ℝ × ℝ := (-3, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_to_focus (P : ℝ × ℝ) :
  C P.1 P.2 → distance P F1 ≤ 7 ∧ ∃ Q : ℝ × ℝ, C Q.1 Q.2 ∧ distance Q F1 = 7 := by
  sorry

#check max_distance_to_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_focus_l1339_133970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_in_class_l1339_133900

theorem number_of_boys_in_class (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 12)
  (h2 : (5 : ℚ) / 6 * girls + (4 : ℚ) / 5 * boys = (girls + boys - 4 : ℕ)) :
  boys = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_in_class_l1339_133900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_configuration_fits_minimum_cube_size_configuration_possible_l1339_133949

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Check if two spheres are tangent -/
def are_tangent (s1 s2 : Sphere) : Prop :=
  distance s1.center s2.center = s1.radius + s2.radius

/-- Check if a sphere is tangent to three faces of a cube -/
def is_tangent_to_faces (s : Sphere) (cube_size : ℝ) : Prop :=
  (s.center.x = s.radius ∨ s.center.x = cube_size - s.radius) ∧
  (s.center.y = s.radius ∨ s.center.y = cube_size - s.radius) ∧
  (s.center.z = s.radius ∨ s.center.z = cube_size - s.radius)

theorem sphere_configuration_fits (cube_size : ℝ) : Prop :=
  let large_sphere : Sphere := { center := { x := cube_size/2, y := cube_size/2, z := cube_size/2 }, radius := 3 }
  let small_spheres : List Sphere := [
    { center := { x := 1, y := 1, z := 1 }, radius := 1 },
    { center := { x := 1, y := 1, z := 5 }, radius := 1 },
    { center := { x := 1, y := 5, z := 1 }, radius := 1 },
    { center := { x := 1, y := 5, z := 5 }, radius := 1 },
    { center := { x := 5, y := 1, z := 1 }, radius := 1 },
    { center := { x := 5, y := 1, z := 5 }, radius := 1 },
    { center := { x := 5, y := 5, z := 1 }, radius := 1 },
    { center := { x := 5, y := 5, z := 5 }, radius := 1 }
  ]
  cube_size = 6 →
  (∀ s ∈ small_spheres, is_tangent_to_faces s cube_size) ∧
  (∀ s ∈ small_spheres, are_tangent s large_sphere)

theorem minimum_cube_size : ℝ := 6

theorem configuration_possible : sphere_configuration_fits minimum_cube_size := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_configuration_fits_minimum_cube_size_configuration_possible_l1339_133949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_evaluation_l1339_133941

-- Define the complex number magnitude
noncomputable def complex_magnitude (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- State the theorem
theorem complex_sum_evaluation : 
  3 * complex_magnitude 1 (-3) + 2 * complex_magnitude 1 3 = 5 * Real.sqrt 10 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_evaluation_l1339_133941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1339_133999

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

def is_solution (x y : ℝ) : Prop :=
  x - 2*y + (integer_part x : ℝ) + 3*fractional_part x - fractional_part y + 3*(integer_part y : ℝ) = 2.5 ∧
  2*x + y - (integer_part x : ℝ) + 2*fractional_part x + 3*fractional_part y - 4*(integer_part y : ℝ) = 12

theorem system_solutions :
  (∀ x y : ℝ, is_solution x y →
    (x = 53/28 ∧ y = -23/14) ∨
    (x = 5/2 ∧ y = -3/2) ∨
    (x = 87/28 ∧ y = -19/14)) ∧
  is_solution (53/28) (-23/14) ∧
  is_solution (5/2) (-3/2) ∧
  is_solution (87/28) (-19/14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1339_133999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_sum_l1339_133938

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Definition of the parabola y^2 = 8x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = 8 * p.x

theorem minimal_distance_sum :
  let A : Point := ⟨2, 0⟩
  let B : Point := ⟨7, 6⟩
  ∀ P : Point, onParabola P →
    distance A P + distance B P ≥ 9 ∧
    ∃ P : Point, onParabola P ∧ distance A P + distance B P = 9 :=
by
  sorry

#check minimal_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_sum_l1339_133938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OQ_OD_l1339_133974

-- Define the points in the Cartesian coordinate system
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (3, 0)
def O : ℝ × ℝ := (0, 0)

-- Define vector OQ
def OQ (m : ℝ) : ℝ × ℝ := (m * A.1 + (1 - m) * B.1, m * A.2 + (1 - m) * B.2)

-- Define the constraint |CD| = 1
def CD_constraint (D : ℝ × ℝ) : Prop := (D.1 - C.1)^2 + (D.2 - C.2)^2 = 1

-- Define the distance between OQ and OD
noncomputable def distance_OQ_OD (m : ℝ) (D : ℝ × ℝ) : ℝ :=
  ((OQ m).1 - D.1)^2 + ((OQ m).2 - D.2)^2

-- Theorem statement
theorem min_distance_OQ_OD :
  ∃ (min_dist : ℝ), min_dist = (3 * Real.sqrt 2 - 1)^2 ∧
  ∀ (m : ℝ) (D : ℝ × ℝ), CD_constraint D →
  distance_OQ_OD m D ≥ min_dist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OQ_OD_l1339_133974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trick_or_treating_count_l1339_133916

def village_blocks : ℕ := 9
def children_odd_block : ℕ := 7
def children_even_block : ℕ := 5
def participation_rate : ℚ := 7/10

def trick_or_treating_children : ℕ :=
  (participation_rate * (((village_blocks + 1) / 2 * children_odd_block) +
    (village_blocks / 2 * children_even_block))).floor.toNat

theorem trick_or_treating_count :
  trick_or_treating_children = 38 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trick_or_treating_count_l1339_133916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_periodic_l1339_133903

noncomputable def b : ℕ → ℝ
  | 0 => Real.sqrt 2 - 1
  | n + 1 => if b n > 1 then b n - 1 else 1 / b n

theorem b_periodic : ∀ n : ℕ, b (n + 5) = b n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_periodic_l1339_133903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_not_in_second_quadrant_l1339_133933

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def inSecondQuadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The line (3a-1)x + (2-a)y - 1 = 0 does not pass through the second quadrant -/
def lineNotInSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3*a - 1)*x + (2 - a)*y - 1 = 0 → ¬(inSecondQuadrant x y)

/-- The range of values for a to ensure the line does not pass through the second quadrant -/
theorem line_not_in_second_quadrant (a : ℝ) :
  lineNotInSecondQuadrant a ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_not_in_second_quadrant_l1339_133933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l1339_133925

/-- A function that generates all 3-digit numbers from a list of 5 digits -/
def generateThreeDigitNumbers (digits : List Nat) : List Nat :=
  sorry

/-- A function that checks if a number is composed of 5 different non-zero digits -/
def isValidFiveDigitNumber (n : Nat) : Bool :=
  sorry

/-- A function to get the digits of a natural number -/
def getDigits (n : Nat) : List Nat :=
  n.repr.data.map (fun c => c.toNat - '0'.toNat)

/-- The main theorem -/
theorem unique_five_digit_number :
  ∃! n : Nat,
    isValidFiveDigitNumber n ∧
    n = (generateThreeDigitNumbers (getDigits n)).sum ∧
    n = 35964 := by
  sorry

#eval getDigits 35964  -- To test the getDigits function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l1339_133925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_meals_sold_l1339_133987

theorem kids_meals_sold (ratio_kids ratio_adults adult_meals : ℕ) 
  (h1 : ratio_kids = 10)
  (h2 : ratio_adults = 7)
  (h3 : adult_meals = 49) : 
  (ratio_kids * adult_meals) / ratio_adults = 70 := by
  -- Proof
  sorry

#eval (10 * 49) / 7  -- This will evaluate to 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_meals_sold_l1339_133987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_volume_after_drilling_l1339_133904

noncomputable section

-- Define the bowling ball parameters
def ball_diameter : ℝ := 24
def ball_radius : ℝ := ball_diameter / 2

-- Define the hole parameters
def hole_depth : ℝ := 5
def small_hole_diameter : ℝ := 2.5
def large_hole_diameter : ℝ := 4

-- Define the number of each type of hole
def num_small_holes : ℝ := 2
def num_large_holes : ℝ := 1

-- Theorem statement
theorem bowling_ball_volume_after_drilling :
  let ball_volume := (4 / 3) * Real.pi * (ball_radius ^ 3)
  let small_hole_volume := Real.pi * ((small_hole_diameter / 2) ^ 2) * hole_depth
  let large_hole_volume := Real.pi * ((large_hole_diameter / 2) ^ 2) * hole_depth
  let total_hole_volume := num_small_holes * small_hole_volume + num_large_holes * large_hole_volume
  ball_volume - total_hole_volume = 2268.375 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_volume_after_drilling_l1339_133904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rally_n_must_be_odd_odd_n_implies_valid_rally_l1339_133906

/-- A structure representing a car in the rally -/
structure Car where
  overtakes : Finset Nat
  overtaken_by : Finset Nat

/-- A structure representing the rally -/
structure Rally where
  n : Nat
  cars : Finset Car
  h_n_ge_2 : n ≥ 2
  h_card_cars : cars.card = n
  h_overtakes_at_most_once : ∀ c, c ∈ cars → c.overtakes.card ≤ 1
  h_different_overtakes : ∀ c1 c2, c1 ∈ cars → c2 ∈ cars → c1 ≠ c2 → c1.overtakes.card ≠ c2.overtakes.card
  h_same_overtaken : ∀ c1 c2, c1 ∈ cars → c2 ∈ cars → c1.overtaken_by.card = c2.overtaken_by.card

/-- The main theorem stating that n must be odd for a valid rally -/
theorem rally_n_must_be_odd (r : Rally) : Odd r.n := by
  sorry

/-- Proof that if n is odd, a valid rally configuration exists -/
theorem odd_n_implies_valid_rally (n : Nat) (h : Odd n) : ∃ r : Rally, r.n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rally_n_must_be_odd_odd_n_implies_valid_rally_l1339_133906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_translation_l1339_133945

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x + Real.pi/3)

noncomputable def translated_function (x : ℝ) : ℝ := Real.sin (x + Real.pi/6)

noncomputable def translation_amount : ℝ := Real.pi/6

theorem sinusoidal_translation :
  ∀ x : ℝ, original_function (x - translation_amount) = translated_function x := by
  intro x
  simp [original_function, translated_function, translation_amount]
  -- The proof goes here
  sorry

#check sinusoidal_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_translation_l1339_133945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1339_133992

/-- The focus of the parabola y^2 = 8x has coordinates (2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  y^2 = 8*x → (2, 0) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1339_133992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l1339_133963

theorem smallest_x_value (x y : ℕ+) (h : (83 : ℚ) / 100 = (y : ℚ) / (192 + x)) : 
  x ≥ 8 ∧ ∃ (y' : ℕ+), (83 : ℚ) / 100 = (y' : ℚ) / (192 + 8) := by
  sorry

#check smallest_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l1339_133963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1339_133918

theorem inequality_proof (n : ℕ) (x : ℝ) (hn : 0 < n) (hx : 0 < x) :
  (x^(2*n-1) - 1) / (2*n-1 : ℝ) ≤ (x^(2*n) - 1) / (2*n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1339_133918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l1339_133950

def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | (x - 1)^2 ≤ 4}

theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l1339_133950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandchildren_ages_l1339_133960

theorem grandchildren_ages (ages : List ℕ) : 
  ages.length = 7 ∧ 
  (∀ i, i ∈ Finset.range 6 → ages[i.succ]! = ages[i]! + 1) ∧
  (ages[0]! + ages[1]! + ages[2]!) / 3 = 6 →
  (ages[4]! + ages[5]! + ages[6]!) / 3 = 10 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandchildren_ages_l1339_133960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1339_133942

noncomputable section

-- Define the hyperbola and its properties
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1 ∧ a > 0

-- Define the focus of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 5 ∧ y = 0

-- Define the eccentricity of a hyperbola
def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Theorem statement
theorem hyperbola_eccentricity (a : ℝ) :
  (∃ x y, hyperbola a x y) →
  (∃ x y, focus x y) →
  eccentricity a 5 = 5/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1339_133942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1339_133946

-- Define the function f
noncomputable def f (x : ℝ) := Real.sqrt (3 - Real.sqrt (5 - Real.sqrt x))

-- Define the domain of f
def domain (f : ℝ → ℝ) := {x : ℝ | ∃ y, f x = y}

-- Theorem statement
theorem domain_of_f :
  domain f = {x : ℝ | 0 ≤ x ∧ x ≤ 25} := by
  sorry

-- Additional helper lemmas if needed
lemma sqrt_nonneg (x : ℝ) : 0 ≤ x → 0 ≤ Real.sqrt x := by
  sorry

lemma sqrt_le_sqrt (x y : ℝ) : 0 ≤ x → x ≤ y → Real.sqrt x ≤ Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1339_133946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_theorem_r33_l1339_133991

theorem ramsey_theorem_r33 (students : Finset Nat) (acquainted : Nat → Nat → Prop) :
  students.card = 6 →
  (∀ a b, a ∈ students → b ∈ students → a ≠ b → (acquainted a b ∨ ¬acquainted a b)) →
  (∃ a b c, a ∈ students ∧ b ∈ students ∧ c ∈ students ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ acquainted a b ∧ acquainted b c ∧ acquainted a c) ∨
  (∃ a b c, a ∈ students ∧ b ∈ students ∧ c ∈ students ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬acquainted a b ∧ ¬acquainted b c ∧ ¬acquainted a c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_theorem_r33_l1339_133991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_list_price_is_40_l1339_133935

-- Define the list price
variable (list_price : ℝ)

-- Define Alice's selling price and commission
def alice_selling_price (list_price : ℝ) : ℝ := list_price - 15
def alice_commission (list_price : ℝ) : ℝ := 0.15 * (alice_selling_price list_price)

-- Define Bob's selling price and commission
def bob_selling_price (list_price : ℝ) : ℝ := list_price - 25
def bob_commission (list_price : ℝ) : ℝ := 0.25 * (bob_selling_price list_price)

-- Theorem: The list price is $40
theorem list_price_is_40 :
  alice_commission list_price = bob_commission list_price → list_price = 40 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval alice_commission 40
#eval bob_commission 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_list_price_is_40_l1339_133935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1339_133915

/-- Parabola type -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 8*x

/-- Line type -/
structure Line where
  k : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = k * (x - 2)

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector type -/
structure Vec where
  x : ℝ
  y : ℝ

def dot_product (v1 v2 : Vec) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Focus of the parabola -/
def focus (p : Parabola) : Point := ⟨2, 0⟩

/-- Theorem statement -/
theorem parabola_line_intersection_slope (p : Parabola) (l : Line) (m : Point)
  (h1 : p.eq = fun x y => y^2 = 8*x)
  (h2 : m.x = -2 ∧ m.y = 2)
  (h3 : l.eq (focus p).x (focus p).y)
  (h4 : ∃ (a b : Point), p.eq a.x a.y ∧ p.eq b.x b.y ∧ l.eq a.x a.y ∧ l.eq b.x b.y)
  (h5 : ∀ (a b : Point), p.eq a.x a.y → p.eq b.x b.y → l.eq a.x a.y → l.eq b.x b.y →
        dot_product ⟨a.x - m.x, a.y - m.y⟩ ⟨b.x - m.x, b.y - m.y⟩ = 0) :
  l.k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l1339_133915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1339_133926

theorem power_inequality (a : ℝ) (n : ℤ) (hn : n ≥ 1) :
  a^n + a^(-n) - 2 ≥ n^2 * (a + a⁻¹ - 2) ∧
  (a^n + a^(-n) - 2 = n^2 * (a + a⁻¹ - 2) ↔ a = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1339_133926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1339_133947

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define our equation
def equation (a : ℝ) : Prop :=
  (floor (a / 2) : ℝ) + (floor (a / 3) : ℝ) + (floor (a / 5) : ℝ) = a

-- Theorem statement
theorem unique_solution :
  ∃! a : ℝ, equation a ∧ a = 0 := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1339_133947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l1339_133920

def a : Fin 3 → ℝ := ![5, -3, 2]
def b : Fin 3 → ℝ := ![-2, 4, -3]

theorem vector_subtraction : (a - 4 • b) = ![13, -19, 14] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l1339_133920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_planes_proof_l1339_133910

noncomputable def plane1 (x y z : ℝ) : Prop := x + 3*y - 2*z + 4 = 0

noncomputable def plane2 (x y z : ℝ) : Prop := 2*x + 6*y - 4*z + 7 = 0

noncomputable def distance_between_planes : ℝ := 15 * Real.sqrt 14 / 28

theorem distance_planes_proof : 
  ∃ (d : ℝ), d = distance_between_planes ∧ 
  ∀ (p q : ℝ × ℝ × ℝ), 
    plane1 p.1 p.2.1 p.2.2 → 
    plane2 q.1 q.2.1 q.2.2 → 
    d = Real.sqrt ((p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2) :=
by
  sorry

#check distance_planes_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_planes_proof_l1339_133910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1339_133936

/-- An equilateral triangle with side length 2 -/
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)
  (side_length : ℝ)
  (is_equilateral : side_length = 2)

/-- A point on the incircle of the triangle -/
def IncirclePoint (t : EquilateralTriangle) := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of squared distances from a point to triangle vertices -/
noncomputable def sum_squared_distances (t : EquilateralTriangle) (p : IncirclePoint t) : ℝ :=
  (distance p t.A)^2 + (distance p t.B)^2 + (distance p t.C)^2

/-- Area of a triangle given side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equilateral_triangle_properties (t : EquilateralTriangle) (p : IncirclePoint t) :
  (sum_squared_distances t p = 5) ∧
  (triangle_area (distance p t.A) (distance p t.B) (distance p t.C) = Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1339_133936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l1339_133982

noncomputable section

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 / 4 = 1

/-- The line equation passing through the right focus with slope 2 -/
def is_on_line (x y : ℝ) : Prop :=
  y = 2 * x - 2

/-- Point A is on both the ellipse and the line -/
def point_A : ℝ × ℝ :=
  (0, -2)

/-- Point B is on both the ellipse and the line -/
def point_B : ℝ × ℝ :=
  (5/3, 4/3)

/-- O is the coordinate origin -/
def point_O : ℝ × ℝ :=
  (0, 0)

/-- The area of triangle AOB -/
def triangle_area (A B O : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := O
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃))

theorem area_of_triangle_AOB :
  is_on_ellipse point_A.1 point_A.2 ∧
  is_on_ellipse point_B.1 point_B.2 ∧
  is_on_line point_A.1 point_A.2 ∧
  is_on_line point_B.1 point_B.2 →
  triangle_area point_A point_B point_O = 25/9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l1339_133982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harkamal_fruit_purchase_cost_l1339_133919

/-- Calculates the discounted price given the original price and discount rate -/
def apply_discount (price : ℚ) (discount_rate : ℚ) : ℚ :=
  price * (1 - discount_rate)

/-- Calculates the total cost after discounts for Harkamal's fruit purchase -/
def total_cost_after_discounts (
  grapes_kg : ℚ)
  (grapes_price_per_kg : ℚ)
  (mangoes_kg : ℚ)
  (mangoes_price_per_kg : ℚ)
  (apples_kg : ℚ)
  (apples_price_per_kg : ℚ)
  (oranges_kg : ℚ)
  (oranges_price_per_kg : ℚ) : ℚ :=
  let grapes_cost := grapes_kg * grapes_price_per_kg
  let mangoes_cost := mangoes_kg * mangoes_price_per_kg
  let apples_cost := apples_kg * apples_price_per_kg
  let oranges_cost := oranges_kg * oranges_price_per_kg
  
  let grapes_discounted := 
    if grapes_cost ≥ 400 then apply_discount grapes_cost (1/10) else grapes_cost
  let mangoes_discounted := 
    if mangoes_cost ≥ 450 then apply_discount mangoes_cost (3/20) else mangoes_cost
  let apples_discounted := 
    if apples_cost ≥ 200 then apply_discount apples_cost (1/20) else apples_cost

  grapes_discounted + mangoes_discounted + apples_discounted + oranges_cost

theorem harkamal_fruit_purchase_cost :
  total_cost_after_discounts 8 70 9 60 5 50 2 30 = 1260.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harkamal_fruit_purchase_cost_l1339_133919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l1339_133921

/-- The mass percentage of hydrogen in barium hydroxide octahydrate -/
noncomputable def mass_percentage_H_in_BaOH2_8H2O : ℝ :=
  let molar_mass_Ba : ℝ := 137.327
  let molar_mass_O : ℝ := 15.999
  let molar_mass_H : ℝ := 1.008
  let total_molar_mass : ℝ := molar_mass_Ba + 10 * molar_mass_O + 18 * molar_mass_H
  let mass_H : ℝ := 18 * molar_mass_H
  (mass_H / total_molar_mass) * 100

/-- Theorem stating that the mass percentage of hydrogen in barium hydroxide octahydrate is approximately 5.754% -/
theorem mass_percentage_H_approx :
  |mass_percentage_H_in_BaOH2_8H2O - 5.754| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_approx_l1339_133921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1339_133995

noncomputable def f (x : ℝ) : ℝ := min (2 * x + 2) (min ((1 / 2) * x + 1) (-(3 / 4) * x + 7))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 17 / 5 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1339_133995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_expression_l1339_133988

theorem min_value_exponential_expression (x : ℝ) :
  (16 : ℝ)^x - (4 : ℝ)^x - 6 * (2 : ℝ)^x + 9 ≥ 135/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_expression_l1339_133988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_equals_instantaneous_speed_l1339_133996

/-- A body moving with constant velocity -/
structure ConstantVelocityMotion where
  v : ℝ  -- Velocity
  b : ℝ  -- Initial position

/-- The position of the body at time t -/
noncomputable def position (m : ConstantVelocityMotion) (t : ℝ) : ℝ :=
  m.v * t + m.b

/-- The average speed between two time points -/
noncomputable def averageSpeed (m : ConstantVelocityMotion) (t1 t2 : ℝ) : ℝ :=
  (position m t2 - position m t1) / (t2 - t1)

/-- The instantaneous speed at any time point -/
def instantaneousSpeed (m : ConstantVelocityMotion) : ℝ :=
  m.v

theorem average_equals_instantaneous_speed (m : ConstantVelocityMotion) (t1 t2 : ℝ) 
    (h : t1 ≠ t2) : 
    averageSpeed m t1 t2 = instantaneousSpeed m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_equals_instantaneous_speed_l1339_133996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l1339_133902

/-- A circle is tangent to the parabola y = x^2 at point (3,9) and passes through the point (0,2).
    The center of this circle has coordinates (-141/11, 128/11). -/
theorem circle_center_coordinates :
  ∃ (center : ℝ × ℝ),
    let (a, b) := center
    (∀ (x y : ℝ), y = x^2 → (x - a)^2 + (y - b)^2 = (3 - a)^2 + (9 - b)^2) ∧
    (3 - a)^2 + (9 - b)^2 = (0 - a)^2 + (2 - b)^2 ∧
    a = -141/11 ∧ b = 128/11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinates_l1339_133902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ukuleles_and_violins_sum_l1339_133948

/-- The number of ukuleles in Francis' family -/
def U : ℕ := sorry

/-- The number of violins in Francis' family -/
def V : ℕ := sorry

/-- The total number of strings -/
def total_strings : ℕ := 40

/-- The number of guitars -/
def guitars : ℕ := 4

/-- The number of strings on each ukulele -/
def ukulele_strings : ℕ := 4

/-- The number of strings on each guitar -/
def guitar_strings : ℕ := 6

/-- The number of strings on each violin -/
def violin_strings : ℕ := 4

/-- Theorem stating that the sum of ukuleles and violins is 4 -/
theorem ukuleles_and_violins_sum :
  U * ukulele_strings + guitars * guitar_strings + V * violin_strings = total_strings →
  U + V = 4 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ukuleles_and_violins_sum_l1339_133948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_circle_l1339_133959

/-- The minimum distance between a point on the parabola y^2 = 2x and 
    any point on a circle with center (2,0) and radius 1 is √3 - 1 -/
theorem min_distance_parabola_to_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = 2 * p.1}
  let circle := {q : ℝ × ℝ | (q.1 - 2)^2 + q.2^2 = 1}
  ∀ M ∈ parabola, ∃ N ∈ circle, 
    ∀ N' ∈ circle, Real.sqrt 3 - 1 ≤ dist M N' ∧ 
    dist M N = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_circle_l1339_133959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1339_133944

open Real

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := exp (log x) / x

theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioc 0 1 → x₂ ∈ Set.Ioc 0 1 → f a x₁ ≥ g x₂) →
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1339_133944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tip_angles_is_720_l1339_133968

/-- An 8-pointed star formed by connecting 8 evenly spaced points on a circle -/
structure EightPointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The number of arcs each tip angle cuts through -/
  arcs_per_tip : ℕ
  /-- Assertion that there are 8 points -/
  points_eq_eight : num_points = 8
  /-- Assertion that each tip cuts through 4 arcs -/
  arcs_eq_four : arcs_per_tip = 4

/-- The sum of all tip angles in an 8-pointed star -/
noncomputable def sum_of_tip_angles (star : EightPointedStar) : ℝ :=
  ↑star.num_points * (↑star.arcs_per_tip * (360 / ↑star.num_points) / 2)

/-- Theorem: The sum of all tip angles in an 8-pointed star is 720° -/
theorem sum_of_tip_angles_is_720 (star : EightPointedStar) :
  sum_of_tip_angles star = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tip_angles_is_720_l1339_133968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_one_sufficient_unnecessary_l1339_133939

def vector_a (l : ℝ) : ℝ × ℝ := (l, -2)
def vector_b (l : ℝ) : ℝ × ℝ := (1 + l, 1)

def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem lambda_one_sufficient_unnecessary :
  (∃ l' : ℝ, l' ≠ 1 ∧ perpendicular (vector_a l') (vector_b l')) ∧
  (∀ l : ℝ, l = 1 → perpendicular (vector_a l) (vector_b l)) :=
by
  sorry

#check lambda_one_sufficient_unnecessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_one_sufficient_unnecessary_l1339_133939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_travel_time_l1339_133909

def cartSequence (n : ℕ) : ℕ := 8 + 9 * (n - 1)

def sumCartSequence (n : ℕ) : ℕ := 
  n * (cartSequence 1 + cartSequence n) / 2

theorem cart_travel_time : 
  (∀ k, k < 21 → sumCartSequence k < 1990) ∧ sumCartSequence 21 ≥ 1990 :=
by
  sorry

#eval sumCartSequence 21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_travel_time_l1339_133909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_properties_l1339_133985

/-- A regression model for variables x and y -/
structure RegressionModel where
  x : List ℝ
  y : List ℝ
  residuals : List ℝ
  r_squared : ℝ
  correlation_coeff : ℝ

/-- The sum of squared residuals for a regression model -/
def sum_squared_residuals (model : RegressionModel) : ℝ :=
  (model.residuals.map (λ r => r^2)).sum

/-- Theorem stating properties of regression analysis -/
theorem regression_properties (model : RegressionModel) :
  (∀ r ∈ model.residuals, r = 0 → model.r_squared = 1) ∧
  (∀ model' : RegressionModel, sum_squared_residuals model' < sum_squared_residuals model →
    model'.r_squared > model.r_squared) ∧
  (∀ model' : RegressionModel, model'.r_squared > model.r_squared →
    -- model' has better fit than model
    True) ∧
  (abs model.correlation_coeff > 0.75 →
    -- strong linear correlation between x and y
    True) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_properties_l1339_133985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_is_equidistant_l1339_133908

/-- The point in the xz-plane that is equidistant from three given points -/
noncomputable def equidistant_point : ℝ × ℝ × ℝ := (41/7, 0, -19/14)

/-- The first given point -/
def point1 : ℝ × ℝ × ℝ := (1, 0, 0)

/-- The second given point -/
def point2 : ℝ × ℝ × ℝ := (0, -2, 3)

/-- The third given point -/
def point3 : ℝ × ℝ × ℝ := (4, 2, -2)

/-- Function to calculate the squared distance between two points in 3D space -/
def squared_distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2

/-- Theorem stating that the equidistant_point is indeed equidistant from the three given points -/
theorem equidistant_point_is_equidistant :
  squared_distance equidistant_point point1 = squared_distance equidistant_point point2 ∧
  squared_distance equidistant_point point1 = squared_distance equidistant_point point3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_is_equidistant_l1339_133908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f2_f3_same_cluster_l1339_133940

/-- Definition of "functions of the same cluster" -/
def same_cluster (f g : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = g (x + a) + b

/-- Given functions -/
noncomputable def f1 (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def f2 (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 4)
noncomputable def f3 (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x
noncomputable def f4 (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x) + 1

/-- Theorem stating that f2 and f3 are functions of the same cluster -/
theorem f2_f3_same_cluster : same_cluster f2 f3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f2_f3_same_cluster_l1339_133940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_c_equals_12_l1339_133967

noncomputable section

-- Define the quadratic equation
def quadratic (b c x : ℝ) : ℝ := 4 * x^2 + b * x + c

-- Define the roots
def root1 (α : ℝ) : ℝ := Real.tan α
def root2 (c α : ℝ) : ℝ := 3 * c * (1 / Real.tan α)

-- State the theorem
theorem quadratic_roots_imply_c_equals_12 (b c α : ℝ) : 
  (∃ x, quadratic b c x = 0 ∧ (x = root1 α ∨ x = root2 c α)) →
  c = 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_c_equals_12_l1339_133967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horsepower_calculation_tractive_force_calculation_l1339_133979

-- Define the given constants
noncomputable def P : ℝ := 1800  -- Weight of the automobile in kg
noncomputable def ρ : ℝ := 1 / 50  -- Coefficient of friction
noncomputable def e : ℝ := 1 / 40  -- Sine of the inclination angle
noncomputable def t : ℝ := 65  -- Time taken to reach maximum speed in seconds
noncomputable def s : ℝ := 900  -- Distance covered to reach maximum speed in meters
noncomputable def v : ℝ := 12  -- Maximum speed in m/s

-- Define the calculation of work done
noncomputable def L : ℝ := (ρ * P + e * P) * s

-- Define the conversion factor for horsepower
noncomputable def horsepower_conversion : ℝ := 75  -- 1 horsepower = 75 kg⋅m/s

-- Theorem for horsepower calculation
theorem horsepower_calculation :
  L / (t * horsepower_conversion) = 15 := by sorry

-- Theorem for tractive force calculation
theorem tractive_force_calculation :
  L / s = 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horsepower_calculation_tractive_force_calculation_l1339_133979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_l1339_133913

theorem divisible_by_eleven (a1 a2 a3 a4 a5 : ℝ) : 
  ∃ l1 l2 l3 l4 l5 : ℤ, 
    (l1 ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (l2 ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (l3 ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (l4 ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (l5 ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (l1 ≠ 0 ∨ l2 ≠ 0 ∨ l3 ≠ 0 ∨ l4 ≠ 0 ∨ l5 ≠ 0) ∧ 
    (∃ k : ℤ, l1 * (a1^2 : ℝ) + l2 * (a2^2 : ℝ) + l3 * (a3^2 : ℝ) + l4 * (a4^2 : ℝ) + l5 * (a5^2 : ℝ) = 11 * (k : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_l1339_133913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_sin_shift_l1339_133943

/-- The center of symmetry of the function y = sin(x - π/5) + 1 -/
theorem center_of_symmetry_sin_shift (k : ℤ) :
  ∃ (x : ℝ) (y : ℝ), x = k * Real.pi + Real.pi / 5 ∧ y = 1 ∧
  (∀ (t : ℝ), Real.sin (x + t - Real.pi / 5) + 1 = Real.sin (x - t - Real.pi / 5) + 1) ∧
  (Real.pi / 5, 1) ∈ {(x, y) | ∃ k : ℤ, x = k * Real.pi + Real.pi / 5 ∧ y = 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_sin_shift_l1339_133943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_hyperbola_ellipse_l1339_133927

theorem distance_hyperbola_ellipse : 
  ∀ (a α : ℝ), a > 0 → 0 ≤ α → α ≤ π/2 →
  ((Real.sqrt 6 * Real.cos α - a)^2 + (5/a - Real.sin α)^2) ≥ (9/7)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_hyperbola_ellipse_l1339_133927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_is_11_5_l1339_133969

/-- Represents the score distribution of students in a math test -/
structure ScoreDistribution where
  score65 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  score105 : ℝ
  sum_to_one : score65 + score75 + score85 + score95 + score105 = 1
  non_negative : score65 ≥ 0 ∧ score75 ≥ 0 ∧ score85 ≥ 0 ∧ score95 ≥ 0 ∧ score105 ≥ 0

/-- Calculates the mean score given a score distribution -/
def mean_score (d : ScoreDistribution) : ℝ :=
  65 * d.score65 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95 + 105 * d.score105

/-- Determines the median score given a score distribution -/
noncomputable def median_score (d : ScoreDistribution) : ℝ :=
  if d.score65 + d.score75 + d.score85 > 0.5 then 85
  else if d.score65 + d.score75 + d.score85 + d.score95 > 0.5 then 95
  else 105

/-- Theorem stating that for the given score distribution, 
    the difference between the median and mean score is 11.5 -/
theorem score_difference_is_11_5 (d : ScoreDistribution) 
    (h1 : d.score65 = 0.2)
    (h2 : d.score75 = 0.25)
    (h3 : d.score85 = 0.15)
    (h4 : d.score95 = 0.3) :
    median_score d - mean_score d = 11.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_is_11_5_l1339_133969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_sides_not_sufficient_for_similarity_l1339_133912

/-- Triangle structure -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Similarity relation between triangles -/
def Similar (T1 T2 : Triangle) : Prop :=
  ∃ (k : ℝ), 
    (T1.side1 = k * T2.side1) ∧
    (T1.side2 = k * T2.side2) ∧
    (T1.side3 = k * T2.side3)

/-- Two triangles with proportional sides are not necessarily similar -/
theorem proportional_sides_not_sufficient_for_similarity :
  ∃ (T1 T2 : Triangle) (k : ℝ),
    (T1.side1 = k * T2.side1) ∧
    (T1.side2 = k * T2.side2) ∧
    ¬ Similar T1 T2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_sides_not_sufficient_for_similarity_l1339_133912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1339_133981

/-- The area of a triangle QCA with given coordinates and conditions -/
theorem triangle_area (x p : ℝ) (h1 : x > 0) (h2 : 0 < p) (h3 : p < 15) : 
  (1/2 : ℝ) * x * (15 - p) = 
  (1/2 : ℝ) * (x - 0) * (15 - p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1339_133981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_properties_l1339_133961

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 9) :
  (∀ x₁ x₂ : ℝ, f a (x₁ + x₂) = f a x₁ * f a x₂) ∧
  (∀ x₁ x₂ : ℝ, f a ((x₁ + x₂) / 2) ≤ (f a x₁ + f a x₂) / 2) :=
by
  sorry

#check exponential_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_properties_l1339_133961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_conversion_l1339_133978

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y ≥ 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ, z)

theorem rectangular_to_cylindrical_conversion :
  let (r, θ, z) := rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 4
  r = 6 ∧ θ = 4 * Real.pi / 3 ∧ z = 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_conversion_l1339_133978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1339_133923

/-- A function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a five-digit positive integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The set of all five-digit positive integers whose product of digits equals 18 -/
def valid_numbers : Finset ℕ := sorry

/-- The main theorem stating that there are exactly 70 such numbers -/
theorem count_valid_numbers : valid_numbers.card = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1339_133923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_six_l1339_133953

theorem tan_a_pi_over_six (a : ℝ) : (3 : ℝ)^a = 9 → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_over_six_l1339_133953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_bisector_l1339_133973

/-- A point is on the angle bisector of the second and fourth quadrants if and only if its coordinates are opposite numbers. -/
theorem point_on_angle_bisector (a b : ℝ) : 
  (∃ (A : ℝ × ℝ), A = (a, b) ∧ A.1 * A.2 < 0 ∧ |A.1| = |A.2|) ↔ a = -b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_bisector_l1339_133973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1339_133931

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube given its side length -/
noncomputable def cubeVolume (side : ℝ) : ℝ :=
  side ^ 3

/-- Calculates the percentage of volume removed from a box -/
noncomputable def percentageVolumeRemoved (boxDim : BoxDimensions) (cubeSide : ℝ) : ℝ :=
  let originalVolume := boxVolume boxDim
  let removedVolume := 8 * cubeVolume cubeSide
  (removedVolume / originalVolume) * 100

/-- Theorem: The percentage of volume removed from the given box is approximately 14.22% -/
theorem volume_removed_percentage :
  let boxDim : BoxDimensions := { length := 20, width := 15, height := 12 }
  let cubeSide : ℝ := 4
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |percentageVolumeRemoved boxDim cubeSide - 14.22| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1339_133931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1339_133907

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / (x - 8)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 8 ∨ x > 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1339_133907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1339_133957

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  f₁ : Point  -- First focus
  f₂ : Point  -- Second focus
  a : ℝ        -- Semi-major axis length

/-- Calculate the distance between two points -/
noncomputable def distance (p₁ p₂ : Point) : ℝ :=
  Real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  distance p e.f₁ + distance p e.f₂ = 2 * e.a

theorem ellipse_minor_axis_length 
  (e : Ellipse) 
  (h₁ : distance e.f₁ e.f₂ = 8) 
  (h₂ : e.a = 5) :
  ∃ (b : ℝ), b^2 = e.a^2 - (distance e.f₁ e.f₂ / 2)^2 ∧ 2 * b = 6 := by
  sorry

#check ellipse_minor_axis_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1339_133957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_first_1500_even_is_1444_l1339_133964

/-- Calculate the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

/-- Calculate the sum of digits for even numbers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ :=
  (List.range (n/2)).map (λ i => num_digits ((i + 1) * 2)) |>.sum

/-- The first 1500 positive even integers end at 3000 -/
def first_1500_even_end : ℕ :=
  3000

/-- The total number of digits used when writing down the first 1500 positive even integers -/
def digits_first_1500_even : ℕ :=
  sum_digits_even first_1500_even_end

theorem digits_first_1500_even_is_1444 : digits_first_1500_even = 1444 := by
  sorry

#eval digits_first_1500_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_first_1500_even_is_1444_l1339_133964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_c_range_l1339_133928

-- Define the function f(x) = xe^x + c
noncomputable def f (x c : ℝ) : ℝ := x * Real.exp x + c

-- Theorem statement
theorem two_zeros_c_range (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x c = 0 ∧ f y c = 0) →
  (∀ z : ℝ, z ≠ x → z ≠ y → f z c ≠ 0) →
  0 < c ∧ c < Real.exp (-1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_c_range_l1339_133928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base6_l1339_133901

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The sum of 555₆, 55₆, and 5₆ in base 6 is equal to 1103₆ -/
theorem sum_in_base6 : 
  decimalToBase6 (base6ToDecimal [5, 5, 5] + base6ToDecimal [5, 5] + base6ToDecimal [5]) = [1, 1, 0, 3] := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_in_base6_l1339_133901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1339_133956

/-- Arithmetic sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Geometric sequence b_n -/
def b : ℕ → ℝ := sorry

/-- Sequence c_n defined as the sum of a_n and b_n -/
def c (n : ℕ) : ℝ := a n + b n

/-- Sum of the first n terms of c_n -/
def S : ℕ → ℝ := sorry

theorem sequence_properties :
  (a 3 = 5) ∧
  (a 5 - 2 * a 2 = 3) ∧
  (b 1 = 3) ∧
  (∀ n, b (n + 1) = 3 * b n) →
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, b n = 3^n) ∧
  (∀ n, S n = n^2 + (3/2) * (3^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1339_133956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1339_133905

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sqrt 3 * Real.sin (2 * x) + 2 * m

-- State the theorem
theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ x m, f (x + T) m = f x m) ∧ 
  (∀ T' > 0, (∀ x m, f (x + T') m = f x m) → T' ≥ T) ∧
  (∀ k : ℤ, ∀ x m, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    (∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 3) x, f y m ≤ f x m)) ∧
  (∃ m : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x m ≥ 0) ∧ 
    (∃ x₀ ∈ Set.Icc 0 (Real.pi / 4), f x₀ m = 0) ∧ m = -1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1339_133905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_theorem_l1339_133977

theorem new_student_weight_theorem (initial_count : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) : 
  initial_count = 19 → 
  initial_avg = 15 → 
  new_avg = 14.8 → 
  (initial_count : ℝ) * initial_avg + new_student_weight = (initial_count + 1 : ℝ) * new_avg →
  new_student_weight = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_theorem_l1339_133977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_relation_l1339_133951

theorem alpha_beta_relation (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) 
  (h5 : Real.sin (α + β) = 2 * Real.sin α) : 
  β > α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_relation_l1339_133951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1339_133990

def t : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | n+2 => if (n+2) % 3 = 0 then 2 + t ((n+2)/3) else 1 / t (n+1)

theorem sequence_property (n : ℕ) (h : n > 0) :
  t n = 3/29 → n = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1339_133990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solutions_l1339_133993

theorem tan_equation_solutions (x : ℝ) :
  0 ≤ x ∧ x < π →
  (Real.tan (4 * x - π / 4) = 1 ↔ x ∈ ({π / 8, 3 * π / 8, 5 * π / 8, 7 * π / 8} : Set ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solutions_l1339_133993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAMB_l1339_133914

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 12 = 1

/-- Point A -/
def A : ℝ × ℝ := (2, 0)

/-- Point B -/
noncomputable def B : ℝ × ℝ := (0, 2 * Real.sqrt 3)

/-- The area of quadrilateral OAMB given a point M -/
noncomputable def area_OAMB (M : ℝ × ℝ) : ℝ :=
  let (x, y) := M
  (A.1 * y + B.2 * x) / 2

/-- The maximum area of quadrilateral OAMB -/
theorem max_area_OAMB :
  (∃ (M : ℝ × ℝ), ellipse M.1 M.2 ∧ M.1 ≥ 0 ∧ M.2 ≥ 0) →
  (∃ (max_area : ℝ), 
    max_area = 2 * Real.sqrt 6 ∧
    ∀ (M : ℝ × ℝ), ellipse M.1 M.2 → M.1 ≥ 0 → M.2 ≥ 0 → area_OAMB M ≤ max_area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAMB_l1339_133914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equals_interval_l1339_133937

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (4 - x^2)}

-- Define set N
def N : Set ℝ := {x : ℝ | (2 : ℝ)^(x - 1) ≥ 1}

-- Define the half-open interval [0, 1)
def interval_zero_one : Set ℝ := Set.Ici 0 ∩ Set.Iio 1

-- Theorem statement
theorem set_intersection_equals_interval : 
  M ∩ (U \ N) = interval_zero_one := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equals_interval_l1339_133937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1339_133952

noncomputable section

def train1_length : ℝ := 220
def train2_length : ℝ := 310
def train1_speed_kmh : ℝ := 80
def train2_speed_kmh : ℝ := 50

def total_length : ℝ := train1_length + train2_length

def kmh_to_ms (speed : ℝ) : ℝ := speed * (1000 / 3600)

def train1_speed_ms : ℝ := kmh_to_ms train1_speed_kmh
def train2_speed_ms : ℝ := kmh_to_ms train2_speed_kmh

def relative_speed : ℝ := train1_speed_ms + train2_speed_ms

def crossing_time : ℝ := total_length / relative_speed

theorem trains_crossing_time :
  ∃ ε > 0, |crossing_time - 14.68| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1339_133952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_mod_3_congruent_mod_3_valid_answers_invalid_answers_l1339_133980

def a : ℕ → ℕ 
  | 0 => 2
  | n + 1 => a n + 2 * 3^(n + 1)

def a_2003 : ℕ := a 2003

theorem a_mod_3 : a_2003 % 3 = 2 := by sorry

theorem congruent_mod_3 (b : ℕ) : b % 3 = 2 ↔ b ≡ a_2003 [MOD 3] := by sorry

theorem valid_answers : (1007 ≡ a_2003 [MOD 3]) ∧ (6002 ≡ a_2003 [MOD 3]) := by sorry

theorem invalid_answers : ¬(2013 ≡ a_2003 [MOD 3]) ∧ ¬(3003 ≡ a_2003 [MOD 3]) := by sorry

#check a_mod_3
#check congruent_mod_3
#check valid_answers
#check invalid_answers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_mod_3_congruent_mod_3_valid_answers_invalid_answers_l1339_133980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_distance_implies_w_l1339_133998

-- Define the function f(x)
noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 6)

-- Helper function to represent the minimum distance from a symmetry center to the axis of symmetry
noncomputable def min_symmetry_distance (g : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem symmetry_distance_implies_w (w : ℝ) (h1 : w > 0) 
  (h2 : ∃ (d : ℝ), d = Real.pi / 3 ∧ d = min_symmetry_distance (f w)) : w = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_distance_implies_w_l1339_133998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_form_circular_arc_l1339_133954

/-- A circle in the Euclidean plane. --/
structure Circle where
  center : Point
  radius : ℝ
  points : Set Point

/-- An arc of a circle between two points. --/
def Circle.arc (circle : Circle) (A B : Point) : Set Point :=
  sorry

/-- The set of points where pairs of circles tangent to the sides of angle ∠ACB at A and B touch each other. --/
def set_of_tangency_points (C A B : Point) : Set Point :=
  sorry

/-- Given an angle with apex C and points A and B on its sides, this theorem states that
    the set of points where pairs of circles tangent to the angle's sides at A and B
    touch each other forms an arc AB of a circle passing through A, B, and D*. --/
theorem tangency_points_form_circular_arc (C A B : Point) (angle : Angle) :
  ∃ (D : Point) (circle : Circle),
    A ∈ circle.points ∧ 
    B ∈ circle.points ∧ 
    D ∈ circle.points ∧
    (∀ (X : Point), X ∈ set_of_tangency_points C A B → X ∈ circle.arc A B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_form_circular_arc_l1339_133954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_propositions_l1339_133976

open Real

-- Define the propositions
def proposition1 : Prop := ∃ α, sin α + cos α = 3/2
def proposition2 : Prop := ∀ x, sin ((5/2)*Real.pi - 2*x) = sin ((5/2)*Real.pi + 2*x)
def proposition3 : Prop := ∀ x, sin (2*(x + Real.pi/8) + 5*Real.pi/4) = sin (2*(Real.pi/4 - x) + 5*Real.pi/4)
def proposition4 : Prop := ∀ x y, 0 < x ∧ x < Real.pi/2 ∧ 0 < y ∧ y < Real.pi/2 ∧ x < y → exp (sin (2*x)) < exp (sin (2*y))
def proposition5 : Prop := ∀ α β, 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ α > β → tan α > tan β
def proposition6 : Prop := ∀ x, 3 * sin (2*x + Real.pi/3) = 3 * sin (2*(x + Real.pi/3))

theorem trigonometric_propositions :
  proposition2 ∧ 
  proposition3 ∧ 
  ¬proposition1 ∧ 
  ¬proposition4 ∧ 
  ¬proposition5 ∧ 
  ¬proposition6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_propositions_l1339_133976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_power_sum_eq_one_iff_two_l1339_133972

theorem cos_sin_power_sum_eq_one_iff_two (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) :
  (∀ x : ℝ, (Real.cos φ) ^ x + (Real.sin φ) ^ x = 1) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_power_sum_eq_one_iff_two_l1339_133972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_theorem_l1339_133986

/-- Calculates the new water depth after placing a cube in a container -/
noncomputable def new_water_depth (a : ℝ) : ℝ :=
  if 0 < a ∧ a < 9 then
    10 / 9 * a
  else if 9 ≤ a ∧ a < 59 then
    a + 1
  else if 59 ≤ a ∧ a ≤ 60 then
    60
  else
    0  -- Invalid input

theorem water_depth_theorem (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 60) :
  new_water_depth a =
    if 0 < a ∧ a < 9 then 10 / 9 * a
    else if 9 ≤ a ∧ a < 59 then a + 1
    else 60 := by
  sorry

#check water_depth_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_theorem_l1339_133986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l1339_133997

/-- The function f(x) = (ax - 1)e^(x-2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * Real.exp (x - 2)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  a * Real.exp (x - 2) + (a * x - 1) * Real.exp (x - 2)

theorem tangent_line_passes_through_point (a : ℝ) :
  (f_derivative a 2 * (3 - 2) = 2 - f a 2) → a = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l1339_133997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_sum_l1339_133966

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := n - 1

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(a n)

-- Define the sum of the first n terms of b_n
noncomputable def S (n : ℕ) : ℝ := 2^n - 1

-- Theorem statement
theorem arithmetic_geometric_sequence_sum :
  (a 2 = 1) ∧ (a 5 = 4) →
  (∀ n : ℕ, a n = n - 1) ∧
  (∀ n : ℕ, S n = 2^n - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_sum_l1339_133966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_l1339_133932

/-- The circumference of the circle defined by x^2 + y^2 - 2x + 2y = 0 is 2√2π -/
theorem circle_circumference (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 2*y = 0) → 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 * Real.pi ∧ c = 2 * Real.pi * Real.sqrt ((x-1)^2 + (y+1)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_l1339_133932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_negative_one_l1339_133934

/-- Given a function f(x) = x^2 - 2x * f'(-1), prove that f'(-1) = -2/3 -/
theorem derivative_at_negative_one (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 - 2*x*(deriv f (-1))) :
  deriv f (-1) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_negative_one_l1339_133934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_implies_cos_sin_sum_l1339_133975

theorem tan_two_implies_cos_sin_sum (a : ℝ) (h : Real.tan a = 2) : 
  Real.cos (2 * a) + Real.sin (2 * a) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_implies_cos_sin_sum_l1339_133975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1339_133971

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := y / 4 + x^2 = 1

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Intersection points M and N -/
noncomputable def M (P : PointOnEllipse) : ℝ × ℝ :=
  ((2 * P.x - P.y) / 4, -(2 * P.x - P.y) / 2)

noncomputable def N (P : PointOnEllipse) : ℝ × ℝ :=
  ((2 * P.x + P.y) / 4, (2 * P.x + P.y) / 2)

/-- The main theorem -/
theorem max_distance_MN (P : PointOnEllipse) :
  distance (M P).1 (M P).2 (N P).1 (N P).2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1339_133971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_apples_l1339_133983

theorem extra_apples (red_apples green_apples students_wanting_fruit : ℕ) 
  (h1 : red_apples = 60)
  (h2 : green_apples = 34)
  (h3 : students_wanting_fruit = 7) :
  red_apples + green_apples - students_wanting_fruit = 87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_apples_l1339_133983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l1339_133911

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial P and the value a -/
structure PolynomialConditions (P : IntPolynomial) (a : ℤ) : Prop where
  a_pos : a > 0
  eq_a : P.eval 1 = a ∧ P.eval 5 = a ∧ P.eval 7 = a ∧ P.eval 9 = a
  eq_neg_a : P.eval 2 = -a ∧ P.eval 4 = -a ∧ P.eval 6 = -a ∧ P.eval 8 = -a

/-- The theorem stating that 336 is the smallest possible value of a -/
theorem smallest_a (P : IntPolynomial) (a : ℤ) (h : PolynomialConditions P a) :
  a ≥ 336 ∧ ∃ (Q : IntPolynomial), PolynomialConditions Q 336 := by
  sorry

#check smallest_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l1339_133911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l1339_133930

/-- A hyperbola passing through (-3, 2√3) and sharing an asymptote with x²/9 - y²/16 = 1 -/
structure SharedAsymptoteHyperbola where
  -- Equation of the hyperbola
  equation : ℝ → ℝ → Prop
  -- The hyperbola passes through (-3, 2√3)
  passes_through : equation (-3) (2 * Real.sqrt 3)
  -- The hyperbola shares an asymptote with x²/9 - y²/16 = 1
  shares_asymptote : ∃ (k : ℝ), ∀ (x y : ℝ),
    equation x y ↔ x^2 / 9 - y^2 / 16 = k

/-- The distance from a focus of the hyperbola to its asymptote is 2 -/
theorem focus_to_asymptote_distance (h : SharedAsymptoteHyperbola) : 
  ∃ (f : ℝ × ℝ) (a : ℝ → ℝ), 
    (∀ (x : ℝ), a x = (4/3) * x) ∧ 
    (∃ (y : ℝ), h.equation (f.1) (f.2)) ∧
    (dist f (⟨0, a 0⟩) = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l1339_133930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_implies_p_l1339_133929

/-- 
Given a parabola y^2 = 2px with p > 0, if the focus has coordinates (2, 0), 
then p = 4.
-/
theorem parabola_focus_implies_p (p : ℝ) : 
  p > 0 → 
  (∀ x y : ℝ, y^2 = 2*p*x) → 
  (2 : ℝ) = p/2 → 
  p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_implies_p_l1339_133929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l1339_133922

/-- The coefficient of x^3 in the expansion of (x+a)(x-2)^5 -/
def coeff_x3 (a : ℝ) : ℝ := -8 * (Nat.choose 5 3) + a * 4 * (Nat.choose 5 2)

/-- Theorem stating that if the coefficient of x^3 in (x+a)(x-2)^5 is -60, then a = 1/2 -/
theorem coefficient_implies_a_value (a : ℝ) : coeff_x3 a = -60 → a = 1/2 := by
  intro h
  -- Expand the definition of coeff_x3
  unfold coeff_x3 at h
  -- Simplify the equation
  simp [Nat.choose] at h
  -- Solve the equation
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l1339_133922
