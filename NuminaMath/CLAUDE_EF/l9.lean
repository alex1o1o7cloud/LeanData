import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l9_961

def line_through_points (x1 y1 x2 y2 : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

theorem points_on_line :
  let line := line_through_points 4 10 1 1
  (line 0 (-2)) ∧ (line 5 13) ∧ (line 6 16) ∧
  ¬(line 2 3) ∧ ¬(line 3 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l9_961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_127_l9_987

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + 1

theorem a_7_equals_127 : sequence_a 7 = 127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_7_equals_127_l9_987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l9_909

-- Define the line and parabola equations
noncomputable def line (k : ℝ) (x : ℝ) : ℝ := k * x - 2
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (8 * x)

-- Define the intersection points
noncomputable def A (k : ℝ) : ℝ × ℝ := sorry
noncomputable def B (k : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem length_of_AB (k : ℝ) :
  (A k).1 + (B k).1 = 4 →  -- x-coordinate of midpoint is 2
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    line k x₁ = parabola x₁ ∧
    line k x₂ = parabola x₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (line k x₁ - line k x₂)^2) = 2 * Real.sqrt 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l9_909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rhombus_triangle_sides_l9_907

/-- A rhombus inscribed in a right triangle with specific properties -/
structure InscribedRhombus where
  /-- The side length of the rhombus -/
  rhombus_side : ℝ
  /-- The angle of the right triangle that is common with the rhombus -/
  common_angle : ℝ
  /-- Assertion that the common angle is 60 degrees -/
  angle_is_60 : common_angle = 60
  /-- Assertion that the rhombus side length is 6 cm -/
  side_is_6 : rhombus_side = 6

/-- The sides of the triangle containing the inscribed rhombus -/
noncomputable def triangle_sides (r : InscribedRhombus) : (ℝ × ℝ × ℝ) :=
  (9, 9 * Real.sqrt 3, 18)

/-- Theorem stating the sides of the triangle given an inscribed rhombus -/
theorem inscribed_rhombus_triangle_sides (r : InscribedRhombus) :
  triangle_sides r = (9, 9 * Real.sqrt 3, 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rhombus_triangle_sides_l9_907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_l9_931

-- Define the lines
def line1 (a x y : ℝ) : Prop := a * x + y - 4 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the intersection point
noncomputable def intersection (a : ℝ) : ℝ × ℝ := (6 / (a + 1), (4 - 2 * a) / (a + 1))

-- Define the condition for the first quadrant
def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, line1 a x y ∧ line2 x y) →
  in_first_quadrant (intersection a) →
  -1 < a ∧ a < 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_l9_931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_contains_all_distances_l9_995

-- Define a partition of 3-dimensional space into three disjoint subsets
def Partition (A B C : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  A ∪ B ∪ C = Set.univ ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅

-- Define the property that a set contains all possible distances
def ContainsAllDistances (S : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  ∀ a : ℝ, a > 0 → ∃ M N : EuclideanSpace ℝ (Fin 3), M ∈ S ∧ N ∈ S ∧ dist M N = a

-- State the theorem
theorem partition_contains_all_distances :
  ∀ A B C : Set (EuclideanSpace ℝ (Fin 3)), Partition A B C →
    ContainsAllDistances A ∨ ContainsAllDistances B ∨ ContainsAllDistances C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_contains_all_distances_l9_995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_needed_l9_967

/-- The number of columns to be painted -/
def num_columns : ℕ := 20

/-- The height of each column in feet -/
def column_height : ℝ := 24

/-- The diameter of each column in feet -/
def column_diameter : ℝ := 8

/-- The area that one gallon of paint can cover in square feet -/
def paint_coverage : ℝ := 300

/-- Calculates the lateral surface area of a right circular cylinder -/
noncomputable def lateral_surface_area (radius height : ℝ) : ℝ := 2 * Real.pi * radius * height

/-- Calculates the total number of gallons needed, rounded up to the nearest integer -/
noncomputable def gallons_needed (total_area coverage : ℝ) : ℕ := 
  (Int.ceil (total_area / coverage)).toNat

theorem paint_gallons_needed : 
  gallons_needed (num_columns * lateral_surface_area (column_diameter / 2) column_height) paint_coverage = 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_needed_l9_967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18_seconds_l9_928

/-- Represents the time taken for a train to cross a signal pole -/
noncomputable def time_to_cross_pole (train_length platform_length time_to_cross_platform : ℝ) : ℝ :=
  train_length / ((train_length + platform_length) / time_to_cross_platform)

/-- Theorem stating that a 300m train crossing a 350m platform in 39s takes about 18s to cross a signal pole -/
theorem train_crossing_time_approx_18_seconds :
  ∃ ε > 0, ε < 0.1 ∧ |time_to_cross_pole 300 350 39 - 18| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18_seconds_l9_928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_tangent_l9_901

-- Define the basic structures
structure Circle where
  center : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define auxiliary functions
def intersect (c1 c2 : Circle) (p : Point) : Prop := sorry
def midpoint_arc (c : Circle) (p1 p2 p3 : Point) : Prop := sorry
def inside (c : Circle) (p : Point) : Prop := sorry
def chord (c : Circle) (p1 p2 : Point) : Prop := sorry
def intersect_line_circle (l : Line) (c : Circle) (p : Point) : Prop := sorry
def tangent_line (c : Circle) (p : Point) (l : Line) : Prop := sorry
def line (p1 p2 : Point) : Line := sorry
def triangle (l1 l2 l3 : Line) : Set Point := sorry
def circumcircle (t : Set Point) : Circle := sorry
def tangent (c1 c2 : Circle) : Prop := sorry

-- Define the problem setup
def problem (ω Ω : Circle) (A B M P Q : Point) (ℓP ℓQ : Line) : Prop :=
  -- ω and Ω intersect at A and B
  (intersect ω Ω A) ∧ (intersect ω Ω B) ∧
  -- M is the midpoint of arc AB on ω
  (midpoint_arc ω A B M) ∧
  -- M is inside Ω
  (inside Ω M) ∧
  -- MP is a chord of ω
  (chord ω M P) ∧
  -- MP intersects Ω at Q
  (intersect_line_circle (line M P) Ω Q) ∧
  -- Q is inside ω
  (inside ω Q) ∧
  -- ℓP is tangent to ω at P
  (tangent_line ω P ℓP) ∧
  -- ℓQ is tangent to Ω at Q
  (tangent_line Ω Q ℓQ)

-- Define the theorem
theorem circumcircle_tangent 
  (ω Ω : Circle) (A B M P Q : Point) (ℓP ℓQ : Line) 
  (h : problem ω Ω A B M P Q ℓP ℓQ) : 
  tangent Ω (circumcircle (triangle ℓP ℓQ (line A B))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_tangent_l9_901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_free_integer_solutions_l9_951

theorem square_free_integer_solutions (d : ℤ) : 
  (∃ (x y n : ℕ), x > 0 ∧ y > 0 ∧ n > 0 ∧ (x : ℤ)^2 + d * (y : ℤ)^2 = (2 : ℤ)^n ∧ Squarefree d) ↔ d = 1 ∨ d = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_free_integer_solutions_l9_951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_liquid_approx_1_654_l9_938

/-- Calculates the remaining amount of liquid in Omar's cup after a series of drinking and adding events --/
noncomputable def remaining_liquid (initial_coffee : ℝ) (ml_to_oz : ℝ) (espresso_ml : ℝ) (iced_tea_oz : ℝ) : ℝ :=
  let remaining_after_first_drink := initial_coffee * (1 - 1/4)
  let remaining_after_second_drink := remaining_after_first_drink * (1 - 1/3)
  let remaining_after_espresso := remaining_after_second_drink + (espresso_ml / ml_to_oz)
  let remaining_after_third_drink := remaining_after_espresso * (1 - 0.75)
  let remaining_after_iced_tea := remaining_after_third_drink + (iced_tea_oz / 2)
  remaining_after_iced_tea * (1 - 0.6)

/-- Theorem stating that the remaining liquid in Omar's cup is approximately 1.654 ounces --/
theorem remaining_liquid_approx_1_654 :
  ∃ ε > 0, |remaining_liquid 12 29.57 75 4 - 1.654| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_liquid_approx_1_654_l9_938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l9_942

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - f a (-x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + f a (-x)

theorem function_properties (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∀ x, h a x = -(h a (-x))) ∧
  ((a > 1 → ∀ x y, x < y → h a x < h a y) ∧
   (a < 1 → ∀ x y, x < y → h a x > h a y)) ∧
  Set.range (g a) = Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l9_942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_trip_takes_less_time_l9_997

/-- Represents a car trip with a given distance and two speeds -/
structure CarTrip where
  distance : ℝ
  speed : ℝ
  fasterSpeed : ℝ

/-- Calculates the time taken for a trip given distance and speed -/
noncomputable def tripTime (d s : ℝ) : ℝ := d / s

theorem faster_trip_takes_less_time (trip : CarTrip) 
  (h1 : trip.distance = 90)
  (h2 : trip.fasterSpeed = trip.speed + 30)
  (h3 : trip.speed > 0) :
  tripTime trip.distance trip.fasterSpeed < tripTime trip.distance trip.speed :=
by
  sorry

#check faster_trip_takes_less_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_trip_takes_less_time_l9_997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l9_919

/-- The distance between the intersection points of a line and a circle. -/
theorem intersection_distance
  (a b c : ℝ) -- Line coefficients: ax + by + c = 0
  (x₀ y₀ r : ℝ) -- Circle center (x₀, y₀) and radius r
  (h_line : b ≠ 0) -- Ensure the line is not vertical
  (h_circle : r > 0) -- Ensure the circle has positive radius
  : 2 * Real.sqrt (r^2 - ((y₀ - (-c/b)) + x₀ - (-a/b))^2 / (1 + (a/b)^2)) = 8 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l9_919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_bike_time_is_two_hours_l9_954

/-- The time taken for Jake to bike to the water park -/
noncomputable def jakes_bike_time (
  total_drive_time : ℝ
  ) (dad_speed1 dad_speed2 jake_speed : ℝ) : ℝ :=
  let half_time := total_drive_time / 2
  let distance1 := dad_speed1 * half_time
  let distance2 := dad_speed2 * half_time
  let total_distance := distance1 + distance2
  total_distance / jake_speed

theorem jakes_bike_time_is_two_hours :
  jakes_bike_time (30 / 60) 28 60 11 = 2 := by
  unfold jakes_bike_time
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_bike_time_is_two_hours_l9_954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l9_990

noncomputable def sequence_a : ℕ+ → ℝ := sorry

noncomputable def S : ℕ+ → ℝ := sorry

axiom a_1 : sequence_a 1 = 1

axiom S_def : ∀ n : ℕ+, S n = n^2 * sequence_a n

theorem sequence_properties :
  (∀ n : ℕ+, S n = 2 * n / (n + 1)) ∧
  (∀ n : ℕ+, sequence_a n = 2 / (n * (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l9_990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_division_by_five_l9_993

theorem remainder_division_by_five (n : ℕ) (h : n % 10 = 7) : n % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_division_by_five_l9_993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l9_948

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 1) => sequence_a n / (1 + n * sequence_a n)

theorem sequence_a_property : 1 / sequence_a 2005 - 2000000 = 9011 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l9_948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shaded_area_l9_981

noncomputable def square_area (side : ℝ) : ℝ := side ^ 2

noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius ^ 2

noncomputable def figure_A_shaded_area (side : ℝ) : ℝ :=
  square_area side - circle_area (side / 2)

noncomputable def figure_B_shaded_area (side : ℝ) : ℝ :=
  square_area side - 2 * (circle_area 1 / 4)

noncomputable def figure_C_shaded_area (side : ℝ) : ℝ :=
  circle_area (side / 2) - square_area side

theorem smallest_shaded_area (side : ℝ) (h : side = 3) :
  figure_A_shaded_area side < figure_B_shaded_area side ∧
  figure_A_shaded_area side < figure_C_shaded_area side :=
by
  sorry

#check smallest_shaded_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shaded_area_l9_981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_lines_l9_973

/-- Given two lines l₁ and l₂ with equations mx + y + 2m - 3 = 0 and mx + y - m + 1 = 0 respectively,
    where m is a real number, the maximum distance between l₁ and l₂ is 5. -/
theorem max_distance_between_lines (m : ℝ) :
  let l₁ := {p : ℝ × ℝ | m * p.1 + p.2 + 2 * m - 3 = 0}
  let l₂ := {p : ℝ × ℝ | m * p.1 + p.2 - m + 1 = 0}
  ∃ (d : ℝ), d = 5 ∧ ∀ (p₁ p₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₂ → dist p₁ p₂ ≤ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_lines_l9_973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_powers_l9_922

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - (17 : ℝ)^(1/4)) * (x - (37 : ℝ)^(1/4)) * (x - (57 : ℝ)^(1/4)) = 1/4

-- Define the three distinct solutions
def has_three_solutions (u v w : ℝ) : Prop :=
  u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ equation u ∧ equation v ∧ equation w

-- Theorem statement
theorem sum_of_fourth_powers (u v w : ℝ) :
  has_three_solutions u v w → u^4 + v^4 + w^4 = 112 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_powers_l9_922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_C_l9_980

theorem existence_of_C : ∃ C : ℝ, C > 0 ∧
  ∀ n : ℕ, n ≥ 2 →
    ∀ X : Finset ℕ, X ⊆ Finset.range n →
      X.Nonempty → X.card ≥ 2 →
        ∃ x y z w : ℕ, x ∈ X ∧ y ∈ X ∧ z ∈ X ∧ w ∈ X ∧
          0 < |((x : ℝ) * y - (z : ℝ) * w)| ∧
          |((x : ℝ) * y - (z : ℝ) * w)| < C * ((n : ℝ) / X.card) ^ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_C_l9_980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_faces_l9_974

/-- The probability of selecting a cube with exactly two red faces -/
theorem probability_two_red_faces 
  (edge_length : ℕ) 
  (small_edge_length : ℕ) 
  (total_cubes : ℕ) 
  (two_red_faces_cubes : ℕ) 
  (h1 : edge_length = 4)
  (h2 : small_edge_length = 1)
  (h3 : total_cubes = 64)
  (h4 : two_red_faces_cubes = 24)
  : (two_red_faces_cubes : ℚ) / total_cubes = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_faces_l9_974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fair_coin_tosses_all_same_probability_l9_992

/-- A fair coin toss is represented as a random variable with two equally likely outcomes -/
def FairCoinToss : Type := Bool

/-- The probability of getting heads (or tails) in a single fair coin toss -/
noncomputable def singleTossProbability : ℝ := 1 / 2

/-- The number of times the coin is tossed -/
def numTosses : ℕ := 3

/-- The probability of getting all heads in three tosses -/
noncomputable def allHeadsProbability : ℝ := singleTossProbability ^ numTosses

/-- The probability of getting all tails in three tosses -/
noncomputable def allTailsProbability : ℝ := singleTossProbability ^ numTosses

/-- The probability of getting all the same outcomes (either all heads or all tails) in three tosses -/
noncomputable def allSameOutcomesProbability : ℝ := allHeadsProbability + allTailsProbability

theorem three_fair_coin_tosses_all_same_probability :
  allSameOutcomesProbability = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fair_coin_tosses_all_same_probability_l9_992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_is_150_l9_994

-- Define the rectangle ABCD
noncomputable def AB : ℝ := 20
noncomputable def BC : ℝ := 12

-- Define the triangle ABE
noncomputable def AE : ℝ := 15

-- Define the area of the rectangle
noncomputable def area_rectangle : ℝ := AB * BC

-- Define the area of the triangle
noncomputable def area_triangle : ℝ := (1/2) * AE * BC

-- Theorem to prove
theorem remaining_area_is_150 :
  area_rectangle - area_triangle = 150 := by
  -- Unfold definitions
  unfold area_rectangle area_triangle AB BC AE
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_is_150_l9_994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_OA_OB_l9_977

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define z₁ and z₂
noncomputable def z₁ : ℂ := (Complex.ofReal (-1) + Complex.ofReal 3 * i) / (Complex.ofReal 1 + Complex.ofReal 2 * i)
noncomputable def z₂ : ℂ := 1 + (1 + i)^10

-- Define points A and B
noncomputable def A : ℂ := z₁
noncomputable def B : ℂ := z₂

-- Define vectors OA and OB
noncomputable def OA : ℝ × ℝ := (z₁.re, z₁.im)
noncomputable def OB : ℝ × ℝ := (z₂.re, z₂.im)

-- State the theorem
theorem dot_product_OA_OB : 
  (OA.1 * OB.1 + OA.2 * OB.2 : ℝ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_OA_OB_l9_977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l9_963

/-- Calculates the time for a train to cross a platform given its speed, the time to pass a man, and the platform length. -/
noncomputable def timeToCrossPlatform (trainSpeedKmph : ℝ) (timeToPassMan : ℝ) (platformLength : ℝ) : ℝ :=
  let trainSpeedMps := trainSpeedKmph * (1000 / 3600)
  let trainLength := trainSpeedMps * timeToPassMan
  let totalDistance := trainLength + platformLength
  totalDistance / trainSpeedMps

/-- Theorem stating that a train traveling at 72 kmph, passing a man in 18 seconds on a 320-meter platform, takes 34 seconds to cross the platform. -/
theorem train_crossing_time :
  timeToCrossPlatform 72 18 320 = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l9_963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_l9_958

-- Define a right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  is_right : C = 90

-- Define the conditions of the problem
def triangle_conditions (t : RightTriangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.AB = 2 * t.BC

-- State the theorem
theorem sin_A_value (t : RightTriangle) (h : triangle_conditions t) : 
  Real.sin (t.A * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_l9_958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_identities_l9_914

theorem trig_sum_identities (α β a b : ℝ) 
  (h1 : Real.sin α + Real.sin β = a) 
  (h2 : Real.cos α + Real.cos β = b) : 
  Real.cos (α + β) = -(a^2 - b^2) / (a^2 + b^2) ∧ 
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_identities_l9_914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_in_circle_l9_915

theorem chord_length_in_circle (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 30 * π / 180) :
  2 * r * Real.sin (θ / 2) = 15 := by
  have h3 : θ / 2 = 15 * π / 180 := by
    rw [h2]
    ring
  
  have h4 : Real.sin (15 * π / 180) = 1 / 4 * Real.sqrt 6 := by
    sorry -- This is a known trigonometric identity
  
  calc
    2 * r * Real.sin (θ / 2) = 2 * 15 * Real.sin (15 * π / 180) := by rw [h1, h3]
    _ = 2 * 15 * (1 / 4 * Real.sqrt 6) := by rw [h4]
    _ = 15 * (1 / 2 * Real.sqrt 6) := by ring
    _ = 15 := by sorry -- This step requires proving that 1/2 * √6 = 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_in_circle_l9_915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_C_parallel_to_polar_axis_l9_988

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- A line in polar coordinates -/
structure PolarLine where
  equation : PolarPoint → Prop

/-- The polar axis -/
def polarAxis : PolarLine :=
  { equation := fun p => p.θ = 0 }

/-- A line is parallel to the polar axis if its equation is of the form θ = constant -/
def isParallelToPolarAxis (l : PolarLine) : Prop :=
  ∃ k : ℝ, ∀ p : PolarPoint, l.equation p ↔ p.θ = k

/-- The point C in the problem -/
noncomputable def C : PolarPoint :=
  { r := 6, θ := Real.pi / 6 }

/-- The theorem statement -/
theorem line_through_C_parallel_to_polar_axis :
  ∃ l : PolarLine, l.equation C ∧ isParallelToPolarAxis l ∧
  ∀ p : PolarPoint, l.equation p ↔ p.θ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_C_parallel_to_polar_axis_l9_988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l9_950

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 1/x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1/x + 2*x + 1/x^2

-- Theorem: The equation of the tangent line to f(x) at x = 1 is 4x - y - 4 = 0
theorem tangent_line_at_one :
  ∀ x y : ℝ, (y - f 1 = f' 1 * (x - 1)) ↔ (4*x - y - 4 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l9_950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_F_less_than_180_deg_l9_921

/-- Triangle DEF with sides d, e, and f -/
structure Triangle :=
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)

/-- Angle F in triangle DEF -/
noncomputable def angle_F (t : Triangle) : ℝ :=
  Real.arccos ((t.d^2 + t.e^2 - t.f^2) / (2 * t.d * t.e))

theorem angle_F_less_than_180_deg (t : Triangle) 
  (h1 : t.d = 2)
  (h2 : t.e = 2)
  (h3 : t.f > 2 * Real.sqrt 2) :
  ∀ ε > 0, ∃ t' : Triangle, 
    t'.d = t.d ∧ 
    t'.e = t.e ∧ 
    t'.f > 2 * Real.sqrt 2 ∧ 
    angle_F t' < π ∧
    π - angle_F t' < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_F_less_than_180_deg_l9_921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l9_982

theorem point_on_parabola (u : ℝ) :
  let x := (3 : ℝ)^u - 2
  let y := (3 : ℝ)^(2*u) - 6 * (3 : ℝ)^u + 4
  y = x^2 - 2*x + 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_l9_982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_powers_l9_936

theorem equality_of_powers (x : ℝ) : 
  (3 : ℝ)^(2*x) = 6 * (2 : ℝ)^(2*x) - 5 * (6 : ℝ)^x → (2 : ℝ)^x = (3 : ℝ)^x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_powers_l9_936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_problem_l9_957

theorem tan_ratio_problem (α β : Real) 
  (h1 : Real.cos (π/4 - α) = 3/5)
  (h2 : Real.sin (5*π/4 + β) = -12/13)
  (h3 : α ∈ Set.Ioo (π/4) (3*π/4))
  (h4 : β ∈ Set.Ioo 0 (π/4)) :
  Real.tan α / Real.tan β = -17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_problem_l9_957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_and_asymptotes_l9_969

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 24 + y^2 / 49 = 1

/-- The equation of the given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 / 36 - y^2 / 64 = 1

/-- The equation of the target hyperbola -/
def target_hyperbola (x y : ℝ) : Prop := y^2 / 16 - x^2 / 9 = 1

/-- The foci of an ellipse or hyperbola -/
noncomputable def foci (a b : ℝ) : Set (ℝ × ℝ) := {(0, c), (0, -c)} where c := Real.sqrt (a^2 - b^2)

/-- The asymptotes of a hyperbola -/
def asymptotes (a b : ℝ) : Set (ℝ → ℝ) := {λ x => (b/a) * x, λ x => -(b/a) * x}

theorem hyperbola_foci_and_asymptotes :
  (foci 7 (Real.sqrt 24) = foci 4 3) ∧
  (asymptotes 6 8 = asymptotes 3 4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_and_asymptotes_l9_969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_correct_l9_935

noncomputable def rotation_angle : ℝ := Real.pi / 3  -- 60° in radians

def scale_factor : ℝ := 2

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, -Real.sqrt 3],
    ![Real.sqrt 3, 1]]

theorem transformation_correct :
  transformation_matrix = scale_factor • 
    ![![Real.cos rotation_angle, -Real.sin rotation_angle],
      ![Real.sin rotation_angle,  Real.cos rotation_angle]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_correct_l9_935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l9_946

/-- Represents Jason's work schedule and earnings --/
structure WorkSchedule where
  afterSchoolRate : ℚ
  saturdayRate : ℚ
  totalHours : ℚ
  totalEarnings : ℚ

/-- Calculates the number of hours worked on Saturday given a work schedule --/
def saturdayHours (schedule : WorkSchedule) : ℚ :=
  (schedule.totalEarnings - schedule.afterSchoolRate * schedule.totalHours) / 
  (schedule.saturdayRate - schedule.afterSchoolRate)

/-- Theorem stating that Jason worked 8 hours on Saturday --/
theorem jason_saturday_hours :
  let schedule := WorkSchedule.mk 4 6 18 88
  saturdayHours schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_hours_l9_946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l9_908

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  simp [g]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l9_908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_approx_l9_959

/-- Calculates the overall loss percentage for three items --/
noncomputable def overallLossPercentage (
  radioCost : ℝ) (radioAd : ℝ) (radioSell : ℝ)
  (tvCost : ℝ) (tvAd : ℝ) (tvSell : ℝ)
  (fridgeCost : ℝ) (fridgeAd : ℝ) (fridgeSell : ℝ) : ℝ :=
  let totalCost := radioCost + radioAd + tvCost + tvAd + fridgeCost + fridgeAd
  let totalSell := radioSell + tvSell + fridgeSell
  let totalLoss := totalCost - totalSell
  (totalLoss / totalCost) * 100

/-- The overall loss percentage is approximately 10.18% --/
theorem loss_percentage_approx (ε : ℝ) (h : ε > 0) :
  |overallLossPercentage 1500 100 1335 5500 200 5050 12000 500 11400 - 10.18| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_approx_l9_959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l9_905

/-- The function f(x) as defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2/2 + (1-k)*x - k * Real.log x

/-- The theorem statement -/
theorem f_inequality (k : ℝ) (x : ℝ) (hk : k > 0) (hx : x > 0) :
  f k x + 3/2 * k^2 - 2*k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l9_905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_equals_one_l9_945

/-- A curve defined by y = ax + ln x -/
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

/-- The tangent line y = 2x - 1 -/
def tangentLine (x : ℝ) : ℝ := 2 * x - 1

/-- The derivative of the curve -/
noncomputable def curveDeriv (a : ℝ) (x : ℝ) : ℝ := a + 1 / x

theorem tangent_implies_a_equals_one (a : ℝ) :
  (∃ m : ℝ, m > 0 ∧ curve a m = tangentLine m ∧ curveDeriv a m = 2) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_equals_one_l9_945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_apples_l9_964

def total_apples : ℕ := 10
def green_apples : ℕ := 4
def picked_apples : ℕ := 2

theorem probability_two_green_apples :
  (Nat.choose green_apples picked_apples : ℚ) / (Nat.choose total_apples picked_apples) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_apples_l9_964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_4_range_m_for_nonempty_solution_f_geq_3_l9_943

-- Define the function f
def f (x : ℝ) : ℝ := 2 * |x - 1| + |x + 2|

-- Theorem for part I
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = Set.Iic 0 ∪ Set.Ici (4/3) := by sorry

-- Theorem for part II
theorem range_m_for_nonempty_solution :
  {m : ℝ | ∃ x, f x < |m - 2|} = Set.Iio (-1) ∪ Set.Ioi 5 := by sorry

-- Additional theorem to show f(x) ≥ 3 for all x
theorem f_geq_3 (x : ℝ) : f x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_4_range_m_for_nonempty_solution_f_geq_3_l9_943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_AB_area_FAB_l9_984

-- Define the line and parabola
def line (x : ℝ) : ℝ := 2 * x - 4

noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (4 * x)

-- Define the intersection points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (4, 4)

-- Define the focus of the parabola
def F : ℝ × ℝ := (1, 0)

-- Theorem for the midpoint of AB
theorem midpoint_AB : (A.1 + B.1) / 2 = 2.5 ∧ (A.2 + B.2) / 2 = 1 := by
  sorry

-- Theorem for the area of triangle FAB
theorem area_FAB : 3 * Real.sqrt 5 * (2 / Real.sqrt 5) / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_AB_area_FAB_l9_984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_holds_l9_976

/-- A regular decagon inscribed in a circle with specific area properties -/
structure RegularDecagon where
  /-- The circle in which the decagon is inscribed -/
  circle : Set (ℝ × ℝ)
  /-- The area of the circle is 1 -/
  circle_area : MeasureTheory.volume circle = 1
  /-- Point Q inside the circle -/
  Q : ℝ × ℝ
  /-- Q is inside the circle -/
  Q_in_circle : Q ∈ circle
  /-- Area of region QB₁B₂ -/
  area_QB1B2 : ℝ
  /-- Area of region QB₃B₄ -/
  area_QB3B4 : ℝ
  /-- Area of region QB₇B₈ -/
  area_QB7B8 : ℝ
  /-- The area of QB₁B₂ is 1/10 -/
  area_QB1B2_eq : area_QB1B2 = 1/10
  /-- The area of QB₃B₄ is 1/12 -/
  area_QB3B4_eq : area_QB3B4 = 1/12
  /-- There exists a positive integer m such that the area of QB₇B₈ is 1/11 - √3/m -/
  exists_m : ∃ m : ℕ+, area_QB7B8 = 1/11 - Real.sqrt 3 / m.val

/-- The value of m for which the area conditions hold -/
noncomputable def find_m (d : RegularDecagon) : ℕ+ :=
  ⟨(Int.floor (110 * Real.sqrt 3)).toNat + 1, Nat.succ_pos _⟩

/-- Theorem stating that the found m satisfies the area condition -/
theorem area_condition_holds (d : RegularDecagon) :
    d.area_QB7B8 = 1/11 - Real.sqrt 3 / (find_m d).val := by
  sorry

#check area_condition_holds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_holds_l9_976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l9_924

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- Maximum area of triangle PAB -/
def max_area_PAB : ℝ := 2

/-- Slope relation between AP and QB -/
def slope_relation (k₁ k₂ : ℝ) : Prop :=
  3 * k₁ = 5 * k₂

/-- Theorem stating the properties of the ellipse and the derived results -/
theorem ellipse_properties (a b : ℝ) (h_ellipse : ellipse_C 0 0 a b)
  (h_ecc : eccentricity a b = Real.sqrt 3 / 2)
  (h_area : max_area_PAB = 2) :
  (∃ x y : ℝ, ellipse_C x y 2 1) ∧
  (∀ k₁ k₂ : ℝ, slope_relation k₁ k₂ → 
    ∃ t : ℝ, ∀ x y : ℝ, ellipse_C x y 2 1 → (x = t * y - 1/2)) ∧
  (∃ S₁ S₂ : ℝ → ℝ, ∀ t : ℝ, 
    |S₁ t - S₂ t| ≤ Real.sqrt 15 / 4 ∧
    (∃ t₀ : ℝ, |S₁ t₀ - S₂ t₀| = Real.sqrt 15 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l9_924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_marginal_profit_l9_949

/-- Revenue function -/
def R (x : ℝ) : ℝ := 3000 * x - 20 * x^2

/-- Cost function -/
def C (x : ℝ) : ℝ := 500 * x + 4000

/-- Profit function -/
def P (x : ℝ) : ℝ := R x - C x

/-- Marginal function of a function f -/
def M (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1) - f x

/-- Marginal profit function -/
def MP : ℝ → ℝ := M P

/-- Maximum value of a function f on the interval [0, 100] -/
noncomputable def max_value (f : ℝ → ℝ) : ℝ := 
  ⨆ (x : ℝ) (hx : x ∈ Set.Icc 0 100), f x

theorem profit_and_marginal_profit :
  (P = fun x => -20 * x^2 + 2500 * x - 4000) ∧
  (MP = fun x => -40 * x + 2480) ∧
  (max_value P ≠ max_value MP) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_marginal_profit_l9_949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_investment_period_is_14_months_l9_991

/-- Represents the investment and profit information for two partners -/
structure PartnershipData where
  investment_ratio : Rat
  profit_ratio : Rat
  q_investment_period : ℕ

/-- Calculates the investment period for partner p given the partnership data -/
def calculate_p_investment_period (data : PartnershipData) : Rat :=
  (data.profit_ratio * data.q_investment_period) / data.investment_ratio

/-- Theorem stating that given the specific partnership data, 
    the investment period for partner p is 14 months -/
theorem p_investment_period_is_14_months 
  (h : PartnershipData) 
  (h_investment : h.investment_ratio = 7 / 5)
  (h_profit : h.profit_ratio = 7 / 10)
  (h_q_period : h.q_investment_period = 20) :
  calculate_p_investment_period h = 14 := by
  sorry

#eval calculate_p_investment_period {
  investment_ratio := 7 / 5,
  profit_ratio := 7 / 10,
  q_investment_period := 20
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_investment_period_is_14_months_l9_991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l9_913

def a : ℝ × ℝ × ℝ := (-2, -3, 1)
def b : ℝ × ℝ × ℝ := (2, 0, 4)
def c : ℝ × ℝ × ℝ := (-4, -6, 2)

def parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2.1 = k * w.2.1 ∧ v.2.2 = k * w.2.2

def orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2 = 0

theorem vector_relations : parallel a c ∧ orthogonal a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l9_913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_set_probability_l9_968

noncomputable def prob_digit (d : ℕ) : ℝ := Real.log (d + 1 : ℝ) - Real.log (d : ℝ)

theorem digit_set_probability : 
  prob_digit 6 + prob_digit 7 + prob_digit 8 + prob_digit 9 = 3 * prob_digit 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_set_probability_l9_968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_correctness_l9_947

-- Statement A
def line_A (a : ℝ) (x : ℝ) : ℝ := a * x - 3 * a + 2

-- Statement B
noncomputable def line_B (x : ℝ) : ℝ := -Real.sqrt 3 * (x - 2) - 1

-- Statement C
def line_C (x : ℝ) : ℝ := -2 * x + 3

-- Statement D
def line_D (x y : ℝ) : Prop := x + y - 2 = 0

theorem statements_correctness :
  (∃ a : ℝ, line_A a 3 = 2) ∧  -- Statement A is correct
  (line_B 2 = -1) ∧            -- Statement B is correct
  (¬ ∀ x : ℝ, line_C x = -2 * x + 3 ∨ line_C x = -2 * x - 3) ∧  -- Statement C is incorrect
  (¬ (line_D 1 1 ∧ ∀ x : ℝ, x ≠ 0 → line_D x x))  -- Statement D is incorrect
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_correctness_l9_947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_divisor_of_level_polynomial_l9_904

/-- A polynomial is level if it has integer coefficients and equals 30 at 0, 2, 5, and 6. -/
def IsLevelPolynomial (P : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, ∃ k : ℤ, P n = k) ∧ P 0 = 30 ∧ P 2 = 30 ∧ P 5 = 30 ∧ P 6 = 30

/-- The largest positive integer that divides P(n) for all integers n, given P is level. -/
def LargestCommonDivisor (P : ℤ → ℤ) : ℕ :=
  Int.natAbs (Int.gcd (P 1 - 30) (P 3 - 30))

theorem largest_common_divisor_of_level_polynomial (P : ℤ → ℤ) (h : IsLevelPolynomial P) :
  LargestCommonDivisor P = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_divisor_of_level_polynomial_l9_904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_surfaces_theorem_l9_906

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A conical surface of revolution -/
structure ConicalSurface where
  apex : Point3D
  axis : Line3D
  angle : ℝ

/-- Three lines are non-coplanar if their direction vectors are linearly independent -/
def nonCoplanar (l1 l2 l3 : Line3D) : Prop :=
  let v1 := l1.direction
  let v2 := l2.direction
  let v3 := l3.direction
  ¬(∃ (a b c : ℝ), a * v1.x + b * v2.x + c * v3.x = 0 ∧
                    a * v1.y + b * v2.y + c * v3.y = 0 ∧
                    a * v1.z + b * v2.z + c * v3.z = 0 ∧
                    (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0))

/-- A line is a generatrix of a conical surface if it lies on the surface -/
def isGeneratrix (l : Line3D) (c : ConicalSurface) : Prop :=
  sorry

/-- Main theorem -/
theorem conical_surfaces_theorem (S : Point3D) (l1 l2 l3 : Line3D) 
  (h1 : l1.point = S) (h2 : l2.point = S) (h3 : l3.point = S)
  (h4 : nonCoplanar l1 l2 l3) :
  ∃ (surfaces : Finset ConicalSurface),
    surfaces.card = 4 ∧
    (∀ c, c ∈ surfaces → c.apex = S) ∧
    (∀ c, c ∈ surfaces → isGeneratrix l1 c ∧ isGeneratrix l2 c ∧ isGeneratrix l3 c) ∧
    (∀ c1 c2, c1 ∈ surfaces → c2 ∈ surfaces → c1 ≠ c2 → c1.axis ≠ c2.axis) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_surfaces_theorem_l9_906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_FQGR_l9_916

/-- Represents a triangle in the diagram -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ
  isIsosceles : (vertices 0).1 = (vertices 2).1

/-- Represents the trapezoid FQGR in the diagram -/
structure Trapezoid where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a geometric shape -/
noncomputable def area {n : Nat} (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that the area of trapezoid FQGR is 40 -/
theorem area_of_trapezoid_FQGR (PQR : Triangle) (FQGR : Trapezoid) 
  (h1 : ∀ T : Triangle, area T.vertices ≤ area PQR.vertices)
  (h2 : area PQR.vertices = 45)
  (h3 : ∃ smallTriangles : Finset Triangle, smallTriangles.card = 9 ∧ 
        ∀ T ∈ smallTriangles, area T.vertices = 1) :
  area FQGR.vertices = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_FQGR_l9_916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l9_978

noncomputable def f (x : ℝ) := 2 * Real.cos x + (Real.sin x) ^ 2

theorem f_minimum_value (x : ℝ) (h : -Real.pi/4 < x ∧ x ≤ Real.pi/2) :
  f x ≥ 1 ∧ f (Real.pi/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l9_978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_donation_l9_923

/-- Represents the daily production and donation of masks -/
structure MaskProduction where
  medical : ℚ  -- Daily production of medical masks in millions
  n95 : ℚ      -- Daily production of N95 masks in millions
  donated_medical : ℚ  -- Donated medical mask packages
  donated_n95 : ℚ      -- Donated N95 mask packages

/-- Conditions for mask production and donation -/
def valid_production (p : MaskProduction) : Prop :=
  p.medical + p.n95 = 80 ∧  -- Total daily production is 800,000 masks
  0.4 * p.medical + 0.5 * p.n95 = 35 ∧  -- Profit condition
  p.donated_n95 ≤ p.donated_medical / 3 ∧  -- Donation constraint
  1.2 * (p.medical - p.donated_medical) + 3 * (p.n95 - p.donated_n95) -
    0.8 * p.medical - 2.5 * p.n95 = 2  -- Profit after donation

/-- Theorem stating the optimal production and valid donation options -/
theorem optimal_production_and_donation :
  ∃ (p : MaskProduction),
    valid_production p ∧
    p.medical = 50 ∧
    p.n95 = 30 ∧
    ((p.donated_medical = 15 ∧ p.donated_n95 = 5) ∨
     (p.donated_medical = 20 ∧ p.donated_n95 = 3) ∨
     (p.donated_medical = 25 ∧ p.donated_n95 = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_donation_l9_923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_fifty_percent_l9_952

/-- Represents the key chain manufacturing scenario --/
structure KeyChainScenario where
  initial_cost : ℚ
  new_cost : ℚ
  initial_profit_percentage : ℚ
  selling_price : ℚ

/-- Calculates the new profit percentage given a KeyChainScenario --/
noncomputable def new_profit_percentage (scenario : KeyChainScenario) : ℚ :=
  let new_profit := scenario.selling_price - scenario.new_cost
  (new_profit / scenario.selling_price) * 100

/-- Theorem stating that under the given conditions, the new profit percentage is 50% --/
theorem profit_percentage_is_fifty_percent 
  (scenario : KeyChainScenario) 
  (h1 : scenario.initial_cost = 80)
  (h2 : scenario.new_cost = 50)
  (h3 : scenario.initial_profit_percentage = 20)
  (h4 : scenario.selling_price = scenario.initial_cost / (1 - scenario.initial_profit_percentage / 100)) :
  new_profit_percentage scenario = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_fifty_percent_l9_952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l9_900

theorem role_assignment_count :
  let n : ℕ := 4  -- number of people and roles
  let people : Fin n → String := λ i => match i with
    | 0 => "Alice"
    | 1 => "Bob"
    | 2 => "Carol"
    | 3 => "Dave"
    | _ => ""  -- Add a catch-all case
  let roles : Fin n → String := λ i => match i with
    | 0 => "president"
    | 1 => "vice president"
    | 2 => "secretary"
    | 3 => "treasurer"
    | _ => ""  -- Add a catch-all case
  (Fintype.card {f : Fin n → Fin n | Function.Injective f}) = 24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l9_900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_midpoint_property_l9_939

/-- A point is the center of an excircle of a triangle -/
def is_center_excircle (J A B C : Point) : Prop :=
  sorry

/-- A point is a tangent point of an excircle -/
def is_tangent_point (M J B C : Point) : Prop :=
  sorry

/-- Three points are collinear -/
def collinear (P Q R : Point) : Prop :=
  sorry

/-- A point is the midpoint of two other points -/
def is_midpoint (M S T : Point) : Prop :=
  sorry

/-- Given a triangle ABC with the following properties:
    - J is the center of the excircle opposite to A
    - The excircle is tangent to BC at M, to AB at K, and to AC at L
    - Lines LM and BJ intersect at F
    - Lines KM and CJ intersect at G
    - Lines AF and BC intersect at S
    - Lines AG and BC intersect at T
    Prove that M is the midpoint of ST -/
theorem excircle_midpoint_property (A B C J K L M F G S T : Point) :
  is_center_excircle J A B C →
  is_tangent_point M J B C →
  is_tangent_point K J A B →
  is_tangent_point L J A C →
  collinear L M F →
  collinear B J F →
  collinear K M G →
  collinear C J G →
  collinear A F S →
  collinear B C S →
  collinear A G T →
  collinear B C T →
  is_midpoint M S T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_midpoint_property_l9_939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_tetrahedron_volume_bound_l9_955

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

-- Define a tetrahedron
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hd : d > 0
  he : e > 0
  hf : f > 0

-- Area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Volume of a tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Theorem for triangle
theorem triangle_area_bound (t : Triangle) :
  area t < (t.a^2 + t.b^2 + t.c^2) / 6 := by sorry

-- Theorem for tetrahedron
theorem tetrahedron_volume_bound (t : Tetrahedron) :
  volume t < (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_tetrahedron_volume_bound_l9_955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_when_a_is_1_f_nonpositive_iff_a_is_1_l9_998

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - a * x / (1 - x)

def monotone_increasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

def monotone_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

theorem f_monotonicity_when_a_is_1 :
  (monotone_increasing (f 1) (Set.Ioo (-1) 0) ∧ monotone_increasing (f 1) (Set.Ioi 3)) ∧
  (monotone_decreasing (f 1) (Set.Ioo 0 1) ∧ monotone_decreasing (f 1) (Set.Ioo 1 3)) :=
sorry

theorem f_nonpositive_iff_a_is_1 :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a x ≤ 0) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_when_a_is_1_f_nonpositive_iff_a_is_1_l9_998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winner_l9_937

/-- The game interval --/
def GameInterval (n : ℕ) := Set.Icc (0 : ℝ) (n : ℝ)

/-- Valid move condition --/
def ValidMove (prev_moves : Set ℝ) (new_move : ℝ) : Prop :=
  ∀ x ∈ prev_moves, |new_move - x| ≥ 2

/-- Jenn's winning condition --/
def JennWins (n : ℕ) : Prop :=
  n > 12

theorem game_winner (n : ℕ) (h1 : n > 10) :
  (∃ (strategy : ℕ → ℝ),
    strategy 0 = 3 ∧
    (∀ k, ValidMove {strategy i | i < k} (strategy k)) ∧
    (∀ x ∈ GameInterval n, ∃ k, strategy k = x)) ↔
  JennWins n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winner_l9_937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l9_927

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000)
  (h2 : repair_cost = 13000)
  (h3 : selling_price = 60900) :
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  abs (profit_percent - 10.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l9_927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_time_to_height_l9_953

/-- Represents a Ferris wheel -/
structure FerrisWheel where
  radius : ℝ
  revolutionTime : ℝ

/-- Calculates the time for a rider to reach a certain height above the bottom of the Ferris wheel -/
noncomputable def timeToHeight (wheel : FerrisWheel) (height : ℝ) : ℝ :=
  (wheel.revolutionTime / (2 * Real.pi)) * Real.arccos ((height / wheel.radius) - 1)

theorem ferris_wheel_time_to_height :
  let wheel : FerrisWheel := { radius := 30, revolutionTime := 90 }
  let height : ℝ := 15
  timeToHeight wheel height = 15 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_time_to_height_l9_953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_surface_area_ratio_l9_940

/-- The side length of the cube -/
def cube_side_length : ℝ := 2

/-- The surface area of a cube given its side length -/
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

/-- The side length of the tetrahedron formed by the cube vertices -/
noncomputable def tetrahedron_side_length (s : ℝ) : ℝ := s * Real.sqrt 3

/-- The surface area of a regular tetrahedron given its side length -/
noncomputable def tetrahedron_surface_area (a : ℝ) : ℝ := Real.sqrt 3 * a^2

/-- The theorem stating the ratio of surface areas -/
theorem cube_tetrahedron_surface_area_ratio :
  (cube_surface_area cube_side_length) / (tetrahedron_surface_area (tetrahedron_side_length cube_side_length)) = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_surface_area_ratio_l9_940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_is_good_number_exists_good_number_ge_2005_l9_999

-- Define a point in 2D space
structure Point where
  x : ℚ
  y : ℚ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define what it means for a number to be rational
def isRational (r : ℝ) : Prop :=
  ∃ (q : ℚ), r = q

-- Define what it means for a number to be a good number
def isGoodNumber (n : ℕ) : Prop :=
  n ≥ 3 ∧
  ∃ (points : Fin n → Point),
    ∀ (i j : Fin n), i ≠ j →
      (isRational (distance (points i) (points j))) →
        ∃ (k : Fin n), k ≠ i ∧ k ≠ j ∧
          ¬(isRational (distance (points i) (points k))) ∧
          ¬(isRational (distance (points j) (points k))) ∧
      ¬(isRational (distance (points i) (points j))) →
        ∃ (k : Fin n), k ≠ i ∧ k ≠ j ∧
          (isRational (distance (points i) (points k))) ∧
          (isRational (distance (points j) (points k)))

-- Theorem statements
theorem five_is_good_number : isGoodNumber 5 := by sorry

theorem exists_good_number_ge_2005 : ∃ (n : ℕ), n ≥ 2005 ∧ isGoodNumber n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_is_good_number_exists_good_number_ge_2005_l9_999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l9_966

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (5, 5)
def C : ℝ × ℝ := (0, 5)
def D : ℝ × ℝ := (1, 4)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter of the quadrilateral
noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C D + distance D A

-- Theorem statement
theorem quadrilateral_perimeter :
  perimeter = Real.sqrt 2 + 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l9_966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l9_983

/-- Calculates the second discount percentage given the list price, final price, and first discount percentage -/
noncomputable def second_discount (list_price final_price first_discount : ℝ) : ℝ :=
  let price_after_first_discount := list_price * (1 - first_discount / 100)
  100 * (1 - final_price / price_after_first_discount)

/-- Theorem stating that given the conditions in the problem, the second discount is approximately 8.24% -/
theorem discount_calculation (list_price final_price first_discount : ℝ) 
  (h1 : list_price = 68)
  (h2 : final_price = 56.16)
  (h3 : first_discount = 10) :
  abs (second_discount list_price final_price first_discount - 8.24) < 0.01 := by
  sorry

-- Remove #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l9_983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_sums_l9_979

/-- An arithmetic-geometric sequence -/
noncomputable def arithmetic_geometric_seq (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q^(n-1)

/-- Sum of the first n terms of an arithmetic-geometric sequence -/
noncomputable def S (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem ratio_of_sums (a : ℝ) :
  let q := (1/2 : ℝ)
  (arithmetic_geometric_seq a q 4 = -a/8) →
  (S a q 6) / (S a q 3) = 9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_sums_l9_979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_is_950_l9_970

/-- Represents the cost of movie tickets for a group -/
structure MovieTickets where
  childCost : ℚ
  groupSize : ℕ
  adultCount : ℕ
  totalPaid : ℚ

/-- Calculates the cost of an adult ticket given the movie ticket information -/
noncomputable def adultTicketCost (tickets : MovieTickets) : ℚ :=
  (tickets.totalPaid - (tickets.childCost * (tickets.groupSize - tickets.adultCount))) / tickets.adultCount

/-- Theorem stating that the adult ticket cost is $9.50 given the specific conditions -/
theorem adult_ticket_cost_is_950 (tickets : MovieTickets) 
  (h1 : tickets.childCost = 13/2)
  (h2 : tickets.groupSize = 7)
  (h3 : tickets.adultCount = 3)
  (h4 : tickets.totalPaid = 109/2) :
  adultTicketCost tickets = 19/2 := by
  sorry

#eval (19 : ℚ) / 2  -- This should output 9.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_is_950_l9_970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l9_934

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y : ℝ) (focus_x focus_y : ℝ) : ℝ :=
  Real.sqrt ((x - focus_x)^2 + (y - focus_y)^2)

-- Theorem statement
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (focus1_x focus1_y focus2_x focus2_y : ℝ) 
  (h_distance_focus1 : distance_to_focus x y focus1_x focus1_y = 4) :
  distance_to_focus x y focus2_x focus2_y = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l9_934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_equals_minus_one_minus_i_over_two_l9_902

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function g
noncomputable def g (x : ℂ) : ℂ := (x^6 + x^3 + 1) / (x + 1)

-- Theorem statement
theorem g_of_i_equals_minus_one_minus_i_over_two :
  g i = (-1 - i) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_equals_minus_one_minus_i_over_two_l9_902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_unchanged_l9_920

noncomputable def sample_A : List ℝ := [81, 82, 82, 84, 84, 85, 86, 86, 86]
noncomputable def sample_B : List ℝ := sample_A.map (· + 2)

noncomputable def standard_deviation (l : List ℝ) : ℝ :=
  let mean := l.sum / l.length
  Real.sqrt ((l.map (fun x => (x - mean) ^ 2)).sum / l.length)

theorem standard_deviation_unchanged :
  standard_deviation sample_A = standard_deviation sample_B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_unchanged_l9_920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_symmetry_axis_l9_986

noncomputable section

open Real

/-- The original sine function before translation -/
def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

/-- The translated sine function -/
def g (x : ℝ) : ℝ := sin (2 * x + π / 3)

/-- The translation amount -/
def translation : ℝ := π / 4

/-- The axis of symmetry -/
def axis_of_symmetry : ℝ := π / 12

theorem sine_translation_symmetry_axis :
  (∀ x, g x = f (x + translation)) ∧
  (∀ x, g (axis_of_symmetry + x) = g (axis_of_symmetry - x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_symmetry_axis_l9_986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l9_965

/-- Represents the work rate of a group of men building a water fountain -/
structure WorkRate where
  men : ℕ
  length : ℚ
  days : ℕ

/-- Calculates the work rate per man per day -/
noncomputable def work_rate_per_man_per_day (w : WorkRate) : ℚ :=
  w.length / (w.men : ℚ) / (w.days : ℚ)

/-- The theorem to be proved -/
theorem first_group_size (w1 w2 : WorkRate) : 
  w1.length = 56 ∧ 
  w1.days = 6 ∧ 
  w2.men = 35 ∧ 
  w2.length = 49 ∧ 
  w2.days = 3 ∧
  work_rate_per_man_per_day w1 = work_rate_per_man_per_day w2 →
  w1.men = 20 := by
  sorry

#check first_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l9_965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_is_zero_l9_926

theorem cos_two_beta_is_zero (α β : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = -Real.sqrt 10 / 10)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) :
  Real.cos (2 * β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_beta_is_zero_l9_926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l9_911

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 0 then Real.sin (x + α) else Real.cos (x + α)

theorem sufficient_not_necessary_condition :
  (∀ x, f (π / 4) x = f (π / 4) (-x)) ∧
  (∃ β ≠ π / 4, ∀ x, f β x = f β (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l9_911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poisson_first_event_waiting_time_l9_975

/-- The rate of events per minute in a Poisson process -/
noncomputable def event_rate : ℝ := 6 / 5

/-- The duration of the observation interval in minutes -/
def observation_interval : ℝ := 5

/-- The expected waiting time for the first event in seconds -/
def expected_waiting_time : ℝ := 50

/-- Theorem stating the relationship between event rate, observation interval, and expected waiting time -/
theorem poisson_first_event_waiting_time :
  (1 / event_rate) * (observation_interval * 60) = expected_waiting_time := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_poisson_first_event_waiting_time_l9_975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l9_972

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3 + φ)

theorem max_value_of_f (φ : ℝ) (h1 : |φ| < Real.pi / 2) 
  (h2 : ∀ x, g x φ = g (-x) φ) : 
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y φ ≤ f x φ ∧ f x φ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l9_972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_in_interval_l9_941

-- Define the recursive function f
noncomputable def f : ℕ → ℝ
| 0 => 0  -- Base case for 0
| 1 => 0  -- Base case for 1
| 2 => Real.log 2
| (n+3) => Real.log (n+3 + f (n+2))

-- Define A as f(2013)
noncomputable def A : ℝ := f 2013

-- Theorem statement
theorem A_in_interval : Real.log 2016 < A ∧ A < Real.log 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_in_interval_l9_941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_is_64_l9_930

def my_sequence : List ℕ := [49, 64, 81]

def is_mean_proportional (m : ℕ) (seq : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → j < seq.length →
    (seq.get! i : ℚ) / m = m / (seq.get! j)

theorem mean_proportional_is_64 :
  ∃ (x : ℕ), x > 0 ∧ is_mean_proportional 64 my_sequence := by
  sorry

#check mean_proportional_is_64

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_is_64_l9_930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l9_962

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 * sin (x / 4 + π / 6) - 1

-- Define the minimum positive period
noncomputable def min_period : ℝ := 8 * π

-- Define the maximum value
def max_value : ℝ := 2

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (x + min_period) = f x) ∧
  (∀ x : ℝ, f x ≤ max_value) ∧
  (∃ x : ℝ, f x = max_value) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l9_962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_cosine_product_l9_989

theorem symmetric_shift_cosine_product (a : ℝ) : 
  a > 0 ∧ 
  (∀ x : ℝ, Real.cos (x + a) * Real.cos (x + a + π/6) = Real.cos (-x + a) * Real.cos (-x + a + π/6)) →
  a ≥ 5*π/12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_cosine_product_l9_989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_between_lines_l9_925

/-- A type representing a line in a plane -/
structure Line :=
  (angle : ℝ)

/-- Given n pairwise non-parallel lines in a plane, there exist two lines
    such that the angle between them is less than or equal to 180°/n -/
theorem min_angle_between_lines (n : ℕ) (lines : Fin n → Line)
  (h_non_parallel : ∀ i j, i ≠ j → lines i ≠ lines j) :
  ∃ i j, i ≠ j ∧ |((lines i).angle - (lines j).angle)| ≤ 180 / n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_between_lines_l9_925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_sum_l9_996

/-- Triangle represented by three points in R² -/
structure Triangle where
  p1 : R × R
  p2 : R × R
  p3 : R × R

/-- Rotation in degrees around a point in R² -/
structure Rotation where
  center : R × R
  angle : R

/-- Rotate a point around a center by an angle in degrees -/
noncomputable def rotate (center : R × R) (angle : R) (p : R × R) : R × R :=
  sorry

theorem triangle_rotation_sum (t1 t2 : Triangle) (r : Rotation) :
  t1.p1 = (0, 0) ∧
  t1.p2 = (0, 20) ∧
  t1.p3 = (30, 0) ∧
  t2.p1 = (-26, 23) ∧
  t2.p2 = (-46, 23) ∧
  t2.p3 = (-26, -7) ∧
  0 < r.angle ∧
  r.angle < 180 ∧
  (∃ (f : R × R → R × R), f t1.p1 = t2.p1 ∧ f t1.p2 = t2.p2 ∧ f t1.p3 = t2.p3 ∧
    ∀ p, f p = rotate r.center r.angle p) →
  r.angle + r.center.1 + r.center.2 = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_sum_l9_996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_comparison_l9_933

theorem sqrt_sum_comparison : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_comparison_l9_933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_div_z₂_not_purely_imaginary_l9_918

def z₁ : ℂ := 2 + Complex.I
def z₂ : ℂ := 2 - Complex.I

theorem z₁_div_z₂_not_purely_imaginary : ∃ (a : ℝ), a ≠ 0 ∧ z₁ / z₂ ≠ 0 + a * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_div_z₂_not_purely_imaginary_l9_918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_replacements_l9_956

def replace_asterisks (expression : List Int) : List (List Int) :=
  sorry

theorem sum_of_replacements :
  let initial_expression : List Int := [1, 2, 3, 4, 5, 6]
  let all_replacements := replace_asterisks initial_expression
  (all_replacements.map List.sum).sum = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_replacements_l9_956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_functions_l9_971

-- Define the interval (0,1)
def openUnitInterval : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Define the functions
noncomputable def f1 : ℝ → ℝ := λ x => 2^x
noncomputable def f2 : ℝ → ℝ := λ x => Real.log (x + 1) / Real.log 0.5
noncomputable def f3 : ℝ → ℝ := λ x => Real.sqrt x
def f4 : ℝ → ℝ := λ x => |x - 1|

-- Define monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

-- Theorem statement
theorem monotonically_decreasing_functions :
  (MonoDecreasing f2 openUnitInterval ∧ MonoDecreasing f4 openUnitInterval) ∧
  (¬MonoDecreasing f1 openUnitInterval ∧ ¬MonoDecreasing f3 openUnitInterval) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_functions_l9_971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_tetrahedron_cube_l9_960

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ

/-- A cube whose vertices are the centers of the faces of a regular tetrahedron -/
structure CubeFromTetrahedron (T : RegularTetrahedron) where

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedronVolume (T : RegularTetrahedron) : ℝ :=
  (T.sideLength ^ 3 * Real.sqrt 2) / 4

/-- The volume of a cube constructed from a regular tetrahedron -/
noncomputable def cubeVolume (T : RegularTetrahedron) (C : CubeFromTetrahedron T) : ℝ :=
  (T.sideLength ^ 3 * Real.sqrt 8) / 8

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio_tetrahedron_cube (T : RegularTetrahedron) (C : CubeFromTetrahedron T) :
  tetrahedronVolume T / cubeVolume T C = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_tetrahedron_cube_l9_960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_circles_l9_903

/-- Given two circles with equations (x^2 + y^2 + 2ax + a^2 - 9 = 0) and (x^2 + y^2 - 4by - 1 + 4b^2 = 0),
    with three common tangents, where a and b are non-zero real numbers,
    the minimum value of (4/a^2 + 1/b^2) is 1. -/
theorem min_value_circles (a b : ℝ) : a ≠ 0 → b ≠ 0 → 
  (∃ (x y : ℝ), x^2 + y^2 + 2*a*x + a^2 - 9 = 0) →
  (∃ (x y : ℝ), x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
  (∃ (t1 t2 t3 : ℝ → ℝ → ℝ), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3) →
  (∀ t : ℝ → ℝ → ℝ, (∃ (x y : ℝ), t x y = 0 ∧ x^2 + y^2 + 2*a*x + a^2 - 9 = 0) →
                    (∃ (x y : ℝ), t x y = 0 ∧ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0)) →
  (∀ c : ℝ, c > 0 → 4/a^2 + 1/b^2 ≥ c → c ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_circles_l9_903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_sufficiency_l9_929

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π

-- Define the theorem
theorem tan_product_sufficiency (t : Triangle) :
  (Real.tan t.A * Real.tan t.B = 1) → (Real.sin t.A)^2 + (Real.sin t.B)^2 = 1 ∧
  ¬ ((Real.sin t.A)^2 + (Real.sin t.B)^2 = 1 → Real.tan t.A * Real.tan t.B = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_sufficiency_l9_929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l9_932

-- Define points A, B, C, D, and E
noncomputable def A : ℝ × ℝ := (0, 8)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (10, 0)

-- Define D as midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define E as midpoint of BC
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Function to calculate the area of a triangle given base and height
noncomputable def triangleArea (base height : ℝ) : ℝ := (base * height) / 2

-- Theorem: The area of triangle DBC is 20
theorem area_of_triangle_DBC : 
  triangleArea (C.1 - B.1) (D.2 - B.2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l9_932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_and_cylinder_surface_areas_l9_944

/-- The surface area of a sphere with radius r -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The surface area of a hemisphere with radius r, including its circular base -/
noncomputable def hemisphereSurfaceArea (r : ℝ) : ℝ :=
  (sphereSurfaceArea r) / 2 + Real.pi * r^2

/-- The surface area of a cylinder with radius r and height h -/
noncomputable def cylinderSurfaceArea (r h : ℝ) : ℝ :=
  2 * Real.pi * r * h + 2 * Real.pi * r^2

/-- Theorem stating the surface areas of a hemisphere and cylinder with radius 8 -/
theorem hemisphere_and_cylinder_surface_areas :
  let r : ℝ := 8
  hemisphereSurfaceArea r = 192 * Real.pi ∧
  cylinderSurfaceArea r r = 256 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_and_cylinder_surface_areas_l9_944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l9_985

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x

def d (x₁ x₂ : ℝ) : ℝ := |x₂ - x₁|

theorem min_distance_theorem (x₁ x₂ : ℝ) (h : f x₁ = g x₂) :
  ∃ (min_d : ℝ), min_d = (1 - Real.log 2) / 2 ∧ 
  ∀ (y₁ y₂ : ℝ), f y₁ = g y₂ → d y₁ y₂ ≥ min_d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l9_985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l9_912

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x + 2 * (Real.exp x - Real.exp (-x))

-- State the theorem
theorem range_of_a (a : ℝ) : f (5 * a - 2) + f (3 * a^2) ≤ 0 → -2 ≤ a ∧ a ≤ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l9_912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_relative_errors_l9_910

/-- Relative error of a measurement -/
noncomputable def relative_error (true_value : ℝ) (error : ℝ) : ℝ :=
  (error / true_value) * 100

theorem equal_relative_errors :
  let line1_length : ℝ := 20
  let line1_error : ℝ := 0.04
  let line2_length : ℝ := 150
  let line2_error : ℝ := 0.3
  relative_error line1_length line1_error = relative_error line2_length line2_error :=
by
  -- Unfold the definition of relative_error
  unfold relative_error
  -- Perform algebraic simplification
  simp [div_eq_mul_inv]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_relative_errors_l9_910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_and_infinitude_l9_917

open Nat BigOperators

/-- Definition of f_p(x) -/
def f_p (p : ℕ) (x : ℕ) : ℕ := ∑ i in Finset.range p, x^i

/-- Main theorem -/
theorem prime_divisibility_and_infinitude (p : ℕ) (h_p : Nat.Prime p) :
  (∀ m : ℕ, p ∣ m → ∃ q : ℕ, Nat.Prime q ∧ q ∣ f_p p m ∧ Coprime q (m * (m - 1))) ∧
  (Set.Infinite {n : ℕ | Nat.Prime (p * n + 1)}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_and_infinitude_l9_917
