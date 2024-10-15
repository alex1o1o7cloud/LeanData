import Mathlib

namespace NUMINAMATH_CALUDE_sector_area_l3494_349432

/-- The area of a circular sector with central angle π/3 and radius 4 is 8π/3 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 4) :
  (1 / 2) * θ * r^2 = 8 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3494_349432


namespace NUMINAMATH_CALUDE_chord_ratio_constant_l3494_349446

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a chord of the ellipse
def chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

-- Define parallel lines
def parallel (A B M N : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (N.1 - M.1) = (B.1 - A.1) * (N.2 - M.2)

-- Define a point on a line passing through the origin
def through_origin (M N : ℝ × ℝ) : Prop :=
  M.2 * N.1 = M.1 * N.2

-- Main theorem
theorem chord_ratio_constant
  (A B M N : ℝ × ℝ)
  (h_AB : chord A B)
  (h_MN : chord M N)
  (h_parallel : parallel A B M N)
  (h_origin : through_origin M N) :
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1/4 * ((N.1 - M.1)^2 + (N.2 - M.2)^2)^2 :=
sorry

end NUMINAMATH_CALUDE_chord_ratio_constant_l3494_349446


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3494_349442

theorem quadratic_inequality_roots (k : ℝ) : 
  (∀ x, x^2 + k*x + 24 > 0 ↔ x < -6 ∨ x > 4) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3494_349442


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3494_349410

theorem equation_solutions_count :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (m n : ℤ), (m, n) ∈ s ↔ m^4 + 8*n^2 + 425 = n^4 + 42*m^2) ∧
    s.card = 16 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3494_349410


namespace NUMINAMATH_CALUDE_exists_2008_acquaintances_l3494_349428

/-- Represents a gathering of people -/
structure Gathering where
  people : Finset Nat
  acquaintances : Nat → Finset Nat
  no_common_acquaintances : ∀ x y, x ∈ people → y ∈ people →
    (acquaintances x).card = (acquaintances y).card →
    (acquaintances x ∩ acquaintances y).card ≤ 1

/-- Main theorem: If there's someone with at least 2008 acquaintances,
    then there's someone with exactly 2008 acquaintances -/
theorem exists_2008_acquaintances (g : Gathering) :
  (∃ x ∈ g.people, (g.acquaintances x).card ≥ 2008) →
  (∃ y ∈ g.people, (g.acquaintances y).card = 2008) := by
  sorry

end NUMINAMATH_CALUDE_exists_2008_acquaintances_l3494_349428


namespace NUMINAMATH_CALUDE_set_intersection_example_l3494_349474

theorem set_intersection_example :
  let A : Set ℤ := {1, 0, 3}
  let B : Set ℤ := {-1, 1, 2, 3}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3494_349474


namespace NUMINAMATH_CALUDE_smallest_positive_value_l3494_349401

theorem smallest_positive_value : 
  let S : Set ℝ := {12 - 4 * Real.sqrt 7, 4 * Real.sqrt 7 - 12, 25 - 6 * Real.sqrt 19, 65 - 15 * Real.sqrt 17, 15 * Real.sqrt 17 - 65}
  ∀ x ∈ S, x > 0 → 12 - 4 * Real.sqrt 7 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_value_l3494_349401


namespace NUMINAMATH_CALUDE_total_wheels_in_both_garages_l3494_349464

-- Define the types of vehicles
inductive Vehicle
| Bicycle
| Tricycle
| Unicycle
| Quadracycle

-- Define the garage contents
def first_garage : List (Vehicle × Nat) :=
  [(Vehicle.Bicycle, 5), (Vehicle.Tricycle, 6), (Vehicle.Unicycle, 9), (Vehicle.Quadracycle, 3)]

def second_garage : List (Vehicle × Nat) :=
  [(Vehicle.Bicycle, 2), (Vehicle.Tricycle, 1), (Vehicle.Unicycle, 3), (Vehicle.Quadracycle, 4)]

-- Define the number of wheels for each vehicle type
def wheels_per_vehicle (v : Vehicle) : Nat :=
  match v with
  | Vehicle.Bicycle => 2
  | Vehicle.Tricycle => 3
  | Vehicle.Unicycle => 1
  | Vehicle.Quadracycle => 4

-- Define the number of missing wheels in the second garage
def missing_wheels : Nat := 3

-- Function to calculate total wheels in a garage
def total_wheels_in_garage (garage : List (Vehicle × Nat)) : Nat :=
  garage.foldl (fun acc (v, count) => acc + wheels_per_vehicle v * count) 0

-- Theorem statement
theorem total_wheels_in_both_garages :
  total_wheels_in_garage first_garage +
  total_wheels_in_garage second_garage - missing_wheels = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_both_garages_l3494_349464


namespace NUMINAMATH_CALUDE_apples_per_person_l3494_349455

/-- Given that Harold gave apples to 3.0 people and the total number of apples given was 45,
    prove that each person received 15 apples. -/
theorem apples_per_person (total_apples : ℕ) (num_people : ℝ) 
    (h1 : total_apples = 45) 
    (h2 : num_people = 3.0) : 
  (total_apples : ℝ) / num_people = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_person_l3494_349455


namespace NUMINAMATH_CALUDE_multiply_is_enlarge_l3494_349425

-- Define the concept of enlarging a number
def enlarge (n : ℕ) (times : ℕ) : ℕ := n * times

-- State the theorem
theorem multiply_is_enlarge :
  ∀ (n : ℕ), 28 * 5 = enlarge 28 5 :=
by
  sorry

end NUMINAMATH_CALUDE_multiply_is_enlarge_l3494_349425


namespace NUMINAMATH_CALUDE_our_circle_center_and_radius_l3494_349481

/-- A circle in the 2D plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle --/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle --/
def Circle.radius (c : Circle) : ℝ := sorry

/-- Our specific circle --/
def our_circle : Circle :=
  { equation := fun x y => x^2 + y^2 - 4*x - 6*y - 3 = 0 }

theorem our_circle_center_and_radius :
  Circle.center our_circle = (2, 3) ∧ Circle.radius our_circle = 4 := by sorry

end NUMINAMATH_CALUDE_our_circle_center_and_radius_l3494_349481


namespace NUMINAMATH_CALUDE_range_of_f_l3494_349466

-- Define the function
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ Set.Iic 8 :=
by sorry

-- Note: Set.Iic 8 represents the set (-∞, 8]

end NUMINAMATH_CALUDE_range_of_f_l3494_349466


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l3494_349441

/-- The line equation 2y - 5x = 10 -/
def line_equation (x y : ℝ) : Prop := 2 * y - 5 * x = 10

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line and the y-axis -/
def intersection_point : ℝ × ℝ := (0, 5)

theorem line_y_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_y_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l3494_349441


namespace NUMINAMATH_CALUDE_fritz_money_l3494_349415

theorem fritz_money (fritz sean rick : ℝ) : 
  sean = fritz / 2 + 4 →
  rick = 3 * sean →
  rick + sean = 96 →
  fritz = 40 := by
sorry

end NUMINAMATH_CALUDE_fritz_money_l3494_349415


namespace NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l3494_349458

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon from the midpoints of another polygon's sides -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_preserved (n : ℕ) (q1 q2 q3 : Polygon) :
  n = 44 →
  q1.vertices.length = n →
  sumXCoordinates q1 = 176 →
  q2 = midpointPolygon q1 →
  q3 = midpointPolygon q2 →
  sumXCoordinates q3 = 176 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l3494_349458


namespace NUMINAMATH_CALUDE_implication_proof_l3494_349435

theorem implication_proof (p q r : Prop) : 
  ((p ∧ ¬q ∧ r) → ((p → q) → r)) ∧
  ((¬p ∧ ¬q ∧ r) → ((p → q) → r)) ∧
  ((p ∧ ¬q ∧ ¬r) → ((p → q) → r)) ∧
  ((¬p ∧ q ∧ r) → ((p → q) → r)) := by
  sorry

end NUMINAMATH_CALUDE_implication_proof_l3494_349435


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3494_349492

theorem rectangle_longer_side (r : ℝ) (h1 : r = 6) (h2 : r > 0) : ∃ l w : ℝ,
  l > w ∧ w = 2 * r ∧ l * w = 2 * (π * r^2) ∧ l = 6 * π :=
sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3494_349492


namespace NUMINAMATH_CALUDE_fruit_distribution_l3494_349478

theorem fruit_distribution (num_students : ℕ) 
  (h1 : 2 * num_students + 6 = num_apples)
  (h2 : 7 * num_students - 5 = num_oranges)
  (h3 : num_oranges = 3 * num_apples + 3) : 
  num_students = 26 := by
sorry

end NUMINAMATH_CALUDE_fruit_distribution_l3494_349478


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3494_349499

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 119 ∧
  (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧
  (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 19) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3494_349499


namespace NUMINAMATH_CALUDE_stirling_second_kind_recurrence_stirling_second_kind_5_3_l3494_349467

def S (n k : ℕ) : ℕ := sorry

theorem stirling_second_kind_recurrence (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  S (n + 1) k = S n (k - 1) + k * S n k := by sorry

theorem stirling_second_kind_5_3 :
  S 5 3 = 25 := by sorry

end NUMINAMATH_CALUDE_stirling_second_kind_recurrence_stirling_second_kind_5_3_l3494_349467


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3494_349496

theorem sum_of_roots_quadratic (α β : ℝ) : 
  (∀ x : ℝ, x^2 + x - 2 = 0 ↔ x = α ∨ x = β) →
  α + β = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3494_349496


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3494_349488

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (sphere_diameter : ℝ) 
  (inner_cube_edge : ℝ) (inner_cube_volume : ℝ) :
  outer_cube_edge = 12 →
  sphere_diameter = outer_cube_edge →
  sphere_diameter = inner_cube_edge * Real.sqrt 3 →
  inner_cube_volume = inner_cube_edge ^ 3 →
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3494_349488


namespace NUMINAMATH_CALUDE_speaker_arrangement_count_l3494_349448

def number_of_speakers : ℕ := 5

theorem speaker_arrangement_count :
  (number_of_speakers.factorial / 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_speaker_arrangement_count_l3494_349448


namespace NUMINAMATH_CALUDE_train_crossing_time_l3494_349420

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 50 →
  train_speed_kmh = 60 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3494_349420


namespace NUMINAMATH_CALUDE_total_dress_designs_l3494_349414

/-- The number of fabric colors available --/
def num_colors : ℕ := 5

/-- The number of patterns available --/
def num_patterns : ℕ := 6

/-- The number of fabric types available --/
def num_fabric_types : ℕ := 2

/-- Theorem stating the total number of possible dress designs --/
theorem total_dress_designs : num_colors * num_patterns * num_fabric_types = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l3494_349414


namespace NUMINAMATH_CALUDE_sum_of_roots_difference_l3494_349498

theorem sum_of_roots_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_difference_l3494_349498


namespace NUMINAMATH_CALUDE_exists_central_island_l3494_349426

/-- A type representing the islands -/
def Island : Type := ℕ

/-- A structure representing the City of Islands -/
structure CityOfIslands (n : ℕ) where
  /-- The set of islands -/
  islands : Finset Island
  /-- The number of islands is n -/
  island_count : islands.card = n
  /-- Connectivity relation between islands -/
  connected : Island → Island → Prop
  /-- Any two islands are connected (directly or indirectly) -/
  all_connected : ∀ (a b : Island), a ∈ islands → b ∈ islands → connected a b
  /-- The special connectivity property for four islands -/
  four_island_property : ∀ (a b c d : Island), 
    a ∈ islands → b ∈ islands → c ∈ islands → d ∈ islands →
    connected a b → connected b c → connected c d →
    (connected a c ∨ connected b d)

/-- The main theorem: there exists an island connected to all others -/
theorem exists_central_island {n : ℕ} (h : n ≥ 1) (city : CityOfIslands n) : 
  ∃ (central : Island), central ∈ city.islands ∧ 
    ∀ (other : Island), other ∈ city.islands → city.connected central other :=
sorry

end NUMINAMATH_CALUDE_exists_central_island_l3494_349426


namespace NUMINAMATH_CALUDE_policeman_speed_l3494_349494

/-- Proves that given the initial conditions of a chase between a policeman and a thief,
    the policeman's speed is 64 km/hr. -/
theorem policeman_speed (initial_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ) :
  initial_distance = 160 →
  thief_speed = 8 →
  thief_distance = 640 →
  ∃ (policeman_speed : ℝ), policeman_speed = 64 :=
by
  sorry


end NUMINAMATH_CALUDE_policeman_speed_l3494_349494


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3494_349421

theorem fraction_sum_equality : (3 : ℚ) / 10 + 5 / 100 - 1 / 1000 = 349 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3494_349421


namespace NUMINAMATH_CALUDE_even_function_derivative_at_zero_l3494_349429

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_derivative_at_zero
  (f : ℝ → ℝ)
  (hf : EvenFunction f)
  (hf' : Differentiable ℝ f) :
  deriv f 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_even_function_derivative_at_zero_l3494_349429


namespace NUMINAMATH_CALUDE_jogger_distance_l3494_349491

/-- Proves that given a jogger who jogs at 12 km/hr, if jogging at 20 km/hr would result in 15 km 
    more distance covered, then the actual distance jogged is 22.5 km. -/
theorem jogger_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  actual_speed = 12 →
  faster_speed = 20 →
  faster_speed * (extra_distance / (faster_speed - actual_speed)) = 
    actual_speed * (extra_distance / (faster_speed - actual_speed)) + extra_distance →
  extra_distance = 15 →
  actual_speed * (extra_distance / (faster_speed - actual_speed)) = 22.5 :=
by sorry


end NUMINAMATH_CALUDE_jogger_distance_l3494_349491


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l3494_349475

/-- Represents the flower planting problem with given conditions -/
structure FlowerPlanting where
  costA3B4 : ℕ  -- Cost for 3 pots of A and 4 pots of B
  costA4B3 : ℕ  -- Cost for 4 pots of A and 3 pots of B
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℚ  -- Survival rate of type A flowers
  survivalRateB : ℚ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Calculates the cost per pot for type A and B flowers -/
def calculateCostPerPot (fp : FlowerPlanting) : ℚ × ℚ := sorry

/-- Calculates the minimum cost and optimal planting strategy -/
def minimumCostStrategy (fp : FlowerPlanting) : ℕ × ℕ × ℕ := sorry

/-- Theorem stating the minimum cost and optimal planting strategy -/
theorem minimum_cost_theorem (fp : FlowerPlanting) 
  (h1 : fp.costA3B4 = 330)
  (h2 : fp.costA4B3 = 300)
  (h3 : fp.totalPots = 400)
  (h4 : fp.survivalRateA = 7/10)
  (h5 : fp.survivalRateB = 9/10)
  (h6 : fp.maxReplacement = 80) :
  minimumCostStrategy fp = (200, 200, 18000) := by sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l3494_349475


namespace NUMINAMATH_CALUDE_a_range_l3494_349418

-- Define the function f(x) piecewise
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x / Real.log a
  else (6 - a) * x - 4 * a

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) →
  1 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3494_349418


namespace NUMINAMATH_CALUDE_number_solution_l3494_349477

theorem number_solution (x : ℝ) (n : ℝ) (h1 : x > 0) (h2 : x / n + x / 25 = 0.06 * x) : n = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l3494_349477


namespace NUMINAMATH_CALUDE_four_objects_three_containers_l3494_349422

/-- The number of ways to distribute n distinct objects into k distinct containers --/
def distributionWays (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 4 distinct objects into 3 distinct containers is 81 --/
theorem four_objects_three_containers : distributionWays 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_objects_three_containers_l3494_349422


namespace NUMINAMATH_CALUDE_sequence_sum_divisible_by_37_l3494_349459

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Represents a sequence of four three-digit integers -/
structure FourTermSequence where
  term1 : ThreeDigitInt
  term2 : ThreeDigitInt
  term3 : ThreeDigitInt
  term4 : ThreeDigitInt
  satisfies_property : 
    term2.hundreds = term1.tens ∧ term2.tens = term1.units ∧
    term3.hundreds = term2.tens ∧ term3.tens = term2.units ∧
    term4.hundreds = term3.tens ∧ term4.tens = term3.units ∧
    term1.hundreds = term4.tens ∧ term1.tens = term4.units

/-- The sum of all terms in the sequence -/
def sequence_sum (seq : FourTermSequence) : Nat :=
  (seq.term1.hundreds * 100 + seq.term1.tens * 10 + seq.term1.units) +
  (seq.term2.hundreds * 100 + seq.term2.tens * 10 + seq.term2.units) +
  (seq.term3.hundreds * 100 + seq.term3.tens * 10 + seq.term3.units) +
  (seq.term4.hundreds * 100 + seq.term4.tens * 10 + seq.term4.units)

theorem sequence_sum_divisible_by_37 (seq : FourTermSequence) :
  37 ∣ sequence_sum seq := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_divisible_by_37_l3494_349459


namespace NUMINAMATH_CALUDE_bike_journey_l3494_349430

theorem bike_journey (v d : ℝ) 
  (h1 : d / (v - 4) - d / v = 1.2)
  (h2 : d / v - d / (v + 4) = 2) :
  d = 160 := by
  sorry

end NUMINAMATH_CALUDE_bike_journey_l3494_349430


namespace NUMINAMATH_CALUDE_max_distinct_numbers_in_circle_l3494_349482

/-- Given a circular arrangement of 2023 numbers where each number is the product of its two neighbors,
    the maximum number of distinct numbers is 1. -/
theorem max_distinct_numbers_in_circle (nums : Fin 2023 → ℝ) 
    (h : ∀ i : Fin 2023, nums i = nums (i - 1) * nums (i + 1)) : 
    Finset.card (Finset.image nums Finset.univ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_numbers_in_circle_l3494_349482


namespace NUMINAMATH_CALUDE_distinct_values_of_c_l3494_349485

/-- Given a complex number c and distinct complex numbers r, s, and t satisfying
    (z - r)(z - s)(z - t) = (z - 2cr)(z - 2cs)(z - 2ct) for all complex z,
    there are exactly 3 distinct possible values of c. -/
theorem distinct_values_of_c (c r s t : ℂ) : 
  r ≠ s ∧ s ≠ t ∧ r ≠ t →
  (∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2*c*r) * (z - 2*c*s) * (z - 2*c*t)) →
  ∃! (values : Finset ℂ), values.card = 3 ∧ c ∈ values := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_of_c_l3494_349485


namespace NUMINAMATH_CALUDE_will_old_cards_l3494_349470

/-- Calculates the number of old baseball cards Will had. -/
def old_cards (cards_per_page : ℕ) (new_cards : ℕ) (pages_used : ℕ) : ℕ :=
  cards_per_page * pages_used - new_cards

/-- Theorem stating that Will had 10 old cards. -/
theorem will_old_cards : old_cards 3 8 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_will_old_cards_l3494_349470


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3494_349424

theorem quadratic_minimum_value (x m : ℝ) : 
  (∀ x, x^2 - 4*x + m ≥ 4) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3494_349424


namespace NUMINAMATH_CALUDE_star_square_sum_l3494_349400

/-- The ★ operation for real numbers -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating that (x + y)^2 ★ (y + x)^2 = 4(x + y)^4 -/
theorem star_square_sum (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_star_square_sum_l3494_349400


namespace NUMINAMATH_CALUDE_taco_truck_beef_amount_l3494_349484

/-- The taco truck problem -/
theorem taco_truck_beef_amount :
  ∀ (beef_amount : ℝ),
    (beef_amount > 0) →
    (0.25 * (beef_amount / 0.25) * (2 - 1.5) = 200) →
    beef_amount = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_taco_truck_beef_amount_l3494_349484


namespace NUMINAMATH_CALUDE_survey_ratings_l3494_349480

theorem survey_ratings (total : ℕ) (excellent_percent : ℚ) (satisfactory_remaining_percent : ℚ) (needs_improvement : ℕ) :
  total = 120 →
  excellent_percent = 15 / 100 →
  satisfactory_remaining_percent = 80 / 100 →
  needs_improvement = 6 →
  ∃ (very_satisfactory_percent : ℚ),
    very_satisfactory_percent = 16 / 100 ∧
    excellent_percent + very_satisfactory_percent + 
    (satisfactory_remaining_percent * (1 - excellent_percent - needs_improvement / total)) +
    (needs_improvement / total) = 1 :=
by sorry

end NUMINAMATH_CALUDE_survey_ratings_l3494_349480


namespace NUMINAMATH_CALUDE_unique_integer_expression_l3494_349413

/-- The function representing the given expression -/
def f (x y : ℕ+) : ℚ := (x^2 + y) / (x * y + 1)

/-- The theorem stating that 1 is the only positive integer expressible
    by the function for at least two distinct pairs of positive integers -/
theorem unique_integer_expression :
  ∀ n : ℕ+, (∃ (x₁ y₁ x₂ y₂ : ℕ+), (x₁, y₁) ≠ (x₂, y₂) ∧ f x₁ y₁ = n ∧ f x₂ y₂ = n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_expression_l3494_349413


namespace NUMINAMATH_CALUDE_tan_alpha_eq_two_implies_fraction_eq_negative_two_l3494_349462

theorem tan_alpha_eq_two_implies_fraction_eq_negative_two (α : Real) 
  (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_two_implies_fraction_eq_negative_two_l3494_349462


namespace NUMINAMATH_CALUDE_cube_side_length_l3494_349427

theorem cube_side_length (volume : ℝ) (x : ℝ) : volume = 8 → x^3 = volume → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l3494_349427


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l3494_349471

-- Define the GDP value in yuan
def gdp : ℝ := 121 * 10^12

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.21 * 10^14

-- Theorem stating that the GDP is equal to its scientific notation representation
theorem gdp_scientific_notation_equality : gdp = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l3494_349471


namespace NUMINAMATH_CALUDE_smallest_multiple_ten_satisfies_ten_is_smallest_l3494_349416

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 500 ∣ (450 * x) → x ≥ 10 := by
  sorry

theorem ten_satisfies : 500 ∣ (450 * 10) := by
  sorry

theorem ten_is_smallest : ∀ y : ℕ, y > 0 ∧ 500 ∣ (450 * y) → y ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_ten_satisfies_ten_is_smallest_l3494_349416


namespace NUMINAMATH_CALUDE_half_of_recipe_l3494_349489

theorem half_of_recipe (original_recipe : ℚ) (half_recipe : ℚ) : 
  original_recipe = 4.5 → half_recipe = original_recipe / 2 → half_recipe = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_half_of_recipe_l3494_349489


namespace NUMINAMATH_CALUDE_abcd_product_l3494_349408

theorem abcd_product (a b c d : ℝ) 
  (ha : a = Real.sqrt (4 + Real.sqrt (5 - a)))
  (hb : b = Real.sqrt (4 + Real.sqrt (5 + b)))
  (hc : c = Real.sqrt (4 - Real.sqrt (5 - c)))
  (hd : d = Real.sqrt (4 - Real.sqrt (5 + d))) :
  a * b * c * d = 11 := by
sorry

end NUMINAMATH_CALUDE_abcd_product_l3494_349408


namespace NUMINAMATH_CALUDE_ironman_age_l3494_349465

/-- Represents the ages of the characters in the problem -/
structure Ages where
  thor : ℕ
  captainAmerica : ℕ
  peterParker : ℕ
  ironman : ℕ

/-- The conditions of the problem -/
def problemConditions (ages : Ages) : Prop :=
  ages.thor = 13 * ages.captainAmerica ∧
  ages.captainAmerica = 7 * ages.peterParker ∧
  ages.ironman = ages.peterParker + 32 ∧
  ages.thor = 1456

/-- The theorem to be proved -/
theorem ironman_age (ages : Ages) :
  problemConditions ages → ages.ironman = 48 := by
  sorry

end NUMINAMATH_CALUDE_ironman_age_l3494_349465


namespace NUMINAMATH_CALUDE_flower_count_l3494_349439

theorem flower_count (bees : ℕ) (diff : ℕ) : bees = 3 → diff = 2 → bees + diff = 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l3494_349439


namespace NUMINAMATH_CALUDE_james_college_cost_l3494_349497

/-- The cost of James's community college units over 2 semesters -/
theorem james_college_cost (units_per_semester : ℕ) (cost_per_unit : ℕ) (num_semesters : ℕ) : 
  units_per_semester = 20 → cost_per_unit = 50 → num_semesters = 2 →
  units_per_semester * cost_per_unit * num_semesters = 2000 := by
  sorry

#check james_college_cost

end NUMINAMATH_CALUDE_james_college_cost_l3494_349497


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3494_349417

theorem weight_loss_challenge (initial_loss : ℝ) (measured_loss : ℝ) (clothes_addition : ℝ) :
  initial_loss = 0.14 →
  measured_loss = 0.1228 →
  (1 - measured_loss) * (1 - initial_loss) = 1 + clothes_addition →
  clothes_addition = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3494_349417


namespace NUMINAMATH_CALUDE_book_cost_price_l3494_349438

theorem book_cost_price (cost : ℝ) : cost = 300 :=
  by
  have h1 : 1.12 * cost + 18 = 1.18 * cost := by sorry
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3494_349438


namespace NUMINAMATH_CALUDE_power_difference_equality_l3494_349457

theorem power_difference_equality : (3^2)^3 - (2^3)^2 = 665 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l3494_349457


namespace NUMINAMATH_CALUDE_total_amount_is_117_l3494_349490

/-- Represents the distribution of money among three parties -/
structure Distribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total amount distributed -/
def total_amount (d : Distribution) : ℝ := d.x + d.y + d.z

/-- Theorem: Given the conditions, the total amount is 117 rupees -/
theorem total_amount_is_117 (d : Distribution) 
  (h1 : d.y = 27)  -- y's share is 27 rupees
  (h2 : d.y = 0.45 * d.x)  -- y gets 45 paisa for each rupee x gets
  (h3 : d.z = 0.50 * d.x)  -- z gets 50 paisa for each rupee x gets
  : total_amount d = 117 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_117_l3494_349490


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3494_349473

def triangle_vertex (y : ℝ) : ℝ × ℝ := (0, y)

theorem triangle_area_theorem (y : ℝ) (h1 : y < 0) :
  let v1 : ℝ × ℝ := (8, 6)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := triangle_vertex y
  let area : ℝ := (1/2) * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)
  area = 24 → y = -4.8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3494_349473


namespace NUMINAMATH_CALUDE_calculation_proof_l3494_349472

theorem calculation_proof (initial_amount : ℝ) (first_percentage : ℝ) 
  (discount_percentage : ℝ) (target_percentage : ℝ) (tax_rate : ℝ) : 
  initial_amount = 4000 ∧ 
  first_percentage = 0.15 ∧ 
  discount_percentage = 0.25 ∧ 
  target_percentage = 0.07 ∧ 
  tax_rate = 0.10 → 
  (1 + tax_rate) * (target_percentage * (1 - discount_percentage) * (first_percentage * initial_amount)) = 34.65 := by
sorry

#eval (1 + 0.10) * (0.07 * (1 - 0.25) * (0.15 * 4000))

end NUMINAMATH_CALUDE_calculation_proof_l3494_349472


namespace NUMINAMATH_CALUDE_tessellation_with_squares_and_triangles_l3494_349405

theorem tessellation_with_squares_and_triangles :
  ∀ m n : ℕ,
  (60 * m + 90 * n = 360) →
  (m = 3 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_tessellation_with_squares_and_triangles_l3494_349405


namespace NUMINAMATH_CALUDE_problem_solution_l3494_349431

theorem problem_solution (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 6) : 
  (a^2 + b^2 = 13) ∧ ((a - b)^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3494_349431


namespace NUMINAMATH_CALUDE_number_of_divisors_210_l3494_349479

theorem number_of_divisors_210 : Nat.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_210_l3494_349479


namespace NUMINAMATH_CALUDE_pens_left_for_lenny_l3494_349409

def total_pens : ℕ := 75 * 15

def friends_percentage : ℚ := 30 / 100
def classmates_percentage : ℚ := 20 / 100
def coworkers_percentage : ℚ := 25 / 100
def neighbors_percentage : ℚ := 15 / 100

def pens_after_friends : ℕ := total_pens - (Nat.floor (friends_percentage * total_pens))
def pens_after_classmates : ℕ := pens_after_friends - (Nat.floor (classmates_percentage * pens_after_friends))
def pens_after_coworkers : ℕ := pens_after_classmates - (Nat.floor (coworkers_percentage * pens_after_classmates))
def pens_after_neighbors : ℕ := pens_after_coworkers - (Nat.floor (neighbors_percentage * pens_after_coworkers))

theorem pens_left_for_lenny : pens_after_neighbors = 403 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_for_lenny_l3494_349409


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l3494_349456

theorem wicket_keeper_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 28)
  (h3 : team_avg_age = 25)
  (h4 : (team_size * team_avg_age - captain_age - wicket_keeper_age) / (team_size - 2) = team_avg_age - 1) :
  wicket_keeper_age - captain_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l3494_349456


namespace NUMINAMATH_CALUDE_f_one_equals_neg_log_four_l3494_349404

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≤ 0 then -x * Real.log (3 - x) else -(-x * Real.log (3 + x))

-- State the theorem
theorem f_one_equals_neg_log_four :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≤ 0, f x = -x * Real.log (3 - x)) →  -- definition for x ≤ 0
  f 1 = -Real.log 4 := by
sorry

end NUMINAMATH_CALUDE_f_one_equals_neg_log_four_l3494_349404


namespace NUMINAMATH_CALUDE_arrangement_with_A_middle_or_sides_arrangement_with_males_grouped_arrangement_with_males_not_grouped_arrangement_with_ABC_order_fixed_arrangement_A_not_left_B_not_right_arrangement_with_extra_female_no_adjacent_arrangement_in_two_rows_arrangement_with_person_between_A_and_B_l3494_349412

-- Common definitions
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- (1)
theorem arrangement_with_A_middle_or_sides :
  (3 * (total_students - 1).factorial) = 72 := by sorry

-- (2)
theorem arrangement_with_males_grouped :
  (male_students.factorial * (total_students - male_students + 1).factorial) = 36 := by sorry

-- (3)
theorem arrangement_with_males_not_grouped :
  (female_students.factorial * male_students.factorial) = 12 := by sorry

-- (4)
theorem arrangement_with_ABC_order_fixed :
  (total_students.factorial / male_students.factorial) = 20 := by sorry

-- (5)
theorem arrangement_A_not_left_B_not_right :
  ((total_students - 1) * (total_students - 1).factorial - 
   (total_students - 2) * (total_students - 2).factorial) = 78 := by sorry

-- (6)
def extra_female_student : ℕ := 1
def new_total_students : ℕ := total_students + extra_female_student

theorem arrangement_with_extra_female_no_adjacent :
  (male_students.factorial * (new_total_students - male_students + 1).factorial) = 144 := by sorry

-- (7)
theorem arrangement_in_two_rows :
  total_students.factorial = 120 := by sorry

-- (8)
theorem arrangement_with_person_between_A_and_B :
  (3 * 2 * male_students.factorial) = 36 := by sorry

end NUMINAMATH_CALUDE_arrangement_with_A_middle_or_sides_arrangement_with_males_grouped_arrangement_with_males_not_grouped_arrangement_with_ABC_order_fixed_arrangement_A_not_left_B_not_right_arrangement_with_extra_female_no_adjacent_arrangement_in_two_rows_arrangement_with_person_between_A_and_B_l3494_349412


namespace NUMINAMATH_CALUDE_a_power_six_bounds_l3494_349451

theorem a_power_six_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := by
  sorry

end NUMINAMATH_CALUDE_a_power_six_bounds_l3494_349451


namespace NUMINAMATH_CALUDE_f_properties_triangle_property_l3494_349437

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f x ≤ max_value) ∧
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧
    max_value = 2 ∧
    max_set = {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} := by sorry

theorem triangle_property :
  ∀ (A B C : ℝ) (a b c : ℝ),
    a = 1 → b = Real.sqrt 3 → f A = 2 →
    (A + B + C = Real.pi) →
    (Real.sin A / a = Real.sin B / b) →
    (Real.sin B / b = Real.sin C / c) →
    (C = Real.pi / 6 ∨ C = Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_f_properties_triangle_property_l3494_349437


namespace NUMINAMATH_CALUDE_problem_solution_l3494_349483

theorem problem_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a * Real.sin (π / 7) + b * Real.cos (π / 7)) / 
       (a * Real.cos (π / 7) - b * Real.sin (π / 7)) = Real.tan (10 * π / 21)) : 
  b / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3494_349483


namespace NUMINAMATH_CALUDE_ned_gave_away_13_games_l3494_349419

/-- The number of games Ned originally had -/
def original_games : ℕ := 19

/-- The number of games Ned currently has -/
def current_games : ℕ := 6

/-- The number of games Ned gave away -/
def games_given_away : ℕ := original_games - current_games

theorem ned_gave_away_13_games : games_given_away = 13 := by
  sorry

end NUMINAMATH_CALUDE_ned_gave_away_13_games_l3494_349419


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l3494_349469

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l3494_349469


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l3494_349434

theorem multiplication_of_powers (a : ℝ) : a * a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l3494_349434


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3494_349402

/-- A function satisfying f(a+b) = f(a) * f(b) for all real a and b, 
    and f(x) > 0 for all real x, with f(1) = 1/3 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) = f a * f b) ∧ 
  (∀ x : ℝ, f x > 0) ∧
  (f 1 = 1/3)

/-- If f satisfies the functional equation, then f(-2) = 9 -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : f (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3494_349402


namespace NUMINAMATH_CALUDE_solar_system_median_moons_l3494_349453

/-- Represents the number of moons for each planet in the solar system -/
def moon_counts : List Nat := [0, 1, 1, 3, 3, 6, 8, 14, 18, 21]

/-- Calculates the median of a sorted list with an even number of elements -/
def median (l : List Nat) : Rat :=
  let n := l.length
  if n % 2 = 0 then
    let mid := n / 2
    (l.get! (mid - 1) + l.get! mid) / 2
  else
    l.get! (n / 2)

theorem solar_system_median_moons :
  median moon_counts = 4.5 := by sorry

end NUMINAMATH_CALUDE_solar_system_median_moons_l3494_349453


namespace NUMINAMATH_CALUDE_isbn_problem_l3494_349443

/-- ISBN check digit calculation -/
def isbn_check_digit (A B C D E F G H I : ℕ) : ℕ :=
  let S := 10*A + 9*B + 8*C + 7*D + 6*E + 5*F + 4*G + 3*H + 2*I
  let r := S % 11
  if r = 0 then 0
  else if r = 1 then 10  -- Represented by 'x' in the problem
  else 11 - r

/-- The problem statement -/
theorem isbn_problem (y : ℕ) : 
  isbn_check_digit 9 6 2 y 7 0 7 0 1 = 5 → y = 7 := by
sorry

end NUMINAMATH_CALUDE_isbn_problem_l3494_349443


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3494_349452

theorem arithmetic_mean_of_fractions : 
  (1 / 3 : ℚ) * ((3 / 7 : ℚ) + (5 / 9 : ℚ) + (2 / 3 : ℚ)) = 104 / 189 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3494_349452


namespace NUMINAMATH_CALUDE_amir_weight_l3494_349461

theorem amir_weight (bulat_weight ilnur_weight : ℝ) : 
  let amir_weight := ilnur_weight + 8
  let daniyar_weight := bulat_weight + 4
  -- The sum of the weights of the heaviest and lightest boys is 2 kg less than the sum of the weights of the other two boys
  (amir_weight + ilnur_weight = daniyar_weight + bulat_weight - 2) →
  -- All four boys together weigh 250 kg
  (amir_weight + ilnur_weight + daniyar_weight + bulat_weight = 250) →
  amir_weight = 67 := by
sorry

end NUMINAMATH_CALUDE_amir_weight_l3494_349461


namespace NUMINAMATH_CALUDE_conditional_equivalence_l3494_349423

theorem conditional_equivalence (R S : Prop) :
  (¬R → S) ↔ (¬S → R) := by sorry

end NUMINAMATH_CALUDE_conditional_equivalence_l3494_349423


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3494_349476

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 80 ∧ n % 8 = 5 ∧ ∀ m, m < 80 ∧ m % 8 = 5 → m ≤ n → n = 77 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3494_349476


namespace NUMINAMATH_CALUDE_line_parallel_to_polar_axis_l3494_349445

/-- Given a point P in polar coordinates (r, θ), prove that the equation r * sin(θ) = 1
    represents a line that passes through P and is parallel to the polar axis. -/
theorem line_parallel_to_polar_axis 
  (r : ℝ) (θ : ℝ) (h1 : r = 2) (h2 : θ = π / 6) :
  r * Real.sin θ = 1 := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_polar_axis_l3494_349445


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3494_349493

/-- Given a mixture where water is 10% of the total volume, if adding 14 liters of water
    results in a new mixture with 25% water, then the initial volume of the mixture was 70 liters. -/
theorem initial_mixture_volume (V : ℝ) : 
  (0.1 * V + 14) / (V + 14) = 0.25 → V = 70 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l3494_349493


namespace NUMINAMATH_CALUDE_nancy_crystal_sets_l3494_349460

/-- The cost of one set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of one set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends in dollars -/
def total_spent : ℕ := 29

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

theorem nancy_crystal_sets : 
  crystal_cost * crystal_sets + metal_cost * metal_sets = total_spent := by
  sorry

end NUMINAMATH_CALUDE_nancy_crystal_sets_l3494_349460


namespace NUMINAMATH_CALUDE_journey_proof_l3494_349468

/-- Represents the distance-time relationship for a journey -/
def distance_from_destination (total_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  total_distance - speed * time

theorem journey_proof (total_distance : ℝ) (speed : ℝ) (time : ℝ) 
  (h1 : total_distance = 174)
  (h2 : speed = 60)
  (h3 : time = 1.5) :
  distance_from_destination total_distance speed time = 84 := by
  sorry

#check journey_proof

end NUMINAMATH_CALUDE_journey_proof_l3494_349468


namespace NUMINAMATH_CALUDE_selection_methods_count_l3494_349411

def num_male_students : ℕ := 5
def num_female_students : ℕ := 4
def num_representatives : ℕ := 4
def min_female_representatives : ℕ := 2

theorem selection_methods_count :
  (Finset.sum (Finset.range (num_representatives - min_female_representatives + 1))
    (λ k => Nat.choose num_female_students (min_female_representatives + k) *
            Nat.choose num_male_students (num_representatives - (min_female_representatives + k))))
  = 81 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l3494_349411


namespace NUMINAMATH_CALUDE_vector_calculation_l3494_349449

/-- Given vectors a, b, and c in ℝ³, prove that a + 2b - 3c equals (-7, -1, -1) -/
theorem vector_calculation (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (2, 0, 1)) 
  (hb : b = (-3, 1, -1)) 
  (hc : c = (1, 1, 0)) : 
  a + 2 • b - 3 • c = (-7, -1, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l3494_349449


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3494_349450

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  (1/2) * a * b = (1/2) * c →  -- Area condition
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3494_349450


namespace NUMINAMATH_CALUDE_race_win_probability_l3494_349447

theorem race_win_probability (pA pB pC pD pE : ℚ) 
  (hA : pA = 1/8) (hB : pB = 1/12) (hC : pC = 1/15) (hD : pD = 1/18) (hE : pE = 1/20)
  (h_mutually_exclusive : ∀ (x y : Fin 5), x ≠ y → pA + pB + pC + pD + pE ≤ 1) :
  pA + pB + pC + pD + pE = 137/360 := by
sorry

end NUMINAMATH_CALUDE_race_win_probability_l3494_349447


namespace NUMINAMATH_CALUDE_treasury_problem_l3494_349463

theorem treasury_problem (T : ℚ) : 
  (T - T / 13 - (T - T / 13) / 17 = 150) → 
  T = 172 + 21 / 32 :=
by sorry

end NUMINAMATH_CALUDE_treasury_problem_l3494_349463


namespace NUMINAMATH_CALUDE_somu_present_age_l3494_349444

/-- Somu's present age -/
def somu_age : ℕ := sorry

/-- Somu's father's present age -/
def father_age : ℕ := sorry

/-- Somu's age is one-third of his father's age -/
axiom current_age_ratio : somu_age = father_age / 3

/-- 9 years ago, Somu was one-fifth of his father's age -/
axiom past_age_ratio : somu_age - 9 = (father_age - 9) / 5

theorem somu_present_age : somu_age = 18 := by sorry

end NUMINAMATH_CALUDE_somu_present_age_l3494_349444


namespace NUMINAMATH_CALUDE_limit_proof_l3494_349406

def a_n (n : ℕ) : ℚ := (7 * n - 1) / (n + 1)

theorem limit_proof (ε : ℚ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |a_n n - 7| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l3494_349406


namespace NUMINAMATH_CALUDE_f_sum_constant_l3494_349436

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem f_sum_constant (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_constant_l3494_349436


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l3494_349440

theorem smallest_perfect_square_divisible_by_5_and_7 :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), n = k^2) ∧ 5 ∣ n ∧ 7 ∣ n ∧
  ∀ (m : ℕ), m > 0 → (∃ (j : ℕ), m = j^2) → 5 ∣ m → 7 ∣ m → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l3494_349440


namespace NUMINAMATH_CALUDE_dice_product_120_probability_l3494_349407

/-- A function representing a standard die roll --/
def standardDie : ℕ → Prop :=
  λ n => 1 ≤ n ∧ n ≤ 6

/-- The probability of a specific outcome when rolling three dice --/
def tripleRollProb : ℚ := (1 : ℚ) / 216

/-- The number of favorable outcomes --/
def favorableOutcomes : ℕ := 6

/-- The probability that the product of three dice rolls equals 120 --/
theorem dice_product_120_probability :
  (favorableOutcomes : ℚ) * tripleRollProb = (1 : ℚ) / 36 :=
sorry

end NUMINAMATH_CALUDE_dice_product_120_probability_l3494_349407


namespace NUMINAMATH_CALUDE_unique_solution_l3494_349454

/-- The price of Lunasa's violin -/
def violin_price : ℝ := sorry

/-- The price of Merlin's trumpet -/
def trumpet_price : ℝ := sorry

/-- The price of Lyrica's piano -/
def piano_price : ℝ := sorry

/-- Condition (a): If violin price is raised by 50% and trumpet price is decreased by 50%,
    violin is $50 more expensive than trumpet -/
axiom condition_a : 1.5 * violin_price = 0.5 * trumpet_price + 50

/-- Condition (b): If trumpet price is raised by 50% and piano price is decreased by 50%,
    trumpet is $50 more expensive than piano -/
axiom condition_b : 1.5 * trumpet_price = 0.5 * piano_price + 50

/-- The percentage m by which violin price is raised and piano price is decreased -/
def m : ℤ := sorry

/-- The price difference n between the adjusted violin and piano prices -/
def n : ℤ := sorry

/-- The relationship between adjusted violin and piano prices -/
axiom price_relationship : (100 + m) * violin_price / 100 = n + (100 - m) * piano_price / 100

theorem unique_solution : m = 80 ∧ n = 80 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3494_349454


namespace NUMINAMATH_CALUDE_ice_cream_fraction_l3494_349403

theorem ice_cream_fraction (initial_amount : ℚ) (lunch_cost : ℚ) (ice_cream_cost : ℚ) : 
  initial_amount = 30 →
  lunch_cost = 10 →
  ice_cream_cost = 5 →
  ice_cream_cost / (initial_amount - lunch_cost) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_fraction_l3494_349403


namespace NUMINAMATH_CALUDE_least_positive_four_digit_octal_l3494_349433

/-- The number of digits required to represent a positive integer in a given base -/
def numDigits (n : ℕ+) (base : ℕ) : ℕ :=
  Nat.log base n.val + 1

/-- Checks if a number requires at least four digits in base 8 -/
def requiresFourDigitsOctal (n : ℕ+) : Prop :=
  numDigits n 8 ≥ 4

theorem least_positive_four_digit_octal :
  ∃ (n : ℕ+), requiresFourDigitsOctal n ∧
    ∀ (m : ℕ+), m < n → ¬requiresFourDigitsOctal m ∧
    n = 512 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_four_digit_octal_l3494_349433


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3494_349495

/-- A quadratic function with a negative leading coefficient -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry (b c : ℝ) : ℝ := 2

theorem quadratic_inequality (b c : ℝ) :
  f b c (axis_of_symmetry b c + 2) < f b c (axis_of_symmetry b c - 1) ∧
  f b c (axis_of_symmetry b c - 1) < f b c (axis_of_symmetry b c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3494_349495


namespace NUMINAMATH_CALUDE_complex_equality_implies_ratio_one_l3494_349486

theorem complex_equality_implies_ratio_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.I : ℂ)^4 = -1 → (a + b * Complex.I)^4 = (a - b * Complex.I)^4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_ratio_one_l3494_349486


namespace NUMINAMATH_CALUDE_truncated_cone_complete_height_l3494_349487

/-- Given a truncated cone with height h, upper radius r, and lower radius R,
    the height H of the corresponding complete cone is hR / (R - r) -/
theorem truncated_cone_complete_height
  (h r R : ℝ) (h_pos : h > 0) (r_pos : r > 0) (R_pos : R > 0) (r_lt_R : r < R) :
  ∃ H : ℝ, H = h * R / (R - r) ∧ H > h := by
  sorry

#check truncated_cone_complete_height

end NUMINAMATH_CALUDE_truncated_cone_complete_height_l3494_349487
