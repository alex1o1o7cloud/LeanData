import Mathlib

namespace NUMINAMATH_CALUDE_doris_earnings_l703_70338

/-- Represents the babysitting scenario for Doris -/
structure BabysittingScenario where
  hourlyRate : ℕ  -- Hourly rate in dollars
  weekdayHours : ℕ  -- Hours worked on each weekday
  saturdayHours : ℕ  -- Hours worked on Saturday
  targetEarnings : ℕ  -- Target earnings in dollars

/-- Calculates the number of weeks needed to reach the target earnings -/
def weeksToReachTarget (scenario : BabysittingScenario) : ℕ :=
  let weeklyEarnings := scenario.hourlyRate * (5 * scenario.weekdayHours + scenario.saturdayHours)
  (scenario.targetEarnings + weeklyEarnings - 1) / weeklyEarnings

/-- Theorem stating that Doris needs 3 weeks to reach her target earnings -/
theorem doris_earnings (scenario : BabysittingScenario) 
  (h1 : scenario.hourlyRate = 20)
  (h2 : scenario.weekdayHours = 3)
  (h3 : scenario.saturdayHours = 5)
  (h4 : scenario.targetEarnings = 1200) :
  weeksToReachTarget scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_doris_earnings_l703_70338


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l703_70355

/-- Given a point M with coordinates (3, -4), its symmetric point with respect to the x-axis has coordinates (3, 4) -/
theorem symmetric_point_wrt_x_axis :
  let M : ℝ × ℝ := (3, -4)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point M = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l703_70355


namespace NUMINAMATH_CALUDE_intersection_properties_y₁_gt_y₂_l703_70332

/-- The quadratic function y₁ -/
def y₁ (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 3

/-- The linear function y₂ -/
def y₂ (x : ℝ) : ℝ := x + 1

/-- Theorem stating the properties of the intersection points and the resulting quadratic function -/
theorem intersection_properties :
  ∀ b m : ℝ,
  (y₁ b (-1) = y₂ (-1)) →
  (y₁ b 4 = y₂ 4) →
  (y₁ b (-1) = 0) →
  (y₁ b 4 = m) →
  (b = -2 ∧ m = 5) :=
sorry

/-- Theorem stating when y₁ > y₂ -/
theorem y₁_gt_y₂ :
  ∀ x : ℝ,
  (y₁ (-2) x > y₂ x) ↔ (x < -1 ∨ x > 4) :=
sorry

end NUMINAMATH_CALUDE_intersection_properties_y₁_gt_y₂_l703_70332


namespace NUMINAMATH_CALUDE_gretchen_weekend_profit_l703_70378

/-- Calculates Gretchen's profit from drawing caricatures over a weekend -/
def weekend_profit (
  full_body_price : ℕ)  -- Price of a full-body caricature
  (face_only_price : ℕ) -- Price of a face-only caricature
  (full_body_count : ℕ) -- Number of full-body caricatures drawn on Saturday
  (face_only_count : ℕ) -- Number of face-only caricatures drawn on Sunday
  (hourly_park_fee : ℕ) -- Hourly park fee
  (hours_per_day : ℕ)   -- Hours worked per day
  (art_supplies_cost : ℕ) -- Daily cost of art supplies
  : ℕ :=
  let total_revenue := full_body_price * full_body_count + face_only_price * face_only_count
  let total_park_fee := hourly_park_fee * hours_per_day * 2
  let total_supplies_cost := art_supplies_cost * 2
  let total_expenses := total_park_fee + total_supplies_cost
  total_revenue - total_expenses

/-- Theorem stating Gretchen's profit for the weekend -/
theorem gretchen_weekend_profit :
  weekend_profit 25 15 24 16 5 6 8 = 764 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_weekend_profit_l703_70378


namespace NUMINAMATH_CALUDE_euler_triangle_inequality_l703_70394

/-- 
For any triangle, let:
  r : radius of the incircle
  R : radius of the circumcircle
  d : distance between the incenter and circumcenter

Then, R ≥ 2r
-/
theorem euler_triangle_inequality (r R d : ℝ) : r > 0 → R > 0 → d > 0 → R ≥ 2 * r := by
  sorry

end NUMINAMATH_CALUDE_euler_triangle_inequality_l703_70394


namespace NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l703_70311

/-- Two lines in the form y = kx + l are parallel if and only if 
    they have the same slope but different y-intercepts -/
theorem lines_parallel_iff_same_slope_diff_intercept 
  (k₁ k₂ l₁ l₂ : ℝ) : 
  (∀ x y : ℝ, y = k₁ * x + l₁ ↔ y = k₂ * x + l₂) ↔ 
  (k₁ = k₂ ∧ l₁ ≠ l₂) :=
by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l703_70311


namespace NUMINAMATH_CALUDE_square_root_of_256_l703_70330

theorem square_root_of_256 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 256) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_256_l703_70330


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l703_70327

theorem arithmetic_sequence_count : 
  ∀ (a d last : ℕ) (n : ℕ),
    a = 2 →
    d = 4 →
    last = 2018 →
    last = a + (n - 1) * d →
    n = 505 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l703_70327


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l703_70371

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 15*x + 35

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! c : ℝ, g c = c ∧ c = -5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l703_70371


namespace NUMINAMATH_CALUDE_one_carton_per_case_l703_70324

/-- The number of cartons in a case -/
def cartons_per_case : ℕ := 1

/-- The number of boxes in each carton -/
def boxes_per_carton : ℕ := 1

/-- The number of paper clips in each box -/
def clips_per_box : ℕ := 500

/-- The total number of paper clips in two cases -/
def total_clips : ℕ := 1000

/-- Theorem stating that there is exactly one carton in a case -/
theorem one_carton_per_case :
  (∀ b : ℕ, b > 0 → 2 * cartons_per_case * b * clips_per_box = total_clips) →
  cartons_per_case = 1 :=
by sorry

end NUMINAMATH_CALUDE_one_carton_per_case_l703_70324


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l703_70308

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distanceToXAxis (y : ℝ) : ℝ := |y|

/-- Point P in the Cartesian coordinate system -/
def P : ℝ × ℝ := (4, -3)

/-- Theorem: The distance from point P(4, -3) to the x-axis is 3 -/
theorem distance_P_to_x_axis :
  distanceToXAxis P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l703_70308


namespace NUMINAMATH_CALUDE_S_is_closed_closed_set_contains_zero_l703_70318

-- Define a closed set
def ClosedSet (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

-- Define the set S
def S : Set ℝ := {x | ∃ a b : ℤ, x = a + b * Real.sqrt 3}

-- Theorem 1: S is a closed set
theorem S_is_closed : ClosedSet S := sorry

-- Theorem 2: Any closed set contains 0
theorem closed_set_contains_zero (T : Set ℝ) (h : ClosedSet T) : (0 : ℝ) ∈ T := sorry

end NUMINAMATH_CALUDE_S_is_closed_closed_set_contains_zero_l703_70318


namespace NUMINAMATH_CALUDE_same_solution_for_both_systems_l703_70387

theorem same_solution_for_both_systems :
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 2*x + 3*y = 7 ∧ 3*x - 2*y = 4 ∧ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_same_solution_for_both_systems_l703_70387


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l703_70367

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x - 1/2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := -1/2

theorem quadratic_discriminant : discriminant a b c = 161/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l703_70367


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l703_70376

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, |x + 5| ≥ m + 2) → m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l703_70376


namespace NUMINAMATH_CALUDE_problem_statement_l703_70385

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 2)^2 + 25/((x - 2)^2) = -x + 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l703_70385


namespace NUMINAMATH_CALUDE_b_value_range_l703_70305

theorem b_value_range (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (b_min b_max : ℝ), 
    (∀ b', (∃ a' c', a' + b' + c' = 3 ∧ a'^2 + b'^2 + c'^2 = 18) → b_min ≤ b' ∧ b' ≤ b_max) ∧
    b_max - b_min = 2 * Real.sqrt (45/4) :=
sorry

end NUMINAMATH_CALUDE_b_value_range_l703_70305


namespace NUMINAMATH_CALUDE_product_eleven_sum_possibilities_l703_70377

theorem product_eleven_sum_possibilities (a b c : ℤ) : 
  a * b * c = -11 → (a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_product_eleven_sum_possibilities_l703_70377


namespace NUMINAMATH_CALUDE_number_division_problem_l703_70306

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7) 
  (h2 : (x - 6) / y = 6) : 
  y = 8 := by sorry

end NUMINAMATH_CALUDE_number_division_problem_l703_70306


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l703_70316

theorem matrix_equation_proof :
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![16/7, -36/7; -12/7, 27/7]
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, 5; 16, -4]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; -4, 1]
  N * A = B + C := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l703_70316


namespace NUMINAMATH_CALUDE_sum_of_q_p_x_values_l703_70345

def p (x : ℝ) : ℝ := |x| + 1

def q (x : ℝ) : ℝ := -x^2

def x_values : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_q_p_x_values :
  (x_values.map (λ x => q (p x))).sum = -59 := by sorry

end NUMINAMATH_CALUDE_sum_of_q_p_x_values_l703_70345


namespace NUMINAMATH_CALUDE_angle_equality_l703_70398

theorem angle_equality (θ : Real) (A B : Set Real) : 
  A = {1, Real.cos θ} → B = {1/2, 1} → A = B → 0 < θ → θ < π/2 → θ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l703_70398


namespace NUMINAMATH_CALUDE_floor_nested_equation_l703_70370

theorem floor_nested_equation (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_nested_equation_l703_70370


namespace NUMINAMATH_CALUDE_probability_no_adjacent_fir_trees_proof_l703_70397

def num_pine_trees : ℕ := 2
def num_cedar_trees : ℕ := 3
def num_fir_trees : ℕ := 4
def total_trees : ℕ := num_pine_trees + num_cedar_trees + num_fir_trees

def probability_no_adjacent_fir_trees : ℚ := 5 / 42

theorem probability_no_adjacent_fir_trees_proof :
  let non_fir_trees := num_pine_trees + num_cedar_trees
  let total_arrangements := Nat.choose total_trees num_fir_trees
  let valid_arrangements := Nat.choose (non_fir_trees + 1) num_fir_trees
  (valid_arrangements : ℚ) / total_arrangements = probability_no_adjacent_fir_trees :=
sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_fir_trees_proof_l703_70397


namespace NUMINAMATH_CALUDE_west_movement_representation_l703_70374

/-- Represents direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Converts a movement to its coordinate representation --/
def toCoordinate (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

theorem west_movement_representation :
  let westMovement : Movement := ⟨80, Direction.West⟩
  toCoordinate westMovement = -80 := by sorry

end NUMINAMATH_CALUDE_west_movement_representation_l703_70374


namespace NUMINAMATH_CALUDE_constant_proof_l703_70343

theorem constant_proof (n : ℤ) (c : ℝ) : 
  (∀ k : ℤ, c * k^2 ≤ 12100 → k ≤ 10) →
  (c * 10^2 ≤ 12100) →
  c = 121 := by
  sorry

end NUMINAMATH_CALUDE_constant_proof_l703_70343


namespace NUMINAMATH_CALUDE_third_angle_measure_l703_70335

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem statement
theorem third_angle_measure (t : Triangle) :
  is_valid_triangle t → t.angle1 = 25 → t.angle2 = 70 → t.angle3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_third_angle_measure_l703_70335


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l703_70360

theorem a_gt_abs_b_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) := by
sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l703_70360


namespace NUMINAMATH_CALUDE_bonus_difference_l703_70357

/-- Prove that given a total bonus of $5,000 divided between two employees,
    where the senior employee receives $1,900 and the junior employee receives $3,100,
    the difference between the junior employee's bonus and the senior employee's bonus is $1,200. -/
theorem bonus_difference (total_bonus senior_bonus junior_bonus : ℕ) : 
  total_bonus = 5000 →
  senior_bonus = 1900 →
  junior_bonus = 3100 →
  junior_bonus - senior_bonus = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bonus_difference_l703_70357


namespace NUMINAMATH_CALUDE_defective_units_percentage_l703_70339

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 4

/-- The percentage of total units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.32

/-- The percentage of total units that are defective -/
def total_defective_percent : ℝ := 8

theorem defective_units_percentage :
  defective_shipped_percent * total_defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l703_70339


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l703_70372

theorem complex_equation_real_solution (a : ℝ) : 
  (((2 * a) / (1 + Complex.I) + 1 + Complex.I).im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l703_70372


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l703_70389

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  C = 2 * B →        -- Angle C is double angle B
  A = 3 * B →        -- Angle A is thrice angle B
  B = 30 :=          -- Angle B is 30°
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l703_70389


namespace NUMINAMATH_CALUDE_car_average_speed_l703_70300

/-- Calculates the average speed of a car over two hours given specific conditions --/
theorem car_average_speed 
  (speed_first_hour : ℝ)
  (headwind_speed : ℝ)
  (speed_second_hour : ℝ)
  (h_speed_first : speed_first_hour = 90)
  (h_headwind : headwind_speed = 10)
  (h_speed_second : speed_second_hour = 55) :
  (speed_first_hour + headwind_speed + speed_second_hour) / 2 = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l703_70300


namespace NUMINAMATH_CALUDE_stating_sum_of_intersections_theorem_l703_70322

/-- The number of lines passing through the origin -/
def num_lines : ℕ := 180

/-- The angle between each line in degrees -/
def angle_between : ℝ := 1

/-- The equation of the line that intersects with all other lines -/
def intersecting_line (x : ℝ) : ℝ := 100 - x

/-- The sum of x-coordinates of intersection points -/
def sum_of_intersections : ℝ := 8950

/-- 
Theorem stating that the sum of x-coordinates of intersections between 
180 lines passing through the origin (forming 1 degree angles) and the 
line y = 100 - x is equal to 8950.
-/
theorem sum_of_intersections_theorem :
  let lines := List.range num_lines
  let intersection_points := lines.map (λ i => 
    let angle := i * angle_between
    let m := Real.tan (angle * π / 180)
    100 / (1 + m))
  intersection_points.sum = sum_of_intersections := by
  sorry


end NUMINAMATH_CALUDE_stating_sum_of_intersections_theorem_l703_70322


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l703_70302

/-- 
For a quadratic equation x^2 + x + m = 0 with m ∈ ℝ, 
the condition "m > 1/4" is neither sufficient nor necessary for real roots.
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  ¬(∀ x : ℝ, x^2 + x + m = 0 → m > 1/4) ∧ 
  ¬(m > 1/4 → ∃ x : ℝ, x^2 + x + m = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l703_70302


namespace NUMINAMATH_CALUDE_money_distribution_l703_70369

theorem money_distribution (a b c d total : ℕ) : 
  a + b + c + d = total →
  2 * a = b →
  5 * a = 2 * c →
  a = d →
  a + b = 1800 →
  total = 4500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l703_70369


namespace NUMINAMATH_CALUDE_car_trip_distance_theorem_l703_70399

/-- Represents a segment of a car trip with speed and duration -/
structure TripSegment where
  speed : ℝ  -- Speed in miles per hour
  duration : ℝ  -- Duration in hours

/-- Calculates the distance traveled for a trip segment -/
def distance_traveled (segment : TripSegment) : ℝ :=
  segment.speed * segment.duration

/-- Represents a car trip with multiple segments -/
def CarTrip : Type := List TripSegment

/-- Calculates the total distance traveled for a car trip -/
def total_distance (trip : CarTrip) : ℝ :=
  trip.map distance_traveled |>.sum

theorem car_trip_distance_theorem (trip : CarTrip) : 
  trip = [
    { speed := 65, duration := 3 },
    { speed := 45, duration := 2 },
    { speed := 55, duration := 4 }
  ] → total_distance trip = 505 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_theorem_l703_70399


namespace NUMINAMATH_CALUDE_solve_equation_l703_70340

theorem solve_equation (x : ℝ) (h : (0.12 / x) * 2 = 12) : x = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l703_70340


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l703_70361

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -1/2x - 2 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -1/2 * x - 2
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = 1/2 * x - 3/2) →  -- Slope of L1 is 1/2
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (x₂ - x₁) * (-1/2) = -(y₂ - y₁) / (x₂ - x₁)) →  -- L1 and L2 are perpendicular
  L2 x y
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l703_70361


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l703_70365

/-- Given a rhombus with area 150 cm² and one diagonal of length 10 cm, 
    prove that the length of the other diagonal is 30 cm. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 150)
  (h_d1 : d1 = 10)
  (h_rhombus_area : area = (d1 * d2) / 2) :
  d2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l703_70365


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_equal_orthocenter_quadrilateral_l703_70325

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Definition of an inscribed quadrilateral -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Definition of an orthocenter of a triangle -/
def isOrthocenter (H : Point) (A B C : Point) : Prop :=
  sorry

/-- Definition of equality between quadrilaterals -/
def quadrilateralEqual (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem inscribed_quadrilateral_equal_orthocenter_quadrilateral 
  (A₁ A₂ A₃ A₄ H₁ H₂ H₃ H₄ : Point) :
  isInscribed (Quadrilateral.mk A₁ A₂ A₃ A₄) →
  isOrthocenter H₁ A₂ A₃ A₄ →
  isOrthocenter H₂ A₁ A₃ A₄ →
  isOrthocenter H₃ A₁ A₂ A₄ →
  isOrthocenter H₄ A₁ A₂ A₃ →
  quadrilateralEqual 
    (Quadrilateral.mk A₁ A₂ A₃ A₄) 
    (Quadrilateral.mk H₁ H₂ H₃ H₄) :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_equal_orthocenter_quadrilateral_l703_70325


namespace NUMINAMATH_CALUDE_bruce_payment_l703_70390

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grapes_quantity grapes_rate mangoes_quantity mangoes_rate : ℕ) : ℕ :=
  grapes_quantity * grapes_rate + mangoes_quantity * mangoes_rate

/-- Theorem stating that Bruce paid 1055 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l703_70390


namespace NUMINAMATH_CALUDE_peach_pie_slices_count_l703_70359

/-- Represents the number of slices in a peach pie -/
def peach_pie_slices : ℕ := sorry

/-- Represents the number of slices in an apple pie -/
def apple_pie_slices : ℕ := 8

/-- Represents the number of customers who ordered apple pie slices -/
def apple_pie_customers : ℕ := 56

/-- Represents the number of customers who ordered peach pie slices -/
def peach_pie_customers : ℕ := 48

/-- Represents the total number of pies sold during the weekend -/
def total_pies_sold : ℕ := 15

theorem peach_pie_slices_count : peach_pie_slices = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_pie_slices_count_l703_70359


namespace NUMINAMATH_CALUDE_distribute_5_2_l703_70312

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_5_2 : distribute 5 2 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_5_2_l703_70312


namespace NUMINAMATH_CALUDE_sector_max_area_l703_70392

/-- Given a sector with perimeter 4, its area is maximized when the central angle equals 2 -/
theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 4) :
  let α := l / r
  let area := (1 / 2) * r * l
  (∀ r' l', 2 * r' + l' = 4 → (1 / 2) * r' * l' ≤ area) →
  α = 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l703_70392


namespace NUMINAMATH_CALUDE_parallelogram_height_l703_70348

theorem parallelogram_height (area base height : ℝ) : 
  area = 120 ∧ base = 12 ∧ area = base * height → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l703_70348


namespace NUMINAMATH_CALUDE_sets_equality_l703_70315

def M : Set ℤ := {u : ℤ | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u : ℤ | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem sets_equality : M = N :=
sorry

end NUMINAMATH_CALUDE_sets_equality_l703_70315


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l703_70358

/-- The focal length of a hyperbola -/
def focal_length (c : ℝ) : ℝ := 2 * c

/-- The equation of a hyperbola -/
def is_hyperbola (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

theorem hyperbola_a_value (a : ℝ) :
  a > 0 →
  focal_length (Real.sqrt 10) = 2 * Real.sqrt 10 →
  is_hyperbola a (Real.sqrt 6) (Real.sqrt 10) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l703_70358


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l703_70310

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem
theorem perpendicular_line_equation (A B C : ℝ) (P₀ : Point2D) :
  let L₁ : Line := { a := A, b := B, c := C }
  let L₂ : Line := { a := B, b := -A, c := -B * P₀.x + A * P₀.y }
  (∀ (x y : ℝ), A * x + B * y + C = 0 → L₁.a * x + L₁.b * y + L₁.c = 0) →
  (L₂.a * P₀.x + L₂.b * P₀.y + L₂.c = 0) →
  (∀ (x y : ℝ), B * x - A * y - B * P₀.x + A * P₀.y = 0 ↔ L₂.a * x + L₂.b * y + L₂.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l703_70310


namespace NUMINAMATH_CALUDE_unique_power_sum_l703_70380

theorem unique_power_sum (k : ℕ) : (∃ (n t : ℕ), t ≥ 2 ∧ 3^k + 5^k = n^t) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_power_sum_l703_70380


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l703_70353

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_condition : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l703_70353


namespace NUMINAMATH_CALUDE_solar_panel_installation_l703_70381

/-- The number of homes that can have solar panels fully installed given the total number of homes,
    panels required per home, and the shortage in supplied panels. -/
def homes_with_panels (total_homes : ℕ) (panels_per_home : ℕ) (panel_shortage : ℕ) : ℕ :=
  ((total_homes * panels_per_home - panel_shortage) / panels_per_home)

/-- Theorem stating that given 20 homes, each requiring 10 solar panels, and a supplier bringing
    50 panels less than required, the number of homes that can have their panels fully installed is 15. -/
theorem solar_panel_installation :
  homes_with_panels 20 10 50 = 15 := by
  sorry

#eval homes_with_panels 20 10 50

end NUMINAMATH_CALUDE_solar_panel_installation_l703_70381


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l703_70383

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧ (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l703_70383


namespace NUMINAMATH_CALUDE_unique_real_root_of_polynomial_l703_70373

theorem unique_real_root_of_polynomial (x : ℝ) :
  x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_root_of_polynomial_l703_70373


namespace NUMINAMATH_CALUDE_square_side_length_l703_70351

/-- Given a square with diagonal length 4, prove that its side length is 2√2. -/
theorem square_side_length (d : ℝ) (h : d = 4) : 
  ∃ s : ℝ, s > 0 ∧ s * s * 2 = d * d ∧ s = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l703_70351


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l703_70393

theorem point_in_second_quadrant (θ : Real) (h : π/2 < θ ∧ θ < π) :
  let P := (Real.tan θ, Real.sin θ)
  P.1 < 0 ∧ P.2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l703_70393


namespace NUMINAMATH_CALUDE_first_number_value_l703_70382

-- Define the custom operation
def custom_op (m n : ℤ) : ℤ := n^2 - m

-- Theorem statement
theorem first_number_value :
  ∃ x : ℤ, custom_op x 3 = 6 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l703_70382


namespace NUMINAMATH_CALUDE_equation_solution_l703_70384

theorem equation_solution (m : ℕ) : 
  ((1^m : ℚ) / (5^m)) * ((1^16 : ℚ) / (4^16)) = 1 / (2 * (10^31)) → m = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l703_70384


namespace NUMINAMATH_CALUDE_ians_money_left_l703_70350

/-- Calculates Ian's remaining money after expenses and taxes -/
def ians_remaining_money (total_hours : ℕ) (first_rate second_rate : ℚ) 
  (spending_ratio tax_rate : ℚ) (monthly_expense : ℚ) : ℚ :=
  let total_earnings := (first_rate * (total_hours / 2 : ℚ)) + (second_rate * (total_hours / 2 : ℚ))
  let spending := total_earnings * spending_ratio
  let taxes := total_earnings * tax_rate
  let total_deductions := spending + taxes + monthly_expense
  total_earnings - total_deductions

theorem ians_money_left :
  ians_remaining_money 8 18 22 (1/2) (1/10) 50 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ians_money_left_l703_70350


namespace NUMINAMATH_CALUDE_odd_numbers_properties_l703_70336

theorem odd_numbers_properties (x y : ℤ) (hx : ∃ k : ℤ, x = 2 * k + 1) (hy : ∃ k : ℤ, y = 2 * k + 1) :
  (∃ m : ℤ, x + y = 2 * m) ∧ 
  (∃ n : ℤ, x - y = 2 * n) ∧ 
  (∃ p : ℤ, x * y = 2 * p + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_numbers_properties_l703_70336


namespace NUMINAMATH_CALUDE_joshuas_share_l703_70313

theorem joshuas_share (total : ℕ) (joshua_share : ℕ) (justin_share : ℕ) : 
  total = 40 → 
  joshua_share = 3 * justin_share → 
  total = joshua_share + justin_share → 
  joshua_share = 30 := by
sorry

end NUMINAMATH_CALUDE_joshuas_share_l703_70313


namespace NUMINAMATH_CALUDE_not_ellipse_for_certain_m_l703_70396

/-- The equation of the curve -/
def curve_equation (m x y : ℝ) : Prop :=
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m)

/-- Definition of an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  1 < m ∧ m < 3 ∧ m ≠ 2

/-- The theorem to be proved -/
theorem not_ellipse_for_certain_m :
  ∀ m : ℝ, (m ≤ 1 ∨ m = 2 ∨ m ≥ 3) →
    ¬(is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_not_ellipse_for_certain_m_l703_70396


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l703_70352

theorem profit_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 60 → selling_price = 63 → 
  (selling_price - cost_price) / cost_price * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l703_70352


namespace NUMINAMATH_CALUDE_unsold_books_l703_70362

theorem unsold_books (total_books : ℕ) : 
  (2 : ℚ) / 3 * total_books * 2 = 144 → 
  (1 : ℚ) / 3 * total_books = 36 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l703_70362


namespace NUMINAMATH_CALUDE_invariant_preserved_cannot_transform_l703_70321

/-- Represents a letter in the English alphabet -/
def Letter := Fin 26

/-- A 4x4 matrix of letters -/
def LetterMatrix := Matrix (Fin 4) (Fin 4) Letter

/-- The operation of incrementing a letter (with wrapping) -/
def nextLetter (l : Letter) : Letter :=
  ⟨(l.val + 1) % 26, by sorry⟩

/-- The invariant property for a 2x2 submatrix -/
def invariant (a b c d : Letter) : ℤ :=
  (a.val + d.val : ℤ) - (b.val + c.val : ℤ)

/-- Theorem: The invariant is preserved under row and column operations -/
theorem invariant_preserved (a b c d : Letter) :
  (invariant a b c d = invariant (nextLetter a) (nextLetter b) c d) ∧
  (invariant a b c d = invariant (nextLetter a) b (nextLetter c) d) :=
sorry

/-- The initial matrix (a) -/
def matrix_a : LetterMatrix := sorry

/-- The target matrix (b) -/
def matrix_b : LetterMatrix := sorry

/-- Theorem: Matrix (a) cannot be transformed into matrix (b) -/
theorem cannot_transform (ops : ℕ) :
  ∀ (m : LetterMatrix), 
    (∃ (i : Fin 4), ∀ (j : Fin 4), m i j = nextLetter (matrix_a i j)) ∨
    (∃ (j : Fin 4), ∀ (i : Fin 4), m i j = nextLetter (matrix_a i j)) →
    m ≠ matrix_b :=
sorry

end NUMINAMATH_CALUDE_invariant_preserved_cannot_transform_l703_70321


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l703_70328

/-- Given a bag with 3 red balls and 2 white balls, the probability of drawing
    at least one white ball when 3 balls are randomly drawn is 9/10. -/
theorem probability_at_least_one_white_ball
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (drawn_balls : ℕ)
  (h1 : total_balls = red_balls + white_balls)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : drawn_balls = 3) :
  (1 - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l703_70328


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l703_70391

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l703_70391


namespace NUMINAMATH_CALUDE_four_number_theorem_l703_70388

theorem four_number_theorem (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (p q r s : ℕ+), a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r := by
  sorry

end NUMINAMATH_CALUDE_four_number_theorem_l703_70388


namespace NUMINAMATH_CALUDE_profit_threshold_l703_70301

-- Define the variables
def cost_per_unit : ℝ := 2
def fixed_cost : ℝ := 500
def selling_price : ℝ := 2.5

-- Define the profit function
def profit (x : ℝ) : ℝ := selling_price * x - (fixed_cost + cost_per_unit * x)

-- Theorem statement
theorem profit_threshold :
  ∀ x : ℝ, profit x > 0 ↔ x > 1000 := by sorry

end NUMINAMATH_CALUDE_profit_threshold_l703_70301


namespace NUMINAMATH_CALUDE_wire_length_ratio_l703_70354

theorem wire_length_ratio (edge_length : ℕ) (wire_pieces : ℕ) (wire_length : ℕ) : 
  edge_length = wire_length ∧ wire_pieces = 12 →
  (wire_pieces * wire_length) / (edge_length^3 * 12) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l703_70354


namespace NUMINAMATH_CALUDE_xiaoSiScore_l703_70395

/-- Represents the correctness of an answer -/
inductive Correctness
| Correct
| Incorrect

/-- Represents a single question in the test -/
structure Question where
  number : Nat
  points : Nat
  correctness : Correctness

/-- Calculates the score for a single question -/
def scoreQuestion (q : Question) : Nat :=
  match q.correctness with
  | Correctness.Correct => q.points
  | Correctness.Incorrect => 0

/-- Xiao Si's test answers -/
def xiaoSiAnswers : List Question :=
  [
    { number := 1, points := 20, correctness := Correctness.Correct },
    { number := 2, points := 20, correctness := Correctness.Incorrect },
    { number := 3, points := 20, correctness := Correctness.Incorrect },
    { number := 4, points := 20, correctness := Correctness.Incorrect },
    { number := 5, points := 20, correctness := Correctness.Incorrect }
  ]

/-- Calculates the total score for the test -/
def calculateTotalScore (answers : List Question) : Nat :=
  answers.foldl (fun acc q => acc + scoreQuestion q) 0

/-- Theorem stating that Xiao Si's total score is 20 points -/
theorem xiaoSiScore : calculateTotalScore xiaoSiAnswers = 20 := by
  sorry


end NUMINAMATH_CALUDE_xiaoSiScore_l703_70395


namespace NUMINAMATH_CALUDE_angle_terminal_side_l703_70323

/-- Given that the terminal side of angle α passes through point (a, 1) and tan α = -1/2, prove that a = -2 -/
theorem angle_terminal_side (α : Real) (a : Real) : 
  (∃ (x y : Real), x = a ∧ y = 1 ∧ Real.tan α = y / x) → 
  Real.tan α = -1/2 → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l703_70323


namespace NUMINAMATH_CALUDE_distance_on_line_l703_70375

/-- The distance between two points on a line --/
theorem distance_on_line (n m p q r s : ℝ) :
  q = n * p + m →
  s = n * r + m →
  Real.sqrt ((r - p)^2 + (s - q)^2) = |r - p| * Real.sqrt (1 + n^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l703_70375


namespace NUMINAMATH_CALUDE_factorial_ratio_l703_70379

theorem factorial_ratio : Nat.factorial 15 / Nat.factorial 14 = 15 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l703_70379


namespace NUMINAMATH_CALUDE_cloth_cost_price_l703_70337

/-- Proves that the cost price of one metre of cloth is 66.25,
    given the selling conditions of a cloth trader. -/
theorem cloth_cost_price
  (meters_sold : ℕ)
  (selling_price : ℚ)
  (profit_per_meter : ℚ)
  (h_meters : meters_sold = 80)
  (h_price : selling_price = 6900)
  (h_profit : profit_per_meter = 20) :
  (selling_price - meters_sold * profit_per_meter) / meters_sold = 66.25 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l703_70337


namespace NUMINAMATH_CALUDE_system_solution_l703_70317

theorem system_solution : ∃ (x y : ℝ), 
  x = 2 ∧ y = -1 ∧ 2*x + y = 3 ∧ -x - y = -1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l703_70317


namespace NUMINAMATH_CALUDE_word_problem_points_word_problem_points_correct_l703_70344

theorem word_problem_points (total_problems : ℕ) (computation_problems : ℕ) 
  (computation_points : ℕ) (total_points : ℕ) : ℕ :=
  let word_problems := total_problems - computation_problems
  let computation_total := computation_problems * computation_points
  let word_total := total_points - computation_total
  word_total / word_problems

#check word_problem_points 30 20 3 110 = 5

theorem word_problem_points_correct : 
  word_problem_points 30 20 3 110 = 5 := by sorry

end NUMINAMATH_CALUDE_word_problem_points_word_problem_points_correct_l703_70344


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l703_70363

theorem right_triangle_base_length 
  (area : ℝ) 
  (hypotenuse : ℝ) 
  (side : ℝ) 
  (h_area : area = 24) 
  (h_hypotenuse : hypotenuse = 10) 
  (h_side : side = 8) : 
  ∃ (base height : ℝ), 
    area = (1/2) * base * height ∧ 
    hypotenuse^2 = base^2 + height^2 ∧ 
    (base = side ∨ height = side) ∧ 
    base = 8 := by
  sorry

#check right_triangle_base_length

end NUMINAMATH_CALUDE_right_triangle_base_length_l703_70363


namespace NUMINAMATH_CALUDE_prob_A_union_B_l703_70303

-- Define the sample space for a fair six-sided die
def Ω : Finset Nat := Finset.range 6

-- Define the probability measure
def P (S : Finset Nat) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Define event A: getting a 3
def A : Finset Nat := {2}

-- Define event B: getting an even number
def B : Finset Nat := {1, 3, 5}

-- Theorem statement
theorem prob_A_union_B : P (A ∪ B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_union_B_l703_70303


namespace NUMINAMATH_CALUDE_solutions_of_x_fourth_minus_16_l703_70307

theorem solutions_of_x_fourth_minus_16 :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_of_x_fourth_minus_16_l703_70307


namespace NUMINAMATH_CALUDE_value_of_expression_l703_70334

theorem value_of_expression (h k : ℤ) : 
  (∃ a : ℤ, 3 * X^3 - h * X - k = a * (X - 3) * (X + 1)) →
  |3 * h - 2 * k| = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l703_70334


namespace NUMINAMATH_CALUDE_remainder_equality_l703_70386

theorem remainder_equality (x : ℕ+) (y : ℤ) 
  (h1 : ∃ k : ℤ, 200 = k * x.val + 5) 
  (h2 : ∃ m : ℤ, y = m * x.val + 5) : 
  y % x.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l703_70386


namespace NUMINAMATH_CALUDE_rectangle_length_equality_l703_70304

/-- Given two rectangles with equal area, where one rectangle measures 15 inches by 24 inches
    and the other is 45 inches wide, the length of the second rectangle is 8 inches. -/
theorem rectangle_length_equality (carol_length carol_width jordan_width : ℕ) 
    (jordan_length : ℚ) : 
  carol_length = 15 ∧ 
  carol_width = 24 ∧ 
  jordan_width = 45 ∧ 
  carol_length * carol_width = jordan_length * jordan_width →
  jordan_length = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_equality_l703_70304


namespace NUMINAMATH_CALUDE_sine_value_from_tangent_cosine_relation_l703_70314

theorem sine_value_from_tangent_cosine_relation (θ : Real) 
  (h1 : 8 * Real.tan θ = 3 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sine_value_from_tangent_cosine_relation_l703_70314


namespace NUMINAMATH_CALUDE_simplify_expression_l703_70346

theorem simplify_expression (x : ℝ) : (2*x)^5 - (3*x^2)*(x^3) = 29*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l703_70346


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l703_70329

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (1 / a + 4 / b) ≥ 9 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l703_70329


namespace NUMINAMATH_CALUDE_divides_trans_divides_mul_l703_70342

/-- Divisibility relation for positive integers -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Transitivity of divisibility -/
theorem divides_trans {a b c : ℕ+} (h1 : divides a b) (h2 : divides b c) : 
  divides a c := by sorry

/-- Product of divisibilities -/
theorem divides_mul {a b c d : ℕ+} (h1 : divides a b) (h2 : divides c d) :
  divides (a * c) (b * d) := by sorry

end NUMINAMATH_CALUDE_divides_trans_divides_mul_l703_70342


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l703_70356

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) 
  (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l703_70356


namespace NUMINAMATH_CALUDE_roberta_garage_sale_records_l703_70364

/-- The number of records Roberta bought at the garage sale -/
def records_bought_at_garage_sale (initial_records : ℕ) (gifted_records : ℕ) (days_per_record : ℕ) (total_listening_days : ℕ) : ℕ :=
  (total_listening_days / days_per_record) - (initial_records + gifted_records)

/-- Theorem stating that Roberta bought 30 records at the garage sale -/
theorem roberta_garage_sale_records : 
  records_bought_at_garage_sale 8 12 2 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_roberta_garage_sale_records_l703_70364


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l703_70331

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The right focus of the hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola) : ℝ × ℝ := sorry

/-- Predicate to check if a point lies on the circle with diameter between two other points -/
def lies_on_circle_diameter (p q r : ℝ × ℝ) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) : 
  lies_on_circle_diameter (0, h.b) (left_vertex h) (right_focus h) →
  eccentricity h = (Real.sqrt 5 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l703_70331


namespace NUMINAMATH_CALUDE_shorter_book_pages_l703_70347

theorem shorter_book_pages (x y : ℕ) (h1 : y = x + 10) (h2 : y / 2 = x) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_shorter_book_pages_l703_70347


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l703_70368

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2 - 1
  ∃ (x y : ℝ), f x y = 0 ∧ x > 0 ∧ x * c = 0 ∧ 2 * c = y * a / b ∧ c^2 = a^2 + b^2
  → c / a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l703_70368


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l703_70333

theorem geometric_sequence_sum (n : ℕ) (a r : ℚ) (h1 : a = 1/3) (h2 : r = 1/3) :
  (a * (1 - r^n) / (1 - r) = 80/243) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l703_70333


namespace NUMINAMATH_CALUDE_horse_cloth_problem_l703_70309

/-- Represents the system of equations for the horse and cloth problem -/
def horse_cloth_system (m n : ℚ) : Prop :=
  m + n = 100 ∧ 3 * m + n / 3 = 100

/-- The horse and cloth problem statement -/
theorem horse_cloth_problem :
  ∃ m n : ℚ, 
    m ≥ 0 ∧ n ≥ 0 ∧  -- Ensuring non-negative numbers of horses
    horse_cloth_system m n :=
by sorry

end NUMINAMATH_CALUDE_horse_cloth_problem_l703_70309


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l703_70319

-- (1) Prove that x = 3 or x = -1 is a solution to 4(x-1)^2 - 16 = 0
theorem problem_1 : ∃ x : ℝ, (x = 3 ∨ x = -1) ∧ 4 * (x - 1)^2 - 16 = 0 := by sorry

-- (2) Prove that ∛(-64) + √16 * √(9/4) + (-√2)^2 = 4
theorem problem_2 : ((-64 : ℝ)^(1/3)) + Real.sqrt 16 * Real.sqrt (9/4) + (-Real.sqrt 2)^2 = 4 := by sorry

-- (3) Prove that if a is the integer part and b is the decimal part of 9 - √13, then 2a + b = 14 - √13
theorem problem_3 (a b : ℝ) (h : a = ⌊9 - Real.sqrt 13⌋ ∧ b = 9 - Real.sqrt 13 - a) :
  2 * a + b = 14 - Real.sqrt 13 := by sorry

-- (4) Define an operation ⊕ and prove that x = 5 or x = -5 is a solution to (4 ⊕ 3) ⊕ x = 24
def circle_plus (a b : ℝ) : ℝ := a^2 - b^2

theorem problem_4 : ∃ x : ℝ, (x = 5 ∨ x = -5) ∧ circle_plus (circle_plus 4 3) x = 24 := by sorry

-- (5) Prove that if ∠1 and ∠2 are parallel, and ∠1 is 36° less than three times ∠2, then ∠1 = 18° or ∠1 = 126°
theorem problem_5 (angle1 angle2 : ℝ) 
  (h1 : angle1 = 3 * angle2 - 36)
  (h2 : angle1 = angle2 ∨ angle1 + angle2 = 180) :
  angle1 = 18 ∨ angle1 = 126 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_l703_70319


namespace NUMINAMATH_CALUDE_unique_k_for_pythagorean_like_equation_l703_70326

theorem unique_k_for_pythagorean_like_equation :
  ∃! k : ℕ+, ∃ a b : ℕ+, a^2 + b^2 = k * a * b := by sorry

end NUMINAMATH_CALUDE_unique_k_for_pythagorean_like_equation_l703_70326


namespace NUMINAMATH_CALUDE_cylinder_diameter_l703_70349

/-- The diameter of a cylinder given its height and volume -/
theorem cylinder_diameter (h : ℝ) (v : ℝ) (h_pos : h > 0) (v_pos : v > 0) :
  let d := 2 * Real.sqrt (9 / Real.pi)
  h = 5 ∧ v = 45 → d * d * Real.pi * h / 4 = v := by
  sorry

end NUMINAMATH_CALUDE_cylinder_diameter_l703_70349


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l703_70341

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 6 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -32/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l703_70341


namespace NUMINAMATH_CALUDE_queen_jack_hands_count_l703_70366

/-- The number of queens in a standard deck --/
def num_queens : ℕ := 4

/-- The number of jacks in a standard deck --/
def num_jacks : ℕ := 4

/-- The total number of queens and jacks --/
def total_queens_jacks : ℕ := num_queens + num_jacks

/-- The number of cards in a hand --/
def hand_size : ℕ := 5

/-- The number of 5-card hands containing only queens and jacks --/
def num_queen_jack_hands : ℕ := Nat.choose total_queens_jacks hand_size

theorem queen_jack_hands_count : num_queen_jack_hands = 56 := by
  sorry

end NUMINAMATH_CALUDE_queen_jack_hands_count_l703_70366


namespace NUMINAMATH_CALUDE_veena_bill_fraction_l703_70320

theorem veena_bill_fraction :
  ∀ (L V A : ℚ),
  V = (1/2) * L →
  A = (3/4) * V →
  V / (L + V + A) = 4/15 := by
sorry

end NUMINAMATH_CALUDE_veena_bill_fraction_l703_70320
