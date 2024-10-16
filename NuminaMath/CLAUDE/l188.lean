import Mathlib

namespace NUMINAMATH_CALUDE_swimmer_distance_proof_l188_18859

/-- Calculates the distance swam against a current given the swimmer's speed in still water,
    the current's speed, and the time spent swimming. -/
def distance_swam_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimmer_speed - current_speed) * time

/-- Proves that a swimmer with a given speed in still water, swimming against a specific current
    for a certain amount of time, covers the expected distance. -/
theorem swimmer_distance_proof (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ)
    (h1 : swimmer_speed = 3)
    (h2 : current_speed = 1.7)
    (h3 : time = 2.3076923076923075)
    : distance_swam_against_current swimmer_speed current_speed time = 3 := by
  sorry

#eval distance_swam_against_current 3 1.7 2.3076923076923075

end NUMINAMATH_CALUDE_swimmer_distance_proof_l188_18859


namespace NUMINAMATH_CALUDE_self_common_tangents_l188_18898

-- Define the concept of a self-common tangent
def has_self_common_tangent (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ x₂ y m b : ℝ), x₁ ≠ x₂ ∧ 
    f x₁ y ∧ f x₂ y ∧
    (∀ x y, f x y → y = m * x + b)

-- Define the four curves
def curve1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def curve2 (x y : ℝ) : Prop := y = x^2 - abs x
def curve3 (x y : ℝ) : Prop := y = 3 * Real.sin x + 4 * Real.cos x
def curve4 (x y : ℝ) : Prop := abs x + 1 = Real.sqrt (4 - y^2)

-- Theorem statement
theorem self_common_tangents :
  has_self_common_tangent curve2 ∧ 
  has_self_common_tangent curve3 ∧
  ¬has_self_common_tangent curve1 ∧
  ¬has_self_common_tangent curve4 :=
sorry

end NUMINAMATH_CALUDE_self_common_tangents_l188_18898


namespace NUMINAMATH_CALUDE_school_ratio_problem_l188_18828

theorem school_ratio_problem (total_students : ℕ) (boys_percentage : ℚ) 
  (represented_students : ℕ) (h1 : total_students = 140) 
  (h2 : boys_percentage = 1/2) (h3 : represented_students = 98) : 
  (represented_students : ℚ) / (boys_percentage * total_students) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_school_ratio_problem_l188_18828


namespace NUMINAMATH_CALUDE_incorrect_statement_C_l188_18886

theorem incorrect_statement_C :
  (∀ a b c : ℚ, a / 4 = c / 5 → (a - 4) / 4 = (c - 5) / 5) ∧
  (∀ a b : ℚ, (a - b) / b = 1 / 7 → a / b = 8 / 7) ∧
  (∃ a b : ℚ, a / b = 2 / 5 ∧ (a ≠ 2 ∨ b ≠ 5)) ∧
  (∀ a b c d : ℚ, a / b = c / d ∧ a / b = 2 / 3 ∧ b - d ≠ 0 → (a - c) / (b - d) = 2 / 3) :=
by sorry


end NUMINAMATH_CALUDE_incorrect_statement_C_l188_18886


namespace NUMINAMATH_CALUDE_transformation_maps_correctly_l188_18850

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Scales a point by a given factor -/
def scale (p : Point) (factor : ℝ) : Point :=
  ⟨p.x * factor, p.y * factor⟩

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Applies scaling followed by reflection across x-axis -/
def scaleAndReflect (p : Point) (factor : ℝ) : Point :=
  reflectX (scale p factor)

theorem transformation_maps_correctly :
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨2, 3⟩
  let A' : Point := ⟨3, -6⟩
  let B' : Point := ⟨6, -9⟩
  (scaleAndReflect A 3 = A') ∧ (scaleAndReflect B 3 = B') := by
  sorry

end NUMINAMATH_CALUDE_transformation_maps_correctly_l188_18850


namespace NUMINAMATH_CALUDE_doubled_volume_cube_edge_l188_18847

/-- The edge length of a cube with double the volume of a 5 cm cube is approximately 6.4 cm. -/
theorem doubled_volume_cube_edge (ε : ℝ) (h : ε > 0) : ∃ x : ℝ, 
  x^3 = 2 * 5^3 ∧ |x - 6.4| < ε :=
sorry

end NUMINAMATH_CALUDE_doubled_volume_cube_edge_l188_18847


namespace NUMINAMATH_CALUDE_divisors_of_572_divisors_of_572a3bc_case1_divisors_of_572a3bc_case2_l188_18820

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_572 :
  count_divisors 572 = 12 :=
sorry

theorem divisors_of_572a3bc_case1 (a b c : ℕ) 
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c)
  (ha_gt : a > 20) (hb_gt : b > 20) (hc_gt : c > 20)
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  count_divisors (572 * a^3 * b * c) = 192 :=
sorry

theorem divisors_of_572a3bc_case2 :
  count_divisors (572 * 31^3 * 32 * 33) = 384 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_572_divisors_of_572a3bc_case1_divisors_of_572a3bc_case2_l188_18820


namespace NUMINAMATH_CALUDE_units_digit_sum_product_l188_18818

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_product : units_digit ((13 * 41) + (27 * 34)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_product_l188_18818


namespace NUMINAMATH_CALUDE_largest_side_of_special_triangle_l188_18899

/-- Given a scalene triangle with sides x and y, and area Δ, satisfying the equation
    x + 2Δ/x = y + 2Δ/y, prove that when x = 60 and y = 63, the largest side is 87. -/
theorem largest_side_of_special_triangle (x y Δ : ℝ) 
  (hx : x = 60)
  (hy : y = 63)
  (h_eq : x + 2 * Δ / x = y + 2 * Δ / y)
  (h_scalene : x ≠ y)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_Δ : Δ > 0) :
  max x (max y (Real.sqrt (x^2 + y^2))) = 87 :=
sorry

end NUMINAMATH_CALUDE_largest_side_of_special_triangle_l188_18899


namespace NUMINAMATH_CALUDE_product_loop_result_l188_18888

def product_loop (i : ℕ) : ℕ :=
  if i < 11 then 1 else i * product_loop (i - 1)

theorem product_loop_result :
  product_loop 12 = 132 :=
by sorry

end NUMINAMATH_CALUDE_product_loop_result_l188_18888


namespace NUMINAMATH_CALUDE_sum_of_angles_in_three_triangles_l188_18877

theorem sum_of_angles_in_three_triangles :
  ∀ (angle1 angle2 angle3 angle4 angle5 angle6 angle7 angle8 angle9 : ℝ),
    angle1 > 0 → angle2 > 0 → angle3 > 0 → angle4 > 0 → angle5 > 0 →
    angle6 > 0 → angle7 > 0 → angle8 > 0 → angle9 > 0 →
    angle1 + angle2 + angle3 = 180 →
    angle4 + angle5 + angle6 = 180 →
    angle7 + angle8 + angle9 = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 + angle8 + angle9 = 540 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_three_triangles_l188_18877


namespace NUMINAMATH_CALUDE_trajectory_constant_sum_distances_l188_18869

/-- The trajectory of points with constant sum of distances to two fixed points -/
theorem trajectory_constant_sum_distances (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-2, 0)
  let F₂ : ℝ × ℝ := (2, 0)
  (‖P - F₁‖ + ‖P - F₂‖ = 4) →
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
by sorry

end NUMINAMATH_CALUDE_trajectory_constant_sum_distances_l188_18869


namespace NUMINAMATH_CALUDE_speed_difference_calculation_l188_18885

/-- Calculates the difference in average speed between no traffic and heavy traffic conditions --/
theorem speed_difference_calculation (distance : ℝ) (heavy_traffic_time : ℝ) (no_traffic_time : ℝ)
  (construction_zones : ℕ) (construction_delay : ℝ) (heavy_traffic_rest_stops : ℕ)
  (no_traffic_rest_stops : ℕ) (rest_stop_duration : ℝ) :
  distance = 200 →
  heavy_traffic_time = 5 →
  no_traffic_time = 4 →
  construction_zones = 2 →
  construction_delay = 0.25 →
  heavy_traffic_rest_stops = 3 →
  no_traffic_rest_stops = 2 →
  rest_stop_duration = 1/6 →
  let heavy_traffic_driving_time := heavy_traffic_time - (construction_zones * construction_delay + heavy_traffic_rest_stops * rest_stop_duration)
  let no_traffic_driving_time := no_traffic_time - (no_traffic_rest_stops * rest_stop_duration)
  let heavy_traffic_speed := distance / heavy_traffic_driving_time
  let no_traffic_speed := distance / no_traffic_driving_time
  ∃ ε > 0, |no_traffic_speed - heavy_traffic_speed - 4.5| < ε :=
by sorry

end NUMINAMATH_CALUDE_speed_difference_calculation_l188_18885


namespace NUMINAMATH_CALUDE_unique_recurrence_sequence_l188_18892

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 > 1 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

/-- The theorem stating the existence and uniqueness of the sequence -/
theorem unique_recurrence_sequence :
  ∃! a : ℕ → ℤ, RecurrenceSequence a :=
sorry

end NUMINAMATH_CALUDE_unique_recurrence_sequence_l188_18892


namespace NUMINAMATH_CALUDE_namjoon_lowest_l188_18826

def board_A : ℝ := 2.4
def board_B : ℝ := 3.2
def board_C : ℝ := 2.8

def eunji_height : ℝ := 8 * board_A
def namjoon_height : ℝ := 4 * board_B
def hoseok_height : ℝ := 5 * board_C

theorem namjoon_lowest : 
  namjoon_height < eunji_height ∧ namjoon_height < hoseok_height :=
by sorry

end NUMINAMATH_CALUDE_namjoon_lowest_l188_18826


namespace NUMINAMATH_CALUDE_toby_friends_percentage_l188_18863

def toby_boy_friends : ℕ := 33
def toby_girl_friends : ℕ := 27

theorem toby_friends_percentage :
  (toby_boy_friends : ℚ) / (toby_boy_friends + toby_girl_friends : ℚ) * 100 = 55 := by
  sorry

end NUMINAMATH_CALUDE_toby_friends_percentage_l188_18863


namespace NUMINAMATH_CALUDE_largest_s_value_l188_18854

theorem largest_s_value (r s : ℕ) : 
  r ≥ s → 
  s ≥ 5 → 
  (r - 2) * s * 61 = (s - 2) * r * 60 → 
  s ≤ 121 ∧ ∃ r' : ℕ, r' ≥ 121 ∧ (r' - 2) * 121 * 61 = (121 - 2) * r' * 60 :=
by sorry

end NUMINAMATH_CALUDE_largest_s_value_l188_18854


namespace NUMINAMATH_CALUDE_multiply_by_nine_divide_by_four_l188_18895

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1/3

-- State the theorem
theorem multiply_by_nine_divide_by_four :
  (repeating_third * 9) / 4 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_nine_divide_by_four_l188_18895


namespace NUMINAMATH_CALUDE_rolling_circle_trajectory_is_line_segment_l188_18889

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point on a circle -/
structure PointOnCircle where
  circle : Circle
  angle : ℝ  -- Angle from the positive x-axis

/-- Represents the trajectory of a point -/
inductive Trajectory
  | LineSegment : (ℝ × ℝ) → (ℝ × ℝ) → Trajectory

/-- The trajectory of a fixed point on a circle rolling inside another circle -/
def rollingCircleTrajectory (stationaryCircle : Circle) (movingCircle : Circle) (fixedPoint : PointOnCircle) : Trajectory :=
  sorry

/-- Main theorem: The trajectory of a fixed point on a circle rolling inside another circle with twice its radius is a line segment -/
theorem rolling_circle_trajectory_is_line_segment
  (stationaryCircle : Circle)
  (movingCircle : Circle)
  (fixedPoint : PointOnCircle)
  (h1 : movingCircle.radius = stationaryCircle.radius / 2)
  (h2 : fixedPoint.circle = movingCircle) :
  ∃ (p q : ℝ × ℝ), rollingCircleTrajectory stationaryCircle movingCircle fixedPoint = Trajectory.LineSegment p q ∧
                    p = stationaryCircle.center :=
  sorry

end NUMINAMATH_CALUDE_rolling_circle_trajectory_is_line_segment_l188_18889


namespace NUMINAMATH_CALUDE_no_good_tetrahedron_inside_good_parallelepiped_l188_18856

-- Define a good polyhedron
def is_good_polyhedron (volume : ℝ) (surface_area : ℝ) : Prop :=
  volume = surface_area

-- Define a tetrahedron
structure Tetrahedron where
  volume : ℝ
  surface_area : ℝ

-- Define a parallelepiped
structure Parallelepiped where
  volume : ℝ
  surface_area : ℝ
  face_areas : Fin 3 → ℝ
  heights : Fin 3 → ℝ

-- Define the property of a tetrahedron being inside a parallelepiped
def tetrahedron_inside_parallelepiped (t : Tetrahedron) (p : Parallelepiped) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  t.volume = (1/3) * t.surface_area * r ∧
  p.heights 0 > 2 * r

-- Theorem statement
theorem no_good_tetrahedron_inside_good_parallelepiped :
  ¬ ∃ (t : Tetrahedron) (p : Parallelepiped),
    is_good_polyhedron t.volume t.surface_area ∧
    is_good_polyhedron p.volume p.surface_area ∧
    tetrahedron_inside_parallelepiped t p :=
sorry

end NUMINAMATH_CALUDE_no_good_tetrahedron_inside_good_parallelepiped_l188_18856


namespace NUMINAMATH_CALUDE_trigonometric_identity_l188_18803

theorem trigonometric_identity (x : ℝ) :
  (1 / Real.cos (2022 * x) + Real.tan (2022 * x) = 1 / 2022) →
  (1 / Real.cos (2022 * x) - Real.tan (2022 * x) = 2022) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l188_18803


namespace NUMINAMATH_CALUDE_sector_angle_l188_18832

/-- Given an arc length of 4 cm and a radius of 2 cm, the central angle of the sector in radians is 2. -/
theorem sector_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 4) (h2 : radius = 2) :
  arc_length / radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l188_18832


namespace NUMINAMATH_CALUDE_inequality_solutions_l188_18868

theorem inequality_solutions :
  (∀ x, x * (x + 2) > x * (3 - x) + 1 ↔ x < -1/2 ∨ x > 1) ∧
  (∀ a x, x^2 - 2*a*x - 8*a^2 ≤ 0 ↔
    (a > 0 ∧ -2*a ≤ x ∧ x ≤ 4*a) ∨
    (a = 0 ∧ x = 0) ∨
    (a < 0 ∧ 4*a ≤ x ∧ x ≤ -2*a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l188_18868


namespace NUMINAMATH_CALUDE_correct_expression_l188_18879

theorem correct_expression : 
  (-((-8 : ℝ) ^ (1/3 : ℝ)) = 2) ∧ 
  (Real.sqrt 9 ≠ 3 ∧ Real.sqrt 9 ≠ -3) ∧
  (-Real.sqrt 16 ≠ 4) ∧
  (Real.sqrt ((-2)^2) ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_correct_expression_l188_18879


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l188_18880

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 3 4 = {x | x^2 + a*x + b < 0}) : 
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/4) ∪ Set.Ici (1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l188_18880


namespace NUMINAMATH_CALUDE_manoj_lending_amount_l188_18872

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℝ := 3900

/-- The interest rate for borrowing (as a decimal) -/
def borrowing_rate : ℝ := 0.06

/-- The interest rate for lending (as a decimal) -/
def lending_rate : ℝ := 0.09

/-- The duration of both borrowing and lending in years -/
def duration : ℝ := 3

/-- Manoj's total gain from the transaction -/
def total_gain : ℝ := 824.85

/-- Calculate simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The sum lent by Manoj to Ramu -/
def lent_amount : ℝ := 4355

theorem manoj_lending_amount :
  simple_interest lent_amount lending_rate duration -
  simple_interest borrowed_amount borrowing_rate duration =
  total_gain := by sorry

end NUMINAMATH_CALUDE_manoj_lending_amount_l188_18872


namespace NUMINAMATH_CALUDE_juanita_daily_cost_l188_18811

/-- The amount Juanita spends on a newspaper from Monday through Saturday -/
def daily_cost : ℝ := sorry

/-- Grant's yearly newspaper cost -/
def grant_yearly_cost : ℝ := 200

/-- Juanita's Sunday newspaper cost -/
def sunday_cost : ℝ := 2

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Difference between Juanita's and Grant's yearly newspaper costs -/
def cost_difference : ℝ := 60

theorem juanita_daily_cost :
  daily_cost * 6 * weeks_per_year + sunday_cost * weeks_per_year = 
  grant_yearly_cost + cost_difference :=
by sorry

end NUMINAMATH_CALUDE_juanita_daily_cost_l188_18811


namespace NUMINAMATH_CALUDE_max_grain_mass_on_platform_l188_18896

/-- Represents a rectangular platform --/
structure Platform where
  length : ℝ
  width : ℝ

/-- Represents the properties of grain --/
structure Grain where
  density : ℝ
  max_angle : ℝ

/-- Calculates the maximum mass of grain that can be loaded onto a platform --/
def max_grain_mass (p : Platform) (g : Grain) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform --/
theorem max_grain_mass_on_platform :
  let p : Platform := { length := 10, width := 5 }
  let g : Grain := { density := 1200, max_angle := 45 }
  max_grain_mass p g = 175000 := by
  sorry

end NUMINAMATH_CALUDE_max_grain_mass_on_platform_l188_18896


namespace NUMINAMATH_CALUDE_incorrect_product_calculation_l188_18817

theorem incorrect_product_calculation (x : ℕ) : 
  (53 * x - 35 * x = 540) → (53 * x = 1590) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_product_calculation_l188_18817


namespace NUMINAMATH_CALUDE_unique_solution_for_star_equation_l188_18827

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

-- State the theorem
theorem unique_solution_for_star_equation :
  ∃! y : ℝ, star 4 y = 16 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_star_equation_l188_18827


namespace NUMINAMATH_CALUDE_second_grade_sample_l188_18835

/-- Given a total sample size and ratios for three grades, 
    calculate the number of students to be drawn from a specific grade. -/
def stratified_sample (total_sample : ℕ) (ratio1 ratio2 ratio3 : ℕ) (target_ratio : ℕ) : ℕ :=
  (target_ratio * total_sample) / (ratio1 + ratio2 + ratio3)

/-- Theorem: Given a total sample of 50 and ratios 3:3:4, 
    the number of students from the grade with ratio 3 is 15. -/
theorem second_grade_sample :
  stratified_sample 50 3 3 4 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_sample_l188_18835


namespace NUMINAMATH_CALUDE_apples_used_for_lunch_l188_18801

theorem apples_used_for_lunch (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 38 → bought = 28 → final = 46 → initial - (final - bought) = 20 := by
sorry

end NUMINAMATH_CALUDE_apples_used_for_lunch_l188_18801


namespace NUMINAMATH_CALUDE_intersection_condition_l188_18833

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}

/-- The set N in ℝ² parameterized by a -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- The theorem stating the necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l188_18833


namespace NUMINAMATH_CALUDE_opposite_points_probability_l188_18829

theorem opposite_points_probability (n : ℕ) (h : n = 12) : 
  (n / 2) / (n.choose 2) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_opposite_points_probability_l188_18829


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_8_l188_18897

/-- A function that returns the product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_with_digit_product_8 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 8 → n ≤ 81 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_8_l188_18897


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l188_18861

theorem computer_literate_female_employees 
  (total_employees : ℕ)
  (female_percentage : ℝ)
  (male_computer_literate_percentage : ℝ)
  (total_computer_literate_percentage : ℝ)
  (h_total : total_employees = 1200)
  (h_female : female_percentage = 0.6)
  (h_male_cl : male_computer_literate_percentage = 0.5)
  (h_total_cl : total_computer_literate_percentage = 0.62) :
  ⌊female_percentage * total_employees - 
   (1 - female_percentage) * male_computer_literate_percentage * total_employees⌋ = 504 :=
by sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l188_18861


namespace NUMINAMATH_CALUDE_common_chord_length_l188_18870

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem common_chord_length :
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l188_18870


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l188_18813

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) → 
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) ∧
  (Real.sqrt (3 * (3/25)) = 5 * (3/25)) := by
sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l188_18813


namespace NUMINAMATH_CALUDE_orange_distribution_l188_18831

-- Define the total number of oranges
def total_oranges : ℕ := 30

-- Define the number of people
def num_people : ℕ := 3

-- Define the minimum number of oranges each person must receive
def min_oranges : ℕ := 3

-- Define the function to calculate the number of ways to distribute oranges
def ways_to_distribute (total : ℕ) (people : ℕ) (min : ℕ) : ℕ :=
  Nat.choose (total - people * min + people - 1) (people - 1)

-- Theorem statement
theorem orange_distribution :
  ways_to_distribute total_oranges num_people min_oranges = 253 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l188_18831


namespace NUMINAMATH_CALUDE_casper_candy_problem_l188_18883

def candy_distribution (initial : ℕ) : ℕ :=
  let day1 := initial * 3 / 4 - 3
  let day2 := day1 * 4 / 5 - 5
  let day3 := day2 * 5 / 6 - 6
  day3

theorem casper_candy_problem :
  ∃ (initial : ℕ), candy_distribution initial = 10 ∧ initial = 678 :=
sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l188_18883


namespace NUMINAMATH_CALUDE_max_students_distribution_l188_18822

theorem max_students_distribution (pens toys : ℕ) (h1 : pens = 451) (h2 : toys = 410) :
  Nat.gcd pens toys = 41 :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l188_18822


namespace NUMINAMATH_CALUDE_item_value_proof_l188_18812

def import_tax_rate : ℝ := 0.07
def tax_threshold : ℝ := 1000
def tax_paid : ℝ := 87.50

theorem item_value_proof (total_value : ℝ) : 
  total_value = 2250 := by
  sorry

end NUMINAMATH_CALUDE_item_value_proof_l188_18812


namespace NUMINAMATH_CALUDE_system_of_inequalities_l188_18843

theorem system_of_inequalities (x : ℝ) : 
  (x - 1 < 3 ∧ x + 1 ≥ (1 + 2*x) / 3) ↔ -2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l188_18843


namespace NUMINAMATH_CALUDE_evaluate_expression_l188_18804

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 3/4) 
  (hz : z = -2) : 
  x^3 * y^2 * z^2 = 9/16 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l188_18804


namespace NUMINAMATH_CALUDE_juice_price_ratio_l188_18836

theorem juice_price_ratio :
  let volume_A : ℝ := 1.25  -- Brand A's volume relative to Brand B
  let price_A : ℝ := 0.85   -- Brand A's price relative to Brand B
  let unit_price_ratio := (price_A / volume_A) / 1  -- Ratio of unit prices (A / B)
  unit_price_ratio = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_juice_price_ratio_l188_18836


namespace NUMINAMATH_CALUDE_mutual_win_exists_l188_18867

/-- Represents the result of a match between two teams -/
inductive MatchResult
| Win
| Draw
| Loss

/-- Calculates points for a given match result -/
def points (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 2
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents a tournament with given number of teams -/
structure Tournament (n : Nat) where
  firstRound : Fin n → Fin n → MatchResult
  secondRound : Fin n → Fin n → MatchResult

/-- Calculates total points for a team after both rounds -/
def totalPoints (t : Tournament n) (team : Fin n) : Nat :=
  sorry

/-- Checks if all teams have different points after the first round -/
def allDifferentFirstRound (t : Tournament n) : Prop :=
  sorry

/-- Checks if all teams have the same points after both rounds -/
def allSameTotal (t : Tournament n) : Prop :=
  sorry

/-- Checks if there exists a pair of teams that have each won once against each other -/
def existsMutualWin (t : Tournament n) : Prop :=
  sorry

/-- Main theorem: If all teams have different points after the first round
    and the same total points after both rounds, then there exists a pair
    of teams that have each won once against each other -/
theorem mutual_win_exists (t : Tournament 20)
    (h1 : allDifferentFirstRound t)
    (h2 : allSameTotal t) :
    existsMutualWin t := by
  sorry

end NUMINAMATH_CALUDE_mutual_win_exists_l188_18867


namespace NUMINAMATH_CALUDE_sin_double_angle_with_tan_l188_18823

theorem sin_double_angle_with_tan (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_with_tan_l188_18823


namespace NUMINAMATH_CALUDE_solution_sum_l188_18866

/-- The solutions of the quadratic equation 2x(5x-11) = -10 -/
def solutions (x : ℝ) : Prop :=
  2 * x * (5 * x - 11) = -10

/-- The rational form of the solutions -/
def rational_form (m n p : ℤ) (x : ℝ) : Prop :=
  (x = (m + Real.sqrt n) / p) ∨ (x = (m - Real.sqrt n) / p)

/-- The theorem statement -/
theorem solution_sum (m n p : ℤ) :
  (∀ x, solutions x → rational_form m n p x) →
  Int.gcd m (Int.gcd n p) = 1 →
  m + n + p = 242 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l188_18866


namespace NUMINAMATH_CALUDE_max_abs_sum_of_quadratic_coeffs_l188_18852

/-- Given a quadratic polynomial ax^2 + bx + c where |ax^2 + bx + c| ≤ 1 for all x in [-1,1],
    the maximum value of |a| + |b| + |c| is 3. -/
theorem max_abs_sum_of_quadratic_coeffs (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 ∧ ∃ a' b' c' : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a' * x^2 + b' * x + c'| ≤ 1) ∧ |a'| + |b'| + |c'| = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_of_quadratic_coeffs_l188_18852


namespace NUMINAMATH_CALUDE_log_has_zero_in_open_interval_l188_18838

theorem log_has_zero_in_open_interval :
  ∃ x, 0 < x ∧ x < 2 ∧ Real.log x = 0 := by sorry

end NUMINAMATH_CALUDE_log_has_zero_in_open_interval_l188_18838


namespace NUMINAMATH_CALUDE_not_sum_product_equal_neg_two_four_sum_product_equal_sqrt_two_plus_two_sqrt_two_sum_product_equal_relation_l188_18849

/-- Definition of sum-product equal number pair -/
def is_sum_product_equal (a b : ℝ) : Prop := a + b = a * b

/-- Theorem 1: (-2, 4) is not a sum-product equal number pair -/
theorem not_sum_product_equal_neg_two_four : ¬ is_sum_product_equal (-2) 4 := by sorry

/-- Theorem 2: (√2+2, √2) is a sum-product equal number pair -/
theorem sum_product_equal_sqrt_two_plus_two_sqrt_two : is_sum_product_equal (Real.sqrt 2 + 2) (Real.sqrt 2) := by sorry

/-- Theorem 3: For (m,n) where m,n ≠ 1, if it's a sum-product equal number pair, then m = n / (n-1) -/
theorem sum_product_equal_relation (m n : ℝ) (hm : m ≠ 1) (hn : n ≠ 1) :
  is_sum_product_equal m n → m = n / (n - 1) := by sorry

end NUMINAMATH_CALUDE_not_sum_product_equal_neg_two_four_sum_product_equal_sqrt_two_plus_two_sqrt_two_sum_product_equal_relation_l188_18849


namespace NUMINAMATH_CALUDE_lineup_count_l188_18830

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) (quarterbacks : ℕ) : ℕ :=
  let remaining := total_members - offensive_linemen - quarterbacks
  offensive_linemen * quarterbacks * remaining * (remaining - 1) * (remaining - 2)

/-- Theorem stating that the number of ways to choose the starting lineup is 5760 -/
theorem lineup_count :
  choose_lineup 12 4 2 = 5760 :=
by sorry

end NUMINAMATH_CALUDE_lineup_count_l188_18830


namespace NUMINAMATH_CALUDE_solution_difference_l188_18894

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ (2 * x^2 - 5 * x - 31) / (x - 3) = 3 * x + 8

-- Define the set of solutions
def solutions : Set ℝ :=
  {x | equation x}

-- State the theorem
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 2 * Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_solution_difference_l188_18894


namespace NUMINAMATH_CALUDE_commission_proof_l188_18851

/-- Calculates the commission earned from selling a coupe and an SUV --/
def calculate_commission (coupe_price : ℝ) (suv_multiplier : ℝ) (commission_rate : ℝ) : ℝ :=
  let suv_price := coupe_price * suv_multiplier
  let total_sales := coupe_price + suv_price
  total_sales * commission_rate

/-- Proves that the commission earned is $1,800 given the specified conditions --/
theorem commission_proof :
  calculate_commission 30000 2 0.02 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_commission_proof_l188_18851


namespace NUMINAMATH_CALUDE_advertising_cost_proof_l188_18871

/-- Proves that the advertising cost is $1000 given the problem conditions -/
theorem advertising_cost_proof 
  (total_customers : ℕ) 
  (purchase_rate : ℚ) 
  (item_cost : ℕ) 
  (profit : ℕ) :
  total_customers = 100 →
  purchase_rate = 4/5 →
  item_cost = 25 →
  profit = 1000 →
  (total_customers : ℚ) * purchase_rate * item_cost - profit = 1000 :=
by sorry

end NUMINAMATH_CALUDE_advertising_cost_proof_l188_18871


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l188_18821

theorem consecutive_integers_sum_of_squares (n : ℤ) : 
  n * (n + 1) * (n + 2) = 12 * (3 * n + 3) → 
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l188_18821


namespace NUMINAMATH_CALUDE_phase_shift_cosine_l188_18806

theorem phase_shift_cosine (x : Real) :
  let f : Real → Real := fun x ↦ 5 * Real.cos (x - π/3 + π/6)
  (∃ (k : Real), ∀ x, f x = 5 * Real.cos (x - k)) ∧
  (∀ k : Real, (∀ x, f x = 5 * Real.cos (x - k)) → k = π/6) :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_cosine_l188_18806


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_2023_l188_18839

theorem closest_multiple_of_15_to_2023 :
  ∀ k : ℤ, |15 * k - 2023| ≥ |2025 - 2023| := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_2023_l188_18839


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l188_18891

-- Define the quadratic function
def y (x m : ℝ) : ℝ := x^2 + 2*m*x - 3*m + 1

-- Define the conditions
def condition1 (p q : ℝ) : Prop := 4*p^2 + 9*q^2 = 2
def condition2 (x p q : ℝ) : Prop := (1/2)*x + 3*p*q = 1

-- State the theorem
theorem quadratic_function_minimum (x p q m : ℝ) :
  condition1 p q →
  condition2 x p q →
  (∀ x', y x' m ≥ 1) →
  (∃ x'', y x'' m = 1) →
  (m = -3 ∨ m = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l188_18891


namespace NUMINAMATH_CALUDE_max_distance_between_generatrices_l188_18846

/-- The maximum distance between two generatrices of two cones with a common base -/
theorem max_distance_between_generatrices (r h H : ℝ) (h_pos : 0 < h) (H_pos : 0 < H) (h_le_H : h ≤ H) :
  ∃ (d : ℝ), d = (h + H) * r / Real.sqrt (r^2 + H^2) ∧
  ∀ (d' : ℝ), d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_generatrices_l188_18846


namespace NUMINAMATH_CALUDE_average_grade_year_before_l188_18807

/-- Calculates the average grade for the year before last, given the following conditions:
  * The student took 6 courses last year with an average grade of 100 points
  * The student took 5 courses the year before
  * The average grade for the entire two-year period was 72 points
-/
theorem average_grade_year_before (courses_last_year : Nat) (avg_grade_last_year : ℝ)
  (courses_year_before : Nat) (avg_grade_two_years : ℝ) :
  courses_last_year = 6 →
  avg_grade_last_year = 100 →
  courses_year_before = 5 →
  avg_grade_two_years = 72 →
  (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) /
    (courses_year_before + courses_last_year) = avg_grade_two_years →
  avg_grade_year_before = 38.4 :=
by
  sorry

#check average_grade_year_before

end NUMINAMATH_CALUDE_average_grade_year_before_l188_18807


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l188_18816

theorem diophantine_equation_solutions :
  {(x, y) : ℕ × ℕ | 3 * x + 2 * y = 21 ∧ x > 0 ∧ y > 0} =
  {(5, 3), (3, 6), (1, 9)} := by
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l188_18816


namespace NUMINAMATH_CALUDE_negative_angle_quadrant_l188_18893

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

def is_in_second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 - 270 < α ∧ α < n * 360 - 180

theorem negative_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_quadrant (-α) := by
  sorry

end NUMINAMATH_CALUDE_negative_angle_quadrant_l188_18893


namespace NUMINAMATH_CALUDE_base5_multiplication_l188_18858

/-- Converts a base 5 number to base 10 --/
def baseConvert5To10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def baseConvert10To5 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 5 numbers --/
def multiplyBase5 (a b : ℕ) : ℕ :=
  baseConvert10To5 (baseConvert5To10 a * baseConvert5To10 b)

theorem base5_multiplication :
  multiplyBase5 132 22 = 4004 := by sorry

end NUMINAMATH_CALUDE_base5_multiplication_l188_18858


namespace NUMINAMATH_CALUDE_cuboid_volume_l188_18825

/-- The volume of a cuboid with given edge lengths -/
theorem cuboid_volume (l w h : ℝ) (hl : l = 3/2 + Real.sqrt (5/3)) 
  (hw : w = 2 + Real.sqrt (3/5)) (hh : h = π / 2) : 
  l * w * h = (3/2 + Real.sqrt (5/3)) * (2 + Real.sqrt (3/5)) * (π / 2) := by
  sorry

#check cuboid_volume

end NUMINAMATH_CALUDE_cuboid_volume_l188_18825


namespace NUMINAMATH_CALUDE_chinese_volleyball_team_probability_l188_18814

theorem chinese_volleyball_team_probability (p_japan p_usa : ℝ) 
  (h_japan : p_japan = 2/3)
  (h_usa : p_usa = 2/5)
  (h_independent : True)  -- This represents the independence assumption
  : (p_japan * (1 - p_usa) + (1 - p_japan) * p_usa) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_chinese_volleyball_team_probability_l188_18814


namespace NUMINAMATH_CALUDE_parallelogram_count_l188_18876

/-- The number of ways to choose 2 items from 4 -/
def choose_2_from_4 : ℕ := 6

/-- The number of horizontal lines -/
def horizontal_lines : ℕ := 4

/-- The number of vertical lines -/
def vertical_lines : ℕ := 4

/-- The number of parallelograms formed -/
def num_parallelograms : ℕ := choose_2_from_4 * choose_2_from_4

theorem parallelogram_count : num_parallelograms = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_l188_18876


namespace NUMINAMATH_CALUDE_system_solution_l188_18805

/- Define the system of equations -/
def equation1 (x y : ℚ) : Prop := 4 * x - 7 * y = -14
def equation2 (x y : ℚ) : Prop := 5 * x + 3 * y = -7

/- Define the solution -/
def solution_x : ℚ := -91/47
def solution_y : ℚ := -42/47

/- Theorem statement -/
theorem system_solution :
  equation1 solution_x solution_y ∧ equation2 solution_x solution_y :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l188_18805


namespace NUMINAMATH_CALUDE_largest_811_triple_l188_18845

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

/-- Converts a list of base-8 digits to a base-10 number -/
def fromBase8 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a list of digits to a base-10 number -/
def toBase10 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is an 8-11 triple -/
def is811Triple (m : ℕ) : Prop :=
  let base8Digits := toBase8 m
  toBase10 base8Digits = 3 * m

/-- The largest 8-11 triple -/
def largestTriple : ℕ := 705

theorem largest_811_triple :
  is811Triple largestTriple ∧
  ∀ m : ℕ, m > largestTriple → ¬is811Triple m :=
by sorry

end NUMINAMATH_CALUDE_largest_811_triple_l188_18845


namespace NUMINAMATH_CALUDE_sum_10_is_35_l188_18824

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  third_term : 2 * a 3 = 5
  sum_4_12 : a 4 + a 12 = 9

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1))

/-- The theorem to be proved -/
theorem sum_10_is_35 (seq : ArithmeticSequence) : sum_n seq 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_10_is_35_l188_18824


namespace NUMINAMATH_CALUDE_archies_backyard_sod_l188_18842

/-- The area of sod needed for Archie's backyard -/
def sod_area (backyard_length backyard_width shed_length shed_width : ℕ) : ℕ :=
  backyard_length * backyard_width - shed_length * shed_width

/-- Theorem stating the correct amount of sod needed for Archie's backyard -/
theorem archies_backyard_sod : sod_area 20 13 3 5 = 245 := by
  sorry

end NUMINAMATH_CALUDE_archies_backyard_sod_l188_18842


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_2010_l188_18862

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2010 is 1. -/
theorem sum_of_digits_8_pow_2010 : ∃ n : ℕ, 8^2010 = 100 * n + 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_2010_l188_18862


namespace NUMINAMATH_CALUDE_birthday_attendees_l188_18890

theorem birthday_attendees (n : ℕ) : 
  (12 * (n + 2) = 16 * n) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_birthday_attendees_l188_18890


namespace NUMINAMATH_CALUDE_quadratic_integral_inequality_l188_18884

/-- For real numbers a, b, c, let f(x) = ax^2 + bx + c. 
    Prove that ∫_{-1}^1 (1 - x^2){f'(x)}^2 dx ≤ 6∫_{-1}^1 {f(x)}^2 dx -/
theorem quadratic_integral_inequality (a b c : ℝ) : 
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  ∫ x in (-1)..1, (1 - x^2) * (deriv f x)^2 ≤ 6 * ∫ x in (-1)..1, (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integral_inequality_l188_18884


namespace NUMINAMATH_CALUDE_books_difference_l188_18853

def summer_reading (june july august : ℕ) : Prop :=
  june = 8 ∧ july = 2 * june ∧ june + july + august = 37

theorem books_difference (june july august : ℕ) 
  (h : summer_reading june july august) : july - august = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_difference_l188_18853


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l188_18857

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 9) :
  let sphere_diameter := outer_cube_edge
  let inscribed_cube_space_diagonal := sphere_diameter
  let inscribed_cube_edge := inscribed_cube_space_diagonal / Real.sqrt 3
  let inscribed_cube_volume := inscribed_cube_edge ^ 3
  inscribed_cube_volume = 81 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l188_18857


namespace NUMINAMATH_CALUDE_limit_of_function_l188_18810

theorem limit_of_function (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π/3| ∧ |x - π/3| < δ →
    |(1 - 2 * Real.cos x) / Real.sin (π - 3 * x) + Real.sqrt 3 / 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_function_l188_18810


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l188_18808

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

-- Theorem statement
theorem intersection_complement_equals_singleton :
  N ∩ (U \ M) = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l188_18808


namespace NUMINAMATH_CALUDE_expression_equals_one_l188_18860

theorem expression_equals_one : 
  (2001 * 2021 + 100) * (1991 * 2031 + 400) / (2011^4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l188_18860


namespace NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l188_18874

/-- Given four points A, B, X, Y on a Cartesian plane where AB is parallel to XY,
    prove that the y-coordinate of Y is 5. -/
theorem parallel_segments_y_coordinate (A B X Y : ℝ × ℝ) : 
  A = (-6, 2) →
  B = (2, -2) →
  X = (-2, 10) →
  Y.1 = 8 →
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) →
  Y.2 = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l188_18874


namespace NUMINAMATH_CALUDE_mikes_candies_l188_18875

theorem mikes_candies (initial_candies : ℕ) : 
  (initial_candies > 0) →
  (initial_candies % 4 = 0) →
  (∃ (sister_took : ℕ), 1 ≤ sister_took ∧ sister_took ≤ 4 ∧
    5 + sister_took = initial_candies * 3 / 4 * 2 / 3 - 24) →
  initial_candies = 64 := by
sorry

end NUMINAMATH_CALUDE_mikes_candies_l188_18875


namespace NUMINAMATH_CALUDE_root_transformation_l188_18887

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) → 
  ((3*r₁)^3 - 9*(3*r₁)^2 + 216 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 216 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 216 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l188_18887


namespace NUMINAMATH_CALUDE_symmetric_line_and_distance_theorem_l188_18840

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x + y + 3 = 0
def l₂ (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the symmetric line l₃
def l₃ (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, -1)

-- Define the line m
def m (x y : ℝ) : Prop := 3 * x + 4 * y + 10 = 0 ∨ x = -2

-- Theorem statement
theorem symmetric_line_and_distance_theorem :
  (∀ x y : ℝ, l₃ x y ↔ l₁ x (-y)) ∧
  (l₂ P.1 P.2 ∧ l₃ P.1 P.2) ∧
  (m P.1 P.2 ∧ 
   ∀ x y : ℝ, m x y → 
     (x * x + y * y = 4 ∨ 
      (3 * x + 4 * y + 10)^2 / (3 * 3 + 4 * 4) = 4)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_and_distance_theorem_l188_18840


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l188_18865

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l188_18865


namespace NUMINAMATH_CALUDE_jonathan_phone_time_l188_18800

/-- 
Given that Jonathan spends some hours on his phone daily, half of which is spent on social media,
and he spends 28 hours on social media in a week, prove that he spends 8 hours on his phone daily.
-/
theorem jonathan_phone_time (x : ℝ) 
  (daily_phone_time : x > 0) 
  (social_media_half : x / 2 * 7 = 28) : 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_jonathan_phone_time_l188_18800


namespace NUMINAMATH_CALUDE_gcd_of_48_72_120_l188_18848

theorem gcd_of_48_72_120 : Nat.gcd 48 (Nat.gcd 72 120) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_48_72_120_l188_18848


namespace NUMINAMATH_CALUDE_horner_method_proof_l188_18841

def horner_polynomial (x : ℝ) : ℝ := 
  ((((2 * x - 5) * x - 4) * x + 3) * x - 6) * x + 7

theorem horner_method_proof : horner_polynomial 5 = 2677 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l188_18841


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l188_18882

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallelLine : Line → Line → Prop)

-- Define the intersection of two planes resulting in a line
variable (planeIntersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection
  (l m : Line) (α β : Plane)
  (h1 : planeIntersection α β = l)
  (h2 : parallelLinePlane m α)
  (h3 : parallelLinePlane m β) :
  parallelLine m l :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l188_18882


namespace NUMINAMATH_CALUDE_given_number_eq_scientific_repr_l188_18878

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number -0.000032 -/
def given_number : ℝ := -0.000032

/-- The scientific notation representation of the given number -/
def scientific_repr : ScientificNotation :=
  { coefficient := -3.2
    exponent := -5
    property := by sorry }

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_repr :
  given_number = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_eq_scientific_repr_l188_18878


namespace NUMINAMATH_CALUDE_evaluate_expression_l188_18837

theorem evaluate_expression : 
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 3 = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l188_18837


namespace NUMINAMATH_CALUDE_pen_pencil_distribution_l188_18864

theorem pen_pencil_distribution (P : ℕ) : 
  (∃ (k : ℕ), P = 20 * k) ↔ 
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 1340 / x = y ∧ P / x = y ∧ x ≤ 20 ∧ 
   ∀ (z : ℕ), z > x → (1340 / z ≠ P / z ∨ 1340 % z ≠ 0 ∨ P % z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_pen_pencil_distribution_l188_18864


namespace NUMINAMATH_CALUDE_max_sphere_ratio_l188_18844

/-- Represents the configuration of spheres within two cones as described in the problem -/
structure SpheresInCones where
  r : ℝ  -- radius of the first two identical spheres
  x : ℝ  -- radius of the third sphere
  R : ℝ  -- radius of the base of the cones
  h : ℝ  -- height of each cone
  s : ℝ  -- slant height of each cone

/-- The conditions given in the problem -/
def problem_conditions (config : SpheresInCones) : Prop :=
  config.r > 0 ∧
  config.x > 0 ∧
  config.R > 0 ∧
  config.h > 0 ∧
  config.s > 0 ∧
  config.h = config.s / 2 ∧
  config.R = 3 * config.r

/-- The theorem stating the maximum ratio of the third sphere's radius to the first sphere's radius -/
theorem max_sphere_ratio (config : SpheresInCones) 
  (h : problem_conditions config) :
  ∃ (t : ℝ), t = config.x / config.r ∧ 
             t ≤ (7 - Real.sqrt 22) / 3 ∧
             ∀ (t' : ℝ), t' = config.x / config.r → t' ≤ t :=
sorry

end NUMINAMATH_CALUDE_max_sphere_ratio_l188_18844


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l188_18802

-- Define the steps of linear regression analysis
inductive RegressionStep
  | collectData
  | drawScatterPlot
  | calculateEquation
  | interpretEquation

-- Define a type for sequences of regression steps
def RegressionSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : RegressionSequence :=
  [RegressionStep.collectData, RegressionStep.drawScatterPlot, 
   RegressionStep.calculateEquation, RegressionStep.interpretEquation]

-- Define a property for variables being linearly related
def linearlyRelated (x y : ℝ → ℝ) : Prop := sorry

-- Theorem stating that given linearly related variables, 
-- the correct sequence is as defined above
theorem correct_regression_sequence 
  (x y : ℝ → ℝ) 
  (h : linearlyRelated x y) : 
  correctSequence = 
    [RegressionStep.collectData, RegressionStep.drawScatterPlot, 
     RegressionStep.calculateEquation, RegressionStep.interpretEquation] :=
by
  sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l188_18802


namespace NUMINAMATH_CALUDE_only_group_D_forms_triangle_l188_18873

/-- Triangle inequality theorem for a set of three lengths -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Groups of line segments -/
def group_A : (ℝ × ℝ × ℝ) := (3, 8, 5)
def group_B : (ℝ × ℝ × ℝ) := (12, 5, 6)
def group_C : (ℝ × ℝ × ℝ) := (5, 5, 10)
def group_D : (ℝ × ℝ × ℝ) := (15, 10, 7)

/-- Theorem: Only group D can form a triangle -/
theorem only_group_D_forms_triangle :
  ¬(triangle_inequality group_A.1 group_A.2.1 group_A.2.2) ∧
  ¬(triangle_inequality group_B.1 group_B.2.1 group_B.2.2) ∧
  ¬(triangle_inequality group_C.1 group_C.2.1 group_C.2.2) ∧
  (triangle_inequality group_D.1 group_D.2.1 group_D.2.2) :=
by sorry

end NUMINAMATH_CALUDE_only_group_D_forms_triangle_l188_18873


namespace NUMINAMATH_CALUDE_max_matching_segment_l188_18815

/-- Two sequences with periods 7 and 13 respectively -/
def Sequence1 : ℕ → ℕ := sorry
def Sequence2 : ℕ → ℕ := sorry

/-- The period of Sequence1 is 7 -/
axiom period_seq1 : ∀ n : ℕ, Sequence1 n = Sequence1 (n + 7)

/-- The period of Sequence2 is 13 -/
axiom period_seq2 : ∀ n : ℕ, Sequence2 n = Sequence2 (n + 13)

/-- Definition of matching initial segment -/
def matching_segment (len : ℕ) : Prop :=
  ∀ i : ℕ, i < len → Sequence1 i = Sequence2 i

/-- The theorem to be proved -/
theorem max_matching_segment :
  (∃ len : ℕ, matching_segment len ∧ len = 18) ∧
  (∀ len : ℕ, len > 18 → ¬matching_segment len) :=
sorry

end NUMINAMATH_CALUDE_max_matching_segment_l188_18815


namespace NUMINAMATH_CALUDE_perpendicular_line_polar_equation_l188_18809

/-- The polar equation of a line perpendicular to the polar axis and passing through 
    the center of the circle ρ = 6cosθ -/
theorem perpendicular_line_polar_equation (ρ θ : ℝ) : 
  (ρ = 6 * Real.cos θ → ∃ c, c = 3 ∧ ρ * Real.cos θ = c) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_polar_equation_l188_18809


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_fgh_supermarkets_count_proof_l188_18855

theorem fgh_supermarkets_count : ℕ → ℕ → ℕ → Prop :=
  fun us_count canada_count total =>
    (us_count = 37) →
    (us_count = canada_count + 14) →
    (total = us_count + canada_count) →
    (total = 60)

-- The proof goes here
theorem fgh_supermarkets_count_proof : fgh_supermarkets_count 37 23 60 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_fgh_supermarkets_count_proof_l188_18855


namespace NUMINAMATH_CALUDE_limit_f_at_zero_l188_18834

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 2) - Real.sqrt 2) / Real.sin (3 * x)

theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds ((Real.sqrt 2) / 24)) :=
sorry

end NUMINAMATH_CALUDE_limit_f_at_zero_l188_18834


namespace NUMINAMATH_CALUDE_polynomial_value_relation_l188_18881

theorem polynomial_value_relation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) :
  6 * x^2 + 9 * y + 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_relation_l188_18881


namespace NUMINAMATH_CALUDE_inequality_proof_l188_18819

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1/(2*a) + 1/(2*b) + 1/(2*c) ≥ 1/(b+c) + 1/(c+a) + 1/(a+b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l188_18819
