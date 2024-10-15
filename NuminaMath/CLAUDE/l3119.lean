import Mathlib

namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l3119_311948

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 20300 = ScientificNotation.mk 2.03 4 sorry := by sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l3119_311948


namespace NUMINAMATH_CALUDE_first_year_x_exceeds_y_l3119_311981

def commodity_x_price (year : ℕ) : ℚ :=
  420/100 + (year - 2001) * 30/100

def commodity_y_price (year : ℕ) : ℚ :=
  440/100 + (year - 2001) * 20/100

theorem first_year_x_exceeds_y :
  (∀ y : ℕ, 2001 < y ∧ y < 2004 → commodity_x_price y ≤ commodity_y_price y) ∧
  commodity_x_price 2004 > commodity_y_price 2004 :=
by sorry

end NUMINAMATH_CALUDE_first_year_x_exceeds_y_l3119_311981


namespace NUMINAMATH_CALUDE_third_term_is_eight_thirds_l3119_311970

/-- The sequence defined by a_n = n - 1/n -/
def a (n : ℕ) : ℚ := n - 1 / n

/-- Theorem: The third term of the sequence a_n is 8/3 -/
theorem third_term_is_eight_thirds : a 3 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_eight_thirds_l3119_311970


namespace NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l3119_311949

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines
variable (parallel : Line → Line → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallelToPlane : Line → Plane → Prop)

-- Define a function that checks if a line is parallel to countless lines in a plane
variable (parallelToCountlessLines : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_countless_lines
  (l b : Line) (α : Plane)
  (h1 : parallel l b)
  (h2 : subset b α) :
  parallelToCountlessLines l α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l3119_311949


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3119_311939

theorem rectangle_dimensions (x y : ℕ+) : 
  x * y = 36 ∧ x + y = 13 → (x = 9 ∧ y = 4) ∨ (x = 4 ∧ y = 9) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3119_311939


namespace NUMINAMATH_CALUDE_lanie_hourly_rate_l3119_311911

/-- Calculates the hourly rate given the fraction of hours worked, total hours, and weekly salary -/
def hourly_rate (fraction_worked : ℚ) (total_hours : ℕ) (weekly_salary : ℕ) : ℚ :=
  weekly_salary / (fraction_worked * total_hours)

/-- Proves that given the specified conditions, the hourly rate is $15 -/
theorem lanie_hourly_rate :
  let fraction_worked : ℚ := 4/5
  let total_hours : ℕ := 40
  let weekly_salary : ℕ := 480
  hourly_rate fraction_worked total_hours weekly_salary = 15 := by
sorry

end NUMINAMATH_CALUDE_lanie_hourly_rate_l3119_311911


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3119_311967

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop :=
  ∃ k > 0, ∀ (s1 s2 : ℝ × ℝ), s1 ∈ t1 → s2 ∈ t2 → ‖s1.1 - s1.2‖ = k * ‖s2.1 - s2.2‖

theorem similar_triangles_side_length 
  (PQR STU : Set (ℝ × ℝ))
  (h_similar : similar_triangles PQR STU)
  (h_PQ : ∃ PQ ∈ PQR, ‖PQ.1 - PQ.2‖ = 7)
  (h_PR : ∃ PR ∈ PQR, ‖PR.1 - PR.2‖ = 9)
  (h_ST : ∃ ST ∈ STU, ‖ST.1 - ST.2‖ = 4.2)
  : ∃ SU ∈ STU, ‖SU.1 - SU.2‖ = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3119_311967


namespace NUMINAMATH_CALUDE_chinese_multiplication_puzzle_l3119_311938

theorem chinese_multiplication_puzzle : 
  ∃! (a b d e p q r : ℕ), 
    (0 ≤ a ∧ a ≤ 9) ∧ 
    (0 ≤ b ∧ b ≤ 9) ∧ 
    (0 ≤ d ∧ d ≤ 9) ∧ 
    (0 ≤ e ∧ e ≤ 9) ∧ 
    (0 ≤ p ∧ p ≤ 9) ∧ 
    (0 ≤ q ∧ q ≤ 9) ∧ 
    (0 ≤ r ∧ r ≤ 9) ∧ 
    (a ≠ b) ∧ 
    (10 * a + b) * (10 * a + b) = 10000 * d + 1000 * e + 100 * p + 10 * q + r ∧
    (10 * a + b) * (10 * a + b) ≡ (10 * a + b) [MOD 100] ∧
    d = 5 ∧ e = 0 ∧ p = 6 ∧ q = 2 ∧ r = 5 ∧ a = 2 ∧ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_chinese_multiplication_puzzle_l3119_311938


namespace NUMINAMATH_CALUDE_water_height_in_cone_l3119_311968

theorem water_height_in_cone (r h : ℝ) (water_ratio : ℝ) :
  r = 16 →
  h = 96 →
  water_ratio = 1/4 →
  (water_ratio * (1/3 * π * r^2 * h) = 1/3 * π * r^2 * (48 * Real.rpow 2 (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_water_height_in_cone_l3119_311968


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l3119_311928

theorem jelly_bean_distribution (total : ℕ) (x y : ℕ) : 
  total = 1200 →
  x + y = total →
  x = 3 * y - 400 →
  x = 800 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l3119_311928


namespace NUMINAMATH_CALUDE_rectangle_area_2_by_3_l3119_311929

/-- A rectangle with width and length in centimeters -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle in square centimeters -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: The area of a rectangle with width 2 cm and length 3 cm is 6 cm² -/
theorem rectangle_area_2_by_3 : 
  let r : Rectangle := { width := 2, length := 3 }
  area r = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_2_by_3_l3119_311929


namespace NUMINAMATH_CALUDE_rectangle_fitting_theorem_l3119_311921

/-- A rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Predicate to check if one rectangle fits inside another -/
def fits_inside (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨ 
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- The main theorem -/
theorem rectangle_fitting_theorem (n : ℕ) (h : n ≥ 2018) 
  (S : Finset Rectangle) 
  (hS : S.card = n + 1) 
  (hSides : ∀ r ∈ S, r.width ∈ Finset.range (n + 1) ∧ r.height ∈ Finset.range (n + 1)) :
  ∃ (A B C : Rectangle), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ 
    fits_inside A B ∧ fits_inside B C :=
by sorry

end NUMINAMATH_CALUDE_rectangle_fitting_theorem_l3119_311921


namespace NUMINAMATH_CALUDE_equilateral_triangle_to_three_layered_quadrilateral_l3119_311924

/-- Represents a polygon with a specified number of sides -/
structure Polygon where
  sides : ℕ
  deriving Repr

/-- Represents a folded shape -/
structure FoldedShape where
  shape : Polygon
  layers : ℕ
  deriving Repr

/-- Represents an equilateral triangle -/
def EquilateralTriangle : Polygon :=
  { sides := 3 }

/-- Represents a quadrilateral -/
def Quadrilateral : Polygon :=
  { sides := 4 }

/-- Folding operation that transforms one shape into another -/
def fold (start : Polygon) (result : FoldedShape) : Prop :=
  sorry

/-- Theorem stating that an equilateral triangle can be folded into a three-layered quadrilateral -/
theorem equilateral_triangle_to_three_layered_quadrilateral :
  ∃ (result : FoldedShape), 
    result.shape = Quadrilateral ∧ 
    result.layers = 3 ∧ 
    fold EquilateralTriangle result :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_to_three_layered_quadrilateral_l3119_311924


namespace NUMINAMATH_CALUDE_babysitter_scream_ratio_l3119_311940

-- Define the variables and constants
def current_rate : ℚ := 16
def new_rate : ℚ := 12
def scream_cost : ℚ := 3
def hours : ℚ := 6
def cost_difference : ℚ := 18

-- Define the theorem
theorem babysitter_scream_ratio :
  let current_cost := current_rate * hours
  let new_cost_without_screams := new_rate * hours
  let new_cost_with_screams := new_cost_without_screams + cost_difference
  let scream_total_cost := new_cost_with_screams - new_cost_without_screams
  let num_screams := scream_total_cost / scream_cost
  num_screams / hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_babysitter_scream_ratio_l3119_311940


namespace NUMINAMATH_CALUDE_cube_root_of_fourth_root_l3119_311972

theorem cube_root_of_fourth_root (a : ℝ) (h : a > 0) :
  (a * a^(1/4))^(1/3) = a^(5/12) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_fourth_root_l3119_311972


namespace NUMINAMATH_CALUDE_singing_percentage_is_32_l3119_311904

/-- Represents the rehearsal schedule and calculates the percentage of time spent singing -/
def rehearsal_schedule (total_time warm_up_time notes_time words_time : ℕ) : ℚ :=
  let singing_time := total_time - warm_up_time - notes_time - words_time
  (singing_time : ℚ) / total_time * 100

/-- Theorem stating that the percentage of time spent singing is 32% -/
theorem singing_percentage_is_32 :
  ∃ (words_time : ℕ), rehearsal_schedule 75 6 30 words_time = 32 := by
  sorry


end NUMINAMATH_CALUDE_singing_percentage_is_32_l3119_311904


namespace NUMINAMATH_CALUDE_parallelogram_rotational_symmetry_l3119_311982

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- A parallelogram is a quadrilateral with opposite sides parallel -/
def is_parallelogram (p : Polygon) : Prop :=
  p.vertices.length = 4 ∧
  ∃ (a b c d : ℝ × ℝ), p.vertices = [a, b, c, d] ∧
    (b.1 - a.1, b.2 - a.2) = (d.1 - c.1, d.2 - c.2) ∧
    (c.1 - b.1, c.2 - b.2) = (a.1 - d.1, a.2 - d.2)

/-- Rotation by 180 degrees around a point -/
def rotate_180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- A polygon coincides with itself after 180-degree rotation -/
def coincides_after_rotation (p : Polygon) : Prop :=
  ∃ (center : ℝ × ℝ), 
    ∀ v ∈ p.vertices, (rotate_180 v center) ∈ p.vertices

theorem parallelogram_rotational_symmetry :
  ∀ (p : Polygon), is_parallelogram p → coincides_after_rotation p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_rotational_symmetry_l3119_311982


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3119_311905

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 6 * x > 23) → x ≤ -3 ∧ 7 - 6 * (-3) > 23 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3119_311905


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3119_311960

theorem sin_120_degrees : 
  ∃ (Q : ℝ × ℝ) (E : ℝ × ℝ),
    (Q.1^2 + Q.2^2 = 1) ∧  -- Q is on the unit circle
    (Real.cos (2*π/3) = Q.1 ∧ Real.sin (2*π/3) = Q.2) ∧  -- Q is at 120°
    (E.2 = 0 ∧ (Q.1 - E.1) * (Q.1 - E.1) + Q.2 * Q.2 = (Q.1 - E.1)^2) →  -- E is the foot of perpendicular
    Real.sin (2*π/3) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3119_311960


namespace NUMINAMATH_CALUDE_specific_circle_distances_l3119_311979

/-- Two circles with given radii and distance between centers -/
structure TwoCircles where
  radius1 : ℝ
  radius2 : ℝ
  center_distance : ℝ

/-- The minimum and maximum distances between points on two circles -/
def circle_distances (c : TwoCircles) : ℝ × ℝ :=
  (c.center_distance - c.radius1 - c.radius2, c.center_distance + c.radius1 + c.radius2)

/-- Theorem stating the minimum and maximum distances for specific circle configuration -/
theorem specific_circle_distances :
  let c : TwoCircles := ⟨2, 3, 8⟩
  circle_distances c = (3, 13) := by sorry

end NUMINAMATH_CALUDE_specific_circle_distances_l3119_311979


namespace NUMINAMATH_CALUDE_trig_identity_l3119_311980

theorem trig_identity (θ : Real) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3119_311980


namespace NUMINAMATH_CALUDE_set_operations_l3119_311983

def A : Set ℝ := {x | 2 * x - 8 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 4}) ∧
  ((Aᶜ ∪ B) = {x : ℝ | 0 < x}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l3119_311983


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3119_311942

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + d) * (x + e)) →
  (∀ x, x^2 + 11*x - 60 = (x + e) * (x - f)) →
  d + e + f = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3119_311942


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l3119_311975

/-- Given two intersecting circles, prove that the equation of the line 
    containing their intersection points can be found by subtracting 
    the equations of the circles. -/
theorem intersection_line_of_circles 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 10) 
  (h2 : (x-1)^2 + (y-3)^2 = 20) : 
  x + 3*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l3119_311975


namespace NUMINAMATH_CALUDE_percentage_with_diploma_l3119_311901

theorem percentage_with_diploma (total : ℝ) 
  (no_diploma_but_choice : ℝ) 
  (no_choice_but_diploma_ratio : ℝ) 
  (job_of_choice : ℝ) :
  no_diploma_but_choice = 0.1 * total →
  no_choice_but_diploma_ratio = 0.15 →
  job_of_choice = 0.4 * total →
  ∃ (with_diploma : ℝ), 
    with_diploma = 0.39 * total ∧
    with_diploma = (job_of_choice - no_diploma_but_choice) + 
                   (no_choice_but_diploma_ratio * (total - job_of_choice)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_with_diploma_l3119_311901


namespace NUMINAMATH_CALUDE_emergency_vehicle_reachable_area_l3119_311945

/-- The area reachable by an emergency vehicle in a desert -/
theorem emergency_vehicle_reachable_area 
  (road_speed : ℝ) 
  (sand_speed : ℝ) 
  (time : ℝ) 
  (h_road_speed : road_speed = 60) 
  (h_sand_speed : sand_speed = 10) 
  (h_time : time = 5/60) : 
  ∃ (area : ℝ), area = 25 + 25 * Real.pi / 36 ∧ 
  area = (road_speed * time)^2 + 4 * (Real.pi * (sand_speed * time)^2 / 4) := by
sorry

end NUMINAMATH_CALUDE_emergency_vehicle_reachable_area_l3119_311945


namespace NUMINAMATH_CALUDE_southton_time_capsule_depth_l3119_311965

theorem southton_time_capsule_depth :
  let southton_depth : ℝ := 9
  let northton_depth : ℝ := 48
  northton_depth = 4 * southton_depth + 12 →
  southton_depth = 9 := by
sorry

end NUMINAMATH_CALUDE_southton_time_capsule_depth_l3119_311965


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3119_311915

theorem fractional_inequality_solution_set (x : ℝ) :
  1 / (x - 1) < -1 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3119_311915


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l3119_311908

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 101 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 101 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l3119_311908


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3119_311906

/-- The line y = x + 1 intersects the circle x^2 + y^2 = 1. -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), y = x + 1 ∧ x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3119_311906


namespace NUMINAMATH_CALUDE_expand_product_l3119_311900

theorem expand_product (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x^3) + 14 * x^5) = 3 / x^3 + 6 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3119_311900


namespace NUMINAMATH_CALUDE_total_votes_proof_l3119_311950

theorem total_votes_proof (total_votes : ℕ) (votes_against : ℕ) : 
  (votes_against = total_votes * 40 / 100) →
  (total_votes - votes_against = votes_against + 70) →
  total_votes = 350 := by
sorry

end NUMINAMATH_CALUDE_total_votes_proof_l3119_311950


namespace NUMINAMATH_CALUDE_same_monotonicity_intervals_l3119_311919

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x + 2

def f' (x : ℝ) : ℝ := 3*x^2 - 12*x + 9

theorem same_monotonicity_intervals :
  (∀ x ∈ Set.Icc 1 2, (∀ y ∈ Set.Icc 1 2, x ≤ y → f x ≥ f y ∧ f' x ≥ f' y)) ∧
  (∀ x ∈ Set.Ioi 3, (∀ y ∈ Set.Ioi 3, x ≤ y → f x ≤ f y ∧ f' x ≤ f' y)) ∧
  (∀ a b : ℝ, a < b ∧ 
    ((a < 1 ∧ b > 1) ∨ (a < 2 ∧ b > 2) ∨ (a < 3 ∧ b > 3)) →
    ¬(∀ x ∈ Set.Icc a b, (∀ y ∈ Set.Icc a b, x ≤ y → 
      (f x ≤ f y ∧ f' x ≤ f' y) ∨ (f x ≥ f y ∧ f' x ≥ f' y)))) :=
by sorry

#check same_monotonicity_intervals

end NUMINAMATH_CALUDE_same_monotonicity_intervals_l3119_311919


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_max_sphere_radius_squared_achievable_l3119_311998

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of three cones and a sphere -/
structure ConeConfiguration where
  cone : Cone
  axisIntersectionDistance : ℝ
  sphereRadius : ℝ

/-- Checks if the configuration is valid -/
def isValidConfiguration (config : ConeConfiguration) : Prop :=
  config.cone.baseRadius = 4 ∧
  config.cone.height = 10 ∧
  config.axisIntersectionDistance = 5

/-- Theorem stating the maximum possible value of r^2 -/
theorem max_sphere_radius_squared (config : ConeConfiguration) 
  (h : isValidConfiguration config) : 
  config.sphereRadius ^ 2 ≤ 100 / 29 := by
  sorry

/-- Theorem stating that the maximum value is achievable -/
theorem max_sphere_radius_squared_achievable : 
  ∃ (config : ConeConfiguration), isValidConfiguration config ∧ config.sphereRadius ^ 2 = 100 / 29 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_max_sphere_radius_squared_achievable_l3119_311998


namespace NUMINAMATH_CALUDE_diagonalSum_is_377_l3119_311966

/-- A hexagon inscribed in a circle with given side lengths -/
structure InscribedHexagon where
  -- Define the side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ
  -- Conditions on side lengths
  AB_length : AB = 41
  other_sides : BC = 91 ∧ CD = 91 ∧ DE = 91 ∧ EF = 91 ∧ FA = 91
  -- Ensure it's inscribed in a circle (this is implicit and we don't prove it)
  inscribed : True

/-- The sum of diagonal lengths from vertex A in the inscribed hexagon -/
def diagonalSum (h : InscribedHexagon) : ℝ :=
  let AC := sorry
  let AD := sorry
  let AE := sorry
  AC + AD + AE

/-- Theorem stating that the sum of diagonal lengths from A is 377 -/
theorem diagonalSum_is_377 (h : InscribedHexagon) : diagonalSum h = 377 := by
  sorry

end NUMINAMATH_CALUDE_diagonalSum_is_377_l3119_311966


namespace NUMINAMATH_CALUDE_jovanas_shells_l3119_311978

/-- The total amount of shells Jovana has after her friends add to her collection -/
def total_shells (initial : ℕ) (friend1 : ℕ) (friend2 : ℕ) : ℕ :=
  initial + friend1 + friend2

/-- Theorem stating that Jovana's total shells equal 37 pounds -/
theorem jovanas_shells :
  total_shells 5 15 17 = 37 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l3119_311978


namespace NUMINAMATH_CALUDE_tan_difference_identity_l3119_311973

theorem tan_difference_identity (n : ℝ) : 
  Real.tan ((n + 1) * π / 180) - Real.tan (n * π / 180) = 
  Real.sin (π / 180) / (Real.cos (n * π / 180) * Real.cos ((n + 1) * π / 180)) := by
sorry

end NUMINAMATH_CALUDE_tan_difference_identity_l3119_311973


namespace NUMINAMATH_CALUDE_triangle_areas_l3119_311951

/-- Given points Q, A, C, and D on the x-y coordinate plane, prove the areas of triangles QCA and ACD. -/
theorem triangle_areas (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  let D : ℝ × ℝ := (3, 0)
  
  let area_QCA := (45 - 3 * p) / 2
  let area_ACD := 22.5

  (∃ (area_function : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ),
    area_function Q C A = area_QCA ∧
    area_function A C D = area_ACD) := by
  sorry

end NUMINAMATH_CALUDE_triangle_areas_l3119_311951


namespace NUMINAMATH_CALUDE_last_two_digits_theorem_l3119_311917

theorem last_two_digits_theorem (n : ℕ) (h : Odd n) :
  (2^(2*n) * (2^(2*n + 1) - 1)) % 100 = 28 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_theorem_l3119_311917


namespace NUMINAMATH_CALUDE_special_lines_intersect_l3119_311952

/-- Given a triangle ABC with incircle center I and excircle center I_A -/
structure Triangle :=
  (A B C I I_A : EuclideanSpace ℝ (Fin 2))

/-- Line passing through orthocenters of triangles formed by vertices, incircle center, and excircle center -/
def special_line (T : Triangle) (v : Fin 3) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The theorem states that the three special lines intersect at a single point -/
theorem special_lines_intersect (T : Triangle) :
  ∃! P, P ∈ (special_line T 0) ∧ P ∈ (special_line T 1) ∧ P ∈ (special_line T 2) :=
sorry

end NUMINAMATH_CALUDE_special_lines_intersect_l3119_311952


namespace NUMINAMATH_CALUDE_simplify_quadratic_radical_l3119_311933

theorem simplify_quadratic_radical (x y : ℝ) (h : x * y < 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) :=
sorry

end NUMINAMATH_CALUDE_simplify_quadratic_radical_l3119_311933


namespace NUMINAMATH_CALUDE_arevalo_dinner_change_l3119_311987

/-- The change calculation for the Arevalo family dinner --/
theorem arevalo_dinner_change (salmon_cost black_burger_cost chicken_katsu_cost : ℝ)
  (service_charge_rate tip_rate : ℝ) (amount_paid : ℝ)
  (h1 : salmon_cost = 40)
  (h2 : black_burger_cost = 15)
  (h3 : chicken_katsu_cost = 25)
  (h4 : service_charge_rate = 0.1)
  (h5 : tip_rate = 0.05)
  (h6 : amount_paid = 100) :
  amount_paid - (salmon_cost + black_burger_cost + chicken_katsu_cost +
    (salmon_cost + black_burger_cost + chicken_katsu_cost) * service_charge_rate +
    (salmon_cost + black_burger_cost + chicken_katsu_cost) * tip_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_arevalo_dinner_change_l3119_311987


namespace NUMINAMATH_CALUDE_gcd_lcm_perfect_square_l3119_311916

theorem gcd_lcm_perfect_square (a b c : ℕ+) 
  (h : ∃ k : ℕ, (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a : ℕ) = k^2) : 
  ∃ m : ℕ, (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a : ℕ) = m^2 := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_perfect_square_l3119_311916


namespace NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_l3119_311912

/-- Represents a mixed repeating decimal as a pair of natural numbers (m, k),
    where m is the number of non-repeating digits after the decimal point,
    and k is the length of the repeating part. -/
structure MixedRepeatingDecimal where
  m : ℕ
  k : ℕ+

/-- Represents an irreducible fraction as a pair of integers (p, q) -/
structure IrreducibleFraction where
  p : ℤ
  q : ℤ
  q_pos : q > 0
  coprime : Int.gcd p q = 1

/-- States that a given irreducible fraction represents a mixed repeating decimal -/
def represents (f : IrreducibleFraction) (d : MixedRepeatingDecimal) : Prop := sorry

/-- The main theorem: If an irreducible fraction represents a mixed repeating decimal,
    then its denominator is divisible by 2 or 5 or both -/
theorem mixed_repeating_decimal_denominator
  (f : IrreducibleFraction)
  (d : MixedRepeatingDecimal)
  (h : represents f d) :
  ∃ (a b : ℕ), f.q = 2^a * 5^b * (2^d.k.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_l3119_311912


namespace NUMINAMATH_CALUDE_field_walking_distance_reduction_l3119_311977

theorem field_walking_distance_reduction : 
  let field_width : ℝ := 6
  let field_height : ℝ := 8
  let daniel_distance := field_width + field_height
  let rachel_distance := Real.sqrt (field_width^2 + field_height^2)
  let percentage_reduction := (daniel_distance - rachel_distance) / daniel_distance * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ abs (percentage_reduction - 29) < ε :=
by sorry

end NUMINAMATH_CALUDE_field_walking_distance_reduction_l3119_311977


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3119_311907

theorem polynomial_factorization (m : ℝ) :
  (∀ x : ℝ, x^2 - 5*x + m = (x - 3) * (x - 2)) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3119_311907


namespace NUMINAMATH_CALUDE_group_average_age_l3119_311990

/-- Given a group of people, prove that their current average age is as calculated -/
theorem group_average_age 
  (n : ℕ) -- number of people in the group
  (youngest_age : ℕ) -- age of the youngest person
  (past_average : ℚ) -- average age when the youngest was born
  (h1 : n = 7) -- there are 7 people
  (h2 : youngest_age = 4) -- the youngest is 4 years old
  (h3 : past_average = 26) -- average age when youngest was born was 26
  : (n : ℚ) * ((n - 1 : ℚ) * past_average + n * (youngest_age : ℚ)) / n = 184 / 7 := by
  sorry

end NUMINAMATH_CALUDE_group_average_age_l3119_311990


namespace NUMINAMATH_CALUDE_shaded_area_13x5_grid_l3119_311922

/-- Represents a rectangular grid with a shaded region --/
structure ShadedGrid where
  width : ℕ
  height : ℕ
  shaded_area : ℝ

/-- Calculates the area of the shaded region in the grid --/
def calculate_shaded_area (grid : ShadedGrid) : ℝ :=
  let total_area := grid.width * grid.height
  let triangle_area := (grid.width * grid.height) / 2
  total_area - triangle_area

/-- Theorem stating that the shaded area of a 13x5 grid with an excluded triangle is 32.5 --/
theorem shaded_area_13x5_grid :
  ∃ (grid : ShadedGrid),
    grid.width = 13 ∧
    grid.height = 5 ∧
    calculate_shaded_area grid = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_13x5_grid_l3119_311922


namespace NUMINAMATH_CALUDE_common_roots_product_l3119_311914

/-- Given two cubic equations with two common roots, prove their product is 4∛5 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ), 
    (u^3 + C*u - 20 = 0) ∧ 
    (v^3 + C*v - 20 = 0) ∧ 
    (w^3 + C*w - 20 = 0) ∧
    (u^3 + D*u^2 - 40 = 0) ∧ 
    (v^3 + D*v^2 - 40 = 0) ∧ 
    (t^3 + D*t^2 - 40 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 4 * Real.rpow 5 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l3119_311914


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l3119_311964

theorem power_of_product_with_negative (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l3119_311964


namespace NUMINAMATH_CALUDE_closed_map_from_compact_preimage_l3119_311926

open Set
open TopologicalSpace
open MetricSpace
open ContinuousMap

theorem closed_map_from_compact_preimage
  {X Y : Type*} [MetricSpace X] [MetricSpace Y]
  (f : C(X, Y))
  (h : ∀ (K : Set Y), IsCompact K → IsCompact (f ⁻¹' K)) :
  ∀ (C : Set X), IsClosed C → IsClosed (f '' C) :=
by sorry

end NUMINAMATH_CALUDE_closed_map_from_compact_preimage_l3119_311926


namespace NUMINAMATH_CALUDE_power_of_negative_power_l3119_311932

theorem power_of_negative_power (x : ℝ) : (-x^4)^3 = -x^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_power_l3119_311932


namespace NUMINAMATH_CALUDE_inequality_proof_l3119_311986

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3119_311986


namespace NUMINAMATH_CALUDE_line_equation_forms_l3119_311992

theorem line_equation_forms (A B C : ℝ) :
  ∃ (φ p : ℝ), ∀ (x y : ℝ),
    A * x + B * y + C = 0 ↔ x * Real.cos φ + y * Real.sin φ = p :=
by sorry

end NUMINAMATH_CALUDE_line_equation_forms_l3119_311992


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l3119_311961

theorem sin_product_equals_one_sixteenth : 
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l3119_311961


namespace NUMINAMATH_CALUDE_prob_at_most_one_mistake_value_l3119_311931

/-- Probability of correct answer for the first question -/
def p1 : ℚ := 3/4

/-- Probability of correct answer for the second question -/
def p2 : ℚ := 1/2

/-- Probability of correct answer for the third question -/
def p3 : ℚ := 1/6

/-- Probability of at most one mistake in the first three questions -/
def prob_at_most_one_mistake : ℚ := 
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem prob_at_most_one_mistake_value : 
  prob_at_most_one_mistake = 11/24 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_mistake_value_l3119_311931


namespace NUMINAMATH_CALUDE_problem_solution_l3119_311969

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) :
  (a^2 + b^2 = 7) ∧ (a < b → a - b = -Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3119_311969


namespace NUMINAMATH_CALUDE_running_match_participants_l3119_311997

theorem running_match_participants : 
  ∀ (n : ℕ), 
  (∃ (participant : ℕ), 
    participant ≤ n ∧ 
    participant > 0 ∧
    n - 1 = 25) →
  n = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_running_match_participants_l3119_311997


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3119_311927

/-- A triangle with two sides of lengths 3 and 4, and the third side length being a root of x^2 - 12x + 35 = 0 has a perimeter of 12. -/
theorem triangle_perimeter : ∃ (a b c : ℝ), 
  a = 3 ∧ b = 4 ∧ c^2 - 12*c + 35 = 0 ∧ 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3119_311927


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3119_311937

theorem rectangle_dimensions :
  ∀ (x y : ℝ), 
    x > 0 ∧ y > 0 →
    x * y = 1/9 →
    y = 3 * x →
    x = Real.sqrt 3 / 9 ∧ y = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3119_311937


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l3119_311918

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 4032 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b'*x + 4032 = (x + r) * (x + s)) ∧
  b = 128 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l3119_311918


namespace NUMINAMATH_CALUDE_circle_area_increase_l3119_311935

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.01 * r
  let old_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - old_area) / old_area = 0.0201 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3119_311935


namespace NUMINAMATH_CALUDE_stimulus_savings_theorem_l3119_311995

def stimulus_distribution (initial_amount : ℚ) : ℚ :=
  let wife_share := initial_amount / 4
  let after_wife := initial_amount - wife_share
  let first_son_share := after_wife * 3 / 8
  let after_first_son := after_wife - first_son_share
  let second_son_share := after_first_son * 25 / 100
  let after_second_son := after_first_son - second_son_share
  let third_son_share := 500
  let after_third_son := after_second_son - third_son_share
  let daughter_share := after_third_son * 15 / 100
  let savings := after_third_son - daughter_share
  savings

theorem stimulus_savings_theorem :
  stimulus_distribution 4000 = 770.3125 := by sorry

end NUMINAMATH_CALUDE_stimulus_savings_theorem_l3119_311995


namespace NUMINAMATH_CALUDE_triple_transmission_more_reliable_l3119_311955

/-- Represents a transmission channel with error probabilities α and β -/
structure TransmissionChannel where
  α : Real
  β : Real
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of decoding as 0 using single transmission when sending 0 -/
def singleTransmissionProb (channel : TransmissionChannel) : Real :=
  1 - channel.α

/-- Probability of decoding as 0 using triple transmission when sending 0 -/
def tripleTransmissionProb (channel : TransmissionChannel) : Real :=
  3 * channel.α * (1 - channel.α)^2 + (1 - channel.α)^3

/-- Theorem stating that triple transmission is more reliable than single transmission for decoding 0 when α < 0.5 -/
theorem triple_transmission_more_reliable (channel : TransmissionChannel) 
    (h : channel.α < 0.5) : 
    singleTransmissionProb channel < tripleTransmissionProb channel := by
  sorry

end NUMINAMATH_CALUDE_triple_transmission_more_reliable_l3119_311955


namespace NUMINAMATH_CALUDE_aizhai_bridge_investment_l3119_311930

/-- Converts a number to scientific notation with a specified number of significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

/-- Checks if two scientific notations are equal -/
def scientific_notation_eq (a : ℝ × ℤ) (b : ℝ × ℤ) : Prop :=
  sorry

theorem aizhai_bridge_investment :
  let investment := 1650000000
  let sig_figs := 3
  let result := to_scientific_notation investment sig_figs
  scientific_notation_eq result (1.65, 9) :=
sorry

end NUMINAMATH_CALUDE_aizhai_bridge_investment_l3119_311930


namespace NUMINAMATH_CALUDE_inverse_trig_inequality_l3119_311956

theorem inverse_trig_inequality : 
  Real.arctan (-5/4) < Real.arcsin (-2/5) ∧ Real.arcsin (-2/5) < Real.arccos (-3/4) := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_inequality_l3119_311956


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3119_311954

def f (a : ℝ) (x : ℝ) := x^2 - 2*a*x

theorem sufficient_not_necessary (a : ℝ) :
  (a < 0 → ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∃ a, 0 ≤ a ∧ a ≤ 1 ∧ ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3119_311954


namespace NUMINAMATH_CALUDE_cos_sin_eq_linear_solution_exists_l3119_311994

theorem cos_sin_eq_linear_solution_exists :
  ∃ x : ℝ, -2/3 ≤ x ∧ x ≤ 2/3 ∧ 
  -3*π/2 ≤ x ∧ x ≤ 3*π/2 ∧
  Real.cos (Real.sin x) = 3*x/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_eq_linear_solution_exists_l3119_311994


namespace NUMINAMATH_CALUDE_largest_n_is_max_l3119_311909

/-- The largest positive integer n such that there exist n real numbers
    satisfying the given inequality. -/
def largest_n : ℕ := 31

/-- The condition that must be satisfied by the n real numbers. -/
def satisfies_condition (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n →
    (1 + x i * x j)^2 ≤ 0.99 * (1 + (x i)^2) * (1 + (x j)^2)

/-- The main theorem stating that largest_n is indeed the largest such n. -/
theorem largest_n_is_max :
  (∃ x : ℕ → ℝ, satisfies_condition x largest_n) ∧
  (∀ m : ℕ, m > largest_n → ¬∃ x : ℕ → ℝ, satisfies_condition x m) :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_max_l3119_311909


namespace NUMINAMATH_CALUDE_gadget_sales_sum_l3119_311934

/-- The sum of an arithmetic sequence with first term 2, common difference 4, and 15 terms -/
def arithmetic_sum : ℕ := sorry

/-- The first term of the sequence -/
def a₁ : ℕ := 2

/-- The common difference of the sequence -/
def d : ℕ := 4

/-- The number of terms in the sequence -/
def n : ℕ := 15

/-- The last term of the sequence -/
def aₙ : ℕ := a₁ + (n - 1) * d

theorem gadget_sales_sum : arithmetic_sum = 450 := by sorry

end NUMINAMATH_CALUDE_gadget_sales_sum_l3119_311934


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3119_311944

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 8
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3119_311944


namespace NUMINAMATH_CALUDE_daria_credit_card_debt_l3119_311957

/-- Calculates the discounted price of an item --/
def discountedPrice (price : ℚ) (discountPercent : ℚ) : ℚ :=
  price * (1 - discountPercent / 100)

/-- Represents Daria's furniture purchases --/
structure Purchases where
  couch : ℚ
  couchDiscount : ℚ
  table : ℚ
  tableDiscount : ℚ
  lamp : ℚ
  rug : ℚ
  rugDiscount : ℚ
  bookshelf : ℚ
  bookshelfDiscount : ℚ

/-- Calculates the total cost of purchases after discounts --/
def totalCost (p : Purchases) : ℚ :=
  discountedPrice p.couch p.couchDiscount +
  discountedPrice p.table p.tableDiscount +
  p.lamp +
  discountedPrice p.rug p.rugDiscount +
  discountedPrice p.bookshelf p.bookshelfDiscount

/-- Theorem: Daria owes $610 on her credit card before interest --/
theorem daria_credit_card_debt (p : Purchases) (savings : ℚ) :
  p.couch = 750 →
  p.couchDiscount = 10 →
  p.table = 100 →
  p.tableDiscount = 5 →
  p.lamp = 50 →
  p.rug = 200 →
  p.rugDiscount = 15 →
  p.bookshelf = 150 →
  p.bookshelfDiscount = 20 →
  savings = 500 →
  totalCost p - savings = 610 := by
  sorry


end NUMINAMATH_CALUDE_daria_credit_card_debt_l3119_311957


namespace NUMINAMATH_CALUDE_hiking_rate_theorem_l3119_311902

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  rate_up : ℝ
  time : ℝ
  route_down_length : ℝ
  rate_down_multiplier : ℝ

/-- The hiking scenario satisfies the given conditions -/
def satisfies_conditions (h : HikingScenario) : Prop :=
  h.time = 2 ∧
  h.route_down_length = 12 ∧
  h.rate_down_multiplier = 1.5

/-- The theorem stating that under the given conditions, the rate going up is 4 miles per day -/
theorem hiking_rate_theorem (h : HikingScenario) 
  (hc : satisfies_conditions h) : h.rate_up = 4 := by
  sorry

end NUMINAMATH_CALUDE_hiking_rate_theorem_l3119_311902


namespace NUMINAMATH_CALUDE_scientific_notation_of_million_l3119_311996

/-- Prove that 1.6369 million is equal to 1.6369 × 10^6 -/
theorem scientific_notation_of_million (x : ℝ) : 
  x * 1000000 = x * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_million_l3119_311996


namespace NUMINAMATH_CALUDE_total_books_combined_l3119_311971

theorem total_books_combined (bryan_books_per_shelf : ℕ) (bryan_shelves : ℕ) 
  (alyssa_books_per_shelf : ℕ) (alyssa_shelves : ℕ) : 
  bryan_books_per_shelf = 56 → 
  bryan_shelves = 9 → 
  alyssa_books_per_shelf = 73 → 
  alyssa_shelves = 12 → 
  bryan_books_per_shelf * bryan_shelves + alyssa_books_per_shelf * alyssa_shelves = 1380 := by
  sorry

end NUMINAMATH_CALUDE_total_books_combined_l3119_311971


namespace NUMINAMATH_CALUDE_min_students_theorem_l3119_311925

/-- The minimum number of students that can be divided into either 18 or 24 teams 
    with a maximum difference of 2 students between team sizes. -/
def min_students : ℕ := 70

/-- Checks if a number can be divided into a given number of teams
    with a maximum difference of 2 students between team sizes. -/
def can_divide (n : ℕ) (teams : ℕ) : Prop :=
  ∃ (base_size : ℕ), 
    (n ≥ base_size * teams) ∧ 
    (n ≤ (base_size + 2) * teams)

theorem min_students_theorem : 
  (can_divide min_students 18) ∧ 
  (can_divide min_students 24) ∧ 
  (∀ m : ℕ, m < min_students → ¬(can_divide m 18 ∧ can_divide m 24)) :=
sorry

end NUMINAMATH_CALUDE_min_students_theorem_l3119_311925


namespace NUMINAMATH_CALUDE_students_excelling_both_tests_l3119_311913

theorem students_excelling_both_tests 
  (total : ℕ) 
  (physical : ℕ) 
  (intellectual : ℕ) 
  (neither : ℕ) 
  (h1 : total = 50) 
  (h2 : physical = 40) 
  (h3 : intellectual = 31) 
  (h4 : neither = 4) :
  physical + intellectual - (total - neither) = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_excelling_both_tests_l3119_311913


namespace NUMINAMATH_CALUDE_joe_paint_usage_l3119_311936

theorem joe_paint_usage (total_paint : ℝ) (used_paint : ℝ) 
  (h1 : total_paint = 360)
  (h2 : used_paint = 225) : 
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint + 
    (1 / 2) * (total_paint - first_week_fraction * total_paint) = used_paint ∧
    first_week_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l3119_311936


namespace NUMINAMATH_CALUDE_product_from_sum_and_difference_l3119_311991

theorem product_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 19) 
  (diff_eq : x - y = 5) : 
  x * y = 84 := by
sorry

end NUMINAMATH_CALUDE_product_from_sum_and_difference_l3119_311991


namespace NUMINAMATH_CALUDE_equation_solutions_l3119_311974

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x + 1)^2 = 8 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 - x - 6 = 0 ↔ x = -3/2 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3119_311974


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3119_311976

-- Define sets A and B as functions of a
def A (a : ℝ) : Set ℝ := {4, a^2}
def B (a : ℝ) : Set ℝ := {a-6, 1+a, 9}

-- Theorem statement
theorem union_of_A_and_B :
  ∃ a : ℝ, (A a ∩ B a = {9}) ∧ (A a ∪ B a = {-9, -2, 4, 9}) :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3119_311976


namespace NUMINAMATH_CALUDE_tan_three_expression_zero_l3119_311923

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (2 - 2 * Real.cos θ) / Real.sin θ - Real.sin θ / (2 + 2 * Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_zero_l3119_311923


namespace NUMINAMATH_CALUDE_octagon_non_intersecting_diagonals_l3119_311988

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of non-intersecting diagonals in a star pattern for a regular polygon -/
def nonIntersectingDiagonals (p : RegularPolygon n) : ℕ := n

/-- Theorem: For an octagon, the number of non-intersecting diagonals in a star pattern
    is equal to the number of sides -/
theorem octagon_non_intersecting_diagonals :
  ∀ (p : RegularPolygon 8), nonIntersectingDiagonals p = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_non_intersecting_diagonals_l3119_311988


namespace NUMINAMATH_CALUDE_min_value_theorem_l3119_311985

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b^2 * c^3 = 256) : 
  a^2 + 8*a*b + 16*b^2 + 2*c^5 ≥ 768 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
    a₀ * b₀^2 * c₀^3 = 256 ∧ 
    a₀^2 + 8*a₀*b₀ + 16*b₀^2 + 2*c₀^5 = 768 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3119_311985


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3119_311920

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3119_311920


namespace NUMINAMATH_CALUDE_min_value_and_range_l3119_311941

theorem min_value_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ min) ∧
  (∀ (x : ℝ), (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ |2*x - 1| - |x + 1|) ↔ 
    -7 ≤ x ∧ x ≤ 11) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_range_l3119_311941


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3119_311953

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- Shorter leg length
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3119_311953


namespace NUMINAMATH_CALUDE_detergent_loads_theorem_l3119_311962

/-- Represents the number of loads of laundry that can be washed with one bottle of detergent -/
def loads_per_bottle (regular_price sale_price cost_per_load : ℚ) : ℚ :=
  (2 * sale_price) / (2 * cost_per_load)

/-- Theorem stating the number of loads that can be washed with one bottle of detergent -/
theorem detergent_loads_theorem (regular_price sale_price cost_per_load : ℚ) 
  (h1 : regular_price = 25)
  (h2 : sale_price = 20)
  (h3 : cost_per_load = 1/4) :
  loads_per_bottle regular_price sale_price cost_per_load = 80 := by
  sorry

#eval loads_per_bottle 25 20 (1/4)

end NUMINAMATH_CALUDE_detergent_loads_theorem_l3119_311962


namespace NUMINAMATH_CALUDE_blue_notebook_cost_l3119_311993

theorem blue_notebook_cost (total_spent : ℕ) (total_notebooks : ℕ) 
  (red_notebooks : ℕ) (red_price : ℕ) (green_notebooks : ℕ) (green_price : ℕ) :
  total_spent = 37 →
  total_notebooks = 12 →
  red_notebooks = 3 →
  red_price = 4 →
  green_notebooks = 2 →
  green_price = 2 →
  ∃ (blue_notebooks : ℕ) (blue_price : ℕ),
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks ∧
    blue_price = 3 ∧
    total_spent = red_notebooks * red_price + green_notebooks * green_price + blue_notebooks * blue_price :=
by sorry

end NUMINAMATH_CALUDE_blue_notebook_cost_l3119_311993


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3119_311984

theorem trigonometric_inequality (x : ℝ) : 
  0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧
  5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3119_311984


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3119_311943

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3119_311943


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l3119_311959

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (total : ℕ) (sample_size : ℕ) (first_sample : ℕ) : ℕ :=
  first_sample + (sample_size - 1) * (total / sample_size)

/-- Theorem stating the largest number in the specific systematic sample -/
theorem largest_number_in_sample :
  largest_sample_number 120 10 7 = 115 := by
  sorry

#eval largest_sample_number 120 10 7

end NUMINAMATH_CALUDE_largest_number_in_sample_l3119_311959


namespace NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l3119_311963

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (white : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 8) :
  (green + yellow) / (green + yellow + white) = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l3119_311963


namespace NUMINAMATH_CALUDE_problem_solution_l3119_311910

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 2 < 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 ≥ 1

-- Theorem statement
theorem problem_solution :
  (¬p) ∧ q :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3119_311910


namespace NUMINAMATH_CALUDE_min_vertical_distance_l3119_311989

-- Define the two functions
def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x : ℝ), vertical_distance x = 8 ∧ 
  ∀ (y : ℝ), vertical_distance y ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l3119_311989


namespace NUMINAMATH_CALUDE_f_decreasing_after_2_l3119_311958

def f (x : ℝ) : ℝ := -(x - 2)^2 + 3

theorem f_decreasing_after_2 :
  ∀ x₁ x₂ : ℝ, 2 < x₁ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_after_2_l3119_311958


namespace NUMINAMATH_CALUDE_sum_of_roots_l3119_311903

/-- The function f(x) = x³ + 3x² + 6x + 14 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

/-- Theorem: If f(a) + f(b) = 20, then a + b = -2 -/
theorem sum_of_roots (a b : ℝ) (h : f a + f b = 20) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3119_311903


namespace NUMINAMATH_CALUDE_combined_salaries_l3119_311999

/-- The combined salaries of four employees given the salary of the fifth and the average of all five -/
theorem combined_salaries (salary_C average_salary : ℕ) 
  (hC : salary_C = 14000)
  (havg : average_salary = 8600) :
  salary_C + 4 * average_salary - 5 * average_salary = 29000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l3119_311999


namespace NUMINAMATH_CALUDE_problem_solution_l3119_311946

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem problem_solution (n : ℕ) : 2 * n * sum_of_digits (3 * n) = 2022 → n = 337 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3119_311946


namespace NUMINAMATH_CALUDE_tangent_line_minimum_sum_l3119_311947

theorem tangent_line_minimum_sum (m n : ℝ) : 
  m > 0 → 
  n > 0 → 
  (∃ x : ℝ, (1/Real.exp 1) * x + m + 1 = Real.log x - n + 2 ∧ 
             (1/Real.exp 1) = 1/x) → 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/a + 1/b ≥ 1/m + 1/n) →
  1/m + 1/n = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_sum_l3119_311947
