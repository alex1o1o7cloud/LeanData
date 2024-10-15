import Mathlib

namespace NUMINAMATH_CALUDE_exam_time_allocation_l185_18599

/-- Represents the time spent on type A problems in an exam -/
def time_on_type_A (total_time minutes : ℕ) (total_questions type_A_questions : ℕ) : ℕ :=
  let type_B_questions := total_questions - type_A_questions
  let time_ratio := 2  -- Type A takes twice as long as Type B
  let total_time_units := type_A_questions * time_ratio + type_B_questions
  (total_time * minutes * type_A_questions * time_ratio) / total_time_units

/-- Theorem: Given the exam conditions, the time spent on type A problems is 120 minutes -/
theorem exam_time_allocation :
  time_on_type_A 3 60 200 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l185_18599


namespace NUMINAMATH_CALUDE_martha_lasagna_cost_l185_18558

/-- The cost of ingredients for Martha's lasagna -/
def lasagna_cost (cheese_price meat_price pasta_price tomato_price : ℝ) : ℝ :=
  1.5 * cheese_price + 0.55 * meat_price + 0.28 * pasta_price + 2.2 * tomato_price

/-- Theorem stating the total cost of ingredients for Martha's lasagna -/
theorem martha_lasagna_cost :
  lasagna_cost 6.30 8.55 2.40 1.79 = 18.76 := by
  sorry


end NUMINAMATH_CALUDE_martha_lasagna_cost_l185_18558


namespace NUMINAMATH_CALUDE_sector_perimeter_l185_18596

theorem sector_perimeter (r : ℝ) (S : ℝ) (h1 : r = 2) (h2 : S = 8) :
  let α := 2 * S / r^2
  let L := r * α
  r + r + L = 12 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l185_18596


namespace NUMINAMATH_CALUDE_remainder_of_product_l185_18580

theorem remainder_of_product (n : ℕ) (h : n = 67545) : (n * 11) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_l185_18580


namespace NUMINAMATH_CALUDE_line_passes_through_point_l185_18591

/-- The line equation passes through the point (3, 1) for all values of m -/
theorem line_passes_through_point :
  ∀ (m : ℝ), (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l185_18591


namespace NUMINAMATH_CALUDE_old_manufacturing_cost_l185_18583

/-- Proves that the old manufacturing cost was $65 given the conditions of the problem -/
theorem old_manufacturing_cost (selling_price : ℝ) (new_manufacturing_cost : ℝ) : 
  selling_price = 100 →
  new_manufacturing_cost = 50 →
  (selling_price - new_manufacturing_cost) / selling_price = 0.5 →
  (selling_price - 0.65 * selling_price) = 65 :=
by sorry

end NUMINAMATH_CALUDE_old_manufacturing_cost_l185_18583


namespace NUMINAMATH_CALUDE_sin_double_angle_for_point_l185_18576

theorem sin_double_angle_for_point (a : ℝ) (θ : ℝ) (h : a > 0) :
  let P : ℝ × ℝ := (-4 * a, 3 * a)
  (∃ r : ℝ, r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ) →
  Real.sin (2 * θ) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_point_l185_18576


namespace NUMINAMATH_CALUDE_expression_simplification_l185_18547

theorem expression_simplification 
  (p q : ℝ) (x : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  (x^(3/p) - x^(3/q)) / ((x^(1/p) + x^(1/q))^2 - 2*x^(1/q)*(x^(1/q) + x^(1/p))) + 
  x^(1/p) / (x^((q-p)/(p*q)) + 1) = x^(1/p) + x^(1/q) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l185_18547


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l185_18571

/-- A geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- A sequence is monotonically increasing -/
def MonotonicallyIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ s (n + 1)

/-- The condition q > 1 is neither necessary nor sufficient for a geometric sequence to be monotonically increasing -/
theorem geometric_sequence_increasing_condition (a q : ℝ) :
  ¬(((q > 1) ↔ MonotonicallyIncreasing (geometricSequence a q))) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l185_18571


namespace NUMINAMATH_CALUDE_polyhedron_ball_covering_inequality_l185_18513

/-- A non-degenerate polyhedron -/
structure Polyhedron where
  nondegenerate : Bool

/-- A collection of balls covering a polyhedron -/
structure BallCovering (P : Polyhedron) where
  n : ℕ
  V : ℝ
  covers_surface : Bool

/-- Theorem: For any non-degenerate polyhedron, there exists a positive constant
    such that any ball covering satisfies the given inequality -/
theorem polyhedron_ball_covering_inequality (P : Polyhedron) 
    (h : P.nondegenerate = true) :
    ∃ c : ℝ, c > 0 ∧ 
    ∀ (B : BallCovering P), B.covers_surface → B.n > c / (B.V ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_ball_covering_inequality_l185_18513


namespace NUMINAMATH_CALUDE_cost_per_person_l185_18512

def total_cost : ℚ := 13500
def num_friends : ℕ := 15

theorem cost_per_person :
  total_cost / num_friends = 900 :=
sorry

end NUMINAMATH_CALUDE_cost_per_person_l185_18512


namespace NUMINAMATH_CALUDE_age_problem_l185_18572

theorem age_problem (age1 age2 : ℕ) : 
  age1 + age2 = 63 →
  age1 = 2 * (age2 - (age1 - age2)) →
  (age1 = 36 ∧ age2 = 27) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l185_18572


namespace NUMINAMATH_CALUDE_colorings_count_l185_18517

/-- Represents the number of colors available --/
def num_colors : ℕ := 4

/-- Represents the number of triangles in the configuration --/
def num_triangles : ℕ := 4

/-- Represents the number of ways to color the first triangle --/
def first_triangle_colorings : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Represents the number of ways to color each subsequent triangle --/
def subsequent_triangle_colorings : ℕ := (num_colors - 1) * (num_colors - 2)

/-- The total number of possible colorings for the entire configuration --/
def total_colorings : ℕ := first_triangle_colorings * subsequent_triangle_colorings^(num_triangles - 1)

theorem colorings_count :
  total_colorings = 5184 :=
sorry

end NUMINAMATH_CALUDE_colorings_count_l185_18517


namespace NUMINAMATH_CALUDE_leadership_arrangements_l185_18505

/-- Represents the number of teachers -/
def num_teachers : ℕ := 5

/-- Represents the number of extracurricular groups -/
def num_groups : ℕ := 3

/-- Represents the maximum number of leaders per group -/
def max_leaders_per_group : ℕ := 2

/-- Represents that teachers A and B cannot lead alone -/
def ab_cannot_lead_alone : Prop := True

/-- The number of different leadership arrangements -/
def num_arrangements : ℕ := 54

/-- Theorem stating that the number of different leadership arrangements
    for the given conditions is equal to 54 -/
theorem leadership_arrangements :
  num_teachers = 5 ∧
  num_groups = 3 ∧
  max_leaders_per_group = 2 ∧
  ab_cannot_lead_alone →
  num_arrangements = 54 := by
  sorry

end NUMINAMATH_CALUDE_leadership_arrangements_l185_18505


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l185_18555

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 7 → x ≥ 4 ∧ 4 < 3*4 - 7 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l185_18555


namespace NUMINAMATH_CALUDE_circular_track_length_l185_18577

/-- Represents a circular track with two runners --/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Represents a meeting point of the runners --/
structure MeetingPoint where
  distance1 : ℝ  -- Distance run by runner 1
  distance2 : ℝ  -- Distance run by runner 2

/-- The theorem to be proved --/
theorem circular_track_length
  (track : CircularTrack)
  (first_meeting : MeetingPoint)
  (second_meeting : MeetingPoint) :
  (first_meeting.distance1 = 100) →
  (second_meeting.distance2 - first_meeting.distance2 = 150) →
  (track.runner1_speed > 0) →
  (track.runner2_speed > 0) →
  (track.length = 500) :=
by sorry

end NUMINAMATH_CALUDE_circular_track_length_l185_18577


namespace NUMINAMATH_CALUDE_percentage_grade_c_l185_18514

def scores : List Nat := [49, 58, 65, 77, 84, 70, 88, 94, 55, 82, 60, 86, 68, 74, 99, 81, 73, 79, 53, 91]

def is_grade_c (score : Nat) : Bool :=
  78 ≤ score ∧ score ≤ 86

def count_grade_c (scores : List Nat) : Nat :=
  scores.filter is_grade_c |>.length

theorem percentage_grade_c : 
  (count_grade_c scores : Rat) / (scores.length : Rat) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_grade_c_l185_18514


namespace NUMINAMATH_CALUDE_gcd_lcm_consecutive_naturals_l185_18525

theorem gcd_lcm_consecutive_naturals (m : ℕ) (h : m > 0) :
  let n := m + 1
  (Nat.gcd m n = 1) ∧ (Nat.lcm m n = m * n) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_consecutive_naturals_l185_18525


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l185_18503

/-- Represents a right circular cone -/
structure RightCircularCone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ  -- Angle from a fixed reference line on the cone surface

/-- Calculates the shortest distance between two points on the surface of a cone -/
def shortestDistanceOnCone (cone : RightCircularCone) (p1 p2 : ConePoint) : ℝ := sorry

/-- Theorem stating the shortest distance between two specific points on a cone -/
theorem shortest_distance_on_specific_cone :
  let cone : RightCircularCone := ⟨450, 300 * Real.sqrt 3⟩
  let p1 : ConePoint := ⟨200, 0⟩
  let p2 : ConePoint := ⟨300 * Real.sqrt 3, π⟩
  shortestDistanceOnCone cone p1 p2 = 200 + 300 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l185_18503


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l185_18540

-- Define the universal set U as the real numbers
def U := ℝ

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(|x|)}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (3 - x)}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {t : ℝ | 1 ≤ t ∧ t < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l185_18540


namespace NUMINAMATH_CALUDE_investment_problem_l185_18534

/-- The investment problem --/
theorem investment_problem (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →  -- Vishal invested 10% more than Trishul
  raghu = 2500 →  -- Raghu invested Rs. 2500
  vishal + trishul + raghu = 7225 →  -- Total sum of investments
  trishul < raghu →  -- Trishul invested less than Raghu
  (raghu - trishul) / raghu * 100 = 10 :=  -- Percentage difference
by sorry

end NUMINAMATH_CALUDE_investment_problem_l185_18534


namespace NUMINAMATH_CALUDE_four_by_four_cube_unpainted_l185_18537

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_units - (cube.size * cube.size * 6 - (cube.size - 2) * (cube.size - 2) * 6)

/-- Theorem stating that a 4x4x4 cube with 4 painted squares per face has 52 unpainted unit cubes -/
theorem four_by_four_cube_unpainted :
  let cube : PaintedCube := { size := 4, total_units := 64, painted_per_face := 4 }
  unpainted_cubes cube = 52 := by
  sorry

end NUMINAMATH_CALUDE_four_by_four_cube_unpainted_l185_18537


namespace NUMINAMATH_CALUDE_cone_volume_increase_l185_18506

/-- The volume of a cone increases by a factor of 8 when its height and radius are doubled -/
theorem cone_volume_increase (r h V : ℝ) (r' h' V' : ℝ) : 
  V = (1/3) * π * r^2 * h →  -- Original volume
  r' = 2*r →                 -- New radius is doubled
  h' = 2*h →                 -- New height is doubled
  V' = (1/3) * π * r'^2 * h' →  -- New volume
  V' = 8*V := by
sorry


end NUMINAMATH_CALUDE_cone_volume_increase_l185_18506


namespace NUMINAMATH_CALUDE_sibling_age_difference_l185_18501

/-- Given three siblings whose ages are in the ratio 3:2:1 and whose total combined age is 90 years,
    the difference between the eldest sibling's age and the youngest sibling's age is 30 years. -/
theorem sibling_age_difference (x : ℝ) (h1 : 3*x + 2*x + x = 90) : 3*x - x = 30 := by
  sorry

end NUMINAMATH_CALUDE_sibling_age_difference_l185_18501


namespace NUMINAMATH_CALUDE_rotating_triangle_path_length_l185_18560

/-- The total path length of point A in a rotating triangle -/
theorem rotating_triangle_path_length (α : ℝ) (h1 : 0 < α) (h2 : α < π / 3) :
  let triangle_rotation := (2 / 3 * π * (1 + Real.sin α) - 2 * α)
  (100 - 1) / 3 * triangle_rotation = 22 * π * (1 + Real.sin α) - 66 * α :=
by sorry

end NUMINAMATH_CALUDE_rotating_triangle_path_length_l185_18560


namespace NUMINAMATH_CALUDE_cos_sin_shift_l185_18569

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (x + 2 * Real.pi / 3) = Real.sin (Real.pi / 3 - (x + Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_shift_l185_18569


namespace NUMINAMATH_CALUDE_octal_sum_equals_2351_l185_18539

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- Theorem stating that the sum of 1457₈ and 672₈ in base 8 is 2351₈ -/
theorem octal_sum_equals_2351 :
  let a := octal_to_decimal [7, 5, 4, 1]  -- 1457₈
  let b := octal_to_decimal [2, 7, 6]     -- 672₈
  decimal_to_octal (a + b) = [1, 5, 3, 2] -- 2351₈
  := by sorry

end NUMINAMATH_CALUDE_octal_sum_equals_2351_l185_18539


namespace NUMINAMATH_CALUDE_projection_coordinates_l185_18553

/-- The plane equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The projection of a point onto a plane -/
def projection (p : Point3D) (plane : Plane) : Point3D :=
  sorry

theorem projection_coordinates :
  let p := Point3D.mk 1 2 (-1)
  let plane := Plane.mk 3 (-1) 2 (-4)
  let proj := projection p plane
  proj.x = 29 / 14 ∧ proj.y = 23 / 14 ∧ proj.z = -2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_projection_coordinates_l185_18553


namespace NUMINAMATH_CALUDE_cost_of_four_birdhouses_l185_18520

/-- The cost to build a given number of birdhouses -/
def cost_of_birdhouses (num_birdhouses : ℕ) : ℚ :=
  let planks_per_house : ℕ := 7
  let nails_per_house : ℕ := 20
  let cost_per_plank : ℚ := 3
  let cost_per_nail : ℚ := 1/20
  num_birdhouses * (planks_per_house * cost_per_plank + nails_per_house * cost_per_nail)

/-- Theorem stating that the cost to build 4 birdhouses is $88 -/
theorem cost_of_four_birdhouses :
  cost_of_birdhouses 4 = 88 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_four_birdhouses_l185_18520


namespace NUMINAMATH_CALUDE_equation_infinite_solutions_l185_18581

theorem equation_infinite_solutions (c : ℝ) : 
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_infinite_solutions_l185_18581


namespace NUMINAMATH_CALUDE_bouncy_balls_per_package_l185_18592

theorem bouncy_balls_per_package (red_packs green_packs yellow_packs total_balls : ℕ) 
  (h1 : red_packs = 4)
  (h2 : yellow_packs = 8)
  (h3 : green_packs = 4)
  (h4 : total_balls = 160) :
  ∃ (balls_per_pack : ℕ), 
    balls_per_pack * (red_packs + yellow_packs + green_packs) = total_balls ∧ 
    balls_per_pack = 10 := by
  sorry

end NUMINAMATH_CALUDE_bouncy_balls_per_package_l185_18592


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l185_18502

/-- Represents the contractor's payment scenario --/
structure ContractorPayment where
  totalDays : ℕ
  absentDays : ℕ
  finePerDay : ℚ
  totalPayment : ℚ

/-- Calculates the daily wage given the contractor's payment scenario --/
def calculateDailyWage (c : ContractorPayment) : ℚ :=
  (c.totalPayment + c.finePerDay * c.absentDays) / (c.totalDays - c.absentDays)

/-- Theorem stating that the daily wage is 25 rupees given the problem conditions --/
theorem contractor_daily_wage :
  let c : ContractorPayment := {
    totalDays := 30,
    absentDays := 10,
    finePerDay := 15/2,
    totalPayment := 425
  }
  calculateDailyWage c = 25 := by sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l185_18502


namespace NUMINAMATH_CALUDE_jar_weight_theorem_l185_18508

theorem jar_weight_theorem (jar_weight : ℝ) (full_weight : ℝ) 
  (h1 : jar_weight = 0.1 * full_weight)
  (h2 : 0 < full_weight) :
  let remaining_fraction : ℝ := 0.5555555555555556
  let remaining_weight := jar_weight + remaining_fraction * (full_weight - jar_weight)
  remaining_weight / full_weight = 0.6 := by sorry

end NUMINAMATH_CALUDE_jar_weight_theorem_l185_18508


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l185_18570

/-- The cost price of one meter of cloth given the selling price and profit per meter -/
theorem cost_price_per_meter 
  (total_meters : ℕ) 
  (selling_price : ℚ) 
  (profit_per_meter : ℚ) : 
  total_meters = 80 → 
  selling_price = 6900 → 
  profit_per_meter = 20 → 
  (selling_price - total_meters * profit_per_meter) / total_meters = 66.25 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l185_18570


namespace NUMINAMATH_CALUDE_sequence_A_l185_18531

theorem sequence_A (a : ℕ → ℕ) : 
  (a 1 = 2) → 
  (∀ n : ℕ, a (n + 1) = a n + n + 1) → 
  (a 20 = 211) :=
by sorry


end NUMINAMATH_CALUDE_sequence_A_l185_18531


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l185_18552

theorem simplify_and_rationalize (x : ℝ) :
  x = 1 / (2 - 1 / (Real.sqrt 5 + 2)) →
  x = (4 + Real.sqrt 5) / 11 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l185_18552


namespace NUMINAMATH_CALUDE_kamal_physics_marks_l185_18538

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks for a student -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

theorem kamal_physics_marks :
  ∀ (marks : StudentMarks),
    marks.english = 76 →
    marks.mathematics = 60 →
    marks.chemistry = 67 →
    marks.biology = 85 →
    average marks = 74 →
    marks.physics = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_kamal_physics_marks_l185_18538


namespace NUMINAMATH_CALUDE_fraction_simplification_l185_18588

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  a / (a - b) - b / (b - a) = (a + b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l185_18588


namespace NUMINAMATH_CALUDE_chef_potatoes_per_week_l185_18530

/-- Calculates the total number of potatoes used by a chef in one week -/
def total_potatoes_per_week (lunch_potatoes : ℕ) (work_days : ℕ) : ℕ :=
  let dinner_potatoes := 2 * lunch_potatoes
  let lunch_total := lunch_potatoes * work_days
  let dinner_total := dinner_potatoes * work_days
  lunch_total + dinner_total

/-- Proves that the chef uses 90 potatoes in one week -/
theorem chef_potatoes_per_week :
  total_potatoes_per_week 5 6 = 90 :=
by
  sorry

#eval total_potatoes_per_week 5 6

end NUMINAMATH_CALUDE_chef_potatoes_per_week_l185_18530


namespace NUMINAMATH_CALUDE_smallest_unrepresentable_odd_number_l185_18589

theorem smallest_unrepresentable_odd_number :
  ∀ n : ℕ, n > 0 → n % 2 = 1 →
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ n = 7^x - 3 * 2^y) ∨ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_unrepresentable_odd_number_l185_18589


namespace NUMINAMATH_CALUDE_smallest_c_is_22_l185_18504

/-- A polynomial with three positive integer roots -/
structure PolynomialWithThreeRoots where
  c : ℤ
  d : ℤ
  root1 : ℤ
  root2 : ℤ
  root3 : ℤ
  root1_pos : root1 > 0
  root2_pos : root2 > 0
  root3_pos : root3 > 0
  is_root1 : root1^3 - c*root1^2 + d*root1 - 2310 = 0
  is_root2 : root2^3 - c*root2^2 + d*root2 - 2310 = 0
  is_root3 : root3^3 - c*root3^2 + d*root3 - 2310 = 0

/-- The smallest possible value of c for a polynomial with three positive integer roots -/
def smallest_c : ℤ := 22

/-- Theorem stating that 22 is the smallest possible value of c -/
theorem smallest_c_is_22 (p : PolynomialWithThreeRoots) : p.c ≥ smallest_c := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_is_22_l185_18504


namespace NUMINAMATH_CALUDE_line_through_two_points_l185_18557

theorem line_through_two_points (m n p : ℝ) :
  (m = 3 * n + 5) ∧ (m + 2 = 3 * (n + p) + 5) → p = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l185_18557


namespace NUMINAMATH_CALUDE_transformations_correct_l185_18568

theorem transformations_correct (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) (h3 : a / c = b / c) (h4 : -2 * a = -2 * b) : 
  (a + 6 = b + 6) ∧ 
  (a / 9 = b / 9) ∧ 
  (a = b) ∧ 
  (a = b) := by
  sorry

end NUMINAMATH_CALUDE_transformations_correct_l185_18568


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l185_18536

/-- The solution to the equation x = 3 + 9 / (2 + 9 / x) -/
theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 9 / (2 + 9 / x) ∧ x = (3 + 3 * Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l185_18536


namespace NUMINAMATH_CALUDE_work_problem_solution_l185_18551

def work_problem (a_days b_days b_worked_days : ℕ) : ℕ :=
  let b_work_rate := 1 / b_days
  let b_completed := b_work_rate * b_worked_days
  let remaining_work := 1 - b_completed
  let a_work_rate := 1 / a_days
  Nat.ceil (remaining_work / a_work_rate)

theorem work_problem_solution :
  work_problem 18 15 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_work_problem_solution_l185_18551


namespace NUMINAMATH_CALUDE_expansion_equality_l185_18586

theorem expansion_equality (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
sorry

end NUMINAMATH_CALUDE_expansion_equality_l185_18586


namespace NUMINAMATH_CALUDE_cubic_decreasing_iff_a_leq_neg_three_l185_18556

/-- A cubic function f(x) = a x^3 + 3 x^2 - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

/-- A function is decreasing on ℝ if for all x, y in ℝ, x < y implies f(x) > f(y) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem cubic_decreasing_iff_a_leq_neg_three (a : ℝ) :
  IsDecreasing (f a) ↔ a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_iff_a_leq_neg_three_l185_18556


namespace NUMINAMATH_CALUDE_intersection_range_l185_18590

theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 6 ∧
    x₂^2 - y₂^2 = 6) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l185_18590


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_complex_fraction_l185_18516

theorem sum_real_imag_parts_complex_fraction : ∃ (z : ℂ), 
  z = (3 - 3 * Complex.I) / (1 - Complex.I) ∧ 
  z.re + z.im = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_complex_fraction_l185_18516


namespace NUMINAMATH_CALUDE_lcm_of_5_6_8_9_l185_18528

theorem lcm_of_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_8_9_l185_18528


namespace NUMINAMATH_CALUDE_allocation_schemes_13_4_l185_18559

/-- The number of ways to allocate outstanding member quotas to classes. -/
def allocationSchemes (totalMembers : ℕ) (numClasses : ℕ) : ℕ :=
  Nat.choose (totalMembers - numClasses + numClasses - 1) (numClasses - 1)

/-- Theorem stating the number of allocation schemes for 13 members to 4 classes. -/
theorem allocation_schemes_13_4 :
  allocationSchemes 13 4 = 220 := by
  sorry

#eval allocationSchemes 13 4

end NUMINAMATH_CALUDE_allocation_schemes_13_4_l185_18559


namespace NUMINAMATH_CALUDE_undergrads_playing_sports_l185_18573

theorem undergrads_playing_sports (total_students : ℕ) 
  (grad_percent : ℚ) (grad_not_playing : ℚ) (undergrad_not_playing : ℚ) 
  (total_not_playing : ℚ) :
  total_students = 800 →
  grad_percent = 1/4 →
  grad_not_playing = 1/2 →
  undergrad_not_playing = 1/5 →
  total_not_playing = 3/10 →
  (total_students : ℚ) * (1 - grad_percent) * (1 - undergrad_not_playing) = 480 :=
by sorry

end NUMINAMATH_CALUDE_undergrads_playing_sports_l185_18573


namespace NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l185_18509

/-- Represents a repeating decimal where the decimal part repeats infinitely -/
def RepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / (999 : ℚ)

/-- The fraction 0.888... -/
def a : ℚ := RepeatingDecimal 0 888

/-- The fraction 1.222... -/
def b : ℚ := RepeatingDecimal 1 222

/-- Theorem stating that the ratio of 0.888... to 1.222... is equal to 8/11 -/
theorem ratio_of_repeating_decimals : a / b = 8 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l185_18509


namespace NUMINAMATH_CALUDE_average_of_multiples_of_four_l185_18523

theorem average_of_multiples_of_four : 
  let numbers := (Finset.range 33).filter (fun n => (n + 8) % 4 = 0)
  let sum := numbers.sum (fun n => n + 8)
  let count := numbers.card
  sum / count = 22 := by sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_four_l185_18523


namespace NUMINAMATH_CALUDE_complex_division_equality_l185_18564

theorem complex_division_equality : (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l185_18564


namespace NUMINAMATH_CALUDE_ratio_difference_l185_18507

/-- Given two positive numbers in a 7:11 ratio where the smaller is 28, 
    prove that the larger exceeds the smaller by 16. -/
theorem ratio_difference (small large : ℝ) : 
  small > 0 ∧ large > 0 ∧ 
  large / small = 11 / 7 ∧ 
  small = 28 → 
  large - small = 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l185_18507


namespace NUMINAMATH_CALUDE_vacation_cost_division_l185_18561

theorem vacation_cost_division (total_cost : ℕ) (cost_difference : ℕ) : 
  (total_cost = 360) →
  (total_cost / 4 + cost_difference = total_cost / 3) →
  (cost_difference = 30) →
  3 = total_cost / (total_cost / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l185_18561


namespace NUMINAMATH_CALUDE_x_squared_ge_one_necessary_not_sufficient_l185_18533

theorem x_squared_ge_one_necessary_not_sufficient :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_ge_one_necessary_not_sufficient_l185_18533


namespace NUMINAMATH_CALUDE_max_intersections_circle_ellipse_triangle_l185_18566

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An ellipse in a 2D plane -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The maximum number of intersection points between a circle and a triangle -/
def max_intersections_circle_triangle : ℕ := 6

/-- The maximum number of intersection points between an ellipse and a triangle -/
def max_intersections_ellipse_triangle : ℕ := 6

/-- The maximum number of intersection points between a circle and an ellipse -/
def max_intersections_circle_ellipse : ℕ := 4

/-- Theorem: The maximum number of intersection points among a circle, an ellipse, and a triangle is 16 -/
theorem max_intersections_circle_ellipse_triangle :
  ∀ (c : Circle) (e : Ellipse) (t : Triangle),
  max_intersections_circle_triangle +
  max_intersections_ellipse_triangle +
  max_intersections_circle_ellipse = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_ellipse_triangle_l185_18566


namespace NUMINAMATH_CALUDE_quadratic_always_positive_triangle_angle_sum_product_zero_implies_factor_zero_factors_nonzero_implies_x_not_roots_l185_18598

-- Statement 1
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by sorry

-- Statement 2
theorem triangle_angle_sum : ∀ a b c : ℝ, 
  0 < a ∧ 0 < b ∧ 0 < c → a + b + c = 180 := by sorry

-- Statement 3
theorem product_zero_implies_factor_zero : ∀ a b c : ℝ, 
  a * b * c = 0 → a = 0 ∨ b = 0 ∨ c = 0 := by sorry

-- Statement 4
theorem factors_nonzero_implies_x_not_roots : ∀ x : ℝ, 
  (x - 1) * (x - 2) ≠ 0 → x ≠ 1 ∧ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_triangle_angle_sum_product_zero_implies_factor_zero_factors_nonzero_implies_x_not_roots_l185_18598


namespace NUMINAMATH_CALUDE_cookie_box_duration_l185_18529

/-- Given a box of cookies and daily consumption, calculate how many days the box will last -/
def cookiesDuration (totalCookies : ℕ) (oldestSonCookies : ℕ) (youngestSonCookies : ℕ) : ℕ :=
  totalCookies / (oldestSonCookies + youngestSonCookies)

/-- Prove that a box of 54 cookies lasts 9 days when 4 cookies are given to the oldest son
    and 2 cookies are given to the youngest son daily -/
theorem cookie_box_duration :
  cookiesDuration 54 4 2 = 9 := by
  sorry

#eval cookiesDuration 54 4 2

end NUMINAMATH_CALUDE_cookie_box_duration_l185_18529


namespace NUMINAMATH_CALUDE_x_plus_y_equals_three_l185_18593

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (a b c p : V)
variable (x y : ℝ)

-- {a, b, c} is a basis of the space
variable (h1 : LinearIndependent ℝ ![a, b, c])
variable (h2 : Submodule.span ℝ {a, b, c} = ⊤)

-- p = 3a + b + c
variable (h3 : p = 3 • a + b + c)

-- {a+b, a-b, c} is another basis of the space
variable (h4 : LinearIndependent ℝ ![a + b, a - b, c])
variable (h5 : Submodule.span ℝ {a + b, a - b, c} = ⊤)

-- p = x(a+b) + y(a-b) + c
variable (h6 : p = x • (a + b) + y • (a - b) + c)

theorem x_plus_y_equals_three : x + y = 3 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_three_l185_18593


namespace NUMINAMATH_CALUDE_book_pages_theorem_l185_18554

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The total number of pages in a book -/
def total_pages (b : Book) : ℕ := b.chapter1_pages + b.chapter2_pages

/-- Theorem: A book with 48 pages in the first chapter and 46 pages in the second chapter has 94 pages in total -/
theorem book_pages_theorem :
  ∀ (b : Book), b.chapter1_pages = 48 → b.chapter2_pages = 46 → total_pages b = 94 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l185_18554


namespace NUMINAMATH_CALUDE_polynomial_factorization_l185_18563

theorem polynomial_factorization (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x - 6 = (x - 2)*(x + 3)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l185_18563


namespace NUMINAMATH_CALUDE_partner_a_capital_l185_18521

/-- Represents the partnership structure and profit distribution --/
structure Partnership where
  total_profit : ℝ
  a_share : ℝ
  b_share : ℝ
  c_share : ℝ
  a_share_def : a_share = (2/3) * total_profit
  bc_share_def : b_share = c_share
  bc_share_sum : b_share + c_share = (1/3) * total_profit

/-- Represents the change in profit rate and its effect on partner a's income --/
structure ProfitChange where
  initial_rate : ℝ
  final_rate : ℝ
  a_income_increase : ℝ
  rate_def : final_rate - initial_rate = 0.02
  initial_rate_def : initial_rate = 0.05
  income_increase_def : a_income_increase = 200

/-- The main theorem stating the capital of partner a --/
theorem partner_a_capital 
  (p : Partnership) 
  (pc : ProfitChange) : 
  ∃ (capital_a : ℝ), capital_a = 300000 := by
  sorry

end NUMINAMATH_CALUDE_partner_a_capital_l185_18521


namespace NUMINAMATH_CALUDE_inequality_proof_l185_18579

theorem inequality_proof (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3*a*(a - x)^5 + 5/2 * a^2 * (a - x)^4 - 1/2 * a^4 * (a - x)^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l185_18579


namespace NUMINAMATH_CALUDE_x_cube_between_x_and_x_square_l185_18574

theorem x_cube_between_x_and_x_square :
  let x : ℚ := -2/5
  x < x^3 ∧ x^3 < x^2 := by sorry

end NUMINAMATH_CALUDE_x_cube_between_x_and_x_square_l185_18574


namespace NUMINAMATH_CALUDE_no_cover_with_changed_tiles_l185_18545

/-- Represents a rectangular floor -/
structure Floor :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a tile configuration -/
structure TileConfig :=
  (twoBytwo : ℕ)  -- number of 2x2 tiles
  (fourByOne : ℕ) -- number of 4x1 tiles

/-- Predicate to check if a floor can be covered by a given tile configuration -/
def canCover (f : Floor) (tc : TileConfig) : Prop :=
  4 * tc.twoBytwo + 4 * tc.fourByOne = f.length * f.width

/-- Main theorem: If a floor can be covered by a tile configuration,
    it cannot be covered by changing the number of tiles by ±1 for each type -/
theorem no_cover_with_changed_tiles (f : Floor) (tc : TileConfig) :
  canCover f tc →
  ¬(canCover f { twoBytwo := tc.twoBytwo + 1, fourByOne := tc.fourByOne - 1 } ∨
    canCover f { twoBytwo := tc.twoBytwo - 1, fourByOne := tc.fourByOne + 1 }) :=
by
  sorry

#check no_cover_with_changed_tiles

end NUMINAMATH_CALUDE_no_cover_with_changed_tiles_l185_18545


namespace NUMINAMATH_CALUDE_quadruple_work_time_l185_18541

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 45
def work_rate_B : ℚ := 1 / 30

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Define the time to complete 4 times the work
def time_for_quadruple_work : ℚ := 4 / combined_work_rate

-- Theorem statement
theorem quadruple_work_time : time_for_quadruple_work = 9/2 := by sorry

end NUMINAMATH_CALUDE_quadruple_work_time_l185_18541


namespace NUMINAMATH_CALUDE_decimal_50_to_ternary_l185_18565

/-- Converts a natural number to its ternary (base-3) representation -/
def to_ternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Checks if a list of digits is a valid ternary number -/
def is_valid_ternary (l : List ℕ) : Prop :=
  l.all (λ d => d < 3)

theorem decimal_50_to_ternary :
  let ternary := to_ternary 50
  is_valid_ternary ternary ∧ ternary = [1, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_decimal_50_to_ternary_l185_18565


namespace NUMINAMATH_CALUDE_quadratic_properties_l185_18543

def f (x : ℝ) := -2 * x^2 + 4 * x + 1

theorem quadratic_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = f (2 - x)) ∧
  (f 1 = 3 ∧ ∀ (x : ℝ), f x ≤ f 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l185_18543


namespace NUMINAMATH_CALUDE_f_quadrants_l185_18524

/-- A linear function in the Cartesian coordinate system -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Quadrants in the Cartesian coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The set of quadrants a linear function passes through -/
def quadrants_passed (f : LinearFunction) : Set Quadrant :=
  sorry

/-- The specific linear function y = -x - 2 -/
def f : LinearFunction :=
  { slope := -1, intercept := -2 }

/-- Theorem stating which quadrants the function y = -x - 2 passes through -/
theorem f_quadrants :
  quadrants_passed f = {Quadrant.II, Quadrant.III, Quadrant.IV} :=
sorry

end NUMINAMATH_CALUDE_f_quadrants_l185_18524


namespace NUMINAMATH_CALUDE_fraction_inequality_l185_18535

theorem fraction_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l185_18535


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l185_18575

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 20 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l185_18575


namespace NUMINAMATH_CALUDE_exam_failure_count_l185_18518

theorem exam_failure_count (total_students : ℕ) (pass_percentage : ℚ) 
  (h1 : total_students = 840)
  (h2 : pass_percentage = 35 / 100) :
  (total_students : ℚ) * (1 - pass_percentage) = 546 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_count_l185_18518


namespace NUMINAMATH_CALUDE_g_g_eq_5_has_two_solutions_l185_18527

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 6

theorem g_g_eq_5_has_two_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g (g x₁) = 5 ∧ g (g x₂) = 5 ∧
  ∀ (x : ℝ), g (g x) = 5 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_g_g_eq_5_has_two_solutions_l185_18527


namespace NUMINAMATH_CALUDE_sin_330_degrees_l185_18567

theorem sin_330_degrees :
  Real.sin (330 * π / 180) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l185_18567


namespace NUMINAMATH_CALUDE_complex_sum_reciprocal_imag_part_l185_18544

theorem complex_sum_reciprocal_imag_part :
  let z : ℂ := 2 + I
  (z + z⁻¹).im = 4/5 := by sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocal_imag_part_l185_18544


namespace NUMINAMATH_CALUDE_lisa_tommy_earnings_difference_l185_18542

-- Define the earnings for each person
def sophia_earnings : ℕ := 10 + 15 + 25
def sarah_earnings : ℕ := 15 + 10 + 20 + 20
def lisa_earnings : ℕ := 20 + 30
def jack_earnings : ℕ := 10 + 10 + 10 + 15 + 15
def tommy_earnings : ℕ := 5 + 5 + 10 + 10

-- Define the total earnings
def total_earnings : ℕ := 180

-- Theorem statement
theorem lisa_tommy_earnings_difference :
  lisa_earnings - tommy_earnings = 20 :=
sorry

end NUMINAMATH_CALUDE_lisa_tommy_earnings_difference_l185_18542


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l185_18548

/-- Given two lines in a 2D plane:
    1. y = 3x + 4
    2. y = x
    This theorem states that the line symmetric to y = 3x + 4
    with respect to y = x has the equation y = (1/3)x - (4/3) -/
theorem symmetric_line_equation :
  let line1 : ℝ → ℝ := λ x => 3 * x + 4
  let line2 : ℝ → ℝ := λ x => x
  let symmetric_line : ℝ → ℝ := λ x => (1/3) * x - (4/3)
  ∀ x y : ℝ,
    (y = line1 x ∧ 
     ∃ x' y', x' = y ∧ y' = x ∧ y' = line2 x') →
    y = symmetric_line x :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l185_18548


namespace NUMINAMATH_CALUDE_largest_four_digit_odd_sum_19_l185_18511

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d ≤ 9

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def all_odd_digits (n : ℕ) : Prop :=
  is_odd_digit (n / 1000) ∧
  is_odd_digit ((n / 100) % 10) ∧
  is_odd_digit ((n / 10) % 10) ∧
  is_odd_digit (n % 10)

theorem largest_four_digit_odd_sum_19 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ all_odd_digits n ∧ digit_sum n = 19 →
  n ≤ 9711 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_odd_sum_19_l185_18511


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_identity_l185_18510

/-- Given two vectors a and b, where a is parallel to b, prove that cos(π/2 + α) = -1/3 -/
theorem parallel_vectors_cos_identity (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α))
  (hb : b = (Real.cos α, 1))
  (hparallel : ∃ (k : ℝ), a = k • b) :
  Real.cos (π/2 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_identity_l185_18510


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l185_18582

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∀ c d : ℝ, (∃ x : ℝ, x^4 + c*x^3 + d*x^2 + c*x + 1 = 0) → a^2 + b^2 ≤ c^2 + d^2) →
  a^2 + b^2 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l185_18582


namespace NUMINAMATH_CALUDE_largest_quantity_l185_18515

theorem largest_quantity (a b c d : ℝ) 
  (h : a + 1 = b - 2 ∧ a + 1 = c + 3 ∧ a + 1 = d - 4) : 
  d = max a (max b c) ∧ d ≥ a ∧ d ≥ b ∧ d ≥ c := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l185_18515


namespace NUMINAMATH_CALUDE_christine_travel_time_l185_18546

/-- Given Christine's travel scenario, prove the time she wandered. -/
theorem christine_travel_time (speed : ℝ) (distance : ℝ) (h1 : speed = 20) (h2 : distance = 80) :
  distance / speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_christine_travel_time_l185_18546


namespace NUMINAMATH_CALUDE_specific_building_height_l185_18595

/-- The height of a building with varying story heights -/
def building_height (total_stories : ℕ) (base_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := total_stories / 2
  let second_half := total_stories - first_half
  (first_half * base_height) + (second_half * (base_height + height_increase))

/-- Theorem stating the height of the specific building described in the problem -/
theorem specific_building_height :
  building_height 20 12 3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_specific_building_height_l185_18595


namespace NUMINAMATH_CALUDE_extra_parts_calculation_l185_18519

/-- The number of extra parts produced compared to the original plan -/
def extra_parts (initial_rate : ℕ) (initial_days : ℕ) (rate_increase : ℕ) (total_parts : ℕ) : ℕ :=
  let total_days := (total_parts - initial_rate * initial_days) / (initial_rate + rate_increase) + initial_days
  let actual_production := initial_rate * initial_days + (initial_rate + rate_increase) * (total_days - initial_days)
  let planned_production := initial_rate * total_days
  actual_production - planned_production

theorem extra_parts_calculation :
  extra_parts 25 3 5 675 = 100 := by
  sorry

end NUMINAMATH_CALUDE_extra_parts_calculation_l185_18519


namespace NUMINAMATH_CALUDE_cubic_value_l185_18549

theorem cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2010 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_l185_18549


namespace NUMINAMATH_CALUDE_rachel_envelope_stuffing_l185_18532

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing
  (total_hours : ℕ)
  (total_envelopes : ℕ)
  (second_hour_envelopes : ℕ)
  (required_rate : ℕ)
  (h1 : total_hours = 8)
  (h2 : total_envelopes = 1500)
  (h3 : second_hour_envelopes = 141)
  (h4 : required_rate = 204) :
  total_envelopes - (required_rate * (total_hours - 2)) - second_hour_envelopes = 135 := by
  sorry

#check rachel_envelope_stuffing

end NUMINAMATH_CALUDE_rachel_envelope_stuffing_l185_18532


namespace NUMINAMATH_CALUDE_encounters_for_2015_trips_relative_speeds_main_encounters_theorem_l185_18584

/-- The number of encounters between two people traveling between two points -/
def encounters (a_trips b_trips : ℕ) : ℕ :=
  let full_cycles := a_trips / 2
  let remainder := a_trips % 2
  3 * full_cycles + if remainder = 0 then 0 else 2

/-- The theorem stating the number of encounters when A reaches point B 2015 times -/
theorem encounters_for_2015_trips : encounters 2015 2015 = 3023 := by
  sorry

/-- The theorem stating the relative speeds of A and B -/
theorem relative_speeds : ∀ (va vb : ℝ), 
  5 * va = 9 * vb → vb = (18/5) * va := by
  sorry

/-- The main theorem proving the number of encounters -/
theorem main_encounters_theorem : 
  ∃ (va vb : ℝ), va > 0 ∧ vb > 0 ∧ 5 * va = 9 * vb ∧ encounters 2015 2015 = 3023 := by
  sorry

end NUMINAMATH_CALUDE_encounters_for_2015_trips_relative_speeds_main_encounters_theorem_l185_18584


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l185_18585

theorem polynomial_factor_coefficients :
  ∀ (a b : ℤ),
  (∃ (c d : ℤ),
    (∀ x : ℝ, a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8 = (3 * x^2 - 2 * x + 2) * (c * x^2 + d * x + 4))) →
  a = -51 ∧ b = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l185_18585


namespace NUMINAMATH_CALUDE_speech_competition_proof_l185_18597

def scores : List ℝ := [91, 89, 88, 92, 90]

theorem speech_competition_proof :
  let n : ℕ := 5
  let avg : ℝ := 90
  let variance : ℝ := (1 : ℝ) / n * (scores.map (λ x => (x - avg)^2)).sum
  (scores.sum / n = avg) ∧ (variance = 2) :=
by sorry

end NUMINAMATH_CALUDE_speech_competition_proof_l185_18597


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l185_18522

theorem quadratic_roots_ratio (p q α β : ℝ) (h1 : α + β = p) (h2 : α * β = 6) 
  (h3 : x^2 - p*x + q = 0 → x = α ∨ x = β) (h4 : p^2 ≠ 12) : 
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l185_18522


namespace NUMINAMATH_CALUDE_cafeteria_pies_l185_18594

theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  (initial_apples - handed_out) / apples_per_pie = 7 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l185_18594


namespace NUMINAMATH_CALUDE_no_rational_solution_l185_18562

theorem no_rational_solution : ¬∃ (x y z : ℚ), (x + y + z = 0) ∧ (x^2 + y^2 + z^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l185_18562


namespace NUMINAMATH_CALUDE_average_weight_after_student_leaves_l185_18550

/-- Represents the class of students with their weights before and after a student leaves -/
structure ClassWeights where
  totalStudents : Nat
  maleWeightSum : ℝ
  femaleWeightSum : ℝ
  leavingStudentWeight : ℝ
  weightIncreaseAfterLeaving : ℝ

/-- Theorem stating the average weight of remaining students after one leaves -/
theorem average_weight_after_student_leaves (c : ClassWeights)
  (h1 : c.totalStudents = 60)
  (h2 : c.leavingStudentWeight = 45)
  (h3 : c.weightIncreaseAfterLeaving = 0.2) :
  (c.maleWeightSum + c.femaleWeightSum - c.leavingStudentWeight) / (c.totalStudents - 1) = 57 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_after_student_leaves_l185_18550


namespace NUMINAMATH_CALUDE_skips_per_meter_l185_18526

theorem skips_per_meter
  (p q r s t u : ℝ)
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (skip_jump : p * q⁻¹ = 1)
  (jump_foot : r * s⁻¹ = 1)
  (foot_meter : t * u⁻¹ = 1) :
  (t * r * p) * (u * s * q)⁻¹ = 1 := by
sorry

end NUMINAMATH_CALUDE_skips_per_meter_l185_18526


namespace NUMINAMATH_CALUDE_random_events_count_l185_18578

theorem random_events_count (total_events : ℕ) 
  (prob_certain : ℚ) (prob_impossible : ℚ) :
  total_events = 10 →
  prob_certain = 2 / 10 →
  prob_impossible = 3 / 10 →
  (total_events : ℚ) * prob_certain + 
  (total_events : ℚ) * prob_impossible + 
  (total_events - 
    (total_events * prob_certain).floor - 
    (total_events * prob_impossible).floor : ℚ) = total_events →
  total_events - 
    (total_events * prob_certain).floor - 
    (total_events * prob_impossible).floor = 5 := by
  sorry

#check random_events_count

end NUMINAMATH_CALUDE_random_events_count_l185_18578


namespace NUMINAMATH_CALUDE_abc_inequality_l185_18587

theorem abc_inequality (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (sum_eq : a + b + c = 6) 
  (prod_sum_eq : a * b + b * c + c * a = 9) : 
  0 < a * b * c ∧ a * b * c < 4 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l185_18587


namespace NUMINAMATH_CALUDE_round_trip_time_l185_18500

/-- Proves that given a round trip with specified conditions, the outbound journey takes 180 minutes -/
theorem round_trip_time (speed_out speed_return : ℝ) (total_time : ℝ) : 
  speed_out = 100 →
  speed_return = 150 →
  total_time = 5 →
  (total_time * speed_out * speed_return) / (speed_out + speed_return) / speed_out * 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_time_l185_18500
