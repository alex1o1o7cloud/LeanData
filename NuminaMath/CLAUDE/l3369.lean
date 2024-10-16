import Mathlib

namespace NUMINAMATH_CALUDE_union_of_sets_l3369_336915

theorem union_of_sets : 
  let S : Set ℕ := {3, 4, 5}
  let T : Set ℕ := {4, 7, 8}
  S ∪ T = {3, 4, 5, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3369_336915


namespace NUMINAMATH_CALUDE_equation_with_72_l3369_336951

/-- The first term of the nth equation in the sequence -/
def first_term (n : ℕ) : ℕ := 2 * n^2

/-- The equation number in which 72 appears as the first term -/
theorem equation_with_72 : {k : ℕ | first_term k = 72} = {6} := by sorry

end NUMINAMATH_CALUDE_equation_with_72_l3369_336951


namespace NUMINAMATH_CALUDE_difference_of_squares_123_23_l3369_336935

theorem difference_of_squares_123_23 : 123^2 - 23^2 = 14600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_123_23_l3369_336935


namespace NUMINAMATH_CALUDE_bed_frame_cost_l3369_336907

theorem bed_frame_cost (bed_price : ℝ) (total_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) :
  bed_price = 10 * total_price →
  discount_rate = 0.2 →
  final_price = (1 - discount_rate) * (bed_price + total_price) →
  final_price = 660 →
  total_price = 75 := by
sorry

end NUMINAMATH_CALUDE_bed_frame_cost_l3369_336907


namespace NUMINAMATH_CALUDE_parkway_elementary_students_l3369_336982

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := 500

/-- The number of boys in the fifth grade -/
def boys : ℕ := 350

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 86 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 115

/-- Theorem stating that the total number of students is 500 -/
theorem parkway_elementary_students :
  total_students = boys + girls_not_soccer + (soccer_players - (boys_soccer_percentage * soccer_players).num) :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_students_l3369_336982


namespace NUMINAMATH_CALUDE_test_has_ten_four_point_questions_l3369_336974

/-- Represents a test with two-point and four-point questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

/-- Checks if a test configuration is valid -/
def is_valid_test (t : Test) : Prop :=
  t.total_questions = t.two_point_questions + t.four_point_questions ∧
  t.total_points = 2 * t.two_point_questions + 4 * t.four_point_questions

/-- Theorem: A test with 100 points and 40 questions has 10 four-point questions -/
theorem test_has_ten_four_point_questions (t : Test) 
  (h1 : t.total_points = 100) 
  (h2 : t.total_questions = 40) 
  (h3 : is_valid_test t) : 
  t.four_point_questions = 10 := by
  sorry

end NUMINAMATH_CALUDE_test_has_ten_four_point_questions_l3369_336974


namespace NUMINAMATH_CALUDE_special_integers_l3369_336997

def is_special (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, 1 < d1 ∧ d1 < n ∧ d1 ∣ n ∧
                1 < d2 ∧ d2 < n ∧ d2 ∣ n ∧
                d1 ≠ d2) ∧
  (∀ d1 d2 : ℕ, 1 < d1 ∧ d1 < n ∧ d1 ∣ n →
                1 < d2 ∧ d2 < n ∧ d2 ∣ n →
                (d1 - d2) ∣ n ∨ (d2 - d1) ∣ n)

theorem special_integers :
  ∀ n : ℕ, is_special n ↔ n = 6 ∨ n = 8 ∨ n = 12 :=
by sorry

end NUMINAMATH_CALUDE_special_integers_l3369_336997


namespace NUMINAMATH_CALUDE_discount_gain_percent_l3369_336934

theorem discount_gain_percent (marked_price : ℝ) (cost_price : ℝ) (discount_rate : ℝ) :
  cost_price = 0.64 * marked_price →
  discount_rate = 0.12 →
  let selling_price := marked_price * (1 - discount_rate)
  let gain_percent := ((selling_price - cost_price) / cost_price) * 100
  gain_percent = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_gain_percent_l3369_336934


namespace NUMINAMATH_CALUDE_no_all_ones_reverse_product_l3369_336969

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number consists only of the digit 1 -/
def allOnes (n : ℕ) : Prop := sorry

/-- 
There does not exist a natural number n > 1 such that n multiplied by 
the number formed by reversing its digits results in a number comprised 
entirely of the digit one.
-/
theorem no_all_ones_reverse_product : 
  ¬ ∃ (n : ℕ), n > 1 ∧ allOnes (n * reverseDigits n) := by
  sorry

end NUMINAMATH_CALUDE_no_all_ones_reverse_product_l3369_336969


namespace NUMINAMATH_CALUDE_survey_respondents_l3369_336971

/-- Represents the number of people preferring each brand in a survey. -/
structure BrandPreference where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents in a survey. -/
def totalRespondents (bp : BrandPreference) : ℕ :=
  bp.x + bp.y + bp.z

/-- Theorem stating that given the conditions of the survey, 
    the total number of respondents is 350. -/
theorem survey_respondents : 
  ∀ (bp : BrandPreference), 
    bp.x = 200 → 
    4 * bp.z = bp.x → 
    2 * bp.z = bp.y → 
    totalRespondents bp = 350 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l3369_336971


namespace NUMINAMATH_CALUDE_scientific_notation_138000_l3369_336979

theorem scientific_notation_138000 :
  138000 = 1.38 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_138000_l3369_336979


namespace NUMINAMATH_CALUDE_rectangle_area_l3369_336989

/-- Proves that a rectangle with length thrice its breadth and perimeter 64 has area 192 -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 64 → l * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3369_336989


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3369_336926

theorem complex_equation_solution (b : ℝ) : 
  (2 - Complex.I) * (4 * Complex.I) = 4 + b * Complex.I → b = 8 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3369_336926


namespace NUMINAMATH_CALUDE_elegant_interval_p_values_l3369_336995

theorem elegant_interval_p_values (a b : ℕ) (m : ℝ) (p : ℕ) :
  (a < m ∧ m < b) →  -- m is in the "elegant interval" (a, b)
  (b = a + 1) →  -- a and b are consecutive positive integers
  (3 < Real.sqrt a + b ∧ Real.sqrt a + b ≤ 13) →  -- satisfies the given inequality
  (∃ x y : ℕ, x = b ∧ y * y = a ∧ b * x + a * y = p) →  -- x = b, y = √a, and bx + ay = p
  (p = 33 ∨ p = 127) :=
by sorry

end NUMINAMATH_CALUDE_elegant_interval_p_values_l3369_336995


namespace NUMINAMATH_CALUDE_jellybean_problem_l3369_336946

theorem jellybean_problem (J : ℕ) : 
  J - 15 + 5 - 4 = 23 → J = 33 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3369_336946


namespace NUMINAMATH_CALUDE_bowl_water_problem_l3369_336998

theorem bowl_water_problem (C : ℝ) (h1 : C > 0) : 
  C / 2 + 4 = 0.7 * C → 0.7 * C = 14 := by
  sorry

end NUMINAMATH_CALUDE_bowl_water_problem_l3369_336998


namespace NUMINAMATH_CALUDE_det_A_eq_46_l3369_336940

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -1; 7, 4, -3; 2, 2, 5]

theorem det_A_eq_46 : A.det = 46 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_46_l3369_336940


namespace NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l3369_336957

/-- Calculates the number of toothpicks in a square grid with a missing corner --/
def toothpicks_in_grid (height : ℕ) (width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height - 1

/-- Theorem: A 15x15 square grid with a missing corner uses 479 toothpicks --/
theorem fifteen_by_fifteen_grid_toothpicks :
  toothpicks_in_grid 15 15 = 479 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l3369_336957


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3369_336933

/-- A rhombus with side length 65 and shorter diagonal 56 has a longer diagonal of length 118 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 65 → shorter_diagonal = 56 → longer_diagonal = 118 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3369_336933


namespace NUMINAMATH_CALUDE_meeting_problem_solution_l3369_336900

/-- Represents the problem of two people moving towards each other --/
structure MeetingProblem where
  total_distance : ℝ
  meeting_time : ℝ
  distance_difference : ℝ
  time_to_b_after_meeting : ℝ

/-- The solution to the meeting problem --/
structure MeetingSolution where
  speed_xiaogang : ℝ
  speed_xiaoqiang : ℝ
  time_to_a_after_meeting : ℝ

/-- Theorem stating the solution to the meeting problem --/
theorem meeting_problem_solution (p : MeetingProblem) 
  (h1 : p.meeting_time = 2)
  (h2 : p.distance_difference = 24)
  (h3 : p.time_to_b_after_meeting = 0.5) :
  ∃ (s : MeetingSolution),
    s.speed_xiaogang = 16 ∧
    s.speed_xiaoqiang = 4 ∧
    s.time_to_a_after_meeting = 8 := by
  sorry

end NUMINAMATH_CALUDE_meeting_problem_solution_l3369_336900


namespace NUMINAMATH_CALUDE_volleyball_net_max_removable_edges_l3369_336913

/-- Represents a volleyball net graph -/
structure VolleyballNet where
  rows : Nat
  cols : Nat

/-- Calculates the number of vertices in the volleyball net graph -/
def VolleyballNet.vertexCount (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1) + net.rows * net.cols

/-- Calculates the total number of edges in the volleyball net graph -/
def VolleyballNet.edgeCount (net : VolleyballNet) : Nat :=
  -- This is a placeholder. The actual calculation would be more complex.
  4 * net.rows * net.cols + net.rows * (net.cols - 1) + net.cols * (net.rows - 1)

/-- Theorem: The maximum number of edges that can be removed without disconnecting
    the graph for a 10x20 volleyball net is 800 -/
theorem volleyball_net_max_removable_edges :
  let net : VolleyballNet := { rows := 10, cols := 20 }
  ∃ (removable : Nat), removable = net.edgeCount - (net.vertexCount - 1) ∧ removable = 800 := by
  sorry


end NUMINAMATH_CALUDE_volleyball_net_max_removable_edges_l3369_336913


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l3369_336908

/-- The market value of a machine after two years, given its initial price and annual depreciation rate. -/
def market_value_after_two_years (initial_price : ℝ) (depreciation_rate : ℝ) : ℝ :=
  initial_price * (1 - depreciation_rate)^2

/-- Theorem stating that a machine purchased for $8000 with a 10% annual depreciation rate
    will have a market value of $6480 after two years. -/
theorem machine_value_after_two_years :
  market_value_after_two_years 8000 0.1 = 6480 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l3369_336908


namespace NUMINAMATH_CALUDE_equal_slope_implies_parallel_l3369_336922

/-- Two lines in a plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Theorem: If two non-intersecting lines have equal slopes, then they are parallel -/
theorem equal_slope_implies_parallel (l1 l2 : Line) 
  (h1 : l1.slope = l2.slope) 
  (h2 : l1.yIntercept ≠ l2.yIntercept) : 
  parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_equal_slope_implies_parallel_l3369_336922


namespace NUMINAMATH_CALUDE_inequality_proof_l3369_336948

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3369_336948


namespace NUMINAMATH_CALUDE_heloise_pets_l3369_336912

theorem heloise_pets (total_pets : ℕ) (dogs_given : ℕ) : 
  total_pets = 189 →
  dogs_given = 10 →
  (∃ (dogs cats : ℕ), 
    dogs + cats = total_pets ∧ 
    dogs * 17 = cats * 10) →
  ∃ (remaining_dogs : ℕ), remaining_dogs = 60 :=
by sorry

end NUMINAMATH_CALUDE_heloise_pets_l3369_336912


namespace NUMINAMATH_CALUDE_isosceles_right_pyramid_leg_length_l3369_336941

/-- Represents a pyramid with an isosceles right triangle base -/
structure IsoscelesRightPyramid where
  height : ℝ
  volume : ℝ
  leg : ℝ

/-- The volume of a pyramid is one-third the product of its base area and height -/
axiom pyramid_volume (p : IsoscelesRightPyramid) : p.volume = (1/3) * (1/2 * p.leg^2) * p.height

/-- Theorem: If a pyramid with an isosceles right triangle base has height 4 and volume 6,
    then the length of the leg of the base triangle is 3 -/
theorem isosceles_right_pyramid_leg_length :
  ∀ (p : IsoscelesRightPyramid), p.height = 4 → p.volume = 6 → p.leg = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_pyramid_leg_length_l3369_336941


namespace NUMINAMATH_CALUDE_three_points_determine_plane_line_and_point_determine_plane_trapezoid_determines_plane_circle_points_not_always_determine_plane_l3369_336990

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- A circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D

/-- Three points are collinear if they lie on the same line -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- A point lies on a circle -/
def point_on_circle (p : Point3D) (c : Circle3D) : Prop := sorry

/-- Three points determine a unique plane -/
theorem three_points_determine_plane (p1 p2 p3 : Point3D) (h : ¬collinear p1 p2 p3) : 
  ∃! plane : Plane3D, point_on_plane p1 plane ∧ point_on_plane p2 plane ∧ point_on_plane p3 plane :=
sorry

/-- A line and a point not on the line determine a unique plane -/
theorem line_and_point_determine_plane (l : Line3D) (p : Point3D) (h : ¬point_on_line p l) :
  ∃! plane : Plane3D, (∀ q : Point3D, point_on_line q l → point_on_plane q plane) ∧ point_on_plane p plane :=
sorry

/-- A trapezoid determines a unique plane -/
theorem trapezoid_determines_plane (p1 p2 p3 p4 : Point3D) 
  (h1 : ∃ l1 l2 : Line3D, point_on_line p1 l1 ∧ point_on_line p2 l1 ∧ point_on_line p3 l2 ∧ point_on_line p4 l2) 
  (h2 : l1 ≠ l2) :
  ∃! plane : Plane3D, point_on_plane p1 plane ∧ point_on_plane p2 plane ∧ point_on_plane p3 plane ∧ point_on_plane p4 plane :=
sorry

/-- The center and two points on a circle do not always determine a unique plane -/
theorem circle_points_not_always_determine_plane :
  ∃ (c : Circle3D) (p1 p2 : Point3D), 
    point_on_circle p1 c ∧ point_on_circle p2 c ∧ 
    ¬(∃! plane : Plane3D, point_on_plane c.center plane ∧ point_on_plane p1 plane ∧ point_on_plane p2 plane) :=
sorry

end NUMINAMATH_CALUDE_three_points_determine_plane_line_and_point_determine_plane_trapezoid_determines_plane_circle_points_not_always_determine_plane_l3369_336990


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l3369_336999

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips and 5 yellow chips, with replacement after the first draw. -/
theorem different_color_chips_probability :
  let blue_chips : ℕ := 7
  let yellow_chips : ℕ := 5
  let total_chips : ℕ := blue_chips + yellow_chips
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_different_colors : ℚ := prob_blue * prob_yellow + prob_yellow * prob_blue
  prob_different_colors = 35 / 72 := by
sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l3369_336999


namespace NUMINAMATH_CALUDE_caroline_lassis_l3369_336947

/-- Represents the number of lassis that can be made with given ingredients -/
def max_lassis (initial_lassis initial_mangoes initial_coconuts available_mangoes available_coconuts : ℚ) : ℚ :=
  min 
    (available_mangoes * (initial_lassis / initial_mangoes))
    (available_coconuts * (initial_lassis / initial_coconuts))

/-- Theorem stating that Caroline can make 55 lassis with the given ingredients -/
theorem caroline_lassis : 
  max_lassis 11 2 4 12 20 = 55 := by
  sorry

#eval max_lassis 11 2 4 12 20

end NUMINAMATH_CALUDE_caroline_lassis_l3369_336947


namespace NUMINAMATH_CALUDE_janinas_pancakes_l3369_336904

/-- Calculates the minimum number of pancakes Janina must sell to cover her expenses -/
theorem janinas_pancakes (rent : ℝ) (supplies : ℝ) (taxes_wages : ℝ) (price_per_pancake : ℝ) :
  rent = 75.50 →
  supplies = 28.40 →
  taxes_wages = 32.10 →
  price_per_pancake = 1.75 →
  ∃ n : ℕ, n ≥ 78 ∧ n * price_per_pancake ≥ rent + supplies + taxes_wages :=
by sorry

end NUMINAMATH_CALUDE_janinas_pancakes_l3369_336904


namespace NUMINAMATH_CALUDE_B_subset_A_l3369_336918

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem B_subset_A (B : Set ℝ) (h : A ∩ B = B) : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l3369_336918


namespace NUMINAMATH_CALUDE_unique_integer_for_complex_sixth_power_l3369_336932

theorem unique_integer_for_complex_sixth_power : 
  ∃! (n : ℤ), ∃ (m : ℤ), (n + Complex.I) ^ 6 = m := by sorry

end NUMINAMATH_CALUDE_unique_integer_for_complex_sixth_power_l3369_336932


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3369_336980

theorem complex_fraction_evaluation :
  let expr := (0.128 / 3.2 + 0.86) / ((5/6) * 1.2 + 0.8) * ((1 + 32/63 - 13/21) * 3.6) / (0.505 * 2/5 - 0.002)
  expr = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3369_336980


namespace NUMINAMATH_CALUDE_birthday_money_allocation_l3369_336976

theorem birthday_money_allocation (total : ℚ) (books snacks apps games : ℚ) : 
  total = 50 ∧ 
  books = (1 : ℚ) / 4 * total ∧
  snacks = (3 : ℚ) / 10 * total ∧
  apps = (7 : ℚ) / 20 * total ∧
  games = total - (books + snacks + apps) →
  games = 5 := by sorry

end NUMINAMATH_CALUDE_birthday_money_allocation_l3369_336976


namespace NUMINAMATH_CALUDE_question_mark_value_l3369_336961

theorem question_mark_value (x : ℝ) : (x * 74) / 30 = 1938.8 → x = 786 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l3369_336961


namespace NUMINAMATH_CALUDE_matthews_water_glass_size_l3369_336972

/-- Given Matthew's water drinking habits, prove the number of ounces in each glass. -/
theorem matthews_water_glass_size 
  (glasses_per_day : ℕ) 
  (bottle_size : ℕ) 
  (fills_per_week : ℕ) 
  (h1 : glasses_per_day = 4)
  (h2 : bottle_size = 35)
  (h3 : fills_per_week = 4) :
  (bottle_size * fills_per_week) / (glasses_per_day * 7) = 5 := by
  sorry

#check matthews_water_glass_size

end NUMINAMATH_CALUDE_matthews_water_glass_size_l3369_336972


namespace NUMINAMATH_CALUDE_greatest_x_value_l3369_336930

theorem greatest_x_value (x : ℤ) (h : (6.1 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 620) :
  x ≤ 2 ∧ ∃ y : ℤ, y > 2 → (6.1 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 620 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3369_336930


namespace NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_l3369_336953

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A diagonal of a quadrilateral is a line segment connecting two non-adjacent vertices. -/
def Quadrilateral.diagonal (q : Quadrilateral) (i j : Fin 4) : ℝ × ℝ → ℝ × ℝ :=
  sorry

/-- Two line segments bisect each other if they intersect at their midpoints. -/
def bisect (seg1 seg2 : ℝ × ℝ → ℝ × ℝ) : Prop :=
  sorry

/-- A parallelogram is a quadrilateral with two pairs of parallel sides. -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- If the diagonals of a quadrilateral bisect each other, then it is a parallelogram. -/
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  (∃ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ 
    bisect (q.diagonal i k) (q.diagonal j l)) →
  is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_l3369_336953


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l3369_336965

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l3369_336965


namespace NUMINAMATH_CALUDE_factorial_305_trailing_zeros_l3369_336903

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 305! ends with 75 zeros -/
theorem factorial_305_trailing_zeros :
  trailingZeros 305 = 75 := by
  sorry

end NUMINAMATH_CALUDE_factorial_305_trailing_zeros_l3369_336903


namespace NUMINAMATH_CALUDE_crayons_lost_theorem_l3369_336959

/-- The number of crayons Paul lost or gave away -/
def crayons_lost_or_given_away (initial_crayons remaining_crayons : ℕ) : ℕ :=
  initial_crayons - remaining_crayons

/-- Theorem: The number of crayons lost or given away is equal to the difference between
    the initial number of crayons and the remaining number of crayons -/
theorem crayons_lost_theorem (initial_crayons remaining_crayons : ℕ) 
  (h : initial_crayons ≥ remaining_crayons) :
  crayons_lost_or_given_away initial_crayons remaining_crayons = initial_crayons - remaining_crayons :=
by
  sorry

#eval crayons_lost_or_given_away 479 134

end NUMINAMATH_CALUDE_crayons_lost_theorem_l3369_336959


namespace NUMINAMATH_CALUDE_expression_simplification_l3369_336962

theorem expression_simplification (a : ℝ) (ha : a ≥ 0) :
  (((2 * (a + 1) + 2 * Real.sqrt (a^2 + 2*a)) / (3*a + 1 - 2 * Real.sqrt (a^2 + 2*a)))^(1/2 : ℝ)) -
  ((Real.sqrt (2*a + 1) - Real.sqrt a)⁻¹ * Real.sqrt (a + 2)) =
  Real.sqrt a / (Real.sqrt (2*a + 1) - Real.sqrt a) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3369_336962


namespace NUMINAMATH_CALUDE_no_integer_solution_l3369_336975

theorem no_integer_solution : ¬∃ (a b : ℤ), a^2 + b^2 = 10^100 + 3 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3369_336975


namespace NUMINAMATH_CALUDE_downstream_distance_is_100_l3369_336939

/-- Represents the properties of a boat traveling in a stream -/
structure BoatTravel where
  downstream_time : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the downstream distance given boat travel properties -/
def downstream_distance (bt : BoatTravel) : ℝ :=
  sorry

/-- Theorem stating that the downstream distance is 100 km given specific conditions -/
theorem downstream_distance_is_100 (bt : BoatTravel) 
  (h1 : bt.downstream_time = 10)
  (h2 : bt.upstream_distance = 200)
  (h3 : bt.upstream_time = 25)
  (h4 : bt.stream_speed = 1) :
  downstream_distance bt = 100 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_is_100_l3369_336939


namespace NUMINAMATH_CALUDE_committee_formation_count_l3369_336921

/-- The number of ways to form a committee with specified conditions -/
def committee_formations (total_members : ℕ) (committee_size : ℕ) (required_members : ℕ) : ℕ :=
  Nat.choose (total_members - required_members) (committee_size - required_members)

/-- Theorem: The number of ways to form a 5-person committee from a 12-member club,
    where two specific members must always be included, is equal to 120. -/
theorem committee_formation_count :
  committee_formations 12 5 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3369_336921


namespace NUMINAMATH_CALUDE_function_increasing_interval_implies_b_bound_l3369_336902

/-- Given a function f(x) = e^x(x^2 - bx) where b is a real number,
    if f(x) has an increasing interval in [1/2, 2],
    then b < 8/3 -/
theorem function_increasing_interval_implies_b_bound 
  (b : ℝ) 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = Real.exp x * (x^2 - b*x)) 
  (h_increasing : ∃ (a c : ℝ), 1/2 ≤ a ∧ c ≤ 2 ∧ StrictMonoOn f (Set.Icc a c)) : 
  b < 8/3 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_interval_implies_b_bound_l3369_336902


namespace NUMINAMATH_CALUDE_max_sum_abc_l3369_336909

theorem max_sum_abc (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : a + b = 719) 
  (h3 : c - a = 915) : 
  (∀ x y z : ℕ, x < y → x + y = 719 → z - x = 915 → x + y + z ≤ 1993) ∧ 
  (∃ x y z : ℕ, x < y ∧ x + y = 719 ∧ z - x = 915 ∧ x + y + z = 1993) :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l3369_336909


namespace NUMINAMATH_CALUDE_find_a_value_l3369_336960

theorem find_a_value (x y a : ℝ) 
  (h1 : x = 2) 
  (h2 : y = 1) 
  (h3 : a * x - y = 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3369_336960


namespace NUMINAMATH_CALUDE_hundredth_stationary_is_hundred_l3369_336988

/-- A function representing the sorting algorithm that swaps adjacent numbers if the larger number is on the left -/
def sortPass (s : List ℕ) : List ℕ := sorry

/-- A predicate that checks if a number at a given index remains stationary during both passes -/
def isStationary (s : List ℕ) (index : ℕ) : Prop := sorry

theorem hundredth_stationary_is_hundred {s : List ℕ} (h1 : s.length = 1982) 
  (h2 : ∀ n, n ∈ s → 1 ≤ n ∧ n ≤ 1982) 
  (h3 : isStationary s 100) : 
  s[99] = 100 := by sorry

end NUMINAMATH_CALUDE_hundredth_stationary_is_hundred_l3369_336988


namespace NUMINAMATH_CALUDE_three_sevenths_minus_forty_percent_l3369_336919

theorem three_sevenths_minus_forty_percent (x : ℝ) : 
  (0.3 * x = 63.0000000000001) → 
  ((3/7) * x - 0.4 * x = 6.00000000000006) := by
sorry

end NUMINAMATH_CALUDE_three_sevenths_minus_forty_percent_l3369_336919


namespace NUMINAMATH_CALUDE_book_sales_proof_l3369_336925

/-- Calculates the number of copies sold given the revenue per book, agent's commission percentage, and total amount kept by the author. -/
def calculate_copies_sold (revenue_per_book : ℚ) (agent_commission_percent : ℚ) (total_kept : ℚ) : ℚ :=
  total_kept / (revenue_per_book * (1 - agent_commission_percent / 100))

/-- Proves that given the specific conditions, the number of copies sold is 900,000. -/
theorem book_sales_proof (revenue_per_book : ℚ) (agent_commission_percent : ℚ) (total_kept : ℚ) 
    (h1 : revenue_per_book = 2)
    (h2 : agent_commission_percent = 10)
    (h3 : total_kept = 1620000) :
  calculate_copies_sold revenue_per_book agent_commission_percent total_kept = 900000 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_proof_l3369_336925


namespace NUMINAMATH_CALUDE_age_difference_l3369_336917

theorem age_difference (louis_age jerica_age matilda_age : ℕ) : 
  louis_age = 14 →
  jerica_age = 2 * louis_age →
  matilda_age = 35 →
  matilda_age - jerica_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3369_336917


namespace NUMINAMATH_CALUDE_alice_average_speed_l3369_336950

/-- Alice's cycling trip -/
theorem alice_average_speed :
  let distance1 : ℝ := 40  -- First segment distance in miles
  let speed1 : ℝ := 8      -- First segment speed in miles per hour
  let distance2 : ℝ := 20  -- Second segment distance in miles
  let speed2 : ℝ := 40     -- Second segment speed in miles per hour
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := distance1 / speed1 + distance2 / speed2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 120 / 11
  := by sorry

end NUMINAMATH_CALUDE_alice_average_speed_l3369_336950


namespace NUMINAMATH_CALUDE_min_value_theorem_l3369_336928

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3369_336928


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_least_one_false_l3369_336916

theorem not_p_and_q_implies_at_least_one_false (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_least_one_false_l3369_336916


namespace NUMINAMATH_CALUDE_percentage_calculation_l3369_336923

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 100 → 
  (P / 100) * (3 / 5 * N) = 36 → 
  P = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3369_336923


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l3369_336927

-- Define the operation [x,y,z]
def bracket (x y z : ℚ) : ℚ := (x + y) / z

-- Theorem statement
theorem nested_bracket_equals_two :
  bracket (bracket 120 60 180) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry


end NUMINAMATH_CALUDE_nested_bracket_equals_two_l3369_336927


namespace NUMINAMATH_CALUDE_quiz_mcq_count_l3369_336955

theorem quiz_mcq_count :
  ∀ (n : ℕ),
  (((1 : ℚ) / 3) ^ n * ((1 : ℚ) / 2) ^ 2 = (1 : ℚ) / 12) →
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_quiz_mcq_count_l3369_336955


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_1001_l3369_336973

-- Define a three-digit number
def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

-- Define the six-digit number formed by repeating the three-digit number
def six_digit_number (a b c : Nat) : Nat :=
  1000 * (three_digit_number a b c) + (three_digit_number a b c)

-- Theorem statement
theorem six_digit_divisible_by_1001 (a b c : Nat) :
  (a < 10) → (b < 10) → (c < 10) →
  (six_digit_number a b c) % 1001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_1001_l3369_336973


namespace NUMINAMATH_CALUDE_refrigerator_production_days_l3369_336970

/-- The number of additional days needed to complete refrigerator production -/
def additional_days (total_required : ℕ) (days_worked : ℕ) (initial_rate : ℕ) (increased_rate : ℕ) : ℕ :=
  let produced := days_worked * initial_rate
  let remaining := total_required - produced
  remaining / increased_rate

theorem refrigerator_production_days : 
  additional_days 1590 12 80 90 = 7 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_production_days_l3369_336970


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l3369_336985

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

theorem tangent_line_at_zero : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    y = m * x + b ∧ 
    (∃ (h : ℝ), h ≠ 0 ∧ (f (0 + h) - f 0) / h = m) ∧
    f 0 = b ∧
    m = 1 ∧ b = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l3369_336985


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l3369_336968

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_implies_a_equals_5 :
  ∀ a : ℝ, (∃ (ε : ℝ), ε > 0 ∧ ∀ x : ℝ, 
    x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l3369_336968


namespace NUMINAMATH_CALUDE_problem_statement_l3369_336986

/-- Given real numbers a and b satisfying the conditions, 
    prove the minimum value of m and the inequality for x, y, z -/
theorem problem_statement 
  (a b : ℝ) 
  (h1 : a * b > 0) 
  (h2 : a^2 * b = 2) 
  (m : ℝ := a * b + a^2) : 
  (∃ (t : ℝ), t = 3 ∧ ∀ m', m' = a * b + a^2 → m' ≥ t) ∧ 
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |x + 2*y + 2*z| ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3369_336986


namespace NUMINAMATH_CALUDE_equation_solution_l3369_336911

theorem equation_solution : ∃! x : ℝ, 90 + 5 * 12 / (x / 3) = 91 ∧ x = 180 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3369_336911


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l3369_336929

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 20)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 168 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l3369_336929


namespace NUMINAMATH_CALUDE_cricket_average_proof_l3369_336914

def cricket_average (total_matches : ℕ) (first_set : ℕ) (second_set : ℕ) 
  (first_avg : ℚ) (second_avg : ℚ) : ℚ :=
  let total_first := first_avg * first_set
  let total_second := second_avg * second_set
  (total_first + total_second) / total_matches

theorem cricket_average_proof :
  cricket_average 10 6 4 41 (35.75) = 38.9 := by
  sorry

#eval cricket_average 10 6 4 41 (35.75)

end NUMINAMATH_CALUDE_cricket_average_proof_l3369_336914


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3369_336910

theorem purely_imaginary_complex_number (a : ℝ) : 
  (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I = (0 : ℂ) + ((a - 1)*I)) → a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3369_336910


namespace NUMINAMATH_CALUDE_probability_of_no_growth_pie_l3369_336983

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ := 7/10

theorem probability_of_no_growth_pie :
  (1 - (Nat.choose (total_pies - growth_pies) (pies_given - growth_pies) : ℚ) / 
   (Nat.choose total_pies pies_given : ℚ)) = probability_no_growth_pie :=
sorry

end NUMINAMATH_CALUDE_probability_of_no_growth_pie_l3369_336983


namespace NUMINAMATH_CALUDE_system_has_solution_solutions_for_a_nonpositive_or_one_solutions_for_a_between_zero_and_two_solutions_for_a_geq_two_l3369_336978

/-- The system of equations has at least one solution for all real a -/
theorem system_has_solution (a : ℝ) : ∃ x y : ℝ, 
  (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x) := by sorry

/-- Solutions for a ≤ 0 or a = 1 -/
theorem solutions_for_a_nonpositive_or_one (a : ℝ) (h : a ≤ 0 ∨ a = 1) : 
  (∃ x y : ℝ, x = 0 ∧ y = 1 - a / 2 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) := by sorry

/-- Solutions for 0 < a < 2, a ≠ 1 -/
theorem solutions_for_a_between_zero_and_two (a : ℝ) (h : 0 < a ∧ a < 2 ∧ a ≠ 1) : 
  (∃ x y : ℝ, x = 0 ∧ y = 1 - a / 2 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = ((2 - a) / a)^2 ∧ y = (2 - a) / a ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) := by sorry

/-- Solutions for a ≥ 2 -/
theorem solutions_for_a_geq_two (a : ℝ) (h : a ≥ 2) : 
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_system_has_solution_solutions_for_a_nonpositive_or_one_solutions_for_a_between_zero_and_two_solutions_for_a_geq_two_l3369_336978


namespace NUMINAMATH_CALUDE_percentage_increase_l3369_336991

theorem percentage_increase (x : ℝ) (h1 : x > 40) (h2 : x = 48) :
  (x - 40) / 40 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3369_336991


namespace NUMINAMATH_CALUDE_square_sum_value_l3369_336943

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3369_336943


namespace NUMINAMATH_CALUDE_no_x_squared_term_l3369_336937

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 2*a*x + a^2) = x^3 + (a^2 - 2*a)*x + a^2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l3369_336937


namespace NUMINAMATH_CALUDE_carriage_sharing_problem_l3369_336977

theorem carriage_sharing_problem (x : ℕ) : 
  (x / 3 : ℚ) + 2 = (x - 9 : ℚ) / 2 ↔ 
  (∃ (total_carriages : ℕ), 
    (x / 3 : ℚ) + 2 = total_carriages ∧ 
    (x - 9 : ℚ) / 2 = total_carriages) :=
sorry

end NUMINAMATH_CALUDE_carriage_sharing_problem_l3369_336977


namespace NUMINAMATH_CALUDE_mixture_alcohol_percentage_l3369_336931

/-- Represents the properties of an alcohol solution -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the volume of alcohol in a solution -/
def alcoholVolume (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

/-- Theorem: Adding 50 mL of 30% alcohol solution to 200 mL of 10% alcohol solution results in 14% alcohol solution -/
theorem mixture_alcohol_percentage 
  (x : Solution) 
  (y : Solution) 
  (h1 : x.volume = 200)
  (h2 : x.alcoholPercentage = 0.1)
  (h3 : y.volume = 50)
  (h4 : y.alcoholPercentage = 0.3) :
  let finalSolution : Solution := {
    volume := x.volume + y.volume,
    alcoholPercentage := (alcoholVolume x + alcoholVolume y) / (x.volume + y.volume)
  }
  finalSolution.alcoholPercentage = 0.14 := by
  sorry

#check mixture_alcohol_percentage

end NUMINAMATH_CALUDE_mixture_alcohol_percentage_l3369_336931


namespace NUMINAMATH_CALUDE_total_distance_covered_l3369_336987

/-- The total distance covered by a fox, rabbit, and deer given their speeds and running times -/
theorem total_distance_covered 
  (fox_speed : ℝ) 
  (rabbit_speed : ℝ) 
  (deer_speed : ℝ) 
  (fox_time : ℝ) 
  (rabbit_time : ℝ) 
  (deer_time : ℝ) 
  (h1 : fox_speed = 50) 
  (h2 : rabbit_speed = 60) 
  (h3 : deer_speed = 80) 
  (h4 : fox_time = 2) 
  (h5 : rabbit_time = 5/3) 
  (h6 : deer_time = 3/2) : 
  fox_speed * fox_time + rabbit_speed * rabbit_time + deer_speed * deer_time = 320 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_covered_l3369_336987


namespace NUMINAMATH_CALUDE_circle_properties_l3369_336901

/-- A circle in the xy-plane is defined by the equation x^2 + y^2 - 6x = 0. -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The center of the circle is the point (h, k) in ℝ² -/
def circle_center : ℝ × ℝ := (3, 0)

/-- The radius of the circle is r -/
def circle_radius : ℝ := 3

/-- Theorem stating that the given equation describes a circle with center (3, 0) and radius 3 -/
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3369_336901


namespace NUMINAMATH_CALUDE_class_average_l3369_336956

theorem class_average (total_students : Nat) (perfect_scores : Nat) (zero_scores : Nat) (rest_average : Nat) : 
  total_students = 20 →
  perfect_scores = 2 →
  zero_scores = 3 →
  rest_average = 40 →
  (perfect_scores * 100 + zero_scores * 0 + (total_students - perfect_scores - zero_scores) * rest_average) / total_students = 40 := by
sorry

end NUMINAMATH_CALUDE_class_average_l3369_336956


namespace NUMINAMATH_CALUDE_binary_multiplication_subtraction_l3369_336964

-- Define binary numbers as natural numbers
def binary_11011 : ℕ := 27
def binary_1101 : ℕ := 13
def binary_1010 : ℕ := 10

-- Define the expected result
def expected_result : ℕ := 409

-- Theorem statement
theorem binary_multiplication_subtraction :
  (binary_11011 * binary_1101) - binary_1010 = expected_result :=
by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_subtraction_l3369_336964


namespace NUMINAMATH_CALUDE_tangent_two_implies_fraction_equals_negative_one_third_l3369_336967

theorem tangent_two_implies_fraction_equals_negative_one_third (α : ℝ) 
  (h : Real.tan α = 2) : 
  (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 + α) - Real.cos (3 * Real.pi / 2 - α)) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_implies_fraction_equals_negative_one_third_l3369_336967


namespace NUMINAMATH_CALUDE_triangle_side_length_l3369_336966

/-- Given a triangle ABC with ∠A = 40°, ∠B = 90°, and AC = 6, prove that BC = 6 * sin(40°) -/
theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let angle (P Q R : ℝ × ℝ) := Real.arccos ((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) / 
    (Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) * Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2))
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  angle B A C = Real.pi / 4.5 →  -- 40°
  angle A B C = Real.pi / 2 →    -- 90°
  dist A C = 6 →
  dist B C = 6 * Real.sin (Real.pi / 4.5) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3369_336966


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3369_336924

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a + 2 > 0) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3369_336924


namespace NUMINAMATH_CALUDE_largest_space_diagonal_squared_of_box_l3369_336944

/-- The square of the largest possible length of the space diagonal of a smaller box -/
def largest_space_diagonal_squared (a b c : ℕ) : ℕ :=
  max
    (a * a + (b / 2) * (b / 2) + c * c)
    (max
      (a * a + b * b + (c / 2) * (c / 2))
      ((a / 2) * (a / 2) + b * b + c * c))

/-- Theorem stating the largest possible space diagonal squared for the given box -/
theorem largest_space_diagonal_squared_of_box :
  largest_space_diagonal_squared 1 2 16 = 258 := by
  sorry

end NUMINAMATH_CALUDE_largest_space_diagonal_squared_of_box_l3369_336944


namespace NUMINAMATH_CALUDE_forest_coverage_growth_rate_l3369_336994

theorem forest_coverage_growth_rate (x : ℝ) : 
  (0.63 * (1 + x)^2 = 0.68) ↔ 
  (∃ (rate : ℝ → ℝ), 
    rate 0 = 0.63 ∧ 
    rate 2 = 0.68 ∧ 
    ∀ t, 0 ≤ t → t ≤ 2 → rate t = 0.63 * (1 + x)^t) :=
sorry

end NUMINAMATH_CALUDE_forest_coverage_growth_rate_l3369_336994


namespace NUMINAMATH_CALUDE_sodium_hydroxide_combined_l3369_336984

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between acetic acid and sodium hydroxide -/
structure Reaction where
  acetic_acid : Moles
  sodium_hydroxide : Moles
  sodium_acetate : Moles

/-- The reaction occurs in a 1:1 molar ratio -/
axiom reaction_ratio (r : Reaction) : r.acetic_acid = r.sodium_hydroxide

/-- The number of moles of sodium acetate formed equals the number of moles of acetic acid used -/
axiom sodium_acetate_formation (r : Reaction) : r.sodium_acetate = r.acetic_acid

theorem sodium_hydroxide_combined (r : Reaction) :
  r.sodium_hydroxide = r.sodium_acetate :=
by sorry

end NUMINAMATH_CALUDE_sodium_hydroxide_combined_l3369_336984


namespace NUMINAMATH_CALUDE_angle_U_is_90_degrees_l3369_336981

-- Define the hexagon FIGURE
structure Hexagon where
  F : ℝ
  I : ℝ
  U : ℝ
  G : ℝ
  R : ℝ
  E : ℝ

-- Define the conditions
def hexagon_conditions (h : Hexagon) : Prop :=
  h.F = h.I ∧ h.I = h.U ∧ 
  h.G + h.E = 180 ∧ 
  h.R + h.U = 180 ∧
  h.F + h.I + h.U + h.G + h.R + h.E = 720

-- Theorem statement
theorem angle_U_is_90_degrees (h : Hexagon) 
  (hc : hexagon_conditions h) : h.U = 90 := by sorry

end NUMINAMATH_CALUDE_angle_U_is_90_degrees_l3369_336981


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l3369_336938

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

/-- The theorem statement -/
theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_eq : a 7 = a 6 + 2 * a 5)
  (h_sqrt : ∃ m n : ℕ, Real.sqrt (a m * a n) = 2 * a 1) :
  (∃ m n : ℕ, (1 : ℝ) / m + 9 / n = 4) ∧
  (∀ m n : ℕ, (1 : ℝ) / m + 9 / n ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_min_value_l3369_336938


namespace NUMINAMATH_CALUDE_arrangements_theorem_l3369_336952

/-- The number of arrangements of 5 people in a row with exactly 1 person between A and B -/
def arrangements_count : ℕ := 36

/-- The total number of people in the row -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

/-- Theorem stating that the number of arrangements is 36 -/
theorem arrangements_theorem :
  (arrangements_count = 36) ∧
  (total_people = 5) ∧
  (people_between = 1) :=
sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l3369_336952


namespace NUMINAMATH_CALUDE_revenue_maximization_l3369_336958

/-- Revenue function for a scenic area with three ticket options -/
def revenue (x : ℝ) : ℝ := -0.1 * x^2 + 1.8 * x + 180

/-- Initial number of people choosing option A -/
def initial_A : ℝ := 20000

/-- Initial number of people choosing option B -/
def initial_B : ℝ := 10000

/-- Initial number of people choosing combined option -/
def initial_combined : ℝ := 10000

/-- Number of people switching from A to combined per 1 yuan decrease -/
def switch_rate_A : ℝ := 400

/-- Number of people switching from B to combined per 1 yuan decrease -/
def switch_rate_B : ℝ := 600

/-- Price of ticket A -/
def price_A : ℝ := 30

/-- Price of ticket B -/
def price_B : ℝ := 50

/-- Initial price of combined ticket -/
def initial_price_combined : ℝ := 70

theorem revenue_maximization :
  ∃ (x : ℝ), x = 9 ∧ 
  revenue x = 188.1 ∧ 
  ∀ y, revenue y ≤ revenue x :=
sorry

#check revenue_maximization

end NUMINAMATH_CALUDE_revenue_maximization_l3369_336958


namespace NUMINAMATH_CALUDE_nonreal_cubic_root_sum_l3369_336936

/-- Given ω is a nonreal cubic root of unity, 
    prove that (2 - ω + 2ω^2)^6 + (2 + ω - 2ω^2)^6 = 38908 -/
theorem nonreal_cubic_root_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ (1 : ℂ) → (2 - ω + 2*ω^2)^6 + (2 + ω - 2*ω^2)^6 = 38908 := by
  sorry

end NUMINAMATH_CALUDE_nonreal_cubic_root_sum_l3369_336936


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3369_336993

/-- Given a quadratic function f(x) = x^2 - 2ax + b where a > 1,
    if both the domain and range of f are [1, a], then b = 5. -/
theorem quadratic_function_property (a b : ℝ) (f : ℝ → ℝ) :
  a > 1 →
  (∀ x, f x = x^2 - 2*a*x + b) →
  (∀ x, x ∈ Set.Icc 1 a ↔ f x ∈ Set.Icc 1 a) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3369_336993


namespace NUMINAMATH_CALUDE_distance_from_negative_one_l3369_336905

theorem distance_from_negative_one : ∀ x : ℝ, |x - (-1)| = 6 ↔ x = 5 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_negative_one_l3369_336905


namespace NUMINAMATH_CALUDE_range_of_m_l3369_336992

/-- The proposition p -/
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0

/-- The proposition q -/
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

/-- q is a sufficient but not necessary condition for p -/
def q_sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ ¬(∀ x, p x m → q x)

theorem range_of_m :
  ∀ m : ℝ, q_sufficient_not_necessary m ↔ -1/3 ≤ m ∧ m ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3369_336992


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l3369_336954

theorem sandcastle_height_difference (miki_height sister_height : ℝ) 
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5) :
  miki_height - sister_height = 0.33 := by
sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l3369_336954


namespace NUMINAMATH_CALUDE_air_quality_probability_l3369_336920

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) 
  (h1 : p_good = 0.75) (h2 : p_consecutive = 0.6) : 
  p_consecutive / p_good = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l3369_336920


namespace NUMINAMATH_CALUDE_inverse_sum_product_l3369_336942

theorem inverse_sum_product (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hsum : 3*x + y/3 + z ≠ 0) : 
  (3*x + y/3 + z)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹ + z⁻¹) = (x*y*z)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l3369_336942


namespace NUMINAMATH_CALUDE_max_squares_covered_l3369_336963

/-- Represents a square card with side length 2 inches -/
structure Card where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- Represents a checkerboard with 1-inch squares -/
structure Checkerboard where
  square_size : ℝ
  square_size_eq : square_size = 1

/-- The maximum number of squares that can be covered by the card -/
def max_covered_squares : ℕ := 16

/-- Theorem stating the maximum number of squares that can be covered -/
theorem max_squares_covered (card : Card) (board : Checkerboard) :
  ∃ (n : ℕ), n = max_covered_squares ∧ 
  n = (max_covered_squares : ℝ) ∧
  ∀ (m : ℕ), (m : ℝ) ≤ (card.side_length / board.square_size) ^ 2 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_squares_covered_l3369_336963


namespace NUMINAMATH_CALUDE_add_1857_minutes_to_noon_l3369_336906

/-- Represents a time of day --/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  is_pm : Bool

/-- Adds minutes to a given time --/
def addMinutes (t : TimeOfDay) (m : Nat) : TimeOfDay :=
  sorry

/-- Checks if two times are equal --/
def timeEqual (t1 t2 : TimeOfDay) : Prop :=
  t1.hours = t2.hours ∧ t1.minutes = t2.minutes ∧ t1.is_pm = t2.is_pm

theorem add_1857_minutes_to_noon :
  let noon := TimeOfDay.mk 12 0 true
  let result := TimeOfDay.mk 6 57 false
  timeEqual (addMinutes noon 1857) result := by
  sorry

end NUMINAMATH_CALUDE_add_1857_minutes_to_noon_l3369_336906


namespace NUMINAMATH_CALUDE_quadratic_equation_linear_term_l3369_336949

theorem quadratic_equation_linear_term 
  (m : ℝ) 
  (h : 2 * m = 6) : 
  ∃ (a b c : ℝ), 
    a * x^2 + b * x + c = 0 ∧ 
    c = 6 ∧ 
    b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_linear_term_l3369_336949


namespace NUMINAMATH_CALUDE_smallest_integer_y_l3369_336996

theorem smallest_integer_y (y : ℤ) : (∀ z : ℤ, z < y → 3 * z - 6 ≥ 15) ∧ 3 * y - 6 < 15 ↔ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l3369_336996


namespace NUMINAMATH_CALUDE_leo_current_weight_l3369_336945

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 104

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 180 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 180

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) →
  leo_weight = 104 := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l3369_336945
