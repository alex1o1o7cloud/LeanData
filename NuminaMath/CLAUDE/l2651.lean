import Mathlib

namespace NUMINAMATH_CALUDE_thirteenth_term_is_15_l2651_265152

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  fifth_term : a 5 = 1
  sum_property : a 8 + a 10 = 16

/-- The 13th term of the arithmetic sequence is 15 -/
theorem thirteenth_term_is_15 (seq : ArithmeticSequence) : seq.a 13 = 15 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_is_15_l2651_265152


namespace NUMINAMATH_CALUDE_book_sales_ratio_l2651_265192

theorem book_sales_ratio : 
  ∀ (T : ℕ), -- Number of books sold on Thursday
  15 + T + T / 5 = 69 → -- Total sales equation
  T / 15 = 3 -- Ratio of Thursday to Wednesday sales
  := by sorry

end NUMINAMATH_CALUDE_book_sales_ratio_l2651_265192


namespace NUMINAMATH_CALUDE_scaling_2_3_to_3_2_l2651_265185

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.x_scale * p.1, t.y_scale * p.2)

/-- The scaling transformation that changes (2, 3) to (3, 2) -/
theorem scaling_2_3_to_3_2 : 
  ∃ (t : ScalingTransformation), apply_scaling t (2, 3) = (3, 2) ∧ 
    t.x_scale = 3/2 ∧ t.y_scale = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_scaling_2_3_to_3_2_l2651_265185


namespace NUMINAMATH_CALUDE_class_size_class_size_is_60_l2651_265149

theorem class_size (cafeteria_students : ℕ) (no_lunch_students : ℕ) : ℕ :=
  let bring_lunch_students := 3 * cafeteria_students
  let total_lunch_students := cafeteria_students + bring_lunch_students
  let total_students := total_lunch_students + no_lunch_students
  total_students

theorem class_size_is_60 : 
  class_size 10 20 = 60 := by sorry

end NUMINAMATH_CALUDE_class_size_class_size_is_60_l2651_265149


namespace NUMINAMATH_CALUDE_shawn_pebble_groups_l2651_265101

theorem shawn_pebble_groups :
  let total_pebbles : ℕ := 40
  let red_pebbles : ℕ := 9
  let blue_pebbles : ℕ := 13
  let remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
  let blue_yellow_diff : ℕ := 7
  let yellow_pebbles : ℕ := blue_pebbles - blue_yellow_diff
  let num_colors : ℕ := 3  -- purple, yellow, and green
  ∃ (num_groups : ℕ),
    num_groups > 0 ∧
    num_groups ∣ remaining_pebbles ∧
    num_groups % num_colors = 0 ∧
    remaining_pebbles / num_groups = yellow_pebbles ∧
    num_groups = 3
  := by sorry

end NUMINAMATH_CALUDE_shawn_pebble_groups_l2651_265101


namespace NUMINAMATH_CALUDE_f_satisfies_condition_l2651_265191

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(1 / (log x))

-- State the theorem
theorem f_satisfies_condition (a : ℝ) (h_a : a > 1) :
  ∀ (x y u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f a (x^u * y^v) ≤ (f a x)^(1/(1*u)) * (f a y)^(1/10) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_condition_l2651_265191


namespace NUMINAMATH_CALUDE_slope_angle_vertical_line_l2651_265150

/-- Given two points A(2, 1) and B(2, 3), prove that the slope angle of the line AB is 90 degrees. -/
theorem slope_angle_vertical_line : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (2, 3)
  let slope_angle := Real.arctan ((B.2 - A.2) / (B.1 - A.1)) * (180 / Real.pi)
  slope_angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_vertical_line_l2651_265150


namespace NUMINAMATH_CALUDE_inequality_proof_l2651_265123

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2651_265123


namespace NUMINAMATH_CALUDE_cookies_problem_l2651_265137

theorem cookies_problem (millie mike frank : ℕ) : 
  millie = 4 →
  mike = 3 * millie →
  frank = mike / 2 - 3 →
  frank = 3 := by
sorry

end NUMINAMATH_CALUDE_cookies_problem_l2651_265137


namespace NUMINAMATH_CALUDE_smallest_third_altitude_l2651_265113

/-- Represents a scalene triangle with altitudes --/
structure ScaleneTriangle where
  -- The lengths of the three sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the three altitudes
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  -- Conditions for a scalene triangle
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Area equality for altitudes
  area_equality : a * h_a = b * h_b ∧ b * h_b = c * h_c

/-- The theorem stating the smallest possible integer length for the third altitude --/
theorem smallest_third_altitude (t : ScaleneTriangle) 
  (h1 : t.h_a = 6 ∨ t.h_b = 6 ∨ t.h_c = 6)
  (h2 : t.h_a = 8 ∨ t.h_b = 8 ∨ t.h_c = 8)
  (h3 : ∃ (n : ℕ), t.h_a = n ∨ t.h_b = n ∨ t.h_c = n) :
  ∃ (h : ScaleneTriangle), h.h_a = 6 ∧ h.h_b = 8 ∧ h.h_c = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_third_altitude_l2651_265113


namespace NUMINAMATH_CALUDE_power_sum_equals_40_l2651_265142

theorem power_sum_equals_40 : (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 + 2^4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_40_l2651_265142


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2651_265114

/-- Calculates the total surface area of a cube with holes --/
def totalSurfaceArea (cubeEdgeLength : ℝ) (holeEdgeLength : ℝ) : ℝ :=
  let originalSurfaceArea := 6 * cubeEdgeLength^2
  let holeArea := 6 * holeEdgeLength^2
  let exposedInsideArea := 6 * 4 * holeEdgeLength^2
  originalSurfaceArea - holeArea + exposedInsideArea

/-- The total surface area of a cube with edge length 4 and holes of side length 2 is 168 --/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 4 2 = 168 := by
  sorry

#eval totalSurfaceArea 4 2

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2651_265114


namespace NUMINAMATH_CALUDE_abs_gt_not_sufficient_nor_necessary_l2651_265183

theorem abs_gt_not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, abs x > abs y ∧ x ≤ y) ∧ 
  (∃ u v : ℝ, u > v ∧ abs u ≤ abs v) := by
sorry

end NUMINAMATH_CALUDE_abs_gt_not_sufficient_nor_necessary_l2651_265183


namespace NUMINAMATH_CALUDE_playground_children_count_l2651_265127

/-- The number of boys on the playground at recess -/
def num_boys : ℕ := 27

/-- The number of girls on the playground at recess -/
def num_girls : ℕ := 35

/-- The total number of children on the playground at recess -/
def total_children : ℕ := num_boys + num_girls

theorem playground_children_count : total_children = 62 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l2651_265127


namespace NUMINAMATH_CALUDE_total_average_donation_l2651_265133

/-- Represents the donation statistics for two units A and B -/
structure DonationStats where
  avg_donation_A : ℝ
  num_people_A : ℕ
  num_people_B : ℕ

/-- The conditions of the donation problem -/
def donation_conditions (stats : DonationStats) : Prop :=
  -- Unit B donated twice as much as unit A
  (stats.avg_donation_A * stats.num_people_A) * 2 = (stats.avg_donation_A - 100) * stats.num_people_B
  -- The average donation per person in unit B is $100 less than the average donation per person in unit A
  ∧ (stats.avg_donation_A - 100) > 0
  -- The number of people in unit A is one-fourth of the number of people in unit B
  ∧ stats.num_people_A * 4 = stats.num_people_B

/-- The theorem stating that the total average donation is $120 -/
theorem total_average_donation (stats : DonationStats) 
  (h : donation_conditions stats) : 
  (stats.avg_donation_A * stats.num_people_A + (stats.avg_donation_A - 100) * stats.num_people_B) / 
  (stats.num_people_A + stats.num_people_B) = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_average_donation_l2651_265133


namespace NUMINAMATH_CALUDE_rhombuses_in_five_by_five_grid_l2651_265156

/-- Represents a grid of equilateral triangles -/
structure TriangleGrid where
  rows : Nat
  cols : Nat

/-- Calculates the number of rhombuses in a triangle grid -/
def count_rhombuses (grid : TriangleGrid) : Nat :=
  sorry

/-- Theorem stating that a 5x5 grid of equilateral triangles contains 30 rhombuses -/
theorem rhombuses_in_five_by_five_grid :
  let grid : TriangleGrid := { rows := 5, cols := 5 }
  count_rhombuses grid = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombuses_in_five_by_five_grid_l2651_265156


namespace NUMINAMATH_CALUDE_exactly_three_two_digit_multiples_l2651_265198

theorem exactly_three_two_digit_multiples :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ (∃! (m : Finset ℕ), 
      (∀ y ∈ m, y ≥ 10 ∧ y ≤ 99 ∧ ∃ k : ℕ, y = k * x) ∧ 
      m.card = 3)) ∧ 
    s.card = 9 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_two_digit_multiples_l2651_265198


namespace NUMINAMATH_CALUDE_inevitable_not_random_l2651_265129

-- Define the Event type
inductive Event
| Random : Event
| Inevitable : Event
| Impossible : Event

-- Define properties of events
def mayOccur (e : Event) : Prop :=
  match e with
  | Event.Random => true
  | Event.Inevitable => true
  | Event.Impossible => false

def willDefinitelyOccur (e : Event) : Prop :=
  match e with
  | Event.Inevitable => true
  | _ => false

-- Theorem: An inevitable event is not a random event
theorem inevitable_not_random (e : Event) :
  willDefinitelyOccur e → e ≠ Event.Random := by
  sorry

end NUMINAMATH_CALUDE_inevitable_not_random_l2651_265129


namespace NUMINAMATH_CALUDE_johnson_family_seating_l2651_265115

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_seatings (n : ℕ) : ℕ := factorial n

def boys_not_adjacent (boys girls : ℕ) : ℕ := 
  2 * factorial boys * factorial girls

theorem johnson_family_seating (boys girls : ℕ) : 
  boys = 5 → girls = 4 → 
  total_seatings (boys + girls) - boys_not_adjacent boys girls = 357120 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l2651_265115


namespace NUMINAMATH_CALUDE_set_intersection_union_problem_l2651_265162

theorem set_intersection_union_problem (a b : ℝ) :
  let M : Set ℝ := {3, 2^a}
  let N : Set ℝ := {a, b}
  (M ∩ N = {2}) → (M ∪ N = {1, 2, 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_problem_l2651_265162


namespace NUMINAMATH_CALUDE_triangle_theorem_l2651_265188

-- Define the triangle ABC
structure Triangle (α : Type*) [Field α] where
  a : α
  b : α
  c : α

-- Define the existence of point P
def exists_unique_point (t : Triangle ℝ) : Prop :=
  t.c ≠ t.a ∧ t.a ≠ t.b

-- Define the angle BAC
noncomputable def angle_BAC (t : Triangle ℝ) : ℝ :=
  Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

-- Main theorem
theorem triangle_theorem (t : Triangle ℝ) :
  (exists_unique_point t) ∧
  (∃ (P : ℝ × ℝ), (angle_BAC t < Real.pi / 3)) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2651_265188


namespace NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l2651_265171

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees between each hour on a clock face -/
def degrees_per_hour : ℕ := full_circle_degrees / clock_hours

/-- The number of hour spaces between 12 and 4 on a clock face -/
def spaces_12_to_4 : ℕ := 4

/-- The smaller angle formed by the hands of a clock at 4 o'clock -/
def clock_angle_at_4 : ℕ := spaces_12_to_4 * degrees_per_hour

theorem smaller_angle_at_4_oclock :
  clock_angle_at_4 = 120 :=
sorry

end NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l2651_265171


namespace NUMINAMATH_CALUDE_b_work_fraction_proof_l2651_265107

/-- The fraction of a day that b works --/
def b_work_fraction : ℚ := 1 / 5

/-- The time it takes a and b together to complete the work (in days) --/
def together_time : ℚ := 12

/-- The time it takes a alone to complete the work (in days) --/
def a_alone_time : ℚ := 20

/-- The time it takes a and b together to complete the work when b works a fraction of a day (in days) --/
def partial_together_time : ℚ := 15

theorem b_work_fraction_proof :
  (1 / a_alone_time + b_work_fraction * (1 / together_time) = 1 / partial_together_time) ∧
  (b_work_fraction > 0) ∧ (b_work_fraction < 1) := by
  sorry

end NUMINAMATH_CALUDE_b_work_fraction_proof_l2651_265107


namespace NUMINAMATH_CALUDE_johns_nap_hours_l2651_265117

/-- Calculates the total hours of naps taken over a given number of days -/
def total_nap_hours (naps_per_week : ℕ) (hours_per_nap : ℕ) (total_days : ℕ) : ℕ :=
  (total_days / 7) * naps_per_week * hours_per_nap

/-- Theorem: John's total nap hours in 70 days -/
theorem johns_nap_hours :
  total_nap_hours 3 2 70 = 60 := by
  sorry

end NUMINAMATH_CALUDE_johns_nap_hours_l2651_265117


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2651_265143

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + a*y + 1 = 0 → y = x) → 
  a = -2 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2651_265143


namespace NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l2651_265186

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 14th term of an arithmetic sequence given its 5th and 8th terms -/
theorem arithmetic_sequence_14th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_5 : a 5 = 6)
  (h_8 : a 8 = 15) :
  a 14 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l2651_265186


namespace NUMINAMATH_CALUDE_triangle_area_l2651_265145

/-- Given a triangle ABC where angle A is 30°, angle B is 45°, and side a is 2,
    prove that the area of the triangle is √3 + 1. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 2 →
  (1/2) * a * b * Real.sin (π - A - B) = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2651_265145


namespace NUMINAMATH_CALUDE_lifting_capacity_proof_l2651_265154

/-- Calculates the total weight a person can lift with both hands after training and specializing,
    given their initial lifting capacity per hand. -/
def totalLiftingCapacity (initialCapacity : ℝ) : ℝ :=
  let doubledCapacity := initialCapacity * 2
  let specializedCapacity := doubledCapacity * 1.1
  specializedCapacity * 2

/-- Proves that given an initial lifting capacity of 80 kg per hand,
    the total weight that can be lifted with both hands after training and specializing is 352 kg. -/
theorem lifting_capacity_proof :
  totalLiftingCapacity 80 = 352 := by
  sorry

#eval totalLiftingCapacity 80

end NUMINAMATH_CALUDE_lifting_capacity_proof_l2651_265154


namespace NUMINAMATH_CALUDE_x_value_l2651_265122

/-- A sequence where the differences between successive terms increase by 3 each time -/
def increasing_diff_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3 * n + 3

/-- The specific sequence from the problem -/
def our_sequence (a : ℕ → ℕ) : Prop :=
  increasing_diff_sequence a ∧ a 0 = 2 ∧ a 5 = 47

theorem x_value (a : ℕ → ℕ) (h : our_sequence a) : a 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2651_265122


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2651_265165

/-- The distance from point A(-2, 1) to the y-axis is 2 -/
theorem distance_to_y_axis : 
  let A : ℝ × ℝ := (-2, 1)
  abs A.1 = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2651_265165


namespace NUMINAMATH_CALUDE_rectangle_sides_l2651_265177

theorem rectangle_sides (S d : ℝ) (h1 : S > 0) (h2 : d ≥ 0) :
  let a := Real.sqrt (S + d^2 / 4) + d / 2
  let b := Real.sqrt (S + d^2 / 4) - d / 2
  a * b = S ∧ a - b = d ∧ a > 0 ∧ b > 0 := by sorry

end NUMINAMATH_CALUDE_rectangle_sides_l2651_265177


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2651_265178

theorem water_tank_capacity : ∃ (C : ℝ), 
  (C > 0) ∧ (0.40 * C - 0.25 * C = 36) ∧ (C = 240) := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2651_265178


namespace NUMINAMATH_CALUDE_sqrt_four_ninths_l2651_265118

theorem sqrt_four_ninths :
  Real.sqrt (4/9) = 2/3 ∨ Real.sqrt (4/9) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_ninths_l2651_265118


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2651_265180

theorem quadratic_form_equivalence : ∀ x : ℝ, x^2 + 6*x - 2 = (x + 3)^2 - 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2651_265180


namespace NUMINAMATH_CALUDE_fertilizer_production_l2651_265166

/-- 
Given:
- m: Initial production in the first quarter (in tons)
- x: Percentage increase in production each quarter (as a decimal)
- n: Production in the third quarter (in tons)

Prove that the production in the third quarter (n) is equal to the initial production (m) 
multiplied by (1 + x)^2.
-/
theorem fertilizer_production (m n : ℝ) (x : ℝ) (h_positive : 0 < x) : 
  m * (1 + x)^2 = n → True :=
by
  sorry

end NUMINAMATH_CALUDE_fertilizer_production_l2651_265166


namespace NUMINAMATH_CALUDE_tan_product_ninth_pi_l2651_265148

theorem tan_product_ninth_pi : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_ninth_pi_l2651_265148


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2651_265139

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  M > 0 ∧
  M % 4 = 3 ∧
  M % 5 = 4 ∧
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  ∀ n : ℕ, n > 0 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → M ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2651_265139


namespace NUMINAMATH_CALUDE_four_point_circle_theorem_l2651_265184

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define what it means for three points to be collinear
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

-- Define what it means for a point to be on a circle
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define what it means for a point to be inside a circle
def insideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

-- The main theorem
theorem four_point_circle_theorem (A B C D : Point) 
  (h : ¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D) :
  ∃ (c : Circle), 
    (onCircle A c ∧ onCircle B c ∧ onCircle C c ∧ (onCircle D c ∨ insideCircle D c)) ∨
    (onCircle A c ∧ onCircle B c ∧ onCircle D c ∧ (onCircle C c ∨ insideCircle C c)) ∨
    (onCircle A c ∧ onCircle C c ∧ onCircle D c ∧ (onCircle B c ∨ insideCircle B c)) ∨
    (onCircle B c ∧ onCircle C c ∧ onCircle D c ∧ (onCircle A c ∨ insideCircle A c)) :=
  sorry

end NUMINAMATH_CALUDE_four_point_circle_theorem_l2651_265184


namespace NUMINAMATH_CALUDE_absolute_value_sum_l2651_265136

theorem absolute_value_sum (y q : ℝ) (h1 : |y - 5| = q) (h2 : y > 5) : y + q = 2*q + 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l2651_265136


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2651_265103

theorem empty_solution_set_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x + 1 ≥ 0) ↔ m ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2651_265103


namespace NUMINAMATH_CALUDE_santiagos_number_l2651_265140

theorem santiagos_number (amelia santiago : ℂ) : 
  amelia * santiago = 20 + 15 * Complex.I ∧ 
  amelia = 4 - 5 * Complex.I →
  santiago = (5 : ℚ) / 41 + (160 : ℚ) / 41 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_santiagos_number_l2651_265140


namespace NUMINAMATH_CALUDE_unique_pronunciations_in_C_l2651_265100

/-- Represents a Chinese character with its pronunciation --/
structure ChineseChar :=
  (char : String)
  (pronunciation : String)

/-- Represents a group of words with underlined characters --/
structure WordGroup :=
  (name : String)
  (underlinedChars : List ChineseChar)

/-- Check if all pronunciations in a list are unique --/
def allUniquePronunciations (chars : List ChineseChar) : Prop :=
  ∀ i j, i ≠ j → (chars.get i).pronunciation ≠ (chars.get j).pronunciation

/-- The four word groups from the problem --/
def groupA : WordGroup := sorry
def groupB : WordGroup := sorry
def groupC : WordGroup := sorry
def groupD : WordGroup := sorry

/-- The main theorem to prove --/
theorem unique_pronunciations_in_C :
  allUniquePronunciations groupC.underlinedChars ∧
  ¬allUniquePronunciations groupA.underlinedChars ∧
  ¬allUniquePronunciations groupB.underlinedChars ∧
  ¬allUniquePronunciations groupD.underlinedChars :=
sorry

end NUMINAMATH_CALUDE_unique_pronunciations_in_C_l2651_265100


namespace NUMINAMATH_CALUDE_service_cost_equations_global_connect_more_cost_effective_l2651_265119

/-- Represents the cost of a mobile communication service based on monthly fee and per-minute rate -/
def service_cost (monthly_fee : ℝ) (per_minute_rate : ℝ) (minutes : ℝ) : ℝ :=
  monthly_fee + per_minute_rate * minutes

/-- Theorem stating the cost equations for Global Connect and Quick Connect services -/
theorem service_cost_equations 
  (x : ℝ) 
  (y1 : ℝ) 
  (y2 : ℝ) : 
  y1 = service_cost 50 0.4 x ∧ 
  y2 = service_cost 0 0.6 x :=
sorry

/-- Theorem stating that Global Connect is more cost-effective for 300 minutes of calls -/
theorem global_connect_more_cost_effective : 
  service_cost 50 0.4 300 < service_cost 0 0.6 300 :=
sorry

end NUMINAMATH_CALUDE_service_cost_equations_global_connect_more_cost_effective_l2651_265119


namespace NUMINAMATH_CALUDE_parallel_vectors_l2651_265190

/-- Given vectors a and b, if a is parallel to (a + b), then the second component of b is -3. -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : ∃ (k : ℝ), a = k • (a + b)) : 
  a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 3 → b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2651_265190


namespace NUMINAMATH_CALUDE_geometric_number_difference_l2651_265141

/-- A function that checks if a 3-digit number has distinct digits forming a geometric sequence --/
def is_geometric_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b * b = a * c

/-- The largest 3-digit number with distinct digits forming a geometric sequence --/
def largest_geometric_number : ℕ := 964

/-- The smallest 3-digit number with distinct digits forming a geometric sequence --/
def smallest_geometric_number : ℕ := 124

theorem geometric_number_difference :
  is_geometric_number largest_geometric_number ∧
  is_geometric_number smallest_geometric_number ∧
  (∀ n : ℕ, is_geometric_number n → 
    smallest_geometric_number ≤ n ∧ n ≤ largest_geometric_number) ∧
  largest_geometric_number - smallest_geometric_number = 840 := by
  sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l2651_265141


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2651_265147

theorem smallest_integer_with_remainders : 
  ∃ x : ℕ, 
    (x > 0) ∧ 
    (x % 5 = 4) ∧ 
    (x % 6 = 5) ∧ 
    (x % 7 = 6) ∧ 
    (∀ y : ℕ, y > 0 → y % 5 = 4 → y % 6 = 5 → y % 7 = 6 → x ≤ y) ∧
    x = 209 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2651_265147


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_largest_inscribed_circle_radius_proof_l2651_265110

/-- The radius of the largest inscribed circle in a square with side length 15,
    outside two congruent equilateral triangles sharing one side and each having
    one vertex on a vertex of the square. -/
theorem largest_inscribed_circle_radius : ℝ :=
  let square_side : ℝ := 15
  let triangle_side : ℝ := (square_side * Real.sqrt 6 - square_side * Real.sqrt 2) / 2
  let circle_radius : ℝ := square_side / 2 - (square_side * Real.sqrt 6 - square_side * Real.sqrt 2) / 8
  circle_radius

/-- Proof that the radius of the largest inscribed circle is correct. -/
theorem largest_inscribed_circle_radius_proof :
  largest_inscribed_circle_radius = 7.5 - (15 * Real.sqrt 6 - 15 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_largest_inscribed_circle_radius_proof_l2651_265110


namespace NUMINAMATH_CALUDE_x_coordinate_of_R_is_one_l2651_265102

/-- The curve on which point R lies -/
def curve (x y : ℝ) : Prop := y = -2 * x^2 + 5 * x - 2

/-- Predicate to check if OMRN is a square -/
def is_square (O M R N : ℝ × ℝ) : Prop := sorry

/-- Theorem stating that the x-coordinate of R is 1 -/
theorem x_coordinate_of_R_is_one 
  (R : ℝ × ℝ) 
  (h1 : curve R.1 R.2)
  (h2 : is_square (0, 0) (R.1, 0) R (0, R.2)) : 
  R.1 = 1 := by sorry

end NUMINAMATH_CALUDE_x_coordinate_of_R_is_one_l2651_265102


namespace NUMINAMATH_CALUDE_thanksgiving_turkey_cost_johns_thanksgiving_cost_l2651_265167

/-- Calculates the total cost of John's Thanksgiving turkey surprise for his employees. -/
theorem thanksgiving_turkey_cost 
  (num_employees : ℕ) 
  (turkey_cost : ℝ) 
  (discount_rate : ℝ) 
  (discount_threshold : ℕ) 
  (delivery_flat_fee : ℝ) 
  (delivery_per_turkey : ℝ) 
  (sales_tax_rate : ℝ) : ℝ :=
  let discounted_turkey_cost := 
    if num_employees > discount_threshold
    then num_employees * turkey_cost * (1 - discount_rate)
    else num_employees * turkey_cost
  let delivery_cost := delivery_flat_fee + num_employees * delivery_per_turkey
  let total_before_tax := discounted_turkey_cost + delivery_cost
  let total_cost := total_before_tax * (1 + sales_tax_rate)
  total_cost

/-- The total cost for John's Thanksgiving surprise is $2,188.35. -/
theorem johns_thanksgiving_cost :
  thanksgiving_turkey_cost 85 25 0.15 50 50 2 0.08 = 2188.35 := by
  sorry

end NUMINAMATH_CALUDE_thanksgiving_turkey_cost_johns_thanksgiving_cost_l2651_265167


namespace NUMINAMATH_CALUDE_jim_initial_tree_rows_l2651_265131

/-- Proves that Jim started with 2 rows of trees given the problem conditions -/
theorem jim_initial_tree_rows : ∀ (initial_rows : ℕ), 
  (∀ (row : ℕ), row > 0 → row ≤ initial_rows + 5 → 4 * row ≤ 56) ∧
  (2 * (4 * (initial_rows + 5)) = 56) →
  initial_rows = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jim_initial_tree_rows_l2651_265131


namespace NUMINAMATH_CALUDE_chocolate_boxes_l2651_265196

theorem chocolate_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 500) (h2 : total_pieces = 3000) :
  total_pieces / pieces_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_l2651_265196


namespace NUMINAMATH_CALUDE_ellipse_condition_l2651_265125

/-- 
A non-degenerate ellipse is represented by the equation 
3x^2 + 9y^2 - 12x + 27y = b if and only if b > -129/4
-/
theorem ellipse_condition (b : ℝ) : 
  (∃ (x y : ℝ), 3*x^2 + 9*y^2 - 12*x + 27*y = b ∧ 
    ∀ (x' y' : ℝ), 3*x'^2 + 9*y'^2 - 12*x' + 27*y' = b → (x', y') ≠ (x, y)) ↔ 
  b > -129/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2651_265125


namespace NUMINAMATH_CALUDE_equation_solutions_l2651_265179

theorem equation_solutions :
  let y₁ : ℝ := (3 + Real.sqrt 15) / 2
  let y₂ : ℝ := (3 - Real.sqrt 15) / 2
  (3 - y₁)^2 + y₁^2 = 12 ∧ (3 - y₂)^2 + y₂^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2651_265179


namespace NUMINAMATH_CALUDE_product_of_roots_l2651_265138

theorem product_of_roots (x : ℝ) : 
  (∃ a b c : ℝ, a * b * c = -9 ∧ 
   ∀ x, 4 * x^3 - 2 * x^2 - 25 * x + 36 = 0 ↔ (x = a ∨ x = b ∨ x = c)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2651_265138


namespace NUMINAMATH_CALUDE_banana_bread_flour_calculation_l2651_265124

/-- Given the recipe for banana bread, calculate the number of cups of flour needed. -/
theorem banana_bread_flour_calculation 
  (flour_per_mush : ℚ)  -- Cups of flour per cup of mush
  (bananas_per_mush : ℚ)  -- Number of bananas per cup of mush
  (total_bananas : ℚ)  -- Total number of bananas used
  (h1 : flour_per_mush = 3)  -- 3 cups of flour per cup of mush
  (h2 : bananas_per_mush = 4)  -- 4 bananas make one cup of mush
  (h3 : total_bananas = 20)  -- Hannah uses 20 bananas
  : (total_bananas / bananas_per_mush) * flour_per_mush = 15 := by
  sorry

#check banana_bread_flour_calculation

end NUMINAMATH_CALUDE_banana_bread_flour_calculation_l2651_265124


namespace NUMINAMATH_CALUDE_factor_sum_18_with_2_l2651_265109

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem factor_sum_18_with_2 (x : ℕ) 
  (h1 : x > 0) 
  (h2 : sum_of_factors x = 18) 
  (h3 : 2 ∣ x) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_factor_sum_18_with_2_l2651_265109


namespace NUMINAMATH_CALUDE_total_cost_with_tax_l2651_265175

def sandwich_price : ℚ := 4
def soda_price : ℚ := 3
def tax_rate : ℚ := 0.1
def sandwich_quantity : ℕ := 7
def soda_quantity : ℕ := 6

theorem total_cost_with_tax :
  let subtotal := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  let tax := subtotal * tax_rate
  let total := subtotal + tax
  total = 50.6 := by sorry

end NUMINAMATH_CALUDE_total_cost_with_tax_l2651_265175


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2651_265104

-- Problem 1
theorem problem_1 : (1/2)⁻¹ - 2 * Real.tan (45 * π / 180) + |1 - Real.sqrt 2| = Real.sqrt 2 - 1 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = Real.sqrt 3 + 2) : 
  (a / (a^2 - 4) + 1 / (2 - a)) / ((2*a + 4) / (a^2 + 4*a + 4)) = -Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2651_265104


namespace NUMINAMATH_CALUDE_remainder_seven_divisors_of_sixtyone_l2651_265130

theorem remainder_seven_divisors_of_sixtyone : 
  (Finset.filter (fun n : ℕ => n > 7 ∧ 61 % n = 7) (Finset.range 62)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_divisors_of_sixtyone_l2651_265130


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_fourth_root_l2651_265176

theorem arithmetic_mean_geq_geometric_mean_fourth_root
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) / 4 ≥ (a * b * c * d) ^ (1/4) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_fourth_root_l2651_265176


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l2651_265157

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Kevin's hopping problem -/
theorem kevin_kangaroo_hops :
  geometricSum (1/2) (1/2) 6 = 63/64 := by
  sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l2651_265157


namespace NUMINAMATH_CALUDE_sum_of_first_15_natural_numbers_mod_11_l2651_265106

theorem sum_of_first_15_natural_numbers_mod_11 :
  (List.range 16).sum % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_15_natural_numbers_mod_11_l2651_265106


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2651_265194

-- Define the function types
variable (f g : ℝ → ℝ)

-- Define the functional equation condition
def functional_equation : Prop :=
  ∀ x y : ℝ, f (x - f y) = x * f y - y * f x + g x

-- State the theorem
theorem functional_equation_solutions :
  functional_equation f g →
  ((∀ x, f x = 0 ∧ g x = 0) ∨ (∀ x, f x = x ∧ g x = 0)) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2651_265194


namespace NUMINAMATH_CALUDE_incorrect_locus_proof_l2651_265193

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 25

/-- The set of points satisfying the circle equation -/
def circle_set : Set (ℝ × ℝ) := {p | circle_equation p.1 p.2}

/-- The statement to be proven false -/
def incorrect_statement : Prop :=
  (∀ p : ℝ × ℝ, ¬(circle_equation p.1 p.2) → p ∉ circle_set) →
  (circle_set = {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = 5^2})

theorem incorrect_locus_proof : ¬incorrect_statement := by
  sorry

end NUMINAMATH_CALUDE_incorrect_locus_proof_l2651_265193


namespace NUMINAMATH_CALUDE_divisibility_problem_l2651_265105

theorem divisibility_problem :
  (∃ n : ℕ, n = 9 ∧ (1100 + n) % 53 = 0 ∧ ∀ k : ℕ, k < n → (1100 + k) % 53 ≠ 0) ∧
  (∃ m : ℕ, m = 0 ∧ (1100 - m) % 71 = 0 ∧ ∀ k : ℕ, k < m → (1100 - k) % 71 ≠ 0) ∧
  (∃ X : ℤ, X = 534 ∧ (1100 + X) % 19 = 0 ∧ (1100 + X) % 43 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2651_265105


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2651_265111

/-- A pyramid with a square base and right-angled isosceles triangular lateral faces -/
structure Pyramid :=
  (base_side : ℝ)

/-- A cube inscribed in a pyramid -/
structure InscribedCube :=
  (side_length : ℝ)

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.side_length ^ 3

/-- Predicate for a cube being properly inscribed in the pyramid -/
def is_properly_inscribed (p : Pyramid) (c : InscribedCube) : Prop :=
  c.side_length > 0 ∧ c.side_length < p.base_side

theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube) 
  (h1 : p.base_side = 2)
  (h2 : is_properly_inscribed p c) :
  cube_volume c = 10 + 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2651_265111


namespace NUMINAMATH_CALUDE_people_in_room_l2651_265189

/-- Given a room with chairs and people, prove the total number of people -/
theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs ∧ 
  chairs - (4 : ℚ) / 5 * chairs = 8 →
  people = 54 := by sorry

end NUMINAMATH_CALUDE_people_in_room_l2651_265189


namespace NUMINAMATH_CALUDE_fifth_match_goals_l2651_265169

/-- Represents the goal-scoring statistics of a football player over 5 matches -/
structure FootballStats where
  total_goals : ℕ
  avg_increase : ℚ

/-- Theorem stating that under given conditions, the player scored 3 goals in the fifth match -/
theorem fifth_match_goals (stats : FootballStats) 
  (h1 : stats.total_goals = 11)
  (h2 : stats.avg_increase = 1/5) : 
  (stats.total_goals : ℚ) - 4 * ((stats.total_goals : ℚ) / 5 - stats.avg_increase) = 3 := by
  sorry

#check fifth_match_goals

end NUMINAMATH_CALUDE_fifth_match_goals_l2651_265169


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2651_265155

-- Define the arithmetic sequence
def arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

-- Define the increasing property
def increasing_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  arithmetic_sequence b 2 →
  increasing_sequence b →
  b 4 * b 5 = 15 →
  b 2 * b 7 = -9 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2651_265155


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2651_265163

/-- A quadratic function satisfying specific conditions -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The function g defined in terms of f -/
def g (a b c m : ℝ) (x : ℝ) : ℝ := f a b c x - 2 * m * x + 2

/-- The theorem stating the properties of f and g -/
theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c 0 = 0)
  (h2 : ∀ x, f a b c (x + 2) - f a b c x = 4 * x) :
  (∀ x, f a b c x = x^2 - 2 * x) ∧
  (∀ m,
    (m ≤ 0 → ∀ x ≥ 1, g a b c m x ≥ 1 - 2 * m) ∧
    (m > 0 → ∀ x ≥ 1, g a b c m x ≥ -m^2 - 2 * m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2651_265163


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2651_265197

theorem cyclic_inequality (x₁ x₂ x₃ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  x₂^2 / x₁ + x₃^2 / x₂ + x₁^2 / x₃ ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2651_265197


namespace NUMINAMATH_CALUDE_optimal_cup_purchase_l2651_265128

/-- Represents the profit optimization problem for cup sales --/
structure CupSalesProblem where
  costA : ℕ
  priceA : ℕ
  costB : ℕ
  priceB : ℕ
  totalCups : ℕ
  budget : ℕ

/-- Calculates the profit for a given number of cup A --/
def profit (p : CupSalesProblem) (x : ℕ) : ℤ :=
  (p.priceA - p.costA) * x + (p.priceB - p.costB) * (p.totalCups - x)

/-- Checks if the purchase is within budget --/
def withinBudget (p : CupSalesProblem) (x : ℕ) : Prop :=
  p.costA * x + p.costB * (p.totalCups - x) ≤ p.budget

/-- Theorem stating the optimal solution and maximum profit --/
theorem optimal_cup_purchase (p : CupSalesProblem) 
  (h1 : p.costA = 100)
  (h2 : p.priceA = 150)
  (h3 : p.costB = 85)
  (h4 : p.priceB = 120)
  (h5 : p.totalCups = 160)
  (h6 : p.budget = 15000) :
  ∃ (x : ℕ), x = 93 ∧ 
             withinBudget p x ∧ 
             profit p x = 6995 ∧ 
             ∀ (y : ℕ), withinBudget p y → profit p y ≤ profit p x :=
by sorry

end NUMINAMATH_CALUDE_optimal_cup_purchase_l2651_265128


namespace NUMINAMATH_CALUDE_square_difference_l2651_265146

theorem square_difference : (625 : ℤ)^2 - (375 : ℤ)^2 = 250000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2651_265146


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_1_to_21_l2651_265195

/-- The sum of an arithmetic series with first term a, last term l, and n terms -/
def arithmetic_series_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The number of terms in an arithmetic series with first term a, last term l, and common difference d -/
def arithmetic_series_length (a l d : ℕ) : ℕ := (l - a) / d + 1

theorem arithmetic_series_sum_1_to_21 : 
  let a := 1  -- first term
  let l := 21 -- last term
  let d := 2  -- common difference
  let n := arithmetic_series_length a l d
  arithmetic_series_sum a l n = 121 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_1_to_21_l2651_265195


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l2651_265112

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π/2 + α) * Real.sin (π + α) * Real.tan (3*π + α)) /
  (Real.cos (3*π/2 + α) * Real.sin (-α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l2651_265112


namespace NUMINAMATH_CALUDE_all_statements_equivalent_l2651_265170

-- Define the propositions
variable (P Q : Prop)

-- Define the equivalence of all statements
theorem all_statements_equivalent :
  (P ↔ Q) ↔ (P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end NUMINAMATH_CALUDE_all_statements_equivalent_l2651_265170


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l2651_265181

theorem triangle_rectangle_ratio : 
  ∀ (t w l : ℝ),
  t > 0 → w > 0 → l > 0 →
  3 * t = 24 →           -- Perimeter of equilateral triangle
  2 * (w + l) = 24 →     -- Perimeter of rectangle
  l = 2 * w →            -- Length is twice the width
  t / w = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l2651_265181


namespace NUMINAMATH_CALUDE_abc_sum_equals_36_l2651_265158

theorem abc_sum_equals_36 (a b c : ℕ+) 
  (h : (4 : ℕ)^(a.val) * (5 : ℕ)^(b.val) * (6 : ℕ)^(c.val) = (8 : ℕ)^8 * (9 : ℕ)^9 * (10 : ℕ)^10) : 
  a.val + b.val + c.val = 36 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_equals_36_l2651_265158


namespace NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l2651_265108

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rectangle in a plane -/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment -/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The maximum number of intersection points between a circle and a rectangle -/
def maxIntersectionPoints (c : Circle) (r : Rectangle) : ℕ :=
  (intersectionPointsCircleLine c (r.corners 0) (r.corners 1)) +
  (intersectionPointsCircleLine c (r.corners 1) (r.corners 2)) +
  (intersectionPointsCircleLine c (r.corners 2) (r.corners 3)) +
  (intersectionPointsCircleLine c (r.corners 3) (r.corners 0))

/-- Theorem: The maximum number of intersection points between a circle and a rectangle is 8 -/
theorem max_intersection_points_circle_rectangle :
  ∀ (c : Circle) (r : Rectangle), maxIntersectionPoints c r ≤ 8 ∧
  ∃ (c : Circle) (r : Rectangle), maxIntersectionPoints c r = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l2651_265108


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_product_l2651_265187

theorem repeating_decimal_fraction_product : ∃ (n d : ℕ), 
  (n ≠ 0 ∧ d ≠ 0) ∧ 
  (∀ (k : ℕ), (0.027 + 0.027 / (1000 ^ k - 1) : ℚ) = n / d) ∧
  (∀ (n' d' : ℕ), n' ≠ 0 ∧ d' ≠ 0 → (∀ (k : ℕ), (0.027 + 0.027 / (1000 ^ k - 1) : ℚ) = n' / d') → n ≤ n' ∧ d ≤ d') ∧
  n * d = 37 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_product_l2651_265187


namespace NUMINAMATH_CALUDE_solution_range_l2651_265135

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem solution_range (a b c : ℝ) :
  f a b c 1.1 < 3 ∧ 
  f a b c 1.2 < 3 ∧ 
  f a b c 1.3 < 3 ∧ 
  f a b c 1.4 > 3 →
  ∃ x, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2651_265135


namespace NUMINAMATH_CALUDE_investment_schemes_count_l2651_265144

/-- The number of projects to be invested -/
def num_projects : ℕ := 4

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The maximum number of projects that can be invested in a single city -/
def max_projects_per_city : ℕ := 2

/-- A function that calculates the number of ways to distribute projects among cities -/
def investment_schemes (projects : ℕ) (cities : ℕ) (max_per_city : ℕ) : ℕ := sorry

/-- Theorem stating that the number of investment schemes is 240 -/
theorem investment_schemes_count : 
  investment_schemes num_projects num_cities max_projects_per_city = 240 := by sorry

end NUMINAMATH_CALUDE_investment_schemes_count_l2651_265144


namespace NUMINAMATH_CALUDE_watermelon_weight_calculation_l2651_265168

/-- The weight of a single watermelon in pounds -/
def watermelon_weight : ℝ := 23

/-- The price per pound of watermelon in dollars -/
def price_per_pound : ℝ := 2

/-- The number of watermelons sold -/
def num_watermelons : ℕ := 18

/-- The total revenue from selling the watermelons in dollars -/
def total_revenue : ℝ := 828

theorem watermelon_weight_calculation :
  watermelon_weight = total_revenue / (price_per_pound * num_watermelons) :=
by sorry

end NUMINAMATH_CALUDE_watermelon_weight_calculation_l2651_265168


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2651_265160

-- Define the trapezoid ABCD
structure Trapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (BC : ℝ)

-- Define the properties of the trapezoid
def is_valid_trapezoid (t : Trapezoid) : Prop :=
  t.AB = 10 ∧ 
  t.CD = 2 * t.AB ∧ 
  t.AD = t.BC ∧ 
  t.AB + t.BC + t.CD + t.AD = 42

-- Theorem statement
theorem trapezoid_side_length 
  (t : Trapezoid) 
  (h : is_valid_trapezoid t) : 
  t.AD = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2651_265160


namespace NUMINAMATH_CALUDE_book_cost_l2651_265134

theorem book_cost (book_cost bookmark_cost : ℚ) 
  (total_cost : book_cost + bookmark_cost = (7/2 : ℚ))
  (price_difference : book_cost = bookmark_cost + 3) : 
  book_cost = (13/4 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_book_cost_l2651_265134


namespace NUMINAMATH_CALUDE_bus_problem_l2651_265173

/-- Calculates the number of students remaining on a bus after a given number of stops,
    where one-third of the students get off at each stop. -/
def studentsRemaining (initialStudents : ℚ) (stops : ℕ) : ℚ :=
  initialStudents * (2/3)^stops

/-- Proves that if a bus starts with 60 students and loses one-third of its passengers
    at each of four stops, the number of students remaining after the fourth stop is 320/27. -/
theorem bus_problem : studentsRemaining 60 4 = 320/27 := by
  sorry

#eval studentsRemaining 60 4

end NUMINAMATH_CALUDE_bus_problem_l2651_265173


namespace NUMINAMATH_CALUDE_work_completion_time_l2651_265151

/-- The number of days it takes A to complete the work alone -/
def days_A : ℝ := 6

/-- The total payment for the work -/
def total_payment : ℝ := 4000

/-- The number of days it takes A, B, and C to complete the work together -/
def days_ABC : ℝ := 3

/-- The payment to C -/
def payment_C : ℝ := 500.0000000000002

/-- The number of days it takes B to complete the work alone -/
def days_B : ℝ := 8

theorem work_completion_time :
  (1 / days_A + 1 / days_B + payment_C / total_payment * (1 / days_ABC) = 1 / days_ABC) ∧
  days_B = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2651_265151


namespace NUMINAMATH_CALUDE_base_85_subtraction_divisibility_l2651_265132

theorem base_85_subtraction_divisibility (b : ℤ) : 
  (0 ≤ b ∧ b ≤ 20) → 
  (∃ k : ℤ, 346841047 * 85^8 + 4 * 85^7 + 1 * 85^5 + 4 * 85^4 + 8 * 85^3 + 6 * 85^2 + 4 * 85 + 3 - b = 17 * k) → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_base_85_subtraction_divisibility_l2651_265132


namespace NUMINAMATH_CALUDE_number_of_dogs_l2651_265116

/-- The number of dogs at a farm, given the number of fish, cats, and total pets. -/
theorem number_of_dogs (fish : ℕ) (cats : ℕ) (total_pets : ℕ) : 
  fish = 72 → cats = 34 → total_pets = 149 → total_pets = fish + cats + 43 :=
by sorry

end NUMINAMATH_CALUDE_number_of_dogs_l2651_265116


namespace NUMINAMATH_CALUDE_uncovered_cells_less_than_mn_div_4_uncovered_cells_less_than_mn_div_5_l2651_265199

/-- Represents a rectangular board with dominoes -/
structure DominoBoard where
  m : ℕ  -- width of the board
  n : ℕ  -- height of the board
  uncovered_cells : ℕ  -- number of uncovered cells

/-- The number of uncovered cells is less than mn/4 -/
theorem uncovered_cells_less_than_mn_div_4 (board : DominoBoard) :
  board.uncovered_cells < (board.m * board.n) / 4 := by
  sorry

/-- The number of uncovered cells is less than mn/5 -/
theorem uncovered_cells_less_than_mn_div_5 (board : DominoBoard) :
  board.uncovered_cells < (board.m * board.n) / 5 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_cells_less_than_mn_div_4_uncovered_cells_less_than_mn_div_5_l2651_265199


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_is_eight_l2651_265120

/-- Represents a key on the keychain -/
inductive Key
| House
| Car
| Work
| Garage
| Other

/-- Represents a pair of adjacent keys -/
structure KeyPair :=
  (first : Key)
  (second : Key)

/-- Represents an arrangement of keys on the keychain -/
structure KeyArrangement :=
  (pair1 : KeyPair)
  (pair2 : KeyPair)
  (single : Key)

/-- Checks if two KeyArrangements are considered identical (up to rotation and reflection) -/
def are_identical (a b : KeyArrangement) : Prop := sorry

/-- The set of all valid key arrangements -/
def valid_arrangements : Set KeyArrangement :=
  {arr | (arr.pair1.first = Key.House ∧ arr.pair1.second = Key.Car) ∨
         (arr.pair1.first = Key.Car ∧ arr.pair1.second = Key.House) ∧
         (arr.pair2.first = Key.Work ∧ arr.pair2.second = Key.Garage) ∨
         (arr.pair2.first = Key.Garage ∧ arr.pair2.second = Key.Work) ∧
         arr.single = Key.Other}

/-- The number of distinct arrangements -/
def distinct_arrangement_count : ℕ := sorry

theorem distinct_arrangements_count_is_eight :
  distinct_arrangement_count = 8 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_is_eight_l2651_265120


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_three_l2651_265161

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (m^2 - 2m - 3) + (m + 1)i is a purely imaginary number, m = 3. -/
theorem purely_imaginary_implies_m_eq_three (m : ℝ) :
  is_purely_imaginary ((m^2 - 2*m - 3 : ℝ) + (m + 1)*I) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_three_l2651_265161


namespace NUMINAMATH_CALUDE_two_week_training_hours_l2651_265172

/-- Calculates the total training hours for two weeks given daily maximum hours -/
def totalTrainingHours (week1MaxHours : ℕ) (week2MaxHours : ℕ) : ℕ :=
  7 * week1MaxHours + 7 * week2MaxHours

/-- Proves that training for 2 hours max per day in week 1 and 3 hours max per day in week 2 results in 35 total hours -/
theorem two_week_training_hours : totalTrainingHours 2 3 = 35 := by
  sorry

#eval totalTrainingHours 2 3

end NUMINAMATH_CALUDE_two_week_training_hours_l2651_265172


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l2651_265159

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  (∀ x : ℝ, x^2 + p*x + 12 = 0 ↔ x = r₁ ∨ x = r₂) →
  |r₁ + r₂| > 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l2651_265159


namespace NUMINAMATH_CALUDE_real_part_of_z_l2651_265182

theorem real_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2) : z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2651_265182


namespace NUMINAMATH_CALUDE_maria_number_puzzle_l2651_265126

theorem maria_number_puzzle (x : ℝ) : 
  (((x + 3) * 2 - 4) / 3 = 10) → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_maria_number_puzzle_l2651_265126


namespace NUMINAMATH_CALUDE_equation_equivalence_l2651_265164

theorem equation_equivalence :
  ∀ (x y : ℝ), (2 * x - y = 3) ↔ (y = 2 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2651_265164


namespace NUMINAMATH_CALUDE_smallest_c_value_l2651_265174

theorem smallest_c_value (c : ℝ) (h : (3 * c + 4) * (c - 2) = 7 * c + 6) :
  c ≥ (9 - Real.sqrt 249) / 6 ∧ ∃ (c₀ : ℝ), (3 * c₀ + 4) * (c₀ - 2) = 7 * c₀ + 6 ∧ c₀ = (9 - Real.sqrt 249) / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2651_265174


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l2651_265121

-- Define the parabola function
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the theorem
theorem parabola_y_relationship (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : parabola (-2) c = y₁)
  (h2 : parabola 0 c = y₂)
  (h3 : parabola (5/3) c = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_parabola_y_relationship_l2651_265121


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_19_mod_26_l2651_265153

theorem largest_five_digit_congruent_to_19_mod_26 : ∃ (n : ℕ), n = 99989 ∧ 
  (∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 26 = 19 → m ≤ n) ∧ 
  10000 ≤ n ∧ n ≤ 99999 ∧ n % 26 = 19 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_19_mod_26_l2651_265153
