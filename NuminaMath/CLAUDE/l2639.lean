import Mathlib

namespace NUMINAMATH_CALUDE_distance_calculation_l2639_263929

/-- Proves that the distance run by A and B is 2250 meters given their running times and the difference in distance covered. -/
theorem distance_calculation (D : ℝ) 
  (h1 : D / 90 * 180 = D + 2250) : D = 2250 := by
  sorry

#check distance_calculation

end NUMINAMATH_CALUDE_distance_calculation_l2639_263929


namespace NUMINAMATH_CALUDE_cereal_box_initial_price_l2639_263989

/-- The initial price of a cereal box given a price reduction and total purchase amount -/
theorem cereal_box_initial_price 
  (price_reduction : ℝ) 
  (num_boxes : ℕ) 
  (total_paid : ℝ) 
  (h1 : price_reduction = 24)
  (h2 : num_boxes = 20)
  (h3 : total_paid = 1600) : 
  ∃ (initial_price : ℝ), 
    num_boxes * (initial_price - price_reduction) = total_paid ∧ 
    initial_price = 104 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_initial_price_l2639_263989


namespace NUMINAMATH_CALUDE_rice_restocking_solution_l2639_263995

def rice_restocking_problem (initial_stock sold final_stock : ℕ) : ℕ :=
  final_stock - (initial_stock - sold)

theorem rice_restocking_solution :
  rice_restocking_problem 55 23 164 = 132 := by
  sorry

end NUMINAMATH_CALUDE_rice_restocking_solution_l2639_263995


namespace NUMINAMATH_CALUDE_lune_area_zero_l2639_263934

/-- The area of a lune formed by a semicircle of diameter 2 sitting on top of a semicircle of diameter 4 is 0 -/
theorem lune_area_zero (π : ℝ) (h : π > 0) : 
  let small_semicircle_area := (1/2) * π * (2/2)^2
  let large_semicircle_sector_area := (1/8) * π * (4/2)^2
  let lune_area := small_semicircle_area - large_semicircle_sector_area
  lune_area = 0 := by sorry

end NUMINAMATH_CALUDE_lune_area_zero_l2639_263934


namespace NUMINAMATH_CALUDE_probability_yellow_second_is_67_135_l2639_263947

/-- Represents the contents of a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Bag X contents -/
def bagX : Bag := { white := 4, black := 5 }

/-- Bag Y contents -/
def bagY : Bag := { yellow := 7, blue := 3 }

/-- Bag Z contents -/
def bagZ : Bag := { yellow := 3, blue := 6 }

/-- Probability of drawing a yellow marble as the second marble -/
def probabilityYellowSecond : ℚ :=
  (bagX.white * bagY.yellow) / (bagX.total * bagY.total) +
  (bagX.black * bagZ.yellow) / (bagX.total * bagZ.total)

theorem probability_yellow_second_is_67_135 : probabilityYellowSecond = 67 / 135 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_second_is_67_135_l2639_263947


namespace NUMINAMATH_CALUDE_minimal_fraction_sum_l2639_263991

theorem minimal_fraction_sum (a b : ℕ+) (h : (45 : ℚ) / 11 < (a : ℚ) / b ∧ (a : ℚ) / b < 5 / 11) :
  (∀ c d : ℕ+, (45 : ℚ) / 11 < (c : ℚ) / d ∧ (c : ℚ) / d < 5 / 11 → c + d ≥ a + b) →
  a = 3 ∧ b = 7 :=
sorry

end NUMINAMATH_CALUDE_minimal_fraction_sum_l2639_263991


namespace NUMINAMATH_CALUDE_convex_polygon_not_divisible_into_nonconvex_quadrilaterals_l2639_263986

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define convexity for a polygon
def IsConvex (P : Polygon) : Prop := sorry

-- Define a quadrilateral as a polygon with exactly 4 vertices
def Quadrilateral (Q : Polygon) : Prop := sorry

-- Define nonconvexity for a quadrilateral
def IsNonConvex (Q : Polygon) : Prop := Quadrilateral Q ∧ ¬IsConvex Q

-- Main theorem
theorem convex_polygon_not_divisible_into_nonconvex_quadrilaterals 
  (M : Polygon) (n : ℕ) (M_i : Fin n → Polygon) :
  IsConvex M →
  (∀ i, IsNonConvex (M_i i)) →
  M ≠ ⋃ i, M_i i :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_not_divisible_into_nonconvex_quadrilaterals_l2639_263986


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2639_263903

/-- Given a line and a circle, prove that if the line is tangent to the circle, then the constant a in the circle equation equals 2 + √5. -/
theorem line_tangent_to_circle (t θ : ℝ) (a : ℝ) (h_a : a > 0) :
  let line : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 1 - t ∧ p.2 = 2 * t
  let circle : ℝ × ℝ → Prop := λ p => ∃ θ, p.1 = Real.cos θ ∧ p.2 = Real.sin θ + a
  (∀ p, line p → ¬ circle p) ∧ (∃ p, line p ∧ (∀ ε > 0, ∃ q, circle q ∧ dist p q < ε)) →
  a = 2 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2639_263903


namespace NUMINAMATH_CALUDE_log_relation_difference_l2639_263976

theorem log_relation_difference (a b c d : ℤ) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) : 
  b - d = 93 := by
sorry

end NUMINAMATH_CALUDE_log_relation_difference_l2639_263976


namespace NUMINAMATH_CALUDE_solution_problem_l2639_263971

theorem solution_problem (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_problem_l2639_263971


namespace NUMINAMATH_CALUDE_power_of_64_l2639_263997

theorem power_of_64 : (64 : ℝ) ^ (5/6 : ℝ) = 32 :=
by
  have h : (64 : ℝ) = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l2639_263997


namespace NUMINAMATH_CALUDE_min_value_expression_l2639_263926

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 81 ∧
  (a^2 + 9*a*b + 9*b^2 + 3*c^2 = 81 ↔ a = 3 ∧ b = 1 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2639_263926


namespace NUMINAMATH_CALUDE_even_square_sum_perfect_square_l2639_263928

theorem even_square_sum_perfect_square (x y z : ℤ) 
  (h_even : Even x)
  (h_odd : Odd y)
  (h_sum : x^2 + y^2 = z^2) :
  4 ∣ x :=
sorry

end NUMINAMATH_CALUDE_even_square_sum_perfect_square_l2639_263928


namespace NUMINAMATH_CALUDE_solve_for_x_l2639_263988

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 20) (h2 : y = 5) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2639_263988


namespace NUMINAMATH_CALUDE_problem_solution_l2639_263931

theorem problem_solution : ∃ x : ℚ, (70 / 100) * x - (1 / 3) * x = 110 ∧ x = 300 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2639_263931


namespace NUMINAMATH_CALUDE_equation_solution_l2639_263952

theorem equation_solution : ∀ x : ℝ, x + 36 / (x - 3) = -9 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2639_263952


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2639_263945

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2639_263945


namespace NUMINAMATH_CALUDE_product_signs_l2639_263975

theorem product_signs (a b c d e : ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  (∃ (pos neg : Finset (ℝ × ℝ)), 
    pos.card = 5 ∧ 
    neg.card = 5 ∧ 
    (∀ p ∈ pos, p.1 + p.2 > 0) ∧
    (∀ p ∈ neg, p.1 + p.2 < 0) ∧
    pos ∩ neg = ∅ ∧
    pos ∪ neg = {(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)}) →
  (∃ (pos_prod neg_prod : Finset (ℝ × ℝ)),
    pos_prod.card = 4 ∧
    neg_prod.card = 6 ∧
    (∀ p ∈ pos_prod, p.1 * p.2 > 0) ∧
    (∀ p ∈ neg_prod, p.1 * p.2 < 0) ∧
    pos_prod ∩ neg_prod = ∅ ∧
    pos_prod ∪ neg_prod = {(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)}) :=
by sorry

end NUMINAMATH_CALUDE_product_signs_l2639_263975


namespace NUMINAMATH_CALUDE_crosswalk_height_l2639_263902

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ  -- Length of one side
  side2 : ℝ  -- Length of adjacent side
  base : ℝ   -- Length of base parallel to side1
  height1 : ℝ -- Height perpendicular to side1
  height2 : ℝ -- Height perpendicular to side2

/-- The area of a parallelogram can be calculated two ways -/
axiom area_equality (p : Parallelogram) : p.side1 * p.height1 = p.side2 * p.height2

/-- Theorem stating the height of the parallelogram perpendicular to the 80-foot side -/
theorem crosswalk_height (p : Parallelogram) 
    (h1 : p.side1 = 60)
    (h2 : p.side2 = 80)
    (h3 : p.base = 30)
    (h4 : p.height1 = 60) :
    p.height2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_height_l2639_263902


namespace NUMINAMATH_CALUDE_parabola_c_value_l2639_263973

/-- A parabola in the xy-plane with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = 4 →  -- vertex at (4, 1)
  p.x_coord 3 = -2 →  -- passes through (-2, 3)
  p.c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2639_263973


namespace NUMINAMATH_CALUDE_correct_num_friends_l2639_263927

/-- The number of friends Jeremie is going with to the amusement park. -/
def num_friends : ℕ := 3

/-- The cost of a ticket in dollars. -/
def ticket_cost : ℕ := 18

/-- The cost of a snack set in dollars. -/
def snack_cost : ℕ := 5

/-- The total cost for Jeremie and her friends in dollars. -/
def total_cost : ℕ := 92

/-- Theorem stating that the number of friends Jeremie is going with is correct. -/
theorem correct_num_friends :
  (num_friends + 1) * (ticket_cost + snack_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_num_friends_l2639_263927


namespace NUMINAMATH_CALUDE_complex_distance_theorem_l2639_263932

theorem complex_distance_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3)
  (h₂ : Complex.abs z₂ = 5)
  (h₃ : Complex.abs (z₁ + z₂) = 6) :
  Complex.abs (z₁ - z₂) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_distance_theorem_l2639_263932


namespace NUMINAMATH_CALUDE_time_after_2023_hours_l2639_263981

def clock_add (current_time : ℕ) (hours_passed : ℕ) : ℕ :=
  (current_time + hours_passed) % 12

theorem time_after_2023_hours :
  clock_add 7 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_after_2023_hours_l2639_263981


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l2639_263907

/-- Given an investment of 3000 units yielding an income of 210 units,
    prove that the investment rate is 7%. -/
theorem investment_rate_calculation (investment : ℝ) (income : ℝ) (rate : ℝ) :
  investment = 3000 →
  income = 210 →
  rate = income / investment * 100 →
  rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l2639_263907


namespace NUMINAMATH_CALUDE_rotation_90_clockwise_effect_l2639_263911

-- Define the shapes
inductive Shape
  | Pentagon
  | Ellipse
  | Rectangle

-- Define the positions on the circle
structure Position :=
  (angle : ℝ)

-- Define the configuration of shapes on the circle
structure Configuration :=
  (pentagon_pos : Position)
  (ellipse_pos : Position)
  (rectangle_pos : Position)

-- Define the rotation operation
def rotate_90_clockwise (config : Configuration) : Configuration :=
  { pentagon_pos := config.ellipse_pos,
    ellipse_pos := config.rectangle_pos,
    rectangle_pos := config.pentagon_pos }

-- Theorem statement
theorem rotation_90_clockwise_effect (initial_config : Configuration) :
  let final_config := rotate_90_clockwise initial_config
  (final_config.pentagon_pos = initial_config.ellipse_pos) ∧
  (final_config.ellipse_pos = initial_config.rectangle_pos) ∧
  (final_config.rectangle_pos = initial_config.pentagon_pos) :=
by
  sorry


end NUMINAMATH_CALUDE_rotation_90_clockwise_effect_l2639_263911


namespace NUMINAMATH_CALUDE_no_solution_iff_a_geq_bound_l2639_263943

theorem no_solution_iff_a_geq_bound (a : ℝ) :
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) ↔ a ≥ (Real.sqrt 3 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_geq_bound_l2639_263943


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2639_263974

/-- If point M (a+3, 2a-2) is on the y-axis, then its coordinates are (0, -8) -/
theorem point_on_y_axis (a : ℝ) : 
  (a + 3 = 0) → ((a + 3, 2*a - 2) : ℝ × ℝ) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2639_263974


namespace NUMINAMATH_CALUDE_f_extrema_and_intersection_range_l2639_263900

-- Define the function f(x) = x³ - 3x - 1
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- State the theorem
theorem f_extrema_and_intersection_range :
  -- f(x) has a maximum at x = -1
  (∀ x : ℝ, f (-1) ≥ f x) ∧
  -- f(x) has a minimum at x = 1
  (∀ x : ℝ, f 1 ≤ f x) ∧
  -- The range of m for which y = m intersects y = f(x) at three distinct points is (-3, 1)
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔ -3 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_and_intersection_range_l2639_263900


namespace NUMINAMATH_CALUDE_water_remaining_l2639_263904

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 4/3 → remaining = initial - used → remaining = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l2639_263904


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l2639_263909

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l2639_263909


namespace NUMINAMATH_CALUDE_total_students_l2639_263949

theorem total_students (n : ℕ) (male_students : ℕ) (selected_students : ℕ) (male_participants : ℕ) (female_participants : ℕ) : 
  male_students = 550 →
  selected_students = n / 10 →
  male_participants = female_participants + 10 →
  male_participants = male_students / 10 →
  selected_students = male_participants + female_participants →
  n = 1000 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2639_263949


namespace NUMINAMATH_CALUDE_annie_thyme_pots_l2639_263992

/-- The number of pots of thyme Annie planted -/
def thyme_pots : ℕ := sorry

/-- The total number of leaves -/
def total_leaves : ℕ := 354

/-- The number of pots of basil -/
def basil_pots : ℕ := 3

/-- The number of pots of rosemary -/
def rosemary_pots : ℕ := 9

/-- The number of leaves per basil plant -/
def leaves_per_basil : ℕ := 4

/-- The number of leaves per rosemary plant -/
def leaves_per_rosemary : ℕ := 18

/-- The number of leaves per thyme plant -/
def leaves_per_thyme : ℕ := 30

theorem annie_thyme_pots : 
  thyme_pots = 6 :=
by sorry

end NUMINAMATH_CALUDE_annie_thyme_pots_l2639_263992


namespace NUMINAMATH_CALUDE_cube_root_simplification_and_rationalization_l2639_263993

theorem cube_root_simplification_and_rationalization :
  let x := (Real.rpow 6 (1/3)) / (Real.rpow 7 (1/3))
  let y := (Real.rpow 8 (1/3)) / (Real.rpow 9 (1/3))
  let z := (Real.rpow 10 (1/3)) / (Real.rpow 11 (1/3))
  x * y * z = (Real.rpow 223948320 (1/3)) / 693 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_and_rationalization_l2639_263993


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l2639_263944

def sum_and_reciprocal_sum_equal (s : Finset ℝ) (n : ℕ) (v : ℝ) : Prop :=
  s.card = n ∧ (∀ x ∈ s, x > 0) ∧ (s.sum id = v) ∧ (s.sum (λ x => 1 / x) = v)

theorem max_value_x_plus_reciprocal 
  (s : Finset ℝ) (h : sum_and_reciprocal_sum_equal s 1001 1002) :
  ∀ x ∈ s, x + 1/x ≤ 4007/1002 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l2639_263944


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_not_necessarily_right_l2639_263951

/-- Triangle ABC with angles in the ratio 3:4:5 is not necessarily a right triangle -/
theorem triangle_with_angle_ratio_not_necessarily_right :
  ∀ (A B C : ℝ),
  (A + B + C = 180) →
  (A : ℝ) / 3 = (B : ℝ) / 4 →
  (B : ℝ) / 4 = (C : ℝ) / 5 →
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_not_necessarily_right_l2639_263951


namespace NUMINAMATH_CALUDE_john_average_change_l2639_263985

def john_scores : List ℝ := [84, 88, 95, 92]

theorem john_average_change : 
  let initial_average := (john_scores.take 3).sum / 3
  let new_average := john_scores.sum / 4
  new_average - initial_average = 0.75 := by sorry

end NUMINAMATH_CALUDE_john_average_change_l2639_263985


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2639_263908

/-- 
Given two functions f(x) = bx² + 5x + 2 and g(x) = -2x - 2,
prove that they intersect at exactly one point when b = 49/16.
-/
theorem quadratic_intersection (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 5 * x + 2 = -2 * x - 2) ↔ b = 49/16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2639_263908


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l2639_263913

/-- Given a flagpole and a building under similar shadow-casting conditions,
    proves that the length of the shadow cast by the flagpole is 45 meters. -/
theorem flagpole_shadow_length
  (flagpole_height : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_building_height : building_height = 20)
  (h_building_shadow : building_shadow = 50)
  (h_similar_conditions : flagpole_height / building_height = building_shadow / building_shadow) :
  flagpole_height * building_shadow / building_height = 45 :=
sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l2639_263913


namespace NUMINAMATH_CALUDE_percent_of_double_is_nine_l2639_263958

theorem percent_of_double_is_nine (x : ℝ) : 
  x > 0 → (0.01 * x * (2 * x) = 9) → x = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_double_is_nine_l2639_263958


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2639_263983

theorem expand_and_simplify (x : ℝ) : (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2639_263983


namespace NUMINAMATH_CALUDE_find_other_number_l2639_263984

theorem find_other_number (a b : ℤ) : 
  (a = 17 ∨ b = 17) → 
  (3 * a + 4 * b = 131) → 
  (a = 21 ∨ b = 21) :=
by sorry

end NUMINAMATH_CALUDE_find_other_number_l2639_263984


namespace NUMINAMATH_CALUDE_cube_root_scaling_l2639_263936

theorem cube_root_scaling (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a^(1/3) = 2.938) (h2 : b^(1/3) = 6.329) (h3 : c = 253600) 
  (h4 : b = 10 * a) (h5 : c = 1000 * b) : 
  c^(1/3) = 63.29 := by sorry

end NUMINAMATH_CALUDE_cube_root_scaling_l2639_263936


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2639_263957

theorem complex_equation_solutions :
  (∃ (x y : ℝ), (x + y) + (y - 1) * I = (2 * x + 3 * y) + (2 * y + 1) * I ∧ x = 4 ∧ y = -2) ∧
  (∃ (x y : ℝ), (x + y - 3) + (x - 2) * I = 0 ∧ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2639_263957


namespace NUMINAMATH_CALUDE_orange_juice_problem_l2639_263930

/-- Calculates the number of servings of orange juice prepared from concentrate -/
def orange_juice_servings (concentrate_cans : ℕ) (concentrate_oz_per_can : ℕ) 
  (water_cans_per_concentrate : ℕ) (oz_per_serving : ℕ) : ℕ :=
  let total_oz := concentrate_cans * concentrate_oz_per_can * (water_cans_per_concentrate + 1)
  total_oz / oz_per_serving

/-- Theorem stating that 60 cans of 5-oz concentrate mixed with 3 cans of water per
    1 can of concentrate yields 200 servings of 6-oz orange juice -/
theorem orange_juice_problem : 
  orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_problem_l2639_263930


namespace NUMINAMATH_CALUDE_day_temperature_difference_l2639_263925

def temperature_difference (lowest highest : ℤ) : ℤ :=
  highest - lowest

theorem day_temperature_difference :
  let lowest : ℤ := -15
  let highest : ℤ := 3
  temperature_difference lowest highest = 18 := by
  sorry

end NUMINAMATH_CALUDE_day_temperature_difference_l2639_263925


namespace NUMINAMATH_CALUDE_base_representation_of_500_l2639_263987

theorem base_representation_of_500 :
  ∃! b : ℕ, b > 1 ∧ b^5 ≤ 500 ∧ 500 < b^6 :=
sorry

end NUMINAMATH_CALUDE_base_representation_of_500_l2639_263987


namespace NUMINAMATH_CALUDE_high_school_total_students_l2639_263969

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  total_sample : ℕ
  grade1_sample : ℕ
  grade2_sample : ℕ
  grade3_total : ℕ
  sample_sum : total_sample = grade1_sample + grade2_sample + (total_sample - grade1_sample - grade2_sample)
  grade3_prob : (total_sample - grade1_sample - grade2_sample) / grade3_total = 1 / 20

/-- The total number of students in the high school is 3600. -/
theorem high_school_total_students (h : HighSchool) 
  (h_total_sample : h.total_sample = 180)
  (h_grade1_sample : h.grade1_sample = 70)
  (h_grade2_sample : h.grade2_sample = 60)
  (h_grade3_total : h.grade3_total = 1000) : 
  h.total_sample * 20 = 3600 := by
  sorry

#check high_school_total_students

end NUMINAMATH_CALUDE_high_school_total_students_l2639_263969


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l2639_263959

/-- The line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0

/-- The second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

/-- The equation of the circle we want to prove -/
def target_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 7*y - 32 = 0

/-- Theorem stating that the target_circle satisfies the given conditions -/
theorem circle_satisfies_conditions :
  ∃ (h k : ℝ), 
    (center_line h k) ∧ 
    (∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → target_circle x y) ∧
    ((h - 1/2)^2 + (k - 7/2)^2 = (33/2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l2639_263959


namespace NUMINAMATH_CALUDE_card_selection_two_suits_l2639_263901

theorem card_selection_two_suits (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) 
  (selection_size : ℕ) (h1 : deck_size = suits * cards_per_suit) 
  (h2 : suits = 4) (h3 : cards_per_suit = 13) (h4 : selection_size = 3) : 
  (suits.choose 2) * (cards_per_suit.choose 2 * cards_per_suit.choose 1 + 
   cards_per_suit.choose 1 * cards_per_suit.choose 2) = 12168 :=
by sorry

end NUMINAMATH_CALUDE_card_selection_two_suits_l2639_263901


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l2639_263954

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - (k - 1) * x - 2 * k + 8 > 0) ↔ -9 < k ∧ k < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l2639_263954


namespace NUMINAMATH_CALUDE_max_distance_complex_l2639_263980

/-- Given a complex number z₁ = i(1-i)³ and any complex number z such that |z| = 1,
    the maximum value of |z - z₁| is 1 + 2√2. -/
theorem max_distance_complex (z : ℂ) : 
  let z₁ : ℂ := Complex.I * (1 - Complex.I)^3
  Complex.abs z = 1 →
  (⨆ (z : ℂ), Complex.abs (z - z₁)) = 1 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2639_263980


namespace NUMINAMATH_CALUDE_car_distance_calculation_car_distance_is_432_l2639_263950

/-- Given a car's journey with known time and alternative speed, calculate the distance. -/
theorem car_distance_calculation (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  let new_time := initial_time * time_ratio
  let distance := new_speed * new_time
  distance

/-- Prove that the distance covered by the car is 432 km. -/
theorem car_distance_is_432 :
  car_distance_calculation 6 48 (3/2) = 432 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_car_distance_is_432_l2639_263950


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1503_odd_integers_l2639_263978

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map (fun x => x * x) |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_squares_of_first_1503_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 1503)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1503_odd_integers_l2639_263978


namespace NUMINAMATH_CALUDE_b_share_calculation_l2639_263917

def total_cost : ℕ := 520
def hours_a : ℕ := 7
def hours_b : ℕ := 8
def hours_c : ℕ := 11

theorem b_share_calculation : 
  let total_hours := hours_a + hours_b + hours_c
  let cost_per_hour := total_cost / total_hours
  cost_per_hour * hours_b = 160 := by
  sorry

end NUMINAMATH_CALUDE_b_share_calculation_l2639_263917


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2639_263906

theorem ratio_of_numbers (A B C : ℝ) (k : ℝ) 
  (h1 : A = k * B)
  (h2 : A = 3 * C)
  (h3 : (A + B + C) / 3 = 88)
  (h4 : A - C = 96) :
  A / B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2639_263906


namespace NUMINAMATH_CALUDE_balls_sold_l2639_263923

/-- Proves that the number of balls sold is 17 given the conditions of the problem -/
theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) :
  selling_price = 720 →
  loss = 5 * cost_price →
  cost_price = 60 →
  selling_price + loss = 17 * cost_price :=
by
  sorry

#check balls_sold

end NUMINAMATH_CALUDE_balls_sold_l2639_263923


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2639_263938

theorem polynomial_coefficient_equality 
  (a b c d : ℚ) :
  (∀ x : ℚ, (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) = 
    18 * x^6 - 2 * x^5 + 16 * x^4 - 28/3 * x^3 + 8/3 * x^2 - 4 * x + 2) →
  b = -1/3 ∧ c = 14/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2639_263938


namespace NUMINAMATH_CALUDE_opposite_hands_theorem_l2639_263955

/-- The time in minutes past 10:00 when the minute hand will be exactly opposite
    the place where the hour hand was four minutes ago, eight minutes from now. -/
def opposite_hands_time : ℝ :=
  let t : ℝ := 29.09090909090909  -- Approximate value of 29 1/11
  t

/-- Theorem stating that the calculated time satisfies the given conditions -/
theorem opposite_hands_theorem :
  let t := opposite_hands_time
  -- Time is between 10:00 and 11:00
  0 < t ∧ t < 60 ∧
  -- Minute hand position 8 minutes from now
  let minute_pos := 6 * (t + 8)
  -- Hour hand position 4 minutes ago
  let hour_pos := 30 + 0.5 * (t - 4)
  -- Hands are opposite (180 degrees apart)
  |minute_pos - hour_pos| = 180 := by
  sorry

#eval opposite_hands_time

end NUMINAMATH_CALUDE_opposite_hands_theorem_l2639_263955


namespace NUMINAMATH_CALUDE_parallelogram_properties_l2639_263965

-- Define the parallelogram vertices as complex numbers
def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

-- Define the parallelogram
def parallelogram (A B C : ℂ) : Prop :=
  ∃ D : ℂ, (C - B) = (D - A) ∧ (D - C) = (B - A)

-- Theorem statement
theorem parallelogram_properties (h : parallelogram A B C) :
  ∃ D : ℂ,
    D = 4 + 3 * Complex.I ∧
    Complex.abs (C - A) = Real.sqrt 17 ∧
    Complex.abs (D - B) = Real.sqrt 18 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l2639_263965


namespace NUMINAMATH_CALUDE_intersection_count_l2639_263922

/-- A point in the Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Checks if a point is the intersection of two lines -/
def is_intersection (p : Point) (k : ℤ) : Prop :=
  p.y = p.x - 3 ∧ p.y = k * p.x - k

/-- The theorem statement -/
theorem intersection_count : 
  ∃ (s : Finset ℤ), s.card = 3 ∧ 
  (∀ k : ℤ, k ∈ s ↔ ∃ p : Point, is_intersection p k) :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l2639_263922


namespace NUMINAMATH_CALUDE_line_circle_intersection_radius_l2639_263939

/-- Given a line and a circle that intersect at two points forming a 120° angle at the origin, 
    prove that the radius of the circle is 2. -/
theorem line_circle_intersection_radius (r : ℝ) (h_r_pos : r > 0) : 
  (∃ (A B : ℝ × ℝ), 
    (3 * A.1 - 4 * A.2 + 5 = 0) ∧ 
    (3 * B.1 - 4 * B.2 + 5 = 0) ∧
    (A.1^2 + A.2^2 = r^2) ∧ 
    (B.1^2 + B.2^2 = r^2) ∧
    (Real.cos (Real.pi / 3) * r^2 = A.1 * B.1 + A.2 * B.2)) →
  r = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_radius_l2639_263939


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2639_263912

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2639_263912


namespace NUMINAMATH_CALUDE_part_one_part_two_l2639_263919

-- Part 1
theorem part_one (x : ℝ) (h1 : x^2 - 4*x + 3 < 0) 
  (h2 : |x - 1| ≤ 2) (h3 : (x + 3) / (x - 2) ≥ 0) : 
  2 < x ∧ x ≤ 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x : ℝ, ¬(x^2 - 4*a*x + 3*a^2 < 0) → 
    ¬(|x - 1| ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0))
  (h_not_nec : ∃ x : ℝ, ¬(x^2 - 4*a*x + 3*a^2 < 0) ∧ 
    (|x - 1| ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0)) :
  a > 3/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2639_263919


namespace NUMINAMATH_CALUDE_ellipse_slope_l2639_263956

theorem ellipse_slope (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e : ℝ := 1/3
  let c : ℝ := a * e
  let k : ℝ := (b^2/a) / (c - (-a))
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    (x = -a ∧ y = 0) ∨ (x = c ∧ y = b^2/a)) →
  k = 2/3 := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_l2639_263956


namespace NUMINAMATH_CALUDE_handshake_count_l2639_263940

/-- The number of players in each team -/
def team_size : ℕ := 6

/-- The number of teams -/
def num_teams : ℕ := 2

/-- The number of referees -/
def num_referees : ℕ := 2

/-- The total number of handshakes -/
def total_handshakes : ℕ :=
  -- Handshakes between teams
  team_size * team_size +
  -- Handshakes within each team
  num_teams * (team_size.choose 2) +
  -- Handshakes with referees
  (num_teams * team_size) * num_referees

theorem handshake_count : total_handshakes = 90 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2639_263940


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l2639_263915

/-- Given a cost per page in cents and a budget in dollars, 
    calculates the maximum number of full pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem stating that with a cost of 3 cents per page and a budget of $25, 
    the maximum number of full pages that can be copied is 833. -/
theorem copy_pages_theorem : max_pages_copied 3 25 = 833 := by
  sorry

#eval max_pages_copied 3 25

end NUMINAMATH_CALUDE_copy_pages_theorem_l2639_263915


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2639_263948

theorem sin_cos_equation_solutions (x : Real) :
  (Real.sin x + Real.cos x + Real.sin x * Real.cos x = 1) ↔
  (∃ k : ℤ, x = k * (2 * Real.pi) ∨ x = Real.pi / 2 + k * (2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2639_263948


namespace NUMINAMATH_CALUDE_tangent_lines_and_intersection_points_l2639_263941

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the line l
def l (m x y : ℝ) : Prop := m * x - y - m + 1 = 0

theorem tangent_lines_and_intersection_points :
  -- Part 1: Tangent lines
  (∀ x y : ℝ, (x + 2*y - 7 = 0 → C x y) ∧ (2*x - y - 4 = 0 → C x y)) ∧
  (x + 2*y - 7 = 0 → x = M.1 ∧ y = M.2) ∧
  (2*x - y - 4 = 0 → x = M.1 ∧ y = M.2) ∧
  -- Part 2: Intersection points
  (∀ m : ℝ, (∃ A B : ℝ × ℝ, 
    l m A.1 A.2 ∧ l m B.1 B.2 ∧ 
    C A.1 A.2 ∧ C B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 17) → 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
sorry


end NUMINAMATH_CALUDE_tangent_lines_and_intersection_points_l2639_263941


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2639_263977

/-- Given a line with equation x - 2y + 1 = 0, prove that the sum of its x-intercept and y-intercept is -1/2 -/
theorem line_intercepts_sum (x y : ℝ) : 
  x - 2*y + 1 = 0 → 
  ∃ (x_int y_int : ℝ), x_int - 2*0 + 1 = 0 ∧ 0 - 2*y_int + 1 = 0 ∧ x_int + y_int = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2639_263977


namespace NUMINAMATH_CALUDE_square_perimeter_7m_l2639_263942

/-- The perimeter of a square with side length 7 meters is 28 meters. -/
theorem square_perimeter_7m : 
  ∀ (s : ℝ), s = 7 → 4 * s = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_7m_l2639_263942


namespace NUMINAMATH_CALUDE_stability_comparison_l2639_263910

-- Define the concept of a data set
def DataSet := List ℝ

-- Define the variance of a data set
def variance (s : DataSet) : ℝ := sorry

-- Define the concept of stability for a data set
def is_more_stable (s1 s2 : DataSet) : Prop := 
  variance s1 < variance s2

-- Theorem statement
theorem stability_comparison (A B : DataSet) 
  (h_mean : (A.sum / A.length) = (B.sum / B.length))
  (h_var_A : variance A = 0.3)
  (h_var_B : variance B = 0.02) :
  is_more_stable B A := by sorry

end NUMINAMATH_CALUDE_stability_comparison_l2639_263910


namespace NUMINAMATH_CALUDE_binomial_60_3_l2639_263998

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l2639_263998


namespace NUMINAMATH_CALUDE_bamboo_tube_rice_problem_l2639_263999

theorem bamboo_tube_rice_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, n < 8 → a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  a 0 + a 1 + a 2 = 39/10 →                         -- bottom three joints
  a 5 + a 6 + a 7 + a 8 = 3 →                       -- top four joints
  a 4 = 1 :=                                        -- middle joint
by sorry

end NUMINAMATH_CALUDE_bamboo_tube_rice_problem_l2639_263999


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2639_263916

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 + i) / (1 + i)
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2639_263916


namespace NUMINAMATH_CALUDE_negative_sqrt_13_less_than_negative_3_l2639_263966

theorem negative_sqrt_13_less_than_negative_3 : -Real.sqrt 13 < -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_13_less_than_negative_3_l2639_263966


namespace NUMINAMATH_CALUDE_equator_arc_length_equals_radius_l2639_263963

/-- The radius of the Earth's equator in kilometers -/
def earth_radius : ℝ := 6370

/-- The length of an arc on a circle, given its radius and angle in radians -/
def arc_length (radius : ℝ) (angle : ℝ) : ℝ := radius * angle

/-- Theorem: The length of an arc on the Earth's equator corresponding to 1 radian 
    is equal to the radius of the Earth's equator -/
theorem equator_arc_length_equals_radius : 
  arc_length earth_radius 1 = earth_radius := by sorry

end NUMINAMATH_CALUDE_equator_arc_length_equals_radius_l2639_263963


namespace NUMINAMATH_CALUDE_hospital_worker_count_l2639_263961

theorem hospital_worker_count 
  (total_workers : ℕ) 
  (chosen_workers : ℕ) 
  (specific_pair_prob : ℚ) : 
  total_workers = 8 → 
  chosen_workers = 2 → 
  specific_pair_prob = 1 / 28 → 
  total_workers - 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_hospital_worker_count_l2639_263961


namespace NUMINAMATH_CALUDE_greatest_fraction_l2639_263962

theorem greatest_fraction (a : ℝ) (m n p : ℝ) 
  (h_a : a < -3)
  (h_m : m = (a + 2) / (a + 3))
  (h_n : n = (a + 1) / (a + 2))
  (h_p : p = a / (a + 1)) :
  m > n ∧ n > p := by
sorry

end NUMINAMATH_CALUDE_greatest_fraction_l2639_263962


namespace NUMINAMATH_CALUDE_min_sin4_plus_2cos4_l2639_263935

theorem min_sin4_plus_2cos4 (x : ℝ) : 
  (Real.sin x)^4 + 2 * (Real.cos x)^4 ≥ (1/2 : ℝ) :=
by
  sorry

#check min_sin4_plus_2cos4

end NUMINAMATH_CALUDE_min_sin4_plus_2cos4_l2639_263935


namespace NUMINAMATH_CALUDE_number_less_than_hundred_million_l2639_263982

theorem number_less_than_hundred_million :
  ∃ x : ℕ,
    x < 100000000 ∧
    x + 1000000 = 100000000 ∧
    x = 99000000 ∧
    x / 1000000 = 99 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_hundred_million_l2639_263982


namespace NUMINAMATH_CALUDE_snakes_not_hiding_l2639_263960

/-- Given a total of 95 snakes and 64 hiding snakes, prove that the number of snakes not hiding is 31. -/
theorem snakes_not_hiding (total_snakes : ℕ) (hiding_snakes : ℕ) 
  (h1 : total_snakes = 95) (h2 : hiding_snakes = 64) : 
  total_snakes - hiding_snakes = 31 := by
  sorry

end NUMINAMATH_CALUDE_snakes_not_hiding_l2639_263960


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l2639_263953

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ring arrangements on four fingers -/
def ring_arrangements : ℕ :=
  let total_rings := 10
  let chosen_rings := 7
  let fingers := 4
  binomial total_rings chosen_rings * 
  factorial chosen_rings * 
  chosen_rings * 
  binomial (chosen_rings + fingers - 2) (fingers - 1)

theorem ring_arrangement_count : ring_arrangements = 264537600 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l2639_263953


namespace NUMINAMATH_CALUDE_vasya_driving_distance_l2639_263921

theorem vasya_driving_distance 
  (total_distance : ℝ) 
  (anton_distance vasya_distance sasha_distance dima_distance : ℝ)
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance)
  : vasya_distance = (2 / 5) * total_distance := by
  sorry

end NUMINAMATH_CALUDE_vasya_driving_distance_l2639_263921


namespace NUMINAMATH_CALUDE_milk_remaining_l2639_263937

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial = 5.5 ∧ given_away = 17/4 → remaining = initial - given_away → remaining = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l2639_263937


namespace NUMINAMATH_CALUDE_item_a_price_correct_l2639_263964

/-- The price of item (a) in won -/
def item_a_price : ℕ := 7 * 1000 + 4 * 100 + 5 * 10

/-- The number of 1000 won coins used -/
def coins_1000 : ℕ := 7

/-- The number of 100 won coins used -/
def coins_100 : ℕ := 4

/-- The number of 10 won coins used -/
def coins_10 : ℕ := 5

theorem item_a_price_correct : item_a_price = 7450 := by
  sorry

end NUMINAMATH_CALUDE_item_a_price_correct_l2639_263964


namespace NUMINAMATH_CALUDE_hoseok_persimmons_l2639_263972

theorem hoseok_persimmons :
  ∀ (jungkook_persimmons hoseok_persimmons : ℕ),
    jungkook_persimmons = 25 →
    jungkook_persimmons = 3 * hoseok_persimmons + 4 →
    hoseok_persimmons = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_persimmons_l2639_263972


namespace NUMINAMATH_CALUDE_log_identity_l2639_263914

theorem log_identity (x : ℝ) : 
  x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3) →
  Real.log x / Real.log 7 = -(Real.log (Real.log 5 / Real.log 3) / Real.log 7) * (Real.log 5 / Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_log_identity_l2639_263914


namespace NUMINAMATH_CALUDE_tangent_from_cosine_central_angle_from_perimeter_and_area_l2639_263979

-- Part 1
theorem tangent_from_cosine (m : ℝ) (α : ℝ) :
  (m : ℝ) = -Real.sqrt 2 / 4 →
  Real.cos α = -1/3 →
  Real.tan α = -2 * Real.sqrt 2 :=
by sorry

-- Part 2
theorem central_angle_from_perimeter_and_area (r l : ℝ) :
  2 * r + l = 8 →
  1/2 * l * r = 3 →
  (l / r = 2/3 ∨ l / r = 6) :=
by sorry

end NUMINAMATH_CALUDE_tangent_from_cosine_central_angle_from_perimeter_and_area_l2639_263979


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l2639_263924

theorem right_triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : A < π / 2) :
  B = π / 2 →
  3 * Real.sin A = 4 * Real.cos A + Real.tan A →
  Real.sin A = 2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l2639_263924


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2639_263990

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EH : ℝ
  ef_eq_gh : EF = GH
  fg_eq_10 : FG = 10
  eh_eq_20 : EH = 20
  right_triangle : EF^2 = 5^2 + 5^2

/-- The perimeter of the trapezoid EFGH is 30 + 10√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  t.EF + t.FG + t.GH + t.EH = 30 + 10 * Real.sqrt 2 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2639_263990


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2639_263994

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 1 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 1) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2639_263994


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l2639_263946

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l2639_263946


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2639_263996

theorem x_plus_y_value (x y : ℝ) (h1 : x - y = 4) (h2 : |x| + |y| = 7) :
  x + y = 7 ∨ x + y = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2639_263996


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2639_263933

-- Define the variables
variable (a b c d x y z : ℝ)

-- Define the given ratios
def ratio_a_to_b : a / b = 2 * x / (3 * y) := by sorry
def ratio_b_to_c : b / c = z / (5 * z) := by sorry
def ratio_a_to_d : a / d = 4 * x / (7 * y) := by sorry
def ratio_d_to_c : d / c = 7 * y / (3 * z) := by sorry

-- State the theorem
theorem ratio_a_to_c (ha : a > 0) (hc : c > 0) : a / c = 2 * x / (15 * y) := by sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2639_263933


namespace NUMINAMATH_CALUDE_strawberry_pies_l2639_263918

/-- The number of pies that can be made from strawberries picked by Christine and Rachel -/
def number_of_pies (christine_pounds : ℕ) (rachel_multiplier : ℕ) (pounds_per_pie : ℕ) : ℕ :=
  (christine_pounds + christine_pounds * rachel_multiplier) / pounds_per_pie

/-- Theorem: Christine and Rachel can make 10 pies given the conditions -/
theorem strawberry_pies : number_of_pies 10 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_pies_l2639_263918


namespace NUMINAMATH_CALUDE_easter_egg_hunt_problem_l2639_263967

/-- Represents the number of eggs of each size found by a child -/
structure EggCount where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total points for a given EggCount -/
def totalPoints (eggs : EggCount) : ℕ :=
  eggs.small + 3 * eggs.medium + 5 * eggs.large

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt_problem :
  let kevin := EggCount.mk 5 0 3
  let bonnie := EggCount.mk 13 7 2
  let george := EggCount.mk 9 6 1
  let cheryl := EggCount.mk 56 30 15
  totalPoints cheryl - (totalPoints kevin + totalPoints bonnie + totalPoints george) = 125 := by
  sorry


end NUMINAMATH_CALUDE_easter_egg_hunt_problem_l2639_263967


namespace NUMINAMATH_CALUDE_prob_at_least_two_girls_is_two_sevenths_l2639_263920

def total_students : ℕ := 8
def boys : ℕ := 5
def girls : ℕ := 3
def selected : ℕ := 3

def prob_at_least_two_girls : ℚ :=
  (Nat.choose girls 2 * Nat.choose boys 1 + Nat.choose girls 3 * Nat.choose boys 0) /
  Nat.choose total_students selected

theorem prob_at_least_two_girls_is_two_sevenths :
  prob_at_least_two_girls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_girls_is_two_sevenths_l2639_263920


namespace NUMINAMATH_CALUDE_valid_outfits_count_l2639_263905

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 5

/-- The number of colors available for each item -/
def num_colors : ℕ := 5

/-- The number of outfits where no two items are the same color -/
def num_valid_outfits : ℕ := num_items * (num_items - 1) * (num_items - 2)

theorem valid_outfits_count :
  num_valid_outfits = 60 :=
by sorry

end NUMINAMATH_CALUDE_valid_outfits_count_l2639_263905


namespace NUMINAMATH_CALUDE_det_A_eq_zero_iff_x_eq_52_19_l2639_263970

def A (x : ℚ) : Matrix (Fin 3) (Fin 3) ℚ :=
  !![3, 1, -1;
     4, x, 2;
     1, 3, 6]

theorem det_A_eq_zero_iff_x_eq_52_19 :
  ∀ x : ℚ, Matrix.det (A x) = 0 ↔ x = 52 / 19 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_zero_iff_x_eq_52_19_l2639_263970


namespace NUMINAMATH_CALUDE_smallest_a_for_integer_roots_and_product_condition_l2639_263968

theorem smallest_a_for_integer_roots_and_product_condition : 
  (∃ (a : ℕ+), 
    (∀ (x : ℤ), x^2 + a*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z) ∧ 
    (∀ (b : ℕ+), b < a → 
      ¬(∀ (x : ℤ), x^2 + b*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z))) ∧
  (∀ (a : ℕ+), 
    (∀ (x : ℤ), x^2 + a*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z) ∧ 
    (∀ (b : ℕ+), b < a → 
      ¬(∀ (x : ℤ), x^2 + b*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z)) → 
    a = 11) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_integer_roots_and_product_condition_l2639_263968
