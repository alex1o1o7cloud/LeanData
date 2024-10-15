import Mathlib

namespace NUMINAMATH_CALUDE_abc_inequality_l3874_387438

/-- Given a = 2/ln(4), b = ln(3)/ln(2), c = 3/2, prove that b > c > a -/
theorem abc_inequality (a b c : ℝ) (ha : a = 2 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) (hc : c = 3 / 2) :
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3874_387438


namespace NUMINAMATH_CALUDE_marbles_distribution_l3874_387445

/-- Given 20 marbles distributed equally among 2 boys, prove that each boy receives 10 marbles. -/
theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 20 →
  num_boys = 2 →
  marbles_per_boy * num_boys = total_marbles →
  marbles_per_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3874_387445


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3874_387404

/-- A quadratic equation in x with parameter m, where one root is zero -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + x + m^2 + 3*m = 0

/-- The theorem stating that m = -3 for the given quadratic equation -/
theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, quadratic_equation m x) ∧ 
  (quadratic_equation m 0) ∧ 
  (m ≠ 0) → 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3874_387404


namespace NUMINAMATH_CALUDE_select_one_from_two_sets_l3874_387419

theorem select_one_from_two_sets (left_set right_set : Finset ℕ) 
  (h1 : left_set.card = 15) (h2 : right_set.card = 20) 
  (h3 : left_set ∩ right_set = ∅) : 
  (left_set ∪ right_set).card = 35 := by
  sorry

end NUMINAMATH_CALUDE_select_one_from_two_sets_l3874_387419


namespace NUMINAMATH_CALUDE_line_properties_l3874_387402

/-- Triangle PQR with vertices P(1, 9), Q(3, 2), and R(9, 2) -/
structure Triangle where
  P : ℝ × ℝ := (1, 9)
  Q : ℝ × ℝ := (3, 2)
  R : ℝ × ℝ := (9, 2)

/-- A line defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Function to calculate the area of a triangle given its vertices -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a line cuts the triangle's area in half -/
def cutsAreaInHalf (t : Triangle) (l : Line) : Prop := sorry

/-- Theorem stating the properties of the line that cuts the triangle's area in half -/
theorem line_properties (t : Triangle) (l : Line) :
  cutsAreaInHalf t l ∧ l.yIntercept = 1 →
  l.slope = 1/3 ∧ l.slope + l.yIntercept = 4/3 := by sorry

end NUMINAMATH_CALUDE_line_properties_l3874_387402


namespace NUMINAMATH_CALUDE_license_plate_increase_l3874_387413

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4 * 2
  (new_plates : ℚ) / old_plates = 135.2 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3874_387413


namespace NUMINAMATH_CALUDE_new_figure_length_l3874_387421

/-- A polygon with adjacent perpendicular sides -/
structure PerpendicularPolygon where
  sides : List ℝ
  adjacent_perpendicular : Bool

/-- The new figure formed by removing four sides from the original polygon -/
def new_figure (p : PerpendicularPolygon) : List ℝ :=
  sorry

/-- Theorem: The total length of segments in the new figure is 22 units -/
theorem new_figure_length (p : PerpendicularPolygon) 
  (h1 : p.adjacent_perpendicular = true)
  (h2 : p.sides = [9, 3, 7, 1, 1]) :
  (new_figure p).sum = 22 := by
  sorry

end NUMINAMATH_CALUDE_new_figure_length_l3874_387421


namespace NUMINAMATH_CALUDE_min_value_inequality_l3874_387464

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 3) : 1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3874_387464


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l3874_387499

def A : Set ℝ := {x | x^2 + x - 12 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem intersection_implies_m_value :
  ∃ m : ℝ, A ∩ B m = {3} → m = -1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l3874_387499


namespace NUMINAMATH_CALUDE_photo_arrangements_l3874_387462

/-- Represents the number of students in the photo --/
def num_students : ℕ := 5

/-- Represents the constraint that B and C must stand together --/
def bc_together : Prop := True

/-- Represents the constraint that A cannot stand next to B --/
def a_not_next_to_b : Prop := True

/-- The number of different arrangements --/
def num_arrangements : ℕ := 36

/-- Theorem stating that the number of arrangements is 36 --/
theorem photo_arrangements :
  (num_students = 5) →
  bc_together →
  a_not_next_to_b →
  num_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3874_387462


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3874_387440

theorem quadratic_inequality_condition (a : ℝ) (h : 0 ≤ a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3874_387440


namespace NUMINAMATH_CALUDE_k_range_l3874_387480

/-- The function y = |log₂ x| is meaningful and not monotonic in the interval (k-1, k+1) -/
def is_meaningful_and_not_monotonic (k : ℝ) : Prop :=
  (k - 1 > 0) ∧ (1 ∈ Set.Ioo (k - 1) (k + 1))

/-- The theorem stating the range of k -/
theorem k_range :
  ∀ k : ℝ, is_meaningful_and_not_monotonic k ↔ k ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l3874_387480


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l3874_387479

/-- Two vectors a and b in ℝ² are collinear if there exists a scalar k such that b = k * a -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given two vectors a = (1, -2) and b = (-2, x) in ℝ², 
    if a and b are collinear, then x = 4 -/
theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-2, x)
  collinear a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l3874_387479


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3874_387453

/-- Given a bus that travels at 43 kmph including stoppages and stops for 8.4 minutes per hour,
    its speed excluding stoppages is 50 kmph. -/
theorem bus_speed_excluding_stoppages (speed_with_stops : ℝ) (stoppage_time : ℝ) :
  speed_with_stops = 43 →
  stoppage_time = 8.4 →
  (60 - stoppage_time) / 60 * speed_with_stops = 50 := by
  sorry

#check bus_speed_excluding_stoppages

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3874_387453


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3874_387452

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (16/7) * x^2 + (32/7) * x - 240/7

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions : 
  q (-5) = 0 ∧ q 3 = 0 ∧ q 2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3874_387452


namespace NUMINAMATH_CALUDE_pencil_dozens_l3874_387410

theorem pencil_dozens (total_pencils : ℕ) (pencils_per_dozen : ℕ) (h1 : total_pencils = 144) (h2 : pencils_per_dozen = 12) :
  total_pencils / pencils_per_dozen = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_dozens_l3874_387410


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l3874_387446

-- Define the number of each type of plant
def num_basil : ℕ := 3
def num_tomato : ℕ := 3
def num_pepper : ℕ := 2

-- Define the number of tomato groups
def num_tomato_groups : ℕ := 2

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ :=
  (Nat.factorial num_basil) *
  (Nat.choose num_tomato 2) *
  (Nat.factorial 2) *
  (Nat.choose (num_basil + 1) num_tomato_groups) *
  (Nat.factorial num_pepper)

-- Theorem statement
theorem plant_arrangement_count :
  num_arrangements = 432 :=
sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l3874_387446


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l3874_387473

/-- Proves that mixing 200 mL of 10% alcohol solution with 600 mL of 30% alcohol solution results in a 25% alcohol mixture -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let y_volume : ℝ := 600
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l3874_387473


namespace NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l3874_387449

theorem cubic_sum_divisible_by_nine (n : ℕ+) : 
  ∃ k : ℤ, (n : ℤ)^3 + (n + 1 : ℤ)^3 + (n + 2 : ℤ)^3 = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l3874_387449


namespace NUMINAMATH_CALUDE_tan_c_in_triangle_l3874_387400

theorem tan_c_in_triangle (A B C : Real) : 
  -- Triangle condition
  A + B + C = π → 
  -- tan A and tan B are roots of 3x^2 - 7x + 2 = 0
  (∃ (x y : Real), x ≠ y ∧ 
    3 * x^2 - 7 * x + 2 = 0 ∧ 
    3 * y^2 - 7 * y + 2 = 0 ∧ 
    x = Real.tan A ∧ 
    y = Real.tan B) → 
  -- Conclusion
  Real.tan C = -7 :=
by sorry

end NUMINAMATH_CALUDE_tan_c_in_triangle_l3874_387400


namespace NUMINAMATH_CALUDE_project_hours_difference_l3874_387466

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 189) 
  (kate_hours : ℕ) (pat_hours : ℕ) (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours = mark_hours / 3) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 105 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3874_387466


namespace NUMINAMATH_CALUDE_fraction_reducibility_implies_divisibility_l3874_387431

theorem fraction_reducibility_implies_divisibility 
  (a b c n l p : ℤ) 
  (h_reducible : ∃ (k m : ℤ), a * l + b = p * k ∧ c * l + n = p * m) : 
  p ∣ (a * n - b * c) := by
sorry

end NUMINAMATH_CALUDE_fraction_reducibility_implies_divisibility_l3874_387431


namespace NUMINAMATH_CALUDE_min_value_expression_l3874_387485

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 2) :
  (x + 40 * y + 4) / (3 * x * y) ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3874_387485


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l3874_387495

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  side : ℝ

/-- Defines the locus equation for a point P relative to an isosceles triangle -/
def locusEquation (P : Point) (triangle : IsoscelesTriangle) (k : ℝ) : Prop :=
  3 * P.x^2 + 2 * P.y^2 - 2 * triangle.height * P.y + triangle.height^2 + 
  2 * (triangle.base / 2)^2 = k * triangle.side^2

/-- States that the locus of points satisfying the equation forms an ellipse -/
theorem locus_is_ellipse (triangle : IsoscelesTriangle) (k : ℝ) 
    (h_k : k > 1) (h_side : triangle.side^2 = (triangle.base / 2)^2 + triangle.height^2) :
  ∃ (center : Point) (a b : ℝ), ∀ (P : Point),
    locusEquation P triangle k ↔ 
    (P.x - center.x)^2 / a^2 + (P.y - center.y)^2 / b^2 = 1 :=
  sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l3874_387495


namespace NUMINAMATH_CALUDE_certain_number_proof_l3874_387424

theorem certain_number_proof (x : ℝ) : 
  (0.15 * x - (1/3) * (0.15 * x)) = 18 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3874_387424


namespace NUMINAMATH_CALUDE_point_inside_circle_l3874_387441

theorem point_inside_circle (a : ℝ) : 
  (∃ (x y : ℝ), x = 2*a ∧ y = a - 1 ∧ x^2 + y^2 - 2*y - 4 < 0) ↔ 
  (-1/5 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3874_387441


namespace NUMINAMATH_CALUDE_gasoline_consumption_reduction_l3874_387451

theorem gasoline_consumption_reduction 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (spending_increase : ℝ) 
  (h1 : price_increase = 0.20) 
  (h2 : spending_increase = 0.14) : 
  let new_price := original_price * (1 + price_increase)
  let new_spending := original_price * original_quantity * (1 + spending_increase)
  let new_quantity := new_spending / new_price
  (original_quantity - new_quantity) / original_quantity = 0.05 := by
sorry

end NUMINAMATH_CALUDE_gasoline_consumption_reduction_l3874_387451


namespace NUMINAMATH_CALUDE_min_values_xy_l3874_387443

/-- Given two positive real numbers x and y satisfying lgx + lgy = lg(x + y + 3),
    prove that the minimum value of xy is 9 and the minimum value of x + y is 6 -/
theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : Real.log x + Real.log y = Real.log (x + y + 3)) : 
    (∀ a b : ℝ, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x * y ≤ a * b) ∧
    (∀ c d : ℝ, c > 0 → d > 0 → Real.log c + Real.log d = Real.log (c + d + 3) → x + y ≤ c + d) ∧
    x * y = 9 ∧ x + y = 6 := by
  sorry


end NUMINAMATH_CALUDE_min_values_xy_l3874_387443


namespace NUMINAMATH_CALUDE_triangle_area_from_rectangle_ratio_l3874_387455

/-- Given a rectangle with length 6 cm and width 4 cm, and a triangle whose area is in a 5:2 ratio
    with the rectangle's area, prove that the area of the triangle is 60 cm². -/
theorem triangle_area_from_rectangle_ratio :
  ∀ (rectangle_length rectangle_width triangle_area : ℝ),
  rectangle_length = 6 →
  rectangle_width = 4 →
  5 * (rectangle_length * rectangle_width) = 2 * triangle_area →
  triangle_area = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_rectangle_ratio_l3874_387455


namespace NUMINAMATH_CALUDE_product_ABC_l3874_387454

theorem product_ABC (m : ℝ) : 
  let A := 4 * m
  let B := m - (1/4 : ℝ)
  let C := m + (1/4 : ℝ)
  A * B * C = 4 * m^3 - (1/4 : ℝ) * m :=
by sorry

end NUMINAMATH_CALUDE_product_ABC_l3874_387454


namespace NUMINAMATH_CALUDE_school_student_count_l3874_387418

theorem school_student_count :
  ∀ (total_students : ℕ),
  (∃ (girls boys : ℕ),
    girls = boys ∧
    girls + boys = total_students ∧
    (girls : ℚ) * (1/5) + (boys : ℚ) * (1/10) = 15) →
  total_students = 100 := by
sorry

end NUMINAMATH_CALUDE_school_student_count_l3874_387418


namespace NUMINAMATH_CALUDE_count_cubes_with_at_most_two_shared_vertices_l3874_387468

/-- Given a cube with edge length n divided into n^3 unit cubes, 
    this function calculates the number of unit cubes that share 
    no more than 2 vertices with any other unit cube. -/
def cubes_with_at_most_two_shared_vertices (n : ℕ) : ℕ :=
  (n^2 * (n^4 - 7*n + 6)) / 2

/-- Theorem stating that the number of unit cubes sharing no more than 2 vertices 
    in a cube of edge length n is given by the formula (1/2) * n^2 * (n^4 - 7n + 6). -/
theorem count_cubes_with_at_most_two_shared_vertices (n : ℕ) :
  cubes_with_at_most_two_shared_vertices n = (n^2 * (n^4 - 7*n + 6)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_count_cubes_with_at_most_two_shared_vertices_l3874_387468


namespace NUMINAMATH_CALUDE_exists_rectangle_same_parity_l3874_387437

/-- Represents a rectangle on a grid -/
structure GridRectangle where
  length : ℕ
  width : ℕ

/-- Represents a square cut into rectangles -/
structure CutSquare where
  side_length : ℕ
  rectangles : List GridRectangle

/-- Checks if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Checks if two numbers have the same parity -/
def same_parity (a b : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨ (¬is_even a ∧ ¬is_even b)

/-- Main theorem: In a square with side length 2009 cut into rectangles,
    there exists at least one rectangle with sides of the same parity -/
theorem exists_rectangle_same_parity (sq : CutSquare) 
    (h1 : sq.side_length = 2009) 
    (h2 : sq.rectangles.length > 0) : 
    ∃ (r : GridRectangle), r ∈ sq.rectangles ∧ same_parity r.length r.width := by
  sorry

end NUMINAMATH_CALUDE_exists_rectangle_same_parity_l3874_387437


namespace NUMINAMATH_CALUDE_solution_range_l3874_387447

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (2 * x - 1) / (x + 1) = 3 - m / (x + 1)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ equation x m ∧ x ≠ -1) → m < 4 ∧ m ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l3874_387447


namespace NUMINAMATH_CALUDE_common_prime_root_quadratics_l3874_387436

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * (p : ℤ) + b = 0 ∧ 
    (p : ℤ)^2 + b * (p : ℤ) + 1100 = 0) →
  a = 274 ∨ a = 40 := by
sorry

end NUMINAMATH_CALUDE_common_prime_root_quadratics_l3874_387436


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3874_387417

def p (x : ℝ) : ℝ := 3*x^4 + 7*x^3 - 13*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (p (-3) = 0) ∧ (p (-2) = 0) ∧ (p (-1) = 0) ∧ (p (1/3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3874_387417


namespace NUMINAMATH_CALUDE_intersection_points_correct_l3874_387484

/-- The number of intersection points of line segments connecting m points on the positive X-axis
    and n points on the positive Y-axis, where no three segments intersect at the same point. -/
def intersectionPoints (m n : ℕ) : ℚ :=
  (m * (m - 1) * n * (n - 1) : ℚ) / 4

/-- Theorem stating that the number of intersection points is correct. -/
theorem intersection_points_correct (m n : ℕ) :
  intersectionPoints m n = (m * (m - 1) * n * (n - 1) : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_correct_l3874_387484


namespace NUMINAMATH_CALUDE_product_equals_fraction_l3874_387457

theorem product_equals_fraction : 12 * 0.5 * 3 * 0.2 = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l3874_387457


namespace NUMINAMATH_CALUDE_group_average_score_l3874_387439

def class_size : ℕ := 14
def class_average : ℝ := 85
def score_differences : List ℝ := [2, 3, -3, -5, 12, 12, 8, 2, -1, 4, -10, -2, 5, 5]

theorem group_average_score :
  let total_score := class_size * class_average + score_differences.sum
  total_score / class_size = 87.29 := by sorry

end NUMINAMATH_CALUDE_group_average_score_l3874_387439


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3874_387498

theorem rectangle_area_problem (x : ℝ) :
  x > 0 ∧
  (5 - (-1)) * (x - (-2)) = 66 →
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3874_387498


namespace NUMINAMATH_CALUDE_tree_height_problem_l3874_387405

/-- Given a square ABCD with trees of heights a, b, c at vertices A, B, C respectively,
    and a point O inside the square equidistant from all vertices,
    prove that the height of the tree at vertex D is √(a² + c² - b²). -/
theorem tree_height_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = a^2 + c^2 - b^2 := by
  sorry


end NUMINAMATH_CALUDE_tree_height_problem_l3874_387405


namespace NUMINAMATH_CALUDE_apartment_complex_households_l3874_387427

/-- The maximum number of households in Jungkook's apartment complex -/
def max_households : ℕ := 2000

/-- The maximum number of buildings in the apartment complex -/
def max_buildings : ℕ := 25

/-- The maximum number of floors per building -/
def max_floors : ℕ := 10

/-- The number of households per floor -/
def households_per_floor : ℕ := 8

/-- Theorem stating that the maximum number of households in the apartment complex is 2000 -/
theorem apartment_complex_households :
  max_households = max_buildings * max_floors * households_per_floor :=
by sorry

end NUMINAMATH_CALUDE_apartment_complex_households_l3874_387427


namespace NUMINAMATH_CALUDE_min_value_expression_l3874_387409

theorem min_value_expression (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a * b + b * c + c * a > 0)
  (h3 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (2 / |a - b| + 2 / |b - c| + 2 / |c - a| + 5 / Real.sqrt (a * b + b * c + c * a)) ≥ 10 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3874_387409


namespace NUMINAMATH_CALUDE_geometric_proof_l3874_387416

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the intersection of two planes resulting in a line
variable (intersect_planes : Plane → Plane → Line)

-- Define the relation of a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem geometric_proof 
  (m n : Line) (α β γ : Plane)
  (h1 : perp_planes α β)
  (h2 : m = intersect_planes α β)
  (h3 : perp_line_plane n α)
  (h4 : line_in_plane n γ) :
  perp_lines m n ∧ perp_planes α γ := by
  sorry

end NUMINAMATH_CALUDE_geometric_proof_l3874_387416


namespace NUMINAMATH_CALUDE_joan_games_this_year_l3874_387469

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := sorry

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

/-- Theorem stating that the number of games Joan went to this year is 4 -/
theorem joan_games_this_year : games_this_year = 4 := by sorry

end NUMINAMATH_CALUDE_joan_games_this_year_l3874_387469


namespace NUMINAMATH_CALUDE_prime_sum_product_l3874_387475

theorem prime_sum_product : 
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ 2 * p + 5 * q = 36 ∧ p * q = 26 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l3874_387475


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3874_387476

theorem trigonometric_identities (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin ((3 * Real.pi) / 2 + α)) : 
  ((2 * Real.sin α - 3 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = 7 / 17) ∧
  (Real.sin α ^ 2 + Real.sin (2 * α) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3874_387476


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_six_l3874_387401

theorem arithmetic_square_root_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_six_l3874_387401


namespace NUMINAMATH_CALUDE_sets_satisfying_union_condition_l3874_387482

theorem sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    (∀ M ∈ S, M ∪ {1} = {1, 2, 3}) ∧ 
    (∀ M, M ∪ {1} = {1, 2, 3} → M ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_sets_satisfying_union_condition_l3874_387482


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l3874_387456

def a (m : ℝ) : ℝ × ℝ := (4, m)
def b : ℝ × ℝ := (1, -2)

theorem magnitude_of_sum (m : ℝ) 
  (h : (a m).1 * b.1 + (a m).2 * b.2 = 0) : 
  Real.sqrt (((a m).1 + 2 * b.1)^2 + ((a m).2 + 2 * b.2)^2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l3874_387456


namespace NUMINAMATH_CALUDE_average_cost_is_14_cents_l3874_387496

/-- Calculates the average cost per pencil in cents, rounded to the nearest whole number -/
def average_cost_per_pencil (num_pencils : ℕ) (catalog_price shipping_cost discount : ℚ) : ℕ :=
  let total_cost_cents := (catalog_price + shipping_cost - discount) * 100
  let average_cost_cents := total_cost_cents / num_pencils
  (average_cost_cents + 1/2).floor.toNat

/-- Proves that the average cost per pencil is 14 cents given the specified conditions -/
theorem average_cost_is_14_cents :
  average_cost_per_pencil 150 15 7.5 1.5 = 14 := by
  sorry

#eval average_cost_per_pencil 150 15 7.5 1.5

end NUMINAMATH_CALUDE_average_cost_is_14_cents_l3874_387496


namespace NUMINAMATH_CALUDE_initial_amount_theorem_l3874_387407

/-- The amount of money in Olivia's wallet before visiting the supermarket. -/
def initial_amount : ℕ := sorry

/-- The amount of money Olivia spent at the supermarket. -/
def amount_spent : ℕ := 16

/-- The amount of money left in Olivia's wallet after visiting the supermarket. -/
def amount_left : ℕ := 78

/-- Theorem stating that the initial amount in Olivia's wallet was $94. -/
theorem initial_amount_theorem : initial_amount = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_amount_theorem_l3874_387407


namespace NUMINAMATH_CALUDE_towels_per_pack_l3874_387435

/-- Given that Tiffany bought 9 packs of towels and 27 towels in total,
    prove that there were 3 towels in each pack. -/
theorem towels_per_pack (total_packs : ℕ) (total_towels : ℕ) 
  (h1 : total_packs = 9) 
  (h2 : total_towels = 27) : 
  total_towels / total_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_towels_per_pack_l3874_387435


namespace NUMINAMATH_CALUDE_triple_hash_twenty_l3874_387497

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_hash_twenty : hash (hash (hash 20)) = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_twenty_l3874_387497


namespace NUMINAMATH_CALUDE_fixed_point_of_square_minus_600_l3874_387406

theorem fixed_point_of_square_minus_600 :
  ∃! (x : ℕ), x = x^2 - 600 :=
by
  -- The unique natural number satisfying the equation is 25
  use 25
  constructor
  · -- Prove that 25 satisfies the equation
    norm_num
  · -- Prove that any natural number satisfying the equation must be 25
    intro y hy
    -- Here we would prove that y = 25
    sorry

#eval (25 : ℕ)^2 - 600  -- This should evaluate to 25

end NUMINAMATH_CALUDE_fixed_point_of_square_minus_600_l3874_387406


namespace NUMINAMATH_CALUDE_binary_representation_of_70_has_7_digits_l3874_387490

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_representation_of_70_has_7_digits :
  (decimal_to_binary 70).length = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_70_has_7_digits_l3874_387490


namespace NUMINAMATH_CALUDE_expression_equality_l3874_387460

theorem expression_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3874_387460


namespace NUMINAMATH_CALUDE_circle_properties_l3874_387420

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2 = 0

-- Define the line L
def L (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the symmetric line
def SymLine (x y : ℝ) : Prop := x - y = 0

-- Define the distance line
def DistLine (x y m : ℝ) : Prop := x + y + m = 0

theorem circle_properties :
  -- 1. Chord length
  (∃ l : ℝ, l = Real.sqrt 6 ∧
    ∀ x y : ℝ, C x y → L x y →
      ∃ x' y' : ℝ, C x' y' ∧ L x' y' ∧
        (x - x')^2 + (y - y')^2 = l^2) ∧
  -- 2. Symmetric circle
  (∀ x y : ℝ, (∃ x' y' : ℝ, C x' y' ∧ SymLine x' y' ∧
    x = y' ∧ y = x') → x^2 + (y-2)^2 = 2) ∧
  -- 3. Distance condition
  (∀ m : ℝ, (abs (m + 2) / Real.sqrt 2 = Real.sqrt 2 / 2) →
    m = -1 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3874_387420


namespace NUMINAMATH_CALUDE_same_end_word_count_l3874_387444

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- A four-letter word with the same first and last letter -/
structure SameEndWord :=
  (first : Fin alphabet_size)
  (second : Fin alphabet_size)
  (third : Fin alphabet_size)

/-- The count of all possible SameEndWords -/
def count_same_end_words : ℕ := alphabet_size * alphabet_size * alphabet_size

theorem same_end_word_count :
  count_same_end_words = 17576 :=
sorry

end NUMINAMATH_CALUDE_same_end_word_count_l3874_387444


namespace NUMINAMATH_CALUDE_ancient_chinese_algorithm_is_successive_subtraction_l3874_387470

/-- An ancient Chinese mathematical algorithm developed during the Song and Yuan dynasties -/
structure AncientChineseAlgorithm where
  name : String
  period : String
  comparable_to_euclidean : Bool

/-- The method of successive subtraction -/
def successive_subtraction : AncientChineseAlgorithm :=
  { name := "Method of Successive Subtraction",
    period := "Song and Yuan dynasties",
    comparable_to_euclidean := true }

/-- Theorem stating that the ancient Chinese algorithm comparable to the Euclidean algorithm
    of division is the method of successive subtraction -/
theorem ancient_chinese_algorithm_is_successive_subtraction :
  ∃ (a : AncientChineseAlgorithm), 
    a.period = "Song and Yuan dynasties" ∧ 
    a.comparable_to_euclidean = true ∧ 
    a = successive_subtraction :=
by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_algorithm_is_successive_subtraction_l3874_387470


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3874_387415

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 * (x + 1/y - 2023) + (y + 1/x)^2 * (y + 1/x - 2023) ≥ -1814505489.667 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 1/y)^2 * (x + 1/y - 2023) + (y + 1/x)^2 * (y + 1/x - 2023) = -1814505489.667 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3874_387415


namespace NUMINAMATH_CALUDE_valid_grid_exists_l3874_387423

/-- A type representing a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two numbers are adjacent in the grid -/
def adjacent (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val)

/-- The main theorem stating the existence of a valid grid -/
theorem valid_grid_exists : ∃ (g : Grid),
  (∀ i j i' j', adjacent i j i' j' → (g i j ∣ g i' j' ∨ g i' j' ∣ g i j)) ∧
  (∀ i j, g i j ≤ 25) ∧
  (∀ i j i' j', (i, j) ≠ (i', j') → g i j ≠ g i' j') :=
sorry

end NUMINAMATH_CALUDE_valid_grid_exists_l3874_387423


namespace NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_zero_l3874_387422

/-- Given a function f(x) = x^2 + a/x where a is a real number,
    if the tangent line at x = 1 is parallel to 2x - y + 1 = 0,
    then a = 0. -/
theorem tangent_line_parallel_implies_a_zero 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + a/x) 
  (h2 : (deriv f 1) = 2) : 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_zero_l3874_387422


namespace NUMINAMATH_CALUDE_geometric_sequence_pairs_l3874_387493

/-- The number of ordered pairs (a, r) satisfying the given conditions -/
def num_pairs : ℕ := 26^3

/-- The base of the logarithm and the exponent in the final equation -/
def base : ℕ := 2015
def exponent : ℕ := 155

theorem geometric_sequence_pairs :
  ∃ (S : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ S ↔ 
      let (a, r) := p
      (a > 0 ∧ r > 0) ∧ (a * r^6 = base^exponent)) ∧
    Finset.card S = num_pairs :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_pairs_l3874_387493


namespace NUMINAMATH_CALUDE_hayden_ironing_time_l3874_387459

/-- The time Hayden spends ironing his clothes over a given number of weeks -/
def ironingTime (shirtTime minutesPerDay : ℕ) (pantsTime minutesPerDay : ℕ) (daysPerWeek : ℕ) (numWeeks : ℕ) : ℕ :=
  (shirtTime + pantsTime) * daysPerWeek * numWeeks

/-- Theorem stating that Hayden spends 160 minutes ironing over 4 weeks -/
theorem hayden_ironing_time :
  ironingTime 5 3 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_hayden_ironing_time_l3874_387459


namespace NUMINAMATH_CALUDE_complex_magnitude_quadratic_l3874_387481

theorem complex_magnitude_quadratic (z : ℂ) : z^2 - 6*z + 25 = 0 → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_quadratic_l3874_387481


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3874_387403

/-- Given an infinite geometric sequence {a_n} with first term 1 and common ratio a - 3/2,
    if the sum of all terms is a, then a = 2. -/
theorem geometric_sequence_sum (a : ℝ) : 
  let a_1 : ℝ := 1
  let q : ℝ := a - 3/2
  let sum : ℝ := a_1 / (1 - q)
  (sum = a) → (a = 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3874_387403


namespace NUMINAMATH_CALUDE_tangent_line_a_zero_max_value_g_positive_a_inequality_a_negative_two_l3874_387433

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x + 1

-- Theorem for the tangent line when a = 0
theorem tangent_line_a_zero :
  ∀ x y : ℝ, f 0 1 = 1 → (2 * x - y - 1 = 0 ↔ y - 1 = 2 * (x - 1)) := by sorry

-- Theorem for the maximum value of g when a > 0
theorem max_value_g_positive_a :
  ∀ a : ℝ, a > 0 → ∃ max_val : ℝ, max_val = g a (1/a) ∧ 
  ∀ x : ℝ, x > 0 → g a x ≤ max_val := by sorry

-- Theorem for the inequality when a = -2
theorem inequality_a_negative_two :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → 
  f (-2) x₁ + f (-2) x₂ + x₁ * x₂ = 0 → 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_a_zero_max_value_g_positive_a_inequality_a_negative_two_l3874_387433


namespace NUMINAMATH_CALUDE_line_segment_parameterization_l3874_387458

/-- Given a line segment connecting points (1,-3) and (6,12) parameterized by
    x = at + b and y = ct + d where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,-3),
    prove that a + c^2 + b^2 + d^2 = 240 -/
theorem line_segment_parameterization (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 6 ∧ c + d = 12) →
  a + c^2 + b^2 + d^2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_l3874_387458


namespace NUMINAMATH_CALUDE_koi_fish_count_l3874_387477

theorem koi_fish_count : ∃ k : ℕ, (2 * k - 14 = 64) ∧ (k = 39) := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_count_l3874_387477


namespace NUMINAMATH_CALUDE_son_father_distance_l3874_387448

/-- 
Given a lamp post, a father, and his son standing on the same straight line,
with their shadows' heads incident at the same point, prove that the distance
between the son and his father is 4.9 meters.
-/
theorem son_father_distance 
  (lamp_height : ℝ) 
  (father_height : ℝ) 
  (son_height : ℝ) 
  (father_lamp_distance : ℝ) 
  (h_lamp : lamp_height = 6)
  (h_father : father_height = 1.8)
  (h_son : son_height = 0.9)
  (h_father_lamp : father_lamp_distance = 2.1)
  (h_shadows : ∀ x : ℝ, father_height / father_lamp_distance = lamp_height / (father_lamp_distance + x) → 
                        son_height / x = father_height / (father_lamp_distance + x)) :
  ∃ x : ℝ, x = 4.9 ∧ 
    father_height / father_lamp_distance = lamp_height / (father_lamp_distance + x) ∧
    son_height / x = father_height / (father_lamp_distance + x) := by
  sorry


end NUMINAMATH_CALUDE_son_father_distance_l3874_387448


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_per_place_setting_l3874_387489

/-- Calculates the cost per place setting for wedding decorations --/
theorem wedding_decoration_cost_per_place_setting 
  (num_tables : ℕ) 
  (tablecloth_cost : ℕ) 
  (place_settings_per_table : ℕ) 
  (roses_per_centerpiece : ℕ) 
  (rose_cost : ℕ) 
  (lilies_per_centerpiece : ℕ) 
  (lily_cost : ℕ) 
  (total_decoration_cost : ℕ) : 
  num_tables = 20 →
  tablecloth_cost = 25 →
  place_settings_per_table = 4 →
  roses_per_centerpiece = 10 →
  rose_cost = 5 →
  lilies_per_centerpiece = 15 →
  lily_cost = 4 →
  total_decoration_cost = 3500 →
  (total_decoration_cost - 
   (num_tables * tablecloth_cost + 
    num_tables * (roses_per_centerpiece * rose_cost + lilies_per_centerpiece * lily_cost))) / 
   (num_tables * place_settings_per_table) = 10 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_per_place_setting_l3874_387489


namespace NUMINAMATH_CALUDE_monitor_height_is_seven_l3874_387494

/-- Represents a rectangular monitor -/
structure RectangularMonitor where
  width : ℝ
  height : ℝ

/-- The circumference of a rectangular monitor -/
def circumference (m : RectangularMonitor) : ℝ :=
  2 * (m.width + m.height)

/-- Theorem: A rectangular monitor with width 12 cm and circumference 38 cm has a height of 7 cm -/
theorem monitor_height_is_seven :
  ∃ (m : RectangularMonitor), m.width = 12 ∧ circumference m = 38 → m.height = 7 :=
by sorry

end NUMINAMATH_CALUDE_monitor_height_is_seven_l3874_387494


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3874_387426

theorem stewart_farm_sheep_count :
  ∀ (num_sheep num_horses : ℕ),
    num_sheep / num_horses = 2 / 7 →
    num_horses * 230 = 12880 →
    num_sheep = 16 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3874_387426


namespace NUMINAMATH_CALUDE_pizza_solution_l3874_387425

/-- Represents the number of slices in a pizza --/
structure PizzaSlices where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased --/
structure PizzasPurchased where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person --/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  others : ℕ

def pizza_theorem (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) : Prop :=
  slices.small = 4 ∧
  slices.large = 8 ∧
  purchased.small = 3 ∧
  purchased.large = 2 ∧
  eaten.bob = eaten.george + 1 ∧
  eaten.susie = (eaten.bob + 1) / 2 ∧
  eaten.others = 9 ∧
  (slices.small * purchased.small + slices.large * purchased.large) - 
    (eaten.george + eaten.bob + eaten.susie + eaten.others) = 10 →
  eaten.george = 6

theorem pizza_solution : 
  ∃ (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten),
    pizza_theorem slices purchased eaten := by
  sorry

end NUMINAMATH_CALUDE_pizza_solution_l3874_387425


namespace NUMINAMATH_CALUDE_derivative_at_pi_half_l3874_387467

/-- Given a function f where f(x) = sin x + 2x * f'(0), prove that f'(π/2) = -2 -/
theorem derivative_at_pi_half (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin x + 2 * x * (deriv f 0)) :
  deriv f (π/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_half_l3874_387467


namespace NUMINAMATH_CALUDE_tara_yoghurt_purchase_l3874_387428

/-- The number of cartons of yoghurt Tara bought -/
def yoghurt_cartons : ℕ := sorry

/-- The number of cartons of ice cream Tara bought -/
def ice_cream_cartons : ℕ := 19

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 7

/-- The cost of one carton of yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- The difference in dollars between ice cream and yoghurt spending -/
def spending_difference : ℕ := 129

theorem tara_yoghurt_purchase : 
  ice_cream_cartons * ice_cream_cost = 
  yoghurt_cartons * yoghurt_cost + spending_difference ∧ 
  yoghurt_cartons = 4 := by sorry

end NUMINAMATH_CALUDE_tara_yoghurt_purchase_l3874_387428


namespace NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l3874_387486

theorem rectangular_solid_on_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l3874_387486


namespace NUMINAMATH_CALUDE_parabola_tangent_property_fixed_point_property_l3874_387471

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define a point on the axis of the parabola
def point_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Define tangent points
def tangent_points (A B : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ point_on_axis G

-- Define the perpendicular condition
def perpendicular (A M N : ℝ × ℝ) : Prop :=
  (M.1 - A.1) * (N.1 - A.1) + (M.2 - A.2) * (N.2 - A.2) = 0

-- Main theorem
theorem parabola_tangent_property (G : ℝ × ℝ) (A B : ℝ × ℝ) :
  tangent_points A B G → A.1 * B.1 + A.2 * B.2 = -3 :=
sorry

-- Fixed point theorem
theorem fixed_point_property (G A M N : ℝ × ℝ) :
  G.1 = 0 ∧ tangent_points A (2, 1) G ∧ parabola M.1 M.2 ∧ parabola N.1 N.2 ∧ perpendicular A M N →
  ∃ t : ℝ, t * (M.1 - 2) + (1 - t) * (N.1 - 2) = 0 ∧
         t * (M.2 - 5) + (1 - t) * (N.2 - 5) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_property_fixed_point_property_l3874_387471


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l3874_387483

theorem floor_abs_negative_real : ⌊|(-58.7 : ℝ)|⌋ = 58 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l3874_387483


namespace NUMINAMATH_CALUDE_diagonal_angle_is_45_degrees_l3874_387411

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define the angle formed by a diagonal and a side of a square
def diagonal_angle (s : Square) : ℝ := sorry

-- Theorem statement
theorem diagonal_angle_is_45_degrees (s : Square) : 
  diagonal_angle s = 45 := by sorry

end NUMINAMATH_CALUDE_diagonal_angle_is_45_degrees_l3874_387411


namespace NUMINAMATH_CALUDE_all_polyhedra_l3874_387492

-- Define the properties of a polyhedron
structure Polyhedron :=
  (has_flat_faces : Bool)
  (has_straight_edges : Bool)
  (has_sharp_corners : Bool)

-- Define the geometric solids
inductive GeometricSolid
  | TriangularPrism
  | SquareFrustum
  | Cube
  | HexagonalPyramid

-- Function to check if a geometric solid is a polyhedron
def is_polyhedron (solid : GeometricSolid) : Polyhedron :=
  match solid with
  | GeometricSolid.TriangularPrism => ⟨true, true, true⟩
  | GeometricSolid.SquareFrustum => ⟨true, true, true⟩
  | GeometricSolid.Cube => ⟨true, true, true⟩
  | GeometricSolid.HexagonalPyramid => ⟨true, true, true⟩

-- Theorem stating that all the given solids are polyhedra
theorem all_polyhedra :
  (is_polyhedron GeometricSolid.TriangularPrism).has_flat_faces ∧
  (is_polyhedron GeometricSolid.TriangularPrism).has_straight_edges ∧
  (is_polyhedron GeometricSolid.TriangularPrism).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_flat_faces ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_straight_edges ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.Cube).has_flat_faces ∧
  (is_polyhedron GeometricSolid.Cube).has_straight_edges ∧
  (is_polyhedron GeometricSolid.Cube).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_flat_faces ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_straight_edges ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_sharp_corners :=
by sorry

end NUMINAMATH_CALUDE_all_polyhedra_l3874_387492


namespace NUMINAMATH_CALUDE_problem_statement_l3874_387487

/-- Given a function g : ℝ → ℝ satisfying the following conditions:
  1) For all x y : ℝ, 2 * x * g y = 3 * y * g x
  2) g 10 = 15
  Prove that g 5 = 45/4 -/
theorem problem_statement (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x) 
  (h2 : g 10 = 15) : 
  g 5 = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3874_387487


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3874_387461

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x, (d * x + e)^2 + f = 4 * x^2 - 28 * x + 49) →
  d * e = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3874_387461


namespace NUMINAMATH_CALUDE_min_value_theorem_l3874_387472

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + 2*y = 3) :
  ∃ (min_val : ℝ), min_val = 8/3 ∧ ∀ (z : ℝ), z = 1/(x-y) + 9/(x+5*y) → z ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3874_387472


namespace NUMINAMATH_CALUDE_x_minus_y_values_l3874_387430

theorem x_minus_y_values (x y : ℝ) 
  (h1 : |x + 1| = 4)
  (h2 : (y + 2)^2 = 4)
  (h3 : x + y ≥ -5) :
  (x - y = -5) ∨ (x - y = 3) ∨ (x - y = 7) :=
by sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l3874_387430


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3874_387463

theorem quadratic_roots_problem (a b c : ℤ) (h_prime : Prime (a + b + c)) :
  let f : ℤ → ℤ := λ x => a * x * x + b * x + c
  (∃ x y : ℕ, x ≠ y ∧ f x = 0 ∧ f y = 0) →  -- roots are distinct positive integers
  (∃ r : ℕ, f r = -55) →                    -- substituting one root gives -55
  (∃ x y : ℕ, x = 2 ∧ y = 7 ∧ f x = 0 ∧ f y = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3874_387463


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3874_387434

def set_A : Set ℝ := {x | |x| ≤ 1}
def set_B : Set ℝ := {y | ∃ x, y = x^2}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3874_387434


namespace NUMINAMATH_CALUDE_soft_drink_price_l3874_387450

/-- The price increase of a soft drink over 10 years -/
def price_increase (initial_price : ℕ) (increase_5p : ℕ) (increase_2p : ℕ) : ℚ :=
  (initial_price + 5 * increase_5p + 2 * increase_2p) / 100

/-- Theorem stating the final price of the soft drink -/
theorem soft_drink_price :
  price_increase 70 4 6 = 102 / 100 := by sorry

end NUMINAMATH_CALUDE_soft_drink_price_l3874_387450


namespace NUMINAMATH_CALUDE_derivative_even_function_at_zero_l3874_387432

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem derivative_even_function_at_zero (f : ℝ → ℝ) (hf : even_function f) 
  (hf' : Differentiable ℝ f) : 
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_even_function_at_zero_l3874_387432


namespace NUMINAMATH_CALUDE_floor_equation_unique_solution_l3874_387442

theorem floor_equation_unique_solution (n : ℕ+) :
  ∃! (a : ℝ), ∀ (n : ℕ+), 4 * ⌊a * n⌋ = n + ⌊a * ⌊a * n⌋⌋ ∧ a = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_unique_solution_l3874_387442


namespace NUMINAMATH_CALUDE_card_area_after_shortening_l3874_387491

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_after_shortening (initial : Rectangle) :
  initial.length = 6 ∧ initial.width = 8 →
  ∃ (shortened : Rectangle), 
    (shortened.length = initial.length ∧ shortened.width = initial.width - 2 ∧ 
     area shortened = 36) →
    area { length := initial.length - 2, width := initial.width } = 32 := by
  sorry

end NUMINAMATH_CALUDE_card_area_after_shortening_l3874_387491


namespace NUMINAMATH_CALUDE_complex_modulus_equal_parts_l3874_387488

theorem complex_modulus_equal_parts (a : ℝ) :
  let z : ℂ := (1 + 2*I) * (a + I)
  (z.re = z.im) → Complex.abs z = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equal_parts_l3874_387488


namespace NUMINAMATH_CALUDE_min_value_fraction_l3874_387465

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  4/x + 1/y ≥ 6 + 4*Real.sqrt 2 ∧
  (4/x + 1/y = 6 + 4*Real.sqrt 2 ↔ x = 2 - Real.sqrt 2 ∧ y = (Real.sqrt 2 - 1)/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3874_387465


namespace NUMINAMATH_CALUDE_rectangle_area_l3874_387408

theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ 
  l = 3 * w ∧ 
  w ^ 2 + l ^ 2 = x ^ 2 ∧
  w * l = (3 / 10) * x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3874_387408


namespace NUMINAMATH_CALUDE_three_balls_four_boxes_l3874_387414

theorem three_balls_four_boxes :
  let num_balls : ℕ := 3
  let num_boxes : ℕ := 4
  num_boxes ^ num_balls = 64 :=
by sorry

end NUMINAMATH_CALUDE_three_balls_four_boxes_l3874_387414


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l3874_387412

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of unique ice cream flavors that can be created -/
def ice_cream_flavors : ℕ := stars_and_bars 4 4

theorem ice_cream_flavors_count : ice_cream_flavors = 35 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l3874_387412


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l3874_387429

theorem point_between_parallel_lines :
  ∃ (b : ℤ),
    (31 - 8 * b) * (20 - 4 * b) < 0 ∧
    b = 4 :=
by sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l3874_387429


namespace NUMINAMATH_CALUDE_triangle_properties_l3874_387478

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle)
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c)
  (h2 : t.a = Real.sqrt 2)
  (h3 : Real.sin t.B * Real.sin t.C = (Real.sin t.A)^2) :
  t.A = π/3 ∧ (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3874_387478


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3874_387474

theorem complex_equation_solution (z : ℂ) : (2 + z) / (2 - z) = I → z = 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3874_387474
