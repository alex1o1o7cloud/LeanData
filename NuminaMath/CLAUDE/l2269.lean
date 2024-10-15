import Mathlib

namespace NUMINAMATH_CALUDE_cos_18_degrees_l2269_226986

theorem cos_18_degrees : Real.cos (18 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l2269_226986


namespace NUMINAMATH_CALUDE_score_difference_l2269_226958

def blue_free_throws : ℕ := 18
def blue_two_pointers : ℕ := 25
def blue_three_pointers : ℕ := 6

def red_free_throws : ℕ := 15
def red_two_pointers : ℕ := 22
def red_three_pointers : ℕ := 5

def blue_score : ℕ := blue_free_throws + 2 * blue_two_pointers + 3 * blue_three_pointers
def red_score : ℕ := red_free_throws + 2 * red_two_pointers + 3 * red_three_pointers

theorem score_difference : blue_score - red_score = 12 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l2269_226958


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2269_226979

theorem quadratic_equation_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (x^2 + 10*x = 45 ∧ x = Real.sqrt a - b) → a + b = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2269_226979


namespace NUMINAMATH_CALUDE_circle_tangency_l2269_226998

/-- Circle C with equation x^2 + y^2 - 2x - 4y + m = 0 -/
def circle_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + m = 0}

/-- Circle D with equation (x+2)^2 + (y+2)^2 = 1 -/
def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 + 2)^2 = 1}

/-- The number of common tangents between two circles -/
def common_tangents (C D : Set (ℝ × ℝ)) : ℕ := sorry

theorem circle_tangency (m : ℝ) :
  common_tangents (circle_C m) circle_D = 3 → m = -11 := by sorry

end NUMINAMATH_CALUDE_circle_tangency_l2269_226998


namespace NUMINAMATH_CALUDE_sum_of_square_roots_l2269_226915

theorem sum_of_square_roots : 
  Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + 
  Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_l2269_226915


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2269_226910

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 3) (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ ≠ 1 - a) :
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2269_226910


namespace NUMINAMATH_CALUDE_vote_count_theorem_l2269_226903

/-- The number of ways to count votes such that candidate A always leads candidate B -/
def vote_count_ways (a b : ℕ) : ℕ :=
  (Nat.factorial (a + b - 1)) / (Nat.factorial (a - 1) * Nat.factorial b) -
  (Nat.factorial (a + b - 1)) / (Nat.factorial a * Nat.factorial (b - 1))

/-- Theorem stating the number of ways for candidate A to maintain a lead throughout the counting process -/
theorem vote_count_theorem (a b : ℕ) (h : a > b) :
  vote_count_ways a b = (Nat.factorial (a + b - 1)) / (Nat.factorial (a - 1) * Nat.factorial b) -
                        (Nat.factorial (a + b - 1)) / (Nat.factorial a * Nat.factorial (b - 1)) :=
by sorry

end NUMINAMATH_CALUDE_vote_count_theorem_l2269_226903


namespace NUMINAMATH_CALUDE_longest_perimeter_l2269_226993

theorem longest_perimeter (x : ℝ) 
  (hx : x > 1)
  (perimeterA : ℝ := 4 + 6*x)
  (perimeterB : ℝ := 2 + 10*x)
  (perimeterC : ℝ := 7 + 5*x)
  (perimeterD : ℝ := 6 + 6*x)
  (perimeterE : ℝ := 1 + 11*x) :
  perimeterE > perimeterA ∧ 
  perimeterE > perimeterB ∧ 
  perimeterE > perimeterC ∧ 
  perimeterE > perimeterD :=
by
  sorry

end NUMINAMATH_CALUDE_longest_perimeter_l2269_226993


namespace NUMINAMATH_CALUDE_f_max_value_l2269_226916

/-- The function f(x) = 3x - x^3 -/
def f (x : ℝ) : ℝ := 3 * x - x^3

/-- The theorem stating that the maximum value of f(x) = 3x - x^3 is 2 for 0 ≤ x ≤ √3 -/
theorem f_max_value :
  ∃ (M : ℝ), M = 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 ∧ f x = M) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2269_226916


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l2269_226953

def a : ℝ × ℝ := (-5, 6)
def b : ℝ × ℝ := (6, 5)

theorem vectors_perpendicular : a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l2269_226953


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2269_226999

/-- 
Theorem: If a line y = kx + 2 is tangent to the ellipse x^2/2 + 2y^2 = 2, 
then k^2 = 3/4.
-/
theorem line_tangent_to_ellipse (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2 → x^2 / 2 + 2 * y^2 = 2) →
  (∃! p : ℝ × ℝ, p.1^2 / 2 + 2 * p.2^2 = 2 ∧ p.2 = k * p.1 + 2) →
  k^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2269_226999


namespace NUMINAMATH_CALUDE_cos_shift_proof_l2269_226918

theorem cos_shift_proof (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 2) : 
  let f := λ x : Real => 2 * Real.cos (2 * x)
  let g := λ x : Real => 2 * Real.cos (2 * x - 2 * φ)
  (∃ x₁ x₂ : Real, |f x₁ - g x₂| = 4 ∧ |x₁ - x₂| = π / 6) → φ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_proof_l2269_226918


namespace NUMINAMATH_CALUDE_quadratic_c_value_l2269_226927

/-- The quadratic function f(x) = -x^2 + cx + 8 is positive only on the open interval (2,6) -/
def quadratic_positive_on_interval (c : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + c*x + 8 > 0) ↔ (2 < x ∧ x < 6)

/-- The value of c for which the quadratic function is positive only on (2,6) is 8 -/
theorem quadratic_c_value : ∃! c : ℝ, quadratic_positive_on_interval c ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_c_value_l2269_226927


namespace NUMINAMATH_CALUDE_parallelogram_angle_E_l2269_226955

structure Parallelogram :=
  (E F G H : Point)

def angle_FGH (p : Parallelogram) : ℝ := sorry
def angle_E (p : Parallelogram) : ℝ := sorry

theorem parallelogram_angle_E (p : Parallelogram) 
  (h : angle_FGH p = 70) : angle_E p = 110 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_E_l2269_226955


namespace NUMINAMATH_CALUDE_jerry_gabriel_toy_difference_l2269_226989

theorem jerry_gabriel_toy_difference (jerry gabriel jaxon : ℕ) 
  (h1 : jerry > gabriel)
  (h2 : gabriel = 2 * jaxon)
  (h3 : jaxon = 15)
  (h4 : jerry + gabriel + jaxon = 83) :
  jerry - gabriel = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_gabriel_toy_difference_l2269_226989


namespace NUMINAMATH_CALUDE_interest_rate_is_one_percent_l2269_226984

/-- Calculate the interest rate given principal, time, and total simple interest -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that given the specific values, the interest rate is 1% -/
theorem interest_rate_is_one_percent :
  let principal : ℚ := 133875
  let time : ℚ := 3
  let total_interest : ℚ := 4016.25
  calculate_interest_rate principal time total_interest = 1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_one_percent_l2269_226984


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2269_226935

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 34 = 17 % 34 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 34 = 17 % 34 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2269_226935


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_126_l2269_226992

theorem percentage_of_360_equals_126 : 
  (126 : ℝ) / 360 * 100 = 35 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_126_l2269_226992


namespace NUMINAMATH_CALUDE_marble_ratio_l2269_226994

theorem marble_ratio (selma_marbles merill_marbles elliot_marbles : ℕ) : 
  selma_marbles = 50 →
  merill_marbles = 30 →
  merill_marbles + elliot_marbles = selma_marbles - 5 →
  merill_marbles / elliot_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l2269_226994


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2269_226924

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2269_226924


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_12_l2269_226944

theorem x_plus_2y_equals_12 (x y : ℝ) (h1 : x = 6) (h2 : y = 3) : x + 2*y = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_12_l2269_226944


namespace NUMINAMATH_CALUDE_bike_vs_drive_time_difference_l2269_226988

theorem bike_vs_drive_time_difference 
  (normal_drive_time : ℝ) 
  (normal_drive_speed : ℝ) 
  (bike_route_reduction : ℝ) 
  (min_bike_speed : ℝ) 
  (max_bike_speed : ℝ) 
  (h1 : normal_drive_time = 45) 
  (h2 : normal_drive_speed = 40) 
  (h3 : bike_route_reduction = 0.2) 
  (h4 : min_bike_speed = 12) 
  (h5 : max_bike_speed = 16) : 
  ∃ (time_difference : ℝ), time_difference = 75 := by
sorry

end NUMINAMATH_CALUDE_bike_vs_drive_time_difference_l2269_226988


namespace NUMINAMATH_CALUDE_perpendicular_diagonals_imply_square_rectangle_is_not_square_l2269_226963

-- Define a quadrilateral
structure Quadrilateral :=
  (has_right_angles : Bool)
  (opposite_sides_parallel_equal : Bool)
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)

-- Define a rectangle
def Rectangle : Quadrilateral :=
  { has_right_angles := true,
    opposite_sides_parallel_equal := true,
    diagonals_bisect := true,
    diagonals_perpendicular := false }

-- Define a square
def Square : Quadrilateral :=
  { has_right_angles := true,
    opposite_sides_parallel_equal := true,
    diagonals_bisect := true,
    diagonals_perpendicular := true }

-- Theorem: A quadrilateral with right angles, opposite sides parallel and equal,
-- and perpendicular diagonals that bisect each other is a square
theorem perpendicular_diagonals_imply_square (q : Quadrilateral) :
  q.has_right_angles = true →
  q.opposite_sides_parallel_equal = true →
  q.diagonals_bisect = true →
  q.diagonals_perpendicular = true →
  q = Square := by
  sorry

-- Theorem: A rectangle is not a square
theorem rectangle_is_not_square : Rectangle ≠ Square := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_diagonals_imply_square_rectangle_is_not_square_l2269_226963


namespace NUMINAMATH_CALUDE_simplify_expression_l2269_226991

theorem simplify_expression :
  ∃ (C : ℝ), C = 2^(1 + Real.sqrt 2) ∧
  (Real.sqrt 3 - 1)^(1 - Real.sqrt 2) / (Real.sqrt 3 + 1)^(1 + Real.sqrt 2) = (4 - 2 * Real.sqrt 3) / C :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2269_226991


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2269_226985

theorem sum_of_x_and_y (x y : ℝ) (hx : x + 2 = 10) (hy : y - 1 = 6) : x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2269_226985


namespace NUMINAMATH_CALUDE_a5_value_l2269_226964

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a5_value (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) 
  (h3 : a 3 = -5) (h7 : a 7 = -1) : a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a5_value_l2269_226964


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2269_226952

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0) ↔ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2269_226952


namespace NUMINAMATH_CALUDE_total_removed_volume_l2269_226945

/-- The edge length of the cube -/
def cube_edge : ℝ := 2

/-- The number of sides in the resulting polygon on each face after slicing -/
def hexadecagon_sides : ℕ := 16

/-- The volume of a single removed tetrahedron -/
noncomputable def tetrahedron_volume : ℝ := 
  let y := 2 * (Real.sqrt 2 - 1)
  let height := 3 - 2 * Real.sqrt 2
  let base_area := (1 / 2) * ((2 - Real.sqrt 2) ^ 2)
  (1 / 3) * base_area * height

/-- The number of corners in a cube -/
def cube_corners : ℕ := 8

/-- Theorem stating the total volume of removed tetrahedra -/
theorem total_removed_volume : 
  cube_corners * tetrahedron_volume = -64 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_removed_volume_l2269_226945


namespace NUMINAMATH_CALUDE_allison_bought_28_items_l2269_226942

/-- The number of craft supply items Allison bought -/
def allison_total (marie_glue : ℕ) (marie_paper : ℕ) (glue_diff : ℕ) (paper_ratio : ℕ) : ℕ :=
  (marie_glue + glue_diff) + (marie_paper / paper_ratio)

/-- Theorem stating the total number of craft supply items Allison bought -/
theorem allison_bought_28_items : allison_total 15 30 8 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_allison_bought_28_items_l2269_226942


namespace NUMINAMATH_CALUDE_average_increase_is_4_l2269_226987

/-- Represents the cricketer's score data -/
structure CricketerScore where
  runs_19th_inning : ℕ
  average_after_19 : ℚ

/-- Calculates the increase in average score -/
def average_increase (score : CricketerScore) : ℚ :=
  let total_runs := score.average_after_19 * 19
  let runs_before_19th := total_runs - score.runs_19th_inning
  let average_before_19th := runs_before_19th / 18
  score.average_after_19 - average_before_19th

/-- Theorem stating the increase in average score -/
theorem average_increase_is_4 (score : CricketerScore) 
  (h1 : score.runs_19th_inning = 96)
  (h2 : score.average_after_19 = 24) :
  average_increase score = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_is_4_l2269_226987


namespace NUMINAMATH_CALUDE_midpoint_locus_l2269_226941

/-- The locus of midpoints between a fixed point and points on a circle -/
theorem midpoint_locus (P : ℝ × ℝ) (c : Set (ℝ × ℝ)) :
  P = (4, -2) →
  c = {(x, y) | x^2 + y^2 = 4} →
  {(x, y) | ∃ (a : ℝ × ℝ), a ∈ c ∧ (x, y) = ((P.1 + a.1) / 2, (P.2 + a.2) / 2)} =
  {(x, y) | (x - 2)^2 + (y + 1)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l2269_226941


namespace NUMINAMATH_CALUDE_train_length_l2269_226959

/-- Proves that a train with the given conditions has a length of 300 meters -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length platform_length : ℝ) : 
  train_speed = 36 * 5/18 → -- Convert 36 km/hr to m/s
  crossing_time = 60 → -- One minute in seconds
  train_length = platform_length →
  train_length + platform_length = train_speed * crossing_time →
  train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2269_226959


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2269_226934

def f (x : ℝ) : ℝ := -x^3 - x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2269_226934


namespace NUMINAMATH_CALUDE_total_pencils_l2269_226907

/-- Given that each child has 2 pencils and there are 11 children, 
    prove that the total number of pencils is 22. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) 
  (h2 : num_children = 11) : 
  pencils_per_child * num_children = 22 := by
sorry

end NUMINAMATH_CALUDE_total_pencils_l2269_226907


namespace NUMINAMATH_CALUDE_shopkeeper_sales_l2269_226931

/-- The number of articles sold by a shopkeeper -/
def articles_sold (cost_price : ℝ) : ℕ :=
  72

/-- The profit percentage made by the shopkeeper -/
def profit_percentage : ℝ :=
  20

/-- The number of articles whose cost price equals the selling price -/
def equivalent_articles : ℕ :=
  60

theorem shopkeeper_sales :
  ∀ (cost_price : ℝ),
  cost_price > 0 →
  (articles_sold cost_price : ℝ) * cost_price =
    equivalent_articles * cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_sales_l2269_226931


namespace NUMINAMATH_CALUDE_stating_not_always_two_triangles_form_rectangle_l2269_226943

/-- Represents a non-isosceles right triangle -/
structure NonIsoscelesRightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  leg1_ne_leg2 : leg1 ≠ leg2
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- Represents a rectangle constructed from non-isosceles right triangles -/
structure RectangleFromTriangles where
  width : ℝ
  height : ℝ
  triangle : NonIsoscelesRightTriangle
  num_triangles : ℕ
  area_equality : width * height = num_triangles * (triangle.leg1 * triangle.leg2 / 2)

/-- 
Theorem stating that it's not always necessary for any two identical non-isosceles 
right triangles to form a rectangle when a larger rectangle is constructed from 
these triangles without gaps or overlaps
-/
theorem not_always_two_triangles_form_rectangle 
  (r : RectangleFromTriangles) : 
  ¬ ∀ (t1 t2 : NonIsoscelesRightTriangle), 
    t1 = r.triangle → t2 = r.triangle → 
    ∃ (w h : ℝ), w * h = t1.leg1 * t1.leg2 + t2.leg1 * t2.leg2 := by
  sorry

end NUMINAMATH_CALUDE_stating_not_always_two_triangles_form_rectangle_l2269_226943


namespace NUMINAMATH_CALUDE_pencils_sold_initially_l2269_226921

-- Define the number of pencils sold at 15% gain
def pencils_at_gain : ℝ := 7.391304347826086

-- Theorem statement
theorem pencils_sold_initially (x : ℝ) :
  (0.85 * x * (1 / (1.15 * pencils_at_gain)) = 1) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencils_sold_initially_l2269_226921


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2269_226932

theorem second_discount_percentage 
  (initial_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (x : ℝ) :
  initial_price = 1000 →
  first_discount = 15 →
  final_price = 830 →
  initial_price * (1 - first_discount / 100) * (1 - x / 100) = final_price :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2269_226932


namespace NUMINAMATH_CALUDE_refrigerator_installation_cost_l2269_226961

/-- Calculates the installation cost for a refrigerator sale --/
theorem refrigerator_installation_cost 
  (purchased_price : ℚ) 
  (discount_rate : ℚ) 
  (transport_cost : ℚ) 
  (profit_rate : ℚ) 
  (final_selling_price : ℚ) 
  (h1 : purchased_price = 12500)
  (h2 : discount_rate = 1/5)
  (h3 : transport_cost = 125)
  (h4 : profit_rate = 4/25)
  (h5 : final_selling_price = 18560) : 
  ∃ (installation_cost : ℚ), installation_cost = 310 := by
  sorry


end NUMINAMATH_CALUDE_refrigerator_installation_cost_l2269_226961


namespace NUMINAMATH_CALUDE_inverse_f_27_equals_3_l2269_226940

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_f_27_equals_3 :
  ∀ f_inv : ℝ → ℝ, 
  (∀ x : ℝ, f_inv (f x) = x) ∧ (∀ y : ℝ, f (f_inv y) = y) → 
  f_inv 27 = 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_f_27_equals_3_l2269_226940


namespace NUMINAMATH_CALUDE_sphere_in_cylindrical_hole_l2269_226997

theorem sphere_in_cylindrical_hole (r : ℝ) (h : ℝ) :
  h = 2 ∧ 
  6^2 + (r - h)^2 = r^2 →
  r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cylindrical_hole_l2269_226997


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2269_226983

/-- Given a geometric sequence {a_n} with sum S_n = b(-2)^(n-1) - a, prove that a/b = -1/2 -/
theorem geometric_sequence_ratio (b a : ℝ) (S : ℕ → ℝ) (a_n : ℕ → ℝ) :
  (∀ n : ℕ, S n = b * (-2)^(n - 1) - a) →
  (∀ n : ℕ, a_n n = S n - S (n - 1)) →
  (a_n 1 = b - a) →
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2269_226983


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l2269_226956

def newspaper_earnings : ℕ := 16
def car_washing_earnings : ℕ := 74
def lawn_mowing_earnings : ℕ := 45
def lemonade_earnings : ℕ := 22
def yard_work_earnings : ℕ := 30

theorem fred_weekend_earnings :
  newspaper_earnings + car_washing_earnings + lawn_mowing_earnings + lemonade_earnings + yard_work_earnings = 187 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l2269_226956


namespace NUMINAMATH_CALUDE_convex_polygon_diagonals_l2269_226954

-- Define a convex polygon type
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool
  interior_angle : ℝ
  all_angles_equal : Bool

-- Theorem statement
theorem convex_polygon_diagonals 
  (p : ConvexPolygon) 
  (h1 : p.is_convex = true) 
  (h2 : p.interior_angle = 150) 
  (h3 : p.all_angles_equal = true) : 
  (p.sides * (p.sides - 3)) / 2 = 54 := by
sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonals_l2269_226954


namespace NUMINAMATH_CALUDE_sandy_dozens_of_marbles_l2269_226901

def melanie_marbles : ℕ := 84
def sandy_multiplier : ℕ := 8
def marbles_per_dozen : ℕ := 12

theorem sandy_dozens_of_marbles :
  (melanie_marbles * sandy_multiplier) / marbles_per_dozen = 56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_dozens_of_marbles_l2269_226901


namespace NUMINAMATH_CALUDE_westeros_max_cursed_roads_l2269_226996

/-- A graph representing the Westeros Empire -/
structure WesterosGraph where
  /-- The number of cities (vertices) in the graph -/
  num_cities : Nat
  /-- The number of roads (edges) in the graph -/
  num_roads : Nat
  /-- The graph is initially connected -/
  is_connected : Bool
  /-- The number of kingdoms formed after cursing some roads -/
  num_kingdoms : Nat

/-- The maximum number of roads that can be cursed -/
def max_cursed_roads (g : WesterosGraph) : Nat :=
  g.num_roads - (g.num_cities - g.num_kingdoms)

/-- Theorem stating the maximum number of roads that can be cursed -/
theorem westeros_max_cursed_roads (g : WesterosGraph) 
  (h1 : g.num_cities = 1000)
  (h2 : g.num_roads = 2017)
  (h3 : g.is_connected = true)
  (h4 : g.num_kingdoms = 7) :
  max_cursed_roads g = 1024 := by
  sorry

#eval max_cursed_roads { num_cities := 1000, num_roads := 2017, is_connected := true, num_kingdoms := 7 }

end NUMINAMATH_CALUDE_westeros_max_cursed_roads_l2269_226996


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l2269_226922

-- Define the types for our geometric objects
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (belongs_to : Point → Line → Prop)
variable (subset_of : Line → Plane → Prop)
variable (intersect_along : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define our specific objects
variable (α β : Plane) (l a b : Line)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (h1 : intersect_along α β l)
  (h2 : subset_of a α)
  (h3 : subset_of b β)
  (h4 : perpendicular a l)
  (h5 : perpendicular b l) :
  ¬(plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l2269_226922


namespace NUMINAMATH_CALUDE_cone_base_radius_l2269_226966

-- Define the surface area of the cone
def surface_area (a : ℝ) : ℝ := a

-- Define the property that the lateral surface unfolds into a semicircle
def lateral_surface_is_semicircle (r l : ℝ) : Prop := 2 * Real.pi * r = Real.pi * l

-- Theorem statement
theorem cone_base_radius (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
    (∃ (l : ℝ), l > 0 ∧ 
      lateral_surface_is_semicircle r l ∧ 
      surface_area a = Real.pi * r^2 + Real.pi * r * l) ∧
    r = Real.sqrt (a / (3 * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2269_226966


namespace NUMINAMATH_CALUDE_complex_power_equivalence_l2269_226928

theorem complex_power_equivalence :
  (Complex.exp (Complex.I * Real.pi * (35 / 180)))^100 = Complex.exp (Complex.I * Real.pi * (20 / 180)) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equivalence_l2269_226928


namespace NUMINAMATH_CALUDE_monotonic_function_theorem_l2269_226977

/-- A monotonic function is either non-increasing or non-decreasing --/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f x ≥ f y)

/-- The main theorem --/
theorem monotonic_function_theorem (f : ℝ → ℝ) (hf : Monotonic f)
    (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) :
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = -x) := by
  sorry


end NUMINAMATH_CALUDE_monotonic_function_theorem_l2269_226977


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2269_226973

theorem repeating_decimal_to_fraction : ∃ (x : ℚ), x = 4/11 ∧ (∀ (n : ℕ), x = (36 * (100^n - 1)) / (99 * 100^n)) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2269_226973


namespace NUMINAMATH_CALUDE_complex_multiplication_l2269_226920

theorem complex_multiplication (i : ℂ) :
  i^2 = -1 →
  (6 - 7*i) * (3 + 6*i) = 60 + 15*i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2269_226920


namespace NUMINAMATH_CALUDE_power_of_two_sum_l2269_226978

theorem power_of_two_sum (m n : ℕ+) (a b : ℝ) 
  (h1 : 2^(m : ℕ) = a) 
  (h2 : 2^(n : ℕ) = b) : 
  2^((m + n : ℕ+) : ℕ) = a * b := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l2269_226978


namespace NUMINAMATH_CALUDE_order_of_abc_l2269_226975

theorem order_of_abc : 
  let a : ℝ := Real.rpow 0.9 (1/3)
  let b : ℝ := Real.rpow (1/3) 0.9
  let c : ℝ := (1/2) * (Real.log 9 / Real.log 27)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2269_226975


namespace NUMINAMATH_CALUDE_f_of_2_equals_7_l2269_226919

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + 2*x - 1

-- State the theorem
theorem f_of_2_equals_7 : f 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_7_l2269_226919


namespace NUMINAMATH_CALUDE_line_perp_plane_and_line_implies_parallel_l2269_226933

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_and_line_implies_parallel
  (l m : Line) (α : Plane)
  (h1 : l ≠ m)
  (h2 : perpToPlane l α)
  (h3 : perp l m) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_and_line_implies_parallel_l2269_226933


namespace NUMINAMATH_CALUDE_equation_solution_l2269_226914

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), x₁ = 1/3 ∧ x₂ = 1/2 ∧ 
  (∀ x : ℚ, 6*x^2 - 3*x - 1 = 2*x - 2 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2269_226914


namespace NUMINAMATH_CALUDE_unanswered_test_theorem_l2269_226930

/-- The number of ways to complete an unanswered multiple-choice test -/
def unanswered_test_completions (num_questions : ℕ) (num_choices : ℕ) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question, 
    there is only one way to complete it if all questions are unanswered -/
theorem unanswered_test_theorem : 
  unanswered_test_completions 4 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_theorem_l2269_226930


namespace NUMINAMATH_CALUDE_mixture_composition_l2269_226909

theorem mixture_composition 
  (p_carbonated : ℝ) 
  (q_carbonated : ℝ) 
  (mixture_carbonated : ℝ) 
  (h1 : p_carbonated = 0.80) 
  (h2 : q_carbonated = 0.55) 
  (h3 : mixture_carbonated = 0.72) :
  let p := (mixture_carbonated - q_carbonated) / (p_carbonated - q_carbonated)
  p = 0.68 := by sorry

end NUMINAMATH_CALUDE_mixture_composition_l2269_226909


namespace NUMINAMATH_CALUDE_find_m_l2269_226939

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, mx + 3 = x ↔ 5 - 2*x = 1) → m = -1/2 := by sorry

end NUMINAMATH_CALUDE_find_m_l2269_226939


namespace NUMINAMATH_CALUDE_equation_solution_l2269_226967

theorem equation_solution : ∃! y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ∧ y = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2269_226967


namespace NUMINAMATH_CALUDE_final_clothing_count_l2269_226947

/-- Calculate the remaining clothes after donations and purchases -/
def remaining_clothes (initial : ℕ) : ℕ :=
  let after_orphanages := initial - (initial / 10 + 3 * (initial / 10))
  let after_shelter := after_orphanages - (after_orphanages / 5)
  let after_purchase := after_shelter + (after_shelter / 5)
  after_purchase - (after_purchase / 8)

/-- Theorem stating the final number of clothing pieces -/
theorem final_clothing_count :
  remaining_clothes 500 = 252 := by
  sorry

end NUMINAMATH_CALUDE_final_clothing_count_l2269_226947


namespace NUMINAMATH_CALUDE_pen_measurement_properties_l2269_226995

def measured_length : Float := 0.06250

-- Function to count significant figures
def count_significant_figures (x : Float) : Nat :=
  sorry

-- Function to determine the place of accuracy
def place_of_accuracy (x : Float) : String :=
  sorry

theorem pen_measurement_properties :
  (count_significant_figures measured_length = 4) ∧
  (place_of_accuracy measured_length = "hundred-thousandth") :=
by sorry

end NUMINAMATH_CALUDE_pen_measurement_properties_l2269_226995


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2269_226990

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_ecc : e = 5/3

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a hyperbola and a point, check if the point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Given a hyperbola, return its foci -/
def foci (h : Hyperbola) : (Point × Point) :=
  let c := h.a * h.e
  (Point.mk (-c) 0, Point.mk c 0)

/-- Given two points, check if they are perpendicular with respect to the origin -/
def are_perpendicular (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

/-- Main theorem -/
theorem hyperbola_equation (h : Hyperbola) (p : Point) :
  let (f1, f2) := foci h
  (is_on_hyperbola h p) ∧
  (p.x = -3 ∧ p.y = -4) ∧
  (are_perpendicular (Point.mk (p.x - f1.x) (p.y - f1.y)) (Point.mk (p.x - f2.x) (p.y - f2.y))) →
  h.a = 3 ∧ h.b = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2269_226990


namespace NUMINAMATH_CALUDE_not_both_odd_l2269_226929

theorem not_both_odd (m n : ℕ) (h : (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020) : 
  ¬(Odd m ∧ Odd n) := by
sorry

end NUMINAMATH_CALUDE_not_both_odd_l2269_226929


namespace NUMINAMATH_CALUDE_antonio_age_is_51_months_l2269_226968

-- Define Isabella's age in months after 18 months
def isabella_age_after_18_months : ℕ := 10 * 12

-- Define the current time difference in months
def time_difference : ℕ := 18

-- Define Isabella's current age in months
def isabella_current_age : ℕ := isabella_age_after_18_months - time_difference

-- Define the relationship between Isabella and Antonio's ages
def antonio_age : ℕ := isabella_current_age / 2

-- Theorem to prove
theorem antonio_age_is_51_months : antonio_age = 51 := by
  sorry


end NUMINAMATH_CALUDE_antonio_age_is_51_months_l2269_226968


namespace NUMINAMATH_CALUDE_max_car_distance_l2269_226900

/-- Represents the maximum distance a car can travel with one tire swap -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  front_tire_life + min front_tire_life rear_tire_life

/-- Theorem stating the maximum distance a car can travel with given tire lifespans -/
theorem max_car_distance :
  let front_tire_life : ℕ := 24000
  let rear_tire_life : ℕ := 36000
  max_distance front_tire_life rear_tire_life = 28800 := by
  sorry

#eval max_distance 24000 36000

end NUMINAMATH_CALUDE_max_car_distance_l2269_226900


namespace NUMINAMATH_CALUDE_candy_cost_620_l2269_226951

/-- Calculates the cost of buying candies given the pricing structure -/
def candy_cost (total_candies : ℕ) : ℕ :=
  let regular_price := 8
  let discount_price := 7
  let candies_per_box := 40
  let discount_threshold := 500
  let full_price_boxes := min (total_candies / candies_per_box) (discount_threshold / candies_per_box)
  let discounted_boxes := (total_candies - full_price_boxes * candies_per_box + candies_per_box - 1) / candies_per_box
  full_price_boxes * regular_price + discounted_boxes * discount_price

theorem candy_cost_620 : candy_cost 620 = 125 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_620_l2269_226951


namespace NUMINAMATH_CALUDE_lydia_road_trip_fuel_usage_l2269_226949

/-- Proves that given the conditions of Lydia's road trip, the fraction of fuel used in the second third is 1/3 --/
theorem lydia_road_trip_fuel_usage 
  (total_fuel : ℝ) 
  (first_third_fuel : ℝ) 
  (h1 : total_fuel = 60) 
  (h2 : first_third_fuel = 30) 
  (h3 : ∃ (second_third_fraction : ℝ), 
    first_third_fuel + second_third_fraction * total_fuel + (second_third_fraction / 2) * total_fuel = total_fuel) :
  ∃ (second_third_fraction : ℝ), second_third_fraction = 1/3 := by
sorry


end NUMINAMATH_CALUDE_lydia_road_trip_fuel_usage_l2269_226949


namespace NUMINAMATH_CALUDE_expression_simplification_l2269_226926

theorem expression_simplification (a : ℝ) (h : a^2 + 4*a + 1 = 0) :
  ((a + 2) / (a^2 - 2*a) + 8 / (4 - a^2)) / ((a^2 - 4) / a) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2269_226926


namespace NUMINAMATH_CALUDE_ferry_time_difference_l2269_226962

/-- Proves that the difference in travel time between Ferry Q and Ferry P is 1 hour -/
theorem ferry_time_difference
  (time_p : ℝ) (speed_p : ℝ) (speed_difference : ℝ) (route_factor : ℝ) :
  time_p = 3 →
  speed_p = 8 →
  speed_difference = 4 →
  route_factor = 2 →
  let distance_p := time_p * speed_p
  let distance_q := route_factor * distance_p
  let speed_q := speed_p + speed_difference
  let time_q := distance_q / speed_q
  time_q - time_p = 1 := by
sorry

end NUMINAMATH_CALUDE_ferry_time_difference_l2269_226962


namespace NUMINAMATH_CALUDE_velvet_area_for_given_box_l2269_226936

/-- The total area of velvet needed to line the inside of a box with given dimensions -/
def total_velvet_area (long_side_length long_side_width short_side_length short_side_width top_bottom_area : ℕ) : ℕ :=
  2 * (long_side_length * long_side_width) +
  2 * (short_side_length * short_side_width) +
  2 * top_bottom_area

/-- Theorem stating that the total area of velvet needed for the given box dimensions is 236 square inches -/
theorem velvet_area_for_given_box : total_velvet_area 8 6 5 6 40 = 236 := by
  sorry

end NUMINAMATH_CALUDE_velvet_area_for_given_box_l2269_226936


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2269_226948

theorem cos_2alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.cos (2*α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2269_226948


namespace NUMINAMATH_CALUDE_bug_probability_after_8_steps_l2269_226913

/-- Probability of being at vertex A after n steps -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 - P (n - 1)) / 3

/-- The probability of being at vertex A after 8 steps in a regular tetrahedron -/
theorem bug_probability_after_8_steps :
  P 8 = 547 / 2187 := by sorry

end NUMINAMATH_CALUDE_bug_probability_after_8_steps_l2269_226913


namespace NUMINAMATH_CALUDE_percentage_problem_l2269_226937

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (50 / 100) * 960 → P = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2269_226937


namespace NUMINAMATH_CALUDE_all_propositions_false_l2269_226969

-- Define the concept of skew lines
def are_skew (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of perpendicular lines
def is_perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of intersecting lines
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of lines in different planes
def in_different_planes (l1 l2 : Line3D) : Prop := sorry

theorem all_propositions_false :
  (∀ l1 l2 : Line3D, in_different_planes l1 l2 → are_skew l1 l2) = False ∧
  (∃! l : Line3D, ∀ l1 l2 : Line3D, are_skew l1 l2 → is_perpendicular l l1 ∧ is_perpendicular l l2) = False ∧
  (∀ l1 l2 l3 l4 : Line3D, are_skew l1 l2 → intersect l3 l1 → intersect l3 l2 → intersect l4 l1 → intersect l4 l2 → are_skew l3 l4) = False ∧
  (∀ a b c : Line3D, are_skew a b → are_skew b c → are_skew a c) = False :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2269_226969


namespace NUMINAMATH_CALUDE_total_amount_correct_l2269_226974

/-- The amount of money Mrs. Hilt needs to share -/
def total_amount : ℝ := 3.75

/-- The number of people sharing the money -/
def number_of_people : ℕ := 3

/-- The amount each person receives -/
def amount_per_person : ℝ := 1.25

/-- Theorem stating that the total amount is correct given the conditions -/
theorem total_amount_correct : 
  total_amount = (number_of_people : ℝ) * amount_per_person :=
by sorry

end NUMINAMATH_CALUDE_total_amount_correct_l2269_226974


namespace NUMINAMATH_CALUDE_three_gorges_dam_capacity_l2269_226946

theorem three_gorges_dam_capacity :
  (16780000 : ℝ) = 1.678 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_three_gorges_dam_capacity_l2269_226946


namespace NUMINAMATH_CALUDE_factorial_sum_l2269_226957

theorem factorial_sum : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 6 * Nat.factorial 5 = 36600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_l2269_226957


namespace NUMINAMATH_CALUDE_proposition_2_l2269_226917

-- Define the basic types
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the proposition we want to prove
theorem proposition_2 
  (m n : Line) (α : Plane) :
  perpendicular_plane m α → parallel m n → perpendicular_plane n α :=
sorry

end NUMINAMATH_CALUDE_proposition_2_l2269_226917


namespace NUMINAMATH_CALUDE_exists_closer_vertex_l2269_226925

-- Define a convex polygon
def ConvexPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point being inside a polygon
def InsidePolygon (p : ℝ × ℝ) (polygon : Set (ℝ × ℝ)) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem exists_closer_vertex 
  (vertices : Set (ℝ × ℝ)) 
  (P Q : ℝ × ℝ) 
  (h_convex : ConvexPolygon vertices)
  (h_P_inside : InsidePolygon P vertices)
  (h_Q_inside : InsidePolygon Q vertices) :
  ∃ V ∈ vertices, distance V Q < distance V P := by
  sorry

end NUMINAMATH_CALUDE_exists_closer_vertex_l2269_226925


namespace NUMINAMATH_CALUDE_peach_baskets_l2269_226912

theorem peach_baskets (red_per_basket : ℕ) (total_red : ℕ) (h1 : red_per_basket = 16) (h2 : total_red = 96) :
  total_red / red_per_basket = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_baskets_l2269_226912


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l2269_226965

/-- Given an office in Singapore with the following conditions:
  * There are 1100 total employees
  * 60% of employees are female
  * 50% of male employees are computer literate
  * 62% of all employees are computer literate
  Prove that the number of female employees who are computer literate is 462 -/
theorem computer_literate_female_employees 
  (total_employees : ℕ) 
  (female_percentage : ℚ)
  (male_literate_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (h1 : total_employees = 1100)
  (h2 : female_percentage = 60 / 100)
  (h3 : male_literate_percentage = 50 / 100)
  (h4 : total_literate_percentage = 62 / 100) :
  ↑⌊(total_literate_percentage * total_employees - 
     male_literate_percentage * ((1 - female_percentage) * total_employees))⌋ = 462 := by
  sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l2269_226965


namespace NUMINAMATH_CALUDE_max_abc_constrained_polynomial_l2269_226980

/-- A polynomial of degree 4 with specific constraints on its coefficients. -/
structure ConstrainedPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  bound_a : a < 3
  bound_b : b < 3
  bound_c : c < 3
  p : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + 1
  no_real_roots : ∀ x : ℝ, p x ≠ 0

/-- The maximum value of abc for polynomials satisfying the given constraints is 18.75. -/
theorem max_abc_constrained_polynomial (poly : ConstrainedPolynomial) :
  ∃ M : ℝ, M = 18.75 ∧ poly.a * poly.b * poly.c ≤ M ∧
  ∀ ε > 0, ∃ poly' : ConstrainedPolynomial, poly'.a * poly'.b * poly'.c > M - ε :=
sorry

end NUMINAMATH_CALUDE_max_abc_constrained_polynomial_l2269_226980


namespace NUMINAMATH_CALUDE_sequence_next_terms_l2269_226970

def sequence1 : ℕ → ℕ
  | 0 => 2
  | n + 1 => sequence1 n + 2

def sequence2 : ℕ → ℕ
  | 0 => 3
  | n + 1 => sequence2 n * 2

def sequence3 : ℕ → ℕ
  | 0 => 36
  | 1 => 11
  | n + 2 => sequence3 n + 2

theorem sequence_next_terms :
  (sequence1 5 = 12 ∧ sequence1 6 = 14) ∧
  (sequence2 5 = 96) ∧
  (sequence3 8 = 44 ∧ sequence3 9 = 19) := by
  sorry

end NUMINAMATH_CALUDE_sequence_next_terms_l2269_226970


namespace NUMINAMATH_CALUDE_total_capsules_sold_l2269_226960

def weekly_earnings_100mg : ℕ := 80
def weekly_earnings_500mg : ℕ := 60
def cost_per_capsule_100mg : ℕ := 5
def cost_per_capsule_500mg : ℕ := 2

def capsules_100mg_per_week : ℕ := weekly_earnings_100mg / cost_per_capsule_100mg
def capsules_500mg_per_week : ℕ := weekly_earnings_500mg / cost_per_capsule_500mg

def total_capsules_2_weeks : ℕ := 2 * (capsules_100mg_per_week + capsules_500mg_per_week)

theorem total_capsules_sold :
  total_capsules_2_weeks = 92 :=
by sorry

end NUMINAMATH_CALUDE_total_capsules_sold_l2269_226960


namespace NUMINAMATH_CALUDE_dimes_spent_l2269_226972

/-- Given Joan's initial and remaining dimes, calculate the number of dimes spent. -/
theorem dimes_spent (initial : ℕ) (remaining : ℕ) (h : remaining ≤ initial) :
  initial - remaining = initial - remaining :=
by sorry

end NUMINAMATH_CALUDE_dimes_spent_l2269_226972


namespace NUMINAMATH_CALUDE_pentagon_sides_solutions_l2269_226905

/-- A pentagon with side lengths satisfying the given conditions -/
structure PentagonSides where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  h_one_side : e = 30
  h_arithmetic : b = a + 2 ∧ c = b + 2 ∧ d = c + 2
  h_smallest : a ≤ 7
  h_sum : a + b + c + d + e > e

/-- The theorem stating that only three specific pentagons satisfy the conditions -/
theorem pentagon_sides_solutions :
  { sides : PentagonSides | 
    (sides.a = 5 ∧ sides.b = 7 ∧ sides.c = 9 ∧ sides.d = 11 ∧ sides.e = 30) ∨
    (sides.a = 6 ∧ sides.b = 8 ∧ sides.c = 10 ∧ sides.d = 12 ∧ sides.e = 30) ∨
    (sides.a = 7 ∧ sides.b = 9 ∧ sides.c = 11 ∧ sides.d = 13 ∧ sides.e = 30) } =
  { sides : PentagonSides | True } :=
sorry

end NUMINAMATH_CALUDE_pentagon_sides_solutions_l2269_226905


namespace NUMINAMATH_CALUDE_zoo_bus_distribution_l2269_226981

theorem zoo_bus_distribution (total_people : ℕ) (num_buses : ℕ) (h1 : total_people = 219) (h2 : num_buses = 3) :
  total_people % num_buses = 0 →
  total_people / num_buses = 73 := by
sorry

end NUMINAMATH_CALUDE_zoo_bus_distribution_l2269_226981


namespace NUMINAMATH_CALUDE_terminal_point_coordinates_l2269_226904

/-- Given sin α = 3/5 and cos α = -4/5, the coordinates of the point on the terminal side of angle α are (-4, 3). -/
theorem terminal_point_coordinates (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : Real.cos α = -4/5) : 
  ∃ (x y : Real), x = -4 ∧ y = 3 ∧ Real.sin α = y / Real.sqrt (x^2 + y^2) ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_terminal_point_coordinates_l2269_226904


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l2269_226976

-- Define the given parameters
def sector_angle : ℝ := 240
def circle_radius : ℝ := 15

-- Define the theorem
theorem cone_volume_over_pi (sector_angle : ℝ) (circle_radius : ℝ) :
  sector_angle = 240 ∧ circle_radius = 15 →
  ∃ (cone_volume : ℝ),
    cone_volume / π = 500 * Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_over_pi_l2269_226976


namespace NUMINAMATH_CALUDE_min_value_theorem_l2269_226938

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ z, z = (2 * x * y) / (x + y - 1) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2269_226938


namespace NUMINAMATH_CALUDE_two_talents_count_l2269_226911

def num_students : ℕ := 120

def num_cant_sing : ℕ := 50
def num_cant_dance : ℕ := 75
def num_cant_act : ℕ := 35

def num_can_sing : ℕ := num_students - num_cant_sing
def num_can_dance : ℕ := num_students - num_cant_dance
def num_can_act : ℕ := num_students - num_cant_act

theorem two_talents_count :
  ∀ (x : ℕ),
    x ≤ num_students →
    (num_can_sing + num_can_dance + num_can_act) - (num_students - x) = 80 + x →
    x = 0 →
    (num_can_sing + num_can_dance + num_can_act) - num_students = 80 :=
by sorry

end NUMINAMATH_CALUDE_two_talents_count_l2269_226911


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2269_226908

theorem sun_radius_scientific_notation : 369000 = 3.69 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2269_226908


namespace NUMINAMATH_CALUDE_koi_fish_count_l2269_226971

/-- Calculates the number of koi fish after three weeks -/
def koi_fish_after_three_weeks (initial_total : ℕ) (initial_goldfish : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : ℕ :=
  let initial_koi := initial_total - initial_goldfish
  let total_koi_added := koi_added_per_day * days
  initial_koi + total_koi_added

theorem koi_fish_count (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) 
    (h1 : initial_total = 280)
    (h2 : koi_added_per_day = 2)
    (h3 : goldfish_added_per_day = 5)
    (h4 : days = 21)
    (h5 : final_goldfish = 200) :
  koi_fish_after_three_weeks initial_total (initial_total - (final_goldfish - goldfish_added_per_day * days)) koi_added_per_day goldfish_added_per_day days final_goldfish = 227 := by
  sorry

#eval koi_fish_after_three_weeks 280 95 2 5 21 200

end NUMINAMATH_CALUDE_koi_fish_count_l2269_226971


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2269_226982

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 →
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - 1)) = (x + 1) / (x - 3) ∧
  (2 + 1) / (2 - 3) = -3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2269_226982


namespace NUMINAMATH_CALUDE_system_solution_unique_l2269_226923

theorem system_solution_unique :
  ∃! (x y : ℝ), 3 * x - 2 * y = 3 ∧ x + 4 * y = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2269_226923


namespace NUMINAMATH_CALUDE_total_apples_picked_l2269_226950

theorem total_apples_picked (benny_apples dan_apples : ℕ) : 
  benny_apples = 2 → dan_apples = 9 → benny_apples + dan_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_picked_l2269_226950


namespace NUMINAMATH_CALUDE_unique_lottery_ticket_l2269_226902

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_lottery_ticket (ticket : ℕ) (neighbor_age : ℕ) :
  is_five_digit ticket →
  digit_sum ticket = neighbor_age →
  (∀ m : ℕ, is_five_digit m → digit_sum m = neighbor_age → m = ticket) →
  ticket = 99999 :=
by sorry

end NUMINAMATH_CALUDE_unique_lottery_ticket_l2269_226902


namespace NUMINAMATH_CALUDE_tomato_theorem_l2269_226906

def tomato_problem (plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 : ℕ) : Prop :=
  plant1 = 15 ∧
  plant2 = 2 * plant1 - 8 ∧
  plant3 = (plant1^2) / 3 ∧
  plant4 = (plant1 + plant2) / 2 ∧
  plant5 = 3 * Int.sqrt (plant1 + plant2) ∧
  plant6 = plant5 ∧
  plant7 = (3 * (plant1 + plant2 + plant3)) / 2 ∧
  plant8 = plant7 ∧
  plant9 = plant1 + plant7 + 6 →
  plant1 + plant2 + plant3 + plant4 + plant5 + plant6 + plant7 + plant8 + plant9 = 692

theorem tomato_theorem : ∃ plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 : ℕ,
  tomato_problem plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_theorem_l2269_226906
