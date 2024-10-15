import Mathlib

namespace NUMINAMATH_CALUDE_transform_point_l2614_261447

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem transform_point (p : ℝ × ℝ) :
  reflectX (rotate180 p) = (p.1, -p.2) := by sorry

end NUMINAMATH_CALUDE_transform_point_l2614_261447


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2614_261499

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def vector_dot_product (v w : ℝ × ℝ) : ℝ := sorry

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := sorry

def vector_length (v : ℝ × ℝ) : ℝ := sorry

def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area 
  (q : Quadrilateral) 
  (h_convex : is_convex q) 
  (h_bd : vector_length (vector_add q.B (vector_add q.D (-q.B))) = 2) 
  (h_perp : vector_dot_product (vector_add q.A (vector_add q.C (-q.A))) 
                               (vector_add q.B (vector_add q.D (-q.B))) = 0) 
  (h_sum : vector_dot_product (vector_add (vector_add q.A (vector_add q.B (-q.A))) 
                                          (vector_add q.D (vector_add q.C (-q.D)))) 
                              (vector_add (vector_add q.B (vector_add q.C (-q.B))) 
                                          (vector_add q.A (vector_add q.D (-q.A)))) = 5) : 
  area q = 3 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2614_261499


namespace NUMINAMATH_CALUDE_meat_for_spring_rolls_l2614_261471

theorem meat_for_spring_rolls (initial_meat : ℝ) (meatball_fraction : ℝ) (remaining_meat : ℝ) : 
  initial_meat = 20 ∧ meatball_fraction = 1/4 ∧ remaining_meat = 12 →
  initial_meat - meatball_fraction * initial_meat - remaining_meat = 3 :=
by sorry

end NUMINAMATH_CALUDE_meat_for_spring_rolls_l2614_261471


namespace NUMINAMATH_CALUDE_jane_daniel_difference_l2614_261421

/-- The width of the streets in Newville -/
def street_width : ℝ := 30

/-- The length of one side of a square block in Newville -/
def block_side : ℝ := 500

/-- The length of Daniel's path around one block -/
def daniel_lap : ℝ := 4 * block_side

/-- The length of Jane's path around one block -/
def jane_lap : ℝ := 4 * (block_side + street_width)

/-- The theorem stating the difference between Jane's and Daniel's lap distances -/
theorem jane_daniel_difference : jane_lap - daniel_lap = 120 := by
  sorry

end NUMINAMATH_CALUDE_jane_daniel_difference_l2614_261421


namespace NUMINAMATH_CALUDE_equilateral_triangle_circles_l2614_261407

theorem equilateral_triangle_circles (rA rB rC : ℝ) : 
  rA < rB ∧ rB < rC →  -- radii form increasing sequence
  ∃ (d : ℝ), rB = rA + d ∧ rC = rA + 2*d →  -- arithmetic sequence
  6 - (rA + rB) = 3.5 →  -- shortest distance between circles A and B
  6 - (rA + rC) = 3 →  -- shortest distance between circles A and C
  (1/6) * (π * rA^2 + π * rB^2 + π * rC^2) = 29*π/24 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circles_l2614_261407


namespace NUMINAMATH_CALUDE_elmer_milton_ratio_l2614_261477

-- Define the daily food intake for each animal
def penelope_intake : ℚ := 20
def greta_intake : ℚ := penelope_intake / 10
def milton_intake : ℚ := greta_intake / 100
def elmer_intake : ℚ := penelope_intake + 60

-- Theorem statement
theorem elmer_milton_ratio : 
  elmer_intake / milton_intake = 4000 := by sorry

end NUMINAMATH_CALUDE_elmer_milton_ratio_l2614_261477


namespace NUMINAMATH_CALUDE_average_books_is_three_l2614_261459

/-- Represents the distribution of books read by book club members -/
structure BookDistribution where
  one_book : Nat
  two_books : Nat
  three_books : Nat
  four_books : Nat
  six_books : Nat

/-- Calculates the average number of books read, rounded to the nearest whole number -/
def averageBooksRead (d : BookDistribution) : Nat :=
  let totalBooks := d.one_book * 1 + d.two_books * 2 + d.three_books * 3 + d.four_books * 4 + d.six_books * 6
  let totalMembers := d.one_book + d.two_books + d.three_books + d.four_books + d.six_books
  (totalBooks + totalMembers / 2) / totalMembers

/-- Theorem stating that the average number of books read is 3 -/
theorem average_books_is_three (d : BookDistribution) 
  (h1 : d.one_book = 4)
  (h2 : d.two_books = 3)
  (h3 : d.three_books = 6)
  (h4 : d.four_books = 2)
  (h5 : d.six_books = 3) : 
  averageBooksRead d = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_books_is_three_l2614_261459


namespace NUMINAMATH_CALUDE_sum_of_roots_tangent_equation_l2614_261483

theorem sum_of_roots_tangent_equation : 
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ < π ∧
    0 < x₂ ∧ x₂ < π ∧
    (Real.tan x₁)^2 - 5 * Real.tan x₁ + 6 = 0 ∧
    (Real.tan x₂)^2 - 5 * Real.tan x₂ + 6 = 0 ∧
    x₁ + x₂ = Real.arctan 3 + Real.arctan 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_tangent_equation_l2614_261483


namespace NUMINAMATH_CALUDE_constant_n_value_l2614_261460

theorem constant_n_value (m n : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + m) = x^2 + n*x + 12) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_n_value_l2614_261460


namespace NUMINAMATH_CALUDE_expression_evaluation_l2614_261468

def f (x : ℚ) : ℚ := (2 * x + 2) / (x - 2)

theorem expression_evaluation :
  let x : ℚ := 3
  let result := f (f x)
  result = 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2614_261468


namespace NUMINAMATH_CALUDE_system_solution_l2614_261461

theorem system_solution :
  ∀ (x y z : ℝ),
    (x + 1) * y * z = 12 ∧
    (y + 1) * z * x = 4 ∧
    (z + 1) * x * y = 4 →
    ((x = 1/3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2614_261461


namespace NUMINAMATH_CALUDE_square_perimeter_quadrupled_l2614_261423

theorem square_perimeter_quadrupled (s : ℝ) (x : ℝ) :
  x = 4 * s →
  4 * x = 4 * (4 * s) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_quadrupled_l2614_261423


namespace NUMINAMATH_CALUDE_cube_root_problem_l2614_261434

theorem cube_root_problem (a b c : ℝ) : 
  (3 * a + 21) ^ (1/3) = 3 → 
  (4 * a - b - 1) ^ (1/2) = 2 → 
  c ^ (1/2) = c → 
  a = 2 ∧ b = 3 ∧ c = 0 ∧ (3 * a + 10 * b + c) ^ (1/2) = 6 ∨ (3 * a + 10 * b + c) ^ (1/2) = -6 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2614_261434


namespace NUMINAMATH_CALUDE_circle_parabola_height_difference_l2614_261417

/-- Given a circle inside the parabola y = 4x^2, tangent at two points,
    prove the height difference between the circle's center and tangency points. -/
theorem circle_parabola_height_difference (a : ℝ) : 
  let parabola (x : ℝ) := 4 * x^2
  let tangency_point := (a, parabola a)
  let circle_center := (0, a^2 + 1/8)
  circle_center.2 - tangency_point.2 = -3 * a^2 + 1/8 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_height_difference_l2614_261417


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2614_261404

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = I - 3 → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2614_261404


namespace NUMINAMATH_CALUDE_descending_eight_digit_numbers_count_l2614_261498

/-- The number of eight-digit numbers where each digit (except the last one) 
    is greater than the following digit. -/
def count_descending_eight_digit_numbers : ℕ :=
  Nat.choose 10 2

/-- Theorem stating that the count of eight-digit numbers with descending digits
    is equal to choosing 2 from 10. -/
theorem descending_eight_digit_numbers_count :
  count_descending_eight_digit_numbers = 45 := by
  sorry

end NUMINAMATH_CALUDE_descending_eight_digit_numbers_count_l2614_261498


namespace NUMINAMATH_CALUDE_triangle_properties_l2614_261495

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0)
  (h2 : t.c = 2)
  (h3 : t.a + t.b = t.a * t.b) : 
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2614_261495


namespace NUMINAMATH_CALUDE_cubic_inequality_with_equality_l2614_261400

theorem cubic_inequality_with_equality (a b : ℝ) :
  a < b → a^3 - 3*a ≤ b^3 - 3*b + 4 ∧
  (a = -1 ∧ b = 1 → a^3 - 3*a = b^3 - 3*b + 4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_with_equality_l2614_261400


namespace NUMINAMATH_CALUDE_simplify_expression_l2614_261494

theorem simplify_expression (m n : ℝ) : 
  m - (m^2 * n + 3 * m - 4 * n) + (2 * n * m^2 - 3 * n) = m^2 * n - 2 * m + n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2614_261494


namespace NUMINAMATH_CALUDE_other_diagonal_length_l2614_261491

/-- A rhombus with known properties -/
structure Rhombus where
  /-- The length of one diagonal -/
  diagonal1 : ℝ
  /-- The area of one of the two equal triangles that make up the rhombus -/
  triangle_area : ℝ
  /-- Assumption that the diagonal1 is positive -/
  diagonal1_pos : 0 < diagonal1
  /-- Assumption that the triangle_area is positive -/
  triangle_area_pos : 0 < triangle_area

/-- The theorem stating the length of the other diagonal given specific conditions -/
theorem other_diagonal_length (r : Rhombus) (h1 : r.diagonal1 = 15) (h2 : r.triangle_area = 75) :
  ∃ diagonal2 : ℝ, diagonal2 = 20 ∧ r.diagonal1 * diagonal2 / 2 = 2 * r.triangle_area := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l2614_261491


namespace NUMINAMATH_CALUDE_number_between_5_and_9_greater_than_7_l2614_261450

theorem number_between_5_and_9_greater_than_7 : ∃! x : ℝ, 5 < x ∧ x < 9 ∧ 7 < x := by
  sorry

end NUMINAMATH_CALUDE_number_between_5_and_9_greater_than_7_l2614_261450


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_1998_l2614_261405

theorem sum_of_four_cubes_1998 : ∃ (a b c d : ℤ), 1998 = a^3 + b^3 + c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_1998_l2614_261405


namespace NUMINAMATH_CALUDE_special_function_inequality_l2614_261489

/-- A function satisfying the given differential inequality -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  diff_twice : ∀ x ∈ domain, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ (deriv f) x
  ineq : ∀ x ∈ domain, x * (deriv^[2] f x) > f x

/-- The main theorem -/
theorem special_function_inequality (φ : SpecialFunction) (x₁ x₂ : ℝ) 
    (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : 
    φ.f x₁ + φ.f x₂ < φ.f (x₁ + x₂) := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2614_261489


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2614_261455

theorem power_fraction_simplification :
  (6^5 * 3^5) / 18^4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2614_261455


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2614_261412

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ p q : ℝ, 16 - 4*x - x^2 = 0 ∧ x = p ∨ x = q) → 
  (∃ p q : ℝ, 16 - 4*p - p^2 = 0 ∧ 16 - 4*q - q^2 = 0 ∧ p + q = 4) :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2614_261412


namespace NUMINAMATH_CALUDE_min_manhattan_distance_l2614_261458

-- Define the manhattan distance function
def manhattan_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 12 = 0

-- State the theorem
theorem min_manhattan_distance :
  ∃ (min_dist : ℝ),
    min_dist = (12 - Real.sqrt 34) / 4 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      on_ellipse x₁ y₁ → on_line x₂ y₂ →
      manhattan_distance x₁ y₁ x₂ y₂ ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_min_manhattan_distance_l2614_261458


namespace NUMINAMATH_CALUDE_difference_even_odd_sums_l2614_261419

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : 
  (sumFirstEvenIntegers 25) - (sumFirstOddIntegers 20) = 250 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_odd_sums_l2614_261419


namespace NUMINAMATH_CALUDE_intersection_line_ellipse_part1_intersection_line_ellipse_part2_l2614_261411

noncomputable section

-- Define the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def ellipse (a : ℝ) (x y : ℝ) : Prop := 3 * x^2 + y^2 = a

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the area of a triangle given the coordinates of its vertices
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem intersection_line_ellipse_part1 (a : ℝ) (x1 y1 x2 y2 : ℝ) :
  line 1 x1 = y1 →
  line 1 x2 = y2 →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  distance x1 y1 x2 y2 = Real.sqrt 10 / 2 →
  a = 2 := by sorry

theorem intersection_line_ellipse_part2 (k a : ℝ) (x1 y1 x2 y2 : ℝ) :
  k ≠ 0 →
  line k x1 = y1 →
  line k x2 = y2 →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  x1 = -2 * x2 →
  ∃ (max_area : ℝ),
    (∀ (k' a' : ℝ) (x1' y1' x2' y2' : ℝ),
      k' ≠ 0 →
      line k' x1' = y1' →
      line k' x2' = y2' →
      ellipse a' x1' y1' →
      ellipse a' x2' y2' →
      x1' = -2 * x2' →
      triangle_area 0 0 x1' y1' x2' y2' ≤ max_area) ∧
    max_area = Real.sqrt 3 / 2 ∧
    a = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_line_ellipse_part1_intersection_line_ellipse_part2_l2614_261411


namespace NUMINAMATH_CALUDE_infinite_decimal_sqrt_l2614_261497

theorem infinite_decimal_sqrt (x y : ℕ) : 
  x ∈ Finset.range 9 → y ∈ Finset.range 9 → 
  (Real.sqrt (x / 9 : ℝ) = y / 9) ↔ ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) := by
sorry

end NUMINAMATH_CALUDE_infinite_decimal_sqrt_l2614_261497


namespace NUMINAMATH_CALUDE_expression_equality_l2614_261448

theorem expression_equality : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2614_261448


namespace NUMINAMATH_CALUDE_calculation_proof_l2614_261462

theorem calculation_proof : (-2)^3 - |2 - 5| / (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2614_261462


namespace NUMINAMATH_CALUDE_lloyds_hourly_rate_l2614_261443

-- Define Lloyd's regular work hours
def regular_hours : ℝ := 7.5

-- Define Lloyd's overtime multiplier
def overtime_multiplier : ℝ := 1.5

-- Define Lloyd's actual work hours on the given day
def actual_hours : ℝ := 10.5

-- Define Lloyd's total earnings for the day
def total_earnings : ℝ := 42

-- Theorem to prove Lloyd's hourly rate
theorem lloyds_hourly_rate :
  ∃ (rate : ℝ),
    rate * regular_hours + 
    (actual_hours - regular_hours) * rate * overtime_multiplier = total_earnings ∧
    rate = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_lloyds_hourly_rate_l2614_261443


namespace NUMINAMATH_CALUDE_front_view_correct_l2614_261441

def ColumnHeights := List Nat

def frontView (columns : List ColumnHeights) : List Nat :=
  columns.map (List.foldl max 0)

theorem front_view_correct (columns : List ColumnHeights) :
  frontView columns = [3, 4, 5, 2] :=
by
  -- The proof would go here
  sorry

#eval frontView [[3, 2], [1, 4, 2], [5], [2, 1]]

end NUMINAMATH_CALUDE_front_view_correct_l2614_261441


namespace NUMINAMATH_CALUDE_distance_to_origin_of_point_on_parabola_l2614_261481

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * C.p * x

theorem distance_to_origin_of_point_on_parabola
  (C : Parabola)
  (A : PointOnParabola C)
  (h1 : Real.sqrt ((A.x - C.p/2)^2 + A.y^2) = 6)  -- Distance from A to focus is 6
  (h2 : A.x = 3)  -- Distance from A to y-axis is 3
  : Real.sqrt (A.x^2 + A.y^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_point_on_parabola_l2614_261481


namespace NUMINAMATH_CALUDE_parabola_shift_l2614_261429

/-- A parabola shifted 1 unit left and 4 units down -/
def shifted_parabola (x : ℝ) : ℝ := 3 * (x + 1)^2 - 4

/-- The original parabola -/
def original_parabola (x : ℝ) : ℝ := 3 * x^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) - 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2614_261429


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2614_261413

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 250 * Real.pi) :
  A = Real.pi * r^2 → r = 5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2614_261413


namespace NUMINAMATH_CALUDE_bigger_part_of_division_l2614_261476

theorem bigger_part_of_division (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 60) (h4 : 10 * x + 22 * y = 780) : max x y = 45 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_of_division_l2614_261476


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l2614_261416

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem reflection_across_y_axis :
  let P : Point2D := { x := -3, y := 5 }
  reflectAcrossYAxis P = { x := 3, y := 5 } := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l2614_261416


namespace NUMINAMATH_CALUDE_linear_function_through_points_l2614_261485

/-- Given a linear function y = ax + a where a is a constant, and the graph of this function 
    passes through the point (1,2), prove that the graph also passes through the point (-2,-1). -/
theorem linear_function_through_points (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x => a * x + a) → 
  (2 = a * 1 + a) → 
  (-1 = a * (-2) + a) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_through_points_l2614_261485


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2614_261425

theorem students_taking_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) :
  total = 500 →
  music = 30 →
  art = 20 →
  both = 10 →
  total - (music + art - both) = 460 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l2614_261425


namespace NUMINAMATH_CALUDE_table_relationship_l2614_261452

def f (x : ℝ) : ℝ := -5 * x^2 - 10 * x

theorem table_relationship : 
  (f 0 = 0) ∧ 
  (f 1 = -15) ∧ 
  (f 2 = -40) ∧ 
  (f 3 = -75) ∧ 
  (f 4 = -120) := by
  sorry

end NUMINAMATH_CALUDE_table_relationship_l2614_261452


namespace NUMINAMATH_CALUDE_enclosed_area_is_one_l2614_261465

-- Define the curves
def curve (x : ℝ) : ℝ := x^2 + 2
def line (x : ℝ) : ℝ := 3*x

-- Define the boundaries
def left_boundary : ℝ := 0
def right_boundary : ℝ := 2

-- Define the area function
noncomputable def area : ℝ := ∫ x in left_boundary..right_boundary, max (curve x - line x) 0 + max (line x - curve x) 0

-- Theorem statement
theorem enclosed_area_is_one : area = 1 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_one_l2614_261465


namespace NUMINAMATH_CALUDE_sum_first_and_ninth_term_l2614_261472

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_first_and_ninth_term : a 1 + a 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_and_ninth_term_l2614_261472


namespace NUMINAMATH_CALUDE_sum_inequality_l2614_261435

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2614_261435


namespace NUMINAMATH_CALUDE_video_dislikes_l2614_261475

theorem video_dislikes (likes : ℕ) (initial_dislikes : ℕ) (additional_dislikes : ℕ) : 
  likes = 3000 → 
  initial_dislikes = likes / 2 + 100 → 
  additional_dislikes = 1000 → 
  initial_dislikes + additional_dislikes = 2600 :=
by sorry

end NUMINAMATH_CALUDE_video_dislikes_l2614_261475


namespace NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l2614_261402

/-- The amount of cargo loaded in the Bahamas -/
def cargo_loaded (initial_cargo final_cargo : ℕ) : ℕ :=
  final_cargo - initial_cargo

/-- Theorem: The amount of cargo loaded in the Bahamas is 8723 tons -/
theorem cargo_loaded_in_bahamas :
  cargo_loaded 5973 14696 = 8723 := by
  sorry

end NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l2614_261402


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2614_261466

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the side of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for an isosceles trapezoid with bases 7 and 13 and area 40, the side length is 5 -/
theorem isosceles_trapezoid_side_length :
  let t : IsoscelesTrapezoid := ⟨7, 13, 40⟩
  side_length t = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2614_261466


namespace NUMINAMATH_CALUDE_cube_surface_area_l2614_261418

/-- Given a cube with volume 27 cubic cm, its surface area is 54 square cm. -/
theorem cube_surface_area (cube : Set ℝ) (volume : ℝ) (surface_area : ℝ) : 
  volume = 27 →
  surface_area = 54 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2614_261418


namespace NUMINAMATH_CALUDE_optimal_plan_is_best_l2614_261436

/-- Represents a bus purchasing plan -/
structure BusPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a bus plan is valid according to the given constraints -/
def isValidPlan (plan : BusPlan) : Prop :=
  plan.typeA + plan.typeB = 10 ∧
  100 * plan.typeA + 150 * plan.typeB ≤ 1200 ∧
  60 * plan.typeA + 100 * plan.typeB ≥ 680

/-- Calculates the total cost of a bus plan in million RMB -/
def totalCost (plan : BusPlan) : ℕ :=
  100 * plan.typeA + 150 * plan.typeB

/-- The optimal bus purchasing plan -/
def optimalPlan : BusPlan :=
  { typeA := 8, typeB := 2 }

/-- Theorem stating that the optimal plan is valid and minimizes the total cost -/
theorem optimal_plan_is_best :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
sorry

#check optimal_plan_is_best

end NUMINAMATH_CALUDE_optimal_plan_is_best_l2614_261436


namespace NUMINAMATH_CALUDE_correct_average_l2614_261432

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 21 ∧ 
  wrong_num = 26 ∧ 
  correct_num = 36 →
  (n : ℚ) * incorrect_avg + (correct_num - wrong_num) = n * 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l2614_261432


namespace NUMINAMATH_CALUDE_linear_function_theorem_l2614_261449

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the domain and range conditions
def domain_condition (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1
def range_condition (y : ℝ) : Prop := 1 ≤ y ∧ y ≤ 9

-- Theorem statement
theorem linear_function_theorem (k b : ℝ) :
  (∀ x, domain_condition x → range_condition (linear_function k b x)) →
  ((k = 2 ∧ b = 7) ∨ (k = -2 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l2614_261449


namespace NUMINAMATH_CALUDE_fixed_points_condition_l2614_261470

/-- A quadratic function with parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - x + c

/-- Theorem stating the condition on c for a quadratic function with specific fixed point properties -/
theorem fixed_points_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = x₁ ∧ f c x₂ = x₂ ∧ x₁ < 2 ∧ 2 < x₂) →
  c < 0 :=
sorry

end NUMINAMATH_CALUDE_fixed_points_condition_l2614_261470


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_eccentricity_l2614_261451

/-- Given a hyperbola and a circle that intersect to form a square, 
    prove that the eccentricity of the hyperbola is √(2 + √2) -/
theorem hyperbola_circle_intersection_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → x^2 + y^2 = c^2 → x^2 = y^2) : 
  c / a = Real.sqrt (2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_eccentricity_l2614_261451


namespace NUMINAMATH_CALUDE_books_grabbed_l2614_261403

/-- Calculates the number of books Henry grabbed from the "free to a good home" box -/
theorem books_grabbed (initial_books : ℕ) (donated_boxes : ℕ) (books_per_box : ℕ) 
  (room_books : ℕ) (coffee_table_books : ℕ) (kitchen_books : ℕ) (final_books : ℕ) : 
  initial_books = 99 →
  donated_boxes = 3 →
  books_per_box = 15 →
  room_books = 21 →
  coffee_table_books = 4 →
  kitchen_books = 18 →
  final_books = 23 →
  final_books - (initial_books - (donated_boxes * books_per_box + room_books + coffee_table_books + kitchen_books)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_books_grabbed_l2614_261403


namespace NUMINAMATH_CALUDE_train_distance_problem_l2614_261431

/-- Proves that the distance between two stations is 450 km, given the conditions of the train problem. -/
theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : v2 > v1) (h4 : d > 0) :
  let t := d / v1
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 50 → d1 + d2 = 450 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2614_261431


namespace NUMINAMATH_CALUDE_y_derivative_l2614_261442

noncomputable def y (x : ℝ) : ℝ := (1 - x^2) / Real.exp x

theorem y_derivative (x : ℝ) : 
  deriv y x = (x^2 - 2*x - 1) / Real.exp x :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2614_261442


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2614_261487

/-- A circle with radius 5, center on the x-axis, and tangent to the line x=3 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : (center.2 = 0)
  radius_is_5 : radius = 5
  tangent_to_x3 : |center.1 - 3| = 5

/-- The equation of the circle is (x-8)^2 + y^2 = 25 or (x+2)^2 + y^2 = 25 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y : ℝ, (x - 8)^2 + y^2 = 25 ∨ (x + 2)^2 + y^2 = 25 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2614_261487


namespace NUMINAMATH_CALUDE_shirt_cost_l2614_261493

theorem shirt_cost (num_shirts : ℕ) (num_pants : ℕ) (pant_cost : ℕ) (total_cost : ℕ) : 
  num_shirts = 10 →
  num_pants = num_shirts / 2 →
  pant_cost = 8 →
  total_cost = 100 →
  num_shirts * (total_cost - num_pants * pant_cost) / num_shirts = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l2614_261493


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l2614_261456

/-- Given a train of length 250 meters that passes a pole in 10 seconds
    and a platform in 60 seconds, prove that the time taken to pass
    only the platform is 50 seconds. -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (pole_passing_time : ℝ)
  (platform_total_passing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : pole_passing_time = 10)
  (h3 : platform_total_passing_time = 60) :
  let train_speed := train_length / pole_passing_time
  let platform_length := train_speed * platform_total_passing_time - train_length
  platform_length / train_speed = 50 := by
sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l2614_261456


namespace NUMINAMATH_CALUDE_digital_earth_functions_l2614_261484

-- Define the concept of Digital Earth
structure DigitalEarth where
  integratesInfo : Bool
  displaysIn3D : Bool
  isDynamic : Bool
  providesExperimentalConditions : Bool

-- Define the correct description of Digital Earth functions
def correctDescription (de : DigitalEarth) : Prop :=
  de.integratesInfo ∧ de.displaysIn3D ∧ de.isDynamic ∧ de.providesExperimentalConditions

-- Theorem stating that the correct description accurately represents Digital Earth functions
theorem digital_earth_functions :
  ∀ (de : DigitalEarth), correctDescription de ↔ 
    (de.integratesInfo = true ∧ 
     de.displaysIn3D = true ∧ 
     de.isDynamic = true ∧ 
     de.providesExperimentalConditions = true) :=
by
  sorry

#check digital_earth_functions

end NUMINAMATH_CALUDE_digital_earth_functions_l2614_261484


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2614_261446

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ Real.sin (Real.arccos (Real.tanh (Real.arcsin x))) = x ∧ x = Real.sqrt (1/2) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2614_261446


namespace NUMINAMATH_CALUDE_a_minus_b_values_l2614_261479

theorem a_minus_b_values (a b : ℝ) (h1 : a < b) (h2 : |a| = 6) (h3 : |b| = 3) :
  a - b = -9 ∨ a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l2614_261479


namespace NUMINAMATH_CALUDE_one_carton_per_case_l2614_261438

/-- Given that each carton contains b boxes, each box contains 200 paper clips,
    and 400 paper clips are contained in 2 cases, prove that there is 1 carton in a case. -/
theorem one_carton_per_case (b : ℕ) (h1 : b > 0) :
  (∃ c : ℕ, c > 0 ∧ c * b * 200 = 200) → c = 1 :=
by sorry

end NUMINAMATH_CALUDE_one_carton_per_case_l2614_261438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_l2614_261430

/-- Given an arithmetic sequence {a_n} with first term a₁ = 1 and common difference d = 3,
    prove that the value of n for which aₙ = 2005 is 669. -/
theorem arithmetic_sequence_2005 (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
  a 1 = 1 →                    -- First term is 1
  ∃ n : ℕ, a n = 2005 ∧ n = 669 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_l2614_261430


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2614_261467

theorem last_two_digits_sum (n : ℕ) : (9^n + 11^n) % 100 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2614_261467


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_geq_neg_two_l2614_261437

theorem inequality_holds_iff_a_geq_neg_two :
  ∀ a : ℝ, (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_geq_neg_two_l2614_261437


namespace NUMINAMATH_CALUDE_specific_extended_parallelepiped_volume_l2614_261490

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped -/
def extended_parallelepiped_volume (l w h : ℝ) : ℝ :=
  (l + 2) * (w + 2) * (h + 2) - (l * w * h)

/-- The theorem stating the volume of the specific extended parallelepiped -/
theorem specific_extended_parallelepiped_volume :
  extended_parallelepiped_volume 5 6 7 = (1272 + 58 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_extended_parallelepiped_volume_l2614_261490


namespace NUMINAMATH_CALUDE_min_value_x2_y2_l2614_261444

theorem min_value_x2_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 196) :
  ∃ (m : ℝ), m = 169 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 196 → x^2 + y^2 ≤ a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x2_y2_l2614_261444


namespace NUMINAMATH_CALUDE_safety_rent_a_truck_cost_per_mile_l2614_261478

/-- The cost per mile for Safety Rent A Truck -/
def safety_cost_per_mile : ℝ := sorry

/-- The base cost for Safety Rent A Truck -/
def safety_base_cost : ℝ := 41.95

/-- The base cost for City Rentals -/
def city_base_cost : ℝ := 38.95

/-- The cost per mile for City Rentals -/
def city_cost_per_mile : ℝ := 0.31

/-- The number of miles for which the total costs are equal -/
def equal_cost_miles : ℝ := 150.0

theorem safety_rent_a_truck_cost_per_mile :
  safety_base_cost + equal_cost_miles * safety_cost_per_mile =
  city_base_cost + equal_cost_miles * city_cost_per_mile ∧
  safety_cost_per_mile = 0.29 := by sorry

end NUMINAMATH_CALUDE_safety_rent_a_truck_cost_per_mile_l2614_261478


namespace NUMINAMATH_CALUDE_composite_probability_six_dice_l2614_261473

/-- The number of sides on a standard die -/
def dieSize : Nat := 6

/-- The number of dice rolled -/
def numDice : Nat := 6

/-- The set of possible outcomes when rolling a die -/
def dieOutcomes : Finset Nat := Finset.range dieSize

/-- The total number of possible outcomes when rolling 6 dice -/
def totalOutcomes : Nat := dieSize ^ numDice

/-- A function that determines if a number is prime -/
def isPrime (n : Nat) : Bool := sorry

/-- A function that determines if a number is composite -/
def isComposite (n : Nat) : Bool := n > 1 ∧ ¬(isPrime n)

/-- The number of outcomes where the product is not composite -/
def nonCompositeOutcomes : Nat := 19

/-- The probability of rolling a composite product -/
def compositeProb : Rat := (totalOutcomes - nonCompositeOutcomes) / totalOutcomes

theorem composite_probability_six_dice :
  compositeProb = 46637 / 46656 := by sorry

end NUMINAMATH_CALUDE_composite_probability_six_dice_l2614_261473


namespace NUMINAMATH_CALUDE_selection_problem_l2614_261409

def total_students : ℕ := 10
def selected_students : ℕ := 3
def students_excluding_c : ℕ := 9
def students_excluding_abc : ℕ := 7

theorem selection_problem :
  (Nat.choose students_excluding_c selected_students) -
  (Nat.choose students_excluding_abc selected_students) = 49 := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l2614_261409


namespace NUMINAMATH_CALUDE_total_bananas_is_110_l2614_261401

/-- The total number of bananas Willie, Charles, and Lucy had originally -/
def total_bananas (willie_bananas charles_bananas lucy_bananas : ℕ) : ℕ :=
  willie_bananas + charles_bananas + lucy_bananas

/-- Theorem stating that the total number of bananas is 110 -/
theorem total_bananas_is_110 :
  total_bananas 48 35 27 = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_bananas_is_110_l2614_261401


namespace NUMINAMATH_CALUDE_second_vessel_capacity_l2614_261406

/-- Proves that the capacity of the second vessel is 6 liters given the problem conditions -/
theorem second_vessel_capacity : 
  ∀ (vessel2_capacity : ℝ),
    -- Given conditions
    let vessel1_capacity : ℝ := 2
    let vessel1_concentration : ℝ := 0.25
    let vessel2_concentration : ℝ := 0.40
    let total_liquid : ℝ := 8
    let final_vessel_capacity : ℝ := 10
    let final_concentration : ℝ := 0.29000000000000004

    -- Total liquid equation
    vessel1_capacity + vessel2_capacity = total_liquid →
    
    -- Alcohol balance equation
    (vessel1_capacity * vessel1_concentration + 
     vessel2_capacity * vessel2_concentration) / final_vessel_capacity = final_concentration →
    
    -- Conclusion
    vessel2_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_vessel_capacity_l2614_261406


namespace NUMINAMATH_CALUDE_bounded_quadratic_coef_sum_l2614_261414

/-- A quadratic polynomial f(x) = ax² + bx + c with |f(x)| ≤ 1 for all x in [0, 2] -/
def BoundedQuadratic (a b c : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |a * x^2 + b * x + c| ≤ 1

/-- The sum of absolute values of coefficients is at most 7 -/
theorem bounded_quadratic_coef_sum (a b c : ℝ) (h : BoundedQuadratic a b c) :
  |a| + |b| + |c| ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_bounded_quadratic_coef_sum_l2614_261414


namespace NUMINAMATH_CALUDE_total_shaded_area_is_72_l2614_261464

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a parallelogram with base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := p.base * p.height

/-- Represents the overlap between shapes -/
structure Overlap where
  width : ℝ
  height : ℝ

/-- Calculates the area of overlap -/
def overlapArea (o : Overlap) : ℝ := o.width * o.height

/-- Theorem: The total shaded area of intersection between the given rectangle and parallelogram is 72 square units -/
theorem total_shaded_area_is_72 (r : Rectangle) (p : Parallelogram) (o : Overlap) : 
  r.width = 4 ∧ r.height = 12 ∧ p.base = 10 ∧ p.height = 4 ∧ o.width = 4 ∧ o.height = 4 →
  rectangleArea r + parallelogramArea p - overlapArea o = 72 := by
  sorry


end NUMINAMATH_CALUDE_total_shaded_area_is_72_l2614_261464


namespace NUMINAMATH_CALUDE_greater_number_proof_l2614_261496

theorem greater_number_proof (x y : ℝ) 
  (sum_eq : x + y = 30)
  (diff_eq : x - y = 6)
  (prod_eq : x * y = 216) :
  max x y = 18 := by
sorry

end NUMINAMATH_CALUDE_greater_number_proof_l2614_261496


namespace NUMINAMATH_CALUDE_smallest_product_increase_l2614_261422

theorem smallest_product_increase (p q r s : ℝ) 
  (h_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s) 
  (h_order : p < q ∧ q < r ∧ r < s) : 
  min (min (min ((p+1)*q*r*s) (p*(q+1)*r*s)) (p*q*(r+1)*s)) (p*q*r*(s+1)) = p*q*r*(s+1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_increase_l2614_261422


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l2614_261408

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 35)
  (h2 : num_groups = 7)
  (h3 : caps_per_group * num_groups = total_caps) :
  caps_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l2614_261408


namespace NUMINAMATH_CALUDE_pecan_pies_count_l2614_261410

def total_pies : ℕ := 13
def apple_pies : ℕ := 2
def pumpkin_pies : ℕ := 7

theorem pecan_pies_count : total_pies - apple_pies - pumpkin_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_pecan_pies_count_l2614_261410


namespace NUMINAMATH_CALUDE_intersection_sum_l2614_261415

/-- Given two lines that intersect at (2,1), prove that a + b = 2 -/
theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 1 + a) → 
  (1 = (1/3) * 2 + b) → 
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2614_261415


namespace NUMINAMATH_CALUDE_units_digit_of_30_factorial_l2614_261433

theorem units_digit_of_30_factorial (n : ℕ) : n = 30 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_30_factorial_l2614_261433


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l2614_261426

/-- A convex polygon -/
structure ConvexPolygon where
  -- Define properties of a convex polygon
  area : ℝ
  is_convex : Bool

/-- A line in 2D space -/
structure Line where
  -- Define properties of a line

/-- A triangle inscribed in a polygon -/
structure InscribedTriangle (M : ConvexPolygon) where
  -- Define properties of an inscribed triangle
  area : ℝ
  side_parallel_to : Line

/-- Theorem statement -/
theorem inscribed_triangle_area_bound (M : ConvexPolygon) (l : Line) :
  (∃ T : InscribedTriangle M, T.side_parallel_to = l ∧ T.area ≥ 3/8 * M.area) ∧
  (∃ M' : ConvexPolygon, ∃ l' : Line, 
    ∀ T : InscribedTriangle M', T.side_parallel_to = l' → T.area ≤ 3/8 * M'.area) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l2614_261426


namespace NUMINAMATH_CALUDE_equal_area_partition_pentagon_l2614_261488

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon (A B C D E : ℝ × ℝ) : Prop := True

-- Define the area of a triangle
def TriangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- State that a point is inside a pentagon
def InsidePentagon (M : ℝ × ℝ) (A B C D E : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem equal_area_partition_pentagon 
  (A B C D E : ℝ × ℝ) 
  (h_pentagon : Pentagon A B C D E)
  (h_convex : sorry) -- Additional hypothesis for convexity
  (h_equal_areas : TriangleArea A B C = TriangleArea B C D ∧ 
                   TriangleArea B C D = TriangleArea C D E ∧ 
                   TriangleArea C D E = TriangleArea D E A ∧ 
                   TriangleArea D E A = TriangleArea E A B) :
  ∃ M : ℝ × ℝ, 
    InsidePentagon M A B C D E ∧
    TriangleArea M A B = TriangleArea M B C ∧
    TriangleArea M B C = TriangleArea M C D ∧
    TriangleArea M C D = TriangleArea M D E ∧
    TriangleArea M D E = TriangleArea M E A :=
sorry

end NUMINAMATH_CALUDE_equal_area_partition_pentagon_l2614_261488


namespace NUMINAMATH_CALUDE_marble_bag_total_l2614_261454

/-- Represents the total number of marbles in a bag with red, blue, and green marbles. -/
def total_marbles (red : ℕ) (blue : ℕ) (green : ℕ) : ℕ := red + blue + green

/-- Theorem: Given a bag of marbles with only red, blue, and green marbles,
    where the ratio of red to blue to green marbles is 2:3:4,
    and there are 36 blue marbles, the total number of marbles in the bag is 108. -/
theorem marble_bag_total :
  ∀ (red blue green : ℕ),
  red = 2 * n ∧ blue = 3 * n ∧ green = 4 * n →
  blue = 36 →
  total_marbles red blue green = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_bag_total_l2614_261454


namespace NUMINAMATH_CALUDE_sin_45_equals_sqrt2_div_2_l2614_261474

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def angle_45 (x y : ℝ) : Prop := x = y ∧ x > 0 ∧ y > 0

def right_isosceles_triangle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1 ∧ x = y ∧ x > 0 ∧ y > 0

theorem sin_45_equals_sqrt2_div_2 :
  ∀ x y : ℝ, unit_circle x y → angle_45 x y → right_isosceles_triangle x y →
  Real.sin (45 * π / 180) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_45_equals_sqrt2_div_2_l2614_261474


namespace NUMINAMATH_CALUDE_crabapple_sequences_l2614_261480

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 11

/-- The number of times Mrs. Crabapple teaches per week -/
def classes_per_week : ℕ := 5

/-- The number of different sequences of crabapple recipients in one week -/
def num_sequences : ℕ := num_students ^ classes_per_week

theorem crabapple_sequences :
  num_sequences = 161051 :=
by sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l2614_261480


namespace NUMINAMATH_CALUDE_crate_stacking_ways_l2614_261492

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to stack crates to achieve a specific height -/
def countStackingWays (dimensions : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to stack 11 crates to 47ft -/
theorem crate_stacking_ways :
  let dimensions : CrateDimensions := { length := 3, width := 4, height := 5 }
  countStackingWays dimensions 11 47 = 2277 := by
  sorry

end NUMINAMATH_CALUDE_crate_stacking_ways_l2614_261492


namespace NUMINAMATH_CALUDE_last_digit_389_base5_is_4_l2614_261420

-- Define a function to convert a decimal number to base-5
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else loop (m / 5) ((m % 5) :: acc)
    loop n []

-- State the theorem
theorem last_digit_389_base5_is_4 :
  (toBase5 389).getLast? = some 4 :=
sorry

end NUMINAMATH_CALUDE_last_digit_389_base5_is_4_l2614_261420


namespace NUMINAMATH_CALUDE_one_tricycle_l2614_261439

/-- The number of cars in the driveway -/
def num_cars : ℕ := 2

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of bikes in the driveway -/
def num_bikes : ℕ := 2

/-- The number of wheels on each bike -/
def wheels_per_bike : ℕ := 2

/-- The number of trash cans in the driveway -/
def num_trash_cans : ℕ := 1

/-- The number of wheels on each trash can -/
def wheels_per_trash_can : ℕ := 2

/-- The number of roller skates (individual skates, not pairs) -/
def num_roller_skates : ℕ := 2

/-- The number of wheels on each roller skate -/
def wheels_per_roller_skate : ℕ := 4

/-- The total number of wheels in the driveway -/
def total_wheels : ℕ := 25

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

theorem one_tricycle :
  ∃ (num_tricycles : ℕ),
    num_tricycles * wheels_per_tricycle =
      total_wheels -
      (num_cars * wheels_per_car +
       num_bikes * wheels_per_bike +
       num_trash_cans * wheels_per_trash_can +
       num_roller_skates * wheels_per_roller_skate) ∧
    num_tricycles = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_tricycle_l2614_261439


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l2614_261424

theorem min_sum_dimensions (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → a * b * c = 3003 → a + b + c ≥ 57 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l2614_261424


namespace NUMINAMATH_CALUDE_odd_function_implies_m_n_equal_one_f_is_decreasing_k_range_l2614_261427

noncomputable def f (m n x : ℝ) : ℝ := (m - 3^x) / (n + 3^x)

theorem odd_function_implies_m_n_equal_one 
  (h : ∀ x, f m n x = -f m n (-x)) : m = 1 ∧ n = 1 := by sorry

theorem f_is_decreasing : 
  ∀ x y, x < y → f 1 1 x > f 1 1 y := by sorry

theorem k_range (t : ℝ) (h1 : t ∈ Set.Icc 0 4) 
  (h2 : f 1 1 (k - 2*t^2) + f 1 1 (4*t - 2*t^2) < 0) : 
  k > -1 := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_n_equal_one_f_is_decreasing_k_range_l2614_261427


namespace NUMINAMATH_CALUDE_exactly_three_ways_l2614_261428

/-- The sum of consecutive integers from a to b, inclusive -/
def consecutiveSum (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- The predicate that checks if a pair (a, b) satisfies the conditions -/
def isValidPair (a b : ℕ) : Prop :=
  a < b ∧ consecutiveSum a b = 91

/-- The theorem stating that there are exactly 3 valid pairs -/
theorem exactly_three_ways :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 3 ∧ ∀ p, p ∈ s ↔ isValidPair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_ways_l2614_261428


namespace NUMINAMATH_CALUDE_selection_count_theorem_l2614_261469

/-- Represents a grid of people -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a selection of people from the grid -/
structure Selection :=
  (grid : Grid)
  (num_selected : ℕ)

/-- Counts the number of valid selections -/
def count_valid_selections (s : Selection) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem selection_count_theorem (g : Grid) (s : Selection) :
  g.rows = 6 ∧ g.cols = 7 ∧ s.grid = g ∧ s.num_selected = 3 →
  count_valid_selections s = 4200 :=
sorry

end NUMINAMATH_CALUDE_selection_count_theorem_l2614_261469


namespace NUMINAMATH_CALUDE_range_of_a_l2614_261486

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {-1, -3, a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (Set.compl A ∩ B a).Nonempty → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2614_261486


namespace NUMINAMATH_CALUDE_star_seven_three_l2614_261440

def star (a b : ℝ) : ℝ := 2*a + 5*b - a*b + 2

theorem star_seven_three : star 7 3 = 10 := by sorry

end NUMINAMATH_CALUDE_star_seven_three_l2614_261440


namespace NUMINAMATH_CALUDE_expression_equality_l2614_261463

theorem expression_equality : -Real.sqrt 4 + |(-Real.sqrt 2 - 1)| + (π - 2013)^0 - (1/5)^0 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2614_261463


namespace NUMINAMATH_CALUDE_simplify_fraction_l2614_261482

theorem simplify_fraction (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  15 * x^2 * z^3 / (9 * x * y * z^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2614_261482


namespace NUMINAMATH_CALUDE_intersection_distance_is_sqrt_2_l2614_261457

-- Define the two equations
def equation1 (x y : ℝ) : Prop := x^2 + y = 12
def equation2 (x y : ℝ) : Prop := x + y = 12

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ equation1 x y ∧ equation2 x y}

-- State the theorem
theorem intersection_distance_is_sqrt_2 :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt 2 = Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_is_sqrt_2_l2614_261457


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l2614_261453

theorem fuel_cost_savings
  (old_efficiency : ℝ)
  (old_fuel_cost : ℝ)
  (efficiency_increase : ℝ)
  (fuel_cost_increase : ℝ)
  (h1 : efficiency_increase = 0.6)
  (h2 : fuel_cost_increase = 0.3)
  : (1 - (1 + fuel_cost_increase) / (1 + efficiency_increase)) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l2614_261453


namespace NUMINAMATH_CALUDE_no_water_overflow_l2614_261445

/-- Represents the dimensions and properties of a cylindrical container and an iron block. -/
structure ContainerProblem where
  container_depth : ℝ
  container_outer_diameter : ℝ
  container_wall_thickness : ℝ
  water_depth : ℝ
  block_diameter : ℝ
  block_height : ℝ

/-- Calculates the volume of water that will overflow when an iron block is placed in a cylindrical container. -/
noncomputable def water_overflow (p : ContainerProblem) : ℝ :=
  let container_inner_radius := (p.container_outer_diameter - 2 * p.container_wall_thickness) / 2
  let initial_water_volume := Real.pi * container_inner_radius ^ 2 * p.water_depth
  let container_max_volume := Real.pi * container_inner_radius ^ 2 * p.container_depth
  let block_volume := Real.pi * (p.block_diameter / 2) ^ 2 * p.block_height
  let new_total_volume := container_max_volume - block_volume
  max (initial_water_volume - new_total_volume) 0

/-- Theorem stating that no water will overflow in the given problem. -/
theorem no_water_overflow : 
  let problem : ContainerProblem := {
    container_depth := 30,
    container_outer_diameter := 22,
    container_wall_thickness := 1,
    water_depth := 27.5,
    block_diameter := 10,
    block_height := 30
  }
  water_overflow problem = 0 := by sorry

end NUMINAMATH_CALUDE_no_water_overflow_l2614_261445
