import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3918_391869

/-- The polynomial we're working with -/
def p (x : ℝ) : ℝ := 5 * (2 * x^5 - x^3 + 2 * x^2 - 3)

/-- The coefficients of the expanded polynomial -/
def coefficients : List ℝ := [10, 0, -5, 10, 0, -15]

/-- Sum of squares of coefficients -/
def sum_of_squares (l : List ℝ) : ℝ := (l.map (λ x => x^2)).sum

theorem sum_of_squares_of_coefficients :
  sum_of_squares coefficients = 450 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3918_391869


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3918_391811

/-- A line tangent to a cubic curve -/
structure TangentLine where
  k : ℝ
  a : ℝ
  b : ℝ

/-- The tangent line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (t : TangentLine) : Prop :=
  3 = t.k * 1 + 1 ∧
  3 = 1^3 + t.a * 1 + t.b ∧
  t.k = 3 * 1^2 + t.a

theorem tangent_line_b_value (t : TangentLine) (h : is_tangent t) : t.b = 3 := by
  sorry

#check tangent_line_b_value

end NUMINAMATH_CALUDE_tangent_line_b_value_l3918_391811


namespace NUMINAMATH_CALUDE_rectangle_square_equal_area_l3918_391896

theorem rectangle_square_equal_area : 
  ∀ (rectangle_width rectangle_length square_side : ℝ),
    rectangle_width = 2 →
    rectangle_length = 18 →
    square_side = 6 →
    rectangle_width * rectangle_length = square_side * square_side := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_equal_area_l3918_391896


namespace NUMINAMATH_CALUDE_min_fence_length_is_28_l3918_391853

/-- Represents a rectangular flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the minimum fence length required for a flower bed with one side against a wall -/
def minFenceLength (fb : FlowerBed) : ℝ :=
  2 * fb.width + fb.length

/-- The specific flower bed in the problem -/
def problemFlowerBed : FlowerBed :=
  { length := 12, width := 8 }

theorem min_fence_length_is_28 :
  minFenceLength problemFlowerBed = 28 := by
  sorry

#eval minFenceLength problemFlowerBed

end NUMINAMATH_CALUDE_min_fence_length_is_28_l3918_391853


namespace NUMINAMATH_CALUDE_inequality_always_true_l3918_391889

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : 
  (a + c > b + d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a - c > b - d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a * c > b * d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a / c > b / d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3918_391889


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3918_391839

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 20| + |x - 18| = |2*x - 36| :=
by
  -- The unique solution is x = 19
  use 19
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3918_391839


namespace NUMINAMATH_CALUDE_union_equals_N_l3918_391846

def M : Set ℝ := {x | x - x < 0}
def N : Set ℝ := {x | -3 < x ∧ x < 3}

theorem union_equals_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_union_equals_N_l3918_391846


namespace NUMINAMATH_CALUDE_teresa_jogging_speed_l3918_391894

theorem teresa_jogging_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 25 → time = 5 → speed = distance / time → speed = 5 :=
by sorry

end NUMINAMATH_CALUDE_teresa_jogging_speed_l3918_391894


namespace NUMINAMATH_CALUDE_matt_flour_bags_matt_flour_bags_correct_l3918_391833

theorem matt_flour_bags (cookies_per_batch : ℕ) (flour_per_batch : ℕ) 
  (flour_per_bag : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) : ℕ :=
  let total_cookies := cookies_eaten + cookies_left
  let total_dozens := total_cookies / cookies_per_batch
  let total_flour := total_dozens * flour_per_batch
  total_flour / flour_per_bag

#check matt_flour_bags 12 2 5 15 105 = 4

theorem matt_flour_bags_correct : matt_flour_bags 12 2 5 15 105 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matt_flour_bags_matt_flour_bags_correct_l3918_391833


namespace NUMINAMATH_CALUDE_quadratic_sum_of_p_q_l3918_391845

/-- Given a quadratic equation 9x^2 - 54x + 63 = 0, when transformed
    into the form (x + p)^2 = q, the sum of p and q is equal to -1 -/
theorem quadratic_sum_of_p_q : ∃ (p q : ℝ),
  (∀ x, 9 * x^2 - 54 * x + 63 = 0 ↔ (x + p)^2 = q) ∧
  p + q = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_p_q_l3918_391845


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l3918_391807

noncomputable def y (a x : ℝ) : ℝ := a * Real.tan (Real.sqrt (a / x - 1))

theorem y_satisfies_equation (a x : ℝ) (h1 : x ≠ 0) (h2 : a / x - 1 ≥ 0) :
  a^2 + (y a x)^2 + 2 * x * Real.sqrt (a * x - x^2) * (deriv (y a) x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l3918_391807


namespace NUMINAMATH_CALUDE_team_a_finishes_faster_l3918_391837

/-- Proves that Team A finishes 3 hours faster than Team R given the specified conditions --/
theorem team_a_finishes_faster (course_distance : ℝ) (team_r_speed : ℝ) (speed_difference : ℝ) :
  course_distance = 300 →
  team_r_speed = 20 →
  speed_difference = 5 →
  let team_a_speed := team_r_speed + speed_difference
  let team_r_time := course_distance / team_r_speed
  let team_a_time := course_distance / team_a_speed
  team_r_time - team_a_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_a_finishes_faster_l3918_391837


namespace NUMINAMATH_CALUDE_river_crossing_theorem_l3918_391883

/-- Calculates the time required for all explorers to cross a river --/
def river_crossing_time (num_explorers : ℕ) (boat_capacity : ℕ) (crossing_time : ℕ) : ℕ :=
  let first_trip := boat_capacity
  let remaining_explorers := num_explorers - first_trip
  let subsequent_trips := (remaining_explorers + 4) / 5  -- Ceiling division
  let total_crossings := 2 * subsequent_trips + 1
  total_crossings * crossing_time

theorem river_crossing_theorem :
  river_crossing_time 60 6 3 = 69 := by
  sorry

end NUMINAMATH_CALUDE_river_crossing_theorem_l3918_391883


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3918_391829

/-- Given a boat traveling downstream with a stream rate of 5 km/hr and covering 84 km in 4 hours,
    the speed of the boat in still water is 16 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (stream_rate : ℝ) (distance : ℝ) (time : ℝ) (boat_speed : ℝ),
    stream_rate = 5 →
    distance = 84 →
    time = 4 →
    distance = (boat_speed + stream_rate) * time →
    boat_speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3918_391829


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3918_391873

theorem polynomial_divisibility (k l m n : ℕ) : 
  ∃ q : Polynomial ℤ, (X^4*k + X^(4*l+1) + X^(4*m+2) + X^(4*n+3)) = (X^3 + X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3918_391873


namespace NUMINAMATH_CALUDE_circle_C_theorem_l3918_391854

/-- Definition of the circle C with parameter t -/
def circle_C (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*t*x - 2*t^2*y + 4*t - 4 = 0

/-- Definition of the line on which the center of C lies -/
def center_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- First possible equation of circle C -/
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 8 = 0

/-- Second possible equation of circle C -/
def circle_C2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 4 = 0

/-- The fixed point that C passes through -/
def fixed_point : ℝ × ℝ := (2, 0)

theorem circle_C_theorem :
  ∀ t : ℝ,
  (∃ x y : ℝ, circle_C t x y ∧ center_line x y) →
  ((∀ x y : ℝ, circle_C t x y ↔ circle_C1 x y) ∨
   (∀ x y : ℝ, circle_C t x y ↔ circle_C2 x y)) ∧
  circle_C t fixed_point.1 fixed_point.2 :=
sorry

end NUMINAMATH_CALUDE_circle_C_theorem_l3918_391854


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3918_391898

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3918_391898


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3918_391858

/-- An arithmetic sequence with the given properties has a common difference of either 1 or -1. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence property
  (h_product : a 1 * a 3 = 8)  -- Given condition: a₁ · a₃ = 8
  (h_second : a 2 = 3)  -- Given condition: a₂ = 3
  : ∃ d : ℝ, (d = 1 ∨ d = -1) ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3918_391858


namespace NUMINAMATH_CALUDE_class_average_weight_l3918_391818

/-- Given two sections A and B in a class, with their respective number of students and average weights,
    prove that the average weight of the whole class is as calculated. -/
theorem class_average_weight 
  (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_A : ℝ) (avg_weight_B : ℝ) :
  students_A = 40 →
  students_B = 20 →
  avg_weight_A = 50 →
  avg_weight_B = 40 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 46.67 :=
by sorry

end NUMINAMATH_CALUDE_class_average_weight_l3918_391818


namespace NUMINAMATH_CALUDE_root_equation_value_l3918_391819

theorem root_equation_value (m : ℝ) (h : m^2 - 3*m - 1 = 0) : 2*m^2 - 6*m + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3918_391819


namespace NUMINAMATH_CALUDE_triangle_properties_l3918_391835

noncomputable section

open Real

/-- Given a triangle ABC with D as the midpoint of AB, prove that under certain conditions,
    angle C is π/3 and the maximum value of CD²/(a²+b²) is 3/8. -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (D : ℝ × ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b - c * cos A = a * (sqrt 3 * sin C - 1) →
  sin (A + B) * cos (C - π / 6) = 3 / 4 →
  D = ((cos A + cos B) / 2, (sin A + sin B) / 2) →
  C = π / 3 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x * x + y * y + x * y) / (4 * (x * x + y * y)) ≤ 3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3918_391835


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l3918_391828

theorem square_perimeter_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 65) (h4 : a^2 - b^2 = 33) : 
  4*a + 4*b = 44 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l3918_391828


namespace NUMINAMATH_CALUDE_thomas_blocks_count_l3918_391899

/-- The number of wooden blocks Thomas used in total -/
def total_blocks (stack1 stack2 stack3 stack4 stack5 : ℕ) : ℕ :=
  stack1 + stack2 + stack3 + stack4 + stack5

/-- Theorem stating the total number of blocks Thomas used -/
theorem thomas_blocks_count :
  ∃ (stack1 stack2 stack3 stack4 stack5 : ℕ),
    stack1 = 7 ∧
    stack2 = stack1 + 3 ∧
    stack3 = stack2 - 6 ∧
    stack4 = stack3 + 10 ∧
    stack5 = 2 * stack2 ∧
    total_blocks stack1 stack2 stack3 stack4 stack5 = 55 :=
by
  sorry


end NUMINAMATH_CALUDE_thomas_blocks_count_l3918_391899


namespace NUMINAMATH_CALUDE_polygon_perimeter_equals_rectangle_perimeter_l3918_391843

/-- A polygon that forms part of a rectangle -/
structure PartialRectanglePolygon where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ

/-- The perimeter of a rectangle -/
def rectanglePerimeter (rect : PartialRectanglePolygon) : ℝ :=
  2 * (rect.length + rect.width)

/-- The perimeter of the polygon that forms part of the rectangle -/
def polygonPerimeter (poly : PartialRectanglePolygon) : ℝ :=
  rectanglePerimeter poly

theorem polygon_perimeter_equals_rectangle_perimeter (poly : PartialRectanglePolygon) :
  polygonPerimeter poly = rectanglePerimeter poly := by
  sorry

#check polygon_perimeter_equals_rectangle_perimeter

end NUMINAMATH_CALUDE_polygon_perimeter_equals_rectangle_perimeter_l3918_391843


namespace NUMINAMATH_CALUDE_binomial_coefficient_and_increase_l3918_391816

variable (n : ℕ)

theorem binomial_coefficient_and_increase :
  (Nat.choose n 2 = n * (n - 1) / 2) ∧
  (Nat.choose (n + 1) 2 - Nat.choose n 2 = n) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_and_increase_l3918_391816


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3918_391803

theorem product_of_three_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * (3 ^ (1/4)))
  (hac : a * c = 50 * (3 ^ (1/4)))
  (hbc : b * c = 18 * (3 ^ (1/4))) :
  a * b * c = 120 * (3 ^ (1/4)) := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3918_391803


namespace NUMINAMATH_CALUDE_books_movies_difference_l3918_391824

theorem books_movies_difference (total_books total_movies : ℕ) 
  (h1 : total_books = 10) 
  (h2 : total_movies = 6) : 
  total_books - total_movies = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_movies_difference_l3918_391824


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3918_391862

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := 2

/-- Theorem: The vertex of the quadratic function f(x) = x^2 - 2x + 3 is at (1, 2) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3918_391862


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l3918_391852

/-- The length of a diagonal in a quadrilateral with given offsets and area -/
theorem diagonal_length_of_quadrilateral (offset1 offset2 area : ℝ) 
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 195) :
  ∃ d : ℝ, d = 26 ∧ (1/2 * d * offset1) + (1/2 * d * offset2) = area :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l3918_391852


namespace NUMINAMATH_CALUDE_sally_has_six_cards_l3918_391812

/-- The number of baseball cards Sally has after selling some to Sara -/
def sallys_remaining_cards (initial_cards torn_cards cards_sold : ℕ) : ℕ :=
  initial_cards - torn_cards - cards_sold

/-- Theorem stating that Sally has 6 cards remaining -/
theorem sally_has_six_cards :
  sallys_remaining_cards 39 9 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_six_cards_l3918_391812


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3918_391826

theorem triangle_angle_calculation (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hC : 0 < B) (hD : B < π) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = π / 3) :
  ∃ (A : ℝ), 
    0 < A ∧ A < π / 2 ∧ 
    Real.sin A = (a * Real.sin B) / b ∧
    A = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3918_391826


namespace NUMINAMATH_CALUDE_spy_is_B_l3918_391895

-- Define the possible roles
inductive Role
| Knight
| Liar
| Spy

-- Define the defendants
inductive Defendant
| A
| B
| C

-- Define a function to represent the role of each defendant
def role : Defendant → Role := sorry

-- Define the answers given by defendants
def answer_A : Bool := sorry
def answer_B : Bool := sorry
def answer_remaining : Bool := sorry

-- Define which defendant was released
def released : Defendant := sorry

-- Define which defendant was asked the final question
def final_asked : Defendant := sorry

-- Axioms based on the problem conditions
axiom different_roles : 
  ∃! (a b c : Defendant), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    role a = Role.Knight ∧ role b = Role.Liar ∧ role c = Role.Spy

axiom judge_deduction : 
  ∃! (spy : Defendant), role spy = Role.Spy

axiom released_not_spy : 
  role released ≠ Role.Spy

axiom final_question_neighbor : 
  final_asked ≠ released ∧ 
  (final_asked = Defendant.A ∨ final_asked = Defendant.B)

-- The theorem to prove
theorem spy_is_B : 
  role Defendant.B = Role.Spy := by sorry

end NUMINAMATH_CALUDE_spy_is_B_l3918_391895


namespace NUMINAMATH_CALUDE_division_of_decimals_l3918_391872

theorem division_of_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3918_391872


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l3918_391802

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3^(n-2) + k, prove that k = -1/9 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n : ℕ, S n = 3^(n - 2) + k) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n - 1)) →
  (∀ n : ℕ, n ≥ 2 → a n / a (n - 1) = a (n + 1) / a n) →
  k = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l3918_391802


namespace NUMINAMATH_CALUDE_equation_solution_l3918_391810

theorem equation_solution (y : ℝ) : 
  (y / 5) / 3 = 15 / (y / 3) → y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3918_391810


namespace NUMINAMATH_CALUDE_square_difference_simplification_l3918_391855

theorem square_difference_simplification (y : ℝ) (h : y^2 ≥ 16) :
  (4 - Real.sqrt (y^2 - 16))^2 = y^2 - 8 * Real.sqrt (y^2 - 16) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_simplification_l3918_391855


namespace NUMINAMATH_CALUDE_distance_calculation_l3918_391863

theorem distance_calculation (D : ℝ) : 
  (1/4 : ℝ) * D + (1/2 : ℝ) * D + 10 = D → D = 40 := by
sorry

end NUMINAMATH_CALUDE_distance_calculation_l3918_391863


namespace NUMINAMATH_CALUDE_circumradius_inscribed_radius_inequality_l3918_391801

/-- A triangle with its circumscribed and inscribed circles -/
structure Triangle where
  -- Radius of the circumscribed circle
  R : ℝ
  -- Radius of the inscribed circle
  r : ℝ
  -- Predicate indicating if the triangle is equilateral
  is_equilateral : Prop

/-- The radius of the circumscribed circle is at least twice the radius of the inscribed circle,
    with equality if and only if the triangle is equilateral -/
theorem circumradius_inscribed_radius_inequality (t : Triangle) :
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ t.is_equilateral) := by
  sorry

end NUMINAMATH_CALUDE_circumradius_inscribed_radius_inequality_l3918_391801


namespace NUMINAMATH_CALUDE_phone_number_probability_correct_probability_l3918_391809

theorem phone_number_probability : ℝ → Prop :=
  fun p => (∀ n : ℕ, n ≤ 3 → n > 0 → (1 - (9/10)^n) ≤ p) ∧ p ≤ 3/10

theorem correct_probability : phone_number_probability (3/10) := by
  sorry

end NUMINAMATH_CALUDE_phone_number_probability_correct_probability_l3918_391809


namespace NUMINAMATH_CALUDE_min_volume_ratio_l3918_391893

theorem min_volume_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * (x + y) * (y + z) * (z + x) / (x * y * z) ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_min_volume_ratio_l3918_391893


namespace NUMINAMATH_CALUDE_a_2018_mod_49_l3918_391831

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : a 2018 % 49 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_2018_mod_49_l3918_391831


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l3918_391825

theorem sqrt_twelve_minus_sqrt_three : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l3918_391825


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3918_391875

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > b,
    if the angle between its asymptotes is 45°, then a/b = √2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ t : ℝ, ∃ x y : ℝ, y = (b / a) * x ∨ y = -(b / a) * x) →
  (Real.pi / 4 : ℝ) = Real.arctan ((b / a - (-b / a)) / (1 + (b / a) * (-b / a))) →
  a / b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3918_391875


namespace NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l3918_391881

theorem square_fraction_is_perfect_square (a b : ℕ+) 
  (h : ∃ k : ℕ, (a + b)^2 = k * (4 * a * b + 1)) : 
  ∃ n : ℕ, (a + b)^2 / (4 * a * b + 1) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l3918_391881


namespace NUMINAMATH_CALUDE_divisors_of_8_factorial_l3918_391871

/-- The number of positive divisors of n! -/
def num_divisors_factorial (n : ℕ) : ℕ :=
  sorry

/-- 8! has 96 positive divisors -/
theorem divisors_of_8_factorial :
  num_divisors_factorial 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8_factorial_l3918_391871


namespace NUMINAMATH_CALUDE_tire_circumference_l3918_391805

/-- The circumference of a tire given its rotations per minute and the car's speed -/
theorem tire_circumference (rotations_per_minute : ℝ) (car_speed_kmh : ℝ) : 
  rotations_per_minute = 400 → car_speed_kmh = 24 → 
  (car_speed_kmh * 1000 / 60) / rotations_per_minute = 1 := by
  sorry

#check tire_circumference

end NUMINAMATH_CALUDE_tire_circumference_l3918_391805


namespace NUMINAMATH_CALUDE_second_sum_proof_l3918_391844

/-- Given a total sum and interest conditions, prove the second sum -/
theorem second_sum_proof (total : ℝ) (first : ℝ) (second : ℝ) : 
  total = 2743 →
  first + second = total →
  (first * 3 / 100 * 8) = (second * 5 / 100 * 3) →
  second = 1688 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_proof_l3918_391844


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3918_391864

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 4 * x + 1 / x ≥ 4 ∧ 
  (4 * x + 1 / x = 4 ↔ x = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3918_391864


namespace NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_fourth_power_l3918_391808

theorem smallest_integer_larger_than_root_sum_fourth_power :
  ∃ n : ℕ, n = 248 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^4) ∧
  (n : ℝ) > (Real.sqrt 5 + Real.sqrt 3)^4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_fourth_power_l3918_391808


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l3918_391867

open Set

-- Define sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < -1} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x ≤ 3 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l3918_391867


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3918_391830

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * Real.sqrt 2 * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3918_391830


namespace NUMINAMATH_CALUDE_exam_duration_l3918_391834

/-- Represents a time on a clock face -/
structure ClockTime where
  hours : ℝ
  minutes : ℝ
  valid : 0 ≤ hours ∧ hours < 12 ∧ 0 ≤ minutes ∧ minutes < 60

/-- Checks if two clock times are equivalent when hour and minute hands are swapped -/
def equivalent_when_swapped (t1 t2 : ClockTime) : Prop :=
  t1.hours = t2.minutes / 5 ∧ t1.minutes = t2.hours * 5

/-- The main theorem statement -/
theorem exam_duration :
  ∀ (start_time end_time : ClockTime),
    9 ≤ start_time.hours ∧ start_time.hours < 10 →
    1 ≤ end_time.hours ∧ end_time.hours < 2 →
    equivalent_when_swapped start_time end_time →
    end_time.hours - start_time.hours + (end_time.minutes - start_time.minutes) / 60 = 60 / 13 :=
sorry

end NUMINAMATH_CALUDE_exam_duration_l3918_391834


namespace NUMINAMATH_CALUDE_problem_solution_l3918_391800

def A (x y : ℝ) : ℝ := 3 * x^2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

theorem problem_solution (x y : ℝ) :
  (A x y + 3 * B x y = 5 * x * y - 2 * x - 4) ∧
  (∀ x, A x y + 3 * B x y = A 0 y + 3 * B 0 y → y = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3918_391800


namespace NUMINAMATH_CALUDE_room_tiling_theorem_l3918_391838

/-- Calculates the number of tiles needed for a room with given dimensions and tile specifications -/
def tiles_needed (room_length room_width border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width - 4 * border_width) + 4 * border_width * border_width
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  let large_tiles := (inner_area + 8) / 9  -- Ceiling division
  border_tiles + large_tiles

/-- The theorem stating that 80 tiles are needed for the given room specifications -/
theorem room_tiling_theorem : tiles_needed 18 14 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_room_tiling_theorem_l3918_391838


namespace NUMINAMATH_CALUDE_initial_leaves_count_l3918_391886

/-- The number of leaves that blew away -/
def leaves_blown_away : ℕ := 244

/-- The number of leaves left -/
def leaves_left : ℕ := 112

/-- The initial number of leaves -/
def initial_leaves : ℕ := leaves_blown_away + leaves_left

theorem initial_leaves_count : initial_leaves = 356 := by
  sorry

end NUMINAMATH_CALUDE_initial_leaves_count_l3918_391886


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3918_391832

-- Define the card numbers for each player
def sora_cards : List Nat := [4, 6]
def heesu_cards : List Nat := [7, 5]
def jiyeon_cards : List Nat := [3, 8]

-- Define a function to calculate the sum of a player's cards
def sum_cards (cards : List Nat) : Nat :=
  cards.sum

-- Theorem statement
theorem heesu_has_greatest_sum :
  sum_cards heesu_cards > sum_cards sora_cards ∧
  sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  sorry


end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3918_391832


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3918_391815

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  (a^3 * b^3) / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  (a^3 * c^3) / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  (b^3 * c^3) / ((b^3 - a^2 * c) * (c^3 - a^2 * b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3918_391815


namespace NUMINAMATH_CALUDE_second_draw_probability_l3918_391804

/-- Represents the total number of items -/
def total_items : ℕ := 10

/-- Represents the number of genuine items -/
def genuine_items : ℕ := 6

/-- Represents the number of defective items -/
def defective_items : ℕ := 4

/-- Represents the probability of drawing a genuine item on the second draw,
    given that the first item drawn is genuine -/
def prob_second_genuine : ℚ := 5 / 9

theorem second_draw_probability :
  total_items = genuine_items + defective_items →
  genuine_items > 0 →
  prob_second_genuine = (genuine_items - 1) / (total_items - 1) :=
by sorry

end NUMINAMATH_CALUDE_second_draw_probability_l3918_391804


namespace NUMINAMATH_CALUDE_mary_baking_cake_l3918_391836

theorem mary_baking_cake (total_flour sugar_needed : ℕ) 
  (h1 : total_flour = 11)
  (h2 : sugar_needed = 7)
  (h3 : total_flour - flour_put_in = sugar_needed + 2) :
  flour_put_in = 2 :=
by sorry

end NUMINAMATH_CALUDE_mary_baking_cake_l3918_391836


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l3918_391876

theorem negative_fractions_comparison : (-1/2 : ℚ) < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l3918_391876


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3918_391874

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = m + 3 * x) ↔ m = 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3918_391874


namespace NUMINAMATH_CALUDE_total_tulips_l3918_391820

def arwen_tulips : ℕ := 20

def elrond_tulips (a : ℕ) : ℕ := 2 * a

theorem total_tulips : arwen_tulips + elrond_tulips arwen_tulips = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_l3918_391820


namespace NUMINAMATH_CALUDE_largest_in_set_l3918_391856

def S : Set ℝ := {0.109, 0.2, 0.111, 0.114, 0.19}

theorem largest_in_set : ∀ x ∈ S, x ≤ 0.2 := by sorry

end NUMINAMATH_CALUDE_largest_in_set_l3918_391856


namespace NUMINAMATH_CALUDE_four_mutually_tangent_circles_exist_l3918_391849

-- Define a circle with a center point and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Theorem statement
theorem four_mutually_tangent_circles_exist : 
  ∃ (c1 c2 c3 c4 : Circle),
    are_externally_tangent c1 c2 ∧
    are_externally_tangent c1 c3 ∧
    are_externally_tangent c1 c4 ∧
    are_externally_tangent c2 c3 ∧
    are_externally_tangent c2 c4 ∧
    are_externally_tangent c3 c4 :=
sorry

end NUMINAMATH_CALUDE_four_mutually_tangent_circles_exist_l3918_391849


namespace NUMINAMATH_CALUDE_product_equality_implies_sum_l3918_391850

theorem product_equality_implies_sum (g h a b : ℝ) :
  (∀ d : ℝ, (8 * d^2 - 4 * d + g) * (2 * d^2 + h * d - 7) = 16 * d^4 - 28 * d^3 + a * h^2 * d^2 - b * d + 49) →
  g + h = -3 := by
sorry

end NUMINAMATH_CALUDE_product_equality_implies_sum_l3918_391850


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3918_391821

theorem complex_magnitude_equality (n : ℝ) :
  n > 0 ∧ Complex.abs (2 + n * Complex.I) = 4 * Real.sqrt 5 ↔ n = 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3918_391821


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3918_391806

/-- The number of teams in the frisbee league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team for the committee -/
def host_committee_size : ℕ := 4

/-- The number of members selected from each non-host team for the committee -/
def non_host_committee_size : ℕ := 2

/-- The total number of members in the tournament committee -/
def total_committee_size : ℕ := 12

/-- Theorem stating the total number of possible tournament committees -/
theorem tournament_committee_count :
  (num_teams : ℕ) * (Nat.choose team_size host_committee_size) *
  (Nat.choose team_size non_host_committee_size ^ (num_teams - 1)) = 340342925 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3918_391806


namespace NUMINAMATH_CALUDE_hostel_expenditure_calculation_hostel_new_expenditure_l3918_391859

/-- Calculates the new total expenditure of a hostel after accommodating more students. -/
theorem hostel_expenditure_calculation (initial_students : ℕ) (budget_decrease : ℚ) 
  (expenditure_increase : ℚ) (new_students : ℕ) : ℚ :=
  let new_total_students := initial_students + new_students
  let original_budget := (new_total_students * budget_decrease + expenditure_increase) / 
    (new_total_students - initial_students)
  let original_expenditure := initial_students * original_budget
  original_expenditure + expenditure_increase

/-- Proves that the new total expenditure of the hostel is 5775 rupees. -/
theorem hostel_new_expenditure : 
  hostel_expenditure_calculation 100 10 400 32 = 5775 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_calculation_hostel_new_expenditure_l3918_391859


namespace NUMINAMATH_CALUDE_not_necessarily_p_or_q_l3918_391880

theorem not_necessarily_p_or_q (P Q : Prop) 
  (h1 : ¬P) 
  (h2 : ¬(P ∧ Q)) : 
  ¬∀ (P Q : Prop), (¬P ∧ ¬(P ∧ Q)) → (P ∨ Q) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_p_or_q_l3918_391880


namespace NUMINAMATH_CALUDE_prove_a_value_l3918_391865

/-- Custom operation @ for positive integers -/
def custom_op (k : ℕ+) (j : ℕ+) : ℕ+ :=
  sorry

/-- Given b and t, prove a = 1060 -/
theorem prove_a_value (b t : ℚ) (h1 : b = 2120) (h2 : t = 1/2) :
  ∃ a : ℚ, t = a / b ∧ a = 1060 := by
  sorry

end NUMINAMATH_CALUDE_prove_a_value_l3918_391865


namespace NUMINAMATH_CALUDE_replaced_man_age_l3918_391878

theorem replaced_man_age (n : ℕ) (avg_increase : ℝ) (man1_age : ℕ) (women_avg_age : ℝ) :
  n = 10 ∧ 
  avg_increase = 6 ∧ 
  man1_age = 18 ∧ 
  women_avg_age = 50 → 
  ∃ (original_avg : ℝ) (man2_age : ℕ),
    n * (original_avg + avg_increase) = n * original_avg + 2 * women_avg_age - (man1_age + man2_age) ∧
    man2_age = 22 := by
  sorry

#check replaced_man_age

end NUMINAMATH_CALUDE_replaced_man_age_l3918_391878


namespace NUMINAMATH_CALUDE_square_area_from_perspective_l3918_391879

-- Define a square
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

-- Define a parallelogram
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ

-- Define the perspective drawing relation
def perspective_drawing (s : Square) (p : Parallelogram) : Prop :=
  (p.side1 = s.side ∨ p.side1 = s.side / 2) ∧ 
  (p.side2 = s.side ∨ p.side2 = s.side / 2)

-- Theorem statement
theorem square_area_from_perspective (s : Square) (p : Parallelogram) :
  perspective_drawing s p → (p.side1 = 4 ∨ p.side2 = 4) → (s.area = 16 ∨ s.area = 64) :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_perspective_l3918_391879


namespace NUMINAMATH_CALUDE_bike_ride_time_l3918_391882

/-- Represents the problem of calculating the time to cover a highway stretch on a bike --/
theorem bike_ride_time (highway_length : Real) (highway_width : Real) (bike_speed : Real) :
  highway_length = 2 → -- 2 miles
  highway_width = 60 / 5280 → -- 60 feet converted to miles
  bike_speed = 6 → -- 6 miles per hour
  (π * highway_length) / bike_speed = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_time_l3918_391882


namespace NUMINAMATH_CALUDE_circle_center_l3918_391823

/-- The equation of a circle in the form (x + h)² + (y + k)² = r², where (h, k) is the center. -/
def CircleEquation (h k r : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (x + h)^2 + (y + k)^2 = r^2

/-- The center of the circle (x + 2)² + y² = 5 is (-2, 0). -/
theorem circle_center :
  ∃ (h k : ℝ), CircleEquation h k (Real.sqrt 5) = CircleEquation 2 0 (Real.sqrt 5) ∧ h = -2 ∧ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3918_391823


namespace NUMINAMATH_CALUDE_complex_number_real_twice_imaginary_l3918_391813

theorem complex_number_real_twice_imaginary (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) / (4 - 3 * Complex.I) + m / 25
  (z.re = 2 * z.im) → m = -1/5 := by
sorry

end NUMINAMATH_CALUDE_complex_number_real_twice_imaginary_l3918_391813


namespace NUMINAMATH_CALUDE_equation_solution_l3918_391897

theorem equation_solution : 
  {x : ℝ | (x + 2)^4 + (x - 4)^4 = 272} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3918_391897


namespace NUMINAMATH_CALUDE_odometer_puzzle_l3918_391861

theorem odometer_puzzle (a b c d : ℕ) (h1 : a ≥ 1) (h2 : a + b + c + d = 10)
  (h3 : ∃ (x : ℕ), 1000 * (d - a) + 100 * (c - b) + 10 * (b - c) + (a - d) = 65 * x) :
  a^2 + b^2 + c^2 + d^2 = 42 := by
sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l3918_391861


namespace NUMINAMATH_CALUDE_pet_sitting_charge_per_night_l3918_391870

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def total_payment : ℕ := 65

theorem pet_sitting_charge_per_night :
  (total_payment : ℚ) / (num_cats + num_dogs : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_sitting_charge_per_night_l3918_391870


namespace NUMINAMATH_CALUDE_verify_coin_weights_l3918_391842

/-- Represents a coin with a denomination and weight -/
structure Coin where
  denomination : ℕ
  weight : ℕ

/-- Represents a balance scale measurement -/
def BalanceMeasurement := List Coin → List Coin → Bool

/-- Checks if the total weight of coins on both sides of the scale is equal -/
def isBalanced (coins1 coins2 : List Coin) : Bool :=
  (coins1.map (λ c => c.weight)).sum = (coins2.map (λ c => c.weight)).sum

/-- Represents the available weight for measurements -/
def WeightValue : ℕ := 9

/-- Theorem stating that it's possible to verify the weights of the coins -/
theorem verify_coin_weights (coins : List Coin) 
  (h1 : coins.length = 4)
  (h2 : coins.map (λ c => c.denomination) = [1, 2, 3, 5])
  (h3 : ∀ c ∈ coins, c.weight = c.denomination)
  (balance : BalanceMeasurement) 
  (h4 : ∀ c1 c2, balance c1 c2 = isBalanced c1 c2) :
  ∃ (measurements : List (List Coin × List Coin)),
    measurements.length ≤ 4 ∧ 
    (∀ m ∈ measurements, balance m.1 m.2 = true) ∧
    (∀ c ∈ coins, c.weight = c.denomination) :=
  sorry

end NUMINAMATH_CALUDE_verify_coin_weights_l3918_391842


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3918_391887

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π →
  B > 0 ∧ B < π →
  C > 0 ∧ C < π →
  A + B + C = π →
  a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C →
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3918_391887


namespace NUMINAMATH_CALUDE_subset_condition_intersection_condition_l3918_391851

open Set Real

-- Define set A
def A : Set ℝ := {x : ℝ | |x + 2| < 3}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- Theorem for part 1
theorem subset_condition (m : ℝ) : A ⊆ B m → m ≤ -5 := by sorry

-- Theorem for part 2
theorem intersection_condition (m n : ℝ) : A ∩ B m = Ioo (-1) n → m = -1 ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_condition_l3918_391851


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_29_4_l3918_391884

theorem floor_plus_self_eq_29_4 (x : ℚ) :
  (⌊x⌋ : ℚ) + x = 29/4 → x = 29/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_29_4_l3918_391884


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3918_391885

theorem quadratic_is_square_of_binomial (x k : ℝ) : 
  (∃ a b : ℝ, x^2 - 20*x + k = (a*x + b)^2) ↔ k = 100 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3918_391885


namespace NUMINAMATH_CALUDE_mark_to_jaydon_ratio_l3918_391857

/-- Represents the number of cans brought by each person -/
structure Cans where
  rachel : ℕ
  jaydon : ℕ
  mark : ℕ

/-- The conditions of the food drive problem -/
def FoodDrive (c : Cans) : Prop :=
  c.mark = 100 ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  c.rachel + c.jaydon + c.mark = 135

/-- The theorem to be proved -/
theorem mark_to_jaydon_ratio (c : Cans) (h : FoodDrive c) : 
  c.mark / c.jaydon = 4 := by
  sorry

#check mark_to_jaydon_ratio

end NUMINAMATH_CALUDE_mark_to_jaydon_ratio_l3918_391857


namespace NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l3918_391866

theorem unique_prime_pair_divisibility : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (p^p + q^q + 1) % (p * q) = 0 → 
    (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l3918_391866


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3918_391848

theorem difference_of_squares_special_case : (527 : ℕ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3918_391848


namespace NUMINAMATH_CALUDE_marias_score_l3918_391892

/-- Given that Maria's score is 50 points more than Tom's and their average score is 105,
    prove that Maria's score is 130. -/
theorem marias_score (tom_score : ℕ) : 
  let maria_score := tom_score + 50
  let average := (maria_score + tom_score) / 2
  average = 105 → maria_score = 130 := by
sorry

end NUMINAMATH_CALUDE_marias_score_l3918_391892


namespace NUMINAMATH_CALUDE_opposite_solutions_imply_a_l3918_391891

theorem opposite_solutions_imply_a (a : ℝ) : 
  (∃ x y : ℝ, 2 * (x - 1) - 6 = 0 ∧ 1 - (3 * a - x) / 3 = 0 ∧ x = -y) → 
  a = -1/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_imply_a_l3918_391891


namespace NUMINAMATH_CALUDE_abs_value_inequality_solution_set_l3918_391868

theorem abs_value_inequality_solution_set :
  {x : ℝ | |x| > 1} = {x : ℝ | x > 1 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_solution_set_l3918_391868


namespace NUMINAMATH_CALUDE_number_ordering_l3918_391814

theorem number_ordering : 
  (3 : ℚ) / 8 < (3 : ℚ) / 4 ∧ 
  (3 : ℚ) / 4 < (7 : ℚ) / 5 ∧ 
  (7 : ℚ) / 5 < (143 : ℚ) / 100 ∧ 
  (143 : ℚ) / 100 < (13 : ℚ) / 8 := by
sorry

end NUMINAMATH_CALUDE_number_ordering_l3918_391814


namespace NUMINAMATH_CALUDE_banana_tree_problem_l3918_391890

/-- The number of bananas initially on the tree -/
def initial_bananas : ℕ := 1180

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left : ℕ := 500

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 170

/-- The number of bananas remaining in Raj's basket -/
def bananas_in_basket : ℕ := 3 * bananas_eaten

theorem banana_tree_problem :
  initial_bananas = bananas_left + bananas_eaten + bananas_in_basket :=
by sorry

end NUMINAMATH_CALUDE_banana_tree_problem_l3918_391890


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3918_391840

theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (sum : ℚ) 
  (h₁ : a₁ = 3) 
  (h₂ : aₙ = 50) 
  (h₃ : sum = 318) : 
  ∃ (n : ℕ) (d : ℚ), n > 1 ∧ d = 47/11 ∧ 
    aₙ = a₁ + (n - 1) * d ∧ 
    sum = (n / 2) * (a₁ + aₙ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3918_391840


namespace NUMINAMATH_CALUDE_function_value_at_negative_l3918_391822

/-- Given a function f(x) = ax³ + bx - c/x + 2, if f(2023) = 6, then f(-2023) = -2 -/
theorem function_value_at_negative (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^3 + b * x - c / x + 2
  f 2023 = 6 → f (-2023) = -2 := by sorry

end NUMINAMATH_CALUDE_function_value_at_negative_l3918_391822


namespace NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l3918_391877

/-- A regular (4k+2)-gon inscribed in a circle -/
structure RegularPolygon (k : ℕ) where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- The vertices of the polygon -/
  vertices : Fin (4*k+2) → ℝ × ℝ
  /-- Condition that the polygon is regular and inscribed -/
  regular_inscribed : ∀ i : Fin (4*k+2), dist O (vertices i) = R

/-- The sum of segments cut by a central angle on diagonals -/
def sum_of_segments (p : RegularPolygon k) : ℝ := sorry

/-- Theorem: The sum of segments equals the radius -/
theorem sum_of_segments_equals_radius (k : ℕ) (p : RegularPolygon k) :
  sum_of_segments p = p.R := by sorry

end NUMINAMATH_CALUDE_sum_of_segments_equals_radius_l3918_391877


namespace NUMINAMATH_CALUDE_no_2008_special_progressions_l3918_391860

theorem no_2008_special_progressions : ¬ ∃ (progressions : Fin 2008 → Set ℕ),
  -- Each set in progressions is an infinite arithmetic progression
  (∀ i, ∃ (a d : ℕ), d > 0 ∧ progressions i = {n : ℕ | ∃ k, n = a + k * d}) ∧
  -- There are finitely many positive integers not in any progression
  (∃ S : Finset ℕ, ∀ n, n ∉ S → ∃ i, n ∈ progressions i) ∧
  -- No two progressions intersect
  (∀ i j, i ≠ j → progressions i ∩ progressions j = ∅) ∧
  -- Each progression contains a prime number bigger than 2008
  (∀ i, ∃ p ∈ progressions i, p > 2008 ∧ Nat.Prime p) :=
by
  sorry

end NUMINAMATH_CALUDE_no_2008_special_progressions_l3918_391860


namespace NUMINAMATH_CALUDE_function_zero_implies_a_range_l3918_391827

theorem function_zero_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1) 1 ∧ 2 * a * x₀ - a + 3 = 0) →
  a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_function_zero_implies_a_range_l3918_391827


namespace NUMINAMATH_CALUDE_sector_perimeter_l3918_391841

theorem sector_perimeter (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 8) :
  2 * r + 2 * area / r = 12 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3918_391841


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l3918_391847

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ N : ℕ, triangular_sum N = 3780 ∧ sum_of_digits N = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l3918_391847


namespace NUMINAMATH_CALUDE_major_premise_is_false_l3918_391817

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in_plane : Line → Plane → Prop)

-- State the theorem
theorem major_premise_is_false :
  ¬(∀ (l : Line) (p : Plane),
    parallel_line_plane l p →
    ∀ (m : Line), contained_in_plane m p → parallel_lines l m) :=
by sorry

end NUMINAMATH_CALUDE_major_premise_is_false_l3918_391817


namespace NUMINAMATH_CALUDE_rice_profit_l3918_391888

/-- Calculates the profit from selling a sack of rice -/
theorem rice_profit (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) :
  weight = 50 ∧ cost = 50 ∧ price_per_kg = 1.20 →
  weight * price_per_kg - cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_profit_l3918_391888
