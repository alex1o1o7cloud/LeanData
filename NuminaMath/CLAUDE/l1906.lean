import Mathlib

namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1906_190650

/-- Given a cylinder with volume 72π cm³ and a cone with the same height
    as the cylinder and half its radius, prove that the volume of the cone is 6π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) →
  (1/3 * π * (r/2)^2 * h = 6 * π) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1906_190650


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1540_l1906_190647

theorem sum_largest_smallest_prime_factors_1540 : 
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ 1540 ∧ 
    q ∣ 1540 ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 1540 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1540_l1906_190647


namespace NUMINAMATH_CALUDE_max_new_lines_theorem_l1906_190679

/-- The maximum number of new lines formed by connecting intersection points 
    of n lines in a plane, where any two lines intersect and no three lines 
    pass through the same point. -/
def max_new_lines (n : ℕ) : ℚ :=
  (1 / 8 : ℚ) * n * (n - 1) * (n - 2) * (n - 3)

/-- Theorem stating the maximum number of new lines formed by connecting 
    intersection points of n lines in a plane, where any two lines intersect 
    and no three lines pass through the same point. -/
theorem max_new_lines_theorem (n : ℕ) (h : n ≥ 3) :
  let original_lines := n
  let any_two_intersect := true
  let no_three_at_same_point := true
  max_new_lines n = (1 / 8 : ℚ) * n * (n - 1) * (n - 2) * (n - 3) :=
by sorry

end NUMINAMATH_CALUDE_max_new_lines_theorem_l1906_190679


namespace NUMINAMATH_CALUDE_minimum_score_for_advanced_algebra_l1906_190661

/-- Represents the minimum score needed in the 4th quarter to achieve a given average -/
def minimum_fourth_quarter_score (
  required_average : ℝ
  ) (first_quarter : ℝ) (second_quarter : ℝ) (third_quarter : ℝ) : ℝ :=
  4 * required_average - (first_quarter + second_quarter + third_quarter)

/-- Proves that the minimum score needed in the 4th quarter is 106% -/
theorem minimum_score_for_advanced_algebra (
  required_average : ℝ
  ) (first_quarter : ℝ) (second_quarter : ℝ) (third_quarter : ℝ)
  (h1 : required_average = 85)
  (h2 : first_quarter = 82)
  (h3 : second_quarter = 77)
  (h4 : third_quarter = 75) :
  minimum_fourth_quarter_score required_average first_quarter second_quarter third_quarter = 106 :=
by
  sorry

#eval minimum_fourth_quarter_score 85 82 77 75

end NUMINAMATH_CALUDE_minimum_score_for_advanced_algebra_l1906_190661


namespace NUMINAMATH_CALUDE_acute_angle_measure_l1906_190623

theorem acute_angle_measure (x : ℝ) : 
  0 < x → x < 90 → (90 - x = (180 - x) / 2 + 20) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_measure_l1906_190623


namespace NUMINAMATH_CALUDE_base_10_number_l1906_190689

-- Define the properties of the number
def is_valid_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a < 8 ∧ b < 8 ∧ c < 8 ∧ d < 8 ∧
  (8 * a + b + c / 8 + d / 64 : ℚ) = (12 * b + b + b / 12 + a / 144 : ℚ)

-- State the theorem
theorem base_10_number (a b c d : ℕ) :
  is_valid_number a b c d → a * 100 + b * 10 + c = 321 :=
by sorry

end NUMINAMATH_CALUDE_base_10_number_l1906_190689


namespace NUMINAMATH_CALUDE_wednesday_sales_l1906_190616

def initial_stock : ℕ := 600
def monday_sales : ℕ := 25
def tuesday_sales : ℕ := 70
def thursday_sales : ℕ := 110
def friday_sales : ℕ := 145
def unsold_percentage : ℚ := 1/4

theorem wednesday_sales :
  ∃ (wednesday_sales : ℕ),
    wednesday_sales = initial_stock * (1 - unsold_percentage) -
      (monday_sales + tuesday_sales + thursday_sales + friday_sales) ∧
    wednesday_sales = 100 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sales_l1906_190616


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l1906_190608

theorem no_function_satisfies_condition : 
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + f y) = f x + Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l1906_190608


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l1906_190666

def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

theorem parallel_vectors_imply_x_equals_two (x : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b x) = k • (4 • (b x) - 2 • a)) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l1906_190666


namespace NUMINAMATH_CALUDE_rectangle_x_value_l1906_190665

/-- A rectangle in a rectangular coordinate system with given properties -/
structure Rectangle where
  x : ℝ
  area : ℝ
  h1 : area = 90

/-- The x-coordinate of the first and last vertices of the rectangle is -9 -/
theorem rectangle_x_value (rect : Rectangle) : rect.x = -9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l1906_190665


namespace NUMINAMATH_CALUDE_three_color_right_triangle_l1906_190636

/-- A color type representing red, blue, and green --/
inductive Color
  | Red
  | Blue
  | Green

/-- A function that assigns a color to each point with integer coordinates --/
def coloring : ℤ × ℤ → Color := sorry

/-- Predicate to check if three points form a right-angled triangle --/
def is_right_triangle (p1 p2 p3 : ℤ × ℤ) : Prop := sorry

theorem three_color_right_triangle 
  (h1 : ∀ p : ℤ × ℤ, coloring p = Color.Red ∨ coloring p = Color.Blue ∨ coloring p = Color.Green)
  (h2 : ∃ p : ℤ × ℤ, coloring p = Color.Red)
  (h3 : ∃ p : ℤ × ℤ, coloring p = Color.Blue)
  (h4 : ∃ p : ℤ × ℤ, coloring p = Color.Green)
  (h5 : coloring (0, 0) = Color.Red)
  (h6 : coloring (0, 1) = Color.Blue) :
  ∃ p1 p2 p3 : ℤ × ℤ, 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 ∧ 
    is_right_triangle p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_three_color_right_triangle_l1906_190636


namespace NUMINAMATH_CALUDE_simplify_fourth_root_and_sum_l1906_190618

theorem simplify_fourth_root_and_sum : ∃ (c d : ℕ+),
  (((3 : ℝ)^5 * 5^3)^(1/4) = c * (d : ℝ)^(1/4)) ∧ 
  (c + d = 378) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_and_sum_l1906_190618


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1906_190635

theorem no_positive_integer_solutions : 
  ¬ ∃ (x y : ℕ+), x^2 + y^2 = x^3 + 2*y := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1906_190635


namespace NUMINAMATH_CALUDE_inequality_proof_l1906_190671

theorem inequality_proof (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1906_190671


namespace NUMINAMATH_CALUDE_sine_cosine_simplification_l1906_190663

theorem sine_cosine_simplification (x y : ℝ) :
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_simplification_l1906_190663


namespace NUMINAMATH_CALUDE_mathematics_teacher_is_C_l1906_190625

-- Define the types for teachers and subjects
inductive Teacher : Type
  | A | B | C | D

inductive Subject : Type
  | Mathematics | Physics | Chemistry | English

-- Define a function to represent the ability to teach a subject
def canTeach : Teacher → Subject → Prop
  | Teacher.A, Subject.Physics => True
  | Teacher.A, Subject.Chemistry => True
  | Teacher.B, Subject.Mathematics => True
  | Teacher.B, Subject.English => True
  | Teacher.C, Subject.Mathematics => True
  | Teacher.C, Subject.Physics => True
  | Teacher.C, Subject.Chemistry => True
  | Teacher.D, Subject.Chemistry => True
  | _, _ => False

-- Define the assignment of teachers to subjects
def assignment : Subject → Teacher
  | Subject.Mathematics => Teacher.C
  | Subject.Physics => Teacher.A
  | Subject.Chemistry => Teacher.D
  | Subject.English => Teacher.B

-- Theorem statement
theorem mathematics_teacher_is_C :
  (∀ s : Subject, canTeach (assignment s) s) ∧
  (∀ t : Teacher, ∃! s : Subject, assignment s = t) ∧
  (∀ s : Subject, ∃! t : Teacher, assignment s = t) →
  assignment Subject.Mathematics = Teacher.C :=
by sorry

end NUMINAMATH_CALUDE_mathematics_teacher_is_C_l1906_190625


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l1906_190601

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)

-- Define the length of a side
def side_length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of a parallelogram
def perimeter (p : Parallelogram) : ℝ :=
  side_length p.A p.B + side_length p.B p.C +
  side_length p.C p.D + side_length p.D p.A

-- Theorem statement
theorem parallelogram_perimeter (ABCD : Parallelogram)
  (h1 : side_length ABCD.A ABCD.B = 14)
  (h2 : side_length ABCD.B ABCD.C = 16) :
  perimeter ABCD = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l1906_190601


namespace NUMINAMATH_CALUDE_existence_and_digit_sum_l1906_190646

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proves the existence of N and its properties -/
theorem existence_and_digit_sum :
  ∃ N : ℕ, N^2 = 36^50 * 50^36 ∧ sum_of_digits N = 54 := by sorry

end NUMINAMATH_CALUDE_existence_and_digit_sum_l1906_190646


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1906_190609

/-- Given a speed of 25 km/hr and a distance of 125 km, the time taken is 5 hours. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (time : ℝ) : 
  speed = 25 → distance = 125 → time = distance / speed → time = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1906_190609


namespace NUMINAMATH_CALUDE_max_a_value_l1906_190680

-- Define the function f(x) = |x-2| + |x-8|
def f (x : ℝ) : ℝ := |x - 2| + |x - 8|

-- State the theorem
theorem max_a_value : 
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), f x ≥ b) → b ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1906_190680


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1906_190600

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ + a₃ + a₁₀ + a₁₁ = 36, a₃ + a₁₀ = 18 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 3 + a 10 + a 11 = 36 →
  a 3 + a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1906_190600


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l1906_190651

theorem add_preserves_inequality (a b c : ℝ) : a < b → a + c < b + c := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l1906_190651


namespace NUMINAMATH_CALUDE_cube_side_ratio_l1906_190613

theorem cube_side_ratio (weight1 weight2 : ℝ) (h1 : weight1 = 6) (h2 : weight2 = 48) :
  ∃ (s1 s2 : ℝ), s1 > 0 ∧ s2 > 0 ∧ s1^3 * weight2 = s2^3 * weight1 ∧ s2 / s1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l1906_190613


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1906_190615

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 5)^2 + (y - 6)^2 + (z - 7)^2 + 2 = 2 →
  x + y + z = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1906_190615


namespace NUMINAMATH_CALUDE_asphalt_cost_per_truckload_l1906_190656

/-- Calculates the cost per truckload of asphalt before tax -/
theorem asphalt_cost_per_truckload
  (road_length : ℝ)
  (road_width : ℝ)
  (coverage_per_truckload : ℝ)
  (tax_rate : ℝ)
  (total_cost_with_tax : ℝ)
  (h1 : road_length = 2000)
  (h2 : road_width = 20)
  (h3 : coverage_per_truckload = 800)
  (h4 : tax_rate = 0.2)
  (h5 : total_cost_with_tax = 4500) :
  (road_length * road_width) / coverage_per_truckload *
  (total_cost_with_tax / (1 + tax_rate)) /
  ((road_length * road_width) / coverage_per_truckload) = 75 := by
sorry

end NUMINAMATH_CALUDE_asphalt_cost_per_truckload_l1906_190656


namespace NUMINAMATH_CALUDE_exactly_five_numbers_l1906_190630

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem exactly_five_numbers :
  ∃! (s : Finset ℕ), s.card = 5 ∧
    ∀ n ∈ s, is_two_digit n ∧
      is_perfect_square (n - reverse_digits n) ∧
      n - reverse_digits n > 16 :=
sorry

end NUMINAMATH_CALUDE_exactly_five_numbers_l1906_190630


namespace NUMINAMATH_CALUDE_plan_y_cost_effective_l1906_190692

/-- The cost in cents for Plan X given the number of minutes used -/
def planXCost (minutes : ℕ) : ℕ := 15 * minutes

/-- The cost in cents for Plan Y given the number of minutes used -/
def planYCost (minutes : ℕ) : ℕ := 3000 + 10 * minutes

/-- The minimum number of minutes for Plan Y to be cost-effective -/
def minMinutes : ℕ := 601

theorem plan_y_cost_effective : 
  ∀ m : ℕ, m ≥ minMinutes → planYCost m < planXCost m :=
by
  sorry

#check plan_y_cost_effective

end NUMINAMATH_CALUDE_plan_y_cost_effective_l1906_190692


namespace NUMINAMATH_CALUDE_distance_between_points_l1906_190649

def point1 : ℝ × ℝ := (3, -5)
def point2 : ℝ × ℝ := (-4, 4)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1906_190649


namespace NUMINAMATH_CALUDE_share_a_plus_c_equals_6952_l1906_190655

def total_money : ℕ := 15800
def ratio_a : ℕ := 5
def ratio_b : ℕ := 9
def ratio_c : ℕ := 6
def ratio_d : ℕ := 5

def total_ratio : ℕ := ratio_a + ratio_b + ratio_c + ratio_d

theorem share_a_plus_c_equals_6952 :
  (ratio_a + ratio_c) * (total_money / total_ratio) = 6952 := by
  sorry

end NUMINAMATH_CALUDE_share_a_plus_c_equals_6952_l1906_190655


namespace NUMINAMATH_CALUDE_fraction_less_than_mode_l1906_190629

def data_list : List ℕ := [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than (l : List ℕ) (n : ℕ) : ℕ :=
  l.filter (· < n) |>.length

theorem fraction_less_than_mode :
  (count_less_than data_list (mode data_list) : ℚ) / data_list.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_mode_l1906_190629


namespace NUMINAMATH_CALUDE_triangular_sum_congruence_l1906_190612

theorem triangular_sum_congruence (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangular_sum_congruence_l1906_190612


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l1906_190685

theorem triangle_existence_condition (a b c : ℝ) :
  (∀ n : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^n + b^n > c^n ∧ b^n + c^n > a^n ∧ c^n + a^n > b^n) ↔
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ ((a ≥ b ∧ b = c) ∨ (b ≥ a ∧ a = c) ∨ (c ≥ a ∧ a = b))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l1906_190685


namespace NUMINAMATH_CALUDE_bee_count_l1906_190643

theorem bee_count (legs_per_bee : ℕ) (total_legs : ℕ) (h1 : legs_per_bee = 6) (h2 : total_legs = 12) :
  total_legs / legs_per_bee = 2 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1906_190643


namespace NUMINAMATH_CALUDE_tangent_line_intersects_ellipse_perpendicularly_l1906_190640

/-- An ellipse with semi-major axis 2 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- A circle with radius 2√5/5 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = (2 * Real.sqrt 5 / 5)^2}

/-- A line tangent to the circle at point (m, n) -/
def TangentLine (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1 + n * p.2 = (2 * Real.sqrt 5 / 5)^2}

/-- The origin of the coordinate system -/
def Origin : ℝ × ℝ := (0, 0)

/-- Two points are perpendicular with respect to the origin -/
def Perpendicular (p q : ℝ × ℝ) : Prop :=
  p.1 * q.1 + p.2 * q.2 = 0

theorem tangent_line_intersects_ellipse_perpendicularly 
  (m n : ℝ) (h : (m, n) ∈ Circle) :
  ∃ (A B : ℝ × ℝ), A ∈ Ellipse ∧ B ∈ Ellipse ∧ 
    A ∈ TangentLine m n ∧ B ∈ TangentLine m n ∧
    Perpendicular (A.1 - Origin.1, A.2 - Origin.2) (B.1 - Origin.1, B.2 - Origin.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersects_ellipse_perpendicularly_l1906_190640


namespace NUMINAMATH_CALUDE_courtyard_width_l1906_190642

/-- Proves that the width of a rectangular courtyard is 18 meters -/
theorem courtyard_width (length : ℝ) (brick_length : ℝ) (brick_width : ℝ) (total_bricks : ℕ) :
  length = 25 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 22500 →
  (length * (total_bricks : ℝ) * brick_length * brick_width) / length = 18 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l1906_190642


namespace NUMINAMATH_CALUDE_endpoint_sum_coordinates_l1906_190641

/-- Given a line segment with one endpoint (6, -2) and midpoint (5, 5),
    the sum of coordinates of the other endpoint is 16. -/
theorem endpoint_sum_coordinates (x y : ℝ) : 
  (6 + x) / 2 = 5 ∧ (-2 + y) / 2 = 5 → x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_coordinates_l1906_190641


namespace NUMINAMATH_CALUDE_linear_function_max_value_l1906_190660

theorem linear_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → a * x - a + 2 ≤ 7) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 4 ∧ a * x - a + 2 = 7) →
  a = 5/3 ∨ a = -5/2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_max_value_l1906_190660


namespace NUMINAMATH_CALUDE_decreasing_at_half_implies_a_le_two_l1906_190604

/-- A quadratic function f(x) = -2x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -4 * x + a

theorem decreasing_at_half_implies_a_le_two (a : ℝ) :
  (f_deriv a (1/2) ≤ 0) → a ≤ 2 := by
  sorry

#check decreasing_at_half_implies_a_le_two

end NUMINAMATH_CALUDE_decreasing_at_half_implies_a_le_two_l1906_190604


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1906_190668

def polynomial (x : ℝ) : ℝ := 6 * (x^5 + 2*x^3 + x + 3)

theorem sum_of_squared_coefficients : 
  (6^2 : ℝ) + (12^2 : ℝ) + (6^2 : ℝ) + (18^2 : ℝ) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1906_190668


namespace NUMINAMATH_CALUDE_prop_variations_l1906_190672

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 2 → x^2 - 5*x + 6 = 0

-- Define the converse
def converse (x : ℝ) : Prop := x^2 - 5*x + 6 = 0 → x = 2

-- Define the inverse
def inverse (x : ℝ) : Prop := x ≠ 2 → x^2 - 5*x + 6 ≠ 0

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop := x^2 - 5*x + 6 ≠ 0 → x ≠ 2

-- Theorem stating the truth values of converse, inverse, and contrapositive
theorem prop_variations :
  (∃ x : ℝ, ¬(converse x)) ∧
  (∃ x : ℝ, ¬(inverse x)) ∧
  (∀ x : ℝ, contrapositive x) :=
sorry

end NUMINAMATH_CALUDE_prop_variations_l1906_190672


namespace NUMINAMATH_CALUDE_min_triangles_in_configuration_l1906_190677

/-- A configuration of lines in a plane. -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersect : Bool

/-- The number of triangular regions formed by a line configuration. -/
def num_triangles (config : LineConfiguration) : ℕ := sorry

/-- Theorem: Given 3000 lines drawn on a plane where no two lines are parallel
    and no three lines intersect at a single point, the number of triangular
    regions formed is at least 2000. -/
theorem min_triangles_in_configuration :
  ∀ (config : LineConfiguration),
    config.num_lines = 3000 →
    config.no_parallel = true →
    config.no_triple_intersect = true →
    num_triangles config ≥ 2000 := by sorry

end NUMINAMATH_CALUDE_min_triangles_in_configuration_l1906_190677


namespace NUMINAMATH_CALUDE_fruit_basket_total_l1906_190639

/-- Calculates the total number of fruits in a basket given the number of oranges and relationships between fruit quantities. -/
def totalFruits (oranges : ℕ) : ℕ :=
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches

/-- Theorem stating that the total number of fruits in the basket is 28. -/
theorem fruit_basket_total : totalFruits 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_l1906_190639


namespace NUMINAMATH_CALUDE_jasmine_solution_percentage_l1906_190602

theorem jasmine_solution_percentage
  (initial_volume : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 80)
  (h2 : added_jasmine = 8)
  (h3 : added_water = 12)
  (h4 : final_percentage = 16)
  : (initial_volume * (final_percentage / 100) - added_jasmine) / initial_volume * 100 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jasmine_solution_percentage_l1906_190602


namespace NUMINAMATH_CALUDE_min_sum_squares_l1906_190653

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 120) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 6100 / 9 ∧ 
  ∃ (y₁' y₂' y₃' : ℝ), y₁'^2 + y₂'^2 + y₃'^2 = 6100 / 9 ∧ 
    y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 2 * y₁' + 3 * y₂' + 4 * y₃' = 120 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1906_190653


namespace NUMINAMATH_CALUDE_triangle_side_length_l1906_190667

-- Define the triangle DEF
structure Triangle (D E F : ℝ) where
  -- Angle sum property of a triangle
  angle_sum : D + E + F = Real.pi

-- Define the main theorem
theorem triangle_side_length 
  (D E F : ℝ) 
  (t : Triangle D E F) 
  (h1 : Real.cos (3 * D - E) + Real.sin (D + E) = 2) 
  (h2 : 6 = 6) :  -- DE = 6, but we use 6 = 6 as Lean doesn't know DE yet
  ∃ (EF : ℝ), EF = 3 * Real.sqrt (2 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1906_190667


namespace NUMINAMATH_CALUDE_ralph_tv_time_l1906_190652

/-- Represents Ralph's TV watching schedule for a week -/
structure TVSchedule where
  weekdayHours : ℝ
  weekdayShows : ℕ × ℕ  -- (number of 1-hour shows, number of 30-minute shows)
  videoGameDays : ℕ
  weekendHours : ℝ
  weekendShows : ℕ × ℕ  -- (number of 1-hour shows, number of 45-minute shows)
  weekendBreak : ℝ

/-- Calculates the total TV watching time for a week given a TV schedule -/
def totalTVTime (schedule : TVSchedule) : ℝ :=
  let weekdayTotal := schedule.weekdayHours * 5
  let weekendTotal := (schedule.weekendHours - schedule.weekendBreak) * 2
  weekdayTotal + weekendTotal

/-- Ralph's actual TV schedule -/
def ralphSchedule : TVSchedule :=
  { weekdayHours := 3
  , weekdayShows := (1, 4)
  , videoGameDays := 3
  , weekendHours := 6
  , weekendShows := (3, 4)
  , weekendBreak := 0.5 }

/-- Theorem stating that Ralph's total TV watching time in one week is 26 hours -/
theorem ralph_tv_time : totalTVTime ralphSchedule = 26 := by
  sorry


end NUMINAMATH_CALUDE_ralph_tv_time_l1906_190652


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1906_190620

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Given nonzero vectors a and b such that ||a|| = ||b|| = 2||a + b||,
    the cosine of the angle between them is -7/8 -/
theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖a‖ = ‖b‖ ∧ ‖a‖ = 2 * ‖a + b‖) : 
  inner a b / (‖a‖ * ‖b‖) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1906_190620


namespace NUMINAMATH_CALUDE_red_faces_up_possible_l1906_190644

/-- Represents a cubic block with one red face and five white faces -/
structure Block where
  redFaceUp : Bool

/-- Represents an n x n chessboard with cubic blocks -/
structure Chessboard (n : ℕ) where
  blocks : Matrix (Fin n) (Fin n) Block

/-- Represents a rotation of blocks in a row or column -/
inductive Rotation
  | Row : Fin n → Rotation
  | Column : Fin n → Rotation

/-- Applies a rotation to the chessboard -/
def applyRotation (board : Chessboard n) (rot : Rotation) : Chessboard n :=
  sorry

/-- Checks if all blocks on the chessboard have their red faces up -/
def allRedFacesUp (board : Chessboard n) : Bool :=
  sorry

/-- Theorem stating that it's possible to turn all red faces up after a finite number of rotations -/
theorem red_faces_up_possible (n : ℕ) :
  ∃ (rotations : List Rotation), ∀ (initial : Chessboard n),
    allRedFacesUp (rotations.foldl applyRotation initial) := by
  sorry

end NUMINAMATH_CALUDE_red_faces_up_possible_l1906_190644


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l1906_190617

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ := sorry

-- Define a function to convert ternary to decimal
def ternary_to_decimal (t : List ℕ) : ℕ := sorry

-- Theorem statement
theorem product_of_binary_and_ternary :
  let binary_num := [true, true, false, true]  -- Represents 1101₂
  let ternary_num := [2, 0, 2]  -- Represents 202₃
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 260 := by sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l1906_190617


namespace NUMINAMATH_CALUDE_bulbs_chosen_l1906_190670

theorem bulbs_chosen (total_bulbs : ℕ) (defective_bulbs : ℕ) (prob_at_least_one_defective : ℝ) :
  total_bulbs = 24 →
  defective_bulbs = 4 →
  prob_at_least_one_defective = 0.3115942028985508 →
  ∃ n : ℕ, n = 2 ∧ (1 - (total_bulbs - defective_bulbs : ℝ) / total_bulbs) ^ n = prob_at_least_one_defective :=
by sorry

end NUMINAMATH_CALUDE_bulbs_chosen_l1906_190670


namespace NUMINAMATH_CALUDE_point_C_coordinates_l1906_190648

def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (7, 2)

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →
  (dist A C = 2 * dist C B) →
  C = (5, 2/3) :=
by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l1906_190648


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1906_190687

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X : Polynomial ℚ)^4 + 3 * X^2 - 4 = (X^2 - 3) * q + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1906_190687


namespace NUMINAMATH_CALUDE_conference_games_l1906_190699

/-- Calculates the number of games within a division -/
def games_within_division (n : ℕ) : ℕ := n * (n - 1)

/-- Calculates the number of games between two divisions -/
def games_between_divisions (n m : ℕ) : ℕ := 2 * n * m

/-- The total number of games in the conference -/
def total_games : ℕ :=
  let div_a := 6
  let div_b := 7
  let div_c := 5
  let within_a := games_within_division div_a
  let within_b := games_within_division div_b
  let within_c := games_within_division div_c
  let between_ab := games_between_divisions div_a div_b
  let between_ac := games_between_divisions div_a div_c
  let between_bc := games_between_divisions div_b div_c
  within_a + within_b + within_c + between_ab + between_ac + between_bc

theorem conference_games : total_games = 306 := by sorry

end NUMINAMATH_CALUDE_conference_games_l1906_190699


namespace NUMINAMATH_CALUDE_cubic_is_odd_rhombus_diagonals_bisect_l1906_190638

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of diagonals bisecting each other
def diagonals_bisect (shape : Type) : Prop := 
  ∀ d1 d2 : shape → ℝ × ℝ, d1 ≠ d2 → ∃ p : ℝ × ℝ, 
    (∃ a b : shape, d1 a = p ∧ d2 b = p) ∧
    (∀ q : shape, d1 q = p ∨ d2 q = p)

-- Define shapes
class Parallelogram (shape : Type)
class Rhombus (shape : Type) extends Parallelogram shape

-- Theorem statements
theorem cubic_is_odd : is_odd_function f := sorry

theorem rhombus_diagonals_bisect (shape : Type) [Rhombus shape] : 
  diagonals_bisect shape := sorry

end NUMINAMATH_CALUDE_cubic_is_odd_rhombus_diagonals_bisect_l1906_190638


namespace NUMINAMATH_CALUDE_wall_building_time_l1906_190673

theorem wall_building_time 
  (men_days_constant : ℕ → ℕ → ℕ) 
  (h1 : men_days_constant 10 6 = men_days_constant 15 4) 
  (h2 : ∀ m d, men_days_constant m d = m * d) :
  (10 : ℚ) * 6 / 15 = 4 :=
sorry

end NUMINAMATH_CALUDE_wall_building_time_l1906_190673


namespace NUMINAMATH_CALUDE_smallest_w_l1906_190634

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (11^2) (936 * w) → 
  ∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (11^2) (936 * v) → 
    w ≤ v →
  w = 4356 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l1906_190634


namespace NUMINAMATH_CALUDE_number_of_recommendation_schemes_l1906_190688

/-- Represents the number of male participants -/
def num_males : ℕ := 2

/-- Represents the number of female participants -/
def num_females : ℕ := 3

/-- Represents the number of participants for the instruments project -/
def instruments_participants : ℕ := 1

/-- Represents the number of participants for the dance project -/
def dance_participants : ℕ := 2

/-- Represents the number of participants for the singing project -/
def singing_participants : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem stating the number of different recommendation schemes -/
theorem number_of_recommendation_schemes :
  choose num_females dance_participants *
  (num_males + (num_females - dance_participants)) *
  num_males = 18 := by sorry

end NUMINAMATH_CALUDE_number_of_recommendation_schemes_l1906_190688


namespace NUMINAMATH_CALUDE_sum_of_squares_constant_l1906_190654

/-- A triangle with side lengths a, b, c and median length m from vertex A to the midpoint of side BC. -/
structure Triangle :=
  (a b c m : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (positive_m : 0 < m)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

/-- The sum of squares of two sides in a triangle given the length of the third side and the median to its midpoint. -/
def sumOfSquares (t : Triangle) : ℝ := t.b^2 + t.c^2

/-- The theorem stating that for a triangle with side length 10 and median length 7,
    the difference between the maximum and minimum possible values of the sum of squares
    of the other two sides is 0. -/
theorem sum_of_squares_constant (t : Triangle) 
  (side_length : t.a = 10)
  (median_length : t.m = 7) :
  ∀ (t' : Triangle), 
    t'.a = t.a → 
    t'.m = t.m → 
    sumOfSquares t = sumOfSquares t' :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_constant_l1906_190654


namespace NUMINAMATH_CALUDE_midpoint_property_l1906_190619

/-- Given two points A and B in a 2D plane, proves that if C is the midpoint of AB,
    then 3 times the x-coordinate of C minus 2 times the y-coordinate of C equals 14. -/
theorem midpoint_property (A B C : ℝ × ℝ) : 
  A = (12, 9) → B = (4, 1) → C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  3 * C.1 - 2 * C.2 = 14 := by
  sorry

#check midpoint_property

end NUMINAMATH_CALUDE_midpoint_property_l1906_190619


namespace NUMINAMATH_CALUDE_apple_basket_theorem_l1906_190657

/-- Represents the number of apples in each basket -/
def baskets : List ℕ := [20, 30, 40, 60, 90]

/-- The total number of apples initially -/
def total : ℕ := baskets.sum

/-- Checks if a number is divisible by 3 -/
def divisibleBy3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Checks if removing a basket results in a valid 2:1 ratio -/
def validRemoval (n : ℕ) : Prop :=
  n ∈ baskets ∧ divisibleBy3 (total - n) ∧
  ∃ x y : ℕ, x + y = total - n ∧ x = 2 * y ∧
  (x ∈ baskets.filter (· ≠ n) ∨ y ∈ baskets.filter (· ≠ n))

/-- The main theorem -/
theorem apple_basket_theorem :
  ∀ n : ℕ, validRemoval n → n = 60 ∨ n = 90 := by sorry

end NUMINAMATH_CALUDE_apple_basket_theorem_l1906_190657


namespace NUMINAMATH_CALUDE_energy_drink_consumption_l1906_190693

/-- Represents the relationship between coding hours and energy drink consumption -/
def energy_drink_relation (hours : ℝ) (drinks : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ hours * drinks = k

theorem energy_drink_consumption 
  (h1 : energy_drink_relation 8 3)
  (h2 : energy_drink_relation 10 x) :
  x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_energy_drink_consumption_l1906_190693


namespace NUMINAMATH_CALUDE_number_division_l1906_190622

theorem number_division (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l1906_190622


namespace NUMINAMATH_CALUDE_no_zero_roots_l1906_190628

theorem no_zero_roots : 
  (∀ x : ℝ, 5 * x^2 - 3 = 47 → x ≠ 0) ∧ 
  (∀ x : ℝ, (3 * x + 2)^2 = (x + 2)^2 → x ≠ 0) ∧ 
  (∀ x : ℝ, (2 * x^2 - 6 : ℝ) = (2 * x - 2 : ℝ) → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_zero_roots_l1906_190628


namespace NUMINAMATH_CALUDE_negative_integer_sum_square_l1906_190626

theorem negative_integer_sum_square (N : ℤ) : 
  N < 0 → N^2 + N = -12 → N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_square_l1906_190626


namespace NUMINAMATH_CALUDE_smallest_yellow_candy_quantity_l1906_190637

def red_candy_cost : ℕ := 8
def green_candy_cost : ℕ := 12
def blue_candy_cost : ℕ := 15
def yellow_candy_cost : ℕ := 24

def red_candy_quantity : ℕ := 10
def green_candy_quantity : ℕ := 18
def blue_candy_quantity : ℕ := 20

def red_total_cost : ℕ := red_candy_cost * red_candy_quantity
def green_total_cost : ℕ := green_candy_cost * green_candy_quantity
def blue_total_cost : ℕ := blue_candy_cost * blue_candy_quantity

theorem smallest_yellow_candy_quantity :
  ∃ (n : ℕ), n > 0 ∧
  (yellow_candy_cost * n) % red_total_cost = 0 ∧
  (yellow_candy_cost * n) % green_total_cost = 0 ∧
  (yellow_candy_cost * n) % blue_total_cost = 0 ∧
  ∀ (m : ℕ), m > 0 →
    (yellow_candy_cost * m) % red_total_cost = 0 →
    (yellow_candy_cost * m) % green_total_cost = 0 →
    (yellow_candy_cost * m) % blue_total_cost = 0 →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_candy_quantity_l1906_190637


namespace NUMINAMATH_CALUDE_new_basis_from_original_l1906_190664

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem new_basis_from_original
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, a - c, b] ∧
  Submodule.span ℝ {a + b, a - c, b} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_new_basis_from_original_l1906_190664


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l1906_190621

def shelf_difference (initial_action_figures : ℕ) (initial_books : ℕ) (added_books : ℕ) : ℕ :=
  initial_action_figures - (initial_books + added_books)

theorem jerry_shelf_difference :
  shelf_difference 7 2 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l1906_190621


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l1906_190611

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- Line passing through the focus of a parabola with slope angle 60° -/
structure FocusLine (c : Parabola) where
  slope : ℝ
  h_slope : slope = Real.sqrt 3
  focus_x : ℝ
  h_focus_x : focus_x = c.p / 2

/-- Theorem stating the ratio of AB to AP is 7/12 -/
theorem parabola_intersection_ratio (c : Parabola) (l : FocusLine c) 
  (A B : ParabolaPoint c) (P : ℝ × ℝ) :
  A.x > 0 → A.y > 0 →  -- A in first quadrant
  B.x > 0 → B.y < 0 →  -- B in fourth quadrant
  P.1 = 0 →  -- P on y-axis
  (A.y - l.focus_x) = l.slope * (A.x - l.focus_x) →  -- A on line l
  (B.y - l.focus_x) = l.slope * (B.x - l.focus_x) →  -- B on line l
  (P.2 - l.focus_x) = l.slope * (P.1 - l.focus_x) →  -- P on line l
  abs (A.x - B.x) / abs (A.x - P.1) = 7 / 12 := by
    sorry

end NUMINAMATH_CALUDE_parabola_intersection_ratio_l1906_190611


namespace NUMINAMATH_CALUDE_least_valid_number_l1906_190681

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧  -- Digit representation
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- All digits are different
    (a = 5 ∨ b = 5 ∨ c = 5 ∨ d = 5) ∧  -- One of the digits is 5
    n % a = 0 ∧ n % b = 0 ∧ n % c = 0 ∧ n % d = 0  -- Divisible by each of its digits

theorem least_valid_number : 
  is_valid_number 1524 ∧ ∀ m : ℕ, is_valid_number m → m ≥ 1524 :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l1906_190681


namespace NUMINAMATH_CALUDE_sequence_sum_comparison_l1906_190696

theorem sequence_sum_comparison (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ k, k > 0 → S k = -a k - (1/2)^(k-1) + 2) : 
  (n ≥ 5 → S n > 2 - 1/(n-1)) ∧ 
  ((n = 3 ∨ n = 4) → S n < 2 - 1/(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_comparison_l1906_190696


namespace NUMINAMATH_CALUDE_expression_simplification_l1906_190606

theorem expression_simplification :
  (((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4)) = 43 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1906_190606


namespace NUMINAMATH_CALUDE_soccer_team_starters_l1906_190695

theorem soccer_team_starters (n : ℕ) (k : ℕ) (q : ℕ) (m : ℕ) 
  (h1 : n = 16) 
  (h2 : k = 7) 
  (h3 : q = 4) 
  (h4 : m = 1) :
  (q.choose m) * ((n - q).choose (k - m)) = 3696 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l1906_190695


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1906_190698

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) →
  (10 * x₂^2 + 15 * x₂ - 20 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1906_190698


namespace NUMINAMATH_CALUDE_probability_y_l1906_190683

theorem probability_y (x y : Set Ω) (z : Set Ω → ℝ) 
  (hx : z x = 0.02)
  (hxy : z (x ∩ y) = 0.10)
  (hcond : z x / z y = 0.2) :
  z y = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_y_l1906_190683


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1906_190614

theorem consecutive_integers_sum_of_cubes (n : ℕ) : 
  n > 0 ∧ (n - 1)^2 + n^2 + (n + 1)^2 = 8555 → 
  (n - 1)^3 + n^3 + (n + 1)^3 = 446949 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1906_190614


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1906_190694

/-- Proves that 11,580,000 is equal to 1.158 × 10^7 in scientific notation -/
theorem scientific_notation_equivalence : 
  11580000 = 1.158 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1906_190694


namespace NUMINAMATH_CALUDE_jerry_average_increase_l1906_190674

theorem jerry_average_increase :
  let initial_average : ℝ := 85
  let fourth_test_score : ℝ := 97
  let total_tests : ℕ := 4
  let sum_first_three : ℝ := initial_average * 3
  let sum_all_four : ℝ := sum_first_three + fourth_test_score
  let new_average : ℝ := sum_all_four / total_tests
  new_average - initial_average = 3 := by sorry

end NUMINAMATH_CALUDE_jerry_average_increase_l1906_190674


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1906_190607

/-- Simple interest calculation -/
theorem simple_interest_principal (interest rate time principal : ℝ) :
  interest = principal * rate * time →
  rate = 0.09 →
  time = 1 →
  interest = 900 →
  principal = 10000 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1906_190607


namespace NUMINAMATH_CALUDE_linear_function_range_l1906_190632

/-- A linear function defined on a closed interval -/
def LinearFunction (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- The domain of the function -/
def Domain : Set ℝ := { x : ℝ | 1/4 ≤ x ∧ x ≤ 3/4 }

theorem linear_function_range (a b : ℝ) (h : a > 0) :
  Set.range (fun x => LinearFunction a b x) = Set.Icc (a/4 + b) (3*a/4 + b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_range_l1906_190632


namespace NUMINAMATH_CALUDE_louise_yellow_pencils_l1906_190675

/-- Proves the number of yellow pencils Louise has --/
theorem louise_yellow_pencils :
  let box_capacity : ℕ := 20
  let red_pencils : ℕ := 20
  let blue_pencils : ℕ := 2 * red_pencils
  let green_pencils : ℕ := red_pencils + blue_pencils
  let total_boxes : ℕ := 8
  let total_capacity : ℕ := total_boxes * box_capacity
  let other_pencils : ℕ := red_pencils + blue_pencils + green_pencils
  let yellow_pencils : ℕ := total_capacity - other_pencils
  yellow_pencils = 40 := by
  sorry

end NUMINAMATH_CALUDE_louise_yellow_pencils_l1906_190675


namespace NUMINAMATH_CALUDE_circle_intersection_condition_l1906_190684

/-- Circle B with equation x^2 + y^2 + b = 0 -/
def circle_B (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + b = 0}

/-- Circle C with equation x^2 + y^2 - 6x + 8y + 16 = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 16 = 0}

/-- Two circles do not intersect -/
def no_intersection (A B : Set (ℝ × ℝ)) : Prop :=
  A ∩ B = ∅

/-- The main theorem -/
theorem circle_intersection_condition (b : ℝ) :
  no_intersection (circle_B b) circle_C →
  (-4 < b ∧ b < 0) ∨ b < -25 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_l1906_190684


namespace NUMINAMATH_CALUDE_percy_dish_cost_l1906_190658

/-- The cost of a meal for three people with a 10% tip --/
def meal_cost (leticia_cost scarlett_cost percy_cost : ℝ) : ℝ :=
  (leticia_cost + scarlett_cost + percy_cost) * 1.1

/-- The theorem stating the cost of Percy's dish --/
theorem percy_dish_cost : 
  ∃ (percy_cost : ℝ), 
    meal_cost 10 13 percy_cost = 44 ∧ 
    percy_cost = 17 := by
  sorry

end NUMINAMATH_CALUDE_percy_dish_cost_l1906_190658


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l1906_190624

-- Define the function f(x) = x^2 + mx + 1
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the property of f being decreasing on an interval
def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem decreasing_quadratic_condition (m : ℝ) :
  isDecreasingOn (f m) 0 5 → m ≤ -10 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l1906_190624


namespace NUMINAMATH_CALUDE_sum_of_squared_complements_geq_two_l1906_190678

theorem sum_of_squared_complements_geq_two 
  (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a + b + c = 1) : 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_complements_geq_two_l1906_190678


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1906_190605

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, if a₁ + a₉ + a₂ + a₈ = 20, then a₃ + a₇ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (sum_condition : a 1 + a 9 + a 2 + a 8 = 20) : 
  a 3 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1906_190605


namespace NUMINAMATH_CALUDE_x_value_proof_l1906_190690

theorem x_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1906_190690


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l1906_190645

theorem polynomial_coefficient_equality (m n : ℤ) : 
  (∀ x : ℝ, (x - 1) * (x + m) = x^2 - n*x - 6) → 
  (m = 6 ∧ n = -5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l1906_190645


namespace NUMINAMATH_CALUDE_snow_clearing_volume_l1906_190631

/-- The volume of snow on a rectangular pathway -/
def snow_volume (length width depth : ℚ) : ℚ :=
  length * width * depth

/-- Proof that the volume of snow on the given pathway is 67.5 cubic feet -/
theorem snow_clearing_volume :
  let length : ℚ := 30
  let width : ℚ := 3
  let depth : ℚ := 3/4
  snow_volume length width depth = 67.5 := by
sorry

end NUMINAMATH_CALUDE_snow_clearing_volume_l1906_190631


namespace NUMINAMATH_CALUDE_mod_sum_powers_seven_l1906_190691

theorem mod_sum_powers_seven : (45^1234 + 27^1234) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_seven_l1906_190691


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1906_190662

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

/-- The last term of an arithmetic sequence -/
def last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  ∃ n : ℕ, 
    last_term 71 2 n = 99 ∧ 
    3 * (arithmetic_sum 71 2 n) = 3825 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1906_190662


namespace NUMINAMATH_CALUDE_abs_negative_thirteen_l1906_190603

theorem abs_negative_thirteen : |(-13 : ℤ)| = 13 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_thirteen_l1906_190603


namespace NUMINAMATH_CALUDE_no_perfect_cube_pair_l1906_190682

theorem no_perfect_cube_pair : ¬ ∃ (a b : ℤ), 
  (∃ (x : ℤ), a^5 * b + 3 = x^3) ∧ (∃ (y : ℤ), a * b^5 + 3 = y^3) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_pair_l1906_190682


namespace NUMINAMATH_CALUDE_complex_division_simplification_l1906_190633

theorem complex_division_simplification (z : ℂ) (h : z = 1 - 2 * I) :
  5 * I / z = -2 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l1906_190633


namespace NUMINAMATH_CALUDE_smallest_c_for_no_five_l1906_190676

theorem smallest_c_for_no_five : ∃ c : ℤ, (∀ x : ℝ, x^2 + c*x + 10 ≠ 5) ∧ 
  (∀ c' : ℤ, c' < c → ∃ x : ℝ, x^2 + c'*x + 10 = 5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_no_five_l1906_190676


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1906_190627

-- Part 1: System of Equations
theorem solve_system_equations :
  ∃! (x y : ℝ), 3 * x + 2 * y = 12 ∧ 2 * x - y = 1 ∧ x = 2 ∧ y = 3 := by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities :
  ∀ x : ℝ, (x - 1 < 2 * x ∧ 2 * (x - 3) ≤ 3 - x) ↔ (-1 < x ∧ x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1906_190627


namespace NUMINAMATH_CALUDE_three_witnesses_are_liars_l1906_190686

-- Define the type for witnesses
inductive Witness : Type
  | one
  | two
  | three
  | four

-- Define a function to represent the statement of each witness
def statement (w : Witness) : Nat :=
  match w with
  | Witness.one => 1
  | Witness.two => 2
  | Witness.three => 3
  | Witness.four => 4

-- Define a predicate to check if a witness is telling the truth
def isTruthful (w : Witness) (numLiars : Nat) : Prop :=
  statement w = numLiars

-- Theorem: Exactly three witnesses are liars
theorem three_witnesses_are_liars :
  ∃! (numLiars : Nat), 
    numLiars = 3 ∧
    (∃! (truthful : Witness), 
      isTruthful truthful numLiars ∧
      ∀ (w : Witness), w ≠ truthful → ¬(isTruthful w numLiars)) :=
by
  sorry


end NUMINAMATH_CALUDE_three_witnesses_are_liars_l1906_190686


namespace NUMINAMATH_CALUDE_base_8_representation_of_157_digit_count_of_157_base_8_l1906_190669

def to_base_8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem base_8_representation_of_157 :
  to_base_8 157 = [2, 3, 5] :=
sorry

theorem digit_count_of_157_base_8 :
  (to_base_8 157).length = 3 :=
sorry

end NUMINAMATH_CALUDE_base_8_representation_of_157_digit_count_of_157_base_8_l1906_190669


namespace NUMINAMATH_CALUDE_carls_garden_area_l1906_190697

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : Nat
  post_spacing : Nat
  longer_side_posts : Nat
  shorter_side_posts : Nat

/-- Calculates the area of the garden given the specifications -/
def calculate_area (g : Garden) : Nat :=
  (g.shorter_side_posts - 1) * g.post_spacing * 
  (g.longer_side_posts - 1) * g.post_spacing

/-- Theorem stating the area of Carl's garden -/
theorem carls_garden_area : 
  ∀ g : Garden, 
    g.total_posts = 24 ∧ 
    g.post_spacing = 5 ∧ 
    g.longer_side_posts = 2 * g.shorter_side_posts ∧
    g.longer_side_posts + g.shorter_side_posts = g.total_posts + 2 →
    calculate_area g = 900 := by
  sorry

end NUMINAMATH_CALUDE_carls_garden_area_l1906_190697


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1906_190659

def g (a b x : ℚ) : ℚ := a * x^3 - 8 * x^2 + b * x - 7

theorem polynomial_remainder (a b : ℚ) :
  (g a b 2 = 1) ∧ (g a b (-3) = -89) → a = -10/3 ∧ b = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1906_190659


namespace NUMINAMATH_CALUDE_sock_drawing_probability_l1906_190610

theorem sock_drawing_probability : 
  ∀ (total_socks : ℕ) (colors : ℕ) (socks_per_color : ℕ) (drawn_socks : ℕ),
    total_socks = colors * socks_per_color →
    total_socks = 10 →
    colors = 5 →
    socks_per_color = 2 →
    drawn_socks = 5 →
    (Nat.choose total_socks drawn_socks : ℚ) ≠ 0 →
    (Nat.choose colors 4 * Nat.choose 4 1 * (socks_per_color ^ 3) : ℚ) / 
    (Nat.choose total_socks drawn_socks : ℚ) = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sock_drawing_probability_l1906_190610
