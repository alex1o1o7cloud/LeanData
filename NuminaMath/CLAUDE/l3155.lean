import Mathlib

namespace NUMINAMATH_CALUDE_scientific_notation_of_1_35_billion_l3155_315512

theorem scientific_notation_of_1_35_billion :
  ∃ (a : ℝ) (n : ℤ), 1.35e9 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 :=
by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_35_billion_l3155_315512


namespace NUMINAMATH_CALUDE_solve_for_b_l3155_315534

theorem solve_for_b (x b : ℝ) : 
  (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 →
  x = 0.3 →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_b_l3155_315534


namespace NUMINAMATH_CALUDE_distance_traveled_l3155_315558

/-- Given a person traveling at 65 km/hr for 3 hours, prove that the distance traveled is 195 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 65)
  (h2 : time = 3)
  (h3 : distance = speed * time) :
  distance = 195 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3155_315558


namespace NUMINAMATH_CALUDE_log_sum_and_product_imply_average_l3155_315586

theorem log_sum_and_product_imply_average (x y : ℝ) : 
  x > 0 → y > 0 → (Real.log x / Real.log y + Real.log y / Real.log x = 4) → x * y = 81 → 
  (x + y) / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_log_sum_and_product_imply_average_l3155_315586


namespace NUMINAMATH_CALUDE_sandwich_ratio_l3155_315594

/-- The number of sandwiches Samson ate at lunch on Monday -/
def lunch_sandwiches : ℕ := 3

/-- The number of sandwiches Samson ate at dinner on Monday -/
def dinner_sandwiches : ℕ := 6

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_sandwiches : ℕ := 1

/-- The difference in total sandwiches eaten between Monday and Tuesday -/
def sandwich_difference : ℕ := 8

theorem sandwich_ratio :
  (dinner_sandwiches : ℚ) / lunch_sandwiches = 2 ∧
  lunch_sandwiches + dinner_sandwiches = tuesday_sandwiches + sandwich_difference :=
sorry

end NUMINAMATH_CALUDE_sandwich_ratio_l3155_315594


namespace NUMINAMATH_CALUDE_power_sum_problem_l3155_315579

theorem power_sum_problem (a b : ℝ) 
  (h1 : a^5 + b^5 = 3) 
  (h2 : a^15 + b^15 = 9) : 
  a^10 + b^10 = 5 := by
sorry

end NUMINAMATH_CALUDE_power_sum_problem_l3155_315579


namespace NUMINAMATH_CALUDE_equation_solution_range_l3155_315526

theorem equation_solution_range (x m : ℝ) : 
  x / (x - 1) - 2 = (3 * m) / (2 * x - 2) → 
  x > 0 → 
  x ≠ 1 → 
  m < 4/3 ∧ m ≠ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3155_315526


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3155_315505

theorem polynomial_divisibility (a : ℤ) : 
  (∃ q : Polynomial ℤ, X^6 - 33•X + 20 = (X^2 - X + a•1) * q) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3155_315505


namespace NUMINAMATH_CALUDE_reflection_matrix_correct_l3155_315589

/-- Reflection matrix over the line y = x -/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1],
    ![1, 0]]

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection over the line y = x -/
def reflect (p : Point2D) : Point2D :=
  ⟨p.y, p.x⟩

theorem reflection_matrix_correct :
  ∀ (p : Point2D),
  let reflected := reflect p
  let matrix_result := reflection_matrix.mulVec ![p.x, p.y]
  matrix_result = ![reflected.x, reflected.y] := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_correct_l3155_315589


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3155_315541

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3155_315541


namespace NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l3155_315575

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_per_face : Nat
  strip_width : Nat
  strip_length : Nat

/-- Calculate the number of unpainted cubes in a painted cube -/
def unpainted_cubes (c : PaintedCube) : Nat :=
  sorry

/-- Theorem stating the number of unpainted cubes in the specific problem -/
theorem unpainted_cubes_6x6x6 :
  let c : PaintedCube := {
    size := 6,
    total_cubes := 216,
    painted_per_face := 10,
    strip_width := 2,
    strip_length := 5
  }
  unpainted_cubes c = 186 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l3155_315575


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l3155_315569

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The inner core of the prism not touching tin foil -/
def inner : PrismDimensions :=
  { length := 2^(5/3),
    width := 2^(8/3),
    height := 2^(5/3) }

/-- The outer prism covered in tin foil -/
def outer : PrismDimensions :=
  { length := inner.length + 2,
    width := inner.width + 2,
    height := inner.height + 2 }

theorem foil_covered_prism_width :
  (inner.length * inner.width * inner.height = 128) →
  (inner.width = 2 * inner.length) →
  (inner.width = 2 * inner.height) →
  (outer.width = 10) := by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l3155_315569


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l3155_315556

/-- Theorem: For a point P(x, y) on the parabola y² = 4x, if its distance from the focus is 4, then x = 3 and y = ±2√3 -/
theorem parabola_point_coordinates (x y : ℝ) :
  y^2 = 4*x →                           -- P is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 16 →                -- Distance from P to focus (1, 0) is 4
  (x = 3 ∧ y = 2*Real.sqrt 3 ∨ y = -2*Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_parabola_point_coordinates_l3155_315556


namespace NUMINAMATH_CALUDE_matrix_power_zero_l3155_315597

open Matrix Complex

theorem matrix_power_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h1 : A * B = B * A)
  (h2 : B.det ≠ 0)
  (h3 : ∀ z : ℂ, Complex.abs z = 1 → Complex.abs ((A + z • B).det) = 1) :
  A ^ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_zero_l3155_315597


namespace NUMINAMATH_CALUDE_elmo_has_24_books_l3155_315574

def elmos_books (elmo_multiplier laura_multiplier stu_books : ℕ) : ℕ :=
  elmo_multiplier * (laura_multiplier * stu_books)

theorem elmo_has_24_books :
  elmos_books 3 2 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_elmo_has_24_books_l3155_315574


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_l3155_315557

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_first_two : a 1 + a 2 = 1
  sum_third_fourth : a 3 + a 4 = 9

/-- The sum of the fourth and fifth terms is either 27 or -27 -/
theorem fourth_fifth_sum (seq : GeometricSequence) : 
  seq.a 4 + seq.a 5 = 27 ∨ seq.a 4 + seq.a 5 = -27 := by
  sorry


end NUMINAMATH_CALUDE_fourth_fifth_sum_l3155_315557


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3155_315550

theorem trigonometric_expression_evaluation :
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) /
  (Real.sin (12 * π / 180) * (4 * Real.cos (12 * π / 180) ^ 2 - 2)) = -4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3155_315550


namespace NUMINAMATH_CALUDE_kelly_chickens_count_l3155_315513

/-- The number of chickens Kelly has -/
def number_of_chickens : ℕ := 8

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 3

/-- The price of a dozen eggs in dollars -/
def price_per_dozen : ℕ := 5

/-- The total amount Kelly makes in 4 weeks in dollars -/
def total_earnings : ℕ := 280

/-- The number of days in 4 weeks -/
def days_in_four_weeks : ℕ := 28

theorem kelly_chickens_count :
  number_of_chickens * eggs_per_chicken_per_day * days_in_four_weeks / 12 * price_per_dozen = total_earnings :=
sorry

end NUMINAMATH_CALUDE_kelly_chickens_count_l3155_315513


namespace NUMINAMATH_CALUDE_not_value_preserving_g_value_preserving_f_condition_l3155_315540

/-- Definition of a value-preserving function on an interval [m, n] -/
def is_value_preserving (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧ 
  Monotone (fun x => f x) ∧
  Set.range (fun x => f x) = Set.Icc m n

/-- The function g(x) = x^2 - 2x -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The function f(x) = 2 + 1/a - 1/(a^2x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + 1/a - 1/(a^2*x)

theorem not_value_preserving_g : ¬ is_value_preserving g 0 1 := by sorry

theorem value_preserving_f_condition (a : ℝ) :
  (∃ m n, is_value_preserving (f a) m n) ↔ (a > 1/2 ∨ a < -3/2) := by sorry

end NUMINAMATH_CALUDE_not_value_preserving_g_value_preserving_f_condition_l3155_315540


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3155_315539

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = 3 * 2^n + k,
    prove that if {a_n} is a geometric sequence, then k = -3. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n, S n = 3 * 2^n + k) →
  (∀ n, a n = S n - S (n-1)) →
  (∀ n, n ≥ 2 → a n * a (n-2) = (a (n-1))^2) →
  k = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3155_315539


namespace NUMINAMATH_CALUDE_dereks_initial_money_l3155_315515

/-- Proves that Derek's initial amount of money was $40 -/
theorem dereks_initial_money :
  ∀ (derek_initial : ℕ) (derek_spent dave_initial dave_spent : ℕ),
  derek_spent = 30 →
  dave_initial = 50 →
  dave_spent = 7 →
  dave_initial - dave_spent = (derek_initial - derek_spent) + 33 →
  derek_initial = 40 := by
  sorry

end NUMINAMATH_CALUDE_dereks_initial_money_l3155_315515


namespace NUMINAMATH_CALUDE_positive_real_floor_product_48_l3155_315517

theorem positive_real_floor_product_48 (x : ℝ) :
  x > 0 ∧ x * ⌊x⌋ = 48 → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_positive_real_floor_product_48_l3155_315517


namespace NUMINAMATH_CALUDE_problem_solution_l3155_315582

theorem problem_solution : 
  (2008^2 - 2007 * 2009 = 1) ∧ 
  ((-0.125)^2011 * 8^2010 = -0.125) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3155_315582


namespace NUMINAMATH_CALUDE_percentage_difference_l3155_315553

theorem percentage_difference : (0.7 * 40) - (4 / 5 * 25) = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3155_315553


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l3155_315522

theorem y_in_terms_of_x (x y : ℝ) : 2 * x + y = 5 → y = 5 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l3155_315522


namespace NUMINAMATH_CALUDE_least_distinct_values_l3155_315516

theorem least_distinct_values (list : List ℕ+) : 
  list.length = 2023 →
  ∃! m : ℕ+, (list.count m = 11 ∧ ∀ n : ℕ+, n ≠ m → list.count n < 11) →
  (∀ k : ℕ+, k < 203 → ∃ x : ℕ+, list.count x > list.count k) →
  ∃ x : ℕ+, list.count x = 203 :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_l3155_315516


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersections_nonagon_intersections_eq_choose_four_l3155_315591

/-- The number of intersection points of diagonals in a regular nonagon -/
theorem nonagon_diagonal_intersections : ℕ := by
  -- Define a regular nonagon
  sorry

/-- The number of ways to choose 4 vertices from 9 vertices -/
def choose_four_from_nine : ℕ := Nat.choose 9 4

/-- Theorem: The number of distinct interior points where two or more diagonals
    intersect in a regular nonagon is equal to choose_four_from_nine -/
theorem nonagon_intersections_eq_choose_four :
  nonagon_diagonal_intersections = choose_four_from_nine := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersections_nonagon_intersections_eq_choose_four_l3155_315591


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l3155_315598

theorem quadratic_root_k_value (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) ∧ (2 * 4^2 + 3 * 4 - k = 0) → k = 44 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l3155_315598


namespace NUMINAMATH_CALUDE_vector_problem_l3155_315531

/-- Given three vectors a, b, c in ℝ² -/
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c : ℝ → ℝ × ℝ := λ m ↦ (-2, m)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Main theorem combining both parts of the problem -/
theorem vector_problem :
  (∃ m : ℝ, perpendicular a (b + c m) → m = -1) ∧
  (∃ k : ℝ, collinear (k • a + b) (2 • a - b) → k = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3155_315531


namespace NUMINAMATH_CALUDE_linear_coefficient_of_equation_l3155_315584

theorem linear_coefficient_of_equation : ∃ (a b c : ℝ), 
  (∀ x, (2*x + 1)*(x - 3) = x^2 + 1) → 
  (∀ x, a*x^2 + b*x + c = 0) ∧ 
  b = -5 := by
  sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_equation_l3155_315584


namespace NUMINAMATH_CALUDE_sqrt_74_between_consecutive_integers_product_l3155_315533

theorem sqrt_74_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ 
  (n : ℝ) < Real.sqrt 74 ∧ 
  Real.sqrt 74 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 72 := by
sorry

end NUMINAMATH_CALUDE_sqrt_74_between_consecutive_integers_product_l3155_315533


namespace NUMINAMATH_CALUDE_percentage_x_more_than_y_l3155_315542

theorem percentage_x_more_than_y : 
  ∀ (x y z : ℝ),
  y = 1.2 * z →
  z = 250 →
  x + y + z = 925 →
  (x - y) / y * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_x_more_than_y_l3155_315542


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3155_315509

theorem triangle_side_calculation (a b : ℝ) (A B : Real) (hpos : 0 < a) :
  a = Real.sqrt 2 →
  B = 60 * π / 180 →
  A = 45 * π / 180 →
  b = a * Real.sin B / Real.sin A →
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3155_315509


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3155_315535

/-- Given a polynomial q(x) = Dx^4 + Ex^2 + Fx + 9, 
    if the remainder when divided by x - 2 is 17, 
    then the remainder when divided by x + 2 is 33. -/
theorem polynomial_remainder (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D*x^4 + E*x^2 + F*x + 9
  (q 2 = 17) → (q (-2) = 33) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3155_315535


namespace NUMINAMATH_CALUDE_g_of_3_l3155_315547

def g (x : ℝ) : ℝ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem g_of_3 : g 3 = 183 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l3155_315547


namespace NUMINAMATH_CALUDE_right_triangle_angle_split_l3155_315532

theorem right_triangle_angle_split (BC AC : ℝ) (h_right : BC = 5 ∧ AC = 12) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let angle_ratio := (1 : ℝ) / 3
  let smaller_segment := AB * (Real.sqrt 3 / 2)
  smaller_segment = 13 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_split_l3155_315532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3155_315577

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 6 = 20) : 
  (a 4 + a 7) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3155_315577


namespace NUMINAMATH_CALUDE_student_correct_answers_l3155_315599

/-- Represents a test score calculation system -/
structure TestScore where
  totalQuestions : ℕ
  score : ℤ
  correctAnswers : ℕ
  incorrectAnswers : ℕ

/-- Theorem: Given the conditions, prove that the student answered 91 questions correctly -/
theorem student_correct_answers
  (test : TestScore)
  (h1 : test.totalQuestions = 100)
  (h2 : test.score = test.correctAnswers - 2 * test.incorrectAnswers)
  (h3 : test.score = 73)
  (h4 : test.correctAnswers + test.incorrectAnswers = test.totalQuestions) :
  test.correctAnswers = 91 := by
  sorry


end NUMINAMATH_CALUDE_student_correct_answers_l3155_315599


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l3155_315566

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | a < x}

theorem set_operations_and_intersection :
  (A ∪ B = {x | 1 < x ∧ x ≤ 8}) ∧
  ((Aᶜ) ∩ B = {x | 1 < x ∧ x < 2}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty ↔ a < 8) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l3155_315566


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l3155_315561

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (348 * π / 180) → n = -12 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l3155_315561


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3155_315545

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan (α + π/4) = 3 * Real.cos (2*α)) : 
  Real.sin (2*α) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3155_315545


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3155_315568

/-- The equation √(x+2) + 2√(x-1) + 3√(3x-2) = 10 has a unique solution x = 2 -/
theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, (x + 2 ≥ 0) ∧ (x - 1 ≥ 0) ∧ (3*x - 2 ≥ 0) ∧
  (Real.sqrt (x + 2) + 2 * Real.sqrt (x - 1) + 3 * Real.sqrt (3*x - 2) = 10) ∧
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3155_315568


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3155_315530

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  unitsDigit (2 * (factorial 1 + factorial 2 + factorial 3 + factorial 4)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3155_315530


namespace NUMINAMATH_CALUDE_darryl_break_even_point_l3155_315563

/-- Calculates the break-even point for Darryl's machine sales -/
theorem darryl_break_even_point 
  (parts_cost : ℕ) 
  (patent_cost : ℕ) 
  (selling_price : ℕ) 
  (h1 : parts_cost = 3600)
  (h2 : patent_cost = 4500)
  (h3 : selling_price = 180) :
  (parts_cost + patent_cost) / selling_price = 45 :=
by sorry

end NUMINAMATH_CALUDE_darryl_break_even_point_l3155_315563


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3155_315508

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^12 + 8*x^11 + 18*x^10 + 2048*x^9 - 1638*x^8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3155_315508


namespace NUMINAMATH_CALUDE_thirty_sixth_bead_is_white_l3155_315595

/-- Represents the color of a bead -/
inductive BeadColor
| Black
| White

/-- Defines the sequence of bead colors -/
def beadSequence : ℕ → BeadColor
| 0 => BeadColor.White
| n + 1 => match (n + 1) % 5 with
  | 1 => BeadColor.White
  | 2 => BeadColor.Black
  | 3 => BeadColor.White
  | 4 => BeadColor.Black
  | _ => BeadColor.White

/-- Theorem: The 36th bead in the sequence is white -/
theorem thirty_sixth_bead_is_white : beadSequence 35 = BeadColor.White := by
  sorry

end NUMINAMATH_CALUDE_thirty_sixth_bead_is_white_l3155_315595


namespace NUMINAMATH_CALUDE_unique_prime_p_l3155_315524

theorem unique_prime_p : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (3 * p^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_p_l3155_315524


namespace NUMINAMATH_CALUDE_robbers_river_crossing_impossibility_l3155_315576

theorem robbers_river_crossing_impossibility :
  ∀ (n : ℕ) (trips : ℕ → ℕ → Prop),
    n = 40 →
    (∀ i j, i < n → j < n → i ≠ j → (trips i j ∨ trips j i)) →
    (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k →
      ¬(trips i j ∧ trips i k)) →
    False :=
by
  sorry

end NUMINAMATH_CALUDE_robbers_river_crossing_impossibility_l3155_315576


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3155_315529

/-- A trinomial x^2 + kx + 9 is a perfect square if and only if k = 6 or k = -6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ (a b : ℝ), ∀ x, x^2 + k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3155_315529


namespace NUMINAMATH_CALUDE_nine_books_arrangement_l3155_315590

/-- Represents a collection of books with specific adjacency requirements -/
structure BookArrangement where
  total_books : Nat
  adjacent_pairs : Nat
  single_books : Nat

/-- Calculates the number of ways to arrange books with adjacency requirements -/
def arrange_books (ba : BookArrangement) : Nat :=
  (2 ^ ba.adjacent_pairs) * Nat.factorial (ba.single_books + ba.adjacent_pairs)

/-- Theorem stating the number of ways to arrange 9 books with 2 adjacent pairs -/
theorem nine_books_arrangement :
  arrange_books ⟨9, 2, 5⟩ = 4 * Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_nine_books_arrangement_l3155_315590


namespace NUMINAMATH_CALUDE_circle_condition_l3155_315511

/-- The equation x^2 + y^2 + 4x - 2y + 5m = 0 represents a circle if and only if m < 1 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + 5*m = 0 ∧ 
   ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 + 4*x - 2*y + 5*m = 0) 
  ↔ m < 1 := by
sorry


end NUMINAMATH_CALUDE_circle_condition_l3155_315511


namespace NUMINAMATH_CALUDE_farm_cows_l3155_315552

/-- Given a farm with cows and horses, prove the number of cows -/
theorem farm_cows (total_horses : ℕ) (ratio_cows : ℕ) (ratio_horses : ℕ) 
  (h1 : total_horses = 6)
  (h2 : ratio_cows = 7)
  (h3 : ratio_horses = 2) :
  (ratio_cows : ℚ) / ratio_horses * total_horses = 21 := by
  sorry

end NUMINAMATH_CALUDE_farm_cows_l3155_315552


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3155_315551

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 3*x - 4 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 - 4*x - 1 = 0

-- Theorem for the solutions of the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 1 ∧ equation1 (-4)) :=
sorry

-- Theorem for the solutions of the second equation
theorem solutions_equation2 : 
  (∃ x : ℝ, equation2 x) ↔ (equation2 (1 + Real.sqrt 6 / 2) ∧ equation2 (1 - Real.sqrt 6 / 2)) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3155_315551


namespace NUMINAMATH_CALUDE_word_probabilities_l3155_315546

def word : String := "дифференцициал"

def is_vowel (c : Char) : Bool :=
  c ∈ ['а', 'е', 'и']

def is_consonant (c : Char) : Bool :=
  c ∈ ['д', 'ф', 'р', 'н', 'ц', 'л']

theorem word_probabilities :
  let total_letters := word.length
  let vowels := (word.toList.filter is_vowel).length
  let consonants := (word.toList.filter is_consonant).length
  (vowels : ℚ) / total_letters = 5 / 12 ∧
  (consonants : ℚ) / total_letters = 7 / 12 ∧
  ((word.toList.filter (· = 'ч')).length : ℚ) / total_letters = 0 := by
  sorry


end NUMINAMATH_CALUDE_word_probabilities_l3155_315546


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bounds_l3155_315562

theorem arithmetic_geometric_mean_difference_bounds (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bounds_l3155_315562


namespace NUMINAMATH_CALUDE_mixture_proportion_l3155_315571

/-- Represents a solution with a given percentage of chemical a -/
structure Solution :=
  (percent_a : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (sol_x : Solution)
  (sol_y : Solution)
  (percent_x : ℝ)
  (percent_mixture_a : ℝ)

/-- The theorem stating the proportion of solution x in the mixture -/
theorem mixture_proportion
  (mix : Mixture)
  (hx : mix.sol_x.percent_a = 0.3)
  (hy : mix.sol_y.percent_a = 0.4)
  (hm : mix.percent_mixture_a = 0.32)
  : mix.percent_x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_mixture_proportion_l3155_315571


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3155_315521

/-- Given a quadratic equation x^2 - bx + 20 = 0 where the product of roots is 20,
    prove that the sum of roots is b. -/
theorem sum_of_roots_quadratic (b : ℝ) : 
  (∃ x y : ℝ, x^2 - b*x + 20 = 0 ∧ y^2 - b*y + 20 = 0 ∧ x*y = 20) → 
  (∃ x y : ℝ, x^2 - b*x + 20 = 0 ∧ y^2 - b*y + 20 = 0 ∧ x + y = b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3155_315521


namespace NUMINAMATH_CALUDE_length_PR_l3155_315572

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def O : ℝ × ℝ := (0, 0)  -- Center of the circle
def radius : ℝ := 10

-- Define points P, Q, and R
variable (P Q R : ℝ × ℝ)

-- State the conditions
variable (h1 : P ∈ Circle O radius)
variable (h2 : Q ∈ Circle O radius)
variable (h3 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12^2)
variable (h4 : R ∈ Circle O radius)
variable (h5 : R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2)

-- State the theorem
theorem length_PR : (P.1 - R.1)^2 + (P.2 - R.2)^2 = 40 := by sorry

end NUMINAMATH_CALUDE_length_PR_l3155_315572


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3155_315549

theorem arithmetic_geometric_sequence : 
  ∃ (a b c d e : ℚ), 
    -- The five numbers
    a = 4 ∧ b = 8 ∧ c = 12 ∧ d = 16 ∧ e = 64/3 ∧
    -- First four form an arithmetic progression
    (b - a = c - b) ∧ (c - b = d - c) ∧
    -- Sum of first four is 40
    (a + b + c + d = 40) ∧
    -- Last three form a geometric progression
    (c^2 = b * d) ∧
    -- Product of outer terms of geometric progression is 32 times the second number
    (c * e = 32 * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3155_315549


namespace NUMINAMATH_CALUDE_coefficient_of_y_l3155_315536

theorem coefficient_of_y (x y a : ℝ) : 
  7 * x + y = 19 → 
  x + a * y = 1 → 
  2 * x + y = 5 → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l3155_315536


namespace NUMINAMATH_CALUDE_production_growth_l3155_315578

theorem production_growth (a : ℝ) (x : ℕ) (y : ℝ) (h : x > 0) :
  y = a * (1 + 0.05) ^ x ↔ 
  (∀ n : ℕ, n ≤ x → 
    (n = 0 → y = a) ∧ 
    (n > 0 → y = a * (1 + 0.05) ^ n)) :=
sorry

end NUMINAMATH_CALUDE_production_growth_l3155_315578


namespace NUMINAMATH_CALUDE_factor_between_l3155_315520

theorem factor_between (n a b : ℕ) (hn : n > 10) (ha : a > 0) (hb : b > 0) 
  (hab : a ≠ b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (heq : n = a^2 + b) : 
  ∃ k : ℕ, k ∣ n ∧ a < k ∧ k < b := by
  sorry

end NUMINAMATH_CALUDE_factor_between_l3155_315520


namespace NUMINAMATH_CALUDE_prob_two_queens_or_two_jacks_standard_deck_l3155_315573

/-- A standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)
  (jacks : ℕ)

/-- The probability of drawing either two queens or at least two jacks
    when selecting 3 cards randomly from a standard deck. -/
def prob_two_queens_or_two_jacks (d : Deck) : ℚ :=
  -- Definition to be proved
  74 / 850

/-- Theorem stating the probability of drawing either two queens or at least two jacks
    when selecting 3 cards randomly from a standard 52-card deck. -/
theorem prob_two_queens_or_two_jacks_standard_deck :
  prob_two_queens_or_two_jacks ⟨52, 4, 4⟩ = 74 / 850 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_two_jacks_standard_deck_l3155_315573


namespace NUMINAMATH_CALUDE_surface_area_rotated_sector_l3155_315506

/-- The surface area of a solid formed by rotating a circular sector about one of its radii. -/
theorem surface_area_rotated_sector (R θ : ℝ) (h_R : R > 0) (h_θ : 0 < θ ∧ θ < 2 * π) :
  let surface_area := 2 * π * R^2 * Real.sin (θ/2) * (Real.cos (θ/2) + 2 * Real.sin (θ/2))
  ∃ (S : ℝ), S = surface_area ∧ 
    S = (π * (R * Real.sin θ)^2) +  -- Area of the circular base
        (π * R * (R * Real.sin θ)) +  -- Curved surface area of the cone
        (2 * π * R * (R * (1 - Real.cos θ)))  -- Surface area of the spherical cap
  := by sorry

end NUMINAMATH_CALUDE_surface_area_rotated_sector_l3155_315506


namespace NUMINAMATH_CALUDE_joke_spread_after_one_minute_l3155_315564

def joke_spread (base : ℕ) (intervals : ℕ) : ℕ :=
  (base^(intervals + 1) - 1) / (base - 1)

theorem joke_spread_after_one_minute :
  joke_spread 6 6 = 55987 :=
by sorry

end NUMINAMATH_CALUDE_joke_spread_after_one_minute_l3155_315564


namespace NUMINAMATH_CALUDE_sqrt_two_triangle_one_l3155_315527

-- Define the triangle operation
def triangle (a b : ℝ) : ℝ := a^2 - a*b

-- Theorem statement
theorem sqrt_two_triangle_one :
  triangle (Real.sqrt 2) 1 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_triangle_one_l3155_315527


namespace NUMINAMATH_CALUDE_swords_per_orc_l3155_315538

theorem swords_per_orc (total_swords : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) :
  total_swords = 1200 →
  num_squads = 10 →
  orcs_per_squad = 8 →
  total_swords / (num_squads * orcs_per_squad) = 15 := by
  sorry

end NUMINAMATH_CALUDE_swords_per_orc_l3155_315538


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l3155_315588

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < 20 → ¬(n % p = 0)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, n > 1 ∧ ¬(is_prime n) ∧ has_no_prime_factor_less_than_20 n ∧
  (∀ m : ℕ, m > 1 → ¬(is_prime m) → has_no_prime_factor_less_than_20 m → m ≥ n) ∧
  n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l3155_315588


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l3155_315501

theorem sin_plus_two_cos_alpha (α : Real) :
  (∃ x y : Real, x = -3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l3155_315501


namespace NUMINAMATH_CALUDE_cindy_hit_eight_l3155_315585

-- Define the set of players
inductive Player : Type
| Alice : Player
| Ben : Player
| Cindy : Player
| Dave : Player
| Ellen : Player

-- Define the score function
def score : Player → ℕ
| Player.Alice => 10
| Player.Ben => 6
| Player.Cindy => 9
| Player.Dave => 15
| Player.Ellen => 19

-- Define the set of possible scores on the dartboard
def dartboard_scores : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define a function to check if a player's score can be composed of two different dartboard scores
def valid_score (p : Player) : Prop :=
  ∃ (a b : ℕ), a ∈ dartboard_scores ∧ b ∈ dartboard_scores ∧ a ≠ b ∧ a + b = score p

-- Theorem: Cindy is the only player who could have hit the section worth 8 points
theorem cindy_hit_eight :
  (∀ p : Player, valid_score p) →
  (∃! p : Player, ∃ (a : ℕ), a ∈ dartboard_scores ∧ a ≠ 8 ∧ a + 8 = score p) ∧
  (∃ (a : ℕ), a ∈ dartboard_scores ∧ a ≠ 8 ∧ a + 8 = score Player.Cindy) :=
by sorry

end NUMINAMATH_CALUDE_cindy_hit_eight_l3155_315585


namespace NUMINAMATH_CALUDE_tic_tac_toe_probability_l3155_315523

/-- Represents a tic-tac-toe board -/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- The number of cells in a tic-tac-toe board -/
def boardSize : Nat := 9

/-- The number of noughts on the board -/
def noughtsCount : Nat := 3

/-- The number of crosses on the board -/
def crossesCount : Nat := 6

/-- The number of ways to choose noughts positions -/
def totalPositions : Nat := Nat.choose boardSize noughtsCount

/-- The number of winning positions for noughts -/
def winningPositions : Nat := 8

/-- Theorem: The probability of 3 noughts being in a winning position is 2/21 -/
theorem tic_tac_toe_probability : 
  (winningPositions : ℚ) / totalPositions = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_probability_l3155_315523


namespace NUMINAMATH_CALUDE_probability_of_successful_meeting_l3155_315528

/-- Friend's train arrival time in minutes after 1:00 -/
def FriendArrivalTime : Type := {t : ℝ // 0 ≤ t ∧ t ≤ 60}

/-- Alex's arrival time in minutes after 1:00 -/
def AlexArrivalTime : Type := {t : ℝ // 0 ≤ t ∧ t ≤ 120}

/-- The waiting time of the friend's train in minutes -/
def WaitingTime : ℝ := 10

/-- The event that Alex arrives while the friend's train is still at the station -/
def SuccessfulMeeting (f : FriendArrivalTime) (a : AlexArrivalTime) : Prop :=
  f.val ≤ a.val ∧ a.val ≤ f.val + WaitingTime

/-- The probability measure for the problem -/
noncomputable def P : Set (FriendArrivalTime × AlexArrivalTime) → ℝ := sorry

/-- The theorem stating the probability of a successful meeting -/
theorem probability_of_successful_meeting :
  P {p : FriendArrivalTime × AlexArrivalTime | SuccessfulMeeting p.1 p.2} = 1/4 := by sorry

end NUMINAMATH_CALUDE_probability_of_successful_meeting_l3155_315528


namespace NUMINAMATH_CALUDE_ratio_xyz_l3155_315596

theorem ratio_xyz (x y z : ℚ) 
  (h1 : (3/4) * y = (1/2) * x) 
  (h2 : (3/10) * x = (1/5) * z) : 
  ∃ (k : ℚ), k > 0 ∧ x = 6*k ∧ y = 4*k ∧ z = 9*k := by
sorry

end NUMINAMATH_CALUDE_ratio_xyz_l3155_315596


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l3155_315583

/-- Proves that the equation of a line passing through point (2, -1) with an inclination angle of π/4 is x - y - 3 = 0 -/
theorem line_equation_through_point_with_inclination (x y : ℝ) :
  let point : ℝ × ℝ := (2, -1)
  let inclination : ℝ := π / 4
  let slope : ℝ := Real.tan inclination
  (y - point.2 = slope * (x - point.1)) → (x - y - 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l3155_315583


namespace NUMINAMATH_CALUDE_unread_messages_proof_l3155_315502

/-- The number of days it takes to read all messages -/
def days : ℕ := 7

/-- The number of messages read per day -/
def messages_read_per_day : ℕ := 20

/-- The number of new messages received per day -/
def new_messages_per_day : ℕ := 6

/-- The initial number of unread messages -/
def initial_messages : ℕ := days * (messages_read_per_day - new_messages_per_day)

theorem unread_messages_proof :
  initial_messages = 98 := by sorry

end NUMINAMATH_CALUDE_unread_messages_proof_l3155_315502


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3155_315570

theorem rhombus_longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side_length = 40 →
  shorter_diagonal = 30 →
  longer_diagonal = 10 * Real.sqrt 55 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3155_315570


namespace NUMINAMATH_CALUDE_non_shaded_area_of_square_with_semicircles_l3155_315507

/-- The area of the non-shaded part of a square with side length 4 and eight congruent semicircles --/
theorem non_shaded_area_of_square_with_semicircles :
  let square_side : ℝ := 4
  let num_semicircles : ℕ := 8
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := π * semicircle_radius ^ 2 / 2
  let total_shaded_area : ℝ := num_semicircles * semicircle_area
  let non_shaded_area : ℝ := square_area - total_shaded_area
  non_shaded_area = 8 := by sorry

end NUMINAMATH_CALUDE_non_shaded_area_of_square_with_semicircles_l3155_315507


namespace NUMINAMATH_CALUDE_x_can_be_any_real_value_l3155_315555

theorem x_can_be_any_real_value
  (x y z w : ℝ)
  (h1 : x / y > z / w)
  (h2 : y ≠ 0 ∧ w ≠ 0)
  (h3 : y * w > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b < 0 ∧ c = 0 ∧
    (x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_x_can_be_any_real_value_l3155_315555


namespace NUMINAMATH_CALUDE_number_division_l3155_315554

theorem number_division : ∃ x : ℝ, x / 0.04 = 400.90000000000003 ∧ x = 16.036 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l3155_315554


namespace NUMINAMATH_CALUDE_striped_shirt_ratio_l3155_315581

theorem striped_shirt_ratio (total : ℕ) (checkered shorts striped : ℕ) : 
  total = 81 →
  total = checkered + striped →
  shorts = checkered + 19 →
  striped = shorts + 8 →
  striped * 3 = total * 2 := by
sorry

end NUMINAMATH_CALUDE_striped_shirt_ratio_l3155_315581


namespace NUMINAMATH_CALUDE_cosine_amplitude_l3155_315519

theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cos (b * x) ≤ 3) ∧
  (∃ x, a * Real.cos (b * x) = 3) ∧
  (∀ x, a * Real.cos (b * x) = a * Real.cos (b * (x + 2 * Real.pi))) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l3155_315519


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3155_315587

theorem circle_intersection_range (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y - 1)^2 = r^2 ∧ r > 0 ∧
   ∃ (x' y' : ℝ), (x' - 2)^2 + (y' - 1)^2 = 1 ∧
   x' = y ∧ y' = x) →
  r ∈ Set.Icc (Real.sqrt 2 - 1) (Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3155_315587


namespace NUMINAMATH_CALUDE_nine_knights_in_room_l3155_315514

/-- Represents a person on the island, either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The total number of people in the room -/
def totalPeople : Nat := 15

/-- Represents the statements made by each person -/
structure Statements where
  sixLiars : Bool  -- "Among my acquaintances in this room, there are exactly six liars"
  noMoreThanSevenKnights : Bool  -- "Among my acquaintances in this room, there are no more than seven knights"

/-- Returns true if the statements are consistent with the person's type and the room's composition -/
def statementsAreConsistent (p : Person) (statements : Statements) (knightCount : Nat) : Bool :=
  match p with
  | Person.Knight => statements.sixLiars = (totalPeople - knightCount - 1 = 6) ∧ 
                     statements.noMoreThanSevenKnights = (knightCount - 1 ≤ 7)
  | Person.Liar => statements.sixLiars ≠ (totalPeople - knightCount - 1 = 6) ∧ 
                   statements.noMoreThanSevenKnights ≠ (knightCount - 1 ≤ 7)

/-- The main theorem: there are exactly 9 knights in the room -/
theorem nine_knights_in_room : 
  ∃ (knightCount : Nat), knightCount = 9 ∧ 
  (∀ (p : Person) (s : Statements), statementsAreConsistent p s knightCount) ∧
  knightCount + (totalPeople - knightCount) = totalPeople :=
sorry

end NUMINAMATH_CALUDE_nine_knights_in_room_l3155_315514


namespace NUMINAMATH_CALUDE_gcd_of_90_and_450_l3155_315504

theorem gcd_of_90_and_450 : Nat.gcd 90 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_450_l3155_315504


namespace NUMINAMATH_CALUDE_tromino_coverage_l3155_315503

/-- Represents a tromino (L-shaped piece formed from three squares) --/
structure Tromino

/-- Represents a chessboard --/
structure Chessboard (n : ℕ) where
  size : n ≥ 7
  odd : Odd n

/-- Counts the number of black squares on the chessboard --/
def black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

/-- Counts the minimum number of trominoes required to cover all black squares --/
def min_trominoes (n : ℕ) : ℕ := (n + 1)^2 / 4

/-- Theorem stating the minimum number of trominoes required to cover all black squares --/
theorem tromino_coverage (n : ℕ) (board : Chessboard n) :
  min_trominoes n = black_squares n := by sorry

end NUMINAMATH_CALUDE_tromino_coverage_l3155_315503


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3155_315592

/-- A geometric sequence with first term 1, second term a, and third term 16 -/
def is_geometric_sequence (a : ℝ) : Prop := ∃ r : ℝ, r ≠ 0 ∧ a = r ∧ 16 = r * r

/-- The condition is necessary but not sufficient -/
theorem geometric_sequence_condition (a : ℝ) :
  (is_geometric_sequence a → a = 4) ∧ ¬(a = 4 → is_geometric_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3155_315592


namespace NUMINAMATH_CALUDE_problem_l3155_315518

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sin x / Real.cos x + 3

theorem problem (a : ℝ) (h : f (Real.log a) = 4) : f (Real.log (1 / a)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_l3155_315518


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3155_315500

-- Define the main equation
def main_equation (x a : ℝ) : Prop :=
  Real.arctan (x / 2) + Real.arctan (2 - x) = a

-- Part 1
theorem part_one :
  ∀ x : ℝ, main_equation x (π / 4) →
    Real.arccos (x / 2) = 2 * π / 3 ∨ Real.arccos (x / 2) = 0 := by sorry

-- Part 2
theorem part_two :
  ∃ x a : ℝ, main_equation x a →
    a ∈ Set.Icc (Real.arctan (1 / (-2 * Real.sqrt 10 - 6))) (Real.arctan (1 / (2 * Real.sqrt 10 - 6))) := by sorry

-- Part 3
theorem part_three :
  ∀ α β a : ℝ, 
    α ∈ Set.Icc 5 15 → β ∈ Set.Icc 5 15 →
    α ≠ β →
    main_equation α a → main_equation β a →
    α + β ≤ 19 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3155_315500


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l3155_315580

theorem quadratic_equation_value (y : ℝ) (h : y = 4) : 3 * y^2 + 4 * y + 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l3155_315580


namespace NUMINAMATH_CALUDE_new_person_weight_l3155_315544

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℚ) (replaced_weight : ℚ) :
  initial_count = 10 →
  weight_increase = 5/2 →
  replaced_weight = 50 →
  ∃ (new_weight : ℚ),
    new_weight = replaced_weight + (initial_count * weight_increase) ∧
    new_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3155_315544


namespace NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_l3155_315548

/-- Calculates the profit percentage for a wheat mixture sale --/
theorem wheat_mixture_profit_percentage
  (wheat1_weight : ℝ)
  (wheat1_price : ℝ)
  (wheat2_weight : ℝ)
  (wheat2_price : ℝ)
  (selling_price : ℝ)
  (h1 : wheat1_weight = 30)
  (h2 : wheat1_price = 11.5)
  (h3 : wheat2_weight = 20)
  (h4 : wheat2_price = 14.25)
  (h5 : selling_price = 15.75) :
  let total_cost := wheat1_weight * wheat1_price + wheat2_weight * wheat2_price
  let total_weight := wheat1_weight + wheat2_weight
  let total_selling_price := total_weight * selling_price
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 25 := by
    sorry

end NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_l3155_315548


namespace NUMINAMATH_CALUDE_straight_line_distance_l3155_315565

/-- The straight-line distance between two points, where one point is 20 yards south
    and 50 yards east of the other, is 10√29 yards. -/
theorem straight_line_distance (south east : ℝ) (h1 : south = 20) (h2 : east = 50) :
  Real.sqrt (south^2 + east^2) = 10 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_straight_line_distance_l3155_315565


namespace NUMINAMATH_CALUDE_teachers_day_theorem_l3155_315567

/-- A directed graph with 200 vertices where each vertex has exactly one outgoing edge -/
structure TeacherGraph where
  vertices : Finset (Fin 200)
  edges : Fin 200 → Fin 200
  edge_property : ∀ v, v ∈ vertices → edges v ≠ v

/-- An independent set in the graph -/
def IndependentSet (G : TeacherGraph) (S : Finset (Fin 200)) : Prop :=
  ∀ u v, u ∈ S → v ∈ S → u ≠ v → G.edges u ≠ v

/-- The theorem stating that there exists an independent set of size at least 67 -/
theorem teachers_day_theorem (G : TeacherGraph) :
  ∃ S : Finset (Fin 200), IndependentSet G S ∧ S.card ≥ 67 := by
  sorry


end NUMINAMATH_CALUDE_teachers_day_theorem_l3155_315567


namespace NUMINAMATH_CALUDE_common_chord_length_l3155_315537

/-- Given two circles with radius 12 and centers 16 units apart,
    the length of their common chord is 8√5. -/
theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 16) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 8 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_common_chord_length_l3155_315537


namespace NUMINAMATH_CALUDE_min_value_expression_l3155_315560

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 32) :
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 96 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 32 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 2*z₀^2 = 96 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3155_315560


namespace NUMINAMATH_CALUDE_periodic_function_equality_l3155_315543

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β), if f(3) = 3, then f(2016) = -3 -/
theorem periodic_function_equality (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)
  f 3 = 3 → f 2016 = -3 := by sorry

end NUMINAMATH_CALUDE_periodic_function_equality_l3155_315543


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l3155_315510

theorem mod_fifteen_equivalence (n : ℤ) : 
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15827 [ZMOD 15] → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l3155_315510


namespace NUMINAMATH_CALUDE_leadership_selection_ways_l3155_315593

/-- The number of ways to choose a president, vice-president, and committee from a group. -/
def choose_leadership (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 3)

/-- The problem statement as a theorem. -/
theorem leadership_selection_ways :
  choose_leadership 10 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_leadership_selection_ways_l3155_315593


namespace NUMINAMATH_CALUDE_marys_remaining_money_equals_50_minus_12p_l3155_315525

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def marys_remaining_money (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary's remaining money is equal to 50 - 12p -/
theorem marys_remaining_money_equals_50_minus_12p (p : ℝ) :
  marys_remaining_money p = 50 - 12 * p := by
  sorry

end NUMINAMATH_CALUDE_marys_remaining_money_equals_50_minus_12p_l3155_315525


namespace NUMINAMATH_CALUDE_half_height_of_triangular_prism_l3155_315559

/-- Given a triangular prism with volume 576 cm³ and base area 3 cm², 
    half of its height is 96 cm. -/
theorem half_height_of_triangular_prism (volume : ℝ) (base_area : ℝ) (height : ℝ) :
  volume = 576 ∧ base_area = 3 ∧ volume = base_area * height →
  height / 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_half_height_of_triangular_prism_l3155_315559
