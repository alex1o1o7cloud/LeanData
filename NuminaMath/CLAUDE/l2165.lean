import Mathlib

namespace NUMINAMATH_CALUDE_find_w_l2165_216522

/-- Given that ( √ 1.21 ) / ( √ 0.81 ) + ( √ 1.44 ) / ( √ w ) = 2.9365079365079367, prove that w = 0.49 -/
theorem find_w (w : ℝ) (h : Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt w = 2.9365079365079367) : 
  w = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_find_w_l2165_216522


namespace NUMINAMATH_CALUDE_f_of_four_equals_thirteen_l2165_216564

/-- Given a function f where f(2x) = 3x^2 + 1 for all x, prove that f(4) = 13 -/
theorem f_of_four_equals_thirteen (f : ℝ → ℝ) (h : ∀ x, f (2 * x) = 3 * x^2 + 1) : 
  f 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_of_four_equals_thirteen_l2165_216564


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2165_216540

theorem rectangle_max_area (perimeter : ℝ) (width : ℝ) (length : ℝ) (area : ℝ) : 
  perimeter = 40 →
  length = 2 * width →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 800 / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2165_216540


namespace NUMINAMATH_CALUDE_clock_hands_angle_l2165_216506

theorem clock_hands_angle (n : ℕ) : 0 < n ∧ n < 720 → (∃ k : ℤ, |11 * n / 2 % 360 - 360 * k| = 1) ↔ n = 262 ∨ n = 458 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_angle_l2165_216506


namespace NUMINAMATH_CALUDE_batsman_average_l2165_216557

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  (previous_total = 16 * previous_average) →
  (previous_total + 85 = 17 * (previous_average + 3)) →
  (previous_average + 3 = 37) := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l2165_216557


namespace NUMINAMATH_CALUDE_triangle_and_function_problem_l2165_216524

open Real

theorem triangle_and_function_problem 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (m α : ℝ) :
  (2 * c * cos B = 2 * a + b) →
  (∀ x, 2 * sin (2 * x + π / 6) + m * cos (2 * x) = 2 * sin (2 * (C / 2 - x) + π / 6) + m * cos (2 * (C / 2 - x))) →
  (2 * sin (α + π / 6) + m * cos α = 6 / 5) →
  cos (2 * α + C) = -7 / 25 := by
sorry

end NUMINAMATH_CALUDE_triangle_and_function_problem_l2165_216524


namespace NUMINAMATH_CALUDE_field_length_calculation_l2165_216565

theorem field_length_calculation (w : ℝ) (l : ℝ) : 
  l = 2 * w →  -- length is double the width
  25 = (1/8) * (l * w) →  -- pond area (5^2) is 1/8 of field area
  l = 20 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l2165_216565


namespace NUMINAMATH_CALUDE_normal_symmetry_l2165_216569

/-- A random variable with normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a normal random variable is less than or equal to a given value -/
def normalCDF (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

theorem normal_symmetry (X : NormalRandomVariable) (a : ℝ) :
  normalCDF X 0 = 1 - normalCDF X (a - 2) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_normal_symmetry_l2165_216569


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2165_216539

theorem expression_simplification_and_evaluation :
  ∀ a : ℕ,
    a ≠ 0 →
    a ≠ 1 →
    2 * a - 3 ≤ 1 →
    a = 2 →
    (a - (2 * a - 1) / a) / ((a^2 - 1) / a) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2165_216539


namespace NUMINAMATH_CALUDE_business_school_class_l2165_216594

theorem business_school_class (p q r s : ℕ+) 
  (h_product : p * q * r * s = 1365)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s) : 
  p + q + r + s = 28 := by
  sorry

end NUMINAMATH_CALUDE_business_school_class_l2165_216594


namespace NUMINAMATH_CALUDE_macaroon_problem_l2165_216581

/-- Proves that the initial number of macaroons is 12 given the problem conditions -/
theorem macaroon_problem (weight_per_macaroon : ℕ) (num_bags : ℕ) (remaining_weight : ℕ) : 
  weight_per_macaroon = 5 →
  num_bags = 4 →
  remaining_weight = 45 →
  ∃ (initial_macaroons : ℕ),
    initial_macaroons = 12 ∧
    initial_macaroons % num_bags = 0 ∧
    (initial_macaroons / num_bags) * weight_per_macaroon * (num_bags - 1) = remaining_weight :=
by sorry

end NUMINAMATH_CALUDE_macaroon_problem_l2165_216581


namespace NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_max_k_no_intersection_l2165_216536

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

theorem tangent_parallel_condition (a : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, f a x = f a 1 + k * (x - 1)) ↔ a = Real.exp 1 := by sorry

theorem extreme_values (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > 0 → ∃ x : ℝ, x = Real.log a ∧ ∀ y : ℝ, y ≠ x → f a x < f a y) := by sorry

theorem max_k_no_intersection :
  ∃ k : ℝ, k = 1 ∧
    (∀ k' : ℝ, (∀ x : ℝ, f 1 x ≠ k' * x - 1) → k' ≤ k) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_max_k_no_intersection_l2165_216536


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_l2165_216562

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_l2165_216562


namespace NUMINAMATH_CALUDE_inequality_solution_l2165_216599

theorem inequality_solution : 
  ∀ x y : ℝ, y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y → 
  ((x = 0 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2)) ∧
  (x = 0 ∧ y = 0 → y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y) ∧
  (x = 1/2 ∧ y = 1/2 → y^2 + y + Real.sqrt (y - x^2 - x*y) ≤ 3*x*y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2165_216599


namespace NUMINAMATH_CALUDE_reversed_digits_multiple_l2165_216582

/-- Given a two-digit number that is k times the sum of its digits, 
    prove that the number formed by reversing its digits is (11 - k) times the sum of its digits. -/
theorem reversed_digits_multiple (k : ℕ) (u v : ℕ) : 
  (u ≤ 9 ∧ v ≤ 9 ∧ u ≠ 0) → 
  (10 * u + v = k * (u + v)) → 
  (10 * v + u = (11 - k) * (u + v)) :=
by sorry

end NUMINAMATH_CALUDE_reversed_digits_multiple_l2165_216582


namespace NUMINAMATH_CALUDE_adjacent_sides_equal_not_imply_rhombus_l2165_216545

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, dist (q.vertices i) (q.vertices j) = dist (q.vertices 0) (q.vertices 1)

-- Define adjacent sides
def adjacent_sides_equal (q : Quadrilateral) : Prop :=
  ∃ i : Fin 4, dist (q.vertices i) (q.vertices (i + 1)) = dist (q.vertices ((i + 1) % 4)) (q.vertices ((i + 2) % 4))

-- Theorem to prove
theorem adjacent_sides_equal_not_imply_rhombus :
  ¬(∀ q : Quadrilateral, adjacent_sides_equal q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_adjacent_sides_equal_not_imply_rhombus_l2165_216545


namespace NUMINAMATH_CALUDE_untouched_shapes_after_game_l2165_216500

-- Define the game state
structure GameState where
  triangles : Nat
  squares : Nat
  pentagons : Nat
  untouchedShapes : Nat
  turn : Nat

-- Define the initial game state
def initialState : GameState :=
  { triangles := 3
  , squares := 4
  , pentagons := 5
  , untouchedShapes := 12
  , turn := 0
  }

-- Define a move function for Petya
def petyaMove (state : GameState) : GameState :=
  { state with
    untouchedShapes := state.untouchedShapes - (if state.turn = 0 then 1 else 0)
    turn := state.turn + 1
  }

-- Define a move function for Vasya
def vasyaMove (state : GameState) : GameState :=
  { state with
    untouchedShapes := state.untouchedShapes - 1
    turn := state.turn + 1
  }

-- Define the final state after 10 turns
def finalState : GameState :=
  (List.range 5).foldl (fun state _ => vasyaMove (petyaMove state)) initialState

-- Theorem statement
theorem untouched_shapes_after_game :
  finalState.untouchedShapes = 6 := by sorry

end NUMINAMATH_CALUDE_untouched_shapes_after_game_l2165_216500


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2165_216552

theorem expand_and_simplify (x : ℝ) : (5*x - 3)*(2*x + 4) = 10*x^2 + 14*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2165_216552


namespace NUMINAMATH_CALUDE_swimming_hours_per_month_l2165_216518

/-- Calculate the required hours per month for freestyle and sidestroke swimming --/
theorem swimming_hours_per_month 
  (total_required : ℕ) 
  (completed : ℕ) 
  (months : ℕ) 
  (h1 : total_required = 1500) 
  (h2 : completed = 180) 
  (h3 : months = 6) :
  (total_required - completed) / months = 220 :=
by sorry

end NUMINAMATH_CALUDE_swimming_hours_per_month_l2165_216518


namespace NUMINAMATH_CALUDE_linear_dependency_condition_l2165_216589

-- Define the vectors
def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (k : ℝ) : Fin 2 → ℝ := ![1, k]

-- Define linear dependency
def is_linearly_dependent (v1 v2 : Fin 2 → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a • v1 + b • v2 = 0)

-- Theorem statement
theorem linear_dependency_condition (k : ℝ) :
  is_linearly_dependent v1 (v2 k) ↔ k = 2 :=
sorry

end NUMINAMATH_CALUDE_linear_dependency_condition_l2165_216589


namespace NUMINAMATH_CALUDE_train_passing_time_l2165_216510

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 375 →
  train_speed = 72 * (1000 / 3600) →
  man_speed = 12 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2165_216510


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2165_216541

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q ≠ 1 →
  (∀ n, a n > 0) →
  (a 3 + a 6 = 2 * a 5) →
  (a 3 + a 4) / (a 4 + a 5) = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2165_216541


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l2165_216596

/-- The number of nonzero terms in the expansion of (x^2+2)(3x^3+2x^2+4)-4(x^4+x^3-3x) -/
theorem nonzero_terms_count : ∃ (p : Polynomial ℝ), 
  p = (X^2 + 2) * (3*X^3 + 2*X^2 + 4) - 4*(X^4 + X^3 - 3*X) ∧ 
  p.support.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l2165_216596


namespace NUMINAMATH_CALUDE_two_face_painted_count_l2165_216504

/-- Represents a cube that has been painted on all faces and cut into smaller cubes --/
structure PaintedCube where
  /-- The number of smaller cubes along each edge of the original cube --/
  edge_count : Nat
  /-- Assumption that the cube is fully painted before cutting --/
  is_fully_painted : Bool

/-- Counts the number of smaller cubes painted on exactly two faces --/
def count_two_face_painted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a cube cut into 27 smaller cubes has 12 cubes painted on two faces --/
theorem two_face_painted_count (cube : PaintedCube) 
  (h1 : cube.edge_count = 3)
  (h2 : cube.is_fully_painted = true) : 
  count_two_face_painted_cubes cube = 12 :=
sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l2165_216504


namespace NUMINAMATH_CALUDE_fraction_equality_l2165_216511

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 10)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2165_216511


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l2165_216554

theorem sum_of_consecutive_integers (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (c = 7) → (a + b + c = 18) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l2165_216554


namespace NUMINAMATH_CALUDE_total_distance_walked_l2165_216501

/-- Given a pace of 2 miles per hour maintained for 8 hours, 
    the total distance walked is 16 miles. -/
theorem total_distance_walked 
  (pace : ℝ) 
  (duration : ℝ) 
  (h1 : pace = 2) 
  (h2 : duration = 8) : 
  pace * duration = 16 := by
sorry

end NUMINAMATH_CALUDE_total_distance_walked_l2165_216501


namespace NUMINAMATH_CALUDE_students_history_not_statistics_l2165_216560

/-- Given a group of students, prove the number taking history but not statistics -/
theorem students_history_not_statistics 
  (total : ℕ) 
  (history : ℕ) 
  (statistics : ℕ) 
  (history_or_statistics : ℕ) 
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 30)
  (h_history_or_statistics : history_or_statistics = 59) :
  history - (history + statistics - history_or_statistics) = 29 := by
  sorry

end NUMINAMATH_CALUDE_students_history_not_statistics_l2165_216560


namespace NUMINAMATH_CALUDE_max_xy_value_l2165_216577

theorem max_xy_value (x y : ℕ+) (h : 5 * x + 3 * y = 100) : x * y ≤ 165 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2165_216577


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l2165_216571

theorem triangle_square_perimeter_difference (d : ℕ) : 
  (∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    3 * a - 4 * b = 1989 ∧ 
    a - b = d ∧ 
    4 * b > 0) ↔ 
  d > 663 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l2165_216571


namespace NUMINAMATH_CALUDE_complete_square_result_l2165_216513

/-- Given a quadratic equation x^2 - 6x + 5 = 0, prove that when completing the square, 
    the resulting equation (x + c)^2 = d has d = 4 -/
theorem complete_square_result (c : ℝ) : 
  ∃ d : ℝ, (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + c)^2 = d) ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_result_l2165_216513


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l2165_216586

theorem function_satisfying_conditions (f : ℚ → ℚ) 
  (h1 : f 0 = 0) 
  (h2 : ∀ x y : ℚ, f (f x + f y) = x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l2165_216586


namespace NUMINAMATH_CALUDE_equilateral_triangle_tiling_l2165_216550

theorem equilateral_triangle_tiling (large_side : ℝ) (small_side : ℝ) : 
  large_side = 15 →
  small_side = 3 →
  (large_side^2 / small_side^2 : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_tiling_l2165_216550


namespace NUMINAMATH_CALUDE_dan_picked_nine_limes_l2165_216595

/-- The number of limes Dan has now -/
def total_limes : ℕ := 13

/-- The number of limes Sara gave to Dan -/
def sara_limes : ℕ := 4

/-- The number of limes Dan picked -/
def dan_picked_limes : ℕ := total_limes - sara_limes

theorem dan_picked_nine_limes : dan_picked_limes = 9 := by sorry

end NUMINAMATH_CALUDE_dan_picked_nine_limes_l2165_216595


namespace NUMINAMATH_CALUDE_quadratic_square_solutions_l2165_216561

theorem quadratic_square_solutions (n : ℕ) : 
  ∃ (p q : ℤ), ∃ (S : Finset ℤ), 
    (Finset.card S = n) ∧ 
    (∀ x : ℤ, x ∈ S ↔ ∃ y : ℕ, x^2 + p * x + q = y^2) ∧
    (∀ x y : ℤ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_solutions_l2165_216561


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l2165_216584

theorem area_between_concentric_circles 
  (r : Real) -- radius of inner circle
  (h1 : r > 0) -- inner radius is positive
  (h2 : 3 * r - r = 4) -- difference between outer and inner radii is 4
  : π * (3 * r)^2 - π * r^2 = 32 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l2165_216584


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2165_216537

/-- Sum of the first n terms of a geometric sequence -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/6
  let r : ℚ := 1/2
  let n : ℕ := 6
  geometricSum a r n = 21/64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2165_216537


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l2165_216531

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 10 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l2165_216531


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_conditions_l2165_216593

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_nonprime_with_conditions (n : ℕ) : 
  n = 289 ↔ 
    (¬is_prime n ∧ 
     n > 25 ∧ 
     has_no_prime_factor_less_than n 15 ∧ 
     sum_of_digits n > 10 ∧
     ∀ m : ℕ, m < n → 
       (¬is_prime m → 
        m ≤ 25 ∨ 
        ¬has_no_prime_factor_less_than m 15 ∨ 
        sum_of_digits m ≤ 10)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_conditions_l2165_216593


namespace NUMINAMATH_CALUDE_solve_z_l2165_216514

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a predicate for purely imaginary numbers
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define the theorem
theorem solve_z (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : ((z + 2) / (1 - i)).im = 0) : 
  z = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_solve_z_l2165_216514


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2165_216587

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | 3 * x - x^2 ≥ 0} = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2165_216587


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l2165_216521

theorem company_picnic_attendance (men_attendance : Real) (women_attendance : Real) 
  (total_attendance : Real) (men_percentage : Real) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.31000000000000007 →
  men_attendance * men_percentage + women_attendance * (1 - men_percentage) = total_attendance →
  men_percentage = 0.45 := by
sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l2165_216521


namespace NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_nine_l2165_216578

/-- Represents the number of balls of each color in the box -/
structure BoxContents where
  purple : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the minimum number of tries to get one blue and one yellow ball -/
def minTriesForBlueAndYellow (box : BoxContents) : Nat :=
  box.purple + 2

/-- Theorem stating the minimum number of tries for the given box contents -/
theorem min_tries_for_blue_and_yellow_is_nine :
  let box : BoxContents := { purple := 7, blue := 5, yellow := 11 }
  minTriesForBlueAndYellow box = 9 := by
  sorry


end NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_nine_l2165_216578


namespace NUMINAMATH_CALUDE_distance_AK_equals_sqrt2_plus_1_l2165_216512

/-- Given a quadrilateral ABCD with vertices A(0, 0), B(0, -1), C(1, 0), D(√2/2, √2/2),
    and K is the intersection point of lines AB and CD,
    prove that the distance AK = √2 + 1 -/
theorem distance_AK_equals_sqrt2_plus_1 :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, -1)
  let C : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))  -- Intersection point of AB and CD
  -- Distance formula
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A K = Real.sqrt 2 + 1 := by sorry

end NUMINAMATH_CALUDE_distance_AK_equals_sqrt2_plus_1_l2165_216512


namespace NUMINAMATH_CALUDE_average_problem_l2165_216575

theorem average_problem (x : ℝ) : (15 + 25 + 35 + x) / 4 = 30 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2165_216575


namespace NUMINAMATH_CALUDE_optimal_bouquet_l2165_216507

/-- Represents the number of flowers Kyle picked last year -/
def last_year_flowers : ℕ := 12

/-- Represents the total number of flowers needed this year -/
def total_flowers : ℕ := 2 * last_year_flowers

/-- Represents the number of roses Kyle picked from his garden this year -/
def picked_roses : ℕ := last_year_flowers / 2

/-- Represents the cost of a rose -/
def rose_cost : ℕ := 3

/-- Represents the cost of a tulip -/
def tulip_cost : ℕ := 2

/-- Represents the cost of a daisy -/
def daisy_cost : ℕ := 1

/-- Represents Kyle's budget constraint -/
def budget : ℕ := 30

/-- Represents the number of additional flowers Kyle needs to buy -/
def flowers_to_buy : ℕ := total_flowers - picked_roses

theorem optimal_bouquet (roses tulips daisies : ℕ) :
  roses + tulips + daisies = flowers_to_buy →
  rose_cost * roses + tulip_cost * tulips + daisy_cost * daisies ≤ budget →
  roses ≤ 9 ∧
  (roses = 9 → tulips = 1 ∧ daisies = 1) :=
sorry

end NUMINAMATH_CALUDE_optimal_bouquet_l2165_216507


namespace NUMINAMATH_CALUDE_three_pairs_satisfy_l2165_216534

/-- The set S of elements -/
inductive S
| A₀ : S
| A₁ : S
| A₂ : S

/-- The operation ⊕ on S -/
def op (x y : S) : S :=
  match x, y with
  | S.A₀, S.A₀ => S.A₀
  | S.A₀, S.A₁ => S.A₁
  | S.A₀, S.A₂ => S.A₂
  | S.A₁, S.A₀ => S.A₁
  | S.A₁, S.A₁ => S.A₂
  | S.A₁, S.A₂ => S.A₀
  | S.A₂, S.A₀ => S.A₂
  | S.A₂, S.A₁ => S.A₀
  | S.A₂, S.A₂ => S.A₁

/-- The theorem stating that there are exactly 3 pairs satisfying the equation -/
theorem three_pairs_satisfy :
  ∃! (pairs : List (S × S)), pairs.length = 3 ∧
    ∀ (x y : S), (op (op x y) x = S.A₀) ↔ (x, y) ∈ pairs :=
by sorry

end NUMINAMATH_CALUDE_three_pairs_satisfy_l2165_216534


namespace NUMINAMATH_CALUDE_limit_of_f_is_one_fourth_l2165_216566

def C (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

def f (n : ℕ) : ℚ := (C n 2 : ℚ) / (2 * n^2 + n : ℚ)

theorem limit_of_f_is_one_fourth :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n - 1/4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_f_is_one_fourth_l2165_216566


namespace NUMINAMATH_CALUDE_kickball_total_players_kickball_problem_l2165_216598

theorem kickball_total_players (wed_morning : ℕ) (wed_afternoon_increase : ℕ) 
  (thu_morning_decrease : ℕ) (thu_lunchtime_decrease : ℕ) : ℕ :=
  let wed_afternoon := wed_morning + wed_afternoon_increase
  let thu_morning := wed_morning - thu_morning_decrease
  let thu_afternoon := thu_morning - thu_lunchtime_decrease
  let wed_total := wed_morning + wed_afternoon
  let thu_total := thu_morning + thu_afternoon
  wed_total + thu_total

theorem kickball_problem :
  kickball_total_players 37 15 9 7 = 138 := by
  sorry

end NUMINAMATH_CALUDE_kickball_total_players_kickball_problem_l2165_216598


namespace NUMINAMATH_CALUDE_tangent_line_slope_positive_l2165_216592

/-- Given a function f: ℝ → ℝ, if the tangent line to the curve y = f(x) at the point (2, f(2)) 
    passes through the point (-1, 2), then f'(2) > 0. -/
theorem tangent_line_slope_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (deriv f 2) * (2 - (-1)) = f 2 - 2) : 
  deriv f 2 > 0 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_slope_positive_l2165_216592


namespace NUMINAMATH_CALUDE_quadratic_congruences_equivalence_l2165_216523

theorem quadratic_congruences_equivalence (p : Nat) (h : Nat.Prime p) :
  (∃ x, (x^2 + x + 3) % p = 0 → ∃ y, (y^2 + y + 25) % p = 0) ∧
  (¬∃ x, (x^2 + x + 3) % p = 0 → ¬∃ y, (y^2 + y + 25) % p = 0) ∧
  (∃ y, (y^2 + y + 25) % p = 0 → ∃ x, (x^2 + x + 3) % p = 0) ∧
  (¬∃ y, (y^2 + y + 25) % p = 0 → ¬∃ x, (x^2 + x + 3) % p = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_congruences_equivalence_l2165_216523


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_8_l2165_216547

/-- A function that returns the product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_with_digit_product_8 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 8 → n ≤ 81 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_8_l2165_216547


namespace NUMINAMATH_CALUDE_banana_arrangements_l2165_216568

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- Theorem: The number of unique arrangements of "BANANA" is 60 -/
theorem banana_arrangements :
  uniqueArrangements 6 [3, 2, 1] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2165_216568


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l2165_216544

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from n balls -/
def possible_lists (n k : ℕ) : ℕ := n ^ k

theorem ball_drawing_theorem : possible_lists n k = 50625 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l2165_216544


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l2165_216548

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLucky (n : ℕ) : Prop := n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf7 (n : ℕ) : Prop := n % 7 = 0

theorem least_non_lucky_multiple_of_7 : 
  (∀ k : ℕ, k > 0 ∧ k < 14 ∧ isMultipleOf7 k → isLucky k) ∧ 
  isMultipleOf7 14 ∧ 
  ¬isLucky 14 := by sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l2165_216548


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_l2165_216532

/-- An arithmetic sequence with given first and second terms -/
def arithmetic_sequence (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * (a₂ - a₁)

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + arithmetic_sequence a₁ a₂ n) / 2

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence
    with a₁ = 1 and a₂ = -3 is -170 -/
theorem arithmetic_sum_10 :
  arithmetic_sum 1 (-3) 10 = -170 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_10_l2165_216532


namespace NUMINAMATH_CALUDE_sum_of_angles_less_than_90_degrees_l2165_216580

/-- A line intersecting two perpendicular planes forms angles α and β with these planes. -/
structure LineIntersectingPerpendicularPlanes where
  α : Real
  β : Real

/-- The theorem states that the sum of angles α and β is always less than 90 degrees. -/
theorem sum_of_angles_less_than_90_degrees (l : LineIntersectingPerpendicularPlanes) :
  l.α + l.β < 90 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_less_than_90_degrees_l2165_216580


namespace NUMINAMATH_CALUDE_grapes_in_robs_bowl_l2165_216549

theorem grapes_in_robs_bowl (rob_grapes : ℕ) 
  (allie_grapes : ℕ) (allyn_grapes : ℕ) : 
  (allie_grapes = rob_grapes + 2) → 
  (allyn_grapes = allie_grapes + 4) → 
  (rob_grapes + allie_grapes + allyn_grapes = 83) → 
  rob_grapes = 25 := by
sorry

end NUMINAMATH_CALUDE_grapes_in_robs_bowl_l2165_216549


namespace NUMINAMATH_CALUDE_area_ratio_is_nine_thirtytwo_l2165_216572

/-- Triangle XYZ with points G, H, I on its sides -/
structure TriangleXYZ where
  /-- Length of side XY -/
  xy : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- Length of side ZX -/
  zx : ℝ
  /-- Ratio of XG to XY -/
  s : ℝ
  /-- Ratio of YH to YZ -/
  t : ℝ
  /-- Ratio of ZI to ZX -/
  u : ℝ
  /-- XY length is 14 -/
  xy_eq : xy = 14
  /-- YZ length is 16 -/
  yz_eq : yz = 16
  /-- ZX length is 18 -/
  zx_eq : zx = 18
  /-- s is positive -/
  s_pos : s > 0
  /-- t is positive -/
  t_pos : t > 0
  /-- u is positive -/
  u_pos : u > 0
  /-- Sum of s, t, u is 3/4 -/
  sum_stu : s + t + u = 3/4
  /-- Sum of squares of s, t, u is 3/8 -/
  sum_sq_stu : s^2 + t^2 + u^2 = 3/8

/-- The ratio of the area of triangle GHI to the area of triangle XYZ -/
def areaRatio (T : TriangleXYZ) : ℝ :=
  1 - T.s * (1 - T.u) - T.t * (1 - T.s) - T.u * (1 - T.t)

theorem area_ratio_is_nine_thirtytwo (T : TriangleXYZ) : 
  areaRatio T = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_nine_thirtytwo_l2165_216572


namespace NUMINAMATH_CALUDE_hospital_workers_count_l2165_216527

theorem hospital_workers_count :
  let total_workers : ℕ := 2 + 3  -- Jack, Jill, and 3 others
  let interview_size : ℕ := 2
  let prob_jack_and_jill : ℚ := 1 / 10  -- 0.1 as a rational number
  total_workers = 5 ∧
  interview_size = 2 ∧
  prob_jack_and_jill = 1 / Nat.choose total_workers interview_size :=
by sorry

end NUMINAMATH_CALUDE_hospital_workers_count_l2165_216527


namespace NUMINAMATH_CALUDE_patsy_guests_l2165_216585

-- Define the problem parameters
def appetizers_per_guest : ℕ := 6
def initial_dozens : ℕ := 3 + 2 + 2
def additional_dozens : ℕ := 8

-- Define the theorem
theorem patsy_guests :
  (((initial_dozens + additional_dozens) * 12) / appetizers_per_guest) = 30 := by
  sorry

end NUMINAMATH_CALUDE_patsy_guests_l2165_216585


namespace NUMINAMATH_CALUDE_max_min_difference_l2165_216590

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Define the interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧
               (∀ x ∈ I, m ≤ f x) ∧
               (M - m = 32) :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l2165_216590


namespace NUMINAMATH_CALUDE_incorrect_equation_l2165_216551

theorem incorrect_equation (a b : ℤ) : 
  (-a + b = -1) → (a + b = 5) → (4*a + b = 14) → (2*a + b ≠ 7) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_l2165_216551


namespace NUMINAMATH_CALUDE_seven_trapezoid_solutions_l2165_216516

/-- The number of solutions for the trapezoidal park problem -/
def trapezoid_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 % 5 = 0 ∧
    p.2 % 5 = 0 ∧
    75 * (p.1 + p.2) / 2 = 2250 ∧
    p.1 ≤ p.2)
    (Finset.product (Finset.range 61) (Finset.range 61))).card

/-- The theorem stating that there are exactly seven solutions -/
theorem seven_trapezoid_solutions : trapezoid_solutions = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_trapezoid_solutions_l2165_216516


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l2165_216570

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits. -/
def binary (bits : List Bool) : ℕ := binaryToNat bits

theorem binary_addition_subtraction :
  let a := binary [true, true, true, false, true]  -- 11101₂
  let b := binary [true, true, false, true]        -- 1101₂
  let c := binary [true, false, true, true, false] -- 10110₂
  let d := binary [true, false, true, true]        -- 1011₂
  let result := binary [true, true, false, true, true] -- 11011₂
  a + b - c + d = result := by sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l2165_216570


namespace NUMINAMATH_CALUDE_solve_for_r_l2165_216528

theorem solve_for_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 3^r) 
  (h2 : 45 = k * 9^r) : 
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_r_l2165_216528


namespace NUMINAMATH_CALUDE_equation_solution_l2165_216520

theorem equation_solution :
  ∃! x : ℝ, (x^2 + 4*x + 5) / (x + 3) = x + 7 ∧ x ≠ -3 :=
by
  use (-8/3)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2165_216520


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2165_216509

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2165_216509


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2165_216505

theorem arithmetic_sequence_problem (x : ℚ) (n : ℕ) : 
  let a₁ := 3 * x - 2
  let a₂ := 7 * x - 15
  let a₃ := 4 * x + 3
  let d := a₂ - a₁
  let aₙ := a₁ + (n - 1) * d
  (a₂ - a₁ = a₃ - a₂) ∧ (aₙ = 4020) → n = 851 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2165_216505


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2165_216508

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2165_216508


namespace NUMINAMATH_CALUDE_simon_fraction_of_alvin_age_l2165_216576

/-- Given that Alvin is 30 years old and Simon is 10 years old, prove that Simon will be 3/7 of Alvin's age in 5 years. -/
theorem simon_fraction_of_alvin_age (alvin_age : ℕ) (simon_age : ℕ) : 
  alvin_age = 30 → simon_age = 10 → (simon_age + 5 : ℚ) / (alvin_age + 5 : ℚ) = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_simon_fraction_of_alvin_age_l2165_216576


namespace NUMINAMATH_CALUDE_inequality_proof_l2165_216519

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) +
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) +
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
    sorry

end NUMINAMATH_CALUDE_inequality_proof_l2165_216519


namespace NUMINAMATH_CALUDE_max_value_constraint_l2165_216533

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 2) :
  x + y^3 + z^2 ≤ 8 ∧ ∃ (a b c : ℝ), a + b^3 + c^2 = 8 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2165_216533


namespace NUMINAMATH_CALUDE_set_condition_l2165_216556

theorem set_condition (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ↔ x^2 - 2*x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_set_condition_l2165_216556


namespace NUMINAMATH_CALUDE_circle_tangency_l2165_216526

theorem circle_tangency (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 1 ∧ (p.1 + 4)^2 + (p.2 - a)^2 = 25) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_l2165_216526


namespace NUMINAMATH_CALUDE_radio_selling_price_l2165_216583

def purchase_price : ℚ := 232
def overhead_expenses : ℚ := 15
def profit_percent : ℚ := 21.457489878542503

def total_cost_price : ℚ := purchase_price + overhead_expenses

def profit_amount : ℚ := (profit_percent / 100) * total_cost_price

def selling_price : ℚ := total_cost_price + profit_amount

theorem radio_selling_price : 
  ∃ (sp : ℚ), sp = selling_price ∧ round sp = 300 := by sorry

end NUMINAMATH_CALUDE_radio_selling_price_l2165_216583


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l2165_216558

/-- The number of letters in an alphabet with specific dot and line properties -/
theorem alphabet_letter_count :
  let dot_and_line : ℕ := 13  -- Letters with both dot and line
  let line_only : ℕ := 24     -- Letters with line but no dot
  let dot_only : ℕ := 3       -- Letters with dot but no line
  let total : ℕ := dot_and_line + line_only + dot_only
  total = 40 := by sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l2165_216558


namespace NUMINAMATH_CALUDE_solve_car_price_l2165_216525

def car_price_problem (total_payment loan_amount interest_rate : ℝ) : Prop :=
  let interest := loan_amount * interest_rate
  let car_price := total_payment - interest
  car_price = 35000

theorem solve_car_price :
  car_price_problem 38000 20000 0.15 :=
sorry

end NUMINAMATH_CALUDE_solve_car_price_l2165_216525


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_range_l2165_216579

theorem fraction_integer_iff_p_range (p : ℕ+) :
  (∃ (k : ℕ+), (3 * p.val + 25 : ℤ) = k.val * (2 * p.val - 5)) ↔ 3 ≤ p.val ∧ p.val ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_range_l2165_216579


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_has_unique_solution_l2165_216563

/-- Represents the number of chickens and rabbits in a cage. -/
structure AnimalCount where
  chickens : ℕ
  rabbits : ℕ

/-- Checks if the given animal count satisfies the problem conditions. -/
def satisfiesConditions (count : AnimalCount) : Prop :=
  count.chickens = 2 * (4 * count.rabbits) - 5 ∧
  2 * count.chickens + count.rabbits = 92

/-- There exists a unique solution to the chicken and rabbit problem. -/
theorem chicken_rabbit_problem_has_unique_solution :
  ∃! count : AnimalCount, satisfiesConditions count :=
sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_has_unique_solution_l2165_216563


namespace NUMINAMATH_CALUDE_quadratic_domain_range_implies_power_l2165_216538

/-- A quadratic function f(x) = x^2 - 4x + 4 + m with domain and range [2, n] -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 4 + m

/-- The theorem stating that if f has domain and range [2, n], then m^n = 8 -/
theorem quadratic_domain_range_implies_power (m n : ℝ) :
  (∀ x, x ∈ Set.Icc 2 n ↔ f m x ∈ Set.Icc 2 n) →
  m^n = 8 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_domain_range_implies_power_l2165_216538


namespace NUMINAMATH_CALUDE_root_in_interval_l2165_216559

-- Define the function
def f (x : ℝ) := x^3 - 2*x - 1

-- State the theorem
theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) →
  (∃ x ∈ Set.Ioo (3/2) 2, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2165_216559


namespace NUMINAMATH_CALUDE_lg_meaningful_iff_first_or_second_quadrant_l2165_216567

open Real

-- Define the meaningful condition for lg(cos θ · tan θ)
def is_meaningful (θ : ℝ) : Prop :=
  sin θ > 0 ∧ sin θ ≠ 1

-- Define the first and second quadrants
def in_first_or_second_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π

-- Theorem statement
theorem lg_meaningful_iff_first_or_second_quadrant (θ : ℝ) :
  is_meaningful θ ↔ in_first_or_second_quadrant θ :=
sorry

end NUMINAMATH_CALUDE_lg_meaningful_iff_first_or_second_quadrant_l2165_216567


namespace NUMINAMATH_CALUDE_odd_composite_quotient_l2165_216517

theorem odd_composite_quotient : 
  let first_four := [9, 15, 21, 25]
  let next_four := [27, 33, 35, 39]
  (first_four.prod : ℚ) / (next_four.prod : ℚ) = 25 / 429 := by
  sorry

end NUMINAMATH_CALUDE_odd_composite_quotient_l2165_216517


namespace NUMINAMATH_CALUDE_melanie_dimes_problem_l2165_216543

/-- The number of dimes Melanie gave her mother -/
def dimes_given_to_mother (initial dimes_from_dad final : ℕ) : ℕ :=
  initial + dimes_from_dad - final

theorem melanie_dimes_problem (initial dimes_from_dad final : ℕ) 
  (h1 : initial = 7)
  (h2 : dimes_from_dad = 8)
  (h3 : final = 11) :
  dimes_given_to_mother initial dimes_from_dad final = 4 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_problem_l2165_216543


namespace NUMINAMATH_CALUDE_sin_45_eq_neg_cos_135_l2165_216535

theorem sin_45_eq_neg_cos_135 : Real.sin (π / 4) = - Real.cos (3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_45_eq_neg_cos_135_l2165_216535


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l2165_216503

/-- The total shaded area of a carpet with a circle and squares -/
theorem carpet_shaded_area (carpet_side : ℝ) (circle_diameter : ℝ) (square_side : ℝ) : 
  carpet_side = 12 →
  carpet_side / circle_diameter = 4 →
  circle_diameter / square_side = 4 →
  (π * (circle_diameter / 2)^2) + (8 * square_side^2) = (9 * π / 4) + (9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l2165_216503


namespace NUMINAMATH_CALUDE_value_of_x_l2165_216555

theorem value_of_x (x y z : ℝ) : 
  x = (1/2) * y → 
  y = (1/4) * z → 
  z = 80 → 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2165_216555


namespace NUMINAMATH_CALUDE_equation_solution_l2165_216573

theorem equation_solution :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^3 - 3*x₁*y₁^2 = 2007) ∧ (y₁^3 - 3*x₁^2*y₁ = 2006) ∧
    (x₂^3 - 3*x₂*y₂^2 = 2007) ∧ (y₂^3 - 3*x₂^2*y₂ = 2006) ∧
    (x₃^3 - 3*x₃*y₃^2 = 2007) ∧ (y₃^3 - 3*x₃^2*y₃ = 2006) →
    (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1003 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2165_216573


namespace NUMINAMATH_CALUDE_sufficient_condition_for_positive_quadratic_l2165_216529

theorem sufficient_condition_for_positive_quadratic (m : ℝ) :
  m > 1 → ∀ x : ℝ, x^2 - 2*x + m > 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_positive_quadratic_l2165_216529


namespace NUMINAMATH_CALUDE_squirrel_stocks_l2165_216515

structure Squirrel where
  mushrooms : ℕ
  hazelnuts : ℕ
  fir_cones : ℕ

def total_items (s : Squirrel) : ℕ := s.mushrooms + s.hazelnuts + s.fir_cones

theorem squirrel_stocks :
  ∃ (zrzecka pizizubka krivoousko : Squirrel),
    -- Each squirrel has 48 mushrooms
    zrzecka.mushrooms = 48 ∧ pizizubka.mushrooms = 48 ∧ krivoousko.mushrooms = 48 ∧
    -- Zrzečka has twice as many hazelnuts as Pizizubka
    zrzecka.hazelnuts = 2 * pizizubka.hazelnuts ∧
    -- Křivoouško has 20 more hazelnuts than Pizizubka
    krivoousko.hazelnuts = pizizubka.hazelnuts + 20 ∧
    -- Together, they have 180 fir cones and 180 hazelnuts
    zrzecka.fir_cones + pizizubka.fir_cones + krivoousko.fir_cones = 180 ∧
    zrzecka.hazelnuts + pizizubka.hazelnuts + krivoousko.hazelnuts = 180 ∧
    -- All squirrels have the same total number of items
    total_items zrzecka = total_items pizizubka ∧
    total_items pizizubka = total_items krivoousko ∧
    -- The correct distribution of items
    zrzecka = { mushrooms := 48, hazelnuts := 80, fir_cones := 40 } ∧
    pizizubka = { mushrooms := 48, hazelnuts := 40, fir_cones := 80 } ∧
    krivoousko = { mushrooms := 48, hazelnuts := 60, fir_cones := 60 } :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_stocks_l2165_216515


namespace NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l2165_216502

/-- Represents a counter in the Ring Mafia game -/
inductive Counter
| Mafia
| Town

/-- Represents the state of the Ring Mafia game -/
structure GameState where
  counters : List Counter
  total_counters : Nat
  mafia_counters : Nat
  town_counters : Nat

/-- Represents a strategy for Tony -/
def TonyStrategy := GameState → List Nat

/-- Represents a strategy for Madeline -/
def MadelineStrategy := GameState → Nat

/-- Checks if a game state is valid according to the rules -/
def is_valid_game_state (state : GameState) : Prop :=
  state.total_counters = 2019 ∧
  state.mafia_counters = 673 ∧
  state.town_counters = 1346 ∧
  state.counters.length = state.total_counters

/-- Checks if the game has ended -/
def game_ended (state : GameState) : Prop :=
  state.mafia_counters = 0 ∨ state.town_counters = 0

/-- Theorem: There is no winning strategy for Tony in Ring Mafia -/
theorem no_winning_strategy_for_tony :
  ∀ (initial_state : GameState),
    is_valid_game_state initial_state →
    ∀ (tony_strategy : TonyStrategy),
    ∃ (madeline_strategy : MadelineStrategy),
    ∃ (final_state : GameState),
      game_ended final_state ∧
      final_state.town_counters = 0 :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l2165_216502


namespace NUMINAMATH_CALUDE_correct_email_sequence_l2165_216530

/-- Represents the steps in sending an email -/
inductive EmailStep
  | OpenEmailBox
  | EnterRecipientAddress
  | EnterSubject
  | EnterContent
  | ClickCompose
  | ClickSend

/-- Represents a sequence of email steps -/
def EmailSequence := List EmailStep

/-- The correct sequence of steps for sending an email -/
def correctEmailSequence : EmailSequence :=
  [EmailStep.OpenEmailBox, EmailStep.ClickCompose, EmailStep.EnterRecipientAddress,
   EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend]

/-- Theorem stating that the given sequence is the correct one for sending an email -/
theorem correct_email_sequence :
  correctEmailSequence =
    [EmailStep.OpenEmailBox, EmailStep.ClickCompose, EmailStep.EnterRecipientAddress,
     EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend] :=
by
  sorry

end NUMINAMATH_CALUDE_correct_email_sequence_l2165_216530


namespace NUMINAMATH_CALUDE_probability_three_of_a_kind_after_reroll_l2165_216597

/-- The probability of getting at least three dice showing the same value after re-rolling the unmatched die in a specific dice configuration. -/
theorem probability_three_of_a_kind_after_reroll (n : ℕ) (p : ℚ) : 
  n = 5 → -- number of dice
  p = 1 / 3 → -- probability we want to prove
  ∃ (X Y : ℕ), -- the two pair values
    X ≠ Y ∧ 
    X ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧ 
    Y ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
  p = (1 : ℚ) / 6 + (1 : ℚ) / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_of_a_kind_after_reroll_l2165_216597


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2165_216553

/-- Calculates the total surface area of a cube with holes --/
def totalSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let internalSurface := 6 * 4 * holeEdge^2
  originalSurface - holeArea + internalSurface

/-- The problem statement --/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2165_216553


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l2165_216591

theorem tank_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l2165_216591


namespace NUMINAMATH_CALUDE_complex_multiplication_l2165_216574

/-- Given that i is the imaginary unit, prove that i(3-4i) = 4 + 3i -/
theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (3 - 4*i) = 4 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2165_216574


namespace NUMINAMATH_CALUDE_lions_escaped_l2165_216588

/-- The number of rhinos that escaped -/
def rhinos : ℕ := 2

/-- The time (in hours) to recover each animal -/
def recovery_time : ℕ := 2

/-- The total time (in hours) spent recovering all animals -/
def total_time : ℕ := 10

/-- The number of lions that escaped -/
def lions : ℕ := (total_time - rhinos * recovery_time) / recovery_time

theorem lions_escaped :
  lions = 3 :=
by sorry

end NUMINAMATH_CALUDE_lions_escaped_l2165_216588


namespace NUMINAMATH_CALUDE_undefined_rational_expression_l2165_216542

theorem undefined_rational_expression (x : ℝ) :
  (x^2 - 16*x + 64 = 0) ↔ (x = 8) :=
by sorry

end NUMINAMATH_CALUDE_undefined_rational_expression_l2165_216542


namespace NUMINAMATH_CALUDE_congruence_problem_l2165_216546

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (100 * n) % 103 = 85 % 103 → n % 103 = 6 % 103 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2165_216546
