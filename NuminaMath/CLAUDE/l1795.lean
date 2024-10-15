import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l1795_179535

theorem sin_cos_equation_solutions (x : ℝ) :
  (0 ≤ x ∧ x < 2 * Real.pi) ∧ (Real.sin x - Real.cos x = Real.sqrt 3 / 2) ↔
  (x = Real.arcsin (Real.sqrt 6 / 4) - Real.pi / 4 ∨
   x = Real.pi - Real.arcsin (Real.sqrt 6 / 4) - Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l1795_179535


namespace NUMINAMATH_CALUDE_concentric_circles_shaded_area_l1795_179533

/-- Given two concentric circles where the smaller circle's radius is half of the larger circle's radius,
    and the area of the larger circle is 144π, the sum of the areas of the upper halves of both circles
    is equal to 90π. -/
theorem concentric_circles_shaded_area (R r : ℝ) : 
  R > 0 ∧ r = R / 2 ∧ π * R^2 = 144 * π → 
  (π * R^2) / 2 + (π * r^2) / 2 = 90 * π := by
  sorry


end NUMINAMATH_CALUDE_concentric_circles_shaded_area_l1795_179533


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l1795_179579

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.00000012 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l1795_179579


namespace NUMINAMATH_CALUDE_terminal_sides_theorem_l1795_179590

/-- Given an angle θ in degrees, returns true if the terminal side of 7θ coincides with the terminal side of θ -/
def terminal_sides_coincide (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + k * 360

/-- The set of angles whose terminal sides coincide with their 7θ counterparts -/
def coinciding_angles : Set ℝ := {0, 60, 120, 180, 240, 300}

theorem terminal_sides_theorem (θ : ℝ) :
  0 ≤ θ ∧ θ < 360 ∧ terminal_sides_coincide θ → θ ∈ coinciding_angles := by
  sorry

end NUMINAMATH_CALUDE_terminal_sides_theorem_l1795_179590


namespace NUMINAMATH_CALUDE_ab_value_l1795_179529

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1795_179529


namespace NUMINAMATH_CALUDE_remainder_of_n_l1795_179561

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^4 % 5 = 1) :
  n % 5 = 1 ∨ n % 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l1795_179561


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1795_179504

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 10*x + 3 ≥ -22) ∧ (∃ x : ℝ, x^2 + 10*x + 3 = -22) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1795_179504


namespace NUMINAMATH_CALUDE_programmer_work_hours_l1795_179552

theorem programmer_work_hours (flow_chart_time : ℚ) (coding_time : ℚ) (debug_time : ℚ) 
  (h1 : flow_chart_time = 1/4)
  (h2 : coding_time = 3/8)
  (h3 : debug_time = 1 - (flow_chart_time + coding_time))
  (h4 : debug_time * 48 = 18) :
  48 = 48 := by sorry

end NUMINAMATH_CALUDE_programmer_work_hours_l1795_179552


namespace NUMINAMATH_CALUDE_max_rectangles_l1795_179577

/-- Represents a cell in the figure -/
inductive Cell
| White
| Black

/-- Represents the figure as a 2D array of cells -/
def Figure := Array (Array Cell)

/-- Checks if a figure has alternating black and white cells -/
def hasAlternatingColors (fig : Figure) : Prop := sorry

/-- Checks if the middle diagonal of a figure is black -/
def hasBlackDiagonal (fig : Figure) : Prop := sorry

/-- Counts the number of black cells in a figure -/
def countBlackCells (fig : Figure) : Nat := sorry

/-- Represents a 1x2 rectangle placement in the figure -/
structure Rectangle where
  row : Nat
  col : Nat

/-- Checks if a rectangle placement is valid (spans one black and one white cell) -/
def isValidRectangle (fig : Figure) (rect : Rectangle) : Prop := sorry

/-- The main theorem -/
theorem max_rectangles (fig : Figure) 
  (h1 : hasAlternatingColors fig)
  (h2 : hasBlackDiagonal fig) :
  (∃ (rects : List Rectangle), 
    (∀ r ∈ rects, isValidRectangle fig r) ∧ 
    rects.length = countBlackCells fig) ∧
  (∀ (rects : List Rectangle), 
    (∀ r ∈ rects, isValidRectangle fig r) → 
    rects.length ≤ countBlackCells fig) := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_l1795_179577


namespace NUMINAMATH_CALUDE_fraction_difference_prime_l1795_179523

theorem fraction_difference_prime (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / p ↔ x = p - 1 ∧ y = p * (p - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_prime_l1795_179523


namespace NUMINAMATH_CALUDE_least_common_multiple_plus_one_l1795_179520

def divisors : List Nat := [2, 3, 5, 7, 8, 9, 10]

theorem least_common_multiple_plus_one : 
  ∃ (n : Nat), n > 1 ∧ 
  (∀ d ∈ divisors, n % d = 1) ∧
  (∀ m : Nat, m > 1 → (∀ d ∈ divisors, m % d = 1) → m ≥ n) ∧
  n = 2521 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_plus_one_l1795_179520


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1795_179566

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 10 = -2) :
  a 4 * a 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1795_179566


namespace NUMINAMATH_CALUDE_line_properties_l1795_179599

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ

/-- The theorem stating the properties of the given parameterized line -/
theorem line_properties (L : ParameterizedLine) : 
  L.point 1 = (2, 5) ∧ L.point 4 = (5, -7) → L.point 0 = (1, 9) := by
  sorry


end NUMINAMATH_CALUDE_line_properties_l1795_179599


namespace NUMINAMATH_CALUDE_problem_statement_l1795_179558

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / (1 - 2*m) + y^2 / (m + 2) = 1 ∧ (1 - 2*m) * (m + 2) < 0

theorem problem_statement :
  (∀ m : ℝ, p m ↔ (m ≤ -1 ∨ m ≥ 2)) ∧
  (∀ m : ℝ, q m ↔ (m < -2 ∨ m > 1/2)) ∧
  (∀ m : ℝ, ¬(p m ∨ q m) ↔ (-1 < m ∧ m ≤ 1/2)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1795_179558


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l1795_179516

def a : ℝ × ℝ := (-1, 1)
def b (m : ℝ) : ℝ × ℝ := (1, m)

theorem vector_dot_product_problem (m : ℝ) : 
  (2 * a - b m) • a = 4 → m = 1 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l1795_179516


namespace NUMINAMATH_CALUDE_whatsapp_messages_l1795_179556

/-- The number of messages sent in a Whatsapp group over four days -/
def total_messages (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem: Given the conditions of the Whatsapp group messages, 
    the total number of messages over four days is 2000 -/
theorem whatsapp_messages : 
  ∀ (monday tuesday wednesday thursday : ℕ),
    monday = 300 →
    tuesday = 200 →
    wednesday = tuesday + 300 →
    thursday = 2 * wednesday →
    total_messages monday tuesday wednesday thursday = 2000 :=
by
  sorry


end NUMINAMATH_CALUDE_whatsapp_messages_l1795_179556


namespace NUMINAMATH_CALUDE_polynomial_inverse_property_l1795_179551

-- Define the polynomials p and P
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def P (A B C : ℝ) (x : ℝ) : ℝ := A * x^2 + B * x + C

-- State the theorem
theorem polynomial_inverse_property 
  (a b c A B C : ℝ) : 
  (∀ x : ℝ, P A B C (p a b c x) = x) → 
  (∀ x : ℝ, p a b c (P A B C x) = x) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_inverse_property_l1795_179551


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1795_179541

theorem five_digit_multiple_of_nine : ∃ (d : ℕ), d < 10 ∧ 56170 + d ≡ 0 [MOD 9] := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1795_179541


namespace NUMINAMATH_CALUDE_polynomial_equality_l1795_179501

theorem polynomial_equality (m n : ℝ) : 
  (∀ x : ℝ, (x + 1) * (2 * x - 3) = 2 * x^2 + m * x + n) → 
  m = -1 ∧ n = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1795_179501


namespace NUMINAMATH_CALUDE_total_blood_cells_l1795_179508

/-- The total number of blood cells in two samples is 7,341, given that the first sample contains 4,221 blood cells and the second sample contains 3,120 blood cells. -/
theorem total_blood_cells (sample1 : Nat) (sample2 : Nat)
  (h1 : sample1 = 4221)
  (h2 : sample2 = 3120) :
  sample1 + sample2 = 7341 := by
  sorry

end NUMINAMATH_CALUDE_total_blood_cells_l1795_179508


namespace NUMINAMATH_CALUDE_least_upper_bound_inequality_inequality_holds_l1795_179515

theorem least_upper_bound_inequality (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ N : ℝ, (∀ x y z : ℝ, 
    |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) →
  N ≥ M :=
by sorry

theorem inequality_holds (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
by sorry

end NUMINAMATH_CALUDE_least_upper_bound_inequality_inequality_holds_l1795_179515


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1795_179571

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 52 →
  a * d + b * c = 83 →
  c * d = 42 →
  a^2 + b^2 + c^2 + d^2 ≤ 38 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1795_179571


namespace NUMINAMATH_CALUDE_correct_result_l1795_179565

theorem correct_result (x : ℝ) (h : x / 3 = 45) : 3 * x = 405 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l1795_179565


namespace NUMINAMATH_CALUDE_vasya_tolya_winning_strategy_l1795_179592

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player
| Tolya : Player

/-- Represents a cell on the board -/
structure Cell :=
(index : Nat)

/-- Represents the game board -/
structure Board :=
(size : Nat)
(boundary_cells : Nat)

/-- Represents the game state -/
structure GameState :=
(board : Board)
(painted_cells : List Cell)
(current_player : Player)

/-- Checks if two cells are adjacent -/
def are_adjacent (c1 c2 : Cell) (board : Board) : Prop :=
  (c1.index + 1) % board.boundary_cells = c2.index ∨
  (c2.index + 1) % board.boundary_cells = c1.index

/-- Checks if two cells are symmetrical with respect to the board center -/
def are_symmetrical (c1 c2 : Cell) (board : Board) : Prop :=
  (c1.index + board.boundary_cells / 2) % board.boundary_cells = c2.index

/-- Determines if a move is valid -/
def is_valid_move (cell : Cell) (state : GameState) : Prop :=
  cell.index < state.board.boundary_cells ∧
  cell ∉ state.painted_cells ∧
  (∀ c ∈ state.painted_cells, ¬(are_adjacent cell c state.board)) ∧
  (∀ c ∈ state.painted_cells, ¬(are_symmetrical cell c state.board))

/-- Theorem: There exists a winning strategy for Vasya and Tolya -/
theorem vasya_tolya_winning_strategy :
  ∃ (strategy : GameState → Cell),
    ∀ (initial_state : GameState),
      initial_state.board.size = 100 ∧
      initial_state.board.boundary_cells = 396 ∧
      initial_state.current_player = Player.Petya →
        ∃ (final_state : GameState),
          final_state.current_player = Player.Petya ∧
          ¬∃ (move : Cell), is_valid_move move final_state :=
sorry

end NUMINAMATH_CALUDE_vasya_tolya_winning_strategy_l1795_179592


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l1795_179530

/-- Given a quadratic function f(x) = 3x^2 + 2x + 4, when shifted 3 units to the left,
    it becomes g(x) = a*x^2 + b*x + c. This theorem proves that a + b + c = 60. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 3*(x+3)^2 + 2*(x+3) + 4 = a*x^2 + b*x + c) → 
  a + b + c = 60 := by
sorry


end NUMINAMATH_CALUDE_quadratic_shift_sum_l1795_179530


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_simplest_form_l1795_179543

theorem subtraction_of_fractions : 
  (9 : ℚ) / 23 - (5 : ℚ) / 69 = (22 : ℚ) / 69 := by
  sorry

theorem simplest_form : 
  ∀ (a b : ℤ), a ≠ 0 → b > 0 → (22 : ℚ) / 69 = (a : ℚ) / b → a = 22 ∧ b = 69 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_simplest_form_l1795_179543


namespace NUMINAMATH_CALUDE_right_angle_point_coordinates_l1795_179567

/-- Given points A, B, and P, where P is on the y-axis and forms a right angle with AB, 
    prove that P has coordinates (0, -11) -/
theorem right_angle_point_coordinates 
  (A B P : ℝ × ℝ)
  (hA : A = (-3, -2))
  (hB : B = (6, 1))
  (hP_y_axis : P.1 = 0)
  (h_right_angle : (P.2 - A.2) * (B.2 - A.2) = -(P.1 - A.1) * (B.1 - A.1)) :
  P = (0, -11) := by
  sorry

end NUMINAMATH_CALUDE_right_angle_point_coordinates_l1795_179567


namespace NUMINAMATH_CALUDE_no_formula_matches_l1795_179562

def x_values : List ℕ := [1, 2, 3, 4, 5]
def y_values : List ℕ := [4, 12, 28, 52, 84]

def formula_a (x : ℕ) : ℕ := 4 * x^2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 3 * x + 1
def formula_c (x : ℕ) : ℕ := 5 * x^3 - 2 * x
def formula_d (x : ℕ) : ℕ := 4 * x^2 + 4 * x

theorem no_formula_matches : 
  ∀ (i : Fin 5), 
    (formula_a (x_values.get i) ≠ y_values.get i) ∧
    (formula_b (x_values.get i) ≠ y_values.get i) ∧
    (formula_c (x_values.get i) ≠ y_values.get i) ∧
    (formula_d (x_values.get i) ≠ y_values.get i) := by
  sorry

end NUMINAMATH_CALUDE_no_formula_matches_l1795_179562


namespace NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l1795_179588

theorem quadratic_inequalities_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) : 
  Set.Ioo (-1/2) (-1/3) = {x : ℝ | b*x^2 - a*x - 1 > 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l1795_179588


namespace NUMINAMATH_CALUDE_no_all_ones_sum_l1795_179542

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

def is_rearrangement (n m : ℕ) : Prop :=
  n.digits 10 ≠ [] ∧ Multiset.ofList (n.digits 10) = Multiset.ofList (m.digits 10)

def all_ones (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1

theorem no_all_ones_sum (N : ℕ) (hN : has_no_zero_digit N) :
  ∀ M : ℕ, is_rearrangement N M → ¬ all_ones (N + M) :=
sorry

end NUMINAMATH_CALUDE_no_all_ones_sum_l1795_179542


namespace NUMINAMATH_CALUDE_coefficient_theorem_l1795_179578

def expression (x : ℝ) : ℝ := 2 * (3 * x - 5) + 5 * (6 - 3 * x^2 + 2 * x) - 9 * (4 * x - 2)

theorem coefficient_theorem :
  ∃ (a b c : ℝ), ∀ x, expression x = a * x^2 + b * x + c ∧ a = -15 ∧ b = -20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_theorem_l1795_179578


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1795_179534

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_inequality : 
  (¬ ∃ x : ℝ, x^2 > 2^x) ↔ (∀ x : ℝ, x^2 ≤ 2^x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1795_179534


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1795_179555

/-- Represents the points on the circle -/
inductive Point : Type
| one | two | three | four | five | six | seven

/-- The next point function, implementing the jumping rules -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.six
  | Point.four => Point.seven
  | Point.five => Point.seven
  | Point.six => Point.two
  | Point.seven => Point.two

/-- Performs n jumps starting from a given point -/
def jumpN (start : Point) (n : ℕ) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpN start n)

/-- The main theorem stating that after 2023 jumps from point 7, the bug lands on point 2 -/
theorem bug_position_after_2023_jumps :
  jumpN Point.seven 2023 = Point.two := by sorry

end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1795_179555


namespace NUMINAMATH_CALUDE_segment_intersection_theorem_l1795_179527

/-- Represents a line in the real plane -/
structure Line where
  -- Add necessary fields

/-- Represents a line segment in the real plane -/
structure Segment where
  -- Add necessary fields

/-- Predicate to check if a line intersects a segment -/
def intersects (l : Line) (s : Segment) : Prop :=
  sorry

/-- Predicate to check if segments are concurrent -/
def concurrent (segments : List Segment) : Prop :=
  sorry

theorem segment_intersection_theorem
  (n : ℕ)
  (segments : List Segment)
  (h_concurrent : concurrent segments)
  (h_count : segments.length = n)
  (h_triple_intersection : ∀ (s1 s2 s3 : Segment),
    s1 ∈ segments → s2 ∈ segments → s3 ∈ segments →
    ∃ (l : Line), intersects l s1 ∧ intersects l s2 ∧ intersects l s3) :
  ∃ (l : Line), ∀ (s : Segment), s ∈ segments → intersects l s :=
sorry

end NUMINAMATH_CALUDE_segment_intersection_theorem_l1795_179527


namespace NUMINAMATH_CALUDE_max_height_formula_l1795_179575

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum possible height of the table formed by right angle folds -/
def max_table_height (t : Triangle) : ℝ := sorry

/-- The specific triangle in the problem -/
def problem_triangle : Triangle := { a := 25, b := 28, c := 31 }

theorem max_height_formula : 
  max_table_height problem_triangle = 42 * Real.sqrt 2582 / 28 := by sorry

end NUMINAMATH_CALUDE_max_height_formula_l1795_179575


namespace NUMINAMATH_CALUDE_value_of_P_closed_under_multiplication_l1795_179528

/-- The polynomial P(x, y) = 2x^2 - 6xy + 5y^2 -/
def P (x y : ℤ) : ℤ := 2*x^2 - 6*x*y + 5*y^2

/-- A number is a value of P if it can be expressed as P(b, c) for some integers b and c -/
def is_value_of_P (a : ℤ) : Prop := ∃ b c : ℤ, P b c = a

/-- If r and s are values of P, then rs is also a value of P -/
theorem value_of_P_closed_under_multiplication (r s : ℤ) 
  (hr : is_value_of_P r) (hs : is_value_of_P s) : 
  is_value_of_P (r * s) := by
  sorry

end NUMINAMATH_CALUDE_value_of_P_closed_under_multiplication_l1795_179528


namespace NUMINAMATH_CALUDE_composition_equation_solution_l1795_179586

theorem composition_equation_solution (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 2 * x + 5) 
  (h2 : ∀ x, φ x = 9 * x + 6) (h3 : δ (φ x) = 3) : x = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l1795_179586


namespace NUMINAMATH_CALUDE_workday_percentage_theorem_l1795_179576

/-- Represents the duration of a workday in minutes -/
def workday_duration : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_duration : ℕ := 2 * first_meeting_duration

/-- Represents the duration of the break in minutes -/
def break_duration : ℕ := 30

/-- Calculates the total time spent in meetings and on break -/
def total_meeting_and_break_time : ℕ :=
  first_meeting_duration + second_meeting_duration + break_duration

/-- Theorem: The percentage of the workday spent in meetings or on break is 35% -/
theorem workday_percentage_theorem :
  (total_meeting_and_break_time : ℚ) / (workday_duration : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_workday_percentage_theorem_l1795_179576


namespace NUMINAMATH_CALUDE_subway_speed_difference_l1795_179593

/-- The speed function of the subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- The theorem stating the existence of the time when the train was 28 km/h slower -/
theorem subway_speed_difference :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ 
  speed 7 - speed t = 28 ∧ 
  t = 5 := by sorry

end NUMINAMATH_CALUDE_subway_speed_difference_l1795_179593


namespace NUMINAMATH_CALUDE_min_value_7x_5y_l1795_179517

theorem min_value_7x_5y (x y : ℕ) 
  (h1 : ∃ k : ℤ, x + 2*y = 5*k)
  (h2 : ∃ m : ℤ, x + y = 3*m)
  (h3 : 2*x + y ≥ 99) :
  7*x + 5*y ≥ 366 := by
sorry

end NUMINAMATH_CALUDE_min_value_7x_5y_l1795_179517


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_is_two_81ths_l1795_179510

/-- A regular tetrahedron with stripes painted on its faces -/
structure StripedTetrahedron where
  /-- The number of faces in a tetrahedron -/
  num_faces : ℕ
  /-- The number of possible stripe orientations per face -/
  orientations_per_face : ℕ
  /-- The total number of possible stripe combinations -/
  total_combinations : ℕ
  /-- The number of favorable outcomes (continuous stripes) -/
  favorable_outcomes : ℕ
  /-- Constraint: num_faces is 4 for a tetrahedron -/
  face_constraint : num_faces = 4
  /-- Constraint: orientations_per_face is 3 -/
  orientation_constraint : orientations_per_face = 3
  /-- Constraint: total_combinations is orientations_per_face^num_faces -/
  combination_constraint : total_combinations = orientations_per_face ^ num_faces
  /-- Constraint: favorable_outcomes is 2 -/
  outcome_constraint : favorable_outcomes = 2

/-- The probability of having a continuous stripe connecting all vertices -/
def continuous_stripe_probability (t : StripedTetrahedron) : ℚ :=
  t.favorable_outcomes / t.total_combinations

/-- Theorem: The probability of a continuous stripe is 2/81 -/
theorem continuous_stripe_probability_is_two_81ths (t : StripedTetrahedron) :
  continuous_stripe_probability t = 2 / 81 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_is_two_81ths_l1795_179510


namespace NUMINAMATH_CALUDE_birthday_spending_l1795_179548

theorem birthday_spending (initial_amount remaining_amount : ℕ) : 
  initial_amount = 7 → remaining_amount = 5 → initial_amount - remaining_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_birthday_spending_l1795_179548


namespace NUMINAMATH_CALUDE_subset_condition_l1795_179537

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1795_179537


namespace NUMINAMATH_CALUDE_range_a_theorem_l1795_179560

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- State the theorem
theorem range_a_theorem : 
  ∀ a : ℝ, (p a ∧ q a) → range_of_a a := by
  sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1795_179560


namespace NUMINAMATH_CALUDE_x_value_l1795_179589

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := 2 * a - b

-- Theorem statement
theorem x_value :
  ∃ x : ℚ, triangle x (triangle 1 3) = 2 ∧ x = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_x_value_l1795_179589


namespace NUMINAMATH_CALUDE_span_equality_iff_multiple_l1795_179525

theorem span_equality_iff_multiple (α₁ β₁ γ₁ α₂ β₂ γ₂ : ℝ) 
  (h₁ : α₁ + β₁ + γ₁ ≠ 0) (h₂ : α₂ + β₂ + γ₂ ≠ 0) :
  Submodule.span ℝ {(α₁, β₁, γ₁)} = Submodule.span ℝ {(α₂, β₂, γ₂)} ↔ 
  ∃ (k : ℝ), k ≠ 0 ∧ (α₁, β₁, γ₁) = (k * α₂, k * β₂, k * γ₂) :=
by sorry

end NUMINAMATH_CALUDE_span_equality_iff_multiple_l1795_179525


namespace NUMINAMATH_CALUDE_surface_area_increase_l1795_179587

/-- The increase in surface area when a cube of edge length a is cut into 27 congruent smaller cubes -/
theorem surface_area_increase (a : ℝ) (h : a > 0) : 
  let original_surface_area := 6 * a^2
  let small_cube_edge := a / 3
  let small_cube_surface_area := 6 * small_cube_edge^2
  let total_new_surface_area := 27 * small_cube_surface_area
  total_new_surface_area - original_surface_area = 12 * a^2 := by
sorry


end NUMINAMATH_CALUDE_surface_area_increase_l1795_179587


namespace NUMINAMATH_CALUDE_widget_production_difference_l1795_179507

/-- Represents the number of widgets produced by David on Tuesday and Wednesday -/
def widget_difference (t : ℝ) : ℝ :=
  let w := 3 * t  -- Tuesday's production rate
  let tuesday_production := w * t
  let wednesday_production := (w + 3) * (t - 3) * 0.9
  tuesday_production - wednesday_production

/-- Theorem stating the difference in widget production between Tuesday and Wednesday -/
theorem widget_production_difference (t : ℝ) :
  widget_difference t = 0.3 * t^2 + 5.4 * t + 8.1 := by
  sorry


end NUMINAMATH_CALUDE_widget_production_difference_l1795_179507


namespace NUMINAMATH_CALUDE_sum_greater_than_8_probability_l1795_179531

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum of two dice is 8 or less -/
def outcomes_8_or_less : ℕ := 26

/-- The probability that the sum of two dice is greater than 8 -/
def prob_sum_greater_than_8 : ℚ := 5 / 18

theorem sum_greater_than_8_probability :
  prob_sum_greater_than_8 = 1 - (outcomes_8_or_less : ℚ) / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_sum_greater_than_8_probability_l1795_179531


namespace NUMINAMATH_CALUDE_sock_drawer_probability_l1795_179594

/-- The total number of socks in the drawer -/
def total_socks : ℕ := 2016

/-- The number of copper socks -/
def copper_socks : ℕ := 2000

/-- The number of colors other than copper -/
def other_colors : ℕ := 8

/-- The number of socks for each color other than copper -/
def socks_per_color : ℕ := 2

/-- The probability of drawing two socks of the same color or one red and one green sock -/
def probability : ℚ := 1999012 / 2031120

theorem sock_drawer_probability :
  (copper_socks.choose 2 + other_colors * socks_per_color.choose 2 + socks_per_color ^ 2) /
  total_socks.choose 2 = probability := by sorry

end NUMINAMATH_CALUDE_sock_drawer_probability_l1795_179594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l1795_179512

/-- Given an arithmetic sequence {a_n} where a_1 + a_7 + a_13 = 4π, 
    prove that tan(a_2 + a_12) = -√3 -/
theorem arithmetic_sequence_tangent (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 1 + a 7 + a 13 = 4 * Real.pi →                  -- given condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l1795_179512


namespace NUMINAMATH_CALUDE_opposite_sign_sum_three_l1795_179563

theorem opposite_sign_sum_three (x y : ℝ) :
  (|x^2 - 4*x + 4| * (2*x - y - 3).sqrt < 0) →
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_three_l1795_179563


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1795_179585

/-- The probability of a license plate containing at least one palindrome -/
theorem license_plate_palindrome_probability :
  let total_arrangements : ℕ := 26^4 * 10^4
  let letter_palindromes : ℕ := 26^2
  let digit_palindromes : ℕ := 10^2
  let both_palindromes : ℕ := letter_palindromes * digit_palindromes
  let palindrome_probability : ℚ := (letter_palindromes * 10^4 + digit_palindromes * 26^4 - both_palindromes) / total_arrangements
  palindrome_probability = 775 / 67600 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1795_179585


namespace NUMINAMATH_CALUDE_divisible_by_eight_l1795_179526

theorem divisible_by_eight (n : ℕ) : ∃ k : ℤ, 6 * n^2 + 4 * n + (-1)^n * 9 + 7 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l1795_179526


namespace NUMINAMATH_CALUDE_sin_405_degrees_l1795_179509

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l1795_179509


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1795_179597

theorem triangle_angle_measure (angle1 angle2 angle3 angle4 : ℝ) :
  angle1 = 34 →
  angle2 = 53 →
  angle3 = 27 →
  angle1 + angle2 + angle3 + angle4 = 180 →
  angle4 = 114 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1795_179597


namespace NUMINAMATH_CALUDE_hammond_marble_weight_l1795_179582

/-- The weight of Hammond's marble statues and discarded marble -/
structure MarbleStatues where
  first_statue : ℕ
  second_statue : ℕ
  remaining_statues : ℕ
  discarded_marble : ℕ

/-- The initial weight of the marble block -/
def initial_weight (m : MarbleStatues) : ℕ :=
  m.first_statue + m.second_statue + 2 * m.remaining_statues + m.discarded_marble

/-- Theorem stating the initial weight of Hammond's marble block -/
theorem hammond_marble_weight :
  ∃ (m : MarbleStatues),
    m.first_statue = 10 ∧
    m.second_statue = 18 ∧
    m.remaining_statues = 15 ∧
    m.discarded_marble = 22 ∧
    initial_weight m = 80 := by
  sorry

end NUMINAMATH_CALUDE_hammond_marble_weight_l1795_179582


namespace NUMINAMATH_CALUDE_price_reduction_l1795_179581

theorem price_reduction (initial_price : ℝ) (first_reduction : ℝ) : 
  first_reduction > 0 ∧ first_reduction < 100 →
  (1 - first_reduction / 100) * (1 - 0.3) = (1 - 0.475) →
  first_reduction = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l1795_179581


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l1795_179545

/-- The age of an employee when hired, given the rule of 70 and years worked until retirement eligibility -/
def age_when_hired (years_worked : ℕ) : ℕ :=
  70 - years_worked

theorem employee_age_when_hired :
  age_when_hired 19 = 51 := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l1795_179545


namespace NUMINAMATH_CALUDE_degree_of_product_polynomial_l1795_179580

/-- The degree of a polynomial (x^2+1)^5 * (x^3+1)^2 * (x+1)^3 -/
theorem degree_of_product_polynomial : ∃ (p : Polynomial ℝ), 
  p = (X^2 + 1)^5 * (X^3 + 1)^2 * (X + 1)^3 ∧ 
  Polynomial.degree p = 19 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_product_polynomial_l1795_179580


namespace NUMINAMATH_CALUDE_range_of_m_l1795_179514

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x > m

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*m*x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1795_179514


namespace NUMINAMATH_CALUDE_power_inequality_l1795_179573

theorem power_inequality (p q a : ℝ) (h1 : p > q) (h2 : q > 1) (h3 : 0 < a) (h4 : a < 1) :
  p ^ a > q ^ a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1795_179573


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1795_179546

-- Define the types for line and plane
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the parallel and perpendicular relations
variable (parallel : L → P → Prop)
variable (perpendicular : L → P → Prop)
variable (plane_perpendicular : P → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : L) (α β : P) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1795_179546


namespace NUMINAMATH_CALUDE_equivalence_conditions_l1795_179544

theorem equivalence_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x < y) ↔ (1 / x > 1 / y) ∧ (x - y < Real.cos x - Real.cos y) ∧ (Real.exp x - Real.exp y < x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_conditions_l1795_179544


namespace NUMINAMATH_CALUDE_division_simplification_l1795_179524

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  4 * x^4 * y^2 / (-2 * x * y) = -2 * x^3 * y :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l1795_179524


namespace NUMINAMATH_CALUDE_maximum_marks_l1795_179568

theorem maximum_marks (victor_marks : ℕ) (max_marks : ℕ) (h1 : victor_marks = 368) (h2 : 92 * max_marks = 100 * victor_marks) : max_marks = 400 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_l1795_179568


namespace NUMINAMATH_CALUDE_sum_of_s_r_at_points_l1795_179506

def r (x : ℝ) : ℝ := |x| + 3

def s (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_s_r_at_points :
  (evaluation_points.map (λ x => s (r x))).sum = -63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_s_r_at_points_l1795_179506


namespace NUMINAMATH_CALUDE_symmetry_axis_shifted_even_function_l1795_179536

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to have an axis of symmetry
def has_axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_axis_shifted_even_function :
  is_even (λ x => f (x + 2)) → has_axis_of_symmetry f 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_shifted_even_function_l1795_179536


namespace NUMINAMATH_CALUDE_total_handshakes_l1795_179532

/-- The number of teams in the tournament -/
def num_teams : ℕ := 3

/-- The number of players in each team -/
def players_per_team : ℕ := 4

/-- The number of referees -/
def num_referees : ℕ := 3

/-- The number of coaches -/
def num_coaches : ℕ := 1

/-- The total number of players -/
def total_players : ℕ := num_teams * players_per_team

/-- The number of officials (referees + coaches) -/
def num_officials : ℕ := num_referees + num_coaches

/-- Theorem stating the total number of handshakes in the tournament -/
theorem total_handshakes : 
  (num_teams * players_per_team * (num_teams - 1) * players_per_team) / 2 + 
  (total_players * num_officials) = 144 := by
  sorry

#eval (num_teams * players_per_team * (num_teams - 1) * players_per_team) / 2 + 
      (total_players * num_officials)

end NUMINAMATH_CALUDE_total_handshakes_l1795_179532


namespace NUMINAMATH_CALUDE_sqrt_3a_plus_2b_l1795_179511

theorem sqrt_3a_plus_2b (a b : ℝ) 
  (h1 : (2*a + 3)^2 = 3^2) 
  (h2 : (5*a + 2*b - 1)^2 = 4^2) : 
  (3*a + 2*b)^2 = 4^2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_3a_plus_2b_l1795_179511


namespace NUMINAMATH_CALUDE_student_number_problem_l1795_179522

theorem student_number_problem (x y : ℝ) : 
  3 * x - y = 110 → x = 110 → y = 220 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1795_179522


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l1795_179505

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 85 ∧ Nat.gcd 36 n = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l1795_179505


namespace NUMINAMATH_CALUDE_remainder_7459_div_9_l1795_179574

theorem remainder_7459_div_9 : 
  7459 % 9 = (7 + 4 + 5 + 9) % 9 := by sorry

end NUMINAMATH_CALUDE_remainder_7459_div_9_l1795_179574


namespace NUMINAMATH_CALUDE_custom_calculator_results_l1795_179547

/-- A custom operation that satisfies specific properties -/
noncomputable def customOp (a b : ℕ) : ℕ :=
  sorry

/-- Addition operation -/
def add : ℕ → ℕ → ℕ := (·+·)

axiom custom_op_self (a : ℕ) : customOp a a = a

axiom custom_op_zero (a : ℕ) : customOp a 0 = 2 * a

axiom custom_op_distributive (a b c d : ℕ) :
  add (customOp a b) (customOp c d) = add (customOp a c) (customOp b d)

theorem custom_calculator_results :
  (customOp (add 2 3) (add 0 3) = 7) ∧
  (customOp 1024 48 = 2000) := by
  sorry

end NUMINAMATH_CALUDE_custom_calculator_results_l1795_179547


namespace NUMINAMATH_CALUDE_train_distance_difference_l1795_179513

/-- Represents the distance traveled by a train given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the difference in distance traveled by two trains -/
theorem train_distance_difference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (total_distance : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 25) 
  (h3 : total_distance = 585) :
  ∃ (time : ℝ), 
    distance speed1 time + distance speed2 time = total_distance ∧ 
    distance speed2 time - distance speed1 time = 65 := by
  sorry

#check train_distance_difference

end NUMINAMATH_CALUDE_train_distance_difference_l1795_179513


namespace NUMINAMATH_CALUDE_overtake_scenario_l1795_179583

/-- Represents the scenario where three people travel at different speeds and overtake each other -/
structure TravelScenario where
  speed_a : ℝ
  speed_b : ℝ
  speed_k : ℝ
  b_delay : ℝ
  overtake_time : ℝ
  k_start_time : ℝ

/-- The theorem statement based on the given problem -/
theorem overtake_scenario (s : TravelScenario) 
  (h1 : s.speed_a = 30)
  (h2 : s.speed_b = 40)
  (h3 : s.speed_k = 60)
  (h4 : s.b_delay = 5)
  (h5 : s.speed_a * s.overtake_time = s.speed_b * (s.overtake_time - s.b_delay))
  (h6 : s.speed_a * s.overtake_time = s.speed_k * s.k_start_time) :
  s.k_start_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_overtake_scenario_l1795_179583


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1795_179521

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    S_n = (1/2)3^(n+1) - a, prove that a = 3/2 -/
theorem geometric_sequence_sum (n : ℕ) (a_n : ℕ → ℝ) (S : ℕ → ℝ) (a : ℝ) :
  (∀ k, S k = (1/2) * 3^(k+1) - a) →
  (∀ k, a_n (k+1) = S (k+1) - S k) →
  (∀ k, a_n (k+2) * a_n k = (a_n (k+1))^2) →
  a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1795_179521


namespace NUMINAMATH_CALUDE_solution_volume_l1795_179564

theorem solution_volume (V : ℝ) : 
  (0.20 * V + 0.60 * 4 = 0.36 * (V + 4)) → V = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l1795_179564


namespace NUMINAMATH_CALUDE_special_sequence_sum_5_l1795_179569

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  roots_property : a 2 * a 4 = 3 ∧ a 2 + a 4 = 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : SpecialArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem: S_5 = 5/2 for the special arithmetic sequence -/
theorem special_sequence_sum_5 (seq : SpecialArithmeticSequence) : 
  sum_n seq 5 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_5_l1795_179569


namespace NUMINAMATH_CALUDE_sequence_property_l1795_179503

/-- Given an arithmetic sequence {aₙ} and a geometric sequence {bₙ} where
    a₃ = b₃ = a, a₆ = b₆ = b, and a > b, prove that if (a₄-b₄)(a₅-b₅) < 0, then ab < 0. -/
theorem sequence_property (a b : ℝ) (aₙ : ℕ → ℝ) (bₙ : ℕ → ℝ) 
    (h_arithmetic : ∀ n : ℕ, aₙ (n + 1) - aₙ n = aₙ 2 - aₙ 1)
    (h_geometric : ∀ n : ℕ, bₙ (n + 1) / bₙ n = bₙ 2 / bₙ 1)
    (h_a3 : aₙ 3 = a) (h_b3 : bₙ 3 = a)
    (h_a6 : aₙ 6 = b) (h_b6 : bₙ 6 = b)
    (h_a_gt_b : a > b) :
  (aₙ 4 - bₙ 4) * (aₙ 5 - bₙ 5) < 0 → a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1795_179503


namespace NUMINAMATH_CALUDE_younger_brother_age_l1795_179554

theorem younger_brother_age (x y : ℕ) 
  (h1 : x + y = 46) 
  (h2 : y = x / 3 + 10) : 
  y = 19 := by
  sorry

end NUMINAMATH_CALUDE_younger_brother_age_l1795_179554


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1795_179598

-- Define the sections of the trip
def section1_distance : ℝ := 600
def section1_speed : ℝ := 30
def section2_distance : ℝ := 300
def section2_speed : ℝ := 15
def section3_distance : ℝ := 500
def section3_speed : ℝ := 25
def section4_distance : ℝ := 400
def section4_speed : ℝ := 40

-- Define the total distance
def total_distance : ℝ := section1_distance + section2_distance + section3_distance + section4_distance

-- Theorem statement
theorem average_speed_calculation :
  let time1 := section1_distance / section1_speed
  let time2 := section2_distance / section2_speed
  let time3 := section3_distance / section3_speed
  let time4 := section4_distance / section4_speed
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  abs (average_speed - 25.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1795_179598


namespace NUMINAMATH_CALUDE_min_value_of_g_l1795_179502

/-- The function f as defined in the problem -/
def f (x₁ x₂ x₃ : ℝ) : ℝ :=
  -2 * (x₁^3 + x₂^3 + x₃^3) + 3 * (x₁^2*(x₂ + x₃) + x₂^2*(x₁ + x₃) + x₃^2*(x₁ + x₂)) - 12*x₁*x₂*x₃

/-- The function g as defined in the problem -/
noncomputable def g (r s t : ℝ) : ℝ :=
  ⨆ (x₃ : ℝ) (h : t ≤ x₃ ∧ x₃ ≤ t + 2), |f r (r + 2) x₃ + s|

/-- The main theorem stating the minimum value of g -/
theorem min_value_of_g :
  (∀ r s t : ℝ, g r s t ≥ 12 * Real.sqrt 3) ∧
  (∃ r₀ s₀ t₀ : ℝ, g r₀ s₀ t₀ = 12 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l1795_179502


namespace NUMINAMATH_CALUDE_eight_chairs_bought_l1795_179540

/-- Represents the chair purchase scenario at Big Lots --/
structure ChairPurchase where
  normalPrice : ℝ
  initialDiscount : ℝ
  additionalDiscount : ℝ
  totalCost : ℝ
  minChairsForAdditionalDiscount : ℕ

/-- Calculates the number of chairs bought given the purchase conditions --/
def calculateChairsBought (purchase : ChairPurchase) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, 8 chairs were bought --/
theorem eight_chairs_bought : 
  let purchase : ChairPurchase := {
    normalPrice := 20,
    initialDiscount := 0.25,
    additionalDiscount := 1/3,
    totalCost := 105,
    minChairsForAdditionalDiscount := 5
  }
  calculateChairsBought purchase = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_chairs_bought_l1795_179540


namespace NUMINAMATH_CALUDE_taras_birthday_money_l1795_179559

theorem taras_birthday_money (P : ℝ) : P * 1.1 = 99 → P = 90 := by
  sorry

end NUMINAMATH_CALUDE_taras_birthday_money_l1795_179559


namespace NUMINAMATH_CALUDE_positive_sqrt_1024_l1795_179539

theorem positive_sqrt_1024 : Real.sqrt 1024 = 32 := by sorry

end NUMINAMATH_CALUDE_positive_sqrt_1024_l1795_179539


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1795_179549

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = -1 ∧
  (f a x₁ = (a + 1) * x₁) ∧
  (f a x₂ = (a + 1) * x₂) ∧
  (∀ x : ℝ, f a x = (a + 1) * x → x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1795_179549


namespace NUMINAMATH_CALUDE_root_value_theorem_l1795_179596

theorem root_value_theorem (m : ℝ) : m^2 - 6*m - 5 = 0 → 11 + 6*m - m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1795_179596


namespace NUMINAMATH_CALUDE_game_question_count_l1795_179550

theorem game_question_count (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3)
  : ∃ (correct_answers : ℕ), 
    correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty ∧ 
    correct_answers = 15 := by
  sorry

end NUMINAMATH_CALUDE_game_question_count_l1795_179550


namespace NUMINAMATH_CALUDE_wanda_crayon_count_l1795_179572

/-- The number of crayons Wanda, Dina, and Jacob have. -/
structure CrayonCount where
  wanda : ℕ
  dina : ℕ
  jacob : ℕ

/-- The given conditions for the crayon problem. -/
def crayon_problem (c : CrayonCount) : Prop :=
  c.dina = 28 ∧
  c.jacob = c.dina - 2 ∧
  c.wanda + c.dina + c.jacob = 116

/-- Theorem stating that Wanda has 62 crayons given the conditions. -/
theorem wanda_crayon_count (c : CrayonCount) (h : crayon_problem c) : c.wanda = 62 := by
  sorry

end NUMINAMATH_CALUDE_wanda_crayon_count_l1795_179572


namespace NUMINAMATH_CALUDE_volunteer_arrangements_l1795_179557

def num_applicants : ℕ := 5
def num_selected : ℕ := 3
def num_events : ℕ := 3

def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

theorem volunteer_arrangements : 
  permutations num_applicants num_selected - 
  permutations (num_applicants - 1) (num_selected - 1) = 48 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangements_l1795_179557


namespace NUMINAMATH_CALUDE_ali_seashells_to_friends_l1795_179595

/-- The number of seashells Ali gave to his friends -/
def seashells_to_friends (initial : ℕ) (to_brothers : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_brothers - 2 * remaining

theorem ali_seashells_to_friends :
  seashells_to_friends 180 30 55 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ali_seashells_to_friends_l1795_179595


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_l1795_179584

theorem reciprocal_sum_equality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (1 / x + 1 / y = 1 / z) → z = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_l1795_179584


namespace NUMINAMATH_CALUDE_mahesh_estimate_less_than_true_value_l1795_179500

theorem mahesh_estimate_less_than_true_value 
  (a b d : ℕ) 
  (h1 : a > b) 
  (h2 : d > 0) : 
  (a - d)^2 - (b + d)^2 < a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_mahesh_estimate_less_than_true_value_l1795_179500


namespace NUMINAMATH_CALUDE_sqrt_of_neg_nine_l1795_179570

-- Define the square root of a complex number
def complex_sqrt (z : ℂ) : Set ℂ :=
  {w : ℂ | w^2 = z}

-- Theorem statement
theorem sqrt_of_neg_nine :
  complex_sqrt (-9 : ℂ) = {3*I, -3*I} :=
sorry

end NUMINAMATH_CALUDE_sqrt_of_neg_nine_l1795_179570


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l1795_179518

theorem gcd_of_polynomial_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) :
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l1795_179518


namespace NUMINAMATH_CALUDE_grid_cut_into_L_shapes_l1795_179591

/-- An L-shaped piece is a shape formed by three squares in an L configuration -/
def LShape : Type := Unit

/-- A grid is a collection of squares arranged in rows and columns -/
def Grid (m n : ℕ) : Type := Fin m → Fin n → Bool

/-- A function that checks if a grid can be cut into L-shaped pieces -/
def can_be_cut_into_L_shapes (g : Grid m n) : Prop := sorry

/-- Main theorem: Any (3n+1) × (3n+1) grid with one square removed can be cut into L-shaped pieces -/
theorem grid_cut_into_L_shapes (n : ℕ) (h : n > 0) :
  ∀ (g : Grid (3*n+1) (3*n+1)), (∃ (i j : Fin (3*n+1)), ¬g i j) →
  can_be_cut_into_L_shapes g :=
sorry

end NUMINAMATH_CALUDE_grid_cut_into_L_shapes_l1795_179591


namespace NUMINAMATH_CALUDE_no_linear_term_implies_p_value_l1795_179519

theorem no_linear_term_implies_p_value (p : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x - 3) * (x^2 + p*x - 1) = a*x^3 + b*x^2 + c) → 
  p = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_p_value_l1795_179519


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1795_179553

theorem product_of_square_roots (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (8 * q^5) = 20 * q^4 * Real.sqrt (3 * q) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1795_179553


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_f_equals_one_condition_l1795_179538

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part I
theorem solution_set_f_geq_2 :
  {x : ℝ | f (x + 2) ≥ 2} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 1/2} := by sorry

-- Theorem for part II
theorem min_value_f :
  ∀ x : ℝ, f x ≥ 1 := by sorry

-- Theorem for the condition when f(x) = 1
theorem f_equals_one_condition (x : ℝ) :
  f x = 1 ↔ 1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_f_equals_one_condition_l1795_179538
