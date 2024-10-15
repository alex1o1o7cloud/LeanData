import Mathlib

namespace NUMINAMATH_CALUDE_other_number_difference_l3325_332513

theorem other_number_difference (x : ℕ) (h1 : x + 42 = 96) : x = 54 := by
  sorry

#check other_number_difference

end NUMINAMATH_CALUDE_other_number_difference_l3325_332513


namespace NUMINAMATH_CALUDE_right_triangle_area_l3325_332568

/-- Right triangle XYZ with altitude foot W -/
structure RightTriangle where
  -- Point X
  X : ℝ × ℝ
  -- Point Y (right angle)
  Y : ℝ × ℝ
  -- Point Z
  Z : ℝ × ℝ
  -- Point W (foot of altitude from Y to XZ)
  W : ℝ × ℝ
  -- XW length
  xw_length : ℝ
  -- WZ length
  wz_length : ℝ
  -- Constraint: XYZ is a right triangle with right angle at Y
  right_angle_at_Y : (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0
  -- Constraint: W is on XZ
  w_on_xz : ∃ t : ℝ, W = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)
  -- Constraint: YW is perpendicular to XZ
  yw_perpendicular_xz : (Y.1 - W.1) * (X.1 - Z.1) + (Y.2 - W.2) * (X.2 - Z.2) = 0
  -- Constraint: XW length is 5
  xw_is_5 : xw_length = 5
  -- Constraint: WZ length is 7
  wz_is_7 : wz_length = 7

/-- The area of the right triangle XYZ -/
def triangleArea (t : RightTriangle) : ℝ := sorry

/-- Theorem: The area of the right triangle XYZ is 6√35 -/
theorem right_triangle_area (t : RightTriangle) : triangleArea t = 6 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3325_332568


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l3325_332535

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

#eval num_diagonals 30  -- This should output 405

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l3325_332535


namespace NUMINAMATH_CALUDE_sufficient_condition_exclusive_or_condition_l3325_332562

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Part 1: p is a sufficient condition for q
theorem sufficient_condition (m : ℝ) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 :=
sorry

-- Part 2: m = 5, "p or q" is true, "p and q" is false
theorem exclusive_or_condition (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → x ∈ Set.Icc (-4) (-1) ∪ Set.Ioc 5 6 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_exclusive_or_condition_l3325_332562


namespace NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_3_l3325_332563

theorem slope_angle_45_implies_a_equals_3 (a : ℝ) :
  (∃ (x y : ℝ), (a - 2) * x - y + 3 = 0 ∧ 
   Real.arctan ((a - 2) : ℝ) = π / 4) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_3_l3325_332563


namespace NUMINAMATH_CALUDE_periodic_placement_exists_l3325_332531

/-- A function that maps integer coordinates to natural numbers -/
def f : ℤ × ℤ → ℕ := sorry

/-- Theorem stating the existence of a function satisfying the required properties -/
theorem periodic_placement_exists : 
  (∀ n : ℕ, ∃ x y : ℤ, f (x, y) = n) ∧ 
  (∀ a b c : ℤ, a ≠ 0 ∨ b ≠ 0 → c ≠ 0 → 
    ∃ k m : ℤ, ∀ x y : ℤ, a * x + b * y = c → 
      f (x + k, y + m) = f (x, y)) :=
by sorry

end NUMINAMATH_CALUDE_periodic_placement_exists_l3325_332531


namespace NUMINAMATH_CALUDE_roots_sum_bound_l3325_332589

theorem roots_sum_bound (v w : ℂ) : 
  v ≠ w → 
  v^2021 = 1 → 
  w^2021 = 1 → 
  Complex.abs (v + w) < Real.sqrt (2 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_bound_l3325_332589


namespace NUMINAMATH_CALUDE_mean_problem_l3325_332572

theorem mean_problem (x y : ℝ) : 
  (6 + 14 + x + 17 + 9 + y + 10) / 7 = 13 → x + y = 35 := by
  sorry

end NUMINAMATH_CALUDE_mean_problem_l3325_332572


namespace NUMINAMATH_CALUDE_carols_piggy_bank_l3325_332559

/-- Represents the contents of Carol's piggy bank -/
structure PiggyBank where
  nickels : ℕ
  dimes : ℕ

/-- The value of the piggy bank in cents -/
def bankValue (bank : PiggyBank) : ℕ :=
  5 * bank.nickels + 10 * bank.dimes

theorem carols_piggy_bank :
  ∃ (bank : PiggyBank),
    bankValue bank = 455 ∧
    bank.nickels = bank.dimes + 7 ∧
    bank.nickels = 35 := by
  sorry

end NUMINAMATH_CALUDE_carols_piggy_bank_l3325_332559


namespace NUMINAMATH_CALUDE_ice_cream_box_cost_l3325_332509

/-- Represents the cost of a box of ice cream bars -/
def box_cost : ℚ := sorry

/-- Number of ice cream bars in a box -/
def bars_per_box : ℕ := 3

/-- Number of friends -/
def num_friends : ℕ := 6

/-- Number of bars each friend wants to eat -/
def bars_per_friend : ℕ := 2

/-- Cost per person -/
def cost_per_person : ℚ := 5

theorem ice_cream_box_cost :
  box_cost = 7.5 := by sorry

end NUMINAMATH_CALUDE_ice_cream_box_cost_l3325_332509


namespace NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l3325_332528

theorem divisibility_implies_gcd_greater_than_one
  (a b x y : ℕ)
  (h : (a^2 + b^2) ∣ (a*x + b*y)) :
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_gcd_greater_than_one_l3325_332528


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3325_332525

theorem arithmetic_calculation : 2011 - (9 * 11 * 11 + 9 * 9 * 11 - 9 * 11) = 130 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3325_332525


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l3325_332529

-- Define the sequence (a_n)
def a : ℕ → ℤ
  | 0 => sorry  -- We don't know the initial value, so we use sorry
  | n + 1 => (a n)^3 + 1999

-- Define what it means for an integer to be a perfect square
def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k^2

-- State the theorem
theorem at_most_one_perfect_square :
  ∃! n : ℕ, is_perfect_square (a n) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l3325_332529


namespace NUMINAMATH_CALUDE_division_value_problem_l3325_332569

theorem division_value_problem (x : ℝ) : 
  (740 / x) - 175 = 10 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l3325_332569


namespace NUMINAMATH_CALUDE_gel_pen_price_ratio_l3325_332537

theorem gel_pen_price_ratio (x y : ℕ) (b g : ℝ) :
  x > 0 ∧ y > 0 ∧ b > 0 ∧ g > 0 →
  (x + y) * g = 4 * (x * b + y * g) →
  (x + y) * b = (1 / 2) * (x * b + y * g) →
  g = 8 * b :=
by
  sorry

end NUMINAMATH_CALUDE_gel_pen_price_ratio_l3325_332537


namespace NUMINAMATH_CALUDE_parabola_sum_l3325_332516

-- Define a quadratic function
def quadratic (p q r : ℝ) : ℝ → ℝ := λ x => p * x^2 + q * x + r

theorem parabola_sum (p q r : ℝ) :
  -- The vertex of the parabola is (3, -1)
  (∀ x, quadratic p q r x ≥ quadratic p q r 3) ∧
  quadratic p q r 3 = -1 ∧
  -- The parabola passes through the point (0, 8)
  quadratic p q r 0 = 8
  →
  p + q + r = 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l3325_332516


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3325_332552

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + m*y + 6 = 0 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3325_332552


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l3325_332502

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ (a^1984 + b^1984 + c^1984 + d^1984 = m * n) := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l3325_332502


namespace NUMINAMATH_CALUDE_printer_problem_l3325_332597

/-- Calculates the time needed to print a given number of pages at a specific rate -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  (pages : ℚ) / (rate : ℚ)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem printer_problem : 
  let pages : ℕ := 300
  let rate : ℕ := 20
  round_to_nearest (print_time pages rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_printer_problem_l3325_332597


namespace NUMINAMATH_CALUDE_correct_ticket_sales_l3325_332548

/-- A structure representing ticket sales for different movie genres --/
structure TicketSales where
  romance : ℕ
  horror : ℕ
  action : ℕ
  comedy : ℕ

/-- Definition of valid ticket sales based on the given conditions --/
def is_valid_ticket_sales (t : TicketSales) : Prop :=
  t.romance = 25 ∧
  t.horror = 3 * t.romance + 18 ∧
  t.action = 2 * t.romance ∧
  5 * t.comedy = 4 * t.horror

/-- Theorem stating the correct number of tickets sold for each genre --/
theorem correct_ticket_sales :
  ∃ (t : TicketSales), is_valid_ticket_sales t ∧
    t.horror = 93 ∧ t.action = 50 ∧ t.comedy = 74 := by
  sorry

end NUMINAMATH_CALUDE_correct_ticket_sales_l3325_332548


namespace NUMINAMATH_CALUDE_sin_neg_five_pi_sixths_l3325_332504

theorem sin_neg_five_pi_sixths : Real.sin (-5 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_five_pi_sixths_l3325_332504


namespace NUMINAMATH_CALUDE_initial_solution_volume_l3325_332592

theorem initial_solution_volume 
  (V : ℝ)  -- Initial volume in liters
  (h1 : 0.20 * V + 3.6 = 0.50 * (V + 3.6))  -- Equation representing the alcohol balance
  : V = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l3325_332592


namespace NUMINAMATH_CALUDE_largest_odd_integer_sum_30_l3325_332519

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def consecutive_odd_integers (m : ℕ) : List ℕ := [m - 4, m - 2, m, m + 2, m + 4]

theorem largest_odd_integer_sum_30 :
  ∃ m : ℕ, 
    (sum_first_n_odd 30 = (consecutive_odd_integers m).sum) ∧
    (List.maximum (consecutive_odd_integers m) = some 184) := by
  sorry

end NUMINAMATH_CALUDE_largest_odd_integer_sum_30_l3325_332519


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l3325_332518

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 / 4) * (p + q + r + s) = 15) : 
  (p + q + r + s) / 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l3325_332518


namespace NUMINAMATH_CALUDE_root_implies_q_equals_four_l3325_332543

theorem root_implies_q_equals_four (p q : ℝ) : 
  (Complex.I * Real.sqrt 3 + 1) ^ 2 + p * (Complex.I * Real.sqrt 3 + 1) + q = 0 → q = 4 := by
sorry

end NUMINAMATH_CALUDE_root_implies_q_equals_four_l3325_332543


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l3325_332567

/-- Represents a geometric figure on a grid paper -/
structure GridFigure where
  -- Add necessary fields to represent the figure

/-- Represents a part of the figure after cutting -/
structure FigurePart where
  -- Add necessary fields to represent a part

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to check if a list of parts can be reassembled into a square -/
def can_form_square (parts : List FigurePart) : Bool :=
  sorry

/-- Function to check if all parts are triangles -/
def all_triangles (parts : List FigurePart) : Bool :=
  sorry

/-- Theorem stating that the figure can be cut and reassembled into a square under given conditions -/
theorem figure_to_square_possible (fig : GridFigure) : 
  (∃ (parts : List FigurePart), parts.length ≤ 4 ∧ can_form_square parts) ∧
  (∃ (parts : List FigurePart), parts.length ≤ 5 ∧ all_triangles parts ∧ can_form_square parts) :=
by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l3325_332567


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3325_332578

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3325_332578


namespace NUMINAMATH_CALUDE_q_value_at_two_l3325_332532

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define q(x) as p(p(x))
def q (x : ℝ) : ℝ := p (p x)

-- Theorem statement
theorem q_value_at_two (h : ∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ q a = 0 ∧ q b = 0 ∧ q c = 0) :
  q 2 = -1 := by
  sorry


end NUMINAMATH_CALUDE_q_value_at_two_l3325_332532


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3325_332544

theorem cube_surface_area_from_volume (V : ℝ) (s : ℝ) (SA : ℝ) : 
  V = 729 → V = s^3 → SA = 6 * s^2 → SA = 486 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3325_332544


namespace NUMINAMATH_CALUDE_carlos_blocks_given_l3325_332582

/-- The number of blocks Carlos gave to Rachel -/
def blocks_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem carlos_blocks_given :
  blocks_given 58 37 = 21 := by
  sorry

end NUMINAMATH_CALUDE_carlos_blocks_given_l3325_332582


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l3325_332547

theorem square_of_linear_expression (x : ℝ) :
  x = -2 → (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l3325_332547


namespace NUMINAMATH_CALUDE_rectangle_matchsticks_distribution_l3325_332588

/-- Calculates the total number of matchsticks in the rectangle -/
def total_matchsticks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Checks if the number of matchsticks can be equally distributed among a given number of children -/
def is_valid_distribution (total_sticks children : ℕ) : Prop :=
  children > 100 ∧ total_sticks % children = 0

theorem rectangle_matchsticks_distribution :
  let total := total_matchsticks 60 10
  ∃ (n : ℕ), is_valid_distribution total n ∧
    ∀ (m : ℕ), m > n → ¬(is_valid_distribution total m) ∧
    n = 127 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_matchsticks_distribution_l3325_332588


namespace NUMINAMATH_CALUDE_book_reading_permutations_l3325_332584

theorem book_reading_permutations :
  let n : ℕ := 5  -- total number of books
  let r : ℕ := 3  -- number of books to read
  Nat.factorial n / Nat.factorial (n - r) = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_permutations_l3325_332584


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l3325_332506

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : IsQuadratic f) 
  (h2 : f 0 = 3) 
  (h3 : ∀ x, f (x + 2) - f x = 4 * x + 2) : 
  ∀ x, f x = x^2 - x + 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_unique_l3325_332506


namespace NUMINAMATH_CALUDE_fraction_equality_l3325_332539

theorem fraction_equality (a b c : ℝ) (h1 : a/3 = b) (h2 : b/4 = c) : a*b/c^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3325_332539


namespace NUMINAMATH_CALUDE_jellybeans_remaining_l3325_332561

/-- Given a jar of jellybeans and a class of students, calculate the remaining jellybeans after some students eat them. -/
theorem jellybeans_remaining (total_jellybeans : ℕ) (total_students : ℕ) (absent_students : ℕ) (jellybeans_per_student : ℕ)
  (h1 : total_jellybeans = 100)
  (h2 : total_students = 24)
  (h3 : absent_students = 2)
  (h4 : jellybeans_per_student = 3) :
  total_jellybeans - (total_students - absent_students) * jellybeans_per_student = 34 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_remaining_l3325_332561


namespace NUMINAMATH_CALUDE_vanya_correct_answers_l3325_332591

/-- The number of questions Sasha asked Vanya -/
def total_questions : ℕ := 50

/-- The number of candies Vanya receives for a correct answer -/
def correct_reward : ℕ := 7

/-- The number of candies Vanya gives for an incorrect answer -/
def incorrect_penalty : ℕ := 3

/-- The number of questions Vanya answered correctly -/
def correct_answers : ℕ := 15

theorem vanya_correct_answers :
  correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty :=
by sorry

end NUMINAMATH_CALUDE_vanya_correct_answers_l3325_332591


namespace NUMINAMATH_CALUDE_find_y_l3325_332523

-- Define the binary operation ⊕
def binary_op (a b c d : ℤ) : ℤ × ℤ := (a + d, b - c)

-- Theorem statement
theorem find_y : ∀ x y : ℤ, 
  binary_op 2 5 1 1 = binary_op x y 2 0 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3325_332523


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_g_nonnegative_l3325_332590

/-- A function that is monotonically decreasing on an interval -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

/-- The function f(x) = x^3 + ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The function g(x) = 3x^2 + 2ax + b -/
def g (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem: If f(x) is monotonically decreasing on (0, 1), then g(0) * g(1) ≥ 0 -/
theorem monotone_decreasing_implies_g_nonnegative (a b c : ℝ) :
  MonotonicallyDecreasing (f a b c) 0 1 → g a b 0 * g a b 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_g_nonnegative_l3325_332590


namespace NUMINAMATH_CALUDE_length_of_GH_l3325_332546

-- Define the lengths of the segments
def AB : ℝ := 11
def CD : ℝ := 5
def FE : ℝ := 13

-- Define the length of GH as the sum of AB, CD, and FE
def GH : ℝ := AB + CD + FE

-- Theorem statement
theorem length_of_GH : GH = 29 := by sorry

end NUMINAMATH_CALUDE_length_of_GH_l3325_332546


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l3325_332579

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 ≠ a2 * c1

/-- The theorem statement -/
theorem parallel_lines_m_equals_one (m : ℝ) :
  parallel_lines 1 (1 + m) (m - 2) (2 * m) 4 16 → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l3325_332579


namespace NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l3325_332560

/-- The area inside a rectangle but outside three quarter circles --/
theorem area_inside_rectangle_outside_circles (CD DA : ℝ) (r₁ r₂ r₃ : ℝ) :
  CD = 3 →
  DA = 5 →
  r₁ = 1 →
  r₂ = 2 →
  r₃ = 3 →
  (CD * DA) - ((π * r₁^2) / 4 + (π * r₂^2) / 4 + (π * r₃^2) / 4) = 15 - (7 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l3325_332560


namespace NUMINAMATH_CALUDE_inequality_subset_l3325_332577

/-- The solution set of the system of inequalities is a subset of 2x^2 - 9x + a < 0 iff a ≤ 9 -/
theorem inequality_subset (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + a < 0) ↔ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_subset_l3325_332577


namespace NUMINAMATH_CALUDE_locus_of_symmetric_point_l3325_332534

/-- Given a parabola y = x^2 and a fixed point A(a, 0) where a ≠ 0, 
    the locus of point Q symmetric to A with respect to a point on the parabola 
    is described by the equation y = (1/2)(x + a)^2 -/
theorem locus_of_symmetric_point (a : ℝ) (ha : a ≠ 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ (x y : ℝ), (y = x^2) → 
      ∃ (qx qy : ℝ), 
        (qx + x = 2 * a ∧ qy + y = 0) → 
        f qx = qy ∧ f qx = (1/2) * (qx + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_symmetric_point_l3325_332534


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3325_332599

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  Odd a ∧ Odd b ∧ Odd c ∧  -- a, b, c are odd
  b = a + 2 ∧ c = b + 2 ∧   -- consecutive with difference 2
  a + b + c = 75            -- sum is 75
  → c = 27 := by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3325_332599


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3325_332514

theorem simplify_complex_fraction (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) :
  1 - (1 / (1 + a^2 / (1 - a^2))) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3325_332514


namespace NUMINAMATH_CALUDE_thirteen_students_in_line_l3325_332553

/-- The number of students in a line, given specific positions of Taehyung and Namjoon -/
def students_in_line (people_between_taehyung_and_namjoon : ℕ) (people_behind_namjoon : ℕ) : ℕ :=
  1 + people_between_taehyung_and_namjoon + 1 + people_behind_namjoon

/-- Theorem stating that there are 13 students in the line -/
theorem thirteen_students_in_line : 
  students_in_line 3 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_students_in_line_l3325_332553


namespace NUMINAMATH_CALUDE_largest_prime_divisor_check_l3325_332573

theorem largest_prime_divisor_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≤ 31 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_check_l3325_332573


namespace NUMINAMATH_CALUDE_cube_split_theorem_l3325_332541

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) : 
  m^2 - m + 1 = 73 → m = 9 :=
by
  sorry

#check cube_split_theorem

end NUMINAMATH_CALUDE_cube_split_theorem_l3325_332541


namespace NUMINAMATH_CALUDE_stream_speed_l3325_332581

/-- Given a boat's speed in still water and its downstream travel time and distance,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ) :
  boat_speed = 24 →
  downstream_time = 7 →
  downstream_distance = 196 →
  ∃ stream_speed : ℝ,
    stream_speed = 4 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l3325_332581


namespace NUMINAMATH_CALUDE_diagonals_not_parallel_in_32gon_l3325_332558

/-- The number of sides in the regular polygon -/
def n : ℕ := 32

/-- The total number of diagonals in an n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of pairs of parallel sides in an n-sided polygon -/
def parallel_side_pairs (n : ℕ) : ℕ := n / 2

/-- The number of diagonals parallel to one pair of sides -/
def diagonals_per_parallel_pair (n : ℕ) : ℕ := (n - 4) / 2

/-- The total number of parallel diagonals -/
def total_parallel_diagonals (n : ℕ) : ℕ :=
  parallel_side_pairs n * diagonals_per_parallel_pair n

/-- The number of diagonals not parallel to any side in a regular 32-gon -/
theorem diagonals_not_parallel_in_32gon :
  total_diagonals n - total_parallel_diagonals n = 240 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_not_parallel_in_32gon_l3325_332558


namespace NUMINAMATH_CALUDE_no_solution_exists_l3325_332500

theorem no_solution_exists : ¬∃ (m n : ℤ), 
  m ≠ n ∧ 
  988 < m ∧ m < 1991 ∧ 
  988 < n ∧ n < 1991 ∧ 
  ∃ (a : ℤ), m * n + n = a ^ 2 ∧ 
  ∃ (b : ℤ), m * n + m = b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3325_332500


namespace NUMINAMATH_CALUDE_max_d_value_l3325_332508

def a (n : ℕ+) : ℕ := n.val ^ 2 + 1000

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (N : ℕ+), d N = 4001 ∧ ∀ (n : ℕ+), d n ≤ 4001 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3325_332508


namespace NUMINAMATH_CALUDE_log_stack_sum_l3325_332576

theorem log_stack_sum : ∀ (a l n : ℕ), 
  a = 15 → l = 4 → n = 12 → 
  (n * (a + l)) / 2 = 114 := by
sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3325_332576


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3325_332557

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, 2h) and y-intercept at (0, -3h), where h ≠ 0, prove that b = 10 -/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + 2 * h) → 
  a * 0^2 + b * 0 + c = -3 * h → 
  b = 10 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3325_332557


namespace NUMINAMATH_CALUDE_candy_count_solution_l3325_332536

def is_valid_candy_count (x : ℕ) : Prop :=
  ∃ (brother_takes : ℕ),
    x % 4 = 0 ∧
    x % 2 = 0 ∧
    2 ≤ brother_takes ∧
    brother_takes ≤ 6 ∧
    (x / 4 * 3 / 3 * 2 - 40 - brother_takes = 10)

theorem candy_count_solution :
  ∀ x : ℕ, is_valid_candy_count x ↔ (x = 108 ∨ x = 112) :=
sorry

end NUMINAMATH_CALUDE_candy_count_solution_l3325_332536


namespace NUMINAMATH_CALUDE_b_investment_l3325_332524

/-- Proves that B's investment is Rs. 12000 given the conditions of the problem -/
theorem b_investment (a_investment b_investment c_investment : ℝ)
  (b_profit : ℝ) (profit_difference : ℝ) :
  a_investment = 8000 →
  c_investment = 12000 →
  b_profit = 3000 →
  profit_difference = 1199.9999999999998 →
  (a_investment / b_investment) * b_profit =
    (c_investment / b_investment) * b_profit - profit_difference →
  b_investment = 12000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_l3325_332524


namespace NUMINAMATH_CALUDE_tank_capacity_l3325_332595

/-- Represents the tank system with its properties -/
structure TankSystem where
  capacity : ℝ
  outletA_time : ℝ
  outletB_time : ℝ
  inlet_rate : ℝ
  combined_extra_time : ℝ

/-- The tank system satisfies the given conditions -/
def satisfies_conditions (ts : TankSystem) : Prop :=
  ts.outletA_time = 5 ∧
  ts.outletB_time = 8 ∧
  ts.inlet_rate = 4 * 60 ∧
  ts.combined_extra_time = 3

/-- The theorem stating that the tank capacity is 1200 litres -/
theorem tank_capacity (ts : TankSystem) 
  (h : satisfies_conditions ts) : ts.capacity = 1200 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3325_332595


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l3325_332565

theorem symmetry_implies_sum (a b : ℝ) :
  (∀ x y : ℝ, y = a * x + 8 ↔ x = -1/2 * y + b) →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l3325_332565


namespace NUMINAMATH_CALUDE_notebook_cost_l3325_332594

theorem notebook_cost (total_spent : ℝ) (backpack_cost : ℝ) (pens_cost : ℝ) (pencils_cost : ℝ) (num_notebooks : ℕ) :
  total_spent = 32 →
  backpack_cost = 15 →
  pens_cost = 1 →
  pencils_cost = 1 →
  num_notebooks = 5 →
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks = 3 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3325_332594


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l3325_332527

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- The event of drawing a specific combination of balls -/
structure Event where
  red : ℕ
  white : ℕ

/-- The bag in the problem -/
def problem_bag : Bag := { red := 5, white := 5 }

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- The event of drawing 3 red balls -/
def event_all_red : Event := { red := 3, white := 0 }

/-- The event of drawing at least 1 white ball -/
def event_at_least_one_white : Event := { red := drawn_balls - 1, white := 1 }

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (e1 e2 : Event) : Prop :=
  e1.red + e1.white = drawn_balls ∧ e2.red + e2.white = drawn_balls ∧ 
  (e1.red + e2.red > problem_bag.red ∨ e1.white + e2.white > problem_bag.white)

/-- The main theorem stating that the two events are mutually exclusive -/
theorem events_mutually_exclusive : 
  mutually_exclusive event_all_red event_at_least_one_white :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l3325_332527


namespace NUMINAMATH_CALUDE_book_cart_total_l3325_332538

/-- Represents the number of books in each category on the top section of the cart -/
structure TopSection where
  history : ℕ
  romance : ℕ
  poetry : ℕ

/-- Represents the number of books in each category on the bottom section of the cart -/
structure BottomSection where
  western : ℕ
  biography : ℕ
  scifi : ℕ

/-- Represents the entire book cart -/
structure BookCart where
  top : TopSection
  bottom : BottomSection
  mystery : ℕ

def total_books (cart : BookCart) : ℕ :=
  cart.top.history + cart.top.romance + cart.top.poetry +
  cart.bottom.western + cart.bottom.biography + cart.bottom.scifi +
  cart.mystery

theorem book_cart_total (cart : BookCart) :
  cart.top.history = 12 →
  cart.top.romance = 8 →
  cart.top.poetry = 4 →
  cart.bottom.western = 5 →
  cart.bottom.biography = 6 →
  cart.bottom.scifi = 3 →
  cart.mystery = 2 * (cart.bottom.western + cart.bottom.biography + cart.bottom.scifi) →
  total_books cart = 66 := by
  sorry

#check book_cart_total

end NUMINAMATH_CALUDE_book_cart_total_l3325_332538


namespace NUMINAMATH_CALUDE_range_of_c_l3325_332533

theorem range_of_c (c : ℝ) : 
  (∀ x > 0, c^2 * x^2 - (c * x + 1) * Real.log x + c * x ≥ 0) ↔ c ≥ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l3325_332533


namespace NUMINAMATH_CALUDE_trig_identity_l3325_332554

theorem trig_identity (α : Real) (h : Real.tan α = -1/2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3325_332554


namespace NUMINAMATH_CALUDE_student_number_problem_l3325_332596

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 106 → x = 122 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3325_332596


namespace NUMINAMATH_CALUDE_read_time_is_two_hours_l3325_332583

/-- Calculates the time taken to read a given number of pages at an increased reading speed. -/
def time_to_read (normal_speed : ℕ) (speed_increase : ℕ) (total_pages : ℕ) : ℚ :=
  total_pages / (normal_speed * speed_increase)

/-- Theorem stating that given the conditions from the problem, the time taken to read is 2 hours. -/
theorem read_time_is_two_hours (normal_speed : ℕ) (speed_increase : ℕ) (total_pages : ℕ)
  (h1 : normal_speed = 12)
  (h2 : speed_increase = 3)
  (h3 : total_pages = 72) :
  time_to_read normal_speed speed_increase total_pages = 2 := by
  sorry

end NUMINAMATH_CALUDE_read_time_is_two_hours_l3325_332583


namespace NUMINAMATH_CALUDE_acme_vowel_soup_strings_l3325_332555

/-- Represents the number of times each vowel appears in the soup -/
def vowel_count : Fin 5 → ℕ
  | 0 => 6  -- A
  | 1 => 6  -- E
  | 2 => 6  -- I
  | 3 => 6  -- O
  | 4 => 3  -- U

/-- The length of the strings to be formed -/
def string_length : ℕ := 6

/-- Calculates the number of possible strings -/
def count_strings : ℕ :=
  (Finset.range 4).sum (λ k =>
    (Nat.choose string_length k) * (4 * vowel_count 0) ^ (string_length - k))

theorem acme_vowel_soup_strings :
  count_strings = 117072 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_strings_l3325_332555


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3325_332585

/-- The set T of points in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x - 1 ∧ y + 3 ≤ 5) ∨
               (5 = y + 3 ∧ x - 1 ≤ 5) ∨
               (x - 1 = y + 3 ∧ 5 ≤ x - 1)}

/-- A ray in the coordinate plane -/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The three rays that make up set T -/
def rays : List Ray :=
  [{ start := (6, 2), direction := (0, -1) },   -- Vertical ray downward
   { start := (6, 2), direction := (-1, 0) },   -- Horizontal ray leftward
   { start := (6, 2), direction := (1, 1) }]    -- Diagonal ray upward

/-- Theorem stating that T consists of three rays with a common point -/
theorem T_is_three_rays_with_common_point :
  ∃ (common_point : ℝ × ℝ) (rs : List Ray),
    common_point = (6, 2) ∧
    rs.length = 3 ∧
    (∀ r ∈ rs, r.start = common_point) ∧
    T = ⋃ r ∈ rs, {p | ∃ t ≥ 0, p = r.start + t • r.direction} :=
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3325_332585


namespace NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_l3325_332550

theorem sum_of_real_and_imaginary_parts : ∃ (z : ℂ), z = 3 - 4*I ∧ z.re + z.im = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imaginary_parts_l3325_332550


namespace NUMINAMATH_CALUDE_find_b_l3325_332542

theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 3 * x - 7)
  (hq : ∀ x, q x = 3 * x - b)
  (h : p (q 5) = 11) :
  b = 9 := by sorry

end NUMINAMATH_CALUDE_find_b_l3325_332542


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l3325_332575

/-- Represents an ellipse equation of the form mx^2 + ny^2 = 1 --/
structure EllipseEquation (m n : ℝ) where
  eq : ∀ x y : ℝ, m * x^2 + n * y^2 = 1

/-- Predicate to check if an ellipse has foci on the y-axis --/
def hasFociOnYAxis (m n : ℝ) : Prop :=
  m > n ∧ n > 0

/-- Theorem stating that m > n > 0 is necessary and sufficient for 
    mx^2 + ny^2 = 1 to represent an ellipse with foci on the y-axis --/
theorem ellipse_foci_on_y_axis_iff (m n : ℝ) :
  hasFociOnYAxis m n ↔ ∃ (e : EllipseEquation m n), True :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l3325_332575


namespace NUMINAMATH_CALUDE_blackboard_numbers_l3325_332580

theorem blackboard_numbers (n : ℕ) (h : n = 1987) :
  let S := n * (n + 1) / 2
  let remaining_sum := S % 7
  ∃ x, x ≤ 6 ∧ (x + 987) % 7 = remaining_sum :=
by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l3325_332580


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3325_332517

theorem multiply_and_simplify (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x)^3) = 25 / 8 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3325_332517


namespace NUMINAMATH_CALUDE_bag_problem_l3325_332549

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the probability of drawing at least 1 white ball when drawing 2 balls -/
def prob_at_least_one_white : ℚ := 4/5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Represents the number of white balls in the bag -/
def num_white_balls : ℕ := sorry

/-- Calculates the probability of drawing exactly k white balls when drawing 2 balls -/
def prob_k_white (k : ℕ) : ℚ := sorry

/-- Calculates the mathematical expectation of the number of white balls drawn -/
def expectation : ℚ := sorry

theorem bag_problem :
  (1 - (choose (total_balls - num_white_balls) 2 : ℚ) / (choose total_balls 2 : ℚ) = prob_at_least_one_white) →
  (num_white_balls = 3 ∧ expectation = 1) :=
by sorry

end NUMINAMATH_CALUDE_bag_problem_l3325_332549


namespace NUMINAMATH_CALUDE_prom_expenses_james_prom_expenses_l3325_332540

theorem prom_expenses (num_people : ℕ) (ticket_cost dinner_cost : ℚ) 
  (tip_percentage : ℚ) (limo_hours : ℕ) (limo_cost_per_hour : ℚ) 
  (tuxedo_rental : ℚ) : ℚ :=
  let total_ticket_cost := num_people * ticket_cost
  let total_dinner_cost := num_people * dinner_cost
  let dinner_tip := total_dinner_cost * tip_percentage
  let total_limo_cost := limo_hours * limo_cost_per_hour
  total_ticket_cost + total_dinner_cost + dinner_tip + total_limo_cost + tuxedo_rental

theorem james_prom_expenses : 
  prom_expenses 4 100 120 0.3 8 80 150 = 1814 := by
  sorry

end NUMINAMATH_CALUDE_prom_expenses_james_prom_expenses_l3325_332540


namespace NUMINAMATH_CALUDE_business_investment_l3325_332507

theorem business_investment (A B : ℕ) (t : ℕ) (r : ℚ) : 
  A = 45000 →
  t = 2 →
  r = 2 / 1 →
  (A * t : ℚ) / (B * t : ℚ) = r →
  B = 22500 := by
  sorry

end NUMINAMATH_CALUDE_business_investment_l3325_332507


namespace NUMINAMATH_CALUDE_speed_calculation_l3325_332520

-- Define the given conditions
def field_area : ℝ := 50
def travel_time_minutes : ℝ := 2

-- Define the theorem
theorem speed_calculation :
  let diagonal := Real.sqrt (2 * field_area)
  let speed_m_per_hour := diagonal / (travel_time_minutes / 60)
  speed_m_per_hour / 1000 = 0.3 := by
sorry

end NUMINAMATH_CALUDE_speed_calculation_l3325_332520


namespace NUMINAMATH_CALUDE_margo_walk_distance_l3325_332511

/-- Calculates the total distance walked given the time to walk to a friend's house,
    the time to return, and the average walking rate for the entire trip. -/
def total_distance (time_to_friend : ℚ) (time_from_friend : ℚ) (avg_rate : ℚ) : ℚ :=
  let total_time : ℚ := time_to_friend + time_from_friend
  let total_time_hours : ℚ := total_time / 60
  avg_rate * total_time_hours

/-- Proves that given the specific conditions, the total distance walked is 1.5 miles. -/
theorem margo_walk_distance :
  let time_to_friend : ℚ := 15
  let time_from_friend : ℚ := 10
  let avg_rate : ℚ := 18/5  -- 3.6 as a rational number
  total_distance time_to_friend time_from_friend avg_rate = 3/2 := by
  sorry

#eval total_distance 15 10 (18/5)

end NUMINAMATH_CALUDE_margo_walk_distance_l3325_332511


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l3325_332526

/-- Represents a triangle -/
structure Triangle where
  side_length : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- Calculates the path length of a vertex of a triangle rotating inside a square -/
def path_length (t : Triangle) (s : Square) : ℝ :=
  sorry

/-- Theorem stating the path length for the given triangle and square -/
theorem triangle_rotation_path_length :
  let t : Triangle := { side_length := 3 }
  let s : Square := { side_length := 6 }
  path_length t s = 24 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l3325_332526


namespace NUMINAMATH_CALUDE_spider_journey_l3325_332505

theorem spider_journey (r : ℝ) (leg : ℝ) (h1 : r = 65) (h2 : leg = 90) :
  let diameter := 2 * r
  let hypotenuse := diameter
  let other_leg := Real.sqrt (hypotenuse ^ 2 - leg ^ 2)
  hypotenuse + leg + other_leg = 220 + 20 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_spider_journey_l3325_332505


namespace NUMINAMATH_CALUDE_billys_age_l3325_332586

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 52) : 
  billy = 39 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l3325_332586


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3325_332530

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ z : ℂ, z = a + 1 - a * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3325_332530


namespace NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_200_l3325_332501

/-- Calculate the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : ℚ :=
  let cost := (num_bars : ℚ) * buy_price / 6
  let revenue := (num_bars : ℚ) * sell_price / 3
  revenue - cost

/-- Prove that the scout troop's profit is $200 -/
theorem scout_troop_profit_is_200 :
  scout_troop_profit 1200 (3 : ℚ) (2 : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_200_l3325_332501


namespace NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l3325_332512

theorem at_least_one_less_than_or_equal_to_one
  (x y z : ℝ)
  (positive_x : 0 < x)
  (positive_y : 0 < y)
  (positive_z : 0 < z)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l3325_332512


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l3325_332574

theorem solve_cubic_equation (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  (x + 1)^3 = x^3 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l3325_332574


namespace NUMINAMATH_CALUDE_value_calculation_l3325_332515

/-- If 0.5% of a value equals 65 paise, then the value is 130 rupees -/
theorem value_calculation (a : ℝ) : (0.005 * a = 65 / 100) → a = 130 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l3325_332515


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l3325_332522

/-- Proves that 2 old oranges were thrown away given the initial, added, and final orange counts. -/
theorem oranges_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) 
    (h1 : initial = 5)
    (h2 : added = 28)
    (h3 : final = 31) :
  initial - (initial + added - final) = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l3325_332522


namespace NUMINAMATH_CALUDE_april_flower_sale_l3325_332571

/-- April's flower sale problem -/
theorem april_flower_sale 
  (price_per_rose : ℕ) 
  (initial_roses : ℕ) 
  (remaining_roses : ℕ) 
  (h1 : price_per_rose = 4)
  (h2 : initial_roses = 13)
  (h3 : remaining_roses = 4) :
  (initial_roses - remaining_roses) * price_per_rose = 36 := by
  sorry

end NUMINAMATH_CALUDE_april_flower_sale_l3325_332571


namespace NUMINAMATH_CALUDE_gcd_102_238_l3325_332556

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l3325_332556


namespace NUMINAMATH_CALUDE_third_angle_is_90_l3325_332587

-- Define a triangle with two known angles
def Triangle (angle1 angle2 : ℝ) :=
  { angle3 : ℝ // angle1 + angle2 + angle3 = 180 }

-- Theorem: In a triangle with angles of 30 and 60 degrees, the third angle is 90 degrees
theorem third_angle_is_90 :
  ∀ (t : Triangle 30 60), t.val = 90 := by
  sorry

end NUMINAMATH_CALUDE_third_angle_is_90_l3325_332587


namespace NUMINAMATH_CALUDE_tv_conditional_probability_l3325_332521

theorem tv_conditional_probability 
  (p_10000 : ℝ) 
  (p_15000 : ℝ) 
  (h1 : p_10000 = 0.80) 
  (h2 : p_15000 = 0.60) : 
  p_15000 / p_10000 = 0.75 := by
sorry

end NUMINAMATH_CALUDE_tv_conditional_probability_l3325_332521


namespace NUMINAMATH_CALUDE_opponent_total_score_l3325_332545

def hockey_team_goals : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

structure GameResult where
  team_score : Nat
  opponent_score : Nat

def is_lost_by_two (game : GameResult) : Bool :=
  game.opponent_score = game.team_score + 2

def is_half_or_double (game : GameResult) : Bool :=
  game.team_score = 2 * game.opponent_score ∨ 2 * game.team_score = game.opponent_score

theorem opponent_total_score (games : List GameResult) : 
  (games.length = 8) →
  (games.map (λ g => g.team_score) = hockey_team_goals) →
  (games.filter is_lost_by_two).length = 3 →
  (games.filter (λ g => ¬(is_lost_by_two g))).all is_half_or_double →
  (games.map (λ g => g.opponent_score)).sum = 56 := by
  sorry

end NUMINAMATH_CALUDE_opponent_total_score_l3325_332545


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3325_332564

theorem other_root_of_quadratic (c : ℝ) : 
  (3^2 - 5*3 + c = 0) → 
  (∃ x : ℝ, x ≠ 3 ∧ x^2 - 5*x + c = 0 ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3325_332564


namespace NUMINAMATH_CALUDE_cube_difference_factorization_l3325_332503

theorem cube_difference_factorization (t : ℝ) : t^3 - 8 = (t - 2) * (t^2 + 2*t + 4) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_factorization_l3325_332503


namespace NUMINAMATH_CALUDE_triangle_side_value_l3325_332570

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + 10 * c = 2 * (Real.sin A + Real.sin B + 10 * Real.sin C) ∧
  A = π / 3 →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3325_332570


namespace NUMINAMATH_CALUDE_math_reading_homework_difference_l3325_332598

theorem math_reading_homework_difference :
  let math_pages : ℕ := 5
  let reading_pages : ℕ := 2
  math_pages - reading_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_math_reading_homework_difference_l3325_332598


namespace NUMINAMATH_CALUDE_abs_equation_solution_l3325_332593

theorem abs_equation_solution : ∃! x : ℚ, |x - 3| = |x - 4| := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l3325_332593


namespace NUMINAMATH_CALUDE_triangle_inequality_l3325_332551

/-- Triangle inequality proof -/
theorem triangle_inequality (a b c s R r : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 → r > 0 →
  s = (a + b + c) / 2 →
  a + b > c → b + c > a → c + a > b →
  (a / (s - a)) + (b / (s - b)) + (c / (s - c)) ≥ 3 * R / r := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3325_332551


namespace NUMINAMATH_CALUDE_max_attempts_for_ten_rooms_l3325_332566

/-- The maximum number of attempts needed to match n keys to n rooms -/
def maxAttempts (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- The number of rooms and keys -/
def numRooms : ℕ := 10

theorem max_attempts_for_ten_rooms :
  maxAttempts numRooms = 45 :=
by sorry

end NUMINAMATH_CALUDE_max_attempts_for_ten_rooms_l3325_332566


namespace NUMINAMATH_CALUDE_part_one_part_two_l3325_332510

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one (x : ℝ) : 
  p x 1 ∨ q x → 1 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (a > 0 ∧ (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)) → 
  1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3325_332510
