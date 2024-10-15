import Mathlib

namespace NUMINAMATH_CALUDE_student_project_assignment_l1405_140539

/-- The number of ways to assign students to projects. -/
def assignmentCount (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then (n - k + 1).factorial * (n.choose k) else 0

/-- Theorem stating the number of ways to assign 6 students to 3 projects. -/
theorem student_project_assignment :
  assignmentCount 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_project_assignment_l1405_140539


namespace NUMINAMATH_CALUDE_courtyard_width_l1405_140555

/-- Represents the dimensions of a brick in centimeters -/
structure Brick where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its length and width -/
def area (length width : ℝ) : ℝ := length * width

/-- Theorem: The width of the courtyard is 15 meters -/
theorem courtyard_width (b : Brick) (c : Courtyard) (total_bricks : ℕ) :
  b.length = 0.2 →
  b.width = 0.1 →
  c.length = 25 →
  total_bricks = 18750 →
  area c.length c.width = (total_bricks : ℝ) * area b.length b.width →
  c.width = 15 := by
  sorry

#check courtyard_width

end NUMINAMATH_CALUDE_courtyard_width_l1405_140555


namespace NUMINAMATH_CALUDE_arctan_sum_right_triangle_l1405_140535

theorem arctan_sum_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 2 * b) :
  Real.arctan (b / a) + Real.arctan (a / b) = π / 2 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_right_triangle_l1405_140535


namespace NUMINAMATH_CALUDE_total_meat_theorem_l1405_140524

/-- The amount of beef needed for one beef hamburger -/
def beef_per_hamburger : ℚ := 4 / 10

/-- The amount of chicken needed for one chicken hamburger -/
def chicken_per_hamburger : ℚ := 2.5 / 5

/-- The number of beef hamburgers to be made -/
def beef_hamburgers : ℕ := 30

/-- The number of chicken hamburgers to be made -/
def chicken_hamburgers : ℕ := 15

/-- The total amount of meat needed for the given number of beef and chicken hamburgers -/
def total_meat_needed : ℚ := beef_per_hamburger * beef_hamburgers + chicken_per_hamburger * chicken_hamburgers

theorem total_meat_theorem : total_meat_needed = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_total_meat_theorem_l1405_140524


namespace NUMINAMATH_CALUDE_painting_ratio_l1405_140510

theorem painting_ratio (monday : ℝ) (total : ℝ) : 
  monday = 30 →
  total = 105 →
  (total - (monday + 2 * monday)) / monday = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_painting_ratio_l1405_140510


namespace NUMINAMATH_CALUDE_valid_quadrilaterals_count_l1405_140546

/-- Represents a quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral :=
  (a b c d : ℕ)

/-- Checks if a quadrilateral is valid according to the problem conditions -/
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.a + q.b + q.c + q.d = 40 ∧
  q.a ≥ 5 ∧ q.b ≥ 5 ∧ q.c ≥ 5 ∧ q.d ≥ 5 ∧
  q.a < q.b + q.c + q.d ∧
  q.b < q.a + q.c + q.d ∧
  q.c < q.a + q.b + q.d ∧
  q.d < q.a + q.b + q.c

/-- Counts the number of valid quadrilaterals -/
def count_valid_quadrilaterals : ℕ := sorry

theorem valid_quadrilaterals_count :
  count_valid_quadrilaterals = 680 := by sorry

end NUMINAMATH_CALUDE_valid_quadrilaterals_count_l1405_140546


namespace NUMINAMATH_CALUDE_acidic_mixture_concentration_l1405_140516

/-- Proves that mixing liquids from two containers with given concentrations
    results in a mixture with the desired concentration. -/
theorem acidic_mixture_concentration
  (volume1 : ℝ) (volume2 : ℝ) (conc1 : ℝ) (conc2 : ℝ) (target_conc : ℝ)
  (x : ℝ) (y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0)
  (h_vol1 : volume1 = 54) (h_vol2 : volume2 = 48)
  (h_conc1 : conc1 = 0.35) (h_conc2 : conc2 = 0.25)
  (h_target : target_conc = 0.75)
  (h_mixture : conc1 * x + conc2 * y = target_conc * (x + y)) :
  (conc1 * x + conc2 * y) / (x + y) = target_conc :=
sorry

end NUMINAMATH_CALUDE_acidic_mixture_concentration_l1405_140516


namespace NUMINAMATH_CALUDE_vector_collinearity_l1405_140570

theorem vector_collinearity (k : ℝ) : 
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (0, 1)
  let v1 : ℝ × ℝ := (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2)
  let v2 : ℝ × ℝ := (k * a.1 + 6 * b.1, k * a.2 + 6 * b.2)
  (∃ (t : ℝ), v1 = (t * v2.1, t * v2.2)) → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1405_140570


namespace NUMINAMATH_CALUDE_sum_product_over_sum_squares_is_zero_l1405_140563

theorem sum_product_over_sum_squares_is_zero 
  (x y z : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) 
  (hsum : x + y + z = 1) : 
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_product_over_sum_squares_is_zero_l1405_140563


namespace NUMINAMATH_CALUDE_triangle_properties_l1405_140540

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define vector CM
variable (CM : ℝ × ℝ)

-- Given conditions
axiom side_angle_relation : 2 * b * Real.cos C = 2 * a - Real.sqrt 3 * c
axiom vector_relation : (0, 0) + CM + CM = (a, 0) + (b * Real.cos C, b * Real.sin C)
axiom cm_length : Real.sqrt (CM.1^2 + CM.2^2) = 1

-- Theorem to prove
theorem triangle_properties :
  B = π / 6 ∧
  (∃ (area : ℝ), area ≤ Real.sqrt 3 / 2 ∧
    ∀ (other_area : ℝ), other_area = 1/2 * a * b * Real.sin C → other_area ≤ area) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1405_140540


namespace NUMINAMATH_CALUDE_like_terms_exponent_equality_l1405_140582

theorem like_terms_exponent_equality (a b : ℤ) : 
  (2 * a + b = 6 ∧ a - b = 3) → a + 2 * b = 3 := by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_equality_l1405_140582


namespace NUMINAMATH_CALUDE_number_puzzle_l1405_140513

theorem number_puzzle : ∃ x : ℝ, 13 * x = x + 180 ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1405_140513


namespace NUMINAMATH_CALUDE_intersection_point_exists_in_interval_l1405_140599

theorem intersection_point_exists_in_interval :
  ∃! x : ℝ, 3 < x ∧ x < 4 ∧ Real.log x = 7 - 2 * x := by sorry

end NUMINAMATH_CALUDE_intersection_point_exists_in_interval_l1405_140599


namespace NUMINAMATH_CALUDE_f_maps_neg_two_three_to_one_neg_six_l1405_140579

/-- The mapping f that transforms a point (x, y) to (x+y, xy) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

/-- Theorem stating that f maps (-2, 3) to (1, -6) -/
theorem f_maps_neg_two_three_to_one_neg_six :
  f (-2, 3) = (1, -6) := by sorry

end NUMINAMATH_CALUDE_f_maps_neg_two_three_to_one_neg_six_l1405_140579


namespace NUMINAMATH_CALUDE_percentage_problem_l1405_140536

theorem percentage_problem (n : ℝ) (h : 1.2 * n = 2400) : 0.2 * n = 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1405_140536


namespace NUMINAMATH_CALUDE_gerald_toy_car_donation_l1405_140529

/-- Proves that the fraction of toy cars Gerald donated is 1/4 -/
theorem gerald_toy_car_donation :
  let initial_cars : ℕ := 20
  let remaining_cars : ℕ := 15
  let donated_cars : ℕ := initial_cars - remaining_cars
  donated_cars / initial_cars = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_gerald_toy_car_donation_l1405_140529


namespace NUMINAMATH_CALUDE_chord_length_l1405_140532

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus at 135°
def line (x y : ℝ) : Prop := y = -x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
  A ≠ B ∧ ‖A - B‖ = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l1405_140532


namespace NUMINAMATH_CALUDE_sqrt_six_irrational_between_two_and_three_l1405_140576

theorem sqrt_six_irrational_between_two_and_three :
  ∃ x : ℝ, Irrational x ∧ 2 < x ∧ x < 3 :=
by
  use Real.sqrt 6
  sorry

end NUMINAMATH_CALUDE_sqrt_six_irrational_between_two_and_three_l1405_140576


namespace NUMINAMATH_CALUDE_unique_number_l1405_140504

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ,
  is_two_digit n ∧
  Odd n ∧
  n % 9 = 0 ∧
  is_perfect_square (digit_product n) ∧
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_number_l1405_140504


namespace NUMINAMATH_CALUDE_max_bishops_on_mountain_board_l1405_140594

/-- A chessboard with two mountains --/
structure MountainChessboard :=
  (black_regions : ℕ)
  (white_regions : ℕ)

/-- The maximum number of non-attacking bishops on a mountain chessboard --/
def max_bishops (board : MountainChessboard) : ℕ :=
  board.black_regions + board.white_regions

/-- Theorem: The maximum number of non-attacking bishops on the given mountain chessboard is 19 --/
theorem max_bishops_on_mountain_board :
  ∃ (board : MountainChessboard), 
    board.black_regions = 11 ∧ 
    board.white_regions = 8 ∧ 
    max_bishops board = 19 := by
  sorry

#eval max_bishops ⟨11, 8⟩

end NUMINAMATH_CALUDE_max_bishops_on_mountain_board_l1405_140594


namespace NUMINAMATH_CALUDE_oliver_candy_to_janet_l1405_140548

theorem oliver_candy_to_janet (initial_candy : ℕ) (remaining_candy : ℕ) : 
  initial_candy = 78 → remaining_candy = 68 → initial_candy - remaining_candy = 10 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_to_janet_l1405_140548


namespace NUMINAMATH_CALUDE_smallest_consecutive_multiples_l1405_140556

theorem smallest_consecutive_multiples : 
  let a := 1735
  ∀ n : ℕ, n < a → ¬(
    (n.succ % 5 = 0) ∧ 
    ((n + 2) % 7 = 0) ∧ 
    ((n + 3) % 9 = 0) ∧ 
    ((n + 4) % 11 = 0)
  ) ∧
  (a % 5 = 0) ∧ 
  ((a + 1) % 7 = 0) ∧ 
  ((a + 2) % 9 = 0) ∧ 
  ((a + 3) % 11 = 0) := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_multiples_l1405_140556


namespace NUMINAMATH_CALUDE_well_digging_time_l1405_140527

/-- Represents the time taken to dig a meter at a given depth -/
def digTime (depth : ℕ) : ℕ := 40 + (depth - 1) * 10

/-- Converts minutes to hours -/
def minutesToHours (minutes : ℕ) : ℚ := minutes / 60

theorem well_digging_time :
  minutesToHours (digTime 21) = 4 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_time_l1405_140527


namespace NUMINAMATH_CALUDE_power_of_power_of_two_l1405_140567

theorem power_of_power_of_two : (2^2)^(2^2) = 256 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_two_l1405_140567


namespace NUMINAMATH_CALUDE_petya_max_candies_l1405_140554

/-- Represents the state of a pile of candies -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Defines a player's move -/
inductive Move
  | take : Nat → Move

/-- Defines the result of a move -/
inductive MoveResult
  | eat : MoveResult
  | throw : MoveResult

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : Option (GameState × MoveResult) :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Simulates the game with given strategies -/
def playGame (initialState : GameState) (petyaStrategy : Strategy) (vasyaStrategy : Strategy) : Nat :=
  sorry

/-- The initial game state -/
def initialGameState : GameState :=
  { piles := List.range 55 |>.map (fun i => { count := i + 1 }) }

theorem petya_max_candies :
  ∀ (petyaStrategy : Strategy),
  ∃ (vasyaStrategy : Strategy),
  playGame initialGameState petyaStrategy vasyaStrategy ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_petya_max_candies_l1405_140554


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l1405_140589

theorem farmer_land_calculation (total_land : ℝ) : 
  (0.05 * 0.9 * total_land + 0.05 * 0.9 * total_land = 90) → total_land = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l1405_140589


namespace NUMINAMATH_CALUDE_remaining_books_l1405_140517

/-- Given an initial number of books and a number of books sold,
    proves that the remaining number of books is equal to
    the difference between the initial number and the number sold. -/
theorem remaining_books (initial : ℕ) (sold : ℕ) (h : sold ≤ initial) :
  initial - sold = initial - sold :=
by sorry

end NUMINAMATH_CALUDE_remaining_books_l1405_140517


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_neq_neg_twelve_fifths_l1405_140530

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (b : ℝ), l1.point.2.2 = b ∧
  (∀ (t u : ℝ),
    l1.point.1 + t * l1.direction.1 ≠ l2.point.1 + u * l2.direction.1 ∨
    l1.point.2.1 + t * l1.direction.2.1 ≠ l2.point.2.1 + u * l2.direction.2.1 ∨
    b + t * l1.direction.2.2 ≠ l2.point.2.2 + u * l2.direction.2.2)

/-- The main theorem -/
theorem lines_skew_iff_b_neq_neg_twelve_fifths :
  ∀ (b : ℝ),
  let l1 : Line3D := ⟨(2, 3, b), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(5, 6, 1), (6, 3, 2)⟩
  are_skew l1 l2 ↔ b ≠ -12/5 := by
  sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_neq_neg_twelve_fifths_l1405_140530


namespace NUMINAMATH_CALUDE_solve_simple_interest_l1405_140511

def simple_interest_problem (principal : ℝ) (interest_paid : ℝ) : Prop :=
  ∃ (rate : ℝ),
    principal = 900 ∧
    interest_paid = 729 ∧
    rate > 0 ∧
    rate < 100 ∧
    interest_paid = (principal * rate * rate) / 100 ∧
    rate = 9

theorem solve_simple_interest :
  ∀ (principal interest_paid : ℝ),
    simple_interest_problem principal interest_paid :=
  sorry

end NUMINAMATH_CALUDE_solve_simple_interest_l1405_140511


namespace NUMINAMATH_CALUDE_restaurant_serving_totals_l1405_140541

/-- Represents the number of food items served at a meal -/
structure MealServing :=
  (hotDogs : ℕ)
  (hamburgers : ℕ)
  (sandwiches : ℕ)
  (salads : ℕ)

/-- Represents the meals served in a day -/
structure DayMeals :=
  (breakfast : MealServing)
  (lunch : MealServing)
  (dinner : MealServing)

def day1 : DayMeals := {
  breakfast := { hotDogs := 15, hamburgers := 8, sandwiches := 6, salads := 10 },
  lunch := { hotDogs := 20, hamburgers := 18, sandwiches := 12, salads := 15 },
  dinner := { hotDogs := 4, hamburgers := 10, sandwiches := 12, salads := 5 }
}

def day2 : DayMeals := {
  breakfast := { hotDogs := 6, hamburgers := 12, sandwiches := 9, salads := 7 },
  lunch := { hotDogs := 10, hamburgers := 20, sandwiches := 16, salads := 12 },
  dinner := { hotDogs := 3, hamburgers := 7, sandwiches := 5, salads := 8 }
}

def day3 : DayMeals := {
  breakfast := { hotDogs := 10, hamburgers := 14, sandwiches := 8, salads := 6 },
  lunch := { hotDogs := 12, hamburgers := 16, sandwiches := 10, salads := 9 },
  dinner := { hotDogs := 8, hamburgers := 9, sandwiches := 7, salads := 10 }
}

theorem restaurant_serving_totals :
  let breakfastLunchTotal := (day1.breakfast.hotDogs + day1.lunch.hotDogs + 
                              day2.breakfast.hotDogs + day2.lunch.hotDogs + 
                              day3.breakfast.hotDogs + day3.lunch.hotDogs) +
                             (day1.breakfast.hamburgers + day1.lunch.hamburgers + 
                              day2.breakfast.hamburgers + day2.lunch.hamburgers + 
                              day3.breakfast.hamburgers + day3.lunch.hamburgers) +
                             (day1.breakfast.sandwiches + day1.lunch.sandwiches + 
                              day2.breakfast.sandwiches + day2.lunch.sandwiches + 
                              day3.breakfast.sandwiches + day3.lunch.sandwiches)
  let saladTotal := day1.breakfast.salads + day1.lunch.salads + day1.dinner.salads +
                    day2.breakfast.salads + day2.lunch.salads + day2.dinner.salads +
                    day3.breakfast.salads + day3.lunch.salads + day3.dinner.salads
  breakfastLunchTotal = 222 ∧ saladTotal = 82 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_serving_totals_l1405_140541


namespace NUMINAMATH_CALUDE_range_of_a_l1405_140502

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ (Set.Ioo 0 1), (x + Real.log a) / Real.exp x - a * Real.log x / x > 0) →
  a ∈ Set.Icc (Real.exp (-1)) 1 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1405_140502


namespace NUMINAMATH_CALUDE_expression_value_l1405_140591

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - 2*z = 0)
  (eq2 : x + 3*y - 28*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y*z) / (y^2 + z^2) = 280/37 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1405_140591


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1405_140551

theorem trigonometric_identity :
  3 * Real.tan (10 * π / 180) + 4 * Real.sqrt 3 * Real.sin (10 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1405_140551


namespace NUMINAMATH_CALUDE_special_triangle_l1405_140537

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Theorem about a special triangle -/
theorem special_triangle (t : Triangle) 
  (hm : Vector2D) 
  (hn : Vector2D) 
  (collinear : hm.x * hn.y = hm.y * hn.x) 
  (dot_product : t.a * t.c * Real.cos t.C = -27) 
  (hm_def : hm = ⟨t.a - t.b, Real.sin t.A + Real.sin t.C⟩) 
  (hn_def : hn = ⟨t.a - t.c, Real.sin (t.A + t.C)⟩) :
  t.C = π/3 ∧ 
  (∃ (min_AB : ℝ), min_AB = 3 * Real.sqrt 6 ∧ 
    ∀ (AB : ℝ), AB ≥ min_AB) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_l1405_140537


namespace NUMINAMATH_CALUDE_integer_solutions_of_difference_of_squares_l1405_140575

theorem integer_solutions_of_difference_of_squares :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^2 - y^2 = 12) ∧
    Finset.card s = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_difference_of_squares_l1405_140575


namespace NUMINAMATH_CALUDE_seating_arrangement_solution_l1405_140506

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  row_size_1 : ℕ
  row_size_2 : ℕ
  rows_of_size_1 : ℕ
  rows_of_size_2 : ℕ

/-- Defines a valid seating arrangement --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = s.row_size_1 * s.rows_of_size_1 + s.row_size_2 * s.rows_of_size_2

/-- The specific seating arrangement for our problem --/
def problem_arrangement : SeatingArrangement :=
  { total_people := 58
  , row_size_1 := 7
  , row_size_2 := 9
  , rows_of_size_1 := 7  -- This value is not given in the problem, but needed for the structure
  , rows_of_size_2 := 1  -- This is what we want to prove
  }

/-- The main theorem to prove --/
theorem seating_arrangement_solution :
  is_valid_arrangement problem_arrangement ∧
  ∀ s : SeatingArrangement,
    s.total_people = problem_arrangement.total_people ∧
    s.row_size_1 = problem_arrangement.row_size_1 ∧
    s.row_size_2 = problem_arrangement.row_size_2 ∧
    is_valid_arrangement s →
    s.rows_of_size_2 = problem_arrangement.rows_of_size_2 :=
by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_solution_l1405_140506


namespace NUMINAMATH_CALUDE_gym_cost_theorem_l1405_140581

/-- Calculates the total cost of two gym memberships for one year -/
def total_gym_cost (cheap_monthly : ℕ) (cheap_signup : ℕ) (months : ℕ) : ℕ :=
  let expensive_monthly := 3 * cheap_monthly
  let cheap_total := cheap_monthly * months + cheap_signup
  let expensive_total := expensive_monthly * months + (expensive_monthly * 4)
  cheap_total + expensive_total

/-- Theorem stating that the total cost for two gym memberships for one year is $650 -/
theorem gym_cost_theorem : total_gym_cost 10 50 12 = 650 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_theorem_l1405_140581


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1405_140557

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 4 * n + 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define T_n
def T (n : ℕ) : ℚ := n / (2 * n + 2)

theorem arithmetic_sequence_properties :
  (a 2 = 9) ∧ (S 5 = 65) →
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (2 * n + 2)) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1405_140557


namespace NUMINAMATH_CALUDE_jam_weight_l1405_140534

/-- Calculates the weight of jam given the initial and final suitcase weights and other item weights --/
theorem jam_weight 
  (initial_weight : ℝ) 
  (final_weight : ℝ) 
  (perfume_weight : ℝ) 
  (perfume_count : ℕ) 
  (chocolate_weight : ℝ) 
  (soap_weight : ℝ) 
  (soap_count : ℕ) 
  (h1 : initial_weight = 5) 
  (h2 : final_weight = 11) 
  (h3 : perfume_weight = 1.2 / 16) 
  (h4 : perfume_count = 5) 
  (h5 : chocolate_weight = 4) 
  (h6 : soap_weight = 5 / 16) 
  (h7 : soap_count = 2) : 
  final_weight - (initial_weight + perfume_weight * perfume_count + chocolate_weight + soap_weight * soap_count) = 1 := by
  sorry

#check jam_weight

end NUMINAMATH_CALUDE_jam_weight_l1405_140534


namespace NUMINAMATH_CALUDE_inequality_proof_l1405_140597

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / (x^4 + y^2) + y / (x^2 + y^4) ≤ 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1405_140597


namespace NUMINAMATH_CALUDE_find_k_value_l1405_140544

theorem find_k_value (k : ℝ) (h : 32 / k = 4) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l1405_140544


namespace NUMINAMATH_CALUDE_range_of_a_l1405_140574

/-- Given sets A and B, and their empty intersection, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | |x - a| ≤ 1}
  let B : Set ℝ := {x | x^2 - 5*x + 4 > 0}
  A ∩ B = ∅ → a ∈ Set.Icc 2 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1405_140574


namespace NUMINAMATH_CALUDE_fish_gone_bad_percentage_l1405_140538

theorem fish_gone_bad_percentage (fish_per_roll fish_bought rolls_made : ℕ) 
  (h1 : fish_per_roll = 40)
  (h2 : fish_bought = 400)
  (h3 : rolls_made = 8) :
  (fish_bought - rolls_made * fish_per_roll) / fish_bought * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fish_gone_bad_percentage_l1405_140538


namespace NUMINAMATH_CALUDE_sum_of_two_angles_in_triangle_l1405_140585

/-- Theorem: In a triangle where one angle is 72°, the sum of the other two angles is 108° -/
theorem sum_of_two_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : B = 72) : 
  A + C = 108 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_angles_in_triangle_l1405_140585


namespace NUMINAMATH_CALUDE_solution_5tuples_l1405_140543

theorem solution_5tuples :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ | 
    let (a, b, c, d, n) := t
    (a + b + c + d = 100) ∧
    (n > 0) ∧
    (a + n = b - n) ∧
    (b - n = c * n) ∧
    (c * n = d / n)} =
  {(24, 26, 25, 25, 1), (12, 20, 4, 64, 4), (0, 18, 1, 81, 9)} :=
by sorry

end NUMINAMATH_CALUDE_solution_5tuples_l1405_140543


namespace NUMINAMATH_CALUDE_sector_perimeter_ratio_l1405_140507

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) (h_positive : 0 < α ∧ 0 < r ∧ 0 < R) 
  (h_area_ratio : (α * r^2) / (α * R^2) = 1/4) : 
  (2*r + α*r) / (2*R + α*R) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sector_perimeter_ratio_l1405_140507


namespace NUMINAMATH_CALUDE_clothing_cost_problem_l1405_140515

theorem clothing_cost_problem (total_spent : ℕ) (num_pieces : ℕ) (piece1_cost : ℕ) (piece2_cost : ℕ) (same_cost_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  piece1_cost = 49 →
  same_cost_piece = 96 →
  total_spent = piece1_cost + piece2_cost + 5 * same_cost_piece →
  piece2_cost = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_clothing_cost_problem_l1405_140515


namespace NUMINAMATH_CALUDE_pencils_remaining_l1405_140522

/-- The number of pencils left in a box after some are taken -/
def pencils_left (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem: Given 79 initial pencils and 4 taken, 75 pencils are left -/
theorem pencils_remaining : pencils_left 79 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l1405_140522


namespace NUMINAMATH_CALUDE_ab_value_l1405_140564

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 18 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1405_140564


namespace NUMINAMATH_CALUDE_sophias_book_length_l1405_140559

theorem sophias_book_length :
  ∀ (total_pages : ℕ),
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 90 →
  total_pages = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_sophias_book_length_l1405_140559


namespace NUMINAMATH_CALUDE_tank_capacity_l1405_140586

/-- Represents the capacity of a tank and its inlet/outlet pipes. -/
structure TankSystem where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- Theorem stating the capacity of the tank given the conditions. -/
theorem tank_capacity (t : TankSystem)
  (h1 : t.outlet_time = 5)
  (h2 : t.inlet_rate = 4 * 60)  -- 4 litres/min converted to litres/hour
  (h3 : t.combined_time = 8)
  : t.capacity = 3200 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1405_140586


namespace NUMINAMATH_CALUDE_min_marked_cells_7x7_l1405_140588

/-- Represents a grid with dimensions (2n-1) x (2n-1) -/
def Grid (n : ℕ) := Fin (2*n - 1) → Fin (2*n - 1) → Bool

/-- Checks if a 1 x 4 strip contains a marked cell -/
def stripContainsMarked (g : Grid 4) (start_row start_col : Fin 7) (isHorizontal : Bool) : Prop :=
  ∃ i : Fin 4, g (if isHorizontal then start_row else start_row + i) 
               (if isHorizontal then start_col + i else start_col) = true

/-- A valid marking satisfies the strip condition for all strips -/
def isValidMarking (g : Grid 4) : Prop :=
  ∀ row col : Fin 7, ∀ isHorizontal : Bool, 
    stripContainsMarked g row col isHorizontal

/-- Counts the number of marked cells in a grid -/
def countMarked (g : Grid 4) : ℕ :=
  (Finset.univ.filter (λ x : Fin 7 × Fin 7 => g x.1 x.2)).card

/-- Main theorem: The minimum number of marked cells in a valid 7x7 grid marking is 12 -/
theorem min_marked_cells_7x7 :
  (∃ g : Grid 4, isValidMarking g ∧ countMarked g = 12) ∧
  (∀ g : Grid 4, isValidMarking g → countMarked g ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_min_marked_cells_7x7_l1405_140588


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_property_l1405_140595

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility_property :
  ∃! (a b m : ℕ), 
    0 < a ∧ a < m ∧
    0 < b ∧ b < m ∧
    (∀ n : ℕ, n > 0 → ∃ k : ℤ, fibonacci n - a * n * b^n = m * k) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_property_l1405_140595


namespace NUMINAMATH_CALUDE_P_zero_equals_eleven_l1405_140503

variables (a b c : ℝ) (P : ℝ → ℝ)

/-- The roots of the cubic equation -/
axiom root_equation : a^3 + 3*a^2 + 5*a + 7 = 0 ∧ 
                      b^3 + 3*b^2 + 5*b + 7 = 0 ∧ 
                      c^3 + 3*c^2 + 5*c + 7 = 0

/-- Properties of polynomial P -/
axiom P_properties : P a = b + c ∧ 
                     P b = a + c ∧ 
                     P c = a + b ∧ 
                     P (a + b + c) = -16

/-- Theorem: P(0) equals 11 -/
theorem P_zero_equals_eleven : P 0 = 11 := by sorry

end NUMINAMATH_CALUDE_P_zero_equals_eleven_l1405_140503


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1405_140550

-- Define vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)

-- Define the perpendicularity condition
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem statement
theorem perpendicular_vectors_x_value :
  ∃ x : ℝ, is_perpendicular (a.1 - x * b.1, a.2 - x * b.2) (a.1 - b.1, a.2 - b.2) ∧ x = -7/3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1405_140550


namespace NUMINAMATH_CALUDE_expression_evaluation_l1405_140558

theorem expression_evaluation : (π - 2023)^0 + |1 - Real.sqrt 3| + Real.sqrt 8 - Real.tan (π / 3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1405_140558


namespace NUMINAMATH_CALUDE_camping_hike_distance_l1405_140542

/-- The total distance hiked by Irwin's family on their camping trip -/
theorem camping_hike_distance 
  (car_to_stream : ℝ) 
  (stream_to_meadow : ℝ) 
  (meadow_to_campsite : ℝ) 
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  car_to_stream + stream_to_meadow + meadow_to_campsite = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l1405_140542


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1405_140562

/-- A line in 2D space defined by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A line in 2D space defined by a standard equation ax + by = c --/
structure StandardLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Convert a parametric line to its standard form --/
def parametricToStandard (l : ParametricLine) : StandardLine :=
  sorry

/-- Check if two lines are perpendicular --/
def arePerpendicular (l1 l2 : StandardLine) : Prop :=
  sorry

/-- The main theorem --/
theorem perpendicular_lines (k : ℝ) : 
  let l1 := ParametricLine.mk (λ t => 1 + 2*t) (λ t => 3 + 2*t)
  let l2 := StandardLine.mk 4 k 1
  arePerpendicular (parametricToStandard l1) l2 → k = 4 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1405_140562


namespace NUMINAMATH_CALUDE_circle_radius_range_equivalence_l1405_140545

/-- A circle in a 2D Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle has exactly two points at distance 1 from x-axis -/
def has_two_points_at_distance_one (c : Circle) : Prop :=
  ∃ (p1 p2 : ℝ × ℝ),
    (p1 ≠ p2) ∧
    (p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2 = c.radius^2 ∧
    (p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2 = c.radius^2 ∧
    (abs p1.2 = 1 ∨ abs p2.2 = 1) ∧
    (∀ (p : ℝ × ℝ), 
      (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 → 
      abs p.2 = 1 → (p = p1 ∨ p = p2))

/-- The main theorem stating the equivalence -/
theorem circle_radius_range_equivalence :
  ∀ (c : Circle),
    c.center = (3, -5) →
    (has_two_points_at_distance_one c ↔ (4 < c.radius ∧ c.radius < 6)) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_range_equivalence_l1405_140545


namespace NUMINAMATH_CALUDE_bill_muffin_batches_l1405_140553

/-- The cost of blueberries in dollars per 6 ounce carton -/
def blueberry_cost : ℚ := 5

/-- The cost of raspberries in dollars per 12 ounce carton -/
def raspberry_cost : ℚ := 3

/-- The amount of fruit in ounces required for each batch of muffins -/
def fruit_per_batch : ℚ := 12

/-- The total savings in dollars by using raspberries instead of blueberries -/
def total_savings : ℚ := 22

/-- The number of batches Bill plans to make -/
def num_batches : ℕ := 3

/-- Theorem stating that given the costs, fruit requirement, and total savings,
    Bill plans to make 3 batches of muffins -/
theorem bill_muffin_batches :
  (blueberry_cost * 2 - raspberry_cost) * (num_batches : ℚ) ≤ total_savings ∧
  (blueberry_cost * 2 - raspberry_cost) * ((num_batches + 1) : ℚ) > total_savings :=
by sorry

end NUMINAMATH_CALUDE_bill_muffin_batches_l1405_140553


namespace NUMINAMATH_CALUDE_smallest_n_for_polynomial_roots_l1405_140518

theorem smallest_n_for_polynomial_roots : ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, 0 < k → k < n →
    ¬∃ (a b : ℤ), ∃ (x y : ℝ),
      0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
      k * x^2 + a * x + b = 0 ∧
      k * y^2 + a * y + b = 0) ∧
  (∃ (a b : ℤ), ∃ (x y : ℝ),
    0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
    n * x^2 + a * x + b = 0 ∧
    n * y^2 + a * y + b = 0) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_polynomial_roots_l1405_140518


namespace NUMINAMATH_CALUDE_star_inequality_l1405_140520

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem star_inequality (x y : ℝ) : 3 * (star x y) ≠ star (3*x) (3*y) := by
  sorry

end NUMINAMATH_CALUDE_star_inequality_l1405_140520


namespace NUMINAMATH_CALUDE_closest_multiple_of_17_to_3513_l1405_140519

theorem closest_multiple_of_17_to_3513 :
  ∀ k : ℤ, |3519 - 3513| ≤ |17 * k - 3513| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_17_to_3513_l1405_140519


namespace NUMINAMATH_CALUDE_cost_of_jeans_l1405_140577

/-- The cost of a pair of jeans -/
def cost_jeans : ℝ := sorry

/-- The cost of a shirt -/
def cost_shirt : ℝ := sorry

/-- The first condition: 3 pairs of jeans and 6 shirts cost $104.25 -/
axiom condition1 : 3 * cost_jeans + 6 * cost_shirt = 104.25

/-- The second condition: 4 pairs of jeans and 5 shirts cost $112.15 -/
axiom condition2 : 4 * cost_jeans + 5 * cost_shirt = 112.15

/-- Theorem stating that the cost of each pair of jeans is $16.85 -/
theorem cost_of_jeans : cost_jeans = 16.85 := by sorry

end NUMINAMATH_CALUDE_cost_of_jeans_l1405_140577


namespace NUMINAMATH_CALUDE_parabola_vertex_l1405_140500

/-- The parabola is defined by the equation y = (x - 1)² - 3 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := -3

/-- Theorem: The vertex of the parabola y = (x - 1)² - 3 has coordinates (1, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola vertex_x) ∧ 
  parabola vertex_x = vertex_y := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1405_140500


namespace NUMINAMATH_CALUDE_andrews_age_l1405_140508

theorem andrews_age (a g : ℚ) 
  (h1 : g = 10 * a)
  (h2 : g - (a + 2) = 57) :
  a = 59 / 9 := by
  sorry

end NUMINAMATH_CALUDE_andrews_age_l1405_140508


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1405_140573

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_four_consecutive (n : ℕ) : ℕ :=
  (Nat.factorial (n - 3)) * (Nat.factorial 4)

theorem seating_arrangements_with_restriction (n : ℕ) (k : ℕ) 
  (h1 : n = 10) (h2 : k = 4) : 
  total_arrangements n - arrangements_with_four_consecutive n = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1405_140573


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1405_140514

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1405_140514


namespace NUMINAMATH_CALUDE_first_class_students_l1405_140523

/-- Represents the number of students in the first class -/
def x : ℕ := sorry

/-- The average mark of the first class -/
def avg_first : ℝ := 40

/-- The number of students in the second class -/
def students_second : ℕ := 50

/-- The average mark of the second class -/
def avg_second : ℝ := 70

/-- The average mark of all students combined -/
def avg_total : ℝ := 58.75

/-- Theorem stating that the number of students in the first class is 30 -/
theorem first_class_students : 
  (x * avg_first + students_second * avg_second) / (x + students_second) = avg_total → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_class_students_l1405_140523


namespace NUMINAMATH_CALUDE_distance_at_two_point_five_l1405_140531

/-- The distance traveled by a ball rolling down an inclined plane -/
def distance (t : ℝ) : ℝ := 10 * t^2

/-- Theorem: The distance traveled at t = 2.5 seconds is 62.5 feet -/
theorem distance_at_two_point_five :
  distance 2.5 = 62.5 := by sorry

end NUMINAMATH_CALUDE_distance_at_two_point_five_l1405_140531


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_l1405_140552

def condition_P (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 0

def condition_Q (x y : ℝ) : Prop := (x - 1) * (y - 1) = 0

theorem P_sufficient_not_necessary :
  (∀ x y : ℝ, condition_P x y → condition_Q x y) ∧
  ¬(∀ x y : ℝ, condition_Q x y → condition_P x y) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_l1405_140552


namespace NUMINAMATH_CALUDE_largest_decimal_l1405_140501

theorem largest_decimal : 
  let a := 0.938
  let b := 0.9389
  let c := 0.93809
  let d := 0.839
  let e := 0.893
  b = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l1405_140501


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_l1405_140587

/-- An arithmetic sequence with given properties -/
def arithmeticSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 3 = 17 ∧ a 6 = 32 ∧ ∀ n, a (n + 1) - a n = a 2 - a 1

theorem fourth_fifth_sum (a : ℕ → ℕ) (h : arithmeticSequence a) : a 4 + a 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_fourth_fifth_sum_l1405_140587


namespace NUMINAMATH_CALUDE_square_difference_l1405_140509

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1405_140509


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1405_140571

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a < 0 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1405_140571


namespace NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l1405_140528

-- Define the basic units
def ounce : ℕ := 1
def pound : ℕ := 16 * ounce
def ton : ℕ := 2000 * pound

-- Define the packet weight
def packet_weight : ℕ := 16 * pound + 4 * ounce

-- Define the gunny bag capacity
def gunny_bag_capacity : ℕ := 13 * ton

-- Theorem statement
theorem one_ton_equals_2000_pounds : 
  (2000 * packet_weight = gunny_bag_capacity) → ton = 2000 * pound := by
  sorry

end NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l1405_140528


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1405_140521

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (x - 14) = 2) ∧ (x = 18) :=
by
  sorry

#check sqrt_equation_solution

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1405_140521


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l1405_140505

theorem sqrt_27_div_sqrt_3_equals_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l1405_140505


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_l1405_140566

theorem sum_of_complex_roots (a₁ a₂ a₃ : ℂ)
  (h1 : a₁^2 + a₂^2 + a₃^2 = 0)
  (h2 : a₁^3 + a₂^3 + a₃^3 = 0)
  (h3 : a₁^4 + a₂^4 + a₃^4 = 0) :
  a₁ + a₂ + a₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_l1405_140566


namespace NUMINAMATH_CALUDE_four_term_expression_l1405_140547

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    (x^3 - 2)^2 + (x^2 + 2*x)^2 = a*x^n₁ + b*x^n₂ + c*x^n₃ + d 
    ∧ n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > 0
    ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_four_term_expression_l1405_140547


namespace NUMINAMATH_CALUDE_highway_extension_ratio_l1405_140526

/-- The ratio of miles built on the second day to the first day of highway extension -/
theorem highway_extension_ratio :
  let current_length : ℕ := 200
  let extended_length : ℕ := 650
  let first_day_miles : ℕ := 50
  let miles_remaining : ℕ := 250
  let second_day_miles : ℕ := extended_length - current_length - first_day_miles - miles_remaining
  (second_day_miles : ℚ) / first_day_miles = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_highway_extension_ratio_l1405_140526


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l1405_140572

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The remainder when 10! is divided by 13 is 6 -/
theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l1405_140572


namespace NUMINAMATH_CALUDE_tangent_line_and_range_l1405_140561

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 18 * x + 12

-- State the theorem
theorem tangent_line_and_range :
  (∃ (m : ℝ), ∀ (x : ℝ), f' 0 * x = m * x ∧ m = 12) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 3 → f x ∈ Set.Icc 0 9) ∧
  (∃ (y : ℝ), y ∈ Set.Icc 0 9 ∧ ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_range_l1405_140561


namespace NUMINAMATH_CALUDE_exponential_decreasing_condition_l1405_140598

theorem exponential_decreasing_condition (a : ℝ) :
  (((a / (a - 1) ≤ 0) → (0 ≤ a ∧ a < 1)) ∧
   (∃ a, 0 ≤ a ∧ a < 1 ∧ a / (a - 1) > 0) ∧
   (∀ x y : ℝ, x < y → a^x > a^y ↔ 0 < a ∧ a < 1)) ↔
  (((a / (a - 1) ≤ 0) → (∀ x y : ℝ, x < y → a^x > a^y)) ∧
   (¬∀ a : ℝ, (a / (a - 1) ≤ 0) → (∀ x y : ℝ, x < y → a^x > a^y))) :=
by sorry

end NUMINAMATH_CALUDE_exponential_decreasing_condition_l1405_140598


namespace NUMINAMATH_CALUDE_secretary_project_hours_l1405_140533

theorem secretary_project_hours (total_hours : ℕ) (ratio1 ratio2 ratio3 : ℕ) 
  (h1 : ratio1 = 3) (h2 : ratio2 = 7) (h3 : ratio3 = 13) 
  (h_total : total_hours = 253) 
  (h_ratio : ratio1 + ratio2 + ratio3 > 0) :
  (ratio3 * total_hours) / (ratio1 + ratio2 + ratio3) = 143 := by
  sorry

end NUMINAMATH_CALUDE_secretary_project_hours_l1405_140533


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l1405_140568

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem subset_implies_a_values (a : ℝ) : 
  B a ⊆ A → a ∈ ({-1/2, 1/3, 0} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l1405_140568


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l1405_140525

theorem cos_arcsin_three_fifths : Real.cos (Real.arcsin (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l1405_140525


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1405_140560

theorem sum_of_roots_quadratic (x : ℝ) : 
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 6
  let sum_of_roots := -b / a
  2 * x^2 - 8 * x + 6 = 0 → sum_of_roots = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1405_140560


namespace NUMINAMATH_CALUDE_kathleen_savings_problem_l1405_140593

/-- Kathleen's savings and expenses problem -/
theorem kathleen_savings_problem (june july august september : ℚ)
  (school_supplies clothes gift book donation : ℚ) :
  june = 21 →
  july = 46 →
  august = 45 →
  september = 32 →
  school_supplies = 12 →
  clothes = 54 →
  gift = 37 →
  book = 25 →
  donation = 10 →
  let october : ℚ := august / 2
  let november : ℚ := 2 * september - 20
  let total_savings : ℚ := june + july + august + september + october + november
  let total_expenses : ℚ := school_supplies + clothes + gift + book + donation
  let aunt_bonus : ℚ := if total_savings > 200 ∧ donation = 10 then 25 else 0
  total_savings - total_expenses + aunt_bonus = 97.5 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_savings_problem_l1405_140593


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1405_140580

theorem solution_set_quadratic_inequality :
  {x : ℝ | x * (x - 1) > 0} = Set.Iio 0 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1405_140580


namespace NUMINAMATH_CALUDE_negative_expression_l1405_140549

/-- Given real numbers U, V, W, X, and Y with the following properties:
    U and W are negative,
    V and Y are positive,
    X is near zero (small in absolute value),
    prove that U - V is negative. -/
theorem negative_expression (U V W X Y : ℝ) 
  (hU : U < 0) (hW : W < 0) 
  (hV : V > 0) (hY : Y > 0) 
  (hX : ∃ ε > 0, abs X < ε ∧ ε < 1) : 
  U - V < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expression_l1405_140549


namespace NUMINAMATH_CALUDE_flour_scoops_to_remove_l1405_140578

-- Define the constants
def total_flour : ℚ := 8
def needed_flour : ℚ := 6
def scoop_size : ℚ := 1/4

-- Theorem statement
theorem flour_scoops_to_remove : 
  (total_flour - needed_flour) / scoop_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_flour_scoops_to_remove_l1405_140578


namespace NUMINAMATH_CALUDE_library_visitors_l1405_140569

def sunday_visitors (avg_non_sunday : ℕ) (avg_total : ℕ) (days_in_month : ℕ) : ℕ :=
  let sundays := (days_in_month + 6) / 7
  let non_sundays := days_in_month - sundays
  ((avg_total * days_in_month) - (avg_non_sunday * non_sundays)) / sundays

theorem library_visitors :
  sunday_visitors 240 285 30 = 510 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l1405_140569


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1405_140512

/-- Calculates the speed of a train given its length, time to cross a bridge, and total length of bridge and train. -/
theorem train_speed_calculation 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130) 
  (h2 : crossing_time = 30) 
  (h3 : total_length = 245) : 
  (total_length - train_length) / crossing_time * 3.6 = 45 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1405_140512


namespace NUMINAMATH_CALUDE_garden_breadth_l1405_140596

/-- The breadth of a rectangular garden with perimeter 600 meters and length 100 meters is 200 meters. -/
theorem garden_breadth (perimeter length breadth : ℝ) 
  (h1 : perimeter = 600)
  (h2 : length = 100)
  (h3 : perimeter = 2 * (length + breadth)) : 
  breadth = 200 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l1405_140596


namespace NUMINAMATH_CALUDE_probability_sum_11_three_dice_l1405_140565

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The target sum we're looking for -/
def targetSum : ℕ := 11

/-- The number of dice being rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of ways to roll a sum of 11 with three dice -/
def favorableOutcomes : ℕ := 24

/-- The probability of rolling a sum of 11 with three standard six-sided dice is 1/9 -/
theorem probability_sum_11_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_sum_11_three_dice_l1405_140565


namespace NUMINAMATH_CALUDE_greg_needs_33_more_l1405_140592

/-- The cost of the scooter in dollars -/
def scooter_cost : ℕ := 90

/-- The amount Greg has saved in dollars -/
def greg_savings : ℕ := 57

/-- The additional amount Greg needs to buy the scooter -/
def additional_amount_needed : ℕ := scooter_cost - greg_savings

/-- Theorem stating that the additional amount Greg needs is $33 -/
theorem greg_needs_33_more :
  additional_amount_needed = 33 :=
by sorry

end NUMINAMATH_CALUDE_greg_needs_33_more_l1405_140592


namespace NUMINAMATH_CALUDE_hramps_are_frafs_and_grups_l1405_140590

-- Define the sets
variable (Erogs Frafs Grups Hramps : Set α)

-- Define the conditions
variable (h1 : Erogs ⊆ Frafs)
variable (h2 : Grups ⊆ Frafs)
variable (h3 : Hramps ⊆ Erogs)
variable (h4 : Hramps ⊆ Grups)
variable (h5 : ∃ x, x ∈ Frafs ∧ x ∈ Grups)

-- Theorem to prove
theorem hramps_are_frafs_and_grups :
  Hramps ⊆ Frafs ∧ Hramps ⊆ Grups :=
sorry

end NUMINAMATH_CALUDE_hramps_are_frafs_and_grups_l1405_140590


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1405_140583

theorem infinite_series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n => 1 / ((2 * (n - 1) * a - (n - 2) * b) * (2 * n * a - (n - 1) * b))
  ∑' n, series n = 1 / ((2 * a - b) * 2 * b) :=
sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1405_140583


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l1405_140584

theorem restaurant_bill_theorem :
  let num_people : ℕ := 7
  let regular_spend : ℕ := 11
  let num_regular : ℕ := 6
  let extra_spend : ℕ := 6
  let total_spend : ℕ := regular_spend * num_regular + 
    (regular_spend * num_regular + (total_spend / num_people + extra_spend))
  total_spend = 84 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l1405_140584
