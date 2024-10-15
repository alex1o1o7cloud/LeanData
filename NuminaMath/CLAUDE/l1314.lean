import Mathlib

namespace NUMINAMATH_CALUDE_max_plus_min_equals_zero_l1314_131497

def f (x : ℝ) := x^3 - 3*x

theorem max_plus_min_equals_zero :
  ∀ m n : ℝ,
  (∀ x : ℝ, f x ≤ m) →
  (∃ x : ℝ, f x = m) →
  (∀ x : ℝ, n ≤ f x) →
  (∃ x : ℝ, f x = n) →
  m + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_zero_l1314_131497


namespace NUMINAMATH_CALUDE_overall_length_is_13_l1314_131460

/-- The length of each ruler in centimeters -/
def ruler_length : ℝ := 10

/-- The mark on the first ruler that aligns with the second ruler -/
def align_mark1 : ℝ := 3

/-- The mark on the second ruler that aligns with the first ruler -/
def align_mark2 : ℝ := 4

/-- The overall length when the rulers are aligned as described -/
def L : ℝ := ruler_length + (ruler_length - align_mark2) - (align_mark2 - align_mark1)

theorem overall_length_is_13 : L = 13 := by
  sorry

end NUMINAMATH_CALUDE_overall_length_is_13_l1314_131460


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1314_131456

theorem walking_speed_problem (x : ℝ) :
  let james_speed := x^2 - 13*x - 30
  let jane_distance := x^2 - 5*x - 66
  let jane_time := x + 6
  let jane_speed := jane_distance / jane_time
  james_speed = jane_speed → james_speed = -4 + 2 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l1314_131456


namespace NUMINAMATH_CALUDE_base_salary_calculation_l1314_131478

/-- Proves that the base salary in the second option is $1600 given the conditions of the problem. -/
theorem base_salary_calculation (monthly_salary : ℝ) (commission_rate : ℝ) (equal_sales : ℝ) 
  (h1 : monthly_salary = 1800)
  (h2 : commission_rate = 0.04)
  (h3 : equal_sales = 5000)
  (h4 : ∃ (base_salary : ℝ), base_salary + commission_rate * equal_sales = monthly_salary) :
  ∃ (base_salary : ℝ), base_salary = 1600 ∧ base_salary + commission_rate * equal_sales = monthly_salary :=
by sorry

end NUMINAMATH_CALUDE_base_salary_calculation_l1314_131478


namespace NUMINAMATH_CALUDE_classroom_notebooks_l1314_131491

theorem classroom_notebooks (total_students : ℕ) 
  (notebooks_group1 : ℕ) (notebooks_group2 : ℕ) : 
  total_students = 28 →
  notebooks_group1 = 5 →
  notebooks_group2 = 3 →
  (total_students / 2 * notebooks_group1 + total_students / 2 * notebooks_group2) = 112 := by
  sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l1314_131491


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1314_131427

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 3 → 
  e = -a - c → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 2 * Complex.I → 
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1314_131427


namespace NUMINAMATH_CALUDE_cost_of_dozen_rolls_l1314_131417

/-- The cost of a dozen rolls given the total spent and number of rolls purchased -/
theorem cost_of_dozen_rolls (total_spent : ℚ) (total_rolls : ℕ) (h1 : total_spent = 15) (h2 : total_rolls = 36) : 
  total_spent / (total_rolls / 12 : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_rolls_l1314_131417


namespace NUMINAMATH_CALUDE_point_x_coordinate_l1314_131488

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_x_coordinate 
  (l : Line) 
  (p : Point) 
  (h1 : l.slope = 3.8666666666666667)
  (h2 : l.yIntercept = 20)
  (h3 : p.y = 600)
  (h4 : pointOnLine p l) :
  p.x = 150 :=
sorry

end NUMINAMATH_CALUDE_point_x_coordinate_l1314_131488


namespace NUMINAMATH_CALUDE_hexagonal_prism_lateral_area_l1314_131482

/-- The lateral surface area of a hexagonal prism with regular hexagon base -/
def lateralSurfaceArea (baseSideLength : ℝ) (lateralEdgeLength : ℝ) : ℝ :=
  6 * baseSideLength * lateralEdgeLength

/-- Theorem: The lateral surface area of a hexagonal prism with base side length 3 and lateral edge length 4 is 72 -/
theorem hexagonal_prism_lateral_area :
  lateralSurfaceArea 3 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_lateral_area_l1314_131482


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l1314_131463

theorem union_necessary_not_sufficient_for_intersection (A B : Set α) :
  (∀ x, x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ A ∩ B) :=
sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l1314_131463


namespace NUMINAMATH_CALUDE_rectangle_area_l1314_131414

/-- The area of a rectangle with length 4 cm and width 2 cm is 8 cm² -/
theorem rectangle_area : 
  ∀ (length width area : ℝ),
  length = 4 →
  width = 2 →
  area = length * width →
  area = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1314_131414


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1314_131474

theorem trigonometric_identity (θ : ℝ) (h : Real.sin (3 * π / 2 + θ) = 1 / 4) :
  (Real.cos (π + θ)) / (Real.cos θ * (Real.cos (π + θ) - 1)) +
  (Real.cos (θ - 2 * π)) / (Real.cos (θ + 2 * π) * Real.cos (θ + π) + Real.cos (-θ)) = 32 / 15 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1314_131474


namespace NUMINAMATH_CALUDE_division_error_problem_l1314_131466

theorem division_error_problem (x : ℝ) (y : ℝ) (h : y > 0) :
  (abs (5 * x - x / y) / (5 * x)) * 100 = 98 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_error_problem_l1314_131466


namespace NUMINAMATH_CALUDE_tape_overlap_l1314_131471

theorem tape_overlap (tape_length : ℕ) (total_length : ℕ) (h1 : tape_length = 275) (h2 : total_length = 512) :
  2 * tape_length - total_length = 38 := by
  sorry

end NUMINAMATH_CALUDE_tape_overlap_l1314_131471


namespace NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l1314_131472

/-- Represents the state of the Ring Mafia game -/
structure GameState where
  total_counters : ℕ
  mafia_counters : ℕ
  town_counters : ℕ

/-- Defines a valid initial state for the Ring Mafia game -/
def valid_initial_state (state : GameState) : Prop :=
  state.total_counters ≥ 3 ∧
  state.total_counters % 2 = 1 ∧
  state.mafia_counters = (state.total_counters - 1) / 3 ∧
  state.town_counters = 2 * (state.total_counters - 1) / 3 ∧
  state.mafia_counters + state.town_counters = state.total_counters

/-- Defines a winning state for Tony -/
def tony_wins (state : GameState) : Prop :=
  state.town_counters > 0 ∧ state.mafia_counters = 0

/-- Represents a strategy for Tony -/
def TonyStrategy := GameState → Set ℕ

/-- Defines the concept of a winning strategy for Tony -/
def winning_strategy (strategy : TonyStrategy) : Prop :=
  ∀ (initial_state : GameState),
    valid_initial_state initial_state →
    ∃ (final_state : GameState),
      tony_wins final_state

/-- The main theorem: Tony does not have a winning strategy -/
theorem no_winning_strategy_for_tony :
  ¬∃ (strategy : TonyStrategy), winning_strategy strategy :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l1314_131472


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l1314_131442

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l1314_131442


namespace NUMINAMATH_CALUDE_sin_30_minus_one_plus_pi_to_zero_l1314_131495

theorem sin_30_minus_one_plus_pi_to_zero (h1 : Real.sin (30 * π / 180) = 1 / 2) 
  (h2 : ∀ x : ℝ, x ^ (0 : ℝ) = 1) : 
  Real.sin (30 * π / 180) - (1 + π) ^ (0 : ℝ) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_minus_one_plus_pi_to_zero_l1314_131495


namespace NUMINAMATH_CALUDE_henry_final_book_count_l1314_131400

def initial_books : ℕ := 99
def boxes_donated : ℕ := 3
def books_per_box : ℕ := 15
def room_books : ℕ := 21
def coffee_table_books : ℕ := 4
def kitchen_books : ℕ := 18
def free_books_taken : ℕ := 12

theorem henry_final_book_count :
  initial_books - 
  (boxes_donated * books_per_box + room_books + coffee_table_books + kitchen_books) + 
  free_books_taken = 23 := by
  sorry

end NUMINAMATH_CALUDE_henry_final_book_count_l1314_131400


namespace NUMINAMATH_CALUDE_magician_earnings_calculation_l1314_131468

/-- The amount of money earned by a magician selling card decks -/
def magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price_per_deck

/-- Theorem: The magician earns $56 -/
theorem magician_earnings_calculation :
  magician_earnings 7 16 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_calculation_l1314_131468


namespace NUMINAMATH_CALUDE_evening_screen_time_l1314_131477

-- Define the total recommended screen time in hours
def total_screen_time_hours : ℕ := 2

-- Define the screen time already used in minutes
def morning_screen_time : ℕ := 45

-- Define the function to calculate remaining screen time
def remaining_screen_time (total_hours : ℕ) (used_minutes : ℕ) : ℕ :=
  total_hours * 60 - used_minutes

-- Theorem statement
theorem evening_screen_time :
  remaining_screen_time total_screen_time_hours morning_screen_time = 75 := by
  sorry

end NUMINAMATH_CALUDE_evening_screen_time_l1314_131477


namespace NUMINAMATH_CALUDE_total_selections_exactly_three_girls_at_most_three_girls_both_boys_and_girls_l1314_131473

-- Define the number of boys and girls
def num_boys : ℕ := 8
def num_girls : ℕ := 5
def total_people : ℕ := num_boys + num_girls
def selection_size : ℕ := 6

-- (1) Total number of ways to select 6 people
theorem total_selections : Nat.choose total_people selection_size = 1716 := by sorry

-- (2) Number of ways to select exactly 3 girls
theorem exactly_three_girls : 
  Nat.choose num_girls 3 * Nat.choose num_boys 3 = 560 := by sorry

-- (3) Number of ways to select at most 3 girls
theorem at_most_three_girls : 
  Nat.choose num_boys 6 + 
  Nat.choose num_boys 5 * Nat.choose num_girls 1 + 
  Nat.choose num_boys 4 * Nat.choose num_girls 2 + 
  Nat.choose num_boys 3 * Nat.choose num_girls 3 = 1568 := by sorry

-- (4) Number of ways to select both boys and girls
theorem both_boys_and_girls : 
  Nat.choose total_people selection_size - Nat.choose num_boys selection_size = 1688 := by sorry

end NUMINAMATH_CALUDE_total_selections_exactly_three_girls_at_most_three_girls_both_boys_and_girls_l1314_131473


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1314_131496

open Real

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ sin (arccos (tan (arcsin x))) = x :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1314_131496


namespace NUMINAMATH_CALUDE_additional_male_workers_l1314_131455

theorem additional_male_workers (initial_female_percent : ℚ) 
                                (final_female_percent : ℚ) 
                                (final_total : ℕ) : ℕ :=
  let initial_female_percent := 60 / 100
  let final_female_percent := 55 / 100
  let final_total := 312
  26

#check additional_male_workers

end NUMINAMATH_CALUDE_additional_male_workers_l1314_131455


namespace NUMINAMATH_CALUDE_total_pencils_after_operations_l1314_131489

/-- 
Given:
- There are initially 43 pencils in a drawer
- There are initially 19 pencils on a desk
- 16 pencils are added to the desk
- 7 pencils are removed from the desk

Prove that the total number of pencils after these operations is 71.
-/
theorem total_pencils_after_operations : 
  ∀ (drawer_initial desk_initial added removed : ℕ),
    drawer_initial = 43 →
    desk_initial = 19 →
    added = 16 →
    removed = 7 →
    drawer_initial + (desk_initial + added - removed) = 71 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pencils_after_operations_l1314_131489


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1314_131410

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (4 * x) * (x + 4) = 2 * (4 * x) + 2 * (x + 4) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1314_131410


namespace NUMINAMATH_CALUDE_magazine_subscription_cost_l1314_131458

theorem magazine_subscription_cost (reduction_percentage : ℝ) (reduction_amount : ℝ) (original_cost : ℝ) : 
  reduction_percentage = 0.30 → 
  reduction_amount = 588 → 
  reduction_percentage * original_cost = reduction_amount → 
  original_cost = 1960 := by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_cost_l1314_131458


namespace NUMINAMATH_CALUDE_first_player_wins_first_player_wins_modified_l1314_131437

/-- Represents the state of the game with two piles of stones -/
structure GameState :=
  (pile1 : Nat)
  (pile2 : Nat)

/-- Represents a move in the game -/
inductive Move
  | TakeFromFirst
  | TakeFromSecond
  | TakeFromBoth
  | TransferToSecond

/-- Defines if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.TakeFromFirst => state.pile1 > 0
  | Move.TakeFromSecond => state.pile2 > 0
  | Move.TakeFromBoth => state.pile1 > 0 && state.pile2 > 0
  | Move.TransferToSecond => state.pile1 > 0

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeFromFirst => ⟨state.pile1 - 1, state.pile2⟩
  | Move.TakeFromSecond => ⟨state.pile1, state.pile2 - 1⟩
  | Move.TakeFromBoth => ⟨state.pile1 - 1, state.pile2 - 1⟩
  | Move.TransferToSecond => ⟨state.pile1 - 1, state.pile2 + 1⟩

/-- Determines if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Bool :=
  state.pile1 = 0 && state.pile2 = 0

/-- Theorem: The first player has a winning strategy in the two-pile stone game -/
theorem first_player_wins (initialState : GameState) 
  (h : initialState = ⟨7, 7⟩) : 
  ∃ (strategy : GameState → Move), 
    ∀ (opponentMove : Move), 
      isValidMove initialState (strategy initialState) ∧ 
      ¬isGameOver (applyMove initialState (strategy initialState)) ∧
      isGameOver (applyMove (applyMove initialState (strategy initialState)) opponentMove) :=
sorry

/-- Theorem: The first player has a winning strategy in the modified two-pile stone game -/
theorem first_player_wins_modified (initialState : GameState) 
  (h : initialState = ⟨7, 7⟩) : 
  ∃ (strategy : GameState → Move), 
    ∀ (opponentMove : Move), 
      isValidMove initialState (strategy initialState) ∧ 
      ¬isGameOver (applyMove initialState (strategy initialState)) ∧
      isGameOver (applyMove (applyMove initialState (strategy initialState)) opponentMove) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_first_player_wins_modified_l1314_131437


namespace NUMINAMATH_CALUDE_magic_numbers_theorem_l1314_131408

/-- Represents the numbers chosen by three people -/
structure Numbers where
  ana : ℕ
  beto : ℕ
  caio : ℕ

/-- Performs one round of exchange -/
def exchange (n : Numbers) : Numbers :=
  { ana := n.beto + n.caio
  , beto := n.ana + n.caio
  , caio := n.ana + n.beto }

/-- The theorem to prove -/
theorem magic_numbers_theorem (initial : Numbers) :
  1 ≤ initial.ana ∧ initial.ana ≤ 50 ∧
  1 ≤ initial.beto ∧ initial.beto ≤ 50 ∧
  1 ≤ initial.caio ∧ initial.caio ≤ 50 →
  let second := exchange initial
  let final := exchange second
  final.ana = 104 ∧ final.beto = 123 ∧ final.caio = 137 →
  initial.ana = 13 ∧ initial.beto = 32 ∧ initial.caio = 46 :=
by
  sorry


end NUMINAMATH_CALUDE_magic_numbers_theorem_l1314_131408


namespace NUMINAMATH_CALUDE_m_eq_two_iff_z_on_y_eq_x_l1314_131446

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := 1 + ((-1 + m) * Complex.I)

-- Define the condition for a point to lie on the line y = x
def lies_on_y_eq_x (z : ℂ) : Prop := z.im = z.re

-- State the theorem
theorem m_eq_two_iff_z_on_y_eq_x :
  ∀ m : ℝ, (m = 2) ↔ lies_on_y_eq_x (z m) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_two_iff_z_on_y_eq_x_l1314_131446


namespace NUMINAMATH_CALUDE_candle_burn_time_l1314_131475

/-- Given that a candle lasts 8 nights and burning it for 2 hours a night uses 6 candles over 24 nights,
    prove that Carmen burns the candle for 1 hour every night in the first scenario. -/
theorem candle_burn_time (candle_duration : ℕ) (nights_per_candle : ℕ) (burn_time_second_scenario : ℕ) 
  (candles_used : ℕ) (total_nights : ℕ) :
  candle_duration = 8 ∧ 
  nights_per_candle = 8 ∧
  burn_time_second_scenario = 2 ∧
  candles_used = 6 ∧
  total_nights = 24 →
  ∃ (burn_time_first_scenario : ℕ), burn_time_first_scenario = 1 :=
by sorry

end NUMINAMATH_CALUDE_candle_burn_time_l1314_131475


namespace NUMINAMATH_CALUDE_M_on_x_axis_M_parallel_to_x_axis_M_distance_from_y_axis_l1314_131440

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a real number m, construct a point M with coordinates (m-1, 2m+3) -/
def M (m : ℝ) : Point := ⟨m - 1, 2 * m + 3⟩

/-- N is a fixed point with coordinates (5, -1) -/
def N : Point := ⟨5, -1⟩

theorem M_on_x_axis (m : ℝ) : 
  M m = ⟨-5/2, 0⟩ ↔ (M m).y = 0 := by sorry

theorem M_parallel_to_x_axis (m : ℝ) :
  M m = ⟨-3, -1⟩ ↔ (M m).y = N.y := by sorry

theorem M_distance_from_y_axis (m : ℝ) :
  (M m = ⟨2, 9⟩ ∨ M m = ⟨-2, 1⟩) ↔ |(M m).x| = 2 := by sorry

end NUMINAMATH_CALUDE_M_on_x_axis_M_parallel_to_x_axis_M_distance_from_y_axis_l1314_131440


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1314_131441

theorem system_of_inequalities_solution (x : ℝ) : 
  (5 / (x + 3) ≥ 1 ∧ x^2 + x - 2 ≥ 0) ↔ ((-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1314_131441


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l1314_131484

/-- A geometric sequence with a_3 = 16 and a_5 = 4 has a_7 = 1 -/
theorem geometric_sequence_a7 (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 3 = 16 →                                 -- given a_3 = 16
  a 5 = 4 →                                  -- given a_5 = 4
  a 7 = 1 :=                                 -- to prove a_7 = 1
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a7_l1314_131484


namespace NUMINAMATH_CALUDE_largest_number_l1314_131419

theorem largest_number (a b c d : ℝ) (h1 : a = Real.sqrt 5) (h2 : b = -1.6) (h3 : c = 0) (h4 : d = 2) :
  max a (max b (max c d)) = a :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1314_131419


namespace NUMINAMATH_CALUDE_test_score_properties_l1314_131453

/-- A test with multiple-choice questions. -/
structure Test where
  num_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ
  max_score : ℕ
  prob_correct : ℝ

/-- Calculate the expected score for a given test. -/
def expected_score (t : Test) : ℝ :=
  t.num_questions * (t.correct_points * t.prob_correct + t.incorrect_points * (1 - t.prob_correct))

/-- Calculate the variance of scores for a given test. -/
def score_variance (t : Test) : ℝ :=
  t.num_questions * (t.correct_points^2 * t.prob_correct + t.incorrect_points^2 * (1 - t.prob_correct) - 
    (t.correct_points * t.prob_correct + t.incorrect_points * (1 - t.prob_correct))^2)

/-- Theorem stating the expected score and variance for the given test conditions. -/
theorem test_score_properties :
  ∃ (t : Test),
    t.num_questions = 25 ∧
    t.correct_points = 4 ∧
    t.incorrect_points = 0 ∧
    t.max_score = 100 ∧
    t.prob_correct = 0.8 ∧
    expected_score t = 80 ∧
    score_variance t = 64 := by
  sorry

end NUMINAMATH_CALUDE_test_score_properties_l1314_131453


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1314_131432

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / 6 - x^2 / 2 = 1

/-- Theorem stating that a hyperbola with the given properties has the specified standard equation -/
theorem hyperbola_standard_equation (h : Hyperbola) 
    (h_slope : h.asymptote_slope = Real.sqrt 3)
    (h_point : h.point = (-1, 3)) :
    ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | standard_equation h p.1 p.2} ↔ 
    (∃ t : ℝ, y = h.asymptote_slope * x + t ∨ y = -h.asymptote_slope * x + t) ∧
    (x = h.point.1 ∧ y = h.point.2) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1314_131432


namespace NUMINAMATH_CALUDE_negative_two_squared_minus_zero_power_six_m_divided_by_two_m_l1314_131470

-- First problem
theorem negative_two_squared_minus_zero_power : ((-2 : ℤ)^2) - ((-2 : ℤ)^0) = 3 := by sorry

-- Second problem
theorem six_m_divided_by_two_m (m : ℝ) (hm : m ≠ 0) : (6 * m) / (2 * m) = 3 := by sorry

end NUMINAMATH_CALUDE_negative_two_squared_minus_zero_power_six_m_divided_by_two_m_l1314_131470


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1314_131483

-- Define the triangle ABC
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  a + b = 11 →
  c = 7 →
  Real.cos A = -1/7 →
  -- Conclusions to prove
  a = 8 ∧
  Real.sin C = Real.sqrt 3 / 2 ∧
  (1/2 : ℝ) * a * b * Real.sin C = 6 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l1314_131483


namespace NUMINAMATH_CALUDE_double_scientific_notation_l1314_131412

theorem double_scientific_notation : 
  let x : ℝ := 1.2 * (10 ^ 6)
  2 * x = 2.4 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_double_scientific_notation_l1314_131412


namespace NUMINAMATH_CALUDE_equation_solutions_l1314_131434

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = 7 ∧ 
    ∀ x : ℝ, 3 * (x - 4) = (x - 4)^2 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (-1 + Real.sqrt 10) / 3 ∧ y₂ = (-1 - Real.sqrt 10) / 3 ∧ 
    ∀ x : ℝ, 3 * x^2 + 2 * x - 3 = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1314_131434


namespace NUMINAMATH_CALUDE_mika_initial_stickers_l1314_131407

/-- Represents the number of stickers Mika has at different stages --/
structure StickerCount where
  initial : ℕ
  after_buying : ℕ
  after_birthday : ℕ
  after_giving : ℕ
  after_decorating : ℕ
  final : ℕ

/-- Defines the sticker transactions Mika goes through --/
def sticker_transactions (s : StickerCount) : Prop :=
  s.after_buying = s.initial + 26 ∧
  s.after_birthday = s.after_buying + 20 ∧
  s.after_giving = s.after_birthday - 6 ∧
  s.after_decorating = s.after_giving - 58 ∧
  s.final = s.after_decorating ∧
  s.final = 2

/-- Theorem stating that Mika initially had 20 stickers --/
theorem mika_initial_stickers :
  ∃ (s : StickerCount), sticker_transactions s ∧ s.initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_mika_initial_stickers_l1314_131407


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_l1314_131462

/-- Check if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 7 + 10 * m

/-- The smallest natural number n such that both n^2 and (n+1)^2 contain the digit 7 -/
theorem smallest_n_with_seven : ∀ n : ℕ, n < 26 →
  ¬(containsSeven (n^2) ∧ containsSeven ((n+1)^2)) ∧
  (containsSeven (26^2) ∧ containsSeven (27^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_l1314_131462


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l1314_131403

theorem crushing_load_calculation (T H D : ℝ) (hT : T = 5) (hH : H = 15) (hD : D = 10) :
  let L := (30 * T^3) / (H * D)
  L = 25 := by
sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l1314_131403


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1314_131476

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 1) :
  1/(x+2*y) + 4/(2*y+3*z) + 9/(3*z+x) ≥ 18 := by
sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ 
  1/(x+2*y) + 4/(2*y+3*z) + 9/(3*z+x) < 18 + ε := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1314_131476


namespace NUMINAMATH_CALUDE_jason_climbing_speed_l1314_131433

/-- Given that Matt climbs at 6 feet per minute and Jason is 42 feet higher than Matt after 7 minutes,
    prove that Jason's climbing speed is 12 feet per minute. -/
theorem jason_climbing_speed (matt_speed : ℝ) (time : ℝ) (height_difference : ℝ) :
  matt_speed = 6 →
  time = 7 →
  height_difference = 42 →
  (time * matt_speed + height_difference) / time = 12 := by
sorry

end NUMINAMATH_CALUDE_jason_climbing_speed_l1314_131433


namespace NUMINAMATH_CALUDE_product_sale_result_l1314_131431

def cost_price : ℝ := 100
def markup_percentage : ℝ := 0.2
def discount_percentage : ℝ := 0.2
def final_selling_price : ℝ := 96

theorem product_sale_result :
  let initial_price := cost_price * (1 + markup_percentage)
  let discounted_price := initial_price * (1 - discount_percentage)
  discounted_price = final_selling_price ∧ 
  cost_price - final_selling_price = 4 := by
sorry

end NUMINAMATH_CALUDE_product_sale_result_l1314_131431


namespace NUMINAMATH_CALUDE_hash_seven_three_l1314_131421

/-- The # operation on real numbers -/
noncomputable def hash (x y : ℝ) : ℝ :=
  sorry

/-- The first condition: x # 0 = x -/
axiom hash_zero (x : ℝ) : hash x 0 = x

/-- The second condition: x # y = y # x -/
axiom hash_comm (x y : ℝ) : hash x y = hash y x

/-- The third condition: (x + 1) # y = (x # y) + 2y + 1 -/
axiom hash_succ (x y : ℝ) : hash (x + 1) y = hash x y + 2 * y + 1

/-- The main theorem: 7 # 3 = 52 -/
theorem hash_seven_three : hash 7 3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_hash_seven_three_l1314_131421


namespace NUMINAMATH_CALUDE_range_of_a_l1314_131430

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - 2 > a^2 - 3*a) → a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1314_131430


namespace NUMINAMATH_CALUDE_polynomial_value_l1314_131494

theorem polynomial_value (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) :
  -2 * x^2 - 3 * y + 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l1314_131494


namespace NUMINAMATH_CALUDE_field_trip_bus_capacity_l1314_131493

theorem field_trip_bus_capacity 
  (total_vehicles : Nat) 
  (people_per_van : Nat) 
  (total_people : Nat) 
  (num_vans : Nat) 
  (num_buses : Nat) 
  (h1 : total_vehicles = num_vans + num_buses)
  (h2 : num_vans = 2)
  (h3 : num_buses = 3)
  (h4 : people_per_van = 8)
  (h5 : total_people = 76) :
  (total_people - num_vans * people_per_van) / num_buses = 20 := by
sorry

end NUMINAMATH_CALUDE_field_trip_bus_capacity_l1314_131493


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1314_131465

theorem inequality_solution_set (x : ℝ) : 
  (3 * x^2) / (1 - (3*x + 1)^(1/3))^2 ≤ x + 2 + (3*x + 1)^(1/3) → 
  -2/3 ≤ x ∧ x < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1314_131465


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1314_131404

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours : ℝ := 54

def overtime_rate : ℝ := regular_rate * (1 + overtime_rate_increase)
def overtime_hours : ℝ := total_hours - regular_hours

def regular_pay : ℝ := regular_rate * regular_hours
def overtime_pay : ℝ := overtime_rate * overtime_hours
def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_compensation :
  total_compensation = 1032 := by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l1314_131404


namespace NUMINAMATH_CALUDE_average_speed_first_half_l1314_131420

theorem average_speed_first_half (total_distance : ℝ) (total_avg_speed : ℝ) : 
  total_distance = 640 →
  total_avg_speed = 40 →
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let first_half_time := first_half_distance / (first_half_distance / (total_distance / (4 * total_avg_speed)))
  let second_half_time := 3 * first_half_time
  first_half_distance / first_half_time = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_first_half_l1314_131420


namespace NUMINAMATH_CALUDE_second_trip_crates_parameters_are_valid_l1314_131479

/-- The number of crates carried in the second trip of a trailer -/
def crates_in_second_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : ℕ :=
  total_crates - (max_trip_weight / min_crate_weight)

/-- Theorem stating that given the specified conditions, the trailer carries 7 crates in the second trip -/
theorem second_trip_crates :
  crates_in_second_trip 12 120 600 = 7 := by
  sorry

/-- Checks if the given parameters satisfy the problem conditions -/
def valid_parameters (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : Prop :=
  total_crates > 0 ∧
  min_crate_weight > 0 ∧
  max_trip_weight > 0 ∧
  min_crate_weight * total_crates > max_trip_weight

/-- Theorem stating that the given parameters satisfy the problem conditions -/
theorem parameters_are_valid :
  valid_parameters 12 120 600 := by
  sorry

end NUMINAMATH_CALUDE_second_trip_crates_parameters_are_valid_l1314_131479


namespace NUMINAMATH_CALUDE_point_order_on_line_l1314_131498

/-- Proves that for points (-3, y₁), (1, y₂), (-1, y₃) lying on the line y = 3x - b, 
    the relationship y₁ < y₃ < y₂ holds. -/
theorem point_order_on_line (b y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 3 * (-3) - b)
  (h₂ : y₂ = 3 * 1 - b)
  (h₃ : y₃ = 3 * (-1) - b) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_point_order_on_line_l1314_131498


namespace NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l1314_131418

theorem average_chocolate_pieces_per_cookie 
  (num_cookies : ℕ) 
  (num_choc_chips : ℕ) 
  (num_mms : ℕ) 
  (h1 : num_cookies = 48) 
  (h2 : num_choc_chips = 108) 
  (h3 : num_mms = num_choc_chips / 3) : 
  (num_choc_chips + num_mms) / num_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l1314_131418


namespace NUMINAMATH_CALUDE_scientific_notation_of_8790000_l1314_131416

theorem scientific_notation_of_8790000 :
  8790000 = 8.79 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8790000_l1314_131416


namespace NUMINAMATH_CALUDE_max_span_sum_of_digits_div_by_8_l1314_131449

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: Maximum span between numbers with sum of digits divisible by 8 -/
theorem max_span_sum_of_digits_div_by_8 (m : ℕ) (h1 : m > 0) (h2 : sumOfDigits m % 8 = 0) :
  ∃ (n : ℕ), n = 15 ∧
    sumOfDigits (m + n) % 8 = 0 ∧
    ∀ k : ℕ, 1 ≤ k → k < n → sumOfDigits (m + k) % 8 ≠ 0 ∧
    ∀ n' : ℕ, n' > n →
      ¬(sumOfDigits (m + n') % 8 = 0 ∧
        ∀ k : ℕ, 1 ≤ k → k < n' → sumOfDigits (m + k) % 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_span_sum_of_digits_div_by_8_l1314_131449


namespace NUMINAMATH_CALUDE_map_segment_to_yards_l1314_131428

/-- Converts a length in inches on a map to yards in reality, given a scale --/
def map_length_to_yards (map_length : ℚ) (scale : ℚ) : ℚ :=
  (map_length * scale) / 3

/-- The scale of the map (feet per inch) --/
def map_scale : ℚ := 500

/-- The length of the line segment on the map (in inches) --/
def line_segment_length : ℚ := 6.25

/-- Theorem: The 6.25-inch line segment on the map represents 1041 2/3 yards in reality --/
theorem map_segment_to_yards :
  map_length_to_yards line_segment_length map_scale = 1041 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_map_segment_to_yards_l1314_131428


namespace NUMINAMATH_CALUDE_area_of_constrained_region_l1314_131405

/-- The area of the region defined by specific constraints in a coordinate plane --/
theorem area_of_constrained_region : 
  let S := {p : ℝ × ℝ | p.1 ≤ 0 ∧ p.2 + p.1 - 1 ≥ 0 ∧ p.2 ≤ 4}
  MeasureTheory.volume S = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_of_constrained_region_l1314_131405


namespace NUMINAMATH_CALUDE_student_selection_sequences_l1314_131480

theorem student_selection_sequences (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  Nat.descFactorial n k = 30240 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_sequences_l1314_131480


namespace NUMINAMATH_CALUDE_coin_problem_l1314_131454

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 32 ∧ 
  total_value = 47/10 →
  ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧ 
    (1/4 : ℚ) * quarters + (1/10 : ℚ) * dimes = total_value ∧
    quarters = 10 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l1314_131454


namespace NUMINAMATH_CALUDE_car_truck_sales_l1314_131490

theorem car_truck_sales (total_vehicles : ℕ) (car_truck_difference : ℕ) : 
  total_vehicles = 69 → car_truck_difference = 27 → 
  ∃ (trucks : ℕ), trucks = 21 ∧ trucks + (trucks + car_truck_difference) = total_vehicles := by
sorry

end NUMINAMATH_CALUDE_car_truck_sales_l1314_131490


namespace NUMINAMATH_CALUDE_floor_sqrt_200_l1314_131443

theorem floor_sqrt_200 : ⌊Real.sqrt 200⌋ = 14 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_200_l1314_131443


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1314_131409

theorem complex_magnitude_example : Complex.abs (-3 + (8/5) * Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1314_131409


namespace NUMINAMATH_CALUDE_total_cost_over_two_years_l1314_131435

/-- Represents the number of games attended and their types -/
structure GameAttendance where
  home : Nat
  away : Nat
  homePlayoff : Nat
  awayPlayoff : Nat

/-- Represents the ticket prices for different game types -/
structure TicketPrices where
  home : Nat
  away : Nat
  homePlayoff : Nat
  awayPlayoff : Nat

/-- Calculates the total cost for a given year -/
def calculateYearlyCost (attendance : GameAttendance) (prices : TicketPrices) : Nat :=
  attendance.home * prices.home +
  attendance.away * prices.away +
  attendance.homePlayoff * prices.homePlayoff +
  attendance.awayPlayoff * prices.awayPlayoff

/-- Theorem stating the total cost over two years -/
theorem total_cost_over_two_years
  (prices : TicketPrices)
  (thisYear : GameAttendance)
  (lastYear : GameAttendance)
  (h1 : prices.home = 60)
  (h2 : prices.away = 75)
  (h3 : prices.homePlayoff = 120)
  (h4 : prices.awayPlayoff = 100)
  (h5 : thisYear.home = 2)
  (h6 : thisYear.away = 2)
  (h7 : thisYear.homePlayoff = 1)
  (h8 : thisYear.awayPlayoff = 0)
  (h9 : lastYear.home = 6)
  (h10 : lastYear.away = 3)
  (h11 : lastYear.homePlayoff = 1)
  (h12 : lastYear.awayPlayoff = 1) :
  calculateYearlyCost thisYear prices + calculateYearlyCost lastYear prices = 1195 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_over_two_years_l1314_131435


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_three_l1314_131492

def A : Set ℝ := {x | |x - 2| < 1}
def B (a : ℝ) : Set ℝ := {y | ∃ x, y = -x^2 + a}

theorem subset_implies_a_geq_three (a : ℝ) (h : A ⊆ B a) : a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_three_l1314_131492


namespace NUMINAMATH_CALUDE_angle_identity_l1314_131457

theorem angle_identity (α : Real) (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : ∃ (x y : Real), x = Real.sin (215 * Real.pi / 180) ∧ 
                        y = Real.cos (215 * Real.pi / 180) ∧ 
                        x = Real.sin α ∧ 
                        y = Real.cos α) : 
  α = 235 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_identity_l1314_131457


namespace NUMINAMATH_CALUDE_line_equation_sum_l1314_131451

/-- Given a line with slope -4 passing through the point (5, 2), 
    prove that if its equation is of the form y = mx + b, then m + b = 18 -/
theorem line_equation_sum (m b : ℝ) : 
  m = -4 → 
  2 = m * 5 + b → 
  m + b = 18 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_sum_l1314_131451


namespace NUMINAMATH_CALUDE_sequence_properties_l1314_131429

/-- Sequence of integers defined by a recursive formula -/
def a : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * a (n + 1) - a n

/-- Theorem stating the properties of the sequence -/
theorem sequence_properties :
  ∀ n : ℕ,
    a (n + 1) > a n ∧
    Nat.gcd (a n) (a (n + 1)) = 1 ∧
    (a n ∣ a (n + 1)^2 - 5) ∧
    (a (n + 1) ∣ a n^2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1314_131429


namespace NUMINAMATH_CALUDE_inequality_proof_l1314_131486

theorem inequality_proof (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1314_131486


namespace NUMINAMATH_CALUDE_negation_equivalence_l1314_131436

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 - 2*x > 0) ↔ (∀ x : ℝ, x < 0 → x^2 - 2*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1314_131436


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_cubes_divisible_by_504_l1314_131481

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℕ) :
  ∃ k : ℕ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_cubes_divisible_by_504_l1314_131481


namespace NUMINAMATH_CALUDE_probability_all_red_in_hat_l1314_131438

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
  | AllRed
  | TwoGreen

/-- The probability of drawing all red chips before two green chips -/
def probability_all_red (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of drawing all red chips -/
theorem probability_all_red_in_hat :
  probability_all_red 7 4 3 = 1/7 :=
sorry

end NUMINAMATH_CALUDE_probability_all_red_in_hat_l1314_131438


namespace NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l1314_131499

/-- A polyhedron is represented as a set of points in ℝ³. -/
def Polyhedron : Type := Set (ℝ × ℝ × ℝ)

/-- The edges of a polyhedron. -/
def edges (P : Polyhedron) : Set (Set (ℝ × ℝ × ℝ)) := sorry

/-- The length of an edge. -/
def edgeLength (e : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The sum of all edge lengths in a polyhedron. -/
def sumEdgeLengths (P : Polyhedron) : ℝ := sorry

/-- The distance between two points in ℝ³. -/
def distance (p q : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The maximum distance between any two points in a polyhedron. -/
def maxDistance (P : Polyhedron) : ℝ := sorry

/-- Theorem: The sum of edge lengths is at least 3 times the maximum distance. -/
theorem polyhedron_edge_length_bound (P : Polyhedron) :
  sumEdgeLengths P ≥ 3 * maxDistance P := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l1314_131499


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1314_131423

/-- Given a geometric sequence with positive terms where (a_3, 1/2*a_5, a_4) form an arithmetic sequence,
    prove that (a_3 + a_5) / (a_4 + a_6) = (√5 - 1) / 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : a 3 + a 4 = a 5) :
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1314_131423


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l1314_131401

def ice_cream_sales (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a + b + c + d = n ∧
    b = (n - a + 1) / 2 ∧
    c = ((n - a - b + 1) / 2 : ℕ) ∧
    d = ((n - a - b - c + 1) / 2 : ℕ) ∧
    d = 1

theorem ice_cream_theorem :
  ∀ n : ℕ, ice_cream_sales n → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l1314_131401


namespace NUMINAMATH_CALUDE_expression_simplification_l1314_131461

theorem expression_simplification 
  (a b c d x y : ℝ) 
  (h : c * x ≠ d * y) : 
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) - 
   d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / 
  (c * x - d * y) = 
  b^2 * x^2 + a^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1314_131461


namespace NUMINAMATH_CALUDE_sequence_general_term_l1314_131469

theorem sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 3) : 
  ∀ n, a n = 3 * 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1314_131469


namespace NUMINAMATH_CALUDE_exists_column_with_many_zeros_l1314_131487

/-- Represents a row in the grid -/
def Row := Fin 6 → Fin 2

/-- The grid -/
def Grid (n : ℕ) := Fin n → Row

/-- Condition: integers in each row are distinct -/
def distinct_rows (g : Grid n) : Prop :=
  ∀ i j, i ≠ j → g i ≠ g j

/-- Condition: for any two rows, their element-wise product exists as a row -/
def product_exists (g : Grid n) : Prop :=
  ∀ i j, ∃ k, ∀ m, g k m = (g i m * g j m : Fin 2)

/-- Count of 0s in a column -/
def zero_count (g : Grid n) (col : Fin 6) : ℕ :=
  (Finset.filter (λ i => g i col = 0) Finset.univ).card

/-- Main theorem -/
theorem exists_column_with_many_zeros (n : ℕ) (hn : n ≥ 2) (g : Grid n)
  (h_distinct : distinct_rows g) (h_product : product_exists g) :
  ∃ col, zero_count g col ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_column_with_many_zeros_l1314_131487


namespace NUMINAMATH_CALUDE_cousin_distribution_l1314_131402

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The number of rooms -/
def num_rooms : ℕ := 4

theorem cousin_distribution :
  distribute num_cousins num_rooms = 52 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_l1314_131402


namespace NUMINAMATH_CALUDE_max_value_ab_l1314_131447

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y ≤ (1/4 : ℝ)) → a * b ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_ab_l1314_131447


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1314_131413

theorem expression_simplification_and_evaluation :
  let x : ℤ := -1
  let original_expression := (x * (x + 1)) - ((x + 2) * (2 - x)) - (2 * (x + 2)^2)
  let simplified_expression := -2 * x^2 - 9 * x - 12
  original_expression = simplified_expression ∧ simplified_expression = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1314_131413


namespace NUMINAMATH_CALUDE_trig_inequality_l1314_131422

theorem trig_inequality : 
  let a := Real.sin (Real.cos (2016 * π / 180))
  let b := Real.sin (Real.sin (2016 * π / 180))
  let c := Real.cos (Real.sin (2016 * π / 180))
  let d := Real.cos (Real.cos (2016 * π / 180))
  c > d ∧ d > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1314_131422


namespace NUMINAMATH_CALUDE_translation_down_3_units_l1314_131415

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 3 * x - 1

def vertical_translation (h : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ h x - d

theorem translation_down_3_units :
  vertical_translation f 3 = g := by sorry

end NUMINAMATH_CALUDE_translation_down_3_units_l1314_131415


namespace NUMINAMATH_CALUDE_cube_root_equivalence_l1314_131445

theorem cube_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/2))^(1/3) = x^(5/6) := by sorry

end NUMINAMATH_CALUDE_cube_root_equivalence_l1314_131445


namespace NUMINAMATH_CALUDE_angle_y_measure_l1314_131452

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  -- Angle X in degrees
  x : ℝ
  -- Triangle sum theorem
  sum_theorem : x + 3*x + 3*x = 180
  -- Non-negativity of angles
  x_nonneg : x ≥ 0

/-- The measure of angle Y in the isosceles triangle is 540/7 degrees -/
theorem angle_y_measure (t : IsoscelesTriangle) : 3 * t.x = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_y_measure_l1314_131452


namespace NUMINAMATH_CALUDE_prime_divisor_problem_l1314_131459

theorem prime_divisor_problem (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p = Nat.sqrt n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p) ∧
  (∃ p : ℕ, Nat.Prime p ∧ p = Nat.sqrt (n + 72) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (n + 72) → q ≤ p) →
  n = 49 ∨ n = 289 :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_problem_l1314_131459


namespace NUMINAMATH_CALUDE_probability_five_heads_ten_coins_l1314_131426

theorem probability_five_heads_ten_coins : 
  let n : ℕ := 10  -- total number of coins
  let k : ℕ := 5   -- number of heads we're looking for
  let p : ℚ := 1/2 -- probability of getting heads on a single coin flip
  Nat.choose n k * p^k * (1-p)^(n-k) = 63/256 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_heads_ten_coins_l1314_131426


namespace NUMINAMATH_CALUDE_sum_distances_bound_l1314_131467

/-- A convex quadrilateral with side lengths p, q, r, s, where p ≤ q ≤ r ≤ s -/
structure ConvexQuadrilateral where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  p_le_q : p ≤ q
  q_le_r : q ≤ r
  r_le_s : r ≤ s
  convex : True  -- Assuming convexity without formal definition

/-- The sum of distances from an interior point to each side of the quadrilateral -/
def sum_distances (quad : ConvexQuadrilateral) (P : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of the sum of distances

/-- Theorem: The sum of distances from any interior point to each side 
    is less than or equal to 3 times the sum of all side lengths -/
theorem sum_distances_bound (quad : ConvexQuadrilateral) (P : ℝ × ℝ) :
  sum_distances quad P ≤ 3 * (quad.p + quad.q + quad.r + quad.s) :=
sorry

end NUMINAMATH_CALUDE_sum_distances_bound_l1314_131467


namespace NUMINAMATH_CALUDE_min_distance_to_plane_l1314_131450

theorem min_distance_to_plane (x y z : ℝ) :
  x + 2*y + 3*z = 1 →
  x^2 + y^2 + z^2 ≥ 1/14 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_plane_l1314_131450


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l1314_131448

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l1314_131448


namespace NUMINAMATH_CALUDE_composite_sum_product_l1314_131406

theorem composite_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a^2 + a*c - c^2 = b^2 + b*d - d^2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a*b + c*d = x*y :=
sorry

end NUMINAMATH_CALUDE_composite_sum_product_l1314_131406


namespace NUMINAMATH_CALUDE_smallest_n_complex_equality_l1314_131444

theorem smallest_n_complex_equality (n : ℕ) (a b c : ℝ) :
  (n > 0) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (∀ k : ℕ, k > 0 ∧ k < n → ¬ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y*I + z*I)^k = (x - y*I - z*I)^k) →
  ((a + b*I + c*I)^n = (a - b*I - c*I)^n) →
  ((b + c) / a = Real.sqrt (12 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_complex_equality_l1314_131444


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_265_l1314_131485

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_101101_equals_octal_265 :
  let binary : List Bool := [true, false, true, true, false, true]
  let decimal : Nat := binary_to_decimal binary
  let octal : List Nat := decimal_to_octal decimal
  octal = [5, 6, 2] := by sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_265_l1314_131485


namespace NUMINAMATH_CALUDE_triangle_properties_triangle_max_area_l1314_131464

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sides form an arithmetic sequence -/
def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Vectors (3, sin B) and (2, sin C) are collinear -/
def areVectorsCollinear (t : Triangle) : Prop :=
  3 * Real.sin t.C = 2 * Real.sin t.B

/-- The product of sides a and c is 8 -/
def hasSideProduct8 (t : Triangle) : Prop :=
  t.a * t.c = 8

theorem triangle_properties (t : Triangle) 
  (h1 : isArithmeticSequence t) 
  (h2 : areVectorsCollinear t) :
  Real.cos t.A = -1/4 := by sorry

theorem triangle_max_area (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : hasSideProduct8 t) :
  ∃ (S : ℝ), S = 2 * Real.sqrt 3 ∧ 
  ∀ (area : ℝ), area ≤ S := by sorry

end NUMINAMATH_CALUDE_triangle_properties_triangle_max_area_l1314_131464


namespace NUMINAMATH_CALUDE_wooden_box_height_is_6_meters_l1314_131439

def wooden_box_length : ℝ := 8
def wooden_box_width : ℝ := 10
def small_box_length : ℝ := 0.04
def small_box_width : ℝ := 0.05
def small_box_height : ℝ := 0.06
def max_small_boxes : ℕ := 4000000

theorem wooden_box_height_is_6_meters :
  let small_box_volume := small_box_length * small_box_width * small_box_height
  let total_volume := small_box_volume * max_small_boxes
  let wooden_box_height := total_volume / (wooden_box_length * wooden_box_width)
  wooden_box_height = 6 := by sorry

end NUMINAMATH_CALUDE_wooden_box_height_is_6_meters_l1314_131439


namespace NUMINAMATH_CALUDE_birthday_cake_is_tradition_l1314_131425

/-- Represents different types of office practices -/
inductive OfficePractice
  | Tradition
  | Balance
  | Concern
  | Relationship

/-- Represents the office birthday cake practice -/
def birthdayCakePractice : OfficePractice := OfficePractice.Tradition

/-- Theorem stating that the office birthday cake practice is a tradition -/
theorem birthday_cake_is_tradition : 
  birthdayCakePractice = OfficePractice.Tradition := by sorry


end NUMINAMATH_CALUDE_birthday_cake_is_tradition_l1314_131425


namespace NUMINAMATH_CALUDE_f_properties_l1314_131411

/-- The function f(x) = a*ln(x) + b/x + c/(x^2) -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- The statement that f has both a maximum and a minimum value -/
def has_max_and_min (f : ℝ → ℝ) : Prop := ∃ (x_max x_min : ℝ), ∀ x, f x ≤ f x_max ∧ f x_min ≤ f x

theorem f_properties (a b c : ℝ) (ha : a ≠ 0) 
  (h_max_min : has_max_and_min (f a b c)) :
  ab > 0 ∧ b^2 + 8*a*c > 0 ∧ a*c < 0 := by sorry

end NUMINAMATH_CALUDE_f_properties_l1314_131411


namespace NUMINAMATH_CALUDE_correct_calculation_l1314_131424

theorem correct_calculation (x : ℝ) : 3 * x - 5 = 103 → x / 3 - 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1314_131424
