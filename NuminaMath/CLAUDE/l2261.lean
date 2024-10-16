import Mathlib

namespace NUMINAMATH_CALUDE_unique_divisor_l2261_226168

def sum_even_two_digit : Nat := 2430

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_sum (n : Nat) : Nat := (n / 10) + (n % 10)

def reverse_digits (n : Nat) : Nat := (n % 10) * 10 + (n / 10)

theorem unique_divisor :
  ∃! n : Nat, is_two_digit n ∧ 
    sum_even_two_digit % n = 0 ∧
    sum_even_two_digit / n = reverse_digits n ∧
    digits_sum (sum_even_two_digit / n) = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisor_l2261_226168


namespace NUMINAMATH_CALUDE_rotating_triangle_path_length_l2261_226196

/-- The total path length of point A in a rotating triangle -/
theorem rotating_triangle_path_length (α : ℝ) (h1 : 0 < α) (h2 : α < π / 3) :
  let triangle_rotation := (2 / 3 * π * (1 + Real.sin α) - 2 * α)
  (100 - 1) / 3 * triangle_rotation = 22 * π * (1 + Real.sin α) - 66 * α :=
by sorry

end NUMINAMATH_CALUDE_rotating_triangle_path_length_l2261_226196


namespace NUMINAMATH_CALUDE_player_a_wins_l2261_226118

/-- Represents a player in the chocolate bar game -/
inductive Player
| A
| B

/-- Represents a move in the chocolate bar game -/
inductive Move
| Single
| Double

/-- Represents the state of the chocolate bar game -/
structure GameState where
  grid : Fin 7 → Fin 7 → Bool
  current_player : Player

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a move is valid for the current player and game state -/
def is_valid_move (gs : GameState) (m : Move) : Bool :=
  match gs.current_player, m with
  | Player.A, Move.Single => true
  | Player.B, _ => true
  | _, _ => false

/-- Applies a move to the game state, returning the new state -/
def apply_move (gs : GameState) (m : Move) : GameState :=
  sorry

/-- Counts the number of squares taken by a player -/
def count_squares (gs : GameState) (p : Player) : Nat :=
  sorry

/-- The main theorem stating that Player A can always secure more than half the squares -/
theorem player_a_wins (init_state : GameState) (strategy_a strategy_b : Strategy) :
  ∃ (final_state : GameState),
    count_squares final_state Player.A > 24 :=
  sorry

end NUMINAMATH_CALUDE_player_a_wins_l2261_226118


namespace NUMINAMATH_CALUDE_calculator_reciprocal_l2261_226119

theorem calculator_reciprocal (x : ℝ) :
  (1 / (1/x - 1)) - 1 = -0.75 → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_calculator_reciprocal_l2261_226119


namespace NUMINAMATH_CALUDE_triangle_side_length_l2261_226132

/-- In a triangle DEF, given angle E, side DE, and side DF, prove the length of EF --/
theorem triangle_side_length (E D F : ℝ) (hE : E = 45 * π / 180) 
  (hDE : D = 100) (hDF : F = 100 * Real.sqrt 2) : 
  ∃ (EF : ℝ), abs (EF - Real.sqrt (10000 + 5176.4)) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2261_226132


namespace NUMINAMATH_CALUDE_dog_distribution_theorem_l2261_226121

/-- The number of ways to distribute 12 dogs into three groups -/
def dog_distribution_ways : ℕ :=
  (Nat.choose 11 3) * (Nat.choose 7 4)

/-- Theorem stating the number of ways to distribute the dogs -/
theorem dog_distribution_theorem : dog_distribution_ways = 5775 := by
  sorry

end NUMINAMATH_CALUDE_dog_distribution_theorem_l2261_226121


namespace NUMINAMATH_CALUDE_pencils_left_problem_l2261_226146

def pencils_left (total_pencils : ℕ) (num_students : ℕ) : ℕ :=
  total_pencils - (num_students * (total_pencils / num_students))

theorem pencils_left_problem :
  pencils_left 42 12 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_left_problem_l2261_226146


namespace NUMINAMATH_CALUDE_exists_recurrence_sequence_l2261_226179

-- Define the sequence type
def RecurrenceSequence (x y : ℝ) := ℕ → ℝ

-- Define the recurrence relation property
def SatisfiesRecurrence (a : RecurrenceSequence x y) : Prop :=
  ∀ n : ℕ, a (n + 2) = x * a (n + 1) + y * a n

-- Define the boundedness property
def SatisfiesBoundedness (a : RecurrenceSequence x y) : Prop :=
  ∀ r : ℝ, r > 0 → ∃ i j : ℕ, i > 0 ∧ j > 0 ∧ |a i| < r ∧ r < |a j|

-- Define the non-zero property
def IsNonZero (a : RecurrenceSequence x y) : Prop :=
  ∀ n : ℕ, a n ≠ 0

-- Main theorem
theorem exists_recurrence_sequence :
  ∃ x y : ℝ, ∃ a : RecurrenceSequence x y,
    SatisfiesRecurrence a ∧ SatisfiesBoundedness a ∧ IsNonZero a := by
  sorry

end NUMINAMATH_CALUDE_exists_recurrence_sequence_l2261_226179


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l2261_226172

/-- 
Given 8 schools, where 17 students didn't get picked for each team,
and 384 total students make the teams, prove that 65 students tried out
for the trivia teams in each school.
-/
theorem trivia_team_tryouts (
  num_schools : ℕ) 
  (students_not_picked : ℕ) 
  (total_students_picked : ℕ) 
  (h1 : num_schools = 8)
  (h2 : students_not_picked = 17)
  (h3 : total_students_picked = 384) :
  num_schools * (65 - students_not_picked) = total_students_picked := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l2261_226172


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2261_226107

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 3 * a 7 = 64 →
  a 5 = 8 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2261_226107


namespace NUMINAMATH_CALUDE_hugo_tom_box_folding_l2261_226185

/-- The number of small boxes Hugo and Tom fold together -/
def small_boxes : ℕ := 4200

/-- The time it takes Hugo to fold a small box (in seconds) -/
def hugo_small_time : ℕ := 3

/-- The time it takes Tom to fold a small or medium box (in seconds) -/
def tom_box_time : ℕ := 4

/-- The total time Hugo and Tom spend folding boxes (in seconds) -/
def total_time : ℕ := 7200

/-- The number of medium boxes Hugo and Tom fold together -/
def medium_boxes : ℕ := 1800

theorem hugo_tom_box_folding :
  small_boxes = (total_time / hugo_small_time) + (total_time / tom_box_time) :=
sorry

end NUMINAMATH_CALUDE_hugo_tom_box_folding_l2261_226185


namespace NUMINAMATH_CALUDE_shirt_cost_proof_l2261_226122

/-- The cost of the shirt Macey wants to buy -/
def shirt_cost : ℚ := 3

/-- The amount Macey has already saved -/
def saved_amount : ℚ := 3/2

/-- The number of weeks Macey needs to save -/
def weeks_to_save : ℕ := 3

/-- The amount Macey saves per week -/
def weekly_savings : ℚ := 1/2

theorem shirt_cost_proof : 
  shirt_cost = saved_amount + weeks_to_save * weekly_savings := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_proof_l2261_226122


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2261_226190

theorem inequality_solution_set (x : ℝ) :
  {x : ℝ | x^4 - 16*x^2 - 36*x > 0} = {x : ℝ | x < -4 ∨ (-4 < x ∧ x < -1) ∨ x > 9} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2261_226190


namespace NUMINAMATH_CALUDE_euro_calculation_l2261_226156

-- Define the € operation
def euro (x y : ℝ) : ℝ := 3 * x * y

-- Theorem statement
theorem euro_calculation : euro 3 (euro 4 5) = 540 := by
  sorry

end NUMINAMATH_CALUDE_euro_calculation_l2261_226156


namespace NUMINAMATH_CALUDE_union_of_sets_l2261_226140

theorem union_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | -2 < x ∧ x < 0} →
  B = {x : ℝ | -1 < x ∧ x < 1} →
  A ∪ B = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2261_226140


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_decrease_l2261_226171

/-- Proves that the decrease in average expenditure per head is 1 rupee
    given the initial conditions of the hostel mess problem. -/
theorem hostel_mess_expenditure_decrease :
  let initial_students : ℕ := 35
  let new_students : ℕ := 7
  let total_students : ℕ := initial_students + new_students
  let initial_expenditure : ℕ := 420
  let expenditure_increase : ℕ := 42
  let new_expenditure : ℕ := initial_expenditure + expenditure_increase
  let initial_average : ℚ := initial_expenditure / initial_students
  let new_average : ℚ := new_expenditure / total_students
  initial_average - new_average = 1 := by
  sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_decrease_l2261_226171


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l2261_226175

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 80) : 
  x^2 + y^2 = 1440 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l2261_226175


namespace NUMINAMATH_CALUDE_patio_tile_count_l2261_226191

/-- Represents a square patio with red tiles along its diagonals -/
structure SquarePatio where
  side_length : ℕ
  red_tiles : ℕ

/-- The number of red tiles on a square patio with given side length -/
def red_tiles_count (s : ℕ) : ℕ := 2 * s - 1

/-- The total number of tiles on a square patio with given side length -/
def total_tiles_count (s : ℕ) : ℕ := s * s

/-- Theorem stating that if a square patio has 61 red tiles, it has 961 total tiles -/
theorem patio_tile_count (p : SquarePatio) (h : p.red_tiles = 61) :
  total_tiles_count p.side_length = 961 := by
  sorry

end NUMINAMATH_CALUDE_patio_tile_count_l2261_226191


namespace NUMINAMATH_CALUDE_johans_house_rooms_l2261_226165

theorem johans_house_rooms (walls_per_room : ℕ) (green_ratio : ℚ) (purple_walls : ℕ) : 
  walls_per_room = 8 →
  green_ratio = 3/5 →
  purple_walls = 32 →
  ∃ (total_rooms : ℕ), total_rooms = 10 ∧ 
    (purple_walls : ℚ) / walls_per_room = (1 - green_ratio) * total_rooms :=
by sorry

end NUMINAMATH_CALUDE_johans_house_rooms_l2261_226165


namespace NUMINAMATH_CALUDE_calculation_result_l2261_226149

theorem calculation_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |0.00067 * 0.338 - (75 * 0.00000102 / 0.00338 * 0.042) - 0.0008| < ε :=
sorry

end NUMINAMATH_CALUDE_calculation_result_l2261_226149


namespace NUMINAMATH_CALUDE_quadratic_sum_l2261_226109

/-- Given a quadratic function f(x) = 8x^2 - 48x - 320, prove that when written in the form a(x+b)^2+c, the sum a + b + c equals -387. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 8*x^2 - 48*x - 320) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = -387 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2261_226109


namespace NUMINAMATH_CALUDE_square_inequality_for_negatives_l2261_226169

theorem square_inequality_for_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_for_negatives_l2261_226169


namespace NUMINAMATH_CALUDE_ages_sum_l2261_226145

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 128 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l2261_226145


namespace NUMINAMATH_CALUDE_real_root_of_cubic_l2261_226195

theorem real_root_of_cubic (a b c : ℂ) (h_a_real : a.im = 0)
  (h_sum : a + b + c = 5)
  (h_sum_prod : a * b + b * c + c * a = 7)
  (h_prod : a * b * c = 2) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_real_root_of_cubic_l2261_226195


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l2261_226183

def G (n : ℕ) : ℕ := 2^(3^n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l2261_226183


namespace NUMINAMATH_CALUDE_log_equation_solution_l2261_226151

theorem log_equation_solution : 
  ∃ (x : ℝ), (Real.log 729 / Real.log (3 * x) = x) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2261_226151


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2261_226166

theorem sum_of_three_numbers : 3/8 + 0.125 + 9.51 = 10.01 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2261_226166


namespace NUMINAMATH_CALUDE_tournament_balls_used_l2261_226131

/-- A tennis tournament with specified conditions -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  balls_per_can : Nat

/-- Calculate the total number of tennis balls used in the tournament -/
def total_balls_used (t : TennisTournament) : Nat :=
  (t.games_per_round.sum * t.cans_per_game * t.balls_per_can)

/-- Theorem stating the total number of tennis balls used in the specific tournament -/
theorem tournament_balls_used :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    balls_per_can := 3
  }
  total_balls_used t = 225 := by sorry

end NUMINAMATH_CALUDE_tournament_balls_used_l2261_226131


namespace NUMINAMATH_CALUDE_debbys_museum_pictures_l2261_226112

theorem debbys_museum_pictures 
  (zoo_pictures : ℕ) 
  (deleted_pictures : ℕ) 
  (remaining_pictures : ℕ) 
  (h1 : zoo_pictures = 24)
  (h2 : deleted_pictures = 14)
  (h3 : remaining_pictures = 22)
  (h4 : remaining_pictures = zoo_pictures + museum_pictures - deleted_pictures) :
  museum_pictures = 12 := by
  sorry

#check debbys_museum_pictures

end NUMINAMATH_CALUDE_debbys_museum_pictures_l2261_226112


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_standard_deck_l2261_226113

/-- A standard deck of cards. -/
structure Deck :=
  (total : Nat)
  (black : Nat)
  (red : Nat)
  (h_total : total = 52)
  (h_half : black = red)
  (h_sum : black + red = total)

/-- The expected number of adjacent pairs with one black and one red card
    in a circular arrangement of cards from a standard deck. -/
def expectedAdjacentPairs (d : Deck) : Rat :=
  (d.total : Rat) * (d.black : Rat) * (d.red : Rat) / ((d.total - 1) : Rat)

theorem expected_adjacent_pairs_standard_deck :
  ∃ (d : Deck), expectedAdjacentPairs d = 1352 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_standard_deck_l2261_226113


namespace NUMINAMATH_CALUDE_sequence_sum_l2261_226103

theorem sequence_sum (a b c d : ℕ) 
  (h1 : b - a = d - c) 
  (h2 : d - a = 24) 
  (h3 : b - a = (d - c) + 2) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) : 
  a + b + c + d = 54 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2261_226103


namespace NUMINAMATH_CALUDE_goldies_earnings_l2261_226138

/-- Calculates the total earnings for pet-sitting over two weeks -/
def total_earnings (hourly_rate : ℕ) (hours_week1 : ℕ) (hours_week2 : ℕ) : ℕ :=
  hourly_rate * hours_week1 + hourly_rate * hours_week2

/-- Proves that Goldie's total earnings for two weeks of pet-sitting is $250 -/
theorem goldies_earnings : total_earnings 5 20 30 = 250 := by
  sorry

end NUMINAMATH_CALUDE_goldies_earnings_l2261_226138


namespace NUMINAMATH_CALUDE_fraction_problem_l2261_226139

theorem fraction_problem (x y : ℕ) (h1 : x + y = 122) (h2 : (x - 19) / (y - 19) = 1 / 5) :
  x / y = 33 / 89 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2261_226139


namespace NUMINAMATH_CALUDE_triangle_similarity_l2261_226126

-- Define the types for points and triangles
variable (Point : Type) (Triangle : Type)

-- Define the necessary relations and properties
variable (is_scalene : Triangle → Prop)
variable (point_on_segment : Point → Point → Point → Prop)
variable (similar_triangles : Triangle → Triangle → Prop)
variable (point_on_line : Point → Point → Point → Prop)
variable (equal_distance : Point → Point → Point → Point → Prop)

-- State the theorem
theorem triangle_similarity 
  (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point) 
  (ABC A₁B₁C₁ A₂B₂C₂ : Triangle) :
  is_scalene ABC →
  point_on_segment A₁ B C →
  point_on_segment B₁ C A →
  point_on_segment C₁ A B →
  similar_triangles A₁B₁C₁ ABC →
  point_on_line A₂ B₁ C₁ →
  equal_distance A A₂ A₁ A₂ →
  point_on_line B₂ C₁ A₁ →
  equal_distance B B₂ B₁ B₂ →
  point_on_line C₂ A₁ B₁ →
  equal_distance C C₂ C₁ C₂ →
  similar_triangles A₂B₂C₂ ABC :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2261_226126


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2261_226177

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4) :
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2261_226177


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_characterization_l2261_226136

-- Part 1
theorem solution_set_part1 (x : ℝ) :
  -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 := by sorry

-- Part 2
def solution_set_part2 (a x : ℝ) : Prop :=
  a * x^2 + 3 * x + 2 > -a * x - 1

theorem solution_set_characterization (a x : ℝ) (ha : a > 0) :
  solution_set_part2 a x ↔
    (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
    (a = 3 ∧ x ≠ -1) ∨
    (a > 3 ∧ (x < -1 ∨ x > -3/a)) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_characterization_l2261_226136


namespace NUMINAMATH_CALUDE_additional_students_l2261_226199

theorem additional_students (initial_students : ℕ) (students_per_computer : ℕ) (target_computers : ℕ) : 
  initial_students = 82 →
  students_per_computer = 2 →
  target_computers = 49 →
  (initial_students + (target_computers - initial_students / students_per_computer) * students_per_computer) - initial_students = 16 := by
sorry

end NUMINAMATH_CALUDE_additional_students_l2261_226199


namespace NUMINAMATH_CALUDE_angle_tangent_product_l2261_226123

theorem angle_tangent_product (A C : ℝ) (h : 5 * (Real.cos A + Real.cos C) + 4 * (Real.cos A * Real.cos C + 1) = 0) :
  Real.tan (A / 2) * Real.tan (C / 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_angle_tangent_product_l2261_226123


namespace NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l2261_226104

/-- A regular polygon with interior angles measuring 150° has 12 sides. -/
theorem regular_polygon_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) → n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l2261_226104


namespace NUMINAMATH_CALUDE_m_range_for_z_in_third_quadrant_l2261_226100

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := m * (3 + Complex.I) - (2 + Complex.I)

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- State the theorem
theorem m_range_for_z_in_third_quadrant :
  ∀ m : ℝ, in_third_quadrant (z m) ↔ m < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_m_range_for_z_in_third_quadrant_l2261_226100


namespace NUMINAMATH_CALUDE_rotate_minus_two_zero_l2261_226129

/-- Rotate a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate_minus_two_zero :
  rotate90Clockwise (-2, 0) = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_minus_two_zero_l2261_226129


namespace NUMINAMATH_CALUDE_donkeys_and_boys_l2261_226101

theorem donkeys_and_boys (b d : ℕ) : 
  (d = b - 1) →  -- Condition 1: When each boy sits on a donkey, one boy is left
  (b / 2 = d - 1) →  -- Condition 2: When two boys sit on each donkey, one donkey is left
  (b = 4 ∧ d = 3) :=  -- Conclusion: There are 4 boys and 3 donkeys
by sorry

end NUMINAMATH_CALUDE_donkeys_and_boys_l2261_226101


namespace NUMINAMATH_CALUDE_converse_correct_l2261_226178

/-- The original statement -/
def original_statement (x : ℝ) : Prop := x^2 = 1 → x = 1

/-- The converse statement -/
def converse_statement (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

/-- Theorem stating that the converse_statement is indeed the converse of the original_statement -/
theorem converse_correct :
  converse_statement = (fun x => ¬(original_statement x)) := by sorry

end NUMINAMATH_CALUDE_converse_correct_l2261_226178


namespace NUMINAMATH_CALUDE_money_distribution_l2261_226120

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : B + C = 330) 
  (h3 : C = 30) : 
  A + C = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2261_226120


namespace NUMINAMATH_CALUDE_equation_solution_l2261_226198

theorem equation_solution (a b c d p : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |p| = 3) :
  ∃! x : ℝ, (a + b) * x^2 + 4 * c * d * x + p^2 = x ∧ x = -3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2261_226198


namespace NUMINAMATH_CALUDE_milk_for_cookies_l2261_226181

/-- Given the ratio of cookies to milk, calculate the cups of milk needed for a given number of cookies -/
def milkNeeded (cookiesReference : ℕ) (quartsReference : ℕ) (cupsPerQuart : ℕ) (cookiesTarget : ℕ) : ℚ :=
  (quartsReference * cupsPerQuart : ℚ) * cookiesTarget / cookiesReference

theorem milk_for_cookies :
  milkNeeded 15 5 4 6 = 8 := by
  sorry

#eval milkNeeded 15 5 4 6

end NUMINAMATH_CALUDE_milk_for_cookies_l2261_226181


namespace NUMINAMATH_CALUDE_shaded_area_of_divided_square_l2261_226117

theorem shaded_area_of_divided_square (side_length : ℝ) (total_squares : ℕ) (shaded_squares : ℕ) : 
  side_length = 10 ∧ total_squares = 25 ∧ shaded_squares = 5 → 
  (side_length^2 / total_squares) * shaded_squares = 20 := by
  sorry

#check shaded_area_of_divided_square

end NUMINAMATH_CALUDE_shaded_area_of_divided_square_l2261_226117


namespace NUMINAMATH_CALUDE_rhombus_area_example_l2261_226125

/-- Given a rhombus with height h and diagonal d, calculates its area -/
def rhombusArea (h d : ℝ) : ℝ := sorry

/-- Theorem: A rhombus with height 12 cm and diagonal 15 cm has an area of 150 cm² -/
theorem rhombus_area_example : rhombusArea 12 15 = 150 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_example_l2261_226125


namespace NUMINAMATH_CALUDE_min_adventurers_l2261_226157

structure AdventurerGroup where
  rubies : Finset Nat
  emeralds : Finset Nat
  sapphires : Finset Nat
  diamonds : Finset Nat

def AdventurerGroup.valid (g : AdventurerGroup) : Prop :=
  g.rubies.card = 5 ∧
  g.emeralds.card = 11 ∧
  g.sapphires.card = 10 ∧
  g.diamonds.card = 6 ∧
  (∀ a ∈ g.diamonds, (a ∈ g.emeralds ∨ a ∈ g.sapphires) ∧ ¬(a ∈ g.emeralds ∧ a ∈ g.sapphires)) ∧
  (∀ a ∈ g.emeralds, (a ∈ g.rubies ∨ a ∈ g.diamonds) ∧ ¬(a ∈ g.rubies ∧ a ∈ g.diamonds))

theorem min_adventurers (g : AdventurerGroup) (h : g.valid) :
  (g.rubies ∪ g.emeralds ∪ g.sapphires ∪ g.diamonds).card ≥ 16 := by
  sorry

#check min_adventurers

end NUMINAMATH_CALUDE_min_adventurers_l2261_226157


namespace NUMINAMATH_CALUDE_marble_difference_is_seventeen_l2261_226147

/-- Calculates the difference in marbles between John and Ben after Ben gives half his marbles to John -/
def marbleDifference (benInitial : ℕ) (johnInitial : ℕ) : ℕ :=
  let benFinal := benInitial - benInitial / 2
  let johnFinal := johnInitial + benInitial / 2
  johnFinal - benFinal

/-- Proves that the difference in marbles between John and Ben is 17 after the transfer -/
theorem marble_difference_is_seventeen :
  marbleDifference 18 17 = 17 := by
  sorry

#eval marbleDifference 18 17

end NUMINAMATH_CALUDE_marble_difference_is_seventeen_l2261_226147


namespace NUMINAMATH_CALUDE_surface_area_theorem_l2261_226106

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the solid RUVWX -/
structure SlicedSolid where
  prism : RightPrism
  V : Point3D
  W : Point3D
  X : Point3D

/-- Calculate the surface area of the sliced solid RUVWX -/
def surface_area_RUVWX (solid : SlicedSolid) : ℝ :=
  sorry

/-- The main theorem stating the surface area of RUVWX -/
theorem surface_area_theorem (solid : SlicedSolid) 
  (h1 : solid.prism.height = 20)
  (h2 : solid.prism.base_side_length = 10)
  (h3 : solid.V = Point3D.mk 5 0 10)
  (h4 : solid.W = Point3D.mk 5 (5 * Real.sqrt 3) 10)
  (h5 : solid.X = Point3D.mk 0 0 10) :
  surface_area_RUVWX solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
sorry

end NUMINAMATH_CALUDE_surface_area_theorem_l2261_226106


namespace NUMINAMATH_CALUDE_definite_integral_exp_abs_x_l2261_226173

theorem definite_integral_exp_abs_x : 
  ∫ x in (-2)..4, Real.exp (|x|) = Real.exp 2 - Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_abs_x_l2261_226173


namespace NUMINAMATH_CALUDE_U_value_l2261_226162

theorem U_value : 
  let U := 1 / (4 - Real.sqrt 9) + 1 / (Real.sqrt 9 - Real.sqrt 8) - 
           1 / (Real.sqrt 8 - Real.sqrt 7) + 1 / (Real.sqrt 7 - Real.sqrt 6) - 
           1 / (Real.sqrt 6 - 3)
  U = 1 := by sorry

end NUMINAMATH_CALUDE_U_value_l2261_226162


namespace NUMINAMATH_CALUDE_solution_ordered_pair_l2261_226154

theorem solution_ordered_pair : ∃ x y : ℝ, 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x + 1) + (y + 1)) ∧
  x = 8 ∧ y = -1 := by
sorry

end NUMINAMATH_CALUDE_solution_ordered_pair_l2261_226154


namespace NUMINAMATH_CALUDE_weight_estimate_error_l2261_226150

/-- The weight of a disk with precise 1-meter diameter in kg -/
def precise_disk_weight : ℝ := 100

/-- The radius of a disk in meters -/
def disk_radius : ℝ := 0.5

/-- The standard deviation of the disk radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The number of disks in the stack -/
def num_disks : ℕ := 100

/-- The expected weight of a single disk with variable radius -/
noncomputable def expected_disk_weight : ℝ := sorry

/-- The error in the weight estimate -/
theorem weight_estimate_error :
  num_disks * expected_disk_weight - (num_disks : ℝ) * precise_disk_weight = 4 := by sorry

end NUMINAMATH_CALUDE_weight_estimate_error_l2261_226150


namespace NUMINAMATH_CALUDE_cubic_root_quadratic_coefficient_l2261_226184

theorem cubic_root_quadratic_coefficient 
  (A B C : ℝ) 
  (r s : ℝ) 
  (h1 : A ≠ 0)
  (h2 : A * r^2 + B * r + C = 0)
  (h3 : A * s^2 + B * s + C = 0) :
  ∃ (p q : ℝ), r^3^2 + p * r^3 + q = 0 ∧ s^3^2 + p * s^3 + q = 0 ∧ 
  p = (B^3 - 3*A*B*C + 2*A*C^2) / A^3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_quadratic_coefficient_l2261_226184


namespace NUMINAMATH_CALUDE_star_computation_l2261_226114

/-- Operation ⭐ defined as (5a + b) / (a - b) -/
def star (a b : ℚ) : ℚ := (5 * a + b) / (a - b)

theorem star_computation :
  star (star 7 (star 2 5)) 3 = -31 := by
  sorry

end NUMINAMATH_CALUDE_star_computation_l2261_226114


namespace NUMINAMATH_CALUDE_problem_statement_l2261_226163

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2023)^2 = 0) : a^b = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2261_226163


namespace NUMINAMATH_CALUDE_expression_equals_one_l2261_226187

theorem expression_equals_one : 
  (150^2 - 12^2) / (90^2 - 18^2) * ((90 - 18)*(90 + 18)) / ((150 - 12)*(150 + 12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2261_226187


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2261_226160

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2261_226160


namespace NUMINAMATH_CALUDE_fraction_difference_to_fifth_power_l2261_226142

theorem fraction_difference_to_fifth_power :
  (3/4 - 1/8)^5 = 3125/32768 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_to_fifth_power_l2261_226142


namespace NUMINAMATH_CALUDE_cylinder_radius_calculation_l2261_226133

theorem cylinder_radius_calculation (shadow_length : ℝ) (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (h1 : shadow_length = 12)
  (h2 : flagpole_height = 1.5)
  (h3 : flagpole_shadow = 3)
  (h4 : flagpole_shadow > 0) -- To avoid division by zero
  : ∃ (radius : ℝ), radius = shadow_length * (flagpole_height / flagpole_shadow) ∧ radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_calculation_l2261_226133


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l2261_226182

theorem sixth_root_of_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l2261_226182


namespace NUMINAMATH_CALUDE_quadratic_radical_condition_l2261_226130

theorem quadratic_radical_condition (x : ℝ) : Real.sqrt ((x - 3)^2) = x - 3 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_condition_l2261_226130


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2261_226115

open Set

theorem inequality_solution_sets 
  (a b c d : ℝ) 
  (h : {x : ℝ | (b / (x + a)) + ((x + d) / (x + c)) < 0} = Ioo (-1) (-1/3) ∪ Ioo (1/2) 1) :
  {x : ℝ | (b * x / (a * x - 1)) + ((d * x - 1) / (c * x - 1)) < 0} = Ioo 1 3 ∪ Ioo (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2261_226115


namespace NUMINAMATH_CALUDE_more_trucks_than_buses_l2261_226108

/-- Given 17 trucks and 9 buses, prove that there are 8 more trucks than buses. -/
theorem more_trucks_than_buses :
  let num_trucks : ℕ := 17
  let num_buses : ℕ := 9
  num_trucks - num_buses = 8 :=
by sorry

end NUMINAMATH_CALUDE_more_trucks_than_buses_l2261_226108


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2261_226189

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 + y^2 + 2*a*x + 2*y + (a-1)^2
  f (0, 0) > 0 := by sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2261_226189


namespace NUMINAMATH_CALUDE_angle_A1C1_B1C_is_60_degrees_l2261_226164

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Calculates the angle between two lines in 3D space -/
def angle_between_lines (p1 p2 p3 p4 : Point3D) : ℝ :=
  sorry

/-- Theorem: In a cube, the angle between A1C1 and B1C is 60 degrees -/
theorem angle_A1C1_B1C_is_60_degrees (cube : Cube) :
  angle_between_lines cube.A1 cube.C1 cube.B1 cube.C = 60 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_A1C1_B1C_is_60_degrees_l2261_226164


namespace NUMINAMATH_CALUDE_stating_not_always_triangle_from_parallelogram_l2261_226167

/-- A stick represents a line segment with a positive length. -/
structure Stick :=
  (length : ℝ)
  (positive : length > 0)

/-- A parallelogram composed of four equal sticks. -/
structure Parallelogram :=
  (stick : Stick)

/-- Represents a potential triangle formed from the parallelogram's sticks. -/
structure PotentialTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Checks if a triangle can be formed given three side lengths. -/
def isValidTriangle (t : PotentialTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side2 + t.side3 > t.side1 ∧
  t.side1 + t.side3 > t.side2

/-- 
Theorem stating that it's not always possible to form a triangle 
from a parallelogram's sticks.
-/
theorem not_always_triangle_from_parallelogram :
  ∃ p : Parallelogram, ¬∃ t : PotentialTriangle, 
    (t.side1 = p.stick.length ∧ t.side2 = p.stick.length ∧ t.side3 = 2 * p.stick.length) ∧
    isValidTriangle t :=
sorry

end NUMINAMATH_CALUDE_stating_not_always_triangle_from_parallelogram_l2261_226167


namespace NUMINAMATH_CALUDE_c_oxen_count_l2261_226111

/-- Represents the number of oxen and months for each person --/
structure GrazingData where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given GrazingData --/
def oxenMonths (data : GrazingData) : ℕ := data.oxen * data.months

/-- Theorem: Given the conditions, c put 15 oxen for grazing --/
theorem c_oxen_count (total_rent : ℚ) (a b : GrazingData) (c_months : ℕ) (c_rent : ℚ) :
  total_rent = 210 →
  a = { oxen := 10, months := 7 } →
  b = { oxen := 12, months := 5 } →
  c_months = 3 →
  c_rent = 54 →
  ∃ (c_oxen : ℕ), 
    let c : GrazingData := { oxen := c_oxen, months := c_months }
    (c_rent / total_rent) * (oxenMonths a + oxenMonths b + oxenMonths c) = oxenMonths c ∧
    c_oxen = 15 := by
  sorry


end NUMINAMATH_CALUDE_c_oxen_count_l2261_226111


namespace NUMINAMATH_CALUDE_log_inequality_l2261_226148

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + 1/x) > 1/(1 + x) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2261_226148


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l2261_226141

theorem triangle_angle_theorem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) (h5 : a = b) :
  let C := Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2)))
  C = Real.arccos (1/4) :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l2261_226141


namespace NUMINAMATH_CALUDE_apple_bag_weight_l2261_226174

theorem apple_bag_weight (empty_weight loaded_weight : ℕ) (num_bags : ℕ) : 
  empty_weight = 500 →
  loaded_weight = 1700 →
  num_bags = 20 →
  (loaded_weight - empty_weight) / num_bags = 60 :=
by sorry

end NUMINAMATH_CALUDE_apple_bag_weight_l2261_226174


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l2261_226144

/-- Julie's summer work and earnings information -/
structure SummerWork where
  hoursPerWeek : ℕ
  weeks : ℕ
  earnings : ℕ

/-- Julie's school year work information -/
structure SchoolYearWork where
  weeks : ℕ
  targetEarnings : ℕ

/-- Calculate the required hours per week during school year -/
def calculateSchoolYearHours (summer : SummerWork) (schoolYear : SchoolYearWork) : ℕ :=
  let hourlyRate := summer.earnings / (summer.hoursPerWeek * summer.weeks)
  let weeklyEarningsNeeded := schoolYear.targetEarnings / schoolYear.weeks
  weeklyEarningsNeeded / hourlyRate

/-- Theorem stating that Julie needs to work 10 hours per week during the school year -/
theorem julie_school_year_hours 
    (summer : SummerWork) 
    (schoolYear : SchoolYearWork) 
    (h1 : summer.hoursPerWeek = 40)
    (h2 : summer.weeks = 10)
    (h3 : summer.earnings = 4000)
    (h4 : schoolYear.weeks = 40)
    (h5 : schoolYear.targetEarnings = 4000) :
  calculateSchoolYearHours summer schoolYear = 10 := by
  sorry

#eval calculateSchoolYearHours 
  { hoursPerWeek := 40, weeks := 10, earnings := 4000 } 
  { weeks := 40, targetEarnings := 4000 }

end NUMINAMATH_CALUDE_julie_school_year_hours_l2261_226144


namespace NUMINAMATH_CALUDE_ellipse_equation_l2261_226105

/-- An ellipse with focal length 2 passing through (-√5, 0) has a standard equation of either x²/5 + y²/4 = 1 or y²/6 + x²/5 = 1 -/
theorem ellipse_equation (f : ℝ) (P : ℝ × ℝ) : 
  f = 2 → P = (-Real.sqrt 5, 0) → 
  (∃ (x y : ℝ), x^2/5 + y^2/4 = 1) ∨ (∃ (x y : ℝ), y^2/6 + x^2/5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2261_226105


namespace NUMINAMATH_CALUDE_vector_properties_l2261_226180

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), ∀ i, v i = c * w i

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0 * w 0 + v 1 * w 1) = 0

theorem vector_properties :
  (∃ k : ℝ, parallel (fun i => k * (a i) + b i) (fun i => a i - 2 * (b i))) ∧
  perpendicular (fun i => (25/3) * (a i) + b i) (fun i => a i - 2 * (b i)) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2261_226180


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l2261_226124

/-- Represents a rectangular prism made of 1-inch cubes -/
structure CubePrism where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a CubePrism -/
def volume (p : CubePrism) : ℕ := p.width * p.length * p.height

/-- Represents the prism of cubes not touching any tin foil -/
def innerPrism (outer : CubePrism) : CubePrism where
  width := outer.width - 2
  length := (outer.width - 2) / 2
  height := (outer.width - 2) / 2

theorem cubes_not_touching_foil (outer : CubePrism) 
  (h1 : outer.width = 10) 
  (h2 : innerPrism outer = { width := 8, length := 4, height := 4 }) : 
  volume (innerPrism outer) = 128 := by
  sorry

end NUMINAMATH_CALUDE_cubes_not_touching_foil_l2261_226124


namespace NUMINAMATH_CALUDE_lateral_to_base_area_ratio_l2261_226188

/-- A cone with its lateral surface unfolded into a sector with a 90° central angle -/
structure UnfoldedCone where
  r : ℝ  -- radius of the base circle
  R : ℝ  -- radius of the unfolded sector (lateral surface)
  h : R = 4 * r  -- condition from the 90° central angle

/-- The ratio of lateral surface area to base area for an UnfoldedCone is 4:1 -/
theorem lateral_to_base_area_ratio (cone : UnfoldedCone) :
  (π * cone.r * cone.R) / (π * cone.r^2) = 4 := by
  sorry

#check lateral_to_base_area_ratio

end NUMINAMATH_CALUDE_lateral_to_base_area_ratio_l2261_226188


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l2261_226186

/-- 
Given a non-zero vector a = (m^2 - 1, m + 1) that is parallel to vector b = (1, -2),
prove that m = 1/2.
-/
theorem parallel_vectors_imply_m_value (m : ℝ) :
  (m^2 - 1 ≠ 0 ∨ m + 1 ≠ 0) →  -- Condition 1: Vector a is non-zero
  ∃ (k : ℝ), k ≠ 0 ∧ k * (m^2 - 1) = 1 ∧ k * (m + 1) = -2 →  -- Condition 2 and 3: Parallel vectors
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l2261_226186


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l2261_226153

theorem reciprocal_equation_solution (x : ℝ) : 
  2 - (1 / (2 - x)^3) = 1 / (2 - x)^3 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l2261_226153


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2261_226116

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2261_226116


namespace NUMINAMATH_CALUDE_angle_subtraction_l2261_226134

/-- Represents an angle in degrees, minutes, and seconds -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Converts an Angle to seconds -/
def angleToSeconds (a : Angle) : ℕ :=
  a.degrees * 3600 + a.minutes * 60 + a.seconds

/-- Converts seconds to an Angle -/
def secondsToAngle (s : ℕ) : Angle :=
  let d := s / 3600
  let m := (s % 3600) / 60
  let sec := s % 60
  ⟨d, m, sec⟩

theorem angle_subtraction :
  let a₁ : Angle := ⟨90, 0, 0⟩
  let a₂ : Angle := ⟨78, 28, 56⟩
  let result : Angle := ⟨11, 31, 4⟩
  angleToSeconds a₁ - angleToSeconds a₂ = angleToSeconds result := by
  sorry

end NUMINAMATH_CALUDE_angle_subtraction_l2261_226134


namespace NUMINAMATH_CALUDE_games_that_didnt_work_l2261_226161

/-- The number of games that didn't work given Ned's game purchases and good games. -/
theorem games_that_didnt_work (friend_games garage_sale_games good_games : ℕ) : 
  friend_games = 50 → garage_sale_games = 27 → good_games = 3 → 
  friend_games + garage_sale_games - good_games = 74 := by
  sorry

end NUMINAMATH_CALUDE_games_that_didnt_work_l2261_226161


namespace NUMINAMATH_CALUDE_bridge_units_correct_l2261_226110

-- Define the units
inductive LengthUnit
| Kilometers

inductive LoadUnit
| Tons

-- Define the bridge properties
structure Bridge where
  length : ℕ
  loadCapacity : ℕ

-- Define the function to assign units
def assignUnits (b : Bridge) : (LengthUnit × LoadUnit) :=
  (LengthUnit.Kilometers, LoadUnit.Tons)

-- Theorem statement
theorem bridge_units_correct (b : Bridge) (h1 : b.length = 1) (h2 : b.loadCapacity = 50) :
  assignUnits b = (LengthUnit.Kilometers, LoadUnit.Tons) := by
  sorry

#check bridge_units_correct

end NUMINAMATH_CALUDE_bridge_units_correct_l2261_226110


namespace NUMINAMATH_CALUDE_triangle_side_length_l2261_226135

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  a = 3 → b = Real.sqrt 13 → B = π / 3 → 
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2261_226135


namespace NUMINAMATH_CALUDE_triangle_height_proof_l2261_226159

/-- Given a triangle with base 4 meters and a constant k = 2 meters, 
    prove that its height is 4 meters when its area satisfies two equations. -/
theorem triangle_height_proof (height : ℝ) (k : ℝ) (base : ℝ) : 
  k = 2 →
  base = 4 →
  (base^2) / (4 * (height - k)) = (1/2) * base * height →
  height = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l2261_226159


namespace NUMINAMATH_CALUDE_score_order_l2261_226158

/-- Represents the scores of contestants in a math competition. -/
structure Scores where
  alice : ℕ
  brian : ℕ
  cindy : ℕ
  donna : ℕ

/-- Conditions for the math competition scores. -/
def valid_scores (s : Scores) : Prop :=
  -- Brian + Donna = Alice + Cindy
  s.brian + s.donna = s.alice + s.cindy ∧
  -- If Brian and Cindy were swapped, Alice + Cindy > Brian + Donna + 10
  s.alice + s.brian > s.cindy + s.donna + 10 ∧
  -- Donna > Brian + Cindy + 20
  s.donna > s.brian + s.cindy + 20 ∧
  -- Total score is 200
  s.alice + s.brian + s.cindy + s.donna = 200

/-- The theorem to prove -/
theorem score_order (s : Scores) (h : valid_scores s) :
  s.donna > s.alice ∧ s.alice > s.brian ∧ s.brian > s.cindy := by
  sorry

end NUMINAMATH_CALUDE_score_order_l2261_226158


namespace NUMINAMATH_CALUDE_equation_simplification_l2261_226170

theorem equation_simplification (Y : ℝ) : ((3.242 * 10 * Y) / 100) = 0.3242 * Y := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l2261_226170


namespace NUMINAMATH_CALUDE_parabola_intersection_l2261_226152

theorem parabola_intersection :
  let f (x : ℝ) := 4 * x^2 + 5 * x - 6
  let g (x : ℝ) := x^2 + 14
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
    x₁ = -4 ∧ x₂ = 5/3 ∧
    f x₁ = 38 ∧ f x₂ = 121/9 ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2261_226152


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2261_226137

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A) →
  (a = 2) →
  (Real.sin (B - C) + Real.sin A = Real.sin (2 * C)) →
  (A = Real.pi / 3) ∧
  ((1/2 * a * b * Real.sin (Real.pi / 3) = 2 * Real.sqrt 3 / 3) ∨
   (1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2261_226137


namespace NUMINAMATH_CALUDE_compare_powers_l2261_226143

theorem compare_powers : (4 ^ 12 : ℕ) < 9 ^ 8 ∧ 9 ^ 8 = 3 ^ 16 := by sorry

end NUMINAMATH_CALUDE_compare_powers_l2261_226143


namespace NUMINAMATH_CALUDE_solve_a_b_l2261_226127

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}

def A (b : ℝ) : Set ℝ := {b, 2}

def complement_U_A (a b : ℝ) : Set ℝ := U a \ A b

theorem solve_a_b (a b : ℝ) : 
  complement_U_A a b = {5} →
  ((a = 2 ∨ a = -4) ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_a_b_l2261_226127


namespace NUMINAMATH_CALUDE_right_triangle_circles_l2261_226197

theorem right_triangle_circles (a b : ℝ) (R r : ℝ) : 
  a = 16 → b = 30 → 
  R = (a^2 + b^2).sqrt / 2 → 
  r = (a * b) / (a + b + (a^2 + b^2).sqrt) → 
  R + r = 23 := by sorry

end NUMINAMATH_CALUDE_right_triangle_circles_l2261_226197


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l2261_226192

/-- Represents a point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- Represents the grid and its shaded squares --/
structure Grid where
  size : Nat
  shaded : List Point

/-- Checks if a grid has horizontal, vertical, and diagonal symmetry --/
def hasSymmetry (g : Grid) : Bool := sorry

/-- Counts the number of additional squares needed for symmetry --/
def additionalSquaresForSymmetry (g : Grid) : Nat := sorry

/-- The initial grid configuration --/
def initialGrid : Grid := {
  size := 6,
  shaded := [⟨2, 5⟩, ⟨3, 3⟩, ⟨4, 2⟩, ⟨6, 1⟩]
}

theorem min_additional_squares_for_symmetry :
  additionalSquaresForSymmetry initialGrid = 9 := by sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l2261_226192


namespace NUMINAMATH_CALUDE_tshirt_jersey_cost_difference_l2261_226102

/-- The amount the Razorback shop makes off each t-shirt -/
def tshirt_profit : ℕ := 192

/-- The amount the Razorback shop makes off each jersey -/
def jersey_profit : ℕ := 34

/-- The difference in cost between a t-shirt and a jersey -/
def cost_difference : ℕ := tshirt_profit - jersey_profit

theorem tshirt_jersey_cost_difference :
  cost_difference = 158 :=
sorry

end NUMINAMATH_CALUDE_tshirt_jersey_cost_difference_l2261_226102


namespace NUMINAMATH_CALUDE_max_product_digits_sum_23_l2261_226176

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- The product of digits of a positive integer -/
def product_of_digits (n : ℕ+) : ℕ := sorry

/-- Theorem: The maximum product of digits for a positive integer with digit sum 23 is 432 -/
theorem max_product_digits_sum_23 :
  ∀ n : ℕ+, sum_of_digits n = 23 → product_of_digits n ≤ 432 :=
sorry

end NUMINAMATH_CALUDE_max_product_digits_sum_23_l2261_226176


namespace NUMINAMATH_CALUDE_pool_capacity_l2261_226155

theorem pool_capacity (C : ℝ) 
  (h1 : 0.45 * C + 300 = 0.75 * C) : C = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l2261_226155


namespace NUMINAMATH_CALUDE_chessboard_paradox_l2261_226193

/-- Represents a part of the chessboard -/
structure ChessboardPart where
  cells : ℕ
  deriving Repr

/-- Represents the chessboard -/
structure Chessboard where
  parts : List ChessboardPart
  totalCells : ℕ
  deriving Repr

/-- Function to rearrange parts of the chessboard -/
def rearrange (c : Chessboard) : Chessboard :=
  c -- Placeholder for rearrangement logic

theorem chessboard_paradox (c : Chessboard) 
  (h1 : c.parts.length = 4)
  (h2 : c.totalCells = 64) :
  (rearrange c).totalCells = 64 :=
sorry

end NUMINAMATH_CALUDE_chessboard_paradox_l2261_226193


namespace NUMINAMATH_CALUDE_m_range_l2261_226128

-- Define the condition function
def condition (x : ℝ) (m : ℝ) : Prop := 0 ≤ x ∧ x ≤ m

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0

-- Define the necessary but not sufficient relationship
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, quadratic_inequality x → condition x m) ∧
  (∃ x, condition x m ∧ ¬quadratic_inequality x)

-- Theorem statement
theorem m_range (m : ℝ) :
  necessary_not_sufficient m ↔ m ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2261_226128


namespace NUMINAMATH_CALUDE_area_equality_l2261_226194

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define midpoints
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

-- Define intersection of lines
def is_intersection (P X₁ Y₁ X₂ Y₂ : ℝ × ℝ) : Prop := sorry

-- Define area of a triangle
def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define area of a quadrilateral
def area_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_equality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_E_midpoint : is_midpoint E A B)
  (h_F_midpoint : is_midpoint F C D)
  (h_G_intersection : is_intersection G A F D E)
  (h_H_intersection : is_intersection H B F C E) :
  area_triangle A G D + area_triangle B H C = area_quadrilateral E H F G := 
sorry

end NUMINAMATH_CALUDE_area_equality_l2261_226194
