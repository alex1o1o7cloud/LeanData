import Mathlib

namespace NUMINAMATH_CALUDE_function_satisfying_equation_l1371_137168

theorem function_satisfying_equation (r s : ℚ) :
  ∀ f : ℚ → ℚ, (∀ x y : ℚ, f (x + f y) = f (x + r) + y + s) →
  (∀ x : ℚ, f x = x + r + s) ∨ (∀ x : ℚ, f x = -x + r - s) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_equation_l1371_137168


namespace NUMINAMATH_CALUDE_division_remainder_l1371_137152

theorem division_remainder : ∃ q : ℕ, 1234567 = 123 * q + 41 ∧ 41 < 123 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1371_137152


namespace NUMINAMATH_CALUDE_baseball_season_games_l1371_137177

/-- Calculates the total number of games played in a baseball season given the number of wins and a relationship between wins and losses. -/
theorem baseball_season_games (wins losses : ℕ) : 
  wins = 101 ∧ wins = 3 * losses + 14 → wins + losses = 130 :=
by sorry

end NUMINAMATH_CALUDE_baseball_season_games_l1371_137177


namespace NUMINAMATH_CALUDE_fortiethSelectedNumber_l1371_137176

/-- Calculates the nth selected number in a sequence -/
def nthSelectedNumber (totalParticipants : ℕ) (numSelected : ℕ) (firstNumber : ℕ) (n : ℕ) : ℕ :=
  let spacing := totalParticipants / numSelected
  (n - 1) * spacing + firstNumber

theorem fortiethSelectedNumber :
  nthSelectedNumber 1000 50 15 40 = 795 := by
  sorry

end NUMINAMATH_CALUDE_fortiethSelectedNumber_l1371_137176


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1371_137175

/-- The quadratic function a(x) -/
def a (x : ℝ) : ℝ := 2*x^2 - 14*x + 20

/-- The shape function y = 2x² -/
def shape (x : ℝ) : ℝ := 2*x^2

theorem quadratic_function_proof :
  a 2 = 0 ∧ a 5 = 0 ∧ ∃ k, ∀ x, a x = k * shape x + (a 0 - k * shape 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1371_137175


namespace NUMINAMATH_CALUDE_expression_evaluation_l1371_137147

theorem expression_evaluation : 
  Real.sin (π / 4) ^ 2 - Real.sqrt 27 + (1 / 2) * ((Real.sqrt 3 - 2006) ^ 0) + 6 * Real.tan (π / 6) = 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1371_137147


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_parabola_vertex_l1371_137155

/-- Given that a, b, c, and d form an arithmetic sequence, and (a, d) is the vertex of y = x^2 - 2x + 5, prove that b + c = 5 -/
theorem arithmetic_sequence_and_parabola_vertex (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) →  -- arithmetic sequence condition
  (a = 1 ∧ d = 4) →  -- vertex condition (derived from y = x^2 - 2x + 5)
  b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_parabola_vertex_l1371_137155


namespace NUMINAMATH_CALUDE_major_axis_length_l1371_137126

/-- Rectangle PQRS with ellipse passing through P and R, foci at Q and S -/
structure EllipseInRectangle where
  /-- Area of the rectangle PQRS -/
  rect_area : ℝ
  /-- Area of the ellipse -/
  ellipse_area : ℝ
  /-- The ellipse passes through P and R, and has foci at Q and S -/
  ellipse_through_PR_foci_QS : Bool

/-- Given the specific rectangle and ellipse, prove the length of the major axis -/
theorem major_axis_length (e : EllipseInRectangle) 
  (h1 : e.rect_area = 4050)
  (h2 : e.ellipse_area = 3240 * Real.pi)
  (h3 : e.ellipse_through_PR_foci_QS = true) : 
  ∃ (major_axis : ℝ), major_axis = 144 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_l1371_137126


namespace NUMINAMATH_CALUDE_amount_to_fifth_sixth_homes_l1371_137129

/-- The amount donated to the fifth and sixth nursing homes combined -/
def amount_fifth_sixth (total donation_1 donation_2 donation_3 donation_4 : ℕ) : ℕ :=
  total - (donation_1 + donation_2 + donation_3 + donation_4)

/-- Theorem stating the amount given to the fifth and sixth nursing homes -/
theorem amount_to_fifth_sixth_homes :
  amount_fifth_sixth 10000 2750 1945 1275 1890 = 2140 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_fifth_sixth_homes_l1371_137129


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1371_137159

theorem largest_integer_with_remainder : 
  ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1371_137159


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1371_137188

theorem perfect_square_condition (x y : ℕ) :
  ∃ (n : ℕ), (x + y)^2 + 3*x + y + 1 = n^2 ↔ x = y :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1371_137188


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l1371_137140

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 2 * y

-- State the theorem
theorem heartsuit_three_eight : heartsuit 3 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l1371_137140


namespace NUMINAMATH_CALUDE_set_equality_l1371_137136

-- Define sets A and B
def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the set we want to prove equal to our result
def S : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_equality : S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1371_137136


namespace NUMINAMATH_CALUDE_shooting_scores_mode_and_variance_l1371_137101

def scores : List ℕ := [8, 9, 9, 10, 10, 7, 8, 9, 10, 10]

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

def mean (l : List ℕ) : ℚ := 
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ := 
  let μ := mean l
  (l.map (λ x => ((x : ℚ) - μ) ^ 2)).sum / l.length

theorem shooting_scores_mode_and_variance :
  mode scores = 10 ∧ variance scores = 1 := by sorry

end NUMINAMATH_CALUDE_shooting_scores_mode_and_variance_l1371_137101


namespace NUMINAMATH_CALUDE_perpendicular_tangent_line_l1371_137112

/-- The curve y = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem perpendicular_tangent_line (a b : ℝ) :
  (∃ (x y : ℝ), a * x - b * y - 2 = 0) →  -- Given line exists
  f 1 = 1 →  -- Point (1,1) is on the curve
  (a / b) * (f' 1) = -1 →  -- Perpendicular condition
  b / a = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangent_line_l1371_137112


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_line_relationships_l1371_137122

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (m n : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Plane) : Prop := sorry

theorem perpendicular_planes_from_line_relationships 
  (m n : Line) (α β : Plane) 
  (h1 : different_lines m n)
  (h2 : different_planes α β)
  (h3 : line_perpendicular_to_plane m α)
  (h4 : lines_parallel m n)
  (h5 : line_parallel_to_plane n β) :
  planes_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_line_relationships_l1371_137122


namespace NUMINAMATH_CALUDE_video_game_players_l1371_137134

theorem video_game_players (lives_per_player : ℕ) (total_lives : ℕ) (h1 : lives_per_player = 8) (h2 : total_lives = 64) :
  total_lives / lives_per_player = 8 :=
by sorry

end NUMINAMATH_CALUDE_video_game_players_l1371_137134


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1371_137125

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1371_137125


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1371_137193

theorem inequality_equivalence (a b : ℝ) (h : a ≠ b) :
  ∀ x, a^2*x + b^2*(1-x) ≥ (a*x + b*(1-x))^2 ↔ 0 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1371_137193


namespace NUMINAMATH_CALUDE_solve_a_and_b_l1371_137156

def A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}

def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem solve_a_and_b :
  ∃ (a b : ℝ),
    (A ∪ B a b = {x | x > -2}) ∧
    (A ∩ B a b = {x | 1 < x ∧ x ≤ 3}) ∧
    a = -4 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_a_and_b_l1371_137156


namespace NUMINAMATH_CALUDE_square_division_exists_l1371_137160

theorem square_division_exists : ∃ (n : ℕ) (a b c : ℝ), 
  n > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ c^2 = n * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_square_division_exists_l1371_137160


namespace NUMINAMATH_CALUDE_inequality_implication_l1371_137143

theorem inequality_implication (a b : ℝ) (h : a < b) : a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1371_137143


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1371_137183

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1371_137183


namespace NUMINAMATH_CALUDE_median_and_mode_of_scores_l1371_137127

/-- Represents the score distribution of students in the competition -/
def score_distribution : List (Nat × Nat) :=
  [(85, 1), (88, 7), (90, 11), (93, 10), (94, 13), (97, 7), (99, 1)]

/-- The total number of students -/
def total_students : Nat := 50

/-- Calculates the median of the given score distribution -/
def median (dist : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Calculates the mode of the given score distribution -/
def mode (dist : List (Nat × Nat)) : Nat :=
  sorry

/-- Theorem stating that the median is 93 and the mode is 94 for the given distribution -/
theorem median_and_mode_of_scores :
  median score_distribution total_students = 93 ∧
  mode score_distribution = 94 :=
sorry

end NUMINAMATH_CALUDE_median_and_mode_of_scores_l1371_137127


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1371_137170

theorem polynomial_simplification (r : ℝ) :
  (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1371_137170


namespace NUMINAMATH_CALUDE_painted_cube_problem_l1371_137184

theorem painted_cube_problem (n : ℕ) : n > 0 →
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l1371_137184


namespace NUMINAMATH_CALUDE_P_equals_Q_l1371_137133

-- Define the sets P and Q
def P : Set ℕ := {2, 3}
def Q : Set ℕ := {3, 2}

-- Theorem stating that P and Q are equal
theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l1371_137133


namespace NUMINAMATH_CALUDE_square_roll_around_octagon_l1371_137153

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- Represents a square -/
def Square := RegularPolygon 4

/-- Represents an octagon -/
def Octagon := RegularPolygon 8

/-- Represents the position of a point on the edge of a square -/
inductive EdgePosition
  | Bottom
  | Left
  | Top
  | Right

/-- Calculates the final position of a point after rolling around an octagon -/
def finalPosition (start : EdgePosition) : EdgePosition :=
  start  -- This is a placeholder, the actual implementation would depend on the proof

theorem square_roll_around_octagon 
  (octagon : Octagon) 
  (square : Square) 
  (start_pos : EdgePosition) :
  start_pos = EdgePosition.Bottom → 
  finalPosition start_pos = EdgePosition.Bottom :=
by sorry

end NUMINAMATH_CALUDE_square_roll_around_octagon_l1371_137153


namespace NUMINAMATH_CALUDE_mans_upward_speed_l1371_137190

/-- Proves that given a man traveling with an average speed of 28.8 km/hr
    and a downward speed of 36 km/hr, his upward speed is 24 km/hr. -/
theorem mans_upward_speed
  (v_avg : ℝ) (v_down : ℝ) (h_avg : v_avg = 28.8)
  (h_down : v_down = 36) :
  let v_up := 2 * v_avg * v_down / (2 * v_down - v_avg)
  v_up = 24 := by sorry

end NUMINAMATH_CALUDE_mans_upward_speed_l1371_137190


namespace NUMINAMATH_CALUDE_three_day_trip_mileage_l1371_137185

theorem three_day_trip_mileage (total_miles : ℕ) (day1_miles : ℕ) (day2_miles : ℕ) 
  (h1 : total_miles = 493) 
  (h2 : day1_miles = 125) 
  (h3 : day2_miles = 223) : 
  total_miles - (day1_miles + day2_miles) = 145 := by
  sorry

end NUMINAMATH_CALUDE_three_day_trip_mileage_l1371_137185


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l1371_137194

theorem smallest_integer_gcd_18_is_6 : 
  ∃ (n : ℕ), n > 100 ∧ Nat.gcd n 18 = 6 ∧ ∀ m, m > 100 ∧ m < n → Nat.gcd m 18 ≠ 6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l1371_137194


namespace NUMINAMATH_CALUDE_athletes_seating_arrangements_l1371_137158

def number_of_arrangements (team_sizes : List Nat) : Nat :=
  (team_sizes.length.factorial) * (team_sizes.map Nat.factorial).prod

theorem athletes_seating_arrangements :
  number_of_arrangements [4, 3, 3] = 5184 := by
  sorry

end NUMINAMATH_CALUDE_athletes_seating_arrangements_l1371_137158


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l1371_137111

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l1371_137111


namespace NUMINAMATH_CALUDE_game_ends_with_two_l1371_137135

/-- Represents the state of the game board -/
structure GameBoard where
  ones : ℕ
  twos : ℕ

/-- Represents a move in the game -/
inductive Move
  | EraseOnes
  | EraseTwos
  | EraseOneTwo

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move with
  | Move.EraseOnes => { ones := board.ones - 2, twos := board.twos + 1 }
  | Move.EraseTwos => { ones := board.ones, twos := board.twos - 1 }
  | Move.EraseOneTwo => { ones := board.ones - 1, twos := board.twos }

/-- The initial state of the game board -/
def initialBoard : GameBoard := { ones := 10, twos := 10 }

/-- Predicate to check if the game is over -/
def gameOver (board : GameBoard) : Prop :=
  board.ones + board.twos = 1

/-- Theorem stating that the game always ends with a two -/
theorem game_ends_with_two :
  ∀ (sequence : List Move),
    let finalBoard := sequence.foldl applyMove initialBoard
    gameOver finalBoard → finalBoard.twos = 1 :=
  sorry

end NUMINAMATH_CALUDE_game_ends_with_two_l1371_137135


namespace NUMINAMATH_CALUDE_min_sum_arc_lengths_l1371_137162

/-- A set of points on a circle consisting of n arcs -/
structure CircleSet (n : ℕ) where
  arcs : Fin n → Set ℝ
  sum_lengths : ℝ

/-- Rotation of a set of points on a circle -/
def rotate (α : ℝ) (F : Set ℝ) : Set ℝ := sorry

/-- Property that for any rotation, the rotated set intersects with the original set -/
def intersects_all_rotations (F : Set ℝ) : Prop :=
  ∀ α : ℝ, (rotate α F ∩ F).Nonempty

/-- Theorem stating the minimum sum of arc lengths -/
theorem min_sum_arc_lengths (n : ℕ) (F : CircleSet n) 
  (h : intersects_all_rotations (⋃ i, F.arcs i)) :
  F.sum_lengths ≥ 180 / n := sorry

end NUMINAMATH_CALUDE_min_sum_arc_lengths_l1371_137162


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l1371_137138

/-- The number of ways to choose 3 boxes out of 4 -/
def choose_boxes : ℕ := 4

/-- The number of ways to distribute the extra white ball -/
def distribute_white : ℕ := 3

/-- The number of ways to distribute the extra black balls -/
def distribute_black : ℕ := 6

/-- The number of ways to distribute the extra red balls -/
def distribute_red : ℕ := 10

/-- The total number of ways to distribute the balls -/
def total_ways : ℕ := choose_boxes * distribute_white * distribute_black * distribute_red

theorem ball_distribution_theorem : total_ways = 720 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l1371_137138


namespace NUMINAMATH_CALUDE_num_paths_is_70_l1371_137192

/-- The number of paths from A to B on a grid with specific movement constraints -/
def num_paths : ℕ :=
  Nat.choose 8 4

/-- Theorem stating that the number of paths is 70 -/
theorem num_paths_is_70 : num_paths = 70 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_is_70_l1371_137192


namespace NUMINAMATH_CALUDE_nth_decimal_35_36_l1371_137195

/-- The fraction 35/36 as a real number -/
def f : ℚ := 35 / 36

/-- Predicate to check if the nth decimal digit of a rational number is 2 -/
def is_nth_decimal_2 (q : ℚ) (n : ℕ) : Prop :=
  (q * 10^n - ⌊q * 10^n⌋) * 10 ≥ 2 ∧ (q * 10^n - ⌊q * 10^n⌋) * 10 < 3

/-- Theorem stating that the nth decimal digit of 35/36 is 2 if and only if n ≥ 2 -/
theorem nth_decimal_35_36 (n : ℕ) : is_nth_decimal_2 f n ↔ n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_nth_decimal_35_36_l1371_137195


namespace NUMINAMATH_CALUDE_quadratic_function_transformation_l1371_137150

-- Define the quadratic function f(x) = ax² + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x) = cx² + 2bx + a
def g (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + 2 * b * x + a

theorem quadratic_function_transformation (a b c : ℝ) :
  (f a b c 0 = 1) ∧ 
  (f a b c 1 = -2) ∧ 
  (f a b c (-1) = 2) →
  (∀ x, g a b c x = x^2 - 4*x - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_transformation_l1371_137150


namespace NUMINAMATH_CALUDE_double_iced_subcubes_count_l1371_137109

/-- Represents a 3D cube with icing on some faces -/
structure IcedCube where
  size : Nat
  top_iced : Bool
  front_iced : Bool
  right_iced : Bool

/-- Counts the number of 1x1x1 subcubes with icing on exactly two faces -/
def count_double_iced_subcubes (cube : IcedCube) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem double_iced_subcubes_count (cake : IcedCube) : 
  cake.size = 5 ∧ cake.top_iced ∧ cake.front_iced ∧ cake.right_iced →
  count_double_iced_subcubes cake = 32 :=
by sorry

end NUMINAMATH_CALUDE_double_iced_subcubes_count_l1371_137109


namespace NUMINAMATH_CALUDE_sandwich_ratio_l1371_137116

theorem sandwich_ratio : ∀ (first_day : ℕ), 
  first_day + (first_day - 2) + 2 = 12 →
  (first_day : ℚ) / 12 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sandwich_ratio_l1371_137116


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1371_137178

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + y = 4}
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ P ∈ line, Real.sqrt (P.1^2 + P.2^2) ≥ d ∧
    ∃ Q ∈ line, Real.sqrt (Q.1^2 + Q.2^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1371_137178


namespace NUMINAMATH_CALUDE_graph_sequence_periodic_l1371_137104

/-- A graph on n vertices -/
def Graph (n : ℕ) := Fin n → Fin n → Prop

/-- The rule for constructing G_(n+1) from G_n -/
def nextGraph (G : Graph n) : Graph n :=
  λ i j => ∃ k, k ≠ i ∧ k ≠ j ∧ G k i ∧ G k j

/-- The sequence of graphs -/
def graphSequence (G₀ : Graph n) : ℕ → Graph n
  | 0 => G₀
  | m + 1 => nextGraph (graphSequence G₀ m)

/-- Two graphs are equal if they have the same edges -/
def graphEqual (G H : Graph n) : Prop :=
  ∀ i j, G i j ↔ H i j

theorem graph_sequence_periodic (n : ℕ) (G₀ : Graph n) :
  ∃ (m₀ T : ℕ), T ≤ 2^n ∧
    ∀ m ≥ m₀, graphEqual (graphSequence G₀ (m + T)) (graphSequence G₀ m) :=
sorry

end NUMINAMATH_CALUDE_graph_sequence_periodic_l1371_137104


namespace NUMINAMATH_CALUDE_nail_painting_problem_l1371_137130

theorem nail_painting_problem (total_nails purple_nails blue_nails : ℕ) 
  (h1 : total_nails = 20)
  (h2 : purple_nails = 6)
  (h3 : blue_nails = 8)
  (h4 : (blue_nails : ℚ) / total_nails - (striped_nails : ℚ) / total_nails = 1/10) :
  striped_nails = 6 :=
by
  sorry

#check nail_painting_problem

end NUMINAMATH_CALUDE_nail_painting_problem_l1371_137130


namespace NUMINAMATH_CALUDE_six_ways_to_make_50_yuan_l1371_137164

/-- The number of ways to make 50 yuan using 5 yuan and 10 yuan notes -/
def ways_to_make_50_yuan : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 5 * p.1 + 10 * p.2 = 50) (Finset.product (Finset.range 11) (Finset.range 6))).card

/-- Theorem stating that there are exactly 6 ways to make 50 yuan using 5 yuan and 10 yuan notes -/
theorem six_ways_to_make_50_yuan : ways_to_make_50_yuan = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_ways_to_make_50_yuan_l1371_137164


namespace NUMINAMATH_CALUDE_teachers_arrangement_count_l1371_137120

def number_of_seats : ℕ := 25
def number_of_teachers : ℕ := 5
def min_gap : ℕ := 2

def arrange_teachers (seats : ℕ) (teachers : ℕ) (gap : ℕ) : ℕ :=
  Nat.choose (seats + teachers - (teachers - 1) * (gap + 1)) teachers

theorem teachers_arrangement_count :
  arrange_teachers number_of_seats number_of_teachers min_gap = 26334 := by
  sorry

end NUMINAMATH_CALUDE_teachers_arrangement_count_l1371_137120


namespace NUMINAMATH_CALUDE_mango_rate_is_59_l1371_137131

/-- Calculates the rate per kg for mangoes given the total amount paid, grape price, grape weight, and mango weight. -/
def mango_rate (total_paid : ℕ) (grape_price : ℕ) (grape_weight : ℕ) (mango_weight : ℕ) : ℕ :=
  (total_paid - grape_price * grape_weight) / mango_weight

/-- Theorem stating that under the given conditions, the mango rate is 59 -/
theorem mango_rate_is_59 :
  mango_rate 975 74 6 9 = 59 := by
  sorry

#eval mango_rate 975 74 6 9

end NUMINAMATH_CALUDE_mango_rate_is_59_l1371_137131


namespace NUMINAMATH_CALUDE_root_sum_squares_l1371_137154

theorem root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 3 ∧ y^2 + 2*h*y = 3 ∧ x^2 + y^2 = 10) → 
  |h| = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1371_137154


namespace NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l1371_137105

theorem consecutive_squares_not_equal_consecutive_fourth_powers :
  ∀ a b : ℕ, a^2 + (a+1)^2 ≠ b^4 + (b+1)^4 := by sorry

end NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l1371_137105


namespace NUMINAMATH_CALUDE_triangle_max_area_l1371_137149

/-- Given a triangle ABC where AB = 9 and BC:AC = 3:4, 
    the maximum possible area of the triangle is 243 / (2√7) square units. -/
theorem triangle_max_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 9 ∧ BC / AC = 3 / 4 → 
  area ≤ 243 / (2 * Real.sqrt 7) := by
sorry


end NUMINAMATH_CALUDE_triangle_max_area_l1371_137149


namespace NUMINAMATH_CALUDE_prob_first_ace_equal_sum_prob_is_one_l1371_137142

/-- Represents a player in the card game -/
inductive Player : Type
| one : Player
| two : Player
| three : Player
| four : Player

/-- The total number of cards in the deck -/
def totalCards : ℕ := 32

/-- The number of aces in the deck -/
def numAces : ℕ := 4

/-- The number of players in the game -/
def numPlayers : ℕ := 4

/-- Calculates the probability of a player getting the first ace -/
def probFirstAce (p : Player) : ℚ :=
  1 / 8

/-- Theorem: The probability of each player getting the first ace is 1/8 -/
theorem prob_first_ace_equal (p : Player) : 
  probFirstAce p = 1 / 8 := by
  sorry

/-- Theorem: The sum of probabilities for all players is 1 -/
theorem sum_prob_is_one : 
  (probFirstAce Player.one) + (probFirstAce Player.two) + 
  (probFirstAce Player.three) + (probFirstAce Player.four) = 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_first_ace_equal_sum_prob_is_one_l1371_137142


namespace NUMINAMATH_CALUDE_parabola_properties_l1371_137119

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Define the directrix
def directrix (x y : ℝ) : Prop := y = -2

-- Define a point on the parabola
def on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define a point on the directrix
def on_directrix (p : ℝ × ℝ) : Prop := directrix p.1 p.2

-- Define the condition PF = FE
def PF_equals_FE (P E : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = (E.1 - focus.1)^2 + (E.2 - focus.2)^2

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem parabola_properties :
  ∀ (P E : ℝ × ℝ),
  on_directrix P →
  on_parabola E →
  E.1 > 0 →
  E.2 > 0 →
  PF_equals_FE P E →
  (∃ (k : ℝ), k * P.1 - P.2 + 2 = 0 ∧ k = 1/Real.sqrt 3) ∧
  (∀ (D : ℝ × ℝ), on_parabola D →
    dot_product (D.1 - P.1, D.2 - P.2) (E.1 - P.1, E.2 - P.2) ≤ -64) ∧
  (∃ (P' : ℝ × ℝ), on_directrix P' ∧
    (P'.1 = 4 ∨ P'.1 = -4) ∧ P'.2 = -2 ∧
    (∀ (D E : ℝ × ℝ), on_parabola D → on_parabola E →
      dot_product (D.1 - P'.1, D.2 - P'.2) (E.1 - P'.1, E.2 - P'.2) = -64)) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1371_137119


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1371_137148

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  solution_set : ∀ x : ℝ, (x < -2 ∨ x > 4) ↔ (a * x^2 + b * x + c > 0)

/-- The main theorem stating the inequality for specific x values -/
theorem quadratic_inequality (f : QuadraticFunction) :
  f.a * 2^2 + f.b * 2 + f.c < f.a * (-1)^2 + f.b * (-1) + f.c ∧
  f.a * (-1)^2 + f.b * (-1) + f.c < f.a * 5^2 + f.b * 5 + f.c :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1371_137148


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1371_137197

theorem complex_sum_equality : 
  8 * Complex.exp (2 * π * I / 13) + 8 * Complex.exp (15 * π * I / 26) = 
  8 * Real.sqrt 3 * Complex.exp (19 * π * I / 52) := by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1371_137197


namespace NUMINAMATH_CALUDE_planes_parallel_if_perp_to_parallel_lines_l1371_137110

-- Define the types for planes and lines
variable (α β : Plane) (l m : Line)

-- Define the relationships between planes and lines
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- State the theorem
theorem planes_parallel_if_perp_to_parallel_lines 
  (h1 : perpendicular l α) 
  (h2 : perpendicular m β) 
  (h3 : parallel_lines l m) : 
  parallel_planes α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perp_to_parallel_lines_l1371_137110


namespace NUMINAMATH_CALUDE_system_solution_l1371_137123

theorem system_solution :
  ∃ (x y : ℝ), 
    (2 * x + Real.sqrt (2 * x + 3 * y) - 3 * y = 5) ∧
    (4 * x^2 + 2 * x + 3 * y - 9 * y^2 = 32) ∧
    (x = 17/4) ∧ (y = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1371_137123


namespace NUMINAMATH_CALUDE_complex_power_simplification_l1371_137167

theorem complex_power_simplification :
  ((2 + Complex.I) / (2 - Complex.I)) ^ 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l1371_137167


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l1371_137186

/-- Sum of tens and ones digits of (3+4)^11 -/
theorem sum_of_digits_of_power : ∃ (n : ℕ), 
  (3 + 4)^11 = n ∧ 
  (n / 10 % 10 + n % 10 = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l1371_137186


namespace NUMINAMATH_CALUDE_table_height_proof_l1371_137124

/-- Given two configurations of a table and two identical wooden blocks,
    prove that the height of the table is 30 inches. -/
theorem table_height_proof (x y : ℝ) : 
  x + 30 - y = 32 ∧ y + 30 - x = 28 → 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_table_height_proof_l1371_137124


namespace NUMINAMATH_CALUDE_smallest_Y_value_l1371_137137

/-- A function that checks if a natural number consists only of digits 0 and 1 -/
def only_zero_and_one (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem stating the smallest possible value of Y -/
theorem smallest_Y_value (S : ℕ) (hS : S > 0) (h_digits : only_zero_and_one S) (h_div : S % 15 = 0) :
  (S / 15 : ℕ) ≥ 74 :=
sorry

end NUMINAMATH_CALUDE_smallest_Y_value_l1371_137137


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1371_137191

theorem expression_simplification_and_evaluation :
  let x := Real.tan (45 * π / 180) + Real.cos (30 * π / 180)
  (x / (x^2 - 1)) * ((x - 1) / x - 2) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1371_137191


namespace NUMINAMATH_CALUDE_quadratic_condition_l1371_137151

/-- The condition for a quadratic equation in x with parameter m -/
def is_quadratic_in_x (m : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, m * x^2 - 3*x = x^2 - m*x + 2 ↔ a * x^2 + b * x + c = 0

/-- Theorem stating that for the given equation to be quadratic in x, m must not equal 1 -/
theorem quadratic_condition (m : ℝ) : is_quadratic_in_x m → m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l1371_137151


namespace NUMINAMATH_CALUDE_manny_marbles_after_sharing_l1371_137182

/-- Given a total number of marbles and a ratio, calculates the number of marbles for each part -/
def marbles_per_part (total : ℕ) (ratio_sum : ℕ) : ℕ := total / ratio_sum

/-- Calculates the initial number of marbles for a person given their ratio part and marbles per part -/
def initial_marbles (ratio_part : ℕ) (marbles_per_part : ℕ) : ℕ := ratio_part * marbles_per_part

/-- Calculates the final number of marbles after giving away some -/
def final_marbles (initial : ℕ) (given_away : ℕ) : ℕ := initial - given_away

theorem manny_marbles_after_sharing (total_marbles : ℕ) (mario_ratio : ℕ) (manny_ratio : ℕ) (shared_marbles : ℕ) :
  total_marbles = 36 →
  mario_ratio = 4 →
  manny_ratio = 5 →
  shared_marbles = 2 →
  final_marbles (initial_marbles manny_ratio (marbles_per_part total_marbles (mario_ratio + manny_ratio))) shared_marbles = 18 := by
  sorry

end NUMINAMATH_CALUDE_manny_marbles_after_sharing_l1371_137182


namespace NUMINAMATH_CALUDE_square_area_increase_l1371_137187

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.2 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.44
  := by sorry

end NUMINAMATH_CALUDE_square_area_increase_l1371_137187


namespace NUMINAMATH_CALUDE_tamika_always_wins_l1371_137106

def tamika_set : Finset ℕ := {11, 12, 13}
def carlos_set : Finset ℕ := {4, 6, 7}

theorem tamika_always_wins :
  ∀ (a b : ℕ) (c d : ℕ),
    a ∈ tamika_set → b ∈ tamika_set → a ≠ b →
    c ∈ carlos_set → d ∈ carlos_set → c ≠ d →
    a * b > c * d := by
  sorry

#check tamika_always_wins

end NUMINAMATH_CALUDE_tamika_always_wins_l1371_137106


namespace NUMINAMATH_CALUDE_sine_monotonicity_l1371_137115

theorem sine_monotonicity (φ : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ |f (π / 6)|) (h3 : f (π / 2) > f π) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sine_monotonicity_l1371_137115


namespace NUMINAMATH_CALUDE_no_xy_term_iff_k_eq_four_l1371_137118

/-- The polynomial multiplication (x+2y)(2x-ky-1) does not contain the term xy if and only if k = 4 -/
theorem no_xy_term_iff_k_eq_four (k : ℝ) : 
  (∀ x y : ℝ, (x + 2*y) * (2*x - k*y - 1) = 2*x^2 - x - 2*k*y^2 - 2*y) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_iff_k_eq_four_l1371_137118


namespace NUMINAMATH_CALUDE_article_price_l1371_137163

theorem article_price (P : ℝ) : 
  P > 0 →                            -- Initial price is positive
  0.9 * (0.8 * P) = 36 →             -- Final price after discounts is $36
  P = 50 :=                          -- Initial price is $50
by sorry

end NUMINAMATH_CALUDE_article_price_l1371_137163


namespace NUMINAMATH_CALUDE_hyperbola_equation_for_given_conditions_l1371_137166

/-- A hyperbola with given eccentricity and foci -/
structure Hyperbola where
  eccentricity : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Theorem: A hyperbola with eccentricity 2 and foci at (-4,0) and (4,0) has the equation x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_for_given_conditions (h : Hyperbola) 
    (h_ecc : h.eccentricity = 2)
    (h_foci : h.focus1 = (-4, 0) ∧ h.focus2 = (4, 0)) :
    ∀ x y : ℝ, hyperbola_equation h x y :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_for_given_conditions_l1371_137166


namespace NUMINAMATH_CALUDE_unique_abc_solution_l1371_137117

/-- Represents a base-7 number with two digits -/
def Base7TwoDigit (a b : Nat) : Nat := 7 * a + b

/-- Represents a base-7 number with one digit -/
def Base7OneDigit (c : Nat) : Nat := c

theorem unique_abc_solution :
  ∀ A B C : Nat,
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigit C 0 →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
    A = 5 ∧ B = 1 ∧ C = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_abc_solution_l1371_137117


namespace NUMINAMATH_CALUDE_water_transfer_problem_l1371_137145

theorem water_transfer_problem (left_initial right_initial : ℕ) 
  (difference_after_transfer : ℕ) (h1 : left_initial = 2800) 
  (h2 : right_initial = 1500) (h3 : difference_after_transfer = 360) :
  ∃ (x : ℕ), x = 470 ∧ 
  left_initial - x = right_initial + x + difference_after_transfer :=
sorry

end NUMINAMATH_CALUDE_water_transfer_problem_l1371_137145


namespace NUMINAMATH_CALUDE_bookstore_earnings_difference_l1371_137141

/-- Represents the earnings difference between two books --/
def earnings_difference (price_top : ℕ) (price_abc : ℕ) (quantity_top : ℕ) (quantity_abc : ℕ) : ℕ :=
  (price_top * quantity_top) - (price_abc * quantity_abc)

/-- Theorem: The earnings difference between "TOP" and "ABC" books is $12 --/
theorem bookstore_earnings_difference :
  earnings_difference 8 23 13 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_earnings_difference_l1371_137141


namespace NUMINAMATH_CALUDE_infinitely_many_an_power_an_mod_8_l1371_137103

theorem infinitely_many_an_power_an_mod_8 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ (8 * n + 3)^(8 * n + 3) ≡ 8 * n + 3 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_an_power_an_mod_8_l1371_137103


namespace NUMINAMATH_CALUDE_mba_committee_size_l1371_137189

theorem mba_committee_size 
  (total_mbas : ℕ) 
  (num_committees : ℕ) 
  (prob_same_committee : ℚ) :
  total_mbas = 6 ∧ 
  num_committees = 2 ∧ 
  prob_same_committee = 2/5 →
  ∃ (committee_size : ℕ), 
    committee_size * num_committees = total_mbas ∧
    committee_size = 3 :=
by sorry

end NUMINAMATH_CALUDE_mba_committee_size_l1371_137189


namespace NUMINAMATH_CALUDE_drums_hit_calculation_l1371_137132

/-- Represents the drumming contest scenario --/
structure DrummingContest where
  entryFee : ℝ
  costPerDrum : ℝ
  earningsStartDrum : ℕ
  earningsPerDrum : ℝ
  bonusRoundDrum : ℕ
  totalLoss : ℝ

/-- Calculates the number of drums hit in the contest --/
def drumsHit (contest : DrummingContest) : ℕ :=
  sorry

/-- Theorem stating the number of drums hit in the given scenario --/
theorem drums_hit_calculation (contest : DrummingContest) 
  (h1 : contest.entryFee = 10)
  (h2 : contest.costPerDrum = 0.02)
  (h3 : contest.earningsStartDrum = 200)
  (h4 : contest.earningsPerDrum = 0.025)
  (h5 : contest.bonusRoundDrum = 250)
  (h6 : contest.totalLoss = 7.5) :
  drumsHit contest = 4500 :=
sorry

end NUMINAMATH_CALUDE_drums_hit_calculation_l1371_137132


namespace NUMINAMATH_CALUDE_no_natural_number_with_sum_of_squared_divisors_perfect_square_l1371_137100

theorem no_natural_number_with_sum_of_squared_divisors_perfect_square :
  ¬ ∃ (n : ℕ), ∃ (d₁ d₂ d₃ d₄ d₅ : ℕ), 
    (∀ d : ℕ, d ∣ n → d ≥ d₅) ∧ 
    (d₁ ∣ n) ∧ (d₂ ∣ n) ∧ (d₃ ∣ n) ∧ (d₄ ∣ n) ∧ (d₅ ∣ n) ∧
    (d₁ < d₂) ∧ (d₂ < d₃) ∧ (d₃ < d₄) ∧ (d₄ < d₅) ∧
    ∃ (m : ℕ), d₁^2 + d₂^2 + d₃^2 + d₄^2 + d₅^2 = m^2 :=
by
  sorry


end NUMINAMATH_CALUDE_no_natural_number_with_sum_of_squared_divisors_perfect_square_l1371_137100


namespace NUMINAMATH_CALUDE_ace_diamond_probability_l1371_137173

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Diamonds in a standard deck -/
def NumDiamonds : ℕ := 13

/-- Probability of drawing an Ace as the first card and a Diamond as the second card -/
def prob_ace_then_diamond (deck : ℕ) (aces : ℕ) (diamonds : ℕ) : ℚ :=
  (aces : ℚ) / (deck : ℚ) * (diamonds : ℚ) / ((deck - 1) : ℚ)

theorem ace_diamond_probability :
  prob_ace_then_diamond StandardDeck NumAces NumDiamonds = 1 / StandardDeck :=
sorry

end NUMINAMATH_CALUDE_ace_diamond_probability_l1371_137173


namespace NUMINAMATH_CALUDE_icosahedron_edges_l1371_137174

/-- A regular icosahedron is a polyhedron with 20 faces and 12 vertices, 
    where each vertex is connected to 5 edges. -/
structure RegularIcosahedron where
  faces : ℕ
  vertices : ℕ
  edges_per_vertex : ℕ
  faces_eq : faces = 20
  vertices_eq : vertices = 12
  edges_per_vertex_eq : edges_per_vertex = 5

/-- The number of edges in a regular icosahedron is 30. -/
theorem icosahedron_edges (i : RegularIcosahedron) : 
  (i.vertices * i.edges_per_vertex) / 2 = 30 := by
  sorry

#check icosahedron_edges

end NUMINAMATH_CALUDE_icosahedron_edges_l1371_137174


namespace NUMINAMATH_CALUDE_specific_triangle_area_l1371_137128

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The ratio of equal sides to base (represented as two integers)
  ratio_equal_to_base : ℕ × ℕ
  -- Condition: altitude is positive
  altitude_pos : altitude > 0
  -- Condition: perimeter is positive
  perimeter_pos : perimeter > 0
  -- Condition: ratio components are positive
  ratio_pos : ratio_equal_to_base.1 > 0 ∧ ratio_equal_to_base.2 > 0

/-- The area of an isosceles triangle with given properties -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles triangle is 75 -/
theorem specific_triangle_area :
  ∃ t : IsoscelesTriangle,
    t.altitude = 10 ∧
    t.perimeter = 40 ∧
    t.ratio_equal_to_base = (5, 3) ∧
    triangle_area t = 75 :=
  sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l1371_137128


namespace NUMINAMATH_CALUDE_point_in_intersection_l1371_137157

-- Define the sets U, A, and B
def U : Set (ℝ × ℝ) := Set.univ
def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem point_in_intersection (m n : ℝ) :
  P ∈ A m ∩ (U \ B n) ↔ m > -1 ∧ n < 5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_intersection_l1371_137157


namespace NUMINAMATH_CALUDE_dagger_example_l1371_137161

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (2 * q / n)

-- Theorem statement
theorem dagger_example : dagger (5/9) (7/6) = 140/3 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l1371_137161


namespace NUMINAMATH_CALUDE_abel_overtake_distance_l1371_137179

/-- Represents the race scenario between Abel and Kelly -/
structure RaceScenario where
  totalDistance : ℝ
  headStart : ℝ
  lossDistance : ℝ

/-- Calculates the distance Abel needs to run to overtake Kelly -/
def distanceToOvertake (race : RaceScenario) : ℝ :=
  race.totalDistance - (race.totalDistance - race.headStart + race.lossDistance)

/-- Theorem stating that Abel needs to run 98 meters to overtake Kelly -/
theorem abel_overtake_distance (race : RaceScenario) 
  (h1 : race.totalDistance = 100)
  (h2 : race.headStart = 3)
  (h3 : race.lossDistance = 0.5) :
  distanceToOvertake race = 98 := by
  sorry

#eval distanceToOvertake { totalDistance := 100, headStart := 3, lossDistance := 0.5 }

end NUMINAMATH_CALUDE_abel_overtake_distance_l1371_137179


namespace NUMINAMATH_CALUDE_dandelion_counts_l1371_137171

/-- Represents the state of dandelions in a meadow on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- Represents the lifecycle of dandelions over three consecutive days -/
structure DandelionLifecycle :=
  (dayBeforeYesterday : DandelionState)
  (yesterday : DandelionState)
  (today : DandelionState)

/-- The theorem statement -/
theorem dandelion_counts 
  (lifecycle : DandelionLifecycle)
  (h1 : lifecycle.yesterday.yellow = 20)
  (h2 : lifecycle.yesterday.white = 14)
  (h3 : lifecycle.today.yellow = 15)
  (h4 : lifecycle.today.white = 11) :
  lifecycle.dayBeforeYesterday.yellow = 25 ∧ 
  lifecycle.today.yellow - (lifecycle.yesterday.white - lifecycle.today.white) = 9 :=
by sorry

end NUMINAMATH_CALUDE_dandelion_counts_l1371_137171


namespace NUMINAMATH_CALUDE_product_97_103_l1371_137181

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l1371_137181


namespace NUMINAMATH_CALUDE_sqrt_23_parts_x_minus_y_value_l1371_137198

-- Part 1: Integer and decimal parts of √23
theorem sqrt_23_parts :
  ∃ (n : ℕ) (d : ℝ), n = 4 ∧ d = Real.sqrt 23 - 4 ∧
  Real.sqrt 23 = n + d ∧ 0 ≤ d ∧ d < 1 := by sorry

-- Part 2: x-y given 9+√3=x+y
theorem x_minus_y_value (x : ℤ) (y : ℝ) 
  (h1 : 9 + Real.sqrt 3 = x + y)
  (h2 : 0 < y) (h3 : y < 1) :
  x - y = 11 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_23_parts_x_minus_y_value_l1371_137198


namespace NUMINAMATH_CALUDE_expression_equality_l1371_137113

theorem expression_equality : (-1)^2 - |(-3)| + (-5) / (-5/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1371_137113


namespace NUMINAMATH_CALUDE_fraction_simplification_l1371_137146

theorem fraction_simplification (x : ℝ) : (x - 1) / 3 + (-2 - 3 * x) / 2 = (-7 * x - 8) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1371_137146


namespace NUMINAMATH_CALUDE_trapezoid_QR_squared_l1371_137172

/-- Represents a trapezoid PQRS with specific properties -/
structure Trapezoid where
  PQ : ℝ
  PS : ℝ
  RS : ℝ
  QR : ℝ
  perp_QR_PQ_RS : True  -- QR is perpendicular to PQ and RS
  perp_diagonals : True -- Diagonals PR and QS are perpendicular

/-- The theorem stating the properties of the specific trapezoid and its conclusion -/
theorem trapezoid_QR_squared (T : Trapezoid) 
  (h1 : T.PQ = Real.sqrt 41)
  (h2 : T.PS = Real.sqrt 2001)
  (h3 : T.PQ + T.RS = Real.sqrt 2082) :
  T.QR ^ 2 = 410 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_QR_squared_l1371_137172


namespace NUMINAMATH_CALUDE_first_four_digits_after_decimal_l1371_137196

theorem first_four_digits_after_decimal (x : ℝ) : 
  x = (5^1001 + 2)^(5/3) → 
  ∃ n : ℕ, 0 ≤ n ∧ n < 10000 ∧ (x - ⌊x⌋) * 10000 = 3333 + n / 10000 :=
sorry

end NUMINAMATH_CALUDE_first_four_digits_after_decimal_l1371_137196


namespace NUMINAMATH_CALUDE_student_committee_candidates_l1371_137180

theorem student_committee_candidates (n : ℕ) : n * (n - 1) = 72 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_candidates_l1371_137180


namespace NUMINAMATH_CALUDE_system_solution_unique_l1371_137169

theorem system_solution_unique (x y z : ℚ) : 
  x + 2*y - z = 100 ∧
  y - z = 25 ∧
  3*x - 5*y + 4*z = 230 →
  x = 101.25 ∧ y = -26.25 ∧ z = -51.25 := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1371_137169


namespace NUMINAMATH_CALUDE_at_least_two_equal_l1371_137165

theorem at_least_two_equal (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2/y + y^2/z + z^2/x = x^2/z + y^2/x + z^2/y) :
  x = y ∨ y = z ∨ z = x := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l1371_137165


namespace NUMINAMATH_CALUDE_monthly_cost_correct_l1371_137108

/-- Represents the monthly cost for online access -/
def monthly_cost : ℝ := 8

/-- Represents the initial app cost -/
def app_cost : ℝ := 5

/-- Represents the number of months of online access -/
def months : ℝ := 2

/-- Represents the total cost for the app and online access -/
def total_cost : ℝ := 21

/-- Proves that the monthly cost for online access is correct -/
theorem monthly_cost_correct : app_cost + months * monthly_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_monthly_cost_correct_l1371_137108


namespace NUMINAMATH_CALUDE_track_width_is_25_feet_l1371_137102

/-- Represents the radii of three concentric circles forming a running track -/
structure TrackRadii where
  inner : ℝ
  middle : ℝ
  outer : ℝ

/-- The width of the track is the difference between the outer and inner radii -/
def track_width (r : TrackRadii) : ℝ := r.outer - r.inner

/-- The theorem states that given the specified differences in circumferences,
    the total width of the track is 25 feet -/
theorem track_width_is_25_feet (r : TrackRadii)
  (h1 : 2 * Real.pi * r.middle - 2 * Real.pi * r.inner = 20 * Real.pi)
  (h2 : 2 * Real.pi * r.outer - 2 * Real.pi * r.middle = 30 * Real.pi) :
  track_width r = 25 := by
  sorry


end NUMINAMATH_CALUDE_track_width_is_25_feet_l1371_137102


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_power_2024_plus_2025_l1371_137114

theorem tens_digit_of_2023_power_2024_plus_2025 :
  (2023^2024 + 2025) % 100 / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_power_2024_plus_2025_l1371_137114


namespace NUMINAMATH_CALUDE_triangle_operation_result_l1371_137107

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a * b / (-6)

-- State the theorem
theorem triangle_operation_result :
  triangle 4 (triangle 3 2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_operation_result_l1371_137107


namespace NUMINAMATH_CALUDE_folded_rope_length_l1371_137199

/-- Represents the length of a rope folded three times -/
structure FoldedRope where
  total_length : ℝ
  distance_1_3 : ℝ

/-- The properties of a rope folded three times as described in the problem -/
def is_valid_folded_rope (rope : FoldedRope) : Prop :=
  rope.distance_1_3 = rope.total_length / 4

/-- The main theorem stating the relationship between the distance between points (1) and (3)
    and the total length of the rope -/
theorem folded_rope_length (rope : FoldedRope) 
  (h : is_valid_folded_rope rope) 
  (h_distance : rope.distance_1_3 = 30) : 
  rope.total_length = 120 := by
  sorry

#check folded_rope_length

end NUMINAMATH_CALUDE_folded_rope_length_l1371_137199


namespace NUMINAMATH_CALUDE_speed_ratio_proof_l1371_137121

/-- Proves that the speed ratio of return to outbound trip is 2:1 given specific conditions -/
theorem speed_ratio_proof (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) : 
  total_distance = 40 ∧ 
  total_time = 6 ∧ 
  return_speed = 10 → 
  (return_speed / (total_distance / 2 / (total_time - total_distance / 2 / return_speed))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_proof_l1371_137121


namespace NUMINAMATH_CALUDE_possible_m_values_l1371_137139

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- Define the theorem
theorem possible_m_values :
  ∀ m : ℝ, (B m ⊆ A m) → (m = 0 ∨ m = 3) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_possible_m_values_l1371_137139


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1371_137144

/-- Proves that the speed of a boat in still water is 20 km/hr given the specified conditions -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed : ℝ),
    (boat_speed + 5) * 0.4 = 10 →
    boat_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1371_137144
