import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l3039_303977

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 > x

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3039_303977


namespace NUMINAMATH_CALUDE_parabola_translation_l3039_303976

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 1 4 (-4)
  let p_translated := translate p 2 (-3)
  y = x^2 + 4*x - 4 →
  y = (x + 4)^2 - 11 ↔
  y = p_translated.a * x^2 + p_translated.b * x + p_translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3039_303976


namespace NUMINAMATH_CALUDE_smaller_city_size_l3039_303905

/-- Proves that given a population density of 80 people per cubic yard, 
    if a larger city with 9000 cubic yards has 208000 more people than a smaller city, 
    then the smaller city has 6400 cubic yards. -/
theorem smaller_city_size (density : ℕ) (larger_city_size : ℕ) (population_difference : ℕ) :
  density = 80 →
  larger_city_size = 9000 →
  population_difference = 208000 →
  (larger_city_size * density) - (population_difference) = 6400 * density :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_city_size_l3039_303905


namespace NUMINAMATH_CALUDE_max_two_digit_div_sum_of_digits_l3039_303995

theorem max_two_digit_div_sum_of_digits :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    ¬(a = 0 ∧ b = 0) →
    (10 * a + b) / (a + b) ≤ 10 ∧
    ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ¬(a = 0 ∧ b = 0) ∧ (10 * a + b) / (a + b) = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_two_digit_div_sum_of_digits_l3039_303995


namespace NUMINAMATH_CALUDE_mike_bought_two_for_friend_l3039_303957

/-- Represents the problem of calculating the number of rose bushes Mike bought for his friend. -/
def mike_rose_bushes_for_friend 
  (total_rose_bushes : ℕ)
  (rose_bush_price : ℕ)
  (total_aloes : ℕ)
  (aloe_price : ℕ)
  (spent_on_self : ℕ) : ℕ :=
  total_rose_bushes - (spent_on_self - total_aloes * aloe_price) / rose_bush_price

/-- Theorem stating that Mike bought 2 rose bushes for his friend. -/
theorem mike_bought_two_for_friend :
  mike_rose_bushes_for_friend 6 75 2 100 500 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mike_bought_two_for_friend_l3039_303957


namespace NUMINAMATH_CALUDE_range_of_f_l3039_303909

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x < 0 then x^2 + 6*x
  else 0  -- We define f as 0 outside the given intervals

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-9 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3039_303909


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l3039_303986

/-- Proves that a train of length 240 m takes 24 seconds to pass a pole, given that it takes 89 seconds to pass a 650 m platform -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 240)
  (h2 : platform_length = 650)
  (h3 : time_to_pass_platform = 89) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l3039_303986


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l3039_303975

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (78/19, 41/19)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3*x - 2*y = 8

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 5*x + 3*y = 27

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l3039_303975


namespace NUMINAMATH_CALUDE_sum_of_squares_l3039_303945

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2*n*b^2 = k^2) :
  ∃ x y : ℕ, a^2 + n*b^2 = x^2 + y^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3039_303945


namespace NUMINAMATH_CALUDE_twelfth_even_multiple_of_4_l3039_303920

/-- The nth term in the sequence of positive integers that are both even and multiples of 4 -/
def evenMultipleOf4 (n : ℕ) : ℕ := 4 * n

/-- Theorem stating that the 12th term in the sequence of positive integers 
    that are both even and multiples of 4 is equal to 48 -/
theorem twelfth_even_multiple_of_4 : evenMultipleOf4 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_even_multiple_of_4_l3039_303920


namespace NUMINAMATH_CALUDE_characterization_of_solutions_l3039_303916

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) * (f x - f y) = (x - y) * f x * f y

/-- The main theorem stating the form of functions satisfying the equation -/
theorem characterization_of_solutions :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ C : ℝ, C ≠ 0 ∧ ∀ x : ℝ, f x = C * x :=
by sorry

end NUMINAMATH_CALUDE_characterization_of_solutions_l3039_303916


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3039_303980

/-- The number of girls in the school -/
def num_girls : ℕ := 739

/-- The number of boys in the school -/
def num_boys : ℕ := 337

/-- The difference between the number of girls and boys -/
def difference : ℕ := num_girls - num_boys

theorem more_girls_than_boys : difference = 402 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3039_303980


namespace NUMINAMATH_CALUDE_curve_decomposition_l3039_303933

-- Define the curve
def curve (x y : ℝ) : Prop := (x + y - 1) * Real.sqrt (x - 1) = 0

-- Define the line x = 1
def line (x y : ℝ) : Prop := x = 1

-- Define the ray x + y - 1 = 0 where x ≥ 1
def ray (x y : ℝ) : Prop := x + y - 1 = 0 ∧ x ≥ 1

-- Theorem statement
theorem curve_decomposition :
  ∀ x y : ℝ, x ≥ 1 → (curve x y ↔ line x y ∨ ray x y) :=
sorry

end NUMINAMATH_CALUDE_curve_decomposition_l3039_303933


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_one_l3039_303979

/-- A geometric sequence with negative terms and a specific sum condition has a common ratio of 1. -/
theorem geometric_sequence_common_ratio_one 
  (a : ℕ+ → ℝ) 
  (h_geometric : ∀ n : ℕ+, a (n + 1) = a n * q) 
  (h_negative : ∀ n : ℕ+, a n < 0) 
  (h_sum : a 3 + a 7 ≥ 2 * a 5) : 
  q = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_one_l3039_303979


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3039_303913

theorem unique_integer_solution : 
  ∀ x y : ℤ, x^2 - 2*x*y + 2*y^2 - 4*y^3 = 0 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3039_303913


namespace NUMINAMATH_CALUDE_game_ends_in_one_round_l3039_303934

/-- Represents a player in the game -/
inductive Player : Type
  | A | B | C | D

/-- The state of the game, containing the token count for each player -/
structure GameState :=
  (tokens : Player → Nat)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := fun p => match p with
    | Player.A => 8
    | Player.B => 9
    | Player.C => 10
    | Player.D => 11 }

/-- Determines if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p, state.tokens p = 0

/-- Determines the player with the most tokens -/
def playerWithMostTokens (state : GameState) : Player :=
  sorry

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Theorem: The game ends after 1 round -/
theorem game_ends_in_one_round :
  gameEnded (playRound initialState) :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_one_round_l3039_303934


namespace NUMINAMATH_CALUDE_box_width_l3039_303938

/-- The width of a rectangular box given its dimensions and cube properties -/
theorem box_width (length height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h1 : length = 10)
  (h2 : height = 4)
  (h3 : cube_volume = 12)
  (h4 : min_cubes = 60) :
  (min_cubes : ℝ) * cube_volume / (length * height) = 18 := by
  sorry

end NUMINAMATH_CALUDE_box_width_l3039_303938


namespace NUMINAMATH_CALUDE_kenny_mushroom_pieces_l3039_303968

/-- The number of mushroom pieces Kenny used on his pizza -/
def kenny_pieces (total_mushrooms : ℕ) (pieces_per_mushroom : ℕ) (karla_pieces : ℕ) (remaining_pieces : ℕ) : ℕ :=
  total_mushrooms * pieces_per_mushroom - (karla_pieces + remaining_pieces)

/-- Theorem stating the number of mushroom pieces Kenny used -/
theorem kenny_mushroom_pieces :
  kenny_pieces 22 4 42 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_kenny_mushroom_pieces_l3039_303968


namespace NUMINAMATH_CALUDE_larger_number_problem_l3039_303983

theorem larger_number_problem (x y : ℕ) : 
  x * y = 30 → x + y = 13 → max x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3039_303983


namespace NUMINAMATH_CALUDE_ship_grain_calculation_l3039_303941

/-- The amount of grain spilled into the water, in tons -/
def grain_spilled : ℕ := 49952

/-- The amount of grain remaining onboard, in tons -/
def grain_remaining : ℕ := 918

/-- The original amount of grain on the ship, in tons -/
def original_grain : ℕ := grain_spilled + grain_remaining

theorem ship_grain_calculation :
  original_grain = 50870 :=
sorry

end NUMINAMATH_CALUDE_ship_grain_calculation_l3039_303941


namespace NUMINAMATH_CALUDE_room_length_l3039_303947

/-- Given a rectangular room with width 4 meters and a paving cost of 950 per square meter
    resulting in a total cost of 20900, the length of the room is 5.5 meters. -/
theorem room_length (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) : 
  width = 4 →
  cost_per_sqm = 950 →
  total_cost = 20900 →
  total_cost = cost_per_sqm * (length * width) →
  length = 5.5 := by
sorry


end NUMINAMATH_CALUDE_room_length_l3039_303947


namespace NUMINAMATH_CALUDE_hayden_ironing_time_l3039_303988

/-- The total time Hayden spends ironing over 4 weeks -/
def total_ironing_time (shirt_time pants_time days_per_week num_weeks : ℕ) : ℕ :=
  (shirt_time + pants_time) * days_per_week * num_weeks

/-- Proof that Hayden spends 160 minutes ironing over 4 weeks -/
theorem hayden_ironing_time :
  total_ironing_time 5 3 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_hayden_ironing_time_l3039_303988


namespace NUMINAMATH_CALUDE_max_value_constraint_l3039_303965

theorem max_value_constraint (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : x^2 + y^2 ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3039_303965


namespace NUMINAMATH_CALUDE_sin_shift_left_l3039_303922

theorem sin_shift_left (x : ℝ) : 
  Real.sin (x + π/4) = Real.sin (x - (-π/4)) := by sorry

end NUMINAMATH_CALUDE_sin_shift_left_l3039_303922


namespace NUMINAMATH_CALUDE_statistics_test_probability_l3039_303908

def word : String := "STATISTICS"
def test_word : String := "TEST"

def letter_count (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem statistics_test_probability :
  let total_tiles := word.length
  let overlapping_tiles := (test_word.toList.eraseDups.filter (λ c => word.contains c))
                            |>.map (λ c => letter_count word c)
                            |>.sum
  (↑overlapping_tiles : ℚ) / total_tiles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_statistics_test_probability_l3039_303908


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3039_303914

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop := a * x - b * y + 8 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem min_value_of_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_line_passes_center : line_eq a b (circle_center.1) (circle_center.2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → line_eq a' b' (circle_center.1) (circle_center.2) → 
    1/a + 1/b ≤ 1/a' + 1/b') ∧ 1/a + 1/b = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3039_303914


namespace NUMINAMATH_CALUDE_set_equality_implies_subset_l3039_303960

theorem set_equality_implies_subset (A B C : Set α) :
  A ∪ B = B ∩ C → A ⊆ C := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_subset_l3039_303960


namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l3039_303949

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of x-axis and y-axis -/
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_graph_is_axes : S = T := by sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l3039_303949


namespace NUMINAMATH_CALUDE_cricket_runs_l3039_303932

theorem cricket_runs (a b c : ℕ) (h1 : 3 * a = b) (h2 : 5 * b = c) (h3 : a + b + c = 95) :
  c = 75 := by
  sorry

end NUMINAMATH_CALUDE_cricket_runs_l3039_303932


namespace NUMINAMATH_CALUDE_negation_of_existence_l3039_303900

theorem negation_of_existence (x : ℝ) : 
  ¬(∃ x > 0, Real.log x > 0) ↔ (∀ x > 0, Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3039_303900


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l3039_303926

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 15

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 588

theorem green_pill_cost_proof :
  green_pill_cost = 15 ∧
  pink_pill_cost = green_pill_cost - 2 ∧
  treatment_days * (green_pill_cost + pink_pill_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l3039_303926


namespace NUMINAMATH_CALUDE_max_value_of_f_l3039_303958

open Real

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ y ∈ Set.Ioo 0 (π / 2), 8 * sin y - tan y ≤ max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3039_303958


namespace NUMINAMATH_CALUDE_union_condition_equiv_range_l3039_303981

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem union_condition_equiv_range (a : ℝ) :
  A a ∪ B = B ↔ a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_union_condition_equiv_range_l3039_303981


namespace NUMINAMATH_CALUDE_valid_solutions_l3039_303906

def is_valid_solution (xyz : ℕ) : Prop :=
  xyz ≥ 100 ∧ xyz ≤ 999 ∧ (456000 + xyz) % 504 = 0

theorem valid_solutions :
  ∀ xyz : ℕ, is_valid_solution xyz ↔ (xyz = 120 ∨ xyz = 624) :=
sorry

end NUMINAMATH_CALUDE_valid_solutions_l3039_303906


namespace NUMINAMATH_CALUDE_multiple_calculation_l3039_303911

theorem multiple_calculation (number : ℝ) (value : ℝ) (multiple : ℝ) : 
  number = -4.5 →
  value = 36 →
  10 * number = value - multiple * number →
  multiple = -18 := by
sorry

end NUMINAMATH_CALUDE_multiple_calculation_l3039_303911


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l3039_303966

/-- Calculates the sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the sum of an arithmetic sequence --/
def sum_arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

/-- The number of days Murtha collects pebbles --/
def total_days : ℕ := 15

/-- The number of days Murtha skips collecting pebbles --/
def skipped_days : ℕ := total_days / 3

/-- Theorem: Murtha's pebble collection after 15 days --/
theorem murtha_pebble_collection :
  sum_first_n total_days - sum_arithmetic_seq 3 3 skipped_days = 75 := by
  sorry

#eval sum_first_n total_days - sum_arithmetic_seq 3 3 skipped_days

end NUMINAMATH_CALUDE_murtha_pebble_collection_l3039_303966


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3039_303940

/-- Given a circle C with equation x^2 - 8y - 7 = -y^2 - 6x, 
    prove that the sum of its center coordinates and radius is 1 + 4√2 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), x^2 - 8*y - 7 = -y^2 - 6*x → (x - a)^2 + (y - b)^2 = r^2) →
    a + b + r = 1 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3039_303940


namespace NUMINAMATH_CALUDE_number_puzzle_l3039_303950

theorem number_puzzle (x : ℝ) : (((3/4 * x) - 25) / 7) + 50 = 100 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3039_303950


namespace NUMINAMATH_CALUDE_election_winner_votes_l3039_303972

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 : ℚ) / 100 * total_votes - (38 : ℚ) / 100 * total_votes = 288) :
  (62 : ℚ) / 100 * total_votes = 744 := by
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3039_303972


namespace NUMINAMATH_CALUDE_leaf_movement_l3039_303942

theorem leaf_movement (forward_distance : ℕ) (num_gusts : ℕ) (total_distance : ℕ) 
  (h1 : forward_distance = 5)
  (h2 : num_gusts = 11)
  (h3 : total_distance = 33) :
  ∃ (backward_distance : ℕ), 
    num_gusts * (forward_distance - backward_distance) = total_distance ∧ 
    backward_distance = 2 :=
by sorry

end NUMINAMATH_CALUDE_leaf_movement_l3039_303942


namespace NUMINAMATH_CALUDE_total_balls_correct_l3039_303903

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 6

/-- The probability of drawing a yellow ball -/
def yellow_probability : ℚ := 3/10

/-- The total number of balls in the bag -/
def total_balls : ℕ := 20

/-- Theorem stating that the total number of balls is correct given the conditions -/
theorem total_balls_correct : 
  (yellow_balls : ℚ) / total_balls = yellow_probability :=
by sorry

end NUMINAMATH_CALUDE_total_balls_correct_l3039_303903


namespace NUMINAMATH_CALUDE_perimeter_710_implies_n_66_l3039_303989

/-- Represents the perimeter of the nth figure in the sequence -/
def perimeter (n : ℕ) : ℕ := 60 + (n - 1) * 10

/-- Theorem stating that if the perimeter of the nth figure is 710 cm, then n is 66 -/
theorem perimeter_710_implies_n_66 : ∃ n : ℕ, perimeter n = 710 ∧ n = 66 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_710_implies_n_66_l3039_303989


namespace NUMINAMATH_CALUDE_contrapositive_zero_product_l3039_303959

theorem contrapositive_zero_product (a b : ℝ) :
  (¬(a = 0 ∨ b = 0) → ab ≠ 0) ↔ (ab = 0 → a = 0 ∨ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_zero_product_l3039_303959


namespace NUMINAMATH_CALUDE_probability_not_losing_l3039_303937

theorem probability_not_losing (p_win p_draw : ℝ) 
  (h_win : p_win = 0.3) 
  (h_draw : p_draw = 0.2) : 
  p_win + p_draw = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_losing_l3039_303937


namespace NUMINAMATH_CALUDE_circle_center_l3039_303970

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- The center of a circle -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 9

/-- Theorem: The center of the circle is (1, 3) -/
theorem circle_center : is_center 1 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3039_303970


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3039_303985

theorem mod_congruence_unique_solution :
  ∃! n : ℕ, n ≤ 6 ∧ n ≡ -7845 [ZMOD 7] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3039_303985


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l3039_303910

theorem smallest_n_divisibility (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 360 → (¬(54 ∣ m^2) ∨ ¬(1280 ∣ m^3))) ∧ 
  (54 ∣ 360^2) ∧ (1280 ∣ 360^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l3039_303910


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3039_303998

theorem square_area_from_diagonal (d : ℝ) (h : d = 3.8) :
  (d^2 / 2) = 7.22 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3039_303998


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l3039_303967

/-- Calculates the total number of ice cream cones sold in a week based on given sales pattern -/
def total_ice_cream_sales (monday : ℕ) (tuesday : ℕ) : ℕ :=
  let wednesday := 2 * tuesday
  let thursday := (3 * wednesday) / 2
  let friday := (3 * thursday) / 4
  let weekend := 2 * friday
  monday + tuesday + wednesday + thursday + friday + weekend

/-- Theorem stating that the total ice cream sales for the week is 163,000 -/
theorem ice_cream_sales_theorem : total_ice_cream_sales 10000 12000 = 163000 := by
  sorry

#eval total_ice_cream_sales 10000 12000

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l3039_303967


namespace NUMINAMATH_CALUDE_intersection_shape_circumference_l3039_303919

/-- The circumference of the shape formed by intersecting quarter circles in a square -/
theorem intersection_shape_circumference (π : ℝ) (side_length : ℝ) : 
  π = 3.141 → side_length = 2 → (4 * π) / 3 = 4.188 := by sorry

end NUMINAMATH_CALUDE_intersection_shape_circumference_l3039_303919


namespace NUMINAMATH_CALUDE_bobs_grade_is_35_l3039_303969

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℚ := jason_grade / 2

theorem bobs_grade_is_35 : bob_grade = 35 := by sorry

end NUMINAMATH_CALUDE_bobs_grade_is_35_l3039_303969


namespace NUMINAMATH_CALUDE_rectangle_area_difference_main_theorem_l3039_303961

theorem rectangle_area_difference : ℕ → Prop :=
fun diff =>
  (∃ (l w : ℕ), l + w = 30 ∧ l * w = 225) ∧  -- Largest area
  (∃ (l w : ℕ), l + w = 30 ∧ l * w = 29) ∧  -- Smallest area
  diff = 225 - 29

theorem main_theorem : rectangle_area_difference 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_main_theorem_l3039_303961


namespace NUMINAMATH_CALUDE_gp_common_ratio_l3039_303978

/-- Given a geometric progression where the ratio of the sum of the first 6 terms
    to the sum of the first 3 terms is 217, prove that the common ratio is 6. -/
theorem gp_common_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217 →
  r = 6 := by
sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l3039_303978


namespace NUMINAMATH_CALUDE_square_sum_equality_l3039_303991

theorem square_sum_equality : 107 * 107 + 93 * 93 = 20098 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3039_303991


namespace NUMINAMATH_CALUDE_not_always_zero_l3039_303939

-- Define the heart operation
def heart (x y : ℝ) : ℝ := |x + y|

-- Theorem stating that the statement is false
theorem not_always_zero : ¬ ∀ x : ℝ, heart x x = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_always_zero_l3039_303939


namespace NUMINAMATH_CALUDE_fraction_equality_with_different_numerator_denominator_relations_l3039_303927

theorem fraction_equality_with_different_numerator_denominator_relations : 
  ∃ (a b c d : ℤ), a < b ∧ c > d ∧ (a : ℚ) / b = (c : ℚ) / d := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_with_different_numerator_denominator_relations_l3039_303927


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l3039_303963

theorem smallest_sum_arithmetic_geometric (A B C D : ℤ) : 
  (∃ d : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →           -- B, C, D form a geometric sequence
  (C = (4 * B) / 3) →         -- C/B = 4/3
  (∀ A' B' C' D' : ℤ, 
    (∃ d' : ℤ, C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' = (4 * B') / 3) → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l3039_303963


namespace NUMINAMATH_CALUDE_david_pushups_count_l3039_303987

def zachary_pushups : ℕ := 35

def david_pushups : ℕ := zachary_pushups + 9

theorem david_pushups_count : david_pushups = 44 := by sorry

end NUMINAMATH_CALUDE_david_pushups_count_l3039_303987


namespace NUMINAMATH_CALUDE_max_value_of_f_l3039_303923

noncomputable def a (x m : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, m + Real.cos x)

noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, -m + Real.cos x)

noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem max_value_of_f (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (-π/6) (π/3), f x₀ m = -4) →
  (∃ x₁ ∈ Set.Icc (-π/6) (π/3), ∀ x ∈ Set.Icc (-π/6) (π/3), f x m ≤ f x₁ m) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x m ≤ -3/2) ∧
  f (π/6) m = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3039_303923


namespace NUMINAMATH_CALUDE_sloth_shoe_theorem_l3039_303946

/-- The number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- The number of complete sets of shoes desired -/
def desired_sets : ℕ := 5

/-- The number of sets of shoes already owned -/
def owned_sets : ℕ := 1

/-- Calculate the number of pairs of shoes needed to be purchased -/
def shoes_to_buy : ℕ :=
  (desired_sets * sloth_feet - owned_sets * sloth_feet) / 2

theorem sloth_shoe_theorem : shoes_to_buy = 6 := by
  sorry

end NUMINAMATH_CALUDE_sloth_shoe_theorem_l3039_303946


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l3039_303915

theorem sqrt_expressions_equality : 
  (Real.sqrt 8 - Real.sqrt (1/2) + Real.sqrt 18 = (9 * Real.sqrt 2) / 2) ∧ 
  ((Real.sqrt 2 + Real.sqrt 3)^2 - Real.sqrt 24 = 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l3039_303915


namespace NUMINAMATH_CALUDE_reading_pages_in_week_l3039_303952

/-- Calculates the total number of pages read in a week -/
def pages_read_in_week (morning_pages : ℕ) (evening_pages : ℕ) (days_in_week : ℕ) : ℕ :=
  (morning_pages + evening_pages) * days_in_week

/-- Theorem: Reading 5 pages in the morning and 10 pages in the evening for a week results in 105 pages read -/
theorem reading_pages_in_week :
  pages_read_in_week 5 10 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_reading_pages_in_week_l3039_303952


namespace NUMINAMATH_CALUDE_sub_committee_count_l3039_303982

/-- The number of people in the committee -/
def totalPeople : ℕ := 8

/-- The size of each sub-committee -/
def subCommitteeSize : ℕ := 2

/-- The number of people who cannot be in the same sub-committee -/
def restrictedPair : ℕ := 1

/-- The number of valid two-person sub-committees -/
def validSubCommittees : ℕ := 27

theorem sub_committee_count :
  (Nat.choose totalPeople subCommitteeSize) - restrictedPair = validSubCommittees :=
sorry

end NUMINAMATH_CALUDE_sub_committee_count_l3039_303982


namespace NUMINAMATH_CALUDE_james_age_is_35_l3039_303990

/-- The age James turned when John turned 35 -/
def james_age : ℕ := sorry

/-- John's age when James turned james_age -/
def john_age : ℕ := 35

/-- Tim's current age -/
def tim_age : ℕ := 79

theorem james_age_is_35 : james_age = 35 :=
  by
    have h1 : tim_age = 2 * john_age - 5 := by sorry
    have h2 : james_age = john_age := by sorry
    sorry

#check james_age_is_35

end NUMINAMATH_CALUDE_james_age_is_35_l3039_303990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3039_303930

/-- Proves that 1, 3, and 5 form a monotonically increasing arithmetic sequence with -1 and 7 -/
theorem arithmetic_sequence_proof : 
  let sequence := [-1, 1, 3, 5, 7]
  (∀ i : Fin 4, sequence[i] < sequence[i+1]) ∧ 
  (∃ d : ℤ, ∀ i : Fin 4, sequence[i+1] - sequence[i] = d) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3039_303930


namespace NUMINAMATH_CALUDE_twentyfour_game_solution_l3039_303936

/-- A type representing the allowed arithmetic operations -/
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

/-- A type representing an arithmetic expression -/
inductive Expr
  | Const (n : Int)
  | BinOp (op : Operation) (e1 e2 : Expr)

/-- Evaluate an expression -/
def eval : Expr → Int
  | Expr.Const n => n
  | Expr.BinOp Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.BinOp Operation.Sub e1 e2 => eval e1 - eval e2
  | Expr.BinOp Operation.Mul e1 e2 => eval e1 * eval e2
  | Expr.BinOp Operation.Div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses all given numbers exactly once -/
def usesAllNumbers (e : Expr) (nums : List Int) : Bool :=
  match e with
  | Expr.Const n => nums == [n]
  | Expr.BinOp _ e1 e2 =>
    let nums1 := nums.filter (λ n => n ∉ collectNumbers e2)
    let nums2 := nums.filter (λ n => n ∉ collectNumbers e1)
    usesAllNumbers e1 nums1 && usesAllNumbers e2 nums2
where
  collectNumbers : Expr → List Int
    | Expr.Const n => [n]
    | Expr.BinOp _ e1 e2 => collectNumbers e1 ++ collectNumbers e2

theorem twentyfour_game_solution :
  ∃ (e : Expr), usesAllNumbers e [3, -5, 6, -8] ∧ eval e = 24 := by
  sorry

end NUMINAMATH_CALUDE_twentyfour_game_solution_l3039_303936


namespace NUMINAMATH_CALUDE_lawsuit_probability_comparison_l3039_303931

theorem lawsuit_probability_comparison :
  let p1_win : ℝ := 0.30
  let p2_win : ℝ := 0.50
  let p3_win : ℝ := 0.40
  let p4_win : ℝ := 0.25
  
  let p1_lose : ℝ := 1 - p1_win
  let p2_lose : ℝ := 1 - p2_win
  let p3_lose : ℝ := 1 - p3_win
  let p4_lose : ℝ := 1 - p4_win
  
  let p_win_all : ℝ := p1_win * p2_win * p3_win * p4_win
  let p_lose_all : ℝ := p1_lose * p2_lose * p3_lose * p4_lose
  
  (p_lose_all - p_win_all) / p_win_all = 9.5
:= by sorry

end NUMINAMATH_CALUDE_lawsuit_probability_comparison_l3039_303931


namespace NUMINAMATH_CALUDE_fractional_method_min_experiments_l3039_303955

/-- The number of division points in the temperature range -/
def division_points : ℕ := 33

/-- The minimum number of experiments needed -/
def min_experiments : ℕ := 7

/-- Theorem stating the minimum number of experiments needed for the given conditions -/
theorem fractional_method_min_experiments :
  ∃ (n : ℕ), 2^n - 1 ≥ division_points ∧ n = min_experiments :=
sorry

end NUMINAMATH_CALUDE_fractional_method_min_experiments_l3039_303955


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l3039_303929

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 200 →
  capacity_ratio = 4 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 250 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l3039_303929


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3039_303996

theorem max_value_sqrt_sum (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
    Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≥ Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y))) ∧
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3039_303996


namespace NUMINAMATH_CALUDE_exam_attendance_calculation_l3039_303974

theorem exam_attendance_calculation (total_topics : ℕ) 
  (all_topics_pass_percent : ℚ) (no_topic_pass_percent : ℚ)
  (one_topic_pass_percent : ℚ) (two_topics_pass_percent : ℚ)
  (four_topics_pass_percent : ℚ) (three_topics_pass_count : ℕ)
  (h1 : total_topics = 5)
  (h2 : all_topics_pass_percent = 1/10)
  (h3 : no_topic_pass_percent = 1/10)
  (h4 : one_topic_pass_percent = 1/5)
  (h5 : two_topics_pass_percent = 1/4)
  (h6 : four_topics_pass_percent = 6/25)
  (h7 : three_topics_pass_count = 500) :
  ∃ total_students : ℕ, total_students = 4546 ∧
  (all_topics_pass_percent + no_topic_pass_percent + one_topic_pass_percent + 
   two_topics_pass_percent + four_topics_pass_percent) * total_students + 
   three_topics_pass_count = total_students :=
by sorry

end NUMINAMATH_CALUDE_exam_attendance_calculation_l3039_303974


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3039_303992

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from a pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  area pan.panDimensions / area pan.pieceDimensions

/-- Theorem stating that a 24x15 inch pan can be divided into exactly 60 pieces of 3x2 inch brownies -/
theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3039_303992


namespace NUMINAMATH_CALUDE_correct_calculation_l3039_303917

theorem correct_calculation (x : ℝ) : 8 * x + 8 = 56 → (x / 8) + 7 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3039_303917


namespace NUMINAMATH_CALUDE_star_three_neg_four_star_not_commutative_l3039_303907

-- Define the new operation "*" for rational numbers
def star (a b : ℚ) : ℚ := 2 * a - 1 + b

-- Theorem 1: 3 * (-4) = 1
theorem star_three_neg_four : star 3 (-4) = 1 := by sorry

-- Theorem 2: 7 * (-3) ≠ (-3) * 7
theorem star_not_commutative : star 7 (-3) ≠ star (-3) 7 := by sorry

end NUMINAMATH_CALUDE_star_three_neg_four_star_not_commutative_l3039_303907


namespace NUMINAMATH_CALUDE_intersection_equals_subset_implies_a_values_l3039_303956

def A (a : ℝ) : Set ℝ := {x | x - a = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equals_subset_implies_a_values (a : ℝ) 
  (h : A a ∩ B a = B a) : 
  a = 1 ∨ a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_subset_implies_a_values_l3039_303956


namespace NUMINAMATH_CALUDE_total_volume_is_85_l3039_303994

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The total volume of n cubes, each with side length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * (cube_volume s)

/-- Carl's cubes -/
def carl_cubes : ℕ := 3
def carl_side_length : ℝ := 3

/-- Kate's cubes -/
def kate_cubes : ℕ := 4
def kate_side_length : ℝ := 1

/-- The theorem stating that the total volume of Carl's and Kate's cubes is 85 -/
theorem total_volume_is_85 : 
  total_volume carl_cubes carl_side_length + total_volume kate_cubes kate_side_length = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_is_85_l3039_303994


namespace NUMINAMATH_CALUDE_angle_complement_half_supplement_l3039_303918

theorem angle_complement_half_supplement (x : ℝ) : 
  (90 - x) = (1/2) * (180 - x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_half_supplement_l3039_303918


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3039_303997

/-- A quadratic function f(x) = mx^2 - 4x + 1 has exactly one root if and only if m ≤ 4 -/
theorem quadratic_one_root (m : ℝ) :
  (∃! x, m * x^2 - 4 * x + 1 = 0) ↔ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3039_303997


namespace NUMINAMATH_CALUDE_cost_of_one_milk_carton_l3039_303928

/-- The cost of 1 one-litre carton of milk, given that 4 cartons cost $4.88 -/
theorem cost_of_one_milk_carton :
  let total_cost : ℚ := 488/100  -- $4.88 represented as a rational number
  let num_cartons : ℕ := 4
  let cost_per_carton : ℚ := total_cost / num_cartons
  cost_per_carton = 122/100  -- $1.22 represented as a rational number
:= by sorry

end NUMINAMATH_CALUDE_cost_of_one_milk_carton_l3039_303928


namespace NUMINAMATH_CALUDE_intersection_implies_range_140_l3039_303984

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 6)^2 + (y - 3)^2 = 7^2
def circle2 (x y k : ℝ) : Prop := (x - 2)^2 + (y - 6)^2 = k + 40

-- Define the intersection condition
def intersect (k : ℝ) : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y k

-- Theorem statement
theorem intersection_implies_range_140 (a b : ℝ) :
  (∀ k : ℝ, a ≤ k ∧ k ≤ b → intersect k) → b - a = 140 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_range_140_l3039_303984


namespace NUMINAMATH_CALUDE_barbara_shopping_l3039_303902

/-- The amount spent on goods other than tuna and water in Barbara's shopping trip -/
def other_goods_cost (tuna_packs : ℕ) (tuna_price : ℚ) (water_bottles : ℕ) (water_price : ℚ) (total_cost : ℚ) : ℚ :=
  total_cost - (tuna_packs * tuna_price + water_bottles * water_price)

/-- Theorem stating that Barbara spent $40 on goods other than tuna and water -/
theorem barbara_shopping :
  other_goods_cost 5 2 4 (3/2) 56 = 40 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_l3039_303902


namespace NUMINAMATH_CALUDE_line_equations_l3039_303971

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y - 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Define the line passing through the origin
def line_through_origin (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the line parallel to 2x + y + 5 = 0
def parallel_line (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the line perpendicular to 2x + y + 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 5 = 0

theorem line_equations :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = M) →
  (line_through_origin M.1 M.2) ∧
  (parallel_line M.1 M.2) ∧
  (perpendicular_line M.1 M.2) := by sorry

end NUMINAMATH_CALUDE_line_equations_l3039_303971


namespace NUMINAMATH_CALUDE_percentage_difference_l3039_303935

theorem percentage_difference : (40 / 100 * 60) - (4 / 5 * 25) = 4 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3039_303935


namespace NUMINAMATH_CALUDE_base7_5213_to_base10_l3039_303924

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- The base 10 representation of 5213₇ is 1823 -/
theorem base7_5213_to_base10 : base7ToBase10 5 2 1 3 = 1823 := by
  sorry

end NUMINAMATH_CALUDE_base7_5213_to_base10_l3039_303924


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l3039_303999

theorem min_k_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0) → k ≥ -4) ∧
  (∃ k : ℝ, k = -4 ∧ 1 / a + 1 / b + k / (a + b) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l3039_303999


namespace NUMINAMATH_CALUDE_distance_not_unique_l3039_303973

/-- Given two segments AB and BC with lengths 4 and 3 respectively, 
    prove that the length of AC cannot be uniquely determined. -/
theorem distance_not_unique (A B C : ℝ × ℝ) 
  (hAB : dist A B = 4) 
  (hBC : dist B C = 3) : 
  ¬ ∃! d, dist A C = d :=
sorry

end NUMINAMATH_CALUDE_distance_not_unique_l3039_303973


namespace NUMINAMATH_CALUDE_target_average_income_l3039_303948

def past_incomes : List ℝ := [406, 413, 420, 436, 395]
def next_two_weeks_avg : ℝ := 365
def total_weeks : ℕ := 7

theorem target_average_income :
  let total_past_income := past_incomes.sum
  let total_next_two_weeks := 2 * next_two_weeks_avg
  let total_income := total_past_income + total_next_two_weeks
  total_income / total_weeks = 400 := by
  sorry

end NUMINAMATH_CALUDE_target_average_income_l3039_303948


namespace NUMINAMATH_CALUDE_dannys_chickens_l3039_303921

/-- Calculates the number of chickens on Dany's farm -/
theorem dannys_chickens (cows sheep : ℕ) (cow_sheep_bushels chicken_bushels total_bushels : ℕ) : 
  cows = 4 →
  sheep = 3 →
  cow_sheep_bushels = 2 →
  chicken_bushels = 3 →
  total_bushels = 35 →
  (cows + sheep) * cow_sheep_bushels + (total_bushels - (cows + sheep) * cow_sheep_bushels) / chicken_bushels = 7 := by
  sorry

end NUMINAMATH_CALUDE_dannys_chickens_l3039_303921


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3039_303954

theorem digit_equation_solution : ∃ (Θ : ℕ), 
  Θ ≤ 9 ∧ 
  252 / Θ = 40 + 2 * Θ ∧ 
  Θ = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3039_303954


namespace NUMINAMATH_CALUDE_abrahams_shopping_budget_l3039_303951

/-- Abraham's shopping problem -/
theorem abrahams_shopping_budget (budget : ℕ) 
  (shower_gel_price shower_gel_quantity : ℕ) 
  (toothpaste_price laundry_detergent_price : ℕ) : 
  budget = 60 →
  shower_gel_price = 4 →
  shower_gel_quantity = 4 →
  toothpaste_price = 3 →
  laundry_detergent_price = 11 →
  budget - (shower_gel_price * shower_gel_quantity + toothpaste_price + laundry_detergent_price) = 30 := by
  sorry


end NUMINAMATH_CALUDE_abrahams_shopping_budget_l3039_303951


namespace NUMINAMATH_CALUDE_profit_percentage_l3039_303912

theorem profit_percentage (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.89 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.89 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3039_303912


namespace NUMINAMATH_CALUDE_smallest_a_value_l3039_303943

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  17 ≤ a ∧ ∀ a' : ℝ, (0 ≤ a' ∧ (∀ x : ℤ, Real.sin (a' * x + b) = Real.sin (17 * x))) → a' ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3039_303943


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l3039_303944

theorem x_plus_2y_equals_10 (x y : ℝ) (h1 : x + y = 19) (h2 : x + 3*y = 1) : 
  x + 2*y = 10 := by
sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l3039_303944


namespace NUMINAMATH_CALUDE_regular_polygon_945_diagonals_has_45_sides_l3039_303993

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 945 diagonals has 45 sides -/
theorem regular_polygon_945_diagonals_has_45_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 945 → n = 45 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_945_diagonals_has_45_sides_l3039_303993


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3039_303904

theorem a_plus_b_value (a b : ℝ) (ha : |a| = 5) (hb : |b| = 2) (hab : a < b) :
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3039_303904


namespace NUMINAMATH_CALUDE_appropriate_word_count_appropriate_lengths_l3039_303953

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration := { d : ℝ // 20 ≤ d ∧ d ≤ 30 }

/-- The optimal speaking rate in words per minute -/
def OptimalSpeakingRate : ℝ := 135

/-- Calculates the number of words for a given duration at the optimal speaking rate -/
def WordCount (duration : PresentationDuration) : ℝ :=
  duration.val * OptimalSpeakingRate

/-- Theorem stating that the appropriate word count is between 2700 and 4050 -/
theorem appropriate_word_count (duration : PresentationDuration) :
  2700 ≤ WordCount duration ∧ WordCount duration ≤ 4050 := by
  sorry

/-- Theorem stating that 3000 and 3700 words are appropriate lengths for the presentation -/
theorem appropriate_lengths :
  ∃ (d1 d2 : PresentationDuration), WordCount d1 = 3000 ∧ WordCount d2 = 3700 := by
  sorry

end NUMINAMATH_CALUDE_appropriate_word_count_appropriate_lengths_l3039_303953


namespace NUMINAMATH_CALUDE_combined_selling_price_l3039_303964

/-- Calculate the combined selling price of two articles given their costs, desired profits, tax rate, and packaging fees. -/
theorem combined_selling_price
  (cost_A cost_B : ℚ)
  (profit_rate_A profit_rate_B : ℚ)
  (tax_rate : ℚ)
  (packaging_fee : ℚ) :
  cost_A = 500 →
  cost_B = 800 →
  profit_rate_A = 1/10 →
  profit_rate_B = 3/20 →
  tax_rate = 1/20 →
  packaging_fee = 50 →
  ∃ (selling_price : ℚ),
    selling_price = 
      (cost_A + cost_A * profit_rate_A) * (1 + tax_rate) + packaging_fee +
      (cost_B + cost_B * profit_rate_B) * (1 + tax_rate) + packaging_fee ∧
    selling_price = 1643.5 := by
  sorry

#check combined_selling_price

end NUMINAMATH_CALUDE_combined_selling_price_l3039_303964


namespace NUMINAMATH_CALUDE_unique_solution_iff_nonzero_l3039_303925

theorem unique_solution_iff_nonzero (a : ℝ) :
  (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_nonzero_l3039_303925


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3039_303901

theorem binomial_coefficient_ratio (m n : ℕ) : 
  (Nat.choose (n + 1) (m + 1) : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 3 →
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) (m - 1) : ℚ) = 5 / 3 →
  m = 3 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3039_303901


namespace NUMINAMATH_CALUDE_vector_sum_coordinates_l3039_303962

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum_coordinates : 2 • a + b = (-3, 4) := by sorry

end NUMINAMATH_CALUDE_vector_sum_coordinates_l3039_303962
