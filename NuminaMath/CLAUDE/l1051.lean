import Mathlib

namespace olympic_arrangements_correct_l1051_105127

/-- The number of ways to arrange athletes in Olympic lanes -/
def olympicArrangements : ℕ := 2520

/-- The number of lanes -/
def numLanes : ℕ := 8

/-- The number of countries -/
def numCountries : ℕ := 4

/-- The number of athletes per country -/
def athletesPerCountry : ℕ := 2

/-- Theorem: The number of ways to arrange the athletes is correct -/
theorem olympic_arrangements_correct :
  olympicArrangements = (numLanes.choose athletesPerCountry) *
                        ((numLanes - athletesPerCountry).choose athletesPerCountry) *
                        ((numLanes - 2 * athletesPerCountry).choose athletesPerCountry) *
                        ((numLanes - 3 * athletesPerCountry).choose athletesPerCountry) :=
by sorry

end olympic_arrangements_correct_l1051_105127


namespace q_investment_value_l1051_105122

/-- Represents the investment and profit division of two business partners -/
structure BusinessInvestment where
  p_investment : ℝ
  q_investment : ℝ
  profit_ratio : ℝ × ℝ

/-- Given the conditions of the problem, prove that q's investment is 45000 -/
theorem q_investment_value (b : BusinessInvestment) 
  (h1 : b.p_investment = 30000)
  (h2 : b.profit_ratio = (2, 3)) :
  b.q_investment = 45000 := by
  sorry

end q_investment_value_l1051_105122


namespace leahs_coins_value_l1051_105181

theorem leahs_coins_value (n p : ℕ) : 
  n + p = 13 →                   -- Total number of coins is 13
  n + 1 = p →                    -- One more nickel would equal pennies
  5 * n + p = 37                 -- Total value in cents
  := by sorry

end leahs_coins_value_l1051_105181


namespace figure_rearrangeable_to_square_l1051_105115

/-- A figure on graph paper can be rearranged into a square if and only if 
    its area (in unit squares) is a perfect square. -/
theorem figure_rearrangeable_to_square (n : ℕ) : 
  (∃ (k : ℕ), n = k^2) ↔ (∃ (s : ℕ), s^2 = n) :=
sorry

end figure_rearrangeable_to_square_l1051_105115


namespace non_collinear_triples_count_l1051_105164

/-- The total number of points -/
def total_points : ℕ := 60

/-- The number of collinear triples -/
def collinear_triples : ℕ := 30

/-- The number of ways to choose three points from the total points -/
def total_triples : ℕ := total_points.choose 3

/-- The number of ways to choose three non-collinear points -/
def non_collinear_triples : ℕ := total_triples - collinear_triples

theorem non_collinear_triples_count : non_collinear_triples = 34190 := by
  sorry

end non_collinear_triples_count_l1051_105164


namespace pizza_fraction_l1051_105113

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice = 1/3 →
  (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end pizza_fraction_l1051_105113


namespace fixed_point_exists_P_on_parabola_l1051_105191

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def line_through (P Q : Point) (x y : ℝ) : Prop :=
  (y - P.y) * (Q.x - P.x) = (x - P.x) * (Q.y - P.y)

-- Define the external angle bisector of ∠APB
def external_angle_bisector (A P B : Point) (x y : ℝ) : Prop :=
  sorry -- Definition of external angle bisector

-- Define a tangent line to the parabola
def tangent_to_parabola (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = 2*(x - x₀)/y₀

-- Main theorem
theorem fixed_point_exists : ∃ (Q : Point), 
  ∀ (A B : Point), 
    parabola A.x A.y → 
    parabola B.x B.y → 
    line_through Q A A.x A.y → 
    line_through Q B B.x B.y → 
    ∃ (T : Point), 
      parabola T.x T.y ∧ 
      external_angle_bisector A P B T.x T.y ∧ 
      tangent_to_parabola T.x T.y T.x T.y :=
by
  -- Proof goes here
  sorry

-- Specific point P satisfies the parabola equation
theorem P_on_parabola : parabola 1 (-2) :=
by
  -- Proof goes here
  sorry

-- Q is the fixed point
def Q : Point := ⟨-3, 2⟩

end fixed_point_exists_P_on_parabola_l1051_105191


namespace total_tickets_won_l1051_105180

/-- Represents the number of tickets Dave used for toys. -/
def tickets_for_toys : ℕ := 8

/-- Represents the number of tickets Dave used for clothes. -/
def tickets_for_clothes : ℕ := 18

/-- Represents the difference in tickets used for clothes versus toys. -/
def difference_clothes_toys : ℕ := 10

/-- Theorem stating that the total number of tickets Dave won is the sum of
    tickets used for toys and clothes. -/
theorem total_tickets_won (hw : tickets_for_clothes = tickets_for_toys + difference_clothes_toys) :
  tickets_for_toys + tickets_for_clothes = 26 := by
  sorry

#check total_tickets_won

end total_tickets_won_l1051_105180


namespace trent_tears_per_three_onions_l1051_105165

def tears_per_three_onions (pots : ℕ) (onions_per_pot : ℕ) (total_tears : ℕ) : ℚ :=
  (3 * total_tears : ℚ) / (pots * onions_per_pot : ℚ)

theorem trent_tears_per_three_onions :
  tears_per_three_onions 6 4 16 = 2 := by
  sorry

end trent_tears_per_three_onions_l1051_105165


namespace polynomial_product_expansion_l1051_105100

theorem polynomial_product_expansion :
  let p₁ : Polynomial ℝ := 5 * X^2 + 3 * X - 4
  let p₂ : Polynomial ℝ := 6 * X^3 + 2 * X^2 - X + 7
  p₁ * p₂ = 30 * X^5 + 28 * X^4 - 23 * X^3 + 24 * X^2 + 25 * X - 28 := by
sorry

end polynomial_product_expansion_l1051_105100


namespace first_to_light_is_match_l1051_105170

/-- Represents items that can be lit --/
inductive LightableItem
  | Match
  | Candle
  | KeroseneLamp
  | Stove

/-- Represents the state of a room --/
structure Room where
  isDark : Bool
  hasMatch : Bool
  items : List LightableItem

/-- Determines the first item that must be lit in a given room --/
def firstItemToLight (room : Room) : LightableItem := by sorry

/-- Theorem: The first item to light in a dark room with a match and other lightable items is the match itself --/
theorem first_to_light_is_match (room : Room) 
  (h1 : room.isDark = true) 
  (h2 : room.hasMatch = true) 
  (h3 : LightableItem.Candle ∈ room.items) 
  (h4 : LightableItem.KeroseneLamp ∈ room.items) 
  (h5 : LightableItem.Stove ∈ room.items) : 
  firstItemToLight room = LightableItem.Match := by sorry

end first_to_light_is_match_l1051_105170


namespace first_player_wins_l1051_105154

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a piece on the chessboard --/
structure Piece :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a player in the game --/
inductive Player
| First
| Second

/-- Defines the game state --/
structure GameState :=
  (board : Chessboard)
  (firstPiece : Piece)
  (secondPiece : Piece)
  (currentPlayer : Player)

/-- Defines a winning strategy for a player --/
def WinningStrategy (player : Player) (game : GameState) : Prop :=
  ∃ (strategy : GameState → ℕ × ℕ), 
    ∀ (opponent_move : ℕ × ℕ), 
      player = game.currentPlayer → 
      ∃ (next_state : GameState), 
        next_state.currentPlayer ≠ player ∧ 
        (∃ (final_state : GameState), final_state.currentPlayer = player ∧ ¬∃ (move : ℕ × ℕ), true)

/-- The main theorem stating that the first player has a winning strategy --/
theorem first_player_wins (game : GameState) : 
  game.board.rows = 3 ∧ 
  game.board.cols = 1000 ∧ 
  game.firstPiece.width = 1 ∧ 
  game.firstPiece.height = 2 ∧ 
  game.secondPiece.width = 2 ∧ 
  game.secondPiece.height = 1 ∧ 
  game.currentPlayer = Player.First → 
  WinningStrategy Player.First game :=
sorry

end first_player_wins_l1051_105154


namespace pipe_filling_time_l1051_105136

/-- Given a tank and two pipes, prove that if one pipe takes T minutes to fill the tank,
    another pipe takes 12 minutes, and both pipes together take 4.8 minutes,
    then T = 8 minutes. -/
theorem pipe_filling_time (T : ℝ) : 
  (T > 0) →  -- T is positive (implied by the context)
  (1 / T + 1 / 12 = 1 / 4.8) →  -- Combined rate equation
  T = 8 := by
  sorry

end pipe_filling_time_l1051_105136


namespace negation_equivalence_l1051_105197

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, ∃ n : ℕ+, n ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, n < x^2) := by
  sorry

end negation_equivalence_l1051_105197


namespace multiple_properties_l1051_105193

-- Define the properties of c and d
def is_multiple_of_4 (n : ℤ) : Prop := ∃ k : ℤ, n = 4 * k
def is_multiple_of_8 (n : ℤ) : Prop := ∃ k : ℤ, n = 8 * k

-- Define the theorem
theorem multiple_properties (c d : ℤ) 
  (hc : is_multiple_of_4 c) (hd : is_multiple_of_8 d) : 
  (is_multiple_of_4 d) ∧ 
  (is_multiple_of_4 (c + d)) ∧ 
  (∃ k : ℤ, c + d = 2 * k) := by
  sorry


end multiple_properties_l1051_105193


namespace diana_age_is_22_l1051_105151

-- Define the ages as natural numbers
def anna_age : ℕ := 48

-- Define the relationships between ages
def brianna_age : ℕ := anna_age / 2
def caitlin_age : ℕ := brianna_age - 5
def diana_age : ℕ := caitlin_age + 3

-- Theorem to prove Diana's age
theorem diana_age_is_22 : diana_age = 22 := by
  sorry

end diana_age_is_22_l1051_105151


namespace min_value_of_z_l1051_105142

/-- Given a set of constraints on x and y, prove that the minimum value of z = 3x - 4y is -1 -/
theorem min_value_of_z (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x - 4 * y ∧ z ≥ -1 ∧ ∀ (w : ℝ), w = 3 * x - 4 * y → w ≥ z :=
by sorry

end min_value_of_z_l1051_105142


namespace fraction_to_decimal_l1051_105138

theorem fraction_to_decimal : (5 : ℚ) / 40 = 0.125 := by
  sorry

end fraction_to_decimal_l1051_105138


namespace battery_difference_is_thirteen_l1051_105158

/-- The number of batteries Tom used in his flashlights -/
def flashlight_batteries : ℕ := 2

/-- The number of batteries Tom used in his toys -/
def toy_batteries : ℕ := 15

/-- The number of batteries Tom used in his controllers -/
def controller_batteries : ℕ := 2

/-- The difference between the number of batteries in Tom's toys and flashlights -/
def battery_difference : ℕ := toy_batteries - flashlight_batteries

theorem battery_difference_is_thirteen : battery_difference = 13 := by
  sorry

end battery_difference_is_thirteen_l1051_105158


namespace sqrt_sum_zero_implies_power_l1051_105137

theorem sqrt_sum_zero_implies_power (a b : ℝ) : 
  Real.sqrt (a + 3) + Real.sqrt (2 - b) = 0 → a^b = 9 := by
sorry

end sqrt_sum_zero_implies_power_l1051_105137


namespace sum_of_15th_set_l1051_105102

/-- Represents the sum of elements in the nth set of a specific sequence of sets of consecutive integers -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  n * (first + last) / 2

/-- The theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end sum_of_15th_set_l1051_105102


namespace squares_after_six_steps_l1051_105189

/-- The number of squares after n steps, given an initial configuration of 5 squares 
    and each step adds 3 squares -/
def num_squares (n : ℕ) : ℕ := 5 + 3 * n

/-- Theorem stating that after 6 steps, there are 23 squares -/
theorem squares_after_six_steps : num_squares 6 = 23 := by
  sorry

end squares_after_six_steps_l1051_105189


namespace part1_min_max_part2_t_range_l1051_105179

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + 2*t*x + t - 1

-- Part 1
theorem part1_min_max :
  let t := 2
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-3) 1, f t x ≥ min) ∧
    (∃ x ∈ Set.Icc (-3) 1, f t x = min) ∧
    (∀ x ∈ Set.Icc (-3) 1, f t x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 1, f t x = max) ∧
    min = -3 ∧ max = 6 :=
sorry

-- Part 2
theorem part2_t_range :
  {t : ℝ | ∀ x ∈ Set.Icc 1 2, f t x > 0} = Set.Ioi 0 :=
sorry

end part1_min_max_part2_t_range_l1051_105179


namespace chessboard_paradox_l1051_105185

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

end chessboard_paradox_l1051_105185


namespace singers_and_dancers_selection_l1051_105194

/-- Represents the number of ways to select singers and dancers from a group -/
def select_singers_and_dancers (total : ℕ) (singers : ℕ) (dancers : ℕ) : ℕ :=
  let both := singers + dancers - total
  let only_singers := singers - both
  let only_dancers := dancers - both
  (only_singers * only_dancers) +
  (both * (only_singers + only_dancers)) +
  (both * (both - 1))

/-- Theorem stating that for 9 people with 7 singers and 5 dancers, 
    there are 32 ways to select 2 people and assign one to sing and one to dance -/
theorem singers_and_dancers_selection :
  select_singers_and_dancers 9 7 5 = 32 := by
  sorry

end singers_and_dancers_selection_l1051_105194


namespace no_integer_solution_l1051_105171

theorem no_integer_solution : ¬∃ (m n : ℤ), m^2 + 1954 = n^2 := by
  sorry

end no_integer_solution_l1051_105171


namespace circle_line_disjoint_radius_l1051_105116

-- Define the circle and line
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the distance function
def distance (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_line_disjoint_radius (O : ℝ × ℝ) (l : Set (ℝ × ℝ)) (r : ℝ) :
  (∃ (a b c : ℝ), l = Line a b c) →
  (distance O l)^2 - (distance O l) - 20 = 0 →
  (distance O l > 0) →
  (∀ p ∈ Circle O r, p ∉ l) →
  r = 4 := by sorry

end circle_line_disjoint_radius_l1051_105116


namespace combination_permutation_relation_combination_symmetry_pascal_identity_permutation_recursive_l1051_105112

-- Define C_n_m as the number of combinations of n items taken m at a time
def C (n m : ℕ) : ℕ := Nat.choose n m

-- Define A_n_m as the number of permutations of n items taken m at a time
def A (n m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

theorem combination_permutation_relation (n m : ℕ) (h : m ≤ n) :
  C n m = A n m / Nat.factorial m := by sorry

theorem combination_symmetry (n m : ℕ) (h : m ≤ n) :
  C n m = C n (n - m) := by sorry

theorem pascal_identity (n r : ℕ) (h : r ≤ n) :
  C (n + 1) r = C n r + C n (r - 1) := by sorry

theorem permutation_recursive (n m : ℕ) (h : m ≤ n) :
  A (n + 2) (m + 2) = (n + 2) * (n + 1) * A n m := by sorry

end combination_permutation_relation_combination_symmetry_pascal_identity_permutation_recursive_l1051_105112


namespace children_on_bus_after_stop_l1051_105153

/-- The number of children on a bus after a stop -/
theorem children_on_bus_after_stop 
  (initial_children : ℕ) 
  (children_who_got_on : ℕ) 
  (h1 : initial_children = 18) 
  (h2 : children_who_got_on = 7) :
  initial_children + children_who_got_on = 25 := by
  sorry

end children_on_bus_after_stop_l1051_105153


namespace unique_valid_subset_l1051_105162

def original_set : Finset ℕ := {1, 3, 4, 5, 7, 8, 9, 11, 12, 14}

def is_valid_subset (s : Finset ℕ) : Prop :=
  s.card = 2 ∧ 
  s ⊆ original_set ∧
  (Finset.sum (original_set \ s) id) / (original_set.card - 2 : ℚ) = 7

theorem unique_valid_subset : ∃! s : Finset ℕ, is_valid_subset s := by
  sorry

end unique_valid_subset_l1051_105162


namespace square_difference_theorem_l1051_105106

theorem square_difference_theorem (a b M : ℝ) : 
  (a + 2*b)^2 = (a - 2*b)^2 + M → M = 8*a*b := by
sorry

end square_difference_theorem_l1051_105106


namespace largest_inscribed_circle_radius_is_2_root_6_l1051_105130

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ :=
  sorry

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral :=
  ⟨13, 10, 8, 11⟩

theorem largest_inscribed_circle_radius_is_2_root_6 :
  largest_inscribed_circle_radius problem_quadrilateral = 2 * Real.sqrt 6 :=
by sorry

end largest_inscribed_circle_radius_is_2_root_6_l1051_105130


namespace quadratic_inequality_range_l1051_105144

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end quadratic_inequality_range_l1051_105144


namespace letter_150_is_Z_l1051_105121

def repeating_sequence : ℕ → Char
  | n => let idx := n % 3
         if idx = 0 then 'Z'
         else if idx = 1 then 'X'
         else 'Y'

theorem letter_150_is_Z : repeating_sequence 150 = 'Z' := by
  sorry

end letter_150_is_Z_l1051_105121


namespace dvd_average_price_l1051_105174

/-- Calculates the average price of DVDs bought from two different price groups -/
theorem dvd_average_price (n1 : ℕ) (p1 : ℚ) (n2 : ℕ) (p2 : ℚ) : 
  n1 = 10 → p1 = 2 → n2 = 5 → p2 = 5 → 
  (n1 * p1 + n2 * p2) / (n1 + n2 : ℚ) = 3 := by
sorry

end dvd_average_price_l1051_105174


namespace cubic_root_quadratic_coefficient_l1051_105145

theorem cubic_root_quadratic_coefficient 
  (A B C : ℝ) 
  (r s : ℝ) 
  (h1 : A ≠ 0)
  (h2 : A * r^2 + B * r + C = 0)
  (h3 : A * s^2 + B * s + C = 0) :
  ∃ (p q : ℝ), r^3^2 + p * r^3 + q = 0 ∧ s^3^2 + p * s^3 + q = 0 ∧ 
  p = (B^3 - 3*A*B*C + 2*A*C^2) / A^3 :=
by sorry

end cubic_root_quadratic_coefficient_l1051_105145


namespace red_cows_produce_more_milk_l1051_105175

/-- The daily milk production of a black cow -/
def black_cow_milk : ℝ := sorry

/-- The daily milk production of a red cow -/
def red_cow_milk : ℝ := sorry

/-- The total milk production of 4 black cows and 3 red cows in 5 days -/
def milk_production_1 : ℝ := 5 * (4 * black_cow_milk + 3 * red_cow_milk)

/-- The total milk production of 3 black cows and 5 red cows in 4 days -/
def milk_production_2 : ℝ := 4 * (3 * black_cow_milk + 5 * red_cow_milk)

theorem red_cows_produce_more_milk :
  milk_production_1 = milk_production_2 → red_cow_milk > black_cow_milk := by
  sorry

end red_cows_produce_more_milk_l1051_105175


namespace expression_value_l1051_105176

theorem expression_value (a : ℚ) (h : a = 1/3) : 
  (3 * a⁻¹ + (2 * a⁻¹) / 3) / a = 33 := by
  sorry

end expression_value_l1051_105176


namespace arithmetic_geometric_sequence_ratio_l1051_105157

theorem arithmetic_geometric_sequence_ratio (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := λ n => d * n.pred
  (a 1 * a 9 = (a 3)^2) →
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5/8 := by sorry

end arithmetic_geometric_sequence_ratio_l1051_105157


namespace inequality_solutions_l1051_105177

theorem inequality_solutions (x : ℝ) : 
  ((-x^2 + x + 6 ≤ 0) ↔ (x ≤ -2 ∨ x ≥ 3)) ∧
  ((x^2 - 2*x - 5 < 2*x) ↔ (-1 < x ∧ x < 5)) := by
  sorry

end inequality_solutions_l1051_105177


namespace double_iced_cubes_count_l1051_105178

/-- Represents a 3D coordinate in the cake --/
structure Coordinate where
  x : Nat
  y : Nat
  z : Nat

/-- The size of the cake --/
def cakeSize : Nat := 5

/-- Checks if a coordinate is on an edge with exactly two iced sides --/
def isDoubleIcedEdge (c : Coordinate) : Bool :=
  -- Top edge (front)
  (c.z = cakeSize - 1 && c.y = 0 && c.x > 0 && c.x < cakeSize - 1) ||
  -- Top edge (left)
  (c.z = cakeSize - 1 && c.x = 0 && c.y > 0 && c.y < cakeSize - 1) ||
  -- Front-left edge
  (c.x = 0 && c.y = 0 && c.z > 0 && c.z < cakeSize - 1)

/-- Counts the number of cubes with icing on exactly two sides --/
def countDoubleIcedCubes : Nat :=
  let coords := List.range cakeSize >>= fun x =>
                List.range cakeSize >>= fun y =>
                List.range cakeSize >>= fun z =>
                [{x := x, y := y, z := z}]
  (coords.filter isDoubleIcedEdge).length

/-- The main theorem to prove --/
theorem double_iced_cubes_count :
  countDoubleIcedCubes = 31 := by
  sorry


end double_iced_cubes_count_l1051_105178


namespace square_perimeter_l1051_105152

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 675 →
  side * side = area →
  perimeter = 4 * side →
  1.5 * perimeter = 90 * Real.sqrt 3 := by
  sorry

end square_perimeter_l1051_105152


namespace triangle_count_on_circle_l1051_105148

theorem triangle_count_on_circle (n : ℕ) (h : n = 10) : 
  Nat.choose n 3 = 120 := by
  sorry

end triangle_count_on_circle_l1051_105148


namespace milk_for_cookies_l1051_105119

/-- Given the ratio of cookies to milk, calculate the cups of milk needed for a given number of cookies -/
def milkNeeded (cookiesReference : ℕ) (quartsReference : ℕ) (cupsPerQuart : ℕ) (cookiesTarget : ℕ) : ℚ :=
  (quartsReference * cupsPerQuart : ℚ) * cookiesTarget / cookiesReference

theorem milk_for_cookies :
  milkNeeded 15 5 4 6 = 8 := by
  sorry

#eval milkNeeded 15 5 4 6

end milk_for_cookies_l1051_105119


namespace winning_scores_count_l1051_105109

/-- Represents a cross country meet with specific rules --/
structure CrossCountryMeet where
  runners_per_team : Nat
  total_runners : Nat
  min_score : Nat
  max_score : Nat

/-- Calculates the total score of all runners --/
def total_meet_score (meet : CrossCountryMeet) : Nat :=
  meet.total_runners * (meet.total_runners + 1) / 2

/-- Defines a valid cross country meet with given parameters --/
def valid_meet : CrossCountryMeet :=
  { runners_per_team := 6
  , total_runners := 12
  , min_score := 21
  , max_score := 38 }

/-- Theorem stating the number of possible winning scores --/
theorem winning_scores_count (meet : CrossCountryMeet) 
  (h1 : meet = valid_meet) 
  (h2 : meet.total_runners = 2 * meet.runners_per_team) 
  (h3 : total_meet_score meet = 78) : 
  (meet.max_score - meet.min_score + 1 : Nat) = 18 := by
  sorry

end winning_scores_count_l1051_105109


namespace unknown_number_is_ten_l1051_105173

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem unknown_number_is_ten :
  ∀ n : ℝ, euro 8 (euro n 5) = 640 → n = 10 := by
  sorry

end unknown_number_is_ten_l1051_105173


namespace parallel_dot_product_perpendicular_angle_l1051_105160

noncomputable section

/-- Two vectors in a plane with given magnitudes and angle between them -/
structure VectorPair where
  a : ℝ × ℝ
  b : ℝ × ℝ
  mag_a : Real.sqrt (a.1^2 + a.2^2) = 1
  mag_b : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2
  θ : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If vectors are parallel, their dot product is ±√2 -/
theorem parallel_dot_product (vp : VectorPair) 
    (h_parallel : ∃ (k : ℝ), vp.a = k • vp.b ∨ vp.b = k • vp.a) : 
    dot_product vp.a vp.b = Real.sqrt 2 ∨ dot_product vp.a vp.b = -Real.sqrt 2 := by
  sorry

/-- Theorem: If a - b is perpendicular to a, then θ = 45° -/
theorem perpendicular_angle (vp : VectorPair) 
    (h_perp : dot_product (vp.a.1 - vp.b.1, vp.a.2 - vp.b.2) vp.a = 0) : 
    vp.θ = Real.pi / 4 := by
  sorry

end parallel_dot_product_perpendicular_angle_l1051_105160


namespace larger_circle_tangent_to_line_and_axes_l1051_105147

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

/-- Checks if a circle is tangent to a line ax + by = c -/
def isTangentToLine (circle : Circle) (a b c : ℝ) : Prop :=
  let (x, y) := circle.center
  |a * x + b * y - c| / Real.sqrt (a^2 + b^2) = circle.radius

/-- Checks if a circle is tangent to both coordinate axes -/
def isTangentToAxes (circle : Circle) : Prop :=
  circle.center.1 = circle.radius ∧ circle.center.2 = circle.radius

/-- The theorem to be proved -/
theorem larger_circle_tangent_to_line_and_axes :
  ∃ (circle : Circle),
    circle.center = (5/2, 5/2) ∧
    circle.radius = 5/2 ∧
    isInFirstQuadrant circle.center ∧
    isTangentToLine circle 3 4 5 ∧
    isTangentToAxes circle ∧
    (∀ (other : Circle),
      isInFirstQuadrant other.center →
      isTangentToLine other 3 4 5 →
      isTangentToAxes other →
      other.radius ≤ circle.radius) :=
  sorry

end larger_circle_tangent_to_line_and_axes_l1051_105147


namespace equal_area_rectangles_l1051_105134

/-- Given two rectangles with equal areas, where one has length 5 and width 24,
    and the other has width 10, prove that the length of the second rectangle is 12. -/
theorem equal_area_rectangles (l₁ w₁ w₂ : ℝ) (h₁ : l₁ = 5) (h₂ : w₁ = 24) (h₃ : w₂ = 10) :
  let a₁ := l₁ * w₁
  let l₂ := a₁ / w₂
  l₂ = 12 := by sorry

end equal_area_rectangles_l1051_105134


namespace solution_set_of_f_neg_x_l1051_105123

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

-- State the theorem
theorem solution_set_of_f_neg_x (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-x) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end solution_set_of_f_neg_x_l1051_105123


namespace sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth_l1051_105150

theorem sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth : 
  10 * 56 * (3/2 + 5/4 + 9/8 + 17/16 + 33/32 + 65/64 - 7) = -1/64 := by
  sorry

end sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth_l1051_105150


namespace quadratic_root_range_l1051_105126

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ y^2 + (a^2 - 1)*y + a - 2 = 0 ∧ x > 1 ∧ y < 1) 
  → a > -2 ∧ a < 1 := by
sorry

end quadratic_root_range_l1051_105126


namespace intersection_size_lower_bound_l1051_105129

theorem intersection_size_lower_bound 
  (n k : ℕ) 
  (A : Fin (k + 1) → Finset (Fin (4 * n))) 
  (h1 : ∀ i, (A i).card = 2 * n) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card ≥ n - n / k := by
  sorry

end intersection_size_lower_bound_l1051_105129


namespace new_people_weight_sum_l1051_105114

/-- Given a group of 8 people with average weight W kg, prove that when two people weighing 68 kg each
    leave and are replaced by two new people, causing the average weight to increase by 5.5 kg,
    the sum of the weights of the two new people is 180 kg. -/
theorem new_people_weight_sum (W : ℝ) : 
  let original_total := 8 * W
  let remaining_total := original_total - 2 * 68
  let new_total := 8 * (W + 5.5)
  new_total - remaining_total = 180 := by
  sorry

/-- The sum of the weights of the two new people is no more than 180 kg. -/
axiom new_people_weight_bound (x y : ℝ) : x + y ≤ 180

/-- Each of the new people weighs more than the original average weight. -/
axiom new_people_weight_lower_bound (x y : ℝ) (W : ℝ) : x > W ∧ y > W

end new_people_weight_sum_l1051_105114


namespace incorrect_expression_l1051_105187

/-- A repeating decimal with non-repeating part X and repeating part Y -/
structure RepeatingDecimal where
  X : ℕ  -- non-repeating part
  Y : ℕ  -- repeating part
  t : ℕ  -- number of digits in X
  u : ℕ  -- number of digits in Y

/-- The value of a repeating decimal -/
def value (E : RepeatingDecimal) : ℚ :=
  sorry

/-- The statement that the expression is incorrect -/
theorem incorrect_expression (E : RepeatingDecimal) :
  ¬(10^E.t * (10^E.u - 1) * value E = E.Y * (E.X - 10)) :=
sorry

end incorrect_expression_l1051_105187


namespace quadratic_complex_roots_condition_l1051_105107

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∃ x : ℂ, x^2 + a*x + 1 = 0 ∧ x.im ≠ 0) →
  a < 2 ∧
  ¬(a < 2 → ∃ x : ℂ, x^2 + a*x + 1 = 0 ∧ x.im ≠ 0) :=
by sorry

end quadratic_complex_roots_condition_l1051_105107


namespace parabola_shift_l1051_105124

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift amount
def shift : ℝ := 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - shift) :=
by
  sorry

end parabola_shift_l1051_105124


namespace staircase_steps_l1051_105188

/-- The number of toothpicks in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 3

/-- The proposition that 8 steps result in 490 toothpicks -/
theorem staircase_steps : toothpicks 8 = 490 := by
  sorry

#check staircase_steps

end staircase_steps_l1051_105188


namespace original_number_of_classes_l1051_105192

theorem original_number_of_classes : 
  ∃! x : ℕ+, 
    (280 % x.val = 0) ∧ 
    (585 % (x.val + 6) = 0) ∧ 
    x.val = 7 := by
  sorry

end original_number_of_classes_l1051_105192


namespace cube_root_of_four_solution_l1051_105108

theorem cube_root_of_four_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) :=
by
  sorry

end cube_root_of_four_solution_l1051_105108


namespace sin_2theta_value_l1051_105131

theorem sin_2theta_value (θ : Real) 
  (h : Real.exp (2 * Real.log 2 * ((-2 : Real) + 2 * Real.sin θ)) + 3 = 
       Real.exp (Real.log 2 * ((1 / 2 : Real) + Real.sin θ))) : 
  Real.sin (2 * θ) = 3 * Real.sqrt 7 / 8 := by
  sorry

end sin_2theta_value_l1051_105131


namespace decimal_to_fraction_l1051_105139

theorem decimal_to_fraction (x : ℚ) (h : x = 3.36) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ n = 84 ∧ d = 25 := by
  sorry

end decimal_to_fraction_l1051_105139


namespace vector_properties_l1051_105159

/-- Given vectors a and b, prove properties about vector c and scalar t. -/
theorem vector_properties (a b : ℝ × ℝ) (h_a : a = (1, 0)) (h_b : b = (-1, 2)) :
  -- Part 1
  (∃ c : ℝ × ℝ, ‖c‖ = 1 ∧ ∃ k : ℝ, c = k • (a - b)) →
  (∃ c : ℝ × ℝ, c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) ∧
  -- Part 2
  (∃ t : ℝ, (2 * t • a - b) • (3 • a + t • b) = 0) →
  (∃ t : ℝ, t = -1 ∨ t = 3 / 2) :=
by sorry

end vector_properties_l1051_105159


namespace determinant_specific_matrix_l1051_105120

theorem determinant_specific_matrix :
  let matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, Real.sin (π / 6); 5, Real.cos (π / 3)]
  Matrix.det matrix = -1 := by
  sorry

end determinant_specific_matrix_l1051_105120


namespace three_digit_square_last_three_l1051_105155

theorem three_digit_square_last_three (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) → (n = n^2 % 1000 ↔ n = 376 ∨ n = 625) := by
  sorry

end three_digit_square_last_three_l1051_105155


namespace square_root_equality_l1051_105167

theorem square_root_equality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x = a + 2 ∧ Real.sqrt x = 2*a - 5) → a = 1 := by
  sorry

end square_root_equality_l1051_105167


namespace jeff_fills_130_boxes_l1051_105168

/-- Calculates the number of boxes Jeff can fill with remaining donuts --/
def donut_boxes : ℕ :=
  let total_donuts := 50 * 30
  let jeff_eats := 3 * 30
  let friends_eat := 10 + 12 + 8
  let given_away := 25 + 50
  let unavailable := jeff_eats + friends_eat + given_away
  let remaining := total_donuts - unavailable
  remaining / 10

/-- Theorem stating that Jeff can fill 130 boxes with remaining donuts --/
theorem jeff_fills_130_boxes : donut_boxes = 130 := by
  sorry

end jeff_fills_130_boxes_l1051_105168


namespace difference_multiplier_proof_l1051_105156

theorem difference_multiplier_proof : ∃ x : ℕ, 
  let sum := 555 + 445
  let difference := 555 - 445
  220040 = sum * (x * difference) + 40 ∧ x = 2 := by
  sorry

end difference_multiplier_proof_l1051_105156


namespace hannah_reading_finish_day_l1051_105143

def days_to_read (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem hannah_reading_finish_day (start_day : ℕ) (num_books : ℕ) :
  start_day = 5 →  -- Friday is represented as 5 (0 = Sunday, 1 = Monday, etc.)
  num_books = 20 →
  day_of_week start_day (days_to_read num_books) = start_day :=
by sorry

end hannah_reading_finish_day_l1051_105143


namespace total_water_in_boxes_l1051_105183

theorem total_water_in_boxes (num_boxes : ℕ) (bottles_per_box : ℕ) (bottle_capacity : ℚ) (fill_ratio : ℚ) : 
  num_boxes = 10 →
  bottles_per_box = 50 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  (num_boxes * bottles_per_box * bottle_capacity * fill_ratio : ℚ) = 4500 := by
sorry

end total_water_in_boxes_l1051_105183


namespace ratio_w_to_y_l1051_105149

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y 
  (h_wx : (w : ℚ) / x = 5 / 2)
  (h_yz : (y : ℚ) / z = 4 / 1)
  (h_zx : (z : ℚ) / x = 2 / 5) :
  w / y = 25 / 16 :=
by sorry

end ratio_w_to_y_l1051_105149


namespace hugo_tom_box_folding_l1051_105146

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

end hugo_tom_box_folding_l1051_105146


namespace largest_number_l1051_105190

/-- Represents a real number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a real number -/
def toReal (d : RepeatingDecimal) : ℝ :=
  sorry

/-- Define the five numbers in the problem -/
def a : ℝ := 8.23456
def b : RepeatingDecimal := ⟨8, [2, 3, 4], [5]⟩
def c : RepeatingDecimal := ⟨8, [2, 3], [4, 5]⟩
def d : RepeatingDecimal := ⟨8, [2], [3, 4, 5]⟩
def e : RepeatingDecimal := ⟨8, [], [2, 3, 4, 5]⟩

theorem largest_number :
  toReal b > a ∧
  toReal b > toReal c ∧
  toReal b > toReal d ∧
  toReal b > toReal e :=
sorry

end largest_number_l1051_105190


namespace comic_book_arrangement_count_comic_book_arrangement_count_is_correct_l1051_105118

/-- The number of ways to arrange comic books from different publishers in a stack -/
theorem comic_book_arrangement_count : Nat :=
  let marvel_books : Nat := 8
  let dc_books : Nat := 6
  let image_books : Nat := 5
  let publisher_groups : Nat := 3

  let marvel_arrangements := Nat.factorial marvel_books
  let dc_arrangements := Nat.factorial dc_books
  let image_arrangements := Nat.factorial image_books
  let group_arrangements := Nat.factorial publisher_groups

  marvel_arrangements * dc_arrangements * image_arrangements * group_arrangements

/-- Proof that the number of arrangements is 20,901,888,000 -/
theorem comic_book_arrangement_count_is_correct : 
  comic_book_arrangement_count = 20901888000 := by
  sorry

end comic_book_arrangement_count_comic_book_arrangement_count_is_correct_l1051_105118


namespace regular_milk_students_l1051_105103

/-- Proof that the number of students who selected regular milk is 3 -/
theorem regular_milk_students (chocolate_milk : ℕ) (strawberry_milk : ℕ) (total_milk : ℕ) :
  chocolate_milk = 2 →
  strawberry_milk = 15 →
  total_milk = 20 →
  total_milk - (chocolate_milk + strawberry_milk) = 3 := by
  sorry

end regular_milk_students_l1051_105103


namespace smallest_n_for_integer_sum_l1051_105166

theorem smallest_n_for_integer_sum : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬∃ (j : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = j) ∧
  n = 24 :=
by sorry

end smallest_n_for_integer_sum_l1051_105166


namespace solve_carlas_drink_problem_l1051_105161

/-- The amount of water Carla drank, given the conditions of the problem -/
def carlas_water_amount (s w : ℝ) : Prop :=
  s = 3 * w - 6 ∧ s + w = 54 → w = 15

theorem solve_carlas_drink_problem :
  ∀ s w : ℝ, carlas_water_amount s w :=
by
  sorry

end solve_carlas_drink_problem_l1051_105161


namespace lucky_lacy_correct_percentage_l1051_105128

/-- Given a total of 5x + 10 problems and x + 2 missed problems, 
    the percentage of correctly answered problems is 80%. -/
theorem lucky_lacy_correct_percentage (x : ℕ) : 
  let total := 5 * x + 10
  let missed := x + 2
  let correct := total - missed
  (correct : ℚ) / total * 100 = 80 :=
by sorry

end lucky_lacy_correct_percentage_l1051_105128


namespace bowling_team_average_weight_l1051_105198

theorem bowling_team_average_weight 
  (original_team_size : ℕ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (new_average_weight : ℝ) 
  (h1 : original_team_size = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 106) :
  ∃ (original_average_weight : ℝ),
    (original_team_size * original_average_weight + new_player1_weight + new_player2_weight) / 
    (original_team_size + 2) = new_average_weight ∧
    original_average_weight = 112 :=
by sorry

end bowling_team_average_weight_l1051_105198


namespace olympic_year_zodiac_l1051_105163

/-- The zodiac signs in order -/
inductive ZodiacSign
| Rat | Ox | Tiger | Rabbit | Dragon | Snake | Horse | Goat | Monkey | Rooster | Dog | Pig

/-- Function to get the zodiac sign for a given year -/
def getZodiacSign (year : Int) : ZodiacSign :=
  match (year - 1) % 12 with
  | 0 => ZodiacSign.Rooster
  | 1 => ZodiacSign.Dog
  | 2 => ZodiacSign.Pig
  | 3 => ZodiacSign.Rat
  | 4 => ZodiacSign.Ox
  | 5 => ZodiacSign.Tiger
  | 6 => ZodiacSign.Rabbit
  | 7 => ZodiacSign.Dragon
  | 8 => ZodiacSign.Snake
  | 9 => ZodiacSign.Horse
  | 10 => ZodiacSign.Goat
  | _ => ZodiacSign.Monkey

theorem olympic_year_zodiac :
  getZodiacSign 2008 = ZodiacSign.Rabbit :=
by sorry

end olympic_year_zodiac_l1051_105163


namespace sufficient_not_necessary_l1051_105196

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x^2 - 1 = 0 → x^3 - x = 0) ∧ 
  (∃ x : ℝ, x^3 - x = 0 ∧ x^2 - 1 ≠ 0) := by
sorry

end sufficient_not_necessary_l1051_105196


namespace product_of_integers_l1051_105111

theorem product_of_integers (A B C D : ℕ+) : 
  A + B + C + D = 100 →
  2^(A:ℕ) = B - 4 →
  C + 6 = D →
  B + C = D + 10 →
  A * B * C * D = 33280 := by
sorry

end product_of_integers_l1051_105111


namespace increasing_function_a_bound_l1051_105195

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 1

-- State the theorem
theorem increasing_function_a_bound (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 3 → y ∈ Set.Icc 1 3 → x < y → f a x < f a y) →
  a ≤ 1/2 := by
  sorry

end increasing_function_a_bound_l1051_105195


namespace g_inv_f_10_l1051_105140

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Assume f and g are bijective
variable (hf : Function.Bijective f)
variable (hg : Function.Bijective g)

-- Define the relationship between f and g
axiom fg_relation : ∀ x, f_inv (g x) = 3 * x - 1

-- Define the inverse functions
axiom f_inverse : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f
axiom g_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

-- State the theorem
theorem g_inv_f_10 : g_inv (f 10) = 11 / 3 := by sorry

end g_inv_f_10_l1051_105140


namespace divisors_odd_iff_perfect_square_l1051_105182

theorem divisors_odd_iff_perfect_square (n : ℕ) : 
  Odd (Finset.card (Nat.divisors n)) ↔ ∃ m : ℕ, n = m ^ 2 := by
sorry

end divisors_odd_iff_perfect_square_l1051_105182


namespace total_price_after_increase_l1051_105125

/-- Calculates the total price for a buyer purchasing jewelry and paintings 
    after a price increase. -/
theorem total_price_after_increase 
  (initial_jewelry_price : ℝ) 
  (initial_painting_price : ℝ)
  (jewelry_price_increase : ℝ)
  (painting_price_increase_percent : ℝ)
  (jewelry_quantity : ℕ)
  (painting_quantity : ℕ)
  (h1 : initial_jewelry_price = 30)
  (h2 : initial_painting_price = 100)
  (h3 : jewelry_price_increase = 10)
  (h4 : painting_price_increase_percent = 20)
  (h5 : jewelry_quantity = 2)
  (h6 : painting_quantity = 5) :
  let new_jewelry_price := initial_jewelry_price + jewelry_price_increase
  let new_painting_price := initial_painting_price * (1 + painting_price_increase_percent / 100)
  let total_price := new_jewelry_price * jewelry_quantity + new_painting_price * painting_quantity
  total_price = 680 := by
sorry


end total_price_after_increase_l1051_105125


namespace sum_a_d_equals_one_l1051_105172

theorem sum_a_d_equals_one 
  (a b c d : ℤ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := by
sorry

end sum_a_d_equals_one_l1051_105172


namespace expression_is_perfect_square_l1051_105186

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the expression
def expression : ℕ := 3^4 * 4^6 * 7^4

-- Define the prime factorization of 4
axiom four_factorization : 4 = 2^2

-- Theorem to prove
theorem expression_is_perfect_square : is_perfect_square expression := by
  sorry

end expression_is_perfect_square_l1051_105186


namespace equation_solution_l1051_105133

theorem equation_solution : ∃! x : ℚ, x + 2/3 = 7/15 + 1/5 - x/2 ∧ x = 0 := by
  sorry

end equation_solution_l1051_105133


namespace f_has_six_zeros_l1051_105117

noncomputable def f (x : ℝ) : ℝ :=
  (1 + x - x^2/2 + x^3/3 - x^4/4 - x^2018/2018 + x^2019/2019) * Real.cos (2*x)

theorem f_has_six_zeros :
  ∃ (S : Finset ℝ), S.card = 6 ∧ 
  (∀ x ∈ S, x ∈ Set.Icc (-3) 4 ∧ f x = 0) ∧
  (∀ x ∈ Set.Icc (-3) 4, f x = 0 → x ∈ S) := by
  sorry

end f_has_six_zeros_l1051_105117


namespace average_problem_l1051_105135

theorem average_problem (x : ℝ) : (15 + 25 + x + 30) / 4 = 23 → x = 22 := by
  sorry

end average_problem_l1051_105135


namespace three_digit_cube_sum_l1051_105105

theorem three_digit_cube_sum : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n = (n / 100)^3 + ((n / 10) % 10)^3 + (n % 10)^3) ∧
  n = 153 := by
  sorry

end three_digit_cube_sum_l1051_105105


namespace sqrt_10_plus_2_range_l1051_105101

theorem sqrt_10_plus_2_range : 5 < Real.sqrt 10 + 2 ∧ Real.sqrt 10 + 2 < 6 := by
  sorry

end sqrt_10_plus_2_range_l1051_105101


namespace least_positive_integer_for_multiple_of_four_l1051_105169

theorem least_positive_integer_for_multiple_of_four :
  ∃ (n : ℕ), n > 0 ∧ (575 + n) % 4 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (575 + m) % 4 = 0 → n ≤ m :=
by sorry

end least_positive_integer_for_multiple_of_four_l1051_105169


namespace intersection_A_B_l1051_105132

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end intersection_A_B_l1051_105132


namespace joey_age_l1051_105199

def ages : List ℕ := [4, 6, 8, 10, 12]

def is_cinema_pair (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def is_soccer_pair (a b : ℕ) : Prop := 
  a < 11 ∧ b < 11 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ 
  ¬(∃ c d, is_cinema_pair c d ∧ (a = c ∨ a = d ∨ b = c ∨ b = d))

theorem joey_age : 
  (∃ a b c d, is_cinema_pair a b ∧ is_soccer_pair c d) →
  (∃! x, x ∈ ages ∧ x ≠ 6 ∧ ¬(∃ y z, (is_cinema_pair x y ∨ is_cinema_pair y x) ∧ 
                                     (is_soccer_pair x z ∨ is_soccer_pair z x))) →
  (∃ x, x ∈ ages ∧ x ≠ 6 ∧ ¬(∃ y z, (is_cinema_pair x y ∨ is_cinema_pair y x) ∧ 
                                    (is_soccer_pair x z ∨ is_soccer_pair z x)) ∧ x = 8) :=
by sorry

end joey_age_l1051_105199


namespace infinite_triplets_exist_l1051_105104

theorem infinite_triplets_exist : 
  ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 0 ∧ a^4 + b^4 + c^4 = 50 := by
  sorry

end infinite_triplets_exist_l1051_105104


namespace exists_solution_for_prime_l1051_105141

theorem exists_solution_for_prime (p : ℕ) (hp : Prime p) :
  ∃ (x y z w : ℤ), x^2 + y^2 + z^2 - w * ↑p = 0 ∧ 0 < w ∧ w < ↑p :=
by sorry

end exists_solution_for_prime_l1051_105141


namespace triangle_inequality_l1051_105110

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
  let f (x : ℝ) := 1 - Real.sqrt (Real.sqrt 3 * Real.tan (x / 2)) + Real.sqrt 3 * Real.tan (x / 2)
  (f A * f B) + (f B * f C) + (f C * f A) ≥ 3 := by
  sorry

end triangle_inequality_l1051_105110


namespace largest_common_divisor_l1051_105184

theorem largest_common_divisor : ∃ (n : ℕ), n = 60 ∧ 
  n ∣ 660 ∧ n < 100 ∧ n ∣ 120 ∧ 
  ∀ (m : ℕ), m ∣ 660 ∧ m < 100 ∧ m ∣ 120 → m ≤ n :=
by sorry

end largest_common_divisor_l1051_105184
