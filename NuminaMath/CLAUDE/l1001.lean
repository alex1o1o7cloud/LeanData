import Mathlib

namespace weight_of_smaller_cube_l1001_100188

/-- Given two cubes of the same material, where the second cube has sides twice
    as long as the first and weighs 40 pounds, prove that the weight of the first
    cube is 5 pounds. -/
theorem weight_of_smaller_cube (s : ℝ) (w : ℝ → ℝ → ℝ) :
  (∀ x y, w x y = (y / x^3) * w 1 1) →  -- weight is proportional to volume
  w (2*s) (8*s^3) = 40 →                -- weight of larger cube
  w s (s^3) = 5 := by
sorry


end weight_of_smaller_cube_l1001_100188


namespace sum_negative_forty_to_sixty_l1001_100143

def sum_range (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_negative_forty_to_sixty :
  sum_range (-40) 60 = 1010 := by
  sorry

end sum_negative_forty_to_sixty_l1001_100143


namespace solve_for_s_l1001_100122

theorem solve_for_s (s t : ℝ) 
  (eq1 : 8 * s + 4 * t = 160) 
  (eq2 : t = 2 * s - 3) : 
  s = 10.75 := by
  sorry

end solve_for_s_l1001_100122


namespace race_problem_l1001_100107

/-- The race problem -/
theorem race_problem (john_speed steve_speed : ℝ) (duration : ℝ) (final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : duration = 28)
  (h4 : final_distance = 2) :
  john_speed * duration - steve_speed * duration - final_distance = 12 := by
  sorry

end race_problem_l1001_100107


namespace phone_call_duration_l1001_100116

/-- Calculates the duration of a phone call given initial credit, cost per minute, and remaining credit -/
def call_duration (initial_credit : ℚ) (cost_per_minute : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_credit - remaining_credit) / cost_per_minute

/-- Proves that given the specified conditions, the call duration is 22 minutes -/
theorem phone_call_duration :
  let initial_credit : ℚ := 30
  let cost_per_minute : ℚ := 16/100
  let remaining_credit : ℚ := 2648/100
  call_duration initial_credit cost_per_minute remaining_credit = 22 := by
sorry


end phone_call_duration_l1001_100116


namespace problem_solution_l1001_100125

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log (abs x)
def g (a : ℝ) (x : ℝ) : ℝ := 1 / (deriv f x) + a * (deriv f x)

-- State the theorem
theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g a x = x + a / x) ∧
  (a > 0 ∧ (∀ x : ℝ, x > 0 → g a x ≥ 2) ∧ ∃ x : ℝ, x > 0 ∧ g a x = 2) →
  (a = 1 ∧
   (∫ x in (3/2)..(2), (2/3 * x + 7/6) - (x + 1/x)) = 7/24 + Real.log 3 - 2 * Real.log 2) :=
by sorry

end

end problem_solution_l1001_100125


namespace first_player_wins_l1001_100152

/-- Represents the state of the game -/
structure GameState where
  player1Pos : Nat
  player2Pos : Nat

/-- Represents a valid move in the game -/
inductive Move where
  | one   : Move
  | two   : Move
  | three : Move
  | four  : Move

/-- The game board size -/
def boardSize : Nat := 101

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (player : Nat) (move : Move) : Bool :=
  match player, move with
  | 1, Move.one   => state.player1Pos + 1 < state.player2Pos
  | 1, Move.two   => state.player1Pos + 2 < state.player2Pos
  | 1, Move.three => state.player1Pos + 3 < state.player2Pos
  | 1, Move.four  => state.player1Pos + 4 < state.player2Pos
  | 2, Move.one   => state.player2Pos - 1 > state.player1Pos
  | 2, Move.two   => state.player2Pos - 2 > state.player1Pos
  | 2, Move.three => state.player2Pos - 3 > state.player1Pos
  | 2, Move.four  => state.player2Pos - 4 > state.player1Pos
  | _, _          => false

/-- Applies a move to the game state -/
def applyMove (state : GameState) (player : Nat) (move : Move) : GameState :=
  match player, move with
  | 1, Move.one   => { state with player1Pos := state.player1Pos + 1 }
  | 1, Move.two   => { state with player1Pos := state.player1Pos + 2 }
  | 1, Move.three => { state with player1Pos := state.player1Pos + 3 }
  | 1, Move.four  => { state with player1Pos := state.player1Pos + 4 }
  | 2, Move.one   => { state with player2Pos := state.player2Pos - 1 }
  | 2, Move.two   => { state with player2Pos := state.player2Pos - 2 }
  | 2, Move.three => { state with player2Pos := state.player2Pos - 3 }
  | 2, Move.four  => { state with player2Pos := state.player2Pos - 4 }
  | _, _          => state

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.player1Pos = boardSize || state.player2Pos = 1

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      let game_result := (sorry : GameState)  -- Simulate game play
      isGameOver game_result ∧ game_result.player1Pos = boardSize :=
sorry


end first_player_wins_l1001_100152


namespace average_weight_problem_l1001_100101

/-- The average weight problem -/
theorem average_weight_problem 
  (weight_A weight_B weight_C weight_D : ℝ)
  (h1 : (weight_A + weight_B + weight_C) / 3 = 60)
  (h2 : weight_A = 87)
  (h3 : (weight_B + weight_C + weight_D + (weight_D + 3)) / 4 = 64) :
  (weight_A + weight_B + weight_C + weight_D) / 4 = 65 := by
  sorry

end average_weight_problem_l1001_100101


namespace number_division_problem_l1001_100145

theorem number_division_problem : ∃! x : ℕ, 
  ∃ q : ℕ, x = 7 * q ∧ q + x + 7 = 175 ∧ x = 147 := by
  sorry

end number_division_problem_l1001_100145


namespace point_coordinates_l1001_100160

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate to check if a point is in the fourth quadrant -/
def inFourthQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

theorem point_coordinates (p : Point) 
  (h1 : inFourthQuadrant p) 
  (h2 : distanceToXAxis p = 2) 
  (h3 : distanceToYAxis p = 5) : 
  p = Point.mk 5 (-2) := by
  sorry

end point_coordinates_l1001_100160


namespace f_less_than_g_implies_a_bound_l1001_100132

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

noncomputable def g (x : ℝ) : ℝ := Real.log x - x / 4 + 3 / (4 * x)

theorem f_less_than_g_implies_a_bound (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ > 1, f a x₁ < g x₂) →
  a > (1 / 3) * Real.exp (1 / 2) :=
by sorry

end f_less_than_g_implies_a_bound_l1001_100132


namespace intersection_of_A_and_B_l1001_100102

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 8}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x - 14 ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 7 ≤ x ∧ x < 8} := by
  sorry

end intersection_of_A_and_B_l1001_100102


namespace jason_seashells_l1001_100126

/-- The number of seashells Jason has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Jason has 36 seashells after starting with 49 and giving away 13 -/
theorem jason_seashells : remaining_seashells 49 13 = 36 := by
  sorry

end jason_seashells_l1001_100126


namespace a_equals_two_l1001_100135

/-- The function f(x) = x^2 - 14x + 52 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 52

/-- The function g(x) = ax + b, where a and b are positive real numbers -/
def g (a b : ℝ) (x : ℝ) : ℝ := a*x + b

/-- Theorem stating that a = 2 given the conditions -/
theorem a_equals_two (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : f (g a b (-5)) = 3) (h2 : f (g a b 0) = 103) : a = 2 := by
  sorry

end a_equals_two_l1001_100135


namespace coloring_books_problem_l1001_100147

theorem coloring_books_problem (total_colored : ℕ) (total_left : ℕ) (num_books : ℕ) :
  total_colored = 20 →
  total_left = 68 →
  num_books = 2 →
  (total_colored + total_left) % num_books = 0 →
  (total_colored + total_left) / num_books = 44 :=
by
  sorry

end coloring_books_problem_l1001_100147


namespace factorial_division_l1001_100120

theorem factorial_division :
  (10 : ℕ).factorial = 3628800 →
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 := by
  sorry

end factorial_division_l1001_100120


namespace inequality_proof_l1001_100161

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l1001_100161


namespace floor_division_equality_l1001_100164

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem: For any positive real number a and any integer n,
    the floor of (floor of a) divided by n is equal to the floor of a divided by n -/
theorem floor_division_equality (a : ℝ) (n : ℤ) (h1 : 0 < a) (h2 : n ≠ 0) :
  floor ((floor a : ℝ) / n) = floor (a / n) := by
  sorry

end floor_division_equality_l1001_100164


namespace rectangle_area_l1001_100108

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
sorry

end rectangle_area_l1001_100108


namespace rational_fraction_implication_l1001_100130

theorem rational_fraction_implication (x : ℝ) :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) →
  (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by sorry

end rational_fraction_implication_l1001_100130


namespace max_subsequent_voters_l1001_100175

/-- Represents a movie rating system where:
  * Ratings are integers from 0 to 10
  * At moment T, the rating was an integer
  * After moment T, each subsequent voter decreased the rating by one unit
-/
structure MovieRating where
  initial_rating : ℕ
  initial_voters : ℕ
  subsequent_votes : List ℕ

/-- The rating at any given moment is the sum of all scores divided by their quantity -/
def current_rating (mr : MovieRating) : ℚ :=
  (mr.initial_rating * mr.initial_voters + mr.subsequent_votes.sum) / 
  (mr.initial_voters + mr.subsequent_votes.length)

/-- The condition that the rating decreases by 1 unit after each vote -/
def decreasing_by_one (mr : MovieRating) : Prop :=
  ∀ i, i < mr.subsequent_votes.length →
    current_rating { mr with 
      subsequent_votes := mr.subsequent_votes.take i
    } - current_rating { mr with 
      subsequent_votes := mr.subsequent_votes.take (i + 1)
    } = 1

/-- The main theorem: The maximum number of viewers who could have voted after moment T is 5 -/
theorem max_subsequent_voters (mr : MovieRating) 
    (h1 : mr.initial_rating ∈ Set.range (fun i => i : ℕ → ℕ) ∩ Set.Icc 0 10)
    (h2 : ∀ v ∈ mr.subsequent_votes, v ∈ Set.range (fun i => i : ℕ → ℕ) ∩ Set.Icc 0 10)
    (h3 : decreasing_by_one mr) :
    mr.subsequent_votes.length ≤ 5 :=
  sorry

end max_subsequent_voters_l1001_100175


namespace spring_math_camp_inconsistency_l1001_100167

theorem spring_math_camp_inconsistency : 
  ¬ ∃ (b g : ℕ), 11 * b + 7 * g = 4046 := by
  sorry

end spring_math_camp_inconsistency_l1001_100167


namespace infinite_geometric_series_first_term_l1001_100149

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = (1 : ℝ) / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r)) :
  a = 60 := by
sorry

end infinite_geometric_series_first_term_l1001_100149


namespace sqrt_198_between_14_and_15_l1001_100115

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end sqrt_198_between_14_and_15_l1001_100115


namespace sum_remainder_l1001_100146

theorem sum_remainder (a b c : ℕ) (ha : a % 36 = 15) (hb : b % 36 = 22) (hc : c % 36 = 9) :
  (a + b + c) % 36 = 10 := by
  sorry

end sum_remainder_l1001_100146


namespace gary_shortage_l1001_100121

def gary_initial_amount : ℝ := 73
def snake_cost : ℝ := 55
def snake_food_cost : ℝ := 12
def habitat_original_cost : ℝ := 35
def habitat_discount_rate : ℝ := 0.15

def total_spent : ℝ := snake_cost + snake_food_cost + 
  (habitat_original_cost * (1 - habitat_discount_rate))

theorem gary_shortage : 
  total_spent - gary_initial_amount = 23.75 := by sorry

end gary_shortage_l1001_100121


namespace pen_notebook_ratio_l1001_100193

/-- Given 50 pens and 40 notebooks, prove that the ratio of pens to notebooks is 5:4 -/
theorem pen_notebook_ratio :
  let num_pens : ℕ := 50
  let num_notebooks : ℕ := 40
  (num_pens : ℚ) / (num_notebooks : ℚ) = 5 / 4 := by
  sorry

end pen_notebook_ratio_l1001_100193


namespace solution_count_l1001_100195

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = x + y + c

/-- The theorem stating the number of solutions based on the value of c -/
theorem solution_count (c : ℝ) :
  (c = 0 ∧ ∃! f : ℝ → ℝ, SatisfiesEquation f c ∧ f = id) ∨
  (c ≠ 0 ∧ ¬∃ f : ℝ → ℝ, SatisfiesEquation f c) :=
sorry

end solution_count_l1001_100195


namespace exists_solution_l1001_100133

theorem exists_solution : ∃ x : ℝ, x + 2.75 + 0.158 = 2.911 := by sorry

end exists_solution_l1001_100133


namespace kyler_wins_one_l1001_100158

structure ChessTournament where
  peter_wins : ℕ
  peter_losses : ℕ
  emma_wins : ℕ
  emma_losses : ℕ
  kyler_losses : ℕ

def kyler_wins (t : ChessTournament) : ℕ :=
  (t.peter_wins + t.emma_wins + t.kyler_losses) - (t.peter_losses + t.emma_losses)

theorem kyler_wins_one (t : ChessTournament) 
  (h1 : t.peter_wins = 4) 
  (h2 : t.peter_losses = 2) 
  (h3 : t.emma_wins = 3) 
  (h4 : t.emma_losses = 3) 
  (h5 : t.kyler_losses = 3) : 
  kyler_wins t = 1 := by
  sorry

end kyler_wins_one_l1001_100158


namespace prob_more_heads_12_coins_l1001_100166

/-- The number of coins flipped -/
def n : ℕ := 12

/-- The probability of getting more heads than tails when flipping n coins -/
def prob_more_heads (n : ℕ) : ℚ :=
  1 / 2 - (n.choose (n / 2)) / (2 ^ n)

theorem prob_more_heads_12_coins : 
  prob_more_heads n = 793 / 2048 := by
  sorry

end prob_more_heads_12_coins_l1001_100166


namespace completely_overlapping_implies_congruent_l1001_100106

/-- Two triangles are completely overlapping if all their corresponding vertices coincide. -/
def CompletelyOverlapping (T1 T2 : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T1 ↔ p ∈ T2

/-- Two triangles are congruent if they have the same size and shape. -/
def Congruent (T1 T2 : Set (ℝ × ℝ)) : Prop :=
  ∃ f : ℝ × ℝ → ℝ × ℝ, Isometry f ∧ f '' T1 = T2

/-- If two triangles completely overlap, then they are congruent. -/
theorem completely_overlapping_implies_congruent
  (T1 T2 : Set (ℝ × ℝ)) (h : CompletelyOverlapping T1 T2) :
  Congruent T1 T2 := by
  sorry

end completely_overlapping_implies_congruent_l1001_100106


namespace abigail_initial_fences_l1001_100198

/-- The number of fences Abigail can build in 8 hours -/
def fences_in_8_hours : ℕ := 8 * 60 / 30

/-- The total number of fences after 8 hours of building -/
def total_fences : ℕ := 26

/-- The number of fences Abigail built initially -/
def initial_fences : ℕ := total_fences - fences_in_8_hours

theorem abigail_initial_fences : initial_fences = 10 := by
  sorry

end abigail_initial_fences_l1001_100198


namespace triangle_side_sum_l1001_100155

theorem triangle_side_sum (a b c : ℝ) (h_angles : a = 60 ∧ b = 30 ∧ c = 90) 
  (h_side : 8 * Real.sqrt 3 = b) : 
  a + b + c = 24 * Real.sqrt 3 + 24 := by
sorry

end triangle_side_sum_l1001_100155


namespace fib_div_three_iff_index_div_four_l1001_100144

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_div_three_iff_index_div_four (n : ℕ) : 
  3 ∣ fib n ↔ 4 ∣ n := by sorry

end fib_div_three_iff_index_div_four_l1001_100144


namespace largest_coefficients_in_expansion_l1001_100179

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def expansion_term (n r : ℕ) : ℕ := 2^r * binomial_coefficient n r

theorem largest_coefficients_in_expansion (n : ℕ) (h : n = 11) :
  (∀ k, k ≠ 5 ∧ k ≠ 6 → expansion_term n 5 ≥ expansion_term n k) ∧
  (∀ k, k ≠ 5 ∧ k ≠ 6 → expansion_term n 6 ≥ expansion_term n k) ∧
  expansion_term n 7 = expansion_term n 8 ∧
  expansion_term n 7 = 42240 ∧
  (∀ k, k ≠ 7 ∧ k ≠ 8 → expansion_term n 7 > expansion_term n k) :=
sorry

end largest_coefficients_in_expansion_l1001_100179


namespace point_on_exponential_graph_tan_value_l1001_100180

theorem point_on_exponential_graph_tan_value :
  ∀ a : ℝ, (3 : ℝ)^a = 9 → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end point_on_exponential_graph_tan_value_l1001_100180


namespace computer_table_markup_l1001_100129

/-- Calculate the percentage markup on a product's cost price. -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- The percentage markup on a computer table -/
theorem computer_table_markup :
  percentage_markup 8340 6672 = 25 := by
  sorry

end computer_table_markup_l1001_100129


namespace smallest_number_divisibility_l1001_100196

theorem smallest_number_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(∀ d ∈ [12, 16, 18, 21, 28, 35, 45], (m - 4) % d = 0)) ∧
  (∀ d ∈ [12, 16, 18, 21, 28, 35, 45], (n - 4) % d = 0) ∧
  n = 5044 :=
by sorry

end smallest_number_divisibility_l1001_100196


namespace cos_135_degrees_l1001_100172

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_degrees_l1001_100172


namespace min_value_expression_l1001_100181

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 4*a + 2)*(b^2 + 4*b + 2)*(c^2 + 4*c + 2)) / (a*b*c) ≥ 512 ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    ((a₀^2 + 4*a₀ + 2)*(b₀^2 + 4*b₀ + 2)*(c₀^2 + 4*c₀ + 2)) / (a₀*b₀*c₀) = 512) :=
by sorry

end min_value_expression_l1001_100181


namespace cone_volume_gravel_pile_l1001_100137

/-- The volume of a cone with base diameter 10 feet and height 80% of its diameter is 200π/3 cubic feet. -/
theorem cone_volume_gravel_pile :
  let base_diameter : ℝ := 10
  let height_ratio : ℝ := 0.8
  let height : ℝ := height_ratio * base_diameter
  let radius : ℝ := base_diameter / 2
  let volume : ℝ := (1 / 3) * π * radius^2 * height
  volume = 200 * π / 3 := by
  sorry

end cone_volume_gravel_pile_l1001_100137


namespace vector_dot_product_l1001_100185

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

theorem vector_dot_product :
  (2 • a + b) • c = -3 := by sorry

end vector_dot_product_l1001_100185


namespace min_value_on_interval_l1001_100171

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -3 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end min_value_on_interval_l1001_100171


namespace sufficient_not_necessary_l1001_100153

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x > 0) ∧ 
  (∃ x, x > 0 ∧ ¬(x > 1)) :=
sorry

end sufficient_not_necessary_l1001_100153


namespace quadratic_problem_l1001_100119

-- Define the quadratic equation
def quadratic (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Theorem statement
theorem quadratic_problem (p q : ℝ) 
  (h1 : quadratic p (q + 1) 2 = 0) : 
  -- 1. Relationship between q and p
  q = -2*p - 5 ∧ 
  -- 2. Two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic p q x₁ = 0 ∧ quadratic p q x₂ = 0 ∧
  -- 3. If equal roots in original equation, roots of modified equation
  (∃ (r : ℝ), ∀ x, quadratic p (q + 1) x = 0 → x = r) → 
    quadratic p q 1 = 0 ∧ quadratic p q 3 = 0 := by
  sorry

end quadratic_problem_l1001_100119


namespace seed_ratio_proof_l1001_100186

def total_seeds : ℕ := 120
def left_seeds : ℕ := 20
def additional_seeds : ℕ := 30
def remaining_seeds : ℕ := 30

theorem seed_ratio_proof :
  let used_seeds := total_seeds - remaining_seeds
  let right_seeds := used_seeds - left_seeds - additional_seeds
  (right_seeds : ℚ) / left_seeds = 2 / 1 := by
sorry

end seed_ratio_proof_l1001_100186


namespace mean_median_difference_l1001_100136

def is_valid_set (x d : ℕ) : Prop :=
  x > 0 ∧ x + 2 > 0 ∧ x + 4 > 0 ∧ x + 7 > 0 ∧ x + d > 0

def median (x : ℕ) : ℕ := x + 4

def mean (x d : ℕ) : ℚ :=
  (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5

theorem mean_median_difference (x d : ℕ) :
  is_valid_set x d →
  mean x d = (median x : ℚ) + 5 →
  d = 32 := by
  sorry

end mean_median_difference_l1001_100136


namespace rod_cutting_l1001_100140

theorem rod_cutting (rod_length : Real) (piece_length : Real) :
  rod_length = 42.5 →
  piece_length = 0.85 →
  Int.floor (rod_length / piece_length) = 50 := by
  sorry

end rod_cutting_l1001_100140


namespace extremum_when_a_zero_range_of_m_l1001_100131

noncomputable section

def g (a x : ℝ) : ℝ := (2 - a) * Real.log x

def h (a x : ℝ) : ℝ := Real.log x + a * x^2

def f (a x : ℝ) : ℝ := g a x + (deriv (h a)) x

theorem extremum_when_a_zero :
  let f₀ := f 0
  ∃ (x_min : ℝ), x_min = 1/2 ∧ 
    (∀ x > 0, f₀ x ≥ f₀ x_min) ∧
    f₀ x_min = 2 - 2 * Real.log 2 ∧
    (∀ M : ℝ, ∃ x > 0, f₀ x > M) :=
sorry

theorem range_of_m (a : ℝ) (h : -8 < a ∧ a < -2) :
  let m_lower := 2 / (3 * Real.exp 2) - 4
  ∀ m > m_lower,
    ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 →
      |f a x₁ - f a x₂| > (m + Real.log 3) * a - 2 * Real.log 3 + 2/3 * Real.log (-a) :=
sorry

end extremum_when_a_zero_range_of_m_l1001_100131


namespace smallest_sum_of_five_primes_with_unique_digits_l1001_100187

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the set of digits used in a number -/
def digitsUsed (n : ℕ) : Finset ℕ := sorry

/-- A function that returns the sum of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem smallest_sum_of_five_primes_with_unique_digits :
  ∃ (primes : List ℕ),
    primes.length = 5 ∧
    (∀ p ∈ primes, isPrime p) ∧
    (digitsUsed (sumList primes) = Finset.range 9) ∧
    (∀ s : List ℕ,
      s.length = 5 →
      (∀ p ∈ s, isPrime p) →
      (digitsUsed (sumList s) = Finset.range 9) →
      sumList primes ≤ sumList s) ∧
    sumList primes = 106 :=
sorry

end smallest_sum_of_five_primes_with_unique_digits_l1001_100187


namespace movie_watching_times_l1001_100134

/-- Represents the duration of the movie in minutes -/
def movie_duration : ℕ := 120

/-- Represents the time difference in minutes between when Camila and Maverick started watching -/
def camila_maverick_diff : ℕ := 30

/-- Represents the time difference in minutes between when Maverick and Daniella started watching -/
def maverick_daniella_diff : ℕ := 45

/-- Represents the number of minutes Daniella has left to watch -/
def daniella_remaining : ℕ := 30

/-- Theorem stating that Camila and Maverick have finished watching when Daniella has 30 minutes left -/
theorem movie_watching_times :
  let camila_watched := movie_duration + maverick_daniella_diff + camila_maverick_diff
  let maverick_watched := movie_duration + maverick_daniella_diff
  let daniella_watched := movie_duration - daniella_remaining
  camila_watched ≥ movie_duration ∧ maverick_watched ≥ movie_duration ∧ daniella_watched < movie_duration :=
by sorry

end movie_watching_times_l1001_100134


namespace rectangle_count_l1001_100139

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of horizontal lines --/
def horizontal_lines : ℕ := 5

/-- The number of vertical lines --/
def vertical_lines : ℕ := 4

/-- The number of lines needed to form a rectangle --/
def lines_for_rectangle : ℕ := 4

/-- The number of horizontal lines needed for a rectangle --/
def horizontal_needed : ℕ := 2

/-- The number of vertical lines needed for a rectangle --/
def vertical_needed : ℕ := 2

theorem rectangle_count : 
  (choose horizontal_lines horizontal_needed) * (choose vertical_lines vertical_needed) = 60 := by
  sorry

end rectangle_count_l1001_100139


namespace intersection_point_is_unique_l1001_100110

/-- The first line: x - 2y - 4 = 0 -/
def line1 (x y : ℝ) : Prop := x - 2*y - 4 = 0

/-- The second line: x + 3y + 6 = 0 -/
def line2 (x y : ℝ) : Prop := x + 3*y + 6 = 0

/-- The intersection point (0, -2) -/
def intersection_point : ℝ × ℝ := (0, -2)

/-- Theorem stating that (0, -2) is the unique intersection point of the two lines -/
theorem intersection_point_is_unique :
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) := by
  sorry

end intersection_point_is_unique_l1001_100110


namespace square_side_length_l1001_100128

theorem square_side_length (s : ℝ) : s > 0 → (4 * s = 2 * s^2) → s = 2 := by
  sorry

end square_side_length_l1001_100128


namespace range_of_h_l1001_100154

def f (x : ℝ) : ℝ := 4 * x - 3

def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h :
  let S : Set ℝ := {y | ∃ x ∈ Set.Icc (-1 : ℝ) 3, h x = y}
  S = Set.Icc (-127 : ℝ) 129 := by
  sorry

end range_of_h_l1001_100154


namespace negation_existential_quadratic_l1001_100174

theorem negation_existential_quadratic :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) := by
  sorry

end negation_existential_quadratic_l1001_100174


namespace work_completion_l1001_100184

/-- The number of men in the first group -/
def men_first : ℕ := 15

/-- The number of days for the first group to complete the work -/
def days_first : ℚ := 25

/-- The number of days for the second group to complete the work -/
def days_second : ℚ := 37/2

/-- The total amount of work in man-days -/
def total_work : ℚ := men_first * days_first

/-- The number of men in the second group -/
def men_second : ℕ := 20

theorem work_completion :
  (men_second : ℚ) * days_second = total_work :=
sorry

end work_completion_l1001_100184


namespace shortest_distance_between_tangents_l1001_100192

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P on C₁
def P : ℝ × ℝ := (2, 1)

-- Define the point Q
def Q : ℝ × ℝ := (0, 2)

-- Define the line l (implicitly by Q and its intersection with C₁)
def l (x y : ℝ) : Prop := ∃ (k : ℝ), y - Q.2 = k * (x - Q.1)

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := x^2 = 2*y - 4

-- Define the tangent lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := ∃ (x₃ y₃ : ℝ), C₁ x₃ y₃ ∧ 2*x*x₃ - 2*y - 2*x₃^2 = 0

def l₂ (x y : ℝ) : Prop := ∃ (x₄ y₄ : ℝ), C₂ x₄ y₄ ∧ 2*x*x₄ - 2*y - x₄^2 + 4 = 0

-- The theorem to prove
theorem shortest_distance_between_tangents :
  ∀ (x₃ : ℝ), l₁ x₃ (x₃^2/4) → l₂ (x₃/2) ((x₃/2)^2/2 + 2) →
  (x₃^2 + 4) / (2 * Real.sqrt (x₃^2 + 1)) ≥ Real.sqrt 3 :=
sorry

end shortest_distance_between_tangents_l1001_100192


namespace fraction_simplification_l1001_100182

theorem fraction_simplification (x y : ℝ) (h : y = x / (1 - 2*x)) :
  (2*x - 3*x*y - 2*y) / (y + x*y - x) = -7/3 := by
  sorry

end fraction_simplification_l1001_100182


namespace solve_equation_l1001_100177

theorem solve_equation (y : ℤ) : 7 + y = 3 ↔ y = -4 := by
  sorry

end solve_equation_l1001_100177


namespace store_employees_l1001_100138

/-- The number of employees in Sergio's store -/
def num_employees : ℕ := 20

/-- The initial average number of items sold per employee -/
def initial_average : ℚ := 75

/-- The new average number of items sold per employee -/
def new_average : ℚ := 783/10

/-- The number of items sold by the top three performers on the next day -/
def top_three_sales : ℕ := 6 + 5 + 4

theorem store_employees :
  (initial_average * num_employees + top_three_sales + 3 * (num_employees - 3)) / num_employees = new_average :=
sorry

end store_employees_l1001_100138


namespace factorizations_of_945_l1001_100104

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  a * b = 945 ∧ is_two_digit a ∧ is_two_digit b

def unique_factorizations : Prop :=
  ∃ (f₁ f₂ : ℕ × ℕ),
    valid_factorization f₁.1 f₁.2 ∧
    valid_factorization f₂.1 f₂.2 ∧
    f₁ ≠ f₂ ∧
    (∀ (g : ℕ × ℕ), valid_factorization g.1 g.2 → g = f₁ ∨ g = f₂)

theorem factorizations_of_945 : unique_factorizations := by
  sorry

end factorizations_of_945_l1001_100104


namespace cistern_fill_time_with_leak_l1001_100142

/-- Proves that a cistern with given fill and empty rates takes 2 additional hours to fill due to a leak -/
theorem cistern_fill_time_with_leak 
  (normal_fill_time : ℝ) 
  (empty_time : ℝ) 
  (h1 : normal_fill_time = 4) 
  (h2 : empty_time = 12) : 
  let fill_rate := 1 / normal_fill_time
  let leak_rate := 1 / empty_time
  let effective_rate := fill_rate - leak_rate
  let time_with_leak := 1 / effective_rate
  time_with_leak - normal_fill_time = 2 := by
  sorry

end cistern_fill_time_with_leak_l1001_100142


namespace last_digit_of_product_l1001_100197

theorem last_digit_of_product : (3^101 * 5^89 * 6^127 * 7^139 * 11^79 * 13^67 * 17^53) % 10 = 2 := by
  sorry

end last_digit_of_product_l1001_100197


namespace import_tax_threshold_l1001_100105

/-- The amount in excess of which the import tax was applied -/
def X : ℝ :=
  1000

/-- The import tax rate -/
def tax_rate : ℝ :=
  0.07

/-- The total value of the item -/
def total_value : ℝ :=
  2250

/-- The import tax paid -/
def tax_paid : ℝ :=
  87.50

theorem import_tax_threshold :
  tax_rate * (total_value - X) = tax_paid :=
by sorry

end import_tax_threshold_l1001_100105


namespace gavins_blue_shirts_l1001_100150

theorem gavins_blue_shirts (total_shirts : ℕ) (green_shirts : ℕ) (blue_shirts : ℕ) :
  total_shirts = 23 →
  green_shirts = 17 →
  total_shirts = green_shirts + blue_shirts →
  blue_shirts = 6 := by
sorry

end gavins_blue_shirts_l1001_100150


namespace triplet_sum_not_one_l1001_100170

theorem triplet_sum_not_one : ∃! (a b c : ℝ), 
  ((a = 1.1 ∧ b = -2.1 ∧ c = 1.0) ∨ 
   (a = 1/2 ∧ b = 1/3 ∧ c = 1/6) ∨ 
   (a = 2 ∧ b = -2 ∧ c = 1) ∨ 
   (a = 0.1 ∧ b = 0.3 ∧ c = 0.6) ∨ 
   (a = -3/2 ∧ b = -5/2 ∧ c = 5)) ∧ 
  a + b + c ≠ 1 := by
sorry

end triplet_sum_not_one_l1001_100170


namespace games_mike_can_buy_l1001_100123

/-- The maximum number of games that can be bought given initial money, spent money, and game cost. -/
def max_games_buyable (initial_money spent_money game_cost : ℕ) : ℕ :=
  (initial_money - spent_money) / game_cost

/-- Theorem stating that given the specific values in the problem, the maximum number of games that can be bought is 4. -/
theorem games_mike_can_buy :
  max_games_buyable 42 10 8 = 4 := by
  sorry

end games_mike_can_buy_l1001_100123


namespace female_population_count_l1001_100118

def total_population : ℕ := 5000
def male_population : ℕ := 2000
def females_with_glasses : ℕ := 900
def female_glasses_percentage : ℚ := 30 / 100

theorem female_population_count : 
  ∃ (female_population : ℕ), 
    female_population = total_population - male_population ∧
    female_population = females_with_glasses / female_glasses_percentage :=
by sorry

end female_population_count_l1001_100118


namespace remainder_thirteen_power_fiftyone_mod_five_l1001_100190

theorem remainder_thirteen_power_fiftyone_mod_five :
  13^51 % 5 = 2 := by sorry

end remainder_thirteen_power_fiftyone_mod_five_l1001_100190


namespace curtain_length_is_101_l1001_100103

/-- The required curtain length in inches, given room height in feet, 
    additional length for pooling, and the conversion factor from feet to inches. -/
def curtain_length (room_height_ft : ℕ) (pooling_inches : ℕ) (inches_per_foot : ℕ) : ℕ :=
  room_height_ft * inches_per_foot + pooling_inches

/-- Theorem stating that the required curtain length is 101 inches 
    for the given conditions. -/
theorem curtain_length_is_101 :
  curtain_length 8 5 12 = 101 := by
  sorry

end curtain_length_is_101_l1001_100103


namespace die_roll_count_l1001_100194

theorem die_roll_count (total_sides : ℕ) (red_sides : ℕ) (prob : ℚ) : 
  total_sides = 10 →
  red_sides = 3 →
  prob = 147/1000 →
  (red_sides / total_sides : ℚ) * (1 - red_sides / total_sides : ℚ)^2 = prob →
  3 = 3 :=
by sorry

end die_roll_count_l1001_100194


namespace train_length_l1001_100159

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 15 → speed * time * (5 / 18) = 375 := by
  sorry

end train_length_l1001_100159


namespace cream_fraction_is_four_ninths_l1001_100113

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : CupState :=
  { cup1 := { coffee := 5, cream := 0 },
    cup2 := { coffee := 0, cream := 5 } }

def pour_half_coffee (state : CupState) : CupState :=
  { cup1 := { coffee := state.cup1.coffee / 2, cream := state.cup1.cream },
    cup2 := { coffee := state.cup2.coffee + state.cup1.coffee / 2, cream := state.cup2.cream } }

def add_cream (state : CupState) : CupState :=
  { cup1 := state.cup1,
    cup2 := { coffee := state.cup2.coffee, cream := state.cup2.cream + 1 } }

def pour_half_back (state : CupState) : CupState :=
  let total_cup2 := state.cup2.coffee + state.cup2.cream
  let half_cup2 := total_cup2 / 2
  let coffee_ratio := state.cup2.coffee / total_cup2
  let cream_ratio := state.cup2.cream / total_cup2
  { cup1 := { coffee := state.cup1.coffee + half_cup2 * coffee_ratio,
              cream := state.cup1.cream + half_cup2 * cream_ratio },
    cup2 := { coffee := state.cup2.coffee - half_cup2 * coffee_ratio,
              cream := state.cup2.cream - half_cup2 * cream_ratio } }

theorem cream_fraction_is_four_ninths :
  let final_state := pour_half_back (add_cream (pour_half_coffee initial_state))
  let total_cup1 := final_state.cup1.coffee + final_state.cup1.cream
  final_state.cup1.cream / total_cup1 = 4 / 9 := by sorry

end cream_fraction_is_four_ninths_l1001_100113


namespace number_of_boys_at_park_l1001_100124

theorem number_of_boys_at_park : 
  ∀ (girls parents groups group_size total_people boys : ℕ),
    girls = 14 →
    parents = 50 →
    groups = 3 →
    group_size = 25 →
    total_people = groups * group_size →
    boys = total_people - (girls + parents) →
    boys = 11 := by
  sorry

end number_of_boys_at_park_l1001_100124


namespace revenue_decrease_percentage_l1001_100178

def old_revenue : ℝ := 72.0
def new_revenue : ℝ := 48.0

theorem revenue_decrease_percentage :
  (old_revenue - new_revenue) / old_revenue * 100 = 33.33 := by
  sorry

end revenue_decrease_percentage_l1001_100178


namespace circle_x_axis_intersection_sum_l1001_100156

/-- The sum of x-coordinates of intersection points between a circle and the x-axis -/
def sum_x_coordinates (h k r : ℝ) : ℝ :=
  2 * h

/-- Theorem: For a circle with center (3, -5) and radius 7, 
    the sum of x-coordinates of its intersection points with the x-axis is 6 -/
theorem circle_x_axis_intersection_sum :
  sum_x_coordinates 3 (-5) 7 = 6 := by
  sorry


end circle_x_axis_intersection_sum_l1001_100156


namespace max_value_4x_3y_l1001_100176

theorem max_value_4x_3y (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 8 →
  4*x + 3*y ≤ Real.sqrt (5184 - 173.33) - 72 :=
by sorry

end max_value_4x_3y_l1001_100176


namespace circle_center_coordinate_sum_l1001_100169

/-- Given two points (5, 3) and (-7, 9) as endpoints of a circle's diameter,
    prove that the sum of the coordinates of the circle's center is 5. -/
theorem circle_center_coordinate_sum : 
  let p1 : ℝ × ℝ := (5, 3)
  let p2 : ℝ × ℝ := (-7, 9)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  center.1 + center.2 = 5 := by
  sorry

end circle_center_coordinate_sum_l1001_100169


namespace elevator_initial_floor_l1001_100168

def elevator_problem (initial_floor final_floor top_floor down_move up_move1 up_move2 : ℕ) : Prop :=
  final_floor = top_floor ∧
  top_floor = 13 ∧
  final_floor = initial_floor - down_move + up_move1 + up_move2 ∧
  down_move = 7 ∧
  up_move1 = 3 ∧
  up_move2 = 8

theorem elevator_initial_floor :
  ∀ initial_floor final_floor top_floor down_move up_move1 up_move2 : ℕ,
    elevator_problem initial_floor final_floor top_floor down_move up_move1 up_move2 →
    initial_floor = 9 :=
by
  sorry

end elevator_initial_floor_l1001_100168


namespace sandy_shorts_cost_l1001_100127

/-- Represents the amount Sandy spent on shorts -/
def S : ℝ := sorry

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := 12.14

/-- The amount Sandy received for returning a jacket -/
def jacket_return : ℝ := 7.43

/-- The net amount Sandy spent on clothes -/
def net_spent : ℝ := 18.7

/-- Theorem stating that Sandy spent $13.99 on shorts -/
theorem sandy_shorts_cost : S = 13.99 :=
  by
    have h : S + shirt_cost - jacket_return = net_spent := by sorry
    sorry


end sandy_shorts_cost_l1001_100127


namespace stratified_sample_size_l1001_100151

/-- Represents the ratio of product types A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a stratified sample -/
structure StratifiedSample where
  ratio : ProductRatio
  type_a_count : ℕ
  total_size : ℕ

/-- Theorem: Given a stratified sample with product ratio 5:2:3 and 15 Type A products,
    the total sample size is 30 -/
theorem stratified_sample_size
  (sample : StratifiedSample)
  (h_ratio : sample.ratio = ⟨5, 2, 3⟩)
  (h_type_a : sample.type_a_count = 15) :
  sample.total_size = 30 := by
  sorry

end stratified_sample_size_l1001_100151


namespace distance_A_to_C_l1001_100157

/-- Proves the distance between city A and C given travel times, distance A to B, and speed ratio -/
theorem distance_A_to_C 
  (eddy_time : ℝ) 
  (freddy_time : ℝ) 
  (distance_AB : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : eddy_time = 3) 
  (h2 : freddy_time = 4) 
  (h3 : distance_AB = 540) 
  (h4 : speed_ratio = 2.4) : 
  distance_AB * freddy_time / (eddy_time * speed_ratio) = 300 := by
sorry

end distance_A_to_C_l1001_100157


namespace average_headcount_rounded_l1001_100112

def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600
def fall_05_06_headcount : ℕ := 11300

def average_headcount : ℚ :=
  (fall_03_04_headcount + fall_04_05_headcount + fall_05_06_headcount) / 3

def round_to_nearest (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

theorem average_headcount_rounded : 
  round_to_nearest average_headcount = 11467 := by sorry

end average_headcount_rounded_l1001_100112


namespace remainder_equality_l1001_100191

theorem remainder_equality (A B D S T u v : ℕ) 
  (h1 : A > B)
  (h2 : S = A % D)
  (h3 : T = B % D)
  (h4 : u = (A + B) % D)
  (h5 : v = (S + T) % D) :
  u = v := by
  sorry

end remainder_equality_l1001_100191


namespace g_50_equals_zero_l1001_100114

theorem g_50_equals_zero
  (g : ℕ → ℕ)
  (h : ∀ a b : ℕ, 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b)) :
  g 50 = 0 := by
sorry

end g_50_equals_zero_l1001_100114


namespace geometric_arithmetic_sequence_problem_l1001_100199

theorem geometric_arithmetic_sequence_problem (b q a d : ℝ) 
  (h1 : b = a + d)
  (h2 : b * q = a + 3 * d)
  (h3 : b * q^2 = a + 6 * d)
  (h4 : b * (b * q) * (b * q^2) = 64) :
  b = 8 / 3 := by
sorry

end geometric_arithmetic_sequence_problem_l1001_100199


namespace tangent_line_equation_l1001_100100

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 4)

-- State the theorem
theorem tangent_line_equation :
  let m := (2 * P.1 + 3) -- Slope of the tangent line
  (5 : ℝ) * x - y - 1 = 0 ↔ y - P.2 = m * (x - P.1) :=
sorry

end tangent_line_equation_l1001_100100


namespace max_value_fraction_l1001_100162

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (M : ℝ), M = 1/4 ∧ 
  (x * y * z * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ M ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * c * (a + b + c)) / ((a + c)^2 * (c + b)^2) = M :=
by sorry

end max_value_fraction_l1001_100162


namespace unique_solutions_l1001_100173

/-- A pair of positive integers (m, n) satisfies the given conditions if
    m^2 - 4n and n^2 - 4m are both perfect squares. -/
def satisfies_conditions (m n : ℕ+) : Prop :=
  ∃ k l : ℕ, (m : ℤ)^2 - 4*(n : ℤ) = (k : ℤ)^2 ∧ (n : ℤ)^2 - 4*(m : ℤ) = (l : ℤ)^2

/-- The theorem stating that the only pairs of positive integers (m, n) satisfying
    the conditions are (4, 4), (5, 6), and (6, 5). -/
theorem unique_solutions :
  ∀ m n : ℕ+, satisfies_conditions m n ↔ 
    ((m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5)) :=
by sorry

end unique_solutions_l1001_100173


namespace large_pizza_cost_is_10_l1001_100117

/-- Represents the cost of a pizza topping --/
structure ToppingCost where
  first : ℝ
  next_two : ℝ
  rest : ℝ

/-- Calculates the total cost of toppings --/
def total_topping_cost (tc : ToppingCost) (num_toppings : ℕ) : ℝ :=
  tc.first + 
  (if num_toppings > 1 then min (num_toppings - 1) 2 * tc.next_two else 0) +
  (if num_toppings > 3 then (num_toppings - 3) * tc.rest else 0)

/-- The cost of a large pizza without toppings --/
def large_pizza_cost (slices : ℕ) (cost_per_slice : ℝ) (num_toppings : ℕ) (tc : ToppingCost) : ℝ :=
  slices * cost_per_slice - total_topping_cost tc num_toppings

/-- Theorem: The cost of a large pizza without toppings is $10.00 --/
theorem large_pizza_cost_is_10 : 
  large_pizza_cost 8 2 7 ⟨2, 1, 0.5⟩ = 10 := by
  sorry

end large_pizza_cost_is_10_l1001_100117


namespace six_digit_number_theorem_l1001_100109

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (m k : ℕ),
    m < 10000 ∧ k < 100 ∧
    n = 100000 * (n / 100000) + 1000 * (n / 1000 % 100) + (n % 1000) ∧
    4 * n = k * 10000 + m ∧
    n = m * 100 + k

theorem six_digit_number_theorem :
  {n : ℕ | is_valid_number n} = {142857, 190476, 238095} :=
sorry

end six_digit_number_theorem_l1001_100109


namespace trig_identity_proof_l1001_100163

theorem trig_identity_proof (a : ℝ) : 
  Real.cos (a + π/6) * Real.sin (a - π/3) + Real.sin (a + π/6) * Real.cos (a - π/3) = 1/2 := by
  sorry

end trig_identity_proof_l1001_100163


namespace initial_apples_count_l1001_100183

/-- The number of apples Sarah initially had in her bag -/
def initial_apples : ℕ := 25

/-- The number of apples Sarah gave to teachers -/
def apples_to_teachers : ℕ := 16

/-- The number of apples Sarah gave to friends -/
def apples_to_friends : ℕ := 5

/-- The number of apples Sarah ate -/
def apples_eaten : ℕ := 1

/-- The number of apples left in Sarah's bag when she got home -/
def apples_left : ℕ := 3

/-- Theorem stating that the initial number of apples equals the sum of apples given away, eaten, and left -/
theorem initial_apples_count : 
  initial_apples = apples_to_teachers + apples_to_friends + apples_eaten + apples_left :=
by sorry

end initial_apples_count_l1001_100183


namespace line_perp_parallel_implies_planes_perp_l1001_100189

-- Define the types for lines and planes
variable (L : Type) (P : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : L → P → Prop)
variable (parallel : L → P → Prop)
variable (planePerpendicular : P → P → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : L) (α β : P) 
  (h1 : perpendicular l α)
  (h2 : parallel l β) :
  planePerpendicular α β :=
sorry

end line_perp_parallel_implies_planes_perp_l1001_100189


namespace intersection_point_of_lines_l1001_100141

theorem intersection_point_of_lines (x y : ℚ) :
  (8 * x - 5 * y = 40) ∧ (6 * x + 2 * y = 14) ↔ x = 75 / 23 ∧ y = -64 / 23 := by
  sorry

end intersection_point_of_lines_l1001_100141


namespace system_solution_independent_of_c_l1001_100148

theorem system_solution_independent_of_c :
  ∀ (c : ℝ),
    2 - 0 + 2*(-1) = 0 ∧
    -2*2 + 0 - 2*(-1) = -2 ∧
    2*2 + c*0 + 3*(-1) = 1 := by
  sorry

end system_solution_independent_of_c_l1001_100148


namespace max_a_value_l1001_100165

def is_lattice_point (x y : ℤ) : Prop := True

def passes_through_lattice_point (m : ℚ) (b : ℤ) : Prop :=
  ∃ x y : ℤ, is_lattice_point x y ∧ 0 < x ∧ x ≤ 200 ∧ y = m * x + b

theorem max_a_value :
  let a : ℚ := 68 / 201
  ∀ m : ℚ, 1/3 < m → m < a →
    ¬(passes_through_lattice_point m 3 ∨ passes_through_lattice_point m 1) ∧
    ∀ a' : ℚ, a < a' →
      ∃ m : ℚ, 1/3 < m ∧ m < a' ∧
        (passes_through_lattice_point m 3 ∨ passes_through_lattice_point m 1) :=
by sorry

end max_a_value_l1001_100165


namespace smaller_number_in_ratio_l1001_100111

theorem smaller_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 2 / 5 → x + y = 21 → min x y = 6 := by
  sorry

end smaller_number_in_ratio_l1001_100111
