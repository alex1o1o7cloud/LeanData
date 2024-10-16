import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3372_337215

theorem binomial_expansion_coefficient (a : ℝ) (h : a ≠ 0) :
  let expansion := fun (x : ℝ) ↦ (x - a / x)^6
  let B := expansion 1  -- Constant term when x = 1
  B = 44 → a = -22/5 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3372_337215


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l3372_337212

theorem max_rabbits_with_traits (N : ℕ) : 
  (∃ (long_ears jump_far both : Finset (Fin N)),
    long_ears.card = 13 ∧ 
    jump_far.card = 17 ∧ 
    both ⊆ long_ears ∧ 
    both ⊆ jump_far ∧ 
    both.card ≥ 3) →
  N ≤ 27 := by
sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l3372_337212


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3372_337291

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (x + 1)}
def B : Set ℝ := {x | 1 / (x + 1) < 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x > 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3372_337291


namespace NUMINAMATH_CALUDE_cotton_amount_l3372_337261

/-- Given:
  * Kevin plants corn and cotton
  * He harvests 30 pounds of corn and x pounds of cotton
  * Corn sells for $5 per pound
  * Cotton sells for $10 per pound
  * Total revenue from selling all corn and cotton is $640
Prove that x = 49 -/
theorem cotton_amount (x : ℝ) : 
  (30 * 5 + x * 10 = 640) → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_cotton_amount_l3372_337261


namespace NUMINAMATH_CALUDE_farm_distance_is_six_l3372_337258

/-- Represents the distance to the farm given the conditions of Bobby's trips -/
def distance_to_farm (initial_gas : ℝ) (supermarket_distance : ℝ) (partial_farm_trip : ℝ) 
  (final_gas : ℝ) (miles_per_gallon : ℝ) : ℝ :=
  let total_miles_driven := (initial_gas - final_gas) * miles_per_gallon
  let known_miles := 2 * supermarket_distance + 2 * partial_farm_trip
  total_miles_driven - known_miles

/-- Theorem stating that the distance to the farm is 6 miles -/
theorem farm_distance_is_six :
  distance_to_farm 12 5 2 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_farm_distance_is_six_l3372_337258


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3372_337243

/-- The range of 'a' for which the ellipse x^2 + 4(y-a)^2 = 4 and 
    the parabola x^2 = 2y have common points -/
theorem ellipse_parabola_intersection_range :
  ∀ a : ℝ, 
  (∃ x y : ℝ, x^2 + 4*(y-a)^2 = 4 ∧ x^2 = 2*y) → 
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3372_337243


namespace NUMINAMATH_CALUDE_rectangular_playground_vertical_length_l3372_337249

/-- The vertical length of a rectangular playground given specific conditions -/
theorem rectangular_playground_vertical_length :
  ∀ (square_side : ℝ) (rect_horizontal : ℝ) (rect_vertical : ℝ),
    square_side = 12 →
    rect_horizontal = 9 →
    4 * square_side = 2 * (rect_horizontal + rect_vertical) →
    rect_vertical = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_playground_vertical_length_l3372_337249


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l3372_337242

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem: The smallest number of identical cubes that can fill a box with dimensions 
    24 inches long, 40 inches wide, and 16 inches deep is 30 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes { length := 24, width := 40, depth := 16 } = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l3372_337242


namespace NUMINAMATH_CALUDE_sum_to_k_perfect_square_l3372_337270

theorem sum_to_k_perfect_square (k : ℕ) :
  (∃ n : ℕ, n < 100 ∧ k * (k + 1) / 2 = n^2) → k = 1 ∨ k = 8 ∨ k = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_perfect_square_l3372_337270


namespace NUMINAMATH_CALUDE_solve_for_y_l3372_337244

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 2*x = y - 4) (h2 : x = -6) : y = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3372_337244


namespace NUMINAMATH_CALUDE_system_solutions_l3372_337232

def equation1 (x y : ℝ) : Prop := (x + 2*y) * (x + 3*y) = x + y

def equation2 (x y : ℝ) : Prop := (2*x + y) * (3*x + y) = -99 * (x + y)

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (-14, 6), (-85/6, 35/6)}

theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3372_337232


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3372_337271

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3372_337271


namespace NUMINAMATH_CALUDE_fishing_ratio_l3372_337241

/-- Given that Tom caught 16 trout and Melanie caught 8 trout, 
    prove that the ratio of Tom's catch to Melanie's catch is 2. -/
theorem fishing_ratio (tom_catch melanie_catch : ℕ) 
  (h1 : tom_catch = 16) (h2 : melanie_catch = 8) : 
  (tom_catch : ℚ) / melanie_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_fishing_ratio_l3372_337241


namespace NUMINAMATH_CALUDE_correct_yeast_experiment_methods_l3372_337235

/-- Represents the method used for counting yeast -/
inductive CountingMethod
| SamplingInspection
| Other

/-- Represents the action taken before extracting culture fluid -/
inductive PreExtractionAction
| GentlyShake
| Other

/-- Represents the measure taken when there are too many yeast cells -/
inductive OvercrowdingMeasure
| AppropriateDilution
| Other

/-- Represents the conditions of the yeast counting experiment -/
structure YeastExperiment where
  countingMethod : CountingMethod
  preExtractionAction : PreExtractionAction
  overcrowdingMeasure : OvercrowdingMeasure

/-- Theorem stating the correct methods and actions for the yeast counting experiment -/
theorem correct_yeast_experiment_methods :
  ∀ (experiment : YeastExperiment),
    experiment.countingMethod = CountingMethod.SamplingInspection ∧
    experiment.preExtractionAction = PreExtractionAction.GentlyShake ∧
    experiment.overcrowdingMeasure = OvercrowdingMeasure.AppropriateDilution :=
by sorry

end NUMINAMATH_CALUDE_correct_yeast_experiment_methods_l3372_337235


namespace NUMINAMATH_CALUDE_coin_machine_theorem_l3372_337281

/-- Represents the coin-changing machine's rules --/
structure CoinMachine where
  quarter_to_nickels : ℕ → ℕ
  nickel_to_pennies : ℕ → ℕ
  penny_to_quarters : ℕ → ℕ

/-- Represents the possible amounts in cents --/
def possible_amounts (m : CoinMachine) (n : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, x = 1 + 74 * k}

/-- The set of given options in cents --/
def given_options : Set ℕ := {175, 325, 449, 549, 823}

theorem coin_machine_theorem (m : CoinMachine) 
  (h1 : m.quarter_to_nickels 1 = 5)
  (h2 : m.nickel_to_pennies 1 = 5)
  (h3 : m.penny_to_quarters 1 = 3) :
  given_options ∩ (possible_amounts m 1) = {823} := by
  sorry

end NUMINAMATH_CALUDE_coin_machine_theorem_l3372_337281


namespace NUMINAMATH_CALUDE_first_part_games_l3372_337259

/-- Prove that the number of games in the first part of the season is 100 -/
theorem first_part_games (total_games : ℕ) (first_win_rate remaining_win_rate overall_win_rate : ℚ) : 
  total_games = 175 →
  first_win_rate = 85/100 →
  remaining_win_rate = 1/2 →
  overall_win_rate = 7/10 →
  ∃ (x : ℕ), x = 100 ∧ 
    first_win_rate * x + remaining_win_rate * (total_games - x) = overall_win_rate * total_games :=
by sorry

end NUMINAMATH_CALUDE_first_part_games_l3372_337259


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l3372_337262

theorem perfect_square_quadratic (x k : ℝ) : 
  (∃ a b : ℝ, x^2 - 18*x + k = (a*x + b)^2) ↔ k = 81 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l3372_337262


namespace NUMINAMATH_CALUDE_a_18_value_l3372_337256

/-- An equal sum sequence with common sum c -/
def EqualSumSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem a_18_value (a : ℕ → ℝ) (h : EqualSumSequence a 5) (h1 : a 1 = 2) :
  a 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_18_value_l3372_337256


namespace NUMINAMATH_CALUDE_initial_speed_problem_l3372_337280

theorem initial_speed_problem (v : ℝ) : 
  (0.5 * v + 1 * (2 * v) = 75) → v = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_problem_l3372_337280


namespace NUMINAMATH_CALUDE_max_receivable_amount_l3372_337275

/-- Represents the denominations of chips available at the casino. -/
inductive ChipDenomination
  | TwentyFive
  | SeventyFive
  | TwoFifty

/-- Represents the number of chips lost for each denomination. -/
structure LostChips where
  twentyFive : ℕ
  seventyFive : ℕ
  twoFifty : ℕ

/-- Calculates the total value of chips based on the number of chips for each denomination. -/
def chipValue (chips : LostChips) : ℕ :=
  25 * chips.twentyFive + 75 * chips.seventyFive + 250 * chips.twoFifty

/-- Represents the conditions of the gambling problem. -/
structure GamblingProblem where
  initialValue : ℕ
  lostChips : LostChips
  haveTwentyFiveLeft : Prop
  haveSeventyFiveLeft : Prop
  haveTwoFiftyLeft : Prop
  totalLostChips : ℕ
  lostTwentyFiveTwiceSeventyFive : Prop
  lostSeventyFiveHalfTwoFifty : Prop

/-- Theorem stating the maximum amount the gambler could have received back. -/
theorem max_receivable_amount (problem : GamblingProblem)
  (h1 : problem.initialValue = 15000)
  (h2 : problem.totalLostChips = 40)
  (h3 : problem.lostTwentyFiveTwiceSeventyFive)
  (h4 : problem.lostSeventyFiveHalfTwoFifty)
  (h5 : problem.haveTwentyFiveLeft)
  (h6 : problem.haveSeventyFiveLeft)
  (h7 : problem.haveTwoFiftyLeft) :
  problem.initialValue - chipValue problem.lostChips = 10000 := by
  sorry

#check max_receivable_amount

end NUMINAMATH_CALUDE_max_receivable_amount_l3372_337275


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3372_337257

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem inequality_equivalence (x : ℝ) (hx : x > 0) :
  (lg x ^ 2 - 3 * lg x + 3) / (lg x - 1) < 1 ↔ x < 10 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3372_337257


namespace NUMINAMATH_CALUDE_total_score_is_219_l3372_337231

/-- Represents a player's score in a basketball game -/
structure PlayerScore where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat

/-- Calculates the total score for a player -/
def calculatePlayerScore (score : PlayerScore) : Nat :=
  2 * score.twoPointers + 3 * score.threePointers + score.freeThrows

/-- Theorem: The total points scored by all players is 219 -/
theorem total_score_is_219 
  (sam : PlayerScore)
  (alex : PlayerScore)
  (jake : PlayerScore)
  (lily : PlayerScore)
  (h_sam : sam = { twoPointers := 20, threePointers := 5, freeThrows := 10 })
  (h_alex : alex = { twoPointers := 15, threePointers := 6, freeThrows := 8 })
  (h_jake : jake = { twoPointers := 10, threePointers := 8, freeThrows := 5 })
  (h_lily : lily = { twoPointers := 12, threePointers := 3, freeThrows := 16 }) :
  calculatePlayerScore sam + calculatePlayerScore alex + 
  calculatePlayerScore jake + calculatePlayerScore lily = 219 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_219_l3372_337231


namespace NUMINAMATH_CALUDE_sixty_has_twelve_divisors_l3372_337217

/-- The number of positive divisors of 60 -/
def num_divisors_60 : ℕ := Finset.card (Nat.divisors 60)

/-- Theorem stating that 60 has exactly 12 positive divisors -/
theorem sixty_has_twelve_divisors : num_divisors_60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sixty_has_twelve_divisors_l3372_337217


namespace NUMINAMATH_CALUDE_second_section_students_correct_l3372_337293

/-- The number of students in the second section of chemistry class X -/
def students_section2 : ℕ := 35

/-- The total number of students in all four sections -/
def total_students : ℕ := 65 + students_section2 + 45 + 42

/-- The overall average of marks per student -/
def overall_average : ℚ := 5195 / 100

/-- Theorem stating that the number of students in the second section is correct -/
theorem second_section_students_correct :
  (65 * 50 + students_section2 * 60 + 45 * 55 + 42 * 45 : ℚ) / total_students = overall_average :=
sorry

end NUMINAMATH_CALUDE_second_section_students_correct_l3372_337293


namespace NUMINAMATH_CALUDE_division_problem_l3372_337226

theorem division_problem (n : ℕ) : n % 12 = 1 ∧ n / 12 = 9 → n = 109 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3372_337226


namespace NUMINAMATH_CALUDE_water_added_to_container_l3372_337285

theorem water_added_to_container (capacity : ℝ) (initial_percentage : ℝ) (final_fraction : ℝ) :
  capacity = 120 →
  initial_percentage = 0.35 →
  final_fraction = 3/4 →
  (final_fraction * capacity) - (initial_percentage * capacity) = 48 :=
by sorry

end NUMINAMATH_CALUDE_water_added_to_container_l3372_337285


namespace NUMINAMATH_CALUDE_calculate_expression_l3372_337218

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3372_337218


namespace NUMINAMATH_CALUDE_parabola_translation_l3372_337269

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (p : Parabola) :
  p.a = -2 ∧ p.b = -4 ∧ p.c = -6 →
  translate p 1 3 = Parabola.mk (-2) 0 (-1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3372_337269


namespace NUMINAMATH_CALUDE_tangent_secant_theorem_l3372_337299

theorem tangent_secant_theorem :
  ∃! (s : Finset ℕ), 
    (∀ t ∈ s, ∃ m n : ℕ, 
      t * t = m * n ∧ 
      m + n = 10 ∧ 
      m ≠ n ∧ 
      m > 0 ∧ 
      n > 0) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_secant_theorem_l3372_337299


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3372_337251

/-- The sum of the infinite series ∑(n=1 to ∞) (n+1) / (n^2(n+2)) is equal to 3/8 + π^2/24 -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (n + 1 : ℝ) / (n^2 * (n + 2)) = 3/8 + π^2/24 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3372_337251


namespace NUMINAMATH_CALUDE_cubic_diophantine_equation_l3372_337292

theorem cubic_diophantine_equation :
  ∀ x y z : ℤ, x^3 + 2*y^3 = 4*z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_diophantine_equation_l3372_337292


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3372_337205

/-- Given a geometric sequence {a_n} where a_2 = 2 and a_5 = 1/4,
    prove that the sum a_1*a_2 + a_2*a_3 + ... + a_5*a_6 equals 341/32. -/
theorem geometric_sequence_sum (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2) →  -- geometric sequence property
  a 2 = 2 →
  a 5 = 1/4 →
  (a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 : ℚ) = 341/32 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3372_337205


namespace NUMINAMATH_CALUDE_work_earnings_equation_l3372_337240

theorem work_earnings_equation (t : ℚ) : (t + 2) * (4 * t - 5) = (2 * t + 1) * (2 * t + 3) + 3 ↔ t = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l3372_337240


namespace NUMINAMATH_CALUDE_starting_player_wins_l3372_337227

/-- Represents a game state --/
structure GameState where
  current : ℕ
  isStartingPlayerTurn : Bool

/-- Represents a valid move in the game --/
def ValidMove (state : GameState) (move : ℕ) : Prop :=
  0 < move ∧ move < state.current

/-- Represents the winning condition of the game --/
def IsWinningState (state : GameState) : Prop :=
  state.current = 1987

/-- Represents a winning strategy for the starting player --/
def WinningStrategy : Type :=
  (state : GameState) → {move : ℕ // ValidMove state move}

/-- The theorem stating that the starting player has a winning strategy --/
theorem starting_player_wins :
  ∃ (strategy : WinningStrategy),
    ∀ (game : ℕ → GameState),
      game 0 = ⟨2, true⟩ →
      (∀ n, game (n + 1) = 
        let move := (strategy (game n)).val
        ⟨(game n).current + move, ¬(game n).isStartingPlayerTurn⟩) →
      ∃ n, IsWinningState (game n) ∧ (game n).isStartingPlayerTurn :=
sorry


end NUMINAMATH_CALUDE_starting_player_wins_l3372_337227


namespace NUMINAMATH_CALUDE_smallest_a_for_nonprime_cube_sum_l3372_337207

theorem smallest_a_for_nonprime_cube_sum :
  ∃ (a : ℕ), a > 0 ∧ (∀ (x : ℤ), ¬ Prime (x^3 + a^3)) ∧
  (∀ (b : ℕ), b > 0 ∧ b < a → ∃ (y : ℤ), Prime (y^3 + b^3)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonprime_cube_sum_l3372_337207


namespace NUMINAMATH_CALUDE_chord_midpoint_trajectory_midpoint_PQ_trajectory_l3372_337246

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 12*y + 24 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 5)

-- Theorem for the trajectory of chord midpoints
theorem chord_midpoint_trajectory (x y : ℝ) : 
  (∃ (a b : ℝ), circle_C a b ∧ circle_C (2*x - a) (2*y - b) ∧ (x + a = 0 ∨ y + b = 5)) →
  x^2 + y^2 + 2*x - 11*y + 30 = 0 :=
sorry

-- Theorem for the trajectory of midpoint M of PQ
theorem midpoint_PQ_trajectory (x y : ℝ) :
  (∃ (q_x q_y : ℝ), circle_C q_x q_y ∧ x = (q_x + point_P.1) / 2 ∧ y = (q_y + point_P.2) / 2) →
  x^2 + y^2 + 2*x - 11*y - 11/4 = 0 :=
sorry

end NUMINAMATH_CALUDE_chord_midpoint_trajectory_midpoint_PQ_trajectory_l3372_337246


namespace NUMINAMATH_CALUDE_cubic_root_product_l3372_337290

theorem cubic_root_product (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + a - 7 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 7 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 7 = 0) →
  a * b * c = 7/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l3372_337290


namespace NUMINAMATH_CALUDE_joggers_speed_ratio_l3372_337263

theorem joggers_speed_ratio (v₁ v₂ : ℝ) (h1 : v₁ > v₂) (h2 : (v₁ + v₂) * 2 = 8) (h3 : (v₁ - v₂) * 4 = 8) : v₁ / v₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_joggers_speed_ratio_l3372_337263


namespace NUMINAMATH_CALUDE_inequality_property_equivalence_l3372_337294

theorem inequality_property_equivalence (t : ℝ) (ht : t > 0) :
  (∃ X : Set ℝ, Set.Infinite X ∧
    ∀ (x y z : ℝ) (a : ℝ) (d : ℝ), x ∈ X → y ∈ X → z ∈ X → d > 0 →
      max (|x - (a - d)|) (max (|y - a|) (|z - (a + d)|)) > t * d) ↔
  t < (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_property_equivalence_l3372_337294


namespace NUMINAMATH_CALUDE_three_digit_cube_units_digit_l3372_337267

theorem three_digit_cube_units_digit :
  ∀ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧ 
    (n = (n % 10)^3) →
    (n = 125 ∨ n = 216 ∨ n = 729) :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_units_digit_l3372_337267


namespace NUMINAMATH_CALUDE_money_transfer_proof_l3372_337296

/-- The amount of money (in won) the older brother gave to the younger brother -/
def money_transferred : ℕ := sorry

/-- The initial amount of money (in won) the older brother had -/
def older_brother_initial : ℕ := 2800

/-- The initial amount of money (in won) the younger brother had -/
def younger_brother_initial : ℕ := 1500

/-- The difference in money (in won) between the brothers after the transfer -/
def final_difference : ℕ := 360

theorem money_transfer_proof :
  (older_brother_initial - money_transferred) - (younger_brother_initial + money_transferred) = final_difference ∧
  money_transferred = 470 := by sorry

end NUMINAMATH_CALUDE_money_transfer_proof_l3372_337296


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3372_337211

open Real

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : exp a + 2*a = exp b + 3*b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3372_337211


namespace NUMINAMATH_CALUDE_value_of_expression_l3372_337289

theorem value_of_expression (x y : ℝ) 
  (eq1 : x + 3 * y = -1) 
  (eq2 : x - 3 * y = 5) : 
  x^2 - 9 * y^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3372_337289


namespace NUMINAMATH_CALUDE_inequality_multiplication_l3372_337200

theorem inequality_multiplication (x y : ℝ) (h : x > y) : 3 * x > 3 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l3372_337200


namespace NUMINAMATH_CALUDE_reyn_placed_25_pieces_l3372_337255

/-- Represents the puzzle distribution and placement problem --/
structure PuzzleProblem where
  total_pieces : Nat
  num_sons : Nat
  pieces_left : Nat
  rhys_multiplier : Nat
  rory_multiplier : Nat

/-- Calculates the number of pieces Reyn placed --/
def reyn_pieces (p : PuzzleProblem) : Nat :=
  let pieces_per_son := p.total_pieces / p.num_sons
  let total_placed := p.total_pieces - p.pieces_left
  total_placed / (1 + p.rhys_multiplier + p.rory_multiplier)

/-- Theorem stating that Reyn placed 25 pieces --/
theorem reyn_placed_25_pieces : 
  let p : PuzzleProblem := {
    total_pieces := 300,
    num_sons := 3,
    pieces_left := 150,
    rhys_multiplier := 2,
    rory_multiplier := 3
  }
  reyn_pieces p = 25 := by
  sorry

end NUMINAMATH_CALUDE_reyn_placed_25_pieces_l3372_337255


namespace NUMINAMATH_CALUDE_percentage_between_55_and_65_l3372_337214

/-- Represents the percentage of students who scored at least 55% on the test -/
def scored_at_least_55 : ℝ := 55

/-- Represents the percentage of students who scored at most 65% on the test -/
def scored_at_most_65 : ℝ := 65

/-- Represents the percentage of students who scored between 55% and 65% (inclusive) on the test -/
def scored_between_55_and_65 : ℝ := scored_at_most_65 - (100 - scored_at_least_55)

theorem percentage_between_55_and_65 : scored_between_55_and_65 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_between_55_and_65_l3372_337214


namespace NUMINAMATH_CALUDE_point_outside_region_implies_a_range_l3372_337245

theorem point_outside_region_implies_a_range (a : ℝ) : 
  (2 - (4 * a^2 + 3 * a - 2) * 2 - 4 ≥ 0) → 
  (a ∈ Set.Icc (-1 : ℝ) (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_point_outside_region_implies_a_range_l3372_337245


namespace NUMINAMATH_CALUDE_exists_special_triangle_l3372_337230

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The length of the median from vertex B to the midpoint of AC. -/
def median_length (t : Triangle) : ℝ := sorry

/-- The length of the angle bisector from vertex C. -/
def angle_bisector_length (t : Triangle) : ℝ := sorry

/-- The length of the altitude from vertex A to BC. -/
def altitude_length (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is scalene (no two sides are equal). -/
def is_scalene (t : Triangle) : Prop := sorry

/-- 
There exists a scalene triangle where the median from B, 
the angle bisector from C, and the altitude from A are all equal.
-/
theorem exists_special_triangle : 
  ∃ t : Triangle, 
    is_scalene t ∧ 
    median_length t = angle_bisector_length t ∧
    angle_bisector_length t = altitude_length t :=
sorry

end NUMINAMATH_CALUDE_exists_special_triangle_l3372_337230


namespace NUMINAMATH_CALUDE_power_of_negative_one_difference_l3372_337260

theorem power_of_negative_one_difference : (-1)^2004 - (-1)^2003 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_one_difference_l3372_337260


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l3372_337272

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallsDrawn : ℕ := 6

def winningProbability : ℚ :=
  1 / (megaBallCount * (winnerBallCount.choose winnerBallsDrawn))

theorem lottery_winning_probability :
  winningProbability = 1 / 476721000 := by sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l3372_337272


namespace NUMINAMATH_CALUDE_typical_flowchart_structure_l3372_337288

-- Define a flowchart structure
structure Flowchart where
  start_points : Nat
  end_points : Nat

-- Define a typical flowchart
def typical_flowchart : Flowchart := ⟨1, 1⟩

-- Theorem stating that a typical flowchart has one start and can have multiple ends
theorem typical_flowchart_structure :
  typical_flowchart.start_points = 1 ∧ typical_flowchart.end_points ≥ 1 := by
  sorry

#check typical_flowchart_structure

end NUMINAMATH_CALUDE_typical_flowchart_structure_l3372_337288


namespace NUMINAMATH_CALUDE_student_distribution_proof_l3372_337297

def distribute_students (n : ℕ) (k : ℕ) (min_per_dorm : ℕ) : ℕ :=
  sorry

theorem student_distribution_proof :
  distribute_students 9 3 4 = 3570 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_proof_l3372_337297


namespace NUMINAMATH_CALUDE_class_A_student_count_l3372_337273

/-- Represents the number of students in class (A) -/
def class_A_students : ℕ := 50

/-- Represents the total number of groups in class (A) -/
def total_groups : ℕ := 8

/-- Represents the number of groups with 6 people -/
def groups_of_six : ℕ := 6

/-- Represents the number of groups with 7 people -/
def groups_of_seven : ℕ := 2

/-- Represents the number of people in each of the smaller groups -/
def people_in_smaller_groups : ℕ := 6

/-- Represents the number of people in each of the larger groups -/
def people_in_larger_groups : ℕ := 7

theorem class_A_student_count : 
  class_A_students = 
    groups_of_six * people_in_smaller_groups + 
    groups_of_seven * people_in_larger_groups ∧
  total_groups = groups_of_six + groups_of_seven := by
  sorry

end NUMINAMATH_CALUDE_class_A_student_count_l3372_337273


namespace NUMINAMATH_CALUDE_gross_profit_calculation_l3372_337202

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 91 →
  gross_profit_percentage = 1.6 →
  ∃ (cost : ℝ), sales_price = cost + gross_profit_percentage * cost ∧
                 gross_profit_percentage * cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_gross_profit_calculation_l3372_337202


namespace NUMINAMATH_CALUDE_reflection_curve_coefficient_product_l3372_337248

/-- The reflection of the curve xy = 1 over the line y = 2x -/
def ReflectedCurve (x y : ℝ) : Prop :=
  ∃ (b c d : ℝ), 12 * x^2 + b * x * y + c * y^2 + d = 0

/-- The product of coefficients b and c in the reflected curve equation -/
def CoefficientProduct (b c : ℝ) : ℝ := b * c

theorem reflection_curve_coefficient_product :
  ∃ (b c : ℝ), ReflectedCurve x y ∧ CoefficientProduct b c = 84 := by
  sorry

end NUMINAMATH_CALUDE_reflection_curve_coefficient_product_l3372_337248


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3372_337279

/-- Given a geometric sequence {a_n} where a_6 + a_8 = 4, 
    prove that a_8(a_4 + 2a_6 + a_8) = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3372_337279


namespace NUMINAMATH_CALUDE_totalChargeDifference_l3372_337265

/-- Represents the pricing structure for an air conditioner company -/
structure ACCompany where
  price : ℝ
  surchargeRate : ℝ
  installationCharge : ℝ
  warrantyFee : ℝ
  maintenanceFee : ℝ

/-- Calculates the total charge for a company -/
def totalCharge (c : ACCompany) : ℝ :=
  c.price + (c.surchargeRate * c.price) + c.installationCharge + c.warrantyFee + c.maintenanceFee

/-- Company X's pricing information -/
def companyX : ACCompany :=
  { price := 575
  , surchargeRate := 0.04
  , installationCharge := 82.50
  , warrantyFee := 125
  , maintenanceFee := 50 }

/-- Company Y's pricing information -/
def companyY : ACCompany :=
  { price := 530
  , surchargeRate := 0.03
  , installationCharge := 93.00
  , warrantyFee := 150
  , maintenanceFee := 40 }

/-- Theorem stating the difference in total charges between Company X and Company Y -/
theorem totalChargeDifference :
  totalCharge companyX - totalCharge companyY = 26.60 := by
  sorry

end NUMINAMATH_CALUDE_totalChargeDifference_l3372_337265


namespace NUMINAMATH_CALUDE_total_cost_construction_materials_l3372_337206

def cement_bags : ℕ := 500
def cement_price_per_bag : ℕ := 10
def sand_lorries : ℕ := 20
def sand_tons_per_lorry : ℕ := 10
def sand_price_per_ton : ℕ := 40

theorem total_cost_construction_materials : 
  cement_bags * cement_price_per_bag + 
  sand_lorries * sand_tons_per_lorry * sand_price_per_ton = 13000 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_construction_materials_l3372_337206


namespace NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l3372_337223

theorem sphere_diameter_triple_volume (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 6
  let v₁ : ℝ := (4 / 3) * π * r₁^3
  let v₂ : ℝ := 3 * v₁
  let r₂ : ℝ := (v₂ * 3 / (4 * π))^(1/3)
  let d₂ : ℝ := 2 * r₂
  d₂ = 18 * (12 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l3372_337223


namespace NUMINAMATH_CALUDE_sandwich_cost_is_30_cents_l3372_337237

/-- The cost of a sandwich given the cost of juice, total money, and number of people. -/
def sandwich_cost (juice_cost total_money : ℚ) (num_people : ℕ) : ℚ :=
  (total_money - num_people * juice_cost) / num_people

/-- Theorem stating that the cost of a sandwich is $0.30 under given conditions. -/
theorem sandwich_cost_is_30_cents :
  sandwich_cost 0.2 2.5 5 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_30_cents_l3372_337237


namespace NUMINAMATH_CALUDE_f_less_than_neg_two_f_two_zeros_iff_l3372_337264

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - exp (x - a) + a

theorem f_less_than_neg_two (x : ℝ) (h : x > 0) : f 0 x < -2 := by
  sorry

theorem f_two_zeros_iff (a : ℝ) :
  (∃ x y, x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_less_than_neg_two_f_two_zeros_iff_l3372_337264


namespace NUMINAMATH_CALUDE_caleb_dandelion_puffs_l3372_337210

/-- Represents the problem of Caleb's dandelion puffs distribution --/
def dandelion_puffs_problem (total : ℕ) (sister grandmother dog : ℕ) (friends num_per_friend : ℕ) : Prop :=
  ∃ (mom : ℕ),
    total = mom + sister + grandmother + dog + (friends * num_per_friend) ∧
    total = 40 ∧
    sister = 3 ∧
    grandmother = 5 ∧
    dog = 2 ∧
    friends = 3 ∧
    num_per_friend = 9

/-- The solution to Caleb's dandelion puffs problem --/
theorem caleb_dandelion_puffs :
  dandelion_puffs_problem 40 3 5 2 3 9 → ∃ (mom : ℕ), mom = 3 := by
  sorry

end NUMINAMATH_CALUDE_caleb_dandelion_puffs_l3372_337210


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3372_337268

theorem complex_magnitude_problem (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (1 + 2*i) / i → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3372_337268


namespace NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l3372_337216

/-- The length of a spiral staircase handrail -/
theorem spiral_staircase_handrail_length 
  (turn_angle : Real) 
  (rise : Real) 
  (radius : Real) 
  (handrail_length : Real) : 
  turn_angle = 315 ∧ 
  rise = 12 ∧ 
  radius = 4 → 
  abs (handrail_length - Real.sqrt (rise^2 + (turn_angle / 360 * 2 * Real.pi * radius)^2)) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l3372_337216


namespace NUMINAMATH_CALUDE_apartment_211_location_l3372_337239

/-- Represents a building with apartments -/
structure Building where
  total_floors : ℕ
  shop_floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the floor and entrance of an apartment in a building -/
def apartment_location (b : Building) (apartment_number : ℕ) : ℕ × ℕ :=
  let residential_floors := b.total_floors - b.shop_floors
  let apartments_per_entrance := residential_floors * b.apartments_per_floor
  let entrance := ((apartment_number - 1) / apartments_per_entrance) + 1
  let position_in_entrance := (apartment_number - 1) % apartments_per_entrance + 1
  let floor := ((position_in_entrance - 1) / b.apartments_per_floor) + b.shop_floors + 1
  (floor, entrance)

theorem apartment_211_location :
  let b := Building.mk 9 1 6
  apartment_location b 211 = (5, 5) := by sorry

end NUMINAMATH_CALUDE_apartment_211_location_l3372_337239


namespace NUMINAMATH_CALUDE_isabel_cupcakes_l3372_337295

theorem isabel_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) :
  todd_ate = 21 →
  packages = 6 →
  cupcakes_per_package = 3 →
  todd_ate + packages * cupcakes_per_package = 39 := by
  sorry

end NUMINAMATH_CALUDE_isabel_cupcakes_l3372_337295


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3372_337283

theorem quadratic_inequality_roots (a : ℝ) :
  (∀ x, x < -4 ∨ x > 5 → x^2 + a*x + 20 > 0) →
  (∀ x, -4 ≤ x ∧ x ≤ 5 → x^2 + a*x + 20 ≤ 0) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3372_337283


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3372_337274

/-- Given a line with equation x*sin(α) - √3*y + 1 = 0, 
    the range of its inclination angle θ is [0, π/6] ∪ [5π/6, π) -/
theorem inclination_angle_range (α : Real) :
  let line := {(x, y) : Real × Real | x * Real.sin α - Real.sqrt 3 * y + 1 = 0}
  let θ := Real.arctan ((Real.sin α) / Real.sqrt 3)
  θ ∈ Set.union (Set.Icc 0 (π / 6)) (Set.Ico (5 * π / 6) π) := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3372_337274


namespace NUMINAMATH_CALUDE_expression_simplification_l3372_337247

theorem expression_simplification (x y : ℝ) : 
  2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3372_337247


namespace NUMINAMATH_CALUDE_class_size_l3372_337213

/-- The number of girls in Tom's class -/
def girls : ℕ := 22

/-- The difference between the number of girls and boys in Tom's class -/
def difference : ℕ := 3

/-- The total number of students in Tom's class -/
def total_students : ℕ := girls + (girls - difference)

theorem class_size : total_students = 41 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3372_337213


namespace NUMINAMATH_CALUDE_vertex_when_m_3_n_values_max_3_m_range_two_points_l3372_337286

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + m - 1

-- Theorem 1: Vertex when m = 3
theorem vertex_when_m_3 :
  let m := 3
  ∃ (x y : ℝ), x = 2 ∧ y = 6 ∧ 
    ∀ (t : ℝ), f m t ≤ f m x :=
sorry

-- Theorem 2: Values of n when maximum is 3
theorem n_values_max_3 :
  let m := 3
  ∀ (n : ℝ), (∀ (x : ℝ), n ≤ x ∧ x ≤ n + 2 → f m x ≤ 3) ∧
             (∃ (x : ℝ), n ≤ x ∧ x ≤ n + 2 ∧ f m x = 3) →
    n = 2 + Real.sqrt 3 ∨ n = -Real.sqrt 3 :=
sorry

-- Theorem 3: Range of m for exactly two points 3 units from x-axis
theorem m_range_two_points :
  ∀ (m : ℝ), (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (f m x₁ = 3 ∨ f m x₁ = -3) ∧ (f m x₂ = 3 ∨ f m x₂ = -3)) ↔
    -6 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_vertex_when_m_3_n_values_max_3_m_range_two_points_l3372_337286


namespace NUMINAMATH_CALUDE_jeff_journey_distance_l3372_337287

-- Define the journey segments
def segment1_speed : ℝ := 80
def segment1_time : ℝ := 6
def segment2_speed : ℝ := 60
def segment2_time : ℝ := 4
def segment3_speed : ℝ := 40
def segment3_time : ℝ := 2

-- Define the total distance function
def total_distance : ℝ := 
  segment1_speed * segment1_time + 
  segment2_speed * segment2_time + 
  segment3_speed * segment3_time

-- Theorem statement
theorem jeff_journey_distance : total_distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_jeff_journey_distance_l3372_337287


namespace NUMINAMATH_CALUDE_right_triangle_among_given_sets_l3372_337221

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem statement
theorem right_triangle_among_given_sets :
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) ∧
  ¬(is_right_triangle 5 (-11) 12) ∧
  is_right_triangle 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_among_given_sets_l3372_337221


namespace NUMINAMATH_CALUDE_inventory_problem_l3372_337266

theorem inventory_problem (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  speedsters = (3 * total) / 4 →
  convertibles = (3 * speedsters) / 5 →
  convertibles = 54 →
  total - speedsters = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_inventory_problem_l3372_337266


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3372_337208

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the non-parallel relation for lines and planes
variable (not_parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α : Plane) :
  parallel_line m n → 
  not_parallel_line_plane n α → 
  not_parallel_line_plane m α → 
  parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3372_337208


namespace NUMINAMATH_CALUDE_rhinoceros_preserve_watering_area_l3372_337254

theorem rhinoceros_preserve_watering_area 
  (initial_population : ℕ)
  (grazing_area_per_rhino : ℕ)
  (population_increase_percent : ℚ)
  (total_preserve_area : ℕ) :
  initial_population = 8000 →
  grazing_area_per_rhino = 100 →
  population_increase_percent = 1/10 →
  total_preserve_area = 890000 →
  let increased_population := initial_population + (initial_population * population_increase_percent).floor
  let total_grazing_area := increased_population * grazing_area_per_rhino
  let watering_area := total_preserve_area - total_grazing_area
  watering_area = 10000 := by
sorry

end NUMINAMATH_CALUDE_rhinoceros_preserve_watering_area_l3372_337254


namespace NUMINAMATH_CALUDE_inverse_function_inequality_l3372_337222

/-- A function satisfying f(x₁x₂) = f(x₁) + f(x₂) for positive x₁ and x₂ -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

/-- Theorem statement for the given problem -/
theorem inverse_function_inequality (f : ℝ → ℝ) (hf : FunctionalEquation f)
    (hfinv : ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f) :
    ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
      f⁻¹ x₁ + f⁻¹ x₂ ≥ 2 * (f⁻¹ (x₁ / 2) * f⁻¹ (x₂ / 2)) :=
  sorry

end NUMINAMATH_CALUDE_inverse_function_inequality_l3372_337222


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3372_337233

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ((-4 ≤ x ∧ x ≤ -1) ∨ (3 ≤ x ∧ x ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3372_337233


namespace NUMINAMATH_CALUDE_line_and_circle_equations_l3372_337278

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space represented by the equation (x-h)² + (y-k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Given two points, determine if a line passes through the first point and is perpendicular to the line connecting the two points -/
def isPerpendicular (p1 p2 : Point) (l : Line) : Prop :=
  -- Line passes through p1
  l.a * p1.x + l.b * p1.y + l.c = 0 ∧
  -- Line is perpendicular to the line connecting p1 and p2
  l.a * (p2.x - p1.x) + l.b * (p2.y - p1.y) = 0

/-- Given two points, determine if a circle has these points as the endpoints of its diameter -/
def isDiameter (p1 p2 : Point) (c : Circle) : Prop :=
  -- Center of the circle is the midpoint of p1 and p2
  c.h = (p1.x + p2.x) / 2 ∧
  c.k = (p1.y + p2.y) / 2 ∧
  -- Radius of the circle is half the distance between p1 and p2
  c.r^2 = ((p2.x - p1.x)^2 + (p2.y - p1.y)^2) / 4

theorem line_and_circle_equations (A B : Point) (l : Line) (C : Circle)
    (hA : A.x = -3 ∧ A.y = -1)
    (hB : B.x = 5 ∧ B.y = 5)
    (hl : l.a = 4 ∧ l.b = 3 ∧ l.c = 15)
    (hC : C.h = 1 ∧ C.k = 2 ∧ C.r = 5) :
    isPerpendicular A B l ∧ isDiameter A B C := by
  sorry

end NUMINAMATH_CALUDE_line_and_circle_equations_l3372_337278


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3372_337238

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h24 : hour < 24
  m60 : minute < 60

/-- Calculates the duration between two times in minutes -/
def duration (t1 t2 : Time) : Nat :=
  (t2.hour - t1.hour) * 60 + (t2.minute - t1.minute)

theorem flight_duration_sum (departure arrival : Time) 
    (h m : Nat) (m_pos : 0 < m) (m_lt_60 : m < 60) :
  departure.hour = 15 ∧ departure.minute = 42 →
  arrival.hour = 18 ∧ arrival.minute = 57 →
  duration departure arrival = h * 60 + m →
  h + m = 18 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l3372_337238


namespace NUMINAMATH_CALUDE_not_order_preserving_isomorphic_Z_Q_l3372_337276

theorem not_order_preserving_isomorphic_Z_Q :
  ¬∃ f : ℤ → ℚ, (∀ q : ℚ, ∃ z : ℤ, f z = q) ∧
    (∀ z₁ z₂ : ℤ, z₁ < z₂ → f z₁ < f z₂) := by
  sorry

end NUMINAMATH_CALUDE_not_order_preserving_isomorphic_Z_Q_l3372_337276


namespace NUMINAMATH_CALUDE_sum_of_ratios_geq_six_l3372_337220

theorem sum_of_ratios_geq_six {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_geq_six_l3372_337220


namespace NUMINAMATH_CALUDE_oliver_earnings_l3372_337298

def laundry_price : ℝ := 2
def day1_laundry : ℝ := 5
def day2_laundry : ℝ := day1_laundry + 5
def day3_laundry : ℝ := 2 * day2_laundry

def total_earnings : ℝ := laundry_price * (day1_laundry + day2_laundry + day3_laundry)

theorem oliver_earnings : total_earnings = 70 := by
  sorry

end NUMINAMATH_CALUDE_oliver_earnings_l3372_337298


namespace NUMINAMATH_CALUDE_expansion_without_x2_x3_terms_l3372_337229

theorem expansion_without_x2_x3_terms (m n : ℝ) : 
  (∀ x, (x^2 + m*x + 1) * (x^2 - 2*x + n) = x^4 + (m*n - 2)*x + n) → 
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_expansion_without_x2_x3_terms_l3372_337229


namespace NUMINAMATH_CALUDE_volunteer_arrangements_l3372_337204

/-- The number of ways to arrange n people among k exits, with each exit having at least one person. -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of permutations of r items chosen from n items. -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

theorem volunteer_arrangements :
  arrangements 5 4 = choose 5 2 * permutations 3 3 ∧ 
  arrangements 5 4 = 240 := by sorry

end NUMINAMATH_CALUDE_volunteer_arrangements_l3372_337204


namespace NUMINAMATH_CALUDE_fixed_points_equality_implies_a_bound_l3372_337236

/-- Given a function f(x) = x^2 - 2x + a, if the set of fixed points of f is equal to the set of fixed points of f ∘ f, then a is greater than or equal to 5/4. -/
theorem fixed_points_equality_implies_a_bound (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + a
  ({x : ℝ | f x = x} = {x : ℝ | f (f x) = x}) →
  a ≥ 5/4 := by
sorry

end NUMINAMATH_CALUDE_fixed_points_equality_implies_a_bound_l3372_337236


namespace NUMINAMATH_CALUDE_disjunction_truth_l3372_337277

theorem disjunction_truth (p q : Prop) : (p ∨ q) → (p ∨ q) :=
  sorry

end NUMINAMATH_CALUDE_disjunction_truth_l3372_337277


namespace NUMINAMATH_CALUDE_congruentAngles_equivalence_sameRemainder_equivalence_l3372_337209

-- Define a type for angles
structure Angle where
  measure : ℝ

-- Define congruence relation for angles
def congruentAngles (a b : Angle) : Prop := a.measure = b.measure

-- Define a type for integers with a specific modulus
structure ModInt (m : ℕ) where
  value : ℤ

-- Define same remainder relation for ModInt
def sameRemainder {m : ℕ} (a b : ModInt m) : Prop := a.value % m = b.value % m

-- Theorem: Congruence of angles is an equivalence relation
theorem congruentAngles_equivalence : Equivalence congruentAngles := by sorry

-- Theorem: Same remainder when divided by a certain number is an equivalence relation
theorem sameRemainder_equivalence (m : ℕ) : Equivalence (@sameRemainder m) := by sorry

end NUMINAMATH_CALUDE_congruentAngles_equivalence_sameRemainder_equivalence_l3372_337209


namespace NUMINAMATH_CALUDE_other_number_proof_l3372_337253

theorem other_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 14)
  (lcm : Nat.lcm a b = 396)
  (ha : a = 36) :
  b = 66 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3372_337253


namespace NUMINAMATH_CALUDE_diamond_value_l3372_337225

/-- Given that ◇5₉ = ◇3₁₁ where ◇ represents a digit, prove that ◇ = 1 -/
theorem diamond_value : ∃ (d : ℕ), d < 10 ∧ d * 9 + 5 = d * 11 + 3 ∧ d = 1 := by sorry

end NUMINAMATH_CALUDE_diamond_value_l3372_337225


namespace NUMINAMATH_CALUDE_ackermann_2_1_l3372_337224

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_2_1 : A 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ackermann_2_1_l3372_337224


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_digits_l3372_337228

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

/-- Extracts the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List ℕ) : Prop :=
  l.Nodup

theorem smallest_four_digit_divisible_by_digits :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n < 10000 ∧
    (∀ d ∈ digits n, d ≠ 0 → isDivisibleBy n d) ∧
    allDistinct (digits n) ∧
    (∀ m, 1000 ≤ m ∧ m < n →
      ¬(∀ d ∈ digits m, d ≠ 0 → isDivisibleBy m d) ∨
      ¬(allDistinct (digits m))) ∧
    n = 1236 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_digits_l3372_337228


namespace NUMINAMATH_CALUDE_largest_product_of_three_l3372_337219

def S : Finset Int := {-4, -3, -1, 5, 6}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → x * y * z ≤ 72) ∧
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 72) :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l3372_337219


namespace NUMINAMATH_CALUDE_center_line_correct_l3372_337284

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : PolarPoint → Prop

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : PolarPoint → Prop

/-- The given circle equation -/
def givenCircle : PolarCircle :=
  { equation := fun p => p.r = 4 * Real.cos p.θ + 6 * Real.sin p.θ }

/-- The line passing through the center of the circle and parallel to the polar axis -/
def centerLine : PolarLine :=
  { equation := fun p => p.r * Real.sin p.θ = 3 }

/-- Theorem stating that the centerLine is correct for the givenCircle -/
theorem center_line_correct (p : PolarPoint) : 
  givenCircle.equation p → centerLine.equation p := by
  sorry

end NUMINAMATH_CALUDE_center_line_correct_l3372_337284


namespace NUMINAMATH_CALUDE_solve_equation_l3372_337282

theorem solve_equation : ∃ y : ℝ, 4 * y + 6 * y = 450 - 10 * (y - 5) ∧ y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3372_337282


namespace NUMINAMATH_CALUDE_min_value_at_six_l3372_337203

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

/-- Theorem stating that f(x) has a minimum value when x = 6 -/
theorem min_value_at_six :
  ∀ x : ℝ, f x ≥ f 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_six_l3372_337203


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3372_337252

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : a * b < 0) :
  a - b = 14 ∨ a - b = -14 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3372_337252


namespace NUMINAMATH_CALUDE_average_daily_sales_l3372_337250

/-- Given the sales of pens over a 13-day period, calculate the average daily sales. -/
theorem average_daily_sales (day1_sales : ℕ) (other_days_sales : ℕ) (num_other_days : ℕ) : 
  day1_sales = 96 →
  other_days_sales = 44 →
  num_other_days = 12 →
  (day1_sales + num_other_days * other_days_sales) / (num_other_days + 1) = 48 :=
by
  sorry

#check average_daily_sales

end NUMINAMATH_CALUDE_average_daily_sales_l3372_337250


namespace NUMINAMATH_CALUDE_base_number_proof_l3372_337201

theorem base_number_proof (k : ℕ) (x : ℤ) 
  (h1 : 21^k ∣ 435961)
  (h2 : x^k - k^7 = 1) :
  x = 2 :=
sorry

end NUMINAMATH_CALUDE_base_number_proof_l3372_337201


namespace NUMINAMATH_CALUDE_rosie_pies_l3372_337234

/-- Calculates the number of pies Rosie can make given the available apples and pears. -/
def calculate_pies (apples_per_3_pies : ℕ) (pears_per_3_pies : ℕ) (available_apples : ℕ) (available_pears : ℕ) : ℕ :=
  min (available_apples * 3 / apples_per_3_pies) (available_pears * 3 / pears_per_3_pies)

/-- Proves that Rosie can make 9 pies with 36 apples and 18 pears, given that she can make 3 pies out of 12 apples and 6 pears. -/
theorem rosie_pies : calculate_pies 12 6 36 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l3372_337234
