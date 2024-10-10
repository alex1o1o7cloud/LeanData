import Mathlib

namespace sqrt_four_squared_l2242_224223

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end sqrt_four_squared_l2242_224223


namespace probability_is_two_over_155_l2242_224278

/-- Represents a 5x5x5 cube with two adjacent faces painted red -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def count_three_painted_faces (cube : PaintedCube) : Nat :=
  1

/-- Calculates the number of unit cubes with no painted faces -/
def count_unpainted_faces (cube : PaintedCube) : Nat :=
  cube.size^3 - (cube.size^2 * 2 - cube.size)

/-- Calculates the total number of ways to choose two unit cubes -/
def total_combinations (cube : PaintedCube) : Nat :=
  (cube.size^3 * (cube.size^3 - 1)) / 2

/-- Calculates the probability of selecting one cube with three painted faces
    and one cube with no painted faces -/
def probability_three_and_none (cube : PaintedCube) : Rat :=
  (count_three_painted_faces cube * count_unpainted_faces cube) / total_combinations cube

/-- The main theorem stating the probability is 2/155 -/
theorem probability_is_two_over_155 (cube : PaintedCube) 
  (h1 : cube.size = 5) 
  (h2 : cube.painted_faces = 2) :
  probability_three_and_none cube = 2 / 155 := by
  sorry

end probability_is_two_over_155_l2242_224278


namespace c_value_is_198_l2242_224288

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for c
def c_equation (a b : ℕ+) : ℂ := (a + b * i)^3 - 107 * i

-- State the theorem
theorem c_value_is_198 :
  ∀ a b c : ℕ+, c_equation a b = c → c = 198 := by
  sorry

end c_value_is_198_l2242_224288


namespace soccer_ball_cost_l2242_224245

theorem soccer_ball_cost (football_cost soccer_cost : ℚ) : 
  (3 * football_cost + soccer_cost = 155) →
  (2 * football_cost + 3 * soccer_cost = 220) →
  soccer_cost = 50 := by
sorry

end soccer_ball_cost_l2242_224245


namespace total_campers_rowing_l2242_224206

theorem total_campers_rowing (morning afternoon evening : ℕ) 
  (h1 : morning = 36) 
  (h2 : afternoon = 13) 
  (h3 : evening = 49) : 
  morning + afternoon + evening = 98 := by
  sorry

end total_campers_rowing_l2242_224206


namespace four_number_inequality_equality_condition_l2242_224295

theorem four_number_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a - b) * (a - c)) / (a + b + c) +
  ((b - c) * (b - d)) / (b + c + d) +
  ((c - d) * (c - a)) / (c + d + a) +
  ((d - a) * (d - b)) / (d + a + b) ≥ 0 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a - b) * (a - c)) / (a + b + c) +
  ((b - c) * (b - d)) / (b + c + d) +
  ((c - d) * (c - a)) / (c + d + a) +
  ((d - a) * (d - b)) / (d + a + b) = 0 ↔
  a = b ∧ b = c ∧ c = d :=
by sorry

end four_number_inequality_equality_condition_l2242_224295


namespace prob_end_after_two_draw_prob_exactly_two_white_l2242_224244

/-- Represents the color of a ping-pong ball -/
inductive BallColor
  | Red
  | White
  | Blue

/-- Represents the box of ping-pong balls -/
structure Box where
  total : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- The probability of drawing a specific color ball from the box -/
def drawProbability (box : Box) (color : BallColor) : Rat :=
  match color with
  | BallColor.Red => box.red / box.total
  | BallColor.White => box.white / box.total
  | BallColor.Blue => box.blue / box.total

/-- The box configuration as per the problem -/
def problemBox : Box := {
  total := 10
  red := 5
  white := 3
  blue := 2
}

/-- The probability of the process ending after two draws -/
def probEndAfterTwoDraw (box : Box) : Rat :=
  (1 - drawProbability box BallColor.Blue) * drawProbability box BallColor.Blue

/-- The probability of exactly drawing 2 white balls -/
def probExactlyTwoWhite (box : Box) : Rat :=
  3 * drawProbability box BallColor.Red * (drawProbability box BallColor.White)^2 +
  drawProbability box BallColor.White * drawProbability box BallColor.White * drawProbability box BallColor.Blue

theorem prob_end_after_two_draw :
  probEndAfterTwoDraw problemBox = 4 / 25 := by sorry

theorem prob_exactly_two_white :
  probExactlyTwoWhite problemBox = 153 / 1000 := by sorry

end prob_end_after_two_draw_prob_exactly_two_white_l2242_224244


namespace second_polygon_sides_l2242_224214

/-- 
Given two regular polygons with the same perimeter, where the first has 50 sides 
and a side length three times as long as the second, prove that the second polygon has 150 sides.
-/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 → 
  50 * (3 * s) = n * s → 
  n = 150 := by
  sorry

end second_polygon_sides_l2242_224214


namespace emilio_gifts_l2242_224202

theorem emilio_gifts (total gifts_from_jorge gifts_from_pedro : ℕ) 
  (h1 : gifts_from_jorge = 6)
  (h2 : gifts_from_pedro = 4)
  (h3 : total = 21) :
  total - gifts_from_jorge - gifts_from_pedro = 11 := by
  sorry

end emilio_gifts_l2242_224202


namespace unique_solution_condition_l2242_224271

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := by
  sorry

end unique_solution_condition_l2242_224271


namespace inequality_solution_set_l2242_224230

-- Define the inequality
def inequality (x : ℝ) : Prop := abs ((2 - x) / x) > (x - 2) / x

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l2242_224230


namespace shirt_cost_l2242_224235

theorem shirt_cost (J S X : ℝ) 
  (eq1 : 3 * J + 2 * S = X)
  (eq2 : 2 * J + 3 * S = 66)
  (eq3 : 3 * J + 2 * S = 2 * J + 3 * S) : 
  S = 13.20 := by
  sorry

end shirt_cost_l2242_224235


namespace amandas_car_round_trip_time_l2242_224239

/-- Given that:
    1. The bus takes 40 minutes to drive 80 miles to the beach.
    2. Amanda's car takes five fewer minutes than the bus for the same trip.
    Prove that Amanda's car takes 70 minutes to make a round trip to the beach. -/
theorem amandas_car_round_trip_time :
  let bus_time : ℕ := 40
  let car_time_difference : ℕ := 5
  let car_one_way_time : ℕ := bus_time - car_time_difference
  car_one_way_time * 2 = 70 := by sorry

end amandas_car_round_trip_time_l2242_224239


namespace simplify_fraction_l2242_224281

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 5) :
  15 * x^3 * y^2 / (10 * x^2 * y^4) = 3 / 25 := by
  sorry

end simplify_fraction_l2242_224281


namespace noah_sticker_count_l2242_224250

/-- Given the number of stickers for Kristoff, calculate the number of stickers Noah has -/
def noahs_stickers (kristoff : ℕ) : ℕ :=
  let riku : ℕ := 25 * kristoff
  let lila : ℕ := 2 * (kristoff + riku)
  kristoff * lila - 3

theorem noah_sticker_count : noahs_stickers 85 = 375697 := by
  sorry

end noah_sticker_count_l2242_224250


namespace subtraction_addition_equality_l2242_224201

theorem subtraction_addition_equality : -32 - (-14) + 4 = -14 := by sorry

end subtraction_addition_equality_l2242_224201


namespace inequality_equivalence_l2242_224226

theorem inequality_equivalence (θ : ℝ) (x : ℝ) :
  (|x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) ↔ (-1 ≤ x ∧ x ≤ -Real.cos (2 * θ)) :=
by sorry

end inequality_equivalence_l2242_224226


namespace prob_two_even_out_of_six_l2242_224256

/-- The probability of rolling an even number on a fair six-sided die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number on a fair six-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice showing even numbers -/
def num_even : ℕ := 2

/-- The number of ways to choose 2 dice out of 6 -/
def ways_to_choose : ℕ := Nat.choose num_dice num_even

theorem prob_two_even_out_of_six :
  (ways_to_choose : ℚ) * prob_even^num_even * prob_odd^(num_dice - num_even) = 15/64 := by
  sorry

end prob_two_even_out_of_six_l2242_224256


namespace quadratic_solution_unique_l2242_224251

theorem quadratic_solution_unique (x : ℝ) :
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0 → x = 4/3 :=
by sorry

end quadratic_solution_unique_l2242_224251


namespace fraction_meaningful_l2242_224285

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 4 / (x - 3)) ↔ x ≠ 3 := by
  sorry

end fraction_meaningful_l2242_224285


namespace starting_player_winning_strategy_l2242_224265

/-- Represents the color of a disk -/
inductive DiskColor
| Red
| Blue

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the state of the chessboard -/
def BoardState (n : Nat) := Position → DiskColor

/-- Checks if a position is within the bounds of the board -/
def isValidPosition (n : Nat) (pos : Position) : Prop :=
  pos.row < n ∧ pos.col < n

/-- Represents a move in the game -/
structure Move :=
  (pos : Position)

/-- Applies a move to the board state -/
def applyMove (n : Nat) (state : BoardState n) (move : Move) : BoardState n :=
  sorry

/-- Checks if a player can make a move -/
def canMove (n : Nat) (state : BoardState n) : Prop :=
  ∃ (move : Move), isValidPosition n move.pos ∧ state move.pos = DiskColor.Blue

/-- Defines a winning strategy for the starting player -/
def hasWinningStrategy (n : Nat) (initialState : BoardState n) : Prop :=
  sorry

/-- The main theorem stating the winning condition for the starting player -/
theorem starting_player_winning_strategy (n : Nat) (initialState : BoardState n) :
  hasWinningStrategy n initialState ↔ 
  initialState ⟨n - 1, n - 1⟩ = DiskColor.Blue :=
sorry

end starting_player_winning_strategy_l2242_224265


namespace topsoil_cost_calculation_l2242_224220

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yard_to_cubic_foot : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def topsoil_volume_cubic_yards : ℝ := 8

/-- Calculate the cost of topsoil given its volume in cubic yards -/
def topsoil_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yard_to_cubic_foot * topsoil_cost_per_cubic_foot

theorem topsoil_cost_calculation :
  topsoil_cost topsoil_volume_cubic_yards = 1728 := by
  sorry

end topsoil_cost_calculation_l2242_224220


namespace rain_probability_l2242_224257

/-- The probability of rain on both Monday and Tuesday given specific conditions -/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (p_tuesday_given_no_monday : ℝ) :
  p_monday = 0.4 →
  p_tuesday = 0.3 →
  p_tuesday_given_no_monday = 0.5 →
  p_monday * p_tuesday = 0.12 :=
by sorry

end rain_probability_l2242_224257


namespace equal_volumes_condition_l2242_224227

/-- Represents the side lengths of the square prisms to be removed from a cube --/
structure PrismSides where
  c : ℝ
  b : ℝ
  a : ℝ

/-- Calculates the volume of the remaining body after removing square prisms --/
def remainingVolume (sides : PrismSides) : ℝ :=
  1 - (sides.c^2 + (sides.b^2 - sides.c^2 * sides.b) + (sides.a^2 - sides.c^2 * sides.a - sides.b^2 * sides.a + sides.c^2 * sides.b))

/-- Theorem stating the conditions for equal volumes --/
theorem equal_volumes_condition (sides : PrismSides) : 
  sides.c = 1/2 ∧ 
  sides.b = (1 + Real.sqrt 17) / 8 ∧ 
  sides.a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64 ∧
  sides.c < sides.b ∧ sides.b < sides.a ∧ sides.a < 1 →
  remainingVolume sides = 1/4 :=
sorry

end equal_volumes_condition_l2242_224227


namespace karen_start_time_l2242_224211

/-- Proves that Karen starts 4 minutes late in the car race --/
theorem karen_start_time (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_win_margin : ℝ) 
  (h1 : karen_speed = 60)
  (h2 : tom_speed = 45)
  (h3 : tom_distance = 24)
  (h4 : karen_win_margin = 4) :
  (tom_distance / tom_speed - (tom_distance + karen_win_margin) / karen_speed) * 60 = 4 := by
  sorry

end karen_start_time_l2242_224211


namespace proposition_p_false_implies_a_range_l2242_224236

theorem proposition_p_false_implies_a_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) → 
  (a < 0 ∨ a > 4) := by
sorry

end proposition_p_false_implies_a_range_l2242_224236


namespace ab_greater_ac_l2242_224277

theorem ab_greater_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end ab_greater_ac_l2242_224277


namespace union_A_B_l2242_224238

def A : Set ℤ := {0, 1}

def B : Set ℤ := {x | (x + 2) * (x - 1) < 0}

theorem union_A_B : A ∪ B = {-1, 0, 1} := by sorry

end union_A_B_l2242_224238


namespace long_division_problem_l2242_224294

theorem long_division_problem :
  let divisor : ℕ := 12
  let quotient : ℕ := 909809
  let dividend : ℕ := divisor * quotient
  dividend = 10917708 := by
sorry

end long_division_problem_l2242_224294


namespace no_solution_to_double_inequality_l2242_224221

theorem no_solution_to_double_inequality :
  ¬∃ (x : ℝ), (3 * x + 2 < (x + 2)^2) ∧ ((x + 2)^2 < 5 * x + 1) := by
  sorry

end no_solution_to_double_inequality_l2242_224221


namespace soda_crates_count_l2242_224263

def bridge_weight_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def num_dryers : ℕ := 3
def dryer_weight : ℕ := 3000
def loaded_truck_weight : ℕ := 24000

def calculate_soda_crates (bridge_weight_limit empty_truck_weight soda_crate_weight 
                           num_dryers dryer_weight loaded_truck_weight : ℕ) : ℕ := 
  let total_dryer_weight := num_dryers * dryer_weight
  let remaining_weight := loaded_truck_weight - empty_truck_weight - total_dryer_weight
  let soda_weight := remaining_weight / 3
  soda_weight / soda_crate_weight

theorem soda_crates_count : 
  calculate_soda_crates bridge_weight_limit empty_truck_weight soda_crate_weight 
                         num_dryers dryer_weight loaded_truck_weight = 20 := by
  sorry

end soda_crates_count_l2242_224263


namespace product_equals_four_l2242_224229

theorem product_equals_four (a b c : ℝ) 
  (h : ∀ (a b c : ℝ), a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) : 
  6 * 15 * 2 = 4 := by
sorry

end product_equals_four_l2242_224229


namespace donut_selections_l2242_224268

theorem donut_selections (n k : ℕ) (hn : n = 5) (hk : k = 4) : 
  Nat.choose (n + k - 1) (k - 1) = 56 := by
  sorry

end donut_selections_l2242_224268


namespace first_occurrence_is_lcm_l2242_224207

/-- Represents the cycle length of letters -/
def letter_cycle : ℕ := 8

/-- Represents the cycle length of digits -/
def digit_cycle : ℕ := 4

/-- Represents the first occurrence of the original sequence -/
def first_occurrence : ℕ := 8

/-- Theorem stating that the first occurrence is the least common multiple of the cycle lengths -/
theorem first_occurrence_is_lcm :
  first_occurrence = Nat.lcm letter_cycle digit_cycle := by sorry

end first_occurrence_is_lcm_l2242_224207


namespace factor_calculation_l2242_224279

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 8 →
  factor * (2 * initial_number + 9) = 75 →
  factor = 3 := by
sorry

end factor_calculation_l2242_224279


namespace cassidy_grounding_period_l2242_224275

/-- Calculates the total grounding period based on initial days, extra days per grade below B, and number of grades below B. -/
def total_grounding_period (initial_days : ℕ) (extra_days_per_grade : ℕ) (grades_below_b : ℕ) : ℕ :=
  initial_days + extra_days_per_grade * grades_below_b

/-- Proves that given the specified conditions, the total grounding period is 26 days. -/
theorem cassidy_grounding_period :
  total_grounding_period 14 3 4 = 26 := by
  sorry

end cassidy_grounding_period_l2242_224275


namespace twins_age_problem_l2242_224200

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 9 → age = 4 := by
  sorry

end twins_age_problem_l2242_224200


namespace possible_x_value_for_simplest_radical_l2242_224284

/-- A number is a simplest quadratic radical if it's of the form √n where n is a positive integer
    and not a perfect square. -/
def is_simplest_quadratic_radical (n : ℝ) : Prop :=
  ∃ (m : ℕ), n = Real.sqrt m ∧ ¬ ∃ (k : ℕ), m = k^2

/-- The proposition states that 2 is a possible value for x that makes √(x+3) 
    the simplest quadratic radical. -/
theorem possible_x_value_for_simplest_radical : 
  ∃ (x : ℝ), is_simplest_quadratic_radical (Real.sqrt (x + 3)) ∧ x = 2 :=
sorry

end possible_x_value_for_simplest_radical_l2242_224284


namespace weights_missing_l2242_224292

/-- Represents a set of weights with equal quantities of each type -/
structure WeightSet where
  quantity : ℕ
  total_mass : ℕ

/-- The weight set described in the problem -/
def problem_weights : WeightSet :=
  { quantity := 0,  -- We don't know the exact quantity, so we use 0 as a placeholder
    total_mass := 606060606060 }  -- Assuming the pattern repeats 6 times for illustration

/-- Theorem stating that at least one weight is missing and more than 10 weights are missing -/
theorem weights_missing (w : WeightSet) :
  (w.total_mass % 72 ≠ 0) ∧ 
  (∃ (a b : ℕ), a + b > 10 ∧ 5*a + 43*b ≡ w.total_mass [MOD 24]) :=
by sorry

end weights_missing_l2242_224292


namespace circle_area_from_parallel_chords_l2242_224246

-- Define the circle C
def C : Real → Real → Prop := sorry

-- Define the two lines
def line1 (x y : Real) : Prop := x - y - 1 = 0
def line2 (x y : Real) : Prop := x - y - 5 = 0

-- Define the chord length
def chord_length : Real := 10

-- Theorem statement
theorem circle_area_from_parallel_chords 
  (h1 : ∃ (x1 y1 x2 y2 : Real), C x1 y1 ∧ C x2 y2 ∧ line1 x1 y1 ∧ line1 x2 y2)
  (h2 : ∃ (x3 y3 x4 y4 : Real), C x3 y3 ∧ C x4 y4 ∧ line2 x3 y3 ∧ line2 x4 y4)
  (h3 : ∀ (x1 y1 x2 y2 : Real), C x1 y1 ∧ C x2 y2 ∧ line1 x1 y1 ∧ line1 x2 y2 → 
        Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = chord_length)
  (h4 : ∀ (x3 y3 x4 y4 : Real), C x3 y3 ∧ C x4 y4 ∧ line2 x3 y3 ∧ line2 x4 y4 → 
        Real.sqrt ((x3 - x4)^2 + (y3 - y4)^2) = chord_length) :
  (∃ (r : Real), ∀ (x y : Real), C x y ↔ (x - 0)^2 + (y - 0)^2 = r^2) ∧ 
  (∃ (area : Real), area = 27 * Real.pi) :=
sorry

end circle_area_from_parallel_chords_l2242_224246


namespace b_5_times_b_9_equals_16_l2242_224298

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b_5_times_b_9_equals_16 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : 2 * a 2 - (a 7)^2 + 2 * a 12 = 0)
  (h3 : geometric_sequence b)
  (h4 : b 7 = a 7) :
  b 5 * b 9 = 16 := by
sorry

end b_5_times_b_9_equals_16_l2242_224298


namespace absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_proof_l2242_224237

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun d => ∃ x y : ℝ,
    (|x - 3| = 15 ∧ |y - 3| = 15) ∧
    x ≠ y ∧
    d = |x - y| ∧
    d = 30

-- The proof is omitted
theorem absolute_value_equation_solution_difference_proof :
  ∃ d : ℝ, absolute_value_equation_solution_difference d :=
by
  sorry

end absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_proof_l2242_224237


namespace gravitational_force_calculation_l2242_224243

/-- Gravitational force calculation -/
theorem gravitational_force_calculation
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances
  (f₁ : ℝ) -- Force at distance d₁
  (h₁ : d₁ = 8000)
  (h₂ : d₂ = 320000)
  (h₃ : f₁ = 150)
  (h₄ : k = f₁ * d₁^2) -- Inverse square law
  : ∃ f₂ : ℝ, f₂ = k / d₂^2 ∧ f₂ = 3 / 32 := by
  sorry

end gravitational_force_calculation_l2242_224243


namespace prob_five_odd_in_six_rolls_l2242_224283

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 6

/-- The number of times we want to get an odd number -/
def target_odd : ℕ := 5

/-- The probability of getting exactly 'target_odd' odd numbers in 'num_rolls' rolls -/
def prob_target_odd : ℚ :=
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd ^ target_odd * (1 - prob_odd) ^ (num_rolls - target_odd)

theorem prob_five_odd_in_six_rolls : prob_target_odd = 3/32 := by
  sorry

end prob_five_odd_in_six_rolls_l2242_224283


namespace class_average_problem_l2242_224204

theorem class_average_problem (percent_high : Real) (percent_mid : Real) (percent_low : Real)
  (score_high : Real) (score_low : Real) (overall_average : Real) :
  percent_high = 15 →
  percent_mid = 50 →
  percent_low = 35 →
  score_high = 100 →
  score_low = 63 →
  overall_average = 76.05 →
  (percent_high * score_high + percent_mid * ((percent_high * score_high + percent_mid * X + percent_low * score_low) / 100) + percent_low * score_low) / 100 = overall_average →
  X = 78 := by
  sorry

end class_average_problem_l2242_224204


namespace coordinates_wrt_x_axis_l2242_224273

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The coordinates of point P -/
def P : ℝ × ℝ := (-2, 3)

/-- Theorem: The coordinates of P(-2, 3) with respect to the x-axis are (-2, -3) -/
theorem coordinates_wrt_x_axis : reflect_x P = (-2, -3) := by
  sorry

end coordinates_wrt_x_axis_l2242_224273


namespace even_function_properties_l2242_224240

-- Define the function f
def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

-- State the theorem
theorem even_function_properties (m : ℝ) :
  (∀ x, f x m = f (-x) m) →
  (m = 1 ∧ (∀ x, f x m = 0 ↔ x = 1 ∨ x = -1)) :=
by sorry

end even_function_properties_l2242_224240


namespace garrett_roses_count_l2242_224258

/-- Mrs. Santiago's red roses -/
def santiago_roses : ℕ := 58

/-- The difference between Mrs. Santiago's and Mrs. Garrett's red roses -/
def rose_difference : ℕ := 34

/-- Mrs. Garrett's red roses -/
def garrett_roses : ℕ := santiago_roses - rose_difference

theorem garrett_roses_count : garrett_roses = 24 := by
  sorry

end garrett_roses_count_l2242_224258


namespace max_value_inequality_l2242_224242

theorem max_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 := by
  sorry

end max_value_inequality_l2242_224242


namespace tan_225_degrees_equals_one_l2242_224219

theorem tan_225_degrees_equals_one : Real.tan (225 * π / 180) = 1 := by
  sorry

end tan_225_degrees_equals_one_l2242_224219


namespace laptop_price_l2242_224205

theorem laptop_price : ∃ (S : ℝ), S = 733 ∧ 
  (0.80 * S - 120 = 0.65 * S - 10) := by
  sorry

end laptop_price_l2242_224205


namespace increase_averages_possible_l2242_224254

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem increase_averages_possible :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end increase_averages_possible_l2242_224254


namespace unique_congruence_solution_l2242_224217

theorem unique_congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 1000000 [ZMOD 9] := by
  sorry

end unique_congruence_solution_l2242_224217


namespace birds_remaining_count_l2242_224272

/-- The number of grey birds initially in the cage -/
def grey_birds : ℕ := 40

/-- The number of white birds next to the cage -/
def white_birds : ℕ := grey_birds + 6

/-- The number of grey birds remaining after half are freed -/
def remaining_grey_birds : ℕ := grey_birds / 2

/-- The total number of birds remaining after ten minutes -/
def total_remaining_birds : ℕ := remaining_grey_birds + white_birds

theorem birds_remaining_count : total_remaining_birds = 66 := by
  sorry

end birds_remaining_count_l2242_224272


namespace min_black_edges_four_black_edges_possible_l2242_224282

structure Cube :=
  (edges : Finset (Fin 12))
  (faces : Finset (Fin 6))
  (edge_coloring : Fin 12 → Bool)
  (face_edges : Fin 6 → Finset (Fin 12))
  (edge_faces : Fin 12 → Finset (Fin 2))

def is_valid_coloring (c : Cube) : Prop :=
  ∀ f : Fin 6, 
    (∃ e ∈ c.face_edges f, c.edge_coloring e = true) ∧ 
    (∃ e ∈ c.face_edges f, c.edge_coloring e = false)

def num_black_edges (c : Cube) : Nat :=
  (c.edges.filter (λ e => c.edge_coloring e = true)).card

theorem min_black_edges (c : Cube) :
  is_valid_coloring c → num_black_edges c ≥ 4 :=
sorry

theorem four_black_edges_possible : 
  ∃ c : Cube, is_valid_coloring c ∧ num_black_edges c = 4 :=
sorry

end min_black_edges_four_black_edges_possible_l2242_224282


namespace rectangular_field_area_l2242_224212

/-- A rectangular field with breadth 60% of length and perimeter 800 m has area 37500 m² -/
theorem rectangular_field_area (length breadth : ℝ) : 
  breadth = 0.6 * length →
  2 * (length + breadth) = 800 →
  length * breadth = 37500 := by
  sorry

end rectangular_field_area_l2242_224212


namespace car_profit_percentage_l2242_224299

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage on purchase and markup percentage on sale. -/
theorem car_profit_percentage
  (P : ℝ)                    -- Original price of the car
  (discount : ℝ)             -- Discount percentage on purchase
  (markup : ℝ)               -- Markup percentage on sale
  (h_discount : discount = 20)
  (h_markup : markup = 45)
  : (((1 - discount / 100) * (1 + markup / 100) - 1) * 100 = 16) := by
  sorry

end car_profit_percentage_l2242_224299


namespace slope_of_line_slope_to_angle_l2242_224233

/-- The slope of the line x + √3 * y - 1 = 0 is -√3/3 -/
theorem slope_of_line (x y : ℝ) : 
  (x + Real.sqrt 3 * y - 1 = 0) → (y = -(1 / Real.sqrt 3) * x + 1 / Real.sqrt 3) :=
by sorry

/-- The slope -√3/3 corresponds to an angle of 150° -/
theorem slope_to_angle (θ : ℝ) :
  Real.tan θ = -(Real.sqrt 3 / 3) → θ = 150 * (Real.pi / 180) :=
by sorry

end slope_of_line_slope_to_angle_l2242_224233


namespace jason_toys_count_l2242_224213

theorem jason_toys_count :
  ∀ (rachel_toys john_toys jason_toys : ℕ),
    rachel_toys = 1 →
    john_toys = rachel_toys + 6 →
    jason_toys = 3 * john_toys →
    jason_toys = 21 :=
by
  sorry

end jason_toys_count_l2242_224213


namespace discounted_price_calculation_l2242_224252

/-- Given a bag marked at $240 with a 50% discount, prove that the discounted price is $120. -/
theorem discounted_price_calculation (marked_price : ℝ) (discount_rate : ℝ) :
  marked_price = 240 →
  discount_rate = 0.5 →
  marked_price * (1 - discount_rate) = 120 := by
  sorry

end discounted_price_calculation_l2242_224252


namespace xyz_mod_nine_l2242_224231

theorem xyz_mod_nine (x y z : ℕ) : 
  x < 9 → y < 9 → z < 9 →
  (x + 3*y + 2*z) % 9 = 0 →
  (2*x + 2*y + z) % 9 = 7 →
  (x + 2*y + 3*z) % 9 = 5 →
  (x*y*z) % 9 = 5 := by
  sorry

end xyz_mod_nine_l2242_224231


namespace emily_team_score_l2242_224293

theorem emily_team_score (total_players : ℕ) (emily_score : ℕ) (other_player_score : ℕ) : 
  total_players = 8 →
  emily_score = 23 →
  other_player_score = 2 →
  emily_score + (total_players - 1) * other_player_score = 37 := by
  sorry

end emily_team_score_l2242_224293


namespace complex_equation_square_sum_l2242_224276

theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : 
  a^2 + b^2 = 5 := by sorry

end complex_equation_square_sum_l2242_224276


namespace decreasing_interval_of_f_l2242_224261

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 5 * Real.pi / 6 ∧ b = Real.pi ∧
  monotonically_decreasing f a b ∧
  ∀ c d, 0 ≤ c ∧ d ≤ Real.pi ∧ c < d ∧ monotonically_decreasing f c d →
    a ≤ c ∧ d ≤ b :=
by sorry

end decreasing_interval_of_f_l2242_224261


namespace min_young_rank_is_11_l2242_224297

/-- Yuna's rank in the running event -/
def yuna_rank : ℕ := 6

/-- The number of people who finished between Yuna and Min-Young -/
def people_between : ℕ := 5

/-- Min-Young's rank in the running event -/
def min_young_rank : ℕ := yuna_rank + people_between

/-- Theorem stating Min-Young's rank -/
theorem min_young_rank_is_11 : min_young_rank = 11 := by
  sorry

end min_young_rank_is_11_l2242_224297


namespace box_volume_l2242_224296

theorem box_volume (x y z : ℝ) 
  (h1 : 2*x + 2*y = 26) 
  (h2 : x + z = 10) 
  (h3 : y + z = 7) : 
  x * y * z = 80 := by
  sorry

end box_volume_l2242_224296


namespace june_bird_eggs_l2242_224255

/-- The number of eggs in each nest in the first tree -/
def eggs_per_nest_tree1 : ℕ := 5

/-- The number of nests in the first tree -/
def nests_in_tree1 : ℕ := 2

/-- The number of eggs in the nest in the second tree -/
def eggs_in_tree2 : ℕ := 3

/-- The number of eggs in the nest in the front yard -/
def eggs_in_front_yard : ℕ := 4

/-- The total number of bird eggs June found -/
def total_eggs : ℕ := nests_in_tree1 * eggs_per_nest_tree1 + eggs_in_tree2 + eggs_in_front_yard

theorem june_bird_eggs : total_eggs = 17 := by
  sorry

end june_bird_eggs_l2242_224255


namespace eating_relationship_l2242_224222

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x | a * x^2 = 1}

def full_eating (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_eating (X Y : Set ℝ) : Prop := 
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem eating_relationship (a : ℝ) : 
  (a ≥ 0) → (full_eating A (B a) ∨ partial_eating A (B a)) ↔ a ∈ ({0, 1, 4} : Set ℝ) :=
sorry

end eating_relationship_l2242_224222


namespace investment_difference_l2242_224280

def initial_investment : ℝ := 500

def jackson_growth_rate : ℝ := 4

def brandon_growth_rate : ℝ := 0.2

def jackson_final_value : ℝ := initial_investment * jackson_growth_rate

def brandon_final_value : ℝ := initial_investment * brandon_growth_rate

theorem investment_difference :
  jackson_final_value - brandon_final_value = 1900 := by sorry

end investment_difference_l2242_224280


namespace intersection_M_N_l2242_224264

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end intersection_M_N_l2242_224264


namespace average_of_six_numbers_l2242_224249

theorem average_of_six_numbers 
  (total_average : ℝ)
  (second_pair_average : ℝ)
  (third_pair_average : ℝ)
  (h1 : total_average = 6.40)
  (h2 : second_pair_average = 6.1)
  (h3 : third_pair_average = 6.9) :
  ∃ (first_pair_average : ℝ),
    first_pair_average = 6.2 ∧
    (first_pair_average + second_pair_average + third_pair_average) / 3 = total_average :=
by sorry

end average_of_six_numbers_l2242_224249


namespace meal_pass_cost_sally_meal_pass_cost_l2242_224228

/-- Calculates the cost of a meal pass for Sally's trip to Sea World --/
theorem meal_pass_cost (savings : ℕ) (parking : ℕ) (entrance : ℕ) (distance : ℕ) (mpg : ℕ) 
  (gas_price : ℕ) (additional_savings : ℕ) : ℕ :=
  let round_trip := 2 * distance
  let gas_needed := round_trip / mpg
  let gas_cost := gas_needed * gas_price
  let known_costs := parking + entrance + gas_cost
  let remaining_costs := known_costs - savings
  additional_savings - remaining_costs

/-- The meal pass for Sally's trip to Sea World costs $25 --/
theorem sally_meal_pass_cost : 
  meal_pass_cost 28 10 55 165 30 3 95 = 25 := by
  sorry

end meal_pass_cost_sally_meal_pass_cost_l2242_224228


namespace equation_solution_l2242_224209

theorem equation_solution (t : ℝ) : 
  (5 * 3^t + Real.sqrt (25 * 9^t) = 50) ↔ (t = Real.log 5 / Real.log 3) :=
by sorry

end equation_solution_l2242_224209


namespace min_points_for_12_monochromatic_triangles_l2242_224210

/-- A coloring of edges in a complete graph with two colors -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- The number of monochromatic triangles in a given coloring -/
def monochromaticTriangles (n : ℕ) (c : TwoColoring n) : ℕ := sorry

/-- The statement that for any two-coloring of Kn, there are at least 12 monochromatic triangles -/
def hasAtLeast12MonochromaticTriangles (n : ℕ) : Prop :=
  ∀ c : TwoColoring n, monochromaticTriangles n c ≥ 12

/-- The theorem stating that 9 is the minimum number of points satisfying the condition -/
theorem min_points_for_12_monochromatic_triangles :
  (hasAtLeast12MonochromaticTriangles 9) ∧ 
  (∀ m : ℕ, m < 9 → ¬(hasAtLeast12MonochromaticTriangles m)) :=
sorry

end min_points_for_12_monochromatic_triangles_l2242_224210


namespace youngest_son_cookies_value_l2242_224289

/-- The number of cookies in a box -/
def total_cookies : ℕ := 54

/-- The number of days the box lasts -/
def days : ℕ := 9

/-- The number of cookies the oldest son gets each day -/
def oldest_son_cookies : ℕ := 4

/-- The number of cookies the youngest son gets each day -/
def youngest_son_cookies : ℕ := (total_cookies - (oldest_son_cookies * days)) / days

theorem youngest_son_cookies_value : youngest_son_cookies = 2 := by
  sorry

end youngest_son_cookies_value_l2242_224289


namespace derivative_at_one_is_one_fourth_l2242_224224

open Real

-- Define the function f
noncomputable def f (f'1 : ℝ) : ℝ → ℝ := λ x => log x - 3 * f'1 * x

-- State the theorem
theorem derivative_at_one_is_one_fourth (f'1 : ℝ) :
  (∀ x > 0, f f'1 x = log x - 3 * f'1 * x) →
  deriv (f f'1) 1 = 1/4 := by sorry

end derivative_at_one_is_one_fourth_l2242_224224


namespace smallest_base_perfect_square_l2242_224234

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 3 → (∃ n : ℕ, 2 * b + 3 = n^2) → b ≥ 11 :=
sorry

end smallest_base_perfect_square_l2242_224234


namespace factorization_xy_minus_8y_l2242_224274

theorem factorization_xy_minus_8y (x y : ℝ) : x * y - 8 * y = y * (x - 8) := by
  sorry

end factorization_xy_minus_8y_l2242_224274


namespace parallelogram_area_l2242_224270

/-- The area of a parallelogram with given base, slant height, and angle --/
theorem parallelogram_area (base slant_height : ℝ) (angle : ℝ) : 
  base = 24 → slant_height = 26 → angle = 40 * π / 180 →
  abs (base * (slant_height * Real.cos angle) - 478) < 0.01 := by
  sorry

end parallelogram_area_l2242_224270


namespace range_of_x_range_of_m_l2242_224262

-- Problem 1
theorem range_of_x (x : ℝ) : (4*x - 3)^2 ≤ 1 → 1/2 ≤ x ∧ x ≤ 1 := by sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 4*x + m < 0 → x^2 - x - 2 > 0) → m ≥ 4 := by sorry

end range_of_x_range_of_m_l2242_224262


namespace range_of_a_l2242_224232

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y + 4 = 2*x*y → 
    x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) → 
  a ≤ 17/4 := by
sorry

end range_of_a_l2242_224232


namespace sqrt_3_simplest_l2242_224291

def is_simplest_sqrt (x : ℝ) (options : List ℝ) : Prop :=
  ∀ y ∈ options, (∃ z : ℝ, z ^ 2 = x) → (∃ w : ℝ, w ^ 2 = y) → x ≤ y

theorem sqrt_3_simplest : 
  is_simplest_sqrt 3 [0.1, 8, (abs a), 3] := by sorry

end sqrt_3_simplest_l2242_224291


namespace sector_area_l2242_224260

/-- Given a circular sector with arc length 3π and central angle 135°, prove its area is 6π. -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r * θ = 3 * Real.pi → 
  θ = 135 * Real.pi / 180 →
  (1 / 2) * r^2 * θ = 6 * Real.pi :=
by sorry

end sector_area_l2242_224260


namespace power_product_simplification_l2242_224218

theorem power_product_simplification (x y : ℝ) :
  (x^3 * y^2)^2 * (x / y^3) = x^7 * y := by sorry

end power_product_simplification_l2242_224218


namespace problem_statement_l2242_224259

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * (b^2 + 1) + b * (b + 2*a) = 40)
  (h2 : a * (b + 1) + b = 8) : 
  1/a^2 + 1/b^2 = 8 := by sorry

end problem_statement_l2242_224259


namespace bucket_water_calculation_l2242_224248

/-- Given an initial amount of water and an amount poured out, 
    calculate the remaining amount of water in the bucket. -/
def water_remaining (initial : ℝ) (poured_out : ℝ) : ℝ :=
  initial - poured_out

/-- Theorem stating that given 0.8 gallon initially and 0.2 gallon poured out,
    the remaining amount is 0.6 gallon. -/
theorem bucket_water_calculation :
  water_remaining 0.8 0.2 = 0.6 := by
  sorry

#eval water_remaining 0.8 0.2

end bucket_water_calculation_l2242_224248


namespace eldorado_license_plates_l2242_224225

/-- The number of letters in the alphabet used for license plates -/
def alphabet_size : ℕ := 26

/-- The number of digits used for license plates -/
def digit_size : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Eldorado -/
def total_license_plates : ℕ := alphabet_size ^ letter_positions * digit_size ^ digit_positions

theorem eldorado_license_plates :
  total_license_plates = 175760000 :=
by sorry

end eldorado_license_plates_l2242_224225


namespace rectangle_side_ratio_l2242_224241

theorem rectangle_side_ratio (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : (a / b) ^ 2 = 4 := by
  sorry

end rectangle_side_ratio_l2242_224241


namespace x_neg_one_is_local_minimum_l2242_224287

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem x_neg_one_is_local_minimum :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1 → |x - (-1)| < δ → f x ≥ f (-1) :=
sorry

end x_neg_one_is_local_minimum_l2242_224287


namespace max_value_of_function_l2242_224203

/-- The function f(x) = x^2(1-3x) has a maximum value of 1/12 in the interval (0, 1/3) -/
theorem max_value_of_function : 
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (1/3) ∧ 
  (∀ x, x ∈ Set.Ioo 0 (1/3) → x^2 * (1 - 3*x) ≤ c) ∧
  c = 1/12 := by
sorry

end max_value_of_function_l2242_224203


namespace product_of_four_consecutive_integers_divisible_by_24_l2242_224267

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k ∧
  ∀ m : ℕ, m > 24 → ¬(∀ n : ℕ, ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) = m * k) :=
by sorry

end product_of_four_consecutive_integers_divisible_by_24_l2242_224267


namespace hyperbola_equation_l2242_224290

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * Real.sqrt 7 * x

-- Define the asymptote passing through (2, √3)
def asymptote_through_point (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3 / 2

-- Define the focus on the directrix condition
def focus_on_directrix (c : ℝ) : Prop :=
  c = Real.sqrt 7

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : asymptote_through_point a b)
  (h4 : ∃ c, focus_on_directrix c ∧ a^2 + b^2 = c^2) :
  ∀ x y, hyperbola a b x y ↔ x^2 / 4 - y^2 / 3 = 1 :=
sorry

end hyperbola_equation_l2242_224290


namespace incircle_excircle_center_distance_l2242_224247

/-- Given a triangle DEF with side lengths, prove the distance between incircle and excircle centers --/
theorem incircle_excircle_center_distance (DE DF EF : ℝ) (h_DE : DE = 20) (h_DF : DF = 21) (h_EF : EF = 29) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let I := Real.sqrt (DE^2 + r^2)
  let E := (DF * I) / DE
  E - I = Real.sqrt 232 / 14 := by sorry

end incircle_excircle_center_distance_l2242_224247


namespace factorization_equality_l2242_224208

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 6 * x = 3 * x * (x * y - 2) := by
  sorry

end factorization_equality_l2242_224208


namespace mukesh_travel_distance_l2242_224269

theorem mukesh_travel_distance : ∀ x : ℝ,
  (x / 90 - x / 120 = 4 / 15) →
  x = 96 := by
  sorry

end mukesh_travel_distance_l2242_224269


namespace max_consecutive_sum_l2242_224216

/-- The sum of consecutive integers from a to (a + n - 1) -/
def sumConsecutive (a : ℤ) (n : ℕ) : ℤ := n * (2 * a + n - 1) / 2

/-- The target sum we want to achieve -/
def targetSum : ℤ := 2015

/-- Theorem stating that the maximum number of consecutive integers summing to 2015 is 4030 -/
theorem max_consecutive_sum :
  (∃ a : ℤ, sumConsecutive a 4030 = targetSum) ∧
  (∀ n : ℕ, n > 4030 → ∀ a : ℤ, sumConsecutive a n ≠ targetSum) :=
sorry

end max_consecutive_sum_l2242_224216


namespace point_outside_circle_l2242_224266

/-- A line intersects a circle at two distinct points if and only if 
    the distance from the circle's center to the line is less than the radius -/
axiom line_intersects_circle_iff_distance_lt_radius 
  (a b : ℝ) : (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    a * x₁ + b * y₁ = 4 ∧ x₁^2 + y₁^2 = 4 ∧
    a * x₂ + b * y₂ = 4 ∧ x₂^2 + y₂^2 = 4) ↔
  (4 / Real.sqrt (a^2 + b^2) < 2)

/-- The distance from a point to the origin is greater than 2 
    if and only if the point is outside the circle with radius 2 centered at the origin -/
axiom outside_circle_iff_distance_gt_radius 
  (a b : ℝ) : Real.sqrt (a^2 + b^2) > 2 ↔ (a, b) ∉ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}

theorem point_outside_circle (a b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    a * x₁ + b * y₁ = 4 ∧ x₁^2 + y₁^2 = 4 ∧
    a * x₂ + b * y₂ = 4 ∧ x₂^2 + y₂^2 = 4) →
  (a, b) ∉ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4} := by
  sorry

end point_outside_circle_l2242_224266


namespace xiaoming_coins_l2242_224253

theorem xiaoming_coins (first_day : ℕ) (second_day : ℕ)
  (h1 : first_day = 22)
  (h2 : second_day = 12) :
  first_day + second_day = 34 := by
  sorry

end xiaoming_coins_l2242_224253


namespace flagpole_shadow_length_l2242_224215

/-- Given a flagpole and a building under similar shadow-casting conditions,
    prove that the flagpole's shadow length is 45 meters. -/
theorem flagpole_shadow_length
  (flagpole_height : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h1 : flagpole_height = 18)
  (h2 : building_height = 24)
  (h3 : building_shadow = 60)
  (h4 : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  flagpole_shadow = 45 :=
by
  sorry


end flagpole_shadow_length_l2242_224215


namespace pokemon_cards_distribution_l2242_224286

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 56) (h2 : num_friends = 4) :
  total_cards / num_friends = 14 := by
  sorry

end pokemon_cards_distribution_l2242_224286
