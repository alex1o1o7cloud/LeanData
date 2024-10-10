import Mathlib

namespace division_simplification_l3111_311158

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  -6 * a^3 * b / (3 * a * b) = -2 * a^2 := by
  sorry

end division_simplification_l3111_311158


namespace quadratic_inequality_range_l3111_311146

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end quadratic_inequality_range_l3111_311146


namespace function_derivative_inequality_l3111_311195

theorem function_derivative_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hf' : ∀ x, deriv f x < 1) :
  ∀ m : ℝ, f (1 - m) - f m > 1 - 2*m → m > 1/2 := by
  sorry

end function_derivative_inequality_l3111_311195


namespace segment_length_product_l3111_311104

theorem segment_length_product (a : ℝ) : 
  (∃ a₁ a₂ : ℝ, 
    (∀ a : ℝ, (3 * a - 5)^2 + (2 * a - 5)^2 = 125 ↔ (a = a₁ ∨ a = a₂)) ∧
    a₁ * a₂ = -749/676) :=
by sorry

end segment_length_product_l3111_311104


namespace theme_park_triplets_l3111_311103

theorem theme_park_triplets (total_cost mother_charge child_charge_per_year : ℚ)
  (h_total_cost : total_cost = 15.25)
  (h_mother_charge : mother_charge = 6.95)
  (h_child_charge : child_charge_per_year = 0.55)
  : ∃ (triplet_age youngest_age : ℕ),
    triplet_age > youngest_age ∧
    youngest_age = 3 ∧
    total_cost = mother_charge + child_charge_per_year * (3 * triplet_age + youngest_age) :=
by sorry

end theme_park_triplets_l3111_311103


namespace chess_tournament_games_l3111_311147

theorem chess_tournament_games (n : ℕ) (h : n = 24) : 
  n * (n - 1) / 2 = 552 := by
  sorry

end chess_tournament_games_l3111_311147


namespace courtyard_width_l3111_311143

/-- The width of a courtyard given its length, brick dimensions, and total number of bricks --/
theorem courtyard_width (length : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℝ) :
  length = 28 →
  brick_length = 0.22 →
  brick_width = 0.12 →
  total_bricks = 13787.878787878788 →
  ∃ width : ℝ, abs (width - 13.012) < 0.001 ∧ 
    length * width * 100 * 100 = total_bricks * brick_length * brick_width * 100 * 100 :=
by sorry

end courtyard_width_l3111_311143


namespace banana_count_l3111_311101

theorem banana_count (bunches_of_eight : Nat) (bananas_per_bunch_eight : Nat)
                     (bunches_of_seven : Nat) (bananas_per_bunch_seven : Nat) :
  bunches_of_eight = 6 →
  bananas_per_bunch_eight = 8 →
  bunches_of_seven = 5 →
  bananas_per_bunch_seven = 7 →
  bunches_of_eight * bananas_per_bunch_eight + bunches_of_seven * bananas_per_bunch_seven = 83 :=
by sorry

end banana_count_l3111_311101


namespace inverse_of_3_mod_199_l3111_311123

theorem inverse_of_3_mod_199 : ∃ x : ℕ, x < 199 ∧ (3 * x) % 199 = 1 :=
by
  use 133
  sorry

end inverse_of_3_mod_199_l3111_311123


namespace quadratic_equations_solutions_l3111_311186

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 3*x = 0) ∧
  (∃ x : ℝ, x^2 - 4*x - 1 = 0) ∧
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ (x = 0 ∨ x = 3)) ∧
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5)) :=
by sorry

end quadratic_equations_solutions_l3111_311186


namespace time_taken_BC_l3111_311163

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 40
def work_rate_B : ℚ := 1 / 60
def work_rate_C : ℚ := 1 / 80

-- Define the work done by A and B
def work_done_A : ℚ := 10 * work_rate_A
def work_done_B : ℚ := 5 * work_rate_B

-- Define the remaining work
def remaining_work : ℚ := 1 - (work_done_A + work_done_B)

-- Define the combined work rate of B and C
def combined_rate_BC : ℚ := work_rate_B + work_rate_C

-- Theorem stating the time taken by B and C to finish the remaining work
theorem time_taken_BC : (remaining_work / combined_rate_BC) = 160 / 7 := by
  sorry

end time_taken_BC_l3111_311163


namespace product_of_terms_l3111_311153

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 = 2 →
  a 2 * a 3 * a 5 * a 6 = 16 := by
sorry

end product_of_terms_l3111_311153


namespace fraction_equality_l3111_311130

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (3 * a) / (3 * b) = a / b := by
  sorry

end fraction_equality_l3111_311130


namespace even_factors_count_l3111_311187

/-- The number of even positive factors of 2^4 * 3^2 * 5 * 7 -/
def num_even_factors (n : ℕ) : ℕ :=
  if n = 2^4 * 3^2 * 5 * 7 then
    (4 * 3 * 2 * 2)  -- 4 choices for 2's exponent (1 to 4), 3 for 3's (0 to 2), 2 for 5's (0 to 1), 2 for 7's (0 to 1)
  else
    0  -- Return 0 if n is not equal to 2^4 * 3^2 * 5 * 7

theorem even_factors_count (n : ℕ) :
  n = 2^4 * 3^2 * 5 * 7 → num_even_factors n = 48 := by
  sorry

end even_factors_count_l3111_311187


namespace bowling_shoe_rental_cost_l3111_311177

/-- The cost to rent bowling shoes for a day, given the following conditions:
  1. The cost per game is $1.75.
  2. A person has $12.80 in total.
  3. The person can bowl a maximum of 7 complete games. -/
theorem bowling_shoe_rental_cost :
  let cost_per_game : ℚ := 175 / 100
  let total_money : ℚ := 1280 / 100
  let max_games : ℕ := 7
  let shoe_rental_cost : ℚ := total_money - (cost_per_game * max_games)
  shoe_rental_cost = 55 / 100 := by sorry

end bowling_shoe_rental_cost_l3111_311177


namespace unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8_l3111_311137

theorem unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n.val = 24 * k) ∧ 
    (7.9 : ℝ) < (n.val : ℝ)^(1/3) ∧ 
    (n.val : ℝ)^(1/3) < 8 ∧
    n.val = 504 := by
  sorry

end unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8_l3111_311137


namespace extreme_points_inequality_l3111_311107

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - 1 - a * Real.log x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  (∀ x, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
  (f a x₁) / x₂ > -7/2 - Real.log 2 :=
sorry

end extreme_points_inequality_l3111_311107


namespace jacket_price_restoration_l3111_311199

theorem jacket_price_restoration :
  ∀ (original_price : ℝ),
  original_price > 0 →
  let price_after_first_reduction := original_price * (1 - 0.2)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.25)
  let required_increase := (original_price / price_after_second_reduction) - 1
  abs (required_increase - 0.6667) < 0.0001 := by
  sorry

end jacket_price_restoration_l3111_311199


namespace exam_questions_count_l3111_311165

theorem exam_questions_count :
  ∀ (a b c : ℕ),
    b = 23 →
    c = 1 →
    a ≥ 1 →
    b ≥ 1 →
    c ≥ 1 →
    a ≥ (6 : ℚ) / 10 * (a + 2 * b + 3 * c) →
    a + b + c = 98 :=
by
  sorry

end exam_questions_count_l3111_311165


namespace sufficient_condition_for_inequality_l3111_311193

theorem sufficient_condition_for_inequality (a b : ℝ) :
  Real.sqrt (a - 1) > Real.sqrt (b - 1) → a > b ∧ b > 0 := by
  sorry

end sufficient_condition_for_inequality_l3111_311193


namespace set_intersection_proof_l3111_311178

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

theorem set_intersection_proof : M ∩ N = {1, 2} := by
  sorry

end set_intersection_proof_l3111_311178


namespace garage_wheels_l3111_311154

/-- The number of bikes that can be assembled -/
def bikes_assembled : ℕ := 7

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- Theorem: The total number of bike wheels in the garage is 14 -/
theorem garage_wheels : bikes_assembled * wheels_per_bike = 14 := by
  sorry

end garage_wheels_l3111_311154


namespace combined_distance_theorem_l3111_311141

def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

def train_A_speed : ℝ := 150
def train_A_time : ℝ := 8

def train_B_speed : ℝ := 180
def train_B_time : ℝ := 6

def train_C_speed : ℝ := 120
def train_C_time : ℝ := 10

theorem combined_distance_theorem :
  train_distance train_A_speed train_A_time +
  train_distance train_B_speed train_B_time +
  train_distance train_C_speed train_C_time = 3480 := by
  sorry

end combined_distance_theorem_l3111_311141


namespace watch_time_calculation_l3111_311170

-- Define constants based on the problem conditions
def regular_season_episodes : ℕ := 22
def third_season_episodes : ℕ := 24
def last_season_extra_episodes : ℕ := 4
def previous_seasons : ℕ := 9
def early_episode_length : ℚ := 1/2
def later_episode_length : ℚ := 3/4
def bonus_episodes : ℕ := 5
def bonus_episode_length : ℚ := 1
def crossover_episode_length : ℚ := 3/2
def marathon_length : ℚ := 5
def daily_watch_time : ℚ := 2

-- Theorem to prove
theorem watch_time_calculation :
  let total_episodes := 
    3 * regular_season_episodes + 2 + -- First three seasons
    6 * regular_season_episodes + -- Seasons 4-9
    (regular_season_episodes + last_season_extra_episodes) -- Last season
  let total_hours := 
    (3 * regular_season_episodes + 2) * early_episode_length + -- First three seasons
    (6 * regular_season_episodes) * later_episode_length + -- Seasons 4-9
    (regular_season_episodes + last_season_extra_episodes) * later_episode_length + -- Last season
    bonus_episodes * bonus_episode_length + -- Bonus episodes
    crossover_episode_length -- Crossover episode
  let remaining_hours := total_hours - marathon_length
  let days_to_finish := remaining_hours / daily_watch_time
  days_to_finish = 77 := by sorry

end watch_time_calculation_l3111_311170


namespace first_day_over_100_l3111_311135

def paperclips (day : ℕ) : ℕ :=
  if day = 0 then 5
  else if day = 1 then 7
  else 7 + 3 * (day - 1)

def dayOfWeek (day : ℕ) : String :=
  match day % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_over_100 :
  (∀ d : ℕ, d < 33 → paperclips d ≤ 100) ∧
  paperclips 33 > 100 ∧
  dayOfWeek 33 = "Friday" := by
  sorry

end first_day_over_100_l3111_311135


namespace board_operations_finite_and_invariant_l3111_311116

/-- Represents the state of the board with n natural numbers -/
def Board := List Nat

/-- Performs one operation on the board, replacing two numbers with their GCD and LCM -/
def performOperation (board : Board) (i j : Nat) : Board :=
  sorry

/-- Checks if the board is in its final state (all pairs are proper) -/
def isFinalState (board : Board) : Bool :=
  sorry

theorem board_operations_finite_and_invariant (initial_board : Board) :
  ∃ (final_board : Board),
    (∀ (sequence : List (Nat × Nat)), 
      isFinalState (sequence.foldl (λ b (i, j) => performOperation b i j) initial_board)) ∧
    (∀ (sequence1 sequence2 : List (Nat × Nat)),
      isFinalState (sequence1.foldl (λ b (i, j) => performOperation b i j) initial_board) ∧
      isFinalState (sequence2.foldl (λ b (i, j) => performOperation b i j) initial_board) →
      sequence1.foldl (λ b (i, j) => performOperation b i j) initial_board =
      sequence2.foldl (λ b (i, j) => performOperation b i j) initial_board) :=
by
  sorry

end board_operations_finite_and_invariant_l3111_311116


namespace min_value_theorem_l3111_311173

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (y / (2*x) + 1 / y) ≥ 2 + Real.sqrt 2 :=
by sorry

end min_value_theorem_l3111_311173


namespace probability_at_most_one_correct_l3111_311126

theorem probability_at_most_one_correct (pA pB : ℚ) : 
  pA = 3/5 → pB = 2/3 → 
  let p_at_most_one := 
    (1 - pA) * (1 - pA) * (1 - pB) * (1 - pB) + 
    2 * pA * (1 - pA) * (1 - pB) * (1 - pB) + 
    2 * (1 - pA) * (1 - pA) * pB * (1 - pB)
  p_at_most_one = 32/225 := by
sorry

end probability_at_most_one_correct_l3111_311126


namespace last_painted_cell_l3111_311184

/-- Represents a cell in a rectangular grid --/
structure Cell where
  row : ℕ
  col : ℕ

/-- Represents a rectangular grid --/
structure Rectangle where
  rows : ℕ
  cols : ℕ

/-- Defines the spiral painting process --/
def spiralPaint (rect : Rectangle) : Cell :=
  sorry

/-- Theorem statement for the last painted cell in a 333 × 444 rectangle --/
theorem last_painted_cell :
  let rect : Rectangle := { rows := 333, cols := 444 }
  spiralPaint rect = { row := 167, col := 278 } :=
sorry

end last_painted_cell_l3111_311184


namespace cube_root_of_eight_l3111_311156

theorem cube_root_of_eight (x y : ℝ) : x^(3*y) = 8 ∧ x = 8 → y = 1/3 := by
  sorry

end cube_root_of_eight_l3111_311156


namespace composition_of_even_is_even_l3111_311113

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (g ∘ g) := by
  sorry

end composition_of_even_is_even_l3111_311113


namespace order_of_expressions_l3111_311191

theorem order_of_expressions : 
  let a : ℝ := (3/5)^(2/5)
  let b : ℝ := (2/5)^(3/5)
  let c : ℝ := (2/5)^(2/5)
  b < c ∧ c < a := by
  sorry

end order_of_expressions_l3111_311191


namespace sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds_l3111_311100

theorem sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds :
  8 < Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) ∧
  Real.sqrt 2 * (Real.sqrt 8 + Real.sqrt 10) < 9 :=
by sorry

end sqrt_two_times_sqrt_eight_plus_sqrt_ten_bounds_l3111_311100


namespace f_values_l3111_311181

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

-- Theorem stating that f(2) = 14 and f(-2) = 2
theorem f_values : f 2 = 14 ∧ f (-2) = 2 := by
  sorry

end f_values_l3111_311181


namespace adjacent_semicircles_perimeter_l3111_311128

/-- The perimeter of a shape formed by two adjacent semicircles with radius 1 --/
theorem adjacent_semicircles_perimeter :
  ∀ (r : ℝ), r = 1 →
  ∃ (perimeter : ℝ), perimeter = 3 * r :=
by sorry

end adjacent_semicircles_perimeter_l3111_311128


namespace system_solutions_l3111_311139

theorem system_solutions (x a : ℝ) : 
  (a = -3*x^2 + 5*x - 2 ∧ (x+2)*a = 4*(x^2 - 1) ∧ x ≠ -2) → 
  ((x = 1 ∧ a = 0) ∨ (x = 0 ∧ a = -2) ∨ (x = -8/3 ∧ a = -110/3)) :=
by sorry

end system_solutions_l3111_311139


namespace polynomial_roots_b_value_l3111_311136

theorem polynomial_roots_b_value (A B C D : ℤ) : 
  (∀ z : ℤ, z > 0 → (z^6 - 10*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 16 = 0) → 
   (∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℤ, 
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₆ > 0 ∧
      z^6 - 10*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 16 = 
      (z - x₁) * (z - x₂) * (z - x₃) * (z - x₄) * (z - x₅) * (z - x₆))) →
  B = -88 := by
sorry

end polynomial_roots_b_value_l3111_311136


namespace chocolate_purchase_shortage_l3111_311179

theorem chocolate_purchase_shortage (chocolate_cost : ℕ) (initial_money : ℕ) (borrowed_money : ℕ) : 
  chocolate_cost = 500 ∧ initial_money = 400 ∧ borrowed_money = 59 →
  chocolate_cost - (initial_money + borrowed_money) = 41 :=
by sorry

end chocolate_purchase_shortage_l3111_311179


namespace equality_of_powers_l3111_311121

theorem equality_of_powers (a b c d : ℕ) :
  a^a * b^(a + b) = c^c * d^(c + d) →
  Nat.gcd a b = 1 →
  Nat.gcd c d = 1 →
  a = c ∧ b = d := by
  sorry

end equality_of_powers_l3111_311121


namespace trajectory_of_moving_point_l3111_311115

theorem trajectory_of_moving_point (x y : ℝ) :
  let segment_length : ℝ := 3
  let point_A : ℝ × ℝ := (3 * x, 0)
  let point_B : ℝ × ℝ := (0, 3 * y / 2)
  let point_C : ℝ × ℝ := (x, y)
  (point_A.1 - point_C.1)^2 + (point_A.2 - point_C.2)^2 = 4 * ((point_C.1 - point_B.1)^2 + (point_C.2 - point_B.2)^2) →
  x^2 + y^2 / 4 = 1 :=
by sorry

end trajectory_of_moving_point_l3111_311115


namespace unique_function_property_l3111_311149

theorem unique_function_property (f : ℕ → ℕ) :
  (∀ n : ℕ, f n + f (f n) = 2 * n) ↔ (∀ n : ℕ, f n = n) := by
  sorry

end unique_function_property_l3111_311149


namespace number_of_possible_sums_l3111_311172

def bag_A : Finset ℕ := {1, 3, 5}
def bag_B : Finset ℕ := {2, 4, 6}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem number_of_possible_sums : possible_sums.card = 5 := by
  sorry

end number_of_possible_sums_l3111_311172


namespace sqrt_product_plus_one_l3111_311109

theorem sqrt_product_plus_one : 
  Real.sqrt ((41 : ℝ) * 40 * 39 * 38 + 1) = 1559 := by sorry

end sqrt_product_plus_one_l3111_311109


namespace inequality_solution_set_l3111_311142

theorem inequality_solution_set (x : ℝ) : 
  x^2 - 2*|x| - 15 > 0 ↔ x < -5 ∨ x > 5 := by
  sorry

end inequality_solution_set_l3111_311142


namespace response_rate_percentage_l3111_311190

/-- Given that 300 responses are needed and the minimum number of questionnaires
    to be mailed is 375, prove that the response rate percentage is 80%. -/
theorem response_rate_percentage
  (responses_needed : ℕ)
  (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 300)
  (h2 : questionnaires_mailed = 375) :
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 80 := by
  sorry

end response_rate_percentage_l3111_311190


namespace distribute_7_balls_3_boxes_l3111_311160

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_7_balls_3_boxes : distribute 7 3 = 36 := by
  sorry

end distribute_7_balls_3_boxes_l3111_311160


namespace derivative_at_one_l3111_311110

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- State the theorem
theorem derivative_at_one : 
  deriv f 1 = 4 := by sorry

end derivative_at_one_l3111_311110


namespace compare_fractions_l3111_311129

theorem compare_fractions : -3/4 > -|-(4/5)| := by
  sorry

end compare_fractions_l3111_311129


namespace consecutive_ones_count_is_3719_l3111_311125

/-- Fibonacci-like sequence for numbers without consecutive 1's -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => F (n + 1) + F n

/-- The number of 12-digit integers with digits 1 or 2 and two consecutive 1's -/
def consecutive_ones_count : ℕ := 2^12 - F 11

theorem consecutive_ones_count_is_3719 : consecutive_ones_count = 3719 := by
  sorry

end consecutive_ones_count_is_3719_l3111_311125


namespace jiAnWinningCases_l3111_311166

/-- Represents the possible moves in rock-paper-scissors -/
inductive Move
  | Rock
  | Paper
  | Scissors

/-- Determines if the first move wins against the second move -/
def wins (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Rock, Move.Scissors => true
  | Move.Paper, Move.Rock => true
  | Move.Scissors, Move.Paper => true
  | _, _ => false

/-- Counts the number of winning cases for the first player -/
def countWinningCases : Nat :=
  List.length (List.filter
    (fun (m1, m2) => wins m1 m2)
    [(Move.Rock, Move.Paper), (Move.Rock, Move.Scissors), (Move.Rock, Move.Rock),
     (Move.Paper, Move.Rock), (Move.Paper, Move.Scissors), (Move.Paper, Move.Paper),
     (Move.Scissors, Move.Rock), (Move.Scissors, Move.Paper), (Move.Scissors, Move.Scissors)])

theorem jiAnWinningCases :
  countWinningCases = 3 := by sorry

end jiAnWinningCases_l3111_311166


namespace well_diameter_l3111_311117

/-- Proves that a circular well with given depth and volume has a specific diameter -/
theorem well_diameter (depth : ℝ) (volume : ℝ) (π : ℝ) :
  depth = 10 →
  volume = 31.41592653589793 →
  π = 3.141592653589793 →
  (2 * (volume / (π * depth)))^(1/2) = 2 := by
  sorry

end well_diameter_l3111_311117


namespace number_ordering_l3111_311162

theorem number_ordering : (2 : ℝ)^30 < 10^10 ∧ 10^10 < 5^15 := by
  sorry

end number_ordering_l3111_311162


namespace midpoint_after_translation_l3111_311114

-- Define the points B and G
def B : ℝ × ℝ := (2, 3)
def G : ℝ × ℝ := (6, 3)

-- Define the translation
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 7, p.2 - 3)

-- Theorem statement
theorem midpoint_after_translation :
  let B' := translate B
  let G' := translate G
  (B'.1 + G'.1) / 2 = -3 ∧ (B'.2 + G'.2) / 2 = 0 :=
by sorry

end midpoint_after_translation_l3111_311114


namespace carrots_removed_count_l3111_311159

-- Define the given constants
def total_carrots : ℕ := 30
def remaining_carrots : ℕ := 27
def total_weight : ℚ := 5.94
def avg_weight_remaining : ℚ := 0.2
def avg_weight_removed : ℚ := 0.18

-- Define the number of removed carrots
def removed_carrots : ℕ := total_carrots - remaining_carrots

-- Theorem statement
theorem carrots_removed_count :
  removed_carrots = 3 := by sorry

end carrots_removed_count_l3111_311159


namespace unique_solution_quadratic_l3111_311124

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = m + 3 * x) ↔ m = 5 := by
  sorry

end unique_solution_quadratic_l3111_311124


namespace inequality_properties_l3111_311108

theorem inequality_properties (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (1 / a > 1 / b) ∧
  (a^(1/5 : ℝ) < b^(1/5 : ℝ)) ∧
  (Real.sqrt (a^2 - a) > Real.sqrt (b^2 - b)) := by
  sorry

end inequality_properties_l3111_311108


namespace total_loaves_served_l3111_311131

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real)
  (h1 : wheat_bread = 0.2)
  (h2 : white_bread = 0.4) :
  wheat_bread + white_bread = 0.6 := by
  sorry

end total_loaves_served_l3111_311131


namespace max_guaranteed_profit_l3111_311105

/-- Represents the number of balls -/
def n : ℕ := 10

/-- Represents the cost of a test and the price of a non-radioactive ball -/
def cost : ℕ := 1

/-- Represents the triangular number function -/
def H (k : ℕ) : ℕ := k * (k + 1) / 2

/-- Theorem stating the maximum guaranteed profit for n balls -/
theorem max_guaranteed_profit :
  ∃ k : ℕ, H k < n ∧ n ≤ H (k + 1) ∧ n - (k + 1) = 5 :=
sorry

end max_guaranteed_profit_l3111_311105


namespace arithmetic_sequence_sum_l3111_311106

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 39) →
  (a 3 + a 6 + a 9 = 33) :=
by sorry

end arithmetic_sequence_sum_l3111_311106


namespace arithmetic_sequence_common_difference_l3111_311127

theorem arithmetic_sequence_common_difference
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : ∃ (d : ℝ), ∃ (m n k : ℤ), a = a + m * d ∧ b = a + n * d ∧ c = a + k * d)
  (h_equation : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  ∃ (d : ℝ), d = 0 ∧ ∃ (m n k : ℤ), a = a + m * d ∧ b = a + n * d ∧ c = a + k * d :=
sorry

end arithmetic_sequence_common_difference_l3111_311127


namespace max_score_15_cards_l3111_311180

/-- Represents the score for a hand of cards -/
def score (r b y : ℕ) : ℕ := r + 2 * r * b + 3 * b * y

/-- The maximum score achievable with 15 cards -/
theorem max_score_15_cards : 
  ∃ (r b y : ℕ), r + b + y = 15 ∧ ∀ (r' b' y' : ℕ), r' + b' + y' = 15 → score r' b' y' ≤ score r b y ∧ score r b y = 168 := by
  sorry

end max_score_15_cards_l3111_311180


namespace min_value_expression_l3111_311171

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 ∧
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) = 8 ↔ x = 2 ∧ y = 2) :=
by sorry

end min_value_expression_l3111_311171


namespace complex_number_in_fourth_quadrant_l3111_311138

/-- The complex number (1+i)/i is in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 + Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l3111_311138


namespace multiples_of_five_relation_l3111_311182

theorem multiples_of_five_relation (a b c : ℤ) : 
  (∃ k l m : ℤ, a = 5 * k ∧ b = 5 * l ∧ c = 5 * m) →  -- a, b, c are multiples of 5
  a < b →                                            -- a < b
  b < c →                                            -- b < c
  c = a + 10 →                                       -- c = a + 10
  (a - b) * (a - c) / (b - c) = -10 := by             -- Prove that the expression equals -10
sorry


end multiples_of_five_relation_l3111_311182


namespace water_moles_in_reaction_l3111_311189

-- Define the chemical reaction
structure ChemicalReaction where
  lithium_nitride : ℚ
  water : ℚ
  lithium_hydroxide : ℚ
  ammonia : ℚ

-- Define the balanced equation
def balanced_equation (r : ChemicalReaction) : Prop :=
  r.lithium_nitride = r.water / 3 ∧ 
  r.lithium_hydroxide = r.water ∧ 
  r.ammonia = r.water / 3

-- Theorem statement
theorem water_moles_in_reaction 
  (r : ChemicalReaction) 
  (h1 : r.lithium_nitride = 1) 
  (h2 : r.lithium_hydroxide = 3) 
  (h3 : balanced_equation r) : 
  r.water = 3 := by sorry

end water_moles_in_reaction_l3111_311189


namespace board_game_cost_l3111_311148

theorem board_game_cost (jump_rope_cost playground_ball_cost dalton_savings uncle_gift additional_needed : ℕ) 
  (h1 : jump_rope_cost = 7)
  (h2 : playground_ball_cost = 4)
  (h3 : dalton_savings = 6)
  (h4 : uncle_gift = 13)
  (h5 : additional_needed = 4) :
  jump_rope_cost + playground_ball_cost + (dalton_savings + uncle_gift + additional_needed) - (dalton_savings + uncle_gift) = 12 := by
  sorry

end board_game_cost_l3111_311148


namespace exam_time_ratio_l3111_311161

/-- Given an examination with the following parameters:
  * Total duration: 3 hours
  * Total number of questions: 200
  * Number of type A problems: 25
  * Time spent on type A problems: 40 minutes

  Prove that the ratio of time spent on type A problems to time spent on type B problems is 2:7. -/
theorem exam_time_ratio :
  let total_time : ℕ := 3 * 60  -- Total time in minutes
  let type_a_time : ℕ := 40     -- Time spent on type A problems
  let type_b_time : ℕ := total_time - type_a_time  -- Time spent on type B problems
  (type_a_time : ℚ) / (type_b_time : ℚ) = 2 / 7 := by
  sorry

end exam_time_ratio_l3111_311161


namespace inscribed_parallelogram_theorem_l3111_311164

/-- A triangle with an inscribed parallelogram -/
structure InscribedParallelogram where
  -- Triangle side lengths
  side1 : ℝ
  side2 : ℝ
  -- Parallelogram side on triangle base
  para_side : ℝ

/-- Properties of the inscribed parallelogram -/
def inscribed_parallelogram_properties (t : InscribedParallelogram) : Prop :=
  t.side1 = 9 ∧ t.side2 = 15 ∧ t.para_side = 6

/-- Theorem about the inscribed parallelogram -/
theorem inscribed_parallelogram_theorem (t : InscribedParallelogram) 
  (h : inscribed_parallelogram_properties t) :
  ∃ (other_side base : ℝ),
    other_side = 4 * Real.sqrt 2 ∧ 
    base = 18 :=
by sorry

end inscribed_parallelogram_theorem_l3111_311164


namespace dryer_sheet_box_cost_l3111_311185

/-- The cost of a box of dryer sheets -/
def box_cost (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (yearly_savings : ℚ) : ℚ :=
  yearly_savings / (loads_per_week * 52 / sheets_per_box)

/-- Theorem stating the cost of a box of dryer sheets -/
theorem dryer_sheet_box_cost :
  box_cost 4 1 104 11 = (11/2 : ℚ) := by
  sorry

end dryer_sheet_box_cost_l3111_311185


namespace remainder_of_power_sum_l3111_311152

theorem remainder_of_power_sum (n : ℕ) : (Nat.pow 6 83 + Nat.pow 8 83) % 49 = 35 := by
  sorry

end remainder_of_power_sum_l3111_311152


namespace triangle_side_lengths_l3111_311150

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_lengths (x : ℕ) :
  (x > 0) →
  (is_valid_triangle (3 * x) 10 (x^2)) →
  (x = 3 ∨ x = 4) :=
by sorry

end triangle_side_lengths_l3111_311150


namespace inequality_equivalence_l3111_311132

theorem inequality_equivalence (x : ℝ) : 
  -2 < (x^2 - 12*x + 20) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 12*x + 20) / (x^2 - 4*x + 8) < 2 ↔ 
  x > 5 :=
sorry

end inequality_equivalence_l3111_311132


namespace trucks_meeting_l3111_311112

/-- Two trucks meeting on a highway --/
theorem trucks_meeting 
  (initial_distance : ℝ) 
  (speed_A speed_B : ℝ) 
  (delay : ℝ) :
  initial_distance = 940 →
  speed_A = 90 →
  speed_B = 80 →
  delay = 1 →
  ∃ (t : ℝ), 
    t > 0 ∧ 
    speed_A * (t + delay) + speed_B * t = initial_distance ∧ 
    speed_A * (t + delay) - speed_B * t = 140 :=
by sorry

end trucks_meeting_l3111_311112


namespace second_year_decrease_is_twenty_percent_l3111_311155

/-- Represents the population change over two years --/
structure PopulationChange where
  initial_population : ℕ
  first_year_increase : ℚ
  final_population : ℕ

/-- Calculates the percentage decrease in the second year --/
def second_year_decrease (pc : PopulationChange) : ℚ :=
  let first_year_population := pc.initial_population * (1 + pc.first_year_increase)
  1 - (pc.final_population : ℚ) / first_year_population

/-- Theorem: Given the specified population change, the second year decrease is 20% --/
theorem second_year_decrease_is_twenty_percent 
  (pc : PopulationChange) 
  (h1 : pc.initial_population = 10000)
  (h2 : pc.first_year_increase = 1/5)
  (h3 : pc.final_population = 9600) : 
  second_year_decrease pc = 1/5 := by
  sorry

end second_year_decrease_is_twenty_percent_l3111_311155


namespace rectangular_solid_diagonal_l3111_311111

theorem rectangular_solid_diagonal (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry

end rectangular_solid_diagonal_l3111_311111


namespace triangle_properties_and_heron_l3111_311169

/-- Triangle properties and Heron's formula -/
theorem triangle_properties_and_heron (r r_a r_b r_c p a b c S : ℝ) 
  (hr : r > 0) (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0)
  (hp : p > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S > 0)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_inradius : r = S / p)
  (h_exradius_a : r_a = S / (p - a))
  (h_exradius_b : r_b = S / (p - b))
  (h_exradius_c : r_c = S / (p - c)) : 
  (r * p = r_a * (p - a)) ∧ 
  (r * r_a = (p - b) * (p - c)) ∧
  (r_b * r_c = p * (p - a)) ∧
  (S^2 = p * (p - a) * (p - b) * (p - c)) ∧
  (S^2 = r * r_a * r_b * r_c) := by
  sorry

end triangle_properties_and_heron_l3111_311169


namespace union_equals_A_implies_p_le_3_l3111_311140

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (p : ℝ) : Set ℝ := {x : ℝ | p + 1 < x ∧ x < 2*p - 1}

-- State the theorem
theorem union_equals_A_implies_p_le_3 (p : ℝ) :
  A ∪ B p = A → p ≤ 3 := by
  sorry

end union_equals_A_implies_p_le_3_l3111_311140


namespace factorial_sum_eq_power_of_three_l3111_311151

theorem factorial_sum_eq_power_of_three (a b c d : ℕ) : 
  a ≤ b → b ≤ c → a.factorial + b.factorial + c.factorial = 3^d →
  ((a, b, c, d) = (1, 1, 1, 1) ∨ (a, b, c, d) = (1, 2, 3, 2) ∨ (a, b, c, d) = (1, 2, 4, 3)) :=
by sorry

end factorial_sum_eq_power_of_three_l3111_311151


namespace coat_discount_proof_l3111_311134

theorem coat_discount_proof :
  ∃ (p q : ℕ), 
    p < 10 ∧ q < 10 ∧
    21250 * (1 - p / 100) * (1 - q / 100) = 19176 ∧
    ((p = 4 ∧ q = 6) ∨ (p = 6 ∧ q = 4)) := by
  sorry

end coat_discount_proof_l3111_311134


namespace modular_congruence_existence_l3111_311174

theorem modular_congruence_existence (a c : ℕ+) (b : ℤ) :
  ∃ x : ℕ+, (c : ℤ) ∣ ((a : ℤ)^(x : ℕ) + x - b) := by
  sorry

end modular_congruence_existence_l3111_311174


namespace sqrt_three_diamond_sqrt_three_l3111_311198

-- Define the binary operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem sqrt_three_diamond_sqrt_three : diamond (Real.sqrt 3) (Real.sqrt 3) = 12 := by
  sorry

end sqrt_three_diamond_sqrt_three_l3111_311198


namespace isosceles_triangle_circle_centers_distance_l3111_311194

/-- For an isosceles triangle with circumradius R and inradius r, 
    the distance d between the centers of the circumscribed and inscribed circles 
    is given by d = √(R(R-2r)). -/
theorem isosceles_triangle_circle_centers_distance 
  (R r d : ℝ) 
  (h_R_pos : R > 0) 
  (h_r_pos : r > 0) 
  (h_isosceles : IsIsosceles) : 
  d = Real.sqrt (R * (R - 2 * r)) := by
  sorry

/-- Represents an isosceles triangle -/
structure IsIsosceles : Prop where
  -- Add necessary fields to represent an isosceles triangle
  -- This is left abstract as the problem doesn't provide specific details

#check isosceles_triangle_circle_centers_distance

end isosceles_triangle_circle_centers_distance_l3111_311194


namespace remainder_equality_l3111_311176

theorem remainder_equality (P P' D Q R R' : ℕ) 
  (h1 : P > P') 
  (h2 : R = P % D) 
  (h3 : R' = P' % D) : 
  ((P + Q) * P') % D = (R * R') % D := by
sorry

end remainder_equality_l3111_311176


namespace f_diff_at_pi_l3111_311144

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 3 * x^2 + 7 * Real.sin x

theorem f_diff_at_pi : f Real.pi - f (-Real.pi) = -2 * Real.pi^3 := by
  sorry

end f_diff_at_pi_l3111_311144


namespace obtain_x_squared_and_xy_l3111_311197

/-- Given positive real numbers x and y, prove that x^2 and xy can be obtained
    using operations of addition, subtraction, multiplication, division, and reciprocal. -/
theorem obtain_x_squared_and_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (f g : ℝ → ℝ → ℝ), f x y = x^2 ∧ g x y = x*y :=
by sorry

end obtain_x_squared_and_xy_l3111_311197


namespace rectangle_length_l3111_311157

theorem rectangle_length (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 4 * width →
  area = length * width →
  area = 100 →
  length = 20 :=
by sorry

end rectangle_length_l3111_311157


namespace negation_of_universal_statement_l3111_311167

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 1) ↔ (∃ x₀ : ℝ, x₀^2 < 1) :=
by sorry

end negation_of_universal_statement_l3111_311167


namespace prize_probability_l3111_311118

theorem prize_probability (p : ℝ) (h : p = 0.9) :
  Nat.choose 5 3 * p^3 * (1 - p)^2 = Nat.choose 5 3 * 0.9^3 * 0.1^2 := by
  sorry

end prize_probability_l3111_311118


namespace toy_spending_ratio_l3111_311120

def Trevor_spending : ℕ := 80
def total_spending : ℕ := 680
def years : ℕ := 4

def spending_ratio (Reed Quinn : ℕ) : Prop :=
  Reed = 2 * Quinn

theorem toy_spending_ratio :
  ∀ Reed Quinn : ℕ,
  (Trevor_spending = Reed + 20) →
  (∃ k : ℕ, Reed = k * Quinn) →
  (years * (Trevor_spending + Reed + Quinn) = total_spending) →
  spending_ratio Reed Quinn :=
by
  sorry

#check toy_spending_ratio

end toy_spending_ratio_l3111_311120


namespace max_value_fourth_root_sum_l3111_311196

theorem max_value_fourth_root_sum (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (abcd : ℝ) ^ (1/4) + ((1-a)*(1-b)*(1-c)*(1-d) : ℝ) ^ (1/4) ≤ 1 :=
by sorry

end max_value_fourth_root_sum_l3111_311196


namespace tram_speed_l3111_311188

/-- The speed of a tram given observation times and tunnel length -/
theorem tram_speed (t_pass : ℝ) (t_tunnel : ℝ) (tunnel_length : ℝ) 
  (h_pass : t_pass = 3)
  (h_tunnel : t_tunnel = 13)
  (h_length : tunnel_length = 100)
  (h_positive : t_pass > 0 ∧ t_tunnel > 0 ∧ tunnel_length > 0) :
  tunnel_length / (t_tunnel - t_pass) = 10 := by
  sorry

end tram_speed_l3111_311188


namespace largest_divisor_of_prime_square_difference_l3111_311145

theorem largest_divisor_of_prime_square_difference (p q : ℕ) 
  (hp : Prime p) (hq : Prime q) (h_order : q < p) : 
  (∀ (d : ℕ), d > 2 → ∃ (p' q' : ℕ), Prime p' ∧ Prime q' ∧ q' < p' ∧ ¬(d ∣ (p'^2 - q'^2))) ∧ 
  (∀ (p' q' : ℕ), Prime p' → Prime q' → q' < p' → 2 ∣ (p'^2 - q'^2)) :=
sorry

end largest_divisor_of_prime_square_difference_l3111_311145


namespace little_krish_sweet_expense_l3111_311133

theorem little_krish_sweet_expense (initial_amount : ℚ) (friend_gift : ℚ) (amount_left : ℚ) :
  initial_amount = 200.50 →
  friend_gift = 25.20 →
  amount_left = 114.85 →
  initial_amount - 2 * friend_gift - amount_left = 35.25 := by
  sorry

end little_krish_sweet_expense_l3111_311133


namespace old_workers_in_sample_l3111_311183

/-- Represents the composition of workers in a unit -/
structure WorkerComposition where
  total : ℕ
  young : ℕ
  old : ℕ
  middleAged : ℕ
  young_count : young ≤ total
  middleAged_relation : middleAged = 2 * old
  total_sum : total = young + old + middleAged

/-- Represents a stratified sample of workers -/
structure StratifiedSample where
  composition : WorkerComposition
  young_sample : ℕ
  young_sample_valid : young_sample ≤ composition.young

/-- Theorem stating the number of old workers in the stratified sample -/
theorem old_workers_in_sample (unit : WorkerComposition) (sample : StratifiedSample)
    (h_unit : unit.total = 430 ∧ unit.young = 160)
    (h_sample : sample.composition = unit ∧ sample.young_sample = 32) :
    (sample.young_sample : ℚ) / unit.young * unit.old = 18 := by
  sorry

end old_workers_in_sample_l3111_311183


namespace complex_exp_thirteen_pi_over_two_equals_i_l3111_311102

theorem complex_exp_thirteen_pi_over_two_equals_i :
  Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end complex_exp_thirteen_pi_over_two_equals_i_l3111_311102


namespace consecutive_numbers_sum_l3111_311119

theorem consecutive_numbers_sum (a b c d : ℤ) : 
  b = a + 1 → 
  c = b + 1 → 
  d = c + 1 → 
  b * c = 2970 → 
  a + d = 113 := by
sorry

end consecutive_numbers_sum_l3111_311119


namespace quadratic_root_n_value_l3111_311192

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x - 5 = 0

-- Define the root form
def root_form (x m n p : ℝ) : Prop := 
  (x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) ∧ 
  m > 0 ∧ n > 0 ∧ p > 0 ∧ Int.gcd ⌊m⌋ (Int.gcd ⌊n⌋ ⌊p⌋) = 1

-- Theorem statement
theorem quadratic_root_n_value : 
  ∃ (x m p : ℝ), quadratic_equation x ∧ root_form x m 31 p := by sorry

end quadratic_root_n_value_l3111_311192


namespace medals_count_l3111_311122

/-- The total number of medals displayed in the sports center -/
def total_medals (gold silver bronze : ℕ) : ℕ :=
  gold + silver + bronze

/-- Theorem: The total number of medals is 67 -/
theorem medals_count : total_medals 19 32 16 = 67 := by
  sorry

end medals_count_l3111_311122


namespace ratio_reduction_l3111_311168

theorem ratio_reduction (x : ℕ) (h : x ≥ 3) :
  (∃ a b : ℕ, a < b ∧ (6 - x : ℚ) / (7 - x) < a / b) ∧
  (∀ a b : ℕ, a < b → (6 - x : ℚ) / (7 - x) < a / b → 4 ≤ a) :=
sorry

end ratio_reduction_l3111_311168


namespace cos_product_special_angles_l3111_311175

theorem cos_product_special_angles : 
  Real.cos ((2 * π) / 5) * Real.cos ((6 * π) / 5) = -(1 / 4) := by
  sorry

end cos_product_special_angles_l3111_311175
