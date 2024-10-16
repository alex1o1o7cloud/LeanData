import Mathlib

namespace NUMINAMATH_CALUDE_tom_reading_pages_l836_83664

/-- Tom's initial reading speed in pages per hour -/
def initial_speed : ℕ := 12

/-- The factor by which Tom increases his reading speed -/
def speed_increase : ℕ := 3

/-- The number of hours Tom reads -/
def reading_time : ℕ := 2

/-- Theorem stating the number of pages Tom can read with increased speed -/
theorem tom_reading_pages : initial_speed * speed_increase * reading_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_pages_l836_83664


namespace NUMINAMATH_CALUDE_bobs_income_changes_l836_83617

def initial_income : ℝ := 2750
def february_increase : ℝ := 0.15
def march_decrease : ℝ := 0.10

theorem bobs_income_changes (initial : ℝ) (increase : ℝ) (decrease : ℝ) :
  initial = initial_income →
  increase = february_increase →
  decrease = march_decrease →
  initial * (1 + increase) * (1 - decrease) = 2846.25 :=
by sorry

end NUMINAMATH_CALUDE_bobs_income_changes_l836_83617


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l836_83618

/-- Given a 50% profit percent, prove that the ratio of the cost price to the selling price is 2:3 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_positive : cost_price > 0 ∧ selling_price > 0)
  (h_profit : selling_price = cost_price * 1.5) : 
  cost_price / selling_price = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l836_83618


namespace NUMINAMATH_CALUDE_intersection_points_on_horizontal_line_l836_83676

/-- Given two lines parameterized by a real number s, 
    prove that their intersection points lie on a horizontal line -/
theorem intersection_points_on_horizontal_line :
  ∀ (s : ℝ), 
  ∃ (x y : ℝ), 
  (2 * x + 3 * y = 6 * s + 4) ∧ 
  (x + 2 * y = 3 * s - 1) → 
  y = -6 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_on_horizontal_line_l836_83676


namespace NUMINAMATH_CALUDE_two_tails_one_head_probability_l836_83626

def coin_toss_probability : ℚ := 3/8

theorem two_tails_one_head_probability :
  let n_coins := 3
  let n_tails := 2
  let n_heads := 1
  let total_outcomes := 2^n_coins
  let favorable_outcomes := n_coins.choose n_tails
  coin_toss_probability = (favorable_outcomes : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_two_tails_one_head_probability_l836_83626


namespace NUMINAMATH_CALUDE_final_color_is_yellow_l836_83671

/-- Represents the color of an elf -/
inductive ElfColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of elves on the island -/
structure ElfState where
  blue : Nat
  red : Nat
  yellow : Nat
  total : Nat
  h_total : blue + red + yellow = total

/-- The score assigned to each color -/
def colorScore (c : ElfColor) : Nat :=
  match c with
  | ElfColor.Blue => 1
  | ElfColor.Red => 2
  | ElfColor.Yellow => 3

/-- The total score of all elves -/
def totalScore (state : ElfState) : Nat :=
  state.blue * colorScore ElfColor.Blue +
  state.red * colorScore ElfColor.Red +
  state.yellow * colorScore ElfColor.Yellow

/-- Theorem: The final color of all elves is yellow -/
theorem final_color_is_yellow (initial_state : ElfState)
  (h_initial : initial_state.blue = 7 ∧ initial_state.red = 10 ∧ initial_state.yellow = 17 ∧ initial_state.total = 34)
  (h_change : ∀ (state : ElfState), totalScore state % 3 = totalScore initial_state % 3)
  (h_final : ∃ (final_state : ElfState), (final_state.blue = final_state.total ∨ final_state.red = final_state.total ∨ final_state.yellow = final_state.total) ∧
              totalScore final_state % 3 = totalScore initial_state % 3) :
  ∃ (final_state : ElfState), final_state.yellow = final_state.total :=
sorry

end NUMINAMATH_CALUDE_final_color_is_yellow_l836_83671


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l836_83662

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 3| < 2}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ (x - 4) / x ≥ 0}

-- Define the interval [4, 5)
def interval : Set ℝ := {x : ℝ | 4 ≤ x ∧ x < 5}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l836_83662


namespace NUMINAMATH_CALUDE_heracles_age_l836_83649

theorem heracles_age (heracles_age audrey_age : ℕ) : 
  audrey_age = heracles_age + 7 →
  audrey_age + 3 = 2 * heracles_age →
  heracles_age = 10 := by
sorry

end NUMINAMATH_CALUDE_heracles_age_l836_83649


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l836_83696

theorem rectangle_shorter_side (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Positive dimensions
  2 * (a + b) = 52 ∧  -- Perimeter condition
  a * b = 168 ∧  -- Area condition
  a ≥ b  -- a is the longer side
  → b = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l836_83696


namespace NUMINAMATH_CALUDE_relay_race_time_reduction_l836_83667

theorem relay_race_time_reduction (T T₁ T₂ T₃ T₄ T₅ : ℝ) 
  (total_time : T = T₁ + T₂ + T₃ + T₄ + T₅)
  (first_runner : T₁ / 2 + T₂ + T₃ + T₄ + T₅ = 0.95 * T)
  (second_runner : T₁ + T₂ / 2 + T₃ + T₄ + T₅ = 0.90 * T)
  (third_runner : T₁ + T₂ + T₃ / 2 + T₄ + T₅ = 0.88 * T)
  (fourth_runner : T₁ + T₂ + T₃ + T₄ / 2 + T₅ = 0.85 * T) :
  T₁ + T₂ + T₃ + T₄ + T₅ / 2 = 0.92 * T := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_reduction_l836_83667


namespace NUMINAMATH_CALUDE_game_ends_in_56_rounds_l836_83621

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : Fin 4 → Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game is over (any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Plays the game until it's over -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Theorem stating the game ends in 56 rounds -/
theorem game_ends_in_56_rounds :
  let initialState : GameState := {
    players := λ i =>
      match i with
      | 0 => ⟨17⟩
      | 1 => ⟨16⟩
      | 2 => ⟨15⟩
      | 3 => ⟨14⟩
    rounds := 0
  }
  let finalState := playGame initialState
  finalState.rounds = 56 ∧ 
  finalState.players 3 = ⟨0⟩ ∧
  isGameOver finalState = true :=
by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_56_rounds_l836_83621


namespace NUMINAMATH_CALUDE_smallest_N_proof_l836_83627

/-- Check if a natural number consists only of threes -/
def all_threes (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 3

/-- The smallest natural number N such that 99N consists only of threes -/
def smallest_N : ℕ := 3367

theorem smallest_N_proof :
  (∀ n < smallest_N, ¬(all_threes (99 * n))) ∧
  (all_threes (99 * smallest_N)) :=
sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l836_83627


namespace NUMINAMATH_CALUDE_lineup_combinations_l836_83646

/-- Represents the number of ways to choose a starting lineup in basketball -/
def choose_lineup (total_players : ℕ) (lineup_size : ℕ) 
  (point_guards : ℕ) (shooting_guards : ℕ) (small_forwards : ℕ) 
  (power_center : ℕ) : ℕ :=
  Nat.choose point_guards 1 * 
  Nat.choose shooting_guards 1 * 
  Nat.choose small_forwards 1 * 
  Nat.choose power_center 1 * 
  Nat.choose (power_center - 1) 1

/-- Theorem stating the number of ways to choose a starting lineup -/
theorem lineup_combinations : 
  choose_lineup 12 5 3 2 4 3 = 144 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l836_83646


namespace NUMINAMATH_CALUDE_base_x_is_8_l836_83670

/-- The base of the numeral system in which 1728 (decimal) is represented as 3362 -/
def base_x : ℕ :=
  sorry

/-- The representation of 1728 in base x -/
def representation : List ℕ :=
  [3, 3, 6, 2]

theorem base_x_is_8 :
  (base_x ^ 3 * representation[0]! +
   base_x ^ 2 * representation[1]! +
   base_x ^ 1 * representation[2]! +
   base_x ^ 0 * representation[3]!) = 1728 ∧
  base_x = 8 :=
sorry

end NUMINAMATH_CALUDE_base_x_is_8_l836_83670


namespace NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l836_83684

theorem prime_divides_mn_minus_one (m n p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_order : m < n ∧ n < p)
  (h_div_m : p ∣ m^2 + 1)
  (h_div_n : p ∣ n^2 + 1) :
  p ∣ m * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l836_83684


namespace NUMINAMATH_CALUDE_slopes_equal_necessary_not_sufficient_for_parallel_l836_83688

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define parallel relation
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

-- Theorem statement
theorem slopes_equal_necessary_not_sufficient_for_parallel :
  -- Given two lines
  ∀ (l1 l2 : Line),
  -- l1 has intercept 1
  l1.intercept = 1 →
  -- Necessary condition
  (parallel l1 l2 → l1.slope = l2.slope) ∧
  -- Not sufficient condition
  ∃ l2 : Line, l1.slope = l2.slope ∧ ¬(parallel l1 l2) :=
by
  sorry

end NUMINAMATH_CALUDE_slopes_equal_necessary_not_sufficient_for_parallel_l836_83688


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l836_83655

-- Define the set of numbers
def S : Finset ℕ := {1, 2, 3, 4}

-- Define the number of elements to choose
def r : ℕ := 3

-- Theorem statement
theorem three_digit_numbers_count : Nat.descFactorial (Finset.card S) r = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l836_83655


namespace NUMINAMATH_CALUDE_marker_selection_combinations_l836_83679

theorem marker_selection_combinations : ∀ n r : ℕ, 
  n = 15 → r = 5 → (n.choose r) = 3003 := by
  sorry

end NUMINAMATH_CALUDE_marker_selection_combinations_l836_83679


namespace NUMINAMATH_CALUDE_simplify_expression_l836_83691

/-- Given an expression 3(3x^2 + 4xy) - a(2x^2 + 3xy - 1), if the simplified result
    does not contain y, then a = 4 and the simplified expression is x^2 + 4 -/
theorem simplify_expression (x y : ℝ) (a : ℝ) :
  (∀ y, 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) = 
   3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1)) →
  a = 4 ∧ 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) = x^2 + 4 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l836_83691


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l836_83690

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, 1 < a ∧ a < 2 → a^2 - 3*a ≤ 0) ∧
  (∃ a, a^2 - 3*a ≤ 0 ∧ ¬(1 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l836_83690


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l836_83641

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -5) (hy : y = 2) :
  ((x + 2*y)^2 - (x - 2*y)*(2*y + x)) / (4*y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l836_83641


namespace NUMINAMATH_CALUDE_tournament_rankings_l836_83628

/-- Represents a volleyball team -/
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_match1 : Match)
(saturday_match2 : Match)
(sunday_final : Match)
(sunday_consolation : Match)
(tiebreaker : Match)

/-- Returns the number of possible ranking sequences for a tournament -/
def possibleRankings (t : Tournament) : ℕ :=
  sorry

/-- Theorem: The number of possible ranking sequences is 16 -/
theorem tournament_rankings (t : Tournament) : possibleRankings t = 16 :=
  sorry

end NUMINAMATH_CALUDE_tournament_rankings_l836_83628


namespace NUMINAMATH_CALUDE_complex_equation_solution_l836_83614

theorem complex_equation_solution (z : ℂ) : (z - 2*I = 3 + 7*I) → z = 3 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l836_83614


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l836_83616

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x^2 - 1) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l836_83616


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l836_83606

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l836_83606


namespace NUMINAMATH_CALUDE_puzzle_sum_l836_83613

theorem puzzle_sum (A B C D : Nat) : 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * 1000 + B - (5000 + C * 10 + 9) = 1000 + D * 100 + 90 + 3 →
  A + B + C + D = 18 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sum_l836_83613


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l836_83693

theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → (k / x) = -1/2 ↔ x = 4) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l836_83693


namespace NUMINAMATH_CALUDE_probability_theorem_l836_83609

def club_sizes : List Nat := [6, 9, 11, 13]

def probability_select_officers (sizes : List Nat) : Rat :=
  let total_probability := sizes.map (fun n => 1 / Nat.choose n 3)
  (1 / sizes.length) * total_probability.sum

theorem probability_theorem :
  probability_select_officers club_sizes = 905 / 55440 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l836_83609


namespace NUMINAMATH_CALUDE_joseph_cards_percentage_l836_83692

theorem joseph_cards_percentage (initial_cards : ℕ) 
  (brother_fraction : ℚ) (friend_cards : ℕ) : 
  initial_cards = 16 →
  brother_fraction = 3/8 →
  friend_cards = 2 →
  (initial_cards - (initial_cards * brother_fraction).floor - friend_cards) / initial_cards * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_joseph_cards_percentage_l836_83692


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l836_83680

def vector_AB (x : ℝ) : ℝ × ℝ := (x, 2*x)
def vector_AC (x : ℝ) : ℝ × ℝ := (-3*x, 2)

def is_obtuse_angle (x : ℝ) : Prop :=
  let dot_product := (vector_AB x).1 * (vector_AC x).1 + (vector_AB x).2 * (vector_AC x).2
  dot_product < 0 ∧ x ≠ -1/3

def range_of_x : Set ℝ :=
  {x | x < -1/3 ∨ (-1/3 < x ∧ x < 0) ∨ x > 4/3}

theorem obtuse_angle_range :
  ∀ x, is_obtuse_angle x ↔ x ∈ range_of_x :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l836_83680


namespace NUMINAMATH_CALUDE_bug_return_probability_l836_83663

/-- Probability of a bug returning to the starting vertex of a regular tetrahedron after n steps -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 - P (n - 1)) / 3

/-- The regular tetrahedron has edge length 1 and the bug starts at vertex A -/
theorem bug_return_probability :
  P 10 = 4921 / 59049 :=
sorry

end NUMINAMATH_CALUDE_bug_return_probability_l836_83663


namespace NUMINAMATH_CALUDE_gift_wrap_sale_total_l836_83652

/-- Calculates the total amount of money collected from selling gift wrap rolls -/
def total_amount_collected (total_rolls : ℕ) (print_rolls : ℕ) (solid_price : ℚ) (print_price : ℚ) : ℚ :=
  (print_rolls * print_price) + ((total_rolls - print_rolls) * solid_price)

/-- Proves that the total amount collected from selling gift wrap rolls is $2340 -/
theorem gift_wrap_sale_total : 
  total_amount_collected 480 210 4 6 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrap_sale_total_l836_83652


namespace NUMINAMATH_CALUDE_exactly_one_correct_proposition_l836_83673

open Real

theorem exactly_one_correct_proposition : ∃! n : Nat, n = 1 ∧
  (¬ (∀ x : ℝ, (x^2 < 1 → -1 < x ∧ x < 1) ↔ ((x > 1 ∨ x < -1) → x^2 > 1))) ∧
  (¬ ((∀ x : ℝ, sin x ≤ 1) ∧ (∀ a b : ℝ, a < b → a^2 < b^2))) ∧
  ((∀ x : ℝ, ¬(x^2 - x > 0)) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  (¬ (∀ x : ℝ, x^2 > 4 → x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_correct_proposition_l836_83673


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l836_83611

theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 36 → x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l836_83611


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l836_83601

/-- Partnership investment problem -/
theorem partnership_investment_ratio 
  (a b c : ℝ) -- Investments of A, B, and C
  (total_profit b_share : ℝ) -- Total profit and B's share
  (h1 : a = 3 * b) -- A invests 3 times as much as B
  (h2 : b < c) -- B invests some fraction of what C invests
  (h3 : total_profit = 8800) -- Total profit is 8800
  (h4 : b_share = 1600) -- B's share is 1600
  (h5 : b_share / total_profit = b / (a + b + c)) -- Profit distribution ratio
  : b / c = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l836_83601


namespace NUMINAMATH_CALUDE_investment_decrease_l836_83694

/-- Given an initial investment that increases by 50% in the first year
    and has a net increase of 4.999999999999982% after two years,
    prove that the percentage decrease in the second year is 30%. -/
theorem investment_decrease (initial : ℝ) (initial_pos : initial > 0) :
  let first_year := initial * 1.5
  let final := initial * 1.04999999999999982
  let second_year_factor := final / first_year
  second_year_factor = 0.7 := by sorry

end NUMINAMATH_CALUDE_investment_decrease_l836_83694


namespace NUMINAMATH_CALUDE_max_value_L_in_triangle_l836_83632

/-- The function L(x, y) = -2x + y -/
def L (x y : ℝ) : ℝ := -2*x + y

/-- The triangle ABC with vertices A(-2, -1), B(0, 1), and C(2, -1) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
       p.1 = -2*a + 0*b + 2*c ∧
       p.2 = -1*a + 1*b - 1*c}

theorem max_value_L_in_triangle :
  ∃ (max : ℝ), max = 3 ∧ 
  ∀ (x y : ℝ), (x, y) ∈ triangle_ABC → L x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_L_in_triangle_l836_83632


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l836_83637

-- Define the quadratic function
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (∀ x ∈ Set.Ioo 2 5, quadratic a b x > 0) ∧ quadratic a b 1 = 1 →
  a ∈ Set.Ioi (3 - 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ a ∈ Set.Icc (-2) (-1), quadratic a (-a-1) x > 0) ∧ quadratic 0 (-1) 1 = 1 →
  x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4) :=
sorry

-- Part 3
theorem min_value_ratio (a b : ℝ) :
  b > 0 ∧ (∀ x : ℝ, quadratic a b x ≥ 0) →
  (a + 2) / b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l836_83637


namespace NUMINAMATH_CALUDE_ski_boat_rental_cost_l836_83608

/-- The cost per hour to rent a ski boat -/
def ski_boat_cost_per_hour : ℝ := 40

/-- The cost to rent a sailboat per day -/
def sailboat_cost_per_day : ℝ := 60

/-- The number of hours per day the boats were rented -/
def hours_per_day : ℝ := 3

/-- The number of days the boats were rented -/
def days_rented : ℝ := 2

/-- The additional cost Aldrich paid compared to Ken -/
def additional_cost : ℝ := 120

theorem ski_boat_rental_cost : 
  ski_boat_cost_per_hour * hours_per_day * days_rented = 
  sailboat_cost_per_day * days_rented + additional_cost := by
  sorry

end NUMINAMATH_CALUDE_ski_boat_rental_cost_l836_83608


namespace NUMINAMATH_CALUDE_opposite_gender_officers_l836_83644

theorem opposite_gender_officers (boys girls : ℕ) (h1 : boys = 18) (h2 : girls = 12) :
  boys * girls + girls * boys = 432 := by
  sorry

end NUMINAMATH_CALUDE_opposite_gender_officers_l836_83644


namespace NUMINAMATH_CALUDE_triangle_inequality_l836_83697

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l836_83697


namespace NUMINAMATH_CALUDE_hotdogs_served_during_dinner_l836_83686

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := 11

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := total_hotdogs - lunch_hotdogs

theorem hotdogs_served_during_dinner : dinner_hotdogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_served_during_dinner_l836_83686


namespace NUMINAMATH_CALUDE_fox_distribution_l836_83622

/-- Fox Distribution Problem -/
theorem fox_distribution (m : ℕ) (a : ℝ) (x y : ℝ) :
  (∀ n : ℕ, n * a + (x - (n - 1) * y - n * a) / m = y) →
  x = (m - 1)^2 * a ∧ 
  y = (m - 1) * a ∧ 
  m - 1 > 0 := by
sorry

end NUMINAMATH_CALUDE_fox_distribution_l836_83622


namespace NUMINAMATH_CALUDE_luncheon_tables_l836_83645

theorem luncheon_tables (invited : ℕ) (no_show : ℕ) (per_table : ℕ) 
  (h1 : invited = 18) 
  (h2 : no_show = 12) 
  (h3 : per_table = 3) : 
  (invited - no_show) / per_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_tables_l836_83645


namespace NUMINAMATH_CALUDE_theresa_shared_crayons_l836_83643

/-- Represents the number of crayons Theresa has initially -/
def initial_crayons : ℕ := 32

/-- Represents the number of crayons Theresa has after sharing -/
def final_crayons : ℕ := 19

/-- Represents the number of crayons Theresa shared -/
def shared_crayons : ℕ := initial_crayons - final_crayons

theorem theresa_shared_crayons : 
  shared_crayons = initial_crayons - final_crayons :=
by sorry

end NUMINAMATH_CALUDE_theresa_shared_crayons_l836_83643


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l836_83639

/-- Given two points P₁(x₁, y₁) and P₂(x₂, y₂) on the line y = -3x + 4,
    if x₁ < x₂, then y₁ > y₂ -/
theorem linear_function_point_relation (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -3 * x₁ + 4 →
  y₂ = -3 * x₂ + 4 →
  x₁ < x₂ →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_point_relation_l836_83639


namespace NUMINAMATH_CALUDE_equation_solution_l836_83685

theorem equation_solution (x : ℝ) : 
  (x = (((-1 + Real.sqrt 21) / 2) ^ 3) ∨ x = (((-1 - Real.sqrt 21) / 2) ^ 3)) →
  6 * x^(1/3) - 3 * (x / x^(2/3)) + 2 * x^(2/3) = 10 + x^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l836_83685


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l836_83675

theorem no_infinite_prime_sequence : 
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧ 
    (∀ n, p n < p (n + 1)) ∧
    (∀ k, p (k + 1) = 2 * p k + 1 ∨ p (k + 1) = 2 * p k - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l836_83675


namespace NUMINAMATH_CALUDE_five_digit_automorphic_number_l836_83623

theorem five_digit_automorphic_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n^2 % 100000 = n :=
sorry

end NUMINAMATH_CALUDE_five_digit_automorphic_number_l836_83623


namespace NUMINAMATH_CALUDE_total_water_intake_l836_83625

def morning_water : Real := 0.26
def afternoon_water : Real := 0.37

theorem total_water_intake : morning_water + afternoon_water = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_total_water_intake_l836_83625


namespace NUMINAMATH_CALUDE_second_question_percentage_l836_83677

theorem second_question_percentage 
  (first_correct : Real) 
  (neither_correct : Real) 
  (both_correct : Real)
  (h1 : first_correct = 0.75)
  (h2 : neither_correct = 0.2)
  (h3 : both_correct = 0.6) :
  ∃ second_correct : Real, 
    second_correct = 0.65 ∧ 
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_second_question_percentage_l836_83677


namespace NUMINAMATH_CALUDE_cyclic_symmetric_count_l836_83620

-- Definition of cyclic symmetric expression
def is_cyclic_symmetric (σ : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, σ a b c = σ b c a ∧ σ a b c = σ c a b

-- Define the three expressions
def σ₁ (a b c : ℝ) : ℝ := a * b * c
def σ₂ (a b c : ℝ) : ℝ := a^2 - b^2 + c^2
noncomputable def σ₃ (A B C : ℝ) : ℝ := Real.cos C * Real.cos (A - B) - (Real.cos C)^2

-- Theorem statement
theorem cyclic_symmetric_count :
  (is_cyclic_symmetric σ₁) ∧
  ¬(is_cyclic_symmetric σ₂) ∧
  (is_cyclic_symmetric σ₃) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_symmetric_count_l836_83620


namespace NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l836_83687

theorem smallest_prime_20_less_than_square : 
  ∃ (n : ℕ), 
    5 = n^2 - 20 ∧ 
    Prime 5 ∧ 
    (∀ (m : ℕ) (p : ℕ), p < 5 → p = m^2 - 20 → ¬ Prime p) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l836_83687


namespace NUMINAMATH_CALUDE_initial_distance_between_students_l836_83657

theorem initial_distance_between_students
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ)
  (h1 : speed1 = 1.6)
  (h2 : speed2 = 1.9)
  (h3 : time = 100)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0)
  (h6 : time > 0) :
  speed1 * time + speed2 * time = 350 := by
sorry

end NUMINAMATH_CALUDE_initial_distance_between_students_l836_83657


namespace NUMINAMATH_CALUDE_jenny_investment_l836_83682

theorem jenny_investment (total : ℝ) (ratio : ℝ) (real_estate : ℝ) : 
  total = 220000 →
  ratio = 7 →
  real_estate = ratio * (total / (ratio + 1)) →
  real_estate = 192500 := by
sorry

end NUMINAMATH_CALUDE_jenny_investment_l836_83682


namespace NUMINAMATH_CALUDE_exists_rational_with_prime_multiples_l836_83604

theorem exists_rational_with_prime_multiples : ∃ x : ℚ, 
  (Nat.Prime (Int.natAbs (Int.floor (10 * x)))) ∧ 
  (Nat.Prime (Int.natAbs (Int.floor (15 * x)))) := by
  sorry

end NUMINAMATH_CALUDE_exists_rational_with_prime_multiples_l836_83604


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_two_l836_83633

theorem trigonometric_sum_equals_sqrt_two :
  (Real.cos (2 * π / 180)) / (Real.sin (47 * π / 180)) +
  (Real.cos (88 * π / 180)) / (Real.sin (133 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_two_l836_83633


namespace NUMINAMATH_CALUDE_min_sum_squares_l836_83659

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 2) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 2 → x^2 + y^2 + z^2 ≥ m ∧ a^2 + b^2 + c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l836_83659


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l836_83666

/-- The area of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (w : ℝ) (h : ℝ) : ℝ :=
  4 * (w + h)^2

/-- Theorem: The area of a square sheet of wrapping paper required to wrap a rectangular box
    with dimensions 2w × w × h, such that the corners of the paper meet at the center of the
    top of the box, is equal to 4(w + h)^2. -/
theorem wrapping_paper_area_theorem (w : ℝ) (h : ℝ) 
    (hw : w > 0) (hh : h > 0) : 
    wrapping_paper_area w h = 4 * (w + h)^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l836_83666


namespace NUMINAMATH_CALUDE_inequality_proof_l836_83638

theorem inequality_proof (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hprod : x * y * z = 1) :
  (∀ (a b : ℝ), 
    ((a = 0 ∧ b = 1) ∨ 
     (a = 1 ∧ b = 0) ∨ 
     (a + b = 1 ∧ a > 0 ∧ b > 0)) →
    1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b)) ≥ 3) ∧
  (x = 1 ∧ y = 1 ∧ z = 1 →
    ∀ (a b : ℝ), a + b = 1 ∧ a > 0 ∧ b > 0 →
      1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b)) = 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l836_83638


namespace NUMINAMATH_CALUDE_podcast_storage_theorem_l836_83665

def podcast_duration : ℕ := 837
def cd_capacity : ℕ := 75

theorem podcast_storage_theorem :
  let num_cds : ℕ := (podcast_duration + cd_capacity - 1) / cd_capacity
  let audio_per_cd : ℚ := podcast_duration / num_cds
  audio_per_cd = 69.75 := by sorry

end NUMINAMATH_CALUDE_podcast_storage_theorem_l836_83665


namespace NUMINAMATH_CALUDE_function_identity_l836_83640

def is_positive_integer (n : ℤ) : Prop := 0 < n

structure PositiveInteger where
  val : ℤ
  pos : is_positive_integer val

def PositiveIntegerFunction := PositiveInteger → PositiveInteger

theorem function_identity (f : PositiveIntegerFunction) : 
  (∀ (a b : PositiveInteger), ∃ (k : ℤ), a.val ^ 2 + (f a).val * (f b).val = ((f a).val + b.val) * k) →
  (∀ (n : PositiveInteger), (f n).val = n.val) := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l836_83640


namespace NUMINAMATH_CALUDE_stratified_sampling_l836_83603

theorem stratified_sampling (total_A total_B sample_A : ℕ) 
  (h1 : total_A = 800)
  (h2 : total_B = 500)
  (h3 : sample_A = 48) :
  (total_B : ℚ) / (total_A + total_B) * sample_A = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l836_83603


namespace NUMINAMATH_CALUDE_rohit_final_position_l836_83612

/-- Represents a 2D position --/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a direction --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Rohit's movement --/
def move (p : Position) (d : Direction) (distance : ℝ) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + distance⟩
  | Direction.East => ⟨p.x + distance, p.y⟩
  | Direction.South => ⟨p.x, p.y - distance⟩
  | Direction.West => ⟨p.x - distance, p.y⟩

/-- Represents a left turn --/
def turnLeft (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.East => Direction.North
  | Direction.South => Direction.East
  | Direction.West => Direction.South

/-- Represents a right turn --/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

theorem rohit_final_position : 
  let start : Position := ⟨0, 0⟩
  let p1 := move start Direction.South 25
  let d1 := turnLeft Direction.South
  let p2 := move p1 d1 20
  let d2 := turnLeft d1
  let p3 := move p2 d2 25
  let d3 := turnRight d2
  let final := move p3 d3 15
  final = ⟨35, 0⟩ := by sorry

end NUMINAMATH_CALUDE_rohit_final_position_l836_83612


namespace NUMINAMATH_CALUDE_ellipse_a_range_l836_83699

theorem ellipse_a_range (a b : ℝ) (e : ℝ) :
  a > b ∧ b > 0 ∧
  e ∈ Set.Icc (1 / Real.sqrt 3) (1 / Real.sqrt 2) ∧
  (∃ (M N : ℝ × ℝ),
    (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
    (N.1^2 / a^2 + N.2^2 / b^2 = 1) ∧
    (M.2 = -M.1 + 1) ∧
    (N.2 = -N.1 + 1) ∧
    (M.1 * N.1 + M.2 * N.2 = 0)) →
  Real.sqrt 5 / 2 ≤ a ∧ a ≤ Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l836_83699


namespace NUMINAMATH_CALUDE_root_properties_l836_83672

theorem root_properties (a b : ℝ) :
  (a - b)^3 + 3*a*b*(a - b) + b^3 - a^3 = 0 ∧
  (∀ a : ℝ, (a - 1)^3 - a*(a - 1)^2 + 1 = 0 ↔ a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_root_properties_l836_83672


namespace NUMINAMATH_CALUDE_gcd_204_85_l836_83689

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l836_83689


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l836_83631

theorem sqrt_fraction_equality : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) = 
  (-2 * Real.sqrt 7 + Real.sqrt 3 + Real.sqrt 5) / 59 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l836_83631


namespace NUMINAMATH_CALUDE_nineteen_only_vegetarian_l836_83678

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat only vegetarian -/
def only_vegetarian (f : FamilyDiet) : ℕ :=
  f.total_veg - f.both_veg_and_non_veg

/-- Theorem stating that 19 people eat only vegetarian in the given family -/
theorem nineteen_only_vegetarian (f : FamilyDiet) 
  (h1 : f.only_non_veg = 9)
  (h2 : f.both_veg_and_non_veg = 12)
  (h3 : f.total_veg = 31) :
  only_vegetarian f = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_only_vegetarian_l836_83678


namespace NUMINAMATH_CALUDE_bobs_money_l836_83635

theorem bobs_money (X : ℝ) :
  X > 0 →
  let day1_remainder := X / 2
  let day2_remainder := day1_remainder * 4 / 5
  let day3_remainder := day2_remainder * 5 / 8
  day3_remainder = 20 →
  X = 80 :=
by sorry

end NUMINAMATH_CALUDE_bobs_money_l836_83635


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l836_83619

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Nat.Prime p ∧ 
    p ∣ (20^3 + 15^4 - 10^5) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (20^3 + 15^4 - 10^5) → q ≤ p ∧ 
    p = 113 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l836_83619


namespace NUMINAMATH_CALUDE_mean_less_than_median_l836_83607

/-- Represents the data for days missed and number of students -/
def days_missed_data : List (Nat × Nat) := [(0, 5), (1, 3), (2, 8), (3, 2), (4, 1), (5, 1)]

/-- Total number of students in the classroom -/
def total_students : Nat := 20

/-- Calculates the median number of days missed -/
def median (data : List (Nat × Nat)) (total : Nat) : ℚ :=
  sorry

/-- Calculates the mean number of days missed -/
def mean (data : List (Nat × Nat)) (total : Nat) : ℚ :=
  sorry

theorem mean_less_than_median :
  mean days_missed_data total_students = median days_missed_data total_students - 3/10 := by
  sorry

end NUMINAMATH_CALUDE_mean_less_than_median_l836_83607


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l836_83653

/-- Given a geometric sequence {a_n} where 2a_3, a_5/2, 3a_1 forms an arithmetic sequence,
    prove that (a_2 + a_5) / (a_9 + a_6) = 1/9 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) (hq : q > 0) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (2 * a 3 - a 5 / 2 = a 5 / 2 - 3 * a 1) →  -- arithmetic sequence condition
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l836_83653


namespace NUMINAMATH_CALUDE_sin_product_equality_l836_83648

theorem sin_product_equality : 
  Real.sin (18 * π / 180) * Real.sin (30 * π / 180) * Real.sin (60 * π / 180) * Real.sin (72 * π / 180) = 
  (Real.sqrt 3 / 8) * Real.sin (36 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l836_83648


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l836_83602

-- Define the standard normal distribution
def standard_normal (ξ : ℝ → ℝ) : Prop :=
  ∃ (μ σ : ℝ), σ > 0 ∧ ∀ x, ξ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

-- Define the probability measure
noncomputable def P (A : Set ℝ) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_probability 
  (ξ : ℝ → ℝ) 
  (h1 : standard_normal ξ) 
  (h2 : P {x | ξ x > 1} = 1/4) : 
  P {x | -1 < ξ x ∧ ξ x < 1} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l836_83602


namespace NUMINAMATH_CALUDE_jorge_goals_last_season_l836_83656

/-- The number of goals Jorge scored last season -/
def goals_last_season : ℕ := sorry

/-- The number of goals Jorge scored this season -/
def goals_this_season : ℕ := 187

/-- The total number of goals Jorge scored -/
def total_goals : ℕ := 343

/-- Theorem stating that Jorge scored 156 goals last season -/
theorem jorge_goals_last_season :
  goals_last_season = total_goals - goals_this_season ∧
  goals_last_season = 156 := by sorry

end NUMINAMATH_CALUDE_jorge_goals_last_season_l836_83656


namespace NUMINAMATH_CALUDE_equation_solution_l836_83634

theorem equation_solution :
  ∃ x : ℝ, 38 + 2 * x^3 = 1250 ∧ x = (606 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l836_83634


namespace NUMINAMATH_CALUDE_smaller_integer_proof_l836_83629

theorem smaller_integer_proof (x y : ℤ) (h1 : x + y = -9) (h2 : y - x = 1) : x = -5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_proof_l836_83629


namespace NUMINAMATH_CALUDE_new_shoes_cost_calculation_l836_83650

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 14.50

/-- The percentage increase in cost per year for new shoes compared to repaired shoes -/
def percentage_increase : ℝ := 0.10344827586206897

/-- The cost of new shoes -/
def new_shoes_cost : ℝ := 2 * (repair_cost + percentage_increase * repair_cost)

theorem new_shoes_cost_calculation :
  new_shoes_cost = 32 := by sorry

end NUMINAMATH_CALUDE_new_shoes_cost_calculation_l836_83650


namespace NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l836_83642

theorem sin_three_pi_half_plus_alpha (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  Real.sin (3 * Real.pi / 2 + α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l836_83642


namespace NUMINAMATH_CALUDE_train_length_l836_83654

theorem train_length (speed : ℝ) (train_length : ℝ) : 
  (train_length + 130) / 15 = speed ∧ 
  (train_length + 250) / 20 = speed → 
  train_length = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l836_83654


namespace NUMINAMATH_CALUDE_shoe_price_ratio_l836_83668

theorem shoe_price_ratio (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := (2/3) * selling_price
  cost_price / marked_price = 1/2 := by sorry

end NUMINAMATH_CALUDE_shoe_price_ratio_l836_83668


namespace NUMINAMATH_CALUDE_unique_card_combination_l836_83669

/-- Represents the colors of the cards --/
inductive Color
  | Green
  | Yellow
  | Blue
  | Red

/-- Represents the set of cards with their numbers --/
def CardSet := Color → Nat

/-- Checks if a number is a valid card number (positive integer less than 10) --/
def isValidCardNumber (n : Nat) : Prop := 0 < n ∧ n < 10

/-- Checks if all numbers in a card set are valid --/
def allValidNumbers (cards : CardSet) : Prop :=
  ∀ c, isValidCardNumber (cards c)

/-- Checks if all numbers in a card set are different --/
def allDifferentNumbers (cards : CardSet) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → cards c₁ ≠ cards c₂

/-- Checks if the product of green and yellow numbers is the green number --/
def greenYellowProduct (cards : CardSet) : Prop :=
  cards Color.Green * cards Color.Yellow = cards Color.Green

/-- Checks if the blue number is the same as the red number --/
def blueRedSame (cards : CardSet) : Prop :=
  cards Color.Blue = cards Color.Red

/-- Checks if the product of red and blue numbers forms a two-digit number
    with green and yellow digits in that order --/
def redBlueProductCondition (cards : CardSet) : Prop :=
  cards Color.Red * cards Color.Blue = 10 * cards Color.Green + cards Color.Yellow

/-- The main theorem stating that the only valid combination is 8, 1, 9, 9 --/
theorem unique_card_combination :
  ∀ cards : CardSet,
    allValidNumbers cards →
    allDifferentNumbers cards →
    greenYellowProduct cards →
    blueRedSame cards →
    redBlueProductCondition cards →
    (cards Color.Green = 8 ∧
     cards Color.Yellow = 1 ∧
     cards Color.Blue = 9 ∧
     cards Color.Red = 9) :=
by sorry

end NUMINAMATH_CALUDE_unique_card_combination_l836_83669


namespace NUMINAMATH_CALUDE_system_solution_l836_83630

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l836_83630


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l836_83681

theorem quadratic_equation_solution : ∃ y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l836_83681


namespace NUMINAMATH_CALUDE_unique_solution_l836_83615

theorem unique_solution (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 20)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 150) :
  a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l836_83615


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l836_83636

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := n

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define T_n as the sum of the first n terms of {1/S_n}
def T (n : ℕ) : ℚ := 2 * (1 - 1 / (n + 1))

theorem arithmetic_geometric_sequence_properties :
  -- The sequence {a_n} is arithmetic with common difference 1
  (∀ n : ℕ, a (n + 1) - a n = 1) ∧
  -- a_1, a_3, a_9 form a geometric sequence
  (a 3)^2 = a 1 * a 9 →
  -- Prove the following:
  (-- 1. General term formula
   (∀ n : ℕ, n ≥ 1 → a n = n) ∧
   -- 2. Sum of first n terms
   (∀ n : ℕ, n ≥ 1 → S n = n * (n + 1) / 2) ∧
   -- 3. T_n < 2
   (∀ n : ℕ, n ≥ 1 → T n < 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l836_83636


namespace NUMINAMATH_CALUDE_oliver_money_l836_83605

/-- 
Given that Oliver:
- Had x dollars in January
- Spent 4 dollars by March
- Received 32 dollars from his mom
- Then had 61 dollars

Prove that x must equal 33.
-/
theorem oliver_money (x : ℤ) 
  (spent : ℤ) 
  (received : ℤ) 
  (final_amount : ℤ) 
  (h1 : spent = 4)
  (h2 : received = 32)
  (h3 : final_amount = 61)
  (h4 : x - spent + received = final_amount) : 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_l836_83605


namespace NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l836_83658

theorem night_day_crew_loading_ratio :
  ∀ (D N B : ℚ),
  N = (2/3) * D →                     -- Night crew has 2/3 as many workers as day crew
  (2/3) * B = D * (B / D) →           -- Day crew loaded 2/3 of all boxes
  (1/3) * B = N * (B / N) →           -- Night crew loaded 1/3 of all boxes
  (B / N) / (B / D) = 3/4              -- Ratio of boxes loaded by each night worker to each day worker
:= by sorry

end NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l836_83658


namespace NUMINAMATH_CALUDE_f_three_pow_ge_f_two_pow_l836_83674

/-- A quadratic function f(x) = ax^2 + bx + c with a > 0 and symmetric about x = 1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) ≥ f(2^x) for all x ∈ ℝ -/
theorem f_three_pow_ge_f_two_pow (a b c : ℝ) (h_a : a > 0) 
  (h_sym : ∀ x, f a b c (1 - x) = f a b c (1 + x)) :
  ∀ x, f a b c (3^x) ≥ f a b c (2^x) := by
  sorry

end NUMINAMATH_CALUDE_f_three_pow_ge_f_two_pow_l836_83674


namespace NUMINAMATH_CALUDE_tax_diminished_percentage_l836_83610

/-- Proves that a 12% increase in consumption and a 23.84% decrease in revenue
    implies a 32% decrease in tax rate. -/
theorem tax_diminished_percentage
  (original_tax : ℝ)
  (original_consumption : ℝ)
  (new_tax : ℝ)
  (new_consumption : ℝ)
  (original_revenue : ℝ)
  (new_revenue : ℝ)
  (h1 : original_tax > 0)
  (h2 : original_consumption > 0)
  (h3 : new_consumption = original_consumption * 1.12)
  (h4 : new_revenue = original_revenue * 0.7616)
  (h5 : original_revenue = original_tax * original_consumption)
  (h6 : new_revenue = new_tax * new_consumption) :
  new_tax = original_tax * 0.68 :=
sorry

end NUMINAMATH_CALUDE_tax_diminished_percentage_l836_83610


namespace NUMINAMATH_CALUDE_yellow_to_red_ratio_l836_83698

/-- Represents the number of chairs of each color in Susan's house. -/
structure ChairCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- The conditions of the chair problem in Susan's house. -/
def susansChairs : ChairCounts → Prop := fun c =>
  c.red = 5 ∧
  c.blue = c.yellow - 2 ∧
  c.red + c.yellow + c.blue = 43

/-- The theorem stating the ratio of yellow to red chairs. -/
theorem yellow_to_red_ratio (c : ChairCounts) (h : susansChairs c) :
  c.yellow / c.red = 4 := by
  sorry

#check yellow_to_red_ratio

end NUMINAMATH_CALUDE_yellow_to_red_ratio_l836_83698


namespace NUMINAMATH_CALUDE_tates_total_education_duration_l836_83661

def normal_high_school_duration : ℕ := 4

def tates_high_school_duration : ℕ := normal_high_school_duration - 1

def tates_college_duration : ℕ := 3 * tates_high_school_duration

theorem tates_total_education_duration :
  tates_high_school_duration + tates_college_duration = 12 := by
  sorry

end NUMINAMATH_CALUDE_tates_total_education_duration_l836_83661


namespace NUMINAMATH_CALUDE_ellipse_tangent_max_area_l836_83651

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the xy-plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a line is tangent to an ellipse -/
def isTangent (l : Line) (e : Ellipse) : Prop :=
  sorry -- Definition of tangent line to ellipse

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry -- Formula for triangle area

/-- The main theorem -/
theorem ellipse_tangent_max_area
  (e : Ellipse)
  (A B C : Point)
  (h1 : e.a^2 = 1 ∧ e.b^2 = 3)
  (h2 : A.x = 1 ∧ A.y = 1)
  (h3 : pointOnEllipse A e)
  (h4 : pointOnEllipse B e)
  (h5 : pointOnEllipse C e)
  (h6 : ∃ (l : Line), isTangent l e ∧ l.a * B.x + l.b * B.y + l.c = 0 ∧ l.a * C.x + l.b * C.y + l.c = 0)
  (h7 : ∀ (B' C' : Point), pointOnEllipse B' e → pointOnEllipse C' e →
        triangleArea A B' C' ≤ triangleArea A B C) :
  ∃ (l : Line), l.a = 1 ∧ l.b = 3 ∧ l.c = 2 ∧
    isTangent l e ∧
    l.a * B.x + l.b * B.y + l.c = 0 ∧
    l.a * C.x + l.b * C.y + l.c = 0 ∧
    triangleArea A B C = 3 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_max_area_l836_83651


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l836_83660

/-- The number of ways to arrange 3 teachers and 3 students in a row, such that no two students are adjacent -/
def arrangementCount : ℕ := 144

/-- The number of teachers -/
def teacherCount : ℕ := 3

/-- The number of students -/
def studentCount : ℕ := 3

/-- The number of spots available for students (always one more than the number of teachers) -/
def studentSpots : ℕ := teacherCount + 1

theorem correct_arrangement_count :
  arrangementCount = 
    (Nat.factorial teacherCount) * 
    (Nat.choose studentSpots studentCount) * 
    (Nat.factorial studentCount) :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l836_83660


namespace NUMINAMATH_CALUDE_brian_oranges_l836_83647

theorem brian_oranges (someone_oranges : ℕ) (brian_difference : ℕ) 
  (h1 : someone_oranges = 12)
  (h2 : brian_difference = 0) : 
  someone_oranges - brian_difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_brian_oranges_l836_83647


namespace NUMINAMATH_CALUDE_archer_probability_l836_83695

theorem archer_probability (p_a p_b : ℝ) (h_p_a : p_a = 1/3) (h_p_b : p_b = 1/2) :
  1 - p_a * p_b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_archer_probability_l836_83695


namespace NUMINAMATH_CALUDE_bus_problem_solution_l836_83600

def bus_problem (initial : ℕ) (stop1_off : ℕ) (stop2_off stop2_on : ℕ) (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off - stop2_off + stop2_on - stop3_off + stop3_on

theorem bus_problem_solution :
  bus_problem 50 15 8 2 4 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_solution_l836_83600


namespace NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l836_83683

theorem cube_root_of_one_sixty_fourth (x : ℝ) : x^3 = 1/64 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l836_83683


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_volume_l836_83624

/-- The volume of a regular hexagonal pyramid with base side length a and lateral surface area 10 times larger than the base area -/
theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) : 
  let base_area := (3 * Real.sqrt 3 / 2) * a^2
  let lateral_area := 10 * base_area
  let height := (3 * a * Real.sqrt 33) / 2
  let volume := (1 / 3) * base_area * height
  volume = (9 * a^3 * Real.sqrt 11) / 4 := by
sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_volume_l836_83624
