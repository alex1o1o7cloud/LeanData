import Mathlib

namespace NUMINAMATH_CALUDE_vanessa_age_proof_l1105_110548

def guesses : List Nat := [32, 34, 36, 40, 42, 45, 48, 52, 55, 58]

def vanessaAge : Nat := 53

theorem vanessa_age_proof :
  -- At least half of the guesses are too low
  (guesses.filter (· < vanessaAge)).length ≥ guesses.length / 2 ∧
  -- Three guesses are off by one
  (guesses.filter (fun x => x = vanessaAge - 1 ∨ x = vanessaAge + 1)).length = 3 ∧
  -- Vanessa's age is a prime number
  Nat.Prime vanessaAge ∧
  -- One guess is exactly correct
  guesses.contains vanessaAge ∧
  -- Vanessa's age is 53
  vanessaAge = 53 := by
  sorry

#eval vanessaAge

end NUMINAMATH_CALUDE_vanessa_age_proof_l1105_110548


namespace NUMINAMATH_CALUDE_shadow_length_ratio_l1105_110514

theorem shadow_length_ratio (α β : Real) 
  (h1 : Real.tan (α - β) = 1 / 3)
  (h2 : Real.tan β = 1) :
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_ratio_l1105_110514


namespace NUMINAMATH_CALUDE_no_valid_numbers_l1105_110520

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def digitSum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem no_valid_numbers : ¬ ∃ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  digitSum n = 27 ∧
  isEven ((n / 10) % 10) ∧
  isEven n :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l1105_110520


namespace NUMINAMATH_CALUDE_one_less_than_negative_one_l1105_110568

theorem one_less_than_negative_one : (-1 : ℤ) - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_negative_one_l1105_110568


namespace NUMINAMATH_CALUDE_det2022_2023_2021_2022_solve_det_eq_16_l1105_110589

-- Definition of second-order determinant
def det2 (a b c d : ℤ) : ℤ := a * d - b * c

-- Theorem 1
theorem det2022_2023_2021_2022 : det2 2022 2023 2021 2022 = 1 := by sorry

-- Theorem 2
theorem solve_det_eq_16 (m : ℤ) : det2 (m + 2) (m - 2) (m - 2) (m + 2) = 16 → m = 2 := by sorry

end NUMINAMATH_CALUDE_det2022_2023_2021_2022_solve_det_eq_16_l1105_110589


namespace NUMINAMATH_CALUDE_point_N_coordinates_l1105_110563

/-- Given point M(5, -6) and vector a = (1, -2), if vector MN = -3 * vector a,
    then the coordinates of point N are (2, 0). -/
theorem point_N_coordinates :
  let M : ℝ × ℝ := (5, -6)
  let a : ℝ × ℝ := (1, -2)
  let N : ℝ × ℝ := (x, y)
  (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2) →
  N = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l1105_110563


namespace NUMINAMATH_CALUDE_sequence_bound_l1105_110549

theorem sequence_bound (x : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → |x i - x j| ≥ 1 / (i + j))
  (h2 : ∀ (i : ℕ), 0 ≤ x i ∧ x i ≤ c) : 
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l1105_110549


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_3_l1105_110524

/-- Set A definition -/
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

/-- Set B definition -/
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

/-- Theorem stating that if B is a subset of A, then m ≤ 3 -/
theorem subset_implies_m_leq_3 (m : ℝ) (h : B m ⊆ A) : m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_3_l1105_110524


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1105_110509

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 8 ∣ m → 6 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1105_110509


namespace NUMINAMATH_CALUDE_maple_taller_than_pine_l1105_110501

-- Define the heights of the trees
def pine_height : ℚ := 15 + 1/4
def maple_height : ℚ := 20 + 2/3

-- Define the height difference
def height_difference : ℚ := maple_height - pine_height

-- Theorem to prove
theorem maple_taller_than_pine :
  height_difference = 5 + 5/12 := by sorry

end NUMINAMATH_CALUDE_maple_taller_than_pine_l1105_110501


namespace NUMINAMATH_CALUDE_fiftycentchange_l1105_110567

/-- Represents the different types of U.S. coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Represents a combination of coins --/
def CoinCombination := List Coin

/-- Checks if a coin combination is valid for 50 cents --/
def isValidCombination (c : CoinCombination) : Bool := sorry

/-- Counts the number of quarters in a combination --/
def countQuarters (c : CoinCombination) : Nat := sorry

/-- Generates all valid coin combinations for 50 cents --/
def allCombinations : List CoinCombination := sorry

/-- The main theorem stating that there are 47 ways to make change for 50 cents --/
theorem fiftycentchange : 
  (allCombinations.filter (fun c => isValidCombination c ∧ countQuarters c ≤ 1)).length = 47 := by
  sorry

end NUMINAMATH_CALUDE_fiftycentchange_l1105_110567


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l1105_110517

theorem quadratic_equation_k_value (x1 x2 k : ℝ) : 
  x1^2 - 6*x1 + k = 0 →
  x2^2 - 6*x2 + k = 0 →
  1/x1 + 1/x2 = 3 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l1105_110517


namespace NUMINAMATH_CALUDE_factory_production_l1105_110577

/-- Calculates the total number of products produced by a factory in 5 days -/
def total_products_in_five_days (refrigerators_per_hour : ℕ) (coolers_difference : ℕ) (hours_per_day : ℕ) : ℕ :=
  let coolers_per_hour := refrigerators_per_hour + coolers_difference
  let products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_hours := 5 * hours_per_day
  products_per_hour * total_hours

/-- Theorem stating that the factory produces 11250 products in 5 days -/
theorem factory_production :
  total_products_in_five_days 90 70 9 = 11250 :=
by
  sorry

end NUMINAMATH_CALUDE_factory_production_l1105_110577


namespace NUMINAMATH_CALUDE_max_goats_after_trading_l1105_110582

/-- Represents the trading system with coconuts, crabs, and goats -/
structure TradingSystem where
  coconuts_per_crab : ℕ
  crabs_per_goat : ℕ
  initial_coconuts : ℕ

/-- Calculates the number of goats obtained from trading coconuts -/
def goats_from_coconuts (ts : TradingSystem) : ℕ :=
  (ts.initial_coconuts / ts.coconuts_per_crab) / ts.crabs_per_goat

/-- Theorem stating that Max will have 19 goats after trading -/
theorem max_goats_after_trading :
  let ts : TradingSystem := {
    coconuts_per_crab := 3,
    crabs_per_goat := 6,
    initial_coconuts := 342
  }
  goats_from_coconuts ts = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_goats_after_trading_l1105_110582


namespace NUMINAMATH_CALUDE_line_intersection_b_range_l1105_110578

theorem line_intersection_b_range (b : ℝ) (h1 : b ≠ 0) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 2 * x + b = 3) →
  -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_b_range_l1105_110578


namespace NUMINAMATH_CALUDE_g_16_48_l1105_110574

/-- A function on ordered pairs of positive integers satisfying specific properties -/
def g : ℕ+ → ℕ+ → ℕ+ :=
  sorry

/-- The first property: g(x,x) = 2x -/
axiom g_diag (x : ℕ+) : g x x = 2 * x

/-- The second property: g(x,y) = g(y,x) -/
axiom g_comm (x y : ℕ+) : g x y = g y x

/-- The third property: (x + y) g(x,y) = x g(x, x + y) -/
axiom g_prop (x y : ℕ+) : (x + y) * g x y = x * g x (x + y)

/-- The main theorem: g(16, 48) = 96 -/
theorem g_16_48 : g 16 48 = 96 :=
  sorry

end NUMINAMATH_CALUDE_g_16_48_l1105_110574


namespace NUMINAMATH_CALUDE_exists_strategy_to_find_genuine_coin_l1105_110531

/-- Represents a coin, which can be either genuine or counterfeit. -/
inductive Coin
| genuine
| counterfeit

/-- Represents the result of weighing two coins. -/
inductive WeighResult
| equal
| left_heavier
| right_heavier

/-- A function that simulates weighing two coins. -/
def weigh : Coin → Coin → WeighResult := sorry

/-- The total number of coins. -/
def total_coins : Nat := 100

/-- A function that represents the distribution of coins. -/
def coin_distribution : Fin total_coins → Coin := sorry

/-- The number of counterfeit coins. -/
def counterfeit_count : Nat := sorry

/-- Assumption that there are more than 0 but less than 99 counterfeit coins. -/
axiom counterfeit_range : 0 < counterfeit_count ∧ counterfeit_count < 99

/-- A strategy is a function that takes the current state and returns the next pair of coins to weigh. -/
def Strategy := List WeighResult → Fin total_coins × Fin total_coins

/-- The theorem stating that there exists a strategy to find a genuine coin. -/
theorem exists_strategy_to_find_genuine_coin :
  ∃ (s : Strategy), ∀ (coin_dist : Fin total_coins → Coin),
    ∃ (n : Nat), n ≤ 99 ∧
      (∃ (i : Fin total_coins), coin_dist i = Coin.genuine ∧
        (∀ (j : Fin total_coins), j ≠ i → coin_dist j = Coin.counterfeit)) :=
sorry

end NUMINAMATH_CALUDE_exists_strategy_to_find_genuine_coin_l1105_110531


namespace NUMINAMATH_CALUDE_fiftieth_term_of_geometric_sequence_l1105_110557

/-- Given a geometric sequence with first term 5 and second term -15,
    the 50th term is -5 * 3^49 -/
theorem fiftieth_term_of_geometric_sequence (a : ℝ) (r : ℝ) :
  a = 5 →
  a * r = -15 →
  a * r^49 = -5 * 3^49 :=
by sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_geometric_sequence_l1105_110557


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1105_110506

theorem floor_equation_solution (n : ℤ) : (Int.floor (n^2 / 4 : ℚ) - Int.floor (n / 2 : ℚ)^2 = 5) ↔ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1105_110506


namespace NUMINAMATH_CALUDE_multiples_of_five_l1105_110592

/-- The largest number n such that there are 999 positive integers 
    between 5 and n (inclusive) that are multiples of 5 is 4995. -/
theorem multiples_of_five (n : ℕ) : 
  (∃ (k : ℕ), k = 999 ∧ 
    (∀ m : ℕ, 5 ≤ m ∧ m ≤ n ∧ m % 5 = 0 ↔ m ∈ Finset.range k)) →
  n = 4995 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_five_l1105_110592


namespace NUMINAMATH_CALUDE_total_handshakes_at_convention_l1105_110537

def num_gremlins : ℕ := 30
def num_imps : ℕ := 25
def num_reconciled_imps : ℕ := 10
def num_unreconciled_imps : ℕ := 15

def handshakes_among_gremlins : ℕ := num_gremlins * (num_gremlins - 1) / 2
def handshakes_among_reconciled_imps : ℕ := num_reconciled_imps * (num_reconciled_imps - 1) / 2
def handshakes_between_gremlins_and_imps : ℕ := num_gremlins * num_imps

theorem total_handshakes_at_convention : 
  handshakes_among_gremlins + handshakes_among_reconciled_imps + handshakes_between_gremlins_and_imps = 1230 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_at_convention_l1105_110537


namespace NUMINAMATH_CALUDE_problem_1_l1105_110516

theorem problem_1 : Real.sqrt (3/2) * Real.sqrt (21/4) / Real.sqrt (7/2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1105_110516


namespace NUMINAMATH_CALUDE_player_a_wins_two_player_player_b_wins_three_player_l1105_110512

/-- Represents a player in the Lazy Checkers game -/
inductive Player : Type
| A : Player
| B : Player
| C : Player

/-- Represents a position on the 5x5 board -/
structure Position :=
(row : Fin 5)
(col : Fin 5)

/-- Represents the state of the Lazy Checkers game -/
structure GameState :=
(board : Position → Option Player)
(current_player : Player)

/-- Represents a winning strategy for a player -/
def WinningStrategy (p : Player) : Type :=
GameState → Position

/-- The rules of Lazy Checkers ensure a valid game state -/
def ValidGameState (state : GameState) : Prop :=
sorry

/-- Theorem: In a two-player Lazy Checkers game, Player A has a winning strategy -/
theorem player_a_wins_two_player :
  ∃ (strategy : WinningStrategy Player.A),
    ∀ (initial_state : GameState),
      ValidGameState initial_state →
      initial_state.current_player = Player.A →
      -- The strategy leads to a win for Player A
      sorry :=
sorry

/-- Theorem: In a three-player Lazy Checkers game, Player B has a winning strategy when cooperating with Player C -/
theorem player_b_wins_three_player :
  ∃ (strategy_b : WinningStrategy Player.B) (strategy_c : WinningStrategy Player.C),
    ∀ (initial_state : GameState),
      ValidGameState initial_state →
      initial_state.current_player = Player.A →
      -- The strategies lead to a win for Player B
      sorry :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_two_player_player_b_wins_three_player_l1105_110512


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1105_110513

theorem boys_to_girls_ratio (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 48) (h2 : boys = 42) : 
  (boys : ℚ) / (total_students - boys : ℚ) = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1105_110513


namespace NUMINAMATH_CALUDE_bananas_left_l1105_110554

/-- The number of bananas initially in the jar -/
def initial_bananas : ℕ := 46

/-- The number of bananas removed from the jar -/
def removed_bananas : ℕ := 5

/-- Theorem: The number of bananas left in the jar is 41 -/
theorem bananas_left : initial_bananas - removed_bananas = 41 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l1105_110554


namespace NUMINAMATH_CALUDE_triangle_left_side_value_l1105_110502

/-- Given a triangle with sides L, R, and B satisfying certain conditions, prove that L = 12 -/
theorem triangle_left_side_value (L R B : ℝ) 
  (h1 : L + R + B = 50)
  (h2 : R = L + 2)
  (h3 : B = 24) : 
  L = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_left_side_value_l1105_110502


namespace NUMINAMATH_CALUDE_initial_test_count_l1105_110544

theorem initial_test_count (initial_avg : ℝ) (improved_avg : ℝ) (lowest_score : ℝ) :
  initial_avg = 35 →
  improved_avg = 40 →
  lowest_score = 20 →
  ∃ n : ℕ,
    n > 1 ∧
    (n : ℝ) * initial_avg = ((n : ℝ) - 1) * improved_avg + lowest_score ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_initial_test_count_l1105_110544


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l1105_110598

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets : ℕ) : 
  total_babies = 1000 →
  triplets = 4 * quadruplets →
  twins = 3 * triplets →
  2 * twins + 3 * triplets + 4 * quadruplets = total_babies →
  4 * quadruplets = 100 := by
  sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l1105_110598


namespace NUMINAMATH_CALUDE_gcf_90_108_l1105_110597

theorem gcf_90_108 : Nat.gcd 90 108 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_90_108_l1105_110597


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1105_110546

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : 
  k = 2021^2 + 2^2021 + 3 → (k^2 + 2^k) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1105_110546


namespace NUMINAMATH_CALUDE_extra_legs_count_l1105_110523

/-- Represents the number of legs for a cow -/
def cow_legs : ℕ := 4

/-- Represents the number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 9

theorem extra_legs_count (num_chickens : ℕ) : 
  cow_legs * num_cows + chicken_legs * num_chickens = 
  2 * (num_cows + num_chickens) + 18 := by
  sorry

end NUMINAMATH_CALUDE_extra_legs_count_l1105_110523


namespace NUMINAMATH_CALUDE_quiz_answer_key_l1105_110595

theorem quiz_answer_key (n : ℕ) : 
  (14 * n^2 = 224) → n = 4 :=
by
  sorry

#check quiz_answer_key

end NUMINAMATH_CALUDE_quiz_answer_key_l1105_110595


namespace NUMINAMATH_CALUDE_system_solution_l1105_110541

theorem system_solution (x y z : ℝ) : 
  x^4 + y^2 + 4 = 5*y*z ∧
  y^4 + z^2 + 4 = 5*z*x ∧
  z^4 + x^2 + 4 = 5*x*y →
  (x = y ∧ y = z ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1105_110541


namespace NUMINAMATH_CALUDE_min_value_theorem_l1105_110587

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 3) (hf_reaches_min : ∃ x, f x a b = 3) :
  (a^2 / b + b^2 / a) ≥ 3 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ (a^2 / b + b^2 / a) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1105_110587


namespace NUMINAMATH_CALUDE_polynomial_roots_coefficient_sum_l1105_110591

theorem polynomial_roots_coefficient_sum (p q r : ℝ) : 
  (∃ a b c : ℝ, 0 < a ∧ a < 2 ∧ 0 < b ∧ b < 2 ∧ 0 < c ∧ c < 2 ∧
    ∀ x : ℝ, x^3 + p*x^2 + q*x + r = (x - a) * (x - b) * (x - c)) →
  -2 < p + q + r ∧ p + q + r < 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_coefficient_sum_l1105_110591


namespace NUMINAMATH_CALUDE_wolf_still_hungry_l1105_110566

/-- Represents the food quantity provided by a hare -/
def hare_food : ℝ := sorry

/-- Represents the food quantity provided by a pig -/
def pig_food : ℝ := sorry

/-- Represents the food quantity needed to satisfy the wolf's hunger -/
def wolf_satiety : ℝ := sorry

/-- The wolf is still hungry after eating 3 pigs and 7 hares -/
axiom hunger_condition : 3 * pig_food + 7 * hare_food < wolf_satiety

/-- The wolf has overeaten after consuming 7 pigs and 1 hare -/
axiom overeating_condition : 7 * pig_food + hare_food > wolf_satiety

/-- Theorem: The wolf will still be hungry after eating 11 hares -/
theorem wolf_still_hungry : 11 * hare_food < wolf_satiety := by
  sorry

end NUMINAMATH_CALUDE_wolf_still_hungry_l1105_110566


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1105_110562

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -1 then Int.ceil (1 / (x + 1))
  else if x < -1 then Int.floor (1 / (x + 1))
  else 0  -- This value doesn't matter as x = -1 is not in the domain

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -1 → g x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1105_110562


namespace NUMINAMATH_CALUDE_sine_equality_l1105_110593

theorem sine_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 ∧ n = 55 → Real.sin (n * π / 180) = Real.sin (845 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sine_equality_l1105_110593


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1105_110583

theorem quadratic_inequality (x : ℝ) : x^2 - 34*x + 225 ≤ 9 ↔ 17 - Real.sqrt 73 ≤ x ∧ x ≤ 17 + Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1105_110583


namespace NUMINAMATH_CALUDE_five_toppings_from_eight_equals_fiftysix_l1105_110558

/-- The number of ways to choose 5 items from a set of 8 items -/
def choose_five_from_eight : ℕ := Nat.choose 8 5

/-- The theorem stating that choosing 5 items from 8 results in 56 combinations -/
theorem five_toppings_from_eight_equals_fiftysix : 
  choose_five_from_eight = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_toppings_from_eight_equals_fiftysix_l1105_110558


namespace NUMINAMATH_CALUDE_root_bounds_l1105_110538

theorem root_bounds (x : ℝ) : 
  x^2014 - 100*x + 1 = 0 → 1/100 ≤ x ∧ x ≤ 100^(1/2013) := by
  sorry

end NUMINAMATH_CALUDE_root_bounds_l1105_110538


namespace NUMINAMATH_CALUDE_daves_phone_files_l1105_110508

theorem daves_phone_files :
  let initial_apps : ℕ := 15
  let initial_files : ℕ := 24
  let final_apps : ℕ := 21
  let app_file_difference : ℕ := 17
  let files_left : ℕ := final_apps - app_file_difference
  files_left = 4 := by sorry

end NUMINAMATH_CALUDE_daves_phone_files_l1105_110508


namespace NUMINAMATH_CALUDE_college_student_count_l1105_110539

/-- Represents the number of students in each category -/
structure StudentCount where
  boys : ℕ
  girls : ℕ
  nonBinary : ℕ

/-- Calculates the total number of students -/
def totalStudents (s : StudentCount) : ℕ :=
  s.boys + s.girls + s.nonBinary

/-- Theorem: Given the ratio and number of girls, prove the total number of students -/
theorem college_student_count :
  ∀ (s : StudentCount),
    s.boys * 5 = s.girls * 8 →
    s.nonBinary * 5 = s.girls * 3 →
    s.girls = 400 →
    totalStudents s = 1280 := by
  sorry


end NUMINAMATH_CALUDE_college_student_count_l1105_110539


namespace NUMINAMATH_CALUDE_cricket_players_count_l1105_110525

/-- The number of cricket players in a game, given the numbers of other players and the total. -/
theorem cricket_players_count 
  (hockey_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (h1 : hockey_players = 12)
  (h2 : football_players = 16)
  (h3 : softball_players = 13)
  (h4 : total_players = 51)
  (h5 : total_players = hockey_players + football_players + softball_players + cricket_players) :
  cricket_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l1105_110525


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1105_110565

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1105_110565


namespace NUMINAMATH_CALUDE_february_first_is_friday_l1105_110505

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that February 13th is a Wednesday, prove that February 1st is a Friday -/
theorem february_first_is_friday 
  (feb13 : FebruaryDate)
  (h13 : feb13.day = 13)
  (hWed : feb13.dayOfWeek = DayOfWeek.Wednesday) :
  ∃ (feb1 : FebruaryDate), feb1.day = 1 ∧ feb1.dayOfWeek = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_february_first_is_friday_l1105_110505


namespace NUMINAMATH_CALUDE_right_triangle_of_orthocenters_l1105_110504

-- Define the circle and points
def Circle : Type := ℂ → Prop
def on_circle (c : Circle) (p : ℂ) : Prop := c p

-- Define the orthocenter function
def orthocenter (a b c : ℂ) : ℂ := sorry

-- Main theorem
theorem right_triangle_of_orthocenters 
  (O A B C D E : ℂ) 
  (c : Circle)
  (on_circle_A : on_circle c A)
  (on_circle_B : on_circle c B)
  (on_circle_C : on_circle c C)
  (on_circle_D : on_circle c D)
  (on_circle_E : on_circle c E)
  (consecutive : sorry) -- Represent that A, B, C, D, E are consecutive
  (equal_chords : AC = BD ∧ BD = CE ∧ CE = DO)
  (H₁ : ℂ := orthocenter A C D)
  (H₂ : ℂ := orthocenter B C D)
  (H₃ : ℂ := orthocenter B C E) :
  ∃ (θ : ℝ), Complex.arg ((H₁ - H₂) / (H₁ - H₃)) = θ ∧ θ = π/2 := by sorry

#check right_triangle_of_orthocenters

end NUMINAMATH_CALUDE_right_triangle_of_orthocenters_l1105_110504


namespace NUMINAMATH_CALUDE_ring_price_calculation_l1105_110540

def total_revenue : ℕ := 80
def necklace_price : ℕ := 12
def necklaces_sold : ℕ := 4
def rings_sold : ℕ := 8

theorem ring_price_calculation : 
  ∃ (ring_price : ℕ), 
    necklaces_sold * necklace_price + rings_sold * ring_price = total_revenue ∧ 
    ring_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_ring_price_calculation_l1105_110540


namespace NUMINAMATH_CALUDE_two_polygons_exist_l1105_110551

/-- Represents a polygon with a given number of sides. -/
structure Polygon where
  sides : ℕ

/-- Calculates the sum of interior angles of a polygon. -/
def sumInteriorAngles (p : Polygon) : ℕ :=
  (p.sides - 2) * 180

/-- Calculates the number of diagonals in a polygon. -/
def numDiagonals (p : Polygon) : ℕ :=
  p.sides * (p.sides - 3) / 2

/-- Theorem stating the existence of two polygons satisfying the given conditions. -/
theorem two_polygons_exist : ∃ (p1 p2 : Polygon),
  (sumInteriorAngles p1 + sumInteriorAngles p2 = 1260) ∧
  (numDiagonals p1 + numDiagonals p2 = 14) ∧
  ((p1.sides = 6 ∧ p2.sides = 5) ∨ (p1.sides = 5 ∧ p2.sides = 6)) := by
  sorry

end NUMINAMATH_CALUDE_two_polygons_exist_l1105_110551


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l1105_110526

-- Define the polar coordinate equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the rectangular coordinate equation
def rectangular_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Theorem stating the equivalence of the two equations
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (polar_equation ρ θ ↔ rectangular_equation x y) :=
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l1105_110526


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1105_110528

theorem expansion_terms_count (G1 G2 : Finset (Fin 4)) 
  (hG1 : G1.card = 4) (hG2 : G2.card = 4) :
  (G1.product G2).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1105_110528


namespace NUMINAMATH_CALUDE_cally_shorts_count_l1105_110553

/-- Represents the number of clothing items a person has. -/
structure ClothingItems where
  whiteShirts : Nat
  coloredShirts : Nat
  shorts : Nat
  pants : Nat

/-- Calculate the total number of clothing items. -/
def totalItems (items : ClothingItems) : Nat :=
  items.whiteShirts + items.coloredShirts + items.shorts + items.pants

theorem cally_shorts_count (totalWashed : Nat) (cally : ClothingItems) (danny : ClothingItems)
    (h1 : totalWashed = 58)
    (h2 : cally.whiteShirts = 10)
    (h3 : cally.coloredShirts = 5)
    (h4 : cally.pants = 6)
    (h5 : danny.whiteShirts = 6)
    (h6 : danny.coloredShirts = 8)
    (h7 : danny.shorts = 10)
    (h8 : danny.pants = 6)
    (h9 : totalWashed = totalItems cally + totalItems danny) :
  cally.shorts = 7 := by
  sorry


end NUMINAMATH_CALUDE_cally_shorts_count_l1105_110553


namespace NUMINAMATH_CALUDE_boys_from_clay_middle_school_l1105_110503

/-- Represents the three schools in the problem -/
inductive School
| Jonas
| Clay
| Pine

/-- Represents the gender of students -/
inductive Gender
| Boy
| Girl

/-- The total number of students at the camp -/
def total_students : ℕ := 150

/-- The number of boys at the camp -/
def total_boys : ℕ := 80

/-- The number of girls at the camp -/
def total_girls : ℕ := 70

/-- The number of students from each school -/
def students_per_school (s : School) : ℕ :=
  match s with
  | School.Jonas => 50
  | School.Clay => 60
  | School.Pine => 40

/-- The number of girls from Jonas Middle School -/
def girls_from_jonas : ℕ := 30

/-- The number of boys from Pine Middle School -/
def boys_from_pine : ℕ := 15

/-- The main theorem to prove -/
theorem boys_from_clay_middle_school :
  (students_per_school School.Clay) -
  (students_per_school School.Clay - 
   (total_boys - boys_from_pine - (students_per_school School.Jonas - girls_from_jonas))) = 45 := by
  sorry

end NUMINAMATH_CALUDE_boys_from_clay_middle_school_l1105_110503


namespace NUMINAMATH_CALUDE_pyramid_frustum_volume_l1105_110575

/-- Given a square pyramid with base edge s and altitude h, 
    if a smaller similar pyramid with altitude h/3 is removed from the apex, 
    the volume of the remaining frustum is 26/27 of the original pyramid's volume. -/
theorem pyramid_frustum_volume 
  (s h : ℝ) 
  (h_pos : 0 < h) 
  (s_pos : 0 < s) : 
  let v_original := (1 / 3) * s^2 * h
  let v_smaller := (1 / 3) * (s / 3)^2 * (h / 3)
  let v_frustum := v_original - v_smaller
  v_frustum = (26 / 27) * v_original := by
  sorry

end NUMINAMATH_CALUDE_pyramid_frustum_volume_l1105_110575


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l1105_110510

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  (∃ (M : ℕ), (N + M) % 7 = 0 ∧ (N + M) % 12 = 0) ∧
  (∀ (K : ℕ), K < 82 → ¬((N + K) % 7 = 0 ∧ (N + K) % 12 = 0)) ∧
  ((N + 82) % 7 = 0 ∧ (N + 82) % 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l1105_110510


namespace NUMINAMATH_CALUDE_unique_triplet_l1105_110560

theorem unique_triplet : ∃! (x y z : ℕ), 
  0 < x ∧ x < y ∧ y < z ∧ 
  Nat.gcd x (Nat.gcd y z) = 1 ∧
  (x + y) % z = 0 ∧
  (y + z) % x = 0 ∧
  (z + x) % y = 0 ∧
  x = 1 ∧ y = 2 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_triplet_l1105_110560


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1105_110588

/-- Represents a repeating decimal -/
def RepeatingDecimal (numerator denominator : ℕ) : ℚ := numerator / denominator

theorem repeating_decimal_sum_diff (a b c : ℚ) :
  a = RepeatingDecimal 6 9 →
  b = RepeatingDecimal 2 9 →
  c = RepeatingDecimal 4 9 →
  a + b - c = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1105_110588


namespace NUMINAMATH_CALUDE_even_function_comparison_l1105_110533

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_comparison (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m (-Real.pi) < f m 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_comparison_l1105_110533


namespace NUMINAMATH_CALUDE_bells_toll_together_l1105_110529

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 5) (hb : b = 8) (hc : c = 11) (hd : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 := by
sorry

end NUMINAMATH_CALUDE_bells_toll_together_l1105_110529


namespace NUMINAMATH_CALUDE_g_monotone_decreasing_l1105_110569

/-- The function g(x) defined in terms of the parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the conditions for g(x) to be monotonically decreasing -/
theorem g_monotone_decreasing (a : ℝ) : 
  (∀ x : ℝ, x < a / 3 → g' a x ≤ 0) ↔ a ∈ Set.Iic (-1) ∪ {0} :=
sorry

end NUMINAMATH_CALUDE_g_monotone_decreasing_l1105_110569


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1105_110580

/-- Given three varieties of tea with prices and mixing ratios, calculate the price of the mixture --/
theorem tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℚ) :
  price1 = 126 →
  price2 = 135 →
  price3 = 175.5 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3) = 153 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l1105_110580


namespace NUMINAMATH_CALUDE_triangle_inequality_l1105_110581

theorem triangle_inequality (R r a b c p : ℝ) 
  (h_R : R > 0) 
  (h_r : r > 0) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) 
  (h_p : p = (a + b + c) / 2) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circumradius : R = (a * b * c) / (4 * p * r)) 
  (h_inradius : r = p * (p - a) * (p - b) * (p - c) / (a * b * c)) :
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ 
  a * b + b * c + c * a ≤ 4 * (R + r)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1105_110581


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1105_110555

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 3, 5}

-- Define set B
def B : Set Nat := {1, 3, 4, 6}

-- Theorem statement
theorem intersection_with_complement : A ∩ (U \ B) = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1105_110555


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l1105_110534

theorem rationalize_denominator_sqrt5 : 
  ∃ (A B C : ℤ), 
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧ 
    A * B * C = 275 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l1105_110534


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l1105_110500

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 10*x + 20

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-2, -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l1105_110500


namespace NUMINAMATH_CALUDE_larger_number_proof_l1105_110584

theorem larger_number_proof (x y : ℝ) (h1 : y > x) (h2 : 5 * y = 6 * x) (h3 : y - x = 12) : y = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1105_110584


namespace NUMINAMATH_CALUDE_jordan_fourth_period_shots_l1105_110572

/-- The number of shots blocked by Jordan in each period of a hockey game --/
structure ShotsBlocked where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Conditions for Jordan's shot-blocking performance --/
def jordan_performance (shots : ShotsBlocked) : Prop :=
  shots.first = 4 ∧
  shots.second = 2 * shots.first ∧
  shots.third = shots.second - 3 ∧
  shots.first + shots.second + shots.third + shots.fourth = 21

/-- Theorem stating that Jordan blocked 4 shots in the fourth period --/
theorem jordan_fourth_period_shots (shots : ShotsBlocked) 
  (h : jordan_performance shots) : shots.fourth = 4 := by
  sorry

#check jordan_fourth_period_shots

end NUMINAMATH_CALUDE_jordan_fourth_period_shots_l1105_110572


namespace NUMINAMATH_CALUDE_problem_statement_l1105_110536

theorem problem_statement : (10 * 7)^3 + (45 * 5)^2 = 393625 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1105_110536


namespace NUMINAMATH_CALUDE_cubic_equation_third_root_l1105_110535

theorem cubic_equation_third_root 
  (a b : ℚ) 
  (h1 : a * (-1)^3 + (a + 3*b) * (-1)^2 + (2*b - 4*a) * (-1) + (10 - a) = 0)
  (h2 : a * 4^3 + (a + 3*b) * 4^2 + (2*b - 4*a) * 4 + (10 - a) = 0)
  : ∃ (x : ℚ), x = -62/19 ∧ 
    a * x^3 + (a + 3*b) * x^2 + (2*b - 4*a) * x + (10 - a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_third_root_l1105_110535


namespace NUMINAMATH_CALUDE_percentage_difference_l1105_110547

theorem percentage_difference (y e w z : ℝ) (P : ℝ) : 
  w = e * (1 - P / 100) →
  e = y * 0.6 →
  z = y * 0.54 →
  z = w * (1 + 0.5000000000000002) →
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1105_110547


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l1105_110543

/-- An unfair eight-sided die with given probabilities -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℝ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℝ
  /-- The probability of rolling an 8 is 3/8 -/
  h1 : prob_eight = 3/8
  /-- The probability of rolling any number from 1 to 7 is 5/56 -/
  h2 : prob_others = 5/56
  /-- The sum of all probabilities is 1 -/
  h3 : prob_eight + 7 * prob_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℝ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.prob_eight * 8

/-- Theorem stating that the expected value of rolling the unfair die is 5.5 -/
theorem unfair_die_expected_value (d : UnfairDie) :
  expected_value d = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l1105_110543


namespace NUMINAMATH_CALUDE_david_presents_l1105_110586

theorem david_presents (christmas : ℕ) (easter : ℕ) (birthday : ℕ) 
  (h1 : christmas = 60)
  (h2 : birthday = 3 * easter)
  (h3 : easter = christmas / 2 - 10) : 
  christmas + easter + birthday = 140 := by
  sorry

end NUMINAMATH_CALUDE_david_presents_l1105_110586


namespace NUMINAMATH_CALUDE_triangle_inequality_l1105_110527

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = Real.pi)
  (h_area : S = (1/2) * a * b * Real.sin C)
  (h_side_a : a = b * Real.sin C / Real.sin A)
  (h_side_b : b = c * Real.sin A / Real.sin B)
  (h_side_c : c = a * Real.sin B / Real.sin C) :
  a^2 * Real.tan (A/2) + b^2 * Real.tan (B/2) + c^2 * Real.tan (C/2) ≥ 4 * S :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1105_110527


namespace NUMINAMATH_CALUDE_buses_meet_at_two_pm_l1105_110552

/-- Represents a bus with departure and arrival times -/
structure Bus where
  departure : ℕ
  arrival : ℕ

/-- The time when two buses meet given their schedules -/
def meeting_time (bus1 bus2 : Bus) : ℕ :=
  sorry

theorem buses_meet_at_two_pm (bus1 bus2 : Bus)
  (h1 : bus1.departure = 11 ∧ bus1.arrival = 16)
  (h2 : bus2.departure = 12 ∧ bus2.arrival = 17) :
  meeting_time bus1 bus2 = 14 :=
sorry

end NUMINAMATH_CALUDE_buses_meet_at_two_pm_l1105_110552


namespace NUMINAMATH_CALUDE_homer_candy_crush_score_l1105_110522

def candy_crush_score (first_try : ℕ) (second_try_difference : ℕ) : ℕ :=
  let second_try := first_try - second_try_difference
  let third_try := 2 * second_try
  first_try + second_try + third_try

theorem homer_candy_crush_score :
  candy_crush_score 400 70 = 1390 := by
  sorry

end NUMINAMATH_CALUDE_homer_candy_crush_score_l1105_110522


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1105_110561

theorem complex_expression_simplification (x y : ℝ) : 
  (2 * x + 3 * Complex.I * y) * (2 * x - 3 * Complex.I * y) + 2 * x = 4 * x^2 + 2 * x - 9 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1105_110561


namespace NUMINAMATH_CALUDE_bells_toll_together_l1105_110585

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 5) (hb : b = 8) (hc : c = 11) (hd : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l1105_110585


namespace NUMINAMATH_CALUDE_engineer_teams_count_l1105_110518

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of ways to form a team of engineers with given constraints -/
def engineerTeams : ℕ :=
  let totalEngineers := 15
  let phdEngineers := 5
  let msEngineers := 6
  let bsEngineers := 4
  let teamSize := 5
  let minPhd := 2
  let minMs := 2
  let minBs := 1
  (choose phdEngineers minPhd) * (choose msEngineers minMs) * (choose bsEngineers minBs)

theorem engineer_teams_count :
  engineerTeams = 600 := by sorry

end NUMINAMATH_CALUDE_engineer_teams_count_l1105_110518


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l1105_110564

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are internally tangent --/
def are_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = abs (c2.radius - c1.radius)

/-- The first circle: x^2 + y^2 - 2x = 0 --/
def circle1 : Circle :=
  { center := (1, 0), radius := 1 }

/-- The second circle: x^2 + y^2 - 2x - 6y - 6 = 0 --/
def circle2 : Circle :=
  { center := (1, 3), radius := 4 }

/-- Theorem stating that the two given circles are internally tangent --/
theorem circles_internally_tangent : are_internally_tangent circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l1105_110564


namespace NUMINAMATH_CALUDE_inequality_proof_l1105_110507

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1105_110507


namespace NUMINAMATH_CALUDE_path_length_2x1x1_block_l1105_110530

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the path traced by a dot on the block -/
def path_length (b : Block) : ℝ := sorry

/-- Theorem stating that the path length for a 2×1×1 block is 4π -/
theorem path_length_2x1x1_block :
  let b : Block := ⟨2, 1, 1⟩
  path_length b = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_path_length_2x1x1_block_l1105_110530


namespace NUMINAMATH_CALUDE_race_distance_is_150_yards_l1105_110545

/-- Represents the race scenario with three racers A, B, and C --/
structure RaceScenario where
  d : ℝ  -- race distance in yards
  a : ℝ  -- speed of racer A
  b : ℝ  -- speed of racer B
  c : ℝ  -- speed of racer C

/-- The race scenario satisfies the given conditions --/
def satisfiesConditions (race : RaceScenario) : Prop :=
  race.d / race.a = (race.d - 30) / race.b ∧  -- A beats B by 30 yards
  race.d / race.b = (race.d - 15) / race.c ∧  -- B beats C by 15 yards
  race.d / race.a = (race.d - 42) / race.c ∧  -- A beats C by 42 yards
  race.c = 0.9 * race.b                       -- C's speed is 10% slower than B's

/-- The theorem stating that the race distance is 150 yards --/
theorem race_distance_is_150_yards (race : RaceScenario) 
  (h : satisfiesConditions race) : race.d = 150 := by
  sorry

#check race_distance_is_150_yards

end NUMINAMATH_CALUDE_race_distance_is_150_yards_l1105_110545


namespace NUMINAMATH_CALUDE_max_points_top_three_l1105_110594

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Calculate the maximum points a team can achieve -/
def max_points_per_team (t : Tournament) : ℕ :=
  (t.num_teams - 1) * t.games_per_pair * t.points_for_win

/-- The main theorem to prove -/
theorem max_points_top_three (t : Tournament) 
  (h1 : t.num_teams = 9)
  (h2 : t.games_per_pair = 2)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  ∃ (max_points : ℕ), max_points = 42 ∧ 
  (∀ (top_three_points : ℕ), top_three_points ≤ max_points) ∧
  (∃ (strategy : Tournament → ℕ), strategy t = max_points) :=
sorry

end NUMINAMATH_CALUDE_max_points_top_three_l1105_110594


namespace NUMINAMATH_CALUDE_unique_solution_system_l1105_110596

/-- The system of equations has exactly one real solution -/
theorem unique_solution_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ x * y - z^2 = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1105_110596


namespace NUMINAMATH_CALUDE_circle_C_properties_l1105_110556

/-- Definition of the circle C -/
def circle_C (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 25}

/-- Theorem stating the properties of circle C and its tangent lines -/
theorem circle_C_properties :
  ∃ (a b : ℝ),
    (a + b + 1 = 0) ∧
    ((-2 - a)^2 + (0 - b)^2 = 25) ∧
    ((5 - a)^2 + (1 - b)^2 = 25) ∧
    (circle_C a b = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 25}) ∧
    (∀ (x y : ℝ), x = -3 → (x, y) ∈ circle_C a b → y = 0 ∨ y ≠ 0) ∧
    (∀ (x y : ℝ), y = (8/15) * (x + 3) → (x, y) ∈ circle_C a b → x = -3 ∨ x ≠ -3) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_C_properties_l1105_110556


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1105_110542

/-- A parabola with directrix y = 1/2 has the standard equation x^2 = -2y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y = 1/2 → (x^2 = -2*p*y ↔ y = -x^2/(2*p))) →
  p = 1 :=
by sorry

#check parabola_standard_equation

end NUMINAMATH_CALUDE_parabola_standard_equation_l1105_110542


namespace NUMINAMATH_CALUDE_expression_evaluation_l1105_110573

theorem expression_evaluation (a b : ℝ) (ha : a = 2023) (hb : b = 2020) : 
  ((3 / (a - b) + 3 * a / (a^3 - b^3) * (a^2 + a*b + b^2) / (a + b)) * 
   (2*a + b) / (a^2 + 2*a*b + b^2)) * 3 / (a + b) = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1105_110573


namespace NUMINAMATH_CALUDE_shooting_scenario_outcomes_l1105_110599

/-- Represents the number of shots fired -/
def total_shots : ℕ := 8

/-- Represents the number of successful hits -/
def total_hits : ℕ := 4

/-- Represents the number of consecutive hits required -/
def consecutive_hits : ℕ := 3

/-- Calculates the number of different outcomes for the shooting scenario -/
def shooting_outcomes : ℕ := total_shots + 1 - total_hits

/-- Theorem stating that the number of different outcomes is 20 -/
theorem shooting_scenario_outcomes : 
  shooting_outcomes = 20 := by sorry

end NUMINAMATH_CALUDE_shooting_scenario_outcomes_l1105_110599


namespace NUMINAMATH_CALUDE_max_satisfying_all_is_50_l1105_110511

/-- Represents the youth summer village population --/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamily : ℕ
  singingInShower : ℕ

/-- The conditions of the problem --/
def problemVillage : Village :=
  { total := 100
  , notWorking := 50
  , withFamily := 25
  , singingInShower := 75 }

/-- The maximum number of people satisfying all conditions --/
def maxSatisfyingAll (v : Village) : ℕ :=
  min (v.total - v.notWorking) (min (v.total - v.withFamily) v.singingInShower)

/-- Theorem stating the maximum number of people satisfying all conditions --/
theorem max_satisfying_all_is_50 :
  maxSatisfyingAll problemVillage = 50 := by sorry

end NUMINAMATH_CALUDE_max_satisfying_all_is_50_l1105_110511


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1105_110576

/-- The hyperbola C with real semi-axis length √3 and the same foci as the ellipse x²/8 + y²/4 = 1 -/
structure Hyperbola where
  /-- The real semi-axis length of the hyperbola -/
  a : ℝ
  /-- The imaginary semi-axis length of the hyperbola -/
  b : ℝ
  /-- The focal distance of the hyperbola -/
  c : ℝ
  /-- The real semi-axis length is √3 -/
  ha : a = Real.sqrt 3
  /-- The focal distance is the same as the ellipse x²/8 + y²/4 = 1 -/
  hc : c = 2
  /-- Relation between a, b, and c in a hyperbola -/
  hab : c^2 = a^2 + b^2

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The line intersecting the hyperbola -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

/-- The dot product of two points with the origin -/
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ * x₂ + y₁ * y₂

/-- The main theorem -/
theorem hyperbola_intersection_theorem (h : Hyperbola) :
  (∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 = 1) ∧
  (∀ k, (∃ x₁ y₁ x₂ y₂, 
    x₁ ≠ x₂ ∧
    hyperbola_equation h x₁ y₁ ∧
    hyperbola_equation h x₂ y₂ ∧
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    dot_product x₁ y₁ x₂ y₂ > 2) ↔
   (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1105_110576


namespace NUMINAMATH_CALUDE_similar_triangles_leg_l1105_110590

/-- Two similar right triangles with legs 12 and 9 in the first triangle,
    and y and 6 in the second triangle, have y = 8 -/
theorem similar_triangles_leg (y : ℝ) : 
  (12 : ℝ) / y = 9 / 6 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_l1105_110590


namespace NUMINAMATH_CALUDE_multiplication_as_difference_of_squares_l1105_110519

theorem multiplication_as_difference_of_squares :
  ∀ a b : ℚ,
  (19 + 2/3) * (20 + 1/3) = (a - b) * (a + b) →
  a = 20 ∧ b = 1/3 := by
sorry

end NUMINAMATH_CALUDE_multiplication_as_difference_of_squares_l1105_110519


namespace NUMINAMATH_CALUDE_correct_multiplication_result_l1105_110515

theorem correct_multiplication_result (result : ℕ) (wrong_digits : List ℕ) :
  result = 867559827931 ∧
  wrong_digits = [8, 6, 7, 5, 2, 7, 9] ∧
  (∃ n : ℕ, n * 98765 = result) →
  ∃ m : ℕ, m * 98765 = 888885 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_result_l1105_110515


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1105_110571

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 4 → x ≥ 4) ∧
  (∃ x : ℝ, x ≥ 4 ∧ ¬(x > 4)) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1105_110571


namespace NUMINAMATH_CALUDE_andrew_mango_purchase_l1105_110570

/-- Calculates the quantity of mangoes purchased given the total amount paid,
    grape quantity, grape price, and mango price. -/
def mango_quantity (total_paid : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  (total_paid - grape_quantity * grape_price) / mango_price

theorem andrew_mango_purchase :
  mango_quantity 908 7 68 48 = 9 := by
  sorry

end NUMINAMATH_CALUDE_andrew_mango_purchase_l1105_110570


namespace NUMINAMATH_CALUDE_min_value_f_and_g_inequality_l1105_110579

noncomputable section

variable (t : ℝ)

def f (x : ℝ) := Real.exp (2 * x) - 2 * t * x

def g (x : ℝ) := -x^2 + 2 * t * Real.exp x - 2 * t^2 + 1/2

theorem min_value_f_and_g_inequality :
  (t ≤ 1 → ∀ x ≥ 0, f t x ≥ 1) ∧
  (t > 1 → ∀ x ≥ 0, f t x ≥ t - t * Real.log t) ∧
  (t = 1 → ∀ x ≥ 0, g t x ≥ 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_min_value_f_and_g_inequality_l1105_110579


namespace NUMINAMATH_CALUDE_largest_house_number_l1105_110521

def phone_number : List Nat := [2, 7, 1, 3, 1, 4, 7]

def sum_digits (digits : List Nat) : Nat :=
  digits.sum

def is_distinct (digits : List Nat) : Prop :=
  digits.length = digits.eraseDups.length

def is_valid_house_number (house : List Nat) : Prop :=
  house.length = 4 ∧ 
  is_distinct house ∧
  sum_digits house = sum_digits phone_number

theorem largest_house_number : 
  ∀ house : List Nat, is_valid_house_number house → 
  house.foldl (fun acc d => acc * 10 + d) 0 ≤ 9871 :=
by sorry

end NUMINAMATH_CALUDE_largest_house_number_l1105_110521


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l1105_110532

theorem product_of_sum_and_difference (a b : ℝ) 
  (sum_eq : a + b = 3) 
  (diff_eq : a - b = 7) : 
  a * b = -10 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l1105_110532


namespace NUMINAMATH_CALUDE_min_dot_product_vectors_l1105_110550

/-- The dot product of vectors (1, x) and (x, x+1) has a minimum value of -1 -/
theorem min_dot_product_vectors : 
  ∃ (min : ℝ), min = -1 ∧ 
  ∀ (x : ℝ), (1 * x + x * (x + 1)) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_vectors_l1105_110550


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l1105_110559

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l1105_110559
