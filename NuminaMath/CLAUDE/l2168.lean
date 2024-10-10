import Mathlib

namespace simplify_irrational_denominator_l2168_216893

theorem simplify_irrational_denominator :
  (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end simplify_irrational_denominator_l2168_216893


namespace quadratic_two_roots_l2168_216812

theorem quadratic_two_roots (a b c : ℝ) (h1 : b > a + c) (h2 : a + c > 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 := by
  sorry

end quadratic_two_roots_l2168_216812


namespace smallest_number_proof_l2168_216869

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % 45 = 0 ∧ (n - 20) % 60 = 0

theorem smallest_number_proof :
  is_divisible_by_all 200 ∧ ∀ m : ℕ, m < 200 → ¬is_divisible_by_all m :=
sorry

end smallest_number_proof_l2168_216869


namespace largest_divisor_of_difference_of_squares_l2168_216868

theorem largest_divisor_of_difference_of_squares (m n : ℕ) :
  (∃ k l : ℕ, m = 2 * k ∧ n = 2 * l) →  -- m and n are even
  n < m →  -- n is less than m
  (∃ d : ℕ, d > 0 ∧ 
    (∀ a b : ℕ, (∃ i j : ℕ, a = 2 * i ∧ b = 2 * j) → b < a → 
      d ∣ (a^2 - b^2)) ∧
    (∀ e : ℕ, e > d → 
      ∃ x y : ℕ, (∃ p q : ℕ, x = 2 * p ∧ y = 2 * q) ∧ y < x ∧ ¬(e ∣ (x^2 - y^2)))) →
  (∃ d : ℕ, d = 16 ∧ d > 0 ∧ 
    (∀ a b : ℕ, (∃ i j : ℕ, a = 2 * i ∧ b = 2 * j) → b < a → 
      d ∣ (a^2 - b^2)) ∧
    (∀ e : ℕ, e > d → 
      ∃ x y : ℕ, (∃ p q : ℕ, x = 2 * p ∧ y = 2 * q) ∧ y < x ∧ ¬(e ∣ (x^2 - y^2)))) :=
by sorry


end largest_divisor_of_difference_of_squares_l2168_216868


namespace oranges_for_24_apples_value_l2168_216887

/-- The number of oranges that can be bought for the price of 24 apples -/
def oranges_for_24_apples (apple_price banana_price cucumber_price orange_price : ℚ) : ℚ :=
  24 * apple_price / orange_price

/-- Theorem stating the number of oranges that can be bought for the price of 24 apples -/
theorem oranges_for_24_apples_value
  (h1 : 12 * apple_price = 6 * banana_price)
  (h2 : 3 * banana_price = 5 * cucumber_price)
  (h3 : 2 * cucumber_price = orange_price)
  : oranges_for_24_apples apple_price banana_price cucumber_price orange_price = 10 := by
  sorry

end oranges_for_24_apples_value_l2168_216887


namespace tan_alpha_value_l2168_216809

theorem tan_alpha_value (α : Real) :
  2 * Real.cos (π / 2 - α) - Real.sin (3 * π / 2 + α) = -Real.sqrt 5 →
  Real.tan α = 2 := by
sorry

end tan_alpha_value_l2168_216809


namespace number_of_literate_employees_l2168_216825

def number_of_illiterate_employees : ℕ := 35
def wage_decrease_per_illiterate : ℕ := 25
def total_wage_decrease : ℕ := 875
def average_salary_decrease : ℕ := 15
def total_employees : ℕ := 58

theorem number_of_literate_employees :
  total_employees - number_of_illiterate_employees = 23 :=
by sorry

end number_of_literate_employees_l2168_216825


namespace perpendicular_planes_l2168_216821

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (l m : Line) 
  (α β : Plane) 
  (h_diff_lines : l ≠ m) 
  (h_diff_planes : α ≠ β) 
  (h_l_perp_m : perp_line l m) 
  (h_l_perp_α : perp_line_plane l α) 
  (h_m_perp_β : perp_line_plane m β) : 
  perp_plane α β :=
sorry

end perpendicular_planes_l2168_216821


namespace sin_cos_power_12_range_l2168_216890

theorem sin_cos_power_12_range (x : ℝ) : 
  (1 : ℝ) / 32 ≤ Real.sin x ^ 12 + Real.cos x ^ 12 ∧ 
  Real.sin x ^ 12 + Real.cos x ^ 12 ≤ 1 := by
  sorry

end sin_cos_power_12_range_l2168_216890


namespace T_simplification_l2168_216834

theorem T_simplification (x : ℝ) : 
  (x - 2)^4 + 8*(x - 2)^3 + 24*(x - 2)^2 + 32*(x - 2) + 16 = x^4 := by
  sorry

end T_simplification_l2168_216834


namespace custom_mult_identity_value_l2168_216874

/-- Custom multiplication operation -/
def custom_mult (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mult_identity_value (a b c : ℝ) :
  (custom_mult a b c 1 2 = 4) →
  (custom_mult a b c 2 3 = 6) →
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x : ℝ, custom_mult a b c x m = x) →
  ∃ m : ℝ, m = 13 ∧ m ≠ 0 ∧ ∀ x : ℝ, custom_mult a b c x m = x :=
by sorry

end custom_mult_identity_value_l2168_216874


namespace parallel_transitivity_false_l2168_216833

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- State the theorem
theorem parallel_transitivity_false 
  (l m : Line) (α : Plane) : 
  (parallel_line_plane l α ∧ parallel_line_plane m α) → 
  parallel_line_line l m :=
sorry

end parallel_transitivity_false_l2168_216833


namespace workshop_workers_l2168_216889

theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (non_technician_salary : ℝ) 
  (num_technicians : ℕ) (h1 : average_salary = 8000) 
  (h2 : technician_salary = 10000) (h3 : non_technician_salary = 6000) 
  (h4 : num_technicians = 7) : 
  ∃ (total_workers : ℕ), total_workers = 14 ∧ 
  (num_technicians : ℝ) * technician_salary + 
  ((total_workers - num_technicians) : ℝ) * non_technician_salary = 
  (total_workers : ℝ) * average_salary :=
sorry

end workshop_workers_l2168_216889


namespace hyperbola_equation_l2168_216891

/-- A hyperbola with foci at (-3,0) and (3,0), and a vertex at (2,0) has the equation x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  let foci_1 : ℝ × ℝ := (-3, 0)
  let foci_2 : ℝ × ℝ := (3, 0)
  let vertex : ℝ × ℝ := (2, 0)
  (x^2 / 4 - y^2 / 5 = 1) ↔ 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      vertex.1 = a ∧
      (foci_2.1 - foci_1.1)^2 / 4 = a^2 + b^2) :=
by sorry

end hyperbola_equation_l2168_216891


namespace round_robin_tournament_games_l2168_216884

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of combinations of n things taken k at a time -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem round_robin_tournament_games :
  num_games 6 = binom 6 2 := by sorry

end round_robin_tournament_games_l2168_216884


namespace equation_solution_l2168_216848

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 8) - 10) + 3 / (Real.sqrt (x - 8) - 5) + 
   4 / (Real.sqrt (x - 8) + 5) + 15 / (Real.sqrt (x - 8) + 10) = 0) ↔ 
  (x = 33 ∨ x = 108) :=
by sorry

end equation_solution_l2168_216848


namespace complex_power_difference_l2168_216819

theorem complex_power_difference (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^2187 - 1/(x^2187) = 0 := by
  sorry

end complex_power_difference_l2168_216819


namespace sqrt_two_squared_l2168_216843

theorem sqrt_two_squared : (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end sqrt_two_squared_l2168_216843


namespace sum_of_fifth_and_eighth_term_l2168_216877

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fifth_and_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 3 * a 10 = 5 ∧ a 3 + a 10 = 3) :
  a 5 + a 8 = 3 := by
sorry

end sum_of_fifth_and_eighth_term_l2168_216877


namespace positive_X_value_l2168_216829

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) (h : hash X 7 = 338) : X = 17 := by
  sorry

end positive_X_value_l2168_216829


namespace equation_solutions_l2168_216859

def solution_set : Set ℝ := {12, 1, -1, -12}

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 9*x - 12) + 1 / (x^2 + 3*x - 18) + 1 / (x^2 - 15*x - 12) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = solution_set :=
by sorry

end equation_solutions_l2168_216859


namespace fifth_root_unity_sum_l2168_216898

theorem fifth_root_unity_sum (x : ℂ) : x^5 = 1 → 1 + x^4 + x^8 + x^12 + x^16 = 0 := by
  sorry

end fifth_root_unity_sum_l2168_216898


namespace largest_non_sum_of_three_distinct_composites_l2168_216892

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬Nat.Prime n

/-- A function that checks if a natural number can be expressed as the sum of three distinct composite numbers -/
def IsSumOfThreeDistinctComposites (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), IsComposite a ∧ IsComposite b ∧ IsComposite c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = n

/-- The theorem stating that 17 is the largest integer that cannot be expressed as the sum of three distinct composite numbers -/
theorem largest_non_sum_of_three_distinct_composites :
  (∀ n > 17, IsSumOfThreeDistinctComposites n) ∧
  ¬IsSumOfThreeDistinctComposites 17 ∧
  (∀ n < 17, ¬IsSumOfThreeDistinctComposites n → ¬IsSumOfThreeDistinctComposites 17) :=
sorry

end largest_non_sum_of_three_distinct_composites_l2168_216892


namespace arithmetic_sequence_sum_l2168_216804

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 3 + a 6 + a 9 = 27 := by
  sorry

end arithmetic_sequence_sum_l2168_216804


namespace percentage_of_girls_l2168_216823

theorem percentage_of_girls (total_students : ℕ) (num_boys : ℕ) : 
  total_students = 400 → num_boys = 80 → 
  (((total_students - num_boys : ℚ) / total_students) * 100 : ℚ) = 80 := by
  sorry

end percentage_of_girls_l2168_216823


namespace hockey_season_length_l2168_216886

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The total number of hockey games in the season -/
def total_games : ℕ := 182

/-- The number of months in the hockey season -/
def season_length : ℕ := total_games / games_per_month

theorem hockey_season_length :
  season_length = 14 :=
sorry

end hockey_season_length_l2168_216886


namespace frosting_cans_needed_l2168_216836

def cakes_day1 : ℕ := 7
def cakes_day2 : ℕ := 12
def cakes_day3 : ℕ := 8
def cakes_day4 : ℕ := 10
def cakes_day5 : ℕ := 15
def cakes_eaten : ℕ := 18
def frosting_per_cake : ℕ := 3

def total_cakes : ℕ := cakes_day1 + cakes_day2 + cakes_day3 + cakes_day4 + cakes_day5
def remaining_cakes : ℕ := total_cakes - cakes_eaten

theorem frosting_cans_needed : remaining_cakes * frosting_per_cake = 102 := by
  sorry

end frosting_cans_needed_l2168_216836


namespace equation_A_is_linear_l2168_216861

/-- An equation is linear in two variables if it can be written in the form ax + by + c = 0,
    where a, b, and c are constants, and a and b are not both zero. -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y + c = 0

/-- The equation (2y-1)/5 = 2 - (3x-2)/4 -/
def equation_A (x y : ℝ) : Prop :=
  (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

theorem equation_A_is_linear :
  is_linear_equation_in_two_variables equation_A :=
sorry

end equation_A_is_linear_l2168_216861


namespace pedestrian_speed_theorem_l2168_216885

theorem pedestrian_speed_theorem :
  ∃ (v : ℝ), v > 0 ∧
  (∀ (t : ℝ), 0 ≤ t ∧ t ≤ 2.5 →
    (if t % 1 < 0.5 then (5 + v) else (5 - v)) * 0.5 +
    (if (t + 0.5) % 1 < 0.5 then (5 + v) else (5 - v)) * 0.5 = 5) ∧
  ((4 * (5 + v) * 0.5 + 3 * (5 - v) * 0.5) / 3.5 > 5) :=
by sorry

end pedestrian_speed_theorem_l2168_216885


namespace quadratic_root_implies_t_value_l2168_216863

theorem quadratic_root_implies_t_value (a t : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (a + 3 * Complex.I : ℂ) ^ 2 - 4 * (a + 3 * Complex.I : ℂ) + t = 0 →
  t = 13 := by
  sorry

end quadratic_root_implies_t_value_l2168_216863


namespace inequality_solution_l2168_216867

theorem inequality_solution (x : ℝ) :
  x > -1 ∧ x ≠ 0 →
  (x^2 / ((x + 1 - Real.sqrt (x + 1))^2) < (x^2 + 3*x + 18) / (x + 1)^2) ↔
  (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 3) :=
by sorry

end inequality_solution_l2168_216867


namespace binomial_17_4_l2168_216803

theorem binomial_17_4 : Nat.choose 17 4 = 2380 := by
  sorry

end binomial_17_4_l2168_216803


namespace tan_product_ninth_pi_l2168_216830

theorem tan_product_ninth_pi : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end tan_product_ninth_pi_l2168_216830


namespace win_in_four_moves_cannot_win_in_ten_moves_min_moves_for_2018_l2168_216826

/-- Represents the state of the game --/
structure GameState where
  coinsA : ℕ  -- Coins in box A
  coinsB : ℕ  -- Coins in box B

/-- Defines a single move in the game --/
inductive Move
  | MoveToB    : Move  -- Move a coin from A to B
  | RemoveFromA : Move  -- Remove coins from A equal to coins in B

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.MoveToB => 
      { coinsA := state.coinsA - 1, coinsB := state.coinsB + 1 }
  | Move.RemoveFromA => 
      { coinsA := state.coinsA - state.coinsB, coinsB := state.coinsB }

/-- Checks if the game is won (i.e., box A is empty) --/
def isGameWon (state : GameState) : Prop := state.coinsA = 0

/-- Theorem: For 6 initial coins, the game can be won in 4 moves --/
theorem win_in_four_moves (initialCoins : ℕ) (h : initialCoins = 6) : 
  ∃ (moves : List Move), moves.length = 4 ∧ 
    isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 }) := by
  sorry

/-- Theorem: For 31 initial coins, the game cannot be won in 10 moves --/
theorem cannot_win_in_ten_moves (initialCoins : ℕ) (h : initialCoins = 31) :
  ∀ (moves : List Move), moves.length = 10 → 
    ¬isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 }) := by
  sorry

/-- Theorem: For 2018 initial coins, the minimum number of moves to win is 89 --/
theorem min_moves_for_2018 (initialCoins : ℕ) (h : initialCoins = 2018) :
  (∃ (moves : List Move), moves.length = 89 ∧ 
    isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 })) ∧
  (∀ (moves : List Move), moves.length < 89 → 
    ¬isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 })) := by
  sorry

end win_in_four_moves_cannot_win_in_ten_moves_min_moves_for_2018_l2168_216826


namespace nine_digit_integers_count_l2168_216897

theorem nine_digit_integers_count : 
  (Finset.range 8).card * (10 ^ 8) = 800000000 := by
  sorry

end nine_digit_integers_count_l2168_216897


namespace lottery_probability_l2168_216842

/-- Represents the lottery setup -/
structure LotterySetup where
  total_people : Nat
  total_tickets : Nat
  winning_tickets : Nat

/-- Calculates the probability of the lottery ending after a specific draw -/
def probability_end_after_draw (setup : LotterySetup) (draw : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem lottery_probability (setup : LotterySetup) :
  setup.total_people = 5 →
  setup.total_tickets = 5 →
  setup.winning_tickets = 3 →
  probability_end_after_draw setup 4 = 3 / 10 := by
  sorry

end lottery_probability_l2168_216842


namespace sum_of_four_consecutive_even_numbers_l2168_216896

theorem sum_of_four_consecutive_even_numbers : 
  let n : ℕ := 32
  let sum := n + (n + 2) + (n + 4) + (n + 6)
  sum = 140 := by
  sorry

end sum_of_four_consecutive_even_numbers_l2168_216896


namespace membership_ratio_is_three_to_one_l2168_216862

/-- Represents the monthly costs and sign-up fees for two gym memberships --/
structure GymMemberships where
  cheap_monthly : ℚ
  cheap_signup : ℚ
  expensive_signup_months : ℚ
  total_first_year : ℚ

/-- Calculates the ratio of expensive gym's monthly cost to cheap gym's monthly cost --/
def membership_ratio (g : GymMemberships) : ℚ :=
  let cheap_yearly := g.cheap_signup + 12 * g.cheap_monthly
  let expensive_yearly := g.total_first_year - cheap_yearly
  let expensive_monthly := expensive_yearly / (g.expensive_signup_months + 12)
  expensive_monthly / g.cheap_monthly

/-- Theorem stating that the membership ratio is 3:1 for the given conditions --/
theorem membership_ratio_is_three_to_one (g : GymMemberships)
    (h1 : g.cheap_monthly = 10)
    (h2 : g.cheap_signup = 50)
    (h3 : g.expensive_signup_months = 4)
    (h4 : g.total_first_year = 650) :
    membership_ratio g = 3 := by
  sorry

end membership_ratio_is_three_to_one_l2168_216862


namespace sons_present_age_l2168_216814

theorem sons_present_age (son_age father_age : ℕ) : 
  father_age = son_age + 45 →
  father_age + 10 = 4 * (son_age + 10) →
  son_age + 15 = 2 * son_age →
  son_age = 15 := by
sorry

end sons_present_age_l2168_216814


namespace saree_price_calculation_l2168_216854

theorem saree_price_calculation (final_price : ℝ) : 
  final_price = 248.625 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.15) * (1 - 0.25) = final_price ∧ 
    original_price = 390 := by
  sorry

end saree_price_calculation_l2168_216854


namespace crates_in_load_l2168_216840

/-- Represents the weight of vegetables in a delivery truck load --/
structure VegetableLoad where
  crateWeight : ℕ     -- Weight of one crate in kilograms
  cartonWeight : ℕ    -- Weight of one carton in kilograms
  numCartons : ℕ      -- Number of cartons in the load
  totalWeight : ℕ     -- Total weight of the load in kilograms

/-- Calculates the number of crates in a vegetable load --/
def numCrates (load : VegetableLoad) : ℕ :=
  (load.totalWeight - load.cartonWeight * load.numCartons) / load.crateWeight

/-- Theorem stating that for the given conditions, the number of crates is 12 --/
theorem crates_in_load :
  ∀ (load : VegetableLoad),
    load.crateWeight = 4 →
    load.cartonWeight = 3 →
    load.numCartons = 16 →
    load.totalWeight = 96 →
    numCrates load = 12 := by
  sorry

end crates_in_load_l2168_216840


namespace rohan_farm_earnings_l2168_216866

/-- Represents a coconut farm with given characteristics -/
structure CoconutFarm where
  size : ℕ  -- farm size in square meters
  trees_per_sqm : ℕ  -- number of trees per square meter
  coconuts_per_tree : ℕ  -- number of coconuts per tree
  harvest_frequency : ℕ  -- harvest frequency in months
  price_per_coconut : ℚ  -- price per coconut in dollars
  time_period : ℕ  -- time period in months

/-- Calculates the earnings from a coconut farm over a given time period -/
def calculate_earnings (farm : CoconutFarm) : ℚ :=
  let total_trees := farm.size * farm.trees_per_sqm
  let total_coconuts_per_harvest := total_trees * farm.coconuts_per_tree
  let number_of_harvests := farm.time_period / farm.harvest_frequency
  let total_coconuts := total_coconuts_per_harvest * number_of_harvests
  total_coconuts * farm.price_per_coconut

/-- Theorem stating that the earnings from Rohan's coconut farm after 6 months is $240 -/
theorem rohan_farm_earnings :
  let farm : CoconutFarm := {
    size := 20,
    trees_per_sqm := 2,
    coconuts_per_tree := 6,
    harvest_frequency := 3,
    price_per_coconut := 1/2,
    time_period := 6
  }
  calculate_earnings farm = 240 := by sorry

end rohan_farm_earnings_l2168_216866


namespace line_equation_conversion_l2168_216817

/-- Given a line expressed as (2, -1) · ((x, y) - (5, -3)) = 0, prove that when written in the form y = mx + b, m = 2 and b = -13 -/
theorem line_equation_conversion :
  ∀ (x y : ℝ),
  (2 : ℝ) * (x - 5) + (-1 : ℝ) * (y - (-3)) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -13 :=
by sorry

end line_equation_conversion_l2168_216817


namespace soccer_team_lineup_count_l2168_216806

theorem soccer_team_lineup_count :
  let total_players : ℕ := 18
  let goalie_count : ℕ := 1
  let defender_count : ℕ := 6
  let forward_count : ℕ := 4
  let remaining_after_goalie : ℕ := total_players - goalie_count
  let remaining_after_defenders : ℕ := remaining_after_goalie - defender_count
  (total_players.choose goalie_count) *
  (remaining_after_goalie.choose defender_count) *
  (remaining_after_defenders.choose forward_count) = 73457760 := by
sorry

end soccer_team_lineup_count_l2168_216806


namespace prime_factors_count_l2168_216865

/-- The total number of prime factors in the given expression -/
def total_prime_factors : ℕ :=
  (2 * 17) + (2 * 13) + (3 * 7) + (5 * 3) + (7 * 19)

/-- The theorem stating that the total number of prime factors in the given expression is 229 -/
theorem prime_factors_count :
  total_prime_factors = 229 := by
  sorry

end prime_factors_count_l2168_216865


namespace nonzero_real_equality_l2168_216805

theorem nonzero_real_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 1 + 1/y) (h2 : y = 1 + 1/x) : y = x := by
  sorry

end nonzero_real_equality_l2168_216805


namespace number_solution_l2168_216879

theorem number_solution : ∃ x : ℝ, 45 - 3 * x = 12 ∧ x = 11 := by sorry

end number_solution_l2168_216879


namespace unpacked_boxes_correct_l2168_216857

-- Define the cookie types
inductive CookieType
  | LemonChaletCremes
  | ThinMints
  | Samoas
  | Trefoils

-- Define the function for boxes per case
def boxesPerCase (c : CookieType) : ℕ :=
  match c with
  | CookieType.LemonChaletCremes => 12
  | CookieType.ThinMints => 15
  | CookieType.Samoas => 10
  | CookieType.Trefoils => 18

-- Define the function for boxes sold
def boxesSold (c : CookieType) : ℕ :=
  match c with
  | CookieType.LemonChaletCremes => 31
  | CookieType.ThinMints => 26
  | CookieType.Samoas => 17
  | CookieType.Trefoils => 44

-- Define the function for unpacked boxes
def unpackedBoxes (c : CookieType) : ℕ :=
  boxesSold c % boxesPerCase c

-- Theorem statement
theorem unpacked_boxes_correct (c : CookieType) :
  unpackedBoxes c =
    match c with
    | CookieType.LemonChaletCremes => 7
    | CookieType.ThinMints => 11
    | CookieType.Samoas => 7
    | CookieType.Trefoils => 8
  := by sorry

end unpacked_boxes_correct_l2168_216857


namespace problem_statement_l2168_216815

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end problem_statement_l2168_216815


namespace green_ball_probability_l2168_216801

-- Define the containers and their contents
def containerA : ℕ × ℕ := (10, 5)  -- (red balls, green balls)
def containerB : ℕ × ℕ := (3, 6)
def containerC : ℕ × ℕ := (3, 6)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Define the probability of selecting a green ball from each container
def greenProbA : ℚ := containerA.2 / (containerA.1 + containerA.2)
def greenProbB : ℚ := containerB.2 / (containerB.1 + containerB.2)
def greenProbC : ℚ := containerC.2 / (containerC.1 + containerC.2)

-- Theorem: The probability of selecting a green ball is 5/9
theorem green_ball_probability :
  containerProb * greenProbA +
  containerProb * greenProbB +
  containerProb * greenProbC = 5 / 9 := by
  sorry

end green_ball_probability_l2168_216801


namespace shifted_parabola_properties_l2168_216853

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- Theorem stating that the shifted parabola is a vertical translation of the original parabola
    and passes through the point (0, 3) -/
theorem shifted_parabola_properties :
  (∃ k : ℝ, ∀ x : ℝ, shifted_parabola x = original_parabola x + k) ∧
  shifted_parabola 0 = 3 := by
  sorry


end shifted_parabola_properties_l2168_216853


namespace sum_of_perimeters_l2168_216888

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * x + 4 * y = 4 * (Real.sqrt 63 + Real.sqrt 22) := by
sorry

end sum_of_perimeters_l2168_216888


namespace sum_base5_equals_l2168_216895

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

/-- The theorem to be proved -/
theorem sum_base5_equals : 
  decimalToBase5 (base5ToDecimal [1, 2, 3] + base5ToDecimal [4, 3, 2] + base5ToDecimal [2, 1, 4]) = 
  [1, 3, 2, 4] := by
  sorry

end sum_base5_equals_l2168_216895


namespace negation_existence_false_l2168_216899

theorem negation_existence_false : ¬(∀ x : ℝ, 2^x + x^2 > 1) := by
  sorry

end negation_existence_false_l2168_216899


namespace hockey_players_count_l2168_216855

/-- The number of hockey players in a games hour -/
def hockey_players (total players : ℕ) (cricket football softball : ℕ) : ℕ :=
  total - (cricket + football + softball)

/-- Theorem stating the number of hockey players -/
theorem hockey_players_count :
  hockey_players 51 10 16 13 = 12 := by
  sorry

end hockey_players_count_l2168_216855


namespace ball_probability_l2168_216832

theorem ball_probability (total white green yellow red purple black blue pink : ℕ) 
  (h_total : total = 500)
  (h_white : white = 200)
  (h_green : green = 80)
  (h_yellow : yellow = 70)
  (h_red : red = 57)
  (h_purple : purple = 33)
  (h_black : black = 30)
  (h_blue : blue = 16)
  (h_pink : pink = 14)
  (h_sum : white + green + yellow + red + purple + black + blue + pink = total) :
  (total - (red + purple + black)) / total = 19 / 25 := by
  sorry

end ball_probability_l2168_216832


namespace simplify_expression_l2168_216849

theorem simplify_expression (x : ℝ) : (3*x + 15) + (100*x + 15) + (10*x - 5) = 113*x + 25 := by
  sorry

end simplify_expression_l2168_216849


namespace f_2_equals_neg_26_l2168_216851

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_2_equals_neg_26 (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end f_2_equals_neg_26_l2168_216851


namespace nonagon_diagonals_l2168_216808

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l2168_216808


namespace jasons_shopping_expenses_l2168_216839

theorem jasons_shopping_expenses (total_spent jacket_price : ℚ) 
  (h1 : total_spent = 14.28)
  (h2 : jacket_price = 4.74) :
  total_spent - jacket_price = 9.54 := by
  sorry

end jasons_shopping_expenses_l2168_216839


namespace optimal_speed_l2168_216807

/-- Fuel cost per unit time as a function of speed -/
noncomputable def fuel_cost (v : ℝ) : ℝ := sorry

/-- Total cost per unit time as a function of speed -/
noncomputable def total_cost (v : ℝ) : ℝ := fuel_cost v + 560

/-- Cost per kilometer as a function of speed -/
noncomputable def cost_per_km (v : ℝ) : ℝ := total_cost v / v

theorem optimal_speed :
  ∃ (k : ℝ), fuel_cost v = k * v^3 ∧  -- Fuel cost is proportional to v^3
  fuel_cost 10 = 35 ∧                 -- At 10 km/h, fuel cost is 35 yuan/hour
  (∀ v, v ≤ 25) →                     -- Maximum speed is 25 km/h
  (∀ v, v > 0 → v ≤ 25 → cost_per_km 20 ≤ cost_per_km v) :=
sorry

end optimal_speed_l2168_216807


namespace imaginary_part_of_z_l2168_216816

theorem imaginary_part_of_z (z : ℂ) (h : z * ((1 + Complex.I)^2 / 2) = 1 + 2 * Complex.I) :
  z.im = -1 := by sorry

end imaginary_part_of_z_l2168_216816


namespace projection_a_onto_b_is_sqrt5_l2168_216800

/-- The projection of vector a onto the direction of vector b is √5 -/
theorem projection_a_onto_b_is_sqrt5 (a b : ℝ × ℝ) : 
  a = (1, 3) → a + b = (-1, 7) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end projection_a_onto_b_is_sqrt5_l2168_216800


namespace no_perfect_square_polynomial_l2168_216871

theorem no_perfect_square_polynomial (n : ℕ) : ∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 ≠ m^2 := by
  sorry

end no_perfect_square_polynomial_l2168_216871


namespace max_value_determines_parameter_l2168_216847

/-- Given a system of linear inequalities and an objective function,
    prove that the maximum value of the objective function
    determines the value of a parameter. -/
theorem max_value_determines_parameter
  (x y z a : ℝ)
  (h1 : x - 3 ≤ 0)
  (h2 : y - a ≤ 0)
  (h3 : x + y ≥ 0)
  (h4 : z = 2*x + y)
  (h5 : ∀ x' y' z', x' - 3 ≤ 0 → y' - a ≤ 0 → x' + y' ≥ 0 → z' = 2*x' + y' → z' ≤ 10)
  (h6 : ∃ x' y' z', x' - 3 ≤ 0 ∧ y' - a ≤ 0 ∧ x' + y' ≥ 0 ∧ z' = 2*x' + y' ∧ z' = 10) :
  a = 4 := by
sorry

end max_value_determines_parameter_l2168_216847


namespace fraction_transformation_impossibility_l2168_216880

theorem fraction_transformation_impossibility : ¬ ∃ (f : ℚ), (
  f = 5/8 ∧
  (∀ (n : ℕ), f = (f.num + n) / (f.den + n) ∨ f = (f.num * n) / (f.den * n)) ∧
  f = 3/5
) := by sorry

end fraction_transformation_impossibility_l2168_216880


namespace sqrt_less_than_y_plus_one_l2168_216838

theorem sqrt_less_than_y_plus_one (y : ℝ) (h : y > 0) : Real.sqrt y < y + 1 := by
  sorry

end sqrt_less_than_y_plus_one_l2168_216838


namespace five_divides_x_l2168_216856

theorem five_divides_x (x y : ℕ) (hx : x > 1) (heq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end five_divides_x_l2168_216856


namespace right_triangle_from_leg_and_projection_l2168_216813

/-- Right triangle determined by one leg and projection of other leg onto hypotenuse -/
theorem right_triangle_from_leg_and_projection
  (a c₂ : ℝ) (ha : a > 0) (hc₂ : c₂ > 0) :
  ∃! (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    c₂ * c = b^2 :=
by sorry

end right_triangle_from_leg_and_projection_l2168_216813


namespace prob_sum_five_or_nine_l2168_216894

def fair_dice_roll : Finset ℕ := Finset.range 6

def sum_outcomes (n : ℕ) : Finset (ℕ × ℕ) :=
  (fair_dice_roll.product fair_dice_roll).filter (fun p => p.1 + p.2 + 2 = n)

def prob_sum (n : ℕ) : ℚ :=
  (sum_outcomes n).card / (fair_dice_roll.card * fair_dice_roll.card : ℕ)

theorem prob_sum_five_or_nine :
  prob_sum 5 = 1/9 ∧ prob_sum 9 = 1/9 :=
sorry

end prob_sum_five_or_nine_l2168_216894


namespace min_sum_squares_with_real_solution_l2168_216882

theorem min_sum_squares_with_real_solution (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → a^2 + b^2 ≥ 4 := by
  sorry

end min_sum_squares_with_real_solution_l2168_216882


namespace partnership_profit_distribution_l2168_216844

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution 
  (total_profit : ℝ) 
  (h_profit : total_profit = 55000) 
  (invest_a invest_b invest_c : ℝ) 
  (h_a_b : invest_a = 3 * invest_b) 
  (h_a_c : invest_a = 2/3 * invest_c) : 
  invest_c / (invest_a + invest_b + invest_c) * total_profit = 9/17 * 55000 := by
sorry

end partnership_profit_distribution_l2168_216844


namespace f_of_one_equals_fourteen_l2168_216818

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

-- State the theorem
theorem f_of_one_equals_fourteen 
  (a b : ℝ) -- Parameters of the function
  (h : f a b (-1) = 10) -- Given condition
  : f a b 1 = 14 := by
  sorry -- Proof is omitted

end f_of_one_equals_fourteen_l2168_216818


namespace negation_of_universal_quantifier_l2168_216822

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) := by
  sorry

end negation_of_universal_quantifier_l2168_216822


namespace system_solution_l2168_216852

theorem system_solution :
  ∃! (x y : ℝ), 
    (x + 2*y = (7 - x) + (7 - 2*y)) ∧ 
    (3*x - 2*y = (x + 2) - (2*y + 2)) ∧
    x = 0 ∧ y = 7/2 := by
  sorry

end system_solution_l2168_216852


namespace f_min_value_l2168_216858

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := (x^2 + 4*x + 5)*(x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

/-- Theorem stating that the minimum value of f(x) is -9 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -9 := by sorry

end f_min_value_l2168_216858


namespace sin_cos_cube_difference_squared_l2168_216883

theorem sin_cos_cube_difference_squared (θ : Real) 
  (h : Real.sin θ - Real.cos θ = (Real.sqrt 6 - Real.sqrt 2) / 2) : 
  24 * (Real.sin θ ^ 3 - Real.cos θ ^ 3) ^ 2 = 12 := by
  sorry

end sin_cos_cube_difference_squared_l2168_216883


namespace pascal_triangle_51_numbers_l2168_216841

theorem pascal_triangle_51_numbers (n : ℕ) : 
  (n + 1 = 51) → Nat.choose n 2 = 1225 := by
  sorry

end pascal_triangle_51_numbers_l2168_216841


namespace reciprocal_sum_theorem_l2168_216824

theorem reciprocal_sum_theorem (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1 / x + 1 / y = 1 / z) : z = (x * y) / (x + y) := by
  sorry

end reciprocal_sum_theorem_l2168_216824


namespace cat_or_bird_percentage_l2168_216802

/-- Represents the survey data from a high school -/
structure SurveyData where
  total_students : ℕ
  dog_owners : ℕ
  cat_owners : ℕ
  bird_owners : ℕ

/-- Calculates the percentage of students owning either cats or birds -/
def percentage_cat_or_bird (data : SurveyData) : ℚ :=
  (data.cat_owners + data.bird_owners : ℚ) / data.total_students * 100

/-- The survey data from the high school -/
def high_school_survey : SurveyData :=
  { total_students := 400
  , dog_owners := 80
  , cat_owners := 50
  , bird_owners := 20 }

/-- Theorem stating that the percentage of students owning either cats or birds is 17.5% -/
theorem cat_or_bird_percentage :
  percentage_cat_or_bird high_school_survey = 35/2 := by
  sorry

end cat_or_bird_percentage_l2168_216802


namespace square_dissection_divisible_perimeter_l2168_216850

theorem square_dissection_divisible_perimeter (n : Nat) (h : n = 2015) :
  ∃ (a b : Nat), a ≤ n ∧ b ≤ n ∧ (2 * (a + b)) % 4 = 0 := by
  sorry

end square_dissection_divisible_perimeter_l2168_216850


namespace car_rental_budget_is_75_l2168_216811

/-- Calculates the budget for a car rental given the daily rate, per-mile rate, and miles driven. -/
def carRentalBudget (dailyRate : ℝ) (perMileRate : ℝ) (milesDriven : ℝ) : ℝ :=
  dailyRate + perMileRate * milesDriven

/-- Theorem: The budget for a car rental with specific rates and mileage is $75.00. -/
theorem car_rental_budget_is_75 :
  let dailyRate : ℝ := 30
  let perMileRate : ℝ := 0.18
  let milesDriven : ℝ := 250.0
  carRentalBudget dailyRate perMileRate milesDriven = 75 := by
  sorry

end car_rental_budget_is_75_l2168_216811


namespace valid_sequences_l2168_216881

def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 8 ∧
  s.count 1 = 2 ∧
  s.count 2 = 2 ∧
  s.count 3 = 2 ∧
  s.count 4 = 2 ∧
  ∃ i j, i + 2 = j ∧ s[i]? = some 1 ∧ s[j]? = some 1 ∧
  ∃ i j, i + 3 = j ∧ s[i]? = some 2 ∧ s[j]? = some 2 ∧
  ∃ i j, i + 4 = j ∧ s[i]? = some 3 ∧ s[j]? = some 3 ∧
  ∃ i j, i + 5 = j ∧ s[i]? = some 4 ∧ s[j]? = some 4

theorem valid_sequences :
  is_valid_sequence [4, 1, 3, 1, 2, 4, 3, 2] ∧
  is_valid_sequence [2, 3, 4, 2, 1, 3, 1, 4] :=
by sorry

end valid_sequences_l2168_216881


namespace courier_strategy_l2168_216828

-- Define the probability of a courier being robbed
variable (p : ℝ) 

-- Define the probability of failure for each strategy
def P2 : ℝ := p^2
def P3 : ℝ := p^2 * (3 - 2*p)
def P4 : ℝ := p^3 * (4 - 3*p)

-- Define the theorem
theorem courier_strategy (h1 : 0 < p) (h2 : p < 1) :
  (0 < p ∧ p < 1/3 → 1 - P4 p > max (1 - P2 p) (1 - P3 p)) ∧
  (1/3 ≤ p ∧ p < 1 → 1 - P2 p ≥ max (1 - P3 p) (1 - P4 p)) :=
sorry

end courier_strategy_l2168_216828


namespace negation_of_existence_negation_of_cubic_inequality_l2168_216846

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_cubic_inequality : 
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end negation_of_existence_negation_of_cubic_inequality_l2168_216846


namespace five_squared_minus_nine_over_five_minus_three_equals_eight_l2168_216878

theorem five_squared_minus_nine_over_five_minus_three_equals_eight :
  (5^2 - 9) / (5 - 3) = 8 := by
  sorry

end five_squared_minus_nine_over_five_minus_three_equals_eight_l2168_216878


namespace dodecagon_diagonals_l2168_216810

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l2168_216810


namespace binary_111_equals_7_l2168_216835

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 111 -/
def binary_111 : List Bool := [true, true, true]

theorem binary_111_equals_7 : binary_to_decimal binary_111 = 7 := by
  sorry

end binary_111_equals_7_l2168_216835


namespace adult_ticket_cost_l2168_216827

theorem adult_ticket_cost (total_tickets : ℕ) (senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  total_tickets = 510 →
  senior_price = 15 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  (total_tickets - senior_tickets) * (total_receipts - senior_tickets * senior_price) / (total_tickets - senior_tickets) = 21 :=
by sorry

end adult_ticket_cost_l2168_216827


namespace distance_between_stations_l2168_216845

/-- The distance between two stations given train travel times and speeds -/
theorem distance_between_stations 
  (train1_speed : ℝ) (train1_time : ℝ) 
  (train2_speed : ℝ) (train2_time : ℝ) 
  (h1 : train1_speed = 20)
  (h2 : train1_time = 5)
  (h3 : train2_speed = 25)
  (h4 : train2_time = 4) :
  train1_speed * train1_time + train2_speed * train2_time = 200 := by
  sorry

#check distance_between_stations

end distance_between_stations_l2168_216845


namespace circles_intersect_l2168_216864

/-- Definition of Circle O₁ -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- Definition of Circle O₂ -/
def circle_O₂ (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- Center of Circle O₁ -/
def center_O₁ : ℝ × ℝ := (0, 1)

/-- Center of Circle O₂ -/
def center_O₂ : ℝ × ℝ := (1, 2)

/-- Radius of Circle O₁ -/
def radius_O₁ : ℝ := 1

/-- Radius of Circle O₂ -/
def radius_O₂ : ℝ := 2

/-- Theorem: Circles O₁ and O₂ are intersecting -/
theorem circles_intersect : 
  (radius_O₁ + radius_O₂ > Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2)) ∧
  (Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2) > |radius_O₂ - radius_O₁|) :=
by sorry

end circles_intersect_l2168_216864


namespace geometric_sequence_second_term_l2168_216831

/-- A geometric sequence with fifth term 48 and sixth term 72 has second term 384/27 -/
theorem geometric_sequence_second_term :
  ∀ (a : ℚ) (r : ℚ),
  a * r^4 = 48 →
  a * r^5 = 72 →
  a * r = 384/27 :=
by sorry

end geometric_sequence_second_term_l2168_216831


namespace digit_sum_problem_l2168_216820

theorem digit_sum_problem :
  ∃ (a b c d : ℕ),
    (1000 ≤ a ∧ a < 10000) ∧
    (1000 ≤ b ∧ b < 10000) ∧
    (1000 ≤ c ∧ c < 10000) ∧
    (1000 ≤ d ∧ d < 10000) ∧
    a + b = 4300 ∧
    c - d = 1542 ∧
    a + c = 5842 :=
by sorry

end digit_sum_problem_l2168_216820


namespace mult_func_property_l2168_216837

/-- A function satisfying f(a+b) = f(a) * f(b) for all real a, b -/
def MultFunc (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a * f b

theorem mult_func_property (f : ℝ → ℝ) (h1 : MultFunc f) (h2 : f 1 = 2) :
  f 0 + f 3 = 9 := by
  sorry

end mult_func_property_l2168_216837


namespace functional_equation_solutions_l2168_216873

noncomputable def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y

theorem functional_equation_solutions (f : ℝ → ℝ) :
  FunctionalEquation f →
  (∀ x : ℝ, f x = 0) ∨
  (∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → f x = 1) ∧ f 0 = a) :=
by sorry

end functional_equation_solutions_l2168_216873


namespace arithmetic_geometric_sequence_relation_l2168_216875

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- A geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  h_geometric : ∀ n, b (n + 1) = b n * q

/-- The main theorem -/
theorem arithmetic_geometric_sequence_relation
  (a : ArithmeticSequence)
  (b : GeometricSequence)
  (h_consecutive : b.b 1 = a.a 5 ∧ b.b 2 = a.a 8 ∧ b.b 3 = a.a 13)
  (h_b2 : b.b 2 = 5) :
  ∀ n, b.b n = 5 * (5/3)^(n-2) := by
  sorry

end arithmetic_geometric_sequence_relation_l2168_216875


namespace greater_than_implies_greater_than_scaled_and_shifted_l2168_216876

theorem greater_than_implies_greater_than_scaled_and_shifted {a b : ℝ} (h : a > b) : 3*a + 5 > 3*b + 5 := by
  sorry

end greater_than_implies_greater_than_scaled_and_shifted_l2168_216876


namespace ben_marbles_count_l2168_216860

theorem ben_marbles_count (ben_marbles : ℕ) (leo_marbles : ℕ) : 
  (leo_marbles = ben_marbles + 20) →
  (ben_marbles + leo_marbles = 132) →
  (ben_marbles = 56) := by
sorry

end ben_marbles_count_l2168_216860


namespace num_factors_of_given_number_l2168_216870

/-- The number of distinct, natural-number factors of 4³ * 5⁴ * 6² -/
def num_factors : ℕ := 135

/-- The given number -/
def given_number : ℕ := 4^3 * 5^4 * 6^2

theorem num_factors_of_given_number :
  (Finset.filter (· ∣ given_number) (Finset.range (given_number + 1))).card = num_factors := by
  sorry

end num_factors_of_given_number_l2168_216870


namespace gold_coin_distribution_l2168_216872

theorem gold_coin_distribution (x y : ℕ) (h1 : x > y) (h2 : x + y = 49) :
  ∃ (k : ℕ), x^2 - y^2 = k * (x - y) → k = 49 := by
  sorry

end gold_coin_distribution_l2168_216872
