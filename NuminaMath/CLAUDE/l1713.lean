import Mathlib

namespace arithmetic_sequence_11th_term_l1713_171305

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2 : a 2 = 3)
  (h_6 : a 6 = 7) :
  a 11 = 12 :=
sorry

end arithmetic_sequence_11th_term_l1713_171305


namespace jeff_storage_usage_l1713_171325

theorem jeff_storage_usage (total_storage : ℝ) (storage_per_song : ℝ) (num_songs : ℕ) (mb_per_gb : ℕ) :
  total_storage = 16 →
  storage_per_song = 30 / 1000 →
  num_songs = 400 →
  mb_per_gb = 1000 →
  total_storage - (↑num_songs * storage_per_song) = 4 :=
by sorry

end jeff_storage_usage_l1713_171325


namespace bret_caught_12_frogs_l1713_171358

-- Define the number of frogs caught by each person
def alster_frogs : ℕ := 2
def quinn_frogs : ℕ := 2 * alster_frogs
def bret_frogs : ℕ := 3 * quinn_frogs

-- Theorem to prove
theorem bret_caught_12_frogs : bret_frogs = 12 := by
  sorry

end bret_caught_12_frogs_l1713_171358


namespace inequality_proof_l1713_171348

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) : 
  (a^b * b^c * c^d * d^a) / (b^d * c^b * d^c * a^d) ≥ 1 := by
  sorry

end inequality_proof_l1713_171348


namespace probability_gpa_at_least_3_6_l1713_171388

/-- Grade points for each letter grade -/
def gradePoints (grade : Char) : ℕ :=
  match grade with
  | 'A' => 4
  | 'B' => 3
  | 'C' => 2
  | 'D' => 1
  | _ => 0

/-- Calculate GPA given a list of grades -/
def calculateGPA (grades : List Char) : ℚ :=
  (grades.map gradePoints).sum / 5

/-- Probability of getting an A in English -/
def pEnglishA : ℚ := 1/4

/-- Probability of getting a B in English -/
def pEnglishB : ℚ := 1/2

/-- Probability of getting an A in History -/
def pHistoryA : ℚ := 2/5

/-- Probability of getting a B in History -/
def pHistoryB : ℚ := 1/2

/-- Theorem stating the probability of achieving a GPA of at least 3.6 -/
theorem probability_gpa_at_least_3_6 :
  let p := pEnglishA * pHistoryA + pEnglishA * pHistoryB + pEnglishB * pHistoryA
  p = 17/40 := by sorry

end probability_gpa_at_least_3_6_l1713_171388


namespace average_of_solutions_is_zero_l1713_171339

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 49}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end average_of_solutions_is_zero_l1713_171339


namespace orange_weight_l1713_171341

theorem orange_weight (apple_weight : ℕ) (bag_capacity : ℕ) (num_bags : ℕ) (total_apple_weight : ℕ) :
  apple_weight = 4 →
  bag_capacity = 49 →
  num_bags = 3 →
  total_apple_weight = 84 →
  ∃ (orange_weight : ℕ),
    orange_weight * (total_apple_weight / apple_weight) = total_apple_weight ∧
    orange_weight = 4 :=
by
  sorry

#check orange_weight

end orange_weight_l1713_171341


namespace snowdrift_depth_ratio_l1713_171335

theorem snowdrift_depth_ratio (initial_depth second_day_depth third_day_snow fourth_day_snow final_depth : ℝ) :
  initial_depth = 20 →
  third_day_snow = 6 →
  fourth_day_snow = 18 →
  final_depth = 34 →
  second_day_depth + third_day_snow + fourth_day_snow = final_depth →
  second_day_depth / initial_depth = 1 / 2 := by
  sorry

end snowdrift_depth_ratio_l1713_171335


namespace arithmetic_sequence_remainder_l1713_171394

theorem arithmetic_sequence_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 8 →
  aₙ = 283 →
  n = (aₙ - a₁) / d + 1 →
  (n * (a₁ + aₙ) / 2) % 8 = 4 := by
  sorry

end arithmetic_sequence_remainder_l1713_171394


namespace mat_weaving_problem_l1713_171373

/-- Given that 4 mat-weaves can weave 4 mats in 4 days, 
    prove that 8 mat-weaves will weave 16 mats in 8 days. -/
theorem mat_weaving_problem (weave_rate : ℕ → ℕ → ℕ → ℕ) :
  weave_rate 4 4 4 = 4 →
  weave_rate 8 16 8 = 16 :=
by
  sorry

end mat_weaving_problem_l1713_171373


namespace H_points_infinite_but_not_all_l1713_171375

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 = 4}

-- Define what it means to be an H point
def is_H_point (P : ℝ × ℝ) : Prop :=
  P ∈ C ∧ ∃ (A B : ℝ × ℝ) (k m : ℝ),
    A ∈ C ∧ B ∈ l ∧
    (∀ x y, y = k * x + m ↔ (x, y) ∈ ({P, A, B} : Set (ℝ × ℝ))) ∧
    (dist P A = dist P B ∨ dist P A = dist A B)

-- Define the set of H points
def H_points : Set (ℝ × ℝ) := {P | is_H_point P}

-- The theorem to be proved
theorem H_points_infinite_but_not_all :
  Set.Infinite H_points ∧ H_points ≠ C :=
sorry


end H_points_infinite_but_not_all_l1713_171375


namespace smallest_number_meeting_criteria_l1713_171336

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def count_even_digits (n : ℕ) : ℕ :=
  (if n % 2 = 0 then 1 else 0) +
  (if (n / 10) % 2 = 0 then 1 else 0) +
  (if (n / 100) % 2 = 0 then 1 else 0) +
  (if (n / 1000) % 2 = 0 then 1 else 0)

def count_odd_digits (n : ℕ) : ℕ := 4 - count_even_digits n

def meets_criteria (n : ℕ) : Prop :=
  is_four_digit n ∧
  divisible_by_9 n ∧
  count_even_digits n = 3 ∧
  count_odd_digits n = 1

theorem smallest_number_meeting_criteria :
  ∀ n : ℕ, meets_criteria n → n ≥ 2043 :=
sorry

end smallest_number_meeting_criteria_l1713_171336


namespace chandler_can_buy_bike_l1713_171343

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 500

/-- The total birthday money Chandler received in dollars -/
def birthday_money : ℕ := 50 + 35 + 15

/-- Chandler's weekly earnings from the paper route in dollars -/
def weekly_earnings : ℕ := 16

/-- The number of weeks required to save enough money for the bike -/
def weeks_to_save : ℕ := 25

/-- Theorem stating that Chandler can buy the bike after saving for 25 weeks -/
theorem chandler_can_buy_bike : 
  birthday_money + weekly_earnings * weeks_to_save = bike_cost :=
sorry

end chandler_can_buy_bike_l1713_171343


namespace vector_decomposition_l1713_171381

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![(-15), 5, 6]
def p : Fin 3 → ℝ := ![0, 5, 1]
def q : Fin 3 → ℝ := ![3, 2, (-1)]
def r : Fin 3 → ℝ := ![(-1), 1, 0]

/-- The theorem to be proved -/
theorem vector_decomposition :
  x = (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r := by
  sorry

end vector_decomposition_l1713_171381


namespace measure_45_minutes_l1713_171312

/-- Represents a string that can be burned --/
structure BurnableString where
  burnTime : ℝ
  nonUniformRate : Bool

/-- Represents a lighter --/
structure Lighter

/-- Represents the state of burning strings --/
inductive BurningState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Function to measure time using burnable strings and a lighter --/
def measureTime (strings : List BurnableString) (lighter : Lighter) : ℝ :=
  sorry

/-- Theorem stating that 45 minutes can be measured --/
theorem measure_45_minutes :
  ∃ (strings : List BurnableString) (lighter : Lighter),
    strings.length = 2 ∧
    (∀ s ∈ strings, s.burnTime = 1 ∧ s.nonUniformRate = true) ∧
    measureTime strings lighter = 0.75 := by
  sorry

end measure_45_minutes_l1713_171312


namespace prob_two_queens_or_at_least_one_jack_l1713_171327

def standard_deck_size : ℕ := 52
def jack_count : ℕ := 4
def queen_count : ℕ := 4

def probability_two_queens_or_at_least_one_jack : ℚ :=
  217 / 882

theorem prob_two_queens_or_at_least_one_jack :
  probability_two_queens_or_at_least_one_jack = 
    (Nat.choose queen_count 2 * (standard_deck_size - queen_count) + 
     (standard_deck_size - jack_count).choose 2 * jack_count + 
     (standard_deck_size - jack_count).choose 1 * Nat.choose jack_count 2 + 
     Nat.choose jack_count 3) / 
    Nat.choose standard_deck_size 3 :=
by
  sorry

#eval probability_two_queens_or_at_least_one_jack

end prob_two_queens_or_at_least_one_jack_l1713_171327


namespace equation_solution_l1713_171350

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3

-- Theorem statement
theorem equation_solution :
  ∃ (x : ℝ), x > 9 ∧ equation x ∧ x = 21 := by
  sorry

end equation_solution_l1713_171350


namespace three_digit_palindrome_squares_l1713_171382

/-- A number is a 3-digit palindrome square if it satisfies these conditions:
1. It is between 100 and 999 (inclusive).
2. It is a perfect square.
3. It reads the same forward and backward. -/
def is_three_digit_palindrome_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  ∃ k, n = k^2 ∧
  (n / 100 = n % 10) ∧ (n / 10 % 10 = (n / 10) % 10)

/-- There are exactly 3 numbers that are 3-digit palindrome squares. -/
theorem three_digit_palindrome_squares :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_three_digit_palindrome_square n) ∧ s.card = 3 :=
sorry

end three_digit_palindrome_squares_l1713_171382


namespace bob_monthly_hours_l1713_171391

/-- Calculates the total hours worked in a month given daily hours, workdays per week, and average weeks per month. -/
def total_monthly_hours (daily_hours : ℝ) (workdays_per_week : ℝ) (avg_weeks_per_month : ℝ) : ℝ :=
  daily_hours * workdays_per_week * avg_weeks_per_month

/-- Proves that Bob's total monthly hours are approximately 216.5 -/
theorem bob_monthly_hours :
  let daily_hours : ℝ := 10
  let workdays_per_week : ℝ := 5
  let avg_weeks_per_month : ℝ := 4.33
  abs (total_monthly_hours daily_hours workdays_per_week avg_weeks_per_month - 216.5) < 0.1 := by
  sorry

#eval total_monthly_hours 10 5 4.33

end bob_monthly_hours_l1713_171391


namespace gcd_lcm_sum_l1713_171397

theorem gcd_lcm_sum : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by
  sorry

end gcd_lcm_sum_l1713_171397


namespace profit_difference_theorem_l1713_171366

/-- Calculates the difference between profit shares of two partners given investments and one partner's profit share. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_invest := invest_a + invest_b + invest_c
  let profit_per_unit := b_profit * total_invest / invest_b
  let a_profit := profit_per_unit * invest_a / total_invest
  let c_profit := profit_per_unit * invest_c / total_invest
  c_profit - a_profit

/-- Theorem stating that given the investments and B's profit share, the difference between A's and C's profit shares is 560. -/
theorem profit_difference_theorem :
  profit_share_difference 8000 10000 12000 1400 = 560 := by
  sorry

end profit_difference_theorem_l1713_171366


namespace largest_common_divisor_l1713_171306

theorem largest_common_divisor : ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 540 ∧ n < 60 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 ∧ m < 60 ∧ m ∣ 180 → m ≤ n :=
by sorry

end largest_common_divisor_l1713_171306


namespace amanda_ticket_sales_l1713_171303

/-- Amanda's ticket sales problem -/
theorem amanda_ticket_sales 
  (total_goal : ℕ) 
  (friends : ℕ) 
  (tickets_per_friend : ℕ) 
  (second_day_sales : ℕ) 
  (h1 : total_goal = 80)
  (h2 : friends = 5)
  (h3 : tickets_per_friend = 4)
  (h4 : second_day_sales = 32) :
  total_goal - (friends * tickets_per_friend + second_day_sales) = 28 := by
sorry


end amanda_ticket_sales_l1713_171303


namespace largest_unexpressible_sum_l1713_171383

def min_num : ℕ := 135
def max_num : ℕ := 144
def target : ℕ := 2024

theorem largest_unexpressible_sum : 
  (∀ n : ℕ, n > target → ∃ k : ℕ, k * min_num ≤ n ∧ n ≤ k * max_num) ∧
  (∀ k : ℕ, k * min_num > target ∨ target > k * max_num) :=
sorry

end largest_unexpressible_sum_l1713_171383


namespace estimated_probability_is_two_ninths_l1713_171359

/-- Represents the outcome of a single trial -/
inductive Outcome
| StopOnThird
| Other

/-- Represents the result of a random simulation -/
structure SimulationResult :=
  (trials : Nat)
  (stopsOnThird : Nat)

/-- Calculates the estimated probability -/
def estimateProbability (result : SimulationResult) : Rat :=
  result.stopsOnThird / result.trials

theorem estimated_probability_is_two_ninths 
  (result : SimulationResult)
  (h1 : result.trials = 18)
  (h2 : result.stopsOnThird = 4) :
  estimateProbability result = 2 / 9 := by
  sorry

end estimated_probability_is_two_ninths_l1713_171359


namespace floor_times_x_eq_54_l1713_171398

theorem floor_times_x_eq_54 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 ∧ abs (x - 7.7143) < 0.0001 := by
  sorry

end floor_times_x_eq_54_l1713_171398


namespace sum_of_odd_numbers_less_than_100_eq_2500_l1713_171347

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def sum_of_odd_numbers_less_than_100 : ℕ := 
  (Finset.range 50).sum (λ i => 2*i + 1)

theorem sum_of_odd_numbers_less_than_100_eq_2500 : 
  sum_of_odd_numbers_less_than_100 = 2500 := by
  sorry

end sum_of_odd_numbers_less_than_100_eq_2500_l1713_171347


namespace total_lives_after_third_level_l1713_171386

-- Define the game parameters
def initial_lives : ℕ := 2
def enemies_defeated : ℕ := 5
def powerups_collected : ℕ := 4
def first_level_penalty : ℕ := 3
def second_level_modifier (x : ℕ) : ℕ := x / 2

-- Define the game rules
def first_level_lives (x : ℕ) : ℕ := initial_lives + 2 * x - first_level_penalty

def second_level_lives (first_level : ℕ) (y : ℕ) : ℕ :=
  first_level + 3 * y - second_level_modifier first_level

def third_level_bonus (x y : ℕ) : ℕ := x + 2 * y - 5

-- The main theorem
theorem total_lives_after_third_level :
  let first_level := first_level_lives enemies_defeated
  let second_level := second_level_lives first_level powerups_collected
  second_level + third_level_bonus enemies_defeated powerups_collected = 25 := by
  sorry

end total_lives_after_third_level_l1713_171386


namespace concert_ticket_cost_l1713_171301

theorem concert_ticket_cost (num_tickets : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) (total_paid : ℚ) :
  num_tickets = 12 →
  discount_rate = 5 / 100 →
  discount_threshold = 10 →
  total_paid = 476 →
  ∃ (original_cost : ℚ), 
    original_cost * (num_tickets - discount_rate * (num_tickets - discount_threshold)) = total_paid ∧
    original_cost = 40 := by
  sorry

end concert_ticket_cost_l1713_171301


namespace contrapositive_example_l1713_171344

theorem contrapositive_example (a b : ℝ) :
  (¬(a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0)) := by sorry

end contrapositive_example_l1713_171344


namespace fraction_product_simplification_l1713_171361

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end fraction_product_simplification_l1713_171361


namespace haleigh_leggings_count_l1713_171369

/-- The number of leggings needed for Haleigh's pets -/
def total_leggings : ℕ :=
  let num_dogs := 4
  let num_cats := 3
  let num_spiders := 2
  let num_parrots := 1
  let dog_legs := 4
  let cat_legs := 4
  let spider_legs := 8
  let parrot_legs := 2
  num_dogs * dog_legs + num_cats * cat_legs + num_spiders * spider_legs + num_parrots * parrot_legs

theorem haleigh_leggings_count : total_leggings = 46 := by
  sorry

end haleigh_leggings_count_l1713_171369


namespace number_whose_quarter_is_nine_more_l1713_171372

theorem number_whose_quarter_is_nine_more (x : ℚ) : (x / 4 = x + 9) → x = -12 := by
  sorry

end number_whose_quarter_is_nine_more_l1713_171372


namespace basketball_lineup_combinations_l1713_171355

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 6
def max_quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations : 
  (Nat.choose (total_players - quadruplets) starters) + 
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) + 
  (Nat.choose quadruplets 2 * Nat.choose (total_players - quadruplets) (starters - 2)) = 4290 :=
sorry

end basketball_lineup_combinations_l1713_171355


namespace arithmetic_sequence_sum_l1713_171368

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_sum : a 1 + a 13 = 10) : 
  a 3 + a 5 + a 7 + a 9 + a 11 = 25 := by
  sorry

end arithmetic_sequence_sum_l1713_171368


namespace arithmetic_calculations_l1713_171321

theorem arithmetic_calculations :
  ((1 : ℤ) - 4 + 8 - 5 = -1) ∧ 
  (24 / (-3 : ℤ) - (-2)^3 = 0) := by
  sorry

end arithmetic_calculations_l1713_171321


namespace f_value_at_3_l1713_171311

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) :
  f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end f_value_at_3_l1713_171311


namespace square_difference_equals_one_l1713_171353

theorem square_difference_equals_one (a b : ℝ) (h : a - b = 1) :
  a^2 - b^2 - 2*b = 1 := by sorry

end square_difference_equals_one_l1713_171353


namespace smallest_angle_measure_l1713_171309

/-- A cyclic quadrilateral with angles in arithmetic progression -/
structure CyclicQuadrilateral where
  -- The smallest angle
  a : ℝ
  -- The common difference in the arithmetic progression
  d : ℝ
  -- Ensures the quadrilateral is cyclic (opposite angles sum to 180°)
  cyclic : a + (a + 3*d) = 180 ∧ (a + d) + (a + 2*d) = 180
  -- Ensures the angles form an arithmetic sequence
  arithmetic_seq : true
  -- The largest angle is 140°
  largest_angle : a + 3*d = 140

/-- 
In a cyclic quadrilateral where the angles form an arithmetic sequence 
and the largest angle is 140°, the smallest angle measures 40°
-/
theorem smallest_angle_measure (q : CyclicQuadrilateral) : q.a = 40 := by
  sorry

end smallest_angle_measure_l1713_171309


namespace remaining_shirt_cost_l1713_171318

/-- Given a set of shirts with known prices, calculate the price of the remaining shirts -/
theorem remaining_shirt_cost (total_shirts : ℕ) (total_cost : ℚ) (known_shirt_count : ℕ) (known_shirt_cost : ℚ) :
  total_shirts = 5 →
  total_cost = 85 →
  known_shirt_count = 3 →
  known_shirt_cost = 15 →
  (total_cost - (known_shirt_count * known_shirt_cost)) / (total_shirts - known_shirt_count) = 20 :=
by
  sorry

end remaining_shirt_cost_l1713_171318


namespace fraction_irreducibility_l1713_171307

theorem fraction_irreducibility (n : ℕ) : 
  (Nat.gcd (2*n^2 + 11*n - 18) (n + 7) = 1) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end fraction_irreducibility_l1713_171307


namespace pitcher_distribution_percentage_l1713_171316

/-- Represents the contents of a pitcher -/
structure Pitcher :=
  (capacity : ℝ)
  (orange_juice : ℝ)
  (apple_juice : ℝ)

/-- Represents the distribution of the pitcher contents into cups -/
structure Distribution :=
  (pitcher : Pitcher)
  (num_cups : ℕ)

/-- The theorem stating the percentage of the pitcher's capacity in each cup -/
theorem pitcher_distribution_percentage (d : Distribution) 
  (h1 : d.pitcher.orange_juice = 2/3 * d.pitcher.capacity)
  (h2 : d.pitcher.apple_juice = 1/3 * d.pitcher.capacity)
  (h3 : d.num_cups = 6)
  (h4 : d.pitcher.capacity > 0) :
  (d.pitcher.capacity / d.num_cups) / d.pitcher.capacity = 1/6 := by
  sorry

#check pitcher_distribution_percentage

end pitcher_distribution_percentage_l1713_171316


namespace third_degree_polynomial_theorem_l1713_171364

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- Property that |g(x)| = 18 for x = 1, 2, 3, 4, 5, 6 -/
def has_absolute_value_18 (g : ThirdDegreePolynomial) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 6 → |g x| = 18

/-- Main theorem: If g is a third-degree polynomial with |g(x)| = 18 for x = 1, 2, 3, 4, 5, 6, then |g(0)| = 162 -/
theorem third_degree_polynomial_theorem (g : ThirdDegreePolynomial) 
  (h : has_absolute_value_18 g) : |g 0| = 162 := by
  sorry

end third_degree_polynomial_theorem_l1713_171364


namespace square_area_ratio_l1713_171310

/-- If the perimeter of one square is 4 times the perimeter of another square,
    then the area of the larger square is 16 times the area of the smaller square. -/
theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l1713_171310


namespace common_root_of_polynomials_l1713_171328

theorem common_root_of_polynomials (a b c d e f g : ℚ) :
  ∃ k : ℚ, k < 0 ∧ k ≠ ⌊k⌋ ∧
  (90 * k^4 + a * k^3 + b * k^2 + c * k + 18 = 0) ∧
  (18 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 90 = 0) :=
by
  use -1/6
  sorry

end common_root_of_polynomials_l1713_171328


namespace trivia_game_total_points_l1713_171349

/-- Given the points scored by three teams in a trivia game, prove that the total points scored is 15. -/
theorem trivia_game_total_points (team_a team_b team_c : ℕ) 
  (h1 : team_a = 2) 
  (h2 : team_b = 9) 
  (h3 : team_c = 4) : 
  team_a + team_b + team_c = 15 := by
  sorry

end trivia_game_total_points_l1713_171349


namespace letter_drawing_probabilities_l1713_171324

/-- Represents a set of letters -/
def LetterSet := Finset Char

/-- Represents a word as a list of characters -/
def Word := List Char

/-- Calculate the number of ways to arrange n items from a set of k items -/
def arrangements (n k : ℕ) : ℕ := 
  (k - n + 1).factorial / (k - n).factorial

/-- Calculate the probability of drawing a specific sequence -/
def probability_specific_sequence (total_letters : ℕ) (word_length : ℕ) : ℚ :=
  1 / arrangements word_length total_letters

/-- Calculate the probability of drawing a sequence with repeated letters -/
def probability_repeated_sequence (total_letters : ℕ) (word_length : ℕ) (permutations : ℕ) : ℚ :=
  permutations / arrangements word_length total_letters

/-- The main theorem to prove -/
theorem letter_drawing_probabilities 
  (s1 : LetterSet) 
  (w1 : Word)
  (s2 : LetterSet)
  (w2 : Word) :
  s1.card = 6 →
  w1.length = 4 →
  s2.card = 6 →
  w2.length = 4 →
  probability_specific_sequence 6 4 = 1 / 360 ∧
  probability_repeated_sequence 6 4 12 = 1 / 30 :=
by sorry

end letter_drawing_probabilities_l1713_171324


namespace peach_difference_l1713_171322

theorem peach_difference (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 17) 
  (h2 : green_peaches = 16) : 
  red_peaches - green_peaches = 1 := by
  sorry

end peach_difference_l1713_171322


namespace vector_equation_m_range_l1713_171377

theorem vector_equation_m_range :
  ∀ (m n x : ℝ),
  (∃ x, (n + 2, n - Real.cos x ^ 2) = (2 * m, m + Real.sin x)) →
  (∀ m', (∃ n' x', (n' + 2, n' - Real.cos x' ^ 2) = (2 * m', m' + Real.sin x')) → 
    0 ≤ m' ∧ m' ≤ 4) ∧
  (∃ n₁ x₁, (n₁ + 2, n₁ - Real.cos x₁ ^ 2) = (2 * 0, 0 + Real.sin x₁)) ∧
  (∃ n₂ x₂, (n₂ + 2, n₂ - Real.cos x₂ ^ 2) = (2 * 4, 4 + Real.sin x₂)) :=
by sorry

end vector_equation_m_range_l1713_171377


namespace derivative_from_limit_l1713_171370

theorem derivative_from_limit (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (x₀ - 2*Δx) - f x₀) / Δx - 2| < ε) →
  deriv f x₀ = -1 := by sorry

end derivative_from_limit_l1713_171370


namespace overall_profit_percentage_l1713_171396

/-- Calculate the overall profit percentage for four items --/
theorem overall_profit_percentage
  (sp_a : ℝ) (cp_percent_a : ℝ)
  (sp_b : ℝ) (cp_percent_b : ℝ)
  (sp_c : ℝ) (cp_percent_c : ℝ)
  (sp_d : ℝ) (cp_percent_d : ℝ)
  (h_sp_a : sp_a = 120)
  (h_cp_percent_a : cp_percent_a = 30)
  (h_sp_b : sp_b = 200)
  (h_cp_percent_b : cp_percent_b = 20)
  (h_sp_c : sp_c = 75)
  (h_cp_percent_c : cp_percent_c = 40)
  (h_sp_d : sp_d = 180)
  (h_cp_percent_d : cp_percent_d = 25) :
  let cp_a := sp_a * (cp_percent_a / 100)
  let cp_b := sp_b * (cp_percent_b / 100)
  let cp_c := sp_c * (cp_percent_c / 100)
  let cp_d := sp_d * (cp_percent_d / 100)
  let total_cp := cp_a + cp_b + cp_c + cp_d
  let total_sp := sp_a + sp_b + sp_c + sp_d
  let total_profit := total_sp - total_cp
  let profit_percentage := (total_profit / total_cp) * 100
  abs (profit_percentage - 280.79) < 0.01 := by
sorry

end overall_profit_percentage_l1713_171396


namespace reading_difference_l1713_171346

-- Define the reading rates in pages per hour
def dustin_rate : ℚ := 75
def sam_rate : ℚ := 24

-- Define the time in hours (40 minutes = 2/3 hour)
def reading_time : ℚ := 2/3

-- Define the function to calculate pages read given rate and time
def pages_read (rate : ℚ) (time : ℚ) : ℚ := rate * time

-- Theorem statement
theorem reading_difference :
  pages_read dustin_rate reading_time - pages_read sam_rate reading_time = 34 := by
  sorry

end reading_difference_l1713_171346


namespace median_mean_equality_l1713_171319

theorem median_mean_equality (n : ℝ) : 
  let s := {n, n + 2, n + 7, n + 10, n + 16}
  n + 7 = 10 → (Finset.sum s id) / 5 = 10 := by
sorry

end median_mean_equality_l1713_171319


namespace divisible_by_six_l1713_171308

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (n : ℤ)^3 + 5*n = 6*k := by
  sorry

end divisible_by_six_l1713_171308


namespace discount_calculation_l1713_171380

/-- The original price of a shirt before discount -/
def original_price : ℚ := 746.68

/-- The discounted price of the shirt -/
def discounted_price : ℚ := 560

/-- The discount rate applied to the shirt -/
def discount_rate : ℚ := 0.25

/-- Theorem stating that the discounted price is equal to the original price minus the discount -/
theorem discount_calculation (original : ℚ) (discount : ℚ) (discounted : ℚ) :
  original = discounted_price ∧ discount = discount_rate ∧ discounted = original * (1 - discount) →
  original = original_price :=
by sorry

end discount_calculation_l1713_171380


namespace multiply_72519_by_9999_l1713_171304

theorem multiply_72519_by_9999 : 72519 * 9999 = 725117481 := by
  sorry

end multiply_72519_by_9999_l1713_171304


namespace production_days_l1713_171376

theorem production_days (n : ℕ) 
  (h1 : (n * 40 + 90) / (n + 1) = 45) : n = 9 := by
  sorry

end production_days_l1713_171376


namespace intersection_point_l1713_171315

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 5 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem intersection_point : 
  ∃! (x y : ℝ), C1 x y ∧ C2 x y ∧ x = 2 ∧ y = 1 := by sorry

end intersection_point_l1713_171315


namespace specific_pyramid_properties_l1713_171313

/-- Represents a straight pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  height : ℝ
  side_face_area : ℝ

/-- Calculates the base edge length of the pyramid -/
def base_edge_length (p : EquilateralPyramid) : ℝ := sorry

/-- Calculates the volume of the pyramid -/
def volume (p : EquilateralPyramid) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid -/
theorem specific_pyramid_properties :
  let p : EquilateralPyramid := { height := 11, side_face_area := 210 }
  base_edge_length p = 30 ∧ volume p = 825 * Real.sqrt 3 := by sorry

end specific_pyramid_properties_l1713_171313


namespace largest_divisible_by_8_l1713_171379

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

def number_format (a : ℕ) : ℕ := 365000 + a * 100 + 20

theorem largest_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    is_divisible_by_8 (number_format 9) ∧
    (is_divisible_by_8 (number_format a) → a ≤ 9) :=
by sorry

end largest_divisible_by_8_l1713_171379


namespace abs_neg_one_third_l1713_171333

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end abs_neg_one_third_l1713_171333


namespace x_squared_minus_y_squared_l1713_171351

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end x_squared_minus_y_squared_l1713_171351


namespace min_value_problem_l1713_171389

theorem min_value_problem (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hab : a * b = 1/4) :
  (∀ x y : ℝ, 0 < x ∧ x < 1 → 0 < y ∧ y < 1 → x * y = 1/4 →
    1 / (1 - x) + 2 / (1 - y) ≥ 4 + 4 * Real.sqrt 2 / 3) ∧
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 ∧
    1 / (1 - x) + 2 / (1 - y) = 4 + 4 * Real.sqrt 2 / 3) :=
by sorry

end min_value_problem_l1713_171389


namespace henrys_money_l1713_171317

theorem henrys_money (initial_amount : ℕ) (birthday_gift : ℕ) (game_cost : ℕ) : 
  initial_amount = 11 → birthday_gift = 18 → game_cost = 10 →
  initial_amount + birthday_gift - game_cost = 19 := by
  sorry

end henrys_money_l1713_171317


namespace quadratic_no_real_roots_l1713_171331

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 + 2*x + 4 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l1713_171331


namespace intersection_of_A_and_B_l1713_171340

def set_A : Set ℝ := {x | x + 2 = 0}
def set_B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {-2} := by sorry

end intersection_of_A_and_B_l1713_171340


namespace min_value_of_function_l1713_171374

theorem min_value_of_function (m n : ℝ) : 
  m > 0 → n > 0 →  -- point in first quadrant
  ∃ (a b : ℝ), (m + a) / 2 + (n + b) / 2 - 2 = 0 →  -- symmetry condition
  2 * a + b + 3 = 0 →  -- (a,b) lies on 2x+y+3=0
  (n - b) / (m - a) = 1 →  -- slope of line of symmetry
  2 * m + n + 3 = 0 →  -- (m,n) lies on 2x+y+3=0
  (1 / m + 8 / n) ≥ 25 / 9 :=
by sorry

end min_value_of_function_l1713_171374


namespace bowling_ball_weight_proof_l1713_171332

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 36

theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 4 * canoe_weight) ∧
  (3 * canoe_weight = 108) →
  bowling_ball_weight = 18 := by
  sorry

end bowling_ball_weight_proof_l1713_171332


namespace daves_ice_cubes_l1713_171360

theorem daves_ice_cubes (original : ℕ) (made : ℕ) (total : ℕ) : 
  made = 7 → total = 9 → original + made = total → original = 2 := by
sorry

end daves_ice_cubes_l1713_171360


namespace remainder_problem_l1713_171326

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 2) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 3 := by
sorry

end remainder_problem_l1713_171326


namespace mark_soup_cans_l1713_171300

/-- The number of cans of soup Mark bought -/
def soup_cans : ℕ := sorry

/-- The cost of one can of soup -/
def soup_cost : ℕ := 2

/-- The number of loaves of bread Mark bought -/
def bread_loaves : ℕ := 2

/-- The cost of one loaf of bread -/
def bread_cost : ℕ := 5

/-- The number of boxes of cereal Mark bought -/
def cereal_boxes : ℕ := 2

/-- The cost of one box of cereal -/
def cereal_cost : ℕ := 3

/-- The number of gallons of milk Mark bought -/
def milk_gallons : ℕ := 2

/-- The cost of one gallon of milk -/
def milk_cost : ℕ := 4

/-- The number of $10 bills Mark used to pay -/
def ten_dollar_bills : ℕ := 4

theorem mark_soup_cans : soup_cans = 8 := by sorry

end mark_soup_cans_l1713_171300


namespace sqrt_expression_equals_512_l1713_171387

theorem sqrt_expression_equals_512 : 
  Real.sqrt ((16^12 + 2^24) / (16^5 + 2^30)) = 512 := by
  sorry

end sqrt_expression_equals_512_l1713_171387


namespace school_ratio_problem_l1713_171363

theorem school_ratio_problem (S T : ℕ) : 
  S / T = 50 →
  (S + 50) / (T + 5) = 25 →
  T = 3 := by
sorry

end school_ratio_problem_l1713_171363


namespace division_remainder_proof_l1713_171330

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 689 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
  sorry

end division_remainder_proof_l1713_171330


namespace range_of_a_l1713_171302

-- Define the sets A, B, and C
def A : Set ℝ := {x | (2 - x) / (3 + x) ≥ 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a*(a + 1) < 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : C a ⊆ (A ∩ B) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end range_of_a_l1713_171302


namespace sin_tan_product_l1713_171395

/-- Given an angle α whose terminal side intersects the unit circle at point P(-1/2, y),
    prove that sinα•tanα = -3/2 -/
theorem sin_tan_product (α : Real) (y : Real) 
    (h1 : Real.cos α = -1/2)  -- x-coordinate of P is -1/2
    (h2 : Real.sin α = y)     -- y-coordinate of P is y
    (h3 : (-1/2)^2 + y^2 = 1) -- P is on the unit circle
    : Real.sin α * Real.tan α = -3/2 := by
  sorry

end sin_tan_product_l1713_171395


namespace min_value_of_quadratic_function_range_l1713_171365

/-- Given a quadratic function f(x) = ax² + 2x + c with range [2,+∞), 
    prove that the minimum value of 1/a + 9/c is 4 -/
theorem min_value_of_quadratic_function_range (a c : ℝ) : 
  a > 0 → 
  (∀ x : ℝ, ax^2 + 2*x + c ≥ 2) → 
  (∃ x : ℝ, ax^2 + 2*x + c = 2) → 
  (∀ y : ℝ, 1/a + 9/c ≥ y) → 
  (∃ y : ℝ, 1/a + 9/c = y ∧ y = 4) :=
by sorry

end min_value_of_quadratic_function_range_l1713_171365


namespace new_cube_edge_theorem_l1713_171338

/-- The edge length of a cube formed by melting five cubes -/
def new_cube_edge (a b c d e : ℝ) : ℝ :=
  (a^3 + b^3 + c^3 + d^3 + e^3) ^ (1/3)

/-- Theorem stating that the edge of the new cube is the cube root of the sum of volumes -/
theorem new_cube_edge_theorem :
  new_cube_edge 6 8 10 12 14 = (6^3 + 8^3 + 10^3 + 12^3 + 14^3) ^ (1/3) :=
by sorry

end new_cube_edge_theorem_l1713_171338


namespace cubic_equation_root_l1713_171329

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5)^3 + a * (3 + Real.sqrt 5)^2 + b * (3 + Real.sqrt 5) - 40 = 0 → b = 64 := by
  sorry

end cubic_equation_root_l1713_171329


namespace power_multiplication_equality_l1713_171356

theorem power_multiplication_equality : (512 : ℝ)^(2/3) * 8 = 512 := by
  sorry

end power_multiplication_equality_l1713_171356


namespace find_number_l1713_171345

theorem find_number (x : ℝ) : 5 + 2 * (x - 3) = 15 → x = 8 := by
  sorry

end find_number_l1713_171345


namespace quadratic_radical_equality_l1713_171320

theorem quadratic_radical_equality (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ a + 2 = k * (3 * a)) → a = 1 := by
  sorry

end quadratic_radical_equality_l1713_171320


namespace ratio_w_to_y_l1713_171367

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
sorry

end ratio_w_to_y_l1713_171367


namespace max_absolute_value_z_l1713_171352

theorem max_absolute_value_z (z : ℂ) (h : Complex.abs (z + 3 + 4 * I) ≤ 2) :
  ∃ (M : ℝ), M = 7 ∧ Complex.abs z ≤ M ∧ ∀ (N : ℝ), Complex.abs z ≤ N → M ≤ N :=
sorry

end max_absolute_value_z_l1713_171352


namespace smithtown_population_ratio_l1713_171392

/-- Represents the population of Smithtown -/
structure Population where
  total : ℝ
  rightHanded : ℝ
  leftHanded : ℝ
  men : ℝ
  women : ℝ
  leftHandedWomen : ℝ

/-- The conditions given in the problem -/
def populationConditions (p : Population) : Prop :=
  p.rightHanded / p.leftHanded = 3 ∧
  p.leftHandedWomen / p.total = 0.2500000000000001 ∧
  p.rightHanded = p.men

/-- The theorem to be proved -/
theorem smithtown_population_ratio
  (p : Population)
  (h : populationConditions p) :
  p.men / p.women = 3 := by
  sorry

end smithtown_population_ratio_l1713_171392


namespace johns_game_percentage_l1713_171354

theorem johns_game_percentage (shots_per_foul : ℕ) (fouls_per_game : ℕ) (total_games : ℕ) (actual_shots : ℕ) :
  shots_per_foul = 2 →
  fouls_per_game = 5 →
  total_games = 20 →
  actual_shots = 112 →
  (actual_shots : ℚ) / ((shots_per_foul * fouls_per_game * total_games) : ℚ) * 100 = 56 := by
  sorry

end johns_game_percentage_l1713_171354


namespace ellipse_foci_distance_l1713_171334

/-- The distance between the foci of an ellipse with center (3, 2), semi-major axis 7, and semi-minor axis 3 is 4√10. -/
theorem ellipse_foci_distance :
  ∀ (center : ℝ × ℝ) (semi_major semi_minor : ℝ),
    center = (3, 2) →
    semi_major = 7 →
    semi_minor = 3 →
    let c := Real.sqrt (semi_major^2 - semi_minor^2)
    2 * c = 4 * Real.sqrt 10 := by
  sorry

end ellipse_foci_distance_l1713_171334


namespace cube_volume_from_surface_area_l1713_171385

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 96 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 64 := by
  sorry

end cube_volume_from_surface_area_l1713_171385


namespace gross_profit_percentage_l1713_171314

/-- Given a sales price and gross profit, calculate the percentage of gross profit relative to the cost -/
theorem gross_profit_percentage 
  (sales_price : ℝ) 
  (gross_profit : ℝ) 
  (h1 : sales_price = 81)
  (h2 : gross_profit = 51) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 170 := by
sorry


end gross_profit_percentage_l1713_171314


namespace pythagorean_theorem_3d_l1713_171393

/-- The Pythagorean theorem extended to a rectangular solid -/
theorem pythagorean_theorem_3d (p q r d : ℝ) 
  (h : d > 0) 
  (h_diagonal : d = Real.sqrt (p^2 + q^2 + r^2)) : 
  p^2 + q^2 + r^2 = d^2 := by
sorry

end pythagorean_theorem_3d_l1713_171393


namespace compound_oxygen_atoms_l1713_171399

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

def molecularWeight (comp : CompoundComposition) (weights : AtomicWeights) : ℝ :=
  comp.carbon * weights.carbon + comp.hydrogen * weights.hydrogen + comp.oxygen * weights.oxygen

theorem compound_oxygen_atoms 
  (comp : CompoundComposition)
  (weights : AtomicWeights)
  (h1 : comp.carbon = 4)
  (h2 : comp.hydrogen = 8)
  (h3 : weights.carbon = 12.01)
  (h4 : weights.hydrogen = 1.008)
  (h5 : weights.oxygen = 16.00)
  (h6 : molecularWeight comp weights = 88) :
  comp.oxygen = 2 := by
sorry

end compound_oxygen_atoms_l1713_171399


namespace shortest_distance_on_specific_cone_l1713_171342

/-- Represents a right circular cone -/
structure RightCircularCone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone -/
noncomputable def shortestDistanceOnCone (cone : RightCircularCone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let cone : RightCircularCone := { baseRadius := 600, height := 200 * Real.sqrt 7 }
  let p1 : ConePoint := { distanceFromVertex := 125 }
  let p2 : ConePoint := { distanceFromVertex := 375 * Real.sqrt 2 }
  shortestDistanceOnCone cone p1 p2 = 125 * Real.sqrt 19 :=
by sorry

end shortest_distance_on_specific_cone_l1713_171342


namespace survey_respondents_l1713_171390

theorem survey_respondents : 
  ∀ (x y : ℕ), 
    x = 60 → -- Number of people who prefer brand X
    x = 3 * y → -- Ratio of preference for X to Y is 3:1
    x + y = 80 -- Total number of respondents
    := by sorry

end survey_respondents_l1713_171390


namespace equation_solution_l1713_171362

theorem equation_solution : ∃! x : ℝ, (1/4 : ℝ)^(4*x + 12) = 16^(x + 3) := by
  sorry

end equation_solution_l1713_171362


namespace set_existence_condition_l1713_171357

theorem set_existence_condition (r : ℝ) (hr : 0 < r ∧ r < 1) :
  (∃ S : Set ℝ, 
    (∀ t : ℝ, (t ∈ S ∨ (t + r) ∈ S ∨ (t + 1) ∈ S) ∧
              (t ∉ S ∨ (t + r) ∉ S) ∧ ((t + r) ∉ S ∨ (t + 1) ∉ S) ∧ (t ∉ S ∨ (t + 1) ∉ S)) ∧
    (∀ t : ℝ, (t ∈ S ∨ (t - r) ∈ S ∨ (t - 1) ∈ S) ∧
              (t ∉ S ∨ (t - r) ∉ S) ∧ ((t - r) ∉ S ∨ (t - 1) ∉ S) ∧ (t ∉ S ∨ (t - 1) ∉ S))) ↔
  (¬ ∃ (a b : ℤ), r = (a : ℝ) / (b : ℝ)) ∨
  (∃ (a b : ℤ), r = (a : ℝ) / (b : ℝ) ∧ 3 ∣ (a + b)) :=
by sorry

end set_existence_condition_l1713_171357


namespace quadratic_inequality_solution_set_l1713_171384

theorem quadratic_inequality_solution_set (c : ℝ) (hc : c > 1) :
  {x : ℝ | x^2 - (c + 1/c)*x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} := by
  sorry

end quadratic_inequality_solution_set_l1713_171384


namespace problem_1_problem_2_problem_3_problem_4_l1713_171323

-- Problem 1
theorem problem_1 : 8 * 77 * 125 = 77000 := by sorry

-- Problem 2
theorem problem_2 : 12 * 98 = 1176 := by sorry

-- Problem 3
theorem problem_3 : 6 * 321 + 6 * 179 = 3000 := by sorry

-- Problem 4
theorem problem_4 : 56 * 101 - 56 = 5600 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1713_171323


namespace quadratic_sum_l1713_171371

theorem quadratic_sum (x : ℝ) : ∃ (a h k : ℝ),
  (3 * x^2 - 6 * x - 2 = a * (x - h)^2 + k) ∧ (a + h + k = -1) := by
  sorry

end quadratic_sum_l1713_171371


namespace chord_inscribed_squares_side_difference_l1713_171378

/-- Given a circle with radius r and a chord at distance h from the center,
    prove that the difference in side lengths of two squares inscribed in the segments
    formed by the chord is 8h/5. -/
theorem chord_inscribed_squares_side_difference
  (r h : ℝ) (hr : r > 0) (hh : 0 < h ∧ h < r) :
  ∃ (a b : ℝ),
    (a > 0 ∧ b > 0) ∧
    (a - h)^2 = r^2 - (a^2 / 4) ∧
    (b + h)^2 = r^2 - (b^2 / 4) ∧
    b - a = (8 * h) / 5 :=
sorry

end chord_inscribed_squares_side_difference_l1713_171378


namespace box_length_l1713_171337

/-- Given a box with specified dimensions and cube properties, prove its length --/
theorem box_length (width : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) :
  width = 16 →
  height = 6 →
  cube_volume = 3 →
  min_cubes = 384 →
  (min_cubes : ℝ) * cube_volume / (width * height) = 12 :=
by
  sorry


end box_length_l1713_171337
