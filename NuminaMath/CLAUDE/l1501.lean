import Mathlib

namespace complex_magnitude_problem_l1501_150163

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I * Real.sqrt 3) = 1) :
  Complex.abs z = 1 / 2 := by
  sorry

end complex_magnitude_problem_l1501_150163


namespace function_properties_l1501_150127

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_symmetry : ∀ x, f (4 - x) = f x) : 
  (∀ x, f (x + 8) = f x) ∧ (f 2019 + f 2020 + f 2021 = 0) := by
  sorry

end function_properties_l1501_150127


namespace two_numbers_property_l1501_150136

theorem two_numbers_property : ∃ x y : ℕ, 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 38) id) - x - y = x * y + 1 ∧
  y - x = 20 := by
sorry

end two_numbers_property_l1501_150136


namespace fibonacci_problem_l1501_150164

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property of arithmetic sequence for Fibonacci numbers
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  fib c - fib b = fib b - fib a

-- Define the main theorem
theorem fibonacci_problem (b : ℕ) :
  is_arithmetic_seq (b - 3) b (b + 3) →
  (b - 3) + b + (b + 3) = 2253 →
  b = 751 := by
  sorry


end fibonacci_problem_l1501_150164


namespace circle_area_half_radius_l1501_150118

/-- The area of a circle with radius 1/2 is π/4 -/
theorem circle_area_half_radius : 
  let r : ℚ := 1/2
  π * r^2 = π/4 := by sorry

end circle_area_half_radius_l1501_150118


namespace clock_twelve_strikes_l1501_150162

/-- Represents a grandfather clock with a given strike interval -/
structure GrandfatherClock where
  strike_interval : ℝ

/-- Calculates the time taken for a given number of strikes -/
def time_for_strikes (clock : GrandfatherClock) (num_strikes : ℕ) : ℝ :=
  clock.strike_interval * (num_strikes - 1)

theorem clock_twelve_strikes (clock : GrandfatherClock) 
  (h : time_for_strikes clock 6 = 30) :
  time_for_strikes clock 12 = 66 := by
  sorry


end clock_twelve_strikes_l1501_150162


namespace babysitting_earnings_l1501_150114

def final_balance (hourly_rate : ℕ) (hours_worked : ℕ) (initial_balance : ℕ) : ℕ :=
  initial_balance + hourly_rate * hours_worked

theorem babysitting_earnings : final_balance 5 7 20 = 55 := by
  sorry

end babysitting_earnings_l1501_150114


namespace rectangle_dimension_change_l1501_150113

theorem rectangle_dimension_change (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let L' := L * (1 + 30 / 100)
  let B' := B * (1 - 20 / 100)
  L' * B' = (L * B) * (1 + 4.0000000000000036 / 100) :=
by sorry

end rectangle_dimension_change_l1501_150113


namespace janice_purchase_l1501_150145

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 50 →
  30 * a + 200 * b + 300 * c = 5000 →
  a = 10 :=
by sorry

end janice_purchase_l1501_150145


namespace sufficient_not_necessary_condition_l1501_150172

def f (a : ℝ) (x : ℝ) : ℝ := (x - a)^2

theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a = 1 → (∀ x y : ℝ, 1 < x → x < y → f a x < f a y)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ (∀ x y : ℝ, 1 < x → x < y → f a x < f a y)) := by
  sorry

end sufficient_not_necessary_condition_l1501_150172


namespace quadratic_factorization_l1501_150111

theorem quadratic_factorization :
  ∀ x : ℝ, 2 * x^2 - 10 * x - 12 = 2 * (x - 6) * (x + 1) := by
  sorry

end quadratic_factorization_l1501_150111


namespace no_prime_solution_l1501_150133

theorem no_prime_solution : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end no_prime_solution_l1501_150133


namespace cheapest_combination_is_12_apples_3_oranges_l1501_150188

/-- Represents the price of a fruit deal -/
structure FruitDeal where
  quantity : Nat
  price : Rat

/-- Represents the fruit options available -/
structure FruitOptions where
  apple_deals : List FruitDeal
  orange_deals : List FruitDeal

/-- Represents a combination of apples and oranges -/
structure FruitCombination where
  apples : Nat
  oranges : Nat

def total_fruits (combo : FruitCombination) : Nat :=
  combo.apples + combo.oranges

def is_valid_combination (combo : FruitCombination) : Prop :=
  total_fruits combo = 15 ∧
  (combo.apples % 2 = 0 ∨ combo.apples % 3 = 0) ∧
  (combo.oranges % 2 = 0 ∨ combo.oranges % 3 = 0)

def cost_of_combination (options : FruitOptions) (combo : FruitCombination) : Rat :=
  sorry

def cheapest_combination (options : FruitOptions) : FruitCombination :=
  sorry

theorem cheapest_combination_is_12_apples_3_oranges
  (options : FruitOptions)
  (h_apple_deals : options.apple_deals = [
    ⟨2, 48/100⟩, ⟨6, 126/100⟩, ⟨12, 224/100⟩
  ])
  (h_orange_deals : options.orange_deals = [
    ⟨2, 60/100⟩, ⟨6, 164/100⟩, ⟨12, 300/100⟩
  ]) :
  cheapest_combination options = ⟨12, 3⟩ ∧
  cost_of_combination options (cheapest_combination options) = 314/100 :=
sorry

end cheapest_combination_is_12_apples_3_oranges_l1501_150188


namespace largest_n_satisfying_inequality_l1501_150151

theorem largest_n_satisfying_inequality : 
  ∀ n : ℤ, (1 : ℚ) / 3 + (n : ℚ) / 7 < 1 ↔ n ≤ 4 :=
by sorry

end largest_n_satisfying_inequality_l1501_150151


namespace original_prices_theorem_l1501_150179

def shirt_discount : Float := 0.20
def shoes_discount : Float := 0.30
def jacket_discount : Float := 0.10

def discounted_shirt_price : Float := 780
def discounted_shoes_price : Float := 2100
def discounted_jacket_price : Float := 2700

def original_shirt_price : Float := discounted_shirt_price / (1 - shirt_discount)
def original_shoes_price : Float := discounted_shoes_price / (1 - shoes_discount)
def original_jacket_price : Float := discounted_jacket_price / (1 - jacket_discount)

theorem original_prices_theorem :
  original_shirt_price = 975 ∧
  original_shoes_price = 3000 ∧
  original_jacket_price = 3000 :=
by sorry

end original_prices_theorem_l1501_150179


namespace trapezoid_area_is_787_5_l1501_150112

/-- Represents a trapezoid ABCD with given measurements -/
structure Trapezoid where
  ab : ℝ
  bc : ℝ
  ad : ℝ
  altitude : ℝ
  slant_height : ℝ

/-- Calculates the area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the given trapezoid is 787.5 -/
theorem trapezoid_area_is_787_5 (t : Trapezoid) 
  (h_ab : t.ab = 40)
  (h_bc : t.bc = 30)
  (h_ad : t.ad = 17)
  (h_altitude : t.altitude = 15)
  (h_slant_height : t.slant_height = 34) :
  trapezoid_area t = 787.5 := by
  sorry

end trapezoid_area_is_787_5_l1501_150112


namespace marble_arrangement_l1501_150126

def arrange_marbles (n : ℕ) (restricted_pairs : ℕ) : ℕ :=
  n.factorial - restricted_pairs * (n - 1).factorial

theorem marble_arrangement :
  arrange_marbles 5 1 = 72 := by sorry

end marble_arrangement_l1501_150126


namespace integer_partition_impossibility_l1501_150184

theorem integer_partition_impossibility : 
  ¬ (∃ (A B C : Set Int), 
    (∀ (n : Int), n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ 
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
    (∀ (n : Int), 
      ((n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
       (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
       (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
       (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
       (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
       (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)))) :=
by sorry

end integer_partition_impossibility_l1501_150184


namespace largest_last_digit_l1501_150129

/-- A string of digits satisfying the given conditions -/
structure DigitString where
  digits : Fin 2050 → Nat
  first_digit_is_two : digits 0 = 2
  divisibility_condition : ∀ i : Fin 2049, 
    (digits i * 10 + digits (i + 1)) % 17 = 0 ∨ 
    (digits i * 10 + digits (i + 1)) % 29 = 0

/-- The theorem stating that the largest possible last digit is 8 -/
theorem largest_last_digit (s : DigitString) : 
  s.digits 2049 ≤ 8 ∧ ∃ s : DigitString, s.digits 2049 = 8 := by
  sorry


end largest_last_digit_l1501_150129


namespace sum_of_first_ten_terms_l1501_150161

/-- Given a sequence {a_n} and its partial sum sequence {S_n}, prove that S_10 = 145 -/
theorem sum_of_first_ten_terms 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S (n + 1) = S n + a n + 3)
  (h2 : a 5 + a 6 = 29) :
  S 10 = 145 := by
sorry

end sum_of_first_ten_terms_l1501_150161


namespace quadratic_root_property_l1501_150124

theorem quadratic_root_property (a : ℝ) : 
  a^2 + 3*a - 1010 = 0 → 2*a^2 + 6*a + 4 = 2024 := by
  sorry

end quadratic_root_property_l1501_150124


namespace johnny_work_days_l1501_150171

def daily_earnings : ℝ := 3 * 7 + 2 * 10 + 4 * 12

theorem johnny_work_days (x : ℝ) (h : x * daily_earnings = 445) : x = 5 := by
  sorry

end johnny_work_days_l1501_150171


namespace cubic_equation_root_l1501_150138

theorem cubic_equation_root (a b : ℚ) : 
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 20 = 0 → 
  b = -79 := by
sorry

end cubic_equation_root_l1501_150138


namespace article_cost_price_l1501_150150

def cost_price : ℝ → Prop :=
  λ c => 
    ∃ s, 
      (s = 1.25 * c) ∧ 
      (s - 14.70 = 1.04 * c) ∧ 
      (c = 70)

theorem article_cost_price : 
  ∃ c, cost_price c :=
sorry

end article_cost_price_l1501_150150


namespace tenth_term_is_19_l1501_150143

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- The nth term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first 9 terms is 81 -/
  sum_9 : S 9 = 81
  /-- The second term is 3 -/
  second_term : a 2 = 3
  /-- The sequence follows the arithmetic sequence property -/
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The 10th term of the specified arithmetic sequence is 19 -/
theorem tenth_term_is_19 (seq : ArithmeticSequence) : seq.a 10 = 19 := by
  sorry

end tenth_term_is_19_l1501_150143


namespace classes_per_semester_l1501_150199

/-- Given the following conditions:
  1. Maddy is in college for 8 semesters.
  2. She needs 120 credits to graduate.
  3. Each class is 3 credits.
  Prove that Maddy needs to take 5 classes per semester. -/
theorem classes_per_semester :
  let total_semesters : ℕ := 8
  let total_credits : ℕ := 120
  let credits_per_class : ℕ := 3
  let classes_per_semester : ℕ := total_credits / (credits_per_class * total_semesters)
  classes_per_semester = 5 := by sorry

end classes_per_semester_l1501_150199


namespace smallest_sum_with_same_probability_l1501_150100

/-- Represents a symmetrical die with faces numbered 1 to 6 -/
structure SymmetricalDie :=
  (faces : Fin 6)

/-- Represents a set of symmetrical dice -/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum when throwing the dice -/
def probability (d : DiceSet) (sum : Nat) : ℝ :=
  sorry

/-- The condition that the sum 2022 is possible with a positive probability -/
def sum_2022_possible (d : DiceSet) : Prop :=
  ∃ p : ℝ, p > 0 ∧ probability d 2022 = p

/-- The theorem stating the smallest possible sum with the same probability as 2022 -/
theorem smallest_sum_with_same_probability (d : DiceSet) 
  (h : sum_2022_possible d) : 
  ∃ p : ℝ, p > 0 ∧ 
    probability d 2022 = p ∧
    probability d 337 = p ∧
    ∀ (sum : Nat), sum < 337 → probability d sum < p :=
  sorry

end smallest_sum_with_same_probability_l1501_150100


namespace final_deficit_is_twelve_l1501_150181

/-- Calculates the final score difference for Liz's basketball game --/
def final_score_difference (initial_deficit : ℕ) 
  (liz_free_throws liz_threes liz_jumps liz_and_ones : ℕ)
  (taylor_threes taylor_jumps : ℕ)
  (opp1_threes : ℕ)
  (opp2_jumps opp2_free_throws : ℕ)
  (opp3_jumps opp3_threes : ℕ) : ℤ :=
  let liz_score := liz_free_throws + 3 * liz_threes + 2 * liz_jumps + 3 * liz_and_ones
  let taylor_score := 3 * taylor_threes + 2 * taylor_jumps
  let opp1_score := 3 * opp1_threes
  let opp2_score := 2 * opp2_jumps + opp2_free_throws
  let opp3_score := 2 * opp3_jumps + 3 * opp3_threes
  let team_score_diff := (liz_score + taylor_score) - (opp1_score + opp2_score + opp3_score)
  initial_deficit - team_score_diff

theorem final_deficit_is_twelve :
  final_score_difference 25 5 4 5 1 2 3 4 4 2 2 1 = 12 := by
  sorry

end final_deficit_is_twelve_l1501_150181


namespace min_value_of_function_l1501_150168

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (12 / x + 4 * x) ≥ 8 * Real.sqrt 3 ∧ ∃ y > 0, 12 / y + 4 * y = 8 * Real.sqrt 3 :=
by sorry

end min_value_of_function_l1501_150168


namespace fundraiser_goal_reached_l1501_150178

/-- Proves that the total amount raised by the group is equal to the total amount needed for the trip. -/
theorem fundraiser_goal_reached (
  num_students : ℕ) 
  (individual_cost : ℕ)
  (collective_expenses : ℕ)
  (day1_raised : ℕ)
  (day2_raised : ℕ)
  (day3_raised : ℕ)
  (num_half_days : ℕ)
  (h1 : num_students = 6)
  (h2 : individual_cost = 450)
  (h3 : collective_expenses = 3000)
  (h4 : day1_raised = 600)
  (h5 : day2_raised = 900)
  (h6 : day3_raised = 400)
  (h7 : num_half_days = 4) :
  (num_students * individual_cost + collective_expenses) = 
  (day1_raised + day2_raised + day3_raised + 
   num_half_days * ((day1_raised + day2_raised + day3_raised) / 2)) := by
  sorry

#eval 6 * 450 + 3000 -- Total needed
#eval 600 + 900 + 400 + 4 * ((600 + 900 + 400) / 2) -- Total raised

end fundraiser_goal_reached_l1501_150178


namespace sqrt_a_minus_two_real_l1501_150187

theorem sqrt_a_minus_two_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) → a ≥ 2 := by
  sorry

end sqrt_a_minus_two_real_l1501_150187


namespace student_average_less_than_true_average_l1501_150131

theorem student_average_less_than_true_average 
  (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2*w + 2*x + y + z) / 6 < (w + x + y + z) / 4 := by
sorry

end student_average_less_than_true_average_l1501_150131


namespace smallest_towel_sets_l1501_150141

def hand_towels_per_set : ℕ := 23
def bath_towels_per_set : ℕ := 29

def total_towels (sets : ℕ) : ℕ :=
  sets * hand_towels_per_set + sets * bath_towels_per_set

theorem smallest_towel_sets :
  ∃ (sets : ℕ),
    (500 ≤ total_towels sets) ∧
    (total_towels sets ≤ 700) ∧
    (∀ (other_sets : ℕ),
      (500 ≤ total_towels other_sets) ∧
      (total_towels other_sets ≤ 700) →
      sets ≤ other_sets) ∧
    sets * hand_towels_per_set = 230 ∧
    sets * bath_towels_per_set = 290 :=
by sorry

end smallest_towel_sets_l1501_150141


namespace probability_of_purple_l1501_150142

def die_sides : ℕ := 6
def red_sides : ℕ := 3
def yellow_sides : ℕ := 2
def blue_sides : ℕ := 1

def prob_red : ℚ := red_sides / die_sides
def prob_blue : ℚ := blue_sides / die_sides

theorem probability_of_purple (h1 : die_sides = red_sides + yellow_sides + blue_sides)
  (h2 : prob_red = red_sides / die_sides)
  (h3 : prob_blue = blue_sides / die_sides) :
  prob_red * prob_blue + prob_blue * prob_red = 1 / 6 :=
sorry

end probability_of_purple_l1501_150142


namespace system_solution_l1501_150105

theorem system_solution :
  let x : ℝ := -13
  let y : ℝ := -1
  let z : ℝ := 2
  (x + y + 16 * z = 18) ∧
  (x - 3 * y + 8 * z = 6) ∧
  (2 * x - y - 4 * z = -33) := by
  sorry

end system_solution_l1501_150105


namespace arithmetic_sequence_60th_term_l1501_150122

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  a₁ : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

theorem arithmetic_sequence_60th_term
  (seq : ArithmeticSequence)
  (h₁ : seq.a₁ = 6)
  (h₁₃ : seq.nthTerm 13 = 32) :
  seq.nthTerm 60 = 803 / 6 := by
  sorry

end arithmetic_sequence_60th_term_l1501_150122


namespace problem_statement_l1501_150159

theorem problem_statement (A : ℤ) (h : A = 43^2011 - 2011^43) : 
  (3 ∣ A) ∧ (A % 11 = 7) ∧ (A % 35 = 6) := by
  sorry

end problem_statement_l1501_150159


namespace vector_dot_product_collinear_l1501_150149

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_collinear :
  ∀ (k : ℝ),
  let m : ℝ × ℝ := (2 * k - 1, k)
  let n : ℝ × ℝ := (4, 1)
  collinear m n → dot_product m n = -17/2 := by
  sorry

end vector_dot_product_collinear_l1501_150149


namespace log_equation_solution_l1501_150128

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * (Real.log x / Real.log 3) = Real.log (4 * x^2) / Real.log 3 → x = 2 := by
  sorry

end log_equation_solution_l1501_150128


namespace chess_tournament_players_l1501_150121

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest 12
  /-- Total number of players is n + 12 -/
  total_players : ℕ := n + 12
  /-- Each player played exactly one game against every other player -/
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  /-- Points earned by n players not in the lowest 12 -/
  top_points : ℕ := n * (n - 1)
  /-- Points earned by 12 lowest-scoring players among themselves -/
  bottom_points : ℕ := 66
  /-- Total points earned in the tournament -/
  total_points : ℕ := total_games

/-- The theorem stating that the total number of players in the tournament is 34 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 34 := by
  sorry


end chess_tournament_players_l1501_150121


namespace increasing_digits_mod_1000_l1501_150193

/-- The number of 8-digit positive integers with digits in increasing order -/
def count_increasing_digits : ℕ := (Nat.choose 17 8)

/-- The theorem stating that the count of such integers is congruent to 310 modulo 1000 -/
theorem increasing_digits_mod_1000 :
  count_increasing_digits % 1000 = 310 := by sorry

end increasing_digits_mod_1000_l1501_150193


namespace five_people_four_rooms_l1501_150166

/-- The number of ways to assign n people to k rooms, where any number of people can be in a room -/
def room_assignments (n k : ℕ) : ℕ := sorry

/-- The specific case for 5 people and 4 rooms -/
theorem five_people_four_rooms : room_assignments 5 4 = 61 := by sorry

end five_people_four_rooms_l1501_150166


namespace square_sum_given_product_and_sum_l1501_150123

theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end square_sum_given_product_and_sum_l1501_150123


namespace plant_purchase_solution_l1501_150182

/-- Represents the prices and quantities of plants A and B -/
structure PlantPurchase where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Calculates the total cost of a plant purchase -/
def total_cost (p : PlantPurchase) : ℝ :=
  p.price_a * p.quantity_a + p.price_b * p.quantity_b

/-- Represents the given conditions from the problem -/
structure ProblemConditions where
  first_phase : PlantPurchase
  second_phase : PlantPurchase
  total_cost_both_phases : ℝ

/-- The main theorem representing the problem and its solution -/
theorem plant_purchase_solution (conditions : ProblemConditions) 
  (h1 : conditions.first_phase.quantity_a = 30)
  (h2 : conditions.first_phase.quantity_b = 15)
  (h3 : total_cost conditions.first_phase = 675)
  (h4 : conditions.second_phase.quantity_a = 12)
  (h5 : conditions.second_phase.quantity_b = 5)
  (h6 : conditions.total_cost_both_phases = 940)
  (h7 : conditions.first_phase.price_a = conditions.second_phase.price_a)
  (h8 : conditions.first_phase.price_b = conditions.second_phase.price_b) :
  ∃ (optimal_plan : PlantPurchase),
    conditions.first_phase.price_a = 20 ∧
    conditions.first_phase.price_b = 5 ∧
    optimal_plan.quantity_a + optimal_plan.quantity_b = 31 ∧
    optimal_plan.quantity_b < 2 * optimal_plan.quantity_a ∧
    total_cost optimal_plan = 320 ∧
    ∀ (other_plan : PlantPurchase),
      other_plan.quantity_a + other_plan.quantity_b = 31 →
      other_plan.quantity_b < 2 * other_plan.quantity_a →
      total_cost other_plan ≥ total_cost optimal_plan := by
  sorry

end plant_purchase_solution_l1501_150182


namespace expand_binomials_l1501_150137

theorem expand_binomials (x : ℝ) : (2*x - 3) * (x + 2) = 2*x^2 + x - 6 := by
  sorry

end expand_binomials_l1501_150137


namespace quartic_roots_l1501_150116

/-- The value of N in the quartic equation --/
def N : ℝ := 10^10

/-- The quartic function --/
def f (x : ℝ) : ℝ := x^4 - (2*N + 1)*x^2 - x + N^2 + N - 1

/-- The first approximate root --/
def root1 : ℝ := 99999.9984

/-- The second approximate root --/
def root2 : ℝ := 100000.0016

/-- Theorem stating that the quartic equation has two approximate roots --/
theorem quartic_roots : 
  ∃ (r1 r2 : ℝ), 
    (abs (r1 - root1) < 0.00005) ∧ 
    (abs (r2 - root2) < 0.00005) ∧ 
    f r1 = 0 ∧ 
    f r2 = 0 := by
  sorry

end quartic_roots_l1501_150116


namespace sum_of_distances_l1501_150175

/-- Given two line segments AB and A'B', with points D and D' on them respectively,
    and a point P on AB, prove that the sum of PD and P'D' is 10/3 units. -/
theorem sum_of_distances (AB A'B' AD A'D' PD : ℝ) (h1 : AB = 8)
    (h2 : A'B' = 6) (h3 : AD = 3) (h4 : A'D' = 1) (h5 : PD = 2)
    (h6 : PD / P'D' = 3 / 2) : PD + P'D' = 10 / 3 := by
  sorry

end sum_of_distances_l1501_150175


namespace initial_amount_was_21_l1501_150125

/-- The initial amount of money in the cookie jar -/
def initial_amount : ℕ := sorry

/-- The amount Doris spent -/
def doris_spent : ℕ := 6

/-- The amount Martha spent -/
def martha_spent : ℕ := doris_spent / 2

/-- The amount left in the cookie jar after spending -/
def amount_left : ℕ := 12

/-- Theorem stating that the initial amount in the cookie jar was 21 dollars -/
theorem initial_amount_was_21 : initial_amount = 21 := by
  sorry

end initial_amount_was_21_l1501_150125


namespace polynomial_simplification_l1501_150169

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5*x - 4) - (x^2 - 2*x + 1) + (3 * x^2 + 4*x - 7) = 4 * x^2 + 11*x - 12 := by
  sorry

end polynomial_simplification_l1501_150169


namespace solve_ages_l1501_150140

/-- Represents the ages of people in the problem -/
structure Ages where
  rehana : ℕ
  phoebe : ℕ
  jacob : ℕ
  xander : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.rehana = 25 ∧
  ages.rehana + 5 = 3 * (ages.phoebe + 5) ∧
  ages.jacob = 3 * ages.phoebe / 5 ∧
  ages.xander = ages.rehana + ages.jacob - 4

/-- The theorem to prove -/
theorem solve_ages : 
  ∃ (ages : Ages), problem_conditions ages ∧ 
    ages.rehana = 25 ∧ 
    ages.phoebe = 5 ∧ 
    ages.jacob = 3 ∧ 
    ages.xander = 24 := by
  sorry

end solve_ages_l1501_150140


namespace coefficient_of_x3_in_expansion_l1501_150183

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^3
def coefficientOfX3 (a b : ℝ) (n : ℕ) : ℝ :=
  (-b)^1 * binomial n 1 * a^3

-- Theorem statement
theorem coefficient_of_x3_in_expansion :
  coefficientOfX3 1 3 7 = -21 := by sorry

end coefficient_of_x3_in_expansion_l1501_150183


namespace problem_statement_l1501_150117

theorem problem_statement (x y : ℝ) 
  (h1 : 2 * x - y = 1) 
  (h2 : x * y = 2) : 
  4 * x^3 * y - 4 * x^2 * y^2 + x * y^3 = 2 := by
  sorry

end problem_statement_l1501_150117


namespace range_of_mu_l1501_150103

theorem range_of_mu (a b μ : ℝ) (ha : a > 0) (hb : b > 0) (hμ : μ > 0) 
  (h : 1/a + 9/b = 1) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 9/b = 1 → a + b ≥ μ) ↔ μ ∈ Set.Ioc 0 16 :=
by sorry

end range_of_mu_l1501_150103


namespace cruise_ship_problem_l1501_150190

/-- Cruise ship problem -/
theorem cruise_ship_problem 
  (distance : ℝ) 
  (x : ℝ) 
  (k : ℝ) 
  (h1 : distance = 5)
  (h2 : 20 ≤ x ∧ x ≤ 50)
  (h3 : 1/15 ≤ k ∧ k ≤ 1/5)
  (h4 : x/40 - k = 5/8) :
  (∃ (x_range : Set ℝ), x_range = {x | 20 ≤ x ∧ x ≤ 40} ∧ 
    ∀ y ∈ x_range, y/40 - k + 1/y ≤ 9/10) ∧
  (∀ y : ℝ, 20 ≤ y ∧ y ≤ 50 →
    (1/15 ≤ k ∧ k < 1/10 → 
      5/y * (y/40 - k + 1/y) ≥ (1 - 10*k^2) / 8) ∧
    (1/10 ≤ k ∧ k ≤ 1/5 → 
      5/y * (y/40 - k + 1/y) ≥ (11 - 20*k) / 80)) :=
sorry

end cruise_ship_problem_l1501_150190


namespace arithmetic_sequence_sum_l1501_150101

theorem arithmetic_sequence_sum (k : ℕ) : 
  let a : ℕ → ℕ := λ n => 1 + 2 * (n - 1)
  let S : ℕ → ℕ := λ n => n * (2 * a 1 + (n - 1) * 2) / 2
  S (k + 2) - S k = 24 → k = 5 := by
sorry

end arithmetic_sequence_sum_l1501_150101


namespace redistribution_result_l1501_150135

/-- Represents the number of marbles each person has -/
structure MarbleCount where
  tyrone : ℚ
  eric : ℚ

/-- The initial distribution of marbles -/
def initial : MarbleCount := { tyrone := 125, eric := 25 }

/-- The number of marbles Tyrone gives to Eric -/
def marbles_given : ℚ := 12.5

/-- The final distribution of marbles after Tyrone gives some to Eric -/
def final : MarbleCount :=
  { tyrone := initial.tyrone - marbles_given,
    eric := initial.eric + marbles_given }

/-- Theorem stating that after redistribution, Tyrone has three times as many marbles as Eric -/
theorem redistribution_result :
  final.tyrone = 3 * final.eric := by sorry

end redistribution_result_l1501_150135


namespace circle_intersection_theorem_l1501_150186

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the radius function for circles
variable (radius : Circle → ℝ)

-- Define the distance function between two points
variable (dist : Point → Point → ℝ)

-- Define the "on_circle" predicate
variable (on_circle : Point → Circle → Prop)

-- Define the "intersect" predicate for two circles
variable (intersect : Circle → Circle → Point → Prop)

-- Define the "interior_point" predicate
variable (interior_point : Point → Circle → Prop)

-- Define the "line_intersect" predicate
variable (line_intersect : Point → Point → Circle → Point → Prop)

-- Define the "equilateral" predicate for triangles
variable (equilateral : Point → Point → Point → Prop)

-- Theorem statement
theorem circle_intersection_theorem 
  (k₁ k₂ : Circle) (O A B S T : Point) (r : ℝ) :
  radius k₁ = r →
  on_circle O k₁ →
  intersect k₁ k₂ A →
  intersect k₁ k₂ B →
  interior_point S k₁ →
  line_intersect B S k₁ T →
  equilateral A O S →
  dist T S = r :=
sorry

end circle_intersection_theorem_l1501_150186


namespace composite_sum_divisibility_l1501_150198

theorem composite_sum_divisibility (s : ℕ) (h1 : s ≥ 4) :
  (∃ (a b c d : ℕ+), (a + b + c + d : ℕ) = s ∧ s ∣ (a * b * c + a * b * d + a * c * d + b * c * d)) ↔
  ¬ Nat.Prime s :=
sorry

end composite_sum_divisibility_l1501_150198


namespace equation_positive_roots_l1501_150153

theorem equation_positive_roots (b : ℝ) : 
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ r₁ ≠ r₂ ∧
  (∀ x : ℝ, x > 0 → ((x - b) * (x - 2) * (x + 1) = 3 * (x - b) * (x + 1)) ↔ (x = r₁ ∨ x = r₂)) :=
sorry

end equation_positive_roots_l1501_150153


namespace hoseoks_number_l1501_150144

theorem hoseoks_number (x : ℤ) : x - 10 = 15 → x + 5 = 30 := by
  sorry

end hoseoks_number_l1501_150144


namespace cone_lateral_surface_angle_l1501_150108

theorem cone_lateral_surface_angle (r h : ℝ) (h_positive : r > 0 ∧ h > 0) :
  (π * r * (r + (r^2 + h^2).sqrt) = 3 * π * r^2) →
  (2 * π * r / (r^2 + h^2).sqrt : ℝ) = π :=
by sorry

end cone_lateral_surface_angle_l1501_150108


namespace f_decreasing_interval_l1501_150170

/-- A quadratic function with specific properties -/
def f : ℝ → ℝ := sorry

/-- The properties of the quadratic function -/
axiom f_prop1 : f (-1) = 0
axiom f_prop2 : f 4 = 0
axiom f_prop3 : f 0 = 4

/-- The decreasing interval of f -/
def decreasing_interval : Set ℝ := {x | x ≥ 3/2}

/-- Theorem stating that the given set is the decreasing interval of f -/
theorem f_decreasing_interval : 
  ∀ x ∈ decreasing_interval, ∀ y ∈ decreasing_interval, x < y → f x > f y :=
sorry

end f_decreasing_interval_l1501_150170


namespace largest_common_divisor_under_60_l1501_150180

theorem largest_common_divisor_under_60 : 
  ∃ (n : ℕ), n ∣ 456 ∧ n ∣ 108 ∧ n < 60 ∧ 
  ∀ (m : ℕ), m ∣ 456 → m ∣ 108 → m < 60 → m ≤ n :=
by
  -- The proof would go here
  sorry

end largest_common_divisor_under_60_l1501_150180


namespace min_balls_correct_l1501_150176

/-- The minimum number of balls that satisfies the given conditions -/
def min_balls : ℕ := 24

/-- The number of white balls -/
def white_balls : ℕ := min_balls / 3

/-- The number of black balls -/
def black_balls : ℕ := 2 * white_balls

/-- The number of pairs of different colors -/
def different_color_pairs : ℕ := min_balls / 4

/-- The number of pairs of the same color -/
def same_color_pairs : ℕ := 3 * different_color_pairs

theorem min_balls_correct :
  (black_balls = 2 * white_balls) ∧
  (black_balls + white_balls = min_balls) ∧
  (same_color_pairs = 3 * different_color_pairs) ∧
  (same_color_pairs + different_color_pairs = min_balls) ∧
  (∀ n : ℕ, n < min_balls → ¬(
    (2 * (n / 3) = n - (n / 3)) ∧
    (3 * (n / 4) = n - (n / 4))
  )) := by
  sorry

#eval min_balls

end min_balls_correct_l1501_150176


namespace pokemon_card_difference_l1501_150107

-- Define the initial number of cards for Sally and Dan
def sally_initial : ℕ := 27
def dan_cards : ℕ := 41

-- Define the number of cards Sally bought
def sally_bought : ℕ := 20

-- Define Sally's total cards after buying
def sally_total : ℕ := sally_initial + sally_bought

-- Theorem to prove
theorem pokemon_card_difference : sally_total - dan_cards = 6 := by
  sorry

end pokemon_card_difference_l1501_150107


namespace pencil_count_l1501_150173

theorem pencil_count (pens pencils : ℕ) 
  (h_ratio : pens * 6 = pencils * 5)
  (h_difference : pencils = pens + 4) :
  pencils = 24 := by
sorry

end pencil_count_l1501_150173


namespace average_age_calculation_l1501_150148

theorem average_age_calculation (fifth_graders : ℕ) (fifth_graders_avg : ℝ)
  (parents : ℕ) (parents_avg : ℝ) (teachers : ℕ) (teachers_avg : ℝ) :
  fifth_graders = 40 ∧ fifth_graders_avg = 10 ∧
  parents = 60 ∧ parents_avg = 35 ∧
  teachers = 10 ∧ teachers_avg = 45 →
  let total_age := fifth_graders * fifth_graders_avg + parents * parents_avg + teachers * teachers_avg
  let total_people := fifth_graders + parents + teachers
  abs ((total_age / total_people) - 26.82) < 0.01 := by
  sorry

end average_age_calculation_l1501_150148


namespace mrs_crabapple_gift_sequences_l1501_150165

/-- Represents Mrs. Crabapple's class setup -/
structure ClassSetup where
  num_students : ℕ
  meetings_per_week : ℕ
  alternating_gifts : Bool
  starts_with_crabapple : Bool

/-- Calculates the number of different gift recipient sequences for a given class setup -/
def num_gift_sequences (setup : ClassSetup) : ℕ :=
  setup.num_students ^ setup.meetings_per_week

/-- Theorem stating the number of different gift recipient sequences for Mrs. Crabapple's class -/
theorem mrs_crabapple_gift_sequences :
  let setup : ClassSetup := {
    num_students := 11,
    meetings_per_week := 4,
    alternating_gifts := true,
    starts_with_crabapple := true
  }
  num_gift_sequences setup = 14641 := by
  sorry

end mrs_crabapple_gift_sequences_l1501_150165


namespace tangent_ellipse_hyperbola_l1501_150195

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, ellipse x y ∧ hyperbola x y m

-- Theorem statement
theorem tangent_ellipse_hyperbola :
  ∀ m, are_tangent m → m = 2 := by
  sorry

end tangent_ellipse_hyperbola_l1501_150195


namespace regular_polygon_150_degrees_l1501_150185

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_150_degrees : 
  ∀ n : ℕ, 
  n > 2 → 
  (180 * (n - 2) : ℝ) = 150 * n → 
  n = 12 :=
by sorry

end regular_polygon_150_degrees_l1501_150185


namespace probability_neighboring_points_l1501_150110

/-- The probability of choosing neighboring points on a circle -/
theorem probability_neighboring_points (n : ℕ) (h : n ≥ 3) :
  (2 : ℚ) / (n - 1) = (n : ℚ) / (n.choose 2) := by
  sorry

end probability_neighboring_points_l1501_150110


namespace two_numbers_difference_l1501_150109

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 162) : 
  |x - y| = 6 * Real.sqrt 7 := by
sorry

end two_numbers_difference_l1501_150109


namespace prime_iff_binomial_congruence_l1501_150192

theorem prime_iff_binomial_congruence (n : ℕ) (hn : n > 0) :
  Nat.Prime n ↔ ∀ k : ℕ, k < n → (Nat.choose (n - 1) k) % n = ((-1 : ℤ) ^ k).toNat % n :=
sorry

end prime_iff_binomial_congruence_l1501_150192


namespace toothpicks_for_2003_base_l1501_150120

def small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

def toothpicks (base : ℕ) : ℕ :=
  let total_triangles := small_triangles base
  3 * total_triangles / 2

theorem toothpicks_for_2003_base :
  toothpicks 2003 = 3010554 :=
by sorry

end toothpicks_for_2003_base_l1501_150120


namespace quiz_competition_participants_l1501_150160

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants * 40 / 100 * 1 / 4 = 30) →
  initial_participants = 300 := by
sorry

end quiz_competition_participants_l1501_150160


namespace circle_properties_l1501_150104

/-- Given a circle with equation x^2 - 24x + y^2 - 4y = -36, 
    prove its center, radius, and the sum of center coordinates and radius. -/
theorem circle_properties : 
  let D : Set (ℝ × ℝ) := {p | (p.1^2 - 24*p.1 + p.2^2 - 4*p.2 = -36)}
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ D ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
    a = 12 ∧ 
    b = 2 ∧ 
    r = 4 * Real.sqrt 7 ∧
    a + b + r = 14 + 4 * Real.sqrt 7 :=
by sorry

end circle_properties_l1501_150104


namespace max_unit_digit_of_2015_divisor_power_l1501_150156

def unit_digit (n : ℕ) : ℕ := n % 10

def is_divisor (d n : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem max_unit_digit_of_2015_divisor_power :
  ∃ (d : ℕ), is_divisor d 2015 ∧
  unit_digit (d^(2015 / d)) = 7 ∧
  ∀ (k : ℕ), is_divisor k 2015 → unit_digit (k^(2015 / k)) ≤ 7 :=
sorry

end max_unit_digit_of_2015_divisor_power_l1501_150156


namespace triangle_circle_perimeter_triangle_circle_perimeter_proof_l1501_150154

/-- The total perimeter of a right triangle with legs 3 and 4, and its inscribed circle -/
theorem triangle_circle_perimeter : ℝ → Prop :=
  fun total_perimeter =>
    ∃ (hypotenuse radius : ℝ),
      -- Triangle properties
      hypotenuse^2 = 3^2 + 4^2 ∧
      -- Circle properties
      radius > 0 ∧
      -- Area of triangle equals semiperimeter times radius
      (3 * 4 / 2 : ℝ) = ((3 + 4 + hypotenuse) / 2) * radius ∧
      -- Total perimeter calculation
      total_perimeter = (3 + 4 + hypotenuse) + 2 * Real.pi * radius ∧
      total_perimeter = 12 + 2 * Real.pi

/-- Proof of the theorem -/
theorem triangle_circle_perimeter_proof : triangle_circle_perimeter (12 + 2 * Real.pi) := by
  sorry

#check triangle_circle_perimeter_proof

end triangle_circle_perimeter_triangle_circle_perimeter_proof_l1501_150154


namespace collinear_points_k_l1501_150167

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_k (k : ℝ) : 
  collinear ⟨1, -4⟩ ⟨3, 2⟩ ⟨6, k/3⟩ → k = 33 := by
  sorry


end collinear_points_k_l1501_150167


namespace polynomial_remainder_l1501_150177

/-- Given a polynomial Q with Q(25) = 50 and Q(50) = 25, 
    the remainder when Q is divided by (x - 25)(x - 50) is -x + 75 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 25 = 50) (h2 : Q 50 = 25) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 25) * (x - 50) * R x + (-x + 75) :=
sorry

end polynomial_remainder_l1501_150177


namespace base6_division_l1501_150130

/-- Converts a base 6 number to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The quotient of 2134₆ divided by 14₆ is equal to 81₆ in base 6 --/
theorem base6_division :
  toBase6 (toBase10 [4, 3, 1, 2] / toBase10 [4, 1]) = [1, 8] := by
  sorry

end base6_division_l1501_150130


namespace possible_total_students_l1501_150102

/-- Represents the possible total number of students -/
inductive TotalStudents
  | seventySix
  | eighty

/-- Checks if a number is a valid group size given the constraints -/
def isValidGroupSize (size : ℕ) : Prop :=
  size = 12 ∨ size = 13 ∨ size = 14

/-- Represents the distribution of students into groups -/
structure StudentDistribution where
  groupSizes : Fin 6 → ℕ
  validSizes : ∀ i, isValidGroupSize (groupSizes i)
  fourGroupsOf13 : (Finset.filter (fun i => groupSizes i = 13) Finset.univ).card = 4
  totalStudents : TotalStudents

/-- The main theorem stating the possible total number of students -/
theorem possible_total_students (d : StudentDistribution) :
    d.totalStudents = TotalStudents.seventySix ∨
    d.totalStudents = TotalStudents.eighty :=
  sorry

end possible_total_students_l1501_150102


namespace polynomial_simplification_l1501_150157

theorem polynomial_simplification (q : ℝ) : 
  (5 * q^4 + 3 * q^3 - 7 * q + 8) + (6 - 9 * q^3 + 4 * q - 3 * q^4) = 
  2 * q^4 - 6 * q^3 - 3 * q + 14 := by
sorry

end polynomial_simplification_l1501_150157


namespace circle_equation_specific_l1501_150189

/-- The standard equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (2, -1) and radius 3 -/
theorem circle_equation_specific : ∀ x y : ℝ,
  circle_equation x y 2 (-1) 3 ↔ (x - 2)^2 + (y + 1)^2 = 9 := by
  sorry

end circle_equation_specific_l1501_150189


namespace line_through_point_parallel_to_line_l1501_150134

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def linesParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (A : Point)
  (l : Line)
  (h_A : A.x = 1 ∧ A.y = 1)
  (h_l : l.a = 3 ∧ l.b = -2 ∧ l.c = 1) :
  ∃ (result : Line),
    result.a = 3 ∧ result.b = -2 ∧ result.c = -1 ∧
    pointOnLine A result ∧
    linesParallel result l :=
  sorry

end line_through_point_parallel_to_line_l1501_150134


namespace fraction_order_l1501_150115

theorem fraction_order : (21 : ℚ) / 17 < 23 / 18 ∧ 23 / 18 < 25 / 19 := by
  sorry

end fraction_order_l1501_150115


namespace sine_shift_and_stretch_l1501_150197

/-- Given a function f(x) = sin(x), prove that shifting it right by π/10 units
    and then stretching the x-coordinates by a factor of 2 results in
    the function g(x) = sin(1/2x - π/10) -/
theorem sine_shift_and_stretch (x : ℝ) :
  let f : ℝ → ℝ := λ t ↦ Real.sin t
  let shift : ℝ → ℝ := λ t ↦ t - π / 10
  let stretch : ℝ → ℝ := λ t ↦ t / 2
  let g : ℝ → ℝ := λ t ↦ Real.sin (1/2 * t - π / 10)
  f (stretch (shift x)) = g x := by
  sorry

end sine_shift_and_stretch_l1501_150197


namespace derivative_greater_than_average_rate_of_change_l1501_150139

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (1 - 2*a) * x - Real.log x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2*a*x + (1 - 2*a) - 1/x

-- Theorem statement
theorem derivative_greater_than_average_rate_of_change 
  (a : ℝ) (x0 x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x0 ≠ (x1 + x2) / 2) :
  f' a x0 > (f a x1 - f a x2) / (x1 - x2) := by
  sorry

end

end derivative_greater_than_average_rate_of_change_l1501_150139


namespace saramago_readers_l1501_150152

/-- Represents the number of workers in Palabras bookstore who have read certain books. -/
structure BookReaders where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  both : ℕ
  neither : ℕ

/-- The conditions given in the problem. -/
def palabras_conditions (r : BookReaders) : Prop :=
  r.total = 150 ∧
  r.kureishi = r.total / 6 ∧
  r.both = 12 ∧
  r.neither = r.saramago - r.both - 1 ∧
  r.saramago - r.both + r.kureishi - r.both + r.both + r.neither = r.total

/-- The theorem to be proved. -/
theorem saramago_readers (r : BookReaders) 
  (h : palabras_conditions r) : r.saramago = 75 := by
  sorry

#check saramago_readers

end saramago_readers_l1501_150152


namespace wednesday_to_tuesday_rainfall_ratio_l1501_150174

/-- Represents the rainfall data for a day -/
structure RainfallData where
  hours : ℝ
  rate : ℝ

/-- Calculates the total rainfall for a given day -/
def totalRainfall (data : RainfallData) : ℝ := data.hours * data.rate

theorem wednesday_to_tuesday_rainfall_ratio :
  let monday : RainfallData := { hours := 7, rate := 1 }
  let tuesday : RainfallData := { hours := 4, rate := 2 }
  let wednesday : RainfallData := { hours := 2, rate := ((23 : ℝ) - totalRainfall monday - totalRainfall tuesday) / 2 }
  wednesday.rate / tuesday.rate = 2 := by sorry

end wednesday_to_tuesday_rainfall_ratio_l1501_150174


namespace division_simplification_l1501_150106

theorem division_simplification (a b c d e f : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) / (e * f) = (a * d) / (b * c * e * f) := by
  sorry

end division_simplification_l1501_150106


namespace exam_score_calculation_l1501_150119

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 150)
  (h3 : correct_answers = 42)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end exam_score_calculation_l1501_150119


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1501_150191

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The condition that a = 1 is sufficient for the lines to be parallel -/
theorem sufficient_condition (a : ℝ) :
  a = 1 → parallel 1 a (-1) (2*a - 1) a (-2) := by sorry

/-- The condition that a = 1 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ 1 ∧ parallel 1 a (-1) (2*a - 1) a (-2) := by sorry

/-- The main theorem stating that a = 1 is sufficient but not necessary -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = 1 → parallel 1 a (-1) (2*a - 1) a (-2)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ parallel 1 a (-1) (2*a - 1) a (-2)) := by sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1501_150191


namespace probability_odd_limit_l1501_150147

/-- Represents the probability of getting an odd number after n button presses -/
def probability_odd (n : ℕ) : ℝ := sorry

/-- The recurrence relation for the probability of getting an odd number -/
axiom probability_recurrence (n : ℕ) : 
  probability_odd (n + 1) = probability_odd n - (1/2) * (probability_odd n)^2

/-- The initial probability (after one button press) is not exactly 1/3 -/
axiom initial_probability : probability_odd 1 ≠ 1/3

/-- Theorem: The probability of getting an odd number converges to 1/3 -/
theorem probability_odd_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |probability_odd n - 1/3| < ε :=
sorry

end probability_odd_limit_l1501_150147


namespace hybrid_cars_with_full_headlights_l1501_150155

/-- Given a car dealership with the following properties:
  * There are 600 cars in total
  * 60% of cars are hybrids
  * 40% of hybrids have only one headlight
  Prove that the number of hybrids with full headlights is 216 -/
theorem hybrid_cars_with_full_headlights 
  (total_cars : ℕ) 
  (hybrid_percentage : ℚ) 
  (one_headlight_percentage : ℚ) 
  (h1 : total_cars = 600)
  (h2 : hybrid_percentage = 60 / 100)
  (h3 : one_headlight_percentage = 40 / 100) :
  ↑total_cars * hybrid_percentage - ↑total_cars * hybrid_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrid_cars_with_full_headlights_l1501_150155


namespace cost_price_calculation_l1501_150132

/-- Proves that if an item is sold for 1260 with a 16% loss, its cost price was 1500 --/
theorem cost_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1260)
  (h2 : loss_percentage = 16) : 
  (selling_price / (1 - loss_percentage / 100)) = 1500 := by
  sorry

end cost_price_calculation_l1501_150132


namespace prove_n_equals_two_l1501_150146

def a (n k : ℕ) : ℕ := (n * k + 1) ^ k

theorem prove_n_equals_two (n : ℕ) : a n (a n (a n 0)) = 343 → n = 2 := by
  sorry

end prove_n_equals_two_l1501_150146


namespace fraction_problem_l1501_150158

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 180 → 
  (1/2 * f * 1/5 * N) + 6 = 1/15 * N → 
  f = 1/3 := by
sorry

end fraction_problem_l1501_150158


namespace unique_parallel_line_l1501_150194

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a point
variable (Point : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the relation of a point lying on a plane
variable (lies_on : Point → Plane → Prop)

-- Define the relation of two planes intersecting
variable (intersect : Plane → Plane → Prop)

-- Define the relation of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the relation of a line passing through a point
variable (passes_through : Line → Point → Prop)

-- Theorem statement
theorem unique_parallel_line 
  (α β : Plane) (A : Point) 
  (h_intersect : intersect α β)
  (h_not_on_α : ¬ lies_on A α)
  (h_not_on_β : ¬ lies_on A β) :
  ∃! l : Line, passes_through l A ∧ parallel_to_plane l α ∧ parallel_to_plane l β :=
sorry

end unique_parallel_line_l1501_150194


namespace pure_imaginary_a_value_l1501_150196

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number z -/
def z (a : ℝ) : ℂ := (a + i) * (3 + 2*i)

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

theorem pure_imaginary_a_value :
  ∃ (a : ℝ), is_pure_imaginary (z a) ∧ a = 2/3 :=
sorry

end pure_imaginary_a_value_l1501_150196
