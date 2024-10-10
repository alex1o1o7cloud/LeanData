import Mathlib

namespace circle_radius_in_square_with_semicircles_l3058_305840

/-- Given a square with side length 108 and semicircles constructed inward on two adjacent sides,
    the radius of a circle touching one side and both semicircles is 27. -/
theorem circle_radius_in_square_with_semicircles (square_side : ℝ) 
  (h_side : square_side = 108) : ∃ (r : ℝ), r = 27 ∧ 
  r + (square_side / 2) = square_side - r := by
  sorry

end circle_radius_in_square_with_semicircles_l3058_305840


namespace average_height_of_four_l3058_305862

/-- Given the heights of four people with specific relationships, prove their average height --/
theorem average_height_of_four (zara_height brixton_height zora_height itzayana_height : ℕ) : 
  zara_height = 64 →
  brixton_height = zara_height →
  zora_height = brixton_height - 8 →
  itzayana_height = zora_height + 4 →
  (zara_height + brixton_height + zora_height + itzayana_height) / 4 = 61 := by
  sorry

end average_height_of_four_l3058_305862


namespace corrected_mean_problem_l3058_305805

/-- Calculates the corrected mean when one observation in a dataset is incorrect -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean + (correct_value - incorrect_value)) / n

/-- Theorem stating that the corrected mean is 36.02 given the problem conditions -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let incorrect_value : ℚ := 47
  let correct_value : ℚ := 48
  corrected_mean n initial_mean incorrect_value correct_value = 3602/100 := by
  sorry

#eval corrected_mean 50 36 47 48

end corrected_mean_problem_l3058_305805


namespace inequality_condition_l3058_305897

theorem inequality_condition (A B C : ℝ) : 
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔ 
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :=
by sorry

end inequality_condition_l3058_305897


namespace linear_equation_implies_a_equals_negative_two_l3058_305858

theorem linear_equation_implies_a_equals_negative_two (a : ℝ) : 
  (∀ x, (a - 2) * x^(|a| - 1) - 2 = 0 → ∃ m k, (a - 2) * x^(|a| - 1) - 2 = m * x + k) → 
  a = -2 :=
by sorry

end linear_equation_implies_a_equals_negative_two_l3058_305858


namespace martin_distance_l3058_305836

/-- The distance traveled by Martin -/
def distance : ℝ := 72.0

/-- Martin's driving speed in miles per hour -/
def speed : ℝ := 12.0

/-- Time taken for Martin's journey in hours -/
def time : ℝ := 6.0

/-- Theorem stating that the distance Martin traveled is equal to his speed multiplied by the time taken -/
theorem martin_distance : distance = speed * time := by
  sorry

end martin_distance_l3058_305836


namespace hanks_total_reading_time_l3058_305812

/-- Represents Hank's weekly reading schedule --/
structure ReadingSchedule where
  newspaper_days : Nat
  newspaper_time : Nat
  magazine_time : Nat
  novel_days : Nat
  novel_time : Nat
  novel_friday_time : Nat
  novel_saturday_multiplier : Nat
  novel_sunday_multiplier : Nat
  scientific_journal_time : Nat
  nonfiction_time : Nat

/-- Calculates the total reading time for a week given a reading schedule --/
def total_reading_time (schedule : ReadingSchedule) : Nat :=
  let newspaper_total := schedule.newspaper_days * schedule.newspaper_time
  let magazine_total := schedule.magazine_time
  let novel_weekday_total := (schedule.novel_days - 3) * schedule.novel_time
  let novel_friday_total := schedule.novel_friday_time
  let novel_saturday_total := schedule.novel_time * schedule.novel_saturday_multiplier
  let novel_sunday_total := schedule.novel_time * schedule.novel_sunday_multiplier
  let scientific_journal_total := schedule.scientific_journal_time
  let nonfiction_total := schedule.nonfiction_time
  newspaper_total + magazine_total + novel_weekday_total + novel_friday_total +
  novel_saturday_total + novel_sunday_total + scientific_journal_total + nonfiction_total

/-- Hank's actual reading schedule --/
def hanks_schedule : ReadingSchedule :=
  { newspaper_days := 5
  , newspaper_time := 30
  , magazine_time := 15
  , novel_days := 5
  , novel_time := 60
  , novel_friday_time := 90
  , novel_saturday_multiplier := 2
  , novel_sunday_multiplier := 3
  , scientific_journal_time := 45
  , nonfiction_time := 40
  }

/-- Theorem stating that Hank's total reading time in a week is 760 minutes --/
theorem hanks_total_reading_time :
  total_reading_time hanks_schedule = 760 := by
  sorry

end hanks_total_reading_time_l3058_305812


namespace range_of_f_inequality_l3058_305864

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem range_of_f_inequality 
  (hdom : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f x ≠ 0 → True)
  (hderiv : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → HasDerivAt f (x^2 + 2 * Real.cos x) x)
  (hf0 : f 0 = 0) :
  {x : ℝ | f (1 + x) + f (x - x^2) > 0} = Set.Ioo (1 - Real.sqrt 2) 1 := by
  sorry

end range_of_f_inequality_l3058_305864


namespace sum_of_consecutive_iff_not_power_of_two_l3058_305835

/-- A function that checks if a number is a sum of consecutive integers -/
def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (start k : ℕ), k ≥ 2 ∧ n = (k * (2 * start + k + 1)) / 2

/-- A function that checks if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- Theorem stating that a positive integer is a sum of two or more consecutive
    positive integers if and only if it is not a power of 2 -/
theorem sum_of_consecutive_iff_not_power_of_two (n : ℕ) (h : n > 0) :
  is_sum_of_consecutive n ↔ ¬ is_power_of_two n :=
sorry

end sum_of_consecutive_iff_not_power_of_two_l3058_305835


namespace d_equals_square_cases_l3058_305819

/-- Function to move the last digit of a number to the first position -/
def moveLastToFirst (n : ℕ) : ℕ := sorry

/-- Function to move the first digit of a number to the last position -/
def moveFirstToLast (n : ℕ) : ℕ := sorry

/-- The d function as described in the problem -/
def d (a : ℕ) : ℕ := 
  let b := moveLastToFirst a
  let c := b * b
  moveFirstToLast c

/-- Theorem stating the possible forms of a when d(a) = a^2 -/
theorem d_equals_square_cases (a : ℕ) (h : 0 < a) : 
  d a = a * a → (a = 2 ∨ a = 3 ∨ ∃ x y : ℕ, a = 20000 + 100 * x + 10 * y + 21) := by
  sorry

end d_equals_square_cases_l3058_305819


namespace sprite_volume_l3058_305877

def maazaVolume : ℕ := 80
def pepsiVolume : ℕ := 144
def totalCans : ℕ := 37

def canVolume : ℕ := Nat.gcd maazaVolume pepsiVolume

theorem sprite_volume :
  ∃ (spriteVolume : ℕ),
    spriteVolume = canVolume * (totalCans - (maazaVolume / canVolume + pepsiVolume / canVolume)) ∧
    spriteVolume = 368 := by
  sorry

end sprite_volume_l3058_305877


namespace coin_game_probability_l3058_305844

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing three coins -/
structure ThreeCoinToss :=
  (first second third : CoinOutcome)

/-- Defines a winning outcome in the Coin Game -/
def is_winning_toss (toss : ThreeCoinToss) : Prop :=
  (toss.first = CoinOutcome.Heads ∧ toss.second = CoinOutcome.Heads ∧ toss.third = CoinOutcome.Heads) ∨
  (toss.first = CoinOutcome.Tails ∧ toss.second = CoinOutcome.Tails ∧ toss.third = CoinOutcome.Tails)

/-- The set of all possible outcomes when tossing three coins -/
def all_outcomes : Finset ThreeCoinToss := sorry

/-- The set of winning outcomes in the Coin Game -/
def winning_outcomes : Finset ThreeCoinToss := sorry

/-- Theorem stating that the probability of winning the Coin Game is 1/4 -/
theorem coin_game_probability : 
  (Finset.card winning_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 4 := by
  sorry

end coin_game_probability_l3058_305844


namespace increasing_function_inequality_l3058_305816

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → f x < f y) →
  (∀ x, x ∈ Set.Iio 2 → f x ≠ 0) →
  f (a - 1) > f (1 - 3 * a) →
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) := by
  sorry


end increasing_function_inequality_l3058_305816


namespace first_movie_length_is_correct_l3058_305807

/-- The length of the first movie in minutes -/
def first_movie_length : ℕ := 90

/-- The length of the second movie in minutes -/
def second_movie_length : ℕ := first_movie_length + 30

/-- The time spent making popcorn in minutes -/
def popcorn_time : ℕ := 10

/-- The time spent making fries in minutes -/
def fries_time : ℕ := 2 * popcorn_time

/-- The total time spent cooking and watching movies in minutes -/
def total_time : ℕ := 4 * 60

theorem first_movie_length_is_correct : 
  first_movie_length + second_movie_length + popcorn_time + fries_time = total_time := by
  sorry

end first_movie_length_is_correct_l3058_305807


namespace min_coins_for_distribution_l3058_305838

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 95) :
  min_additional_coins num_friends initial_coins = 25 := by
  sorry

end min_coins_for_distribution_l3058_305838


namespace weeks_to_save_for_shirt_l3058_305855

/-- Calculate the number of weeks needed to save for a shirt -/
theorem weeks_to_save_for_shirt (total_cost saved_amount savings_rate : ℚ) : 
  total_cost = 3 →
  saved_amount = 3/2 →
  savings_rate = 1/2 →
  (total_cost - saved_amount) / savings_rate = 3 := by
  sorry

#check weeks_to_save_for_shirt

end weeks_to_save_for_shirt_l3058_305855


namespace ny_mets_fans_count_l3058_305854

theorem ny_mets_fans_count (total_fans : ℕ) (yankees_mets_ratio : ℚ) (mets_redsox_ratio : ℚ) :
  total_fans = 390 →
  yankees_mets_ratio = 3 / 2 →
  mets_redsox_ratio = 4 / 5 →
  ∃ (yankees mets redsox : ℕ),
    yankees + mets + redsox = total_fans ∧
    (yankees : ℚ) / mets = yankees_mets_ratio ∧
    (mets : ℚ) / redsox = mets_redsox_ratio ∧
    mets = 104 :=
by sorry

end ny_mets_fans_count_l3058_305854


namespace hotel_room_pricing_l3058_305828

theorem hotel_room_pricing (total_rooms : ℕ) (double_rooms : ℕ) (single_room_cost : ℕ) (total_revenue : ℕ) :
  total_rooms = 260 →
  double_rooms = 196 →
  single_room_cost = 35 →
  total_revenue = 14000 →
  ∃ (double_room_cost : ℕ),
    double_room_cost = 60 ∧
    total_revenue = (total_rooms - double_rooms) * single_room_cost + double_rooms * double_room_cost :=
by
  sorry

#check hotel_room_pricing

end hotel_room_pricing_l3058_305828


namespace hyperbola_asymptotes_l3058_305856

/-- The equations of the asymptotes of the hyperbola y²/9 - x²/4 = 1 are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := λ x y => y^2/9 - x^2/4 - 1
  ∀ x y : ℝ, h x y = 0 →
  ∃ k : ℝ, k = 3/2 ∧
  (∀ ε > 0, ∃ M : ℝ, ∀ x y : ℝ, h x y = 0 ∧ |x| > M →
    (|y - k*x| < ε*|x| ∨ |y + k*x| < ε*|x|)) :=
by sorry

end hyperbola_asymptotes_l3058_305856


namespace div_power_eq_reciprocal_pow_l3058_305821

/-- Definition of division power for rational numbers -/
def div_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (div_power a (n - 1))

/-- Theorem: Division power is equivalent to reciprocal exponentiation -/
theorem div_power_eq_reciprocal_pow (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  div_power a n = (1 / a) ^ (n - 2) :=
sorry

end div_power_eq_reciprocal_pow_l3058_305821


namespace problem_solution_l3058_305834

theorem problem_solution (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := by
sorry

end problem_solution_l3058_305834


namespace quadratic_factorization_l3058_305868

theorem quadratic_factorization (p q : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q)) →
  p - q = 36 :=
by sorry

end quadratic_factorization_l3058_305868


namespace clothing_popularity_l3058_305889

/-- Represents the sales of clothing on a given day in July -/
def sales (n : ℕ) : ℕ :=
  if n ≤ 13 then 3 * n else 65 - 2 * n

/-- Represents the cumulative sales up to a given day in July -/
def cumulative_sales (n : ℕ) : ℕ :=
  if n ≤ 13 then (3 + 3 * n) * n / 2 else 273 + (51 - n) * (n - 13)

/-- The day when the clothing becomes popular -/
def popular_start : ℕ := 12

/-- The day when the clothing is no longer popular -/
def popular_end : ℕ := 22

theorem clothing_popularity :
  (∀ n : ℕ, n ≥ popular_start → n ≤ popular_end → cumulative_sales n ≥ 200) ∧
  (∀ n : ℕ, n > popular_end → sales n < 20) ∧
  popular_end - popular_start + 1 = 11 := by sorry

end clothing_popularity_l3058_305889


namespace monotonicity_and_range_l3058_305845

noncomputable def f (a x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + a^2

theorem monotonicity_and_range :
  (∀ x > 0, ∀ y > 0, (2-Real.sqrt 2)/2 < x → x < y → y < (2+Real.sqrt 2)/2 → f 2 y < f 2 x) ∧
  (∀ x > 0, ∀ y > 0, 0 < x → x < y → y < (2-Real.sqrt 2)/2 → f 2 x < f 2 y) ∧
  (∀ x > 0, ∀ y > 0, (2+Real.sqrt 2)/2 < x → x < y → f 2 x < f 2 y) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, x < y → f a x ≥ f a y) → a ≥ 19/6) :=
sorry

end monotonicity_and_range_l3058_305845


namespace calvin_winning_condition_l3058_305886

/-- The game state represents the current configuration of coins on the circle. -/
structure GameState where
  n : ℕ
  coins : Fin (2 * n + 1) → Bool

/-- A player's move in the game. -/
inductive Move
  | calvin : Fin (2 * n + 1) → Move
  | hobbes : Option (Fin (2 * n + 1)) → Move

/-- Applies a move to the current game state. -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Counts the number of tails in the current game state. -/
def countTails (state : GameState) : ℕ :=
  sorry

/-- Determines if a player has a winning strategy for the game. -/
def hasWinningStrategy (n k : ℕ) (player : Bool) : Prop :=
  sorry

/-- The main theorem stating the conditions for Calvin's victory. -/
theorem calvin_winning_condition (n k : ℕ) (h1 : n > 1) (h2 : k ≥ 1) :
  hasWinningStrategy n k true ↔ k ≤ n + 1 :=
  sorry

end calvin_winning_condition_l3058_305886


namespace quadratic_always_real_root_l3058_305827

theorem quadratic_always_real_root (b : ℝ) : 
  ∃ x : ℝ, x^2 + b*x - 20 = 0 := by
  sorry

end quadratic_always_real_root_l3058_305827


namespace matrix_max_min_element_l3058_305873

theorem matrix_max_min_element
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (p : Fin m → ℝ)
  (q : Fin n → ℝ)
  (hp : ∀ i, p i > 0)
  (hq : ∀ j, q j > 0) :
  ∃ (k : Fin m) (l : Fin n),
    (∀ j, (a k + b l) / (p k + q l) ≥ (a k + b j) / (p k + q j)) ∧
    (∀ i, (a k + b l) / (p k + q l) ≤ (a i + b l) / (p i + q l)) :=
by sorry

end matrix_max_min_element_l3058_305873


namespace first_division_percentage_l3058_305843

theorem first_division_percentage 
  (total_students : ℕ) 
  (second_division_percentage : ℚ)
  (just_passed_count : ℕ) :
  total_students = 300 →
  second_division_percentage = 54/100 →
  just_passed_count = 51 →
  ∃ (first_division_percentage : ℚ),
    first_division_percentage = 29/100 ∧
    first_division_percentage + second_division_percentage + (just_passed_count : ℚ) / total_students = 1 :=
by sorry

end first_division_percentage_l3058_305843


namespace fruits_remaining_proof_l3058_305802

def initial_apples : ℕ := 7
def initial_oranges : ℕ := 8
def initial_mangoes : ℕ := 15

def apples_taken : ℕ := 2
def oranges_taken : ℕ := 2 * apples_taken
def mangoes_taken : ℕ := (2 * initial_mangoes) / 3

def remaining_fruits : ℕ :=
  (initial_apples - apples_taken) +
  (initial_oranges - oranges_taken) +
  (initial_mangoes - mangoes_taken)

theorem fruits_remaining_proof :
  remaining_fruits = 14 := by sorry

end fruits_remaining_proof_l3058_305802


namespace equation_solution_l3058_305891

theorem equation_solution (x : ℝ) (h : 5 / (4 + 1/x) = 1) : x = 1 := by
  sorry

end equation_solution_l3058_305891


namespace probability_even_sum_l3058_305853

theorem probability_even_sum (wheel1_even : ℚ) (wheel1_odd : ℚ) 
  (wheel2_even : ℚ) (wheel2_odd : ℚ) : 
  wheel1_even = 1/4 →
  wheel1_odd = 3/4 →
  wheel2_even = 2/3 →
  wheel2_odd = 1/3 →
  wheel1_even + wheel1_odd = 1 →
  wheel2_even + wheel2_odd = 1 →
  wheel1_even * wheel2_even + wheel1_odd * wheel2_odd = 5/12 := by
  sorry

end probability_even_sum_l3058_305853


namespace dot_product_range_l3058_305867

/-- The locus M is defined as the set of points (x, y) satisfying x²/3 + y² = 1 -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

/-- F is the point (2, 0) -/
def F : ℝ × ℝ := (2, 0)

/-- Given two points on M, compute their dot product with respect to F -/
def dot_product_with_F (C D : ℝ × ℝ) : ℝ :=
  let FC := (C.1 - F.1, C.2 - F.2)
  let FD := (D.1 - F.1, D.2 - F.2)
  FC.1 * FD.1 + FC.2 * FD.2

/-- The main theorem stating the range of the dot product -/
theorem dot_product_range (C D : ℝ × ℝ) (hC : C ∈ M) (hD : D ∈ M) 
  (h_line : ∃ (k : ℝ), C.2 = k * (C.1 - 2) ∧ D.2 = k * (D.1 - 2)) :
  1/3 < dot_product_with_F C D ∧ dot_product_with_F C D ≤ 1 := by
  sorry


end dot_product_range_l3058_305867


namespace pencil_profit_calculation_pencil_profit_proof_l3058_305811

theorem pencil_profit_calculation (purchase_quantity : ℕ) (purchase_price : ℚ) 
  (selling_price : ℚ) (desired_profit : ℚ) (sold_quantity : ℕ) : Prop :=
  purchase_quantity = 2000 →
  purchase_price = 15/100 →
  selling_price = 30/100 →
  desired_profit = 150 →
  sold_quantity = 1500 →
  (sold_quantity : ℚ) * selling_price - (purchase_quantity : ℚ) * purchase_price = desired_profit

/-- Proof that selling 1500 pencils at $0.30 each results in a profit of $150 
    when 2000 pencils were purchased at $0.15 each. -/
theorem pencil_profit_proof :
  pencil_profit_calculation 2000 (15/100) (30/100) 150 1500 := by
  sorry

end pencil_profit_calculation_pencil_profit_proof_l3058_305811


namespace arithmetic_sequence_properties_l3058_305875

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  h1 : a 1 < 0
  h2 : a 10 + a 15 = a 12
  h3 : ∀ n, a n = a 1 + (n - 1) * d
  h4 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n m, n < m → seq.a n < seq.a m) ∧
  (∀ n, n ≠ 12 ∧ n ≠ 13 → seq.S 12 ≤ seq.S n ∧ seq.S 13 ≤ seq.S n) :=
sorry

end arithmetic_sequence_properties_l3058_305875


namespace equation_has_solution_l3058_305872

theorem equation_has_solution (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := by
  sorry

end equation_has_solution_l3058_305872


namespace smallest_sum_of_digits_l3058_305888

/-- Represents an n-digit number with all digits equal to d -/
def digits_number (n : ℕ+) (d : ℕ) : ℕ :=
  d * (10^n.val - 1) / 9

/-- The equation C_n - A_n = B_n^2 holds for at least two distinct values of n -/
def equation_holds (a b c : ℕ) : Prop :=
  ∃ n m : ℕ+, n ≠ m ∧
    digits_number (2*n) c - digits_number n a = (digits_number n b)^2 ∧
    digits_number (2*m) c - digits_number m a = (digits_number m b)^2

theorem smallest_sum_of_digits :
  ∀ a b c : ℕ,
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    equation_holds a b c →
    ∀ x y z : ℕ,
      0 < x ∧ x < 10 →
      0 < y ∧ y < 10 →
      0 < z ∧ z < 10 →
      equation_holds x y z →
      5 ≤ x + y + z :=
by sorry

end smallest_sum_of_digits_l3058_305888


namespace min_value_bisecting_line_l3058_305857

/-- The minimum value of 1/a + 1/b for a line ax + by - 1 = 0 bisecting a specific circle -/
theorem min_value_bisecting_line (a b : ℝ) : 
  a * b > 0 → 
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 - 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + 1 = 0 → (a * x + b * y - 1) * (a * x + b * y - 1) ≤ (a^2 + b^2) * ((x-1)^2 + (y-2)^2)) →
  (1/a + 1/b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_bisecting_line_l3058_305857


namespace xy_fraction_sum_l3058_305895

theorem xy_fraction_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 := by
  sorry

end xy_fraction_sum_l3058_305895


namespace ragnar_wood_chopping_l3058_305863

/-- Represents the number of blocks of wood obtained from chopping trees over a period of time. -/
structure WoodChopping where
  trees_per_day : ℕ
  days : ℕ
  total_blocks : ℕ

/-- Calculates the number of blocks of wood obtained from one tree. -/
def blocks_per_tree (w : WoodChopping) : ℚ :=
  w.total_blocks / (w.trees_per_day * w.days)

/-- Theorem stating that given the specific conditions, the number of blocks per tree is 3. -/
theorem ragnar_wood_chopping :
  let w : WoodChopping := { trees_per_day := 2, days := 5, total_blocks := 30 }
  blocks_per_tree w = 3 := by sorry

end ragnar_wood_chopping_l3058_305863


namespace emptyBoxes_l3058_305849

/-- Represents the state of the two boxes with pebbles -/
structure BoxState :=
  (p : ℕ)
  (q : ℕ)

/-- Defines a single step operation on the boxes -/
inductive Step
  | Remove : Step
  | TripleP : Step
  | TripleQ : Step

/-- Applies a single step to a BoxState -/
def applyStep (state : BoxState) (step : Step) : BoxState :=
  match step with
  | Step.Remove => ⟨state.p - 1, state.q - 1⟩
  | Step.TripleP => ⟨state.p * 3, state.q⟩
  | Step.TripleQ => ⟨state.p, state.q * 3⟩

/-- Checks if a BoxState is empty (both boxes have 0 pebbles) -/
def isEmpty (state : BoxState) : Prop :=
  state.p = 0 ∧ state.q = 0

/-- Defines if it's possible to empty both boxes from a given initial state -/
def canEmpty (initial : BoxState) : Prop :=
  ∃ (steps : List Step), isEmpty (steps.foldl applyStep initial)

/-- The main theorem to be proved -/
theorem emptyBoxes (p q : ℕ) :
  canEmpty ⟨p, q⟩ ↔ p % 2 = q % 2 := by sorry


end emptyBoxes_l3058_305849


namespace negation_of_universal_proposition_l3058_305890

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≤ 1) ↔ (∃ x₀ : ℝ, x₀^2 > 1) := by sorry

end negation_of_universal_proposition_l3058_305890


namespace mn_value_l3058_305860

theorem mn_value (m n : ℤ) : 
  (∀ x : ℤ, x^2 + m*x - 15 = (x + 3)*(x + n)) → m*n = 10 := by
  sorry

end mn_value_l3058_305860


namespace alBr3_weight_calculation_l3058_305824

/-- Calculates the weight of AlBr3 given moles and isotope data -/
def weightAlBr3 (moles : ℝ) (alMass : ℝ) (br79Mass br81Mass : ℝ) (br79Abundance br81Abundance : ℝ) : ℝ :=
  let brAvgMass := br79Mass * br79Abundance + br81Mass * br81Abundance
  let molarMass := alMass + 3 * brAvgMass
  moles * molarMass

/-- The weight of 4 moles of AlBr3 is approximately 1067.2344 grams -/
theorem alBr3_weight_calculation :
  let moles : ℝ := 4
  let alMass : ℝ := 27
  let br79Mass : ℝ := 79
  let br81Mass : ℝ := 81
  let br79Abundance : ℝ := 0.5069
  let br81Abundance : ℝ := 0.4931
  ∃ ε > 0, |weightAlBr3 moles alMass br79Mass br81Mass br79Abundance br81Abundance - 1067.2344| < ε :=
by sorry

end alBr3_weight_calculation_l3058_305824


namespace fixed_point_on_circle_l3058_305896

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The parabola x^2 = 12y -/
def on_parabola (p : Point) : Prop :=
  p.x^2 = 12 * p.y

/-- The line y = -3 -/
def on_line (p : Point) : Prop :=
  p.y = -3

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The circle is tangent to the line y = -3 -/
def tangent_to_line (c : Circle) : Prop :=
  c.center.y + c.radius = -3

/-- Main theorem -/
theorem fixed_point_on_circle :
  ∀ (c : Circle),
    on_parabola c.center →
    tangent_to_line c →
    on_circle ⟨0, 3⟩ c :=
sorry

end fixed_point_on_circle_l3058_305896


namespace arithmetic_mean_of_one_and_four_l3058_305848

theorem arithmetic_mean_of_one_and_four :
  (1 + 4) / 2 = 5/2 := by
  sorry

end arithmetic_mean_of_one_and_four_l3058_305848


namespace bananas_left_l3058_305878

def dozen : Nat := 12

theorem bananas_left (initial : Nat) (eaten : Nat) : 
  initial = dozen → eaten = 1 → initial - eaten = 11 := by
  sorry

end bananas_left_l3058_305878


namespace travel_distance_l3058_305846

/-- Proves that given a person traveling equal distances at speeds of 5 km/hr, 10 km/hr, and 15 km/hr,
    and taking a total time of 11 minutes, the total distance traveled is 1.5 km. -/
theorem travel_distance (d : ℝ) : 
  d / 5 + d / 10 + d / 15 = 11 / 60 → 3 * d = 1.5 := by
  sorry

end travel_distance_l3058_305846


namespace negative_expression_l3058_305825

theorem negative_expression : 
  (|(-4)| > 0) ∧ (-(-4) > 0) ∧ ((-4)^2 > 0) ∧ (-4^2 < 0) :=
by sorry

end negative_expression_l3058_305825


namespace frog_to_hamster_ratio_l3058_305894

-- Define the lifespans of the animals
def bat_lifespan : ℕ := 10
def hamster_lifespan : ℕ := bat_lifespan - 6

-- Define the total lifespan
def total_lifespan : ℕ := 30

-- Define the frog's lifespan as a function of the hamster's
def frog_lifespan : ℕ := total_lifespan - (bat_lifespan + hamster_lifespan)

-- Theorem to prove
theorem frog_to_hamster_ratio :
  frog_lifespan / hamster_lifespan = 4 :=
by sorry

end frog_to_hamster_ratio_l3058_305894


namespace average_speed_swim_run_l3058_305866

/-- 
Given a swimmer who swims at 1 mile per hour and runs at 11 miles per hour,
their average speed for these two events (assuming equal distances for both)
is 11/6 miles per hour.
-/
theorem average_speed_swim_run :
  let swim_speed : ℝ := 1
  let run_speed : ℝ := 11
  let total_distance : ℝ := 2 -- Assuming 1 mile each for swimming and running
  let swim_time : ℝ := 1 -- Time to swim 1 mile at 1 mph
  let run_time : ℝ := 1 / 11 -- Time to run 1 mile at 11 mph
  let total_time : ℝ := swim_time + run_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 11 / 6 := by sorry

end average_speed_swim_run_l3058_305866


namespace simplify_fraction_division_l3058_305808

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x^2 - 6*x + 8 ≠ 0) 
  (h2 : x^2 - 8*x + 15 ≠ 0) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  (x - 3) / (x^2 - 6*x + 8) :=
by
  sorry

end simplify_fraction_division_l3058_305808


namespace cylinder_volume_unchanged_l3058_305859

/-- Theorem: For a cylinder with radius 5 inches and height 4 inches, 
    the value of x that keeps the volume unchanged when the radius 
    is increased by x and the height is decreased by x is 5 - 2√10. -/
theorem cylinder_volume_unchanged (R H : ℝ) (x : ℝ) : 
  R = 5 → H = 4 → 
  π * R^2 * H = π * (R + x)^2 * (H - x) → 
  x = 5 - 2 * Real.sqrt 10 :=
by sorry

end cylinder_volume_unchanged_l3058_305859


namespace planning_committee_selections_l3058_305820

def student_council_size : ℕ := 6

theorem planning_committee_selections :
  (Nat.choose student_council_size 3 = 20) →
  (Nat.choose student_council_size 3 = 20) := by
  sorry

end planning_committee_selections_l3058_305820


namespace smallest_k_with_odd_solutions_l3058_305800

/-- The number of positive integral solutions to the equation 2xy - 3x - 5y = k -/
def num_solutions (k : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 * p.2 - 3 * p.1 - 5 * p.2 = k) (Finset.product (Finset.range 1000) (Finset.range 1000))).card

/-- Predicate to check if a number is odd -/
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_k_with_odd_solutions :
  (∀ k < 5, ¬(is_odd (num_solutions k))) ∧ 
  (is_odd (num_solutions 5)) :=
sorry

end smallest_k_with_odd_solutions_l3058_305800


namespace logarithm_relation_l3058_305837

theorem logarithm_relation (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hdist : a ≠ b ∧ a ≠ x ∧ b ≠ x)
  (heq : 4 * (Real.log x / Real.log a)^3 + 5 * (Real.log x / Real.log b)^3 = 7 * (Real.log x)^3) :
  ∃ k, b = a^k ∧ k = (3/5)^(1/3) :=
by sorry

end logarithm_relation_l3058_305837


namespace profit_thirty_for_thirtyfive_l3058_305884

/-- Calculates the profit percentage when selling a different number of articles than the cost price basis -/
def profit_percentage (sold : ℕ) (cost_basis : ℕ) : ℚ :=
  let profit := cost_basis - sold
  (profit / sold) * 100

/-- Theorem stating that selling 30 articles at the price of 35 articles' cost results in a profit of 1/6 * 100% -/
theorem profit_thirty_for_thirtyfive :
  profit_percentage 30 35 = 100 / 6 := by
  sorry

end profit_thirty_for_thirtyfive_l3058_305884


namespace quadratic_one_root_l3058_305870

theorem quadratic_one_root (b c : ℝ) 
  (h1 : ∃! x : ℝ, x^2 + b*x + c = 0)
  (h2 : b = 2*c - 1) : 
  c = 1/2 := by
sorry

end quadratic_one_root_l3058_305870


namespace quadratic_root_sum_inverse_cubes_l3058_305810

theorem quadratic_root_sum_inverse_cubes 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * r^2 + b * r + c = 0)
  (h3 : a * s^2 + b * s + c = 0)
  (h4 : r ≠ s)
  (h5 : a + b + c = 0) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3*a^2 + 3*a*b) / (a + b)^3 :=
sorry

end quadratic_root_sum_inverse_cubes_l3058_305810


namespace decreasing_order_l3058_305880

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the even property of f(x-1)
axiom f_even : ∀ x : ℝ, f (-x - 1) = f (x - 1)

-- Define the decreasing property of f on [-1,+∞)
axiom f_decreasing : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ ≥ -1 → x₂ ≥ -1 → 
  (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log (7/2) / Real.log (1/2))
noncomputable def b : ℝ := f (Real.log (7/2) / Real.log (1/3))
noncomputable def c : ℝ := f (Real.log (3/2) / Real.log 2)

-- The theorem to prove
theorem decreasing_order : b > a ∧ a > c := by sorry

end decreasing_order_l3058_305880


namespace smallest_integer_quadratic_inequality_four_satisfies_inequality_four_is_smallest_l3058_305809

theorem smallest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 36 ≤ 0 → n ≥ 4 :=
by sorry

theorem four_satisfies_inequality :
  4^2 - 13*4 + 36 ≤ 0 :=
by sorry

theorem four_is_smallest :
  ∀ n : ℤ, n < 4 → n^2 - 13*n + 36 > 0 :=
by sorry

end smallest_integer_quadratic_inequality_four_satisfies_inequality_four_is_smallest_l3058_305809


namespace subsidy_calculation_l3058_305829

/-- Represents the "Home Appliances to the Countryside" initiative subsidy calculation -/
theorem subsidy_calculation (x : ℝ) : 
  (20 * x * 0.13 = 2340) ↔ 
  (∃ (subsidy_rate : ℝ) (num_phones : ℕ) (total_subsidy : ℝ),
    subsidy_rate = 0.13 ∧ 
    num_phones = 20 ∧ 
    total_subsidy = 2340 ∧
    num_phones * (x * subsidy_rate) = total_subsidy) :=
by sorry

end subsidy_calculation_l3058_305829


namespace unique_solution_for_t_l3058_305831

/-- A non-zero digit is an integer between 1 and 9, inclusive. -/
def NonZeroDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The expression as a function of k and t -/
def expression (k t : NonZeroDigit) : ℤ :=
  808 + 10 * k.val + 80 * k.val + 8 - (1600 + 6 * t.val + 6)

theorem unique_solution_for_t :
  ∃! (t : NonZeroDigit), ∀ (k : NonZeroDigit),
    ∃ (n : ℤ), expression k t = n ∧ n % 10 = 2 :=
by sorry

end unique_solution_for_t_l3058_305831


namespace range_of_a_l3058_305815

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 - (a+1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) ∧ 
  ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1)) ∧
  (∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → 
  ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1)) :=
by sorry

end range_of_a_l3058_305815


namespace billy_apple_ratio_l3058_305801

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := 9

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := 4 * friday_apples

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := total_apples - (monday_apples + wednesday_apples + thursday_apples + friday_apples)

/-- The ratio of apples eaten on Tuesday to Monday -/
def tuesday_to_monday_ratio : ℚ := tuesday_apples / monday_apples

theorem billy_apple_ratio : tuesday_to_monday_ratio = 2 := by
  sorry

end billy_apple_ratio_l3058_305801


namespace range_of_a_l3058_305841

/-- A function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 15 - 2*a

/-- Predicate to check if there are exactly two positive integers in an open interval -/
def exactly_two_positive_integers (lower upper : ℝ) : Prop :=
  ∃ (n m : ℕ), n < m ∧ 
    (∀ (k : ℕ), lower < k ∧ k < upper ↔ k = n ∨ k = m)

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
    exactly_two_positive_integers x₁ x₂) →
  (31/10 < a ∧ a ≤ 19/6) :=
sorry

end range_of_a_l3058_305841


namespace calculate_expression_l3058_305850

theorem calculate_expression : (π - 2023)^0 + |-9| - 3^2 = 1 := by
  sorry

end calculate_expression_l3058_305850


namespace vacant_seats_l3058_305874

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 62/100) : 
  (total_seats : ℚ) * (1 - filled_percentage) = 228 := by
  sorry

end vacant_seats_l3058_305874


namespace cylindrical_tank_volume_increase_l3058_305852

theorem cylindrical_tank_volume_increase (R H : ℝ) (hR : R = 10) (hH : H = 5) :
  ∃ k : ℝ, k > 0 ∧
  (π * (k * R)^2 * H - π * R^2 * H = π * R^2 * (H + k) - π * R^2 * H) ∧
  k = (1 + Real.sqrt 101) / 10 := by
sorry

end cylindrical_tank_volume_increase_l3058_305852


namespace unique_k_value_l3058_305830

theorem unique_k_value : ∃! k : ℝ, ∀ x : ℝ, x * (2 * x + 3) < k ↔ -5/2 < x ∧ x < 1 := by
  sorry

end unique_k_value_l3058_305830


namespace projection_problem_l3058_305833

def vector1 : ℝ × ℝ := (3, -2)
def vector2 : ℝ × ℝ := (2, 5)

def is_projection (v p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), p = (k * v.1, k * v.2)

theorem projection_problem (v : ℝ × ℝ) (p : ℝ × ℝ) :
  is_projection v p ∧ is_projection v p →
  p = (133/50, 49/50) := by sorry

end projection_problem_l3058_305833


namespace negative_five_meters_decrease_l3058_305879

-- Define a type for distance changes
inductive DistanceChange
| Increase (amount : ℤ)
| Decrease (amount : ℤ)

-- Define a function to interpret integers as distance changes
def interpretDistance (d : ℤ) : DistanceChange :=
  if d > 0 then DistanceChange.Increase d
  else DistanceChange.Decrease (-d)

-- Theorem statement
theorem negative_five_meters_decrease :
  interpretDistance (-5) = DistanceChange.Decrease 5 :=
by sorry

end negative_five_meters_decrease_l3058_305879


namespace triangle_area_is_36_l3058_305823

/-- The area of a triangle formed by three lines in a 2D plane -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

theorem triangle_area_is_36 :
  let line1 := fun (x : ℝ) ↦ 8
  let line2 := fun (x : ℝ) ↦ 2 + x
  let line3 := fun (x : ℝ) ↦ 2 - x
  triangleArea line1 line2 line3 = 36 := by
  sorry

end triangle_area_is_36_l3058_305823


namespace european_stamps_cost_l3058_305851

/-- Represents a country with its stamp counts and price --/
structure Country where
  name : String
  price : ℚ
  count_80s : ℕ
  count_90s : ℕ

/-- Calculates the total cost of stamps for a country in both decades --/
def totalCost (c : Country) : ℚ :=
  c.price * (c.count_80s + c.count_90s)

/-- The set of European countries in Laura's collection --/
def europeanCountries : List Country :=
  [{ name := "France", price := 9/100, count_80s := 10, count_90s := 12 },
   { name := "Spain", price := 7/100, count_80s := 18, count_90s := 16 }]

theorem european_stamps_cost :
  List.sum (europeanCountries.map totalCost) = 436/100 := by
  sorry

end european_stamps_cost_l3058_305851


namespace planes_parallel_if_perpendicular_to_same_line_l3058_305892

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (l : Line) (α β : Plane) (h1 : α ≠ β) :
  perpendicular l α → perpendicular l β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l3058_305892


namespace unique_solution_count_l3058_305869

/-- A system of equations has exactly one solution -/
def has_unique_solution (k : ℝ) : Prop :=
  ∃! x y : ℝ, x^2 + y^2 = 2*k^2 ∧ k*x - y = 2*k

/-- The number of real values of k for which the system has a unique solution -/
theorem unique_solution_count :
  ∃ S : Finset ℝ, (∀ k : ℝ, k ∈ S ↔ has_unique_solution k) ∧ S.card = 3 :=
sorry

end unique_solution_count_l3058_305869


namespace share_premium_percentage_l3058_305814

/-- Calculates the premium percentage on shares given investment details -/
theorem share_premium_percentage
  (total_investment : ℝ)
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : total_investment = 14400)
  (h2 : face_value = 100)
  (h3 : dividend_rate = 0.07)
  (h4 : total_dividend = 840) :
  (total_investment / (total_dividend / (dividend_rate * face_value)) - face_value) / face_value * 100 = 20 := by
  sorry

end share_premium_percentage_l3058_305814


namespace stocker_wait_time_l3058_305832

def total_shopping_time : ℕ := 90
def shopping_time : ℕ := 42
def cart_wait_time : ℕ := 3
def employee_wait_time : ℕ := 13
def checkout_wait_time : ℕ := 18

theorem stocker_wait_time :
  total_shopping_time - shopping_time - (cart_wait_time + employee_wait_time + checkout_wait_time) = 14 := by
  sorry

end stocker_wait_time_l3058_305832


namespace square_area_in_triangle_l3058_305882

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A square in a 2D plane -/
structure Square where
  corners : Fin 4 → ℝ × ℝ

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The area of a square -/
def Square.area (s : Square) : ℝ := sorry

/-- Predicate to check if a square lies within a triangle -/
def Square.liesWithin (s : Square) (t : Triangle) : Prop := sorry

/-- Theorem: The area of any square lying within a triangle does not exceed half of the area of that triangle -/
theorem square_area_in_triangle (t : Triangle) (s : Square) :
  s.liesWithin t → s.area ≤ (1/2) * t.area := by sorry

end square_area_in_triangle_l3058_305882


namespace mrs_crocker_chicken_l3058_305839

def chicken_problem (lyndee_pieces : ℕ) (friend_pieces : ℕ) (num_friends : ℕ) : Prop :=
  lyndee_pieces = 1 ∧ friend_pieces = 2 ∧ num_friends = 5 →
  lyndee_pieces + friend_pieces * num_friends = 11

theorem mrs_crocker_chicken : chicken_problem 1 2 5 := by
  sorry

end mrs_crocker_chicken_l3058_305839


namespace investment_interest_rate_l3058_305881

theorem investment_interest_rate 
  (initial_investment : ℝ)
  (first_duration : ℝ)
  (first_rate : ℝ)
  (second_duration : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_duration = 9/12)
  (h3 : first_rate = 0.09)
  (h4 : second_duration = 9/12)
  (h5 : final_value = 17218.50) :
  ∃ s : ℝ, 
    s = 0.10 ∧ 
    final_value = initial_investment * (1 + first_duration * first_rate) * (1 + second_duration * s) := by
  sorry

end investment_interest_rate_l3058_305881


namespace smallest_transformed_sum_l3058_305822

/-- The number of faces on a standard die -/
def facesOnDie : ℕ := 6

/-- The target sum we want to achieve -/
def targetSum : ℕ := 1994

/-- The function to calculate the transformed sum given the number of dice -/
def transformedSum (n : ℕ) : ℕ := 7 * n - targetSum

/-- The theorem stating the smallest possible value of the transformed sum -/
theorem smallest_transformed_sum :
  ∃ (n : ℕ), 
    (n * facesOnDie ≥ targetSum) ∧ 
    (∀ m : ℕ, m * facesOnDie ≥ targetSum → n ≤ m) ∧
    (transformedSum n = 337) := by
  sorry

end smallest_transformed_sum_l3058_305822


namespace reciprocal_of_negative_three_l3058_305893

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end reciprocal_of_negative_three_l3058_305893


namespace inverse_variation_l3058_305818

/-- Given that a and b vary inversely, prove that when a = 800 and b = 0.5, 
    then b = 0.125 when a = 3200 -/
theorem inverse_variation (a b : ℝ) (h : a * b = 800 * 0.5) :
  3200 * (1 / 8) = 800 * 0.5 := by
  sorry

end inverse_variation_l3058_305818


namespace subsequence_appears_l3058_305871

/-- Defines the sequence where each digit after the first four is the last digit of the sum of the previous four digits -/
def digit_sequence : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| n + 4 => (digit_sequence n + digit_sequence (n + 1) + digit_sequence (n + 2) + digit_sequence (n + 3)) % 10

/-- Checks if the subsequence 8123 appears starting at position n in the sequence -/
def appears_at (n : ℕ) : Prop :=
  digit_sequence n = 8 ∧
  digit_sequence (n + 1) = 1 ∧
  digit_sequence (n + 2) = 2 ∧
  digit_sequence (n + 3) = 3

/-- Theorem stating that the subsequence 8123 appears in the sequence -/
theorem subsequence_appears : ∃ n : ℕ, appears_at n := by
  sorry

end subsequence_appears_l3058_305871


namespace exists_strictly_convex_function_with_constraints_l3058_305817

-- Define the function type
def StrictlyConvexFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    f x₁ + f x₂ < 2 * f ((x₁ + x₂) / 2)

-- Main theorem
theorem exists_strictly_convex_function_with_constraints : 
  ∃ f : ℝ → ℝ,
    (∀ x : ℝ, x > 0 → f x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 4 → ∃ x : ℝ, x > 0 ∧ f x = y) ∧
    StrictlyConvexFunction f :=
sorry

end exists_strictly_convex_function_with_constraints_l3058_305817


namespace parabola_focus_and_directrix_l3058_305865

/-- Given a parabola with equation y² = 8x, prove its focus coordinates and directrix equation -/
theorem parabola_focus_and_directrix :
  ∀ (x y : ℝ), y^2 = 8*x →
  (∃ (focus_x focus_y : ℝ), focus_x = 2 ∧ focus_y = 0) ∧
  (∃ (k : ℝ), k = -2 ∧ ∀ (x : ℝ), x = k → x ∈ {x | x = -2}) :=
by sorry

end parabola_focus_and_directrix_l3058_305865


namespace equation_holds_l3058_305803

theorem equation_holds (n : ℕ+) : 
  (n^2)^2 + n^2 + 1 = (n^2 + n + 1) * ((n-1)^2 + (n-1) + 1) := by
  sorry

#check equation_holds

end equation_holds_l3058_305803


namespace jennifer_spending_l3058_305806

theorem jennifer_spending (total : ℝ) (sandwich_fraction : ℝ) (museum_fraction : ℝ) (book_fraction : ℝ)
  (h_total : total = 150)
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum : museum_fraction = 1/6)
  (h_book : book_fraction = 1/2) :
  total - (sandwich_fraction * total + museum_fraction * total + book_fraction * total) = 20 := by
  sorry

end jennifer_spending_l3058_305806


namespace at_least_one_alarm_probability_l3058_305887

theorem at_least_one_alarm_probability (p_A p_B : ℝ) 
  (h_A : 0 ≤ p_A ∧ p_A ≤ 1) 
  (h_B : 0 ≤ p_B ∧ p_B ≤ 1) : 
  1 - (1 - p_A) * (1 - p_B) = p_A + p_B - p_A * p_B :=
by sorry

end at_least_one_alarm_probability_l3058_305887


namespace sin_square_plus_sin_minus_one_range_l3058_305847

theorem sin_square_plus_sin_minus_one_range :
  ∀ x : ℝ, -5/4 ≤ Real.sin x ^ 2 + Real.sin x - 1 ∧ Real.sin x ^ 2 + Real.sin x - 1 ≤ 1 := by
  sorry

end sin_square_plus_sin_minus_one_range_l3058_305847


namespace taeyeon_height_l3058_305804

theorem taeyeon_height (seonghee_height : ℝ) (taeyeon_ratio : ℝ) :
  seonghee_height = 134.5 →
  taeyeon_ratio = 1.06 →
  taeyeon_ratio * seonghee_height = 142.57 := by
  sorry

end taeyeon_height_l3058_305804


namespace only_setD_cannot_form_triangle_l3058_305826

/-- A set of three line segments that might form a triangle -/
structure TriangleSegments where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three line segments can form a triangle -/
def canFormTriangle (t : TriangleSegments) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The four sets of line segments given in the problem -/
def setA : TriangleSegments := ⟨3, 4, 5⟩
def setB : TriangleSegments := ⟨5, 10, 8⟩
def setC : TriangleSegments := ⟨5, 4.5, 8⟩
def setD : TriangleSegments := ⟨7, 7, 15⟩

/-- Theorem: Among the given sets, only set D cannot form a triangle -/
theorem only_setD_cannot_form_triangle :
  canFormTriangle setA ∧ 
  canFormTriangle setB ∧ 
  canFormTriangle setC ∧ 
  ¬canFormTriangle setD := by
  sorry

end only_setD_cannot_form_triangle_l3058_305826


namespace correct_calculation_l3058_305842

theorem correct_calculation (a : ℝ) : 2 * a^4 * 3 * a^5 = 6 * a^9 := by
  sorry

end correct_calculation_l3058_305842


namespace integer_solution_for_inequalities_l3058_305898

theorem integer_solution_for_inequalities : 
  ∃! (n : ℤ), n + 15 > 16 ∧ -3 * n^2 > -27 :=
by
  -- Proof goes here
  sorry

end integer_solution_for_inequalities_l3058_305898


namespace money_division_l3058_305883

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 2400 →
  r - q = 3000 :=
by sorry

end money_division_l3058_305883


namespace max_tiles_on_floor_l3058_305876

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit in one direction -/
def tilesInOneDimension (floorSize tileSize : ℕ) : ℕ :=
  floorSize / tileSize

/-- Calculates the total number of tiles for a given orientation -/
def totalTiles (floor tile : Dimensions) : ℕ :=
  (tilesInOneDimension floor.length tile.length) * (tilesInOneDimension floor.width tile.width)

/-- Theorem stating the maximum number of tiles that can be accommodated -/
theorem max_tiles_on_floor (floor : Dimensions) (tile : Dimensions) 
    (h_floor : floor = ⟨1000, 210⟩) (h_tile : tile = ⟨35, 30⟩) :
  max (totalTiles floor tile) (totalTiles floor ⟨tile.width, tile.length⟩) = 198 := by
  sorry

end max_tiles_on_floor_l3058_305876


namespace tangent_line_correct_l3058_305813

-- Define the curve
def curve (x : ℝ) : ℝ := -x^2 + 6*x

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := -2*x + 6

-- Define the proposed tangent line
def tangent_line (x : ℝ) : ℝ := 6*x

theorem tangent_line_correct :
  -- The tangent line passes through the origin
  tangent_line 0 = 0 ∧
  -- The tangent line touches the curve at some point
  ∃ x : ℝ, curve x = tangent_line x ∧
  -- The slope of the tangent line equals the derivative of the curve at the point of tangency
  curve_derivative x = 6 := by
  sorry

end tangent_line_correct_l3058_305813


namespace trigonometric_identities_l3058_305861

theorem trigonometric_identities :
  (2 * Real.sin (30 * π / 180) - Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1 / 2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + 
   Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) := by
  sorry

end trigonometric_identities_l3058_305861


namespace like_terms_exponent_equality_l3058_305885

theorem like_terms_exponent_equality (a b : ℝ) (m : ℝ) : 
  (∃ k : ℝ, -2 * a^(2-m) * b^3 = k * (-2 * a^(4-3*m) * b^3)) → m = 1 := by
sorry

end like_terms_exponent_equality_l3058_305885


namespace variations_difference_l3058_305899

theorem variations_difference (n : ℕ) : n ^ 3 = n * (n - 1) * (n - 2) + 225 ↔ n = 9 := by
  sorry

end variations_difference_l3058_305899
