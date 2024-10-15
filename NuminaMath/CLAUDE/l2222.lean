import Mathlib

namespace NUMINAMATH_CALUDE_marks_songs_per_gig_l2222_222207

/-- Represents the number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Represents the number of gigs Mark does in two weeks -/
def number_of_gigs : ℕ := days_in_two_weeks / 2

/-- Represents the duration of a short song in minutes -/
def short_song_duration : ℕ := 5

/-- Represents the duration of a long song in minutes -/
def long_song_duration : ℕ := 2 * short_song_duration

/-- Represents the number of short songs per gig -/
def short_songs_per_gig : ℕ := 2

/-- Represents the number of long songs per gig -/
def long_songs_per_gig : ℕ := 1

/-- Represents the total playing time for all gigs in minutes -/
def total_playing_time : ℕ := 280

/-- Theorem: Given the conditions, Mark plays 7 songs at each gig -/
theorem marks_songs_per_gig :
  ∃ (songs_per_gig : ℕ),
    songs_per_gig = short_songs_per_gig + long_songs_per_gig +
      ((total_playing_time / number_of_gigs) -
       (short_songs_per_gig * short_song_duration + long_songs_per_gig * long_song_duration)) /
      short_song_duration ∧
    songs_per_gig = 7 :=
by sorry

end NUMINAMATH_CALUDE_marks_songs_per_gig_l2222_222207


namespace NUMINAMATH_CALUDE_show_length_ratio_l2222_222228

theorem show_length_ratio (first_show_length second_show_length total_time : ℕ) 
  (h1 : first_show_length = 30)
  (h2 : total_time = 150)
  (h3 : second_show_length = total_time - first_show_length) :
  second_show_length / first_show_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_show_length_ratio_l2222_222228


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l2222_222283

-- Define the repeating decimals
def repeating_decimal_0_8 : ℚ := 8/9
def repeating_decimal_2_4 : ℚ := 22/9

-- State the theorem
theorem repeating_decimal_fraction :
  repeating_decimal_0_8 / repeating_decimal_2_4 = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l2222_222283


namespace NUMINAMATH_CALUDE_days_worked_by_c_l2222_222212

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 6

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 100

/-- Represents the total earnings of all three people -/
def total_earnings : ℕ := 1480

/-- Represents the ratio of daily wages for a, b, and c -/
def wage_ratio : Fin 3 → ℕ 
  | 0 => 3
  | 1 => 4
  | 2 => 5

/-- 
Proves that given the conditions, the number of days worked by person c is 4
-/
theorem days_worked_by_c : 
  ∃ (days_c : ℕ), 
    days_c * wage_c + 
    days_a * (wage_ratio 0 * wage_c / wage_ratio 2) + 
    days_b * (wage_ratio 1 * wage_c / wage_ratio 2) = 
    total_earnings ∧ days_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_by_c_l2222_222212


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l2222_222294

/-- An arithmetic sequence with the property that removing one term results in a geometric sequence -/
def ArithmeticSequenceWithGeometricSubsequence (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  n ≥ 4 ∧ 
  d ≠ 0 ∧ 
  (∀ i, a i ≠ 0) ∧
  (∀ i, i < n → a (i + 1) = a i + d) ∧
  ∃ k, k < n ∧ 
    (∀ i j, i < j ∧ j < n ∧ i ≠ k ∧ j ≠ k → 
      (a j)^2 = a i * a (if j < k then j + 1 else j))

theorem arithmetic_sequence_with_geometric_subsequence 
  (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : 
  ArithmeticSequenceWithGeometricSubsequence n a d → 
  n = 4 ∧ (a 1 / d = -4 ∨ a 1 / d = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_subsequence_l2222_222294


namespace NUMINAMATH_CALUDE_shirts_count_l2222_222287

/-- Given a ratio of pants : shorts : shirts and the number of pants, 
    calculate the number of shirts -/
def calculate_shirts (pants_ratio : ℕ) (shorts_ratio : ℕ) (shirts_ratio : ℕ) 
                     (num_pants : ℕ) : ℕ :=
  (num_pants / pants_ratio) * shirts_ratio

/-- Prove that given the ratio 7 : 7 : 10 for pants : shorts : shirts, 
    and 14 pants, there are 20 shirts -/
theorem shirts_count : calculate_shirts 7 7 10 14 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shirts_count_l2222_222287


namespace NUMINAMATH_CALUDE_solve_colored_copies_l2222_222227

def colored_copies_problem (colored_cost white_cost : ℚ) (total_copies : ℕ) (total_cost : ℚ) : Prop :=
  ∃ (colored_copies : ℕ),
    colored_copies ≤ total_copies ∧
    colored_cost * colored_copies + white_cost * (total_copies - colored_copies) = total_cost ∧
    colored_copies = 50

theorem solve_colored_copies :
  colored_copies_problem (10/100) (5/100) 400 (45/2) :=
sorry

end NUMINAMATH_CALUDE_solve_colored_copies_l2222_222227


namespace NUMINAMATH_CALUDE_layla_and_alan_apples_l2222_222218

def maggie_apples : ℕ := 40
def kelsey_apples : ℕ := 28
def total_people : ℕ := 4
def average_apples : ℕ := 30

theorem layla_and_alan_apples :
  ∃ (layla_apples alan_apples : ℕ),
    maggie_apples + kelsey_apples + layla_apples + alan_apples = total_people * average_apples ∧
    layla_apples + alan_apples = 52 :=
by sorry

end NUMINAMATH_CALUDE_layla_and_alan_apples_l2222_222218


namespace NUMINAMATH_CALUDE_sixth_equation_pattern_l2222_222262

/-- The sum of n consecutive odd numbers starting from a given odd number -/
def sum_consecutive_odds (start : ℕ) (n : ℕ) : ℕ :=
  (start + n - 1) * n

/-- The nth cube -/
def cube (n : ℕ) : ℕ := n^3

theorem sixth_equation_pattern : sum_consecutive_odds 31 6 = cube 6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_equation_pattern_l2222_222262


namespace NUMINAMATH_CALUDE_final_flow_rate_l2222_222289

/-- Represents the flow rate of cleaner through a pipe at different time intervals --/
structure FlowRate :=
  (initial : ℝ)
  (after15min : ℝ)
  (final : ℝ)

/-- Theorem stating that given the initial conditions and total cleaner used, 
    the final flow rate must be 4 ounces per minute --/
theorem final_flow_rate 
  (flow : FlowRate)
  (total_time : ℝ)
  (total_cleaner : ℝ)
  (h1 : flow.initial = 2)
  (h2 : flow.after15min = 3)
  (h3 : total_time = 30)
  (h4 : total_cleaner = 80)
  : flow.final = 4 := by
  sorry

#check final_flow_rate

end NUMINAMATH_CALUDE_final_flow_rate_l2222_222289


namespace NUMINAMATH_CALUDE_r_profit_share_is_one_third_of_total_l2222_222298

/-- Represents the capital and investment duration of an investor -/
structure Investor where
  capital : ℝ
  duration : ℝ

/-- Calculates the profit share of an investor -/
def profitShare (i : Investor) : ℝ := i.capital * i.duration

/-- Theorem: Given the conditions, r's share of the total profit is one-third of the total profit -/
theorem r_profit_share_is_one_third_of_total
  (p q r : Investor)
  (h1 : 4 * p.capital = 6 * q.capital)
  (h2 : 6 * q.capital = 10 * r.capital)
  (h3 : p.duration = 2)
  (h4 : q.duration = 3)
  (h5 : r.duration = 5)
  (total_profit : ℝ)
  : profitShare r = total_profit / 3 := by
  sorry

#check r_profit_share_is_one_third_of_total

end NUMINAMATH_CALUDE_r_profit_share_is_one_third_of_total_l2222_222298


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2222_222204

/-- Simple interest calculation -/
theorem simple_interest_principal (interest rate time principal : ℝ) :
  interest = principal * (rate / 100) * time →
  rate = 6.666666666666667 →
  time = 4 →
  interest = 160 →
  principal = 600 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2222_222204


namespace NUMINAMATH_CALUDE_geometric_sequence_product_equality_l2222_222242

/-- Given four non-zero real numbers, prove that forming a geometric sequence
    is sufficient but not necessary for their product equality. -/
theorem geometric_sequence_product_equality (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → a * d = b * c ∧
  ∃ a' b' c' d' : ℝ, a' * d' = b' * c' ∧ ¬(∃ r : ℝ, b' = a' * r ∧ c' = b' * r ∧ d' = c' * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_equality_l2222_222242


namespace NUMINAMATH_CALUDE_calculate_annual_interest_rate_l2222_222208

/-- Given an initial charge and the amount owed after one year with simple annual interest,
    calculate the annual interest rate. -/
theorem calculate_annual_interest_rate
  (initial_charge : ℝ)
  (amount_owed_after_year : ℝ)
  (h1 : initial_charge = 35)
  (h2 : amount_owed_after_year = 37.1)
  (h3 : amount_owed_after_year = initial_charge * (1 + interest_rate))
  : interest_rate = 0.06 :=
sorry

end NUMINAMATH_CALUDE_calculate_annual_interest_rate_l2222_222208


namespace NUMINAMATH_CALUDE_smallest_c_in_special_progression_l2222_222231

theorem smallest_c_in_special_progression (a b c : ℕ) : 
  a > b ∧ b > c ∧ c > 0 →  -- a, b, c are positive integers with a > b > c
  (b * b = a * c) →        -- a, b, c form a geometric progression
  (a + b = 2 * c) →        -- a, c, b form an arithmetic progression
  c ≥ 1 ∧                  -- c is at least 1
  (∀ k : ℕ, k > 0 ∧ k < c →
    ¬∃ x y : ℕ, x > y ∧ y > k ∧ 
    (y * y = x * k) ∧ 
    (x + y = 2 * k)) →     -- c is the smallest value satisfying the conditions
  c = 1                    -- The smallest possible value of c is 1
:= by sorry

end NUMINAMATH_CALUDE_smallest_c_in_special_progression_l2222_222231


namespace NUMINAMATH_CALUDE_expected_rounds_range_l2222_222284

/-- Represents the game between players A and B -/
structure Game where
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The expected number of rounds in the game -/
noncomputable def expected_rounds (g : Game) : ℝ :=
  2 * (1 - (2 * g.p * (1 - g.p))^10) / (1 - 2 * g.p * (1 - g.p))

/-- Theorem stating the range of the expected number of rounds -/
theorem expected_rounds_range (g : Game) :
  2 < expected_rounds g ∧ expected_rounds g ≤ 4 - (1/2)^8 :=
sorry

end NUMINAMATH_CALUDE_expected_rounds_range_l2222_222284


namespace NUMINAMATH_CALUDE_max_queens_2017_l2222_222297

/-- Represents a chessboard of size n x n -/
def Chessboard (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a queen at position (x, y) attacks at most one other queen -/
def attacks_at_most_one (board : Chessboard 2017) (x y : Fin 2017) : Prop :=
  ∃! (x' y' : Fin 2017), x' ≠ x ∨ y' ≠ y ∧ board x' y' = true ∧
    (x' = x ∨ y' = y ∨ (x' : ℤ) - (x : ℤ) = (y' : ℤ) - (y : ℤ) ∨ 
     (x' : ℤ) - (x : ℤ) = (y : ℤ) - (y' : ℤ))

/-- The property that each queen on the board attacks at most one other queen -/
def valid_placement (board : Chessboard 2017) : Prop :=
  ∀ x y, board x y = true → attacks_at_most_one board x y

/-- Counts the number of queens on the board -/
def count_queens (board : Chessboard 2017) : ℕ :=
  (Finset.univ.filter (λ x : Fin 2017 × Fin 2017 => board x.1 x.2 = true)).card

/-- The main theorem: there exists a valid placement with 673359 queens -/
theorem max_queens_2017 : 
  ∃ (board : Chessboard 2017), valid_placement board ∧ count_queens board = 673359 :=
sorry

end NUMINAMATH_CALUDE_max_queens_2017_l2222_222297


namespace NUMINAMATH_CALUDE_peter_stamps_l2222_222246

theorem peter_stamps (M : ℕ) : 
  M > 1 ∧ 
  M % 5 = 2 ∧ 
  M % 11 = 2 ∧ 
  M % 13 = 2 → 
  (∀ n : ℕ, n > 1 ∧ n % 5 = 2 ∧ n % 11 = 2 ∧ n % 13 = 2 → n ≥ M) → 
  M = 717 := by
sorry

end NUMINAMATH_CALUDE_peter_stamps_l2222_222246


namespace NUMINAMATH_CALUDE_equation_solution_l2222_222268

theorem equation_solution :
  ∀ (A B C : ℕ),
    3 * A - A = 10 →
    B + A = 12 →
    C - B = 6 →
    A ≠ B →
    B ≠ C →
    A ≠ C →
    C = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2222_222268


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2222_222269

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 108 → 
  (n : ℝ) * (180 - interior_angle) = 360 → 
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2222_222269


namespace NUMINAMATH_CALUDE_sum_digits_count_numeric_hex_below_2000_l2222_222245

/-- Converts a decimal number to hexadecimal --/
def decimalToHex (n : ℕ) : String := sorry

/-- Counts positive hexadecimal numbers below a given hexadecimal number
    that contain only numeric digits (0-9) --/
def countNumericHex (hex : String) : ℕ := sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the digits of the count of positive hexadecimal numbers
    below the hexadecimal representation of 2000 that contain only numeric digits (0-9) is 25 --/
theorem sum_digits_count_numeric_hex_below_2000 :
  sumDigits (countNumericHex (decimalToHex 2000)) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_digits_count_numeric_hex_below_2000_l2222_222245


namespace NUMINAMATH_CALUDE_min_hours_to_drive_l2222_222225

/-- The legal blood alcohol content (BAC) limit for driving -/
def legal_bac_limit : ℝ := 0.2

/-- The initial BAC after drinking -/
def initial_bac : ℝ := 0.8

/-- The rate at which BAC decreases per hour -/
def bac_decrease_rate : ℝ := 0.5

/-- The minimum number of hours to wait before driving -/
def min_wait_hours : ℕ := 2

/-- Theorem stating the minimum number of hours to wait before driving -/
theorem min_hours_to_drive :
  (initial_bac * (1 - bac_decrease_rate) ^ min_wait_hours ≤ legal_bac_limit) ∧
  (∀ h : ℕ, h < min_wait_hours → initial_bac * (1 - bac_decrease_rate) ^ h > legal_bac_limit) :=
sorry

end NUMINAMATH_CALUDE_min_hours_to_drive_l2222_222225


namespace NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l2222_222248

/-- Represents a number in base 9 of the form ab4d -/
structure Base9Number where
  a : Nat
  b : Nat
  d : Nat
  a_nonzero : a ≠ 0

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : Nat :=
  729 * n.a + 81 * n.b + 36 + n.d

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

theorem base9_perfect_square_last_digit 
  (n : Base9Number) 
  (h : isPerfectSquare (toDecimal n)) : 
  n.d = 0 ∨ n.d = 1 ∨ n.d = 4 ∨ n.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l2222_222248


namespace NUMINAMATH_CALUDE_cartesian_to_polar_l2222_222299

theorem cartesian_to_polar :
  let x : ℝ := -2
  let y : ℝ := 2 * Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * π / 3
  ρ = 4 ∧ Real.cos θ = x / ρ ∧ Real.sin θ = y / ρ := by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_l2222_222299


namespace NUMINAMATH_CALUDE_price_A_base_correct_min_A_bundles_correct_l2222_222280

-- Define the price of type A seedlings at the base
def price_A_base : ℝ := 20

-- Define the price of type B seedlings at the base
def price_B_base : ℝ := 30

-- Define the total number of bundles to purchase
def total_bundles : ℕ := 100

-- Define the maximum spending limit
def max_spending : ℝ := 2400

-- Theorem for the price of type A seedlings at the base
theorem price_A_base_correct :
  ∃ (x : ℝ), x > 0 ∧ 300 / x - 300 / (1.5 * x) = 5 ∧ x = price_A_base :=
sorry

-- Theorem for the minimum number of type A seedlings to purchase
theorem min_A_bundles_correct :
  ∃ (m : ℕ), m ≥ 60 ∧
    ∀ (n : ℕ), n < m →
      price_A_base * n + price_B_base * (total_bundles - n) > max_spending :=
sorry

end NUMINAMATH_CALUDE_price_A_base_correct_min_A_bundles_correct_l2222_222280


namespace NUMINAMATH_CALUDE_no_linear_term_implies_sum_zero_l2222_222234

theorem no_linear_term_implies_sum_zero (a b : ℝ) :
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + a*b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_sum_zero_l2222_222234


namespace NUMINAMATH_CALUDE_johns_number_l2222_222230

theorem johns_number : ∃ x : ℝ, (2 * (3 * x - 6) + 20 = 122) ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_l2222_222230


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2222_222221

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 54 →
  time_pole = 18 →
  ∃ platform_length : ℝ, abs (platform_length - 600.18) < 0.01 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l2222_222221


namespace NUMINAMATH_CALUDE_min_value_cubic_fraction_l2222_222293

theorem min_value_cubic_fraction (x : ℝ) (h : x > 9) :
  x^3 / (x - 9) ≥ 325 ∧ ∃ y > 9, y^3 / (y - 9) = 325 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cubic_fraction_l2222_222293


namespace NUMINAMATH_CALUDE_salary_difference_l2222_222292

def initial_salary : ℝ := 30000
def hansel_raise_percent : ℝ := 0.10
def gretel_raise_percent : ℝ := 0.15

def hansel_new_salary : ℝ := initial_salary * (1 + hansel_raise_percent)
def gretel_new_salary : ℝ := initial_salary * (1 + gretel_raise_percent)

theorem salary_difference :
  gretel_new_salary - hansel_new_salary = 1500 := by
  sorry

end NUMINAMATH_CALUDE_salary_difference_l2222_222292


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_three_sum_l2222_222210

theorem consecutive_integers_sqrt_three_sum (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_three_sum_l2222_222210


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2222_222288

theorem solution_set_inequality (x : ℝ) :
  (abs x - 2) * (x - 1) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 1 ∨ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2222_222288


namespace NUMINAMATH_CALUDE_rug_selling_price_l2222_222282

/-- Proves that the selling price per rug is $60, given the cost price, number of rugs, and total profit --/
theorem rug_selling_price 
  (cost_price : ℝ) 
  (num_rugs : ℕ) 
  (total_profit : ℝ) 
  (h1 : cost_price = 40) 
  (h2 : num_rugs = 20) 
  (h3 : total_profit = 400) : 
  (cost_price * num_rugs + total_profit) / num_rugs = 60 := by
  sorry

end NUMINAMATH_CALUDE_rug_selling_price_l2222_222282


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2222_222247

theorem vector_sum_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x + 2, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2222_222247


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2222_222279

/-- The universal set U -/
def U : Set ℕ := {1, 2, 3, 4, 5}

/-- Set A -/
def A : Set ℕ := {1, 3, 4}

/-- Set B -/
def B : Set ℕ := {4, 5}

/-- Theorem stating that the intersection of A and the complement of B with respect to U is {1, 3} -/
theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2222_222279


namespace NUMINAMATH_CALUDE_quadratic_radicals_simplification_l2222_222217

theorem quadratic_radicals_simplification :
  (∀ a b m n : ℝ, a > 0 ∧ b > 0 ∧ m > 0 ∧ n > 0 →
    m^2 + n^2 = a ∧ m * n = Real.sqrt b →
    Real.sqrt (a + 2 * Real.sqrt b) = m + n) ∧
  Real.sqrt (6 + 2 * Real.sqrt 5) = Real.sqrt 5 + 1 ∧
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 ∧
  (∀ a : ℝ, Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5 →
    a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radicals_simplification_l2222_222217


namespace NUMINAMATH_CALUDE_roots_triangle_condition_l2222_222219

/-- A cubic equation with coefficients p, q, and r -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic equation form a triangle -/
def roots_form_triangle (eq : CubicEquation) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧
    u^3 + eq.p * u^2 + eq.q * u + eq.r = 0 ∧
    v^3 + eq.p * v^2 + eq.q * v + eq.r = 0 ∧
    w^3 + eq.p * w^2 + eq.q * w + eq.r = 0 ∧
    u + v > w ∧ u + w > v ∧ v + w > u

/-- The theorem stating the condition for roots to form a triangle -/
theorem roots_triangle_condition (eq : CubicEquation) :
  roots_form_triangle eq ↔ eq.p^3 - 4 * eq.p * eq.q + 8 * eq.r > 0 :=
sorry

end NUMINAMATH_CALUDE_roots_triangle_condition_l2222_222219


namespace NUMINAMATH_CALUDE_chord_length_l2222_222259

theorem chord_length (R : ℝ) (AB AC : ℝ) (h1 : R = 8) (h2 : AB = 10) 
  (h3 : AC = (2 * Real.pi * R) / 3) : 
  (AC : ℝ) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2222_222259


namespace NUMINAMATH_CALUDE_perimeter_of_square_C_l2222_222238

/-- Given squares A, B, and C with specific perimeter relationships, 
    prove that the perimeter of C is 100. -/
theorem perimeter_of_square_C (A B C : ℝ) : 
  (A > 0) →  -- A is positive (side length of a square)
  (B > 0) →  -- B is positive (side length of a square)
  (C > 0) →  -- C is positive (side length of a square)
  (4 * A = 20) →  -- Perimeter of A is 20
  (4 * B = 40) →  -- Perimeter of B is 40
  (C = A + 2 * B) →  -- Side length of C relationship
  (4 * C = 100) :=  -- Perimeter of C is 100
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_C_l2222_222238


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2222_222275

/-- Given a geometric sequence {a_n} with a_1 = 8 and a_4 = 64, the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 8 →                     -- first term condition
  a 4 = 64 →                    -- fourth term condition
  q = 2 :=                      -- conclusion: common ratio is 2
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2222_222275


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l2222_222254

/-- Fixed point F₁ -/
def F₁ : ℝ × ℝ := (-4, 0)

/-- Fixed point F₂ -/
def F₂ : ℝ × ℝ := (4, 0)

/-- The set of points M satisfying the condition |MF₁| + |MF₂| = 8 -/
def trajectory : Set (ℝ × ℝ) :=
  {M : ℝ × ℝ | dist M F₁ + dist M F₂ = 8}

/-- Theorem stating that the trajectory is a line segment -/
theorem trajectory_is_line_segment :
  ∃ (A B : ℝ × ℝ), trajectory = {M : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l2222_222254


namespace NUMINAMATH_CALUDE_erica_earnings_l2222_222260

/-- The amount of money earned per kilogram of fish -/
def price_per_kg : ℝ := 20

/-- The amount of fish caught in the past four months in kilograms -/
def past_four_months_catch : ℝ := 80

/-- The amount of fish caught today in kilograms -/
def today_catch : ℝ := 2 * past_four_months_catch

/-- The total amount of fish caught in kilograms -/
def total_catch : ℝ := past_four_months_catch + today_catch

/-- Erica's total earnings for the past four months including today -/
def total_earnings : ℝ := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end NUMINAMATH_CALUDE_erica_earnings_l2222_222260


namespace NUMINAMATH_CALUDE_problem_statement_l2222_222233

theorem problem_statement : (5/12 : ℝ)^2022 * (-2.4)^2023 = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2222_222233


namespace NUMINAMATH_CALUDE_movie_marathon_duration_l2222_222216

theorem movie_marathon_duration :
  let movie1 : ℝ := 2
  let movie2 : ℝ := movie1 * 1.5
  let movie3 : ℝ := movie1 + movie2 - 1
  movie1 + movie2 + movie3 = 9 := by sorry

end NUMINAMATH_CALUDE_movie_marathon_duration_l2222_222216


namespace NUMINAMATH_CALUDE_geometric_number_difference_l2222_222258

/-- A function that checks if a 3-digit number is geometric --/
def is_geometric (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    r > 0 ∧
    b = (a : ℚ) * r ∧
    c = (b : ℚ) * r

/-- A function that checks if a number starts with an even digit --/
def starts_with_even (n : ℕ) : Prop :=
  ∃ (a : ℕ), n = 100 * a + n % 100 ∧ Even a

/-- The theorem to be proved --/
theorem geometric_number_difference :
  ∃ (max min : ℕ),
    is_geometric max ∧
    is_geometric min ∧
    starts_with_even max ∧
    starts_with_even min ∧
    (∀ n, is_geometric n ∧ starts_with_even n → n ≤ max) ∧
    (∀ n, is_geometric n ∧ starts_with_even n → n ≥ min) ∧
    max - min = 403 :=
sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l2222_222258


namespace NUMINAMATH_CALUDE_tile_arrangements_l2222_222274

/-- The number of distinguishable arrangements of tiles -/
def distinguishable_arrangements (red blue green yellow : ℕ) : ℕ :=
  Nat.factorial (red + blue + green + yellow) /
  (Nat.factorial red * Nat.factorial blue * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 red tile, 2 blue tiles, 2 green tiles, and 4 yellow tiles is 3780 -/
theorem tile_arrangements :
  distinguishable_arrangements 1 2 2 4 = 3780 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l2222_222274


namespace NUMINAMATH_CALUDE_disjoint_circles_condition_l2222_222264

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4
def circle2 (x y a : ℝ) : Prop := x^2 + (y - a)^2 = 1

def circles_disjoint (a : ℝ) : Prop :=
  ∀ x y, ¬(circle1 x y ∧ circle2 x y a)

theorem disjoint_circles_condition (a : ℝ) :
  circles_disjoint a ↔ (a > 1 + 2 * Real.sqrt 2 ∨ a < 1 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_disjoint_circles_condition_l2222_222264


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2222_222200

theorem solution_set_inequality (x : ℝ) : (x - 2) / x < 0 ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2222_222200


namespace NUMINAMATH_CALUDE_remainder_problem_l2222_222222

theorem remainder_problem (N : ℤ) : 
  N % 37 = 1 → N % 296 = 260 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2222_222222


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2222_222244

theorem nested_fraction_equality : 1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2222_222244


namespace NUMINAMATH_CALUDE_line_xz_plane_intersection_l2222_222214

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_xz_plane_intersection (p₁ p₂ q : ℝ × ℝ × ℝ) : 
  p₁ = (2, 3, 5) → 
  p₂ = (4, 0, 9) → 
  (∃ t : ℝ, q = p₁ + t • (p₂ - p₁)) → 
  q.2 = 0 → 
  q = (4, 0, 9) := by
  sorry

#check line_xz_plane_intersection

end NUMINAMATH_CALUDE_line_xz_plane_intersection_l2222_222214


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l2222_222224

theorem smallest_m_no_real_roots : 
  let equation (m x : ℝ) := 3 * x * ((m + 1) * x - 5) - x^2 + 8
  ∀ m : ℤ, (∀ x : ℝ, equation m x ≠ 0) → m ≥ 2 ∧ 
  ∃ m' : ℤ, m' < 2 ∧ ∃ x : ℝ, equation (m' : ℝ) x = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l2222_222224


namespace NUMINAMATH_CALUDE_complex_point_location_l2222_222257

theorem complex_point_location (z : ℂ) (h : z = Complex.I * 2) : 
  z.re = 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l2222_222257


namespace NUMINAMATH_CALUDE_boat_distance_downstream_l2222_222209

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : time = 4) : 
  boat_speed + stream_speed * time = 112 := by
  sorry


end NUMINAMATH_CALUDE_boat_distance_downstream_l2222_222209


namespace NUMINAMATH_CALUDE_star_eight_ten_l2222_222250

/-- Custom operation * for rational numbers -/
def star (m n p : ℚ) (x y : ℚ) : ℚ := m * x + n * y + p

/-- Theorem stating that if 3 * 5 = 30 and 4 * 6 = 425, then 8 * 10 = 2005 -/
theorem star_eight_ten (m n p : ℚ) 
  (h1 : star m n p 3 5 = 30)
  (h2 : star m n p 4 6 = 425) : 
  star m n p 8 10 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_star_eight_ten_l2222_222250


namespace NUMINAMATH_CALUDE_christine_savings_theorem_l2222_222220

/-- Calculates Christine's savings for the month based on her sales and commission structure -/
def christine_savings (
  electronics_rate : ℚ)
  (clothing_rate : ℚ)
  (furniture_rate : ℚ)
  (domestic_electronics : ℚ)
  (domestic_clothing : ℚ)
  (domestic_furniture : ℚ)
  (international_electronics : ℚ)
  (international_clothing : ℚ)
  (international_furniture : ℚ)
  (exchange_rate : ℚ)
  (tax_rate : ℚ)
  (personal_needs_rate : ℚ)
  (investment_rate : ℚ) : ℚ :=
  let domestic_commission := 
    electronics_rate * domestic_electronics +
    clothing_rate * domestic_clothing +
    furniture_rate * domestic_furniture
  let international_commission := 
    (electronics_rate * international_electronics +
    clothing_rate * international_clothing +
    furniture_rate * international_furniture) * exchange_rate
  let tax := international_commission * tax_rate
  let post_tax_international := international_commission - tax
  let international_savings := 
    post_tax_international * (1 - personal_needs_rate - investment_rate)
  domestic_commission + international_savings

theorem christine_savings_theorem :
  christine_savings 0.15 0.10 0.20 12000 8000 4000 5000 3000 2000 1.10 0.25 0.55 0.30 = 3579.4375 := by
  sorry

#eval christine_savings 0.15 0.10 0.20 12000 8000 4000 5000 3000 2000 1.10 0.25 0.55 0.30

end NUMINAMATH_CALUDE_christine_savings_theorem_l2222_222220


namespace NUMINAMATH_CALUDE_prime_quadruple_theorem_l2222_222270

def is_valid_quadruple (p₁ p₂ p₃ p₄ : Nat) : Prop :=
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
  p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
  p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882

theorem prime_quadruple_theorem :
  ∀ p₁ p₂ p₃ p₄ : Nat,
  is_valid_quadruple p₁ p₂ p₃ p₄ ↔
  ((p₁, p₂, p₃, p₄) = (2, 5, 19, 37) ∨
   (p₁, p₂, p₃, p₄) = (2, 11, 19, 31) ∨
   (p₁, p₂, p₃, p₄) = (2, 13, 19, 29)) :=
by sorry

end NUMINAMATH_CALUDE_prime_quadruple_theorem_l2222_222270


namespace NUMINAMATH_CALUDE_bugs_meeting_time_l2222_222267

/-- The time for two bugs to meet again at the starting point on two tangent circles -/
theorem bugs_meeting_time (r1 r2 v1 v2 : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) 
  (hv1 : v1 = 3 * Real.pi) (hv2 : v2 = 4 * Real.pi) : 
  ∃ t : ℝ, t = 48 ∧ 
  (∃ n1 n2 : ℕ, t * v1 = n1 * (2 * Real.pi * r1) ∧ 
               t * v2 = n2 * (2 * Real.pi * r2)) := by
  sorry

#check bugs_meeting_time

end NUMINAMATH_CALUDE_bugs_meeting_time_l2222_222267


namespace NUMINAMATH_CALUDE_cosine_equation_solvability_l2222_222290

theorem cosine_equation_solvability (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y - Real.cos y ^ 2 + m - 3 = 0) ↔ 0 ≤ m ∧ m ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equation_solvability_l2222_222290


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l2222_222295

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (m : Line) (α β : Plane) :
  perpendicular m β → parallel m α → perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l2222_222295


namespace NUMINAMATH_CALUDE_fertilizer_transport_l2222_222276

theorem fertilizer_transport (x y t : ℕ) : 
  (x * t = (x - 4) * (t + 10)) →
  (y * t = (y - 3) * (t + 10)) →
  (x * t - y * t = 60) →
  (x - 4 = 8) ∧ (y - 3 = 6) ∧ (t + 10 = 30) :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_transport_l2222_222276


namespace NUMINAMATH_CALUDE_horner_v4_value_l2222_222252

def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc c => horner_step acc c x) 0

theorem horner_v4_value :
  let coeffs := [1, 0, 1, 0, 0, -2, 3]
  let x := 2
  let v0 := 3
  let v1 := horner_step v0 (-2) x
  let v2 := horner_step v1 0 x
  let v3 := horner_step v2 1 x
  let v4 := horner_step v3 0 x
  v4 = 34 ∧ horner_method coeffs x = f x := by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l2222_222252


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l2222_222235

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, (40 * 30 + (12 + x) * 3) / 5 = 1212 := by sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l2222_222235


namespace NUMINAMATH_CALUDE_octahedron_non_blue_probability_l2222_222256

structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (green_faces : ℕ)

def non_blue_probability (o : Octahedron) : ℚ :=
  (o.red_faces + o.green_faces : ℚ) / o.total_faces

theorem octahedron_non_blue_probability :
  ∀ o : Octahedron,
  o.total_faces = 8 →
  o.blue_faces = 3 →
  o.red_faces = 3 →
  o.green_faces = 2 →
  non_blue_probability o = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_non_blue_probability_l2222_222256


namespace NUMINAMATH_CALUDE_max_dot_product_l2222_222266

-- Define the hyperbola
def hyperbola (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) - (P.2^2 / 9) = 1

-- Define point A in terms of P and t
def point_A (P : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * P.1, t * P.2)

-- Define the dot product condition
def dot_product_condition (P : ℝ × ℝ) (t : ℝ) : Prop :=
  (point_A P t).1 * P.1 + (point_A P t).2 * P.2 = 64

-- Define point B
def B : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem max_dot_product 
  (P : ℝ × ℝ) 
  (t : ℝ) 
  (h1 : hyperbola P) 
  (h2 : dot_product_condition P t) :
  ∃ (M : ℝ), M = 24/5 ∧ ∀ (A : ℝ × ℝ), A = point_A P t → |B.1 * A.1 + B.2 * A.2| ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l2222_222266


namespace NUMINAMATH_CALUDE_negation_equivalence_l2222_222215

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2222_222215


namespace NUMINAMATH_CALUDE_mixture_weight_l2222_222240

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (brand_a_weight : ℝ) (brand_b_weight : ℝ) 
  (mix_ratio_a : ℝ) (mix_ratio_b : ℝ) (total_volume : ℝ) : 
  brand_a_weight = 900 →
  brand_b_weight = 750 →
  mix_ratio_a = 3 →
  mix_ratio_b = 2 →
  total_volume = 4 →
  (mix_ratio_a * total_volume * brand_a_weight + mix_ratio_b * total_volume * brand_b_weight) / 
  ((mix_ratio_a + mix_ratio_b) * 1000) = 3.36 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l2222_222240


namespace NUMINAMATH_CALUDE_no_roots_for_equation_l2222_222285

theorem no_roots_for_equation : ∀ x : ℝ, ¬(Real.sqrt (7 - x) = x * Real.sqrt (7 - x) - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_roots_for_equation_l2222_222285


namespace NUMINAMATH_CALUDE_simplify_expressions_l2222_222277

theorem simplify_expressions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (Real.sqrt (2 / 3) + 3 * Real.sqrt (1 / 6) - (1 / 2) * Real.sqrt 54 = -(2 * Real.sqrt 6) / 3) ∧
    (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2222_222277


namespace NUMINAMATH_CALUDE_week_cycling_distance_l2222_222241

/-- Represents the cycling data for a single day -/
structure DailyRide where
  base_distance : Float
  speed_bonus : Float

/-- Calculates the effective distance for a single day -/
def effective_distance (ride : DailyRide) : Float :=
  ride.base_distance * (1 + ride.speed_bonus)

/-- Calculates the total effective distance for the week -/
def total_effective_distance (rides : List DailyRide) : Float :=
  rides.map effective_distance |> List.sum

/-- The main theorem: proves that the total effective distance is 367.05 km -/
theorem week_cycling_distance : 
  let monday : DailyRide := { base_distance := 40, speed_bonus := 0.05 }
  let tuesday : DailyRide := { base_distance := 50, speed_bonus := 0.03 }
  let wednesday : DailyRide := { base_distance := 25, speed_bonus := 0.07 }
  let thursday : DailyRide := { base_distance := 65, speed_bonus := 0.04 }
  let friday : DailyRide := { base_distance := 78, speed_bonus := 0.06 }
  let saturday : DailyRide := { base_distance := 58.5, speed_bonus := 0.02 }
  let sunday : DailyRide := { base_distance := 33.5, speed_bonus := 0.10 }
  let week_rides : List DailyRide := [monday, tuesday, wednesday, thursday, friday, saturday, sunday]
  total_effective_distance week_rides = 367.05 := by
  sorry


end NUMINAMATH_CALUDE_week_cycling_distance_l2222_222241


namespace NUMINAMATH_CALUDE_intersection_height_l2222_222232

/-- Triangle ABC with vertices A(0, 7), B(3, 0), and C(9, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Horizontal line y = t intersecting AB at T and AC at U -/
structure Intersection (ABC : Triangle) (t : ℝ) :=
  (T : ℝ × ℝ)
  (U : ℝ × ℝ)

/-- The area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem intersection_height (ABC : Triangle) (t : ℝ) (intr : Intersection ABC t) :
  ABC.A = (0, 7) ∧ ABC.B = (3, 0) ∧ ABC.C = (9, 0) →
  triangleArea ABC.A intr.T intr.U = 18 →
  t = 7 - Real.sqrt 42 :=
by sorry

end NUMINAMATH_CALUDE_intersection_height_l2222_222232


namespace NUMINAMATH_CALUDE_zero_in_interval_l2222_222263

-- Define the function f(x) = 2x + 3x
def f (x : ℝ) : ℝ := 2*x + 3*x

-- Theorem stating that the zero of f(x) is in the interval (-1, 0)
theorem zero_in_interval :
  ∃ x, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2222_222263


namespace NUMINAMATH_CALUDE_triangle_side_range_l2222_222237

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = 180

-- Theorem statement
theorem triangle_side_range (x : ℝ) :
  (∃ t1 t2 : Triangle, 
    validTriangle t1 ∧ validTriangle t2 ∧
    t1.b = 2 ∧ t2.b = 2 ∧
    t1.B = 60 ∧ t2.B = 60 ∧
    t1.a = x ∧ t2.a = x ∧
    t1 ≠ t2) →
  2 < x ∧ x < (4 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2222_222237


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l2222_222286

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem suzanna_bike_ride :
  let rate : ℝ := 0.75 / 5  -- miles per minute
  let time : ℝ := 45        -- minutes
  distance_traveled rate time = 6.75 := by
  sorry


end NUMINAMATH_CALUDE_suzanna_bike_ride_l2222_222286


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2222_222271

/-- A two-digit number satisfying the given conditions -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ (x : ℕ), 
    let tens := n / 10
    let ones := n % 10
    tens = ones^2 - 9 ∧
    10 * ones + tens = n - 27

theorem unique_two_digit_number : 
  ∃! (n : ℕ), TwoDigitNumber n ∧ n = 74 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2222_222271


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2222_222261

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 5| > 1} = {x : ℝ | x < 2 ∨ x > 3} := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2222_222261


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l2222_222296

theorem medicine_price_reduction (x : ℝ) :
  (100 : ℝ) > 0 ∧ (81 : ℝ) > 0 →
  (∃ (initial_price final_price : ℝ),
    initial_price = 100 ∧
    final_price = 81 ∧
    final_price = initial_price * (1 - x) * (1 - x)) →
  100 * (1 - x)^2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l2222_222296


namespace NUMINAMATH_CALUDE_roller_coaster_capacity_l2222_222205

theorem roller_coaster_capacity 
  (total_cars : ℕ) 
  (total_capacity : ℕ) 
  (four_seater_cars : ℕ) 
  (four_seater_capacity : ℕ) 
  (h1 : total_cars = 15)
  (h2 : total_capacity = 72)
  (h3 : four_seater_cars = 9)
  (h4 : four_seater_capacity = 4) :
  (total_capacity - four_seater_cars * four_seater_capacity) / (total_cars - four_seater_cars) = 6 := by
sorry

end NUMINAMATH_CALUDE_roller_coaster_capacity_l2222_222205


namespace NUMINAMATH_CALUDE_lesser_fraction_l2222_222249

theorem lesser_fraction (x y : ℚ) (sum_eq : x + y = 17/24) (prod_eq : x * y = 1/8) :
  min x y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2222_222249


namespace NUMINAMATH_CALUDE_choose_15_3_l2222_222206

theorem choose_15_3 : Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_choose_15_3_l2222_222206


namespace NUMINAMATH_CALUDE_temperature_frequency_l2222_222236

def temperatures : List ℤ := [-2, 0, 3, -1, 1, 0, 4]

theorem temperature_frequency :
  (temperatures.filter (λ t => t > 0)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_temperature_frequency_l2222_222236


namespace NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l2222_222201

theorem smallest_divisor_sum_of_squares (n : ℕ) : n ≥ 2 →
  (∃ (a b : ℕ), a > 1 ∧ a ∣ n ∧ b ∣ n ∧
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    n = a^2 + b^2) →
  n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l2222_222201


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l2222_222226

theorem opposite_of_negative_sqrt_two : -(-(Real.sqrt 2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l2222_222226


namespace NUMINAMATH_CALUDE_ratio_problem_l2222_222239

theorem ratio_problem (a b c x : ℝ) 
  (h1 : a / c = 3 / 7)
  (h2 : b / c = x / 7)
  (h3 : (a + b + c) / c = 2) :
  b / (a + c) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2222_222239


namespace NUMINAMATH_CALUDE_jane_earnings_l2222_222272

def payment_per_bulb : ℚ := 0.50
def tulip_bulbs : ℕ := 20
def daffodil_bulbs : ℕ := 30

def iris_bulbs : ℕ := tulip_bulbs / 2
def crocus_bulbs : ℕ := daffodil_bulbs * 3

def total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs

def total_payment : ℚ := payment_per_bulb * total_bulbs

theorem jane_earnings : total_payment = 75 := by
  sorry

end NUMINAMATH_CALUDE_jane_earnings_l2222_222272


namespace NUMINAMATH_CALUDE_circle_tangent_problem_l2222_222251

/-- Two lines l₁ and l₂ are perpendicular if their slopes multiply to -1 -/
def perpendicular (a : ℝ) : Prop := a * (1/a) = -1

/-- A line ax + by + c = 0 is tangent to the circle x² + y² = r² 
    if the distance from (0,0) to the line equals r -/
def tangent_to_circle (a b c r : ℝ) : Prop :=
  (c / (a^2 + b^2).sqrt)^2 = r^2

theorem circle_tangent_problem (a : ℝ) :
  perpendicular a →
  tangent_to_circle 1 0 2 (b^2).sqrt →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_problem_l2222_222251


namespace NUMINAMATH_CALUDE_expand_expression_l2222_222253

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2222_222253


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l2222_222255

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l2222_222255


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l2222_222265

/-- The maximum number of students for equal distribution of pens and pencils -/
theorem max_students_equal_distribution (pens pencils : ℕ) : 
  pens = 891 → pencils = 810 → 
  (∃ (max_students : ℕ), 
    max_students = Nat.gcd pens pencils ∧ 
    max_students > 0 ∧
    pens % max_students = 0 ∧ 
    pencils % max_students = 0 ∧
    ∀ (n : ℕ), n > max_students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) := by
  sorry

#eval Nat.gcd 891 810  -- Expected output: 81

end NUMINAMATH_CALUDE_max_students_equal_distribution_l2222_222265


namespace NUMINAMATH_CALUDE_y_value_l2222_222203

theorem y_value (x y : ℝ) (h1 : x^2 = y - 7) (h2 : x = 7) : y = 56 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2222_222203


namespace NUMINAMATH_CALUDE_expression_simplification_l2222_222211

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  let f := (((x^2 - x) / (x^2 - 2*x + 1) + 2 / (x - 1)) / ((x^2 - 4) / (x^2 - 1)))
  x = 3 → f = 4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2222_222211


namespace NUMINAMATH_CALUDE_unique_point_equal_angles_l2222_222202

/-- The ellipse equation x²/4 + y² = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The focus F = (2, 0) -/
def F : ℝ × ℝ := (2, 0)

/-- A chord AB passing through F -/
def is_chord_through_F (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = (2 + t * (B.1 - 2), t * B.2) ∧ 
             is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2

/-- Angles APF and BPF are equal -/
def equal_angles (P A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - P.1))^2 = (B.2 / (B.1 - P.1))^2

/-- The main theorem -/
theorem unique_point_equal_angles :
  ∃! (p : ℝ), p > 0 ∧ 
    (∀ (A B : ℝ × ℝ), is_chord_through_F A B → 
      equal_angles (p, 0) A B) ∧ 
    p = 2 := by sorry

end NUMINAMATH_CALUDE_unique_point_equal_angles_l2222_222202


namespace NUMINAMATH_CALUDE_georgia_carnation_cost_l2222_222278

-- Define the cost of a single carnation
def single_carnation_cost : ℚ := 1/2

-- Define the cost of a dozen carnations
def dozen_carnation_cost : ℚ := 4

-- Define the number of teachers
def num_teachers : ℕ := 5

-- Define the number of friends
def num_friends : ℕ := 14

-- Theorem statement
theorem georgia_carnation_cost : 
  (num_teachers : ℚ) * dozen_carnation_cost + (num_friends : ℚ) * single_carnation_cost = 27 := by
  sorry

end NUMINAMATH_CALUDE_georgia_carnation_cost_l2222_222278


namespace NUMINAMATH_CALUDE_twin_primes_difference_divisible_by_twelve_l2222_222229

/-- Twin primes are prime numbers that differ by 2 -/
def IsTwinPrime (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ (q = p + 2 ∨ p = q + 2)

/-- The main theorem statement -/
theorem twin_primes_difference_divisible_by_twelve 
  (p q r s : ℕ) 
  (hp : p > 3) 
  (hq : q > 3) 
  (hr : r > 3) 
  (hs : s > 3) 
  (hpq : IsTwinPrime p q) 
  (hrs : IsTwinPrime r s) : 
  12 ∣ (p * r - q * s) := by
  sorry

end NUMINAMATH_CALUDE_twin_primes_difference_divisible_by_twelve_l2222_222229


namespace NUMINAMATH_CALUDE_problem_statement_l2222_222213

def f (x : ℝ) := x^3 - x^2

theorem problem_statement :
  (∀ m n : ℝ, m > 0 → n > 0 → m * n > 1 → max (f m) (f n) ≥ 0) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a ≠ b → f a = f b → a + b > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2222_222213


namespace NUMINAMATH_CALUDE_ab_minimum_value_l2222_222281

theorem ab_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 3) :
  a * b ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_ab_minimum_value_l2222_222281


namespace NUMINAMATH_CALUDE_sets_problem_l2222_222223

-- Define the sets A, B, and C
def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem statement
theorem sets_problem (a : ℝ) :
  (A ∪ B = {x : ℝ | 4 ≤ x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 8 ≤ x ∧ x < 10}) ∧
  (Set.Nonempty (A ∩ C a) ↔ a < 8) := by
  sorry

end NUMINAMATH_CALUDE_sets_problem_l2222_222223


namespace NUMINAMATH_CALUDE_triangle_function_k_range_l2222_222243

-- Define the function f(x) = kx + 2
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the property of being a "triangle function" on a domain
def is_triangle_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ (x y z : ℝ), a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ a ≤ z ∧ z ≤ b →
    f x + f y > f z ∧ f y + f z > f x ∧ f z + f x > f y

-- State the theorem
theorem triangle_function_k_range :
  ∀ k : ℝ, is_triangle_function (f k) 1 4 ↔ -2/7 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_function_k_range_l2222_222243


namespace NUMINAMATH_CALUDE_factorization_of_5x_cubed_minus_125x_l2222_222273

theorem factorization_of_5x_cubed_minus_125x (x : ℝ) :
  5 * x^3 - 125 * x = 5 * x * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_5x_cubed_minus_125x_l2222_222273


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2222_222291

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {0, 2, 3}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2222_222291
