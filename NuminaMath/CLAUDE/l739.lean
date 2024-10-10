import Mathlib

namespace number_problem_l739_73910

theorem number_problem (x : ℤ) (h : x + 1015 = 3016) : x = 2001 := by
  sorry

end number_problem_l739_73910


namespace product_of_three_numbers_l739_73922

theorem product_of_three_numbers (a b c : ℕ) : 
  a * b * c = 224 →
  a < b →
  b < c →
  2 * a = c →
  ∃ (x y z : ℕ), x * y * z = 224 ∧ 2 * x = z ∧ x < y ∧ y < z :=
by sorry

end product_of_three_numbers_l739_73922


namespace negation_equivalence_l739_73986

theorem negation_equivalence (a : ℝ) : (¬(a < 0)) ↔ (¬(a^2 > a)) := by sorry

end negation_equivalence_l739_73986


namespace gcd_values_for_special_m_l739_73967

theorem gcd_values_for_special_m (m n : ℕ) (h : m + 6 = 9 * m) : 
  Nat.gcd m n = 3 ∨ Nat.gcd m n = 6 := by
sorry

end gcd_values_for_special_m_l739_73967


namespace evaluate_expression_l739_73923

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l739_73923


namespace num_different_selections_eq_six_l739_73901

/-- Represents the set of attractions -/
inductive Attraction : Type
  | A : Attraction
  | B : Attraction
  | C : Attraction

/-- Represents a selection of two attractions -/
def Selection := Finset Attraction

/-- The set of all possible selections -/
def all_selections : Finset Selection :=
  sorry

/-- Predicate to check if two selections are different -/
def different_selections (s1 s2 : Selection) : Prop :=
  s1 ≠ s2

/-- The number of ways two people can choose different selections -/
def num_different_selections : ℕ :=
  sorry

/-- Theorem: The number of ways two people can choose two out of three attractions,
    such that their choices are different, is equal to 6 -/
theorem num_different_selections_eq_six :
  num_different_selections = 6 :=
sorry

end num_different_selections_eq_six_l739_73901


namespace neds_video_games_l739_73911

theorem neds_video_games (non_working : ℕ) (price_per_game : ℕ) (total_earned : ℕ) :
  non_working = 6 →
  price_per_game = 7 →
  total_earned = 63 →
  non_working + (total_earned / price_per_game) = 15 :=
by sorry

end neds_video_games_l739_73911


namespace sharmila_average_earnings_l739_73964

/-- Represents Sharmila's work schedule and earnings --/
structure WorkSchedule where
  job1_long_days : Nat -- Number of 10-hour days in job 1
  job1_short_days : Nat -- Number of 8-hour days in job 1
  job1_hourly_rate : ℚ -- Hourly rate for job 1
  job1_long_day_bonus : ℚ -- Bonus for 10-hour days in job 1
  job2_hours : Nat -- Hours worked in job 2
  job2_hourly_rate : ℚ -- Hourly rate for job 2
  job2_bonus : ℚ -- Bonus for job 2

/-- Calculates the average hourly earnings --/
def average_hourly_earnings (schedule : WorkSchedule) : ℚ :=
  let job1_hours := schedule.job1_long_days * 10 + schedule.job1_short_days * 8
  let job1_earnings := job1_hours * schedule.job1_hourly_rate + schedule.job1_long_days * schedule.job1_long_day_bonus
  let job2_earnings := schedule.job2_hours * schedule.job2_hourly_rate + schedule.job2_bonus
  let total_earnings := job1_earnings + job2_earnings
  let total_hours := job1_hours + schedule.job2_hours
  total_earnings / total_hours

/-- Sharmila's work schedule --/
def sharmila_schedule : WorkSchedule := {
  job1_long_days := 3
  job1_short_days := 2
  job1_hourly_rate := 15
  job1_long_day_bonus := 20
  job2_hours := 5
  job2_hourly_rate := 12
  job2_bonus := 10
}

/-- Theorem stating Sharmila's average hourly earnings --/
theorem sharmila_average_earnings :
  average_hourly_earnings sharmila_schedule = 16.08 := by
  sorry


end sharmila_average_earnings_l739_73964


namespace program_list_orders_l739_73973

/-- Represents the number of items in the program list -/
def n : ℕ := 6

/-- Represents the number of items that must be adjacent -/
def adjacent_items : ℕ := 2

/-- Represents the number of slots available for inserting the item that can't be first -/
def available_slots : ℕ := n - 1

/-- Calculates the number of different orders for the program list -/
def program_orders : ℕ :=
  (Nat.factorial (n - adjacent_items + 1)) *
  (Nat.choose available_slots 1) *
  (Nat.factorial adjacent_items)

theorem program_list_orders :
  program_orders = 192 := by sorry

end program_list_orders_l739_73973


namespace angle_330_equivalent_to_negative_30_l739_73975

/-- Two angles have the same terminal side if they are equivalent modulo 360° -/
def same_terminal_side (a b : ℝ) : Prop := a % 360 = b % 360

/-- The problem statement -/
theorem angle_330_equivalent_to_negative_30 :
  same_terminal_side 330 (-30) := by sorry

end angle_330_equivalent_to_negative_30_l739_73975


namespace percentage_correct_second_question_l739_73919

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered the second question correctly. -/
theorem percentage_correct_second_question
  (total : ℝ) -- Total number of students
  (first_correct : ℝ) -- Number of students who answered the first question correctly
  (both_correct : ℝ) -- Number of students who answered both questions correctly
  (neither_correct : ℝ) -- Number of students who answered neither question correctly
  (h1 : first_correct = 0.75 * total) -- 75% answered the first question correctly
  (h2 : both_correct = 0.25 * total) -- 25% answered both questions correctly
  (h3 : neither_correct = 0.2 * total) -- 20% answered neither question correctly
  : (total - neither_correct - (first_correct - both_correct)) / total = 0.3 := by
  sorry


end percentage_correct_second_question_l739_73919


namespace theorem_60752_infinite_primes_4k_plus_1_l739_73917

-- Theorem from problem 60752
theorem theorem_60752 (N : ℕ) (a : ℕ) (h : N = a^2 + 1) :
  ∃ p : ℕ, Prime p ∧ p ∣ N ∧ ∃ k : ℕ, p = 4 * k + 1 := sorry

theorem infinite_primes_4k_plus_1 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Prime p ∧ ∃ k : ℕ, p = 4 * k + 1 := by sorry

end theorem_60752_infinite_primes_4k_plus_1_l739_73917


namespace quadratic_roots_constraint_l739_73937

/-- A quadratic function f(x) = x^2 + 2bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

/-- The equation f(x) + x + b = 0 -/
def g (b c x : ℝ) : ℝ := f b c x + x + b

theorem quadratic_roots_constraint (b c : ℝ) :
  f b c 1 = 0 ∧
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-3) (-2) ∧ x₂ ∈ Set.Ioo 0 1 ∧
    g b c x₁ = 0 ∧ g b c x₂ = 0) →
  b ∈ Set.Ioo (-5/2) (-1/2) :=
by sorry

end quadratic_roots_constraint_l739_73937


namespace star_equal_is_four_lines_l739_73990

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Define the union of four lines
def four_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

-- Theorem statement
theorem star_equal_is_four_lines : star_equal_set = four_lines := by
  sorry

end star_equal_is_four_lines_l739_73990


namespace divisors_of_36_l739_73977

theorem divisors_of_36 : 
  ∃ (divs : List Nat), 
    (∀ d, d ∈ divs ↔ d ∣ 36) ∧ 
    divs.length = 9 ∧
    divs = [1, 2, 3, 4, 6, 9, 12, 18, 36] :=
by sorry

end divisors_of_36_l739_73977


namespace m_range_l739_73944

theorem m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m * x + 1 > 0) → 
  m > -2 ∧ m < 0 :=
by sorry

end m_range_l739_73944


namespace sqrt_difference_less_than_sqrt_of_difference_l739_73970

theorem sqrt_difference_less_than_sqrt_of_difference 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end sqrt_difference_less_than_sqrt_of_difference_l739_73970


namespace factor_expression_l739_73966

theorem factor_expression (x : ℝ) : 63 * x + 45 = 9 * (7 * x + 5) := by
  sorry

end factor_expression_l739_73966


namespace water_cup_pricing_equation_l739_73998

/-- Represents the pricing of a Huashan brand water cup -/
def water_cup_pricing (x : ℝ) : Prop :=
  let first_discount := x - 5
  let second_discount := 0.8 * first_discount
  second_discount = 60

/-- The equation representing the water cup pricing after discounts -/
theorem water_cup_pricing_equation (x : ℝ) :
  water_cup_pricing x ↔ 0.8 * (x - 5) = 60 := by sorry

end water_cup_pricing_equation_l739_73998


namespace unfair_coin_flip_probability_l739_73974

/-- The probability of flipping exactly k tails in n flips of an unfair coin -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 3 tails in 8 flips of an unfair coin with 2/3 probability of tails -/
theorem unfair_coin_flip_probability : 
  binomial_probability 8 3 (2/3) = 448/6561 := by
  sorry

end unfair_coin_flip_probability_l739_73974


namespace sum_of_powers_l739_73907

theorem sum_of_powers : (-2)^4 + (-2)^(3/2) + (-2)^1 + 2^1 + 2^(3/2) + 2^4 = 32 := by
  sorry

end sum_of_powers_l739_73907


namespace arithmetic_sequence_seventh_term_l739_73939

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 = 4)
  (h_sum2 : a 2 + a 3 = 8) :
  a 7 = 13 := by
sorry

end arithmetic_sequence_seventh_term_l739_73939


namespace incorrect_inequality_l739_73955

theorem incorrect_inequality (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) : 
  ¬(a * b > b^2) := by
sorry

end incorrect_inequality_l739_73955


namespace log_relation_l739_73935

theorem log_relation (x k : ℝ) (h1 : Real.log 3 / Real.log 4 = x) (h2 : Real.log 64 / Real.log 2 = k * x) : k = 3 := by
  sorry

end log_relation_l739_73935


namespace oocyte_characteristics_l739_73929

/-- Represents the hair length trait in rabbits -/
inductive HairTrait
  | Long
  | Short

/-- Represents the phase of meiosis -/
inductive MeioticPhase
  | First
  | Second

/-- Represents a heterozygous rabbit with long hair trait dominant over short hair trait -/
structure HeterozygousRabbit where
  dominantTrait : HairTrait
  recessiveTrait : HairTrait
  totalGenes : ℕ
  genesPerOocyte : ℕ
  nucleotideTypes : ℕ
  allelesSeperationPhase : MeioticPhase

/-- Main theorem about the characteristics of oocytes in a heterozygous rabbit -/
theorem oocyte_characteristics (rabbit : HeterozygousRabbit)
  (h1 : rabbit.dominantTrait = HairTrait.Long)
  (h2 : rabbit.recessiveTrait = HairTrait.Short)
  (h3 : rabbit.totalGenes = 20)
  (h4 : rabbit.genesPerOocyte = 4)
  (h5 : rabbit.nucleotideTypes = 4)
  (h6 : rabbit.allelesSeperationPhase = MeioticPhase.First) :
  let maxShortHairOocytes := rabbit.totalGenes / rabbit.genesPerOocyte / 2
  maxShortHairOocytes = 5 ∧
  rabbit.nucleotideTypes = 4 ∧
  rabbit.allelesSeperationPhase = MeioticPhase.First :=
by sorry

end oocyte_characteristics_l739_73929


namespace shaded_square_area_l739_73984

/-- Represents a figure with five squares and two right-angled triangles -/
structure GeometricFigure where
  square1 : ℝ
  square2 : ℝ
  square3 : ℝ
  square4 : ℝ
  square5 : ℝ

/-- The theorem stating the area of the shaded square -/
theorem shaded_square_area (fig : GeometricFigure) 
  (h1 : fig.square1 = 5)
  (h2 : fig.square2 = 8)
  (h3 : fig.square3 = 32) :
  fig.square5 = 45 := by
  sorry


end shaded_square_area_l739_73984


namespace algebraic_expression_equals_one_l739_73995

theorem algebraic_expression_equals_one
  (m n : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_diff : m - n = 1/2) :
  (m^2 - n^2) / (2*m^2 + 2*m*n) / (m - (2*m*n - n^2) / m) = 1 := by
sorry

end algebraic_expression_equals_one_l739_73995


namespace smallest_number_with_remainder_one_l739_73941

theorem smallest_number_with_remainder_one : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 9 = 0 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) ∧
  (∀ m : ℕ, 0 < m ∧ m < n → ¬(m % 9 = 0 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → m % k = 1))) :=
by sorry

end smallest_number_with_remainder_one_l739_73941


namespace bankers_discount_example_l739_73943

/-- Given a true discount and a sum due, calculate the banker's discount -/
def bankers_discount (true_discount : ℚ) (sum_due : ℚ) : ℚ :=
  (true_discount * sum_due) / (sum_due - true_discount)

/-- Theorem: The banker's discount is 78 given a true discount of 66 and a sum due of 429 -/
theorem bankers_discount_example : bankers_discount 66 429 = 78 := by
  sorry

end bankers_discount_example_l739_73943


namespace three_distinct_real_roots_l739_73978

/-- The cubic function f(x) = x^3 - 3x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

/-- Theorem stating the condition for three distinct real roots -/
theorem three_distinct_real_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end three_distinct_real_roots_l739_73978


namespace sufficient_not_necessary_l739_73953

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0)) :=
by sorry

end sufficient_not_necessary_l739_73953


namespace adultChildRatioIsTwo_l739_73927

/-- Represents the ticket prices and attendance information for a show -/
structure ShowInfo where
  adultTicketPrice : ℚ
  childTicketPrice : ℚ
  totalReceipts : ℚ
  numAdults : ℕ

/-- Calculates the ratio of adults to children given show information -/
def adultChildRatio (info : ShowInfo) : ℚ :=
  let numChildren := (info.totalReceipts - info.adultTicketPrice * info.numAdults) / info.childTicketPrice
  info.numAdults / numChildren

/-- Theorem stating that the ratio of adults to children is 2:1 for the given show information -/
theorem adultChildRatioIsTwo (info : ShowInfo) 
    (h1 : info.adultTicketPrice = 11/2)
    (h2 : info.childTicketPrice = 5/2)
    (h3 : info.totalReceipts = 1026)
    (h4 : info.numAdults = 152) : 
  adultChildRatio info = 2 := by
  sorry

#eval adultChildRatio {
  adultTicketPrice := 11/2,
  childTicketPrice := 5/2,
  totalReceipts := 1026,
  numAdults := 152
}

end adultChildRatioIsTwo_l739_73927


namespace cookie_division_l739_73942

/-- The area of a cookie piece when a cookie with total area 81.12 cm² is divided equally among 6 friends -/
theorem cookie_division (total_area : ℝ) (num_friends : ℕ) 
  (h1 : total_area = 81.12)
  (h2 : num_friends = 6) :
  total_area / num_friends = 13.52 := by
  sorry

end cookie_division_l739_73942


namespace stock_fall_amount_l739_73979

/-- Represents the daily change in stock value -/
structure StockChange where
  morning_rise : ℚ
  afternoon_fall : ℚ

/-- Models the stock behavior over time -/
def stock_value (initial_value : ℚ) (daily_change : StockChange) (days : ℕ) : ℚ :=
  initial_value + (daily_change.morning_rise - daily_change.afternoon_fall) * days

/-- Theorem stating the condition for the stock to reach a specific value -/
theorem stock_fall_amount (initial_value target_value : ℚ) (days : ℕ) :
  let morning_rise := 2
  ∀ afternoon_fall : ℚ,
    stock_value initial_value ⟨morning_rise, afternoon_fall⟩ (days - 1) < target_value ∧
    stock_value initial_value ⟨morning_rise, afternoon_fall⟩ days ≥ target_value →
    afternoon_fall = 98 / 99 :=
by sorry

end stock_fall_amount_l739_73979


namespace remainder_twelve_remainder_107_is_least_unique_divisor_l739_73926

def least_number : ℕ := 540

-- 540 leaves a remainder of 5 when divided by 12
theorem remainder_twelve : least_number % 12 = 5 := by sorry

-- 107 leaves a remainder of 5 when 540 is divided by it
theorem remainder_107 : least_number % 107 = 5 := by sorry

-- 540 is the least number that leaves a remainder of 5 when divided by some numbers
theorem is_least (n : ℕ) : n < least_number → ¬(∃ m : ℕ, m > 1 ∧ n % m = 5) := by sorry

-- 107 is the only number (other than 12) that leaves a remainder of 5 when 540 is divided by it
theorem unique_divisor (n : ℕ) : n ≠ 12 → n ≠ 107 → least_number % n ≠ 5 := by sorry

end remainder_twelve_remainder_107_is_least_unique_divisor_l739_73926


namespace solution_set_f_leq_6_range_of_a_l739_73972

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a < -3 ∨ a > 5} := by sorry

end solution_set_f_leq_6_range_of_a_l739_73972


namespace sequence_fourth_term_l739_73906

theorem sequence_fourth_term 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h : ∀ n, S n = (n + 1 : ℚ) / (n + 2 : ℚ)) : 
  a 4 = 1 / 30 := by
  sorry

end sequence_fourth_term_l739_73906


namespace arithmetic_progression_problem_l739_73918

/-- An arithmetic progression with its sum sequence -/
structure ArithmeticProgression where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_progression_problem (seq : ArithmeticProgression) 
    (h1 : seq.a 1 + (seq.a 2)^2 = -3)
    (h2 : seq.S 5 = 10) :
  seq.a 9 = 20 := by
  sorry

end arithmetic_progression_problem_l739_73918


namespace discount_rate_for_profit_margin_l739_73976

/-- Proves that a 20% discount rate maintains a 20% profit margin for a toy gift box. -/
theorem discount_rate_for_profit_margin :
  let cost_price : ℝ := 160
  let marked_price : ℝ := 240
  let profit_margin : ℝ := 0.2
  let discount_rate : ℝ := 0.2
  let discounted_price : ℝ := marked_price * (1 - discount_rate)
  let profit : ℝ := discounted_price - cost_price
  profit / cost_price = profit_margin :=
by
  sorry

#check discount_rate_for_profit_margin

end discount_rate_for_profit_margin_l739_73976


namespace inequality_proof_l739_73994

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hx : x = Real.sqrt (a^2 + b^2)) (hy : y = Real.sqrt (c^2 + d^2)) :
  x * y ≥ a * c + b * d := by
  sorry

end inequality_proof_l739_73994


namespace greatest_c_for_quadratic_range_l739_73946

theorem greatest_c_for_quadratic_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ -6) ↔ c ≤ 9 :=
sorry

end greatest_c_for_quadratic_range_l739_73946


namespace unique_magnitude_quadratic_l739_73959

theorem unique_magnitude_quadratic : ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m := by
  sorry

end unique_magnitude_quadratic_l739_73959


namespace smallest_number_proof_l739_73905

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  (a ≤ b ∧ b ≤ c) ∨ (a ≤ c ∧ c ≤ b) ∨ (b ≤ a ∧ a ≤ c) ∨ 
  (b ≤ c ∧ c ≤ a) ∨ (c ≤ a ∧ a ≤ b) ∨ (c ≤ b ∧ b ≤ a) →  -- Median condition
  b = 28 →                 -- Median is 28
  c = 34 →                 -- Largest number is 6 more than median
  a = 28                   -- Smallest number is 28
:= by sorry

end smallest_number_proof_l739_73905


namespace expression_evaluation_l739_73958

theorem expression_evaluation (m n : ℚ) (hm : m = -1) (hn : n = 1/2) :
  (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n^2) / (m^3 - m * n^2)) = -2 := by
  sorry

end expression_evaluation_l739_73958


namespace wire_ratio_l739_73997

/-- Given a wire of length 21 cm cut into two pieces, where the shorter piece is 5.999999999999998 cm long,
    prove that the ratio of the shorter piece to the longer piece is 2:5. -/
theorem wire_ratio (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 21 →
  shorter_length = 5.999999999999998 →
  let longer_length := total_length - shorter_length
  shorter_length / longer_length = 2 / 5 := by
  sorry

end wire_ratio_l739_73997


namespace cubic_expression_property_l739_73936

theorem cubic_expression_property (a b : ℝ) :
  a * (3^3) + b * 3 - 5 = 20 → a * ((-3)^3) + b * (-3) - 5 = -30 := by
  sorry

end cubic_expression_property_l739_73936


namespace condition_necessary_not_sufficient_l739_73980

-- Define a sequence
def Sequence := ℕ → ℝ

-- Define the property of being a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the given condition
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)

-- State the theorem
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) := by
  sorry

end condition_necessary_not_sufficient_l739_73980


namespace abc_sum_root_l739_73902

theorem abc_sum_root (a b c : ℝ) 
  (h1 : b + c = 7) 
  (h2 : c + a = 8) 
  (h3 : a + b = 9) : 
  Real.sqrt (a * b * c * (a + b + c)) = 12 * Real.sqrt 5 := by
  sorry

end abc_sum_root_l739_73902


namespace factorization_equality_l739_73992

theorem factorization_equality (a b : ℝ) : 
  a^2 - b^2 + 4*a + 2*b + 3 = (a + b + 1)*(a - b + 3) := by
sorry

end factorization_equality_l739_73992


namespace movie_and_popcorn_expense_l739_73996

/-- The fraction of allowance spent on movie ticket and popcorn -/
theorem movie_and_popcorn_expense (B : ℝ) (m p : ℝ) 
  (hm : m = (1/4) * (B - p)) 
  (hp : p = (1/10) * (B - m)) : 
  (m + p) / B = 4/13 := by
  sorry

end movie_and_popcorn_expense_l739_73996


namespace ab_product_l739_73938

theorem ab_product (a b : ℚ) (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * a * b = 800 := by
  sorry

end ab_product_l739_73938


namespace mono_increasing_and_even_shift_implies_l739_73950

/-- A function that is monotonically increasing on [1,+∞) and f(x+1) is even -/
def MonoIncreasingAndEvenShift (f : ℝ → ℝ) : Prop :=
  (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y) ∧
  (∀ x, f (x + 1) = f (-x + 1))

/-- Theorem: If f is monotonically increasing on [1,+∞) and f(x+1) is even, then f(-2) > f(2) -/
theorem mono_increasing_and_even_shift_implies (f : ℝ → ℝ) 
  (h : MonoIncreasingAndEvenShift f) : f (-2) > f 2 := by
  sorry

end mono_increasing_and_even_shift_implies_l739_73950


namespace ellipse_condition_l739_73957

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  (5 - m > 0) ∧ (m + 3 > 0) ∧ (5 - m ≠ m + 3)

/-- The condition -3 < m < 5 -/
def condition (m : ℝ) : Prop :=
  -3 < m ∧ m < 5

theorem ellipse_condition (m : ℝ) :
  (is_ellipse m → condition m) ∧ 
  ¬(condition m → is_ellipse m) :=
sorry

end ellipse_condition_l739_73957


namespace sequence_properties_l739_73909

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := 2^(n - 1)

def sum_sequence_a (n : ℕ) : ℕ := n^2

def sum_sequence_ab (n : ℕ) : ℕ := (2 * n - 3) * 2^n + 3

theorem sequence_properties :
  (∀ n, sum_sequence_a n = n^2) →
  sequence_b 2 = 2 →
  sequence_b 5 = 16 →
  (∀ n, sequence_a n = 2 * n - 1) ∧
  (∀ n, sequence_b n = 2^(n - 1)) ∧
  (∀ n, sum_sequence_ab n = (2 * n - 3) * 2^n + 3) := by
  sorry

end sequence_properties_l739_73909


namespace fractional_equation_root_l739_73982

theorem fractional_equation_root (k : ℚ) : 
  (∃ x : ℚ, x ≠ 1 ∧ (2 * k) / (x - 1) - 3 / (1 - x) = 1) → k = -3/2 := by
  sorry

end fractional_equation_root_l739_73982


namespace set_operations_l739_73963

def A : Set ℤ := {x : ℤ | |x| < 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5}

theorem set_operations :
  (B ∩ C = {3}) ∧
  (B ∪ C = {1, 2, 3, 4, 5}) ∧
  (A ∪ (B ∩ C) = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}) ∧
  (A ∩ (A \ (B ∪ C)) = {-5, -4, -3, -2, -1, 0}) := by
sorry

end set_operations_l739_73963


namespace xy_and_expression_values_l739_73930

theorem xy_and_expression_values (x y : ℝ) 
  (h1 : x - 2*y = 3) 
  (h2 : (x - 2)*(y + 1) = 2) : 
  x*y = 1 ∧ (x^2 - 2)*(2*y^2 - 1) = -9 := by
  sorry

end xy_and_expression_values_l739_73930


namespace bells_toll_together_once_l739_73916

def bell_intervals : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23]

def lcm_list (L : List ℕ) : ℕ :=
  L.foldl Nat.lcm 1

theorem bells_toll_together_once (intervals : List ℕ) (duration : ℕ) : 
  intervals = bell_intervals → duration = 60 * 60 → 
  (duration / (lcm_list intervals) + 1 : ℕ) = 1 :=
by sorry

end bells_toll_together_once_l739_73916


namespace equation_solution_l739_73987

theorem equation_solution (x : ℝ) : 
  (Real.cos (2 * x / 5) - Real.cos (2 * Real.pi / 15))^2 + 
  (Real.sin (2 * x / 3) - Real.sin (4 * Real.pi / 9))^2 = 0 ↔ 
  ∃ t : ℤ, x = 29 * Real.pi / 3 + 15 * Real.pi * (t : ℝ) :=
by sorry

end equation_solution_l739_73987


namespace min_value_shifted_sine_l739_73908

theorem min_value_shifted_sine (φ : ℝ) (h_φ : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2*x - π/3)
  ∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₀ ≤ f x ∧ f x₀ = -Real.sqrt 3 / 2 := by
  sorry

end min_value_shifted_sine_l739_73908


namespace haley_marbles_l739_73956

/-- The number of boys who love to play marbles -/
def num_marble_boys : ℕ := 13

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 2

/-- The total number of marbles Haley has -/
def total_marbles : ℕ := num_marble_boys * marbles_per_boy

theorem haley_marbles : total_marbles = 26 := by
  sorry

end haley_marbles_l739_73956


namespace cody_remaining_games_l739_73947

theorem cody_remaining_games (initial_games given_away_games : ℕ) 
  (h1 : initial_games = 9)
  (h2 : given_away_games = 4) :
  initial_games - given_away_games = 5 := by
  sorry

end cody_remaining_games_l739_73947


namespace money_division_theorem_l739_73954

theorem money_division_theorem (p q r : ℕ) (total : ℚ) :
  p + q + r = 22 →  -- Ratio sum: 3 + 7 + 12 = 22
  (7 * total / 22 - 3 * total / 22 = 2800) →  -- Difference between q and p's shares
  (12 * total / 22 - 7 * total / 22 = 3500) :=  -- Difference between r and q's shares
by sorry

end money_division_theorem_l739_73954


namespace triangle_in_radius_l739_73991

/-- Given a triangle with perimeter 36 cm and area 45 cm², prove that its in radius is 2.5 cm. -/
theorem triangle_in_radius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : P = 36) 
  (h_area : A = 45) 
  (h_in_radius : A = r * (P / 2)) : r = 2.5 := by
  sorry

end triangle_in_radius_l739_73991


namespace parabola_directrix_l739_73952

/-- The equation of the directrix of the parabola y = 4x^2 -/
theorem parabola_directrix (x y : ℝ) :
  (y = 4 * x^2) →  -- Given parabola equation
  ∃ (d : ℝ), d = -1/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = 4 * x₀^2 → y₀ ≥ d) ∧
              (∀ ε > 0, ∃ (x₁ y₁ : ℝ), y₁ = 4 * x₁^2 ∧ y₁ < d + ε) :=
by sorry

end parabola_directrix_l739_73952


namespace company_fund_distribution_l739_73912

/-- Represents the company fund distribution problem -/
theorem company_fund_distribution (n : ℕ) 
  (h1 : 50 * n + 130 = 60 * n - 10) : 
  60 * n - 10 = 830 :=
by
  sorry

#check company_fund_distribution

end company_fund_distribution_l739_73912


namespace negative_movement_l739_73921

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) : Direction :=
  if distance > 0 then Direction.East else Direction.West

-- Define the theorem
theorem negative_movement :
  (movement 30 = Direction.East) →
  (movement (-50) = Direction.West) :=
by
  sorry

end negative_movement_l739_73921


namespace greatest_integer_for_all_real_domain_l739_73985

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), x^2 + b*x + 15 ≠ 0) ∧ 
  (∀ (c : ℤ), (∀ (x : ℝ), x^2 + c*x + 15 ≠ 0) → c ≤ b) ∧ 
  b = 7 := by
  sorry

end greatest_integer_for_all_real_domain_l739_73985


namespace volume_of_rotated_region_l739_73932

/-- The volume of the solid formed by rotating the region bounded by y = 2x - x^2 and y = 2x^2 - 4x around the x-axis. -/
theorem volume_of_rotated_region : ∃ V : ℝ,
  (∀ x y : ℝ, (y = 2*x - x^2 ∨ y = 2*x^2 - 4*x) → 
    V = π * ∫ x in (0)..(2), ((2*x^2 - 4*x)^2 - (2*x - x^2)^2)) ∧
  V = (16 * π) / 5 := by sorry

end volume_of_rotated_region_l739_73932


namespace necessary_but_not_sufficient_condition_l739_73960

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∀ x, -1 ≤ x ∧ x < 2 → x ≤ a) ∧ 
  (∃ x, x ≤ a ∧ (x < -1 ∨ x ≥ 2)) →
  a ≥ 2 :=
by sorry

end necessary_but_not_sufficient_condition_l739_73960


namespace dave_initial_apps_l739_73968

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := 15

/-- The number of apps Dave added -/
def added_apps : ℕ := 71

/-- The number of apps Dave had left after deleting some -/
def remaining_apps : ℕ := 14

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := added_apps + 1

theorem dave_initial_apps : 
  initial_apps + added_apps - deleted_apps = remaining_apps :=
by sorry

end dave_initial_apps_l739_73968


namespace stratified_sampling_grade10_l739_73981

theorem stratified_sampling_grade10 (total_students : ℕ) (grade10_students : ℕ) (sample_size : ℕ) :
  total_students = 700 →
  grade10_students = 300 →
  sample_size = 35 →
  (grade10_students * sample_size) / total_students = 15 :=
by
  sorry

end stratified_sampling_grade10_l739_73981


namespace billion_to_scientific_notation_l739_73961

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (40.9 * 1000000000) =
    ScientificNotation.mk 4.09 9 (by sorry) := by sorry

end billion_to_scientific_notation_l739_73961


namespace complex_on_line_l739_73915

theorem complex_on_line (z : ℂ) (a : ℝ) : 
  z = (1 - a * Complex.I) / Complex.I →
  (z.re : ℝ) + 2 * (z.im : ℝ) + 5 = 0 →
  a = 3 := by
sorry

end complex_on_line_l739_73915


namespace thought_number_is_729_l739_73969

/-- 
Given a three-digit number, if each of the numbers 109, 704, and 124 
matches it exactly in one digit place, then the number is 729.
-/
theorem thought_number_is_729 (x : ℕ) : 
  (100 ≤ x ∧ x < 1000) → 
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 109 / 10^d % 10)) →
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 704 / 10^d % 10)) →
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 124 / 10^d % 10)) →
  x = 729 := by
sorry


end thought_number_is_729_l739_73969


namespace trigonometric_identities_l739_73988

theorem trigonometric_identities :
  -- Part 1
  ¬∃x : ℝ, x = Real.sin (-14 / 3 * π) + Real.cos (20 / 3 * π) + Real.tan (-53 / 6 * π) ∧
  -- Part 2
  Real.tan (675 * π / 180) - Real.sin (-330 * π / 180) - Real.cos (960 * π / 180) = 0 :=
by sorry

end trigonometric_identities_l739_73988


namespace matrix_homomorphism_implies_equal_dim_l739_73989

-- Define the set of valid dimensions
def ValidDim : Set ℕ := {2, 3}

-- Define the property of the bijective function
def IsMatrixHomomorphism {n p : ℕ} (f : Matrix (Fin n) (Fin n) ℂ → Matrix (Fin p) (Fin p) ℂ) : Prop :=
  ∀ X Y : Matrix (Fin n) (Fin n) ℂ, f (X * Y) = f X * f Y

-- The main theorem
theorem matrix_homomorphism_implies_equal_dim (n p : ℕ) 
  (hn : n ∈ ValidDim) (hp : p ∈ ValidDim) :
  (∃ f : Matrix (Fin n) (Fin n) ℂ → Matrix (Fin p) (Fin p) ℂ, 
    Function.Bijective f ∧ IsMatrixHomomorphism f) → n = p := by
  sorry

end matrix_homomorphism_implies_equal_dim_l739_73989


namespace probability_black_ball_l739_73949

/-- The probability of drawing a black ball from a bag of colored balls. -/
theorem probability_black_ball (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ) :
  total = red + white + black →
  total = 6 →
  red = 1 →
  white = 2 →
  black = 3 →
  (black : ℚ) / total = 1 / 2 := by
  sorry

end probability_black_ball_l739_73949


namespace right_triangle_area_rational_l739_73934

/-- A right-angled triangle with integer coordinates -/
structure RightTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The area of a right-angled triangle with integer coordinates -/
def area (t : RightTriangle) : ℚ :=
  (|t.a * t.d - t.b * t.c| : ℚ) / 2

/-- Theorem: The area of a right-angled triangle with integer coordinates is always rational -/
theorem right_triangle_area_rational (t : RightTriangle) : 
  ∃ (q : ℚ), area t = q :=
sorry

end right_triangle_area_rational_l739_73934


namespace solve_equation_l739_73933

theorem solve_equation (m : ℝ) : (m - 4)^2 = (1/16)⁻¹ → m = 8 ∨ m = 0 := by
  sorry

end solve_equation_l739_73933


namespace wolf_still_hungry_l739_73971

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

end wolf_still_hungry_l739_73971


namespace complex_magnitude_theorem_l739_73993

theorem complex_magnitude_theorem (a b : ℂ) (x : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = x - 6 * Complex.I →
  x = 3 * Real.sqrt 21 :=
by sorry

end complex_magnitude_theorem_l739_73993


namespace smallest_valid_six_digit_number_l739_73962

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2

def append_three_digits (n : ℕ) (m : ℕ) : ℕ :=
  n * 1000 + m

theorem smallest_valid_six_digit_number :
  ∃ (n m : ℕ),
    is_valid_three_digit n ∧
    m < 1000 ∧
    let six_digit := append_three_digits n m
    six_digit = 122040 ∧
    six_digit % 4 = 0 ∧
    six_digit % 5 = 0 ∧
    six_digit % 6 = 0 ∧
    ∀ (n' m' : ℕ),
      is_valid_three_digit n' ∧
      m' < 1000 ∧
      let six_digit' := append_three_digits n' m'
      six_digit' % 4 = 0 ∧
      six_digit' % 5 = 0 ∧
      six_digit' % 6 = 0 →
      six_digit ≤ six_digit' :=
by
  sorry

end smallest_valid_six_digit_number_l739_73962


namespace largest_inscribed_circle_radius_is_2_sqrt_8_l739_73999

/-- A quadrilateral with given side lengths -/
structure Quadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the largest inscribed circle radius for the given quadrilateral is 2√8 -/
theorem largest_inscribed_circle_radius_is_2_sqrt_8 :
  let q : Quadrilateral := ⟨15, 10, 8, 13⟩
  largest_inscribed_circle_radius q = 2 * Real.sqrt 8 := by sorry

end largest_inscribed_circle_radius_is_2_sqrt_8_l739_73999


namespace price_ratio_theorem_l739_73925

theorem price_ratio_theorem (cost_price : ℝ) (first_price second_price : ℝ) :
  first_price = cost_price * (1 + 1.4) ∧
  second_price = cost_price * (1 - 0.2) →
  second_price / first_price = 1 / 3 := by
  sorry

end price_ratio_theorem_l739_73925


namespace flags_on_circular_track_l739_73983

/-- The number of flags needed on a circular track -/
def num_flags (track_length : ℕ) (flag_interval : ℕ) : ℕ :=
  (track_length / flag_interval) + 1

/-- Theorem: 5 flags are needed for a 400m track with 90m intervals -/
theorem flags_on_circular_track :
  num_flags 400 90 = 5 := by
  sorry

end flags_on_circular_track_l739_73983


namespace sum_of_squares_orthogonal_matrix_l739_73945

theorem sum_of_squares_orthogonal_matrix (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A.transpose = A⁻¹) : 
  (A 0 0)^2 + (A 0 1)^2 + (A 0 2)^2 + 
  (A 1 0)^2 + (A 1 1)^2 + (A 1 2)^2 + 
  (A 2 0)^2 + (A 2 1)^2 + (A 2 2)^2 = 3 := by
  sorry

end sum_of_squares_orthogonal_matrix_l739_73945


namespace f_has_max_and_min_iff_m_in_range_l739_73948

/-- The function f with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + (m + 6)

/-- The discriminant of f' -/
def discriminant (m : ℝ) : ℝ := (2*m)^2 - 4*3*(m + 6)

theorem f_has_max_and_min_iff_m_in_range (m : ℝ) :
  (∃ (a b : ℝ), ∀ x, f m x ≤ f m a ∧ f m x ≥ f m b) ↔ 
  m < -3 ∨ m > 6 :=
sorry

end f_has_max_and_min_iff_m_in_range_l739_73948


namespace certain_number_value_l739_73914

theorem certain_number_value : ∃ x : ℝ, 15 * x = 165 ∧ x = 11 := by
  sorry

end certain_number_value_l739_73914


namespace first_half_speed_l739_73924

theorem first_half_speed (total_distance : ℝ) (first_half_distance : ℝ) (second_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 20 →
  first_half_distance = 10 →
  second_half_speed = 10 →
  average_speed = 10.909090909090908 →
  (total_distance / (first_half_distance / (total_distance / average_speed - first_half_distance / second_half_speed) + first_half_distance / second_half_speed)) = 12 :=
by sorry

end first_half_speed_l739_73924


namespace maisy_new_job_earnings_l739_73920

/-- Represents Maisy's job options and calculates the difference in earnings -/
def earnings_difference (current_hours : ℕ) (current_wage : ℕ) (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ) : ℕ :=
  let current_earnings := current_hours * current_wage
  let new_earnings := new_hours * new_wage + bonus
  new_earnings - current_earnings

/-- Proves that Maisy will earn $15 more at her new job -/
theorem maisy_new_job_earnings :
  earnings_difference 8 10 4 15 35 = 15 := by
  sorry

end maisy_new_job_earnings_l739_73920


namespace tile_ratio_l739_73903

/-- Given a square pattern with black and white tiles and a white border added, 
    calculate the ratio of black to white tiles. -/
theorem tile_ratio (initial_black initial_white border_width : ℕ) 
  (h1 : initial_black = 5)
  (h2 : initial_white = 20)
  (h3 : border_width = 1)
  (h4 : initial_black + initial_white = (initial_black + initial_white).sqrt ^ 2) :
  let total_side := (initial_black + initial_white).sqrt + 2 * border_width
  let total_tiles := total_side ^ 2
  let added_white := total_tiles - (initial_black + initial_white)
  let final_white := initial_white + added_white
  (initial_black : ℚ) / final_white = 5 / 44 := by sorry

end tile_ratio_l739_73903


namespace triangle_angle_not_all_greater_than_60_l739_73900

theorem triangle_angle_not_all_greater_than_60 :
  ¬ ∃ (a b c : ℝ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧  -- angles are positive
    (a + b + c = 180) ∧        -- sum of angles in a triangle is 180°
    (60 < a ∧ 60 < b ∧ 60 < c) -- all angles greater than 60°
  := by sorry

end triangle_angle_not_all_greater_than_60_l739_73900


namespace pounds_per_pillow_is_two_l739_73940

-- Define the constants from the problem
def feathers_per_pound : ℕ := 300
def total_feathers : ℕ := 3600
def number_of_pillows : ℕ := 6

-- Define the function to calculate pounds of feathers needed per pillow
def pounds_per_pillow : ℚ :=
  (total_feathers / feathers_per_pound) / number_of_pillows

-- Theorem to prove
theorem pounds_per_pillow_is_two : pounds_per_pillow = 2 := by
  sorry


end pounds_per_pillow_is_two_l739_73940


namespace arithmetic_square_root_of_9_l739_73904

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end arithmetic_square_root_of_9_l739_73904


namespace u_converges_to_zero_l739_73951

open Real

variable (f : ℝ → ℝ)
variable (u : ℕ → ℝ)

-- f is non-decreasing
axiom f_nondecreasing : ∀ x y, x ≤ y → f x ≤ f y

-- f(y) - f(x) < y - x for all real numbers x and y > x
axiom f_contractive : ∀ x y, x < y → f y - f x < y - x

-- Recurrence relation for u
axiom u_recurrence : ∀ n : ℕ, u (n + 2) = f (u (n + 1)) - f (u n)

-- Theorem to prove
theorem u_converges_to_zero : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n| < ε) :=
sorry

end u_converges_to_zero_l739_73951


namespace cube_sum_equality_l739_73965

theorem cube_sum_equality (a b c : ℕ+) (h : a^3 + b^3 + c^3 = 3*a*b*c) : a = b ∧ b = c := by
  sorry

end cube_sum_equality_l739_73965


namespace log_difference_l739_73913

theorem log_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) :
  b - d = 93 := by
  sorry

end log_difference_l739_73913


namespace move_right_result_l739_73931

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally in the Cartesian coordinate system -/
def moveHorizontal (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The initial point (-1, 2) -/
def initialPoint : Point :=
  { x := -1, y := 2 }

/-- The number of units to move right -/
def moveRightUnits : ℝ := 3

/-- The final point after moving -/
def finalPoint : Point := moveHorizontal initialPoint moveRightUnits

/-- Theorem: Moving the initial point 3 units to the right results in (2, 2) -/
theorem move_right_result :
  finalPoint = { x := 2, y := 2 } := by
  sorry

end move_right_result_l739_73931


namespace padma_valuable_cards_l739_73928

theorem padma_valuable_cards (padma_initial : ℕ) (robert_initial : ℕ) (total_traded : ℕ) 
  (padma_received : ℕ) (robert_received : ℕ) (robert_traded : ℕ) 
  (h1 : padma_initial = 75)
  (h2 : robert_initial = 88)
  (h3 : total_traded = 35)
  (h4 : padma_received = 10)
  (h5 : robert_received = 15)
  (h6 : robert_traded = 8) :
  ∃ (padma_valuable : ℕ), 
    padma_valuable + robert_received = total_traded ∧ 
    padma_valuable = 20 :=
by sorry

end padma_valuable_cards_l739_73928
