import Mathlib

namespace NUMINAMATH_CALUDE_meaningful_reciprocal_l1893_189313

theorem meaningful_reciprocal (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_reciprocal_l1893_189313


namespace NUMINAMATH_CALUDE_f_decreasing_iff_a_in_range_l1893_189323

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2*a)^x else Real.log x / Real.log a + 1/3

-- Define the property of f being decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

-- State the theorem
theorem f_decreasing_iff_a_in_range (a : ℝ) :
  is_decreasing (f a) ↔ 0 < a ∧ a ≤ 1/3 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_a_in_range_l1893_189323


namespace NUMINAMATH_CALUDE_most_reasonable_sampling_methods_l1893_189376

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a sampling survey --/
structure Survey where
  totalItems : ℕ
  sampleSize : ℕ
  hasStrata : Bool
  hasStructure : Bool

/-- Determines the most reasonable sampling method for a given survey --/
def mostReasonableSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasStrata then SamplingMethod.Stratified
  else if s.hasStructure then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The three surveys described in the problem --/
def survey1 : Survey := { totalItems := 15, sampleSize := 5, hasStrata := false, hasStructure := false }
def survey2 : Survey := { totalItems := 240, sampleSize := 20, hasStrata := true, hasStructure := false }
def survey3 : Survey := { totalItems := 25 * 38, sampleSize := 25, hasStrata := false, hasStructure := true }

/-- Theorem stating the most reasonable sampling methods for the given surveys --/
theorem most_reasonable_sampling_methods :
  (mostReasonableSamplingMethod survey1 = SamplingMethod.SimpleRandom) ∧
  (mostReasonableSamplingMethod survey2 = SamplingMethod.Stratified) ∧
  (mostReasonableSamplingMethod survey3 = SamplingMethod.Systematic) :=
sorry


end NUMINAMATH_CALUDE_most_reasonable_sampling_methods_l1893_189376


namespace NUMINAMATH_CALUDE_final_time_and_sum_l1893_189302

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Converts a time to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  sorry

/-- Theorem: Given the starting time and duration, prove the final time and sum -/
theorem final_time_and_sum 
  (start : Time)
  (durationHours durationMinutes durationSeconds : Nat) : 
  start.hours = 3 ∧ start.minutes = 0 ∧ start.seconds = 0 →
  durationHours = 313 ∧ durationMinutes = 45 ∧ durationSeconds = 56 →
  let finalTime := to12HourFormat (addTime start durationHours durationMinutes durationSeconds)
  finalTime.hours = 4 ∧ finalTime.minutes = 45 ∧ finalTime.seconds = 56 ∧
  finalTime.hours + finalTime.minutes + finalTime.seconds = 105 :=
by sorry

end NUMINAMATH_CALUDE_final_time_and_sum_l1893_189302


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1893_189324

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - a^2) < 0} = {x : ℝ | a^2 < x ∧ x < a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1893_189324


namespace NUMINAMATH_CALUDE_prime_condition_l1893_189301

theorem prime_condition (p : ℕ) : 
  Prime p → Prime (p^4 - 3*p^2 + 9) → p = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_condition_l1893_189301


namespace NUMINAMATH_CALUDE_negative_sum_l1893_189395

theorem negative_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  b + c < 0 := by
sorry

end NUMINAMATH_CALUDE_negative_sum_l1893_189395


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_slope_product_l1893_189390

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem lines_perpendicular_iff_slope_product (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) :
  (∀ x y, A₁ * x + B₁ * y + C₁ = 0) ∧ (∀ x y, A₂ * x + B₂ * y + C₂ = 0) →
  (A₁ * A₂) / (B₁ * B₂) = -1 ↔ 
  (∀ x₁ y₁ x₂ y₂, A₁ * x₁ + B₁ * y₁ + C₁ = 0 ∧ A₂ * x₂ + B₂ * y₂ + C₂ = 0 →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (x₂ - x₁) = (y₂ - y₁) * (y₂ - y₁) ∧
     (x₂ - x₁) * (y₂ - y₁) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_slope_product_l1893_189390


namespace NUMINAMATH_CALUDE_fraction_addition_l1893_189318

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1893_189318


namespace NUMINAMATH_CALUDE_prob_at_least_three_primes_l1893_189366

/-- The number of sides on each die -/
def numSides : ℕ := 12

/-- The number of prime numbers on a 12-sided die -/
def numPrimes : ℕ := 5

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling a prime number on a single die -/
def probPrime : ℚ := numPrimes / numSides

/-- The probability of not rolling a prime number on a single die -/
def probNotPrime : ℚ := 1 - probPrime

/-- The probability of rolling at least three prime numbers on five dice -/
def probAtLeastThreePrimes : ℚ := 40625 / 622080

/-- Theorem stating the probability of rolling at least three prime numbers on five 12-sided dice -/
theorem prob_at_least_three_primes :
  (Finset.sum (Finset.range 3) (λ k => Nat.choose numDice k * probPrime ^ k * probNotPrime ^ (numDice - k))) = 1 - probAtLeastThreePrimes :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_three_primes_l1893_189366


namespace NUMINAMATH_CALUDE_min_value_function_l1893_189306

theorem min_value_function (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (6 * x^2 + 9 * x + 2 * y^2 + 3 * y + 20) / (9 * (x + y + 2)) ≥ 4 * Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l1893_189306


namespace NUMINAMATH_CALUDE_defective_and_shipped_percentage_l1893_189378

/-- The percentage of defective units produced -/
def defective_rate : ℝ := 0.08

/-- The percentage of defective units shipped -/
def shipped_rate : ℝ := 0.05

/-- The percentage of units that are both defective and shipped -/
def defective_and_shipped_rate : ℝ := defective_rate * shipped_rate

theorem defective_and_shipped_percentage :
  defective_and_shipped_rate = 0.004 := by sorry

end NUMINAMATH_CALUDE_defective_and_shipped_percentage_l1893_189378


namespace NUMINAMATH_CALUDE_equation_solution_l1893_189308

theorem equation_solution : 
  ∃ x : ℝ, (10 : ℝ)^(2*x) * (100 : ℝ)^(3*x) = (1000 : ℝ)^7 ∧ x = 21/8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1893_189308


namespace NUMINAMATH_CALUDE_courtyard_length_l1893_189328

/-- The length of a rectangular courtyard given its width and paving details. -/
theorem courtyard_length (width : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℕ) : 
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 16000 →
  width * (total_bricks : ℝ) * brick_length * brick_width / width = 20 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l1893_189328


namespace NUMINAMATH_CALUDE_michael_pet_sitting_cost_l1893_189357

/-- Calculates the total cost of pet sitting for one night -/
def pet_sitting_cost (num_cats num_dogs num_parrots num_fish : ℕ) 
                     (cost_per_cat cost_per_dog cost_per_parrot cost_per_fish : ℕ) : ℕ :=
  num_cats * cost_per_cat + 
  num_dogs * cost_per_dog + 
  num_parrots * cost_per_parrot + 
  num_fish * cost_per_fish

/-- Theorem: The total cost of pet sitting for Michael's pets for one night is $106 -/
theorem michael_pet_sitting_cost : 
  pet_sitting_cost 2 3 1 4 13 18 10 4 = 106 := by
  sorry

end NUMINAMATH_CALUDE_michael_pet_sitting_cost_l1893_189357


namespace NUMINAMATH_CALUDE_caterpillar_to_scorpion_ratio_l1893_189345

/-- Represents Calvin's bug collection -/
structure BugCollection where
  roaches : ℕ
  scorpions : ℕ
  crickets : ℕ
  caterpillars : ℕ

/-- Calvin's bug collection satisfies the given conditions -/
def calvins_collection : BugCollection where
  roaches := 12
  scorpions := 3
  crickets := 6  -- half as many crickets as roaches
  caterpillars := 6  -- to be proven

theorem caterpillar_to_scorpion_ratio (c : BugCollection) 
  (h1 : c.roaches = 12)
  (h2 : c.scorpions = 3)
  (h3 : c.crickets = c.roaches / 2)
  (h4 : c.roaches + c.scorpions + c.crickets + c.caterpillars = 27) :
  c.caterpillars / c.scorpions = 2 := by
  sorry

#check caterpillar_to_scorpion_ratio calvins_collection

end NUMINAMATH_CALUDE_caterpillar_to_scorpion_ratio_l1893_189345


namespace NUMINAMATH_CALUDE_triangle_problem_l1893_189382

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (2 * b + Real.sqrt 3 * c) * Real.cos A + Real.sqrt 3 * a * Real.cos C = 0 →
  A = 5 * π / 6 ∧
  (a = 2 → 
    ∃ (lower upper : ℝ), lower = 2 ∧ upper = 2 * Real.sqrt 3 ∧
    ∀ (x : ℝ), (∃ (b' c' : ℝ), b' + Real.sqrt 3 * c' = x ∧
      b' / (Real.sin B) = c' / (Real.sin C) ∧
      (2 * b' + Real.sqrt 3 * c') * Real.cos A + Real.sqrt 3 * a * Real.cos C = 0) →
    lower < x ∧ x < upper) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1893_189382


namespace NUMINAMATH_CALUDE_remainder_of_3n_mod_7_l1893_189358

theorem remainder_of_3n_mod_7 (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3n_mod_7_l1893_189358


namespace NUMINAMATH_CALUDE_complex_power_2009_l1893_189335

theorem complex_power_2009 (i : ℂ) (h : i^2 = -1) : i^2009 = i := by sorry

end NUMINAMATH_CALUDE_complex_power_2009_l1893_189335


namespace NUMINAMATH_CALUDE_square_diagonals_properties_l1893_189355

structure Square where
  diagonals_perpendicular : Prop
  diagonals_equal : Prop

theorem square_diagonals_properties (s : Square) :
  (s.diagonals_perpendicular ∨ s.diagonals_equal) ∧
  (s.diagonals_perpendicular ∧ s.diagonals_equal) ∧
  ¬(¬s.diagonals_perpendicular) := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_properties_l1893_189355


namespace NUMINAMATH_CALUDE_min_value_expression_l1893_189394

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ 6 * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b = 6 * Real.sqrt 3 ↔ 
    a^2 = 1/6 ∧ b = -1/(2*a) ∧ c = 2*a) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1893_189394


namespace NUMINAMATH_CALUDE_investment_profit_ratio_l1893_189307

/-- Represents the profit ratio between two investors based on their capital and investment duration. -/
def profit_ratio (capital_a capital_b : ℕ) (duration_a duration_b : ℚ) : ℚ × ℚ :=
  let contribution_a := capital_a * duration_a
  let contribution_b := capital_b * duration_b
  (contribution_a, contribution_b)

/-- Theorem stating that given the specified investments and durations, the profit ratio is 2:1. -/
theorem investment_profit_ratio :
  let (ratio_a, ratio_b) := profit_ratio 27000 36000 12 (9/2)
  ratio_a / ratio_b = 2 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_ratio_l1893_189307


namespace NUMINAMATH_CALUDE_investment_profit_sharing_l1893_189389

/-- Represents the capital contribution of an investor over a year -/
def capital_contribution (initial_investment : ℕ) (doubled_after_six_months : Bool) : ℕ :=
  if doubled_after_six_months
  then initial_investment * 6 + (initial_investment * 2) * 6
  else initial_investment * 12

/-- Represents the profit-sharing ratio between two investors -/
def profit_sharing_ratio (a_contribution : ℕ) (b_contribution : ℕ) : Prop :=
  a_contribution = b_contribution

theorem investment_profit_sharing :
  let a_initial_investment : ℕ := 3000
  let b_initial_investment : ℕ := 4500
  let a_doubles_capital : Bool := true
  let b_doubles_capital : Bool := false
  
  let a_contribution := capital_contribution a_initial_investment a_doubles_capital
  let b_contribution := capital_contribution b_initial_investment b_doubles_capital
  
  profit_sharing_ratio a_contribution b_contribution :=
by
  sorry

end NUMINAMATH_CALUDE_investment_profit_sharing_l1893_189389


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1893_189399

theorem contrapositive_equivalence (x : ℝ) :
  (¬ (-2 < x ∧ x < 2) → ¬ (x^2 < 4)) ↔ ((x ≤ -2 ∨ x ≥ 2) → x^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1893_189399


namespace NUMINAMATH_CALUDE_kenya_peanuts_l1893_189353

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_difference : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_difference = 48) :
  jose_peanuts + kenya_difference = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l1893_189353


namespace NUMINAMATH_CALUDE_sum_repeating_decimals_eq_l1893_189361

/-- The sum of the repeating decimals 0.141414... and 0.272727... -/
def sum_repeating_decimals : ℚ :=
  let a : ℚ := 14 / 99  -- 0.141414...
  let b : ℚ := 27 / 99  -- 0.272727...
  a + b

/-- Theorem: The sum of the repeating decimals 0.141414... and 0.272727... is 41/99 -/
theorem sum_repeating_decimals_eq :
  sum_repeating_decimals = 41 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_repeating_decimals_eq_l1893_189361


namespace NUMINAMATH_CALUDE_green_hat_cost_l1893_189341

/-- Proves the cost of green hats given the total number of hats, cost of blue hats, 
    total price, and number of green hats. -/
theorem green_hat_cost 
  (total_hats : ℕ) 
  (blue_hat_cost : ℕ) 
  (total_price : ℕ) 
  (green_hats : ℕ) 
  (h1 : total_hats = 85) 
  (h2 : blue_hat_cost = 6) 
  (h3 : total_price = 540) 
  (h4 : green_hats = 30) : 
  (total_price - blue_hat_cost * (total_hats - green_hats)) / green_hats = 7 := by
  sorry

end NUMINAMATH_CALUDE_green_hat_cost_l1893_189341


namespace NUMINAMATH_CALUDE_aquarium_illness_percentage_l1893_189377

theorem aquarium_illness_percentage (total_visitors : ℕ) (healthy_visitors : ℕ) : 
  total_visitors = 500 → 
  healthy_visitors = 300 → 
  (total_visitors - healthy_visitors : ℚ) / total_visitors * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_illness_percentage_l1893_189377


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1893_189397

theorem quadratic_maximum : 
  (∀ s : ℝ, -3 * s^2 + 36 * s + 7 ≤ 115) ∧ 
  (∃ s : ℝ, -3 * s^2 + 36 * s + 7 = 115) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1893_189397


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l1893_189327

theorem belt_and_road_population_scientific_notation :
  let billion : ℝ := 10^9
  4.4 * billion = 4.4 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l1893_189327


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1893_189365

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 + 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1893_189365


namespace NUMINAMATH_CALUDE_valid_paths_count_l1893_189388

/-- Represents the grid dimensions -/
structure GridDimensions where
  width : Nat
  height : Nat

/-- Represents a forbidden vertical segment -/
structure ForbiddenSegment where
  x : Nat
  y_start : Nat
  y_end : Nat

/-- Calculates the number of valid paths on the grid -/
def countValidPaths (grid : GridDimensions) (forbidden : List ForbiddenSegment) : Nat :=
  sorry

/-- The main theorem stating the number of valid paths -/
theorem valid_paths_count :
  let grid := GridDimensions.mk 10 4
  let forbidden := [
    ForbiddenSegment.mk 5 1 3,
    ForbiddenSegment.mk 6 1 3
  ]
  countValidPaths grid forbidden = 329 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l1893_189388


namespace NUMINAMATH_CALUDE_soccer_team_goals_l1893_189364

theorem soccer_team_goals (total_players : ℕ) (total_goals : ℕ) (games_played : ℕ) 
  (h1 : total_players = 24)
  (h2 : total_goals = 150)
  (h3 : games_played = 15)
  (h4 : (total_players / 3) * games_played = total_goals - 30) : 
  30 = total_goals - (total_players / 3) * games_played := by
sorry

end NUMINAMATH_CALUDE_soccer_team_goals_l1893_189364


namespace NUMINAMATH_CALUDE_smallest_special_number_l1893_189351

def is_special (n : ℕ) : Prop :=
  (n > 3429) ∧ (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number :
  ∀ m : ℕ, is_special m → m ≥ 3450 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l1893_189351


namespace NUMINAMATH_CALUDE_A_3_1_l1893_189348

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_1 : A 3 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_A_3_1_l1893_189348


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1893_189347

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : b = (a + c) / 2) (h5 : b^2 = a^2 - c^2) : 
  let e := c / a
  0 < e ∧ e < 1 ∧ e = 3/5 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1893_189347


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1893_189386

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
    common difference d = 2, and a₅ = 10, prove that S₁₀ = 110 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * 2)) →  -- sum formula
  a 5 = 10 →  -- given condition
  S 10 = 110 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1893_189386


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l1893_189326

/-- Given a positive integer, returns the number obtained by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_by_eleven (A : ℕ) (h : A > 0) :
  let B := reverse_digits A
  (11 ∣ (A + B)) ∨ (11 ∣ (A - B)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l1893_189326


namespace NUMINAMATH_CALUDE_cost_per_book_is_5_l1893_189319

/-- The cost to produce each book -/
def cost_per_book : ℝ := 5

/-- The selling price of each book -/
def selling_price : ℝ := 20

/-- The total profit -/
def total_profit : ℝ := 120

/-- The number of customers -/
def num_customers : ℕ := 4

/-- The number of books each customer buys -/
def books_per_customer : ℕ := 2

/-- The theorem stating the cost to make each book -/
theorem cost_per_book_is_5 : 
  cost_per_book = 5 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_book_is_5_l1893_189319


namespace NUMINAMATH_CALUDE_base_of_negative_four_cubed_l1893_189396

def power_expression : ℤ → ℕ → ℤ := (·^·)

theorem base_of_negative_four_cubed :
  ∃ (base : ℤ), power_expression base 3 = power_expression (-4) 3 ∧ base = -4 :=
sorry

end NUMINAMATH_CALUDE_base_of_negative_four_cubed_l1893_189396


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l1893_189310

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define the largest prime factor function
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_sum_of_divisors_450 :
  largest_prime_factor (sum_of_divisors 450) = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l1893_189310


namespace NUMINAMATH_CALUDE_college_students_count_l1893_189374

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 6) (h2 : girls = 200) :
  boys + girls = 440 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l1893_189374


namespace NUMINAMATH_CALUDE_range_of_t_l1893_189387

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, (|x - t| < 1 → 1 < x ∧ x ≤ 4)) →
  (2 ≤ t ∧ t ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l1893_189387


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l1893_189393

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (different : Line → Line → Prop)
variable (non_coincident : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : different m n) 
  (h2 : non_coincident α β) 
  (h3 : perpendicular m α) 
  (h4 : parallel m β) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l1893_189393


namespace NUMINAMATH_CALUDE_dvd_cost_l1893_189368

/-- Given that two identical DVDs cost $36, prove that six of these DVDs cost $108. -/
theorem dvd_cost (two_dvd_cost : ℕ) (h : two_dvd_cost = 36) : 
  (6 * two_dvd_cost / 2 : ℚ) = 108 := by sorry

end NUMINAMATH_CALUDE_dvd_cost_l1893_189368


namespace NUMINAMATH_CALUDE_tangent_chord_distance_l1893_189349

theorem tangent_chord_distance (R a : ℝ) (h : R > 0) :
  let x := R
  let m := 2 * R
  16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_chord_distance_l1893_189349


namespace NUMINAMATH_CALUDE_paint_intensity_after_replacement_l1893_189340

/-- Calculates the new paint intensity after partial replacement -/
def new_paint_intensity (initial_intensity : ℝ) (replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

/-- Theorem: Given the specified conditions, the new paint intensity is 0.4 (40%) -/
theorem paint_intensity_after_replacement :
  let initial_intensity : ℝ := 0.5
  let replacement_intensity : ℝ := 0.25
  let replacement_fraction : ℝ := 0.4
  new_paint_intensity initial_intensity replacement_intensity replacement_fraction = 0.4 := by
sorry

#eval new_paint_intensity 0.5 0.25 0.4

end NUMINAMATH_CALUDE_paint_intensity_after_replacement_l1893_189340


namespace NUMINAMATH_CALUDE_division_equality_l1893_189372

theorem division_equality : (124 : ℚ) / (8 + 14 * 3) = 62 / 25 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l1893_189372


namespace NUMINAMATH_CALUDE_transaction_conservation_l1893_189333

/-- Represents the transaction in the restaurant problem -/
structure RestaurantTransaction where
  initial_payment : ℕ
  people : ℕ
  overcharge : ℕ
  refund_per_person : ℕ
  assistant_kept : ℕ

/-- The actual cost of the meal -/
def actual_cost (t : RestaurantTransaction) : ℕ :=
  t.initial_payment - t.overcharge

/-- The amount effectively paid by the customers -/
def effective_payment (t : RestaurantTransaction) : ℕ :=
  t.people * (t.initial_payment / t.people - t.refund_per_person)

/-- Theorem stating that the total amount involved is conserved -/
theorem transaction_conservation (t : RestaurantTransaction) 
  (h1 : t.initial_payment = 30)
  (h2 : t.people = 3)
  (h3 : t.overcharge = 5)
  (h4 : t.refund_per_person = 1)
  (h5 : t.assistant_kept = 2) :
  effective_payment t + (t.people * t.refund_per_person) + t.assistant_kept = t.initial_payment := by
  sorry

#check transaction_conservation

end NUMINAMATH_CALUDE_transaction_conservation_l1893_189333


namespace NUMINAMATH_CALUDE_f_analytical_expression_k_range_for_monotonicity_l1893_189331

-- Part 1
def f₁ (x : ℝ) := x^2 - 3*x + 2

theorem f_analytical_expression :
  ∀ x, f₁ (x + 1) = x^2 - 3*x + 2 →
  ∃ g : ℝ → ℝ, (∀ x, g x = x^2 - 6*x + 6) ∧ (∀ x, g x = f₁ x) :=
sorry

-- Part 2
def f₂ (k : ℝ) (x : ℝ) := x^2 - 2*k*x - 8

theorem k_range_for_monotonicity :
  ∀ k, (∀ x ∈ Set.Icc 1 4, Monotone (f₂ k)) →
  k ≥ 4 ∨ k ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_analytical_expression_k_range_for_monotonicity_l1893_189331


namespace NUMINAMATH_CALUDE_triangle_value_l1893_189369

theorem triangle_value (p : ℝ) (h1 : ∃ triangle : ℝ, triangle + p = 67) 
  (h2 : 3 * (67) - p = 185) : 
  ∃ triangle : ℝ, triangle = 51 ∧ triangle + p = 67 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l1893_189369


namespace NUMINAMATH_CALUDE_factor_expression_l1893_189354

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = 18 * x^5 * (4 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1893_189354


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1893_189384

/-- Given a hyperbola with the specified properties, prove its equation is x²/8 - y²/8 = 1 -/
theorem hyperbola_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (c / a = Real.sqrt 2) →                   -- Eccentricity is √2
  (4 / c = 1) →                             -- Slope of line through F(-c,0) and P(0,4) is 1
  (a = b) →                                 -- Equilateral hyperbola
  (∀ x y : ℝ, x^2 / 8 - y^2 / 8 = 1) :=     -- Resulting equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1893_189384


namespace NUMINAMATH_CALUDE_factory_equation_holds_l1893_189300

/-- Represents a factory's part processing scenario -/
def factory_scenario (x : ℝ) : Prop :=
  x > 0 ∧ 
  (100 / x) + (400 / (2 * x)) = 6

/-- Theorem stating the equation holds for the given scenario -/
theorem factory_equation_holds : 
  ∀ x : ℝ, x > 0 → (100 / x) + (400 / (2 * x)) = 6 ↔ factory_scenario x :=
by
  sorry

#check factory_equation_holds

end NUMINAMATH_CALUDE_factory_equation_holds_l1893_189300


namespace NUMINAMATH_CALUDE_not_parabola_l1893_189315

theorem not_parabola (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  ¬∃ (a b c : Real), ∀ (x y : Real),
    x^2 * Real.sin α + y^2 * Real.cos α = 1 ↔ y = a*x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l1893_189315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1893_189342

/-- An arithmetic sequence {a_n} with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3 + a 4 + a 5 = 12 →
  a 6 = 2 →
  a 2 + a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1893_189342


namespace NUMINAMATH_CALUDE_additional_marbles_needed_l1893_189334

def friends : ℕ := 12
def current_marbles : ℕ := 50

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem additional_marbles_needed : 
  sum_first_n friends - current_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_needed_l1893_189334


namespace NUMINAMATH_CALUDE_triangle_side_value_l1893_189367

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b = a + c, B = 30°, and the area is 3/2, then b = 1 + √3 -/
theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  2 * b = a + c →
  B = π / 6 →
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  b = 1 + Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_value_l1893_189367


namespace NUMINAMATH_CALUDE_prove_initial_person_count_l1893_189373

/-- The initial number of persons in a group where:
  - The average weight increase is 4.2 kg when a new person replaces one of the original group.
  - The weight of the person leaving is 65 kg.
  - The weight of the new person is 98.6 kg.
-/
def initialPersonCount : ℕ := 8

theorem prove_initial_person_count :
  let avgWeightIncrease : ℚ := 21/5
  let oldPersonWeight : ℚ := 65
  let newPersonWeight : ℚ := 493/5
  (newPersonWeight - oldPersonWeight) / avgWeightIncrease = initialPersonCount := by
  sorry

end NUMINAMATH_CALUDE_prove_initial_person_count_l1893_189373


namespace NUMINAMATH_CALUDE_circular_film_radius_l1893_189316

/-- The radius of a circular film formed by a non-mixing liquid on water -/
theorem circular_film_radius 
  (volume : ℝ) 
  (thickness : ℝ) 
  (radius : ℝ) 
  (h1 : volume = 400) 
  (h2 : thickness = 0.2) 
  (h3 : π * radius^2 * thickness = volume) : 
  radius = Real.sqrt (2000 / π) := by
sorry

end NUMINAMATH_CALUDE_circular_film_radius_l1893_189316


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1893_189370

theorem arithmetic_expression_equality : (4 + 6 * 3) - (2 * 3) + 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1893_189370


namespace NUMINAMATH_CALUDE_smallest_number_with_weight_2000_l1893_189305

/-- The weight of a number is the sum of its digits -/
def weight (n : ℕ) : ℕ := sorry

/-- Construct a number with a leading digit followed by a sequence of nines -/
def constructNumber (lead : ℕ) (nines : ℕ) : ℕ := sorry

theorem smallest_number_with_weight_2000 :
  ∀ n : ℕ, weight n = 2000 → n ≥ constructNumber 2 222 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_weight_2000_l1893_189305


namespace NUMINAMATH_CALUDE_parabola_min_y_l1893_189339

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  y + x = (y - x)^2 + 3*(y - x) + 3

/-- Theorem stating the minimum value of y for points on the parabola -/
theorem parabola_min_y :
  (∀ x y : ℝ, parabola_eq x y → y ≥ -1/2) ∧
  (∃ x y : ℝ, parabola_eq x y ∧ y = -1/2) := by
sorry

end NUMINAMATH_CALUDE_parabola_min_y_l1893_189339


namespace NUMINAMATH_CALUDE_sand_received_by_city_c_l1893_189398

/-- The amount of sand received by City C given the total sand and amounts received by other cities -/
theorem sand_received_by_city_c 
  (total : ℝ) 
  (city_a : ℝ) 
  (city_b : ℝ) 
  (city_d : ℝ) 
  (h_total : total = 95) 
  (h_city_a : city_a = 16.5) 
  (h_city_b : city_b = 26) 
  (h_city_d : city_d = 28) : 
  total - (city_a + city_b + city_d) = 24.5 := by
sorry

end NUMINAMATH_CALUDE_sand_received_by_city_c_l1893_189398


namespace NUMINAMATH_CALUDE_find_number_l1893_189371

theorem find_number (G N : ℕ) (h1 : G = 4) (h2 : N % G = 6) (h3 : 1856 % G = 4) : N = 1862 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1893_189371


namespace NUMINAMATH_CALUDE_hausdorff_dim_countable_union_l1893_189338

open MeasureTheory

-- Define a countable collection of sets
variable {α : Type*} [MeasurableSpace α]
variable (A : ℕ → Set α)

-- Define Hausdorff dimension
noncomputable def hausdorffDim (S : Set α) : ℝ := sorry

-- State the theorem
theorem hausdorff_dim_countable_union :
  hausdorffDim (⋃ i, A i) = ⨆ i, hausdorffDim (A i) := by sorry

end NUMINAMATH_CALUDE_hausdorff_dim_countable_union_l1893_189338


namespace NUMINAMATH_CALUDE_point_positions_l1893_189309

/-- Define a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point is in the first octant -/
def isInFirstOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.z > 0

/-- Check if a point is in the second octant -/
def isInSecondOctant (p : Point3D) : Prop :=
  p.x < 0 ∧ p.y > 0 ∧ p.z > 0

/-- Check if a point is in the eighth octant -/
def isInEighthOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y < 0 ∧ p.z < 0

/-- Check if a point lies in the YOZ plane -/
def isInYOZPlane (p : Point3D) : Prop :=
  p.x = 0

/-- Check if a point lies on the OY axis -/
def isOnOYAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.z = 0

/-- Check if a point is at the origin -/
def isAtOrigin (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0 ∧ p.z = 0

theorem point_positions :
  let A : Point3D := ⟨3, 2, 6⟩
  let B : Point3D := ⟨-2, 3, 1⟩
  let C : Point3D := ⟨1, -4, -2⟩
  let D : Point3D := ⟨1, -2, -1⟩
  let E : Point3D := ⟨0, 4, 1⟩
  let F : Point3D := ⟨0, 2, 0⟩
  let P : Point3D := ⟨0, 0, 0⟩
  isInFirstOctant A ∧
  isInSecondOctant B ∧
  isInEighthOctant C ∧
  isInEighthOctant D ∧
  isInYOZPlane E ∧
  isOnOYAxis F ∧
  isAtOrigin P := by
  sorry

end NUMINAMATH_CALUDE_point_positions_l1893_189309


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l1893_189321

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l1893_189321


namespace NUMINAMATH_CALUDE_difference_2010th_2008th_triangular_l1893_189325

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_2010th_2008th_triangular : 
  triangular_number 2010 - triangular_number 2008 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_difference_2010th_2008th_triangular_l1893_189325


namespace NUMINAMATH_CALUDE_tank_weight_calculation_l1893_189329

def tank_capacity : ℝ := 200
def empty_tank_weight : ℝ := 80
def fill_percentage : ℝ := 0.8
def water_weight_per_gallon : ℝ := 8

theorem tank_weight_calculation : 
  let water_volume : ℝ := tank_capacity * fill_percentage
  let water_weight : ℝ := water_volume * water_weight_per_gallon
  let total_weight : ℝ := empty_tank_weight + water_weight
  total_weight = 1360 := by sorry

end NUMINAMATH_CALUDE_tank_weight_calculation_l1893_189329


namespace NUMINAMATH_CALUDE_trig_expression_value_l1893_189385

theorem trig_expression_value (α : ℝ) (h : Real.tan α = -2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / Real.sin α ^ 2 = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l1893_189385


namespace NUMINAMATH_CALUDE_average_increase_l1893_189343

theorem average_increase (x₁ x₂ x₃ : ℝ) :
  (x₁ + x₂ + x₃) / 3 = 5 →
  ((x₁ + 2) + (x₂ + 2) + (x₃ + 2)) / 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_average_increase_l1893_189343


namespace NUMINAMATH_CALUDE_gcd_cube_plus_eight_and_n_plus_three_l1893_189383

theorem gcd_cube_plus_eight_and_n_plus_three (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 + 2^3) (n + 3) = 9 := by sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_eight_and_n_plus_three_l1893_189383


namespace NUMINAMATH_CALUDE_cross_section_area_theorem_l1893_189336

/-- Regular quadrilateral prism with given dimensions -/
structure RegularQuadrilateralPrism where
  a : ℝ
  base_edge : ℝ
  height : ℝ
  h_base_edge : base_edge = a
  h_height : height = 2 * a

/-- Plane passing through diagonal B₁D₁ and midpoint of edge DC -/
structure CuttingPlane (prism : RegularQuadrilateralPrism) where
  diagonal : ℝ × ℝ × ℝ
  midpoint : ℝ × ℝ × ℝ
  h_diagonal : diagonal = (prism.a, prism.a, prism.height)
  h_midpoint : midpoint = (prism.a / 2, prism.a, 0)

/-- Area of cross-section created by cutting plane -/
noncomputable def cross_section_area (prism : RegularQuadrilateralPrism) (plane : CuttingPlane prism) : ℝ :=
  (3 * prism.a^2 * Real.sqrt 33) / 8

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_theorem (prism : RegularQuadrilateralPrism) (plane : CuttingPlane prism) :
  cross_section_area prism plane = (3 * prism.a^2 * Real.sqrt 33) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_theorem_l1893_189336


namespace NUMINAMATH_CALUDE_garden_width_is_eleven_l1893_189344

/-- Represents a rectangular garden with specific dimensions. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 2
  perimeter_formula : perimeter = 2 * (length + width)

/-- Theorem: The width of a rectangular garden with perimeter 48m and length 2m more than width is 11m. -/
theorem garden_width_is_eleven (garden : RectangularGarden) 
    (h_perimeter : garden.perimeter = 48) : garden.width = 11 := by
  sorry

#check garden_width_is_eleven

end NUMINAMATH_CALUDE_garden_width_is_eleven_l1893_189344


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1893_189352

theorem quadratic_equation_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   a * x₁^2 - (3*a + 1) * x₁ + 2*(a + 1) = 0 ∧
   a * x₂^2 - (3*a + 1) * x₂ + 2*(a + 1) = 0 ∧
   x₁ - x₁*x₂ + x₂ = 1 - a) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1893_189352


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l1893_189379

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = b * Complex.I) →  -- z is purely imaginary
  (∃ r : ℝ, (z + 2) / (1 + Complex.I) = r) →  -- (z+2)/(1+i) is real
  z = -2 * Complex.I :=  -- z = -2i
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l1893_189379


namespace NUMINAMATH_CALUDE_count_pairs_theorem_l1893_189356

/-- The number of integer pairs (m, n) satisfying the given inequality -/
def count_pairs : ℕ := 1000

/-- The lower bound for m -/
def m_lower_bound : ℕ := 1

/-- The upper bound for m -/
def m_upper_bound : ℕ := 3000

/-- Predicate to check if a pair (m, n) satisfies the inequality -/
def satisfies_inequality (m n : ℕ) : Prop :=
  (5 : ℝ)^n < (3 : ℝ)^m ∧ (3 : ℝ)^m < (3 : ℝ)^(m+1) ∧ (3 : ℝ)^(m+1) < (5 : ℝ)^(n+1)

theorem count_pairs_theorem :
  ∃ S : Finset (ℕ × ℕ),
    S.card = count_pairs ∧
    (∀ (m n : ℕ), (m, n) ∈ S ↔ 
      m_lower_bound ≤ m ∧ m ≤ m_upper_bound ∧ satisfies_inequality m n) :=
sorry

end NUMINAMATH_CALUDE_count_pairs_theorem_l1893_189356


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1893_189304

theorem cubic_equation_solution :
  ∃! x : ℝ, (x^3 - x^2) / (x^2 + 3*x + 2) + x = -3 ∧ x ≠ -1 ∧ x ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1893_189304


namespace NUMINAMATH_CALUDE_colored_paper_count_l1893_189362

theorem colored_paper_count (used left : ℕ) (h1 : used = 9) (h2 : left = 12) :
  used + left = 21 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_count_l1893_189362


namespace NUMINAMATH_CALUDE_schur_like_inequality_l1893_189392

theorem schur_like_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) ≥ a + b + c :=
sorry

end NUMINAMATH_CALUDE_schur_like_inequality_l1893_189392


namespace NUMINAMATH_CALUDE_initial_number_proof_l1893_189332

theorem initial_number_proof : ∃ (N : ℕ), N > 0 ∧ (N - 10) % 21 = 0 ∧ ∀ (M : ℕ), M > 0 → (M - 10) % 21 = 0 → M ≥ N := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1893_189332


namespace NUMINAMATH_CALUDE_program_output_is_twenty_l1893_189317

/-- The result of evaluating the arithmetic expression (3+2)*4 -/
def program_result : ℕ := (3 + 2) * 4

/-- Theorem stating that the result of the program is 20 -/
theorem program_output_is_twenty : program_result = 20 := by
  sorry

end NUMINAMATH_CALUDE_program_output_is_twenty_l1893_189317


namespace NUMINAMATH_CALUDE_roots_equation_r_value_l1893_189360

theorem roots_equation_r_value (m p : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  (r = 16/3) := by
sorry

end NUMINAMATH_CALUDE_roots_equation_r_value_l1893_189360


namespace NUMINAMATH_CALUDE_additional_toothpicks_3_to_5_l1893_189350

/-- The number of toothpicks needed for a staircase of n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else if n = 2 then 10
  else if n = 3 then 18
  else toothpicks (n - 1) + 2 * n + 2

theorem additional_toothpicks_3_to_5 :
  toothpicks 5 - toothpicks 3 = 22 :=
sorry

end NUMINAMATH_CALUDE_additional_toothpicks_3_to_5_l1893_189350


namespace NUMINAMATH_CALUDE_bookstore_sales_l1893_189314

/-- Given a store that sold 72 books and has a ratio of books to bookmarks sold of 9:2,
    prove that the number of bookmarks sold is 16. -/
theorem bookstore_sales (books_sold : ℕ) (book_ratio : ℕ) (bookmark_ratio : ℕ) 
    (h1 : books_sold = 72)
    (h2 : book_ratio = 9)
    (h3 : bookmark_ratio = 2) :
    (books_sold * bookmark_ratio) / book_ratio = 16 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_l1893_189314


namespace NUMINAMATH_CALUDE_circle_combined_value_l1893_189311

/-- The combined value of circumference and area for a circle with radius 13 cm -/
theorem circle_combined_value :
  let r : ℝ := 13
  let π : ℝ := Real.pi
  let circumference : ℝ := 2 * π * r
  let area : ℝ := π * r^2
  abs ((circumference + area) - 612.6105) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_circle_combined_value_l1893_189311


namespace NUMINAMATH_CALUDE_min_even_integers_l1893_189337

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 24 →
  a + b + c + d = 39 →
  a + b + c + d + e + f = 58 →
  ∃ (count : ℕ), count ≥ 2 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) ∧
    ∀ (other_count : ℕ), 
      other_count = (if Even a then 1 else 0) + 
                    (if Even b then 1 else 0) + 
                    (if Even c then 1 else 0) + 
                    (if Even d then 1 else 0) + 
                    (if Even e then 1 else 0) + 
                    (if Even f then 1 else 0) →
      other_count ≥ count := by
sorry

end NUMINAMATH_CALUDE_min_even_integers_l1893_189337


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l1893_189330

theorem unique_solution_to_equation :
  ∃! y : ℝ, y ≠ 2 ∧ y ≠ -2 ∧
  (-12 * y) / (y^2 - 4) = (3 * y) / (y + 2) - 9 / (y - 2) ∧
  y = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l1893_189330


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l1893_189375

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear

/-- The proportion of angular speeds for a system of four meshed gears -/
def angular_speed_proportion (g : GearSystem) : Prop :=
  ∃ (k : ℝ), k > 0 ∧
    g.A.speed = k * g.B.teeth * g.C.teeth * g.D.teeth ∧
    g.B.speed = k * g.A.teeth * g.C.teeth * g.D.teeth ∧
    g.C.speed = k * g.A.teeth * g.B.teeth * g.D.teeth ∧
    g.D.speed = k * g.A.teeth * g.B.teeth * g.C.teeth

theorem gear_speed_proportion (g : GearSystem) :
  angular_speed_proportion g → True :=
by
  sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l1893_189375


namespace NUMINAMATH_CALUDE_aquarium_length_l1893_189391

theorem aquarium_length (L : ℝ) : 
  L > 0 → 
  3 * (1/4 * L * 6 * 3) = 54 → 
  L = 4 := by
sorry

end NUMINAMATH_CALUDE_aquarium_length_l1893_189391


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1893_189363

theorem inequality_equivalence (a b c : ℕ+) :
  (∀ x y z : ℝ, (x - y) ^ a.val * (x - z) ^ b.val * (y - z) ^ c.val ≥ 0) ↔ 
  (∃ m n p : ℕ, a.val = 2 * m ∧ b.val = 2 * n ∧ c.val = 2 * p) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1893_189363


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1893_189322

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k * x₁^2 - (2*k + 4) * x₁ + (k - 6) = 0 ∧ 
    k * x₂^2 - (2*k + 4) * x₂ + (k - 6) = 0) ↔ 
  (k > -2/5 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1893_189322


namespace NUMINAMATH_CALUDE_max_price_correct_optimal_price_correct_max_profit_correct_l1893_189303

/-- Represents the beverage pricing and sales model for a food company. -/
structure BeverageModel where
  initial_price : ℝ
  initial_cost : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  marketing_cost : ℝ → ℝ
  sales_decrease : ℝ → ℝ

/-- The maximum price that ensures the total profit is not lower than the initial profit. -/
def max_price (model : BeverageModel) : ℝ :=
  model.initial_price + 5

/-- The price that maximizes the total profit under the new marketing strategy. -/
def optimal_price (model : BeverageModel) : ℝ := 19

/-- The maximum total profit under the new marketing strategy. -/
def max_profit (model : BeverageModel) : ℝ := 45.45

/-- Theorem stating the correctness of the maximum price. -/
theorem max_price_correct (model : BeverageModel) 
  (h1 : model.initial_price = 15)
  (h2 : model.initial_cost = 10)
  (h3 : model.initial_sales = 80000)
  (h4 : model.price_sensitivity = 8000) :
  max_price model = 20 := by sorry

/-- Theorem stating the correctness of the optimal price for maximum profit. -/
theorem optimal_price_correct (model : BeverageModel) 
  (h1 : model.initial_cost = 10)
  (h2 : ∀ x, x ≥ 16 → model.marketing_cost x = (33/4) * (x - 16))
  (h3 : ∀ x, model.sales_decrease x = 0.8 / ((x - 15)^2)) :
  optimal_price model = 19 := by sorry

/-- Theorem stating the correctness of the maximum total profit. -/
theorem max_profit_correct (model : BeverageModel) 
  (h1 : model.initial_cost = 10)
  (h2 : ∀ x, x ≥ 16 → model.marketing_cost x = (33/4) * (x - 16))
  (h3 : ∀ x, model.sales_decrease x = 0.8 / ((x - 15)^2)) :
  max_profit model = 45.45 := by sorry

end NUMINAMATH_CALUDE_max_price_correct_optimal_price_correct_max_profit_correct_l1893_189303


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1893_189359

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_a6 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 = 1 → a 7 = 16 → a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1893_189359


namespace NUMINAMATH_CALUDE_three_not_in_range_iff_c_gt_four_l1893_189312

/-- The function g(x) = x^2 + 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c

/-- 3 is not in the range of g(x) if and only if c > 4 -/
theorem three_not_in_range_iff_c_gt_four (c : ℝ) :
  (∀ x : ℝ, g c x ≠ 3) ↔ c > 4 := by
  sorry

end NUMINAMATH_CALUDE_three_not_in_range_iff_c_gt_four_l1893_189312


namespace NUMINAMATH_CALUDE_intersection_symmetry_l1893_189320

/-- The line y = ax + 1 intersects the curve x^2 + y^2 + bx - y = 1 at two points
    which are symmetric about the line x + y = 0. -/
theorem intersection_symmetry (a b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Line equation
    (y₁ = a * x₁ + 1) ∧ (y₂ = a * x₂ + 1) ∧
    -- Curve equation
    (x₁^2 + y₁^2 + b * x₁ - y₁ = 1) ∧ (x₂^2 + y₂^2 + b * x₂ - y₂ = 1) ∧
    -- Symmetry condition
    (x₁ + y₁ = -(x₂ + y₂)) ∧
    -- Distinct points
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_symmetry_l1893_189320


namespace NUMINAMATH_CALUDE_right_triangle_30_deg_side_half_hypotenuse_l1893_189381

/-- Theorem: In a right-angled triangle with one angle of 30°, 
    the length of the side opposite to the 30° angle is equal to 
    half the length of the hypotenuse. -/
theorem right_triangle_30_deg_side_half_hypotenuse 
  (A B C : ℝ × ℝ) -- Three points representing the vertices of the triangle
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) -- Right angle condition
  (angle_30_deg : ∃ i j k, i^2 + j^2 = k^2 ∧ i / k = 1 / 2) -- 30° angle condition
  : ∃ side hypotenuse, side = hypotenuse / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_30_deg_side_half_hypotenuse_l1893_189381


namespace NUMINAMATH_CALUDE_incorrect_absolute_value_expression_l1893_189346

theorem incorrect_absolute_value_expression : 
  ((-|5|)^2 = 25) ∧ 
  (|((-5)^2)| = 25) ∧ 
  ((-|5|)^2 = 25) ∧ 
  ¬((|(-5)|)^2 = 25) := by sorry

end NUMINAMATH_CALUDE_incorrect_absolute_value_expression_l1893_189346


namespace NUMINAMATH_CALUDE_abs_negative_eleven_l1893_189380

theorem abs_negative_eleven : abs (-11 : ℤ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_eleven_l1893_189380
