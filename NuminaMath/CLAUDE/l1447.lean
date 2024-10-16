import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1447_144740

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁ ≠ x₂ ∧
   x₁^2 + (m+2)*x₁ + m + 5 = 0 ∧
   x₂^2 + (m+2)*x₂ + m + 5 = 0) →
  -5 < m ∧ m ≤ -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1447_144740


namespace NUMINAMATH_CALUDE_omars_kite_height_l1447_144732

/-- Omar's kite raising problem -/
theorem omars_kite_height 
  (omar_time : ℝ) 
  (jasper_rate_multiplier : ℝ) 
  (jasper_height : ℝ) 
  (jasper_time : ℝ) :
  omar_time = 12 →
  jasper_rate_multiplier = 3 →
  jasper_height = 600 →
  jasper_time = 10 →
  (omar_time * (jasper_height / jasper_time) / jasper_rate_multiplier) = 240 := by
  sorry

#check omars_kite_height

end NUMINAMATH_CALUDE_omars_kite_height_l1447_144732


namespace NUMINAMATH_CALUDE_range_of_a_l1447_144729

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x ≥ a
def q (x : ℝ) : Prop := |x - 1| < 1

-- Define the property that p is necessary but not sufficient for q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  necessary_not_sufficient a → a ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1447_144729


namespace NUMINAMATH_CALUDE_silverware_probability_l1447_144717

/-- The probability of selecting one fork, one spoon, and one knife when
    randomly removing three pieces of silverware from a drawer. -/
theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 6)
  (h2 : spoons = 8)
  (h3 : knives = 6) :
  (forks * spoons * knives : ℚ) / (Nat.choose (forks + spoons + knives) 3) = 24 / 95 :=
by sorry

end NUMINAMATH_CALUDE_silverware_probability_l1447_144717


namespace NUMINAMATH_CALUDE_cubic_equation_root_magnitude_l1447_144742

theorem cubic_equation_root_magnitude (k : ℝ) : 
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  (k = -1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_magnitude_l1447_144742


namespace NUMINAMATH_CALUDE_basketball_competition_probabilities_l1447_144736

/-- Represents a team in the basketball competition -/
inductive Team : Type
| A
| B
| C

/-- The probability of one team winning against another -/
def win_probability (winner loser : Team) : ℚ :=
  match winner, loser with
  | Team.A, Team.B => 2/3
  | Team.A, Team.C => 2/3
  | Team.B, Team.C => 1/2
  | Team.B, Team.A => 1/3
  | Team.C, Team.A => 1/3
  | Team.C, Team.B => 1/2
  | _, _ => 0

/-- Team A gets a bye in the first match -/
def first_match_bye : Team := Team.A

/-- The probability that Team B is eliminated after the first three matches -/
def prob_b_eliminated_three_matches : ℚ := 11/36

/-- The probability that Team A wins the championship in only four matches -/
def prob_a_wins_four_matches : ℚ := 8/27

/-- The probability that a fifth match is needed -/
def prob_fifth_match_needed : ℚ := 35/54

theorem basketball_competition_probabilities :
  (prob_b_eliminated_three_matches = 11/36) ∧
  (prob_a_wins_four_matches = 8/27) ∧
  (prob_fifth_match_needed = 35/54) :=
by sorry

end NUMINAMATH_CALUDE_basketball_competition_probabilities_l1447_144736


namespace NUMINAMATH_CALUDE_carnation_percentage_l1447_144792

/-- Represents a floral arrangement with different types of flowers -/
structure FloralArrangement where
  total : ℕ
  pink_roses : ℕ
  red_roses : ℕ
  white_roses : ℕ
  pink_carnations : ℕ
  red_carnations : ℕ
  white_carnations : ℕ

/-- Conditions for the floral arrangement -/
def valid_arrangement (f : FloralArrangement) : Prop :=
  -- Half of the pink flowers are roses
  f.pink_roses = f.pink_carnations
  -- One-third of the red flowers are carnations
  ∧ 3 * f.red_carnations = f.red_roses + f.red_carnations
  -- Three-fifths of the flowers are pink
  ∧ 5 * (f.pink_roses + f.pink_carnations) = 3 * f.total
  -- Total flowers equals sum of all flower types
  ∧ f.total = f.pink_roses + f.red_roses + f.white_roses + 
              f.pink_carnations + f.red_carnations + f.white_carnations

/-- Theorem: The percentage of carnations in a valid floral arrangement is 50% -/
theorem carnation_percentage (f : FloralArrangement) 
  (h : valid_arrangement f) : 
  (f.pink_carnations + f.red_carnations + f.white_carnations) * 2 = f.total := by
  sorry

#check carnation_percentage

end NUMINAMATH_CALUDE_carnation_percentage_l1447_144792


namespace NUMINAMATH_CALUDE_suit_pants_cost_l1447_144726

theorem suit_pants_cost (budget initial_budget remaining : ℕ) 
  (shirt_cost coat_cost socks_cost belt_cost shoes_cost : ℕ) :
  initial_budget = 200 →
  shirt_cost = 30 →
  coat_cost = 38 →
  socks_cost = 11 →
  belt_cost = 18 →
  shoes_cost = 41 →
  remaining = 16 →
  ∃ (pants_cost : ℕ),
    pants_cost = initial_budget - (shirt_cost + coat_cost + socks_cost + belt_cost + shoes_cost + remaining) ∧
    pants_cost = 46 :=
by sorry

end NUMINAMATH_CALUDE_suit_pants_cost_l1447_144726


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1447_144767

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (3^15 + 11^13)) →
  2 ≤ Nat.minFac (3^15 + 11^13) ∧ Nat.minFac (3^15 + 11^13) = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1447_144767


namespace NUMINAMATH_CALUDE_complex_equal_parts_l1447_144772

theorem complex_equal_parts (a : ℝ) : 
  let z : ℂ := (1 - a * Complex.I) / (2 + Complex.I)
  (z.re = z.im) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l1447_144772


namespace NUMINAMATH_CALUDE_first_robber_guarantee_l1447_144791

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : Nat
  maxBags : Nat

/-- Calculates the guaranteed minimum coins for the first robber --/
def guaranteedCoins (game : CoinGame) : Nat :=
  game.totalCoins - (game.maxBags - 1) * (game.totalCoins / (2 * game.maxBags - 1))

/-- Theorem: In a game with 300 coins and 11 max bags, the first robber can guarantee at least 146 coins --/
theorem first_robber_guarantee (game : CoinGame) 
  (h1 : game.totalCoins = 300) 
  (h2 : game.maxBags = 11) : 
  guaranteedCoins game ≥ 146 := by
  sorry

#eval guaranteedCoins { totalCoins := 300, maxBags := 11 }

end NUMINAMATH_CALUDE_first_robber_guarantee_l1447_144791


namespace NUMINAMATH_CALUDE_function_inequality_l1447_144706

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, (x * log x) * deriv f x < f x) : 
  2 * f (sqrt e) > f e := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1447_144706


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_zero_l1447_144769

theorem set_equality_implies_sum_zero (x y : ℝ) : 
  ({x, y, x + y} : Set ℝ) = {0, x^2, x*y} → x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_zero_l1447_144769


namespace NUMINAMATH_CALUDE_soccer_lineup_selections_l1447_144743

/-- The number of players in the soccer team -/
def team_size : ℕ := 16

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to select the starting lineup -/
def lineup_selections : ℕ := 409500

/-- Theorem: The number of ways to select a starting lineup of 5 players from a team of 16,
    where one player (utility) cannot be selected for a specific position (goalkeeper),
    is equal to 409,500. -/
theorem soccer_lineup_selections :
  (team_size - 1) *  -- Goalkeeper selection (excluding utility player)
  (team_size - 1) *  -- Defender selection
  (team_size - 2) *  -- Midfielder selection
  (team_size - 3) *  -- Forward selection
  (team_size - 4)    -- Utility player selection (excluding goalkeeper)
  = lineup_selections := by sorry

end NUMINAMATH_CALUDE_soccer_lineup_selections_l1447_144743


namespace NUMINAMATH_CALUDE_linear_equation_mn_l1447_144711

theorem linear_equation_mn (m n : ℝ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, x^(4-3*|m|) + y^(3*|n|) = a*x + b*y + c) →
  m * n < 0 →
  0 < m + n →
  m + n ≤ 3 →
  m - n = 4/3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_mn_l1447_144711


namespace NUMINAMATH_CALUDE_jerry_current_average_l1447_144707

def jerry_average_score (current_average : ℝ) (raise_by : ℝ) (fourth_test_score : ℝ) : Prop :=
  let total_score_needed := 4 * (current_average + raise_by)
  3 * current_average + fourth_test_score = total_score_needed

theorem jerry_current_average : 
  ∃ (A : ℝ), jerry_average_score A 2 89 ∧ A = 81 :=
sorry

end NUMINAMATH_CALUDE_jerry_current_average_l1447_144707


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1447_144735

theorem sum_of_absolute_values_zero (a b : ℝ) : 
  |a + 3| + |2*b - 4| = 0 → a + b = -1 := by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1447_144735


namespace NUMINAMATH_CALUDE_greenhill_soccer_kicks_l1447_144710

/-- Given a soccer team with total players and goalies, calculate the number of penalty kicks required --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: For a team with 25 players including 4 goalies, 96 penalty kicks are required --/
theorem greenhill_soccer_kicks : penalty_kicks 25 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_greenhill_soccer_kicks_l1447_144710


namespace NUMINAMATH_CALUDE_smallest_prime_with_condition_l1447_144783

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_prime_with_condition : 
  ∃ (p : ℕ), 
    Prime p ∧ 
    is_two_digit p ∧ 
    tens_digit p = 3 ∧ 
    ¬(Prime (reverse_digits p + 5)) ∧
    ∀ (q : ℕ), Prime q ∧ is_two_digit q ∧ tens_digit q = 3 ∧ ¬(Prime (reverse_digits q + 5)) → p ≤ q ∧
    p = 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_condition_l1447_144783


namespace NUMINAMATH_CALUDE_angle_properties_l1447_144713

theorem angle_properties (α : Real) (y : Real) :
  -- Angle α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- Point P on its terminal side has coordinates (-√2, y)
  ∃ P : Real × Real, P = (-Real.sqrt 2, y) →
  -- sin α = (√2/4)y
  Real.sin α = (Real.sqrt 2 / 4) * y →
  -- Prove: tan α = -√3
  Real.tan α = -Real.sqrt 3 ∧
  -- Prove: (3sin α · cos α) / (4sin²α + 2cos²α) = -3√3/14
  (3 * Real.sin α * Real.cos α) / (4 * Real.sin α ^ 2 + 2 * Real.cos α ^ 2) = -3 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l1447_144713


namespace NUMINAMATH_CALUDE_point_position_l1447_144721

theorem point_position (x : ℝ) (h1 : x < -2) (h2 : |x - (-2)| = 5) : x = -7 := by
  sorry

end NUMINAMATH_CALUDE_point_position_l1447_144721


namespace NUMINAMATH_CALUDE_platform_length_platform_length_approx_l1447_144794

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) : ℝ :=
  let train_speed := train_length / pole_time
  let platform_length := train_speed * platform_time - train_length
  platform_length

/-- The platform length is approximately 300 meters -/
theorem platform_length_approx :
  let result := platform_length 300 36 18
  ∃ ε > 0, abs (result - 300) < ε :=
sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_approx_l1447_144794


namespace NUMINAMATH_CALUDE_ships_converge_l1447_144723

/-- Represents a ship with a given round trip duration -/
structure Ship where
  roundTripDays : ℕ

/-- Represents the fleet of ships -/
def Fleet : List Ship := [
  { roundTripDays := 2 },
  { roundTripDays := 3 },
  { roundTripDays := 5 }
]

/-- The number of days after which all ships converge -/
def convergenceDays : ℕ := 30

/-- Theorem stating that the ships converge after the specified number of days -/
theorem ships_converge :
  ∀ (ship : Ship), ship ∈ Fleet → convergenceDays % ship.roundTripDays = 0 := by
  sorry

#check ships_converge

end NUMINAMATH_CALUDE_ships_converge_l1447_144723


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1447_144771

def numbers : List ℤ := [1871, 2011, 2059, 2084, 2113, 2167, 2198, 2210]

theorem mean_of_remaining_numbers :
  ∀ (subset : List ℤ),
    subset ⊆ numbers →
    subset.length = 6 →
    (subset.sum : ℚ) / 6 = 2100 →
    let remaining := numbers.filter (λ x => x ∉ subset)
    (remaining.sum : ℚ) / 2 = 2056.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1447_144771


namespace NUMINAMATH_CALUDE_f_not_monotonic_exists_even_f_exists_three_zeros_l1447_144724

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs (x + a)

-- Theorem 1: f is not monotonic for any a
theorem f_not_monotonic : ∀ a : ℝ, ¬(Monotone (f a)) := by sorry

-- Theorem 2: There exists an 'a' for which f is even
theorem exists_even_f : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by sorry

-- Theorem 3: There exists a negative 'a' for which f has three zeros
theorem exists_three_zeros : ∃ a : ℝ, a < 0 ∧ (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) := by sorry

end NUMINAMATH_CALUDE_f_not_monotonic_exists_even_f_exists_three_zeros_l1447_144724


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1447_144796

theorem quadratic_roots_expression (a b : ℝ) : 
  a^2 + a - 3 = 0 → b^2 + b - 3 = 0 → 4 * b^2 - a^3 = (53 + 8 * Real.sqrt 13) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1447_144796


namespace NUMINAMATH_CALUDE_consecutive_divisible_numbers_l1447_144728

theorem consecutive_divisible_numbers :
  ∃ n : ℕ, 100 ≤ n ∧ n < 200 ∧ 
    3 ∣ n ∧ 5 ∣ (n + 1) ∧ 7 ∣ (n + 2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_divisible_numbers_l1447_144728


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1447_144748

theorem arithmetic_mean_geq_geometric_mean {x y : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1447_144748


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1447_144708

/-- Given a right triangle with sides 3, 4, and 5, x is the side length of a square
    inscribed with one vertex at the right angle, and y is the side length of a square
    inscribed with one side on the hypotenuse. -/
def triangle_with_squares (x y : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a^2 + b^2 = c^2 ∧
    a = 3 ∧ b = 4 ∧ c = 5 ∧
    x / 4 = (3 - x) / 3 ∧
    4/3 * y + y + 3/4 * y = 5

theorem inscribed_squares_ratio :
  ∀ x y : ℝ, triangle_with_squares x y → x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1447_144708


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1447_144712

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n = 4 ∧ 
  (15 ∣ (9679 - n)) ∧ 
  ∀ (m : ℕ), m < n → ¬(15 ∣ (9679 - m)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1447_144712


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1447_144759

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 4 * r^2 + 5 * r - 3) - (r^3 + 5 * r^2 + 9 * r - 6) = r^3 - r^2 - 4 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1447_144759


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_l1447_144782

theorem sin_cos_sum_equals_sqrt_sum : 
  Real.sin (26 * π / 3) + Real.cos (-17 * π / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_l1447_144782


namespace NUMINAMATH_CALUDE_factory_output_l1447_144739

/-- Calculates the number of batteries manufactured by robots in a given time period. -/
def batteries_manufactured (gather_time min_per_battery : ℕ) (create_time min_per_battery : ℕ) 
  (num_robots : ℕ) (total_time hours : ℕ) : ℕ :=
  let total_time_minutes := total_time * 60
  let time_per_battery := gather_time + create_time
  let batteries_per_robot_per_hour := 60 / time_per_battery
  let batteries_per_hour := num_robots * batteries_per_robot_per_hour
  batteries_per_hour * total_time

/-- The number of batteries manufactured by 10 robots in 5 hours is 200. -/
theorem factory_output : batteries_manufactured 6 9 10 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_l1447_144739


namespace NUMINAMATH_CALUDE_mikes_pens_l1447_144718

theorem mikes_pens (initial_pens : ℕ) (final_pens : ℕ) : 
  initial_pens = 5 → final_pens = 40 → ∃ M : ℕ, 
    2 * (initial_pens + M) - 10 = final_pens ∧ M = 20 := by
  sorry

end NUMINAMATH_CALUDE_mikes_pens_l1447_144718


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1447_144755

theorem decimal_to_fraction : (2.36 : ℚ) = 59 / 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1447_144755


namespace NUMINAMATH_CALUDE_cos_585_degrees_l1447_144704

theorem cos_585_degrees : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_585_degrees_l1447_144704


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1447_144789

theorem repeating_decimal_sum (a b : ℕ+) (h1 : (35 : ℚ) / 99 = (a : ℚ) / b) 
  (h2 : Nat.gcd a.val b.val = 1) : a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1447_144789


namespace NUMINAMATH_CALUDE_even_digits_529_base9_l1447_144731

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-9 representation of 529₁₀ is 2 -/
theorem even_digits_529_base9 : 
  countEvenDigits (toBase9 529) = 2 :=
sorry

end NUMINAMATH_CALUDE_even_digits_529_base9_l1447_144731


namespace NUMINAMATH_CALUDE_xiao_ming_book_price_l1447_144730

/-- The price of Xiao Ming's book satisfies 15 < x < 20, given that:
    1. Classmate A guessed the price is at least 20.
    2. Classmate B guessed the price is at most 15.
    3. Xiao Ming said both classmates are wrong. -/
theorem xiao_ming_book_price (x : ℝ) 
  (hA : x < 20)  -- Xiao Ming said A is wrong, so price is less than 20
  (hB : x > 15)  -- Xiao Ming said B is wrong, so price is greater than 15
  : 15 < x ∧ x < 20 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_book_price_l1447_144730


namespace NUMINAMATH_CALUDE_cargo_weight_calculation_l1447_144725

/-- Calculates the total cargo weight after loading and unloading activities -/
def total_cargo_weight (initial_cargo : Real) (additional_cargo : Real) (unloaded_cargo : Real) 
  (short_ton_to_kg : Real) (pound_to_kg : Real) : Real :=
  (initial_cargo * short_ton_to_kg) + (additional_cargo * short_ton_to_kg) - (unloaded_cargo * pound_to_kg)

/-- Theorem stating the total cargo weight after loading and unloading activities -/
theorem cargo_weight_calculation :
  let initial_cargo : Real := 5973.42
  let additional_cargo : Real := 8723.18
  let unloaded_cargo : Real := 2256719.55
  let short_ton_to_kg : Real := 907.18474
  let pound_to_kg : Real := 0.45359237
  total_cargo_weight initial_cargo additional_cargo unloaded_cargo short_ton_to_kg pound_to_kg = 12302024.7688159 := by
  sorry


end NUMINAMATH_CALUDE_cargo_weight_calculation_l1447_144725


namespace NUMINAMATH_CALUDE_conic_family_inscribed_in_square_l1447_144753

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | (p.1 = 3 ∨ p.1 = -3) ∧ p.2 ∈ [-3, 3] ∨
       (p.2 = 3 ∨ p.2 = -3) ∧ p.1 ∈ [-3, 3]}

-- Define the differential equation
def diff_eq (x y : ℝ) (dy_dx : ℝ) : Prop :=
  (9 - x^2) * dy_dx^2 = (9 - y^2)

-- Define a family of conics
def conic_family (C : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * (Real.cos C) * x * y = 9 * (Real.sin C)^2

-- State the theorem
theorem conic_family_inscribed_in_square :
  ∀ C : ℝ, ∃ p : ℝ × ℝ,
    p ∈ square ∧
    (∃ x y dy_dx : ℝ, diff_eq x y dy_dx ∧
      conic_family C x y ∧
      (x = p.1 ∧ y = p.2)) :=
sorry

end NUMINAMATH_CALUDE_conic_family_inscribed_in_square_l1447_144753


namespace NUMINAMATH_CALUDE_basketball_scores_l1447_144790

/-- Represents the scores of a team over four quarters -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℚ)

/-- Checks if a sequence of four numbers is geometric -/
def isGeometric (s : TeamScores) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a sequence of four numbers is arithmetic -/
def isArithmetic (s : TeamScores) : Prop :=
  ∃ d : ℚ, s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℚ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def firstHalfScore (s : TeamScores) : ℚ :=
  s.q1 + s.q2

theorem basketball_scores (teamA teamB : TeamScores) :
  teamA.q1 = teamB.q1 →  -- Tied at the end of first quarter
  isGeometric teamA →    -- Team A's scores form a geometric sequence
  isArithmetic teamB →   -- Team B's scores form an arithmetic sequence
  totalScore teamA = totalScore teamB + 2 →  -- Team A won by two points
  totalScore teamA ≤ 80 →  -- Team A's total score is not more than 80
  totalScore teamB ≤ 80 →  -- Team B's total score is not more than 80
  firstHalfScore teamA + firstHalfScore teamB = 41 :=
by sorry

end NUMINAMATH_CALUDE_basketball_scores_l1447_144790


namespace NUMINAMATH_CALUDE_planning_committee_selection_l1447_144788

theorem planning_committee_selection (n : ℕ) : 
  (n.choose 2 = 21) → (n.choose 4 = 35) := by
  sorry

end NUMINAMATH_CALUDE_planning_committee_selection_l1447_144788


namespace NUMINAMATH_CALUDE_train_to_subway_ratio_l1447_144784

/-- Represents the travel times for Andrew's journey from Manhattan to the Bronx -/
structure TravelTimes where
  total : ℝ
  subway : ℝ
  biking : ℝ
  train : ℝ

/-- Theorem stating the ratio of train time to subway time -/
theorem train_to_subway_ratio (t : TravelTimes) 
  (h1 : t.total = 38)
  (h2 : t.subway = 10)
  (h3 : t.biking = 8)
  (h4 : t.train = t.total - t.subway - t.biking) :
  t.train / t.subway = 2 := by
  sorry

#check train_to_subway_ratio

end NUMINAMATH_CALUDE_train_to_subway_ratio_l1447_144784


namespace NUMINAMATH_CALUDE_min_socks_in_box_min_socks_even_black_l1447_144709

/-- Represents a box of socks -/
structure SockBox where
  red : ℕ
  black : ℕ

/-- The probability of drawing two red socks from the box -/
def prob_two_red (box : SockBox) : ℚ :=
  (box.red / (box.red + box.black)) * ((box.red - 1) / (box.red + box.black - 1))

/-- The total number of socks in the box -/
def total_socks (box : SockBox) : ℕ := box.red + box.black

theorem min_socks_in_box :
  ∃ (box : SockBox), prob_two_red box = 1/2 ∧
    ∀ (other : SockBox), prob_two_red other = 1/2 → total_socks box ≤ total_socks other :=
sorry

theorem min_socks_even_black :
  ∃ (box : SockBox), prob_two_red box = 1/2 ∧ box.black % 2 = 0 ∧
    ∀ (other : SockBox), prob_two_red other = 1/2 ∧ other.black % 2 = 0 →
      total_socks box ≤ total_socks other :=
sorry

end NUMINAMATH_CALUDE_min_socks_in_box_min_socks_even_black_l1447_144709


namespace NUMINAMATH_CALUDE_curve_length_of_right_square_prism_l1447_144749

/-- Represents a right square prism -/
structure RightSquarePrism where
  sideEdge : ℝ
  baseEdge : ℝ

/-- Calculates the total length of curves on the surface of a right square prism
    formed by points at a given distance from a vertex -/
def totalCurveLength (prism : RightSquarePrism) (distance : ℝ) : ℝ :=
  sorry

/-- The theorem statement -/
theorem curve_length_of_right_square_prism :
  let prism : RightSquarePrism := ⟨4, 4⟩
  totalCurveLength prism 3 = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_curve_length_of_right_square_prism_l1447_144749


namespace NUMINAMATH_CALUDE_right_triangle_construction_condition_l1447_144761

/-- Given a right triangle ABC with leg AC = b and perimeter 2s, 
    prove that the construction is possible if and only if b < s -/
theorem right_triangle_construction_condition 
  (b s : ℝ) 
  (h_positive_b : 0 < b) 
  (h_positive_s : 0 < s) 
  (h_perimeter : ∃ (c : ℝ), b + c + (b^2 + c^2).sqrt = 2*s) :
  (∃ (c : ℝ), c > 0 ∧ b^2 + c^2 = ((2*s - b - c)^2)) ↔ b < s :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_construction_condition_l1447_144761


namespace NUMINAMATH_CALUDE_angle_sum_bounds_l1447_144766

theorem angle_sum_bounds (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_sum_sin_sq : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ < 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_bounds_l1447_144766


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1447_144795

theorem largest_solution_of_equation : 
  ∃ (y : ℝ), y = 5 ∧ 
  3 * y^2 + 30 * y - 90 = y * (y + 18) ∧
  ∀ (z : ℝ), 3 * z^2 + 30 * z - 90 = z * (z + 18) → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1447_144795


namespace NUMINAMATH_CALUDE_pencil_cost_difference_l1447_144756

def joy_pencils : ℕ := 30
def colleen_pencils : ℕ := 50
def pencil_cost : ℕ := 4

theorem pencil_cost_difference : 
  colleen_pencils * pencil_cost - joy_pencils * pencil_cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_difference_l1447_144756


namespace NUMINAMATH_CALUDE_range_of_m_l1447_144774

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def B : Set ℝ := {x | -1 < x ∧ x < 5}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem range_of_m (m : ℝ) :
  (Set.compl (A m) ∩ B).Nonempty → m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1447_144774


namespace NUMINAMATH_CALUDE_arthur_sword_problem_l1447_144758

theorem arthur_sword_problem (A B : ℕ) : 
  5 * A + 7 * B = 49 → A - B = 5 := by
sorry

end NUMINAMATH_CALUDE_arthur_sword_problem_l1447_144758


namespace NUMINAMATH_CALUDE_square_root_equality_l1447_144799

theorem square_root_equality (x a : ℝ) (hx : x > 0) :
  (Real.sqrt x = 2 * a - 1 ∧ Real.sqrt x = -a + 2) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l1447_144799


namespace NUMINAMATH_CALUDE_smallest_p_in_prime_sum_l1447_144777

theorem smallest_p_in_prime_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p + q = r →
  1 < p →
  p < q →
  r > 10 →
  ∃ (p' : ℕ), p' = 2 ∧ (∀ (p'' : ℕ), 
    Nat.Prime p'' → 
    p'' + q = r → 
    1 < p'' → 
    p'' < q → 
    r > 10 → 
    p' ≤ p'') :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_in_prime_sum_l1447_144777


namespace NUMINAMATH_CALUDE_scaled_building_height_l1447_144745

/-- Calculates the height of a scaled model building given the original building's height and the volumes of water held in the top portions of both the original and the model. -/
theorem scaled_building_height
  (original_height : ℝ)
  (original_volume : ℝ)
  (model_volume : ℝ)
  (h_original_height : original_height = 120)
  (h_original_volume : original_volume = 30000)
  (h_model_volume : model_volume = 0.03)
  : ∃ (model_height : ℝ), model_height = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_scaled_building_height_l1447_144745


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1447_144727

def repeating_decimal_1 : ℚ := 1 / 9
def repeating_decimal_2 : ℚ := 2 / 99
def repeating_decimal_3 : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 + repeating_decimal_3 = 134 / 999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1447_144727


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l1447_144786

/-- The surface area of a cuboid with given dimensions -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * height + length * breadth + breadth * height)

/-- Theorem: The surface area of a cuboid with length 10 cm, breadth 8 cm, and height 6 cm is 376 cm² -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 10 8 6 = 376 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l1447_144786


namespace NUMINAMATH_CALUDE_skating_time_for_average_l1447_144776

def minutes_per_day_1 : ℕ := 80
def days_1 : ℕ := 4
def minutes_per_day_2 : ℕ := 105
def days_2 : ℕ := 3
def total_days : ℕ := 8
def target_average : ℕ := 95

theorem skating_time_for_average :
  (minutes_per_day_1 * days_1 + minutes_per_day_2 * days_2 + 125) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_skating_time_for_average_l1447_144776


namespace NUMINAMATH_CALUDE_fraction_to_repeating_decimal_value_of_expression_l1447_144744

def repeating_decimal (n d : ℕ) (a b c d : ℕ) : Prop :=
  (n : ℚ) / d = (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999

theorem fraction_to_repeating_decimal :
  repeating_decimal 7 26 2 6 9 2 :=
sorry

theorem value_of_expression (a b c d : ℕ) :
  repeating_decimal 7 26 a b c d → 3 * a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_to_repeating_decimal_value_of_expression_l1447_144744


namespace NUMINAMATH_CALUDE_labor_costs_theorem_l1447_144781

/-- Calculates the overall labor costs for one day given the number of workers and their wages. -/
def overall_labor_costs (
  num_construction_workers : ℕ)
  (num_electricians : ℕ)
  (num_plumbers : ℕ)
  (construction_worker_wage : ℚ)
  (electrician_wage_multiplier : ℚ)
  (plumber_wage_multiplier : ℚ) : ℚ :=
  (num_construction_workers * construction_worker_wage) +
  (num_electricians * (electrician_wage_multiplier * construction_worker_wage)) +
  (num_plumbers * (plumber_wage_multiplier * construction_worker_wage))

/-- Proves that the overall labor costs for one day is $650 given the specified conditions. -/
theorem labor_costs_theorem :
  overall_labor_costs 2 1 1 100 2 (5/2) = 650 := by
  sorry

#eval overall_labor_costs 2 1 1 100 2 (5/2)

end NUMINAMATH_CALUDE_labor_costs_theorem_l1447_144781


namespace NUMINAMATH_CALUDE_smallest_first_term_divisible_by_11_l1447_144700

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

-- Define the sum of seven consecutive terms starting from k
def sumSevenTerms (k : ℕ) : ℤ := 
  (arithmeticSequence k) + 
  (arithmeticSequence (k + 1)) + 
  (arithmeticSequence (k + 2)) + 
  (arithmeticSequence (k + 3)) + 
  (arithmeticSequence (k + 4)) + 
  (arithmeticSequence (k + 5)) + 
  (arithmeticSequence (k + 6))

-- The theorem to prove
theorem smallest_first_term_divisible_by_11 :
  ∃ k : ℕ, (sumSevenTerms k) % 11 = 0 ∧ 
  ∀ m : ℕ, m < k → (sumSevenTerms m) % 11 ≠ 0 ∧
  arithmeticSequence k = 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_first_term_divisible_by_11_l1447_144700


namespace NUMINAMATH_CALUDE_ratio_of_11th_terms_l1447_144770

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := n * (a₁ + (n - 1) / 2 * d)

theorem ratio_of_11th_terms
  (a₁ d₁ a₂ d₂ : ℚ)
  (h : ∀ n : ℕ, sum_arithmetic_sequence a₁ d₁ n / sum_arithmetic_sequence a₂ d₂ n = (7 * n + 1) / (4 * n + 27)) :
  (arithmetic_sequence a₁ d₁ 11) / (arithmetic_sequence a₂ d₂ 11) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_11th_terms_l1447_144770


namespace NUMINAMATH_CALUDE_equation_solutions_may_days_l1447_144793

-- Define the interval [0°, 360°]
def angle_interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 360}

-- Define the equation cos³α - cosα = 0
def equation (α : ℝ) : Prop := Real.cos α ^ 3 - Real.cos α = 0

-- Define the day of the week as an enumeration
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to get the day of the week for a given day in May
def day_in_may (day : ℕ) : DayOfWeek := sorry

-- Theorem 1: There are exactly 5 values of α in [0°, 360°] that satisfy cos³α - cosα = 0
theorem equation_solutions :
  ∃ (S : Finset ℝ), S.card = 5 ∧ (∀ α ∈ S, α ∈ angle_interval ∧ equation α) ∧
    (∀ α, α ∈ angle_interval → equation α → α ∈ S) :=
  sorry

-- Theorem 2: If the 5th day of May is Thursday, then the 16th day of May is Monday
theorem may_days :
  day_in_may 5 = DayOfWeek.Thursday → day_in_may 16 = DayOfWeek.Monday :=
  sorry

end NUMINAMATH_CALUDE_equation_solutions_may_days_l1447_144793


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l1447_144780

theorem sum_of_squares_zero (a b c : ℝ) : 
  (a - 2)^2 + (b + 3)^2 + (c - 7)^2 = 0 → a + b + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l1447_144780


namespace NUMINAMATH_CALUDE_f_three_quadrants_iff_a_range_l1447_144737

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x - 1
  else x^3 - a * x + |x - 2|

/-- The graph of f passes through exactly three quadrants -/
def passes_through_three_quadrants (a : ℝ) : Prop :=
  ∃ x y z : ℝ,
    (x < 0 ∧ f a x < 0) ∧
    (y > 0 ∧ f a y > 0) ∧
    ((z < 0 ∧ f a z > 0) ∨ (z > 0 ∧ f a z < 0)) ∧
    ∀ w : ℝ, (w < 0 ∧ f a w > 0) → (z < 0 ∧ f a z > 0) ∧
             (w > 0 ∧ f a w < 0) → (z > 0 ∧ f a z < 0)

/-- Main theorem: f passes through exactly three quadrants iff a < 0 or a > 2 -/
theorem f_three_quadrants_iff_a_range (a : ℝ) :
  passes_through_three_quadrants a ↔ a < 0 ∨ a > 2 := by
  sorry


end NUMINAMATH_CALUDE_f_three_quadrants_iff_a_range_l1447_144737


namespace NUMINAMATH_CALUDE_recommended_intake_proof_l1447_144779

/-- Recommended intake of added sugar for men per day (in calories) -/
def recommended_intake : ℝ := 150

/-- Calories in the soft drink -/
def soft_drink_calories : ℝ := 2500

/-- Percentage of calories from added sugar in the soft drink -/
def soft_drink_sugar_percentage : ℝ := 0.05

/-- Calories of added sugar in each candy bar -/
def candy_bar_sugar_calories : ℝ := 25

/-- Number of candy bars consumed -/
def candy_bars_consumed : ℕ := 7

/-- Percentage by which Mark exceeded the recommended intake -/
def excess_percentage : ℝ := 1

theorem recommended_intake_proof :
  let soft_drink_sugar := soft_drink_calories * soft_drink_sugar_percentage
  let candy_sugar := candy_bar_sugar_calories * candy_bars_consumed
  let total_sugar := soft_drink_sugar + candy_sugar
  total_sugar = recommended_intake * (1 + excess_percentage) :=
by sorry

end NUMINAMATH_CALUDE_recommended_intake_proof_l1447_144779


namespace NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l1447_144798

/-- The Tuning Day Method function -/
def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

/-- Check if a fraction is simpler than another -/
def isSimpler (a b c d : ℕ) : Bool :=
  a + b < c + d ∨ (a + b = c + d ∧ a < c)

theorem tuning_day_method_pi_approximation :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let step1 : ℚ := tuningDayMethod 10 31 15 49
  let step2 : ℚ := tuningDayMethod 10 31 5 16
  let step3 : ℚ := tuningDayMethod 15 47 5 16
  let step4 : ℚ := tuningDayMethod 15 47 20 63
  initial_lower < Real.pi ∧ Real.pi < initial_upper ∧
  step1 = 16 / 5 ∧
  step2 = 47 / 15 ∧
  step3 = 63 / 20 ∧
  step4 = 22 / 7 ∧
  isSimpler 22 7 63 20 ∧
  isSimpler 22 7 47 15 ∧
  isSimpler 22 7 16 5 ∧
  47 / 15 < Real.pi ∧ Real.pi < 22 / 7 :=
by sorry

end NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l1447_144798


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1447_144746

theorem complex_number_quadrant : ∃ (z : ℂ), z = (Complex.I : ℂ) / (1 + Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1447_144746


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_achieved_l1447_144785

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (-x)^2) ≥ 2 * Real.sqrt 2 :=
sorry

theorem min_value_sqrt_sum_achieved : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (-x)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_achieved_l1447_144785


namespace NUMINAMATH_CALUDE_eunji_shopping_l1447_144751

theorem eunji_shopping (initial_money : ℝ) : 
  initial_money * (1 - 1/4) * (1 - 1/3) = 1600 → initial_money = 3200 := by
  sorry

end NUMINAMATH_CALUDE_eunji_shopping_l1447_144751


namespace NUMINAMATH_CALUDE_a_less_than_abs_a_implies_negative_l1447_144760

theorem a_less_than_abs_a_implies_negative (a : ℝ) : a < |a| → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_abs_a_implies_negative_l1447_144760


namespace NUMINAMATH_CALUDE_sqrt_2_3_5_not_arithmetic_progression_l1447_144775

theorem sqrt_2_3_5_not_arithmetic_progression : ¬ ∃ (d : ℝ), Real.sqrt 3 = Real.sqrt 2 + d ∧ Real.sqrt 5 = Real.sqrt 2 + 2 * d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_3_5_not_arithmetic_progression_l1447_144775


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1447_144752

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1447_144752


namespace NUMINAMATH_CALUDE_father_son_age_sum_l1447_144716

theorem father_son_age_sum :
  ∀ (F S : ℕ),
  F > 0 ∧ S > 0 →
  F / S = 7 / 4 →
  (F + 10) / (S + 10) = 5 / 3 →
  F + S = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_l1447_144716


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l1447_144705

theorem power_of_three_mod_eight : 3^2007 % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l1447_144705


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l1447_144750

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - (2 * Q.quadrilateral_faces + 5 * Q.pentagonal_faces)

/-- Theorem: A specific convex polyhedron Q has 323 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 10,
    pentagonal_faces := 4
  }
  space_diagonals Q = 323 := by
  sorry


end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l1447_144750


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1447_144738

theorem complex_fraction_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  (1 / (a + 1) - 1 / (a^2 - 1)) / (a / (a - 1) - a) = -1 / (a^2 + a) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1447_144738


namespace NUMINAMATH_CALUDE_six_pieces_per_small_load_l1447_144763

/-- Given a total number of clothing pieces, the number of pieces in the first load,
    and the number of small loads, calculate the number of pieces in each small load. -/
def clothingPerSmallLoad (total : ℕ) (firstLoad : ℕ) (smallLoads : ℕ) : ℕ :=
  (total - firstLoad) / smallLoads

/-- Theorem stating that with 47 total pieces, 17 in the first load, and 5 small loads,
    each small load contains 6 pieces of clothing. -/
theorem six_pieces_per_small_load :
  clothingPerSmallLoad 47 17 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_pieces_per_small_load_l1447_144763


namespace NUMINAMATH_CALUDE_max_area_CDFE_l1447_144778

/-- A square with side length 1 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 1 ∧ B.2 = 0 ∧ C.1 = 1 ∧ C.2 = 1 ∧ D.1 = 0 ∧ D.2 = 1)

/-- Points E and F on sides AB and AD respectively -/
def PointsEF (s : Square) (x : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((x, 0), (0, x))

/-- Area of quadrilateral CDFE -/
def AreaCDFE (s : Square) (x : ℝ) : ℝ :=
  x * (1 - x)

/-- The maximum area of quadrilateral CDFE is 1/4 -/
theorem max_area_CDFE (s : Square) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → AreaCDFE s y ≤ AreaCDFE s x ∧
  AreaCDFE s x = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_area_CDFE_l1447_144778


namespace NUMINAMATH_CALUDE_berry_theorem_l1447_144719

def berry_problem (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries)

theorem berry_theorem : berry_problem 26 10 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_berry_theorem_l1447_144719


namespace NUMINAMATH_CALUDE_network_connections_l1447_144765

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  (n * k) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l1447_144765


namespace NUMINAMATH_CALUDE_total_triangles_is_53_l1447_144722

/-- Represents a rectangular figure with internal divisions -/
structure RectangularFigure where
  /-- The number of smallest right triangles -/
  small_right_triangles : ℕ
  /-- The number of isosceles triangles with base equal to the width -/
  width_isosceles_triangles : ℕ
  /-- The number of isosceles triangles with base equal to half the length -/
  half_length_isosceles_triangles : ℕ
  /-- The number of large right triangles -/
  large_right_triangles : ℕ
  /-- The number of large isosceles triangles with base equal to the full width -/
  large_isosceles_triangles : ℕ

/-- Calculates the total number of triangles in the figure -/
def total_triangles (figure : RectangularFigure) : ℕ :=
  figure.small_right_triangles +
  figure.width_isosceles_triangles +
  figure.half_length_isosceles_triangles +
  figure.large_right_triangles +
  figure.large_isosceles_triangles

/-- The specific rectangular figure described in the problem -/
def problem_figure : RectangularFigure :=
  { small_right_triangles := 24
  , width_isosceles_triangles := 6
  , half_length_isosceles_triangles := 8
  , large_right_triangles := 12
  , large_isosceles_triangles := 3
  }

/-- Theorem stating that the total number of triangles in the problem figure is 53 -/
theorem total_triangles_is_53 : total_triangles problem_figure = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_53_l1447_144722


namespace NUMINAMATH_CALUDE_simplify_expression_l1447_144733

theorem simplify_expression (x : ℝ) : (3*x)^4 + (4*x)*(x^5) = 81*x^4 + 4*x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1447_144733


namespace NUMINAMATH_CALUDE_average_difference_l1447_144773

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 110) 
  (hbc : (b + c) / 2 = 150) : 
  a - c = -80 := by sorry

end NUMINAMATH_CALUDE_average_difference_l1447_144773


namespace NUMINAMATH_CALUDE_infinite_n_squared_plus_one_divides_and_not_divides_factorial_l1447_144757

theorem infinite_n_squared_plus_one_divides_and_not_divides_factorial :
  (∃ S : Set ℤ, Set.Infinite S ∧ ∀ n ∈ S, (n^2 + 1) ∣ n!) ∧
  (∃ T : Set ℤ, Set.Infinite T ∧ ∀ n ∈ T, ¬((n^2 + 1) ∣ n!)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_n_squared_plus_one_divides_and_not_divides_factorial_l1447_144757


namespace NUMINAMATH_CALUDE_quotient_approx_l1447_144715

theorem quotient_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ |0.284973 / 29 - 0.009827| < ε :=
sorry

end NUMINAMATH_CALUDE_quotient_approx_l1447_144715


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l1447_144701

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles 
  (c1 : ℝ × ℝ → ℝ) 
  (c2 : ℝ × ℝ → ℝ) 
  (h1 : ∀ x y, c1 (x, y) = x^2 - 6*x + y^2 - 8*y + 4)
  (h2 : ∀ x y, c2 (x, y) = x^2 + 8*x + y^2 + 12*y + 36) :
  let d := Real.sqrt 149 - Real.sqrt 21 - 4
  ∃ p1 p2, c1 p1 = 0 ∧ c2 p2 = 0 ∧ 
    d = Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) - 
        (Real.sqrt 21 + 4) ∧
    ∀ q1 q2, c1 q1 = 0 → c2 q2 = 0 → 
      d ≤ Real.sqrt ((q1.1 - q2.1)^2 + (q1.2 - q2.2)^2) - 
          (Real.sqrt 21 + 4) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l1447_144701


namespace NUMINAMATH_CALUDE_josephine_milk_sales_l1447_144702

/-- The total amount of milk sold by Josephine on Sunday morning -/
def total_milk_sold (container_2L : ℕ) (container_075L : ℕ) (container_05L : ℕ) : ℝ :=
  (container_2L * 2) + (container_075L * 0.75) + (container_05L * 0.5)

/-- Theorem stating that Josephine sold 10 liters of milk given the specified containers -/
theorem josephine_milk_sales : total_milk_sold 3 2 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_josephine_milk_sales_l1447_144702


namespace NUMINAMATH_CALUDE_months_C_is_three_l1447_144747

/-- Represents the number of months C put his oxen for grazing -/
def months_C : ℕ := sorry

/-- Total rent of the pasture in rupees -/
def total_rent : ℕ := 280

/-- Number of oxen A put for grazing -/
def oxen_A : ℕ := 10

/-- Number of months A put his oxen for grazing -/
def months_A : ℕ := 7

/-- Number of oxen B put for grazing -/
def oxen_B : ℕ := 12

/-- Number of months B put his oxen for grazing -/
def months_B : ℕ := 5

/-- Number of oxen C put for grazing -/
def oxen_C : ℕ := 15

/-- C's share of rent in rupees -/
def rent_C : ℕ := 72

/-- Theorem stating that C put his oxen for grazing for 3 months -/
theorem months_C_is_three : months_C = 3 := by sorry

end NUMINAMATH_CALUDE_months_C_is_three_l1447_144747


namespace NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l1447_144720

/-- The asymptotic lines of the hyperbola 3x^2 - y^2 = 3 are y = ± √3 x -/
theorem hyperbola_asymptotic_lines :
  let hyperbola := {(x, y) : ℝ × ℝ | 3 * x^2 - y^2 = 3}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x}
  (Set.range fun t : ℝ => (t, Real.sqrt 3 * t)) ∪ (Set.range fun t : ℝ => (t, -Real.sqrt 3 * t)) =
    {p | p ∈ asymptotic_lines ∧ p ∉ hyperbola ∧ ∀ ε > 0, ∃ q ∈ hyperbola, dist p q < ε} := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l1447_144720


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1447_144734

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1447_144734


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1447_144762

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|x - 3| < 1 → x^2 + x - 6 > 0)) ∧
  (∃ x : ℝ, x^2 + x - 6 > 0 ∧ ¬(|x - 3| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1447_144762


namespace NUMINAMATH_CALUDE_geometry_propositions_l1447_144714

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (intersect : Line → Line → Prop)

-- Theorem statement
theorem geometry_propositions 
  (a b c : Line) (α β : Plane) :
  (∀ (α β : Plane) (c : Line), 
    parallel_plane α β → perpendicular_plane c α → perpendicular_plane c β) ∧
  (∀ (a b c : Line),
    perpendicular a c → perpendicular b c → 
    (parallel a b ∨ skew a b ∨ intersect a b)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1447_144714


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1447_144754

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1447_144754


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1447_144797

/-- The area of an isosceles trapezoid with given dimensions -/
theorem isosceles_trapezoid_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let leg := a
  let base1 := b
  let base2 := c
  let height := Real.sqrt (a^2 - ((c - b)/2)^2)
  (base1 + base2) * height / 2 = 36 ↔ a = 5 ∧ b = 6 ∧ c = 12 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1447_144797


namespace NUMINAMATH_CALUDE_parallelogram_roots_l1447_144768

def polynomial (a : ℝ) (z : ℂ) : ℂ :=
  z^4 - 6*z^3 + 11*a*z^2 - 3*(2*a^2 + 3*a - 3)*z + 1

def forms_parallelogram (roots : List ℂ) : Prop :=
  ∃ (w₁ w₂ : ℂ), roots = [w₁, -w₁, w₂, -w₂]

theorem parallelogram_roots (a : ℝ) :
  (∃ (roots : List ℂ), (∀ z ∈ roots, polynomial a z = 0) ∧
                       roots.length = 4 ∧
                       forms_parallelogram roots) ↔ a = 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l1447_144768


namespace NUMINAMATH_CALUDE_set_operation_equality_l1447_144741

def U : Finset Nat := {1,2,3,4,5,6,7}
def A : Finset Nat := {2,4,5,7}
def B : Finset Nat := {3,4,5}

theorem set_operation_equality : (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_equality_l1447_144741


namespace NUMINAMATH_CALUDE_sum_between_equals_1999_l1447_144703

theorem sum_between_equals_1999 :
  ∀ x y : ℕ, x < y →
  (((x + 1 + (y - 1)) / 2) * (y - x - 1) = 1999) →
  ((x = 1998 ∧ y = 2000) ∨ (x = 998 ∧ y = 1001)) :=
by sorry

end NUMINAMATH_CALUDE_sum_between_equals_1999_l1447_144703


namespace NUMINAMATH_CALUDE_firecracker_sales_properties_l1447_144787

/-- Electronic firecracker sales model -/
structure FirecrackerSales where
  cost : ℝ
  demand : ℝ → ℝ
  price_range : Set ℝ

/-- Daily profit function -/
def daily_profit (model : FirecrackerSales) (x : ℝ) : ℝ :=
  (x - model.cost) * model.demand x

theorem firecracker_sales_properties (model : FirecrackerSales) 
  (h_cost : model.cost = 80)
  (h_demand : ∀ x, model.demand x = -2 * x + 320)
  (h_range : model.price_range = {x | 80 ≤ x ∧ x ≤ 160}) :
  (∀ x ∈ model.price_range, daily_profit model x = -2 * x^2 + 480 * x - 25600) ∧
  (∃ max_price ∈ model.price_range, 
    (∀ x ∈ model.price_range, daily_profit model x ≤ daily_profit model max_price) ∧
    daily_profit model max_price = 3200 ∧
    max_price = 120) ∧
  (∃ price ∈ model.price_range, daily_profit model price = 2400 ∧ price = 100) := by
  sorry

end NUMINAMATH_CALUDE_firecracker_sales_properties_l1447_144787


namespace NUMINAMATH_CALUDE_max_planes_from_three_parallel_lines_l1447_144764

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- Determines if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Determines if a line lies on a plane -/
def lineOnPlane (l : Line3D) (p : Plane) : Prop :=
  sorry

/-- Determines if a plane is defined by two lines -/
def planeDefinedByLines (p : Plane) (l1 l2 : Line3D) : Prop :=
  sorry

/-- The main theorem: maximum number of planes defined by three parallel lines -/
theorem max_planes_from_three_parallel_lines (l1 l2 l3 : Line3D) 
  (h_parallel_12 : parallel l1 l2) 
  (h_parallel_23 : parallel l2 l3) 
  (h_parallel_13 : parallel l1 l3) :
  ∃ (p1 p2 p3 : Plane), 
    (∀ (p : Plane), (planeDefinedByLines p l1 l2 ∨ planeDefinedByLines p l2 l3 ∨ planeDefinedByLines p l1 l3) → 
      (p = p1 ∨ p = p2 ∨ p = p3)) ∧
    (∃ (p : Plane), planeDefinedByLines p l1 l2 ∧ planeDefinedByLines p l2 l3 ∧ planeDefinedByLines p l1 l3) →
      (p1 = p2 ∧ p2 = p3) :=
by
  sorry

end NUMINAMATH_CALUDE_max_planes_from_three_parallel_lines_l1447_144764
