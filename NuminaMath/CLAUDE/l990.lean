import Mathlib

namespace NUMINAMATH_CALUDE_correlation_coefficient_and_fit_l990_99042

/-- Represents the correlation coefficient in regression analysis -/
def correlation_coefficient : ℝ := sorry

/-- Represents the goodness of fit of a regression model -/
def goodness_of_fit : ℝ := sorry

/-- States that as the absolute value of the correlation coefficient 
    approaches 1, the goodness of fit improves -/
theorem correlation_coefficient_and_fit :
  ∀ ε > 0, ∃ δ > 0, ∀ R : ℝ,
    |R| > 1 - δ → goodness_of_fit > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_and_fit_l990_99042


namespace NUMINAMATH_CALUDE_sum_max_min_f_l990_99067

def f (x : ℝ) := -x^2 + 2*x + 3

theorem sum_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_max_min_f_l990_99067


namespace NUMINAMATH_CALUDE_marble_bag_total_l990_99066

/-- Given a bag of marbles with red:blue:green ratio of 2:4:5 and 40 blue marbles,
    the total number of marbles is 110. -/
theorem marble_bag_total (red blue green total : ℕ) : 
  red + blue + green = total →
  red = 2 * n ∧ blue = 4 * n ∧ green = 5 * n →
  blue = 40 →
  total = 110 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_total_l990_99066


namespace NUMINAMATH_CALUDE_boats_left_l990_99047

def total_boats : ℕ := 30
def fish_eaten_percentage : ℚ := 1/5
def boats_shot : ℕ := 2

theorem boats_left : 
  total_boats - (total_boats * fish_eaten_percentage).floor - boats_shot = 22 := by
sorry

end NUMINAMATH_CALUDE_boats_left_l990_99047


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l990_99048

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c 0 = 1 →
  (∃ m : ℝ, m = -1 ∧
    (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b c x ≥ 2 * x + m) ∧
    (∀ m' : ℝ, m' > m →
      ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f a b c x < 2 * x + m')) →
  (∀ x, f a b c x = x^2 - x + 1) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l990_99048


namespace NUMINAMATH_CALUDE_complex_power_195_deg_36_l990_99040

theorem complex_power_195_deg_36 :
  (Complex.exp (195 * π / 180 * Complex.I)) ^ 36 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_195_deg_36_l990_99040


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l990_99012

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l990_99012


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l990_99018

/-- The degree of (5x^3 + 7)^10 is 30 -/
theorem degree_of_polynomial (x : ℝ) : 
  Polynomial.degree ((5 * X + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l990_99018


namespace NUMINAMATH_CALUDE_integer_1025_column_l990_99037

def column_sequence := ["B", "C", "D", "E", "A"]

theorem integer_1025_column :
  let n := 1025 - 1
  let column_index := n % (List.length column_sequence)
  List.get! column_sequence column_index = "E" := by
  sorry

end NUMINAMATH_CALUDE_integer_1025_column_l990_99037


namespace NUMINAMATH_CALUDE_shift_down_quadratic_l990_99000

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the transformation (shift down by 2 units)
def shift_down (y : ℝ) : ℝ := y - 2

-- Define the resulting function after transformation
def resulting_function (x : ℝ) : ℝ := x^2 - 2

-- Theorem stating that shifting the original function down by 2 units
-- results in the resulting function
theorem shift_down_quadratic :
  ∀ x : ℝ, shift_down (original_function x) = resulting_function x :=
by
  sorry


end NUMINAMATH_CALUDE_shift_down_quadratic_l990_99000


namespace NUMINAMATH_CALUDE_bake_sale_profit_split_l990_99084

/-- The number of dozens of cookies John makes -/
def dozens : ℕ := 6

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The selling price of each cookie in dollars -/
def selling_price : ℚ := 3/2

/-- The cost to make each cookie in dollars -/
def cost_per_cookie : ℚ := 1/4

/-- The amount each charity receives in dollars -/
def charity_amount : ℚ := 45

/-- The number of charities John splits the profit between -/
def num_charities : ℕ := 2

theorem bake_sale_profit_split :
  (dozens * cookies_per_dozen * selling_price - dozens * cookies_per_dozen * cost_per_cookie) / charity_amount = num_charities := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_profit_split_l990_99084


namespace NUMINAMATH_CALUDE_range_of_m_l990_99091

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l990_99091


namespace NUMINAMATH_CALUDE_crayon_ratio_l990_99098

theorem crayon_ratio (total : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 15 → blue = 3 → red = total - blue → (red : ℚ) / blue = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_crayon_ratio_l990_99098


namespace NUMINAMATH_CALUDE_ab_value_l990_99062

-- Define the sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

-- State the theorem
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l990_99062


namespace NUMINAMATH_CALUDE_sequence_sum_l990_99003

theorem sequence_sum (n : ℕ) (y : ℕ → ℕ) (h1 : y 1 = 2) 
  (h2 : ∀ k ∈ Finset.range (n - 1), y (k + 1) = y k + k + 1) : 
  Finset.sum (Finset.range n) (λ k => y (k + 1)) = 2 * n + (n - 1) * n * (n + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l990_99003


namespace NUMINAMATH_CALUDE_regression_line_estimate_l990_99028

/-- Represents a linear regression line y = ax + b -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on the regression line -/
def RegressionLine.evaluate (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

theorem regression_line_estimate :
  ∀ (line : RegressionLine),
    line.slope = 1.23 →
    line.evaluate 4 = 5 →
    line.evaluate 2 = 2.54 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_estimate_l990_99028


namespace NUMINAMATH_CALUDE_solve_for_y_l990_99076

theorem solve_for_y : ∃ y : ℚ, ((2^5 : ℚ) * y) / ((8^2 : ℚ) * (3^5 : ℚ)) = 1/6 ∧ y = 81 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l990_99076


namespace NUMINAMATH_CALUDE_california_texas_plates_equal_l990_99072

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_digits^3 * num_letters^3

/-- Theorem stating that California and Texas can issue the same number of license plates -/
theorem california_texas_plates_equal : california_plates = texas_plates := by
  sorry

end NUMINAMATH_CALUDE_california_texas_plates_equal_l990_99072


namespace NUMINAMATH_CALUDE_solve_equation_l990_99083

theorem solve_equation (x : ℝ) (h : x + 1 = 3) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l990_99083


namespace NUMINAMATH_CALUDE_ring_element_equality_l990_99031

variable {A : Type*} [Ring A] [Finite A]

theorem ring_element_equality (a b : A) (h : (a * b - 1) * b = 0) : 
  b * (a * b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ring_element_equality_l990_99031


namespace NUMINAMATH_CALUDE_janet_hourly_earnings_l990_99094

/-- Calculates Janet's hourly earnings for moderating social media posts -/
theorem janet_hourly_earnings (cents_per_post : ℚ) (seconds_per_post : ℕ) : 
  cents_per_post = 25 → seconds_per_post = 10 → 
  (3600 / seconds_per_post) * cents_per_post = 9000 := by
  sorry

#check janet_hourly_earnings

end NUMINAMATH_CALUDE_janet_hourly_earnings_l990_99094


namespace NUMINAMATH_CALUDE_cube_of_negative_two_ab_squared_l990_99099

theorem cube_of_negative_two_ab_squared (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_two_ab_squared_l990_99099


namespace NUMINAMATH_CALUDE_solution_set_cubic_inequality_l990_99039

theorem solution_set_cubic_inequality :
  {x : ℝ | x + x^3 ≥ 0} = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_cubic_inequality_l990_99039


namespace NUMINAMATH_CALUDE_squares_pattern_squares_figure_100_l990_99038

/-- The number of squares in figure n -/
def num_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

/-- The sequence of squares follows the given pattern for the first four figures -/
theorem squares_pattern :
  num_squares 0 = 1 ∧
  num_squares 1 = 7 ∧
  num_squares 2 = 19 ∧
  num_squares 3 = 37 := by sorry

/-- The number of squares in figure 100 is 30301 -/
theorem squares_figure_100 :
  num_squares 100 = 30301 := by sorry

end NUMINAMATH_CALUDE_squares_pattern_squares_figure_100_l990_99038


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l990_99008

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 2*x - 5 = 0) ↔ ((x + 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l990_99008


namespace NUMINAMATH_CALUDE_mass_of_cao_l990_99074

/-- Calculates the mass of a given number of moles of a compound -/
def calculate_mass (moles : ℝ) (atomic_mass_ca : ℝ) (atomic_mass_o : ℝ) : ℝ :=
  moles * (atomic_mass_ca + atomic_mass_o)

/-- Theorem: The mass of 8 moles of CaO containing only 42Ca is 464 grams -/
theorem mass_of_cao : calculate_mass 8 42 16 = 464 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_cao_l990_99074


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_2_l990_99033

def sumOfIntegersEndingIn2 (lower upper : ℕ) : ℕ :=
  let firstTerm := (lower + 2 - lower % 10)
  let lastTerm := (upper - upper % 10 + 2)
  let numTerms := (lastTerm - firstTerm) / 10 + 1
  numTerms * (firstTerm + lastTerm) / 2

theorem sum_of_integers_ending_in_2 :
  sumOfIntegersEndingIn2 60 460 = 10280 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_2_l990_99033


namespace NUMINAMATH_CALUDE_mean_home_runs_l990_99077

def number_of_players : ℕ := 3 + 5 + 3 + 1

def total_home_runs : ℕ := 5 * 3 + 8 * 5 + 9 * 3 + 11 * 1

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (number_of_players : ℚ) = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l990_99077


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l990_99009

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_pentagons : ℕ := n.choose k

/-- The probability of forming a convex pentagon -/
def probability : ℚ := convex_pentagons / total_selections

theorem convex_pentagon_probability :
  probability = 1 / 1755 :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l990_99009


namespace NUMINAMATH_CALUDE_power_of_power_three_l990_99068

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l990_99068


namespace NUMINAMATH_CALUDE_orange_apple_cost_l990_99089

/-- The cost of oranges and apples given specific quantities and price per kilo -/
theorem orange_apple_cost (orange_price apple_price : ℕ) 
  (orange_quantity apple_quantity : ℕ) : 
  orange_price = 29 → 
  apple_price = 29 → 
  orange_quantity = 6 → 
  apple_quantity = 5 → 
  orange_price * orange_quantity + apple_price * apple_quantity = 319 :=
by
  sorry

#check orange_apple_cost

end NUMINAMATH_CALUDE_orange_apple_cost_l990_99089


namespace NUMINAMATH_CALUDE_income_calculation_l990_99030

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 7 = expenditure * 8 →
  income = expenditure + savings →
  savings = 5000 →
  income = 40000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l990_99030


namespace NUMINAMATH_CALUDE_ryan_english_hours_l990_99081

/-- Ryan's learning schedule --/
structure LearningSchedule where
  days : ℕ
  english_total : ℕ
  chinese_daily : ℕ

/-- Calculates the daily hours spent on learning English --/
def daily_english_hours (schedule : LearningSchedule) : ℚ :=
  schedule.english_total / schedule.days

/-- Theorem stating Ryan's daily English learning hours --/
theorem ryan_english_hours (schedule : LearningSchedule) 
  (h1 : schedule.days = 2)
  (h2 : schedule.english_total = 12)
  (h3 : schedule.chinese_daily = 5) :
  daily_english_hours schedule = 6 := by
  sorry


end NUMINAMATH_CALUDE_ryan_english_hours_l990_99081


namespace NUMINAMATH_CALUDE_yeongsoo_initial_amount_l990_99004

/-- Given the initial amounts of money for Yeongsoo, Hyogeun, and Woong,
    this function returns their final amounts after the transactions. -/
def final_amounts (y h w : ℕ) : ℕ × ℕ × ℕ :=
  (y - 200 + 1000, h + 200 - 500, w + 500 - 1000)

/-- Theorem stating that Yeongsoo's initial amount was 1200 won -/
theorem yeongsoo_initial_amount :
  ∃ (h w : ℕ), final_amounts 1200 h w = (2000, 2000, 2000) :=
sorry

end NUMINAMATH_CALUDE_yeongsoo_initial_amount_l990_99004


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l990_99063

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 8 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l990_99063


namespace NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l990_99050

theorem perpendicular_vectors_tan_theta (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h2 : a = (Real.cos θ, 2))
  (h3 : b = (-1, Real.sin θ))
  (h4 : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l990_99050


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_6_l990_99073

/-- A geometric sequence with its partial sums -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- The main theorem -/
theorem geometric_sequence_sum_6 (seq : GeometricSequence) 
    (h2 : seq.S 2 = 3) (h4 : seq.S 4 = 15) : seq.S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_6_l990_99073


namespace NUMINAMATH_CALUDE_weight_of_b_l990_99085

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43) 
  (h2 : (a + b) / 2 = 40) 
  (h3 : (b + c) / 2 = 43) : 
  b = 37 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l990_99085


namespace NUMINAMATH_CALUDE_triangle_side_length_l990_99036

theorem triangle_side_length (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) →
  (|a + b - c| + |a - b - c| = 10) → 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l990_99036


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l990_99027

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ k : ℕ, 2^k ∣ (9^456 - 3^684) ∧ 
  ∀ m : ℕ, 2^m ∣ (9^456 - 3^684) → m ≤ k :=
by
  use 459
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l990_99027


namespace NUMINAMATH_CALUDE_cos_2x_value_l990_99043

theorem cos_2x_value (x : Real) (h : Real.sin (π / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l990_99043


namespace NUMINAMATH_CALUDE_basketball_lineup_theorem_l990_99017

/-- The number of ways to choose 7 starters from 18 players, including a set of 3 triplets,
    with exactly two of the triplets in the starting lineup. -/
def basketball_lineup_count : ℕ := sorry

/-- The total number of players on the team. -/
def total_players : ℕ := 18

/-- The number of triplets in the team. -/
def triplets : ℕ := 3

/-- The number of starters to be chosen. -/
def starters : ℕ := 7

/-- The number of triplets that must be in the starting lineup. -/
def triplets_in_lineup : ℕ := 2

theorem basketball_lineup_theorem : 
  basketball_lineup_count = (Nat.choose triplets triplets_in_lineup) * 
    (Nat.choose (total_players - triplets) (starters - triplets_in_lineup)) := by sorry

end NUMINAMATH_CALUDE_basketball_lineup_theorem_l990_99017


namespace NUMINAMATH_CALUDE_mikes_basketball_games_l990_99007

theorem mikes_basketball_games (points_per_game : ℕ) (total_points : ℕ) (h1 : points_per_game = 4) (h2 : total_points = 24) :
  total_points / points_per_game = 6 := by
  sorry

end NUMINAMATH_CALUDE_mikes_basketball_games_l990_99007


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l990_99029

-- Define the repeating decimal 0.3333...
def repeating_3 : ℚ := 1 / 3

-- Define the repeating decimal 0.2121...
def repeating_21 : ℚ := 7 / 33

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_3 + repeating_21 = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l990_99029


namespace NUMINAMATH_CALUDE_difference_of_squares_262_258_l990_99049

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_262_258_l990_99049


namespace NUMINAMATH_CALUDE_football_tournament_scheduling_l990_99035

theorem football_tournament_scheduling (n : ℕ) (h_even : Even n) :
  ∃ schedule : Fin (n - 1) → Fin n → Fin n,
    (∀ round : Fin (n - 1), ∀ team : Fin n, 
      schedule round team ≠ team ∧ 
      (∀ other_team : Fin n, schedule round team = other_team → schedule round other_team = team)) ∧
    (∀ team1 team2 : Fin n, team1 ≠ team2 → 
      ∃! round : Fin (n - 1), schedule round team1 = team2 ∨ schedule round team2 = team1) := by
  sorry

end NUMINAMATH_CALUDE_football_tournament_scheduling_l990_99035


namespace NUMINAMATH_CALUDE_cubic_sum_l990_99080

theorem cubic_sum (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : p * q + p * r + q * r = 3) 
  (h3 : p * q * r = -2) : 
  p^3 + q^3 + r^3 = 89 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_l990_99080


namespace NUMINAMATH_CALUDE_certain_number_problem_l990_99046

theorem certain_number_problem (h : 2994 / 14.5 = 179) : 
  ∃ x : ℝ, x / 1.45 = 17.9 ∧ x = 25.955 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l990_99046


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l990_99013

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ≥ 3 → x > 2) ∧
  ¬(∀ x : ℝ, x > 2 → x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l990_99013


namespace NUMINAMATH_CALUDE_parabola_shift_l990_99082

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 3

-- Theorem stating that the shifted parabola is the result of the described transformations
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l990_99082


namespace NUMINAMATH_CALUDE_matinee_price_correct_l990_99021

/-- The price of a matinee ticket that satisfies the given conditions -/
def matinee_price : ℝ :=
  let evening_price : ℝ := 12
  let threed_price : ℝ := 20
  let matinee_count : ℕ := 200
  let evening_count : ℕ := 300
  let threed_count : ℕ := 100
  let total_revenue : ℝ := 6600
  5

/-- Theorem stating that the matinee price satisfies the given conditions -/
theorem matinee_price_correct :
  let evening_price : ℝ := 12
  let threed_price : ℝ := 20
  let matinee_count : ℕ := 200
  let evening_count : ℕ := 300
  let threed_count : ℕ := 100
  let total_revenue : ℝ := 6600
  matinee_price * matinee_count + evening_price * evening_count + threed_price * threed_count = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_matinee_price_correct_l990_99021


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_l990_99069

/-- Conference attendees -/
structure Conference where
  total : ℕ
  writers : ℕ
  editors : ℕ
  both : ℕ
  neither : ℕ

/-- Conference constraints -/
def valid_conference (c : Conference) : Prop :=
  c.total = 100 ∧
  c.writers = 35 ∧
  c.editors > 38 ∧
  c.neither = 2 * c.both ∧
  c.total = c.writers + c.editors - c.both + c.neither

/-- Theorem: The maximum number of people who can be both writers and editors is 26 -/
theorem max_both_writers_and_editors (c : Conference) (h : valid_conference c) :
  c.both ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_max_both_writers_and_editors_l990_99069


namespace NUMINAMATH_CALUDE_triangle_property_l990_99015

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b² + c² - a²) / cos A = 2 and (a cos B - b cos A) / (a cos B + b cos A) - b/c = 1,
    then bc = 1 and the area of triangle ABC is √3/4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  ((b^2 + c^2 - a^2) / Real.cos A = 2) →
  ((a * Real.cos B - b * Real.cos A) / (a * Real.cos B + b * Real.cos A) - b / c = 1) →
  (b * c = 1 ∧ (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l990_99015


namespace NUMINAMATH_CALUDE_original_equals_scientific_l990_99087

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 280000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 2.8
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l990_99087


namespace NUMINAMATH_CALUDE_deck_size_l990_99058

/-- The number of cards in a deck of playing cards. -/
def num_cards : ℕ := 52

/-- The number of hearts on each card. -/
def hearts_per_card : ℕ := 4

/-- The cost of each cow in dollars. -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars. -/
def total_cost : ℕ := 83200

/-- The number of cows in Devonshire. -/
def num_cows : ℕ := total_cost / cost_per_cow

/-- The number of hearts in the deck. -/
def num_hearts : ℕ := num_cows / 2

theorem deck_size :
  num_cards = num_hearts / hearts_per_card ∧
  num_cows = 2 * num_hearts ∧
  num_cows * cost_per_cow = total_cost :=
by sorry

end NUMINAMATH_CALUDE_deck_size_l990_99058


namespace NUMINAMATH_CALUDE_log2_derivative_l990_99064

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log2_derivative_l990_99064


namespace NUMINAMATH_CALUDE_circular_sector_properties_l990_99025

/-- A circular sector with given area and perimeter -/
structure CircularSector where
  area : ℝ
  perimeter : ℝ

/-- The central angle of a circular sector -/
def central_angle (s : CircularSector) : ℝ := sorry

/-- The chord length of a circular sector -/
def chord_length (s : CircularSector) : ℝ := sorry

/-- Theorem stating the properties of a specific circular sector -/
theorem circular_sector_properties :
  let s : CircularSector := { area := 1, perimeter := 4 }
  (central_angle s = 2) ∧ (chord_length s = 2 * Real.sin 1) := by sorry

end NUMINAMATH_CALUDE_circular_sector_properties_l990_99025


namespace NUMINAMATH_CALUDE_shara_shells_after_vacation_l990_99057

/-- Calculates the total number of shells after a vacation -/
def total_shells_after_vacation (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (fourth_day_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + fourth_day_shells

/-- Proves that Shara has 41 shells after her vacation -/
theorem shara_shells_after_vacation :
  total_shells_after_vacation 20 5 3 6 = 41 := by
  sorry

end NUMINAMATH_CALUDE_shara_shells_after_vacation_l990_99057


namespace NUMINAMATH_CALUDE_widget_earnings_calculation_l990_99059

/-- Calculates the earnings per widget given the hourly wage, work hours, 
    required widget production, and total weekly earnings. -/
def earnings_per_widget (hourly_wage : ℚ) (work_hours : ℕ) 
  (required_widgets : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - hourly_wage * work_hours) / required_widgets

theorem widget_earnings_calculation : 
  earnings_per_widget (12.5) 40 500 580 = (16 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_widget_earnings_calculation_l990_99059


namespace NUMINAMATH_CALUDE_camp_skills_l990_99070

theorem camp_skills (total : ℕ) (cant_sing cant_dance cant_perform : ℕ) :
  total = 100 ∧
  cant_sing = 42 ∧
  cant_dance = 65 ∧
  cant_perform = 29 →
  ∃ (only_sing only_dance only_perform sing_dance sing_perform dance_perform : ℕ),
    only_sing + only_dance + only_perform + sing_dance + sing_perform + dance_perform = total ∧
    only_dance + only_perform + dance_perform = cant_sing ∧
    only_sing + only_perform + sing_perform = cant_dance ∧
    only_sing + only_dance + sing_dance = cant_perform ∧
    sing_dance + sing_perform + dance_perform = 64 :=
by sorry

end NUMINAMATH_CALUDE_camp_skills_l990_99070


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l990_99054

theorem two_digit_number_sum (tens : ℕ) (units : ℕ) : 
  tens < 10 → 
  units = 6 * tens → 
  10 * tens + units = 16 → 
  tens + units = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l990_99054


namespace NUMINAMATH_CALUDE_sum_of_roots_even_function_l990_99006

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define a function that has exactly four roots
def HasFourRoots (f : ℝ → ℝ) : Prop := ∃ a b c d : ℝ, 
  (a < b ∧ b < c ∧ c < d) ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem sum_of_roots_even_function (f : ℝ → ℝ) 
  (h_even : EvenFunction f) (h_four_roots : HasFourRoots f) : 
  ∃ a b c d : ℝ, (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧ (a + b + c + d = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_function_l990_99006


namespace NUMINAMATH_CALUDE_sport_popularity_order_l990_99051

/-- Represents a sport with its popularity fraction -/
structure Sport where
  name : String
  popularity : Rat

/-- Determines if one sport is more popular than another -/
def morePopularThan (s1 s2 : Sport) : Prop :=
  s1.popularity > s2.popularity

theorem sport_popularity_order (basketball tennis volleyball : Sport)
  (h_basketball : basketball.name = "Basketball" ∧ basketball.popularity = 9/24)
  (h_tennis : tennis.name = "Tennis" ∧ tennis.popularity = 8/24)
  (h_volleyball : volleyball.name = "Volleyball" ∧ volleyball.popularity = 7/24) :
  morePopularThan basketball tennis ∧ 
  morePopularThan tennis volleyball ∧
  [basketball.name, tennis.name, volleyball.name] = ["Basketball", "Tennis", "Volleyball"] :=
by sorry

end NUMINAMATH_CALUDE_sport_popularity_order_l990_99051


namespace NUMINAMATH_CALUDE_farm_has_eleven_goats_l990_99090

/-- Represents the number of animals on a farm -/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Defines the conditions of the farm -/
def valid_farm (f : Farm) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows = f.goats + 4 ∧
  f.goats + f.cows + f.pigs = 56

/-- Theorem stating that a valid farm has 11 goats -/
theorem farm_has_eleven_goats (f : Farm) (h : valid_farm f) : f.goats = 11 := by
  sorry

#check farm_has_eleven_goats

end NUMINAMATH_CALUDE_farm_has_eleven_goats_l990_99090


namespace NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l990_99024

theorem quadratic_maximum (s : ℝ) : -7 * s^2 + 56 * s - 18 ≤ 94 := by sorry

theorem quadratic_maximum_achieved : ∃ s : ℝ, -7 * s^2 + 56 * s - 18 = 94 := by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l990_99024


namespace NUMINAMATH_CALUDE_percentage_calculation_l990_99075

theorem percentage_calculation (total : ℝ) (part : ℝ) (h1 : total = 500) (h2 : part = 125) :
  (part / total) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l990_99075


namespace NUMINAMATH_CALUDE_point_P_y_coordinate_l990_99010

theorem point_P_y_coordinate :
  ∀ (x y : ℝ),
  (|y| = (1/2) * |x|) →  -- Distance from x-axis is half the distance from y-axis
  (|x| = 18) →           -- Point P is 18 units from y-axis
  y = 9 := by            -- The y-coordinate of point P is 9
sorry

end NUMINAMATH_CALUDE_point_P_y_coordinate_l990_99010


namespace NUMINAMATH_CALUDE_divisibility_condition_l990_99034

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
    (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l990_99034


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l990_99023

theorem prime_power_divisibility : 
  (∃ p : ℕ, p ≥ 7 ∧ Nat.Prime p ∧ (p^4 - 1) % 48 = 0) ∧ 
  (∃ q : ℕ, q ≥ 7 ∧ Nat.Prime q ∧ (q^4 - 1) % 48 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l990_99023


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l990_99078

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- The number to be represented -/
def number : ℕ := 2400000

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  a := 2.4
  n := 6
  h1 := by sorry
  h2 := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem correct_scientific_notation : 
  (scientificForm.a * (10 : ℝ) ^ scientificForm.n) = number := by sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l990_99078


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l990_99097

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, x^2 + b*x - 3 = 0 ↔ x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l990_99097


namespace NUMINAMATH_CALUDE_root_of_cubic_equation_l990_99005

theorem root_of_cubic_equation :
  ∃ x : ℝ, (1/2 : ℝ) * x^3 + 4 = 0 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_of_cubic_equation_l990_99005


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l990_99060

/-- Given points A and B, and vector a, proves that if AB is perpendicular to a, then k = 1 -/
theorem perpendicular_vectors_k_value (k : ℝ) :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, k)
  let a : ℝ × ℝ := (-1, 2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (AB.1 * a.1 + AB.2 * a.2 = 0) → k = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l990_99060


namespace NUMINAMATH_CALUDE_all_digits_appear_as_cube_units_l990_99053

theorem all_digits_appear_as_cube_units : ∀ d : Nat, d < 10 → ∃ n : Nat, n^3 % 10 = d := by
  sorry

end NUMINAMATH_CALUDE_all_digits_appear_as_cube_units_l990_99053


namespace NUMINAMATH_CALUDE_inverse_difference_l990_99026

theorem inverse_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x * y + 1) :
  1 / x - 1 / y = -1 - 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_l990_99026


namespace NUMINAMATH_CALUDE_circle_circumference_area_equal_diameter_l990_99011

/-- When the circumference and area of a circle are numerically equal, the diameter is 4. -/
theorem circle_circumference_area_equal_diameter (r : ℝ) :
  2 * Real.pi * r = Real.pi * r^2 → 2 * r = 4 := by sorry

end NUMINAMATH_CALUDE_circle_circumference_area_equal_diameter_l990_99011


namespace NUMINAMATH_CALUDE_segment_length_specific_case_l990_99002

/-- A rectangle with an inscribed circle and a diagonal intersecting the circle -/
structure RectangleWithCircle where
  /-- Length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- Length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The circle is tangent to three sides of the rectangle -/
  circle_tangent : Bool
  /-- The diagonal intersects the circle at two points -/
  diagonal_intersects : Bool

/-- The length of the segment AB formed by the intersection of the diagonal with the circle -/
def segment_length (r : RectangleWithCircle) : ℝ :=
  sorry

/-- Theorem stating the length of AB in the specific case -/
theorem segment_length_specific_case :
  let r : RectangleWithCircle := {
    short_side := 2,
    long_side := 4,
    circle_tangent := true,
    diagonal_intersects := true
  }
  segment_length r = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_segment_length_specific_case_l990_99002


namespace NUMINAMATH_CALUDE_final_water_level_change_l990_99014

def water_level_change (initial_change : ℝ) (subsequent_change : ℝ) : ℝ :=
  initial_change + subsequent_change

theorem final_water_level_change :
  water_level_change (-3) 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_water_level_change_l990_99014


namespace NUMINAMATH_CALUDE_joes_pocket_money_l990_99086

theorem joes_pocket_money (initial_money : ℚ) : 
  (1 - (1/9 + 2/5)) * initial_money = 220 → initial_money = 450 := by
  sorry

end NUMINAMATH_CALUDE_joes_pocket_money_l990_99086


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_over_twelve_l990_99093

theorem cos_squared_minus_sin_squared_pi_over_twelve (π : Real) :
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_over_twelve_l990_99093


namespace NUMINAMATH_CALUDE_hypotenuse_length_l990_99079

/-- A right triangle with specific medians -/
structure RightTriangle where
  /-- Length of one leg -/
  x : ℝ
  /-- Length of the other leg -/
  y : ℝ
  /-- The triangle is right-angled -/
  right_angle : x ^ 2 + y ^ 2 > 0
  /-- One median has length 3 -/
  median1 : x ^ 2 + (y / 2) ^ 2 = 3 ^ 2
  /-- The other median has length 2√13 -/
  median2 : y ^ 2 + (x / 2) ^ 2 = (2 * Real.sqrt 13) ^ 2

/-- The hypotenuse of the right triangle is 8√1.1 -/
theorem hypotenuse_length (t : RightTriangle) : 
  Real.sqrt (4 * (t.x ^ 2 + t.y ^ 2)) = 8 * Real.sqrt 1.1 := by
  sorry

#check hypotenuse_length

end NUMINAMATH_CALUDE_hypotenuse_length_l990_99079


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l990_99065

/-- Calculates the simple interest for a loan -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem annual_interest_calculation (loan_amount : ℝ) (interest_rate : ℝ) :
  loan_amount = 9000 →
  interest_rate = 0.09 →
  simple_interest loan_amount interest_rate 1 = 810 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l990_99065


namespace NUMINAMATH_CALUDE_candy_boxes_problem_l990_99045

/-- Given that Paul bought 6 boxes of chocolate candy and 4 boxes of caramel candy,
    with a total of 90 candies, and each box contains the same number of pieces,
    prove that there are 9 pieces of candy in each box. -/
theorem candy_boxes_problem (pieces_per_box : ℕ) : 
  (6 * pieces_per_box + 4 * pieces_per_box = 90) → pieces_per_box = 9 := by
sorry

end NUMINAMATH_CALUDE_candy_boxes_problem_l990_99045


namespace NUMINAMATH_CALUDE_modified_triangle_invalid_zero_area_l990_99092

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- The original triangle ABC -/
def originalTriangle : Triangle :=
  { a := 12, b := 7, c := 10 }

/-- The modified triangle with doubled AB and AC -/
def modifiedTriangle : Triangle :=
  { a := 24, b := 14, c := 10 }

/-- Theorem stating that the modified triangle is not valid and has zero area -/
theorem modified_triangle_invalid_zero_area :
  ¬(isValidTriangle modifiedTriangle) ∧ 
  (∃ area : ℝ, area = 0 ∧ area ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_modified_triangle_invalid_zero_area_l990_99092


namespace NUMINAMATH_CALUDE_price_restoration_l990_99019

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (increase_percentage : ℝ) : 
  reduced_price = original_price * (1 - 0.2) →
  reduced_price * (1 + increase_percentage) = original_price →
  increase_percentage = 0.25 := by
  sorry

#check price_restoration

end NUMINAMATH_CALUDE_price_restoration_l990_99019


namespace NUMINAMATH_CALUDE_unique_solution_3x_minus_5y_equals_z2_l990_99055

theorem unique_solution_3x_minus_5y_equals_z2 :
  ∀ x y z : ℕ+, 3^(x : ℕ) - 5^(y : ℕ) = (z : ℕ)^2 → x = 2 ∧ y = 2 ∧ z = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_3x_minus_5y_equals_z2_l990_99055


namespace NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l990_99020

/-- Given two distinct natural numbers, the maximum number of perfect squares
    among the pairwise products of these numbers and their +2 counterparts is 2. -/
theorem max_perfect_squares_pairwise_products (a b : ℕ) (h : a ≠ b) :
  let products := {a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y, x = y^2) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y, x = y^2) → s.card ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l990_99020


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l990_99044

theorem negative_fraction_comparison : -3/5 < -4/7 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l990_99044


namespace NUMINAMATH_CALUDE_triangular_number_all_equal_digits_l990_99001

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def all_digits_equal (num : ℕ) (digit : ℕ) : Prop :=
  ∀ d, d ∈ num.digits 10 → d = digit

theorem triangular_number_all_equal_digits :
  {a : ℕ | a < 10 ∧ ∃ n : ℕ, n ≥ 4 ∧ all_digits_equal (triangular_number n) a} = {5, 6} := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_all_equal_digits_l990_99001


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l990_99041

theorem x_minus_y_equals_three (x y : ℝ) (h : |x - 2| + (y + 1)^2 = 0) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l990_99041


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l990_99016

theorem sum_of_squares_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l990_99016


namespace NUMINAMATH_CALUDE_quadrilateral_area_l990_99056

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_at_B_and_D (q : Quadrilateral) : Prop := sorry
def diagonal_length (q : Quadrilateral) (p1 p2 : ℝ × ℝ) : ℝ := sorry
def side_length (q : Quadrilateral) (p1 p2 : ℝ × ℝ) : ℝ := sorry
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area 
  (q : Quadrilateral)
  (h1 : is_right_angled_at_B_and_D q)
  (h2 : diagonal_length q q.A q.C = 5)
  (h3 : side_length q q.B q.C = 4)
  (h4 : side_length q q.A q.D = 3) :
  area q = 12 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l990_99056


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_3_plus_2sqrt2_l990_99088

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_equals_3_plus_2sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a + 2*b = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_equals_3_plus_2sqrt2_l990_99088


namespace NUMINAMATH_CALUDE_city_population_problem_l990_99096

theorem city_population_problem (p : ℝ) : 
  0.85 * (p + 800) = p + 824 ↔ p = 960 :=
by sorry

end NUMINAMATH_CALUDE_city_population_problem_l990_99096


namespace NUMINAMATH_CALUDE_two_digit_number_equation_l990_99061

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- The property that the unit digit is 3 greater than the tens digit -/
def unit_is_three_greater (n : TwoDigitNumber) : Prop :=
  n.units = n.tens + 3

/-- The property that the square of the unit digit equals the two-digit number -/
def square_of_unit_is_number (n : TwoDigitNumber) : Prop :=
  n.units ^ 2 = 10 * n.tens + n.units

/-- Theorem: For a two-digit number satisfying the given conditions, 
    the tens digit x satisfies the equation x^2 - 5x + 6 = 0 -/
theorem two_digit_number_equation (n : TwoDigitNumber) 
  (h1 : unit_is_three_greater n) 
  (h2 : square_of_unit_is_number n) : 
  n.tens ^ 2 - 5 * n.tens + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_equation_l990_99061


namespace NUMINAMATH_CALUDE_square_division_theorem_l990_99032

-- Define the square and point P
def Square (E F G H : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := E
  let (x₂, y₂) := F
  let (x₃, y₃) := G
  let (x₄, y₄) := H
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16 ∧
  (x₃ - x₂)^2 + (y₃ - y₂)^2 = 16 ∧
  (x₄ - x₃)^2 + (y₄ - y₃)^2 = 16 ∧
  (x₁ - x₄)^2 + (y₁ - y₄)^2 = 16

def PointOnSide (P : ℝ × ℝ) (E H : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * E.1 + (1 - t) * H.1, t * E.2 + (1 - t) * H.2)

-- Define the area division property
def DivideAreaEqually (P : ℝ × ℝ) (E F G H : ℝ × ℝ) : Prop :=
  let area_EFP := abs ((F.1 - E.1) * (P.2 - E.2) - (P.1 - E.1) * (F.2 - E.2)) / 2
  let area_FGP := abs ((G.1 - F.1) * (P.2 - F.2) - (P.1 - F.1) * (G.2 - F.2)) / 2
  let area_GHP := abs ((H.1 - G.1) * (P.2 - G.2) - (P.1 - G.1) * (H.2 - G.2)) / 2
  let area_HEP := abs ((E.1 - H.1) * (P.2 - H.2) - (P.1 - H.1) * (E.2 - H.2)) / 2
  area_EFP = area_FGP ∧ area_FGP = area_GHP ∧ area_GHP = area_HEP

-- State the theorem
theorem square_division_theorem (E F G H P : ℝ × ℝ) :
  Square E F G H →
  PointOnSide P E H →
  DivideAreaEqually P E F G H →
  (F.1 - P.1)^2 + (F.2 - P.2)^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l990_99032


namespace NUMINAMATH_CALUDE_graces_initial_fruits_l990_99095

/-- The number of Graces --/
def num_graces : ℕ := 3

/-- The number of Muses --/
def num_muses : ℕ := 9

/-- Represents the distribution of fruits --/
structure FruitDistribution where
  initial_grace : ℕ  -- Initial number of fruits each Grace had
  given_to_muse : ℕ  -- Number of fruits each Grace gave to each Muse

/-- Theorem stating the conditions and the result to be proved --/
theorem graces_initial_fruits (fd : FruitDistribution) : 
  -- Each Grace gives fruits to each Muse
  (fd.initial_grace ≥ num_muses * fd.given_to_muse) →
  -- After exchange, Graces and Muses have the same number of fruits
  (fd.initial_grace - num_muses * fd.given_to_muse = num_graces * fd.given_to_muse) →
  -- Initial number of fruits each Grace had is 12
  fd.initial_grace = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_graces_initial_fruits_l990_99095


namespace NUMINAMATH_CALUDE_coefficient_of_y_squared_l990_99052

def polynomial (x : ℝ) : ℝ :=
  (1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11 + x^12 - x^13 + x^14 - x^15 + x^16 - x^17)

def y (x : ℝ) : ℝ := x + 1

theorem coefficient_of_y_squared (x : ℝ) : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ a₁₅ a₁₆ a₁₇ : ℝ), 
    polynomial x = a₀ + a₁ * y x + a₂ * (y x)^2 + a₃ * (y x)^3 + a₄ * (y x)^4 + 
                   a₅ * (y x)^5 + a₆ * (y x)^6 + a₇ * (y x)^7 + a₈ * (y x)^8 + 
                   a₉ * (y x)^9 + a₁₀ * (y x)^10 + a₁₁ * (y x)^11 + a₁₂ * (y x)^12 + 
                   a₁₃ * (y x)^13 + a₁₄ * (y x)^14 + a₁₅ * (y x)^15 + a₁₆ * (y x)^16 + 
                   a₁₇ * (y x)^17 ∧
    a₂ = 816 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_y_squared_l990_99052


namespace NUMINAMATH_CALUDE_inequality_solution_and_sqrt2_l990_99022

-- Define the inequality
def inequality (x : ℝ) : Prop := (5/2 * x - 1) > 3 * x

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2}

-- Theorem statement
theorem inequality_solution_and_sqrt2 :
  (∀ x, inequality x ↔ x ∈ solution_set) ∧
  ¬ inequality (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_and_sqrt2_l990_99022


namespace NUMINAMATH_CALUDE_lesser_fraction_l990_99071

theorem lesser_fraction (x y : ℝ) (sum_eq : x + y = 7/8) (prod_eq : x * y = 1/12) :
  min x y = (7 - Real.sqrt 17) / 16 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l990_99071
