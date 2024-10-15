import Mathlib

namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3331_333128

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3331_333128


namespace NUMINAMATH_CALUDE_main_theorem_l3331_333188

/-- Proposition p -/
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0

/-- Proposition q -/
def q (x m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

/-- Set A: values of x that satisfy p -/
def A : Set ℝ := {x | p x}

/-- Set B: values of x that satisfy q -/
def B (m : ℝ) : Set ℝ := {x | q x m}

/-- Main theorem: If ¬q is a sufficient but not necessary condition for ¬p,
    then m > 1 or m < -2 -/
theorem main_theorem (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, p x ∧ q x m) →
  m > 1 ∨ m < -2 :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l3331_333188


namespace NUMINAMATH_CALUDE_max_days_to_eat_candies_l3331_333158

/-- The total number of candies Vasya received -/
def total_candies : ℕ := 777

/-- The function that calculates the total number of candies eaten over n days,
    where a is the number of candies eaten on the first day -/
def candies_eaten (n a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- The proposition that states 37 is the maximum number of days
    in which Vasya can eat all the candies -/
theorem max_days_to_eat_candies :
  ∃ (a : ℕ), candies_eaten 37 a = total_candies ∧
  ∀ (n : ℕ), n > 37 → ∀ (b : ℕ), candies_eaten n b ≠ total_candies :=
sorry

end NUMINAMATH_CALUDE_max_days_to_eat_candies_l3331_333158


namespace NUMINAMATH_CALUDE_dividend_rate_for_given_stock_l3331_333118

/-- Represents a stock with its characteristics -/
structure Stock where
  nominal_percentage : ℝ  -- The nominal percentage of the stock
  quote : ℝ             -- The quoted price of the stock
  yield : ℝ              -- The yield of the stock as a percentage

/-- Calculates the dividend rate of a stock -/
def dividend_rate (s : Stock) : ℝ :=
  s.nominal_percentage

/-- Theorem stating that for a 25% stock quoted at 125 with a 20% yield, the dividend rate is 25 -/
theorem dividend_rate_for_given_stock :
  let s : Stock := { nominal_percentage := 25, quote := 125, yield := 20 }
  dividend_rate s = 25 := by
  sorry


end NUMINAMATH_CALUDE_dividend_rate_for_given_stock_l3331_333118


namespace NUMINAMATH_CALUDE_puzzle_solution_l3331_333138

/-- A function that represents the puzzle rule --/
def puzzleRule (n : ℕ) : ℕ := sorry

/-- The puzzle conditions --/
axiom rule_111 : puzzleRule 111 = 9
axiom rule_444 : puzzleRule 444 = 12
axiom rule_777 : puzzleRule 777 = 15

/-- The theorem to prove --/
theorem puzzle_solution : ∃ (n : ℕ), puzzleRule n = 15 ∧ n = 777 := by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3331_333138


namespace NUMINAMATH_CALUDE_simplify_fraction_l3331_333173

theorem simplify_fraction : (120 : ℚ) / 1800 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3331_333173


namespace NUMINAMATH_CALUDE_circle_C_properties_l3331_333141

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y = 0
def line2 (x y : ℝ) : Prop := x - y - 4 = 0
def line3 (x y : ℝ) : Prop := x + y = 0

-- Define tangency
def is_tangent (circle : (ℝ → ℝ → Prop)) (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ line x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - x)^2 + (y' - y)^2 ≥ 2

-- State the theorem
theorem circle_C_properties :
  is_tangent circle_C line1 ∧
  is_tangent circle_C line2 ∧
  ∃ (x y : ℝ), circle_C x y ∧ line3 x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l3331_333141


namespace NUMINAMATH_CALUDE_expression_has_17_digits_l3331_333115

-- Define a function to calculate the number of digits in a number
def numDigits (n : ℕ) : ℕ := sorry

-- Define the expression
def expression : ℕ := (8 * 10^10) * (10 * 10^5)

-- Theorem statement
theorem expression_has_17_digits : numDigits expression = 17 := by sorry

end NUMINAMATH_CALUDE_expression_has_17_digits_l3331_333115


namespace NUMINAMATH_CALUDE_lap_distance_l3331_333153

theorem lap_distance (boys_laps girls_laps girls_miles : ℚ) : 
  boys_laps = 27 →
  girls_laps = boys_laps + 9 →
  girls_miles = 27 →
  girls_miles / girls_laps = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_lap_distance_l3331_333153


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3331_333126

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) 
  (h2 : hyperbola a (-2) 1) :
  ∃ (k : ℝ), k = 1 ∨ k = -1 ∧ 
  ∀ (x y : ℝ), (x + k*y = 0) ↔ (∀ ε > 0, ∃ t > 0, ∀ t' ≥ t, 
    ∃ x' y', hyperbola a x' y' ∧ 
    ((x' - x)^2 + (y' - y)^2 < ε^2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3331_333126


namespace NUMINAMATH_CALUDE_solve_system_l3331_333190

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : 2 * x + y = 21) : 
  y = 27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3331_333190


namespace NUMINAMATH_CALUDE_plant_prices_and_minimum_cost_l3331_333176

/-- The price of a pot of green radish -/
def green_radish_price : ℝ := 4

/-- The price of a pot of spider plant -/
def spider_plant_price : ℝ := 12

/-- The total number of pots to be purchased -/
def total_pots : ℕ := 120

/-- The number of spider plant pots that minimizes the total cost -/
def optimal_spider_pots : ℕ := 80

/-- The number of green radish pots that minimizes the total cost -/
def optimal_green_radish_pots : ℕ := 40

theorem plant_prices_and_minimum_cost :
  (green_radish_price + spider_plant_price = 16) ∧
  (80 / green_radish_price = 2 * (120 / spider_plant_price)) ∧
  (optimal_spider_pots + optimal_green_radish_pots = total_pots) ∧
  (optimal_green_radish_pots ≤ optimal_spider_pots / 2) ∧
  (∀ a : ℕ, a + (total_pots - a) = total_pots →
    (total_pots - a) ≤ a / 2 →
    spider_plant_price * a + green_radish_price * (total_pots - a) ≥
    spider_plant_price * optimal_spider_pots + green_radish_price * optimal_green_radish_pots) :=
by sorry

end NUMINAMATH_CALUDE_plant_prices_and_minimum_cost_l3331_333176


namespace NUMINAMATH_CALUDE_same_terminal_side_470_110_l3331_333134

/-- Two angles have the same terminal side if their difference is a multiple of 360° --/
def same_terminal_side (α β : ℝ) : Prop := ∃ k : ℤ, α = β + k * 360

/-- The theorem states that 470° has the same terminal side as 110° --/
theorem same_terminal_side_470_110 : same_terminal_side 470 110 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_470_110_l3331_333134


namespace NUMINAMATH_CALUDE_sequence_expression_l3331_333194

theorem sequence_expression (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) - 2 * a n = 2^n) →
  ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_expression_l3331_333194


namespace NUMINAMATH_CALUDE_mikes_games_this_year_l3331_333117

def total_games : ℕ := 54
def last_year_games : ℕ := 39
def missed_games : ℕ := 41

theorem mikes_games_this_year : 
  total_games - last_year_games = 15 := by sorry

end NUMINAMATH_CALUDE_mikes_games_this_year_l3331_333117


namespace NUMINAMATH_CALUDE_show_revenue_l3331_333103

/-- Calculate the total revenue from two shows given the number of attendees and ticket price -/
theorem show_revenue (first_show_attendees : ℕ) (ticket_price : ℕ) : 
  first_show_attendees * ticket_price + (3 * first_show_attendees) * ticket_price = 20000 :=
by
  sorry

#check show_revenue 200 25

end NUMINAMATH_CALUDE_show_revenue_l3331_333103


namespace NUMINAMATH_CALUDE_distinct_prime_factors_sum_and_product_l3331_333175

def number : Nat := 420

theorem distinct_prime_factors_sum_and_product :
  (Finset.sum (Nat.factors number).toFinset id = 17) ∧
  (Finset.prod (Nat.factors number).toFinset id = 210) := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_sum_and_product_l3331_333175


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l3331_333164

theorem solve_cubic_equation :
  ∃ x : ℝ, x = -15.625 ∧ 3 * x^(1/3) - 5 * (x / x^(2/3)) = 10 + 2 * x^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l3331_333164


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3331_333146

theorem fruit_basket_count : 
  let apples_per_basket : ℕ := 9
  let oranges_per_basket : ℕ := 15
  let bananas_per_basket : ℕ := 14
  let num_baskets : ℕ := 4
  let fruits_per_basket : ℕ := apples_per_basket + oranges_per_basket + bananas_per_basket
  let fruits_in_three_baskets : ℕ := 3 * fruits_per_basket
  let fruits_in_fourth_basket : ℕ := fruits_per_basket - 6
  fruits_in_three_baskets + fruits_in_fourth_basket = 70
  := by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3331_333146


namespace NUMINAMATH_CALUDE_rectangle_area_l3331_333192

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3331_333192


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3331_333137

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 * y = k) (h4 : 2^2 * 10 = k) (h5 : y = 4000) : x = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3331_333137


namespace NUMINAMATH_CALUDE_hat_number_sum_l3331_333106

/-- A four-digit perfect square number with tens digit 0 and non-zero units digit -/
def ValidNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^2 ∧ (n / 10) % 10 = 0 ∧ n % 10 ≠ 0

/-- The property that two numbers have the same units digit -/
def SameUnitsDigit (a b : ℕ) : Prop := a % 10 = b % 10

/-- The property that a number has an even units digit -/
def EvenUnitsDigit (n : ℕ) : Prop := n % 2 = 0

theorem hat_number_sum :
  ∀ a b c : ℕ,
    ValidNumber a ∧ ValidNumber b ∧ ValidNumber c ∧
    SameUnitsDigit b c ∧
    EvenUnitsDigit a ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a + b + c = 14612 := by
  sorry

end NUMINAMATH_CALUDE_hat_number_sum_l3331_333106


namespace NUMINAMATH_CALUDE_singing_competition_stats_l3331_333159

def scores : List ℝ := [9.40, 9.40, 9.50, 9.50, 9.50, 9.60, 9.60, 9.60, 9.60, 9.60, 9.70, 9.70, 9.70, 9.70, 9.80, 9.80, 9.80, 9.90]

def median (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem singing_competition_stats :
  median scores = 9.60 ∧ mode scores = 9.60 := by sorry

end NUMINAMATH_CALUDE_singing_competition_stats_l3331_333159


namespace NUMINAMATH_CALUDE_arithmetic_combination_exists_l3331_333116

theorem arithmetic_combination_exists : ∃ (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ) (h : ℕ → ℕ → ℕ),
  (f 1 (g 2 3)) * (h 4 5) = 100 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_combination_exists_l3331_333116


namespace NUMINAMATH_CALUDE_inverse_mod_million_l3331_333156

theorem inverse_mod_million (C D M : Nat) : 
  C = 123456 → 
  D = 142857 → 
  M = 814815 → 
  (C * D * M) % 1000000 = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_mod_million_l3331_333156


namespace NUMINAMATH_CALUDE_rabbit_travel_time_l3331_333184

/-- Proves that a rabbit traveling at 5 miles per hour takes 24 minutes to cover 2 miles -/
theorem rabbit_travel_time :
  let speed : ℝ := 5  -- miles per hour
  let distance : ℝ := 2  -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 24 := by sorry

end NUMINAMATH_CALUDE_rabbit_travel_time_l3331_333184


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3331_333107

theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 3^n + 2) 
  (h2 : r = 4^s - s) 
  (h3 : n = 3) : 
  r = 4^29 - 29 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3331_333107


namespace NUMINAMATH_CALUDE_vector_subtraction_l3331_333162

theorem vector_subtraction :
  let v₁ : Fin 3 → ℝ := ![-2, 5, -1]
  let v₂ : Fin 3 → ℝ := ![7, -3, 6]
  v₁ - v₂ = ![-9, 8, -7] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3331_333162


namespace NUMINAMATH_CALUDE_children_count_l3331_333197

theorem children_count (pencils_per_child : ℕ) (skittles_per_child : ℕ) (total_pencils : ℕ) : 
  pencils_per_child = 2 → 
  skittles_per_child = 13 → 
  total_pencils = 18 → 
  total_pencils / pencils_per_child = 9 := by
sorry

end NUMINAMATH_CALUDE_children_count_l3331_333197


namespace NUMINAMATH_CALUDE_arc_ray_configuration_theorem_l3331_333149

/-- Given a geometric configuration with circular arcs and rays, 
    we define constants u_ij and v_ij. This theorem proves a relationship between these constants. -/
theorem arc_ray_configuration_theorem 
  (u12 v12 u23 v23 : ℝ) 
  (h1 : u12 = v12) 
  (h2 : u12 = v23) 
  (h3 : u23 = v12) : 
  u23 = v23 := by sorry

end NUMINAMATH_CALUDE_arc_ray_configuration_theorem_l3331_333149


namespace NUMINAMATH_CALUDE_event_A_necessary_not_sufficient_for_B_l3331_333139

/- Define the bag contents -/
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

/- Define the events -/
def event_A (drawn_red : ℕ) : Prop := drawn_red ≥ 1
def event_B (drawn_red : ℕ) : Prop := drawn_red = 1

/- Define the relationship between events -/
theorem event_A_necessary_not_sufficient_for_B :
  (∀ (drawn_red : ℕ), event_B drawn_red → event_A drawn_red) ∧
  (∃ (drawn_red : ℕ), event_A drawn_red ∧ ¬event_B drawn_red) :=
sorry

end NUMINAMATH_CALUDE_event_A_necessary_not_sufficient_for_B_l3331_333139


namespace NUMINAMATH_CALUDE_annual_car_insurance_cost_l3331_333198

/-- Theorem: If a person spends 40000 dollars on car insurance over a decade,
    then their annual car insurance cost is 4000 dollars. -/
theorem annual_car_insurance_cost (total_cost : ℕ) (years : ℕ) (annual_cost : ℕ) :
  total_cost = 40000 →
  years = 10 →
  annual_cost = total_cost / years →
  annual_cost = 4000 := by
  sorry

end NUMINAMATH_CALUDE_annual_car_insurance_cost_l3331_333198


namespace NUMINAMATH_CALUDE_polynomial_equality_l3331_333199

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x - 2) * (x + 3) = x^2 + a*x + b) → (a = 1 ∧ b = -6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3331_333199


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3331_333183

theorem exponent_multiplication (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3331_333183


namespace NUMINAMATH_CALUDE_subtraction_problem_l3331_333166

theorem subtraction_problem : 943 - 87 = 856 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3331_333166


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l3331_333168

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 6752 :=
sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l3331_333168


namespace NUMINAMATH_CALUDE_complex_magnitude_l3331_333119

theorem complex_magnitude (z : ℂ) (h : z * (1 + 2*I) = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3331_333119


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_range_l3331_333172

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 + a * log x

-- State the theorem
theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (∀ x : ℝ, 0 < x → (deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂))) →
  0 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_range_l3331_333172


namespace NUMINAMATH_CALUDE_b_profit_is_4000_l3331_333130

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  total_profit : ℕ
  a_investment_ratio : ℕ
  a_time_ratio : ℕ

/-- Calculates B's profit in the joint business venture -/
def calculate_b_profit (jb : JointBusiness) : ℕ :=
  jb.total_profit / (jb.a_investment_ratio * jb.a_time_ratio + 1)

/-- Theorem stating that B's profit is 4000 given the specified conditions -/
theorem b_profit_is_4000 (jb : JointBusiness) 
  (h1 : jb.total_profit = 28000)
  (h2 : jb.a_investment_ratio = 3)
  (h3 : jb.a_time_ratio = 2) : 
  calculate_b_profit jb = 4000 := by
  sorry

#eval calculate_b_profit { total_profit := 28000, a_investment_ratio := 3, a_time_ratio := 2 }

end NUMINAMATH_CALUDE_b_profit_is_4000_l3331_333130


namespace NUMINAMATH_CALUDE_star_four_three_l3331_333181

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2 + 2*a*b

-- State the theorem
theorem star_four_three : star 4 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l3331_333181


namespace NUMINAMATH_CALUDE_stall_owner_earnings_l3331_333143

/-- Represents the stall owner's game with ping-pong balls -/
structure BallGame where
  yellow_balls : ℕ := 3
  white_balls : ℕ := 2
  balls_drawn : ℕ := 3
  same_color_reward : ℕ := 5
  diff_color_cost : ℕ := 1
  daily_players : ℕ := 100
  days_in_month : ℕ := 30

/-- Calculates the expected monthly earnings of the stall owner -/
def expected_monthly_earnings (game : BallGame) : ℚ :=
  let total_balls := game.yellow_balls + game.white_balls
  let prob_same_color := (game.yellow_balls.choose game.balls_drawn) / (total_balls.choose game.balls_drawn)
  let daily_earnings := game.daily_players * (game.diff_color_cost * (1 - prob_same_color) - game.same_color_reward * prob_same_color)
  daily_earnings * game.days_in_month

/-- Theorem stating the expected monthly earnings of the stall owner -/
theorem stall_owner_earnings (game : BallGame) : 
  expected_monthly_earnings game = 1200 := by
  sorry

end NUMINAMATH_CALUDE_stall_owner_earnings_l3331_333143


namespace NUMINAMATH_CALUDE_b_investment_is_1000_l3331_333100

/-- Represents the business partnership between a and b -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  total_profit : ℕ
  management_fee_percent : ℚ
  a_total_received : ℕ

/-- Calculates b's investment given the partnership details -/
def calculate_b_investment (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that b's investment is 1000 given the problem conditions -/
theorem b_investment_is_1000 (p : Partnership) 
  (h1 : p.a_investment = 2000)
  (h2 : p.total_profit = 9600)
  (h3 : p.management_fee_percent = 1/10)
  (h4 : p.a_total_received = 4416) :
  calculate_b_investment p = 1000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_1000_l3331_333100


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3331_333101

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 15^4 →
  m % 10 = 9 →
  n % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3331_333101


namespace NUMINAMATH_CALUDE_dan_picked_nine_apples_l3331_333178

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The total number of apples picked by Benny and Dan -/
def total_apples : ℕ := 11

/-- The number of apples Dan picked -/
def dan_apples : ℕ := total_apples - benny_apples

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_apples_l3331_333178


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l3331_333189

/-- The longest side of a triangle with vertices at (1,1), (4,5), and (7,1) has a length of 6 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ), 
  a = (1, 1) ∧ b = (4, 5) ∧ c = (7, 1) ∧
  ∀ (d : ℝ), d = max (dist a b) (max (dist b c) (dist c a)) → d = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l3331_333189


namespace NUMINAMATH_CALUDE_paiges_dresser_capacity_l3331_333148

/-- Calculates the total number of clothing pieces a dresser can hold. -/
def dresser_capacity (drawers : ℕ) (pieces_per_drawer : ℕ) : ℕ :=
  drawers * pieces_per_drawer

/-- Proves that Paige's dresser can hold 8 pieces of clothing. -/
theorem paiges_dresser_capacity :
  dresser_capacity 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_paiges_dresser_capacity_l3331_333148


namespace NUMINAMATH_CALUDE_largest_multiple_less_than_negative_fifty_l3331_333129

theorem largest_multiple_less_than_negative_fifty :
  ∀ n : ℤ, n * 12 < -50 → n * 12 ≤ -48 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_less_than_negative_fifty_l3331_333129


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3331_333108

theorem no_function_satisfies_inequality :
  ∀ f : ℕ → ℕ, ∃ m n : ℕ, (m + f n)^2 < 3 * (f m)^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3331_333108


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3331_333165

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Main theorem -/
theorem fibonacci_arithmetic_sequence (a b d : ℕ) : 
  (∀ n ≥ 3, fib n = fib (n - 1) + fib (n - 2)) →  -- Fibonacci recurrence relation
  (fib a < fib b ∧ fib b < fib d) →  -- Increasing sequence
  (fib d - fib b = fib b - fib a) →  -- Arithmetic sequence
  d = b + 2 →  -- Given condition
  a + b + d = 1000 →  -- Given condition
  a = 332 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3331_333165


namespace NUMINAMATH_CALUDE_cone_central_angle_l3331_333171

/-- Given a cone where the lateral area is twice the area of its base,
    prove that the central angle of the sector of the unfolded side is 180 degrees. -/
theorem cone_central_angle (r R : ℝ) (h : r > 0) (H : R > 0) : 
  π * r * R = 2 * π * r^2 → (180 : ℝ) * (2 * π * r) / (π * R) = 180 :=
by sorry

end NUMINAMATH_CALUDE_cone_central_angle_l3331_333171


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l3331_333157

def f (x : ℝ) := 2*x + x^3 - 2

theorem unique_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l3331_333157


namespace NUMINAMATH_CALUDE_mark_piggy_bank_problem_l3331_333185

/-- Given a total amount of money and a total number of bills (one and two dollar bills only),
    calculate the number of one dollar bills. -/
def one_dollar_bills (total_money : ℕ) (total_bills : ℕ) : ℕ :=
  total_bills - (total_money - total_bills)

/-- Theorem stating that given 87 dollars in total and 58 bills,
    the number of one dollar bills is 29. -/
theorem mark_piggy_bank_problem :
  one_dollar_bills 87 58 = 29 := by
  sorry

end NUMINAMATH_CALUDE_mark_piggy_bank_problem_l3331_333185


namespace NUMINAMATH_CALUDE_matching_color_probability_l3331_333151

def total_jellybeans_ava : ℕ := 4
def total_jellybeans_ben : ℕ := 8

def green_jellybeans_ava : ℕ := 2
def red_jellybeans_ava : ℕ := 2
def green_jellybeans_ben : ℕ := 2
def red_jellybeans_ben : ℕ := 3

theorem matching_color_probability :
  let p_green := (green_jellybeans_ava / total_jellybeans_ava) * (green_jellybeans_ben / total_jellybeans_ben)
  let p_red := (red_jellybeans_ava / total_jellybeans_ava) * (red_jellybeans_ben / total_jellybeans_ben)
  p_green + p_red = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_matching_color_probability_l3331_333151


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3331_333174

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs (x + a)

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3331_333174


namespace NUMINAMATH_CALUDE_inequality_proof_l3331_333113

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4*x + y + 2*z) * (2*x + y + 8*z) ≥ (375/2) * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3331_333113


namespace NUMINAMATH_CALUDE_last_digit_of_base4_conversion_l3331_333132

def base5_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

def decimal_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

def base5_number : List Nat := [4, 3, 2, 1]

theorem last_digit_of_base4_conversion :
  (decimal_to_base4 (base5_to_decimal base5_number)).getLast? = some 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_base4_conversion_l3331_333132


namespace NUMINAMATH_CALUDE_square_sum_simplification_l3331_333195

theorem square_sum_simplification : 99^2 + 202 * 99 + 101^2 = 40000 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_simplification_l3331_333195


namespace NUMINAMATH_CALUDE_imaginary_number_on_real_axis_l3331_333109

theorem imaginary_number_on_real_axis (z : ℂ) :
  (∃ b : ℝ, z = b * I) →  -- z is a pure imaginary number
  (∃ r : ℝ, (z + 2) / (1 - I) = r) →  -- point lies on real axis
  z = -2 * I :=
by sorry

end NUMINAMATH_CALUDE_imaginary_number_on_real_axis_l3331_333109


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3331_333167

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) := 
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 5 * a 7 = -3 * Real.sqrt 3 →
  a 2 * a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3331_333167


namespace NUMINAMATH_CALUDE_fraction_simplification_l3331_333161

theorem fraction_simplification :
  (3^1006 + 3^1004) / (3^1006 - 3^1004) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3331_333161


namespace NUMINAMATH_CALUDE_paint_for_smaller_statues_l3331_333155

/-- The amount of paint (in pints) required for a statue of given height (in feet) -/
def paint_required (height : ℝ) : ℝ := sorry

/-- The number of statues to be painted -/
def num_statues : ℕ := 320

/-- The height (in feet) of the original statue -/
def original_height : ℝ := 8

/-- The height (in feet) of the new statues -/
def new_height : ℝ := 2

/-- The amount of paint (in pints) required for the original statue -/
def original_paint : ℝ := 2

theorem paint_for_smaller_statues :
  paint_required new_height * num_statues = 10 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_smaller_statues_l3331_333155


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l3331_333140

-- Define the given constants
def rate_up : ℝ := 6
def time_up : ℝ := 2
def distance_down : ℝ := 18

-- Define the theorem
theorem hiking_rate_ratio :
  let distance_up := rate_up * time_up
  let time_down := time_up
  let rate_down := distance_down / time_down
  rate_down / rate_up = 1.5 := by
sorry

end NUMINAMATH_CALUDE_hiking_rate_ratio_l3331_333140


namespace NUMINAMATH_CALUDE_sector_area_l3331_333160

theorem sector_area (α : Real) (r : Real) (h1 : α = 2 * Real.pi / 3) (h2 : r = Real.sqrt 3) : 
  (1/2) * α * r^2 = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3331_333160


namespace NUMINAMATH_CALUDE_increasing_function_range_function_below_one_range_function_range_when_a_geq_two_l3331_333182

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x - a)

-- Theorem 1
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x + x < f a y + y) ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem function_below_one_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < 1) ↔ 3/2 < a ∧ a < 2 := by sorry

-- Theorem 3 (partial, due to multiple conditions)
theorem function_range_when_a_geq_two (a : ℝ) (h : a ≥ 2) :
  ∃ l u : ℝ, ∀ x : ℝ, x ∈ Set.Icc 2 4 → l ≤ f a x ∧ f a x ≤ u := by sorry

end NUMINAMATH_CALUDE_increasing_function_range_function_below_one_range_function_range_when_a_geq_two_l3331_333182


namespace NUMINAMATH_CALUDE_tile_problem_l3331_333145

theorem tile_problem (total_tiles : ℕ) (total_edges : ℕ) (triangular_tiles : ℕ) (square_tiles : ℕ) : 
  total_tiles = 25 →
  total_edges = 84 →
  total_tiles = triangular_tiles + square_tiles →
  total_edges = 3 * triangular_tiles + 4 * square_tiles →
  square_tiles = 9 := by
sorry

end NUMINAMATH_CALUDE_tile_problem_l3331_333145


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3331_333142

theorem alpha_beta_sum (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 75*x + 1236) / (x^2 + 60*x - 3120)) →
  α + β = 139 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3331_333142


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3331_333133

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) - (Real.sqrt 98 / Real.sqrt 32) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3331_333133


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3331_333125

theorem tenth_term_of_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, a n = n * (n + 1) / 2) : 
  a 10 = 55 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3331_333125


namespace NUMINAMATH_CALUDE_product_xyz_l3331_333154

theorem product_xyz (x y z : ℕ+) 
  (h1 : x + 2*y = z) 
  (h2 : x^2 - 4*y^2 + z^2 = 310) : 
  x*y*z = 4030 ∨ x*y*z = 23870 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l3331_333154


namespace NUMINAMATH_CALUDE_direct_proportion_function_m_l3331_333170

theorem direct_proportion_function_m (m : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 3) * x^(m^2 - 8) = k * x) ↔ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_function_m_l3331_333170


namespace NUMINAMATH_CALUDE_quartic_polynomial_value_l3331_333136

def is_monic_quartic (q : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, q x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem quartic_polynomial_value (q : ℝ → ℝ) :
  is_monic_quartic q →
  q 1 = 3 →
  q 2 = 6 →
  q 3 = 11 →
  q 4 = 18 →
  q 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_value_l3331_333136


namespace NUMINAMATH_CALUDE_sara_movie_rental_l3331_333127

def movie_problem (theater_ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) : Prop :=
  let theater_total : ℚ := theater_ticket_price * num_tickets
  let rental_price : ℚ := total_spent - theater_total - bought_movie_price
  rental_price = 159/100

theorem sara_movie_rental :
  movie_problem (1062/100) 2 (1395/100) (3678/100) :=
by
  sorry

end NUMINAMATH_CALUDE_sara_movie_rental_l3331_333127


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3331_333102

theorem sum_of_solutions_is_zero (y : ℝ) (x₁ x₂ : ℝ) : 
  y = 6 → x₁^2 + y^2 = 145 → x₂^2 + y^2 = 145 → x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3331_333102


namespace NUMINAMATH_CALUDE_partnership_investment_period_ratio_l3331_333110

/-- Proves that the ratio of investment periods is 2:1 given the partnership conditions --/
theorem partnership_investment_period_ratio :
  ∀ (investment_A investment_B period_A period_B profit_A profit_B : ℚ),
    investment_A = 3 * investment_B →
    ∃ k : ℚ, period_A = k * period_B →
    profit_B = 4500 →
    profit_A + profit_B = 31500 →
    profit_A / profit_B = (investment_A * period_A) / (investment_B * period_B) →
    period_A / period_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_period_ratio_l3331_333110


namespace NUMINAMATH_CALUDE_stating_angle_bisector_division_l3331_333187

/-- Represents a parallelogram with sides of length 8 and 3 -/
structure Parallelogram where
  long_side : ℝ
  short_side : ℝ
  long_side_eq : long_side = 8
  short_side_eq : short_side = 3

/-- Represents the three parts of the divided side -/
structure DividedSide where
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ

/-- 
Theorem stating that the angle bisectors of the two angles adjacent to the longer side 
divide the opposite side into three parts with lengths 3, 2, and 3.
-/
theorem angle_bisector_division (p : Parallelogram) : 
  ∃ (d : DividedSide), d.part1 = 3 ∧ d.part2 = 2 ∧ d.part3 = 3 ∧ 
  d.part1 + d.part2 + d.part3 = p.long_side :=
sorry

end NUMINAMATH_CALUDE_stating_angle_bisector_division_l3331_333187


namespace NUMINAMATH_CALUDE_min_magnitude_a_minus_c_l3331_333124

noncomputable section

-- Define the plane vectors
variable (a b c : ℝ × ℝ)

-- Define the conditions
def magnitude_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
def magnitude_b_minus_c : ℝ := Real.sqrt (((b.1 - c.1) ^ 2) + ((b.2 - c.2) ^ 2))
def angle_between_a_and_b : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (magnitude_a a * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))))

-- State the theorem
theorem min_magnitude_a_minus_c (h1 : magnitude_a a = 2)
                                (h2 : magnitude_b_minus_c b c = 1)
                                (h3 : angle_between_a_and_b a b = π / 3) :
  ∃ (min_value : ℝ), ∀ (a' b' c' : ℝ × ℝ),
    magnitude_a a' = 2 →
    magnitude_b_minus_c b' c' = 1 →
    angle_between_a_and_b a' b' = π / 3 →
    Real.sqrt (((a'.1 - c'.1) ^ 2) + ((a'.2 - c'.2) ^ 2)) ≥ min_value ∧
    min_value = Real.sqrt 3 - 1 :=
  sorry

end

end NUMINAMATH_CALUDE_min_magnitude_a_minus_c_l3331_333124


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_proof_l3331_333121

/-- The least positive base-10 number that requires seven digits in binary representation -/
def least_seven_digit_binary : ℕ := 64

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binary_digits (n : ℕ) : ℕ := sorry

theorem least_seven_digit_binary_proof :
  (binary_digits least_seven_digit_binary = 7) ∧
  (∀ m : ℕ, m > 0 ∧ m < least_seven_digit_binary → binary_digits m < 7) :=
sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_proof_l3331_333121


namespace NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l3331_333169

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def fair_12_sided_die : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair 12-sided die -/
def prob_each_outcome : ℚ := 1 / 12

/-- The expected value of rolling a fair 12-sided die -/
def expected_value : ℚ := (fair_12_sided_die.sum (λ x => (x + 1) * prob_each_outcome))

/-- Theorem: The expected value of rolling a fair 12-sided die is 6.5 -/
theorem expected_value_fair_12_sided_die : expected_value = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l3331_333169


namespace NUMINAMATH_CALUDE_amount_subtracted_l3331_333112

theorem amount_subtracted (number : ℝ) (result : ℝ) (amount : ℝ) : 
  number = 150 →
  result = 50 →
  0.60 * number - amount = result →
  amount = 40 := by
sorry

end NUMINAMATH_CALUDE_amount_subtracted_l3331_333112


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l3331_333144

def total_participants : ℕ := 18
def tribe1_size : ℕ := 10
def tribe2_size : ℕ := 8
def quitters : ℕ := 3

theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_participants quitters
  let ways_from_tribe1 := Nat.choose tribe1_size quitters
  let ways_from_tribe2 := Nat.choose tribe2_size quitters
  (ways_from_tribe1 + ways_from_tribe2 : ℚ) / total_ways = 11 / 51 := by
sorry

end NUMINAMATH_CALUDE_survivor_quitters_probability_l3331_333144


namespace NUMINAMATH_CALUDE_multiply_by_8050_equals_80_5_l3331_333105

theorem multiply_by_8050_equals_80_5 : ∃ x : ℝ, 8050 * x = 80.5 ∧ x = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_8050_equals_80_5_l3331_333105


namespace NUMINAMATH_CALUDE_smallest_n_for_2012_terms_l3331_333163

theorem smallest_n_for_2012_terms (n : ℕ) : (∀ m : ℕ, (m + 1)^2 ≥ 2012 → n ≤ m) ↔ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_2012_terms_l3331_333163


namespace NUMINAMATH_CALUDE_sector_max_area_l3331_333111

/-- The maximum area of a sector with circumference 40 is 100 -/
theorem sector_max_area (C : ℝ) (h : C = 40) : 
  ∃ (A : ℝ), A = 100 ∧ ∀ (r θ : ℝ), r > 0 → θ > 0 → r * θ + 2 * r = C → 
    (1/2 : ℝ) * r^2 * θ ≤ A := by sorry

end NUMINAMATH_CALUDE_sector_max_area_l3331_333111


namespace NUMINAMATH_CALUDE_range_of_k_l3331_333193

theorem range_of_k (k : ℝ) : 
  (k ≠ 0) → 
  (k^2 * 1^2 - 6*k*1 + 8 ≥ 0) → 
  ((k ≥ 4) ∨ (k ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l3331_333193


namespace NUMINAMATH_CALUDE_toby_friend_wins_l3331_333180

/-- Juggling contest between Toby and his friend -/
theorem toby_friend_wins (toby_rotations : ℕ → ℕ) (friend_apples : ℕ) (friend_rotations : ℕ → ℕ) : 
  friend_apples = 4 ∧ 
  (∀ n, friend_rotations n = 101) ∧ 
  (∀ n, toby_rotations n = 80) → 
  friend_apples * friend_rotations 0 = 404 ∧ 
  ∀ k, k * toby_rotations 0 ≤ friend_apples * friend_rotations 0 :=
by sorry

end NUMINAMATH_CALUDE_toby_friend_wins_l3331_333180


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3331_333104

def days_in_week : ℕ := 7

def probability_sun : ℝ := 0.3
def probability_light_rain : ℝ := 0.5
def probability_heavy_rain : ℝ := 0.2

def light_rain_amount : ℝ := 3
def heavy_rain_amount : ℝ := 8

def daily_expected_rainfall : ℝ :=
  probability_sun * 0 + 
  probability_light_rain * light_rain_amount + 
  probability_heavy_rain * heavy_rain_amount

theorem expected_weekly_rainfall : 
  days_in_week * daily_expected_rainfall = 21.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3331_333104


namespace NUMINAMATH_CALUDE_systematic_sampling_third_group_l3331_333123

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) : ℕ → ℕ :=
  fun n => firstSelected + (n - 1) * (totalItems / sampleSize)

theorem systematic_sampling_third_group 
  (totalItems : ℕ) 
  (sampleSize : ℕ) 
  (groupSize : ℕ) 
  (numGroups : ℕ) 
  (firstSelected : ℕ) :
  totalItems = 300 →
  sampleSize = 20 →
  groupSize = 20 →
  numGroups = 15 →
  firstSelected = 6 →
  totalItems = groupSize * numGroups →
  systematicSample totalItems sampleSize firstSelected 3 = 36 := by
  sorry

#check systematic_sampling_third_group

end NUMINAMATH_CALUDE_systematic_sampling_third_group_l3331_333123


namespace NUMINAMATH_CALUDE_simplify_expression_l3331_333131

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    (((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 5)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 5)) =
     (1 - (1/2) * Real.sqrt 3) * (2 ^ (-Real.sqrt 5))) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c.val)) ∧
    ((1 - (1/2) * Real.sqrt 3) * (2 ^ (-Real.sqrt 5)) = a - b * Real.sqrt c) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3331_333131


namespace NUMINAMATH_CALUDE_binomial_variance_example_l3331_333196

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(8, 3/4) is 3/2 -/
theorem binomial_variance_example :
  let X : BinomialRV := ⟨8, 3/4, by norm_num, by norm_num⟩
  variance X = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l3331_333196


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3331_333186

theorem trigonometric_identities :
  (Real.cos (75 * π / 180))^2 = (2 - Real.sqrt 3) / 4 ∧
  Real.tan (1 * π / 180) + Real.tan (44 * π / 180) + Real.tan (1 * π / 180) * Real.tan (44 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3331_333186


namespace NUMINAMATH_CALUDE_r_fourth_plus_inv_r_fourth_l3331_333191

theorem r_fourth_plus_inv_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inv_r_fourth_l3331_333191


namespace NUMINAMATH_CALUDE_tan_thirty_degrees_l3331_333122

theorem tan_thirty_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirty_degrees_l3331_333122


namespace NUMINAMATH_CALUDE_unknown_number_value_l3331_333179

theorem unknown_number_value (x n : ℝ) : 
  x = 12 → 5 + n / x = 6 - 5 / x → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l3331_333179


namespace NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l3331_333152

theorem unique_cube_difference_nineteen :
  ∃! (x y : ℕ), x^3 - y^3 = 19 ∧ x = 3 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l3331_333152


namespace NUMINAMATH_CALUDE_nested_cube_root_simplification_l3331_333147

theorem nested_cube_root_simplification (N : ℝ) (h : N > 1) :
  (N^3 * (N^5 * N^3)^(1/3))^(1/3) = N^(5/3) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_simplification_l3331_333147


namespace NUMINAMATH_CALUDE_square_area_is_25_l3331_333114

/-- A square in the coordinate plane with given vertex coordinates -/
structure Square where
  x₁ : ℝ
  x₂ : ℝ

/-- The area of the square -/
def square_area (s : Square) : ℝ :=
  (5 : ℝ) ^ 2

/-- Theorem stating that the area of the square is 25 -/
theorem square_area_is_25 (s : Square) : square_area s = 25 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_25_l3331_333114


namespace NUMINAMATH_CALUDE_expression_simplification_l3331_333177

theorem expression_simplification :
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3331_333177


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3331_333120

/-- The line passing through (-1, 1) and perpendicular to x + y = 0 has equation x - y + 2 = 0 -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + 1 = 0 ∧ y - 1 = 0) →  -- Point (-1, 1)
  (∀ x y, x + y = 0 → True) →  -- Given line x + y = 0
  x - y + 2 = 0 := by  -- Resulting perpendicular line
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3331_333120


namespace NUMINAMATH_CALUDE_min_cos_sum_with_tan_product_l3331_333150

theorem min_cos_sum_with_tan_product (x y m : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hm : m > 2) 
  (h : Real.tan x * Real.tan y = m) : 
  Real.cos x + Real.cos y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_cos_sum_with_tan_product_l3331_333150


namespace NUMINAMATH_CALUDE_circle_C_distance_range_l3331_333135

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 2)^2 = 25

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (7, 4)

-- Define the function for |PA|^2 + |PB|^2
def sum_of_squared_distances (P : ℝ × ℝ) : ℝ :=
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 +
  (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2

-- Theorem statement
theorem circle_C_distance_range :
  ∀ P : ℝ × ℝ, circle_C P.1 P.2 →
  103 ≤ sum_of_squared_distances P ∧ sum_of_squared_distances P ≤ 123 :=
sorry

end NUMINAMATH_CALUDE_circle_C_distance_range_l3331_333135
