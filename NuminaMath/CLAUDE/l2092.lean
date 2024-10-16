import Mathlib

namespace NUMINAMATH_CALUDE_dark_king_game_winner_l2092_209234

/-- The dark king game on an n × m chessboard -/
def DarkKingGame (n m : ℕ) :=
  { board : Set (ℕ × ℕ) // board ⊆ (Finset.range n).product (Finset.range m) }

/-- A player in the dark king game -/
inductive Player
| First
| Second

/-- A winning strategy for a player in the dark king game -/
def WinningStrategy (n m : ℕ) (p : Player) :=
  ∃ (strategy : DarkKingGame n m → ℕ × ℕ),
    ∀ (game : DarkKingGame n m),
      (strategy game ∉ game.val) →
      (strategy game).1 < n ∧ (strategy game).2 < m

/-- The main theorem about the dark king game -/
theorem dark_king_game_winner (n m : ℕ) :
  (n % 2 = 0 ∨ m % 2 = 0) → WinningStrategy n m Player.First ∧
  (n % 2 = 1 ∧ m % 2 = 1) → WinningStrategy n m Player.Second :=
sorry

end NUMINAMATH_CALUDE_dark_king_game_winner_l2092_209234


namespace NUMINAMATH_CALUDE_min_value_zero_at_k_eq_two_l2092_209206

/-- The quadratic function f(x, y) depending on parameter k -/
def f (k : ℝ) (x y : ℝ) : ℝ :=
  4 * x^2 - 6 * k * x * y + (3 * k^2 + 2) * y^2 - 4 * x - 4 * y + 6

/-- Theorem stating that k = 2 is the unique value for which the minimum of f is 0 -/
theorem min_value_zero_at_k_eq_two :
  ∃! k : ℝ, (∀ x y : ℝ, f k x y ≥ 0) ∧ (∃ x y : ℝ, f k x y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_zero_at_k_eq_two_l2092_209206


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l2092_209220

theorem integer_roots_quadratic (a : ℤ) : 
  (∃ x y : ℤ, x^2 - a*x + 9*a = 0 ∧ y^2 - a*y + 9*a = 0 ∧ x ≠ y) ↔ 
  a ∈ ({100, -64, 48, -12, 36, 0} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l2092_209220


namespace NUMINAMATH_CALUDE_f_properties_l2092_209244

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x

-- State the theorem
theorem f_properties :
  -- Part 1: Monotonicity of f
  (∀ x < -3, ∀ y < -3, x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioo (-3) 1, ∀ y ∈ Set.Ioo (-3) 1, x < y → f x > f y) ∧
  (∀ x > 1, ∀ y > 1, x < y → f x < f y) ∧
  -- Part 2: Minimum value condition
  (∀ c : ℝ, (∀ x ∈ Set.Icc (-4) c, f x ≥ -5) ∧ (∃ x ∈ Set.Icc (-4) c, f x = -5) ↔ c ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2092_209244


namespace NUMINAMATH_CALUDE_deck_cost_l2092_209285

/-- The cost of the deck of playing cards given the allowances and sticker purchases -/
theorem deck_cost (lola_allowance dora_allowance : ℕ)
                  (sticker_boxes : ℕ)
                  (dora_sticker_packs : ℕ)
                  (h1 : lola_allowance = 9)
                  (h2 : dora_allowance = 9)
                  (h3 : sticker_boxes = 2)
                  (h4 : dora_sticker_packs = 2) :
  let total_allowance := lola_allowance + dora_allowance
  let total_sticker_packs := 2 * dora_sticker_packs
  let sticker_cost := sticker_boxes * 2
  total_allowance - sticker_cost = 10 := by
sorry

end NUMINAMATH_CALUDE_deck_cost_l2092_209285


namespace NUMINAMATH_CALUDE_cosine_sine_sum_zero_l2092_209295

theorem cosine_sine_sum_zero (x : ℝ) (h : Real.cos (π / 6 - x) = -Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_zero_l2092_209295


namespace NUMINAMATH_CALUDE_jimin_class_size_l2092_209290

/-- The number of students in Jimin's class -/
def total_students : ℕ := 45

/-- The number of students who like Korean -/
def korean_students : ℕ := 38

/-- The number of students who like math -/
def math_students : ℕ := 39

/-- The number of students who like both Korean and math -/
def both_subjects : ℕ := 32

/-- Theorem stating the total number of students in Jimin's class -/
theorem jimin_class_size :
  total_students = korean_students + math_students - both_subjects :=
by sorry

end NUMINAMATH_CALUDE_jimin_class_size_l2092_209290


namespace NUMINAMATH_CALUDE_rachel_bought_three_tables_l2092_209200

/-- Represents the number of minutes spent on each piece of furniture -/
def time_per_furniture : ℕ := 4

/-- Represents the total number of chairs bought -/
def num_chairs : ℕ := 7

/-- Represents the total time spent assembling all furniture -/
def total_time : ℕ := 40

/-- Calculates the number of tables bought -/
def num_tables : ℕ :=
  (total_time - time_per_furniture * num_chairs) / time_per_furniture

theorem rachel_bought_three_tables :
  num_tables = 3 :=
sorry

end NUMINAMATH_CALUDE_rachel_bought_three_tables_l2092_209200


namespace NUMINAMATH_CALUDE_eddie_study_games_l2092_209217

/-- Calculates the maximum number of games that can be played in a study block -/
def max_games (study_block_minutes : ℕ) (homework_minutes : ℕ) (game_duration : ℕ) : ℕ :=
  (study_block_minutes - homework_minutes) / game_duration

/-- Theorem stating that given the specific conditions, the maximum number of games is 7 -/
theorem eddie_study_games :
  max_games 60 25 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eddie_study_games_l2092_209217


namespace NUMINAMATH_CALUDE_heroes_on_back_l2092_209250

/-- The number of heroes Will drew on the front of the paper -/
def heroes_on_front : ℕ := 2

/-- The total number of heroes Will drew -/
def total_heroes : ℕ := 9

/-- Theorem: The number of heroes Will drew on the back of the paper is 7 -/
theorem heroes_on_back : total_heroes - heroes_on_front = 7 := by
  sorry

end NUMINAMATH_CALUDE_heroes_on_back_l2092_209250


namespace NUMINAMATH_CALUDE_cos_value_third_quadrant_l2092_209292

theorem cos_value_third_quadrant (θ : Real) :
  tanθ = Real.sqrt 2 / 4 →
  θ > π ∧ θ < 3 * π / 2 →
  cosθ = -2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_value_third_quadrant_l2092_209292


namespace NUMINAMATH_CALUDE_max_min_difference_z_l2092_209237

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (sum_squares_condition : x^2 + y^2 + z^2 = 29) :
  ∃ (z_max z_min : ℝ),
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 29) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 29) → z' ≥ z_min) ∧
    z_max - z_min = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l2092_209237


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2092_209270

theorem complex_equation_solution : ∃ (a : ℝ), 
  (1 - Complex.I : ℂ) = (2 + a * Complex.I) / (1 + Complex.I) ∧ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2092_209270


namespace NUMINAMATH_CALUDE_sum_of_greatest_b_values_l2092_209275

theorem sum_of_greatest_b_values (b : ℝ) : 
  4 * b^4 - 41 * b^2 + 100 = 0 → 
  ∃ (b1 b2 : ℝ), b1 ≥ b2 ∧ b2 ≥ 0 ∧ 
    (4 * b1^4 - 41 * b1^2 + 100 = 0) ∧ 
    (4 * b2^4 - 41 * b2^2 + 100 = 0) ∧ 
    b1 + b2 = 4.5 ∧
    ∀ (x : ℝ), (4 * x^4 - 41 * x^2 + 100 = 0) → x ≤ b1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_greatest_b_values_l2092_209275


namespace NUMINAMATH_CALUDE_nth_group_sum_correct_l2092_209222

/-- The sum of the n-th group in the sequence of positive integers grouped as 1, 2+3, 4+5+6, 7+8+9+10, ... -/
def nth_group_sum (n : ℕ) : ℕ :=
  (n * (n^2 + 1)) / 2

/-- The first element of the n-th group -/
def first_element (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

/-- The last element of the n-th group -/
def last_element (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem nth_group_sum_correct (n : ℕ) (h : n > 0) :
  nth_group_sum n = (n * (first_element n + last_element n)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_nth_group_sum_correct_l2092_209222


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2092_209240

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2092_209240


namespace NUMINAMATH_CALUDE_two_digit_integer_count_l2092_209286

/-- A function that counts the number of three-digit integers less than 1000 with exactly two different digits. -/
def count_two_digit_integers : ℕ :=
  let case1 := 9  -- Numbers with one digit as zero
  let case2 := 9 * 9 * 3  -- Numbers with two non-zero digits
  case1 + case2

/-- Theorem stating that the count of three-digit integers less than 1000 with exactly two different digits is 252. -/
theorem two_digit_integer_count : count_two_digit_integers = 252 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integer_count_l2092_209286


namespace NUMINAMATH_CALUDE_lauryn_company_employees_l2092_209223

/-- The number of men working for Lauryn's company -/
def num_men : ℕ := 80

/-- The difference between the number of women and men -/
def women_men_diff : ℕ := 20

/-- The total number of people working for Lauryn's company -/
def total_employees : ℕ := num_men + (num_men + women_men_diff)

theorem lauryn_company_employees :
  total_employees = 180 :=
by sorry

end NUMINAMATH_CALUDE_lauryn_company_employees_l2092_209223


namespace NUMINAMATH_CALUDE_twenty_fives_sum_1000_l2092_209214

/-- A list of integers representing a grouping of fives -/
def Grouping : Type := List Nat

/-- The number of fives in a grouping -/
def count_fives : Grouping → Nat
  | [] => 0
  | (x::xs) => (x.digits 10).length + count_fives xs

/-- The sum of a grouping -/
def sum_grouping : Grouping → Nat
  | [] => 0
  | (x::xs) => x + sum_grouping xs

/-- A valid grouping of 20 fives that sums to 1000 -/
theorem twenty_fives_sum_1000 : ∃ (g : Grouping), 
  (count_fives g = 20) ∧ (sum_grouping g = 1000) := by
  sorry

end NUMINAMATH_CALUDE_twenty_fives_sum_1000_l2092_209214


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l2092_209219

def sum_of_factors (n : ℕ) : ℕ := sorry

theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l2092_209219


namespace NUMINAMATH_CALUDE_coconut_moving_theorem_l2092_209202

/-- The number of coconuts Barbie can carry in one trip -/
def barbie_capacity : ℕ := 4

/-- The number of coconuts Bruno can carry in one trip -/
def bruno_capacity : ℕ := 8

/-- The number of trips Barbie and Bruno make together -/
def num_trips : ℕ := 12

/-- The total number of coconuts Barbie and Bruno can move -/
def total_coconuts : ℕ := (barbie_capacity + bruno_capacity) * num_trips

theorem coconut_moving_theorem : total_coconuts = 144 := by
  sorry

end NUMINAMATH_CALUDE_coconut_moving_theorem_l2092_209202


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l2092_209268

theorem jason_pokemon_cards (initial_cards : ℕ) (bought_cards : ℕ) :
  initial_cards = 1342 →
  bought_cards = 536 →
  initial_cards - bought_cards = 806 :=
by sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l2092_209268


namespace NUMINAMATH_CALUDE_jenny_max_sales_l2092_209269

/-- Represents a neighborhood where Jenny can sell cookies. -/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ

/-- Calculates the total sales for a given neighborhood. -/
def totalSales (n : Neighborhood) (pricePerBox : ℕ) : ℕ :=
  n.homes * n.boxesPerHome * pricePerBox

/-- Theorem stating that the maximum amount Jenny can make is $50. -/
theorem jenny_max_sales : 
  let neighborhoodA : Neighborhood := { homes := 10, boxesPerHome := 2 }
  let neighborhoodB : Neighborhood := { homes := 5, boxesPerHome := 5 }
  let pricePerBox : ℕ := 2
  max (totalSales neighborhoodA pricePerBox) (totalSales neighborhoodB pricePerBox) = 50 := by
  sorry

end NUMINAMATH_CALUDE_jenny_max_sales_l2092_209269


namespace NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l2092_209261

theorem sqrt_eight_and_one_ninth (x : ℝ) : x = Real.sqrt (8 + 1/9) → x = Real.sqrt 73 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l2092_209261


namespace NUMINAMATH_CALUDE_number_equals_scientific_notation_l2092_209238

-- Define the number we want to represent in scientific notation
def number : ℕ := 11700000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.17 * (10 ^ 7)

-- Theorem stating that the number is equal to its scientific notation representation
theorem number_equals_scientific_notation : (number : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_number_equals_scientific_notation_l2092_209238


namespace NUMINAMATH_CALUDE_brent_candy_count_l2092_209225

/-- Calculates the total number of candy pieces Brent has left after trick-or-treating and giving some to his sister. -/
def total_candy_left (kitkat : ℕ) (nerds : ℕ) (initial_lollipops : ℕ) (baby_ruth : ℕ) (given_lollipops : ℕ) : ℕ :=
  let hershey := 3 * kitkat
  let reese := baby_ruth / 2
  let remaining_lollipops := initial_lollipops - given_lollipops
  kitkat + hershey + nerds + baby_ruth + reese + remaining_lollipops

theorem brent_candy_count : 
  total_candy_left 5 8 11 10 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_brent_candy_count_l2092_209225


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l2092_209224

theorem smallest_integer_in_consecutive_set (n : ℤ) : 
  (n + 4 < 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4)) / 5)) → n ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l2092_209224


namespace NUMINAMATH_CALUDE_cos_equality_l2092_209243

theorem cos_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) : 
  n = 43 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_l2092_209243


namespace NUMINAMATH_CALUDE_smallest_solution_congruences_l2092_209249

theorem smallest_solution_congruences :
  ∃ x : ℕ, x > 0 ∧
    x % 2 = 1 ∧
    x % 3 = 2 ∧
    x % 4 = 3 ∧
    x % 5 = 4 ∧
    (∀ y : ℕ, y > 0 →
      y % 2 = 1 →
      y % 3 = 2 →
      y % 4 = 3 →
      y % 5 = 4 →
      y ≥ x) ∧
  x = 59 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruences_l2092_209249


namespace NUMINAMATH_CALUDE_basil_planter_problem_l2092_209263

theorem basil_planter_problem (total_seeds : Nat) (large_planters : Nat) (seeds_per_large : Nat) (seeds_per_small : Nat) :
  total_seeds = 200 →
  large_planters = 4 →
  seeds_per_large = 20 →
  seeds_per_small = 4 →
  (total_seeds - large_planters * seeds_per_large) / seeds_per_small = 30 := by
  sorry

end NUMINAMATH_CALUDE_basil_planter_problem_l2092_209263


namespace NUMINAMATH_CALUDE_unique_solution_l2092_209296

theorem unique_solution (m n : ℕ+) 
  (eq : 2 * m.val + 3 = 5 * n.val - 2)
  (ineq : 5 * n.val - 2 < 15) :
  m.val = 5 ∧ n.val = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2092_209296


namespace NUMINAMATH_CALUDE_parabola_properties_l2092_209283

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  x₁ : ℝ
  x₂ : ℝ
  y : ℝ → ℝ
  h₁ : a ≠ 0
  h₂ : x₁ ≠ x₂
  h₃ : ∀ x, y x = x^2 + (1 - 2*a)*x + a^2
  h₄ : y x₁ = 0
  h₅ : y x₂ = 0

/-- Main theorem about the parabola -/
theorem parabola_properties (p : Parabola) :
  (0 < p.a ∧ p.a < 1/4 ∧ p.x₁ < 0 ∧ p.x₂ < 0) ∧
  (p.y 0 - 2 = -p.x₁ - p.x₂ → p.a = -3) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2092_209283


namespace NUMINAMATH_CALUDE_biased_coin_probability_l2092_209289

theorem biased_coin_probability : ∃ (h : ℝ),
  (0 < h ∧ h < 1) ∧
  (Nat.choose 6 2 * h^2 * (1-h)^4 = Nat.choose 6 3 * h^3 * (1-h)^3) ∧
  (Nat.choose 6 4 * h^4 * (1-h)^2 = 19440 / 117649) :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l2092_209289


namespace NUMINAMATH_CALUDE_boat_upstream_downstream_ratio_l2092_209242

/-- Given a boat with speed in still water and a stream with its own speed,
    prove that the ratio of time taken to row upstream to downstream is 2:1 -/
theorem boat_upstream_downstream_ratio
  (boat_speed : ℝ) (stream_speed : ℝ)
  (h1 : boat_speed = 54)
  (h2 : stream_speed = 18) :
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check boat_upstream_downstream_ratio

end NUMINAMATH_CALUDE_boat_upstream_downstream_ratio_l2092_209242


namespace NUMINAMATH_CALUDE_x_value_from_ratios_l2092_209246

theorem x_value_from_ratios (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c) :
  x = 2 * a * b * c / (a * c + b * c - a * b) := by
sorry

end NUMINAMATH_CALUDE_x_value_from_ratios_l2092_209246


namespace NUMINAMATH_CALUDE_expression_evaluation_l2092_209293

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2092_209293


namespace NUMINAMATH_CALUDE_chip_price_is_two_l2092_209251

/-- The price of a packet of chips -/
def chip_price : ℝ := sorry

/-- The price of a packet of corn chips -/
def corn_chip_price : ℝ := 1.5

/-- The number of packets of chips John buys -/
def num_chips : ℕ := 15

/-- The number of packets of corn chips John buys -/
def num_corn_chips : ℕ := 10

/-- John's total budget -/
def total_budget : ℝ := 45

theorem chip_price_is_two :
  chip_price * num_chips + corn_chip_price * num_corn_chips = total_budget →
  chip_price = 2 := by sorry

end NUMINAMATH_CALUDE_chip_price_is_two_l2092_209251


namespace NUMINAMATH_CALUDE_sum_of_d_and_f_is_zero_l2092_209288

/-- Given three complex numbers a + bi, c + di, and 3e + fi, prove that d + f = 0 
    under the following conditions:
    1) b = 2
    2) c = -a - 2e
    3) The sum of the three complex numbers is 2i
-/
theorem sum_of_d_and_f_is_zero 
  (a b c d e f : ℂ) 
  (h1 : b = 2)
  (h2 : c = -a - 2*e)
  (h3 : a + b*Complex.I + c + d*Complex.I + 3*e + f*Complex.I = 2*Complex.I) :
  d + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_d_and_f_is_zero_l2092_209288


namespace NUMINAMATH_CALUDE_divisor_count_problem_l2092_209274

def τ (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_count_problem :
  (Finset.filter (fun n => τ n > 2 ∧ τ (τ n) = 2) (Finset.range 1001)).card = 184 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_problem_l2092_209274


namespace NUMINAMATH_CALUDE_cheolsu_number_problem_l2092_209215

theorem cheolsu_number_problem (x : ℚ) : 
  x + (-5/12) - (-5/2) = 1/3 → x = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_cheolsu_number_problem_l2092_209215


namespace NUMINAMATH_CALUDE_f_3_minus_f_4_l2092_209284

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_3_minus_f_4 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 5)
  (h_f_1 : f 1 = 1)
  (h_f_2 : f 2 = 2) :
  f 3 - f 4 = -1 := by sorry

end NUMINAMATH_CALUDE_f_3_minus_f_4_l2092_209284


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2092_209241

theorem simplify_trig_expression :
  1 / Real.sin (15 * π / 180) - 1 / Real.cos (15 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2092_209241


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2092_209299

theorem inequality_solution_set : 
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ x > 2} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2092_209299


namespace NUMINAMATH_CALUDE_root_absolute_value_greater_than_four_l2092_209221

theorem root_absolute_value_greater_than_four (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 16 = 0 → 
  r₂^2 + p*r₂ + 16 = 0 → 
  (abs r₁ > 4) ∨ (abs r₂ > 4) := by
sorry

end NUMINAMATH_CALUDE_root_absolute_value_greater_than_four_l2092_209221


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l2092_209218

theorem unique_solution_3x_4y_5z : 
  ∀ x y z : ℕ+, 
    (3 : ℕ)^(x : ℕ) + (4 : ℕ)^(y : ℕ) = (5 : ℕ)^(z : ℕ) → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

#check unique_solution_3x_4y_5z

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l2092_209218


namespace NUMINAMATH_CALUDE_divisibility_counterexample_l2092_209260

theorem divisibility_counterexample : 
  ∃ (a b c : ℤ), (a ∣ b * c) ∧ ¬(a ∣ b) ∧ ¬(a ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_counterexample_l2092_209260


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l2092_209247

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ r : Fin 180, p^2 % 180 = r.val ∧ (r.val = 1 ∨ r.val = 145) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l2092_209247


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2092_209209

def M : Set ℤ := {-1, 0, 2, 4}
def N : Set ℤ := {0, 2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2092_209209


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2092_209253

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {x | -1 < x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2092_209253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2092_209271

/-- An arithmetic sequence with given start, end, and common difference -/
def arithmetic_sequence (start end_ diff : ℕ) : List ℕ :=
  let n := (end_ - start) / diff + 1
  List.range n |>.map (fun i => start + i * diff)

/-- The problem statement -/
theorem arithmetic_sequence_length :
  (arithmetic_sequence 20 150 5).length = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2092_209271


namespace NUMINAMATH_CALUDE_points_on_line_l2092_209262

-- Define the points
def p1 : ℝ × ℝ := (4, 8)
def p2 : ℝ × ℝ := (2, 2)
def p3 : ℝ × ℝ := (3, 5)
def p4 : ℝ × ℝ := (0, -2)
def p5 : ℝ × ℝ := (1, 1)
def p6 : ℝ × ℝ := (5, 11)
def p7 : ℝ × ℝ := (6, 14)

-- Function to check if a point lies on the line
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (p1.2 - p2.2) / (p1.1 - p2.1)
  let b := p1.2 - m * p1.1
  p.2 = m * p.1 + b

-- Theorem stating which points lie on the line
theorem points_on_line :
  lies_on_line p3 ∧ lies_on_line p6 ∧ lies_on_line p7 ∧
  ¬lies_on_line p4 ∧ ¬lies_on_line p5 :=
sorry

end NUMINAMATH_CALUDE_points_on_line_l2092_209262


namespace NUMINAMATH_CALUDE_product_equality_l2092_209212

theorem product_equality : 3.6 * 0.4 * 1.5 = 2.16 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2092_209212


namespace NUMINAMATH_CALUDE_expression_simplification_l2092_209236

theorem expression_simplification (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - 
  (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 = 
  (2*(a + b + c - d))^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2092_209236


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_primes_l2092_209255

theorem pythagorean_triple_with_primes (x y z : ℤ) :
  x^2 + y^2 = z^2 →
  (Prime y ∧ y > 5 ∧ Prime z ∧ z > 5) ∨
  (Prime x ∧ x > 5 ∧ Prime z ∧ z > 5) ∨
  (Prime x ∧ x > 5 ∧ Prime y ∧ y > 5) →
  60 ∣ x ∨ 60 ∣ y ∨ 60 ∣ z :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_primes_l2092_209255


namespace NUMINAMATH_CALUDE_container_fill_fraction_l2092_209235

theorem container_fill_fraction (initial_percentage : ℝ) (added_water : ℝ) (capacity : ℝ) : 
  initial_percentage = 0.3 →
  added_water = 27 →
  capacity = 60 →
  (initial_percentage * capacity + added_water) / capacity = 0.75 := by
sorry

end NUMINAMATH_CALUDE_container_fill_fraction_l2092_209235


namespace NUMINAMATH_CALUDE_remainder_problem_l2092_209227

theorem remainder_problem (N : ℕ) (h : N % 899 = 63) : N % 29 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2092_209227


namespace NUMINAMATH_CALUDE_max_p_value_l2092_209248

/-- Given a function f(x) = e^x and real numbers m, n, p satisfying certain conditions,
    the maximum value of p is 2ln(2) - ln(3). -/
theorem max_p_value (f : ℝ → ℝ) (m n p : ℝ) 
    (h1 : ∀ x, f x = Real.exp x)
    (h2 : f (m + n) = f m + f n)
    (h3 : f (m + n + p) = f m + f n + f p) :
    p ≤ 2 * Real.log 2 - Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_max_p_value_l2092_209248


namespace NUMINAMATH_CALUDE_noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive_l2092_209203

/-- Represents the outcome of tossing 3 coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | THH
  | HTT
  | THT
  | TTH
  | TTT

/-- The event of having no more than one head -/
def noMoreThanOneHead (outcome : CoinToss) : Prop :=
  match outcome with
  | CoinToss.HTT | CoinToss.THT | CoinToss.TTH | CoinToss.TTT => True
  | _ => False

/-- The event of having at least two heads -/
def atLeastTwoHeads (outcome : CoinToss) : Prop :=
  match outcome with
  | CoinToss.HHH | CoinToss.HHT | CoinToss.HTH | CoinToss.THH => True
  | _ => False

/-- Theorem stating that "No more than one head" and "At least two heads" are mutually exclusive -/
theorem noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : CoinToss), ¬(noMoreThanOneHead outcome ∧ atLeastTwoHeads outcome) :=
by
  sorry

end NUMINAMATH_CALUDE_noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive_l2092_209203


namespace NUMINAMATH_CALUDE_cururu_jump_theorem_l2092_209278

/-- Represents the number of jumps of each type -/
structure JumpCount where
  typeI : ℕ
  typeII : ℕ

/-- Checks if a given jump count reaches the target position -/
def reachesTarget (jumps : JumpCount) (targetEast targetNorth : ℤ) : Prop :=
  10 * jumps.typeI - 20 * jumps.typeII = targetEast ∧
  30 * jumps.typeI - 40 * jumps.typeII = targetNorth

theorem cururu_jump_theorem :
  (∃ jumps : JumpCount, reachesTarget jumps 190 950) ∧
  (¬ ∃ jumps : JumpCount, reachesTarget jumps 180 950) := by
  sorry

#check cururu_jump_theorem

end NUMINAMATH_CALUDE_cururu_jump_theorem_l2092_209278


namespace NUMINAMATH_CALUDE_product_consecutive_integers_even_l2092_209258

theorem product_consecutive_integers_even (n : ℤ) : ∃ k : ℤ, n * (n + 1) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_product_consecutive_integers_even_l2092_209258


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_symmetric_roots_l2092_209245

/-- The quadratic function f(x) = x^2 - 2kx - 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 1

theorem quadratic_intersects_x_axis (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 :=
sorry

theorem symmetric_roots :
  f 0 1 = 0 ∧ f 0 (-1) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_symmetric_roots_l2092_209245


namespace NUMINAMATH_CALUDE_pencils_given_l2092_209201

theorem pencils_given (initial : ℕ) (final : ℕ) (given : ℕ) : 
  initial = 51 → final = 57 → given = final - initial → given = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_l2092_209201


namespace NUMINAMATH_CALUDE_cube_sum_over_product_equals_three_l2092_209216

theorem cube_sum_over_product_equals_three
  (p q r : ℝ)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_sum : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_equals_three_l2092_209216


namespace NUMINAMATH_CALUDE_exchange_40_dollars_l2092_209211

/-- Represents the exchange rate between Japanese Yen and Canadian Dollars -/
structure ExchangeRate where
  yen : ℕ
  dollars : ℕ

/-- Calculates the amount of yen received for a given amount of dollars based on an exchange rate -/
def exchange (rate : ExchangeRate) (dollars : ℕ) : ℕ :=
  (rate.yen * dollars) / rate.dollars

theorem exchange_40_dollars :
  let rate : ExchangeRate := { yen := 7500, dollars := 65 }
  exchange rate 40 = 4615 := by
  sorry

end NUMINAMATH_CALUDE_exchange_40_dollars_l2092_209211


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l2092_209228

-- Define propositions P and Q
def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l2092_209228


namespace NUMINAMATH_CALUDE_beef_price_per_pound_l2092_209294

/-- The price of beef per pound given the total cost, number of packs, and weight per pack -/
def price_per_pound (total_cost : ℚ) (num_packs : ℕ) (weight_per_pack : ℚ) : ℚ :=
  total_cost / (num_packs * weight_per_pack)

/-- Theorem: The price of beef per pound is $5.50 -/
theorem beef_price_per_pound :
  price_per_pound 110 5 4 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_beef_price_per_pound_l2092_209294


namespace NUMINAMATH_CALUDE_total_students_suggestion_l2092_209281

theorem total_students_suggestion (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 324)
  (h2 : bacon = 374)
  (h3 : tomatoes = 128) :
  mashed_potatoes + bacon + tomatoes = 826 := by
  sorry

end NUMINAMATH_CALUDE_total_students_suggestion_l2092_209281


namespace NUMINAMATH_CALUDE_a_range_l2092_209291

theorem a_range (a : ℝ) (h1 : a > 0) :
  let p := ∀ x y : ℝ, x < y → a^x < a^y
  let q := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
  (¬p ∧ ¬q) ∧ (p ∨ q) → a ∈ Set.Ioo 0 1 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2092_209291


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l2092_209267

/-- The line equation in terms of a, x, and y -/
def line_equation (a x y : ℝ) : Prop :=
  (3*a - 1)*x + (2 - a)*y - 1 = 0

/-- The second quadrant -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- The line does not pass through the second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y, line_equation a x y → ¬ second_quadrant x y

/-- The main theorem -/
theorem line_not_in_second_quadrant_iff (a : ℝ) :
  not_in_second_quadrant a ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l2092_209267


namespace NUMINAMATH_CALUDE_fractal_sequence_2000_and_sum_l2092_209233

/-- The fractal sequence a_n -/
def fractal_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let k := (n + 1).log2
    let prev_len := 2^k - 1
    if n + 1 ≤ prev_len then fractal_sequence n
    else if n + 1 = 2^k then k + 1
    else fractal_sequence (n - prev_len)

/-- Sum of the first n terms of the fractal sequence -/
def fractal_sum (n : ℕ) : ℕ :=
  (List.range n).map fractal_sequence |>.sum

theorem fractal_sequence_2000_and_sum :
  fractal_sequence 1999 = 2 ∧ fractal_sum 2000 = 4004 := by
  sorry


end NUMINAMATH_CALUDE_fractal_sequence_2000_and_sum_l2092_209233


namespace NUMINAMATH_CALUDE_inequality_proof_l2092_209232

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2092_209232


namespace NUMINAMATH_CALUDE_prob_green_marble_l2092_209266

/-- The probability of drawing a green marble from a box of 90 marbles -/
theorem prob_green_marble (total_marbles : ℕ) (prob_white : ℝ) (prob_red_or_blue : ℝ) :
  total_marbles = 90 →
  prob_white = 1 / 6 →
  prob_red_or_blue = 0.6333333333333333 →
  ∃ (prob_green : ℝ), prob_green = 0.2 ∧ prob_white + prob_red_or_blue + prob_green = 1 :=
by sorry

end NUMINAMATH_CALUDE_prob_green_marble_l2092_209266


namespace NUMINAMATH_CALUDE_three_divisions_not_imply_symmetry_l2092_209230

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon for this problem
  mk :: 

/-- A division of a polygon is a way to split it into two equal parts. -/
structure Division (P : Polygon) where
  -- We don't need to define the full structure of a division for this problem
  mk ::

/-- A symmetry of a polygon is either a center of symmetry or an axis of symmetry. -/
inductive Symmetry (P : Polygon)
  | Center : Symmetry P
  | Axis : Symmetry P

/-- A polygon has three divisions if there exist three distinct ways to split it into two equal parts. -/
def has_three_divisions (P : Polygon) : Prop :=
  ∃ (d1 d2 d3 : Division P), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

/-- A polygon has a symmetry if it has either a center of symmetry or an axis of symmetry. -/
def has_symmetry (P : Polygon) : Prop :=
  ∃ (s : Symmetry P), true

/-- 
The existence of three ways to divide a polygon into two equal parts 
does not necessarily imply the existence of a center or axis of symmetry for that polygon.
-/
theorem three_divisions_not_imply_symmetry :
  ∃ (P : Polygon), has_three_divisions P ∧ ¬has_symmetry P :=
sorry

end NUMINAMATH_CALUDE_three_divisions_not_imply_symmetry_l2092_209230


namespace NUMINAMATH_CALUDE_cyclist_hill_time_l2092_209287

/-- Calculates the total time for a cyclist to climb and descend a hill. -/
theorem cyclist_hill_time (hill_length : Real) (climbing_speed_kmh : Real) : 
  hill_length = 400 ∧ 
  climbing_speed_kmh = 7.2 →
  (let climbing_speed_ms := climbing_speed_kmh * (1000 / 3600)
   let descending_speed_ms := 2 * climbing_speed_ms
   let time_climbing := hill_length / climbing_speed_ms
   let time_descending := hill_length / descending_speed_ms
   time_climbing + time_descending) = 300 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_hill_time_l2092_209287


namespace NUMINAMATH_CALUDE_ann_has_eight_bags_l2092_209207

/-- Represents the number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1800

/-- Represents the total number of apples Ann has -/
def anns_total_apples : ℕ := 1800

/-- Represents the number of apples in one of Pam's bags -/
def pams_bag_count : ℕ := 3 * geralds_bag_count

/-- Represents the number of apples in one of Ann's bags -/
def anns_bag_count : ℕ := 2 * pams_bag_count

/-- Theorem stating that Ann has 8 bags of apples -/
theorem ann_has_eight_bags : 
  anns_total_apples / anns_bag_count = 8 ∧ 
  anns_total_apples % anns_bag_count = 0 :=
by sorry

end NUMINAMATH_CALUDE_ann_has_eight_bags_l2092_209207


namespace NUMINAMATH_CALUDE_mel_weight_proof_l2092_209229

/-- Mel's weight in pounds -/
def mels_weight : ℝ := 70

/-- Brenda's weight in pounds -/
def brendas_weight : ℝ := 220

/-- Relationship between Brenda's and Mel's weights -/
def weight_relationship (m : ℝ) : Prop := brendas_weight = 3 * m + 10

theorem mel_weight_proof : 
  weight_relationship mels_weight ∧ mels_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_mel_weight_proof_l2092_209229


namespace NUMINAMATH_CALUDE_evaluate_expression_l2092_209280

theorem evaluate_expression : (2^3)^2 - (3^2)^3 = -665 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2092_209280


namespace NUMINAMATH_CALUDE_stairs_in_building_correct_stairs_count_l2092_209205

theorem stairs_in_building (ned_speed : ℕ) (bomb_time_left : ℕ) (time_spent_running : ℕ) (diffuse_time : ℕ) : ℕ :=
  let total_run_time := time_spent_running + (bomb_time_left - diffuse_time)
  total_run_time / ned_speed

theorem correct_stairs_count : stairs_in_building 11 72 165 17 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stairs_in_building_correct_stairs_count_l2092_209205


namespace NUMINAMATH_CALUDE_club_officer_selection_count_l2092_209297

/-- Represents a club with members of two genders -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of ways to select a president, vice-president, and secretary -/
def selectOfficers (club : Club) : Nat :=
  club.total_members * club.boys * (club.total_members - 2)

/-- The theorem to prove -/
theorem club_officer_selection_count (club : Club) 
  (h1 : club.total_members = 30)
  (h2 : club.boys = 15)
  (h3 : club.girls = 15)
  (h4 : club.total_members = club.boys + club.girls) :
  selectOfficers club = 12600 := by
  sorry

#eval selectOfficers { total_members := 30, boys := 15, girls := 15 }

end NUMINAMATH_CALUDE_club_officer_selection_count_l2092_209297


namespace NUMINAMATH_CALUDE_job_completion_time_l2092_209239

/-- Represents the workforce and time required to complete a job -/
structure JobInfo where
  initialWorkforce : ℕ
  initialDays : ℕ
  extraWorkers : ℕ
  joinInterval : ℕ

/-- Calculates the total time required to complete the job given the job information -/
def calculateTotalTime (job : JobInfo) : ℕ :=
  sorry

/-- Theorem stating that for the given job information, the total time is 12 days -/
theorem job_completion_time (job : JobInfo) 
  (h1 : job.initialWorkforce = 20)
  (h2 : job.initialDays = 15)
  (h3 : job.extraWorkers = 10)
  (h4 : job.joinInterval = 5) : 
  calculateTotalTime job = 12 :=
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2092_209239


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l2092_209252

theorem parabola_equation_from_hyperbola_focus : ∃ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 / 6 = 1) →
  c^2 = a^2 + b^2 →
  (∀ x y : ℝ, y^2 = 4 * c * x ↔ y^2 = 12 * x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l2092_209252


namespace NUMINAMATH_CALUDE_solve_equation_l2092_209277

theorem solve_equation : ∃ x : ℚ, 3 * x + 15 = (1/3) * (8 * x - 24) ∧ x = -69 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2092_209277


namespace NUMINAMATH_CALUDE_first_sequence_30th_term_l2092_209279

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The 30th term of the first arithmetic sequence is 178 -/
theorem first_sequence_30th_term :
  arithmeticSequenceTerm 4 6 30 = 178 := by
  sorry

end NUMINAMATH_CALUDE_first_sequence_30th_term_l2092_209279


namespace NUMINAMATH_CALUDE_place_value_ratio_l2092_209273

/-- The number we're analyzing -/
def number : ℚ := 90347.6208

/-- The place value of a digit in a decimal number -/
def place_value (digit : ℕ) (position : ℤ) : ℚ := (digit : ℚ) * 10 ^ position

/-- The position of the digit 0 in the number (counting from right, with decimal point at 0) -/
def zero_position : ℤ := 4

/-- The position of the digit 6 in the number (counting from right, with decimal point at 0) -/
def six_position : ℤ := -1

theorem place_value_ratio :
  place_value 1 zero_position / place_value 1 six_position = 100000 := by sorry

end NUMINAMATH_CALUDE_place_value_ratio_l2092_209273


namespace NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l2092_209265

/-- Given two right triangles ABC and ABD sharing hypotenuse AB, 
    where BC = 3, AC = a, and AD = 4, prove that BD = √(a² - 7) -/
theorem right_triangles_shared_hypotenuse 
  (a : ℝ) 
  (h : a ≥ Real.sqrt 7) : 
  ∃ (AB BC AC AD BD : ℝ),
    BC = 3 ∧ 
    AC = a ∧ 
    AD = 4 ∧
    AB ^ 2 = AC ^ 2 + BC ^ 2 ∧ 
    AB ^ 2 = AD ^ 2 + BD ^ 2 ∧
    BD = Real.sqrt (a ^ 2 - 7) := by
  sorry


end NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l2092_209265


namespace NUMINAMATH_CALUDE_odd_power_sum_divisibility_l2092_209264

theorem odd_power_sum_divisibility (k : ℕ) (x y : ℤ) :
  (∃ q : ℤ, x^(2*k-1) + y^(2*k-1) = (x+y) * q) →
  (∃ r : ℤ, x^(2*k+1) + y^(2*k+1) = (x+y) * r) :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisibility_l2092_209264


namespace NUMINAMATH_CALUDE_min_value_problem_l2092_209259

theorem min_value_problem (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (hab : a + b = 1) (hmn : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2092_209259


namespace NUMINAMATH_CALUDE_volunteers_distribution_l2092_209210

theorem volunteers_distribution (n : ℕ) (k : ℕ) : 
  n = 6 → k = 4 → (k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n) = 1564 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_distribution_l2092_209210


namespace NUMINAMATH_CALUDE_total_crayons_l2092_209256

/-- Given a box of crayons where there are four times as many red crayons as blue crayons,
    and there are 3 blue crayons, prove that the total number of crayons is 15. -/
theorem total_crayons (blue : ℕ) (red : ℕ) (h1 : blue = 3) (h2 : red = 4 * blue) :
  blue + red = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l2092_209256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2092_209276

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 - a 9 + a 17 = 7) :
  a 3 + a 15 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2092_209276


namespace NUMINAMATH_CALUDE_alice_plate_stacking_l2092_209298

theorem alice_plate_stacking (initial_plates : ℕ) (first_addition : ℕ) (total_plates : ℕ) : 
  initial_plates = 27 → 
  first_addition = 37 → 
  total_plates = 83 → 
  total_plates - (initial_plates + first_addition) = 19 := by
sorry

end NUMINAMATH_CALUDE_alice_plate_stacking_l2092_209298


namespace NUMINAMATH_CALUDE_determine_F_l2092_209226

def first_number (D E : ℕ) : ℕ := 9000000 + 600000 + 100000 * D + 10000 + 1000 * E + 800 + 2

def second_number (D E F : ℕ) : ℕ := 5000000 + 400000 + 100000 * E + 10000 * D + 2000 + 100 + 10 * F

theorem determine_F :
  ∀ D E F : ℕ,
  D < 10 → E < 10 → F < 10 →
  (first_number D E) % 3 = 0 →
  (second_number D E F) % 3 = 0 →
  F = 2 := by
sorry

end NUMINAMATH_CALUDE_determine_F_l2092_209226


namespace NUMINAMATH_CALUDE_system_solution_range_l2092_209208

theorem system_solution_range (x y k : ℝ) : 
  (2 * x - 3 * y = 5) → 
  (2 * x - y = k) → 
  (x > y) → 
  (k > -5) :=
sorry

end NUMINAMATH_CALUDE_system_solution_range_l2092_209208


namespace NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l2092_209282

theorem triangle_angle_A_is_30_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B)
  (h3 : 0 < A ∧ A < π)
  (h4 : 0 < B ∧ B < π)
  (h5 : 0 < C ∧ C < π)
  (h6 : A + B + C = π)
  (h7 : a / Real.sin A = b / Real.sin B)
  (h8 : b / Real.sin B = c / Real.sin C)
  : A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l2092_209282


namespace NUMINAMATH_CALUDE_coloring_count_l2092_209257

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The number of parts to be colored -/
def num_parts : ℕ := 3

/-- A function that calculates the number of coloring possibilities -/
def count_colorings : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Theorem stating that the number of valid colorings is 60 -/
theorem coloring_count : count_colorings = 60 := by
  sorry

end NUMINAMATH_CALUDE_coloring_count_l2092_209257


namespace NUMINAMATH_CALUDE_smallest_a_satisfying_equation_l2092_209231

theorem smallest_a_satisfying_equation :
  ∃ a : ℝ, (a = -Real.sqrt (62/5)) ∧
    (∀ b : ℝ, (8*Real.sqrt ((3*b)^2 + 2^2) - 5*b^2 - 2) / (Real.sqrt (2 + 5*b^2) + 4) = 3 → a ≤ b) ∧
    (8*Real.sqrt ((3*a)^2 + 2^2) - 5*a^2 - 2) / (Real.sqrt (2 + 5*a^2) + 4) = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_satisfying_equation_l2092_209231


namespace NUMINAMATH_CALUDE_two_digit_sum_divisible_by_17_l2092_209254

/-- A function that reverses a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

/-- A predicate that checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_sum_divisible_by_17 :
  ∀ A : ℕ, is_two_digit A →
    (A + reverse_digits A) % 17 = 0 ↔ A = 89 ∨ A = 98 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisible_by_17_l2092_209254


namespace NUMINAMATH_CALUDE_num_divisors_5400_multiple_of_5_l2092_209213

/-- The number of positive divisors of 5400 that are multiples of 5 -/
def num_divisors_multiple_of_5 (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 5400 that are multiples of 5 is 24 -/
theorem num_divisors_5400_multiple_of_5 :
  num_divisors_multiple_of_5 5400 = 24 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_5400_multiple_of_5_l2092_209213


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l2092_209204

def downstream_distance : ℝ := 24
def upstream_distance : ℝ := 18
def time : ℝ := 3
def current_speed : ℝ := 2

def man_speed : ℝ := 6

theorem man_speed_in_still_water :
  (downstream_distance / time = man_speed + current_speed) ∧
  (upstream_distance / time = man_speed - current_speed) :=
by sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l2092_209204


namespace NUMINAMATH_CALUDE_sin_B_range_in_acute_triangle_l2092_209272

theorem sin_B_range_in_acute_triangle (A B C : Real) (a b c : Real) (S : Real) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  S = (1 / 2) * b * c * Real.sin A →
  a^2 = 2 * S + (b - c)^2 →
  3 / 5 < Real.sin B ∧ Real.sin B < 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_B_range_in_acute_triangle_l2092_209272
