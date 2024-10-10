import Mathlib

namespace cylinder_volume_relation_l520_52031

/-- Given two cylinders X and Y with the following properties:
    1. The height of X equals the diameter of Y
    2. The diameter of X equals the height of Y (denoted as k)
    3. The volume of X is three times the volume of Y
    This theorem states that the volume of Y can be expressed as (1/4) π k^3 cubic units. -/
theorem cylinder_volume_relation (k : ℝ) (hk : k > 0) :
  ∃ (r_x h_x r_y : ℝ),
    r_x > 0 ∧ h_x > 0 ∧ r_y > 0 ∧
    h_x = 2 * r_y ∧
    2 * r_x = k ∧
    π * r_x^2 * h_x = 3 * (π * r_y^2 * k) ∧
    π * r_y^2 * k = (1/4) * π * k^3 :=
sorry

end cylinder_volume_relation_l520_52031


namespace range_of_m_for_false_proposition_l520_52035

theorem range_of_m_for_false_proposition :
  (∀ x : ℝ, x^2 - m*x - m > 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
by
  sorry

end range_of_m_for_false_proposition_l520_52035


namespace min_value_x_plus_3y_l520_52049

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (y + 3) = 1 / 4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + 1) + 1 / (b + 3) = 1 / 4 → 
  x + 3 * y ≤ a + 3 * b ∧ x + 3 * y = 6 + 8 * Real.sqrt 3 :=
sorry

end min_value_x_plus_3y_l520_52049


namespace square_of_85_l520_52029

theorem square_of_85 : 85^2 = 7225 := by
  sorry

end square_of_85_l520_52029


namespace arithmetic_sequence_sum_2018_l520_52006

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term : a 1 = -2018
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + (n * (n - 1) / 2) * seq.d

/-- The main theorem -/
theorem arithmetic_sequence_sum_2018 (seq : ArithmeticSequence) :
  (sum_n seq 2015 / 2015) - (sum_n seq 2013 / 2013) = 2 →
  sum_n seq 2018 = -2018 :=
by
  sorry

end arithmetic_sequence_sum_2018_l520_52006


namespace binomial_10_3_l520_52020

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l520_52020


namespace triangle_arithmetic_sides_cosine_identity_l520_52090

/-- In a triangle ABC where the sides form an arithmetic sequence, 
    5 cos A - 4 cos A cos C + 5 cos C equals 8 -/
theorem triangle_arithmetic_sides_cosine_identity 
  (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  2 * b = a + c →
  5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 8 := by
  sorry

end triangle_arithmetic_sides_cosine_identity_l520_52090


namespace ellipse_and_line_properties_l520_52045

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  k : ℝ

/-- Theorem stating the properties of the ellipse and line -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (h1 : C.a^2 / C.b^2 = 4 / 3)  -- Eccentricity condition
  (h2 : 1 / C.a^2 + (9/4) / C.b^2 = 1)  -- Point (1, 3/2) lies on the ellipse
  (l : Line)
  (h3 : ∀ x y, y = l.k * (x - 1))  -- Line equation
  (h4 : ∃ x1 y1 x2 y2, 
    x1^2 / C.a^2 + y1^2 / C.b^2 = 1 ∧
    x2^2 / C.a^2 + y2^2 / C.b^2 = 1 ∧
    y1 = l.k * (x1 - 1) ∧
    y2 = l.k * (x2 - 1) ∧
    x1 * x2 + y1 * y2 = -2)  -- Intersection points and dot product condition
  : C.a^2 = 4 ∧ C.b^2 = 3 ∧ l.k^2 = 2 := by sorry

end ellipse_and_line_properties_l520_52045


namespace paco_initial_cookies_l520_52062

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 21

/-- The number of cookies Paco had left -/
def cookies_left : ℕ := 7

/-- The initial number of cookies Paco had -/
def initial_cookies : ℕ := cookies_eaten + cookies_left

theorem paco_initial_cookies : initial_cookies = 28 := by sorry

end paco_initial_cookies_l520_52062


namespace problem_statement_l520_52056

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2015 + b^2016 = -1 := by
sorry

end problem_statement_l520_52056


namespace peters_nickels_problem_l520_52098

theorem peters_nickels_problem :
  ∃! n : ℕ, 40 < n ∧ n < 400 ∧ 
    n % 4 = 2 ∧ n % 5 = 2 ∧ n % 7 = 2 ∧ n = 142 := by
  sorry

end peters_nickels_problem_l520_52098


namespace factorization_of_2a_5_minus_8a_l520_52040

theorem factorization_of_2a_5_minus_8a (a : ℝ) : 
  2 * a^5 - 8 * a = 2 * a * (a^2 + 2) * (a + Real.sqrt 2) * (a - Real.sqrt 2) := by
  sorry

end factorization_of_2a_5_minus_8a_l520_52040


namespace farm_leg_count_l520_52019

/-- The number of legs for animals on a farm --/
def farm_legs (total_animals : ℕ) (num_ducks : ℕ) (duck_legs : ℕ) (dog_legs : ℕ) : ℕ :=
  let num_dogs := total_animals - num_ducks
  num_ducks * duck_legs + num_dogs * dog_legs

/-- Theorem stating the total number of legs on the farm --/
theorem farm_leg_count : farm_legs 11 6 2 4 = 32 := by
  sorry

end farm_leg_count_l520_52019


namespace largest_three_digit_base5_l520_52072

-- Define a function to convert a three-digit base-5 number to base-10
def base5ToBase10 (a b c : Nat) : Nat :=
  a * 5^2 + b * 5^1 + c * 5^0

-- Theorem statement
theorem largest_three_digit_base5 : 
  base5ToBase10 4 4 4 = 124 := by sorry

end largest_three_digit_base5_l520_52072


namespace gas_fill_calculation_l520_52032

theorem gas_fill_calculation (cost_today : ℝ) (rollback : ℝ) (friday_fill : ℝ) 
  (total_spend : ℝ) (total_liters : ℝ) 
  (h1 : cost_today = 1.4)
  (h2 : rollback = 0.4)
  (h3 : friday_fill = 25)
  (h4 : total_spend = 39)
  (h5 : total_liters = 35) :
  ∃ (today_fill : ℝ), 
    today_fill = 10 ∧ 
    cost_today * today_fill + (cost_today - rollback) * friday_fill = total_spend ∧
    today_fill + friday_fill = total_liters :=
by sorry

end gas_fill_calculation_l520_52032


namespace sum_of_max_min_a_l520_52092

theorem sum_of_max_min_a (a b : ℝ) (h : a - 2*a*b + 2*a*b^2 + 4 = 0) :
  ∃ (a_max a_min : ℝ),
    (∀ x : ℝ, (∃ y : ℝ, x - 2*x*y + 2*x*y^2 + 4 = 0) → x ≤ a_max ∧ x ≥ a_min) ∧
    a_max + a_min = -8 :=
sorry

end sum_of_max_min_a_l520_52092


namespace square_binomial_constant_l520_52047

theorem square_binomial_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 50*x + c = (x + a)^2 + b) → c = 625 := by
  sorry

end square_binomial_constant_l520_52047


namespace ferris_wheel_problem_l520_52026

theorem ferris_wheel_problem (capacity : ℕ) (waiting : ℕ) (h1 : capacity = 56) (h2 : waiting = 92) :
  waiting - capacity = 36 := by
  sorry

end ferris_wheel_problem_l520_52026


namespace game_ends_after_54_rounds_l520_52033

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  playerA : Player
  playerB : Player
  playerC : Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- Main theorem: The game ends after exactly 54 rounds -/
theorem game_ends_after_54_rounds :
  let initialState : GameState := {
    playerA := { tokens := 20 },
    playerB := { tokens := 19 },
    playerC := { tokens := 18 },
    rounds := 0
  }
  ∃ (finalState : GameState),
    (finalState.rounds = 54) ∧
    (gameEnded finalState) ∧
    (∀ (intermediateState : GameState),
      intermediateState.rounds < 54 →
      ¬(gameEnded intermediateState)) :=
  sorry

end game_ends_after_54_rounds_l520_52033


namespace sum_fifth_sixth_is_180_l520_52064

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_first_two : a 1 + a 2 = 20
  sum_third_fourth : a 3 + a 4 = 60

/-- The sum of the fifth and sixth terms of the geometric sequence is 180 -/
theorem sum_fifth_sixth_is_180 (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 180 := by
  sorry

end sum_fifth_sixth_is_180_l520_52064


namespace piggy_bank_pennies_l520_52043

theorem piggy_bank_pennies (num_compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) : 
  num_compartments = 12 → 
  initial_pennies = 2 → 
  added_pennies = 6 → 
  (num_compartments * (initial_pennies + added_pennies)) = 96 := by
sorry

end piggy_bank_pennies_l520_52043


namespace least_marbles_nine_marbles_marbles_solution_l520_52046

theorem least_marbles (n : ℕ) : n > 0 ∧ n % 6 = 3 ∧ n % 4 = 1 → n ≥ 9 :=
by sorry

theorem nine_marbles : 9 % 6 = 3 ∧ 9 % 4 = 1 :=
by sorry

theorem marbles_solution : ∃ (n : ℕ), n > 0 ∧ n % 6 = 3 ∧ n % 4 = 1 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 6 = 3 ∧ m % 4 = 1 → n ≤ m :=
by sorry

end least_marbles_nine_marbles_marbles_solution_l520_52046


namespace sum_of_roots_l520_52018

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end sum_of_roots_l520_52018


namespace gcd_of_three_numbers_l520_52058

theorem gcd_of_three_numbers : Nat.gcd 12222 (Nat.gcd 18333 36666) = 6111 := by
  sorry

end gcd_of_three_numbers_l520_52058


namespace gwen_homework_l520_52076

def homework_problem (math_problems science_problems finished_problems : ℕ) : Prop :=
  let total_problems := math_problems + science_problems
  total_problems - finished_problems = 5

theorem gwen_homework :
  homework_problem 18 11 24 := by sorry

end gwen_homework_l520_52076


namespace flagpole_height_correct_l520_52080

/-- The height of the flagpole in feet -/
def flagpole_height : ℝ := 48

/-- The length of the flagpole's shadow in feet -/
def flagpole_shadow : ℝ := 72

/-- The height of the reference pole in feet -/
def reference_pole_height : ℝ := 18

/-- The length of the reference pole's shadow in feet -/
def reference_pole_shadow : ℝ := 27

/-- Theorem stating that the flagpole height is correct given the shadow lengths -/
theorem flagpole_height_correct :
  flagpole_height * reference_pole_shadow = reference_pole_height * flagpole_shadow :=
by sorry

end flagpole_height_correct_l520_52080


namespace mural_hourly_rate_l520_52055

/-- Calculates the hourly rate for painting a mural given its dimensions, painting rate, and total charge -/
theorem mural_hourly_rate (length width : ℝ) (paint_rate : ℝ) (total_charge : ℝ) :
  length = 20 ∧ width = 15 ∧ paint_rate = 20 ∧ total_charge = 15000 →
  total_charge / (length * width * paint_rate / 60) = 150 := by
  sorry

#check mural_hourly_rate

end mural_hourly_rate_l520_52055


namespace hyperbola_eccentricity_l520_52028

theorem hyperbola_eccentricity :
  let hyperbola := fun (x y : ℝ) => x^2 / 5 - y^2 / 4 = 1
  ∃ (e : ℝ), e = (3 * Real.sqrt 5) / 5 ∧
    ∀ (x y : ℝ), hyperbola x y → 
      e = Real.sqrt ((x^2 / 5) + (y^2 / 4)) / Real.sqrt (x^2 / 5) :=
by sorry

end hyperbola_eccentricity_l520_52028


namespace candy_to_drink_ratio_l520_52077

def deal_price : ℚ := 20
def ticket_price : ℚ := 8
def popcorn_price : ℚ := ticket_price - 3
def drink_price : ℚ := popcorn_price + 1
def savings : ℚ := 2

def normal_total : ℚ := deal_price + savings
def candy_price : ℚ := normal_total - (ticket_price + popcorn_price + drink_price)

theorem candy_to_drink_ratio : candy_price / drink_price = 1 / 2 := by
  sorry

end candy_to_drink_ratio_l520_52077


namespace a_range_l520_52083

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

/-- Theorem stating the range of a given the conditions -/
theorem a_range (a : ℝ) :
  (∃! x, f a x = 0) ∧ 
  (∀ x, f a x = 0 → x < 0) →
  a < -Real.sqrt 2 := by
  sorry

end a_range_l520_52083


namespace dead_to_total_ratio_is_three_to_five_l520_52069

/-- Represents the ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the problem setup -/
structure FlowerProblem where
  desired_flowers : ℕ
  seeds_per_pack : ℕ
  price_per_pack : ℕ
  total_spent : ℕ

/-- Calculates the ratio of dead seeds to total seeds -/
def dead_to_total_ratio (p : FlowerProblem) : Ratio :=
  let total_packs := p.total_spent / p.price_per_pack
  let total_seeds := total_packs * p.seeds_per_pack
  let dead_seeds := total_seeds - p.desired_flowers
  { numerator := dead_seeds, denominator := total_seeds }

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

/-- The main theorem to prove -/
theorem dead_to_total_ratio_is_three_to_five (p : FlowerProblem) 
    (h1 : p.desired_flowers = 20)
    (h2 : p.seeds_per_pack = 25)
    (h3 : p.price_per_pack = 5)
    (h4 : p.total_spent = 10) : 
    simplify_ratio (dead_to_total_ratio p) = { numerator := 3, denominator := 5 } := by
  sorry

end dead_to_total_ratio_is_three_to_five_l520_52069


namespace crazy_silly_school_books_l520_52005

theorem crazy_silly_school_books (num_movies : ℕ) (movie_book_diff : ℕ) : 
  num_movies = 17 → movie_book_diff = 6 → num_movies - movie_book_diff = 11 := by
  sorry

end crazy_silly_school_books_l520_52005


namespace chess_club_mixed_groups_l520_52038

theorem chess_club_mixed_groups 
  (total_children : ℕ) 
  (total_groups : ℕ) 
  (group_size : ℕ) 
  (boy_boy_games : ℕ) 
  (girl_girl_games : ℕ) : 
  total_children = 90 →
  total_groups = 30 →
  group_size = 3 →
  boy_boy_games = 30 →
  girl_girl_games = 14 →
  (∃ (mixed_groups : ℕ), 
    mixed_groups = 23 ∧ 
    mixed_groups + (boy_boy_games / 3) + (girl_girl_games / 3) = total_groups) :=
by sorry

end chess_club_mixed_groups_l520_52038


namespace fifteenth_term_is_53_l520_52036

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (a₁ a₂ a₃ : ℤ) : ℕ → ℤ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

/-- Theorem: The 15th term of the specific arithmetic sequence is 53 -/
theorem fifteenth_term_is_53 :
  arithmetic_sequence (-3) 1 5 15 = 53 := by
  sorry

end fifteenth_term_is_53_l520_52036


namespace phone_profit_maximization_l520_52030

theorem phone_profit_maximization
  (profit_A_B : ℕ → ℕ → ℕ)
  (h1 : profit_A_B 1 1 = 600)
  (h2 : profit_A_B 3 2 = 1400)
  (total_phones : ℕ)
  (h3 : total_phones = 20)
  (h4 : ∀ x y : ℕ, x + y = total_phones → 3 * y ≤ 2 * x) :
  ∃ (x y : ℕ),
    x + y = total_phones ∧
    3 * y ≤ 2 * x ∧
    ∀ (a b : ℕ), a + b = total_phones → 3 * b ≤ 2 * a →
      profit_A_B x y ≥ profit_A_B a b ∧
      profit_A_B x y = 5600 ∧
      x = 12 ∧ y = 8 :=
by sorry

end phone_profit_maximization_l520_52030


namespace floor_sum_abcd_l520_52057

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 2010) (h2 : c^2 + d^2 = 2010) (h3 : a * c = 1020) (h4 : b * d = 1020) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end floor_sum_abcd_l520_52057


namespace patricia_barrels_l520_52053

/-- Given a scenario where Patricia has some barrels, proves that the number of barrels is 4 -/
theorem patricia_barrels : 
  ∀ (barrel_capacity : ℝ) (flow_rate : ℝ) (fill_time : ℝ) (num_barrels : ℕ),
  barrel_capacity = 7 →
  flow_rate = 3.5 →
  fill_time = 8 →
  (flow_rate * fill_time : ℝ) = (↑num_barrels * barrel_capacity) →
  num_barrels = 4 := by
sorry

end patricia_barrels_l520_52053


namespace total_earning_calculation_l520_52041

theorem total_earning_calculation (days_a days_b days_c : ℕ) 
  (wage_ratio_a wage_ratio_b wage_ratio_c : ℕ) (wage_c : ℕ) :
  days_a = 6 →
  days_b = 9 →
  days_c = 4 →
  wage_ratio_a = 3 →
  wage_ratio_b = 4 →
  wage_ratio_c = 5 →
  wage_c = 110 →
  (days_a * (wage_c * wage_ratio_a / wage_ratio_c) +
   days_b * (wage_c * wage_ratio_b / wage_ratio_c) +
   days_c * wage_c) = 1628 :=
by sorry

end total_earning_calculation_l520_52041


namespace distance_to_nearest_town_l520_52017

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) →
  (¬ (d ≤ 7)) →
  (¬ (d ≤ 6)) →
  (d ≠ 9) →
  7 < d ∧ d < 8 :=
by sorry

end distance_to_nearest_town_l520_52017


namespace negation_of_universal_statement_l520_52089

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 1/4 ≤ 0) ↔ (∃ x : ℝ, x^2 - x + 1/4 > 0) :=
by sorry

end negation_of_universal_statement_l520_52089


namespace profit_percentage_calculation_l520_52000

/-- Calculate the profit percentage given the cost price and selling price -/
theorem profit_percentage_calculation (cost_price selling_price : ℚ) :
  cost_price = 60 →
  selling_price = 78 →
  (selling_price - cost_price) / cost_price * 100 = 30 := by
sorry

end profit_percentage_calculation_l520_52000


namespace max_ab_value_l520_52066

/-- The maximum value of ab given the conditions -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a * 2 - b * (-1) = 2) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → (x - 2)^2 + (y + 1)^2 = 4) :
  a * b ≤ 1/2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * 2 - b * (-1) = 2 ∧ a * b = 1/2 :=
sorry

end max_ab_value_l520_52066


namespace correct_average_unchanged_l520_52078

theorem correct_average_unchanged (n : ℕ) (initial_avg : ℚ) (error1 : ℚ) (wrong_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 18 →
  wrong_num = 13 →
  correct_num = 31 →
  (n : ℚ) * initial_avg - error1 - wrong_num + correct_num = (n : ℚ) * initial_avg :=
by sorry

end correct_average_unchanged_l520_52078


namespace saplings_in_park_l520_52091

theorem saplings_in_park (total_trees : ℕ) (ancient_oaks : ℕ) (fir_trees : ℕ) : 
  total_trees = 96 → ancient_oaks = 15 → fir_trees = 23 → 
  total_trees - (ancient_oaks + fir_trees) = 58 := by
  sorry

end saplings_in_park_l520_52091


namespace first_load_theorem_l520_52050

/-- Calculates the number of pieces of clothing in the first load -/
def first_load_pieces (total_pieces : ℕ) (num_equal_loads : ℕ) (pieces_per_equal_load : ℕ) : ℕ :=
  total_pieces - (num_equal_loads * pieces_per_equal_load)

/-- Theorem stating that given 59 total pieces of clothing, with 9 equal loads of 3 pieces each,
    the number of pieces in the first load is 32. -/
theorem first_load_theorem :
  first_load_pieces 59 9 3 = 32 := by
  sorry

end first_load_theorem_l520_52050


namespace P_on_y_axis_l520_52063

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the y-axis -/
def is_on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- The point P with coordinates (0, -4) -/
def P : Point :=
  { x := 0, y := -4 }

/-- Theorem: The point P(0, -4) lies on the y-axis -/
theorem P_on_y_axis : is_on_y_axis P := by
  sorry

end P_on_y_axis_l520_52063


namespace cube_greater_than_one_iff_l520_52042

theorem cube_greater_than_one_iff (x : ℝ) : x > 1 ↔ x^3 > 1 := by sorry

end cube_greater_than_one_iff_l520_52042


namespace tan_theta_plus_pi_fourth_l520_52068

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) 
  (h2 : Real.cos (θ - Real.pi / 4) = 3 / 5) : 
  Real.tan (θ + Real.pi / 4) = 3 / 4 := by
  sorry

end tan_theta_plus_pi_fourth_l520_52068


namespace ball_count_l520_52034

/-- Given a box of balls with specific properties, prove the total number of balls -/
theorem ball_count (orange purple yellow total : ℕ) : 
  orange + purple + yellow = total →  -- Total balls
  orange = 2 * n →  -- Ratio condition for orange
  purple = 3 * n →  -- Ratio condition for purple
  yellow = 4 * n →  -- Ratio condition for yellow
  yellow = 32 →     -- Given number of yellow balls
  total = 72 :=     -- Prove total number of balls
by sorry

end ball_count_l520_52034


namespace exists_cost_price_l520_52039

/-- The cost price of a watch satisfying the given conditions --/
def cost_price : ℝ → Prop := fun C =>
  3 * (0.925 * C + 265) = 3 * C * 1.053

/-- Theorem stating the existence of a cost price satisfying the conditions --/
theorem exists_cost_price : ∃ C : ℝ, cost_price C := by
  sorry

end exists_cost_price_l520_52039


namespace opposite_signs_inequality_l520_52013

theorem opposite_signs_inequality (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end opposite_signs_inequality_l520_52013


namespace not_q_necessary_not_sufficient_for_not_p_l520_52001

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that ¬q is a necessary but not sufficient condition for ¬p
theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) :=
sorry

end not_q_necessary_not_sufficient_for_not_p_l520_52001


namespace not_on_inverse_proportion_graph_l520_52065

def inverse_proportion (x y : ℝ) : Prop := x * y = 6

def point_on_graph (p : ℝ × ℝ) : Prop :=
  inverse_proportion p.1 p.2

theorem not_on_inverse_proportion_graph :
  point_on_graph (-2, -3) ∧
  point_on_graph (-3, -2) ∧
  ¬point_on_graph (1, 5) ∧
  point_on_graph (4, 1.5) :=
sorry

end not_on_inverse_proportion_graph_l520_52065


namespace double_age_proof_l520_52086

/-- The number of years in the future when Richard will be twice as old as Scott -/
def years_until_double_age : ℕ := 8

theorem double_age_proof (david_current_age richard_current_age scott_current_age : ℕ) 
  (h1 : david_current_age = 14)
  (h2 : richard_current_age = david_current_age + 6)
  (h3 : david_current_age = scott_current_age + 8) :
  richard_current_age + years_until_double_age = 2 * (scott_current_age + years_until_double_age) := by
  sorry

#check double_age_proof

end double_age_proof_l520_52086


namespace binomial_expansion_5_plus_4_cubed_l520_52067

theorem binomial_expansion_5_plus_4_cubed : (5 + 4)^3 = 729 := by
  -- The proof goes here
  sorry

end binomial_expansion_5_plus_4_cubed_l520_52067


namespace problem_statement_l520_52093

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1) : 
  m = 10 := by
sorry

end problem_statement_l520_52093


namespace purchase_savings_l520_52084

/-- Calculates the total savings on a purchase given the original and discounted prices -/
def calculateSavings (originalPrice discountedPrice : ℚ) (quantity : ℕ) : ℚ :=
  (originalPrice - discountedPrice) * quantity

/-- Calculates the discounted price given the original price and discount percentage -/
def calculateDiscountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage)

theorem purchase_savings :
  let folderQuantity : ℕ := 7
  let folderPrice : ℚ := 3
  let folderDiscount : ℚ := 0.25
  let penQuantity : ℕ := 4
  let penPrice : ℚ := 1.5
  let penDiscount : ℚ := 0.1
  let folderSavings := calculateSavings folderPrice (calculateDiscountedPrice folderPrice folderDiscount) folderQuantity
  let penSavings := calculateSavings penPrice (calculateDiscountedPrice penPrice penDiscount) penQuantity
  folderSavings + penSavings = 5.85 := by
  sorry


end purchase_savings_l520_52084


namespace arithmetic_sequence_and_sum_l520_52027

-- Define the arithmetic sequence a_n and its sum S_n
def a (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- Define T_n as the sum of first n terms of 1/S_n
def T (n : ℕ) : ℝ := sorry

-- State the given conditions
axiom S_3_eq_15 : S 3 = 15
axiom a_3_plus_a_8 : a 3 + a 8 = 2 * a 5 + 2

-- State the theorem to be proved
theorem arithmetic_sequence_and_sum (n : ℕ) : 
  a n = 2 * n + 1 ∧ T n < 3/4 := by sorry

end arithmetic_sequence_and_sum_l520_52027


namespace certain_number_subtraction_l520_52016

theorem certain_number_subtraction (x : ℝ) (y : ℝ) : 
  (3 * x = (y - x) + 4) → (x = 5) → (y = 16) := by
  sorry

end certain_number_subtraction_l520_52016


namespace completing_square_quadratic_l520_52074

theorem completing_square_quadratic (x : ℝ) : 
  x^2 + 8*x - 3 = 0 ↔ (x + 4)^2 = 19 :=
sorry

end completing_square_quadratic_l520_52074


namespace power_not_all_ones_l520_52082

theorem power_not_all_ones (a n : ℕ) : a > 1 → n > 1 → ¬∃ s : ℕ, a^n = 2^s - 1 := by
  sorry

end power_not_all_ones_l520_52082


namespace solution_set_equals_open_interval_l520_52059

def solution_set : Set ℝ := {x | x^2 - 9*x + 14 < 0 ∧ 2*x + 3 > 0}

theorem solution_set_equals_open_interval :
  solution_set = Set.Ioo 2 7 := by sorry

end solution_set_equals_open_interval_l520_52059


namespace smallest_w_proof_l520_52023

def smallest_w : ℕ := 79092

theorem smallest_w_proof :
  ∀ w : ℕ,
  w > 0 →
  (∃ k : ℕ, 1452 * w = 2^4 * 3^3 * 13^3 * k) →
  w ≥ smallest_w :=
by
  sorry

#check smallest_w_proof

end smallest_w_proof_l520_52023


namespace equation_system_solution_l520_52071

theorem equation_system_solution :
  ∀ (x y z : ℤ),
    (4 : ℝ) ^ (x^2 + 2*x*y + 1) = (z + 2 : ℝ) * 7^(|y| - 1) →
    Real.sin ((3 * Real.pi * ↑z) / 2) = 1 →
    ((x = 1 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 1 ∧ z = -1)) :=
by sorry

end equation_system_solution_l520_52071


namespace integer_expression_is_integer_l520_52010

theorem integer_expression_is_integer (n : ℤ) : ∃ m : ℤ, (n / 3 + n^2 / 2 + n^3 / 6 : ℚ) = m := by
  sorry

end integer_expression_is_integer_l520_52010


namespace min_value_theorem_l520_52094

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.rpow 2 (x - 3) = Real.rpow (1 / 2) y) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.rpow 2 (a - 3) = Real.rpow (1 / 2) b → 
    1 / x + 4 / y ≤ 1 / a + 4 / b) ∧ 1 / x + 4 / y = 3 := by
  sorry

#check min_value_theorem

end min_value_theorem_l520_52094


namespace square_area_increase_l520_52079

theorem square_area_increase (x : ℝ) : (x + 3)^2 - x^2 = 45 → x^2 = 36 := by
  sorry

end square_area_increase_l520_52079


namespace lucy_flour_problem_l520_52048

/-- The amount of flour Lucy had at the start of the week -/
def initial_flour : ℝ := 500

/-- The amount of flour Lucy used for baking cookies -/
def used_flour : ℝ := 240

/-- The amount of flour Lucy needs to buy to have a full bag -/
def flour_to_buy : ℝ := 370

theorem lucy_flour_problem :
  (initial_flour - used_flour) / 2 + flour_to_buy = initial_flour :=
by sorry

end lucy_flour_problem_l520_52048


namespace B_power_15_minus_3_power_14_l520_52060

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end B_power_15_minus_3_power_14_l520_52060


namespace integer_k_not_dividing_binomial_coefficient_l520_52054

theorem integer_k_not_dividing_binomial_coefficient (k : ℤ) : 
  k ≠ 1 ↔ ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ n ∈ S, ¬(n + k : ℤ) ∣ (Nat.choose (2 * n) n : ℤ) :=
by sorry

end integer_k_not_dividing_binomial_coefficient_l520_52054


namespace debate_participants_l520_52022

theorem debate_participants (third_school : ℕ) 
  (h1 : third_school + (third_school + 40) + 2 * (third_school + 40) = 920) : 
  third_school = 200 := by
sorry

end debate_participants_l520_52022


namespace factor_expression_l520_52009

theorem factor_expression (y : ℝ) : 3 * y * (y - 5) + 4 * (y - 5) = (3 * y + 4) * (y - 5) := by
  sorry

end factor_expression_l520_52009


namespace tangent_parallel_points_tangent_parallel_result_point_coordinates_l520_52024

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

theorem tangent_parallel_result :
  {x : ℝ | f' x = 4} = {1, -1} :=
by sorry

theorem point_coordinates :
  {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = f x ∧ f' x = 4} = {(1, 0), (-1, -4)} :=
by sorry

end tangent_parallel_points_tangent_parallel_result_point_coordinates_l520_52024


namespace product_even_odd_is_odd_l520_52007

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem product_even_odd_is_odd (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry

end product_even_odd_is_odd_l520_52007


namespace f_one_eq_zero_f_x_minus_one_lt_zero_iff_f_inequality_iff_l520_52088

/-- An increasing function f satisfying f(x/y) = f(x) - f(y) -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  domain : ∀ x, x > 0 → f x ≠ 0
  increasing : ∀ x y, x < y → f x < f y
  special_prop : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

variable (f : ℝ → ℝ) [SpecialFunction f]

/-- f(1) = 0 -/
theorem f_one_eq_zero : f 1 = 0 := by sorry

/-- f(x-1) < 0 iff x ∈ (1, 2) -/
theorem f_x_minus_one_lt_zero_iff (x : ℝ) : f (x - 1) < 0 ↔ 1 < x ∧ x < 2 := by sorry

/-- If f(2) = 1, then f(x+3) - f(1/x) < 2 iff x ∈ (0, 1) -/
theorem f_inequality_iff (h : f 2 = 1) (x : ℝ) : f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < 1 := by sorry

end f_one_eq_zero_f_x_minus_one_lt_zero_iff_f_inequality_iff_l520_52088


namespace quiz_win_probability_l520_52075

/-- Represents a quiz with multiple-choice questions. -/
structure Quiz where
  num_questions : ℕ
  num_choices : ℕ

/-- Represents the outcome of a quiz attempt. -/
structure QuizOutcome where
  correct_answers : ℕ

/-- The probability of getting a single question correct. -/
def single_question_probability (q : Quiz) : ℚ :=
  1 / q.num_choices

/-- The probability of winning the quiz. -/
def win_probability (q : Quiz) : ℚ :=
  let p := single_question_probability q
  (p ^ q.num_questions) +  -- All correct
  q.num_questions * (p ^ 3 * (1 - p))  -- Exactly 3 correct

/-- The theorem stating the probability of winning the quiz. -/
theorem quiz_win_probability (q : Quiz) (h1 : q.num_questions = 4) (h2 : q.num_choices = 3) :
  win_probability q = 1 / 9 := by
  sorry

#eval win_probability {num_questions := 4, num_choices := 3}

end quiz_win_probability_l520_52075


namespace smallest_n_perfect_powers_l520_52002

theorem smallest_n_perfect_powers : ∃ (n : ℕ), 
  (n = 151875) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (∃ k : ℕ, 3 * m = k^2) → 
    (∃ l : ℕ, 5 * m = l^5) → False) ∧
  (∃ k : ℕ, 3 * n = k^2) ∧
  (∃ l : ℕ, 5 * n = l^5) := by
sorry

end smallest_n_perfect_powers_l520_52002


namespace fraction_product_simplification_l520_52099

/-- The product of fractions in the sequence -/
def fraction_product : ℕ → ℚ
| 0 => 3 / 1
| n + 1 => fraction_product n * ((3 * (n + 1) + 6) / (3 * (n + 1)))

/-- The last term in the sequence -/
def last_term : ℚ := 3003 / 2997

/-- The number of terms in the sequence -/
def num_terms : ℕ := 999

theorem fraction_product_simplification :
  fraction_product num_terms * last_term = 1001 := by
  sorry


end fraction_product_simplification_l520_52099


namespace initial_tagged_fish_calculation_l520_52073

/-- Calculates the number of initially tagged fish in a pond -/
def initiallyTaggedFish (totalFish : ℕ) (secondCatchTotal : ℕ) (secondCatchTagged : ℕ) : ℕ :=
  (totalFish * secondCatchTagged) / secondCatchTotal

theorem initial_tagged_fish_calculation :
  initiallyTaggedFish 1250 50 2 = 50 := by
  sorry

end initial_tagged_fish_calculation_l520_52073


namespace soda_consumption_proof_l520_52011

/-- The number of soda bottles Debby bought -/
def total_soda_bottles : ℕ := 360

/-- The number of days the soda bottles lasted -/
def days_lasted : ℕ := 40

/-- The number of soda bottles Debby drank per day -/
def soda_bottles_per_day : ℕ := total_soda_bottles / days_lasted

theorem soda_consumption_proof : soda_bottles_per_day = 9 := by
  sorry

end soda_consumption_proof_l520_52011


namespace triangle_side_range_l520_52095

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def hasTwoSolutions (t : Triangle) : Prop :=
  t.b * Real.sin t.A < t.a ∧ t.a < t.b

-- Theorem statement
theorem triangle_side_range (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.A = π / 6)
  (h3 : hasTwoSolutions t) :
  1 < t.a ∧ t.a < 2 :=
by
  sorry

end triangle_side_range_l520_52095


namespace center_of_specific_pyramid_l520_52097

/-- The center of the circumscribed sphere of a triangular pyramid -/
def center_of_circumscribed_sphere (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The center of the circumscribed sphere of a triangular pyramid with
    vertices at (1,0,1), (1,1,0), (0,1,1), and (0,0,0) has coordinates (1/2, 1/2, 1/2) -/
theorem center_of_specific_pyramid :
  let A : ℝ × ℝ × ℝ := (1, 0, 1)
  let B : ℝ × ℝ × ℝ := (1, 1, 0)
  let C : ℝ × ℝ × ℝ := (0, 1, 1)
  let D : ℝ × ℝ × ℝ := (0, 0, 0)
  center_of_circumscribed_sphere A B C D = (1/2, 1/2, 1/2) := by sorry

end center_of_specific_pyramid_l520_52097


namespace square_perimeter_l520_52012

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 200) (h2 : side^2 = area) :
  4 * side = 40 * Real.sqrt 2 := by
  sorry

end square_perimeter_l520_52012


namespace binary_110011_equals_51_l520_52051

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the given binary number is equal to 51 in decimal -/
theorem binary_110011_equals_51 :
  binary_to_decimal binary_number = 51 := by
  sorry

end binary_110011_equals_51_l520_52051


namespace greatest_n_perfect_square_l520_52025

/-- Sum of squares from 1 to n -/
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Product of sum of squares -/
def product_sum_squares (n : ℕ) : ℕ :=
  (sum_squares n) * (sum_squares (2 * n) - sum_squares n)

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Main theorem -/
theorem greatest_n_perfect_square :
  (∀ k : ℕ, k ≤ 2023 → is_perfect_square (product_sum_squares k) → k ≤ 1921) ∧
  is_perfect_square (product_sum_squares 1921) := by sorry

end greatest_n_perfect_square_l520_52025


namespace bottles_left_l520_52070

theorem bottles_left (initial : Float) (maria_drank : Float) (sister_drank : Float) :
  initial = 45.0 →
  maria_drank = 14.0 →
  sister_drank = 8.0 →
  initial - maria_drank - sister_drank = 23.0 :=
by sorry

end bottles_left_l520_52070


namespace trick_deck_total_spent_l520_52037

/-- The total amount spent by Tom and his friend on trick decks -/
theorem trick_deck_total_spent 
  (price_per_deck : ℕ)
  (tom_decks : ℕ)
  (friend_decks : ℕ)
  (h1 : price_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : friend_decks = 5) :
  price_per_deck * (tom_decks + friend_decks) = 64 :=
by sorry

end trick_deck_total_spent_l520_52037


namespace square_area_on_parabola_l520_52044

/-- A square with two vertices on a parabola and one side on a line -/
structure SquareOnParabola where
  /-- The side length of the square -/
  side : ℝ
  /-- The y-intercept of the line parallel to y = 2x - 22 that contains two vertices of the square -/
  b : ℝ
  /-- Two vertices of the square lie on the parabola y = x^2 -/
  vertices_on_parabola : ∃ (x₁ x₂ : ℝ), x₁^2 = (2 * x₁ + b) ∧ x₂^2 = (2 * x₂ + b) ∧ (x₁ - x₂)^2 + (x₁^2 - x₂^2)^2 = side^2
  /-- One side of the square lies on the line y = 2x - 22 -/
  side_on_line : side = |b + 22| / Real.sqrt 5

/-- The theorem stating the possible areas of the square -/
theorem square_area_on_parabola (s : SquareOnParabola) :
  s.side^2 = 115.2 ∨ s.side^2 = 156.8 := by
  sorry

end square_area_on_parabola_l520_52044


namespace largest_multiple_of_9_below_75_l520_52052

theorem largest_multiple_of_9_below_75 : ∃ n : ℕ, n * 9 = 72 ∧ 72 < 75 ∧ ∀ m : ℕ, m * 9 < 75 → m * 9 ≤ 72 :=
by sorry

end largest_multiple_of_9_below_75_l520_52052


namespace time_to_work_l520_52021

-- Define the variables
def speed_to_work : ℝ := 80
def speed_to_home : ℝ := 120
def total_time : ℝ := 3

-- Define the theorem
theorem time_to_work : 
  ∃ (distance : ℝ),
    distance / speed_to_work + distance / speed_to_home = total_time ∧
    (distance / speed_to_work) * 60 = 108 := by
  sorry

end time_to_work_l520_52021


namespace parabola_roots_difference_l520_52008

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_roots_difference (a b c : ℝ) :
  (∃ (h k : ℝ), h = 3 ∧ k = -3 ∧ ∀ x, parabola a b c x = a * (x - h)^2 + k) →
  parabola a b c 5 = 9 →
  (∃ (m n : ℝ), m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0) →
  ∃ (m n : ℝ), m - n = 2 :=
sorry

end parabola_roots_difference_l520_52008


namespace little_d_can_win_l520_52081

/-- Represents a point in the 3D lattice grid -/
structure LatticePoint where
  x : Int
  y : Int
  z : Int

/-- Represents a plane perpendicular to a coordinate axis -/
inductive Plane
  | X (y z : Int)
  | Y (x z : Int)
  | Z (x y : Int)

/-- Represents the state of the game -/
structure GameState where
  markedPoints : Set LatticePoint
  munchedPlanes : Set Plane

/-- Represents a move by Little D -/
def LittleDMove := LatticePoint

/-- Represents a move by Big Z -/
def BigZMove := Plane

/-- A strategy for Little D is a function that takes the current game state
    and returns the next move -/
def LittleDStrategy := GameState → LittleDMove

/-- Check if n consecutive points are marked on a line parallel to a coordinate axis -/
def hasConsecutiveMarkedPoints (state : GameState) (n : Nat) : Prop :=
  ∃ (start : LatticePoint) (axis : Fin 3),
    ∀ i : Fin n,
      let point : LatticePoint :=
        match axis with
        | 0 => ⟨start.x + i.val, start.y, start.z⟩
        | 1 => ⟨start.x, start.y + i.val, start.z⟩
        | 2 => ⟨start.x, start.y, start.z + i.val⟩
      point ∈ state.markedPoints

/-- The main theorem: Little D can win for any n -/
theorem little_d_can_win (n : Nat) :
  ∃ (strategy : LittleDStrategy),
    ∀ (bigZMoves : Nat → BigZMove),
      ∃ (finalState : GameState),
        hasConsecutiveMarkedPoints finalState n :=
  sorry

end little_d_can_win_l520_52081


namespace smallest_scalene_triangle_with_prime_perimeter_l520_52087

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  b = a + 2 ∧ c = b + 2

theorem smallest_scalene_triangle_with_prime_perimeter :
  ∃ (a b c : ℕ),
    areConsecutiveOddPrimes a b c ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    a + b + c = 23 ∧
    (∀ (x y z : ℕ),
      areConsecutiveOddPrimes x y z →
      isValidTriangle x y z →
      isPrime (x + y + z) →
      x + y + z ≥ 23) :=
sorry

end smallest_scalene_triangle_with_prime_perimeter_l520_52087


namespace john_total_distance_l520_52014

/-- Calculates the total distance driven given two separate trips with different speeds and durations. -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that John's total driving distance is 235 miles. -/
theorem john_total_distance :
  total_distance 35 2 55 3 = 235 := by
  sorry

end john_total_distance_l520_52014


namespace stack_probability_theorem_l520_52003

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the configuration of crates in the stack -/
structure StackConfiguration where
  count_3ft : ℕ
  count_4ft : ℕ
  count_6ft : ℕ

def crate_dimensions : CrateDimensions :=
  { length := 3, width := 4, height := 6 }

def total_crates : ℕ := 12

def target_height : ℕ := 50

def valid_configuration (config : StackConfiguration) : Prop :=
  config.count_3ft + config.count_4ft + config.count_6ft = total_crates ∧
  3 * config.count_3ft + 4 * config.count_4ft + 6 * config.count_6ft = target_height

def count_valid_configurations : ℕ := 30690

def total_possible_configurations : ℕ := 3^total_crates

theorem stack_probability_theorem :
  (count_valid_configurations : ℚ) / total_possible_configurations = 10230 / 531441 :=
sorry

end stack_probability_theorem_l520_52003


namespace inequality_range_l520_52061

/-- Given that m < (e^x) / (x*e^x - x + 1) has exactly two integer solutions, 
    prove that the range of m is [e^2 / (2e^2 - 1), 1) -/
theorem inequality_range (m : ℝ) : 
  (∃! (a b : ℤ), ∀ (x : ℤ), m < (Real.exp x) / (x * Real.exp x - x + 1) ↔ x = a ∨ x = b) →
  m ∈ Set.Ici (Real.exp 2 / (2 * Real.exp 2 - 1)) ∩ Set.Iio 1 :=
by sorry

end inequality_range_l520_52061


namespace combined_tax_rate_l520_52085

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.4) 
  (h2 : mindy_rate = 0.3) 
  (h3 : income_ratio = 3) :
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.325 := by
  sorry

end combined_tax_rate_l520_52085


namespace min_value_implies_a_l520_52004

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem min_value_implies_a (a : ℝ) :
  (∃ m : ℝ, m = 5 ∧ ∀ x : ℝ, f x a ≥ m) → a = -6 ∨ a = 4 := by sorry

end min_value_implies_a_l520_52004


namespace newspaper_conference_max_overlap_l520_52015

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 35 →
  editors > 38 →
  writers + editors + x = total →
  x ≤ 26 :=
by sorry

end newspaper_conference_max_overlap_l520_52015


namespace campground_distance_l520_52096

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance :
  distance_to_campground 60 5 = 300 := by
  sorry

end campground_distance_l520_52096
