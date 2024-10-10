import Mathlib

namespace jose_bottle_caps_l1394_139494

/-- 
Given that Jose starts with some bottle caps, gets 2 more from Rebecca, 
and ends up with 9 bottle caps, prove that he started with 7 bottle caps.
-/
theorem jose_bottle_caps (x : ℕ) : x + 2 = 9 → x = 7 := by
  sorry

end jose_bottle_caps_l1394_139494


namespace player_a_winning_strategy_l1394_139413

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents a cubic polynomial ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Represents the state of the game -/
structure GameState where
  polynomial : CubicPolynomial
  current_player : Player
  moves_left : Nat

/-- Represents a move in the game -/
structure Move where
  value : ℤ
  position : Nat

/-- Function to apply a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Predicate to check if a polynomial has three distinct integer roots -/
def has_three_distinct_integer_roots (p : CubicPolynomial) : Prop :=
  sorry

/-- Theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initial_state : GameState),
      initial_state.current_player = Player.A →
      initial_state.moves_left = 3 →
      ∀ (b_moves : Fin 2 → Move),
        let final_state := apply_move (apply_move (apply_move initial_state (strategy initial_state)) (b_moves 0)) (b_moves 1)
        has_three_distinct_integer_roots final_state.polynomial :=
  sorry

end player_a_winning_strategy_l1394_139413


namespace sum_of_polynomials_l1394_139436

-- Define the polynomials f, g, and h
def f (x : ℝ) : ℝ := -2 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 7
def h (x : ℝ) : ℝ := 6 * x^2 + 3 * x + 2

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -2 * x^3 - x^2 + 9 * x - 10 := by
  sorry

end sum_of_polynomials_l1394_139436


namespace zain_coin_count_l1394_139499

/-- Represents the count of each coin type --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ

/-- Calculates the total number of coins --/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.quarters + coins.dimes + coins.nickels

/-- Zain's coin count given Emerie's coin count --/
def zainCoins (emerieCoins : CoinCount) : CoinCount :=
  { quarters := emerieCoins.quarters + 10
  , dimes := emerieCoins.dimes + 10
  , nickels := emerieCoins.nickels + 10 }

theorem zain_coin_count (emerieCoins : CoinCount) 
  (h1 : emerieCoins.quarters = 6)
  (h2 : emerieCoins.dimes = 7)
  (h3 : emerieCoins.nickels = 5) : 
  totalCoins (zainCoins emerieCoins) = 48 := by
  sorry

end zain_coin_count_l1394_139499


namespace ratio_sum_equality_l1394_139458

theorem ratio_sum_equality (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : c / d = 3 / 4) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) 
  (h5 : b + d ≠ 0) : 
  (a + c) / (b + d) = 3 / 4 := by
sorry

end ratio_sum_equality_l1394_139458


namespace max_popsicles_for_zoe_l1394_139491

/-- Represents the pricing options for popsicles -/
structure PopsicleOptions where
  single_price : ℕ
  four_pack_price : ℕ
  seven_pack_price : ℕ

/-- Calculates the maximum number of popsicles that can be bought with a given budget -/
def max_popsicles (options : PopsicleOptions) (budget : ℕ) : ℕ :=
  sorry

/-- The store's pricing options -/
def store_options : PopsicleOptions :=
  { single_price := 2
  , four_pack_price := 3
  , seven_pack_price := 5 }

/-- Zoe's budget -/
def zoe_budget : ℕ := 11

/-- Theorem: The maximum number of popsicles Zoe can buy with $11 is 14 -/
theorem max_popsicles_for_zoe :
  max_popsicles store_options zoe_budget = 14 := by
  sorry

end max_popsicles_for_zoe_l1394_139491


namespace johns_allowance_l1394_139417

theorem johns_allowance (A : ℝ) : 
  A > 0 →
  (4/15) * A = 0.88 →
  A = 3.30 := by
sorry

end johns_allowance_l1394_139417


namespace tan_sum_equals_double_tan_l1394_139453

theorem tan_sum_equals_double_tan (α β : Real) 
  (h : 3 * Real.sin β = Real.sin (2 * α + β)) : 
  Real.tan (α + β) = 2 * Real.tan α := by
  sorry

end tan_sum_equals_double_tan_l1394_139453


namespace delegates_without_badges_l1394_139482

theorem delegates_without_badges (total : Nat) (preprinted : Nat) : 
  total = 36 → preprinted = 16 → (total - preprinted - (total - preprinted) / 2) = 10 := by
  sorry

end delegates_without_badges_l1394_139482


namespace tomato_plants_l1394_139415

theorem tomato_plants (n : ℕ) (sum : ℕ) : 
  n = 12 → sum = 186 → 
  ∃ a d : ℕ, 
    (∀ i : ℕ, i ≤ n → a + (i - 1) * d = sum / n + (2 * i - n - 1) / 2) ∧
    (a + (n - 1) * d = 21) :=
by sorry

end tomato_plants_l1394_139415


namespace average_b_c_l1394_139455

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : c - a = 50) :
  (b + c) / 2 = 70 := by
  sorry

end average_b_c_l1394_139455


namespace parallelogram_base_length_l1394_139456

theorem parallelogram_base_length
  (area : ℝ)
  (base : ℝ)
  (altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base)
  (h3 : area = base * altitude) :
  base = 7 := by
sorry

end parallelogram_base_length_l1394_139456


namespace sin_theta_value_l1394_139412

theorem sin_theta_value (a : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real.sin θ = -Real.sqrt 2 / 2 := by
  sorry

end sin_theta_value_l1394_139412


namespace quadratic_equation_solution_l1394_139467

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := by sorry

end quadratic_equation_solution_l1394_139467


namespace hyperbola_theorem_l1394_139466

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧ 
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧ 
  A ≠ B

-- Theorem statement
theorem hyperbola_theorem 
  (center : ℝ × ℝ) 
  (right_focus : ℝ × ℝ) 
  (right_vertex : ℝ × ℝ) 
  (A B : ℝ × ℝ) :
  center = (0, 0) →
  right_focus = (2, 0) →
  right_vertex = (Real.sqrt 3, 0) →
  intersection_points A B →
  (∀ x y, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
sorry

end hyperbola_theorem_l1394_139466


namespace collinear_points_b_value_l1394_139414

theorem collinear_points_b_value :
  ∀ b : ℚ,
  (∃ (m c : ℚ), 
    (m * 4 + c = -6) ∧
    (m * (b + 3) + c = -1) ∧
    (m * (-3 * b + 4) + c = 5)) →
  b = 11 / 26 :=
by sorry

end collinear_points_b_value_l1394_139414


namespace fourth_root_equation_solutions_l1394_139462

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x^(1/4) = 18 / (9 - x^(1/4))) ↔ (x = 81 ∨ x = 1296) := by
  sorry

end fourth_root_equation_solutions_l1394_139462


namespace quadratic_roots_bound_l1394_139422

theorem quadratic_roots_bound (a b : ℝ) (α β : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + b = 0) →  -- Equation has real roots
  (α^2 + a*α + b = 0) →           -- α is a root
  (β^2 + a*β + b = 0) →           -- β is a root
  (α ≠ β) →                       -- Roots are distinct
  (2 * abs a < 4 + b ∧ abs b < 4 ↔ abs α < 2 ∧ abs β < 2) :=
by sorry

end quadratic_roots_bound_l1394_139422


namespace jones_elementary_population_l1394_139450

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ),
    boys_percentage = 30 / 100 →
    (boys_percentage * total_students : ℚ).num = 90 →
    total_students = 300 :=
by sorry

end jones_elementary_population_l1394_139450


namespace monday_temperature_l1394_139403

def sunday_temp : ℝ := 40
def tuesday_temp : ℝ := 65
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℕ := 7

theorem monday_temperature (monday_temp : ℝ) :
  (sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp →
  monday_temp = 50 := by
sorry

end monday_temperature_l1394_139403


namespace correct_savings_amount_l1394_139479

/-- Represents a bank with its interest calculation method -/
structure Bank where
  name : String
  calculateInterest : (principal : ℝ) → ℝ

/-- Calculates the amount needed to save given a bank's interest calculation -/
def amountToSave (initialFunds : ℝ) (totalExpenses : ℝ) (bank : Bank) : ℝ :=
  totalExpenses - initialFunds - bank.calculateInterest initialFunds

/-- Theorem stating the correct amount to save for each bank -/
theorem correct_savings_amount 
  (initialFunds : ℝ) 
  (totalExpenses : ℝ) 
  (bettaBank gammaBank omegaBank epsilonBank : Bank) 
  (h1 : initialFunds = 150000)
  (h2 : totalExpenses = 182200)
  (h3 : bettaBank.calculateInterest initialFunds = 2720.33)
  (h4 : gammaBank.calculateInterest initialFunds = 3375)
  (h5 : omegaBank.calculateInterest initialFunds = 2349.13)
  (h6 : epsilonBank.calculateInterest initialFunds = 2264.11) :
  (amountToSave initialFunds totalExpenses bettaBank = 29479.67) ∧
  (amountToSave initialFunds totalExpenses gammaBank = 28825) ∧
  (amountToSave initialFunds totalExpenses omegaBank = 29850.87) ∧
  (amountToSave initialFunds totalExpenses epsilonBank = 29935.89) :=
by sorry


end correct_savings_amount_l1394_139479


namespace sculpture_cost_in_pesos_l1394_139448

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Mexican pesos -/
def usd_to_mxn : ℝ := 20

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating that the cost of the sculpture in Mexican pesos is 400 -/
theorem sculpture_cost_in_pesos :
  (sculpture_cost_nad / usd_to_nad) * usd_to_mxn = 400 := by
  sorry

end sculpture_cost_in_pesos_l1394_139448


namespace rectangle_length_l1394_139476

/-- The perimeter of a rectangle -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangle with perimeter 1200 and width 500, its length is 100 -/
theorem rectangle_length (p w : ℝ) (h1 : p = 1200) (h2 : w = 500) :
  ∃ l : ℝ, perimeter l w = p ∧ l = 100 := by
  sorry

end rectangle_length_l1394_139476


namespace iron_content_calculation_l1394_139472

theorem iron_content_calculation (initial_mass : ℝ) (impurities_mass : ℝ) 
  (impurities_iron_percent : ℝ) (iron_content_increase : ℝ) :
  initial_mass = 500 →
  impurities_mass = 200 →
  impurities_iron_percent = 12.5 →
  iron_content_increase = 20 →
  ∃ (remaining_iron : ℝ),
    remaining_iron = 187.5 ∧
    remaining_iron = 
      (initial_mass * ((impurities_mass * impurities_iron_percent / 100) / 
      (initial_mass - impurities_mass) + iron_content_increase / 100) / 100) * 
      (initial_mass - impurities_mass) -
      (impurities_mass * impurities_iron_percent / 100) := by
  sorry

end iron_content_calculation_l1394_139472


namespace num_sequences_eq_248832_l1394_139452

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of class sessions per week -/
def sessions_per_week : ℕ := 5

/-- The number of different sequences of students solving problems in one week -/
def num_sequences : ℕ := num_students ^ sessions_per_week

/-- Theorem stating that the number of different sequences of students solving problems in one week is 248,832 -/
theorem num_sequences_eq_248832 : num_sequences = 248832 := by sorry

end num_sequences_eq_248832_l1394_139452


namespace fraction_ratio_equality_l1394_139497

theorem fraction_ratio_equality : ∃ x : ℚ, (5 / 34) / (7 / 48) = x / (1 / 13) ∧ x = 5 / 64 := by
  sorry

end fraction_ratio_equality_l1394_139497


namespace jacob_dinner_calories_l1394_139451

/-- Calculates Jacob's dinner calories based on his daily goal, breakfast, lunch, and excess calories --/
theorem jacob_dinner_calories
  (daily_goal : ℕ)
  (breakfast : ℕ)
  (lunch : ℕ)
  (excess : ℕ)
  (h1 : daily_goal = 1800)
  (h2 : breakfast = 400)
  (h3 : lunch = 900)
  (h4 : excess = 600) :
  daily_goal + excess - (breakfast + lunch) = 1100 :=
by sorry

end jacob_dinner_calories_l1394_139451


namespace power_36_equals_power_16_9_l1394_139426

theorem power_36_equals_power_16_9 (m n : ℤ) : 
  (36 : ℝ) ^ (m + n) = (16 : ℝ) ^ (m * n) * (9 : ℝ) ^ (m * n) := by
  sorry

end power_36_equals_power_16_9_l1394_139426


namespace trig_expression_simplification_l1394_139487

theorem trig_expression_simplification (α : ℝ) : 
  (Real.tan (2 * π + α)) / (Real.tan (α + π) - Real.cos (-α) + Real.sin (π / 2 - α)) = 1 := by
  sorry

end trig_expression_simplification_l1394_139487


namespace race_result_l1394_139471

-- Define the type for athlete positions
inductive Position
| First
| Second
| Third
| Fourth

-- Define a function to represent the statements of athletes
def athleteStatement (pos : Position) : Prop :=
  match pos with
  | Position.First => pos = Position.First
  | Position.Second => pos ≠ Position.First
  | Position.Third => pos = Position.First
  | Position.Fourth => pos = Position.Fourth

-- Define the theorem
theorem race_result :
  ∃ (p₁ p₂ p₃ p₄ : Position),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    (athleteStatement p₁ ∧ athleteStatement p₂ ∧ athleteStatement p₃ ∧ ¬athleteStatement p₄) ∧
    p₃ = Position.First :=
by sorry


end race_result_l1394_139471


namespace share_investment_interest_rate_l1394_139488

/-- Calculates the interest rate for a share investment -/
theorem share_investment_interest_rate 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : face_value = 52) 
  (h2 : dividend_rate = 0.09) 
  (h3 : market_value = 39) : 
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

#check share_investment_interest_rate

end share_investment_interest_rate_l1394_139488


namespace tangent_line_to_circle_l1394_139439

theorem tangent_line_to_circle (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0) →  -- circle equation
  (x = 1 ∧ y = Real.sqrt 3) →  -- point of tangency
  (x - Real.sqrt 3 * y + 2 = 0)  -- equation of tangent line
:= by sorry

end tangent_line_to_circle_l1394_139439


namespace river_depth_l1394_139427

/-- Given a river with specified width, flow rate, and volume flow rate, calculate its depth. -/
theorem river_depth (width : ℝ) (flow_rate_kmph : ℝ) (volume_flow_rate : ℝ) :
  width = 75 →
  flow_rate_kmph = 4 →
  volume_flow_rate = 35000 →
  (volume_flow_rate / (flow_rate_kmph * 1000 / 60) / width) = 7 := by
  sorry

#check river_depth

end river_depth_l1394_139427


namespace max_expression_proof_l1394_139421

/-- The maximum value of c * a^b - d given the constraints --/
def max_expression : ℕ := 625

/-- The set of possible values for a, b, c, and d --/
def value_set : Finset ℕ := {0, 1, 4, 5}

/-- Proposition: The maximum value of c * a^b - d is 625, given the constraints --/
theorem max_expression_proof :
  ∀ a b c d : ℕ,
    a ∈ value_set → b ∈ value_set → c ∈ value_set → d ∈ value_set →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    c * a^b - d ≤ max_expression :=
by sorry

end max_expression_proof_l1394_139421


namespace ampersand_composition_l1394_139405

-- Define the operations
def ampersand (y : ℤ) : ℤ := 2 * (7 - y)
def ampersandbar (y : ℤ) : ℤ := 2 * (y - 7)

-- State the theorem
theorem ampersand_composition : ampersandbar (ampersand (-13)) = 66 := by
  sorry

end ampersand_composition_l1394_139405


namespace pythagorean_triple_properties_l1394_139418

/-- Given a Pythagorean triple (a, b, c) where c is the hypotenuse,
    prove that certain expressions are perfect squares and
    that certain equations are solvable in integers. -/
theorem pythagorean_triple_properties (a b c : ℤ) 
  (h : a^2 + b^2 = c^2) : -- Pythagorean triple condition
  (∃ (k₁ k₂ k₃ k₄ : ℤ), 
    2*(c-a)*(c-b) = k₁^2 ∧ 
    2*(c-a)*(c+b) = k₂^2 ∧ 
    2*(c+a)*(c-b) = k₃^2 ∧ 
    2*(c+a)*(c+b) = k₄^2) ∧ 
  (∃ (x y : ℤ), 
    x + y + (2*x*y).sqrt = c ∧ 
    x + y - (2*x*y).sqrt = c) :=
by sorry

end pythagorean_triple_properties_l1394_139418


namespace paper_length_proof_l1394_139486

/-- Given a rectangular sheet of paper with specific dimensions and margins,
    prove that the length of the sheet is 10 inches. -/
theorem paper_length_proof (paper_width : Real) (margin : Real) (picture_area : Real) :
  paper_width = 8.5 →
  margin = 1.5 →
  picture_area = 38.5 →
  ∃ (paper_length : Real),
    paper_length = 10 ∧
    picture_area = (paper_length - 2 * margin) * (paper_width - 2 * margin) :=
by sorry

end paper_length_proof_l1394_139486


namespace adjacent_numbers_selection_l1394_139477

theorem adjacent_numbers_selection (n : ℕ) (k : ℕ) : 
  n = 49 → k = 6 → 
  (Nat.choose n k) - (Nat.choose (n - k + 1) k) = 
  (Nat.choose n k) - (Nat.choose 44 k) := by
  sorry

end adjacent_numbers_selection_l1394_139477


namespace smallest_y_with_24_factors_l1394_139481

theorem smallest_y_with_24_factors (y : ℕ) 
  (h1 : (Nat.divisors y).card = 24)
  (h2 : 20 ∣ y)
  (h3 : 35 ∣ y) :
  y ≥ 1120 ∧ ∃ (z : ℕ), z ≥ 1120 ∧ (Nat.divisors z).card = 24 ∧ 20 ∣ z ∧ 35 ∣ z :=
by sorry

end smallest_y_with_24_factors_l1394_139481


namespace complex_equation_solution_l1394_139438

theorem complex_equation_solution (z : ℂ) : 
  (1 : ℂ) + Complex.I * Real.sqrt 3 = z * ((1 : ℂ) - Complex.I * Real.sqrt 3) →
  z = -(1/2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
sorry

end complex_equation_solution_l1394_139438


namespace onion_harvest_scientific_notation_l1394_139464

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem onion_harvest_scientific_notation :
  toScientificNotation 325000000 = ScientificNotation.mk 3.25 8 (by sorry) :=
sorry

end onion_harvest_scientific_notation_l1394_139464


namespace final_price_percentage_l1394_139446

/-- The price change over 5 years -/
def price_change (p : ℝ) : ℝ :=
  p * 1.3 * 0.8 * 1.25 * 0.9 * 1.15

/-- Theorem stating the final price is 134.55% of the original price -/
theorem final_price_percentage (p : ℝ) (hp : p > 0) :
  price_change p / p = 1.3455 := by
  sorry

end final_price_percentage_l1394_139446


namespace original_group_size_l1394_139406

theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 8 ∧ absent_men = 3 ∧ final_days = 10 →
  ∃ original_size : ℕ, 
    original_size > absent_men ∧
    (original_size : ℚ) / initial_days = (original_size - absent_men) / final_days ∧
    original_size = 15 := by
  sorry

end original_group_size_l1394_139406


namespace brothers_age_difference_l1394_139409

/-- The age difference between two brothers -/
def age_difference (mark_age john_age : ℕ) : ℕ :=
  mark_age - john_age

theorem brothers_age_difference :
  ∀ (mark_age john_age parents_age : ℕ),
    mark_age = 18 →
    parents_age = 5 * john_age →
    parents_age = 40 →
    age_difference mark_age john_age = 10 := by
  sorry

end brothers_age_difference_l1394_139409


namespace bisector_sum_squares_l1394_139432

/-- Given a triangle with side lengths a and b, angle C, and its angle bisector l and
    exterior angle bisector l', the sum of squares of these bisectors is equal to
    (64 R^2 S^2) / ((a^2 - b^2)^2), where R is the circumradius and S is the area of the triangle. -/
theorem bisector_sum_squares (a b l l' R S : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 0 < l) (hl' : 0 < l') (hR : 0 < R) (hS : 0 < S) :
  l'^2 + l^2 = (64 * R^2 * S^2) / ((a^2 - b^2)^2) := by
  sorry

end bisector_sum_squares_l1394_139432


namespace first_car_distance_l1394_139447

theorem first_car_distance (total_distance : ℝ) (second_car_distance : ℝ) (side_distance : ℝ) (final_distance : ℝ) 
  (h1 : total_distance = 113)
  (h2 : second_car_distance = 35)
  (h3 : side_distance = 15)
  (h4 : final_distance = 28) :
  ∃ x : ℝ, x = 17.5 ∧ total_distance - (2 * x + side_distance + second_car_distance) = final_distance :=
by
  sorry


end first_car_distance_l1394_139447


namespace stewart_farm_horse_food_l1394_139461

/-- Calculates the total amount of horse food needed per day on a farm -/
def total_horse_food (num_sheep : ℕ) (sheep_ratio horse_ratio : ℕ) (food_per_horse : ℕ) : ℕ :=
  let num_horses := (num_sheep * horse_ratio) / sheep_ratio
  num_horses * food_per_horse

/-- Theorem: The Stewart farm needs 12,880 ounces of horse food per day -/
theorem stewart_farm_horse_food :
  total_horse_food 40 5 7 230 = 12880 := by
  sorry

end stewart_farm_horse_food_l1394_139461


namespace dark_tiles_fraction_l1394_139480

/-- Represents a 4x4 block of tiles on the floor -/
structure Block where
  size : Nat
  dark_tiles : Nat

/-- Represents the entire tiled floor -/
structure Floor where
  block : Block

/-- The fraction of dark tiles in the floor -/
def dark_fraction (f : Floor) : Rat :=
  f.block.dark_tiles / (f.block.size * f.block.size)

theorem dark_tiles_fraction (f : Floor) 
  (h1 : f.block.size = 4)
  (h2 : f.block.dark_tiles = 12) : 
  dark_fraction f = 3/4 := by
  sorry

end dark_tiles_fraction_l1394_139480


namespace cubic_polynomial_c_value_l1394_139460

theorem cubic_polynomial_c_value 
  (g : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, g x = x^3 + a*x^2 + b*x + c) 
  (h2 : ∃ r₁ r₂ r₃ : ℕ, (∀ i, Odd (r₁ + 2*i) ∧ g (r₁ + 2*i) = 0) ∧ 
                        (r₁ < r₂) ∧ (r₂ < r₃) ∧ 
                        g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0)
  (h3 : a + b + c = -11) :
  c = -15 := by
  sorry

end cubic_polynomial_c_value_l1394_139460


namespace probability_x_equals_y_l1394_139420

-- Define the range for x and y
def valid_range (x : ℝ) : Prop := -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi

-- Define the condition for x and y
def condition (x y : ℝ) : Prop := Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Define the total number of valid pairs
def total_pairs : ℕ := 121

-- Define the number of pairs where X = Y
def equal_pairs : ℕ := 11

-- State the theorem
theorem probability_x_equals_y :
  (∀ x y : ℝ, valid_range x → valid_range y → condition x y) →
  (equal_pairs : ℕ) / (total_pairs : ℕ) = 1 / 11 :=
sorry

end probability_x_equals_y_l1394_139420


namespace max_correct_answers_l1394_139443

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 60 → 
  correct_points = 5 → 
  blank_points = 0 → 
  incorrect_points = -2 → 
  total_score = 150 → 
  (∃ (correct blank incorrect : ℕ), 
    correct + blank + incorrect = total_questions ∧ 
    correct_points * correct + blank_points * blank + incorrect_points * incorrect = total_score ∧ 
    ∀ (other_correct : ℕ), 
      (∃ (other_blank other_incorrect : ℕ), 
        other_correct + other_blank + other_incorrect = total_questions ∧ 
        correct_points * other_correct + blank_points * other_blank + incorrect_points * other_incorrect = total_score) → 
      other_correct ≤ 38) ∧ 
  (∃ (blank incorrect : ℕ), 
    38 + blank + incorrect = total_questions ∧ 
    correct_points * 38 + blank_points * blank + incorrect_points * incorrect = total_score) :=
by sorry

end max_correct_answers_l1394_139443


namespace lower_variance_implies_more_stable_l1394_139425

/-- Represents a participant in the math competition -/
structure Participant where
  name : String
  average_score : ℝ
  variance : ℝ

/-- Defines what it means for a participant to have more stable performance -/
def has_more_stable_performance (p1 p2 : Participant) : Prop :=
  p1.average_score = p2.average_score ∧ p1.variance < p2.variance

/-- Theorem stating that the participant with lower variance has more stable performance -/
theorem lower_variance_implies_more_stable
  (xiao_li xiao_zhang : Participant)
  (h1 : xiao_li.name = "Xiao Li")
  (h2 : xiao_zhang.name = "Xiao Zhang")
  (h3 : xiao_li.average_score = 95)
  (h4 : xiao_zhang.average_score = 95)
  (h5 : xiao_li.variance = 0.55)
  (h6 : xiao_zhang.variance = 1.35) :
  has_more_stable_performance xiao_li xiao_zhang :=
sorry

end lower_variance_implies_more_stable_l1394_139425


namespace triangle_theorem_l1394_139434

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b)
  (h2 : Real.cos t.B = 1/4)
  (h3 : t.a + t.b + t.c = 5) :
  Real.sin t.C / Real.sin t.A = 2 ∧ t.b = 2 := by
  sorry

end triangle_theorem_l1394_139434


namespace coronavirus_state_after_three_days_l1394_139430

/-- Represents the state of Coronavirus cases on a given day -/
structure CoronavirusState where
  positiveCase : ℕ
  hospitalizedCase : ℕ
  deaths : ℕ

/-- Calculates the next day's Coronavirus state based on the current state and rates -/
def nextDayState (state : CoronavirusState) (newCaseRate : ℝ) (recoveryRate : ℝ) 
                 (hospitalizationRate : ℝ) (hospitalizationIncreaseRate : ℝ)
                 (deathRate : ℝ) (deathIncreaseRate : ℝ) : CoronavirusState :=
  sorry

/-- Theorem stating the Coronavirus state after 3 days given initial conditions -/
theorem coronavirus_state_after_three_days 
  (initialState : CoronavirusState)
  (newCaseRate : ℝ)
  (recoveryRate : ℝ)
  (hospitalizationRate : ℝ)
  (hospitalizationIncreaseRate : ℝ)
  (deathRate : ℝ)
  (deathIncreaseRate : ℝ)
  (h1 : initialState.positiveCase = 2000)
  (h2 : newCaseRate = 0.15)
  (h3 : recoveryRate = 0.05)
  (h4 : hospitalizationRate = 0.03)
  (h5 : hospitalizationIncreaseRate = 0.10)
  (h6 : deathRate = 0.02)
  (h7 : deathIncreaseRate = 0.05) :
  let day1 := nextDayState initialState newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  let day2 := nextDayState day1 newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  let day3 := nextDayState day2 newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  day3.positiveCase = 2420 ∧ day3.hospitalizedCase = 92 ∧ day3.deaths = 57 :=
sorry

end coronavirus_state_after_three_days_l1394_139430


namespace number_of_men_l1394_139489

theorem number_of_men (M W B : ℕ) (Ww Wb : ℚ) : 
  M * 6 = W * Ww ∧ 
  W * Ww = 7 * B * Wb ∧ 
  M * 6 + W * Ww + B * Wb = 90 →
  M = 5 := by
sorry

end number_of_men_l1394_139489


namespace system_solution_correct_l1394_139435

theorem system_solution_correct (x y : ℚ) :
  x = 2 ∧ y = 1/2 → (x - 2*y = 1 ∧ 2*x + 2*y = 5) := by
  sorry

end system_solution_correct_l1394_139435


namespace differential_of_y_l1394_139429

noncomputable def y (x : ℝ) : ℝ := Real.arctan (Real.sinh x) + (Real.sinh x) * Real.log (Real.cosh x)

theorem differential_of_y (x : ℝ) :
  deriv y x = Real.cosh x * (1 + Real.log (Real.cosh x)) :=
by sorry

end differential_of_y_l1394_139429


namespace fifth_month_sale_is_6500_l1394_139495

/-- Calculates the sale in the fifth month given the sales for other months and the average -/
def fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) : ℕ :=
  6 * avg - (m1 + m2 + m3 + m4 + m6)

/-- Theorem stating that the sale in the fifth month is 6500 -/
theorem fifth_month_sale_is_6500 :
  fifth_month_sale 6400 7000 6800 7200 5100 6500 = 6500 := by
  sorry

end fifth_month_sale_is_6500_l1394_139495


namespace dannys_physics_marks_l1394_139416

/-- Danny's marks in different subjects and average -/
structure DannyMarks where
  english : ℕ
  mathematics : ℕ
  chemistry : ℕ
  biology : ℕ
  average : ℕ

/-- The theorem stating Danny's marks in Physics -/
theorem dannys_physics_marks (marks : DannyMarks) 
  (h1 : marks.english = 76)
  (h2 : marks.mathematics = 65)
  (h3 : marks.chemistry = 67)
  (h4 : marks.biology = 75)
  (h5 : marks.average = 73)
  (h6 : (marks.english + marks.mathematics + marks.chemistry + marks.biology + marks.average * 5 - (marks.english + marks.mathematics + marks.chemistry + marks.biology)) / 5 = marks.average) :
  marks.average * 5 - (marks.english + marks.mathematics + marks.chemistry + marks.biology) = 82 := by
  sorry


end dannys_physics_marks_l1394_139416


namespace worksheets_to_memorize_l1394_139493

/-- Calculate the number of worksheets that can be memorized given study conditions --/
theorem worksheets_to_memorize (
  chapters : ℕ)
  (hours_per_chapter : ℝ)
  (hours_per_worksheet : ℝ)
  (max_hours_per_day : ℝ)
  (break_duration : ℝ)
  (breaks_per_day : ℕ)
  (snack_breaks : ℕ)
  (snack_break_duration : ℝ)
  (lunch_duration : ℝ)
  (study_days : ℕ)
  (h1 : chapters = 2)
  (h2 : hours_per_chapter = 3)
  (h3 : hours_per_worksheet = 1.5)
  (h4 : max_hours_per_day = 4)
  (h5 : break_duration = 1/6)  -- 10 minutes in hours
  (h6 : breaks_per_day = 4)
  (h7 : snack_breaks = 3)
  (h8 : snack_break_duration = 1/6)  -- 10 minutes in hours
  (h9 : lunch_duration = 0.5)  -- 30 minutes in hours
  (h10 : study_days = 4) :
  ⌊(study_days * (max_hours_per_day - (breaks_per_day * break_duration + snack_breaks * snack_break_duration + lunch_duration)) - chapters * hours_per_chapter) / hours_per_worksheet⌋ = 2 :=
by sorry

end worksheets_to_memorize_l1394_139493


namespace quadratic_inequality_solution_set_l1394_139457

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 + (2 - a) * x - 2 < 0}
  (a = 0 → S = {x : ℝ | x < 1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | x < 1 ∨ x > -2/a}) ∧
  (a = -2 → S = {x : ℝ | x ≠ 1}) ∧
  (a < -2 → S = {x : ℝ | x < -2/a ∨ x > 1}) ∧
  (a > 0 → S = {x : ℝ | -2/a < x ∧ x < 1}) :=
by sorry

end quadratic_inequality_solution_set_l1394_139457


namespace line_intersects_ellipse_l1394_139473

-- Define the line
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Theorem statement
theorem line_intersects_ellipse :
  ∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end line_intersects_ellipse_l1394_139473


namespace min_value_of_squares_l1394_139490

theorem min_value_of_squares (t : ℝ) :
  ∃ (a b : ℝ), 2 * a + 3 * b = t ∧
  ∀ (x y : ℝ), 2 * x + 3 * y = t → a^2 + b^2 ≤ x^2 + y^2 ∧
  a^2 + b^2 = (13 * t^2) / 169 := by
sorry

end min_value_of_squares_l1394_139490


namespace dinosaur_book_cost_l1394_139484

def dictionary_cost : ℕ := 5
def cookbook_cost : ℕ := 5
def saved_amount : ℕ := 19
def additional_needed : ℕ := 2

theorem dinosaur_book_cost :
  dictionary_cost + cookbook_cost + (saved_amount + additional_needed - (dictionary_cost + cookbook_cost)) = 11 := by
  sorry

end dinosaur_book_cost_l1394_139484


namespace investment_division_l1394_139478

/-- 
Given a total amount of 3200 divided into two parts, where one part is invested at 3% 
and the other at 5%, and the total annual interest is 144, prove that the amount 
of the first part (invested at 3%) is 800.
-/
theorem investment_division (x : ℝ) : 
  x ≥ 0 ∧ 
  3200 - x ≥ 0 ∧ 
  0.03 * x + 0.05 * (3200 - x) = 144 → 
  x = 800 := by
  sorry

#check investment_division

end investment_division_l1394_139478


namespace sum_of_squares_zero_implies_sum_nine_l1394_139408

theorem sum_of_squares_zero_implies_sum_nine (a b c : ℝ) 
  (h : 2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) : 
  a + b + c = 9 := by
  sorry

end sum_of_squares_zero_implies_sum_nine_l1394_139408


namespace sophie_wallet_problem_l1394_139468

theorem sophie_wallet_problem :
  ∃ (x y z : ℕ), 
    x + y + z = 60 ∧
    x + 2*y + 5*z = 175 ∧
    x = 5 := by
  sorry

end sophie_wallet_problem_l1394_139468


namespace cube_root_seven_to_sixth_l1394_139474

theorem cube_root_seven_to_sixth (x : ℝ) (h : x = 7^(1/3)) : x^6 = 49 := by
  sorry

end cube_root_seven_to_sixth_l1394_139474


namespace marta_took_ten_books_l1394_139445

/-- The number of books Marta took off the shelf -/
def books_taken (initial_books : ℝ) (remaining_books : ℕ) : ℝ :=
  initial_books - remaining_books

/-- Theorem stating that Marta took 10 books off the shelf -/
theorem marta_took_ten_books : books_taken 38.0 28 = 10 := by
  sorry

end marta_took_ten_books_l1394_139445


namespace equation_solutions_l1394_139433

theorem equation_solutions :
  (∃ x : ℝ, (2 / (x - 2) = 3 / x) ∧ (x = 6)) ∧
  (∃ x : ℝ, (4 / (x^2 - 1) = (x + 2) / (x - 1) - 1) ∧ (x = 1/3)) :=
by sorry

end equation_solutions_l1394_139433


namespace sin_cos_sum_equivalence_l1394_139441

theorem sin_cos_sum_equivalence (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry

end sin_cos_sum_equivalence_l1394_139441


namespace fred_newspaper_earnings_l1394_139410

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings (total_earnings washing_earnings : ℕ) : ℕ :=
  total_earnings - washing_earnings

theorem fred_newspaper_earnings :
  newspaper_earnings 90 74 = 16 := by
  sorry

end fred_newspaper_earnings_l1394_139410


namespace circumference_difference_of_concentric_circles_l1394_139431

/-- Given two concentric circles with the specified properties, 
    prove the difference in their circumferences --/
theorem circumference_difference_of_concentric_circles 
  (r_inner : ℝ) (r_outer : ℝ) (h1 : r_outer = r_inner + 15) 
  (h2 : 2 * r_inner = 50) : 
  2 * π * r_outer - 2 * π * r_inner = 30 * π := by
  sorry

end circumference_difference_of_concentric_circles_l1394_139431


namespace max_x_minus_y_l1394_139454

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l1394_139454


namespace puzzle_solution_l1394_139483

theorem puzzle_solution (D E F : ℕ) 
  (h1 : D + E + F = 16)
  (h2 : F + D + 1 = 16)
  (h3 : E - 1 = D)
  (h4 : D ≠ E ∧ D ≠ F ∧ E ≠ F)
  (h5 : D < 10 ∧ E < 10 ∧ F < 10) : E = 1 := by
  sorry

#check puzzle_solution

end puzzle_solution_l1394_139483


namespace circles_intersect_iff_distance_between_radii_sum_and_diff_l1394_139400

/-- Two circles intersect if and only if the distance between their centers
    is greater than the absolute difference of their radii and less than
    the sum of their radii. -/
theorem circles_intersect_iff_distance_between_radii_sum_and_diff
  (R r d : ℝ) (h : R ≥ r) :
  (∃ (p : ℝ × ℝ), (p.1 - 0)^2 + (p.2 - 0)^2 = R^2 ∧ 
                  (p.1 - d)^2 + p.2^2 = r^2) ↔
  (R - r < d ∧ d < R + r) :=
sorry

end circles_intersect_iff_distance_between_radii_sum_and_diff_l1394_139400


namespace certain_number_equation_l1394_139449

theorem certain_number_equation (x : ℝ) : 15 * x + 16 * x + 19 * x + 11 = 161 ↔ x = 3 := by
  sorry

end certain_number_equation_l1394_139449


namespace parallel_condition_l1394_139496

/-- Two lines in the form of ax + by + c = 0 and dx + ey + f = 0 are parallel if and only if ae = bd -/
def are_parallel (a b c d e f : ℝ) : Prop := a * e = b * d

/-- The first line: ax + 2y - 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0

/-- The second line: x + (a + 1)y + 4 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (a = -2 → are_parallel a 2 (-1) 1 (a + 1) 4) ∧
  (∃ b : ℝ, b ≠ -2 ∧ are_parallel b 2 (-1) 1 (b + 1) 4) :=
sorry

end parallel_condition_l1394_139496


namespace solution_characterization_l1394_139470

/-- The set of all solutions to the equation ab + bc + ca = 2(a + b + c) in natural numbers -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)}

/-- The equation ab + bc + ca = 2(a + b + c) -/
def SatisfiesEquation (t : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t
  a * b + b * c + c * a = 2 * (a + b + c)

theorem solution_characterization :
  ∀ t : ℕ × ℕ × ℕ, t ∈ SolutionSet ↔ SatisfiesEquation t :=
sorry

end solution_characterization_l1394_139470


namespace angle_convergence_point_l1394_139428

theorem angle_convergence_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by sorry

end angle_convergence_point_l1394_139428


namespace cats_adopted_l1394_139463

/-- Proves the number of cats adopted given the shelter's cat population changes -/
theorem cats_adopted (initial_cats : ℕ) (new_cats : ℕ) (kittens_born : ℕ) (cat_picked_up : ℕ) (final_cats : ℕ) :
  initial_cats = 6 →
  new_cats = 12 →
  kittens_born = 5 →
  cat_picked_up = 1 →
  final_cats = 19 →
  initial_cats + new_cats - (initial_cats + new_cats + kittens_born - cat_picked_up - final_cats) = 3 :=
by sorry

end cats_adopted_l1394_139463


namespace complex_magnitude_problem_l1394_139485

theorem complex_magnitude_problem (x y : ℝ) (h : (5 : ℂ) - x * I = y + 1 - 3 * I) : 
  Complex.abs (x - y * I) = 5 := by
sorry

end complex_magnitude_problem_l1394_139485


namespace inscribed_octagon_area_inscribed_octagon_area_is_1400_l1394_139411

/-- The area of an inscribed octagon in a square -/
theorem inscribed_octagon_area (square_perimeter : ℝ) (h1 : square_perimeter = 160) : ℝ :=
  let square_side := square_perimeter / 4
  let triangle_leg := square_side / 4
  let triangle_area := (1 / 2) * triangle_leg * triangle_leg
  let total_triangle_area := 4 * triangle_area
  let square_area := square_side * square_side
  square_area - total_triangle_area

/-- The area of the inscribed octagon is 1400 square centimeters -/
theorem inscribed_octagon_area_is_1400 (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  inscribed_octagon_area square_perimeter h1 = 1400 := by
  sorry

end inscribed_octagon_area_inscribed_octagon_area_is_1400_l1394_139411


namespace polygon_sides_l1394_139424

theorem polygon_sides (n : ℕ) (x : ℝ) : 
  n ≥ 3 →
  0 < x →
  x < 180 →
  (n - 2) * 180 - x + (180 - x) = 500 →
  n = 4 ∨ n = 5 :=
sorry

end polygon_sides_l1394_139424


namespace intersection_P_Q_l1394_139469

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x * (x - 1) ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = {x : ℝ | x > 1} := by sorry

end intersection_P_Q_l1394_139469


namespace sam_distance_l1394_139465

/-- The distance traveled by Sam given his walking speed and duration -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that Sam's traveled distance is 8 miles -/
theorem sam_distance :
  let speed := 4 -- miles per hour
  let time := 2 -- hours
  distance_traveled speed time = 8 := by sorry

end sam_distance_l1394_139465


namespace cos_150_degrees_l1394_139459

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l1394_139459


namespace max_a_for_four_near_zero_points_l1394_139440

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- Definition of a "near-zero point" -/
def is_near_zero_point (f : ℝ → ℝ) (x : ℤ) : Prop :=
  |f x| ≤ 1/4

theorem max_a_for_four_near_zero_points (a b c : ℝ) (ha : a > 0) :
  (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    is_near_zero_point (quadratic_function a b c) x₁ ∧
    is_near_zero_point (quadratic_function a b c) x₂ ∧
    is_near_zero_point (quadratic_function a b c) x₃ ∧
    is_near_zero_point (quadratic_function a b c) x₄) →
  a ≤ 1/4 :=
sorry

end max_a_for_four_near_zero_points_l1394_139440


namespace undefined_at_eleven_l1394_139423

theorem undefined_at_eleven (x : ℝ) : 
  (∃ y, (3 * x^2 + 5) / (x^2 - 22*x + 121) = y) ↔ x ≠ 11 :=
by sorry

end undefined_at_eleven_l1394_139423


namespace marble_picking_ways_l1394_139401

/-- The number of ways to pick up at least one marble from a set of marbles -/
def pick_marbles_ways (red green yellow black pink : ℕ) : ℕ :=
  ((red + 1) * (green + 1) * (yellow + 1) * (black + 1) * (pink + 1)) - 1

/-- Theorem: There are 95 ways to pick up at least one marble from a set
    containing 3 red marbles, 2 green marbles, and one each of yellow, black, and pink marbles -/
theorem marble_picking_ways :
  pick_marbles_ways 3 2 1 1 1 = 95 := by
  sorry

end marble_picking_ways_l1394_139401


namespace triangle_incircle_path_length_l1394_139402

theorem triangle_incircle_path_length (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sides : a = 6 ∧ b = 8 ∧ c = 10) : 
  let s := (a + b + c) / 2
  let r := (s * (s - a) * (s - b) * (s - c)).sqrt / s
  (a + b + c) - 2 * r = 12 := by
  sorry

end triangle_incircle_path_length_l1394_139402


namespace square_equation_solution_l1394_139419

theorem square_equation_solution : ∃! x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 := by sorry

end square_equation_solution_l1394_139419


namespace odd_function_value_l1394_139442

theorem odd_function_value (f : ℝ → ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x^2 + 1/x) →  -- definition of f for x > 0
  f (-1) = -2 := by sorry

end odd_function_value_l1394_139442


namespace equation_solutions_l1394_139407

theorem equation_solutions : 
  {(x, y) : ℕ × ℕ | x^2 + 6*x*y - 7*y^2 = 2009 ∧ x > 0 ∧ y > 0} = 
  {(252, 251), (42, 35), (42, 1)} := by sorry

end equation_solutions_l1394_139407


namespace sum_last_two_digits_modified_fibonacci_factorial_series_l1394_139492

def modifiedFibonacciFactorialSeries : List Nat := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

def lastTwoDigits (n : Nat) : Nat :=
  n % 100

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_last_two_digits_modified_fibonacci_factorial_series :
  (modifiedFibonacciFactorialSeries.map (λ x => lastTwoDigits (factorial x))).sum % 100 = 5 := by
  sorry

end sum_last_two_digits_modified_fibonacci_factorial_series_l1394_139492


namespace infinite_decimal_digits_l1394_139404

/-- The decimal representation of 1 / (2^3 * 5^4 * 3^2) has infinitely many digits after the decimal point. -/
theorem infinite_decimal_digits (n : ℕ) : ∃ (k : ℕ), k > n ∧ 
  (10^k * (1 : ℚ) / (2^3 * 5^4 * 3^2)).num ≠ 0 :=
sorry

end infinite_decimal_digits_l1394_139404


namespace base8_perfect_square_c_not_unique_l1394_139475

/-- Represents a number in base 8 of the form ab5c -/
structure Base8Number where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_valid : b < 8
  c_valid : c < 8

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.a + 64 * n.b + 40 + n.c

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem base8_perfect_square_c_not_unique :
  ∃ (n1 n2 : Base8Number),
    n1.a = n2.a ∧ n1.b = n2.b ∧ n1.c ≠ n2.c ∧
    isPerfectSquare (toDecimal n1) ∧
    isPerfectSquare (toDecimal n2) := by
  sorry

end base8_perfect_square_c_not_unique_l1394_139475


namespace intersection_lines_l1394_139498

-- Define the fixed points M₁ and M₂
def M₁ : ℝ × ℝ := (26, 1)
def M₂ : ℝ × ℝ := (2, 1)

-- Define the point P
def P : ℝ × ℝ := (-2, 3)

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (((x - M₁.1)^2 + (y - M₁.2)^2) / ((x - M₂.1)^2 + (y - M₂.2)^2)) = 25

-- Define the trajectory of M
def trajectory (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

-- Define the chord length condition
def chord_length (l : ℝ → ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
    y₁ = l x₁ ∧ y₂ = l x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 64

-- Theorem statement
theorem intersection_lines :
  ∀ (l : ℝ → ℝ),
    (∀ x, l x = -2 ∨ l x = (-5/12) * x + 23/6) ↔
    (∀ M, distance_ratio M → trajectory M.1 M.2) ∧
    chord_length l ∧
    l P.1 = P.2 :=
sorry

end intersection_lines_l1394_139498


namespace power_of_five_l1394_139444

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end power_of_five_l1394_139444


namespace lulu_ice_cream_expense_l1394_139437

theorem lulu_ice_cream_expense (initial_amount : ℝ) (ice_cream_cost : ℝ) (final_cash : ℝ) :
  initial_amount = 65 →
  final_cash = 24 →
  final_cash = (4/5) * (1/2) * (initial_amount - ice_cream_cost) →
  ice_cream_cost = 5 := by
  sorry

end lulu_ice_cream_expense_l1394_139437
