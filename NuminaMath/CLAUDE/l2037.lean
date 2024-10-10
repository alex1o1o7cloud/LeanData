import Mathlib

namespace C_power_100_l2037_203727

def C : Matrix (Fin 2) (Fin 2) ℝ := !![5, -1; 12, 3]

theorem C_power_100 : 
  C^100 = (3^99 : ℝ) • !![1, 100; 6000, -200] := by sorry

end C_power_100_l2037_203727


namespace mrs_wonderful_class_size_l2037_203713

theorem mrs_wonderful_class_size :
  ∀ (girls : ℕ) (boys : ℕ),
  boys = girls + 3 →
  girls * girls + boys * boys + 10 + 8 = 450 →
  girls + boys = 29 :=
by
  sorry

end mrs_wonderful_class_size_l2037_203713


namespace sandy_change_is_13_5_l2037_203748

/-- Represents the prices and quantities of drinks in Sandy's order -/
structure DrinkOrder where
  cappuccino_price : ℝ
  iced_tea_price : ℝ
  cafe_latte_price : ℝ
  espresso_price : ℝ
  mocha_price : ℝ
  hot_chocolate_price : ℝ
  cappuccino_qty : ℕ
  iced_tea_qty : ℕ
  cafe_latte_qty : ℕ
  espresso_qty : ℕ
  mocha_qty : ℕ
  hot_chocolate_qty : ℕ

/-- Calculates the total cost of the drink order -/
def total_cost (order : DrinkOrder) : ℝ :=
  order.cappuccino_price * order.cappuccino_qty +
  order.iced_tea_price * order.iced_tea_qty +
  order.cafe_latte_price * order.cafe_latte_qty +
  order.espresso_price * order.espresso_qty +
  order.mocha_price * order.mocha_qty +
  order.hot_chocolate_price * order.hot_chocolate_qty

/-- Calculates the change from a given payment amount -/
def calculate_change (payment : ℝ) (order : DrinkOrder) : ℝ :=
  payment - total_cost order

/-- Theorem stating that Sandy's change is $13.5 -/
theorem sandy_change_is_13_5 :
  let sandy_order : DrinkOrder := {
    cappuccino_price := 2,
    iced_tea_price := 3,
    cafe_latte_price := 1.5,
    espresso_price := 1,
    mocha_price := 2.5,
    hot_chocolate_price := 2,
    cappuccino_qty := 4,
    iced_tea_qty := 3,
    cafe_latte_qty := 5,
    espresso_qty := 3,
    mocha_qty := 2,
    hot_chocolate_qty := 2
  }
  calculate_change 50 sandy_order = 13.5 := by
  sorry

end sandy_change_is_13_5_l2037_203748


namespace system_solutions_l2037_203776

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  y = 2 * x^2 - 1 ∧ z = 2 * y^2 - 1 ∧ x = 2 * z^2 - 1

/-- The set of solutions to the system -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(1, 1, 1), (-1/2, -1/2, -1/2)} ∪
  {(Real.cos (2 * Real.pi / 9), Real.cos (4 * Real.pi / 9), -Real.cos (Real.pi / 9)),
   (Real.cos (4 * Real.pi / 9), -Real.cos (Real.pi / 9), Real.cos (2 * Real.pi / 9)),
   (-Real.cos (Real.pi / 9), Real.cos (2 * Real.pi / 9), Real.cos (4 * Real.pi / 9))} ∪
  {(Real.cos (2 * Real.pi / 7), -Real.cos (3 * Real.pi / 7), -Real.cos (Real.pi / 7)),
   (-Real.cos (3 * Real.pi / 7), -Real.cos (Real.pi / 7), Real.cos (2 * Real.pi / 7)),
   (-Real.cos (Real.pi / 7), Real.cos (2 * Real.pi / 7), -Real.cos (3 * Real.pi / 7))}

/-- Theorem stating that the solutions set contains all and only the solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℝ, (x, y, z) ∈ solutions ↔ system x y z :=
sorry

end system_solutions_l2037_203776


namespace octal_addition_1275_164_l2037_203725

/-- Converts an octal (base 8) number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Represents an octal number -/
structure OctalNumber where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 8

/-- Addition of two octal numbers -/
def octal_add (a b : OctalNumber) : OctalNumber :=
  ⟨ -- implementation details omitted
    sorry,
    sorry ⟩

theorem octal_addition_1275_164 :
  let a : OctalNumber := ⟨[1, 2, 7, 5], sorry⟩
  let b : OctalNumber := ⟨[1, 6, 4], sorry⟩
  let result : OctalNumber := octal_add a b
  result.digits = [1, 5, 0, 3] := by
  sorry

end octal_addition_1275_164_l2037_203725


namespace car_travel_distance_l2037_203708

def initial_distance : ℝ := 192
def initial_gallons : ℝ := 6
def efficiency_increase : ℝ := 0.1
def new_gallons : ℝ := 8

theorem car_travel_distance : 
  let initial_mpg := initial_distance / initial_gallons
  let new_mpg := initial_mpg * (1 + efficiency_increase)
  new_mpg * new_gallons = 281.6 := by
  sorry

end car_travel_distance_l2037_203708


namespace fifth_term_is_five_l2037_203707

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem fifth_term_is_five :
  fibonacci_like_sequence 4 = 5 := by
  sorry

end fifth_term_is_five_l2037_203707


namespace geometric_sequence_a6_l2037_203781

/-- A geometric sequence with a₂ = 4 and a₄ = 8 has a₆ = 16 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a2 : a 2 = 4) (h_a4 : a 4 = 8) : a 6 = 16 := by
  sorry

end geometric_sequence_a6_l2037_203781


namespace optimal_game_result_exact_distinct_rows_l2037_203737

/-- Represents the game board -/
def GameBoard := Fin (2^100) → Fin 100 → Bool

/-- Player A's strategy -/
def StrategyA := GameBoard → Fin 100 → Fin 100

/-- Player B's strategy -/
def StrategyB := GameBoard → Fin 100 → Fin 100

/-- Counts the number of distinct rows in a game board -/
def countDistinctRows (board : GameBoard) : ℕ := sorry

/-- Simulates the game with given strategies -/
def playGame (stratA : StrategyA) (stratB : StrategyB) : GameBoard := sorry

/-- The main theorem stating the result of the game -/
theorem optimal_game_result :
  ∀ (stratA : StrategyA) (stratB : StrategyB),
  ∃ (optimalA : StrategyA) (optimalB : StrategyB),
  countDistinctRows (playGame optimalA stratB) ≥ 2^50 ∧
  countDistinctRows (playGame stratA optimalB) ≤ 2^50 := by sorry

/-- The final theorem stating the exact number of distinct rows -/
theorem exact_distinct_rows :
  ∃ (optimalA : StrategyA) (optimalB : StrategyB),
  countDistinctRows (playGame optimalA optimalB) = 2^50 := by sorry

end optimal_game_result_exact_distinct_rows_l2037_203737


namespace min_difference_l2037_203706

/-- Represents a 4-digit positive integer ABCD -/
def FourDigitNum (a b c d : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d

/-- Represents a 2-digit positive integer -/
def TwoDigitNum (x y : Nat) : Nat :=
  10 * x + y

/-- The difference between a 4-digit number and the product of its two 2-digit parts -/
def Difference (a b c d : Nat) : Nat :=
  FourDigitNum a b c d - TwoDigitNum a b * TwoDigitNum c d

theorem min_difference :
  ∀ (a b c d : Nat),
    a ≠ 0 → c ≠ 0 →
    a < 10 → b < 10 → c < 10 → d < 10 →
    Difference a b c d ≥ 109 :=
by sorry

end min_difference_l2037_203706


namespace sum_of_intercepts_l2037_203736

-- Define the parabola
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 5

-- Define the x-intercept
def x_intercept : ℝ := parabola 0

-- Define the y-intercepts
def y_intercepts : Set ℝ := {y | parabola y = 0}

-- Theorem statement
theorem sum_of_intercepts :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧ x_intercept + b + c = 8 :=
sorry

end sum_of_intercepts_l2037_203736


namespace no_integer_solutions_l2037_203766

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≤ 9 ∧ d % 3 ≠ 0 ∧ d % 7 ≠ 0

def has_valid_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem no_integer_solutions (p : ℕ) (hp : Prime p) (hp_gt : p > 5) (hp_digits : has_valid_digits p) :
  ¬∃ (x y : ℤ), x^4 + p = 3 * y^4 :=
sorry

end no_integer_solutions_l2037_203766


namespace composite_has_at_least_three_factors_l2037_203715

/-- A positive integer is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The number of factors of a natural number. -/
def numFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- Theorem: A composite number has at least three factors. -/
theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
    3 ≤ numFactors n := by
  sorry

end composite_has_at_least_three_factors_l2037_203715


namespace inequality_range_l2037_203787

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end inequality_range_l2037_203787


namespace grammar_club_committee_probability_l2037_203735

/-- The number of boys in the Grammar club -/
def num_boys : ℕ := 15

/-- The number of girls in the Grammar club -/
def num_girls : ℕ := 15

/-- The size of the committee to be formed -/
def committee_size : ℕ := 5

/-- The minimum number of boys required in the committee -/
def min_boys : ℕ := 2

/-- The probability of forming a committee with at least 2 boys and at least 1 girl -/
def committee_probability : ℚ := 515 / 581

/-- Theorem stating the probability of forming a committee with the given conditions -/
theorem grammar_club_committee_probability :
  let total_members := num_boys + num_girls
  let valid_committees := (Finset.range (committee_size + 1)).filter (λ k => k ≥ min_boys ∧ k < committee_size)
    |>.sum (λ k => (Nat.choose num_boys k) * (Nat.choose num_girls (committee_size - k)))
  let total_committees := Nat.choose total_members committee_size
  (valid_committees : ℚ) / total_committees = committee_probability := by
  sorry

#check grammar_club_committee_probability

end grammar_club_committee_probability_l2037_203735


namespace factorize_expression_1_l2037_203732

theorem factorize_expression_1 (x y : ℝ) :
  x^2*y - 4*x*y + 4*y = y*(x-2)^2 := by sorry

end factorize_expression_1_l2037_203732


namespace overtime_pay_rate_ratio_l2037_203797

/-- Proves that the ratio of overtime pay rate to regular pay rate is 2:1 given specific conditions -/
theorem overtime_pay_rate_ratio (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) (overtime_hours : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 180 →
  overtime_hours = 10 →
  (total_pay - regular_rate * regular_hours) / overtime_hours / regular_rate = 2 := by
  sorry

end overtime_pay_rate_ratio_l2037_203797


namespace tangent_slope_at_2_10_l2037_203780

/-- The slope of the tangent line to y = x^2 + 3x at (2, 10) is 7 -/
theorem tangent_slope_at_2_10 : 
  let f (x : ℝ) := x^2 + 3*x
  let A : ℝ × ℝ := (2, 10)
  let slope := (deriv f) A.1
  slope = 7 := by sorry

end tangent_slope_at_2_10_l2037_203780


namespace fraction_comparison_l2037_203782

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_comparison_l2037_203782


namespace lcm_hcf_problem_l2037_203704

theorem lcm_hcf_problem (n : ℕ) : 
  Nat.lcm 8 n = 24 → Nat.gcd 8 n = 4 → n = 12 := by
  sorry

end lcm_hcf_problem_l2037_203704


namespace total_fuel_consumption_l2037_203724

-- Define fuel consumption rates
def highway_consumption_60 : ℝ := 3
def highway_consumption_70 : ℝ := 3.5
def city_consumption_30 : ℝ := 5
def city_consumption_15 : ℝ := 4.5

-- Define driving durations and speeds
def day1_highway_60 : ℝ := 2
def day1_highway_70 : ℝ := 1
def day1_city_30 : ℝ := 4

def day2_highway_70 : ℝ := 3
def day2_city_15 : ℝ := 3
def day2_city_30 : ℝ := 1

def day3_highway_60 : ℝ := 1.5
def day3_city_30 : ℝ := 3
def day3_city_15 : ℝ := 1

-- Theorem statement
theorem total_fuel_consumption :
  let day1 := day1_highway_60 * 60 * highway_consumption_60 +
              day1_highway_70 * 70 * highway_consumption_70 +
              day1_city_30 * 30 * city_consumption_30
  let day2 := day2_highway_70 * 70 * highway_consumption_70 +
              day2_city_15 * 15 * city_consumption_15 +
              day2_city_30 * 30 * city_consumption_30
  let day3 := day3_highway_60 * 60 * highway_consumption_60 +
              day3_city_30 * 30 * city_consumption_30 +
              day3_city_15 * 15 * city_consumption_15
  day1 + day2 + day3 = 3080 := by
  sorry

end total_fuel_consumption_l2037_203724


namespace base_conversion_four_digits_l2037_203757

theorem base_conversion_four_digits (b : ℕ) : b > 1 → (
  (256 < b^4) ∧ (b^3 ≤ 256) ↔ b = 5
) := by sorry

end base_conversion_four_digits_l2037_203757


namespace ratio_problem_l2037_203702

theorem ratio_problem (y : ℚ) : (1 : ℚ) / 3 = y / 5 → y = 5 / 3 := by
  sorry

end ratio_problem_l2037_203702


namespace negative_comparison_l2037_203786

theorem negative_comparison : -2023 > -2024 := by
  sorry

end negative_comparison_l2037_203786


namespace product_of_sums_equals_difference_of_powers_l2037_203739

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end product_of_sums_equals_difference_of_powers_l2037_203739


namespace volleyballs_count_l2037_203721

/-- The number of volleyballs in Reynald's purchase --/
def volleyballs : ℕ :=
  let total_balls : ℕ := 145
  let soccer_balls : ℕ := 20
  let basketballs : ℕ := soccer_balls + 5
  let tennis_balls : ℕ := 2 * soccer_balls
  let baseballs : ℕ := soccer_balls + 10
  total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

/-- Theorem stating that the number of volleyballs is 30 --/
theorem volleyballs_count : volleyballs = 30 := by
  sorry

end volleyballs_count_l2037_203721


namespace matthew_owns_26_cheap_shares_l2037_203733

/-- Calculates the number of shares of the less valuable stock Matthew owns --/
def calculate_less_valuable_shares (total_assets : ℕ) (expensive_share_price : ℕ) (expensive_shares : ℕ) : ℕ :=
  let cheap_share_price := expensive_share_price / 2
  let expensive_stock_value := expensive_share_price * expensive_shares
  let cheap_stock_value := total_assets - expensive_stock_value
  cheap_stock_value / cheap_share_price

/-- Proves that Matthew owns 26 shares of the less valuable stock --/
theorem matthew_owns_26_cheap_shares :
  calculate_less_valuable_shares 2106 78 14 = 26 := by
  sorry

end matthew_owns_26_cheap_shares_l2037_203733


namespace parent_current_age_l2037_203730

-- Define the son's age next year
def sons_age_next_year : ℕ := 8

-- Define the relation between parent's and son's age
def parent_age_relation (parent_age son_age : ℕ) : Prop :=
  parent_age = 5 * son_age

-- Theorem to prove
theorem parent_current_age : 
  ∃ (parent_age : ℕ), parent_age_relation parent_age (sons_age_next_year - 1) ∧ parent_age = 35 :=
by
  sorry

end parent_current_age_l2037_203730


namespace lauren_mail_count_l2037_203758

/-- The number of pieces of mail Lauren sent on Monday -/
def monday : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday : ℕ := monday + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday : ℕ := tuesday - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday : ℕ := wednesday + 15

/-- The total number of pieces of mail Lauren sent over four days -/
def total_mail : ℕ := monday + tuesday + wednesday + thursday

theorem lauren_mail_count : total_mail = 295 := by sorry

end lauren_mail_count_l2037_203758


namespace ferry_tourists_l2037_203701

/-- Calculates the total number of tourists transported by a ferry --/
def totalTourists (trips : ℕ) (initialTourists : ℕ) (decrease : ℕ) : ℕ :=
  trips * (2 * initialTourists - (trips - 1) * decrease) / 2

/-- Theorem: The ferry transports 904 tourists in total --/
theorem ferry_tourists : totalTourists 8 120 2 = 904 := by
  sorry

end ferry_tourists_l2037_203701


namespace trig_identity_l2037_203720

theorem trig_identity : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end trig_identity_l2037_203720


namespace fraction_simplification_l2037_203700

theorem fraction_simplification 
  (b c d x y z : ℝ) :
  (c * x * (b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3) + 
   d * z * (b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3)) / 
  (c * x + d * z) = 
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
by sorry

end fraction_simplification_l2037_203700


namespace school_home_time_ratio_l2037_203788

/-- Represents the road segments in Xiaoming's journey --/
inductive RoadSegment
| Flat
| Uphill
| Downhill

/-- Represents the direction of Xiaoming's journey --/
inductive Direction
| ToSchool
| ToHome

/-- Calculates the time taken for a segment of the journey --/
def segmentTime (segment : RoadSegment) (direction : Direction) : ℚ :=
  match segment, direction with
  | RoadSegment.Flat, _ => 1 / 3
  | RoadSegment.Uphill, Direction.ToSchool => 1
  | RoadSegment.Uphill, Direction.ToHome => 1
  | RoadSegment.Downhill, Direction.ToSchool => 1 / 4
  | RoadSegment.Downhill, Direction.ToHome => 1 / 2

/-- Calculates the total time for a journey in a given direction --/
def journeyTime (direction : Direction) : ℚ :=
  segmentTime RoadSegment.Flat direction +
  2 * segmentTime RoadSegment.Uphill direction +
  segmentTime RoadSegment.Downhill direction

/-- Main theorem: The ratio of time to school vs time to home is 19:16 --/
theorem school_home_time_ratio :
  (journeyTime Direction.ToSchool) / (journeyTime Direction.ToHome) = 19 / 16 := by
  sorry


end school_home_time_ratio_l2037_203788


namespace existence_of_divisible_power_sum_l2037_203711

theorem existence_of_divisible_power_sum (p : Nat) (h_prime : Prime p) (h_p_gt_10 : p > 10) :
  ∃ m n : Nat, m > 0 ∧ n > 0 ∧ m + n < p ∧ (p ∣ (5^m * 7^n - 1)) := by
  sorry

end existence_of_divisible_power_sum_l2037_203711


namespace cubic_function_range_l2037_203710

/-- Given a cubic function f(x) = ax³ + bx where a and b are real constants,
    if f(2) = 2 and f'(2) = 9, then the range of f(x) for x ∈ ℝ is [-2, 18]. -/
theorem cubic_function_range (a b : ℝ) :
  (∀ x, f x = a * x^3 + b * x) →
  f 2 = 2 →
  (∀ x, deriv f x = 3 * a * x^2 + b) →
  deriv f 2 = 9 →
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ 18 :=
by sorry


end cubic_function_range_l2037_203710


namespace complex_calculation_l2037_203709

theorem complex_calculation : 
  (2/3 * Real.sqrt 180) * (0.4 * 300)^3 - (0.4 * 180 - 1/3 * (0.4 * 180)) = 15454377.6 := by
  sorry

end complex_calculation_l2037_203709


namespace fib_100_mod_5_l2037_203703

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Theorem statement
theorem fib_100_mod_5 : fib 99 % 5 = 0 := by
  sorry

end fib_100_mod_5_l2037_203703


namespace base_height_proof_l2037_203729

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_valid_inches : inches < 12

/-- Converts a Height to feet -/
def heightToFeet (h : Height) : ℚ :=
  h.feet + h.inches / 12

theorem base_height_proof (sculpture_height : Height) 
    (h_sculpture_height : sculpture_height = ⟨2, 10, by norm_num⟩) 
    (combined_height : ℚ) 
    (h_combined_height : combined_height = 3) :
  let base_height := combined_height - heightToFeet sculpture_height
  base_height * 12 = 2 := by
  sorry

end base_height_proof_l2037_203729


namespace complex_calculation_result_l2037_203792

theorem complex_calculation_result : (13.672 * 125 + 136.72 * 12.25 - 1367.2 * 1.875) / 17.09 = 60.5 := by
  sorry

end complex_calculation_result_l2037_203792


namespace empty_square_existence_l2037_203717

/-- Represents a chessboard with rooks -/
structure Chessboard :=
  (size : ℕ)
  (rooks : Finset (ℕ × ℕ))

/-- Defines a valid chessboard configuration -/
def is_valid_configuration (board : Chessboard) : Prop :=
  board.size = 50 ∧ 
  board.rooks.card = 50 ∧
  ∀ (r1 r2 : ℕ × ℕ), r1 ∈ board.rooks → r2 ∈ board.rooks → r1 ≠ r2 → 
    r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2

/-- Defines an empty square on the chessboard -/
def has_empty_square (board : Chessboard) (k : ℕ) : Prop :=
  ∃ (x y : ℕ), x + k ≤ board.size ∧ y + k ≤ board.size ∧
    ∀ (i j : ℕ), i < k → j < k → (x + i, y + j) ∉ board.rooks

/-- The main theorem -/
theorem empty_square_existence (board : Chessboard) (h : is_valid_configuration board) :
  ∀ k : ℕ, (k ≤ 7 ↔ ∀ (board : Chessboard), is_valid_configuration board → has_empty_square board k) :=
sorry

end empty_square_existence_l2037_203717


namespace problem1_l2037_203796

theorem problem1 (a b : ℝ) : 3 * (a^2 - a*b) - 5 * (a*b + 2*a^2 - 1) = -7*a^2 - 8*a*b + 5 := by
  sorry

end problem1_l2037_203796


namespace tan_eight_pi_thirds_l2037_203743

theorem tan_eight_pi_thirds : Real.tan (8 * π / 3) = -Real.sqrt 3 := by
  sorry

end tan_eight_pi_thirds_l2037_203743


namespace alice_bob_number_sum_l2037_203779

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B →
    (¬(is_prime A) ∧ ¬(¬is_prime A)) →
    (¬(is_prime B) ∧ is_perfect_square B) →
    is_perfect_square (50 * B + A) →
    A + B = 43 := by
  sorry

end alice_bob_number_sum_l2037_203779


namespace lightest_pumpkin_weight_l2037_203771

/-- Given three pumpkins with weights A, B, and C, prove that A = 5 -/
theorem lightest_pumpkin_weight (A B C : ℕ) 
  (h1 : A ≤ B) (h2 : B ≤ C)
  (h3 : A + B = 12) (h4 : A + C = 13) (h5 : B + C = 15) : 
  A = 5 := by
  sorry

end lightest_pumpkin_weight_l2037_203771


namespace hyperbola_vertices_distance_l2037_203745

theorem hyperbola_vertices_distance (x y : ℝ) :
  (x^2 / 121 - y^2 / 49 = 1) →
  (∃ v₁ v₂ : ℝ × ℝ, v₁.1 = -11 ∧ v₁.2 = 0 ∧ v₂.1 = 11 ∧ v₂.2 = 0 ∧
    (v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2 = 22^2) :=
by sorry

end hyperbola_vertices_distance_l2037_203745


namespace geometric_sequence_common_ratio_l2037_203754

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence condition
def arithmeticSequenceCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 1 = a 3

-- Main theorem
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : isGeometricSequence a q)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : arithmeticSequenceCondition a) :
  q = (Real.sqrt 5 + 1) / 2 :=
sorry

end geometric_sequence_common_ratio_l2037_203754


namespace crayon_difference_l2037_203785

def karen_crayons : ℕ := 639
def cindy_crayons : ℕ := 504
def peter_crayons : ℕ := 752
def rachel_crayons : ℕ := 315

theorem crayon_difference :
  (max karen_crayons (max cindy_crayons (max peter_crayons rachel_crayons))) -
  (min karen_crayons (min cindy_crayons (min peter_crayons rachel_crayons))) = 437 := by
  sorry

end crayon_difference_l2037_203785


namespace revenue_increase_when_doubled_l2037_203777

/-- Production function model -/
noncomputable def Q (A K L α₁ α₂ : ℝ) : ℝ := A * K^α₁ * L^α₂

/-- Theorem: When α₁ + α₂ > 1, doubling inputs more than doubles revenue -/
theorem revenue_increase_when_doubled
  (A K L α₁ α₂ : ℝ)
  (h_A : A > 0)
  (h_α₁ : 0 < α₁ ∧ α₁ < 1)
  (h_α₂ : 0 < α₂ ∧ α₂ < 1)
  (h_sum : α₁ + α₂ > 1) :
  Q A (2 * K) (2 * L) α₁ α₂ > 2 * Q A K L α₁ α₂ :=
sorry

end revenue_increase_when_doubled_l2037_203777


namespace solution_of_system_l2037_203722

variable (a b c x y z : ℝ)

theorem solution_of_system :
  (a * x + b * y - c * z = 2 * a * b) →
  (a * x - b * y + c * z = 2 * a * c) →
  (-a * x + b * y - c * z = 2 * b * c) →
  (x = b + c ∧ y = a + c ∧ z = a + b) :=
by sorry

end solution_of_system_l2037_203722


namespace correlation_strength_linear_correlation_strength_l2037_203744

-- Define the correlation coefficient r
variable (r : ℝ) 

-- Define the absolute value of r
def abs_r := |r|

-- Define the property that r is a valid correlation coefficient
def is_valid_corr_coeff (r : ℝ) : Prop := -1 ≤ r ∧ r ≤ 1

-- Define the degree of correlation as a function of |r|
def degree_of_correlation (abs_r : ℝ) : ℝ := abs_r

-- Define the degree of linear correlation as a function of |r|
def degree_of_linear_correlation (abs_r : ℝ) : ℝ := abs_r

-- Theorem 1: As |r| increases, the degree of correlation increases
theorem correlation_strength (r1 r2 : ℝ) 
  (h1 : is_valid_corr_coeff r1) (h2 : is_valid_corr_coeff r2) :
  abs_r r1 < abs_r r2 → degree_of_correlation (abs_r r1) < degree_of_correlation (abs_r r2) :=
sorry

-- Theorem 2: As |r| approaches 1, the degree of linear correlation strengthens
theorem linear_correlation_strength (r : ℝ) (h : is_valid_corr_coeff r) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', is_valid_corr_coeff r' →
    abs_r r' > 1 - δ → degree_of_linear_correlation (abs_r r') > 1 - ε :=
sorry

end correlation_strength_linear_correlation_strength_l2037_203744


namespace complex_quadratic_roots_l2037_203773

theorem complex_quadratic_roots : 
  ∃ (z₁ z₂ : ℂ), z₁ = Complex.I ∧ z₂ = -3 - 2*Complex.I ∧
  (∀ z : ℂ, z^2 + 2*z = -3 + 4*Complex.I ↔ z = z₁ ∨ z = z₂) :=
by sorry

end complex_quadratic_roots_l2037_203773


namespace fifth_term_constant_binomial_l2037_203723

theorem fifth_term_constant_binomial (n : ℕ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 4) * (-2)^4 * k = (Nat.choose n 4) * (-2)^4) → 
  n = 12 := by
sorry

end fifth_term_constant_binomial_l2037_203723


namespace photo_framing_yards_l2037_203742

/-- Calculates the minimum number of linear yards of framing needed for an enlarged photo with border. -/
def min_framing_yards (original_width : ℕ) (original_height : ℕ) (enlarge_factor : ℕ) (border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  let yards_needed := (perimeter_inches + 35) / 36  -- Ceiling division
  yards_needed

/-- Theorem stating that for a 5x7 inch photo enlarged 4 times with a 3-inch border,
    the minimum number of linear yards of framing needed is 4. -/
theorem photo_framing_yards :
  min_framing_yards 5 7 4 3 = 4 := by
  sorry

end photo_framing_yards_l2037_203742


namespace fans_with_all_items_fans_with_all_items_is_27_l2037_203760

def total_fans : ℕ := 5000
def tshirt_interval : ℕ := 90
def cap_interval : ℕ := 45
def scarf_interval : ℕ := 60

theorem fans_with_all_items : ℕ := by
  -- The number of fans who received all three promotional items
  -- is equal to the floor division of total_fans by the LCM of
  -- tshirt_interval, cap_interval, and scarf_interval
  sorry

-- Prove that fans_with_all_items equals 27
theorem fans_with_all_items_is_27 : fans_with_all_items = 27 := by
  sorry

end fans_with_all_items_fans_with_all_items_is_27_l2037_203760


namespace factorization_problem1_l2037_203772

theorem factorization_problem1 (m a : ℝ) : m * (a - 3) + 2 * (3 - a) = (a - 3) * (m - 2) := by
  sorry

end factorization_problem1_l2037_203772


namespace correct_value_proof_l2037_203769

theorem correct_value_proof (n : ℕ) (initial_mean correct_mean wrong_value : ℚ) 
  (h1 : n = 30)
  (h2 : initial_mean = 250)
  (h3 : correct_mean = 251)
  (h4 : wrong_value = 135) :
  ∃ (correct_value : ℚ),
    correct_value = 165 ∧
    n * correct_mean = n * initial_mean - wrong_value + correct_value :=
by
  sorry

end correct_value_proof_l2037_203769


namespace ball_returns_to_bella_l2037_203714

/-- Represents the number of girls in the circle -/
def n : ℕ := 13

/-- Represents the number of positions to move in each throw -/
def k : ℕ := 6

/-- Represents the position after a certain number of throws -/
def position (throws : ℕ) : ℕ :=
  (1 + throws * k) % n

/-- Theorem: The ball returns to Bella after exactly 13 throws -/
theorem ball_returns_to_bella :
  position 13 = 1 ∧ ∀ m : ℕ, m < 13 → position m ≠ 1 :=
sorry

end ball_returns_to_bella_l2037_203714


namespace chess_club_girls_count_l2037_203761

theorem chess_club_girls_count (total_members : ℕ) (present_members : ℕ) 
  (h1 : total_members = 32)
  (h2 : present_members = 20)
  (h3 : ∃ (boys girls : ℕ), boys + girls = total_members ∧ boys + girls / 2 = present_members) :
  ∃ (girls : ℕ), girls = 24 ∧ ∃ (boys : ℕ), boys + girls = total_members := by
sorry

end chess_club_girls_count_l2037_203761


namespace divisibility_by_five_l2037_203734

theorem divisibility_by_five (n : ℕ) : (76 * n^5 + 115 * n^4 + 19 * n) % 5 = 0 := by
  sorry

end divisibility_by_five_l2037_203734


namespace unique_b_for_three_integer_solutions_l2037_203750

theorem unique_b_for_three_integer_solutions :
  ∃! b : ℤ, ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x : ℤ, x ∈ s ↔ x^2 + b*x + 5 ≤ 0 :=
sorry

end unique_b_for_three_integer_solutions_l2037_203750


namespace find_a_value_l2037_203740

/-- The problem statement translated to Lean 4 --/
theorem find_a_value (a : ℝ) :
  (∀ x y : ℝ, 2*x - y + a ≥ 0 ∧ 3*x + y - 3 ≤ 0 →
    4*x + 3*y ≤ 8) ∧
  (∃ x y : ℝ, 2*x - y + a ≥ 0 ∧ 3*x + y - 3 ≤ 0 ∧
    4*x + 3*y = 8) →
  a = 2 := by
  sorry

end find_a_value_l2037_203740


namespace pta_spending_ratio_l2037_203752

/-- Proves the ratio of money spent on food for faculty to the amount left after buying school supplies -/
theorem pta_spending_ratio (initial_savings : ℚ) (school_supplies_fraction : ℚ) (final_amount : ℚ)
  (h1 : initial_savings = 400)
  (h2 : school_supplies_fraction = 1/4)
  (h3 : final_amount = 150)
  : (initial_savings * (1 - school_supplies_fraction) - final_amount) / 
    (initial_savings * (1 - school_supplies_fraction)) = 1/2 := by
  sorry

end pta_spending_ratio_l2037_203752


namespace cylinder_ellipse_eccentricity_l2037_203784

/-- The eccentricity of an ellipse formed by intersecting a cylinder with a plane -/
theorem cylinder_ellipse_eccentricity (d : ℝ) (θ : ℝ) (h_d : d = 12) (h_θ : θ = π / 6) :
  let r := d / 2
  let b := r
  let a := r / Real.cos θ
  let c := Real.sqrt (a^2 - b^2)
  c / a = 1 / 2 := by sorry

end cylinder_ellipse_eccentricity_l2037_203784


namespace quadratic_inequality_solution_l2037_203716

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 8 ↔ -8 < x ∧ x < 1 := by sorry

end quadratic_inequality_solution_l2037_203716


namespace sum_of_x_and_y_l2037_203778

theorem sum_of_x_and_y (x y : ℚ) 
  (hx : |x| = 5)
  (hy : |y| = 2)
  (hxy : |x - y| = x - y) :
  x + y = 7 ∨ x + y = 3 := by
sorry

end sum_of_x_and_y_l2037_203778


namespace interest_rate_proof_l2037_203726

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * ((1 + r) ^ t - 1)

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  |x - y| < ε

theorem interest_rate_proof (P : ℝ) (CI : ℝ) (t : ℕ) (r : ℝ) 
  (h1 : P = 400)
  (h2 : CI = 100)
  (h3 : t = 2)
  (h4 : compound_interest P r t = CI) :
  approx_equal r 0.11803398875 0.00000001 :=
sorry

end interest_rate_proof_l2037_203726


namespace money_exchange_solution_money_exchange_unique_l2037_203775

/-- Represents the money exchange process between three friends -/
def money_exchange (a b c : ℕ) : Prop :=
  let step1_1 := a - b - c
  let step1_2 := 2 * b
  let step1_3 := 2 * c
  let step2_1 := 2 * (a - b - c)
  let step2_2 := 3 * b - a - 3 * c
  let step2_3 := 4 * c
  let step3_1 := 4 * (a - b - c)
  let step3_2 := 6 * b - 2 * a - 6 * c
  let step3_3 := 4 * c - 2 * (a - b - c) - (3 * b - a - 3 * c)
  step3_1 = 8 ∧ step3_2 = 8 ∧ step3_3 = 8

/-- Theorem stating that the initial amounts of 13, 7, and 4 écus result in each friend having 8 écus after the exchanges -/
theorem money_exchange_solution :
  money_exchange 13 7 4 :=
sorry

/-- Theorem stating that 13, 7, and 4 are the only initial amounts that result in each friend having 8 écus after the exchanges -/
theorem money_exchange_unique :
  ∀ a b c : ℕ, money_exchange a b c → (a = 13 ∧ b = 7 ∧ c = 4) :=
sorry

end money_exchange_solution_money_exchange_unique_l2037_203775


namespace dodecagon_square_area_ratio_l2037_203767

theorem dodecagon_square_area_ratio :
  ∀ (square_side : ℝ) (dodecagon_area : ℝ),
    square_side = 2 →
    dodecagon_area = 3 →
    ∃ (shaded_area : ℝ),
      shaded_area = (square_side^2 - dodecagon_area) / 4 ∧
      shaded_area / dodecagon_area = 1 / 12 := by
  sorry

end dodecagon_square_area_ratio_l2037_203767


namespace cinema_seating_l2037_203712

/-- The number of people sitting between the far right and far left audience members -/
def people_between : ℕ := 30

/-- The total number of people sitting in the chairs -/
def total_people : ℕ := people_between + 2

theorem cinema_seating : total_people = 32 := by
  sorry

end cinema_seating_l2037_203712


namespace inequality_solution_l2037_203793

theorem inequality_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (2 * x) / (x + 1) + (x - 3) / (3 * x) ≤ 4 ↔ x < -1 ∨ x > -1/2 :=
by sorry

end inequality_solution_l2037_203793


namespace sum_20_terms_eq_2870_l2037_203756

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n^2

-- Define the sum of the first n terms of the sequence
def sum_a (n : ℕ) : ℕ := 
  (n * (n + 1) * (2 * n + 1)) / 6

-- Theorem statement
theorem sum_20_terms_eq_2870 :
  sum_a 20 = 2870 := by sorry

end sum_20_terms_eq_2870_l2037_203756


namespace unmarked_trees_l2037_203753

def total_trees : ℕ := 200
def mark_interval_out : ℕ := 5
def mark_interval_back : ℕ := 8

theorem unmarked_trees :
  let marked_out := total_trees / mark_interval_out
  let marked_back := total_trees / mark_interval_back
  let overlap := total_trees / (mark_interval_out * mark_interval_back)
  let total_marked := marked_out + marked_back - overlap
  total_trees - total_marked = 140 := by
  sorry

end unmarked_trees_l2037_203753


namespace quiz_competition_participants_l2037_203719

/-- The number of participants who started the national quiz competition -/
def initial_participants : ℕ := 300

/-- The fraction of participants remaining after the first round -/
def first_round_remaining : ℚ := 2/5

/-- The fraction of participants remaining after the second round, relative to those who remained after the first round -/
def second_round_remaining : ℚ := 1/4

/-- The number of participants remaining after the second round -/
def final_participants : ℕ := 30

theorem quiz_competition_participants :
  (↑initial_participants * first_round_remaining * second_round_remaining : ℚ) = ↑final_participants :=
sorry

end quiz_competition_participants_l2037_203719


namespace fraction_equation_solution_l2037_203770

theorem fraction_equation_solution : ∃ x : ℚ, (1 / 2 - 1 / 3 : ℚ) = 1 / x ∧ x = 6 := by
  sorry

end fraction_equation_solution_l2037_203770


namespace repeating_decimal_to_fraction_l2037_203768

theorem repeating_decimal_to_fraction :
  ∀ (a b : ℕ) (x : ℚ),
    (x = 0.4 + (31 : ℚ) / (990 : ℚ)) →
    (x = (427 : ℚ) / (990 : ℚ)) :=
by
  sorry

end repeating_decimal_to_fraction_l2037_203768


namespace grocery_store_soda_count_l2037_203789

/-- Given a grocery store inventory, prove the number of regular soda bottles -/
theorem grocery_store_soda_count 
  (diet_soda : ℕ) 
  (regular_soda_diff : ℕ) 
  (h1 : diet_soda = 53)
  (h2 : regular_soda_diff = 26) :
  diet_soda + regular_soda_diff = 79 := by
  sorry

end grocery_store_soda_count_l2037_203789


namespace quadratic_perfect_square_l2037_203765

theorem quadratic_perfect_square (x : ℝ) : 
  (∃ a : ℝ, x^2 + 10*x + 25 = (x + a)^2) ∧ 
  (∀ c : ℝ, c ≠ 25 → ¬∃ a : ℝ, x^2 + 10*x + c = (x + a)^2) :=
sorry

end quadratic_perfect_square_l2037_203765


namespace change_in_responses_l2037_203762

theorem change_in_responses (initial_yes initial_no final_yes final_no : ℚ) 
  (h1 : initial_yes = 50 / 100)
  (h2 : initial_no = 50 / 100)
  (h3 : final_yes = 70 / 100)
  (h4 : final_no = 30 / 100)
  (h5 : initial_yes + initial_no = 1)
  (h6 : final_yes + final_no = 1) :
  ∃ (min_change max_change : ℚ),
    min_change ≥ 0 ∧
    max_change ≤ 1 ∧
    min_change ≤ max_change ∧
    max_change - min_change = 30 / 100 :=
by sorry

end change_in_responses_l2037_203762


namespace sun_city_population_l2037_203718

theorem sun_city_population (willowdale roseville sun : ℕ) : 
  willowdale = 2000 →
  roseville = 3 * willowdale - 500 →
  sun = 2 * roseville + 1000 →
  sun = 12000 := by
sorry

end sun_city_population_l2037_203718


namespace proposition_truth_values_l2037_203705

theorem proposition_truth_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p)) : 
  ¬q := by
  sorry

end proposition_truth_values_l2037_203705


namespace original_equals_scientific_l2037_203798

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we want to express in scientific notation -/
def original_number : ℕ := 135000

/-- The proposed scientific notation representation -/
def scientific_form : ScientificNotation := {
  coefficient := 1.35
  exponent := 5
  coeff_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l2037_203798


namespace find_e_l2037_203791

/-- Given two functions f and g, and a composition condition, prove the value of e. -/
theorem find_e (b e : ℝ) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * x + b)
  (g : ℝ → ℝ) (hg : ∀ x, g x = b * x + 5)
  (h_comp : ∀ x, f (g x) = 15 * x + e) : 
  e = 15 := by sorry

end find_e_l2037_203791


namespace no_solution_for_equation_l2037_203790

theorem no_solution_for_equation : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (1 / x + 1 / y = 3 / (x + y)) := by
  sorry

end no_solution_for_equation_l2037_203790


namespace accumulate_small_steps_necessary_not_sufficient_l2037_203794

-- Define the concept of "reaching a thousand miles"
def reach_thousand_miles : Prop := sorry

-- Define the concept of "accumulating small steps"
def accumulate_small_steps : Prop := sorry

-- Xunzi's saying as an axiom
axiom xunzi_saying : ¬accumulate_small_steps → ¬reach_thousand_miles

-- Define what it means to be a necessary condition
def is_necessary_condition (condition goal : Prop) : Prop :=
  ¬condition → ¬goal

-- Define what it means to be a sufficient condition
def is_sufficient_condition (condition goal : Prop) : Prop :=
  condition → goal

-- Theorem to prove
theorem accumulate_small_steps_necessary_not_sufficient :
  (is_necessary_condition accumulate_small_steps reach_thousand_miles) ∧
  ¬(is_sufficient_condition accumulate_small_steps reach_thousand_miles) := by
  sorry

end accumulate_small_steps_necessary_not_sufficient_l2037_203794


namespace arccos_one_half_equals_pi_third_l2037_203795

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l2037_203795


namespace expression_evaluation_l2037_203783

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := 2
  2 * x^2 - y^2 + (2 * y^2 - 3 * x^2) - (2 * y^2 + x^2) = -6 :=
by
  sorry

end expression_evaluation_l2037_203783


namespace readers_all_three_genres_l2037_203755

/-- Represents the number of readers for each genre and their intersections --/
structure ReaderCounts where
  total : ℕ
  sf : ℕ
  lw : ℕ
  hf : ℕ
  sf_lw : ℕ
  sf_hf : ℕ
  lw_hf : ℕ

/-- The principle of inclusion-exclusion for three sets --/
def inclusionExclusion (r : ReaderCounts) (x : ℕ) : Prop :=
  r.total = r.sf + r.lw + r.hf - r.sf_lw - r.sf_hf - r.lw_hf + x

/-- The theorem stating the number of readers who read all three genres --/
theorem readers_all_three_genres (r : ReaderCounts) 
  (h_total : r.total = 800)
  (h_sf : r.sf = 300)
  (h_lw : r.lw = 600)
  (h_hf : r.hf = 400)
  (h_sf_lw : r.sf_lw = 175)
  (h_sf_hf : r.sf_hf = 150)
  (h_lw_hf : r.lw_hf = 250) :
  ∃ x, inclusionExclusion r x ∧ x = 75 := by
  sorry

end readers_all_three_genres_l2037_203755


namespace rectangle_diagonal_l2037_203751

/-- The diagonal length of a rectangle with sides 30√3 cm and 30 cm is 60 cm. -/
theorem rectangle_diagonal : ℝ → Prop :=
  fun diagonal =>
    let side1 := 30 * Real.sqrt 3
    let side2 := 30
    diagonal ^ 2 = side1 ^ 2 + side2 ^ 2 →
    diagonal = 60

-- The proof is omitted
axiom rectangle_diagonal_proof : rectangle_diagonal 60

#check rectangle_diagonal_proof

end rectangle_diagonal_l2037_203751


namespace solution_for_F_l2037_203774

/-- Definition of function F --/
def F (a b c : ℝ) : ℝ := a * b^2 - c

/-- Theorem stating that 1/6 is the solution to F(a,5,10) = F(a,7,14) --/
theorem solution_for_F : ∃ a : ℝ, F a 5 10 = F a 7 14 ∧ a = 1/6 := by
  sorry

end solution_for_F_l2037_203774


namespace circular_arrangement_students_l2037_203741

/-- Given a circular arrangement of students, if the 8th student is directly opposite the 33rd student, then the total number of students is 52. -/
theorem circular_arrangement_students (n : ℕ) : 
  (∃ (a b : ℕ), a = 8 ∧ b = 33 ∧ a < b ∧ b - a = n - (b - a)) → n = 52 := by
  sorry

end circular_arrangement_students_l2037_203741


namespace bank_exceeds_500_on_day_9_l2037_203728

def deposit_amount (day : ℕ) : ℕ :=
  if day ≤ 1 then 3
  else if day % 2 = 0 then 3 * deposit_amount (day - 2)
  else deposit_amount (day - 1)

def total_amount (day : ℕ) : ℕ :=
  List.sum (List.map deposit_amount (List.range (day + 1)))

theorem bank_exceeds_500_on_day_9 :
  total_amount 8 ≤ 500 ∧ total_amount 9 > 500 :=
sorry

end bank_exceeds_500_on_day_9_l2037_203728


namespace abs_neg_seven_l2037_203749

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end abs_neg_seven_l2037_203749


namespace three_digit_numbers_with_repetition_l2037_203763

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of digits in the numbers we're forming -/
def num_places : ℕ := 3

/-- The number of non-zero digits available for the first place -/
def non_zero_digits : ℕ := num_digits - 1

/-- The total number of three-digit numbers (including those without repetition) -/
def total_numbers : ℕ := non_zero_digits * num_digits * num_digits

/-- The number of three-digit numbers without repetition -/
def numbers_without_repetition : ℕ := non_zero_digits * (num_digits - 1) * (num_digits - 2)

theorem three_digit_numbers_with_repetition :
  total_numbers - numbers_without_repetition = 252 := by
  sorry

end three_digit_numbers_with_repetition_l2037_203763


namespace translation_theorem_l2037_203799

/-- The original function -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 4

/-- The translated function -/
def g (x : ℝ) : ℝ := -(x + 1)^2 + 1

/-- Translation parameters -/
def left_shift : ℝ := 2
def down_shift : ℝ := 3

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + left_shift) - down_shift := by
  sorry

end translation_theorem_l2037_203799


namespace zero_success_probability_l2037_203738

/-- Probability of success in a single trial -/
def p : ℚ := 2 / 7

/-- Number of trials -/
def n : ℕ := 7

/-- Probability of exactly k successes in n Bernoulli trials with success probability p -/
def binomialProbability (k : ℕ) : ℚ :=
  (n.choose k) * p ^ k * (1 - p) ^ (n - k)

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is equal to (5/7)^7 -/
theorem zero_success_probability : 
  binomialProbability 0 = (5 / 7) ^ 7 := by sorry

end zero_success_probability_l2037_203738


namespace marble_probability_l2037_203764

theorem marble_probability (box1 box2 : Nat) : 
  box1 + box2 = 36 →
  (box1 * box2 : Rat) = 36 →
  (∃ black1 black2 : Nat, 
    black1 ≤ box1 ∧ 
    black2 ≤ box2 ∧ 
    (black1 * black2 : Rat) / (box1 * box2) = 25 / 36) →
  (∃ white1 white2 : Nat,
    white1 = box1 - black1 ∧
    white2 = box2 - black2 ∧
    (white1 * white2 : Rat) / (box1 * box2) = 169 / 324) :=
by sorry

end marble_probability_l2037_203764


namespace hyperbola_m_value_l2037_203731

/-- A hyperbola with equation mx² + y² = 1 where the length of its imaginary axis 
    is twice the length of its real axis -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, m * x^2 + y^2 = 1
  axis_ratio : (imaginary_axis_length : ℝ) = 2 * (real_axis_length : ℝ)

/-- The value of m for a hyperbola with the given properties is -1/4 -/
theorem hyperbola_m_value (h : Hyperbola) : h.m = -1/4 := by
  sorry

end hyperbola_m_value_l2037_203731


namespace word_count_is_370_l2037_203747

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of five-letter words with exactly two A's and at least one O -/
def word_count : ℕ :=
  let a_freq := 6
  let e_freq := 4
  let i_freq := 5
  let o_freq := 3
  let u_freq := 2
  let word_length := 5
  let a_count := 2
  let remaining_letters := word_length - a_count
  let ways_to_place_a := choose word_length a_count
  let ways_to_place_o_and_others := 
    (choose remaining_letters 1) * (e_freq + i_freq + u_freq)^2 +
    (choose remaining_letters 2) * (e_freq + i_freq + u_freq) +
    (choose remaining_letters 3)
  ways_to_place_a * ways_to_place_o_and_others

theorem word_count_is_370 : word_count = 370 := by
  sorry

end word_count_is_370_l2037_203747


namespace six_integers_mean_twice_mode_l2037_203746

theorem six_integers_mean_twice_mode (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 
  x ≤ 100 ∧ y ≤ 100 ∧
  y > x ∧
  (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x →
  x = 16 := by
sorry

end six_integers_mean_twice_mode_l2037_203746


namespace complex_magnitude_sqrt_5_l2037_203759

def complex (a b : ℝ) := a + b * Complex.I

theorem complex_magnitude_sqrt_5 (a b : ℝ) (h : a / (1 - Complex.I) = 1 - b * Complex.I) :
  Complex.abs (complex a b) = Real.sqrt 5 := by
  sorry

end complex_magnitude_sqrt_5_l2037_203759
