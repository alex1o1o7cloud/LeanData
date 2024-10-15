import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l3558_355829

theorem triangle_properties (a b c A B C : Real) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Given conditions
  π/2 < A ∧ -- A is obtuse
  a * Real.sin B = b * Real.cos B ∧
  C = π/6 →
  -- Conclusions
  A = 2*π/3 ∧
  1 < Real.cos A + Real.cos B + Real.cos C ∧
  Real.cos A + Real.cos B + Real.cos C ≤ 5/4 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3558_355829


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3558_355811

theorem largest_n_with_unique_k : 
  (∀ n : ℕ+, n > 1 → 
    ¬(∃! k : ℤ, (3 : ℚ)/7 < (n : ℚ)/((n : ℚ) + k) ∧ (n : ℚ)/((n : ℚ) + k) < 8/19)) ∧
  (∃! k : ℤ, (3 : ℚ)/7 < (1 : ℚ)/((1 : ℚ) + k) ∧ (1 : ℚ)/((1 : ℚ) + k) < 8/19) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3558_355811


namespace NUMINAMATH_CALUDE_prob_dime_is_25_143_l3558_355835

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 900
  | Coin.Dime => 500
  | Coin.Penny => 200

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Dime + coin_count Coin.Penny

/-- The probability of picking a dime from the jar -/
def prob_dime : ℚ := coin_count Coin.Dime / total_coins

theorem prob_dime_is_25_143 : prob_dime = 25 / 143 := by
  sorry


end NUMINAMATH_CALUDE_prob_dime_is_25_143_l3558_355835


namespace NUMINAMATH_CALUDE_exponent_division_l3558_355888

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^4 / a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3558_355888


namespace NUMINAMATH_CALUDE_sum_m_n_range_l3558_355831

/-- A quadratic function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating that given the conditions, m + n is in [-4, 0] -/
theorem sum_m_n_range (m n : ℝ) (h1 : m ≤ n) (h2 : ∀ x ∈ Set.Icc m n, -1 ≤ f x ∧ f x ≤ 3) :
  -4 ≤ m + n ∧ m + n ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sum_m_n_range_l3558_355831


namespace NUMINAMATH_CALUDE_inequality_proof_l3558_355833

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : c^2 + a*b = a^2 + b^2) : 
  c^2 + a*b ≤ a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3558_355833


namespace NUMINAMATH_CALUDE_negative_fifteen_inequality_l3558_355873

theorem negative_fifteen_inequality (a b : ℝ) (h : a > b) : -15 * a < -15 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_fifteen_inequality_l3558_355873


namespace NUMINAMATH_CALUDE_day_318_is_monday_l3558_355890

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a specific day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Given that the 45th day of 2003 is a Monday, 
    prove that the 318th day of 2003 is also a Monday -/
theorem day_318_is_monday (d45 d318 : DayInYear) 
  (h1 : d45.dayNumber = 45)
  (h2 : d45.dayOfWeek = DayOfWeek.Monday)
  (h3 : d318.dayNumber = 318) :
  d318.dayOfWeek = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_day_318_is_monday_l3558_355890


namespace NUMINAMATH_CALUDE_function_value_proof_l3558_355887

theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x - 1)
  (h2 : f 2 = 3) :
  f 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l3558_355887


namespace NUMINAMATH_CALUDE_peter_wins_iff_n_odd_l3558_355864

/-- Represents the state of a cup (empty or filled) -/
inductive CupState
| Empty : CupState
| Filled : CupState

/-- Represents a player in the game -/
inductive Player
| Peter : Player
| Vasya : Player

/-- The game state on a 2n-gon -/
structure GameState (n : ℕ) where
  cups : Fin (2 * n) → CupState
  currentPlayer : Player

/-- Checks if two positions are symmetric with respect to the center of the 2n-gon -/
def isSymmetric (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + j.val) % (2 * n) = 0

/-- A valid move in the game -/
inductive Move (n : ℕ)
| Single : Fin (2 * n) → Move n
| Double : (i j : Fin (2 * n)) → isSymmetric n i j → Move n

/-- Applies a move to the game state -/
def applyMove (n : ℕ) (state : GameState n) (move : Move n) : GameState n :=
  sorry

/-- Checks if a player has a winning strategy -/
def hasWinningStrategy (n : ℕ) (player : Player) : Prop :=
  sorry

/-- The main theorem: Peter has a winning strategy if and only if n is odd -/
theorem peter_wins_iff_n_odd (n : ℕ) :
  hasWinningStrategy n Player.Peter ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_peter_wins_iff_n_odd_l3558_355864


namespace NUMINAMATH_CALUDE_annual_forest_gathering_handshakes_count_l3558_355806

/-- The number of handshakes at the Annual Forest Gathering -/
def annual_forest_gathering_handshakes (num_goblins num_elves : ℕ) : ℕ :=
  (num_goblins.choose 2) + (num_goblins * num_elves)

/-- Theorem stating the number of handshakes at the Annual Forest Gathering -/
theorem annual_forest_gathering_handshakes_count :
  annual_forest_gathering_handshakes 25 18 = 750 := by
  sorry

end NUMINAMATH_CALUDE_annual_forest_gathering_handshakes_count_l3558_355806


namespace NUMINAMATH_CALUDE_lcm_of_4_6_15_l3558_355812

theorem lcm_of_4_6_15 : Nat.lcm (Nat.lcm 4 6) 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_6_15_l3558_355812


namespace NUMINAMATH_CALUDE_kah_to_zah_conversion_l3558_355868

/-- Conversion rate between zahs and tols -/
def zah_to_tol : ℚ := 24 / 15

/-- Conversion rate between tols and kahs -/
def tol_to_kah : ℚ := 15 / 9

/-- The number of kahs we want to convert -/
def kahs_to_convert : ℕ := 2000

/-- The expected number of zahs after conversion -/
def expected_zahs : ℕ := 750

theorem kah_to_zah_conversion :
  (kahs_to_convert : ℚ) / (zah_to_tol * tol_to_kah) = expected_zahs := by
  sorry

end NUMINAMATH_CALUDE_kah_to_zah_conversion_l3558_355868


namespace NUMINAMATH_CALUDE_school_population_l3558_355865

/-- Given a school with boys and girls, prove the total number of students is 900 -/
theorem school_population (total boys girls : ℕ) : 
  total = boys + girls →
  boys = 90 →
  girls = (90 * total) / 100 →
  total = 900 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l3558_355865


namespace NUMINAMATH_CALUDE_intersection_perpendicular_implies_a_value_l3558_355830

-- Define the line equation
def line_eq (x y a : ℝ) : Prop := x - y + a = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop := sorry

theorem intersection_perpendicular_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), line_eq A.1 A.2 a ∧ line_eq B.1 B.2 a ∧ 
                     circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
                     perpendicular A B circle_center) →
  a = 0 ∨ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_implies_a_value_l3558_355830


namespace NUMINAMATH_CALUDE_definite_integral_equality_l3558_355844

theorem definite_integral_equality : ∫ x in (1 : ℝ)..3, (2 * x - 1 / x^2) = 22 / 3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_equality_l3558_355844


namespace NUMINAMATH_CALUDE_number_equation_l3558_355891

theorem number_equation (x : ℝ) : x - 2 + 4 = 9 ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_number_equation_l3558_355891


namespace NUMINAMATH_CALUDE_store_transaction_loss_l3558_355846

def selling_price : ℝ := 60

theorem store_transaction_loss (cost_price_1 cost_price_2 : ℝ) 
  (h1 : (selling_price - cost_price_1) / cost_price_1 = 1/2)
  (h2 : (cost_price_2 - selling_price) / cost_price_2 = 1/2) :
  2 * selling_price - (cost_price_1 + cost_price_2) = -selling_price / 3 := by
  sorry

end NUMINAMATH_CALUDE_store_transaction_loss_l3558_355846


namespace NUMINAMATH_CALUDE_brownies_per_person_l3558_355823

/-- Given a pan of brownies cut into columns and rows, calculate how many brownies each person can eat. -/
theorem brownies_per_person 
  (columns : ℕ) 
  (rows : ℕ) 
  (people : ℕ) 
  (h1 : columns = 6) 
  (h2 : rows = 3) 
  (h3 : people = 6) 
  : (columns * rows) / people = 3 := by
  sorry

end NUMINAMATH_CALUDE_brownies_per_person_l3558_355823


namespace NUMINAMATH_CALUDE_largest_palindrome_multiple_of_6_l3558_355859

def is_palindrome (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 = n % 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_palindrome_multiple_of_6 :
  ∀ n : ℕ, is_palindrome n → n % 6 = 0 → n ≤ 888 ∧
  (∃ m : ℕ, is_palindrome m ∧ m % 6 = 0 ∧ m = 888) ∧
  sum_of_digits 888 = 24 :=
sorry

end NUMINAMATH_CALUDE_largest_palindrome_multiple_of_6_l3558_355859


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l3558_355840

/-- Given a geometric sequence {aₙ} where the sum of the first n terms
    is Sₙ = 3ⁿ + r, prove that r = -1 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, S n = 3^n + r) →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  (a 1 = S 1) →
  (∀ n : ℕ, n ≥ 2 → a (n+1) = 3 * a n) →
  r = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l3558_355840


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l3558_355858

theorem geometric_mean_of_4_and_9 :
  ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l3558_355858


namespace NUMINAMATH_CALUDE_rotate_from_one_to_six_l3558_355850

/-- Represents a face of a standard six-sided die -/
inductive DieFace
| one
| two
| three
| four
| five
| six

/-- Represents the state of a die with visible faces -/
structure DieState where
  top : DieFace
  front : DieFace
  right : DieFace

/-- Defines the opposite face relation for a standard die -/
def opposite_face (f : DieFace) : DieFace :=
  match f with
  | DieFace.one => DieFace.six
  | DieFace.two => DieFace.five
  | DieFace.three => DieFace.four
  | DieFace.four => DieFace.three
  | DieFace.five => DieFace.two
  | DieFace.six => DieFace.one

/-- Simulates a 90° clockwise rotation of the die -/
def rotate_90_clockwise (s : DieState) : DieState :=
  { top := s.right
  , front := s.top
  , right := opposite_face s.front }

/-- Theorem: After a 90° clockwise rotation from a state where 1 is visible,
    the opposite face (6) becomes visible -/
theorem rotate_from_one_to_six (initial : DieState) 
    (h : initial.top = DieFace.one) : 
    ∃ (rotated : DieState), rotated = rotate_90_clockwise initial ∧ 
    (rotated.top = DieFace.six ∨ rotated.front = DieFace.six ∨ rotated.right = DieFace.six) :=
  sorry


end NUMINAMATH_CALUDE_rotate_from_one_to_six_l3558_355850


namespace NUMINAMATH_CALUDE_lending_interest_rate_lending_rate_is_six_percent_l3558_355805

/-- Calculates the lending interest rate given the borrowing details and yearly gain -/
theorem lending_interest_rate 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (duration : ℝ) 
  (yearly_gain : ℝ) : ℝ :=
let borrowed_interest := borrowed_amount * borrowing_rate * duration / 100
let total_gain := yearly_gain * duration
let lending_rate := (total_gain + borrowed_interest) * 100 / (borrowed_amount * duration)
lending_rate

/-- The lending interest rate is 6% given the specified conditions -/
theorem lending_rate_is_six_percent : 
  lending_interest_rate 5000 4 2 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lending_interest_rate_lending_rate_is_six_percent_l3558_355805


namespace NUMINAMATH_CALUDE_smallest_number_above_threshold_l3558_355807

theorem smallest_number_above_threshold : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℚ := 1.1
  let filtered := numbers.filter (λ x => x ≥ threshold)
  filtered.minimum? = some 1.2 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_above_threshold_l3558_355807


namespace NUMINAMATH_CALUDE_vector_angle_proof_l3558_355869

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_proof (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 4) 
  (h3 : (a + b) • a = 0) : 
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_proof_l3558_355869


namespace NUMINAMATH_CALUDE_smallest_n_for_four_sum_divisible_by_four_l3558_355804

theorem smallest_n_for_four_sum_divisible_by_four :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b + c + d) % 4 = 0) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (a + b + c + d) % 4 ≠ 0) ∧
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_four_sum_divisible_by_four_l3558_355804


namespace NUMINAMATH_CALUDE_pencils_given_to_joyce_l3558_355853

theorem pencils_given_to_joyce (initial_pencils : ℝ) (remaining_pencils : ℕ) 
  (h1 : initial_pencils = 51.0)
  (h2 : remaining_pencils = 45) :
  initial_pencils - remaining_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_to_joyce_l3558_355853


namespace NUMINAMATH_CALUDE_no_real_solutions_for_quadratic_inequality_l3558_355813

theorem no_real_solutions_for_quadratic_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_quadratic_inequality_l3558_355813


namespace NUMINAMATH_CALUDE_chess_players_lost_to_ai_l3558_355854

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) : 
  total_players = 40 → never_lost_fraction = 1/4 → 
  (total_players : ℚ) * (1 - never_lost_fraction) = 30 := by
  sorry

end NUMINAMATH_CALUDE_chess_players_lost_to_ai_l3558_355854


namespace NUMINAMATH_CALUDE_prob_heads_11th_toss_l3558_355867

/-- A fair coin is a coin with equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting heads on a single toss of a fair coin -/
def prob_heads (p : ℝ) : ℝ := p

/-- The number of tosses -/
def num_tosses : ℕ := 10

/-- The number of heads observed -/
def heads_observed : ℕ := 7

/-- Theorem: The probability of getting heads on the 11th toss of a fair coin is 0.5,
    given that the coin was tossed 10 times with 7 heads as the result -/
theorem prob_heads_11th_toss (p : ℝ) (h : fair_coin p) :
  prob_heads p = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_prob_heads_11th_toss_l3558_355867


namespace NUMINAMATH_CALUDE_coin_denomination_problem_l3558_355871

theorem coin_denomination_problem (total_coins : ℕ) (unknown_coins : ℕ) (known_coins : ℕ) 
  (known_coin_value : ℕ) (total_value : ℕ) (x : ℕ) :
  total_coins = 324 →
  unknown_coins = 220 →
  known_coins = total_coins - unknown_coins →
  known_coin_value = 25 →
  total_value = 7000 →
  unknown_coins * x + known_coins * known_coin_value = total_value →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_coin_denomination_problem_l3558_355871


namespace NUMINAMATH_CALUDE_train_length_calculation_l3558_355856

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) → 
  time_to_pass = 44 →
  bridge_length = 140 →
  train_speed * time_to_pass - bridge_length = 410 :=
by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3558_355856


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l3558_355801

theorem fraction_subtraction_equality : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l3558_355801


namespace NUMINAMATH_CALUDE_xy_value_l3558_355837

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(5*y) = 729) : 
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3558_355837


namespace NUMINAMATH_CALUDE_letter_lock_max_letters_l3558_355847

theorem letter_lock_max_letters (n : ℕ) : 
  (n ^ 3 - 1 ≤ 215) ∧ (∀ m : ℕ, m > n → m ^ 3 - 1 > 215) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_letter_lock_max_letters_l3558_355847


namespace NUMINAMATH_CALUDE_evaluate_expression_l3558_355876

theorem evaluate_expression : ((-2)^3)^(1/3) - (-1)^0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3558_355876


namespace NUMINAMATH_CALUDE_side_xy_length_l3558_355880

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ) := X + Y + Z = 180

-- Define the right angle
def RightAngle (Z : ℝ) := Z = 90

-- Define the area of the triangle
def TriangleArea (A : ℝ) := A = 36

-- Define the angles of the triangle
def AngleX (X : ℝ) := X = 30
def AngleY (Y : ℝ) := Y = 60

-- Theorem statement
theorem side_xy_length 
  (X Y Z A : ℝ) 
  (tri : Triangle X Y Z) 
  (right : RightAngle Z) 
  (area : TriangleArea A) 
  (angleX : AngleX X) 
  (angleY : AngleY Y) : 
  ∃ (XY : ℝ), XY = Real.sqrt (36 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_side_xy_length_l3558_355880


namespace NUMINAMATH_CALUDE_matthews_friends_l3558_355894

theorem matthews_friends (initial_crackers initial_cakes cakes_per_person : ℕ) 
  (h1 : initial_crackers = 10)
  (h2 : initial_cakes = 8)
  (h3 : cakes_per_person = 2)
  (h4 : initial_cakes % cakes_per_person = 0) :
  initial_cakes / cakes_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l3558_355894


namespace NUMINAMATH_CALUDE_fraction_inequality_implies_numerator_inequality_l3558_355821

theorem fraction_inequality_implies_numerator_inequality
  (a b c : ℝ) (hc : c ≠ 0) :
  a / c^2 > b / c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_implies_numerator_inequality_l3558_355821


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3558_355818

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - m*x - 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (f (Real.sqrt 2) = 0 → f (-Real.sqrt 2 / 2) = 0 ∧ m = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3558_355818


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3558_355842

theorem quadratic_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3558_355842


namespace NUMINAMATH_CALUDE_first_month_sale_l3558_355808

def average_sale : ℕ := 7000
def num_months : ℕ := 6
def sale_month2 : ℕ := 6524
def sale_month3 : ℕ := 5689
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6000
def sale_month6 : ℕ := 12557

theorem first_month_sale (sale_month1 : ℕ) : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6 = average_sale * num_months →
  sale_month1 = average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) :=
by
  sorry

#eval average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6)

end NUMINAMATH_CALUDE_first_month_sale_l3558_355808


namespace NUMINAMATH_CALUDE_angle_complement_supplement_l3558_355874

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = (1/3) * (180 - x) ↔ x = 45 := by sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_l3558_355874


namespace NUMINAMATH_CALUDE_meaningful_fraction_iff_x_gt_three_l3558_355828

theorem meaningful_fraction_iff_x_gt_three (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_iff_x_gt_three_l3558_355828


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3558_355845

theorem complex_product_magnitude : 
  Complex.abs ((3 - 4 * Complex.I) * (5 + 12 * Complex.I) * (2 - 7 * Complex.I)) = 65 * Real.sqrt 53 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3558_355845


namespace NUMINAMATH_CALUDE_log_problem_l3558_355841

theorem log_problem : Real.log (648 * Real.rpow 6 (1/3)) / Real.log (Real.rpow 6 (1/3)) = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l3558_355841


namespace NUMINAMATH_CALUDE_smallest_coprime_to_180_l3558_355875

theorem smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ x < 7 → Nat.gcd x 180 ≠ 1 ∧ Nat.gcd 7 180 = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_180_l3558_355875


namespace NUMINAMATH_CALUDE_triangle_properties_l3558_355898

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A + Real.sin t.B = 5/4 * Real.sin t.C)
  (h2 : t.a + t.b + t.c = 9)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sin t.C) :
  t.C = 4 ∧ Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3558_355898


namespace NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l3558_355849

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  rows : Nat
  cols : Nat

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  ((board.cols + 1) / 2) * board.rows

/-- Theorem: A 31x29 checkerboard has 465 black squares -/
theorem chubby_checkerboard_black_squares :
  let board : Checkerboard := ⟨31, 29⟩
  count_black_squares board = 465 := by
  sorry

#eval count_black_squares ⟨31, 29⟩

end NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l3558_355849


namespace NUMINAMATH_CALUDE_chocolate_boxes_sold_l3558_355877

/-- The number of chocolate biscuit boxes sold by Kaylee -/
def chocolate_boxes : ℕ :=
  let total_boxes : ℕ := 33
  let lemon_boxes : ℕ := 12
  let oatmeal_boxes : ℕ := 4
  let remaining_boxes : ℕ := 12
  total_boxes - (lemon_boxes + oatmeal_boxes + remaining_boxes)

theorem chocolate_boxes_sold :
  chocolate_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_sold_l3558_355877


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3558_355817

/-- The number of students in fourth grade at the end of the year -/
def final_students (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial number of students, the number of students who left,
    and the number of new students, prove that the final number of students is 47 -/
theorem fourth_grade_students :
  final_students 11 6 42 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3558_355817


namespace NUMINAMATH_CALUDE_jacks_second_half_time_l3558_355800

/-- Proves that Jack's time for the second half of the hill is 6 seconds -/
theorem jacks_second_half_time
  (jack_first_half : ℕ)
  (jack_finishes_before : ℕ)
  (jill_total_time : ℕ)
  (h1 : jack_first_half = 19)
  (h2 : jack_finishes_before = 7)
  (h3 : jill_total_time = 32) :
  jill_total_time - jack_finishes_before - jack_first_half = 6 := by
  sorry

#check jacks_second_half_time

end NUMINAMATH_CALUDE_jacks_second_half_time_l3558_355800


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3558_355886

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (hc : c ≠ 0) :
  a / c = b / c → a = b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3558_355886


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3558_355885

theorem system_of_equations_solutions :
  -- System (1)
  (∃ x y : ℝ, 3 * y - 4 * x = 0 ∧ 4 * x + y = 8 ∧ x = 3/2 ∧ y = 2) ∧
  -- System (2)
  (∃ x y : ℝ, x + y = 3 ∧ (x - 1)/4 + y/2 = 3/4 ∧ x = 2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3558_355885


namespace NUMINAMATH_CALUDE_book_series_first_year_l3558_355814

/-- Represents the publication years of a book series -/
def BookSeries (a : ℕ) : List ℕ :=
  List.range 7 |>.map (fun i => a + 7 * i)

/-- The theorem stating the properties of the book series -/
theorem book_series_first_year :
  ∀ a : ℕ,
  (BookSeries a).length = 7 ∧
  (∀ i j, i < j → (BookSeries a).get i < (BookSeries a).get j) ∧
  (BookSeries a).sum = 13524 →
  a = 1911 := by
sorry


end NUMINAMATH_CALUDE_book_series_first_year_l3558_355814


namespace NUMINAMATH_CALUDE_birds_nest_building_distance_l3558_355881

/-- Calculates the total distance covered by birds making round trips to collect nest materials. -/
def total_distance_covered (num_birds : ℕ) (num_trips : ℕ) (distance_to_materials : ℕ) : ℕ :=
  num_birds * num_trips * (2 * distance_to_materials)

/-- Theorem stating that two birds making 10 round trips each to collect materials 200 miles away cover a total distance of 8000 miles. -/
theorem birds_nest_building_distance :
  total_distance_covered 2 10 200 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_birds_nest_building_distance_l3558_355881


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l3558_355815

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 - 2*x^3 - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l3558_355815


namespace NUMINAMATH_CALUDE_other_candidate_votes_l3558_355866

theorem other_candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ)
  (h_total : total_votes = 8500)
  (h_invalid : invalid_percent = 25 / 100)
  (h_winner : winner_percent = 60 / 100) :
  ⌊(1 - winner_percent) * ((1 - invalid_percent) * total_votes)⌋ = 2550 := by
  sorry

end NUMINAMATH_CALUDE_other_candidate_votes_l3558_355866


namespace NUMINAMATH_CALUDE_max_homes_first_neighborhood_l3558_355855

def revenue_first (n : ℕ) : ℕ := 4 * n

def revenue_second : ℕ := 50

theorem max_homes_first_neighborhood :
  ∀ n : ℕ, revenue_first n ≤ revenue_second → n ≤ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_max_homes_first_neighborhood_l3558_355855


namespace NUMINAMATH_CALUDE_weight_after_jogging_first_week_l3558_355883

/-- Calculates the weight after one week of jogging given the initial weight and weight loss. -/
def weight_after_one_week (initial_weight weight_loss : ℕ) : ℕ :=
  initial_weight - weight_loss

/-- Proves that given an initial weight of 92 kg and a weight loss of 56 kg in the first week,
    the weight after the first week is equal to 36 kg. -/
theorem weight_after_jogging_first_week :
  weight_after_one_week 92 56 = 36 := by
  sorry

#eval weight_after_one_week 92 56

end NUMINAMATH_CALUDE_weight_after_jogging_first_week_l3558_355883


namespace NUMINAMATH_CALUDE_negative_y_implies_m_gt_2_smallest_m_solution_l3558_355843

-- Define the equation
def equation (y m : ℝ) : Prop := 4 * y + 2 * m + 1 = 2 * y + 5

-- Define the inequality
def inequality (x m : ℝ) : Prop := x - 1 > (m * x + 1) / 2

theorem negative_y_implies_m_gt_2 :
  (∃ y, y < 0 ∧ equation y m) → m > 2 :=
sorry

theorem smallest_m_solution :
  m = 3 → (∀ x, inequality x m ↔ x < -3) :=
sorry

end NUMINAMATH_CALUDE_negative_y_implies_m_gt_2_smallest_m_solution_l3558_355843


namespace NUMINAMATH_CALUDE_car_speed_change_l3558_355870

theorem car_speed_change (V : ℝ) (x : ℝ) (h_V : V > 0) (h_x : x > 0) : 
  V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_change_l3558_355870


namespace NUMINAMATH_CALUDE_machine_precision_test_l3558_355809

-- Define the sample data
def sample_data : List (Float × Nat) := [(3.0, 2), (3.5, 6), (3.8, 9), (4.4, 7), (4.5, 1)]

-- Define the hypothesized variance
def sigma_0_squared : Float := 0.1

-- Define the significance level
def alpha : Float := 0.05

-- Define the degrees of freedom
def df : Nat := 24

-- Function to calculate sample variance
def calculate_sample_variance (data : List (Float × Nat)) : Float :=
  sorry

-- Function to calculate chi-square test statistic
def calculate_chi_square (sample_variance : Float) (n : Nat) (sigma_0_squared : Float) : Float :=
  sorry

-- Function to get critical value from chi-square distribution
def get_chi_square_critical (alpha : Float) (df : Nat) : Float :=
  sorry

theorem machine_precision_test (data : List (Float × Nat)) (alpha : Float) (df : Nat) (sigma_0_squared : Float) :
  let sample_variance := calculate_sample_variance data
  let chi_square_obs := calculate_chi_square sample_variance data.length sigma_0_squared
  let chi_square_crit := get_chi_square_critical alpha df
  chi_square_obs > chi_square_crit :=
by
  sorry

#check machine_precision_test sample_data alpha df sigma_0_squared

end NUMINAMATH_CALUDE_machine_precision_test_l3558_355809


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_l3558_355889

theorem largest_of_three_consecutive_integers (n : ℤ) 
  (h : (n - 1) + n + (n + 1) = 90) : 
  max (n - 1) (max n (n + 1)) = 31 := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_l3558_355889


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l3558_355878

-- Define the cost price and selling price
def cost_price : ℚ := 1500
def selling_price : ℚ := 1200

-- Define the loss percentage calculation
def loss_percentage (cp sp : ℚ) : ℚ := (cp - sp) / cp * 100

-- Theorem statement
theorem loss_percentage_calculation :
  loss_percentage cost_price selling_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l3558_355878


namespace NUMINAMATH_CALUDE_choir_average_age_l3558_355893

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12)
  (h2 : num_males = 13)
  (h3 : avg_age_females = 32)
  (h4 : avg_age_males = 33)
  (h5 : num_females + num_males = 25) :
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  let total_members := num_females + num_males
  total_age / total_members = 32.52 := by
sorry

end NUMINAMATH_CALUDE_choir_average_age_l3558_355893


namespace NUMINAMATH_CALUDE_line_through_first_third_quadrants_l3558_355892

/-- A line y = kx passes through the first and third quadrants if and only if k > 0 -/
theorem line_through_first_third_quadrants (k : ℝ) (h1 : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) ↔ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_third_quadrants_l3558_355892


namespace NUMINAMATH_CALUDE_emily_calculation_l3558_355863

theorem emily_calculation (n : ℕ) (h : n = 50) : n^2 - 99 = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_emily_calculation_l3558_355863


namespace NUMINAMATH_CALUDE_f_at_2_l3558_355825

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem f_at_2 : f 2 = 123 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l3558_355825


namespace NUMINAMATH_CALUDE_min_third_side_length_l3558_355848

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Checks if the given sides satisfy the triangle inequality -/
def satisfies_triangle_inequality (a b c : ℕ+) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem: The minimum length of the third side in a triangle with two sides
    being multiples of 42 and 72 respectively is 7 -/
theorem min_third_side_length (t : Triangle) 
    (h1 : ∃ (k : ℕ+), t.a = 42 * k ∨ t.b = 42 * k ∨ t.c = 42 * k)
    (h2 : ∃ (m : ℕ+), t.a = 72 * m ∨ t.b = 72 * m ∨ t.c = 72 * m)
    (h3 : satisfies_triangle_inequality t.a t.b t.c) :
    min t.a (min t.b t.c) ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_third_side_length_l3558_355848


namespace NUMINAMATH_CALUDE_cubic_sum_plus_eight_l3558_355861

theorem cubic_sum_plus_eight (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 8 = 978 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_plus_eight_l3558_355861


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l3558_355895

/-- Given a man's speed in still water and downstream speed, calculate his upstream speed -/
theorem mans_upstream_speed (v_still : ℝ) (v_downstream : ℝ) (h1 : v_still = 75) (h2 : v_downstream = 90) :
  v_still - (v_downstream - v_still) = 60 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_speed_l3558_355895


namespace NUMINAMATH_CALUDE_number_multiplication_problem_l3558_355838

theorem number_multiplication_problem (x : ℝ) : 15 * x = x + 196 → 15 * x = 210 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_problem_l3558_355838


namespace NUMINAMATH_CALUDE_square_plus_integer_equality_find_integer_l3558_355851

theorem square_plus_integer_equality (y : ℝ) : ∃ k : ℤ, y^2 + 12*y + 40 = (y + 6)^2 + k := by
  sorry

theorem find_integer : ∃ k : ℤ, ∀ y : ℝ, y^2 + 12*y + 40 = (y + 6)^2 + k ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_integer_equality_find_integer_l3558_355851


namespace NUMINAMATH_CALUDE_contradiction_proof_l3558_355879

theorem contradiction_proof (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (product_inequality : a * c + b * d > 1) 
  (all_nonnegative : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : 
  False := by
sorry

end NUMINAMATH_CALUDE_contradiction_proof_l3558_355879


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l3558_355899

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l3558_355899


namespace NUMINAMATH_CALUDE_square_of_integer_root_l3558_355822

theorem square_of_integer_root (n : ℕ) : 
  ∃ (m : ℤ), (2 : ℝ) + 2 * Real.sqrt (28 * (n^2 : ℝ) + 1) = m → 
  ∃ (k : ℤ), m = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_integer_root_l3558_355822


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3558_355834

theorem perfect_square_condition (y m : ℝ) : 
  (∃ k : ℝ, y^2 - 8*y + m = k^2) → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3558_355834


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l3558_355827

theorem arithmetic_progression_ratio (a d : ℝ) : 
  (15 * a + 105 * d = 4 * (8 * a + 28 * d)) → (a / d = -7 / 17) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l3558_355827


namespace NUMINAMATH_CALUDE_system_solution_l3558_355820

theorem system_solution : 
  ∀ x y : ℝ, 
    x^2 + 3*x*y = 18 ∧ x*y + 3*y^2 = 6 → 
      (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3558_355820


namespace NUMINAMATH_CALUDE_find_number_l3558_355816

theorem find_number : ∃ x : ℝ, (0.4 * x = 0.75 * 100 + 50) ∧ (x = 312.5) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3558_355816


namespace NUMINAMATH_CALUDE_percentage_of_students_liking_donuts_l3558_355882

theorem percentage_of_students_liking_donuts : 
  ∀ (total_donuts : ℕ) (total_students : ℕ) (donuts_per_student : ℕ),
    total_donuts = 4 * 12 →
    total_students = 30 →
    donuts_per_student = 2 →
    (((total_donuts / donuts_per_student) / total_students) * 100 : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_liking_donuts_l3558_355882


namespace NUMINAMATH_CALUDE_two_digit_number_insertion_theorem_l3558_355802

theorem two_digit_number_insertion_theorem :
  ∃! (S : Finset Nat),
    (∀ n ∈ S, 10 ≤ n ∧ n < 100) ∧
    (∀ n ∉ S, ¬(10 ≤ n ∧ n < 100)) ∧
    (∀ n ∈ S,
      ∃ d : Nat,
      d < 10 ∧
      (100 * (n / 10) + 10 * d + (n % 10) = 9 * n)) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_insertion_theorem_l3558_355802


namespace NUMINAMATH_CALUDE_fold_crease_length_l3558_355819

/-- Represents a rectangular sheet of paper -/
structure Sheet :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the crease formed by folding the sheet -/
def crease_length (s : Sheet) : ℝ :=
  sorry

/-- The theorem stating the length of the crease -/
theorem fold_crease_length (s : Sheet) 
  (h1 : s.length = 8) 
  (h2 : s.width = 6) : 
  crease_length s = 7.5 :=
sorry

end NUMINAMATH_CALUDE_fold_crease_length_l3558_355819


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l3558_355824

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the area of a specific triangle -/
theorem triangle_area_is_one (t : Triangle) 
  (h1 : (Real.cos t.B / t.b) + (Real.cos t.C / t.c) = (Real.sin t.A / (2 * Real.sin t.C)))
  (h2 : Real.sqrt 3 * t.b * Real.cos t.C = (2 * t.a - Real.sqrt 3 * t.c) * Real.cos t.B)
  (h3 : ∃ (r : ℝ), Real.sin t.A = r * Real.sin t.B ∧ Real.sin t.B = r * Real.sin t.C) :
  (1/2) * t.a * t.c * Real.sin t.B = 1 := by
  sorry

#check triangle_area_is_one

end NUMINAMATH_CALUDE_triangle_area_is_one_l3558_355824


namespace NUMINAMATH_CALUDE_symmetry_of_even_functions_l3558_355897

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a point
def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_even_functions :
  (∀ f : ℝ → ℝ, IsEven f → IsSymmetricAbout (fun x ↦ f (x + 2)) (-2)) ∧
  (∀ f : ℝ → ℝ, IsEven (fun x ↦ f (x + 2)) → IsSymmetricAbout f 2) := by
  sorry


end NUMINAMATH_CALUDE_symmetry_of_even_functions_l3558_355897


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3558_355832

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) → a ∈ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3558_355832


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3558_355860

theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3558_355860


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3558_355803

theorem water_tank_capacity : ∀ x : ℚ, 
  (3/4 : ℚ) * x - (1/3 : ℚ) * x = 15 → x = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3558_355803


namespace NUMINAMATH_CALUDE_part_one_part_two_l3558_355896

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x| ≥ 2}
def B (a : ℝ) : Set ℝ := {x | (x - 2*a)*(x + 3) < 0}

-- Part I
theorem part_one : 
  A 3 ∩ B 3 = {x | -3 < x ∧ x ≤ -2 ∨ 2 ≤ x ∧ x < 6} := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a > 0) :
  A a ∪ B a = Set.univ → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3558_355896


namespace NUMINAMATH_CALUDE_triangle_value_l3558_355852

theorem triangle_value (triangle q p : ℤ) 
  (eq1 : triangle + q = 73)
  (eq2 : 2 * (triangle + q) + p = 172)
  (eq3 : p = 26) : 
  triangle = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l3558_355852


namespace NUMINAMATH_CALUDE_vector_sum_l3558_355862

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b : ℝ → ℝ × ℝ := λ x ↦ (2, x)
def c : ℝ → ℝ × ℝ := λ m ↦ (m, -3)

-- Define the parallel and perpendicular conditions
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem vector_sum (x m : ℝ) 
  (h1 : parallel a (b x)) 
  (h2 : perpendicular (b x) (c m)) : 
  x + m = -10 := by sorry

end NUMINAMATH_CALUDE_vector_sum_l3558_355862


namespace NUMINAMATH_CALUDE_fraction_modification_l3558_355839

theorem fraction_modification (a b c d x : ℚ) : 
  a ≠ b →
  b ≠ 0 →
  (2 * a + x) / (3 * b + x) = c / d →
  ∃ (k₁ k₂ : ℚ), c = k₁ * x ∧ d = k₂ * x →
  x = (3 * b * c - 2 * a * d) / (d - c) := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l3558_355839


namespace NUMINAMATH_CALUDE_telephone_number_increase_l3558_355884

/-- The number of possible n-digit telephone numbers with a non-zero first digit -/
def telephone_numbers (n : ℕ) : ℕ := 9 * 10^(n - 1)

/-- The increase in telephone numbers when moving from 6 to 7 digits -/
def increase_in_numbers : ℕ := telephone_numbers 7 - telephone_numbers 6

theorem telephone_number_increase :
  increase_in_numbers = 81 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_telephone_number_increase_l3558_355884


namespace NUMINAMATH_CALUDE_chess_tournament_ordering_l3558_355872

/-- A structure representing a chess tournament -/
structure ChessTournament (N : ℕ) where
  beats : Fin N → Fin N → Prop

/-- The tournament property described in the problem -/
def has_tournament_property {N : ℕ} (M : ℕ) (t : ChessTournament N) : Prop :=
  ∀ (players : Fin (M + 1) → Fin N),
    (∀ i : Fin M, t.beats (players i) (players (i + 1))) →
    t.beats (players 0) (players M)

/-- The theorem to be proved -/
theorem chess_tournament_ordering
  {N M : ℕ} (h_N : N > M) (h_M : M > 1)
  (t : ChessTournament N)
  (h_prop : has_tournament_property M t) :
  ∃ f : Fin N ≃ Fin N,
    ∀ a b : Fin N, (a : ℕ) ≥ (b : ℕ) + M - 1 → t.beats (f a) (f b) :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_ordering_l3558_355872


namespace NUMINAMATH_CALUDE_z_is_real_z_is_imaginary_z_is_pure_imaginary_l3558_355857

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 12) (a^2 - 5*a + 6)

-- Theorem for real values of z
theorem z_is_real (a : ℝ) : (z a).im = 0 ↔ a = 2 ∨ a = 3 := by sorry

-- Theorem for imaginary values of z
theorem z_is_imaginary (a : ℝ) : (z a).im ≠ 0 ↔ a ≠ 2 ∧ a ≠ 3 := by sorry

-- Theorem for pure imaginary values of z
theorem z_is_pure_imaginary (a : ℝ) : (z a).re = 0 ∧ (z a).im ≠ 0 ↔ a = 4 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_imaginary_z_is_pure_imaginary_l3558_355857


namespace NUMINAMATH_CALUDE_plan_b_rate_l3558_355836

/-- Represents the cost of a call under Plan A -/
def costPlanA (minutes : ℕ) : ℚ :=
  if minutes ≤ 6 then 60/100
  else 60/100 + (minutes - 6) * (6/100)

/-- Represents the cost of a call under Plan B -/
def costPlanB (rate : ℚ) (minutes : ℕ) : ℚ :=
  rate * minutes

/-- The duration at which both plans charge the same amount -/
def equalDuration : ℕ := 12

theorem plan_b_rate : ∃ (rate : ℚ), 
  costPlanA equalDuration = costPlanB rate equalDuration ∧ rate = 8/100 := by
  sorry

end NUMINAMATH_CALUDE_plan_b_rate_l3558_355836


namespace NUMINAMATH_CALUDE_initial_worksheets_l3558_355826

theorem initial_worksheets (graded : ℕ) (new_worksheets : ℕ) (total : ℕ) :
  graded = 7 → new_worksheets = 36 → total = 63 →
  ∃ initial : ℕ, initial - graded + new_worksheets = total ∧ initial = 34 :=
by sorry

end NUMINAMATH_CALUDE_initial_worksheets_l3558_355826


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3558_355810

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (1, 2) and b = (-1, m), if they are perpendicular, then m = 1/2 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, m)
  perpendicular a b → m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3558_355810
