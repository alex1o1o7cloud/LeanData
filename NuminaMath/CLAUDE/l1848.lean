import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_two_squared_l1848_184847

theorem sqrt_two_squared : (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l1848_184847


namespace NUMINAMATH_CALUDE_ac_plus_bd_equals_23_l1848_184871

theorem ac_plus_bd_equals_23 
  (a b c d : ℝ) 
  (h1 : a + b + c = 6)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = -9) :
  a * c + b * d = 23 := by
sorry

end NUMINAMATH_CALUDE_ac_plus_bd_equals_23_l1848_184871


namespace NUMINAMATH_CALUDE_unique_seating_arrangement_l1848_184895

-- Define the types of representatives
inductive Representative
| Martian
| Venusian
| Earthling

-- Define the seating arrangement as a function from chair number to representative
def SeatingArrangement := Fin 10 → Representative

-- Define the rules for valid seating arrangements
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  -- Martian must occupy chair 1
  arr 0 = Representative.Martian ∧
  -- Earthling must occupy chair 10
  arr 9 = Representative.Earthling ∧
  -- Representatives must be arranged in clockwise order: Martian, Venusian, Earthling, repeating
  (∀ i : Fin 10, arr i = Representative.Martian → arr ((i + 1) % 10) = Representative.Venusian) ∧
  (∀ i : Fin 10, arr i = Representative.Venusian → arr ((i + 1) % 10) = Representative.Earthling) ∧
  (∀ i : Fin 10, arr i = Representative.Earthling → arr ((i + 1) % 10) = Representative.Martian)

-- Theorem stating that there is exactly one valid seating arrangement
theorem unique_seating_arrangement :
  ∃! arr : SeatingArrangement, is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_unique_seating_arrangement_l1848_184895


namespace NUMINAMATH_CALUDE_equation_solutions_l1848_184833

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  x^4 = 2*x^2 + (floor x)

/-- The set of solutions to the equation -/
def solution_set : Set ℝ :=
  {0, Real.sqrt (1 + Real.sqrt 2), -1}

/-- Theorem stating that the solution set contains exactly the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1848_184833


namespace NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l1848_184887

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The area of the trapezoid
  area : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- The length of one parallel side (shorter)
  base_short : ℝ
  -- The length of the other parallel side (longer)
  base_long : ℝ
  -- The length of the non-parallel sides (legs)
  leg : ℝ
  -- The trapezoid is isosceles
  isosceles : True
  -- The lines containing the legs intersect at a right angle
  right_angle_intersection : True
  -- The area is calculated correctly
  area_eq : area = (base_short + base_long) * height / 2

/-- Theorem about a specific isosceles trapezoid -/
theorem specific_trapezoid_dimensions :
  ∃ t : IsoscelesTrapezoid,
    t.area = 12 ∧
    t.height = 2 ∧
    t.base_short = 4 ∧
    t.base_long = 8 ∧
    t.leg = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_dimensions_l1848_184887


namespace NUMINAMATH_CALUDE_complex_power_difference_l1848_184864

theorem complex_power_difference (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^2187 - 1/(x^2187) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1848_184864


namespace NUMINAMATH_CALUDE_mod_19_equivalence_l1848_184819

theorem mod_19_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ 42568 % 19 = n % 19 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_19_equivalence_l1848_184819


namespace NUMINAMATH_CALUDE_ones_digit_of_33_power_power_of_3_cycle_power_mod_4_main_theorem_l1848_184815

theorem ones_digit_of_33_power (n : ℕ) : n > 0 → (33^n) % 10 = (3^n) % 10 := by sorry

theorem power_of_3_cycle (n : ℕ) : (3^n) % 10 = (3^(n % 4)) % 10 := by sorry

theorem power_mod_4 (a b : ℕ) : a > 0 → b > 0 → (a^b) % 4 = (a % 4)^(b % 4) % 4 := by sorry

theorem main_theorem : (33^(33 * 7^7)) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_33_power_power_of_3_cycle_power_mod_4_main_theorem_l1848_184815


namespace NUMINAMATH_CALUDE_fenced_field_area_l1848_184886

/-- A rectangular field with specific fencing requirements -/
structure FencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing : ℝ
  uncovered_side_eq : uncovered_side = 20
  fencing_eq : uncovered_side + 2 * width = fencing
  fencing_length : fencing = 88

/-- The area of a rectangular field -/
def field_area (f : FencedField) : ℝ :=
  f.length * f.width

/-- Theorem stating that a field with the given specifications has an area of 680 square feet -/
theorem fenced_field_area (f : FencedField) : field_area f = 680 := by
  sorry

end NUMINAMATH_CALUDE_fenced_field_area_l1848_184886


namespace NUMINAMATH_CALUDE_beatrix_books_l1848_184823

theorem beatrix_books (beatrix alannah queen : ℕ) 
  (h1 : alannah = beatrix + 20)
  (h2 : queen = alannah + alannah / 5)
  (h3 : beatrix + alannah + queen = 140) : 
  beatrix = 30 := by
  sorry

end NUMINAMATH_CALUDE_beatrix_books_l1848_184823


namespace NUMINAMATH_CALUDE_distance_between_stations_l1848_184849

/-- The distance between two stations given train travel times and speeds -/
theorem distance_between_stations 
  (train1_speed : ℝ) (train1_time : ℝ) 
  (train2_speed : ℝ) (train2_time : ℝ) 
  (h1 : train1_speed = 20)
  (h2 : train1_time = 5)
  (h3 : train2_speed = 25)
  (h4 : train2_time = 4) :
  train1_speed * train1_time + train2_speed * train2_time = 200 := by
  sorry

#check distance_between_stations

end NUMINAMATH_CALUDE_distance_between_stations_l1848_184849


namespace NUMINAMATH_CALUDE_john_square_calculation_l1848_184827

theorem john_square_calculation (n : ℕ) (h : n = 50) :
  n^2 + 101 = (n + 1)^2 → n^2 - 99 = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_john_square_calculation_l1848_184827


namespace NUMINAMATH_CALUDE_Q_equals_G_l1848_184897

-- Define the sets Q and G
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_G : Q = G := by sorry

end NUMINAMATH_CALUDE_Q_equals_G_l1848_184897


namespace NUMINAMATH_CALUDE_problem_statement_l1848_184874

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1848_184874


namespace NUMINAMATH_CALUDE_stock_worth_l1848_184810

theorem stock_worth (X : ℝ) : 
  (0.2 * X * 1.1 + 0.8 * X * 0.95) - X = -250 → X = 12500 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_l1848_184810


namespace NUMINAMATH_CALUDE_inequality_proof_l1848_184838

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1848_184838


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l1848_184839

theorem quadratic_two_roots (a b c : ℝ) (h1 : b > a + c) (h2 : a + c > 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l1848_184839


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1848_184857

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1848_184857


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1848_184898

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  n = Nat.gcd (1557 - 7) (Nat.gcd (2037 - 5) (2765 - 9)) ∧
  1557 % n = 7 ∧
  2037 % n = 5 ∧
  2765 % n = 9 ∧
  ∀ (m : ℕ), m > n → 
    (1557 % m = 7 ∧ 2037 % m = 5 ∧ 2765 % m = 9) → False :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1848_184898


namespace NUMINAMATH_CALUDE_sequence_sum_l1848_184883

theorem sequence_sum (A B C D E F G H : ℤ) : 
  C = 3 ∧ 
  A + B + C = 27 ∧
  B + C + D = 27 ∧
  C + D + E = 27 ∧
  D + E + F = 27 ∧
  E + F + G = 27 ∧
  F + G + H = 27 →
  A + H = 27 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l1848_184883


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1848_184829

theorem reciprocal_sum_theorem (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1 / x + 1 / y = 1 / z) : z = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1848_184829


namespace NUMINAMATH_CALUDE_green_ball_probability_l1848_184875

-- Define the containers and their contents
def containerA : ℕ × ℕ := (10, 5)  -- (red balls, green balls)
def containerB : ℕ × ℕ := (3, 6)
def containerC : ℕ × ℕ := (3, 6)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Define the probability of selecting a green ball from each container
def greenProbA : ℚ := containerA.2 / (containerA.1 + containerA.2)
def greenProbB : ℚ := containerB.2 / (containerB.1 + containerB.2)
def greenProbC : ℚ := containerC.2 / (containerC.1 + containerC.2)

-- Theorem: The probability of selecting a green ball is 5/9
theorem green_ball_probability :
  containerProb * greenProbA +
  containerProb * greenProbB +
  containerProb * greenProbC = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1848_184875


namespace NUMINAMATH_CALUDE_photo_difference_l1848_184813

/-- The number of photos taken by Lisa -/
def L : ℕ := 50

/-- The number of photos taken by Mike -/
def M : ℕ := sorry

/-- The number of photos taken by Norm -/
def N : ℕ := 110

/-- The total of Lisa and Mike's photos is less than the sum of Mike's and Norm's -/
axiom photo_sum_inequality : L + M < M + N

/-- Norm's photos are 10 more than twice Lisa's photos -/
axiom norm_photos_relation : N = 2 * L + 10

theorem photo_difference : (M + N) - (L + M) = 60 := by sorry

end NUMINAMATH_CALUDE_photo_difference_l1848_184813


namespace NUMINAMATH_CALUDE_win_in_four_moves_cannot_win_in_ten_moves_min_moves_for_2018_l1848_184877

/-- Represents the state of the game --/
structure GameState where
  coinsA : ℕ  -- Coins in box A
  coinsB : ℕ  -- Coins in box B

/-- Defines a single move in the game --/
inductive Move
  | MoveToB    : Move  -- Move a coin from A to B
  | RemoveFromA : Move  -- Remove coins from A equal to coins in B

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.MoveToB => 
      { coinsA := state.coinsA - 1, coinsB := state.coinsB + 1 }
  | Move.RemoveFromA => 
      { coinsA := state.coinsA - state.coinsB, coinsB := state.coinsB }

/-- Checks if the game is won (i.e., box A is empty) --/
def isGameWon (state : GameState) : Prop := state.coinsA = 0

/-- Theorem: For 6 initial coins, the game can be won in 4 moves --/
theorem win_in_four_moves (initialCoins : ℕ) (h : initialCoins = 6) : 
  ∃ (moves : List Move), moves.length = 4 ∧ 
    isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 }) := by
  sorry

/-- Theorem: For 31 initial coins, the game cannot be won in 10 moves --/
theorem cannot_win_in_ten_moves (initialCoins : ℕ) (h : initialCoins = 31) :
  ∀ (moves : List Move), moves.length = 10 → 
    ¬isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 }) := by
  sorry

/-- Theorem: For 2018 initial coins, the minimum number of moves to win is 89 --/
theorem min_moves_for_2018 (initialCoins : ℕ) (h : initialCoins = 2018) :
  (∃ (moves : List Move), moves.length = 89 ∧ 
    isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 })) ∧
  (∀ (moves : List Move), moves.length < 89 → 
    ¬isGameWon (moves.foldl applyMove { coinsA := initialCoins, coinsB := 0 })) := by
  sorry

end NUMINAMATH_CALUDE_win_in_four_moves_cannot_win_in_ten_moves_min_moves_for_2018_l1848_184877


namespace NUMINAMATH_CALUDE_simplify_expression_l1848_184825

theorem simplify_expression (x : ℝ) : (3*x + 15) + (100*x + 15) + (10*x - 5) = 113*x + 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1848_184825


namespace NUMINAMATH_CALUDE_second_rate_is_five_percent_l1848_184890

-- Define the total amount, first part, and interest rates
def total_amount : ℚ := 3200
def first_part : ℚ := 800
def first_rate : ℚ := 3 / 100
def total_interest : ℚ := 144

-- Define the second part
def second_part : ℚ := total_amount - first_part

-- Define the interest from the first part
def interest_first : ℚ := first_part * first_rate

-- Define the interest from the second part
def interest_second : ℚ := total_interest - interest_first

-- Define the interest rate of the second part
def second_rate : ℚ := interest_second / second_part

-- Theorem to prove
theorem second_rate_is_five_percent : second_rate = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_rate_is_five_percent_l1848_184890


namespace NUMINAMATH_CALUDE_courier_strategy_l1848_184811

-- Define the probability of a courier being robbed
variable (p : ℝ) 

-- Define the probability of failure for each strategy
def P2 : ℝ := p^2
def P3 : ℝ := p^2 * (3 - 2*p)
def P4 : ℝ := p^3 * (4 - 3*p)

-- Define the theorem
theorem courier_strategy (h1 : 0 < p) (h2 : p < 1) :
  (0 < p ∧ p < 1/3 → 1 - P4 p > max (1 - P2 p) (1 - P3 p)) ∧
  (1/3 ≤ p ∧ p < 1 → 1 - P2 p ≥ max (1 - P3 p) (1 - P4 p)) :=
sorry

end NUMINAMATH_CALUDE_courier_strategy_l1848_184811


namespace NUMINAMATH_CALUDE_complex_modulus_one_l1848_184896

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l1848_184896


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1848_184841

theorem digit_sum_problem :
  ∃ (a b c d : ℕ),
    (1000 ≤ a ∧ a < 10000) ∧
    (1000 ≤ b ∧ b < 10000) ∧
    (1000 ≤ c ∧ c < 10000) ∧
    (1000 ≤ d ∧ d < 10000) ∧
    a + b = 4300 ∧
    c - d = 1542 ∧
    a + c = 5842 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1848_184841


namespace NUMINAMATH_CALUDE_sons_present_age_l1848_184873

theorem sons_present_age (son_age father_age : ℕ) : 
  father_age = son_age + 45 →
  father_age + 10 = 4 * (son_age + 10) →
  son_age + 15 = 2 * son_age →
  son_age = 15 := by
sorry

end NUMINAMATH_CALUDE_sons_present_age_l1848_184873


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1848_184803

/-- A geometric sequence with fifth term 48 and sixth term 72 has second term 384/27 -/
theorem geometric_sequence_second_term :
  ∀ (a : ℚ) (r : ℚ),
  a * r^4 = 48 →
  a * r^5 = 72 →
  a * r = 384/27 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1848_184803


namespace NUMINAMATH_CALUDE_car_rental_budget_is_75_l1848_184830

/-- Calculates the budget for a car rental given the daily rate, per-mile rate, and miles driven. -/
def carRentalBudget (dailyRate : ℝ) (perMileRate : ℝ) (milesDriven : ℝ) : ℝ :=
  dailyRate + perMileRate * milesDriven

/-- Theorem: The budget for a car rental with specific rates and mileage is $75.00. -/
theorem car_rental_budget_is_75 :
  let dailyRate : ℝ := 30
  let perMileRate : ℝ := 0.18
  let milesDriven : ℝ := 250.0
  carRentalBudget dailyRate perMileRate milesDriven = 75 := by
  sorry

end NUMINAMATH_CALUDE_car_rental_budget_is_75_l1848_184830


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_correct_l1848_184817

/-- A natural number is a perfect square if it's equal to some natural number squared. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- A natural number is a perfect cube if it's equal to some natural number cubed. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- The smallest natural number that satisfies the given conditions. -/
def SmallestSatisfyingNumber : ℕ := 216

/-- Theorem stating that SmallestSatisfyingNumber is the smallest natural number
    that when multiplied by 2 becomes a perfect square and
    when multiplied by 3 becomes a perfect cube. -/
theorem smallest_satisfying_number_correct :
  (IsPerfectSquare (2 * SmallestSatisfyingNumber)) ∧
  (IsPerfectCube (3 * SmallestSatisfyingNumber)) ∧
  (∀ n : ℕ, n < SmallestSatisfyingNumber →
    ¬(IsPerfectSquare (2 * n) ∧ IsPerfectCube (3 * n))) := by
  sorry

#eval SmallestSatisfyingNumber -- Should output 216

end NUMINAMATH_CALUDE_smallest_satisfying_number_correct_l1848_184817


namespace NUMINAMATH_CALUDE_line_equation_conversion_l1848_184853

/-- Given a line expressed as (2, -1) · ((x, y) - (5, -3)) = 0, prove that when written in the form y = mx + b, m = 2 and b = -13 -/
theorem line_equation_conversion :
  ∀ (x y : ℝ),
  (2 : ℝ) * (x - 5) + (-1 : ℝ) * (y - (-3)) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -13 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_conversion_l1848_184853


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l1848_184894

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_of_factorials_15 :
  sum_of_factorials 15 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l1848_184894


namespace NUMINAMATH_CALUDE_ball_probability_l1848_184804

theorem ball_probability (total white green yellow red purple black blue pink : ℕ) 
  (h_total : total = 500)
  (h_white : white = 200)
  (h_green : green = 80)
  (h_yellow : yellow = 70)
  (h_red : red = 57)
  (h_purple : purple = 33)
  (h_black : black = 30)
  (h_blue : blue = 16)
  (h_pink : pink = 14)
  (h_sum : white + green + yellow + red + purple + black + blue + pink = total) :
  (total - (red + purple + black)) / total = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1848_184804


namespace NUMINAMATH_CALUDE_sqrt_less_than_y_plus_one_l1848_184881

theorem sqrt_less_than_y_plus_one (y : ℝ) (h : y > 0) : Real.sqrt y < y + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_y_plus_one_l1848_184881


namespace NUMINAMATH_CALUDE_percentage_of_girls_l1848_184828

theorem percentage_of_girls (total_students : ℕ) (num_boys : ℕ) : 
  total_students = 400 → num_boys = 80 → 
  (((total_students - num_boys : ℚ) / total_students) * 100 : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l1848_184828


namespace NUMINAMATH_CALUDE_nonzero_real_equality_l1848_184808

theorem nonzero_real_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 1 + 1/y) (h2 : y = 1 + 1/x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_equality_l1848_184808


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l1848_184845

/-- The number of flips in the sequence -/
def num_flips : ℕ := 10

/-- The number of fixed flips (fifth and sixth must be heads) -/
def fixed_flips : ℕ := 2

/-- The number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- 
Theorem: The number of distinct sequences of coin flips, 
where two specific flips are fixed, is equal to 2^(total flips - fixed flips)
-/
theorem coin_flip_sequences : 
  outcomes_per_flip ^ (num_flips - fixed_flips) = 256 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l1848_184845


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l1848_184891

/-- The line equation mx - y + 2m + 1 = 0 passes through the point (-2, 1) for all values of m. -/
theorem fixed_point_of_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l1848_184891


namespace NUMINAMATH_CALUDE_crates_in_load_l1848_184867

/-- Represents the weight of vegetables in a delivery truck load --/
structure VegetableLoad where
  crateWeight : ℕ     -- Weight of one crate in kilograms
  cartonWeight : ℕ    -- Weight of one carton in kilograms
  numCartons : ℕ      -- Number of cartons in the load
  totalWeight : ℕ     -- Total weight of the load in kilograms

/-- Calculates the number of crates in a vegetable load --/
def numCrates (load : VegetableLoad) : ℕ :=
  (load.totalWeight - load.cartonWeight * load.numCartons) / load.crateWeight

/-- Theorem stating that for the given conditions, the number of crates is 12 --/
theorem crates_in_load :
  ∀ (load : VegetableLoad),
    load.crateWeight = 4 →
    load.cartonWeight = 3 →
    load.numCartons = 16 →
    load.totalWeight = 96 →
    numCrates load = 12 := by
  sorry

end NUMINAMATH_CALUDE_crates_in_load_l1848_184867


namespace NUMINAMATH_CALUDE_binomial_17_4_l1848_184860

theorem binomial_17_4 : Nat.choose 17 4 = 2380 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_4_l1848_184860


namespace NUMINAMATH_CALUDE_log_equation_solution_l1848_184814

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9 →
  x = 2^(54/5) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1848_184814


namespace NUMINAMATH_CALUDE_obtuse_angles_are_second_quadrant_l1848_184846

-- Define angle types
def ObtuseAngle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def SecondQuadrantAngle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def FirstQuadrantAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def ThirdQuadrantAngle (θ : ℝ) : Prop := -180 < θ ∧ θ < -90

-- Theorem statement
theorem obtuse_angles_are_second_quadrant :
  ∀ θ : ℝ, ObtuseAngle θ ↔ SecondQuadrantAngle θ := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angles_are_second_quadrant_l1848_184846


namespace NUMINAMATH_CALUDE_lottery_probability_l1848_184802

/-- Represents the lottery setup -/
structure LotterySetup where
  total_people : Nat
  total_tickets : Nat
  winning_tickets : Nat

/-- Calculates the probability of the lottery ending after a specific draw -/
def probability_end_after_draw (setup : LotterySetup) (draw : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem lottery_probability (setup : LotterySetup) :
  setup.total_people = 5 →
  setup.total_tickets = 5 →
  setup.winning_tickets = 3 →
  probability_end_after_draw setup 4 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l1848_184802


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1848_184801

theorem adult_ticket_cost (total_tickets : ℕ) (senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  total_tickets = 510 →
  senior_price = 15 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  (total_tickets - senior_tickets) * (total_receipts - senior_tickets * senior_price) / (total_tickets - senior_tickets) = 21 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1848_184801


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1848_184842

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (l m : Line) 
  (α β : Plane) 
  (h_diff_lines : l ≠ m) 
  (h_diff_planes : α ≠ β) 
  (h_l_perp_m : perp_line l m) 
  (h_l_perp_α : perp_line_plane l α) 
  (h_m_perp_β : perp_line_plane m β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1848_184842


namespace NUMINAMATH_CALUDE_half_volume_convex_hull_cube_l1848_184885

theorem half_volume_convex_hull_cube : ∃ a : ℝ, 0 < a ∧ a < 1 ∧ 
  2 * (a^3 + (1-a)^3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_half_volume_convex_hull_cube_l1848_184885


namespace NUMINAMATH_CALUDE_jimmy_change_l1848_184872

def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5

def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def num_folders : ℕ := 2

def bill_amount : ℕ := 50

theorem jimmy_change :
  bill_amount - (num_pens * pen_cost + num_notebooks * notebook_cost + num_folders * folder_cost) = 25 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_change_l1848_184872


namespace NUMINAMATH_CALUDE_number_of_literate_employees_l1848_184854

def number_of_illiterate_employees : ℕ := 35
def wage_decrease_per_illiterate : ℕ := 25
def total_wage_decrease : ℕ := 875
def average_salary_decrease : ℕ := 15
def total_employees : ℕ := 58

theorem number_of_literate_employees :
  total_employees - number_of_illiterate_employees = 23 :=
by sorry

end NUMINAMATH_CALUDE_number_of_literate_employees_l1848_184854


namespace NUMINAMATH_CALUDE_bracelet_cost_calculation_josh_bracelet_cost_l1848_184882

theorem bracelet_cost_calculation (bracelet_price : ℝ) (num_bracelets : ℕ) (cookie_cost : ℝ) (money_left : ℝ) : ℝ :=
  let total_earned := bracelet_price * num_bracelets
  let total_after_cookies := cookie_cost + money_left
  let supply_cost := (total_earned - total_after_cookies) / num_bracelets
  supply_cost

theorem josh_bracelet_cost :
  bracelet_cost_calculation 1.5 12 3 3 = 1 := by sorry

end NUMINAMATH_CALUDE_bracelet_cost_calculation_josh_bracelet_cost_l1848_184882


namespace NUMINAMATH_CALUDE_saree_price_calculation_l1848_184862

theorem saree_price_calculation (final_price : ℝ) : 
  final_price = 248.625 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.15) * (1 - 0.25) = final_price ∧ 
    original_price = 390 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l1848_184862


namespace NUMINAMATH_CALUDE_binary_111_equals_7_l1848_184812

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 111 -/
def binary_111 : List Bool := [true, true, true]

theorem binary_111_equals_7 : binary_to_decimal binary_111 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_111_equals_7_l1848_184812


namespace NUMINAMATH_CALUDE_cat_or_bird_percentage_l1848_184876

/-- Represents the survey data from a high school -/
structure SurveyData where
  total_students : ℕ
  dog_owners : ℕ
  cat_owners : ℕ
  bird_owners : ℕ

/-- Calculates the percentage of students owning either cats or birds -/
def percentage_cat_or_bird (data : SurveyData) : ℚ :=
  (data.cat_owners + data.bird_owners : ℚ) / data.total_students * 100

/-- The survey data from the high school -/
def high_school_survey : SurveyData :=
  { total_students := 400
  , dog_owners := 80
  , cat_owners := 50
  , bird_owners := 20 }

/-- Theorem stating that the percentage of students owning either cats or birds is 17.5% -/
theorem cat_or_bird_percentage :
  percentage_cat_or_bird high_school_survey = 35/2 := by
  sorry

end NUMINAMATH_CALUDE_cat_or_bird_percentage_l1848_184876


namespace NUMINAMATH_CALUDE_right_triangle_from_leg_and_projection_l1848_184840

/-- Right triangle determined by one leg and projection of other leg onto hypotenuse -/
theorem right_triangle_from_leg_and_projection
  (a c₂ : ℝ) (ha : a > 0) (hc₂ : c₂ > 0) :
  ∃! (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    c₂ * c = b^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_leg_and_projection_l1848_184840


namespace NUMINAMATH_CALUDE_circle_and_inscribed_square_l1848_184869

/-- Given a circle with circumference 72π and an inscribed square with vertices touching the circle,
    prove that the radius is 36 and the side length of the square is 36√2. -/
theorem circle_and_inscribed_square (C : ℝ) (r : ℝ) (s : ℝ) :
  C = 72 * Real.pi →  -- Circumference of the circle
  C = 2 * Real.pi * r →  -- Relation between circumference and radius
  s^2 * 2 = (2 * r)^2 →  -- Relation between square side and circle diameter
  r = 36 ∧ s = 36 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_and_inscribed_square_l1848_184869


namespace NUMINAMATH_CALUDE_distance_traveled_downstream_l1848_184884

/-- Calculate the distance traveled downstream by a boat -/
theorem distance_traveled_downstream 
  (boat_speed : ℝ) 
  (current_speed : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : current_speed = 5)
  (h3 : time_minutes = 24) :
  let effective_speed := boat_speed + current_speed
  let time_hours := time_minutes / 60
  effective_speed * time_hours = 10 := by
sorry

end NUMINAMATH_CALUDE_distance_traveled_downstream_l1848_184884


namespace NUMINAMATH_CALUDE_tan_product_ninth_pi_l1848_184821

theorem tan_product_ninth_pi : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_ninth_pi_l1848_184821


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1848_184889

theorem sum_of_two_numbers (s l : ℝ) : 
  s = 3.5 →
  l = 3 * s →
  s + l = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1848_184889


namespace NUMINAMATH_CALUDE_optimal_speed_l1848_184835

/-- Fuel cost per unit time as a function of speed -/
noncomputable def fuel_cost (v : ℝ) : ℝ := sorry

/-- Total cost per unit time as a function of speed -/
noncomputable def total_cost (v : ℝ) : ℝ := fuel_cost v + 560

/-- Cost per kilometer as a function of speed -/
noncomputable def cost_per_km (v : ℝ) : ℝ := total_cost v / v

theorem optimal_speed :
  ∃ (k : ℝ), fuel_cost v = k * v^3 ∧  -- Fuel cost is proportional to v^3
  fuel_cost 10 = 35 ∧                 -- At 10 km/h, fuel cost is 35 yuan/hour
  (∀ v, v ≤ 25) →                     -- Maximum speed is 25 km/h
  (∀ v, v > 0 → v ≤ 25 → cost_per_km 20 ≤ cost_per_km v) :=
sorry

end NUMINAMATH_CALUDE_optimal_speed_l1848_184835


namespace NUMINAMATH_CALUDE_x_range_l1848_184865

theorem x_range (x : ℝ) 
  (h : ∀ a b : ℝ, a^2 + b^2 = 1 → a + Real.sqrt 3 * b ≤ |x^2 - 1|) : 
  x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_x_range_l1848_184865


namespace NUMINAMATH_CALUDE_parallel_transitivity_false_l1848_184805

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- State the theorem
theorem parallel_transitivity_false 
  (l m : Line) (α : Plane) : 
  (parallel_line_plane l α ∧ parallel_line_plane m α) → 
  parallel_line_line l m :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_false_l1848_184805


namespace NUMINAMATH_CALUDE_rectangle_probability_in_n_gon_l1848_184888

theorem rectangle_probability_in_n_gon (n : ℕ) (h1 : Even n) (h2 : n > 4) :
  let P := (3 : ℚ) / ((n - 1) * (n - 3))
  P = (Nat.choose (n / 2) 2 : ℚ) / (Nat.choose n 4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_probability_in_n_gon_l1848_184888


namespace NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l1848_184868

theorem pascal_triangle_51_numbers (n : ℕ) : 
  (n + 1 = 51) → Nat.choose n 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l1848_184868


namespace NUMINAMATH_CALUDE_f_of_one_equals_fourteen_l1848_184863

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

-- State the theorem
theorem f_of_one_equals_fourteen 
  (a b : ℝ) -- Parameters of the function
  (h : f a b (-1) = 10) -- Given condition
  : f a b 1 = 14 := by
  sorry -- Proof is omitted

end NUMINAMATH_CALUDE_f_of_one_equals_fourteen_l1848_184863


namespace NUMINAMATH_CALUDE_pattern_equality_l1848_184832

theorem pattern_equality (n : ℕ) (h : n > 1) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_pattern_equality_l1848_184832


namespace NUMINAMATH_CALUDE_square_dissection_divisible_perimeter_l1848_184826

theorem square_dissection_divisible_perimeter (n : Nat) (h : n = 2015) :
  ∃ (a b : Nat), a ≤ n ∧ b ≤ n ∧ (2 * (a + b)) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_dissection_divisible_perimeter_l1848_184826


namespace NUMINAMATH_CALUDE_triple_base_square_exponent_l1848_184809

theorem triple_base_square_exponent 
  (a b y : ℝ) 
  (hb : b ≠ 0) 
  (hr : (3 * a) ^ (2 * b) = a ^ b * y ^ b) : 
  y = 9 * a := 
sorry

end NUMINAMATH_CALUDE_triple_base_square_exponent_l1848_184809


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l1848_184834

theorem soccer_team_lineup_count :
  let total_players : ℕ := 18
  let goalie_count : ℕ := 1
  let defender_count : ℕ := 6
  let forward_count : ℕ := 4
  let remaining_after_goalie : ℕ := total_players - goalie_count
  let remaining_after_defenders : ℕ := remaining_after_goalie - defender_count
  (total_players.choose goalie_count) *
  (remaining_after_goalie.choose defender_count) *
  (remaining_after_defenders.choose forward_count) = 73457760 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l1848_184834


namespace NUMINAMATH_CALUDE_circle_equation_with_hyperbola_asymptotes_as_tangents_l1848_184893

/-- The standard equation of a circle with center (0,5) and tangents that are the asymptotes of the hyperbola x^2 - y^2 = 1 -/
theorem circle_equation_with_hyperbola_asymptotes_as_tangents :
  ∃ (r : ℝ),
    (∀ (x y : ℝ), x^2 + (y - 5)^2 = r^2 ↔
      (∃ (t : ℝ), (x = t ∧ y = t + 5) ∨ (x = -t ∧ y = -t + 5))) ∧
    r^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_hyperbola_asymptotes_as_tangents_l1848_184893


namespace NUMINAMATH_CALUDE_george_coin_value_l1848_184822

/-- Calculates the total value of coins given the number of nickels and dimes -/
def totalCoinValue (totalCoins : ℕ) (nickels : ℕ) (nickelValue : ℚ) (dimeValue : ℚ) : ℚ :=
  let dimes := totalCoins - nickels
  nickels * nickelValue + dimes * dimeValue

theorem george_coin_value :
  totalCoinValue 28 4 (5 / 100) (10 / 100) = 260 / 100 := by
  sorry

end NUMINAMATH_CALUDE_george_coin_value_l1848_184822


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1848_184856

theorem tan_alpha_value (α : Real) :
  2 * Real.cos (π / 2 - α) - Real.sin (3 * π / 2 + α) = -Real.sqrt 5 →
  Real.tan α = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1848_184856


namespace NUMINAMATH_CALUDE_frosting_cans_needed_l1848_184843

def cakes_day1 : ℕ := 7
def cakes_day2 : ℕ := 12
def cakes_day3 : ℕ := 8
def cakes_day4 : ℕ := 10
def cakes_day5 : ℕ := 15
def cakes_eaten : ℕ := 18
def frosting_per_cake : ℕ := 3

def total_cakes : ℕ := cakes_day1 + cakes_day2 + cakes_day3 + cakes_day4 + cakes_day5
def remaining_cakes : ℕ := total_cakes - cakes_eaten

theorem frosting_cans_needed : remaining_cakes * frosting_per_cake = 102 := by
  sorry

end NUMINAMATH_CALUDE_frosting_cans_needed_l1848_184843


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l1848_184858

theorem least_three_digit_multiple_of_nine : 
  ∀ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ n % 9 = 0 → n ≥ 108 :=
by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l1848_184858


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1848_184892

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 28) :
  (face_perimeter / 4) ^ 3 = 343 := by
  sorry

#check cube_volume_from_face_perimeter

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1848_184892


namespace NUMINAMATH_CALUDE_complex_equation_first_quadrant_l1848_184879

/-- Given a complex equation, prove the resulting point is in the first quadrant -/
theorem complex_equation_first_quadrant (a b : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = b + Complex.I → 
  a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_first_quadrant_l1848_184879


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1848_184837

/-- Given an ellipse with equation 16x^2 + 9y^2 = 144, its major axis length is 8 -/
theorem ellipse_major_axis_length :
  ∀ (x y : ℝ), 16 * x^2 + 9 * y^2 = 144 → ∃ (a b : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    ((a ≥ b ∧ 2 * a = 8) ∨ (b > a ∧ 2 * b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1848_184837


namespace NUMINAMATH_CALUDE_mult_func_property_l1848_184880

/-- A function satisfying f(a+b) = f(a) * f(b) for all real a, b -/
def MultFunc (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a * f b

theorem mult_func_property (f : ℝ → ℝ) (h1 : MultFunc f) (h2 : f 1 = 2) :
  f 0 + f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mult_func_property_l1848_184880


namespace NUMINAMATH_CALUDE_auction_starting_value_l1848_184844

/-- The starting value of an auction satisfying the given conditions -/
def auctionStartingValue : ℝ → Prop := fun S =>
  let harryFirstBid := S + 200
  let secondBidderBid := 2 * harryFirstBid
  let thirdBidderBid := secondBidderBid + 3 * harryFirstBid
  let harryFinalBid := 4000
  harryFinalBid = thirdBidderBid + 1500

theorem auction_starting_value : ∃ S, auctionStartingValue S ∧ S = 300 := by
  sorry

end NUMINAMATH_CALUDE_auction_starting_value_l1848_184844


namespace NUMINAMATH_CALUDE_partnership_profit_distribution_l1848_184848

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution 
  (total_profit : ℝ) 
  (h_profit : total_profit = 55000) 
  (invest_a invest_b invest_c : ℝ) 
  (h_a_b : invest_a = 3 * invest_b) 
  (h_a_c : invest_a = 2/3 * invest_c) : 
  invest_c / (invest_a + invest_b + invest_c) * total_profit = 9/17 * 55000 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_distribution_l1848_184848


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1848_184818

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 3 + a 6 + a 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1848_184818


namespace NUMINAMATH_CALUDE_carter_road_trip_l1848_184824

/-- The duration of Carter's road trip without pit stops -/
def road_trip_duration : ℝ := 13.33

/-- The theorem stating the duration of Carter's road trip without pit stops -/
theorem carter_road_trip :
  let stop_interval : ℝ := 2 -- Hours between leg-stretching stops
  let food_stops : ℕ := 2 -- Number of additional food stops
  let gas_stops : ℕ := 3 -- Number of additional gas stops
  let pit_stop_duration : ℝ := 1/3 -- Duration of each pit stop in hours (20 minutes)
  let total_trip_duration : ℝ := 18 -- Total trip duration including pit stops in hours
  
  road_trip_duration = total_trip_duration - 
    (⌊total_trip_duration / stop_interval⌋ + food_stops + gas_stops) * pit_stop_duration :=
by
  sorry

end NUMINAMATH_CALUDE_carter_road_trip_l1848_184824


namespace NUMINAMATH_CALUDE_system_solution_l1848_184800

theorem system_solution :
  ∃! (x y : ℝ), 
    (x + 2*y = (7 - x) + (7 - 2*y)) ∧ 
    (3*x - 2*y = (x + 2) - (2*y + 2)) ∧
    x = 0 ∧ y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1848_184800


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1848_184855

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1848_184855


namespace NUMINAMATH_CALUDE_positive_X_value_l1848_184820

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) (h : hash X 7 = 338) : X = 17 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l1848_184820


namespace NUMINAMATH_CALUDE_T_simplification_l1848_184859

theorem T_simplification (x : ℝ) : 
  (x - 2)^4 + 8*(x - 2)^3 + 24*(x - 2)^2 + 32*(x - 2) + 16 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_T_simplification_l1848_184859


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1848_184852

theorem imaginary_part_of_z (z : ℂ) (h : z * ((1 + Complex.I)^2 / 2) = 1 + 2 * Complex.I) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1848_184852


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1848_184851

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (3 * n) % 30 = 2412 % 30 ∧ ∀ (m : ℕ), m > 0 → (3 * m) % 30 = 2412 % 30 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1848_184851


namespace NUMINAMATH_CALUDE_maria_green_towels_l1848_184878

/-- The number of green towels Maria bought -/
def green_towels : ℕ := sorry

/-- The total number of towels Maria had initially -/
def total_towels : ℕ := green_towels + 44

/-- The number of towels Maria had after giving some away -/
def remaining_towels : ℕ := total_towels - 65

theorem maria_green_towels : green_towels = 40 :=
  by
    have h1 : remaining_towels = 19 := sorry
    sorry

#check maria_green_towels

end NUMINAMATH_CALUDE_maria_green_towels_l1848_184878


namespace NUMINAMATH_CALUDE_g_72_value_l1848_184870

-- Define the properties of function g
def PositiveInteger (n : ℕ) : Prop := n > 0

def g_properties (g : ℕ → ℕ) : Prop :=
  (∀ n, PositiveInteger n → PositiveInteger (g n)) ∧
  (∀ n, PositiveInteger n → g (n + 1) > g n) ∧
  (∀ m n, PositiveInteger m → PositiveInteger n → g (m * n) = g m * g n) ∧
  (∀ m n, m ≠ n → m^n = n^m → (g m = 2*n ∨ g n = 2*m))

-- Theorem statement
theorem g_72_value (g : ℕ → ℕ) (h : g_properties g) : g 72 = 294912 := by
  sorry

end NUMINAMATH_CALUDE_g_72_value_l1848_184870


namespace NUMINAMATH_CALUDE_function_value_at_three_l1848_184850

theorem function_value_at_three (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l1848_184850


namespace NUMINAMATH_CALUDE_third_piece_coverage_l1848_184807

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a piece that covers some number of squares -/
structure Piece :=
  (squares_covered : ℕ)

/-- The theorem stating that if two pieces cover 3 squares in a 4x4 grid, 
    the third piece must cover 13 squares -/
theorem third_piece_coverage 
  (grid : Grid) 
  (piece1 piece2 : Piece) :
  grid.size = 4 →
  piece1.squares_covered = 2 →
  piece2.squares_covered = 1 →
  (∃ (piece3 : Piece), 
    piece1.squares_covered + piece2.squares_covered + piece3.squares_covered = grid.size * grid.size ∧
    piece3.squares_covered = 13) :=
by sorry

end NUMINAMATH_CALUDE_third_piece_coverage_l1848_184807


namespace NUMINAMATH_CALUDE_michaels_bunnies_l1848_184899

theorem michaels_bunnies (total_pets : ℕ) (dog_percent : ℚ) (cat_percent : ℚ) 
  (h1 : total_pets = 36)
  (h2 : dog_percent = 25 / 100)
  (h3 : cat_percent = 50 / 100)
  (h4 : dog_percent + cat_percent < 1) :
  (1 - dog_percent - cat_percent) * total_pets = 9 := by
  sorry

end NUMINAMATH_CALUDE_michaels_bunnies_l1848_184899


namespace NUMINAMATH_CALUDE_hockey_players_count_l1848_184806

/-- The number of hockey players in a games hour -/
def hockey_players (total players : ℕ) (cricket football softball : ℕ) : ℕ :=
  total - (cricket + football + softball)

/-- Theorem stating the number of hockey players -/
theorem hockey_players_count :
  hockey_players 51 10 16 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l1848_184806


namespace NUMINAMATH_CALUDE_shifted_parabola_properties_l1848_184861

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- Theorem stating that the shifted parabola is a vertical translation of the original parabola
    and passes through the point (0, 3) -/
theorem shifted_parabola_properties :
  (∃ k : ℝ, ∀ x : ℝ, shifted_parabola x = original_parabola x + k) ∧
  shifted_parabola 0 = 3 := by
  sorry


end NUMINAMATH_CALUDE_shifted_parabola_properties_l1848_184861


namespace NUMINAMATH_CALUDE_prob_sum_less_than_15_l1848_184831

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ 3

/-- The number of outcomes where the sum is less than 15 -/
def favorableOutcomes : ℕ := totalOutcomes - 26

/-- The probability of rolling three fair six-sided dice and getting a sum less than 15 -/
theorem prob_sum_less_than_15 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 95 / 108 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_15_l1848_184831


namespace NUMINAMATH_CALUDE_scientific_notation_of_113800_l1848_184816

theorem scientific_notation_of_113800 :
  ∃ (a : ℝ) (n : ℤ), 
    113800 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ a < 10 ∧
    a = 1.138 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_113800_l1848_184816


namespace NUMINAMATH_CALUDE_problem_solution_l1848_184836

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 1)^2 + 16/(x - 1)^2 = 7 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1848_184836


namespace NUMINAMATH_CALUDE_projection_a_onto_b_is_sqrt5_l1848_184866

/-- The projection of vector a onto the direction of vector b is √5 -/
theorem projection_a_onto_b_is_sqrt5 (a b : ℝ × ℝ) : 
  a = (1, 3) → a + b = (-1, 7) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_is_sqrt5_l1848_184866
