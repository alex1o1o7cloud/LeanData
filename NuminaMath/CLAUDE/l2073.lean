import Mathlib

namespace NUMINAMATH_CALUDE_multiplication_division_sum_l2073_207335

theorem multiplication_division_sum : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_sum_l2073_207335


namespace NUMINAMATH_CALUDE_constant_term_in_expansion_l2073_207318

theorem constant_term_in_expansion (n : ℕ) (h : n > 0) 
  (h_coeff : (n.choose 2) - (n.choose 1) = 44) : 
  let general_term (r : ℕ) := (n.choose r) * (33 - 11 * r) / 2
  ∃ (k : ℕ), k = 4 ∧ general_term (k - 1) = 0 :=
sorry

end NUMINAMATH_CALUDE_constant_term_in_expansion_l2073_207318


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2073_207351

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2073_207351


namespace NUMINAMATH_CALUDE_computer_table_markup_l2073_207372

/-- Calculate the percentage markup on a product's cost price. -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- The percentage markup on a computer table -/
theorem computer_table_markup :
  percentage_markup 8340 6672 = 25 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l2073_207372


namespace NUMINAMATH_CALUDE_faster_train_length_l2073_207307

/-- Given two trains moving in the same direction, this theorem calculates the length of the faster train. -/
theorem faster_train_length (v_fast v_slow : ℝ) (t_cross : ℝ) (h1 : v_fast = 72) (h2 : v_slow = 36) (h3 : t_cross = 18) :
  (v_fast - v_slow) * (5 / 18) * t_cross = 180 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_length_l2073_207307


namespace NUMINAMATH_CALUDE_rahul_share_l2073_207379

/-- Calculates the share of payment for a worker given the total payment and the time taken by each worker --/
def calculate_share (total_payment : ℚ) (time_worker1 time_worker2 : ℚ) : ℚ :=
  let work_rate1 := 1 / time_worker1
  let work_rate2 := 1 / time_worker2
  let combined_rate := work_rate1 + work_rate2
  let share_ratio := work_rate1 / combined_rate
  total_payment * share_ratio

/-- Proves that Rahul's share of the payment is 900 given the conditions --/
theorem rahul_share :
  let total_payment : ℚ := 2250
  let rahul_time : ℚ := 3
  let rajesh_time : ℚ := 2
  calculate_share total_payment rahul_time rajesh_time = 900 := by
  sorry

#eval calculate_share 2250 3 2

end NUMINAMATH_CALUDE_rahul_share_l2073_207379


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2073_207309

theorem perfect_square_condition (n : ℕ+) :
  (∃ m : ℕ, n^4 + 2*n^3 + 5*n^2 + 12*n + 5 = m^2) ↔ (n = 1 ∨ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2073_207309


namespace NUMINAMATH_CALUDE_total_hours_worked_l2073_207336

/-- Represents the hours worked by Thomas, Toby, and Rebecca in one week -/
structure WorkHours where
  thomas : ℕ
  toby : ℕ
  rebecca : ℕ

/-- Calculates the total hours worked by all three people -/
def totalHours (h : WorkHours) : ℕ :=
  h.thomas + h.toby + h.rebecca

/-- Theorem stating the total hours worked given the conditions -/
theorem total_hours_worked :
  ∀ h : WorkHours,
    (∃ x : ℕ, h.thomas = x ∧
              h.toby = 2 * x - 10 ∧
              h.rebecca = h.toby - 8 ∧
              h.rebecca = 56) →
    totalHours h = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_l2073_207336


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l2073_207365

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 20) :
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2) = 2 * Real.sqrt 104 :=
by sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l2073_207365


namespace NUMINAMATH_CALUDE_books_needed_l2073_207341

/-- The number of books each person has -/
structure BookCounts where
  darryl : ℕ
  lamont : ℕ
  loris : ℕ

/-- The conditions of the problem -/
def book_problem (b : BookCounts) : Prop :=
  b.darryl = 20 ∧
  b.lamont = 2 * b.darryl ∧
  b.darryl + b.lamont + b.loris = 97

/-- The theorem to prove -/
theorem books_needed (b : BookCounts) (h : book_problem b) : 
  b.lamont - b.loris = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_needed_l2073_207341


namespace NUMINAMATH_CALUDE_odd_function_sum_l2073_207388

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 4 = 5) :
  f 4 + f (-4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2073_207388


namespace NUMINAMATH_CALUDE_square_division_l2073_207370

theorem square_division (s : ℝ) (x : ℝ) : 
  s = 2 →  -- side length of the square
  (4 * (1/2 * s * x) + (s^2 - 4 * (1/2 * s * x))) = (s^2 / 5) →  -- equal areas condition
  x = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_square_division_l2073_207370


namespace NUMINAMATH_CALUDE_exists_solution_l2073_207395

theorem exists_solution : ∃ x : ℝ, x + 2.75 + 0.158 = 2.911 := by sorry

end NUMINAMATH_CALUDE_exists_solution_l2073_207395


namespace NUMINAMATH_CALUDE_polynomial_with_no_integer_roots_but_modular_roots_l2073_207324

/-- Definition of the polynomial P(x) = (x³ + 3)(x² + 1)(x² + 2)(x² - 2) -/
def P (x : ℤ) : ℤ := (x^3 + 3) * (x^2 + 1) * (x^2 + 2) * (x^2 - 2)

/-- Theorem stating the existence of a polynomial with the required properties -/
theorem polynomial_with_no_integer_roots_but_modular_roots :
  (∀ x : ℤ, P x ≠ 0) ∧
  (∀ n : ℕ, n > 0 → ∃ x : ℤ, P x % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_with_no_integer_roots_but_modular_roots_l2073_207324


namespace NUMINAMATH_CALUDE_gcd_1515_600_l2073_207317

theorem gcd_1515_600 : Nat.gcd 1515 600 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1515_600_l2073_207317


namespace NUMINAMATH_CALUDE_female_population_count_l2073_207393

def total_population : ℕ := 5000
def male_population : ℕ := 2000
def females_with_glasses : ℕ := 900
def female_glasses_percentage : ℚ := 30 / 100

theorem female_population_count : 
  ∃ (female_population : ℕ), 
    female_population = total_population - male_population ∧
    female_population = females_with_glasses / female_glasses_percentage :=
by sorry

end NUMINAMATH_CALUDE_female_population_count_l2073_207393


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2073_207340

-- Define the properties of our target number
def is_valid (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ ∃ k : ℕ, n = 17 * k

-- State the theorem
theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, is_valid n → n ≥ 102 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2073_207340


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_quadratic_equation_distinct_roots_2_l2073_207331

theorem quadratic_equation_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + (a + 1) * x - 2 = 0 ∧ a * y^2 + (a + 1) * y - 2 = 0) ↔
  (a < -5 - 2 * Real.sqrt 6 ∨ (-5 + 2 * Real.sqrt 6 < a ∧ a < 0) ∨ a > 0) :=
sorry

theorem quadratic_equation_distinct_roots_2 (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (1 - a) * x^2 + (a + 1) * x - 2 = 0 ∧ (1 - a) * y^2 + (a + 1) * y - 2 = 0) ↔
  (a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_quadratic_equation_distinct_roots_2_l2073_207331


namespace NUMINAMATH_CALUDE_remaining_investment_rate_l2073_207328

-- Define the investment amounts and rates
def total_investment : ℝ := 12000
def investment1_amount : ℝ := 5000
def investment1_rate : ℝ := 0.03
def investment2_amount : ℝ := 4000
def investment2_rate : ℝ := 0.045
def desired_income : ℝ := 600

-- Define the remaining investment amount
def remaining_investment : ℝ := total_investment - (investment1_amount + investment2_amount)

-- Define the income from the first two investments
def known_income : ℝ := investment1_amount * investment1_rate + investment2_amount * investment2_rate

-- Define the required income from the remaining investment
def required_income : ℝ := desired_income - known_income

-- Theorem to prove
theorem remaining_investment_rate : 
  (required_income / remaining_investment) = 0.09 := by sorry

end NUMINAMATH_CALUDE_remaining_investment_rate_l2073_207328


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2073_207322

theorem smaller_number_problem (x y : ℝ) 
  (sum_eq : x + y = 18) 
  (diff_eq : x - y = 8) : 
  min x y = 5 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2073_207322


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_7_l2073_207329

theorem sum_of_sixth_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_mod_7_l2073_207329


namespace NUMINAMATH_CALUDE_expression_simplification_l2073_207378

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 5*(2 + x) - 7*(1 - 3*x) = 31*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2073_207378


namespace NUMINAMATH_CALUDE_catman_do_whiskers_l2073_207362

theorem catman_do_whiskers (princess_puff_whiskers : ℕ) (catman_do_whiskers : ℕ) : 
  princess_puff_whiskers = 14 →
  catman_do_whiskers = 2 * princess_puff_whiskers - 6 →
  catman_do_whiskers = 22 := by
  sorry

end NUMINAMATH_CALUDE_catman_do_whiskers_l2073_207362


namespace NUMINAMATH_CALUDE_employed_female_percentage_l2073_207366

/-- Represents the percentage of a population --/
def Percentage := Finset (Fin 100)

theorem employed_female_percentage
  (total_employed : Percentage)
  (employed_males : Percentage)
  (h1 : total_employed.card = 60)
  (h2 : employed_males.card = 48) :
  (total_employed.card - employed_males.card : ℚ) / total_employed.card * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_employed_female_percentage_l2073_207366


namespace NUMINAMATH_CALUDE_cake_bread_weight_difference_l2073_207355

/-- Given that 4 cakes weigh 800 g and 3 cakes plus 5 pieces of bread weigh 1100 g,
    prove that a cake is 100 g heavier than a piece of bread. -/
theorem cake_bread_weight_difference :
  ∀ (cake_weight bread_weight : ℕ),
    4 * cake_weight = 800 →
    3 * cake_weight + 5 * bread_weight = 1100 →
    cake_weight - bread_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_bread_weight_difference_l2073_207355


namespace NUMINAMATH_CALUDE_phone_call_duration_l2073_207344

/-- Calculates the duration of a phone call given initial credit, cost per minute, and remaining credit -/
def call_duration (initial_credit : ℚ) (cost_per_minute : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_credit - remaining_credit) / cost_per_minute

/-- Proves that given the specified conditions, the call duration is 22 minutes -/
theorem phone_call_duration :
  let initial_credit : ℚ := 30
  let cost_per_minute : ℚ := 16/100
  let remaining_credit : ℚ := 2648/100
  call_duration initial_credit cost_per_minute remaining_credit = 22 := by
sorry


end NUMINAMATH_CALUDE_phone_call_duration_l2073_207344


namespace NUMINAMATH_CALUDE_jonathan_tax_calculation_l2073_207333

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def localTaxInCents (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a tax rate of 2.4%, the local tax amount is 60 cents. -/
theorem jonathan_tax_calculation :
  localTaxInCents 25 2.4 = 60 := by
  sorry

#eval localTaxInCents 25 2.4

end NUMINAMATH_CALUDE_jonathan_tax_calculation_l2073_207333


namespace NUMINAMATH_CALUDE_shortest_path_length_l2073_207306

/-- A regular tetrahedron with unit edge length -/
structure UnitRegularTetrahedron where
  -- We don't need to define the structure explicitly for this problem

/-- The shortest path on the surface of a unit regular tetrahedron between midpoints of opposite edges -/
def shortest_path (t : UnitRegularTetrahedron) : ℝ :=
  sorry -- Definition of the shortest path

/-- Theorem: The shortest path on the surface of a unit regular tetrahedron 
    between the midpoints of its opposite edges has a length of 1 -/
theorem shortest_path_length (t : UnitRegularTetrahedron) : 
  shortest_path t = 1 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_length_l2073_207306


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2073_207391

theorem partial_fraction_decomposition :
  ∀ (x : ℝ), x ≠ 10 → x ≠ -2 →
  (6 * x - 4) / (x^2 - 8 * x - 20) = 
  (14 / 3) / (x - 10) + (4 / 3) / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2073_207391


namespace NUMINAMATH_CALUDE_plot_perimeter_l2073_207389

/-- Proves that the perimeter of a rectangular plot is 300 meters given specific conditions -/
theorem plot_perimeter : 
  ∀ (width length perimeter : ℝ),
  length = width + 10 →
  1950 = (perimeter * 6.5) →
  perimeter = 2 * (length + width) →
  perimeter = 300 := by
sorry

end NUMINAMATH_CALUDE_plot_perimeter_l2073_207389


namespace NUMINAMATH_CALUDE_movie_watching_times_l2073_207396

/-- Represents the duration of the movie in minutes -/
def movie_duration : ℕ := 120

/-- Represents the time difference in minutes between when Camila and Maverick started watching -/
def camila_maverick_diff : ℕ := 30

/-- Represents the time difference in minutes between when Maverick and Daniella started watching -/
def maverick_daniella_diff : ℕ := 45

/-- Represents the number of minutes Daniella has left to watch -/
def daniella_remaining : ℕ := 30

/-- Theorem stating that Camila and Maverick have finished watching when Daniella has 30 minutes left -/
theorem movie_watching_times :
  let camila_watched := movie_duration + maverick_daniella_diff + camila_maverick_diff
  let maverick_watched := movie_duration + maverick_daniella_diff
  let daniella_watched := movie_duration - daniella_remaining
  camila_watched ≥ movie_duration ∧ maverick_watched ≥ movie_duration ∧ daniella_watched < movie_duration :=
by sorry

end NUMINAMATH_CALUDE_movie_watching_times_l2073_207396


namespace NUMINAMATH_CALUDE_a_divides_b_l2073_207321

theorem a_divides_b (a b : ℕ) (h1 : a > 1) (h2 : b > 1)
  (r : ℕ → ℕ)
  (h3 : ∀ n : ℕ, n > 0 → r n = b^n % a^n)
  (h4 : ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (r n : ℚ) < 2^n / n) :
  a ∣ b :=
by sorry

end NUMINAMATH_CALUDE_a_divides_b_l2073_207321


namespace NUMINAMATH_CALUDE_jack_email_difference_l2073_207357

theorem jack_email_difference : 
  ∀ (morning_emails afternoon_emails morning_letters afternoon_letters : ℕ),
  morning_emails = 10 →
  afternoon_emails = 3 →
  morning_letters = 12 →
  afternoon_letters = 44 →
  morning_emails - afternoon_emails = 7 :=
by sorry

end NUMINAMATH_CALUDE_jack_email_difference_l2073_207357


namespace NUMINAMATH_CALUDE_first_player_wins_l2073_207373

/-- Represents the state of the game -/
structure GameState where
  player1Pos : Nat
  player2Pos : Nat

/-- Represents a valid move in the game -/
inductive Move where
  | one   : Move
  | two   : Move
  | three : Move
  | four  : Move

/-- The game board size -/
def boardSize : Nat := 101

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (player : Nat) (move : Move) : Bool :=
  match player, move with
  | 1, Move.one   => state.player1Pos + 1 < state.player2Pos
  | 1, Move.two   => state.player1Pos + 2 < state.player2Pos
  | 1, Move.three => state.player1Pos + 3 < state.player2Pos
  | 1, Move.four  => state.player1Pos + 4 < state.player2Pos
  | 2, Move.one   => state.player2Pos - 1 > state.player1Pos
  | 2, Move.two   => state.player2Pos - 2 > state.player1Pos
  | 2, Move.three => state.player2Pos - 3 > state.player1Pos
  | 2, Move.four  => state.player2Pos - 4 > state.player1Pos
  | _, _          => false

/-- Applies a move to the game state -/
def applyMove (state : GameState) (player : Nat) (move : Move) : GameState :=
  match player, move with
  | 1, Move.one   => { state with player1Pos := state.player1Pos + 1 }
  | 1, Move.two   => { state with player1Pos := state.player1Pos + 2 }
  | 1, Move.three => { state with player1Pos := state.player1Pos + 3 }
  | 1, Move.four  => { state with player1Pos := state.player1Pos + 4 }
  | 2, Move.one   => { state with player2Pos := state.player2Pos - 1 }
  | 2, Move.two   => { state with player2Pos := state.player2Pos - 2 }
  | 2, Move.three => { state with player2Pos := state.player2Pos - 3 }
  | 2, Move.four  => { state with player2Pos := state.player2Pos - 4 }
  | _, _          => state

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.player1Pos = boardSize || state.player2Pos = 1

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      let game_result := (sorry : GameState)  -- Simulate game play
      isGameOver game_result ∧ game_result.player1Pos = boardSize :=
sorry


end NUMINAMATH_CALUDE_first_player_wins_l2073_207373


namespace NUMINAMATH_CALUDE_number_in_interval_l2073_207314

theorem number_in_interval (y : ℝ) (h : y = (1/y) * (-y) + 5) : 2 < y ∧ y ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_number_in_interval_l2073_207314


namespace NUMINAMATH_CALUDE_import_tax_threshold_l2073_207382

/-- The amount in excess of which the import tax was applied -/
def X : ℝ :=
  1000

/-- The import tax rate -/
def tax_rate : ℝ :=
  0.07

/-- The total value of the item -/
def total_value : ℝ :=
  2250

/-- The import tax paid -/
def tax_paid : ℝ :=
  87.50

theorem import_tax_threshold :
  tax_rate * (total_value - X) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l2073_207382


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2073_207350

-- Problem 1
theorem problem_1 : 0.25 * 1.25 * 32 = 10 := by sorry

-- Problem 2
theorem problem_2 : 4/5 * 5/11 + 5/11 / 5 = 5/11 := by sorry

-- Problem 3
theorem problem_3 : 7 - 4/9 - 5/9 = 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2073_207350


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2073_207339

/-- The first line: x - 2y - 4 = 0 -/
def line1 (x y : ℝ) : Prop := x - 2*y - 4 = 0

/-- The second line: x + 3y + 6 = 0 -/
def line2 (x y : ℝ) : Prop := x + 3*y + 6 = 0

/-- The intersection point (0, -2) -/
def intersection_point : ℝ × ℝ := (0, -2)

/-- Theorem stating that (0, -2) is the unique intersection point of the two lines -/
theorem intersection_point_is_unique :
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2073_207339


namespace NUMINAMATH_CALUDE_tony_age_at_end_of_period_l2073_207304

/-- Represents Tony's age at the start of the period -/
def initial_age : ℕ := 14

/-- Represents Tony's age at the end of the period -/
def final_age : ℕ := initial_age + 1

/-- Represents the number of days Tony worked at his initial age -/
def days_at_initial_age : ℕ := 46

/-- Represents the number of days Tony worked at his final age -/
def days_at_final_age : ℕ := 100 - days_at_initial_age

/-- Represents Tony's daily earnings at a given age -/
def daily_earnings (age : ℕ) : ℚ := 1.9 * age

/-- Represents Tony's total earnings during the period -/
def total_earnings : ℚ := 3750

theorem tony_age_at_end_of_period :
  final_age = 15 ∧
  days_at_initial_age + days_at_final_age = 100 ∧
  daily_earnings initial_age * days_at_initial_age +
  daily_earnings final_age * days_at_final_age = total_earnings :=
sorry

end NUMINAMATH_CALUDE_tony_age_at_end_of_period_l2073_207304


namespace NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l2073_207343

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l2073_207343


namespace NUMINAMATH_CALUDE_large_pizza_cost_is_10_l2073_207358

/-- Represents the cost of a pizza topping --/
structure ToppingCost where
  first : ℝ
  next_two : ℝ
  rest : ℝ

/-- Calculates the total cost of toppings --/
def total_topping_cost (tc : ToppingCost) (num_toppings : ℕ) : ℝ :=
  tc.first + 
  (if num_toppings > 1 then min (num_toppings - 1) 2 * tc.next_two else 0) +
  (if num_toppings > 3 then (num_toppings - 3) * tc.rest else 0)

/-- The cost of a large pizza without toppings --/
def large_pizza_cost (slices : ℕ) (cost_per_slice : ℝ) (num_toppings : ℕ) (tc : ToppingCost) : ℝ :=
  slices * cost_per_slice - total_topping_cost tc num_toppings

/-- Theorem: The cost of a large pizza without toppings is $10.00 --/
theorem large_pizza_cost_is_10 : 
  large_pizza_cost 8 2 7 ⟨2, 1, 0.5⟩ = 10 := by
  sorry

end NUMINAMATH_CALUDE_large_pizza_cost_is_10_l2073_207358


namespace NUMINAMATH_CALUDE_expensive_rock_cost_l2073_207313

/-- Given a mixture of two types of rock, prove the cost of the more expensive rock -/
theorem expensive_rock_cost 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (cheap_rock_cost : ℝ) 
  (cheap_rock_weight : ℝ) 
  (expensive_rock_weight : ℝ)
  (h1 : total_weight = 24)
  (h2 : total_cost = 800)
  (h3 : cheap_rock_cost = 30)
  (h4 : cheap_rock_weight = 8)
  (h5 : expensive_rock_weight = 8)
  : (total_cost - cheap_rock_cost * cheap_rock_weight) / (total_weight - cheap_rock_weight) = 35 := by
  sorry

end NUMINAMATH_CALUDE_expensive_rock_cost_l2073_207313


namespace NUMINAMATH_CALUDE_min_sum_four_reals_l2073_207319

theorem min_sum_four_reals (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ + x₂ ≥ 12)
  (h2 : x₁ + x₃ ≥ 13)
  (h3 : x₁ + x₄ ≥ 14)
  (h4 : x₃ + x₄ ≥ 22)
  (h5 : x₂ + x₃ ≥ 23)
  (h6 : x₂ + x₄ ≥ 24) :
  x₁ + x₂ + x₃ + x₄ ≥ 37 ∧ ∃ a b c d : ℝ, a + b + c + d = 37 ∧ 
    a + b ≥ 12 ∧ a + c ≥ 13 ∧ a + d ≥ 14 ∧ c + d ≥ 22 ∧ b + c ≥ 23 ∧ b + d ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_four_reals_l2073_207319


namespace NUMINAMATH_CALUDE_fib_div_three_iff_index_div_four_l2073_207364

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_div_three_iff_index_div_four (n : ℕ) : 
  3 ∣ fib n ↔ 4 ∣ n := by sorry

end NUMINAMATH_CALUDE_fib_div_three_iff_index_div_four_l2073_207364


namespace NUMINAMATH_CALUDE_digital_sum_property_l2073_207332

/-- Digital sum of a natural number -/
def digitalSum (n : ℕ) : ℕ := sorry

/-- Proposition: M satisfies S(Mk) = S(M) for all 1 ≤ k ≤ M iff M = 10^l - 1 for some l -/
theorem digital_sum_property (M : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digitalSum (M * k) = digitalSum M) ↔
  ∃ l : ℕ, l > 0 ∧ M = 10^l - 1 :=
sorry

end NUMINAMATH_CALUDE_digital_sum_property_l2073_207332


namespace NUMINAMATH_CALUDE_f_bounds_in_R_f_attains_bounds_l2073_207325

/-- The triangular region R with vertices A(4,1), B(-1,-6), C(-3,2) -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (4*a - b - 3*c, a - 6*b + 2*c)}

/-- The function to be maximized and minimized -/
def f (p : ℝ × ℝ) : ℝ := 4 * p.1 - 3 * p.2

theorem f_bounds_in_R :
  ∀ p ∈ R, -18 ≤ f p ∧ f p ≤ 14 :=
by sorry

theorem f_attains_bounds :
  (∃ p ∈ R, f p = -18) ∧ (∃ p ∈ R, f p = 14) :=
by sorry

end NUMINAMATH_CALUDE_f_bounds_in_R_f_attains_bounds_l2073_207325


namespace NUMINAMATH_CALUDE_units_digit_3_2009_l2073_207320

def units_digit (n : ℕ) : ℕ := n % 10

def power_3_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem units_digit_3_2009 : units_digit (3^2009) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_3_2009_l2073_207320


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2073_207360

/-- Represents the ratio of product types A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a stratified sample -/
structure StratifiedSample where
  ratio : ProductRatio
  type_a_count : ℕ
  total_size : ℕ

/-- Theorem: Given a stratified sample with product ratio 5:2:3 and 15 Type A products,
    the total sample size is 30 -/
theorem stratified_sample_size
  (sample : StratifiedSample)
  (h_ratio : sample.ratio = ⟨5, 2, 3⟩)
  (h_type_a : sample.type_a_count = 15) :
  sample.total_size = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2073_207360


namespace NUMINAMATH_CALUDE_square_roots_problem_l2073_207361

theorem square_roots_problem (a : ℝ) (n : ℝ) : 
  (2*a + 3)^2 = n ∧ (a - 18)^2 = n → n = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2073_207361


namespace NUMINAMATH_CALUDE_circle_area_increase_l2073_207377

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  (π * (2 * r)^2 - π * r^2) / (π * r^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2073_207377


namespace NUMINAMATH_CALUDE_gavins_blue_shirts_l2073_207359

theorem gavins_blue_shirts (total_shirts : ℕ) (green_shirts : ℕ) (blue_shirts : ℕ) :
  total_shirts = 23 →
  green_shirts = 17 →
  total_shirts = green_shirts + blue_shirts →
  blue_shirts = 6 := by
sorry

end NUMINAMATH_CALUDE_gavins_blue_shirts_l2073_207359


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_of_special_triangle_l2073_207392

/-- Given a triangle ABC where side b = 2√3 and angles A, B, C form an arithmetic sequence,
    the radius of the circumscribed circle is 2. -/
theorem circumscribed_circle_radius_of_special_triangle (A B C : Real) (a b c : Real) :
  b = 2 * Real.sqrt 3 →
  ∃ (d : Real), B = (A + C) / 2 ∧ A + d = B ∧ B + d = C →
  A + B + C = Real.pi →
  2 * Real.sin B = b / 2 →
  2 = 2 * Real.sin B / b * 2 * Real.sqrt 3 := by
  sorry

#check circumscribed_circle_radius_of_special_triangle

end NUMINAMATH_CALUDE_circumscribed_circle_radius_of_special_triangle_l2073_207392


namespace NUMINAMATH_CALUDE_complex_division_theorem_l2073_207369

theorem complex_division_theorem : 
  let z₁ : ℂ := Complex.mk 1 (-1)
  let z₂ : ℂ := Complex.mk 3 1
  z₂ / z₁ = Complex.mk 1 2 := by
sorry

end NUMINAMATH_CALUDE_complex_division_theorem_l2073_207369


namespace NUMINAMATH_CALUDE_problem_solution_l2073_207337

theorem problem_solution :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  (abs (-3) - 4 * Real.sin (π/4) + Real.sqrt 8 + (π - 3)^0 = 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2073_207337


namespace NUMINAMATH_CALUDE_stock_change_theorem_l2073_207326

/-- The overall percent change in a stock after two days of trading -/
def overall_percent_change (day1_decrease : ℝ) (day2_increase : ℝ) : ℝ :=
  (((1 - day1_decrease) * (1 + day2_increase)) - 1) * 100

/-- Theorem stating the overall percent change for the given scenario -/
theorem stock_change_theorem :
  overall_percent_change 0.25 0.35 = 1.25 := by
  sorry

#eval overall_percent_change 0.25 0.35

end NUMINAMATH_CALUDE_stock_change_theorem_l2073_207326


namespace NUMINAMATH_CALUDE_square_of_negative_two_l2073_207330

theorem square_of_negative_two : (-2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_l2073_207330


namespace NUMINAMATH_CALUDE_identity_implies_a_minus_b_zero_l2073_207311

theorem identity_implies_a_minus_b_zero 
  (x : ℚ) 
  (h_pos : x > 0) 
  (h_identity : ∀ x > 0, a / (2^x - 1) + b / (2^x + 2) = (2 * 2^x + 1) / ((2^x - 1) * (2^x + 2))) : 
  a - b = 0 :=
by sorry

end NUMINAMATH_CALUDE_identity_implies_a_minus_b_zero_l2073_207311


namespace NUMINAMATH_CALUDE_quadratic_problem_l2073_207394

-- Define the quadratic equation
def quadratic (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Theorem statement
theorem quadratic_problem (p q : ℝ) 
  (h1 : quadratic p (q + 1) 2 = 0) : 
  -- 1. Relationship between q and p
  q = -2*p - 5 ∧ 
  -- 2. Two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic p q x₁ = 0 ∧ quadratic p q x₂ = 0 ∧
  -- 3. If equal roots in original equation, roots of modified equation
  (∃ (r : ℝ), ∀ x, quadratic p (q + 1) x = 0 → x = r) → 
    quadratic p q 1 = 0 ∧ quadratic p q 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2073_207394


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2073_207374

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x > 0) ∧ 
  (∃ x, x > 0 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2073_207374


namespace NUMINAMATH_CALUDE_sum_negative_forty_to_sixty_l2073_207363

def sum_range (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_negative_forty_to_sixty :
  sum_range (-40) 60 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_sum_negative_forty_to_sixty_l2073_207363


namespace NUMINAMATH_CALUDE_g_50_equals_zero_l2073_207342

theorem g_50_equals_zero
  (g : ℕ → ℕ)
  (h : ∀ a b : ℕ, 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b)) :
  g 50 = 0 := by
sorry

end NUMINAMATH_CALUDE_g_50_equals_zero_l2073_207342


namespace NUMINAMATH_CALUDE_imaginary_part_z_2017_l2073_207305

theorem imaginary_part_z_2017 : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) / (1 - i)
  Complex.im (z^2017) = Complex.im i := by sorry

end NUMINAMATH_CALUDE_imaginary_part_z_2017_l2073_207305


namespace NUMINAMATH_CALUDE_cookies_eaten_l2073_207353

theorem cookies_eaten (initial_cookies bought_cookies final_cookies : ℕ) :
  initial_cookies = 40 →
  bought_cookies = 37 →
  final_cookies = 75 →
  initial_cookies + bought_cookies - final_cookies = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l2073_207353


namespace NUMINAMATH_CALUDE_factorial_division_l2073_207383

theorem factorial_division :
  (10 : ℕ).factorial = 3628800 →
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2073_207383


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2073_207385

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ a₂ : ℤ) (h₁ : a₁ = 2) (h₂ : a₂ = 1) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = -7 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2073_207385


namespace NUMINAMATH_CALUDE_function_is_2x_l2073_207323

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem function_is_2x (f : ℝ → ℝ) 
  (h₁ : f (-1) = -2)
  (h₂ : f 0 = 0)
  (h₃ : f 1 = 2)
  (h₄ : f 2 = 4) :
  ∀ x, f x = 2 * x := by
sorry

end NUMINAMATH_CALUDE_function_is_2x_l2073_207323


namespace NUMINAMATH_CALUDE_triangle_projection_relation_l2073_207302

/-- In a triangle with sides a, b, and c, where a > b > c and a = 2(b - c),
    and p is the projection of side c onto a, the equation 4c + 8p = 3a holds. -/
theorem triangle_projection_relation (a b c p : ℝ) : 
  a > b → b > c → a = 2*(b - c) → 4*c + 8*p = 3*a := by
  sorry

end NUMINAMATH_CALUDE_triangle_projection_relation_l2073_207302


namespace NUMINAMATH_CALUDE_max_candy_leftover_l2073_207308

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l2073_207308


namespace NUMINAMATH_CALUDE_problem_solution_l2073_207397

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log (abs x)
def g (a : ℝ) (x : ℝ) : ℝ := 1 / (deriv f x) + a * (deriv f x)

-- State the theorem
theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g a x = x + a / x) ∧
  (a > 0 ∧ (∀ x : ℝ, x > 0 → g a x ≥ 2) ∧ ∃ x : ℝ, x > 0 ∧ g a x = 2) →
  (a = 1 ∧
   (∫ x in (3/2)..(2), (2/3 * x + 7/6) - (x + 1/x)) = 7/24 + Real.log 3 - 2 * Real.log 2) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2073_207397


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2073_207301

/-- Given a square with perimeter 160 units divided into 4 rectangles, where each rectangle
    has one side equal to half of the square's side length and the other side equal to the
    full side length of the square, the perimeter of one of these rectangles is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (rectangle_count : ℕ) 
  (h1 : square_perimeter = 160)
  (h2 : rectangle_count = 4)
  (h3 : ∀ r : ℝ, r > 0 → ∃ (s w : ℝ), s = r ∧ w = r / 2 ∧ 
       4 * r = square_perimeter ∧
       rectangle_count * (s * w) = r * r) :
  ∃ (rectangle_perimeter : ℝ), rectangle_perimeter = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2073_207301


namespace NUMINAMATH_CALUDE_factorizations_of_945_l2073_207381

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  a * b = 945 ∧ is_two_digit a ∧ is_two_digit b

def unique_factorizations : Prop :=
  ∃ (f₁ f₂ : ℕ × ℕ),
    valid_factorization f₁.1 f₁.2 ∧
    valid_factorization f₂.1 f₂.2 ∧
    f₁ ≠ f₂ ∧
    (∀ (g : ℕ × ℕ), valid_factorization g.1 g.2 → g = f₁ ∨ g = f₂)

theorem factorizations_of_945 : unique_factorizations := by
  sorry

end NUMINAMATH_CALUDE_factorizations_of_945_l2073_207381


namespace NUMINAMATH_CALUDE_remainder_sum_l2073_207356

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 53) 
  (hb : b % 50 = 24) : 
  (a + b) % 20 = 17 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2073_207356


namespace NUMINAMATH_CALUDE_three_number_problem_l2073_207303

theorem three_number_problem :
  ∃ (X Y Z : ℤ),
    (X = (35 * X) / 100 + 60) ∧
    (X = (70 * Y) / 200 + Y / 2) ∧
    (Y = 2 * Z^2) ∧
    (X = 92) ∧
    (Y = 108) ∧
    (Z = 7) := by
  sorry

end NUMINAMATH_CALUDE_three_number_problem_l2073_207303


namespace NUMINAMATH_CALUDE_curtain_length_is_101_l2073_207380

/-- The required curtain length in inches, given room height in feet, 
    additional length for pooling, and the conversion factor from feet to inches. -/
def curtain_length (room_height_ft : ℕ) (pooling_inches : ℕ) (inches_per_foot : ℕ) : ℕ :=
  room_height_ft * inches_per_foot + pooling_inches

/-- Theorem stating that the required curtain length is 101 inches 
    for the given conditions. -/
theorem curtain_length_is_101 :
  curtain_length 8 5 12 = 101 := by
  sorry

end NUMINAMATH_CALUDE_curtain_length_is_101_l2073_207380


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l2073_207327

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720° -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l2073_207327


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l2073_207352

theorem modular_inverse_13_mod_101 : ∃ x : ℤ, (13 * x) % 101 = 1 ∧ 0 ≤ x ∧ x < 101 :=
by
  use 70
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l2073_207352


namespace NUMINAMATH_CALUDE_initial_cheerleaders_initial_cheerleaders_correct_l2073_207338

theorem initial_cheerleaders (initial_football_players : ℕ) 
                             (quit_football_players : ℕ) 
                             (quit_cheerleaders : ℕ) 
                             (remaining_total : ℕ) : ℕ :=
  let initial_cheerleaders := 16
  have h1 : initial_football_players = 13 := by sorry
  have h2 : quit_football_players = 10 := by sorry
  have h3 : quit_cheerleaders = 4 := by sorry
  have h4 : remaining_total = 15 := by sorry
  have h5 : initial_football_players - quit_football_players + 
            (initial_cheerleaders - quit_cheerleaders) = remaining_total := by sorry
  initial_cheerleaders

theorem initial_cheerleaders_correct : initial_cheerleaders 13 10 4 15 = 16 := by sorry

end NUMINAMATH_CALUDE_initial_cheerleaders_initial_cheerleaders_correct_l2073_207338


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2073_207354

theorem intersection_of_sets (A B : Set ℝ) (a : ℝ) : 
  A = {-1, 0, 1} → 
  B = {a + 1, 2 * a} → 
  A ∩ B = {0} → 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2073_207354


namespace NUMINAMATH_CALUDE_internet_rate_proof_l2073_207334

/-- The regular monthly internet rate without discount -/
def regular_rate : ℝ := 50

/-- The discounted rate as a fraction of the regular rate -/
def discount_rate : ℝ := 0.95

/-- The number of months -/
def num_months : ℕ := 4

/-- The total payment for the given number of months -/
def total_payment : ℝ := 190

theorem internet_rate_proof : 
  regular_rate * discount_rate * num_months = total_payment := by
  sorry

#check internet_rate_proof

end NUMINAMATH_CALUDE_internet_rate_proof_l2073_207334


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2073_207346

theorem mean_equality_implies_z_value : ∃ z : ℝ,
  (6 + 15 + 9 + 20) / 4 = (13 + z) / 2 → z = 12 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2073_207346


namespace NUMINAMATH_CALUDE_solve_for_s_l2073_207375

theorem solve_for_s (s t : ℝ) 
  (eq1 : 8 * s + 4 * t = 160) 
  (eq2 : t = 2 * s - 3) : 
  s = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_s_l2073_207375


namespace NUMINAMATH_CALUDE_rotated_ellipse_sum_l2073_207367

/-- Represents an ellipse rotated 90 degrees around its center. -/
structure RotatedEllipse where
  h' : ℝ  -- x-coordinate of the center
  k' : ℝ  -- y-coordinate of the center
  a' : ℝ  -- length of the semi-major axis
  b' : ℝ  -- length of the semi-minor axis

/-- Theorem stating the sum of parameters for a specific rotated ellipse. -/
theorem rotated_ellipse_sum (e : RotatedEllipse) 
  (center_x : e.h' = 3) 
  (center_y : e.k' = -5) 
  (major_axis : e.a' = 4) 
  (minor_axis : e.b' = 2) : 
  e.h' + e.k' + e.a' + e.b' = 4 := by
  sorry

end NUMINAMATH_CALUDE_rotated_ellipse_sum_l2073_207367


namespace NUMINAMATH_CALUDE_simplify_fraction_l2073_207345

theorem simplify_fraction : (45 : ℚ) / 75 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2073_207345


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l2073_207310

noncomputable section

open Real

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (λ x => x * log x) x = 1 + log x :=
sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l2073_207310


namespace NUMINAMATH_CALUDE_max_value_fraction_l2073_207387

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (M : ℝ), M = 1/4 ∧ 
  (x * y * z * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ M ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * c * (a + b + c)) / ((a + c)^2 * (c + b)^2) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2073_207387


namespace NUMINAMATH_CALUDE_range_of_h_l2073_207349

def f (x : ℝ) : ℝ := 4 * x - 3

def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h :
  let S : Set ℝ := {y | ∃ x ∈ Set.Icc (-1 : ℝ) 3, h x = y}
  S = Set.Icc (-127 : ℝ) 129 := by
  sorry

end NUMINAMATH_CALUDE_range_of_h_l2073_207349


namespace NUMINAMATH_CALUDE_square_side_length_l2073_207371

theorem square_side_length (s : ℝ) : s > 0 → (4 * s = 2 * s^2) → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2073_207371


namespace NUMINAMATH_CALUDE_games_mike_can_buy_l2073_207376

/-- The maximum number of games that can be bought given initial money, spent money, and game cost. -/
def max_games_buyable (initial_money spent_money game_cost : ℕ) : ℕ :=
  (initial_money - spent_money) / game_cost

/-- Theorem stating that given the specific values in the problem, the maximum number of games that can be bought is 4. -/
theorem games_mike_can_buy :
  max_games_buyable 42 10 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_mike_can_buy_l2073_207376


namespace NUMINAMATH_CALUDE_first_valve_fills_in_two_hours_l2073_207300

/-- Represents the time in hours taken by the first valve to fill the pool -/
def first_valve_fill_time (pool_capacity : ℝ) (both_valves_fill_time : ℝ) (valve_difference : ℝ) : ℝ :=
  2

/-- Theorem stating that under given conditions, the first valve takes 2 hours to fill the pool -/
theorem first_valve_fills_in_two_hours 
  (pool_capacity : ℝ) 
  (both_valves_fill_time : ℝ) 
  (valve_difference : ℝ) 
  (h1 : pool_capacity = 12000)
  (h2 : both_valves_fill_time = 48 / 60) -- Convert 48 minutes to hours
  (h3 : valve_difference = 50) :
  first_valve_fill_time pool_capacity both_valves_fill_time valve_difference = 2 := by
  sorry

#check first_valve_fills_in_two_hours

end NUMINAMATH_CALUDE_first_valve_fills_in_two_hours_l2073_207300


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l2073_207312

-- Define a relation for numbers with the same prime factors
def same_prime_factors (x y : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ x ↔ p ∣ y)

theorem power_of_two_plus_one (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_same_factors : same_prime_factors (b^m - 1) (b^n - 1)) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l2073_207312


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2073_207386

theorem polynomial_factor_implies_coefficients 
  (p q : ℚ) 
  (h : ∃ (a b : ℚ), px^4 + qx^3 + 45*x^2 - 25*x + 10 = (5*x^2 - 3*x + 2)*(a*x^2 + b*x + 5)) :
  p = 25/2 ∧ q = -65/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2073_207386


namespace NUMINAMATH_CALUDE_gary_shortage_l2073_207384

def gary_initial_amount : ℝ := 73
def snake_cost : ℝ := 55
def snake_food_cost : ℝ := 12
def habitat_original_cost : ℝ := 35
def habitat_discount_rate : ℝ := 0.15

def total_spent : ℝ := snake_cost + snake_food_cost + 
  (habitat_original_cost * (1 - habitat_discount_rate))

theorem gary_shortage : 
  total_spent - gary_initial_amount = 23.75 := by sorry

end NUMINAMATH_CALUDE_gary_shortage_l2073_207384


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2073_207347

theorem intersection_point_of_lines (x y : ℚ) :
  (8 * x - 5 * y = 40) ∧ (6 * x + 2 * y = 14) ↔ x = 75 / 23 ∧ y = -64 / 23 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2073_207347


namespace NUMINAMATH_CALUDE_number_division_problem_l2073_207399

theorem number_division_problem : ∃! x : ℕ, 
  ∃ q : ℕ, x = 7 * q ∧ q + x + 7 = 175 ∧ x = 147 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2073_207399


namespace NUMINAMATH_CALUDE_jason_seashells_l2073_207398

/-- The number of seashells Jason has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Jason has 36 seashells after starting with 49 and giving away 13 -/
theorem jason_seashells : remaining_seashells 49 13 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_seashells_l2073_207398


namespace NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l2073_207348

/-- Proves that a cistern with given fill and empty rates takes 2 additional hours to fill due to a leak -/
theorem cistern_fill_time_with_leak 
  (normal_fill_time : ℝ) 
  (empty_time : ℝ) 
  (h1 : normal_fill_time = 4) 
  (h2 : empty_time = 12) : 
  let fill_rate := 1 / normal_fill_time
  let leak_rate := 1 / empty_time
  let effective_rate := fill_rate - leak_rate
  let time_with_leak := 1 / effective_rate
  time_with_leak - normal_fill_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_with_leak_l2073_207348


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2073_207316

theorem simplify_and_evaluate (m : ℚ) (h : m = 2) : 
  ((2 * m + 1) / m - 1) / ((m^2 - 1) / m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2073_207316


namespace NUMINAMATH_CALUDE_burger_share_inches_l2073_207368

-- Define the length of the burger in feet
def burger_length_feet : ℝ := 1

-- Define the number of people sharing the burger
def num_people : ℕ := 2

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem to prove
theorem burger_share_inches : 
  (burger_length_feet * feet_to_inches) / num_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_burger_share_inches_l2073_207368


namespace NUMINAMATH_CALUDE_largest_n_value_l2073_207390

/-- The largest possible value of n for regular polygons Q1 (m-gon) and Q2 (n-gon) 
    satisfying the given conditions -/
theorem largest_n_value (m n : ℕ) : m ≥ n → n ≥ 3 → 
  (m - 2) * n = (n - 2) * m * 8 / 7 → 
  (∀ k, k > n → (k - 2) * m ≠ (m - 2) * k * 8 / 7) →
  n = 112 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_value_l2073_207390


namespace NUMINAMATH_CALUDE_sausage_problem_l2073_207315

/-- Represents the sausage problem --/
theorem sausage_problem (total_meat : ℕ) (total_links : ℕ) (remaining_meat : ℕ) 
  (h1 : total_meat = 10) 
  (h2 : total_links = 40)
  (h3 : remaining_meat = 112) : 
  (total_meat * 16 - remaining_meat) / (total_meat * 16 / total_links) = 12 := by
  sorry

#check sausage_problem

end NUMINAMATH_CALUDE_sausage_problem_l2073_207315
