import Mathlib

namespace NUMINAMATH_CALUDE_binary_110010_equals_50_l1069_106943

-- Define the binary number as a list of digits
def binary_number : List Nat := [1, 1, 0, 0, 1, 0]

-- Function to convert binary to decimal
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem statement
theorem binary_110010_equals_50 :
  binary_to_decimal binary_number = 50 := by
  sorry

end NUMINAMATH_CALUDE_binary_110010_equals_50_l1069_106943


namespace NUMINAMATH_CALUDE_remainder_proof_l1069_106952

theorem remainder_proof :
  let n : ℕ := 174
  let d₁ : ℕ := 34
  let d₂ : ℕ := 5
  (n % d₁ = 4) ∧ (n % d₂ = 4) :=
by sorry

end NUMINAMATH_CALUDE_remainder_proof_l1069_106952


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_six_satisfies_condition_seven_does_not_satisfy_l1069_106944

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

theorem six_satisfies_condition :
  6^2 - 11*6 + 28 < 0 :=
by sorry

theorem seven_does_not_satisfy :
  7^2 - 11*7 + 28 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_six_satisfies_condition_seven_does_not_satisfy_l1069_106944


namespace NUMINAMATH_CALUDE_math_only_count_l1069_106908

def brainiac_survey (total : ℕ) (rebus math logic : ℕ) 
  (rebus_math rebus_logic math_logic all_three neither : ℕ) : Prop :=
  total = 500 ∧
  rebus = 2 * math ∧
  logic = math ∧
  rebus_math = 72 ∧
  rebus_logic = 40 ∧
  math_logic = 36 ∧
  all_three = 10 ∧
  neither = 20

theorem math_only_count 
  (total rebus math logic rebus_math rebus_logic math_logic all_three neither : ℕ) :
  brainiac_survey total rebus math logic rebus_math rebus_logic math_logic all_three neither →
  math - rebus_math - math_logic + all_three = 54 :=
by sorry

end NUMINAMATH_CALUDE_math_only_count_l1069_106908


namespace NUMINAMATH_CALUDE_simplify_power_product_l1069_106949

theorem simplify_power_product (x : ℝ) : (x^5 * x^3)^2 = x^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_product_l1069_106949


namespace NUMINAMATH_CALUDE_derivative_x_cos_x_l1069_106938

theorem derivative_x_cos_x (x : ℝ) :
  deriv (fun x => x * Real.cos x) x = Real.cos x - x * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_cos_x_l1069_106938


namespace NUMINAMATH_CALUDE_concert_ticket_ratio_l1069_106906

theorem concert_ticket_ratio (initial_amount : ℚ) (motorcycle_cost : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 5000)
  (h2 : motorcycle_cost = 2800)
  (h3 : final_amount = 825)
  (h4 : ∃ (concert_cost : ℚ),
    final_amount = (initial_amount - motorcycle_cost - concert_cost) * (3/4)) :
  ∃ (concert_cost : ℚ),
    concert_cost / (initial_amount - motorcycle_cost) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_ratio_l1069_106906


namespace NUMINAMATH_CALUDE_odd_function_value_l1069_106950

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_value (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f 3 = 7) :
  f (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l1069_106950


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l1069_106965

theorem nested_fraction_simplification : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l1069_106965


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1069_106924

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1069_106924


namespace NUMINAMATH_CALUDE_baseball_game_total_baseball_game_total_is_643_l1069_106927

/-- Represents the statistics of a baseball team for a single day -/
structure DayStats where
  misses : ℕ
  hits : ℕ
  singles : ℕ
  doubles : ℕ
  triples : ℕ
  homeRuns : ℕ

/-- Represents the statistics of a baseball team for three days -/
structure TeamStats where
  day1 : DayStats
  day2 : DayStats
  day3 : DayStats

theorem baseball_game_total (teamA teamB : TeamStats) : ℕ :=
  let totalMisses := teamA.day1.misses + teamA.day2.misses + teamA.day3.misses +
                     teamB.day1.misses + teamB.day2.misses + teamB.day3.misses
  let totalSingles := teamA.day1.singles + teamA.day2.singles + teamA.day3.singles +
                      teamB.day1.singles + teamB.day2.singles + teamB.day3.singles
  let totalDoubles := teamA.day1.doubles + teamA.day2.doubles + teamA.day3.doubles +
                      teamB.day1.doubles + teamB.day2.doubles + teamB.day3.doubles
  let totalTriples := teamA.day1.triples + teamA.day2.triples + teamA.day3.triples +
                      teamB.day1.triples + teamB.day2.triples + teamB.day3.triples
  let totalHomeRuns := teamA.day1.homeRuns + teamA.day2.homeRuns + teamA.day3.homeRuns +
                       teamB.day1.homeRuns + teamB.day2.homeRuns + teamB.day3.homeRuns
  totalMisses + totalSingles + totalDoubles + totalTriples + totalHomeRuns

theorem baseball_game_total_is_643 :
  let teamA : TeamStats := {
    day1 := { misses := 60, hits := 30, singles := 15, doubles := 0, triples := 0, homeRuns := 15 },
    day2 := { misses := 68, hits := 17, singles := 11, doubles := 6, triples := 0, homeRuns := 0 },
    day3 := { misses := 100, hits := 20, singles := 10, doubles := 0, triples := 5, homeRuns := 5 }
  }
  let teamB : TeamStats := {
    day1 := { misses := 90, hits := 30, singles := 15, doubles := 0, triples := 0, homeRuns := 15 },
    day2 := { misses := 56, hits := 28, singles := 19, doubles := 9, triples := 0, homeRuns := 0 },
    day3 := { misses := 120, hits := 24, singles := 12, doubles := 0, triples := 6, homeRuns := 6 }
  }
  baseball_game_total teamA teamB = 643 := by
  sorry

#check baseball_game_total_is_643

end NUMINAMATH_CALUDE_baseball_game_total_baseball_game_total_is_643_l1069_106927


namespace NUMINAMATH_CALUDE_exists_function_1995_double_l1069_106988

/-- The number of iterations in the problem -/
def iterations : ℕ := 1995

/-- Definition of function iteration -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, n => n
  | k + 1, n => f (iterate f k n)

/-- Theorem stating the existence of a function satisfying the condition -/
theorem exists_function_1995_double :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, iterate f iterations n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_exists_function_1995_double_l1069_106988


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_l1069_106992

theorem not_arithmetic_sequence : ¬∃ (a d : ℝ) (m n k : ℤ), 
  a + (m - 1 : ℝ) * d = 1 ∧ 
  a + (n - 1 : ℝ) * d = Real.sqrt 2 ∧ 
  a + (k - 1 : ℝ) * d = 3 ∧ 
  n = m + 1 ∧ 
  k = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_l1069_106992


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l1069_106959

theorem difference_of_cubes_divisible_by_eight (a b : ℤ) : 
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 = 8*k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l1069_106959


namespace NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1069_106942

theorem negation_of_proposition_is_true :
  let p := (∀ x y : ℝ, x + y = 5 → x = 2 ∧ y = 3)
  ¬p = True :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1069_106942


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1069_106905

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 8 →
  4*x + 3*y ≤ 63 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1069_106905


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1069_106936

theorem quadratic_real_solutions (m : ℝ) : 
  (∀ x : ℝ, x^2 + x + m = 0 → ∃ y : ℝ, y^2 + y + m = 0) ∧ 
  (∃ n : ℝ, n ≥ 1/4 ∧ ∃ z : ℝ, z^2 + z + n = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1069_106936


namespace NUMINAMATH_CALUDE_election_vote_majority_l1069_106957

/-- In an election with two candidates, prove the vote majority for the winner. -/
theorem election_vote_majority
  (total_votes : ℕ)
  (winning_percentage : ℚ)
  (h_total : total_votes = 700)
  (h_percentage : winning_percentage = 70 / 100) :
  (winning_percentage * total_votes : ℚ).floor -
  ((1 - winning_percentage) * total_votes : ℚ).floor = 280 := by
  sorry

end NUMINAMATH_CALUDE_election_vote_majority_l1069_106957


namespace NUMINAMATH_CALUDE_antonette_age_l1069_106989

theorem antonette_age (a t : ℝ) 
  (h1 : t = 3 * a)  -- Tom is thrice as old as Antonette
  (h2 : a + t = 54) -- The sum of their ages is 54
  : a = 13.5 := by  -- Prove that Antonette's age is 13.5
  sorry

end NUMINAMATH_CALUDE_antonette_age_l1069_106989


namespace NUMINAMATH_CALUDE_inequality_proof_l1069_106903

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_bound_x : x ≥ (1 : ℝ) / 2) (h_bound_y : y ≥ (1 : ℝ) / 2) (h_bound_z : z ≥ (1 : ℝ) / 2)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (1/x + 1/y - 1/z) * (1/x - 1/y + 1/z) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1069_106903


namespace NUMINAMATH_CALUDE_base_8_of_2023_l1069_106913

/-- Converts a base-10 number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The base-8 representation of 2023 (base 10) is 3747 -/
theorem base_8_of_2023 : toBase8 2023 = 3747 := by
  sorry

end NUMINAMATH_CALUDE_base_8_of_2023_l1069_106913


namespace NUMINAMATH_CALUDE_star_example_l1069_106951

/-- The ⬥ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + 2*y) * (x - y)

/-- Theorem stating that 5 ⬥ (2 ⬥ 3) = -143 -/
theorem star_example : star 5 (star 2 3) = -143 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l1069_106951


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_1_l1069_106984

theorem factorization_of_4x_squared_minus_1 (x : ℝ) : 4 * x^2 - 1 = (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_1_l1069_106984


namespace NUMINAMATH_CALUDE_first_day_over_500_l1069_106985

def paperclips (k : ℕ) : ℕ := 4 * 3^k

theorem first_day_over_500 : 
  (∃ k : ℕ, paperclips k > 500) ∧ 
  (∀ j : ℕ, j < 5 → paperclips j ≤ 500) ∧ 
  (paperclips 5 > 500) :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_500_l1069_106985


namespace NUMINAMATH_CALUDE_expansion_equals_difference_of_squares_l1069_106974

theorem expansion_equals_difference_of_squares (x y : ℝ) : 
  (5*y - x) * (-5*y - x) = x^2 - 25*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_difference_of_squares_l1069_106974


namespace NUMINAMATH_CALUDE_repeating_decimal_135_equals_5_37_l1069_106911

def repeating_decimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_135_equals_5_37 :
  repeating_decimal 1 3 5 = 5 / 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_135_equals_5_37_l1069_106911


namespace NUMINAMATH_CALUDE_arithmetic_sum_33_l1069_106975

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum_33 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_33_l1069_106975


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1069_106968

/-- The quadratic function y = x^2 - 4x + 3 is equivalent to y = (x-2)^2 - 1 -/
theorem quadratic_equivalence :
  ∀ x : ℝ, x^2 - 4*x + 3 = (x - 2)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1069_106968


namespace NUMINAMATH_CALUDE_fraction_cubed_l1069_106917

theorem fraction_cubed : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cubed_l1069_106917


namespace NUMINAMATH_CALUDE_cool_drink_jasmine_percentage_l1069_106979

/-- Represents the initial percentage of jasmine water in the solution -/
def initial_percentage : ℝ := 5

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 90

/-- The volume of jasmine added in liters -/
def added_jasmine : ℝ := 8

/-- The volume of water added in liters -/
def added_water : ℝ := 2

/-- The final percentage of jasmine in the solution -/
def final_percentage : ℝ := 12.5

/-- The final volume of the solution in liters -/
def final_volume : ℝ := initial_volume + added_jasmine + added_water

theorem cool_drink_jasmine_percentage :
  (initial_percentage / 100) * initial_volume + added_jasmine = 
  (final_percentage / 100) * final_volume :=
sorry

end NUMINAMATH_CALUDE_cool_drink_jasmine_percentage_l1069_106979


namespace NUMINAMATH_CALUDE_bowling_team_weight_specific_bowling_problem_l1069_106901

/-- Given a bowling team with initial players and weights, prove the weight of a new player --/
theorem bowling_team_weight (initial_players : ℕ) (initial_avg_weight : ℝ) 
  (new_player1_weight : ℝ) (new_avg_weight : ℝ) : ℝ :=
  let total_initial_weight := initial_players * initial_avg_weight
  let new_total_players := initial_players + 2
  let new_total_weight := new_total_players * new_avg_weight
  let new_players_total_weight := new_total_weight - total_initial_weight
  let new_player2_weight := new_players_total_weight - new_player1_weight
  new_player2_weight

/-- The specific bowling team problem --/
theorem specific_bowling_problem : 
  bowling_team_weight 7 76 110 78 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_specific_bowling_problem_l1069_106901


namespace NUMINAMATH_CALUDE_integral_absolute_value_l1069_106923

theorem integral_absolute_value : 
  ∫ x in (0 : ℝ)..2, (2 - |1 - x|) = 3 := by sorry

end NUMINAMATH_CALUDE_integral_absolute_value_l1069_106923


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1069_106978

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^2 + 2 * a * x < 1 - 3 * a) ↔ a < 1/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1069_106978


namespace NUMINAMATH_CALUDE_cucumber_packing_l1069_106918

theorem cucumber_packing (total_cucumbers : ℕ) (basket_capacity : ℕ) 
  (h1 : total_cucumbers = 216)
  (h2 : basket_capacity = 23) :
  ∃ (filled_baskets : ℕ) (remaining_cucumbers : ℕ),
    filled_baskets * basket_capacity + remaining_cucumbers = total_cucumbers ∧
    filled_baskets = 9 ∧
    remaining_cucumbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_packing_l1069_106918


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1069_106991

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-4 : ℝ) 2) = {x | (2 - x) / (x + 4) > 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1069_106991


namespace NUMINAMATH_CALUDE_expression_simplification_l1069_106999

theorem expression_simplification (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 6) :
  (a - b)^2 + b*(3*a - b) - a^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1069_106999


namespace NUMINAMATH_CALUDE_cube_root_nested_expression_l1069_106967

theorem cube_root_nested_expression (x : ℝ) (h : x ≥ 0) :
  (x * (x * x^(1/3))^(1/2))^(1/3) = x^(5/9) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_nested_expression_l1069_106967


namespace NUMINAMATH_CALUDE_cube_cannot_cover_5x5_square_l1069_106990

/-- Represents the four possible directions on a chessboard -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the state of the cube -/
structure CubeState :=
  (position : Position)
  (topFace : Fin 6)
  (faceDirections : Fin 6 → Direction)

/-- The set of all positions a cube can visit given its initial state -/
def visitablePositions (initialState : CubeState) : Set Position :=
  sorry

/-- A 5x5 square on the chessboard -/
def square5x5 (topLeft : Position) : Set Position :=
  { p : Position | 
    topLeft.x ≤ p.x ∧ p.x < topLeft.x + 5 ∧
    topLeft.y - 4 ≤ p.y ∧ p.y ≤ topLeft.y }

/-- Theorem stating that the cube cannot cover any 5x5 square -/
theorem cube_cannot_cover_5x5_square (initialState : CubeState) :
  ∀ topLeft : Position, ¬(square5x5 topLeft ⊆ visitablePositions initialState) :=
sorry

end NUMINAMATH_CALUDE_cube_cannot_cover_5x5_square_l1069_106990


namespace NUMINAMATH_CALUDE_cost_calculation_l1069_106966

theorem cost_calculation (N P M : ℚ) 
  (eq1 : 13 * N + 26 * P + 19 * M = 25)
  (eq2 : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := by
sorry

end NUMINAMATH_CALUDE_cost_calculation_l1069_106966


namespace NUMINAMATH_CALUDE_exponent_simplification_l1069_106921

theorem exponent_simplification :
  3000 * (3000 ^ 2500) * 2 = 2 * 3000 ^ 2501 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l1069_106921


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1069_106971

theorem x_plus_2y_equals_10 (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : x + 2*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1069_106971


namespace NUMINAMATH_CALUDE_writer_book_frequency_l1069_106954

theorem writer_book_frequency
  (years_writing : ℕ)
  (avg_earnings_per_book : ℝ)
  (total_earnings : ℝ)
  (h1 : years_writing = 20)
  (h2 : avg_earnings_per_book = 30000)
  (h3 : total_earnings = 3600000) :
  (years_writing * 12 : ℝ) / (total_earnings / avg_earnings_per_book) = 2 := by
  sorry

end NUMINAMATH_CALUDE_writer_book_frequency_l1069_106954


namespace NUMINAMATH_CALUDE_volunteer_hours_theorem_l1069_106907

/-- Calculates the total hours volunteered per year given the frequency per month and hours per session -/
def total_volunteer_hours_per_year (sessions_per_month : ℕ) (hours_per_session : ℕ) : ℕ :=
  sessions_per_month * 12 * hours_per_session

/-- Proves that volunteering twice a month for 3 hours each time results in 72 hours per year -/
theorem volunteer_hours_theorem :
  total_volunteer_hours_per_year 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_hours_theorem_l1069_106907


namespace NUMINAMATH_CALUDE_fraction_value_l1069_106995

theorem fraction_value (y : ℝ) (h : 4 - 9/y + 9/(y^2) = 0) : 3/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1069_106995


namespace NUMINAMATH_CALUDE_train_speed_l1069_106958

/-- Proves that the speed of a train is 45 km/hr given specific conditions --/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : bridge_length = 225)
  (h3 : crossing_time = 30)
  (h4 : (1 : ℝ) / 3.6 = 1 / 3.6) : -- Conversion factor
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1069_106958


namespace NUMINAMATH_CALUDE_track_width_l1069_106953

theorem track_width (r₁ r₂ : ℝ) : 
  (2 * π * r₁ = 100 * π) →
  (2 * π * r₁ - 2 * π * r₂ = 16 * π) →
  (r₁ - r₂ = 8) := by
sorry

end NUMINAMATH_CALUDE_track_width_l1069_106953


namespace NUMINAMATH_CALUDE_power_of_power_l1069_106933

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1069_106933


namespace NUMINAMATH_CALUDE_lawrence_county_kids_l1069_106922

/-- The number of kids from Lawrence county going to camp -/
def kids_camp : ℕ := 610769

/-- The number of kids from Lawrence county staying home -/
def kids_home : ℕ := 590796

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_camp + kids_home

/-- Theorem stating that the total number of kids in Lawrence county
    is equal to the sum of kids going to camp and kids staying home -/
theorem lawrence_county_kids : total_kids = 1201565 := by sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_l1069_106922


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1069_106970

theorem smallest_integer_with_remainders : 
  ∃ n : ℕ, 
    n > 1 ∧
    n % 3 = 2 ∧ 
    n % 7 = 2 ∧ 
    n % 5 = 1 ∧
    (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 7 = 2 ∧ m % 5 = 1 → n ≤ m) ∧
    n = 86 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1069_106970


namespace NUMINAMATH_CALUDE_sign_selection_theorem_l1069_106939

theorem sign_selection_theorem (n : ℕ) (a : ℕ → ℕ) 
  (h_n : n ≥ 2)
  (h_a : ∀ k ∈ Finset.range n, 0 < a k ∧ a k ≤ k + 1)
  (h_even : Even (Finset.sum (Finset.range n) a)) :
  ∃ f : ℕ → Int, (∀ k, f k = 1 ∨ f k = -1) ∧ 
    Finset.sum (Finset.range n) (λ k => (f k) * (a k)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_selection_theorem_l1069_106939


namespace NUMINAMATH_CALUDE_books_loaned_out_l1069_106997

/-- Proves that the number of books loaned out is 50, given the initial number of books,
    the return rate, and the final number of books. -/
theorem books_loaned_out
  (initial_books : ℕ)
  (return_rate : ℚ)
  (final_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_books = 65) :
  ∃ (loaned_books : ℕ), loaned_books = 50 ∧
    final_books = initial_books - (1 - return_rate) * loaned_books :=
by sorry

end NUMINAMATH_CALUDE_books_loaned_out_l1069_106997


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_l1069_106945

theorem sum_of_squares_quadratic_roots : 
  let a : ℝ := 1
  let b : ℝ := -15
  let c : ℝ := 6
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁^2 + r₂^2 = 213 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_l1069_106945


namespace NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_angle_of_inclination_range_l1069_106980

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Statement for the unique perpendicular tangent line
theorem unique_perpendicular_tangent (a : ℝ) :
  (∃! x : ℝ, f' a x = -1) ↔ a = 3 :=
sorry

-- Statement for the equation of the perpendicular tangent line
theorem perpendicular_tangent_equation (a : ℝ) (h : a = 3) :
  ∃ x y : ℝ, 3*x + y - 8 = 0 ∧ y = f a x ∧ f' a x = -1 :=
sorry

-- Statement for the range of the angle of inclination
theorem angle_of_inclination_range (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, -π/4 < Real.arctan (f' a x) ∧ Real.arctan (f' a x) < π/2 :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_angle_of_inclination_range_l1069_106980


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l1069_106937

theorem tangent_line_minimum_value (k b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧
    k = 1 / (2 * Real.sqrt x₀) ∧
    b = Real.sqrt x₀ / 2 + 1 ∧
    k * x₀ + b = Real.sqrt x₀ + 1) →
  k^2 + b^2 - 2*b ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l1069_106937


namespace NUMINAMATH_CALUDE_power_of_5000_times_2_l1069_106986

theorem power_of_5000_times_2 : ∃ n : ℕ, 2 * (5000 ^ 150) = 10 ^ n ∧ n = 600 := by
  sorry

end NUMINAMATH_CALUDE_power_of_5000_times_2_l1069_106986


namespace NUMINAMATH_CALUDE_work_completion_days_l1069_106981

-- Define the daily work done by a man and a boy
variable (M B : ℝ)

-- Define the total work to be done
variable (W : ℝ)

-- Define the number of days for the first group
variable (D : ℝ)

-- Theorem statement
theorem work_completion_days 
  (h1 : M = 2 * B) -- A man's daily work is twice that of a boy
  (h2 : (13 * M + 24 * B) * 4 = W) -- 13 men and 24 boys complete the work in 4 days
  (h3 : (12 * M + 16 * B) * D = W) -- 12 men and 16 boys complete the work in D days
  : D = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l1069_106981


namespace NUMINAMATH_CALUDE_dilative_rotation_commutes_l1069_106916

/-- A transformation consisting of a rotation and scaling -/
structure DilativeRotation where
  center : ℝ × ℝ
  angle : ℝ
  scale : ℝ

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Apply a dilative rotation to a point -/
def applyDilativeRotation (t : DilativeRotation) (p : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Apply a dilative rotation to a triangle -/
def applyDilativeRotationToTriangle (t : DilativeRotation) (tri : Triangle) : Triangle :=
  sorry

/-- Theorem stating that the order of rotation and scaling is interchangeable -/
theorem dilative_rotation_commutes (t : DilativeRotation) (tri : Triangle) :
  let t1 := DilativeRotation.mk t.center t.angle 1
  let t2 := DilativeRotation.mk t.center 0 t.scale
  applyDilativeRotationToTriangle t2 (applyDilativeRotationToTriangle t1 tri) =
  applyDilativeRotationToTriangle t1 (applyDilativeRotationToTriangle t2 tri) :=
  sorry

end NUMINAMATH_CALUDE_dilative_rotation_commutes_l1069_106916


namespace NUMINAMATH_CALUDE_gcd_problem_l1069_106998

theorem gcd_problem (a : ℤ) (h : 1610 ∣ a) :
  Nat.gcd (Int.natAbs (a^2 + 9*a + 35)) (Int.natAbs (a + 5)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1069_106998


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_digit_removal_l1069_106960

theorem two_numbers_sum_and_digit_removal (x y : ℕ) : 
  x + y = 2014 ∧ 
  3 * (x / 100) = y + 6 ∧ 
  x > y → 
  (x = 1963 ∧ y = 51) ∨ (x = 51 ∧ y = 1963) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_digit_removal_l1069_106960


namespace NUMINAMATH_CALUDE_roof_collapse_leaves_l1069_106932

theorem roof_collapse_leaves (roof_capacity : ℕ) (leaves_per_pound : ℕ) (days_to_collapse : ℕ) :
  roof_capacity = 500 →
  leaves_per_pound = 1000 →
  days_to_collapse = 5000 →
  (roof_capacity * leaves_per_pound) / days_to_collapse = 100 :=
by sorry

end NUMINAMATH_CALUDE_roof_collapse_leaves_l1069_106932


namespace NUMINAMATH_CALUDE_smaller_square_area_l1069_106969

/-- The area of the smaller square formed by inscribing two right triangles in a larger square --/
theorem smaller_square_area (s : ℝ) (h : s = 4) : 
  let diagonal_smaller := s
  let side_smaller := diagonal_smaller / Real.sqrt 2
  side_smaller ^ 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_l1069_106969


namespace NUMINAMATH_CALUDE_sin_sum_max_value_l1069_106983

open Real

theorem sin_sum_max_value (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π) 
  (h₂ : 0 < x₂ ∧ x₂ < π) 
  (h₃ : 0 < x₃ ∧ x₃ < π) 
  (h_sum : x₁ + x₂ + x₃ = π) : 
  sin x₁ + sin x₂ + sin x₃ ≤ 2 * sqrt 3 / 3 := by
  sorry

#check sin_sum_max_value

end NUMINAMATH_CALUDE_sin_sum_max_value_l1069_106983


namespace NUMINAMATH_CALUDE_initial_wall_count_l1069_106987

theorem initial_wall_count (total_containers ceiling_containers leftover_containers tiled_walls : ℕ) 
  (h1 : total_containers = 16)
  (h2 : ceiling_containers = 1)
  (h3 : leftover_containers = 3)
  (h4 : tiled_walls = 1)
  (h5 : ∀ w1 w2 : ℕ, w1 ≠ 0 → w2 ≠ 0 → (total_containers - ceiling_containers - leftover_containers) / w1 = 
                     (total_containers - ceiling_containers - leftover_containers) / w2 → w1 = w2) :
  total_containers - ceiling_containers - leftover_containers + tiled_walls = 13 := by
  sorry

end NUMINAMATH_CALUDE_initial_wall_count_l1069_106987


namespace NUMINAMATH_CALUDE_complex_fourth_power_equality_implies_ratio_one_l1069_106912

theorem complex_fourth_power_equality_implies_ratio_one 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_equality_implies_ratio_one_l1069_106912


namespace NUMINAMATH_CALUDE_tan_product_eighths_pi_l1069_106973

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_pi_l1069_106973


namespace NUMINAMATH_CALUDE_quaternary_201_equals_33_l1069_106928

/-- Converts a quaternary (base-4) number to its decimal (base-10) equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem quaternary_201_equals_33 :
  quaternary_to_decimal [1, 0, 2] = 33 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_201_equals_33_l1069_106928


namespace NUMINAMATH_CALUDE_team_B_better_image_l1069_106972

-- Define the structure for a team
structure Team where
  members : ℕ
  avg_height : ℝ
  height_variance : ℝ

-- Define the two teams
def team_A : Team := { members := 20, avg_height := 160, height_variance := 10.5 }
def team_B : Team := { members := 20, avg_height := 160, height_variance := 1.2 }

-- Define a function to determine which team has a better performance image
def better_performance_image (t1 t2 : Team) : Prop :=
  t1.avg_height = t2.avg_height ∧ t1.height_variance < t2.height_variance

-- Theorem statement
theorem team_B_better_image : 
  better_performance_image team_B team_A :=
sorry

end NUMINAMATH_CALUDE_team_B_better_image_l1069_106972


namespace NUMINAMATH_CALUDE_video_game_shelves_l1069_106914

/-- Calculates the minimum number of shelves needed to display video games -/
def minimum_shelves_needed (total_games : ℕ) (action_games : ℕ) (adventure_games : ℕ) (simulation_games : ℕ) (shelf_capacity : ℕ) (special_display_per_genre : ℕ) : ℕ :=
  let remaining_action := action_games - special_display_per_genre
  let remaining_adventure := adventure_games - special_display_per_genre
  let remaining_simulation := simulation_games - special_display_per_genre
  let action_shelves := (remaining_action + shelf_capacity - 1) / shelf_capacity
  let adventure_shelves := (remaining_adventure + shelf_capacity - 1) / shelf_capacity
  let simulation_shelves := (remaining_simulation + shelf_capacity - 1) / shelf_capacity
  action_shelves + adventure_shelves + simulation_shelves + 1

theorem video_game_shelves :
  minimum_shelves_needed 163 73 51 39 84 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_video_game_shelves_l1069_106914


namespace NUMINAMATH_CALUDE_isosceles_triangle_yw_length_l1069_106931

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  dist t.X t.Z = dist t.Y t.Z

-- Define the point W on XZ
def W (t : Triangle) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem isosceles_triangle_yw_length 
  (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : dist t.X t.Y = 3) 
  (h3 : dist t.X t.Z = 5) 
  (h4 : dist t.Y t.Z = 5) 
  (h5 : dist (W t) t.Z = 2) : 
  dist (W t) t.Y = Real.sqrt 18.5 := 
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_yw_length_l1069_106931


namespace NUMINAMATH_CALUDE_cucumber_price_is_four_l1069_106963

/-- Represents the price of cucumbers per kilo -/
def cucumber_price : ℝ := sorry

/-- Theorem: Given Peter's shopping details, the price of cucumbers per kilo is $4 -/
theorem cucumber_price_is_four :
  let initial_amount : ℝ := 500
  let potatoes_kilo : ℝ := 6
  let potatoes_price : ℝ := 2
  let tomatoes_kilo : ℝ := 9
  let tomatoes_price : ℝ := 3
  let cucumbers_kilo : ℝ := 5
  let bananas_kilo : ℝ := 3
  let bananas_price : ℝ := 5
  let remaining_amount : ℝ := 426
  initial_amount - 
    (potatoes_kilo * potatoes_price + 
     tomatoes_kilo * tomatoes_price + 
     cucumbers_kilo * cucumber_price + 
     bananas_kilo * bananas_price) = remaining_amount →
  cucumber_price = 4 := by sorry

end NUMINAMATH_CALUDE_cucumber_price_is_four_l1069_106963


namespace NUMINAMATH_CALUDE_circle_problem_l1069_106904

noncomputable section

-- Define the line l: y = kx
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define circle C₁: (x-1)² + y² = 1
def circle_C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (3, Real.sqrt 3)

-- Define the tangency condition for C₂ and l at M
def tangent_C₂_l (k : ℝ) : Prop := line_l k 3 (Real.sqrt 3)

-- Define the external tangency condition for C₁ and C₂
def external_tangent_C₁_C₂ (m n R : ℝ) : Prop :=
  (m - 1)^2 + n^2 = (1 + R)^2

-- Main theorem
theorem circle_problem (k : ℝ) :
  (∃ m n R, external_tangent_C₁_C₂ m n R ∧ tangent_C₂_l k) →
  (k = Real.sqrt 3 / 3) ∧
  (∃ A B : ℝ × ℝ, circle_C₁ A.1 A.2 ∧ circle_C₁ B.1 B.2 ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3)) ∧
  (∃ m n : ℝ, ((m = 4 ∧ n = 0) ∨ (m = 0 ∧ n = 4 * Real.sqrt 3)) ∧
    (∀ x y : ℝ, (x - m)^2 + (y - n)^2 = (if m = 4 then 4 else 36))) :=
sorry

end

end NUMINAMATH_CALUDE_circle_problem_l1069_106904


namespace NUMINAMATH_CALUDE_steve_commute_time_l1069_106940

-- Define the parameters
def distance_to_work : ℝ := 35
def speed_back : ℝ := 17.5

-- Define the theorem
theorem steve_commute_time :
  let speed_to_work : ℝ := speed_back / 2
  let time_to_work : ℝ := distance_to_work / speed_to_work
  let time_from_work : ℝ := distance_to_work / speed_back
  let total_time : ℝ := time_to_work + time_from_work
  total_time = 6 := by sorry

end NUMINAMATH_CALUDE_steve_commute_time_l1069_106940


namespace NUMINAMATH_CALUDE_total_fruits_is_137_l1069_106909

/-- The number of fruits picked by George, Amelia, and Olivia -/
def total_fruits (george_oranges : ℕ) (amelia_apples : ℕ) (amelia_orange_diff : ℕ) 
  (george_apple_diff : ℕ) (olivia_time : ℕ) (olivia_orange_rate : ℕ) (olivia_apple_rate : ℕ) 
  (olivia_time_unit : ℕ) : ℕ :=
  let george_apples := amelia_apples + george_apple_diff
  let amelia_oranges := george_oranges - amelia_orange_diff
  let olivia_cycles := olivia_time / olivia_time_unit
  let olivia_oranges := olivia_orange_rate * olivia_cycles
  let olivia_apples := olivia_apple_rate * olivia_cycles
  george_oranges + george_apples + amelia_oranges + amelia_apples + olivia_oranges + olivia_apples

/-- Theorem stating that the total number of fruits picked is 137 -/
theorem total_fruits_is_137 : 
  total_fruits 45 15 18 5 30 3 2 5 = 137 := by
  sorry


end NUMINAMATH_CALUDE_total_fruits_is_137_l1069_106909


namespace NUMINAMATH_CALUDE_third_player_win_probability_probability_third_player_wins_l1069_106930

/-- The probability of winning for the third player in a four-player 
    coin flipping game where players take turns and the first to flip 
    heads wins. -/
theorem third_player_win_probability : ℝ :=
  2 / 63

/-- The game ends when a player flips heads -/
axiom game_ends_on_heads : Prop

/-- There are four players taking turns -/
axiom four_players : Prop

/-- Players take turns in order -/
axiom turns_in_order : Prop

/-- Each flip has a 1/2 probability of heads -/
axiom fair_coin : Prop

theorem probability_third_player_wins : 
  game_ends_on_heads → four_players → turns_in_order → fair_coin →
  third_player_win_probability = 2 / 63 :=
sorry

end NUMINAMATH_CALUDE_third_player_win_probability_probability_third_player_wins_l1069_106930


namespace NUMINAMATH_CALUDE_difference_between_numbers_l1069_106956

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 84) (h2 : a = 36) (h3 : b = 48) :
  b - a = 12 := by
sorry

end NUMINAMATH_CALUDE_difference_between_numbers_l1069_106956


namespace NUMINAMATH_CALUDE_quinary_to_octal_conversion_polynomial_evaluation_l1069_106935

-- Define the polynomial f(x)
def f (x : ℕ) : ℕ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

-- Define the quinary to decimal conversion function
def quinary_to_decimal (q : ℕ) : ℕ :=
  (q / 1000) * 5^3 + ((q / 100) % 10) * 5^2 + ((q / 10) % 10) * 5^1 + (q % 10)

-- Define the decimal to octal conversion function
def decimal_to_octal (d : ℕ) : ℕ :=
  (d / 64) * 100 + ((d / 8) % 8) * 10 + (d % 8)

theorem quinary_to_octal_conversion :
  decimal_to_octal (quinary_to_decimal 1234) = 302 := by sorry

theorem polynomial_evaluation :
  f 3 = 21324 := by sorry

end NUMINAMATH_CALUDE_quinary_to_octal_conversion_polynomial_evaluation_l1069_106935


namespace NUMINAMATH_CALUDE_fibonacci_harmonic_sum_l1069_106977

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the harmonic series
def H : ℕ → ℚ
  | 0 => 0
  | (n + 1) => H n + 1 / (n + 1)

-- State the theorem
theorem fibonacci_harmonic_sum :
  (∑' n : ℕ, (fib (n + 1) : ℚ) / ((n + 2) * H (n + 1) * H (n + 2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_harmonic_sum_l1069_106977


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1069_106982

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1069_106982


namespace NUMINAMATH_CALUDE_complex_square_equation_l1069_106946

theorem complex_square_equation (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  a + b * Complex.I = 4 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_equation_l1069_106946


namespace NUMINAMATH_CALUDE_simplify_expression_l1069_106934

theorem simplify_expression (a b : ℝ) :
  (-2 * a^2 * b)^3 / (-2 * a * b) * (1/3 * a^2 * b^3) = 4/3 * a^7 * b^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1069_106934


namespace NUMINAMATH_CALUDE_porter_monthly_earnings_l1069_106925

/-- Porter's daily wage in dollars -/
def daily_wage : ℕ := 8

/-- Number of regular working days per week -/
def regular_days : ℕ := 5

/-- Overtime bonus rate as a percentage -/
def overtime_bonus_rate : ℕ := 50

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculate Porter's monthly earnings with overtime every week -/
def monthly_earnings_with_overtime : ℕ :=
  let regular_weekly_earnings := daily_wage * regular_days
  let overtime_daily_earnings := daily_wage + (daily_wage * overtime_bonus_rate / 100)
  let weekly_earnings_with_overtime := regular_weekly_earnings + overtime_daily_earnings
  weekly_earnings_with_overtime * weeks_per_month

theorem porter_monthly_earnings :
  monthly_earnings_with_overtime = 208 := by
  sorry

end NUMINAMATH_CALUDE_porter_monthly_earnings_l1069_106925


namespace NUMINAMATH_CALUDE_principal_calculation_l1069_106962

/-- Prove that the principal is 9200 given the specified conditions -/
theorem principal_calculation (r t : ℝ) (h1 : r = 0.12) (h2 : t = 3) : 
  ∃ P : ℝ, P - (P * r * t) = P - 5888 ∧ P = 9200 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1069_106962


namespace NUMINAMATH_CALUDE_green_green_pairs_l1069_106948

/-- Represents the distribution of shirt colors and pairs in a classroom --/
structure Classroom where
  total_students : ℕ
  red_students : ℕ
  green_students : ℕ
  total_pairs : ℕ
  red_red_pairs : ℕ

/-- The theorem states that given the classroom conditions, 
    the number of pairs where both students wear green is 35 --/
theorem green_green_pairs (c : Classroom) 
  (h1 : c.total_students = 144)
  (h2 : c.red_students = 63)
  (h3 : c.green_students = 81)
  (h4 : c.total_pairs = 72)
  (h5 : c.red_red_pairs = 26)
  (h6 : c.total_students = c.red_students + c.green_students) :
  c.total_pairs - c.red_red_pairs - (c.red_students - 2 * c.red_red_pairs) = 35 := by
  sorry

#check green_green_pairs

end NUMINAMATH_CALUDE_green_green_pairs_l1069_106948


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_iff_equal_and_in_set_l1069_106994

def is_valid_pair (m n : ℤ) : Prop :=
  ∃ (q : ℚ), (Real.sqrt (n + Real.sqrt 2016) + Real.sqrt (m - Real.sqrt 2016) : ℝ) = q

def valid_n_set : Set ℤ := {505, 254, 130, 65, 50, 46, 45}

theorem sqrt_sum_rational_iff_equal_and_in_set (m n : ℤ) :
  is_valid_pair m n ↔ (m = n ∧ n ∈ valid_n_set) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_iff_equal_and_in_set_l1069_106994


namespace NUMINAMATH_CALUDE_expected_sixes_is_one_third_l1069_106976

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 
  0 * (prob_not_six ^ 2) + 
  1 * (2 * prob_six * prob_not_six) + 
  2 * (prob_six ^ 2)

theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_sixes_is_one_third_l1069_106976


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1069_106955

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1069_106955


namespace NUMINAMATH_CALUDE_min_value_of_f_l1069_106902

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - Real.sqrt 3 * abs x + 1) + Real.sqrt (x^2 + Real.sqrt 3 * abs x + 3)

theorem min_value_of_f :
  (∀ x : ℝ, f x ≥ Real.sqrt 7) ∧
  f (Real.sqrt 3 / 4) = Real.sqrt 7 ∧
  f (-Real.sqrt 3 / 4) = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1069_106902


namespace NUMINAMATH_CALUDE_age_problem_l1069_106900

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = b / 2 →
  a + b + c + d = 44 →
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1069_106900


namespace NUMINAMATH_CALUDE_final_cost_is_12_l1069_106910

def purchase1 : ℚ := 2.45
def purchase2 : ℚ := 7.60
def purchase3 : ℚ := 3.15
def discount_rate : ℚ := 0.1

def total_before_discount : ℚ := purchase1 + purchase2 + purchase3
def discount_amount : ℚ := total_before_discount * discount_rate
def total_after_discount : ℚ := total_before_discount - discount_amount

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 0.5 then x.floor else x.ceil

theorem final_cost_is_12 :
  round_to_nearest_dollar total_after_discount = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_cost_is_12_l1069_106910


namespace NUMINAMATH_CALUDE_savings_difference_l1069_106926

def initial_amount : ℝ := 10000

def option1_discounts : List ℝ := [0.20, 0.20, 0.10]
def option2_discounts : List ℝ := [0.40, 0.05, 0.05]

def apply_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * (1 - d)) amount

theorem savings_difference : 
  apply_discounts initial_amount option1_discounts - 
  apply_discounts initial_amount option2_discounts = 345 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l1069_106926


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_four_l1069_106947

theorem smallest_n_divisible_by_four :
  ∃ (n : ℕ), (7 * (n - 3)^5 - n^2 + 16*n - 30) % 4 = 0 ∧
  ∀ (m : ℕ), m < n → (7 * (m - 3)^5 - m^2 + 16*m - 30) % 4 ≠ 0 ∧
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_four_l1069_106947


namespace NUMINAMATH_CALUDE_ram_bicycle_sale_loss_percentage_l1069_106961

/-- Calculates the percentage loss on the second bicycle sold by Ram -/
theorem ram_bicycle_sale_loss_percentage :
  let selling_price : ℚ := 990
  let total_cost : ℚ := 1980
  let profit_percentage_first : ℚ := 10 / 100

  let cost_price_first : ℚ := selling_price / (1 + profit_percentage_first)
  let cost_price_second : ℚ := total_cost - cost_price_first
  let loss_second : ℚ := cost_price_second - selling_price
  let loss_percentage_second : ℚ := (loss_second / cost_price_second) * 100

  loss_percentage_second = 25 / 3 := by sorry

end NUMINAMATH_CALUDE_ram_bicycle_sale_loss_percentage_l1069_106961


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1069_106996

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℕ),
    a₁ = 1 →
    aₙ = 46 →
    d = 3 →
    aₙ = a₁ + (n - 1) * d →
    n = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1069_106996


namespace NUMINAMATH_CALUDE_power_result_l1069_106919

theorem power_result (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(2*m - 3*n) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_power_result_l1069_106919


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1069_106915

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (Complex.I * (a - 1) : ℂ).im ≠ 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1069_106915


namespace NUMINAMATH_CALUDE_circle_equation_given_conditions_l1069_106929

/-- A circle C in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle given its center and radius -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the y-axis if its center's x-coordinate equals its radius -/
def tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius ∨ c.center.1 = -c.radius

/-- A point (x, y) lies on the line x - 3y = 0 -/
def on_line (x y : ℝ) : Prop :=
  x - 3*y = 0

theorem circle_equation_given_conditions :
  ∀ (C : Circle),
    tangent_to_y_axis C →
    C.radius = 4 →
    on_line C.center.1 C.center.2 →
    ∀ (x y : ℝ),
      circle_equation C x y ↔ 
        ((x - 4)^2 + (y - 4/3)^2 = 16 ∨ (x + 4)^2 + (y + 4/3)^2 = 16) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_given_conditions_l1069_106929


namespace NUMINAMATH_CALUDE_three_tenths_plus_four_thousandths_l1069_106941

theorem three_tenths_plus_four_thousandths : 
  (3 : ℚ) / 10 + (4 : ℚ) / 1000 = (304 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_three_tenths_plus_four_thousandths_l1069_106941


namespace NUMINAMATH_CALUDE_fourth_quadrant_angle_l1069_106964

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

/-- An angle is in the fourth quadrant if it's between 270° and 360° -/
def is_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360

/-- If α is in the first quadrant, then 360° - α is in the fourth quadrant -/
theorem fourth_quadrant_angle (α : ℝ) (h : is_first_quadrant α) : 
  is_fourth_quadrant (360 - α) := by
  sorry

end NUMINAMATH_CALUDE_fourth_quadrant_angle_l1069_106964


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l1069_106920

theorem inscribed_triangle_area (r : ℝ) (a b c : ℝ) (h_radius : r = 5) 
  (h_ratio : ∃ (k : ℝ), a = 4*k ∧ b = 5*k ∧ c = 6*k) 
  (h_inscribed : c = 2*r) : 
  (1/2 : ℝ) * a * b = 250/9 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l1069_106920


namespace NUMINAMATH_CALUDE_simplify_absolute_value_l1069_106993

theorem simplify_absolute_value : |(-5^2 + 6)| = 19 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_l1069_106993
