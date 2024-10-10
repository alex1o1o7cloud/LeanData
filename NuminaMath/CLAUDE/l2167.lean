import Mathlib

namespace equation_solutions_l2167_216771

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

end equation_solutions_l2167_216771


namespace cheese_options_correct_l2167_216700

/-- Represents the number of cheese options -/
def cheese_options : ℕ := 3

/-- Represents the number of meat options -/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options -/
def vegetable_options : ℕ := 5

/-- Represents the total number of topping combinations -/
def total_combinations : ℕ := 57

/-- Theorem stating that the number of cheese options is correct -/
theorem cheese_options_correct : 
  cheese_options * (meat_options * (vegetable_options - 1) + 
  (meat_options - 1) * vegetable_options) = total_combinations :=
sorry

end cheese_options_correct_l2167_216700


namespace smallest_prime_factor_in_C_l2167_216704

def C : Set Nat := {51, 53, 54, 55, 57}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ 
    (∀ (m : Nat), m ∈ C → 
      (∃ (p : Nat), Nat.Prime p ∧ p ∣ n) → 
      (∃ (q : Nat), Nat.Prime q ∧ q ∣ m ∧ p ≤ q)) ∧
    n = 54 := by
  sorry

end smallest_prime_factor_in_C_l2167_216704


namespace game_probabilities_l2167_216707

/-- Represents the outcome of a single trial -/
inductive Outcome
  | win
  | loss

/-- Represents the result of 4 trials -/
def GameResult := List Outcome

/-- Counts the number of wins in a game result -/
def countWins : GameResult → Nat
  | [] => 0
  | (Outcome.win :: rest) => 1 + countWins rest
  | (Outcome.loss :: rest) => countWins rest

/-- The sample space of all possible game results -/
def sampleSpace : List GameResult := sorry

/-- The probability of an event occurring -/
def probability (event : GameResult → Bool) : Rat :=
  (sampleSpace.filter event).length / sampleSpace.length

/-- Winning at least once -/
def winAtLeastOnce (result : GameResult) : Bool :=
  countWins result ≥ 1

/-- Winning at most twice -/
def winAtMostTwice (result : GameResult) : Bool :=
  countWins result ≤ 2

theorem game_probabilities :
  probability winAtLeastOnce = 5/16 ∧
  probability winAtMostTwice = 11/16 := by
  sorry

end game_probabilities_l2167_216707


namespace average_weight_abc_l2167_216703

/-- Given the average weight of a and b is 40 kg, the average weight of b and c is 44 kg,
    and the weight of b is 33 kg, prove that the average weight of a, b, and c is 45 kg. -/
theorem average_weight_abc (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 44)
  (h3 : b = 33) :
  (a + b + c) / 3 = 45 := by
  sorry


end average_weight_abc_l2167_216703


namespace work_completion_time_l2167_216718

theorem work_completion_time (x_total_days y_completion_days : ℕ) 
  (x_work_days : ℕ) (h1 : x_total_days = 20) (h2 : x_work_days = 10) 
  (h3 : y_completion_days = 12) : 
  (x_total_days * y_completion_days) / (y_completion_days - x_work_days) = 24 :=
by sorry

end work_completion_time_l2167_216718


namespace regular_21gon_symmetry_sum_l2167_216758

theorem regular_21gon_symmetry_sum : 
  let n : ℕ := 21
  let L' : ℕ := n  -- number of lines of symmetry
  let R' : ℚ := 360 / n  -- smallest positive angle of rotational symmetry in degrees
  L' + R' = 38.142857 := by sorry

end regular_21gon_symmetry_sum_l2167_216758


namespace triple_base_square_exponent_l2167_216777

theorem triple_base_square_exponent 
  (a b y : ℝ) 
  (hb : b ≠ 0) 
  (hr : (3 * a) ^ (2 * b) = a ^ b * y ^ b) : 
  y = 9 * a := 
sorry

end triple_base_square_exponent_l2167_216777


namespace john_square_calculation_l2167_216792

theorem john_square_calculation (n : ℕ) (h : n = 50) :
  n^2 + 101 = (n + 1)^2 → n^2 - 99 = (n - 1)^2 := by
  sorry

end john_square_calculation_l2167_216792


namespace sum_of_three_numbers_l2167_216760

theorem sum_of_three_numbers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 := by
  sorry

end sum_of_three_numbers_l2167_216760


namespace solve_clubsuit_equation_l2167_216709

-- Define the ♣ operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- State the theorem
theorem solve_clubsuit_equation :
  ∃ A : ℝ, (clubsuit A 7 = 61) ∧ (A = 2 * Real.sqrt 30 / 3) :=
by sorry

end solve_clubsuit_equation_l2167_216709


namespace smallest_n_congruence_l2167_216765

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (3 * n) % 30 = 2412 % 30 ∧ ∀ (m : ℕ), m > 0 → (3 * m) % 30 = 2412 % 30 → n ≤ m :=
by sorry

end smallest_n_congruence_l2167_216765


namespace third_piece_coverage_l2167_216772

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

end third_piece_coverage_l2167_216772


namespace no_solution_exists_l2167_216751

theorem no_solution_exists (a b : ℝ) : a^2 + 3*b^2 + 2 > 3*a*b := by
  sorry

end no_solution_exists_l2167_216751


namespace inverse_variation_problem_l2167_216795

/-- Given that a² and √b vary inversely, if a = 3 when b = 64, then b = 18 when ab = 72 -/
theorem inverse_variation_problem (a b : ℝ) : 
  (∃ k : ℝ, ∀ a b : ℝ, a^2 * Real.sqrt b = k) →  -- a² and √b vary inversely
  (3^2 * Real.sqrt 64 = 3 * 64) →                -- a = 3 when b = 64
  (a * b = 72) →                                 -- ab = 72
  b = 18 := by
sorry

end inverse_variation_problem_l2167_216795


namespace quadratic_roots_opposite_l2167_216749

theorem quadratic_roots_opposite (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 2*a)*x + (a - 1) = 0 ∧ 
               y^2 + (a^2 - 2*a)*y + (a - 1) = 0 ∧ 
               x = -y) → 
  a = 0 := by
sorry

end quadratic_roots_opposite_l2167_216749


namespace largest_n_unique_k_l2167_216715

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n = 63 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 8/15)) :=
by sorry

end largest_n_unique_k_l2167_216715


namespace area_30_60_90_triangle_l2167_216734

theorem area_30_60_90_triangle (a : ℝ) (h : a > 0) :
  let triangle_area := (1/2) * a * (a / Real.sqrt 3)
  triangle_area = (32 * Real.sqrt 3) / 3 ↔ a = 8 := by
sorry

end area_30_60_90_triangle_l2167_216734


namespace photo_difference_l2167_216773

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

end photo_difference_l2167_216773


namespace sum_of_q_p_x_values_l2167_216720

def p (x : ℝ) : ℝ := |x| + 1

def q (x : ℝ) : ℝ := -x^2

def x_values : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_q_p_x_values :
  (x_values.map (λ x => q (p x))).sum = -59 := by sorry

end sum_of_q_p_x_values_l2167_216720


namespace opposite_of_negative_two_l2167_216767

theorem opposite_of_negative_two : 
  ∃ x : ℤ, -x = -2 ∧ x = 2 := by sorry

end opposite_of_negative_two_l2167_216767


namespace streetlight_distance_l2167_216716

/-- The distance between streetlights in meters -/
def interval : ℝ := 60

/-- The number of streetlights -/
def num_streetlights : ℕ := 45

/-- The distance from the first to the last streetlight in kilometers -/
def distance_km : ℝ := 2.64

theorem streetlight_distance :
  (interval * (num_streetlights - 1)) / 1000 = distance_km := by
  sorry

end streetlight_distance_l2167_216716


namespace abscissa_of_point_M_l2167_216766

/-- Given a point M with coordinates (1,1), prove that its abscissa is 1 -/
theorem abscissa_of_point_M (M : ℝ × ℝ) (h : M = (1, 1)) : M.1 = 1 := by
  sorry

end abscissa_of_point_M_l2167_216766


namespace ones_digit_of_33_power_power_of_3_cycle_power_mod_4_main_theorem_l2167_216780

theorem ones_digit_of_33_power (n : ℕ) : n > 0 → (33^n) % 10 = (3^n) % 10 := by sorry

theorem power_of_3_cycle (n : ℕ) : (3^n) % 10 = (3^(n % 4)) % 10 := by sorry

theorem power_mod_4 (a b : ℕ) : a > 0 → b > 0 → (a^b) % 4 = (a % 4)^(b % 4) % 4 := by sorry

theorem main_theorem : (33^(33 * 7^7)) % 10 = 7 := by sorry

end ones_digit_of_33_power_power_of_3_cycle_power_mod_4_main_theorem_l2167_216780


namespace log_equation_solution_l2167_216793

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9 →
  x = 2^(54/5) := by
sorry

end log_equation_solution_l2167_216793


namespace expression_equals_hundred_l2167_216757

theorem expression_equals_hundred : (7.5 * 7.5 + 37.5 + 2.5 * 2.5) = 100 := by
  sorry

end expression_equals_hundred_l2167_216757


namespace cube_root_unity_sum_l2167_216744

theorem cube_root_unity_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^2005 + z^2006 + z^2008 + z^2009 = -2 := by sorry

end cube_root_unity_sum_l2167_216744


namespace tangent_line_quadratic_l2167_216752

theorem tangent_line_quadratic (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∀ x : ℝ, x + 1 = (0^2 + a*0 + b) + (2*0 + a)*x) →
  a = 1 ∧ b = 1 := by
sorry

end tangent_line_quadratic_l2167_216752


namespace stock_worth_l2167_216783

theorem stock_worth (X : ℝ) : 
  (0.2 * X * 1.1 + 0.8 * X * 0.95) - X = -250 → X = 12500 := by
  sorry

end stock_worth_l2167_216783


namespace jimmy_change_l2167_216779

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

end jimmy_change_l2167_216779


namespace problem_solution_l2167_216797

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 1)^2 + 16/(x - 1)^2 = 7 + 3 * Real.sqrt 5 := by
  sorry

end problem_solution_l2167_216797


namespace circles_intersection_range_l2167_216731

/-- Two circles C₁ and C₂ defined by their equations -/
def C₁ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 4 = 0
def C₂ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0

/-- The condition for two circles to intersect -/
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, C₁ m x y ∧ C₂ m x y

/-- The theorem stating the range of m for which the circles intersect -/
theorem circles_intersection_range :
  ∀ m : ℝ, circles_intersect m ↔ (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
sorry

end circles_intersection_range_l2167_216731


namespace mel_weight_is_70_l2167_216740

/-- Mel's weight in pounds -/
def mel_weight : ℝ := 70

/-- Brenda's weight in pounds -/
def brenda_weight : ℝ := 220

/-- Theorem stating that Mel's weight is 70 pounds, given the problem conditions -/
theorem mel_weight_is_70 : 
  mel_weight = 70 ∧ 
  brenda_weight = 3 * mel_weight + 10 ∧ 
  brenda_weight = 220 := by
  sorry

end mel_weight_is_70_l2167_216740


namespace smallest_satisfying_number_correct_l2167_216782

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

end smallest_satisfying_number_correct_l2167_216782


namespace same_terminal_side_as_610_degrees_l2167_216712

theorem same_terminal_side_as_610_degrees :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 + 250) ↔ (∃ n : ℤ, θ = n * 360 + 610) :=
by sorry

end same_terminal_side_as_610_degrees_l2167_216712


namespace george_coin_value_l2167_216789

/-- Calculates the total value of coins given the number of nickels and dimes -/
def totalCoinValue (totalCoins : ℕ) (nickels : ℕ) (nickelValue : ℚ) (dimeValue : ℚ) : ℚ :=
  let dimes := totalCoins - nickels
  nickels * nickelValue + dimes * dimeValue

theorem george_coin_value :
  totalCoinValue 28 4 (5 / 100) (10 / 100) = 260 / 100 := by
  sorry

end george_coin_value_l2167_216789


namespace g_72_value_l2167_216762

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

end g_72_value_l2167_216762


namespace x_range_l2167_216788

theorem x_range (x : ℝ) 
  (h : ∀ a b : ℝ, a^2 + b^2 = 1 → a + Real.sqrt 3 * b ≤ |x^2 - 1|) : 
  x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 := by
sorry

end x_range_l2167_216788


namespace cistern_emptying_l2167_216726

/-- If a pipe can empty 3/4 of a cistern in 12 minutes, then it will empty 1/2 of the cistern in 8 minutes. -/
theorem cistern_emptying (empty_rate : ℚ) (empty_time : ℕ) (target_time : ℕ) :
  empty_rate = 3/4 ∧ empty_time = 12 ∧ target_time = 8 →
  (target_time : ℚ) * (empty_rate / empty_time) = 1/2 := by
  sorry

end cistern_emptying_l2167_216726


namespace quadratic_root_condition_l2167_216733

/-- For the equation 7x^2-(m+13)x+m^2-m-2=0 to have one root greater than 1 
    and one root less than 1, m must satisfy -2 < m < 4 -/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ < 1 ∧ 
    7 * x₁^2 - (m + 13) * x₁ + m^2 - m - 2 = 0 ∧
    7 * x₂^2 - (m + 13) * x₂ + m^2 - m - 2 = 0) ↔
  -2 < m ∧ m < 4 :=
by sorry

end quadratic_root_condition_l2167_216733


namespace tiling_count_is_27_l2167_216701

/-- Represents a 2 × 24 grid divided into three 3 × 8 subrectangles -/
structure Grid :=
  (subrectangles : Fin 3 → Unit)

/-- Represents the number of ways to tile a single 3 × 8 subrectangle -/
def subrectangle_tiling_count : ℕ := 3

/-- Calculates the total number of ways to tile the entire 2 × 24 grid -/
def total_tiling_count (g : Grid) : ℕ :=
  subrectangle_tiling_count ^ 3

/-- Theorem stating that the total number of tiling ways is 27 -/
theorem tiling_count_is_27 (g : Grid) : total_tiling_count g = 27 := by
  sorry

end tiling_count_is_27_l2167_216701


namespace initial_men_correct_l2167_216717

/-- Represents the initial number of men employed by Nhai -/
def initial_men : ℕ := 100

/-- Represents the total length of the highway in kilometers -/
def highway_length : ℕ := 2

/-- Represents the initial number of days allocated for the project -/
def total_days : ℕ := 50

/-- Represents the initial number of work hours per day -/
def initial_hours_per_day : ℕ := 8

/-- Represents the number of days after which 1/3 of the work is completed -/
def days_for_one_third : ℕ := 25

/-- Represents the fraction of work completed after 25 days -/
def work_completed_fraction : ℚ := 1/3

/-- Represents the number of additional men hired -/
def additional_men : ℕ := 60

/-- Represents the new number of work hours per day after hiring additional men -/
def new_hours_per_day : ℕ := 10

theorem initial_men_correct :
  initial_men * total_days * initial_hours_per_day =
  (initial_men + additional_men) * (total_days - days_for_one_third) * new_hours_per_day :=
by sorry

end initial_men_correct_l2167_216717


namespace xyz_value_l2167_216750

theorem xyz_value (x y z s : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 12)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
  (h3 : x + y + z = s) :
  x * y * z = -8 := by
sorry

end xyz_value_l2167_216750


namespace escalator_steps_l2167_216714

theorem escalator_steps :
  ∀ (n : ℕ) (k : ℚ),
    k > 0 →
    (18 / k) * (k + 1) = n →
    (27 / (2 * k)) * (2 * k + 1) = n →
    n = 54 := by
  sorry

end escalator_steps_l2167_216714


namespace auditorium_sampling_is_systematic_l2167_216743

structure Auditorium where
  rows : Nat
  seats_per_row : Nat

def systematic_sampling (a : Auditorium) (seat_number : Nat) : Prop :=
  seat_number > 0 ∧ 
  seat_number ≤ a.seats_per_row ∧
  ∀ (row : Nat), row > 0 → row ≤ a.rows → 
    ∃ (student : Nat), student = (row - 1) * a.seats_per_row + seat_number

theorem auditorium_sampling_is_systematic (a : Auditorium) (h1 : a.rows = 30) (h2 : a.seats_per_row = 20) : 
  systematic_sampling a 15 :=
sorry

end auditorium_sampling_is_systematic_l2167_216743


namespace quadratic_function_properties_l2167_216730

theorem quadratic_function_properties (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end quadratic_function_properties_l2167_216730


namespace dans_initial_cards_l2167_216732

/-- The number of baseball cards Dan had initially -/
def initial_cards : ℕ := sorry

/-- The number of torn cards -/
def torn_cards : ℕ := 8

/-- The number of cards Sam bought -/
def cards_sold : ℕ := 15

/-- The number of cards Dan has after selling to Sam -/
def remaining_cards : ℕ := 82

theorem dans_initial_cards : initial_cards = 105 := by
  sorry

end dans_initial_cards_l2167_216732


namespace least_three_digit_multiple_of_nine_l2167_216796

theorem least_three_digit_multiple_of_nine : 
  ∀ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ n % 9 = 0 → n ≥ 108 :=
by
  sorry

end least_three_digit_multiple_of_nine_l2167_216796


namespace julie_school_year_hours_l2167_216756

/-- Given Julie's summer work details and school year earnings goal, calculate her required weekly hours during the school year. -/
theorem julie_school_year_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) 
  (school_year_earnings : ℕ) 
  (h1 : summer_weeks = 12)
  (h2 : summer_hours_per_week = 40)
  (h3 : summer_earnings = 6000)
  (h4 : school_year_weeks = 36)
  (h5 : school_year_earnings = 9000) :
  (school_year_earnings * summer_weeks * summer_hours_per_week) / 
  (summer_earnings * school_year_weeks) = 20 := by
sorry

end julie_school_year_hours_l2167_216756


namespace prime_pairs_perfect_square_l2167_216725

theorem prime_pairs_perfect_square :
  ∀ a b : ℕ,
  Prime a → Prime b → a > 0 → b > 0 →
  (∃ k : ℕ, 3 * a^2 * b + 16 * a * b^2 = k^2) →
  ((a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3)) :=
sorry

end prime_pairs_perfect_square_l2167_216725


namespace square_area_from_diagonal_l2167_216775

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end square_area_from_diagonal_l2167_216775


namespace complex_number_opposite_parts_l2167_216747

theorem complex_number_opposite_parts (a : ℝ) : 
  let z : ℂ := a / (1 - 2*I) + Complex.abs I
  (Complex.re z = -Complex.im z) → a = -5/3 := by
sorry

end complex_number_opposite_parts_l2167_216747


namespace unique_solution_trigonometric_equation_l2167_216724

theorem unique_solution_trigonometric_equation :
  ∃! x : Real,
    0 < x ∧ x < 180 ∧
    Real.tan (120 - x) = (Real.sin 120 - Real.sin x) / (Real.cos 120 - Real.cos x) ∧
    x = 100 := by
  sorry

end unique_solution_trigonometric_equation_l2167_216724


namespace function_value_at_three_l2167_216764

theorem function_value_at_three (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := by
  sorry

end function_value_at_three_l2167_216764


namespace range_of_a_l2167_216721

def f : Set ℝ → Set ℝ := sorry

def A : Set ℝ := {x | ∃ y ∈ Set.Icc 7 15, f {y} = {2 * x + 1}}

def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ A ∨ x ∈ B a) ↔ 3 ≤ a ∧ a < 6 := by sorry

end range_of_a_l2167_216721


namespace ant_position_100_l2167_216710

-- Define the ant's position after n steps
def ant_position (n : ℕ) : ℕ × ℕ :=
  (n * (n + 1) / 2, n * (n + 1) / 2)

-- Theorem statement
theorem ant_position_100 : ant_position 100 = (5050, 5050) := by
  sorry

end ant_position_100_l2167_216710


namespace quadratic_function_theorem_l2167_216755

/-- Given a quadratic function y = ax² + bx - 1 where a ≠ 0, 
    if the graph passes through the point (1, 1), then a + b + 1 = 3 -/
theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  (a * 1^2 + b * 1 - 1 = 1) → (a + b + 1 = 3) := by
  sorry

end quadratic_function_theorem_l2167_216755


namespace alvin_egg_rolls_l2167_216738

/-- Given the egg roll consumption of Matthew, Patrick, and Alvin, prove that Alvin ate 4 egg rolls. -/
theorem alvin_egg_rolls (matthew patrick alvin : ℕ) : 
  matthew = 3 * patrick →  -- Matthew eats three times as many egg rolls as Patrick
  patrick = alvin / 2 →    -- Patrick eats half as many egg rolls as Alvin
  matthew = 6 →            -- Matthew ate 6 egg rolls
  alvin = 4 := by           -- Prove that Alvin ate 4 egg rolls
sorry

end alvin_egg_rolls_l2167_216738


namespace inequality_proof_l2167_216799

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
  sorry

end inequality_proof_l2167_216799


namespace complex_equation_first_quadrant_l2167_216786

/-- Given a complex equation, prove the resulting point is in the first quadrant -/
theorem complex_equation_first_quadrant (a b : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = b + Complex.I → 
  a > 0 ∧ b > 0 := by
  sorry

end complex_equation_first_quadrant_l2167_216786


namespace problem_solution_l2167_216753

/-- If 2a - b + 3 = 0, then 2(2a + b) - 4b = -6 -/
theorem problem_solution (a b : ℝ) (h : 2*a - b + 3 = 0) : 
  2*(2*a + b) - 4*b = -6 := by
sorry

end problem_solution_l2167_216753


namespace stationery_cost_l2167_216706

/-- Given the cost of stationery items, prove the total cost of a specific combination. -/
theorem stationery_cost (E P M : ℕ) : 
  (E + 3 * P + 2 * M = 240) →
  (2 * E + 5 * P + 4 * M = 440) →
  (3 * E + 4 * P + 6 * M = 520) :=
by sorry

end stationery_cost_l2167_216706


namespace restaurant_bill_l2167_216776

theorem restaurant_bill (food_cost : ℝ) (service_fee_percent : ℝ) (tip : ℝ) : 
  food_cost = 50 ∧ service_fee_percent = 12 ∧ tip = 5 →
  food_cost + (service_fee_percent / 100) * food_cost + tip = 61 :=
by sorry

end restaurant_bill_l2167_216776


namespace organization_growth_l2167_216736

/-- Represents the number of people in the organization after a given number of years. -/
def people_count (initial_total : ℕ) (leaders : ℕ) (years : ℕ) : ℕ :=
  leaders + (initial_total - leaders) * (3^years)

/-- Theorem stating the number of people in the organization after 5 years. -/
theorem organization_growth :
  people_count 15 5 5 = 2435 := by
  sorry

#eval people_count 15 5 5

end organization_growth_l2167_216736


namespace max_sum_with_lcm_constraint_max_sum_with_lcm_constraint_achievable_l2167_216708

theorem max_sum_with_lcm_constraint (m n : ℕ) : 
  m > 0 → n > 0 → m < 500 → n < 500 → Nat.lcm m n = (m - n)^2 → m + n ≤ 840 := by
  sorry

theorem max_sum_with_lcm_constraint_achievable : 
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m < 500 ∧ n < 500 ∧ Nat.lcm m n = (m - n)^2 ∧ m + n = 840 := by
  sorry

end max_sum_with_lcm_constraint_max_sum_with_lcm_constraint_achievable_l2167_216708


namespace valid_sequences_count_l2167_216735

/-- Represents the colors of the houses -/
inductive Color
  | Orange
  | Red
  | Blue
  | Yellow
  | Green
  | Purple

/-- A sequence of colored houses -/
def HouseSequence := List Color

/-- Checks if a color appears before another in a sequence -/
def appearsBefore (c1 c2 : Color) (seq : HouseSequence) : Prop :=
  ∃ i j, i < j ∧ seq.getD i c1 = c1 ∧ seq.getD j c2 = c2

/-- Checks if two colors are adjacent in a sequence -/
def areAdjacent (c1 c2 : Color) (seq : HouseSequence) : Prop :=
  ∃ i, (seq.getD i c1 = c1 ∧ seq.getD (i+1) c2 = c2) ∨ 
       (seq.getD i c2 = c2 ∧ seq.getD (i+1) c1 = c1)

/-- Checks if a sequence is valid according to the given conditions -/
def isValidSequence (seq : HouseSequence) : Prop :=
  seq.length = 6 ∧ 
  seq.Nodup ∧
  appearsBefore Color.Orange Color.Red seq ∧
  appearsBefore Color.Blue Color.Yellow seq ∧
  areAdjacent Color.Red Color.Green seq ∧
  ¬(areAdjacent Color.Blue Color.Yellow seq) ∧
  ¬(areAdjacent Color.Blue Color.Red seq)

/-- The main theorem to be proved -/
theorem valid_sequences_count :
  ∃! (validSeqs : List HouseSequence), 
    (∀ seq, seq ∈ validSeqs ↔ isValidSequence seq) ∧ 
    validSeqs.length = 3 := by sorry

end valid_sequences_count_l2167_216735


namespace max_sum_squares_fibonacci_l2167_216748

theorem max_sum_squares_fibonacci (m n : ℕ) : 
  m ∈ Finset.range 1982 → 
  n ∈ Finset.range 1982 → 
  (n^2 - m*n - m^2)^2 = 1 → 
  m^2 + n^2 ≤ 3524578 := by
sorry

end max_sum_squares_fibonacci_l2167_216748


namespace coin_flip_sequences_l2167_216784

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

end coin_flip_sequences_l2167_216784


namespace juanita_dessert_cost_l2167_216794

/-- Represents the menu prices and discounts for the brownie dessert --/
structure BrownieMenu where
  brownie_base : ℝ := 2.50
  regular_scoop : ℝ := 1.00
  premium_scoop : ℝ := 1.25
  deluxe_scoop : ℝ := 1.50
  syrup : ℝ := 0.50
  nuts : ℝ := 1.50
  whipped_cream : ℝ := 0.75
  cherry : ℝ := 0.25
  tuesday_discount : ℝ := 0.10
  wednesday_discount : ℝ := 0.50
  sunday_discount : ℝ := 0.25

/-- Represents Juanita's order --/
structure JuanitaOrder where
  regular_scoops : ℕ := 2
  premium_scoops : ℕ := 1
  deluxe_scoops : ℕ := 0
  syrups : ℕ := 2
  has_nuts : Bool := true
  has_whipped_cream : Bool := true
  has_cherry : Bool := true

/-- Calculates the total cost of Juanita's dessert --/
def calculate_total_cost (menu : BrownieMenu) (order : JuanitaOrder) : ℝ :=
  let discounted_brownie := menu.brownie_base * (1 - menu.tuesday_discount)
  let ice_cream_cost := order.regular_scoops * menu.regular_scoop + 
                        order.premium_scoops * menu.premium_scoop + 
                        order.deluxe_scoops * menu.deluxe_scoop
  let syrup_cost := order.syrups * menu.syrup
  let topping_cost := (if order.has_nuts then menu.nuts else 0) +
                      (if order.has_whipped_cream then menu.whipped_cream else 0) +
                      (if order.has_cherry then menu.cherry else 0)
  discounted_brownie + ice_cream_cost + syrup_cost + topping_cost

/-- Theorem stating that Juanita's dessert costs $9.00 --/
theorem juanita_dessert_cost (menu : BrownieMenu) (order : JuanitaOrder) :
  calculate_total_cost menu order = 9.00 := by
  sorry


end juanita_dessert_cost_l2167_216794


namespace word_problem_points_word_problem_points_correct_l2167_216719

theorem word_problem_points (total_problems : ℕ) (computation_problems : ℕ) 
  (computation_points : ℕ) (total_points : ℕ) : ℕ :=
  let word_problems := total_problems - computation_problems
  let computation_total := computation_problems * computation_points
  let word_total := total_points - computation_total
  word_total / word_problems

#check word_problem_points 30 20 3 110 = 5

theorem word_problem_points_correct : 
  word_problem_points 30 20 3 110 = 5 := by sorry

end word_problem_points_word_problem_points_correct_l2167_216719


namespace cubic_three_monotonic_intervals_l2167_216723

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The derivative of f -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem cubic_three_monotonic_intervals (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f_deriv a x = 0 ∧ f_deriv a y = 0) ↔ a < 0 := by
  sorry

end cubic_three_monotonic_intervals_l2167_216723


namespace area_to_paint_dining_room_l2167_216741

/-- The area to be painted on a wall with a painting hanging on it -/
def area_to_paint (wall_height wall_length painting_height painting_length : ℝ) : ℝ :=
  wall_height * wall_length - painting_height * painting_length

/-- Theorem: The area to be painted is 135 square feet -/
theorem area_to_paint_dining_room : 
  area_to_paint 10 15 3 5 = 135 := by
  sorry

end area_to_paint_dining_room_l2167_216741


namespace ac_plus_bd_equals_23_l2167_216763

theorem ac_plus_bd_equals_23 
  (a b c d : ℝ) 
  (h1 : a + b + c = 6)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = -9) :
  a * c + b * d = 23 := by
sorry

end ac_plus_bd_equals_23_l2167_216763


namespace polar_to_cartesian_equivalence_polar_equation_is_line_l2167_216722

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (Real.sin θ + Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop := y + x = 1

-- Theorem statement
theorem polar_to_cartesian_equivalence :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    line_equation x y := by
  sorry

-- The main theorem stating that the polar equation represents a line
theorem polar_equation_is_line :
  ∃ (a b c : ℝ), ∀ (r θ : ℝ), 
    polar_equation r θ → 
    ∃ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ a*x + b*y = c := by
  sorry

end polar_to_cartesian_equivalence_polar_equation_is_line_l2167_216722


namespace dynaco_price_is_44_l2167_216705

/-- Calculates the price per share of Dynaco stock given the total number of shares sold,
    number of Dynaco shares sold, price per share of Microtron stock, and average price
    per share of all stocks sold. -/
def dynaco_price_per_share (total_shares : ℕ) (dynaco_shares : ℕ) 
    (microtron_price : ℚ) (average_price : ℚ) : ℚ :=
  let microtron_shares := total_shares - dynaco_shares
  let total_revenue := (total_shares : ℚ) * average_price
  let microtron_revenue := (microtron_shares : ℚ) * microtron_price
  (total_revenue - microtron_revenue) / (dynaco_shares : ℚ)

/-- Theorem stating that given the specific conditions from the problem,
    the price per share of Dynaco stock is $44. -/
theorem dynaco_price_is_44 :
  dynaco_price_per_share 300 150 36 40 = 44 := by
  sorry

end dynaco_price_is_44_l2167_216705


namespace obtuse_angles_are_second_quadrant_l2167_216787

-- Define angle types
def ObtuseAngle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def SecondQuadrantAngle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def FirstQuadrantAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def ThirdQuadrantAngle (θ : ℝ) : Prop := -180 < θ ∧ θ < -90

-- Theorem statement
theorem obtuse_angles_are_second_quadrant :
  ∀ θ : ℝ, ObtuseAngle θ ↔ SecondQuadrantAngle θ := by
  sorry

end obtuse_angles_are_second_quadrant_l2167_216787


namespace solve_marble_problem_l2167_216739

def marble_problem (katrina_marbles : ℕ) : Prop :=
  let amanda_marbles : ℕ := 2 * katrina_marbles - 12
  let mabel_marbles : ℕ := 5 * katrina_marbles
  let carlos_marbles : ℕ := 3 * katrina_marbles
  mabel_marbles = 85 ∧ mabel_marbles - (amanda_marbles + carlos_marbles) = 12

theorem solve_marble_problem :
  ∃ (katrina_marbles : ℕ), marble_problem katrina_marbles := by
  sorry

end solve_marble_problem_l2167_216739


namespace tan_eq_two_solution_set_l2167_216727

theorem tan_eq_two_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + (-1)^k * Real.arctan 2} =
  {x : ℝ | Real.tan x = 2} := by sorry

end tan_eq_two_solution_set_l2167_216727


namespace ellipse_major_axis_length_l2167_216798

/-- Given an ellipse with equation 16x^2 + 9y^2 = 144, its major axis length is 8 -/
theorem ellipse_major_axis_length :
  ∀ (x y : ℝ), 16 * x^2 + 9 * y^2 = 144 → ∃ (a b : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    ((a ≥ b ∧ 2 * a = 8) ∨ (b > a ∧ 2 * b = 8)) :=
by sorry

end ellipse_major_axis_length_l2167_216798


namespace carter_road_trip_l2167_216791

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

end carter_road_trip_l2167_216791


namespace circle_and_inscribed_square_l2167_216761

/-- Given a circle with circumference 72π and an inscribed square with vertices touching the circle,
    prove that the radius is 36 and the side length of the square is 36√2. -/
theorem circle_and_inscribed_square (C : ℝ) (r : ℝ) (s : ℝ) :
  C = 72 * Real.pi →  -- Circumference of the circle
  C = 2 * Real.pi * r →  -- Relation between circumference and radius
  s^2 * 2 = (2 * r)^2 →  -- Relation between square side and circle diameter
  r = 36 ∧ s = 36 * Real.sqrt 2 := by
  sorry

end circle_and_inscribed_square_l2167_216761


namespace six_digit_divisible_by_9_and_22_l2167_216742

theorem six_digit_divisible_by_9_and_22 : ∃! n : ℕ, 
  220140 ≤ n ∧ n < 220150 ∧ 
  n % 9 = 0 ∧ 
  n % 22 = 0 ∧
  n = 520146 := by
sorry

end six_digit_divisible_by_9_and_22_l2167_216742


namespace scientific_notation_of_113800_l2167_216781

theorem scientific_notation_of_113800 :
  ∃ (a : ℝ) (n : ℤ), 
    113800 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ a < 10 ∧
    a = 1.138 ∧ n = 5 := by
  sorry

end scientific_notation_of_113800_l2167_216781


namespace stratum_c_sample_size_l2167_216713

/-- Calculates the sample size for a stratum in stratified sampling -/
def stratumSampleSize (stratumSize : ℕ) (totalPopulation : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (stratumSize * totalSampleSize) / totalPopulation

theorem stratum_c_sample_size :
  let stratum_a_size : ℕ := 400
  let stratum_b_size : ℕ := 800
  let stratum_c_size : ℕ := 600
  let total_population : ℕ := stratum_a_size + stratum_b_size + stratum_c_size
  let total_sample_size : ℕ := 90
  stratumSampleSize stratum_c_size total_population total_sample_size = 30 := by
  sorry

#eval stratumSampleSize 600 1800 90

end stratum_c_sample_size_l2167_216713


namespace mod_19_equivalence_l2167_216774

theorem mod_19_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ 42568 % 19 = n % 19 ∧ n = 3 := by
  sorry

end mod_19_equivalence_l2167_216774


namespace pattern_equality_l2167_216770

theorem pattern_equality (n : ℕ) (h : n > 1) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) :=
by sorry

end pattern_equality_l2167_216770


namespace remainder_98_102_div_11_l2167_216702

theorem remainder_98_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end remainder_98_102_div_11_l2167_216702


namespace freelancer_earnings_l2167_216737

def calculate_final_amount (initial_amount : ℚ) : ℚ :=
  let first_client_payment := initial_amount / 2
  let second_client_payment := first_client_payment * (1 + 2/5)
  let third_client_payment := 2 * (first_client_payment + second_client_payment)
  let average_first_three := (first_client_payment + second_client_payment + third_client_payment) / 3
  let fourth_client_payment := average_first_three * (1 + 1/10)
  initial_amount + first_client_payment + second_client_payment + third_client_payment + fourth_client_payment

theorem freelancer_earnings (initial_amount : ℚ) :
  initial_amount = 4000 → calculate_final_amount initial_amount = 23680 :=
by sorry

end freelancer_earnings_l2167_216737


namespace min_value_geometric_sequence_l2167_216746

/-- Given a geometric sequence with first term a₁ = 2, 
    the smallest possible value of 6a₂ + 7a₃ is -18/7 -/
theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) : 
  a₁ = 2 → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) → 
  (∀ b₂ b₃ : ℝ, (∃ s : ℝ, b₂ = a₁ * s ∧ b₃ = b₂ * s) → 
    6 * a₂ + 7 * a₃ ≤ 6 * b₂ + 7 * b₃) → 
  6 * a₂ + 7 * a₃ = -18/7 :=
sorry

end min_value_geometric_sequence_l2167_216746


namespace truck_journey_distance_l2167_216729

/-- A problem about a semi truck's journey on paved and dirt roads. -/
theorem truck_journey_distance :
  let time_paved : ℝ := 2 -- Time spent on paved road in hours
  let time_dirt : ℝ := 3 -- Time spent on dirt road in hours
  let speed_dirt : ℝ := 32 -- Speed on dirt road in mph
  let speed_paved : ℝ := speed_dirt + 20 -- Speed on paved road in mph
  let distance_dirt : ℝ := speed_dirt * time_dirt -- Distance on dirt road
  let distance_paved : ℝ := speed_paved * time_paved -- Distance on paved road
  let total_distance : ℝ := distance_dirt + distance_paved -- Total distance of the trip
  total_distance = 200 := by sorry

end truck_journey_distance_l2167_216729


namespace initial_money_calculation_l2167_216754

/-- The amount of money Rachel and Sarah had when they left home -/
def initial_money : ℝ := 50

/-- The amount spent on gasoline -/
def gasoline_cost : ℝ := 8

/-- The amount spent on lunch -/
def lunch_cost : ℝ := 15.65

/-- The amount spent on gifts for grandma (per person) -/
def gift_cost : ℝ := 5

/-- The amount received from grandma (per person) -/
def grandma_gift : ℝ := 10

/-- The amount of money they have for the return trip -/
def return_trip_money : ℝ := 36.35

/-- The number of people (Rachel and Sarah) -/
def num_people : ℕ := 2

theorem initial_money_calculation :
  initial_money = 
    return_trip_money + 
    (gasoline_cost + lunch_cost + num_people * gift_cost) - 
    (num_people * grandma_gift) := by
  sorry

end initial_money_calculation_l2167_216754


namespace intersection_perimeter_constant_l2167_216728

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A plane parallel to two opposite edges of a regular tetrahedron -/
structure ParallelPlane (t : RegularTetrahedron) where
  -- We don't need to define the plane explicitly, just its existence

/-- The figure obtained from intersecting a regular tetrahedron with a parallel plane -/
def IntersectionFigure (t : RegularTetrahedron) (p : ParallelPlane t) : Type :=
  -- We don't need to define the figure explicitly, just its existence
  Unit

/-- The perimeter of an intersection figure -/
noncomputable def perimeter (t : RegularTetrahedron) (p : ParallelPlane t) (f : IntersectionFigure t p) : ℝ :=
  2 * t.edge_length

/-- Theorem: The perimeter of any intersection figure is equal to 2a -/
theorem intersection_perimeter_constant (t : RegularTetrahedron) 
  (p : ParallelPlane t) (f : IntersectionFigure t p) :
  perimeter t p f = 2 * t.edge_length :=
by
  sorry

end intersection_perimeter_constant_l2167_216728


namespace maria_green_towels_l2167_216785

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

end maria_green_towels_l2167_216785


namespace car_sales_second_day_l2167_216778

theorem car_sales_second_day 
  (total_sales : ℕ) 
  (first_day_sales : ℕ) 
  (third_day_sales : ℕ) 
  (h1 : total_sales = 57)
  (h2 : first_day_sales = 14)
  (h3 : third_day_sales = 27) :
  total_sales - first_day_sales - third_day_sales = 16 := by
  sorry

end car_sales_second_day_l2167_216778


namespace beatrix_books_l2167_216790

theorem beatrix_books (beatrix alannah queen : ℕ) 
  (h1 : alannah = beatrix + 20)
  (h2 : queen = alannah + alannah / 5)
  (h3 : beatrix + alannah + queen = 140) : 
  beatrix = 30 := by
  sorry

end beatrix_books_l2167_216790


namespace positive_real_inequality_l2167_216759

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 
  (x^2011 + y^2011) / (x^2009 + y^2009) + 
  (y^2011 + z^2011) / (y^2009 + z^2009) + 
  (z^2011 + x^2011) / (z^2009 + x^2009) ≥ 1/3 := by
  sorry

end positive_real_inequality_l2167_216759


namespace prob_sum_less_than_15_l2167_216769

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

end prob_sum_less_than_15_l2167_216769


namespace clock_chimes_theorem_l2167_216745

/-- Represents the number of chimes at a given hour -/
def chimes_at_hour (hour : ℕ) : ℕ := hour

/-- Represents the time taken for a given number of chimes -/
def time_for_chimes (chimes : ℕ) : ℕ :=
  if chimes ≤ 1 then chimes else chimes - 1 + 1

/-- The theorem statement -/
theorem clock_chimes_theorem (hour : ℕ) (chimes : ℕ) (time : ℕ) 
  (h1 : hour = 2 → chimes = 2)
  (h2 : hour = 2 → time = 2)
  (h3 : hour = 12 → chimes = 12) :
  hour = 12 → time_for_chimes chimes = 12 := by
  sorry

#check clock_chimes_theorem

end clock_chimes_theorem_l2167_216745


namespace auction_starting_value_l2167_216768

/-- The starting value of an auction satisfying the given conditions -/
def auctionStartingValue : ℝ → Prop := fun S =>
  let harryFirstBid := S + 200
  let secondBidderBid := 2 * harryFirstBid
  let thirdBidderBid := secondBidderBid + 3 * harryFirstBid
  let harryFinalBid := 4000
  harryFinalBid = thirdBidderBid + 1500

theorem auction_starting_value : ∃ S, auctionStartingValue S ∧ S = 300 := by
  sorry

end auction_starting_value_l2167_216768


namespace complement_intersection_real_l2167_216711

open Set

theorem complement_intersection_real (A B : Set ℝ) 
  (hA : A = {x : ℝ | 3 ≤ x ∧ x < 7})
  (hB : B = {x : ℝ | 2 < x ∧ x < 10}) :
  (A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ 7 ≤ x} := by sorry

end complement_intersection_real_l2167_216711
