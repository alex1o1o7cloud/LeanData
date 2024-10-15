import Mathlib

namespace NUMINAMATH_CALUDE_expected_ones_is_one_third_l1048_104815

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The expected number of 1's when rolling two standard dice -/
def expected_ones : ℚ := 2 * (prob_one * prob_one) + 1 * (2 * prob_one * prob_not_one)

theorem expected_ones_is_one_third : expected_ones = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_one_third_l1048_104815


namespace NUMINAMATH_CALUDE_max_sum_xy_l1048_104821

def max_value_xy (x y : ℝ) : Prop :=
  (Real.log y / Real.log ((x^2 + y^2) / 2) ≥ 1) ∧
  ((x, y) ≠ (0, 0)) ∧
  (x^2 + y^2 ≠ 2)

theorem max_sum_xy :
  ∀ x y : ℝ, max_value_xy x y → x + y ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xy_l1048_104821


namespace NUMINAMATH_CALUDE_percentage_of_120_to_80_l1048_104869

theorem percentage_of_120_to_80 : 
  (120 : ℝ) / 80 * 100 = 150 := by sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_80_l1048_104869


namespace NUMINAMATH_CALUDE_second_group_size_l1048_104823

theorem second_group_size (sum_first : ℕ) (count_first : ℕ) (avg_second : ℚ) (avg_total : ℚ) 
  (h1 : sum_first = 84)
  (h2 : count_first = 7)
  (h3 : avg_second = 21)
  (h4 : avg_total = 18) :
  ∃ (count_second : ℕ), 
    (sum_first + count_second * avg_second) / (count_first + count_second) = avg_total ∧ 
    count_second = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l1048_104823


namespace NUMINAMATH_CALUDE_carrie_fourth_day_miles_l1048_104808

/-- Represents Carrie's four-day trip --/
structure CarrieTrip where
  day1_miles : ℕ
  day2_miles : ℕ
  day3_miles : ℕ
  day4_miles : ℕ
  charge_interval : ℕ
  total_charges : ℕ

/-- Theorem: Given the conditions of Carrie's trip, she drove 189 miles on the fourth day --/
theorem carrie_fourth_day_miles (trip : CarrieTrip)
  (h1 : trip.day1_miles = 135)
  (h2 : trip.day2_miles = trip.day1_miles + 124)
  (h3 : trip.day3_miles = 159)
  (h4 : trip.charge_interval = 106)
  (h5 : trip.total_charges = 7)
  : trip.day4_miles = 189 := by
  sorry

#check carrie_fourth_day_miles

end NUMINAMATH_CALUDE_carrie_fourth_day_miles_l1048_104808


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_bound_gcd_lcm_sum_equality_condition_l1048_104870

theorem gcd_lcm_sum_bound (a b : ℕ) (h1 : a * b > 2) :
  let d := Nat.gcd a b
  let l := Nat.lcm a b
  (∃ k : ℕ, d + l = k * (a + b)) →
  (d + l) / (a + b) ≤ (a + b) / 4 := by
sorry

theorem gcd_lcm_sum_equality_condition (a b : ℕ) (h1 : a * b > 2) :
  let d := Nat.gcd a b
  let l := Nat.lcm a b
  (∃ k : ℕ, d + l = k * (a + b)) →
  ((d + l) / (a + b) = (a + b) / 4 ↔
   ∃ (x y : ℕ), a = d * x ∧ b = d * y ∧ x = y + 2) := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_bound_gcd_lcm_sum_equality_condition_l1048_104870


namespace NUMINAMATH_CALUDE_range_of_a_l1048_104803

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*x + 3*a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1048_104803


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1048_104896

theorem multiplication_puzzle : ∃ (a b : Nat), 
  a < 10000 ∧ 
  b < 1000 ∧ 
  a / 1000 = 3 ∧ 
  a % 100 = 20 ∧
  b / 100 = 3 ∧
  (a * (b % 10)) % 10000 = 9060 ∧
  ((a * (b / 10)) / 10000) * 10000 + ((a * (b / 10)) % 10000) = 62510 ∧
  a * b = 1157940830 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1048_104896


namespace NUMINAMATH_CALUDE_optimal_split_positions_l1048_104807

/-- The number N as defined in the problem -/
def N : ℕ := 10^1001 - 1

/-- Function to calculate the sum when splitting at position m -/
def S (m : ℕ) : ℕ := 2 * 10^m + 10^(1992 - m) - 10

/-- Function to calculate the product when splitting at position m -/
def P (m : ℕ) : ℕ := 2 * 10^1992 + 9 - 18 * 10^m - 10^(1992 - m)

/-- Theorem stating the optimal split positions for sum and product -/
theorem optimal_split_positions :
  (∀ m, m ≠ 996 → S 996 ≤ S m) ∧
  (∀ m, m ≠ 995 → P 995 ≥ P m) :=
sorry


end NUMINAMATH_CALUDE_optimal_split_positions_l1048_104807


namespace NUMINAMATH_CALUDE_coin_array_digit_sum_l1048_104841

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The total number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem coin_array_digit_sum :
  ∃ (N : ℕ), triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_digit_sum_l1048_104841


namespace NUMINAMATH_CALUDE_equation_solution_l1048_104839

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4} = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1048_104839


namespace NUMINAMATH_CALUDE_quadratic_roots_count_l1048_104895

/-- The number of real roots of the quadratic function y = x^2 + x - 1 is 2 -/
theorem quadratic_roots_count : 
  let f : ℝ → ℝ := fun x ↦ x^2 + x - 1
  (∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0) ∧ 
  (∀ (x y z : ℝ), f x = 0 → f y = 0 → f z = 0 → x = y ∨ x = z ∨ y = z) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_count_l1048_104895


namespace NUMINAMATH_CALUDE_wax_sculpture_theorem_l1048_104813

/-- Proves that the total number of wax sticks used is 20 --/
theorem wax_sculpture_theorem (large_sticks small_sticks : ℕ) 
  (h1 : large_sticks = 4)
  (h2 : small_sticks = 2)
  (small_animals large_animals : ℕ)
  (h3 : small_animals = 3 * large_animals)
  (total_small_sticks : ℕ)
  (h4 : total_small_sticks = 12)
  (h5 : total_small_sticks = small_animals * small_sticks) :
  total_small_sticks + large_animals * large_sticks = 20 := by
sorry

end NUMINAMATH_CALUDE_wax_sculpture_theorem_l1048_104813


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1048_104818

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 4, 5}
def N : Finset Nat := {1, 3}

theorem intersection_with_complement : M ∩ (U \ N) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1048_104818


namespace NUMINAMATH_CALUDE_ratio_comparison_l1048_104810

theorem ratio_comparison (a : ℚ) (h : a > 3) : (3 : ℚ) / 4 < a / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_l1048_104810


namespace NUMINAMATH_CALUDE_n_plus_one_in_terms_of_m_l1048_104809

theorem n_plus_one_in_terms_of_m (m n : ℕ) 
  (h1 : m * n = 121) 
  (h2 : (m + 1) * (n + 1) = 1000) : 
  n + 1 = 879 - m := by
sorry

end NUMINAMATH_CALUDE_n_plus_one_in_terms_of_m_l1048_104809


namespace NUMINAMATH_CALUDE_concatenated_digits_2015_l1048_104817

/-- The number of digits in a positive integer n -/
def num_digits (n : ℕ+) : ℕ := sorry

/-- The sum of digits for all numbers from 1 to n -/
def sum_digits (n : ℕ) : ℕ := sorry

theorem concatenated_digits_2015 : sum_digits 2015 = 6953 := by sorry

end NUMINAMATH_CALUDE_concatenated_digits_2015_l1048_104817


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1048_104871

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1048_104871


namespace NUMINAMATH_CALUDE_compute_F_3_f_5_l1048_104851

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem compute_F_3_f_5 : F 3 (f 5) = 24 := by sorry

end NUMINAMATH_CALUDE_compute_F_3_f_5_l1048_104851


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l1048_104834

/-- Given two lines in a plane, this function returns the line that is symmetric to the first line with respect to the second line. -/
def symmetricLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line y = 2x + 1 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y ↦ y = 2 * x + 1

/-- The line y = x - 2 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y ↦ y = x - 2

/-- The line x - 2y - 7 = 0 -/
def lineL : ℝ → ℝ → Prop :=
  fun x y ↦ x - 2 * y - 7 = 0

theorem symmetric_line_equation :
  symmetricLine line1 line2 = lineL :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l1048_104834


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1048_104822

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℝ)
  (group1 : ℕ)
  (avg1 : ℝ)
  (group2 : ℕ)
  (avg2 : ℝ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_group1 : group1 = 2)
  (h_avg1 : avg1 = 3.4)
  (h_group2 : group2 = 2)
  (h_avg2 : avg2 = 3.85) :
  let remaining := total - group1 - group2
  let sum_all := total * avg_all
  let sum1 := group1 * avg1
  let sum2 := group2 * avg2
  let sum_remaining := sum_all - sum1 - sum2
  sum_remaining / remaining = 4.6 := by sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1048_104822


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1048_104888

/-- Distance between foci of an ellipse -/
theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1048_104888


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l1048_104824

theorem unique_solution_linear_system
  (a b c d : ℝ)
  (h : a * d - c * b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y = 0 ∧ c * x + d * y = 0 → x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l1048_104824


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1048_104886

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1048_104886


namespace NUMINAMATH_CALUDE_problem_solution_l1048_104833

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1048_104833


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l1048_104832

/-- A square pyramid is a polyhedron with a square base and triangular lateral faces -/
structure SquarePyramid where
  base : Nat
  lateral_faces : Nat
  base_edges : Nat
  lateral_edges : Nat
  base_vertices : Nat
  apex : Nat

/-- Properties of a square pyramid -/
def square_pyramid : SquarePyramid :=
  { base := 1
  , lateral_faces := 4
  , base_edges := 4
  , lateral_edges := 4
  , base_vertices := 4
  , apex := 1 }

/-- The sum of faces, edges, and vertices of a square pyramid is 18 -/
theorem square_pyramid_sum :
  (square_pyramid.base + square_pyramid.lateral_faces) +
  (square_pyramid.base_edges + square_pyramid.lateral_edges) +
  (square_pyramid.base_vertices + square_pyramid.apex) = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l1048_104832


namespace NUMINAMATH_CALUDE_trip_cost_calculation_l1048_104876

def initial_odometer : ℕ := 85300
def final_odometer : ℕ := 85335
def fuel_efficiency : ℚ := 25
def gas_price : ℚ := 21/5  -- $4.20 represented as a rational number

def trip_cost : ℚ :=
  (final_odometer - initial_odometer : ℚ) / fuel_efficiency * gas_price

theorem trip_cost_calculation :
  trip_cost = 588/100 := by sorry

end NUMINAMATH_CALUDE_trip_cost_calculation_l1048_104876


namespace NUMINAMATH_CALUDE_four_integer_pairs_satisfying_equation_l1048_104859

theorem four_integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 1) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_integer_pairs_satisfying_equation_l1048_104859


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1048_104860

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 17 + (2 * x) + 15 + (2 * x + 6) + (3 * x - 5)) / 6 = 30 → x = 137 / 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1048_104860


namespace NUMINAMATH_CALUDE_multiply_826446281_by_11_twice_l1048_104845

theorem multiply_826446281_by_11_twice :
  826446281 * 11 * 11 = 100000000001 := by
  sorry

end NUMINAMATH_CALUDE_multiply_826446281_by_11_twice_l1048_104845


namespace NUMINAMATH_CALUDE_linear_inequality_m_value_l1048_104842

theorem linear_inequality_m_value (m : ℝ) : 
  (∀ x, ∃ a b, (m - 2) * x^(|m - 1|) - 3 > 6 ↔ a * x + b > 0) → 
  (m - 2 ≠ 0) → 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_m_value_l1048_104842


namespace NUMINAMATH_CALUDE_tangency_point_satisfies_equations_unique_tangency_point_l1048_104844

/-- The point of tangency for two parabolas -/
def point_of_tangency : ℝ × ℝ := (-7, -24)

/-- The first parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32

/-- The second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  parabola1 point_of_tangency.1 point_of_tangency.2 ∧
  parabola2 point_of_tangency.1 point_of_tangency.2 := by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem unique_tangency_point :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency := by sorry

end NUMINAMATH_CALUDE_tangency_point_satisfies_equations_unique_tangency_point_l1048_104844


namespace NUMINAMATH_CALUDE_no_common_points_l1048_104816

theorem no_common_points : 
  ¬∃ (x y : ℝ), (x^2 + y^2 = 4) ∧ (x^2 + 2*y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_common_points_l1048_104816


namespace NUMINAMATH_CALUDE_probability_three_one_color_l1048_104805

/-- The probability of drawing 3 balls of one color and 1 of another color
    from a set of 20 balls (12 black, 8 white) when 4 are drawn at random -/
theorem probability_three_one_color (black_balls white_balls total_balls drawn : ℕ) 
  (h1 : black_balls = 12)
  (h2 : white_balls = 8)
  (h3 : total_balls = black_balls + white_balls)
  (h4 : drawn = 4) :
  (Nat.choose black_balls 3 * Nat.choose white_balls 1 +
   Nat.choose black_balls 1 * Nat.choose white_balls 3) / 
  Nat.choose total_balls drawn = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_three_one_color_l1048_104805


namespace NUMINAMATH_CALUDE_chess_game_probability_l1048_104862

/-- The probability of player A winning a chess game -/
def prob_A_win : ℝ := 0.3

/-- The probability of the chess game ending in a draw -/
def prob_draw : ℝ := 0.5

/-- The probability of player B not losing the chess game -/
def prob_B_not_lose : ℝ := 1 - prob_A_win

theorem chess_game_probability : prob_B_not_lose = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l1048_104862


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l1048_104899

theorem gcd_of_powers_of_101 (h : Prime 101) :
  Nat.gcd (101^5 + 1) (101^5 + 101^3 + 101 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l1048_104899


namespace NUMINAMATH_CALUDE_new_energy_vehicles_analysis_l1048_104847

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 149.24 * x - 33.64

-- Define the stock data for years 2017 to 2021
def stock_data : List (ℕ × ℝ) := [
  (1, 153.4),
  (2, 260.8),
  (3, 380.2),
  (4, 492.0),
  (5, 784.0)
]

-- Theorem statement
theorem new_energy_vehicles_analysis :
  -- 1. Predicted stock for 2023 exceeds 1000 million vehicles
  (regression_eq 7 > 1000) ∧
  -- 2. Stock shows increasing trend from 2017 to 2021
  (∀ i j, i < j → i ∈ List.map Prod.fst stock_data → j ∈ List.map Prod.fst stock_data →
    (stock_data.find? (λ p => p.fst = i)).map Prod.snd < (stock_data.find? (λ p => p.fst = j)).map Prod.snd) ∧
  -- 3. Residual for 2021 is 71.44
  (((stock_data.find? (λ p => p.fst = 5)).map Prod.snd).getD 0 - regression_eq 5 = 71.44) := by
  sorry

end NUMINAMATH_CALUDE_new_energy_vehicles_analysis_l1048_104847


namespace NUMINAMATH_CALUDE_alice_original_seat_l1048_104890

/-- Represents the possible seats in the lecture hall -/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six
  | seven

/-- Represents the movement of a person -/
inductive Movement
  | left : Nat → Movement
  | right : Nat → Movement
  | stay : Movement
  | switch : Movement

/-- Represents a person and their movement -/
structure Person where
  name : String
  movement : Movement

/-- The state of the seating arrangement -/
structure SeatingArrangement where
  seats : Vector Person 7
  aliceOriginalSeat : Seat
  aliceFinalSeat : Seat

def isEndSeat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.seven

/-- The theorem to prove -/
theorem alice_original_seat
  (arrangement : SeatingArrangement)
  (beth_moves_right : arrangement.seats[1].movement = Movement.right 1)
  (carla_moves_left : arrangement.seats[2].movement = Movement.left 2)
  (dana_elly_switch : arrangement.seats[3].movement = Movement.switch ∧
                      arrangement.seats[4].movement = Movement.switch)
  (fiona_moves_left : arrangement.seats[5].movement = Movement.left 1)
  (grace_stays : arrangement.seats[6].movement = Movement.stay)
  (alice_ends_in_end_seat : isEndSeat arrangement.aliceFinalSeat) :
  arrangement.aliceOriginalSeat = Seat.five := by
  sorry

end NUMINAMATH_CALUDE_alice_original_seat_l1048_104890


namespace NUMINAMATH_CALUDE_pen_price_calculation_l1048_104820

/-- Given the purchase of pens and pencils with known quantities and prices,
    prove that the average price of a pen is $14.00. -/
theorem pen_price_calculation (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) 
    (pencil_price : ℚ) (pen_price : ℚ) : 
    num_pens = 30 → 
    num_pencils = 75 → 
    total_cost = 570 → 
    pencil_price = 2 → 
    pen_price = (total_cost - num_pencils * pencil_price) / num_pens → 
    pen_price = 14 := by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l1048_104820


namespace NUMINAMATH_CALUDE_age_double_time_l1048_104885

/-- Given Julio's current age is 42 and James' current age is 8,
    this theorem proves that it will take 26 years for Julio's age to be twice James' age. -/
theorem age_double_time (julio_age : ℕ) (james_age : ℕ) (h1 : julio_age = 42) (h2 : james_age = 8) :
  ∃ (years : ℕ), julio_age + years = 2 * (james_age + years) ∧ years = 26 := by
  sorry

end NUMINAMATH_CALUDE_age_double_time_l1048_104885


namespace NUMINAMATH_CALUDE_fraction_transformation_l1048_104835

theorem fraction_transformation (x : ℤ) : 
  x = 437 → (537 - x : ℚ) / (463 + x) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1048_104835


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1048_104881

theorem smallest_common_multiple_of_8_and_6 :
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 ∧ 8 ∣ m ∧ 6 ∣ m → n ≤ m :=
by
  use 24
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1048_104881


namespace NUMINAMATH_CALUDE_lowest_energy_point_min_energy_at_two_l1048_104852

/-- Represents the energy function for an athlete during a 4-hour training session. -/
noncomputable def Q (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 1 then
    10000 - 3600 * t
  else if 1 < t ∧ t ≤ 4 then
    400 + 1200 * t + 4800 / t
  else
    0

/-- Theorem stating that the athlete's energy reaches its lowest point at t = 2 hours with a value of 5200kJ. -/
theorem lowest_energy_point :
  ∀ t : ℝ, 0 < t ∧ t ≤ 4 → Q t ≥ 5200 ∧ Q 2 = 5200 := by sorry

/-- Corollary stating that the minimum energy occurs at t = 2. -/
theorem min_energy_at_two :
  ∀ t : ℝ, 0 < t ∧ t ≤ 4 → Q 2 ≤ Q t := by sorry

end NUMINAMATH_CALUDE_lowest_energy_point_min_energy_at_two_l1048_104852


namespace NUMINAMATH_CALUDE_third_job_hourly_rate_l1048_104879

-- Define the problem parameters
def total_earnings : ℝ := 430
def first_job_hours : ℝ := 15
def first_job_rate : ℝ := 8
def second_job_sales : ℝ := 1000
def second_job_commission_rate : ℝ := 0.1
def third_job_hours : ℝ := 12
def tax_deduction : ℝ := 50

-- Define the theorem
theorem third_job_hourly_rate :
  let first_job_earnings := first_job_hours * first_job_rate
  let second_job_earnings := second_job_sales * second_job_commission_rate
  let combined_wages := first_job_earnings + second_job_earnings
  let combined_wages_after_tax := combined_wages - tax_deduction
  let third_job_earnings := total_earnings - combined_wages_after_tax
  third_job_earnings / third_job_hours = 21.67 := by
  sorry

end NUMINAMATH_CALUDE_third_job_hourly_rate_l1048_104879


namespace NUMINAMATH_CALUDE_inequality_theorem_l1048_104829

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hy₁ : x₁ * y₁ - z₁^2 > 0) (hy₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1048_104829


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1048_104800

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1048_104800


namespace NUMINAMATH_CALUDE_division_problem_l1048_104889

theorem division_problem (n : ℕ) : 
  n % 23 = 19 ∧ n / 23 = 17 → (10 * n) / 23 + (10 * n) % 23 = 184 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1048_104889


namespace NUMINAMATH_CALUDE_value_range_of_f_l1048_104874

def f (x : ℝ) := 3 * x - 1

theorem value_range_of_f :
  Set.Icc (-16 : ℝ) 5 = Set.image f (Set.Ico (-5 : ℝ) 2) := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l1048_104874


namespace NUMINAMATH_CALUDE_warehouse_capacity_prove_warehouse_capacity_l1048_104858

/-- The total capacity of a grain storage warehouse --/
theorem warehouse_capacity : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_bins twenty_ton_bins twenty_ton_capacity fifteen_ton_capacity =>
    total_bins = 30 ∧
    twenty_ton_bins = 12 ∧
    twenty_ton_capacity = 20 ∧
    fifteen_ton_capacity = 15 →
    (twenty_ton_bins * twenty_ton_capacity) +
    ((total_bins - twenty_ton_bins) * fifteen_ton_capacity) = 510

/-- Proof of the warehouse capacity theorem --/
theorem prove_warehouse_capacity :
  warehouse_capacity 30 12 20 15 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_capacity_prove_warehouse_capacity_l1048_104858


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1048_104866

/-- A point in the second quadrant with given distances to the axes has specific coordinates -/
theorem point_in_second_quadrant (P : ℝ × ℝ) : 
  P.1 < 0 ∧ P.2 > 0 ∧  -- P is in the second quadrant
  |P.2| = 5 ∧          -- distance to x-axis is 5
  |P.1| = 3            -- distance to y-axis is 3
  → P = (-3, 5) := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1048_104866


namespace NUMINAMATH_CALUDE_point_N_coordinates_l1048_104804

def M : ℝ × ℝ := (3, -4)
def a : ℝ × ℝ := (1, -2)

theorem point_N_coordinates : 
  ∀ N : ℝ × ℝ, 
  (N.1 - M.1, N.2 - M.2) = (-2 * a.1, -2 * a.2) → 
  N = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l1048_104804


namespace NUMINAMATH_CALUDE_runner_solution_l1048_104843

def runner_problem (t : ℕ) : Prop :=
  let first_runner := 2
  let second_runner := 4
  let third_runner := t
  let meeting_time := 44
  (meeting_time % first_runner = 0) ∧
  (meeting_time % second_runner = 0) ∧
  (meeting_time % third_runner = 0) ∧
  (first_runner < third_runner) ∧
  (second_runner < third_runner) ∧
  (∀ t' < meeting_time, t' % first_runner = 0 → t' % second_runner = 0 → t' % third_runner ≠ 0)

theorem runner_solution : runner_problem 11 := by
  sorry

end NUMINAMATH_CALUDE_runner_solution_l1048_104843


namespace NUMINAMATH_CALUDE_copenhagen_aarhus_distance_l1048_104846

/-- The distance between two city centers with a detour -/
def distance_with_detour (map_distance : ℝ) (scale : ℝ) (detour_increase : ℝ) : ℝ :=
  map_distance * scale * (1 + detour_increase)

/-- Theorem: The distance between Copenhagen and Aarhus is 420 km -/
theorem copenhagen_aarhus_distance :
  distance_with_detour 35 10 0.2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_copenhagen_aarhus_distance_l1048_104846


namespace NUMINAMATH_CALUDE_eva_total_marks_2019_l1048_104831

/-- Eva's marks in different subjects and semesters -/
structure EvaMarks where
  maths_second : ℕ
  arts_second : ℕ
  science_second : ℕ
  maths_first : ℕ
  arts_first : ℕ
  science_first : ℕ

/-- Calculate the total marks for Eva in 2019 -/
def total_marks (marks : EvaMarks) : ℕ :=
  marks.maths_first + marks.arts_first + marks.science_first +
  marks.maths_second + marks.arts_second + marks.science_second

/-- Theorem stating Eva's total marks in 2019 -/
theorem eva_total_marks_2019 (marks : EvaMarks)
  (h1 : marks.maths_second = 80)
  (h2 : marks.arts_second = 90)
  (h3 : marks.science_second = 90)
  (h4 : marks.maths_first = marks.maths_second + 10)
  (h5 : marks.arts_first = marks.arts_second - 15)
  (h6 : marks.science_first = marks.science_second - marks.science_second / 3) :
  total_marks marks = 485 := by
  sorry


end NUMINAMATH_CALUDE_eva_total_marks_2019_l1048_104831


namespace NUMINAMATH_CALUDE_product_inequality_l1048_104887

theorem product_inequality (a b c d : ℝ) 
  (sum_zero : a + b + c = 0)
  (d_def : d = max (abs a) (max (abs b) (abs c))) :
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l1048_104887


namespace NUMINAMATH_CALUDE_equality_proof_l1048_104892

theorem equality_proof (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l1048_104892


namespace NUMINAMATH_CALUDE_squirrel_mushroom_collection_l1048_104849

/-- Represents the number of mushrooms in each clearing --/
def MushroomSequence : Type := List Nat

/-- The total number of mushrooms collected by the squirrel --/
def TotalMushrooms : Nat := 60

/-- The number of clearings visited by the squirrel --/
def NumberOfClearings : Nat := 10

/-- Checks if a given sequence is valid according to the problem conditions --/
def IsValidSequence (seq : MushroomSequence) : Prop :=
  seq.length = NumberOfClearings ∧
  seq.sum = TotalMushrooms ∧
  seq.all (· > 0)

/-- The correct sequence of mushrooms collected in each clearing --/
def CorrectSequence : MushroomSequence := [5, 2, 11, 8, 2, 12, 3, 7, 2, 8]

/-- Theorem stating that the CorrectSequence is a valid solution to the problem --/
theorem squirrel_mushroom_collection :
  IsValidSequence CorrectSequence :=
sorry

end NUMINAMATH_CALUDE_squirrel_mushroom_collection_l1048_104849


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l1048_104897

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(abs (x - 5) + abs (x + 3) < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l1048_104897


namespace NUMINAMATH_CALUDE_soccer_leagues_games_l1048_104838

/-- Calculate the number of games in a round-robin tournament -/
def gamesInLeague (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of games played across three leagues -/
def totalGames (a b c : ℕ) : ℕ := gamesInLeague a + gamesInLeague b + gamesInLeague c

theorem soccer_leagues_games :
  totalGames 20 25 30 = 925 := by
  sorry

end NUMINAMATH_CALUDE_soccer_leagues_games_l1048_104838


namespace NUMINAMATH_CALUDE_not_perfect_power_of_ten_sixes_and_zeros_l1048_104865

def is_composed_of_ten_sixes_and_zeros (n : ℕ) : Prop :=
  ∃ k, n = 6666666666 * 10^k

theorem not_perfect_power_of_ten_sixes_and_zeros (n : ℕ) 
  (h : is_composed_of_ten_sixes_and_zeros n) : 
  ¬ ∃ (a b : ℕ), b > 1 ∧ n = a^b :=
sorry

end NUMINAMATH_CALUDE_not_perfect_power_of_ten_sixes_and_zeros_l1048_104865


namespace NUMINAMATH_CALUDE_money_puzzle_l1048_104827

theorem money_puzzle (x : ℝ) : x = 800 ↔ 4 * x - 2000 = 2000 - x := by sorry

end NUMINAMATH_CALUDE_money_puzzle_l1048_104827


namespace NUMINAMATH_CALUDE_cooking_time_is_five_l1048_104811

def recommended_cooking_time (cooked_time seconds_remaining : ℕ) : ℚ :=
  (cooked_time + seconds_remaining) / 60

theorem cooking_time_is_five :
  recommended_cooking_time 45 255 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_is_five_l1048_104811


namespace NUMINAMATH_CALUDE_triangle_side_length_l1048_104848

theorem triangle_side_length (A B C : Real) (angleB : Real) (sideAB sideAC : Real) :
  angleB = π / 4 →
  sideAB = 100 →
  sideAC = 100 →
  (∃! bc : Real, bc = sideAB * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1048_104848


namespace NUMINAMATH_CALUDE_problem_solution_l1048_104837

noncomputable def g (θ : Real) (x : Real) : Real := x * Real.sin θ - Real.log x - Real.sin θ

noncomputable def f (θ : Real) (x : Real) : Real := g θ x + (2*x - 1) / (2*x^2)

theorem problem_solution (θ : Real) (h1 : θ ∈ Set.Ioo 0 Real.pi) 
  (h2 : ∀ x ≥ 1, Monotone (g θ)) : 
  (θ = Real.pi/2) ∧ 
  (∀ x ∈ Set.Icc 1 2, f θ x > (deriv (f θ)) x + 1/2) ∧
  (∀ k > 1, ∃ x > 0, Real.exp x - x - 1 < k * g θ (x+1)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1048_104837


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1048_104825

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + 1
  let r : ℕ := 3^s - s + 2
  r = 19676 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1048_104825


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1048_104857

/-- Given a total number of marbles, with blue marbles being three times
    the number of red marbles, and a specific number of red marbles,
    prove the number of yellow marbles. -/
theorem yellow_marbles_count
  (total : ℕ)
  (red : ℕ)
  (h1 : total = 85)
  (h2 : red = 14) :
  total - (red + 3 * red) = 29 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1048_104857


namespace NUMINAMATH_CALUDE_johns_father_age_multiple_l1048_104894

/-- 
Given John's age, the sum of John and his father's ages, and the relationship between
John's father's age and John's age, this theorem proves the multiple of John's age
that represents his father's age without the additional 32 years.
-/
theorem johns_father_age_multiple 
  (john_age : ℕ)
  (sum_ages : ℕ)
  (father_age_relation : ℕ → ℕ)
  (h1 : john_age = 15)
  (h2 : sum_ages = 77)
  (h3 : father_age_relation m = m * john_age + 32)
  (h4 : sum_ages = john_age + father_age_relation m) :
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_father_age_multiple_l1048_104894


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l1048_104826

theorem insurance_coverage_percentage 
  (total_cost : ℝ) 
  (out_of_pocket : ℝ) 
  (h1 : total_cost = 500) 
  (h2 : out_of_pocket = 200) : 
  (total_cost - out_of_pocket) / total_cost * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l1048_104826


namespace NUMINAMATH_CALUDE_A_subset_B_A_eq_B_when_single_element_l1048_104819

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | f a b x = x}
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

-- Theorem 1: A ⊆ B
theorem A_subset_B (a b : ℝ) : A a b ⊆ B a b := by sorry

-- Theorem 2: If A has only one element, then A = B
theorem A_eq_B_when_single_element (a b : ℝ) :
  (∃! x, x ∈ A a b) → A a b = B a b := by sorry

end NUMINAMATH_CALUDE_A_subset_B_A_eq_B_when_single_element_l1048_104819


namespace NUMINAMATH_CALUDE_cuboid_sum_of_edges_l1048_104891

/-- Represents the dimensions of a rectangular cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the dimensions form a geometric progression -/
def isGeometricProgression (d : CuboidDimensions) : Prop :=
  ∃ q : ℝ, q > 0 ∧ d.length = q * d.width ∧ d.width = q * d.height

/-- Calculates the volume of a rectangular cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the surface area of a rectangular cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.width * d.height + d.height * d.length)

/-- Calculates the sum of all edges of a rectangular cuboid -/
def sumOfEdges (d : CuboidDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

/-- Theorem: For a rectangular cuboid with volume 8, surface area 32, and dimensions 
    forming a geometric progression, the sum of all edges is 32 -/
theorem cuboid_sum_of_edges : 
  ∀ d : CuboidDimensions, 
    volume d = 8 → 
    surfaceArea d = 32 → 
    isGeometricProgression d → 
    sumOfEdges d = 32 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_sum_of_edges_l1048_104891


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_in_pascal_l1048_104898

/-- Represents a row in Pascal's triangle -/
def PascalRow := List Nat

/-- Generates the next row of Pascal's triangle given the current row -/
def nextPascalRow (row : PascalRow) : PascalRow :=
  sorry

/-- Checks if a number is a four-digit number -/
def isFourDigit (n : Nat) : Bool :=
  1000 ≤ n ∧ n ≤ 9999

/-- Finds the nth four-digit number in Pascal's triangle -/
def nthFourDigitInPascal (n : Nat) : Nat :=
  sorry

theorem third_smallest_four_digit_in_pascal :
  nthFourDigitInPascal 3 = 1002 :=
sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_in_pascal_l1048_104898


namespace NUMINAMATH_CALUDE_lunch_combo_count_l1048_104867

/-- Represents the number of options for each food category --/
structure FoodOptions where
  lettuce : Nat
  tomatoes : Nat
  olives : Nat
  bread : Nat
  fruit : Nat
  soup : Nat

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- Calculates the total number of lunch combo options --/
def lunchComboOptions (options : FoodOptions) : Nat :=
  let remainingItems := options.olives + options.bread + options.fruit
  let remainingChoices := choose remainingItems 3
  options.lettuce * options.tomatoes * remainingChoices * options.soup

/-- Theorem stating the number of lunch combo options --/
theorem lunch_combo_count (options : FoodOptions) 
  (h1 : options.lettuce = 4)
  (h2 : options.tomatoes = 5)
  (h3 : options.olives = 6)
  (h4 : options.bread = 3)
  (h5 : options.fruit = 4)
  (h6 : options.soup = 3) :
  lunchComboOptions options = 17160 := by
  sorry

#eval lunchComboOptions { lettuce := 4, tomatoes := 5, olives := 6, bread := 3, fruit := 4, soup := 3 }

end NUMINAMATH_CALUDE_lunch_combo_count_l1048_104867


namespace NUMINAMATH_CALUDE_x_value_l1048_104880

theorem x_value (x y z : ℝ) (h1 : x = y) (h2 : x = 2*z) (h3 : x*y*z = 256) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1048_104880


namespace NUMINAMATH_CALUDE_hiking_problem_l1048_104840

/-- Proves that the number of people in each van is 5, given the conditions of the hiking problem --/
theorem hiking_problem (num_cars num_taxis num_vans : ℕ) 
                       (people_per_car people_per_taxi total_people : ℕ) 
                       (h1 : num_cars = 3)
                       (h2 : num_taxis = 6)
                       (h3 : num_vans = 2)
                       (h4 : people_per_car = 4)
                       (h5 : people_per_taxi = 6)
                       (h6 : total_people = 58)
                       (h7 : total_people = num_cars * people_per_car + 
                                            num_taxis * people_per_taxi + 
                                            num_vans * (total_people - num_cars * people_per_car - num_taxis * people_per_taxi) / num_vans) : 
  (total_people - num_cars * people_per_car - num_taxis * people_per_taxi) / num_vans = 5 := by
  sorry

end NUMINAMATH_CALUDE_hiking_problem_l1048_104840


namespace NUMINAMATH_CALUDE_merchant_markup_problem_l1048_104863

theorem merchant_markup_problem (markup_percentage : ℝ) : 
  (∀ cost_price : ℝ, cost_price > 0 →
    let marked_price := cost_price * (1 + markup_percentage / 100)
    let discounted_price := marked_price * (1 - 25 / 100)
    let profit_percentage := (discounted_price - cost_price) / cost_price * 100
    profit_percentage = 20) →
  markup_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_merchant_markup_problem_l1048_104863


namespace NUMINAMATH_CALUDE_three_digit_primes_with_digit_product_189_l1048_104856

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def target_set : Set ℕ := {379, 397, 739, 937}

theorem three_digit_primes_with_digit_product_189 :
  ∀ n : ℕ, is_three_digit n ∧ Nat.Prime n ∧ digit_product n = 189 ↔ n ∈ target_set :=
sorry

end NUMINAMATH_CALUDE_three_digit_primes_with_digit_product_189_l1048_104856


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l1048_104806

theorem solution_implies_m_value (m : ℚ) : 
  (m * (-3) - 8 = 15 + m) → m = -23/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l1048_104806


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1048_104830

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
def distPointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def intersects (c : Circle) (l : Line) : Prop :=
  distPointToLine c.center l < c.radius

theorem line_circle_intersection (c : Circle) (l : Line) :
  distPointToLine c.center l < c.radius → intersects c l := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1048_104830


namespace NUMINAMATH_CALUDE_house_sale_tax_percentage_l1048_104812

theorem house_sale_tax_percentage (market_value : ℝ) (over_market_percentage : ℝ) 
  (num_people : ℕ) (amount_per_person : ℝ) :
  market_value = 500000 →
  over_market_percentage = 0.20 →
  num_people = 4 →
  amount_per_person = 135000 →
  (market_value * (1 + over_market_percentage) - num_people * amount_per_person) / 
    (market_value * (1 + over_market_percentage)) = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_tax_percentage_l1048_104812


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1048_104884

theorem quadratic_roots_problem (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + 2*k*x + k^2 = x + 1 ↔ x = x₁ ∨ x = x₂) ∧
    (3*x₁ - x₂)*(x₁ - 3*x₂) = 19) →
  k = 0 ∨ k = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1048_104884


namespace NUMINAMATH_CALUDE_russom_subway_tickets_l1048_104854

theorem russom_subway_tickets (bus_tickets : ℕ) (max_envelopes : ℕ) (subway_tickets : ℕ) : 
  bus_tickets = 18 →
  max_envelopes = 6 →
  bus_tickets % max_envelopes = 0 →
  subway_tickets % max_envelopes = 0 →
  subway_tickets > 0 →
  ∀ n : ℕ, n < subway_tickets → n % max_envelopes ≠ 0 ∨ n = 0 →
  subway_tickets = 6 :=
by sorry

end NUMINAMATH_CALUDE_russom_subway_tickets_l1048_104854


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1048_104850

theorem fraction_equation_solution :
  ∃ x : ℝ, x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1048_104850


namespace NUMINAMATH_CALUDE_total_cost_15_pencils_9_notebooks_l1048_104883

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 8 pencils and 5 notebooks cost $3.90 -/
axiom first_condition : 8 * pencil_cost + 5 * notebook_cost = 3.90

/-- The second given condition: 6 pencils and 4 notebooks cost $2.96 -/
axiom second_condition : 6 * pencil_cost + 4 * notebook_cost = 2.96

/-- The theorem to be proved -/
theorem total_cost_15_pencils_9_notebooks : 
  15 * pencil_cost + 9 * notebook_cost = 7.26 := by sorry

end NUMINAMATH_CALUDE_total_cost_15_pencils_9_notebooks_l1048_104883


namespace NUMINAMATH_CALUDE_fruit_packing_lcm_l1048_104868

theorem fruit_packing_lcm : Nat.lcm 18 (Nat.lcm 9 (Nat.lcm 12 6)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_fruit_packing_lcm_l1048_104868


namespace NUMINAMATH_CALUDE_existence_of_point_l1048_104873

theorem existence_of_point (f : ℝ → ℝ) (h_pos : ∀ x, f x > 0) (h_nondec : ∀ x y, x ≤ y → f x ≤ f y) :
  ∃ a : ℝ, f (a + 1 / f a) < 2 * f a := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_l1048_104873


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1048_104875

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- The equation of the new ellipse -/
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The point through which the new ellipse passes -/
def point : ℝ × ℝ := (3, -2)

theorem ellipse_theorem :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), given_ellipse x y → 
      ∃ (f1 f2 : ℝ × ℝ), (f1.1 = c ∧ f1.2 = 0) ∧ (f2.1 = -c ∧ f2.2 = 0) ∧
        (x - f1.1)^2 + y^2 + (x - f2.1)^2 + y^2 = 
        ((x - f1.1)^2 + y^2)^(1/2) + ((x - f2.1)^2 + y^2)^(1/2)) ∧
    (∀ (x y : ℝ), new_ellipse x y →
      ∃ (f1 f2 : ℝ × ℝ), (f1.1 = c ∧ f1.2 = 0) ∧ (f2.1 = -c ∧ f2.2 = 0) ∧
        (x - f1.1)^2 + y^2 + (x - f2.1)^2 + y^2 = 
        ((x - f1.1)^2 + y^2)^(1/2) + ((x - f2.1)^2 + y^2)^(1/2))) ∧
  new_ellipse point.1 point.2 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_theorem_l1048_104875


namespace NUMINAMATH_CALUDE_triangle_properties_l1048_104878

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 2 ∧ (1/2 * t.b * t.c * Real.sin t.A) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1048_104878


namespace NUMINAMATH_CALUDE_triangle_and_division_counts_l1048_104855

/-- The number of non-congruent triangles formed by m equally spaced points on a circle -/
def num_triangles (m : ℕ) : ℕ :=
  let k := m / 6
  match m % 6 with
  | 0 => 3*k^2 - 3*k + 1
  | 1 => 3*k^2 - 2*k
  | 2 => 3*k^2 - k
  | 3 => 3*k^2
  | 4 => 3*k^2 + k
  | 5 => 3*k^2 + 2*k
  | _ => 0  -- This case should never occur

/-- The number of ways to divide m identical items into 3 groups -/
def num_divisions (m : ℕ) : ℕ :=
  let k := m / 6
  match m % 6 with
  | 0 => 3*k^2
  | 1 => 3*k^2 + k
  | 2 => 3*k^2 + 2*k
  | 3 => 3*k^2 + 3*k + 1
  | 4 => 3*k^2 + 4*k + 1
  | 5 => 3*k^2 + 5*k + 2
  | _ => 0  -- This case should never occur

theorem triangle_and_division_counts (m : ℕ) (h : m ≥ 3) :
  (num_triangles m = num_triangles m) ∧ (num_divisions m = num_divisions m) :=
sorry

end NUMINAMATH_CALUDE_triangle_and_division_counts_l1048_104855


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l1048_104882

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_solution (a : ℕ → ℝ) (r : ℝ) :
  is_geometric_sequence a r →
  r > 0 →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l1048_104882


namespace NUMINAMATH_CALUDE_solution_existence_l1048_104802

/-- Given a positive real number a, prove the existence of real solutions for two systems of equations involving parameter m. -/
theorem solution_existence (a : ℝ) (ha : a > 0) :
  (∀ m : ℝ, ∃ x y : ℝ, y = m * x + a ∧ 1 / x - 1 / y = 1 / a) ∧
  (∀ m : ℝ, (m ≤ 0 ∨ m ≥ 4) → ∃ x y : ℝ, y = m * x - a ∧ 1 / x - 1 / y = 1 / a) :=
by sorry

end NUMINAMATH_CALUDE_solution_existence_l1048_104802


namespace NUMINAMATH_CALUDE_fraction_equality_l1048_104872

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 5) : (2 * a + 3 * b) / a = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1048_104872


namespace NUMINAMATH_CALUDE_fraction_inequality_l1048_104864

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 → (4 * x + 3 ≤ 9 - 3 * x ↔ -1 ≤ x ∧ x ≤ 6/7) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1048_104864


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1048_104801

theorem polygon_diagonals (n : ℕ) : n ≥ 3 → (n - 3 = 5 ↔ n = 8) := by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1048_104801


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1048_104853

theorem basketball_team_selection (total_players : Nat) (twins : Nat) (lineup_size : Nat) : 
  total_players = 12 →
  twins = 2 →
  lineup_size = 5 →
  (twins * (total_players - twins).choose (lineup_size - 1)) = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1048_104853


namespace NUMINAMATH_CALUDE_decagon_division_impossible_l1048_104861

/-- Represents a division of a polygon into colored triangles -/
structure TriangleDivision (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  valid_division : black_sides - white_sides = n

/-- Checks if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := ∃ k, n = 3 * k

theorem decagon_division_impossible :
  ¬ ∃ (d : TriangleDivision 10),
    divisible_by_three d.black_sides ∧
    divisible_by_three d.white_sides :=
sorry

end NUMINAMATH_CALUDE_decagon_division_impossible_l1048_104861


namespace NUMINAMATH_CALUDE_function_value_at_ten_l1048_104877

theorem function_value_at_ten (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 7*x*y + 3*y^2 = f (3*x - y) + 3*x^2 + 2) : 
  f 10 = -123 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_ten_l1048_104877


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l1048_104893

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define a line with negative slope
def negative_slope_line (k t : ℝ) (x y : ℝ) : Prop := y = k*x + t ∧ k < 0

-- Define the centroid condition
def centroid_origin (a b c : ℝ × ℝ) : Prop :=
  a.1 + b.1 + c.1 = 0 ∧ a.2 + b.2 + c.2 = 0

-- Define the area ratio condition
def area_ratio_condition (b m a c o : ℝ × ℝ) : Prop :=
  2 * (b.1 - m.1) * (a.2 - m.2) = 3 * (c.1 - m.1) * (o.2 - m.2)

-- Main theorem
theorem ellipse_slope_theorem (a b c m : ℝ × ℝ) (k t : ℝ) :
  point_on_ellipse a ∧ point_on_ellipse b ∧ point_on_ellipse c ∧
  negative_slope_line k t b.1 b.2 ∧
  negative_slope_line k t c.1 c.2 ∧
  m.1 = 0 ∧
  centroid_origin a b c ∧
  area_ratio_condition b m a c (0, 0) →
  k = -3*Real.sqrt 3/2 ∨ k = -Real.sqrt 3/6 := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l1048_104893


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l1048_104828

def total_balls : ℕ := 40
def red_balls : ℕ := 16
def blue_balls : ℕ := 12
def white_balls : ℕ := 8
def yellow_balls : ℕ := 4
def sample_size : ℕ := 10

def stratified_sample_red : ℕ := 4
def stratified_sample_blue : ℕ := 3
def stratified_sample_white : ℕ := 2
def stratified_sample_yellow : ℕ := 1

theorem stratified_sampling_probability :
  (Nat.choose yellow_balls stratified_sample_yellow *
   Nat.choose white_balls stratified_sample_white *
   Nat.choose blue_balls stratified_sample_blue *
   Nat.choose red_balls stratified_sample_red) /
  Nat.choose total_balls sample_size =
  (Nat.choose 4 1 * Nat.choose 8 2 * Nat.choose 12 3 * Nat.choose 16 4) /
  Nat.choose 40 10 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l1048_104828


namespace NUMINAMATH_CALUDE_intersection_point_l1048_104814

/-- Curve C₁ is defined by y = √x for x ≥ 0 -/
def C₁ (x y : ℝ) : Prop := y = Real.sqrt x ∧ x ≥ 0

/-- Curve C₂ is defined by x² + y² = 2 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- The point (1, 1) is the unique intersection point of curves C₁ and C₂ -/
theorem intersection_point : 
  (∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ C₂ p.1 p.2) ∧ 
  (C₁ 1 1 ∧ C₂ 1 1) := by
  sorry

#check intersection_point

end NUMINAMATH_CALUDE_intersection_point_l1048_104814


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l1048_104836

theorem common_factor_of_polynomial (x : ℝ) :
  ∃ (k : ℝ), 2*x^2 - 8*x = 2*x*k :=
by
  sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l1048_104836
