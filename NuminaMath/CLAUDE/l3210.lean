import Mathlib

namespace NUMINAMATH_CALUDE_chinese_character_number_puzzle_l3210_321040

theorem chinese_character_number_puzzle :
  ∃! (A B C D : Nat),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    A * 10 + B = 19 ∧
    C * 10 + D = 62 ∧
    (A * 1000 + B * 100 + C * 10 + D) - (A * 1000 + A * 100 + B * 10 + B) = 124 := by
  sorry

end NUMINAMATH_CALUDE_chinese_character_number_puzzle_l3210_321040


namespace NUMINAMATH_CALUDE_common_root_quadratic_l3210_321093

theorem common_root_quadratic (a b : ℝ) : 
  (∃! t : ℝ, t^2 + a*t + b = 0 ∧ t^2 + b*t + a = 0) → 
  (a + b + 1 = 0 ∧ a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_l3210_321093


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l3210_321034

/-- Given an arc length of 4 and a central angle of 2 radians, the radius of the circle is 2. -/
theorem circle_radius_from_arc_and_angle (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) 
    (h1 : arc_length = 4)
    (h2 : central_angle = 2)
    (h3 : arc_length = radius * central_angle) : 
  radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l3210_321034


namespace NUMINAMATH_CALUDE_correct_combined_average_l3210_321070

def num_students : ℕ := 100
def math_avg : ℚ := 85
def science_avg : ℚ := 89
def num_incorrect : ℕ := 5

def incorrect_math_marks : List ℕ := [76, 80, 95, 70, 90]
def correct_math_marks : List ℕ := [86, 70, 75, 90, 100]
def incorrect_science_marks : List ℕ := [105, 60, 80, 92, 78]
def correct_science_marks : List ℕ := [95, 70, 90, 82, 88]

theorem correct_combined_average :
  let math_total := num_students * math_avg + (correct_math_marks.sum - incorrect_math_marks.sum)
  let science_total := num_students * science_avg + (correct_science_marks.sum - incorrect_science_marks.sum)
  let combined_total := math_total + science_total
  let combined_avg := combined_total / (2 * num_students)
  combined_avg = 87.1 := by sorry

end NUMINAMATH_CALUDE_correct_combined_average_l3210_321070


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_bisector_proportion_l3210_321095

/-- Represents a triangle with side lengths and an angle bisector -/
structure BisectedTriangle where
  -- Side lengths
  p : ℝ
  q : ℝ
  r : ℝ
  -- Length of angle bisector segments
  u : ℝ
  v : ℝ
  -- Conditions
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r
  triangle_ineq : p < q + r ∧ q < p + r ∧ r < p + q
  bisector_sum : u + v = p

/-- The angle bisector theorem holds for this triangle -/
theorem angle_bisector_theorem (t : BisectedTriangle) : t.u / t.q = t.v / t.r := sorry

/-- The main theorem: proving the proportion involving v and r -/
theorem bisector_proportion (t : BisectedTriangle) : t.v / t.r = t.p / (t.q + t.r) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_bisector_proportion_l3210_321095


namespace NUMINAMATH_CALUDE_equation1_unique_solution_equation2_no_solution_l3210_321004

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  5 / (2 * x) - 1 / (x - 3) = 0

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  1 / (x - 2) = 4 / (x^2 - 4)

-- Theorem for the first equation
theorem equation1_unique_solution :
  ∃! x : ℝ, equation1 x ∧ x = 5 :=
sorry

-- Theorem for the second equation
theorem equation2_no_solution :
  ∀ x : ℝ, ¬ equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_unique_solution_equation2_no_solution_l3210_321004


namespace NUMINAMATH_CALUDE_sum_digits_first_2002_even_integers_l3210_321002

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The nth positive even integer -/
def nthEvenInteger (n : ℕ) : ℕ := sorry

/-- The sum of digits for the first n positive even integers -/
def sumDigitsFirstNEvenIntegers (n : ℕ) : ℕ := sorry

/-- Theorem: The total number of digits used to write the first 2002 positive even integers is 7456 -/
theorem sum_digits_first_2002_even_integers : 
  sumDigitsFirstNEvenIntegers 2002 = 7456 := by sorry

end NUMINAMATH_CALUDE_sum_digits_first_2002_even_integers_l3210_321002


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l3210_321073

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l3210_321073


namespace NUMINAMATH_CALUDE_prism_diagonal_angle_l3210_321013

/-- Given a right prism with a right triangular base, where one acute angle of the base is α
    and the largest lateral face is a square, this theorem states that the angle β between
    the intersecting diagonals of the other two lateral faces is arccos(2 / √(8 + sin²(2α))) -/
theorem prism_diagonal_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
  ∃ (β : ℝ),
    β = Real.arccos (2 / Real.sqrt (8 + Real.sin (2 * α) ^ 2)) ∧
    0 ≤ β ∧
    β ≤ π :=
sorry

end NUMINAMATH_CALUDE_prism_diagonal_angle_l3210_321013


namespace NUMINAMATH_CALUDE_intersection_integer_iff_k_valid_l3210_321092

/-- A point in the Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Check if a point is the intersection of two lines -/
def is_intersection (p : Point) (k : ℤ) : Prop :=
  p.y = p.x - 2 ∧ p.y = k * p.x + k

/-- The set of valid k values -/
def valid_k : Set ℤ := {-2, 0, 2, 4}

/-- Main theorem: The intersection is an integer point iff k is in the valid set -/
theorem intersection_integer_iff_k_valid (k : ℤ) :
  (∃ p : Point, is_intersection p k) ↔ k ∈ valid_k :=
sorry

end NUMINAMATH_CALUDE_intersection_integer_iff_k_valid_l3210_321092


namespace NUMINAMATH_CALUDE_amusing_numbers_l3210_321084

def is_amusing (x : Nat) : Prop :=
  (1000 ≤ x ∧ x < 10000) ∧
  ∃ y : Nat, (1000 ≤ y ∧ y < 10000) ∧
  (y % x = 0) ∧
  (∀ i : Fin 4,
    let x_digit := (x / (10 ^ i.val)) % 10
    let y_digit := (y / (10 ^ i.val)) % 10
    (x_digit = 0 ∧ y_digit = 1) ∨
    (x_digit = 9 ∧ y_digit = 8) ∨
    (x_digit ≠ 0 ∧ x_digit ≠ 9 ∧ (y_digit = x_digit - 1 ∨ y_digit = x_digit + 1)))

theorem amusing_numbers :
  is_amusing 1111 ∧ is_amusing 1091 ∧ is_amusing 1109 ∧ is_amusing 1089 :=
sorry

end NUMINAMATH_CALUDE_amusing_numbers_l3210_321084


namespace NUMINAMATH_CALUDE_additional_male_workers_hired_l3210_321041

theorem additional_male_workers_hired (
  initial_female_percentage : ℚ)
  (final_female_percentage : ℚ)
  (final_total_employees : ℕ)
  (h1 : initial_female_percentage = 3/5)
  (h2 : final_female_percentage = 11/20)
  (h3 : final_total_employees = 240) :
  (final_total_employees : ℚ) - (final_female_percentage * final_total_employees) / initial_female_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_additional_male_workers_hired_l3210_321041


namespace NUMINAMATH_CALUDE_linear_function_domain_range_l3210_321026

def LinearFunction (k b : ℚ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_domain_range 
  (k b : ℚ) 
  (h_domain : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → LinearFunction k b x ∈ Set.Icc (-4 : ℝ) 1) 
  (h_range : Set.Icc (-4 : ℝ) 1 ⊆ Set.range (LinearFunction k b)) :
  (k = 5/6 ∧ b = -3/2) ∨ (k = -5/6 ∧ b = -3/2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_domain_range_l3210_321026


namespace NUMINAMATH_CALUDE_kelly_games_l3210_321091

theorem kelly_games (games_given_away : ℕ) (games_left : ℕ) : games_given_away = 91 → games_left = 92 → games_given_away + games_left = 183 :=
by
  sorry

end NUMINAMATH_CALUDE_kelly_games_l3210_321091


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_equation_l3210_321014

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_equation :
  let a : ℝ := -16
  let b : ℝ := 48
  let c : ℝ := -75
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_equation_l3210_321014


namespace NUMINAMATH_CALUDE_tan_double_angle_l3210_321052

theorem tan_double_angle (α : Real) :
  (2 * Real.cos α + Real.sin α) / (Real.cos α - 2 * Real.sin α) = -1 →
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3210_321052


namespace NUMINAMATH_CALUDE_draw_condition_butterfly_wins_condition_l3210_321024

/-- Represents the outcome of the spider web game -/
inductive GameOutcome
  | Draw
  | ButterflyWins

/-- Defines the spider web game structure and rules -/
structure SpiderWebGame where
  K : Nat  -- Number of rings
  R : Nat  -- Number of radii
  butterfly_moves_first : Bool
  K_ge_2 : K ≥ 2
  R_ge_3 : R ≥ 3

/-- Determines the outcome of the spider web game -/
def game_outcome (game : SpiderWebGame) : GameOutcome :=
  if game.K ≥ Nat.ceil (game.R / 2) then
    GameOutcome.Draw
  else
    GameOutcome.ButterflyWins

/-- Theorem stating the conditions for a draw in the spider web game -/
theorem draw_condition (game : SpiderWebGame) :
  game_outcome game = GameOutcome.Draw ↔ game.K ≥ Nat.ceil (game.R / 2) :=
sorry

/-- Theorem stating the conditions for butterfly winning in the spider web game -/
theorem butterfly_wins_condition (game : SpiderWebGame) :
  game_outcome game = GameOutcome.ButterflyWins ↔ game.K < Nat.ceil (game.R / 2) :=
sorry

end NUMINAMATH_CALUDE_draw_condition_butterfly_wins_condition_l3210_321024


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3210_321001

theorem inserted_numbers_sum (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ 
  2 < a ∧ a < b ∧ b < 12 ∧
  (∃ r : ℝ, r > 0 ∧ a = 2 * r ∧ b = 2 * r^2) ∧
  (∃ d : ℝ, b = a + d ∧ 12 = b + d) →
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3210_321001


namespace NUMINAMATH_CALUDE_g_of_3_eq_125_l3210_321021

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 4 * x^2 - 7 * x + 2

theorem g_of_3_eq_125 : g 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_125_l3210_321021


namespace NUMINAMATH_CALUDE_vacant_seats_l3210_321012

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 75 / 100) : 
  (1 - filled_percentage) * total_seats = 150 := by
sorry


end NUMINAMATH_CALUDE_vacant_seats_l3210_321012


namespace NUMINAMATH_CALUDE_complex_number_problem_l3210_321074

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property of being a purely imaginary number
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define the property of being a real number
def isRealNumber (z : ℂ) : Prop := z.im = 0

-- Theorem statement
theorem complex_number_problem (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isRealNumber ((z + 2) / (1 - i))) : 
  z = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3210_321074


namespace NUMINAMATH_CALUDE_expression_simplification_l3210_321058

theorem expression_simplification (x y z : ℝ) : 
  ((x + z) - (y - 2*z)) - ((x - 2*z) - (y + z)) = 6*z := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3210_321058


namespace NUMINAMATH_CALUDE_mcdonalds_coupon_value_l3210_321065

/-- Proves that given an original cost of $7.50, a senior citizen discount of 20%,
    and a final payment of $4, the coupon value that makes this possible is $2.50. -/
theorem mcdonalds_coupon_value :
  let original_cost : ℝ := 7.50
  let senior_discount : ℝ := 0.20
  let final_payment : ℝ := 4.00
  let coupon_value : ℝ := 2.50
  (1 - senior_discount) * (original_cost - coupon_value) = final_payment := by
sorry

end NUMINAMATH_CALUDE_mcdonalds_coupon_value_l3210_321065


namespace NUMINAMATH_CALUDE_range_of_m_l3210_321083

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -7 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 7}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 3*m - 2}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3210_321083


namespace NUMINAMATH_CALUDE_andrew_grapes_purchase_l3210_321020

/-- The quantity of grapes (in kg) purchased by Andrew -/
def grapes_quantity : ℕ := sorry

/-- The cost of grapes per kg -/
def grapes_cost_per_kg : ℕ := 54

/-- The quantity of mangoes (in kg) purchased by Andrew -/
def mangoes_quantity : ℕ := 10

/-- The cost of mangoes per kg -/
def mangoes_cost_per_kg : ℕ := 62

/-- The total amount paid by Andrew -/
def total_paid : ℕ := 1376

theorem andrew_grapes_purchase :
  grapes_quantity * grapes_cost_per_kg + 
  mangoes_quantity * mangoes_cost_per_kg = total_paid ∧
  grapes_quantity = 14 := by sorry

end NUMINAMATH_CALUDE_andrew_grapes_purchase_l3210_321020


namespace NUMINAMATH_CALUDE_football_players_count_l3210_321048

theorem football_players_count (total_players cricket_players hockey_players softball_players : ℕ) 
  (h1 : total_players = 59)
  (h2 : cricket_players = 16)
  (h3 : hockey_players = 12)
  (h4 : softball_players = 13) :
  total_players - (cricket_players + hockey_players + softball_players) = 18 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l3210_321048


namespace NUMINAMATH_CALUDE_truck_catch_up_time_is_fifteen_l3210_321097

/-- Represents a vehicle with a constant speed -/
structure Vehicle where
  speed : ℝ

/-- Represents the state of the vehicles at a given time -/
structure VehicleState where
  bus : Vehicle
  truck : Vehicle
  car : Vehicle
  time : ℝ
  busTruckDistance : ℝ
  truckCarDistance : ℝ

/-- The initial state of the vehicles -/
def initialState : VehicleState := sorry

/-- The state after the car catches up with the truck -/
def carTruckCatchUpState : VehicleState := sorry

/-- The state after the car catches up with the bus -/
def carBusCatchUpState : VehicleState := sorry

/-- The state after the truck catches up with the bus -/
def truckBusCatchUpState : VehicleState := sorry

/-- The time it takes for the truck to catch up with the bus after the car catches up with the bus -/
def truckCatchUpTime : ℝ := truckBusCatchUpState.time - carBusCatchUpState.time

theorem truck_catch_up_time_is_fifteen :
  truckCatchUpTime = 15 := by sorry

end NUMINAMATH_CALUDE_truck_catch_up_time_is_fifteen_l3210_321097


namespace NUMINAMATH_CALUDE_double_base_cost_increase_l3210_321016

/-- The cost function for a given base value -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem stating that doubling the base value results in a cost that is 1600% of the original -/
theorem double_base_cost_increase (t : ℝ) (b : ℝ) :
  cost t (2 * b) = 16 * cost t b :=
by sorry

end NUMINAMATH_CALUDE_double_base_cost_increase_l3210_321016


namespace NUMINAMATH_CALUDE_range_of_a_l3210_321039

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ - 1 < 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3210_321039


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3210_321043

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : angle_between a b = Real.pi / 4)
  (h2 : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2)
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 3) :
  Real.sqrt (((2 * a.1 - b.1) ^ 2) + ((2 * a.2 - b.2) ^ 2)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3210_321043


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3210_321010

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3210_321010


namespace NUMINAMATH_CALUDE_kitty_dusting_time_l3210_321019

/-- Represents the cleaning activities and their durations in Kitty's living room --/
structure CleaningActivities where
  pickingUpToys : ℕ
  vacuuming : ℕ
  cleaningWindows : ℕ
  totalWeeks : ℕ
  totalMinutes : ℕ

/-- Calculates the time spent dusting furniture each week --/
def dustingTime (c : CleaningActivities) : ℕ :=
  let otherTasksTime := c.pickingUpToys + c.vacuuming + c.cleaningWindows
  let totalOtherTasksTime := otherTasksTime * c.totalWeeks
  let totalDustingTime := c.totalMinutes - totalOtherTasksTime
  totalDustingTime / c.totalWeeks

/-- Theorem stating that Kitty spends 10 minutes each week dusting furniture --/
theorem kitty_dusting_time :
  ∀ (c : CleaningActivities),
    c.pickingUpToys = 5 →
    c.vacuuming = 20 →
    c.cleaningWindows = 15 →
    c.totalWeeks = 4 →
    c.totalMinutes = 200 →
    dustingTime c = 10 := by
  sorry

end NUMINAMATH_CALUDE_kitty_dusting_time_l3210_321019


namespace NUMINAMATH_CALUDE_odd_squares_sum_power_of_two_l3210_321045

theorem odd_squares_sum_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ x y : ℤ, Odd x ∧ Odd y ∧ x^2 + 7*y^2 = 2^n := by
  sorry

end NUMINAMATH_CALUDE_odd_squares_sum_power_of_two_l3210_321045


namespace NUMINAMATH_CALUDE_balls_in_original_positions_l3210_321007

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The number of transpositions performed -/
def num_transpositions : ℕ := 3

/-- The probability of a ball being in its original position after the transpositions -/
def prob_original_position : ℚ := 127 / 343

/-- The expected number of balls in their original positions after the transpositions -/
def expected_original_positions : ℚ := 889 / 343

theorem balls_in_original_positions :
  num_balls * prob_original_position = expected_original_positions := by sorry

end NUMINAMATH_CALUDE_balls_in_original_positions_l3210_321007


namespace NUMINAMATH_CALUDE_same_color_probability_l3210_321037

theorem same_color_probability (blue_balls yellow_balls : ℕ) 
  (h_blue : blue_balls = 8) (h_yellow : yellow_balls = 5) : 
  let total_balls := blue_balls + yellow_balls
  let prob_blue := blue_balls / total_balls
  let prob_yellow := yellow_balls / total_balls
  prob_blue ^ 2 + prob_yellow ^ 2 = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3210_321037


namespace NUMINAMATH_CALUDE_least_divisible_by_first_ten_l3210_321033

theorem least_divisible_by_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_first_ten_l3210_321033


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l3210_321075

theorem quadratic_constant_term (m : ℝ) : 
  (∀ x, m * x^2 + 2 * x + m^2 - 1 = 0) → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l3210_321075


namespace NUMINAMATH_CALUDE_range_of_m_l3210_321018

def proposition_p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*m) - y^2 / (m-1) = 1 ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ 
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ c < a

def proposition_q (m : ℝ) : Prop :=
  ∃ e : ℝ, 1 < e ∧ e < 2 ∧
  ∃ x y : ℝ, y^2 / 5 - x^2 / m = 1 ∧
  e^2 = (5 + m) / 5

theorem range_of_m :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) → 0 < m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3210_321018


namespace NUMINAMATH_CALUDE_geometry_problem_l3210_321008

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the line 2x + y = 0
def line_center (x y : ℝ) : Prop := 2*x + y = 0

theorem geometry_problem :
  -- Conditions
  (∀ x y, line_l x y → (x = 2 ∧ y = -1) → True) ∧  -- l passes through P(2,-1)
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, line_l x y ↔ x/a + y/b = 1) ∧  -- Sum of intercepts is 2
  (∃ m, line_center m (-2*m) ∧ ∀ x y, circle_M x y → line_center x y) ∧  -- M's center on 2x+y=0
  (∀ x y, circle_M x y → line_l x y → (x = 2 ∧ y = -1)) →  -- M tangent to l at P
  -- Conclusions
  (∀ x y, line_l x y ↔ x + y = 1) ∧  -- Equation of line l
  (∀ x y, circle_M x y ↔ (x - 1)^2 + (y + 2)^2 = 2) ∧  -- Equation of circle M
  (∃ y₁ y₂, y₁ < y₂ ∧ circle_M 0 y₁ ∧ circle_M 0 y₂ ∧ y₂ - y₁ = 2)  -- Length of chord on y-axis
  := by sorry

end NUMINAMATH_CALUDE_geometry_problem_l3210_321008


namespace NUMINAMATH_CALUDE_milk_added_to_full_can_l3210_321094

/-- Represents the contents of a can with milk and water -/
structure Can where
  milk : ℝ
  water : ℝ

/-- Represents the ratios of milk to water -/
structure Ratio where
  milk : ℝ
  water : ℝ

def Can.ratio (can : Can) : Ratio :=
  { milk := can.milk, water := can.water }

def Can.total (can : Can) : ℝ :=
  can.milk + can.water

theorem milk_added_to_full_can 
  (initial_ratio : Ratio) 
  (final_ratio : Ratio) 
  (capacity : ℝ) :
  initial_ratio.milk / initial_ratio.water = 4 / 3 →
  final_ratio.milk / final_ratio.water = 2 / 1 →
  capacity = 36 →
  ∃ (initial_can final_can : Can),
    initial_can.ratio = initial_ratio ∧
    final_can.ratio = final_ratio ∧
    final_can.total = capacity ∧
    final_can.water = initial_can.water ∧
    final_can.milk - initial_can.milk = 72 / 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_added_to_full_can_l3210_321094


namespace NUMINAMATH_CALUDE_coin_difference_l3210_321003

def coin_values : List Nat := [1, 5, 10, 25, 50]
def target_amount : Nat := 65

def min_coins (amount : Nat) (coins : List Nat) : Nat :=
  sorry

def max_coins (amount : Nat) (coins : List Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins target_amount coin_values - min_coins target_amount coin_values = 62 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_l3210_321003


namespace NUMINAMATH_CALUDE_eugene_pencils_l3210_321079

/-- Calculates the total number of pencils Eugene has after receiving more. -/
def total_pencils (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Eugene's total pencils -/
theorem eugene_pencils : total_pencils 51 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l3210_321079


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3210_321066

theorem arithmetic_computation : -7 * 5 - (-4 * -2) + (-9 * -6) = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3210_321066


namespace NUMINAMATH_CALUDE_min_sum_of_product_2010_l3210_321017

theorem min_sum_of_product_2010 :
  ∃ (min : ℕ), min = 78 ∧
  ∀ (a b c : ℕ), 
    a > 0 → b > 0 → c > 0 →
    a * b * c = 2010 →
    a + b + c ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2010_l3210_321017


namespace NUMINAMATH_CALUDE_exists_expression_equal_100_l3210_321011

/-- Represents a sequence of digits with operators between them -/
inductive DigitExpression
  | single : Nat → DigitExpression
  | add : DigitExpression → DigitExpression → DigitExpression
  | sub : DigitExpression → DigitExpression → DigitExpression

/-- Evaluates a DigitExpression to its integer value -/
def evaluate : DigitExpression → Int
  | DigitExpression.single n => n
  | DigitExpression.add a b => evaluate a + evaluate b
  | DigitExpression.sub a b => evaluate a - evaluate b

/-- Checks if a DigitExpression uses the digits 1 to 9 in order -/
def usesDigitsInOrder : DigitExpression → Bool := sorry

/-- The main theorem stating that there exists a valid expression equaling 100 -/
theorem exists_expression_equal_100 : 
  ∃ (expr : DigitExpression), usesDigitsInOrder expr ∧ evaluate expr = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_expression_equal_100_l3210_321011


namespace NUMINAMATH_CALUDE_square_side_length_l3210_321087

theorem square_side_length (perimeter : ℚ) (h : perimeter = 12 / 25) :
  perimeter / 4 = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3210_321087


namespace NUMINAMATH_CALUDE_min_value_f_l3210_321050

/-- The function f(x) = (x^2 + 2) / x has a minimum value of 2√2 for x > 1 -/
theorem min_value_f (x : ℝ) (h : x > 1) : (x^2 + 2) / x ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l3210_321050


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_l3210_321036

theorem smallest_solution_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 → x ≥ -Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_l3210_321036


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l3210_321047

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l3210_321047


namespace NUMINAMATH_CALUDE_example_theorem_l3210_321044

-- Define the necessary types and structures

-- State the theorem
theorem example_theorem (hypothesis1 : Type) (hypothesis2 : Type) : conclusion_type :=
  -- The proof would go here, but we're using sorry as requested
  sorry

-- Additional definitions or lemmas if needed

end NUMINAMATH_CALUDE_example_theorem_l3210_321044


namespace NUMINAMATH_CALUDE_darnel_sprint_distance_l3210_321015

theorem darnel_sprint_distance (jogged_distance : Real) (additional_sprint : Real) :
  jogged_distance = 0.75 →
  additional_sprint = 0.13 →
  jogged_distance + additional_sprint = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_darnel_sprint_distance_l3210_321015


namespace NUMINAMATH_CALUDE_log3_20_approximation_l3210_321081

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.477

-- Define the target value
def target_value : ℝ := 2.7

-- State the theorem
theorem log3_20_approximation :
  let log3_20 := (1 + log10_2_approx) / log10_3_approx
  abs (log3_20 - target_value) < 0.05 := by sorry

end NUMINAMATH_CALUDE_log3_20_approximation_l3210_321081


namespace NUMINAMATH_CALUDE_ab_max_and_sum_min_l3210_321000

theorem ab_max_and_sum_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 7 * b = 10) :
  (ab ≤ 25 / 21) ∧ (3 / a + 7 / b ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_ab_max_and_sum_min_l3210_321000


namespace NUMINAMATH_CALUDE_max_product_sum_1998_l3210_321054

theorem max_product_sum_1998 :
  ∀ x y : ℤ, x + y = 1998 → x * y ≤ 998001 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_1998_l3210_321054


namespace NUMINAMATH_CALUDE_expression_evaluation_l3210_321053

theorem expression_evaluation :
  let x : ℚ := -1/3
  (3*x + 2) * (3*x - 2) - 5*x*(x - 1) - (2*x - 1)^2 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3210_321053


namespace NUMINAMATH_CALUDE_exists_self_intersecting_net_l3210_321076

/-- A tetrahedron is represented by its four vertices in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A net of a tetrahedron is represented by the 2D coordinates of its vertices -/
structure TetrahedronNet where
  vertices : Fin 4 → ℝ × ℝ

/-- A function that determines if a tetrahedron net self-intersects -/
def self_intersects (net : TetrahedronNet) : Prop :=
  sorry

/-- A function that cuts a tetrahedron along three edges not belonging to the same face -/
def cut_tetrahedron (t : Tetrahedron) : TetrahedronNet :=
  sorry

/-- The main theorem: there exists a tetrahedron whose net self-intersects -/
theorem exists_self_intersecting_net :
  ∃ t : Tetrahedron, self_intersects (cut_tetrahedron t) :=
sorry

end NUMINAMATH_CALUDE_exists_self_intersecting_net_l3210_321076


namespace NUMINAMATH_CALUDE_price_reduction_problem_l3210_321085

/-- The price reduction problem -/
theorem price_reduction_problem (reduced_price : ℝ) (extra_oil : ℝ) (total_money : ℝ) 
  (h1 : reduced_price = 15)
  (h2 : extra_oil = 6)
  (h3 : total_money = 900) :
  let original_price := total_money / (total_money / reduced_price - extra_oil)
  let percentage_reduction := (original_price - reduced_price) / original_price * 100
  ∃ (ε : ℝ), ε > 0 ∧ abs (percentage_reduction - 10) < ε :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_problem_l3210_321085


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3210_321078

theorem coin_flip_probability : 
  let p_heads : ℝ := 1/2  -- probability of getting heads on a single flip
  let n : ℕ := 5  -- number of flips
  let target_sequence := List.replicate 4 true ++ [false]  -- HTTT (true for heads, false for tails)
  
  (target_sequence.map (fun h => if h then p_heads else 1 - p_heads)).prod = 1/32 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3210_321078


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l3210_321023

/-- 
Calculates the number of games required in a single-elimination tournament
to declare a winner, given the number of teams participating.
-/
def gamesRequired (numTeams : ℕ) : ℕ := numTeams - 1

/-- 
Theorem: In a single-elimination tournament with 25 teams and no possibility of ties,
the number of games required to declare a winner is 24.
-/
theorem single_elimination_tournament_games :
  gamesRequired 25 = 24 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l3210_321023


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l3210_321031

def K' : ℚ := 1/1 + 1/2 + 1/3 + 1/4 + 1/5

def T (n : ℕ) : ℚ := n * (5^(n-1)) * K'

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_T :
  ∀ n : ℕ, n > 0 → (is_integer (T n) ↔ n ≥ 24) ∧
  ∀ m : ℕ, m < 24 → ¬ is_integer (T m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l3210_321031


namespace NUMINAMATH_CALUDE_y_value_proof_l3210_321025

theorem y_value_proof (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (5 * y) * Real.sqrt (7 * y) * Real.sqrt (21 * y) = 21) : 
  y = 1 / Real.rpow 20 (1/4) :=
sorry

end NUMINAMATH_CALUDE_y_value_proof_l3210_321025


namespace NUMINAMATH_CALUDE_proposition_equivalences_and_set_equality_l3210_321088

-- Define the proposition
def P (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def Q (x : ℝ) : Prop := x = 1 ∨ x = 2

-- Define the sets P and S
def setP : Set ℝ := {x | -1 < x ∧ x < 3}
def setS (a : ℝ) : Set ℝ := {x | x^2 + (a+1)*x + a < 0}

theorem proposition_equivalences_and_set_equality :
  (∀ x, Q x → P x) ∧
  (∀ x, ¬(P x) → ¬(Q x)) ∧
  (∀ x, ¬(Q x) → ¬(P x)) ∧
  ∃ a, setP = setS a ∧ a = -3 := by sorry

end NUMINAMATH_CALUDE_proposition_equivalences_and_set_equality_l3210_321088


namespace NUMINAMATH_CALUDE_cat_gemstone_difference_l3210_321005

/-- Given three cats with gemstone collars, prove the difference between Spaatz's 
    gemstones and half of Frankie's gemstones. -/
theorem cat_gemstone_difference (binkie frankie spaatz : ℕ) : 
  binkie = 24 →
  spaatz = 1 →
  binkie = 4 * frankie →
  spaatz = frankie →
  spaatz - (frankie / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_gemstone_difference_l3210_321005


namespace NUMINAMATH_CALUDE_paths_through_F_l3210_321068

/-- The number of paths on a grid from (0,0) to (a,b) -/
def gridPaths (a b : ℕ) : ℕ := Nat.choose (a + b) a

/-- The coordinates of point E -/
def E : ℕ × ℕ := (0, 0)

/-- The coordinates of point F -/
def F : ℕ × ℕ := (5, 2)

/-- The coordinates of point G -/
def G : ℕ × ℕ := (6, 5)

/-- The total number of steps from E to G -/
def totalSteps : ℕ := G.1 - E.1 + G.2 - E.2

theorem paths_through_F : 
  gridPaths (F.1 - E.1) (F.2 - E.2) * gridPaths (G.1 - F.1) (G.2 - F.2) = 84 ∧
  totalSteps = 12 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_F_l3210_321068


namespace NUMINAMATH_CALUDE_complex_number_location_l3210_321027

theorem complex_number_location :
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (0 < z.re) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3210_321027


namespace NUMINAMATH_CALUDE_least_divisor_for_perfect_square_l3210_321072

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem least_divisor_for_perfect_square :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (16800 / n) ∧ 
  ∀ m : ℕ, m > 0 → is_perfect_square (16800 / m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_divisor_for_perfect_square_l3210_321072


namespace NUMINAMATH_CALUDE_bee_speed_l3210_321080

/-- The speed of a bee flying between flowers -/
theorem bee_speed (time_to_rose time_to_poppy : ℝ)
  (distance_difference speed_difference : ℝ)
  (h1 : time_to_rose = 10)
  (h2 : time_to_poppy = 6)
  (h3 : distance_difference = 8)
  (h4 : speed_difference = 3) :
  ∃ (speed_to_rose : ℝ),
    speed_to_rose * time_to_rose = 
    (speed_to_rose + speed_difference) * time_to_poppy + distance_difference ∧
    speed_to_rose = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_bee_speed_l3210_321080


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3210_321069

-- Define the length of the train in meters
def train_length : ℝ := 310

-- Define the length of the platform in meters
def platform_length : ℝ := 210

-- Define the time taken to cross the platform in seconds
def crossing_time : ℝ := 26

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / crossing_time
  let speed_kmhr := speed_ms * conversion_factor
  speed_kmhr = 72 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3210_321069


namespace NUMINAMATH_CALUDE_sum_x_y_value_l3210_321071

theorem sum_x_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : x + 3 * y = -1) : 
  x + y = 29 / 13 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_value_l3210_321071


namespace NUMINAMATH_CALUDE_group_size_proof_l3210_321056

theorem group_size_proof (total_paise : ℕ) (contribution : ℕ → ℕ) : 
  (total_paise = 1369) →
  (∀ n : ℕ, contribution n = n) →
  (∃ n : ℕ, n * contribution n = total_paise) →
  (∃ n : ℕ, n * n = total_paise) →
  (∃ n : ℕ, n = 37) :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l3210_321056


namespace NUMINAMATH_CALUDE_race_time_difference_l3210_321042

/-- 
Given a 1000-meter race where runner A completes the race in 192 seconds and 
is 40 meters ahead of runner B at the finish line, prove that A beats B by 7.68 seconds.
-/
theorem race_time_difference (race_distance : ℝ) (a_time : ℝ) (distance_difference : ℝ) : 
  race_distance = 1000 →
  a_time = 192 →
  distance_difference = 40 →
  (race_distance / a_time) * (distance_difference / race_distance) * a_time = 7.68 :=
by sorry

end NUMINAMATH_CALUDE_race_time_difference_l3210_321042


namespace NUMINAMATH_CALUDE_supplementary_to_complementary_ratio_l3210_321096

/-- 
Given an angle of 45 degrees, prove that the ratio of its supplementary angle 
to its complementary angle is 3:1.
-/
theorem supplementary_to_complementary_ratio 
  (angle : ℝ) 
  (h_angle : angle = 45) 
  (h_supplementary : ℝ → ℝ → Prop)
  (h_complementary : ℝ → ℝ → Prop)
  (h_supp_def : ∀ x y, h_supplementary x y ↔ x + y = 180)
  (h_comp_def : ∀ x y, h_complementary x y ↔ x + y = 90) :
  (180 - angle) / (90 - angle) = 3 := by
sorry

end NUMINAMATH_CALUDE_supplementary_to_complementary_ratio_l3210_321096


namespace NUMINAMATH_CALUDE_grocery_store_soda_bottles_l3210_321064

/-- 
Given a grocery store with regular and diet soda bottles, this theorem proves 
the number of diet soda bottles, given the number of regular soda bottles and 
the difference between regular and diet soda bottles.
-/
theorem grocery_store_soda_bottles 
  (regular_soda : ℕ) 
  (difference : ℕ) 
  (h1 : regular_soda = 67)
  (h2 : regular_soda = difference + diet_soda) : 
  diet_soda = 9 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_bottles_l3210_321064


namespace NUMINAMATH_CALUDE_sector_perimeter_l3210_321030

/-- The perimeter of a circular sector with a central angle of 180 degrees and a radius of 28.000000000000004 cm is 143.96459430079216 cm. -/
theorem sector_perimeter : 
  let r : ℝ := 28.000000000000004
  let θ : ℝ := 180
  let arc_length : ℝ := (θ / 360) * 2 * Real.pi * r
  let perimeter : ℝ := arc_length + 2 * r
  perimeter = 143.96459430079216 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3210_321030


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3210_321022

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

theorem inequality_solution_set 
  (a : ℝ)
  (h1 : ∀ x y : ℝ, x < y → f a x > f a y)
  (h2 : ∀ x : ℝ, f a (-x) = -(f a x)) :
  {t : ℝ | f a (2*t + 1) + f a (t - 5) ≤ 0} = {t : ℝ | t ≥ 4/3} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3210_321022


namespace NUMINAMATH_CALUDE_sequence_equality_l3210_321006

theorem sequence_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ (i : ℕ), a i = i := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l3210_321006


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3210_321077

/-- Given two perpendicular lines (3a+2)x+(1-4a)y+8=0 and (5a-2)x+(a+4)y-7=0, prove that a = 0 or a = 12/11 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  ((3*a+2) * (5*a-2) + (1-4*a) * (a+4) = 0) → (a = 0 ∨ a = 12/11) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3210_321077


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l3210_321028

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define the universal set R (real numbers)
def R : Set ℝ := univ

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 6} := by sorry

-- Theorem for the union of complement of B and A
theorem union_complement_B_A : (R \ B) ∪ A = {x | x < 6 ∨ 9 ≤ x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_A_l3210_321028


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3210_321032

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y k : ℚ) : 
  (y = 4*x - 2) ∧ (y = -3*x + 9) ∧ (y = 2*x + k) → k = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3210_321032


namespace NUMINAMATH_CALUDE_non_decreasing_integers_count_l3210_321063

/-- The number of digits in the integers we're considering -/
def n : ℕ := 11

/-- The number of possible digit values (1 to 9) -/
def k : ℕ := 9

/-- The number of 11-digit positive integers with non-decreasing digits -/
def non_decreasing_integers : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem non_decreasing_integers_count : non_decreasing_integers = 75582 := by
  sorry

end NUMINAMATH_CALUDE_non_decreasing_integers_count_l3210_321063


namespace NUMINAMATH_CALUDE_third_number_proof_l3210_321062

theorem third_number_proof (a b c : ℝ) : 
  (a + b + c) / 3 = 48 → 
  (a + b) / 2 = 56 → 
  c = 32 := by
sorry

end NUMINAMATH_CALUDE_third_number_proof_l3210_321062


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3210_321049

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x ≤ -3) ∧
  (∃ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x = -3) →
  a = Real.sqrt 6 + 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l3210_321049


namespace NUMINAMATH_CALUDE_cinnamon_swirls_theorem_l3210_321089

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- The total number of cinnamon swirl pieces prepared -/
def total_pieces : ℕ := num_people * janes_pieces

theorem cinnamon_swirls_theorem :
  total_pieces = 12 :=
sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_theorem_l3210_321089


namespace NUMINAMATH_CALUDE_factorization_equality_l3210_321057

theorem factorization_equality (a b : ℝ) : 2*a - 8*a*b^2 = 2*a*(1-2*b)*(1+2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3210_321057


namespace NUMINAMATH_CALUDE_angle_bisection_quadrant_l3210_321009

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * Real.pi)

theorem angle_bisection_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_angle_bisection_quadrant_l3210_321009


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3210_321090

theorem sqrt_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq : a + b = c + d) (ineq : a < c ∧ c ≤ d ∧ d < b) :
  Real.sqrt a + Real.sqrt b < Real.sqrt c + Real.sqrt d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3210_321090


namespace NUMINAMATH_CALUDE_distance_per_block_l3210_321060

/-- Proves that the distance of each block is 1/8 mile -/
theorem distance_per_block (total_time : ℚ) (total_blocks : ℕ) (speed : ℚ) :
  total_time = 10 / 60 →
  total_blocks = 16 →
  speed = 12 →
  (speed * total_time) / total_blocks = 1 / 8 := by
  sorry

#check distance_per_block

end NUMINAMATH_CALUDE_distance_per_block_l3210_321060


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3210_321029

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 5 * p - 7 = 0) → 
  (3 * q^2 + 5 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3210_321029


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l3210_321038

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {2, 4}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l3210_321038


namespace NUMINAMATH_CALUDE_frog_weight_ratio_l3210_321051

/-- The ratio of the weight of the largest frog to the smallest frog is 10 -/
theorem frog_weight_ratio :
  ∀ (small_frog large_frog : ℝ),
  large_frog = 120 →
  large_frog = small_frog + 108 →
  large_frog / small_frog = 10 := by
sorry

end NUMINAMATH_CALUDE_frog_weight_ratio_l3210_321051


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_is_1_A_intersect_B_equals_B_l3210_321082

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem intersection_A_complement_B_when_a_is_1 :
  A ∩ (Set.univ \ B 1) = {x | -2 ≤ x ∧ x ≤ 0} ∪ {x | 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem A_intersect_B_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry


end NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_is_1_A_intersect_B_equals_B_l3210_321082


namespace NUMINAMATH_CALUDE_toys_ratio_l3210_321055

def num_friends : ℕ := 4
def total_toys : ℕ := 118

theorem toys_ratio : 
  ∃ (toys_to_B : ℕ), 
    toys_to_B * num_friends = total_toys ∧ 
    (toys_to_B : ℚ) / total_toys = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_toys_ratio_l3210_321055


namespace NUMINAMATH_CALUDE_yarn_ball_ratio_l3210_321067

/-- Given three balls of yarn, where:
    - The third ball is three times as large as the first ball
    - 27 feet of yarn was used for the third ball
    - 18 feet of yarn was used for the second ball
    Prove that the ratio of the size of the first ball to the size of the second ball is 1:2 -/
theorem yarn_ball_ratio :
  ∀ (first_ball second_ball third_ball : ℝ),
  third_ball = 3 * first_ball →
  third_ball = 27 →
  second_ball = 18 →
  first_ball / second_ball = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_yarn_ball_ratio_l3210_321067


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3210_321099

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (|x + 1| ≤ 4) → (-6 ≤ x ∧ x ≤ 3) ∧
  ∃ y : ℝ, -6 ≤ y ∧ y ≤ 3 ∧ |y + 1| > 4 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3210_321099


namespace NUMINAMATH_CALUDE_ryan_learning_time_l3210_321046

/-- Represents the time Ryan spends on learning languages in hours -/
structure LearningTime where
  total : ℝ
  english : ℝ
  chinese : ℝ

/-- Theorem: Given Ryan's total learning time and English learning time, 
    prove that his Chinese learning time is the difference -/
theorem ryan_learning_time (rt : LearningTime) 
  (h1 : rt.total = 3) 
  (h2 : rt.english = 2) 
  (h3 : rt.total = rt.english + rt.chinese) : 
  rt.chinese = 1 := by
sorry

end NUMINAMATH_CALUDE_ryan_learning_time_l3210_321046


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_five_l3210_321059

theorem floor_ceiling_sum_five (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 5) ↔ (2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_five_l3210_321059


namespace NUMINAMATH_CALUDE_pepsi_volume_l3210_321035

theorem pepsi_volume (maaza : ℕ) (sprite : ℕ) (total_cans : ℕ) (pepsi : ℕ) : 
  maaza = 40 →
  sprite = 368 →
  total_cans = 69 →
  (maaza + sprite + pepsi) % total_cans = 0 →
  pepsi = 75 :=
by sorry

end NUMINAMATH_CALUDE_pepsi_volume_l3210_321035


namespace NUMINAMATH_CALUDE_product_of_roots_l3210_321098

theorem product_of_roots (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 75*x - 36 = (x - p) * (x - q) * (x - r)) →
  (∃ p q r : ℝ, x^3 - 15*x^2 + 75*x - 36 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 36) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l3210_321098


namespace NUMINAMATH_CALUDE_students_count_l3210_321061

/-- The total number of students in an arrangement of rows -/
def totalStudents (rows : ℕ) (studentsPerRow : ℕ) (lastRowStudents : ℕ) : ℕ :=
  (rows - 1) * studentsPerRow + lastRowStudents

/-- Theorem: Given 8 rows of students, where 7 rows have 6 students each 
    and the last row has 5 students, the total number of students is 47. -/
theorem students_count : totalStudents 8 6 5 = 47 := by
  sorry

end NUMINAMATH_CALUDE_students_count_l3210_321061


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l3210_321086

theorem negation_of_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x + 1| ≥ 0) ↔ (∃ x : ℝ, |x + 1| < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l3210_321086
