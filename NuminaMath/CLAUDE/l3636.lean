import Mathlib

namespace NUMINAMATH_CALUDE_equation_is_linear_l3636_363682

/-- A linear equation in one variable -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Check if an equation is a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The specific equation 3x = 2x -/
def f (x : ℝ) : ℝ := 3 * x - 2 * x

theorem equation_is_linear : is_linear_equation f := by sorry

end NUMINAMATH_CALUDE_equation_is_linear_l3636_363682


namespace NUMINAMATH_CALUDE_base3_addition_theorem_l3636_363684

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else convert (m / 3) ((m % 3) :: acc)
    convert n []

theorem base3_addition_theorem :
  let a := base3ToDecimal [2]
  let b := base3ToDecimal [0, 2, 1]
  let c := base3ToDecimal [1, 2, 2]
  let d := base3ToDecimal [2, 1, 1, 1]
  let e := base3ToDecimal [2, 2, 0, 1]
  let sum := a + b + c + d + e
  decimalToBase3 sum = [1, 0, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base3_addition_theorem_l3636_363684


namespace NUMINAMATH_CALUDE_max_factors_of_power_l3636_363649

-- Define the type for positive integers from 1 to 15
def PositiveIntegerTo15 : Type := {x : ℕ // 1 ≤ x ∧ x ≤ 15}

-- Define the function to count factors
def countFactors (m : ℕ) : ℕ := sorry

-- Define the function to calculate b^n
def powerFunction (b n : PositiveIntegerTo15) : ℕ := sorry

-- Theorem statement
theorem max_factors_of_power :
  ∃ (b n : PositiveIntegerTo15),
    ∀ (b' n' : PositiveIntegerTo15),
      countFactors (powerFunction b n) ≥ countFactors (powerFunction b' n') ∧
      countFactors (powerFunction b n) = 496 :=
sorry

end NUMINAMATH_CALUDE_max_factors_of_power_l3636_363649


namespace NUMINAMATH_CALUDE_black_highest_probability_l3636_363630

-- Define the bag contents
def total_balls : ℕ := 8
def white_balls : ℕ := 1
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2
def black_balls : ℕ := 3

-- Define probabilities
def prob_white : ℚ := white_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_yellow : ℚ := yellow_balls / total_balls
def prob_black : ℚ := black_balls / total_balls

-- Theorem statement
theorem black_highest_probability :
  prob_black > prob_white ∧ 
  prob_black > prob_red ∧ 
  prob_black > prob_yellow :=
sorry

end NUMINAMATH_CALUDE_black_highest_probability_l3636_363630


namespace NUMINAMATH_CALUDE_volleyball_tournament_games_l3636_363613

/-- The number of games played in a volleyball tournament -/
def tournament_games (n : ℕ) (g : ℕ) : ℕ :=
  (n * (n - 1) * g) / 2

/-- Theorem: A volleyball tournament with 10 teams, where each team plays 4 games
    with every other team, has a total of 180 games. -/
theorem volleyball_tournament_games :
  tournament_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_games_l3636_363613


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3636_363692

def team_size : ℕ := 12
def center_players : ℕ := 2
def lineup_size : ℕ := 4

theorem starting_lineup_combinations :
  (center_players) * (team_size - 1) * (team_size - 2) * (team_size - 3) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3636_363692


namespace NUMINAMATH_CALUDE_sqrt_ab_eq_a_plus_b_iff_zero_l3636_363667

theorem sqrt_ab_eq_a_plus_b_iff_zero (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * b) = a + b ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ab_eq_a_plus_b_iff_zero_l3636_363667


namespace NUMINAMATH_CALUDE_equation_solution_l3636_363604

theorem equation_solution (x : ℝ) : x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3636_363604


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3636_363695

/-- The function that constructs the number 5678N from a single digit N -/
def constructNumber (N : ℕ) : ℕ := 5678 * 10 + N

/-- Predicate to check if a natural number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

/-- Theorem stating that 4 is the largest single-digit number N such that 5678N is divisible by 6 -/
theorem largest_digit_divisible_by_six :
  ∀ N : ℕ, isSingleDigit N → N > 4 → ¬(constructNumber N % 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3636_363695


namespace NUMINAMATH_CALUDE_clothes_transport_equals_savings_l3636_363623

/-- Represents Mr. Yadav's monthly financial breakdown --/
structure MonthlyFinances where
  salary : ℝ
  consumable_rate : ℝ
  clothes_transport_rate : ℝ
  savings_rate : ℝ

/-- Calculates the yearly savings based on monthly finances --/
def yearly_savings (m : MonthlyFinances) : ℝ :=
  12 * m.savings_rate * m.salary

/-- Theorem stating that the monthly amount spent on clothes and transport
    is equal to the monthly savings --/
theorem clothes_transport_equals_savings
  (m : MonthlyFinances)
  (h1 : m.consumable_rate = 0.6)
  (h2 : m.clothes_transport_rate = 0.5 * (1 - m.consumable_rate))
  (h3 : m.savings_rate = 1 - m.consumable_rate - m.clothes_transport_rate)
  (h4 : yearly_savings m = 48456) :
  m.clothes_transport_rate * m.salary = m.savings_rate * m.salary :=
by sorry

end NUMINAMATH_CALUDE_clothes_transport_equals_savings_l3636_363623


namespace NUMINAMATH_CALUDE_binary_calculation_proof_l3636_363685

/-- Converts a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ := sorry

/-- Converts a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String := sorry

/-- Binary division (integer division) -/
def binary_div (a b : String) : String := sorry

/-- Binary multiplication -/
def binary_mul (a b : String) : String := sorry

theorem binary_calculation_proof : 
  let a := "1101110"
  let b := "100"
  let c := "1101"
  let result := "10010001"
  binary_mul (binary_div a b) c = result := by sorry

end NUMINAMATH_CALUDE_binary_calculation_proof_l3636_363685


namespace NUMINAMATH_CALUDE_books_loaned_out_l3636_363650

theorem books_loaned_out (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) :
  initial_books = 150 →
  final_books = 100 →
  return_rate = 3/5 →
  (initial_books - final_books : ℚ) / (1 - return_rate) = 125 :=
by sorry

end NUMINAMATH_CALUDE_books_loaned_out_l3636_363650


namespace NUMINAMATH_CALUDE_problem_statement_l3636_363641

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem problem_statement (x : ℝ) 
  (h : deriv f x = 2 * f x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin (2 * x)) = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3636_363641


namespace NUMINAMATH_CALUDE_x_equals_two_l3636_363626

theorem x_equals_two (some_number : ℝ) (h : x + some_number = 3) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_two_l3636_363626


namespace NUMINAMATH_CALUDE_jeans_cost_l3636_363640

theorem jeans_cost (total_cost coat_cost shoe_cost : ℕ) (h1 : total_cost = 110) (h2 : coat_cost = 40) (h3 : shoe_cost = 30) : 
  ∃ (jeans_cost : ℕ), jeans_cost * 2 + coat_cost + shoe_cost = total_cost ∧ jeans_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_l3636_363640


namespace NUMINAMATH_CALUDE_sum_of_k_values_for_distinct_integer_solutions_l3636_363634

theorem sum_of_k_values_for_distinct_integer_solutions : ∃ (S : Finset ℤ), 
  (∀ k ∈ S, ∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 12 = 0 ∧ 3 * y^2 - k * y + 12 = 0) ∧ 
  (∀ k : ℤ, (∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 12 = 0 ∧ 3 * y^2 - k * y + 12 = 0) → k ∈ S) ∧
  (Finset.sum S id = 0) := by
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_for_distinct_integer_solutions_l3636_363634


namespace NUMINAMATH_CALUDE_young_li_age_is_20_l3636_363670

/-- Young Li's age this year -/
def young_li_age : ℕ := 20

/-- Old Li's age this year -/
def old_li_age : ℕ := young_li_age * 5 / 2

theorem young_li_age_is_20 :
  (old_li_age = young_li_age * 5 / 2) ∧
  (old_li_age + 10 = (young_li_age + 10) * 2) →
  young_li_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_young_li_age_is_20_l3636_363670


namespace NUMINAMATH_CALUDE_baseball_players_l3636_363656

/-- Given a club with the following properties:
  * There are 310 people in total
  * 138 people play tennis
  * 94 people play both tennis and baseball
  * 11 people do not play any sport
  Prove that 255 people play baseball -/
theorem baseball_players (total : ℕ) (tennis : ℕ) (both : ℕ) (none : ℕ) 
  (h1 : total = 310)
  (h2 : tennis = 138)
  (h3 : both = 94)
  (h4 : none = 11) :
  total - (tennis - both) - none = 255 := by
  sorry

#eval 310 - (138 - 94) - 11

end NUMINAMATH_CALUDE_baseball_players_l3636_363656


namespace NUMINAMATH_CALUDE_geometric_figure_area_l3636_363669

theorem geometric_figure_area (x : ℝ) : 
  x > 0 →
  (3*x)^2 + (4*x)^2 + (1/2) * (3*x) * (4*x) = 1200 →
  x = Real.sqrt (1200/31) := by
sorry

end NUMINAMATH_CALUDE_geometric_figure_area_l3636_363669


namespace NUMINAMATH_CALUDE_race_time_proof_l3636_363690

/-- In a 1000-meter race, runner A beats runner B by either 25 meters or 10 seconds. -/
theorem race_time_proof (v : ℝ) (t : ℝ) (h1 : v > 0) (h2 : t > 0) : 
  (1000 = v * t ∧ 975 = v * (t + 10)) → t = 400 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l3636_363690


namespace NUMINAMATH_CALUDE_average_price_theorem_l3636_363654

theorem average_price_theorem (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  let p := (2 * a * b) / (a + b)
  a < p ∧ p < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_average_price_theorem_l3636_363654


namespace NUMINAMATH_CALUDE_glass_bowl_purchase_price_l3636_363673

theorem glass_bowl_purchase_price 
  (total_bowls : ℕ) 
  (sold_bowls : ℕ) 
  (selling_price : ℚ) 
  (percentage_gain : ℚ) :
  total_bowls = 118 →
  sold_bowls = 102 →
  selling_price = 15 →
  percentage_gain = 8050847457627118 / 100000000000000000 →
  ∃ (purchase_price : ℚ),
    purchase_price = 12 ∧
    sold_bowls * selling_price - total_bowls * purchase_price = 
      (percentage_gain / 100) * (total_bowls * purchase_price) := by
  sorry

end NUMINAMATH_CALUDE_glass_bowl_purchase_price_l3636_363673


namespace NUMINAMATH_CALUDE_expression_simplification_l3636_363612

theorem expression_simplification (x y : ℝ) :
  (2 * x - (3 * y - (2 * x + 1))) - ((3 * y - (2 * x + 1)) - 2 * x) = 8 * x - 6 * y + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3636_363612


namespace NUMINAMATH_CALUDE_line_through_M_parallel_to_line1_line_through_N_perpendicular_to_line2_l3636_363688

-- Define the points M and N
def M : ℝ × ℝ := (1, -2)
def N : ℝ × ℝ := (2, -3)

-- Define the lines given in the conditions
def line1 (x y : ℝ) : Prop := 2*x - y + 5 = 0
def line2 (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the parallel and perpendicular conditions
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Theorem for the first line
theorem line_through_M_parallel_to_line1 :
  ∃ (a b c : ℝ), 
    (a * M.1 + b * M.2 + c = 0) ∧ 
    (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 2 * x - y - 4 = 0) ∧
    parallel (a / b) 2 :=
sorry

-- Theorem for the second line
theorem line_through_N_perpendicular_to_line2 :
  ∃ (a b c : ℝ), 
    (a * N.1 + b * N.2 + c = 0) ∧ 
    (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 2 * x + y - 1 = 0) ∧
    perpendicular (a / b) (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_line_through_M_parallel_to_line1_line_through_N_perpendicular_to_line2_l3636_363688


namespace NUMINAMATH_CALUDE_min_distance_complex_l3636_363698

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 2) :
  ∃ (min_val : ℝ), min_val = 2*Real.sqrt 2 - 2 ∧
    ∀ (w : ℂ), Complex.abs (w - (1 + 2*I)) = 2 → Complex.abs (w - 3) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_l3636_363698


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3636_363655

-- Define the sets corresponding to p and q
def set_p : Set ℝ := {x | (1 - x^2) / (|x| - 2) < 0}
def set_q : Set ℝ := {x | x^2 + x - 6 > 0}

-- State the theorem
theorem p_necessary_not_sufficient_for_q :
  (set_q ⊆ set_p) ∧ (set_q ≠ set_p) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3636_363655


namespace NUMINAMATH_CALUDE_forty_knocks_to_knicks_l3636_363694

-- Define the units
def Knick : Type := ℚ
def Knack : Type := ℚ
def Knock : Type := ℚ

-- Define the conversion rates
def knicks_to_knacks : ℚ := 3 / 8
def knacks_to_knocks : ℚ := 5 / 4

-- Theorem statement
theorem forty_knocks_to_knicks :
  (40 : ℚ) * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 128 / 3 := by
  sorry

end NUMINAMATH_CALUDE_forty_knocks_to_knicks_l3636_363694


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l3636_363605

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l3636_363605


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3636_363680

theorem arctan_equation_solution :
  ∃ y : ℝ, 2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/y) = π/3 ∧ y = 13.25 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3636_363680


namespace NUMINAMATH_CALUDE_system_solvability_l3636_363606

/-- The set of values for parameter a such that the system has at least one solution -/
def ValidAValues : Set ℝ := {a | a < 0 ∨ a ≥ 2/3}

/-- The system of equations -/
def System (a b x y : ℝ) : Prop :=
  x = |y + a| + 4/a ∧ x^2 + y^2 + 24 + b*(2*y + b) = 10*x

/-- Theorem stating the condition for the existence of a solution -/
theorem system_solvability (a : ℝ) :
  (∃ b x y, System a b x y) ↔ a ∈ ValidAValues :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l3636_363606


namespace NUMINAMATH_CALUDE_circular_garden_max_area_l3636_363603

theorem circular_garden_max_area (fence_length : ℝ) (h : fence_length = 200) :
  let radius := fence_length / (2 * Real.pi)
  let area := Real.pi * radius ^ 2
  area = 10000 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_max_area_l3636_363603


namespace NUMINAMATH_CALUDE_integer_divisor_problem_l3636_363645

theorem integer_divisor_problem :
  ∀ (a b c : ℕ), 1 < a → a < b → b < c →
  (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
  ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_integer_divisor_problem_l3636_363645


namespace NUMINAMATH_CALUDE_bee_flight_count_l3636_363679

/-- Represents the energy content of honey in terms of bee flight distance -/
def honey_energy : ℕ := 7000

/-- Represents the amount of honey available -/
def honey_amount : ℕ := 10

/-- Represents the distance each bee should fly -/
def flight_distance : ℕ := 1

/-- Theorem: Given the energy content of honey and the amount available,
    calculate the number of bees that can fly a specified distance -/
theorem bee_flight_count :
  (honey_energy * honey_amount) / flight_distance = 70000 := by
  sorry

end NUMINAMATH_CALUDE_bee_flight_count_l3636_363679


namespace NUMINAMATH_CALUDE_square_sum_value_l3636_363609

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3636_363609


namespace NUMINAMATH_CALUDE_train_length_l3636_363653

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 63 → time_s = 20 → length_m = speed_kmh * (1000 / 3600) * time_s → length_m = 350 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3636_363653


namespace NUMINAMATH_CALUDE_painted_cells_count_l3636_363646

/-- Represents a rectangular grid with alternating painted rows and columns. -/
structure PaintedGrid where
  rows : ℕ
  cols : ℕ
  unpainted : ℕ

/-- The number of painted cells in a PaintedGrid. -/
def num_painted (grid : PaintedGrid) : ℕ :=
  grid.rows * grid.cols - grid.unpainted

theorem painted_cells_count (grid : PaintedGrid) :
  grid.rows = 5 ∧ grid.cols = 75 ∧ grid.unpainted = 74 →
  num_painted grid = 301 := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_count_l3636_363646


namespace NUMINAMATH_CALUDE_counterexample_exists_l3636_363661

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3636_363661


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l3636_363635

theorem opposite_of_negative_one_fourth :
  -((-1 : ℚ) / 4) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l3636_363635


namespace NUMINAMATH_CALUDE_annas_age_problem_l3636_363600

theorem annas_age_problem :
  ∃! x : ℕ+, 
    (∃ m : ℕ, (x : ℤ) - 4 = m^2) ∧ 
    (∃ n : ℕ, (x : ℤ) + 3 = n^3) ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_annas_age_problem_l3636_363600


namespace NUMINAMATH_CALUDE_brownies_shared_with_guests_l3636_363615

/-- The number of brownies shared with dinner guests -/
def brownies_shared (total : ℕ) (tina_days : ℕ) (husband_days : ℕ) (left : ℕ) : ℕ :=
  total - (2 * tina_days + husband_days) - left

/-- Theorem stating the number of brownies shared with dinner guests -/
theorem brownies_shared_with_guests :
  brownies_shared 24 5 5 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_brownies_shared_with_guests_l3636_363615


namespace NUMINAMATH_CALUDE_point_movement_l3636_363618

/-- The possible final positions of a point that starts 3 units from the origin,
    moves 4 units right, and then 1 unit left. -/
def final_positions : Set ℤ :=
  {0, 6}

/-- The theorem stating the possible final positions of the point. -/
theorem point_movement (A : ℤ) : 
  (abs A = 3) → 
  ((A + 4 - 1) ∈ final_positions) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_l3636_363618


namespace NUMINAMATH_CALUDE_final_turtle_count_l3636_363663

/-- Number of turtle statues on Grandma Molly's lawn after four years -/
def turtle_statues : ℕ :=
  let year1 := 4
  let year2 := year1 * 4
  let year3_before_breakage := year2 + 12
  let year3_after_breakage := year3_before_breakage - 3
  let year4_new_statues := 3 * 2
  year3_after_breakage + year4_new_statues

theorem final_turtle_count : turtle_statues = 31 := by
  sorry

end NUMINAMATH_CALUDE_final_turtle_count_l3636_363663


namespace NUMINAMATH_CALUDE_probability_two_even_toys_l3636_363683

def number_of_toys : ℕ := 21
def number_of_even_toys : ℕ := 10

theorem probability_two_even_toys :
  let p := (number_of_even_toys / number_of_toys) * ((number_of_even_toys - 1) / (number_of_toys - 1))
  p = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_even_toys_l3636_363683


namespace NUMINAMATH_CALUDE_Q_3_volume_l3636_363611

/-- Recursive definition of the volume of Qᵢ -/
def Q_volume : ℕ → ℚ
  | 0 => 1
  | n + 1 => Q_volume n + 4 * 4^n * (1 / 27)^(n + 1)

/-- The volume of Q₃ is 73/81 -/
theorem Q_3_volume : Q_volume 3 = 73 / 81 := by
  sorry

end NUMINAMATH_CALUDE_Q_3_volume_l3636_363611


namespace NUMINAMATH_CALUDE_rhombus_from_equal_triangle_perimeters_l3636_363657

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The perimeter of a triangle defined by three points -/
def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Theorem: If the perimeters of triangles ABO, BCO, CDO, and DAO are equal
    in a convex quadrilateral ABCD where O is the intersection of diagonals,
    then ABCD is a rhombus -/
theorem rhombus_from_equal_triangle_perimeters (q : Quadrilateral) 
  (h_convex : isConvex q) :
  let O := diagonalIntersection q
  (trianglePerimeter q.A q.B O = trianglePerimeter q.B q.C O) ∧
  (trianglePerimeter q.B q.C O = trianglePerimeter q.C q.D O) ∧
  (trianglePerimeter q.C q.D O = trianglePerimeter q.D q.A O) →
  (q.A.x - q.B.x)^2 + (q.A.y - q.B.y)^2 = 
  (q.B.x - q.C.x)^2 + (q.B.y - q.C.y)^2 ∧
  (q.B.x - q.C.x)^2 + (q.B.y - q.C.y)^2 = 
  (q.C.x - q.D.x)^2 + (q.C.y - q.D.y)^2 ∧
  (q.C.x - q.D.x)^2 + (q.C.y - q.D.y)^2 = 
  (q.D.x - q.A.x)^2 + (q.D.y - q.A.y)^2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_from_equal_triangle_perimeters_l3636_363657


namespace NUMINAMATH_CALUDE_fruit_cost_solution_l3636_363633

/-- Given a system of linear equations representing the cost of fruits,
    prove that the solution satisfies the given equations. -/
theorem fruit_cost_solution (x y z : ℝ) : 
  x + 2 * y = 8.9 ∧ 
  2 * z + 3 * y = 23 ∧ 
  3 * z + 4 * x = 30.1 →
  x = 2.5 ∧ y = 3.2 ∧ z = 6.7 := by
sorry

end NUMINAMATH_CALUDE_fruit_cost_solution_l3636_363633


namespace NUMINAMATH_CALUDE_max_value_ahn_operation_l3636_363628

theorem max_value_ahn_operation :
  ∃ (max : ℕ), max = 600 ∧
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 →
  3 * (300 - n) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_ahn_operation_l3636_363628


namespace NUMINAMATH_CALUDE_cymbal_strike_interval_l3636_363619

def beats_between_triangle_strikes : ℕ := 2
def lcm_cymbal_triangle_strikes : ℕ := 14

theorem cymbal_strike_interval :
  ∃ (c : ℕ), c > 0 ∧ Nat.lcm c beats_between_triangle_strikes = lcm_cymbal_triangle_strikes ∧ c = 14 := by
  sorry

end NUMINAMATH_CALUDE_cymbal_strike_interval_l3636_363619


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l3636_363652

theorem farmer_tomatoes (picked : ℕ) (left : ℕ) (h1 : picked = 83) (h2 : left = 14) :
  picked + left = 97 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l3636_363652


namespace NUMINAMATH_CALUDE_fiftyMeterDashIsSuitable_suitableSurveyIsCorrect_l3636_363672

/-- Represents a survey option -/
inductive SurveyOption
  | A
  | B
  | C
  | D

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  requiresPrecision : Bool
  easyToConduct : Bool
  nonDestructive : Bool
  manageableSubjects : Bool

/-- Defines the characteristics of a comprehensive survey method -/
def isComprehensiveSurvey (c : SurveyCharacteristics) : Prop :=
  c.requiresPrecision ∧ c.easyToConduct ∧ c.nonDestructive ∧ c.manageableSubjects

/-- Characteristics of the 50-meter dash survey -/
def fiftyMeterDashSurvey : SurveyCharacteristics :=
  { requiresPrecision := true
    easyToConduct := true
    nonDestructive := true
    manageableSubjects := true }

/-- Theorem stating that the 50-meter dash survey is suitable for a comprehensive survey method -/
theorem fiftyMeterDashIsSuitable : isComprehensiveSurvey fiftyMeterDashSurvey :=
  sorry

/-- Function to determine the suitable survey option -/
def suitableSurveyOption : SurveyOption :=
  SurveyOption.A

/-- Theorem stating that the suitable survey option is correct -/
theorem suitableSurveyIsCorrect : suitableSurveyOption = SurveyOption.A :=
  sorry

end NUMINAMATH_CALUDE_fiftyMeterDashIsSuitable_suitableSurveyIsCorrect_l3636_363672


namespace NUMINAMATH_CALUDE_polar_curve_length_l3636_363689

noncomputable def curve_length (ρ : Real → Real) (φ₁ φ₂ : Real) : Real :=
  ∫ x in φ₁..φ₂, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem polar_curve_length :
  let ρ : Real → Real := fun φ ↦ 2 * (1 - Real.cos φ)
  curve_length ρ (-Real.pi) (-Real.pi/2) = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_length_l3636_363689


namespace NUMINAMATH_CALUDE_percent_relation_l3636_363648

/-- Given that x is p percent of y, prove that p = 100x / y -/
theorem percent_relation (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3636_363648


namespace NUMINAMATH_CALUDE_selection_schemes_with_women_l3636_363639

/-- The number of ways to select 4 individuals from 4 men and 2 women, with at least 1 woman included -/
def selection_schemes (total_men : ℕ) (total_women : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose (total_men + total_women) to_select - Nat.choose total_men to_select

theorem selection_schemes_with_women (total_men : ℕ) (total_women : ℕ) (to_select : ℕ) 
    (h1 : total_men = 4)
    (h2 : total_women = 2)
    (h3 : to_select = 4) :
  selection_schemes total_men total_women to_select = 14 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_with_women_l3636_363639


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l3636_363614

theorem polynomial_functional_equation (a b c d : ℝ) :
  let f (x : ℝ) := a * x^3 + b * x^2 + c * x + d
  (∀ x, f x * f (-x) = f (x^3)) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l3636_363614


namespace NUMINAMATH_CALUDE_karen_savings_l3636_363617

/-- The sum of a geometric series with initial term 2, common ratio 3, and 7 terms -/
def geometric_sum : ℕ → ℚ
| 0 => 0
| n + 1 => 2 * (3^(n+1) - 1) / (3 - 1)

/-- The theorem stating that the sum of the geometric series after 7 days is 2186 -/
theorem karen_savings : geometric_sum 7 = 2186 := by
  sorry

end NUMINAMATH_CALUDE_karen_savings_l3636_363617


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3636_363659

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (y : ℕ), 5 * x = y^2) ∧ 
    (∃ (z : ℕ), 3 * x = z^3) → 
    x ≥ 1125) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3636_363659


namespace NUMINAMATH_CALUDE_water_added_to_tank_l3636_363627

theorem water_added_to_tank (tank_capacity : ℚ) (initial_fraction : ℚ) (final_fraction : ℚ) : 
  tank_capacity = 32 →
  initial_fraction = 3/4 →
  final_fraction = 7/8 →
  final_fraction * tank_capacity - initial_fraction * tank_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l3636_363627


namespace NUMINAMATH_CALUDE_total_earnings_of_three_workers_l3636_363693

/-- The total earnings of three workers given their combined earnings -/
theorem total_earnings_of_three_workers
  (earnings_a : ℕ) (earnings_b : ℕ) (earnings_c : ℕ)
  (h1 : earnings_a + earnings_c = 400)
  (h2 : earnings_b + earnings_c = 300)
  (h3 : earnings_c = 100) :
  earnings_a + earnings_b + earnings_c = 600 :=
by sorry

end NUMINAMATH_CALUDE_total_earnings_of_three_workers_l3636_363693


namespace NUMINAMATH_CALUDE_sin_theta_value_l3636_363636

/-- Definition of determinant for 2x2 matrix -/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: If the determinant of the given matrix is 1/2, then sin θ = ±√3/2 -/
theorem sin_theta_value (θ : ℝ) (h : det (Real.sin (θ/2)) (Real.cos (θ/2)) (Real.cos (3*θ/2)) (Real.sin (3*θ/2)) = 1/2) :
  Real.sin θ = Real.sqrt 3 / 2 ∨ Real.sin θ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3636_363636


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3636_363686

theorem inequality_solution_set : 
  {x : ℝ | (x - 1) / (2 * x + 1) ≤ 0} = Set.Ioo (-1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3636_363686


namespace NUMINAMATH_CALUDE_find_a_l3636_363697

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x + 1/x)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The coefficient of x^k in the expansion of (x + 1/x)^n -/
def coefficient (n k : ℕ) : ℝ := sorry

theorem find_a : ∃ (a : ℝ), 
  (coefficient 6 3 * a + coefficient 6 2) = 30 ∧ 
  ∀ (b : ℝ), (coefficient 6 3 * b + coefficient 6 2) = 30 → b = a :=
sorry

end NUMINAMATH_CALUDE_find_a_l3636_363697


namespace NUMINAMATH_CALUDE_remaining_customers_l3636_363637

theorem remaining_customers (initial : ℕ) (left : ℕ) (remaining : ℕ) : 
  initial = 14 → left = 11 → remaining = initial - left → remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_customers_l3636_363637


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l3636_363658

/-- Proves that given a car's average speed over two hours and its speed in the second hour, 
    we can determine its speed in the first hour. -/
theorem car_speed_first_hour 
  (average_speed : ℝ) 
  (second_hour_speed : ℝ) 
  (h1 : average_speed = 90) 
  (h2 : second_hour_speed = 60) : 
  ∃ (first_hour_speed : ℝ), 
    first_hour_speed = 120 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l3636_363658


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3636_363621

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) :
  (∀ x, x^2 - 16*x + 63 = (x - a)*(x - b)) →
  3*b - a = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3636_363621


namespace NUMINAMATH_CALUDE_ln_increasing_on_positive_reals_l3636_363638

-- Define the open interval (0, +∞)
def openPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem ln_increasing_on_positive_reals :
  StrictMonoOn Real.log openPositiveReals :=
sorry

end NUMINAMATH_CALUDE_ln_increasing_on_positive_reals_l3636_363638


namespace NUMINAMATH_CALUDE_tan_75_degrees_l3636_363610

theorem tan_75_degrees (h1 : 75 = 60 + 15) 
                        (h2 : Real.tan (60 * π / 180) = Real.sqrt 3) 
                        (h3 : Real.tan (15 * π / 180) = 2 - Real.sqrt 3) : 
  Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l3636_363610


namespace NUMINAMATH_CALUDE_prism_triangle_areas_sum_l3636_363625

/-- Represents a rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the sum of areas of all triangles with vertices at corners of the prism -/
def sumTriangleAreas (prism : RectangularPrism) : ℝ :=
  sorry

/-- Represents the result of sumTriangleAreas as m + √n + √p -/
structure AreaSum where
  m : ℤ
  n : ℤ
  p : ℤ

/-- Converts the sum of triangle areas to the AreaSum form -/
def toAreaSum (sum : ℝ) : AreaSum :=
  sorry

theorem prism_triangle_areas_sum (prism : RectangularPrism) 
  (h1 : prism.a = 1) (h2 : prism.b = 1) (h3 : prism.c = 2) : 
  let sum := sumTriangleAreas prism
  let result := toAreaSum sum
  result.m + result.n + result.p = 41 :=
sorry

end NUMINAMATH_CALUDE_prism_triangle_areas_sum_l3636_363625


namespace NUMINAMATH_CALUDE_range_of_expression_l3636_363687

theorem range_of_expression (x y : ℝ) 
  (h : 4 * x^2 - 2 * Real.sqrt 3 * x * y + 4 * y^2 = 13) : 
  10 - 4 * Real.sqrt 3 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 10 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3636_363687


namespace NUMINAMATH_CALUDE_problem_statement_l3636_363601

theorem problem_statement (x y : ℝ) :
  |x + y - 6| + (x - y + 3)^2 = 0 → 3*x - y = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3636_363601


namespace NUMINAMATH_CALUDE_number_equation_l3636_363642

theorem number_equation (x : ℝ) : 100 - x = x + 40 ↔ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3636_363642


namespace NUMINAMATH_CALUDE_cubic_function_min_value_l3636_363691

/-- Given a cubic function f(x) with a known maximum value on [-2, 2],
    prove that its minimum value on the same interval is -37. -/
theorem cubic_function_min_value (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = 2 * x^3 - 6 * x^2 + m) →
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) →
  (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≥ f x) →
  ∃ x ∈ Set.Icc (-2) 2, f x = -37 ∧ ∀ y ∈ Set.Icc (-2) 2, f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_min_value_l3636_363691


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3636_363699

theorem sum_of_four_numbers : 1432 + 3214 + 2143 + 4321 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3636_363699


namespace NUMINAMATH_CALUDE_ratio_of_powers_l3636_363629

theorem ratio_of_powers : (2^17 * 3^19) / 6^18 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_powers_l3636_363629


namespace NUMINAMATH_CALUDE_rectangle_midpoint_distances_theorem_l3636_363647

def rectangle_midpoint_distances : ℝ := by
  -- Define the vertices of the rectangle
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (3, 4)
  let D : ℝ × ℝ := (0, 4)

  -- Define the midpoints of each side
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : ℝ × ℝ := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  -- Calculate distances from A to each midpoint
  let d_AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let d_AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let d_AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  let d_AP := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

  -- Sum of distances
  let total_distance := d_AM + d_AN + d_AO + d_AP

  -- Prove that the total distance equals the expected value
  sorry

theorem rectangle_midpoint_distances_theorem :
  rectangle_midpoint_distances = (3 * Real.sqrt 2) / 2 + Real.sqrt 13 + (Real.sqrt 73) / 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_distances_theorem_l3636_363647


namespace NUMINAMATH_CALUDE_mod_product_equality_l3636_363624

theorem mod_product_equality (m : ℕ) : 
  (256 * 738 ≡ m [ZMOD 75]) → 
  (0 ≤ m ∧ m < 75) → 
  m = 53 := by
sorry

end NUMINAMATH_CALUDE_mod_product_equality_l3636_363624


namespace NUMINAMATH_CALUDE_set_of_values_for_a_l3636_363674

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + a = 0}

theorem set_of_values_for_a (a : ℝ) : 
  (∀ B : Set ℝ, B ⊆ A a → B = ∅ ∨ B = A a) → 
  (a > 1 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_set_of_values_for_a_l3636_363674


namespace NUMINAMATH_CALUDE_sports_league_games_l3636_363631

/-- Calculates the total number of games in a sports league season. -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  (total_teams * (intra_division_games * (teams_per_division - 1) + 
  inter_division_games * teams_per_division)) / 2

/-- Theorem stating the total number of games in the given sports league setup -/
theorem sports_league_games : 
  total_games 16 8 3 1 = 232 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l3636_363631


namespace NUMINAMATH_CALUDE_problem_solution_l3636_363696

theorem problem_solution (x : ℝ) : (0.65 * x = 0.20 * 422.50) → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3636_363696


namespace NUMINAMATH_CALUDE_binary_to_decimal_111_l3636_363602

theorem binary_to_decimal_111 : 
  (1 : ℕ) * 2^0 + (1 : ℕ) * 2^1 + (1 : ℕ) * 2^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_111_l3636_363602


namespace NUMINAMATH_CALUDE_sum_squares_three_halves_l3636_363678

theorem sum_squares_three_halves
  (x y z : ℝ)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (sum_zero : x + y + z = 0)
  (power_equality : x^4 + y^4 + z^4 = x^6 + y^6 + z^6) :
  x^2 + y^2 + z^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_three_halves_l3636_363678


namespace NUMINAMATH_CALUDE_perpendicular_line_through_P_l3636_363620

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x - y + 4 = 0

theorem perpendicular_line_through_P : 
  (∀ x y : ℝ, given_line x y → (∃ m : ℝ, m * (2*x - y) = -1)) ∧ 
  perpendicular_line point_P.1 point_P.2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_P_l3636_363620


namespace NUMINAMATH_CALUDE_sum_of_extrema_l3636_363616

theorem sum_of_extrema (x y : ℝ) (h : 1 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 4) :
  ∃ (min max : ℝ),
    (∀ z w : ℝ, 1 ≤ z^2 + w^2 ∧ z^2 + w^2 ≤ 4 → min ≤ z^2 - z*w + w^2 ∧ z^2 - z*w + w^2 ≤ max) ∧
    min + max = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l3636_363616


namespace NUMINAMATH_CALUDE_marble_distribution_l3636_363664

theorem marble_distribution (total : ℕ) (first second third : ℚ) : 
  total = 78 →
  first = 3 * second + 2 →
  second = third / 2 →
  first + second + third = total →
  (first = 40 ∧ second = 38/3 ∧ third = 76/3) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3636_363664


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l3636_363651

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem base_conversion_theorem :
  let base := 5
  let T := 0
  let P := 1
  let Q := 2
  let R := 3
  let S := 4
  let dividend_base5 := [P, Q, R, S, R, Q, P]
  let divisor_base5 := [Q, R, Q]
  (to_decimal dividend_base5 base = 24336) ∧
  (to_decimal divisor_base5 base = 67) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l3636_363651


namespace NUMINAMATH_CALUDE_triangle_angle_properties_l3636_363644

theorem triangle_angle_properties (α : Real) 
  (h1 : 0 < α ∧ α < π) -- α is an internal angle of a triangle
  (h2 : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 ∧ 
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = -25/7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_properties_l3636_363644


namespace NUMINAMATH_CALUDE_binomial_coefficient_prime_power_bound_l3636_363608

theorem binomial_coefficient_prime_power_bound 
  (p : Nat) (n k α : Nat) (h_prime : Prime p) 
  (h_divides : p ^ α ∣ Nat.choose n k) : 
  p ^ α ≤ n :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_prime_power_bound_l3636_363608


namespace NUMINAMATH_CALUDE_product_remainder_seven_l3636_363643

theorem product_remainder_seven (a b : ℕ) (ha : a = 326) (hb : b = 57) :
  (a * b) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_seven_l3636_363643


namespace NUMINAMATH_CALUDE_rectangle_area_implies_y_l3636_363607

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    if the area of the rectangle is 45 square units and y > 0, then y = 9. -/
theorem rectangle_area_implies_y (y : ℝ) : y > 0 → 5 * y = 45 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_implies_y_l3636_363607


namespace NUMINAMATH_CALUDE_stamps_given_l3636_363675

theorem stamps_given (x : ℕ) (y : ℕ) : 
  (7 * x : ℕ) / (4 * x) = 7 / 4 →  -- Initial ratio
  ((7 * x - y) : ℕ) / (4 * x + y) = 6 / 5 →  -- Final ratio
  (7 * x - y) = (4 * x + y) + 8 →  -- Final difference
  y = 8 := by sorry

end NUMINAMATH_CALUDE_stamps_given_l3636_363675


namespace NUMINAMATH_CALUDE_bookcase_weight_limit_l3636_363660

/-- The maximum weight a bookcase can hold given a collection of items and their weights --/
theorem bookcase_weight_limit (hardcover_count : ℕ) (hardcover_weight : ℚ)
  (textbook_count : ℕ) (textbook_weight : ℚ)
  (knickknack_count : ℕ) (knickknack_weight : ℚ)
  (overweight : ℚ) :
  hardcover_count = 70 →
  hardcover_weight = 1/2 →
  textbook_count = 30 →
  textbook_weight = 2 →
  knickknack_count = 3 →
  knickknack_weight = 6 →
  overweight = 33 →
  (hardcover_count : ℚ) * hardcover_weight +
  (textbook_count : ℚ) * textbook_weight +
  (knickknack_count : ℚ) * knickknack_weight - overweight = 80 :=
by sorry

end NUMINAMATH_CALUDE_bookcase_weight_limit_l3636_363660


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3636_363671

/-- Given a quadratic equation 6x^2 + 5x + q with roots (-5 ± i√323) / 12, q equals 14.5 -/
theorem quadratic_root_value (q : ℝ) : 
  (∀ x : ℂ, 6 * x^2 + 5 * x + q = 0 ↔ x = (-5 + Complex.I * Real.sqrt 323) / 12 ∨ x = (-5 - Complex.I * Real.sqrt 323) / 12) →
  q = 14.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3636_363671


namespace NUMINAMATH_CALUDE_intersection_point_D_l3636_363632

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 1

/-- The normal line equation at point (2, 4) -/
def normal_line (x y : ℝ) : Prop := y = -1/4 * x + 9/2

theorem intersection_point_D :
  let C : ℝ × ℝ := (2, 4)
  let D : ℝ × ℝ := (-2, 5)
  parabola C.1 C.2 →
  parabola D.1 D.2 ∧
  normal_line D.1 D.2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_D_l3636_363632


namespace NUMINAMATH_CALUDE_g_of_neg_three_eq_eight_l3636_363665

/-- Given functions f and g, prove that g(-3) = 8 -/
theorem g_of_neg_three_eq_eight
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 4 * x - 7)
  (hg : ∀ x, g (f x) = 3 * x^2 + 4 * x + 1) :
  g (-3) = 8 := by
sorry

end NUMINAMATH_CALUDE_g_of_neg_three_eq_eight_l3636_363665


namespace NUMINAMATH_CALUDE_value_of_m_l3636_363662

theorem value_of_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x + m
  let g : ℝ → ℝ := λ x => x^2 - 2*x + 2*m + 8
  3 * f 5 = g 5 → m = -22 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l3636_363662


namespace NUMINAMATH_CALUDE_sticker_difference_l3636_363676

/-- Proves the difference in stickers received by Mandy and Justin -/
theorem sticker_difference (initial_stickers : ℕ) 
  (friends : ℕ) (stickers_per_friend : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 72 →
  friends = 3 →
  stickers_per_friend = 4 →
  remaining_stickers = 42 →
  ∃ (mandy_stickers justin_stickers : ℕ),
    mandy_stickers = friends * stickers_per_friend + 2 ∧
    justin_stickers < mandy_stickers ∧
    initial_stickers = remaining_stickers + friends * stickers_per_friend + mandy_stickers + justin_stickers ∧
    mandy_stickers - justin_stickers = 10 :=
by sorry

end NUMINAMATH_CALUDE_sticker_difference_l3636_363676


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3636_363622

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3636_363622


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3636_363668

theorem tangent_point_coordinates (f : ℝ → ℝ) (h : f = λ x ↦ Real.exp x) :
  ∃ (x y : ℝ), x = 1 ∧ y = Real.exp 1 ∧
  (∀ t : ℝ, f t = Real.exp t) ∧
  (∃ m : ℝ, ∀ t : ℝ, y - f x = m * (t - x) ∧ 0 = m * (-x)) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3636_363668


namespace NUMINAMATH_CALUDE_fertilizer_prices_l3636_363677

theorem fertilizer_prices (price_A price_B : ℝ)
  (h1 : price_A = price_B + 100)
  (h2 : 2 * price_A + price_B = 1700) :
  price_A = 600 ∧ price_B = 500 := by
  sorry

end NUMINAMATH_CALUDE_fertilizer_prices_l3636_363677


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_l3636_363681

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define iron as a constant in our universe
variable (iron : U)

-- Theorem statement
theorem iron_conducts_electricity 
  (h1 : ∀ x, Metal x → ConductsElectricity x) 
  (h2 : Metal iron) : 
  ConductsElectricity iron := by
  sorry


end NUMINAMATH_CALUDE_iron_conducts_electricity_l3636_363681


namespace NUMINAMATH_CALUDE_non_decreasing_lists_count_l3636_363666

def number_of_balls : ℕ := 12
def draws : ℕ := 3

def combinations_with_repetition (n r : ℕ) : ℕ :=
  Nat.choose (n + r - 1) r

theorem non_decreasing_lists_count :
  combinations_with_repetition number_of_balls draws = 364 := by
  sorry

end NUMINAMATH_CALUDE_non_decreasing_lists_count_l3636_363666
