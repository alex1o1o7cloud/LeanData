import Mathlib

namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l901_90121

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l901_90121


namespace NUMINAMATH_CALUDE_mr_slinkums_shipment_count_l901_90167

theorem mr_slinkums_shipment_count : ∀ (total : ℕ), 
  (75 : ℚ) / 100 * total = 150 → total = 200 := by
  sorry

end NUMINAMATH_CALUDE_mr_slinkums_shipment_count_l901_90167


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l901_90187

theorem inequality_system_solution_range (x m : ℝ) : 
  ((x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l901_90187


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l901_90100

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l901_90100


namespace NUMINAMATH_CALUDE_inverse_proportionality_l901_90169

/-- Given that α is inversely proportional to β, prove that if α = 4 when β = 9, 
    then α = -1/2 when β = -72 -/
theorem inverse_proportionality (α β : ℝ) (h : ∃ k, ∀ x y, x * y = k → α = x ∧ β = y) :
  (α = 4 ∧ β = 9) → (β = -72 → α = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l901_90169


namespace NUMINAMATH_CALUDE_quadratic_factorization_problem_l901_90127

theorem quadratic_factorization_problem :
  ∀ (a b : ℕ), 
    (∀ x : ℝ, x^2 - 20*x + 96 = (x - a)*(x - b)) →
    a > b →
    2*b - a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_problem_l901_90127


namespace NUMINAMATH_CALUDE_salary_decrease_theorem_l901_90101

/-- Represents the decrease in average salary of all employees per day -/
def salary_decrease (illiterate_count : ℕ) (literate_count : ℕ) (old_wage : ℕ) (new_wage : ℕ) : ℚ :=
  let total_employees := illiterate_count + literate_count
  let wage_decrease := old_wage - new_wage
  let total_decrease := illiterate_count * wage_decrease
  (total_decrease : ℚ) / total_employees

/-- Theorem stating the decrease in average salary of all employees per day -/
theorem salary_decrease_theorem :
  salary_decrease 20 10 25 10 = 10 := by sorry

end NUMINAMATH_CALUDE_salary_decrease_theorem_l901_90101


namespace NUMINAMATH_CALUDE_no_tangent_line_with_slope_three_halves_for_sine_l901_90182

theorem no_tangent_line_with_slope_three_halves_for_sine :
  ¬∃ (x : ℝ), Real.cos x = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_tangent_line_with_slope_three_halves_for_sine_l901_90182


namespace NUMINAMATH_CALUDE_value_of_expression_l901_90170

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : 3*a - 3*b - 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l901_90170


namespace NUMINAMATH_CALUDE_vasyas_number_l901_90115

theorem vasyas_number (n : ℕ) 
  (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : (n / 100) + (n % 10) = 1)
  (h3 : (n / 100) * ((n / 10) % 10) = 4) :
  n = 140 := by sorry

end NUMINAMATH_CALUDE_vasyas_number_l901_90115


namespace NUMINAMATH_CALUDE_journey_proportions_l901_90183

theorem journey_proportions 
  (total_distance : ℝ) 
  (rail_proportion bus_proportion : ℝ) 
  (h1 : rail_proportion > 0)
  (h2 : bus_proportion > 0)
  (h3 : rail_proportion + bus_proportion < 1) :
  ∃ (foot_proportion : ℝ),
    foot_proportion > 0 ∧ 
    rail_proportion + bus_proportion + foot_proportion = 1 := by
  sorry

end NUMINAMATH_CALUDE_journey_proportions_l901_90183


namespace NUMINAMATH_CALUDE_wall_length_with_mirrors_l901_90189

/-- The length of a rectangular wall with specific mirror configurations -/
theorem wall_length_with_mirrors (square_side : ℝ) (circle_diameter : ℝ) (wall_width : ℝ)
  (h_square : square_side = 18)
  (h_circle : circle_diameter = 20)
  (h_width : wall_width = 32)
  (h_combined_area : square_side ^ 2 + π * (circle_diameter / 2) ^ 2 = wall_width * wall_length / 2) :
  wall_length = (324 + 100 * π) / 16 := by
  sorry

#check wall_length_with_mirrors

end NUMINAMATH_CALUDE_wall_length_with_mirrors_l901_90189


namespace NUMINAMATH_CALUDE_starburst_candies_l901_90186

theorem starburst_candies (mm_ratio : ℕ) (starburst_ratio : ℕ) (total_mm : ℕ) : ℕ :=
  let starburst_count := (starburst_ratio * total_mm) / mm_ratio
  by
    sorry

#check starburst_candies 13 8 143 = 88

end NUMINAMATH_CALUDE_starburst_candies_l901_90186


namespace NUMINAMATH_CALUDE_lindas_savings_l901_90104

-- Define the problem parameters
def furniture_ratio : ℚ := 5/8
def tv_ratio : ℚ := 1/4
def tv_discount : ℚ := 15/100
def furniture_discount : ℚ := 10/100
def initial_tv_cost : ℚ := 320
def exchange_rate : ℚ := 11/10

-- Define the theorem
theorem lindas_savings : 
  ∃ (savings : ℚ),
    savings * tv_ratio * (1 - tv_discount) = initial_tv_cost * (1 - tv_discount) ∧
    savings * furniture_ratio * (1 - furniture_discount) * exchange_rate = 
      savings * furniture_ratio * (1 - furniture_discount) ∧
    savings = 1088 :=
sorry

end NUMINAMATH_CALUDE_lindas_savings_l901_90104


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l901_90112

theorem rectangle_measurement_error (x : ℝ) : 
  (((1 + x / 100) * 0.9) = 1.08) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l901_90112


namespace NUMINAMATH_CALUDE_age_sum_is_37_l901_90133

/-- Given the ages of A, B, and C, prove their sum is 37 -/
theorem age_sum_is_37 (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 14) :
  a + b + c = 37 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_is_37_l901_90133


namespace NUMINAMATH_CALUDE_equation_simplification_l901_90134

theorem equation_simplification (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) + 3) * (x - 1) = 1 + 3 * (x - 1) ∧
  3 * x / (1 - x) * (x - 1) = -3 * x ∧
  1 + 3 * (x - 1) = -3 * x :=
sorry

end NUMINAMATH_CALUDE_equation_simplification_l901_90134


namespace NUMINAMATH_CALUDE_expression_equality_l901_90146

theorem expression_equality : 99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1 = 92199816 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l901_90146


namespace NUMINAMATH_CALUDE_prove_postcard_selection_l901_90139

def postcardSelection (typeA : ℕ) (typeB : ℕ) (teachers : ℕ) : Prop :=
  typeA = 2 ∧ typeB = 3 ∧ teachers = 4 →
  (Nat.choose teachers typeA + Nat.choose (teachers - 1) (typeA - 1)) = 10

theorem prove_postcard_selection :
  postcardSelection 2 3 4 :=
by
  sorry

end NUMINAMATH_CALUDE_prove_postcard_selection_l901_90139


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l901_90118

-- Problem 1
theorem problem_1 : (-2)^2 + |-4| - 18 * (-1/3) = 14 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2*(3*a^2*b - 2*a*b^2) - 4*(-a*b^2 + a^2*b) = 2*a^2*b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l901_90118


namespace NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l901_90163

/-- Represents the problem of Patty paying her siblings with cookies to do her chores. -/
structure CookieChoresProblem where
  money_available : ℕ  -- Amount of money Patty has in dollars
  pack_cost : ℕ  -- Cost of one pack of cookies in dollars
  cookies_per_pack : ℕ  -- Number of cookies in each pack
  cookies_per_chore : ℕ  -- Number of cookies given for each chore
  chores_per_week : ℕ  -- Number of chores each kid has per week

/-- Calculates the number of weeks Patty can avoid doing chores. -/
def weeks_without_chores (problem : CookieChoresProblem) : ℕ :=
  let packs_bought := problem.money_available / problem.pack_cost
  let total_cookies := packs_bought * problem.cookies_per_pack
  let cookies_per_week := problem.cookies_per_chore * problem.chores_per_week
  total_cookies / cookies_per_week

/-- Theorem stating that given the problem conditions, Patty can avoid doing chores for 10 weeks. -/
theorem patty_avoids_chores_for_ten_weeks (problem : CookieChoresProblem) 
  (h1 : problem.money_available = 15)
  (h2 : problem.pack_cost = 3)
  (h3 : problem.cookies_per_pack = 24)
  (h4 : problem.cookies_per_chore = 3)
  (h5 : problem.chores_per_week = 4) :
  weeks_without_chores problem = 10 := by
  sorry

#eval weeks_without_chores { 
  money_available := 15, 
  pack_cost := 3, 
  cookies_per_pack := 24, 
  cookies_per_chore := 3, 
  chores_per_week := 4 
}

end NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l901_90163


namespace NUMINAMATH_CALUDE_dvd_book_capacity_l901_90191

theorem dvd_book_capacity (total_capacity : ℕ) (current_count : ℕ) (h1 : total_capacity = 126) (h2 : current_count = 81) :
  total_capacity - current_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_capacity_l901_90191


namespace NUMINAMATH_CALUDE_M_mod_500_l901_90148

/-- A sequence of positive integers whose binary representations have exactly 6 ones -/
def T : ℕ → ℕ :=
  sorry

/-- The 500th term in the sequence T -/
def M : ℕ :=
  T 500

theorem M_mod_500 : M % 500 = 198 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_500_l901_90148


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l901_90190

theorem sqrt_18_div_sqrt_8 : Real.sqrt 18 / Real.sqrt 8 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l901_90190


namespace NUMINAMATH_CALUDE_calculate_expression_l901_90154

theorem calculate_expression : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l901_90154


namespace NUMINAMATH_CALUDE_string_average_length_l901_90153

/-- Given 6 strings where 2 strings have an average length of 70 cm
    and the other 4 strings have an average length of 85 cm,
    prove that the average length of all 6 strings is 80 cm. -/
theorem string_average_length :
  let total_strings : ℕ := 6
  let group1_strings : ℕ := 2
  let group2_strings : ℕ := 4
  let group1_avg_length : ℝ := 70
  let group2_avg_length : ℝ := 85
  (total_strings = group1_strings + group2_strings) →
  (group1_strings * group1_avg_length + group2_strings * group2_avg_length) / total_strings = 80 :=
by sorry

end NUMINAMATH_CALUDE_string_average_length_l901_90153


namespace NUMINAMATH_CALUDE_work_completion_proof_l901_90102

/-- The number of days it takes for person B to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A and B work together -/
def work_together_days : ℝ := 7

/-- The fraction of work left after A and B work together for 7 days -/
def work_left : ℝ := 0.18333333333333335

/-- The number of days it takes for person A to complete the work alone -/
def a_days : ℝ := 15

theorem work_completion_proof :
  (work_together_days * (1 / a_days + 1 / b_days) = 1 - work_left) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l901_90102


namespace NUMINAMATH_CALUDE_fibonacci_square_property_l901_90116

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Proposition: N^2 = 1 + n(N + n) iff (N, n) are consecutive Fibonacci numbers -/
theorem fibonacci_square_property (N n : ℕ) (hN : N > 0) (hn : n > 0) :
  N^2 = 1 + n * (N + n) ↔ ∃ i : ℕ, i > 0 ∧ N = fib (i + 1) ∧ n = fib i :=
sorry

end NUMINAMATH_CALUDE_fibonacci_square_property_l901_90116


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l901_90161

def total_spent : ℝ := 300

def small_doll_discount : ℝ := 2

theorem large_monkey_doll_cost (large_cost : ℝ) 
  (h1 : large_cost > 0)
  (h2 : total_spent / (large_cost - small_doll_discount) = total_spent / large_cost + 25) :
  large_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l901_90161


namespace NUMINAMATH_CALUDE_area_of_triple_square_l901_90172

/-- Given a square (square I) with diagonal length a + b√2, 
    prove that the area of a square (square II) that is three times 
    the area of square I is 3a^2 + 6ab√2 + 6b^2 -/
theorem area_of_triple_square (a b : ℝ) : 
  let diagonal_I := a + b * Real.sqrt 2
  let area_II := 3 * (diagonal_I^2 / 2)
  area_II = 3 * a^2 + 6 * a * b * Real.sqrt 2 + 6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triple_square_l901_90172


namespace NUMINAMATH_CALUDE_bus_cost_proof_l901_90107

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 6.85

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℝ := 9.85

theorem bus_cost_proof : bus_cost = 1.50 := by sorry

end NUMINAMATH_CALUDE_bus_cost_proof_l901_90107


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l901_90131

theorem product_of_sums_and_differences (W X Y Z : ℝ) : 
  W = (Real.sqrt 2025 + Real.sqrt 2024) →
  X = (-Real.sqrt 2025 - Real.sqrt 2024) →
  Y = (Real.sqrt 2025 - Real.sqrt 2024) →
  Z = (Real.sqrt 2024 - Real.sqrt 2025) →
  W * X * Y * Z = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l901_90131


namespace NUMINAMATH_CALUDE_board_zero_condition_l901_90180

/-- Represents a board with positive integers -/
def Board (m n : ℕ) := Fin m → Fin n → ℕ+

/-- Checks if two positions are adjacent on the board -/
def adjacent (m n : ℕ) (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Represents a move on the board -/
structure Move (m n : ℕ) where
  x1 : Fin m
  y1 : Fin n
  x2 : Fin m
  y2 : Fin n
  k : ℤ
  adj : adjacent m n x1.val y1.val x2.val y2.val

/-- Applies a move to the board -/
def applyMove (b : Board m n) (move : Move m n) : Board m n :=
  sorry

/-- Checks if a position is on a black square in chessboard coloring -/
def isBlack (x y : ℕ) : Bool :=
  (x + y) % 2 = 0

/-- Calculates the sum of numbers on black squares -/
def sumBlack (b : Board m n) : ℕ :=
  sorry

/-- Calculates the sum of numbers on white squares -/
def sumWhite (b : Board m n) : ℕ :=
  sorry

/-- Represents a sequence of moves -/
def MoveSequence (m n : ℕ) := List (Move m n)

/-- Applies a sequence of moves to the board -/
def applyMoveSequence (b : Board m n) (moves : MoveSequence m n) : Board m n :=
  sorry

/-- Checks if all numbers on the board are zero -/
def allZero (b : Board m n) : Prop :=
  ∀ x y, (b x y : ℕ) = 0

theorem board_zero_condition (m n : ℕ) :
  ∀ (b : Board m n),
    (∃ (moves : MoveSequence m n), allZero (applyMoveSequence b moves)) ↔
    (sumBlack b = sumWhite b) :=
  sorry

end NUMINAMATH_CALUDE_board_zero_condition_l901_90180


namespace NUMINAMATH_CALUDE_evaluate_expression_l901_90179

theorem evaluate_expression : 3000 * (3000^3001)^2 = 3000^6003 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l901_90179


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l901_90173

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_sum : unitsDigit (42^2 + 25^3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l901_90173


namespace NUMINAMATH_CALUDE_intern_distribution_theorem_l901_90114

def distribute_interns (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem intern_distribution_theorem :
  distribute_interns 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_intern_distribution_theorem_l901_90114


namespace NUMINAMATH_CALUDE_room_length_l901_90145

/-- The length of a room given its width, total paving cost, and paving rate per square meter. -/
theorem room_length (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) 
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 12375)
  (h_rate : rate_per_sqm = 600) : 
  total_cost / rate_per_sqm / width = 5.5 := by
sorry

#eval (12375 / 600 / 3.75 : Float)

end NUMINAMATH_CALUDE_room_length_l901_90145


namespace NUMINAMATH_CALUDE_circumscribing_circle_diameter_l901_90162

theorem circumscribing_circle_diameter (n : ℕ) (r : ℝ) :
  n = 8 ∧ r = 2 →
  let R := (2 * r) / (2 * Real.sin (π / n))
  2 * (R + r) = 2 * (4 / Real.sqrt (2 - Real.sqrt 2) + 2) := by sorry

end NUMINAMATH_CALUDE_circumscribing_circle_diameter_l901_90162


namespace NUMINAMATH_CALUDE_prime_pairs_square_sum_l901_90126

theorem prime_pairs_square_sum (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ n : ℕ, p^2 + 5*p*q + 4*q^2 = n^2) ↔ ((p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11)) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_square_sum_l901_90126


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l901_90175

theorem jason_pokemon_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 676 → cards_bought = 224 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 452 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l901_90175


namespace NUMINAMATH_CALUDE_AC_length_approx_l901_90132

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (q : Quadrilateral) : Prop :=
  let dist := λ p1 p2 : ℝ × ℝ => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 15 ∧ 
  dist q.D q.C = 24 ∧ 
  dist q.A q.D = 9

-- Theorem statement
theorem AC_length_approx (q : Quadrilateral) 
  (h : satisfies_conditions q) : 
  ∃ ε > 0, |dist q.A q.C - 30.7| < ε :=
sorry

#check AC_length_approx

end NUMINAMATH_CALUDE_AC_length_approx_l901_90132


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l901_90193

theorem simplify_sqrt_sum (x : ℝ) :
  Real.sqrt (4 * x^2 - 8 * x + 4) + Real.sqrt (4 * x^2 + 8 * x + 4) = 2 * (|x - 1| + |x + 1|) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l901_90193


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l901_90147

theorem continued_fraction_sum (w x y : ℕ+) :
  (97 : ℚ) / 19 = w + 1 / (x + 1 / y) →
  (w : ℕ) + x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l901_90147


namespace NUMINAMATH_CALUDE_arithmetic_seq_problem_l901_90168

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given conditions for the arithmetic sequence -/
structure ArithSeqConditions (a : ℕ → ℚ) : Prop :=
  (is_arith : ArithmeticSequence a)
  (prod_eq : a 5 * a 7 = 6)
  (sum_eq : a 2 + a 10 = 5)

/-- Theorem statement -/
theorem arithmetic_seq_problem (a : ℕ → ℚ) (h : ArithSeqConditions a) :
  (a 10 - a 6 = 2) ∨ (a 10 - a 6 = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_problem_l901_90168


namespace NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l901_90199

-- First inequality
theorem solution_set_inequality1 (x : ℝ) :
  x ≠ 2 →
  ((x + 1) / (x - 2) ≥ 3) ↔ (2 < x ∧ x ≤ 7/2) :=
sorry

-- Second inequality
theorem solution_set_inequality2 (x a : ℝ) :
  x^2 - a*x - 2*a^2 ≤ 0 ↔
    (a = 0 ∧ x = 0) ∨
    (a > 0 ∧ -a ≤ x ∧ x ≤ 2*a) ∨
    (a < 0 ∧ 2*a ≤ x ∧ x ≤ -a) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l901_90199


namespace NUMINAMATH_CALUDE_equation_solution_l901_90166

theorem equation_solution (a b x : ℝ) : 
  (a * Real.sin x + b) / (b * Real.cos x + a) = (a * Real.cos x + b) / (b * Real.sin x + a) ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 4) ∨ 
  (b = Real.sqrt 2 * a ∧ ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 4) ∨
  (b = -Real.sqrt 2 * a ∧ ∃ k : ℤ, x = (2 * k + 1) * Real.pi) := by
sorry


end NUMINAMATH_CALUDE_equation_solution_l901_90166


namespace NUMINAMATH_CALUDE_peanut_butter_price_increase_peanut_butter_problem_l901_90117

/-- Calculates the new average price of returned peanut butter cans after a price increase -/
theorem peanut_butter_price_increase (initial_avg_price : ℚ) (num_cans : ℕ) 
  (price_increase : ℚ) (num_returned : ℕ) (remaining_avg_price : ℚ) : ℚ :=
  let total_initial_cost := initial_avg_price * num_cans
  let new_price_per_can := initial_avg_price * (1 + price_increase)
  let total_new_cost := new_price_per_can * num_cans
  let num_remaining := num_cans - num_returned
  let total_remaining_cost := remaining_avg_price * num_remaining
  let total_returned_cost := total_new_cost - total_remaining_cost
  let new_avg_returned_price := total_returned_cost / num_returned
  new_avg_returned_price

/-- The new average price of the two returned peanut butter cans is 65.925 cents -/
theorem peanut_butter_problem : 
  peanut_butter_price_increase (36.5 / 100) 6 (15 / 100) 2 (30 / 100) = 65925 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_price_increase_peanut_butter_problem_l901_90117


namespace NUMINAMATH_CALUDE_solve_sandwich_problem_l901_90185

def sandwich_problem (sandwich_cost : ℕ) (paid_amount : ℕ) (change_received : ℕ) : Prop :=
  let spent_amount := paid_amount - change_received
  let num_sandwiches := spent_amount / sandwich_cost
  num_sandwiches = 3

theorem solve_sandwich_problem :
  sandwich_problem 5 20 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_sandwich_problem_l901_90185


namespace NUMINAMATH_CALUDE_initial_amount_proof_l901_90159

/-- Proves that given specific conditions, the initial amount of money is 30000 --/
theorem initial_amount_proof (rate : ℝ) (time : ℝ) (difference : ℝ) : 
  rate = 0.20 →
  time = 2 →
  difference = 723.0000000000146 →
  (fun P : ℝ => P * ((1 + rate / 2) ^ (2 * time) - (1 + rate) ^ time)) difference = 30000 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l901_90159


namespace NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l901_90144

/-- The number of dinners sold on each day of the week --/
structure DinnerSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the dinner sales problem --/
def dinner_problem (sales : DinnerSales) : Prop :=
  sales.monday = 40 ∧
  sales.tuesday = sales.monday + 40 ∧
  sales.thursday = sales.wednesday + 3 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 203

/-- The theorem stating the ratio of Wednesday's sales to Tuesday's sales --/
theorem wednesday_to_tuesday_ratio (sales : DinnerSales) 
  (h : dinner_problem sales) : 
  (sales.wednesday : ℚ) / sales.tuesday = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l901_90144


namespace NUMINAMATH_CALUDE_flow_across_cut_equals_flow_from_single_vertex_l901_90171

-- Define a network
variable (N : Type*) [Fintype N]

-- Define the flow function
variable (f : Set N → Set N → ℝ)

-- Define the set of all vertices
variable (V : Set N)

-- Theorem statement
theorem flow_across_cut_equals_flow_from_single_vertex
  (S : Set N) (s : N) (h_s_in_S : s ∈ S) (h_S_subset_V : S ⊆ V) :
  f S (V \ S) = f {s} V :=
sorry

end NUMINAMATH_CALUDE_flow_across_cut_equals_flow_from_single_vertex_l901_90171


namespace NUMINAMATH_CALUDE_zoo_visitors_l901_90138

def num_cars : ℕ := 3
def people_per_car : ℕ := 21

theorem zoo_visitors : num_cars * people_per_car = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l901_90138


namespace NUMINAMATH_CALUDE_appropriate_word_count_l901_90198

def speech_duration_min : ℝ := 40
def speech_duration_max : ℝ := 50
def speech_rate : ℝ := 160
def word_count : ℕ := 7600

theorem appropriate_word_count : 
  speech_duration_min * speech_rate ≤ word_count ∧ 
  word_count ≤ speech_duration_max * speech_rate := by
  sorry

end NUMINAMATH_CALUDE_appropriate_word_count_l901_90198


namespace NUMINAMATH_CALUDE_coin_count_l901_90160

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 440

theorem coin_count (n : ℕ) : 
  n * (quarter_value + dime_value + nickel_value) = total_value → 
  n = 11 := by sorry

end NUMINAMATH_CALUDE_coin_count_l901_90160


namespace NUMINAMATH_CALUDE_binomial_coeff_n_n_l901_90177

theorem binomial_coeff_n_n (n : ℕ) : (n.choose n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coeff_n_n_l901_90177


namespace NUMINAMATH_CALUDE_agricultural_product_prices_l901_90123

/-- Given two linear equations representing the cost of agricultural products A and B,
    prove that the unique solution for the prices of A and B is (120, 150). -/
theorem agricultural_product_prices (x y : ℚ) : 
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720) → x = 120 ∧ y = 150 := by
  sorry

end NUMINAMATH_CALUDE_agricultural_product_prices_l901_90123


namespace NUMINAMATH_CALUDE_basketball_price_correct_l901_90156

/-- The price of a basketball that satisfies the given conditions -/
def basketball_price : ℚ := 29

/-- The number of basketballs bought by Coach A -/
def basketballs_count : ℕ := 10

/-- The price of each baseball -/
def baseball_price : ℚ := 5/2

/-- The number of baseballs bought by Coach B -/
def baseballs_count : ℕ := 14

/-- The price of the baseball bat -/
def bat_price : ℚ := 18

/-- The difference in spending between Coach A and Coach B -/
def spending_difference : ℚ := 237

theorem basketball_price_correct : 
  basketballs_count * basketball_price = 
  (baseballs_count * baseball_price + bat_price + spending_difference) :=
by sorry

end NUMINAMATH_CALUDE_basketball_price_correct_l901_90156


namespace NUMINAMATH_CALUDE_triangle_third_side_l901_90110

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 8 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c ≠ 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_l901_90110


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l901_90155

/-- A pyramid with a square base and equilateral triangular lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (edge_length : ℝ)
  (touches_base_center : True)
  (touches_apex : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.edge_length ^ 3

/-- The main theorem: volume of the inscribed cube in the given pyramid -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
  (h_base : p.base_side = 2) :
  cube_volume p c = 2 * Real.sqrt 6 / 9 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_inscribed_cube_volume_l901_90155


namespace NUMINAMATH_CALUDE_altered_prism_edges_l901_90135

/-- Represents a rectangular prism that has been altered by truncating vertices and cutting faces diagonally -/
structure AlteredPrism where
  initialEdges : Nat
  initialVertices : Nat
  initialFaces : Nat
  newEdgesPerTruncatedVertex : Nat
  newEdgesPerCutFace : Nat

/-- Calculates the total number of edges in the altered prism -/
def totalEdges (p : AlteredPrism) : Nat :=
  p.initialEdges + 
  (p.initialVertices * p.newEdgesPerTruncatedVertex) + 
  (p.initialFaces * p.newEdgesPerCutFace)

/-- Theorem stating that the altered rectangular prism has 42 edges -/
theorem altered_prism_edges :
  ∀ (p : AlteredPrism),
    p.initialEdges = 12 →
    p.initialVertices = 8 →
    p.initialFaces = 6 →
    p.newEdgesPerTruncatedVertex = 3 →
    p.newEdgesPerCutFace = 1 →
    totalEdges p = 42 := by
  sorry

#check altered_prism_edges

end NUMINAMATH_CALUDE_altered_prism_edges_l901_90135


namespace NUMINAMATH_CALUDE_second_month_sale_is_6927_l901_90164

/-- Calculates the sale amount for the second month given the sales of other months and the average sale --/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the second month is 6927 given the problem conditions --/
theorem second_month_sale_is_6927 :
  calculate_second_month_sale 6435 6855 7230 6562 6191 6700 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_is_6927_l901_90164


namespace NUMINAMATH_CALUDE_eight_coin_flip_probability_l901_90151

theorem eight_coin_flip_probability :
  let n : ℕ := 8
  let p : ℚ := 1 / 2  -- probability of heads for a fair coin
  let prob_seven_heads : ℚ := n.choose (n - 1) * p^(n - 1) * (1 - p)
  let prob_seven_tails : ℚ := n.choose (n - 1) * (1 - p)^(n - 1) * p
  prob_seven_heads + prob_seven_tails = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_eight_coin_flip_probability_l901_90151


namespace NUMINAMATH_CALUDE_green_minus_blue_disks_l901_90120

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green
  | Red

/-- Represents the ratio of disks of each color -/
def colorRatio : Fin 4 → Nat
  | 0 => 3  -- Blue
  | 1 => 7  -- Yellow
  | 2 => 8  -- Green
  | 3 => 4  -- Red

/-- The total number of disks in the bag -/
def totalDisks : Nat := 176

/-- Calculates the number of disks of a given color based on the ratio and total disks -/
def disksOfColor (color : Fin 4) : Nat :=
  (colorRatio color * totalDisks) / (colorRatio 0 + colorRatio 1 + colorRatio 2 + colorRatio 3)

/-- Theorem: There are 40 more green disks than blue disks in the bag -/
theorem green_minus_blue_disks : disksOfColor 2 - disksOfColor 0 = 40 := by
  sorry

end NUMINAMATH_CALUDE_green_minus_blue_disks_l901_90120


namespace NUMINAMATH_CALUDE_train_speed_calculation_l901_90188

theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 250 →
  bridge_length = 150 →
  time = 41.142857142857146 →
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  let speed_kmh := speed_ms * 3.6
  ⌊speed_kmh⌋ = 35 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l901_90188


namespace NUMINAMATH_CALUDE_quadratic_inequality_min_value_l901_90158

/-- Given a quadratic inequality with an empty solution set and a condition on its coefficients,
    prove that a certain expression has a minimum value of 4. -/
theorem quadratic_inequality_min_value (a b c : ℝ) :
  (∀ x, (1/a) * x^2 + b*x + c ≥ 0) →
  a * b > 1 →
  ∀ T, T = 1/(2*(a*b - 1)) + (a*(b + 2*c))/(a*b - 1) →
  T ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_min_value_l901_90158


namespace NUMINAMATH_CALUDE_percentage_difference_l901_90109

theorem percentage_difference : 
  (40 * 80 / 100) - (25 * 4 / 5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l901_90109


namespace NUMINAMATH_CALUDE_range_of_b_l901_90184

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ Real.sqrt (x^2 + y^2) + |x - 4| = 5}

-- Define the point B
def B (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the symmetry condition
def symmetricPoints (b : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 p6 : ℝ × ℝ),
    p1 ∈ C ∧ p2 ∈ C ∧ p3 ∈ C ∧ p4 ∈ C ∧ p5 ∈ C ∧ p6 ∈ C ∧
    p1 ≠ p2 ∧ p3 ≠ p4 ∧ p5 ≠ p6 ∧
    (p1.1 + p2.1) / 2 = b ∧ (p3.1 + p4.1) / 2 = b ∧ (p5.1 + p6.1) / 2 = b

-- Theorem statement
theorem range_of_b :
  ∀ b : ℝ, (∀ p ∈ C, Real.sqrt ((p.1)^2 + (p.2)^2) + |p.1 - 4| = 5) →
            symmetricPoints b →
            2 < b ∧ b < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l901_90184


namespace NUMINAMATH_CALUDE_regression_lines_common_point_l901_90149

-- Define the type for a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the type for a line in 2D space
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Define the theorem
theorem regression_lines_common_point
  (s t : ℝ)
  (l₁ l₂ : Line)
  (h₁ : pointOnLine ⟨s, t⟩ l₁)
  (h₂ : pointOnLine ⟨s, t⟩ l₂) :
  ∃ (p : Point), pointOnLine p l₁ ∧ pointOnLine p l₂ :=
by sorry

end NUMINAMATH_CALUDE_regression_lines_common_point_l901_90149


namespace NUMINAMATH_CALUDE_g_is_even_l901_90129

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by
  sorry

end NUMINAMATH_CALUDE_g_is_even_l901_90129


namespace NUMINAMATH_CALUDE_system_solution_l901_90137

theorem system_solution (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z ≠ 0) (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) (hxy : x + y ≠ 0)
  (eq1 : 1/x + 1/(x+y) = 1/a)
  (eq2 : 1/y + 1/(z+x) = 1/b)
  (eq3 : 1/z + 1/(x+y) = 1/c) :
  x = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(-a + b + c)) ∧
  y = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(a - b + c)) ∧
  z = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(a + b - c)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l901_90137


namespace NUMINAMATH_CALUDE_diameter_equation_l901_90192

/-- Given a circle and a line, if a diameter of the circle intersects with the line at the midpoint
    of the chord cut by the circle, then the equation of the line on which this diameter lies
    is 2x + y - 3 = 0. -/
theorem diameter_equation (x y : ℝ) :
  (∃ (a b : ℝ), (x - 2)^2 + (y + 1)^2 = 16 ∧ x - 2*y + 3 = 0 ∧
   (a - 2)^2 + (b + 1)^2 = 16 ∧
   (x + a)/2 - 2 = 0 ∧ (y + b)/2 + 1 = 0) →
  (2*x + y - 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_diameter_equation_l901_90192


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l901_90106

/-- The area of a square containing four circles, each with a radius of 10 inches
    and touching two sides of the square and two other circles. -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 10) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 1600 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l901_90106


namespace NUMINAMATH_CALUDE_mailman_theorem_l901_90130

def mailman_problem (total_mail : ℕ) (mail_per_block : ℕ) : ℕ :=
  total_mail / mail_per_block

theorem mailman_theorem : 
  mailman_problem 192 48 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mailman_theorem_l901_90130


namespace NUMINAMATH_CALUDE_sum_reciprocal_f_equals_251_385_l901_90140

/-- The function f(n) that returns the integer closest to the cube root of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of 1/f(k) from k=1 to 2023 -/
def sum_reciprocal_f : ℚ :=
  (Finset.range 2023).sum (λ k => 1 / (f (k + 1) : ℚ))

/-- The theorem stating that the sum of 1/f(k) from k=1 to 2023 is equal to 251.385 -/
theorem sum_reciprocal_f_equals_251_385 : sum_reciprocal_f = 251385 / 1000 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_f_equals_251_385_l901_90140


namespace NUMINAMATH_CALUDE_train_length_l901_90122

/-- The length of a train given its speed, bridge length, and crossing time -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 255.03 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 119.97 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l901_90122


namespace NUMINAMATH_CALUDE_triangle_max_area_l901_90128

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.cos C + c * Real.cos A = 3 →
  a^2 + c^2 = 9 + a*c →
  (∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧
    ∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ S) →
  S = (9 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l901_90128


namespace NUMINAMATH_CALUDE_binomial_9_6_l901_90105

theorem binomial_9_6 : Nat.choose 9 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_6_l901_90105


namespace NUMINAMATH_CALUDE_expression_is_integer_l901_90152

theorem expression_is_integer (a b c : ℝ) 
  (h1 : a^2 + b^2 = 2*c^2)
  (h2 : a ≠ b)
  (h3 : c ≠ -a)
  (h4 : c ≠ -b) :
  ∃ n : ℤ, ((a+b+2*c)*(2*a^2-b^2-c^2)) / ((a-b)*(a+c)*(b+c)) = n := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_l901_90152


namespace NUMINAMATH_CALUDE_exponent_sum_equality_l901_90197

theorem exponent_sum_equality : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_equality_l901_90197


namespace NUMINAMATH_CALUDE_rhombus_properties_l901_90174

/-- Properties of a rhombus -/
structure Rhombus where
  /-- The diagonals of a rhombus are perpendicular to each other -/
  diagonals_perpendicular : Prop
  /-- The diagonals of a rhombus bisect each other -/
  diagonals_bisect : Prop

theorem rhombus_properties (R : Rhombus) : 
  (R.diagonals_perpendicular ∨ R.diagonals_bisect) ∧ 
  (R.diagonals_perpendicular ∧ R.diagonals_bisect) ∧ 
  ¬(¬R.diagonals_perpendicular) := by
  sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_rhombus_properties_l901_90174


namespace NUMINAMATH_CALUDE_no_fixed_extreme_points_l901_90142

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 3

/-- The derivative of f -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem: There do not exist real numbers a and b such that f has two distinct extreme points that are also fixed points -/
theorem no_fixed_extreme_points :
  ¬ ∃ (a b x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    f' a b x₁ = 0 ∧
    f' a b x₂ = 0 ∧
    f a b x₁ = x₁ ∧
    f a b x₂ = x₂ := by
  sorry


end NUMINAMATH_CALUDE_no_fixed_extreme_points_l901_90142


namespace NUMINAMATH_CALUDE_cos_420_degrees_l901_90136

theorem cos_420_degrees : Real.cos (420 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_degrees_l901_90136


namespace NUMINAMATH_CALUDE_world_population_scientific_notation_l901_90125

/-- The world's population in billions -/
def world_population : ℝ := 8

/-- Scientific notation representation of a number -/
def scientific_notation (n : ℝ) (base : ℝ) (exponent : ℤ) : Prop :=
  n = base * (10 : ℝ) ^ exponent ∧ 1 ≤ base ∧ base < 10

/-- Theorem: The world population of 8 billion in scientific notation is 8 × 10^9 -/
theorem world_population_scientific_notation :
  scientific_notation (world_population * 1000000000) 8 9 := by
  sorry

end NUMINAMATH_CALUDE_world_population_scientific_notation_l901_90125


namespace NUMINAMATH_CALUDE_ribbon_used_wendy_ribbon_problem_l901_90103

/-- Given the total amount of ribbon and the amount left after wrapping presents,
    prove that the amount used for wrapping is the difference between the two. -/
theorem ribbon_used (total : ℕ) (leftover : ℕ) (h : leftover ≤ total) :
  total - leftover = (total - leftover : ℕ) :=
by sorry

/-- Wendy's ribbon problem -/
theorem wendy_ribbon_problem :
  let total := 84
  let leftover := 38
  total - leftover = 46 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_used_wendy_ribbon_problem_l901_90103


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l901_90181

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * (((-15 - Real.sqrt 145) / 8) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 8) + u = 0) → 
  u = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l901_90181


namespace NUMINAMATH_CALUDE_company_a_bottles_company_a_bottles_proof_l901_90165

/-- Proves that Company A sold 300 bottles given the problem conditions -/
theorem company_a_bottles : ℕ :=
  let company_a_price : ℚ := 4
  let company_b_price : ℚ := 7/2
  let company_b_bottles : ℕ := 350
  let revenue_difference : ℚ := 25
  300

theorem company_a_bottles_proof (company_a_price : ℚ) (company_b_price : ℚ) 
  (company_b_bottles : ℕ) (revenue_difference : ℚ) :
  company_a_price = 4 →
  company_b_price = 7/2 →
  company_b_bottles = 350 →
  revenue_difference = 25 →
  company_a_price * company_a_bottles = 
    company_b_price * company_b_bottles + revenue_difference :=
by sorry

end NUMINAMATH_CALUDE_company_a_bottles_company_a_bottles_proof_l901_90165


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l901_90194

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (m + 3, m - 1)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_m_range :
  ∀ m : ℝ, in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l901_90194


namespace NUMINAMATH_CALUDE_least_integer_square_condition_l901_90196

theorem least_integer_square_condition (x : ℤ) : x^2 = 3*(2*x) + 50 → x ≥ -4 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_condition_l901_90196


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l901_90143

theorem arithmetic_expression_equals_24 :
  (8 * 9 / 6) + 8 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l901_90143


namespace NUMINAMATH_CALUDE_walter_zoo_time_l901_90195

def time_at_zoo (seal_time penguin_factor elephant_time : ℕ) : ℕ :=
  seal_time + (seal_time * penguin_factor) + elephant_time

theorem walter_zoo_time :
  time_at_zoo 13 8 13 = 130 :=
by sorry

end NUMINAMATH_CALUDE_walter_zoo_time_l901_90195


namespace NUMINAMATH_CALUDE_pages_with_text_l901_90113

/-- Given a book with the following properties:
  * It has 98 pages in total
  * Half of the pages are filled with images
  * 11 pages are for introduction
  * The remaining pages are equally split between blank and text
  Prove that the number of pages with text is 19 -/
theorem pages_with_text (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : 
  total_pages = 98 →
  image_pages = total_pages / 2 →
  intro_pages = 11 →
  (total_pages - image_pages - intro_pages) % 2 = 0 →
  (total_pages - image_pages - intro_pages) / 2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_pages_with_text_l901_90113


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l901_90108

-- Define the triangle vertices
def P : ℝ × ℝ := (-8, 5)
def Q : ℝ × ℝ := (-15, -19)
def R : ℝ × ℝ := (1, -7)

-- Define the equation of the angle bisector
def angle_bisector_equation (a c : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + c = 0

-- Theorem statement
theorem angle_bisector_sum (a c : ℝ) :
  (∃ x y, angle_bisector_equation a c x y ∧
          (x, y) ≠ P ∧
          (∃ t : ℝ, (1 - t) • P + t • Q = (x, y) ∨ (1 - t) • P + t • R = (x, y))) →
  a + c = 89 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l901_90108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l901_90157

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement to be proved -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 6*(a 3) + 8 = 0 →
  (a 15)^2 - 6*(a 15) + 8 = 0 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l901_90157


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l901_90111

theorem red_shirt_pairs (total_students : ℕ) (green_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) 
  (h1 : total_students = 144)
  (h2 : green_students = 65)
  (h3 : red_students = 79)
  (h4 : total_pairs = 72)
  (h5 : green_green_pairs = 27)
  (h6 : total_students = green_students + red_students)
  (h7 : total_pairs * 2 = total_students) :
  ∃ red_red_pairs : ℕ, red_red_pairs = 34 ∧ 
    red_red_pairs + green_green_pairs + (green_students - 2 * green_green_pairs) = total_pairs :=
by
  sorry


end NUMINAMATH_CALUDE_red_shirt_pairs_l901_90111


namespace NUMINAMATH_CALUDE_total_weight_of_hay_bales_l901_90150

/-- Calculates the total weight of hay bales in a barn after adding new bales -/
theorem total_weight_of_hay_bales
  (initial_bales : ℕ)
  (initial_weight : ℕ)
  (total_bales : ℕ)
  (new_weight : ℕ)
  (h1 : initial_bales = 73)
  (h2 : initial_weight = 45)
  (h3 : total_bales = 96)
  (h4 : new_weight = 50)
  (h5 : total_bales > initial_bales) :
  initial_bales * initial_weight + (total_bales - initial_bales) * new_weight = 4435 :=
by sorry

#check total_weight_of_hay_bales

end NUMINAMATH_CALUDE_total_weight_of_hay_bales_l901_90150


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l901_90176

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (A = 1/3 ∧ B = 2/3 ∧ C = 1/3) ∧
    (∀ x : ℚ, x ≠ -2 ∧ x^2 + x + 1 ≠ 0 →
      (x + 1)^2 / ((x + 2) * (x^2 + x + 1)) =
      A / (x + 2) + (B * x + C) / (x^2 + x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l901_90176


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_l901_90119

theorem tom_fruit_purchase (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) :
  apple_quantity = 8 →
  apple_rate = 70 →
  mango_quantity = 9 →
  mango_rate = 75 →
  apple_quantity * apple_rate + mango_quantity * mango_rate = 1235 := by
  sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_l901_90119


namespace NUMINAMATH_CALUDE_subtracted_value_problem_solution_l901_90124

theorem subtracted_value (chosen_number : ℕ) (final_answer : ℕ) : ℕ :=
  let divided_result := chosen_number / 8
  divided_result - final_answer

theorem problem_solution :
  subtracted_value 1376 12 = 160 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_problem_solution_l901_90124


namespace NUMINAMATH_CALUDE_rug_area_theorem_l901_90178

/-- Given three overlapping rugs, prove their combined area is 212 square meters -/
theorem rug_area_theorem (total_covered_area single_layer_area double_layer_area triple_layer_area : ℝ) :
  total_covered_area = 140 →
  double_layer_area = 24 →
  triple_layer_area = 24 →
  single_layer_area = total_covered_area - double_layer_area - triple_layer_area →
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = 212 :=
by sorry

end NUMINAMATH_CALUDE_rug_area_theorem_l901_90178


namespace NUMINAMATH_CALUDE_acid_solution_replacement_l901_90141

/-- Proves that the fraction of original 50% acid solution replaced with 20% acid solution to obtain a 35% acid solution is 0.5 -/
theorem acid_solution_replacement (V : ℝ) (h : V > 0) :
  let original_concentration : ℝ := 0.5
  let replacement_concentration : ℝ := 0.2
  let final_concentration : ℝ := 0.35
  let x : ℝ := (original_concentration - final_concentration) / (original_concentration - replacement_concentration)
  x = 0.5 := by sorry

end NUMINAMATH_CALUDE_acid_solution_replacement_l901_90141
