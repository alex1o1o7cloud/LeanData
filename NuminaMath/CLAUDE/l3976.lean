import Mathlib

namespace NUMINAMATH_CALUDE_twentieth_common_number_l3976_397610

/-- The mth term of the first sequence -/
def a (m : ℕ) : ℕ := 4 * m - 1

/-- The nth term of the second sequence -/
def b (n : ℕ) : ℕ := 3 * n + 2

/-- The kth common number between the two sequences -/
def common_number (k : ℕ) : ℕ := 12 * k - 1

theorem twentieth_common_number :
  ∃ m n : ℕ, a m = b n ∧ a m = common_number 20 ∧ common_number 20 = 239 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_common_number_l3976_397610


namespace NUMINAMATH_CALUDE_smallest_n_with_non_decimal_digit_in_g_l3976_397694

/-- Sum of digits in base-three representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-six representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- Check if a number in base-twelve contains a digit not in {0, 1, ..., 9} -/
def has_non_decimal_digit (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem smallest_n_with_non_decimal_digit_in_g : 
  (∃ n : ℕ, n > 0 ∧ has_non_decimal_digit (g n)) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < 32 → ¬has_non_decimal_digit (g m)) ∧
  has_non_decimal_digit (g 32) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_non_decimal_digit_in_g_l3976_397694


namespace NUMINAMATH_CALUDE_smallest_x_value_l3976_397647

theorem smallest_x_value (x : ℚ) : 
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) → x ≥ -7/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3976_397647


namespace NUMINAMATH_CALUDE_triangle_inequality_l3976_397699

/-- Given a triangle with side lengths a, b, c and area T, 
    prove that a^2 + b^2 + c^2 ≥ 4√3 T, 
    with equality if and only if the triangle is equilateral -/
theorem triangle_inequality (a b c T : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : T > 0)
  (h_T : T = Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * T ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * T ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3976_397699


namespace NUMINAMATH_CALUDE_complex_number_operations_l3976_397629

theorem complex_number_operations (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 2 - 3*I) 
  (hz₂ : z₂ = (15 - 5*I) / (2 + I^2)) : 
  (z₁ - z₂ = -13 + 2*I) ∧ (z₁ * z₂ = 15 - 55*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_operations_l3976_397629


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3976_397685

/-- A circle in the Cartesian plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Symmetry of a circle with respect to a line -/
def isSymmetric (c : Circle) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem circle_symmetry_line (a : ℝ) :
  let c : Circle := { equation := fun x y => x^2 + y^2 - 4*x - 8*y + 19 = 0 }
  let l : Line := { equation := fun x y => x + 2*y - a = 0 }
  isSymmetric c l → a = 10 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3976_397685


namespace NUMINAMATH_CALUDE_second_number_is_204_l3976_397673

def number_list : List ℕ := [201, 204, 205, 206, 209, 209, 210, 212, 212]

theorem second_number_is_204 : number_list[1] = 204 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_204_l3976_397673


namespace NUMINAMATH_CALUDE_courtyard_length_l3976_397639

theorem courtyard_length 
  (breadth : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℝ) 
  (h1 : breadth = 12)
  (h2 : brick_length = 0.15)
  (h3 : brick_width = 0.13)
  (h4 : num_bricks = 11076.923076923076) :
  (num_bricks * brick_length * brick_width) / breadth = 18 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l3976_397639


namespace NUMINAMATH_CALUDE_monday_rainfall_value_l3976_397626

/-- The rainfall recorded over three days in centimeters -/
def total_rainfall : ℝ := 0.6666666666666666

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℝ := 0.4166666666666667

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℝ := 0.08333333333333333

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℝ := total_rainfall - (tuesday_rainfall + wednesday_rainfall)

theorem monday_rainfall_value : monday_rainfall = 0.16666666666666663 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_value_l3976_397626


namespace NUMINAMATH_CALUDE_smallest_candy_count_l3976_397657

theorem smallest_candy_count : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0) → False) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l3976_397657


namespace NUMINAMATH_CALUDE_smiley_red_smile_l3976_397686

def tulip_smiley (red_smile : ℕ) : Prop :=
  let red_eyes : ℕ := 8 * 2
  let yellow_background : ℕ := 9 * red_smile
  red_eyes + red_smile + yellow_background = 196

theorem smiley_red_smile :
  ∃ (red_smile : ℕ), tulip_smiley red_smile ∧ red_smile = 18 :=
by sorry

end NUMINAMATH_CALUDE_smiley_red_smile_l3976_397686


namespace NUMINAMATH_CALUDE_kishore_savings_percentage_l3976_397683

-- Define Mr. Kishore's expenses and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 2500
def savings : ℕ := 2000

-- Define total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define total salary
def total_salary : ℕ := total_expenses + savings

-- Theorem to prove
theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_kishore_savings_percentage_l3976_397683


namespace NUMINAMATH_CALUDE_car_average_speed_l3976_397652

/-- The average speed of a car given its speeds for two hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h : speed1 = 145 ∧ speed2 = 60) :
  (speed1 + speed2) / 2 = 102.5 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l3976_397652


namespace NUMINAMATH_CALUDE_solve_inequality_l3976_397621

theorem solve_inequality (x : ℝ) : 
  (x + 5) / 2 - 1 < (3 * x + 2) / 2 ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_solve_inequality_l3976_397621


namespace NUMINAMATH_CALUDE_termites_count_workers_composition_l3976_397628

/-- The number of termites in the construction project -/
def num_termites : ℕ := 861 - 239

/-- The total number of workers in the construction project -/
def total_workers : ℕ := 861

/-- The number of monkeys in the construction project -/
def num_monkeys : ℕ := 239

/-- Theorem stating that the number of termites is 622 -/
theorem termites_count : num_termites = 622 := by
  sorry

/-- Theorem stating that the total number of workers is the sum of monkeys and termites -/
theorem workers_composition : total_workers = num_monkeys + num_termites := by
  sorry

end NUMINAMATH_CALUDE_termites_count_workers_composition_l3976_397628


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_theorem_l3976_397697

/-- Represents the financial data of a person -/
structure FinancialData where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings -/
def calculateExpenditure (data : FinancialData) : ℕ :=
  data.income - data.savings

/-- Calculates the ratio of income to expenditure -/
def incomeToExpenditureRatio (data : FinancialData) : Rat :=
  data.income / (calculateExpenditure data)

/-- Theorem stating that for a person with an income of 20000 and savings of 4000,
    the ratio of income to expenditure is 5/4 -/
theorem income_expenditure_ratio_theorem (data : FinancialData)
    (h1 : data.income = 20000)
    (h2 : data.savings = 4000) :
    incomeToExpenditureRatio data = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_income_expenditure_ratio_theorem_l3976_397697


namespace NUMINAMATH_CALUDE_infinitely_many_m_for_binomial_equality_l3976_397605

theorem infinitely_many_m_for_binomial_equality :
  ∀ n : ℕ, n ≥ 4 →
  ∃ m : ℕ, m ≥ 2 ∧
    m = (n^2 - 3*n + 2) / 2 ∧
    Nat.choose m 2 = 3 * Nat.choose n 4 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_m_for_binomial_equality_l3976_397605


namespace NUMINAMATH_CALUDE_apartment_keys_theorem_l3976_397693

/-- The number of keys needed for apartment complexes -/
def keys_needed (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_apartment

/-- Theorem: Given two apartment complexes with 12 apartments each, 
    and requiring 3 keys per apartment, the total number of keys needed is 72 -/
theorem apartment_keys_theorem :
  keys_needed 2 12 3 = 72 := by
  sorry


end NUMINAMATH_CALUDE_apartment_keys_theorem_l3976_397693


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_to_geometric_l3976_397689

/-- Represents a geometric progression with first term a and common ratio q -/
structure GeometricProgression (α : Type*) [Field α] where
  a : α
  q : α

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  2 * y = x + z

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  y * y = x * z

theorem geometric_to_arithmetic_to_geometric 
  {α : Type*} [Field α] (gp : GeometricProgression α) :
  is_arithmetic_progression gp.a (gp.a * gp.q + 2) (gp.a * gp.q^2) ∧
  is_geometric_progression gp.a (gp.a * gp.q + 2) (gp.a * gp.q^2 + 9) →
  (gp.a = 64 ∧ gp.q = 5/4) ∨ (gp.a = 64/25 ∧ gp.q = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_to_geometric_l3976_397689


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3976_397636

/-- Given a principal amount and an interest rate, proves that if the simple interest
    for 2 years is $600 and the compound interest for 2 years is $609,
    then the interest rate is 3% per annum. -/
theorem interest_rate_calculation (P r : ℝ) : 
  P * r * 2 = 600 →
  P * ((1 + r)^2 - 1) = 609 →
  r = 0.03 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3976_397636


namespace NUMINAMATH_CALUDE_wire_cut_problem_l3976_397681

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 90 →
  ratio = 2 / 7 →
  shorter_length = ratio * (total_length - shorter_length) →
  shorter_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_problem_l3976_397681


namespace NUMINAMATH_CALUDE_y1_value_l3976_397661

theorem y1_value (y1 y2 y3 : ℝ) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1) 
  (h2 : (1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1/9) : 
  y1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_y1_value_l3976_397661


namespace NUMINAMATH_CALUDE_paint_fraction_first_week_l3976_397617

/-- Proves that the fraction of paint used in the first week is 1/9 -/
theorem paint_fraction_first_week (total_paint : ℝ) (paint_used : ℝ) 
  (h1 : total_paint = 360)
  (h2 : paint_used = 104)
  (h3 : ∀ f : ℝ, paint_used = f * total_paint + 1/5 * (total_paint - f * total_paint)) :
  ∃ f : ℝ, f = 1/9 ∧ paint_used = f * total_paint + 1/5 * (total_paint - f * total_paint) :=
by sorry

end NUMINAMATH_CALUDE_paint_fraction_first_week_l3976_397617


namespace NUMINAMATH_CALUDE_equation_solution_l3976_397633

theorem equation_solution : 
  ∃ y : ℚ, 3 * (4 * y - 5) + 1 = -3 * (2 - 5 * y) ∧ y = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3976_397633


namespace NUMINAMATH_CALUDE_f_g_properties_l3976_397644

/-- The absolute value function -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|

/-- The function g defined in terms of f -/
def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x - f m (x + m)

/-- The theorem stating the properties of f and g -/
theorem f_g_properties :
  ∃ (m : ℝ), m > 0 ∧
  (∀ x, g m x ≥ -1) ∧
  (∃ x, g m x = -1) ∧
  m = 1 ∧
  ∀ (a b : ℝ), |a| < m → |b| < m → a ≠ 0 → f m (a * b) > |a| * f m (b / a) :=
sorry

end NUMINAMATH_CALUDE_f_g_properties_l3976_397644


namespace NUMINAMATH_CALUDE_jian_has_second_most_l3976_397670

-- Define the number of notebooks for each person
def jian_notebooks : ℕ := 3
def doyun_notebooks : ℕ := 5
def siu_notebooks : ℕ := 2

-- Define a function to determine if a person has the second most notebooks
def has_second_most (x y z : ℕ) : Prop :=
  (x > y ∧ x < z) ∨ (x > z ∧ x < y)

-- Theorem statement
theorem jian_has_second_most :
  has_second_most jian_notebooks siu_notebooks doyun_notebooks :=
sorry

end NUMINAMATH_CALUDE_jian_has_second_most_l3976_397670


namespace NUMINAMATH_CALUDE_number_equals_five_times_difference_l3976_397651

theorem number_equals_five_times_difference : ∃! x : ℝ, x = 5 * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_number_equals_five_times_difference_l3976_397651


namespace NUMINAMATH_CALUDE_cody_game_count_l3976_397659

def final_game_count (initial_games : ℕ) (games_to_jake : ℕ) (games_to_sarah : ℕ) (new_games : ℕ) : ℕ :=
  initial_games - (games_to_jake + games_to_sarah) + new_games

theorem cody_game_count :
  final_game_count 9 4 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cody_game_count_l3976_397659


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3976_397698

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3976_397698


namespace NUMINAMATH_CALUDE_exists_sequence_to_1981_no_sequence_to_1982_l3976_397611

-- Define the machine operations
def multiply_by_3 (n : ℕ) : ℕ := 3 * n
def add_4 (n : ℕ) : ℕ := n + 4

-- Define a sequence of operations
inductive Operation
| Mult3 : Operation
| Add4 : Operation

def apply_operation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Mult3 => multiply_by_3 n
  | Operation.Add4 => add_4 n

def apply_sequence (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl apply_operation start

-- Theorem statements
theorem exists_sequence_to_1981 :
  ∃ (ops : List Operation), apply_sequence 1 ops = 1981 :=
sorry

theorem no_sequence_to_1982 :
  ¬∃ (ops : List Operation), apply_sequence 1 ops = 1982 :=
sorry

end NUMINAMATH_CALUDE_exists_sequence_to_1981_no_sequence_to_1982_l3976_397611


namespace NUMINAMATH_CALUDE_units_digit_problem_l3976_397634

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 7 ∧ n = (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) + 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3976_397634


namespace NUMINAMATH_CALUDE_binary_101110_to_octal_56_l3976_397692

def binary_to_octal (b : List Bool) : Nat :=
  let binary_to_decimal := b.foldl (λ acc x => 2 * acc + if x then 1 else 0) 0
  let decimal_to_octal := binary_to_decimal.digits 8
  decimal_to_octal.foldl (λ acc x => 10 * acc + x) 0

theorem binary_101110_to_octal_56 :
  binary_to_octal [true, false, true, true, true, false] = 56 := by
  sorry

end NUMINAMATH_CALUDE_binary_101110_to_octal_56_l3976_397692


namespace NUMINAMATH_CALUDE_unique_c_value_l3976_397680

theorem unique_c_value (c : ℝ) : c + ⌊c⌋ = 23.2 → c = 11.7 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l3976_397680


namespace NUMINAMATH_CALUDE_problem_solution_l3976_397696

theorem problem_solution (x : ℝ) (h : x - Real.sqrt (x^2 - 4) + 1 / (x + Real.sqrt (x^2 - 4)) = 10) :
  x^2 - Real.sqrt (x^4 - 16) + 1 / (x^2 - Real.sqrt (x^4 - 16)) = 237/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3976_397696


namespace NUMINAMATH_CALUDE_all_hungarian_teams_face_foreign_l3976_397649

-- Define the total number of teams
def total_teams : ℕ := 8

-- Define the number of Hungarian teams
def hungarian_teams : ℕ := 3

-- Define the number of foreign teams
def foreign_teams : ℕ := total_teams - hungarian_teams

-- Define the probability of all Hungarian teams facing foreign opponents
def prob_all_hungarian_foreign : ℚ := 4/7

-- Theorem statement
theorem all_hungarian_teams_face_foreign :
  (foreign_teams.choose hungarian_teams * hungarian_teams.factorial) / 
  (total_teams.choose 2 * (total_teams / 2).factorial) = prob_all_hungarian_foreign := by
  sorry

end NUMINAMATH_CALUDE_all_hungarian_teams_face_foreign_l3976_397649


namespace NUMINAMATH_CALUDE_geometric_sequence_and_parabola_vertex_l3976_397643

/-- Given that a, b, c, and d form a geometric sequence, and the vertex of the curve y = x^2 - 2x + 3 is (b, c), then ad = 2 -/
theorem geometric_sequence_and_parabola_vertex (a b c d : ℝ) : 
  (∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, x^2 - 2*x + 3 ≥ c) →  -- vertex condition
  (b^2 - 2*b + 3 = c) →  -- vertex condition
  a * d = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_parabola_vertex_l3976_397643


namespace NUMINAMATH_CALUDE_square_is_quadratic_and_power_l3976_397684

/-- A function f: ℝ → ℝ is a power function if there exists a real number a such that f(x) = x^a for all x in the domain of f. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x, f x = x ^ a

/-- A function f: ℝ → ℝ is a quadratic function if there exist real numbers a, b, and c with a ≠ 0 such that f(x) = ax^2 + bx + c for all x in ℝ. -/
def IsQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 is both a quadratic function and a power function. -/
theorem square_is_quadratic_and_power :
  let f : ℝ → ℝ := fun x ↦ x^2
  IsQuadraticFunction f ∧ IsPowerFunction f := by
  sorry

end NUMINAMATH_CALUDE_square_is_quadratic_and_power_l3976_397684


namespace NUMINAMATH_CALUDE_jack_jogging_speed_l3976_397687

-- Define the given conditions
def melt_time : ℚ := 10 / 60  -- 10 minutes converted to hours
def num_blocks : ℕ := 16
def block_length : ℚ := 1 / 8  -- in miles

-- Define the total distance
def total_distance : ℚ := num_blocks * block_length

-- Define the required speed
def required_speed : ℚ := total_distance / melt_time

-- Theorem statement
theorem jack_jogging_speed :
  required_speed = 12 := by sorry

end NUMINAMATH_CALUDE_jack_jogging_speed_l3976_397687


namespace NUMINAMATH_CALUDE_part_i_part_ii_l3976_397667

-- Define propositions P and Q
def P (m : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, -x^2 + 3*m - 1 ≤ 0

def Q (m a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, m - a*x ≤ 0

-- Part (i)
theorem part_i (m : ℝ) : 
  (¬(P m) ∧ ¬(Q m 1) ∧ (P m ∨ Q m 1)) → (1/3 < m ∧ m ≤ 1) :=
sorry

-- Part (ii)
theorem part_ii (m a : ℝ) :
  ((P m → Q m a) ∧ ¬(Q m a → P m)) → (a ≥ 1/3 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l3976_397667


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3976_397607

open Real

theorem trig_equation_solution (n : ℤ) : 
  let x : ℝ := π / 6 * (3 * ↑n + 1)
  tan (2 * x) * sin (2 * x) - 3 * sqrt 3 * (1 / tan (2 * x)) * cos (2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3976_397607


namespace NUMINAMATH_CALUDE_bug_return_probability_l3976_397635

/-- Probability of the bug being at the starting corner after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting corner on its eighth move -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3976_397635


namespace NUMINAMATH_CALUDE_problem_solution_l3976_397691

theorem problem_solution : 
  ((2023 - Real.sqrt 5) ^ 0 - 2 + abs (Real.sqrt 3 - 1) = Real.sqrt 3 - 2) ∧
  ((Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) + (Real.sqrt 15 * Real.sqrt 3) / Real.sqrt 5 = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3976_397691


namespace NUMINAMATH_CALUDE_base_27_to_3_conversion_l3976_397655

/-- Converts a single digit from base 27 to its three-digit representation in base 3 -/
def convert_digit_27_to_3 (d : Nat) : Nat × Nat × Nat :=
  (d / 9, (d % 9) / 3, d % 3)

/-- Converts a number from base 27 to base 3 -/
def convert_27_to_3 (n : Nat) : List Nat :=
  let digits := n.digits 27
  List.join (digits.map (fun d => let (a, b, c) := convert_digit_27_to_3 d; [a, b, c]))

theorem base_27_to_3_conversion :
  convert_27_to_3 652 = [0, 2, 0, 0, 1, 2, 0, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_27_to_3_conversion_l3976_397655


namespace NUMINAMATH_CALUDE_puzzle_min_cost_l3976_397631

/-- Represents the cost structure and purchase requirement for puzzles -/
structure PuzzlePurchase where
  single_cost : ℕ  -- Cost of a single puzzle
  box_cost : ℕ    -- Cost of a box of puzzles
  box_size : ℕ    -- Number of puzzles in a box
  required : ℕ    -- Number of puzzles required

/-- Calculates the minimum cost for purchasing the required number of puzzles -/
def minCost (p : PuzzlePurchase) : ℕ :=
  let boxes := p.required / p.box_size
  let singles := p.required % p.box_size
  boxes * p.box_cost + singles * p.single_cost

/-- Theorem stating that the minimum cost for 25 puzzles is $210 -/
theorem puzzle_min_cost :
  let p : PuzzlePurchase := {
    single_cost := 10,
    box_cost := 50,
    box_size := 6,
    required := 25
  }
  minCost p = 210 := by
  sorry


end NUMINAMATH_CALUDE_puzzle_min_cost_l3976_397631


namespace NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l3976_397690

theorem no_distinct_naturals_satisfying_equation :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + 1 : ℚ) / a = ((b + 1 : ℚ) / b + (c + 1 : ℚ) / c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l3976_397690


namespace NUMINAMATH_CALUDE_first_half_total_score_l3976_397619

/-- Represents the score of a team in a basketball game -/
structure Score where
  quarter1 : ℚ
  quarter2 : ℚ
  quarter3 : ℚ
  quarter4 : ℚ

/-- The Eagles' score -/
def eagles : Score :=
  { quarter1 := 1/2,
    quarter2 := 1/2 * 2,
    quarter3 := 1/2 * 2^2,
    quarter4 := 1/2 * 2^3 }

/-- The Tigers' score -/
def tigers : Score :=
  { quarter1 := 5,
    quarter2 := 5,
    quarter3 := 5,
    quarter4 := 5 }

/-- Total score for a team -/
def totalScore (s : Score) : ℚ :=
  s.quarter1 + s.quarter2 + s.quarter3 + s.quarter4

/-- First half score for a team -/
def firstHalfScore (s : Score) : ℚ :=
  s.quarter1 + s.quarter2

/-- Theorem stating the total first half score -/
theorem first_half_total_score :
  ⌈firstHalfScore eagles⌉ + ⌈firstHalfScore tigers⌉ = 19 ∧
  eagles.quarter1 = tigers.quarter1 ∧
  totalScore eagles = totalScore tigers + 2 ∧
  totalScore eagles ≤ 100 ∧
  totalScore tigers ≤ 100 :=
sorry


end NUMINAMATH_CALUDE_first_half_total_score_l3976_397619


namespace NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l3976_397616

theorem sqrt_eight_times_sqrt_two : Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l3976_397616


namespace NUMINAMATH_CALUDE_family_average_age_l3976_397662

theorem family_average_age (grandparents_avg : ℝ) (parents_avg : ℝ) (grandchildren_avg : ℝ)
  (h1 : grandparents_avg = 64)
  (h2 : parents_avg = 39)
  (h3 : grandchildren_avg = 6) :
  (2 * grandparents_avg + 2 * parents_avg + 3 * grandchildren_avg) / 7 = 32 := by
  sorry

end NUMINAMATH_CALUDE_family_average_age_l3976_397662


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3976_397678

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- State the theorem
theorem quadratic_function_range (a : ℝ) : 
  (∀ x ∈ Set.Ioc 0 1, |f a x| ≤ 1) → a ∈ Set.Icc (-2) 0 \ {0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3976_397678


namespace NUMINAMATH_CALUDE_two_consecutive_increases_l3976_397620

theorem two_consecutive_increases (initial : ℝ) (increase1 : ℝ) (increase2 : ℝ) : 
  let after_first_increase := initial * (1 + increase1 / 100)
  let final_number := after_first_increase * (1 + increase2 / 100)
  initial = 1256 ∧ increase1 = 325 ∧ increase2 = 147 → final_number = 6000.54 := by
sorry

end NUMINAMATH_CALUDE_two_consecutive_increases_l3976_397620


namespace NUMINAMATH_CALUDE_product_of_geometric_terms_l3976_397675

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

/-- The main theorem -/
theorem product_of_geometric_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n ≠ 0) →
  a 6 - a 7 ^ 2 + a 8 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 2 * b 8 * b 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_geometric_terms_l3976_397675


namespace NUMINAMATH_CALUDE_circle_radius_largest_radius_l3976_397604

/-- A circle tangent to both x and y axes with center (r,r) passing through (9,2) has radius 17 or 5 -/
theorem circle_radius (r : ℝ) : 
  (r > 0) → 
  ((9 - r)^2 + (2 - r)^2 = r^2) → 
  (r = 17 ∨ r = 5) :=
by sorry

/-- The largest possible radius of a circle tangent to both x and y axes and passing through (9,2) is 17 -/
theorem largest_radius : 
  ∃ (r : ℝ), (r > 0) ∧ 
  ((9 - r)^2 + (2 - r)^2 = r^2) ∧ 
  (∀ (s : ℝ), (s > 0) ∧ ((9 - s)^2 + (2 - s)^2 = s^2) → s ≤ r) ∧
  r = 17 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_largest_radius_l3976_397604


namespace NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l3976_397601

/-- Represents an algorithm --/
inductive Algorithm
  | ProductOfTwoNumbers
  | DistancePointToLine
  | QuadraticInequality
  | TrapezoidArea

/-- Determines if an algorithm requires a conditional branch structure --/
def requires_conditional_branch (a : Algorithm) : Prop :=
  match a with
  | Algorithm.QuadraticInequality => True
  | _ => False

/-- Theorem stating that only solving a quadratic inequality requires a conditional branch structure --/
theorem quadratic_inequality_requires_conditional_branch :
  ∀ (a : Algorithm), requires_conditional_branch a ↔ a = Algorithm.QuadraticInequality :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l3976_397601


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3976_397653

theorem quadratic_square_of_binomial (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 + 22 * x + 9 = (r * x + s)^2) →
  a = 121 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3976_397653


namespace NUMINAMATH_CALUDE_mother_twice_age_2040_l3976_397660

/-- The year when Tina's mother's age is twice Tina's age -/
def year_mother_twice_age (tina_birth_year : ℕ) (tina_age_2010 : ℕ) (mother_age_multiplier_2010 : ℕ) : ℕ :=
  tina_birth_year + (mother_age_multiplier_2010 - 2) * tina_age_2010

theorem mother_twice_age_2040 :
  year_mother_twice_age 2000 10 5 = 2040 := by
  sorry

#eval year_mother_twice_age 2000 10 5

end NUMINAMATH_CALUDE_mother_twice_age_2040_l3976_397660


namespace NUMINAMATH_CALUDE_quotient_calculation_l3976_397650

theorem quotient_calculation (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) 
  (h1 : dividend = 149)
  (h2 : divisor = 16)
  (h3 : remainder = 5)
  (h4 : dividend = divisor * 9 + remainder) :
  9 = dividend / divisor := by
sorry

end NUMINAMATH_CALUDE_quotient_calculation_l3976_397650


namespace NUMINAMATH_CALUDE_trundic_word_count_l3976_397637

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 15

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of required letters (A and B) -/
def required_letters : ℕ := 2

/-- Calculates the number of valid words in the Trundic language -/
def count_valid_words (alphabet_size : ℕ) (max_word_length : ℕ) (required_letters : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid words in the Trundic language -/
theorem trundic_word_count :
  count_valid_words alphabet_size max_word_length required_letters = 35180 :=
sorry

end NUMINAMATH_CALUDE_trundic_word_count_l3976_397637


namespace NUMINAMATH_CALUDE_apex_at_vertex_a_l3976_397666

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle with pillars -/
structure TriangleWithPillars where
  A : Point3D
  B : Point3D
  C : Point3D
  heightA : ℝ
  heightB : ℝ
  heightC : ℝ

/-- Check if three points form an equilateral triangle on the ground (z = 0) -/
def isEquilateral (t : TriangleWithPillars) : Prop :=
  let d1 := (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2
  let d2 := (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2
  let d3 := (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2
  d1 = d2 ∧ d2 = d3 ∧ t.A.z = 0 ∧ t.B.z = 0 ∧ t.C.z = 0

/-- Find the point directly below the apex of the inclined plane -/
def apexProjection (t : TriangleWithPillars) : Point3D :=
  { x := t.A.x, y := t.A.y, z := 0 }

/-- Theorem: The apex projection is at vertex A for the given triangle -/
theorem apex_at_vertex_a (t : TriangleWithPillars) :
  isEquilateral t ∧ t.heightA = 10 ∧ t.heightB = 8 ∧ t.heightC = 6 →
  apexProjection t = t.A :=
by sorry

end NUMINAMATH_CALUDE_apex_at_vertex_a_l3976_397666


namespace NUMINAMATH_CALUDE_zeros_of_f_l3976_397608

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Theorem statement
theorem zeros_of_f :
  ∃ (a b c : ℝ), (a = -1 ∧ b = 1 ∧ c = 2) ∧
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3976_397608


namespace NUMINAMATH_CALUDE_ab_nonzero_sufficient_for_a_nonzero_l3976_397646

theorem ab_nonzero_sufficient_for_a_nonzero (a b : ℝ) : 
  (∀ a b, a * b ≠ 0 → a ≠ 0) ∧ 
  ¬(∀ a b, a ≠ 0 → a * b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ab_nonzero_sufficient_for_a_nonzero_l3976_397646


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3976_397622

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  let binary : List Bool := [true, false, true, false, true, true, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let quaternary : List ℕ := decimal_to_quaternary decimal
  quaternary = [2, 2, 1, 3] :=
by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3976_397622


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3976_397627

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/49
  let a₃ : ℚ := 64/343
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → (7^n * a₁ = 4^n)) → r = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3976_397627


namespace NUMINAMATH_CALUDE_f_iter_has_two_roots_l3976_397606

def f (x : ℝ) : ℝ := x^2 + 2018*x + 1

def f_iter (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n+1 => f ∘ f_iter n

theorem f_iter_has_two_roots (n : ℕ+) : ∃ (x y : ℝ), x ≠ y ∧ f_iter n x = 0 ∧ f_iter n y = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_iter_has_two_roots_l3976_397606


namespace NUMINAMATH_CALUDE_complex_number_location_l3976_397676

/-- Given a complex number z satisfying z * (-1 + 3*I) = 1 + 7*I,
    prove that z is located in the fourth quadrant of the complex plane. -/
theorem complex_number_location (z : ℂ) (h : z * (-1 + 3*I) = 1 + 7*I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3976_397676


namespace NUMINAMATH_CALUDE_inverse_g_one_over_120_l3976_397613

noncomputable def g (x : ℝ) : ℝ := (x^5 + 1) / 5

theorem inverse_g_one_over_120 :
  g⁻¹ (1/120) = ((-23/24) : ℝ)^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_one_over_120_l3976_397613


namespace NUMINAMATH_CALUDE_base_conversion_sum_approx_l3976_397648

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 6, 2]  -- 263 in base 8
def num2 : List Nat := [3, 1]     -- 13 in base 3
def num3 : List Nat := [3, 4, 2]  -- 243 in base 7
def num4 : List Nat := [5, 3]     -- 35 in base 6

-- State the theorem
theorem base_conversion_sum_approx :
  let x1 := baseToDecimal num1 8
  let x2 := baseToDecimal num2 3
  let x3 := baseToDecimal num3 7
  let x4 := baseToDecimal num4 6
  abs ((x1 / x2 + x3 / x4 : ℚ) - 35.442) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_approx_l3976_397648


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l3976_397674

theorem probability_four_twos_in_five_rolls (p : ℝ) :
  p = 1 / 8 →
  (5 : ℝ) * p^4 * (1 - p) = 35 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l3976_397674


namespace NUMINAMATH_CALUDE_largest_quotient_is_30_l3976_397672

def S : Set Int := {-30, -5, -1, 0, 3, 9}

theorem largest_quotient_is_30 : 
  ∀ a b : Int, a ∈ S → b ∈ S → b ≠ 0 → a / b ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_is_30_l3976_397672


namespace NUMINAMATH_CALUDE_plane_sphere_sum_l3976_397695

-- Define the origin
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the fixed point (2a, 2b, 2c)
def fixed_point (a b c : ℝ) : ℝ × ℝ × ℝ := (2*a, 2*b, 2*c)

-- Define the points A, B, C on the axes
def A (α : ℝ) : ℝ × ℝ × ℝ := (α, 0, 0)
def B (β : ℝ) : ℝ × ℝ × ℝ := (0, β, 0)
def C (γ : ℝ) : ℝ × ℝ × ℝ := (0, 0, γ)

-- Define the center of the sphere
def sphere_center (p q r : ℝ) : ℝ × ℝ × ℝ := (p, q, r)

-- State the theorem
theorem plane_sphere_sum (a b c p q r α β γ : ℝ) 
  (h1 : A α ≠ O) (h2 : B β ≠ O) (h3 : C γ ≠ O)
  (h4 : sphere_center p q r ≠ O)
  (h5 : ∃ (t : ℝ), t * (2*a) / α + t * (2*b) / β + t * (2*c) / γ = t) 
  (h6 : ∀ (x y z : ℝ), (x - p)^2 + (y - q)^2 + (z - r)^2 = p^2 + q^2 + r^2 → 
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = α ∧ y = 0 ∧ z = 0) ∨ 
    (x = 0 ∧ y = β ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = γ)) :
  (2*a)/p + (2*b)/q + (2*c)/r = 2 := by
sorry

end NUMINAMATH_CALUDE_plane_sphere_sum_l3976_397695


namespace NUMINAMATH_CALUDE_parabola_focus_l3976_397669

/-- A parabola is defined by the equation y² = -16x + 64. -/
def parabola (x y : ℝ) : Prop := y^2 = -16*x + 64

/-- The focus of a parabola is a point on its axis of symmetry. -/
def is_focus (x y : ℝ) : Prop := sorry

/-- The focus of the parabola y² = -16x + 64 is at (0, 0). -/
theorem parabola_focus :
  is_focus 0 0 ∧ ∀ x y, parabola x y → is_focus x y → x = 0 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l3976_397669


namespace NUMINAMATH_CALUDE_broken_line_intersections_l3976_397623

/-- A broken line is represented as a list of points in the plane -/
def BrokenLine := List (Real × Real)

/-- The length of a broken line -/
def length (bl : BrokenLine) : Real :=
  sorry

/-- Checks if a broken line is inside the unit square -/
def isInsideUnitSquare (bl : BrokenLine) : Prop :=
  sorry

/-- Counts the number of intersections between a broken line and a line parallel to the x-axis -/
def intersectionsWithHorizontalLine (bl : BrokenLine) (y : Real) : Nat :=
  sorry

/-- Counts the number of intersections between a broken line and a line parallel to the y-axis -/
def intersectionsWithVerticalLine (bl : BrokenLine) (x : Real) : Nat :=
  sorry

/-- The main theorem -/
theorem broken_line_intersections (bl : BrokenLine) 
  (h1 : length bl = 1000)
  (h2 : isInsideUnitSquare bl) :
  (∃ y : Real, y ∈ Set.Icc 0 1 ∧ intersectionsWithHorizontalLine bl y ≥ 500) ∨
  (∃ x : Real, x ∈ Set.Icc 0 1 ∧ intersectionsWithVerticalLine bl x ≥ 500) :=
sorry

end NUMINAMATH_CALUDE_broken_line_intersections_l3976_397623


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3976_397665

theorem cube_root_equation_solution :
  ∀ x : ℝ, (10 - 6 * x)^(1/3 : ℝ) = -2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3976_397665


namespace NUMINAMATH_CALUDE_smallest_divisible_by_79_and_83_l3976_397682

theorem smallest_divisible_by_79_and_83 :
  ∃ (m : ℕ), 
    m > 0 ∧
    79 ∣ (m^3 - 3*m^2 + 2*m) ∧
    83 ∣ (m^3 - 3*m^2 + 2*m) ∧
    (∀ (k : ℕ), k > 0 ∧ k < m → ¬(79 ∣ (k^3 - 3*k^2 + 2*k) ∧ 83 ∣ (k^3 - 3*k^2 + 2*k))) ∧
    m = 3715 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_79_and_83_l3976_397682


namespace NUMINAMATH_CALUDE_solve_equation_l3976_397679

theorem solve_equation : ∃ r : ℤ, 19 - 3 = 2 + r ∧ r = 14 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3976_397679


namespace NUMINAMATH_CALUDE_fraction_equality_l3976_397630

theorem fraction_equality (a b : ℝ) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3976_397630


namespace NUMINAMATH_CALUDE_fib_divisibility_implies_fib_number_l3976_397654

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Condition: For every positive integer m, there exists a positive integer n such that m | Fₙ - k -/
def condition (k : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → ∃ n : ℕ, n > 0 ∧ (fib n - k) % m = 0

/-- Main theorem: If the condition holds, then k is a Fibonacci number -/
theorem fib_divisibility_implies_fib_number (k : ℕ) (h : condition k) :
  ∃ n : ℕ, fib n = k :=
sorry

end NUMINAMATH_CALUDE_fib_divisibility_implies_fib_number_l3976_397654


namespace NUMINAMATH_CALUDE_card_game_remainder_l3976_397645

def deck_size : ℕ := 60
def hand_size : ℕ := 12

def possible_remainders : List ℕ := [20, 40, 60, 80, 0]

theorem card_game_remainder :
  ∃ (r : ℕ), r ∈ possible_remainders ∧ 
  (Nat.choose deck_size hand_size) % 100 = r :=
sorry

end NUMINAMATH_CALUDE_card_game_remainder_l3976_397645


namespace NUMINAMATH_CALUDE_sqrt_one_implies_one_l3976_397632

theorem sqrt_one_implies_one (a : ℝ) : Real.sqrt a = 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_implies_one_l3976_397632


namespace NUMINAMATH_CALUDE_expression_evaluation_l3976_397688

theorem expression_evaluation (a b : ℝ) (h1 : a = 6) (h2 : b = 2) :
  ((3 / (a + b))^2) * (a - b) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3976_397688


namespace NUMINAMATH_CALUDE_least_N_for_prime_condition_l3976_397612

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a multiple of 12 -/
def isMultipleOf12 (n : ℕ) : Prop := sorry

/-- The theorem statement -/
theorem least_N_for_prime_condition : 
  ∃ (N : ℕ), N > 0 ∧ 
  (∀ (n : ℕ), isPrime (1 + N * 2^n) ↔ isMultipleOf12 n) ∧
  (∀ (M : ℕ), M > 0 → M < N → 
    ¬(∀ (n : ℕ), isPrime (1 + M * 2^n) ↔ isMultipleOf12 n)) ∧
  N = 556 := by
  sorry

end NUMINAMATH_CALUDE_least_N_for_prime_condition_l3976_397612


namespace NUMINAMATH_CALUDE_dogSchoolCount_l3976_397618

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  roll : ℕ
  sitStay : ℕ
  stayRoll : ℕ
  sitRoll : ℕ
  allThree : ℕ
  none : ℕ

/-- Calculates the total number of dogs in the school -/
def totalDogs (d : DogTricks) : ℕ :=
  d.allThree +
  (d.sitRoll - d.allThree) +
  (d.stayRoll - d.allThree) +
  (d.sitStay - d.allThree) +
  (d.sit - d.sitRoll - d.sitStay + d.allThree) +
  (d.stay - d.stayRoll - d.sitStay + d.allThree) +
  (d.roll - d.sitRoll - d.stayRoll + d.allThree) +
  d.none

/-- Theorem stating that the total number of dogs in the school is 84 -/
theorem dogSchoolCount (d : DogTricks)
  (h1 : d.sit = 50)
  (h2 : d.stay = 29)
  (h3 : d.roll = 34)
  (h4 : d.sitStay = 17)
  (h5 : d.stayRoll = 12)
  (h6 : d.sitRoll = 18)
  (h7 : d.allThree = 9)
  (h8 : d.none = 9) :
  totalDogs d = 84 := by
  sorry

end NUMINAMATH_CALUDE_dogSchoolCount_l3976_397618


namespace NUMINAMATH_CALUDE_three_digit_permutation_sum_l3976_397638

/-- A three-digit number with no zeros -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ d, d ∣ n → d ≠ 0

/-- Sum of all distinct permutations of the digits of a number -/
def SumOfPermutations (n : ℕ) : ℕ := sorry

theorem three_digit_permutation_sum (n : ℕ) :
  ThreeDigitNumber n → SumOfPermutations n = 2775 → n = 889 ∨ n = 997 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_permutation_sum_l3976_397638


namespace NUMINAMATH_CALUDE_point_A_in_third_quadrant_l3976_397614

/-- The point A with coordinates (sin 2018°, tan 117°) is in the third quadrant -/
theorem point_A_in_third_quadrant :
  let x : ℝ := Real.sin (2018 * π / 180)
  let y : ℝ := Real.tan (117 * π / 180)
  x < 0 ∧ y < 0 := by sorry

end NUMINAMATH_CALUDE_point_A_in_third_quadrant_l3976_397614


namespace NUMINAMATH_CALUDE_not_difference_of_squares_l3976_397671

/-- The difference of squares formula cannot be directly applied to (-x+y)(x-y) -/
theorem not_difference_of_squares (x y : ℝ) : 
  ¬ ∃ (a b : ℝ), (-x + y) * (x - y) = a^2 - b^2 :=
sorry

end NUMINAMATH_CALUDE_not_difference_of_squares_l3976_397671


namespace NUMINAMATH_CALUDE_inequality_solution_l3976_397624

theorem inequality_solution (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ x ∈ ({-2, -1, 0, 1, 2, 3} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3976_397624


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3976_397668

theorem cubic_roots_sum (u v w : ℝ) : 
  (u - Real.rpow 17 (1/3 : ℝ)) * (u - Real.rpow 67 (1/3 : ℝ)) * (u - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  (v - Real.rpow 17 (1/3 : ℝ)) * (v - Real.rpow 67 (1/3 : ℝ)) * (v - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  (w - Real.rpow 17 (1/3 : ℝ)) * (w - Real.rpow 67 (1/3 : ℝ)) * (w - Real.rpow 137 (1/3 : ℝ)) = 2/5 ∧
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  u^3 + v^3 + w^3 = 221 + 6/5 - 3 * 1549 :=
by sorry


end NUMINAMATH_CALUDE_cubic_roots_sum_l3976_397668


namespace NUMINAMATH_CALUDE_b_current_age_l3976_397609

/-- Given two people A and B, proves that B's current age is 39 years
    under the given conditions. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- A's age in 10 years equals twice B's age from 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39 :=                  -- B's current age is 39 years
by sorry

end NUMINAMATH_CALUDE_b_current_age_l3976_397609


namespace NUMINAMATH_CALUDE_robert_can_finish_both_books_l3976_397656

/-- Represents the number of pages Robert can read per hour -/
def reading_speed : ℕ := 120

/-- Represents the number of pages in the first book -/
def book1_pages : ℕ := 360

/-- Represents the number of pages in the second book -/
def book2_pages : ℕ := 180

/-- Represents the number of hours Robert has available for reading -/
def available_time : ℕ := 7

/-- Theorem stating that Robert can finish both books within the available time -/
theorem robert_can_finish_both_books :
  (book1_pages / reading_speed + book2_pages / reading_speed : ℚ) ≤ available_time :=
sorry

end NUMINAMATH_CALUDE_robert_can_finish_both_books_l3976_397656


namespace NUMINAMATH_CALUDE_factor_w6_minus_81_l3976_397664

theorem factor_w6_minus_81 (w : ℝ) : 
  w^6 - 81 = (w - 3) * (w^2 + 3*w + 9) * (w^3 + 9) := by sorry

end NUMINAMATH_CALUDE_factor_w6_minus_81_l3976_397664


namespace NUMINAMATH_CALUDE_prob_even_card_l3976_397603

/-- The probability of drawing a card with an even number from a set of cards -/
theorem prob_even_card (total_cards : ℕ) (even_cards : ℕ) 
  (h1 : total_cards = 6) 
  (h2 : even_cards = 3) : 
  (even_cards : ℚ) / total_cards = 1 / 2 := by
  sorry

#check prob_even_card

end NUMINAMATH_CALUDE_prob_even_card_l3976_397603


namespace NUMINAMATH_CALUDE_count_ordered_pairs_eq_six_l3976_397677

/-- The number of ordered pairs of positive integers (M, N) satisfying M/8 = 4/N -/
def count_ordered_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 32) (Finset.product (Finset.range 33) (Finset.range 33))).card

theorem count_ordered_pairs_eq_six : count_ordered_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_eq_six_l3976_397677


namespace NUMINAMATH_CALUDE_fifteen_sided_figure_area_l3976_397641

/-- A fifteen-sided figure on a 1 cm × 1 cm graph paper -/
structure FifteenSidedFigure where
  full_squares : ℕ
  small_triangles : ℕ
  h_full_squares : full_squares = 10
  h_small_triangles : small_triangles = 10

/-- The area of the fifteen-sided figure is 15 cm² -/
theorem fifteen_sided_figure_area (fig : FifteenSidedFigure) : 
  (fig.full_squares : ℝ) + (fig.small_triangles : ℝ) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_figure_area_l3976_397641


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3976_397625

theorem birds_and_storks_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (new_birds : ℕ) : 
  initial_birds = 3 → initial_storks = 2 → new_birds = 5 →
  initial_birds + initial_storks + new_birds = 10 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3976_397625


namespace NUMINAMATH_CALUDE_holly_weekly_pill_count_l3976_397615

/-- Calculates the total number of pills Holly takes in a week -/
def total_weekly_pills : ℕ :=
  let insulin_daily := 2
  let bp_daily := 3
  let anticonvulsant_daily := 2 * bp_daily
  let calcium_every_other_day := 3 * insulin_daily
  let vitamin_d_twice_weekly := 4
  let multivitamin_thrice_weekly := 1
  let anxiety_sunday := 3 * bp_daily

  7 * insulin_daily + 
  7 * bp_daily + 
  7 * anticonvulsant_daily + 
  (7 / 2) * calcium_every_other_day +
  2 * vitamin_d_twice_weekly + 
  3 * multivitamin_thrice_weekly + 
  anxiety_sunday

theorem holly_weekly_pill_count : total_weekly_pills = 118 := by
  sorry

end NUMINAMATH_CALUDE_holly_weekly_pill_count_l3976_397615


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l3976_397663

/-- A circle inscribed in a square with sides parallel to the axes -/
structure InscribedCircle where
  /-- The equation of the circle: x^2 + y^2 + 2x - 8y = 0 -/
  eq : ∀ (x y : ℝ), x^2 + y^2 + 2*x - 8*y = 0

/-- The area of the square that inscribes the circle -/
def squareArea (c : InscribedCircle) : ℝ := 68

/-- Theorem: The area of the square that inscribes the circle is 68 square units -/
theorem inscribed_circle_square_area (c : InscribedCircle) : squareArea c = 68 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l3976_397663


namespace NUMINAMATH_CALUDE_interest_problem_l3976_397658

/-- Given a sum P put at simple interest for 10 years, if increasing the interest rate
    by 5% results in Rs. 150 more interest, then P = 300. -/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 150 → P = 300 := by
  sorry


end NUMINAMATH_CALUDE_interest_problem_l3976_397658


namespace NUMINAMATH_CALUDE_f_and_g_properties_l3976_397640

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as the derivative of f
def g : ℝ → ℝ := f'

-- Axioms based on the problem conditions
axiom f_diff : ∀ x, HasDerivAt f (f' x) x
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- Theorem to prove
theorem f_and_g_properties :
  f (-1) = f 4 ∧ g (-1/2) = 0 :=
sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l3976_397640


namespace NUMINAMATH_CALUDE_weight_plate_problem_l3976_397602

theorem weight_plate_problem (num_plates : ℕ) (weight_increase : ℝ) (felt_weight : ℝ) :
  num_plates = 10 →
  weight_increase = 0.2 →
  felt_weight = 360 →
  (felt_weight / (1 + weight_increase)) / num_plates = 30 := by
  sorry

end NUMINAMATH_CALUDE_weight_plate_problem_l3976_397602


namespace NUMINAMATH_CALUDE_distance_AF_l3976_397600

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : Parabola x y

-- Define the property of the midpoint
def MidpointProperty (A : PointOnParabola) : Prop :=
  (A.x + Focus.1) / 2 = 2

-- Theorem statement
theorem distance_AF (A : PointOnParabola) 
  (h : MidpointProperty A) : 
  Real.sqrt ((A.x - Focus.1)^2 + (A.y - Focus.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_AF_l3976_397600


namespace NUMINAMATH_CALUDE_smallest_k_proof_l3976_397642

/-- The smallest integer k for which x^2 - x + 2 - k = 0 has two distinct real roots -/
def smallest_k : ℕ := 2

/-- The quadratic equation x^2 - x + 2 - k = 0 -/
def quadratic (x k : ℝ) : Prop := x^2 - x + 2 - k = 0

theorem smallest_k_proof :
  (∀ k < smallest_k, ¬∃ x y : ℝ, x ≠ y ∧ quadratic x k ∧ quadratic y k) ∧
  (∃ x y : ℝ, x ≠ y ∧ quadratic x smallest_k ∧ quadratic y smallest_k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_proof_l3976_397642
