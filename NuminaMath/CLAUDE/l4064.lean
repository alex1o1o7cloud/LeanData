import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l4064_406404

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l4064_406404


namespace NUMINAMATH_CALUDE_bakery_calculations_l4064_406409

-- Define the bakery's parameters
def cost_price : ℝ := 4
def selling_price : ℝ := 10
def clearance_price : ℝ := 2
def min_loaves : ℕ := 15
def max_loaves : ℕ := 30
def baked_loaves : ℕ := 21

-- Define the demand frequencies
def demand_freq : List (ℕ × ℕ) := [(15, 10), (18, 8), (21, 7), (24, 3), (27, 2)]

-- Calculate the probability of demand being at least 21 loaves
def prob_demand_ge_21 : ℚ := 2/5

-- Calculate the daily profit when demand is 15 loaves
def profit_demand_15 : ℝ := 78

-- Calculate the average daily profit over 30 days
def avg_daily_profit : ℝ := 103.6

theorem bakery_calculations :
  (prob_demand_ge_21 = 2/5) ∧
  (profit_demand_15 = 78) ∧
  (avg_daily_profit = 103.6) := by
  sorry

end NUMINAMATH_CALUDE_bakery_calculations_l4064_406409


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_plus_one_l4064_406455

theorem gcd_of_powers_of_101_plus_one (h : Nat.Prime 101) :
  Nat.gcd (101^5 + 1) (101^5 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_plus_one_l4064_406455


namespace NUMINAMATH_CALUDE_quarters_left_l4064_406441

/-- Given that Adam started with 88 quarters and spent 9 quarters at the arcade,
    prove that he had 79 quarters left. -/
theorem quarters_left (initial_quarters spent_quarters : ℕ) 
  (h1 : initial_quarters = 88)
  (h2 : spent_quarters = 9) :
  initial_quarters - spent_quarters = 79 := by
  sorry

end NUMINAMATH_CALUDE_quarters_left_l4064_406441


namespace NUMINAMATH_CALUDE_homework_difference_l4064_406485

def math_homework_pages : ℕ := 3
def reading_homework_pages : ℕ := 4

theorem homework_difference : reading_homework_pages - math_homework_pages = 1 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_l4064_406485


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l4064_406460

/-- Proves that the percentage in excess for the first side is 12% given the conditions of the problem -/
theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) :
  (L * (1 + x / 100) * (W * 0.95) = L * W * 1.064) →
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l4064_406460


namespace NUMINAMATH_CALUDE_labourer_absence_proof_l4064_406402

def total_days : ℕ := 25
def daily_wage : ℚ := 2
def daily_fine : ℚ := 1/2
def total_received : ℚ := 75/2

def days_absent : ℕ := 5

theorem labourer_absence_proof :
  ∃ (days_worked : ℕ),
    days_worked + days_absent = total_days ∧
    daily_wage * days_worked - daily_fine * days_absent = total_received :=
by sorry

end NUMINAMATH_CALUDE_labourer_absence_proof_l4064_406402


namespace NUMINAMATH_CALUDE_median_is_106_l4064_406470

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n (1 ≤ n ≤ 150) appears n times -/
def special_list : List ℕ := sorry

/-- The length of the special list -/
def special_list_length : ℕ := sum_to_n 150

/-- The median index of the special list -/
def median_index : ℕ := special_list_length / 2 + 1

theorem median_is_106 : 
  ∃ (l : List ℕ), l = special_list ∧ l.length = special_list_length ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 150 → (l.count n = n)) ∧
  (l.nthLe (median_index - 1) sorry = 106) :=
sorry

end NUMINAMATH_CALUDE_median_is_106_l4064_406470


namespace NUMINAMATH_CALUDE_max_fraction_sum_l4064_406449

theorem max_fraction_sum (x y : ℝ) :
  (Real.sqrt 3 * x - y + Real.sqrt 3 ≥ 0) →
  (Real.sqrt 3 * x + y - Real.sqrt 3 ≤ 0) →
  (y ≥ 0) →
  (∀ x' y' : ℝ, (Real.sqrt 3 * x' - y' + Real.sqrt 3 ≥ 0) →
                (Real.sqrt 3 * x' + y' - Real.sqrt 3 ≤ 0) →
                (y' ≥ 0) →
                ((y' + 1) / (x' + 3) ≤ (y + 1) / (x + 3))) →
  x + y = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l4064_406449


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l4064_406433

/-- Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 5 and -3, then a = 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l4064_406433


namespace NUMINAMATH_CALUDE_lauren_tuesday_earnings_l4064_406499

/-- Calculates Lauren's earnings from her social media channel -/
def laurens_earnings (commercial_rate : ℚ) (subscription_rate : ℚ) (commercial_views : ℕ) (subscriptions : ℕ) : ℚ :=
  commercial_rate * commercial_views + subscription_rate * subscriptions

theorem lauren_tuesday_earnings :
  let commercial_rate : ℚ := 1/2
  let subscription_rate : ℚ := 1
  let commercial_views : ℕ := 100
  let subscriptions : ℕ := 27
  laurens_earnings commercial_rate subscription_rate commercial_views subscriptions = 77 := by
sorry

end NUMINAMATH_CALUDE_lauren_tuesday_earnings_l4064_406499


namespace NUMINAMATH_CALUDE_ball_pit_problem_l4064_406478

theorem ball_pit_problem (total_balls : ℕ) (red_fraction : ℚ) (neither_red_nor_blue : ℕ) :
  total_balls = 360 →
  red_fraction = 1/4 →
  neither_red_nor_blue = 216 →
  (total_balls - red_fraction * total_balls - neither_red_nor_blue) / 
  (total_balls - red_fraction * total_balls) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ball_pit_problem_l4064_406478


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l4064_406440

/-- Represents a cross country meet with two teams of 6 runners each. -/
structure CrossCountryMeet where
  runners_per_team : Nat
  total_runners : Nat
  min_score : Nat
  max_score : Nat

/-- Calculates the number of possible winning scores in a cross country meet. -/
def possible_winning_scores (meet : CrossCountryMeet) : Nat :=
  meet.max_score - meet.min_score + 1

/-- Theorem stating the number of possible winning scores in the given cross country meet setup. -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.runners_per_team = 6 ∧
    meet.total_runners = 12 ∧
    meet.min_score = 21 ∧
    meet.max_score = 39 ∧
    possible_winning_scores meet = 19 := by
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l4064_406440


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l4064_406479

/-- Calculates the total number of people that can sit in a bus given the seating arrangement --/
theorem bus_seating_capacity (left_seats right_seats seat_capacity back_seat_capacity : ℕ) 
  (h1 : left_seats = 15)
  (h2 : right_seats = left_seats - 3)
  (h3 : seat_capacity = 3)
  (h4 : back_seat_capacity = 8) :
  left_seats * seat_capacity + right_seats * seat_capacity + back_seat_capacity = 89 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l4064_406479


namespace NUMINAMATH_CALUDE_runner_speed_l4064_406452

/-- Calculates the speed of a runner overtaking a parade -/
theorem runner_speed (parade_length : ℝ) (parade_speed : ℝ) (runner_time : ℝ) :
  parade_length = 2 →
  parade_speed = 3 →
  runner_time = 0.222222222222 →
  parade_length / runner_time = 9 :=
by sorry

end NUMINAMATH_CALUDE_runner_speed_l4064_406452


namespace NUMINAMATH_CALUDE_association_ticket_sales_l4064_406445

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  male_members : ℕ
  female_members : ℕ
  male_avg_tickets : ℝ
  female_avg_tickets : ℝ
  overall_avg_tickets : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem association_ticket_sales (a : Association) 
  (h1 : a.female_members = 2 * a.male_members)
  (h2 : a.overall_avg_tickets = 66)
  (h3 : a.male_avg_tickets = 58) :
  a.female_avg_tickets = 70 := by
  sorry


end NUMINAMATH_CALUDE_association_ticket_sales_l4064_406445


namespace NUMINAMATH_CALUDE_difference_average_median_l4064_406442

theorem difference_average_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((1 + (a + 1) + (2*a + b) + (a + b + 1)) / 4) - ((a + 1 + (a + b + 1)) / 2)| = 1/4 := by
sorry

end NUMINAMATH_CALUDE_difference_average_median_l4064_406442


namespace NUMINAMATH_CALUDE_line_segment_param_sum_of_squares_l4064_406486

/-- Given a line segment connecting (-3,9) and (2,12), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (-3,9), prove that a^2 + b^2 + c^2 + d^2 = 124 -/
theorem line_segment_param_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = -3 ∧ d = 9) →
  (a + b = 2 ∧ c + d = 12) →
  a^2 + b^2 + c^2 + d^2 = 124 :=
by sorry


end NUMINAMATH_CALUDE_line_segment_param_sum_of_squares_l4064_406486


namespace NUMINAMATH_CALUDE_average_b_c_l4064_406423

theorem average_b_c (a b c : ℝ) : 
  (a + b) / 2 = 45 →
  c - a = 10 →
  (b + c) / 2 = 50 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l4064_406423


namespace NUMINAMATH_CALUDE_dividend_divisor_remainder_l4064_406413

theorem dividend_divisor_remainder (x y : ℕ+) :
  (x : ℝ) / (y : ℝ) = 96.12 →
  (x : ℝ) % (y : ℝ) = 1.44 →
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_remainder_l4064_406413


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l4064_406403

theorem sum_remainder_mod_nine : (8243 + 8244 + 8245 + 8246) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l4064_406403


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l4064_406420

-- Define the doubling time in seconds
def doubling_time : ℕ := 15

-- Define the total time in seconds
def total_time : ℕ := 4 * 60

-- Define the final number of bacteria
def final_bacteria : ℕ := 2097152

-- Theorem statement
theorem initial_bacteria_count :
  ∃ (initial : ℕ), 
    initial * (2 ^ (total_time / doubling_time)) = final_bacteria ∧
    initial = 32 := by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l4064_406420


namespace NUMINAMATH_CALUDE_skating_minutes_needed_for_average_l4064_406456

def skating_schedule (days : Nat) (hours_per_day : Nat) : Nat :=
  days * hours_per_day * 60

def total_minutes_8_days : Nat :=
  skating_schedule 6 1 + skating_schedule 2 2

def average_minutes_per_day : Nat := 100

def total_days : Nat := 10

theorem skating_minutes_needed_for_average :
  skating_schedule 6 1 + skating_schedule 2 2 + 400 = total_days * average_minutes_per_day := by
  sorry

end NUMINAMATH_CALUDE_skating_minutes_needed_for_average_l4064_406456


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4064_406416

theorem perfect_square_condition (Z K : ℤ) : 
  (500 < Z ∧ Z < 1000) →
  K > 1 →
  Z = K * K^2 →
  (∃ n : ℤ, Z = n^2) →
  K = 9 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l4064_406416


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_min_value_three_iff_l4064_406464

-- Define the function f
def f (x a : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - a)

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 5} = {x : ℝ | x ≤ -2 ∨ x ≥ 4/3} := by sorry

-- Theorem for part (2)
theorem min_value_three_iff :
  (∃ x : ℝ, f x a = 3) ∧ (∀ x : ℝ, f x a ≥ 3) ↔ a = 2 ∨ a = -4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_min_value_three_iff_l4064_406464


namespace NUMINAMATH_CALUDE_pencil_sale_l4064_406467

theorem pencil_sale (x : ℕ) : 
  (2 * x) + (6 * 3) + (2 * 1) = 24 → x = 2 := by sorry

end NUMINAMATH_CALUDE_pencil_sale_l4064_406467


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4064_406462

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (z + 2) * (1 + i^3) = 2

-- Theorem statement
theorem complex_equation_solution :
  ∃ z : ℂ, equation z ∧ z = -1 + i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4064_406462


namespace NUMINAMATH_CALUDE_dropped_student_score_l4064_406482

theorem dropped_student_score 
  (initial_count : ℕ) 
  (remaining_count : ℕ) 
  (initial_average : ℚ) 
  (new_average : ℚ) 
  (h1 : initial_count = 16)
  (h2 : remaining_count = 15)
  (h3 : initial_average = 62.5)
  (h4 : new_average = 62)
  : (initial_count : ℚ) * initial_average - (remaining_count : ℚ) * new_average = 70 :=
by sorry

end NUMINAMATH_CALUDE_dropped_student_score_l4064_406482


namespace NUMINAMATH_CALUDE_admission_methods_l4064_406430

theorem admission_methods (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = 8 → k = 2 → s = 3 → (n.choose k) * s = 84 :=
by sorry

end NUMINAMATH_CALUDE_admission_methods_l4064_406430


namespace NUMINAMATH_CALUDE_fathers_catch_l4064_406471

/-- The number of fishes Hazel caught -/
def hazel_catch : ℕ := 48

/-- The total number of fishes caught by Hazel and her father -/
def total_catch : ℕ := 94

/-- Hazel's father's catch is the difference between the total catch and Hazel's catch -/
theorem fathers_catch (hazel_catch : ℕ) (total_catch : ℕ) : 
  total_catch - hazel_catch = 46 :=
by sorry

end NUMINAMATH_CALUDE_fathers_catch_l4064_406471


namespace NUMINAMATH_CALUDE_function_value_at_two_l4064_406474

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = 3 * x - 2) :
  f 2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l4064_406474


namespace NUMINAMATH_CALUDE_solve_equations_l4064_406426

-- Define the equations
def equation1 (y : ℝ) : Prop := 2.4 * y - 9.8 = 1.4 * y - 9
def equation2 (x : ℝ) : Prop := x - 3 = (3/2) * x + 1

-- State the theorem
theorem solve_equations :
  (∃ y : ℝ, equation1 y ∧ y = 0.8) ∧
  (∃ x : ℝ, equation2 x ∧ x = -8) := by sorry

end NUMINAMATH_CALUDE_solve_equations_l4064_406426


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l4064_406437

open Real

theorem triangle_angle_equality (A B : ℝ) (a b : ℝ) 
  (h1 : sin A / a = cos B / b) 
  (h2 : a = b) : 
  B = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l4064_406437


namespace NUMINAMATH_CALUDE_brenda_bracelets_l4064_406447

/-- Given the number of bracelets and total number of stones, 
    calculate the number of stones per bracelet -/
def stones_per_bracelet (num_bracelets : ℕ) (total_stones : ℕ) : ℕ :=
  total_stones / num_bracelets

/-- Theorem: Given 3 bracelets and 36 total stones, 
    there will be 12 stones per bracelet -/
theorem brenda_bracelets : stones_per_bracelet 3 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_brenda_bracelets_l4064_406447


namespace NUMINAMATH_CALUDE_expression_simplification_l4064_406448

theorem expression_simplification (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a*b + b*c + c*a + 3)⁻¹ * ((a*b)⁻¹ + (b*c)⁻¹ + (c*a)⁻¹ + 3) = (a*b*c)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4064_406448


namespace NUMINAMATH_CALUDE_slope_values_theorem_l4064_406488

def valid_slopes : List ℕ := [81, 192, 399, 501, 1008, 2019]

def intersects_parabola (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁^2 = k * x₁ + 2020 ∧ x₂^2 = k * x₂ + 2020

theorem slope_values_theorem :
  ∀ k : ℝ, k > 0 → intersects_parabola k → k ∈ valid_slopes.map (λ x => x : ℕ → ℝ) := by
  sorry

end NUMINAMATH_CALUDE_slope_values_theorem_l4064_406488


namespace NUMINAMATH_CALUDE_right_triangle_sin_x_l4064_406422

theorem right_triangle_sin_x (X Y Z : Real) (sinX cosX tanX : Real) :
  -- Right triangle XYZ with ∠Y = 90°
  (X^2 + Y^2 = Z^2) →
  -- 4sinX = 5cosX
  (4 * sinX = 5 * cosX) →
  -- tanX = XY/YZ
  (tanX = X / Y) →
  -- sinX = 5√41 / 41
  sinX = 5 * Real.sqrt 41 / 41 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_x_l4064_406422


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l4064_406492

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence (α : Type*) [Field α] where
  a : ℕ → α
  d : α
  h_nonzero : d ≠ 0
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The 13th term of the arithmetic sequence is 28 -/
theorem arithmetic_sequence_13th_term
  {α : Type*} [Field α] (seq : ArithmeticSequence α)
  (h_geometric : seq.a 9 * seq.a 1 = (seq.a 5)^2)
  (h_sum : seq.a 1 + 3 * seq.a 5 + seq.a 9 = 20) :
  seq.a 13 = 28 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l4064_406492


namespace NUMINAMATH_CALUDE_vector_on_line_and_parallel_l4064_406419

def line_x (t : ℝ) : ℝ := 5 * t + 3
def line_y (t : ℝ) : ℝ := t + 3

def vector_a : ℝ := 18
def vector_b : ℝ := 6

def parallel_vector_x : ℝ := 3
def parallel_vector_y : ℝ := 1

theorem vector_on_line_and_parallel :
  (∃ t : ℝ, line_x t = vector_a ∧ line_y t = vector_b) ∧
  (∃ k : ℝ, vector_a = k * parallel_vector_x ∧ vector_b = k * parallel_vector_y) :=
sorry

end NUMINAMATH_CALUDE_vector_on_line_and_parallel_l4064_406419


namespace NUMINAMATH_CALUDE_equality_of_expressions_l4064_406438

theorem equality_of_expressions :
  (-2^7 = (-2)^7) ∧
  (-3^2 ≠ (-3)^2) ∧
  (-3 * 2^3 ≠ -3^2 * 2) ∧
  (-((-3)^2) ≠ -((-2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l4064_406438


namespace NUMINAMATH_CALUDE_c_is_power_of_two_l4064_406480

/-- Represents a nonempty string of base-ten digits -/
structure DigitString where
  digits : List Nat
  nonempty : digits.length > 0

/-- Represents the number of ways to split a DigitString into parts divisible by m -/
def c (m : Nat) (S : DigitString) : Nat :=
  sorry

/-- Main theorem: c(S) is always 0 or a power of 2 -/
theorem c_is_power_of_two (m : Nat) (S : DigitString) (h : m > 1) :
  ∃ k : Nat, c m S = 0 ∨ c m S = 2^k :=
sorry

end NUMINAMATH_CALUDE_c_is_power_of_two_l4064_406480


namespace NUMINAMATH_CALUDE_average_bottle_price_l4064_406483

def large_bottles : ℕ := 1325
def small_bottles : ℕ := 750
def large_bottle_price : ℚ := 189/100
def small_bottle_price : ℚ := 138/100

theorem average_bottle_price :
  let total_cost : ℚ := large_bottles * large_bottle_price + small_bottles * small_bottle_price
  let total_bottles : ℕ := large_bottles + small_bottles
  let average_price : ℚ := total_cost / total_bottles
  ∃ ε > 0, |average_price - 17/10| < ε ∧ ε < 1/100 :=
by
  sorry

end NUMINAMATH_CALUDE_average_bottle_price_l4064_406483


namespace NUMINAMATH_CALUDE_total_books_count_l4064_406493

def initial_books : ℕ := 35
def bought_books : ℕ := 21

theorem total_books_count : initial_books + bought_books = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l4064_406493


namespace NUMINAMATH_CALUDE_computer_profit_calculation_l4064_406484

theorem computer_profit_calculation (C : ℝ) :
  (C + 0.4 * C = 2240) → (C + 0.6 * C = 2560) := by
  sorry

end NUMINAMATH_CALUDE_computer_profit_calculation_l4064_406484


namespace NUMINAMATH_CALUDE_alices_preferred_number_l4064_406450

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem alices_preferred_number (n : ℕ) 
  (h1 : 70 < n ∧ n < 140)
  (h2 : n % 13 = 0)
  (h3 : n % 3 ≠ 0)
  (h4 : sum_of_digits n % 4 = 0) :
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_alices_preferred_number_l4064_406450


namespace NUMINAMATH_CALUDE_min_value_theorem_l4064_406497

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → 2/x + 3/y ≥ 5 + 2 * Real.sqrt 6) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.log (x + y) = 0 ∧ 2/x + 3/y = 5 + 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4064_406497


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l4064_406473

theorem cubic_equation_sum (r s t : ℝ) : 
  r^3 - 4*r^2 + 4*r = 6 →
  s^3 - 4*s^2 + 4*s = 6 →
  t^3 - 4*t^2 + 4*t = 6 →
  r*s/t + s*t/r + t*r/s = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l4064_406473


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l4064_406410

/-- A nonzero digit is a natural number between 1 and 9. -/
def NonzeroDigit : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- Convert a three-digit number represented by three digits to a natural number. -/
def threeDigitToNat (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Convert a two-digit number represented by two digits to a natural number. -/
def twoDigitToNat (a b : ℕ) : ℕ := 10 * a + b

/-- The main theorem: the sum of the four numbers always has 5 digits. -/
theorem sum_has_five_digits (A B C : NonzeroDigit) :
  ∃ (n : ℕ), 10000 ≤ 21478 + threeDigitToNat A.val 5 9 + twoDigitToNat B.val 4 + twoDigitToNat C.val 6 ∧
             21478 + threeDigitToNat A.val 5 9 + twoDigitToNat B.val 4 + twoDigitToNat C.val 6 < 100000 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l4064_406410


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l4064_406424

def is_arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |>.sum

theorem arithmetic_progression_problem (a : ℕ → ℤ) (S : ℤ) :
  is_arithmetic_progression a →
  is_increasing_sequence a →
  S = sum_first_n_terms a 5 →
  a 6 * a 11 > S + 15 →
  a 9 * a 8 < S + 39 →
  a 1 ∈ ({-9, -8, -7, -6, -4, -3, -2, -1} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l4064_406424


namespace NUMINAMATH_CALUDE_minimum_sum_of_parameters_l4064_406457

theorem minimum_sum_of_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a + 1 / b = 1) → (a + b ≥ 4) ∧ (∃ a b, 1 / a + 1 / b = 1 ∧ a + b = 4) :=
sorry

end NUMINAMATH_CALUDE_minimum_sum_of_parameters_l4064_406457


namespace NUMINAMATH_CALUDE_eighteen_twelve_over_fiftyfour_six_l4064_406443

theorem eighteen_twelve_over_fiftyfour_six : (18 ^ 12) / (54 ^ 6) = 46656 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_twelve_over_fiftyfour_six_l4064_406443


namespace NUMINAMATH_CALUDE_max_side_length_24_perimeter_l4064_406406

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different_sides : a ≠ b ∧ b ≠ c ∧ a ≠ c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  perimeter_24 : a + b + c = 24

/-- The maximum length of any side in a triangle with perimeter 24 and different integer side lengths is 12 -/
theorem max_side_length_24_perimeter (t : Triangle) : t.a ≤ 12 ∧ t.b ≤ 12 ∧ t.c ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_side_length_24_perimeter_l4064_406406


namespace NUMINAMATH_CALUDE_cellphone_surveys_count_l4064_406458

/-- Represents the weekly survey data for a worker --/
structure SurveyData where
  regularRate : ℕ
  totalSurveys : ℕ
  cellphoneRateIncrease : ℚ
  totalEarnings : ℕ

/-- Calculates the number of cellphone surveys given the survey data --/
def calculateCellphoneSurveys (data : SurveyData) : ℕ :=
  sorry

/-- Theorem stating that the number of cellphone surveys is 50 for the given data --/
theorem cellphone_surveys_count (data : SurveyData) 
  (h1 : data.regularRate = 30)
  (h2 : data.totalSurveys = 100)
  (h3 : data.cellphoneRateIncrease = 1/5)
  (h4 : data.totalEarnings = 3300) :
  calculateCellphoneSurveys data = 50 := by
  sorry

end NUMINAMATH_CALUDE_cellphone_surveys_count_l4064_406458


namespace NUMINAMATH_CALUDE_range_f_characterization_l4064_406428

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1

-- Define the range of f(x) on [0, 2]
def range_f (a : ℝ) : Set ℝ :=
  { y | ∃ x ∈ Set.Icc 0 2, f a x = y }

-- Theorem statement
theorem range_f_characterization (a : ℝ) :
  range_f a =
    if a < 0 then Set.Icc (-1) (3 - 4*a)
    else if a < 1 then Set.Icc (-1 - a^2) (3 - 4*a)
    else if a < 2 then Set.Icc (-1 - a^2) (-1)
    else Set.Icc (3 - 4*a) (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_f_characterization_l4064_406428


namespace NUMINAMATH_CALUDE_BE_length_l4064_406429

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 3 ∧ dist B C = 4 ∧ dist C A = 5

-- Define points D and E on ray AB
def points_on_ray (A B D E : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ > 1 ∧ t₂ > t₁ ∧ D = A + t₁ • (B - A) ∧ E = A + t₂ • (B - A)

-- Define point F as intersection of circumcircles
def point_F (A B C D E F : ℝ × ℝ) : Prop :=
  F ≠ C ∧
  ∃ r₁ r₂ : ℝ,
    dist A F = r₁ ∧ dist C F = r₁ ∧ dist D F = r₁ ∧
    dist E F = r₂ ∧ dist B F = r₂ ∧ dist C F = r₂

-- Main theorem
theorem BE_length (A B C D E F : ℝ × ℝ) :
  triangle_ABC A B C →
  points_on_ray A B D E →
  point_F A B C D E F →
  dist D F = 3 →
  dist E F = 8 →
  dist B E = 3 + Real.sqrt 34.6 :=
sorry

end NUMINAMATH_CALUDE_BE_length_l4064_406429


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4064_406427

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4064_406427


namespace NUMINAMATH_CALUDE_mcdonalds_fries_cost_l4064_406487

/-- The cost of one pack of fries at McDonald's -/
def fries_cost : ℝ := 2

/-- The cost of a burger at McDonald's -/
def burger_cost : ℝ := 5

/-- The cost of a salad at McDonald's -/
def salad_cost (f : ℝ) : ℝ := 3 * f

/-- The total cost of the meal at McDonald's -/
def total_cost (f : ℝ) : ℝ := salad_cost f + burger_cost + 2 * f

theorem mcdonalds_fries_cost :
  fries_cost = 2 ∧ total_cost fries_cost = 15 :=
sorry

end NUMINAMATH_CALUDE_mcdonalds_fries_cost_l4064_406487


namespace NUMINAMATH_CALUDE_sum_of_divisors_36_l4064_406408

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_36 : sum_of_divisors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_36_l4064_406408


namespace NUMINAMATH_CALUDE_tangent_length_circle_tangent_length_l4064_406417

/-- The length of a tangent to a circle from an external point -/
theorem tangent_length (r d l : ℝ) (hr : r > 0) (hd : d > r) : 
  r = 36 → d = 85 → l = 77 → l^2 = d^2 - r^2 := by
  sorry

/-- The main theorem stating the length of the tangent -/
theorem circle_tangent_length : 
  ∃ (l : ℝ), l = 77 ∧ l^2 = 85^2 - 36^2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_circle_tangent_length_l4064_406417


namespace NUMINAMATH_CALUDE_smallest_valid_seating_eighteen_is_valid_smallest_seating_is_eighteen_l4064_406411

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  totalChairs : Nat
  seatedPeople : Nat

/-- Checks if a seating arrangement is valid (any new person must sit next to someone). -/
def isValidSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.totalChairs ≥ table.seatedPeople ∧
  table.totalChairs % table.seatedPeople = 0 ∧
  table.totalChairs / table.seatedPeople ≤ 4

/-- The theorem stating the smallest valid number of seated people for a 72-chair table. -/
theorem smallest_valid_seating :
  ∀ (table : CircularTable),
    table.totalChairs = 72 →
    isValidSeating table →
    table.seatedPeople ≥ 18 :=
by
  sorry

/-- The theorem stating that 18 is a valid seating arrangement for a 72-chair table. -/
theorem eighteen_is_valid :
  isValidSeating { totalChairs := 72, seatedPeople := 18 } :=
by
  sorry

/-- The main theorem combining the above results to prove 18 is the smallest valid seating. -/
theorem smallest_seating_is_eighteen :
  ∃ (table : CircularTable),
    table.totalChairs = 72 ∧
    table.seatedPeople = 18 ∧
    isValidSeating table ∧
    ∀ (otherTable : CircularTable),
      otherTable.totalChairs = 72 →
      isValidSeating otherTable →
      otherTable.seatedPeople ≥ table.seatedPeople :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_eighteen_is_valid_smallest_seating_is_eighteen_l4064_406411


namespace NUMINAMATH_CALUDE_runners_meet_time_l4064_406425

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.minutes + m
  { hours := t.hours + totalMinutes / 60
    minutes := totalMinutes % 60 }

theorem runners_meet_time (startTime : Time) (lapTime1 lapTime2 lapTime3 : Nat) : 
  startTime.hours = 7 ∧ startTime.minutes = 45 ∧
  lapTime1 = 5 ∧ lapTime2 = 8 ∧ lapTime3 = 10 →
  let meetTime := addMinutes startTime (Nat.lcm lapTime1 (Nat.lcm lapTime2 lapTime3))
  meetTime.hours = 8 ∧ meetTime.minutes = 25 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_time_l4064_406425


namespace NUMINAMATH_CALUDE_log_simplification_l4064_406435

-- Define the variables as positive real numbers
variable (a b d e y z : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hd : 0 < d) (he : 0 < e) (hy : 0 < y) (hz : 0 < z)

-- State the theorem
theorem log_simplification :
  Real.log (a / b) + Real.log (b / e) + Real.log (e / d) - Real.log (a * z / (d * y)) = Real.log (d * y / z) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l4064_406435


namespace NUMINAMATH_CALUDE_tan_2x_and_sin_x_plus_pi_4_l4064_406432

theorem tan_2x_and_sin_x_plus_pi_4 (x : ℝ) 
  (h1 : |Real.tan x| = 2) 
  (h2 : x ∈ Set.Ioo (π / 2) π) : 
  Real.tan (2 * x) = 4 / 3 ∧ 
  Real.sin (x + π / 4) = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_2x_and_sin_x_plus_pi_4_l4064_406432


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l4064_406401

theorem quadratic_equation_root (x : ℝ) : x^2 - 6*x - 4 = 0 ↔ x = Real.sqrt 5 - 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l4064_406401


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l4064_406476

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The total maximum number of intersection points -/
def total_max_intersections : ℕ := max_circle_intersections + max_line_circle_intersections

theorem max_intersections_three_circles_one_line :
  total_max_intersections = 12 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l4064_406476


namespace NUMINAMATH_CALUDE_teacher_discount_l4064_406439

theorem teacher_discount (students : ℕ) (pens_per_student : ℕ) (notebooks_per_student : ℕ) 
  (binders_per_student : ℕ) (highlighters_per_student : ℕ) 
  (pen_cost : ℚ) (notebook_cost : ℚ) (binder_cost : ℚ) (highlighter_cost : ℚ) 
  (amount_spent : ℚ) : 
  students = 30 →
  pens_per_student = 5 →
  notebooks_per_student = 3 →
  binders_per_student = 1 →
  highlighters_per_student = 2 →
  pen_cost = 1/2 →
  notebook_cost = 5/4 →
  binder_cost = 17/4 →
  highlighter_cost = 3/4 →
  amount_spent = 260 →
  (students * pens_per_student * pen_cost + 
   students * notebooks_per_student * notebook_cost + 
   students * binders_per_student * binder_cost + 
   students * highlighters_per_student * highlighter_cost) - amount_spent = 100 := by
  sorry

end NUMINAMATH_CALUDE_teacher_discount_l4064_406439


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l4064_406451

theorem geometric_progression_proof (x : ℝ) (r : ℝ) : 
  (((30 + x) / (10 + x) = r) ∧ ((90 + x) / (30 + x) = r)) ↔ (x = 0 ∧ r = 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l4064_406451


namespace NUMINAMATH_CALUDE_set_equality_sum_l4064_406431

theorem set_equality_sum (x y : ℝ) (A B : Set ℝ) : 
  A = {2, y} → B = {x, 3} → A = B → x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_sum_l4064_406431


namespace NUMINAMATH_CALUDE_kwi_wins_l4064_406489

/-- Represents a frog in the race -/
structure Frog where
  name : String
  jump_length : ℚ
  jumps_per_time_unit : ℚ

/-- Calculates the time taken to complete the race for a given frog -/
def race_time (f : Frog) (race_distance : ℚ) : ℚ :=
  (race_distance / f.jump_length) / f.jumps_per_time_unit

/-- The race distance in decimeters -/
def total_race_distance : ℚ := 400

/-- Kwa, the first frog -/
def kwa : Frog := ⟨"Kwa", 6, 2⟩

/-- Kwi, the second frog -/
def kwi : Frog := ⟨"Kwi", 4, 3⟩

theorem kwi_wins : race_time kwi total_race_distance < race_time kwa total_race_distance := by
  sorry

end NUMINAMATH_CALUDE_kwi_wins_l4064_406489


namespace NUMINAMATH_CALUDE_max_a_value_l4064_406495

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → Real.exp x + Real.sin x - 2*x ≥ a*x^2 + 1) → 
  a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l4064_406495


namespace NUMINAMATH_CALUDE_intersecting_lines_regions_l4064_406466

/-- The number of regions created by n intersecting lines -/
def num_regions (n : ℕ) : ℕ := (n * n - n + 2) / 2 + 1

/-- Theorem stating that for any n ≥ 5, there exists a configuration of n intersecting lines
    that divides the plane into at least n regions -/
theorem intersecting_lines_regions (n : ℕ) (h : n ≥ 5) :
  num_regions n ≥ n :=
by sorry

end NUMINAMATH_CALUDE_intersecting_lines_regions_l4064_406466


namespace NUMINAMATH_CALUDE_math_only_students_l4064_406496

theorem math_only_students (total : ℕ) (math foreign_lang science : Finset ℕ) :
  total = 120 →
  (∀ s, s ∈ math ∪ foreign_lang ∪ science) →
  math.card = 85 →
  foreign_lang.card = 65 →
  science.card = 50 →
  (math ∩ foreign_lang ∩ science).card = 20 →
  (math \ (foreign_lang ∪ science)).card = 52 :=
by sorry

end NUMINAMATH_CALUDE_math_only_students_l4064_406496


namespace NUMINAMATH_CALUDE_blind_students_count_l4064_406465

theorem blind_students_count (total : ℕ) (deaf_ratio : ℕ) : 
  total = 180 → deaf_ratio = 3 → 
  ∃ (blind : ℕ), blind = 45 ∧ total = blind + deaf_ratio * blind :=
by
  sorry

end NUMINAMATH_CALUDE_blind_students_count_l4064_406465


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l4064_406469

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l4064_406469


namespace NUMINAMATH_CALUDE_division_remainder_problem_l4064_406412

theorem division_remainder_problem (k : ℕ+) (h : ∃ b : ℕ, 80 = b * k^2 + 8) :
  ∃ q : ℕ, 140 = q * k + 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l4064_406412


namespace NUMINAMATH_CALUDE_carpet_length_l4064_406481

/-- Given a rectangular carpet with width 4 feet covering 75% of a 48 square feet room,
    prove that the length of the carpet is 9 feet. -/
theorem carpet_length (room_area : ℝ) (carpet_width : ℝ) (coverage_percent : ℝ) :
  room_area = 48 →
  carpet_width = 4 →
  coverage_percent = 0.75 →
  (room_area * coverage_percent) / carpet_width = 9 :=
by sorry

end NUMINAMATH_CALUDE_carpet_length_l4064_406481


namespace NUMINAMATH_CALUDE_rectangle_y_value_l4064_406475

/-- Given a rectangle with vertices at (2, y), (10, y), (2, -1), and (10, -1),
    where y is negative and the area is 96 square units, prove that y = -13. -/
theorem rectangle_y_value (y : ℝ) : 
  y < 0 → -- y is negative
  (10 - 2) * |(-1) - y| = 96 → -- area of the rectangle is 96
  y = -13 := by sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l4064_406475


namespace NUMINAMATH_CALUDE_cipher_solution_l4064_406494

/-- Represents a mapping from letters to digits -/
def Cipher := Char → Nat

/-- The condition that each letter represents a unique digit -/
def is_valid_cipher (c : Cipher) : Prop :=
  ∀ x y : Char, c x = c y → x = y

/-- The value of a word under a given cipher -/
def word_value (c : Cipher) (w : String) : Nat :=
  w.foldl (λ acc d => 10 * acc + c d) 0

/-- The main theorem -/
theorem cipher_solution (c : Cipher) 
  (h1 : is_valid_cipher c)
  (h2 : word_value c "СЕКРЕТ" - word_value c "ОТКРЫТ" = 20010)
  (h3 : c 'Т' = 9) :
  word_value c "СЕК" = 392 ∧ c 'О' = 2 :=
sorry

end NUMINAMATH_CALUDE_cipher_solution_l4064_406494


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l4064_406405

theorem negative_fractions_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l4064_406405


namespace NUMINAMATH_CALUDE_lulu_piggy_bank_l4064_406407

theorem lulu_piggy_bank (initial_amount : ℝ) : 
  (4/5 * (1/2 * (initial_amount - 5))) = 24 → initial_amount = 65 := by
  sorry

end NUMINAMATH_CALUDE_lulu_piggy_bank_l4064_406407


namespace NUMINAMATH_CALUDE_violet_hiking_time_l4064_406444

/-- Proves that Violet and her dog can spend 4 hours hiking given the conditions --/
theorem violet_hiking_time :
  let violet_water_per_hour : ℚ := 800 / 1000  -- Convert ml to L
  let dog_water_per_hour : ℚ := 400 / 1000     -- Convert ml to L
  let total_water_capacity : ℚ := 4.8          -- In L
  
  (total_water_capacity / (violet_water_per_hour + dog_water_per_hour) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_violet_hiking_time_l4064_406444


namespace NUMINAMATH_CALUDE_impossible_cover_l4064_406446

/-- Represents an L-trimino piece -/
structure LTrimino where
  covers : Nat
  covers_eq : covers = 3

/-- Represents a 3x5 board with special squares -/
structure Board where
  total_squares : Nat
  total_squares_eq : total_squares = 15
  special_squares : Nat
  special_squares_eq : special_squares = 6

/-- States that it's impossible to cover the board with L-triminos -/
theorem impossible_cover (b : Board) (l : LTrimino) : 
  ¬∃ (n : Nat), n * l.covers = b.total_squares ∧ n ≥ b.special_squares :=
sorry

end NUMINAMATH_CALUDE_impossible_cover_l4064_406446


namespace NUMINAMATH_CALUDE_square_properties_l4064_406415

/-- Given a square with diagonal length 12√2 cm, prove its perimeter and inscribed circle area -/
theorem square_properties (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  let perimeter := 4 * s
  let r := s / 2
  let inscribed_circle_area := Real.pi * r^2
  (perimeter = 48 ∧ inscribed_circle_area = 36 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l4064_406415


namespace NUMINAMATH_CALUDE_exists_set_with_150_primes_l4064_406477

/-- The number of primes less than 1000 -/
def primes_lt_1000 : ℕ := 168

/-- Function that counts the number of primes in a set of 2002 consecutive integers starting from n -/
def count_primes (n : ℕ) : ℕ := sorry

theorem exists_set_with_150_primes :
  ∃ n : ℕ, count_primes n = 150 :=
sorry

end NUMINAMATH_CALUDE_exists_set_with_150_primes_l4064_406477


namespace NUMINAMATH_CALUDE_correct_quadratic_not_in_options_l4064_406418

theorem correct_quadratic_not_in_options : ∀ b c : ℝ,
  (∃ x y : ℝ, x + y = 10 ∧ x * y = c) →  -- From the first student's roots
  (∃ u v : ℝ, u + v = b ∧ u * v = -10) →  -- From the second student's roots
  (b = -10 ∧ c = -10) →  -- Derived from the conditions
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x^2 - 10*x - 10 = 0) →
  (x^2 + b*x + c ≠ x^2 - 9*x + 10) ∧
  (x^2 + b*x + c ≠ x^2 + 9*x + 10) ∧
  (x^2 + b*x + c ≠ x^2 - 9*x + 12) ∧
  (x^2 + b*x + c ≠ x^2 + 10*x - 21) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_not_in_options_l4064_406418


namespace NUMINAMATH_CALUDE_decreasing_exponential_function_range_l4064_406453

theorem decreasing_exponential_function_range :
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 →
  (∀ x y : ℝ, x < y → a^x > a^y) →
  a ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_exponential_function_range_l4064_406453


namespace NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l4064_406434

-- Define sets A and B
def A : Set ℝ := {x | 4 * x - 3 > 0}
def B : Set ℝ := {x | x - 6 < 0}

-- State the theorem
theorem union_of_A_and_B_is_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l4064_406434


namespace NUMINAMATH_CALUDE_student_rank_l4064_406491

theorem student_rank (total : Nat) (left_rank : Nat) (right_rank : Nat) : 
  total = 20 → left_rank = 8 → right_rank = total - left_rank + 1 → right_rank = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l4064_406491


namespace NUMINAMATH_CALUDE_squirrel_nut_distance_l4064_406461

theorem squirrel_nut_distance (total_time : ℝ) (speed_without_nut : ℝ) (speed_with_nut : ℝ) 
  (h1 : total_time = 1200)
  (h2 : speed_without_nut = 5)
  (h3 : speed_with_nut = 3) :
  ∃ x : ℝ, x = 2250 ∧ x / speed_without_nut + x / speed_with_nut = total_time :=
by sorry

end NUMINAMATH_CALUDE_squirrel_nut_distance_l4064_406461


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l4064_406463

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (total : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 8 3 4 1 = 576 :=
by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l4064_406463


namespace NUMINAMATH_CALUDE_unique_non_expressible_l4064_406454

/-- Checks if a number can be expressed as x^2 + y^5 for some integers x and y -/
def isExpressible (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + y^5

/-- The list of numbers to check -/
def numberList : List ℤ := [59170, 59149, 59130, 59121, 59012]

/-- Theorem stating that 59121 is the only number in the list that cannot be expressed as x^2 + y^5 -/
theorem unique_non_expressible :
  ∀ n ∈ numberList, n ≠ 59121 → isExpressible n ∧ ¬isExpressible 59121 := by
  sorry

end NUMINAMATH_CALUDE_unique_non_expressible_l4064_406454


namespace NUMINAMATH_CALUDE_valid_selections_count_l4064_406421

/-- The number of male athletes -/
def num_males : ℕ := 4

/-- The number of female athletes -/
def num_females : ℕ := 5

/-- The total number of athletes to be chosen -/
def num_chosen : ℕ := 3

/-- The function to calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of valid selections -/
def total_selections : ℕ := 
  choose num_males 1 * choose num_females 2 + 
  choose num_males 2 * choose num_females 1

theorem valid_selections_count : total_selections = 70 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_count_l4064_406421


namespace NUMINAMATH_CALUDE_snickers_for_nintendo_switch_l4064_406459

def snickers_needed (total_points_needed : ℕ) (chocolate_bunnies_sold : ℕ) (points_per_bunny : ℕ) (points_per_snickers : ℕ) : ℕ :=
  let points_from_bunnies := chocolate_bunnies_sold * points_per_bunny
  let remaining_points := total_points_needed - points_from_bunnies
  remaining_points / points_per_snickers

theorem snickers_for_nintendo_switch : 
  snickers_needed 2000 8 100 25 = 48 := by
  sorry

end NUMINAMATH_CALUDE_snickers_for_nintendo_switch_l4064_406459


namespace NUMINAMATH_CALUDE_complex_subtraction_l4064_406400

theorem complex_subtraction : (4 - 3*I) - (7 - 5*I) = -3 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l4064_406400


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l4064_406468

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 210 →
  8 * a = b - 11 →
  8 * a = c + 11 →
  a * b * c = 4173.75 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l4064_406468


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l4064_406414

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting with 2, ending with 2014, 
    and having a common difference of 4 contains exactly 504 terms -/
theorem arithmetic_sequence_2_to_2014 : 
  arithmetic_sequence_length 2 2014 4 = 504 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l4064_406414


namespace NUMINAMATH_CALUDE_quadratic_range_on_interval_l4064_406490

/-- A quadratic function defined on a closed interval -/
def QuadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The range of a quadratic function on a closed interval -/
def QuadraticRange (a b c : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc (-1 : ℝ) 2, y = QuadraticFunction a b c x}

theorem quadratic_range_on_interval
  (a b c : ℝ) (h : a > 0) :
  QuadraticRange a b c =
    Set.Icc (min (a - b + c) (c - b^2 / (4 * a))) (4 * a + 2 * b + c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_on_interval_l4064_406490


namespace NUMINAMATH_CALUDE_original_number_problem_l4064_406498

theorem original_number_problem (x : ℝ) : x * 1.2 = 480 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l4064_406498


namespace NUMINAMATH_CALUDE_polynomial_sum_l4064_406436

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 160 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l4064_406436


namespace NUMINAMATH_CALUDE_cable_length_l4064_406472

/-- The length of a curve defined by the intersection of a sphere and a plane --/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 →
  x * y + y * z + z * x = 18 →
  ∃ (curve_length : ℝ), curve_length = 4 * Real.pi * Real.sqrt (23 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cable_length_l4064_406472
