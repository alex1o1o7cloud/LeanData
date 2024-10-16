import Mathlib

namespace NUMINAMATH_CALUDE_matrix_equality_l1855_185582

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![10, 6], ![-4, 2]]) : 
  B * A = ![![10, 6], ![-4, 2]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_l1855_185582


namespace NUMINAMATH_CALUDE_order_of_abc_l1855_185522

theorem order_of_abc (a b c : ℝ) (ha : a = 17/18) (hb : b = Real.cos (1/3)) (hc : c = 3 * Real.sin (1/3)) :
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1855_185522


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l1855_185526

theorem point_in_third_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (-m^2 - 1, -1)
  P.1 < 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l1855_185526


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1855_185580

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1855_185580


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1855_185523

/-- Given the relationships between Mickey's, Jayden's, and Coraline's numbers, 
    prove that their sum is 180. -/
theorem sum_of_numbers (mickey jayden coraline : ℕ) 
    (h1 : mickey = jayden + 20)
    (h2 : jayden = coraline - 40)
    (h3 : coraline = 80) : 
  mickey + jayden + coraline = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1855_185523


namespace NUMINAMATH_CALUDE_perfect_fourth_power_in_range_l1855_185539

theorem perfect_fourth_power_in_range : ∃! K : ℤ,
  (K > 0) ∧
  (∃ Z : ℤ, 1000 < Z ∧ Z < 2000 ∧ Z = K * K^3) ∧
  (∃ n : ℤ, K^4 = n^4) :=
by sorry

end NUMINAMATH_CALUDE_perfect_fourth_power_in_range_l1855_185539


namespace NUMINAMATH_CALUDE_valid_sequences_12_l1855_185551

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def valid_sequences (n : ℕ) : ℕ :=
  fibonacci (n + 2)

theorem valid_sequences_12 :
  valid_sequences 12 = 377 :=
by sorry

#eval valid_sequences 12

end NUMINAMATH_CALUDE_valid_sequences_12_l1855_185551


namespace NUMINAMATH_CALUDE_goldfish_preference_total_l1855_185505

theorem goldfish_preference_total : 
  let johnson_class := 30
  let johnson_ratio := (1 : ℚ) / 6
  let feldstein_class := 45
  let feldstein_ratio := (2 : ℚ) / 3
  let henderson_class := 36
  let henderson_ratio := (1 : ℚ) / 5
  let dias_class := 50
  let dias_ratio := (3 : ℚ) / 5
  let norris_class := 25
  let norris_ratio := (2 : ℚ) / 5
  ⌊johnson_class * johnson_ratio⌋ +
  ⌊feldstein_class * feldstein_ratio⌋ +
  ⌊henderson_class * henderson_ratio⌋ +
  ⌊dias_class * dias_ratio⌋ +
  ⌊norris_class * norris_ratio⌋ = 82 :=
by sorry


end NUMINAMATH_CALUDE_goldfish_preference_total_l1855_185505


namespace NUMINAMATH_CALUDE_benny_market_money_l1855_185576

/-- The amount of money Benny took to the market --/
def money_taken : ℕ → ℕ → ℕ → ℕ
  | num_kids, apples_per_kid, cost_per_apple =>
    num_kids * apples_per_kid * cost_per_apple

theorem benny_market_money :
  money_taken 18 5 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_benny_market_money_l1855_185576


namespace NUMINAMATH_CALUDE_college_choices_theorem_l1855_185511

/-- The number of colleges --/
def n : ℕ := 6

/-- The number of colleges to be chosen --/
def k : ℕ := 3

/-- The number of colleges with scheduling conflict --/
def conflict : ℕ := 2

/-- Function to calculate the number of ways to choose colleges --/
def chooseColleges (n k conflict : ℕ) : ℕ :=
  Nat.choose (n - conflict) k + conflict * Nat.choose (n - conflict) (k - 1)

/-- Theorem stating that the number of ways to choose colleges is 16 --/
theorem college_choices_theorem :
  chooseColleges n k conflict = 16 := by sorry

end NUMINAMATH_CALUDE_college_choices_theorem_l1855_185511


namespace NUMINAMATH_CALUDE_limit_f_at_one_l1855_185592

open Real

noncomputable def f (x : ℝ) : ℝ := (2 - x) ^ (sin (π * x / 2) / log (2 - x))

theorem limit_f_at_one : 
  Filter.Tendsto f (nhds 1) (nhds (Real.exp 1)) := by sorry

end NUMINAMATH_CALUDE_limit_f_at_one_l1855_185592


namespace NUMINAMATH_CALUDE_sqrt_10_power_identity_l1855_185500

theorem sqrt_10_power_identity : (Real.sqrt 10 + 3)^2023 * (Real.sqrt 10 - 3)^2022 = Real.sqrt 10 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_power_identity_l1855_185500


namespace NUMINAMATH_CALUDE_solve_for_y_l1855_185564

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - 3 * y = 9) (h2 : x + y = 8) : y = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1855_185564


namespace NUMINAMATH_CALUDE_product_of_absolute_sum_l1855_185555

theorem product_of_absolute_sum (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 / (b*c) + b^2 / (c*a) + c^2 / (a*b) = 1) : 
  (Complex.abs (a + b + c) - 1) * 
  (Complex.abs (a + b + c) - (1 + Real.sqrt 3)) * 
  (Complex.abs (a + b + c) - (1 - Real.sqrt 3)) = 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_absolute_sum_l1855_185555


namespace NUMINAMATH_CALUDE_range_of_a_l1855_185534

def M (a : ℝ) : Set ℝ := { x | -1 < x - a ∧ x - a < 2 }
def N : Set ℝ := { x | x^2 ≥ x }

theorem range_of_a (a : ℝ) : M a ∪ N = Set.univ → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1855_185534


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1855_185556

/-- Given three partners a, b, and c with their capital investments and profit shares,
    prove that the total profit is 16500. -/
theorem total_profit_calculation (a b c : ℕ) (profit_b : ℕ) :
  (2 * a = 3 * b) →  -- Twice a's capital equals thrice b's capital
  (b = 4 * c) →      -- b's capital is 4 times c's capital
  (profit_b = 6000) →  -- b's share of the profit is 6000
  (∃ (total_profit : ℕ), 
    total_profit = 16500 ∧
    total_profit * 4 = profit_b * 11) := by
  sorry

#check total_profit_calculation

end NUMINAMATH_CALUDE_total_profit_calculation_l1855_185556


namespace NUMINAMATH_CALUDE_line_equation_proof_l1855_185518

theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (-2, 1)
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let slope : ℝ := Real.tan angle
  let line_eq := (y - point.2 = slope * (x - point.1))
  line_eq ↔ (y - 1 = Real.sqrt 3 * (x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1855_185518


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1855_185597

theorem trigonometric_equation_solution (x : Real) :
  8.419 * Real.sin x + Real.sqrt (2 - Real.sin x ^ 2) + Real.sin x * Real.sqrt (2 - Real.sin x ^ 2) = 3 ↔
  ∃ k : ℤ, x = π / 2 + 2 * π * k :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1855_185597


namespace NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l1855_185528

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (a^2 - 2a)x + y = 0 -/
def slope1 (a : ℝ) : ℝ := -(a^2 - 2*a)

/-- The slope of the line 3x + y + 1 = 0 -/
def slope2 : ℝ := -3

/-- The lines (a^2 - 2a)x + y = 0 and 3x + y + 1 = 0 are parallel -/
def lines_are_parallel (a : ℝ) : Prop := are_parallel (slope1 a) slope2

theorem a_equals_3_necessary_not_sufficient :
  (∀ a : ℝ, lines_are_parallel a → a = 3) ∧
  ¬(∀ a : ℝ, a = 3 → lines_are_parallel a) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l1855_185528


namespace NUMINAMATH_CALUDE_particle_catch_up_time_l1855_185519

/-- The speed of the first particle in meters per minute -/
def speed_first : ℝ := 5

/-- The time delay between the two particles entering the pipe in minutes -/
def time_delay : ℝ := 6.8

/-- The initial speed of the second particle in meters per minute -/
def initial_speed_second : ℝ := 3

/-- The acceleration of the second particle in meters per minute per minute -/
def acceleration_second : ℝ := 0.5

/-- The time when the second particle catches up with the first -/
def catch_up_time : ℝ := 17

theorem particle_catch_up_time :
  let distance_first (t : ℝ) := speed_first * (t + time_delay)
  let distance_second (t : ℝ) := (initial_speed_second + 0.5 * acceleration_second * (t - 1)) * t / 2
  distance_first catch_up_time = distance_second catch_up_time :=
sorry

#check particle_catch_up_time

end NUMINAMATH_CALUDE_particle_catch_up_time_l1855_185519


namespace NUMINAMATH_CALUDE_no_two_digit_sum_reverse_21_l1855_185578

theorem no_two_digit_sum_reverse_21 : 
  ¬ ∃ (N : ℕ), 
    10 ≤ N ∧ N < 100 ∧ 
    (N + (10 * (N % 10) + N / 10) = 21) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_sum_reverse_21_l1855_185578


namespace NUMINAMATH_CALUDE_closet_probability_l1855_185501

/-- The number of shirts in the closet -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the closet -/
def num_shorts : ℕ := 8

/-- The number of pairs of socks in the closet -/
def num_socks : ℕ := 7

/-- The total number of articles of clothing in the closet -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles to be drawn -/
def draw_count : ℕ := 4

/-- The probability of drawing 2 shirts, 1 pair of shorts, and 1 pair of socks -/
theorem closet_probability : 
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 1 * Nat.choose num_socks 1) / 
  Nat.choose total_articles draw_count = 56 / 399 := by
  sorry

end NUMINAMATH_CALUDE_closet_probability_l1855_185501


namespace NUMINAMATH_CALUDE_total_carrots_l1855_185584

theorem total_carrots (sally fred mary : ℕ) 
  (h1 : sally = 6) 
  (h2 : fred = 4) 
  (h3 : mary = 10) : 
  sally + fred + mary = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l1855_185584


namespace NUMINAMATH_CALUDE_count_valid_quadruples_l1855_185508

def valid_quadruple (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
  a^2 + b^2 + c^2 + d^2 = 9 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81

theorem count_valid_quadruples :
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ q ∈ s, valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2) ∧
    (∀ a b c d, valid_quadruple a b c d → (a, b, c, d) ∈ s) ∧
    s.card = 15 :=
sorry

end NUMINAMATH_CALUDE_count_valid_quadruples_l1855_185508


namespace NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_l1855_185586

-- Sequence 1
def sequence1 (n : ℕ) : ℚ :=
  (-5^n + (-1)^(n-1) * 3 * 2^(n+1)) / (2 * 5^n + (-1)^(n-1) * 2^(n+1))

def sequence1_recurrence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ ∀ n, n ≥ 1 → a (n+1) = (a n + 3) / (2 * a n - 4)

theorem sequence1_correct :
  sequence1_recurrence sequence1 := by sorry

-- Sequence 2
def sequence2 (n : ℕ) : ℚ :=
  (6*n - 11) / (3*n - 4)

def sequence2_recurrence (a : ℕ → ℚ) : Prop :=
  a 1 = 5 ∧ ∀ n, n ≥ 1 → a (n+1) = (a n - 4) / (a n - 3)

theorem sequence2_correct :
  sequence2_recurrence sequence2 := by sorry

end NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_l1855_185586


namespace NUMINAMATH_CALUDE_income_b_is_7200_l1855_185515

/-- Represents the monthly income and expenditure of two individuals -/
structure MonthlyFinances where
  income_ratio : Rat × Rat
  expenditure_ratio : Rat × Rat
  savings_a : ℕ
  savings_b : ℕ

/-- Calculates the monthly income of the second individual given the financial data -/
def calculate_income_b (finances : MonthlyFinances) : ℕ :=
  sorry

/-- Theorem stating that given the specific financial data, the income of b is 7200 -/
theorem income_b_is_7200 (finances : MonthlyFinances) 
  (h1 : finances.income_ratio = (5, 6))
  (h2 : finances.expenditure_ratio = (3, 4))
  (h3 : finances.savings_a = 1800)
  (h4 : finances.savings_b = 1600) :
  calculate_income_b finances = 7200 := by
  sorry

end NUMINAMATH_CALUDE_income_b_is_7200_l1855_185515


namespace NUMINAMATH_CALUDE_total_squares_5x5_with_2_removed_l1855_185550

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)
  (removed : ℕ)

/-- Calculates the total number of squares in a grid --/
def total_squares (g : Grid) : ℕ :=
  sorry

/-- The theorem to prove --/
theorem total_squares_5x5_with_2_removed :
  ∃ (g : Grid), g.size = 5 ∧ g.removed = 2 ∧ total_squares g = 55 :=
sorry

end NUMINAMATH_CALUDE_total_squares_5x5_with_2_removed_l1855_185550


namespace NUMINAMATH_CALUDE_average_marks_is_76_l1855_185510

def english_marks : ℕ := 73
def math_marks : ℕ := 69
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 64
def biology_marks : ℕ := 82

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks_is_76 : (total_marks : ℚ) / num_subjects = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_is_76_l1855_185510


namespace NUMINAMATH_CALUDE_notebooks_left_l1855_185545

theorem notebooks_left (total : ℕ) (yeonju_fraction : ℚ) (minji_fraction : ℚ) : 
  total = 28 → 
  yeonju_fraction = 1/4 →
  minji_fraction = 3/7 →
  total - (yeonju_fraction * total).floor - (minji_fraction * total).floor = 9 := by
sorry

end NUMINAMATH_CALUDE_notebooks_left_l1855_185545


namespace NUMINAMATH_CALUDE_lemon_permutations_l1855_185552

theorem lemon_permutations :
  (Finset.range 5).card.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_lemon_permutations_l1855_185552


namespace NUMINAMATH_CALUDE_barbara_savings_weeks_l1855_185541

/-- Calculates the number of weeks needed to save for a wristwatch -/
def weeks_to_save (watch_cost : ℕ) (weekly_allowance : ℕ) (current_savings : ℕ) : ℕ :=
  ((watch_cost - current_savings) + weekly_allowance - 1) / weekly_allowance

/-- Proves that Barbara needs 16 more weeks to save for the watch -/
theorem barbara_savings_weeks :
  weeks_to_save 100 5 20 = 16 := by
sorry

end NUMINAMATH_CALUDE_barbara_savings_weeks_l1855_185541


namespace NUMINAMATH_CALUDE_people_per_car_l1855_185585

/-- Proves that if 63 people are equally divided among 9 cars, then the number of people in each car is 7. -/
theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (people_per_car : ℕ) 
  (h1 : total_people = 63) 
  (h2 : num_cars = 9) 
  (h3 : total_people = num_cars * people_per_car) : 
  people_per_car = 7 := by
  sorry

end NUMINAMATH_CALUDE_people_per_car_l1855_185585


namespace NUMINAMATH_CALUDE_negative_less_than_positive_l1855_185503

theorem negative_less_than_positive : ∀ x y : ℝ, x < 0 → 0 < y → x < y := by sorry

end NUMINAMATH_CALUDE_negative_less_than_positive_l1855_185503


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1855_185529

theorem sum_of_squares_and_products (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 52)
  (sum_of_products : a*b + b*c + c*a = 24) : 
  a + b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1855_185529


namespace NUMINAMATH_CALUDE_elite_academy_games_l1855_185504

/-- The number of teams in the Elite Academy Basketball League -/
def num_teams : ℕ := 8

/-- The number of times each team plays every other team -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 3

/-- The total number of games in a season for the Elite Academy Basketball League -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem elite_academy_games :
  total_games = 108 := by sorry

end NUMINAMATH_CALUDE_elite_academy_games_l1855_185504


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_range_f_inequality_when_a_zero_l1855_185513

noncomputable section

def f (a x : ℝ) : ℝ := (x - a) * Real.log x - x

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ↔ a ∈ Set.Iic (-1 / Real.exp 1) :=
sorry

theorem f_inequality_when_a_zero (x : ℝ) (hx : x > 0) :
  f 0 x ≥ x * (Real.exp (-x) - 1) - 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_range_f_inequality_when_a_zero_l1855_185513


namespace NUMINAMATH_CALUDE_medians_form_right_triangle_l1855_185532

/-- Given a triangle ABC with sides a, b, c and corresponding medians m_a, m_b, m_c,
    if m_a ⊥ m_b, then m_a^2 + m_b^2 = m_c^2 -/
theorem medians_form_right_triangle (a b c m_a m_b m_c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_medians : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_perp : m_a * m_b = 0) : 
  m_a^2 + m_b^2 = m_c^2 := by
sorry

end NUMINAMATH_CALUDE_medians_form_right_triangle_l1855_185532


namespace NUMINAMATH_CALUDE_shirts_sold_l1855_185599

theorem shirts_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 49 → remaining = 28 → sold = initial - remaining → sold = 21 := by
sorry

end NUMINAMATH_CALUDE_shirts_sold_l1855_185599


namespace NUMINAMATH_CALUDE_student_weights_l1855_185571

theorem student_weights (A B C D : ℕ) : 
  A < B ∧ B < C ∧ C < D →
  A + B = 45 →
  A + C = 49 →
  B + C = 54 →
  B + D = 60 →
  C + D = 64 →
  D = 35 := by
sorry

end NUMINAMATH_CALUDE_student_weights_l1855_185571


namespace NUMINAMATH_CALUDE_coincidence_time_l1855_185568

-- Define the movement pattern
def move_distance (n : ℕ) : ℤ := if n % 2 = 0 then -n else n

-- Define the position after n moves
def position (n : ℕ) : ℤ := (List.range n).map move_distance |>.sum

-- Define the total distance traveled after n moves
def total_distance (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the speed
def speed : ℕ := 4

-- Define the position of point A
def point_A : ℤ := -24

-- Theorem to prove
theorem coincidence_time :
  ∃ n : ℕ, position n = point_A ∧ (total_distance n / speed : ℚ) = 294 := by
  sorry


end NUMINAMATH_CALUDE_coincidence_time_l1855_185568


namespace NUMINAMATH_CALUDE_function_s_is_identity_max_value_12345_l1855_185546

/-- A non-zero real-valued function satisfying the given functional equation. -/
def FunctionS (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≠ 0) ∧
  (∀ x y z : ℝ, f (x^2 + y * f z) = x * f x + z * f y)

/-- The theorem stating that any function in S is the identity function. -/
theorem function_s_is_identity (f : ℝ → ℝ) (hf : FunctionS f) :
  ∀ x : ℝ, f x = x :=
sorry

/-- The maximum value of f(12345) for f in S is 12345. -/
theorem max_value_12345 (f : ℝ → ℝ) (hf : FunctionS f) :
  f 12345 = 12345 :=
sorry

end NUMINAMATH_CALUDE_function_s_is_identity_max_value_12345_l1855_185546


namespace NUMINAMATH_CALUDE_donation_calculation_l1855_185547

/-- Calculates the total donation amount for a person who started donating at age 13 and is now 33 years old, donating 5k each year. -/
def total_donation (start_age : ℕ) (current_age : ℕ) (annual_donation : ℕ) : ℕ :=
  (current_age - start_age) * annual_donation

/-- Theorem stating that the total donation for a person starting at age 13, now 33, donating 5k annually is 100k. -/
theorem donation_calculation (start_age : ℕ) (current_age : ℕ) (annual_donation : ℕ) 
  (h1 : start_age = 13) 
  (h2 : current_age = 33) 
  (h3 : annual_donation = 5000) : 
  total_donation start_age current_age annual_donation = 100000 := by
  sorry

#eval total_donation 13 33 5000

end NUMINAMATH_CALUDE_donation_calculation_l1855_185547


namespace NUMINAMATH_CALUDE_semicircle_sum_limit_l1855_185521

/-- Theorem: As the number of divisions approaches infinity, the sum of the lengths of semicircles
    constructed on equal parts of a circle's diameter approaches the semi-circumference of the original circle. -/
theorem semicircle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * (D / n) / 2) - π * D / 2| < ε :=
sorry

end NUMINAMATH_CALUDE_semicircle_sum_limit_l1855_185521


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l1855_185533

theorem greatest_integer_difference (x y : ℝ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l1855_185533


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1855_185560

/-- The repeating decimal 0.868686... -/
def repeating_decimal : ℚ := 0.868686

/-- The fraction 86/99 -/
def fraction : ℚ := 86 / 99

/-- Theorem stating that the repeating decimal 0.868686... equals the fraction 86/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1855_185560


namespace NUMINAMATH_CALUDE_prob_odd_then_even_eq_17_45_l1855_185583

/-- A box containing 6 cards numbered 1 to 6 -/
def Box : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of drawing a specific card from the box -/
def prob_draw (n : ℕ) : ℚ := if n ∈ Box then 1 / 6 else 0

/-- The probability of drawing an even number from the remaining cards after drawing 'a' -/
def prob_even_after (a : ℕ) : ℚ :=
  let remaining := Box.filter (λ x => x > a)
  let even_remaining := remaining.filter (λ x => x % 2 = 0)
  (even_remaining.card : ℚ) / remaining.card

/-- The probability of the event: first draw is odd and second draw is even -/
def prob_odd_then_even : ℚ :=
  (prob_draw 1 * prob_even_after 1) +
  (prob_draw 3 * prob_even_after 3) +
  (prob_draw 5 * prob_even_after 5)

theorem prob_odd_then_even_eq_17_45 : prob_odd_then_even = 17 / 45 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_then_even_eq_17_45_l1855_185583


namespace NUMINAMATH_CALUDE_group_size_solve_group_size_l1855_185530

/-- The number of persons in the group -/
def n : ℕ := sorry

/-- The age of the replaced person -/
def replaced_age : ℕ := 45

/-- The age of the new person -/
def new_age : ℕ := 15

/-- The decrease in average age -/
def avg_decrease : ℕ := 3

theorem group_size :
  (n * replaced_age - (replaced_age - new_age)) = (n * (replaced_age - avg_decrease)) :=
sorry

theorem solve_group_size : n = 10 :=
sorry

end NUMINAMATH_CALUDE_group_size_solve_group_size_l1855_185530


namespace NUMINAMATH_CALUDE_ship_supplies_l1855_185572

theorem ship_supplies (x : ℝ) : 
  x > 0 →
  (x - 2/5 * x) * (1 - 3/5) = 96 →
  x = 400 :=
by sorry

end NUMINAMATH_CALUDE_ship_supplies_l1855_185572


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1855_185507

theorem sphere_surface_area (V : ℝ) (r : ℝ) (h : V = 72 * Real.pi) :
  (4 * Real.pi * r^2 : ℝ) = 36 * 2^(2/3) * Real.pi ↔ (4/3 * Real.pi * r^3 : ℝ) = V := by
  sorry

#check sphere_surface_area

end NUMINAMATH_CALUDE_sphere_surface_area_l1855_185507


namespace NUMINAMATH_CALUDE_max_value_fraction_l1855_185520

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -3 ≤ x' ∧ x' ≤ -1 → 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1855_185520


namespace NUMINAMATH_CALUDE_janet_earnings_per_hour_l1855_185525

-- Define the payment rates for each type of post
def text_post_rate : ℚ := 0.25
def image_post_rate : ℚ := 0.30
def video_post_rate : ℚ := 0.40

-- Define the number of posts checked in an hour
def text_posts_per_hour : ℕ := 130
def image_posts_per_hour : ℕ := 90
def video_posts_per_hour : ℕ := 30

-- Define the USD to EUR exchange rate
def usd_to_eur_rate : ℚ := 0.85

-- Calculate the earnings per hour in EUR
def earnings_per_hour_eur : ℚ :=
  (text_post_rate * text_posts_per_hour +
   image_post_rate * image_posts_per_hour +
   video_post_rate * video_posts_per_hour) * usd_to_eur_rate

-- Theorem to prove
theorem janet_earnings_per_hour :
  earnings_per_hour_eur = 60.775 := by sorry

end NUMINAMATH_CALUDE_janet_earnings_per_hour_l1855_185525


namespace NUMINAMATH_CALUDE_football_team_size_l1855_185598

/-- The number of players on a football team -/
def total_players : ℕ := 70

/-- The number of throwers on the team -/
def throwers : ℕ := 46

/-- The number of right-handed players on the team -/
def right_handed : ℕ := 62

/-- All throwers are right-handed -/
axiom throwers_are_right_handed : throwers ≤ right_handed

/-- One third of non-throwers are left-handed -/
axiom one_third_left_handed :
  3 * (total_players - throwers - (right_handed - throwers)) = total_players - throwers

theorem football_team_size :
  total_players = 70 :=
sorry

end NUMINAMATH_CALUDE_football_team_size_l1855_185598


namespace NUMINAMATH_CALUDE_no_integer_solution_l1855_185554

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), x * (x - y) + y * (y - z) + z * (z - x) = 3 ∧ x > y ∧ y > z :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1855_185554


namespace NUMINAMATH_CALUDE_fault_line_current_movement_l1855_185596

/-- The movement of a fault line over two years -/
structure FaultLineMovement where
  total : ℝ  -- Total movement over two years
  previous : ℝ  -- Movement in the previous year
  current : ℝ  -- Movement in the current year

/-- Theorem: Given the total movement and previous year's movement, 
    calculate the current year's movement -/
theorem fault_line_current_movement 
  (f : FaultLineMovement) 
  (h1 : f.total = 6.5) 
  (h2 : f.previous = 5.25) 
  (h3 : f.total = f.previous + f.current) : 
  f.current = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_current_movement_l1855_185596


namespace NUMINAMATH_CALUDE_sum_squares_consecutive_even_numbers_l1855_185591

/-- Given 6 consecutive even numbers with a sum of 72, prove that the sum of their squares is 1420 -/
theorem sum_squares_consecutive_even_numbers :
  ∀ (a : ℕ), 
  (∃ (n : ℕ), a = 2*n) →  -- a is even
  (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) + (a + 10) = 72) →  -- sum is 72
  (a^2 + (a + 2)^2 + (a + 4)^2 + (a + 6)^2 + (a + 8)^2 + (a + 10)^2 = 1420) :=
by sorry


end NUMINAMATH_CALUDE_sum_squares_consecutive_even_numbers_l1855_185591


namespace NUMINAMATH_CALUDE_president_and_vice_captain_selection_l1855_185548

/-- The number of people to choose from -/
def n : ℕ := 5

/-- The number of positions to fill -/
def k : ℕ := 2

/-- Theorem: The number of ways to select a class president and a vice-captain 
    from a group of n people, where one person cannot hold both positions, 
    is equal to n * (n - 1) -/
theorem president_and_vice_captain_selection : n * (n - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_president_and_vice_captain_selection_l1855_185548


namespace NUMINAMATH_CALUDE_cube_shape_product_l1855_185559

/-- Represents a 3D shape constructed from identical cubes. -/
structure CubeShape where
  /-- The number of cubes in the shape. -/
  num_cubes : ℕ
  /-- Predicate that returns true if the shape satisfies the given views. -/
  satisfies_views : Bool

/-- The minimum number of cubes that can form the shape satisfying the given views. -/
def min_cubes : ℕ := 8

/-- The maximum number of cubes that can form the shape satisfying the given views. -/
def max_cubes : ℕ := 16

/-- Theorem stating that the product of the maximum and minimum number of cubes is 128. -/
theorem cube_shape_product :
  min_cubes * max_cubes = 128 ∧
  ∀ shape : CubeShape, shape.satisfies_views →
    min_cubes ≤ shape.num_cubes ∧ shape.num_cubes ≤ max_cubes :=
by sorry

end NUMINAMATH_CALUDE_cube_shape_product_l1855_185559


namespace NUMINAMATH_CALUDE_f_zero_gt_f_four_l1855_185561

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State that f is differentiable on ℝ
variable (hf : Differentiable ℝ f)

-- Define the condition that f(x) = x² + 2f''(2)x - 3
variable (hf_eq : ∀ x, f x = x^2 + 2 * (deriv^[2] f 2) * x - 3)

-- Theorem to prove
theorem f_zero_gt_f_four : f 0 > f 4 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_gt_f_four_l1855_185561


namespace NUMINAMATH_CALUDE_prob_at_least_twice_eq_target_l1855_185562

/-- The probability of hitting a target in one shot -/
def p : ℝ := 0.6

/-- The number of shots taken -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots -/
def prob_at_least_twice (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * p^2 * (1 - p) + (n.choose 3) * p^3

theorem prob_at_least_twice_eq_target : 
  prob_at_least_twice p n = 0.648 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_twice_eq_target_l1855_185562


namespace NUMINAMATH_CALUDE_candy_ratio_is_three_l1855_185535

/-- The ratio of Jennifer's candies to Bob's candies -/
def candy_ratio (emily_candies bob_candies : ℕ) : ℚ :=
  (2 * emily_candies) / bob_candies

/-- Theorem: The ratio of Jennifer's candies to Bob's candies is 3 -/
theorem candy_ratio_is_three :
  candy_ratio 6 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_is_three_l1855_185535


namespace NUMINAMATH_CALUDE_condition_iff_prime_l1855_185577

def satisfies_condition (n : ℕ) : Prop :=
  (n = 2) ∨ (n > 2 ∧ ∀ k : ℕ, 2 ≤ k → k < n → ¬(k ∣ n))

theorem condition_iff_prime (n : ℕ) : satisfies_condition n ↔ Nat.Prime n :=
  sorry

end NUMINAMATH_CALUDE_condition_iff_prime_l1855_185577


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l1855_185502

theorem sandy_correct_sums :
  ∀ (c i : ℕ),
  c + i = 30 →
  3 * c - 2 * i = 55 →
  c = 23 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l1855_185502


namespace NUMINAMATH_CALUDE_a_values_l1855_185574

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem a_values (h : ∀ a : ℝ, B a ⊆ A) : 
  {a : ℝ | B a ⊆ A} = {0, 1/3, 1/5} := by sorry

end NUMINAMATH_CALUDE_a_values_l1855_185574


namespace NUMINAMATH_CALUDE_differential_savings_proof_l1855_185569

def calculate_differential_savings (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  income * (old_rate - new_rate)

theorem differential_savings_proof (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) 
  (h1 : income = 48000)
  (h2 : old_rate = 0.45)
  (h3 : new_rate = 0.30) :
  calculate_differential_savings income old_rate new_rate = 7200 := by
  sorry

end NUMINAMATH_CALUDE_differential_savings_proof_l1855_185569


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1855_185538

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) :
  (∃ k : ℕ, k ≥ 2 ∧
    Nat.choose n k * x^(n - k) * a^k = 210 ∧
    Nat.choose n (k + 1) * x^(n - k - 1) * a^(k + 1) = 504 ∧
    Nat.choose n (k + 2) * x^(n - k - 2) * a^(k + 2) = 1260) →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1855_185538


namespace NUMINAMATH_CALUDE_corn_donation_l1855_185531

def total_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

theorem corn_donation :
  let total_ears := total_bushels * ears_per_bushel
  let given_away_ears := total_ears - remaining_ears
  given_away_ears / ears_per_bushel = 24 :=
by sorry

end NUMINAMATH_CALUDE_corn_donation_l1855_185531


namespace NUMINAMATH_CALUDE_willie_had_48_bananas_l1855_185557

/-- Given the total number of bananas and Charles' initial bananas, 
    calculate Willie's initial bananas. -/
def willies_bananas (total : ℝ) (charles_initial : ℝ) : ℝ :=
  total - charles_initial

/-- Theorem stating that Willie had 48.0 bananas given the problem conditions. -/
theorem willie_had_48_bananas : 
  willies_bananas 83 35 = 48 := by
  sorry

#eval willies_bananas 83 35

end NUMINAMATH_CALUDE_willie_had_48_bananas_l1855_185557


namespace NUMINAMATH_CALUDE_multiply_subtract_distribute_compute_expression_l1855_185581

theorem multiply_subtract_distribute (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_distribute_compute_expression_l1855_185581


namespace NUMINAMATH_CALUDE_min_delivery_time_l1855_185524

theorem min_delivery_time (n : Nat) (hn : n = 63) :
  let S := Fin n → Fin n
  (∃ (f : S), Function.Bijective f) →
  (∀ (f : S), Function.Bijective f →
    (∃ (i : Fin n), (i.val + 1) * (f i).val + 1 ≥ 1024)) ∧
  (∃ (f : S), Function.Bijective f ∧
    ∀ (i : Fin n), (i.val + 1) * (f i).val + 1 ≤ 1024) :=
by sorry

end NUMINAMATH_CALUDE_min_delivery_time_l1855_185524


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1855_185516

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1855_185516


namespace NUMINAMATH_CALUDE_nate_running_distance_l1855_185588

/-- The total distance Nate ran given the length of a football field and additional distance -/
def total_distance (field_length : ℝ) (additional_distance : ℝ) : ℝ :=
  4 * field_length + additional_distance

/-- Theorem stating that Nate's total running distance is 1172 meters -/
theorem nate_running_distance :
  total_distance 168 500 = 1172 := by
  sorry

end NUMINAMATH_CALUDE_nate_running_distance_l1855_185588


namespace NUMINAMATH_CALUDE_blocks_with_one_face_painted_10_2_l1855_185514

/-- Represents a cube made of smaller blocks -/
structure BlockCube where
  largeSideLength : ℕ
  smallSideLength : ℕ
  
/-- Calculates the number of blocks with only one face painted -/
def BlockCube.blocksWithOneFacePainted (cube : BlockCube) : ℕ :=
  let blocksPerEdge := cube.largeSideLength / cube.smallSideLength
  let surfaceBlocks := 6 * blocksPerEdge * blocksPerEdge
  let edgeBlocks := 12 * blocksPerEdge - 24
  surfaceBlocks - edgeBlocks - 8

theorem blocks_with_one_face_painted_10_2 :
  (BlockCube.blocksWithOneFacePainted { largeSideLength := 10, smallSideLength := 2 }) = 54 := by
  sorry

end NUMINAMATH_CALUDE_blocks_with_one_face_painted_10_2_l1855_185514


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l1855_185594

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  ∃ d, c - b = b - a ∧ b - a = d →  -- a, b, c form an arithmetic sequence
  a * b * c = 64 →  -- product is 64
  b ≥ 4 ∧ ∃ (a' b' c' : ℝ), a' * b' * c' = 64 ∧ b' = 4 ∧ 
    ∃ d', c' - b' = b' - a' ∧ b' - a' = d' :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l1855_185594


namespace NUMINAMATH_CALUDE_solution_sum_l1855_185595

theorem solution_sum (c d : ℝ) : 
  c^2 - 6*c + 15 = 27 →
  d^2 - 6*d + 15 = 27 →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l1855_185595


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l1855_185590

theorem sqrt_three_squared : Real.sqrt 3 * Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l1855_185590


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1855_185573

theorem rectangle_area_diagonal_relation (length width perimeter diagonal area : ℝ) 
  (h_ratio : length / width = 5 / 2)
  (h_perimeter : perimeter = 2 * (length + width))
  (h_perimeter_value : perimeter = 28)
  (h_diagonal : diagonal^2 = length^2 + width^2)
  (h_area : area = length * width) :
  ∃ k : ℝ, k = 10 / 29 ∧ area = k * diagonal^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1855_185573


namespace NUMINAMATH_CALUDE_sin_cos_shift_l1855_185575

theorem sin_cos_shift (x : ℝ) : Real.cos (2 * (x - Real.pi / 8) - Real.pi / 4) = Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l1855_185575


namespace NUMINAMATH_CALUDE_tangent_roots_sine_cosine_ratio_l1855_185509

theorem tangent_roots_sine_cosine_ratio (α β p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (Real.sin (α + β)) / (Real.cos (α - β)) = -p / (q + 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_roots_sine_cosine_ratio_l1855_185509


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1855_185566

/-- The area of a triangle with sides 16, 30, and 34 is 240 -/
theorem triangle_area : ℝ → Prop :=
  fun a : ℝ =>
    let s1 : ℝ := 16
    let s2 : ℝ := 30
    let s3 : ℝ := 34
    (s1 * s1 + s2 * s2 = s3 * s3) →  -- Pythagorean theorem condition
    (a = (1 / 2) * s1 * s2) →        -- Area formula for right triangle
    a = 240

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 240 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1855_185566


namespace NUMINAMATH_CALUDE_steve_final_marbles_l1855_185558

/-- Represents the number of marbles each person has -/
structure Marbles where
  sam : ℕ
  sally : ℕ
  steve : ℕ

/-- The initial state of marbles -/
def initial_state : Marbles :=
  { sam := 14,
    sally := 9,
    steve := 7 }

/-- The final state of marbles after Sam gives 3 to Sally and Steve each -/
def final_state : Marbles :=
  { sam := initial_state.sam - 6,
    sally := initial_state.sally + 3,
    steve := initial_state.steve + 3 }

theorem steve_final_marbles :
  (initial_state.sam = 2 * initial_state.steve) →
  (initial_state.sally = initial_state.sam - 5) →
  (final_state.sam = 8) →
  (final_state.steve = 10) := by
  sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l1855_185558


namespace NUMINAMATH_CALUDE_frank_apples_l1855_185553

theorem frank_apples (frank : ℕ) (susan : ℕ) : 
  susan = 3 * frank →  -- Susan picked 3 times as many apples as Frank
  (2 * frank / 3 + 3 * susan / 2 : ℚ) = 78 →  -- Remaining apples after Frank sold 1/3 and Susan gave out 1/2
  frank = 36 := by
sorry

end NUMINAMATH_CALUDE_frank_apples_l1855_185553


namespace NUMINAMATH_CALUDE_smallest_a_for_quadratic_roots_l1855_185570

theorem smallest_a_for_quadratic_roots (a : ℕ) (b c : ℝ) : 
  (∃ x y : ℝ, 
    x ≠ y ∧ 
    0 < x ∧ x ≤ 1/1000 ∧ 
    0 < y ∧ y ≤ 1/1000 ∧ 
    a * x^2 + b * x + c = 0 ∧ 
    a * y^2 + b * y + c = 0) →
  a ≥ 1001000 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_quadratic_roots_l1855_185570


namespace NUMINAMATH_CALUDE_worker_completion_time_proof_l1855_185544

/-- Represents a worker with their working days and payment --/
structure Worker where
  days : ℕ
  payment : ℕ

/-- Calculates the time it would take a worker to complete the entire job --/
def timeToCompleteJob (w : Worker) (totalPayment : ℕ) : ℕ :=
  w.days * (totalPayment / w.payment)

theorem worker_completion_time_proof (w1 w2 w3 : Worker) 
  (h1 : w1.days = 6 ∧ w1.payment = 36)
  (h2 : w2.days = 3 ∧ w2.payment = 12)
  (h3 : w3.days = 8 ∧ w3.payment = 24) :
  let totalPayment := w1.payment + w2.payment + w3.payment
  (timeToCompleteJob w1 totalPayment = 12) ∧
  (timeToCompleteJob w2 totalPayment = 18) ∧
  (timeToCompleteJob w3 totalPayment = 24) := by
  sorry

#check worker_completion_time_proof

end NUMINAMATH_CALUDE_worker_completion_time_proof_l1855_185544


namespace NUMINAMATH_CALUDE_unique_x_with_703_factors_l1855_185587

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- x^x has exactly 703 positive factors -/
def has_703_factors (x : ℕ) : Prop :=
  num_factors (x^x) = 703

theorem unique_x_with_703_factors :
  ∃! x : ℕ, x > 0 ∧ has_703_factors x ∧ x = 18 := by sorry

end NUMINAMATH_CALUDE_unique_x_with_703_factors_l1855_185587


namespace NUMINAMATH_CALUDE_quadratic_roots_l1855_185565

theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  (2*(a + b))^2 - 4*3*a*(b + c) > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1855_185565


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l1855_185506

theorem rational_coefficient_terms_count : ℕ :=
  let expansion := (fun (x y : ℝ) => x * Real.rpow 3 (1/4) + y * Real.rpow 5 (1/3)) ^ 400
  let total_terms := 401
  let rational_coeff_count := Finset.filter (fun k => 
    (k % 4 = 0) ∧ ((400 - k) % 3 = 0)
  ) (Finset.range (total_terms))
  34

#check rational_coefficient_terms_count

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l1855_185506


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1855_185536

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area (a b h x : ℝ) (hb : b > a) (hh : h > 0) (hx : 0 < x ∧ x < h) :
  let rectangle_area := (b - a) * x * (h - x) / h
  rectangle_area = (b - a) * x * (h - x) / h := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1855_185536


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l1855_185579

theorem largest_n_for_unique_k : ∃ (n : ℕ), n > 0 ∧ 
  (∃! (k : ℤ), (9 : ℚ)/17 < n/(n + k) ∧ n/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < m/(m + k) ∧ m/(m + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l1855_185579


namespace NUMINAMATH_CALUDE_range_of_a_when_P_or_Q_false_l1855_185567

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Define proposition P
def P (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- Define the set of a values where either P or Q is false
def A : Set ℝ := {a | -1 < a ∧ a < 0 ∨ 0 < a ∧ a < 1}

-- Theorem statement
theorem range_of_a_when_P_or_Q_false :
  ∀ a : ℝ, (¬P a ∨ ¬Q a) ↔ a ∈ A :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_P_or_Q_false_l1855_185567


namespace NUMINAMATH_CALUDE_charles_whistle_count_l1855_185542

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end NUMINAMATH_CALUDE_charles_whistle_count_l1855_185542


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1855_185527

/-- Proves that given a principal of 5000, if increasing the interest rate by 3%
    results in 300 more interest over the same time period, then the time period is 2 years. -/
theorem simple_interest_problem (R : ℚ) (T : ℚ) : 
  (5000 * (R + 3) / 100 * T = 5000 * R / 100 * T + 300) → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1855_185527


namespace NUMINAMATH_CALUDE_equation_is_linear_l1855_185537

def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (A B C : ℝ), ∀ x y, f x y = A * x + B * y + C

def equation (x y : ℝ) : ℝ := 2 * x + 3 * y - 4

theorem equation_is_linear :
  is_linear_equation_in_two_variables equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_l1855_185537


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1855_185517

theorem min_value_quadratic_form (x y : ℝ) : 2 * x^2 + 3 * x * y + 4 * y^2 + 5 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1855_185517


namespace NUMINAMATH_CALUDE_fox_catches_rabbits_l1855_185563

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the game setup -/
structure GameSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  foxSpeed : ℝ
  rabbitSpeed : ℝ

/-- Checks if the fox can catch both rabbits -/
def canCatchBothRabbits (setup : GameSetup) : Prop :=
  setup.foxSpeed ≥ 1 + Real.sqrt 2

theorem fox_catches_rabbits (setup : GameSetup) 
  (h1 : setup.A = ⟨0, 0⟩) 
  (h2 : setup.B = ⟨1, 0⟩) 
  (h3 : setup.C = ⟨1, 1⟩) 
  (h4 : setup.D = ⟨0, 1⟩)
  (h5 : setup.rabbitSpeed = 1) :
  canCatchBothRabbits setup ↔ 
    ∀ (t : ℝ), t ≥ 0 → 
      ∃ (foxPos : Point),
        (foxPos.x - setup.C.x)^2 + (foxPos.y - setup.C.y)^2 ≤ (setup.foxSpeed * t)^2 ∧
        ((foxPos.x = setup.B.x + t ∧ foxPos.y = 0) ∨
         (foxPos.x = 0 ∧ foxPos.y = setup.D.y + t) ∨
         (foxPos.x = 0 ∧ foxPos.y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_fox_catches_rabbits_l1855_185563


namespace NUMINAMATH_CALUDE_percentage_relationship_l1855_185543

theorem percentage_relationship (a b : ℝ) (h : a = 1.5 * b) :
  3 * b = 2 * a := by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1855_185543


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1855_185512

theorem tan_alpha_value (α : Real) (h : Real.tan α = 3/4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64/25 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1855_185512


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l1855_185540

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0 and f(1-x) = f(1+x),
    prove that f(3^x) > f(2^x) for all x > 0 -/
theorem quadratic_function_inequality (a b c : ℝ) (x : ℝ) 
  (h1 : a > 0) 
  (h2 : ∀ y, a*(1-y)^2 + b*(1-y) + c = a*(1+y)^2 + b*(1+y) + c) 
  (h3 : x > 0) : 
  a*(3^x)^2 + b*(3^x) + c > a*(2^x)^2 + b*(2^x) + c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l1855_185540


namespace NUMINAMATH_CALUDE_rohit_final_position_l1855_185593

/-- Represents a direction in 2D space -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position in 2D space -/
structure Position where
  x : Int
  y : Int

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : Nat
  direction : Direction

/-- Function to update position based on a movement -/
def updatePosition (pos : Position) (move : Movement) : Position :=
  match move.direction with
  | Direction.North => { x := pos.x,     y := pos.y + move.distance }
  | Direction.South => { x := pos.x,     y := pos.y - move.distance }
  | Direction.East  => { x := pos.x + move.distance, y := pos.y }
  | Direction.West  => { x := pos.x - move.distance, y := pos.y }

/-- Function to turn left -/
def turnLeft (dir : Direction) : Direction :=
  match dir with
  | Direction.North => Direction.West
  | Direction.West  => Direction.South
  | Direction.South => Direction.East
  | Direction.East  => Direction.North

/-- Function to turn right -/
def turnRight (dir : Direction) : Direction :=
  match dir with
  | Direction.North => Direction.East
  | Direction.East  => Direction.South
  | Direction.South => Direction.West
  | Direction.West  => Direction.North

/-- Rohit's movements -/
def rohitMovements : List Movement :=
  [ { distance := 25, direction := Direction.South },
    { distance := 20, direction := Direction.East },
    { distance := 25, direction := Direction.North },
    { distance := 15, direction := Direction.East } ]

theorem rohit_final_position (startPos : Position := { x := 0, y := 0 }) :
  let finalPos := rohitMovements.foldl updatePosition startPos
  finalPos.x = 35 ∧ finalPos.y = 0 := by sorry

end NUMINAMATH_CALUDE_rohit_final_position_l1855_185593


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1855_185549

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  h1 : a 3 = 10
  h2 : a 12 = 31

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 1 = 16/3) ∧ 
  (∀ n : ℕ, seq.a (n + 1) - seq.a n = 7/3) ∧
  (∀ n : ℕ, seq.a n = 7/3 * n + 3) ∧
  (seq.a 18 = 45) ∧
  (∀ n : ℕ, seq.a n ≠ 85) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1855_185549


namespace NUMINAMATH_CALUDE_pqr_value_exists_l1855_185589

theorem pqr_value_exists :
  ∃ (p q r : ℝ), (p * q) * (q * r) * (r * p) = 16 ∧ p * q * r = 4 :=
by sorry

end NUMINAMATH_CALUDE_pqr_value_exists_l1855_185589
