import Mathlib

namespace NUMINAMATH_CALUDE_a_plus_b_equals_34_l606_60667

theorem a_plus_b_equals_34 (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 30) / (x - 3)) →
  A + B = 34 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_34_l606_60667


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_typist_salary_problem_l606_60655

theorem salary_decrease_percentage 
  (original_salary : ℝ) 
  (increase_percentage : ℝ) 
  (final_salary : ℝ) : ℝ :=
  let increased_salary := original_salary * (1 + increase_percentage / 100)
  let decrease_percentage := (increased_salary - final_salary) / increased_salary * 100
  decrease_percentage

theorem typist_salary_problem : 
  salary_decrease_percentage 2000 10 2090 = 5 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_typist_salary_problem_l606_60655


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l606_60650

/-- The area of a regular hexagon inscribed in a circle -/
theorem inscribed_hexagon_area (circle_area : ℝ) (h : circle_area = 100 * Real.pi) :
  let r := (circle_area / Real.pi).sqrt
  let hexagon_area := 6 * (r^2 * Real.sqrt 3 / 4)
  hexagon_area = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l606_60650


namespace NUMINAMATH_CALUDE_m_arit_fib_seq_periodic_l606_60663

/-- An m-arithmetic Fibonacci sequence -/
def MAritFibSeq (m : ℕ) := ℕ → Fin m

/-- The period of an m-arithmetic Fibonacci sequence -/
def Period (m : ℕ) (v : MAritFibSeq m) (r : ℕ) : Prop :=
  ∀ n, v n = v (n + r)

theorem m_arit_fib_seq_periodic (m : ℕ) (v : MAritFibSeq m) :
  ∃ r : ℕ, r ≤ m^2 ∧ Period m v r := by
  sorry

end NUMINAMATH_CALUDE_m_arit_fib_seq_periodic_l606_60663


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l606_60604

theorem max_value_of_sum_of_roots (a b c : ℝ) 
  (sum_eq_two : a + b + c = 2)
  (a_ge_neg_one : a ≥ -1)
  (b_ge_neg_two : b ≥ -2)
  (c_ge_neg_three : c ≥ -3) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 6 ∧
    ∀ (x : ℝ), x = Real.sqrt (4 * a + 2) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 14) → x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l606_60604


namespace NUMINAMATH_CALUDE_fib_mod_10_periodic_fib_mod_10_smallest_period_l606_60606

/-- Fibonacci sequence modulo 10 -/
def fib_mod_10 : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (fib_mod_10 n + fib_mod_10 (n + 1)) % 10

/-- The period of the Fibonacci sequence modulo 10 -/
def fib_mod_10_period : ℕ := 60

/-- Theorem: The Fibonacci sequence modulo 10 has a period of 60 -/
theorem fib_mod_10_periodic :
  ∀ n : ℕ, fib_mod_10 (n + fib_mod_10_period) = fib_mod_10 n :=
by
  sorry

/-- Theorem: 60 is the smallest positive period of the Fibonacci sequence modulo 10 -/
theorem fib_mod_10_smallest_period :
  ∀ k : ℕ, k > 0 → k < fib_mod_10_period →
    ∃ n : ℕ, fib_mod_10 (n + k) ≠ fib_mod_10 n :=
by
  sorry

end NUMINAMATH_CALUDE_fib_mod_10_periodic_fib_mod_10_smallest_period_l606_60606


namespace NUMINAMATH_CALUDE_max_value_of_expression_l606_60669

theorem max_value_of_expression (x : ℝ) : 
  x^4 / (x^8 + 4*x^6 + 2*x^4 + 8*x^2 + 16) ≤ 1/31 ∧ 
  ∃ y : ℝ, y^4 / (y^8 + 4*y^6 + 2*y^4 + 8*y^2 + 16) = 1/31 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l606_60669


namespace NUMINAMATH_CALUDE_janet_snowball_percentage_l606_60616

/-- The number of snowballs Janet made -/
def janet_snowballs : ℕ := 50

/-- The number of snowballs Janet's brother made -/
def brother_snowballs : ℕ := 150

/-- The total number of snowballs made -/
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

/-- The percentage of snowballs Janet made -/
def janet_percentage : ℚ := (janet_snowballs : ℚ) / (total_snowballs : ℚ) * 100

theorem janet_snowball_percentage : janet_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_snowball_percentage_l606_60616


namespace NUMINAMATH_CALUDE_binomial_18_choose_6_l606_60621

theorem binomial_18_choose_6 : Nat.choose 18 6 = 13260 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_6_l606_60621


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l606_60618

/-- The minimum distance between two points on parallel lines -/
theorem min_distance_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y => x + 3 * y - 9 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => x + 3 * y + 1 = 0
  ∃ (d : ℝ), d = Real.sqrt 10 ∧
    ∀ (P₁ P₂ : ℝ × ℝ), l₁ P₁.1 P₁.2 → l₂ P₂.1 P₂.2 →
      Real.sqrt ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parallel_lines_l606_60618


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l606_60631

-- Define the points
def p1 : ℝ × ℝ := (1, 3)
def p2 : ℝ × ℝ := (5, -1)
def p3 : ℝ × ℝ := (10, 3)
def p4 : ℝ × ℝ := (5, 7)

-- Define the ellipse
def ellipse (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let center := ((p1.1 + p3.1) / 2, (p1.2 + p3.2) / 2)
  let a := (p3.1 - p1.1) / 2
  let b := (p4.2 - p2.2) / 2
  (center.1 = (p2.1 + p4.1) / 2) ∧ 
  (center.2 = (p2.2 + p4.2) / 2) ∧
  (a > b) ∧ (b > 0)

-- Theorem statement
theorem ellipse_foci_distance (h : ellipse p1 p2 p3 p4) :
  let a := (p3.1 - p1.1) / 2
  let b := (p4.2 - p2.2) / 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 4.25 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l606_60631


namespace NUMINAMATH_CALUDE_smallest_solution_l606_60627

-- Define the equation
def equation (x : ℝ) : Prop := x * (abs x) + 3 * x = 5 * x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | equation x}

-- State the theorem
theorem smallest_solution :
  ∃ (x : ℝ), x ∈ solution_set ∧ ∀ (y : ℝ), y ∈ solution_set → x ≤ y ∧ x = -1 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l606_60627


namespace NUMINAMATH_CALUDE_ellipse_complementary_angles_point_l606_60625

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

-- Define the right focus of ellipse C
def right_focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the right focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - right_focus.1)

-- Define the property of complementary angles of inclination
def complementary_angles (P A B : ℝ × ℝ) : Prop :=
  (A.2 - P.2) / (A.1 - P.1) + (B.2 - P.2) / (B.1 - P.1) = 0

-- Main theorem
theorem ellipse_complementary_angles_point :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    k ≠ 0 →
    line_through_focus k A.1 A.2 →
    line_through_focus k B.1 B.2 →
    ellipse_C A.1 A.2 →
    ellipse_C B.1 B.2 →
    A ≠ B →
    complementary_angles P A B :=
sorry

end NUMINAMATH_CALUDE_ellipse_complementary_angles_point_l606_60625


namespace NUMINAMATH_CALUDE_adjacent_even_sum_l606_60660

theorem adjacent_even_sum (numbers : Vector ℕ 2019) : 
  ∃ i : Fin 2019, Even ((numbers.get i) + (numbers.get ((i + 1) % 2019))) :=
sorry

end NUMINAMATH_CALUDE_adjacent_even_sum_l606_60660


namespace NUMINAMATH_CALUDE_largest_fraction_l606_60613

theorem largest_fraction : 
  (26 : ℚ) / 51 > 101 / 203 ∧ 
  (26 : ℚ) / 51 > 47 / 93 ∧ 
  (26 : ℚ) / 51 > 5 / 11 ∧ 
  (26 : ℚ) / 51 > 199 / 401 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l606_60613


namespace NUMINAMATH_CALUDE_melinda_coffees_l606_60678

/-- The cost of one doughnut in dollars -/
def doughnut_cost : ℚ := 45/100

/-- The total cost of Harold's purchase on Monday in dollars -/
def harold_total : ℚ := 491/100

/-- The number of doughnuts Harold bought on Monday -/
def harold_doughnuts : ℕ := 3

/-- The number of coffees Harold bought on Monday -/
def harold_coffees : ℕ := 4

/-- The total cost of Melinda's purchase on Tuesday in dollars -/
def melinda_total : ℚ := 759/100

/-- The number of doughnuts Melinda bought on Tuesday -/
def melinda_doughnuts : ℕ := 5

/-- Theorem stating that Melinda bought 6 large coffees on Tuesday -/
theorem melinda_coffees : ℕ := by
  sorry


end NUMINAMATH_CALUDE_melinda_coffees_l606_60678


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l606_60639

/-- Given a cistern with normal fill time and leak-affected fill time, 
    calculate the time to empty through the leak. -/
theorem cistern_emptying_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 2) 
  (h2 : leak_fill_time = 4) : 
  (1 / (1 / normal_fill_time - 1 / leak_fill_time)) = 4 := by
  sorry

#check cistern_emptying_time

end NUMINAMATH_CALUDE_cistern_emptying_time_l606_60639


namespace NUMINAMATH_CALUDE_equal_expressions_l606_60695

theorem equal_expressions (x : ℝ) : 2 * x - 1 = 3 * x + 3 ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l606_60695


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l606_60681

theorem infinite_sum_equality (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  let f : ℕ → ℝ := fun n => 1 / ((n * c - (n - 1) * d) * ((n + 1) * c - n * d))
  let series := ∑' n, f n
  series = 1 / ((c - d) * d) := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l606_60681


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l606_60607

theorem right_triangle_inequality (a b c h : ℝ) (n : ℕ) (h1 : 0 < n) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < h) 
  (h6 : a^2 + b^2 = c^2) (h7 : a * b = c * h) (h8 : a + b < c + h) :
  a^n + b^n < c^n + h^n := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l606_60607


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l606_60614

/-- Given that quantities a and b vary inversely, prove that b = 0.375 when a = 1600 -/
theorem inverse_variation_problem (a b : ℝ) (h1 : a * b = 800 * 0.5) 
  (h2 : (2 * 800) * (b / 2) = a * b + 200) : 
  (a = 1600) → (b = 0.375) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l606_60614


namespace NUMINAMATH_CALUDE_q_is_false_l606_60617

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_is_false_l606_60617


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l606_60664

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : 1 ≤ a - 2*b ∧ a - 2*b ≤ 3) : 
  -11/3 ≤ a + 3*b ∧ a + 3*b ≤ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l606_60664


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l606_60615

theorem line_slope_intercept_product (m b : ℚ) : 
  m = -3/4 → b = 3/2 → m * b < -1 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l606_60615


namespace NUMINAMATH_CALUDE_four_digit_sum_with_reverse_l606_60620

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Returns the reversed digits of a four-digit number -/
def reverseDigits (x : FourDigitNumber) : FourDigitNumber :=
  sorry

/-- The sum of a number and its reverse -/
def sumWithReverse (x : FourDigitNumber) : ℕ :=
  x.val + (reverseDigits x).val

theorem four_digit_sum_with_reverse (x : FourDigitNumber) :
  x.val % 10 ≠ 0 →
  (sumWithReverse x) % 100 = 0 →
  sumWithReverse x = 11000 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_with_reverse_l606_60620


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l606_60602

theorem polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) (h2 : 40 * n = 360) :
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l606_60602


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l606_60670

theorem polynomial_evaluation (x y p q : ℝ) 
  (h1 : x + y = -p) 
  (h2 : x * y = q) : 
  x * (1 + y) - y * (x * y - 1) - x^2 * y = p * q + q - p :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l606_60670


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l606_60657

def total_players : ℕ := 15
def predetermined_players : ℕ := 3
def players_to_choose : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (total_players - predetermined_players) players_to_choose = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l606_60657


namespace NUMINAMATH_CALUDE_function_inequality_l606_60653

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 2) * (deriv^[2] f x) > 0) : 
  f 2 < f 0 ∧ f 0 < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l606_60653


namespace NUMINAMATH_CALUDE_melanie_dimes_l606_60698

theorem melanie_dimes (initial_dimes : ℕ) : 
  (initial_dimes - 7 + 4 = 5) → initial_dimes = 8 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l606_60698


namespace NUMINAMATH_CALUDE_library_books_total_l606_60603

/-- The total number of books obtained from the library -/
def total_books (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 54 initial books and 23 additional books, the total is 77 -/
theorem library_books_total : total_books 54 23 = 77 := by
  sorry

end NUMINAMATH_CALUDE_library_books_total_l606_60603


namespace NUMINAMATH_CALUDE_arithmetic_progression_unique_solution_l606_60619

theorem arithmetic_progression_unique_solution (n₁ n₂ : ℕ) (hn : n₁ ≠ n₂) :
  ∃! (a₁ d : ℚ),
    (∀ (n : ℕ), n * (2 * a₁ + (n - 1) * d) / 2 = n^2) ∧
    (n₁ * (2 * a₁ + (n₁ - 1) * d) / 2 = n₁^2) ∧
    (n₂ * (2 * a₁ + (n₂ - 1) * d) / 2 = n₂^2) ∧
    a₁ = 1 ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_unique_solution_l606_60619


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l606_60630

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 2 > 3 * (1 - x) ∧ 1 - 2 * x ≤ 2) ↔ x > (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l606_60630


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l606_60682

theorem sum_reciprocal_inequality (a b c : ℝ) (h : a + b + c = 3) :
  1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l606_60682


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l606_60629

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l606_60629


namespace NUMINAMATH_CALUDE_month_days_l606_60612

theorem month_days (days_took_capsules days_forgot_capsules : ℕ) 
  (h1 : days_took_capsules = 29)
  (h2 : days_forgot_capsules = 2) : 
  days_took_capsules + days_forgot_capsules = 31 := by
sorry

end NUMINAMATH_CALUDE_month_days_l606_60612


namespace NUMINAMATH_CALUDE_no_nonneg_int_solutions_l606_60692

theorem no_nonneg_int_solutions : 
  ¬∃ (x : ℕ), 4 * (x - 2) > 2 * (3 * x + 5) := by
sorry

end NUMINAMATH_CALUDE_no_nonneg_int_solutions_l606_60692


namespace NUMINAMATH_CALUDE_bus_capacity_l606_60685

/-- A bus with seats on both sides and a back seat -/
structure Bus where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  back_seat_capacity : Nat

/-- Calculate the total number of people that can sit in the bus -/
def total_capacity (b : Bus) : Nat :=
  (b.left_seats + b.right_seats) * b.people_per_seat + b.back_seat_capacity

/-- Theorem stating the total capacity of the bus -/
theorem bus_capacity :
  ∃ (b : Bus),
    b.left_seats = 15 ∧
    b.right_seats = b.left_seats - 3 ∧
    b.people_per_seat = 3 ∧
    b.back_seat_capacity = 7 ∧
    total_capacity b = 88 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l606_60685


namespace NUMINAMATH_CALUDE_window_installation_time_l606_60640

theorem window_installation_time (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) 
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_per_window = 4) : 
  (total_windows - installed_windows) * time_per_window = 36 := by
  sorry

end NUMINAMATH_CALUDE_window_installation_time_l606_60640


namespace NUMINAMATH_CALUDE_no_solution_exists_l606_60641

theorem no_solution_exists : ¬ ∃ (n m r : ℕ), 
  n ≥ 1 ∧ m ≥ 1 ∧ r ≥ 1 ∧ n^5 + 49^m = 1221^r := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l606_60641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l606_60687

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions,
    prove that the sum of its third and fourth terms is 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h1 : isArithmeticSequence a) 
    (h2 : a 1 + a 2 = 10) 
    (h3 : a 4 = a 3 + 2) : 
  a 3 + a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l606_60687


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l606_60691

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 29989 ∧ m = 73) :
  ∃ x : ℕ, x = 21 ∧ 
    (∀ y : ℕ, (n + y) % m = 0 → y ≥ x) ∧
    (n + x) % m = 0 :=
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l606_60691


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l606_60697

theorem smallest_x_with_remainders : ∃ (x : ℕ), 
  (x % 5 = 4) ∧ 
  (x % 6 = 5) ∧ 
  (x % 7 = 6) ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(y % 5 = 4 ∧ y % 6 = 5 ∧ y % 7 = 6)) ∧
  x = 209 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l606_60697


namespace NUMINAMATH_CALUDE_student_D_most_stable_smallest_variance_most_stable_l606_60634

-- Define the variances for each student
def variance_A : ℝ := 6
def variance_B : ℝ := 5.5
def variance_C : ℝ := 10
def variance_D : ℝ := 3.8

-- Define a function to determine if a student has the most stable performance
def has_most_stable_performance (student_variance : ℝ) : Prop :=
  student_variance ≤ variance_A ∧
  student_variance ≤ variance_B ∧
  student_variance ≤ variance_C ∧
  student_variance ≤ variance_D

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable : has_most_stable_performance variance_D := by
  sorry

-- Theorem stating that the student with the smallest variance has the most stable performance
theorem smallest_variance_most_stable :
  ∀ (student_variance : ℝ),
    has_most_stable_performance student_variance →
    student_variance = min (min (min variance_A variance_B) variance_C) variance_D := by
  sorry

end NUMINAMATH_CALUDE_student_D_most_stable_smallest_variance_most_stable_l606_60634


namespace NUMINAMATH_CALUDE_volume_circumscribed_sphere_folded_rectangle_l606_60623

/-- The volume of the circumscribed sphere of a tetrahedron formed by folding a rectangle --/
theorem volume_circumscribed_sphere_folded_rectangle (a b : ℝ) (ha : a = 4) (hb : b = 3) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = (125/6) * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_circumscribed_sphere_folded_rectangle_l606_60623


namespace NUMINAMATH_CALUDE_least_m_satisfying_condition_l606_60633

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The problem statement -/
theorem least_m_satisfying_condition : ∃ m : ℕ, 
  (m > 0) ∧ 
  (∀ k : ℕ, k > 0 → k < m → 
    ¬(∃ p : ℕ, trailingZeros k = p ∧ 
      trailingZeros (2 * k) = ⌊(5 * p : ℚ) / 2⌋)) ∧
  (∃ p : ℕ, trailingZeros m = p ∧ 
    trailingZeros (2 * m) = ⌊(5 * p : ℚ) / 2⌋) ∧
  m = 25 := by
  sorry

end NUMINAMATH_CALUDE_least_m_satisfying_condition_l606_60633


namespace NUMINAMATH_CALUDE_pineapple_problem_l606_60699

theorem pineapple_problem (pineapple_cost : ℕ) (rings_per_pineapple : ℕ) 
  (rings_per_sale : ℕ) (sale_price : ℕ) (total_profit : ℕ) :
  pineapple_cost = 3 →
  rings_per_pineapple = 12 →
  rings_per_sale = 4 →
  sale_price = 5 →
  total_profit = 72 →
  ∃ (num_pineapples : ℕ),
    num_pineapples * (rings_per_pineapple / rings_per_sale * sale_price - pineapple_cost) = total_profit ∧
    num_pineapples = 6 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_problem_l606_60699


namespace NUMINAMATH_CALUDE_consecutive_even_product_l606_60651

theorem consecutive_even_product : 442 * 444 * 446 = 87526608 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_product_l606_60651


namespace NUMINAMATH_CALUDE_first_dog_consumption_l606_60638

/-- Represents the weekly food consumption of three dogs -/
structure DogFoodConsumption where
  first_dog : ℝ
  second_dog : ℝ
  third_dog : ℝ

/-- The total weekly food consumption of the three dogs -/
def total_consumption (d : DogFoodConsumption) : ℝ :=
  d.first_dog + d.second_dog + d.third_dog

theorem first_dog_consumption :
  ∃ (d : DogFoodConsumption),
    total_consumption d = 15 ∧
    d.second_dog = 2 * d.first_dog ∧
    d.third_dog = 6 ∧
    d.first_dog = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_dog_consumption_l606_60638


namespace NUMINAMATH_CALUDE_sector_perimeter_l606_60654

/-- Given a sector with central angle 54° and radius 20 cm, its perimeter is (6π + 40) cm -/
theorem sector_perimeter (θ : Real) (r : Real) : 
  θ = 54 * Real.pi / 180 → r = 20 → 
  (θ * r) + 2 * r = 6 * Real.pi + 40 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l606_60654


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l606_60635

/-- Given a two-digit number n = 10a + b, where a and b are single digits,
    if the difference between n and its reverse is 7 times the sum of its digits,
    then the sum of n and its reverse is 99. -/
theorem two_digit_number_sum (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) (ha_pos : a > 0) :
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l606_60635


namespace NUMINAMATH_CALUDE_rohan_salary_l606_60636

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 5000

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 1000

theorem rohan_salary :
  monthly_salary * (1 - (food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage) / 100) = savings :=
by sorry

end NUMINAMATH_CALUDE_rohan_salary_l606_60636


namespace NUMINAMATH_CALUDE_cube_shadow_problem_l606_60662

/-- Given a cube with edge length 2 cm and a light source x cm above one upper vertex,
    if the shadow area (excluding the area beneath the cube) is 192 cm²,
    then the greatest integer not exceeding 1000x is 25780. -/
theorem cube_shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 192
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := (total_shadow_area).sqrt
  x = (shadow_side - cube_edge) / 2 →
  ⌊1000 * x⌋ = 25780 := by sorry

end NUMINAMATH_CALUDE_cube_shadow_problem_l606_60662


namespace NUMINAMATH_CALUDE_total_comics_in_box_l606_60622

-- Define the problem parameters
def pages_per_comic : ℕ := 25
def found_pages : ℕ := 150
def untorn_comics : ℕ := 5

-- State the theorem
theorem total_comics_in_box : 
  (found_pages / pages_per_comic) + untorn_comics = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_comics_in_box_l606_60622


namespace NUMINAMATH_CALUDE_square_sum_simplification_l606_60637

theorem square_sum_simplification (a : ℝ) : a^2 + 2*a^2 = 3*a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_simplification_l606_60637


namespace NUMINAMATH_CALUDE_inequality_proof_l606_60628

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2 / 3 ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l606_60628


namespace NUMINAMATH_CALUDE_smallest_number_is_three_l606_60601

/-- Represents the systematic sampling of classes -/
structure ClassSampling where
  total_classes : Nat
  selected_classes : Nat
  sum_of_selected : Nat

/-- Calculates the smallest number in the systematic sample -/
def smallest_number (sampling : ClassSampling) : Nat :=
  let interval := sampling.total_classes / sampling.selected_classes
  (sampling.sum_of_selected - (interval * (sampling.selected_classes - 1) * sampling.selected_classes / 2)) / sampling.selected_classes

/-- Theorem: The smallest number in the given systematic sample is 3 -/
theorem smallest_number_is_three (sampling : ClassSampling) 
  (h1 : sampling.total_classes = 30)
  (h2 : sampling.selected_classes = 5)
  (h3 : sampling.sum_of_selected = 75) :
  smallest_number sampling = 3 := by
  sorry

#eval smallest_number { total_classes := 30, selected_classes := 5, sum_of_selected := 75 }

end NUMINAMATH_CALUDE_smallest_number_is_three_l606_60601


namespace NUMINAMATH_CALUDE_joseph_baseball_cards_l606_60677

theorem joseph_baseball_cards (X : ℚ) : 
  X - (3/8) * X - 2 = (1/2) * X → X = 16 := by
  sorry

end NUMINAMATH_CALUDE_joseph_baseball_cards_l606_60677


namespace NUMINAMATH_CALUDE_reflection_line_is_x_equals_zero_l606_60672

-- Define the points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (5, 7)
def R : ℝ × ℝ := (-2, 5)
def P' : ℝ × ℝ := (-1, 2)
def Q' : ℝ × ℝ := (-5, 7)
def R' : ℝ × ℝ := (2, 5)

-- Define the reflection line
def M : Set (ℝ × ℝ) := {(x, y) | x = 0}

-- Theorem statement
theorem reflection_line_is_x_equals_zero :
  (∀ (x y : ℝ), (x, y) ∈ M ↔ x = 0) ∧
  (P.1 + P'.1 = 0) ∧ (P.2 = P'.2) ∧
  (Q.1 + Q'.1 = 0) ∧ (Q.2 = Q'.2) ∧
  (R.1 + R'.1 = 0) ∧ (R.2 = R'.2) :=
sorry


end NUMINAMATH_CALUDE_reflection_line_is_x_equals_zero_l606_60672


namespace NUMINAMATH_CALUDE_card_distribution_l606_60665

theorem card_distribution (total : ℕ) (black red : ℕ) (spades diamonds hearts clubs : ℕ) : 
  total = 13 →
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  total = spades + diamonds + hearts + clubs →
  black = spades + clubs →
  red = diamonds + hearts →
  clubs = 6 := by
sorry

end NUMINAMATH_CALUDE_card_distribution_l606_60665


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_127_l606_60690

theorem first_nonzero_digit_of_1_over_127 :
  ∃ (n : ℕ), n > 0 ∧ (1000 : ℚ) / 127 = 7 + n / 127 ∧ n < 127 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_127_l606_60690


namespace NUMINAMATH_CALUDE_triangle_problem_l606_60608

/-- Given a triangle ABC with vertex A at (5,1), altitude CH from AB with equation x-2y-5=0,
    and median BM from AC with equation 2x-y-1=0, prove the coordinates of B and the equation
    of the perpendicular bisector of BC. -/
theorem triangle_problem (B : ℝ × ℝ) (perpBisectorBC : ℝ → ℝ → ℝ) : 
  let A : ℝ × ℝ := (5, 1)
  let altitude_CH (x y : ℝ) := x - 2*y - 5 = 0
  let median_BM (x y : ℝ) := 2*x - y - 1 = 0
  B = (3, 5) ∧ 
  (∀ x y, perpBisectorBC x y = 0 ↔ 21*x + 24*y + 43 = 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l606_60608


namespace NUMINAMATH_CALUDE_scientific_notation_of_32100000_l606_60626

theorem scientific_notation_of_32100000 : 
  32100000 = 3.21 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_32100000_l606_60626


namespace NUMINAMATH_CALUDE_invalid_triangle_after_transformation_l606_60696

theorem invalid_triangle_after_transformation (DE DF EF : ℝ) 
  (h_original_valid : DE + DF > EF ∧ DE + EF > DF ∧ DF + EF > DE)
  (h_DE : DE = 8)
  (h_DF : DF = 9)
  (h_EF : EF = 5)
  (DE' DF' EF' : ℝ)
  (h_DE' : DE' = 3 * DE)
  (h_DF' : DF' = 2 * DF)
  (h_EF' : EF' = EF) :
  ¬(DE' + DF' > EF' ∧ DE' + EF' > DF' ∧ DF' + EF' > DE') :=
by sorry

end NUMINAMATH_CALUDE_invalid_triangle_after_transformation_l606_60696


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l606_60683

theorem last_two_digits_sum (n : ℕ) : n = 7^15 + 13^15 → n % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l606_60683


namespace NUMINAMATH_CALUDE_range_of_m_chord_length_l606_60675

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
sorry

-- Theorem for the length of chord MN when m = 4
theorem chord_length :
  let m : ℝ := 4
  ∃ M N : ℝ × ℝ,
    circle_equation M.1 M.2 m ∧
    circle_equation N.1 N.2 m ∧
    line_equation M.1 M.2 ∧
    line_equation N.1 N.2 ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_chord_length_l606_60675


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l606_60652

theorem gcd_sum_and_sum_of_squares (a b : ℕ+) (h : Nat.Coprime a b) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l606_60652


namespace NUMINAMATH_CALUDE_valentines_packs_given_away_l606_60694

def initial_valentines : ℕ := 450
def remaining_valentines : ℕ := 70
def valentines_per_pack : ℕ := 10

theorem valentines_packs_given_away : 
  (initial_valentines - remaining_valentines) / valentines_per_pack = 38 := by
  sorry

end NUMINAMATH_CALUDE_valentines_packs_given_away_l606_60694


namespace NUMINAMATH_CALUDE_stadium_empty_seats_l606_60648

/-- The number of empty seats in a stadium -/
def empty_seats (total_seats people_present : ℕ) : ℕ :=
  total_seats - people_present

/-- Theorem: Given a stadium with 92 seats and 47 people present, there are 45 empty seats -/
theorem stadium_empty_seats : empty_seats 92 47 = 45 := by
  sorry

end NUMINAMATH_CALUDE_stadium_empty_seats_l606_60648


namespace NUMINAMATH_CALUDE_problems_per_page_l606_60645

theorem problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 110) 
  (h2 : finished_problems = 47) 
  (h3 : remaining_pages = 7) 
  (h4 : finished_problems < total_problems) : 
  (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_page_l606_60645


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l606_60686

def A : Set ℝ := {x : ℝ | |x| ≤ 2}
def B : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l606_60686


namespace NUMINAMATH_CALUDE_line_equation_l606_60605

/-- A line passing through (1,1) with y-intercept 3 has equation 2x + y - 3 = 0 -/
theorem line_equation (x y : ℝ) : 
  (2 * 1 + 1 - 3 = 0) ∧ 
  (2 * 0 + 3 - 3 = 0) ∧ 
  (∀ x y, y = -2 * x + 3) → 
  2 * x + y - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_line_equation_l606_60605


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l606_60674

/-- The area of a triangle inscribed in a circle, given the circle's radius and the ratio of the triangle's sides. -/
theorem inscribed_triangle_area
  (r : ℝ) -- radius of the circle
  (a b c : ℝ) -- ratios of the triangle's sides
  (h_positive : r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0) -- positivity conditions
  (h_ratio : a^2 + b^2 = c^2) -- Pythagorean theorem condition for the ratios
  (h_diameter : c * (a + b + c)⁻¹ * 2 * r = c) -- condition relating the longest side to the diameter
  : (1/2 * a * b * (a + b + c)⁻¹ * 2 * r)^2 = 216/25 ∧ r = 3 ∧ (a, b, c) = (3, 4, 5) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l606_60674


namespace NUMINAMATH_CALUDE_right_triangle_angle_A_l606_60668

theorem right_triangle_angle_A (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.cos B = Real.sqrt 3 / 2) : A = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_A_l606_60668


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l606_60659

theorem simplify_trig_expression :
  let x : Real := 10 * π / 180  -- 10 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.cos (17 * x) ^ 2)) = Real.tan x :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l606_60659


namespace NUMINAMATH_CALUDE_percentage_given_away_l606_60611

def total_amount : ℝ := 100
def amount_kept : ℝ := 80

theorem percentage_given_away : 
  (total_amount - amount_kept) / total_amount * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_given_away_l606_60611


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l606_60666

/-- The surface area of a rectangular solid -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth :
  ∃ (h : ℝ), h > 0 ∧ surface_area 5 4 h = 58 → h = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l606_60666


namespace NUMINAMATH_CALUDE_truck_travel_distance_l606_60624

/-- Represents the distance a truck can travel -/
def truck_distance (miles_per_gallon : ℝ) (initial_gallons : ℝ) (added_gallons : ℝ) : ℝ :=
  miles_per_gallon * (initial_gallons + added_gallons)

/-- Theorem: A truck traveling 3 miles per gallon with 12 gallons initially and 18 gallons added can travel 90 miles -/
theorem truck_travel_distance :
  truck_distance 3 12 18 = 90 := by
  sorry

#eval truck_distance 3 12 18

end NUMINAMATH_CALUDE_truck_travel_distance_l606_60624


namespace NUMINAMATH_CALUDE_map_scale_proportion_l606_60679

/-- Represents the scale of a map -/
structure MapScale where
  cm : ℝ  -- centimeters on the map
  km : ℝ  -- kilometers in reality

/-- 
Given a map scale where 15 cm represents 90 km, 
proves that 20 cm represents 120 km on the same map
-/
theorem map_scale_proportion (scale : MapScale) 
  (h : scale.cm = 15 ∧ scale.km = 90) : 
  ∃ (new_scale : MapScale), 
    new_scale.cm = 20 ∧ 
    new_scale.km = 120 ∧
    new_scale.km / new_scale.cm = scale.km / scale.cm := by
  sorry


end NUMINAMATH_CALUDE_map_scale_proportion_l606_60679


namespace NUMINAMATH_CALUDE_vectors_in_plane_implies_x_eq_neg_one_l606_60680

-- Define the vectors
def a (x : ℝ) : Fin 3 → ℝ := ![1, x, -2]
def b : Fin 3 → ℝ := ![0, 1, 2]
def c : Fin 3 → ℝ := ![1, 0, 0]

-- Define the condition that vectors lie in the same plane
def vectors_in_same_plane (x : ℝ) : Prop :=
  ∃ (m n : ℝ), a x = m • b + n • c

-- Theorem statement
theorem vectors_in_plane_implies_x_eq_neg_one :
  ∀ x : ℝ, vectors_in_same_plane x → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_vectors_in_plane_implies_x_eq_neg_one_l606_60680


namespace NUMINAMATH_CALUDE_solve_equation_l606_60689

theorem solve_equation : ∃ x : ℝ, x + 1 - 2 + 3 - 4 = 5 - 6 + 7 - 8 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l606_60689


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l606_60658

/-- The curve y = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- Point P₀ -/
def P₀ : ℝ × ℝ := (-1, -4)

/-- The slope of the line parallel to the tangent at P₀ -/
def m : ℝ := 4

/-- The equation of the line perpendicular to the tangent at P₀ -/
def l (x y : ℝ) : Prop := x + 4*y + 17 = 0

theorem tangent_point_and_perpendicular_line :
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ f' x = m) →  -- P₀ exists in third quadrant with slope m
  (P₀.1 = -1 ∧ P₀.2 = -4) ∧  -- P₀ has coordinates (-1, -4)
  (∀ (x y : ℝ), l x y ↔ y - P₀.2 = -(1/m) * (x - P₀.1)) :=  -- l is perpendicular to tangent at P₀
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l606_60658


namespace NUMINAMATH_CALUDE_equation_equivalence_l606_60673

theorem equation_equivalence (a c x y : ℤ) (m n p : ℕ) : 
  (a^9*x*y - a^8*y - a^7*x = a^6*(c^3 - 1)) →
  ((a^m*x - a^n)*(a^p*y - a^3) = a^6*c^3) →
  m*n*p = 90 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l606_60673


namespace NUMINAMATH_CALUDE_photographer_arrangement_exists_l606_60609

-- Define a type for photographers
def Photographer := Fin 6

-- Define a type for positions in the plane
def Position := ℝ × ℝ

-- Define a function to check if a photographer is between two others
def isBetween (p₁ p₂ p₃ : Position) : Prop := sorry

-- Define a function to check if two photographers can see each other
def canSee (positions : Photographer → Position) (p₁ p₂ : Photographer) : Prop :=
  ∀ p₃, p₃ ≠ p₁ ∧ p₃ ≠ p₂ → ¬ isBetween (positions p₁) (positions p₃) (positions p₂)

-- State the theorem
theorem photographer_arrangement_exists :
  ∃ (positions : Photographer → Position),
    ∀ p, (∃! (s : Finset Photographer), s.card = 4 ∧ ∀ p' ∈ s, canSee positions p p') :=
sorry

end NUMINAMATH_CALUDE_photographer_arrangement_exists_l606_60609


namespace NUMINAMATH_CALUDE_square_13_on_top_l606_60632

/-- Represents a 5x5 grid of numbers -/
def Grid := Fin 5 → Fin 5 → Fin 25

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  fun i j => ⟨i.val * 5 + j.val + 1, by sorry⟩

/-- Represents a folding operation on the grid -/
def Fold := Grid → Grid

/-- Fold the top half over the bottom half -/
def fold1 : Fold := sorry

/-- Fold the bottom half over the top half -/
def fold2 : Fold := sorry

/-- Fold the left half over the right half -/
def fold3 : Fold := sorry

/-- Fold the right half over the left half -/
def fold4 : Fold := sorry

/-- Fold diagonally from bottom left to top right -/
def fold5 : Fold := sorry

/-- The final configuration after all folds -/
def final_grid : Grid :=
  fold5 (fold4 (fold3 (fold2 (fold1 initial_grid))))

/-- The theorem stating that square 13 is on top after all folds -/
theorem square_13_on_top :
  final_grid 0 0 = ⟨13, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_square_13_on_top_l606_60632


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l606_60647

/-- Given an arithmetic sequence with first three terms a-1, a+1, 2a+3, 
    prove that its general formula is a_n = 2n - 3 -/
theorem arithmetic_sequence_formula 
  (a : ℝ) 
  (seq : ℕ → ℝ) 
  (h1 : seq 1 = a - 1) 
  (h2 : seq 2 = a + 1) 
  (h3 : seq 3 = 2*a + 3) 
  (h_arithmetic : ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)) :
  ∀ n : ℕ, seq n = 2*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l606_60647


namespace NUMINAMATH_CALUDE_cubic_polynomial_fits_points_l606_60649

def f (x : ℝ) : ℝ := -10 * x^3 + 20 * x^2 - 60 * x + 200

theorem cubic_polynomial_fits_points :
  f 0 = 200 ∧
  f 1 = 150 ∧
  f 2 = 80 ∧
  f 3 = 0 ∧
  f 4 = -140 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_fits_points_l606_60649


namespace NUMINAMATH_CALUDE_mode_of_sample_data_l606_60643

def sample_data : List Int := [-2, 0, 6, 3, 6]

def mode (data : List Int) : Int :=
  data.foldl (fun acc x => if data.count x > data.count acc then x else acc) 0

theorem mode_of_sample_data :
  mode sample_data = 6 := by sorry

end NUMINAMATH_CALUDE_mode_of_sample_data_l606_60643


namespace NUMINAMATH_CALUDE_password_count_l606_60671

/-- The number of case-insensitive English letters -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letters in the password -/
def num_password_letters : ℕ := 2

/-- The number of digits in the password -/
def num_password_digits : ℕ := 2

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of possible passwords -/
def num_passwords : ℕ := 
  permutations num_letters num_password_letters * permutations num_digits num_password_digits

theorem password_count : 
  num_passwords = permutations num_letters num_password_letters * permutations num_digits num_password_digits :=
by sorry

end NUMINAMATH_CALUDE_password_count_l606_60671


namespace NUMINAMATH_CALUDE_randy_farm_trees_l606_60684

/-- Calculates the total number of trees on Randy's farm -/
def total_trees (mango_trees : ℕ) (coconut_trees : ℕ) : ℕ :=
  mango_trees + coconut_trees

/-- Theorem: Given Randy's farm conditions, the total number of trees is 85 -/
theorem randy_farm_trees :
  let mango_trees : ℕ := 60
  let coconut_trees : ℕ := mango_trees / 2 - 5
  total_trees mango_trees coconut_trees = 85 := by
  sorry

end NUMINAMATH_CALUDE_randy_farm_trees_l606_60684


namespace NUMINAMATH_CALUDE_kayla_waiting_time_l606_60646

/-- The number of years Kayla needs to wait before reaching the minimum driving age -/
def years_until_driving (minimum_age : ℕ) (kimiko_age : ℕ) : ℕ :=
  minimum_age - kimiko_age / 2

/-- Proof that Kayla needs to wait 5 years before she can start driving -/
theorem kayla_waiting_time :
  years_until_driving 18 26 = 5 := by
  sorry

end NUMINAMATH_CALUDE_kayla_waiting_time_l606_60646


namespace NUMINAMATH_CALUDE_gunther_typing_words_l606_60661

-- Define the typing speeds and durations
def first_phase_speed : ℕ := 160
def first_phase_duration : ℕ := 2 * 60
def second_phase_speed : ℕ := 200
def second_phase_duration : ℕ := 3 * 60
def third_phase_speed : ℕ := 140
def third_phase_duration : ℕ := 4 * 60

-- Define the interval duration (in minutes)
def interval_duration : ℕ := 3

-- Function to calculate words typed in a phase
def words_in_phase (speed : ℕ) (duration : ℕ) : ℕ :=
  (duration / interval_duration) * speed

-- Theorem statement
theorem gunther_typing_words :
  words_in_phase first_phase_speed first_phase_duration +
  words_in_phase second_phase_speed second_phase_duration +
  words_in_phase third_phase_speed third_phase_duration = 29600 := by
  sorry

end NUMINAMATH_CALUDE_gunther_typing_words_l606_60661


namespace NUMINAMATH_CALUDE_composition_may_have_no_fixed_point_l606_60610

-- Define a type for our functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to have a fixed point
def has_fixed_point (f : RealFunction) : Prop :=
  ∃ x : ℝ, f x = x

-- State the theorem
theorem composition_may_have_no_fixed_point :
  ∃ (f g : RealFunction),
    has_fixed_point f ∧ 
    has_fixed_point g ∧ 
    ¬(has_fixed_point (f ∘ g)) :=
sorry

end NUMINAMATH_CALUDE_composition_may_have_no_fixed_point_l606_60610


namespace NUMINAMATH_CALUDE_min_side_length_of_A_l606_60644

-- Define the squares
structure Square where
  sideLength : ℕ

-- Define the configuration
structure SquareConfiguration where
  A : Square
  B : Square
  C : Square
  D : Square
  vertexCondition : A.sideLength = B.sideLength + C.sideLength + D.sideLength
  areaCondition : A.sideLength^2 / 2 = B.sideLength^2 + C.sideLength^2 + D.sideLength^2

-- Theorem statement
theorem min_side_length_of_A (config : SquareConfiguration) :
  config.A.sideLength ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_side_length_of_A_l606_60644


namespace NUMINAMATH_CALUDE_power_sine_inequality_l606_60656

theorem power_sine_inequality (α : Real) (x₁ x₂ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < x₁)
  (h3 : x₁ < x₂) :
  (x₂ / x₁) ^ (Real.sin α) > 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sine_inequality_l606_60656


namespace NUMINAMATH_CALUDE_triangle_x_coordinate_l606_60688

/-- 
Given a triangle with vertices (x, 0), (7, 4), and (7, -4),
if the area of the triangle is 32, then x = -1.
-/
theorem triangle_x_coordinate (x : ℝ) : 
  let v1 : ℝ × ℝ := (x, 0)
  let v2 : ℝ × ℝ := (7, 4)
  let v3 : ℝ × ℝ := (7, -4)
  let base : ℝ := |v2.2 - v3.2|
  let height : ℝ := |7 - x|
  let area : ℝ := (1/2) * base * height
  area = 32 → x = -1 := by
sorry

end NUMINAMATH_CALUDE_triangle_x_coordinate_l606_60688


namespace NUMINAMATH_CALUDE_power_three_thirds_of_675_l606_60600

theorem power_three_thirds_of_675 : (675 : ℝ) ^ (3/3) = 675 := by
  sorry

end NUMINAMATH_CALUDE_power_three_thirds_of_675_l606_60600


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l606_60693

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 675 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l606_60693


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l606_60676

theorem quadratic_solution_difference_squared :
  ∀ (α β : ℝ),
    α ≠ β →
    α^2 - 3*α + 2 = 0 →
    β^2 - 3*β + 2 = 0 →
    (α - β)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l606_60676


namespace NUMINAMATH_CALUDE_min_value_theorem_l606_60642

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, a * x + b * y - 2 = 0 → x^2 + y^2 - 6*x - 4*y - 12 = 0) →
  (∃ x y : ℝ, a * x + b * y - 2 = 0 ∧ x^2 + y^2 - 6*x - 4*y - 12 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, a' * x + b' * y - 2 = 0 → x^2 + y^2 - 6*x - 4*y - 12 = 0) →
    (∃ x y : ℝ, a' * x + b' * y - 2 = 0 ∧ x^2 + y^2 - 6*x - 4*y - 12 = 0) →
    3/a + 2/b ≤ 3/a' + 2/b') →
  3/a + 2/b = 25/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l606_60642
