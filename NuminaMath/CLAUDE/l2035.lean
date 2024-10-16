import Mathlib

namespace NUMINAMATH_CALUDE_infinitely_many_primes_not_in_S_a_l2035_203536

-- Define the set S_a
def S_a (a : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ b : ℕ, Odd b ∧ p ∣ (2^(2^a))^b - 1}

-- State the theorem
theorem infinitely_many_primes_not_in_S_a :
  ∀ a : ℕ, a > 0 → Set.Infinite {p : ℕ | Nat.Prime p ∧ p ∉ S_a a} :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_not_in_S_a_l2035_203536


namespace NUMINAMATH_CALUDE_conversation_on_weekday_l2035_203592

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to check if a day is a weekday
def isWeekday (d : Day) : Prop :=
  d ≠ Day.Saturday ∧ d ≠ Day.Sunday

-- Define the brothers
structure Brother :=
  (liesOnSaturday : Bool)
  (liesOnSunday : Bool)
  (willLieTomorrow : Bool)

-- Define the conversation
def conversation (day : Day) (brother1 brother2 : Brother) : Prop :=
  brother1.liesOnSaturday = true
  ∧ brother1.liesOnSunday = true
  ∧ brother2.willLieTomorrow = true
  ∧ (day = Day.Saturday → ¬brother1.liesOnSaturday)
  ∧ (day = Day.Sunday → ¬brother1.liesOnSunday)
  ∧ (isWeekday day → ¬brother2.willLieTomorrow)

-- Theorem: The conversation occurs on a weekday
theorem conversation_on_weekday (day : Day) (brother1 brother2 : Brother) :
  conversation day brother1 brother2 → isWeekday day :=
by sorry

end NUMINAMATH_CALUDE_conversation_on_weekday_l2035_203592


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2035_203557

theorem gcd_of_specific_numbers : 
  let m : ℕ := 3333333
  let n : ℕ := 99999999
  Nat.gcd m n = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2035_203557


namespace NUMINAMATH_CALUDE_brendans_tax_payment_is_correct_l2035_203520

/-- Calculates Brendan's weekly tax payment based on his work schedule and income reporting --/
def brendans_weekly_tax_payment (
  waiter_hourly_wage : ℚ)
  (barista_hourly_wage : ℚ)
  (waiter_shift_hours : List ℚ)
  (barista_shift_hours : List ℚ)
  (waiter_hourly_tips : ℚ)
  (barista_hourly_tips : ℚ)
  (waiter_tax_rate : ℚ)
  (barista_tax_rate : ℚ)
  (waiter_reported_tips_ratio : ℚ)
  (barista_reported_tips_ratio : ℚ) : ℚ :=
  let waiter_total_hours := waiter_shift_hours.sum
  let barista_total_hours := barista_shift_hours.sum
  let waiter_wage_income := waiter_total_hours * waiter_hourly_wage
  let barista_wage_income := barista_total_hours * barista_hourly_wage
  let waiter_total_tips := waiter_total_hours * waiter_hourly_tips
  let barista_total_tips := barista_total_hours * barista_hourly_tips
  let waiter_reported_tips := waiter_total_tips * waiter_reported_tips_ratio
  let barista_reported_tips := barista_total_tips * barista_reported_tips_ratio
  let waiter_reported_income := waiter_wage_income + waiter_reported_tips
  let barista_reported_income := barista_wage_income + barista_reported_tips
  let waiter_tax := waiter_reported_income * waiter_tax_rate
  let barista_tax := barista_reported_income * barista_tax_rate
  waiter_tax + barista_tax

theorem brendans_tax_payment_is_correct :
  brendans_weekly_tax_payment 6 8 [8, 8, 12] [6] 12 5 (1/5) (1/4) (1/3) (1/2) = 71.75 := by
  sorry

end NUMINAMATH_CALUDE_brendans_tax_payment_is_correct_l2035_203520


namespace NUMINAMATH_CALUDE_math_team_selection_l2035_203591

theorem math_team_selection (girls boys : ℕ) (h1 : girls = 4) (h2 : boys = 6) :
  (girls.choose 2) * (boys.choose 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l2035_203591


namespace NUMINAMATH_CALUDE_sequence_differences_l2035_203542

def a (n : ℕ) : ℕ := n^2 + 1

def first_difference (n : ℕ) : ℕ := a (n + 1) - a n

def second_difference (n : ℕ) : ℕ := first_difference (n + 1) - first_difference n

def third_difference (n : ℕ) : ℕ := second_difference (n + 1) - second_difference n

theorem sequence_differences :
  (∀ n : ℕ, first_difference n = 2*n + 1) ∧
  (∀ n : ℕ, second_difference n = 2) ∧
  (∀ n : ℕ, third_difference n = 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_differences_l2035_203542


namespace NUMINAMATH_CALUDE_max_min_difference_c_l2035_203543

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, x + y + z = 6 ∧ x^2 + y^2 + z^2 = 18) → c_min ≤ x ∧ x ≤ c_max) ∧
    c_max - c_min = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l2035_203543


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2035_203509

/-- Given two vectors a and b in ℝ², if they are parallel and a = (4,2) and b = (x,3), then x = 6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![x, 3]
  (∃ (k : ℝ), b = k • a) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2035_203509


namespace NUMINAMATH_CALUDE_expand_and_subtract_l2035_203532

theorem expand_and_subtract (x : ℝ) : (x + 3) * (2 * x - 5) - (2 * x + 1) = 2 * x^2 - x - 16 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_subtract_l2035_203532


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l2035_203564

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  marks_per_correct : ℕ
  total_score : ℕ
  correct_answers : ℕ

/-- Calculates the marks lost per wrong answer -/
def marks_lost_per_wrong (exam : Examination) : ℚ :=
  (exam.marks_per_correct * exam.correct_answers - exam.total_score) / (exam.total_questions - exam.correct_answers)

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
  (h1 : exam.total_questions = 60)
  (h2 : exam.marks_per_correct = 4)
  (h3 : exam.total_score = 140)
  (h4 : exam.correct_answers = 40) : 
  marks_lost_per_wrong exam = 1 := by
  sorry

#eval marks_lost_per_wrong ⟨60, 4, 140, 40⟩

end NUMINAMATH_CALUDE_marks_lost_is_one_l2035_203564


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2035_203573

theorem quadratic_inequality_problem (m n : ℝ) (h1 : ∀ x : ℝ, x^2 - 3*x + m < 0 ↔ 1 < x ∧ x < n) :
  m = 2 ∧ n = 2 ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → m*a + 2*n*b = 3 → a*b ≤ 9/32) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ m*a + 2*n*b = 3 ∧ a*b = 9/32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2035_203573


namespace NUMINAMATH_CALUDE_percentage_of_red_cars_l2035_203584

theorem percentage_of_red_cars (total_cars : ℕ) (honda_cars : ℕ) 
  (honda_red_percentage : ℚ) (non_honda_red_percentage : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  honda_red_percentage = 90 / 100 →
  non_honda_red_percentage = 225 / 1000 →
  (((honda_red_percentage * honda_cars) + 
    (non_honda_red_percentage * (total_cars - honda_cars))) / total_cars) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_cars_l2035_203584


namespace NUMINAMATH_CALUDE_max_tax_revenue_l2035_203593

-- Define the market conditions
def supply_function (P : ℝ) : ℝ := 6 * P - 312
def demand_slope : ℝ := 4
def tax_rate : ℝ := 30
def consumer_price : ℝ := 118

-- Define the demand function
def demand_function (P : ℝ) : ℝ := 688 - demand_slope * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := (288 - 2.4 * t) * t

-- Theorem statement
theorem max_tax_revenue :
  ∃ (t : ℝ), ∀ (t' : ℝ), tax_revenue t ≥ tax_revenue t' ∧ tax_revenue t = 8640 := by
  sorry


end NUMINAMATH_CALUDE_max_tax_revenue_l2035_203593


namespace NUMINAMATH_CALUDE_prob_select_two_after_transfer_l2035_203514

/-- Represents the label on a ball -/
inductive Label
  | one
  | two
  | three

/-- Represents a bag of balls -/
structure Bag where
  ones : Nat
  twos : Nat
  threes : Nat

/-- Initial state of bag A -/
def bagA : Bag := ⟨3, 2, 1⟩

/-- Initial state of bag B -/
def bagB : Bag := ⟨2, 1, 1⟩

/-- Probability of selecting a ball with a specific label from a bag -/
def probSelect (bag : Bag) (label : Label) : Rat :=
  match label with
  | Label.one => bag.ones / (bag.ones + bag.twos + bag.threes)
  | Label.two => bag.twos / (bag.ones + bag.twos + bag.threes)
  | Label.three => bag.threes / (bag.ones + bag.twos + bag.threes)

/-- Probability of selecting a ball labeled 2 from bag B after transfer -/
def probSelectTwoAfterTransfer : Rat :=
  (probSelect bagA Label.one) * (probSelect ⟨bagB.ones + 1, bagB.twos, bagB.threes⟩ Label.two) +
  (probSelect bagA Label.two) * (probSelect ⟨bagB.ones, bagB.twos + 1, bagB.threes⟩ Label.two) +
  (probSelect bagA Label.three) * (probSelect ⟨bagB.ones, bagB.twos, bagB.threes + 1⟩ Label.two)

theorem prob_select_two_after_transfer :
  probSelectTwoAfterTransfer = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_two_after_transfer_l2035_203514


namespace NUMINAMATH_CALUDE_triangle_existence_and_area_l2035_203518

theorem triangle_existence_and_area 
  (a b c : ℝ) 
  (h : |a - Real.sqrt 8| + Real.sqrt (b^2 - 5) + (c - Real.sqrt 3)^2 = 0) : 
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_and_area_l2035_203518


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l2035_203561

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_two_digit_prime_factor_of_150_choose_75 :
  ∃ (p : ℕ), p = 47 ∧ 
    Prime p ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ binomial 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial 150 75 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_150_choose_75_l2035_203561


namespace NUMINAMATH_CALUDE_percentage_difference_l2035_203558

theorem percentage_difference (x y p : ℝ) (h : x = y * (1 + p / 100)) : 
  p = 100 * (x - y) / y :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2035_203558


namespace NUMINAMATH_CALUDE_cube_root_squared_eq_81_l2035_203526

theorem cube_root_squared_eq_81 (x : ℝ) :
  (x ^ (1/3)) ^ 2 = 81 → x = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_squared_eq_81_l2035_203526


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l2035_203549

theorem quadratic_single_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 7 * x + m = 0) ↔ m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l2035_203549


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2035_203570

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2035_203570


namespace NUMINAMATH_CALUDE_polynomial_value_at_3_l2035_203565

-- Define a monic polynomial of degree 4
def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_value_at_3 
  (p : ℝ → ℝ) 
  (h_monic : is_monic_degree_4 p) 
  (h1 : p 1 = 1) 
  (h2 : p (-1) = -1) 
  (h3 : p 2 = 2) 
  (h4 : p (-2) = -2) : 
  p 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_3_l2035_203565


namespace NUMINAMATH_CALUDE_quadratic_triple_root_l2035_203503

/-- For a quadratic equation ax^2 + bx + c = 0, one root is triple the other 
    if and only if 3b^2 = 16ac -/
theorem quadratic_triple_root (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) ↔ 
  3 * b^2 = 16 * a * c :=
sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_l2035_203503


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_ratio_l2035_203505

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem arithmetic_to_geometric_ratio 
  (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d ∧ 
  d ≠ 0 ∧
  (∀ n : ℕ, a n ≠ 0) ∧
  ((is_geometric_sequence (λ n => a n) ∧ is_geometric_sequence (λ n => a (n + 1))) ∨
   (is_geometric_sequence (λ n => a n) ∧ is_geometric_sequence (λ n => a (n + 2))) ∨
   (is_geometric_sequence (λ n => a (n + 1)) ∧ is_geometric_sequence (λ n => a (n + 2)))) →
  a 0 / d = 1 ∨ a 0 / d = -4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_ratio_l2035_203505


namespace NUMINAMATH_CALUDE_arcade_vending_machines_total_beverages_in_arcade_l2035_203511

/-- Given the conditions of vending machines in an arcade, calculate the total number of beverages --/
theorem arcade_vending_machines (num_machines : ℕ) 
  (front_position : ℕ) (back_position : ℕ) 
  (top_position : ℕ) (bottom_position : ℕ) : ℕ :=
  let beverages_per_column := front_position + back_position - 1
  let rows_per_machine := top_position + bottom_position - 1
  let beverages_per_machine := beverages_per_column * rows_per_machine
  num_machines * beverages_per_machine

/-- Prove that the total number of beverages in the arcade is 3696 --/
theorem total_beverages_in_arcade : 
  arcade_vending_machines 28 14 20 3 2 = 3696 := by
  sorry

end NUMINAMATH_CALUDE_arcade_vending_machines_total_beverages_in_arcade_l2035_203511


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2035_203540

/-- The distance between the vertices of the hyperbola x^2/144 - y^2/49 = 1 is 24 -/
theorem hyperbola_vertex_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/144 - y^2/49 = 1}
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ hyperbola ∧ v2 ∈ hyperbola ∧ ‖v1 - v2‖ = 24 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2035_203540


namespace NUMINAMATH_CALUDE_shaded_area_problem_l2035_203502

theorem shaded_area_problem (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 4 →
  triangle_base = 4 →
  triangle_height = 3 →
  square_side * square_side - (1 / 2 * triangle_base * triangle_height) = 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l2035_203502


namespace NUMINAMATH_CALUDE_roots_squared_relation_l2035_203571

-- Define the polynomials h(x) and p(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem roots_squared_relation (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → p a b c (x^2) = 0) →
  a = -1 ∧ b = -2 ∧ c = 16 :=
by sorry

end NUMINAMATH_CALUDE_roots_squared_relation_l2035_203571


namespace NUMINAMATH_CALUDE_solve_equation_l2035_203519

theorem solve_equation : ∃ x : ℝ, (2 * x + 5) / 7 = 15 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2035_203519


namespace NUMINAMATH_CALUDE_qian_receives_23_yuan_l2035_203578

/-- Represents the amount of money paid by each person for each meal -/
structure MealPayments where
  zhao_lunch : ℕ
  qian_lunch : ℕ
  sun_lunch : ℕ
  zhao_dinner : ℕ
  qian_dinner : ℕ

/-- Calculates the amount Qian should receive from Li -/
def amount_qian_receives (payments : MealPayments) : ℕ :=
  let total_cost := payments.zhao_lunch + payments.qian_lunch + payments.sun_lunch +
                    payments.zhao_dinner + payments.qian_dinner
  let cost_per_person := total_cost / 4
  let qian_paid := payments.qian_lunch + payments.qian_dinner
  qian_paid - cost_per_person

/-- The main theorem stating that Qian should receive 23 yuan from Li -/
theorem qian_receives_23_yuan (payments : MealPayments) 
  (h1 : payments.zhao_lunch = 23)
  (h2 : payments.qian_lunch = 41)
  (h3 : payments.sun_lunch = 56)
  (h4 : payments.zhao_dinner = 48)
  (h5 : payments.qian_dinner = 32) :
  amount_qian_receives payments = 23 := by
  sorry


end NUMINAMATH_CALUDE_qian_receives_23_yuan_l2035_203578


namespace NUMINAMATH_CALUDE_max_cables_for_given_network_l2035_203508

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  total_employees : ℕ
  brand_x_count : ℕ
  brand_y_count : ℕ
  max_connections_per_computer : ℕ
  (total_is_sum : total_employees = brand_x_count + brand_y_count)
  (max_connections_positive : max_connections_per_computer > 0)

/-- The maximum number of cables that can be used in the network. -/
def max_cables (network : ComputerNetwork) : ℕ :=
  min (network.brand_x_count * network.max_connections_per_computer)
      (network.brand_y_count * network.max_connections_per_computer)

/-- The theorem stating the maximum number of cables for the given network configuration. -/
theorem max_cables_for_given_network :
  ∃ (network : ComputerNetwork),
    network.total_employees = 40 ∧
    network.brand_x_count = 25 ∧
    network.brand_y_count = 15 ∧
    network.max_connections_per_computer = 3 ∧
    max_cables network = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_cables_for_given_network_l2035_203508


namespace NUMINAMATH_CALUDE_max_G_ratio_is_six_fifths_l2035_203572

/-- Represents a four-digit number --/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧ 
             hundreds ≥ 0 ∧ hundreds ≤ 9 ∧ 
             tens ≥ 0 ∧ tens ≤ 9 ∧ 
             units ≥ 0 ∧ units ≤ 9

/-- Defines a "difference 2 multiple" --/
def isDifference2Multiple (n : FourDigitNumber) : Prop :=
  n.thousands - n.hundreds = 2 ∧ n.tens - n.units = 4

/-- Defines a "difference 3 multiple" --/
def isDifference3Multiple (n : FourDigitNumber) : Prop :=
  n.thousands - n.hundreds = 3 ∧ n.tens - n.units = 6

/-- Calculates the sum of digits --/
def G (n : FourDigitNumber) : Nat :=
  n.thousands + n.hundreds + n.tens + n.units

/-- Calculates F(p,q) --/
def F (p q : FourDigitNumber) : Int :=
  (1000 * p.thousands + 100 * p.hundreds + 10 * p.tens + p.units -
   (1000 * q.thousands + 100 * q.hundreds + 10 * q.tens + q.units)) / 10

/-- Main theorem --/
theorem max_G_ratio_is_six_fifths 
  (p q : FourDigitNumber) 
  (h1 : isDifference2Multiple p)
  (h2 : isDifference3Multiple q)
  (h3 : p.units = 3)
  (h4 : q.units = 3)
  (h5 : ∃ k : Int, F p q / (G p - G q + 3) = k) :
  ∀ (p' q' : FourDigitNumber), 
    isDifference2Multiple p' → 
    isDifference3Multiple q' → 
    p'.units = 3 → 
    q'.units = 3 → 
    (∃ k : Int, F p' q' / (G p' - G q' + 3) = k) → 
    (G p : ℚ) / (G q) ≥ (G p' : ℚ) / (G q') := by
  sorry

end NUMINAMATH_CALUDE_max_G_ratio_is_six_fifths_l2035_203572


namespace NUMINAMATH_CALUDE_part_one_part_two_l2035_203544

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (0, 1)
def c : ℝ × ℝ := (1, -2)

/-- The theorem for the first part of the problem -/
theorem part_one : ∃ (m n : ℝ), a = m • b + n • c ∧ m = 3 ∧ n = 2 := by sorry

/-- The theorem for the second part of the problem -/
theorem part_two : 
  (∃ (d : ℝ × ℝ), ∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) ∧ 
  (∀ (d : ℝ × ℝ), (∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) → 
    Real.sqrt 2 / 2 ≤ Real.sqrt (d.1^2 + d.2^2)) ∧
  (∃ (d : ℝ × ℝ), (∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) ∧ 
    Real.sqrt (d.1^2 + d.2^2) = Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2035_203544


namespace NUMINAMATH_CALUDE_max_F_value_l2035_203531

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_four_digit : thousands ≥ 1 ∧ thousands ≤ 9 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Defines an eternal number -/
def is_eternal (m : FourDigitNumber) : Prop :=
  m.hundreds + m.tens + m.units = 12

/-- Swaps digits to create N -/
def swap_digits (m : FourDigitNumber) : FourDigitNumber :=
  { thousands := m.hundreds,
    hundreds := m.thousands,
    tens := m.units,
    units := m.tens,
    is_four_digit := by sorry }

/-- Defines the function F(M) -/
def F (m : FourDigitNumber) : Int :=
  let n := swap_digits m
  let m_value := 1000 * m.thousands + 100 * m.hundreds + 10 * m.tens + m.units
  let n_value := 1000 * n.thousands + 100 * n.hundreds + 10 * n.tens + n.units
  (m_value - n_value) / 9

/-- Main theorem -/
theorem max_F_value (m : FourDigitNumber) 
  (h_eternal : is_eternal m)
  (h_diff : m.hundreds - m.units = m.thousands)
  (h_div : (F m) % 9 = 0) :
  F m ≤ 9 ∧ ∃ (m' : FourDigitNumber), is_eternal m' ∧ m'.hundreds - m'.units = m'.thousands ∧ (F m') % 9 = 0 ∧ F m' = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_F_value_l2035_203531


namespace NUMINAMATH_CALUDE_larger_square_perimeter_l2035_203545

theorem larger_square_perimeter
  (small_square_perimeter : ℝ)
  (shaded_area : ℝ)
  (h1 : small_square_perimeter = 72)
  (h2 : shaded_area = 160) :
  let small_side := small_square_perimeter / 4
  let small_area := small_side ^ 2
  let large_area := small_area + shaded_area
  let large_side := Real.sqrt large_area
  let large_perimeter := 4 * large_side
  large_perimeter = 88 := by
sorry

end NUMINAMATH_CALUDE_larger_square_perimeter_l2035_203545


namespace NUMINAMATH_CALUDE_wire_division_l2035_203555

/-- Given a wire that can be divided into two parts of 120 cm each with 2.4 cm left over,
    prove that when divided into three equal parts, each part is 80.8 cm long. -/
theorem wire_division (wire_length : ℝ) (h1 : wire_length = 2 * 120 + 2.4) :
  wire_length / 3 = 80.8 := by
sorry

end NUMINAMATH_CALUDE_wire_division_l2035_203555


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l2035_203523

def g (x : ℝ) := 10 * x^4 - 16 * x^2 + 3

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/5) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l2035_203523


namespace NUMINAMATH_CALUDE_unique_valid_triple_l2035_203552

/-- Represents an ordered triple of integers (a, b, c) satisfying the given conditions -/
structure ValidTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  a_ge_2 : a ≥ 2
  b_ge_1 : b ≥ 1
  log_cond : (Real.log b) / (Real.log a) = c^2
  sum_cond : a + b + c = 100

/-- There exists exactly one ordered triple of integers satisfying the given conditions -/
theorem unique_valid_triple : ∃! t : ValidTriple, True := by sorry

end NUMINAMATH_CALUDE_unique_valid_triple_l2035_203552


namespace NUMINAMATH_CALUDE_monthly_interest_rate_equation_l2035_203597

/-- The monthly interest rate that satisfies the compound interest equation for a loan of $200 with $22 interest charged in the second month. -/
theorem monthly_interest_rate_equation : ∃ r : ℝ, 200 * (1 + r)^2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_monthly_interest_rate_equation_l2035_203597


namespace NUMINAMATH_CALUDE_system_solution_existence_l2035_203500

theorem system_solution_existence (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |x| + |y| = |b|) ↔ |a| ≤ |b| ∧ |b| ≤ Real.sqrt 2 * |a| :=
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2035_203500


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2035_203528

/-- A rhombus with perimeter 32 has side length 8 -/
theorem rhombus_side_length (perimeter : ℝ) (h1 : perimeter = 32) : perimeter / 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2035_203528


namespace NUMINAMATH_CALUDE_inverse_proportion_m_range_l2035_203527

/-- Given an inverse proportion function y = (1-m)/x passing through points (1, y₁) and (2, y₂),
    where y₁ > y₂, prove that m < 1 -/
theorem inverse_proportion_m_range (y₁ y₂ m : ℝ) : 
  y₁ = 1 - m → 
  y₂ = (1 - m) / 2 → 
  y₁ > y₂ → 
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_range_l2035_203527


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2035_203517

theorem power_fraction_equality : 
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73/89 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2035_203517


namespace NUMINAMATH_CALUDE_smallest_multiple_l2035_203563

theorem smallest_multiple : ∃ (a : ℕ), 
  (a % 3 = 0) ∧ 
  ((a - 1) % 4 = 0) ∧ 
  ((a - 2) % 5 = 0) ∧ 
  (∀ b : ℕ, b < a → ¬((b % 3 = 0) ∧ ((b - 1) % 4 = 0) ∧ ((b - 2) % 5 = 0))) ∧
  a = 57 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2035_203563


namespace NUMINAMATH_CALUDE_fruit_basket_combinations_l2035_203524

/-- The number of ways to choose apples for a fruit basket -/
def apple_choices : ℕ := 3

/-- The number of ways to choose oranges for a fruit basket -/
def orange_choices : ℕ := 8

/-- The total number of fruit basket combinations -/
def total_combinations : ℕ := apple_choices * orange_choices

/-- Theorem stating the number of possible fruit baskets -/
theorem fruit_basket_combinations :
  total_combinations = 36 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_combinations_l2035_203524


namespace NUMINAMATH_CALUDE_triangle_inequality_from_condition_l2035_203546

theorem triangle_inequality_from_condition 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : ∀ (A B C : ℝ), A > 0 → B > 0 → C > 0 → 
    A * a * (B * b + C * c) + B * b * (C * c + A * a) + C * c * (A * a + B * b) > 
    (1/2) * (A * B * c^2 + B * C * a^2 + C * A * b^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_condition_l2035_203546


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2035_203510

/-- Proves that for a rectangular plot with given conditions, the length is 60 meters more than the breadth. -/
theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 80 ∧ 
  length > breadth ∧ 
  (4 * breadth + 2 * (length - breadth)) * 26.5 = 5300 →
  length - breadth = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l2035_203510


namespace NUMINAMATH_CALUDE_cos_shift_l2035_203587

open Real

theorem cos_shift (f g : ℝ → ℝ) : 
  (∀ x, f x = cos (2 * x + π / 4)) → 
  (∀ x, g x = (1 / 2) * (deriv f x)) → 
  (∀ x, f x = g (x - π / 4)) := by
sorry

end NUMINAMATH_CALUDE_cos_shift_l2035_203587


namespace NUMINAMATH_CALUDE_evaluate_expression_l2035_203521

theorem evaluate_expression (S : ℝ) : 
  S = 1 / (4 - Real.sqrt 10) - 1 / (Real.sqrt 10 - Real.sqrt 9) + 
      1 / (Real.sqrt 9 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
      1 / (Real.sqrt 7 - 3) → 
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2035_203521


namespace NUMINAMATH_CALUDE_binomial_coefficient_properties_l2035_203548

theorem binomial_coefficient_properties (p : ℕ) (hp : Nat.Prime p) :
  (∀ k, p ∣ (Nat.choose (p - 1) k ^ 2 - 1)) ∧
  (∀ s, Even s → p ∣ (Finset.sum (Finset.range p) (λ k => Nat.choose (p - 1) k ^ s))) ∧
  (∀ s, Odd s → (Finset.sum (Finset.range p) (λ k => Nat.choose (p - 1) k ^ s)) % p = 1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_properties_l2035_203548


namespace NUMINAMATH_CALUDE_sequence_ratio_l2035_203588

/-- Given two sequences, one arithmetic and one geometric, prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℚ) : 
  ((-1 : ℚ) - a₁ = a₁ - a₂) ∧ 
  (a₁ - a₂ = a₂ - (-4)) ∧ 
  ((-1 : ℚ) * b₁ = b₁ * b₂) ∧ 
  (b₁ * b₂ = b₂ * b₃) ∧ 
  (b₂ * b₃ = b₃ * (-4)) → 
  (a₂ - a₁) / b₂ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2035_203588


namespace NUMINAMATH_CALUDE_platform_height_l2035_203535

/-- Given two configurations of identical rectangular prisms on a platform,
    prove that the platform height is 37 inches. -/
theorem platform_height (l w : ℝ) : 
  l + 37 - w = 40 → w + 37 - l = 34 → 37 = 37 := by
  sorry

end NUMINAMATH_CALUDE_platform_height_l2035_203535


namespace NUMINAMATH_CALUDE_equation_solution_l2035_203598

theorem equation_solution (x : ℝ) : 
  |x - 3| + x^2 = 10 ↔ 
  x = (-1 + Real.sqrt 53) / 2 ∨ 
  x = (1 + Real.sqrt 29) / 2 ∨ 
  x = (1 - Real.sqrt 29) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2035_203598


namespace NUMINAMATH_CALUDE_point_set_equivalence_l2035_203522

theorem point_set_equivalence (x y : ℝ) : 
  y^2 - y = x^2 - x ↔ y = x ∨ y = 1 - x := by sorry

end NUMINAMATH_CALUDE_point_set_equivalence_l2035_203522


namespace NUMINAMATH_CALUDE_cyclists_max_daily_distance_l2035_203550

theorem cyclists_max_daily_distance (distance_to_boston distance_to_atlanta : ℕ) 
  (h1 : distance_to_boston = 840) 
  (h2 : distance_to_atlanta = 440) : 
  (Nat.gcd distance_to_boston distance_to_atlanta) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_max_daily_distance_l2035_203550


namespace NUMINAMATH_CALUDE_mary_initial_weight_l2035_203568

/-- Mary's weight changes and final weight --/
structure WeightChanges where
  initial_loss : ℕ
  final_weight : ℕ

/-- Calculate Mary's initial weight given her weight changes --/
def calculate_initial_weight (changes : WeightChanges) : ℕ :=
  changes.final_weight         -- Start with final weight
  + changes.initial_loss * 3   -- Add back the triple loss
  - changes.initial_loss * 2   -- Subtract the double gain
  - 6                          -- Subtract the final gain
  + changes.initial_loss       -- Add back the initial loss

/-- Theorem stating that Mary's initial weight was 99 pounds --/
theorem mary_initial_weight :
  let changes : WeightChanges := { initial_loss := 12, final_weight := 81 }
  calculate_initial_weight changes = 99 := by
  sorry


end NUMINAMATH_CALUDE_mary_initial_weight_l2035_203568


namespace NUMINAMATH_CALUDE_apple_buying_problem_l2035_203507

/-- Proves that given the conditions of the apple-buying problem, each man bought 30 apples. -/
theorem apple_buying_problem (men women man_apples woman_apples total_apples : ℕ) 
  (h1 : men = 2)
  (h2 : women = 3)
  (h3 : man_apples + 20 = woman_apples)
  (h4 : men * man_apples + women * woman_apples = total_apples)
  (h5 : total_apples = 210) :
  man_apples = 30 := by
sorry

end NUMINAMATH_CALUDE_apple_buying_problem_l2035_203507


namespace NUMINAMATH_CALUDE_cycle_original_price_l2035_203537

/-- The original price of a cycle sold at a loss -/
def original_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem: The original price of a cycle is 1750, given a selling price of 1610 and a loss of 8% -/
theorem cycle_original_price : 
  original_price 1610 8 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l2035_203537


namespace NUMINAMATH_CALUDE_cereal_servings_l2035_203589

def cereal_problem (total_cereal : ℝ) (serving_size : ℝ) : Prop :=
  total_cereal = 24.5 ∧ 
  serving_size = 1.75 →
  (total_cereal / serving_size : ℝ) = 14

theorem cereal_servings : cereal_problem 24.5 1.75 := by
  sorry

end NUMINAMATH_CALUDE_cereal_servings_l2035_203589


namespace NUMINAMATH_CALUDE_donny_savings_l2035_203559

theorem donny_savings (monday : ℕ) (wednesday : ℕ) (thursday_spent : ℕ) :
  monday = 15 →
  wednesday = 13 →
  thursday_spent = 28 →
  ∃ tuesday : ℕ, 
    tuesday = 28 ∧ 
    monday + tuesday + wednesday = 2 * thursday_spent :=
by sorry

end NUMINAMATH_CALUDE_donny_savings_l2035_203559


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2035_203513

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (c - y) = (a - y) * (b - y) →
  x / y = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2035_203513


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2035_203541

/-- The complex number z is in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + (1 - i) / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2035_203541


namespace NUMINAMATH_CALUDE_minimize_y_l2035_203582

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2

/-- The theorem stating that (a+b)/2 minimizes y -/
theorem minimize_y (a b : ℝ) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y x_min a b ≤ y x a b ∧ x_min = (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_y_l2035_203582


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2035_203583

/-- Given Oscar's and Charlie's bus ride lengths, prove the difference between them -/
theorem bus_ride_difference (oscar_ride : ℝ) (charlie_ride : ℝ)
  (h1 : oscar_ride = 0.75)
  (h2 : charlie_ride = 0.25) :
  oscar_ride - charlie_ride = 0.50 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2035_203583


namespace NUMINAMATH_CALUDE_dinner_cost_is_120_l2035_203538

/-- Calculates the cost of dinner before tip given the total cost, ticket price, number of tickets, limo hourly rate, limo hours, and tip percentage. -/
def dinner_cost (total : ℚ) (ticket_price : ℚ) (num_tickets : ℕ) (limo_rate : ℚ) (limo_hours : ℕ) (tip_percent : ℚ) : ℚ :=
  let ticket_cost := ticket_price * num_tickets
  let limo_cost := limo_rate * limo_hours
  let dinner_with_tip := total - (ticket_cost + limo_cost)
  dinner_with_tip / (1 + tip_percent)

/-- Proves that the cost of dinner before tip is $120 given the specified conditions. -/
theorem dinner_cost_is_120 :
  dinner_cost 836 100 2 80 6 (30/100) = 120 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_is_120_l2035_203538


namespace NUMINAMATH_CALUDE_inequality_solution_range_of_a_l2035_203575

-- Define the functions f and g
def f (x : ℝ) := |x - 4|
def g (x : ℝ) := |2*x + 1|

-- Theorem for the first part of the problem
theorem inequality_solution :
  ∀ x : ℝ, f x < g x ↔ x < -5 ∨ x > 1 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9/4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_of_a_l2035_203575


namespace NUMINAMATH_CALUDE_red_peaches_count_l2035_203551

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := sorry

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 11

/-- The difference between green and red peaches -/
def difference : ℕ := 6

/-- Theorem stating that the number of red peaches is 5 -/
theorem red_peaches_count : red_peaches = 5 := by
  sorry

/-- The relationship between green and red peaches -/
axiom green_red_relation : green_peaches = red_peaches + difference


end NUMINAMATH_CALUDE_red_peaches_count_l2035_203551


namespace NUMINAMATH_CALUDE_semicircle_chord_projection_l2035_203553

/-- Given a semicircle with diameter 2R and a chord intersecting the semicircle and its tangent,
    prove that the condition AC^2 + CD^2 + BD^2 = 4a^2 has a solution for the projection of C on AB
    if and only if a^2 ≥ R^2, and that this solution is unique. -/
theorem semicircle_chord_projection (R a : ℝ) (h : R > 0) :
  ∃! x, x > 0 ∧ x < 2*R ∧ 
    2*R*x + (4*R^2*(2*R - x)^2)/x^2 + (4*R^2*(2*R - x))/x = 4*a^2 ↔ 
  a^2 ≥ R^2 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_chord_projection_l2035_203553


namespace NUMINAMATH_CALUDE_orangeade_price_day1_l2035_203533

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℚ
  day : ℕ

/-- Represents the amount of orangeade made on a given day -/
structure OrangeadeAmount where
  amount : ℚ
  day : ℕ

/-- Represents the revenue from selling orangeade on a given day -/
def revenue (price : OrangeadePrice) (amount : OrangeadeAmount) : ℚ :=
  price.price * amount.amount

theorem orangeade_price_day1 (juice : ℚ) 
  (amount_day1 : OrangeadeAmount) 
  (amount_day2 : OrangeadeAmount)
  (price_day1 : OrangeadePrice)
  (price_day2 : OrangeadePrice) :
  amount_day1.amount = 2 * juice →
  amount_day2.amount = 3 * juice →
  amount_day1.day = 1 →
  amount_day2.day = 2 →
  price_day1.day = 1 →
  price_day2.day = 2 →
  price_day2.price = 2/5 →
  revenue price_day1 amount_day1 = revenue price_day2 amount_day2 →
  price_day1.price = 3/5 := by
  sorry

#eval (3 : ℚ) / 5  -- Should output 0.6

end NUMINAMATH_CALUDE_orangeade_price_day1_l2035_203533


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2035_203525

theorem simplify_polynomial (w : ℝ) : 
  3*w + 4 - 6*w - 5 + 7*w + 8 - 9*w - 10 + 2*w^2 = 2*w^2 - 5*w - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2035_203525


namespace NUMINAMATH_CALUDE_equation_solution_l2035_203504

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x - 3 = 5 ∧ x = 112 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2035_203504


namespace NUMINAMATH_CALUDE_black_ants_count_l2035_203595

theorem black_ants_count (total_ants red_ants : ℕ) 
  (h1 : total_ants = 900) 
  (h2 : red_ants = 413) : 
  total_ants - red_ants = 487 := by
  sorry

end NUMINAMATH_CALUDE_black_ants_count_l2035_203595


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l2035_203585

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateSampleSize (workshops : List Workshop) (sampledUnits : ℕ) (sampledWorkshopQuantity : ℕ) : ℕ :=
  let totalQuantity := workshops.map (·.quantity) |>.sum
  (sampledUnits * totalQuantity) / sampledWorkshopQuantity

theorem stratified_sampling_size :
  let workshops := [
    { quantity := 120 },  -- Workshop A
    { quantity := 80 },   -- Workshop B
    { quantity := 60 }    -- Workshop C
  ]
  let sampledUnits := 3
  let sampledWorkshopQuantity := 60
  calculateSampleSize workshops sampledUnits sampledWorkshopQuantity = 13 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_size_l2035_203585


namespace NUMINAMATH_CALUDE_segment_distance_sum_l2035_203599

/-- Represents a line segment with a midpoint -/
structure Segment where
  length : ℝ
  midpoint : ℝ

/-- The function relating distances from midpoints -/
def distance_relation (x y : ℝ) : Prop := y / x = 5 / 3

theorem segment_distance_sum 
  (ab : Segment) 
  (a'b' : Segment) 
  (h1 : ab.length = 3) 
  (h2 : a'b'.length = 5) 
  (x : ℝ) 
  (y : ℝ) 
  (h3 : distance_relation x y) 
  (h4 : x = 2) : 
  x + y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_segment_distance_sum_l2035_203599


namespace NUMINAMATH_CALUDE_max_distance_is_25km_l2035_203569

def car_position (t : ℝ) : ℝ := 40 * t

def motorcycle_position (t : ℝ) : ℝ := 16 * t^2 + 9

def distance (t : ℝ) : ℝ := |motorcycle_position t - car_position t|

theorem max_distance_is_25km :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 2 → distance t ≥ distance s ∧
  distance t = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_is_25km_l2035_203569


namespace NUMINAMATH_CALUDE_marco_juice_mixture_l2035_203556

/-- Calculates the remaining mixture after giving some away -/
def remaining_mixture (apple_juice orange_juice given_away : ℚ) : ℚ :=
  apple_juice + orange_juice - given_away

/-- Proves that the remaining mixture is 13/4 gallons -/
theorem marco_juice_mixture :
  let apple_juice : ℚ := 4
  let orange_juice : ℚ := 7/4
  let given_away : ℚ := 5/2
  remaining_mixture apple_juice orange_juice given_away = 13/4 := by
sorry

end NUMINAMATH_CALUDE_marco_juice_mixture_l2035_203556


namespace NUMINAMATH_CALUDE_digit_divisible_by_9_l2035_203579

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

theorem digit_divisible_by_9 :
  is_divisible_by_9 5274 ∧ 
  ∀ B : ℕ, B ≤ 9 → B ≠ 4 → ¬(is_divisible_by_9 (5270 + B)) :=
by sorry

end NUMINAMATH_CALUDE_digit_divisible_by_9_l2035_203579


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l2035_203512

theorem smaller_solution_quadratic : ∃ (x y : ℝ), 
  x < y ∧ 
  x^2 - 12*x - 28 = 0 ∧ 
  y^2 - 12*y - 28 = 0 ∧
  x = -2 ∧
  ∀ z : ℝ, z^2 - 12*z - 28 = 0 → z = x ∨ z = y := by
  sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l2035_203512


namespace NUMINAMATH_CALUDE_sum_ratio_equality_l2035_203574

theorem sum_ratio_equality (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_sum_ratio_equality_l2035_203574


namespace NUMINAMATH_CALUDE_train_length_l2035_203566

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 9 → ∃ (length : ℝ), abs (length - 150.03) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l2035_203566


namespace NUMINAMATH_CALUDE_total_loaves_served_l2035_203576

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real) :
  wheat_bread = 0.2 → white_bread = 0.4 → wheat_bread + white_bread = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l2035_203576


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2035_203581

theorem and_sufficient_not_necessary_for_or :
  (∀ p q : Prop, p ∧ q → p ∨ q) ∧
  (∃ p q : Prop, p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2035_203581


namespace NUMINAMATH_CALUDE_problem_solving_probability_l2035_203506

theorem problem_solving_probability :
  let p_xavier : ℚ := 1/4
  let p_yvonne : ℚ := 2/3
  let p_zelda : ℚ := 5/8
  let p_william : ℚ := 7/10
  (p_xavier * p_yvonne * p_william * (1 - p_zelda) : ℚ) = 7/160 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l2035_203506


namespace NUMINAMATH_CALUDE_new_oranges_added_l2035_203515

theorem new_oranges_added (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 50 → thrown_away = 40 → final = 34 → 
  final - (initial - thrown_away) = 24 := by
  sorry

end NUMINAMATH_CALUDE_new_oranges_added_l2035_203515


namespace NUMINAMATH_CALUDE_solution_set_m_2_range_of_m_l2035_203539

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| + |2 * x + m|

-- Theorem 1: Solution set for f(x) ≤ 3 when m = 2
theorem solution_set_m_2 :
  {x : ℝ | f x 2 ≤ 3} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem 2: Range of m values for f(x) ≤ |2x - 3| with x ∈ [0, 1]
theorem range_of_m :
  {m : ℝ | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x m ≤ |2 * x - 3|} = {m : ℝ | -3 ≤ m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_m_2_range_of_m_l2035_203539


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_proof_l2035_203560

/-- Calculates the speed of a man given the parameters of a train passing him. -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- Given the specific parameters, proves that the man's speed is approximately 0.832 m/s. -/
theorem man_speed_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |man_speed_calculation 700 63 41.9966402687785 - 0.832| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_proof_l2035_203560


namespace NUMINAMATH_CALUDE_cookies_eaten_l2035_203554

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem cookies_eaten :
  initial_cookies - remaining_cookies = 9 :=
by sorry

end NUMINAMATH_CALUDE_cookies_eaten_l2035_203554


namespace NUMINAMATH_CALUDE_construction_rearrangements_l2035_203567

def word : String := "CONSTRUCTION"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  vowels.length.factorial

def consonant_arrangements : ℕ :=
  consonants.length.factorial / ((consonants.countP (· = 'C')).factorial *
                                 (consonants.countP (· = 'T')).factorial *
                                 (consonants.countP (· = 'N')).factorial)

theorem construction_rearrangements :
  vowel_arrangements * consonant_arrangements = 30240 := by
  sorry

end NUMINAMATH_CALUDE_construction_rearrangements_l2035_203567


namespace NUMINAMATH_CALUDE_expression_evaluation_l2035_203596

theorem expression_evaluation (x : ℝ) (h : x > 2) :
  Real.sqrt (x^2 / (1 - (x^2 - 4) / x^2)) = x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2035_203596


namespace NUMINAMATH_CALUDE_translation_problem_l2035_203590

-- Part 1
def part1 (A B A' B' : ℝ × ℝ) : Prop :=
  A = (-2, -1) ∧ B = (1, -3) ∧ A' = (2, 3) → B' = (5, 1)

-- Part 2
def part2 (A B A' B' : ℝ × ℝ) (m n : ℝ) : Prop :=
  A = (m, n) ∧ B = (2*n, m) ∧ A' = (3*m, n) ∧ B' = (6*n, m) → m = 2*n

-- Part 3
def part3 (A B A' B' : ℝ × ℝ) (m n : ℝ) : Prop :=
  A = (m, n+1) ∧ B = (n-1, n-2) ∧ A' = (2*n-5, 2*m+3) ∧ B' = (2*m+3, n+3) →
  A = (6, 10) ∧ B = (8, 7)

theorem translation_problem :
  ∀ (A B A' B' : ℝ × ℝ) (m n : ℝ),
    part1 A B A' B' ∧
    part2 A B A' B' m n ∧
    part3 A B A' B' m n :=
by sorry

end NUMINAMATH_CALUDE_translation_problem_l2035_203590


namespace NUMINAMATH_CALUDE_bobbys_blocks_l2035_203501

/-- The number of blocks Bobby's father gave him -/
def blocks_from_father (initial_blocks final_blocks : ℕ) : ℕ :=
  final_blocks - initial_blocks

/-- Proof that Bobby's father gave him 6 blocks -/
theorem bobbys_blocks :
  blocks_from_father 2 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_blocks_l2035_203501


namespace NUMINAMATH_CALUDE_marbles_lost_found_difference_l2035_203577

/-- Given Josh's marble collection scenario, prove the difference between lost and found marbles. -/
theorem marbles_lost_found_difference (initial : ℕ) (lost : ℕ) (found : ℕ) 
  (h1 : initial = 4)
  (h2 : lost = 16)
  (h3 : found = 8) :
  lost - found = 8 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_found_difference_l2035_203577


namespace NUMINAMATH_CALUDE_birthday_money_l2035_203534

theorem birthday_money (age : ℕ) (money : ℕ) : 
  age = 3 * 3 →
  money = 5 * age →
  money = 45 := by
sorry

end NUMINAMATH_CALUDE_birthday_money_l2035_203534


namespace NUMINAMATH_CALUDE_points_in_first_quadrant_l2035_203562

theorem points_in_first_quadrant (x y : ℝ) : 
  y > -x + 3 ∧ y > 3*x - 1 → x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_CALUDE_points_in_first_quadrant_l2035_203562


namespace NUMINAMATH_CALUDE_no_integer_solution_for_divisibility_l2035_203529

theorem no_integer_solution_for_divisibility : ¬∃ (x y : ℤ), (x^2 + y^2 + x + y) ∣ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_divisibility_l2035_203529


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2035_203547

theorem billion_to_scientific_notation :
  ∀ (x : ℝ), x = 26.62 * 1000000000 → x = 2.662 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2035_203547


namespace NUMINAMATH_CALUDE_grocer_sale_problem_l2035_203586

theorem grocer_sale_problem (sale1 sale2 sale3 sale5 average_sale : ℕ) 
  (h1 : sale1 = 5700)
  (h2 : sale2 = 8550)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 14045)
  (h_avg : average_sale = 7800) :
  ∃ sale4 : ℕ, 
    sale4 = 3850 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = average_sale :=
by sorry

end NUMINAMATH_CALUDE_grocer_sale_problem_l2035_203586


namespace NUMINAMATH_CALUDE_part_one_part_two_l2035_203530

/-- Given positive numbers a, b, c, d such that ad = bc and a + d > b + c, 
    then |a - d| > |b - c| -/
theorem part_one (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_ad_bc : a * d = b * c) (h_sum : a + d > b + c) :
  |a - d| > |b - c| := by sorry

/-- Given positive numbers a, b, c, d and a real number t such that 
    t * √(a² + b²) * √(c² + d²) = √(a⁴ + c⁴) + √(b⁴ + d⁴), then t ≥ √2 -/
theorem part_two (a b c d t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_eq : t * Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2) = 
          Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)) :
  t ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2035_203530


namespace NUMINAMATH_CALUDE_train_length_calculation_second_train_length_l2035_203594

/-- Calculates the length of the second train given the conditions of the problem -/
theorem train_length_calculation (length1 : ℝ) (speed1 speed2 : ℝ) (crossing_time : ℝ) : ℝ :=
  let km_per_hr_to_m_per_s : ℝ := 1000 / 3600
  let speed1_m_per_s : ℝ := speed1 * km_per_hr_to_m_per_s
  let speed2_m_per_s : ℝ := speed2 * km_per_hr_to_m_per_s
  let relative_speed : ℝ := speed1_m_per_s + speed2_m_per_s
  let total_distance : ℝ := relative_speed * crossing_time
  total_distance - length1

/-- The length of the second train is approximately 160 meters -/
theorem second_train_length :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_length_calculation 140 60 40 10.799136069114471 - 160| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_second_train_length_l2035_203594


namespace NUMINAMATH_CALUDE_sum_of_combinations_l2035_203580

theorem sum_of_combinations : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l2035_203580


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2035_203516

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2035_203516
