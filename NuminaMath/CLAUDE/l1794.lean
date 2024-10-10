import Mathlib

namespace max_visible_cubes_12_l1794_179426

/-- Represents a cube formed by unit cubes --/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from any single point --/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  3 * cube.size^2 - 3 * (cube.size - 1) + 1

/-- Theorem stating that for a 12 × 12 × 12 cube, the maximum number of visible unit cubes is 400 --/
theorem max_visible_cubes_12 :
  max_visible_cubes { size := 12 } = 400 := by
  sorry

#eval max_visible_cubes { size := 12 }

end max_visible_cubes_12_l1794_179426


namespace seventh_root_unity_sum_l1794_179471

theorem seventh_root_unity_sum (q : ℂ) (h : q^7 = 1) :
  q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6) = 
    if q = 1 then (3 : ℂ) / 2 else -2 := by sorry

end seventh_root_unity_sum_l1794_179471


namespace arithmetic_sequence_properties_l1794_179450

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties :
  ∀ d : ℤ,
  (arithmetic_sequence 23 d 6 > 0) →
  (arithmetic_sequence 23 d 7 < 0) →
  (d = -4) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 23 d n ≤ 78) ∧
  (∀ n : ℕ, n ≤ 12 ↔ sum_arithmetic_sequence 23 d n > 0) :=
sorry

end arithmetic_sequence_properties_l1794_179450


namespace floor_tiles_l1794_179409

theorem floor_tiles (n : ℕ) : 
  n % 3 = 0 ∧ 
  2 * (2 * n / 3) - 1 = 49 → 
  n^2 - (n / 3)^2 = 1352 := by
sorry

end floor_tiles_l1794_179409


namespace certain_number_calculation_l1794_179444

theorem certain_number_calculation : 5 * 3 + 4 = 19 := by
  sorry

end certain_number_calculation_l1794_179444


namespace correct_product_after_decimal_error_l1794_179418

theorem correct_product_after_decimal_error (incorrect_product : ℝ) 
  (h1 : incorrect_product = 12.04) : 
  ∃ (factor1 factor2 : ℝ), 
    (0.01 ≤ factor1 ∧ factor1 < 1) ∧ 
    (factor1 * 100 * factor2 = incorrect_product) ∧
    (factor1 * factor2 = 0.1204) := by
  sorry

end correct_product_after_decimal_error_l1794_179418


namespace radical_product_simplification_l1794_179479

theorem radical_product_simplification (m : ℝ) (h : m > 0) :
  Real.sqrt (50 * m) * Real.sqrt (5 * m) * Real.sqrt (45 * m) = 15 * m * Real.sqrt (10 * m) :=
by sorry

end radical_product_simplification_l1794_179479


namespace problem_solution_l1794_179474

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x) else x - 4 / x

theorem problem_solution (a : ℝ) :
  (a = 1 → ∃! x, f a x = 3 ∧ x = 4) ∧
  (a ≤ -1 →
    (∃ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ ∧
      f a x₁ = 3 ∧ f a x₂ = 3 ∧ f a x₃ = 3 ∧
      x₃ - x₂ = x₂ - x₁) →
    a = -11/6) :=
sorry

end problem_solution_l1794_179474


namespace fraction_of_muscle_gain_as_fat_l1794_179458

/-- Calculates the fraction of muscle gain that is fat given initial weight, muscle gain percentage, and final weight. -/
theorem fraction_of_muscle_gain_as_fat 
  (initial_weight : ℝ) 
  (muscle_gain_percentage : ℝ) 
  (final_weight : ℝ) 
  (h1 : initial_weight = 120)
  (h2 : muscle_gain_percentage = 0.20)
  (h3 : final_weight = 150) :
  (final_weight - initial_weight - muscle_gain_percentage * initial_weight) / (muscle_gain_percentage * initial_weight) = 1/4 := by
  sorry

end fraction_of_muscle_gain_as_fat_l1794_179458


namespace smallest_integer_y_smallest_integer_y_is_six_l1794_179451

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < -8) ↔ (y ≥ 6) := by sorry

theorem smallest_integer_y_is_six : ∃ (y : ℤ), (7 - 3 * y < -8) ∧ (∀ (z : ℤ), (7 - 3 * z < -8) → z ≥ y) ∧ y = 6 := by sorry

end smallest_integer_y_smallest_integer_y_is_six_l1794_179451


namespace monic_quartic_specific_values_l1794_179443

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h_neg2 : f (-2) = 0)
  (h_1 : f 1 = -2)
  (h_3 : f 3 = -6)
  (h_5 : f 5 = -10) :
  f 0 = 29 := by
  sorry

end monic_quartic_specific_values_l1794_179443


namespace unique_solution_for_equation_l1794_179464

theorem unique_solution_for_equation (x y : ℝ) :
  (x - 6)^2 + (y - 7)^2 + (x - y)^2 = 1/3 ↔ x = 19/3 ∧ y = 20/3 := by
  sorry

end unique_solution_for_equation_l1794_179464


namespace membership_change_fall_increase_value_l1794_179421

/-- The percentage increase in membership during fall -/
def fall_increase : ℝ := sorry

/-- The percentage decrease in membership during spring -/
def spring_decrease : ℝ := 19

/-- The total percentage increase from original to spring membership -/
def total_increase : ℝ := 12.52

/-- Theorem stating the relationship between fall increase, spring decrease, and total increase -/
theorem membership_change :
  (1 + fall_increase / 100) * (1 - spring_decrease / 100) = 1 + total_increase / 100 :=
sorry

/-- The fall increase is approximately 38.91% -/
theorem fall_increase_value : 
  ∃ ε > 0, |fall_increase - 38.91| < ε :=
sorry

end membership_change_fall_increase_value_l1794_179421


namespace gym_occupancy_l1794_179425

theorem gym_occupancy (initial_people : ℕ) (people_came_in : ℕ) (people_left : ℕ) 
  (h1 : initial_people = 16) 
  (h2 : people_came_in = 5) 
  (h3 : people_left = 2) : 
  initial_people + people_came_in - people_left = 19 :=
by sorry

end gym_occupancy_l1794_179425


namespace arthur_muffins_l1794_179455

theorem arthur_muffins (james_muffins : ℕ) (arthur_muffins : ℕ) 
  (h1 : james_muffins = 12 * arthur_muffins) 
  (h2 : james_muffins = 1380) : 
  arthur_muffins = 115 := by
sorry

end arthur_muffins_l1794_179455


namespace final_S_value_l1794_179481

def sequence_A : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_A n + 1

def sequence_S : ℕ → ℕ
  | 0 => 0
  | n + 1 => sequence_S n + sequence_A (n + 1)

theorem final_S_value :
  ∃ n : ℕ, sequence_S n ≤ 36 ∧ sequence_S (n + 1) > 36 ∧ sequence_S (n + 1) = 45 :=
by
  sorry

end final_S_value_l1794_179481


namespace problem_solution_l1794_179438

theorem problem_solution (x y : ℝ) :
  (Real.sqrt (x - 3 * y) + |x^2 - 9|) / ((x + 3)^2) = 0 →
  Real.sqrt (x + 2) / Real.sqrt (y + 1) = Real.sqrt 5 / 2 := by
  sorry

end problem_solution_l1794_179438


namespace intersection_subsets_count_l1794_179410

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}

theorem intersection_subsets_count :
  Finset.card (Finset.powerset (M ∩ N)) = 4 := by
  sorry

end intersection_subsets_count_l1794_179410


namespace multiplication_subtraction_equality_l1794_179473

theorem multiplication_subtraction_equality : 72 * 989 - 12 * 989 = 59340 := by sorry

end multiplication_subtraction_equality_l1794_179473


namespace expense_increase_percentage_l1794_179470

def monthly_salary : ℝ := 5750
def initial_savings_rate : ℝ := 0.20
def new_savings : ℝ := 230

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - new_savings
  (expense_increase / initial_expenses) * 100 = 20 := by
  sorry

end expense_increase_percentage_l1794_179470


namespace reading_order_l1794_179412

variable (a b c d : ℝ)

theorem reading_order (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := by
  sorry

end reading_order_l1794_179412


namespace max_pieces_20x24_cake_l1794_179475

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents a piece of cake -/
structure CakePiece where
  size : Dimensions

/-- Represents the whole cake -/
structure Cake where
  size : Dimensions

/-- Calculates the maximum number of pieces that can be cut from a cake -/
def maxPieces (cake : Cake) (piece : CakePiece) : ℕ :=
  let horizontal := (cake.size.length / piece.size.length) * (cake.size.width / piece.size.width)
  let vertical := (cake.size.length / piece.size.width) * (cake.size.width / piece.size.length)
  max horizontal vertical

theorem max_pieces_20x24_cake (cake : Cake) (piece : CakePiece) :
  cake.size = Dimensions.mk 20 24 →
  piece.size = Dimensions.mk 4 4 →
  maxPieces cake piece = 30 := by
  sorry

#eval maxPieces (Cake.mk (Dimensions.mk 20 24)) (CakePiece.mk (Dimensions.mk 4 4))

end max_pieces_20x24_cake_l1794_179475


namespace final_payment_calculation_l1794_179489

/-- Calculates the final amount John will pay for three articles with given costs and discounts, including sales tax. -/
theorem final_payment_calculation (cost_A cost_B cost_C : ℝ)
  (discount_A discount_B discount_C : ℝ) (sales_tax_rate : ℝ)
  (h_cost_A : cost_A = 200)
  (h_cost_B : cost_B = 300)
  (h_cost_C : cost_C = 400)
  (h_discount_A : discount_A = 0.5)
  (h_discount_B : discount_B = 0.3)
  (h_discount_C : discount_C = 0.4)
  (h_sales_tax : sales_tax_rate = 0.05) :
  let discounted_A := cost_A * (1 - discount_A)
  let discounted_B := cost_B * (1 - discount_B)
  let discounted_C := cost_C * (1 - discount_C)
  let total_discounted := discounted_A + discounted_B + discounted_C
  let final_amount := total_discounted * (1 + sales_tax_rate)
  final_amount = 577.5 := by sorry


end final_payment_calculation_l1794_179489


namespace parallel_vectors_k_value_l1794_179432

/-- Given vectors a and b, if (k*a + b) is parallel to (a - 3*b), then k = -1/3 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2))
    (hb : b = (-3, 2))
    (h_parallel : ∃ (t : ℝ), t • (k • a + b) = (a - 3 • b)) :
  k = -1/3 := by
  sorry

end parallel_vectors_k_value_l1794_179432


namespace quadratic_root_range_l1794_179487

theorem quadratic_root_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a = 0 → x > 1) → 
  (3 < a ∧ a ≤ 4) :=
sorry

end quadratic_root_range_l1794_179487


namespace shooting_probability_l1794_179400

/-- The probability of scoring less than 9 in a shooting practice -/
def prob_less_than_9 (prob_10 prob_9 prob_8 : ℝ) : Prop :=
  prob_10 = 0.24 ∧ prob_9 = 0.28 ∧ prob_8 = 0.19 →
  1 - (prob_10 + prob_9) = 0.29

theorem shooting_probability : 
  ∃ (prob_10 prob_9 prob_8 : ℝ), prob_less_than_9 prob_10 prob_9 prob_8 := by
  sorry

end shooting_probability_l1794_179400


namespace pencil_cost_l1794_179485

theorem pencil_cost (pen_price pencil_price : ℚ) : 
  3 * pen_price + 2 * pencil_price = 165/100 →
  4 * pen_price + 7 * pencil_price = 303/100 →
  pencil_price = 19155/100000 :=
by sorry

end pencil_cost_l1794_179485


namespace mean_score_problem_l1794_179435

theorem mean_score_problem (m_mean a_mean : ℝ) (m a : ℕ) 
  (h1 : m_mean = 75)
  (h2 : a_mean = 65)
  (h3 : m = 2 * a / 3) :
  (m_mean * m + a_mean * a) / (m + a) = 69 := by
sorry

end mean_score_problem_l1794_179435


namespace sum_3x_4y_equals_60_l1794_179480

theorem sum_3x_4y_equals_60 
  (x y N : ℝ) 
  (h1 : 3 * x + 4 * y = N) 
  (h2 : 6 * x - 4 * y = 12) 
  (h3 : x * y = 72) : 
  3 * x + 4 * y = 60 := by
sorry

end sum_3x_4y_equals_60_l1794_179480


namespace increasing_function_m_range_l1794_179430

def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_function_m_range :
  ∀ m : ℝ, (∀ x > 2, (∀ h > 0, f m (x + h) > f m x)) ↔ m < 5/2 :=
sorry

end increasing_function_m_range_l1794_179430


namespace inequalities_given_sum_positive_l1794_179437

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequalities_given_sum_positive_l1794_179437


namespace power_difference_l1794_179403

theorem power_difference (m n : ℕ) (h1 : 3^m = 8) (h2 : 3^n = 2) : 3^(m-n) = 4 := by
  sorry

end power_difference_l1794_179403


namespace thalassa_population_estimate_l1794_179490

-- Define the initial population in 2020
def initial_population : ℕ := 500

-- Define the doubling period in years
def doubling_period : ℕ := 30

-- Define the target year
def target_year : ℕ := 2075

-- Define the base year
def base_year : ℕ := 2020

-- Function to calculate the number of complete doubling periods
def complete_doubling_periods (start_year end_year doubling_period : ℕ) : ℕ :=
  (end_year - start_year) / doubling_period

-- Function to estimate population after a number of complete doubling periods
def population_after_doubling (initial_pop doubling_periods : ℕ) : ℕ :=
  initial_pop * (2 ^ doubling_periods)

-- Theorem statement
theorem thalassa_population_estimate :
  let complete_periods := complete_doubling_periods base_year target_year doubling_period
  let pop_at_last_complete_period := population_after_doubling initial_population complete_periods
  let pop_at_next_complete_period := pop_at_last_complete_period * 2
  (pop_at_last_complete_period + pop_at_next_complete_period) / 2 = 1500 := by
  sorry

end thalassa_population_estimate_l1794_179490


namespace least_b_is_five_l1794_179496

/-- A triangle with angles a, b, c in degrees, where a, b, c are prime numbers and a > b > c -/
structure PrimeAngleTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_prime : Nat.Prime a
  b_prime : Nat.Prime b
  c_prime : Nat.Prime c
  angle_sum : a + b + c = 180
  a_gt_b : a > b
  b_gt_c : b > c
  not_right : a ≠ 90 ∧ b ≠ 90 ∧ c ≠ 90

/-- The least possible value of b in a PrimeAngleTriangle is 5 -/
theorem least_b_is_five (t : PrimeAngleTriangle) : t.b ≥ 5 := by
  sorry

end least_b_is_five_l1794_179496


namespace general_term_formula_first_term_l1794_179404

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

/-- The general term of the sequence -/
def a (n : ℕ+) : ℤ := 4 * n.val - 5

/-- Theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : a n = S n - S (n - 1) := by
  sorry

/-- Theorem stating that the formula holds for the first term -/
theorem first_term : a 1 = S 1 := by
  sorry

end general_term_formula_first_term_l1794_179404


namespace platform_length_l1794_179461

/-- Given a train of length 600 m that takes 78 seconds to cross a platform
    and 52 seconds to cross a signal pole, prove that the length of the platform is 300 m. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 600)
  (h2 : time_platform = 78)
  (h3 : time_pole = 52) :
  (train_length * time_platform / time_pole) - train_length = 300 :=
by sorry

end platform_length_l1794_179461


namespace infinite_powers_of_two_l1794_179491

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n -/
noncomputable def a (n : ℕ) : ℤ :=
  floor (n * Real.sqrt 2)

/-- Statement: There are infinitely many n such that a_n is a power of 2 -/
theorem infinite_powers_of_two : ∀ k : ℕ, ∃ n > k, ∃ m : ℕ, a n = 2^m :=
sorry

end infinite_powers_of_two_l1794_179491


namespace monomial_replacement_four_terms_l1794_179431

/-- Given an expression (x^4 - 3)^2 + (x^3 + *)^2, where * is to be replaced by a monomial,
    prove that replacing * with (x^3 + 3x) results in an expression with exactly four terms
    after squaring and combining like terms. -/
theorem monomial_replacement_four_terms (x : ℝ) : 
  let original_expr := (x^4 - 3)^2 + (x^3 + (x^3 + 3*x))^2
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    original_expr = a * x^n₁ + b * x^n₂ + c * x^n₃ + d * x^n₄ ∧
    n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > n₄ ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end monomial_replacement_four_terms_l1794_179431


namespace trig_identity_l1794_179441

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end trig_identity_l1794_179441


namespace max_value_sum_of_roots_l1794_179478

theorem max_value_sum_of_roots (x y z : ℝ) 
  (sum_eq_two : x + y + z = 2)
  (x_geq_neg_one : x ≥ -1)
  (y_geq_neg_two : y ≥ -2)
  (z_geq_neg_one : z ≥ -1) :
  ∃ (M : ℝ), M = 4 * Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a + b + c = 2 → a ≥ -1 → b ≥ -2 → c ≥ -1 →
  Real.sqrt (3 * a^2 + 3) + Real.sqrt (3 * b^2 + 6) + Real.sqrt (3 * c^2 + 3) ≤ M :=
by sorry

end max_value_sum_of_roots_l1794_179478


namespace triangle_side_sum_l1794_179452

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) :
  B = 2 * A →
  Real.cos A = 4 / 5 →
  (1 / 2) * a * b * Real.sin C = 468 / 25 →
  a + b = 13 := by
sorry

end triangle_side_sum_l1794_179452


namespace student_calculation_difference_l1794_179493

theorem student_calculation_difference : 
  let number : ℝ := 100.00000000000003
  let correct_answer := number * (4/5 : ℝ)
  let student_answer := number / (4/5 : ℝ)
  student_answer - correct_answer = 45.00000000000002 := by
sorry

end student_calculation_difference_l1794_179493


namespace sqrt_equation_solution_l1794_179433

theorem sqrt_equation_solution :
  let f (x : ℝ) := Real.sqrt (3 * x - 5) + 14 / Real.sqrt (3 * x - 5)
  ∀ x : ℝ, f x = 8 ↔ x = (23 + 8 * Real.sqrt 2) / 3 ∨ x = (23 - 8 * Real.sqrt 2) / 3 := by
  sorry

end sqrt_equation_solution_l1794_179433


namespace min_rounds_for_sole_winner_l1794_179495

/-- Represents a chess tournament -/
structure ChessTournament where
  num_players : ℕ
  num_rounds : ℕ
  points_per_win : ℚ
  points_per_draw : ℚ
  points_per_loss : ℚ

/-- Checks if a tournament configuration allows for a sole winner -/
def has_sole_winner (t : ChessTournament) : Prop :=
  ∃ (leader_score : ℚ) (max_other_score : ℚ),
    leader_score > max_other_score ∧
    leader_score ≤ t.num_rounds * t.points_per_win ∧
    max_other_score ≤ (t.num_rounds - 1) * t.points_per_win + t.points_per_draw

/-- The main theorem stating the minimum number of rounds for a sole winner -/
theorem min_rounds_for_sole_winner :
  ∀ (t : ChessTournament),
    t.num_players = 10 →
    t.points_per_win = 1 →
    t.points_per_draw = 1/2 →
    t.points_per_loss = 0 →
    (∀ n : ℕ, n < 7 → ¬(has_sole_winner {num_players := t.num_players,
                                         num_rounds := n,
                                         points_per_win := t.points_per_win,
                                         points_per_draw := t.points_per_draw,
                                         points_per_loss := t.points_per_loss})) ∧
    (has_sole_winner {num_players := t.num_players,
                      num_rounds := 7,
                      points_per_win := t.points_per_win,
                      points_per_draw := t.points_per_draw,
                      points_per_loss := t.points_per_loss}) :=
by
  sorry

end min_rounds_for_sole_winner_l1794_179495


namespace statement_A_statement_D_l1794_179477

-- Statement A
theorem statement_A (a b c : ℝ) : 
  a / (c^2 + 1) > b / (c^2 + 1) → a > b :=
by sorry

-- Statement D
theorem statement_D (a b : ℝ) :
  -1 < 2*a + b ∧ 2*a + b < 1 ∧ -1 < a - b ∧ a - b < 2 →
  -3 < 4*a - b ∧ 4*a - b < 5 :=
by sorry

end statement_A_statement_D_l1794_179477


namespace quadratic_equal_roots_l1794_179416

/-- Discriminant of a quadratic equation ax² + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Condition for a quadratic equation to have two equal real roots -/
def has_two_equal_real_roots (a b c : ℝ) : Prop := discriminant a b c = 0

theorem quadratic_equal_roots :
  has_two_equal_real_roots 1 (-2) 1 ∧
  ¬has_two_equal_real_roots 1 (-3) 2 ∧
  ¬has_two_equal_real_roots 1 (-2) 3 ∧
  ¬has_two_equal_real_roots 1 0 (-9) :=
sorry

end quadratic_equal_roots_l1794_179416


namespace stratified_sampling_sophomores_l1794_179447

theorem stratified_sampling_sophomores (total_students : ℕ) (sophomore_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : sophomore_students = 1500)
  (h3 : sample_size = 600) :
  (sophomore_students : ℚ) / total_students * sample_size = 200 := by
  sorry

end stratified_sampling_sophomores_l1794_179447


namespace complex_sum_of_parts_l1794_179448

theorem complex_sum_of_parts (z : ℂ) (h : z * Complex.I = -1 + Complex.I) : 
  z.re + z.im = 2 := by
  sorry

end complex_sum_of_parts_l1794_179448


namespace equation_solution_l1794_179498

theorem equation_solution (k : ℝ) : 
  (7 * (-1)^3 - 3 * (-1)^2 + k * (-1) + 5 = 0) → 
  (k^3 + 2 * k^2 - 11 * k - 85 = -105) := by
  sorry

end equation_solution_l1794_179498


namespace nine_crosses_fit_on_chessboard_l1794_179483

/-- Represents a cross pentomino -/
structure CrossPentomino :=
  (size : ℕ := 5)

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- The area of a cross pentomino -/
def cross_pentomino_area (c : CrossPentomino) : ℕ := c.size

/-- The area of a chessboard -/
def chessboard_area (b : Chessboard) : ℕ := b.rows * b.cols

/-- Theorem: Nine cross pentominoes can fit on an 8x8 chessboard -/
theorem nine_crosses_fit_on_chessboard :
  ∃ (c : CrossPentomino) (b : Chessboard),
    b.rows = 8 ∧ b.cols = 8 ∧
    9 * (cross_pentomino_area c) ≤ chessboard_area b :=
by sorry

end nine_crosses_fit_on_chessboard_l1794_179483


namespace stratified_sampling_11th_grade_l1794_179445

theorem stratified_sampling_11th_grade (total_students : ℕ) (eleventh_grade_students : ℕ) (sample_size : ℕ) :
  total_students = 5000 →
  eleventh_grade_students = 1500 →
  sample_size = 30 →
  (eleventh_grade_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 9 := by
  sorry

end stratified_sampling_11th_grade_l1794_179445


namespace football_team_yardage_l1794_179415

theorem football_team_yardage (initial_loss : ℤ) (gain : ℤ) (final_progress : ℤ) : 
  gain = 9 ∧ final_progress = 4 → initial_loss = 5 :=
by
  sorry

end football_team_yardage_l1794_179415


namespace angle_sum_90_l1794_179488

-- Define the necessary structures
structure Plane :=
(π : Type)

structure Line :=
(l : Type)

-- Define the perpendicular relation between a line and a plane
def perpendicular (p : Line) (π : Plane) : Prop :=
sorry

-- Define the angle between a line and a plane
def angle_line_plane (l : Line) (π : Plane) : ℝ :=
sorry

-- Define the angle between two lines
def angle_between_lines (l1 l2 : Line) : ℝ :=
sorry

-- State the theorem
theorem angle_sum_90 (p : Line) (π : Plane) (l : Line) 
  (h : perpendicular p π) :
  angle_line_plane l π + angle_between_lines l p = 90 :=
sorry

end angle_sum_90_l1794_179488


namespace correct_recommendation_count_l1794_179457

/-- Represents the number of recommendation spots for each language -/
structure SpotDistribution :=
  (korean : Nat)
  (japanese : Nat)
  (russian : Nat)

/-- Represents the gender distribution of candidates -/
structure CandidateDistribution :=
  (female : Nat)
  (male : Nat)

/-- Calculates the number of different recommendation methods -/
def recommendationMethods (spots : SpotDistribution) (candidates : CandidateDistribution) : Nat :=
  sorry

/-- Theorem stating the number of different recommendation methods -/
theorem correct_recommendation_count :
  let spots : SpotDistribution := ⟨2, 2, 1⟩
  let candidates : CandidateDistribution := ⟨3, 2⟩
  recommendationMethods spots candidates = 24 := by
  sorry

end correct_recommendation_count_l1794_179457


namespace complement_union_A_B_l1794_179482

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end complement_union_A_B_l1794_179482


namespace range_of_a_l1794_179408

theorem range_of_a (x a : ℝ) : 
  (∀ x, x < 0 → x < a) ∧ (∃ x, x ≥ 0 ∧ x < a) → 
  a > 0 ∧ ∀ ε > 0, ∃ b, b > a ∧ b < a + ε :=
by sorry

end range_of_a_l1794_179408


namespace function_properties_l1794_179456

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem function_properties :
  ∀ (a : ℝ),
  (∀ (x : ℝ), -5 ≤ x ∧ x ≤ 5 → 
    (a = -1 → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 37) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 37) ∧
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≥ 1) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 1)) ∧
    ((-5 < a ∧ a < 5) ↔ 
      (∃ (y z : ℝ), -5 ≤ y ∧ y < z ∧ z ≤ 5 ∧ f a y > f a z)) ∧
    ((-5 < a ∧ a < 0) → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 27 - 10*a) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 27 - 10*a)) ∧
    ((0 ≤ a ∧ a < 5) → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 27 + 10*a) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 27 + 10*a))) :=
by sorry

end function_properties_l1794_179456


namespace range_of_a_l1794_179427

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a = 0}

theorem range_of_a : ∀ a : ℝ, (A ∪ B a = A) ↔ (0 ≤ a ∧ a < 4) := by sorry

end range_of_a_l1794_179427


namespace personal_trainer_cost_l1794_179492

-- Define the given conditions
def old_hourly_wage : ℚ := 40
def raise_percentage : ℚ := 5 / 100
def hours_per_day : ℚ := 8
def days_per_week : ℚ := 5
def old_bills : ℚ := 600
def leftover : ℚ := 980

-- Define the theorem
theorem personal_trainer_cost :
  let new_hourly_wage := old_hourly_wage * (1 + raise_percentage)
  let weekly_earnings := new_hourly_wage * hours_per_day * days_per_week
  let total_expenses := weekly_earnings - leftover
  total_expenses - old_bills = 100 := by sorry

end personal_trainer_cost_l1794_179492


namespace arithmetic_sequence_properties_l1794_179462

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first 5 terms is 10
  sum_5 : (5 : ℚ) / 2 * (2 * a + 4 * d) = 10
  -- Sum of first 50 terms is 150
  sum_50 : (50 : ℚ) / 2 * (2 * a + 49 * d) = 150

/-- Properties of the 55th term and sum of first 55 terms -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  let sum_55 := (55 : ℚ) / 2 * (2 * seq.a + 54 * seq.d)
  let term_55 := seq.a + 54 * seq.d
  sum_55 = 171 ∧ term_55 = 4.31 := by
  sorry

end arithmetic_sequence_properties_l1794_179462


namespace proper_subsets_of_m_n_l1794_179463

def S : Set (Set Char) := {{}, {'m'}, {'n'}}

theorem proper_subsets_of_m_n :
  {A : Set Char | A ⊂ {'m', 'n'}} = S := by sorry

end proper_subsets_of_m_n_l1794_179463


namespace center_square_side_length_l1794_179422

theorem center_square_side_length : 
  let large_square_side : ℝ := 120
  let total_area : ℝ := large_square_side ^ 2
  let l_shape_area : ℝ := (1 / 5) * total_area
  let center_square_area : ℝ := total_area - 4 * l_shape_area
  let center_square_side : ℝ := Real.sqrt center_square_area
  center_square_side = 54 := by
  sorry

end center_square_side_length_l1794_179422


namespace point_coordinates_wrt_origin_l1794_179424

/-- The coordinates of a point in a 2D Cartesian coordinate system. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Theorem: The coordinates of the point (1, -2) with respect to the origin
    in a Cartesian coordinate system are (1, -2). -/
theorem point_coordinates_wrt_origin (p : Point2D) (h : p = ⟨1, -2⟩) :
  p.x = 1 ∧ p.y = -2 := by
  sorry

end point_coordinates_wrt_origin_l1794_179424


namespace longest_side_of_triangle_l1794_179469

theorem longest_side_of_triangle (x : ℝ) : 
  5 + (2*x + 3) + (3*x - 2) = 41 →
  max 5 (max (2*x + 3) (3*x - 2)) = 19 := by
sorry

end longest_side_of_triangle_l1794_179469


namespace num_paths_equals_1287_l1794_179486

/-- The number of blocks to the right -/
def blocks_right : ℕ := 8

/-- The number of blocks up -/
def blocks_up : ℕ := 5

/-- The total number of moves -/
def total_moves : ℕ := blocks_right + blocks_up

/-- The number of different shortest paths -/
def num_paths : ℕ := Nat.choose total_moves blocks_up

theorem num_paths_equals_1287 : num_paths = 1287 := by
  sorry

end num_paths_equals_1287_l1794_179486


namespace common_solution_y_value_l1794_179466

theorem common_solution_y_value : ∃ (x y : ℝ), 
  (x^2 + y^2 - 16 = 0) ∧ 
  (x^2 - 3*y + 12 = 0) → 
  y = 4 := by sorry

end common_solution_y_value_l1794_179466


namespace right_triangle_side_length_l1794_179411

theorem right_triangle_side_length 
  (X Y Z : ℝ) 
  (h_right_angle : X^2 + Y^2 = Z^2)  -- Y is the right angle
  (h_cos : Real.cos X = 3/5)
  (h_hypotenuse : Z = 10) :
  Y = 8 := by
sorry

end right_triangle_side_length_l1794_179411


namespace account_balance_difference_l1794_179407

/-- Computes the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Computes the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- The difference between two account balances after 25 years -/
theorem account_balance_difference : 
  let jessica_balance := compound_interest 12000 0.025 50
  let mark_balance := simple_interest 15000 0.06 25
  ∃ ε > 0, abs (jessica_balance - mark_balance - 3136) < ε :=
sorry

end account_balance_difference_l1794_179407


namespace sandwich_slices_count_l1794_179436

/-- Given the total number of sandwiches and the total number of bread slices,
    calculate the number of slices per sandwich. -/
def slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) : ℚ :=
  total_slices / total_sandwiches

/-- Theorem stating that for 5 sandwiches and 15 slices, each sandwich consists of 3 slices. -/
theorem sandwich_slices_count :
  slices_per_sandwich 5 15 = 3 := by
  sorry

end sandwich_slices_count_l1794_179436


namespace line_perp_from_plane_perp_l1794_179465

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_from_plane_perp
  (m n : Line) (α β : Plane)
  (h1 : perp_line_plane m α)
  (h2 : perp_line_plane n β)
  (h3 : perp_plane α β) :
  perp_line m n :=
sorry

end line_perp_from_plane_perp_l1794_179465


namespace sum_of_possible_n_values_l1794_179406

/-- Given natural numbers 15, 12, and n, where the product of any two is divisible by the third,
    the sum of all possible values of n is 260. -/
theorem sum_of_possible_n_values : ∃ (S : Finset ℕ),
  (∀ n ∈ S, n > 0 ∧ 
    (15 * 12) % n = 0 ∧ 
    (15 * n) % 12 = 0 ∧ 
    (12 * n) % 15 = 0) ∧
  (∀ n > 0, 
    (15 * 12) % n = 0 ∧ 
    (15 * n) % 12 = 0 ∧ 
    (12 * n) % 15 = 0 → n ∈ S) ∧
  S.sum id = 260 := by
  sorry


end sum_of_possible_n_values_l1794_179406


namespace systematic_sampling_first_sample_first_sample_is_18_l1794_179401

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sampleSize : ℕ
  interval : ℕ
  firstSample : ℕ
  eighteenthSample : ℕ

/-- Theorem stating the relationship between the first and eighteenth samples in systematic sampling -/
theorem systematic_sampling_first_sample
  (s : SystematicSampling)
  (h1 : s.population = 1000)
  (h2 : s.sampleSize = 40)
  (h3 : s.interval = s.population / s.sampleSize)
  (h4 : s.eighteenthSample = 443)
  (h5 : s.eighteenthSample = s.firstSample + 17 * s.interval) :
  s.firstSample = 18 := by
  sorry

/-- Main theorem proving the first sample number in the given scenario -/
theorem first_sample_is_18
  (population : ℕ)
  (sampleSize : ℕ)
  (eighteenthSample : ℕ)
  (h1 : population = 1000)
  (h2 : sampleSize = 40)
  (h3 : eighteenthSample = 443) :
  ∃ (s : SystematicSampling),
    s.population = population ∧
    s.sampleSize = sampleSize ∧
    s.interval = population / sampleSize ∧
    s.eighteenthSample = eighteenthSample ∧
    s.firstSample = 18 := by
  sorry

end systematic_sampling_first_sample_first_sample_is_18_l1794_179401


namespace two_solutions_l1794_179420

/-- The number of ordered pairs of integers (x, y) satisfying x^4 + y^2 = 4y -/
def count_solutions : ℕ := 2

/-- Predicate that checks if a pair of integers satisfies the equation -/
def satisfies_equation (x y : ℤ) : Prop :=
  x^4 + y^2 = 4*y

theorem two_solutions :
  (∃! (s : Finset (ℤ × ℤ)), s.card = count_solutions ∧ 
    ∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_equation p.1 p.2) :=
sorry

end two_solutions_l1794_179420


namespace measuring_cups_l1794_179402

theorem measuring_cups (a : Int) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : Int),
    (b ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (c ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (d ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (e ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (f ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (a = 625*b + 125*c + 25*d + 5*e + f) :=
by sorry

end measuring_cups_l1794_179402


namespace negative_plus_abs_neg_l1794_179468

theorem negative_plus_abs_neg (a : ℝ) (h : a < 0) : a + |-a| = 0 := by
  sorry

end negative_plus_abs_neg_l1794_179468


namespace total_ants_l1794_179476

theorem total_ants (red_ants : ℕ) (black_ants : ℕ) 
  (h1 : red_ants = 413) (h2 : black_ants = 487) : 
  red_ants + black_ants = 900 := by
  sorry

end total_ants_l1794_179476


namespace square_area_from_vertices_l1794_179440

/-- The area of a square with adjacent vertices at (0,3) and (4,0) is 25. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (4, 0)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end square_area_from_vertices_l1794_179440


namespace total_food_consumed_theorem_l1794_179434

/-- Represents the amount of food a dog eats per meal -/
structure MealPortion where
  dry : Float
  wet : Float

/-- Represents the feeding schedule for a dog -/
structure FeedingSchedule where
  portion : MealPortion
  mealsPerDay : Nat

/-- Conversion rates for dry and wet food -/
def dryFoodConversion : Float := 3.2  -- cups per pound
def wetFoodConversion : Float := 2.8  -- cups per pound

/-- Feeding schedules for each dog -/
def momoSchedule : FeedingSchedule := { portion := { dry := 1.3, wet := 0.7 }, mealsPerDay := 2 }
def fifiSchedule : FeedingSchedule := { portion := { dry := 1.6, wet := 0.5 }, mealsPerDay := 2 }
def gigiSchedule : FeedingSchedule := { portion := { dry := 2.0, wet := 1.0 }, mealsPerDay := 3 }

/-- Calculate total food consumed by all dogs in pounds -/
def totalFoodConsumed (momo fifi gigi : FeedingSchedule) : Float :=
  let totalDry := (momo.portion.dry * momo.mealsPerDay.toFloat +
                   fifi.portion.dry * fifi.mealsPerDay.toFloat +
                   gigi.portion.dry * gigi.mealsPerDay.toFloat) / dryFoodConversion
  let totalWet := (momo.portion.wet * momo.mealsPerDay.toFloat +
                   fifi.portion.wet * fifi.mealsPerDay.toFloat +
                   gigi.portion.wet * gigi.mealsPerDay.toFloat) / wetFoodConversion
  totalDry + totalWet

/-- Theorem: The total amount of food consumed by all three dogs in a day is approximately 5.6161 pounds -/
theorem total_food_consumed_theorem :
  Float.abs (totalFoodConsumed momoSchedule fifiSchedule gigiSchedule - 5.6161) < 0.0001 := by
  sorry

end total_food_consumed_theorem_l1794_179434


namespace max_students_equal_distribution_l1794_179467

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1001) (h_pencils : pencils = 910) : 
  (∃ (students : ℕ), 
    students > 0 ∧ 
    pens % students = 0 ∧ 
    pencils % students = 0 ∧ 
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔ 
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
sorry

end max_students_equal_distribution_l1794_179467


namespace three_dozen_quarters_value_l1794_179414

/-- Proves that 3 dozen quarters is equal to $9 --/
theorem three_dozen_quarters_value : 
  let dozen : ℕ := 12
  let quarter_value : ℕ := 25  -- in cents
  let cents_per_dollar : ℕ := 100
  (3 * dozen * quarter_value) / cents_per_dollar = 9 := by
  sorry

end three_dozen_quarters_value_l1794_179414


namespace james_monthly_income_l1794_179413

/-- Represents a subscription tier with its subscriber count and price --/
structure SubscriptionTier where
  subscribers : ℕ
  price : ℚ

/-- Calculates the total monthly income for James from Twitch subscriptions --/
def calculate_monthly_income (tier1 tier2 tier3 : SubscriptionTier) : ℚ :=
  tier1.subscribers * tier1.price +
  tier2.subscribers * tier2.price +
  tier3.subscribers * tier3.price

/-- Theorem stating that James' monthly income from Twitch subscriptions is $2522.50 --/
theorem james_monthly_income :
  let tier1 := SubscriptionTier.mk (120 + 10) (499 / 100)
  let tier2 := SubscriptionTier.mk (50 + 25) (999 / 100)
  let tier3 := SubscriptionTier.mk (30 + 15) (2499 / 100)
  calculate_monthly_income tier1 tier2 tier3 = 252250 / 100 := by
  sorry


end james_monthly_income_l1794_179413


namespace kiarra_age_l1794_179423

/-- Given the ages of several people and their relationships, prove Kiarra's age --/
theorem kiarra_age (bea job figaro harry kiarra : ℕ) 
  (h1 : kiarra = 2 * bea)
  (h2 : job = 3 * bea)
  (h3 : figaro = job + 7)
  (h4 : harry * 2 = figaro)
  (h5 : harry = 26) :
  kiarra = 30 := by
  sorry

end kiarra_age_l1794_179423


namespace water_jars_problem_l1794_179446

theorem water_jars_problem (total_volume : ℚ) (x : ℕ) : 
  total_volume = 42 →
  (x : ℚ) * (1/4 + 1/2 + 1) = total_volume →
  3 * x = 72 :=
by sorry

end water_jars_problem_l1794_179446


namespace g_13_l1794_179460

def g (n : ℕ) : ℕ := n^2 + 2*n + 41

theorem g_13 : g 13 = 236 := by
  sorry

end g_13_l1794_179460


namespace original_recipe_butter_l1794_179419

/-- Represents a bread recipe with butter and flour quantities -/
structure BreadRecipe where
  butter : ℝ  -- Amount of butter in ounces
  flour : ℝ   -- Amount of flour in cups

/-- The original bread recipe -/
def original_recipe : BreadRecipe := { butter := 0, flour := 5 }

/-- The scaled up recipe -/
def scaled_recipe : BreadRecipe := { butter := 12, flour := 20 }

/-- The scale factor between the original and scaled recipe -/
def scale_factor : ℝ := 4

theorem original_recipe_butter :
  original_recipe.butter = 3 :=
by
  sorry


end original_recipe_butter_l1794_179419


namespace only_prop3_true_l1794_179472

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define the limit of a sequence
def LimitOf (a : Sequence) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - L| < ε

-- Define the four propositions
def Prop1 (a : Sequence) (A : ℝ) : Prop :=
  LimitOf (fun n => (a n)^2) (A^2) → LimitOf a A

def Prop2 (a : Sequence) (A : ℝ) : Prop :=
  (∀ n, a n > 0) → LimitOf a A → A > 0

def Prop3 (a : Sequence) (A : ℝ) : Prop :=
  LimitOf a A → LimitOf (fun n => (a n)^2) (A^2)

def Prop4 (a b : Sequence) : Prop :=
  LimitOf (fun n => a n - b n) 0 → 
  (∃ L, LimitOf a L ∧ LimitOf b L)

-- Theorem stating that only Prop3 is always true
theorem only_prop3_true : 
  (∃ a A, ¬ Prop1 a A) ∧
  (∃ a A, ¬ Prop2 a A) ∧
  (∀ a A, Prop3 a A) ∧
  (∃ a b, ¬ Prop4 a b) := by
  sorry

end only_prop3_true_l1794_179472


namespace meaningful_fraction_l1794_179428

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 2)) ↔ x ≠ 2 := by sorry

end meaningful_fraction_l1794_179428


namespace washington_dc_july_4th_avg_temp_l1794_179497

def washington_dc_july_4th_temps : List ℝ := [90, 90, 90, 79, 71]

theorem washington_dc_july_4th_avg_temp :
  (washington_dc_july_4th_temps.sum / washington_dc_july_4th_temps.length : ℝ) = 84 := by
  sorry

end washington_dc_july_4th_avg_temp_l1794_179497


namespace indefinite_integral_ln_4x2_plus_1_l1794_179453

open Real

theorem indefinite_integral_ln_4x2_plus_1 (x : ℝ) :
  (deriv fun x => x * log (4 * x^2 + 1) - 8 * x + 4 * arctan (2 * x)) x = log (4 * x^2 + 1) := by
  sorry

end indefinite_integral_ln_4x2_plus_1_l1794_179453


namespace consecutive_odd_numbers_divisibility_l1794_179449

theorem consecutive_odd_numbers_divisibility (a b c : ℤ) : 
  (∃ k : ℤ, b = 2 * k + 1) →  -- b is odd
  (a = b - 2) →              -- a is the previous odd number
  (c = b + 2) →              -- c is the next odd number
  ∃ m : ℤ, a * b * c + 4 * b = m * b^3 :=
by sorry

end consecutive_odd_numbers_divisibility_l1794_179449


namespace exp_ln_five_l1794_179417

theorem exp_ln_five : Real.exp (Real.log 5) = 5 := by sorry

end exp_ln_five_l1794_179417


namespace more_polygons_without_A1_l1794_179429

-- Define the number of points on the circle
def n : ℕ := 16

-- Define the function to calculate the number of polygons including A1
def polygons_with_A1 (n : ℕ) : ℕ :=
  (2^(n-1) : ℕ) - (n : ℕ)

-- Define the function to calculate the number of polygons not including A1
def polygons_without_A1 (n : ℕ) : ℕ :=
  (2^(n-1) : ℕ) - (n : ℕ) - ((n-1).choose 2)

-- State the theorem
theorem more_polygons_without_A1 :
  polygons_without_A1 n > polygons_with_A1 n :=
by sorry

end more_polygons_without_A1_l1794_179429


namespace video_dislikes_calculation_l1794_179484

/-- Calculates the final number of dislikes for a video given initial likes, 
    initial dislikes formula, and additional dislikes. -/
def final_dislikes (initial_likes : ℕ) (additional_dislikes : ℕ) : ℕ :=
  (initial_likes / 2 + 100) + additional_dislikes

/-- Theorem stating that for a video with 3000 initial likes and 1000 additional dislikes,
    the final number of dislikes is 2600. -/
theorem video_dislikes_calculation :
  final_dislikes 3000 1000 = 2600 := by
  sorry

end video_dislikes_calculation_l1794_179484


namespace arithmetic_geometric_k4_l1794_179442

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d k : ℕ → ℕ) : Prop :=
  (∃ (c : ℝ), c ≠ 0 ∧ ∀ n, a (n + 1) = a n + c) ∧
  (∃ (q : ℝ), q ≠ 0 ∧ q ≠ 1 ∧ ∀ n, a (k (n + 1)) = a (k n) * q) ∧
  k 1 = 1 ∧ k 2 = 2 ∧ k 3 = 6

theorem arithmetic_geometric_k4 (a : ℕ → ℝ) (d k : ℕ → ℕ) :
  arithmetic_geometric_sequence a d k → k 4 = 22 := by
  sorry

end arithmetic_geometric_k4_l1794_179442


namespace QR_length_l1794_179494

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the point N on QR
def N_on_QR (Q R N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ N = (1 - t) • Q + t • R

-- Define the ratio condition for N on QR
def N_divides_QR_in_ratio (Q R N : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, dist Q N = 2 * x ∧ dist N R = 3 * x

-- Main theorem
theorem QR_length 
  (P Q R N : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (pr_length : dist P R = 5)
  (pq_length : dist P Q = 7)
  (n_on_qr : N_on_QR Q R N)
  (n_divides_qr : N_divides_QR_in_ratio Q R N)
  (pn_length : dist P N = 4) :
  dist Q R = 5 * Real.sqrt 3.9 := by
  sorry


end QR_length_l1794_179494


namespace f_domain_l1794_179459

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^2))

theorem f_domain : Set.Icc (-1 : ℝ) 1 = {x : ℝ | ∃ y, f x = y} :=
sorry

end f_domain_l1794_179459


namespace quiz_scores_l1794_179439

theorem quiz_scores (nicole kim cherry : ℕ) 
  (h1 : nicole = kim - 3)
  (h2 : kim = cherry + 8)
  (h3 : nicole = 22) : 
  cherry = 17 := by sorry

end quiz_scores_l1794_179439


namespace four_digit_number_theorem_l1794_179405

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

def satisfies_conditions (n : ℕ) : Prop :=
  is_four_digit n ∧
  n % 3 = 0 ∧
  first_two_digits n - last_two_digits n = 11

def solution_set : Set ℕ := {1302, 1605, 1908, 2211, 2514, 2817, 3120, 3423, 3726, 4029, 4332, 4635, 4938, 5241, 5544, 5847, 6150, 6453, 6756, 7059, 7362, 7665, 7968, 8271, 8574, 8877, 9180, 9483, 9786, 10089, 10392, 10695, 10998}

theorem four_digit_number_theorem :
  {n : ℕ | satisfies_conditions n} = solution_set := by sorry

end four_digit_number_theorem_l1794_179405


namespace get_ready_time_l1794_179454

/-- The time it takes for Jack and his two toddlers to get ready -/
def total_time (jack_socks jack_shoes jack_jacket toddler_socks toddler_shoes toddler_shoelaces : ℕ) : ℕ :=
  let jack_time := jack_socks + jack_shoes + jack_jacket
  let toddler_time := toddler_socks + toddler_shoes + 2 * toddler_shoelaces
  jack_time + 2 * toddler_time

theorem get_ready_time :
  total_time 2 4 3 2 5 1 = 27 :=
by sorry

end get_ready_time_l1794_179454


namespace negation_of_universal_proposition_l1794_179499

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, 3^n ≥ n^2 + 1) ↔ (∃ n₀ : ℕ, 3^n₀ < n₀^2 + 1) := by
  sorry

end negation_of_universal_proposition_l1794_179499
