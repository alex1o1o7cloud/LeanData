import Mathlib

namespace NUMINAMATH_CALUDE_zeros_in_Q_l1275_127502

/-- R_k represents an integer whose base-ten representation consists of k consecutive ones -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- Q is the quotient of R_30 and R_6 -/
def Q : ℕ := R 30 / R 6

/-- count_zeros counts the number of zeros in the base-ten representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 25 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l1275_127502


namespace NUMINAMATH_CALUDE_lemon_juice_for_dozen_cupcakes_l1275_127566

/-- The number of tablespoons of lemon juice provided by one lemon -/
def tablespoons_per_lemon : ℕ := 4

/-- The number of lemons needed for 3 dozen cupcakes -/
def lemons_for_three_dozen : ℕ := 9

/-- The number of tablespoons of lemon juice needed for a dozen cupcakes -/
def tablespoons_for_dozen : ℕ := 12

/-- Proves that the number of tablespoons of lemon juice needed for a dozen cupcakes is 12 -/
theorem lemon_juice_for_dozen_cupcakes : 
  tablespoons_for_dozen = (lemons_for_three_dozen * tablespoons_per_lemon) / 3 :=
by sorry

end NUMINAMATH_CALUDE_lemon_juice_for_dozen_cupcakes_l1275_127566


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1275_127513

theorem quadratic_form_equivalence :
  ∀ (x : ℝ), 3 * x^2 + 9 * x + 20 = 3 * (x + 3/2)^2 + 53/4 ∧
  ∃ (h : ℝ), h = -3/2 ∧ ∀ (x : ℝ), 3 * x^2 + 9 * x + 20 = 3 * (x - h)^2 + 53/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1275_127513


namespace NUMINAMATH_CALUDE_average_increase_is_four_l1275_127597

/-- Represents a cricket player's performance --/
structure CricketPerformance where
  innings : ℕ
  totalRuns : ℕ
  newInningsRuns : ℕ

/-- Calculates the average runs per innings --/
def average (cp : CricketPerformance) : ℚ :=
  cp.totalRuns / cp.innings

/-- Calculates the new average after playing an additional innings --/
def newAverage (cp : CricketPerformance) : ℚ :=
  (cp.totalRuns + cp.newInningsRuns) / (cp.innings + 1)

/-- Theorem: The increase in average is 4 runs --/
theorem average_increase_is_four (cp : CricketPerformance) 
  (h1 : cp.innings = 10)
  (h2 : average cp = 18)
  (h3 : cp.newInningsRuns = 62) : 
  newAverage cp - average cp = 4 := by
  sorry


end NUMINAMATH_CALUDE_average_increase_is_four_l1275_127597


namespace NUMINAMATH_CALUDE_initial_men_count_l1275_127582

theorem initial_men_count (provisions : ℕ) : ∃ (initial_men : ℕ),
  (provisions / (initial_men * 20) = provisions / ((initial_men + 200) * 15)) ∧
  initial_men = 600 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1275_127582


namespace NUMINAMATH_CALUDE_stock_value_after_fluctuations_l1275_127570

theorem stock_value_after_fluctuations (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let limit_up := 1.1
  let limit_down := 0.9
  let final_value := initial_value * (limit_up ^ 5) * (limit_down ^ 5)
  final_value < initial_value :=
by sorry

end NUMINAMATH_CALUDE_stock_value_after_fluctuations_l1275_127570


namespace NUMINAMATH_CALUDE_cos_215_minus_1_l1275_127512

theorem cos_215_minus_1 : Real.cos (215 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_215_minus_1_l1275_127512


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l1275_127534

/-- Given a function f(x) = (ln x) / x - a with two zeros m and n, prove that mn > e² -/
theorem zeros_product_greater_than_e_squared (a : ℝ) (m n : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (Real.log x) / x - a = 0 ∧ (Real.log y) / y - a = 0) →
  (Real.log m) / m - a = 0 →
  (Real.log n) / n - a = 0 →
  m * n > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l1275_127534


namespace NUMINAMATH_CALUDE_point_outside_circle_l1275_127517

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 24}
  P ∉ circle := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1275_127517


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l1275_127574

theorem movie_ticket_cost (M : ℝ) : 
  (2 * M + 5 * M = 35) → M = 5 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l1275_127574


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1275_127599

theorem basketball_lineup_count :
  let total_players : ℕ := 12
  let point_guards : ℕ := 1
  let other_players : ℕ := 5
  Nat.choose total_players point_guards * Nat.choose (total_players - point_guards) other_players = 5544 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1275_127599


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1275_127536

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₃ = 6 and a₆ = 3, prove a₉ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 6)
  (h_6 : a 6 = 3) :
  a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1275_127536


namespace NUMINAMATH_CALUDE_no_infinite_arithmetic_progression_in_squares_l1275_127529

theorem no_infinite_arithmetic_progression_in_squares :
  ¬ ∃ (a d : ℕ) (f : ℕ → ℕ),
    (∀ n, f n < f (n + 1)) ∧
    (∀ n, ∃ k, f n = k^2) ∧
    (∀ n, f (n + 1) - f n = d) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_arithmetic_progression_in_squares_l1275_127529


namespace NUMINAMATH_CALUDE_odd_sequence_concat_theorem_l1275_127543

def odd_sequence (n : ℕ) : List ℕ :=
  List.filter (λ x => x % 2 = 1) (List.range (n + 1))

def concat_digits (lst : List ℕ) : ℕ := sorry

def digit_sum (n : ℕ) : ℕ := sorry

theorem odd_sequence_concat_theorem :
  let seq := odd_sequence 103
  let A := concat_digits seq
  (Nat.digits 10 A).length = 101 ∧ A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_sequence_concat_theorem_l1275_127543


namespace NUMINAMATH_CALUDE_puzzle_solution_l1275_127515

def addition_puzzle (F I V N E : Nat) : Prop :=
  F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E ∧
  I ≠ V ∧ I ≠ N ∧ I ≠ E ∧
  V ≠ N ∧ V ≠ E ∧
  N ≠ E ∧
  F = 8 ∧
  I % 2 = 0 ∧
  1000 * N + 100 * I + 10 * N + E = 100 * F + 10 * I + V + 100 * F + 10 * I + V

theorem puzzle_solution :
  ∀ F I V N E, addition_puzzle F I V N E → V = 5 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1275_127515


namespace NUMINAMATH_CALUDE_apples_distribution_l1275_127564

/-- Given 48 apples distributed evenly among 7 children, prove that 1 child receives fewer than 7 apples -/
theorem apples_distribution (total_apples : Nat) (num_children : Nat) 
  (h1 : total_apples = 48) 
  (h2 : num_children = 7) : 
  (num_children - (total_apples % num_children)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_apples_distribution_l1275_127564


namespace NUMINAMATH_CALUDE_union_subset_iff_m_range_no_m_for_equality_l1275_127555

-- Define the sets P and S
def P : Set ℝ := {x : ℝ | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ m}

-- Theorem 1: (P ∪ S) ⊆ P if and only if m ∈ (-∞, 3]
theorem union_subset_iff_m_range (m : ℝ) : 
  (P ∪ S m) ⊆ P ↔ m ≤ 3 :=
sorry

-- Theorem 2: There does not exist an m such that P = S
theorem no_m_for_equality : 
  ¬∃ m : ℝ, P = S m :=
sorry

end NUMINAMATH_CALUDE_union_subset_iff_m_range_no_m_for_equality_l1275_127555


namespace NUMINAMATH_CALUDE_employee_pay_l1275_127556

theorem employee_pay (total_pay m_pay n_pay : ℝ) : 
  total_pay = 550 →
  m_pay = 1.2 * n_pay →
  m_pay + n_pay = total_pay →
  n_pay = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l1275_127556


namespace NUMINAMATH_CALUDE_smallest_positive_constant_inequality_l1275_127540

theorem smallest_positive_constant_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (∃ c : ℝ, c > 0 ∧ ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 →
    Real.sqrt (x * y * z) + c * Real.sqrt (|x - y|) ≥ (x + y + z) / 3) ∧
  (∀ c : ℝ, c > 0 ∧ (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 →
    Real.sqrt (x * y * z) + c * Real.sqrt (|x - y|) ≥ (x + y + z) / 3) → c ≥ 1) ∧
  (Real.sqrt (x * y * z) + Real.sqrt (|x - y|) ≥ (x + y + z) / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_constant_inequality_l1275_127540


namespace NUMINAMATH_CALUDE_hyperbola_intersection_length_l1275_127535

/-- Given a hyperbola with imaginary axis length 4 and eccentricity √6/2,
    if a line through the left focus intersects the left branch at points A and B
    such that |AB| is the arithmetic mean of |AF₂| and |BF₂|, then |AB| = 8√2 -/
theorem hyperbola_intersection_length
  (b : ℝ) (e : ℝ) (A B F₁ F₂ : ℝ × ℝ)
  (h_b : b = 2)
  (h_e : e = Real.sqrt 6 / 2)
  (h_foci : F₁.1 < F₂.1)
  (h_left_branch : A.1 < F₁.1 ∧ B.1 < F₁.1)
  (h_line : ∃ (m k : ℝ), A.2 = m * A.1 + k ∧ B.2 = m * B.1 + k ∧ F₁.2 = m * F₁.1 + k)
  (h_arithmetic_mean : 2 * dist A B = dist A F₂ + dist B F₂)
  (h_hyperbola : dist A F₂ - dist A F₁ = dist B F₂ - dist B F₁) :
  dist A B = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_length_l1275_127535


namespace NUMINAMATH_CALUDE_equation_solution_l1275_127523

theorem equation_solution : 
  ∀ x : ℝ, (x + 4)^2 = 5*(x + 4) ↔ x = -4 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1275_127523


namespace NUMINAMATH_CALUDE_one_negative_number_l1275_127518

theorem one_negative_number (numbers : List ℝ := [-2, 1/2, 0, 3]) : 
  (numbers.filter (λ x => x < 0)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_negative_number_l1275_127518


namespace NUMINAMATH_CALUDE_percentage_calculation_l1275_127558

theorem percentage_calculation (N P : ℝ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1275_127558


namespace NUMINAMATH_CALUDE_oranges_from_third_tree_l1275_127521

/-- The number of oranges picked from the third tree -/
def oranges_third_tree (total : ℕ) (first : ℕ) (second : ℕ) : ℕ :=
  total - (first + second)

/-- Theorem stating that the number of oranges picked from the third tree is 120 -/
theorem oranges_from_third_tree :
  oranges_third_tree 260 80 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_oranges_from_third_tree_l1275_127521


namespace NUMINAMATH_CALUDE_f_value_at_3_l1275_127537

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = 4) : f a b 3 = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1275_127537


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1275_127553

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 336) + (Real.sqrt 192 / Real.sqrt 144) = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1275_127553


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1275_127525

/-- Proves that the cost of each adult ticket is $31.50 given the problem conditions --/
theorem adult_ticket_cost : 
  let child_ticket_cost : ℚ := 15/2
  let total_bill : ℚ := 138
  let total_tickets : ℕ := 12
  ∀ (adult_tickets : ℕ) (child_tickets : ℕ) (adult_ticket_cost : ℚ),
    child_tickets = adult_tickets + 8 →
    adult_tickets + child_tickets = total_tickets →
    adult_tickets * adult_ticket_cost + child_tickets * child_ticket_cost = total_bill →
    adult_ticket_cost = 63/2 :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1275_127525


namespace NUMINAMATH_CALUDE_expression_value_l1275_127567

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 2 * y + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1275_127567


namespace NUMINAMATH_CALUDE_unique_prime_factors_count_l1275_127550

def product : ℕ := 102 * 103 * 105 * 107

theorem unique_prime_factors_count :
  (Nat.factors product).toFinset.card = 7 := by sorry

end NUMINAMATH_CALUDE_unique_prime_factors_count_l1275_127550


namespace NUMINAMATH_CALUDE_birthday_cake_red_candles_l1275_127532

/-- The number of red candles on a birthday cake -/
def red_candles (total_candles yellow_candles blue_candles : ℕ) : ℕ :=
  total_candles - (yellow_candles + blue_candles)

/-- Theorem stating the number of red candles used for the birthday cake -/
theorem birthday_cake_red_candles :
  red_candles 79 27 38 = 14 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cake_red_candles_l1275_127532


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l1275_127549

/-- Given a waiter's salary and tips, where the tips are 7/4 of the salary,
    prove that the fraction of total income from tips is 7/11. -/
theorem waiter_income_fraction (salary : ℚ) (tips : ℚ) (h : tips = (7 / 4) * salary) :
  tips / (salary + tips) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l1275_127549


namespace NUMINAMATH_CALUDE_function_analysis_l1275_127596

/-- Given a real number a and a function f(x) = x²(x-a), this theorem proves:
    (I) If f'(1) = 3, then a = 0 and the equation of the tangent line at (1, f(1)) is 3x - y - 2 = 0
    (II) The maximum value of f(x) in the interval [0, 2] is max{8 - 4a, 0} for a < 3 and 0 for a ≥ 3 -/
theorem function_analysis (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 * (x - a)) :
  (deriv f 1 = 3 → a = 0 ∧ ∀ x y, 3*x - y - 2 = 0 ↔ y = f x ∧ x = 1) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ max (8 - 4*a) 0 ∧ (a ≥ 3 → f x ≤ 0)) :=
sorry

end NUMINAMATH_CALUDE_function_analysis_l1275_127596


namespace NUMINAMATH_CALUDE_jane_bagels_l1275_127519

theorem jane_bagels (b m : ℕ) : 
  b + m = 5 →
  (75 * b + 50 * m) % 100 = 0 →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_jane_bagels_l1275_127519


namespace NUMINAMATH_CALUDE_smallest_absolute_value_rational_l1275_127544

theorem smallest_absolute_value_rational : 
  ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_rational_l1275_127544


namespace NUMINAMATH_CALUDE_cookies_per_row_l1275_127563

theorem cookies_per_row (num_trays : ℕ) (rows_per_tray : ℕ) (total_cookies : ℕ) :
  num_trays = 4 →
  rows_per_tray = 5 →
  total_cookies = 120 →
  total_cookies / (num_trays * rows_per_tray) = 6 := by
sorry

end NUMINAMATH_CALUDE_cookies_per_row_l1275_127563


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_6_million_l1275_127577

theorem scientific_notation_of_1_6_million :
  ∃ (a : ℝ) (n : ℤ), 1600000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.6 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_6_million_l1275_127577


namespace NUMINAMATH_CALUDE_ninas_school_students_l1275_127524

theorem ninas_school_students : ∀ (n m : ℕ),
  n = 5 * m →
  n + m = 4800 →
  (n - 200) + (m + 200) = 2 * (m + 200) →
  n = 4000 := by
sorry

end NUMINAMATH_CALUDE_ninas_school_students_l1275_127524


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l1275_127522

def systematic_sampling (population : ℕ) (sample_size : ℕ) (first_drawn : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ := 
  let interval := population / sample_size
  let sequence := fun n => first_drawn + (n - 1) * interval
  let n := (range_start - first_drawn + interval - 1) / interval
  sequence n

theorem systematic_sampling_result :
  systematic_sampling 960 32 9 401 430 = 429 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l1275_127522


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1275_127579

theorem intersection_implies_a_value (A B : Set ℝ) (a : ℝ) :
  A = {-1, 1, 3} →
  B = {a + 2, a^2 + 4} →
  A ∩ B = {3} →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1275_127579


namespace NUMINAMATH_CALUDE_arithmetic_mean_sum_l1275_127593

theorem arithmetic_mean_sum (x y : ℝ) : 
  let s : Finset ℝ := {6, 13, 18, 4, x, y}
  (s.sum id) / s.card = 12 → x + y = 31 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sum_l1275_127593


namespace NUMINAMATH_CALUDE_leftover_value_is_685_l1275_127573

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

def roll_size : RollSize := { quarters := 30, dimes := 60 }

def michael_coins : CoinCount := { quarters := 94, dimes := 184 }
def sara_coins : CoinCount := { quarters := 137, dimes := 312 }

def total_coins : CoinCount := {
  quarters := michael_coins.quarters + sara_coins.quarters,
  dimes := michael_coins.dimes + sara_coins.dimes
}

def leftover_coins : CoinCount := {
  quarters := total_coins.quarters % roll_size.quarters,
  dimes := total_coins.dimes % roll_size.dimes
}

def leftover_value : Rat :=
  (leftover_coins.quarters : Rat) * (1/4) + (leftover_coins.dimes : Rat) * (1/10)

theorem leftover_value_is_685 : leftover_value = 685/100 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_685_l1275_127573


namespace NUMINAMATH_CALUDE_expand_expression_l1275_127542

theorem expand_expression (x : ℝ) : 20 * (3 * x + 7 - 2 * x^2) = 60 * x + 140 - 40 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1275_127542


namespace NUMINAMATH_CALUDE_iron_chain_links_count_l1275_127505

/-- Represents a piece of an iron chain -/
structure ChainPiece where
  length : ℝ
  links : ℕ

/-- Calculates the length of a chain piece given the number of links and internal diameter -/
def chainLength (links : ℕ) (internalDiameter : ℝ) : ℝ :=
  (links : ℝ) * internalDiameter + 1

theorem iron_chain_links_count :
  let shortPiece : ChainPiece := ⟨22, 9⟩
  let longPiece : ChainPiece := ⟨36, 15⟩
  let internalDiameter : ℝ := 7/3

  (longPiece.links = shortPiece.links + 6) ∧
  (chainLength shortPiece.links internalDiameter = shortPiece.length) ∧
  (chainLength longPiece.links internalDiameter = longPiece.length) :=
by
  sorry

end NUMINAMATH_CALUDE_iron_chain_links_count_l1275_127505


namespace NUMINAMATH_CALUDE_property_set_characterization_l1275_127504

/-- A number is a prime power if it's of the form p^k where p is prime and k ≥ 1 -/
def IsPrimePower (n : Nat) : Prop :=
  ∃ (p k : Nat), Prime p ∧ k ≥ 1 ∧ n = p^k

/-- A perfect square n satisfies the property if for all its divisors a ≥ 15, a + 15 is a prime power -/
def SatisfiesProperty (n : Nat) : Prop :=
  ∃ m : Nat, n = m^2 ∧ ∀ a : Nat, a ≥ 15 → a ∣ n → IsPrimePower (a + 15)

/-- The set of all perfect squares satisfying the property -/
def PropertySet : Set Nat :=
  {n : Nat | SatisfiesProperty n}

/-- The theorem stating that the set of perfect squares satisfying the property
    is exactly {1, 4, 9, 16, 49, 64, 196} -/
theorem property_set_characterization :
  PropertySet = {1, 4, 9, 16, 49, 64, 196} := by
  sorry


end NUMINAMATH_CALUDE_property_set_characterization_l1275_127504


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1275_127575

theorem arithmetic_mean_geq_geometric_mean {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1275_127575


namespace NUMINAMATH_CALUDE_quadratic_equations_intersection_l1275_127500

theorem quadratic_equations_intersection (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x : ℝ, x ∈ M ↔ x^2 - p*x + 6 = 0) ∧
    (∀ x : ℝ, x ∈ N ↔ x^2 + 6*x - q = 0) ∧
    (M ∩ N = {2})) →
  p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_intersection_l1275_127500


namespace NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l1275_127539

/-- An arithmetic sequence with first term 100 and common difference -4 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  100 - 4 * (n - 1)

theorem arithmetic_sequence_25th_term :
  arithmetic_sequence 25 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l1275_127539


namespace NUMINAMATH_CALUDE_A_xor_B_equals_one_three_l1275_127565

-- Define the ⊕ operation
def setXor (M P : Set ℝ) : Set ℝ := {x | x ∈ M ∨ x ∈ P ∧ x ∉ M ∩ P}

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

-- Theorem statement
theorem A_xor_B_equals_one_three : setXor A B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_xor_B_equals_one_three_l1275_127565


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1275_127554

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4*x) * (7 + 4*x) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1275_127554


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1275_127552

theorem complex_equation_solution (z : ℂ) :
  (2 * z - Complex.I) * (2 - Complex.I) = 5 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1275_127552


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1275_127581

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + y * z = 30) (h2 : y * z + z * x = 36) (h3 : z * x + x * y = 42) :
  x + y + z = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1275_127581


namespace NUMINAMATH_CALUDE_expression_evaluation_l1275_127511

theorem expression_evaluation (x : ℝ) (h : x = 6) :
  (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8) = 43264 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1275_127511


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l1275_127578

/-- Calculates the total number of visitors to a library in a week given specific conditions --/
theorem library_visitors_theorem (monday_visitors : ℕ) 
  (h1 : monday_visitors = 50)
  (h2 : ∃ tuesday_visitors : ℕ, tuesday_visitors = 2 * monday_visitors)
  (h3 : ∃ wednesday_visitors : ℕ, wednesday_visitors = 2 * monday_visitors)
  (h4 : ∃ thursday_visitors : ℕ, thursday_visitors = 3 * (2 * monday_visitors))
  (h5 : ∃ weekend_visitors : ℕ, weekend_visitors = 3 * 20) :
  monday_visitors + 
  (2 * monday_visitors) + 
  (2 * monday_visitors) + 
  (3 * (2 * monday_visitors)) + 
  (3 * 20) = 610 := by
    sorry

end NUMINAMATH_CALUDE_library_visitors_theorem_l1275_127578


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l1275_127587

theorem sum_of_quadratic_roots (a b c d e : ℝ) (h : ∀ x, a * x^2 + b * x + c = d * x + e) :
  let x₁ := (10 + 4 * Real.sqrt 5) / 2
  let x₂ := (10 - 4 * Real.sqrt 5) / 2
  x₁ + x₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l1275_127587


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1275_127557

theorem power_of_two_equality (y : ℕ) : (1 / 8 : ℝ) * 2^40 = 2^y → y = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1275_127557


namespace NUMINAMATH_CALUDE_roots_of_equation_l1275_127590

theorem roots_of_equation : 
  let f (x : ℝ) := 21 / (x^2 - 9) - 3 / (x - 3) - 1
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -7 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1275_127590


namespace NUMINAMATH_CALUDE_ones_digit_of_nine_to_27_l1275_127586

def ones_digit (n : ℕ) : ℕ := n % 10

def power_of_nine_ones_digit (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

theorem ones_digit_of_nine_to_27 :
  ones_digit (9^27) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_nine_to_27_l1275_127586


namespace NUMINAMATH_CALUDE_smallest_k_for_cos_squared_one_l1275_127594

theorem smallest_k_for_cos_squared_one :
  ∃ k : ℕ+, 
    (∀ m : ℕ+, m < k → (Real.cos ((m.val ^ 2 + 7 ^ 2 : ℝ) * Real.pi / 180)) ^ 2 ≠ 1) ∧
    (Real.cos ((k.val ^ 2 + 7 ^ 2 : ℝ) * Real.pi / 180)) ^ 2 = 1 ∧
    k = 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_cos_squared_one_l1275_127594


namespace NUMINAMATH_CALUDE_slope_product_theorem_l1275_127508

theorem slope_product_theorem (m n p : ℝ) : 
  m ≠ 0 ∧ n ≠ 0 ∧ p ≠ 0 →  -- none of the lines are horizontal
  (∃ θ₁ θ₂ θ₃ : ℝ, 
    θ₁ = 3 * θ₂ ∧  -- L₁ makes three times the angle with the horizontal as L₂
    θ₃ = θ₁ / 2 ∧  -- L₃ makes half the angle of L₁
    m = Real.tan θ₁ ∧ 
    n = Real.tan θ₂ ∧ 
    p = Real.tan θ₃) →
  m = 3 * n →  -- L₁ has 3 times the slope of L₂
  m = 5 * p →  -- L₁ has 5 times the slope of L₃
  m * n * p = Real.sqrt 3 / 15 := by
sorry

end NUMINAMATH_CALUDE_slope_product_theorem_l1275_127508


namespace NUMINAMATH_CALUDE_faster_speed_proof_l1275_127516

/-- Proves that the faster speed is 12 kmph given the problem conditions -/
theorem faster_speed_proof (distance : ℝ) (slow_speed : ℝ) (late_time : ℝ) (early_time : ℝ) 
  (h1 : distance = 24)
  (h2 : slow_speed = 9)
  (h3 : late_time = 1/3)  -- 20 minutes in hours
  (h4 : early_time = 1/3) -- 20 minutes in hours
  : ∃ (fast_speed : ℝ), 
    distance / slow_speed - distance / fast_speed = late_time + early_time ∧ 
    fast_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_faster_speed_proof_l1275_127516


namespace NUMINAMATH_CALUDE_intersection_sum_l1275_127509

theorem intersection_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) → 
  (3 = (1/3) * 3 + d) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1275_127509


namespace NUMINAMATH_CALUDE_scientific_notation_of_6_1757_million_l1275_127598

theorem scientific_notation_of_6_1757_million :
  let original_number : ℝ := 6.1757 * 1000000
  original_number = 6.1757 * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_6_1757_million_l1275_127598


namespace NUMINAMATH_CALUDE_competition_probabilities_l1275_127559

def score_prob (p1 p2 p3 : ℝ) : ℝ × ℝ :=
  let prob_300 := p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
  let prob_400 := p1 * p2 * p3
  (prob_300, prob_300 + prob_400)

theorem competition_probabilities :
  let (prob_300, prob_at_least_300) := score_prob 0.8 0.7 0.6
  prob_300 = 0.228 ∧ prob_at_least_300 = 0.564 := by
  sorry

end NUMINAMATH_CALUDE_competition_probabilities_l1275_127559


namespace NUMINAMATH_CALUDE_power_function_coefficient_l1275_127571

theorem power_function_coefficient (m : ℝ) : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = (m^2 + 2*m - 2) * x^4 ∧ ∃ (k : ℝ), ∀ x, y x = x^k) → 
  m = 1 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_power_function_coefficient_l1275_127571


namespace NUMINAMATH_CALUDE_parallel_vectors_iff_y_eq_3_l1275_127514

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The problem statement -/
theorem parallel_vectors_iff_y_eq_3 :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (6, y)
  parallel a b ↔ y = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_iff_y_eq_3_l1275_127514


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l1275_127589

theorem complex_exponential_to_rectangular : 2 * Complex.exp (15 * π * I / 4) = Complex.mk (Real.sqrt 2) (- Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l1275_127589


namespace NUMINAMATH_CALUDE_bricks_used_l1275_127551

/-- Calculates the total number of bricks used in a construction project --/
theorem bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) 
  (h1 : courses_per_wall = 15)
  (h2 : bricks_per_course = 25)
  (h3 : total_walls = 8) : 
  (total_walls - 1) * courses_per_wall * bricks_per_course + 
  (courses_per_wall - 1) * bricks_per_course = 2975 := by
  sorry

end NUMINAMATH_CALUDE_bricks_used_l1275_127551


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_one_l1275_127526

theorem sqrt_sum_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b > 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_one_l1275_127526


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1275_127527

theorem quadratic_equation_solution :
  ∃ (m n p : ℕ) (x₁ x₂ : ℚ),
    -- The equation is satisfied by both solutions
    x₁ * (5 * x₁ - 11) = -2 ∧
    x₂ * (5 * x₂ - 11) = -2 ∧
    -- Solutions are in the required form
    x₁ = (m + Real.sqrt n) / p ∧
    x₂ = (m - Real.sqrt n) / p ∧
    -- m, n, and p have a greatest common divisor of 1
    Nat.gcd m (Nat.gcd n p) = 1 ∧
    -- Sum of m, n, and p is 102
    m + n + p = 102 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1275_127527


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l1275_127506

/-- Given a right triangle with two sides of 8 units and 15 units, 
    the least possible length of the third side is √161 units. -/
theorem least_possible_third_side_length : ∀ a b c : ℝ,
  a = 8 →
  b = 15 →
  (a = c ∧ b * b = c * c - a * a) ∨ 
  (b = c ∧ a * a = c * c - b * b) ∨
  (c * c = a * a + b * b) →
  c ≥ Real.sqrt 161 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l1275_127506


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l1275_127531

theorem P_greater_than_Q : ∀ x : ℝ, (x^2 + 2) > 2*x := by sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l1275_127531


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1275_127546

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp a α → perp b α → parallel a b := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1275_127546


namespace NUMINAMATH_CALUDE_mascot_sales_theorem_l1275_127562

/-- Represents the monthly sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -2 * x + 360

/-- Represents the monthly sales profit as a function of selling price -/
def sales_profit (x : ℝ) : ℝ := sales_quantity x * (x - 30)

theorem mascot_sales_theorem :
  -- 1. The linear function satisfies the given conditions
  (sales_quantity 30 = 300 ∧ sales_quantity 45 = 270) ∧
  -- 2. The maximum profit occurs at x = 105 and equals 11250
  (∀ x : ℝ, sales_profit x ≤ sales_profit 105) ∧
  (sales_profit 105 = 11250) ∧
  -- 3. The minimum selling price for profit ≥ 10000 is 80
  (∀ x : ℝ, x ≥ 80 → sales_profit x ≥ 10000) ∧
  (∀ x : ℝ, x < 80 → sales_profit x < 10000) :=
by sorry

end NUMINAMATH_CALUDE_mascot_sales_theorem_l1275_127562


namespace NUMINAMATH_CALUDE_original_mixture_composition_l1275_127576

def original_mixture (acid water : ℝ) : Prop :=
  acid > 0 ∧ water > 0

def after_adding_water (acid water : ℝ) : Prop :=
  acid / (acid + water + 2) = 1/4

def after_adding_acid (acid water : ℝ) : Prop :=
  (acid + 3) / (acid + water + 5) = 2/5

theorem original_mixture_composition (acid water : ℝ) :
  original_mixture acid water →
  after_adding_water acid water →
  after_adding_acid acid water →
  acid / (acid + water) = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_original_mixture_composition_l1275_127576


namespace NUMINAMATH_CALUDE_choose_two_from_even_set_l1275_127568

theorem choose_two_from_even_set (n : ℕ) (k : ℕ) (h1 : n > 0) (h2 : n = 2 * k) :
  Nat.choose n 2 = k * (2 * k - 1) :=
by sorry

end NUMINAMATH_CALUDE_choose_two_from_even_set_l1275_127568


namespace NUMINAMATH_CALUDE_travis_cereal_cost_l1275_127507

/-- The amount Travis spends on cereal in a year -/
def cereal_cost (boxes_per_week : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (boxes_per_week : ℚ) * cost_per_box * (weeks_per_year : ℚ)

/-- Proof that Travis spends $312.00 on cereal in a year -/
theorem travis_cereal_cost :
  cereal_cost 2 3 52 = 312 := by
  sorry

end NUMINAMATH_CALUDE_travis_cereal_cost_l1275_127507


namespace NUMINAMATH_CALUDE_window_purchase_savings_l1275_127538

/-- Represents the window store's pricing and discount structure -/
structure WindowStore where
  regularPrice : ℕ := 120
  freeWindowThreshold : ℕ := 5
  bulkDiscountThreshold : ℕ := 10
  bulkDiscountRate : ℚ := 0.05

/-- Calculates the cost of windows for an individual purchase -/
def individualCost (store : WindowStore) (quantity : ℕ) : ℚ :=
  let freeWindows := quantity / store.freeWindowThreshold
  let paidWindows := quantity - freeWindows
  let basePrice := paidWindows * store.regularPrice
  if quantity > store.bulkDiscountThreshold
  then basePrice * (1 - store.bulkDiscountRate)
  else basePrice

/-- Calculates the cost of windows for a collective purchase -/
def collectiveCost (store : WindowStore) (quantities : List ℕ) : ℚ :=
  let totalQuantity := quantities.sum
  let freeWindows := totalQuantity / store.freeWindowThreshold
  let paidWindows := totalQuantity - freeWindows
  let basePrice := paidWindows * store.regularPrice
  basePrice * (1 - store.bulkDiscountRate)

/-- Theorem statement for the window purchase problem -/
theorem window_purchase_savings (store : WindowStore) :
  let gregQuantity := 9
  let susanQuantity := 13
  let individualTotal := individualCost store gregQuantity + individualCost store susanQuantity
  let collectiveTotal := collectiveCost store [gregQuantity, susanQuantity]
  individualTotal - collectiveTotal = 162 := by
  sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l1275_127538


namespace NUMINAMATH_CALUDE_program_cost_is_40_92_l1275_127560

/-- Represents the cost calculation for a computer program run -/
def program_cost_calculation (milliseconds_per_second : ℝ) 
                             (os_overhead : ℝ) 
                             (cost_per_millisecond : ℝ) 
                             (tape_mounting_cost : ℝ) 
                             (program_runtime_seconds : ℝ) : ℝ :=
  let total_milliseconds := program_runtime_seconds * milliseconds_per_second
  os_overhead + (cost_per_millisecond * total_milliseconds) + tape_mounting_cost

/-- Theorem stating that the total cost for the given program run is $40.92 -/
theorem program_cost_is_40_92 : 
  program_cost_calculation 1000 1.07 0.023 5.35 1.5 = 40.92 := by
  sorry

end NUMINAMATH_CALUDE_program_cost_is_40_92_l1275_127560


namespace NUMINAMATH_CALUDE_polygon_internal_diagonals_l1275_127588

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields and constraints
  sides : ℕ
  sides_eq : sides = n
  sides_ge_3 : n ≥ 3

/-- A diagonal of a polygon -/
structure Diagonal (p : Polygon n) where
  -- Add necessary fields and constraints

/-- Predicate to check if a diagonal is completely inside the polygon -/
def is_inside (d : Diagonal p) : Prop :=
  -- Define the condition for a diagonal to be inside the polygon
  sorry

/-- The number of complete internal diagonals in a polygon -/
def num_internal_diagonals (p : Polygon n) : ℕ :=
  -- Define the number of internal diagonals
  sorry

/-- Theorem: Any polygon with more than 3 sides has at least one internal diagonal,
    and the minimum number of internal diagonals is n-3 -/
theorem polygon_internal_diagonals (n : ℕ) (h : n > 3) (p : Polygon n) :
  (∃ d : Diagonal p, is_inside d) ∧ num_internal_diagonals p = n - 3 :=
sorry

end NUMINAMATH_CALUDE_polygon_internal_diagonals_l1275_127588


namespace NUMINAMATH_CALUDE_tan_pi_plus_theta_l1275_127591

theorem tan_pi_plus_theta (θ : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 3/5 ∧ y = 4/5 ∧ 
   x = Real.cos θ ∧ y = Real.sin θ) →
  Real.tan (π + θ) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_pi_plus_theta_l1275_127591


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1275_127533

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 24) : 
  max x (max (x + 1) (x + 2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1275_127533


namespace NUMINAMATH_CALUDE_product_of_three_terms_l1275_127528

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem product_of_three_terms
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 5 = 4) :
  a 4 * a 5 * a 6 = 64 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_terms_l1275_127528


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1275_127545

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.14 * L) * ((1 - 0.01 * x) * W) = 1.083 * (L * W) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1275_127545


namespace NUMINAMATH_CALUDE_unobserved_planet_exists_l1275_127583

/-- A planet in the Zoolander system -/
structure Planet where
  id : Nat

/-- The planetary system of star Zoolander -/
structure ZoolanderSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  closest_planet : Planet → Planet
  num_planets : planets.card = 2015
  different_distances : ∀ p q r s : Planet, p ≠ q → r ≠ s → (p, q) ≠ (r, s) → distance p q ≠ distance r s
  closest_is_closest : ∀ p q : Planet, p ≠ q → p ∈ planets → q ∈ planets → 
    distance p (closest_planet p) ≤ distance p q

/-- There exists a planet that is not observed by any astronomer -/
theorem unobserved_planet_exists (z : ZoolanderSystem) : 
  ∃ p : Planet, p ∈ z.planets ∧ ∀ q : Planet, q ∈ z.planets → z.closest_planet q ≠ p :=
sorry

end NUMINAMATH_CALUDE_unobserved_planet_exists_l1275_127583


namespace NUMINAMATH_CALUDE_expected_squares_under_attack_l1275_127585

/-- The number of squares on a chessboard -/
def board_size : ℕ := 64

/-- The number of rooks placed on the board -/
def num_rooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def prob_not_attacked_by_one : ℚ := 49 / 64

/-- The expected number of squares under attack by three randomly placed rooks on a chessboard -/
theorem expected_squares_under_attack :
  let prob_attacked := 1 - prob_not_attacked_by_one ^ num_rooks
  (board_size : ℚ) * prob_attacked = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_squares_under_attack_l1275_127585


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1275_127501

/-- Given f(x) = (2x-1)e^x - a(x^2+x) and g(x) = -ax^2 - a, where a ∈ ℝ,
    if f(x) ≥ g(x) for all x ∈ ℝ, then 1 ≤ a ≤ 4e^(3/2) -/
theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, (2*x - 1) * Real.exp x - a*(x^2 + x) ≥ -a*x^2 - a) →
  1 ≤ a ∧ a ≤ 4 * Real.exp (3/2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1275_127501


namespace NUMINAMATH_CALUDE_regular_tetrahedron_unordered_pairs_l1275_127547

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The number of edges in a regular tetrahedron -/
  num_edges : ℕ
  /-- The property that any two edges determine the same plane -/
  edges_same_plane : Unit

/-- The number of unordered pairs of edges in a regular tetrahedron -/
def num_unordered_pairs (t : RegularTetrahedron) : ℕ :=
  (t.num_edges * (t.num_edges - 1)) / 2

/-- Theorem stating that the number of unordered pairs of edges in a regular tetrahedron is 15 -/
theorem regular_tetrahedron_unordered_pairs :
  ∀ t : RegularTetrahedron, t.num_edges = 6 → num_unordered_pairs t = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_regular_tetrahedron_unordered_pairs_l1275_127547


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l1275_127510

theorem no_real_sqrt_negative_quadratic : ∀ x : ℝ, ¬ ∃ y : ℝ, y^2 = -(x^2 + x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l1275_127510


namespace NUMINAMATH_CALUDE_arrangement_ratio_l1275_127548

def C : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
def T : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
def S : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)
def M : ℕ := Nat.factorial 6 / Nat.factorial 2

theorem arrangement_ratio : (C - T + S) / M = 126 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_ratio_l1275_127548


namespace NUMINAMATH_CALUDE_nanometer_to_meter_one_nanometer_def_l1275_127503

/-- Proves that 28 nanometers is equal to 2.8 × 10^(-8) meters. -/
theorem nanometer_to_meter : 
  (28 : ℝ) * (1e-9 : ℝ) = (2.8 : ℝ) * (1e-8 : ℝ) := by
  sorry

/-- Defines the conversion factor from nanometers to meters. -/
def nanometer_to_meter_conversion : ℝ := 1e-9

/-- Proves that 1 nanometer is equal to 10^(-9) meters. -/
theorem one_nanometer_def : 
  (1 : ℝ) * nanometer_to_meter_conversion = (1e-9 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_nanometer_to_meter_one_nanometer_def_l1275_127503


namespace NUMINAMATH_CALUDE_circle_bisection_l1275_127520

/-- Circle represented by its equation -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A circle bisects another circle if the line through their intersection points passes through the center of the bisected circle -/
def bisects (c1 c2 : Circle) : Prop :=
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y ↔ c1.equation x y ∧ c2.equation x y) ∧
                         l c2.center.1 c2.center.2

theorem circle_bisection (a b : ℝ) :
  let c1 : Circle := ⟨(a, b), λ x y => (x - a)^2 + (y - b)^2 = b^2 + 1⟩
  let c2 : Circle := ⟨(-1, -1), λ x y => (x + 1)^2 + (y + 1)^2 = 4⟩
  bisects c1 c2 → a^2 + 2*a + 2*b + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_bisection_l1275_127520


namespace NUMINAMATH_CALUDE_year_square_minus_product_l1275_127595

theorem year_square_minus_product (n : ℕ) : n^2 - (n - 1) * n = n :=
by sorry

end NUMINAMATH_CALUDE_year_square_minus_product_l1275_127595


namespace NUMINAMATH_CALUDE_circle_equations_valid_l1275_127572

-- Define the points
def M : ℝ × ℝ := (-1, 1)
def N : ℝ × ℝ := (0, 2)
def Q : ℝ × ℝ := (2, 0)

-- Define the equations of the circles
def circle_C1_eq (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 5/2
def circle_C2_eq (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 5/2)^2 = 5/2

-- Define the line MN
def line_MN_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem circle_equations_valid :
  -- Circle C1 passes through M, N, and Q
  (circle_C1_eq M.1 M.2 ∧ circle_C1_eq N.1 N.2 ∧ circle_C1_eq Q.1 Q.2) ∧
  -- C2 is the reflection of C1 about line MN
  (∀ x y : ℝ, circle_C1_eq x y ↔ 
    ∃ x' y' : ℝ, circle_C2_eq x' y' ∧ 
    ((x + x')/2 - (y + y')/2 + 2 = 0) ∧
    (y' - y)/(x' - x) = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_valid_l1275_127572


namespace NUMINAMATH_CALUDE_bianca_carrots_l1275_127584

/-- The number of carrots Bianca picked on the first day -/
def first_day_carrots : ℕ := 23

/-- The number of carrots Bianca threw out after the first day -/
def thrown_out_carrots : ℕ := 10

/-- The number of carrots Bianca picked on the second day -/
def second_day_carrots : ℕ := 47

/-- The total number of carrots Bianca has at the end -/
def total_carrots : ℕ := 60

theorem bianca_carrots :
  first_day_carrots - thrown_out_carrots + second_day_carrots = total_carrots :=
by sorry

end NUMINAMATH_CALUDE_bianca_carrots_l1275_127584


namespace NUMINAMATH_CALUDE_specific_factory_production_l1275_127580

/-- A factory produces toys with the following parameters:
  * weekly_production: The total number of toys produced in a week
  * work_days: The number of days worked in a week
  * constant_daily_production: Whether the daily production is constant throughout the week
-/
structure ToyFactory where
  weekly_production : ℕ
  work_days : ℕ
  constant_daily_production : Prop

/-- Calculate the daily toy production for a given factory -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.work_days

/-- Theorem stating that for a factory producing 6500 toys per week,
    working 5 days a week, with constant daily production,
    the daily production is 1300 toys -/
theorem specific_factory_production :
  ∀ (factory : ToyFactory),
    factory.weekly_production = 6500 ∧
    factory.work_days = 5 ∧
    factory.constant_daily_production →
    daily_production factory = 1300 := by
  sorry

end NUMINAMATH_CALUDE_specific_factory_production_l1275_127580


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1275_127592

theorem divisibility_equivalence (m n : ℤ) :
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1275_127592


namespace NUMINAMATH_CALUDE_cube_root_problem_l1275_127569

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1275_127569


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l1275_127561

/-- The number of letters in the word "BALLOON" -/
def n : ℕ := 7

/-- The number of times the letter 'L' appears in "BALLOON" -/
def l_count : ℕ := 2

/-- The number of times the letter 'O' appears in "BALLOON" -/
def o_count : ℕ := 2

/-- The number of unique arrangements of the letters in "BALLOON" -/
def balloon_arrangements : ℕ := n.factorial / (l_count.factorial * o_count.factorial)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l1275_127561


namespace NUMINAMATH_CALUDE_club_membership_count_l1275_127541

theorem club_membership_count :
  let tennis : ℕ := 138
  let baseball : ℕ := 255
  let both : ℕ := 94
  let neither : ℕ := 11
  tennis + baseball - both + neither = 310 :=
by sorry

end NUMINAMATH_CALUDE_club_membership_count_l1275_127541


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l1275_127530

theorem negation_of_exists_lt_is_forall_ge :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l1275_127530
