import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l3463_346383

/-- The decimal representation of x as 0.36̅ -/
def x : ℚ := 36 / 99

/-- The reciprocal of the common fraction form of 0.36̅ -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_repeating_decimal :
  (1 : ℚ) / x = reciprocal := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l3463_346383


namespace NUMINAMATH_CALUDE_baseball_tickets_sold_l3463_346313

theorem baseball_tickets_sold (fair_tickets : ℕ) (baseball_tickets : ℕ) : 
  fair_tickets = 25 →
  2 * fair_tickets + 6 = baseball_tickets →
  baseball_tickets = 56 := by
  sorry

end NUMINAMATH_CALUDE_baseball_tickets_sold_l3463_346313


namespace NUMINAMATH_CALUDE_log_101600_equals_2x_l3463_346356

theorem log_101600_equals_2x (x : ℝ) (h : Real.log 102 = x) : Real.log 101600 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_log_101600_equals_2x_l3463_346356


namespace NUMINAMATH_CALUDE_simplify_expressions_l3463_346301

theorem simplify_expressions :
  (∃ x y : ℝ, x^2 = 8 ∧ y^2 = 3 ∧ 
    x + 2*y - (3*y - Real.sqrt 2) = 3*Real.sqrt 2 - y) ∧
  (∃ a b : ℝ, a^2 = 2 ∧ b^2 = 3 ∧ 
    (a - b)^2 + 2*Real.sqrt (1/3) * 3*a = 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3463_346301


namespace NUMINAMATH_CALUDE_selection_problem_l3463_346398

def number_of_ways_to_select (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem selection_problem (total_students : ℕ) (selected_students : ℕ) 
  (h_total : total_students = 10) 
  (h_selected : selected_students = 4) : 
  (number_of_ways_to_select 8 2) + (number_of_ways_to_select 8 3) = 84 := by
  sorry

#check selection_problem

end NUMINAMATH_CALUDE_selection_problem_l3463_346398


namespace NUMINAMATH_CALUDE_min_value_of_f_l3463_346349

def f (x : ℝ) := x^3 - 3*x + 1

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = -1 ∧ ∀ y ∈ Set.Icc 0 3, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3463_346349


namespace NUMINAMATH_CALUDE_vacuum_cleaner_theorem_l3463_346389

def vacuum_cleaner_problem (initial_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) 
  (dog_walking_earnings : List ℕ) (discount_percent : ℕ) : ℕ × ℕ :=
  let discounted_cost := initial_cost - (initial_cost * discount_percent / 100)
  let total_savings := initial_savings + weekly_allowance * 3 + dog_walking_earnings.sum
  let amount_needed := discounted_cost - total_savings
  let weekly_savings := weekly_allowance + dog_walking_earnings.getLast!
  let weeks_needed := (amount_needed + weekly_savings - 1) / weekly_savings
  (amount_needed, weeks_needed)

theorem vacuum_cleaner_theorem : 
  vacuum_cleaner_problem 420 65 25 [40, 50, 30] 15 = (97, 2) := by
  sorry

end NUMINAMATH_CALUDE_vacuum_cleaner_theorem_l3463_346389


namespace NUMINAMATH_CALUDE_power_of_power_l3463_346342

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3463_346342


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3463_346338

theorem greatest_x_quadratic_inequality :
  ∃ (x : ℝ), x^2 - 6*x + 8 ≤ 0 ∧
  ∀ (y : ℝ), y^2 - 6*y + 8 ≤ 0 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3463_346338


namespace NUMINAMATH_CALUDE_sandwich_problem_l3463_346394

theorem sandwich_problem (sandwich_cost soda_cost total_cost : ℚ) 
                         (num_sodas : ℕ) :
  sandwich_cost = 245/100 →
  soda_cost = 87/100 →
  num_sodas = 4 →
  total_cost = 838/100 →
  ∃ (num_sandwiches : ℕ), 
    num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost ∧
    num_sandwiches = 2 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_problem_l3463_346394


namespace NUMINAMATH_CALUDE_dime_difference_l3463_346372

/-- Represents the number of each type of coin in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- The total number of coins in the piggy bank -/
def total_coins : ℕ := 120

/-- The total value of coins in cents -/
def total_value : ℕ := 1240

/-- Calculates the total number of coins for a given CoinCount -/
def count_coins (c : CoinCount) : ℕ :=
  c.nickels + c.dimes + c.quarters + c.half_dollars

/-- Calculates the total value in cents for a given CoinCount -/
def calculate_value (c : CoinCount) : ℕ :=
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters + 50 * c.half_dollars

/-- Defines a valid CoinCount that satisfies the problem conditions -/
def is_valid_count (c : CoinCount) : Prop :=
  count_coins c = total_coins ∧ calculate_value c = total_value

/-- Finds the maximum number of dimes possible -/
def max_dimes : ℕ := 128

/-- Finds the minimum number of dimes possible -/
def min_dimes : ℕ := 2

theorem dime_difference :
  ∃ (max min : CoinCount),
    is_valid_count max ∧
    is_valid_count min ∧
    max.dimes = max_dimes ∧
    min.dimes = min_dimes ∧
    max_dimes - min_dimes = 126 := by
  sorry

end NUMINAMATH_CALUDE_dime_difference_l3463_346372


namespace NUMINAMATH_CALUDE_inequality_proof_l3463_346392

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Define the set T
def T : Set ℝ := {a | -Real.sqrt 3 < a ∧ a < Real.sqrt 3}

-- Theorem statement
theorem inequality_proof (h : ∀ x : ℝ, ∀ a ∈ T, f x > a^2) (m n : ℝ) (hm : m ∈ T) (hn : n ∈ T) :
  Real.sqrt 3 * |m + n| < |m * n + 3| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3463_346392


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3463_346300

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m - 2) + (m + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3463_346300


namespace NUMINAMATH_CALUDE_gala_trees_count_l3463_346354

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- Determines if an orchard satisfies the given conditions -/
def satisfies_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji + o.cross_pollinated = 153 ∧
  o.pure_fuji = 3 * o.total / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem gala_trees_count (o : Orchard) :
  satisfies_conditions o → o.pure_gala = 45 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_count_l3463_346354


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l3463_346333

theorem smallest_positive_integer_satisfying_congruences : 
  ∃ (x : ℕ), x > 0 ∧ 
  (42 * x + 14) % 26 = 4 ∧ 
  x % 5 = 3 ∧
  (∀ (y : ℕ), y > 0 ∧ (42 * y + 14) % 26 = 4 ∧ y % 5 = 3 → x ≤ y) ∧
  x = 38 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l3463_346333


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3463_346310

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3463_346310


namespace NUMINAMATH_CALUDE_fraction_power_seven_l3463_346321

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_seven_l3463_346321


namespace NUMINAMATH_CALUDE_count_cubic_polynomials_satisfying_property_l3463_346340

/-- A polynomial function of degree 3 -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The property that f(x)f(-x) = f(x³) -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that there are exactly 16 cubic polynomials satisfying the property -/
theorem count_cubic_polynomials_satisfying_property :
  ∃! (s : Finset (ℝ → ℝ)),
    (∀ f ∈ s, ∃ a b c d, f = CubicPolynomial a b c d ∧ SatisfiesProperty f) ∧
    Finset.card s = 16 := by sorry

end NUMINAMATH_CALUDE_count_cubic_polynomials_satisfying_property_l3463_346340


namespace NUMINAMATH_CALUDE_unique_factorization_l3463_346320

/-- The set of all positive integers that cannot be written as a sum of an arithmetic progression with difference d, having at least two terms and consisting of positive integers. -/
def M (d : ℕ) : Set ℕ :=
  {n : ℕ | ∀ (a k : ℕ), k ≥ 2 → n ≠ (k * (2 * a + (k - 1) * d)) / 2}

/-- A is the set M₁ -/
def A : Set ℕ := M 1

/-- B is the set M₂ without the element 2 -/
def B : Set ℕ := M 2 \ {2}

/-- C is the set M₃ -/
def C : Set ℕ := M 3

/-- Every element in C can be uniquely expressed as a product of an element from A and an element from B -/
theorem unique_factorization (c : ℕ) (hc : c ∈ C) :
  ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ c = a * b :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_l3463_346320


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3463_346395

theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 36 → x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3463_346395


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3463_346352

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3463_346352


namespace NUMINAMATH_CALUDE_probability_green_is_25_56_l3463_346380

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def probability_green (containers : List Container) : ℚ :=
  let total_containers := containers.length
  let prob_green_per_container := containers.map (fun c => c.green / (c.red + c.green))
  (prob_green_per_container.sum) / total_containers

/-- The containers as described in the problem -/
def problem_containers : List Container :=
  [⟨8, 4⟩, ⟨2, 5⟩, ⟨2, 5⟩, ⟨4, 4⟩]

/-- The theorem stating the probability of selecting a green ball -/
theorem probability_green_is_25_56 :
  probability_green problem_containers = 25 / 56 := by
  sorry


end NUMINAMATH_CALUDE_probability_green_is_25_56_l3463_346380


namespace NUMINAMATH_CALUDE_purely_imaginary_quotient_implies_a_l3463_346376

def z₁ (a : ℝ) : ℂ := a + 2 * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

theorem purely_imaginary_quotient_implies_a (a : ℝ) :
  (z₁ a / z₂).re = 0 → (z₁ a / z₂).im ≠ 0 → a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_quotient_implies_a_l3463_346376


namespace NUMINAMATH_CALUDE_binomial_coefficient_25_7_l3463_346305

theorem binomial_coefficient_25_7 
  (h1 : Nat.choose 23 5 = 33649)
  (h2 : Nat.choose 23 6 = 42504)
  (h3 : Nat.choose 23 7 = 33649) : 
  Nat.choose 25 7 = 152306 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_25_7_l3463_346305


namespace NUMINAMATH_CALUDE_loss_per_meter_is_five_l3463_346345

/-- Calculates the loss per meter of cloth given the total cloth sold, total selling price, and cost price per meter. -/
def loss_per_meter (total_cloth : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  let total_cost_price := total_cloth * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_cloth

/-- Proves that the loss per meter of cloth is 5 rupees given the specific conditions. -/
theorem loss_per_meter_is_five :
  loss_per_meter 450 18000 45 = 5 := by
  sorry

#eval loss_per_meter 450 18000 45

end NUMINAMATH_CALUDE_loss_per_meter_is_five_l3463_346345


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_eq_one_l3463_346332

/-- The function f(x) = ae^x - sin(x) has an extreme value at x = 0 -/
def has_extreme_value_at_zero (a : ℝ) : Prop :=
  let f := fun x => a * Real.exp x - Real.sin x
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x ≤ f 0 ∨ f x ≥ f 0

/-- If f(x) = ae^x - sin(x) has an extreme value at x = 0, then a = 1 -/
theorem extreme_value_implies_a_eq_one :
  ∀ a : ℝ, has_extreme_value_at_zero a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_eq_one_l3463_346332


namespace NUMINAMATH_CALUDE_bowl_capacity_sum_l3463_346311

theorem bowl_capacity_sum : 
  let second_bowl : ℕ := 600
  let first_bowl : ℕ := (3 * second_bowl) / 4
  let third_bowl : ℕ := first_bowl / 2
  let fourth_bowl : ℕ := second_bowl / 3
  second_bowl + first_bowl + third_bowl + fourth_bowl = 1475 :=
by sorry

end NUMINAMATH_CALUDE_bowl_capacity_sum_l3463_346311


namespace NUMINAMATH_CALUDE_diamond_ratio_equals_five_thirds_l3463_346323

-- Define the diamond operation
def diamond (n m : ℤ) : ℤ := n^2 * m^3

-- Theorem statement
theorem diamond_ratio_equals_five_thirds :
  (diamond 3 5) / (diamond 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_ratio_equals_five_thirds_l3463_346323


namespace NUMINAMATH_CALUDE_tan_beta_value_l3463_346367

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3463_346367


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3463_346396

/-- Given m > 0, n > 0, and the line y = (1/e)x + m + 1 is tangent to the curve y = ln x - n + 2,
    the minimum value of 1/m + 1/n is 4. -/
theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_tangent : ∃ x : ℝ, (1 / Real.exp 1) * x + m + 1 = Real.log x - n + 2 ∧
                        (1 / Real.exp 1) = 1 / x) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1 / m₀ + 1 / n₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3463_346396


namespace NUMINAMATH_CALUDE_polynomial_equality_l3463_346390

/-- Two distinct quadratic polynomials with leading coefficient 1 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c
def g (d e : ℝ) (x : ℝ) : ℝ := x^2 + d*x + e

/-- The theorem stating the point of equality for the polynomials -/
theorem polynomial_equality (b c d e : ℝ) 
  (h_distinct : b ≠ d ∨ c ≠ e)
  (h_sum_eq : f b c 1 + f b c 10 + f b c 100 = g d e 1 + g d e 10 + g d e 100) :
  f b c 37 = g d e 37 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3463_346390


namespace NUMINAMATH_CALUDE_calculate_F_2_f_3_l3463_346307

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 3*a + 2
def F (a b : ℝ) : ℝ := b^2 + a + 1

-- State the theorem
theorem calculate_F_2_f_3 : F 2 (f 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculate_F_2_f_3_l3463_346307


namespace NUMINAMATH_CALUDE_asian_games_ticket_scientific_notation_l3463_346370

theorem asian_games_ticket_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ 113700 = a * (10 : ℝ) ^ b ∧ a = 1.137 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_asian_games_ticket_scientific_notation_l3463_346370


namespace NUMINAMATH_CALUDE_cell_count_after_3_hours_l3463_346351

/-- The number of cells after a given number of half-hour intervals, starting with one cell -/
def cell_count (n : ℕ) : ℕ := 2^n

/-- The number of half-hour intervals in 3 hours -/
def intervals_in_3_hours : ℕ := 6

theorem cell_count_after_3_hours :
  cell_count intervals_in_3_hours = 64 := by
  sorry

end NUMINAMATH_CALUDE_cell_count_after_3_hours_l3463_346351


namespace NUMINAMATH_CALUDE_number_calculation_l3463_346322

theorem number_calculation (x : ℝ) (h : 0.45 * x = 162) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3463_346322


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3463_346375

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3463_346375


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_l3463_346382

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_periodicity :
  (∀ n, 10 ∣ (fib (n + 60) - fib n)) ∧
  (∀ k, 1 ≤ k → k < 60 → ∃ n, ¬(10 ∣ (fib (n + k) - fib n))) ∧
  (∀ n, 100 ∣ (fib (n + 300) - fib n)) ∧
  (∀ k, 1 ≤ k → k < 300 → ∃ n, ¬(100 ∣ (fib (n + k) - fib n))) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_l3463_346382


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3463_346335

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  t : ℝ
  h1 : 0 < d
  h2 : a 1 = 1
  h3 : ∀ n, 2 * (a n * a (n + 1) + 1) = t * n * (1 + a n)
  h4 : ∀ n, a (n + 1) = a n + d

/-- The general term of the arithmetic sequence is 2n - 1 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, n > 0 → seq.a n = 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3463_346335


namespace NUMINAMATH_CALUDE_square_semicircle_perimeter_l3463_346358

theorem square_semicircle_perimeter (π : Real) (h : π > 0) : 
  let square_side : Real := 4 / π
  let semicircle_radius : Real := square_side / 2
  let num_semicircles : Nat := 4
  num_semicircles * (π * semicircle_radius) = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_semicircle_perimeter_l3463_346358


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l3463_346350

/-- Given a quadratic function y = (x - a)² + a - 1, where a is a constant,
    and (m, n) is a point on the graph with m > 0, prove that if m > 2a, then n > -5/4. -/
theorem quadratic_function_inequality (a m n : ℝ) : 
  m > 0 → 
  n = (m - a)^2 + a - 1 → 
  m > 2*a → 
  n > -5/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l3463_346350


namespace NUMINAMATH_CALUDE_homework_ratio_proof_l3463_346369

/-- Given a total of 15 problems and 6 problems finished, 
    prove that the simplified ratio of problems still to complete 
    to problems already finished is 3:2. -/
theorem homework_ratio_proof (total : ℕ) (finished : ℕ) 
    (h1 : total = 15) (h2 : finished = 6) : 
    (total - finished) / Nat.gcd (total - finished) finished = 3 ∧ 
    finished / Nat.gcd (total - finished) finished = 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_ratio_proof_l3463_346369


namespace NUMINAMATH_CALUDE_exponent_sum_l3463_346346

theorem exponent_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_l3463_346346


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l3463_346316

theorem sphere_radius_from_surface_area :
  ∀ (S R : ℝ), S = 4 * Real.pi → S = 4 * Real.pi * R^2 → R = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l3463_346316


namespace NUMINAMATH_CALUDE_hyperbola_upper_focus_l3463_346315

/-- Given a hyperbola with equation y^2/16 - x^2/9 = 1, prove that the coordinates of the upper focus are (0, 5) -/
theorem hyperbola_upper_focus (x y : ℝ) :
  (y^2 / 16) - (x^2 / 9) = 1 →
  ∃ (a b c : ℝ),
    a = 4 ∧
    b = 3 ∧
    c^2 = a^2 + b^2 ∧
    (0, c) = (0, 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_upper_focus_l3463_346315


namespace NUMINAMATH_CALUDE_max_value_expression_l3463_346327

theorem max_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((2 - x) * (2 - y) * (2 - z)) + 1 / ((2 + x) * (2 + y) * (2 + z))) ≤ 12 / 27 ∧
  (1 / ((2 - 1) * (2 - 1) * (2 - 1)) + 1 / ((2 + 1) * (2 + 1) * (2 + 1))) = 12 / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3463_346327


namespace NUMINAMATH_CALUDE_journey_distance_is_20km_l3463_346336

/-- Represents a round trip journey with a horizontal section and a hill. -/
structure Journey where
  total_time : ℝ
  horizontal_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Calculates the total distance covered in a journey. -/
def total_distance (j : Journey) : ℝ :=
  j.total_time * j.horizontal_speed

/-- Theorem stating that for the given journey parameters, the total distance is 20 km. -/
theorem journey_distance_is_20km (j : Journey)
  (h_time : j.total_time = 5)
  (h_horizontal : j.horizontal_speed = 4)
  (h_uphill : j.uphill_speed = 3)
  (h_downhill : j.downhill_speed = 6) :
  total_distance j = 20 := by
  sorry

#eval total_distance { total_time := 5, horizontal_speed := 4, uphill_speed := 3, downhill_speed := 6 }

end NUMINAMATH_CALUDE_journey_distance_is_20km_l3463_346336


namespace NUMINAMATH_CALUDE_odd_function_sum_l3463_346328

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_sum (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : HasPeriod (fun x ↦ f (2 * x + 1)) 5) (h3 : f 1 = 5) :
  f 2009 + f 2010 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3463_346328


namespace NUMINAMATH_CALUDE_balloon_division_l3463_346318

theorem balloon_division (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k + 4) ↔ (∃ m : ℕ, n = 7 * m + 4) :=
by sorry

end NUMINAMATH_CALUDE_balloon_division_l3463_346318


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3463_346368

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3463_346368


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3463_346329

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 5, 7]
  let B : Matrix (Fin 2) (Fin 3) ℤ := !![2, 1, 4; 1, 0, -2]
  A * B = !![5, 3, 14; 17, 5, 6] := by
sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3463_346329


namespace NUMINAMATH_CALUDE_work_completion_proof_l3463_346378

/-- The number of days it takes the first group to complete the work -/
def days_group1 : ℕ := 35

/-- The number of days it takes the second group to complete the work -/
def days_group2 : ℕ := 50

/-- The number of men in the second group -/
def men_group2 : ℕ := 7

/-- The number of men in the first group -/
def men_group1 : ℕ := men_group2 * days_group2 / days_group1

theorem work_completion_proof : men_group1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3463_346378


namespace NUMINAMATH_CALUDE_medium_stores_sampled_l3463_346359

/-- Represents the total number of stores in the city -/
def total_stores : ℕ := 1500

/-- Represents the ratio of large stores in the city -/
def large_ratio : ℕ := 1

/-- Represents the ratio of medium stores in the city -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores in the city -/
def small_ratio : ℕ := 9

/-- Represents the total number of stores to be sampled -/
def sample_size : ℕ := 30

/-- Theorem stating that the number of medium-sized stores to be sampled is 10 -/
theorem medium_stores_sampled : ℕ := by
  sorry

end NUMINAMATH_CALUDE_medium_stores_sampled_l3463_346359


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l3463_346325

theorem sum_and_ratio_to_difference (x y : ℝ) 
  (sum_eq : x + y = 399)
  (ratio_eq : x / y = 0.9) : 
  y - x = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l3463_346325


namespace NUMINAMATH_CALUDE_milk_distribution_l3463_346326

/-- The milk distribution problem -/
theorem milk_distribution (container_a capacity_a quantity_b quantity_c : ℚ) : 
  capacity_a = 1264 →
  quantity_b + quantity_c = capacity_a →
  quantity_b + 158 = quantity_c - 158 →
  (capacity_a - (quantity_b + 158)) / capacity_a * 100 = 50 := by
  sorry

#check milk_distribution

end NUMINAMATH_CALUDE_milk_distribution_l3463_346326


namespace NUMINAMATH_CALUDE_max_min_sum_xy_xz_yz_l3463_346377

theorem max_min_sum_xy_xz_yz (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  ∃ (M m : ℝ), (∀ t : ℝ, t = x*y + x*z + y*z → t ≤ M) ∧ 
                (∀ t : ℝ, t = x*y + x*z + y*z → m ≤ t) ∧ 
                M + 10*m = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_xy_xz_yz_l3463_346377


namespace NUMINAMATH_CALUDE_two_thirds_of_number_is_fifty_l3463_346373

theorem two_thirds_of_number_is_fifty (y : ℝ) : (2 / 3 : ℝ) * y = 50 → y = 75 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_number_is_fifty_l3463_346373


namespace NUMINAMATH_CALUDE_paint_O_circles_l3463_346303

theorem paint_O_circles (num_circles : Nat) (num_colors : Nat) : 
  num_circles = 4 → num_colors = 3 → num_colors ^ num_circles = 81 := by
  sorry

#check paint_O_circles

end NUMINAMATH_CALUDE_paint_O_circles_l3463_346303


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3463_346384

def total_workers : ℕ := 7
def technicians_salary : ℕ := 8000
def rest_salary : ℕ := 6000

theorem workshop_average_salary :
  (total_workers * technicians_salary) / total_workers = technicians_salary :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3463_346384


namespace NUMINAMATH_CALUDE_min_distance_for_ten_trees_l3463_346360

/-- Calculates the minimum distance to water trees -/
def min_distance_to_water_trees (num_trees : ℕ) (tree_spacing : ℕ) : ℕ :=
  let max_trees_per_trip := 2  -- Xiao Zhang can water 2 trees per trip
  let num_full_trips := (num_trees - 1) / max_trees_per_trip
  let trees_on_last_trip := (num_trees - 1) % max_trees_per_trip + 1
  let full_trip_distance := num_full_trips * (max_trees_per_trip * tree_spacing * 2)
  let last_trip_distance := trees_on_last_trip * tree_spacing
  full_trip_distance + last_trip_distance

/-- The theorem to be proved -/
theorem min_distance_for_ten_trees :
  min_distance_to_water_trees 10 10 = 410 :=
sorry

end NUMINAMATH_CALUDE_min_distance_for_ten_trees_l3463_346360


namespace NUMINAMATH_CALUDE_original_vocabulary_l3463_346355

/-- The number of words learned per day -/
def words_per_day : ℕ := 10

/-- The number of days in 2 years -/
def days_in_two_years : ℕ := 365 * 2

/-- The percentage increase in vocabulary -/
def percentage_increase : ℚ := 1 / 2

theorem original_vocabulary (original : ℕ) : 
  (original : ℚ) + (original : ℚ) * percentage_increase = 
    (words_per_day * days_in_two_years : ℚ) → 
  original = 14600 := by sorry

end NUMINAMATH_CALUDE_original_vocabulary_l3463_346355


namespace NUMINAMATH_CALUDE_stone_pile_combination_l3463_346308

/-- Two piles are considered similar if their sizes differ by at most a factor of two -/
def similar (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A combining operation takes two piles and creates a new pile with their combined size -/
def combine (x y : ℕ) : ℕ := x + y

/-- A sequence of combining operations -/
def combineSequence : List (ℕ × ℕ) → List ℕ
  | [] => []
  | (x, y) :: rest => combine x y :: combineSequence rest

/-- The theorem states that for any number of stones, there exists a sequence of
    combining operations that results in a single pile, using only similar piles -/
theorem stone_pile_combination (n : ℕ) :
  ∃ (seq : List (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ seq → similar x y) ∧
    (combineSequence seq = [n]) ∧
    (seq.foldl (λ acc (x, y) => acc - 1) n = 1) :=
  sorry

end NUMINAMATH_CALUDE_stone_pile_combination_l3463_346308


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l3463_346371

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l3463_346371


namespace NUMINAMATH_CALUDE_cube_monotonicity_l3463_346302

theorem cube_monotonicity (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end NUMINAMATH_CALUDE_cube_monotonicity_l3463_346302


namespace NUMINAMATH_CALUDE_rowing_distance_problem_l3463_346365

/-- Proves that the distance to a destination is 72 km given specific rowing conditions -/
theorem rowing_distance_problem (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : 
  rowing_speed = 10 → 
  current_speed = 2 → 
  total_time = 15 → 
  (rowing_speed + current_speed) * (rowing_speed - current_speed) * total_time / 
    (rowing_speed + current_speed + rowing_speed - current_speed) = 72 := by
  sorry

#check rowing_distance_problem

end NUMINAMATH_CALUDE_rowing_distance_problem_l3463_346365


namespace NUMINAMATH_CALUDE_pink_highlighters_l3463_346387

theorem pink_highlighters (total : ℕ) (yellow : ℕ) (blue : ℕ) (h1 : yellow = 2) (h2 : blue = 5) (h3 : total = 11) :
  total - yellow - blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_l3463_346387


namespace NUMINAMATH_CALUDE_loss_ratio_is_one_third_l3463_346334

/-- Represents a baseball team's season statistics -/
structure BaseballSeason where
  total_games : ℕ
  away_games : ℕ
  home_game_wins : ℕ
  away_game_wins : ℕ
  home_game_wins_extra_innings : ℕ

/-- Calculates the ratio of away game losses to home game losses not in extra innings -/
def loss_ratio (season : BaseballSeason) : Rat :=
  let home_games := season.total_games - season.away_games
  let home_game_losses := home_games - season.home_game_wins
  let away_game_losses := season.away_games - season.away_game_wins
  away_game_losses / home_game_losses

/-- Theorem stating the loss ratio for the given season is 1/3 -/
theorem loss_ratio_is_one_third (season : BaseballSeason) 
  (h1 : season.total_games = 45)
  (h2 : season.away_games = 15)
  (h3 : season.home_game_wins = 6)
  (h4 : season.away_game_wins = 7)
  (h5 : season.home_game_wins_extra_innings = 3) :
  loss_ratio season = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_loss_ratio_is_one_third_l3463_346334


namespace NUMINAMATH_CALUDE_complex_trajectory_l3463_346357

theorem complex_trajectory (x y : ℝ) (z : ℂ) :
  z = x + y * I ∧ Complex.abs (z - 1) = x →
  y^2 = 2 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_trajectory_l3463_346357


namespace NUMINAMATH_CALUDE_largest_nine_digit_divisible_by_127_l3463_346324

theorem largest_nine_digit_divisible_by_127 :
  ∀ n : ℕ, n ≤ 999999999 ∧ n % 127 = 0 → n ≤ 999999945 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_nine_digit_divisible_by_127_l3463_346324


namespace NUMINAMATH_CALUDE_henry_final_book_count_l3463_346348

/-- Calculates the final number of books Henry has after decluttering and acquiring new ones. -/
def final_book_count (initial_books : ℕ) (boxed_books : ℕ) (room_books : ℕ) 
  (coffee_table_books : ℕ) (cookbooks : ℕ) (new_books : ℕ) : ℕ :=
  initial_books - (3 * boxed_books + room_books + coffee_table_books + cookbooks) + new_books

/-- Theorem stating that Henry ends up with 23 books after the process. -/
theorem henry_final_book_count : 
  final_book_count 99 15 21 4 18 12 = 23 := by
  sorry

end NUMINAMATH_CALUDE_henry_final_book_count_l3463_346348


namespace NUMINAMATH_CALUDE_pancake_flour_calculation_l3463_346391

/-- Given a recipe for 20 pancakes requiring 3 cups of flour,
    prove that 27 cups of flour are needed for 180 pancakes. -/
theorem pancake_flour_calculation
  (original_pancakes : ℕ)
  (original_flour : ℕ)
  (desired_pancakes : ℕ)
  (h1 : original_pancakes = 20)
  (h2 : original_flour = 3)
  (h3 : desired_pancakes = 180) :
  (desired_pancakes / original_pancakes) * original_flour = 27 :=
by sorry

end NUMINAMATH_CALUDE_pancake_flour_calculation_l3463_346391


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_half_l3463_346331

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_f_at_pi_half : 
  (deriv f) (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_half_l3463_346331


namespace NUMINAMATH_CALUDE_obtuse_minus_right_is_acute_straight_minus_acute_is_obtuse_l3463_346363

/-- Definition of an obtuse angle -/
def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- Definition of a right angle -/
def is_right_angle (α : ℝ) : Prop := α = 90

/-- Definition of an acute angle -/
def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

/-- Definition of a straight angle -/
def is_straight_angle (α : ℝ) : Prop := α = 180

/-- Theorem: When an obtuse angle is cut by a right angle, the remaining angle is acute -/
theorem obtuse_minus_right_is_acute (α β : ℝ) 
  (h1 : is_obtuse_angle α) (h2 : is_right_angle β) : 
  is_acute_angle (α - β) := by sorry

/-- Theorem: When a straight angle is cut by an acute angle, the remaining angle is obtuse -/
theorem straight_minus_acute_is_obtuse (α β : ℝ) 
  (h1 : is_straight_angle α) (h2 : is_acute_angle β) : 
  is_obtuse_angle (α - β) := by sorry

end NUMINAMATH_CALUDE_obtuse_minus_right_is_acute_straight_minus_acute_is_obtuse_l3463_346363


namespace NUMINAMATH_CALUDE_dalton_needs_four_more_l3463_346385

/-- The amount of additional money Dalton needs to buy his desired items -/
def additional_money_needed (jump_rope_cost board_game_cost ball_cost saved_allowance uncle_gift : ℕ) : ℕ :=
  let total_cost := jump_rope_cost + board_game_cost + ball_cost
  let available_money := saved_allowance + uncle_gift
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Dalton needs $4 more to buy his desired items -/
theorem dalton_needs_four_more :
  additional_money_needed 7 12 4 6 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_four_more_l3463_346385


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3463_346366

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
    a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10 + a₁₁*(x-1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3463_346366


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3463_346344

theorem imaginary_part_of_z (z : ℂ) (h : (Complex.I - 1) * z = Complex.I) : 
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3463_346344


namespace NUMINAMATH_CALUDE_wong_grandchildren_probability_l3463_346388

/-- The number of grandchildren Mr. Wong has -/
def num_grandchildren : ℕ := 12

/-- The probability of a grandchild being male (or female) -/
def gender_probability : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_gender_prob : ℚ := 793/1024

theorem wong_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_gender_outcomes := (num_grandchildren.choose (num_grandchildren / 2))
  (total_outcomes - equal_gender_outcomes : ℚ) / total_outcomes = unequal_gender_prob :=
sorry

end NUMINAMATH_CALUDE_wong_grandchildren_probability_l3463_346388


namespace NUMINAMATH_CALUDE_sandy_sums_theorem_l3463_346304

theorem sandy_sums_theorem (correct_marks : ℕ → ℕ) (incorrect_marks : ℕ → ℕ) 
  (total_marks : ℕ) (correct_sums : ℕ) :
  (correct_marks = λ n => 3 * n) →
  (incorrect_marks = λ n => 2 * n) →
  (total_marks = 50) →
  (correct_sums = 22) →
  (∃ (total_sums : ℕ), 
    total_sums = correct_sums + (total_sums - correct_sums) ∧
    total_marks = correct_marks correct_sums - incorrect_marks (total_sums - correct_sums) ∧
    total_sums = 30) :=
by sorry

end NUMINAMATH_CALUDE_sandy_sums_theorem_l3463_346304


namespace NUMINAMATH_CALUDE_ratio_problem_l3463_346379

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3463_346379


namespace NUMINAMATH_CALUDE_special_list_median_l3463_346361

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list of integers where each n appears n times for 1 ≤ n ≤ 150 -/
def special_list : List ℕ := sorry

/-- The median of a list is the middle value when the list is sorted -/
def median (l : List ℕ) : ℕ := sorry

theorem special_list_median :
  median special_list = 107 := by sorry

end NUMINAMATH_CALUDE_special_list_median_l3463_346361


namespace NUMINAMATH_CALUDE_find_certain_number_l3463_346399

theorem find_certain_number (N : ℚ) : (5/6 * N) - (5/16 * N) = 100 → N = 192 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l3463_346399


namespace NUMINAMATH_CALUDE_unique_solution_l3463_346343

/-- Represents a three-digit number (abc) -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of the five rearranged numbers plus the original number -/
def N : Nat := 3306

/-- The equation that needs to be satisfied -/
def satisfiesEquation (n : ThreeDigitNumber) : Prop :=
  N + n.toNat = 222 * (n.a + n.b + n.c)

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesEquation n ∧ n.toNat = 753 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3463_346343


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3463_346306

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 7 ∨ b = 7 ∨ c = 7)

/-- Checks if a natural number contains the digit 3 -/
def containsThree (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 3 ∨ b = 3 ∨ c = 3)

theorem sum_of_numbers (A : ThreeDigitNumber) (B C : TwoDigitNumber) :
  ((containsSeven A.val ∧ containsSeven B.val) ∨
   (containsSeven A.val ∧ containsSeven C.val) ∨
   (containsSeven B.val ∧ containsSeven C.val)) →
  (containsThree B.val ∧ containsThree C.val) →
  (A.val + B.val + C.val = 208) →
  (B.val + C.val = 76) →
  A.val + B.val + C.val = 247 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3463_346306


namespace NUMINAMATH_CALUDE_fir_trees_not_adjacent_probability_l3463_346312

/-- The number of pine trees -/
def pine_trees : ℕ := 4

/-- The number of cedar trees -/
def cedar_trees : ℕ := 5

/-- The number of fir trees -/
def fir_trees : ℕ := 6

/-- The total number of trees -/
def total_trees : ℕ := pine_trees + cedar_trees + fir_trees

/-- The probability that no two fir trees are next to one another when planted in a random order -/
theorem fir_trees_not_adjacent_probability : 
  (Nat.choose (pine_trees + cedar_trees + 1) fir_trees : ℚ) / 
  (Nat.choose total_trees fir_trees) = 6 / 143 := by sorry

end NUMINAMATH_CALUDE_fir_trees_not_adjacent_probability_l3463_346312


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l3463_346337

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^(1/3)}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}

-- Define the open interval (1, +∞)
def open_interval : Set ℝ := {x : ℝ | x > 1}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l3463_346337


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l3463_346330

/-- A calendrical system where leap years occur every three years -/
structure ModifiedCalendar where
  leapYearInterval : ℕ
  leapYearInterval_eq : leapYearInterval = 3

/-- The number of years in the period we're considering -/
def periodLength : ℕ := 100

/-- The maximum number of leap years in the given period -/
def maxLeapYears (c : ModifiedCalendar) : ℕ :=
  periodLength / c.leapYearInterval

/-- Theorem stating that the maximum number of leap years in a 100-year period is 33 -/
theorem max_leap_years_in_period (c : ModifiedCalendar) :
  maxLeapYears c = 33 := by
  sorry

#check max_leap_years_in_period

end NUMINAMATH_CALUDE_max_leap_years_in_period_l3463_346330


namespace NUMINAMATH_CALUDE_number_of_zeros_l3463_346314

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else Real.log x - 1

-- Define what it means for x to be a zero of f
def isZero (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem number_of_zeros : ∃ (a b : ℝ), a ≠ b ∧ isZero a ∧ isZero b ∧ ∀ c, isZero c → c = a ∨ c = b := by
  sorry

end NUMINAMATH_CALUDE_number_of_zeros_l3463_346314


namespace NUMINAMATH_CALUDE_fraction_ordering_l3463_346374

theorem fraction_ordering : (8 : ℚ) / 25 < 6 / 17 ∧ 6 / 17 < 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3463_346374


namespace NUMINAMATH_CALUDE_shopping_theorem_l3463_346364

def shopping_scenario (initial_money : ℝ) : Prop :=
  let after_first_store := initial_money / 2 - 2000
  let after_second_store := after_first_store / 2 - 2000
  after_second_store = 0

theorem shopping_theorem : 
  ∃ (initial_money : ℝ), shopping_scenario initial_money ∧ initial_money = 12000 :=
sorry

end NUMINAMATH_CALUDE_shopping_theorem_l3463_346364


namespace NUMINAMATH_CALUDE_line_equation_proof_l3463_346362

/-- Given a line defined by the equation (3, -4) · ((x, y) - (-2, 8)) = 0,
    prove that its slope-intercept form y = mx + b has m = 3/4 and b = 9.5 -/
theorem line_equation_proof :
  let line_eq := fun (x y : ℝ) => (3 * (x + 2) + (-4) * (y - 8) = 0)
  ∃ (m b : ℝ), m = 3/4 ∧ b = 9.5 ∧ ∀ x y, line_eq x y ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3463_346362


namespace NUMINAMATH_CALUDE_negation_relationship_l3463_346339

theorem negation_relationship (x : ℝ) :
  (¬(x^2 + x - 6 > 0) → ¬(16 - x^2 < 0)) ∧
  ¬(¬(16 - x^2 < 0) → ¬(x^2 + x - 6 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_relationship_l3463_346339


namespace NUMINAMATH_CALUDE_inequality_proof_l3463_346309

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + a*b + b*c + c*a + a*b*c = 7) :
  Real.sqrt (a^2 + b^2 + 2) + Real.sqrt (b^2 + c^2 + 2) + Real.sqrt (c^2 + a^2 + 2) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3463_346309


namespace NUMINAMATH_CALUDE_pizzas_served_today_l3463_346381

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

theorem pizzas_served_today : total_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l3463_346381


namespace NUMINAMATH_CALUDE_joan_books_l3463_346317

theorem joan_books (initial_books sold_books : ℕ) 
  (h1 : initial_books = 33)
  (h2 : sold_books = 26) :
  initial_books - sold_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_books_l3463_346317


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3463_346353

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 9) 
  (h2 : x * y = 2) : 
  x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3463_346353


namespace NUMINAMATH_CALUDE_cliff_total_rocks_l3463_346397

/-- Represents the number of rocks in Cliff's collection --/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  metamorphic : ℕ

/-- Conditions for Cliff's rock collection --/
def cliff_collection : RockCollection → Prop := fun r =>
  r.sedimentary = 2 * r.igneous ∧
  r.metamorphic = 2 * r.igneous ∧
  2 * r.igneous = 3 * 40 ∧
  r.sedimentary / 5 + r.metamorphic * 3 / 4 + 40 = (r.igneous + r.sedimentary + r.metamorphic) / 5

theorem cliff_total_rocks :
  ∀ r : RockCollection, cliff_collection r → r.igneous + r.sedimentary + r.metamorphic = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_cliff_total_rocks_l3463_346397


namespace NUMINAMATH_CALUDE_stock_percentage_return_l3463_346319

/-- Calculate the percentage return on a stock given the income and investment. -/
theorem stock_percentage_return (income : ℝ) (investment : ℝ) :
  income = 650 →
  investment = 6240 →
  abs ((income / investment * 100) - 10.42) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_return_l3463_346319


namespace NUMINAMATH_CALUDE_pirate_costume_group_size_l3463_346393

theorem pirate_costume_group_size 
  (costume_cost : ℕ) 
  (total_spent : ℕ) 
  (h1 : costume_cost = 5)
  (h2 : total_spent = 40) :
  total_spent / costume_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_costume_group_size_l3463_346393


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l3463_346347

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l3463_346347


namespace NUMINAMATH_CALUDE_parabola_focus_at_hyperbola_vertex_l3463_346341

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the right vertex of the hyperbola
def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ y = 0 ∧ x > 0

-- Define the standard form of a parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Theorem statement
theorem parabola_focus_at_hyperbola_vertex :
  ∃ (x₀ y₀ : ℝ), right_vertex x₀ y₀ →
  ∃ (p : ℝ), p > 0 ∧ ∀ (x y : ℝ), parabola (x - x₀) y p ↔ y^2 = 16 * x :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_at_hyperbola_vertex_l3463_346341


namespace NUMINAMATH_CALUDE_A_intersect_CᵣB_equals_zero_one_l3463_346386

-- Define the universal set
def 𝕌 : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 5}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Define the complement of B in ℝ
def CᵣB : Set ℝ := 𝕌 \ B

-- Theorem statement
theorem A_intersect_CᵣB_equals_zero_one : A ∩ CᵣB = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_CᵣB_equals_zero_one_l3463_346386
