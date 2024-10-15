import Mathlib

namespace NUMINAMATH_CALUDE_complex_square_calculation_l1230_123072

theorem complex_square_calculation (z : ℂ) : z = 2 + 3*I → z^2 = -5 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_calculation_l1230_123072


namespace NUMINAMATH_CALUDE_sqrt_calculations_l1230_123013

theorem sqrt_calculations :
  (∀ (x : ℝ), x ≥ 0 → Real.sqrt (x ^ 2) = x) ∧
  (Real.sqrt 21 * Real.sqrt 3 / Real.sqrt 7 = 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l1230_123013


namespace NUMINAMATH_CALUDE_bd_squared_equals_sixteen_l1230_123029

theorem bd_squared_equals_sixteen
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 5)
  (h3 : 3 * a - 2 * b + 4 * c - d = 17)
  : (b - d)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bd_squared_equals_sixteen_l1230_123029


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1230_123019

-- Problem 1
theorem problem_1 : (1 - Real.sqrt 3) ^ 0 + |-Real.sqrt 2| - 2 * Real.cos (π / 4) + (1 / 4)⁻¹ = 5 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x₁ x₂ : ℝ, x₁ = (3 + 2 * Real.sqrt 3) / 3 ∧ 
                                 x₂ = (3 - 2 * Real.sqrt 3) / 3 ∧ 
                                 3 * x₁^2 - 6 * x₁ - 1 = 0 ∧
                                 3 * x₂^2 - 6 * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1230_123019


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1230_123078

/-- Given a cubic polynomial x^3 + px + q = 0 where p and q are rational,
    if 3 - √5 is a root and the polynomial has an integer root,
    then this integer root must be -6. -/
theorem cubic_polynomial_integer_root
  (p q : ℚ)
  (h1 : ∃ (x : ℝ), x^3 + p*x + q = 0)
  (h2 : (3 - Real.sqrt 5)^3 + p*(3 - Real.sqrt 5) + q = 0)
  (h3 : ∃ (r : ℤ), r^3 + p*r + q = 0) :
  ∃ (r : ℤ), r^3 + p*r + q = 0 ∧ r = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1230_123078


namespace NUMINAMATH_CALUDE_book_selling_price_l1230_123015

/-- Given a book with cost price CP, prove that the original selling price is 720 Rs. -/
theorem book_selling_price (CP : ℝ) : 
  (1.1 * CP = 880) →  -- Condition for 10% gain
  (∃ OSP, OSP = 0.9 * CP) →  -- Condition for 10% loss
  (∃ OSP, OSP = 720) :=
by sorry

end NUMINAMATH_CALUDE_book_selling_price_l1230_123015


namespace NUMINAMATH_CALUDE_reciprocal_of_2022_l1230_123004

theorem reciprocal_of_2022 : (2022⁻¹ : ℝ) = 1 / 2022 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_2022_l1230_123004


namespace NUMINAMATH_CALUDE_lollipops_left_after_sharing_l1230_123039

def raspberry_lollipops : ℕ := 51
def mint_lollipops : ℕ := 121
def chocolate_lollipops : ℕ := 9
def blueberry_lollipops : ℕ := 232
def num_friends : ℕ := 13

theorem lollipops_left_after_sharing :
  (raspberry_lollipops + mint_lollipops + chocolate_lollipops + blueberry_lollipops) % num_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_left_after_sharing_l1230_123039


namespace NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_two_fifths_i_l1230_123050

theorem complex_modulus_three_fourths_minus_two_fifths_i :
  Complex.abs (3/4 - (2/5)*Complex.I) = 17/20 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_two_fifths_i_l1230_123050


namespace NUMINAMATH_CALUDE_simplify_expression_solve_cubic_equation_l1230_123084

-- Problem 1
theorem simplify_expression (a b : ℝ) : 2*a*(a-2*b) - (2*a-b)^2 = -2*a^2 - b^2 := by
  sorry

-- Problem 2
theorem solve_cubic_equation : ∃ x : ℝ, (x-1)^3 - 3 = 3/8 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_cubic_equation_l1230_123084


namespace NUMINAMATH_CALUDE_line_equation_l1230_123067

/-- Given a line parameterized by (x, y) = (3t + 6, 5t - 10) where t is a real number,
    prove that the equation of this line in the form y = mx + b is y = (5/3)x - 20. -/
theorem line_equation (t x y : ℝ) : 
  (x = 3 * t + 6 ∧ y = 5 * t - 10) → 
  y = (5/3) * x - 20 := by sorry

end NUMINAMATH_CALUDE_line_equation_l1230_123067


namespace NUMINAMATH_CALUDE_basketball_game_difference_l1230_123060

theorem basketball_game_difference (total_games won_games lost_games : ℕ) : 
  total_games = 62 →
  won_games > lost_games →
  won_games = 45 →
  lost_games = 17 →
  won_games - lost_games = 28 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_difference_l1230_123060


namespace NUMINAMATH_CALUDE_polynomial_equality_l1230_123008

theorem polynomial_equality : 2090^3 + 2089 * 2090^2 - 2089^2 * 2090 + 2089^3 = 4179 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1230_123008


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l1230_123028

/-- Given that the equation x^2 + y^2 - x + y + m = 0 represents a circle,
    prove that m < 1/2 -/
theorem circle_equation_m_range (m : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - 1/2)^2 + (y + 1/2)^2 = r^2) →
  m < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l1230_123028


namespace NUMINAMATH_CALUDE_constant_function_l1230_123055

def BoundedAbove (f : ℤ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℤ, f n ≤ M

theorem constant_function (f : ℤ → ℝ) 
  (h_bound : BoundedAbove f)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n :=
sorry

end NUMINAMATH_CALUDE_constant_function_l1230_123055


namespace NUMINAMATH_CALUDE_max_value_of_sum_cube_roots_l1230_123003

open Real

theorem max_value_of_sum_cube_roots (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 100) : 
  let S := (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + 
           (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)
  S ≤ 8 / 7 ^ (1/3) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_cube_roots_l1230_123003


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l1230_123068

theorem product_divisible_by_sum_implies_inequality 
  (m n : ℕ) 
  (h : (m * n) % (m + n) = 0) : 
  m + n ≤ n^2 := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l1230_123068


namespace NUMINAMATH_CALUDE_hourglass_problem_l1230_123006

/-- Given two hourglasses that can measure exactly 15 minutes, 
    where one measures 7 minutes, the other measures 2 minutes. -/
theorem hourglass_problem :
  ∀ (x : ℕ), 
    (∃ (n m k : ℕ), n * 7 + m * x + k * (x - 1) = 15 ∧ 
                     n > 0 ∧ m ≥ 0 ∧ k ≥ 0 ∧ 
                     (m = 0 ∨ k = 0)) → 
    x = 2 :=
by sorry

end NUMINAMATH_CALUDE_hourglass_problem_l1230_123006


namespace NUMINAMATH_CALUDE_expand_product_l1230_123076

theorem expand_product (x : ℝ) : (x + 4) * (x^2 - 5*x - 6) = x^3 - x^2 - 26*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1230_123076


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1230_123098

theorem inequality_solution_sets (a : ℝ) :
  let f := fun x => a * x^2 - (a + 2) * x + 2
  (a = -1 → {x : ℝ | f x < 0} = {x : ℝ | x < -2 ∨ x > 1}) ∧
  (a = 0 → {x : ℝ | f x < 0} = {x : ℝ | x > 1}) ∧
  (a < 0 → {x : ℝ | f x < 0} = {x : ℝ | x < 2/a ∨ x > 1}) ∧
  (0 < a ∧ a < 2 → {x : ℝ | f x < 0} = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → {x : ℝ | f x < 0} = ∅) ∧
  (a > 2 → {x : ℝ | f x < 0} = {x : ℝ | 2/a < x ∧ x < 1}) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1230_123098


namespace NUMINAMATH_CALUDE_no_universal_divisor_l1230_123030

-- Define a function to represent the concatenation of digits
def concat_digits (a b : ℕ) : ℕ := sorry

-- Define a function to represent the concatenation of three digits
def concat_three_digits (a n b : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_universal_divisor :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, 
    a ≠ 0 → b ≠ 0 → a < 10 → b < 10 → 
    (concat_three_digits a n b) % (concat_digits a b) = 0 := by sorry

end NUMINAMATH_CALUDE_no_universal_divisor_l1230_123030


namespace NUMINAMATH_CALUDE_grid_game_winner_parity_l1230_123032

/-- Represents the state of a string in the grid game -/
inductive StringState
| Uncut
| Cut

/-- Represents a player in the grid game -/
inductive Player
| First
| Second

/-- Represents the grid game state -/
structure GridGame where
  m : ℕ
  n : ℕ
  strings : Array (Array StringState)

/-- Determines the winner of the grid game based on the dimensions -/
def gridGameWinner (game : GridGame) : Player :=
  if (game.m + game.n) % 2 == 0 then Player.Second else Player.First

/-- The main theorem: The winner of the grid game is determined by the parity of m + n -/
theorem grid_game_winner_parity (game : GridGame) :
  gridGameWinner game = 
    if (game.m + game.n) % 2 == 0 then Player.Second else Player.First :=
by sorry

end NUMINAMATH_CALUDE_grid_game_winner_parity_l1230_123032


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1230_123012

/-- A sequence of 8 positive real numbers -/
def Sequence := Fin 8 → ℝ

/-- Predicate to check if a sequence is positive -/
def is_positive (s : Sequence) : Prop :=
  ∀ i, s i > 0

/-- Predicate to check if a sequence is geometric -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ i : Fin 7, s (i + 1) = q * s i

theorem sufficient_but_not_necessary_condition (s : Sequence) 
  (h_positive : is_positive s) :
  (s 0 + s 7 < s 3 + s 4 → ¬ is_geometric s) ∧
  ∃ s' : Sequence, is_positive s' ∧ ¬ is_geometric s' ∧ s' 0 + s' 7 ≥ s' 3 + s' 4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1230_123012


namespace NUMINAMATH_CALUDE_sqrt_four_twentyfifths_equals_two_fifths_l1230_123033

theorem sqrt_four_twentyfifths_equals_two_fifths : 
  Real.sqrt (4 / 25) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_twentyfifths_equals_two_fifths_l1230_123033


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l1230_123061

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l1230_123061


namespace NUMINAMATH_CALUDE_perfect_apples_count_l1230_123005

/-- Represents the number of perfect apples in a batch with given conditions -/
def number_of_perfect_apples (total_apples : ℕ) 
  (small_ratio medium_ratio large_ratio : ℚ)
  (unripe_ratio partly_ripe_ratio fully_ripe_ratio : ℚ) : ℕ :=
  22

/-- Theorem stating the number of perfect apples under given conditions -/
theorem perfect_apples_count : 
  number_of_perfect_apples 60 (1/4) (1/2) (1/4) (1/3) (1/6) (1/2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_perfect_apples_count_l1230_123005


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1230_123018

/-- Given two parallel vectors a and b, prove that k = -1/2 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-1, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ a = t • b) →
  k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1230_123018


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1230_123082

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 509) :
  ∃ (k : ℕ), k = 14 ∧
  (∀ m : ℕ, m < k → ¬((n - m) % 9 = 0 ∧ (n - m) % 15 = 0)) ∧
  (n - k) % 9 = 0 ∧ (n - k) % 15 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1230_123082


namespace NUMINAMATH_CALUDE_election_vote_difference_l1230_123099

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 4400 →
  candidate_percentage = 30 / 100 →
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -1760 := by
  sorry

end NUMINAMATH_CALUDE_election_vote_difference_l1230_123099


namespace NUMINAMATH_CALUDE_exactly_three_proper_sets_l1230_123085

/-- A set of weights is proper if it can balance any weight from 1 to 200 grams uniquely -/
def IsProperSet (s : Multiset ℕ) : Prop :=
  (s.sum = 200) ∧
  (∀ w : ℕ, w ≥ 1 ∧ w ≤ 200 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The number of different proper sets of weights -/
def NumberOfProperSets : ℕ := 3

/-- Theorem stating that there are exactly 3 different proper sets of weights -/
theorem exactly_three_proper_sets :
  (∃ (sets : Finset (Multiset ℕ)), sets.card = NumberOfProperSets ∧
    (∀ s : Multiset ℕ, s ∈ sets ↔ IsProperSet s)) ∧
  (¬∃ (sets : Finset (Multiset ℕ)), sets.card > NumberOfProperSets ∧
    (∀ s : Multiset ℕ, s ∈ sets ↔ IsProperSet s)) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_proper_sets_l1230_123085


namespace NUMINAMATH_CALUDE_unique_row_with_41_l1230_123048

/-- The number of rows in Pascal's Triangle containing 41 -/
def rows_containing_41 : ℕ := 1

/-- 41 is prime -/
axiom prime_41 : Nat.Prime 41

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- 41 appears as a binomial coefficient -/
axiom exists_41_binomial : ∃ n k : ℕ, binomial n k = 41

theorem unique_row_with_41 : 
  (∃! r : ℕ, ∃ k : ℕ, binomial r k = 41) ∧ rows_containing_41 = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_row_with_41_l1230_123048


namespace NUMINAMATH_CALUDE_delta_phi_equation_solution_l1230_123035

-- Define the functions δ and φ
def δ (x : ℚ) : ℚ := 4 * x + 9
def φ (x : ℚ) : ℚ := 9 * x + 6

-- State the theorem
theorem delta_phi_equation_solution :
  ∃ x : ℚ, δ (φ x) = 10 ∧ x = -23 / 36 := by
  sorry

end NUMINAMATH_CALUDE_delta_phi_equation_solution_l1230_123035


namespace NUMINAMATH_CALUDE_total_salary_after_layoffs_l1230_123080

def total_employees : ℕ := 450
def employees_2000 : ℕ := 150
def employees_2500 : ℕ := 200
def employees_3000 : ℕ := 100

def layoff_round1_2000 : ℚ := 0.20
def layoff_round1_2500 : ℚ := 0.25
def layoff_round1_3000 : ℚ := 0.15

def layoff_round2_2000 : ℚ := 0.10
def layoff_round2_2500 : ℚ := 0.15
def layoff_round2_3000 : ℚ := 0.05

def salary_2000 : ℕ := 2000
def salary_2500 : ℕ := 2500
def salary_3000 : ℕ := 3000

theorem total_salary_after_layoffs :
  let remaining_2000 := employees_2000 - ⌊employees_2000 * layoff_round1_2000⌋ - ⌊(employees_2000 - ⌊employees_2000 * layoff_round1_2000⌋) * layoff_round2_2000⌋
  let remaining_2500 := employees_2500 - ⌊employees_2500 * layoff_round1_2500⌋ - ⌊(employees_2500 - ⌊employees_2500 * layoff_round1_2500⌋) * layoff_round2_2500⌋
  let remaining_3000 := employees_3000 - ⌊employees_3000 * layoff_round1_3000⌋ - ⌊(employees_3000 - ⌊employees_3000 * layoff_round1_3000⌋) * layoff_round2_3000⌋
  remaining_2000 * salary_2000 + remaining_2500 * salary_2500 + remaining_3000 * salary_3000 = 776500 := by
sorry

end NUMINAMATH_CALUDE_total_salary_after_layoffs_l1230_123080


namespace NUMINAMATH_CALUDE_probability_all_cocaptains_l1230_123024

def team1_size : ℕ := 6
def team2_size : ℕ := 9
def team3_size : ℕ := 10
def cocaptains_per_team : ℕ := 3
def num_teams : ℕ := 3
def selected_members : ℕ := 3

theorem probability_all_cocaptains :
  (1 / num_teams) * (
    1 / (team1_size.choose selected_members) +
    1 / (team2_size.choose selected_members) +
    1 / (team3_size.choose selected_members)
  ) = 53 / 2520 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_cocaptains_l1230_123024


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l1230_123063

/-- The required total of age and years of employment for retirement -/
def retirement_total : ℕ := 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1988

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible for retirement -/
def retirement_year : ℕ := 2007

theorem retirement_total_is_70 :
  retirement_total = 
    (retirement_year - hire_year) + -- Years of employment
    (retirement_year - hire_year + hire_age) -- Age at retirement
  := by sorry

end NUMINAMATH_CALUDE_retirement_total_is_70_l1230_123063


namespace NUMINAMATH_CALUDE_interest_problem_l1230_123053

/-- Proves that given the conditions of the interest problem, the principal amount must be 400 -/
theorem interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) : 
  (P * (R + 6) * 10 / 100 - P * R * 10 / 100 = 240) → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l1230_123053


namespace NUMINAMATH_CALUDE_problem_statement_l1230_123087

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem problem_statement :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ m : ℝ, (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) → m ≤ -1/3) ∧
  (∀ a > 0, (∃ x₀ ≥ 1, f x₀ < a * (-x₀^2 + 3*x₀)) → 
    ((1/2*(Real.exp 1 + Real.exp (-1)) < a ∧ a < Real.exp 1 → a^(Real.exp 1 - 1) > Real.exp (a - 1)) ∧
     (a > Real.exp 1 → a^(Real.exp 1 - 1) < Real.exp (a - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1230_123087


namespace NUMINAMATH_CALUDE_billy_ferris_wheel_rides_l1230_123001

/-- The number of times Billy rode the ferris wheel -/
def F : ℕ := sorry

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def ticket_cost : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

theorem billy_ferris_wheel_rides : F = 7 := by
  sorry

end NUMINAMATH_CALUDE_billy_ferris_wheel_rides_l1230_123001


namespace NUMINAMATH_CALUDE_star_value_of_a_l1230_123070

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^3

-- Theorem statement
theorem star_value_of_a : 
  ∃ a : ℝ, star a 3 = 15 ∧ a = 21 :=
by sorry

end NUMINAMATH_CALUDE_star_value_of_a_l1230_123070


namespace NUMINAMATH_CALUDE_x_plus_3_over_x_is_fraction_l1230_123089

/-- A fraction is an expression with a variable in the denominator. -/
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, f x = (n x) / (d x) ∧ d x ≠ 0

/-- The expression (x + 3) / x is a fraction. -/
theorem x_plus_3_over_x_is_fraction :
  is_fraction (λ x => (x + 3) / x) :=
sorry

end NUMINAMATH_CALUDE_x_plus_3_over_x_is_fraction_l1230_123089


namespace NUMINAMATH_CALUDE_returning_players_l1230_123064

theorem returning_players (new_players : ℕ) (total_groups : ℕ) (players_per_group : ℕ) : 
  new_players = 48 → total_groups = 9 → players_per_group = 6 →
  total_groups * players_per_group - new_players = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_returning_players_l1230_123064


namespace NUMINAMATH_CALUDE_mario_age_is_four_l1230_123052

/-- Mario and Maria's ages satisfy the given conditions -/
structure AgesProblem where
  mario : ℕ
  maria : ℕ
  sum_ages : mario + maria = 7
  age_difference : mario = maria + 1

/-- Mario's age is 4 given the conditions -/
theorem mario_age_is_four (p : AgesProblem) : p.mario = 4 := by
  sorry

end NUMINAMATH_CALUDE_mario_age_is_four_l1230_123052


namespace NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l1230_123022

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- The polar equation of a line parallel to the polar axis -/
def isPolarLineParallelToAxis (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ θ, f θ * Real.sin θ = k

theorem polar_line_through_point_parallel_to_axis 
  (P : PolarPoint) 
  (h_P : P.ρ = 2 ∧ P.θ = π/3) :
  isPolarLineParallelToAxis (fun θ ↦ Real.sqrt 3 / Real.sin θ) ∧ 
  (Real.sqrt 3 / Real.sin P.θ) * Real.sin P.θ = P.ρ * Real.sin P.θ :=
sorry

end NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l1230_123022


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1230_123057

theorem negative_fraction_comparison : -3/5 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1230_123057


namespace NUMINAMATH_CALUDE_special_line_equation_l1230_123014

/-- A line passing through (1,2) with its y-intercept twice its x-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1,2) -/
  passes_through : m + b = 2
  /-- The y-intercept is twice the x-intercept -/
  intercept_condition : b = 2 * (-b / m)

/-- The equation of the special line is either y = 2x or 2x + y - 4 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -2 ∧ l.b = 4) := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l1230_123014


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l1230_123042

theorem wrong_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 180)
  (h3 : real_avg = 178)
  (h4 : actual_height = 106)
  : ∃ wrong_height : ℝ,
    (n * initial_avg - wrong_height + actual_height) / n = real_avg ∧ 
    wrong_height = 176 := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l1230_123042


namespace NUMINAMATH_CALUDE_equation_solution_l1230_123010

theorem equation_solution : ∃ (a b : ℤ), a^2 * b^2 + a^2 + b^2 + 1 = 2005 ∧ (a = 2 ∧ b = 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1230_123010


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_quadratic_roots_l1230_123037

theorem sum_reciprocals_of_quadratic_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b)
  (eq_a : a^2 + a - 2007 = 0) (eq_b : b^2 + b - 2007 = 0) :
  1/a + 1/b = 1/2007 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_quadratic_roots_l1230_123037


namespace NUMINAMATH_CALUDE_triangle_base_calculation_l1230_123056

/-- Given a triangle with area 46 cm² and height 10 cm, prove its base is 9.2 cm -/
theorem triangle_base_calculation (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 46 →
  height = 10 →
  area = (base * height) / 2 →
  base = 9.2 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_calculation_l1230_123056


namespace NUMINAMATH_CALUDE_point_in_region_range_l1230_123094

theorem point_in_region_range (a : ℝ) : 
  (2 * a + 2 < 4) → (a ∈ Set.Iio 1) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_range_l1230_123094


namespace NUMINAMATH_CALUDE_max_prime_area_rectangle_l1230_123088

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def rectangleArea (l w : ℕ) : ℕ := l * w

def rectanglePerimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_prime_area_rectangle (l w : ℕ) :
  rectanglePerimeter l w = 40 →
  isPrime (rectangleArea l w) →
  rectangleArea l w ≤ 19 ∧
  (rectangleArea l w = 19 → (l = 1 ∧ w = 19) ∨ (l = 19 ∧ w = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_prime_area_rectangle_l1230_123088


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l1230_123054

/-- Given a line l symmetric to the line 2x - 3y + 4 = 0 with respect to the line x = 1,
    prove that the equation of line l is 2x + 3y - 8 = 0 -/
theorem symmetric_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (2 - x, y) ∈ {(x, y) | 2*x - 3*y + 4 = 0}) →
  l = {(x, y) | 2*x + 3*y - 8 = 0} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l1230_123054


namespace NUMINAMATH_CALUDE_inequality_solution_l1230_123079

theorem inequality_solution :
  ∀ x : ℝ, (2 < (3 * x) / (4 * x - 7) ∧ (3 * x) / (4 * x - 7) ≤ 9) ↔ 
    (21 / 11 < x ∧ x ≤ 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1230_123079


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1230_123007

theorem quadratic_equation_solution :
  let equation := fun x : ℝ => 2 * x^2 + 6 * x - 1
  let solution1 := -3/2 + Real.sqrt 11 / 2
  let solution2 := -3/2 - Real.sqrt 11 / 2
  equation solution1 = 0 ∧ equation solution2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1230_123007


namespace NUMINAMATH_CALUDE_class_composition_l1230_123034

theorem class_composition (d m : ℕ) : 
  (d : ℚ) / (d + m : ℚ) = 3/5 →
  ((d - 1 : ℚ) / (d + m - 3 : ℚ) = 5/8) →
  d = 21 ∧ m = 14 := by
sorry

end NUMINAMATH_CALUDE_class_composition_l1230_123034


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1230_123086

theorem smallest_solution_of_equation : ∃ x : ℝ, 
  (∀ y : ℝ, y^4 - 26*y^2 + 169 = 0 → x ≤ y) ∧ 
  x^4 - 26*x^2 + 169 = 0 ∧ 
  x = -Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1230_123086


namespace NUMINAMATH_CALUDE_exists_non_zero_sign_function_l1230_123000

/-- Given functions on a blackboard -/
def f₁ (x : ℝ) : ℝ := x + 1
def f₂ (x : ℝ) : ℝ := x^2 + 1
def f₃ (x : ℝ) : ℝ := x^3 + 1
def f₄ (x : ℝ) : ℝ := x^4 + 1

/-- The set of functions that can be constructed from the given functions -/
inductive ConstructibleFunction : (ℝ → ℝ) → Prop
  | base₁ : ConstructibleFunction f₁
  | base₂ : ConstructibleFunction f₂
  | base₃ : ConstructibleFunction f₃
  | base₄ : ConstructibleFunction f₄
  | sub (f g : ℝ → ℝ) : ConstructibleFunction f → ConstructibleFunction g → ConstructibleFunction (λ x => f x - g x)
  | mul (f g : ℝ → ℝ) : ConstructibleFunction f → ConstructibleFunction g → ConstructibleFunction (λ x => f x * g x)

/-- The theorem to be proved -/
theorem exists_non_zero_sign_function :
  ∃ (f : ℝ → ℝ), ConstructibleFunction f ∧ f ≠ 0 ∧
  (∀ x > 0, f x ≥ 0) ∧ (∀ x < 0, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_zero_sign_function_l1230_123000


namespace NUMINAMATH_CALUDE_missing_number_is_sixty_l1230_123009

/-- Given that the average of 20, 40, and 60 is 5 more than the average of 10, x, and 35,
    prove that x = 60. -/
theorem missing_number_is_sixty :
  ∃ x : ℝ, (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 := by
sorry

end NUMINAMATH_CALUDE_missing_number_is_sixty_l1230_123009


namespace NUMINAMATH_CALUDE_second_cook_selection_l1230_123066

theorem second_cook_selection (n : ℕ) (k : ℕ) : n = 9 ∧ k = 1 → Nat.choose n k = 9 := by
  sorry

end NUMINAMATH_CALUDE_second_cook_selection_l1230_123066


namespace NUMINAMATH_CALUDE_sum_of_valid_starting_values_l1230_123044

def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 4 * n + 1

def apply_transform (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | m + 1 => transform (apply_transform n m)

def valid_starting_values : List ℕ :=
  (List.range 100).filter (λ n => apply_transform n 6 = 1)

theorem sum_of_valid_starting_values :
  valid_starting_values.sum = 85 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_starting_values_l1230_123044


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1230_123059

theorem simplify_square_roots : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1230_123059


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l1230_123065

theorem integer_solutions_of_inequalities :
  {x : ℤ | (2 * x - 1 < x + 1) ∧ (1 - 2 * (x - 1) ≤ 3)} = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l1230_123065


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l1230_123046

theorem geometric_sum_problem : 
  let a : ℚ := 1/2
  let r : ℚ := -1/3
  let n : ℕ := 6
  let S := (a * (1 - r^n)) / (1 - r)
  S = 91/243 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l1230_123046


namespace NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l1230_123051

/-- A function that returns true if a number satisfies the given property --/
def satisfies_property (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
  ∃ (a b : ℕ),
    n = 10 * a + b ∧  -- n is represented as 10a + b
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧  -- a and b are single digits
    (n - (a + b) / 2) % 10 = 4  -- the property holds

/-- The theorem stating that exactly two numbers satisfy the property --/
theorem exactly_two_numbers_satisfy :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n, n ∈ s ↔ satisfies_property n :=
sorry

end NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l1230_123051


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1230_123090

/-- Represents a digit in base d -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number AB in base d to its decimal representation -/
def toDecimal (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d 
  (d : ℕ) 
  (h_d : d > 7) 
  (A B : Digit d) 
  (h_sum : toDecimal d A B + toDecimal d A A = 1 * d * d + 7 * d + 2) :
  (A.val - B.val : ℤ) = 4 :=
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1230_123090


namespace NUMINAMATH_CALUDE_average_of_20_and_22_l1230_123027

theorem average_of_20_and_22 : (20 + 22) / 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_20_and_22_l1230_123027


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l1230_123038

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℚ) 
  (num_group1 : Nat) 
  (avg_age_group1 : ℚ) 
  (num_group2 : Nat) 
  (avg_age_group2 : ℚ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  : ℚ :=
  by
    sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l1230_123038


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1230_123041

theorem pure_imaginary_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : ∃ (z : ℂ), z.re = 0 ∧ z = (5 - 9 * Complex.I) * (x + y * Complex.I)) : 
  x / y = -9 / 5 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1230_123041


namespace NUMINAMATH_CALUDE_correct_regression_equation_l1230_123025

/-- Represents the selling price of a product in yuan/piece -/
def selling_price : ℝ → Prop :=
  λ x => x > 0

/-- Represents the sales volume of a product in pieces -/
def sales_volume : ℝ → Prop :=
  λ y => y > 0

/-- Represents a negative correlation between sales volume and selling price -/
def negative_correlation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The regression equation for sales volume based on selling price -/
def regression_equation (x : ℝ) : ℝ :=
  -10 * x + 200

theorem correct_regression_equation :
  (∀ x, selling_price x → sales_volume (regression_equation x)) ∧
  negative_correlation regression_equation :=
sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l1230_123025


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l1230_123075

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 20 →
  offset1 = 6 →
  area = 150 →
  ∃ offset2 : ℝ, 
    area = (diagonal * (offset1 + offset2)) / 2 ∧
    offset2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l1230_123075


namespace NUMINAMATH_CALUDE_garden_flowers_l1230_123077

/-- Represents a rectangular garden with a rose planted in it. -/
structure Garden where
  rows_front : ℕ  -- Number of rows in front of the rose
  rows_back : ℕ   -- Number of rows behind the rose
  cols_right : ℕ  -- Number of columns to the right of the rose
  cols_left : ℕ   -- Number of columns to the left of the rose

/-- Calculates the total number of flowers in the garden. -/
def total_flowers (g : Garden) : ℕ :=
  (g.rows_front + g.rows_back + 1) * (g.cols_right + g.cols_left + 1)

/-- Theorem stating that a garden with the given properties has 462 flowers. -/
theorem garden_flowers :
  ∀ (g : Garden),
    g.rows_front = 6 ∧
    g.rows_back = 15 ∧
    g.cols_right = 12 ∧
    g.cols_left = 8 →
    total_flowers g = 462 := by
  sorry

end NUMINAMATH_CALUDE_garden_flowers_l1230_123077


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l1230_123058

theorem fourth_root_simplification :
  (2^8 * 3^2 * 5^3)^(1/4 : ℝ) = 4 * (1125 : ℝ)^(1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l1230_123058


namespace NUMINAMATH_CALUDE_lakeisha_lawn_mowing_l1230_123062

/-- The amount LaKeisha charges per square foot of lawn -/
def charge_per_sqft : ℚ := 1/10

/-- The cost of the book set -/
def book_cost : ℚ := 150

/-- The length of each lawn -/
def lawn_length : ℕ := 20

/-- The width of each lawn -/
def lawn_width : ℕ := 15

/-- The number of lawns already mowed -/
def lawns_mowed : ℕ := 3

/-- The additional square feet LaKeisha needs to mow -/
def additional_sqft : ℕ := 600

theorem lakeisha_lawn_mowing :
  (lawn_length * lawn_width * lawns_mowed * charge_per_sqft) + 
  (additional_sqft * charge_per_sqft) = book_cost :=
sorry

end NUMINAMATH_CALUDE_lakeisha_lawn_mowing_l1230_123062


namespace NUMINAMATH_CALUDE_linear_equation_exponent_relation_l1230_123021

/-- If 2x^(m-1) + 3y^(2n-1) = 7 is a linear equation in x and y, then m - 2n = 0 -/
theorem linear_equation_exponent_relation (m n : ℕ) :
  (∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(m-1) + 3 * y^(2*n-1) = a * x + b * y + c) →
  m - 2*n = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_relation_l1230_123021


namespace NUMINAMATH_CALUDE_hours_in_year_correct_hours_in_year_l1230_123040

theorem hours_in_year : ℕ → ℕ → ℕ → Prop :=
  fun hours_per_day days_per_year hours_per_year =>
    hours_per_day = 24 ∧ days_per_year = 365 →
    hours_per_year = hours_per_day * days_per_year

theorem correct_hours_in_year : hours_in_year 24 365 8760 := by
  sorry

end NUMINAMATH_CALUDE_hours_in_year_correct_hours_in_year_l1230_123040


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1230_123047

theorem volleyball_lineup_combinations (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 5) : 
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1230_123047


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1230_123031

theorem quadratic_equivalence :
  ∀ x y : ℝ, y = x^2 - 8*x - 1 ↔ y = (x - 4)^2 - 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1230_123031


namespace NUMINAMATH_CALUDE_sin_cos_shift_l1230_123036

theorem sin_cos_shift (x : ℝ) : Real.sin (x/2) = Real.cos ((x-π)/2 - π/4) := by sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l1230_123036


namespace NUMINAMATH_CALUDE_tower_height_difference_l1230_123020

/-- Given the heights of three towers and their relationships, prove the height difference between two of them. -/
theorem tower_height_difference 
  (cn_tower_height : ℝ)
  (cn_space_needle_diff : ℝ)
  (eiffel_tower_height : ℝ)
  (h1 : cn_tower_height = 553)
  (h2 : cn_space_needle_diff = 369)
  (h3 : eiffel_tower_height = 330) :
  eiffel_tower_height - (cn_tower_height - cn_space_needle_diff) = 146 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_difference_l1230_123020


namespace NUMINAMATH_CALUDE_factors_of_polynomial_l1230_123097

theorem factors_of_polynomial (x : ℝ) : 
  (x^4 - 4*x^2 + 4 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2)) ∧ 
  (x^4 - 4*x^2 + 4 ≠ (x - 1) * (x^3 + x^2 + x + 1)) ∧
  (x^4 - 4*x^2 + 4 ≠ (x^2 + 2) * (x^2 - 2)) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_polynomial_l1230_123097


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l1230_123002

theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 6 → choices = 6 → 1 - (1 - 1 / choices) ^ n = 31031 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l1230_123002


namespace NUMINAMATH_CALUDE_total_profit_is_35000_l1230_123017

/-- Represents the business subscription and profit distribution problem --/
structure BusinessProblem where
  total_subscription : ℕ
  a_more_than_b : ℕ
  b_more_than_c : ℕ
  c_profit : ℕ

/-- Calculates the total profit based on the given business problem --/
def calculate_total_profit (problem : BusinessProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the total profit is 35000 --/
theorem total_profit_is_35000 : 
  let problem := BusinessProblem.mk 50000 4000 5000 8400
  calculate_total_profit problem = 35000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_35000_l1230_123017


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l1230_123091

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l1230_123091


namespace NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_remaining_fuel_formula_l1230_123083

/-- Represents the fuel consumption model of a car -/
structure CarFuelModel where
  initial_fuel : ℝ
  consumption_rate : ℝ

/-- Calculates the remaining fuel after a given time -/
def remaining_fuel (model : CarFuelModel) (t : ℝ) : ℝ :=
  model.initial_fuel - model.consumption_rate * t

/-- Theorem stating the remaining fuel after 3 hours for a specific car model -/
theorem remaining_fuel_after_three_hours
  (model : CarFuelModel)
  (h1 : model.initial_fuel = 100)
  (h2 : model.consumption_rate = 6) :
  remaining_fuel model 3 = 82 := by
  sorry

/-- Theorem proving the general formula for remaining fuel -/
theorem remaining_fuel_formula
  (model : CarFuelModel)
  (h1 : model.initial_fuel = 100)
  (h2 : model.consumption_rate = 6)
  (t : ℝ) :
  remaining_fuel model t = 100 - 6 * t := by
  sorry

end NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_remaining_fuel_formula_l1230_123083


namespace NUMINAMATH_CALUDE_xyz_value_l1230_123096

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10)
  (h3 : x + y = 2 * z) :
  x * y * z = 6 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l1230_123096


namespace NUMINAMATH_CALUDE_bound_on_c_l1230_123071

theorem bound_on_c (a b c : ℝ) 
  (sum_condition : a + 2 * b + c = 1) 
  (square_sum_condition : a^2 + b^2 + c^2 = 1) : 
  -2/3 ≤ c ∧ c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bound_on_c_l1230_123071


namespace NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l1230_123043

theorem one_fourth_in_one_eighth : (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l1230_123043


namespace NUMINAMATH_CALUDE_fraction_equality_l1230_123023

theorem fraction_equality (x y : ℝ) (h : x / y = 1 / 2) :
  (x - y) / (x + y) = -1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1230_123023


namespace NUMINAMATH_CALUDE_decimal_sum_l1230_123095

theorem decimal_sum : 0.3 + 0.08 + 0.007 = 0.387 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l1230_123095


namespace NUMINAMATH_CALUDE_remainder_5_2024_mod_17_l1230_123069

theorem remainder_5_2024_mod_17 : 5^2024 % 17 = 16 := by sorry

end NUMINAMATH_CALUDE_remainder_5_2024_mod_17_l1230_123069


namespace NUMINAMATH_CALUDE_corrected_mean_l1230_123073

/-- Given 100 observations with an initial mean of 45, and three incorrect recordings
    (60 as 35, 52 as 25, and 85 as 40), the corrected mean is 45.97. -/
theorem corrected_mean (n : ℕ) (initial_mean : ℝ) 
  (error1 error2 error3 : ℝ) (h1 : n = 100) (h2 : initial_mean = 45)
  (h3 : error1 = 60 - 35) (h4 : error2 = 52 - 25) (h5 : error3 = 85 - 40) :
  let total_error := error1 + error2 + error3
  let initial_sum := n * initial_mean
  let corrected_sum := initial_sum + total_error
  corrected_sum / n = 45.97 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_l1230_123073


namespace NUMINAMATH_CALUDE_patients_ages_problem_l1230_123092

theorem patients_ages_problem : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x - y = 44 ∧ x * y = 1280 ∧ x = 64 ∧ y = 20 := by
  sorry

end NUMINAMATH_CALUDE_patients_ages_problem_l1230_123092


namespace NUMINAMATH_CALUDE_one_thirteenth_150th_digit_l1230_123016

def decimal_representation (n : ℕ) : ℕ := 
  match n % 6 with
  | 1 => 0
  | 2 => 7
  | 3 => 6
  | 4 => 9
  | 5 => 2
  | 0 => 3
  | _ => 0  -- This case should never occur, but Lean requires it for exhaustiveness

theorem one_thirteenth_150th_digit : 
  decimal_representation 150 = 3 := by
sorry


end NUMINAMATH_CALUDE_one_thirteenth_150th_digit_l1230_123016


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1230_123026

theorem geometric_sequence_middle_term (b : ℝ) (h : b > 0) :
  (∃ s : ℝ, s ≠ 0 ∧ 10 * s = b ∧ b * s = 1/3) → b = Real.sqrt (10/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l1230_123026


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1230_123045

theorem trigonometric_equation_solution (x : ℝ) :
  (5.32 * Real.sin (2 * x) * Real.sin (6 * x) * Real.cos (4 * x) + (1/4) * Real.cos (12 * x) = 0) ↔
  (∃ k : ℤ, x = (π / 8) * (2 * k + 1)) ∨
  (∃ k : ℤ, x = (π / 12) * (6 * k + 1) ∨ x = (π / 12) * (6 * k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1230_123045


namespace NUMINAMATH_CALUDE_prob_difference_games_l1230_123074

/-- Probability of getting heads on a single toss of the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails on a single toss of the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game A -/
def p_win_game_a : ℚ := p_heads^4 + p_tails^4

/-- Probability of winning Game C -/
def p_win_game_c : ℚ := p_heads^5 + p_tails^5 + p_heads^3 * p_tails^2 + p_tails^3 * p_heads^2

/-- The difference in probabilities between winning Game A and Game C -/
theorem prob_difference_games : p_win_game_a - p_win_game_c = 3/64 := by sorry

end NUMINAMATH_CALUDE_prob_difference_games_l1230_123074


namespace NUMINAMATH_CALUDE_frank_reading_speed_l1230_123093

-- Define the parameters of the problem
def total_pages : ℕ := 193
def total_chapters : ℕ := 15
def total_days : ℕ := 660

-- Define the function to calculate chapters read per day
def chapters_per_day : ℚ := total_chapters / total_days

-- Theorem statement
theorem frank_reading_speed :
  chapters_per_day = 15 / 660 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l1230_123093


namespace NUMINAMATH_CALUDE_probability_score_difference_not_exceeding_three_l1230_123011

def group_A : List ℕ := [88, 89, 90]
def group_B : List ℕ := [87, 88, 92]

def total_possibilities : ℕ := group_A.length * group_B.length

def favorable_outcomes : ℕ :=
  (group_A.length * group_B.length) - 
  (group_A.filter (λ x => x = 88)).length * 
  (group_B.filter (λ x => x = 92)).length

theorem probability_score_difference_not_exceeding_three :
  (favorable_outcomes : ℚ) / total_possibilities = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_score_difference_not_exceeding_three_l1230_123011


namespace NUMINAMATH_CALUDE_canoe_trip_time_rita_canoe_trip_time_l1230_123049

/-- Calculates the total time for a round trip given upstream and downstream speeds and distance -/
theorem canoe_trip_time (upstream_speed downstream_speed distance : ℝ) :
  upstream_speed > 0 →
  downstream_speed > 0 →
  distance > 0 →
  (distance / upstream_speed) + (distance / downstream_speed) =
    (upstream_speed + downstream_speed) * distance / (upstream_speed * downstream_speed) := by
  sorry

/-- Proves that Rita's canoe trip takes 8 hours -/
theorem rita_canoe_trip_time :
  let upstream_speed : ℝ := 3
  let downstream_speed : ℝ := 9
  let distance : ℝ := 18
  (distance / upstream_speed) + (distance / downstream_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_canoe_trip_time_rita_canoe_trip_time_l1230_123049


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_not_l1230_123081

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |a*x + 1|

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem negation_of_existence_is_universal_not :
  (¬ ∃ a : ℝ, is_even_function (f a)) ↔ (∀ a : ℝ, ¬ is_even_function (f a)) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_not_l1230_123081
