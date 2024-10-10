import Mathlib

namespace select_at_least_one_first_class_l60_6094

theorem select_at_least_one_first_class :
  let total_parts : ℕ := 10
  let first_class_parts : ℕ := 6
  let second_class_parts : ℕ := 4
  let parts_to_select : ℕ := 3
  let total_combinations := Nat.choose total_parts parts_to_select
  let all_second_class := Nat.choose second_class_parts parts_to_select
  total_combinations - all_second_class = 116 := by
  sorry

end select_at_least_one_first_class_l60_6094


namespace quadratic_distinct_roots_l60_6006

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
sorry

end quadratic_distinct_roots_l60_6006


namespace quadratic_inequality_max_value_l60_6039

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    ∀ k : ℝ, (b^2) / (a^2 + c^2) ≤ k → k ≤ M) :=
by sorry

end quadratic_inequality_max_value_l60_6039


namespace equation_solution_l60_6091

-- Define the operation "*"
def star_op (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- State the theorem
theorem equation_solution :
  ∃! x : ℝ, star_op (x - 4) 1 = 0 ∧ x = 5 :=
by
  sorry

end equation_solution_l60_6091


namespace total_eggs_collected_l60_6096

def benjamin_eggs : ℕ := 6

def carla_eggs : ℕ := 3 * benjamin_eggs

def trisha_eggs : ℕ := benjamin_eggs - 4

def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem total_eggs_collected :
  total_eggs = 26 := by sorry

end total_eggs_collected_l60_6096


namespace F_2017_composition_l60_6021

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

def F_comp (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => x
  | n+1 => F (F_comp n x)

theorem F_2017_composition (x : ℝ) : F_comp 2017 x = (x + 1)^(3^2017) - 1 := by
  sorry

end F_2017_composition_l60_6021


namespace triangle_area_theorem_l60_6025

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 * x * 2*x = 64) → x = 8 := by
  sorry

end triangle_area_theorem_l60_6025


namespace total_oranges_equals_147_l60_6078

/-- Represents the number of oranges picked by Mary on Monday -/
def mary_monday_oranges : ℕ := 14

/-- Represents the number of oranges picked by Jason on Monday -/
def jason_monday_oranges : ℕ := 41

/-- Represents the number of oranges picked by Amanda on Monday -/
def amanda_monday_oranges : ℕ := 56

/-- Represents the number of apples picked by Mary on Tuesday -/
def mary_tuesday_apples : ℕ := 22

/-- Represents the number of grapefruits picked by Jason on Tuesday -/
def jason_tuesday_grapefruits : ℕ := 15

/-- Represents the number of oranges picked by Amanda on Tuesday -/
def amanda_tuesday_oranges : ℕ := 36

/-- Represents the number of apples picked by Keith on Monday -/
def keith_monday_apples : ℕ := 38

/-- Represents the number of plums picked by Keith on Tuesday -/
def keith_tuesday_plums : ℕ := 47

/-- The total number of oranges picked over two days -/
def total_oranges : ℕ := mary_monday_oranges + jason_monday_oranges + amanda_monday_oranges + amanda_tuesday_oranges

theorem total_oranges_equals_147 : total_oranges = 147 := by
  sorry

end total_oranges_equals_147_l60_6078


namespace unique_value_in_set_l60_6065

theorem unique_value_in_set (a : ℝ) : 1 ∈ ({a, a + 1, a^2} : Set ℝ) → a = -1 := by
  sorry

end unique_value_in_set_l60_6065


namespace gift_box_weight_l60_6095

/-- The weight of an empty gift box -/
def empty_box_weight (num_tangerines : ℕ) (tangerine_weight : ℝ) (total_weight : ℝ) : ℝ :=
  total_weight - (num_tangerines : ℝ) * tangerine_weight

/-- Theorem: The weight of the empty gift box is 0.46 kg -/
theorem gift_box_weight :
  empty_box_weight 30 0.36 11.26 = 0.46 := by sorry

end gift_box_weight_l60_6095


namespace quadratic_expression_value_l60_6051

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 12) 
  (eq2 : x + 4 * y = 18) : 
  17 * x^2 + 24 * x * y + 17 * y^2 = 532 := by
  sorry

end quadratic_expression_value_l60_6051


namespace square_area_relation_l60_6035

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := a + b
  let area_I := (diagonal_I^2) / 2
  let area_II := 2 * area_I
  area_II = (a + b)^2 := by
sorry

end square_area_relation_l60_6035


namespace f_range_l60_6046

def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem f_range : ∀ x ∈ Set.Icc 0 3, 1 ≤ f x ∧ f x ≤ 5 := by sorry

end f_range_l60_6046


namespace translation_right_3_units_l60_6019

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateRight (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem translation_right_3_units :
  let A : Point := { x := 2, y := -1 }
  let A' : Point := translateRight A 3
  A'.x = 5 ∧ A'.y = -1 := by
sorry

end translation_right_3_units_l60_6019


namespace cyclic_sum_nonnegative_l60_6098

theorem cyclic_sum_nonnegative (a b c k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hk_lower : k ≥ 0) (hk_upper : k < 2) : 
  (a^2 - b*c)/(b^2 + c^2 + k*a^2) + 
  (b^2 - a*c)/(a^2 + c^2 + k*b^2) + 
  (c^2 - a*b)/(a^2 + b^2 + k*c^2) ≥ 0 := by
sorry

end cyclic_sum_nonnegative_l60_6098


namespace average_of_seven_thirteen_and_n_l60_6000

theorem average_of_seven_thirteen_and_n (N : ℝ) (h1 : 15 < N) (h2 : N < 25) :
  (7 + 13 + N) / 3 = 12 := by
  sorry

end average_of_seven_thirteen_and_n_l60_6000


namespace amount_saved_is_30_l60_6092

/-- Calculates the amount saved after clearing debt given income and expenses -/
def amountSavedAfterDebt (monthlyIncome : ℕ) (initialExpense : ℕ) (reducedExpense : ℕ) : ℕ :=
  let initialPeriod := 6
  let reducedPeriod := 4
  let initialDebt := initialPeriod * initialExpense - initialPeriod * monthlyIncome
  let totalIncome := (initialPeriod + reducedPeriod) * monthlyIncome
  let totalExpense := initialPeriod * initialExpense + reducedPeriod * reducedExpense
  totalIncome - (totalExpense + initialDebt)

/-- Theorem: Given the specified income and expenses, the amount saved after clearing debt is 30 -/
theorem amount_saved_is_30 :
  amountSavedAfterDebt 69 70 60 = 30 := by
  sorry

end amount_saved_is_30_l60_6092


namespace tournament_results_count_l60_6033

-- Define the teams
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

-- Define a match result
inductive MatchResult : Type
| Win : Team → MatchResult
| Loss : Team → MatchResult

-- Define a tournament result
structure TournamentResult : Type :=
(saturday1 : MatchResult)  -- E vs F
(saturday2 : MatchResult)  -- G vs H
(sunday1 : MatchResult)    -- 1st vs 2nd
(sunday2 : MatchResult)    -- 3rd vs 4th

def count_tournament_results : ℕ :=
  -- The actual count will be implemented in the proof
  sorry

theorem tournament_results_count :
  count_tournament_results = 16 := by
  sorry

end tournament_results_count_l60_6033


namespace class_2003_ice_cream_picnic_student_ticket_cost_l60_6059

/-- The cost of a student ticket for the Class of 2003 ice cream picnic -/
def student_ticket_cost : ℚ := sorry

/-- The theorem stating the cost of a student ticket for the Class of 2003 ice cream picnic -/
theorem class_2003_ice_cream_picnic_student_ticket_cost :
  let total_tickets : ℕ := 193
  let non_student_ticket_cost : ℚ := 3/2
  let total_revenue : ℚ := 413/2
  let student_tickets : ℕ := 83
  let non_student_tickets : ℕ := total_tickets - student_tickets
  student_ticket_cost * student_tickets + non_student_ticket_cost * non_student_tickets = total_revenue ∧
  student_ticket_cost = 1/2
  := by sorry

end class_2003_ice_cream_picnic_student_ticket_cost_l60_6059


namespace simplify_expression_l60_6071

theorem simplify_expression : (5 * 10^9) / (2 * 10^5 * 5) = 5000 := by sorry

end simplify_expression_l60_6071


namespace square_plus_reciprocal_square_l60_6017

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1 / x) = 1.5) :
  x^2 + (1 / x^2) = 0.25 := by
  sorry

end square_plus_reciprocal_square_l60_6017


namespace coin_flip_probability_difference_l60_6064

-- Define a fair coin
def fair_coin_prob : ℚ := 1 / 2

-- Define the number of flips
def total_flips : ℕ := 5

-- Define the number of heads for the first probability
def heads_count_1 : ℕ := 3

-- Define the number of heads for the second probability
def heads_count_2 : ℕ := 5

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Function to calculate probability of exactly k heads in n flips
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem coin_flip_probability_difference : 
  (prob_k_heads total_flips heads_count_1 fair_coin_prob - 
   prob_k_heads total_flips heads_count_2 fair_coin_prob) = 9 / 32 := by sorry

end coin_flip_probability_difference_l60_6064


namespace lee_test_probability_l60_6047

theorem lee_test_probability (p_physics : ℝ) (p_chem_given_no_physics : ℝ) 
  (h1 : p_physics = 5/8)
  (h2 : p_chem_given_no_physics = 2/3) :
  (1 - p_physics) * p_chem_given_no_physics = 1/4 := by
  sorry

end lee_test_probability_l60_6047


namespace integral_proof_l60_6038

theorem integral_proof (x C : ℝ) : 
  (deriv (λ x => 1 / (2 * (2 * Real.sin x - 3 * Real.cos x)^2) + C)) x = 
  (2 * Real.cos x + 3 * Real.sin x) / (2 * Real.sin x - 3 * Real.cos x)^3 := by
  sorry

end integral_proof_l60_6038


namespace sum_of_consecutive_odd_primes_has_three_factors_l60_6073

/-- Two natural numbers are consecutive primes if they are both prime and there is no prime between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

/-- A natural number is the product of at least three factors greater than 1 if it can be written as the product of three or more natural numbers, each greater than 1. -/
def ProductOfAtLeastThreeFactors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ n = a * b * c

/-- For any two consecutive odd prime numbers, their sum is a product of at least three positive integers greater than 1. -/
theorem sum_of_consecutive_odd_primes_has_three_factors (p q : ℕ) :
  ConsecutivePrimes p q → Odd p → Odd q → ProductOfAtLeastThreeFactors (p + q) := by
  sorry

end sum_of_consecutive_odd_primes_has_three_factors_l60_6073


namespace system_integer_solutions_l60_6049

theorem system_integer_solutions (a b c d : ℤ) 
  (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) : 
  (a * d - b * c = 1) ∨ (a * d - b * c = -1) := by sorry

end system_integer_solutions_l60_6049


namespace descendants_characterization_l60_6028

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The set of descendants of 1 -/
inductive Descendant : ℚ → Prop where
  | base : Descendant 1
  | left (x : ℚ) : Descendant x → Descendant (x + 1)
  | right (x : ℚ) : Descendant x → Descendant (x / (x + 1))

/-- Theorem: All descendants of 1 are of the form F_(n±1) / F_n, where n > 1 -/
theorem descendants_characterization (q : ℚ) :
  Descendant q ↔ ∃ n : ℕ, n > 1 ∧ (q = (fib (n + 1) : ℚ) / fib n ∨ q = (fib (n - 1) : ℚ) / fib n) :=
sorry

end descendants_characterization_l60_6028


namespace point_c_coordinates_l60_6066

/-- Given points A and B in ℝ², and a relationship between vectors AC and CB,
    prove that point C has specific coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (3, 0) → 
  C - A = -2 • (B - C) → 
  C = (4, -3) := by
sorry

end point_c_coordinates_l60_6066


namespace ravi_coins_value_is_350_l60_6079

/-- Calculates the total value of Ravi's coins given the number of nickels --/
def raviCoinsValue (nickels : ℕ) : ℚ :=
  let quarters := nickels + 2
  let dimes := quarters + 4
  let nickelValue : ℚ := 5 / 100
  let quarterValue : ℚ := 25 / 100
  let dimeValue : ℚ := 10 / 100
  nickels * nickelValue + quarters * quarterValue + dimes * dimeValue

/-- Theorem stating that Ravi's coins are worth $3.50 given the conditions --/
theorem ravi_coins_value_is_350 : raviCoinsValue 6 = 7/2 := by
  sorry

end ravi_coins_value_is_350_l60_6079


namespace min_grass_seed_amount_is_75_l60_6041

/-- Represents a bag of grass seed with its weight and price -/
structure GrassSeedBag where
  weight : ℕ
  price : ℚ

/-- Finds the minimum amount of grass seed that can be purchased given the constraints -/
def minGrassSeedAmount (bags : List GrassSeedBag) (maxWeight : ℕ) (exactCost : ℚ) : ℕ :=
  sorry

theorem min_grass_seed_amount_is_75 :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 13.82 },
    { weight := 10, price := 20.43 },
    { weight := 25, price := 32.25 }
  ]
  let maxWeight : ℕ := 80
  let exactCost : ℚ := 98.75

  minGrassSeedAmount bags maxWeight exactCost = 75 := by sorry

end min_grass_seed_amount_is_75_l60_6041


namespace tv_screen_diagonal_l60_6004

theorem tv_screen_diagonal (d : ℝ) : d > 0 → d^2 = 17^2 + 76 → d = Real.sqrt 365 := by
  sorry

end tv_screen_diagonal_l60_6004


namespace calculate_total_earnings_l60_6093

/-- Represents the number of days it takes for a person to complete the job alone. -/
structure WorkRate where
  days : ℕ
  days_pos : days > 0

/-- Represents the daily work rate of a person. -/
def daily_rate (w : WorkRate) : ℚ := 1 / w.days

/-- Calculates the total daily rate when multiple people work together. -/
def total_daily_rate (rates : List ℚ) : ℚ := rates.sum

/-- Represents the earnings of the workers. -/
structure Earnings where
  total : ℚ
  total_pos : total > 0

/-- Main theorem: Given the work rates and b's earnings, prove the total earnings. -/
theorem calculate_total_earnings
  (a b c : WorkRate)
  (h_a : a.days = 6)
  (h_b : b.days = 8)
  (h_c : c.days = 12)
  (b_earnings : ℚ)
  (h_b_earnings : b_earnings = 390)
  : ∃ (e : Earnings), e.total = 1170 := by
  sorry

end calculate_total_earnings_l60_6093


namespace unique_solution_is_four_l60_6087

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem unique_solution_is_four :
  ∃! x : ℝ, 2 * (f x) - 19 = f (x - 4) :=
by
  -- The proof goes here
  sorry

end unique_solution_is_four_l60_6087


namespace base5ToBinary_110_equals_11110_l60_6007

-- Define a function to convert a number from base 5 to decimal
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define a function to convert a decimal number to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else go (m / 2) ((m % 2) :: acc)
  go n []

-- Theorem statement
theorem base5ToBinary_110_equals_11110 :
  decimalToBinary (base5ToDecimal [0, 1, 1]) = [1, 1, 1, 1, 0] := by
  sorry

end base5ToBinary_110_equals_11110_l60_6007


namespace divisibility_in_sequence_l60_6032

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2^(sequence_a n) + 2

theorem divisibility_in_sequence (m n : ℕ) (h : m < n) :
  ∃ k : ℕ, sequence_a n = k * sequence_a m := by
  sorry

end divisibility_in_sequence_l60_6032


namespace farey_sequence_mediant_l60_6061

theorem farey_sequence_mediant (r s : ℕ+) : 
  (6:ℚ)/11 < r/s ∧ r/s < (5:ℚ)/9 ∧ 
  (∀ (r' s' : ℕ+), (6:ℚ)/11 < r'/s' ∧ r'/s' < (5:ℚ)/9 → s ≤ s') →
  s - r = 9 :=
by sorry

end farey_sequence_mediant_l60_6061


namespace al_original_portion_l60_6067

/-- Represents the investment scenario with four participants --/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ
  dave : ℝ

/-- The investment scenario satisfies the given conditions --/
def ValidInvestment (i : Investment) : Prop :=
  i.al + i.betty + i.clare + i.dave = 1200 ∧
  (i.al - 150) + (2 * i.betty) + (2 * i.clare) + (3 * i.dave) = 1800

/-- Theorem stating that Al's original portion was $450 --/
theorem al_original_portion (i : Investment) (h : ValidInvestment i) : i.al = 450 := by
  sorry

#check al_original_portion

end al_original_portion_l60_6067


namespace degree_of_3ab_l60_6030

-- Define a monomial type
def Monomial := List (String × Nat)

-- Define a function to calculate the degree of a monomial
def degree (m : Monomial) : Nat :=
  m.foldl (fun acc (_, power) => acc + power) 0

-- Define our specific monomial 3ab
def monomial_3ab : Monomial := [("a", 1), ("b", 1)]

-- Theorem statement
theorem degree_of_3ab : degree monomial_3ab = 2 := by
  sorry

end degree_of_3ab_l60_6030


namespace a_formula_l60_6036

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) - a n + 2

theorem a_formula (n : ℕ) : a n = n^2 + n := by
  sorry

end a_formula_l60_6036


namespace field_trip_students_field_trip_problem_l60_6070

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) : ℕ :=
  let total_capacity := van_capacity * num_vans
  let num_students := total_capacity - num_adults
  num_students

theorem field_trip_problem : 
  field_trip_students 4 2 6 = 2 := by
  sorry

end field_trip_students_field_trip_problem_l60_6070


namespace factorial_ratio_l60_6012

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio (n : ℕ) (h : n > 0) : 
  factorial n / factorial (n - 1) = n := by
  sorry

end factorial_ratio_l60_6012


namespace hyperbola_equation_l60_6009

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if its asymptote intersects the circle with its foci as diameter
    at the point (2, 1) in the first quadrant, then a = 2 and b = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = (b / a) * x) ∧
    (2^2 + 1^2 = c^2) ∧
    (a^2 + b^2 = c^2)) →
  a = 2 ∧ b = 1 := by sorry

end hyperbola_equation_l60_6009


namespace quadratic_equation_coefficients_l60_6022

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
    (∀ x, 3 * x^2 = 5 * x - 1) →
    (∀ x, a * x^2 + b * x + c = 0) →
    (∀ x, 3 * x^2 - 5 * x + 1 = 0) →
    a = 3 ∧ b = -5 := by
  sorry

end quadratic_equation_coefficients_l60_6022


namespace arithmetic_sequence_sum_l60_6013

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 4 + a 7 + a 11 = 44 →
  a 3 + a 5 + a 10 = 33 := by
sorry

end arithmetic_sequence_sum_l60_6013


namespace largest_of_four_consecutive_odds_l60_6090

theorem largest_of_four_consecutive_odds (x : ℤ) : 
  (x % 2 = 1) →                           -- x is odd
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →  -- average is 24
  (x + 6 = 27) :=                         -- largest number is 27
by sorry

end largest_of_four_consecutive_odds_l60_6090


namespace valid_parameterizations_l60_6089

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 4

-- Define the parameterizations
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)
def param_C (t : ℝ) : ℝ × ℝ := (-1 + 2 * t, -6 + 4 * t)
def param_D (t : ℝ) : ℝ × ℝ := (3 + t, 2 + 3 * t)
def param_E (t : ℝ) : ℝ × ℝ := (-4 - 2 * t, -12 - 4 * t)

-- Theorem stating which parameterizations are valid
theorem valid_parameterizations :
  (∀ t, line_equation (param_A t).1 (param_A t).2) ∧
  (∀ t, line_equation (param_B t).1 (param_B t).2) ∧
  ¬(∀ t, line_equation (param_C t).1 (param_C t).2) ∧
  ¬(∀ t, line_equation (param_D t).1 (param_D t).2) ∧
  ¬(∀ t, line_equation (param_E t).1 (param_E t).2) := by
  sorry

end valid_parameterizations_l60_6089


namespace sum_positive_implies_one_positive_l60_6018

theorem sum_positive_implies_one_positive (a b : ℝ) : a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end sum_positive_implies_one_positive_l60_6018


namespace quadratic_root_relation_l60_6005

theorem quadratic_root_relation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (2 * x₁^2 - (2*m + 1)*x₁ + m^2 - 9*m + 39 = 0) ∧ 
    (2 * x₂^2 - (2*m + 1)*x₂ + m^2 - 9*m + 39 = 0) ∧ 
    (x₂ = 2 * x₁)) → 
  (m = 10 ∨ m = 7) :=
by sorry

end quadratic_root_relation_l60_6005


namespace most_reasonable_sampling_method_l60_6077

/-- Represents the different sampling methods --/
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

/-- Represents the educational stages --/
inductive EducationalStage
| Primary
| JuniorHigh
| SeniorHigh

/-- Represents whether there are significant differences in vision conditions --/
def HasSignificantDifferences : Prop := True

/-- The most reasonable sampling method given the conditions --/
def MostReasonableSamplingMethod : SamplingMethod := SamplingMethod.StratifiedByEducationalStage

theorem most_reasonable_sampling_method
  (h1 : HasSignificantDifferences → ∀ (s1 s2 : EducationalStage), s1 ≠ s2 → ∃ (diff : ℝ), diff > 0)
  (h2 : ¬HasSignificantDifferences → ∀ (gender1 gender2 : Bool), ∀ (ε : ℝ), ε > 0 → ∃ (diff : ℝ), diff < ε)
  : MostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by sorry

end most_reasonable_sampling_method_l60_6077


namespace tan_105_degrees_l60_6099

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l60_6099


namespace fred_seashells_l60_6082

theorem fred_seashells (initial_seashells : ℕ) (given_seashells : ℕ) : 
  initial_seashells = 47 → given_seashells = 25 → initial_seashells - given_seashells = 22 := by
  sorry

end fred_seashells_l60_6082


namespace triangle_arctan_sum_l60_6069

theorem triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_C := 2 * Real.pi / 3
  (a^2 + b^2 + c^2 = 2 * a * b * Real.cos angle_C + 2 * b * c + 2 * c * a) →
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 := by
sorry

end triangle_arctan_sum_l60_6069


namespace initial_pens_l60_6081

def double_weekly (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => 2 * double_weekly initial n

theorem initial_pens (initial : ℕ) :
  double_weekly initial 4 = 32 → initial = 2 := by
  sorry

end initial_pens_l60_6081


namespace divisor_degree_l60_6054

-- Define the degrees of the polynomials
def deg_dividend : ℕ := 15
def deg_quotient : ℕ := 9
def deg_remainder : ℕ := 4

-- Theorem statement
theorem divisor_degree :
  ∀ (deg_divisor : ℕ),
    deg_dividend = deg_divisor + deg_quotient ∧
    deg_remainder < deg_divisor →
    deg_divisor = 6 := by
  sorry

end divisor_degree_l60_6054


namespace kittens_and_mice_count_l60_6088

/-- The number of children carrying baskets -/
def num_children : ℕ := 12

/-- The number of baskets each child carries -/
def baskets_per_child : ℕ := 3

/-- The number of cats in each basket -/
def cats_per_basket : ℕ := 1

/-- The number of kittens each cat has -/
def kittens_per_cat : ℕ := 12

/-- The number of mice each kitten carries -/
def mice_per_kitten : ℕ := 4

/-- The total number of kittens and mice carried by the children -/
def total_kittens_and_mice : ℕ :=
  let total_baskets := num_children * baskets_per_child
  let total_cats := total_baskets * cats_per_basket
  let total_kittens := total_cats * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice

theorem kittens_and_mice_count : total_kittens_and_mice = 2160 := by
  sorry

end kittens_and_mice_count_l60_6088


namespace cricket_run_rate_theorem_l60_6014

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_part_overs = 10)
  (h3 : game.first_part_run_rate = 6.2)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 5.5 := by
  sorry

#eval required_run_rate { 
  total_overs := 50, 
  first_part_overs := 10, 
  first_part_run_rate := 6.2, 
  target_runs := 282 
}

end cricket_run_rate_theorem_l60_6014


namespace perpendicular_parallel_relationships_l60_6050

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_relationships 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β) :
  (parallel α β → line_perpendicular l m) ∧
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end perpendicular_parallel_relationships_l60_6050


namespace necessary_not_sufficient_condition_l60_6040

theorem necessary_not_sufficient_condition (x : ℝ) :
  (∀ y : ℝ, y > 2 → y > 1) ∧ (∃ z : ℝ, z > 1 ∧ z ≤ 2) := by
  sorry

end necessary_not_sufficient_condition_l60_6040


namespace junior_freshman_ratio_l60_6048

theorem junior_freshman_ratio (f j : ℕ) (hf : f > 0) (hj : j > 0)
  (h_participants : (1 : ℚ) / 4 * f = (1 : ℚ) / 2 * j) :
  j / f = (1 : ℚ) / 2 := by
sorry

end junior_freshman_ratio_l60_6048


namespace adam_savings_l60_6034

theorem adam_savings (x : ℝ) : x + 13 = 92 → x = 79 := by
  sorry

end adam_savings_l60_6034


namespace CD_parallel_BE_l60_6068

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define points C and D
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)

-- Define a line passing through C and a point (x, y) on Γ
def line_through_C (x y : ℝ) : Set (ℝ × ℝ) :=
  {(t, u) | ∃ (k : ℝ), u - C.2 = k * (t - C.1) ∧ t ≠ C.1}

-- Define point A as an intersection of the line and Γ
def A (x y : ℝ) : ℝ × ℝ :=
  (x, y)

-- Define point B as the other intersection of the line and Γ
def B (x y : ℝ) : ℝ × ℝ :=
  sorry

-- Define point E as the intersection of AD and x=3
def E (x y : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem CD_parallel_BE (x y : ℝ) :
  Γ x y →
  (x, y) ∈ line_through_C x y →
  (B x y).1 ≠ x →
  (let slope_CD := (D.2 - C.2) / (D.1 - C.1)
   let slope_BE := (E x y).2 / ((E x y).1 - (B x y).1)
   slope_CD = slope_BE) :=
sorry

end CD_parallel_BE_l60_6068


namespace comic_book_problem_l60_6045

theorem comic_book_problem (x y : ℕ) : 
  (y + 7 = 5 * (x - 7)) →
  (y - 9 = 3 * (x + 9)) →
  (x = 39 ∧ y = 153) := by
sorry

end comic_book_problem_l60_6045


namespace solution_existence_l60_6008

theorem solution_existence (x y : ℝ) : 
  |x + 1| + (y - 8)^2 = 0 → x = -1 ∧ y = 8 := by
  sorry

end solution_existence_l60_6008


namespace fraction_simplification_l60_6085

theorem fraction_simplification (x : ℝ) (h : x ≠ 0) :
  ((x + 3)^2 + (x + 3)*(x - 3)) / (2*x) = x + 3 := by
  sorry

end fraction_simplification_l60_6085


namespace three_percentage_problem_l60_6026

theorem three_percentage_problem (x y : ℝ) 
  (h1 : 3 = 0.25 * x) 
  (h2 : 3 = 0.50 * y) : 
  x - y = 6 ∧ x + y = 18 := by
  sorry

end three_percentage_problem_l60_6026


namespace cyclist_club_members_count_l60_6031

/-- The set of digits that can be used in the identification numbers. -/
def ValidDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 9}

/-- The number of digits in each identification number. -/
def IdentificationNumberLength : Nat := 3

/-- The total number of possible identification numbers. -/
def TotalIdentificationNumbers : Nat := ValidDigits.card ^ IdentificationNumberLength

/-- Theorem stating that the total number of possible identification numbers is 512. -/
theorem cyclist_club_members_count :
  TotalIdentificationNumbers = 512 := by sorry

end cyclist_club_members_count_l60_6031


namespace multiple_factor_statement_l60_6011

theorem multiple_factor_statement (h : 8 * 9 = 72) : ¬(∃ k : ℕ, 72 = 8 * k ∧ ∃ m : ℕ, 72 = m * 8) :=
sorry

end multiple_factor_statement_l60_6011


namespace complex_multiplication_l60_6062

theorem complex_multiplication : ∃ (i : ℂ), i^2 = -1 ∧ (3 - 4*i) * (-7 + 2*i) = -13 + 34*i :=
by sorry

end complex_multiplication_l60_6062


namespace system_solution_proof_l60_6003

theorem system_solution_proof :
  ∃ (x y z : ℝ),
    x = 48 ∧ y = 16 ∧ z = 12 ∧
    (x * y) / (5 * x + 4 * y) = 6 ∧
    (x * z) / (3 * x + 2 * z) = 8 ∧
    (y * z) / (3 * y + 5 * z) = 6 := by
  sorry

end system_solution_proof_l60_6003


namespace yoghurt_cost_l60_6042

/-- Given Tara's purchase of ice cream and yoghurt, prove the cost of each yoghurt carton. -/
theorem yoghurt_cost (ice_cream_cartons : ℕ) (yoghurt_cartons : ℕ) 
  (ice_cream_cost : ℕ) (price_difference : ℕ) :
  ice_cream_cartons = 19 →
  yoghurt_cartons = 4 →
  ice_cream_cost = 7 →
  price_difference = 129 →
  (ice_cream_cartons * ice_cream_cost - price_difference) / yoghurt_cartons = 1 := by
  sorry

end yoghurt_cost_l60_6042


namespace unique_solution_factorial_equation_l60_6055

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem unique_solution_factorial_equation :
  ∃! n : ℕ, 3 * n * factorial n + 2 * factorial n = 40320 :=
sorry

end unique_solution_factorial_equation_l60_6055


namespace hexagon_circle_comparison_l60_6027

theorem hexagon_circle_comparison : ∃ (h r : ℝ),
  h > 0 ∧ r > 0 ∧
  (3 * Real.sqrt 3 / 2) * h^2 = 6 * h ∧  -- Hexagon area equals perimeter
  π * r^2 = 2 * π * r ∧                  -- Circle area equals perimeter
  (Real.sqrt 3 / 2) * h = r ∧            -- Apothem equals radius
  r = 2 := by sorry

end hexagon_circle_comparison_l60_6027


namespace train_length_l60_6044

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 225 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end train_length_l60_6044


namespace distance_sum_equals_3root2_l60_6052

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = -t ∧ y = 1 + t

-- Define point P in Cartesian coordinates
def point_P : ℝ × ℝ := (0, 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ A.1 A.2 ∧ circle_C A.1 A.2 ∧
    line_l t₂ B.1 B.2 ∧ circle_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem distance_sum_equals_3root2 (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 3 * Real.sqrt 2 :=
by sorry

end distance_sum_equals_3root2_l60_6052


namespace quadratic_equation_root_l60_6023

theorem quadratic_equation_root (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - 65 = 0) ∧ (a * 5^2 + 3 * 5 - 65 = 0) → a = 2 := by
  sorry

end quadratic_equation_root_l60_6023


namespace invalid_votes_percentage_l60_6060

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_valid_votes : ℕ)
  (h_total : total_votes = 560000)
  (h_percentage : candidate_a_percentage = 70 / 100)
  (h_valid_votes : candidate_a_valid_votes = 333200) :
  (total_votes - (candidate_a_valid_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
by sorry

end invalid_votes_percentage_l60_6060


namespace arithmetic_expression_equals_eighteen_l60_6056

theorem arithmetic_expression_equals_eighteen :
  8 / 2 - 3 - 10 + 3 * 9 = 18 := by
  sorry

end arithmetic_expression_equals_eighteen_l60_6056


namespace max_value_at_e_l60_6029

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_at_e :
  ∀ x : ℝ, x > 0 → f x ≤ f (Real.exp 1) :=
by sorry

end max_value_at_e_l60_6029


namespace regular_polygon_sides_l60_6086

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) / n = 156 → n = 15 := by
  sorry

end regular_polygon_sides_l60_6086


namespace omino_tilings_2_by_10_l60_6097

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Number of omino tilings for a 1-by-n rectangle -/
def ominoTilings1ByN (n : ℕ) : ℕ := fib (n + 1)

/-- Number of omino tilings for a 2-by-n rectangle -/
def ominoTilings2ByN (n : ℕ) : ℕ := (ominoTilings1ByN n) ^ 2

theorem omino_tilings_2_by_10 : ominoTilings2ByN 10 = 3025 := by
  sorry

end omino_tilings_2_by_10_l60_6097


namespace sqrt_inequality_l60_6002

theorem sqrt_inequality (p q x₁ x₂ : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1) :
  p * Real.sqrt x₁ + q * Real.sqrt x₂ ≤ Real.sqrt (p * x₁ + q * x₂) := by
  sorry

end sqrt_inequality_l60_6002


namespace john_streaming_hours_l60_6020

/-- Calculates the number of hours streamed per day given the total weekly earnings,
    hourly rate, and number of streaming days per week. -/
def hours_streamed_per_day (weekly_earnings : ℕ) (hourly_rate : ℕ) (streaming_days : ℕ) : ℚ :=
  (weekly_earnings : ℚ) / (hourly_rate : ℚ) / (streaming_days : ℚ)

/-- Proves that John streams 4 hours per day given the problem conditions. -/
theorem john_streaming_hours :
  let weekly_earnings := 160
  let hourly_rate := 10
  let days_per_week := 7
  let days_off := 3
  let streaming_days := days_per_week - days_off
  hours_streamed_per_day weekly_earnings hourly_rate streaming_days = 4 := by
  sorry

#eval hours_streamed_per_day 160 10 4

end john_streaming_hours_l60_6020


namespace geometric_mean_scaling_l60_6076

theorem geometric_mean_scaling (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) 
  (h₅ : a₅ > 0) (h₆ : a₆ > 0) (h₇ : a₇ > 0) (h₈ : a₈ > 0) :
  (((5 * a₁) * (5 * a₂) * (5 * a₃) * (5 * a₄) * (5 * a₅) * (5 * a₆) * (5 * a₇) * (5 * a₈)) ^ (1/8 : ℝ)) = 
  5 * ((a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ * a₈) ^ (1/8 : ℝ)) :=
by sorry

end geometric_mean_scaling_l60_6076


namespace correct_guesser_is_D_l60_6057

-- Define the set of suspects
inductive Suspect : Type
| A | B | C | D | E | F

-- Define the set of passersby
inductive Passerby : Type
| A | B | C | D

-- Define a function to represent each passerby's guess
def guess (p : Passerby) (s : Suspect) : Prop :=
  match p with
  | Passerby.A => s = Suspect.D ∨ s = Suspect.E
  | Passerby.B => s ≠ Suspect.C
  | Passerby.C => s = Suspect.A ∨ s = Suspect.B ∨ s = Suspect.F
  | Passerby.D => s ≠ Suspect.D ∧ s ≠ Suspect.E ∧ s ≠ Suspect.F

-- Theorem statement
theorem correct_guesser_is_D :
  ∃! (thief : Suspect),
    ∃! (correct_passerby : Passerby),
      (∀ (p : Passerby), p ≠ correct_passerby → ¬guess p thief) ∧
      guess correct_passerby thief ∧
      correct_passerby = Passerby.D :=
by sorry

end correct_guesser_is_D_l60_6057


namespace factorization_proof_l60_6015

theorem factorization_proof (x : ℝ) : 72 * x^2 + 108 * x + 36 = 36 * (2 * x + 1) * (x + 1) := by
  sorry

end factorization_proof_l60_6015


namespace x_equals_four_l60_6063

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  fun (a, b) (c, d) => (a - c, b + d)

/-- Theorem stating that x = 4 given the conditions -/
theorem x_equals_four :
  ∀ y : ℤ, star (4, 1) (1, -2) = star (x, y) (1, 4) → x = 4 :=
by
  sorry

#check x_equals_four

end x_equals_four_l60_6063


namespace dancing_preference_fraction_l60_6083

def total_students : ℕ := 200
def like_dancing_percent : ℚ := 70 / 100
def dislike_dancing_percent : ℚ := 30 / 100
def honest_like_percent : ℚ := 85 / 100
def dishonest_like_percent : ℚ := 15 / 100
def honest_dislike_percent : ℚ := 80 / 100
def dishonest_dislike_percent : ℚ := 20 / 100

theorem dancing_preference_fraction :
  let like_dancing := (like_dancing_percent * total_students : ℚ)
  let dislike_dancing := (dislike_dancing_percent * total_students : ℚ)
  let say_like := (honest_like_percent * like_dancing + dishonest_dislike_percent * dislike_dancing : ℚ)
  let actually_dislike_say_like := (dishonest_dislike_percent * dislike_dancing : ℚ)
  actually_dislike_say_like / say_like = 12 / 131 := by
  sorry

end dancing_preference_fraction_l60_6083


namespace gcd_50404_40303_l60_6024

theorem gcd_50404_40303 : Nat.gcd 50404 40303 = 1 := by
  sorry

end gcd_50404_40303_l60_6024


namespace original_number_proof_l60_6080

theorem original_number_proof : 
  ∃ x : ℝ, 3 * (2 * (3 * x) - 9) = 90 ∧ x = 6.5 := by sorry

end original_number_proof_l60_6080


namespace product_xyz_is_one_l60_6001

theorem product_xyz_is_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (hy_nonzero : y ≠ 0) 
  (hz_nonzero : z ≠ 0) : 
  x * y * z = 1 := by
sorry

end product_xyz_is_one_l60_6001


namespace crystal_mass_problem_l60_6016

theorem crystal_mass_problem (a b : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (0.04 * a / 4 = 0.05 * b / 3) → 
  ((a + 20) / (b + 20) = 1.5) → 
  (a = 100 ∧ b = 60) := by
  sorry

end crystal_mass_problem_l60_6016


namespace diamond_equation_l60_6084

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_equation : 
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 := by
  sorry

end diamond_equation_l60_6084


namespace money_redistribution_l60_6072

-- Define the initial amounts for each person
variable (a b c d : ℝ)

-- Define the redistribution function
def redistribute (x y z w : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (x - (y + z + w), 2*y, 2*z, 2*w)

-- Theorem statement
theorem money_redistribution (h1 : c = 24) :
  let (a', b', c', d') := redistribute a b c d
  let (a'', b'', c'', d'') := redistribute b' a' c' d'
  let (a''', b''', c''', d''') := redistribute c'' a'' b'' d''
  let (a_final, b_final, c_final, d_final) := redistribute d''' a''' b''' c'''
  c_final = c → a + b + c + d = 96 := by
  sorry

end money_redistribution_l60_6072


namespace f_derivative_l60_6074

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def f (x : ℝ) : ℝ :=
  binomial 4 0 - binomial 4 1 * x + binomial 4 2 * x^2 - binomial 4 3 * x^3 + binomial 4 4 * x^4

theorem f_derivative (x : ℝ) : deriv f x = 4 * (-1 + x)^3 := by
  sorry

end f_derivative_l60_6074


namespace intersecting_line_equation_l60_6043

/-- A line that intersects a circle and a hyperbola with specific properties -/
structure IntersectingLine (a : ℝ) where
  m : ℝ
  b : ℝ
  intersects_circle : ∀ x y, y = m * x + b → x^2 + y^2 = a^2
  intersects_hyperbola : ∀ x y, y = m * x + b → x^2 - y^2 = a^2
  trisects : ∀ (x₁ x₂ x₃ x₄ : ℝ),
    (x₁^2 + (m * x₁ + b)^2 = a^2) →
    (x₂^2 + (m * x₂ + b)^2 = a^2) →
    (x₃^2 - (m * x₃ + b)^2 = a^2) →
    (x₄^2 - (m * x₄ + b)^2 = a^2) →
    (x₁ - x₂)^2 = (1/9) * (x₃ - x₄)^2

/-- The equation of the intersecting line is y = ±(2√5/5)x or y = ±(2√5/5)a -/
theorem intersecting_line_equation (a : ℝ) (l : IntersectingLine a) :
  (l.m = 2 * Real.sqrt 5 / 5 ∧ l.b = 0) ∨
  (l.m = -2 * Real.sqrt 5 / 5 ∧ l.b = 0) ∨
  (l.m = 0 ∧ l.b = 2 * a * Real.sqrt 5 / 5) ∨
  (l.m = 0 ∧ l.b = -2 * a * Real.sqrt 5 / 5) :=
sorry

end intersecting_line_equation_l60_6043


namespace quotient_derivative_property_l60_6037

/-- Two differentiable functions satisfying the property that the derivative of their quotient
    is equal to the quotient of their derivatives. -/
theorem quotient_derivative_property (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.exp (4 * x)
  let g : ℝ → ℝ := fun x => Real.exp (2 * x)
  (deriv (f / g)) x = (deriv f x) / (deriv g x) := by sorry

end quotient_derivative_property_l60_6037


namespace fixed_point_exponential_function_l60_6010

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end fixed_point_exponential_function_l60_6010


namespace largest_inscribed_equilateral_triangle_area_l60_6058

theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
  let circle_radius : ℝ := r
  let triangle_side_length : ℝ := r * Real.sqrt 3
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side_length ^ 2
  triangle_area = 75 * Real.sqrt 3 := by
  sorry

end largest_inscribed_equilateral_triangle_area_l60_6058


namespace simple_interest_problem_l60_6075

/-- Given a sum P put at simple interest rate R% for 1 year, 
    if increasing the rate by 6% results in Rs. 30 more interest, 
    then P = 500. -/
theorem simple_interest_problem (P R : ℝ) 
  (h1 : P * (R + 6) / 100 = P * R / 100 + 30) : 
  P = 500 := by
  sorry

end simple_interest_problem_l60_6075


namespace arithmetic_sum_equals_twenty_l60_6053

/-- The sum of an arithmetic sequence with 5 terms, where the first term is 2 and the last term is 6 -/
def arithmetic_sum : ℕ :=
  let n := 5  -- number of days
  let a₁ := 2 -- first day's distance
  let aₙ := 6 -- last day's distance
  n * (a₁ + aₙ) / 2

/-- The theorem states that the arithmetic sum defined above equals 20 -/
theorem arithmetic_sum_equals_twenty : arithmetic_sum = 20 := by
  sorry

end arithmetic_sum_equals_twenty_l60_6053
