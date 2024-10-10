import Mathlib

namespace committee_vote_change_l2028_202891

theorem committee_vote_change (total : ℕ) (a b a' b' : ℕ) : 
  total = 300 →
  a + b = total →
  b > a →
  a' + b' = total →
  a' - b' = 3 * (b - a) →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by sorry

end committee_vote_change_l2028_202891


namespace midpoint_coordinate_sum_l2028_202859

/-- Given that N(5, -1) is the midpoint of segment CD and C has coordinates (11, 10),
    prove that the sum of the coordinates of point D is -13. -/
theorem midpoint_coordinate_sum (N C D : ℝ × ℝ) : 
  N = (5, -1) →
  C = (11, 10) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = -13 := by
  sorry

end midpoint_coordinate_sum_l2028_202859


namespace intersection_empty_implies_a_geq_5_not_p_sufficient_not_necessary_implies_a_leq_2_l2028_202881

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem for part (1)
theorem intersection_empty_implies_a_geq_5 (a : ℝ) :
  (a > 0) → (A ∩ B a = ∅) → a ≥ 5 := by sorry

-- Theorem for part (2)
theorem not_p_sufficient_not_necessary_implies_a_leq_2 (a : ℝ) :
  (a > 0) → (∀ x, ¬(p x) → q a x) → (∃ x, q a x ∧ p x) → a ≤ 2 := by sorry

end intersection_empty_implies_a_geq_5_not_p_sufficient_not_necessary_implies_a_leq_2_l2028_202881


namespace starting_to_running_current_ratio_l2028_202870

/-- Proves the ratio of starting current to running current for machinery units -/
theorem starting_to_running_current_ratio
  (num_units : ℕ)
  (running_current : ℝ)
  (min_transformer_load : ℝ)
  (h1 : num_units = 3)
  (h2 : running_current = 40)
  (h3 : min_transformer_load = 240)
  : min_transformer_load / (num_units * running_current) = 2 := by
  sorry

#check starting_to_running_current_ratio

end starting_to_running_current_ratio_l2028_202870


namespace polynomial_equality_l2028_202807

theorem polynomial_equality (t s : ℚ) : 
  (∀ x : ℚ, (3*x^2 - 4*x + 9) * (5*x^2 + t*x + 12) = 15*x^4 + s*x^3 + 33*x^2 + 12*x + 108) 
  ↔ 
  (t = 37/5 ∧ s = 11/5) :=
by sorry

end polynomial_equality_l2028_202807


namespace total_miles_traveled_l2028_202866

theorem total_miles_traveled (initial_reading additional_distance : Real) 
  (h1 : initial_reading = 212.3)
  (h2 : additional_distance = 372.0) : 
  initial_reading + additional_distance = 584.3 := by
sorry

end total_miles_traveled_l2028_202866


namespace B_completes_work_in_8_days_l2028_202830

/-- The number of days B takes to complete the work alone -/
def B : ℕ := 8

/-- The rate at which A completes the work -/
def rate_A : ℚ := 1 / 20

/-- The rate at which B completes the work -/
def rate_B : ℚ := 1 / B

/-- The amount of work completed by A and B together in 3 days -/
def work_together : ℚ := 3 * (rate_A + rate_B)

/-- The amount of work completed by B alone in 3 days -/
def work_B_alone : ℚ := 3 * rate_B

theorem B_completes_work_in_8_days :
  work_together + work_B_alone = 1 ∧ B = 8 := by
  sorry

end B_completes_work_in_8_days_l2028_202830


namespace joan_remaining_kittens_l2028_202873

def initial_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_remaining_kittens :
  initial_kittens - kittens_given_away = 6 := by
  sorry

end joan_remaining_kittens_l2028_202873


namespace standard_pairs_parity_l2028_202817

/-- Represents a color on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard -/
structure Chessboard (m n : ℕ) where
  colors : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of blue squares on the boundary (excluding corners) -/
def countBlueBoundary (board : Chessboard m n) : ℕ :=
  sorry

/-- Counts the number of standard pairs on the chessboard -/
def countStandardPairs (board : Chessboard m n) : ℕ :=
  sorry

/-- Theorem stating that the parity of standard pairs is determined by the parity of blue boundary squares -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Even (countStandardPairs board) ↔ Even (countBlueBoundary board) :=
sorry

end standard_pairs_parity_l2028_202817


namespace real_part_of_w_cubed_l2028_202808

theorem real_part_of_w_cubed (w : ℂ) 
  (h1 : w.im > 0)
  (h2 : Complex.abs w = 5)
  (h3 : (w^2 - w) • (w^3 - w) = 0) :
  (w^3).re = -73 := by
  sorry

end real_part_of_w_cubed_l2028_202808


namespace exists_number_divisible_by_2n_with_only_1_and_2_l2028_202871

/-- A function that checks if a natural number only contains digits 1 and 2 in its decimal representation -/
def onlyOneAndTwo (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- For every natural number n, there exists a number divisible by 2^n 
    whose decimal representation uses only the digits 1 and 2 -/
theorem exists_number_divisible_by_2n_with_only_1_and_2 :
  ∀ n : ℕ, ∃ N : ℕ, 2^n ∣ N ∧ onlyOneAndTwo N :=
by sorry

end exists_number_divisible_by_2n_with_only_1_and_2_l2028_202871


namespace angle_measure_proof_l2028_202892

theorem angle_measure_proof (x : ℝ) : x + (4 * x + 5) = 90 → x = 17 := by
  sorry

end angle_measure_proof_l2028_202892


namespace cubic_root_sum_l2028_202835

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 9*p - 3 = 0 →
  q^3 - 8*q^2 + 9*q - 3 = 0 →
  r^3 - 8*r^2 + 9*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 83/43 := by
sorry

end cubic_root_sum_l2028_202835


namespace cubic_is_closed_log_not_closed_sqrt_closed_condition_l2028_202899

-- Define a closed function
def is_closed_function (f : ℝ → ℝ) : Prop :=
  (∃ (a b : ℝ), a < b ∧ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)) ∧
    (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y))

-- Theorem for the cubic function
theorem cubic_is_closed : is_closed_function (fun x => -x^3) :=
sorry

-- Theorem for the logarithmic function
theorem log_not_closed : ¬ is_closed_function (fun x => 2*x - Real.log x) :=
sorry

-- Theorem for the square root function
theorem sqrt_closed_condition (k : ℝ) : 
  is_closed_function (fun x => k + Real.sqrt (x + 2)) ↔ -9/4 < k ∧ k ≤ -2 :=
sorry

end cubic_is_closed_log_not_closed_sqrt_closed_condition_l2028_202899


namespace lawn_care_time_l2028_202867

/-- The time it takes Max to mow the lawn, in minutes -/
def mow_time : ℕ := 40

/-- The time it takes Max to fertilize the lawn, in minutes -/
def fertilize_time : ℕ := 2 * mow_time

/-- The total time it takes Max to both mow and fertilize the lawn, in minutes -/
def total_time : ℕ := mow_time + fertilize_time

theorem lawn_care_time : total_time = 120 := by
  sorry

end lawn_care_time_l2028_202867


namespace orange_ring_weight_l2028_202898

/-- The weight of the orange ring in an experiment -/
theorem orange_ring_weight (purple_weight white_weight total_weight : ℚ)
  (h1 : purple_weight = 33/100)
  (h2 : white_weight = 21/50)
  (h3 : total_weight = 83/100) :
  total_weight - (purple_weight + white_weight) = 2/25 := by
  sorry

#eval (83/100 : ℚ) - ((33/100 : ℚ) + (21/50 : ℚ))

end orange_ring_weight_l2028_202898


namespace total_value_is_20_31_l2028_202837

/-- Represents the value of coins in U.S. Dollars -/
def total_value : ℝ :=
  let us_quarter_value : ℝ := 0.25
  let us_nickel_value : ℝ := 0.05
  let canadian_dime_value : ℝ := 0.10
  let euro_cent_value : ℝ := 0.01
  let british_pence_value : ℝ := 0.01
  let cad_to_usd : ℝ := 0.8
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.4
  let us_quarters : ℝ := 4 * 10 * us_quarter_value
  let us_nickels : ℝ := 9 * 10 * us_nickel_value
  let canadian_dimes : ℝ := 6 * 10 * canadian_dime_value * cad_to_usd
  let euro_cents : ℝ := 5 * 10 * euro_cent_value * eur_to_usd
  let british_pence : ℝ := 3 * 10 * british_pence_value * gbp_to_usd
  us_quarters + us_nickels + canadian_dimes + euro_cents + british_pence

/-- Theorem stating that the total value of Rocco's coins is $20.31 -/
theorem total_value_is_20_31 : total_value = 20.31 := by
  sorry

end total_value_is_20_31_l2028_202837


namespace sphere_radius_ratio_l2028_202841

/-- The maximum ratio of the radius of the third sphere to the radius of the first sphere
    in a specific geometric configuration. -/
theorem sphere_radius_ratio (r x : ℝ) (h1 : r > 0) (h2 : x > 0) : 
  let R := 3 * r
  let t := x / r
  let α := π / 3
  let cone_height := R / 2
  let slant_height := R
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 3 → 
    2 * Real.cos θ ≤ (3 - 2*t) / Real.sqrt (t^2 + 2*t)) →
  (3 * t^2 - 14 * t + 9 = 0) →
  t ≤ 3 / 2 →
  t = (7 - Real.sqrt 22) / 3 := by
sorry

end sphere_radius_ratio_l2028_202841


namespace translation_point_difference_l2028_202823

/-- Given points A and B, their translations A₁ and B₁, prove that a - b = -8 -/
theorem translation_point_difference (A B A₁ B₁ : ℝ × ℝ) (a b : ℝ) 
  (h1 : A = (1, -3))
  (h2 : B = (2, 1))
  (h3 : A₁ = (a, 2))
  (h4 : B₁ = (-1, b))
  (h5 : ∃ (v : ℝ × ℝ), A₁ = A + v ∧ B₁ = B + v) :
  a - b = -8 := by
  sorry

end translation_point_difference_l2028_202823


namespace rowing_conference_votes_l2028_202879

theorem rowing_conference_votes 
  (num_coaches : ℕ) 
  (num_rowers : ℕ) 
  (votes_per_coach : ℕ) 
  (h1 : num_coaches = 36) 
  (h2 : num_rowers = 60) 
  (h3 : votes_per_coach = 5) : 
  (num_coaches * votes_per_coach) / num_rowers = 3 :=
by sorry

end rowing_conference_votes_l2028_202879


namespace frog_climb_proof_l2028_202811

def well_depth : ℝ := 4

def climb_distances : List ℝ := [1.2, 1.4, 1.1, 1.2]
def slide_distances : List ℝ := [0.4, 0.5, 0.3, 0.2]

def net_distance_climbed : ℝ := 
  List.sum (List.zipWith (·-·) climb_distances slide_distances)

def total_distance_covered : ℝ := 
  List.sum climb_distances + List.sum slide_distances

def fifth_climb_distance : ℝ := 1.2

theorem frog_climb_proof :
  (well_depth - net_distance_climbed = 0.5) ∧
  (total_distance_covered = 6.3) ∧
  (net_distance_climbed + fifth_climb_distance > well_depth) := by
  sorry

end frog_climb_proof_l2028_202811


namespace intersection_sum_l2028_202839

theorem intersection_sum (c d : ℝ) :
  (∀ x y : ℝ, x = (1/3) * y + c ↔ y = (1/3) * x + d) →
  3 = (1/3) * 0 + c →
  0 = (1/3) * 3 + d →
  c + d = 2 := by
sorry

end intersection_sum_l2028_202839


namespace num_possible_lists_eq_50625_l2028_202878

/-- The number of balls in the bin -/
def num_balls : ℕ := 15

/-- The number of draws -/
def num_draws : ℕ := 4

/-- The number of possible lists when drawing 'num_draws' times from 'num_balls' with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem stating that the number of possible lists is 50625 -/
theorem num_possible_lists_eq_50625 : num_possible_lists = 50625 := by
  sorry

end num_possible_lists_eq_50625_l2028_202878


namespace total_cost_with_tax_l2028_202864

def earbuds_cost : ℝ := 200
def smartwatch_cost : ℝ := 300
def earbuds_tax_rate : ℝ := 0.15
def smartwatch_tax_rate : ℝ := 0.12

theorem total_cost_with_tax : 
  earbuds_cost * (1 + earbuds_tax_rate) + smartwatch_cost * (1 + smartwatch_tax_rate) = 566 := by
  sorry

end total_cost_with_tax_l2028_202864


namespace arithmetic_sequence_divisibility_l2028_202800

theorem arithmetic_sequence_divisibility (a : ℕ) :
  ∃! k : Fin 7, ∃ n : ℕ, n = a + k * 30 ∧ n % 7 = 0 :=
sorry

end arithmetic_sequence_divisibility_l2028_202800


namespace arbitrarily_large_N_exists_l2028_202803

def is_increasing_seq (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, x n < x (n + 1)

def limit_zero (x : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n / n| < ε

theorem arbitrarily_large_N_exists (x : ℕ → ℝ) 
  (h_pos : ∀ n, x n > 0)
  (h_inc : is_increasing_seq x)
  (h_lim : limit_zero x) :
  ∀ M : ℕ, ∃ N > M, ∀ i : ℕ, 1 ≤ i → i < N → x i + x (2*N - i) < 2 * x N :=
sorry

end arbitrarily_large_N_exists_l2028_202803


namespace average_of_middle_two_l2028_202812

theorem average_of_middle_two (total_avg : ℝ) (first_two_avg : ℝ) (last_two_avg : ℝ) :
  total_avg = 3.95 →
  first_two_avg = 3.4 →
  last_two_avg = 4.600000000000001 →
  (6 * total_avg - 2 * first_two_avg - 2 * last_two_avg) / 2 = 3.85 := by
  sorry

end average_of_middle_two_l2028_202812


namespace dice_probability_l2028_202847

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def dice_outcome := ℕ × ℕ

def favorable_outcome (outcome : dice_outcome) : Prop :=
  is_prime outcome.1 ∧ is_perfect_square outcome.2

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 6

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 :=
sorry

end dice_probability_l2028_202847


namespace solution_count_l2028_202819

/-- For any integer k > 1, there exist at least 3k + 1 distinct triples of positive integers (m, n, r) 
    satisfying the equation mn + nr + mr = k(m + n + r). -/
theorem solution_count (k : ℕ) (h : k > 1) : 
  ∃ S : Finset (ℕ × ℕ × ℕ), 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m > 0 ∧ n > 0 ∧ r > 0) ∧ 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m * n + n * r + m * r = k * (m + n + r)) ∧
    S.card ≥ 3 * k + 1 :=
sorry

end solution_count_l2028_202819


namespace opera_house_rows_l2028_202829

/-- Represents an opera house with a certain number of rows -/
structure OperaHouse where
  rows : ℕ

/-- Represents a show at the opera house -/
structure Show where
  earnings : ℕ
  occupancyRate : ℚ

/-- Calculates the total number of seats in the opera house -/
def totalSeats (oh : OperaHouse) : ℕ := oh.rows * 10

/-- Calculates the number of tickets sold for a show -/
def ticketsSold (s : Show) : ℕ := s.earnings / 10

/-- Theorem: Given the conditions, the opera house has 150 rows -/
theorem opera_house_rows (oh : OperaHouse) (s : Show) :
  totalSeats oh = ticketsSold s / s.occupancyRate →
  s.earnings = 12000 →
  s.occupancyRate = 4/5 →
  oh.rows = 150 := by
  sorry


end opera_house_rows_l2028_202829


namespace union_of_M_and_N_l2028_202854

-- Define the sets M and N
def M : Set ℕ := {a : ℕ | a = 0 ∨ ∃ x, x = a}
def N : Set ℕ := {1, 2}

-- State the theorem
theorem union_of_M_and_N :
  (∃ a : ℕ, M = {a, 0}) →  -- M = {a, 0}
  N = {1, 2} →             -- N = {1, 2}
  M ∩ N = {1} →            -- M ∩ N = {1}
  M ∪ N = {0, 1, 2} :=     -- M ∪ N = {0, 1, 2}
by sorry

end union_of_M_and_N_l2028_202854


namespace arithmetic_sequence_sum_l2028_202802

/-- Given an arithmetic sequence {a_n} where a_1 + a_5 + a_9 = 6, prove that a_2 + a_8 = 4 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  (a 1 + a 5 + a 9 = 6) →                           -- given condition
  (a 2 + a 8 = 4) :=                                -- conclusion to prove
by sorry

end arithmetic_sequence_sum_l2028_202802


namespace book_sale_revenue_l2028_202876

/-- Given a collection of books where 2/3 were sold for $3.50 each and 40 remained unsold,
    prove that the total amount received for the sold books is $280. -/
theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) :
  (2 : ℚ) / 3 * total_books + 40 = total_books →
  price_per_book = (7 : ℚ) / 2 →
  ((2 : ℚ) / 3 * total_books) * price_per_book = 280 := by
  sorry

end book_sale_revenue_l2028_202876


namespace gcd_12345_6789_l2028_202848

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l2028_202848


namespace train_platform_crossing_time_l2028_202810

/-- Given a train of length 1100 meters that crosses a tree in 110 seconds,
    prove that it takes 180 seconds to pass a platform of length 700 meters. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1100
  let tree_crossing_time : ℝ := 110
  let platform_length : ℝ := 700
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  total_distance / train_speed = 180 := by sorry

end train_platform_crossing_time_l2028_202810


namespace quadratic_roots_relation_l2028_202851

/-- Given a quadratic equation x^2 + bx + c = 0 whose roots are each three more than
    the roots of 2x^2 - 4x - 8 = 0, prove that c = 11 -/
theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ p q : ℝ, (2 * p^2 - 4 * p - 8 = 0) ∧ 
              (2 * q^2 - 4 * q - 8 = 0) ∧ 
              ((p + 3)^2 + b * (p + 3) + c = 0) ∧ 
              ((q + 3)^2 + b * (q + 3) + c = 0)) →
  c = 11 := by
sorry


end quadratic_roots_relation_l2028_202851


namespace roots_of_polynomial_l2028_202863

def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (p 1 = 0) ∧ (p 2 = 0) ∧ (p 4 = 0) ∧
  (∀ x : ℝ, p x = 0 → x = 1 ∨ x = 2 ∨ x = 4) :=
by sorry

end roots_of_polynomial_l2028_202863


namespace mikes_total_spending_l2028_202846

/-- Represents Mike's shopping expenses -/
structure ShoppingExpenses where
  food : ℝ
  wallet : ℝ
  shirt : ℝ

/-- Calculates the total spending given Mike's shopping expenses -/
def totalSpending (expenses : ShoppingExpenses) : ℝ :=
  expenses.food + expenses.wallet + expenses.shirt

/-- Theorem stating Mike's total spending given the problem conditions -/
theorem mikes_total_spending :
  ∀ (expenses : ShoppingExpenses),
    expenses.food = 30 →
    expenses.wallet = expenses.food + 60 →
    expenses.shirt = expenses.wallet / 3 →
    totalSpending expenses = 150 := by
  sorry


end mikes_total_spending_l2028_202846


namespace product_expansion_l2028_202838

theorem product_expansion (x : ℝ) : (x + 2) * (x^2 + 3*x + 4) = x^3 + 5*x^2 + 10*x + 8 := by
  sorry

end product_expansion_l2028_202838


namespace problem_statement_l2028_202828

theorem problem_statement (x y z : ℝ) 
  (h1 : x * z / (x + y) + y * x / (y + z) + z * y / (z + x) = 2)
  (h2 : z * y / (x + y) + x * z / (y + z) + y * x / (z + x) = 3) :
  y / (x + y) + z / (y + z) + x / (z + x) = 1 := by
sorry

end problem_statement_l2028_202828


namespace no_even_three_digit_sum_27_l2028_202893

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_even_three_digit_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ Even n :=
sorry

end no_even_three_digit_sum_27_l2028_202893


namespace sum_of_cubes_l2028_202874

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := by
  sorry

end sum_of_cubes_l2028_202874


namespace gabby_shopping_funds_l2028_202861

theorem gabby_shopping_funds (total_cost available_funds : ℕ) : 
  total_cost = 165 → available_funds = 110 → total_cost - available_funds = 55 := by
  sorry

end gabby_shopping_funds_l2028_202861


namespace distribute_men_and_women_l2028_202816

/- Define the number of men and women -/
def num_men : ℕ := 4
def num_women : ℕ := 5

/- Define the size of each group -/
def group_size : ℕ := 3

/- Define a function to calculate the number of ways to distribute people -/
def distribute_people (m : ℕ) (w : ℕ) : ℕ :=
  let ways_group1 := (m.choose 1) * (w.choose 2)
  let ways_group2 := ((m - 1).choose 1) * ((w - 2).choose 2)
  ways_group1 * ways_group2 / 2

/- Theorem statement -/
theorem distribute_men_and_women :
  distribute_people num_men num_women = 180 :=
by sorry

end distribute_men_and_women_l2028_202816


namespace negative_three_times_two_l2028_202896

theorem negative_three_times_two : (-3 : ℤ) * 2 = -6 := by
  sorry

end negative_three_times_two_l2028_202896


namespace consecutive_product_plus_one_is_square_l2028_202890

theorem consecutive_product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end consecutive_product_plus_one_is_square_l2028_202890


namespace binomial_expansion_properties_l2028_202832

/-- Given the expansion of (1+2x)^10, prove properties about its coefficients -/
theorem binomial_expansion_properties :
  let n : ℕ := 10
  let expansion := fun (k : ℕ) => (n.choose k) * (2^k)
  let sum_first_three := 1 + 2 * n.choose 1 + 4 * n.choose 2
  -- Condition: sum of coefficients of first three terms is 201
  sum_first_three = 201 →
  -- 1. The binomial coefficient is largest for the 6th term
  (∀ k, k ≠ 5 → n.choose 5 ≥ n.choose k) ∧
  -- 2. The coefficient is largest for the 8th term
  (∀ k, k ≠ 7 → expansion 7 ≥ expansion k) :=
by sorry

end binomial_expansion_properties_l2028_202832


namespace triangle_similarity_BL_calculation_l2028_202862

theorem triangle_similarity_BL_calculation (AD BC AL BL LD LC : ℝ) 
  (h_similar : AD / BC = AL / BL ∧ AL / BL = LD / LC) :
  (∀ AB BD : ℝ, 
    (AB = 6 * Real.sqrt 13 ∧ AD = 6 ∧ BD = 12 * Real.sqrt 3) → 
    BL = 16 * Real.sqrt 3 - 12) ∧
  (∀ AB BD : ℝ, 
    (AB = 30 ∧ AD = 6 ∧ BD = 12 * Real.sqrt 6) → 
    BL = (16 * Real.sqrt 6 - 6) / 5) := by
  sorry

end triangle_similarity_BL_calculation_l2028_202862


namespace largest_sum_is_253_33_l2028_202897

/-- Represents a trapezium ABCD with specific angle properties -/
structure Trapezium where
  -- Internal angles in arithmetic progression
  b : ℝ
  e : ℝ
  -- Smallest angle is 35°
  smallest_angle : b = 35
  -- Sum of internal angles is 360°
  angle_sum : 4 * b + 6 * e = 360

/-- The largest possible sum of the two largest angles in the trapezium -/
def largest_sum_of_two_largest_angles (t : Trapezium) : ℝ :=
  2 * t.b + 5 * t.e

/-- Theorem stating the largest possible sum of the two largest angles -/
theorem largest_sum_is_253_33 (t : Trapezium) :
  largest_sum_of_two_largest_angles t = 253.33 := by
  sorry


end largest_sum_is_253_33_l2028_202897


namespace jennifer_fish_count_l2028_202805

/-- The number of tanks Jennifer has already built -/
def built_tanks : ℕ := 3

/-- The number of fish each built tank can hold -/
def fish_per_built_tank : ℕ := 15

/-- The number of tanks Jennifer plans to build -/
def planned_tanks : ℕ := 3

/-- The number of fish each planned tank can hold -/
def fish_per_planned_tank : ℕ := 10

/-- The total number of fish Jennifer wants to house -/
def total_fish : ℕ := built_tanks * fish_per_built_tank + planned_tanks * fish_per_planned_tank

theorem jennifer_fish_count : total_fish = 75 := by
  sorry

end jennifer_fish_count_l2028_202805


namespace gcf_5_factorial_6_factorial_l2028_202868

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcf_5_factorial_6_factorial : 
  Nat.gcd (factorial 5) (factorial 6) = factorial 5 := by
  sorry

end gcf_5_factorial_6_factorial_l2028_202868


namespace function_inequality_solution_set_l2028_202821

open Real

def isSolutionSet (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, f x < exp x ↔ x ∈ S

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv : ∀ x, deriv f x < f x)
  (hf_even : ∀ x, f (x + 2) = f (-x + 2))
  (hf_init : f 0 = exp 4) :
  isSolutionSet f (Set.Ici 4) :=
sorry

end function_inequality_solution_set_l2028_202821


namespace sqrt_equation_solution_l2028_202843

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (3 * x - 4)) = 4 → x = 173 / 3 := by
sorry

end sqrt_equation_solution_l2028_202843


namespace hay_shortage_farmer_hay_shortage_l2028_202809

/-- Calculates the hay shortage for a farmer given specific conditions --/
theorem hay_shortage (original_harvest : ℕ) (original_acres : ℕ) (additional_acres : ℕ)
                     (num_horses : ℕ) (hay_per_horse : ℕ) (feeding_months : ℕ) : ℤ :=
  let total_acres := original_acres + additional_acres
  let total_harvest := (original_harvest / original_acres) * total_acres
  let daily_consumption := hay_per_horse * num_horses
  let monthly_consumption := daily_consumption * 30
  let total_consumption := monthly_consumption * feeding_months
  total_consumption - total_harvest

/-- Proves that the farmer will be short by 1896 bales of hay --/
theorem farmer_hay_shortage :
  hay_shortage 560 5 7 9 3 4 = 1896 := by
  sorry

end hay_shortage_farmer_hay_shortage_l2028_202809


namespace binary_101_equals_5_l2028_202814

/- Define a function to convert binary to decimal -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/- Define the binary number 101 -/
def binary_101 : List Bool := [true, false, true]

/- Theorem statement -/
theorem binary_101_equals_5 :
  binary_to_decimal binary_101 = 5 := by sorry

end binary_101_equals_5_l2028_202814


namespace equation_solutions_l2028_202875

theorem equation_solutions :
  (∀ x : ℝ, 25 * x^2 = 81 ↔ x = 9/5 ∨ x = -9/5) ∧
  (∀ x : ℝ, (x - 2)^2 = 25 ↔ x = 7 ∨ x = -3) :=
by sorry

end equation_solutions_l2028_202875


namespace divisible_by_seven_count_l2028_202801

theorem divisible_by_seven_count : 
  (∃! (s : Finset Nat), 
    (∀ k ∈ s, k < 100 ∧ k > 0) ∧ 
    (∀ k ∈ s, ∀ n : Nat, n > 0 → (2 * (3^(6*n)) + k * (2^(3*n+1)) - 1) % 7 = 0) ∧
    s.card = 14) := by sorry

end divisible_by_seven_count_l2028_202801


namespace union_of_A_and_B_l2028_202840

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end union_of_A_and_B_l2028_202840


namespace max_value_cubic_function_l2028_202855

theorem max_value_cubic_function (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), -y^3 + 6*y^2 - m ≤ -x^3 + 6*x^2 - m) ∧
  (∃ (z : ℝ), -z^3 + 6*z^2 - m = 12) →
  m = 20 := by
sorry

end max_value_cubic_function_l2028_202855


namespace expense_increase_percentage_l2028_202853

theorem expense_increase_percentage (salary : ℝ) (initial_savings_rate : ℝ) (new_savings : ℝ) :
  salary = 5500 →
  initial_savings_rate = 0.2 →
  new_savings = 220 →
  let initial_savings := salary * initial_savings_rate
  let initial_expenses := salary - initial_savings
  let expense_increase := initial_savings - new_savings
  (expense_increase / initial_expenses) * 100 = 20 := by
  sorry

end expense_increase_percentage_l2028_202853


namespace sqrt_36_minus_k_squared_minus_6_equals_zero_l2028_202880

theorem sqrt_36_minus_k_squared_minus_6_equals_zero (k : ℝ) :
  Real.sqrt (36 - k^2) - 6 = 0 ↔ k = 0 := by sorry

end sqrt_36_minus_k_squared_minus_6_equals_zero_l2028_202880


namespace spherical_coords_reflection_l2028_202895

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    prove that the point (x, y, -z) has spherical coordinates (ρ, θ, π - φ) -/
theorem spherical_coords_reflection (x y z ρ θ φ : Real) 
  (h1 : ρ > 0) 
  (h2 : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h3 : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h4 : x = ρ * Real.sin φ * Real.cos θ)
  (h5 : y = ρ * Real.sin φ * Real.sin θ)
  (h6 : z = ρ * Real.cos φ)
  (h7 : ρ = 4)
  (h8 : θ = Real.pi / 4)
  (h9 : φ = Real.pi / 6) :
  ∃ (ρ' θ' φ' : Real),
    ρ' = ρ ∧
    θ' = θ ∧
    φ' = Real.pi - φ ∧
    x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    -z = ρ' * Real.cos φ' ∧
    ρ' > 0 ∧
    0 ≤ θ' ∧ θ' < 2 * Real.pi ∧
    0 ≤ φ' ∧ φ' ≤ Real.pi :=
by sorry

end spherical_coords_reflection_l2028_202895


namespace sum_remainder_mod_seven_l2028_202852

theorem sum_remainder_mod_seven : 
  (51730 % 7 + 51731 % 7 + 51732 % 7 + 51733 % 7 + 51734 % 7 + 51735 % 7) % 7 = 5 := by
  sorry

end sum_remainder_mod_seven_l2028_202852


namespace triangle_properties_l2028_202858

-- Define the triangle ABC
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (2, -2)

-- Define the altitude line equation
def altitude_equation (x y : ℝ) : Prop :=
  2 * x + y - 2 = 0

-- Define the circumcircle equation
def circumcircle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x + 4 * y - 8 = 0

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, altitude_equation x y ↔ 
    (x - A.1) * (B.2 - C.2) = (y - A.2) * (B.1 - C.1)) ∧
  (∀ x y : ℝ, circumcircle_equation x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end triangle_properties_l2028_202858


namespace exists_convex_quadrilateral_geometric_progression_l2028_202806

/-- A convex quadrilateral with sides a₁, a₂, a₃, a₄ and diagonals d₁, d₂ -/
structure ConvexQuadrilateral where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  d₁ : ℝ
  d₂ : ℝ
  a₁_pos : a₁ > 0
  a₂_pos : a₂ > 0
  a₃_pos : a₃ > 0
  a₄_pos : a₄ > 0
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  convex : a₁ + a₂ + a₃ > a₄ ∧
           a₁ + a₂ + a₄ > a₃ ∧
           a₁ + a₃ + a₄ > a₂ ∧
           a₂ + a₃ + a₄ > a₁

/-- Predicate to check if a sequence forms a geometric progression -/
def IsGeometricProgression (seq : List ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ i : Fin (seq.length - 1), seq[i.val + 1] = seq[i.val] * r

/-- Theorem stating the existence of a convex quadrilateral with sides and diagonals
    forming a geometric progression -/
theorem exists_convex_quadrilateral_geometric_progression :
  ∃ q : ConvexQuadrilateral, IsGeometricProgression [q.a₁, q.a₂, q.a₃, q.a₄, q.d₁, q.d₂] :=
sorry

end exists_convex_quadrilateral_geometric_progression_l2028_202806


namespace fraction_meaningful_iff_not_neg_two_l2028_202827

theorem fraction_meaningful_iff_not_neg_two (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end fraction_meaningful_iff_not_neg_two_l2028_202827


namespace emma_calculation_l2028_202884

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem emma_calculation (a b : ℕ) (ha : is_two_digit a) (hb : b > 0) :
  (reverse_digits a * b - 18 = 120) → (a * b = 192) := by
  sorry

end emma_calculation_l2028_202884


namespace fibonacci_sum_cube_square_l2028_202887

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define a predicate for Fibonacci numbers
def isFibonacci (n : ℕ) : Prop := ∃ k, fib k = n

-- Define the theorem
theorem fibonacci_sum_cube_square :
  ∀ a b : ℕ,
  isFibonacci a ∧ 49 < a ∧ a < 61 ∧
  isFibonacci b ∧ 59 < b ∧ b < 71 →
  a^3 + b^2 = 170096 :=
sorry

end fibonacci_sum_cube_square_l2028_202887


namespace cyclist_speed_l2028_202860

/-- The cyclist's problem -/
theorem cyclist_speed :
  ∀ (expected_speed actual_speed : ℝ),
  expected_speed > 0 →
  actual_speed > 0 →
  actual_speed = expected_speed + 1 →
  96 / actual_speed = 96 / expected_speed - 2 →
  96 / expected_speed = 1.25 →
  actual_speed = 16 := by
  sorry

end cyclist_speed_l2028_202860


namespace mans_speed_with_current_l2028_202869

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 11.2)
  (h2 : current_speed = 3.4) : 
  speed_against_current + 2 * current_speed = 18 := by
  sorry

#check mans_speed_with_current

end mans_speed_with_current_l2028_202869


namespace four_digit_sum_mod_1000_l2028_202822

def four_digit_sum : ℕ := sorry

theorem four_digit_sum_mod_1000 : four_digit_sum % 1000 = 320 := by sorry

end four_digit_sum_mod_1000_l2028_202822


namespace unique_four_digit_number_l2028_202882

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := 
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := 
  (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem unique_four_digit_number : 
  ∃! n : ℕ, 
    is_four_digit n ∧ 
    digit_sum n = 18 ∧ 
    middle_digits_sum n = 10 ∧ 
    thousands_minus_units n = 2 ∧ 
    n % 9 = 0 ∧
    n = 5643 :=
by
  sorry

end unique_four_digit_number_l2028_202882


namespace gcd_lcm_product_24_54_l2028_202894

theorem gcd_lcm_product_24_54 : Nat.gcd 24 54 * Nat.lcm 24 54 = 1296 := by
  sorry

end gcd_lcm_product_24_54_l2028_202894


namespace complex_number_location_l2028_202813

theorem complex_number_location :
  let z : ℂ := (3 + Complex.I) / (1 + Complex.I)
  (0 < z.re) ∧ (z.im < 0) :=
by sorry

end complex_number_location_l2028_202813


namespace find_m_l2028_202825

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 7

-- State the theorem
theorem find_m (m : ℝ) : (∀ x, f (1/2 * x - 1) = 2 * x + 3) → f m = 6 → m = -1/4 := by
  sorry

end find_m_l2028_202825


namespace sports_competition_results_l2028_202856

-- Define the probabilities of School A winning each event
def p1 : ℝ := 0.5
def p2 : ℝ := 0.4
def p3 : ℝ := 0.8

-- Define the score for winning an event
def win_score : ℕ := 10

-- Define the probability of School A winning the championship
def prob_A_wins : ℝ := p1 * p2 * p3 + p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3

-- Define the distribution of School B's total score
def dist_B : List (ℝ × ℝ) := [
  (0, (1 - p1) * (1 - p2) * (1 - p3)),
  (10, p1 * (1 - p2) * (1 - p3) + (1 - p1) * p2 * (1 - p3) + (1 - p1) * (1 - p2) * p3),
  (20, p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3),
  (30, p1 * p2 * p3)
]

-- Define the expectation of School B's total score
def exp_B : ℝ := (dist_B.map (λ p => p.1 * p.2)).sum

-- Theorem statement
theorem sports_competition_results :
  prob_A_wins = 0.6 ∧ exp_B = 13 := by sorry

end sports_competition_results_l2028_202856


namespace fraction_value_l2028_202844

theorem fraction_value : (2222 - 2123)^2 / 121 = 81 := by
  sorry

end fraction_value_l2028_202844


namespace allocation_methods_l2028_202865

def doctors : ℕ := 2
def nurses : ℕ := 4
def schools : ℕ := 2
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

theorem allocation_methods :
  (Nat.choose doctors doctors_per_school) * (Nat.choose nurses nurses_per_school) = 12 := by
  sorry

end allocation_methods_l2028_202865


namespace jeffs_towers_count_l2028_202824

/-- The number of sandcastles on Mark's beach -/
def marks_sandcastles : ℕ := 20

/-- The number of towers per sandcastle on Mark's beach -/
def marks_towers_per_castle : ℕ := 10

/-- The ratio of Jeff's sandcastles to Mark's sandcastles -/
def jeff_to_mark_ratio : ℕ := 3

/-- The total number of sandcastles and towers on both beaches -/
def total_objects : ℕ := 580

/-- The number of towers per sandcastle on Jeff's beach -/
def jeffs_towers_per_castle : ℕ := 5

theorem jeffs_towers_count : jeffs_towers_per_castle = 5 := by
  sorry

end jeffs_towers_count_l2028_202824


namespace transformed_quadratic_roots_l2028_202845

theorem transformed_quadratic_roots 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * r^2 + b * r + c = 0) 
  (h3 : a * s^2 + b * s + c = 0) : 
  (a * r + b)^2 - b * (a * r + b) + a * c = 0 ∧ 
  (a * s + b)^2 - b * (a * s + b) + a * c = 0 := by
  sorry

end transformed_quadratic_roots_l2028_202845


namespace ratio_q_to_r_l2028_202818

/-- 
Given a total amount of 1210 divided among three persons p, q, and r,
where the ratio of p to q is 5:4 and r receives 400,
prove that the ratio of q to r is 9:10.
-/
theorem ratio_q_to_r (total : ℕ) (p q r : ℕ) (h1 : total = 1210) 
  (h2 : p + q + r = total) (h3 : 5 * q = 4 * p) (h4 : r = 400) :
  9 * r = 10 * q := by
  sorry


end ratio_q_to_r_l2028_202818


namespace train_length_l2028_202834

/-- Given a train that crosses a tree in 120 seconds and takes 200 seconds to pass
    a platform 800 m long, moving at a constant speed, prove that the length of the train
    is 1200 meters. -/
theorem train_length (tree_time platform_time platform_length : ℝ)
    (h1 : tree_time = 120)
    (h2 : platform_time = 200)
    (h3 : platform_length = 800) :
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
  sorry

end train_length_l2028_202834


namespace journey_speed_calculation_l2028_202831

/-- Proves that given a journey of 12 hours covering 560 km, where the first half of the distance 
is traveled at 35 kmph, the speed for the second half of the journey is 70 kmph. -/
theorem journey_speed_calculation (total_time : ℝ) (total_distance : ℝ) (first_half_speed : ℝ) :
  total_time = 12 →
  total_distance = 560 →
  first_half_speed = 35 →
  (total_distance / 2) / first_half_speed + (total_distance / 2) / ((total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed)) = total_time →
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 70 := by
  sorry

#check journey_speed_calculation

end journey_speed_calculation_l2028_202831


namespace nth_root_two_inequality_l2028_202842

theorem nth_root_two_inequality (n : ℕ) (h : n ≥ 2) :
  (2 : ℝ) ^ (1 / n) - 1 ≤ Real.sqrt (2 / (n * (n - 1))) := by
  sorry

end nth_root_two_inequality_l2028_202842


namespace chairs_remaining_l2028_202857

def classroom_chairs (total red yellow blue green orange : ℕ) : Prop :=
  total = 62 ∧
  red = 4 ∧
  yellow = 2 * red ∧
  blue = 3 * yellow ∧
  green = blue / 2 ∧
  orange = green + 2 ∧
  total = red + yellow + blue + green + orange

def lisa_borrows (total borrowed : ℕ) : Prop :=
  borrowed = total / 10

def carla_borrows (remaining borrowed : ℕ) : Prop :=
  borrowed = remaining / 5

theorem chairs_remaining 
  (total red yellow blue green orange : ℕ)
  (lisa_borrowed carla_borrowed : ℕ)
  (h1 : classroom_chairs total red yellow blue green orange)
  (h2 : lisa_borrows total lisa_borrowed)
  (h3 : carla_borrows (total - lisa_borrowed) carla_borrowed) :
  total - lisa_borrowed - carla_borrowed = 45 :=
sorry

end chairs_remaining_l2028_202857


namespace factorial_equation_l2028_202889

theorem factorial_equation : 6 * 10 * 4 * 168 = Nat.factorial 8 := by
  sorry

end factorial_equation_l2028_202889


namespace multiplication_addition_equality_l2028_202833

theorem multiplication_addition_equality : 85 * 1500 + (1 / 2) * 1500 = 128250 := by
  sorry

end multiplication_addition_equality_l2028_202833


namespace complex_product_l2028_202815

/-- Given complex numbers Q, E, and D, prove that their product is -25i. -/
theorem complex_product (Q E D : ℂ) : 
  Q = 3 + 4*I ∧ E = -I ∧ D = 3 - 4*I → Q * E * D = -25 * I :=
by sorry

end complex_product_l2028_202815


namespace mary_ray_difference_l2028_202886

/-- The number of chickens taken by each person -/
structure ChickenDistribution where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken distribution problem -/
def valid_distribution (d : ChickenDistribution) : Prop :=
  d.john = d.mary + 5 ∧
  d.ray < d.mary ∧
  d.ray = 10 ∧
  d.john = d.ray + 11

/-- The theorem stating the difference between Mary's and Ray's chickens -/
theorem mary_ray_difference (d : ChickenDistribution) 
  (h : valid_distribution d) : d.mary - d.ray = 6 := by
  sorry

#check mary_ray_difference

end mary_ray_difference_l2028_202886


namespace negation_of_universal_positive_square_l2028_202820

theorem negation_of_universal_positive_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

end negation_of_universal_positive_square_l2028_202820


namespace horner_method_correct_l2028_202849

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 4x^2 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x - 4

theorem horner_method_correct :
  f 3 = horner [2, 3, 4, 5, -4] 3 ∧ horner [2, 3, 4, 5, -4] 3 = 290 := by
  sorry

end horner_method_correct_l2028_202849


namespace equal_area_perimeter_rectangle_dimensions_l2028_202850

/-- A rectangle with integer side lengths where the area equals the perimeter. -/
structure EqualAreaPerimeterRectangle where
  width : ℕ
  length : ℕ
  area_eq_perimeter : width * length = 2 * (width + length)

/-- The possible dimensions of a rectangle with integer side lengths where the area equals the perimeter. -/
def valid_dimensions : Set (ℕ × ℕ) :=
  {(4, 4), (3, 6), (6, 3)}

/-- Theorem stating that the only valid dimensions for a rectangle with integer side lengths
    where the area equals the perimeter are 4x4, 3x6, or 6x3. -/
theorem equal_area_perimeter_rectangle_dimensions (r : EqualAreaPerimeterRectangle) :
  (r.width, r.length) ∈ valid_dimensions := by
  sorry

#check equal_area_perimeter_rectangle_dimensions

end equal_area_perimeter_rectangle_dimensions_l2028_202850


namespace range_of_3x_plus_2y_l2028_202883

theorem range_of_3x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 4) : 
  2 ≤ 3*x + 2*y ∧ 3*x + 2*y ≤ 9.5 :=
by sorry

end range_of_3x_plus_2y_l2028_202883


namespace hexagon_diagonal_intersection_probability_l2028_202804

/-- A convex hexagon -/
structure ConvexHexagon where
  -- We don't need to define the structure of a hexagon for this problem

/-- A diagonal of a hexagon -/
structure Diagonal (h : ConvexHexagon) where
  -- We don't need to define the structure of a diagonal for this problem

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (h : ConvexHexagon) (d1 d2 : Diagonal h) : Prop :=
  sorry  -- Definition not provided, as it's not necessary for the statement

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def intersection_probability (h : ConvexHexagon) : ℚ :=
  sorry  -- Definition not provided, as it's not necessary for the statement

/-- Theorem stating that the probability of two randomly chosen diagonals 
    intersecting inside a convex hexagon is 5/12 -/
theorem hexagon_diagonal_intersection_probability (h : ConvexHexagon) :
  intersection_probability h = 5 / 12 :=
sorry

end hexagon_diagonal_intersection_probability_l2028_202804


namespace origin_not_in_convex_hull_probability_l2028_202885

/-- The unit circle in the complex plane -/
def S1 : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- The probability that the origin is not contained in the convex hull of n randomly selected points from S¹ -/
noncomputable def probability (n : ℕ) : ℝ := 1 - (n : ℝ) / 2^(n - 1)

/-- Theorem: The probability that the origin is not contained in the convex hull of seven randomly selected points from S¹ is 57/64 -/
theorem origin_not_in_convex_hull_probability :
  probability 7 = 57 / 64 := by sorry

end origin_not_in_convex_hull_probability_l2028_202885


namespace B_2_2_l2028_202826

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2 : B 2 2 = 16 := by
  sorry

end B_2_2_l2028_202826


namespace smallest_positive_leading_coeff_l2028_202872

/-- A quadratic polynomial that takes integer values for all integer inputs. -/
def IntegerValuedQuadratic (a b c : ℚ) : ℤ → ℤ :=
  fun x => ⌊a * x^2 + b * x + c⌋

/-- The property that a quadratic polynomial takes integer values for all integer inputs. -/
def IsIntegerValued (a b c : ℚ) : Prop :=
  ∀ x : ℤ, (IntegerValuedQuadratic a b c x : ℚ) = a * x^2 + b * x + c

/-- The smallest positive leading coefficient of an integer-valued quadratic polynomial is 1/2. -/
theorem smallest_positive_leading_coeff :
  (∃ a b c : ℚ, a > 0 ∧ IsIntegerValued a b c) ∧
  (∀ a b c : ℚ, a > 0 → IsIntegerValued a b c → a ≥ 1/2) ∧
  (∃ b c : ℚ, IsIntegerValued (1/2) b c) :=
sorry

end smallest_positive_leading_coeff_l2028_202872


namespace cube_distance_to_plane_l2028_202877

/-- Given a cube with side length 10 and three vertices adjacent to the closest vertex A
    at heights 10, 11, and 12 above a plane, prove that the distance from A to the plane
    is (33-√294)/3 -/
theorem cube_distance_to_plane (cube_side : ℝ) (height_1 height_2 height_3 : ℝ) :
  cube_side = 10 →
  height_1 = 10 →
  height_2 = 11 →
  height_3 = 12 →
  ∃ (distance : ℝ), distance = (33 - Real.sqrt 294) / 3 ∧
    distance = min height_1 (min height_2 height_3) - 
      Real.sqrt ((cube_side^2 - (height_2 - height_1)^2) / 4 +
                 (cube_side^2 - (height_3 - height_1)^2) / 4 +
                 (cube_side^2 - (height_3 - height_2)^2) / 4) := by
  sorry

end cube_distance_to_plane_l2028_202877


namespace x_intercept_is_four_l2028_202836

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 4 -/
theorem x_intercept_is_four :
  let l : Line := { x₁ := 10, y₁ := 3, x₂ := -10, y₂ := -7 }
  x_intercept l = 4 := by
  sorry

end x_intercept_is_four_l2028_202836


namespace pythagorean_triple_identification_l2028_202888

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 3 4 6 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 9 12 15 :=
by sorry

end pythagorean_triple_identification_l2028_202888
