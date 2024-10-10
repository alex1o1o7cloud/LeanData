import Mathlib

namespace system_solution_l2310_231007

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x = -10 - 3 * y) ∧ 
  (4 * x = 5 * y - 35) ∧ 
  (x = -155 / 47) ∧ 
  (y = 205 / 47) := by
  sorry

end system_solution_l2310_231007


namespace lizzy_money_theorem_l2310_231001

/-- Calculates the final amount Lizzy has after lending money and receiving it back with interest -/
def final_amount (initial : ℝ) (loan : ℝ) (interest_rate : ℝ) : ℝ :=
  initial - loan + loan * (1 + interest_rate)

/-- Theorem stating that given the specific conditions, Lizzy will have $33 -/
theorem lizzy_money_theorem :
  let initial := 30
  let loan := 15
  let interest_rate := 0.2
  final_amount initial loan interest_rate = 33 := by
  sorry

end lizzy_money_theorem_l2310_231001


namespace min_value_of_f_inequality_condition_l2310_231016

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 2|

-- Statement 1: The minimum value of f(x) is 4
theorem min_value_of_f : ∃ (x : ℝ), f x = 4 ∧ ∀ (y : ℝ), f y ≥ 4 :=
sorry

-- Statement 2: f(x) ≥ |a+4| - |a-3| for all x if and only if a ≤ 3/2
theorem inequality_condition (a : ℝ) : 
  (∀ (x : ℝ), f x ≥ |a + 4| - |a - 3|) ↔ a ≤ 3/2 :=
sorry

end min_value_of_f_inequality_condition_l2310_231016


namespace range_of_a_l2310_231004

def statement_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def statement_q (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

theorem range_of_a :
  ∀ a : ℝ, (statement_p a ∨ statement_q a) ∧ ¬(statement_p a ∧ statement_q a) →
    (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) := by
  sorry

end range_of_a_l2310_231004


namespace probability_three_odd_less_than_eighth_l2310_231092

def range_size : ℕ := 2023
def odd_count : ℕ := (range_size + 1) / 2

theorem probability_three_odd_less_than_eighth :
  (odd_count : ℚ) / range_size *
  ((odd_count - 1) : ℚ) / (range_size - 1) *
  ((odd_count - 2) : ℚ) / (range_size - 2) <
  1 / 8 :=
sorry

end probability_three_odd_less_than_eighth_l2310_231092


namespace sum_of_k_values_l2310_231077

theorem sum_of_k_values : ∃ (S : Finset ℕ), 
  (∀ k ∈ S, ∃ j : ℕ, j > 0 ∧ k > 0 ∧ (1 : ℚ) / j + 1 / k = (1 : ℚ) / 4) ∧
  (∀ k : ℕ, k > 0 → (∃ j : ℕ, j > 0 ∧ (1 : ℚ) / j + 1 / k = (1 : ℚ) / 4) → k ∈ S) ∧
  Finset.sum S id = 51 :=
sorry

end sum_of_k_values_l2310_231077


namespace jack_initial_marbles_l2310_231060

/-- The number of marbles Jack shared with Rebecca -/
def shared_marbles : ℕ := 33

/-- The number of marbles Jack had after sharing -/
def remaining_marbles : ℕ := 29

/-- The initial number of marbles Jack had -/
def initial_marbles : ℕ := shared_marbles + remaining_marbles

theorem jack_initial_marbles : initial_marbles = 62 := by
  sorry

end jack_initial_marbles_l2310_231060


namespace f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l2310_231042

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 :=
sorry

end f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l2310_231042


namespace smallest_cut_length_l2310_231040

theorem smallest_cut_length : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ 8 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → (8 - y) + (15 - y) > 17 - y) ∧
  (8 - x) + (15 - x) ≤ 17 - x ∧
  x = 6 := by
  sorry

end smallest_cut_length_l2310_231040


namespace valentina_share_ratio_l2310_231073

/-- The length of the burger in inches -/
def burger_length : ℚ := 12

/-- The length of each person's share in inches -/
def share_length : ℚ := 6

/-- The ratio of Valentina's share to the whole burger -/
def valentina_ratio : ℚ × ℚ := (share_length, burger_length)

theorem valentina_share_ratio :
  valentina_ratio = (1, 2) := by sorry

end valentina_share_ratio_l2310_231073


namespace count_divisible_integers_l2310_231039

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1806 : ℤ) ∣ (m^2 - 2)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (1806 : ℤ) ∣ (m^2 - 2) → m ∈ S) ∧
    Finset.card S = 2 :=
by sorry

end count_divisible_integers_l2310_231039


namespace readers_intersection_l2310_231066

theorem readers_intersection (total : ℕ) (sci_fi : ℕ) (literary : ℕ) 
  (h1 : total = 250) (h2 : sci_fi = 180) (h3 : literary = 88) :
  sci_fi + literary - total = 18 := by
  sorry

end readers_intersection_l2310_231066


namespace sum_of_squares_not_prime_l2310_231082

theorem sum_of_squares_not_prime (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : a * b = c * d) : 
  ¬ Nat.Prime (Int.natAbs (a^2 + b^2 + c^2 + d^2)) :=
by sorry

end sum_of_squares_not_prime_l2310_231082


namespace monthly_fee_plan_a_correct_l2310_231029

/-- The monthly fee for Plan A in a cell phone company's text-messaging plans. -/
def monthly_fee_plan_a : ℝ := 9

/-- The cost per text message for Plan A. -/
def cost_per_text_plan_a : ℝ := 0.25

/-- The cost per text message for Plan B. -/
def cost_per_text_plan_b : ℝ := 0.40

/-- The number of text messages at which both plans cost the same. -/
def equal_cost_messages : ℕ := 60

/-- Theorem stating that the monthly fee for Plan A is correct. -/
theorem monthly_fee_plan_a_correct :
  monthly_fee_plan_a = 
    equal_cost_messages * (cost_per_text_plan_b - cost_per_text_plan_a) :=
by sorry

end monthly_fee_plan_a_correct_l2310_231029


namespace arithmetic_sequence_problem_l2310_231062

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47,
    the 8th term is 71. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_4th : a 4 = 23) 
    (h_6th : a 6 = 47) : 
  a 8 = 71 := by
  sorry


end arithmetic_sequence_problem_l2310_231062


namespace bus_children_count_l2310_231074

theorem bus_children_count (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 26 → additional = 38 → total = initial + additional → total = 64 := by
sorry

end bus_children_count_l2310_231074


namespace sqrt_three_multiplication_l2310_231033

theorem sqrt_three_multiplication : Real.sqrt 3 * (2 * Real.sqrt 3 - 2) = 6 - 2 * Real.sqrt 3 := by
  sorry

end sqrt_three_multiplication_l2310_231033


namespace parabola_properties_l2310_231051

-- Define the parabola function
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem parabola_properties (a b c : ℝ) :
  (parabola a b c (-2) = 0) →
  (parabola a b c (-1) = 4) →
  (parabola a b c 0 = 6) →
  (parabola a b c 1 = 6) →
  (a < 0) ∧
  (∀ x, parabola a b c x ≤ parabola a b c (1/2)) ∧
  (parabola a b c (1/2) = 25/4) := by
  sorry

end parabola_properties_l2310_231051


namespace division_problem_l2310_231027

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1637) (h3 : a = b * q + 5) : q = 6 := by
  sorry

end division_problem_l2310_231027


namespace amount_after_two_years_l2310_231012

theorem amount_after_two_years (initial_amount : ℝ) (increase_ratio : ℝ) :
  initial_amount = 70400 →
  increase_ratio = 1 / 8 →
  initial_amount * (1 + increase_ratio)^2 = 89070 :=
by sorry

end amount_after_two_years_l2310_231012


namespace sum_interior_angles_limited_diagonal_polygon_l2310_231091

/-- A polygon where at most 6 diagonals can be drawn from any vertex -/
structure LimitedDiagonalPolygon where
  vertices : ℕ
  diagonals_limit : vertices - 3 = 6

/-- The sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

/-- Theorem: The sum of interior angles of a LimitedDiagonalPolygon is 1260° -/
theorem sum_interior_angles_limited_diagonal_polygon (p : LimitedDiagonalPolygon) :
  sum_interior_angles p.vertices = 1260 := by
  sorry

#eval sum_interior_angles 9  -- Expected output: 1260

end sum_interior_angles_limited_diagonal_polygon_l2310_231091


namespace three_numbers_sum_l2310_231090

theorem three_numbers_sum (a b c x y z : ℝ) : 
  (x + y = z + a) → 
  (x + z = y + b) → 
  (y + z = x + c) → 
  (x = (a + b - c) / 2) ∧ 
  (y = (a - b + c) / 2) ∧ 
  (z = (-a + b + c) / 2) := by
  sorry

end three_numbers_sum_l2310_231090


namespace card_purchase_cost_l2310_231094

/-- Calculates the total cost of cards purchased from two boxes, including sales tax. -/
def total_cost (price1 : ℚ) (price2 : ℚ) (count1 : ℕ) (count2 : ℕ) (tax_rate : ℚ) : ℚ :=
  let subtotal := price1 * count1 + price2 * count2
  subtotal * (1 + tax_rate)

/-- Proves that the total cost of 8 cards from the first box and 12 cards from the second box, including 7% sales tax, is $33.17. -/
theorem card_purchase_cost : 
  total_cost (25/20) (35/20) 8 12 (7/100) = 3317/100 := by
  sorry

#eval total_cost (25/20) (35/20) 8 12 (7/100)

end card_purchase_cost_l2310_231094


namespace divisibility_criterion_l2310_231068

theorem divisibility_criterion (p : ℕ) (hp : Nat.Prime p) :
  (∀ x y : ℕ, x > 0 → y > 0 → p ∣ (x + y)^19 - x^19 - y^19) ↔ p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19 :=
sorry

end divisibility_criterion_l2310_231068


namespace range_of_g_l2310_231083

noncomputable def g (x : ℝ) : ℝ := Real.arctan (x^2) + Real.arctan ((2 - 2*x^2) / (1 + 2*x^2))

theorem range_of_g : ∀ x : ℝ, g x = Real.arctan 2 := by
  sorry

end range_of_g_l2310_231083


namespace gwen_homework_problems_l2310_231002

/-- The number of math problems Gwen had -/
def math_problems : ℕ := 18

/-- The number of science problems Gwen had -/
def science_problems : ℕ := 11

/-- The number of problems Gwen finished at school -/
def finished_at_school : ℕ := 24

/-- The number of problems Gwen had to do for homework -/
def homework_problems : ℕ := math_problems + science_problems - finished_at_school

theorem gwen_homework_problems :
  homework_problems = 5 := by sorry

end gwen_homework_problems_l2310_231002


namespace log_inequality_range_l2310_231020

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_inequality_range (a : ℝ) :
  log a (2/5) < 1 ↔ (0 < a ∧ a < 2/5) ∨ a > 1 :=
by sorry

end log_inequality_range_l2310_231020


namespace square_side_length_with_equal_perimeter_circle_l2310_231054

theorem square_side_length_with_equal_perimeter_circle (r : ℝ) :
  ∃ (s : ℝ), (4 * s = 2 * π * r) → (s = 3 * π / 2) :=
by
  sorry

end square_side_length_with_equal_perimeter_circle_l2310_231054


namespace distance_A_to_C_distance_A_to_C_is_300_l2310_231009

/-- The distance between city A and city C given the travel times and speeds of Eddy and Freddy -/
theorem distance_A_to_C (eddy_time : ℝ) (freddy_time : ℝ) (distance_A_to_B : ℝ) (speed_ratio : ℝ) : ℝ :=
  let eddy_speed := distance_A_to_B / eddy_time
  let freddy_speed := eddy_speed / speed_ratio
  freddy_speed * freddy_time

/-- The actual distance between city A and city C is 300 km -/
theorem distance_A_to_C_is_300 : distance_A_to_C 3 4 450 2 = 300 := by
  sorry

end distance_A_to_C_distance_A_to_C_is_300_l2310_231009


namespace exactly_two_trains_on_time_l2310_231034

-- Define the probabilities of each train arriving on time
def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

-- Define the probability of exactly two trains arriving on time
def P_exactly_two : ℝ := 
  P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C

-- Theorem statement
theorem exactly_two_trains_on_time : P_exactly_two = 0.398 := by
  sorry

end exactly_two_trains_on_time_l2310_231034


namespace diesel_tank_capacity_l2310_231036

/-- Given the cost of a certain volume of diesel fuel and the cost of a full tank,
    calculate the capacity of the tank in liters. -/
theorem diesel_tank_capacity 
  (fuel_volume : ℝ) 
  (fuel_cost : ℝ) 
  (full_tank_cost : ℝ) 
  (h1 : fuel_volume = 36) 
  (h2 : fuel_cost = 18) 
  (h3 : full_tank_cost = 32) : 
  (full_tank_cost / (fuel_cost / fuel_volume)) = 64 := by
  sorry

#check diesel_tank_capacity

end diesel_tank_capacity_l2310_231036


namespace absolute_value_inequality_l2310_231063

theorem absolute_value_inequality (x : ℝ) : 
  |((5 - x) / 3)| < 2 ↔ -1 < x ∧ x < 11 := by sorry

end absolute_value_inequality_l2310_231063


namespace greatest_two_digit_multiple_of_17_l2310_231079

theorem greatest_two_digit_multiple_of_17 :
  ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 17 ∣ m → m ≤ n) ∧
  17 ∣ n ∧ 10 ≤ n ∧ n ≤ 99 :=
by sorry

end greatest_two_digit_multiple_of_17_l2310_231079


namespace rationalize_and_simplify_l2310_231071

theorem rationalize_and_simplify :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end rationalize_and_simplify_l2310_231071


namespace max_gum_pieces_is_31_l2310_231089

/-- Represents the number of coins Quentavious has -/
structure Coins where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Represents the exchange rates for gum pieces -/
structure ExchangeRates where
  nickel_rate : ℕ
  dime_rate : ℕ
  quarter_rate : ℕ

/-- Represents the maximum number of coins that can be exchanged -/
structure MaxExchange where
  max_nickels : ℕ
  max_dimes : ℕ
  max_quarters : ℕ

/-- Calculates the maximum number of gum pieces Quentavious can get -/
def max_gum_pieces (coins : Coins) (rates : ExchangeRates) (max_exchange : MaxExchange) 
  (keep_nickels keep_dimes : ℕ) : ℕ :=
  let exchangeable_nickels := min (coins.nickels - keep_nickels) max_exchange.max_nickels
  let exchangeable_dimes := min (coins.dimes - keep_dimes) max_exchange.max_dimes
  let exchangeable_quarters := min coins.quarters max_exchange.max_quarters
  exchangeable_nickels * rates.nickel_rate + 
  exchangeable_dimes * rates.dime_rate + 
  exchangeable_quarters * rates.quarter_rate

/-- Theorem stating that the maximum number of gum pieces Quentavious can get is 31 -/
theorem max_gum_pieces_is_31 
  (coins : Coins)
  (rates : ExchangeRates)
  (max_exchange : MaxExchange)
  (h_coins : coins = ⟨5, 6, 4⟩)
  (h_rates : rates = ⟨2, 3, 5⟩)
  (h_max_exchange : max_exchange = ⟨3, 4, 2⟩)
  (h_keep_nickels : 2 ≤ coins.nickels)
  (h_keep_dimes : 1 ≤ coins.dimes) :
  max_gum_pieces coins rates max_exchange 2 1 = 31 :=
sorry

end max_gum_pieces_is_31_l2310_231089


namespace quadratic_roots_reciprocal_sum_l2310_231024

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) :
  x₁^2 - 5*x₁ + 4 = 0 →
  x₂^2 - 5*x₂ + 4 = 0 →
  x₁ ≠ x₂ →
  (1 / x₁) + (1 / x₂) = 5/4 := by
  sorry

end quadratic_roots_reciprocal_sum_l2310_231024


namespace judys_score_is_25_l2310_231010

/-- Represents the scoring system for a math competition -/
structure ScoringSystem where
  correctPoints : Int
  incorrectPoints : Int

/-- Represents a participant's answers in the competition -/
structure Answers where
  total : Nat
  correct : Nat
  incorrect : Nat
  unanswered : Nat

/-- Calculates the score based on the scoring system and answers -/
def calculateScore (system : ScoringSystem) (answers : Answers) : Int :=
  system.correctPoints * answers.correct + system.incorrectPoints * answers.incorrect

/-- Theorem: Judy's score in the math competition is 25 points -/
theorem judys_score_is_25 (system : ScoringSystem) (answers : Answers) :
  system.correctPoints = 2 →
  system.incorrectPoints = -1 →
  answers.total = 30 →
  answers.correct = 15 →
  answers.incorrect = 5 →
  answers.unanswered = 10 →
  calculateScore system answers = 25 := by
  sorry

#eval calculateScore { correctPoints := 2, incorrectPoints := -1 }
                     { total := 30, correct := 15, incorrect := 5, unanswered := 10 }

end judys_score_is_25_l2310_231010


namespace composite_function_evaluation_l2310_231003

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composite_function_evaluation :
  g (f (g 3)) = 192 := by
  sorry

end composite_function_evaluation_l2310_231003


namespace multiplication_addition_equality_l2310_231013

theorem multiplication_addition_equality : 45 * 72 + 28 * 45 = 4500 := by
  sorry

end multiplication_addition_equality_l2310_231013


namespace celine_payment_l2310_231023

/-- Represents the daily charge for borrowing a book -/
def daily_charge : ℚ := 1/2

/-- Represents the number of days in May -/
def days_in_may : ℕ := 31

/-- Represents the number of books Celine borrowed -/
def books_borrowed : ℕ := 3

/-- Represents the number of days Celine kept the first book -/
def days_first_book : ℕ := 20

/-- Calculates the total amount Celine paid for borrowing the books -/
def total_amount : ℚ :=
  daily_charge * days_first_book +
  daily_charge * days_in_may * 2

theorem celine_payment :
  total_amount = 41 :=
sorry

end celine_payment_l2310_231023


namespace pages_per_book_l2310_231075

/-- Given that Frank took 12 days to finish each book and 492 days to finish all 41 books,
    prove that each book had 492 pages. -/
theorem pages_per_book (days_per_book : ℕ) (total_days : ℕ) (total_books : ℕ) :
  days_per_book = 12 →
  total_days = 492 →
  total_books = 41 →
  (total_days / days_per_book) * days_per_book = 492 := by
  sorry

#check pages_per_book

end pages_per_book_l2310_231075


namespace parallelogram_on_circle_l2310_231031

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if a point is on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a function to check if a quadrilateral is a parallelogram
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2)

theorem parallelogram_on_circle (a c : ℝ × ℝ) (γ : Circle) :
  ∃ (b d : ℝ × ℝ), onCircle b γ ∧ onCircle d γ ∧ isParallelogram a b c d :=
sorry

end parallelogram_on_circle_l2310_231031


namespace sophomore_selection_l2310_231015

/-- Calculates the number of sophomores selected for a study tour using proportional allocation -/
theorem sophomore_selection (freshmen sophomore junior total_spots : ℕ) : 
  freshmen = 240 →
  sophomore = 260 →
  junior = 300 →
  total_spots = 40 →
  (sophomore * total_spots) / (freshmen + sophomore + junior) = 26 := by
  sorry

end sophomore_selection_l2310_231015


namespace angle_bisector_ratio_bound_angle_bisector_ratio_bound_tight_l2310_231059

/-- A triangle with sides a and b, and corresponding angle bisectors t_a and t_b -/
structure Triangle where
  a : ℝ
  b : ℝ
  t_a : ℝ
  t_b : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < t_a ∧ 0 < t_b
  h_triangle : a < b + t_b ∧ b < a + t_a ∧ t_b < a + b
  h_bisector_a : t_a < (2 * b * (a + b)) / (a + 2 * b)
  h_bisector_b : t_b < (2 * a * (a + b)) / (2 * a + b)

/-- The upper bound for the ratio of sum of angle bisectors to sum of sides is 4/3 -/
theorem angle_bisector_ratio_bound (T : Triangle) :
  (T.t_a + T.t_b) / (T.a + T.b) < 4/3 :=
sorry

/-- The upper bound 4/3 is the least possible -/
theorem angle_bisector_ratio_bound_tight :
  ∀ ε > 0, ∃ T : Triangle, (4/3 - ε) < (T.t_a + T.t_b) / (T.a + T.b) :=
sorry

end angle_bisector_ratio_bound_angle_bisector_ratio_bound_tight_l2310_231059


namespace ascending_order_abc_l2310_231086

theorem ascending_order_abc :
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4/5 : ℝ)
  let c := Real.sin 1
  a < c ∧ c < b := by sorry

end ascending_order_abc_l2310_231086


namespace miss_adamson_classes_l2310_231030

/-- The number of classes Miss Adamson has -/
def number_of_classes (students_per_class : ℕ) (sheets_per_student : ℕ) (total_sheets : ℕ) : ℕ :=
  total_sheets / (students_per_class * sheets_per_student)

/-- Proof that Miss Adamson has 4 classes -/
theorem miss_adamson_classes :
  number_of_classes 20 5 400 = 4 := by
  sorry

end miss_adamson_classes_l2310_231030


namespace average_tv_sets_is_48_l2310_231044

/-- The average number of TV sets in 5 electronic shops -/
def average_tv_sets : ℚ :=
  let shops := 5
  let tv_sets := [20, 30, 60, 80, 50]
  (tv_sets.sum : ℚ) / shops

/-- Theorem: The average number of TV sets in the 5 electronic shops is 48 -/
theorem average_tv_sets_is_48 : average_tv_sets = 48 := by
  sorry

end average_tv_sets_is_48_l2310_231044


namespace negation_of_set_implication_l2310_231064

theorem negation_of_set_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B ≠ A → A ∩ B ≠ B) :=
by sorry

end negation_of_set_implication_l2310_231064


namespace eight_paths_A_to_C_l2310_231026

/-- Represents a simple directed graph with four nodes -/
structure DirectedGraph :=
  (paths_A_to_B : ℕ)
  (paths_B_to_C : ℕ)
  (paths_B_to_D : ℕ)
  (paths_D_to_C : ℕ)

/-- Calculates the total number of paths from A to C -/
def total_paths_A_to_C (g : DirectedGraph) : ℕ :=
  g.paths_A_to_B * (g.paths_B_to_C + g.paths_B_to_D * g.paths_D_to_C)

/-- Theorem stating that for the given graph configuration, there are 8 paths from A to C -/
theorem eight_paths_A_to_C :
  ∃ (g : DirectedGraph),
    g.paths_A_to_B = 2 ∧
    g.paths_B_to_C = 3 ∧
    g.paths_B_to_D = 1 ∧
    g.paths_D_to_C = 1 ∧
    total_paths_A_to_C g = 8 :=
by sorry

end eight_paths_A_to_C_l2310_231026


namespace sin_has_P_pi_property_P4_central_sym_monotone_P0_P3_implies_periodic_l2310_231046

-- Definition of P(a) property
def has_P_property (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, ∃ a, f (x + a) = f (-x)

-- Statement 1
theorem sin_has_P_pi_property : has_P_property Real.sin π :=
  sorry

-- Definition of central symmetry about a point
def centrally_symmetric (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

-- Definition of monotonically decreasing on an interval
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Definition of monotonically increasing on an interval
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Statement 3
theorem P4_central_sym_monotone (f : ℝ → ℝ) 
  (h1 : has_P_property f 4)
  (h2 : centrally_symmetric f (1, 0))
  (h3 : ∃ ε > 0, monotone_decreasing_on f (-1-ε) (-1+ε)) :
  monotone_decreasing_on f (-2) (-1) ∧ monotone_increasing_on f 1 2 :=
  sorry

-- Definition of periodic function
def periodic (f : ℝ → ℝ) : Prop :=
  ∃ p ≠ 0, ∀ x, f (x + p) = f x

-- Statement 4
theorem P0_P3_implies_periodic (f g : ℝ → ℝ)
  (h1 : f ≠ 0)
  (h2 : has_P_property f 0)
  (h3 : has_P_property f 3)
  (h4 : ∀ x₁ x₂, |f x₁ - f x₂| ≥ |g x₁ - g x₂|) :
  periodic g :=
  sorry

end sin_has_P_pi_property_P4_central_sym_monotone_P0_P3_implies_periodic_l2310_231046


namespace complex_equation_solution_l2310_231097

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I → a + b * Complex.I = 1 + 2 * Complex.I :=
by sorry

end complex_equation_solution_l2310_231097


namespace intersection_implies_a_value_l2310_231072

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a = {x | -2 ≤ x ∧ x ≤ 1}) → a = -2 := by
  sorry

end intersection_implies_a_value_l2310_231072


namespace sales_difference_is_25_l2310_231014

-- Define the prices and quantities for each company
def company_a_price : ℝ := 4
def company_b_price : ℝ := 3.5
def company_a_quantity : ℕ := 300
def company_b_quantity : ℕ := 350

-- Define the sales difference function
def sales_difference : ℝ :=
  (company_b_price * company_b_quantity) - (company_a_price * company_a_quantity)

-- Theorem statement
theorem sales_difference_is_25 : sales_difference = 25 := by
  sorry

end sales_difference_is_25_l2310_231014


namespace arithmetic_calculation_l2310_231076

theorem arithmetic_calculation : (1 + 2) * (3 - 4) + 5 = 2 := by
  sorry

end arithmetic_calculation_l2310_231076


namespace original_curve_equation_l2310_231043

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem original_curve_equation
  (x y x' y' : ℝ) -- Real variables for coordinates
  (h1 : x' = 5 * x) -- Scaling transformation for x
  (h2 : y' = 3 * y) -- Scaling transformation for y
  (h3 : 2 * x' ^ 2 + 8 * y' ^ 2 = 1) -- Equation of transformed curve
  : 50 * x ^ 2 + 72 * y ^ 2 = 1 := by
  sorry

end original_curve_equation_l2310_231043


namespace system_solution_ratio_l2310_231038

theorem system_solution_ratio (x y c d : ℝ) :
  (4 * x - 2 * y = c) →
  (6 * y - 12 * x = d) →
  d ≠ 0 →
  (∃ x y, (4 * x - 2 * y = c) ∧ (6 * y - 12 * x = d)) →
  c / d = -1 / 3 := by
sorry

end system_solution_ratio_l2310_231038


namespace A_power_difference_l2310_231025

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 0, 1]

theorem A_power_difference :
  A^20 - 2 • A^19 = !![0, 3; 0, -1] := by sorry

end A_power_difference_l2310_231025


namespace marco_strawberries_weight_l2310_231037

/-- The weight of Marco's strawberries in pounds -/
def marco_weight : ℝ := 37 - 22

/-- Theorem stating that Marco's strawberries weighed 15 pounds -/
theorem marco_strawberries_weight :
  marco_weight = 15 := by sorry

end marco_strawberries_weight_l2310_231037


namespace apples_left_l2310_231078

/-- Given that Mike picked 7.0 apples, Nancy ate 3.0 apples, and Keith picked 6.0 apples,
    prove that the number of apples left is 10.0. -/
theorem apples_left (mike_picked : ℝ) (nancy_ate : ℝ) (keith_picked : ℝ)
    (h1 : mike_picked = 7.0)
    (h2 : nancy_ate = 3.0)
    (h3 : keith_picked = 6.0) :
    mike_picked + keith_picked - nancy_ate = 10.0 := by
  sorry

end apples_left_l2310_231078


namespace x_intercept_of_specific_line_l2310_231065

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := sorry

/-- The line passing through (4, 6) and (8, 2) -/
def specific_line : Line := { x₁ := 4, y₁ := 6, x₂ := 8, y₂ := 2 }

theorem x_intercept_of_specific_line :
  x_intercept specific_line = 10 := by sorry

end x_intercept_of_specific_line_l2310_231065


namespace lamp_purchasing_problem_l2310_231022

/-- Represents a purchasing plan for energy-saving lamps -/
structure LampPlan where
  typeA : ℕ
  typeB : ℕ
  cost : ℕ

/-- Checks if a plan satisfies the given constraints -/
def isValidPlan (plan : LampPlan) : Prop :=
  plan.typeA + plan.typeB = 50 ∧
  2 * plan.typeB ≤ plan.typeA ∧
  plan.typeA ≤ 3 * plan.typeB

/-- Calculates the cost of a plan given the prices of lamps -/
def calculateCost (priceA priceB : ℕ) (plan : LampPlan) : ℕ :=
  priceA * plan.typeA + priceB * plan.typeB

/-- Main theorem to prove -/
theorem lamp_purchasing_problem :
  ∃ (priceA priceB : ℕ) (plans : List LampPlan),
    priceA + 3 * priceB = 26 ∧
    3 * priceA + 2 * priceB = 29 ∧
    priceA = 5 ∧
    priceB = 7 ∧
    plans.length = 4 ∧
    (∀ plan ∈ plans, isValidPlan plan) ∧
    (∃ bestPlan ∈ plans,
      bestPlan.typeA = 37 ∧
      bestPlan.typeB = 13 ∧
      calculateCost priceA priceB bestPlan = 276 ∧
      ∀ plan ∈ plans, calculateCost priceA priceB bestPlan ≤ calculateCost priceA priceB plan) :=
sorry

end lamp_purchasing_problem_l2310_231022


namespace partnership_gain_is_18000_l2310_231028

/-- Represents the annual gain of a partnership given the investments and one partner's share. -/
def partnership_annual_gain (x : ℚ) (a_share : ℚ) : ℚ :=
  let a_invest_time : ℚ := x * 12
  let b_invest_time : ℚ := 2 * x * 6
  let c_invest_time : ℚ := 3 * x * 4
  let total_invest_time : ℚ := a_invest_time + b_invest_time + c_invest_time
  (total_invest_time / a_invest_time) * a_share

/-- 
Given:
- A invests x at the beginning
- B invests 2x after 6 months
- C invests 3x after 8 months
- A's share of the gain is 6000

Prove that the total annual gain of the partnership is 18000.
-/
theorem partnership_gain_is_18000 (x : ℚ) (h : x > 0) :
  partnership_annual_gain x 6000 = 18000 := by
  sorry

end partnership_gain_is_18000_l2310_231028


namespace shirt_discount_l2310_231045

/-- Given a shirt with an original price and a sale price, calculate the discount amount. -/
def discount (original_price sale_price : ℕ) : ℕ :=
  original_price - sale_price

/-- Theorem stating that for a shirt with an original price of $22 and a sale price of $16, 
    the discount amount is $6. -/
theorem shirt_discount :
  let original_price : ℕ := 22
  let sale_price : ℕ := 16
  discount original_price sale_price = 6 := by
sorry

end shirt_discount_l2310_231045


namespace tent_max_profit_l2310_231019

/-- Represents the purchase and sales information for tents --/
structure TentInfo where
  regular_purchase_price : ℝ
  sunshade_purchase_price : ℝ
  regular_selling_price : ℝ
  sunshade_selling_price : ℝ
  total_budget : ℝ

/-- Represents the constraints on tent purchases --/
structure TentConstraints where
  min_regular_tents : ℕ
  regular_not_exceeding_sunshade : Bool

/-- Calculates the maximum profit given tent information and constraints --/
def max_profit (info : TentInfo) (constraints : TentConstraints) : ℝ :=
  sorry

/-- Theorem stating the maximum profit for the given scenario --/
theorem tent_max_profit :
  let info : TentInfo := {
    regular_purchase_price := 150,
    sunshade_purchase_price := 300,
    regular_selling_price := 180,
    sunshade_selling_price := 380,
    total_budget := 9000
  }
  let constraints : TentConstraints := {
    min_regular_tents := 12,
    regular_not_exceeding_sunshade := true
  }
  max_profit info constraints = 2280 := by sorry

end tent_max_profit_l2310_231019


namespace area_between_squares_l2310_231000

/-- The area of the region between two concentric squares -/
theorem area_between_squares (outer_side : ℝ) (inner_side : ℝ) 
  (h_outer : outer_side = 6) 
  (h_inner : inner_side = 4) :
  outer_side ^ 2 - inner_side ^ 2 = 20 := by
  sorry

end area_between_squares_l2310_231000


namespace max_profit_at_six_l2310_231099

/-- The profit function for a certain product -/
def profit_function (x : ℝ) : ℝ := -2 * x^3 + 18 * x^2

/-- The derivative of the profit function -/
def profit_derivative (x : ℝ) : ℝ := -6 * x^2 + 36 * x

theorem max_profit_at_six :
  ∃ (x : ℝ), x > 0 ∧
  (∀ (y : ℝ), y > 0 → profit_function y ≤ profit_function x) ∧
  x = 6 := by
  sorry

end max_profit_at_six_l2310_231099


namespace min_value_sum_of_fractions_l2310_231021

theorem min_value_sum_of_fractions (a b : ℤ) (h : a ≠ b) :
  (a^2 + b^2 : ℚ) / (a^2 - b^2) + (a^2 - b^2 : ℚ) / (a^2 + b^2) ≥ 2 :=
sorry

end min_value_sum_of_fractions_l2310_231021


namespace sum_of_logarithms_l2310_231041

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem sum_of_logarithms (a b : ℝ) (ha : (10 : ℝ) ^ a = 2) (hb : b = log10 5) :
  a + b = 1 := by
  sorry

end sum_of_logarithms_l2310_231041


namespace perpendicular_chords_theorem_l2310_231098

/-- 
Given a circle with radius R and two perpendicular chords intersecting at point M,
this theorem proves two properties:
1. The sum of squares of the four segments formed by the intersection is 4R^2.
2. If the distance from the center to M is d, the sum of squares of chord lengths is 8R^2 - 4d^2.
-/
theorem perpendicular_chords_theorem (R d : ℝ) (h : d ≥ 0) :
  ∃ (AM MB CM MD : ℝ),
    (AM ≥ 0) ∧ (MB ≥ 0) ∧ (CM ≥ 0) ∧ (MD ≥ 0) ∧
    (AM^2 + MB^2 + CM^2 + MD^2 = 4 * R^2) ∧
    ∃ (AB CD : ℝ),
      (AB ≥ 0) ∧ (CD ≥ 0) ∧
      (AB^2 + CD^2 = 8 * R^2 - 4 * d^2) := by
  sorry

end perpendicular_chords_theorem_l2310_231098


namespace polynomial_factorization_l2310_231061

theorem polynomial_factorization (m x y : ℝ) : 4*m*x^2 - m*y^2 = m*(2*x+y)*(2*x-y) := by
  sorry

end polynomial_factorization_l2310_231061


namespace expression_never_equals_33_l2310_231032

theorem expression_never_equals_33 (x y : ℤ) :
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end expression_never_equals_33_l2310_231032


namespace quadratic_inequality_l2310_231008

theorem quadratic_inequality (x : ℝ) : (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 := by
  sorry

end quadratic_inequality_l2310_231008


namespace symmetry_wrt_y_axis_l2310_231067

/-- Given a point P with coordinates (x, y), the point symmetrical to P
    with respect to the y-axis has coordinates (-x, y) -/
def symmetrical_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-(P.1), P.2)

theorem symmetry_wrt_y_axis :
  let P : ℝ × ℝ := (3, -5)
  symmetrical_point P = (-3, -5) := by
  sorry

end symmetry_wrt_y_axis_l2310_231067


namespace schedule_theorem_l2310_231069

/-- The number of periods in a day -/
def num_periods : ℕ := 7

/-- The number of courses to be scheduled -/
def num_courses : ℕ := 4

/-- Calculates the number of ways to schedule distinct courses in non-consecutive periods -/
def schedule_ways (periods : ℕ) (courses : ℕ) : ℕ :=
  (Nat.choose (periods - courses + 1) courses) * (Nat.factorial courses)

/-- Theorem stating that there are 1680 ways to schedule 4 distinct courses in a 7-period day
    with no two courses in consecutive periods -/
theorem schedule_theorem : 
  schedule_ways num_periods num_courses = 1680 := by
  sorry

end schedule_theorem_l2310_231069


namespace evaluate_expression_l2310_231011

theorem evaluate_expression : (10^8) / (2 * (10^5) * (1/2)) = 1000 := by
  sorry

end evaluate_expression_l2310_231011


namespace sum_of_specific_T_l2310_231017

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    (3 * n) / 2
  else
    (3 * n - 1) / 2

theorem sum_of_specific_T : T 18 + T 34 + T 51 = 154 := by
  sorry

end sum_of_specific_T_l2310_231017


namespace cannot_distinguish_normal_l2310_231035

/-- Represents the three types of people on the island -/
inductive PersonType
  | Knight
  | Liar
  | Normal

/-- Represents a statement that can be true or false -/
structure Statement where
  content : Prop

/-- A function that determines whether a given person type would make a given statement -/
def wouldMakeStatement (personType : PersonType) (statement : Statement) : Prop :=
  match personType with
  | PersonType.Knight => statement.content
  | PersonType.Liar => ¬statement.content
  | PersonType.Normal => True

/-- The main theorem stating that it's impossible to distinguish a normal person from a knight or liar with any finite number of statements -/
theorem cannot_distinguish_normal (n : ℕ) :
  ∃ (statements : Fin n → Statement),
    (∀ i, wouldMakeStatement PersonType.Normal (statements i)) ∧
    ((∀ i, wouldMakeStatement PersonType.Knight (statements i)) ∨
     (∀ i, wouldMakeStatement PersonType.Liar (statements i))) :=
sorry

end cannot_distinguish_normal_l2310_231035


namespace inverse_of_i_minus_two_i_inv_l2310_231080

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem inverse_of_i_minus_two_i_inv (h : i^2 = -1) :
  (i - 2 * i⁻¹)⁻¹ = -i / 3 := by
  sorry

end inverse_of_i_minus_two_i_inv_l2310_231080


namespace equation_solution_l2310_231087

theorem equation_solution : ∃ y : ℤ, (2010 + 2*y)^2 = 4*y^2 ∧ y = -1005 := by
  sorry

end equation_solution_l2310_231087


namespace loss_percentage_book1_l2310_231048

-- Define the total cost of both books
def total_cost : ℝ := 450

-- Define the cost of the first book (sold at a loss)
def cost_book1 : ℝ := 262.5

-- Define the gain percentage on the second book
def gain_percentage : ℝ := 0.19

-- Define the function to calculate the selling price of the second book
def selling_price_book2 (cost : ℝ) : ℝ := cost * (1 + gain_percentage)

-- Define the theorem
theorem loss_percentage_book1 : 
  let cost_book2 := total_cost - cost_book1
  let sp := selling_price_book2 cost_book2
  let loss_percentage := (cost_book1 - sp) / cost_book1 * 100
  loss_percentage = 15 := by sorry

end loss_percentage_book1_l2310_231048


namespace pizza_solution_l2310_231056

/-- The number of pizzas made by Craig and Heather over two days -/
def pizza_problem (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ) : Prop :=
  let total := craig_day1 + craig_day2 + heather_day1 + heather_day2
  craig_day1 = 40 ∧
  craig_day2 = craig_day1 + 60 ∧
  heather_day1 = 4 * craig_day1 ∧
  total = 380 ∧
  craig_day2 - heather_day2 = 20

theorem pizza_solution :
  ∃ (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ),
    pizza_problem craig_day1 craig_day2 heather_day1 heather_day2 :=
by
  sorry

end pizza_solution_l2310_231056


namespace ticket_revenue_calculation_l2310_231081

/-- Calculates the total revenue from ticket sales given the following conditions:
  * Child ticket cost: $6
  * Adult ticket cost: $9
  * Total tickets sold: 225
  * Number of adult tickets: 175
-/
theorem ticket_revenue_calculation (child_cost adult_cost total_tickets adult_tickets : ℕ) 
  (h1 : child_cost = 6)
  (h2 : adult_cost = 9)
  (h3 : total_tickets = 225)
  (h4 : adult_tickets = 175) :
  child_cost * (total_tickets - adult_tickets) + adult_cost * adult_tickets = 1875 :=
by sorry

end ticket_revenue_calculation_l2310_231081


namespace mixture_composition_l2310_231053

/-- Represents a solution with percentages of materials A and B -/
structure Solution :=
  (percentA : ℝ)
  (percentB : ℝ)
  (sum_to_100 : percentA + percentB = 100)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solutionX : Solution)
  (solutionY : Solution)
  (finalPercentA : ℝ)

theorem mixture_composition 
  (m : Mixture)
  (hX : m.solutionX.percentA = 20 ∧ m.solutionX.percentB = 80)
  (hY : m.solutionY.percentA = 30 ∧ m.solutionY.percentB = 70)
  (hFinal : m.finalPercentA = 22) :
  100 - m.finalPercentA = 78 := by
  sorry

end mixture_composition_l2310_231053


namespace hyperbola_focus_to_asymptote_distance_l2310_231085

/-- The distance from any focus of the hyperbola x^2 - y^2 = 1 to any of its asymptotes is 1 -/
theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), x^2 - y^2 = 1 →
  ∀ (fx : ℝ), fx^2 = 2 →
  ∀ (a b : ℝ), a^2 = 1 ∧ b^2 = 1 ∧ a * b = 0 →
  |a * fx + b * 0| / Real.sqrt (a^2 + b^2) = 1 :=
by sorry


end hyperbola_focus_to_asymptote_distance_l2310_231085


namespace difference_of_squares_l2310_231093

theorem difference_of_squares (a b : ℝ) : a^2 - 9*b^2 = (a + 3*b) * (a - 3*b) := by
  sorry

end difference_of_squares_l2310_231093


namespace smaller_root_of_equation_l2310_231052

theorem smaller_root_of_equation (x : ℝ) : 
  (x - 2/3) * (x - 2/3) + (x - 2/3) * (x - 1/3) = 0 →
  (x = 1/2 ∨ x = 2/3) ∧ 1/2 < 2/3 := by
sorry

end smaller_root_of_equation_l2310_231052


namespace exam_average_l2310_231006

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) : 
  n1 = 15 → 
  n2 = 10 → 
  avg1 = 75 / 100 → 
  avg2 = 90 / 100 → 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 81 / 100 := by
  sorry

end exam_average_l2310_231006


namespace seokjin_paper_left_l2310_231049

/-- Given the initial number of sheets, number of notebooks, and pages per notebook,
    calculate the remaining sheets of paper. -/
def remaining_sheets (initial_sheets : ℕ) (num_notebooks : ℕ) (pages_per_notebook : ℕ) : ℕ :=
  initial_sheets - (num_notebooks * pages_per_notebook)

/-- Theorem stating that given 100 initial sheets, 3 notebooks with 30 pages each,
    the remaining sheets is 10. -/
theorem seokjin_paper_left : remaining_sheets 100 3 30 = 10 := by
  sorry

end seokjin_paper_left_l2310_231049


namespace calculation_proof_l2310_231018

theorem calculation_proof (h1 : 9 + 3/4 = 9.75) (h2 : 975/100 = 9.75) (h3 : 0.142857 = 1/7) :
  4/7 * (9 + 3/4) + 9.75 * 2/7 + 0.142857 * 975/100 = 9.75 := by
  sorry

end calculation_proof_l2310_231018


namespace birds_optimal_speed_l2310_231084

/-- Represents the problem of finding the optimal speed for Mr. Bird's commute --/
theorem birds_optimal_speed (d : ℝ) (t : ℝ) : 
  d > 0 → -- distance is positive
  t > 0 → -- time is positive
  d = 50 * (t + 1/12) → -- equation for 50 mph
  d = 70 * (t - 1/12) → -- equation for 70 mph
  ∃ (speed : ℝ), 
    speed = d / t ∧ 
    speed = 70 ∧ 
    speed > 50 :=
by sorry

end birds_optimal_speed_l2310_231084


namespace time_to_install_remaining_windows_l2310_231070

/-- Calculates the time to install remaining windows -/
def timeToInstallRemaining (totalWindows installedWindows timePerWindow : ℕ) : ℕ :=
  (totalWindows - installedWindows) * timePerWindow

/-- Theorem: Time to install remaining windows is 18 hours -/
theorem time_to_install_remaining_windows :
  timeToInstallRemaining 9 6 6 = 18 := by
  sorry

end time_to_install_remaining_windows_l2310_231070


namespace quadratic_factorization_l2310_231095

theorem quadratic_factorization (x : ℝ) :
  ∃ (m n : ℤ), 6 * x^2 - 5 * x - 6 = (6 * x + m) * (x + n) ∧ m - n = 5 := by
  sorry

end quadratic_factorization_l2310_231095


namespace unique_triple_solution_l2310_231047

theorem unique_triple_solution (x y p : ℕ+) (h_prime : Nat.Prime p) 
  (h_p : p = x^2 + 1) (h_y : 2*p^2 = y^2 + 1) : 
  (x = 2 ∧ y = 7 ∧ p = 5) := by
  sorry

end unique_triple_solution_l2310_231047


namespace square_congruent_one_iff_one_or_minus_one_l2310_231055

theorem square_congruent_one_iff_one_or_minus_one (p : Nat) (hp : Prime p) :
  ∀ a : Nat, a^2 ≡ 1 [ZMOD p] ↔ a ≡ 1 [ZMOD p] ∨ a ≡ p - 1 [ZMOD p] := by
  sorry

end square_congruent_one_iff_one_or_minus_one_l2310_231055


namespace estimate_larger_than_original_l2310_231005

theorem estimate_larger_than_original (x y ε : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) : 
  (x + ε) - (y - ε) > x - y := by
  sorry

end estimate_larger_than_original_l2310_231005


namespace negation_of_existence_negation_of_quadratic_equation_l2310_231088

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 + x + 1 = 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≠ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_equation_l2310_231088


namespace consecutive_color_draw_probability_l2310_231096

def orange_chips : Nat := 4
def green_chips : Nat := 3
def blue_chips : Nat := 5
def total_chips : Nat := orange_chips + green_chips + blue_chips

def satisfying_arrangements : Nat := orange_chips.factorial * green_chips.factorial * blue_chips.factorial

theorem consecutive_color_draw_probability : 
  (satisfying_arrangements : ℚ) / (total_chips.factorial : ℚ) = 1 / 665280 := by
  sorry

end consecutive_color_draw_probability_l2310_231096


namespace cos_240_degrees_l2310_231057

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l2310_231057


namespace polygon_D_has_largest_area_l2310_231058

-- Define the basic shapes
def square_area : ℝ := 1
def isosceles_right_triangle_area : ℝ := 0.5
def parallelogram_area : ℝ := 1

-- Define the polygons
def polygon_A_area : ℝ := 3 * square_area + 2 * isosceles_right_triangle_area
def polygon_B_area : ℝ := 2 * square_area + 4 * isosceles_right_triangle_area
def polygon_C_area : ℝ := square_area + 2 * isosceles_right_triangle_area + parallelogram_area
def polygon_D_area : ℝ := 4 * square_area + parallelogram_area
def polygon_E_area : ℝ := 2 * square_area + 3 * isosceles_right_triangle_area + parallelogram_area

-- Theorem statement
theorem polygon_D_has_largest_area :
  polygon_D_area > polygon_A_area ∧
  polygon_D_area > polygon_B_area ∧
  polygon_D_area > polygon_C_area ∧
  polygon_D_area > polygon_E_area :=
by sorry

end polygon_D_has_largest_area_l2310_231058


namespace fence_cost_l2310_231050

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 57) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 3876 := by sorry

end fence_cost_l2310_231050
