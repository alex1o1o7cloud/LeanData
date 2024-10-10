import Mathlib

namespace min_value_quadratic_form_l451_45151

theorem min_value_quadratic_form (x y : ℝ) : x^2 + 3*x*y + y^2 ≥ 0 ∧ 
  (x^2 + 3*x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end min_value_quadratic_form_l451_45151


namespace quadratic_roots_and_isosceles_triangle_l451_45103

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a function to check if three sides form an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (c = a ∧ b ≠ c)

-- Statement of the theorem
theorem quadratic_roots_and_isosceles_triangle :
  (∀ k : ℝ, discriminant k > 0) ∧
  (∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧
    is_isosceles x y 4) → (k = 3 ∨ k = 4)) := by
  sorry

end quadratic_roots_and_isosceles_triangle_l451_45103


namespace pencils_added_l451_45158

theorem pencils_added (initial : ℕ) (final : ℕ) (h1 : initial = 115) (h2 : final = 215) :
  final - initial = 100 := by
sorry

end pencils_added_l451_45158


namespace only_C_not_proportional_l451_45119

-- Define the groups of line segments
def group_A : (ℚ × ℚ × ℚ × ℚ) := (3, 6, 2, 4)
def group_B : (ℚ × ℚ × ℚ × ℚ) := (1, 2, 2, 4)
def group_C : (ℚ × ℚ × ℚ × ℚ) := (4, 6, 5, 10)
def group_D : (ℚ × ℚ × ℚ × ℚ) := (1, 1/2, 1/6, 1/3)

-- Define a function to check if a group is proportional
def is_proportional (group : ℚ × ℚ × ℚ × ℚ) : Prop :=
  let (a, b, c, d) := group
  a / b = c / d

-- Theorem stating that only group C is not proportional
theorem only_C_not_proportional :
  is_proportional group_A ∧
  is_proportional group_B ∧
  ¬is_proportional group_C ∧
  is_proportional group_D :=
by sorry

end only_C_not_proportional_l451_45119


namespace logical_equivalence_l451_45125

variable (E W : Prop)

-- E: Pink elephants on planet α have purple eyes
-- W: Wild boars on planet β have long noses

theorem logical_equivalence :
  ((E → ¬W) ↔ (W → ¬E)) ∧ ((E → ¬W) ↔ (¬E ∨ ¬W)) := by sorry

end logical_equivalence_l451_45125


namespace adults_who_ate_correct_l451_45164

/-- Represents the number of adults who had their meal -/
def adults_who_ate : ℕ := 21

/-- The total number of adults -/
def total_adults : ℕ := 55

/-- The total number of children -/
def total_children : ℕ := 70

/-- The number of adults the meal can fully cater for -/
def meal_capacity_adults : ℕ := 70

/-- The number of children the meal can fully cater for -/
def meal_capacity_children : ℕ := 90

/-- The number of children that can be catered with the remaining food after some adults eat -/
def remaining_children_capacity : ℕ := 63

theorem adults_who_ate_correct :
  adults_who_ate * meal_capacity_children / meal_capacity_adults +
  remaining_children_capacity = meal_capacity_children :=
by sorry

end adults_who_ate_correct_l451_45164


namespace inverse_power_of_two_l451_45197

theorem inverse_power_of_two : 2⁻¹ = (1 : ℚ) / 2 := by sorry

end inverse_power_of_two_l451_45197


namespace number_of_team_formations_l451_45162

def male_athletes : ℕ := 5
def female_athletes : ℕ := 5
def team_size : ℕ := 6
def ma_long_selected : Prop := true
def ding_ning_selected : Prop := true

def remaining_male_athletes : ℕ := male_athletes - 1
def remaining_female_athletes : ℕ := female_athletes - 1
def remaining_slots : ℕ := team_size - 2

theorem number_of_team_formations :
  (Nat.choose remaining_male_athletes (remaining_slots / 2))^2 *
  (Nat.factorial remaining_slots) =
  number_of_ways_to_form_teams :=
sorry

end number_of_team_formations_l451_45162


namespace job_completion_time_l451_45136

def job_completion (x : ℝ) : Prop :=
  ∃ (y : ℝ),
    (1 / (x + 5) + 1 / (x + 3) + 1 / (2 * y) = 1 / x) ∧
    (1 / (x + 3) + 1 / y = 1 / x) ∧
    (y > 0) ∧ (x > 0)

theorem job_completion_time : ∃ (x : ℝ), job_completion x ∧ x = 3 := by
  sorry

end job_completion_time_l451_45136


namespace sum_remainder_zero_l451_45171

def arithmetic_sum (a₁ aₙ n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem sum_remainder_zero : 
  let a₁ := 6
  let d := 6
  let aₙ := 288
  let n := (aₙ - a₁) / d + 1
  (arithmetic_sum a₁ aₙ n) % 8 = 0 := by
sorry

end sum_remainder_zero_l451_45171


namespace coin_arrangements_count_l451_45172

/-- The number of indistinguishable gold coins -/
def num_gold_coins : ℕ := 5

/-- The number of indistinguishable silver coins -/
def num_silver_coins : ℕ := 5

/-- The total number of coins -/
def total_coins : ℕ := num_gold_coins + num_silver_coins

/-- The number of ways to arrange the gold and silver coins -/
def color_arrangements : ℕ := Nat.choose total_coins num_gold_coins

/-- The number of possible orientations to avoid face-to-face adjacency -/
def orientation_arrangements : ℕ := total_coins + 1

/-- The total number of distinguishable arrangements -/
def total_arrangements : ℕ := color_arrangements * orientation_arrangements

theorem coin_arrangements_count :
  total_arrangements = 2772 :=
sorry

end coin_arrangements_count_l451_45172


namespace slope_angle_range_l451_45146

/-- A line passing through the point (0, -2) and intersecting the unit circle -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (0, -2) -/
  passes_through_point : k * 0 - 2 = -2
  /-- The line intersects the unit circle -/
  intersects_circle : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y = k * x - 2

/-- The slope angle of a line -/
noncomputable def slope_angle (l : IntersectingLine) : ℝ :=
  Real.arctan l.k

/-- Theorem: The range of the slope angle for lines intersecting the unit circle and passing through (0, -2) -/
theorem slope_angle_range (l : IntersectingLine) :
  π/3 ≤ slope_angle l ∧ slope_angle l ≤ 2*π/3 :=
sorry

end slope_angle_range_l451_45146


namespace wang_loss_l451_45193

/-- Represents the financial transaction in Mr. Wang's store --/
structure Transaction where
  gift_cost : ℕ
  gift_price : ℕ
  payment : ℕ
  change_given : ℕ
  returned_to_neighbor : ℕ

/-- Calculates the loss in the transaction --/
def calculate_loss (t : Transaction) : ℕ :=
  t.change_given + t.gift_cost + t.returned_to_neighbor - t.payment

/-- Theorem stating that Mr. Wang's loss in the given transaction is $97 --/
theorem wang_loss (t : Transaction) 
  (h1 : t.gift_cost = 18)
  (h2 : t.gift_price = 21)
  (h3 : t.payment = 100)
  (h4 : t.change_given = 79)
  (h5 : t.returned_to_neighbor = 100) : 
  calculate_loss t = 97 := by
  sorry

#eval calculate_loss { gift_cost := 18, gift_price := 21, payment := 100, change_given := 79, returned_to_neighbor := 100 }

end wang_loss_l451_45193


namespace min_orders_is_three_l451_45153

/-- Represents the shopping problem with given conditions -/
structure ShoppingProblem where
  item_price : ℕ  -- Original price of each item in yuan
  item_count : ℕ  -- Number of items
  discount_rate : ℚ  -- Discount rate (e.g., 0.6 for 60% off)
  additional_discount_threshold : ℕ  -- Threshold for additional discount in yuan
  additional_discount_amount : ℕ  -- Amount of additional discount in yuan

/-- Calculates the total cost after discounts for a given number of orders -/
def total_cost (problem : ShoppingProblem) (num_orders : ℕ) : ℚ :=
  sorry

/-- Theorem stating that 3 is the minimum number of orders that minimizes the total cost -/
theorem min_orders_is_three (problem : ShoppingProblem) 
  (h1 : problem.item_price = 48)
  (h2 : problem.item_count = 42)
  (h3 : problem.discount_rate = 0.6)
  (h4 : problem.additional_discount_threshold = 300)
  (h5 : problem.additional_discount_amount = 100) :
  ∀ n : ℕ, n ≠ 3 → total_cost problem 3 ≤ total_cost problem n :=
sorry

end min_orders_is_three_l451_45153


namespace collinear_points_imply_a_values_l451_45152

-- Define the points A, B, and C in the plane
def A (a : ℝ) : ℝ × ℝ := (1, -a)
def B (a : ℝ) : ℝ × ℝ := (2, a^2)
def C (a : ℝ) : ℝ × ℝ := (3, a^3)

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Theorem statement
theorem collinear_points_imply_a_values (a : ℝ) :
  collinear (A a) (B a) (C a) → a = 0 ∨ a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 := by
  sorry

end collinear_points_imply_a_values_l451_45152


namespace x_plus_y_value_l451_45116

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : |y| + x - y = 12) : 
  x + y = 18/5 := by sorry

end x_plus_y_value_l451_45116


namespace total_weekly_eggs_l451_45137

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens supplied to Store A daily -/
def store_a_dozens : ℕ := 5

/-- The number of eggs supplied to Store B daily -/
def store_b_eggs : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem total_weekly_eggs : 
  (store_a_dozens * dozen + store_b_eggs) * days_in_week = 630 := by
  sorry

end total_weekly_eggs_l451_45137


namespace original_average_weight_l451_45148

/-- Given a group of students, prove that the original average weight was 28 kg -/
theorem original_average_weight
  (n : ℕ) -- number of original students
  (x : ℝ) -- original average weight
  (w : ℝ) -- weight of new student
  (y : ℝ) -- new average weight after admitting the new student
  (hn : n = 29)
  (hw : w = 13)
  (hy : y = 27.5)
  (h_new_avg : (n : ℝ) * x + w = (n + 1 : ℝ) * y) :
  x = 28 :=
sorry

end original_average_weight_l451_45148


namespace hexadecimal_to_decimal_l451_45180

/-- Given that the hexadecimal number (10k5)₆ (where k is a positive integer) 
    is equivalent to the decimal number 239, prove that k = 3. -/
theorem hexadecimal_to_decimal (k : ℕ+) : 
  (1 * 6^3 + k * 6 + 5) = 239 → k = 3 := by
  sorry

end hexadecimal_to_decimal_l451_45180


namespace banana_orange_equivalence_l451_45127

/-- The cost of fruits at Lola's Fruit Stand -/
structure FruitCost where
  banana_apple_ratio : ℚ  -- 4 bananas = 3 apples
  apple_orange_ratio : ℚ  -- 9 apples = 5 oranges

/-- The theorem stating the relationship between bananas and oranges -/
theorem banana_orange_equivalence (fc : FruitCost) 
  (h1 : fc.banana_apple_ratio = 4 / 3)
  (h2 : fc.apple_orange_ratio = 9 / 5) : 
  24 * (fc.apple_orange_ratio * fc.banana_apple_ratio) = 10 := by
  sorry

#check banana_orange_equivalence

end banana_orange_equivalence_l451_45127


namespace cindy_envelopes_l451_45143

def envelopes_problem (initial_envelopes : ℕ) (num_friends : ℕ) (envelopes_per_friend : ℕ) : Prop :=
  initial_envelopes - (num_friends * envelopes_per_friend) = 22

theorem cindy_envelopes : envelopes_problem 37 5 3 := by
  sorry

end cindy_envelopes_l451_45143


namespace probability_of_letter_in_mathematics_l451_45185

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

theorem probability_of_letter_in_mathematics :
  (mathematics.toList.toFinset.card : ℚ) / alphabet.card = 4 / 13 := by
  sorry

end probability_of_letter_in_mathematics_l451_45185


namespace unique_factorial_difference_divisibility_l451_45133

theorem unique_factorial_difference_divisibility :
  ∃! (x : ℕ), x > 0 ∧ (Nat.factorial x - Nat.factorial (x - 4)) / 29 = 1 :=
by
  -- The unique value is 8
  use 8
  -- Proof goes here
  sorry

end unique_factorial_difference_divisibility_l451_45133


namespace secretaries_working_hours_l451_45179

theorem secretaries_working_hours (t₁ t₂ t₃ : ℝ) : 
  t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 →  -- Ensuring positive working times
  t₂ = 2 * t₁ →               -- Ratio condition for t₂
  t₃ = 5 * t₁ →               -- Ratio condition for t₃
  t₃ = 75 →                   -- Longest working time
  t₁ + t₂ + t₃ = 120 :=       -- Combined total
by sorry


end secretaries_working_hours_l451_45179


namespace complement_of_M_l451_45128

def U : Set Int := {-1, -2, -3, -4}
def M : Set Int := {-2, -3}

theorem complement_of_M : U \ M = {-1, -4} := by
  sorry

end complement_of_M_l451_45128


namespace complex_calculation_l451_45138

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a - 4*b = 1 + 18*I := by
sorry

end complex_calculation_l451_45138


namespace health_risk_factors_l451_45147

theorem health_risk_factors (total_population : ℕ) 
  (prob_one_factor : ℚ) 
  (prob_two_factors : ℚ) 
  (prob_all_given_AB : ℚ) :
  prob_one_factor = 1/10 →
  prob_two_factors = 14/100 →
  prob_all_given_AB = 1/3 →
  total_population > 0 →
  ∃ (num_no_factors : ℕ) (num_not_A : ℕ),
    (num_no_factors : ℚ) / (num_not_A : ℚ) = 21/55 ∧
    num_no_factors + num_not_A = 76 :=
by sorry

end health_risk_factors_l451_45147


namespace not_necessarily_right_triangle_l451_45123

/-- Given a triangle ABC with sides a, b, c in the ratio 2:3:4, 
    it is not necessarily a right triangle -/
theorem not_necessarily_right_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (ratio : ∃ (k : ℝ), k > 0 ∧ a = 2*k ∧ b = 3*k ∧ c = 4*k) : 
  ¬(a^2 + b^2 = c^2) := by
  sorry

end not_necessarily_right_triangle_l451_45123


namespace exponent_multiplication_l451_45177

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l451_45177


namespace potato_sale_revenue_l451_45101

/-- Calculates the revenue from selling potatoes given the total weight, damaged weight, bag size, and price per bag. -/
def potato_revenue (total_weight damaged_weight bag_size price_per_bag : ℕ) : ℕ :=
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_size
  num_bags * price_per_bag

/-- Theorem stating that the revenue from selling potatoes under given conditions is $9144. -/
theorem potato_sale_revenue :
  potato_revenue 6500 150 50 72 = 9144 := by
  sorry

end potato_sale_revenue_l451_45101


namespace f_4_1981_l451_45102

/-- A function satisfying the given recursive properties -/
def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Power tower of 2 with given height -/
def power_tower_2 : ℕ → ℕ
| 0 => 1
| n + 1 => 2^(power_tower_2 n)

/-- The main theorem to prove -/
theorem f_4_1981 : f 4 1981 = power_tower_2 1984 - 3 := by
  sorry

end f_4_1981_l451_45102


namespace solve_system_l451_45150

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - 2 * y = 7) 
  (eq2 : x + 3 * y = 8) : 
  x = 37 / 11 := by
sorry

end solve_system_l451_45150


namespace custom_op_result_l451_45112

/-- Define the custom operation * -/
def custom_op (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1

/-- Theorem stating the result of the custom operation given the conditions -/
theorem custom_op_result (a b : ℝ) :
  (custom_op a b 3 5 = 15) →
  (custom_op a b 4 7 = 28) →
  (custom_op a b 1 1 = -11) := by
  sorry

end custom_op_result_l451_45112


namespace additional_money_needed_l451_45161

/-- The amount of money Michael has initially -/
def michaels_money : ℕ := 50

/-- The cost of the cake -/
def cake_cost : ℕ := 20

/-- The cost of the bouquet -/
def bouquet_cost : ℕ := 36

/-- The cost of the balloons -/
def balloon_cost : ℕ := 5

/-- The total cost of all items -/
def total_cost : ℕ := cake_cost + bouquet_cost + balloon_cost

/-- The theorem stating how much more money Michael needs -/
theorem additional_money_needed : total_cost - michaels_money = 11 := by
  sorry

end additional_money_needed_l451_45161


namespace largest_prime_divisor_of_factorial_sum_l451_45181

theorem largest_prime_divisor_of_factorial_sum (n : ℕ) : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14 * 2) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 13 + Nat.factorial 14 * 2) → q ≤ p :=
by sorry

end largest_prime_divisor_of_factorial_sum_l451_45181


namespace loan_b_more_cost_effective_l451_45115

/-- Calculates the total repayable amount for a loan -/
def totalRepayable (principal : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  principal + principal * interestRate * years

/-- Represents the loan options available to Mike -/
structure LoanOption where
  principal : ℝ
  interestRate : ℝ
  years : ℝ

/-- Theorem stating that Loan B is more cost-effective than Loan A -/
theorem loan_b_more_cost_effective (carPrice savings : ℝ) (loanA loanB : LoanOption) :
  carPrice = 35000 ∧
  savings = 5000 ∧
  loanA.principal = 25000 ∧
  loanA.interestRate = 0.07 ∧
  loanA.years = 5 ∧
  loanB.principal = 20000 ∧
  loanB.interestRate = 0.05 ∧
  loanB.years = 4 →
  totalRepayable loanB.principal loanB.interestRate loanB.years <
  totalRepayable loanA.principal loanA.interestRate loanA.years :=
by sorry

end loan_b_more_cost_effective_l451_45115


namespace bitcoin_transfer_theorem_l451_45117

/-- Represents the state of bitcoin holdings for the three businessmen -/
structure BitcoinState where
  sasha : ℕ
  pasha : ℕ
  arkasha : ℕ

/-- Performs the series of transfers described in the problem -/
def perform_transfers (initial : BitcoinState) : BitcoinState :=
  let state1 := BitcoinState.mk (initial.sasha - initial.pasha) (2 * initial.pasha) initial.arkasha
  let state2 := BitcoinState.mk (state1.sasha - initial.arkasha) state1.pasha (2 * initial.arkasha)
  let state3 := BitcoinState.mk (2 * state2.sasha) (state2.pasha - state2.sasha - state2.arkasha) (2 * state2.arkasha)
  BitcoinState.mk (state3.sasha + state3.sasha) (state3.pasha + state3.sasha) (state3.arkasha - state3.sasha - state3.pasha)

/-- The theorem stating the initial and final states of bitcoin holdings -/
theorem bitcoin_transfer_theorem (initial : BitcoinState) :
  initial.sasha = 13 ∧ initial.pasha = 7 ∧ initial.arkasha = 4 ↔
  let final := perform_transfers initial
  final.sasha = 8 ∧ final.pasha = 8 ∧ final.arkasha = 8 :=
sorry

end bitcoin_transfer_theorem_l451_45117


namespace sine_cosine_shift_l451_45114

open Real

theorem sine_cosine_shift (ω : ℝ) :
  (∀ x, sin (ω * (x + π / 3)) = cos (ω * x)) → ω = 3 / 2 := by
  sorry

end sine_cosine_shift_l451_45114


namespace eighth_grade_trip_contribution_l451_45131

theorem eighth_grade_trip_contribution (total : ℕ) (months : ℕ) 
  (h1 : total = 49685) 
  (h2 : months = 5) : 
  ∃ (students : ℕ) (contribution : ℕ), 
    students * contribution * months = total ∧ 
    students = 19 ∧ 
    contribution = 523 := by
sorry

end eighth_grade_trip_contribution_l451_45131


namespace arrangement_count_is_correct_l451_45140

/-- The number of ways to arrange 4 volunteers and 1 elder in a row, with the elder in the middle -/
def arrangementCount : ℕ := 24

/-- The number of volunteers -/
def numVolunteers : ℕ := 4

/-- The number of elders -/
def numElders : ℕ := 1

/-- The total number of people -/
def totalPeople : ℕ := numVolunteers + numElders

theorem arrangement_count_is_correct :
  arrangementCount = Nat.factorial numVolunteers := by
  sorry


end arrangement_count_is_correct_l451_45140


namespace hundred_thousand_scientific_notation_l451_45121

-- Define scientific notation
def scientific_notation (n : ℝ) (x : ℝ) (y : ℤ) : Prop :=
  n = x * (10 : ℝ) ^ y ∧ 1 ≤ x ∧ x < 10

-- Theorem statement
theorem hundred_thousand_scientific_notation :
  scientific_notation 100000 1 5 :=
by sorry

end hundred_thousand_scientific_notation_l451_45121


namespace perpendicular_bisector_equation_l451_45129

/-- Given two points M and N in the plane, this theorem states that
    the equation of the perpendicular bisector of line segment MN
    is x - y + 3 = 0. -/
theorem perpendicular_bisector_equation (M N : ℝ × ℝ) :
  M = (-1, 6) →
  N = (3, 2) →
  ∃ (f : ℝ → ℝ), 
    (∀ x y, f x = y ↔ x - y + 3 = 0) ∧
    (∀ p : ℝ × ℝ, f p.1 = p.2 ↔ 
      (p.1 - M.1)^2 + (p.2 - M.2)^2 = (p.1 - N.1)^2 + (p.2 - N.2)^2 ∧
      (p.1 - M.1) * (N.1 - M.1) + (p.2 - M.2) * (N.2 - M.2) = 0) :=
by sorry

end perpendicular_bisector_equation_l451_45129


namespace failed_both_subjects_percentage_l451_45182

def total_candidates : ℕ := 3000
def failed_english_percent : ℚ := 49 / 100
def failed_hindi_percent : ℚ := 36 / 100
def passed_english_alone : ℕ := 630

theorem failed_both_subjects_percentage :
  let passed_english_alone_percent : ℚ := passed_english_alone / total_candidates
  let passed_english_percent : ℚ := 1 - failed_english_percent
  let passed_hindi_percent : ℚ := 1 - failed_hindi_percent
  let passed_both_percent : ℚ := passed_english_percent - passed_english_alone_percent
  let passed_hindi_alone_percent : ℚ := passed_hindi_percent - passed_both_percent
  let failed_both_percent : ℚ := 1 - (passed_english_alone_percent + passed_hindi_alone_percent + passed_both_percent)
  failed_both_percent = 15 / 100 := by
  sorry

end failed_both_subjects_percentage_l451_45182


namespace count_distinct_five_digit_numbers_l451_45124

/-- The number of distinct five-digit numbers that can be formed by selecting 2 digits
    from the set of odd digits {1, 3, 5, 7, 9} and 3 digits from the set of even digits
    {0, 2, 4, 6, 8}. -/
def distinct_five_digit_numbers : ℕ :=
  let odd_digits : Finset ℕ := {1, 3, 5, 7, 9}
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  10560

/-- Theorem stating that the number of distinct five-digit numbers formed under the given
    conditions is equal to 10560. -/
theorem count_distinct_five_digit_numbers :
  distinct_five_digit_numbers = 10560 := by
  sorry

end count_distinct_five_digit_numbers_l451_45124


namespace next_multiple_age_sum_digits_l451_45132

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents the family with Joey, Chloe, and Zoe -/
structure Family where
  joey : Person
  chloe : Person
  zoe : Person

/-- Returns true if n is a multiple of m -/
def isMultiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The main theorem -/
theorem next_multiple_age_sum_digits (f : Family) : 
  f.zoe.age = 1 →
  f.joey.age = f.chloe.age + 1 →
  (∃ n : ℕ, n ≥ 1 ∧ n ≤ 9 ∧ isMultiple (f.chloe.age + n - 1) n) →
  (∀ m : ℕ, m < f.chloe.age - 1 → ¬isMultiple (f.chloe.age + m - 1) m) →
  (∃ k : ℕ, isMultiple (f.joey.age + k) (f.zoe.age + k) ∧ 
    (∀ j : ℕ, j < k → ¬isMultiple (f.joey.age + j) (f.zoe.age + j)) ∧
    sumOfDigits (f.joey.age + k) = 12) :=
sorry

end next_multiple_age_sum_digits_l451_45132


namespace justin_age_l451_45168

/-- Prove that Justin's age is 26 years -/
theorem justin_age :
  ∀ (justin_age jessica_age james_age : ℕ),
  (jessica_age = justin_age + 6) →
  (james_age = jessica_age + 7) →
  (james_age + 5 = 44) →
  justin_age = 26 := by
sorry

end justin_age_l451_45168


namespace difference_8_in_96348621_l451_45154

/-- The difference between the local value and the face value of a digit in a number -/
def localFaceDifference (n : ℕ) (d : ℕ) (p : ℕ) : ℕ :=
  d * (10 ^ p) - d

/-- The position of a digit in a number, counting from right to left and starting at 0 -/
def digitPosition (n : ℕ) (d : ℕ) : ℕ :=
  sorry -- Implementation not required for the statement

theorem difference_8_in_96348621 :
  localFaceDifference 96348621 8 (digitPosition 96348621 8) = 7992 := by
  sorry

end difference_8_in_96348621_l451_45154


namespace smallest_integer_fourth_root_l451_45174

theorem smallest_integer_fourth_root (p : ℕ) (q : ℕ) (s : ℝ) : 
  (0 < q) → 
  (0 < s) → 
  (s < 1 / 2000) → 
  (p^(1/4 : ℝ) = q + s) → 
  (∀ (p' : ℕ) (q' : ℕ) (s' : ℝ), 
    0 < q' → 0 < s' → s' < 1 / 2000 → p'^(1/4 : ℝ) = q' + s' → p' ≥ p) →
  q = 8 := by
sorry

end smallest_integer_fourth_root_l451_45174


namespace ratio_of_repeating_decimals_l451_45108

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

/-- The main theorem stating that the ratio of 0.overline{63} to 0.overline{21} is 3 -/
theorem ratio_of_repeating_decimals : 
  (RepeatingDecimal 6 3) / (RepeatingDecimal 2 1) = 3 := by sorry

end ratio_of_repeating_decimals_l451_45108


namespace glass_volume_l451_45169

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V) -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference between optimist's and pessimist's water volumes
  : V = 230 := by
sorry

end glass_volume_l451_45169


namespace equation_one_solution_equation_two_solutions_equation_three_solution_l451_45156

-- Equation 1
theorem equation_one_solution (x : ℝ) :
  (x^2 + 2) * |2*x - 5| = 0 ↔ x = 5/2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  (x - 3)^3 * x = 0 ↔ x = 0 ∨ x = 3 := by sorry

-- Equation 3
theorem equation_three_solution (x : ℝ) :
  |x^4 + 1| = x^4 + x ↔ x = 1 := by sorry

end equation_one_solution_equation_two_solutions_equation_three_solution_l451_45156


namespace log_101600_div_3_l451_45178

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_101600_div_3 : log (101600 / 3) = 0.1249 := by
  -- Given conditions
  have h1 : log 102 = 0.3010 := by sorry
  have h2 : log 3 = 0.4771 := by sorry

  -- Proof steps
  sorry

end log_101600_div_3_l451_45178


namespace probability_jack_or_queen_l451_45163

theorem probability_jack_or_queen (total_cards : ℕ) (jack_queen_count : ℕ) 
  (h1 : total_cards = 104) 
  (h2 : jack_queen_count = 16) : 
  (jack_queen_count : ℚ) / total_cards = 2 / 13 := by
  sorry

end probability_jack_or_queen_l451_45163


namespace max_value_of_f_l451_45198

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧  -- Minimum value is -2
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →  -- Minimum value is achieved
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) ∧   -- Maximum value is at most 1
  (∃ x ∈ Set.Icc 0 1, f a x = 1)     -- Maximum value 1 is achieved
  := by sorry

end max_value_of_f_l451_45198


namespace graph_is_pair_of_lines_l451_45166

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

/-- The graph is a pair of straight lines -/
theorem graph_is_pair_of_lines :
  ∃ f g : ℝ → ℝ,
    (is_straight_line f ∧ is_straight_line g) ∧
    ∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x) :=
sorry

end graph_is_pair_of_lines_l451_45166


namespace root_equation_implies_b_equals_four_l451_45195

theorem root_equation_implies_b_equals_four
  (a b c : ℕ)
  (ha : a > 1)
  (hb : b > 1)
  (hc : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → (N^3 * (N^2 * N^(1/c))^(1/b))^(1/a) = N^(39/48)) :
  b = 4 :=
sorry

end root_equation_implies_b_equals_four_l451_45195


namespace complex_simplification_and_multiplication_l451_45188

theorem complex_simplification_and_multiplication :
  3 * ((4 - 3*Complex.I) - (2 + 5*Complex.I)) = 6 - 24*Complex.I := by
  sorry

end complex_simplification_and_multiplication_l451_45188


namespace additional_telephone_lines_l451_45167

theorem additional_telephone_lines :
  (9 * 10^6 : ℕ) - (9 * 10^5 : ℕ) = 81 * 10^5 := by
  sorry

end additional_telephone_lines_l451_45167


namespace simplify_complex_fraction_l451_45149

def i : ℂ := Complex.I

theorem simplify_complex_fraction :
  (4 + 2 * i) / (4 - 2 * i) - (4 - 2 * i) / (4 + 2 * i) = 8 * i / 5 :=
by sorry

end simplify_complex_fraction_l451_45149


namespace square_root_equation_solution_l451_45170

theorem square_root_equation_solution :
  ∃ x : ℝ, (Real.sqrt 289 - Real.sqrt x / Real.sqrt 25 = 12) ∧ x = 625 := by
  sorry

end square_root_equation_solution_l451_45170


namespace cos_sin_sum_equals_sqrt3_over_2_l451_45184

theorem cos_sin_sum_equals_sqrt3_over_2 :
  Real.cos (6 * π / 180) * Real.cos (36 * π / 180) + 
  Real.sin (6 * π / 180) * Real.cos (54 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end cos_sin_sum_equals_sqrt3_over_2_l451_45184


namespace minji_water_intake_l451_45145

theorem minji_water_intake (morning_intake : Real) (afternoon_intake : Real)
  (h1 : morning_intake = 0.26)
  (h2 : afternoon_intake = 0.37) :
  morning_intake + afternoon_intake = 0.63 := by
sorry

end minji_water_intake_l451_45145


namespace arrangements_proof_l451_45199

def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3
def tasks : ℕ := 3

def arrangements_with_at_least_one_girl : ℕ :=
  Nat.choose total_people selected_people * Nat.factorial tasks -
  Nat.choose boys selected_people * Nat.factorial tasks

theorem arrangements_proof : arrangements_with_at_least_one_girl = 186 := by
  sorry

end arrangements_proof_l451_45199


namespace evelyn_bottle_caps_l451_45189

/-- The number of bottle caps Evelyn starts with -/
def initial_caps : ℕ := 18

/-- The number of bottle caps Evelyn finds -/
def found_caps : ℕ := 63

/-- The total number of bottle caps Evelyn ends up with -/
def total_caps : ℕ := initial_caps + found_caps

theorem evelyn_bottle_caps : total_caps = 81 := by
  sorry

end evelyn_bottle_caps_l451_45189


namespace deal_or_no_deal_elimination_l451_45141

/-- Represents the game setup and elimination process -/
structure DealOrNoDeal where
  totalBoxes : Nat
  highValueBoxes : Nat
  eliminatedBoxes : Nat

/-- Checks if the chance of holding a high-value box is at least 1/2 -/
def hasAtLeastHalfChance (game : DealOrNoDeal) : Prop :=
  let remainingBoxes := game.totalBoxes - game.eliminatedBoxes
  2 * game.highValueBoxes ≥ remainingBoxes

/-- The main theorem to prove -/
theorem deal_or_no_deal_elimination 
  (game : DealOrNoDeal) 
  (h1 : game.totalBoxes = 26)
  (h2 : game.highValueBoxes = 6)
  (h3 : game.eliminatedBoxes = 15) : 
  hasAtLeastHalfChance game :=
sorry

end deal_or_no_deal_elimination_l451_45141


namespace no_solution_inequality_l451_45159

theorem no_solution_inequality :
  ¬∃ (x : ℝ), (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4 :=
by sorry

end no_solution_inequality_l451_45159


namespace base8_52_equals_base10_42_l451_45192

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

theorem base8_52_equals_base10_42 :
  base8ToBase10 [5, 2] = 42 := by
  sorry

end base8_52_equals_base10_42_l451_45192


namespace line_tangent_to_circle_l451_45196

/-- The line y = a is tangent to the circle x^2 + y^2 - 2y = 0 if and only if a = 0 or a = 2 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, y = a → x^2 + y^2 - 2*y = 0 → (x = 0 ∧ (y = a + 1 ∨ y = a - 1))) ↔ (a = 0 ∨ a = 2) := by
sorry

end line_tangent_to_circle_l451_45196


namespace frame_width_is_five_l451_45113

/-- Represents a frame with square openings -/
structure SquareFrame where
  numOpenings : ℕ
  openingPerimeter : ℝ
  totalPerimeter : ℝ

/-- Calculates the width of the frame -/
def frameWidth (frame : SquareFrame) : ℝ :=
  sorry

/-- Theorem stating that for a frame with 3 square openings, 
    an opening perimeter of 60 cm, and a total perimeter of 180 cm, 
    the frame width is 5 cm -/
theorem frame_width_is_five :
  let frame : SquareFrame := {
    numOpenings := 3,
    openingPerimeter := 60,
    totalPerimeter := 180
  }
  frameWidth frame = 5 := by sorry

end frame_width_is_five_l451_45113


namespace teairra_clothing_count_l451_45111

/-- The number of shirts and pants Teairra has which are neither plaid nor purple -/
def non_plaid_purple_count (total_shirts : ℕ) (total_pants : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) : ℕ :=
  (total_shirts - plaid_shirts) + (total_pants - purple_pants)

/-- Theorem stating that Teairra has 21 items that are neither plaid nor purple -/
theorem teairra_clothing_count : non_plaid_purple_count 5 24 3 5 = 21 := by
  sorry

end teairra_clothing_count_l451_45111


namespace arithmetic_sequence_sine_problem_l451_45109

theorem arithmetic_sequence_sine_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 = 10 * Real.pi / 3 →                    -- given condition
  Real.sin (a 4 + a 7) = -Real.sqrt 3 / 2 :=        -- conclusion to prove
by sorry

end arithmetic_sequence_sine_problem_l451_45109


namespace polynomial_root_implies_k_value_l451_45186

theorem polynomial_root_implies_k_value : 
  ∀ k : ℚ, (3 : ℚ)^3 + 7*(3 : ℚ)^2 + k*(3 : ℚ) + 23 = 0 → k = -113/3 := by
  sorry

end polynomial_root_implies_k_value_l451_45186


namespace fourth_term_is_eleven_l451_45173

/-- A sequence of 5 terms with specific properties -/
def CanSequence (a : Fin 5 → ℕ) : Prop :=
  a 0 = 2 ∧ 
  a 1 = 4 ∧ 
  a 2 = 7 ∧ 
  a 4 = 16 ∧
  ∀ i : Fin 3, (a (i + 1) - a i) - (a (i + 2) - a (i + 1)) = 1

theorem fourth_term_is_eleven (a : Fin 5 → ℕ) (h : CanSequence a) : a 3 = 11 := by
  sorry

end fourth_term_is_eleven_l451_45173


namespace complex_number_and_pure_imaginary_l451_45187

-- Define the complex number z
def z : ℂ := sorry

-- Define the real number m
def m : ℝ := sorry

-- Theorem statement
theorem complex_number_and_pure_imaginary :
  (Complex.abs z = Real.sqrt 2) ∧
  (z.im = 1) ∧
  (z.re < 0) ∧
  (z = -1 + Complex.I) ∧
  (∃ (k : ℝ), m^2 + m + m * z^2 = k * Complex.I) →
  (z = -1 + Complex.I) ∧ (m = -1) := by
  sorry

end complex_number_and_pure_imaginary_l451_45187


namespace equal_division_of_money_l451_45135

theorem equal_division_of_money (total_amount : ℚ) (num_people : ℕ) 
  (h1 : total_amount = 5.25) (h2 : num_people = 7) :
  total_amount / num_people = 0.75 := by
  sorry

end equal_division_of_money_l451_45135


namespace division_problem_l451_45126

theorem division_problem (x : ℝ) : 45 / x = 900 → x = 0.05 := by
  sorry

end division_problem_l451_45126


namespace line_parallel_properties_l451_45191

-- Define the structure for a line
structure Line where
  slope : ℝ
  angle : ℝ

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop :=
  l1.angle = l2.angle

-- Theorem statement
theorem line_parallel_properties (l1 l2 : Line) :
  (l1.slope = l2.slope → parallel l1 l2) ∧
  (l1.angle = l2.angle → parallel l1 l2) ∧
  (parallel l1 l2 → l1.angle = l2.angle) :=
sorry

end line_parallel_properties_l451_45191


namespace polynomial_identities_identity1_identity2_identity3_identity4_l451_45155

-- Define the polynomial identities
theorem polynomial_identities (a b : ℝ) :
  ((a + b) * (a^2 - a*b + b^2) = a^3 + b^3) ∧
  ((a - b) * (a^2 + a*b + b^2) = a^3 - b^3) ∧
  ((a + 2*b) * (a^2 - 2*a*b + 4*b^2) = a^3 + 8*b^3) ∧
  (a^3 - 8 = (a - 2) * (a^2 + 2*a + 4)) :=
by sorry

-- Prove each identity separately
theorem identity1 (a b : ℝ) : (a + b) * (a^2 - a*b + b^2) = a^3 + b^3 :=
by sorry

theorem identity2 (a b : ℝ) : (a - b) * (a^2 + a*b + b^2) = a^3 - b^3 :=
by sorry

theorem identity3 (a b : ℝ) : (a + 2*b) * (a^2 - 2*a*b + 4*b^2) = a^3 + 8*b^3 :=
by sorry

theorem identity4 (a : ℝ) : a^3 - 8 = (a - 2) * (a^2 + 2*a + 4) :=
by sorry

end polynomial_identities_identity1_identity2_identity3_identity4_l451_45155


namespace circle_symmetry_max_k_l451_45120

/-- Given a circle C with center (a,b) and radius 2 passing through (0,2),
    and a line 2x-ky-k=0 with respect to which two points on C are symmetric,
    the maximum value of k is 4√5/5 -/
theorem circle_symmetry_max_k :
  ∀ (a b k : ℝ),
  (a^2 + (b-2)^2 = 4) →  -- circle equation passing through (0,2)
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ((x₁ - a)^2 + (y₁ - b)^2 = 4) ∧  -- point 1 on circle
    ((x₂ - a)^2 + (y₂ - b)^2 = 4) ∧  -- point 2 on circle
    (2*((x₁ + x₂)/2) - k*((y₁ + y₂)/2) - k = 0) ∧  -- midpoint on line
    (2*a - k*b - k = 0)) →  -- line passes through center
  k ≤ 4 * Real.sqrt 5 / 5 :=
sorry

end circle_symmetry_max_k_l451_45120


namespace solve_bag_problem_l451_45165

def bag_problem (total_balls : ℕ) (prob_two_red : ℚ) (red_balls : ℕ) : Prop :=
  total_balls = 10 ∧
  prob_two_red = 1 / 15 ∧
  (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) = prob_two_red

theorem solve_bag_problem :
  ∃ (red_balls : ℕ), bag_problem 10 (1 / 15) red_balls ∧ red_balls = 3 :=
sorry

end solve_bag_problem_l451_45165


namespace fraction_value_preservation_l451_45194

theorem fraction_value_preservation (original_numerator original_denominator increase_numerator : ℕ) 
  (h1 : original_numerator = 3)
  (h2 : original_denominator = 16)
  (h3 : increase_numerator = 6) :
  ∃ (increase_denominator : ℕ),
    (original_numerator + increase_numerator) / (original_denominator + increase_denominator) = 
    original_numerator / original_denominator ∧ 
    increase_denominator = 32 := by
  sorry

end fraction_value_preservation_l451_45194


namespace rectangles_combinable_l451_45105

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a square divided into four rectangles -/
structure DividedSquare where
  side : ℝ
  r1 : Rectangle
  r2 : Rectangle
  r3 : Rectangle
  r4 : Rectangle

/-- Assumption that the sum of areas of two non-adjacent rectangles equals the sum of areas of the other two -/
def equal_area_pairs (s : DividedSquare) : Prop :=
  area s.r1 + area s.r3 = area s.r2 + area s.r4

/-- The theorem to be proved -/
theorem rectangles_combinable (s : DividedSquare) (h : equal_area_pairs s) :
  (s.r1.width = s.r3.width ∨ s.r1.height = s.r3.height) :=
sorry

end rectangles_combinable_l451_45105


namespace smallest_product_l451_45107

def S : Finset Int := {-7, -5, -1, 1, 3}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y = -21 ∧ ∀ (c d : Int), c ∈ S → d ∈ S → x * y ≤ c * d :=
sorry

end smallest_product_l451_45107


namespace point_on_line_l451_45106

/-- A line in the xy-plane with slope m and y-intercept b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Theorem: For a line with slope 4 and y-intercept 4, 
    the point (199, 800) lies on this line -/
theorem point_on_line : 
  let l : Line := { m := 4, b := 4 }
  let p : Point := { x := 199, y := 800 }
  p.onLine l := by sorry

end point_on_line_l451_45106


namespace circle_tangency_l451_45118

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (m : ℝ) : 
  externally_tangent (0, 0) (2, 4) (Real.sqrt 5) (Real.sqrt (20 + m)) → m = -15 := by
  sorry

#check circle_tangency

end circle_tangency_l451_45118


namespace regression_lines_intersection_l451_45104

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passes_through (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : passes_through l₁ s t)
  (h₂ : passes_through l₂ s t) :
  ∃ (x y : ℝ), passes_through l₁ x y ∧ passes_through l₂ x y ∧ x = s ∧ y = t :=
sorry

end regression_lines_intersection_l451_45104


namespace optimal_stamp_combination_l451_45110

/-- The minimum number of stamps needed to make 50 cents using only 5-cent and 7-cent stamps -/
def min_stamps : ℕ := 8

/-- The number of 5-cent stamps used in the optimal solution -/
def num_5cent : ℕ := 3

/-- The number of 7-cent stamps used in the optimal solution -/
def num_7cent : ℕ := 5

theorem optimal_stamp_combination :
  (∀ x y : ℕ, 5 * x + 7 * y = 50 → x + y ≥ min_stamps) ∧
  5 * num_5cent + 7 * num_7cent = 50 ∧
  num_5cent + num_7cent = min_stamps := by
  sorry

end optimal_stamp_combination_l451_45110


namespace max_volume_triangular_pyramid_l451_45100

/-- A triangular pyramid P-ABC with given side lengths -/
structure TriangularPyramid where
  PA : ℝ
  PB : ℝ
  AB : ℝ
  BC : ℝ
  CA : ℝ

/-- The volume of a triangular pyramid -/
def volume (t : TriangularPyramid) : ℝ := sorry

/-- The maximum volume of a triangular pyramid with specific side lengths -/
theorem max_volume_triangular_pyramid :
  ∀ t : TriangularPyramid,
  t.PA = 3 ∧ t.PB = 3 ∧ t.AB = 2 ∧ t.BC = 2 ∧ t.CA = 2 →
  volume t ≤ 2 * Real.sqrt 6 / 3 :=
sorry

end max_volume_triangular_pyramid_l451_45100


namespace polygon_E_largest_area_l451_45190

-- Define the polygons and their areas
def polygon_A_area : ℝ := 4
def polygon_B_area : ℝ := 4.5
def polygon_C_area : ℝ := 4.5
def polygon_D_area : ℝ := 5
def polygon_E_area : ℝ := 5.5

-- Define a function to compare areas
def has_largest_area (x y z w v : ℝ) : Prop :=
  v ≥ x ∧ v ≥ y ∧ v ≥ z ∧ v ≥ w

-- Theorem statement
theorem polygon_E_largest_area :
  has_largest_area polygon_A_area polygon_B_area polygon_C_area polygon_D_area polygon_E_area :=
sorry

end polygon_E_largest_area_l451_45190


namespace percentage_green_tiles_l451_45142

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 10
def tiles_per_sqft : ℝ := 4
def green_tile_cost : ℝ := 3
def red_tile_cost : ℝ := 1.5
def total_cost : ℝ := 2100

theorem percentage_green_tiles :
  let total_area : ℝ := courtyard_length * courtyard_width
  let total_tiles : ℝ := total_area * tiles_per_sqft
  let green_tiles : ℝ := (total_cost - red_tile_cost * total_tiles) / (green_tile_cost - red_tile_cost)
  (green_tiles / total_tiles) * 100 = 40 := by
sorry

end percentage_green_tiles_l451_45142


namespace three_numbers_sum_l451_45160

theorem three_numbers_sum (x y z M : ℚ) : 
  x + y + z = 48 ∧ 
  x - 5 = M ∧ 
  y + 9 = M ∧ 
  z / 5 = M → 
  M = 52 / 7 := by
sorry

end three_numbers_sum_l451_45160


namespace sum_equals_350_l451_45157

theorem sum_equals_350 : 247 + 53 + 47 + 3 = 350 := by
  sorry

end sum_equals_350_l451_45157


namespace expression_evaluation_l451_45134

theorem expression_evaluation (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) :
  1 - (a^2 + 2*a*b + b^2) / (a^2 - a*b) / ((a + b) / (a - b)) = -1 :=
by sorry

end expression_evaluation_l451_45134


namespace flyers_left_to_hand_out_l451_45139

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed_out : ℕ) 
  (rose_handed_out : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed_out = 120)
  (h3 : rose_handed_out = 320) :
  total_flyers - (jack_handed_out + rose_handed_out) = 796 :=
by sorry

end flyers_left_to_hand_out_l451_45139


namespace expand_product_l451_45130

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 2*x + 5) = x^4 + 2*x^3 - 4*x^2 - 18*x - 45 := by
  sorry

end expand_product_l451_45130


namespace power_mod_nineteen_l451_45144

theorem power_mod_nineteen : 11^2048 ≡ 16 [MOD 19] := by
  sorry

end power_mod_nineteen_l451_45144


namespace cooking_and_weaving_count_l451_45122

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cooking_only : ℕ
  cooking_and_yoga : ℕ
  all_curriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem cooking_and_weaving_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 25)
  (h2 : cp.cooking = 15)
  (h3 : cp.weaving = 8)
  (h4 : cp.cooking_only = 2)
  (h5 : cp.cooking_and_yoga = 7)
  (h6 : cp.all_curriculums = 3) :
  cp.cooking - cp.cooking_only - (cp.cooking_and_yoga - cp.all_curriculums) = 9 := by
  sorry


end cooking_and_weaving_count_l451_45122


namespace optimal_transport_solution_l451_45183

/-- Represents a vehicle type with its carrying capacity and freight cost. -/
structure VehicleType where
  capacity : ℕ
  cost : ℕ

/-- Represents the transportation problem. -/
structure TransportProblem where
  totalVegetables : ℕ
  totalVehicles : ℕ
  vehicleA : VehicleType
  vehicleB : VehicleType
  vehicleC : VehicleType

/-- Represents a solution to the transportation problem. -/
structure TransportSolution where
  numA : ℕ
  numB : ℕ
  numC : ℕ
  totalCost : ℕ

/-- Checks if a solution is valid for a given problem. -/
def isValidSolution (problem : TransportProblem) (solution : TransportSolution) : Prop :=
  solution.numA + solution.numB + solution.numC = problem.totalVehicles ∧
  solution.numA * problem.vehicleA.capacity +
  solution.numB * problem.vehicleB.capacity +
  solution.numC * problem.vehicleC.capacity ≥ problem.totalVegetables ∧
  solution.totalCost = solution.numA * problem.vehicleA.cost +
                       solution.numB * problem.vehicleB.cost +
                       solution.numC * problem.vehicleC.cost

/-- Theorem stating the optimal solution for the given problem. -/
theorem optimal_transport_solution (problem : TransportProblem)
  (h1 : problem.totalVegetables = 240)
  (h2 : problem.totalVehicles = 16)
  (h3 : problem.vehicleA = ⟨10, 800⟩)
  (h4 : problem.vehicleB = ⟨16, 1000⟩)
  (h5 : problem.vehicleC = ⟨20, 1200⟩) :
  ∃ (solution : TransportSolution),
    isValidSolution problem solution ∧
    solution.numA = 4 ∧
    solution.numB = 10 ∧
    solution.numC = 2 ∧
    solution.totalCost = 15600 ∧
    (∀ (otherSolution : TransportSolution),
      isValidSolution problem otherSolution →
      otherSolution.totalCost ≥ solution.totalCost) :=
sorry

end optimal_transport_solution_l451_45183


namespace equation_solutions_l451_45176

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ > 0 ∧ x₂ > 0) ∧
    (∀ (x : ℝ), x > 0 → 
      ((1/3) * (4*x^2 - 1) = (x^2 - 60*x - 12) * (x^2 + 30*x + 6)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = 30 + Real.sqrt 905 ∧
    x₂ = -15 + 4 * Real.sqrt 14 ∧
    4 * Real.sqrt 14 > 15 :=
by sorry

end equation_solutions_l451_45176


namespace sum_of_x_y_z_l451_45175

theorem sum_of_x_y_z (x y z : ℝ) : y = 3*x → z = 2*y → x + y + z = 10*x := by
  sorry

end sum_of_x_y_z_l451_45175
