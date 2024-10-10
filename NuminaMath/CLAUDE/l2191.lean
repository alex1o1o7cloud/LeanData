import Mathlib

namespace boxes_with_neither_l2191_219123

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ)
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : crayons = 4)
  (h4 : both = 3) :
  total - (markers + crayons - both) = 6 := by
sorry

end boxes_with_neither_l2191_219123


namespace initial_red_marbles_l2191_219112

theorem initial_red_marbles (r g : ℕ) : 
  r * 3 = g * 5 → 
  (r - 18) * 4 = g + 27 → 
  r = 29 := by
sorry

end initial_red_marbles_l2191_219112


namespace max_container_weight_l2191_219107

def can_transport (k : ℕ) : Prop :=
  ∀ (distribution : List ℕ),
    (distribution.sum = 1500) →
    (∀ x ∈ distribution, x ≤ k ∧ x > 0) →
    ∃ (platform_loads : List ℕ),
      (platform_loads.length = 25) ∧
      (∀ load ∈ platform_loads, load ≤ 80) ∧
      (platform_loads.sum = 1500)

theorem max_container_weight :
  (can_transport 26) ∧ ¬(can_transport 27) := by sorry

end max_container_weight_l2191_219107


namespace intersection_range_l2191_219144

/-- The line equation y = a(x + 2) -/
def line (a x : ℝ) : ℝ := a * (x + 2)

/-- The curve equation x^2 - y|y| = 1 -/
def curve (x y : ℝ) : Prop := x^2 - y * abs y = 1

/-- The number of intersection points between the line and the curve -/
def intersection_count (a : ℝ) : ℕ := sorry

/-- The theorem stating the range of a for exactly 2 intersection points -/
theorem intersection_range :
  ∀ a : ℝ, intersection_count a = 2 ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
sorry

end intersection_range_l2191_219144


namespace min_rooms_in_apartment_l2191_219126

/-- Represents an apartment with rooms and doors. -/
structure Apartment where
  rooms : ℕ
  doors : ℕ
  at_most_one_door_between_rooms : Bool
  at_most_one_door_to_outside : Bool

/-- Checks if the apartment configuration is valid. -/
def is_valid_apartment (a : Apartment) : Prop :=
  a.at_most_one_door_between_rooms ∧
  a.at_most_one_door_to_outside ∧
  a.doors = 12

/-- Theorem: The minimum number of rooms in a valid apartment is 5. -/
theorem min_rooms_in_apartment (a : Apartment) 
  (h : is_valid_apartment a) : a.rooms ≥ 5 := by
  sorry

#check min_rooms_in_apartment

end min_rooms_in_apartment_l2191_219126


namespace geometric_sequence_fifth_term_l2191_219158

/-- Given a geometric sequence {a_n}, prove that if a_1 = 2 and a_9 = 8, then a_5 = 4 -/
theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) 
  (h_first : a 1 = 2) 
  (h_ninth : a 9 = 8) : 
  a 5 = 4 := by
sorry

end geometric_sequence_fifth_term_l2191_219158


namespace joan_has_nine_balloons_l2191_219133

/-- The number of blue balloons that Sally has -/
def sally_balloons : ℕ := 5

/-- The number of blue balloons that Jessica has -/
def jessica_balloons : ℕ := 2

/-- The total number of blue balloons -/
def total_balloons : ℕ := 16

/-- The number of blue balloons that Joan has -/
def joan_balloons : ℕ := total_balloons - (sally_balloons + jessica_balloons)

theorem joan_has_nine_balloons : joan_balloons = 9 := by
  sorry

end joan_has_nine_balloons_l2191_219133


namespace cistern_emptying_time_l2191_219176

/-- Represents the cistern problem -/
theorem cistern_emptying_time 
  (volume : ℝ) 
  (time_with_tap : ℝ) 
  (tap_rate : ℝ) 
  (h1 : volume = 480) 
  (h2 : time_with_tap = 24) 
  (h3 : tap_rate = 4) : 
  (volume / (volume / time_with_tap - tap_rate) = 30) := by
  sorry

#check cistern_emptying_time

end cistern_emptying_time_l2191_219176


namespace restaurant_bill_share_l2191_219194

/-- Calculate each person's share of a restaurant bill with tip -/
theorem restaurant_bill_share 
  (total_bill : ℝ) 
  (num_people : ℕ) 
  (tip_percentage : ℝ) 
  (h1 : total_bill = 211)
  (h2 : num_people = 5)
  (h3 : tip_percentage = 0.15) : 
  (total_bill * (1 + tip_percentage)) / num_people = 48.53 := by
sorry

end restaurant_bill_share_l2191_219194


namespace equal_cost_at_60_messages_l2191_219130

/-- Represents a text messaging plan with a per-message cost and a monthly fee. -/
structure TextPlan where
  perMessageCost : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages under a specific plan. -/
def totalCost (plan : TextPlan) (messages : ℚ) : ℚ :=
  plan.perMessageCost * messages + plan.monthlyFee

/-- The number of messages at which all plans have the same cost. -/
def equalCostMessages (planA planB planC : TextPlan) : ℚ :=
  60

theorem equal_cost_at_60_messages (planA planB planC : TextPlan) 
    (hA : planA = ⟨0.25, 9⟩) 
    (hB : planB = ⟨0.40, 0⟩)
    (hC : planC = ⟨0.20, 12⟩) : 
    let messages := equalCostMessages planA planB planC
    totalCost planA messages = totalCost planB messages ∧ 
    totalCost planA messages = totalCost planC messages :=
  sorry

end equal_cost_at_60_messages_l2191_219130


namespace smallest_two_digit_with_digit_product_12_l2191_219167

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end smallest_two_digit_with_digit_product_12_l2191_219167


namespace quadratic_equation_roots_l2191_219188

theorem quadratic_equation_roots (a b c : ℝ) (h : a = 1 ∧ b = -8 ∧ c = 16) :
  ∃! x : ℝ, x^2 - 8*x + 16 = 0 := by
  sorry

end quadratic_equation_roots_l2191_219188


namespace parabola_directrix_l2191_219182

/-- Given a parabola with equation y = ax^2 and directrix y = 1, prove that a = -1/4 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Condition 1: Equation of the parabola
  (∃ y : ℝ, y = 1 ∧ ∀ x : ℝ, y ≠ a * x^2) →  -- Condition 2: Equation of the directrix
  a = -1/4 := by
sorry

end parabola_directrix_l2191_219182


namespace test_probabilities_l2191_219115

/-- Probability of A passing the test -/
def prob_A : ℝ := 0.8

/-- Probability of B passing the test -/
def prob_B : ℝ := 0.6

/-- Probability of C passing the test -/
def prob_C : ℝ := 0.5

/-- Probability that all three pass the test -/
def prob_all_pass : ℝ := prob_A * prob_B * prob_C

/-- Probability that at least one passes the test -/
def prob_at_least_one_pass : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem test_probabilities :
  prob_all_pass = 0.24 ∧ prob_at_least_one_pass = 0.96 := by
  sorry

end test_probabilities_l2191_219115


namespace stating_chess_tournament_players_l2191_219151

/-- The number of players in a chess tournament -/
def num_players : ℕ := 19

/-- The total number of games played in the tournament -/
def total_games : ℕ := 342

/-- 
Theorem stating that the number of players in the chess tournament is correct,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  2 * num_players * (num_players - 1) = total_games := by
  sorry

end stating_chess_tournament_players_l2191_219151


namespace fixed_fee_calculation_l2191_219156

theorem fixed_fee_calculation (feb_bill march_bill : ℝ) 
  (h : feb_bill = 18.72 ∧ march_bill = 33.78) :
  ∃ (fixed_fee hourly_rate : ℝ),
    fixed_fee + hourly_rate = feb_bill ∧
    fixed_fee + 3 * hourly_rate = march_bill ∧
    fixed_fee = 11.19 := by
sorry

end fixed_fee_calculation_l2191_219156


namespace solution_satisfies_equations_l2191_219192

theorem solution_satisfies_equations :
  let x : ℚ := 67 / 9
  let y : ℚ := 22 / 3
  (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 8) := by
  sorry

end solution_satisfies_equations_l2191_219192


namespace abs_less_of_even_increasing_fn_l2191_219153

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- State the theorem
theorem abs_less_of_even_increasing_fn (a b : ℝ) 
  (h_even : is_even f) 
  (h_incr : is_increasing_on_nonneg f) 
  (h_less : f a < f b) : 
  |a| < |b| := by
  sorry

end abs_less_of_even_increasing_fn_l2191_219153


namespace max_value_trig_expression_l2191_219150

theorem max_value_trig_expression (a b c : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.cos (2 * θ) ≤ Real.sqrt (a^2 + b^2 + 2 * c^2)) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.cos (2 * θ) = Real.sqrt (a^2 + b^2 + 2 * c^2)) :=
by sorry

end max_value_trig_expression_l2191_219150


namespace odd_function_constant_term_zero_l2191_219152

def f (a b c x : ℝ) : ℝ := a * x^3 - b * x + c

theorem odd_function_constant_term_zero (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) → c = 0 := by
  sorry

end odd_function_constant_term_zero_l2191_219152


namespace biased_dice_probability_l2191_219163

def num_rolls : ℕ := 10
def num_sixes : ℕ := 4
def prob_six : ℚ := 1/3
def prob_not_six : ℚ := 2/3

theorem biased_dice_probability :
  (Nat.choose num_rolls num_sixes) * (prob_six ^ num_sixes) * (prob_not_six ^ (num_rolls - num_sixes)) = 13440/59049 :=
by sorry

end biased_dice_probability_l2191_219163


namespace baseball_team_selection_l2191_219125

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the team -/
def total_players : ℕ := 18

/-- The number of quadruplets that must be included -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 9

theorem baseball_team_selection :
  choose (total_players - quadruplets) (starters - quadruplets) = 2002 := by
  sorry

end baseball_team_selection_l2191_219125


namespace fishing_problem_l2191_219135

/-- The number of fish caught by Ollie -/
def ollie_fish : ℕ := 5

/-- The number of fish caught by Angus relative to Patrick -/
def angus_more_than_patrick : ℕ := 4

/-- The number of fish Ollie caught fewer than Angus -/
def ollie_fewer_than_angus : ℕ := 7

/-- The number of fish caught by Patrick -/
def patrick_fish : ℕ := 8

theorem fishing_problem :
  ollie_fish + ollie_fewer_than_angus - angus_more_than_patrick = patrick_fish := by
  sorry

end fishing_problem_l2191_219135


namespace sum_of_pairwise_quotients_geq_three_halves_l2191_219168

theorem sum_of_pairwise_quotients_geq_three_halves 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3/2 := by
  sorry

end sum_of_pairwise_quotients_geq_three_halves_l2191_219168


namespace dot_product_bounds_l2191_219128

theorem dot_product_bounds (a b : ℝ) :
  let v : ℝ × ℝ := (a, b)
  let u : ℝ → ℝ × ℝ := fun θ ↦ (Real.cos θ, Real.sin θ)
  ∀ θ, -Real.sqrt (a^2 + b^2) ≤ (v.1 * (u θ).1 + v.2 * (u θ).2) ∧
       (v.1 * (u θ).1 + v.2 * (u θ).2) ≤ Real.sqrt (a^2 + b^2) ∧
       (∃ θ₁, v.1 * (u θ₁).1 + v.2 * (u θ₁).2 = Real.sqrt (a^2 + b^2)) ∧
       (∃ θ₂, v.1 * (u θ₂).1 + v.2 * (u θ₂).2 = -Real.sqrt (a^2 + b^2)) :=
by
  sorry

end dot_product_bounds_l2191_219128


namespace accurate_to_tenths_l2191_219180

/-- Represents a decimal number with its integer and fractional parts -/
structure DecimalNumber where
  integerPart : Int
  fractionalPart : Nat
  fractionalDigits : Nat

/-- Defines accuracy to a certain decimal place -/
def accurateTo (n : DecimalNumber) (place : Nat) : Prop :=
  n.fractionalDigits ≥ place

/-- The decimal number 3.72 -/
def number : DecimalNumber :=
  { integerPart := 3,
    fractionalPart := 72,
    fractionalDigits := 2 }

/-- The tenths place -/
def tenthsPlace : Nat := 1

theorem accurate_to_tenths :
  accurateTo number tenthsPlace := by sorry

end accurate_to_tenths_l2191_219180


namespace power_fraction_simplification_l2191_219155

theorem power_fraction_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5/3 := by
  sorry

end power_fraction_simplification_l2191_219155


namespace cos_225_degrees_l2191_219175

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l2191_219175


namespace olly_ferrets_l2191_219129

theorem olly_ferrets (num_dogs : ℕ) (num_cats : ℕ) (total_shoes : ℕ) :
  num_dogs = 3 →
  num_cats = 2 →
  total_shoes = 24 →
  ∃ (num_ferrets : ℕ),
    num_ferrets * 4 + num_dogs * 4 + num_cats * 4 = total_shoes ∧
    num_ferrets = 1 :=
by sorry

end olly_ferrets_l2191_219129


namespace negation_equivalence_l2191_219166

theorem negation_equivalence (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 2^x₀ * (x₀ - a) > 1) ↔
  (∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1) :=
by sorry

end negation_equivalence_l2191_219166


namespace milford_lake_algae_increase_l2191_219178

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original current : ℕ) : ℕ := current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end milford_lake_algae_increase_l2191_219178


namespace complex_modulus_equation_l2191_219177

theorem complex_modulus_equation (t : ℝ) (h1 : t > 0) :
  Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 2 → t = Real.sqrt 41 := by
  sorry

end complex_modulus_equation_l2191_219177


namespace simplify_expression_value_given_condition_value_given_equations_l2191_219131

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3*(x+y)^2 - 7*(x+y) + 8*(x+y)^2 + 6*(x+y) = 11*(x+y)^2 - (x+y) := by sorry

-- Part 2
theorem value_given_condition (a : ℝ) (h : a^2 + 2*a = 3) :
  3*a^2 + 6*a - 14 = -5 := by sorry

-- Part 3
theorem value_given_equations (a b c d : ℝ) 
  (h1 : a - 3*b = 3) (h2 : 2*b + c = 5) (h3 : c - 4*d = -7) :
  (a - 2*b) - (3*b - c) - (c + 4*d) = -9 := by sorry

end simplify_expression_value_given_condition_value_given_equations_l2191_219131


namespace max_product_of_roots_l2191_219111

theorem max_product_of_roots (m : ℝ) : 
  let product_of_roots := m / 5
  let discriminant := 100 - 20 * m
  (discriminant ≥ 0) →  -- Condition for real roots
  product_of_roots ≤ 1 ∧ 
  (product_of_roots = 1 ↔ m = 5) :=
by sorry

end max_product_of_roots_l2191_219111


namespace walking_days_problem_l2191_219116

/-- 
Given:
- Jackie walks 2 miles per day
- Jessie walks 1.5 miles per day
- Over d days, Jackie walks 3 miles more than Jessie

Prove that d = 6
-/
theorem walking_days_problem (d : ℝ) 
  (h1 : 2 * d = 1.5 * d + 3) : d = 6 := by
  sorry

end walking_days_problem_l2191_219116


namespace soccer_lineup_selections_l2191_219198

/-- The number of players in the soccer team -/
def team_size : ℕ := 16

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to select the starting lineup -/
def lineup_selections : ℕ := 409500

/-- Theorem: The number of ways to select a starting lineup of 5 players from a team of 16,
    where one player (utility) cannot be selected for a specific position (goalkeeper),
    is equal to 409,500. -/
theorem soccer_lineup_selections :
  (team_size - 1) *  -- Goalkeeper selection (excluding utility player)
  (team_size - 1) *  -- Defender selection
  (team_size - 2) *  -- Midfielder selection
  (team_size - 3) *  -- Forward selection
  (team_size - 4)    -- Utility player selection (excluding goalkeeper)
  = lineup_selections := by sorry

end soccer_lineup_selections_l2191_219198


namespace problem_statement_l2191_219124

theorem problem_statement (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_a : a ≥ 1/a + 2/b) (h_b : b ≥ 3/a + 2/b) :
  (a + b ≥ 4) ∧ 
  (a^2 + b^2 ≥ 3 + 2*Real.sqrt 6) ∧ 
  (1/a + 1/b < 1 + Real.sqrt 2 / 2) := by
sorry

end problem_statement_l2191_219124


namespace mangoes_rate_per_kg_l2191_219113

/-- Given the conditions of Harkamal's purchase, prove that the rate per kg of mangoes is 55. -/
theorem mangoes_rate_per_kg (grapes_quantity : ℕ) (grapes_rate : ℕ) (mangoes_quantity : ℕ) (total_paid : ℕ) :
  grapes_quantity = 8 →
  grapes_rate = 80 →
  mangoes_quantity = 9 →
  total_paid = 1135 →
  (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity = 55 := by
  sorry

#eval (1135 - 8 * 80) / 9  -- This should evaluate to 55

end mangoes_rate_per_kg_l2191_219113


namespace dividend_division_theorem_l2191_219101

theorem dividend_division_theorem : ∃ (q r : ℕ), 
  220030 = (555 + 445) * q + r ∧ 
  r < (555 + 445) ∧ 
  r = 30 ∧ 
  q = 2 * (555 - 445) :=
by sorry

end dividend_division_theorem_l2191_219101


namespace problem_1_problem_2_l2191_219127

-- Problem 1
theorem problem_1 : (Real.sqrt 7 - Real.sqrt 3) * (Real.sqrt 7 + Real.sqrt 3) - (Real.sqrt 6 + Real.sqrt 2)^2 = -4 - 4 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 : (3 * Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 9/2 := by
  sorry

end problem_1_problem_2_l2191_219127


namespace equation_solution_l2191_219170

theorem equation_solution (x y : ℤ) (hy : y ≠ 0) :
  (2 : ℝ) ^ ((x - y : ℝ) / y) - (3 / 2 : ℝ) * y = 1 ↔
  ∃ n : ℕ, x = ((2 * n + 1) * (2 ^ (2 * n + 1) - 2)) / 3 ∧
           y = (2 ^ (2 * n + 1) - 2) / 3 :=
by sorry

end equation_solution_l2191_219170


namespace eight_solutions_for_triple_f_l2191_219162

def f (x : ℝ) : ℝ := |1 - 2*x|

theorem eight_solutions_for_triple_f (x : ℝ) :
  x ∈ Set.Icc 0 1 →
  ∃! (solutions : Finset ℝ),
    (∀ s ∈ solutions, f (f (f s)) = (1/2) * s) ∧
    Finset.card solutions = 8 :=
sorry

end eight_solutions_for_triple_f_l2191_219162


namespace value_of_one_item_l2191_219172

/-- Given two persons with equal capitals, each consisting of items of equal value and coins,
    prove that the value of one item is (p - m) / (a - b) --/
theorem value_of_one_item
  (a b : ℕ) (m p : ℝ) (h : a ≠ b)
  (equal_capitals : a * x + m = b * x + p)
  (x : ℝ) :
  x = (p - m) / (a - b) :=
by sorry

end value_of_one_item_l2191_219172


namespace ice_cream_bill_calculation_l2191_219190

/-- The final bill for four ice cream sundaes with a 20% tip -/
def final_bill (sundae1 sundae2 sundae3 sundae4 : ℝ) (tip_percentage : ℝ) : ℝ :=
  let total_cost := sundae1 + sundae2 + sundae3 + sundae4
  let tip := tip_percentage * total_cost
  total_cost + tip

/-- Theorem stating that the final bill for the given sundae prices and tip percentage is $42.00 -/
theorem ice_cream_bill_calculation :
  final_bill 7.50 10.00 8.50 9.00 0.20 = 42.00 := by
  sorry


end ice_cream_bill_calculation_l2191_219190


namespace arithmetic_sequence_common_difference_l2191_219184

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7)
  (h3 : ∃ d, arithmetic_sequence a d) :
  ∃ d, arithmetic_sequence a d ∧ d = 2 :=
sorry

end arithmetic_sequence_common_difference_l2191_219184


namespace a_most_stable_l2191_219187

/-- Represents a person's shooting performance data -/
structure ShootingData where
  name : String
  variance : Real

/-- Defines stability of shooting performance based on variance -/
def isMoreStable (a b : ShootingData) : Prop :=
  a.variance < b.variance

/-- Theorem: A has the most stable shooting performance -/
theorem a_most_stable (a b c d : ShootingData)
  (ha : a.name = "A" ∧ a.variance = 0.6)
  (hb : b.name = "B" ∧ b.variance = 1.1)
  (hc : c.name = "C" ∧ c.variance = 0.9)
  (hd : d.name = "D" ∧ d.variance = 1.2) :
  isMoreStable a b ∧ isMoreStable a c ∧ isMoreStable a d :=
sorry

end a_most_stable_l2191_219187


namespace triangle_side_length_l2191_219149

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  b^2 - 6*b + 8 = 0 →
  b = 2 := by sorry

end triangle_side_length_l2191_219149


namespace power_of_power_three_l2191_219136

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end power_of_power_three_l2191_219136


namespace rectangle_area_l2191_219157

/-- The area of a rectangle with length 20 cm and width 25 cm is 500 cm² -/
theorem rectangle_area : 
  ∀ (rectangle : Set ℝ) (length width area : ℝ),
  length = 20 →
  width = 25 →
  area = length * width →
  area = 500 := by sorry

end rectangle_area_l2191_219157


namespace absolute_value_equality_l2191_219191

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x + 2| → x = 1/2 := by
  sorry

end absolute_value_equality_l2191_219191


namespace quadratic_inequality_always_negative_l2191_219181

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end quadratic_inequality_always_negative_l2191_219181


namespace radius_B_is_three_fifths_l2191_219110

/-- A structure representing the configuration of circles A, B, C, and D. -/
structure CircleConfiguration where
  /-- Radius of circle A -/
  radius_A : ℝ
  /-- Radius of circle B -/
  radius_B : ℝ
  /-- Radius of circle D -/
  radius_D : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externally_tangent : Prop
  /-- Circles A, B, and C are internally tangent to circle D -/
  internally_tangent : Prop
  /-- Circles B and C are congruent -/
  B_C_congruent : Prop
  /-- The center of D is tangent to circle A at one point -/
  D_center_tangent_A : Prop

/-- Theorem stating that given the specific configuration of circles, the radius of circle B is 3/5. -/
theorem radius_B_is_three_fifths (config : CircleConfiguration)
  (h1 : config.radius_A = 2)
  (h2 : config.radius_D = 3) :
  config.radius_B = 3/5 := by
  sorry


end radius_B_is_three_fifths_l2191_219110


namespace quadratic_radical_range_l2191_219122

theorem quadratic_radical_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3 - x) ↔ x ≤ 3 := by
  sorry

end quadratic_radical_range_l2191_219122


namespace initial_volume_proof_l2191_219121

/-- Given a solution with initial volume V and 5% alcohol concentration,
    adding 2.5 liters of alcohol and 7.5 liters of water results in a
    9% alcohol concentration. Prove that V must be 40 liters. -/
theorem initial_volume_proof (V : ℝ) : 
  (0.05 * V + 2.5) / (V + 10) = 0.09 → V = 40 := by sorry

end initial_volume_proof_l2191_219121


namespace starting_team_combinations_l2191_219179

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The total number of team members --/
def totalMembers : ℕ := 20

/-- The number of players in the starting team --/
def startingTeamSize : ℕ := 9

/-- The number of goalkeepers --/
def numGoalkeepers : ℕ := 2

/-- Theorem stating the number of ways to choose the starting team --/
theorem starting_team_combinations : 
  (choose totalMembers numGoalkeepers) * (choose (totalMembers - numGoalkeepers) (startingTeamSize - numGoalkeepers)) = 6046560 := by
  sorry

end starting_team_combinations_l2191_219179


namespace four_digit_integer_proof_l2191_219183

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_proof (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 17)
  (h3 : middle_digits_sum n = 8)
  (h4 : thousands_minus_units n = 3)
  (h5 : n % 7 = 0) :
  n = 6443 := by
  sorry

end four_digit_integer_proof_l2191_219183


namespace line_segment_no_intersection_l2191_219139

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
structure LineSegment where
  p1 : Point
  p2 : Point

/-- Checks if a line segment intersects both x and y axes -/
def intersectsBothAxes (l : LineSegment) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    ((l.p1.x + t * (l.p2.x - l.p1.x) = 0 ∧ l.p1.y + t * (l.p2.y - l.p1.y) ≠ 0) ∨
     (l.p1.x + t * (l.p2.x - l.p1.x) ≠ 0 ∧ l.p1.y + t * (l.p2.y - l.p1.y) = 0))

theorem line_segment_no_intersection :
  let p1 : Point := ⟨-3, 4⟩
  let p2 : Point := ⟨-5, 1⟩
  let segment : LineSegment := ⟨p1, p2⟩
  ¬(intersectsBothAxes segment) :=
by
  sorry

end line_segment_no_intersection_l2191_219139


namespace power_five_2023_mod_11_l2191_219185

theorem power_five_2023_mod_11 : 5^2023 ≡ 4 [ZMOD 11] := by sorry

end power_five_2023_mod_11_l2191_219185


namespace tangent_line_perpendicular_l2191_219160

def f (x : ℝ) : ℝ := x - x^3 - 1

theorem tangent_line_perpendicular (a : ℝ) : 
  (∃ k : ℝ, k = (deriv f) 1 ∧ k * (-4/a) = -1) → a = -8 := by
  sorry

end tangent_line_perpendicular_l2191_219160


namespace office_payroll_is_75000_l2191_219145

/-- Calculates the total monthly payroll for office workers given the following conditions:
  * There are 15 factory workers with a total monthly payroll of $30,000
  * There are 30 office workers
  * The average monthly salary of an office worker exceeds that of a factory worker by $500
-/
def office_workers_payroll (
  factory_workers : ℕ)
  (factory_payroll : ℕ)
  (office_workers : ℕ)
  (salary_difference : ℕ) : ℕ :=
  let factory_avg_salary := factory_payroll / factory_workers
  let office_avg_salary := factory_avg_salary + salary_difference
  office_workers * office_avg_salary

/-- Theorem stating that the total monthly payroll for office workers is $75,000 -/
theorem office_payroll_is_75000 :
  office_workers_payroll 15 30000 30 500 = 75000 := by
  sorry

end office_payroll_is_75000_l2191_219145


namespace quadratic_positivity_quadratic_positivity_range_l2191_219137

/-- Given a quadratic function f(x) = x^2 + 2x + a, if f(x) > 0 for all x ≥ 1,
    then a > -3. -/
theorem quadratic_positivity (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x^2 + 2*x + a > 0) → a > -3 := by
  sorry

/-- The range of a for which f(x) = x^2 + 2x + a is positive for all x ≥ 1
    is the open interval (-3, +∞). -/
theorem quadratic_positivity_range :
  {a : ℝ | ∀ x : ℝ, x ≥ 1 → x^2 + 2*x + a > 0} = Set.Ioi (-3) := by
  sorry

end quadratic_positivity_quadratic_positivity_range_l2191_219137


namespace product_xyz_equals_negative_two_l2191_219103

theorem product_xyz_equals_negative_two 
  (x y z : ℝ) 
  (h1 : x + 2 / y = 2) 
  (h2 : y + 2 / z = 2) : 
  x * y * z = -2 := by sorry

end product_xyz_equals_negative_two_l2191_219103


namespace f_properties_l2191_219165

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_properties :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧
  (∀ x, f x ≤ 0) ∧
  (f 0 = 0) ∧
  (∀ x, f x ≥ -4) ∧
  (f 2 = -4) :=
by sorry

end f_properties_l2191_219165


namespace power_of_two_digit_sum_five_l2191_219169

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem power_of_two_digit_sum_five (n : ℕ) : 
  sum_of_digits (2^n) = 5 ↔ n = 5 := by sorry

end power_of_two_digit_sum_five_l2191_219169


namespace scientific_notation_of_given_number_l2191_219142

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The given number in millions -/
def givenNumber : ℝ := 141260

theorem scientific_notation_of_given_number :
  toScientificNotation givenNumber = ScientificNotation.mk 1.4126 5 (by sorry) :=
sorry

end scientific_notation_of_given_number_l2191_219142


namespace vector_addition_l2191_219147

theorem vector_addition : 
  let v1 : Fin 2 → ℝ := ![3, -7]
  let v2 : Fin 2 → ℝ := ![-6, 11]
  v1 + v2 = ![(-3), 4] := by sorry

end vector_addition_l2191_219147


namespace intersection_right_triangle_l2191_219161

/-- Given a line and a circle in the Cartesian plane, if they intersect at two points
    forming a right triangle with the circle's center, then the parameter of the line
    and circle equations must be -1. -/
theorem intersection_right_triangle (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (a * A.1 + A.2 - 2 = 0 ∧ (A.1 - 1)^2 + (A.2 - a)^2 = 16) ∧
    (a * B.1 + B.2 - 2 = 0 ∧ (B.1 - 1)^2 + (B.2 - a)^2 = 16) ∧
    A ≠ B ∧
    let C : ℝ × ℝ := (1, a)
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) →
  a = -1 := by
sorry


end intersection_right_triangle_l2191_219161


namespace changhyeon_money_problem_l2191_219102

theorem changhyeon_money_problem (initial_money : ℕ) : 
  (initial_money / 2 - 300) / 2 - 400 = 0 → initial_money = 2200 := by
  sorry

end changhyeon_money_problem_l2191_219102


namespace sin_cos_sum_47_43_l2191_219189

theorem sin_cos_sum_47_43 : Real.sin (47 * π / 180) * Real.cos (43 * π / 180) + Real.cos (47 * π / 180) * Real.sin (43 * π / 180) = 1 := by
  sorry

end sin_cos_sum_47_43_l2191_219189


namespace binomial_coefficient_16_4_l2191_219109

theorem binomial_coefficient_16_4 : Nat.choose 16 4 = 1820 := by
  -- The proof goes here
  sorry

end binomial_coefficient_16_4_l2191_219109


namespace gcd_lcm_product_30_75_l2191_219114

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end gcd_lcm_product_30_75_l2191_219114


namespace arithmetic_series_sum_example_l2191_219108

/-- Sum of an arithmetic series -/
def arithmetic_series_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℚ :=
  let n : ℚ := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

/-- Theorem: The sum of the arithmetic series with first term -35, last term 1, and common difference 2 is -323 -/
theorem arithmetic_series_sum_example : 
  arithmetic_series_sum (-35) 1 2 = -323 := by sorry

end arithmetic_series_sum_example_l2191_219108


namespace andrei_apple_spending_l2191_219148

/-- Calculates Andrei's monthly spending on apples given the original price, price increase percentage, discount percentage, and amount bought per month. -/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease / 100)
  let discountedPrice := newPrice * (1 - discount / 100)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles under the given conditions. -/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 10 10 2 = 99 := by
  sorry

end andrei_apple_spending_l2191_219148


namespace unique_solution_value_l2191_219159

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant b^2 - 4ac must be zero -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 4x^2 + mx + 16 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  4*x^2 + m*x + 16 = 0

theorem unique_solution_value :
  ∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, has_one_solution 4 m 16) ∧ m = 16 :=
sorry

end unique_solution_value_l2191_219159


namespace grape_sales_properties_l2191_219106

/-- Represents the properties of the grape sales scenario -/
structure GrapeSales where
  initial_price : ℝ
  initial_volume : ℝ
  cost_price : ℝ
  price_reduction_effect : ℝ

/-- Calculates the daily sales profit for a given price reduction -/
def daily_profit (g : GrapeSales) (price_reduction : ℝ) : ℝ :=
  let new_price := g.initial_price - price_reduction
  let new_volume := g.initial_volume + price_reduction * g.price_reduction_effect
  (new_price - g.cost_price) * new_volume

/-- Calculates the profit as a function of selling price -/
def profit_function (g : GrapeSales) (x : ℝ) : ℝ :=
  (x - g.cost_price) * (g.initial_volume + (g.initial_price - x) * g.price_reduction_effect)

/-- Theorem stating the properties of the grape sales scenario -/
theorem grape_sales_properties (g : GrapeSales) 
  (h1 : g.initial_price = 30)
  (h2 : g.initial_volume = 60)
  (h3 : g.cost_price = 15)
  (h4 : g.price_reduction_effect = 10) :
  daily_profit g 2 = 1040 ∧ 
  (∃ (x : ℝ), x = 51/2 ∧ ∀ (y : ℝ), profit_function g y ≤ profit_function g x) ∧
  (∃ (max_profit : ℝ), max_profit = 1102.5 ∧ 
    ∀ (y : ℝ), profit_function g y ≤ max_profit) := by
  sorry

end grape_sales_properties_l2191_219106


namespace no_natural_solution_l2191_219141

theorem no_natural_solution (x y z : ℕ) : (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / x ≠ 1 := by
  sorry

end no_natural_solution_l2191_219141


namespace vector_problem_l2191_219195

/-- Given four points in a plane, prove that if certain vector conditions are met,
    then the coordinates of point D and the value of k are as specified. -/
theorem vector_problem (A B C D : ℝ × ℝ) (k : ℝ) : 
  A = (1, 3) →
  B = (2, -2) →
  C = (4, -1) →
  B - A = D - C →
  ∃ (t : ℝ), t • (k • (B - A) - (C - B)) = (B - A) + 3 • (C - B) →
  D = (5, -6) ∧ k = -1/3 := by
  sorry

end vector_problem_l2191_219195


namespace no_linear_factor_with_integer_coefficients_l2191_219199

theorem no_linear_factor_with_integer_coefficients :
  ∀ (a b c d : ℤ), (∀ (x y z : ℝ), 
    a*x + b*y + c*z + d ≠ 0 ∨ 
    x^2 - y^2 - z^2 + 3*y*z + x + 2*y - z ≠ (a*x + b*y + c*z + d) * 
      ((x^2 - y^2 - z^2 + 3*y*z + x + 2*y - z) / (a*x + b*y + c*z + d))) :=
by sorry

end no_linear_factor_with_integer_coefficients_l2191_219199


namespace rectangular_region_area_l2191_219154

/-- The area of a rectangular region enclosed by lines derived from given equations -/
theorem rectangular_region_area (a : ℝ) (ha : a > 0) :
  let eq1 (x y : ℝ) := (2 * x - a * y)^2 = 25 * a^2
  let eq2 (x y : ℝ) := (5 * a * x + 2 * y)^2 = 36 * a^2
  let area := (120 * a^2) / Real.sqrt (100 * a^2 + 16 + 100 * a^4)
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    eq1 x1 y1 ∧ eq1 x2 y2 ∧ eq1 x3 y3 ∧ eq1 x4 y4 ∧
    eq2 x1 y1 ∧ eq2 x2 y2 ∧ eq2 x3 y3 ∧ eq2 x4 y4 ∧
    (x1 - x2) * (y1 - y3) = area :=
by sorry

end rectangular_region_area_l2191_219154


namespace cubic_equation_root_magnitude_l2191_219197

theorem cubic_equation_root_magnitude (k : ℝ) : 
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  (k = -1 ∨ k = 3) :=
by sorry

end cubic_equation_root_magnitude_l2191_219197


namespace arithmetic_evaluation_l2191_219196

theorem arithmetic_evaluation : 4 * (9 - 6) - 8 = 4 := by
  sorry

end arithmetic_evaluation_l2191_219196


namespace subset_implies_a_range_l2191_219100

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end subset_implies_a_range_l2191_219100


namespace quadratic_is_square_of_binomial_l2191_219134

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  a = 4 → ∃ (r s : ℝ), a * x^2 + 16 * x + 16 = (r * x + s)^2 := by
  sorry

end quadratic_is_square_of_binomial_l2191_219134


namespace distance_from_center_to_chords_l2191_219119

/-- A circle with two chords drawn through the ends of a diameter -/
structure CircleWithChords where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first chord -/
  chord1_length : ℝ
  /-- The length of the second chord -/
  chord2_length : ℝ
  /-- The chords intersect on the circumference -/
  chords_intersect_on_circumference : True
  /-- The chords are drawn through the ends of a diameter -/
  chords_through_diameter_ends : True
  /-- The first chord has length 12 -/
  chord1_is_12 : chord1_length = 12
  /-- The second chord has length 16 -/
  chord2_is_16 : chord2_length = 16

/-- The theorem stating the distances from the center to the chords -/
theorem distance_from_center_to_chords (c : CircleWithChords) :
  ∃ (d1 d2 : ℝ), d1 = 8 ∧ d2 = 6 ∧
  d1 = c.chord2_length / 2 ∧
  d2 = c.chord1_length / 2 :=
sorry

end distance_from_center_to_chords_l2191_219119


namespace payment_difference_l2191_219140

/-- The original price of the dish -/
def original_price : Float := 24.00000000000002

/-- The discount percentage -/
def discount_percent : Float := 0.10

/-- The tip percentage -/
def tip_percent : Float := 0.15

/-- The discounted price of the dish -/
def discounted_price : Float := original_price * (1 - discount_percent)

/-- John's tip amount -/
def john_tip : Float := original_price * tip_percent

/-- Jane's tip amount -/
def jane_tip : Float := discounted_price * tip_percent

/-- John's total payment -/
def john_total : Float := discounted_price + john_tip

/-- Jane's total payment -/
def jane_total : Float := discounted_price + jane_tip

/-- Theorem stating the difference between John's and Jane's payments -/
theorem payment_difference : john_total - jane_total = 0.3600000000000003 := by
  sorry

end payment_difference_l2191_219140


namespace percent_equality_l2191_219105

theorem percent_equality (x : ℝ) : (60 / 100 * 600 = 50 / 100 * x) → x = 720 := by
  sorry

end percent_equality_l2191_219105


namespace sum_of_two_numbers_l2191_219174

theorem sum_of_two_numbers : ∃ (a b : ℤ), 
  (a = |(-10)| + 1) ∧ 
  (b = -(2) - 1) ∧ 
  (a + b = 8) := by
  sorry

end sum_of_two_numbers_l2191_219174


namespace extreme_values_and_sum_l2191_219164

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem extreme_values_and_sum (α β : ℝ) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x ≥ -2 ∧ f x ≤ 2) ∧
  f (α - Real.pi / 6) = 2 * Real.sqrt 5 / 5 ∧
  Real.sin (β - α) = Real.sqrt 10 / 10 ∧
  α ∈ Set.Icc (Real.pi / 4) Real.pi ∧
  β ∈ Set.Icc Real.pi (3 * Real.pi / 2) →
  (∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = -2) ∧
  (∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = 2) ∧
  α + β = 7 * Real.pi / 4 :=
by sorry

end extreme_values_and_sum_l2191_219164


namespace pages_to_read_tonight_l2191_219104

def pages_three_nights_ago : ℕ := 20

def pages_two_nights_ago (x : ℕ) : ℕ := x^2 + 5

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + sum_of_digits (n / 10))

def pages_last_night (x : ℕ) : ℕ := 3 * sum_of_digits x

def total_pages : ℕ := 500

theorem pages_to_read_tonight : 
  total_pages - (pages_three_nights_ago + 
                 pages_two_nights_ago pages_three_nights_ago + 
                 pages_last_night (pages_two_nights_ago pages_three_nights_ago)) = 48 := by
  sorry

end pages_to_read_tonight_l2191_219104


namespace max_cookies_eaten_l2191_219120

/-- Given two people sharing 30 cookies, where one eats twice as many as the other,
    the maximum number of cookies the person eating fewer could have eaten is 10. -/
theorem max_cookies_eaten (total : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ) : 
  total = 30 →
  bella_cookies = 2 * andy_cookies →
  total = andy_cookies + bella_cookies →
  andy_cookies ≤ 10 :=
by sorry

end max_cookies_eaten_l2191_219120


namespace total_eggs_collected_l2191_219118

/-- The number of dozen eggs Benjamin collects per day -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects per day -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects per day -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem total_eggs_collected :
  total_eggs = 26 := by sorry

end total_eggs_collected_l2191_219118


namespace digit_47_is_6_l2191_219138

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating cycle in the decimal representation of 1/17 -/
def cycle_length : Nat := 16

/-- The 47th digit after the decimal point in the decimal representation of 1/17 -/
def digit_47 : Nat := decimal_rep_1_17[(47 - 1) % cycle_length]

theorem digit_47_is_6 : digit_47 = 6 := by sorry

end digit_47_is_6_l2191_219138


namespace certain_number_operations_l2191_219171

theorem certain_number_operations (x : ℝ) : 
  (((x + 5) * 2) / 5) - 5 = 62.5 / 2 → x = 85.625 := by
  sorry

end certain_number_operations_l2191_219171


namespace janice_earnings_l2191_219132

/-- Calculates the total earnings for Janice's work week -/
def calculate_earnings (days_worked : ℕ) (daily_rate : ℕ) (overtime_rate : ℕ) (overtime_shifts : ℕ) : ℕ :=
  days_worked * daily_rate + overtime_shifts * overtime_rate

/-- Proves that Janice's earnings for the week equal $195 -/
theorem janice_earnings : calculate_earnings 5 30 15 3 = 195 := by
  sorry

end janice_earnings_l2191_219132


namespace markup_is_twenty_percent_l2191_219143

/-- Calculates the markup percentage given cost price, discount, and profit percentage. -/
def markup_percentage (cost_price discount : ℚ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage) - discount
  let markup := selling_price - cost_price
  (markup / cost_price) * 100

/-- Theorem stating that under the given conditions, the markup percentage is 20%. -/
theorem markup_is_twenty_percent :
  markup_percentage 180 50 (20/100) = 20 := by
sorry

end markup_is_twenty_percent_l2191_219143


namespace sqrt_sum_equal_product_equal_l2191_219173

-- Problem 1
theorem sqrt_sum_equal : Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24 = 3 * Real.sqrt 6 := by sorry

-- Problem 2
theorem product_equal : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 := by sorry

end sqrt_sum_equal_product_equal_l2191_219173


namespace prime_square_plus_two_l2191_219193

theorem prime_square_plus_two (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → p = 3 :=
by sorry

end prime_square_plus_two_l2191_219193


namespace factorial_equation_solutions_l2191_219117

theorem factorial_equation_solutions :
  ∀ (x y : ℕ) (z : ℤ),
    (Odd z) →
    (Nat.factorial x + Nat.factorial y = 24 * z + 2017) →
    ((x = 1 ∧ y = 4 ∧ z = -83) ∨
     (x = 4 ∧ y = 1 ∧ z = -83) ∨
     (x = 1 ∧ y = 5 ∧ z = -79) ∨
     (x = 5 ∧ y = 1 ∧ z = -79)) :=
by sorry


end factorial_equation_solutions_l2191_219117


namespace bucket_weight_bucket_weight_proof_l2191_219146

/-- 
Given a bucket that weighs p kilograms when three-fourths full of water
and q kilograms when one-third full of water, this theorem proves that
the weight of the bucket when full is (8p - 3q) / 5 kilograms.
-/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let three_fourths_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 3 * q) / 5
  full_weight

/-- The proof of the bucket_weight theorem. -/
theorem bucket_weight_proof (p q : ℝ) : 
  bucket_weight p q = (8 * p - 3 * q) / 5 := by
  sorry

end bucket_weight_bucket_weight_proof_l2191_219146


namespace total_is_260_l2191_219186

/-- Represents the ratio of money shared among four people -/
structure MoneyRatio :=
  (a b c d : ℕ)

/-- Calculates the total amount of money shared given a ratio and the first person's share -/
def totalShared (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  firstShare * (ratio.a + ratio.b + ratio.c + ratio.d)

/-- Theorem stating that for the given ratio and first share, the total is 260 -/
theorem total_is_260 (ratio : MoneyRatio) (h1 : ratio.a = 1) (h2 : ratio.b = 2) 
    (h3 : ratio.c = 7) (h4 : ratio.d = 3) (h5 : firstShare = 20) : 
    totalShared ratio firstShare = 260 := by
  sorry


end total_is_260_l2191_219186
