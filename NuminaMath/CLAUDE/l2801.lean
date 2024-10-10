import Mathlib

namespace count_squares_below_line_l2801_280110

/-- The number of 1x1 squares in the first quadrant lying entirely below the line 6x + 216y = 1296 -/
def squaresBelowLine : ℕ :=
  -- Definition goes here
  sorry

/-- The equation of the line -/
def lineEquation (x y : ℝ) : Prop :=
  6 * x + 216 * y = 1296

theorem count_squares_below_line :
  squaresBelowLine = 537 := by
  sorry

end count_squares_below_line_l2801_280110


namespace unique_solution_l2801_280156

/-- The functional equation that f must satisfy for all real x and y -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)

/-- The theorem stating that the only function satisfying the equation is f(x) = x + 2 -/
theorem unique_solution (f : ℝ → ℝ) (h : functional_equation f) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end unique_solution_l2801_280156


namespace tshirt_sales_optimization_l2801_280186

/-- Represents the profit function for T-shirt sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 3000

/-- Represents the sales volume function based on price increase -/
def sales_volume (x : ℝ) : ℝ := 300 - 10 * x

theorem tshirt_sales_optimization :
  let initial_price : ℝ := 40
  let purchase_price : ℝ := 30
  let target_profit : ℝ := 3360
  let optimal_increase : ℝ := 2
  let max_profit_price : ℝ := 50
  let max_profit : ℝ := 4000
  
  -- Part 1: Prove that increasing the price by 2 yuan yields the target profit
  (∃ x : ℝ, x ≥ 0 ∧ profit_function x = target_profit ∧
    ∀ y : ℝ, y ≥ 0 ∧ profit_function y = target_profit → x ≤ y) ∧
  profit_function optimal_increase = target_profit ∧
  
  -- Part 2: Prove that setting the price to 50 yuan maximizes profit
  (∀ x : ℝ, profit_function x ≤ max_profit) ∧
  profit_function (max_profit_price - initial_price) = max_profit := by
  sorry

end tshirt_sales_optimization_l2801_280186


namespace unique_mythical_with_most_divisors_l2801_280166

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_mythical (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → ∃ p : ℕ, is_prime p ∧ d = p - 2

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_mythical_with_most_divisors :
  is_mythical 135 ∧
  ∀ n : ℕ, is_mythical n → number_of_divisors n ≤ number_of_divisors 135 ∧
  (number_of_divisors n = number_of_divisors 135 → n = 135) :=
sorry

end unique_mythical_with_most_divisors_l2801_280166


namespace walking_rate_ratio_l2801_280123

theorem walking_rate_ratio 
  (D : ℝ) -- Distance to school
  (R : ℝ) -- Usual walking rate
  (R' : ℝ) -- New walking rate
  (h1 : D = R * 21) -- Usual time equation
  (h2 : D = R' * 18) -- New time equation
  : R' / R = 7 / 6 := by sorry

end walking_rate_ratio_l2801_280123


namespace given_equation_is_quadratic_l2801_280120

/-- An equation is quadratic in one variable if it can be expressed in the form ax² + bx + c = 0, where a ≠ 0 and x is the variable. --/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (x - 1)(x + 2) = 1 --/
def given_equation (x : ℝ) : ℝ :=
  (x - 1) * (x + 2) - 1

theorem given_equation_is_quadratic :
  is_quadratic_one_var given_equation :=
sorry

end given_equation_is_quadratic_l2801_280120


namespace no_solution_exists_l2801_280145

theorem no_solution_exists : ¬∃ x : ℝ, 2 * ((x - 3) / 2 + 3) = x + 6 := by
  sorry

end no_solution_exists_l2801_280145


namespace intersection_A_B_l2801_280114

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {-2, 0, 2}

theorem intersection_A_B : A ∩ B = {0, 2} := by
  sorry

end intersection_A_B_l2801_280114


namespace regular_octagon_area_l2801_280130

/-- Regular octagon with area A, longest diagonal d_max, and shortest diagonal d_min -/
structure RegularOctagon where
  A : ℝ
  d_max : ℝ
  d_min : ℝ

/-- The area of a regular octagon is equal to the product of its longest and shortest diagonals -/
theorem regular_octagon_area (o : RegularOctagon) : o.A = o.d_max * o.d_min := by
  sorry

end regular_octagon_area_l2801_280130


namespace complex_square_root_l2801_280142

theorem complex_square_root (p q : ℕ+) (h : (p + q * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  p + q * Complex.I = 4 + 3 * Complex.I :=
by sorry

end complex_square_root_l2801_280142


namespace girls_count_in_classroom_l2801_280132

theorem girls_count_in_classroom (ratio_girls : ℕ) (ratio_boys : ℕ) 
  (total_count : ℕ) (h1 : ratio_girls = 4) (h2 : ratio_boys = 3) 
  (h3 : total_count = 43) :
  (ratio_girls * total_count - ratio_girls) / (ratio_girls + ratio_boys) = 24 := by
  sorry

end girls_count_in_classroom_l2801_280132


namespace butterfat_mixture_l2801_280165

theorem butterfat_mixture (initial_volume : ℝ) (initial_butterfat : ℝ) 
  (added_volume : ℝ) (added_butterfat : ℝ) (target_butterfat : ℝ) :
  initial_volume = 8 →
  initial_butterfat = 0.3 →
  added_butterfat = 0.1 →
  target_butterfat = 0.2 →
  added_volume = 8 →
  (initial_volume * initial_butterfat + added_volume * added_butterfat) / 
  (initial_volume + added_volume) = target_butterfat :=
by
  sorry

#check butterfat_mixture

end butterfat_mixture_l2801_280165


namespace negation_of_existence_negation_of_quadratic_existence_l2801_280178

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ c > 0, p c) ↔ (∀ c > 0, ¬p c) :=
by sorry

def has_solution (c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + c = 0

theorem negation_of_quadratic_existence :
  (¬∃ c > 0, has_solution c) ↔ (∀ c > 0, ¬has_solution c) :=
by sorry

end negation_of_existence_negation_of_quadratic_existence_l2801_280178


namespace factory_door_production_l2801_280129

/-- Calculates the number of doors produced by a car factory given various production changes -/
theorem factory_door_production
  (doors_per_car : ℕ)
  (initial_plan : ℕ)
  (shortage_decrease : ℕ)
  (pandemic_cut : Rat)
  (h1 : doors_per_car = 5)
  (h2 : initial_plan = 200)
  (h3 : shortage_decrease = 50)
  (h4 : pandemic_cut = 1/2) :
  (initial_plan - shortage_decrease) * pandemic_cut * doors_per_car = 375 := by
  sorry

end factory_door_production_l2801_280129


namespace child_ticket_cost_child_ticket_cost_is_6_l2801_280191

/-- Proves that the cost of a child's ticket is 6 dollars given the specified conditions -/
theorem child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (children_attending : ℕ) : ℕ :=
  let child_ticket_cost := (total_revenue - adult_ticket_cost * (total_tickets - children_attending)) / children_attending
  have h1 : adult_ticket_cost = 9 := by sorry
  have h2 : total_tickets = 225 := by sorry
  have h3 : total_revenue = 1875 := by sorry
  have h4 : children_attending = 50 := by sorry
  have h5 : child_ticket_cost * children_attending + adult_ticket_cost * (total_tickets - children_attending) = total_revenue := by sorry
  6

theorem child_ticket_cost_is_6 : child_ticket_cost 9 225 1875 50 = 6 := by sorry

end child_ticket_cost_child_ticket_cost_is_6_l2801_280191


namespace quadratic_coefficient_l2801_280113

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b < 0 ∧ 
  (∀ x, x^2 + b*x + 1/5 = (x+n)^2 + 1/20) →
  b = -Real.sqrt (3/5) := by
  sorry

end quadratic_coefficient_l2801_280113


namespace buckingham_palace_visitor_difference_l2801_280134

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 317

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := 295

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := current_day_visitors - previous_day_visitors

theorem buckingham_palace_visitor_difference :
  visitor_difference = 22 :=
by sorry

end buckingham_palace_visitor_difference_l2801_280134


namespace system_solution_l2801_280128

theorem system_solution (a b : ℝ) :
  ∃ (x y : ℝ), 
    (x + y = a ∧ Real.tan x * Real.tan y = b) ∧
    ((b ≠ 1 → 
      ∃ (k : ℤ), 
        x = (a + 2 * k * Real.pi + Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2 ∧
        y = (a - 2 * k * Real.pi - Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2) ∨
     (b ≠ 1 → 
      ∃ (k : ℤ), 
        x = (a + 2 * k * Real.pi - Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2 ∧
        y = (a - 2 * k * Real.pi + Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2)) ∧
    (b = 1 ∧ ∃ (k : ℤ), a = Real.pi / 2 + k * Real.pi → y = Real.pi / 2 + k * Real.pi - x) :=
by
  sorry


end system_solution_l2801_280128


namespace prob_different_colors_is_three_fourths_l2801_280136

/-- The set of colors for shorts -/
inductive ShortsColor
| Red
| Blue
| Green

/-- The set of colors for jerseys -/
inductive JerseyColor
| Red
| Blue
| Green
| Yellow

/-- The probability of choosing different colors for shorts and jersey -/
def prob_different_colors : ℚ := 3/4

/-- Theorem stating that the probability of choosing different colors for shorts and jersey is 3/4 -/
theorem prob_different_colors_is_three_fourths :
  prob_different_colors = 3/4 := by sorry

end prob_different_colors_is_three_fourths_l2801_280136


namespace nigels_money_ratio_l2801_280160

/-- Represents Nigel's money transactions and proves the final ratio --/
theorem nigels_money_ratio :
  ∀ (original : ℝ) (given_away : ℝ),
  original > 0 →
  given_away > 0 →
  original + 45 - given_away + 80 - 25 = 2 * original + 25 →
  (original + 45 + 80 - 25) / original = 3 :=
by sorry

end nigels_money_ratio_l2801_280160


namespace point_on_line_l2801_280193

/-- Given five points O, A, B, C, D on a straight line with specified distances,
    and points P and Q satisfying certain ratio conditions, prove that OQ has the given value. -/
theorem point_on_line (a b c d : ℝ) :
  let O := 0
  let A := 2 * a
  let B := 4 * b
  let C := 5 * c
  let D := 7 * d
  let P := (14 * b * d - 10 * a * c) / (2 * a - 4 * b + 7 * d - 5 * c)
  let Q := (14 * c * d - 10 * b * c) / (5 * c - 7 * d)
  (A - P) / (P - D) = (B - P) / (P - C) →
  (Q - C) / (D - Q) = (B - C) / (D - C) →
  Q = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) :=
by sorry

end point_on_line_l2801_280193


namespace min_sheets_theorem_l2801_280133

/-- The minimum number of sheets in a pad of paper -/
def min_sheets_in_pad : ℕ := 36

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The number of days Evelyn takes off per week -/
def days_off : ℕ := 2

/-- The number of sheets Evelyn uses per working day -/
def sheets_per_day : ℕ := 12

/-- Theorem stating that the minimum number of sheets in a pad of paper is 36 -/
theorem min_sheets_theorem : 
  min_sheets_in_pad = (weekdays - days_off) * sheets_per_day :=
by sorry

end min_sheets_theorem_l2801_280133


namespace raccoon_stall_time_l2801_280196

/-- The time (in minutes) the first lock stalls the raccoons -/
def T1 : ℕ := 5

/-- The time (in minutes) the second lock stalls the raccoons -/
def T2 : ℕ := 3 * T1 - 3

/-- The time (in minutes) both locks together stall the raccoons -/
def both_locks : ℕ := 5 * T2

theorem raccoon_stall_time : both_locks = 60 := by
  sorry

end raccoon_stall_time_l2801_280196


namespace min_value_of_x_l2801_280181

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 2 + (1/2) * Real.log x) : x ≥ 4 := by
  sorry

end min_value_of_x_l2801_280181


namespace p_sufficient_not_necessary_l2801_280138

-- Define the propositions p and q
def p (a : ℝ) : Prop := 1/a > 1/4

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∃ a : ℝ, p a ∧ q a) ∧ (∃ a : ℝ, ¬p a ∧ q a) ∧ (∀ a : ℝ, p a → q a) :=
sorry

end p_sufficient_not_necessary_l2801_280138


namespace first_discount_calculation_l2801_280199

theorem first_discount_calculation (original_price final_price second_discount : ℝ) 
  (h1 : original_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5)
  : ∃ first_discount : ℝ, 
    first_discount = 20 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end first_discount_calculation_l2801_280199


namespace sum_of_cubes_plus_one_divisible_by_5_l2801_280177

def sum_of_cubes_plus_one (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => (i + 1)^3 + 1)

theorem sum_of_cubes_plus_one_divisible_by_5 :
  5 ∣ sum_of_cubes_plus_one 50 := by
sorry

end sum_of_cubes_plus_one_divisible_by_5_l2801_280177


namespace admission_cost_proof_l2801_280155

/-- Calculates the total cost of admission tickets for a group -/
def total_cost (adult_price child_price : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := num_children + 25
  let adult_cost := num_adults * adult_price
  let child_cost := num_children * child_price
  adult_cost + child_cost

/-- Proves that the total cost for the given group is $720 -/
theorem admission_cost_proof :
  total_cost 15 8 15 = 720 := by
  sorry

end admission_cost_proof_l2801_280155


namespace equation_solution_l2801_280101

theorem equation_solution : ∃! x : ℝ, (4 : ℝ) ^ (x + 1) = (64 : ℝ) ^ (1/3) := by
  sorry

end equation_solution_l2801_280101


namespace textbook_order_solution_l2801_280140

/-- Represents the textbook order problem -/
structure TextbookOrder where
  red_cost : ℝ
  trad_cost : ℝ
  red_price_ratio : ℝ
  quantity_diff : ℕ
  total_quantity : ℕ
  max_trad_quantity : ℕ
  max_total_cost : ℝ

/-- Theorem stating the solution to the textbook order problem -/
theorem textbook_order_solution (order : TextbookOrder)
  (h1 : order.red_cost = 14000)
  (h2 : order.trad_cost = 7000)
  (h3 : order.red_price_ratio = 1.4)
  (h4 : order.quantity_diff = 300)
  (h5 : order.total_quantity = 1000)
  (h6 : order.max_trad_quantity = 400)
  (h7 : order.max_total_cost = 12880) :
  ∃ (red_price trad_price min_cost : ℝ),
    red_price = 14 ∧
    trad_price = 10 ∧
    min_cost = 12400 ∧
    red_price = order.red_price_ratio * trad_price ∧
    order.red_cost / red_price - order.trad_cost / trad_price = order.quantity_diff ∧
    min_cost ≤ order.max_total_cost ∧
    (∀ (trad_quantity : ℕ),
      trad_quantity ≤ order.max_trad_quantity →
      trad_quantity * trad_price + (order.total_quantity - trad_quantity) * red_price ≥ min_cost) :=
by sorry

end textbook_order_solution_l2801_280140


namespace symmetric_complex_number_l2801_280175

/-- Given that z is symmetric to 2/(1-i) with respect to the imaginary axis, prove that z = -1 + i -/
theorem symmetric_complex_number (z : ℂ) : 
  (z.re = -(2 / (1 - I)).re ∧ z.im = (2 / (1 - I)).im) → z = -1 + I :=
by sorry

end symmetric_complex_number_l2801_280175


namespace crayons_left_l2801_280105

theorem crayons_left (total : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) : 
  total = 120 →
  kiley_fraction = 3/8 →
  joe_fraction = 5/9 →
  (total - kiley_fraction * total) - joe_fraction * (total - kiley_fraction * total) = 33 := by
  sorry

end crayons_left_l2801_280105


namespace power_function_properties_l2801_280151

-- Define the power function f(x) = x^α
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 3 / Real.log 9)

-- Theorem statement
theorem power_function_properties :
  -- The function passes through (9,3)
  f 9 = 3 ∧
  -- f(x) is increasing on its domain
  (∀ x y, x < y → x > 0 → y > 0 → f x < f y) ∧
  -- When x ≥ 4, f(x) ≥ 2
  (∀ x, x ≥ 4 → f x ≥ 2) ∧
  -- When x₂ > x₁ > 0, (f(x₁) + f(x₂))/2 < f((x₁ + x₂)/2)
  (∀ x₁ x₂, x₂ > x₁ → x₁ > 0 → (f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by sorry

end power_function_properties_l2801_280151


namespace watch_loss_percentage_loss_percentage_is_ten_percent_l2801_280185

/-- Proves that the loss percentage is 10% for a watch sale scenario --/
theorem watch_loss_percentage : ℝ → Prop :=
  λ L : ℝ =>
    let cost_price : ℝ := 2000
    let selling_price : ℝ := cost_price - (L / 100 * cost_price)
    let new_selling_price : ℝ := cost_price + (4 / 100 * cost_price)
    new_selling_price = selling_price + 280 →
    L = 10

/-- The loss percentage is indeed 10% --/
theorem loss_percentage_is_ten_percent : watch_loss_percentage 10 := by
  sorry

end watch_loss_percentage_loss_percentage_is_ten_percent_l2801_280185


namespace polynomial_equality_l2801_280183

theorem polynomial_equality (x : ℝ) : 
  let k : ℝ := -9
  let a : ℝ := 15
  let b : ℝ := 72
  (3 * x^2 - 4 * x + 5) * (5 * x^2 + k * x + 8) = 
    15 * x^4 - 47 * x^3 + a * x^2 - b * x + 40 := by
  sorry

end polynomial_equality_l2801_280183


namespace prime_divisors_50_factorial_l2801_280139

/-- The number of prime divisors of 50! -/
def num_prime_divisors_50_factorial : ℕ := sorry

/-- Theorem stating that the number of prime divisors of 50! is 15 -/
theorem prime_divisors_50_factorial :
  num_prime_divisors_50_factorial = 15 := by sorry

end prime_divisors_50_factorial_l2801_280139


namespace sufficient_but_not_necessary_condition_l2801_280144

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end sufficient_but_not_necessary_condition_l2801_280144


namespace power_comparison_l2801_280153

theorem power_comparison : 2^1997 > 5^850 := by
  sorry

end power_comparison_l2801_280153


namespace unique_solution_mn_l2801_280116

theorem unique_solution_mn : ∃! (m n : ℕ+), 10 * m * n = 45 - 5 * m - 3 * n := by
  sorry

end unique_solution_mn_l2801_280116


namespace intersection_of_A_and_B_l2801_280100

def A : Set ℕ := {x | x - 4 < 0}
def B : Set ℕ := {0, 1, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 3} := by
  sorry

end intersection_of_A_and_B_l2801_280100


namespace tournament_games_l2801_280137

theorem tournament_games (total_teams : Nat) (preliminary_teams : Nat) (preliminary_matches : Nat) :
  total_teams = 24 →
  preliminary_teams = 16 →
  preliminary_matches = 8 →
  preliminary_teams = 2 * preliminary_matches →
  (total_games : Nat) = preliminary_matches + (total_teams - preliminary_matches) - 1 →
  total_games = 23 := by
  sorry

end tournament_games_l2801_280137


namespace divisor_count_equals_equation_solutions_l2801_280176

/-- The prime factorization of 2310 -/
def prime_factors : List Nat := [2, 3, 5, 7, 11]

/-- The exponent of 2310 in the number we're considering -/
def exponent : Nat := 2310

/-- A function that counts the number of positive integer divisors of n^exponent 
    that are divisible by exactly 48 positive integers, 
    where n is the product of the prime factors -/
def count_specific_divisors (prime_factors : List Nat) (exponent : Nat) : Nat :=
  sorry

/-- A function that counts the number of solutions (a,b,c,d,e) to the equation 
    (a+1)(b+1)(c+1)(d+1)(e+1) = 48, where a,b,c,d,e are non-negative integers -/
def count_equation_solutions : Nat :=
  sorry

/-- The main theorem stating the equality of the two counting functions -/
theorem divisor_count_equals_equation_solutions : 
  count_specific_divisors prime_factors exponent = count_equation_solutions :=
  sorry

end divisor_count_equals_equation_solutions_l2801_280176


namespace weighted_sum_square_inequality_l2801_280192

theorem weighted_sum_square_inequality (x y a b : ℝ) 
  (h1 : a + b = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) : 
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 := by
  sorry

end weighted_sum_square_inequality_l2801_280192


namespace plane_perpendicularity_l2801_280187

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m : Line) (n : Line) (α β γ : Plane) :
  parallel m α → perpendicular m β → perpendicularPlanes α β :=
sorry

end plane_perpendicularity_l2801_280187


namespace inverse_proportion_problem_l2801_280170

/-- Given that x and y are inversely proportional, prove that when x = -12, y = -39.0625 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 50 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) :
  x = -12 → y = -39.0625 := by
sorry

end inverse_proportion_problem_l2801_280170


namespace factor_quadratic_l2801_280174

theorem factor_quadratic (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end factor_quadratic_l2801_280174


namespace min_value_of_function_min_value_achieved_l2801_280109

theorem min_value_of_function (x : ℝ) (h : x > 0) : 4 * x + 9 / x ≥ 12 :=
sorry

theorem min_value_achieved : ∃ x : ℝ, x > 0 ∧ 4 * x + 9 / x = 12 :=
sorry

end min_value_of_function_min_value_achieved_l2801_280109


namespace mikaela_savings_l2801_280106

-- Define the hourly rate
def hourly_rate : ℕ := 10

-- Define the hours worked in the first month
def first_month_hours : ℕ := 35

-- Define the additional hours worked in the second month
def additional_hours : ℕ := 5

-- Define the fraction of earnings spent on personal needs
def spent_fraction : ℚ := 4/5

-- Function to calculate total earnings
def total_earnings (rate : ℕ) (hours1 : ℕ) (hours2 : ℕ) : ℕ :=
  rate * (hours1 + hours2)

-- Function to calculate savings
def savings (total : ℕ) (spent_frac : ℚ) : ℚ :=
  (1 - spent_frac) * total

-- Theorem statement
theorem mikaela_savings :
  savings (total_earnings hourly_rate first_month_hours (first_month_hours + additional_hours)) spent_fraction = 150 := by
  sorry


end mikaela_savings_l2801_280106


namespace correct_young_sample_size_l2801_280179

/-- Represents the stratified sampling problem for a company's employees. -/
structure CompanySampling where
  total_employees : ℕ
  young_employees : ℕ
  sample_size : ℕ
  young_in_sample : ℕ

/-- Theorem stating the correct number of young employees in the sample. -/
theorem correct_young_sample_size (c : CompanySampling) 
    (h1 : c.total_employees = 200)
    (h2 : c.young_employees = 120)
    (h3 : c.sample_size = 25)
    (h4 : c.young_in_sample = c.young_employees * c.sample_size / c.total_employees) :
  c.young_in_sample = 15 := by
  sorry

end correct_young_sample_size_l2801_280179


namespace boys_without_calculators_l2801_280135

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ)
  (h1 : total_boys = 20)
  (h2 : students_with_calculators = 26)
  (h3 : girls_with_calculators = 15) :
  total_boys - (students_with_calculators - girls_with_calculators) = 9 := by
sorry

end boys_without_calculators_l2801_280135


namespace find_a_and_b_l2801_280111

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = {x | 0 < x ∧ x ≤ 2}) ∧
    (A ∪ B a b = {x | x > -2}) ∧
    a = -1 ∧
    b = -2 := by
  sorry

end find_a_and_b_l2801_280111


namespace circle_equation_with_PQ_diameter_l2801_280118

/-- Given circle equation -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + 3 = 0

/-- Given line equation -/
def given_line (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

/-- Intersection points P and Q -/
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  given_circle P.1 P.2 ∧ given_line P.1 P.2 ∧
  given_circle Q.1 Q.2 ∧ given_line Q.1 Q.2 ∧
  P ≠ Q

/-- Circle equation with PQ as diameter -/
def circle_PQ_diameter (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem statement -/
theorem circle_equation_with_PQ_diameter
  (P Q : ℝ × ℝ) (h : intersection_points P Q) :
  ∀ x y : ℝ, circle_PQ_diameter x y ↔
    (x - P.1)^2 + (y - P.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 / 4 :=
sorry

end circle_equation_with_PQ_diameter_l2801_280118


namespace initial_paint_amount_l2801_280147

/-- The amount of paint Jimin used for his house -/
def paint_for_house : ℝ := 4.3

/-- The amount of paint Jimin used for his friend's house -/
def paint_for_friend : ℝ := 4.3

/-- The amount of paint remaining after painting both houses -/
def paint_remaining : ℝ := 8.8

/-- The initial amount of paint Jimin had -/
def initial_paint : ℝ := paint_for_house + paint_for_friend + paint_remaining

theorem initial_paint_amount : initial_paint = 17.4 := by sorry

end initial_paint_amount_l2801_280147


namespace jamie_remaining_capacity_l2801_280197

/-- Jamie's bathroom limit in ounces -/
def bathroom_limit : ℕ := 32

/-- Amount of milk Jamie consumed in ounces -/
def milk_consumed : ℕ := 8

/-- Amount of grape juice Jamie consumed in ounces -/
def grape_juice_consumed : ℕ := 16

/-- Total amount of liquid Jamie consumed before the test -/
def total_consumed : ℕ := milk_consumed + grape_juice_consumed

/-- Theorem: Jamie can drink 8 ounces during the test before needing the bathroom -/
theorem jamie_remaining_capacity : bathroom_limit - total_consumed = 8 := by
  sorry

end jamie_remaining_capacity_l2801_280197


namespace equal_first_two_numbers_l2801_280158

theorem equal_first_two_numbers (a : Fin 17 → ℕ) 
  (h : ∀ i : Fin 17, (a i) ^ (a (i + 1)) = (a ((i + 1) % 17)) ^ (a ((i + 2) % 17))) : 
  a 0 = a 1 := by
  sorry

end equal_first_two_numbers_l2801_280158


namespace prime_power_divisibility_l2801_280124

theorem prime_power_divisibility (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p ∣ a^n → p^n ∣ a^n := by
  sorry

end prime_power_divisibility_l2801_280124


namespace circle_diameter_from_area_l2801_280127

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (D : ℝ) :
  A = 400 * Real.pi →
  A = Real.pi * r^2 →
  D = 2 * r →
  D = 40 := by sorry

end circle_diameter_from_area_l2801_280127


namespace existence_of_special_numbers_l2801_280159

theorem existence_of_special_numbers :
  ∃ (S : Finset ℕ), Finset.card S = 100 ∧
  ∀ (a b c d e : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
  (a * b * c * d * e) % (a + b + c + d + e) = 0 :=
by sorry

end existence_of_special_numbers_l2801_280159


namespace inequality_not_holding_l2801_280150

theorem inequality_not_holding (x y : ℝ) (h : x > y) : ¬(-3*x > -3*y) := by
  sorry

end inequality_not_holding_l2801_280150


namespace polynomial_simplification_l2801_280182

theorem polynomial_simplification (x : ℝ) :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) =
  2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end polynomial_simplification_l2801_280182


namespace solution_set_x_squared_geq_four_l2801_280164

theorem solution_set_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} := by sorry

end solution_set_x_squared_geq_four_l2801_280164


namespace power_gt_one_iff_product_gt_zero_l2801_280173

theorem power_gt_one_iff_product_gt_zero {a b : ℝ} (ha : a > 0) (ha' : a ≠ 1) :
  a^b > 1 ↔ (a - 1) * b > 0 := by
  sorry

end power_gt_one_iff_product_gt_zero_l2801_280173


namespace board_number_generation_l2801_280121

theorem board_number_generation (target : ℕ := 2020) : ∃ a b : ℕ, 20 * a + 21 * b = target := by
  sorry

end board_number_generation_l2801_280121


namespace marks_age_multiple_l2801_280143

theorem marks_age_multiple (mark_current_age : ℕ) (aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age - 3 = 3 * (aaron_current_age - 3) + 1 →
  ∃ x : ℕ, mark_current_age + 4 = x * (aaron_current_age + 4) + 2 →
  x = 2 := by
sorry

end marks_age_multiple_l2801_280143


namespace binary_11010_equals_octal_32_l2801_280184

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal [false, true, false, true, true]) = [3, 2] := by
  sorry

end binary_11010_equals_octal_32_l2801_280184


namespace probability_all_white_is_correct_l2801_280112

def total_balls : ℕ := 18
def white_balls : ℕ := 8
def black_balls : ℕ := 10
def drawn_balls : ℕ := 7

def probability_all_white : ℚ :=
  (Nat.choose white_balls drawn_balls) / (Nat.choose total_balls drawn_balls)

theorem probability_all_white_is_correct :
  probability_all_white = 1 / 3980 := by
  sorry

end probability_all_white_is_correct_l2801_280112


namespace valid_sequences_count_l2801_280103

def is_valid_sequence (s : List ℕ) : Prop :=
  s.length = 8 ∧
  s.toFinset.card = 8 ∧
  ∀ x ∈ s, 1 ≤ x ∧ x ≤ 11 ∧
  ∀ n, 1 ≤ n ∧ n ≤ 8 → (s.take n).sum % n = 0

theorem valid_sequences_count :
  ∃! (sequences : List (List ℕ)),
    sequences.length = 8 ∧
    ∀ s ∈ sequences, is_valid_sequence s :=
by sorry

end valid_sequences_count_l2801_280103


namespace chloe_win_prob_is_25_91_l2801_280190

/-- Represents the probability of rolling a specific number on a six-sided die -/
def roll_probability : ℚ := 1 / 6

/-- Represents the probability of not rolling a '6' on a six-sided die -/
def not_six_probability : ℚ := 5 / 6

/-- Calculates the probability of Chloe winning on her nth turn -/
def chloe_win_nth_turn (n : ℕ) : ℚ :=
  (not_six_probability ^ (3 * n - 1)) * roll_probability

/-- Calculates the sum of the geometric series representing Chloe's win probability -/
def chloe_win_probability : ℚ :=
  (chloe_win_nth_turn 1) / (1 - (not_six_probability ^ 3))

/-- Theorem stating that the probability of Chloe winning is 25/91 -/
theorem chloe_win_prob_is_25_91 : chloe_win_probability = 25 / 91 := by
  sorry

end chloe_win_prob_is_25_91_l2801_280190


namespace max_d_minus_r_value_l2801_280194

theorem max_d_minus_r_value : ∃ (d r : ℕ), 
  (2017 % d = r) ∧ (1029 % d = r) ∧ (725 % d = r) ∧
  (∀ (d' r' : ℕ), (2017 % d' = r') ∧ (1029 % d' = r') ∧ (725 % d' = r') → d' - r' ≤ d - r) ∧
  (d - r = 35) := by
  sorry

end max_d_minus_r_value_l2801_280194


namespace expression_evaluation_l2801_280104

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  (2 / (x + 1) + 1 / (x - 2)) / ((x - 1) / (x - 2)) = Real.sqrt 3 := by
  sorry

end expression_evaluation_l2801_280104


namespace trajectory_is_ellipse_l2801_280148

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of circle M -/
def circle_M : Circle :=
  { center := (-1, 0), radius := 1 }

/-- Definition of circle N -/
def circle_N : Circle :=
  { center := (1, 0), radius := 5 }

/-- Definition of external tangency -/
def is_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Definition of internal tangency -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

/-- Theorem: The trajectory of the center of circle P is an ellipse -/
theorem trajectory_is_ellipse (P : ℝ × ℝ) :
  is_externally_tangent { center := P, radius := 0 } circle_M →
  is_internally_tangent { center := P, radius := 0 } circle_N →
  P.1^2 / 9 + P.2^2 / 8 = 1 := by sorry

end trajectory_is_ellipse_l2801_280148


namespace time_to_paint_remaining_rooms_l2801_280157

/-- Given a painting job with the following conditions:
  - There are 10 rooms in total to be painted
  - Each room takes 8 hours to paint
  - 8 rooms have already been painted
This theorem proves that it will take 16 hours to paint the remaining rooms. -/
theorem time_to_paint_remaining_rooms :
  let total_rooms : ℕ := 10
  let painted_rooms : ℕ := 8
  let time_per_room : ℕ := 8
  let remaining_rooms := total_rooms - painted_rooms
  let time_for_remaining := remaining_rooms * time_per_room
  time_for_remaining = 16 := by sorry

end time_to_paint_remaining_rooms_l2801_280157


namespace max_revenue_theorem_l2801_280163

/-- Represents the advertising allocation problem for a company --/
structure AdvertisingProblem where
  totalTime : ℝ
  totalBudget : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Represents a solution to the advertising allocation problem --/
structure AdvertisingSolution where
  timeA : ℝ
  timeB : ℝ
  revenue : ℝ

/-- Checks if a solution is valid for a given problem --/
def isValidSolution (p : AdvertisingProblem) (s : AdvertisingSolution) : Prop :=
  s.timeA ≥ 0 ∧ s.timeB ≥ 0 ∧
  s.timeA + s.timeB ≤ p.totalTime ∧
  s.timeA * p.rateA + s.timeB * p.rateB ≤ p.totalBudget ∧
  s.revenue = s.timeA * p.revenueA + s.timeB * p.revenueB

/-- Theorem stating that the given solution maximizes revenue --/
theorem max_revenue_theorem (p : AdvertisingProblem)
  (h1 : p.totalTime = 300)
  (h2 : p.totalBudget = 90000)
  (h3 : p.rateA = 500)
  (h4 : p.rateB = 200)
  (h5 : p.revenueA = 0.3)
  (h6 : p.revenueB = 0.2) :
  ∃ (s : AdvertisingSolution),
    isValidSolution p s ∧
    s.timeA = 100 ∧
    s.timeB = 200 ∧
    s.revenue = 70 ∧
    ∀ (s' : AdvertisingSolution), isValidSolution p s' → s'.revenue ≤ s.revenue :=
sorry

end max_revenue_theorem_l2801_280163


namespace skirt_ratio_is_two_thirds_l2801_280161

-- Define the number of skirts in each valley
def purple_skirts : ℕ := 10
def azure_skirts : ℕ := 60

-- Define the relationship between Purple and Seafoam Valley skirts
def seafoam_skirts : ℕ := 4 * purple_skirts

-- Define the ratio of Seafoam to Azure Valley skirts
def skirt_ratio : Rat := seafoam_skirts / azure_skirts

-- Theorem to prove
theorem skirt_ratio_is_two_thirds : skirt_ratio = 2 / 3 := by
  sorry

end skirt_ratio_is_two_thirds_l2801_280161


namespace digital_earth_properties_digital_earth_properties_complete_l2801_280102

/-- Represents the concept of the digital Earth -/
structure DigitalEarth where
  /-- Geographic information technology is the foundation -/
  geoInfoTechFoundation : Prop
  /-- Ability to simulate reality -/
  simulatesReality : Prop
  /-- Manages Earth's information digitally through computer networks -/
  managesInfoDigitally : Prop
  /-- Method of information storage (centralized or not) -/
  centralizedStorage : Bool

/-- Theorem stating the correct properties of the digital Earth -/
theorem digital_earth_properties :
  ∀ (de : DigitalEarth),
    de.geoInfoTechFoundation ∧
    de.simulatesReality ∧
    de.managesInfoDigitally ∧
    ¬de.centralizedStorage :=
by
  sorry

/-- Theorem stating that these are the only correct properties -/
theorem digital_earth_properties_complete (de : DigitalEarth) :
  (de.geoInfoTechFoundation ∧ de.simulatesReality ∧ de.managesInfoDigitally ∧ ¬de.centralizedStorage) ↔
  (de.geoInfoTechFoundation ∧ de.simulatesReality ∧ de.managesInfoDigitally) :=
by
  sorry

end digital_earth_properties_digital_earth_properties_complete_l2801_280102


namespace regular_decagon_diagonal_side_difference_l2801_280188

/-- In a regular decagon inscribed in a circle, the difference between the length of the diagonal 
    connecting vertices 3 apart and the side length is equal to the radius of the circumcircle. -/
theorem regular_decagon_diagonal_side_difference (R : ℝ) : 
  let side_length := 2 * R * Real.sin (π / 10)
  let diagonal_length := 2 * R * Real.sin (3 * π / 10)
  diagonal_length - side_length = R := by sorry

end regular_decagon_diagonal_side_difference_l2801_280188


namespace additional_discount_percentage_l2801_280107

/-- Represents the price of duty shoes in cents -/
def full_price : ℕ := 8500

/-- Represents the first discount percentage for officers who served at least a year -/
def first_discount_percent : ℕ := 20

/-- Represents the price paid by officers who served at least three years in cents -/
def price_three_years : ℕ := 5100

/-- Calculates the price after the first discount -/
def price_after_first_discount : ℕ := full_price - (full_price * first_discount_percent / 100)

/-- Represents the additional discount percentage for officers who served at least three years -/
def additional_discount_percent : ℕ := 25

theorem additional_discount_percentage :
  (price_after_first_discount - price_three_years) * 100 / price_after_first_discount = additional_discount_percent := by
  sorry

end additional_discount_percentage_l2801_280107


namespace negation_of_existence_cubic_inequality_negation_l2801_280125

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ ≤ 0) ↔ (∀ x : ℝ, f x > 0) :=
by sorry

theorem cubic_inequality_negation :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end negation_of_existence_cubic_inequality_negation_l2801_280125


namespace problem_statement_l2801_280119

theorem problem_statement (a b c : ℝ) 
  (eq_condition : a - 2*b + c = 0) 
  (ineq_condition : a + 2*b + c < 0) : 
  b < 0 ∧ b^2 - a*c ≥ 0 := by
  sorry

end problem_statement_l2801_280119


namespace systematic_sampling_smallest_number_l2801_280167

/-- Systematic sampling theorem for a specific case -/
theorem systematic_sampling_smallest_number
  (total_items : ℕ)
  (sample_size : ℕ)
  (highest_drawn : ℕ)
  (h1 : total_items = 32)
  (h2 : sample_size = 8)
  (h3 : highest_drawn = 31)
  (h4 : highest_drawn ≤ total_items)
  : ∃ (smallest_drawn : ℕ),
    smallest_drawn = 3 ∧
    smallest_drawn > 0 ∧
    smallest_drawn ≤ highest_drawn ∧
    (highest_drawn - smallest_drawn) % (total_items / sample_size) = 0 :=
by
  sorry


end systematic_sampling_smallest_number_l2801_280167


namespace triangle_case1_triangle_case2_l2801_280146

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangleConditions (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧ Real.sin t.B = 2 * Real.sin t.A

-- Theorem 1: When C = π/3
theorem triangle_case1 (t : Triangle) (h : triangleConditions t) (hC : t.C = π / 3) :
  t.a = 2 ∧ t.b = 4 := by sorry

-- Theorem 2: When cos C = 1/4
theorem triangle_case2 (t : Triangle) (h : triangleConditions t) (hC : Real.cos t.C = 1 / 4) :
  (1 / 2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 15) / 4 := by sorry

end triangle_case1_triangle_case2_l2801_280146


namespace vending_machine_probability_l2801_280154

-- Define the number of toys
def num_toys : ℕ := 10

-- Define the cost range of toys
def min_cost : ℚ := 1/2
def max_cost : ℚ := 5

-- Define the cost increment
def cost_increment : ℚ := 1/2

-- Define Sam's initial quarters
def initial_quarters : ℕ := 10

-- Define the cost of Sam's favorite toy
def favorite_toy_cost : ℚ := 3

-- Define the function to calculate toy prices
def toy_price (n : ℕ) : ℚ := min_cost + (n - 1) * cost_increment

-- Define the probability of needing to break the twenty-dollar bill
def prob_break_bill : ℚ := 14/15

-- Theorem statement
theorem vending_machine_probability :
  (∀ n ∈ Finset.range num_toys, toy_price n ≤ max_cost) →
  (∀ n ∈ Finset.range num_toys, toy_price n ≥ min_cost) →
  (∀ n ∈ Finset.range (num_toys - 1), toy_price (n + 1) = toy_price n + cost_increment) →
  (favorite_toy_cost ∈ Finset.image toy_price (Finset.range num_toys)) →
  (initial_quarters * (1/4 : ℚ) < favorite_toy_cost) →
  (prob_break_bill = 14/15) :=
sorry

end vending_machine_probability_l2801_280154


namespace square_of_sum_l2801_280108

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end square_of_sum_l2801_280108


namespace unique_cube_prime_l2801_280195

theorem unique_cube_prime (p : ℕ) : Prime p → (∃ n : ℕ, 2 * p + 1 = n ^ 3) ↔ p = 13 := by
  sorry

end unique_cube_prime_l2801_280195


namespace sum_of_cubes_l2801_280122

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : a * b + a * c + b * c = 7) 
  (h3 : a * b * c = -10) : 
  a^3 + b^3 + c^3 = 132 := by
  sorry

end sum_of_cubes_l2801_280122


namespace function_analysis_l2801_280115

def f (x : ℝ) := x^3 - 3*x^2 - 9*x + 11

theorem function_analysis :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1)) ∧
  f (-1) = 16 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f x ≥ f 3) ∧
  f 3 = -16 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) 1, f x < f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo 1 (1 + ε), f x > f 1) ∧
  f 1 = 0 :=
sorry

end function_analysis_l2801_280115


namespace meal_cost_theorem_l2801_280126

-- Define variables for item costs
variable (s c p k : ℝ)

-- Define the equations from the given meals
def meal1_equation : Prop := 2 * s + 5 * c + 2 * p + 3 * k = 6.30
def meal2_equation : Prop := 3 * s + 8 * c + 2 * p + 4 * k = 8.40

-- Theorem to prove
theorem meal_cost_theorem 
  (h1 : meal1_equation s c p k)
  (h2 : meal2_equation s c p k) :
  s + c + p + k = 3.15 := by
  sorry


end meal_cost_theorem_l2801_280126


namespace unique_solution_xy_l2801_280189

theorem unique_solution_xy (x y : ℕ) :
  x * (x + 1) = 4 * y * (y + 1) → (x = 0 ∧ y = 0) := by
  sorry

end unique_solution_xy_l2801_280189


namespace min_turn_angles_sum_l2801_280149

/-- Represents a broken line path in a circular arena -/
structure BrokenLinePath where
  /-- Radius of the circular arena in meters -/
  arena_radius : ℝ
  /-- Total length of the path in meters -/
  total_length : ℝ
  /-- List of angles between consecutive segments in radians -/
  turn_angles : List ℝ

/-- Theorem: The sum of turn angles in a broken line path is at least 2998 radians
    given the specified arena radius and path length -/
theorem min_turn_angles_sum (path : BrokenLinePath)
    (h_radius : path.arena_radius = 10)
    (h_length : path.total_length = 30000) :
    (path.turn_angles.sum ≥ 2998) := by
  sorry


end min_turn_angles_sum_l2801_280149


namespace min_socks_for_pair_is_four_l2801_280172

/-- Represents a drawer of socks with three colors -/
structure SockDrawer :=
  (white : Nat)
  (green : Nat)
  (red : Nat)

/-- Ensures that there is at least one sock of each color -/
def hasAllColors (drawer : SockDrawer) : Prop :=
  drawer.white > 0 ∧ drawer.green > 0 ∧ drawer.red > 0

/-- The minimum number of socks needed to ensure at least two of the same color -/
def minSocksForPair (drawer : SockDrawer) : Nat :=
  4

theorem min_socks_for_pair_is_four (drawer : SockDrawer) 
  (h : hasAllColors drawer) : 
  minSocksForPair drawer = 4 := by
  sorry

#check min_socks_for_pair_is_four

end min_socks_for_pair_is_four_l2801_280172


namespace lunch_pizzas_calculation_l2801_280141

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem lunch_pizzas_calculation :
  total_pizzas - dinner_pizzas = 9 := by
  sorry

end lunch_pizzas_calculation_l2801_280141


namespace square_grid_15_toothpicks_l2801_280168

/-- Calculates the total number of toothpicks in a square grid -/
def toothpicks_in_square_grid (side_length : ℕ) : ℕ :=
  2 * side_length * (side_length + 1)

/-- Theorem: A square grid with sides of 15 toothpicks uses 480 toothpicks in total -/
theorem square_grid_15_toothpicks :
  toothpicks_in_square_grid 15 = 480 := by
  sorry

end square_grid_15_toothpicks_l2801_280168


namespace largest_multiple_of_15_under_400_l2801_280162

theorem largest_multiple_of_15_under_400 : ∃ n : ℕ, n * 15 = 390 ∧ 
  390 < 400 ∧ 
  (∀ m : ℕ, m * 15 < 400 → m * 15 ≤ 390) := by
  sorry

end largest_multiple_of_15_under_400_l2801_280162


namespace sallys_earnings_l2801_280198

theorem sallys_earnings (first_month_earnings : ℝ) : 
  first_month_earnings + (first_month_earnings * 1.1) = 2100 → 
  first_month_earnings = 1000 := by
sorry

end sallys_earnings_l2801_280198


namespace tangent_line_at_point_l2801_280169

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

theorem tangent_line_at_point (x₀ y₀ : ℝ) (h : y₀ = f x₀) :
  let m := (3*x₀^2 - 4*x₀ - 4)  -- Derivative of f at x₀
  (5 : ℝ) * x + y - 2 = 0 ↔ y - y₀ = m * (x - x₀) ∧ x₀ = 1 ∧ y₀ = -3 :=
by sorry

#check tangent_line_at_point

end tangent_line_at_point_l2801_280169


namespace sum_reciprocals_inequality_l2801_280152

theorem sum_reciprocals_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 + 1 / b^2 ≥ 8) ∧ (1 / a + 1 / b + 1 / (a * b) ≥ 8) := by
  sorry

end sum_reciprocals_inequality_l2801_280152


namespace beryl_radishes_l2801_280131

def radishes_problem (first_basket : ℕ) (difference : ℕ) : Prop :=
  let second_basket := first_basket + difference
  let total := first_basket + second_basket
  total = 88

theorem beryl_radishes : radishes_problem 37 14 := by
  sorry

end beryl_radishes_l2801_280131


namespace count_with_zero_1000_l2801_280117

def count_with_zero (n : ℕ) : ℕ :=
  (n + 1) - (9 * 9 * 10)

theorem count_with_zero_1000 : count_with_zero 1000 = 181 := by
  sorry

end count_with_zero_1000_l2801_280117


namespace remainder_3_pow_23_mod_11_l2801_280180

theorem remainder_3_pow_23_mod_11 : 3^23 % 11 = 5 := by sorry

end remainder_3_pow_23_mod_11_l2801_280180


namespace absolute_difference_always_less_than_one_l2801_280171

theorem absolute_difference_always_less_than_one :
  ∀ (m : ℝ), ∀ (x : ℝ), |x - m| < 1 :=
by sorry

end absolute_difference_always_less_than_one_l2801_280171
