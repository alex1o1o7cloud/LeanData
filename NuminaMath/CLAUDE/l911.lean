import Mathlib

namespace system_solution_l911_91156

theorem system_solution :
  ∃! (x y : ℚ), (4 * x - 3 * y = -7) ∧ (5 * x + 4 * y = -2) ∧ x = -34/31 ∧ y = 27/31 := by
  sorry

end system_solution_l911_91156


namespace age_ratio_constant_l911_91134

/-- Given two people p and q, where the ratio of their present ages is 3:4 and their total age is 28,
    prove that p's age was always 3/4 of q's age at any point in the past. -/
theorem age_ratio_constant
  (p q : ℕ) -- present ages of p and q
  (h1 : p * 4 = q * 3) -- ratio of present ages is 3:4
  (h2 : p + q = 28) -- total present age is 28
  (t : ℕ) -- time in the past
  (h3 : t ≤ min p q) -- ensure t is not greater than either age
  : (p - t) * 4 = (q - t) * 3 :=
sorry

end age_ratio_constant_l911_91134


namespace root_modulus_preservation_l911_91189

theorem root_modulus_preservation (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∀ z : ℂ, z^3 + Complex.abs a*z^2 + Complex.abs b*z + Complex.abs c = 0 → Complex.abs z = 1) :=
by sorry

end root_modulus_preservation_l911_91189


namespace square_sum_reciprocal_l911_91155

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end square_sum_reciprocal_l911_91155


namespace yellow_peaches_undetermined_l911_91101

def basket_peaches (red green yellow : ℕ) : Prop :=
  red = 7 ∧ green = 8 ∧ green = red + 1

theorem yellow_peaches_undetermined :
  ∀ (red green yellow : ℕ),
    basket_peaches red green yellow →
    ¬∃ (n : ℕ), ∀ (y : ℕ), basket_peaches red green y → y = n :=
by
  sorry

end yellow_peaches_undetermined_l911_91101


namespace car_average_speed_l911_91126

/-- Proves that the average speed of a car is 40 km/h given the specified conditions -/
theorem car_average_speed : ∀ (s : ℝ), s > 0 →
  ∃ (v : ℝ), v > 0 ∧
  (s / (s / (2 * (v + 30)) + s / (1.4 * v)) = v) ∧
  v = 40 := by
sorry

end car_average_speed_l911_91126


namespace white_then_red_prob_is_one_thirtieth_l911_91195

/-- A type representing the colors of balls in the bag -/
inductive Color
| Red | Blue | Green | Yellow | Purple | White

/-- The total number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of colored balls (excluding white) in the bag -/
def colored_balls : ℕ := 5

/-- The probability of drawing a specific ball from the bag -/
def draw_probability (n : ℕ) : ℚ := 1 / n

/-- The probability of drawing the white ball first and the red ball second -/
def white_then_red_probability : ℚ :=
  draw_probability total_balls * draw_probability colored_balls

/-- Theorem stating that the probability of drawing the white ball first
    and the red ball second is 1/30 -/
theorem white_then_red_prob_is_one_thirtieth :
  white_then_red_probability = 1 / 30 := by
  sorry

end white_then_red_prob_is_one_thirtieth_l911_91195


namespace duck_pond_problem_l911_91151

theorem duck_pond_problem :
  let small_pond_total : ℕ := 30
  let small_pond_green_ratio : ℚ := 1/5
  let large_pond_green_ratio : ℚ := 3/25
  let total_green_ratio : ℚ := 3/20
  ∃ (large_pond_total : ℕ),
    (small_pond_green_ratio * small_pond_total + large_pond_green_ratio * large_pond_total : ℚ) = 
    total_green_ratio * (small_pond_total + large_pond_total) ∧
    large_pond_total = 50 :=
by
  sorry

end duck_pond_problem_l911_91151


namespace lending_scenario_l911_91185

/-- Proves that given the conditions of the lending scenario, the principal amount is 3500 Rs. -/
theorem lending_scenario (P : ℝ) 
  (h1 : P + (P * 0.1 * 3) = 1.3 * P)  -- B owes A after 3 years
  (h2 : P + (P * 0.12 * 3) = 1.36 * P)  -- C owes B after 3 years
  (h3 : 1.36 * P - 1.3 * P = 210)  -- B's gain over 3 years
  : P = 3500 := by
  sorry

#check lending_scenario

end lending_scenario_l911_91185


namespace salary_calculation_l911_91198

def monthly_salary : ℝ → Prop := λ s => 
  let original_savings := 0.2 * s
  let original_expenses := 0.8 * s
  let increased_expenses := 1.2 * original_expenses
  s - increased_expenses = 250

theorem salary_calculation : ∃ s : ℝ, monthly_salary s ∧ s = 6250 := by
  sorry

end salary_calculation_l911_91198


namespace donated_area_is_108_45_l911_91128

/-- Calculates the total area of cloth donated given the areas and percentages of three cloths. -/
def total_donated_area (cloth1_area cloth2_area cloth3_area : ℝ)
  (cloth1_keep_percent cloth2_keep_percent cloth3_keep_percent : ℝ) : ℝ :=
  let cloth1_donate := cloth1_area * (1 - cloth1_keep_percent)
  let cloth2_donate := cloth2_area * (1 - cloth2_keep_percent)
  let cloth3_donate := cloth3_area * (1 - cloth3_keep_percent)
  cloth1_donate + cloth2_donate + cloth3_donate

/-- Theorem stating that the total donated area is 108.45 square inches. -/
theorem donated_area_is_108_45 :
  total_donated_area 100 65 48 0.4 0.55 0.6 = 108.45 := by
  sorry

end donated_area_is_108_45_l911_91128


namespace new_person_weight_l911_91103

theorem new_person_weight 
  (n : ℕ) 
  (initial_weight : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) :
  n = 8 →
  weight_increase = 2.5 →
  replaced_weight = 75 →
  initial_weight + (n : ℝ) * weight_increase = initial_weight - replaced_weight + 95 :=
by sorry

end new_person_weight_l911_91103


namespace imaginary_power_sum_l911_91160

theorem imaginary_power_sum : Complex.I ^ 22 + Complex.I ^ 222 = -2 := by sorry

end imaginary_power_sum_l911_91160


namespace range_of_a_l911_91192

-- Define the propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*m*a + 12*a^2 < 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ x^2 / (m - 1) + y^2 / (2 - m - c) = 1

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (∀ m : ℝ, (¬(p m a) → ¬(q m)) ∧ ∃ m : ℝ, ¬(p m a) ∧ q m) →
  a ∈ Set.Icc (1/3 : ℝ) (3/8 : ℝ) :=
sorry

end range_of_a_l911_91192


namespace square_rectangle_area_relationship_l911_91179

theorem square_rectangle_area_relationship : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, 3 * (x - 2)^2 = (x - 3) * (x + 4) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 19/2 := by
  sorry

end square_rectangle_area_relationship_l911_91179


namespace simplify_expression_l911_91141

theorem simplify_expression (a : ℝ) (h : a < 2) : 
  Real.sqrt ((a - 2)^2) + a - 1 = 1 := by
sorry

end simplify_expression_l911_91141


namespace triangle_angle_not_all_greater_60_l911_91149

theorem triangle_angle_not_all_greater_60 :
  ∀ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Angles are positive
  (a + b + c = 180) →        -- Sum of angles in a triangle is 180°
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by
  sorry

end triangle_angle_not_all_greater_60_l911_91149


namespace composite_evaluation_l911_91132

/-- A polynomial with coefficients either 0 or 1 -/
def BinaryPolynomial (P : Polynomial ℤ) : Prop :=
  ∀ i, P.coeff i = 0 ∨ P.coeff i = 1

/-- A polynomial is nonconstant -/
def IsNonconstant (P : Polynomial ℤ) : Prop :=
  ∃ i > 0, P.coeff i ≠ 0

theorem composite_evaluation
  (P : Polynomial ℤ)
  (h_binary : BinaryPolynomial P)
  (h_factorizable : ∃ (f g : Polynomial ℤ), P = f * g ∧ IsNonconstant f ∧ IsNonconstant g) :
  ∃ (a b : ℤ), a > 1 ∧ b > 1 ∧ P.eval 2 = a * b :=
sorry

end composite_evaluation_l911_91132


namespace percentage_increase_l911_91199

theorem percentage_increase (x : ℝ) (h : x = 123.2) : 
  (x - 88) / 88 * 100 = 40 := by
  sorry

end percentage_increase_l911_91199


namespace p_necessary_not_sufficient_for_q_l911_91133

-- Define the conditions
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - a ≤ 0
def q (a : ℝ) : Prop := a > 0 ∨ a < -1

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ 
  (∃ a : ℝ, p a ∧ ¬(q a)) :=
by sorry

end p_necessary_not_sufficient_for_q_l911_91133


namespace equivalent_discount_l911_91102

theorem equivalent_discount (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.15
  let second_discount := 0.25
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let equivalent_discount := 1 - (price_after_second / original_price)
  equivalent_discount = 0.3625 := by
sorry

end equivalent_discount_l911_91102


namespace lea_binders_purchase_l911_91190

/-- The number of binders Léa bought -/
def num_binders : ℕ := 3

/-- The cost of one book in dollars -/
def book_cost : ℕ := 16

/-- The cost of one binder in dollars -/
def binder_cost : ℕ := 2

/-- The cost of one notebook in dollars -/
def notebook_cost : ℕ := 1

/-- The number of notebooks Léa bought -/
def num_notebooks : ℕ := 6

/-- The total cost of Léa's purchases in dollars -/
def total_cost : ℕ := 28

theorem lea_binders_purchase :
  book_cost + binder_cost * num_binders + notebook_cost * num_notebooks = total_cost :=
by sorry

end lea_binders_purchase_l911_91190


namespace smallest_number_divisible_l911_91140

theorem smallest_number_divisible (n : ℕ) : n = 257 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) % 8 = 0 ∧ (m + 7) % 11 = 0 ∧ (m + 7) % 24 = 0)) ∧
  (n + 7) % 8 = 0 ∧ (n + 7) % 11 = 0 ∧ (n + 7) % 24 = 0 :=
by sorry

#check smallest_number_divisible

end smallest_number_divisible_l911_91140


namespace interest_calculation_l911_91191

def deposit : ℝ := 30000
def term : ℝ := 3
def interest_rate : ℝ := 0.047
def tax_rate : ℝ := 0.2

def pre_tax_interest : ℝ := deposit * interest_rate * term
def after_tax_interest : ℝ := pre_tax_interest * (1 - tax_rate)
def total_withdrawal : ℝ := deposit + after_tax_interest

theorem interest_calculation :
  after_tax_interest = 3372 ∧ total_withdrawal = 33372 := by
  sorry

end interest_calculation_l911_91191


namespace monthly_growth_rate_correct_max_daily_tourists_may_correct_l911_91118

-- Define the number of tourists in February and April
def tourists_february : ℝ := 16000
def tourists_april : ℝ := 25000

-- Define the number of tourists from May 1st to May 21st
def tourists_may_21 : ℝ := 21250

-- Define the monthly average growth rate
def monthly_growth_rate : ℝ := 0.25

-- Define the function to calculate the growth over two months
def two_month_growth (initial : ℝ) (rate : ℝ) : ℝ :=
  initial * (1 + rate) ^ 2

-- Define the function to calculate the maximum number of tourists in May
def max_tourists_may (rate : ℝ) : ℝ :=
  tourists_april * (1 + rate)

-- Theorem 1: Prove the monthly average growth rate
theorem monthly_growth_rate_correct :
  two_month_growth tourists_february monthly_growth_rate = tourists_april :=
sorry

-- Theorem 2: Prove the maximum average number of tourists per day in the last 10 days of May
theorem max_daily_tourists_may_correct :
  (max_tourists_may monthly_growth_rate - tourists_may_21) / 10 = 100000 :=
sorry

end monthly_growth_rate_correct_max_daily_tourists_may_correct_l911_91118


namespace no_prime_satisfies_equation_l911_91130

theorem no_prime_satisfies_equation : ¬∃ (p : ℕ), 
  Nat.Prime p ∧ 
  (2 * p^2 + 5 * p + 3) + (5 * p^2 + p + 2) + (p^2 + 1) + (2 * p^2 + 4 * p + 3) + (p^2 + 6) = 
  (7 * p^2 + 6 * p + 5) + (4 * p^2 + 3 * p + 2) + (p^2 + 2 * p) := by
  sorry

#check no_prime_satisfies_equation

end no_prime_satisfies_equation_l911_91130


namespace product_of_one_fourth_and_one_half_l911_91143

theorem product_of_one_fourth_and_one_half : (1 / 4 : ℚ) * (1 / 2 : ℚ) = 1 / 8 := by
  sorry

end product_of_one_fourth_and_one_half_l911_91143


namespace inverse_proportion_percentage_change_l911_91144

/-- Proves the percentage decrease in b when a increases by q% for inversely proportional variables -/
theorem inverse_proportion_percentage_change 
  (a b : ℝ) (q : ℝ) (c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : q > 0) :
  (a * b = c) →  -- inverse proportionality condition
  let a' := a * (1 + q / 100)  -- a increased by q%
  let b' := c / a'  -- new b value
  (b - b') / b * 100 = 100 * q / (100 + q) :=
by sorry

end inverse_proportion_percentage_change_l911_91144


namespace log_equation_implies_y_equals_nine_l911_91178

theorem log_equation_implies_y_equals_nine 
  (x y : ℝ) 
  (h : x > 0) 
  (h2x : 2*x > 0) 
  (hy : y > 0) : 
  (Real.log x / Real.log 3) * (Real.log (2*x) / Real.log x) * (Real.log y / Real.log (2*x)) = 2 → 
  y = 9 := by
sorry

end log_equation_implies_y_equals_nine_l911_91178


namespace line_slope_problem_l911_91177

theorem line_slope_problem (a : ℝ) : 
  (3 * a - 7) / (a - 2) = 2 → a = 3 := by
  sorry

end line_slope_problem_l911_91177


namespace spider_dressing_8_pairs_l911_91174

/-- The number of ways a spider can put on n pairs of socks and shoes -/
def spiderDressingWays (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (2^n)

/-- Theorem: For 8 pairs of socks and shoes, the number of ways is 81729648000 -/
theorem spider_dressing_8_pairs :
  spiderDressingWays 8 = 81729648000 := by
  sorry

end spider_dressing_8_pairs_l911_91174


namespace quadratic_roots_l911_91162

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_roots (b c : ℝ) :
  (f b c (-2) = 5) →
  (f b c (-1) = 0) →
  (f b c 0 = -3) →
  (f b c 1 = -4) →
  (f b c 2 = -3) →
  (f b c 4 = 5) →
  (∃ x, f b c x = 0) →
  (∀ x, f b c x = 0 ↔ (x = -1 ∨ x = 3)) :=
by sorry

end quadratic_roots_l911_91162


namespace largest_eight_digit_with_even_digits_l911_91113

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
  n ≤ 99986420 :=
sorry

end largest_eight_digit_with_even_digits_l911_91113


namespace cloth_selling_price_l911_91120

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Proves that the total selling price for 85 meters of cloth with a profit of Rs. 25 per meter 
    and a cost price of Rs. 80 per meter is Rs. 8925 -/
theorem cloth_selling_price :
  totalSellingPrice 85 25 80 = 8925 := by
  sorry

end cloth_selling_price_l911_91120


namespace sum_odd_integers_7_to_35_l911_91197

/-- The sum of odd integers from 7 to 35 (inclusive) is 315 -/
theorem sum_odd_integers_7_to_35 : 
  (Finset.range 15).sum (fun i => 2 * i + 7) = 315 := by
  sorry

end sum_odd_integers_7_to_35_l911_91197


namespace algebraic_expression_value_l911_91145

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : y - x = -1) 
  (h2 : x * y = 2) : 
  -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3 = -4 := by
sorry

end algebraic_expression_value_l911_91145


namespace solution_set_abs_inequality_l911_91110

theorem solution_set_abs_inequality :
  {x : ℝ | |2 - x| < 5} = {x : ℝ | -3 < x ∧ x < 7} := by sorry

end solution_set_abs_inequality_l911_91110


namespace avery_donation_total_l911_91123

/-- Proves that the total number of clothes Avery donates is 16 -/
theorem avery_donation_total (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 4 →
  pants = 2 * shirts →
  shorts = pants / 2 →
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_total_l911_91123


namespace book_chapters_l911_91148

/-- The number of chapters in a book, given the number of chapters read per day and the number of days taken to finish the book. -/
def total_chapters (chapters_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  chapters_per_day * days_to_finish

/-- Theorem stating that the total number of chapters in the book is 220,448. -/
theorem book_chapters :
  total_chapters 332 664 = 220448 := by
  sorry

end book_chapters_l911_91148


namespace smallest_sum_is_14_l911_91129

/-- Represents a pentagon arrangement of numbers 1 through 10 -/
structure PentagonArrangement where
  vertices : Fin 5 → Fin 10
  sides : Fin 5 → Fin 10
  all_used : ∀ n : Fin 10, (n ∈ Set.range vertices) ∨ (n ∈ Set.range sides)
  distinct : Function.Injective vertices ∧ Function.Injective sides

/-- The sum along each side of the pentagon -/
def side_sum (arr : PentagonArrangement) : ℕ → ℕ
| 0 => arr.vertices 0 + arr.sides 0 + arr.vertices 1
| 1 => arr.vertices 1 + arr.sides 1 + arr.vertices 2
| 2 => arr.vertices 2 + arr.sides 2 + arr.vertices 3
| 3 => arr.vertices 3 + arr.sides 3 + arr.vertices 4
| 4 => arr.vertices 4 + arr.sides 4 + arr.vertices 0
| _ => 0

/-- The arrangement is valid if all side sums are equal -/
def is_valid_arrangement (arr : PentagonArrangement) : Prop :=
  ∀ i j : Fin 5, side_sum arr i = side_sum arr j

/-- The main theorem: the smallest possible sum is 14 -/
theorem smallest_sum_is_14 :
  ∃ (arr : PentagonArrangement), is_valid_arrangement arr ∧
  (∀ i : Fin 5, side_sum arr i = 14) ∧
  (∀ arr' : PentagonArrangement, is_valid_arrangement arr' →
    ∀ i : Fin 5, side_sum arr' i ≥ 14) :=
  sorry

end smallest_sum_is_14_l911_91129


namespace tournament_balls_count_l911_91158

def tournament_rounds : ℕ := 7

def games_per_round : List ℕ := [64, 32, 16, 8, 4, 2, 1]

def cans_per_game : ℕ := 6

def balls_per_can : ℕ := 4

def total_balls : ℕ := (games_per_round.sum * cans_per_game * balls_per_can)

theorem tournament_balls_count :
  total_balls = 3048 :=
by sorry

end tournament_balls_count_l911_91158


namespace lcm_product_hcf_l911_91111

theorem lcm_product_hcf (a b : ℕ+) (h1 : Nat.lcm a b = 750) (h2 : a * b = 18750) :
  Nat.gcd a b = 25 := by
  sorry

end lcm_product_hcf_l911_91111


namespace football_season_games_l911_91112

/-- The number of months in the football season -/
def season_months : ℕ := 17

/-- The number of games played each month -/
def games_per_month : ℕ := 19

/-- The total number of games played during the season -/
def total_games : ℕ := season_months * games_per_month

theorem football_season_games :
  total_games = 323 :=
by sorry

end football_season_games_l911_91112


namespace equation_solutions_l911_91161

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 36 = 0 ↔ x = 6 ∨ x = -6) ∧
  (∀ x : ℝ, (x+1)^3 + 27 = 0 ↔ x = -4) :=
by sorry

end equation_solutions_l911_91161


namespace max_value_xy_x_minus_y_l911_91186

theorem max_value_xy_x_minus_y (x y : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) :
  x * y * (x - y) ≤ (1 : ℝ) / 4 := by
  sorry

end max_value_xy_x_minus_y_l911_91186


namespace jan_claims_l911_91196

/-- The number of claims each agent can handle --/
structure AgentClaims where
  missy : ℕ
  john : ℕ
  jan : ℕ

/-- Conditions for the insurance claims problem --/
def insurance_claims_conditions (claims : AgentClaims) : Prop :=
  claims.missy = 41 ∧
  claims.missy = claims.john + 15 ∧
  claims.john = claims.jan + (claims.jan / 10) * 3

/-- Theorem stating that under the given conditions, Jan can handle 20 claims --/
theorem jan_claims (claims : AgentClaims) 
  (h : insurance_claims_conditions claims) : claims.jan = 20 := by
  sorry


end jan_claims_l911_91196


namespace common_divisors_9240_10080_l911_91136

theorem common_divisors_9240_10080 : Nat.card {d : ℕ | d ∣ 9240 ∧ d ∣ 10080} = 48 := by
  sorry

end common_divisors_9240_10080_l911_91136


namespace min_cards_to_form_square_l911_91194

/-- Represents the width of the rectangular card in centimeters -/
def card_width : ℕ := 20

/-- Represents the length of the rectangular card in centimeters -/
def card_length : ℕ := 8

/-- Represents the area of a single card in square centimeters -/
def card_area : ℕ := card_width * card_length

/-- Represents the side length of the smallest square that can be formed -/
def square_side : ℕ := Nat.lcm card_width card_length

/-- Represents the area of the smallest square that can be formed -/
def square_area : ℕ := square_side * square_side

/-- The minimum number of cards needed to form the smallest square -/
def min_cards : ℕ := square_area / card_area

theorem min_cards_to_form_square : min_cards = 10 := by
  sorry

end min_cards_to_form_square_l911_91194


namespace retail_price_calculation_l911_91142

def calculate_retail_price (wholesale : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let intended_price := wholesale * (1 + profit_percent)
  intended_price * (1 - discount_percent)

def overall_retail_price (w1 w2 w3 : ℝ) (p1 p2 p3 : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  calculate_retail_price w1 p1 d1 +
  calculate_retail_price w2 p2 d2 +
  calculate_retail_price w3 p3 d3

theorem retail_price_calculation :
  overall_retail_price 90 120 200 0.20 0.30 0.25 0.10 0.15 0.05 = 467.30 := by
  sorry

end retail_price_calculation_l911_91142


namespace three_digit_sum_reduction_l911_91165

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  let sum := d1 + d2 + d3
  let n_plus_3 := n + 3
  let d1_new := n_plus_3 / 100
  let d2_new := (n_plus_3 / 10) % 10
  let d3_new := n_plus_3 % 10
  let sum_new := d1_new + d2_new + d3_new
  sum_new = sum / 3

theorem three_digit_sum_reduction :
  ∀ n : ℕ, is_valid_number n ↔ n = 117 ∨ n = 207 ∨ n = 108 :=
sorry

end three_digit_sum_reduction_l911_91165


namespace quadratic_symmetry_axis_l911_91157

/-- A quadratic function passing through points (-4,m) and (2,m) has its axis of symmetry at x = -1 -/
theorem quadratic_symmetry_axis (f : ℝ → ℝ) (m : ℝ) : 
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is a quadratic function
  f (-4) = m →                                    -- f passes through (-4,m)
  f 2 = m →                                       -- f passes through (2,m)
  (∀ x, f (x - 1) = f (-x - 1)) :=                -- axis of symmetry is x = -1
by sorry

end quadratic_symmetry_axis_l911_91157


namespace cupcake_price_is_one_fifty_l911_91152

/-- Represents the daily production and prices of bakery items -/
structure BakeryProduction where
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ
  cookie_price_per_packet : ℚ
  biscuit_price_per_packet : ℚ

/-- Calculates the price of a cupcake given the bakery production and total earnings -/
def calculate_cupcake_price (prod : BakeryProduction) (days : ℕ) (total_earnings : ℚ) : ℚ :=
  let total_cookies_earnings := prod.cookie_packets_per_day * days * prod.cookie_price_per_packet
  let total_biscuits_earnings := prod.biscuit_packets_per_day * days * prod.biscuit_price_per_packet
  let cupcakes_earnings := total_earnings - total_cookies_earnings - total_biscuits_earnings
  cupcakes_earnings / (prod.cupcakes_per_day * days)

/-- Theorem stating that the cupcake price is $1.50 given the specified conditions -/
theorem cupcake_price_is_one_fifty :
  let prod : BakeryProduction := {
    cupcakes_per_day := 20,
    cookie_packets_per_day := 10,
    biscuit_packets_per_day := 20,
    cookie_price_per_packet := 2,
    biscuit_price_per_packet := 1
  }
  let days : ℕ := 5
  let total_earnings : ℚ := 350
  calculate_cupcake_price prod days total_earnings = 3/2 := by
  sorry


end cupcake_price_is_one_fifty_l911_91152


namespace cats_owners_percentage_l911_91124

/-- The percentage of students who own cats, given 75 out of 450 students own cats. -/
def percentage_cats_owners : ℚ :=
  75 / 450 * 100

/-- Theorem: The percentage of students who own cats is 16.6% (recurring). -/
theorem cats_owners_percentage :
  percentage_cats_owners = 50 / 3 := by
  sorry

end cats_owners_percentage_l911_91124


namespace f_satisfies_equation_l911_91117

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem f_satisfies_equation : ∀ x : ℝ, 2 * (f x) + f (-x) = 3 * x := by
  sorry

end f_satisfies_equation_l911_91117


namespace least_possible_bc_length_l911_91183

theorem least_possible_bc_length 
  (AB AC DC BD : ℝ) 
  (hAB : AB = 8) 
  (hAC : AC = 10) 
  (hDC : DC = 7) 
  (hBD : BD = 15) : 
  ∃ (BC : ℕ), BC = 9 ∧ 
    BC > AC - AB ∧ 
    BC > BD - DC ∧ 
    ∀ (n : ℕ), n < 9 → (n ≤ AC - AB ∨ n ≤ BD - DC) :=
by sorry

end least_possible_bc_length_l911_91183


namespace special_polynomial_form_l911_91180

/-- A polynomial satisfying the given functional equation. -/
structure SpecialPolynomial where
  P : ℝ → ℝ
  equation_holds : ∀ (a b c : ℝ),
    P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) =
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem special_polynomial_form (p : SpecialPolynomial) :
  ∃ (a b : ℝ), (∀ x, p.P x = a * x + b) ∨ (∀ x, p.P x = a * x^2 + b * x) := by
  sorry

end special_polynomial_form_l911_91180


namespace smallest_constant_for_triangle_sides_l911_91131

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem smallest_constant_for_triangle_sides (t : Triangle) :
  (t.a^2 + t.b^2) / (t.a * t.b) ≥ 2 ∧
  ∀ N, (∀ t' : Triangle, (t'.a^2 + t'.b^2) / (t'.a * t'.b) < N) → N ≥ 2 :=
sorry

end smallest_constant_for_triangle_sides_l911_91131


namespace complex_polygon_area_l911_91163

/-- A complex polygon with specific properties -/
structure ComplexPolygon where
  sides : Nat
  side_length : ℝ
  perimeter : ℝ
  is_perpendicular : Bool
  is_congruent : Bool

/-- The area of the complex polygon -/
noncomputable def polygon_area (p : ComplexPolygon) : ℝ :=
  96

/-- Theorem stating the area of the specific complex polygon -/
theorem complex_polygon_area 
  (p : ComplexPolygon) 
  (h1 : p.sides = 32) 
  (h2 : p.perimeter = 64) 
  (h3 : p.is_perpendicular = true) 
  (h4 : p.is_congruent = true) : 
  polygon_area p = 96 := by
  sorry


end complex_polygon_area_l911_91163


namespace tv_screen_height_l911_91122

/-- The height of a rectangular TV screen given its area and width -/
theorem tv_screen_height (area width : ℝ) (h_area : area = 21) (h_width : width = 3) :
  area / width = 7 := by
  sorry

end tv_screen_height_l911_91122


namespace minimum_dresses_for_six_colors_one_style_l911_91170

theorem minimum_dresses_for_six_colors_one_style 
  (num_colors : ℕ) 
  (num_styles : ℕ) 
  (max_extraction_time : ℕ) 
  (h1 : num_colors = 10)
  (h2 : num_styles = 9)
  (h3 : max_extraction_time = 60) :
  ∃ (min_dresses : ℕ),
    (∀ (n : ℕ), n < min_dresses → 
      ¬(∃ (style : ℕ), style < num_styles ∧ 
        (∃ (colors : Finset ℕ), colors.card = 6 ∧ 
          (∀ (c : ℕ), c ∈ colors → c < num_colors) ∧
          (∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2 → 
            ∃ (t1 t2 : ℕ), t1 < max_extraction_time ∧ t2 < max_extraction_time ∧ t1 ≠ t2)))) ∧
    (∃ (style : ℕ), style < num_styles ∧ 
      (∃ (colors : Finset ℕ), colors.card = 6 ∧ 
        (∀ (c : ℕ), c ∈ colors → c < num_colors) ∧
        (∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2 → 
          ∃ (t1 t2 : ℕ), t1 < max_extraction_time ∧ t2 < max_extraction_time ∧ t1 ≠ t2))) ∧
    min_dresses = 46 :=
by
  sorry

end minimum_dresses_for_six_colors_one_style_l911_91170


namespace dividend_calculation_l911_91193

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 19)
  (h3 : remainder = 2) : 
  divisor * quotient + remainder = 686 := by
sorry

end dividend_calculation_l911_91193


namespace fixed_point_of_exponential_function_l911_91106

/-- Given a > 0 and a ≠ 1, prove that the function f(x) = a^(x+2) - 2 
    always passes through the point (-2, -1) regardless of the value of a -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 2
  f (-2) = -1 := by sorry

end fixed_point_of_exponential_function_l911_91106


namespace principal_calculation_l911_91182

/-- Proves that the principal amount is 1500 given the specified conditions --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2.4 →
  amount = 1680 →
  (1 + rate * time) * 1500 = amount :=
by sorry

end principal_calculation_l911_91182


namespace sum_of_roots_quadratic_equation_l911_91121

theorem sum_of_roots_quadratic_equation :
  let a : ℝ := -3
  let b : ℝ := -27
  let c : ℝ := 81
  let equation := fun x : ℝ => a * x^2 + b * x + c
  ∃ r s : ℝ, equation r = 0 ∧ equation s = 0 ∧ r + s = -b / a :=
by
  sorry

end sum_of_roots_quadratic_equation_l911_91121


namespace shifted_parabola_equation_l911_91164

/-- Represents a parabola in the form y = (x - h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { h := 0, k := 0 }

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (units : ℝ) : Parabola :=
  { h := p.h - units, k := p.k }

/-- The equation of a parabola in terms of x -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem shifted_parabola_equation :
  let shifted := shift_parabola original_parabola 2
  ∀ x, parabola_equation shifted x = (x + 2)^2 := by
  sorry

end shifted_parabola_equation_l911_91164


namespace leisure_park_ticket_cost_l911_91167

/-- The cost of tickets for a family visit to a leisure park -/
theorem leisure_park_ticket_cost :
  ∀ (child_ticket : ℕ),
  child_ticket * 5 + (child_ticket + 8) * 2 + (child_ticket + 4) * 2 = 150 →
  child_ticket + 8 = 22 := by
  sorry

end leisure_park_ticket_cost_l911_91167


namespace publishing_profit_inequality_minimum_sets_answer_is_four_thousand_l911_91146

/-- Represents the number of thousands of sets -/
def x : ℝ := 4

/-- Fixed cost in yuan -/
def fixed_cost : ℝ := 80000

/-- Cost increase per set in yuan -/
def cost_increase : ℝ := 20

/-- Price per set in yuan -/
def price : ℝ := 100

/-- Underwriter's share of sales -/
def underwriter_share : ℝ := 0.3

/-- Publishing house's desired profit margin -/
def profit_margin : ℝ := 0.1

/-- The inequality that must be satisfied for the publishing house to achieve its desired profit -/
theorem publishing_profit_inequality :
  fixed_cost + cost_increase * 1000 * x ≤ price * (1 - underwriter_share - profit_margin) * 1000 * x :=
sorry

/-- The minimum number of sets (in thousands) that satisfies the inequality -/
theorem minimum_sets :
  x = ⌈(fixed_cost / (price * (1 - underwriter_share - profit_margin) * 1000 - cost_increase * 1000))⌉ :=
sorry

/-- Proof that 4,000 sets is the correct answer when rounded to the nearest thousand -/
theorem answer_is_four_thousand :
  ⌊x * 1000 / 1000 + 0.5⌋ * 1000 = 4000 :=
sorry

end publishing_profit_inequality_minimum_sets_answer_is_four_thousand_l911_91146


namespace triangle_side_length_l911_91107

/-- An equilateral triangle divided into three congruent trapezoids -/
structure TriangleDivision where
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- The length of the shorter base of each trapezoid -/
  trapezoid_short_base : ℝ
  /-- The length of the longer base of each trapezoid -/
  trapezoid_long_base : ℝ
  /-- The length of the legs of each trapezoid -/
  trapezoid_leg : ℝ
  /-- The trapezoids are congruent -/
  congruent_trapezoids : trapezoid_long_base = 2 * trapezoid_short_base
  /-- The triangle is divided into three trapezoids -/
  triangle_composition : triangle_side = trapezoid_short_base + 2 * trapezoid_leg
  /-- The perimeter of each trapezoid is 10 + 5√3 -/
  trapezoid_perimeter : trapezoid_short_base + trapezoid_long_base + 2 * trapezoid_leg = 10 + 5 * Real.sqrt 3

/-- Theorem: The side length of the equilateral triangle is 6 + 3√3 -/
theorem triangle_side_length (td : TriangleDivision) : td.triangle_side = 6 + 3 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l911_91107


namespace treble_double_plus_five_l911_91138

theorem treble_double_plus_five (initial_number : ℕ) : initial_number = 15 → 
  3 * (2 * initial_number + 5) = 105 := by
  sorry

end treble_double_plus_five_l911_91138


namespace sqrt_2x_minus_1_condition_l911_91105

-- Define the condition for the square root to be meaningful
def is_meaningful (x : ℝ) : Prop := 2 * x - 1 ≥ 0

-- State the theorem
theorem sqrt_2x_minus_1_condition (x : ℝ) :
  is_meaningful x ↔ x ≥ 1/2 := by
  sorry

end sqrt_2x_minus_1_condition_l911_91105


namespace minute_hand_rotation_l911_91127

-- Define the constants
def full_rotation_minutes : ℝ := 60
def full_rotation_degrees : ℝ := 360
def minutes_moved : ℝ := 10

-- Define the theorem
theorem minute_hand_rotation : 
  -(minutes_moved / full_rotation_minutes * full_rotation_degrees * (π / 180)) = -π/3 := by
  sorry

end minute_hand_rotation_l911_91127


namespace min_a_value_l911_91150

/-- The minimum value of a that satisfies the given inequality for all positive x -/
theorem min_a_value (a : ℝ) : 
  (∀ x > 0, Real.log (2 * x) - (a * Real.exp x) / 2 ≤ Real.log a) → 
  a ≥ 2 / Real.exp 1 :=
sorry

end min_a_value_l911_91150


namespace katy_summer_reading_l911_91172

/-- The number of books Katy read in June -/
def june_books : ℕ := 8

/-- The number of books Katy read in July -/
def july_books : ℕ := 2 * june_books

/-- The number of books Katy read in August -/
def august_books : ℕ := july_books - 3

/-- The total number of books Katy read during the summer -/
def total_summer_books : ℕ := june_books + july_books + august_books

/-- Theorem stating that Katy read 37 books during the summer -/
theorem katy_summer_reading : total_summer_books = 37 := by
  sorry

end katy_summer_reading_l911_91172


namespace contrapositive_true_l911_91109

theorem contrapositive_true : 
  (∀ a : ℝ, a > 2 → a^2 > 4) ↔ (∀ a : ℝ, a ≤ 2 → a^2 ≤ 4) :=
by sorry

end contrapositive_true_l911_91109


namespace complex_equation_result_l911_91187

theorem complex_equation_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : a + 2 * i = i * (b - i)) : 
  a - b = -3 := by sorry

end complex_equation_result_l911_91187


namespace a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one_l911_91135

theorem a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one :
  ∃ (a : ℝ), (a^2 < 1 → a < 1) ∧ ¬(a < 1 → a^2 < 1) :=
by sorry

end a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one_l911_91135


namespace iris_shopping_cost_l911_91125

-- Define the quantities and prices
def num_jackets : ℕ := 3
def price_jacket : ℕ := 10
def num_shorts : ℕ := 2
def price_shorts : ℕ := 6
def num_pants : ℕ := 4
def price_pants : ℕ := 12

-- Define the total cost function
def total_cost : ℕ :=
  num_jackets * price_jacket +
  num_shorts * price_shorts +
  num_pants * price_pants

-- Theorem stating the total cost is $90
theorem iris_shopping_cost : total_cost = 90 := by
  sorry

end iris_shopping_cost_l911_91125


namespace four_term_expression_l911_91168

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ), (x^3 - 2)^2 + (x^2 + 2*x)^2 = a*x^6 + b*x^4 + c*x^2 + d ∧ 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) := by
  sorry

end four_term_expression_l911_91168


namespace sheena_sewing_hours_per_week_l911_91188

/-- Proves that Sheena sews 4 hours per week given the problem conditions -/
theorem sheena_sewing_hours_per_week 
  (time_per_dress : ℕ) 
  (num_dresses : ℕ) 
  (total_weeks : ℕ) 
  (h1 : time_per_dress = 12)
  (h2 : num_dresses = 5)
  (h3 : total_weeks = 15) :
  (time_per_dress * num_dresses) / total_weeks = 4 :=
by sorry

end sheena_sewing_hours_per_week_l911_91188


namespace probability_one_head_in_three_tosses_l911_91100

theorem probability_one_head_in_three_tosses :
  let n : ℕ := 3  -- number of tosses
  let k : ℕ := 1  -- number of heads we want
  let p : ℚ := 1/2  -- probability of heads on a single toss
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 := by
sorry

end probability_one_head_in_three_tosses_l911_91100


namespace symmetric_circle_correct_l911_91153

/-- The equation of circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- The equation of the symmetry line -/
def symmetry_line (x y : ℝ) : Prop := y = -x - 4

/-- The equation of the symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

/-- Theorem stating that the given symmetric circle is correct -/
theorem symmetric_circle_correct :
  ∀ (x y : ℝ), (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ symmetry_line ((x + x₀)/2) ((y + y₀)/2)) →
  symmetric_circle x y :=
sorry

end symmetric_circle_correct_l911_91153


namespace triangle_problem_l911_91181

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that if 2b*sin(B) = (2a+c)*sin(A) + (2c+a)*sin(C), b = √3, and A = π/4,
then B = 2π/3 and the area of the triangle is (3 - √3)/4.
-/
theorem triangle_problem (a b c A B C : ℝ) : 
  2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C →
  b = Real.sqrt 3 →
  A = π / 4 →
  B = 2 * π / 3 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = (3 - Real.sqrt 3) / 4 := by
  sorry


end triangle_problem_l911_91181


namespace overlapping_sticks_length_l911_91116

/-- The total length of overlapping wooden sticks -/
def total_length (n : ℕ) (stick_length overlap : ℝ) : ℝ :=
  stick_length + (n - 1) * (stick_length - overlap)

/-- Theorem: The total length of 30 wooden sticks, each 25 cm long, 
    when overlapped by 6 cm, is equal to 576 cm -/
theorem overlapping_sticks_length :
  total_length 30 25 6 = 576 := by
  sorry

end overlapping_sticks_length_l911_91116


namespace english_only_enrollment_l911_91115

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 60)
  (h2 : both = 18)
  (h3 : german = 36)
  (h4 : total ≥ german)
  (h5 : german ≥ both) :
  total - (german - both) - both = 24 :=
by sorry

end english_only_enrollment_l911_91115


namespace product_of_ten_proper_fractions_equals_one_tenth_l911_91184

theorem product_of_ten_proper_fractions_equals_one_tenth :
  ∃ (a b c d e f g h i j : ℚ),
    (0 < a ∧ a < 1) ∧
    (0 < b ∧ b < 1) ∧
    (0 < c ∧ c < 1) ∧
    (0 < d ∧ d < 1) ∧
    (0 < e ∧ e < 1) ∧
    (0 < f ∧ f < 1) ∧
    (0 < g ∧ g < 1) ∧
    (0 < h ∧ h < 1) ∧
    (0 < i ∧ i < 1) ∧
    (0 < j ∧ j < 1) ∧
    a * b * c * d * e * f * g * h * i * j = 1/10 :=
by sorry

end product_of_ten_proper_fractions_equals_one_tenth_l911_91184


namespace cost_price_calculation_l911_91137

/-- Proves that if an article is sold at 800 with a profit of 25%, then its cost price is 640. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 800)
  (h2 : profit_percentage = 25) :
  let cost_price := selling_price / (1 + profit_percentage / 100)
  cost_price = 640 := by sorry

end cost_price_calculation_l911_91137


namespace number_relationship_l911_91171

theorem number_relationship (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b :=
by sorry

end number_relationship_l911_91171


namespace no_perfect_squares_in_sequence_l911_91175

/-- Definition of the sequence x_n -/
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

/-- Theorem stating that no term in the sequence is a perfect square -/
theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℤ, x n = m * m :=
  sorry

end no_perfect_squares_in_sequence_l911_91175


namespace complex_exponential_sum_l911_91154

theorem complex_exponential_sum (α β : ℝ) : 
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/4 : ℂ) + (3/7 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/4 : ℂ) - (3/7 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l911_91154


namespace largest_prime_factor_of_6241_l911_91169

theorem largest_prime_factor_of_6241 : ∃ (p : ℕ), p.Prime ∧ p ∣ 6241 ∧ ∀ (q : ℕ), q.Prime → q ∣ 6241 → q ≤ p :=
by sorry

end largest_prime_factor_of_6241_l911_91169


namespace vikas_questions_l911_91159

theorem vikas_questions (total : ℕ) (r v a : ℕ) : 
  total = 24 →
  r + v + a = total →
  7 * v = 3 * r →
  3 * a = 2 * v →
  v = 6 := by
sorry

end vikas_questions_l911_91159


namespace tourist_contact_probability_l911_91166

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability 
  (group1_size : ℕ) 
  (group2_size : ℕ) 
  (p : ℝ) 
  (h1 : group1_size = 6) 
  (h2 : group2_size = 7) 
  (h3 : 0 ≤ p ∧ p ≤ 1) : 
  contact_probability p = 1 - (1 - p) ^ (group1_size * group2_size) :=
by sorry

end tourist_contact_probability_l911_91166


namespace photo_count_proof_l911_91147

def final_photo_count (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_photos : ℕ) (friend_photos : ℕ) (deleted_after_edit : ℕ) : ℕ :=
  initial_photos - deleted_bad_shots + cat_photos + friend_photos - deleted_after_edit

theorem photo_count_proof (x : ℕ) : 
  final_photo_count 63 7 15 x 3 = 68 + x := by
  sorry

end photo_count_proof_l911_91147


namespace min_abs_z_complex_l911_91139

theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 3*I) + Complex.abs (z - 4) = 5) :
  ∃ (min_abs : ℝ), min_abs = 12/5 ∧ ∀ w : ℂ, Complex.abs (w - 3*I) + Complex.abs (w - 4) = 5 → Complex.abs w ≥ min_abs :=
sorry

end min_abs_z_complex_l911_91139


namespace metal_detector_time_busier_days_is_30_l911_91108

/-- Represents the time Mark spends on courthouse activities in a week -/
structure CourthouseTime where
  totalWeeklyTime : ℕ
  parkingTime : ℕ
  walkingTime : ℕ
  workDays : ℕ
  lessCrowdedDays : ℕ
  metalDetectorTimeLessCrowded : ℕ

/-- Calculates the time spent on metal detector on busier days -/
def metalDetectorTimeBusierDays (ct : CourthouseTime) : ℕ :=
  let totalParkingWalkingTime := ct.workDays * (ct.parkingTime + ct.walkingTime)
  let totalMetalDetectorTime := ct.totalWeeklyTime - totalParkingWalkingTime
  let metalDetectorTimeLessCrowdedTotal := ct.lessCrowdedDays * ct.metalDetectorTimeLessCrowded
  let metalDetectorTimeBusierTotal := totalMetalDetectorTime - metalDetectorTimeLessCrowdedTotal
  metalDetectorTimeBusierTotal / (ct.workDays - ct.lessCrowdedDays)

theorem metal_detector_time_busier_days_is_30 (ct : CourthouseTime) :
  ct.totalWeeklyTime = 130 ∧
  ct.parkingTime = 5 ∧
  ct.walkingTime = 3 ∧
  ct.workDays = 5 ∧
  ct.lessCrowdedDays = 3 ∧
  ct.metalDetectorTimeLessCrowded = 10 →
  metalDetectorTimeBusierDays ct = 30 := by
  sorry


end metal_detector_time_busier_days_is_30_l911_91108


namespace only_log29_undetermined_l911_91176

-- Define the given logarithms
def log7 : ℝ := 0.8451
def log10 : ℝ := 1

-- Define a function to represent whether a logarithm can be determined
def can_determine (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), f log7 log10 = Real.log x

-- State the theorem
theorem only_log29_undetermined :
  ¬(can_determine 29) ∧ 
  can_determine (5/9) ∧ 
  can_determine 35 ∧ 
  can_determine 700 ∧ 
  can_determine 0.6 := by
  sorry


end only_log29_undetermined_l911_91176


namespace art_show_earnings_l911_91104

def extra_large_price : ℕ := 150
def large_price : ℕ := 100
def medium_price : ℕ := 80
def small_price : ℕ := 60

def extra_large_sold : ℕ := 3
def large_sold : ℕ := 5
def medium_sold : ℕ := 8
def small_sold : ℕ := 10

def large_discount : ℚ := 0.1
def sales_tax : ℚ := 0.05

def total_earnings : ℚ := 2247

theorem art_show_earnings :
  let extra_large_total := extra_large_price * extra_large_sold
  let large_total := large_price * large_sold * (1 - large_discount)
  let medium_total := medium_price * medium_sold
  let small_total := small_price * small_sold
  let subtotal := extra_large_total + large_total + medium_total + small_total
  let tax := subtotal * sales_tax
  (subtotal + tax : ℚ) = total_earnings := by
sorry

end art_show_earnings_l911_91104


namespace complex_magnitude_l911_91114

theorem complex_magnitude (z : ℂ) : z = (2 - Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l911_91114


namespace third_derivative_x5_minus_7x3_plus_2_l911_91119

/-- The third derivative of x^5 - 7x^3 + 2 is 60x^2 - 42 -/
theorem third_derivative_x5_minus_7x3_plus_2 (x : ℝ) :
  (deriv^[3] (fun x => x^5 - 7*x^3 + 2)) x = 60*x^2 - 42 := by
  sorry

end third_derivative_x5_minus_7x3_plus_2_l911_91119


namespace function_properties_l911_91173

/-- The function f(x) = x^2 - 6x + 10 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

/-- The interval [2, 5) -/
def I : Set ℝ := Set.Icc 2 5

theorem function_properties :
  (∃ (m : ℝ), m = 1 ∧ ∀ x ∈ I, f x ≥ m) ∧
  (¬∃ (M : ℝ), ∀ x ∈ I, f x ≤ M) :=
sorry

end function_properties_l911_91173
