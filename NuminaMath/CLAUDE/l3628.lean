import Mathlib

namespace circle_circumference_ratio_l3628_362855

/-- The ratio of the new circumference to the original circumference when the radius is increased by 2 units -/
theorem circle_circumference_ratio (r : ℝ) (h : r > 0) :
  (2 * Real.pi * (r + 2)) / (2 * Real.pi * r) = 1 + 2 / r := by
  sorry

#check circle_circumference_ratio

end circle_circumference_ratio_l3628_362855


namespace sum_first_eight_primes_mod_tenth_prime_l3628_362818

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]
def tenth_prime : Nat := 29

theorem sum_first_eight_primes_mod_tenth_prime :
  (first_eight_primes.sum) % tenth_prime = 19 := by sorry

end sum_first_eight_primes_mod_tenth_prime_l3628_362818


namespace triangle_side_indeterminate_l3628_362860

/-- Given a triangle ABC with AB = 3 and AC = 2, the length of BC cannot be uniquely determined. -/
theorem triangle_side_indeterminate (A B C : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A B = 3) → (d A C = 2) → 
  ¬∃! x : ℝ, d B C = x :=
by sorry

end triangle_side_indeterminate_l3628_362860


namespace jack_jill_equal_payment_l3628_362853

/-- Represents the pizza order and consumption details --/
structure PizzaOrder where
  totalSlices : ℕ
  baseCost : ℚ
  pepperoniCost : ℚ
  jackPepperoniSlices : ℕ
  jackCheeseSlices : ℕ

/-- Calculates the total cost of the pizza --/
def totalCost (order : PizzaOrder) : ℚ :=
  order.baseCost + order.pepperoniCost

/-- Calculates the cost per slice --/
def costPerSlice (order : PizzaOrder) : ℚ :=
  totalCost order / order.totalSlices

/-- Calculates Jack's payment --/
def jackPayment (order : PizzaOrder) : ℚ :=
  costPerSlice order * (order.jackPepperoniSlices + order.jackCheeseSlices)

/-- Calculates Jill's payment --/
def jillPayment (order : PizzaOrder) : ℚ :=
  costPerSlice order * (order.totalSlices - order.jackPepperoniSlices - order.jackCheeseSlices)

/-- Theorem: Jack and Jill pay the same amount for their share of the pizza --/
theorem jack_jill_equal_payment (order : PizzaOrder)
  (h1 : order.totalSlices = 12)
  (h2 : order.baseCost = 12)
  (h3 : order.pepperoniCost = 3)
  (h4 : order.jackPepperoniSlices = 4)
  (h5 : order.jackCheeseSlices = 2) :
  jackPayment order = jillPayment order := by
  sorry

end jack_jill_equal_payment_l3628_362853


namespace smallest_number_proof_l3628_362874

/-- The smallest natural number that is divisible by 55 and has exactly 117 distinct divisors -/
def smallest_number : ℕ := 12390400

/-- Count the number of distinct divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_proof :
  smallest_number % 55 = 0 ∧
  count_divisors smallest_number = 117 ∧
  ∀ m : ℕ, m < smallest_number → (m % 55 = 0 ∧ count_divisors m = 117) → False :=
by sorry

end smallest_number_proof_l3628_362874


namespace sum_of_fractions_bound_l3628_362846

theorem sum_of_fractions_bound (x y z : ℝ) (h : |x*y*z| = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 :=
sorry

end sum_of_fractions_bound_l3628_362846


namespace ordering_abc_l3628_362880

theorem ordering_abc : 
  let a : ℝ := 1/11
  let b : ℝ := Real.sqrt (1/10)
  let c : ℝ := Real.log (11/10)
  b > c ∧ c > a := by sorry

end ordering_abc_l3628_362880


namespace hyperbola_asymptote_l3628_362810

/-- The equation of the asymptote of a hyperbola -/
def asymptote_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The focal length of a hyperbola -/
def focal_length (c : ℝ) : ℝ := 2 * c

theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / 16 - y^2 / b^2 = 1}
  let f : ℝ := focal_length 5
  asymptote_equation 4 3 = {(x, y) | y = (3 / 4) * x ∨ y = -(3 / 4) * x} :=
by sorry

end hyperbola_asymptote_l3628_362810


namespace intersection_x_value_l3628_362800

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m1 b1 a2 b2 c2 : ℚ) : ℚ :=
  (c2 - b2 + b1) / (m1 + a2)

/-- Theorem: The x-coordinate of the intersection point of y = 4x - 29 and 3x + y = 105 is 134/7 -/
theorem intersection_x_value :
  intersection_x 4 (-29) 3 1 105 = 134 / 7 := by
  sorry

end intersection_x_value_l3628_362800


namespace sum_place_values_equals_350077055735_l3628_362886

def numeral : ℕ := 95378637153370261

def place_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def sum_place_values : ℕ :=
  -- Two 3's
  place_value 3 11 + place_value 3 1 +
  -- Three 7's
  place_value 7 10 + place_value 7 6 + place_value 7 2 +
  -- Four 5's
  place_value 5 13 + place_value 5 4 + place_value 5 3 + place_value 5 0

theorem sum_place_values_equals_350077055735 :
  sum_place_values = 350077055735 := by sorry

end sum_place_values_equals_350077055735_l3628_362886


namespace cos_theta_plus_5pi_6_l3628_362856

theorem cos_theta_plus_5pi_6 (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (θ / 2 + π / 6) = 2 / 3) : 
  Real.cos (θ + 5 * π / 6) = -4 * Real.sqrt 5 / 9 := by
  sorry

end cos_theta_plus_5pi_6_l3628_362856


namespace first_class_rate_total_l3628_362811

-- Define the pass rate
def pass_rate : ℝ := 0.95

-- Define the rate of first-class products among qualified products
def first_class_rate_qualified : ℝ := 0.20

-- Theorem statement
theorem first_class_rate_total (pass_rate : ℝ) (first_class_rate_qualified : ℝ) :
  pass_rate * first_class_rate_qualified = 0.19 := by
  sorry

end first_class_rate_total_l3628_362811


namespace tshirt_cost_calculation_l3628_362841

def sweatshirt_cost : ℕ := 15
def num_sweatshirts : ℕ := 3
def num_tshirts : ℕ := 2
def total_spent : ℕ := 65

theorem tshirt_cost_calculation :
  ∃ (tshirt_cost : ℕ), 
    num_sweatshirts * sweatshirt_cost + num_tshirts * tshirt_cost = total_spent ∧
    tshirt_cost = 10 := by
  sorry

end tshirt_cost_calculation_l3628_362841


namespace sum_of_evaluations_is_32_l3628_362875

/-- The expression to be evaluated -/
def expression : List ℕ := [1, 2, 3, 4, 5, 6]

/-- A sign assignment is a list of booleans, where true represents + and false represents - -/
def SignAssignment := List Bool

/-- Evaluate the expression given a sign assignment -/
def evaluate (signs : SignAssignment) : ℤ :=
  sorry

/-- Generate all possible sign assignments -/
def allSignAssignments : List SignAssignment :=
  sorry

/-- Calculate the sum of all evaluations -/
def sumOfEvaluations : ℤ :=
  sorry

/-- The main theorem: The sum of all evaluations is 32 -/
theorem sum_of_evaluations_is_32 : sumOfEvaluations = 32 := by
  sorry

end sum_of_evaluations_is_32_l3628_362875


namespace prob_two_out_of_three_germinate_l3628_362822

/-- The probability of exactly k successes in n independent Bernoulli trials 
    with probability p of success for each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly 2 successes out of 3 trials 
    with probability 4/5 of success for each trial -/
theorem prob_two_out_of_three_germinate : 
  binomial_probability 3 2 (4/5) = 48/125 := by
  sorry

end prob_two_out_of_three_germinate_l3628_362822


namespace candy_remaining_l3628_362869

theorem candy_remaining (initial : ℝ) (talitha_took : ℝ) (solomon_took : ℝ) (maya_took : ℝ)
  (h1 : initial = 1012.5)
  (h2 : talitha_took = 283.7)
  (h3 : solomon_took = 398.2)
  (h4 : maya_took = 197.6) :
  initial - (talitha_took + solomon_took + maya_took) = 133 := by
  sorry

end candy_remaining_l3628_362869


namespace smallest_positive_term_l3628_362826

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem smallest_positive_term (d : ℚ) :
  let a := arithmetic_sequence (-12) d
  let S := arithmetic_sum (-12) d
  S 13 = 0 →
  (∀ k < 8, a k ≤ 0) ∧ a 8 > 0 :=
by sorry

end smallest_positive_term_l3628_362826


namespace complex_product_real_l3628_362836

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := x - 2 * I
  (z₁ * z₂).im = 0 → x = 4 := by
sorry

end complex_product_real_l3628_362836


namespace fixed_point_of_linear_function_l3628_362871

theorem fixed_point_of_linear_function (k : ℝ) : 
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end fixed_point_of_linear_function_l3628_362871


namespace chess_tournament_participants_l3628_362851

theorem chess_tournament_participants (n : ℕ) : 
  n > 3 → 
  (n * (n - 1)) / 2 = 26 → 
  n = 14 := by
sorry

end chess_tournament_participants_l3628_362851


namespace original_orange_price_l3628_362840

theorem original_orange_price 
  (price_increase : ℝ) 
  (original_mango_price : ℝ) 
  (new_total_cost : ℝ) 
  (h1 : price_increase = 0.15)
  (h2 : original_mango_price = 50)
  (h3 : new_total_cost = 1035) :
  ∃ (original_orange_price : ℝ),
    original_orange_price = 40 ∧
    new_total_cost = 10 * (original_orange_price * (1 + price_increase)) + 
                     10 * (original_mango_price * (1 + price_increase)) :=
by sorry

end original_orange_price_l3628_362840


namespace max_value_of_a_l3628_362892

theorem max_value_of_a : 
  (∀ x : ℝ, (x + a)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) → 
  (∀ b : ℝ, (∀ x : ℝ, (x + b)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) → b ≤ 2) ∧
  (∀ x : ℝ, (x + 2)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) :=
by sorry

end max_value_of_a_l3628_362892


namespace problem_statement_l3628_362867

theorem problem_statement (x y : ℝ) (h : (x + 2*y)^3 + x^3 + 2*x + 2*y = 0) : 
  x + y - 1 = -1 := by sorry

end problem_statement_l3628_362867


namespace sqrt_equation_solutions_l3628_362881

theorem sqrt_equation_solutions : 
  {x : ℝ | Real.sqrt (4 * x - 3) + 18 / Real.sqrt (4 * x - 3) = 9} = {3, 9.75} := by
sorry

end sqrt_equation_solutions_l3628_362881


namespace formula_correctness_l3628_362868

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 2

theorem formula_correctness : 
  (f 2 = 10) ∧ 
  (f 3 = 21) ∧ 
  (f 4 = 38) ∧ 
  (f 5 = 61) ∧ 
  (f 6 = 90) := by
  sorry

end formula_correctness_l3628_362868


namespace soft_drink_storage_l3628_362819

theorem soft_drink_storage (initial_small : ℕ) (initial_big : ℕ) 
  (percent_big_sold : ℚ) (total_remaining : ℕ) :
  initial_small = 6000 →
  initial_big = 15000 →
  percent_big_sold = 14 / 100 →
  total_remaining = 18180 →
  ∃ (percent_small_sold : ℚ),
    percent_small_sold = 12 / 100 ∧
    (initial_small - initial_small * percent_small_sold) +
    (initial_big - initial_big * percent_big_sold) = total_remaining :=
by sorry

end soft_drink_storage_l3628_362819


namespace unique_pricing_l3628_362858

/-- Represents the price of a sewage treatment equipment model in thousand dollars. -/
structure ModelPrice where
  price : ℝ

/-- Represents the pricing of two sewage treatment equipment models A and B. -/
structure EquipmentPricing where
  modelA : ModelPrice
  modelB : ModelPrice

/-- Checks if the given equipment pricing satisfies the problem conditions. -/
def satisfiesConditions (pricing : EquipmentPricing) : Prop :=
  pricing.modelA.price = pricing.modelB.price + 5 ∧
  2 * pricing.modelA.price + 3 * pricing.modelB.price = 45

/-- Theorem stating that the only pricing satisfying the conditions is A at 12 and B at 7. -/
theorem unique_pricing :
  ∀ (pricing : EquipmentPricing),
    satisfiesConditions pricing →
    pricing.modelA.price = 12 ∧ pricing.modelB.price = 7 := by
  sorry

end unique_pricing_l3628_362858


namespace basketball_season_games_l3628_362804

/-- The total number of games played by a basketball team in a season -/
def total_games : ℕ := 93

/-- The number of games in the first segment -/
def first_segment : ℕ := 40

/-- The number of games in the second segment -/
def second_segment : ℕ := 30

/-- The win rate for the first segment -/
def first_rate : ℚ := 1/2

/-- The win rate for the second segment -/
def second_rate : ℚ := 3/5

/-- The win rate for the remaining games -/
def remaining_rate : ℚ := 17/20

/-- The overall win rate for the season -/
def overall_rate : ℚ := 31/50

theorem basketball_season_games :
  let remaining_games := total_games - first_segment - second_segment
  let total_wins := (first_rate * first_segment) + (second_rate * second_segment) + (remaining_rate * remaining_games)
  total_wins = overall_rate * total_games := by sorry

#eval total_games

end basketball_season_games_l3628_362804


namespace intersecting_segments_l3628_362812

/-- Given two intersecting line segments PQ and RS, prove that x + y = 145 -/
theorem intersecting_segments (x y : ℝ) : 
  (60 + (y + 5) = 180) →  -- Linear pair on PQ
  (4 * x = y + 5) →       -- Vertically opposite angles
  x + y = 145 := by sorry

end intersecting_segments_l3628_362812


namespace quadratic_inequality_properties_l3628_362838

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < 1 ∨ x > 3}) :
  c < 0 ∧
  a + 2*b + 4*c < 0 ∧
  {x : ℝ | c*x + a < 0} = {x : ℝ | x > -1/3} :=
by sorry

end quadratic_inequality_properties_l3628_362838


namespace smallest_value_of_x_plus_yz_l3628_362864

theorem smallest_value_of_x_plus_yz (x y z : ℕ+) (h : x * y + z = 160) :
  ∃ (a b c : ℕ+), a * b + c = 160 ∧ a + b * c = 64 ∧ ∀ (p q r : ℕ+), p * q + r = 160 → p + q * r ≥ 64 := by
  sorry

end smallest_value_of_x_plus_yz_l3628_362864


namespace star_six_three_l3628_362801

-- Define the * operation
def star (a b : ℤ) : ℤ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_six_three : star 6 3 = 3 := by sorry

end star_six_three_l3628_362801


namespace vector_linear_combination_l3628_362827

/-- Given two planar vectors a and b, prove that their linear combination results in the specified vector. -/
theorem vector_linear_combination (a b : ℝ × ℝ) :
  a = (1, 1) → b = (1, -1) → (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by sorry

end vector_linear_combination_l3628_362827


namespace automobile_distance_l3628_362896

/-- Proves the total distance traveled by an automobile given specific conditions -/
theorem automobile_distance (a r : ℝ) : 
  let first_half_distance : ℝ := a / 4
  let first_half_time : ℝ := 2 * r
  let first_half_speed : ℝ := first_half_distance / first_half_time
  let second_half_speed : ℝ := 2 * first_half_speed
  let second_half_time : ℝ := 2 * 60 -- 2 minutes in seconds
  let second_half_distance : ℝ := second_half_speed * second_half_time
  let total_distance_feet : ℝ := first_half_distance + second_half_distance
  let total_distance_yards : ℝ := total_distance_feet / 3
  total_distance_yards = 121 * a / 12 :=
by sorry

end automobile_distance_l3628_362896


namespace apple_distribution_l3628_362824

theorem apple_distribution (total_apples : ℕ) (new_people : ℕ) (apple_reduction : ℕ) 
  (h1 : total_apples = 2750)
  (h2 : new_people = 60)
  (h3 : apple_reduction = 12) :
  ∃ (original_people : ℕ),
    (total_apples / original_people : ℚ) - 
    (total_apples / (original_people + new_people) : ℚ) = apple_reduction ∧
    total_apples / original_people = 30 := by
  sorry

end apple_distribution_l3628_362824


namespace quadratic_unique_solution_l3628_362839

theorem quadratic_unique_solution (b d : ℤ) : 
  (∃! x : ℝ, b * x^2 + 24 * x + d = 0) →
  b + d = 41 →
  b < d →
  b = 9 ∧ d = 32 := by
sorry

end quadratic_unique_solution_l3628_362839


namespace max_sum_of_digits_24hour_watch_l3628_362883

-- Define the valid range for hours and minutes
def valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23
def valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem statement
theorem max_sum_of_digits_24hour_watch : 
  ∀ h m, valid_hour h → valid_minute m →
  sum_of_digits h + sum_of_digits m ≤ 24 :=
by sorry

end max_sum_of_digits_24hour_watch_l3628_362883


namespace david_investment_time_l3628_362816

/-- Simple interest calculation -/
def simpleInterest (principal time rate : ℝ) : ℝ :=
  principal * (1 + time * rate)

theorem david_investment_time :
  ∀ (rate : ℝ),
  rate > 0 →
  simpleInterest 710 3 rate = 815 →
  simpleInterest 710 4 rate = 850 :=
by
  sorry

end david_investment_time_l3628_362816


namespace trivia_team_group_size_l3628_362899

theorem trivia_team_group_size 
  (total_students : ℕ) 
  (unpicked_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 35) 
  (h2 : unpicked_students = 11) 
  (h3 : num_groups = 4) :
  (total_students - unpicked_students) / num_groups = 6 :=
by sorry

end trivia_team_group_size_l3628_362899


namespace smallest_square_enclosing_circle_l3628_362837

theorem smallest_square_enclosing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by sorry

end smallest_square_enclosing_circle_l3628_362837


namespace intersecting_lines_k_value_l3628_362803

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersecting_lines_k_value (x y k : ℚ) : 
  (y = 6 * x + 4) ∧ 
  (y = -3 * x - 30) ∧ 
  (y = 4 * x + k) → 
  k = -32/9 := by sorry

end intersecting_lines_k_value_l3628_362803


namespace assignment_schemes_with_girl_count_l3628_362890

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The number of tasks to be assigned -/
def num_tasks : ℕ := 3

/-- The function to calculate the number of assignment schemes -/
def assignment_schemes (b g s t : ℕ) : ℕ :=
  Nat.descFactorial (b + g) s - Nat.descFactorial b s

/-- Theorem stating that the number of assignment schemes with at least one girl is 186 -/
theorem assignment_schemes_with_girl_count :
  assignment_schemes num_boys num_girls num_selected num_tasks = 186 := by
  sorry

end assignment_schemes_with_girl_count_l3628_362890


namespace boats_geometric_sum_l3628_362877

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boats_geometric_sum :
  geometric_sum 5 3 5 = 605 := by
  sorry

end boats_geometric_sum_l3628_362877


namespace afternoon_more_than_morning_l3628_362843

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The difference in emails between afternoon and morning -/
def email_difference : ℕ := afternoon_emails - morning_emails

theorem afternoon_more_than_morning : email_difference = 2 := by
  sorry

end afternoon_more_than_morning_l3628_362843


namespace olivia_hourly_rate_l3628_362859

/-- Olivia's hourly rate given her work hours and total earnings --/
theorem olivia_hourly_rate (monday_hours wednesday_hours friday_hours total_earnings : ℕ) :
  monday_hours = 4 →
  wednesday_hours = 3 →
  friday_hours = 6 →
  total_earnings = 117 →
  (total_earnings : ℚ) / (monday_hours + wednesday_hours + friday_hours : ℚ) = 9 := by
  sorry

end olivia_hourly_rate_l3628_362859


namespace oil_price_reduction_l3628_362873

/-- Given a 25% reduction in oil price, prove the reduced price per kg is 30 Rs. --/
theorem oil_price_reduction (original_price : ℝ) (h1 : original_price > 0) : 
  let reduced_price := 0.75 * original_price
  (600 / reduced_price) = (600 / original_price) + 5 →
  reduced_price = 30 := by
  sorry

end oil_price_reduction_l3628_362873


namespace quadrilateral_area_implies_k_l3628_362872

/-- A quadrilateral with vertices A(0,3), B(0,k), C(5,10), and D(5,0) -/
structure Quadrilateral (k : ℝ) :=
  (A : ℝ × ℝ := (0, 3))
  (B : ℝ × ℝ := (0, k))
  (C : ℝ × ℝ := (5, 10))
  (D : ℝ × ℝ := (5, 0))

/-- The area of a quadrilateral -/
def area (q : Quadrilateral k) : ℝ :=
  sorry

/-- Theorem stating that if k > 3 and the area of the quadrilateral is 50, then k = 13 -/
theorem quadrilateral_area_implies_k (k : ℝ) (q : Quadrilateral k)
    (h1 : k > 3)
    (h2 : area q = 50) :
  k = 13 := by
  sorry

end quadrilateral_area_implies_k_l3628_362872


namespace larger_number_proof_l3628_362887

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 4186) →
  (a = 23 * 13) →
  (b = 23 * 14) →
  max a b = 322 := by
sorry

end larger_number_proof_l3628_362887


namespace condition_sufficient_not_necessary_l3628_362866

theorem condition_sufficient_not_necessary :
  (∀ k : ℤ, Real.sin (2 * k * Real.pi + Real.pi / 4) = Real.sqrt 2 / 2) ∧
  (∃ x : ℝ, Real.sin x = Real.sqrt 2 / 2 ∧ ∀ k : ℤ, x ≠ 2 * k * Real.pi + Real.pi / 4) :=
by sorry

end condition_sufficient_not_necessary_l3628_362866


namespace income_change_approx_23_86_percent_l3628_362884

def job_a_initial_weekly : ℚ := 60
def job_a_final_weekly : ℚ := 78
def job_a_quarterly_bonus : ℚ := 50

def job_b_initial_weekly : ℚ := 100
def job_b_final_weekly : ℚ := 115
def job_b_initial_biannual_bonus : ℚ := 200
def job_b_bonus_increase_rate : ℚ := 0.1

def weekly_expenses : ℚ := 30
def weeks_per_quarter : ℕ := 13

def initial_quarterly_income : ℚ :=
  job_a_initial_weekly * weeks_per_quarter + job_a_quarterly_bonus +
  job_b_initial_weekly * weeks_per_quarter + job_b_initial_biannual_bonus / 2

def final_quarterly_income : ℚ :=
  job_a_final_weekly * weeks_per_quarter + job_a_quarterly_bonus +
  job_b_final_weekly * weeks_per_quarter + 
  (job_b_initial_biannual_bonus * (1 + job_b_bonus_increase_rate)) / 2

def quarterly_expenses : ℚ := weekly_expenses * weeks_per_quarter

def initial_effective_income : ℚ := initial_quarterly_income - quarterly_expenses
def final_effective_income : ℚ := final_quarterly_income - quarterly_expenses

def income_change_percentage : ℚ :=
  (final_effective_income - initial_effective_income) / initial_effective_income * 100

theorem income_change_approx_23_86_percent : 
  ∃ ε > 0, abs (income_change_percentage - 23.86) < ε :=
sorry

end income_change_approx_23_86_percent_l3628_362884


namespace card_area_problem_l3628_362805

theorem card_area_problem (length width : ℝ) 
  (h1 : length = 4 ∧ width = 6)
  (h2 : (length - 1) * width = 18 ∨ length * (width - 1) = 18) :
  (if (length - 1) * width = 18 
   then length * (width - 1) 
   else (length - 1) * width) = 20 := by
  sorry

end card_area_problem_l3628_362805


namespace remainder_problem_l3628_362878

theorem remainder_problem (k : ℕ+) (h : 60 % k.val ^ 2 = 6) : 100 % k.val = 1 := by
  sorry

end remainder_problem_l3628_362878


namespace sophie_rearrangement_time_l3628_362847

def name_length : ℕ := 6
def rearrangements_per_minute : ℕ := 18

theorem sophie_rearrangement_time :
  (name_length.factorial / rearrangements_per_minute : ℚ) / 60 = 2 / 3 := by
  sorry

end sophie_rearrangement_time_l3628_362847


namespace largest_number_l3628_362895

theorem largest_number (a b c d e : ℕ) :
  a = 30^20 ∧
  b = 10^30 ∧
  c = 30^10 + 20^20 ∧
  d = (30 + 10)^20 ∧
  e = (30 * 20)^10 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end largest_number_l3628_362895


namespace part_one_part_two_l3628_362882

-- Define the solution set M
def M (a : ℝ) := {x : ℝ | a * x^2 + 5 * x - 2 > 0}

-- Part 1
theorem part_one (a : ℝ) : 2 ∈ M a → a > -2 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  M a = {x : ℝ | 1/2 < x ∧ x < 2} →
  {x : ℝ | a * x^2 - 5 * x + a^2 - 1 > 0} = {x : ℝ | -3 < x ∧ x < 1/2} := by sorry

end part_one_part_two_l3628_362882


namespace smallest_n_for_sqrt_inequality_l3628_362870

theorem smallest_n_for_sqrt_inequality : 
  ∀ n : ℕ+, n < 101 → ¬(Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.05) ∧ 
  (Real.sqrt 101 - Real.sqrt 100 < 0.05) := by
  sorry

end smallest_n_for_sqrt_inequality_l3628_362870


namespace cubic_polynomial_properties_l3628_362832

/-- The cubic polynomial f(x) = x³ + px + q -/
noncomputable def f (p q x : ℝ) : ℝ := x^3 + p*x + q

theorem cubic_polynomial_properties (p q : ℝ) :
  (p ≥ 0 → ∀ x y : ℝ, x < y → f p q x < f p q y) ∧ 
  (p < 0 → ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f p q x = 0 ∧ f p q y = 0 ∧ f p q z = 0) ∧
  (p < 0 → ∃! x y : ℝ, x ≠ y ∧ (∀ z : ℝ, f p q x ≤ f p q z) ∧ (∀ z : ℝ, f p q y ≥ f p q z)) ∧
  (p < 0 → ∃ x y : ℝ, x ≠ y ∧ (∀ z : ℝ, f p q x ≤ f p q z) ∧ (∀ z : ℝ, f p q y ≥ f p q z) ∧ x = -y) :=
by sorry

end cubic_polynomial_properties_l3628_362832


namespace triangular_prism_volume_l3628_362889

/-- Given a rectangle ABCD with dimensions as specified, prove that the volume of the
    triangular prism formed by folding is 594. -/
theorem triangular_prism_volume (A B C D P : ℝ × ℝ) : 
  let AB := 13 * Real.sqrt 3
  let BC := 12 * Real.sqrt 3
  -- A, B, C, D form a rectangle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = AB^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = BC^2 ∧
  -- P is the intersection of diagonals
  (A.1 - C.1) * (B.2 - D.2) = (A.2 - C.2) * (B.1 - D.1) ∧
  P.1 = (A.1 + C.1) / 2 ∧ P.2 = (A.2 + C.2) / 2 →
  -- Volume of the triangular prism after folding
  (1/6) * AB * BC * Real.sqrt (AB^2 + BC^2 - (AB^2 * BC^2) / (AB^2 + BC^2)) = 594 := by
  sorry


end triangular_prism_volume_l3628_362889


namespace equation_with_added_constant_l3628_362865

theorem equation_with_added_constant (y : ℝ) (n : ℝ) :
  y^4 - 20*y + 1 = 22 ∧ n = 3 →
  y^4 - 20*y + (1 + n) = 22 + n :=
by sorry

end equation_with_added_constant_l3628_362865


namespace natasha_exercise_time_l3628_362828

/-- Proves that Natasha exercised for 30 minutes daily given the conditions of the problem -/
theorem natasha_exercise_time :
  ∀ (d : ℕ),
  let natasha_daily_time : ℕ := 30
  let natasha_total_time : ℕ := d * natasha_daily_time
  let esteban_daily_time : ℕ := 10
  let esteban_days : ℕ := 9
  let esteban_total_time : ℕ := esteban_daily_time * esteban_days
  let total_exercise_time : ℕ := 5 * 60
  natasha_total_time + esteban_total_time = total_exercise_time →
  natasha_daily_time = 30 :=
by
  sorry

#check natasha_exercise_time

end natasha_exercise_time_l3628_362828


namespace simplify_fraction_l3628_362834

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l3628_362834


namespace intersection_theorem_l3628_362898

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | x < 1}

theorem intersection_theorem : M ∩ (Set.univ \ N) = Set.Icc 1 2 := by
  sorry

end intersection_theorem_l3628_362898


namespace max_value_of_f_l3628_362857

-- Define the function f(x) = -3x^2 + 9
def f (x : ℝ) : ℝ := -3 * x^2 + 9

-- Theorem stating that the maximum value of f(x) is 9
theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
by
  sorry


end max_value_of_f_l3628_362857


namespace inequality_proof_l3628_362897

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end inequality_proof_l3628_362897


namespace circles_tangent_sum_l3628_362885

-- Define the circles and line
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 1
def circle_C2 (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the external tangency condition
def externally_tangent (a b : ℝ) : Prop := (a - 1)^2 + (b + 3)^2 > 4

-- Define the equal tangent length condition
def equal_tangent_length (a b : ℝ) : Prop := 
  ∃ m : ℝ, (4 + 2*a + 2*b)*m + 5 - a^2 - (1 + b)^2 = 0

-- State the theorem
theorem circles_tangent_sum (a b : ℝ) :
  externally_tangent a b →
  equal_tangent_length a b →
  a + b = -2 := by sorry

end circles_tangent_sum_l3628_362885


namespace sum_of_coefficients_l3628_362888

theorem sum_of_coefficients (P a b c d e f : ℕ) : 
  20112011 = a * P^5 + b * P^4 + c * P^3 + d * P^2 + e * P + f →
  a < P ∧ b < P ∧ c < P ∧ d < P ∧ e < P ∧ f < P →
  a + b + c + d + e + f = 36 := by
  sorry

end sum_of_coefficients_l3628_362888


namespace largest_alternating_geometric_sequence_l3628_362879

def is_valid_sequence (a b c d : ℕ) : Prop :=
  a > b ∧ b < c ∧ c > d ∧
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  ∃ (r : ℚ), b = a / r ∧ c = a / (r^2) ∧ d = a / (r^3)

theorem largest_alternating_geometric_sequence :
  ∀ (n : ℕ), n ≤ 9999 →
    is_valid_sequence (n / 1000 % 10) (n / 100 % 10) (n / 10 % 10) (n % 10) →
    n ≤ 9632 :=
sorry

end largest_alternating_geometric_sequence_l3628_362879


namespace arithmetic_simplification_l3628_362808

theorem arithmetic_simplification : (4 + 6 + 2) / 3 - 2 / 3 = 10 / 3 := by
  sorry

end arithmetic_simplification_l3628_362808


namespace congruence_solution_l3628_362823

theorem congruence_solution (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 → x % 6 = 1 % 6 := by
  sorry

end congruence_solution_l3628_362823


namespace max_m_value_l3628_362844

theorem max_m_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = x + 2 * y) :
  ∀ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x * y = x + 2 * y → x * y ≥ m - 2) → m ≤ 10 :=
sorry

end max_m_value_l3628_362844


namespace complement_of_A_l3628_362813

def A : Set ℝ := {y | ∃ x, y = 2^x}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = Set.Iic 0 := by sorry

end complement_of_A_l3628_362813


namespace sum_of_digits_An_l3628_362842

-- Define the product An
def An (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc i => acc * (10^(2^i) - 1)) 9

-- Define the sum of digits function
def sumOfDigits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sumOfDigits (m / 10)

-- State the theorem
theorem sum_of_digits_An (n : ℕ) :
  sumOfDigits (An n) = 9 * 2^n := by
  sorry

end sum_of_digits_An_l3628_362842


namespace min_value_of_f_l3628_362833

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2023

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 1996) :=
sorry

end min_value_of_f_l3628_362833


namespace student_number_problem_l3628_362802

theorem student_number_problem (x : ℝ) : 7 * x - 150 = 130 → x = 40 := by
  sorry

end student_number_problem_l3628_362802


namespace bridge_length_calculation_l3628_362848

/-- Calculates the length of a bridge given the parameters of an elephant train passing through it. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_cm : ℝ) (time_to_pass : ℝ) : 
  train_length = 15 →
  train_speed_cm = 275 →
  time_to_pass = 48 →
  (train_speed_cm / 100 * time_to_pass) - train_length = 117 := by
  sorry

#check bridge_length_calculation

end bridge_length_calculation_l3628_362848


namespace floor_sum_example_l3628_362861

theorem floor_sum_example : ⌊(-13.7 : ℝ)⌋ + ⌊(13.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l3628_362861


namespace inequality_range_l3628_362829

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := by sorry

end inequality_range_l3628_362829


namespace max_profit_l3628_362820

-- Define the linear relationship between price and quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 50) * (sales_quantity x)

-- Theorem statement
theorem max_profit :
  ∃ (x : ℝ), x = 70 ∧ profit x = 800 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end max_profit_l3628_362820


namespace golf_ball_goal_l3628_362825

theorem golf_ball_goal (goal : ℕ) (saturday : ℕ) (sunday : ℕ) : 
  goal = 48 → saturday = 16 → sunday = 18 → 
  goal - (saturday + sunday) = 14 :=
by sorry

end golf_ball_goal_l3628_362825


namespace min_value_sum_of_distances_min_value_achievable_l3628_362845

theorem min_value_sum_of_distances (x : ℝ) :
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2) ≥ Real.sqrt 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2) = Real.sqrt 5 :=
by sorry

end min_value_sum_of_distances_min_value_achievable_l3628_362845


namespace four_heads_before_three_tails_l3628_362862

/-- The probability of getting 4 consecutive heads before 3 consecutive tails
    when repeatedly flipping a fair coin -/
def q : ℚ := 31 / 63

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop := p (1 / 2)

/-- The event of getting 4 consecutive heads -/
def four_heads : ℕ → Prop := λ n => ∀ i, i ∈ Finset.range 4 → n + i = 1

/-- The event of getting 3 consecutive tails -/
def three_tails : ℕ → Prop := λ n => ∀ i, i ∈ Finset.range 3 → n + i = 0

/-- The probability of an event occurring before another event
    when repeatedly performing an experiment -/
def prob_before (p : ℚ) (event1 event2 : ℕ → Prop) : Prop :=
  ∃ n : ℕ, (∀ k < n, ¬event1 k ∧ ¬event2 k) ∧ event1 n ∧ (∀ k ≤ n, ¬event2 k)

theorem four_heads_before_three_tails :
  fair_coin (λ p => prob_before q four_heads three_tails) :=
sorry

end four_heads_before_three_tails_l3628_362862


namespace min_value_x_plus_2y_l3628_362876

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/b = 2 → x + 2*y ≤ a + 2*b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 2 ∧ x₀ + 2*y₀ = (3 + 2*Real.sqrt 2)/2 :=
sorry

end min_value_x_plus_2y_l3628_362876


namespace right_triangle_in_circle_l3628_362807

theorem right_triangle_in_circle (d : ℝ) (b : ℝ) (a : ℝ) : 
  d = 10 → b = 8 → a * a + b * b = d * d → a = 6 :=
by sorry

end right_triangle_in_circle_l3628_362807


namespace line_proof_l3628_362893

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y + 10 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line3 (x y : ℝ) : Prop := 3*x - 2*y + 4 = 0
def result_line (x y : ℝ) : Prop := 2*x + 3*y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    result_line x y ∧
    perpendicular (3/2) (-2/3) :=
by sorry

end line_proof_l3628_362893


namespace not_geometric_progression_l3628_362850

theorem not_geometric_progression : 
  ¬∃ (a r : ℝ) (p q k : ℤ), 
    p ≠ q ∧ q ≠ k ∧ p ≠ k ∧ 
    a * r^(p-1) = 10 ∧ 
    a * r^(q-1) = 11 ∧ 
    a * r^(k-1) = 12 := by
  sorry

end not_geometric_progression_l3628_362850


namespace lcm_of_ratio_and_hcf_l3628_362814

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 2 / 3 → 
  Nat.gcd a b = 6 → 
  Nat.lcm a b = 36 := by
sorry

end lcm_of_ratio_and_hcf_l3628_362814


namespace sum_of_u_and_v_l3628_362831

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = 17) 
  (eq2 : 5 * u + 3 * v = -1) : 
  u + v = -41 / 29 := by
sorry

end sum_of_u_and_v_l3628_362831


namespace sock_pairs_theorem_l3628_362852

theorem sock_pairs_theorem (n : ℕ) : 
  (2 * n * (2 * n - 1)) / 2 = 42 → n = 6 := by
  sorry

end sock_pairs_theorem_l3628_362852


namespace quadratic_factorization_l3628_362817

theorem quadratic_factorization :
  ∀ x : ℝ, x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
sorry

end quadratic_factorization_l3628_362817


namespace max_value_parabola_l3628_362849

/-- The maximum value of y = -3x^2 + 6, where x is a real number, is 6. -/
theorem max_value_parabola :
  ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), -3 * x^2 + 6 ≤ M :=
sorry

end max_value_parabola_l3628_362849


namespace average_increase_is_eight_l3628_362806

/-- Represents a cricketer's batting statistics -/
structure CricketerStats where
  initialInnings : Nat
  initialTotalRuns : Nat
  newInningScore : Nat
  newAverage : Nat

/-- Calculates the increase in average for a cricketer -/
def averageIncrease (stats : CricketerStats) : Nat :=
  stats.newAverage - (stats.initialTotalRuns / stats.initialInnings)

/-- Theorem: Given the specific conditions, the average increase is 8 runs -/
theorem average_increase_is_eight (stats : CricketerStats) 
  (h1 : stats.initialInnings = 9)
  (h2 : stats.newInningScore = 200)
  (h3 : stats.newAverage = 128)
  (h4 : stats.initialTotalRuns + stats.newInningScore = (stats.initialInnings + 1) * stats.newAverage) :
  averageIncrease stats = 8 := by
  sorry

#eval averageIncrease { initialInnings := 9, initialTotalRuns := 1080, newInningScore := 200, newAverage := 128 }

end average_increase_is_eight_l3628_362806


namespace no_solution_to_inequality_system_l3628_362830

theorem no_solution_to_inequality_system :
  ¬∃ (x y z t : ℝ), 
    (abs x > abs (y - z + t)) ∧
    (abs y > abs (x - z + t)) ∧
    (abs z > abs (x - y + t)) ∧
    (abs t > abs (x - y + z)) := by
  sorry

end no_solution_to_inequality_system_l3628_362830


namespace lollipop_distribution_theorem_l3628_362854

/-- Represents the lollipop distribution rules and class attendance --/
structure LollipopDistribution where
  mainTeacherRatio : ℕ  -- Students per lollipop for main teacher
  assistantRatio : ℕ    -- Students per lollipop for assistant
  assistantThreshold : ℕ -- Threshold for assistant to start giving lollipops
  initialStudents : ℕ   -- Initial number of students
  lateStudents : List ℕ  -- List of additional students joining later

/-- Calculates the total number of lollipops given away --/
def totalLollipops (d : LollipopDistribution) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, 21 lollipops will be given away --/
theorem lollipop_distribution_theorem :
  let d : LollipopDistribution := {
    mainTeacherRatio := 5,
    assistantRatio := 7,
    assistantThreshold := 30,
    initialStudents := 45,
    lateStudents := [10, 5, 5]
  }
  totalLollipops d = 21 := by
  sorry

end lollipop_distribution_theorem_l3628_362854


namespace master_bedroom_size_l3628_362809

theorem master_bedroom_size (total_area : ℝ) (common_area : ℝ) (guest_ratio : ℝ) :
  total_area = 2300 →
  common_area = 1000 →
  guest_ratio = 1/4 →
  ∃ (master_size : ℝ),
    master_size = 1040 ∧
    total_area = common_area + master_size + guest_ratio * master_size :=
by
  sorry

end master_bedroom_size_l3628_362809


namespace point_coordinates_l3628_362891

def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ x y : ℝ,
    third_quadrant x y →
    distance_to_x_axis y = 2 →
    distance_to_y_axis x = 5 →
    (x, y) = (-5, -2) :=
by
  sorry

end point_coordinates_l3628_362891


namespace time_differences_not_constant_l3628_362835

/-- Represents the relationship between height and time for the sliding car experiment -/
def slide_data : List (ℝ × ℝ) :=
  [(10, 4.23), (20, 3.00), (30, 2.45), (40, 2.13), (50, 1.89), (60, 1.71), (70, 1.59)]

/-- Calculates the time difference between two consecutive measurements -/
def time_diff (data : List (ℝ × ℝ)) (i : ℕ) : ℝ :=
  match data.get? i, data.get? (i+1) with
  | some (_, t1), some (_, t2) => t1 - t2
  | _, _ => 0

/-- Theorem stating that time differences are not constant -/
theorem time_differences_not_constant : ∃ i j, i ≠ j ∧ i < slide_data.length - 1 ∧ j < slide_data.length - 1 ∧ time_diff slide_data i ≠ time_diff slide_data j :=
sorry

end time_differences_not_constant_l3628_362835


namespace travel_options_count_l3628_362821

/-- The number of bus options from A to B -/
def bus_options : ℕ := 5

/-- The number of train options from A to B -/
def train_options : ℕ := 6

/-- The number of boat options from A to B -/
def boat_options : ℕ := 2

/-- The total number of travel options from A to B -/
def total_options : ℕ := bus_options + train_options + boat_options

theorem travel_options_count :
  total_options = 13 :=
by sorry

end travel_options_count_l3628_362821


namespace tiling_8x1_board_remainder_l3628_362815

/-- Represents a tiling of an 8x1 board -/
structure Tiling :=
  (num_1x1 : ℕ)
  (num_2x1 : ℕ)
  (h_sum : num_1x1 + 2 * num_2x1 = 8)

/-- Calculates the number of valid colorings for a given tiling -/
def validColorings (t : Tiling) : ℕ :=
  3^(t.num_1x1 + t.num_2x1) - 3 * 2^(t.num_1x1 + t.num_2x1) + 3

/-- The set of all possible tilings -/
def allTilings : Finset Tiling :=
  sorry

theorem tiling_8x1_board_remainder (M : ℕ) (h_M : M = (allTilings.sum validColorings)) :
  M % 1000 = 328 :=
sorry

end tiling_8x1_board_remainder_l3628_362815


namespace line_opposite_sides_range_l3628_362894

/-- The range of 'a' for a line x + y - a = 0 with (0, 0) and (1, 1) on opposite sides -/
theorem line_opposite_sides_range (a : ℝ) : 
  (∀ x y : ℝ, x + y - a = 0 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (0 < a ∧ a < 2) :=
sorry

end line_opposite_sides_range_l3628_362894


namespace parallelogram_count_formula_l3628_362863

/-- An equilateral triangle with side length n, tiled with n^2 smaller equilateral triangles -/
structure TiledTriangle (n : ℕ) where
  side_length : ℕ := n
  num_small_triangles : ℕ := n^2

/-- The number of parallelograms in a tiled equilateral triangle -/
def count_parallelograms (t : TiledTriangle n) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- Theorem stating that the number of parallelograms in a tiled equilateral triangle
    is equal to 3 * (n+2 choose 4) -/
theorem parallelogram_count_formula (n : ℕ) (t : TiledTriangle n) :
  count_parallelograms t = 3 * Nat.choose (n + 2) 4 := by
  sorry

end parallelogram_count_formula_l3628_362863
