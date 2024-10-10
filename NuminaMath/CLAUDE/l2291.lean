import Mathlib

namespace function_properties_l2291_229132

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem function_properties (a b : ℝ) :
  (f a b 1 = 3) →
  ((3 * a + 2 * b) = 0) →
  (a = -6 ∧ b = 9) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≤ 15) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≥ -12) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = 15) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = -12) :=
by
  sorry

end function_properties_l2291_229132


namespace optimal_purchase_plan_l2291_229130

/-- Represents the purchase and selling prices of keychains --/
structure KeychainPrices where
  purchase_a : ℕ
  purchase_b : ℕ
  selling_a : ℕ
  selling_b : ℕ

/-- Represents the purchase plan for keychains --/
structure PurchasePlan where
  quantity_a : ℕ
  quantity_b : ℕ

/-- Calculates the total purchase cost for a given plan --/
def total_purchase_cost (prices : KeychainPrices) (plan : PurchasePlan) : ℕ :=
  prices.purchase_a * plan.quantity_a + prices.purchase_b * plan.quantity_b

/-- Calculates the total profit for a given plan --/
def total_profit (prices : KeychainPrices) (plan : PurchasePlan) : ℕ :=
  (prices.selling_a - prices.purchase_a) * plan.quantity_a +
  (prices.selling_b - prices.purchase_b) * plan.quantity_b

/-- Theorem: The optimal purchase plan maximizes profit --/
theorem optimal_purchase_plan (prices : KeychainPrices)
  (h_prices : prices.purchase_a = 30 ∧ prices.purchase_b = 25 ∧
              prices.selling_a = 45 ∧ prices.selling_b = 37) :
  ∃ (plan : PurchasePlan),
    plan.quantity_a + plan.quantity_b = 80 ∧
    total_purchase_cost prices plan ≤ 2200 ∧
    total_profit prices plan = 1080 ∧
    ∀ (other_plan : PurchasePlan),
      other_plan.quantity_a + other_plan.quantity_b = 80 →
      total_purchase_cost prices other_plan ≤ 2200 →
      total_profit prices other_plan ≤ total_profit prices plan :=
sorry

end optimal_purchase_plan_l2291_229130


namespace olivine_stones_difference_l2291_229181

theorem olivine_stones_difference (agate_stones olivine_stones diamond_stones : ℕ) : 
  agate_stones = 30 →
  olivine_stones > agate_stones →
  diamond_stones = olivine_stones + 11 →
  agate_stones + olivine_stones + diamond_stones = 111 →
  olivine_stones = agate_stones + 5 := by
sorry

end olivine_stones_difference_l2291_229181


namespace average_sq_feet_per_person_closest_to_500000_l2291_229108

/-- The population of the United States in 1980 -/
def us_population : ℕ := 226504825

/-- The area of the United States in square miles -/
def us_area : ℕ := 3615122

/-- The number of square feet in one square mile -/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- The options for the average number of square feet per person -/
def options : List ℕ := [5000, 10000, 50000, 100000, 500000]

/-- The theorem stating that the average number of square feet per person
    is closest to 500,000 among the given options -/
theorem average_sq_feet_per_person_closest_to_500000 :
  let total_sq_feet : ℕ := us_area * sq_feet_per_sq_mile
  let avg_sq_feet_per_person : ℚ := (total_sq_feet : ℚ) / us_population
  (500000 : ℚ) = options.argmin (fun x => |avg_sq_feet_per_person - x|) := by
  sorry

end average_sq_feet_per_person_closest_to_500000_l2291_229108


namespace square_roots_theorem_l2291_229104

theorem square_roots_theorem (a : ℝ) :
  (∃ x : ℝ, x^2 = (a + 3)^2 ∧ x^2 = (2*a - 9)^2) →
  (a + 3)^2 = 25 :=
by sorry

end square_roots_theorem_l2291_229104


namespace fraction_equation_solution_l2291_229151

theorem fraction_equation_solution : 
  ∃ x : ℚ, (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) ∧ (x = 3/2) :=
by sorry

end fraction_equation_solution_l2291_229151


namespace triangle_properties_l2291_229103

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Given conditions
  b * (1 + Real.cos C) = c * (2 - Real.cos B) →
  C = π / 3 →
  1/2 * a * b * Real.sin C = 4 * Real.sqrt 3 →
  -- Conclusions to prove
  (a + b = 2 * c ∧ c = 4) := by
  sorry


end triangle_properties_l2291_229103


namespace man_ownership_proof_l2291_229145

/-- The fraction of the business owned by the man -/
def man_ownership : ℚ := 2/3

/-- The value of the entire business in rupees -/
def business_value : ℕ := 60000

/-- The amount received from selling 3/4 of the man's shares in rupees -/
def sale_amount : ℕ := 30000

/-- The fraction of the man's shares that were sold -/
def sold_fraction : ℚ := 3/4

theorem man_ownership_proof :
  man_ownership * sold_fraction * business_value = sale_amount :=
sorry

end man_ownership_proof_l2291_229145


namespace intersection_distance_sum_l2291_229157

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 1 + (2/Real.sqrt 5)*t ∧ y = 1 + (1/Real.sqrt 5)*t

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ A.1 A.2 ∧ curve_C A.1 A.2 ∧
    line_l t₂ B.1 B.2 ∧ curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem intersection_distance_sum :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
  4 * Real.sqrt 15 :=
sorry

end intersection_distance_sum_l2291_229157


namespace min_value_expression_l2291_229111

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  (4 + x) * (1 + x) / x ≥ 9 ∧ ∃ y > 0, (4 + y) * (1 + y) / y = 9 :=
sorry

end min_value_expression_l2291_229111


namespace tan_sum_pi_twelfths_l2291_229184

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end tan_sum_pi_twelfths_l2291_229184


namespace intersection_of_P_and_Q_l2291_229116

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = -x + 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≤ 2} := by
  sorry

end intersection_of_P_and_Q_l2291_229116


namespace phone_selling_price_l2291_229159

theorem phone_selling_price 
  (total_phones : ℕ) 
  (initial_investment : ℚ) 
  (profit_ratio : ℚ) :
  total_phones = 200 →
  initial_investment = 3000 →
  profit_ratio = 1/3 →
  (initial_investment + profit_ratio * initial_investment) / total_phones = 20 := by
sorry

end phone_selling_price_l2291_229159


namespace smallest_divisor_with_remainder_l2291_229143

theorem smallest_divisor_with_remainder (n : ℕ) : 
  (∃ (k : ℕ), n = 10 * k) ∧ 
  (19^19 + 19) % n = 18 ∧ 
  (∀ m : ℕ, m < n → m % 10 = 0 → (19^19 + 19) % m ≠ 18) → 
  n = 10 := by
sorry

end smallest_divisor_with_remainder_l2291_229143


namespace chemistry_class_b_count_l2291_229147

/-- Represents the number of students who earn each grade in a chemistry class. -/
structure GradeDistribution where
  a : ℝ  -- Number of students earning A
  b : ℝ  -- Number of students earning B
  c : ℝ  -- Number of students earning C
  d : ℝ  -- Number of students earning D

/-- The grade distribution in a chemistry class of 50 students satisfies given probability ratios. -/
def chemistryClass (g : GradeDistribution) : Prop :=
  g.a = 0.5 * g.b ∧
  g.c = 1.2 * g.b ∧
  g.d = 0.3 * g.b ∧
  g.a + g.b + g.c + g.d = 50

/-- The number of students earning a B in the chemistry class is 50/3. -/
theorem chemistry_class_b_count :
  ∀ g : GradeDistribution, chemistryClass g → g.b = 50 / 3 := by
  sorry

end chemistry_class_b_count_l2291_229147


namespace equation_solutions_l2291_229133

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 8 = x + 4 ∧ x = 6) ∧
  (∃ x : ℚ, 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) ∧ x = -2) ∧
  (∃ x : ℚ, (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ∧ x = -20) ∧
  (∃ y : ℚ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 ∧ y = -1) :=
by sorry

end equation_solutions_l2291_229133


namespace tv_screen_horizontal_length_l2291_229155

/-- Represents a rectangular TV screen --/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- Theorem: Given a TV screen with horizontal to vertical ratio of 9:12 and
    diagonal of 32 inches, the horizontal length is 25.6 inches --/
theorem tv_screen_horizontal_length 
  (tv : TVScreen) 
  (ratio : tv.horizontal / tv.vertical = 9 / 12) 
  (diag : tv.diagonal = 32) :
  tv.horizontal = 25.6 := by
  sorry

#check tv_screen_horizontal_length

end tv_screen_horizontal_length_l2291_229155


namespace basketball_sales_solution_l2291_229115

/-- Represents the cost and sales information for basketballs --/
structure BasketballSales where
  cost_a : ℝ  -- Cost of A brand basketball
  cost_b : ℝ  -- Cost of B brand basketball
  price_a : ℝ  -- Original selling price of A brand basketball
  markup_b : ℝ  -- Markup percentage for B brand basketball
  discount_a : ℝ  -- Discount percentage for A brand basketball

/-- Theorem stating the solution to the basketball sales problem --/
theorem basketball_sales_solution (s : BasketballSales) : 
  (40 * s.cost_a + 40 * s.cost_b = 7200) →
  (50 * s.cost_a + 30 * s.cost_b = 7400) →
  (s.price_a = 140) →
  (s.markup_b = 0.3) →
  (40 * (s.price_a - s.cost_a) + 10 * (s.price_a * (1 - s.discount_a / 100) - s.cost_a) + 
   30 * s.cost_b * s.markup_b = 2440) →
  (s.cost_a = 100 ∧ s.cost_b = 80 ∧ s.discount_a = 8) := by
  sorry


end basketball_sales_solution_l2291_229115


namespace rectangle_b_product_l2291_229129

theorem rectangle_b_product : ∀ b₁ b₂ : ℝ,
  (∃ (rect : Set (ℝ × ℝ)), 
    rect = {(x, y) | 3 ≤ y ∧ y ≤ 8 ∧ ((x = 2 ∧ b₁ ≤ x) ∨ (x = b₁ ∧ x ≤ 2)) ∧
            ((x = 2 ∧ x ≤ b₂) ∨ (x = b₂ ∧ 2 ≤ x))} ∧
    (∀ (p q : ℝ × ℝ), p ∈ rect ∧ q ∈ rect → 
      (p.1 = q.1 ∨ p.2 = q.2) ∧ 
      (p.1 ≠ q.1 ∨ p.2 ≠ q.2))) →
  b₁ * b₂ = -21 :=
by sorry


end rectangle_b_product_l2291_229129


namespace hidden_primes_average_l2291_229172

-- Define the type for our cards
structure Card where
  visible : ℕ
  hidden : ℕ

-- Define the property of being consecutive primes
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p > q) ∧ ∀ k, q < k → k < p → ¬Nat.Prime k

-- State the theorem
theorem hidden_primes_average (card1 card2 : Card) :
  card1.visible = 18 →
  card2.visible = 27 →
  card1.visible + card1.hidden = card2.visible + card2.hidden →
  ConsecutivePrimes card1.hidden card2.hidden →
  card1.hidden - card2.hidden = 9 →
  (card1.hidden + card2.hidden) / 2 = 15 := by
  sorry

end hidden_primes_average_l2291_229172


namespace smallest_k_value_l2291_229183

/-- Given positive integers a, b, c, d, e, and k satisfying the conditions,
    prove that the smallest possible value for k is 522. -/
theorem smallest_k_value (a b c d e k : ℕ+) 
  (eq1 : a + 2*b + 3*c + 4*d + 5*e = k)
  (eq2 : 5*a = 4*b)
  (eq3 : 4*b = 3*c)
  (eq4 : 3*c = 2*d)
  (eq5 : 2*d = e) :
  k ≥ 522 ∧ (∃ (a' b' c' d' e' : ℕ+), 
    a' + 2*b' + 3*c' + 4*d' + 5*e' = 522 ∧
    5*a' = 4*b' ∧ 4*b' = 3*c' ∧ 3*c' = 2*d' ∧ 2*d' = e') := by
  sorry

end smallest_k_value_l2291_229183


namespace solution_part1_solution_part2_l2291_229118

/-- A system of linear equations in two variables x and y -/
structure LinearSystem where
  a : ℝ
  eq1 : ℝ → ℝ → ℝ := λ x y => x + 2 * y - a
  eq2 : ℝ → ℝ → ℝ := λ x y => 2 * x - y - 1

/-- The system has a solution when both equations equal zero -/
def HasSolution (s : LinearSystem) (x y : ℝ) : Prop :=
  s.eq1 x y = 0 ∧ s.eq2 x y = 0

theorem solution_part1 (s : LinearSystem) :
  HasSolution s 1 1 → s.a = 3 := by sorry

theorem solution_part2 (s : LinearSystem) :
  s.a = -2 → HasSolution s 0 (-1) ∧
  (∀ x y : ℝ, HasSolution s x y → x = 0 ∧ y = -1) := by sorry

end solution_part1_solution_part2_l2291_229118


namespace geometric_to_arithmetic_squares_possible_l2291_229156

/-- Given three real numbers that form a geometric sequence and are not all equal,
    it's possible for their squares to form an arithmetic sequence. -/
theorem geometric_to_arithmetic_squares_possible (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 1 ∧ b = a * r ∧ c = b * r) →  -- Geometric sequence condition
  (a ≠ b ∨ b ≠ c) →                          -- Not all equal condition
  ∃ x y z : ℝ, x = a^2 ∧ y = b^2 ∧ z = c^2 ∧  -- Squares of a, b, c
            y - x = z - y                    -- Arithmetic sequence condition
    := by sorry

end geometric_to_arithmetic_squares_possible_l2291_229156


namespace arithmetic_sequence_first_term_l2291_229135

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = a 0 + n * d) →  -- arithmetic sequence definition
  (a 19 = 205) →             -- given condition a_20 = 205 (index starts at 0)
  (a 0 = 91) :=              -- prove a_1 = 91 (a_1 is a 0 in 0-indexed notation)
by
  sorry

end arithmetic_sequence_first_term_l2291_229135


namespace map_distance_twenty_cm_distance_l2291_229100

-- Define the scale of the map
def map_scale (cm : ℝ) (km : ℝ) : Prop := cm * (54 / 9) = km

-- Theorem statement
theorem map_distance (cm : ℝ) :
  map_scale 9 54 → map_scale cm (cm * 6) :=
by
  sorry

-- The specific case for 20 cm
theorem twenty_cm_distance :
  map_scale 9 54 → map_scale 20 120 :=
by
  sorry

end map_distance_twenty_cm_distance_l2291_229100


namespace distance_calculation_l2291_229182

/-- The speed of light in km/s -/
def speed_of_light : ℝ := 3 * 10^5

/-- The time it takes for light to reach Earth from Proxima Centauri in years -/
def travel_time : ℝ := 4

/-- The number of seconds in a year -/
def seconds_per_year : ℝ := 3 * 10^7

/-- The distance from Proxima Centauri to Earth in km -/
def distance_to_proxima_centauri : ℝ := speed_of_light * travel_time * seconds_per_year

theorem distance_calculation :
  distance_to_proxima_centauri = 3.6 * 10^13 := by
  sorry

end distance_calculation_l2291_229182


namespace metro_earnings_proof_l2291_229137

/-- Calculates the earnings from ticket sales given the ticket price, average tickets sold per minute, and duration in minutes. -/
def calculate_earnings (ticket_price : ℝ) (tickets_per_minute : ℝ) (duration : ℝ) : ℝ :=
  ticket_price * tickets_per_minute * duration

/-- Proves that the earnings from ticket sales for the given conditions equal $90. -/
theorem metro_earnings_proof (ticket_price : ℝ) (tickets_per_minute : ℝ) (duration : ℝ)
  (h1 : ticket_price = 3)
  (h2 : tickets_per_minute = 5)
  (h3 : duration = 6) :
  calculate_earnings ticket_price tickets_per_minute duration = 90 := by
  sorry

end metro_earnings_proof_l2291_229137


namespace exponential_regression_model_l2291_229192

/-- Given a model y = ce^(kx) and its transformed linear equation z = 0.3x + 4 where z = ln y,
    prove that c = e^4 and k = 0.3 -/
theorem exponential_regression_model (c k : ℝ) : 
  (∀ x y : ℝ, y = c * Real.exp (k * x)) →
  (∀ x z : ℝ, z = 0.3 * x + 4) →
  (∀ y : ℝ, z = Real.log y) →
  c = Real.exp 4 ∧ k = 0.3 := by
  sorry

end exponential_regression_model_l2291_229192


namespace operation_2011_result_l2291_229139

def operation_result (n : ℕ) : ℕ :=
  match n % 3 with
  | 1 => 133
  | 2 => 55
  | 0 => 250
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

theorem operation_2011_result :
  operation_result 2011 = 133 :=
sorry

end operation_2011_result_l2291_229139


namespace red_balls_in_box_l2291_229177

/-- Given a box with an initial number of red balls and a number of red balls added,
    calculate the final number of red balls in the box. -/
def final_red_balls (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The final number of red balls in the box is 7 when starting with 5 and adding 2. -/
theorem red_balls_in_box : final_red_balls 5 2 = 7 := by
  sorry

end red_balls_in_box_l2291_229177


namespace only_B_is_certain_l2291_229138

-- Define the type for events
inductive Event : Type
  | A : Event  -- It will be sunny on New Year's Day in 2020
  | B : Event  -- The sun rises from the east
  | C : Event  -- The TV is turned on and broadcasting the news
  | D : Event  -- Drawing a red ball from a box without any red balls

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.B => True
  | _ => False

-- Theorem statement
theorem only_B_is_certain :
  ∀ e : Event, is_certain e ↔ e = Event.B :=
by sorry

end only_B_is_certain_l2291_229138


namespace age_inconsistency_l2291_229149

/-- Given the ages of Sandy, Molly, and Noah, this theorem proves that the given conditions lead to a contradiction. -/
theorem age_inconsistency (S M N : ℕ) : 
  (M = S + 20) →  -- Sandy is younger than Molly by 20 years
  (S : ℚ) / M = 7 / 9 →  -- The ratio of Sandy's age to Molly's age is 7:9
  S + M + N = 120 →  -- The sum of their ages is 120
  (N - M : ℚ) = (1 / 2 : ℚ) * (M - S) →  -- The age difference between Noah and Molly is half that between Sandy and Molly
  False :=
by
  sorry

#eval 70 + 90  -- This evaluates to 160, which is already greater than 120


end age_inconsistency_l2291_229149


namespace whitewashing_cost_is_7580_l2291_229121

/-- Calculates the cost of white washing a trapezoidal room with given dimensions and conditions -/
def whitewashingCost (length width height1 height2 : ℕ) (doorCount windowCount : ℕ) 
  (doorLength doorWidth windowLength windowWidth : ℕ) (decorationArea : ℕ) (ratePerSqFt : ℕ) : ℕ :=
  let totalWallArea := 2 * (length * height1 + width * height2)
  let doorArea := doorCount * doorLength * doorWidth
  let windowArea := windowCount * windowLength * windowWidth
  let adjustedArea := totalWallArea - doorArea - windowArea - decorationArea
  adjustedArea * ratePerSqFt

/-- Theorem stating that the cost of white washing the given trapezoidal room is 7580 -/
theorem whitewashing_cost_is_7580 : 
  whitewashingCost 25 15 12 8 2 3 6 3 4 3 10 10 = 7580 := by
  sorry

end whitewashing_cost_is_7580_l2291_229121


namespace inequality_proof_l2291_229162

theorem inequality_proof (x y : ℝ) (h : x > y) : -2 * x < -2 * y := by
  sorry

end inequality_proof_l2291_229162


namespace book_cost_l2291_229160

theorem book_cost (initial_money : ℕ) (notebooks : ℕ) (notebook_cost : ℕ) (books : ℕ) (money_left : ℕ) : 
  initial_money = 56 →
  notebooks = 7 →
  notebook_cost = 4 →
  books = 2 →
  money_left = 14 →
  (initial_money - money_left - notebooks * notebook_cost) / books = 7 := by
  sorry

end book_cost_l2291_229160


namespace road_trip_total_hours_l2291_229136

/-- Calculates the total hours driven on a road trip -/
def total_hours_driven (days : ℕ) (jade_hours_per_day : ℕ) (krista_hours_per_day : ℕ) : ℕ :=
  days * (jade_hours_per_day + krista_hours_per_day)

/-- Proves that the total hours driven by Jade and Krista over 3 days equals 42 hours -/
theorem road_trip_total_hours : total_hours_driven 3 8 6 = 42 := by
  sorry

#eval total_hours_driven 3 8 6

end road_trip_total_hours_l2291_229136


namespace ring_toss_total_earnings_l2291_229106

/-- The ring toss game's earnings over a period of days -/
def ring_toss_earnings (days : ℕ) (daily_income : ℕ) : ℕ :=
  days * daily_income

/-- Theorem: The ring toss game's total earnings over 3 days at $140 per day is $420 -/
theorem ring_toss_total_earnings : ring_toss_earnings 3 140 = 420 := by
  sorry

end ring_toss_total_earnings_l2291_229106


namespace binomial_expansions_l2291_229188

theorem binomial_expansions (a b : ℝ) : 
  ((a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3) ∧ 
  ((a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4) ∧
  ((a + b)^5 = a^5 + 5*a^4*b + 10*a^3*b^2 + 10*a^2*b^3 + 5*a*b^4 + b^5) :=
by sorry

end binomial_expansions_l2291_229188


namespace zeros_arithmetic_sequence_implies_a_value_l2291_229120

/-- A cubic polynomial function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + x + a

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 1

/-- Theorem: If the zeros of f form an arithmetic sequence, then a = -23/54 -/
theorem zeros_arithmetic_sequence_implies_a_value (a : ℝ) : 
  (∃ r s t : ℝ, (f a r = 0 ∧ f a s = 0 ∧ f a t = 0) ∧ 
   (s - r = t - s) ∧ (r < s ∧ s < t)) → 
  a = -23/54 := by
  sorry

end zeros_arithmetic_sequence_implies_a_value_l2291_229120


namespace g_of_5_equals_18_l2291_229191

/-- Given a function g where g(x) = 4x - 2 for all x, prove that g(5) = 18 -/
theorem g_of_5_equals_18 (g : ℝ → ℝ) (h : ∀ x, g x = 4 * x - 2) : g 5 = 18 := by
  sorry

end g_of_5_equals_18_l2291_229191


namespace sheersCost_is_40_l2291_229198

/-- The cost of window treatments for a house with 3 windows, where each window
    requires a pair of sheers and a pair of drapes. -/
def WindowTreatmentsCost (sheersCost : ℚ) : ℚ :=
  3 * (sheersCost + 60)

/-- Theorem stating that the cost of a pair of sheers is $40, given the conditions. -/
theorem sheersCost_is_40 :
  ∃ (sheersCost : ℚ), WindowTreatmentsCost sheersCost = 300 ∧ sheersCost = 40 :=
sorry

end sheersCost_is_40_l2291_229198


namespace units_digit_of_2_pow_20_minus_1_l2291_229142

theorem units_digit_of_2_pow_20_minus_1 : 
  (2^20 - 1) % 10 = 5 := by sorry

end units_digit_of_2_pow_20_minus_1_l2291_229142


namespace max_value_of_function_l2291_229117

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ (y : ℝ), y = 1/27 ∧ ∀ (z : ℝ), 0 < z ∧ z < 1/2 → x^2 * (1 - 2*x) ≤ y := by
  sorry

end max_value_of_function_l2291_229117


namespace illegal_parking_percentage_l2291_229122

theorem illegal_parking_percentage (total_cars : ℕ) (towed_cars : ℕ) (illegal_cars : ℕ) :
  towed_cars = (2 : ℕ) * total_cars / 100 →
  (80 : ℕ) * illegal_cars / 100 = illegal_cars - towed_cars →
  illegal_cars * 100 / total_cars = 10 := by
  sorry

end illegal_parking_percentage_l2291_229122


namespace opposite_silver_is_orange_l2291_229168

/-- Represents the colors of the cube faces -/
inductive Color
  | Blue
  | Orange
  | Black
  | Yellow
  | Silver
  | Violet

/-- Represents the positions of the cube faces -/
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

/-- Represents a view of the cube -/
structure View where
  top : Color
  front : Color
  right : Color

/-- The cube with its colored faces -/
structure Cube where
  faces : Position → Color

def first_view : View :=
  { top := Color.Blue, front := Color.Yellow, right := Color.Violet }

def second_view : View :=
  { top := Color.Blue, front := Color.Silver, right := Color.Violet }

def third_view : View :=
  { top := Color.Blue, front := Color.Black, right := Color.Violet }

theorem opposite_silver_is_orange (c : Cube) :
  (c.faces Position.Front = Color.Silver) →
  (c.faces Position.Top = Color.Blue) →
  (c.faces Position.Right = Color.Violet) →
  (c.faces Position.Back = Color.Orange) :=
by sorry

end opposite_silver_is_orange_l2291_229168


namespace small_panda_bamboo_consumption_l2291_229102

/-- The amount of bamboo eaten by small pandas each day -/
def small_panda_bamboo : ℝ := 100

/-- The number of small panda bears -/
def num_small_pandas : ℕ := 4

/-- The number of bigger panda bears -/
def num_big_pandas : ℕ := 5

/-- The amount of bamboo eaten by each bigger panda bear per day -/
def big_panda_bamboo : ℝ := 40

/-- The total amount of bamboo eaten by all pandas in a week -/
def total_weekly_bamboo : ℝ := 2100

theorem small_panda_bamboo_consumption :
  small_panda_bamboo * num_small_pandas +
  big_panda_bamboo * num_big_pandas =
  total_weekly_bamboo / 7 :=
by sorry

end small_panda_bamboo_consumption_l2291_229102


namespace simplest_quadratic_root_l2291_229194

theorem simplest_quadratic_root (x : ℝ) : 
  (∃ (k : ℚ), Real.sqrt (x + 1) = k * Real.sqrt (5 / 2)) → x = 9 := by
  sorry

end simplest_quadratic_root_l2291_229194


namespace solution_set_theorem_l2291_229195

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, f x + x * (deriv f x) > 0)

-- Define the theorem
theorem solution_set_theorem :
  {x : ℝ | (deriv f (Real.sqrt (x + 1))) > Real.sqrt (x - 1) * f (Real.sqrt (x^2 - 1))} =
  {x : ℝ | 1 ≤ x ∧ x < 2} :=
sorry

end solution_set_theorem_l2291_229195


namespace cube_root_equation_solution_l2291_229110

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  sorry

end cube_root_equation_solution_l2291_229110


namespace pearl_string_value_l2291_229174

/-- Represents a string of pearls with a given middle pearl value and decreasing rates on each side. -/
structure PearlString where
  middleValue : ℕ
  decreaseRate1 : ℕ
  decreaseRate2 : ℕ

/-- Calculates the total value of the pearl string. -/
def totalValue (ps : PearlString) : ℕ :=
  ps.middleValue + 16 * ps.middleValue - 16 * 17 * ps.decreaseRate1 / 2 +
  16 * ps.middleValue - 16 * 17 * ps.decreaseRate2 / 2

/-- Calculates the value of the fourth pearl from the middle on the more expensive side. -/
def fourthPearlValue (ps : PearlString) : ℕ :=
  ps.middleValue - 4 * min ps.decreaseRate1 ps.decreaseRate2

/-- The main theorem stating the conditions and the result to be proven. -/
theorem pearl_string_value (ps : PearlString) :
  ps.decreaseRate1 = 3000 →
  ps.decreaseRate2 = 4500 →
  totalValue ps = 25 * fourthPearlValue ps →
  ps.middleValue = 90000 := by
  sorry

end pearl_string_value_l2291_229174


namespace order_of_sqrt_differences_l2291_229173

theorem order_of_sqrt_differences :
  let m : ℝ := Real.sqrt 6 - Real.sqrt 5
  let n : ℝ := Real.sqrt 7 - Real.sqrt 6
  let p : ℝ := Real.sqrt 8 - Real.sqrt 7
  m > n ∧ n > p :=
by sorry

end order_of_sqrt_differences_l2291_229173


namespace frog_eggs_theorem_l2291_229126

/-- The number of eggs laid by a frog in a year -/
def eggs_laid : ℕ := 593

/-- The fraction of eggs that don't dry up -/
def not_dried_up : ℚ := 9/10

/-- The fraction of remaining eggs that are not eaten -/
def not_eaten : ℚ := 3/10

/-- The fraction of remaining eggs that hatch -/
def hatch_rate : ℚ := 1/4

/-- The number of frogs that hatch -/
def frogs_hatched : ℕ := 40

theorem frog_eggs_theorem :
  ↑frogs_hatched = ⌈(↑eggs_laid * not_dried_up * not_eaten * hatch_rate)⌉ := by sorry

end frog_eggs_theorem_l2291_229126


namespace negative_expressions_l2291_229185

theorem negative_expressions (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := by
  sorry

end negative_expressions_l2291_229185


namespace paint_usage_fraction_l2291_229127

theorem paint_usage_fraction (total_paint : ℚ) (total_used : ℚ) : 
  total_paint = 360 →
  total_used = 264 →
  let first_week_fraction := (5 : ℚ) / 9
  let remaining_after_first := total_paint * (1 - first_week_fraction)
  let second_week_usage := remaining_after_first / 5
  total_used = total_paint * first_week_fraction + second_week_usage →
  first_week_fraction = 5 / 9 := by
sorry

end paint_usage_fraction_l2291_229127


namespace right_triangle_side_relation_l2291_229171

theorem right_triangle_side_relation (a d : ℝ) :
  (a > 0) →
  (d > 0) →
  (a ≤ a + 2*d) →
  (a + 2*d ≤ a + 4*d) →
  (a + 4*d)^2 = a^2 + (a + 2*d)^2 →
  a = d*(1 + Real.sqrt 7) :=
by sorry

end right_triangle_side_relation_l2291_229171


namespace complex_calculations_l2291_229193

theorem complex_calculations :
  (∃ (i : ℂ), i * i = -1) →
  (∃ (z₁ z₂ : ℂ),
    (1 - i) * (-1/2 + (Real.sqrt 3)/2 * i) * (1 + i) = z₁ ∧
    z₁ = -1 + Real.sqrt 3 * i ∧
    (2 + 2*i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i))^2010 = z₂ ∧
    z₂ = -1 - 2*i) :=
by sorry

end complex_calculations_l2291_229193


namespace root_equation_q_value_l2291_229197

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 2/b)^2 - p*(a + 2/b) + q = 0) →
  ((b + 2/a)^2 - p*(b + 2/a) + q = 0) →
  (q = 25/3) := by
sorry

end root_equation_q_value_l2291_229197


namespace athlete_arrangement_count_l2291_229150

theorem athlete_arrangement_count : ℕ := by
  -- Define the number of athletes and tracks
  let num_athletes : ℕ := 6
  let num_tracks : ℕ := 6

  -- Define the restrictions for athletes A and B
  let a_possible_tracks : ℕ := 4  -- A can't be on 1st or 2nd track
  let b_possible_tracks : ℕ := 2  -- B must be on 5th or 6th track

  -- Define the number of remaining athletes to be arranged
  let remaining_athletes : ℕ := num_athletes - 2  -- excluding A and B

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := b_possible_tracks * a_possible_tracks * (Nat.factorial remaining_athletes)

  -- Prove that the total number of arrangements is 144
  sorry

end athlete_arrangement_count_l2291_229150


namespace student_take_home_pay_l2291_229124

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takeHomePay (baseSalary : ℝ) (bonus : ℝ) (taxRate : ℝ) : ℝ :=
  let totalEarnings := baseSalary + bonus
  let taxAmount := totalEarnings * taxRate
  totalEarnings - taxAmount

/-- Theorem: The take-home pay for a well-performing student is 26,100 rubles --/
theorem student_take_home_pay :
  takeHomePay 25000 5000 0.13 = 26100 := by
  sorry

#eval takeHomePay 25000 5000 0.13

end student_take_home_pay_l2291_229124


namespace problem_statement_l2291_229175

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 3 * a * (b^2 - 1) = b * (1 - a^2)) : 
  (1 / a + 3 / b = a + 3 * b) ∧ 
  (a^(3/2) * b^(1/2) + 3 * a^(1/2) * b^(3/2) ≥ 2 * Real.sqrt 3) := by
  sorry

end problem_statement_l2291_229175


namespace count_perfect_square_factors_51200_l2291_229114

/-- The number of factors of 51200 that are perfect squares -/
def perfect_square_factors_of_51200 : ℕ :=
  (Finset.range 6).card * (Finset.range 2).card

/-- Theorem stating that the number of factors of 51200 that are perfect squares is 12 -/
theorem count_perfect_square_factors_51200 :
  perfect_square_factors_of_51200 = 12 := by
  sorry

end count_perfect_square_factors_51200_l2291_229114


namespace reciprocal_sum_of_roots_l2291_229169

theorem reciprocal_sum_of_roots (a b c : ℚ) (α β : ℚ) :
  a ≠ 0 →
  (∃ x y : ℚ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) →
  (∀ x : ℚ, a * x^2 + b * x + c = 0 → (α = 1/x ∨ β = 1/x)) →
  a = 6 ∧ b = 5 ∧ c = 7 →
  α + β = -5/7 := by
sorry

end reciprocal_sum_of_roots_l2291_229169


namespace pen_cost_proof_l2291_229190

theorem pen_cost_proof (total_students : ℕ) (total_cost : ℚ) : ∃ (buyers pens_per_student cost_per_pen : ℕ),
  total_students = 40 ∧
  total_cost = 2091 / 100 ∧
  buyers > total_students / 2 ∧
  buyers ≤ total_students ∧
  pens_per_student % 2 = 1 ∧
  pens_per_student > 1 ∧
  Nat.Prime cost_per_pen ∧
  buyers * pens_per_student * cost_per_pen = 2091 ∧
  cost_per_pen = 47 := by
sorry

end pen_cost_proof_l2291_229190


namespace cubic_odd_and_increasing_l2291_229186

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end cubic_odd_and_increasing_l2291_229186


namespace unique_cubic_zero_a_range_l2291_229119

/-- A cubic function with a unique positive zero point -/
structure UniqueCubicZero where
  a : ℝ
  f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * x^2 + 1
  x₀ : ℝ
  x₀_pos : x₀ > 0
  x₀_zero : f x₀ = 0
  unique_zero : ∀ x, f x = 0 → x = x₀

/-- The range of 'a' for a cubic function with a unique positive zero point -/
theorem unique_cubic_zero_a_range (c : UniqueCubicZero) : c.a < -2 := by
  sorry

end unique_cubic_zero_a_range_l2291_229119


namespace candy_cost_l2291_229154

theorem candy_cost (initial_amount pencil_cost remaining_amount : ℕ) 
  (h1 : initial_amount = 43)
  (h2 : pencil_cost = 20)
  (h3 : remaining_amount = 18) :
  initial_amount - pencil_cost - remaining_amount = 5 := by
  sorry

end candy_cost_l2291_229154


namespace binomial_coefficient_ratio_l2291_229123

theorem binomial_coefficient_ratio : ∀ a₀ a₁ a₂ a₃ a₄ a₅ : ℤ,
  (∀ x : ℤ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 :=
by sorry

end binomial_coefficient_ratio_l2291_229123


namespace inverse_of_A_l2291_229131

-- Define matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 1, 8]

-- Define the proposed inverse of A
def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1/2, 0; -1/16, 1/8]

-- Theorem statement
theorem inverse_of_A : A⁻¹ = A_inv := by sorry

end inverse_of_A_l2291_229131


namespace fraction_equality_l2291_229189

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 := by
  sorry

end fraction_equality_l2291_229189


namespace unique_prime_square_sum_l2291_229166

theorem unique_prime_square_sum (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → ∃ (n : ℕ), p^(q+1) + q^(p+1) = n^2 → p = 2 ∧ q = 2 := by
  sorry

end unique_prime_square_sum_l2291_229166


namespace min_club_members_l2291_229167

theorem min_club_members (N : ℕ) : N < 80 ∧ 
  ((N - 5) % 8 = 0 ∨ (N - 5) % 7 = 0) ∧ 
  N % 9 = 7 → 
  N ≥ 61 :=
by sorry

end min_club_members_l2291_229167


namespace marked_up_percentage_l2291_229105

theorem marked_up_percentage 
  (cost_price selling_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 540)
  (h2 : selling_price = 456)
  (h3 : discount_percentage = 26.570048309178745) :
  (((selling_price / (1 - discount_percentage / 100) - cost_price) / cost_price) * 100 = 15) :=
by sorry

end marked_up_percentage_l2291_229105


namespace finley_tickets_l2291_229152

theorem finley_tickets (total_tickets : ℕ) (ratio_jensen : ℕ) (ratio_finley : ℕ) : 
  total_tickets = 400 →
  ratio_jensen = 4 →
  ratio_finley = 11 →
  (3 * total_tickets / 4) * ratio_finley / (ratio_jensen + ratio_finley) = 220 := by
  sorry

end finley_tickets_l2291_229152


namespace supermarket_purchase_cost_l2291_229196

/-- Calculates the total cost of items with given quantities, prices, and discounts -/
def totalCost (quantities : List ℕ) (prices : List ℚ) (discounts : List ℚ) : ℚ :=
  List.sum (List.zipWith3 (fun q p d => q * p * (1 - d)) quantities prices discounts)

/-- The problem statement -/
theorem supermarket_purchase_cost : 
  let quantities : List ℕ := [24, 6, 5, 3]
  let prices : List ℚ := [9/5, 17/10, 17/5, 56/5]
  let discounts : List ℚ := [1/5, 1/5, 0, 1/10]
  totalCost quantities prices discounts = 4498/50
  := by sorry

end supermarket_purchase_cost_l2291_229196


namespace trig_problem_l2291_229179

theorem trig_problem (α β : Real) 
  (h1 : Real.sin (Real.pi - α) - 2 * Real.sin (Real.pi / 2 + α) = 0) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.sin α * Real.cos α + Real.sin α ^ 2 = 6 / 5 ∧ Real.tan β = 3 := by
  sorry

end trig_problem_l2291_229179


namespace square_area_17m_l2291_229113

theorem square_area_17m (side_length : ℝ) (h : side_length = 17) :
  side_length * side_length = 289 := by
  sorry

end square_area_17m_l2291_229113


namespace same_volume_prisms_l2291_229148

def edge_lengths : List ℕ := [12, 18, 20, 24, 30, 33, 70, 24, 154]

def is_valid_prism (a b c : ℕ) : Bool :=
  a ∈ edge_lengths ∧ b ∈ edge_lengths ∧ c ∈ edge_lengths

def prism_volume (a b c : ℕ) : ℕ := a * b * c

theorem same_volume_prisms :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℕ),
    is_valid_prism a₁ b₁ c₁ ∧
    is_valid_prism a₂ b₂ c₂ ∧
    is_valid_prism a₃ b₃ c₃ ∧
    prism_volume a₁ b₁ c₁ = prism_volume a₂ b₂ c₂ ∧
    prism_volume a₂ b₂ c₂ = prism_volume a₃ b₃ c₃ ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧
    (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) :=
by
  sorry

#check same_volume_prisms

end same_volume_prisms_l2291_229148


namespace extreme_points_range_l2291_229161

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + a + 1) * Real.exp x

theorem extreme_points_range (a : ℝ) (x₁ x₂ : ℝ) (h₁ : a > 0) (h₂ : x₁ < x₂)
  (h₃ : ∀ x, f a x = 0 → x = x₁ ∨ x = x₂)
  (h₄ : ∀ m : ℝ, m * x₁ - f a x₂ / Real.exp x₁ > 0) :
  ∀ m : ℝ, m ≥ 2 ↔ m * x₁ - f a x₂ / Real.exp x₁ > 0 :=
by sorry

end extreme_points_range_l2291_229161


namespace book_pages_count_l2291_229146

/-- The number of pages Liam read in a week -/
def total_pages : ℕ :=
  let first_three_days := 3 * 40
  let next_three_days := 3 * 50
  let seventh_day_first_session := 15
  let seventh_day_second_session := 2 * seventh_day_first_session
  first_three_days + next_three_days + seventh_day_first_session + seventh_day_second_session

/-- Theorem stating that the total number of pages in the book is 315 -/
theorem book_pages_count : total_pages = 315 := by
  sorry

end book_pages_count_l2291_229146


namespace samantha_routes_l2291_229170

/-- Represents a location on a grid --/
structure Location :=
  (x : ℤ) (y : ℤ)

/-- Calculates the number of shortest paths between two locations --/
def num_shortest_paths (start finish : Location) : ℕ :=
  sorry

/-- Samantha's home location relative to the southwest corner of City Park --/
def home : Location :=
  { x := -1, y := -3 }

/-- Southwest corner of City Park --/
def park_sw : Location :=
  { x := 0, y := 0 }

/-- Northeast corner of City Park --/
def park_ne : Location :=
  { x := 0, y := 0 }

/-- Samantha's school location relative to the northeast corner of City Park --/
def school : Location :=
  { x := 3, y := 1 }

/-- Library location relative to the school --/
def library : Location :=
  { x := 2, y := 1 }

/-- Total number of routes Samantha can take --/
def total_routes : ℕ :=
  (num_shortest_paths home park_sw) *
  (num_shortest_paths park_ne school) *
  (num_shortest_paths school library)

theorem samantha_routes :
  total_routes = 48 :=
sorry

end samantha_routes_l2291_229170


namespace goods_train_length_l2291_229178

/-- Calculate the length of a goods train given the speeds of two trains moving in opposite directions and the time taken for the goods train to pass an observer in the other train. -/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) (passing_time : ℝ) :
  man_train_speed = 30 →
  goods_train_speed = 82 →
  passing_time = 9 →
  ∃ (length : ℝ), abs (length - 279.99) < 0.01 :=
by
  sorry

end goods_train_length_l2291_229178


namespace soft_taco_price_is_correct_l2291_229112

/-- The price of a hard shell taco -/
def hard_shell_price : ℝ := 5

/-- The number of hard shell tacos bought by the family -/
def family_hard_shells : ℕ := 4

/-- The number of soft tacos bought by the family -/
def family_soft_shells : ℕ := 3

/-- The number of other customers -/
def other_customers : ℕ := 10

/-- The number of soft tacos bought by each other customer -/
def soft_tacos_per_customer : ℕ := 2

/-- The total revenue during lunch rush -/
def total_revenue : ℝ := 66

/-- The price of a soft taco -/
def soft_taco_price : ℝ := 2

theorem soft_taco_price_is_correct :
  soft_taco_price * (family_soft_shells + other_customers * soft_tacos_per_customer) +
  hard_shell_price * family_hard_shells = total_revenue :=
by sorry

end soft_taco_price_is_correct_l2291_229112


namespace hyperbola_equation_l2291_229141

/-- A curve in the rectangular coordinate system (xOy) -/
structure Curve where
  -- The equation of the curve is implicitly defined by this function
  equation : ℝ → ℝ → Prop

/-- The eccentricity of a curve -/
def eccentricity (c : Curve) : ℝ := sorry

/-- Whether a point lies on a curve -/
def lies_on (p : ℝ × ℝ) (c : Curve) : Prop :=
  c.equation p.1 p.2

/-- The standard equation of a hyperbola -/
def is_standard_hyperbola_equation (c : Curve) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, c.equation x y ↔ y^2 - x^2 = a^2

theorem hyperbola_equation (c : Curve) 
  (h_ecc : eccentricity c = Real.sqrt 2)
  (h_point : lies_on (1, Real.sqrt 2) c) :
  is_standard_hyperbola_equation c ∧ 
  ∃ x y : ℝ, c.equation x y ↔ y^2 - x^2 = 1 := by sorry

end hyperbola_equation_l2291_229141


namespace sum_of_integers_l2291_229125

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 10)
  (eq2 : y - z + w = 15)
  (eq3 : z - w + x = 9)
  (eq4 : w - x + y = 4) :
  x + y + z + w = 38 := by
sorry

end sum_of_integers_l2291_229125


namespace two_tangent_lines_l2291_229164

/-- A line that passes through a point and intersects a parabola at only one point. -/
structure TangentLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The point through which the line passes -/
  point : ℝ × ℝ
  /-- The parabola equation in the form y^2 = ax -/
  parabola_coeff : ℝ

/-- The number of lines passing through a given point and tangent to a parabola -/
def count_tangent_lines (point : ℝ × ℝ) (parabola_coeff : ℝ) : ℕ :=
  sorry

/-- Theorem: There are exactly two lines that pass through point M(2, 4) 
    and intersect the parabola y^2 = 8x at only one point -/
theorem two_tangent_lines : count_tangent_lines (2, 4) 8 = 2 := by
  sorry

end two_tangent_lines_l2291_229164


namespace polynomial_expansion_l2291_229107

theorem polynomial_expansion (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) =
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end polynomial_expansion_l2291_229107


namespace units_digit_sum_squares_odd_plus_7_1011_l2291_229163

/-- The units digit of the sum of squares of the first n odd positive integers plus 7 -/
def units_digit_sum_squares_odd_plus_7 (n : ℕ) : ℕ :=
  (((List.range n).map (fun i => (2 * i + 1) ^ 2)).sum + 7) % 10

/-- Theorem stating that the units digit of the sum of squares of the first 1011 odd positive integers plus 7 is 2 -/
theorem units_digit_sum_squares_odd_plus_7_1011 :
  units_digit_sum_squares_odd_plus_7 1011 = 2 := by
  sorry

end units_digit_sum_squares_odd_plus_7_1011_l2291_229163


namespace function_always_negative_iff_a_in_range_l2291_229199

theorem function_always_negative_iff_a_in_range :
  ∀ (a : ℝ), (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end function_always_negative_iff_a_in_range_l2291_229199


namespace inverse_proportionality_l2291_229153

theorem inverse_proportionality (x y : ℝ) (P : ℝ) : 
  (x + y = 30) → (x - y = 12) → (x * y = P) → (3 * (P / 3) = 63) := by
  sorry

end inverse_proportionality_l2291_229153


namespace parallel_line_through_point_l2291_229176

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (given_line : Line) (p : Point) :
  ∃ (result_line : Line),
    pointOnLine p result_line ∧
    parallel result_line given_line ∧
    result_line.a = 2 ∧
    result_line.b = 1 ∧
    result_line.c = -1 :=
  sorry

end parallel_line_through_point_l2291_229176


namespace display_rows_count_l2291_229134

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The total number of cans in the first n rows of the display -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

theorem display_rows_count :
  ∃ n : ℕ, total_cans n = 169 ∧ n = 10 :=
sorry

end display_rows_count_l2291_229134


namespace parallelogram_area_l2291_229101

theorem parallelogram_area (side1 side2 angle : ℝ) (h_side1 : side1 = 20) (h_side2 : side2 = 30) (h_angle : angle = 40 * π / 180) :
  let height := side1 * Real.sin angle
  let area := side2 * height
  ∃ ε > 0, |area - 385.68| < ε :=
sorry

end parallelogram_area_l2291_229101


namespace triangle_perimeter_bound_l2291_229128

theorem triangle_perimeter_bound (a b c : ℝ) : 
  a = 7 → b = 23 → (a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + c → ↑n > p ∧ ∀ (m : ℕ), ↑m > p → m ≥ n :=
sorry

end triangle_perimeter_bound_l2291_229128


namespace square_park_area_l2291_229144

/-- The area of a square park with a side length of 30 meters is 900 square meters. -/
theorem square_park_area : 
  ∀ (park_side_length : ℝ), 
  park_side_length = 30 → 
  park_side_length * park_side_length = 900 :=
by
  sorry

end square_park_area_l2291_229144


namespace correct_quotient_proof_l2291_229158

theorem correct_quotient_proof (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 63) : D / 21 = 36 := by
  sorry

end correct_quotient_proof_l2291_229158


namespace annies_initial_apples_l2291_229180

theorem annies_initial_apples (initial_apples total_apples apples_from_nathan : ℕ) :
  total_apples = initial_apples + apples_from_nathan →
  apples_from_nathan = 6 →
  total_apples = 12 →
  initial_apples = 6 := by
sorry

end annies_initial_apples_l2291_229180


namespace selection_methods_count_l2291_229140

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of boys in the class -/
def num_boys : ℕ := 4

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The total number of students to be selected -/
def num_selected : ℕ := 4

/-- The number of ways to select 4 members from 4 boys and 2 girls, with at least 1 girl -/
def num_selections : ℕ := 
  binomial num_girls 1 * binomial num_boys 3 + 
  binomial num_girls 2 * binomial num_boys 2

theorem selection_methods_count : num_selections = 14 := by sorry

end selection_methods_count_l2291_229140


namespace condition_p_sufficient_not_necessary_l2291_229187

theorem condition_p_sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end condition_p_sufficient_not_necessary_l2291_229187


namespace fruit_arrangement_theorem_l2291_229109

def number_of_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) (group3 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2 * Nat.factorial group3)

theorem fruit_arrangement_theorem :
  number_of_arrangements 7 4 2 1 = 105 := by
  sorry

end fruit_arrangement_theorem_l2291_229109


namespace product_mod_25_l2291_229165

theorem product_mod_25 (m : ℕ) : 
  95 * 115 * 135 ≡ m [MOD 25] → 0 ≤ m → m < 25 → m = 0 := by
  sorry

end product_mod_25_l2291_229165
