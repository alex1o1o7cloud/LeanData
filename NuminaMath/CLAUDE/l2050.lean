import Mathlib

namespace money_division_l2050_205045

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  3 * q = 7 * p →
  3 * r = 4 * q →
  q - p = 2800 →
  r - q = 3500 :=
by sorry

end money_division_l2050_205045


namespace a_range_l2050_205076

theorem a_range (p : ∀ x > 0, x + 1/x ≥ a^2 - a) 
                (q : ∃ x : ℝ, x + |x - 1| = 2*a) : 
  a ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry

end a_range_l2050_205076


namespace sweet_potato_price_is_correct_l2050_205054

/-- The price of each sweet potato in Alice's grocery order --/
def sweet_potato_price : ℚ :=
  let minimum_spend : ℚ := 35
  let chicken_price : ℚ := 6 * (3/2)
  let lettuce_price : ℚ := 3
  let tomato_price : ℚ := 5/2
  let broccoli_price : ℚ := 2 * 2
  let sprouts_price : ℚ := 5/2
  let sweet_potato_count : ℕ := 4
  let additional_spend : ℚ := 11
  let total_without_potatoes : ℚ := chicken_price + lettuce_price + tomato_price + broccoli_price + sprouts_price
  let potato_total : ℚ := minimum_spend - additional_spend - total_without_potatoes
  potato_total / sweet_potato_count

theorem sweet_potato_price_is_correct : sweet_potato_price = 3/4 := by
  sorry

end sweet_potato_price_is_correct_l2050_205054


namespace root_in_interval_implies_m_range_l2050_205067

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem root_in_interval_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x + m = 0) → -2 ≤ m ∧ m ≤ 2 := by
  sorry

end root_in_interval_implies_m_range_l2050_205067


namespace boat_distance_problem_l2050_205097

/-- Proves that given a boat with speed 9 kmph in standing water, a stream with speed 1.5 kmph,
    and a round trip time of 24 hours, the distance to the destination is 105 km. -/
theorem boat_distance_problem (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 1.5 →
  total_time = 24 →
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time →
  distance = 105 := by
sorry


end boat_distance_problem_l2050_205097


namespace dog_grouping_combinations_l2050_205003

def total_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 5
def group3_size : ℕ := 3

def buster_in_group1 : Prop := True
def whiskers_in_group2 : Prop := True

def remaining_dogs : ℕ := total_dogs - 2
def remaining_group1 : ℕ := group1_size - 1
def remaining_group2 : ℕ := group2_size - 1

theorem dog_grouping_combinations :
  buster_in_group1 →
  whiskers_in_group2 →
  Nat.choose remaining_dogs remaining_group1 * Nat.choose (remaining_dogs - remaining_group1) remaining_group2 = 4200 := by
  sorry

end dog_grouping_combinations_l2050_205003


namespace area_of_triangle_AGE_l2050_205083

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- Point E on side BC such that BE = 2 and EC = 3 -/
def E : ℝ × ℝ := (5, 2)

/-- Point G is the second intersection of circumcircle of ABE with diagonal BD -/
def G : Square → ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AGE (s : Square) :
  triangle_area s.A (G s) E = 44.5 := by sorry

end area_of_triangle_AGE_l2050_205083


namespace fraction_simplification_l2050_205029

theorem fraction_simplification :
  1 / (1 / ((1/2)^2) + 1 / ((1/2)^3) + 1 / ((1/2)^4) + 1 / ((1/2)^5)) = 1 / 60 := by
  sorry

end fraction_simplification_l2050_205029


namespace odd_function_extension_l2050_205079

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = x^3 + x + 1) :
  ∀ x < 0, f x = x^3 + x - 1 :=
by sorry

end odd_function_extension_l2050_205079


namespace farmer_james_animals_l2050_205074

/-- Represents the number of heads for each animal type -/
def heads : Fin 3 → ℕ
  | 0 => 2  -- Hens
  | 1 => 3  -- Peacocks
  | 2 => 6  -- Zombie hens

/-- Represents the number of legs for each animal type -/
def legs : Fin 3 → ℕ
  | 0 => 8  -- Hens
  | 1 => 9  -- Peacocks
  | 2 => 12 -- Zombie hens

/-- The total number of heads on the farm -/
def total_heads : ℕ := 800

/-- The total number of legs on the farm -/
def total_legs : ℕ := 2018

/-- Calculates the total number of animals on the farm -/
def total_animals : ℕ := (total_legs - total_heads) / 6

theorem farmer_james_animals :
  total_animals = 203 ∧
  (∃ (h p z : ℕ),
    h * heads 0 + p * heads 1 + z * heads 2 = total_heads ∧
    h * legs 0 + p * legs 1 + z * legs 2 = total_legs ∧
    h + p + z = total_animals) :=
by sorry

#eval total_animals

end farmer_james_animals_l2050_205074


namespace quarter_circles_sum_limit_l2050_205036

/-- The sum of the lengths of quarter-circles approaches πC as n approaches infinity -/
theorem quarter_circles_sum_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |2 * n * (π * (C / (2 * π * n)) / 4) - π * C| < ε :=
by sorry

end quarter_circles_sum_limit_l2050_205036


namespace expenditure_recording_l2050_205002

/-- Represents the sign of a financial transaction -/
inductive TransactionSign
| Positive
| Negative

/-- Represents a financial transaction -/
structure Transaction where
  amount : ℕ
  sign : TransactionSign

/-- Records a transaction with the given amount and sign -/
def recordTransaction (amount : ℕ) (sign : TransactionSign) : Transaction :=
  { amount := amount, sign := sign }

/-- The rule for recording incomes and expenditures -/
axiom opposite_signs : 
  ∀ (income expenditure : Transaction), 
    income.sign = TransactionSign.Positive → 
    expenditure.sign = TransactionSign.Negative

/-- The main theorem -/
theorem expenditure_recording 
  (income : Transaction) 
  (h_income : income = recordTransaction 500 TransactionSign.Positive) :
  ∃ (expenditure : Transaction), 
    expenditure = recordTransaction 200 TransactionSign.Negative :=
sorry

end expenditure_recording_l2050_205002


namespace b_completes_in_20_days_l2050_205007

/-- The number of days it takes for worker A to complete the work alone -/
def days_a : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 7

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.18333333333333335

/-- The number of days it takes for worker B to complete the work alone -/
def days_b : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work in 20 days -/
theorem b_completes_in_20_days :
  (days_together * (1 / days_a + 1 / days_b) = 1 - work_left) →
  days_b = 20 := by
  sorry

end b_completes_in_20_days_l2050_205007


namespace water_transfer_difference_l2050_205084

theorem water_transfer_difference (suho_original seohyun_original : ℚ) : 
  suho_original ≥ 0 →
  seohyun_original ≥ 0 →
  (suho_original - 7/3) = (seohyun_original + 7/3 + 3/2) →
  suho_original - seohyun_original = 37/6 :=
by sorry

end water_transfer_difference_l2050_205084


namespace distribute_five_items_three_bags_l2050_205025

/-- The number of ways to distribute n distinct items into k identical bags --/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 distinct items into 3 identical bags results in 51 ways --/
theorem distribute_five_items_three_bags : distribute 5 3 = 51 := by sorry

end distribute_five_items_three_bags_l2050_205025


namespace equation_solution_l2050_205012

theorem equation_solution : ∃ x : ℝ, 3^(x - 1) = (1 : ℝ) / 9 ∧ x = -1 := by
  sorry

end equation_solution_l2050_205012


namespace marble_probability_l2050_205052

theorem marble_probability (total red blue : ℕ) (h1 : total = 20) (h2 : red = 7) (h3 : blue = 5) :
  let white := total - (red + blue)
  (red + white : ℚ) / total = 3 / 4 := by sorry

end marble_probability_l2050_205052


namespace geometric_sequence_formula_l2050_205046

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 3 = 10 →
  a 4 + a 6 = 5/4 →
  ∃ (q : ℝ), ∀ n : ℕ, a n = 2^(4-n) :=
by sorry

end geometric_sequence_formula_l2050_205046


namespace optimal_price_and_quantity_l2050_205028

/-- Represents the sales and pricing model for a product -/
structure SalesModel where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialSalesVolume : ℝ
  priceElasticity : ℝ
  targetProfit : ℝ
  maxCost : ℝ

/-- Calculates the sales volume for a given selling price -/
def salesVolume (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  model.initialSalesVolume - model.priceElasticity * (sellingPrice - model.initialSellingPrice)

/-- Calculates the profit for a given selling price -/
def profit (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  (sellingPrice - model.initialPurchasePrice) * (salesVolume model sellingPrice)

/-- Calculates the cost for a given selling price -/
def cost (model : SalesModel) (sellingPrice : ℝ) : ℝ :=
  model.initialPurchasePrice * (salesVolume model sellingPrice)

/-- Theorem stating that the optimal selling price and purchase quantity satisfy the constraints -/
theorem optimal_price_and_quantity (model : SalesModel) 
  (h_model : model = { 
    initialPurchasePrice := 40,
    initialSellingPrice := 50,
    initialSalesVolume := 500,
    priceElasticity := 10,
    targetProfit := 8000,
    maxCost := 10000
  }) :
  ∃ (optimalPrice optimalQuantity : ℝ),
    optimalPrice = 80 ∧
    optimalQuantity = 200 ∧
    profit model optimalPrice = model.targetProfit ∧
    cost model optimalPrice ≤ model.maxCost :=
  sorry

end optimal_price_and_quantity_l2050_205028


namespace cost_of_roses_shoes_l2050_205090

/-- The cost of Rose's shoes given Mary and Rose's shopping details -/
theorem cost_of_roses_shoes 
  (mary_rose_total : ℝ → ℝ → Prop)  -- Mary and Rose spent the same total amount
  (mary_sunglasses_cost : ℝ)        -- Cost of each pair of Mary's sunglasses
  (mary_sunglasses_quantity : ℕ)    -- Number of pairs of sunglasses Mary bought
  (mary_jeans_cost : ℝ)             -- Cost of Mary's jeans
  (rose_cards_cost : ℝ)             -- Cost of each deck of Rose's basketball cards
  (rose_cards_quantity : ℕ)         -- Number of decks of basketball cards Rose bought
  (h1 : mary_sunglasses_cost = 50)
  (h2 : mary_sunglasses_quantity = 2)
  (h3 : mary_jeans_cost = 100)
  (h4 : rose_cards_cost = 25)
  (h5 : rose_cards_quantity = 2)
  (h6 : mary_rose_total (mary_sunglasses_cost * mary_sunglasses_quantity + mary_jeans_cost) 
                        (rose_cards_cost * rose_cards_quantity + rose_shoes_cost))
  : rose_shoes_cost = 150 := by
  sorry


end cost_of_roses_shoes_l2050_205090


namespace gold_pucks_count_gold_pucks_theorem_l2050_205095

theorem gold_pucks_count : ℕ → Prop :=
  fun total_gold : ℕ =>
    ∃ (pucks_per_box : ℕ),
      -- Each box has the same number of pucks
      3 * pucks_per_box = 40 + total_gold ∧
      -- One box contains all black pucks and 1/7 of gold pucks
      pucks_per_box = 40 + total_gold / 7 ∧
      -- The number of gold pucks is 140
      total_gold = 140

-- The proof of the theorem
theorem gold_pucks_theorem : gold_pucks_count 140 := by
  sorry

end gold_pucks_count_gold_pucks_theorem_l2050_205095


namespace B_3_2_eq_4_l2050_205086

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_3_2_eq_4 : B 3 2 = 4 := by sorry

end B_3_2_eq_4_l2050_205086


namespace max_value_fraction_l2050_205080

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (2*a + b)^2 = 1 + 6*a*b) :
  (a * b) / (2*a + b + 1) ≤ 1/6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (2*a₀ + b₀)^2 = 1 + 6*a₀*b₀ ∧ (a₀ * b₀) / (2*a₀ + b₀ + 1) = 1/6 :=
sorry

end max_value_fraction_l2050_205080


namespace gas_price_calculation_l2050_205001

theorem gas_price_calculation (expected_cash : ℝ) : 
  (12 * (expected_cash / 12) = 10 * (expected_cash / 12 + 0.3)) →
  expected_cash / 12 + 0.3 = 1.8 := by
  sorry

end gas_price_calculation_l2050_205001


namespace whole_number_between_bounds_l2050_205009

theorem whole_number_between_bounds (N : ℕ) (h : 7.5 < (N : ℝ) / 3 ∧ (N : ℝ) / 3 < 8) : N = 23 := by
  sorry

end whole_number_between_bounds_l2050_205009


namespace inequality_problem_l2050_205031

theorem inequality_problem (x y z : ℝ) (a : ℝ) : 
  (x^2 + y^2 + z^2 = 1) → 
  ((-3 : ℝ) ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3) ∧ 
  ((∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) ↔ 
   (a ≥ 4 ∨ a ≤ 0)) :=
by sorry

end inequality_problem_l2050_205031


namespace problem_1_problem_2_l2050_205085

-- Problem 1
theorem problem_1 : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (30 * π / 180) + (-1/2)⁻¹ = Real.sqrt 3 - 3 :=
sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, x * (x + 6) = -5 ↔ x = -5 ∨ x = -1 :=
sorry

end problem_1_problem_2_l2050_205085


namespace subset_necessary_not_sufficient_l2050_205048

def A (a : ℕ) : Set ℕ := {1, a}
def B : Set ℕ := {1, 2, 3}

theorem subset_necessary_not_sufficient (a : ℕ) :
  (A a ⊆ B ↔ a = 3) ↔ False ∧
  (a = 3 → A a ⊆ B) ∧
  ¬(A a ⊆ B → a = 3) :=
sorry

end subset_necessary_not_sufficient_l2050_205048


namespace investment_profit_ratio_l2050_205075

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℚ
  time : ℕ

/-- Calculates the profit ratio of two partners -/
def profitRatio (p q : Partner) : ℚ × ℚ :=
  let pProfit := p.investment * p.time
  let qProfit := q.investment * q.time
  (pProfit, qProfit)

theorem investment_profit_ratio :
  let p : Partner := ⟨7, 5⟩
  let q : Partner := ⟨5, 14⟩
  profitRatio p q = (1, 2) := by
  sorry

end investment_profit_ratio_l2050_205075


namespace ellipse_sum_l2050_205042

/-- The sum of h, k, a, and b for a specific ellipse -/
theorem ellipse_sum (h k a b : ℝ) : 
  ((3 : ℝ) = h) → ((-5 : ℝ) = k) → ((7 : ℝ) = a) → ((2 : ℝ) = b) → 
  h + k + a + b = 7 := by
  sorry

end ellipse_sum_l2050_205042


namespace f_composition_of_one_l2050_205088

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_one (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x + 2) : f (f (f 1)) = 53 := by
  sorry

end f_composition_of_one_l2050_205088


namespace range_of_m_l2050_205005

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 := by sorry

end range_of_m_l2050_205005


namespace six_power_plus_one_same_digits_l2050_205030

def has_same_digits (m : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, (m / 10^k) % 10 = d

theorem six_power_plus_one_same_digits :
  {n : ℕ | n > 0 ∧ has_same_digits (6^n + 1)} = {1, 5} := by sorry

end six_power_plus_one_same_digits_l2050_205030


namespace sqrt_x_minus_3_real_l2050_205011

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) → x ≥ 3 := by
  sorry

end sqrt_x_minus_3_real_l2050_205011


namespace profit_percentage_is_20_percent_l2050_205063

def selling_price : ℝ := 250
def cost_price : ℝ := 208.33

theorem profit_percentage_is_20_percent :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end profit_percentage_is_20_percent_l2050_205063


namespace inequality_to_interval_l2050_205096

theorem inequality_to_interval : 
  {x : ℝ | -8 ≤ x ∧ x < 15} = Set.Icc (-8) 15 := by sorry

end inequality_to_interval_l2050_205096


namespace parabola_directrix_l2050_205070

/-- Given a parabola with equation x = (1/8)y^2, its directrix has equation x = -2 -/
theorem parabola_directrix (x y : ℝ) :
  (x = (1/8) * y^2) → (∃ (p : ℝ), p > 0 ∧ x = (1/(4*p)) * y^2 ∧ -p = -2) :=
by sorry

end parabola_directrix_l2050_205070


namespace number_puzzle_l2050_205062

theorem number_puzzle (x : ℝ) : 0.5 * x = 0.25 * x + 2 → x = 8 := by
  sorry

end number_puzzle_l2050_205062


namespace complex_sum_power_l2050_205015

theorem complex_sum_power (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) :
  (x / (x + y))^2013 + (y / (x + y))^2013 = -2 := by
  sorry

end complex_sum_power_l2050_205015


namespace hundredths_place_of_seven_twentieths_l2050_205035

theorem hundredths_place_of_seven_twentieths (x : ℚ) : 
  x = 7 / 20 → (x * 100).floor % 10 = 5 := by
  sorry

end hundredths_place_of_seven_twentieths_l2050_205035


namespace apex_high_debate_points_l2050_205026

theorem apex_high_debate_points :
  ∀ (total_points : ℚ),
  total_points > 0 →
  ∃ (remaining_points : ℕ),
  (1/5 : ℚ) * total_points + (1/3 : ℚ) * total_points + 12 + remaining_points = total_points ∧
  remaining_points ≤ 18 ∧
  remaining_points = 18 :=
by sorry

end apex_high_debate_points_l2050_205026


namespace alton_daily_earnings_l2050_205040

/-- Calculates daily earnings given weekly rent, weekly profit, and number of workdays --/
def daily_earnings (weekly_rent : ℚ) (weekly_profit : ℚ) (workdays : ℕ) : ℚ :=
  (weekly_rent + weekly_profit) / workdays

/-- Proves that given the specified conditions, daily earnings are $11.20 --/
theorem alton_daily_earnings :
  let weekly_rent : ℚ := 20
  let weekly_profit : ℚ := 36
  let workdays : ℕ := 5
  daily_earnings weekly_rent weekly_profit workdays = 11.2 := by
sorry

end alton_daily_earnings_l2050_205040


namespace fuel_mixture_problem_l2050_205013

/-- Proves that the volume of fuel A added is 82 gallons given the specified conditions -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_A : ℝ) (ethanol_B : ℝ) (total_ethanol : ℝ) 
  (h1 : tank_capacity = 208)
  (h2 : ethanol_A = 0.12)
  (h3 : ethanol_B = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_A : ℝ), fuel_A = 82 ∧ 
  ethanol_A * fuel_A + ethanol_B * (tank_capacity - fuel_A) = total_ethanol :=
by sorry

end fuel_mixture_problem_l2050_205013


namespace min_value_sequence_l2050_205024

theorem min_value_sequence (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∀ k, ∃ r, a (k + 1) = a k + r) →  -- Arithmetic progression
  (∀ k, ∃ q, a (k + 1) = a k * q) →  -- Geometric progression
  (a 7 = a 6 + 2 * a 5) →  -- Given condition
  (Real.sqrt (a m * a n) = 4 * a 1) →  -- Given condition
  (∃ min_val : ℝ, min_val = 1 + Real.sqrt 5 / 3 ∧
    ∀ p q : ℕ, 1 / p + 5 / q ≥ min_val) :=
by sorry

end min_value_sequence_l2050_205024


namespace color_film_fraction_l2050_205050

/-- Given a film festival selection process, prove the fraction of color films in the selection. -/
theorem color_film_fraction (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw : ℚ := 40 * x
  let total_color : ℚ := 10 * y
  let selected_bw : ℚ := (y / x) * (total_bw / 100)
  let selected_color : ℚ := total_color
  let total_selected : ℚ := selected_bw + selected_color
  (selected_color / total_selected) = 25 / 26 := by
  sorry

end color_film_fraction_l2050_205050


namespace arithmetic_sequence_common_difference_l2050_205047

/-- 
Given an arithmetic sequence with:
- 20 terms
- First term is 4
- Sum of the sequence is 650

Prove that the common difference is 3
-/
theorem arithmetic_sequence_common_difference :
  ∀ (d : ℚ),
  (20 : ℚ) / 2 * (2 * 4 + (20 - 1) * d) = 650 →
  d = 3 := by
  sorry

end arithmetic_sequence_common_difference_l2050_205047


namespace quadratic_coefficient_l2050_205064

/-- A quadratic function with vertex at (-3, 2) passing through (2, -43) has a = -9/5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x + 3)^2 + 2) → 
  (a * 2^2 + b * 2 + c = -43) →
  a = -9/5 := by
  sorry

end quadratic_coefficient_l2050_205064


namespace field_dimensions_l2050_205081

theorem field_dimensions (m : ℝ) : (3*m + 11) * m = 100 → m = 5 := by
  sorry

end field_dimensions_l2050_205081


namespace family_size_l2050_205087

theorem family_size (purification_cost : ℚ) (water_per_person : ℚ) (family_cost : ℚ) :
  purification_cost = 1 →
  water_per_person = 1/2 →
  family_cost = 3 →
  (family_cost / (purification_cost * water_per_person) : ℚ) = 6 :=
by sorry

end family_size_l2050_205087


namespace cos_seven_pi_four_l2050_205034

theorem cos_seven_pi_four : Real.cos (7 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_seven_pi_four_l2050_205034


namespace max_shelves_with_five_books_together_l2050_205018

/-- Given 1300 books and k shelves, this theorem states that 18 is the largest value of k
    for which there will always be at least 5 books on the same shelf
    before and after any rearrangement. -/
theorem max_shelves_with_five_books_together (k : ℕ) : 
  (∀ (arrangement₁ arrangement₂ : Fin k → Fin 1300 → Prop), 
    (∀ b, ∃! s, arrangement₁ s b) → 
    (∀ b, ∃! s, arrangement₂ s b) → 
    (∃ s : Fin k, ∃ (books : Finset (Fin 1300)), 
      books.card = 5 ∧ 
      (∀ b ∈ books, arrangement₁ s b ∧ arrangement₂ s b))) ↔ 
  k ≤ 18 :=
sorry

end max_shelves_with_five_books_together_l2050_205018


namespace freds_allowance_l2050_205022

/-- Proves that Fred's weekly allowance is 16 dollars given the problem conditions -/
theorem freds_allowance (spent_on_movies : ℝ) (car_wash_earnings : ℝ) (final_amount : ℝ) :
  spent_on_movies = car_wash_earnings - 6 →
  final_amount = 14 →
  spent_on_movies = 8 →
  spent_on_movies * 2 = 16 :=
by
  sorry

#check freds_allowance

end freds_allowance_l2050_205022


namespace geometric_sum_base_case_l2050_205041

theorem geometric_sum_base_case (a : ℝ) (h : a ≠ 1) :
  1 + a = (1 - a^2) / (1 - a) := by sorry

end geometric_sum_base_case_l2050_205041


namespace car_speed_first_hour_l2050_205017

/-- Proves that the speed of a car in the first hour is 60 km/h given the conditions -/
theorem car_speed_first_hour 
  (x : ℝ) -- Speed in the first hour
  (h1 : x > 0) -- Assuming speed is positive
  (h2 : (x + 30) / 2 = 45) -- Average speed equation
  : x = 60 := by
  sorry

end car_speed_first_hour_l2050_205017


namespace inequality_proof_l2050_205089

theorem inequality_proof (x m : ℝ) (a b c : ℝ) :
  (∀ x, |x - 3| + |x - m| ≥ 2*m) →
  a > 0 → b > 0 → c > 0 → a + b + c = 1 →
  (∃ m_max : ℝ, m_max = 1 ∧ 
    (∀ m', (∀ x, |x - 3| + |x - m'| ≥ 2*m') → m' ≤ m_max)) ∧
  (4*a^2 + 9*b^2 + c^2 ≥ 36/49) ∧
  (4*a^2 + 9*b^2 + c^2 = 36/49 ↔ a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
by sorry


end inequality_proof_l2050_205089


namespace arithmetic_calculations_l2050_205077

theorem arithmetic_calculations : 
  (1 * (-30) - 4 * (-4) = -14) ∧ 
  ((-2)^2 - (1/7) * (-3-4) = 5) := by
sorry

end arithmetic_calculations_l2050_205077


namespace sqrt_5_greater_than_2_l2050_205068

theorem sqrt_5_greater_than_2 : Real.sqrt 5 > 2 := by
  sorry

end sqrt_5_greater_than_2_l2050_205068


namespace base_8_subtraction_example_l2050_205055

/-- Subtraction in base 8 -/
def base_8_subtraction (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 8 -/
def to_base_8 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 8 to base 10 -/
def from_base_8 (n : ℕ) : ℕ :=
  sorry

theorem base_8_subtraction_example :
  base_8_subtraction (from_base_8 7463) (from_base_8 3154) = from_base_8 4317 :=
sorry

end base_8_subtraction_example_l2050_205055


namespace seminar_attendance_l2050_205059

/-- The total number of people who attended the seminars given the attendance for math and music seminars -/
theorem seminar_attendance (math_attendees music_attendees both_attendees : ℕ) 
  (h1 : math_attendees = 75)
  (h2 : music_attendees = 61)
  (h3 : both_attendees = 12) :
  math_attendees + music_attendees - both_attendees = 124 := by
  sorry

#check seminar_attendance

end seminar_attendance_l2050_205059


namespace rope_length_proof_l2050_205037

theorem rope_length_proof (r : ℝ) : 
  r > 0 → 
  π * 20^2 - π * r^2 = 942.8571428571429 → 
  r = 10 := by
sorry

end rope_length_proof_l2050_205037


namespace smallest_positive_integer_e_l2050_205073

theorem smallest_positive_integer_e (a b c d e : ℤ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = -3 ∨ x = 7 ∨ x = 11 ∨ x = -1/4) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 ∧ 
    (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ x = -3 ∨ x = 7 ∨ x = 11 ∨ x = -1/4) →
    e' ≥ e) →
  e = 231 :=
by sorry

end smallest_positive_integer_e_l2050_205073


namespace arithmetic_sequence_sum_l2050_205082

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 - a 5 + a 15 = 20 →
  a 3 + a 19 = 40 := by
  sorry

end arithmetic_sequence_sum_l2050_205082


namespace f_range_l2050_205051

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.Icc (-2 : ℝ) 2, ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end f_range_l2050_205051


namespace yellow_balls_count_l2050_205056

theorem yellow_balls_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob : ℚ) : 
  red = 10 →
  yellow + red = total →
  prob = 2 / 5 →
  (red : ℚ) / total = prob →
  yellow = 15 :=
by
  sorry

end yellow_balls_count_l2050_205056


namespace quadratic_points_relationship_l2050_205004

/-- A quadratic function f(x) = -x² + 2x + c --/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- The y-coordinate of a point (x, f(x)) on the graph of f --/
def y (c : ℝ) (x : ℝ) : ℝ := f c x

theorem quadratic_points_relationship (c : ℝ) :
  let y₁ := y c (-1)
  let y₂ := y c 3
  let y₃ := y c 5
  y₁ = y₂ ∧ y₂ > y₃ := by sorry

end quadratic_points_relationship_l2050_205004


namespace min_cost_all_B_trucks_l2050_205058

-- Define the capacities of trucks A and B
def truck_A_capacity : ℝ := 5
def truck_B_capacity : ℝ := 3

-- Define the cost per ton for trucks A and B
def cost_per_ton_A : ℝ := 100
def cost_per_ton_B : ℝ := 150

-- Define the total number of trucks
def total_trucks : ℕ := 5

-- Define the cost function
def cost_function (a : ℝ) : ℝ := 50 * a + 2250

-- Theorem statement
theorem min_cost_all_B_trucks :
  ∀ a : ℝ, 0 ≤ a ∧ a ≤ total_trucks →
  cost_function 0 ≤ cost_function a :=
by sorry

end min_cost_all_B_trucks_l2050_205058


namespace race_orders_theorem_l2050_205033

-- Define the number of racers
def num_racers : ℕ := 6

-- Define the function to calculate the number of possible orders
def possible_orders (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem race_orders_theorem : possible_orders num_racers = 720 := by
  sorry

end race_orders_theorem_l2050_205033


namespace residue_of_seven_power_l2050_205023

theorem residue_of_seven_power (n : ℕ) : 7^1234 ≡ 4 [ZMOD 13] := by
  sorry

end residue_of_seven_power_l2050_205023


namespace no_perfect_square_pairs_l2050_205014

theorem no_perfect_square_pairs : ¬∃ (x y : ℕ+), ∃ (z : ℕ+), (x * y + 1) * (x * y + x + 2) = z ^ 2 := by
  sorry

end no_perfect_square_pairs_l2050_205014


namespace total_situps_is_510_l2050_205078

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps : ℕ := 45

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ :=
  barney_situps * barney_minutes +
  carrie_situps * carrie_minutes +
  jerrie_situps * jerrie_minutes

/-- Theorem stating that the total number of sit-ups is 510 -/
theorem total_situps_is_510 : total_situps = 510 := by
  sorry

end total_situps_is_510_l2050_205078


namespace parts_per_day_to_finish_ahead_l2050_205021

theorem parts_per_day_to_finish_ahead (total_parts : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_parts_per_day : ℕ) :
  total_parts = 408 →
  total_days = 15 →
  initial_days = 3 →
  initial_parts_per_day = 24 →
  ∃ (x : ℕ), x = 29 ∧ 
    (initial_days * initial_parts_per_day + (total_days - initial_days) * x > total_parts) ∧
    ∀ (y : ℕ), y < x → (initial_days * initial_parts_per_day + (total_days - initial_days) * y ≤ total_parts) :=
by sorry

end parts_per_day_to_finish_ahead_l2050_205021


namespace line_slope_intercept_sum_l2050_205099

/-- Given a line passing through points (1, -3) and (-1, 3), 
    prove that the sum of its slope and y-intercept is -3 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → 
    ((x = 1 ∧ y = -3) ∨ (x = -1 ∧ y = 3))) → 
  m + b = -3 :=
by sorry

end line_slope_intercept_sum_l2050_205099


namespace mechanic_work_hours_l2050_205057

theorem mechanic_work_hours (rate1 rate2 total_hours total_charge : ℕ) 
  (h1 : rate1 = 45)
  (h2 : rate2 = 85)
  (h3 : total_hours = 20)
  (h4 : total_charge = 1100) :
  ∃ (hours1 hours2 : ℕ), 
    hours1 + hours2 = total_hours ∧ 
    rate1 * hours1 + rate2 * hours2 = total_charge ∧
    hours2 = 5 := by
  sorry

end mechanic_work_hours_l2050_205057


namespace bug_travel_distance_l2050_205093

theorem bug_travel_distance (r : ℝ) (s : ℝ) (h1 : r = 65) (h2 : s = 100) :
  let d := 2 * r
  let x := Real.sqrt (d^2 - s^2)
  d + s + x = 313 :=
by sorry

end bug_travel_distance_l2050_205093


namespace face_card_proportion_l2050_205044

theorem face_card_proportion (p : ℝ) : 
  (p ≥ 0) → (p ≤ 1) → (1 - (1 - p)^3 = 19/27) → p = 1/3 := by
sorry

end face_card_proportion_l2050_205044


namespace matrix_cube_proof_l2050_205066

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_cube_proof : A ^ 3 = !![(-4), 2; (-2), 1] := by
  sorry

end matrix_cube_proof_l2050_205066


namespace squirrel_nut_difference_example_l2050_205065

/-- Given a tree with squirrels and nuts, calculate the difference between their quantities -/
def squirrel_nut_difference (num_squirrels num_nuts : ℕ) : ℤ :=
  (num_squirrels : ℤ) - (num_nuts : ℤ)

/-- Theorem: In a tree with 4 squirrels and 2 nuts, the difference between
    the number of squirrels and nuts is 2 -/
theorem squirrel_nut_difference_example : squirrel_nut_difference 4 2 = 2 := by
  sorry

end squirrel_nut_difference_example_l2050_205065


namespace lego_pieces_sold_l2050_205020

/-- The number of single Lego pieces sold -/
def single_pieces : ℕ := sorry

/-- The total earnings in cents -/
def total_earnings : ℕ := 1000

/-- The number of double pieces sold -/
def double_pieces : ℕ := 45

/-- The number of triple pieces sold -/
def triple_pieces : ℕ := 50

/-- The number of quadruple pieces sold -/
def quadruple_pieces : ℕ := 165

/-- The cost of each circle in cents -/
def circle_cost : ℕ := 1

theorem lego_pieces_sold :
  single_pieces = 100 :=
by sorry

end lego_pieces_sold_l2050_205020


namespace molecular_weight_3_moles_CaOH2_l2050_205019

/-- Atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Number of Calcium atoms in Ca(OH)2 -/
def num_Ca : ℕ := 1

/-- Number of Oxygen atoms in Ca(OH)2 -/
def num_O : ℕ := 2

/-- Number of Hydrogen atoms in Ca(OH)2 -/
def num_H : ℕ := 2

/-- Number of moles of Ca(OH)2 -/
def num_moles : ℝ := 3

/-- Molecular weight of Ca(OH)2 in g/mol -/
def molecular_weight_CaOH2 : ℝ :=
  num_Ca * atomic_weight_Ca + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_3_moles_CaOH2 :
  num_moles * molecular_weight_CaOH2 = 222.30 := by
  sorry

end molecular_weight_3_moles_CaOH2_l2050_205019


namespace initial_stock_calculation_l2050_205061

theorem initial_stock_calculation (sold : ℕ) (unsold_percentage : ℚ) 
  (h1 : sold = 402)
  (h2 : unsold_percentage = 665/1000) : 
  ∃ initial_stock : ℕ, 
    initial_stock = 1200 ∧ 
    (1 - unsold_percentage) * initial_stock = sold :=
by sorry

end initial_stock_calculation_l2050_205061


namespace largest_power_dividing_factorial_l2050_205092

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem largest_power_dividing_factorial :
  (∀ m : ℕ, m > 7 → ¬(divides (18^m) (factorial 30))) ∧
  (divides (18^7) (factorial 30)) := by
sorry

end largest_power_dividing_factorial_l2050_205092


namespace mcnugget_theorem_l2050_205060

/-- Represents the possible package sizes for Chicken McNuggets -/
def nugget_sizes : List ℕ := [6, 9, 20]

/-- Checks if a number can be expressed as a combination of nugget sizes -/
def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

/-- The largest number that cannot be ordered -/
def largest_unorderable : ℕ := 43

/-- Main theorem: 43 is the largest number that cannot be ordered -/
theorem mcnugget_theorem :
  (∀ m > largest_unorderable, is_orderable m) ∧
  ¬(is_orderable largest_unorderable) :=
sorry

end mcnugget_theorem_l2050_205060


namespace plot_area_approx_360_l2050_205016

/-- Calculates the area of a rectangular plot given its breadth, where the length is 25% less than the breadth -/
def plot_area (breadth : ℝ) : ℝ :=
  let length := 0.75 * breadth
  length * breadth

/-- The breadth of the plot -/
def plot_breadth : ℝ := 21.908902300206645

/-- Theorem stating that the area of the plot is approximately 360 square meters -/
theorem plot_area_approx_360 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |plot_area plot_breadth - 360| < ε :=
sorry

end plot_area_approx_360_l2050_205016


namespace amount_calculation_l2050_205000

theorem amount_calculation (x : ℝ) (amount : ℝ) (h1 : x = 25.0) (h2 : 2 * x = 3 * x - amount) : amount = 25.0 := by
  sorry

end amount_calculation_l2050_205000


namespace max_non_intersecting_points_l2050_205072

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A function that checks if a broken line formed by a list of points intersects itself -/
def is_self_intersecting (points : List Point) : Prop :=
  sorry

/-- The property that any permutation of points forms a non-self-intersecting broken line -/
def all_permutations_non_intersecting (points : List Point) : Prop :=
  ∀ perm : List Point, perm.Perm points → ¬(is_self_intersecting perm)

theorem max_non_intersecting_points :
  ∃ (points : List Point),
    points.length = 4 ∧
    all_permutations_non_intersecting points ∧
    ∀ (larger_set : List Point),
      larger_set.length > 4 →
      ¬(all_permutations_non_intersecting larger_set) :=
sorry

end max_non_intersecting_points_l2050_205072


namespace mollys_current_age_l2050_205049

/-- Represents the ages of Sandy and Molly -/
structure Ages where
  sandy : ℕ
  molly : ℕ

/-- The ratio of Sandy's age to Molly's age is 4:3 -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.molly = 3 * ages.sandy

/-- Sandy will be 42 years old in 6 years -/
def sandy_future_age (ages : Ages) : Prop :=
  ages.sandy + 6 = 42

theorem mollys_current_age (ages : Ages) :
  age_ratio ages → sandy_future_age ages → ages.molly = 27 := by
  sorry

end mollys_current_age_l2050_205049


namespace total_distance_traveled_l2050_205039

def trip_duration : ℕ := 12
def speed1 : ℕ := 70
def time1 : ℕ := 3
def speed2 : ℕ := 80
def time2 : ℕ := 4
def speed3 : ℕ := 65
def time3 : ℕ := 3
def speed4 : ℕ := 90
def time4 : ℕ := 2

theorem total_distance_traveled :
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4 = 905 :=
by
  sorry

#check total_distance_traveled

end total_distance_traveled_l2050_205039


namespace candies_remaining_is_155_l2050_205010

/-- The number of candies remaining after Carlos ate his share -/
def candies_remaining : ℕ :=
  let red : ℕ := 60
  let yellow : ℕ := 3 * red - 30
  let blue : ℕ := (2 * yellow) / 4
  let green : ℕ := 40
  let purple : ℕ := green / 3
  let silver : ℕ := 15
  let gold : ℕ := silver / 2
  let total : ℕ := red + yellow + blue + green + purple + silver + gold
  let eaten : ℕ := yellow + (green * 3 / 4) + (blue / 3)
  total - eaten

theorem candies_remaining_is_155 : candies_remaining = 155 := by
  sorry

end candies_remaining_is_155_l2050_205010


namespace negation_of_proposition_l2050_205091

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0)) ↔ 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 - 1 ≤ 0) :=
by sorry

end negation_of_proposition_l2050_205091


namespace brandy_safe_caffeine_l2050_205094

/-- The maximum safe amount of caffeine that can be consumed per day (in mg) -/
def max_safe_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink (in mg) -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumed -/
def drinks_consumed : ℕ := 4

/-- The remaining amount of caffeine Brandy can safely consume (in mg) -/
def remaining_safe_caffeine : ℕ := max_safe_caffeine - (caffeine_per_drink * drinks_consumed)

theorem brandy_safe_caffeine : remaining_safe_caffeine = 20 := by
  sorry

end brandy_safe_caffeine_l2050_205094


namespace sqrt_a_sqrt_a_eq_a_pow_three_fourths_l2050_205027

theorem sqrt_a_sqrt_a_eq_a_pow_three_fourths (a : ℝ) (h : a > 0) :
  Real.sqrt (a * Real.sqrt a) = a ^ (3/4) := by
  sorry

end sqrt_a_sqrt_a_eq_a_pow_three_fourths_l2050_205027


namespace chocolate_bar_count_l2050_205008

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 17

/-- The number of chocolate bars in each small box -/
def choc_per_small_box : ℕ := 26

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * choc_per_small_box

theorem chocolate_bar_count : total_chocolate_bars = 442 := by
  sorry

end chocolate_bar_count_l2050_205008


namespace circle_center_and_radius_l2050_205006

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧
    radius = Real.sqrt 5 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l2050_205006


namespace herb_count_at_spring_end_l2050_205098

def spring_duration : ℕ := 6

def initial_basil : ℕ := 3
def initial_parsley : ℕ := 1
def initial_mint : ℕ := 2
def initial_rosemary : ℕ := 1
def initial_thyme : ℕ := 1

def basil_growth_rate : ℕ → ℕ := λ weeks => 2^(weeks / 2)
def parsley_growth_rate : ℕ → ℕ := λ weeks => weeks
def mint_growth_rate : ℕ → ℕ := λ weeks => 3^(weeks / 4)

def extra_basil_week : ℕ := 3
def mint_stop_week : ℕ := 3
def parsley_loss_week : ℕ := 5
def parsley_loss_amount : ℕ := 2

def final_basil_count : ℕ := initial_basil * basil_growth_rate spring_duration + 1
def final_parsley_count : ℕ := initial_parsley + parsley_growth_rate spring_duration - parsley_loss_amount
def final_mint_count : ℕ := initial_mint * mint_growth_rate mint_stop_week
def final_rosemary_count : ℕ := initial_rosemary
def final_thyme_count : ℕ := initial_thyme

theorem herb_count_at_spring_end :
  final_basil_count + final_parsley_count + final_mint_count + 
  final_rosemary_count + final_thyme_count = 35 := by
  sorry

end herb_count_at_spring_end_l2050_205098


namespace flower_stitches_l2050_205069

/-- Proves that given the conditions, the number of stitches required to embroider one flower is 60. -/
theorem flower_stitches (
  stitches_per_minute : ℕ)
  (unicorn_stitches : ℕ)
  (godzilla_stitches : ℕ)
  (num_unicorns : ℕ)
  (num_flowers : ℕ)
  (total_minutes : ℕ)
  (h1 : stitches_per_minute = 4)
  (h2 : unicorn_stitches = 180)
  (h3 : godzilla_stitches = 800)
  (h4 : num_unicorns = 3)
  (h5 : num_flowers = 50)
  (h6 : total_minutes = 1085)
  : (total_minutes * stitches_per_minute - (num_unicorns * unicorn_stitches + godzilla_stitches)) / num_flowers = 60 :=
sorry

end flower_stitches_l2050_205069


namespace special_polynomial_f_one_l2050_205038

/-- A polynomial function satisfying a specific equation -/
def SpecialPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧
    (∀ x : ℝ, x ≠ 0 → f (x - 1) + f x + f (x + 1) = (f x)^2 / (2027 * x))

/-- The theorem stating that for a special polynomial, f(1) must equal 6081 -/
theorem special_polynomial_f_one (f : ℝ → ℝ) (hf : SpecialPolynomial f) : f 1 = 6081 := by
  sorry

end special_polynomial_f_one_l2050_205038


namespace subtract_fractions_l2050_205053

theorem subtract_fractions : (7 : ℚ) / 9 - (5 : ℚ) / 6 = (-1 : ℚ) / 18 := by
  sorry

end subtract_fractions_l2050_205053


namespace jimin_weight_l2050_205032

theorem jimin_weight (T J : ℝ) (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 := by
  sorry

end jimin_weight_l2050_205032


namespace total_payment_is_195_l2050_205071

def monthly_rate : ℝ := 50

def discount_rate (month : ℕ) : ℝ :=
  match month with
  | 1 => 0.05
  | 2 => 0.07
  | 3 => 0.10
  | 4 => 0.12
  | _ => 0

def late_fee_rate (month : ℕ) : ℝ :=
  match month with
  | 1 => 0.03
  | 2 => 0.02
  | 3 => 0.04
  | 4 => 0.03
  | _ => 0

def payment_amount (month : ℕ) (on_time : Bool) : ℝ :=
  if on_time then
    monthly_rate * (1 - discount_rate month)
  else
    monthly_rate * (1 + late_fee_rate month)

def total_payment : ℝ :=
  payment_amount 1 true +
  payment_amount 2 false +
  payment_amount 3 true +
  payment_amount 4 false

theorem total_payment_is_195 : total_payment = 195 := by
  sorry

end total_payment_is_195_l2050_205071


namespace inequality_proof_l2050_205043

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : 1 / a + 1 / b = 1) (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l2050_205043
