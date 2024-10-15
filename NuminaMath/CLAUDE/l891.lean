import Mathlib

namespace NUMINAMATH_CALUDE_bruce_mangoes_l891_89178

/-- Calculates the quantity of mangoes purchased given the total amount paid,
    the quantity and price of grapes, and the price of mangoes. -/
def mangoes_purchased (total_paid : ℕ) (grape_qty : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_paid - grape_qty * grape_price) / mango_price : ℕ)

/-- Proves that Bruce purchased 9 kg of mangoes given the problem conditions. -/
theorem bruce_mangoes :
  mangoes_purchased 1055 8 70 55 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bruce_mangoes_l891_89178


namespace NUMINAMATH_CALUDE_curve_M_properties_l891_89199

-- Define the curve M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^(1/2) + p.2^(1/2) = 1}

-- Theorem statement
theorem curve_M_properties :
  (∃ (p : ℝ × ℝ), p ∈ M ∧ Real.sqrt (p.1^2 + p.2^2) < Real.sqrt 2 / 2) ∧
  (∀ (S : Set (ℝ × ℝ)), S ⊆ M → MeasureTheory.volume S ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_curve_M_properties_l891_89199


namespace NUMINAMATH_CALUDE_reduced_oil_price_l891_89147

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  price_reduction_percent : ℝ
  additional_amount : ℝ
  fixed_cost : ℝ

/-- Theorem stating the reduced price of oil given the conditions --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction)
  (h1 : scenario.price_reduction_percent = 20)
  (h2 : scenario.additional_amount = 4)
  (h3 : scenario.fixed_cost = 600)
  (h4 : scenario.reduced_price = scenario.original_price * (1 - scenario.price_reduction_percent / 100))
  (h5 : scenario.fixed_cost = (scenario.fixed_cost / scenario.original_price + scenario.additional_amount) * scenario.reduced_price) :
  scenario.reduced_price = 30 := by
  sorry

#check reduced_oil_price

end NUMINAMATH_CALUDE_reduced_oil_price_l891_89147


namespace NUMINAMATH_CALUDE_afternoon_shells_l891_89124

theorem afternoon_shells (morning_shells : ℕ) (total_shells : ℕ) 
  (h1 : morning_shells = 292) 
  (h2 : total_shells = 616) : 
  total_shells - morning_shells = 324 := by
sorry

end NUMINAMATH_CALUDE_afternoon_shells_l891_89124


namespace NUMINAMATH_CALUDE_line_symmetry_theorem_l891_89129

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Two lines are symmetric about the y-axis -/
def symmetric_about_y_axis (l₁ l₂ : Line) : Prop :=
  l₁.slope = -l₂.slope ∧ l₁.intercept = l₂.intercept

theorem line_symmetry_theorem (b : ℝ) :
  let l₁ : Line := { slope := -2, intercept := b }
  let l₂ : Line := { slope := 2, intercept := 4 }
  symmetric_about_y_axis l₁ l₂ ∧ l₂.contains 1 6 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_theorem_l891_89129


namespace NUMINAMATH_CALUDE_second_day_charge_l891_89125

theorem second_day_charge (day1_charge : ℝ) (day3_charge : ℝ) (attendance_ratio : Fin 3 → ℝ) (average_charge : ℝ) :
  day1_charge = 15 →
  day3_charge = 2.5 →
  attendance_ratio 0 = 2 →
  attendance_ratio 1 = 5 →
  attendance_ratio 2 = 13 →
  average_charge = 5 →
  ∃ day2_charge : ℝ,
    day2_charge = 7.5 ∧
    average_charge * (attendance_ratio 0 + attendance_ratio 1 + attendance_ratio 2) =
      day1_charge * attendance_ratio 0 + day2_charge * attendance_ratio 1 + day3_charge * attendance_ratio 2 :=
by
  sorry


end NUMINAMATH_CALUDE_second_day_charge_l891_89125


namespace NUMINAMATH_CALUDE_debt_equality_time_l891_89161

/-- The number of days for two debts to become equal -/
def daysUntilEqualDebt (initialDebt1 initialDebt2 interestRate1 interestRate2 : ℚ) : ℚ :=
  (initialDebt2 - initialDebt1) / (initialDebt1 * interestRate1 - initialDebt2 * interestRate2)

/-- Theorem: Darren and Fergie's debts will be equal after 25 days -/
theorem debt_equality_time : 
  daysUntilEqualDebt 200 300 (8/100) (4/100) = 25 := by sorry

end NUMINAMATH_CALUDE_debt_equality_time_l891_89161


namespace NUMINAMATH_CALUDE_factorial_ratio_100_98_l891_89196

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_100_98 : factorial 100 / factorial 98 = 9900 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_100_98_l891_89196


namespace NUMINAMATH_CALUDE_vector_properties_l891_89109

/-- Given vectors a and b in ℝ², prove they are perpendicular and satisfy certain magnitude properties -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (2, 4)) (h2 : b = (-2, 1)) :
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ 
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l891_89109


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l891_89175

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 11) / Real.sqrt (x^2 + 5) ≥ 2 * Real.sqrt 6 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 11) / Real.sqrt (x^2 + 5) = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l891_89175


namespace NUMINAMATH_CALUDE_antonella_remaining_money_l891_89131

/-- Represents the types of Canadian coins -/
inductive CanadianCoin
  | Loonie
  | Toonie

/-- The value of a Canadian coin in dollars -/
def coin_value : CanadianCoin → ℕ
  | CanadianCoin.Loonie => 1
  | CanadianCoin.Toonie => 2

/-- Calculates the total value of coins -/
def total_value (coins : List CanadianCoin) : ℕ :=
  coins.map coin_value |>.sum

theorem antonella_remaining_money :
  let total_coins : ℕ := 10
  let toonie_count : ℕ := 4
  let loonie_count : ℕ := total_coins - toonie_count
  let initial_coins : List CanadianCoin := 
    List.replicate toonie_count CanadianCoin.Toonie ++ List.replicate loonie_count CanadianCoin.Loonie
  let frappuccino_cost : ℕ := 3
  total_value initial_coins - frappuccino_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_antonella_remaining_money_l891_89131


namespace NUMINAMATH_CALUDE_vector_problem_l891_89158

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem vector_problem (h1 : e₁ ≠ 0) (h2 : e₂ ≠ 0) (h3 : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (h4 : B - A = 3 • e₁ + k • e₂) 
  (h5 : C - B = 4 • e₁ + e₂) 
  (h6 : D - C = 8 • e₁ - 9 • e₂)
  (h7 : ∃ (t : ℝ), D - A = t • (B - A)) :
  k = -2 := by sorry

end NUMINAMATH_CALUDE_vector_problem_l891_89158


namespace NUMINAMATH_CALUDE_problem_solution_l891_89130

-- Define the condition from the problem
def condition (m : ℝ) : Prop :=
  ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2

-- Define the theorem
theorem problem_solution :
  (∃ m : ℝ, condition m) →
  (∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5) ∧
  (∀ x y z : ℝ, 3*x + 4*y + 5*z = 5 → x^2 + y^2 + z^2 ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l891_89130


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l891_89143

/-- Given a line L1 with equation 2x + 3y - 6 = 0 and a point P (0, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 3x - 2y - 6 = 0 -/
theorem perpendicular_line_equation :
  let L1 : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y - 6 = 0}
  let P : ℝ × ℝ := (0, -3)
  let L2 : Set (ℝ × ℝ) := {(x, y) | 3 * x - 2 * y - 6 = 0}
  (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 3 * x - 2 * y - 6 = 0) ∧
  P ∈ L2 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L1 → x₁ ≠ x₂ →
    ((x₁ - x₂) * (x - 0) + (y₁ - y₂) * (y + 3) = 0 ↔ (x, y) ∈ L2)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l891_89143


namespace NUMINAMATH_CALUDE_sqrt_3_plus_sqrt_7_less_than_2sqrt_5_l891_89123

theorem sqrt_3_plus_sqrt_7_less_than_2sqrt_5 : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_plus_sqrt_7_less_than_2sqrt_5_l891_89123


namespace NUMINAMATH_CALUDE_infinite_series_sum_l891_89119

theorem infinite_series_sum : 
  let series := fun k : ℕ => (3^(2^k)) / ((5^(2^k)) - 2)
  ∑' k, series k = 1 / (Real.sqrt 5 - 1) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l891_89119


namespace NUMINAMATH_CALUDE_complex_absolute_value_l891_89180

theorem complex_absolute_value (z : ℂ) : z = 1 + I → Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l891_89180


namespace NUMINAMATH_CALUDE_system_of_equations_l891_89188

theorem system_of_equations (x y m : ℝ) : 
  x - y = 5 → 
  x + 2*y = 3*m - 1 → 
  2*x + y = 13 → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l891_89188


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l891_89111

theorem square_minus_product_plus_square : 7^2 - 5*6 + 6^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l891_89111


namespace NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l891_89177

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_of_two_equals_negative_twenty_six 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) :
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l891_89177


namespace NUMINAMATH_CALUDE_mac_loses_three_dollars_l891_89110

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of coins in dollars -/
def total_value (coin : String) (count : ℕ) : ℚ :=
  (coin_value coin * count : ℚ) / 100

/-- Represents Mac's trade of dimes for quarters -/
def dimes_for_quarter : ℕ := 3

/-- Represents Mac's trade of nickels for quarters -/
def nickels_for_quarter : ℕ := 7

/-- Number of quarters Mac trades for using dimes -/
def quarters_from_dimes : ℕ := 20

/-- Number of quarters Mac trades for using nickels -/
def quarters_from_nickels : ℕ := 20

/-- Theorem stating that Mac loses $3.00 in his trades -/
theorem mac_loses_three_dollars :
  total_value "quarter" (quarters_from_dimes + quarters_from_nickels) -
  (total_value "dime" (dimes_for_quarter * quarters_from_dimes) +
   total_value "nickel" (nickels_for_quarter * quarters_from_nickels)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_mac_loses_three_dollars_l891_89110


namespace NUMINAMATH_CALUDE_complex_power_sum_l891_89183

theorem complex_power_sum (z : ℂ) (h : z + (1 / z) = 2 * Real.cos (5 * π / 180)) :
  z^2010 + (1 / z^2010) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l891_89183


namespace NUMINAMATH_CALUDE_root_sum_transformation_l891_89141

theorem root_sum_transformation (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → (β^3 - β - 1 = 0) → (γ^3 - γ - 1 = 0) →
  ((1 - α) / (1 + α)) + ((1 - β) / (1 + β)) + ((1 - γ) / (1 + γ)) = 1 := by
sorry

end NUMINAMATH_CALUDE_root_sum_transformation_l891_89141


namespace NUMINAMATH_CALUDE_rectangle_area_change_l891_89108

theorem rectangle_area_change (initial_area : ℝ) : 
  initial_area = 500 →
  (0.8 * 1.2 * initial_area) = 480 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l891_89108


namespace NUMINAMATH_CALUDE_multiply_72_68_l891_89193

theorem multiply_72_68 : 72 * 68 = 4896 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_multiply_72_68_l891_89193


namespace NUMINAMATH_CALUDE_microphotonics_allocation_l891_89187

/-- Represents the budget allocation for Megatech Corporation -/
structure BudgetAllocation where
  total : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ
  basic_astrophysics_degrees : ℝ
  microphotonics : ℝ

/-- The theorem stating that the microphotonics allocation is 14% given the conditions -/
theorem microphotonics_allocation
  (budget : BudgetAllocation)
  (h1 : budget.total = 100)
  (h2 : budget.home_electronics = 19)
  (h3 : budget.food_additives = 10)
  (h4 : budget.genetically_modified_microorganisms = 24)
  (h5 : budget.industrial_lubricants = 8)
  (h6 : budget.basic_astrophysics_degrees = 90)
  (h7 : budget.microphotonics = budget.total - (budget.home_electronics + budget.food_additives + budget.genetically_modified_microorganisms + budget.industrial_lubricants + (budget.basic_astrophysics_degrees / 360 * 100))) :
  budget.microphotonics = 14 := by
  sorry

end NUMINAMATH_CALUDE_microphotonics_allocation_l891_89187


namespace NUMINAMATH_CALUDE_square_of_negative_product_l891_89159

theorem square_of_negative_product (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l891_89159


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l891_89135

/-- A line passing through (2,3) with equal absolute intercepts -/
theorem line_through_point_equal_intercepts :
  ∃ (m c : ℝ), 
    (3 = 2 * m + c) ∧ 
    (|c| = |c / m|) ∧
    (∀ x y : ℝ, y = m * x + c ↔ y = x + 1) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l891_89135


namespace NUMINAMATH_CALUDE_correct_num_pregnant_dogs_l891_89107

/-- The number of pregnant dogs Chuck has. -/
def num_pregnant_dogs : ℕ := 3

/-- The number of puppies each pregnant dog gives birth to. -/
def puppies_per_dog : ℕ := 4

/-- The number of shots each puppy needs. -/
def shots_per_puppy : ℕ := 2

/-- The cost of each shot in dollars. -/
def cost_per_shot : ℕ := 5

/-- The total cost of all shots in dollars. -/
def total_cost : ℕ := 120

/-- Theorem stating that the number of pregnant dogs is correct given the conditions. -/
theorem correct_num_pregnant_dogs :
  num_pregnant_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = total_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_num_pregnant_dogs_l891_89107


namespace NUMINAMATH_CALUDE_dogwood_trees_in_park_l891_89145

theorem dogwood_trees_in_park (current trees_today trees_tomorrow total : ℕ) : 
  trees_today = 41 → 
  trees_tomorrow = 20 → 
  total = 100 → 
  current + trees_today + trees_tomorrow = total → 
  current = 39 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_park_l891_89145


namespace NUMINAMATH_CALUDE_widgets_sold_is_3125_l891_89134

/-- Represents Jenna's wholesale business --/
structure WholesaleBusiness where
  buy_price : ℝ
  sell_price : ℝ
  rent : ℝ
  tax_rate : ℝ
  worker_salary : ℝ
  num_workers : ℕ
  profit_after_tax : ℝ

/-- Calculates the number of widgets sold given the business parameters --/
def widgets_sold (b : WholesaleBusiness) : ℕ :=
  sorry

/-- Theorem stating that the number of widgets sold is 3125 --/
theorem widgets_sold_is_3125 (jenna : WholesaleBusiness) 
  (h1 : jenna.buy_price = 3)
  (h2 : jenna.sell_price = 8)
  (h3 : jenna.rent = 10000)
  (h4 : jenna.tax_rate = 0.2)
  (h5 : jenna.worker_salary = 2500)
  (h6 : jenna.num_workers = 4)
  (h7 : jenna.profit_after_tax = 4000) :
  widgets_sold jenna = 3125 :=
sorry

end NUMINAMATH_CALUDE_widgets_sold_is_3125_l891_89134


namespace NUMINAMATH_CALUDE_monthly_savings_prediction_l891_89191

/-- Linear regression equation for monthly savings prediction -/
def linear_regression (x : ℝ) (b_hat : ℝ) (a_hat : ℝ) : ℝ :=
  b_hat * x + a_hat

theorem monthly_savings_prediction 
  (n : ℕ) (x_bar : ℝ) (b_hat : ℝ) (a_hat : ℝ) :
  n = 10 →
  x_bar = 8 →
  b_hat = 0.3 →
  a_hat = -0.4 →
  linear_regression 7 b_hat a_hat = 1.7 :=
by sorry

end NUMINAMATH_CALUDE_monthly_savings_prediction_l891_89191


namespace NUMINAMATH_CALUDE_salad_dressing_weight_l891_89163

theorem salad_dressing_weight (bowl_capacity : ℝ) (oil_fraction vinegar_fraction : ℝ)
  (oil_density vinegar_density lemon_juice_density : ℝ) :
  bowl_capacity = 200 →
  oil_fraction = 3/5 →
  vinegar_fraction = 1/4 →
  oil_density = 5 →
  vinegar_density = 4 →
  lemon_juice_density = 2.5 →
  let lemon_juice_fraction : ℝ := 1 - oil_fraction - vinegar_fraction
  let oil_volume : ℝ := bowl_capacity * oil_fraction
  let vinegar_volume : ℝ := bowl_capacity * vinegar_fraction
  let lemon_juice_volume : ℝ := bowl_capacity * lemon_juice_fraction
  let total_weight : ℝ := oil_volume * oil_density + vinegar_volume * vinegar_density + lemon_juice_volume * lemon_juice_density
  total_weight = 875 := by
sorry


end NUMINAMATH_CALUDE_salad_dressing_weight_l891_89163


namespace NUMINAMATH_CALUDE_harold_grocery_expense_l891_89106

/-- Harold's monthly finances --/
def harold_finances (grocery_expense : ℚ) : Prop :=
  let monthly_income : ℚ := 2500
  let rent : ℚ := 700
  let car_payment : ℚ := 300
  let utilities : ℚ := car_payment / 2
  let fixed_expenses : ℚ := rent + car_payment + utilities
  let remaining_after_fixed : ℚ := monthly_income - fixed_expenses
  let retirement_savings : ℚ := (remaining_after_fixed - grocery_expense) / 2
  retirement_savings = 650 ∧ remaining_after_fixed - retirement_savings - grocery_expense = 650

theorem harold_grocery_expense : 
  ∃ (expense : ℚ), harold_finances expense ∧ expense = 50 :=
sorry

end NUMINAMATH_CALUDE_harold_grocery_expense_l891_89106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l891_89148

/-- An arithmetic sequence is represented by its sums of first n, 2n, and 3n terms. -/
structure ArithmeticSequenceSums where
  S : ℝ  -- Sum of first n terms
  T : ℝ  -- Sum of first 2n terms
  R : ℝ  -- Sum of first 3n terms

/-- 
For any arithmetic sequence, given the sums of its first n, 2n, and 3n terms,
the sum of the first 3n terms equals three times the difference between
the sum of the first 2n terms and the sum of the first n terms.
-/
theorem arithmetic_sequence_sum_relation (seq : ArithmeticSequenceSums) : 
  seq.R = 3 * (seq.T - seq.S) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l891_89148


namespace NUMINAMATH_CALUDE_cirrus_to_cumulus_ratio_l891_89105

theorem cirrus_to_cumulus_ratio :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 144 →
    cumulonimbus = 3 →
    cumulus = 12 * cumulonimbus →
    ∃ k : ℕ, cirrus = k * cumulus →
    cirrus / cumulus = 4 :=
by sorry

end NUMINAMATH_CALUDE_cirrus_to_cumulus_ratio_l891_89105


namespace NUMINAMATH_CALUDE_product_in_B_l891_89120

-- Define the sets A and B
def A : Set ℤ := {x | ∃ (a b k m : ℤ), x = m * a^2 + k * a * b + m * b^2}
def B : Set ℤ := {x | ∃ (a b k m : ℤ), x = a^2 + k * a * b + m^2 * b^2}

-- State the theorem
theorem product_in_B (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ B := by
  sorry

end NUMINAMATH_CALUDE_product_in_B_l891_89120


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l891_89179

theorem polygon_exterior_angles (n : ℕ) (h : n > 2) :
  (n : ℝ) * 60 = 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l891_89179


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l891_89133

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, |x| + |2023 - x| < 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l891_89133


namespace NUMINAMATH_CALUDE_roots_sum_to_four_l891_89162

theorem roots_sum_to_four : ∃ (x y : ℝ), x^2 - 4*x - 1 = 0 ∧ y^2 - 4*y - 1 = 0 ∧ x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_to_four_l891_89162


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l891_89116

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {y | (y - 2) * (y + 3) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l891_89116


namespace NUMINAMATH_CALUDE_equal_coins_count_l891_89140

/-- Represents the value of a coin in cents -/
def coin_value (coin_type : String) : ℕ :=
  match coin_type with
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 120

/-- Represents the number of different types of coins -/
def num_coin_types : ℕ := 3

theorem equal_coins_count (num_each : ℕ) :
  (num_each * coin_value "nickel" +
   num_each * coin_value "dime" +
   num_each * coin_value "quarter" = total_value) →
  (num_each * num_coin_types = 9) := by
  sorry

#check equal_coins_count

end NUMINAMATH_CALUDE_equal_coins_count_l891_89140


namespace NUMINAMATH_CALUDE_apple_distribution_l891_89113

theorem apple_distribution (x y : ℕ) : 
  (y - 1 = x + 1) →
  (y + 1 = 3 * (x - 1)) →
  (x = 3 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l891_89113


namespace NUMINAMATH_CALUDE_min_discriminant_l891_89169

/-- A quadratic trinomial that satisfies the problem conditions -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ
  nonnegative : ∀ x, a * x^2 + b * x + c ≥ 0
  below_curve : ∀ x, abs x < 1 → a * x^2 + b * x + c ≤ 1 / Real.sqrt (1 - x^2)

/-- The discriminant of a quadratic trinomial -/
def discriminant (q : QuadraticTrinomial) : ℝ := q.b^2 - 4 * q.a * q.c

/-- The theorem stating the minimum value of the discriminant -/
theorem min_discriminant :
  (∀ q : QuadraticTrinomial, discriminant q ≥ -4) ∧
  (∃ q : QuadraticTrinomial, discriminant q = -4) := by sorry

end NUMINAMATH_CALUDE_min_discriminant_l891_89169


namespace NUMINAMATH_CALUDE_simplify_expression_l891_89167

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l891_89167


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l891_89156

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 1 / Real.rpow 6 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b) + b / (6 * c) + c / (9 * a)) = 1 / Real.rpow 6 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l891_89156


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l891_89185

theorem modulus_of_complex_fraction :
  let z : ℂ := (1 - 2*I) / (3 - I)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l891_89185


namespace NUMINAMATH_CALUDE_product_equality_l891_89142

def prod (a : ℕ → ℕ) (m n : ℕ) : ℕ :=
  if m > n then 1 else (List.range (n - m + 1)).foldl (fun acc i => acc * a (i + m)) 1

theorem product_equality :
  (prod (fun k => 2 * k - 1) 1 1008) * (prod (fun k => 2 * k) 1 1007) = prod id 1 2015 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l891_89142


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_two_l891_89126

theorem binomial_coefficient_n_plus_one_choose_two (n : ℕ) : 
  Nat.choose (n + 1) 2 = (n + 1) * n / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_two_l891_89126


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l891_89150

theorem divisible_by_twelve (n : ℤ) : 12 ∣ n^2 * (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l891_89150


namespace NUMINAMATH_CALUDE_range_of_sum_l891_89170

theorem range_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + 4*y^2 = 1) :
  0 < x + y ∧ x + y < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_sum_l891_89170


namespace NUMINAMATH_CALUDE_excess_calories_is_770_l891_89164

/-- Calculates the excess calories consumed by James after snacking and exercising -/
def excess_calories : ℕ :=
  let cheezit_bags : ℕ := 3
  let cheezit_oz_per_bag : ℕ := 2
  let cheezit_cal_per_oz : ℕ := 150
  let chocolate_bars : ℕ := 2
  let chocolate_cal_per_bar : ℕ := 250
  let popcorn_cal : ℕ := 500
  let run_minutes : ℕ := 40
  let run_cal_per_minute : ℕ := 12
  let swim_minutes : ℕ := 30
  let swim_cal_per_minute : ℕ := 15
  let cycle_minutes : ℕ := 20
  let cycle_cal_per_minute : ℕ := 10

  let total_calories_consumed : ℕ := 
    cheezit_bags * cheezit_oz_per_bag * cheezit_cal_per_oz +
    chocolate_bars * chocolate_cal_per_bar +
    popcorn_cal

  let total_calories_burned : ℕ := 
    run_minutes * run_cal_per_minute +
    swim_minutes * swim_cal_per_minute +
    cycle_minutes * cycle_cal_per_minute

  total_calories_consumed - total_calories_burned

theorem excess_calories_is_770 : excess_calories = 770 := by
  sorry

end NUMINAMATH_CALUDE_excess_calories_is_770_l891_89164


namespace NUMINAMATH_CALUDE_contest_order_l891_89117

-- Define the contestants
variable (Andy Beth Carol Dave : ℝ)

-- Define the conditions
axiom sum_equality : Andy + Carol = Beth + Dave
axiom interchange_inequality : Beth + Andy > Dave + Carol
axiom carol_highest : Carol > Andy + Beth
axiom nonnegative_scores : Andy ≥ 0 ∧ Beth ≥ 0 ∧ Carol ≥ 0 ∧ Dave ≥ 0

-- Theorem to prove
theorem contest_order : Carol > Beth ∧ Beth > Andy ∧ Andy > Dave := by
  sorry

end NUMINAMATH_CALUDE_contest_order_l891_89117


namespace NUMINAMATH_CALUDE_opposite_equal_implies_zero_abs_equal_implies_equal_or_opposite_sum_product_condition_implies_abs_equality_abs_plus_self_nonnegative_l891_89195

-- Statement 1
theorem opposite_equal_implies_zero (x : ℝ) : x = -x → x = 0 := by sorry

-- Statement 2
theorem abs_equal_implies_equal_or_opposite (a b : ℝ) : |a| = |b| → a = b ∨ a = -b := by sorry

-- Statement 3
theorem sum_product_condition_implies_abs_equality (a b : ℝ) : 
  a + b < 0 → ab > 0 → |7*a + 3*b| = -(7*a + 3*b) := by sorry

-- Statement 4
theorem abs_plus_self_nonnegative (m : ℚ) : |m| + m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_opposite_equal_implies_zero_abs_equal_implies_equal_or_opposite_sum_product_condition_implies_abs_equality_abs_plus_self_nonnegative_l891_89195


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l891_89155

/-- A quadrilateral inscribed in a circle with given side lengths -/
structure InscribedQuadrilateral where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The lengths of the four sides of the quadrilateral
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Theorem stating that for a quadrilateral inscribed in a circle with radius 300√2
    and three sides of lengths 300, 300, and 150√2, the fourth side has length 300√2 -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 300 * Real.sqrt 2)
  (h_side1 : q.side1 = 300)
  (h_side2 : q.side2 = 300)
  (h_side3 : q.side3 = 150 * Real.sqrt 2) :
  q.side4 = 300 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l891_89155


namespace NUMINAMATH_CALUDE_sum_f_positive_l891_89184

-- Define the function f
def f (x : ℝ) : ℝ := x + x^3

-- State the theorem
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) 
  (h₂ : x₂ + x₃ > 0) 
  (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l891_89184


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l891_89137

theorem cubic_equation_solution :
  let f : ℝ → ℝ := λ x => (x + 1)^3 + (3 - x)^3
  ∃ (x₁ x₂ : ℝ), 
    (f x₁ = 35 ∧ f x₂ = 35) ∧ 
    (x₁ = 1 + Real.sqrt (19/3) / 2) ∧ 
    (x₂ = 1 - Real.sqrt (19/3) / 2) ∧
    (∀ x : ℝ, f x = 35 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l891_89137


namespace NUMINAMATH_CALUDE_front_axle_wheels_count_l891_89198

/-- Represents a truck with a specific wheel configuration -/
structure Truck where
  total_wheels : ℕ
  wheels_per_axle : ℕ
  front_axle_wheels : ℕ
  toll : ℚ

/-- Calculates the number of axles for a given truck -/
def num_axles (t : Truck) : ℕ :=
  (t.total_wheels - t.front_axle_wheels) / t.wheels_per_axle + 1

/-- Calculates the toll for a given number of axles -/
def toll_formula (x : ℕ) : ℚ :=
  (3/2) + (3/2) * (x - 2)

/-- Theorem stating that a truck with the given specifications has 2 wheels on its front axle -/
theorem front_axle_wheels_count (t : Truck) 
    (h1 : t.total_wheels = 18)
    (h2 : t.wheels_per_axle = 4)
    (h3 : t.toll = 6)
    (h4 : t.toll = toll_formula (num_axles t)) :
    t.front_axle_wheels = 2 := by
  sorry

end NUMINAMATH_CALUDE_front_axle_wheels_count_l891_89198


namespace NUMINAMATH_CALUDE_chloe_trivia_score_l891_89104

/-- The total points scored in a trivia game with three rounds -/
def trivia_game_score (round1 : Int) (round2 : Int) (round3 : Int) : Int :=
  round1 + round2 + round3

/-- Theorem: The total points at the end of Chloe's trivia game is 86 -/
theorem chloe_trivia_score : trivia_game_score 40 50 (-4) = 86 := by
  sorry

end NUMINAMATH_CALUDE_chloe_trivia_score_l891_89104


namespace NUMINAMATH_CALUDE_sequence_ratio_range_l891_89138

theorem sequence_ratio_range (x y a₁ a₂ b₁ b₂ : ℝ) 
  (h_arith : a₁ - x = a₂ - a₁ ∧ a₂ - a₁ = y - a₂)
  (h_geom : b₁ / x = b₂ / b₁ ∧ b₂ / b₁ = y / b₂) :
  (a₁ + a₂)^2 / (b₁ * b₂) ≥ 4 ∨ (a₁ + a₂)^2 / (b₁ * b₂) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_range_l891_89138


namespace NUMINAMATH_CALUDE_painted_cube_probability_l891_89139

theorem painted_cube_probability : 
  let cube_side : ℕ := 5
  let total_cubes : ℕ := cube_side ^ 3
  let painted_faces : ℕ := 2
  let two_face_painted : ℕ := 4 * (cube_side - 1)
  let no_face_painted : ℕ := total_cubes - 2 * cube_side^2 + two_face_painted
  let total_combinations : ℕ := total_cubes.choose 2
  let favorable_outcomes : ℕ := two_face_painted * no_face_painted
  (favorable_outcomes : ℚ) / total_combinations = 728 / 3875 := by
sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l891_89139


namespace NUMINAMATH_CALUDE_sailboat_two_sail_speed_l891_89171

/-- Represents the speed of a sailboat in knots -/
structure SailboatSpeed :=
  (speed : ℝ)

/-- Represents the travel conditions for a sailboat -/
structure TravelConditions :=
  (oneSpeedSail : SailboatSpeed)
  (twoSpeedSail : SailboatSpeed)
  (timeOneSail : ℝ)
  (timeTwoSail : ℝ)
  (totalDistance : ℝ)
  (nauticalMileToLandMile : ℝ)

/-- The main theorem stating the speed of the sailboat with two sails -/
theorem sailboat_two_sail_speed 
  (conditions : TravelConditions)
  (h1 : conditions.oneSpeedSail.speed = 25)
  (h2 : conditions.timeOneSail = 4)
  (h3 : conditions.timeTwoSail = 4)
  (h4 : conditions.totalDistance = 345)
  (h5 : conditions.nauticalMileToLandMile = 1.15)
  : conditions.twoSpeedSail.speed = 50 := by
  sorry

#check sailboat_two_sail_speed

end NUMINAMATH_CALUDE_sailboat_two_sail_speed_l891_89171


namespace NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l891_89192

theorem sin_n_squared_not_converge_to_zero :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ |Real.sin (n^2 : ℝ)| ≥ ε :=
sorry

end NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l891_89192


namespace NUMINAMATH_CALUDE_jerky_order_fulfillment_l891_89136

/-- Calculates the number of days required to fulfill a jerky order -/
def days_to_fulfill_order (order : ℕ) (in_stock : ℕ) (production_rate : ℕ) : ℕ :=
  ((order - in_stock) + production_rate - 1) / production_rate

/-- Theorem: Given the specific conditions, it takes 4 days to fulfill the order -/
theorem jerky_order_fulfillment :
  days_to_fulfill_order 60 20 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerky_order_fulfillment_l891_89136


namespace NUMINAMATH_CALUDE_students_without_gift_l891_89115

theorem students_without_gift (total_students : ℕ) (h : total_students = 2016) :
  (∃ (no_gift : ℕ), no_gift = Nat.totient total_students ∧
    ∀ (n : ℕ), n ≥ 2 → no_gift = total_students - (total_students / n) * n) := by
  sorry

end NUMINAMATH_CALUDE_students_without_gift_l891_89115


namespace NUMINAMATH_CALUDE_max_sign_changes_is_n_minus_one_sign_changes_bounded_l891_89102

/-- The maximum number of sign changes for the first element in a sequence of n real numbers 
    under the described averaging process. -/
def max_sign_changes (n : ℕ) : ℕ :=
  n - 1

/-- The theorem stating that the maximum number of sign changes for the first element
    is n-1 for any positive integer n. -/
theorem max_sign_changes_is_n_minus_one (n : ℕ) (hn : n > 0) : 
  max_sign_changes n = n - 1 := by
  sorry

/-- A helper function to represent the averaging operation on a sequence of real numbers. -/
def average_operation (seq : List ℝ) (i : ℕ) : List ℝ :=
  sorry

/-- A predicate to check if a number has changed sign. -/
def sign_changed (a b : ℝ) : Prop :=
  (a ≥ 0 ∧ b < 0) ∨ (a < 0 ∧ b ≥ 0)

/-- A function to count the number of sign changes in a₁ after a sequence of operations. -/
def count_sign_changes (initial_seq : List ℝ) (operations : List ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for any initial sequence and any sequence of operations,
    the number of sign changes in a₁ is at most n-1. -/
theorem sign_changes_bounded (n : ℕ) (hn : n > 0) 
  (initial_seq : List ℝ) (h_seq : initial_seq.length = n)
  (operations : List ℕ) :
  count_sign_changes initial_seq operations ≤ max_sign_changes n := by
  sorry

end NUMINAMATH_CALUDE_max_sign_changes_is_n_minus_one_sign_changes_bounded_l891_89102


namespace NUMINAMATH_CALUDE_square_roots_sum_l891_89122

theorem square_roots_sum (x y : ℝ) : 
  x^2 = 16 → y^2 = 9 → x^2 + y^2 + x - 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_l891_89122


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l891_89189

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 6*x - 16 = 0) ∧
  (∃ x : ℝ, 2*x^2 - 3*x - 5 = 0) →
  (∃ x : ℝ, x = 8 ∨ x = -2) ∧
  (∃ x : ℝ, x = 5/2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l891_89189


namespace NUMINAMATH_CALUDE_enchiladas_ordered_l891_89176

/-- The number of enchiladas you ordered -/
def your_enchiladas : ℕ := 3

/-- The cost of each taco in dollars -/
def taco_cost : ℚ := 9/10

/-- Your bill in dollars (without tax) -/
def your_bill : ℚ := 78/10

/-- Your friend's bill in dollars (without tax) -/
def friend_bill : ℚ := 127/10

/-- The cost of each enchilada in dollars -/
def enchilada_cost : ℚ := 2

theorem enchiladas_ordered :
  (2 * taco_cost + your_enchiladas * enchilada_cost = your_bill) ∧
  (3 * taco_cost + 5 * enchilada_cost = friend_bill) :=
by sorry

end NUMINAMATH_CALUDE_enchiladas_ordered_l891_89176


namespace NUMINAMATH_CALUDE_smallest_perimeter_consecutive_integer_triangle_l891_89152

/-- A triangle with consecutive integer side lengths greater than 1 -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  greater_than_one : a > 1

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The triangle inequality -/
def satisfies_triangle_inequality (t : ConsecutiveIntegerTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem smallest_perimeter_consecutive_integer_triangle :
  ∀ t : ConsecutiveIntegerTriangle,
  satisfies_triangle_inequality t →
  perimeter t ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_consecutive_integer_triangle_l891_89152


namespace NUMINAMATH_CALUDE_baseball_season_games_l891_89149

/-- The number of baseball games in a season -/
def games_in_season (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem: There are 14 baseball games in a season -/
theorem baseball_season_games :
  games_in_season 7 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_baseball_season_games_l891_89149


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_l891_89186

theorem greatest_integer_with_gcf_four : ∃ n : ℕ, n < 200 ∧ 
  Nat.gcd n 24 = 4 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 24 = 4 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_l891_89186


namespace NUMINAMATH_CALUDE_sum_of_integers_l891_89173

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 245) (h2 : x.val * y.val = 120) :
  (x.val : ℝ) + y.val = Real.sqrt 485 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l891_89173


namespace NUMINAMATH_CALUDE_special_number_property_l891_89121

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a natural number has all identical digits -/
def has_identical_digits (n : ℕ) : Prop := sorry

/-- The unique three-digit number satisfying the given conditions -/
def special_number : ℕ := 105

theorem special_number_property :
  ∃! (N : ℕ), 
    100 ≤ N ∧ N < 1000 ∧ 
    has_identical_digits (N + digit_sum N) ∧
    has_identical_digits (N - digit_sum N) ∧
    N = special_number := by
  sorry

end NUMINAMATH_CALUDE_special_number_property_l891_89121


namespace NUMINAMATH_CALUDE_choir_average_age_l891_89182

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 8)
  (h2 : num_males = 17)
  (h3 : avg_age_females = 28)
  (h4 : avg_age_males = 32) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 768 / 25 := by
sorry

end NUMINAMATH_CALUDE_choir_average_age_l891_89182


namespace NUMINAMATH_CALUDE_revenue_change_l891_89174

theorem revenue_change (R : ℝ) (p : ℝ) (h1 : R > 0) :
  (R + p / 100 * R) * (1 - p / 100) = R * (1 - 4 / 100) →
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l891_89174


namespace NUMINAMATH_CALUDE_same_color_probability_l891_89114

/-- The number of red plates -/
def red_plates : ℕ := 6

/-- The number of blue plates -/
def blue_plates : ℕ := 5

/-- The number of green plates -/
def green_plates : ℕ := 3

/-- The total number of plates -/
def total_plates : ℕ := red_plates + blue_plates + green_plates

/-- The number of ways to choose 3 plates from the total number of plates -/
def total_ways : ℕ := Nat.choose total_plates 3

/-- The number of ways to choose 3 red plates -/
def red_ways : ℕ := Nat.choose red_plates 3

/-- The number of ways to choose 3 blue plates -/
def blue_ways : ℕ := Nat.choose blue_plates 3

/-- The number of ways to choose 3 green plates -/
def green_ways : ℕ := Nat.choose green_plates 3

/-- The total number of favorable outcomes (all same color) -/
def favorable_outcomes : ℕ := red_ways + blue_ways + green_ways

/-- The probability of selecting three plates of the same color -/
theorem same_color_probability : 
  (favorable_outcomes : ℚ) / total_ways = 31 / 364 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l891_89114


namespace NUMINAMATH_CALUDE_largest_subset_size_150_l891_89100

/-- A function that returns the size of the largest subset of integers from 1 to n 
    where no member is 4 times another member -/
def largest_subset_size (n : ℕ) : ℕ := 
  sorry

/-- The theorem to be proved -/
theorem largest_subset_size_150 : largest_subset_size 150 = 142 := by
  sorry

end NUMINAMATH_CALUDE_largest_subset_size_150_l891_89100


namespace NUMINAMATH_CALUDE_bus_passengers_l891_89132

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 64 →
  num_stops = 4 →
  (initial_students : ℚ) * (2/3)^num_stops = 1024/81 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l891_89132


namespace NUMINAMATH_CALUDE_total_addresses_is_40_l891_89190

/-- The number of commencement addresses given by Governor Sandoval -/
def sandoval_addresses : ℕ := 12

/-- The number of commencement addresses given by Governor Hawkins -/
def hawkins_addresses : ℕ := sandoval_addresses / 2

/-- The number of commencement addresses given by Governor Sloan -/
def sloan_addresses : ℕ := sandoval_addresses + 10

/-- The total number of commencement addresses given by all three governors -/
def total_addresses : ℕ := sandoval_addresses + hawkins_addresses + sloan_addresses

theorem total_addresses_is_40 : total_addresses = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_addresses_is_40_l891_89190


namespace NUMINAMATH_CALUDE_complex_sum_properties_l891_89151

open Complex

/-- Given complex numbers z and u with the specified properties, prove the required statements -/
theorem complex_sum_properties (α β : ℝ) (z u : ℂ) 
  (hz : z = Complex.exp (I * α))  -- z = cos α + i sin α
  (hu : u = Complex.exp (I * β))  -- u = cos β + i sin β
  (hsum : z + u = (4/5 : ℂ) + (3/5 : ℂ) * I) : 
  (Complex.tan (α + β) = 24/7) ∧ (z^2 + u^2 + z*u = 0) := by
  sorry


end NUMINAMATH_CALUDE_complex_sum_properties_l891_89151


namespace NUMINAMATH_CALUDE_max_sectional_area_of_cone_l891_89197

/-- The maximum sectional area of a cone --/
theorem max_sectional_area_of_cone (θ : Real) (l : Real) : 
  θ = π / 3 → l = 3 → (∀ α, 0 ≤ α ∧ α ≤ 2*π/3 → (1/2) * l^2 * Real.sin α ≤ 9/2) ∧ 
  ∃ α, 0 ≤ α ∧ α ≤ 2*π/3 ∧ (1/2) * l^2 * Real.sin α = 9/2 :=
by sorry

#check max_sectional_area_of_cone

end NUMINAMATH_CALUDE_max_sectional_area_of_cone_l891_89197


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l891_89146

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = 2^x + 1) :
  ∀ x < 0, f x = -2^(-x) - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l891_89146


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l891_89101

theorem necessary_not_sufficient_condition (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l891_89101


namespace NUMINAMATH_CALUDE_root_cubic_expression_l891_89127

theorem root_cubic_expression (m : ℝ) : 
  m^2 + 3*m - 2022 = 0 → m^3 + 4*m^2 - 2019*m - 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_expression_l891_89127


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l891_89172

theorem rebus_puzzle_solution :
  ∀ (A B C : ℕ),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A ≠ B → B ≠ C → A ≠ C →
    A < 10 → B < 10 → C < 10 →
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) = (100 * A + 10 * C + C) →
    (100 * A + 10 * C + C) = 1416 →
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l891_89172


namespace NUMINAMATH_CALUDE_children_on_bus_l891_89194

theorem children_on_bus (initial_children : ℕ) (children_who_got_on : ℕ) : 
  initial_children = 18 → children_who_got_on = 7 → 
  initial_children + children_who_got_on = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_l891_89194


namespace NUMINAMATH_CALUDE_intersection_M_N_l891_89181

def M : Set ℝ := {x | 2 * x - x^2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l891_89181


namespace NUMINAMATH_CALUDE_birds_total_distance_l891_89168

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def flight_time : ℕ := 2

def total_distance : ℕ := eagle_speed * flight_time + falcon_speed * flight_time + 
                           pelican_speed * flight_time + hummingbird_speed * flight_time

theorem birds_total_distance : total_distance = 248 := by
  sorry

end NUMINAMATH_CALUDE_birds_total_distance_l891_89168


namespace NUMINAMATH_CALUDE_channel_count_is_164_l891_89165

/-- Calculates the final number of channels after a series of changes --/
def final_channels (initial : ℕ) : ℕ :=
  let after_first := initial - 20 + 12
  let after_second := after_first - 10 + 8
  let after_third := after_second + 15 - 5
  let overlap := (after_third * 10) / 100
  let after_fourth := after_third + (25 - overlap)
  after_fourth + 7 - 3

/-- Theorem stating that given the initial number of channels and the series of changes, 
    the final number of channels is 164 --/
theorem channel_count_is_164 : final_channels 150 = 164 := by
  sorry

end NUMINAMATH_CALUDE_channel_count_is_164_l891_89165


namespace NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l891_89144

theorem simultaneous_pipe_filling_time 
  (fill_time_A : ℝ) 
  (fill_time_B : ℝ) 
  (h1 : fill_time_A = 50) 
  (h2 : fill_time_B = 75) : 
  (1 / (1 / fill_time_A + 1 / fill_time_B)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l891_89144


namespace NUMINAMATH_CALUDE_signals_coincide_l891_89128

def town_hall_period : ℕ := 18
def library_period : ℕ := 24
def fire_station_period : ℕ := 36

def coincidence_time : ℕ := 72

theorem signals_coincide :
  coincidence_time = Nat.lcm town_hall_period (Nat.lcm library_period fire_station_period) :=
by sorry

end NUMINAMATH_CALUDE_signals_coincide_l891_89128


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l891_89112

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), (433124 + m) % 17 = 0 → m ≥ n) ∧ 
  (433124 + n) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l891_89112


namespace NUMINAMATH_CALUDE_min_value_function_l891_89166

theorem min_value_function (x : ℝ) (h : x > 3) : 
  (1 / (x - 3)) + x ≥ 5 ∧ ∃ y > 3, (1 / (y - 3)) + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l891_89166


namespace NUMINAMATH_CALUDE_correct_notebooks_A_correct_min_full_price_sales_l891_89154

/-- Represents the bookstore problem with notebooks of two types -/
structure BookstoreProblem where
  total_notebooks : ℕ
  cost_price_A : ℕ
  cost_price_B : ℕ
  total_cost : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ
  discount_A : ℚ
  profit_threshold : ℕ

/-- The specific instance of the bookstore problem -/
def problem : BookstoreProblem :=
  { total_notebooks := 350
  , cost_price_A := 12
  , cost_price_B := 15
  , total_cost := 4800
  , selling_price_A := 20
  , selling_price_B := 25
  , discount_A := 0.7
  , profit_threshold := 2348 }

/-- The number of type A notebooks purchased -/
def notebooks_A (p : BookstoreProblem) : ℕ := sorry

/-- The number of type B notebooks purchased -/
def notebooks_B (p : BookstoreProblem) : ℕ := sorry

/-- The minimum number of notebooks of each type that must be sold at full price -/
def min_full_price_sales (p : BookstoreProblem) : ℕ := sorry

/-- Theorem stating the correct number of type A notebooks -/
theorem correct_notebooks_A : notebooks_A problem = 150 := by sorry

/-- Theorem stating the correct minimum number of full-price sales -/
theorem correct_min_full_price_sales : min_full_price_sales problem = 128 := by sorry

end NUMINAMATH_CALUDE_correct_notebooks_A_correct_min_full_price_sales_l891_89154


namespace NUMINAMATH_CALUDE_sqrt_10_irrational_l891_89118

theorem sqrt_10_irrational : Irrational (Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_irrational_l891_89118


namespace NUMINAMATH_CALUDE_danielles_rooms_l891_89153

theorem danielles_rooms (d h g : ℕ) : 
  h = 3 * d →  -- Heidi's apartment has 3 times as many rooms as Danielle's
  g = h / 9 →  -- Grant's apartment has 1/9 as many rooms as Heidi's
  g = 2 →      -- Grant's apartment has 2 rooms
  d = 6        -- Prove that Danielle's apartment has 6 rooms
:= by sorry

end NUMINAMATH_CALUDE_danielles_rooms_l891_89153


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l891_89160

/-- An arithmetic sequence with first four terms a, x, b, 2x has the property that a/b = 1/3 -/
theorem arithmetic_sequence_ratio (a x b : ℝ) : 
  (∃ d : ℝ, x - a = d ∧ b - x = d ∧ 2*x - b = d) → a/b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l891_89160


namespace NUMINAMATH_CALUDE_toms_spending_ratio_l891_89157

def monthly_allowance : ℚ := 12
def first_week_spending_ratio : ℚ := 1/3
def remaining_money : ℚ := 6

theorem toms_spending_ratio :
  let first_week_spending := monthly_allowance * first_week_spending_ratio
  let money_after_first_week := monthly_allowance - first_week_spending
  let second_week_spending := money_after_first_week - remaining_money
  second_week_spending / money_after_first_week = 1/4 := by
sorry

end NUMINAMATH_CALUDE_toms_spending_ratio_l891_89157


namespace NUMINAMATH_CALUDE_inequality_proof_l891_89103

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l891_89103
