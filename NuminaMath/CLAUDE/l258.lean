import Mathlib

namespace smallest_solution_abs_equation_l258_25891

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 4 * x + 3 ∧
  (∀ (y : ℝ), y * |y| = 4 * y + 3 → x ≤ y) ∧
  x = -3 :=
by sorry

end smallest_solution_abs_equation_l258_25891


namespace min_value_ab_l258_25821

theorem min_value_ab (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 4/b = Real.sqrt (a*b)) : 
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 :=
sorry

end min_value_ab_l258_25821


namespace g_extreme_points_l258_25860

noncomputable def f (x : ℝ) : ℝ := Real.log x - x - 1

noncomputable def g (x : ℝ) : ℝ := x * f x + (1/2) * x^2 + 2 * x

noncomputable def g' (x : ℝ) : ℝ := f x + 3

theorem g_extreme_points :
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ < 1 ∧
    3 < x₂ ∧ x₂ < 4 ∧
    g' x₁ = 0 ∧ g' x₂ = 0 ∧
    (∀ x ∈ Set.Ioo 0 1, x ≠ x₁ → g' x ≠ 0) ∧
    (∀ x ∈ Set.Ioo 3 4, x ≠ x₂ → g' x ≠ 0) :=
sorry

end g_extreme_points_l258_25860


namespace contrapositive_equivalence_l258_25824

theorem contrapositive_equivalence :
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end contrapositive_equivalence_l258_25824


namespace range_and_minimum_l258_25868

theorem range_and_minimum (x y a : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : x^2 - y^2 = 2)
  (h_ineq : (1 / (2*x^2)) + (2*y/x) < a) :
  (0 < y/x ∧ y/x < 1) ∧ a ≥ 2 := by
  sorry

end range_and_minimum_l258_25868


namespace chess_tournament_games_l258_25869

theorem chess_tournament_games (n : Nat) (games : Fin n → Nat) :
  n = 5 →
  (∀ i j : Fin n, i ≠ j → games i + games j ≤ n - 1) →
  (∃ p : Fin n, games p = 4) →
  (∃ p : Fin n, games p = 3) →
  (∃ p : Fin n, games p = 2) →
  (∃ p : Fin n, games p = 1) →
  (∃ p : Fin n, games p = 2) :=
by sorry

end chess_tournament_games_l258_25869


namespace earrings_price_decrease_l258_25897

/-- Given a pair of earrings with the following properties:
  - Purchase price: $240
  - Original markup: 25% of the selling price
  - Gross profit after price decrease: $16
  Prove that the percentage decrease in the selling price is 5% -/
theorem earrings_price_decrease (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) :
  purchase_price = 240 →
  markup_percentage = 0.25 →
  gross_profit = 16 →
  let original_selling_price := purchase_price / (1 - markup_percentage)
  let new_selling_price := original_selling_price - gross_profit
  let price_decrease := original_selling_price - new_selling_price
  let percentage_decrease := price_decrease / original_selling_price * 100
  percentage_decrease = 5 := by
  sorry

end earrings_price_decrease_l258_25897


namespace mask_price_problem_l258_25895

theorem mask_price_problem (first_total second_total : ℚ) 
  (price_increase : ℚ) (quantity_increase : ℕ) :
  first_total = 500000 →
  second_total = 770000 →
  price_increase = 1.4 →
  quantity_increase = 10000 →
  ∃ (first_price first_quantity : ℚ),
    first_price * first_quantity = first_total ∧
    price_increase * first_price * (first_quantity + quantity_increase) = second_total ∧
    first_price = 5 := by
  sorry

end mask_price_problem_l258_25895


namespace triangle_side_length_l258_25877

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  B = π / 6 →
  c = 2 * Real.sqrt 3 →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = 2 := by
  sorry

end triangle_side_length_l258_25877


namespace age_ratio_after_two_years_l258_25849

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the specified conditions. -/
theorem age_ratio_after_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 27 → 
  man_age = son_age + 29 → 
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end age_ratio_after_two_years_l258_25849


namespace conic_sections_properties_l258_25851

-- Define the equations for the conic sections
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1
def ellipse_eq (x y : ℝ) : Prop := x^2 / 35 + y^2 = 1
def parabola_eq (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Define the theorem
theorem conic_sections_properties :
  -- Proposition ②
  (∃ e₁ e₂ : ℝ, quadratic_eq e₁ ∧ quadratic_eq e₂ ∧ 0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1) ∧
  -- Proposition ③
  (∃ c : ℝ, (∀ x y : ℝ, hyperbola_eq x y → x^2 - c^2 = 25) ∧
            (∀ x y : ℝ, ellipse_eq x y → x^2 + c^2 = 35)) ∧
  -- Proposition ④
  (∀ p : ℝ, p > 0 →
    ∃ x₀ y₀ r : ℝ,
      -- Circle equation
      (∀ x y : ℝ, (x - x₀)^2 + (y - y₀)^2 = r^2 →
        -- Tangent to directrix
        x = -p ∨
        -- Passes through focus
        (x₀ = p/2 ∧ y₀ = 0 ∧ r = p/2))) :=
sorry

end conic_sections_properties_l258_25851


namespace sufficient_but_not_necessary_l258_25871

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the theorem
theorem sufficient_but_not_necessary
  (a b : Line) (α β : Plane)
  (h_diff : a ≠ b)
  (h_parallel : parallel α β)
  (h_perp : perpendicular a α) :
  (∃ (c : Line), c ≠ b ∧ perpendicularLines a c ∧ ¬lineParallelPlane c β) ∧
  (lineParallelPlane b β → perpendicularLines a b) ∧
  ¬(perpendicularLines a b → lineParallelPlane b β) :=
sorry

end sufficient_but_not_necessary_l258_25871


namespace cannot_compare_greening_areas_l258_25811

-- Define the structure for a city
structure City where
  total_area : ℝ
  greening_coverage_rate : ℝ

-- Define the greening coverage area
def greening_coverage_area (city : City) : ℝ :=
  city.total_area * city.greening_coverage_rate

-- Theorem statement
theorem cannot_compare_greening_areas (city_a city_b : City)
  (h_a : city_a.greening_coverage_rate = 0.10)
  (h_b : city_b.greening_coverage_rate = 0.08) :
  ¬ (∀ (a b : City), a.greening_coverage_rate > b.greening_coverage_rate →
    greening_coverage_area a > greening_coverage_area b) :=
by sorry

end cannot_compare_greening_areas_l258_25811


namespace grunters_win_probability_l258_25861

/-- The probability of a team winning a single game -/
def win_probability : ℚ := 4/5

/-- The number of games in the series -/
def num_games : ℕ := 5

/-- The probability of winning all games in the series -/
def win_all_probability : ℚ := (4/5)^5

theorem grunters_win_probability :
  win_all_probability = 1024/3125 := by
  sorry

end grunters_win_probability_l258_25861


namespace max_point_difference_is_n_l258_25843

/-- Represents a hockey tournament with n teams -/
structure HockeyTournament where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- The maximum point difference between consecutively ranked teams -/
def maxPointDifference (t : HockeyTournament) : ℕ := t.n

/-- Theorem: The maximum point difference between consecutively ranked teams is n -/
theorem max_point_difference_is_n (t : HockeyTournament) : 
  maxPointDifference t = t.n := by sorry

end max_point_difference_is_n_l258_25843


namespace circle_center_and_radius_l258_25875

/-- Given a circle with equation x^2 + y^2 - 4x = 0, prove that its center is (2, 0) and its radius is 2 -/
theorem circle_center_and_radius :
  let equation := fun (x y : ℝ) => x^2 + y^2 - 4*x
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (2, 0) ∧
    radius = 2 := by
  sorry

end circle_center_and_radius_l258_25875


namespace cost_increase_is_six_percent_l258_25880

/-- Represents the cost components of manufacturing a car -/
structure CarCost where
  rawMaterial : ℝ
  labor : ℝ
  overheads : ℝ

/-- Calculates the total cost of manufacturing a car -/
def totalCost (cost : CarCost) : ℝ :=
  cost.rawMaterial + cost.labor + cost.overheads

/-- Represents the cost ratio in the first year -/
def initialRatio : CarCost :=
  { rawMaterial := 4
    labor := 3
    overheads := 2 }

/-- Calculates the new cost after applying percentage changes -/
def newCost (cost : CarCost) : CarCost :=
  { rawMaterial := cost.rawMaterial * 1.1
    labor := cost.labor * 1.08
    overheads := cost.overheads * 0.95 }

/-- Theorem stating that the total cost increase is 6% -/
theorem cost_increase_is_six_percent :
  (totalCost (newCost initialRatio) - totalCost initialRatio) / totalCost initialRatio * 100 = 6 := by
  sorry


end cost_increase_is_six_percent_l258_25880


namespace right_triangle_area_l258_25874

theorem right_triangle_area (AB AC : ℝ) (h1 : AB = 12) (h2 : AC = 5) :
  let BC : ℝ := Real.sqrt (AB^2 - AC^2)
  (1 / 2) * AC * BC = (5 * Real.sqrt 119) / 2 := by sorry

end right_triangle_area_l258_25874


namespace original_price_calculation_l258_25838

/-- Given an article sold for $115 with a 15% gain, prove that the original price was $100 --/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 115 ∧ gain_percent = 15 → 
  ∃ (original_price : ℝ), 
    original_price = 100 ∧ 
    selling_price = original_price * (1 + gain_percent / 100) :=
by sorry

end original_price_calculation_l258_25838


namespace female_student_fraction_l258_25827

theorem female_student_fraction :
  ∀ (f m : ℝ),
  f + m = 1 →
  (5/6 : ℝ) * f + (2/3 : ℝ) * m = 0.7333333333333333 →
  f = 0.4 := by
sorry

end female_student_fraction_l258_25827


namespace no_solution_linear_system_l258_25899

theorem no_solution_linear_system :
  ¬ ∃ (x y z : ℝ),
    (3 * x - 4 * y + z = 10) ∧
    (6 * x - 8 * y + 2 * z = 5) ∧
    (2 * x - y - z = 4) :=
by
  sorry

end no_solution_linear_system_l258_25899


namespace larger_number_proof_l258_25816

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1370) (h3 : L = 6 * S + 15) : L = 1641 := by
  sorry

end larger_number_proof_l258_25816


namespace overtime_rate_is_five_l258_25829

/-- Calculates the overtime pay rate given daily wage, total earnings, days worked, and overtime hours. -/
def overtime_pay_rate (daily_wage : ℚ) (total_earnings : ℚ) (days_worked : ℕ) (overtime_hours : ℕ) : ℚ :=
  (total_earnings - daily_wage * days_worked) / overtime_hours

/-- Proves that given the conditions, the overtime pay rate is $5 per hour. -/
theorem overtime_rate_is_five :
  let daily_wage : ℚ := 150
  let total_earnings : ℚ := 770
  let days_worked : ℕ := 5
  let overtime_hours : ℕ := 4
  overtime_pay_rate daily_wage total_earnings days_worked overtime_hours = 5 := by
  sorry

#eval overtime_pay_rate 150 770 5 4

end overtime_rate_is_five_l258_25829


namespace kellys_sister_visit_l258_25837

def vacation_length : ℕ := 3 * 7

def travel_days : ℕ := 1 + 1 + 2 + 2

def grandparents_days : ℕ := 5

def brother_days : ℕ := 5

theorem kellys_sister_visit (sister_days : ℕ) : 
  sister_days = vacation_length - (travel_days + grandparents_days + brother_days) → 
  sister_days = 5 := by
  sorry

end kellys_sister_visit_l258_25837


namespace unique_solution_l258_25888

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that f satisfies for all positive integers m and n -/
def SatisfiesEquation (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (f (f m)^2 + 2 * (f n)^2) = m^2 + 2 * n^2

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := λ n => n

/-- The theorem stating that the identity function is the only one satisfying the equation -/
theorem unique_solution :
  ∀ f : PositiveIntFunction, SatisfiesEquation f ↔ f = identityFunction :=
sorry

end unique_solution_l258_25888


namespace haman_initial_trays_l258_25872

/-- Represents the number of eggs in a standard tray -/
def eggs_per_tray : ℕ := 30

/-- Represents the number of trays Haman dropped -/
def dropped_trays : ℕ := 2

/-- Represents the number of trays added after the accident -/
def added_trays : ℕ := 7

/-- Represents the total number of eggs sold -/
def total_eggs_sold : ℕ := 540

/-- Theorem stating that Haman initially collected 13 trays of eggs -/
theorem haman_initial_trays :
  (total_eggs_sold / eggs_per_tray - added_trays + dropped_trays : ℕ) = 13 := by
  sorry

end haman_initial_trays_l258_25872


namespace units_digit_of_expression_l258_25873

theorem units_digit_of_expression : 
  (3 * 19 * 1981 - 3^4) % 10 = 6 := by
  sorry

end units_digit_of_expression_l258_25873


namespace sqrt3_times_3_minus_sqrt3_range_l258_25810

theorem sqrt3_times_3_minus_sqrt3_range :
  2 < Real.sqrt 3 * (3 - Real.sqrt 3) ∧ Real.sqrt 3 * (3 - Real.sqrt 3) < 3 := by
  sorry

end sqrt3_times_3_minus_sqrt3_range_l258_25810


namespace inequality_proof_l258_25882

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end inequality_proof_l258_25882


namespace product_closest_to_2500_l258_25803

def options : List ℝ := [2500, 2600, 250, 260, 25000]

def product : ℝ := 0.0003125 * 8125312

theorem product_closest_to_2500 : 
  ∀ x ∈ options, |product - 2500| ≤ |product - x| :=
sorry

end product_closest_to_2500_l258_25803


namespace semicircle_perimeter_equilateral_triangle_l258_25855

/-- The perimeter of a region formed by three semicircular arcs,
    each constructed on a side of an equilateral triangle with side length 1,
    is equal to 3π/2. -/
theorem semicircle_perimeter_equilateral_triangle :
  let triangle_side_length : ℝ := 1
  let semicircle_radius : ℝ := triangle_side_length / 2
  let num_sides : ℕ := 3
  let perimeter : ℝ := num_sides * (π * semicircle_radius)
  perimeter = 3 * π / 2 := by sorry

end semicircle_perimeter_equilateral_triangle_l258_25855


namespace intersection_of_M_and_N_l258_25806

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2*x)}
def N : Set ℝ := {y | ∃ x, y = Real.exp x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x < 1/2} :=
sorry

end intersection_of_M_and_N_l258_25806


namespace trader_profit_above_goal_l258_25850

theorem trader_profit_above_goal 
  (profit : ℝ) 
  (required_amount : ℝ) 
  (donation : ℝ) 
  (half_profit : ℝ) 
  (h1 : profit = 960) 
  (h2 : required_amount = 610) 
  (h3 : donation = 310) 
  (h4 : half_profit = profit / 2) : 
  half_profit + donation - required_amount = 180 := by
sorry

end trader_profit_above_goal_l258_25850


namespace tangent_slope_at_point_one_l258_25834

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = 3 ∧ f' 1 = 1 :=
by sorry

end tangent_slope_at_point_one_l258_25834


namespace exists_winning_strategy_l258_25854

/-- Represents the color of a hat -/
inductive HatColor
| Black
| White

/-- Represents an agent's guess about their own hat color -/
def Guess := HatColor

/-- Represents a strategy function that takes the observed hat color and returns a guess -/
def Strategy := HatColor → Guess

/-- Represents the outcome of applying strategies to a pair of hat colors -/
def Outcome (c1 c2 : HatColor) (s1 s2 : Strategy) : Prop :=
  (s1 c2 = c1) ∨ (s2 c1 = c2)

/-- Theorem stating that there exists a pair of strategies that guarantees
    at least one correct guess for any combination of hat colors -/
theorem exists_winning_strategy :
  ∃ (s1 s2 : Strategy), ∀ (c1 c2 : HatColor), Outcome c1 c2 s1 s2 := by
  sorry

end exists_winning_strategy_l258_25854


namespace house_transaction_net_worth_change_l258_25841

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  houseValue : Int

/-- Calculates the net worth of a person -/
def netWorth (state : FinancialState) : Int :=
  state.cash + state.houseValue

/-- Represents a house transaction between two people -/
def houseTransaction (buyer seller : FinancialState) (price : Int) : (FinancialState × FinancialState) :=
  ({ cash := buyer.cash - price, houseValue := seller.houseValue },
   { cash := seller.cash + price, houseValue := 0 })

theorem house_transaction_net_worth_change 
  (initialA initialB : FinancialState)
  (houseValue firstPrice secondPrice : Int) :
  initialA.cash = 15000 →
  initialA.houseValue = 12000 →
  initialB.cash = 13000 →
  initialB.houseValue = 0 →
  houseValue = 12000 →
  firstPrice = 14000 →
  secondPrice = 10000 →
  let (afterFirstA, afterFirstB) := houseTransaction initialB initialA firstPrice
  let (finalB, finalA) := houseTransaction afterFirstA afterFirstB secondPrice
  netWorth finalA - netWorth initialA = 4000 ∧
  netWorth finalB - netWorth initialB = -4000 := by sorry


end house_transaction_net_worth_change_l258_25841


namespace completing_square_transformation_l258_25846

/-- Given a quadratic equation x^2 - 2x - 4 = 0, prove that when transformed
    into the form (x-1)^2 = a using the completing the square method,
    the value of a is 5. -/
theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 4 = 0) → ∃ a : ℝ, ((x - 1)^2 = a) ∧ (a = 5) := by
  sorry

end completing_square_transformation_l258_25846


namespace square_ratios_area_ratio_diagonal_ratio_l258_25885

/-- Given two squares where the perimeter of one is 4 times the other, 
    prove the ratios of their areas and diagonals -/
theorem square_ratios (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a ^ 2 = 16 * b ^ 2 ∧ a * Real.sqrt 2 = 4 * (b * Real.sqrt 2) := by
  sorry

/-- The area of the larger square is 16 times the area of the smaller square -/
theorem area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a ^ 2 = 16 * b ^ 2 := by
  sorry

/-- The diagonal of the larger square is 4 times the diagonal of the smaller square -/
theorem diagonal_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a * Real.sqrt 2 = 4 * (b * Real.sqrt 2) := by
  sorry

end square_ratios_area_ratio_diagonal_ratio_l258_25885


namespace min_value_problem_l258_25862

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + 3 * b = 6) :
  (3 / a + 2 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 6 ∧ 3 / a₀ + 2 / b₀ = 4 :=
by sorry

end min_value_problem_l258_25862


namespace circle_equation_l258_25815

-- Define the point P
def P : ℝ × ℝ := (-2, 1)

-- Define the line y = x + 1
def line1 (x y : ℝ) : Prop := y = x + 1

-- Define the line 3x + 4y - 11 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the circle C
def circle_C (c : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = 18

-- Define the symmetry condition
def symmetric_point (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ 
  (p1.1 + p2.1) / 2 = x ∧ 
  (p1.2 + p2.2) / 2 = y

-- Define the intersection condition
def intersects (c : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), 
  circle_C c A.1 A.2 ∧ 
  circle_C c B.1 B.2 ∧
  line2 A.1 A.2 ∧ 
  line2 B.1 B.2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36

-- Theorem statement
theorem circle_equation (c : ℝ × ℝ) :
  symmetric_point P c ∧ intersects c →
  ∀ (x y : ℝ), circle_C c x y ↔ x^2 + (y+1)^2 = 18 :=
sorry

end circle_equation_l258_25815


namespace triangle_side_length_l258_25848

theorem triangle_side_length
  (a b c : ℝ)
  (A : ℝ)
  (area : ℝ)
  (h1 : a + b + c = 20)
  (h2 : area = 10 * Real.sqrt 3)
  (h3 : A = π / 3) :
  a = 7 := by
  sorry

end triangle_side_length_l258_25848


namespace square_area_with_four_circles_l258_25826

theorem square_area_with_four_circles (r : ℝ) (h : r = 8) : 
  (2 * (2 * r))^2 = 1024 := by
  sorry

end square_area_with_four_circles_l258_25826


namespace sum_of_solutions_is_two_l258_25833

theorem sum_of_solutions_is_two :
  ∃ (x y : ℤ), x^2 = x + 224 ∧ y^2 = y + 224 ∧ x + y = 2 ∧
  ∀ (z : ℤ), z^2 = z + 224 → z = x ∨ z = y :=
sorry

end sum_of_solutions_is_two_l258_25833


namespace mechanism_composition_l258_25884

/-- Represents a mechanism with small and large parts. -/
structure Mechanism where
  total_parts : ℕ
  small_parts : ℕ
  large_parts : ℕ
  total_eq : total_parts = small_parts + large_parts

/-- Property: Among any 12 parts, there is at least one small part. -/
def has_small_in_12 (m : Mechanism) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 12 → (∃ (x : ℕ), x ∈ subset ∧ x ≤ m.small_parts)

/-- Property: Among any 15 parts, there is at least one large part. -/
def has_large_in_15 (m : Mechanism) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 15 → (∃ (x : ℕ), x ∈ subset ∧ x > m.small_parts)

/-- Main theorem: If a mechanism satisfies the given conditions, it has 11 large parts and 14 small parts. -/
theorem mechanism_composition (m : Mechanism) 
    (h_total : m.total_parts = 25)
    (h_small : has_small_in_12 m)
    (h_large : has_large_in_15 m) : 
    m.large_parts = 11 ∧ m.small_parts = 14 := by
  sorry


end mechanism_composition_l258_25884


namespace female_kittens_count_l258_25866

theorem female_kittens_count (initial_cats : ℕ) (total_cats : ℕ) (male_kittens : ℕ) : 
  initial_cats = 2 → total_cats = 7 → male_kittens = 2 → 
  total_cats - initial_cats - male_kittens = 3 := by
sorry

end female_kittens_count_l258_25866


namespace sum_of_solutions_eq_twelve_l258_25878

theorem sum_of_solutions_eq_twelve : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 6)^2 = 50 ∧ (x₂ - 6)^2 = 50 ∧ x₁ + x₂ = 12 := by
  sorry

end sum_of_solutions_eq_twelve_l258_25878


namespace danny_soda_remaining_l258_25881

theorem danny_soda_remaining (bottles : ℕ) (consumed : ℚ) (given_away : ℚ) : 
  bottles = 3 → 
  consumed = 9/10 → 
  given_away = 7/10 → 
  (1 - consumed) + 2 * (1 - given_away) = 7/10 :=
by sorry

end danny_soda_remaining_l258_25881


namespace tree_height_l258_25825

/-- Given Jane's height, Jane's shadow length, and the tree's shadow length,
    prove that the tree's height is 30 meters. -/
theorem tree_height (jane_height jane_shadow tree_shadow : ℝ)
  (h1 : jane_height = 1.5)
  (h2 : jane_shadow = 0.5)
  (h3 : tree_shadow = 10)
  (h4 : ∀ (obj1 obj2 : ℝ), obj1 / jane_shadow = jane_height / jane_shadow → obj1 / obj2 = jane_height / jane_shadow) :
  jane_height / jane_shadow * tree_shadow = 30 := by
  sorry

end tree_height_l258_25825


namespace train_speed_crossing_bridge_l258_25830

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 25) :
  (train_length + bridge_length) / crossing_time = 16 :=
by sorry

end train_speed_crossing_bridge_l258_25830


namespace initial_peaches_count_l258_25842

/-- Represents the state of the fruit bowl on a given day -/
structure FruitBowl :=
  (day : Nat)
  (ripe : Nat)
  (unripe : Nat)

/-- Updates the fruit bowl state for the next day -/
def nextDay (bowl : FruitBowl) : FruitBowl :=
  let newRipe := bowl.ripe + 2
  let newUnripe := bowl.unripe - 2
  { day := bowl.day + 1, ripe := newRipe, unripe := newUnripe }

/-- Represents eating 3 peaches on day 3 -/
def eatPeaches (bowl : FruitBowl) : FruitBowl :=
  { bowl with ripe := bowl.ripe - 3 }

/-- The initial state of the fruit bowl -/
def initialBowl : FruitBowl :=
  { day := 0, ripe := 4, unripe := 13 }

/-- The final state of the fruit bowl after 5 days -/
def finalBowl : FruitBowl :=
  (nextDay ∘ nextDay ∘ eatPeaches ∘ nextDay ∘ nextDay ∘ nextDay) initialBowl

/-- Theorem stating that the initial number of peaches was 17 -/
theorem initial_peaches_count :
  initialBowl.ripe + initialBowl.unripe = 17 ∧
  finalBowl.ripe = finalBowl.unripe + 7 :=
by sorry


end initial_peaches_count_l258_25842


namespace train_length_l258_25863

/-- Given a train crossing a pole at a speed of 60 km/hr in 18 seconds,
    prove that the length of the train is 300 meters. -/
theorem train_length (speed : ℝ) (time_seconds : ℝ) (length : ℝ) :
  speed = 60 →
  time_seconds = 18 →
  length = speed * (time_seconds / 3600) * 1000 →
  length = 300 := by sorry

end train_length_l258_25863


namespace quadratic_no_intersection_l258_25823

/-- A quadratic function that doesn't intersect the x-axis has c > 1 -/
theorem quadratic_no_intersection (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + c ≠ 0) → c > 1 := by
sorry

end quadratic_no_intersection_l258_25823


namespace unique_prime_generating_x_l258_25889

theorem unique_prime_generating_x (x : ℕ+) 
  (h : Nat.Prime (x^5 + x + 1)) : x = 1 := by
  sorry

end unique_prime_generating_x_l258_25889


namespace f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l258_25817

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 :=
sorry

end f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l258_25817


namespace expression_evaluation_inequality_system_solution_l258_25801

-- Part 1
theorem expression_evaluation :
  Real.sqrt 12 + |Real.sqrt 3 - 2| - 2 * Real.tan (60 * π / 180) + (1/3)⁻¹ = 5 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (x + 3 * (x - 2) ≥ 2 ∧ (1 + 2 * x) / 3 > x - 1) ↔ (2 ≤ x ∧ x < 3) := by
  sorry

end expression_evaluation_inequality_system_solution_l258_25801


namespace product_of_numbers_l258_25820

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 21) (sum_squares_eq : x^2 + y^2 = 527) :
  x * y = -43 := by sorry

end product_of_numbers_l258_25820


namespace equal_sums_l258_25865

-- Define the range of numbers
def N : ℕ := 999999

-- Function to determine if a number's nearest perfect square is odd
def nearest_square_odd (n : ℕ) : Prop := sorry

-- Function to determine if a number's nearest perfect square is even
def nearest_square_even (n : ℕ) : Prop := sorry

-- Sum of numbers with odd nearest perfect square
def sum_odd_group : ℕ := sorry

-- Sum of numbers with even nearest perfect square
def sum_even_group : ℕ := sorry

-- Theorem stating that the sums are equal
theorem equal_sums : sum_odd_group = sum_even_group := by sorry

end equal_sums_l258_25865


namespace function_properties_l258_25847

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < -Real.exp 1) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) :
  let t := Real.sqrt (x₂ / x₁)
  (deriv (f a)) ((3 * x₁ + x₂) / 4) < 0 ∧ 
  (t - 1) * (a + Real.sqrt 3) = -2 * Real.sqrt 3 := by
sorry

end function_properties_l258_25847


namespace inequality_solution_implies_m_negative_l258_25805

/-- 
Given a real number m, prove that if the solution set of the inequality 
(mx-1)(x-2) > 0 is {x | 1/m < x < 2}, then m < 0.
-/
theorem inequality_solution_implies_m_negative (m : ℝ) : 
  (∀ x, (m * x - 1) * (x - 2) > 0 ↔ 1/m < x ∧ x < 2) → m < 0 := by
  sorry

end inequality_solution_implies_m_negative_l258_25805


namespace karina_to_brother_age_ratio_l258_25836

-- Define the given information
def karina_birth_year : ℕ := 1970
def karina_current_age : ℕ := 40
def brother_birth_year : ℕ := 1990

-- Define the current year based on Karina's age
def current_year : ℕ := karina_birth_year + karina_current_age

-- Calculate brother's age
def brother_current_age : ℕ := current_year - brother_birth_year

-- Theorem to prove
theorem karina_to_brother_age_ratio :
  (karina_current_age : ℚ) / (brother_current_age : ℚ) = 2 := by
  sorry

end karina_to_brother_age_ratio_l258_25836


namespace triangle_circles_radius_sum_l258_25886

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (AC : ℝ) (BC : ℝ)

/-- Circle with given radius -/
structure Circle :=
  (radius : ℝ)

/-- Represents the radius of circle Q in the form m - n√k -/
structure RadiusForm :=
  (m : ℕ) (n : ℕ) (k : ℕ)

/-- Main theorem statement -/
theorem triangle_circles_radius_sum (ABC : Triangle) (P Q : Circle) (r : RadiusForm) :
  ABC.AB = 130 →
  ABC.AC = 130 →
  ABC.BC = 78 →
  P.radius = 25 →
  -- Circle P is tangent to AC and BC
  -- Circle Q is externally tangent to P and tangent to AB and BC
  -- No point of circle Q lies outside of triangle ABC
  Q.radius = r.m - r.n * Real.sqrt r.k →
  r.m > 0 →
  r.n > 0 →
  r.k > 0 →
  -- k is the product of distinct primes
  r.m + r.n * r.k = 131 := by
  sorry

end triangle_circles_radius_sum_l258_25886


namespace aquarium_width_l258_25807

theorem aquarium_width (length height : ℝ) (volume_final : ℝ) : 
  length = 4 → height = 3 → volume_final = 54 → 
  ∃ (width : ℝ), 3 * ((length * width * height) / 4) = volume_final ∧ width = 6 := by
sorry

end aquarium_width_l258_25807


namespace no_real_roots_x_squared_plus_five_l258_25812

theorem no_real_roots_x_squared_plus_five :
  ∀ x : ℝ, x^2 + 5 ≠ 0 := by
  sorry

end no_real_roots_x_squared_plus_five_l258_25812


namespace point_distance_product_l258_25814

theorem point_distance_product : ∃ y₁ y₂ : ℝ,
  ((-1 - 4)^2 + (y₁ - 3)^2 = 8^2) ∧
  ((-1 - 4)^2 + (y₂ - 3)^2 = 8^2) ∧
  y₁ ≠ y₂ ∧
  y₁ * y₂ = -30 := by
sorry

end point_distance_product_l258_25814


namespace perpendicular_line_parallel_planes_l258_25832

structure Plane where
  -- Define a plane

structure Line where
  -- Define a line

def perpendicular (l : Line) (p : Plane) : Prop :=
  -- Define what it means for a line to be perpendicular to a plane
  sorry

def parallel (p1 p2 : Plane) : Prop :=
  -- Define what it means for two planes to be parallel
  sorry

def contains (p : Plane) (l : Line) : Prop :=
  -- Define what it means for a plane to contain a line
  sorry

def perpendicular_lines (l1 l2 : Line) : Prop :=
  -- Define what it means for two lines to be perpendicular
  sorry

theorem perpendicular_line_parallel_planes 
  (m : Line) (n : Line) (α β : Plane) :
  perpendicular m α → contains β n → parallel α β → perpendicular_lines m n :=
by sorry

end perpendicular_line_parallel_planes_l258_25832


namespace fifth_term_of_arithmetic_sequence_l258_25892

def arithmeticSequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℤ) (d : ℤ) 
  (h12 : arithmeticSequence a d 12 = 25)
  (h13 : arithmeticSequence a d 13 = 29) :
  arithmeticSequence a d 5 = -3 := by
sorry

end fifth_term_of_arithmetic_sequence_l258_25892


namespace best_meeting_days_l258_25896

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define the team members
inductive Member
| Alice
| Bob
| Cindy
| Dave
| Eve

-- Define the availability function
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Alice, Day.Saturday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Cindy, Day.Wednesday => false
  | Member.Cindy, Day.Saturday => false
  | Member.Dave, Day.Monday => false
  | Member.Dave, Day.Tuesday => false
  | Member.Dave, Day.Thursday => false
  | Member.Eve, Day.Thursday => false
  | Member.Eve, Day.Friday => false
  | Member.Eve, Day.Saturday => false
  | _, _ => true

-- Define the function to count available members on a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cindy, Member.Dave, Member.Eve]).length

-- Theorem statement
theorem best_meeting_days :
  (∀ d : Day, availableCount d ≤ 3) ∧
  (availableCount Day.Monday = 3) ∧
  (availableCount Day.Tuesday = 3) ∧
  (availableCount Day.Wednesday = 3) ∧
  (availableCount Day.Friday = 3) ∧
  (∀ d : Day, availableCount d = 3 → d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Friday) :=
by sorry


end best_meeting_days_l258_25896


namespace b_sequence_max_at_4_l258_25800

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sequence_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

def b_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := (1 + arithmetic_sequence a₁ d n) / arithmetic_sequence a₁ d n

theorem b_sequence_max_at_4 (a₁ d : ℚ) (h₁ : a₁ = -5/2) (h₂ : sequence_sum a₁ d 4 = 2 * sequence_sum a₁ d 2 + 4) :
  ∀ n : ℕ, n ≥ 1 → b_sequence a₁ d 4 ≥ b_sequence a₁ d n :=
sorry

end b_sequence_max_at_4_l258_25800


namespace line_and_chord_problem_l258_25879

-- Define the circle M
def circle_M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l
def line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}

-- Define the midpoint P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

theorem line_and_chord_problem :
  point_P = ((point_A.1 + point_B.1) / 2, (point_A.2 + point_B.2) / 2) ∧
  point_A ∈ circle_M ∧ point_B ∈ circle_M ∧
  point_A ∈ line_l ∧ point_B ∈ line_l →
  (∀ p : ℝ × ℝ, p ∈ line_l ↔ p.1 + p.2 = 2) ∧
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 2 * Real.sqrt 2 :=
sorry

end line_and_chord_problem_l258_25879


namespace expression_evaluation_l258_25857

theorem expression_evaluation : (3^2 - 1) - (4^2 - 2) + (5^2 - 3) = 16 := by
  sorry

end expression_evaluation_l258_25857


namespace paint_combinations_count_l258_25867

/-- The number of available paint colors -/
def num_colors : ℕ := 6

/-- The number of available painting tools -/
def num_tools : ℕ := 4

/-- The number of combinations of color and different tools for two objects -/
def num_combinations : ℕ := num_colors * num_tools * (num_tools - 1)

theorem paint_combinations_count :
  num_combinations = 72 := by
  sorry

end paint_combinations_count_l258_25867


namespace cos_alpha_value_l258_25809

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 3 / 3) : 
  Real.cos α = (-2 * Real.sqrt 3 + Real.sqrt 6) / 6 := by
  sorry

end cos_alpha_value_l258_25809


namespace arithmetic_equation_l258_25893

theorem arithmetic_equation : 12.1212 + 17.0005 - 9.1103 = 20.0114 := by
  sorry

end arithmetic_equation_l258_25893


namespace contrapositive_rhombus_diagonals_l258_25808

-- Define a quadrilateral type
structure Quadrilateral where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the property of being a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define the property of having perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

-- State the theorem
theorem contrapositive_rhombus_diagonals :
  (∀ q : Quadrilateral, ¬(has_perpendicular_diagonals q) → ¬(is_rhombus q)) ↔
  (∀ q : Quadrilateral, is_rhombus q → has_perpendicular_diagonals q) :=
by sorry

end contrapositive_rhombus_diagonals_l258_25808


namespace square_cardinality_continuum_l258_25898

/-- A square in the 2D plane -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- The unit interval [0, 1] -/
def UnitInterval : Set ℝ :=
  {x | 0 ≤ x ∧ x ≤ 1}

theorem square_cardinality_continuum :
  Cardinal.mk (Square) = Cardinal.mk (UnitInterval) :=
sorry

end square_cardinality_continuum_l258_25898


namespace is_projection_matrix_l258_25822

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

theorem is_projection_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![20/49, 20/49; 20/49, 29/49]
  projection_matrix A := by
  sorry

end is_projection_matrix_l258_25822


namespace river_flow_volume_l258_25819

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 12) 
  (h_width : width = 35) 
  (h_flow_rate : flow_rate_kmph = 8) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 56000 := by
  sorry

end river_flow_volume_l258_25819


namespace probability_of_purple_marble_l258_25853

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) :
  p_blue = 0.25 →
  p_green = 0.4 →
  p_blue + p_green + p_purple = 1 →
  p_purple = 0.35 := by
sorry

end probability_of_purple_marble_l258_25853


namespace second_guide_children_l258_25844

/-- Given information about zoo guides and children -/
structure ZooTour where
  total_children : ℕ
  first_guide_children : ℕ

/-- Theorem: The second guide spoke to 25 children -/
theorem second_guide_children (tour : ZooTour) 
  (h1 : tour.total_children = 44)
  (h2 : tour.first_guide_children = 19) :
  tour.total_children - tour.first_guide_children = 25 := by
  sorry

#eval 44 - 19  -- Expected output: 25

end second_guide_children_l258_25844


namespace number_of_lineups_l258_25876

def team_size : ℕ := 15
def lineup_size : ℕ := 5

def cannot_play_together : Prop := true
def at_least_one_must_play : Prop := true

theorem number_of_lineups : 
  ∃ (n : ℕ), n = Nat.choose (team_size - 2) (lineup_size - 1) * 2 + 
             Nat.choose (team_size - 3) (lineup_size - 2) ∧
  n = 1210 := by
  sorry

end number_of_lineups_l258_25876


namespace general_admission_ticket_cost_l258_25804

theorem general_admission_ticket_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (general_admission_tickets : ℕ)
  (student_ticket_cost : ℕ)
  (h1 : total_tickets = 525)
  (h2 : total_revenue = 2876)
  (h3 : general_admission_tickets = 388)
  (h4 : student_ticket_cost = 4) :
  ∃ (general_admission_cost : ℕ),
    general_admission_cost * general_admission_tickets +
    student_ticket_cost * (total_tickets - general_admission_tickets) =
    total_revenue ∧
    general_admission_cost = 6 :=
by sorry

end general_admission_ticket_cost_l258_25804


namespace negation_of_p_l258_25856

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) : 
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end negation_of_p_l258_25856


namespace a_minus_b_bounds_l258_25831

theorem a_minus_b_bounds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := by sorry

end a_minus_b_bounds_l258_25831


namespace photo_collection_l258_25852

theorem photo_collection (total photos : ℕ) (tim paul tom : ℕ) : 
  total = 152 →
  tim = total - 100 →
  paul = tim + 10 →
  total = tim + paul + tom →
  tom = 38 := by
sorry

end photo_collection_l258_25852


namespace liar_proportion_is_half_l258_25839

/-- Represents the proportion of liars in a village -/
def proportion_of_liars : ℝ := sorry

/-- The proportion of liars is between 0 and 1 -/
axiom proportion_bounds : 0 ≤ proportion_of_liars ∧ proportion_of_liars ≤ 1

/-- The proportion of liars is indistinguishable from the proportion of truth-tellers when roles are reversed -/
axiom indistinguishable_proportion : proportion_of_liars = 1 - proportion_of_liars

theorem liar_proportion_is_half : proportion_of_liars = 1/2 := by sorry

end liar_proportion_is_half_l258_25839


namespace cube_sum_from_sum_and_product_l258_25859

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end cube_sum_from_sum_and_product_l258_25859


namespace triangle_abc_proof_l258_25883

theorem triangle_abc_proof (a b c : ℝ) (A B : ℝ) :
  a = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  B = 45 * (π / 180) →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  b = 2 * Real.sqrt 2 ∧ A = π / 3 := by
  sorry


end triangle_abc_proof_l258_25883


namespace probability_between_C_and_D_l258_25858

/-- Given a line segment AB with points A, B, C, D, and E, prove that the probability
    of a randomly selected point on AB being between C and D is 1/2. -/
theorem probability_between_C_and_D (A B C D E : ℝ) : 
  A < B ∧ 
  B - A = 4 * (E - A) ∧
  B - A = 8 * (B - D) ∧
  D - A = 3 * (E - A) ∧
  B - D = 5 * (B - E) ∧
  C = D + (1/8) * (B - A) →
  (C - D) / (B - A) = 1/2 := by
sorry

end probability_between_C_and_D_l258_25858


namespace contractor_payment_l258_25813

/-- A contractor's payment calculation --/
theorem contractor_payment
  (total_days : ℕ)
  (work_pay : ℚ)
  (fine : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : work_pay = 25)
  (h3 : fine = 7.5)
  (h4 : absent_days = 4)
  : (total_days - absent_days : ℚ) * work_pay - (absent_days : ℚ) * fine = 620 := by
  sorry

end contractor_payment_l258_25813


namespace course_size_l258_25818

theorem course_size (total : ℕ) 
  (h1 : (3 : ℚ) / 10 * total + (3 : ℚ) / 10 * total + (2 : ℚ) / 10 * total + 
        (1 : ℚ) / 10 * total + 12 + 5 = total) : 
  total = 170 := by
  sorry

end course_size_l258_25818


namespace expected_male_athletes_expected_male_athletes_eq_twelve_l258_25835

/-- Given a team of athletes with a specific male-to-total ratio,
    calculate the expected number of male athletes in a stratified sample. -/
theorem expected_male_athletes 
  (total_athletes : ℕ) 
  (male_ratio : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = 84) 
  (h2 : male_ratio = 4/7) 
  (h3 : sample_size = 21) : 
  ℕ := by
  sorry

#check expected_male_athletes

theorem expected_male_athletes_eq_twelve 
  (total_athletes : ℕ) 
  (male_ratio : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = 84) 
  (h2 : male_ratio = 4/7) 
  (h3 : sample_size = 21) : 
  expected_male_athletes total_athletes male_ratio sample_size h1 h2 h3 = 12 := by
  sorry

end expected_male_athletes_expected_male_athletes_eq_twelve_l258_25835


namespace perimeter_gt_four_times_circumradius_l258_25870

/-- An acute-angled triangle with its perimeter and circumradius -/
structure AcuteTriangle where
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The circumradius of the triangle
  circumradius : ℝ
  -- Condition ensuring the triangle is acute-angled (this is a simplification)
  is_acute : perimeter > 0 ∧ circumradius > 0

/-- Theorem stating that for any acute-angled triangle, its perimeter is greater than 4 times its circumradius -/
theorem perimeter_gt_four_times_circumradius (t : AcuteTriangle) : t.perimeter > 4 * t.circumradius := by
  sorry


end perimeter_gt_four_times_circumradius_l258_25870


namespace odds_against_third_horse_l258_25894

/-- Represents the probability of a horse winning a race -/
def probability (p q : ℚ) : ℚ := q / (p + q)

/-- Given three horses in a race with no ties, calculates the odds against the third horse winning -/
theorem odds_against_third_horse 
  (prob_x prob_y : ℚ) 
  (hx : prob_x = probability 3 1) 
  (hy : prob_y = probability 2 3) 
  (h_sum : prob_x + prob_y < 1) :
  ∃ (p q : ℚ), p / q = 17 / 3 ∧ probability p q = 1 - prob_x - prob_y := by
sorry


end odds_against_third_horse_l258_25894


namespace four_numbers_with_one_sixth_property_l258_25840

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The three-digit number obtained by removing the leftmost digit of a four-digit number -/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- The property that the three-digit number obtained by removing the leftmost digit is one sixth of the original number -/
def HasOneSixthProperty (n : ℕ) : Prop :=
  FourDigitNumber n ∧ RemoveLeftmostDigit n = n / 6

/-- The theorem stating that there are exactly 4 numbers satisfying the property -/
theorem four_numbers_with_one_sixth_property :
  ∃! (s : Finset ℕ), s.card = 4 ∧ ∀ n, n ∈ s ↔ HasOneSixthProperty n :=
sorry

end four_numbers_with_one_sixth_property_l258_25840


namespace solution_sum_l258_25828

theorem solution_sum (x y : ℤ) : 
  (x : ℝ) * Real.log 27 * (Real.log 13)⁻¹ = 27 * Real.log y / Real.log 13 →
  y > 70 →
  ∀ z, z > 70 → z < y →
  x + y = 117 := by
sorry

end solution_sum_l258_25828


namespace parabola_reflection_sum_l258_25845

theorem parabola_reflection_sum (a b c : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x + c + 3
  let g := fun x : ℝ => -a * x^2 - b * x - c - 3
  ∀ x, f x + g x = 0 := by sorry

end parabola_reflection_sum_l258_25845


namespace three_hour_charge_l258_25887

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price of the first hour
  additional_hour : ℕ  -- Price of each additional hour
  first_hour_premium : first_hour = additional_hour + 30  -- First hour costs $30 more
  five_hour_total : first_hour + 4 * additional_hour = 400  -- Total for 5 hours is $400

/-- Theorem stating that given the pricing structure, the total charge for 3 hours is $252. -/
theorem three_hour_charge (p : TherapyPricing) : 
  p.first_hour + 2 * p.additional_hour = 252 := by
  sorry


end three_hour_charge_l258_25887


namespace product_trailing_zeros_l258_25802

/-- The number of trailing zeros in n -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 20 raised to the power of 50 -/
def a : ℕ := 20^50

/-- 50 raised to the power of 20 -/
def b : ℕ := 50^20

/-- The main theorem stating that the number of trailing zeros
    in the product of 20^50 and 50^20 is 90 -/
theorem product_trailing_zeros : trailingZeros (a * b) = 90 := by sorry

end product_trailing_zeros_l258_25802


namespace hyperbola_equation_l258_25864

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1

-- Define the properties of the hyperbola
def has_focus (h : Hyperbola) (fx fy : ℝ) : Prop :=
  h.a^2 + h.b^2 = fx^2 + fy^2

def passes_through (h : Hyperbola) (px py : ℝ) : Prop :=
  h.eq px py

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  has_focus h (Real.sqrt 6) 0 →
  passes_through h (-5) 2 →
  h.a^2 = 5 ∧ h.b^2 = 1 :=
sorry

end hyperbola_equation_l258_25864


namespace adjacent_probability_l258_25890

/-- Represents the number of students in the group photo --/
def total_students : ℕ := 6

/-- Represents the number of rows in the seating arrangement --/
def num_rows : ℕ := 3

/-- Represents the number of seats per row --/
def seats_per_row : ℕ := 2

/-- Calculates the total number of seating arrangements --/
def total_arrangements : ℕ := Nat.factorial total_students

/-- Calculates the number of favorable arrangements where Abby and Bridget are adjacent but not in the middle row --/
def favorable_arrangements : ℕ := 4 * 2 * Nat.factorial (total_students - 2)

/-- Represents the probability of Abby and Bridget being adjacent but not in the middle row --/
def probability : ℚ := favorable_arrangements / total_arrangements

/-- Theorem stating that the probability of Abby and Bridget being adjacent but not in the middle row is 4/15 --/
theorem adjacent_probability :
  probability = 4 / 15 := by sorry

end adjacent_probability_l258_25890
