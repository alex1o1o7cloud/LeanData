import Mathlib

namespace highway_project_employees_l760_76039

/-- Represents the highway construction project -/
structure HighwayProject where
  initial_workforce : ℕ
  total_length : ℕ
  initial_days : ℕ
  initial_hours_per_day : ℕ
  days_worked : ℕ
  work_completed : ℚ
  remaining_days : ℕ
  new_hours_per_day : ℕ

/-- Calculates the number of additional employees needed to complete the project on time -/
def additional_employees_needed (project : HighwayProject) : ℕ :=
  sorry

/-- Theorem stating that 60 additional employees are needed for the given project -/
theorem highway_project_employees (project : HighwayProject) 
  (h1 : project.initial_workforce = 100)
  (h2 : project.total_length = 2)
  (h3 : project.initial_days = 50)
  (h4 : project.initial_hours_per_day = 8)
  (h5 : project.days_worked = 25)
  (h6 : project.work_completed = 1/3)
  (h7 : project.remaining_days = 25)
  (h8 : project.new_hours_per_day = 10) :
  additional_employees_needed project = 60 :=
sorry

end highway_project_employees_l760_76039


namespace intersection_point_l760_76044

def P : ℝ × ℝ × ℝ := (10, -1, 3)
def Q : ℝ × ℝ × ℝ := (20, -11, 8)
def R : ℝ × ℝ × ℝ := (3, 8, -9)
def S : ℝ × ℝ × ℝ := (5, 0, 6)

def line_PQ (t : ℝ) : ℝ × ℝ × ℝ :=
  (P.1 + t * (Q.1 - P.1), P.2.1 + t * (Q.2.1 - P.2.1), P.2.2 + t * (Q.2.2 - P.2.2))

def line_RS (s : ℝ) : ℝ × ℝ × ℝ :=
  (R.1 + s * (S.1 - R.1), R.2.1 + s * (S.2.1 - R.2.1), R.2.2 + s * (S.2.2 - R.2.2))

theorem intersection_point :
  ∃ t s : ℝ, line_PQ t = line_RS s ∧ line_PQ t = (11, -2, 3.5) := by sorry

end intersection_point_l760_76044


namespace shaded_area_is_14_l760_76046

/-- Represents the grid dimensions --/
structure GridDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle --/
def rectangleArea (w h : ℕ) : ℕ := w * h

/-- Calculates the area of a right-angled triangle --/
def triangleArea (base height : ℕ) : ℕ := base * height / 2

/-- Theorem stating that the shaded area in the grid is 14 square units --/
theorem shaded_area_is_14 (grid : GridDimensions) 
    (h1 : grid.width = 12)
    (h2 : grid.height = 4) : 
  rectangleArea grid.width grid.height - triangleArea grid.width grid.height = 14 := by
  sorry

end shaded_area_is_14_l760_76046


namespace sum_three_squares_to_four_fractions_l760_76074

theorem sum_three_squares_to_four_fractions (A B C : ℤ) :
  ∃ (x y z : ℝ), 
    (A : ℝ)^2 + (B : ℝ)^2 + (C : ℝ)^2 = 
      ((A * (x^2 + y^2 - z^2) + B * (2*x*z) + C * (2*y*z)) / (x^2 + y^2 + z^2))^2 +
      ((A * (2*x*z) - B * (x^2 + y^2 - z^2)) / (x^2 + y^2 + z^2))^2 +
      ((B * (2*y*z) - C * (2*x*z)) / (x^2 + y^2 + z^2))^2 +
      ((C * (x^2 + y^2 - z^2) - A * (2*y*z)) / (x^2 + y^2 + z^2))^2 :=
by sorry

end sum_three_squares_to_four_fractions_l760_76074


namespace paint_remaining_l760_76087

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  (initial_paint - initial_paint / 4) / 2 = 3 / 8 := by
sorry

end paint_remaining_l760_76087


namespace parabola_vertex_l760_76006

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The vertex coordinates of the parabola -/
def vertex : ℝ × ℝ := (3, -4)

/-- Theorem: The vertex coordinates of the parabola y = x^2 - 6x + 5 are (3, -4) -/
theorem parabola_vertex : 
  ∀ x : ℝ, parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 := by
  sorry

end parabola_vertex_l760_76006


namespace puppy_food_consumption_l760_76073

/-- Given the cost of a puppy, the duration of food supply, the amount and cost of food per bag,
    and the total cost, this theorem proves the daily food consumption of the puppy. -/
theorem puppy_food_consumption
  (puppy_cost : ℚ)
  (food_duration_weeks : ℕ)
  (food_per_bag : ℚ)
  (cost_per_bag : ℚ)
  (total_cost : ℚ)
  (h1 : puppy_cost = 10)
  (h2 : food_duration_weeks = 3)
  (h3 : food_per_bag = 7/2)
  (h4 : cost_per_bag = 2)
  (h5 : total_cost = 14) :
  (total_cost - puppy_cost) / cost_per_bag * food_per_bag / (food_duration_weeks * 7 : ℚ) = 1/3 :=
sorry

end puppy_food_consumption_l760_76073


namespace fraction_of_number_l760_76037

theorem fraction_of_number : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5020 = 753 := by
  sorry

end fraction_of_number_l760_76037


namespace mass_CO2_from_CO_combustion_l760_76047

/-- The mass of CO2 produced from the complete combustion of CO -/
def mass_CO2_produced (initial_moles_CO : ℝ) (molar_mass_CO2 : ℝ) : ℝ :=
  initial_moles_CO * molar_mass_CO2

/-- The balanced chemical reaction coefficient for CO2 -/
def CO2_coefficient : ℚ := 2

/-- The balanced chemical reaction coefficient for CO -/
def CO_coefficient : ℚ := 2

theorem mass_CO2_from_CO_combustion 
  (initial_moles_CO : ℝ)
  (molar_mass_CO2 : ℝ)
  (h1 : initial_moles_CO = 3)
  (h2 : molar_mass_CO2 = 44.01) :
  mass_CO2_produced initial_moles_CO molar_mass_CO2 = 132.03 := by
  sorry

end mass_CO2_from_CO_combustion_l760_76047


namespace complex_equation_solution_l760_76084

theorem complex_equation_solution :
  ∃ (x y : ℝ), (-5 + 2 * Complex.I) * x - (3 - 4 * Complex.I) * y = 2 - Complex.I ∧
  x = -5/14 ∧ y = -1/14 := by
sorry

end complex_equation_solution_l760_76084


namespace class_size_proof_l760_76088

theorem class_size_proof (avg_age : ℝ) (avg_age_5 : ℝ) (avg_age_9 : ℝ) (age_15th : ℕ) : 
  avg_age = 15 → 
  avg_age_5 = 14 → 
  avg_age_9 = 16 → 
  age_15th = 11 → 
  ∃ (N : ℕ), N = 15 ∧ N * avg_age = 5 * avg_age_5 + 9 * avg_age_9 + age_15th :=
by
  sorry

end class_size_proof_l760_76088


namespace negation_equivalence_l760_76081

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end negation_equivalence_l760_76081


namespace modulus_of_complex_power_l760_76000

theorem modulus_of_complex_power (z : ℂ) :
  z = 2 - 3 * Real.sqrt 2 * Complex.I →
  Complex.abs (z^4) = 484 := by
sorry

end modulus_of_complex_power_l760_76000


namespace trig_ratio_problem_l760_76060

theorem trig_ratio_problem (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) :
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 := by
  sorry

end trig_ratio_problem_l760_76060


namespace april_price_achieves_profit_l760_76092

/-- Represents the sales and pricing data for a desk lamp over several months -/
structure LampSalesData where
  cost_price : ℝ
  selling_price_jan_mar : ℝ
  sales_jan : ℕ
  sales_mar : ℕ
  price_reduction_sales_increase : ℝ
  desired_profit_apr : ℝ

/-- Calculates the selling price in April that achieves the desired profit -/
def calculate_april_price (data : LampSalesData) : ℝ :=
  sorry

/-- Theorem stating that the calculated April price achieves the desired profit -/
theorem april_price_achieves_profit (data : LampSalesData) 
  (h1 : data.cost_price = 25)
  (h2 : data.selling_price_jan_mar = 40)
  (h3 : data.sales_jan = 256)
  (h4 : data.sales_mar = 400)
  (h5 : data.price_reduction_sales_increase = 4)
  (h6 : data.desired_profit_apr = 4200) :
  let april_price := calculate_april_price data
  let april_sales := data.sales_mar + data.price_reduction_sales_increase * (data.selling_price_jan_mar - april_price)
  (april_price - data.cost_price) * april_sales = data.desired_profit_apr ∧ april_price = 35 :=
sorry

end april_price_achieves_profit_l760_76092


namespace revenue_calculation_l760_76014

/-- Calculates the total revenue given the salary expense and ratio of salary to stock purchase --/
def total_revenue (salary_expense : ℚ) (salary_ratio : ℚ) (stock_ratio : ℚ) : ℚ :=
  salary_expense * (salary_ratio + stock_ratio) / salary_ratio

/-- Proves that the total revenue is 3000 given the conditions --/
theorem revenue_calculation :
  let salary_expense : ℚ := 800
  let salary_ratio : ℚ := 4
  let stock_ratio : ℚ := 11
  total_revenue salary_expense salary_ratio stock_ratio = 3000 := by
sorry

#eval total_revenue 800 4 11

end revenue_calculation_l760_76014


namespace f_definition_f_of_five_l760_76008

noncomputable def f : ℝ → ℝ := λ u => (u^3 + 6*u^2 + 21*u + 40) / 27

theorem f_definition (x : ℝ) : f (3*x - 1) = x^3 + x^2 + x + 1 := by sorry

theorem f_of_five : f 5 = 140 / 9 := by sorry

end f_definition_f_of_five_l760_76008


namespace soccer_league_teams_l760_76072

/-- The number of teams in a soccer league where each team plays every other team once 
    and the total number of games is 105. -/
def num_teams : ℕ := 15

/-- The total number of games played in the league. -/
def total_games : ℕ := 105

/-- Formula for the number of games in a round-robin tournament. -/
def games_formula (n : ℕ) : ℕ := n * (n - 1) / 2

theorem soccer_league_teams : 
  games_formula num_teams = total_games ∧ num_teams > 0 :=
sorry

end soccer_league_teams_l760_76072


namespace meat_for_hamburgers_l760_76064

/-- Given that 3 pounds of meat make 8 hamburgers, prove that 9 pounds of meat are needed for 24 hamburgers -/
theorem meat_for_hamburgers (meat_per_8 : ℝ) (hamburgers : ℝ) 
  (h1 : meat_per_8 = 3) 
  (h2 : hamburgers = 24) : 
  (meat_per_8 / 8) * hamburgers = 9 := by
  sorry

end meat_for_hamburgers_l760_76064


namespace stock_price_increase_percentage_l760_76077

theorem stock_price_increase_percentage (total_stocks : ℕ) (higher_price_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : higher_price_stocks = 1080)
  (h3 : higher_price_stocks > total_stocks - higher_price_stocks) :
  let lower_price_stocks := total_stocks - higher_price_stocks
  (higher_price_stocks - lower_price_stocks) / lower_price_stocks * 100 = 20 := by
  sorry

end stock_price_increase_percentage_l760_76077


namespace ab_neg_necessary_not_sufficient_for_hyperbola_l760_76038

/-- Represents a conic section in the form ax^2 + by^2 = c -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is a hyperbola -/
def IsHyperbola (conic : ConicSection) : Prop :=
  sorry  -- The actual definition would depend on the formal definition of a hyperbola

/-- The main theorem stating that ab < 0 is necessary but not sufficient for a hyperbola -/
theorem ab_neg_necessary_not_sufficient_for_hyperbola :
  (∀ conic : ConicSection, IsHyperbola conic → conic.a * conic.b < 0) ∧
  (∃ conic : ConicSection, conic.a * conic.b < 0 ∧ ¬IsHyperbola conic) :=
sorry

end ab_neg_necessary_not_sufficient_for_hyperbola_l760_76038


namespace max_food_per_guest_l760_76085

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (max_food : ℚ) : 
  total_food = 337 → min_guests = 169 → max_food = 2 → 
  max_food = (total_food : ℚ) / min_guests :=
by sorry

end max_food_per_guest_l760_76085


namespace cross_shaped_graph_paper_rectangles_l760_76051

/-- Calculates the number of rectangles in a grid --/
def rectangleCount (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Calculates the sum of squares from 1 to n --/
def sumOfSquares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

/-- The side length of the original square graph paper in mm --/
def originalSideLength : ℕ := 30

/-- The side length of the cut-away corner squares in mm --/
def cornerSideLength : ℕ := 10

/-- The total number of smallest squares in the original graph paper --/
def totalSmallestSquares : ℕ := 900

theorem cross_shaped_graph_paper_rectangles :
  let totalRectangles := rectangleCount originalSideLength originalSideLength
  let cornerRectangles := 4 * rectangleCount cornerSideLength originalSideLength
  let remainingSquares := 2 * sumOfSquares originalSideLength - sumOfSquares (originalSideLength - 2 * cornerSideLength)
  totalRectangles - cornerRectangles - remainingSquares = 144130 := by
  sorry

end cross_shaped_graph_paper_rectangles_l760_76051


namespace interest_rate_calculation_l760_76042

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432) :
  ∃ (rate : ℝ), 
    rate > 0 ∧ 
    rate = (interest_paid * 100) / (principal * rate) ∧ 
    rate = 6 := by
  sorry

end interest_rate_calculation_l760_76042


namespace triangle_problem_l760_76099

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c A B C : ℝ) : Prop :=
  -- Add necessary conditions for a valid triangle here
  True

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : Real.sqrt 3 * c * Real.cos A + a * Real.sin C = Real.sqrt 3 * c)
  (h_sum : b + c = 5)
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end triangle_problem_l760_76099


namespace points_below_line_l760_76096

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d

-- Define the geometric sequence
def geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q

theorem points_below_line (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 2 →
  geometric_sequence 1 y₁ y₂ 2 →
  x₁ > y₁ ∧ x₂ > y₂ := by
  sorry

end points_below_line_l760_76096


namespace car_discount_proof_l760_76089

/-- Proves that the discount on a car is 20% of the original price given the specified conditions -/
theorem car_discount_proof (P : ℝ) (D : ℝ) : 
  P > 0 →  -- Assuming positive original price
  D > 0 →  -- Assuming positive discount
  D < P →  -- Discount is less than original price
  (P - D + 0.45 * (P - D)) = (P + 0.16 * P) →  -- Selling price equation
  D = 0.2 * P :=  -- Conclusion: discount is 20% of original price
by sorry  -- Proof is omitted

end car_discount_proof_l760_76089


namespace special_arithmetic_l760_76055

/-- In a country with non-standard arithmetic, prove that if 1/5 of 8 equals 4,
    and 1/4 of a number X equals 10, then X must be 16. -/
theorem special_arithmetic (country_fifth : ℚ → ℚ) (X : ℚ) :
  country_fifth 8 = 4 →
  country_fifth X = 10 →
  X = 16 :=
by
  sorry

#check special_arithmetic

end special_arithmetic_l760_76055


namespace building_heights_sum_l760_76034

theorem building_heights_sum (h1 h2 h3 h4 : ℝ) : 
  h1 = 100 →
  h2 = h1 / 2 →
  h3 = h2 / 2 →
  h4 = h3 / 5 →
  h1 + h2 + h3 + h4 = 180 := by
  sorry

end building_heights_sum_l760_76034


namespace college_board_committee_count_l760_76031

/-- Represents a college board. -/
structure Board :=
  (total_members : ℕ)
  (professors : ℕ)
  (non_professors : ℕ)
  (h_total : total_members = professors + non_professors)

/-- Represents a committee formed from the board. -/
structure Committee :=
  (size : ℕ)
  (min_professors : ℕ)

/-- Calculates the number of valid committees for a given board and committee requirements. -/
def count_valid_committees (board : Board) (committee : Committee) : ℕ :=
  sorry

/-- The specific board in the problem. -/
def college_board : Board :=
  { total_members := 15
  , professors := 7
  , non_professors := 8
  , h_total := by rfl }

/-- The specific committee requirements in the problem. -/
def required_committee : Committee :=
  { size := 5
  , min_professors := 2 }

theorem college_board_committee_count :
  count_valid_committees college_board required_committee = 2457 :=
sorry

end college_board_committee_count_l760_76031


namespace ordering_abc_l760_76015

theorem ordering_abc :
  let a : ℝ := Real.log 2
  let b : ℝ := 2023 / 2022
  let c : ℝ := Real.log 2023 / Real.log 2022
  a < c ∧ c < b := by sorry

end ordering_abc_l760_76015


namespace y_intercept_of_parallel_line_l760_76098

/-- A line in the xy-plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3/2, point := (0, 6) } →
  b.point = (4, 2) →
  y_intercept b = -4 := by
sorry

end y_intercept_of_parallel_line_l760_76098


namespace rectangle_lengths_l760_76018

/-- Given a square and two rectangles with specific properties, prove their lengths -/
theorem rectangle_lengths (square_side : ℝ) (rect1_width rect2_width : ℝ) 
  (h1 : square_side = 6)
  (h2 : rect1_width = 4)
  (h3 : rect2_width = 3)
  (h4 : square_side * square_side = rect1_width * (square_side * square_side / rect1_width))
  (h5 : rect2_width * (square_side * square_side / (2 * rect2_width)) = square_side * square_side / 2) :
  (square_side * square_side / rect1_width, square_side * square_side / (2 * rect2_width)) = (9, 6) := by
  sorry

#check rectangle_lengths

end rectangle_lengths_l760_76018


namespace max_b_for_zero_in_range_l760_76002

/-- The maximum value of b such that 0 is in the range of the quadratic function g(x) = x^2 - 7x + b is 49/4. -/
theorem max_b_for_zero_in_range : 
  ∀ b : ℝ, (∃ x : ℝ, x^2 - 7*x + b = 0) ↔ b ≤ 49/4 :=
sorry

end max_b_for_zero_in_range_l760_76002


namespace geometric_series_sum_l760_76025

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/3 is 3/2 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/3
  let S : ℝ := ∑' n, a * r^n
  S = 3/2 := by sorry

end geometric_series_sum_l760_76025


namespace solution_set_inequality_l760_76082

/-- Given that the solution set of ax^2 + 5x + b > 0 is {x | 2 < x < 3},
    prove that the solution set of bx^2 - 5x + a > 0 is (-1/2, -1/3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x + b > 0 ↔ 2 < x ∧ x < 3) →
  (∀ x : ℝ, b*x^2 - 5*x + a > 0 ↔ -1/2 < x ∧ x < -1/3) :=
sorry

end solution_set_inequality_l760_76082


namespace rectangle_area_l760_76032

/-- A rectangle with length twice its width and perimeter 84 cm has an area of 392 cm² -/
theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  length = 2 * width →
  perimeter = 84 →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 392 := by
  sorry

end rectangle_area_l760_76032


namespace billboard_fully_lit_probability_l760_76017

/-- The number of words in the billboard text -/
def num_words : ℕ := 5

/-- The probability of seeing the billboard fully lit -/
def fully_lit_probability : ℚ := 1 / num_words

/-- Theorem stating that the probability of seeing the billboard fully lit is 1/5 -/
theorem billboard_fully_lit_probability :
  fully_lit_probability = 1 / 5 := by sorry

end billboard_fully_lit_probability_l760_76017


namespace three_digit_number_operations_l760_76027

theorem three_digit_number_operations (a b c : Nat) 
  (h1 : a > 0) 
  (h2 : a < 10) 
  (h3 : b < 10) 
  (h4 : c < 10) : 
  ((2 * a + 3) * 5 + b) * 10 + c - 150 = 100 * a + 10 * b + c := by
  sorry

end three_digit_number_operations_l760_76027


namespace truck_fill_rate_l760_76041

/-- The rate at which a person can fill a truck with stone blocks per hour -/
def fill_rate : ℕ → Prop :=
  λ r => 
    -- Truck capacity
    let capacity : ℕ := 6000
    -- Number of people working initially
    let initial_workers : ℕ := 2
    -- Number of hours initial workers work
    let initial_hours : ℕ := 4
    -- Total number of workers after more join
    let total_workers : ℕ := 8
    -- Number of hours all workers work together
    let final_hours : ℕ := 2
    -- Total time to fill the truck
    let total_time : ℕ := 6

    -- The truck is filled when the sum of blocks filled in both phases equals the capacity
    (initial_workers * initial_hours * r) + (total_workers * final_hours * r) = capacity

theorem truck_fill_rate : fill_rate 250 := by
  sorry

end truck_fill_rate_l760_76041


namespace square_tiles_count_l760_76012

/-- Represents a box of pentagonal and square tiles -/
structure TileBox where
  pentagonal : ℕ
  square : ℕ

/-- The total number of tiles in the box -/
def TileBox.total (box : TileBox) : ℕ := box.pentagonal + box.square

/-- The total number of edges in the box -/
def TileBox.edges (box : TileBox) : ℕ := 5 * box.pentagonal + 4 * box.square

theorem square_tiles_count (box : TileBox) : 
  box.total = 30 ∧ box.edges = 122 → box.square = 28 := by
  sorry

end square_tiles_count_l760_76012


namespace local_tax_deduction_l760_76022

-- Define Carl's hourly wage in dollars
def carlHourlyWage : ℝ := 25

-- Define the local tax rate as a percentage
def localTaxRate : ℝ := 2.0

-- Define the conversion rate from dollars to cents
def dollarsToCents : ℝ := 100

-- Theorem to prove
theorem local_tax_deduction :
  (carlHourlyWage * dollarsToCents * (localTaxRate / 100)) = 50 := by
  sorry


end local_tax_deduction_l760_76022


namespace lambda_n_lower_bound_l760_76028

/-- The ratio of the longest to the shortest distance between any two of n points in the plane -/
def lambda_n (n : ℕ) : ℝ := sorry

/-- Theorem: For n ≥ 4, λ_n ≥ 2 * sin((n-2)π/(2n)) -/
theorem lambda_n_lower_bound (n : ℕ) (h : n ≥ 4) : 
  lambda_n n ≥ 2 * Real.sin ((n - 2) * Real.pi / (2 * n)) := by sorry

end lambda_n_lower_bound_l760_76028


namespace origin_outside_circle_l760_76019

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let circle_equation (x y : ℝ) := x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2
  circle_equation 0 0 > 0 := by
  sorry

end origin_outside_circle_l760_76019


namespace painted_cube_theorem_l760_76043

/-- Given a cube of side length n, painted blue on all faces and split into unit cubes,
    if exactly one-third of the total faces of the unit cubes are blue, then n = 3 -/
theorem painted_cube_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by
  sorry

end painted_cube_theorem_l760_76043


namespace product_of_square_roots_l760_76061

theorem product_of_square_roots (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 126 * q * Real.sqrt q :=
by sorry

end product_of_square_roots_l760_76061


namespace A_eq_union_l760_76093

/-- The set of real numbers a > 0 such that either y = a^x is not monotonically increasing on R
    or ax^2 - ax + 1 > 0 does not hold for all x ∈ R, but at least one of these conditions is true. -/
def A : Set ℝ :=
  {a : ℝ | a > 0 ∧
    (¬(∀ x y : ℝ, x < y → a^x < a^y) ∨ ¬(∀ x : ℝ, a*x^2 - a*x + 1 > 0)) ∧
    ((∀ x y : ℝ, x < y → a^x < a^y) ∨ (∀ x : ℝ, a*x^2 - a*x + 1 > 0))}

/-- The theorem stating that A is equal to the interval (0,1] union [4,+∞) -/
theorem A_eq_union : A = Set.Ioo 0 1 ∪ Set.Ici 4 := by sorry

end A_eq_union_l760_76093


namespace vector_decomposition_l760_76054

theorem vector_decomposition (e₁ e₂ a : ℝ × ℝ) :
  e₁ = (1, 2) →
  e₂ = (-2, 3) →
  a = (-1, 2) →
  a = (1/7 : ℝ) • e₁ + (4/7 : ℝ) • e₂ := by sorry

end vector_decomposition_l760_76054


namespace cinema_ticket_pricing_l760_76045

theorem cinema_ticket_pricing (adult_price : ℚ) : 
  (10 * adult_price + 6 * (adult_price / 2) = 35) →
  ((12 * adult_price + 8 * (adult_price / 2)) * (9 / 10) = 504 / 13) := by
  sorry

end cinema_ticket_pricing_l760_76045


namespace rect_to_polar_conversion_l760_76013

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion (x y : ℝ) (h : (x, y) = (8, 2 * Real.sqrt 3)) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 2 * Real.sqrt 19 ∧ θ = Real.pi / 6 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end rect_to_polar_conversion_l760_76013


namespace tony_school_years_l760_76059

/-- The total number of years Tony went to school to become an astronaut -/
def total_school_years (first_degree_years : ℕ) (additional_degrees : ℕ) (graduate_degree_years : ℕ) : ℕ :=
  first_degree_years + additional_degrees * first_degree_years + graduate_degree_years

/-- Theorem stating that Tony went to school for 14 years -/
theorem tony_school_years :
  total_school_years 4 2 2 = 14 := by
  sorry

end tony_school_years_l760_76059


namespace erwans_shopping_trip_l760_76067

/-- Proves that the price of each shirt is $80 given the conditions of Erwan's shopping trip -/
theorem erwans_shopping_trip (shoe_price : ℝ) (shirt_price : ℝ) :
  shoe_price = 200 →
  (shoe_price * 0.7 + 2 * shirt_price) * 0.95 = 285 →
  shirt_price = 80 :=
by sorry

end erwans_shopping_trip_l760_76067


namespace sum_inequality_l760_76062

theorem sum_inequality {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end sum_inequality_l760_76062


namespace expression_value_l760_76058

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : abs m = 2)  -- |m| = 2
  : (3 * c * d) / (4 * m) + m^2 - 5 * (a + b) = 35/8 ∨ 
    (3 * c * d) / (4 * m) + m^2 - 5 * (a + b) = 29/8 :=
by sorry

end expression_value_l760_76058


namespace isabella_hair_length_l760_76069

/-- The length of Isabella's hair before the haircut -/
def hair_length_before : ℕ := sorry

/-- The length of Isabella's hair after the haircut -/
def hair_length_after : ℕ := 9

/-- The length of hair that was cut off -/
def hair_length_cut : ℕ := 9

/-- Theorem stating that the length of Isabella's hair before the haircut
    is equal to the sum of the length after the haircut and the length cut off -/
theorem isabella_hair_length : hair_length_before = hair_length_after + hair_length_cut := by
  sorry

end isabella_hair_length_l760_76069


namespace no_triple_squares_l760_76007

theorem no_triple_squares : ¬∃ (m n k : ℕ), 
  (∃ a : ℕ, m^2 + n + k = a^2) ∧ 
  (∃ b : ℕ, n^2 + k + m = b^2) ∧ 
  (∃ c : ℕ, k^2 + m + n = c^2) := by
sorry

end no_triple_squares_l760_76007


namespace car_price_calculation_l760_76011

theorem car_price_calculation (reduced_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) : 
  reduced_price = 7500 ∧ 
  discount_percentage = 25 ∧ 
  reduced_price = original_price * (1 - discount_percentage / 100) → 
  original_price = 10000 := by
sorry

end car_price_calculation_l760_76011


namespace compound_molecular_weight_l760_76094

/-- Atomic weight in atomic mass units (amu) -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "H" => 1.008
  | "Br" => 79.904
  | "O" => 15.999
  | "C" => 12.011
  | "N" => 14.007
  | "S" => 32.065
  | _ => 0  -- Default case for unknown elements

/-- Number of atoms of each element in the compound -/
def atom_count (element : String) : ℕ :=
  match element with
  | "H" => 2
  | "Br" => 1
  | "O" => 3
  | "C" => 1
  | "N" => 1
  | "S" => 2
  | _ => 0  -- Default case for elements not in the compound

/-- Calculate the molecular weight of the compound -/
def molecular_weight : ℝ :=
  (atomic_weight "H" * atom_count "H") +
  (atomic_weight "Br" * atom_count "Br") +
  (atomic_weight "O" * atom_count "O") +
  (atomic_weight "C" * atom_count "C") +
  (atomic_weight "N" * atom_count "N") +
  (atomic_weight "S" * atom_count "S")

/-- Theorem stating that the molecular weight of the compound is 220.065 amu -/
theorem compound_molecular_weight : molecular_weight = 220.065 := by
  sorry

end compound_molecular_weight_l760_76094


namespace ones_digit_factorial_sum_10_l760_76075

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def ones_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => factorial (n + 1) + factorial_sum n

theorem ones_digit_factorial_sum_10 :
  ones_digit (factorial_sum 10) = 3 := by sorry

end ones_digit_factorial_sum_10_l760_76075


namespace f_composition_equals_negative_262144_l760_76070

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if ¬(z.re = 0 ∧ z.im = 0) then z^2
  else if 0 < z.re then -z^2
  else z^3

-- State the theorem
theorem f_composition_equals_negative_262144 :
  f (f (f (f (1 + I)))) = -262144 := by
  sorry

end f_composition_equals_negative_262144_l760_76070


namespace smallest_possible_b_l760_76056

-- Define the conditions
def no_triangle_2ab (a b : ℝ) : Prop := 2 + a ≤ b
def no_triangle_inverse (a b : ℝ) : Prop := 1/b + 1/a ≤ 2

-- State the theorem
theorem smallest_possible_b :
  ∀ a b : ℝ, 2 < a → a < b → no_triangle_2ab a b → no_triangle_inverse a b →
  b ≥ (5 + Real.sqrt 17) / 4 :=
sorry

end smallest_possible_b_l760_76056


namespace trig_identity_l760_76065

theorem trig_identity : 
  Real.cos (π / 3) * Real.tan (π / 4) + 3 / 4 * (Real.tan (π / 6))^2 - Real.sin (π / 6) + (Real.cos (π / 6))^2 = 1 := by
  sorry

end trig_identity_l760_76065


namespace completing_square_quadratic_l760_76016

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 4*x - 2 = 0) ↔ ((x - 2)^2 = 6) :=
sorry

end completing_square_quadratic_l760_76016


namespace average_of_data_set_l760_76071

def data_set : List ℤ := [3, -2, 4, 1, 4]

theorem average_of_data_set :
  (data_set.sum : ℚ) / data_set.length = 2 := by
  sorry

end average_of_data_set_l760_76071


namespace triangle_third_sides_l760_76076

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.a / t2.a = k ∧ 
    t1.b / t2.b = k ∧ 
    t1.c / t2.c = k

theorem triangle_third_sides 
  (t1 t2 : Triangle) 
  (h_similar : similar t1 t2) 
  (h_not_congruent : t1 ≠ t2) 
  (h_t1_sides : t1.a = 12 ∧ t1.b = 18) 
  (h_t2_sides : t2.a = 12 ∧ t2.b = 18) : 
  (t1.c = 27/2 ∧ t2.c = 8) ∨ (t1.c = 8 ∧ t2.c = 27/2) :=
sorry

end triangle_third_sides_l760_76076


namespace no_triple_perfect_squares_l760_76086

theorem no_triple_perfect_squares : 
  ¬ ∃ (a b c : ℕ+), 
    (∃ (x y z : ℕ), (a^2 * b * c + 2 : ℕ) = x^2 ∧ 
                    (b^2 * c * a + 2 : ℕ) = y^2 ∧ 
                    (c^2 * a * b + 2 : ℕ) = z^2) :=
by sorry

end no_triple_perfect_squares_l760_76086


namespace bill_eric_age_difference_l760_76057

/-- The age difference between two brothers, given their total age and the older brother's age. -/
def age_difference (total_age : ℕ) (older_brother_age : ℕ) : ℕ :=
  older_brother_age - (total_age - older_brother_age)

/-- Theorem stating the age difference between Bill and Eric -/
theorem bill_eric_age_difference :
  let total_age : ℕ := 28
  let bill_age : ℕ := 16
  age_difference total_age bill_age = 4 := by
  sorry

end bill_eric_age_difference_l760_76057


namespace regression_analysis_l760_76078

-- Define the data points
def data : List (ℝ × ℝ) := [(5, 17), (6, 20), (8, 25), (9, 28), (12, 35)]

-- Define the regression equation
def regression_equation (x : ℝ) (a : ℝ) : ℝ := 2.6 * x + a

-- Theorem statement
theorem regression_analysis :
  -- 1. Center point
  (let x_mean := (data.map Prod.fst).sum / data.length
   let y_mean := (data.map Prod.snd).sum / data.length
   (x_mean, y_mean) = (8, 25)) ∧
  -- 2. Y-intercept
  (∃ a : ℝ, a = 4.2 ∧
    regression_equation 8 a = 25) ∧
  -- 3. Residual when x = 5
  (let a := 4.2
   let y_pred := regression_equation 5 a
   let y_actual := 17
   y_actual - y_pred = -0.2) := by
  sorry


end regression_analysis_l760_76078


namespace dimitri_burger_calories_l760_76033

/-- Calculates the number of calories per burger given the daily burger consumption and total calories over two days. -/
def calories_per_burger (burgers_per_day : ℕ) (total_calories : ℕ) : ℕ :=
  total_calories / (burgers_per_day * 2)

/-- Theorem stating that given Dimitri's burger consumption and calorie intake, each burger contains 20 calories. -/
theorem dimitri_burger_calories :
  calories_per_burger 3 120 = 20 := by
  sorry

end dimitri_burger_calories_l760_76033


namespace student_average_mark_l760_76004

/-- Given a student's marks in 5 subjects, prove that the average mark in 4 subjects
    (excluding physics) is 70, when the total marks are 280 more than the physics marks. -/
theorem student_average_mark (physics chemistry maths biology english : ℕ) :
  physics + chemistry + maths + biology + english = physics + 280 →
  (chemistry + maths + biology + english) / 4 = 70 := by
  sorry

end student_average_mark_l760_76004


namespace overlapping_strips_area_l760_76036

/-- Represents a rectangular strip with given length and width -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Calculates the number of overlaps between n strips -/
def numOverlaps (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The total area covered by 5 overlapping strips -/
theorem overlapping_strips_area :
  let strips : List Strip := List.replicate 5 ⟨12, 1⟩
  let totalStripArea := (strips.map stripArea).sum
  let overlapArea := numOverlaps 5
  totalStripArea - overlapArea = 50 := by
  sorry


end overlapping_strips_area_l760_76036


namespace cos_sum_fifteenth_l760_76079

theorem cos_sum_fifteenth : Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (8 * Real.pi / 15) = 1 := by
  sorry

end cos_sum_fifteenth_l760_76079


namespace rectangle_max_area_l760_76030

/-- Given a rectangle with perimeter 60, its maximum possible area is 225 -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 60 →
  x * y ≤ 225 :=
by
  sorry

end rectangle_max_area_l760_76030


namespace patricks_age_to_roberts_age_ratio_l760_76026

/-- Given that Robert will turn 30 after 2 years and Patrick is 14 years old now,
    prove that the ratio of Patrick's age to Robert's age is 1:2 -/
theorem patricks_age_to_roberts_age_ratio :
  ∀ (roberts_age patricks_age : ℕ),
  roberts_age + 2 = 30 →
  patricks_age = 14 →
  patricks_age / roberts_age = 1 / 2 := by
sorry

end patricks_age_to_roberts_age_ratio_l760_76026


namespace count_special_numbers_eq_384_l760_76024

/-- Counts 4-digit numbers beginning with 2 that have exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := 10 -- Total number of digits (0-9)
  let first_digit := 2 -- First digit is always 2
  let remaining_digits := digits - 1 -- Excluding 2
  let configurations := 2 -- Two main configurations: 2 is repeated or not

  -- When 2 is one of the repeated digits
  let case1 := 3 * remaining_digits * remaining_digits

  -- When 2 is not one of the repeated digits
  let case2 := 3 * remaining_digits * remaining_digits

  case1 + case2

theorem count_special_numbers_eq_384 : count_special_numbers = 384 := by
  sorry

end count_special_numbers_eq_384_l760_76024


namespace ratio_problem_l760_76048

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end ratio_problem_l760_76048


namespace five_T_three_l760_76005

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 7*b + 2*a*b

-- Theorem statement
theorem five_T_three : T 5 3 = 71 := by
  sorry

end five_T_three_l760_76005


namespace three_lines_intersection_l760_76020

/-- Three lines intersect at a single point if and only if k = -2/7 --/
theorem three_lines_intersection (x y k : ℚ) : 
  (y = 3*x + 2 ∧ y = -4*x - 14 ∧ y = 2*x + k) ↔ k = -2/7 := by
  sorry

end three_lines_intersection_l760_76020


namespace weightlifter_total_weight_l760_76091

/-- The weight a weightlifter can lift in each hand, in pounds. -/
def weight_per_hand : ℕ := 10

/-- The total weight a weightlifter can lift at once, in pounds. -/
def total_weight : ℕ := weight_per_hand * 2

/-- Theorem stating that the total weight a weightlifter can lift at once is 20 pounds. -/
theorem weightlifter_total_weight : total_weight = 20 := by sorry

end weightlifter_total_weight_l760_76091


namespace unique_solution_condition_l760_76090

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+8)*(x-6) = -55 + k*x) ↔ (k = -10 + 2*Real.sqrt 21 ∨ k = -10 - 2*Real.sqrt 21) := by
  sorry

end unique_solution_condition_l760_76090


namespace circle_area_l760_76003

theorem circle_area (r : ℝ) (h : 6 / (2 * Real.pi * r) = 2 * r) : 
  Real.pi * r^2 = 3/2 := by
  sorry

end circle_area_l760_76003


namespace major_premise_incorrect_l760_76066

theorem major_premise_incorrect : ¬(∀ (a : ℝ) (n : ℕ), n > 0 → (a^(1/n : ℝ))^n = a) := by
  sorry

end major_premise_incorrect_l760_76066


namespace max_product_of_three_primes_l760_76050

theorem max_product_of_three_primes (x y z : ℕ) : 
  Prime x → Prime y → Prime z →
  x ≠ y → x ≠ z → y ≠ z →
  x + y + z = 49 →
  x * y * z ≤ 4199 := by
sorry

end max_product_of_three_primes_l760_76050


namespace quadratic_properties_l760_76029

-- Define the quadratic function
def f (x : ℝ) := -x^2 + 9*x - 20

-- Theorem statement
theorem quadratic_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≥ 0 → x ≤ max) ∧
  (∃ (max : ℝ), f max ≥ 0 ∧ max = 5) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x^2 - 9*x + 20 = 0) :=
by sorry

end quadratic_properties_l760_76029


namespace sum_of_squares_l760_76049

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 := by
  sorry

end sum_of_squares_l760_76049


namespace x_squared_minus_nine_y_squared_l760_76001

theorem x_squared_minus_nine_y_squared (x y : ℝ) 
  (eq1 : x + 3*y = -1) 
  (eq2 : x - 3*y = 5) : 
  x^2 - 9*y^2 = -5 := by
sorry

end x_squared_minus_nine_y_squared_l760_76001


namespace square_difference_quotient_l760_76040

theorem square_difference_quotient : (347^2 - 333^2) / 14 = 680 := by
  sorry

end square_difference_quotient_l760_76040


namespace orthogonal_vectors_l760_76083

/-- Two vectors in ℝ³ -/
def v1 : Fin 3 → ℝ := ![3, -1, 4]
def v2 (x : ℝ) : Fin 3 → ℝ := ![x, 4, -2]

/-- Dot product of two vectors in ℝ³ -/
def dot_product (u v : Fin 3 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

/-- The theorem stating that x = 4 makes v1 and v2 orthogonal -/
theorem orthogonal_vectors :
  ∃ x : ℝ, dot_product v1 (v2 x) = 0 ∧ x = 4 := by
  sorry

#check orthogonal_vectors

end orthogonal_vectors_l760_76083


namespace pyramid_properties_l760_76068

-- Define the cone and pyramid
structure Cone where
  height : ℝ
  slantHeight : ℝ

structure Pyramid where
  cone : Cone
  OB : ℝ

-- Define the properties of the cone and pyramid
def isValidCone (c : Cone) : Prop :=
  c.height = 4 ∧ c.slantHeight = 5

def isValidPyramid (p : Pyramid) : Prop :=
  isValidCone p.cone ∧ p.OB = 3

-- Define the properties to be proved
def pyramidVolume (p : Pyramid) : ℝ := sorry

def dihedralAngleAB (p : Pyramid) : ℝ := sorry

def circumscribedSphereRadius (p : Pyramid) : ℝ := sorry

-- Main theorem
theorem pyramid_properties (p : Pyramid) 
  (h : isValidPyramid p) : 
  ∃ (v d r : ℝ),
    pyramidVolume p = v ∧
    dihedralAngleAB p = d ∧
    circumscribedSphereRadius p = r :=
  sorry

end pyramid_properties_l760_76068


namespace hula_hoop_radius_l760_76021

theorem hula_hoop_radius (diameter : ℝ) (h : diameter = 14) : diameter / 2 = 7 := by
  sorry

end hula_hoop_radius_l760_76021


namespace existence_equivalence_l760_76009

theorem existence_equivalence (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 :=
by sorry

end existence_equivalence_l760_76009


namespace triangle_quadratic_no_solution_l760_76035

theorem triangle_quadratic_no_solution (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 + c^2 - a^2)^2 - 4*(b^2)*(c^2) < 0 :=
sorry

end triangle_quadratic_no_solution_l760_76035


namespace f_inequality_l760_76080

/-- A quadratic function with positive leading coefficient and axis of symmetry at x=1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) > f(2^x) for x > 0 -/
theorem f_inequality (a b c : ℝ) (h_a : a > 0) (h_sym : ∀ x, f a b c (2 - x) = f a b c x) :
  ∀ x > 0, f a b c (3^x) > f a b c (2^x) := by
  sorry

end f_inequality_l760_76080


namespace set_operations_l760_76063

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}

-- Define the theorem
theorem set_operations :
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  (A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x < 0}) := by
  sorry

end set_operations_l760_76063


namespace square_rectangle_intersection_l760_76053

theorem square_rectangle_intersection (EFGH_side_length MO LO shaded_area : ℝ) :
  EFGH_side_length = 8 →
  MO = 12 →
  LO = 8 →
  shaded_area = (MO * LO) / 2 →
  shaded_area = EFGH_side_length * (EFGH_side_length - EM) →
  EM = 2 :=
by sorry

end square_rectangle_intersection_l760_76053


namespace longest_segment_in_cylinder_l760_76010

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Theorem statement
theorem longest_segment_in_cylinder :
  let diagonal := Real.sqrt (cylinder_height ^ 2 + (2 * cylinder_radius) ^ 2)
  diagonal = 10 * Real.sqrt 2 := by sorry

end longest_segment_in_cylinder_l760_76010


namespace quadratic_roots_same_sign_l760_76052

theorem quadratic_roots_same_sign (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + 2*x₁ + m = 0 ∧ 
   x₂^2 + 2*x₂ + m = 0 ∧
   (x₁ > 0 ∧ x₂ > 0 ∨ x₁ < 0 ∧ x₂ < 0)) →
  (0 < m ∧ m ≤ 1) :=
by sorry

end quadratic_roots_same_sign_l760_76052


namespace units_digit_sum_octal_l760_76095

/-- Converts a number from base 8 to base 10 -/
def octalToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 -/
def decimalToOctal (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a number in base 8 -/
def unitsDigitOctal (n : ℕ) : ℕ := sorry

theorem units_digit_sum_octal :
  unitsDigitOctal (decimalToOctal (octalToDecimal 45 + octalToDecimal 67)) = 4 := by
  sorry

end units_digit_sum_octal_l760_76095


namespace point_on_angle_negative_pi_third_l760_76023

/-- Given a point P(2,y) on the terminal side of angle -π/3, prove that y = -2√3 -/
theorem point_on_angle_negative_pi_third (y : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 2 ∧ P.2 = y ∧ P.2 / P.1 = Real.tan (-π/3)) → 
  y = -2 * Real.sqrt 3 := by
  sorry

end point_on_angle_negative_pi_third_l760_76023


namespace delta_value_l760_76097

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 3 → Δ = -9 := by
  sorry

end delta_value_l760_76097
