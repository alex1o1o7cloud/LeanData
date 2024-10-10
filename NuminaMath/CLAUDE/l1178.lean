import Mathlib

namespace charlies_bus_ride_l1178_117857

theorem charlies_bus_ride (oscars_ride : ℝ) (difference : ℝ) :
  oscars_ride = 0.75 →
  oscars_ride = difference + charlies_ride →
  difference = 0.5 →
  charlies_ride = 0.25 :=
by
  sorry

end charlies_bus_ride_l1178_117857


namespace college_students_count_l1178_117854

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 210) :
  boys + girls = 546 := by
  sorry

end college_students_count_l1178_117854


namespace project_hours_difference_l1178_117863

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 189) 
  (h_pat_kate : ∃ k : ℕ, pat = 2 * k ∧ kate = k) 
  (h_pat_mark : ∃ m : ℕ, mark = 3 * pat ∧ pat = m) 
  (h_sum : pat + kate + mark = total_hours) :
  mark - kate = 105 :=
by sorry

end project_hours_difference_l1178_117863


namespace mail_difference_l1178_117849

/-- Proves that the difference between mail sent on Thursday and Wednesday is 15 --/
theorem mail_difference (monday tuesday wednesday thursday : ℕ) : 
  monday = 65 →
  tuesday = monday + 10 →
  wednesday = tuesday - 5 →
  thursday > wednesday →
  monday + tuesday + wednesday + thursday = 295 →
  thursday - wednesday = 15 :=
by sorry

end mail_difference_l1178_117849


namespace polynomial_division_remainder_l1178_117810

theorem polynomial_division_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (S : ℝ → ℝ),
    (∀ x, x^50 = (x^2 - 5*x + 6) * Q x + S x) ∧
    (∃ a b : ℝ, ∀ x, S x = a * x + b) ∧
    S x = (3^50 - 2^50) * x + (4^50 - 6^50) := by
  sorry

end polynomial_division_remainder_l1178_117810


namespace same_number_of_atoms_l1178_117828

/-- The number of atoms in a mole of a substance -/
def atoms_per_mole (substance : String) : ℕ :=
  match substance with
  | "H₃PO₄" => 8
  | "H₂O₂" => 4
  | _ => 0

/-- The number of moles of a substance -/
def moles (substance : String) : ℚ :=
  match substance with
  | "H₃PO₄" => 1/5
  | "H₂O₂" => 2/5
  | _ => 0

/-- The total number of atoms in a given amount of a substance -/
def total_atoms (substance : String) : ℚ :=
  (moles substance) * (atoms_per_mole substance)

theorem same_number_of_atoms : total_atoms "H₃PO₄" = total_atoms "H₂O₂" := by
  sorry

end same_number_of_atoms_l1178_117828


namespace square_sum_ge_mixed_products_l1178_117882

theorem square_sum_ge_mixed_products (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end square_sum_ge_mixed_products_l1178_117882


namespace profit_for_five_yuan_reduction_optimal_price_reduction_l1178_117815

/-- Represents the product details and sales dynamics -/
structure ProductSales where
  cost : ℕ  -- Cost per unit in yuan
  originalPrice : ℕ  -- Original selling price per unit in yuan
  initialSales : ℕ  -- Initial sales volume
  salesIncrease : ℕ  -- Increase in sales for every 1 yuan price reduction

/-- Calculates the profit for a given price reduction -/
def calculateProfit (p : ProductSales) (priceReduction : ℕ) : ℕ :=
  let newPrice := p.originalPrice - priceReduction
  let newSales := p.initialSales + p.salesIncrease * priceReduction
  (newPrice - p.cost) * newSales

/-- Theorem for the profit calculation with a 5 yuan price reduction -/
theorem profit_for_five_yuan_reduction (p : ProductSales) 
  (h1 : p.cost = 16) (h2 : p.originalPrice = 30) (h3 : p.initialSales = 200) (h4 : p.salesIncrease = 20) :
  calculateProfit p 5 = 2700 := by sorry

/-- Theorem for the optimal price reduction to achieve 2860 yuan profit -/
theorem optimal_price_reduction (p : ProductSales) 
  (h1 : p.cost = 16) (h2 : p.originalPrice = 30) (h3 : p.initialSales = 200) (h4 : p.salesIncrease = 20) :
  ∃ (x : ℕ), calculateProfit p x = 2860 ∧ 
    ∀ (y : ℕ), calculateProfit p y = 2860 → x ≤ y := by sorry

end profit_for_five_yuan_reduction_optimal_price_reduction_l1178_117815


namespace quadratic_function_a_range_l1178_117853

/-- Given a quadratic function y = (ax - 1)(x - a), this theorem proves that the range of a
    satisfying specific conditions about its roots and axis of symmetry is (0, 1). -/
theorem quadratic_function_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, (a * x - 1) * (x - a) > 0 ↔ x < a ∨ x > 1/a) ∧
  ((a^2 + 1) / (2 * a) > 0) ∧
  ¬(∀ x : ℝ, (a * x - 1) * (x - a) < 0 ↔ x < a ∨ x > 1/a)
  ↔ 0 < a ∧ a < 1 :=
by sorry

end quadratic_function_a_range_l1178_117853


namespace doritos_distribution_l1178_117862

theorem doritos_distribution (total_bags : ℕ) (doritos_fraction : ℚ) (num_piles : ℕ) : 
  total_bags = 200 →
  doritos_fraction = 2 / 5 →
  num_piles = 5 →
  (total_bags : ℚ) * doritos_fraction / num_piles = 16 := by
  sorry

end doritos_distribution_l1178_117862


namespace square_of_zero_is_not_positive_l1178_117819

theorem square_of_zero_is_not_positive : ¬ (∀ x : ℕ, x^2 > 0) := by
  sorry

end square_of_zero_is_not_positive_l1178_117819


namespace hyperbola_min_eccentricity_l1178_117836

/-- Given an ellipse and a hyperbola with coinciding foci, and a line intersecting
    the right branch of the hyperbola, when the eccentricity of the hyperbola is minimized,
    the equation of the hyperbola is x^2/5 - y^2/4 = 1 -/
theorem hyperbola_min_eccentricity 
  (ellipse : ℝ → ℝ → Prop)
  (hyperbola : ℝ → ℝ → ℝ → ℝ → Prop)
  (line : ℝ → ℝ → Prop)
  (h_ellipse : ∀ x y, ellipse x y ↔ x^2/16 + y^2/7 = 1)
  (h_hyperbola : ∀ a b x y, a > b ∧ b > 0 → (hyperbola a b x y ↔ x^2/a^2 - y^2/b^2 = 1))
  (h_foci : ∀ a b, hyperbola a b (-3) 0 ∧ hyperbola a b 3 0)
  (h_line : ∀ x y, line x y ↔ x - y = 1)
  (h_intersect : ∃ x y, hyperbola a b x y ∧ line x y ∧ x > 0)
  (h_min_eccentricity : ∀ a' b', (∃ x y, hyperbola a' b' x y ∧ line x y) → 
    (a^2 - b^2)/(a^2) ≤ (a'^2 - b'^2)/(a'^2)) :
  hyperbola 5 4 x y :=
sorry

end hyperbola_min_eccentricity_l1178_117836


namespace polynomial_product_l1178_117860

variables (a b : ℚ)

theorem polynomial_product (a b : ℚ) :
  (-3 * a^2 * b) * (-2 * a * b + b - 3) = 6 * a^3 * b^2 - 3 * a^2 * b^2 + 9 * a^2 * b :=
by sorry

end polynomial_product_l1178_117860


namespace smallest_n_inequality_l1178_117826

theorem smallest_n_inequality (x y z : ℝ) :
  (∃ (n : ℕ), ∀ (a b c : ℝ), (a^2 + b^2 + c^2) ≤ n * (a^4 + b^4 + c^4)) ∧
  (∀ (n : ℕ), (∀ (a b c : ℝ), (a^2 + b^2 + c^2) ≤ n * (a^4 + b^4 + c^4)) → n ≥ 3) ∧
  ((x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) :=
by sorry

end smallest_n_inequality_l1178_117826


namespace increasing_f_implies_k_leq_one_l1178_117841

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x + 1

-- State the theorem
theorem increasing_f_implies_k_leq_one :
  ∀ k : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 3 → f k x < f k y) → k ≤ 1 := by
  sorry

end increasing_f_implies_k_leq_one_l1178_117841


namespace deal_or_no_deal_boxes_l1178_117804

theorem deal_or_no_deal_boxes (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) : 
  total_boxes = 30 →
  high_value_boxes = 7 →
  (high_value_boxes : ℚ) / ((total_boxes - eliminated_boxes) : ℚ) ≥ 2 / 3 →
  eliminated_boxes ≥ 20 :=
by sorry

end deal_or_no_deal_boxes_l1178_117804


namespace milk_cost_l1178_117887

/-- The cost of a gallon of milk given the following conditions:
  * 4 pounds of coffee beans and 2 gallons of milk were bought
  * A pound of coffee beans costs $2.50
  * The total cost is $17
-/
theorem milk_cost (coffee_pounds : ℕ) (milk_gallons : ℕ) 
  (coffee_price : ℚ) (total_cost : ℚ) :
  coffee_pounds = 4 →
  milk_gallons = 2 →
  coffee_price = 5/2 →
  total_cost = 17 →
  ∃ (milk_price : ℚ), 
    milk_price * milk_gallons + coffee_price * coffee_pounds = total_cost ∧
    milk_price = 7/2 :=
by sorry

end milk_cost_l1178_117887


namespace school_dinosaur_cost_l1178_117834

def dinosaur_model_cost : ℕ := 100

def kindergarten_models : ℕ := 2
def elementary_models : ℕ := 2 * kindergarten_models
def high_school_models : ℕ := 3 * kindergarten_models

def total_models : ℕ := kindergarten_models + elementary_models + high_school_models

def discount_rate : ℚ :=
  if total_models > 10 then 1/10
  else if total_models > 5 then 1/20
  else 0

def discounted_price : ℚ := dinosaur_model_cost * (1 - discount_rate)

def total_cost : ℚ := total_models * discounted_price

theorem school_dinosaur_cost : total_cost = 1080 := by
  sorry

end school_dinosaur_cost_l1178_117834


namespace tarantula_perimeter_is_16_l1178_117875

/-- Represents a rectangle with width and height in inches -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the tarantula-shaped figure -/
structure TarantulaShape where
  body : Rectangle
  legs : Rectangle

/-- Calculates the perimeter of the tarantula-shaped figure -/
def tarantulaPerimeter (t : TarantulaShape) : ℝ :=
  2 * (t.body.width + t.body.height)

theorem tarantula_perimeter_is_16 :
  ∀ t : TarantulaShape,
    t.body.width = 3 ∧
    t.body.height = 10 ∧
    t.legs.width = 5 ∧
    t.legs.height = 3 →
    tarantulaPerimeter t = 16 := by
  sorry

#check tarantula_perimeter_is_16

end tarantula_perimeter_is_16_l1178_117875


namespace train_speed_calculation_l1178_117892

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1178_117892


namespace vehicle_speeds_l1178_117895

/-- Represents the initial speeds and distance of two vehicles --/
structure VehicleData where
  bus_speed : ℝ
  car_speed : ℝ
  final_distance : ℝ

/-- Calculates the total distance traveled by both vehicles --/
def total_distance (data : VehicleData) : ℝ :=
  2 * data.bus_speed + 2 * data.car_speed + 2 * data.bus_speed + 2 * (data.car_speed - 10)

/-- Theorem stating the initial speeds of the vehicles --/
theorem vehicle_speeds : ∃ (data : VehicleData),
  data.car_speed = data.bus_speed + 8 ∧
  data.final_distance = 384 ∧
  total_distance data = data.final_distance ∧
  data.bus_speed = 46.5 ∧
  data.car_speed = 54.5 := by
  sorry

end vehicle_speeds_l1178_117895


namespace max_value_product_l1178_117888

theorem max_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + 2*y + 3*z = 1) :
  x^2 * y^2 * z ≤ 4/16807 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ x^2 * y^2 * z = 4/16807 :=
by sorry

end max_value_product_l1178_117888


namespace min_sum_of_distances_min_sum_of_distances_achievable_l1178_117894

theorem min_sum_of_distances (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ Real.sqrt 20 := by
  sorry

theorem min_sum_of_distances_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = Real.sqrt 20 := by
  sorry

end min_sum_of_distances_min_sum_of_distances_achievable_l1178_117894


namespace football_season_length_l1178_117879

/-- The number of football games in one month -/
def games_per_month : ℝ := 323.0

/-- The total number of football games in the season -/
def total_games : ℕ := 5491

/-- The number of months in the football season -/
def season_months : ℕ := 17

/-- Theorem stating that the number of months in the season is 17 -/
theorem football_season_length :
  (total_games : ℝ) / games_per_month = season_months := by
  sorry

end football_season_length_l1178_117879


namespace unequal_probabilities_after_adding_balls_l1178_117823

/-- Represents the contents of the bag -/
structure BagContents where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing a specific color ball -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.white + bag.red : ℚ)

/-- The initial contents of the bag -/
def initialBag : BagContents := { white := 1, red := 2 }

/-- The bag after adding 1 white ball and 2 red balls -/
def updatedBag : BagContents := { white := initialBag.white + 1, red := initialBag.red + 2 }

theorem unequal_probabilities_after_adding_balls :
  probability updatedBag updatedBag.white ≠ probability updatedBag updatedBag.red := by
  sorry

end unequal_probabilities_after_adding_balls_l1178_117823


namespace remaining_backpack_price_l1178_117835

-- Define the problem parameters
def total_backpacks : ℕ := 48
def total_cost : ℕ := 576
def swap_meet_sold : ℕ := 17
def swap_meet_price : ℕ := 18
def dept_store_sold : ℕ := 10
def dept_store_price : ℕ := 25
def total_profit : ℕ := 442

-- Define the theorem
theorem remaining_backpack_price :
  let remaining_backpacks := total_backpacks - (swap_meet_sold + dept_store_sold)
  let swap_meet_revenue := swap_meet_sold * swap_meet_price
  let dept_store_revenue := dept_store_sold * dept_store_price
  let total_revenue := total_cost + total_profit
  let remaining_revenue := total_revenue - (swap_meet_revenue + dept_store_revenue)
  remaining_revenue / remaining_backpacks = 22 := by
  sorry

end remaining_backpack_price_l1178_117835


namespace x_plus_y_positive_l1178_117856

theorem x_plus_y_positive (x y : ℝ) (h1 : x * y < 0) (h2 : x > |y|) : x + y > 0 := by
  sorry

end x_plus_y_positive_l1178_117856


namespace nested_average_equals_seven_sixths_l1178_117893

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- The main theorem
theorem nested_average_equals_seven_sixths :
  avg3 (avg3 2 1 0) (avg2 1 2) 1 = 7/6 := by sorry

end nested_average_equals_seven_sixths_l1178_117893


namespace positive_real_solution_l1178_117806

theorem positive_real_solution (x : ℝ) : 
  x > 0 → x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16 → 
  15 * x^2 + 32 * x - 256 = 0 := by
  sorry

end positive_real_solution_l1178_117806


namespace max_value_of_f_l1178_117838

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 2 * Real.sin x + Real.cos x) / (Real.sin x + Real.sqrt (1 - Real.sin x))

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f x = M) :=
sorry

end max_value_of_f_l1178_117838


namespace sum_three_squares_not_7_mod_8_l1178_117872

theorem sum_three_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 := by
  sorry

end sum_three_squares_not_7_mod_8_l1178_117872


namespace adult_ticket_cost_l1178_117890

/-- Proves that the cost of an adult ticket is $5.50 given the specified conditions -/
theorem adult_ticket_cost : 
  let child_ticket_cost : ℝ := 3.50
  let total_tickets : ℕ := 21
  let total_cost : ℝ := 83.50
  let child_tickets : ℕ := 16
  let adult_tickets : ℕ := total_tickets - child_tickets
  let adult_ticket_cost : ℝ := (total_cost - child_ticket_cost * child_tickets) / adult_tickets
  adult_ticket_cost = 5.50 := by sorry

end adult_ticket_cost_l1178_117890


namespace electricity_price_correct_l1178_117802

/-- The electricity price per kWh in Coco's town -/
def electricity_price : ℝ := 0.1

/-- Coco's oven consumption rate in kWh -/
def oven_consumption_rate : ℝ := 2.4

/-- The number of hours Coco used his oven -/
def hours_used : ℝ := 25

/-- The amount Coco paid for using his oven -/
def amount_paid : ℝ := 6

/-- Theorem stating that the electricity price is correct -/
theorem electricity_price_correct : 
  electricity_price = amount_paid / (oven_consumption_rate * hours_used) :=
by sorry

end electricity_price_correct_l1178_117802


namespace reciprocal_sum_contains_two_l1178_117873

theorem reciprocal_sum_contains_two (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1 →
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := by
sorry

end reciprocal_sum_contains_two_l1178_117873


namespace trajectory_and_intersection_l1178_117816

/-- The trajectory of point P given fixed points A and B -/
def trajectory (x y : ℝ) : Prop :=
  x^2 + y^2/2 = 1 ∧ x ≠ 1 ∧ x ≠ -1

/-- The line intersecting the trajectory -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 1

/-- Theorem stating the properties of the trajectory and intersection -/
theorem trajectory_and_intersection :
  ∀ (x y : ℝ),
  (∀ (x' y' : ℝ), (y' / (x' + 1)) * (y' / (x' - 1)) = -2 → trajectory x' y') ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    trajectory x1 y1 ∧ trajectory x2 y2 ∧
    intersecting_line x1 y1 ∧ intersecting_line x2 y2 ∧
    ((x1 - x2)^2 + (y1 - y2)^2)^(1/2 : ℝ) = 4 * Real.sqrt 2 / 3) :=
by sorry

end trajectory_and_intersection_l1178_117816


namespace angle_b_measure_l1178_117896

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180
  isosceles : B = C
  angle_relation : C = 3 * A

/-- Theorem: In the given isosceles triangle, angle B measures 540/7 degrees -/
theorem angle_b_measure (t : IsoscelesTriangle) : t.B = 540 / 7 := by
  sorry

end angle_b_measure_l1178_117896


namespace cistern_leak_empty_time_l1178_117899

/-- Given a cistern with normal fill time and leak-affected fill time, 
    calculate the time it takes for the leak to empty the full cistern. -/
theorem cistern_leak_empty_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 12) 
  (h2 : leak_fill_time = normal_fill_time + 2) : 
  (1 / ((1 / normal_fill_time) - (1 / leak_fill_time))) = 84 := by
  sorry

end cistern_leak_empty_time_l1178_117899


namespace total_gifts_received_l1178_117830

def gifts_from_emilio : ℕ := 11
def gifts_from_jorge : ℕ := 6
def gifts_from_pedro : ℕ := 4

theorem total_gifts_received : 
  gifts_from_emilio + gifts_from_jorge + gifts_from_pedro = 21 := by
  sorry

end total_gifts_received_l1178_117830


namespace train_speed_ratio_l1178_117808

/-- Prove that the ratio of the speeds of two trains is 2:1 given specific conditions --/
theorem train_speed_ratio :
  let train_length : ℝ := 150  -- Length of each train in meters
  let crossing_time : ℝ := 8   -- Time taken to cross in seconds
  let faster_speed : ℝ := 90   -- Speed of faster train in km/h

  let total_distance : ℝ := 2 * train_length
  let relative_speed : ℝ := total_distance / crossing_time
  let faster_speed_ms : ℝ := faster_speed * 1000 / 3600
  let slower_speed_ms : ℝ := relative_speed - faster_speed_ms

  (faster_speed_ms / slower_speed_ms : ℝ) = 2 := by sorry

end train_speed_ratio_l1178_117808


namespace range_of_m_l1178_117852

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) → 
  (m ≥ 4 ∨ m ≤ 0) := by
sorry

end range_of_m_l1178_117852


namespace sufficient_not_necessary_l1178_117827

theorem sufficient_not_necessary (x y a m : ℝ) :
  (∀ x y a m : ℝ, (|x - a| < m ∧ |y - a| < m) → |x - y| < 2*m) ∧
  (∃ x y a m : ℝ, |x - y| < 2*m ∧ ¬(|x - a| < m ∧ |y - a| < m)) :=
sorry

end sufficient_not_necessary_l1178_117827


namespace trumpet_section_fraction_l1178_117866

/-- The fraction of students in the trumpet section -/
def trumpet_fraction : ℝ := sorry

/-- The fraction of students in the trombone section -/
def trombone_fraction : ℝ := 0.12

/-- The fraction of students in either the trumpet or trombone section -/
def trumpet_or_trombone_fraction : ℝ := 0.63

theorem trumpet_section_fraction :
  trumpet_fraction = 0.51 :=
by
  sorry

end trumpet_section_fraction_l1178_117866


namespace value_of_y_l1178_117801

theorem value_of_y : ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end value_of_y_l1178_117801


namespace circle_center_and_radius_l1178_117844

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a circle equation, returns its properties (center and radius) -/
def circle_properties (eq : CircleEquation) : CircleProperties :=
  sorry

theorem circle_center_and_radius 
  (eq : CircleEquation) 
  (h : eq = ⟨-6, 0, 0⟩) : 
  circle_properties eq = ⟨(3, 0), 3⟩ :=
sorry

end circle_center_and_radius_l1178_117844


namespace daily_rental_cost_l1178_117889

/-- Represents the daily car rental cost problem -/
theorem daily_rental_cost (total_cost : ℝ) (miles_driven : ℝ) (per_mile_rate : ℝ) :
  total_cost = 46.12 ∧
  miles_driven = 214.0 ∧
  per_mile_rate = 0.08 →
  ∃ (daily_rate : ℝ), daily_rate = 29.00 ∧ total_cost = daily_rate + miles_driven * per_mile_rate :=
by sorry

end daily_rental_cost_l1178_117889


namespace power_sum_equality_l1178_117897

theorem power_sum_equality : -2^2005 + (-2)^2006 + 2^2007 - 2^2008 = 2^2005 := by
  sorry

end power_sum_equality_l1178_117897


namespace store_traffic_proof_l1178_117805

/-- The number of people who entered the store in the first hour -/
def first_hour_entries : ℕ := 94

/-- The number of people who entered the store in the second hour -/
def second_hour_entries : ℕ := 18

/-- The number of people who left the store in the second hour -/
def second_hour_exits : ℕ := 9

/-- The number of people in the store after two hours -/
def final_count : ℕ := 76

/-- The number of people who left during the first hour -/
def first_hour_exits : ℕ := 27

theorem store_traffic_proof :
  first_hour_entries - first_hour_exits + second_hour_entries - second_hour_exits = final_count :=
by sorry

end store_traffic_proof_l1178_117805


namespace square_pyramid_dihedral_angle_cosine_l1178_117886

/-- A pyramid with a square base and specific properties -/
structure SquarePyramid where
  -- The length of the congruent edges
  edge_length : ℝ
  -- The measure of the dihedral angle between faces PQR and PRS
  dihedral_angle : ℝ
  -- Angle QPR is 45°
  angle_QPR_is_45 : angle_QPR = Real.pi / 4
  -- The base is square (implied by the problem setup)
  base_is_square : True

/-- The theorem statement -/
theorem square_pyramid_dihedral_angle_cosine 
  (P : SquarePyramid) 
  (a b : ℝ) 
  (h : Real.cos P.dihedral_angle = a + Real.sqrt b) : 
  a + b = 1 := by
  sorry

end square_pyramid_dihedral_angle_cosine_l1178_117886


namespace imaginary_part_of_z_l1178_117832

-- Define the complex number z
def z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)

-- Theorem stating that the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
  sorry

end imaginary_part_of_z_l1178_117832


namespace prob_at_least_one_white_correct_l1178_117812

def total_balls : ℕ := 9
def red_balls : ℕ := 5
def white_balls : ℕ := 4

def prob_at_least_one_white : ℚ := 13 / 18

theorem prob_at_least_one_white_correct :
  let prob_two_red : ℚ := (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))
  1 - prob_two_red = prob_at_least_one_white := by
  sorry

end prob_at_least_one_white_correct_l1178_117812


namespace complex_power_magnitude_l1178_117813

theorem complex_power_magnitude (z : ℂ) (h : z = 4/5 + 3/5 * I) :
  Complex.abs (z^8) = 1 := by sorry

end complex_power_magnitude_l1178_117813


namespace benny_pie_price_l1178_117864

/-- Calculates the price per pie needed to achieve a desired profit given the number and cost of pumpkin and cherry pies -/
def price_per_pie (Np Nc : ℕ) (Cp Cc Pr : ℚ) : ℚ :=
  (Np * Cp + Nc * Cc + Pr) / (Np + Nc)

theorem benny_pie_price :
  let Np : ℕ := 10  -- Number of pumpkin pies
  let Nc : ℕ := 12  -- Number of cherry pies
  let Cp : ℚ := 3   -- Cost to make each pumpkin pie
  let Cc : ℚ := 5   -- Cost to make each cherry pie
  let Pr : ℚ := 20  -- Desired profit
  price_per_pie Np Nc Cp Cc Pr = 5 := by
sorry

end benny_pie_price_l1178_117864


namespace passing_grade_fraction_l1178_117877

theorem passing_grade_fraction (students_A students_B students_C students_D students_F : ℚ) :
  students_A = 1/4 →
  students_B = 1/2 →
  students_C = 1/8 →
  students_D = 1/12 →
  students_F = 1/24 →
  students_A + students_B + students_C = 7/8 := by
  sorry

end passing_grade_fraction_l1178_117877


namespace lcm_gcd_problem_l1178_117847

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 5040 → 
  Nat.gcd a b = 24 → 
  a = 240 → 
  b = 504 := by sorry

end lcm_gcd_problem_l1178_117847


namespace harkamal_payment_l1178_117824

/-- Calculates the final amount paid after discount and tax --/
def calculate_final_amount (fruits : List (String × ℕ × ℕ)) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := (fruits.map (λ (_, quantity, price) => quantity * price)).sum
  let discounted_total := total_cost * (1 - discount_rate)
  let final_amount := discounted_total * (1 + tax_rate)
  final_amount

/-- Theorem stating the final amount Harkamal paid --/
theorem harkamal_payment : 
  let fruits := [
    ("Grapes", 8, 70),
    ("Mangoes", 9, 55),
    ("Apples", 4, 40),
    ("Oranges", 6, 30),
    ("Pineapples", 2, 90),
    ("Cherries", 5, 100)
  ]
  let discount_rate : ℚ := 5 / 100
  let tax_rate : ℚ := 10 / 100
  calculate_final_amount fruits discount_rate tax_rate = 2168375 / 1000 := by
  sorry

#eval calculate_final_amount [
  ("Grapes", 8, 70),
  ("Mangoes", 9, 55),
  ("Apples", 4, 40),
  ("Oranges", 6, 30),
  ("Pineapples", 2, 90),
  ("Cherries", 5, 100)
] (5 / 100) (10 / 100)

end harkamal_payment_l1178_117824


namespace point_inside_circle_l1178_117891

theorem point_inside_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := by
sorry

end point_inside_circle_l1178_117891


namespace balloon_difference_l1178_117820

-- Define the initial conditions
def allan_initial : ℕ := 6
def jake_initial : ℕ := 2
def jake_bought : ℕ := 3
def allan_bought : ℕ := 4
def claire_from_jake : ℕ := 2
def claire_from_allan : ℕ := 3

-- Theorem statement
theorem balloon_difference :
  (allan_initial + allan_bought - claire_from_allan) -
  (jake_initial + jake_bought - claire_from_jake) = 4 := by
  sorry

end balloon_difference_l1178_117820


namespace fraction_multiplication_and_addition_l1178_117817

theorem fraction_multiplication_and_addition : (2 : ℚ) / 9 * 5 / 11 + 1 / 3 = 43 / 99 := by
  sorry

end fraction_multiplication_and_addition_l1178_117817


namespace second_scenario_cost_l1178_117803

/-- The cost of a single shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a single trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a single tie -/
def tie_cost : ℝ := sorry

/-- The first scenario: 6 shirts, 4 trousers, and 2 ties cost $80 -/
def scenario1 : Prop := 6 * shirt_cost + 4 * trouser_cost + 2 * tie_cost = 80

/-- The third scenario: 5 shirts, 3 trousers, and 2 ties cost $110 -/
def scenario3 : Prop := 5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 110

/-- Theorem: Given scenario1 and scenario3, the cost of 4 shirts, 2 trousers, and 2 ties is $50 -/
theorem second_scenario_cost (h1 : scenario1) (h3 : scenario3) : 
  4 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50 := by sorry

end second_scenario_cost_l1178_117803


namespace tank_inflow_rate_l1178_117880

/-- Given two tanks with equal capacity, prove the inflow rate of the slower-filling tank -/
theorem tank_inflow_rate (capacity : ℝ) (fast_rate slow_rate : ℝ) (time_diff : ℝ) :
  capacity > 0 →
  fast_rate > 0 →
  slow_rate > 0 →
  time_diff > 0 →
  capacity / fast_rate + time_diff = capacity / slow_rate →
  capacity = 20 →
  fast_rate = 4 →
  time_diff = 5 →
  slow_rate = 2 := by sorry

end tank_inflow_rate_l1178_117880


namespace tangent_line_power_function_l1178_117811

theorem tangent_line_power_function (n : ℝ) :
  (2 : ℝ) ^ n = 8 →
  let f := λ x : ℝ => x ^ n
  let f' := λ x : ℝ => n * x ^ (n - 1)
  let tangent_slope := f' 2
  let tangent_eq := λ x y : ℝ => tangent_slope * (x - 2) = y - 8
  tangent_eq = λ x y : ℝ => 12 * x - y - 16 = 0 := by sorry

end tangent_line_power_function_l1178_117811


namespace curve_classification_l1178_117874

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop := 3 * x^2 + m * y^2 = 1

-- Define the possible curve types
inductive CurveType
  | TwoLines
  | Ellipse
  | Circle
  | Hyperbola

-- Theorem statement
theorem curve_classification (m : ℝ) : 
  ∃ (t : CurveType), ∀ (x y : ℝ), curve_equation x y m → 
    (t = CurveType.TwoLines ∨ 
     t = CurveType.Ellipse ∨ 
     t = CurveType.Circle ∨ 
     t = CurveType.Hyperbola) :=
sorry

end curve_classification_l1178_117874


namespace circle_center_sum_l1178_117821

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 14*y - 11 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h + 14*k - 11)) →
  h + k = -4 := by
sorry

end circle_center_sum_l1178_117821


namespace martin_position_l1178_117898

/-- Represents a queue with the given properties -/
structure Queue where
  total : ℕ
  martin_pos : ℕ
  friend1_pos : ℕ
  friend2_pos : ℕ
  total_multiple_of_3 : total % 3 = 0
  martin_centered : martin_pos - 1 = total - martin_pos
  friend1_behind : friend1_pos > martin_pos
  friend2_behind : friend2_pos > martin_pos
  friend1_is_19th : friend1_pos = 19
  friend2_is_28th : friend2_pos = 28

/-- The theorem stating Martin's position in the queue -/
theorem martin_position (q : Queue) : q.martin_pos = 17 := by
  sorry

end martin_position_l1178_117898


namespace train_travel_time_l1178_117800

theorem train_travel_time (initial_time : ℝ) (increase1 increase2 increase3 : ℝ) :
  initial_time = 19.5 ∧ 
  increase1 = 0.3 ∧ 
  increase2 = 0.25 ∧ 
  increase3 = 0.2 → 
  initial_time / ((1 + increase1) * (1 + increase2) * (1 + increase3)) = 10 := by
sorry

end train_travel_time_l1178_117800


namespace intersection_point_on_circle_l1178_117807

theorem intersection_point_on_circle (m : ℝ) :
  ∃ (x y r : ℝ),
    r > 0 ∧
    m * x + y + 2 * m = 0 ∧
    x - m * y + 2 * m = 0 ∧
    (x - 2)^2 + (y - 4)^2 = r^2 →
    2 * Real.sqrt 2 ≤ r ∧ r ≤ 4 * Real.sqrt 2 := by
  sorry

end intersection_point_on_circle_l1178_117807


namespace real_part_of_complex_fraction_l1178_117859

theorem real_part_of_complex_fraction (θ : ℝ) :
  let z : ℂ := Complex.exp (θ * Complex.I)
  Complex.abs z = 1 →
  (1 / (2 - z)).re = (2 - Real.cos θ) / (5 - 4 * Real.cos θ) := by
sorry

end real_part_of_complex_fraction_l1178_117859


namespace trapezoid_area_equality_l1178_117848

/-- Represents a trapezoid divided into triangles and a pentagon as described in the problem -/
structure DividedTrapezoid where
  /-- Area of the central pentagon -/
  Q : ℝ
  /-- Area of the triangle adjacent to one lateral side -/
  s₁ : ℝ
  /-- Area of the triangle adjacent to the shorter base -/
  s₂ : ℝ
  /-- Area of the triangle adjacent to the other lateral side -/
  s₃ : ℝ
  /-- Area of the triangle between s₁ and s₂ -/
  x : ℝ
  /-- Area of the triangle between s₂ and s₃ -/
  y : ℝ
  /-- The sum of areas of triangles adjacent to one side and the shorter base equals half the sum of x, y, s₂, and Q -/
  h₁ : s₁ + x + s₂ = (x + y + s₂ + Q) / 2
  /-- The sum of areas of triangles adjacent to the shorter base and the other side equals half the sum of x, y, s₂, and Q -/
  h₂ : s₂ + y + s₃ = (x + y + s₂ + Q) / 2

/-- The sum of the areas of the three triangles adjacent to the lateral sides and the shorter base 
    of the trapezoid is equal to the area of the pentagon -/
theorem trapezoid_area_equality (t : DividedTrapezoid) : t.s₁ + t.s₂ + t.s₃ = t.Q := by
  sorry

end trapezoid_area_equality_l1178_117848


namespace dodecagon_diagonals_l1178_117829

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: The number of distinct diagonals in a convex dodecagon is 54 -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l1178_117829


namespace two_digit_number_property_l1178_117876

theorem two_digit_number_property (c d h : ℕ) (m : ℕ) (y : ℤ) :
  c < 10 →
  d < 10 →
  m = 10 * c + d →
  m = h * (c + d) →
  (10 * d + c : ℤ) = y * (c + d) →
  y = 12 - h :=
by sorry

end two_digit_number_property_l1178_117876


namespace sphere_volume_calculation_l1178_117858

-- Define the sphere and plane
def Sphere : Type := Unit
def Plane : Type := Unit

-- Define the properties of the intersection
def intersection_diameter (s : Sphere) (p : Plane) : ℝ := 6

-- Define the distance from the center of the sphere to the plane
def center_to_plane_distance (s : Sphere) (p : Plane) : ℝ := 4

-- Define the volume of a sphere
def sphere_volume (s : Sphere) : ℝ := sorry

-- Theorem statement
theorem sphere_volume_calculation (s : Sphere) (p : Plane) :
  sphere_volume s = (500 * Real.pi) / 3 :=
by sorry

end sphere_volume_calculation_l1178_117858


namespace fraction_equality_l1178_117850

theorem fraction_equality (x : ℝ) : (3 + x) / (5 + x) = (1 + x) / (2 + x) ↔ x = 1 := by
  sorry

end fraction_equality_l1178_117850


namespace not_equivalent_to_0_0000042_l1178_117846

theorem not_equivalent_to_0_0000042 : ¬ (2.1 * 10^(-6) = 0.0000042) :=
by
  have h1 : 0.0000042 = 4.2 * 10^(-6) := by sorry
  sorry

end not_equivalent_to_0_0000042_l1178_117846


namespace smallest_cube_box_volume_l1178_117869

/-- Represents the dimensions of a pyramid -/
structure PyramidDimensions where
  height : ℝ
  baseSide : ℝ

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ := side ^ 3

/-- Theorem: The smallest cube-shaped box that can contain a pyramid with given dimensions has a volume of 3375 cubic inches -/
theorem smallest_cube_box_volume
  (pyramid : PyramidDimensions)
  (h_height : pyramid.height = 15)
  (h_base : pyramid.baseSide = 14) :
  cubeVolume (max pyramid.height pyramid.baseSide) = 3375 := by
  sorry

#eval cubeVolume 15  -- Should output 3375

end smallest_cube_box_volume_l1178_117869


namespace absolute_value_equation_solution_l1178_117881

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- The unique solution is y = 2
  use 2
  constructor
  · -- Prove that y = 2 satisfies the equation
    sorry
  · -- Prove that any y satisfying the equation must equal 2
    sorry

end absolute_value_equation_solution_l1178_117881


namespace reading_time_calculation_l1178_117878

theorem reading_time_calculation (pages_book1 pages_book2 : ℕ) 
  (time_book1 time_book2 : ℚ) (pages_to_read1 pages_to_read2 : ℕ) :
  pages_book1 = 4 →
  pages_book2 = 5 →
  time_book1 = 2 →
  time_book2 = 3 →
  pages_to_read1 = 36 →
  pages_to_read2 = 25 →
  (pages_to_read1 * (time_book1 / pages_book1) + 
   pages_to_read2 * (time_book2 / pages_book2)) = 33 :=
by
  sorry

end reading_time_calculation_l1178_117878


namespace pastry_sale_revenue_l1178_117855

/-- Calculates the total money made from selling discounted pastries. -/
theorem pastry_sale_revenue (cupcake_price cookie_price : ℚ)
  (cupcakes_sold cookies_sold : ℕ) : 
  cupcake_price = 3 ∧ cookie_price = 2 ∧ cupcakes_sold = 16 ∧ cookies_sold = 8 →
  (cupcake_price / 2 * cupcakes_sold + cookie_price / 2 * cookies_sold : ℚ) = 32 := by
  sorry

#check pastry_sale_revenue

end pastry_sale_revenue_l1178_117855


namespace circle_line_intersection_sum_l1178_117861

/-- Given a circle with radius 4 centered at the origin and a line y = 4 - (2 - √3)x
    intersecting the circle at points A and B, the sum of the length of segment AB
    and the length of the shorter arc AB is 4√(2 - √3) + (2π/3) -/
theorem circle_line_intersection_sum (A B : ℝ × ℝ) : 
  let r : ℝ := 4
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | y = 4 - (2 - Real.sqrt 3) * x}
  A ∈ circle ∧ A ∈ line ∧ B ∈ circle ∧ B ∈ line ∧ A ≠ B →
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := angle * r
  segment_length + min arc_length (2 * π * r - arc_length) = 4 * Real.sqrt (2 - Real.sqrt 3) + (2 * π / 3) := by
  sorry

end circle_line_intersection_sum_l1178_117861


namespace lily_lottery_tickets_l1178_117868

/-- Represents the number of lottery tickets sold -/
def n : ℕ := 5

/-- The price of the i-th ticket -/
def ticket_price (i : ℕ) : ℕ := i

/-- The total amount collected from selling n tickets -/
def total_collected (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The profit Lily keeps -/
def profit : ℕ := 4

/-- The prize money for the lottery winner -/
def prize : ℕ := 11

theorem lily_lottery_tickets :
  (total_collected n = prize + profit) ∧
  (∀ m : ℕ, m ≠ n → total_collected m ≠ prize + profit) :=
by sorry

end lily_lottery_tickets_l1178_117868


namespace bookstore_repricing_l1178_117870

theorem bookstore_repricing (n : Nat) (p₁ p₂ : Nat) (h₁ : n = 1452) (h₂ : p₁ = 42) (h₃ : p₂ = 45) :
  (n * p₁) % p₂ = 9 := by
  sorry

end bookstore_repricing_l1178_117870


namespace binomial_expansion_coefficient_l1178_117831

theorem binomial_expansion_coefficient (a : ℝ) : 
  (6 : ℕ) * a^5 * (Real.sqrt 3 / 6) = -Real.sqrt 3 → a = -1 := by
  sorry

end binomial_expansion_coefficient_l1178_117831


namespace power_function_through_point_l1178_117843

/-- Given a power function f(x) = x^a that passes through the point (2, 4), prove that f(x) = x^2 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →
  f 2 = 4 →
  ∀ x, f x = x ^ 2 :=
by sorry

end power_function_through_point_l1178_117843


namespace smallest_result_l1178_117839

def S : Finset Nat := {3, 5, 7, 11, 13, 17}

def process (a b c : Nat) : Nat :=
  max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

def valid_selection (a b c : Nat) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : Nat), valid_selection a b c ∧
    process a b c = 36 ∧
    ∀ (x y z : Nat), valid_selection x y z → process x y z ≥ 36 := by
  sorry

end smallest_result_l1178_117839


namespace adams_to_ricks_ratio_l1178_117825

/-- Represents the cost of lunch for each person -/
structure LunchCost where
  adam : ℚ
  rick : ℚ
  jose : ℚ

/-- The conditions of the lunch scenario -/
def lunch_scenario (cost : LunchCost) : Prop :=
  cost.rick = cost.jose ∧ 
  cost.jose = 45 ∧
  cost.adam + cost.rick + cost.jose = 120

/-- The theorem stating the ratio of Adam's lunch cost to Rick's lunch cost -/
theorem adams_to_ricks_ratio (cost : LunchCost) :
  lunch_scenario cost → cost.adam / cost.rick = 2 / 3 := by
  sorry

end adams_to_ricks_ratio_l1178_117825


namespace simplify_trig_expression_l1178_117885

theorem simplify_trig_expression :
  let tan60 : ℝ := Real.sqrt 3
  let cot60 : ℝ := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7/3 := by
  sorry

end simplify_trig_expression_l1178_117885


namespace tie_record_score_difference_l1178_117865

/-- The league record average score per player per round -/
def league_record : ℕ := 287

/-- The number of players in a team -/
def players_per_team : ℕ := 4

/-- The number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- The total score of George's team after 9 rounds -/
def team_score_9_rounds : ℕ := 10440

/-- The minimum average score needed per player in the final round to tie the record -/
def min_avg_score_final_round : ℕ := (league_record * players_per_team * rounds_per_season - team_score_9_rounds) / players_per_team

/-- The difference between the league record average and the minimum average score needed -/
def score_difference : ℕ := league_record - min_avg_score_final_round

theorem tie_record_score_difference : score_difference = 27 := by
  sorry

end tie_record_score_difference_l1178_117865


namespace trajectory_equation_l1178_117842

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_O x y

-- Define the midpoint M of PQ
def midpoint_M (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), point_Q qx qy ∧ x = (qx + point_P.1) / 2 ∧ y = (qy + point_P.2) / 2

-- Theorem: The trajectory of M forms the equation (x + 1/2)² + y² = 1
theorem trajectory_equation :
  ∀ (x y : ℝ), midpoint_M x y ↔ (x + 1/2)^2 + y^2 = 1 :=
sorry

end trajectory_equation_l1178_117842


namespace r_profit_share_l1178_117822

/-- Represents a partner in the business partnership --/
inductive Partner
| P
| Q
| R

/-- Represents the initial share ratio of each partner --/
def initial_share_ratio (p : Partner) : Rat :=
  match p with
  | Partner.P => 1/2
  | Partner.Q => 1/3
  | Partner.R => 1/4

/-- The number of months after which P withdraws half of their capital --/
def withdrawal_month : Nat := 2

/-- The total number of months for the profit calculation --/
def total_months : Nat := 12

/-- The total profit to be divided --/
def total_profit : ℚ := 378

/-- Calculates the effective share ratio for a partner over the entire period --/
def effective_share_ratio (p : Partner) : Rat :=
  match p with
  | Partner.P => (initial_share_ratio Partner.P * withdrawal_month + initial_share_ratio Partner.P / 2 * (total_months - withdrawal_month)) / total_months
  | _ => initial_share_ratio p

/-- Calculates a partner's share of the profit --/
def profit_share (p : Partner) : ℚ :=
  (effective_share_ratio p / (effective_share_ratio Partner.P + effective_share_ratio Partner.Q + effective_share_ratio Partner.R)) * total_profit

/-- The main theorem stating R's share of the profit --/
theorem r_profit_share : profit_share Partner.R = 108 := by
  sorry


end r_profit_share_l1178_117822


namespace max_product_under_constraints_l1178_117809

theorem max_product_under_constraints (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : 10 * x + 15 * y = 150) (h2 : x^2 + y^2 ≤ 100) :
  x * y ≤ 37.5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
  10 * x₀ + 15 * y₀ = 150 ∧ x₀^2 + y₀^2 ≤ 100 ∧ x₀ * y₀ = 37.5 :=
sorry

end max_product_under_constraints_l1178_117809


namespace two_counterexamples_l1178_117845

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has any digit equal to 0 -/
def has_zero_digit (n : ℕ) : Bool := sorry

/-- The main theorem -/
theorem two_counterexamples : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, sum_of_digits n = 4 ∧ ¬has_zero_digit n ∧ ¬Nat.Prime n) ∧ 
    s.card = 2 := by sorry

end two_counterexamples_l1178_117845


namespace equal_numbers_from_equal_powers_l1178_117837

theorem equal_numbers_from_equal_powers (a : Fin 17 → ℕ) 
  (h : ∀ i : Fin 16, (a i) ^ (a (i + 1)) = (a (i + 1)) ^ (a ((i + 2) % 17))) : 
  ∀ i j : Fin 17, a i = a j := by
  sorry

end equal_numbers_from_equal_powers_l1178_117837


namespace race_graph_representation_l1178_117840

-- Define the types of contestants
inductive Contestant
| Snail
| Horse

-- Define the movement pattern
structure MovementPattern where
  contestant : Contestant
  isConsistent : Bool
  hasRest : Bool
  initialSpeed : ℕ
  finalPosition : ℕ

-- Define the graph types
inductive GraphType
| FluctuatingSpeed
| SteadySlowWinnerVsFastStartStop
| ConsistentlyIncreasing

-- Define the race outcome
def raceOutcome (snailPattern : MovementPattern) (horsePattern : MovementPattern) : GraphType :=
  if snailPattern.isConsistent ∧ 
     snailPattern.initialSpeed < horsePattern.initialSpeed ∧ 
     horsePattern.hasRest ∧ 
     snailPattern.finalPosition > horsePattern.finalPosition
  then GraphType.SteadySlowWinnerVsFastStartStop
  else GraphType.FluctuatingSpeed

-- Theorem statement
theorem race_graph_representation 
  (snail : MovementPattern) 
  (horse : MovementPattern) 
  (h_snail_contestant : snail.contestant = Contestant.Snail)
  (h_horse_contestant : horse.contestant = Contestant.Horse)
  (h_snail_consistent : snail.isConsistent = true)
  (h_snail_slow : snail.initialSpeed < horse.initialSpeed)
  (h_horse_rest : horse.hasRest = true)
  (h_snail_wins : snail.finalPosition > horse.finalPosition) :
  raceOutcome snail horse = GraphType.SteadySlowWinnerVsFastStartStop :=
by sorry

end race_graph_representation_l1178_117840


namespace p_and_q_true_l1178_117884

theorem p_and_q_true (a b c : ℝ) : 
  ((a > b → a + c > b + c) ∧ ((a > b ∧ b > 0) → a * c > b * c)) := by
  sorry

end p_and_q_true_l1178_117884


namespace jessica_roses_cut_l1178_117818

/-- The number of roses Jessica cut from her garden -/
def roses_cut : ℕ := 99

theorem jessica_roses_cut :
  let initial_roses : ℕ := 17
  let roses_thrown : ℕ := 8
  let roses_now : ℕ := 42
  let roses_given : ℕ := 6
  (initial_roses - roses_thrown + roses_cut / 3 = roses_now) ∧
  (roses_cut / 3 + roses_given = roses_now - initial_roses + roses_thrown + roses_given) →
  roses_cut = 99 := by
sorry

end jessica_roses_cut_l1178_117818


namespace rising_number_Q_l1178_117814

/-- Definition of a rising number -/
def is_rising_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = 1000*a + 100*b + 10*c + d ∧ 
  a < b ∧ b < c ∧ c < d ∧ a + d = b + c

/-- Function F as defined in the problem -/
def F (m : ℕ) : ℚ :=
  let m' := 1000*(m/10%10) + 100*(m/100%10) + 10*(m/1000) + (m%10)
  (m' - m) / 99

/-- Main theorem -/
theorem rising_number_Q (P Q : ℕ) (x y z t : ℕ) : 
  is_rising_number P ∧ 
  is_rising_number Q ∧
  P = 1000 + 100*x + 10*y + z ∧
  Q = 1000*x + 100*t + 60 + z ∧
  ∃ (k : ℤ), F P + F Q = k * 7 →
  Q = 3467 := by sorry

end rising_number_Q_l1178_117814


namespace complex_number_in_fourth_quadrant_l1178_117833

theorem complex_number_in_fourth_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i^3) / (1 - i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l1178_117833


namespace greatest_divisor_with_remainders_l1178_117883

theorem greatest_divisor_with_remainders : Nat.gcd (3461 - 23) (4783 - 41) = 2 := by
  sorry

end greatest_divisor_with_remainders_l1178_117883


namespace symmetry_implies_axis_1_5_l1178_117871

/-- A function f is symmetric about the line x = 1.5 if f(x) = f(3 - x) for all x. -/
def is_symmetric_about_1_5 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (3 - x)

/-- The line x = 1.5 is an axis of symmetry for a function f if 
    for any point (x, f(x)) on the graph, the point (3 - x, f(x)) is also on the graph. -/
def is_axis_of_symmetry_1_5 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y → f (3 - x) = y

theorem symmetry_implies_axis_1_5 (f : ℝ → ℝ) :
  is_symmetric_about_1_5 f → is_axis_of_symmetry_1_5 f :=
by
  sorry

end symmetry_implies_axis_1_5_l1178_117871


namespace water_displaced_volume_squared_l1178_117851

/-- The square of the volume of water displaced by a cube in a cylindrical barrel -/
theorem water_displaced_volume_squared
  (barrel_radius : ℝ)
  (barrel_height : ℝ)
  (cube_side_length : ℝ)
  (h_radius : barrel_radius = 5)
  (h_height : barrel_height = 10)
  (h_side : cube_side_length = 6) :
  let diagonal := cube_side_length * Real.sqrt 3
  let triangle_side := barrel_radius * Real.sqrt 3
  let tetrahedron_leg := (5 * Real.sqrt 6) / 2
  let volume := (375 * Real.sqrt 6) / 8
  volume ^ 2 = 843750 / 64 := by
  sorry

#eval (843750 / 64 : Float)  -- Should output approximately 13141.855

end water_displaced_volume_squared_l1178_117851


namespace intersection_M_N_l1178_117867

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_M_N : M ∩ N = {x | -2 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l1178_117867
