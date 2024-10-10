import Mathlib

namespace endpoint_from_midpoint_and_one_endpoint_l4105_410506

/-- Given a line segment with midpoint (3, 4) and one endpoint at (0, -1), 
    the other endpoint is at (6, 9). -/
theorem endpoint_from_midpoint_and_one_endpoint :
  let midpoint : ℝ × ℝ := (3, 4)
  let endpoint1 : ℝ × ℝ := (0, -1)
  let endpoint2 : ℝ × ℝ := (6, 9)
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end endpoint_from_midpoint_and_one_endpoint_l4105_410506


namespace shopkeeper_bananas_l4105_410547

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) : 
  oranges = 600 →
  (510 : ℝ) + 0.95 * bananas = 0.89 * (oranges + bananas) →
  bananas = 400 := by
sorry

end shopkeeper_bananas_l4105_410547


namespace number_division_problem_l4105_410555

theorem number_division_problem : ∃ x : ℝ, x / 5 = 75 + x / 6 ∧ x = 2250 := by
  sorry

end number_division_problem_l4105_410555


namespace morning_speed_calculation_l4105_410559

theorem morning_speed_calculation 
  (total_time : ℝ) 
  (distance : ℝ) 
  (evening_speed : ℝ) 
  (h1 : total_time = 1) 
  (h2 : distance = 18) 
  (h3 : evening_speed = 30) : 
  ∃ morning_speed : ℝ, 
    distance / morning_speed + distance / evening_speed = total_time ∧ 
    morning_speed = 45 := by
  sorry

end morning_speed_calculation_l4105_410559


namespace midpoint_trajectory_l4105_410574

/-- The trajectory of the midpoint of a line segment from a point on a hyperbola to its perpendicular projection on a line -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₁ y₁ : ℝ, 
    -- Q(x₁, y₁) is on the hyperbola x^2 - y^2 = 1
    x₁^2 - y₁^2 = 1 ∧ 
    -- N(2x - x₁, 2y - y₁) is on the line x + y = 2
    (2*x - x₁) + (2*y - y₁) = 2 ∧ 
    -- PQ is perpendicular to the line x + y = 2
    (y - y₁) = (x - x₁) ∧ 
    -- P(x, y) is the midpoint of QN
    x = (x₁ + (2*x - x₁)) / 2 ∧ 
    y = (y₁ + (2*y - y₁)) / 2) →
  -- The trajectory equation of P(x, y)
  2*x^2 - 2*y^2 - 2*x + 2*y - 1 = 0 :=
by sorry

end midpoint_trajectory_l4105_410574


namespace mom_bought_39_shirts_l4105_410533

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 13

/-- The number of packages mom bought -/
def packages_bought : ℕ := 3

/-- The total number of t-shirts mom bought -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_bought_39_shirts : total_shirts = 39 := by
  sorry

end mom_bought_39_shirts_l4105_410533


namespace max_boxes_is_240_l4105_410518

/-- Represents the weight of a box in pounds -/
inductive BoxWeight
  | light : BoxWeight  -- 10-pound box
  | heavy : BoxWeight  -- 40-pound box

/-- Calculates the total weight of a pair of boxes (one light, one heavy) -/
def pairWeight : ℕ := 50

/-- Represents the maximum weight capacity of a truck in pounds -/
def truckCapacity : ℕ := 2000

/-- Represents the number of trucks available for delivery -/
def numTrucks : ℕ := 3

/-- Calculates the maximum number of boxes that can be shipped in each delivery -/
def maxBoxesPerDelivery : ℕ := 
  (truckCapacity / pairWeight) * 2 * numTrucks

/-- Theorem stating that the maximum number of boxes that can be shipped in each delivery is 240 -/
theorem max_boxes_is_240 : maxBoxesPerDelivery = 240 := by
  sorry

end max_boxes_is_240_l4105_410518


namespace integer_solution_problem_l4105_410557

theorem integer_solution_problem :
  ∀ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
    a > b ∧ b > c ∧ c > d →
    a * b + c * d = 34 →
    a * c - b * d = 19 →
    ((a = 1 ∧ b = 4 ∧ c = -5 ∧ d = -6) ∨
     (a = -1 ∧ b = -4 ∧ c = 5 ∧ d = 6)) :=
by sorry

end integer_solution_problem_l4105_410557


namespace hyperbola_asymptotes_l4105_410512

/-- Given two hyperbolas with the same asymptotes, prove the value of T -/
theorem hyperbola_asymptotes (T : ℚ) : 
  (∀ x y, y^2 / 49 - x^2 / 25 = 1 → 
    ∃ k, y = k * x ∧ k^2 = 49 / 25) ∧
  (∀ x y, x^2 / T - y^2 / 18 = 1 → 
    ∃ k, y = k * x ∧ k^2 = 18 / T) →
  T = 450 / 49 := by
sorry

end hyperbola_asymptotes_l4105_410512


namespace carolyn_practice_time_l4105_410539

/-- Calculates the total practice time for Carolyn in a month --/
def total_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (practice_days_per_week : ℕ) (weeks_in_month : ℕ) : ℕ :=
  let violin_time := piano_time * violin_multiplier
  let daily_practice_time := piano_time + violin_time
  let weekly_practice_time := daily_practice_time * practice_days_per_week
  weekly_practice_time * weeks_in_month

/-- Proves that Carolyn's total practice time in a month with 4 weeks is 1920 minutes --/
theorem carolyn_practice_time :
  total_practice_time 20 3 6 4 = 1920 := by
  sorry

end carolyn_practice_time_l4105_410539


namespace function_value_determines_parameter_l4105_410513

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3^x + 1 else x^2 + a*x

theorem function_value_determines_parameter (a : ℝ) : f a (f a 0) = 6 → a = 1 := by
  sorry

end function_value_determines_parameter_l4105_410513


namespace cost_of_500_cookies_l4105_410528

/-- The cost in dollars for buying a number of cookies -/
def cookie_cost (num_cookies : ℕ) : ℚ :=
  (num_cookies * 2) / 100

/-- Proof that buying 500 cookies costs 10 dollars -/
theorem cost_of_500_cookies : cookie_cost 500 = 10 := by
  sorry

end cost_of_500_cookies_l4105_410528


namespace stream_rate_calculation_l4105_410545

/-- Proves that the rate of a stream is 5 km/hr given the boat's speed in still water,
    distance traveled downstream, and time taken. -/
theorem stream_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 16 →
  distance = 84 →
  time = 4 →
  ∃ stream_rate : ℝ, 
    stream_rate = 5 ∧
    distance = (boat_speed + stream_rate) * time :=
by
  sorry


end stream_rate_calculation_l4105_410545


namespace xyz_values_l4105_410543

theorem xyz_values (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq1 : x * y = 30)
  (eq2 : x * z = 60)
  (eq3 : x + y + z = 27) :
  x = (27 + Real.sqrt 369) / 2 ∧
  y = 60 / ((27 + Real.sqrt 369) / 2) ∧
  z = 30 / ((27 + Real.sqrt 369) / 2) := by
sorry

end xyz_values_l4105_410543


namespace log_sum_40_25_l4105_410583

theorem log_sum_40_25 : Real.log 40 + Real.log 25 = 3 * Real.log 10 := by
  sorry

end log_sum_40_25_l4105_410583


namespace hyperbola_eccentricity_l4105_410598

/-- Given a hyperbola and a circle, if the length of the chord intercepted on the hyperbola's
    asymptotes by the circle is 2, then the eccentricity of the hyperbola is √6/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x + 5 = 0}
  let asymptotes := {(x, y) : ℝ × ℝ | y = b/a * x ∨ y = -b/a * x}
  let chord_length := 2
  chord_length = Real.sqrt (4 - 9 * b^2 / (a^2 + b^2)) * 2 →
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 6 / 2 :=
by sorry

end hyperbola_eccentricity_l4105_410598


namespace sin_80_cos_20_minus_cos_80_sin_20_l4105_410516

theorem sin_80_cos_20_minus_cos_80_sin_20 : 
  Real.sin (80 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_80_cos_20_minus_cos_80_sin_20_l4105_410516


namespace delta_value_l4105_410505

theorem delta_value (Δ : ℤ) (h : 5 * (-3) = Δ - 3) : Δ = -12 := by
  sorry

end delta_value_l4105_410505


namespace negation_equivalence_l4105_410565

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 > Real.exp x) ↔ (∀ x : ℝ, x^2 ≤ Real.exp x) := by sorry

end negation_equivalence_l4105_410565


namespace boat_speed_in_still_water_boat_speed_in_still_water_proof_l4105_410511

/-- Given a boat that travels 15 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 10 km/hr. -/
theorem boat_speed_in_still_water : ℝ → ℝ → Prop :=
  fun (along_stream : ℝ) (against_stream : ℝ) =>
    along_stream = 15 ∧ against_stream = 5 →
    ∃ (boat_speed stream_speed : ℝ),
      boat_speed + stream_speed = along_stream ∧
      boat_speed - stream_speed = against_stream ∧
      boat_speed = 10

/-- Proof of the theorem -/
theorem boat_speed_in_still_water_proof :
  boat_speed_in_still_water 15 5 := by
  sorry

end boat_speed_in_still_water_boat_speed_in_still_water_proof_l4105_410511


namespace solution_set_of_system_l4105_410552

theorem solution_set_of_system : 
  let S : Set (ℝ × ℝ) := {(x, y) | x - y = 0 ∧ x^2 + y = 2}
  S = {(1, 1), (-2, -2)} := by sorry

end solution_set_of_system_l4105_410552


namespace survey_respondents_l4105_410576

theorem survey_respondents (brand_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  brand_x = 200 →
  ratio_x = 4 →
  ratio_y = 1 →
  ∃ total : ℕ, total = brand_x + (brand_x * ratio_y / ratio_x) ∧ total = 250 :=
by
  sorry

end survey_respondents_l4105_410576


namespace expansion_has_four_nonzero_terms_l4105_410501

def expansion (x : ℝ) : ℝ := (x^2 + 2) * (3*x^3 - x^2 + 4) - 2 * (x^4 - 3*x^3 + x^2)

def count_nonzero_terms (p : ℝ → ℝ) : ℕ := sorry

theorem expansion_has_four_nonzero_terms :
  count_nonzero_terms expansion = 4 := by sorry

end expansion_has_four_nonzero_terms_l4105_410501


namespace equation_solution_l4105_410504

theorem equation_solution : 
  ∀ y : ℝ, (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end equation_solution_l4105_410504


namespace jake_monday_sales_l4105_410531

/-- The number of candy pieces Jake sold on Monday -/
def monday_sales : ℕ := 15

/-- The initial number of candy pieces Jake had -/
def initial_candy : ℕ := 80

/-- The number of candy pieces Jake sold on Tuesday -/
def tuesday_sales : ℕ := 58

/-- The number of candy pieces Jake had left by Wednesday -/
def wednesday_leftover : ℕ := 7

/-- Theorem stating that the number of candy pieces Jake sold on Monday is 15 -/
theorem jake_monday_sales : 
  monday_sales = initial_candy - tuesday_sales - wednesday_leftover := by
  sorry

end jake_monday_sales_l4105_410531


namespace square_sum_geq_neg_double_product_l4105_410536

theorem square_sum_geq_neg_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end square_sum_geq_neg_double_product_l4105_410536


namespace quadratic_solution_l4105_410579

theorem quadratic_solution (a : ℝ) : (1 : ℝ)^2 + 1 + 2*a = 0 → a = -1 := by
  sorry

end quadratic_solution_l4105_410579


namespace max_value_theorem_l4105_410563

theorem max_value_theorem (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) :
  ∃ (max : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 5 → a + 2*b + 3*c ≤ max) ∧
  (∃ (x₀ y₀ z₀ : ℝ), x₀^2 + y₀^2 + z₀^2 = 5 ∧ x₀ + 2*y₀ + 3*z₀ = max) ∧
  max = Real.sqrt 70 :=
sorry

end max_value_theorem_l4105_410563


namespace inequality_proof_l4105_410597

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 / b) + Real.sqrt (b^2 / a) ≥ Real.sqrt a + Real.sqrt b :=
by sorry

end inequality_proof_l4105_410597


namespace sum_expression_equals_1215_l4105_410578

theorem sum_expression_equals_1215 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3*a + 8*b + 24*c + 37*d = 2018) : 
  3*b + 8*c + 24*d + 37*a = 1215 := by
  sorry

end sum_expression_equals_1215_l4105_410578


namespace difference_of_squares_l4105_410538

theorem difference_of_squares : 75^2 - 25^2 = 5000 := by
  sorry

end difference_of_squares_l4105_410538


namespace root_value_l4105_410509

theorem root_value (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 2*x + a = 0 ↔ x = x₁ ∨ x = x₂) → 
  (x₁ + 2*x₂ = 3 - Real.sqrt 2) →
  x₂ = 1 - Real.sqrt 2 := by
sorry

end root_value_l4105_410509


namespace space_shuttle_speed_conversion_l4105_410566

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- Theorem: A space shuttle orbiting at 4 km/s is traveling at 14400 km/h -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 4 = 14400 := by
  sorry

#eval km_per_second_to_km_per_hour 4

end space_shuttle_speed_conversion_l4105_410566


namespace uncle_height_difference_l4105_410560

/-- Given James was initially 2/3 as tall as his uncle who is 72 inches tall,
    and James grew 10 inches, prove that his uncle is now 14 inches taller than James. -/
theorem uncle_height_difference (james_initial_ratio : ℚ) (uncle_height : ℕ) (james_growth : ℕ) :
  james_initial_ratio = 2 / 3 →
  uncle_height = 72 →
  james_growth = 10 →
  uncle_height - (james_initial_ratio * uncle_height + james_growth) = 14 :=
by
  sorry

end uncle_height_difference_l4105_410560


namespace algebraic_expression_value_l4105_410558

theorem algebraic_expression_value (a b : ℝ) 
  (sum_eq : a + b = 5) 
  (product_eq : a * b = 2) : 
  a^2 - a*b + b^2 = 19 := by
sorry

end algebraic_expression_value_l4105_410558


namespace journey_speed_calculation_l4105_410521

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
  (first_part_time : ℝ) (first_part_speed : ℝ) :
  total_distance = 24 →
  total_time = 8 →
  first_part_time = 4 →
  first_part_speed = 4 →
  (total_distance - first_part_time * first_part_speed) / (total_time - first_part_time) = 2 :=
by
  sorry

end journey_speed_calculation_l4105_410521


namespace unique_dataset_l4105_410514

def is_valid_dataset (x : Fin 4 → ℕ) : Prop :=
  (∀ i, x i > 0) ∧
  (x 0 ≤ x 1) ∧ (x 1 ≤ x 2) ∧ (x 2 ≤ x 3) ∧
  (x 0 + x 1 + x 2 + x 3 = 8) ∧
  ((x 1 + x 2) / 2 = 2) ∧
  ((x 0 - 2)^2 + (x 1 - 2)^2 + (x 2 - 2)^2 + (x 3 - 2)^2 = 4)

theorem unique_dataset :
  ∀ x : Fin 4 → ℕ, is_valid_dataset x → (x 0 = 1 ∧ x 1 = 1 ∧ x 2 = 3 ∧ x 3 = 3) :=
sorry

end unique_dataset_l4105_410514


namespace equal_distance_at_time_l4105_410532

/-- The time in minutes past 3 o'clock when the minute hand is at the same distance 
    to the left of 12 as the hour hand is to the right of 12 -/
def time_equal_distance : ℚ := 13 + 11/13

theorem equal_distance_at_time (t : ℚ) : 
  t = time_equal_distance →
  (180 - 6 * t = 90 + 0.5 * t) := by sorry


end equal_distance_at_time_l4105_410532


namespace triangle_altitude_proof_l4105_410588

def triangle_altitude (a b c : ℝ) : Prop :=
  let tan_BCA := 1
  let tan_BAC := 1 / 7
  let perimeter := 24 + 18 * Real.sqrt 2
  let h := 3
  -- The altitude from B to AC has length h
  (tan_BCA = 1 ∧ tan_BAC = 1 / 7 ∧ 
   a + b + c = perimeter) → 
  h = 3

theorem triangle_altitude_proof : 
  ∃ (a b c : ℝ), triangle_altitude a b c :=
sorry

end triangle_altitude_proof_l4105_410588


namespace unique_positive_solution_l4105_410537

theorem unique_positive_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_positive_solution_l4105_410537


namespace system_solution_l4105_410541

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -2) ∧ (9 * x + 5 * y = 9) ∧ (x = 17/47) ∧ (y = 54/47) := by
  sorry

end system_solution_l4105_410541


namespace diamonds_in_tenth_figure_l4105_410550

/-- The number of diamonds in the outer circle of the nth figure -/
def outer_diamonds (n : ℕ) : ℕ := 4 + 6 * (n - 1)

/-- The total number of diamonds in the nth figure -/
def total_diamonds (n : ℕ) : ℕ := 3 * n^2 + n

theorem diamonds_in_tenth_figure : total_diamonds 10 = 310 := by sorry

end diamonds_in_tenth_figure_l4105_410550


namespace likelihood_number_is_probability_l4105_410529

/-- A number representing the likelihood of a random event occurring -/
def likelihood_number : ℝ := sorry

/-- The term for the number representing the likelihood of a random event occurring -/
def probability_term : String := sorry

/-- The theorem stating that the term for the number representing the likelihood of a random event occurring is "probability" -/
theorem likelihood_number_is_probability : probability_term = "probability" := by
  sorry

end likelihood_number_is_probability_l4105_410529


namespace equation_solutions_l4105_410580

-- Define the equation
def equation (x : ℝ) : Prop :=
  (59 - 3*x)^(1/4) + (17 + 3*x)^(1/4) = 4

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 20 ∨ x = -10) :=
by sorry

end equation_solutions_l4105_410580


namespace valid_quadruples_l4105_410570

def is_valid_quadruple (p q r n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ 
  ¬(3 ∣ (p + q)) ∧
  p + q = r * (p - q)^n

theorem valid_quadruples :
  ∀ p q r n : ℕ,
    is_valid_quadruple p q r n →
    ((p = 2 ∧ q = 3 ∧ r = 5 ∧ Even n) ∨
     (p = 3 ∧ q = 2 ∧ r = 5) ∨
     (p = 5 ∧ q = 3 ∧ r = 1 ∧ n = 3) ∨
     (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 2) ∨
     (p = 5 ∧ q = 3 ∧ r = 8 ∧ n = 1) ∨
     (p = 3 ∧ q = 5 ∧ r = 1 ∧ n = 3) ∨
     (p = 3 ∧ q = 5 ∧ r = 2 ∧ n = 2) ∨
     (p = 3 ∧ q = 5 ∧ r = 8 ∧ n = 1)) :=
by sorry


end valid_quadruples_l4105_410570


namespace profit_and_marginal_profit_maxima_l4105_410568

/-- Revenue function -/
def R (x : ℕ) : ℚ := 3000 * x - 20 * x^2

/-- Cost function -/
def C (x : ℕ) : ℚ := 500 * x + 4000

/-- Profit function -/
def p (x : ℕ) : ℚ := R x - C x

/-- Marginal function -/
def M (f : ℕ → ℚ) (x : ℕ) : ℚ := f (x + 1) - f x

/-- Marginal profit function -/
def Mp (x : ℕ) : ℚ := M p x

theorem profit_and_marginal_profit_maxima :
  (∃ x : ℕ, x ≤ 100 ∧ ∀ y : ℕ, y ≤ 100 → p y ≤ p x) ∧
  (∃ x : ℕ, x ≤ 100 ∧ ∀ y : ℕ, y ≤ 100 → Mp y ≤ Mp x) ∧
  (∀ x : ℕ, x ≤ 100 → p x ≤ 74120) ∧
  (∀ x : ℕ, x ≤ 100 → Mp x ≤ 2440) ∧
  (∃ x : ℕ, x ≤ 100 ∧ p x = 74120) ∧
  (∃ x : ℕ, x ≤ 100 ∧ Mp x = 2440) :=
by sorry

end profit_and_marginal_profit_maxima_l4105_410568


namespace sufficient_not_necessary_condition_l4105_410556

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, a * x^2 + 2 * a * x + 1 > 0) → 
  (0 < a ∧ a < 1) → 
  ¬ ((0 < a ∧ a < 1) ↔ (∀ x, a * x^2 + 2 * a * x + 1 > 0)) :=
sorry

end sufficient_not_necessary_condition_l4105_410556


namespace value_of_a_l4105_410591

theorem value_of_a (a b c : ℤ) 
  (sum_ab : a + b = 2)
  (opposite_bc : b + c = 0)
  (abs_c : |c| = 1) :
  a = 3 ∨ a = 1 :=
by sorry

end value_of_a_l4105_410591


namespace f_composition_eq_exp_l4105_410502

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x else 3*x - 1

theorem f_composition_eq_exp (a : ℝ) :
  {a : ℝ | f (f a) = 2^(f a)} = Set.Ici (2/3) := by sorry

end f_composition_eq_exp_l4105_410502


namespace existence_of_critical_point_and_positive_function_l4105_410585

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem existence_of_critical_point_and_positive_function :
  (∃ t : ℝ, t ∈ Set.Ioo (1/2) 1 ∧ (deriv (f 1)) t = 0 ∧
    ∀ t' : ℝ, t' ∈ Set.Ioo (1/2) 1 ∧ (deriv (f 1)) t' = 0 → t' = t) ∧
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) :=
by sorry

end existence_of_critical_point_and_positive_function_l4105_410585


namespace complement_union_theorem_l4105_410599

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 2, 3}

-- Define set B
def B : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l4105_410599


namespace nell_initial_cards_l4105_410519

/-- Nell's initial number of baseball cards -/
def initial_cards : ℕ := sorry

/-- Number of cards Nell gave to Jeff -/
def cards_given : ℕ := 28

/-- Number of cards Nell has left -/
def cards_left : ℕ := 276

/-- Theorem stating that Nell's initial number of cards was 304 -/
theorem nell_initial_cards : initial_cards = 304 := by
  sorry

end nell_initial_cards_l4105_410519


namespace custom_op_5_3_l4105_410548

-- Define the custom operation
def custom_op (m n : ℕ) : ℕ := n ^ 2 - m

-- Theorem statement
theorem custom_op_5_3 : custom_op 5 3 = 4 := by
  sorry

end custom_op_5_3_l4105_410548


namespace min_obtuse_angles_convex_octagon_min_obtuse_angles_convex_octagon_proof_l4105_410510

/-- The minimum number of obtuse interior angles in a convex octagon -/
theorem min_obtuse_angles_convex_octagon : ℕ :=
  let exterior_angles : ℕ := 8
  let sum_exterior_angles : ℕ := 360
  5

/-- Proof of the minimum number of obtuse interior angles in a convex octagon -/
theorem min_obtuse_angles_convex_octagon_proof :
  min_obtuse_angles_convex_octagon = 5 := by
  sorry

end min_obtuse_angles_convex_octagon_min_obtuse_angles_convex_octagon_proof_l4105_410510


namespace polynomial_equality_l4105_410590

theorem polynomial_equality : 
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 10406040101 := by
  sorry

end polynomial_equality_l4105_410590


namespace todd_repayment_l4105_410544

/-- Calculates the amount Todd repaid his brother --/
def amount_repaid (loan : ℝ) (ingredients_cost : ℝ) (snow_cones_sold : ℕ) (price_per_snow_cone : ℝ) (remaining_money : ℝ) : ℝ :=
  (snow_cones_sold : ℝ) * price_per_snow_cone - ingredients_cost + loan - remaining_money

/-- Proves that Todd repaid his brother $110 --/
theorem todd_repayment : 
  amount_repaid 100 75 200 0.75 65 = 110 := by
  sorry

#eval amount_repaid 100 75 200 0.75 65

end todd_repayment_l4105_410544


namespace candy_ratio_l4105_410577

theorem candy_ratio (m_and_m : ℕ) (starburst : ℕ) : 
  (7 : ℕ) * starburst = (4 : ℕ) * m_and_m → m_and_m = 56 → starburst = 32 := by
  sorry

end candy_ratio_l4105_410577


namespace tenth_term_of_sequence_l4105_410564

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem tenth_term_of_sequence : 
  let a : ℚ := 5
  let r : ℚ := 3/4
  geometric_sequence a r 10 = 98415/262144 := by
  sorry

end tenth_term_of_sequence_l4105_410564


namespace madison_distance_l4105_410586

/-- Represents the distance between two locations on a map --/
structure MapDistance where
  inches : ℝ

/-- Represents a travel duration --/
structure TravelTime where
  hours : ℝ

/-- Represents a speed --/
structure Speed where
  mph : ℝ

/-- Represents a map scale --/
structure MapScale where
  inches_per_mile : ℝ

/-- Calculates the actual distance traveled given speed and time --/
def calculate_distance (speed : Speed) (time : TravelTime) : ℝ :=
  speed.mph * time.hours

/-- Calculates the map distance given actual distance and map scale --/
def calculate_map_distance (actual_distance : ℝ) (scale : MapScale) : MapDistance :=
  { inches := actual_distance * scale.inches_per_mile }

/-- The main theorem --/
theorem madison_distance (travel_time : TravelTime) (speed : Speed) (scale : MapScale) :
  travel_time.hours = 3.5 →
  speed.mph = 60 →
  scale.inches_per_mile = 0.023809523809523808 →
  (calculate_map_distance (calculate_distance speed travel_time) scale).inches = 5 := by
  sorry

end madison_distance_l4105_410586


namespace white_l_shapes_count_l4105_410572

/-- Represents a 5x5 grid with white and non-white squares -/
def Grid := Matrix (Fin 5) (Fin 5) Bool

/-- Represents an "L" shape composed of three squares -/
def LShape := List (Fin 5 × Fin 5)

/-- Returns true if all squares in the L-shape are white -/
def isWhite (g : Grid) (l : LShape) : Bool :=
  l.all (fun (i, j) => g i j)

/-- Returns the number of distinct all-white L-shapes in the grid -/
def countWhiteLShapes (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are 24 distinct ways to choose an all-white L-shape -/
theorem white_l_shapes_count (g : Grid) : countWhiteLShapes g = 24 := by
  sorry

end white_l_shapes_count_l4105_410572


namespace trip_duration_l4105_410582

/-- Proves that the trip duration is 24 hours given the specified conditions -/
theorem trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 35 →
  initial_time = 4 →
  additional_speed = 53 →
  average_speed = 50 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 24 := by
  sorry

end trip_duration_l4105_410582


namespace f_extreme_value_and_negative_range_l4105_410554

/-- The function f(x) defined on (0, +∞) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x + (Real.log x - 2) / x + 1

theorem f_extreme_value_and_negative_range :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x) ∧
  f 0 (Real.exp 3) = 1 / Real.exp 3 + 1 ∧
  ∀ (m : ℝ), (∀ (x : ℝ), x > 0 → f m x < 0) ↔ m < -1 / Real.exp 3 :=
by sorry

end f_extreme_value_and_negative_range_l4105_410554


namespace smallest_number_l4105_410508

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -1 := by
  sorry

end smallest_number_l4105_410508


namespace wheat_cost_is_30_l4105_410551

/-- Represents the farm's cultivation scenario -/
structure FarmScenario where
  totalLand : ℕ
  cornCost : ℕ
  totalBudget : ℕ
  wheatAcres : ℕ

/-- Calculates the cost of wheat cultivation per acre -/
def wheatCostPerAcre (scenario : FarmScenario) : ℕ :=
  (scenario.totalBudget - (scenario.cornCost * (scenario.totalLand - scenario.wheatAcres))) / scenario.wheatAcres

/-- Theorem stating the cost of wheat cultivation per acre is 30 -/
theorem wheat_cost_is_30 (scenario : FarmScenario) 
    (h1 : scenario.totalLand = 500)
    (h2 : scenario.cornCost = 42)
    (h3 : scenario.totalBudget = 18600)
    (h4 : scenario.wheatAcres = 200) :
  wheatCostPerAcre scenario = 30 := by
  sorry

#eval wheatCostPerAcre { totalLand := 500, cornCost := 42, totalBudget := 18600, wheatAcres := 200 }

end wheat_cost_is_30_l4105_410551


namespace quadratic_inequality_solution_set_l4105_410571

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 5*x - 14 < 0 ↔ -2 < x ∧ x < 7 := by sorry

end quadratic_inequality_solution_set_l4105_410571


namespace rabbit_apple_collection_l4105_410522

theorem rabbit_apple_collection (rabbit_apples_per_basket deer_apples_per_basket : ℕ)
  (rabbit_baskets deer_baskets total_apples : ℕ) :
  rabbit_apples_per_basket = 5 →
  deer_apples_per_basket = 6 →
  rabbit_baskets = deer_baskets + 3 →
  rabbit_apples_per_basket * rabbit_baskets = total_apples →
  deer_apples_per_basket * deer_baskets = total_apples →
  rabbit_apples_per_basket * rabbit_baskets = 90 :=
by sorry

end rabbit_apple_collection_l4105_410522


namespace triangle_area_theorem_l4105_410523

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the areas of the smaller triangles
def small_triangle_areas (T : Triangle) : ℕ × ℕ × ℕ := (16, 25, 64)

-- Define the theorem
theorem triangle_area_theorem (T : Triangle) : 
  let (a1, a2, a3) := small_triangle_areas T
  (a1 : ℝ) + a2 + a3 > 0 →
  (∃ (l1 l2 l3 : ℝ), l1 > 0 ∧ l2 > 0 ∧ l3 > 0 ∧ 
    l1^2 = a1 ∧ l2^2 = a2 ∧ l3^2 = a3) →
  (∃ (A : ℝ), A = (l1 + l2 + l3)^2 * (a1 + a2 + a3) / (l1^2 + l2^2 + l3^2)) →
  A = 30345 :=
sorry

end triangle_area_theorem_l4105_410523


namespace ivanna_dorothy_ratio_l4105_410573

/-- Represents the scores of the three students -/
structure Scores where
  tatuya : ℚ
  ivanna : ℚ
  dorothy : ℚ

/-- The conditions of the quiz scores -/
def quiz_conditions (s : Scores) : Prop :=
  s.dorothy = 90 ∧
  (s.tatuya + s.ivanna + s.dorothy) / 3 = 84 ∧
  s.tatuya = 2 * s.ivanna ∧
  ∃ x : ℚ, 0 < x ∧ x < 1 ∧ s.ivanna = x * s.dorothy

/-- The theorem stating the ratio of Ivanna's score to Dorothy's score -/
theorem ivanna_dorothy_ratio (s : Scores) (h : quiz_conditions s) :
  s.ivanna / s.dorothy = 3 / 5 := by
  sorry

end ivanna_dorothy_ratio_l4105_410573


namespace divisibility_equivalence_l4105_410507

theorem divisibility_equivalence (n : ℤ) : 
  let A := n % 1000
  let B := n / 1000
  let k := A - B
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) := by
  sorry

end divisibility_equivalence_l4105_410507


namespace proportion_third_number_l4105_410562

theorem proportion_third_number (y : ℝ) : 
  (0.75 : ℝ) / 1.35 = y / 9 → y = 5 := by
  sorry

end proportion_third_number_l4105_410562


namespace factor_polynomial_l4105_410589

theorem factor_polynomial (x : ℝ) : 60 * x^4 - 150 * x^8 = -30 * x^4 * (5 * x^4 - 2) := by
  sorry

end factor_polynomial_l4105_410589


namespace second_player_cannot_win_l4105_410530

-- Define the game of tic-tac-toe
structure TicTacToe :=
  (board : Matrix (Fin 3) (Fin 3) (Option Bool))
  (current_player : Bool)

-- Define optimal play
def optimal_play (game : TicTacToe) : Bool := sorry

-- Define the winning condition
def is_win (game : TicTacToe) (player : Bool) : Prop := sorry

-- Define the draw condition
def is_draw (game : TicTacToe) : Prop := sorry

-- Theorem: If the first player plays optimally, the second player cannot win
theorem second_player_cannot_win (game : TicTacToe) :
  optimal_play game → ¬(is_win game false) :=
by sorry

end second_player_cannot_win_l4105_410530


namespace count_integer_solutions_l4105_410540

theorem count_integer_solutions : ∃! (s : Finset (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ s ↔ p.1 > 0 ∧ p.2 > 0 ∧ 8 / p.1 + 6 / p.2 = 1) ∧ 
  s.card = 5 := by
sorry

end count_integer_solutions_l4105_410540


namespace sugar_in_recipe_l4105_410520

/-- Given a cake recipe with specific flour requirements and a relation between
    sugar and remaining flour, this theorem proves the amount of sugar needed. -/
theorem sugar_in_recipe (total_flour remaining_flour sugar : ℕ) : 
  total_flour = 14 →
  remaining_flour = total_flour - 4 →
  remaining_flour = sugar + 1 →
  sugar = 9 := by
  sorry

end sugar_in_recipe_l4105_410520


namespace complex_power_eight_l4105_410546

theorem complex_power_eight (a b : ℝ) (h : (a : ℂ) + Complex.I = 1 - b * Complex.I) :
  (a + b * Complex.I) ^ 8 = 16 := by sorry

end complex_power_eight_l4105_410546


namespace f_2011_equals_sin_l4105_410553

noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => Real.cos
  | n + 1 => deriv (f n)

theorem f_2011_equals_sin : f 2011 = Real.sin := by sorry

end f_2011_equals_sin_l4105_410553


namespace find_b_l4105_410575

theorem find_b (b c : ℝ) : 
  (∀ x, (3 * x^2 - 4 * x + 5/2) * (2 * x^2 + b * x + c) = 
        6 * x^4 - 11 * x^3 + 13 * x^2 - 15/2 * x + 10/2) → 
  b = -1 := by
sorry

end find_b_l4105_410575


namespace candy_distribution_l4105_410517

theorem candy_distribution (adam james rubert : ℕ) 
  (h1 : rubert = 4 * james) 
  (h2 : james = 3 * adam) 
  (h3 : adam + james + rubert = 96) : 
  adam = 6 := by
sorry

end candy_distribution_l4105_410517


namespace equation_proof_l4105_410595

theorem equation_proof (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b := by
  sorry

end equation_proof_l4105_410595


namespace arithmetic_sequence_problem_l4105_410549

/-- Given an arithmetic sequence a, prove that if a₂ + a₈ = 12, then a₅ = 6 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_sum : a 2 + a 8 = 12) : 
  a 5 = 6 := by
sorry

end arithmetic_sequence_problem_l4105_410549


namespace arctan_sum_equation_l4105_410503

theorem arctan_sum_equation (y : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/y) = π/4 →
  y = -43/3 := by
  sorry

end arctan_sum_equation_l4105_410503


namespace bookstore_purchasing_plans_l4105_410515

theorem bookstore_purchasing_plans :
  let n : ℕ := 3 -- number of books
  let select_at_least_one (k : ℕ) : ℕ := 
    Finset.card (Finset.powerset (Finset.range k) \ {∅})
  select_at_least_one n = 7 := by
  sorry

end bookstore_purchasing_plans_l4105_410515


namespace purely_imaginary_complex_l4105_410561

theorem purely_imaginary_complex (a : ℝ) : 
  (((2 : ℂ) + a * Complex.I) / ((1 : ℂ) - Complex.I) + (1 : ℂ) / ((1 : ℂ) + Complex.I)).re = 0 ↔ a = 3 :=
by sorry

end purely_imaginary_complex_l4105_410561


namespace intersection_point_k_value_l4105_410535

/-- Given two lines -3x + y = k and 2x + y = 10 that intersect at x = -5, prove that k = 35 -/
theorem intersection_point_k_value :
  ∀ (x y k : ℝ),
  (-3 * x + y = k) →
  (2 * x + y = 10) →
  (x = -5) →
  (k = 35) :=
by sorry

end intersection_point_k_value_l4105_410535


namespace factorial_ratio_eq_120_l4105_410526

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio_eq_120 :
  factorial 10 / (factorial 7 * factorial 3) = 120 := by
  sorry

end factorial_ratio_eq_120_l4105_410526


namespace expression_evaluation_l4105_410596

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end expression_evaluation_l4105_410596


namespace compute_expression_l4105_410525

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end compute_expression_l4105_410525


namespace last_digit_of_7_to_1032_l4105_410527

theorem last_digit_of_7_to_1032 : ∃ n : ℕ, 7^1032 ≡ 1 [ZMOD 10] := by
  sorry

end last_digit_of_7_to_1032_l4105_410527


namespace unique_n_less_than_180_l4105_410587

theorem unique_n_less_than_180 : ∃! n : ℕ, n < 180 ∧ n % 8 = 5 := by
  sorry

end unique_n_less_than_180_l4105_410587


namespace sequence_theorem_l4105_410569

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ∈ ({0, 1} : Set ℕ)) ∧
  (∀ n, a n + a (n + 1) ≠ a (n + 2) + a (n + 3)) ∧
  (∀ n, a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_theorem (a : ℕ → ℕ) (h : sequence_property a) (h1 : a 1 = 0) :
  a 2020 = 1 := by
  sorry

end sequence_theorem_l4105_410569


namespace brick_surface_area_l4105_410581

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm² -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
sorry

end brick_surface_area_l4105_410581


namespace composite_10201_composite_10101_l4105_410592

-- Definition for composite numbers
def IsComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Theorem 1: 10201 is composite in any base > 2
theorem composite_10201 (x : ℕ) (h : x > 2) : IsComposite (x^4 + 2*x^2 + 1) := by
  sorry

-- Theorem 2: 10101 is composite in any base ≥ 2
theorem composite_10101 (x : ℕ) (h : x ≥ 2) : IsComposite (x^4 + x^2 + 1) := by
  sorry

end composite_10201_composite_10101_l4105_410592


namespace vasyas_numbers_l4105_410524

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) (h3 : y ≠ 0) :
  x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end vasyas_numbers_l4105_410524


namespace assignment_time_ratio_l4105_410534

theorem assignment_time_ratio : 
  let total_time : ℕ := 120
  let first_part : ℕ := 25
  let third_part : ℕ := 45
  let second_part : ℕ := total_time - (first_part + third_part)
  (second_part : ℚ) / first_part = 2 := by
  sorry

end assignment_time_ratio_l4105_410534


namespace smallest_valid_n_l4105_410584

def is_valid_pairing (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ i ∈ Finset.range 1008, f i ≠ f (2017 - i) ∧ f i ∈ Finset.range 2016 ∧ f (2017 - i) ∈ Finset.range 2016) ∧
    (∀ i ∈ Finset.range 1008, (i + 1) * (2017 - i) ≤ n)

theorem smallest_valid_n : (∀ m < 1017072, ¬ is_valid_pairing m) ∧ is_valid_pairing 1017072 := by
  sorry

end smallest_valid_n_l4105_410584


namespace exists_k_composite_for_all_n_l4105_410593

theorem exists_k_composite_for_all_n : ∃ k : ℕ, ∀ n : ℕ, ∃ m : ℕ, m > 1 ∧ m ∣ (k * 2^n + 1) := by
  sorry

end exists_k_composite_for_all_n_l4105_410593


namespace jake_brought_six_balloons_l4105_410500

/-- The number of balloons Jake brought to the park -/
def jakes_balloons (allans_initial_balloons allans_bought_balloons : ℕ) : ℕ :=
  allans_initial_balloons + allans_bought_balloons + 1

/-- Theorem stating that Jake brought 6 balloons to the park -/
theorem jake_brought_six_balloons :
  jakes_balloons 2 3 = 6 := by
  sorry

end jake_brought_six_balloons_l4105_410500


namespace roses_picked_l4105_410567

theorem roses_picked (tulips flowers_used extra_flowers : ℕ) : 
  tulips = 4 →
  flowers_used = 11 →
  extra_flowers = 4 →
  flowers_used + extra_flowers - tulips = 11 :=
by
  sorry

end roses_picked_l4105_410567


namespace pablo_puzzle_speed_l4105_410542

/-- Represents the number of pieces Pablo can put together per hour -/
def pieces_per_hour : ℕ := sorry

/-- The number of puzzles with 300 pieces -/
def puzzles_300 : ℕ := 8

/-- The number of puzzles with 500 pieces -/
def puzzles_500 : ℕ := 5

/-- The number of pieces in a 300-piece puzzle -/
def pieces_300 : ℕ := 300

/-- The number of pieces in a 500-piece puzzle -/
def pieces_500 : ℕ := 500

/-- The maximum number of hours Pablo works each day -/
def hours_per_day : ℕ := 7

/-- The number of days it takes Pablo to complete all puzzles -/
def days_to_complete : ℕ := 7

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := puzzles_300 * pieces_300 + puzzles_500 * pieces_500

/-- The total number of hours Pablo spends on puzzles -/
def total_hours : ℕ := hours_per_day * days_to_complete

theorem pablo_puzzle_speed : pieces_per_hour = 100 := by
  sorry

end pablo_puzzle_speed_l4105_410542


namespace inequality_theorem_l4105_410594

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ - z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ - z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔
   x₁ * y₁ * x₂ * y₂ - z₁^2 * x₂^2 - z₂^2 * x₁^2 = 0) :=
by sorry


end inequality_theorem_l4105_410594
