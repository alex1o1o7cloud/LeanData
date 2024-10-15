import Mathlib

namespace NUMINAMATH_CALUDE_point_transformation_theorem_l898_89802

def rotate90CounterClockwise (center x : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := x
  (cx - (py - cy), cy + (px - cx))

def reflectAboutYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem point_transformation_theorem (a b : ℝ) :
  let p := (a, b)
  let center := (2, 6)
  let transformed := reflectAboutYEqualsX (rotate90CounterClockwise center p)
  transformed = (-7, 4) → b - a = 15 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l898_89802


namespace NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l898_89836

/-- Calculates the number of bills received when withdrawing money from two banks -/
def number_of_bills (amount_per_bank : ℕ) (num_banks : ℕ) (bill_value : ℕ) : ℕ :=
  (amount_per_bank * num_banks) / bill_value

/-- Proves that withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdraw_300_from_two_banks_in_20_bills : 
  number_of_bills 300 2 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l898_89836


namespace NUMINAMATH_CALUDE_fraction_product_l898_89865

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5 : ℚ) * (3 / 6 : ℚ) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l898_89865


namespace NUMINAMATH_CALUDE_original_number_proof_l898_89888

theorem original_number_proof : 
  ∀ x : ℝ, ((x / 8) * 16 + 20) / 4 = 34 → x = 58 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l898_89888


namespace NUMINAMATH_CALUDE_worker_savings_fraction_l898_89864

theorem worker_savings_fraction (P : ℝ) (S : ℝ) (h1 : P > 0) (h2 : 0 ≤ S ∧ S ≤ 1) 
  (h3 : 12 * S * P = 2 * (1 - S) * P) : S = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_fraction_l898_89864


namespace NUMINAMATH_CALUDE_group_size_after_new_member_l898_89818

theorem group_size_after_new_member (n : ℕ) : 
  (n * 14 + 32) / (n + 1) = 15 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_group_size_after_new_member_l898_89818


namespace NUMINAMATH_CALUDE_square_difference_pattern_l898_89831

theorem square_difference_pattern (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l898_89831


namespace NUMINAMATH_CALUDE_binomial_coefficient_60_2_l898_89839

theorem binomial_coefficient_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_60_2_l898_89839


namespace NUMINAMATH_CALUDE_roots_difference_squared_l898_89837

theorem roots_difference_squared (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 4 * x₁ - 3 = 0) →
  (2 * x₂^2 + 4 * x₂ - 3 = 0) →
  (x₁ - x₂)^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l898_89837


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l898_89896

/-- Represents the speed and distance calculations for a boat in a stream -/
def boat_in_stream (boat_speed : ℝ) (downstream_distance : ℝ) : ℝ :=
  let stream_speed := downstream_distance - boat_speed
  boat_speed - stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : boat_speed = 11) 
  (h2 : downstream_distance = 16) : 
  boat_in_stream boat_speed downstream_distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l898_89896


namespace NUMINAMATH_CALUDE_water_height_in_cylinder_l898_89858

/-- Given a cone with base radius 10 cm and height 15 cm, when its volume of water is poured into a cylinder with base radius 20 cm, the height of water in the cylinder is 1.25 cm. -/
theorem water_height_in_cylinder (π : ℝ) : 
  let cone_radius : ℝ := 10
  let cone_height : ℝ := 15
  let cylinder_radius : ℝ := 20
  let cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
  let cylinder_height : ℝ := cone_volume / (π * cylinder_radius^2)
  cylinder_height = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_water_height_in_cylinder_l898_89858


namespace NUMINAMATH_CALUDE_lowest_price_pet_food_l898_89885

/-- Calculates the final price of a pet food container after two consecutive discounts -/
def final_price (msrp : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  msrp * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the lowest possible price of a $35 pet food container
    after a maximum 30% discount and an additional 20% discount is $19.60 -/
theorem lowest_price_pet_food :
  final_price 35 0.3 0.2 = 19.60 := by
  sorry

end NUMINAMATH_CALUDE_lowest_price_pet_food_l898_89885


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l898_89859

theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) :
  total_investment = 15000 →
  first_investment = 5000 →
  second_investment = 6000 →
  first_rate = 0.03 →
  second_rate = 0.045 →
  desired_income = 800 →
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment = 0.095 := by sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l898_89859


namespace NUMINAMATH_CALUDE_square_area_ratio_l898_89834

/-- Given a large square and a small square with coinciding centers,
    if the area of the cross formed by the small square is 17 times
    the area of the small square, then the area of the large square
    is 81 times the area of the small square. -/
theorem square_area_ratio (small_side large_side : ℝ) : 
  small_side > 0 →
  large_side > 0 →
  2 * large_side * small_side - small_side^2 = 17 * small_side^2 →
  large_side^2 = 81 * small_side^2 := by
  sorry

#check square_area_ratio

end NUMINAMATH_CALUDE_square_area_ratio_l898_89834


namespace NUMINAMATH_CALUDE_y_derivative_l898_89814

noncomputable section

open Real

def y (x : ℝ) : ℝ := (1/6) * log ((1 - sinh (2*x)) / (2 + sinh (2*x)))

theorem y_derivative (x : ℝ) : 
  deriv y x = cosh (2*x) / (sinh (2*x)^2 + sinh (2*x) - 2) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l898_89814


namespace NUMINAMATH_CALUDE_weight_change_result_l898_89895

/-- Calculates the final weight after a series of weight changes -/
def finalWeight (initialWeight : ℕ) (initialLoss : ℕ) : ℕ :=
  let weightAfterFirstLoss := initialWeight - initialLoss
  let weightAfterSecondGain := weightAfterFirstLoss + 2 * initialLoss
  let weightAfterThirdLoss := weightAfterSecondGain - 3 * initialLoss
  let finalWeightGain := 3  -- half of a dozen
  weightAfterThirdLoss + finalWeightGain

/-- Theorem stating that the final weight is 78 pounds -/
theorem weight_change_result : finalWeight 99 12 = 78 := by
  sorry

end NUMINAMATH_CALUDE_weight_change_result_l898_89895


namespace NUMINAMATH_CALUDE_unique_rectangle_dimensions_l898_89861

theorem unique_rectangle_dimensions : 
  ∃! (a b : ℕ), 
    b > a ∧ 
    a > 0 ∧ 
    b > 0 ∧
    (a - 4) * (b - 4) = 2 * (a * b) / 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_dimensions_l898_89861


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l898_89892

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4)^2 → area = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l898_89892


namespace NUMINAMATH_CALUDE_population_increase_l898_89812

theorem population_increase (birth_rate : ℚ) (death_rate : ℚ) (seconds_per_day : ℕ) :
  birth_rate = 7 / 2 →
  death_rate = 3 / 2 →
  seconds_per_day = 24 * 3600 →
  (birth_rate - death_rate) * seconds_per_day = 172800 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l898_89812


namespace NUMINAMATH_CALUDE_a_minus_b_values_l898_89897

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  (a - b = 4) ∨ (a - b = 8) :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l898_89897


namespace NUMINAMATH_CALUDE_points_A_B_D_collinear_l898_89870

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_AB (a b : V) : V := a + 2 • b
def vector_BC (a b : V) : V := -5 • a + 6 • b
def vector_CD (a b : V) : V := 7 • a - 2 • b

theorem points_A_B_D_collinear (a b : V) :
  ∃ (k : ℝ), vector_AB a b = k • (vector_AB a b + vector_BC a b + vector_CD a b) :=
sorry

end NUMINAMATH_CALUDE_points_A_B_D_collinear_l898_89870


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l898_89854

/-- Two circles in the xy-plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0

/-- The property that the two circles have exactly three common tangents -/
def has_three_common_tangents (c : TwoCircles) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2 * c.a * x + c.a^2 - 4 = 0 ∧
                x^2 + y^2 - 4 * c.b * y - 1 + 4 * c.b^2 = 0

/-- The theorem stating the minimum value of 1/a^2 + 1/b^2 -/
theorem min_value_sum_reciprocal_squares (c : TwoCircles) 
  (h : has_three_common_tangents c) : 
  (∀ ε > 0, ∃ (c' : TwoCircles), has_three_common_tangents c' ∧ 
    1 / c'.a^2 + 1 / c'.b^2 < 1 + ε) ∧
  (∀ (c' : TwoCircles), has_three_common_tangents c' → 
    1 / c'.a^2 + 1 / c'.b^2 ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l898_89854


namespace NUMINAMATH_CALUDE_mrs_hilt_shortage_l898_89835

def initial_amount : ℚ := 375 / 100
def pencil_cost : ℚ := 115 / 100
def eraser_cost : ℚ := 85 / 100
def notebook_cost : ℚ := 225 / 100

theorem mrs_hilt_shortage :
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -50 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_shortage_l898_89835


namespace NUMINAMATH_CALUDE_snooker_tournament_revenue_l898_89813

theorem snooker_tournament_revenue
  (vip_price : ℚ)
  (general_price : ℚ)
  (total_tickets : ℕ)
  (ticket_difference : ℕ)
  (h1 : vip_price = 45)
  (h2 : general_price = 20)
  (h3 : total_tickets = 320)
  (h4 : ticket_difference = 276)
  : ∃ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = total_tickets ∧
    vip_tickets = general_tickets - ticket_difference ∧
    vip_price * vip_tickets + general_price * general_tickets = 6950 := by
  sorry

#check snooker_tournament_revenue

end NUMINAMATH_CALUDE_snooker_tournament_revenue_l898_89813


namespace NUMINAMATH_CALUDE_percent_commutation_l898_89878

theorem percent_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 45) : 0.4 * (0.3 * x) = 45 := by
  sorry

end NUMINAMATH_CALUDE_percent_commutation_l898_89878


namespace NUMINAMATH_CALUDE_max_basketballs_l898_89867

-- Define the cost of soccer balls and basketballs
def cost_3_soccer_2_basket : ℕ := 490
def cost_2_soccer_4_basket : ℕ := 660
def total_balls : ℕ := 62
def max_total_cost : ℕ := 6750

-- Define the function to calculate the total cost
def total_cost (soccer_balls : ℕ) (basketballs : ℕ) : ℕ :=
  let soccer_cost := (cost_3_soccer_2_basket * 2 - cost_2_soccer_4_basket * 3) / 2
  let basket_cost := (cost_2_soccer_4_basket * 3 - cost_3_soccer_2_basket * 2) / 2
  soccer_cost * soccer_balls + basket_cost * basketballs

-- Theorem to prove
theorem max_basketballs :
  ∃ (m : ℕ), m = 39 ∧
  (∀ (n : ℕ), n > m → total_cost (total_balls - n) n > max_total_cost) ∧
  total_cost (total_balls - m) m ≤ max_total_cost :=
sorry

end NUMINAMATH_CALUDE_max_basketballs_l898_89867


namespace NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l898_89811

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the set of solutions
def solution_set : Set ℝ := {x | x < -1 ∨ x > 3}

-- State the theorem
theorem log_inequality_equiv_solution_set :
  ∀ x : ℝ, lg (x^2 - 2*x - 3) ≥ 0 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l898_89811


namespace NUMINAMATH_CALUDE_dog_food_insufficient_l898_89891

/-- Proves that the amount of dog food remaining after two weeks is negative -/
theorem dog_food_insufficient (num_dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) 
  (initial_food : ℚ) (days : ℕ) :
  num_dogs = 5 →
  food_per_meal = 3/4 →
  meals_per_day = 3 →
  initial_food = 45 →
  days = 14 →
  initial_food - (num_dogs * food_per_meal * meals_per_day * days) < 0 :=
by sorry

end NUMINAMATH_CALUDE_dog_food_insufficient_l898_89891


namespace NUMINAMATH_CALUDE_h_j_composition_l898_89898

theorem h_j_composition (c d : ℝ) (h : ℝ → ℝ) (j : ℝ → ℝ)
  (h_def : ∀ x, h x = c * x + d)
  (j_def : ∀ x, j x = 3 * x - 4)
  (composition : ∀ x, j (h x) = 4 * x + 3) :
  c + d = 11 / 3 := by
sorry

end NUMINAMATH_CALUDE_h_j_composition_l898_89898


namespace NUMINAMATH_CALUDE_sin_equality_n_512_l898_89851

theorem sin_equality_n_512 (n : ℤ) :
  -100 ≤ n ∧ n ≤ 100 ∧ Real.sin (n * π / 180) = Real.sin (512 * π / 180) → n = 28 :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_n_512_l898_89851


namespace NUMINAMATH_CALUDE_train_journey_times_l898_89866

/-- Proves that given the conditions of two trains running late, their usual journey times are both 2 hours -/
theorem train_journey_times (speed_ratio_A speed_ratio_B : ℚ) (delay_A delay_B : ℚ) 
  (h1 : speed_ratio_A = 4/5)
  (h2 : speed_ratio_B = 3/4)
  (h3 : delay_A = 1/2)  -- 30 minutes in hours
  (h4 : delay_B = 2/3)  -- 40 minutes in hours
  : ∃ (T_A T_B : ℚ), T_A = 2 ∧ T_B = 2 ∧ 
    (1/speed_ratio_A) * T_A = T_A + delay_A ∧
    (1/speed_ratio_B) * T_B = T_B + delay_B :=
by sorry


end NUMINAMATH_CALUDE_train_journey_times_l898_89866


namespace NUMINAMATH_CALUDE_median_salary_is_worker_salary_l898_89822

/-- Represents a position in the company -/
inductive Position
  | CEO
  | GeneralManager
  | Manager
  | Supervisor
  | Worker

/-- Information about a position: number of employees and salary -/
structure PositionInfo where
  count : Nat
  salary : Nat

/-- Company salary data -/
def companySalaries : List (Position × PositionInfo) :=
  [(Position.CEO, ⟨1, 150000⟩),
   (Position.GeneralManager, ⟨3, 100000⟩),
   (Position.Manager, ⟨12, 80000⟩),
   (Position.Supervisor, ⟨8, 55000⟩),
   (Position.Worker, ⟨35, 30000⟩)]

/-- Total number of employees -/
def totalEmployees : Nat :=
  companySalaries.foldr (fun (_, info) acc => acc + info.count) 0

/-- Theorem: The median salary of the company is $30,000 -/
theorem median_salary_is_worker_salary :
  let salaries := companySalaries.map (fun (_, info) => info.salary)
  let counts := companySalaries.map (fun (_, info) => info.count)
  let medianIndex := (totalEmployees + 1) / 2
  ∃ (i : Nat), i < salaries.length ∧
    (counts.take i).sum < medianIndex ∧
    medianIndex ≤ (counts.take (i + 1)).sum ∧
    salaries[i]! = 30000 := by
  sorry

#eval totalEmployees -- Should output 59

end NUMINAMATH_CALUDE_median_salary_is_worker_salary_l898_89822


namespace NUMINAMATH_CALUDE_square_difference_540_460_l898_89801

theorem square_difference_540_460 : 540^2 - 460^2 = 80000 := by sorry

end NUMINAMATH_CALUDE_square_difference_540_460_l898_89801


namespace NUMINAMATH_CALUDE_perfect_square_condition_l898_89840

theorem perfect_square_condition (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 18*x + k = (a*x + b)^2) ↔ k = 81 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l898_89840


namespace NUMINAMATH_CALUDE_acute_triangle_tangent_difference_range_l898_89852

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² - a² = ac, then 1 < 1/tan(A) - 1/tan(B) < 2√3/3 -/
theorem acute_triangle_tangent_difference_range 
  (A B C : ℝ) (a b c : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sides : b^2 - a^2 = a*c) :
  1 < 1 / Real.tan A - 1 / Real.tan B ∧ 
  1 / Real.tan A - 1 / Real.tan B < 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_tangent_difference_range_l898_89852


namespace NUMINAMATH_CALUDE_gcd_20020_11011_l898_89800

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_20020_11011_l898_89800


namespace NUMINAMATH_CALUDE_taxi_fare_distance_l898_89819

/-- Represents the taxi fare structure and proves the distance for each fare segment -/
theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (total_distance : ℝ) (total_fare : ℝ) :
  initial_fare = 8 →
  subsequent_fare = 0.8 →
  total_distance = 8 →
  total_fare = 39.2 →
  ∃ (d : ℝ), d > 0 ∧ d = 1/5 ∧
    total_fare = initial_fare + subsequent_fare * ((total_distance - d) / d) :=
by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_distance_l898_89819


namespace NUMINAMATH_CALUDE_unique_solution_for_euler_equation_l898_89876

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- The statement to prove -/
theorem unique_solution_for_euler_equation :
  ∀ a n : ℕ, a ≠ 0 ∧ n ≠ 0 → (φ (a^n + n) = 2^n) → (a = 2 ∧ n = 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_euler_equation_l898_89876


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l898_89820

/-- A geometric sequence with positive terms -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l898_89820


namespace NUMINAMATH_CALUDE_tenth_triangular_number_l898_89846

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tenth_triangular_number : triangular_number 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_tenth_triangular_number_l898_89846


namespace NUMINAMATH_CALUDE_sock_counting_l898_89848

theorem sock_counting (initial : ℕ) (thrown_away : ℕ) (new_bought : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + new_bought = initial + new_bought - thrown_away :=
by sorry

end NUMINAMATH_CALUDE_sock_counting_l898_89848


namespace NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_problem_l898_89847

/-- Calculates the interest rate of an additional investment given initial investment parameters and desired total return rate. -/
theorem additional_investment_rate 
  (initial_investment : ℝ) 
  (initial_rate : ℝ) 
  (additional_investment : ℝ) 
  (total_rate : ℝ) : ℝ :=
  let total_investment := initial_investment + additional_investment
  let initial_income := initial_investment * initial_rate
  let total_desired_income := total_investment * total_rate
  let additional_income_needed := total_desired_income - initial_income
  additional_income_needed / additional_investment

/-- Proves that the additional investment rate is 0.08 (8%) given the specific problem parameters. -/
theorem barbata_investment_problem : 
  additional_investment_rate 1400 0.05 700 0.06 = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_problem_l898_89847


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l898_89877

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l898_89877


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_three_l898_89833

theorem opposite_of_sqrt_three : 
  -(Real.sqrt 3) = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_three_l898_89833


namespace NUMINAMATH_CALUDE_parallel_no_common_points_relation_l898_89806

-- Define the concept of lines in a space
axiom Line : Type

-- Define the parallel relation between lines
axiom parallel : Line → Line → Prop

-- Define the property of having no common points
axiom no_common_points : Line → Line → Prop

-- Define the theorem
theorem parallel_no_common_points_relation (a b : Line) :
  (parallel a b → no_common_points a b) ∧
  ¬(no_common_points a b → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_no_common_points_relation_l898_89806


namespace NUMINAMATH_CALUDE_marble_problem_l898_89860

theorem marble_problem (a b : ℚ) : 
  let brian := 3 * a
  let caden := 4 * brian
  let daryl := 6 * caden
  b = 6 →
  a + (brian - b) + caden + daryl = 240 →
  a = 123 / 44 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l898_89860


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l898_89886

theorem max_value_of_trig_function :
  ∀ x : ℝ, (π / (1 + Real.tan x ^ 2)) ≤ π :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l898_89886


namespace NUMINAMATH_CALUDE_general_inequality_l898_89803

theorem general_inequality (x : ℝ) (n : ℕ) (h : x > 0) (hn : n > 0) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_general_inequality_l898_89803


namespace NUMINAMATH_CALUDE_farm_problem_l898_89810

/-- Proves that given the conditions of the farm problem, the number of hens is 24 -/
theorem farm_problem (hens cows : ℕ) : 
  hens + cows = 48 →
  2 * hens + 4 * cows = 144 →
  hens = 24 := by
sorry

end NUMINAMATH_CALUDE_farm_problem_l898_89810


namespace NUMINAMATH_CALUDE_x_value_l898_89842

theorem x_value : ∃ x : ℝ, (0.65 * x = 0.20 * 617.50) ∧ (x = 190) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l898_89842


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l898_89824

theorem quadratic_root_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x - 6 = 0 ∧ x = -3) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l898_89824


namespace NUMINAMATH_CALUDE_max_value_theorem_l898_89889

theorem max_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_constraint : x^2 - x*y + 2*y^2 = 8) :
  x^2 + x*y + 2*y^2 ≤ (72 + 32*Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l898_89889


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_integer_sqrt_sum_squares_not_integer_1_sqrt_sum_squares_not_integer_2_sqrt_sum_squares_not_integer_3_l898_89843

theorem sqrt_sum_squares_integer (x y : ℤ) : x = 25530 ∧ y = 29464 →
  ∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_1 (x y : ℤ) : x = 37615 ∧ y = 26855 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_2 (x y : ℤ) : x = 15123 ∧ y = 32477 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_3 (x y : ℤ) : x = 28326 ∧ y = 28614 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_integer_sqrt_sum_squares_not_integer_1_sqrt_sum_squares_not_integer_2_sqrt_sum_squares_not_integer_3_l898_89843


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l898_89881

/-- The distance between two complex numbers 2+3i and -2+2i is √17 -/
theorem distance_between_complex_points : 
  Complex.abs ((2 : ℂ) + 3*I - ((-2 : ℂ) + 2*I)) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l898_89881


namespace NUMINAMATH_CALUDE_perimeter_of_PQRSU_l898_89844

-- Define the points as 2D vectors
def P : ℝ × ℝ := (0, 8)
def Q : ℝ × ℝ := (4, 8)
def R : ℝ × ℝ := (4, 4)
def S : ℝ × ℝ := (9, 0)
def U : ℝ × ℝ := (0, 0)

-- Define the conditions
def PQ_length : ℝ := 4
def PU_length : ℝ := 8
def US_length : ℝ := 9

-- Define the right angles
def angle_PUQ_is_right : (P.1 - U.1) * (Q.1 - U.1) + (P.2 - U.2) * (Q.2 - U.2) = 0 := by sorry
def angle_UPQ_is_right : (U.1 - P.1) * (Q.1 - P.1) + (U.2 - P.2) * (Q.2 - P.2) = 0 := by sorry
def angle_PQR_is_right : (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 := by sorry

-- Define the theorem
theorem perimeter_of_PQRSU : 
  let perimeter := PQ_length + 
                   Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) + 
                   Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) + 
                   US_length + 
                   PU_length
  perimeter = 25 + Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_PQRSU_l898_89844


namespace NUMINAMATH_CALUDE_arg_cube_equals_pi_l898_89855

theorem arg_cube_equals_pi (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3)
  (h₂ : Complex.abs z₂ = 5)
  (h₃ : Complex.abs (z₁ + z₂) = 7) :
  (Complex.arg (z₁ - z₂))^3 = π := by
  sorry

end NUMINAMATH_CALUDE_arg_cube_equals_pi_l898_89855


namespace NUMINAMATH_CALUDE_sphere_cross_section_distance_l898_89825

theorem sphere_cross_section_distance (V : ℝ) (A : ℝ) (r : ℝ) (r_cross : ℝ) (d : ℝ) :
  V = 4 * Real.sqrt 3 * Real.pi →
  (4 / 3) * Real.pi * r^3 = V →
  A = Real.pi →
  Real.pi * r_cross^2 = A →
  d^2 + r_cross^2 = r^2 →
  d = Real.sqrt 2 := by
  sorry

#check sphere_cross_section_distance

end NUMINAMATH_CALUDE_sphere_cross_section_distance_l898_89825


namespace NUMINAMATH_CALUDE_time_addition_sum_l898_89883

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the resulting time -/
def addTime (start : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- Converts a 24-hour time to 12-hour format -/
def to12Hour (t : Time) : Time :=
  sorry

theorem time_addition_sum (startTime : Time) :
  let endTime := to12Hour (addTime startTime 145 50 15)
  endTime.hours + endTime.minutes + endTime.seconds = 69 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_sum_l898_89883


namespace NUMINAMATH_CALUDE_patricia_hair_length_l898_89890

/-- Given Patricia's hair growth scenario, prove the desired hair length after donation -/
theorem patricia_hair_length 
  (current_length : ℕ) 
  (donation_length : ℕ) 
  (growth_needed : ℕ) 
  (h1 : current_length = 14)
  (h2 : donation_length = 23)
  (h3 : growth_needed = 21) : 
  current_length + growth_needed - donation_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_patricia_hair_length_l898_89890


namespace NUMINAMATH_CALUDE_intersection_value_l898_89853

-- Define the complex plane
variable (z : ℂ)

-- Define the first equation |z - 3| = 3|z + 3|
def equation1 (z : ℂ) : Prop := Complex.abs (z - 3) = 3 * Complex.abs (z + 3)

-- Define the second equation |z| = k
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the condition of intersection at exactly one point
def single_intersection (k : ℝ) : Prop :=
  ∃! z, equation1 z ∧ equation2 z k

-- The theorem to prove
theorem intersection_value :
  ∃! k, k > 0 ∧ single_intersection k ∧ k = 4.5 :=
sorry

end NUMINAMATH_CALUDE_intersection_value_l898_89853


namespace NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l898_89807

/-- The diameter of a circular field given the cost of fencing. -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- The diameter of the specific circular field is approximately 34 meters. -/
theorem specific_field_diameter : 
  abs (circular_field_diameter 2 213.63 - 34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l898_89807


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l898_89816

/-- Given three consecutive odd numbers where the first is 7, 
    prove that the multiple of the third number that satisfies 
    the equation is 3 --/
theorem consecutive_odd_numbers_equation (n : ℕ) : 
  let first := 7
  let second := first + 2
  let third := second + 2
  8 * first = n * third + 5 + 2 * second → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l898_89816


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l898_89875

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 9x + 4 is 1 -/
theorem quadratic_discriminant : discriminant 5 (-9) 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l898_89875


namespace NUMINAMATH_CALUDE_proposition_p_negation_and_range_l898_89862

theorem proposition_p_negation_and_range (a : ℝ) :
  (¬∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ∧ 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0 → 0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_negation_and_range_l898_89862


namespace NUMINAMATH_CALUDE_rope_remaining_l898_89817

theorem rope_remaining (initial_length : ℝ) (fraction_to_allan : ℝ) (fraction_to_jack : ℝ) :
  initial_length = 20 ∧ 
  fraction_to_allan = 1/4 ∧ 
  fraction_to_jack = 2/3 →
  initial_length * (1 - fraction_to_allan) * (1 - fraction_to_jack) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rope_remaining_l898_89817


namespace NUMINAMATH_CALUDE_pauls_money_duration_l898_89856

/-- 
Given Paul's earnings from mowing lawns and weed eating, and his weekly spending rate,
prove that the money will last for 2 weeks.
-/
theorem pauls_money_duration (lawn_earnings weed_earnings weekly_spending : ℕ) 
  (h1 : lawn_earnings = 3)
  (h2 : weed_earnings = 3)
  (h3 : weekly_spending = 3) :
  (lawn_earnings + weed_earnings) / weekly_spending = 2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l898_89856


namespace NUMINAMATH_CALUDE_base_k_conversion_l898_89809

/-- Given that 44 in base k equals 36 in base 10, prove that 67 in base 10 equals 103 in base k. -/
theorem base_k_conversion (k : ℕ) (h : 4 * k + 4 = 36) : 
  (67 : ℕ).digits k = [3, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_k_conversion_l898_89809


namespace NUMINAMATH_CALUDE_additional_sheep_problem_l898_89828

theorem additional_sheep_problem (mary_sheep : ℕ) (bob_additional : ℕ) :
  mary_sheep = 300 →
  (mary_sheep + 266 = 2 * mary_sheep + bob_additional - 69) →
  bob_additional = 35 := by
  sorry

end NUMINAMATH_CALUDE_additional_sheep_problem_l898_89828


namespace NUMINAMATH_CALUDE_conic_parabola_focus_coincidence_l898_89899

/-- Given a conic section and a parabola, prove that the parameter m of the conic section is 9 when their foci coincide. -/
theorem conic_parabola_focus_coincidence (m : ℝ) : 
  m ≠ 0 → m ≠ 5 → 
  (∃ (x y : ℝ), x^2 / m + y^2 / 5 = 1) →
  (∃ (x y : ℝ), y^2 = 8*x) →
  (∃ (x₀ y₀ : ℝ), x₀^2 / m + y₀^2 / 5 = 1 ∧ y₀^2 = 8*x₀ ∧ x₀ = 2 ∧ y₀ = 0) →
  m = 9 :=
by sorry

end NUMINAMATH_CALUDE_conic_parabola_focus_coincidence_l898_89899


namespace NUMINAMATH_CALUDE_circumcircle_radius_l898_89832

/-- Given a triangle ABC with side length a = 2 and sin A = 1/3, 
    the radius R of its circumcircle is 3. -/
theorem circumcircle_radius (A B C : ℝ × ℝ) (a : ℝ) (sin_A : ℝ) :
  a = 2 →
  sin_A = 1/3 →
  let R := (a / 2) / sin_A
  R = 3 := by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_l898_89832


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l898_89804

theorem arithmetic_evaluation : 2 * (5 - 2) - 5^2 = -19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l898_89804


namespace NUMINAMATH_CALUDE_mike_work_hours_l898_89857

def wash_time : ℕ := 10
def oil_change_time : ℕ := 15
def tire_change_time : ℕ := 30
def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def tire_sets_changed : ℕ := 2

theorem mike_work_hours : 
  (cars_washed * wash_time + cars_oil_changed * oil_change_time + tire_sets_changed * tire_change_time) / 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_work_hours_l898_89857


namespace NUMINAMATH_CALUDE_toms_profit_l898_89884

/-- Calculate Tom's total profit from lawn mowing and side jobs --/
theorem toms_profit (small_lawns : ℕ) (small_price : ℕ)
                    (medium_lawns : ℕ) (medium_price : ℕ)
                    (large_lawns : ℕ) (large_price : ℕ)
                    (gas_expense : ℕ) (maintenance_expense : ℕ)
                    (weed_jobs : ℕ) (weed_price : ℕ)
                    (hedge_jobs : ℕ) (hedge_price : ℕ)
                    (rake_jobs : ℕ) (rake_price : ℕ) :
  small_lawns = 4 →
  small_price = 12 →
  medium_lawns = 3 →
  medium_price = 15 →
  large_lawns = 1 →
  large_price = 20 →
  gas_expense = 17 →
  maintenance_expense = 5 →
  weed_jobs = 2 →
  weed_price = 10 →
  hedge_jobs = 3 →
  hedge_price = 8 →
  rake_jobs = 1 →
  rake_price = 12 →
  (small_lawns * small_price + medium_lawns * medium_price + large_lawns * large_price +
   weed_jobs * weed_price + hedge_jobs * hedge_price + rake_jobs * rake_price) -
  (gas_expense + maintenance_expense) = 147 :=
by sorry

end NUMINAMATH_CALUDE_toms_profit_l898_89884


namespace NUMINAMATH_CALUDE_problem_solution_l898_89823

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) ∧ 
   (∀ m' : ℝ, m' < m → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a * b > m') ∧
   m = 1/4) ∧
  (∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -2 ≤ x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l898_89823


namespace NUMINAMATH_CALUDE_debt_payment_difference_l898_89868

/-- Given a debt paid in 40 installments with specific conditions, 
    prove the difference between later and earlier payments. -/
theorem debt_payment_difference (first_payment : ℝ) (average_payment : ℝ) 
    (h1 : first_payment = 410)
    (h2 : average_payment = 442.5) : 
    ∃ (difference : ℝ), 
      20 * first_payment + 20 * (first_payment + difference) = 40 * average_payment ∧ 
      difference = 65 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_difference_l898_89868


namespace NUMINAMATH_CALUDE_negative_difference_l898_89872

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l898_89872


namespace NUMINAMATH_CALUDE_correct_categorization_l898_89882

def numbers : List ℚ := [15, -3/8, 0, 0.15, -30, -12.8, 22/5, 20]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n
def is_fraction (q : ℚ) : Prop := ¬(is_integer q)
def is_positive_integer (q : ℚ) : Prop := is_integer q ∧ q > 0
def is_negative_fraction (q : ℚ) : Prop := is_fraction q ∧ q < 0
def is_non_negative (q : ℚ) : Prop := q ≥ 0

def integer_set : Set ℚ := {q ∈ numbers | is_integer q}
def fraction_set : Set ℚ := {q ∈ numbers | is_fraction q}
def positive_integer_set : Set ℚ := {q ∈ numbers | is_positive_integer q}
def negative_fraction_set : Set ℚ := {q ∈ numbers | is_negative_fraction q}
def non_negative_set : Set ℚ := {q ∈ numbers | is_non_negative q}

theorem correct_categorization :
  integer_set = {15, 0, -30, 20} ∧
  fraction_set = {-3/8, 0.15, -12.8, 22/5} ∧
  positive_integer_set = {15, 20} ∧
  negative_fraction_set = {-3/8, -12.8} ∧
  non_negative_set = {15, 0, 0.15, 22/5, 20} := by
  sorry

end NUMINAMATH_CALUDE_correct_categorization_l898_89882


namespace NUMINAMATH_CALUDE_resort_tips_multiple_l898_89869

theorem resort_tips_multiple (total_months : Nat) (special_month_fraction : Real) 
  (h1 : total_months = 7)
  (h2 : special_month_fraction = 0.5)
  (average_other_months : Real)
  (special_month_tips : Real)
  (h3 : special_month_tips = special_month_fraction * (average_other_months * (total_months - 1) + special_month_tips))
  (h4 : ∃ (m : Real), special_month_tips = m * average_other_months) :
  ∃ (m : Real), special_month_tips = 6 * average_other_months :=
by sorry

end NUMINAMATH_CALUDE_resort_tips_multiple_l898_89869


namespace NUMINAMATH_CALUDE_square_root_of_four_l898_89894

theorem square_root_of_four : 
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l898_89894


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l898_89849

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l898_89849


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l898_89829

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The x-axis symmetry operation -/
def xAxisSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetric_point_wrt_x_axis :
  let original := Point3D.mk (-2) 1 9
  xAxisSymmetry original = Point3D.mk (-2) (-1) (-9) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l898_89829


namespace NUMINAMATH_CALUDE_coefficient_x4_is_negative_seven_l898_89838

/-- The coefficient of x^4 in the expanded expression -/
def coefficient_x4 (a b c d e f g : ℤ) : ℤ :=
  5 * a - 3 * 0 + 4 * (-3)

/-- The expression to be expanded -/
def expression (x : ℚ) : ℚ :=
  5 * (x^4 - 2*x^3 + x^2) - 3 * (x^2 - x + 1) + 4 * (x^6 - 3*x^4 + x^3)

theorem coefficient_x4_is_negative_seven :
  coefficient_x4 1 (-2) 1 0 (-1) 1 = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_negative_seven_l898_89838


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l898_89841

theorem divisibility_equivalence (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l898_89841


namespace NUMINAMATH_CALUDE_wendy_scholarship_amount_l898_89863

theorem wendy_scholarship_amount 
  (wendy kelly nina : ℕ)  -- Scholarship amounts for each person
  (h1 : nina = kelly - 8000)  -- Nina's scholarship is $8000 less than Kelly's
  (h2 : kelly = 2 * wendy)    -- Kelly's scholarship is twice Wendy's
  (h3 : wendy + kelly + nina = 92000)  -- Total scholarship amount
  : wendy = 20000 := by
  sorry

end NUMINAMATH_CALUDE_wendy_scholarship_amount_l898_89863


namespace NUMINAMATH_CALUDE_factor_of_expression_l898_89850

theorem factor_of_expression (x y z : ℝ) :
  ∃ (k : ℝ), x^2 - y^2 - z^2 + 2*y*z + 3*x + 2*y - 4*z = (x + y - z) * k := by
  sorry

end NUMINAMATH_CALUDE_factor_of_expression_l898_89850


namespace NUMINAMATH_CALUDE_cos_250_over_sin_200_equals_1_l898_89815

theorem cos_250_over_sin_200_equals_1 :
  (Real.cos (250 * π / 180)) / (Real.sin (200 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_250_over_sin_200_equals_1_l898_89815


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l898_89821

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l898_89821


namespace NUMINAMATH_CALUDE_commute_days_calculation_l898_89871

theorem commute_days_calculation (x : ℕ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : ∃ a b c : ℕ, 
    a + b + c = x ∧  -- Total days
    b + c = 6 ∧      -- Bus to work
    a + c = 18 ∧     -- Bus from work
    a + b = 14) :    -- Train commutes
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_commute_days_calculation_l898_89871


namespace NUMINAMATH_CALUDE_largest_multiple_of_7_less_than_neg_95_l898_89805

theorem largest_multiple_of_7_less_than_neg_95 :
  ∀ n : ℤ, n * 7 < -95 → n * 7 ≤ -98 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_7_less_than_neg_95_l898_89805


namespace NUMINAMATH_CALUDE_inequality_proof_l898_89874

theorem inequality_proof (x a : ℝ) (h : x > a ∧ a > 0) : x^2 > x*a ∧ x*a > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l898_89874


namespace NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l898_89873

theorem smallest_positive_angle_with_same_terminal_side (angle : ℝ) : 
  angle = 1000 →
  (∃ (k : ℤ), angle = 280 + 360 * k) →
  (∀ (x : ℝ), 0 ≤ x ∧ x < 360 ∧ (∃ (m : ℤ), angle = x + 360 * m) → x ≥ 280) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l898_89873


namespace NUMINAMATH_CALUDE_evaluate_g_l898_89880

/-- The function g(x) = 3x^2 - 6x + 5 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- Theorem: 3g(2) + 2g(-4) = 169 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-4) = 169 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l898_89880


namespace NUMINAMATH_CALUDE_compound_weight_l898_89845

/-- Given a compound with a molecular weight of 1050, 
    the total weight of 6 moles of this compound is 6300 grams. -/
theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 1050 → moles = 6 → moles * molecular_weight = 6300 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l898_89845


namespace NUMINAMATH_CALUDE_angle_set_inclusion_l898_89808

def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

theorem angle_set_inclusion : N ⊆ M ∧ M ⊆ P := by sorry

end NUMINAMATH_CALUDE_angle_set_inclusion_l898_89808


namespace NUMINAMATH_CALUDE_max_remainder_division_l898_89887

theorem max_remainder_division (n : ℕ) : 
  (n % 6 < 6) → (n / 6 = 18) → (n % 6 = 5) → n = 113 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_division_l898_89887


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l898_89830

/-- Represents a valid arrangement of multiples of 2013 in a table -/
def ValidArrangement : Type :=
  { arr : Fin 11 → Fin 11 // Function.Injective arr ∧ 
    ∀ i : Fin 11, (2013 * (arr i + 1)) % (i + 1) = 0 }

/-- The number of valid arrangements -/
def numValidArrangements : ℕ := sorry

theorem count_valid_arrangements : numValidArrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l898_89830


namespace NUMINAMATH_CALUDE_no_triple_squares_l898_89893

theorem no_triple_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n.val^2 + 1 = a^2) ∧ (3 * n.val^2 + 1 = b^2) ∧ (6 * n.val^2 + 1 = c^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_triple_squares_l898_89893


namespace NUMINAMATH_CALUDE_equations_not_intersecting_at_roots_l898_89826

theorem equations_not_intersecting_at_roots : ∀ (x : ℝ),
  (x = 0 ∨ x = 3) →
  (x = x - 3) →
  False :=
by sorry

#check equations_not_intersecting_at_roots

end NUMINAMATH_CALUDE_equations_not_intersecting_at_roots_l898_89826


namespace NUMINAMATH_CALUDE_correct_calculation_l898_89827

theorem correct_calculation (x y : ℝ) : 6 * x * y^2 - 3 * y^2 * x = 3 * x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l898_89827


namespace NUMINAMATH_CALUDE_young_inequality_l898_89879

theorem young_inequality (A B p q : ℝ) (hA : A > 0) (hB : B > 0) (hp : p > 0) (hq : q > 0) (hpq : 1/p + 1/q = 1) :
  A^(1/p) * B^(1/q) ≤ A/p + B/q :=
by sorry

end NUMINAMATH_CALUDE_young_inequality_l898_89879
