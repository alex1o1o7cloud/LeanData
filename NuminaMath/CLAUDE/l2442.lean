import Mathlib

namespace NUMINAMATH_CALUDE_petya_vasya_journey_l2442_244285

/-- Represents the problem of Petya and Vasya's journey to the football match. -/
theorem petya_vasya_journey
  (distance : ℝ)
  (walking_speed : ℝ)
  (bicycle_speed_multiplier : ℝ)
  (late_time : ℝ)
  (h1 : distance = 4)
  (h2 : walking_speed = 4)
  (h3 : bicycle_speed_multiplier = 3)
  (h4 : late_time = 10)
  (h5 : distance / walking_speed * 60 - late_time = 50) :
  let bicycle_speed := walking_speed * bicycle_speed_multiplier
  let half_distance := distance / 2
  let walking_time := half_distance / walking_speed * 60
  let cycling_time := half_distance / bicycle_speed * 60
  let total_time := walking_time + cycling_time
  50 - total_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_journey_l2442_244285


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_l2442_244233

theorem sqrt_450_equals_15 : Real.sqrt 450 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_l2442_244233


namespace NUMINAMATH_CALUDE_parabola_directrix_l2442_244232

/-- A parabola with equation y = 1/4 * x^2 has a directrix with equation y = -1 -/
theorem parabola_directrix (x y : ℝ) :
  y = (1/4) * x^2 → ∃ (k : ℝ), k = -1 ∧ (∀ (x₀ y₀ : ℝ), y₀ = k → 
    (x₀ - x)^2 + (y₀ - y)^2 = (y₀ - (y + 1/4))^2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2442_244232


namespace NUMINAMATH_CALUDE_experiment_success_probability_l2442_244278

-- Define the experiment setup
structure ExperimentSetup where
  box1_total : ℕ := 10
  box1_a : ℕ := 7
  box1_b : ℕ := 3
  box2_total : ℕ := 10
  box2_red : ℕ := 5
  box3_total : ℕ := 10
  box3_red : ℕ := 8

-- Define the probability of success
def probability_of_success (setup : ExperimentSetup) : ℚ :=
  let p1 := (setup.box1_a : ℚ) / setup.box1_total * setup.box2_red / setup.box2_total
  let p2 := (setup.box1_b : ℚ) / setup.box1_total * setup.box3_red / setup.box3_total
  p1 + p2

-- Theorem statement
theorem experiment_success_probability (setup : ExperimentSetup) :
  probability_of_success setup = 59 / 100 := by
  sorry

end NUMINAMATH_CALUDE_experiment_success_probability_l2442_244278


namespace NUMINAMATH_CALUDE_complex_number_second_quadrant_l2442_244271

theorem complex_number_second_quadrant (a : ℝ) : 
  let z : ℂ := (a + 3*Complex.I)/Complex.I + a*Complex.I
  (z.re = 0) ∧ (z.im < 0) ∧ (z.re < 0) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_second_quadrant_l2442_244271


namespace NUMINAMATH_CALUDE_gizmos_produced_75_workers_2_hours_l2442_244299

/-- Represents the production rates and worker information for a manufacturing plant. -/
structure ProductionData where
  gadget_rate : ℝ  -- Gadgets produced per worker per hour
  gizmo_rate : ℝ   -- Gizmos produced per worker per hour
  workers : ℕ      -- Number of workers
  hours : ℝ        -- Number of hours worked

/-- Calculates the number of gizmos produced given production data. -/
def gizmos_produced (data : ProductionData) : ℝ :=
  data.gizmo_rate * data.workers * data.hours

/-- States that the number of gizmos produced by 75 workers in 2 hours is 450. -/
theorem gizmos_produced_75_workers_2_hours :
  let data : ProductionData := {
    gadget_rate := 2,
    gizmo_rate := 3,
    workers := 75,
    hours := 2
  }
  gizmos_produced data = 450 := by sorry

end NUMINAMATH_CALUDE_gizmos_produced_75_workers_2_hours_l2442_244299


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_imply_real_coefficients_l2442_244276

theorem complex_conjugate_roots_imply_real_coefficients (a b : ℝ) :
  (∃ x y : ℝ, y ≠ 0 ∧ 
    (Complex.I * y + x) ^ 2 + (6 + Complex.I * a) * (Complex.I * y + x) + (13 + Complex.I * b) = 0 ∧
    (Complex.I * -y + x) ^ 2 + (6 + Complex.I * a) * (Complex.I * -y + x) + (13 + Complex.I * b) = 0) →
  a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_imply_real_coefficients_l2442_244276


namespace NUMINAMATH_CALUDE_fruit_platter_kiwis_l2442_244253

theorem fruit_platter_kiwis 
  (total : ℕ) 
  (oranges apples bananas kiwis : ℕ) 
  (h_total : oranges + apples + bananas + kiwis = total)
  (h_apples : apples = 3 * oranges)
  (h_bananas : bananas = 4 * apples)
  (h_kiwis : kiwis = 5 * bananas)
  (h_total_value : total = 540) :
  kiwis = 420 := by
  sorry

end NUMINAMATH_CALUDE_fruit_platter_kiwis_l2442_244253


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2442_244225

/-- An arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def is_geometric (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The sequence a_n + 2^n * b_n forms an arithmetic sequence for n = 1, 3, 5 -/
def special_sequence_arithmetic (a b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, (a 3 + 4 * b 3) - (a 1 + 2 * b 1) = d ∧
            (a 5 + 8 * b 5) - (a 3 + 4 * b 3) = d

theorem geometric_sequence_ratio (a b : ℕ → ℝ) :
  is_arithmetic a →
  is_geometric b →
  special_sequence_arithmetic a b →
  b 3 * b 7 / (b 4 ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2442_244225


namespace NUMINAMATH_CALUDE_typhoon_tree_difference_l2442_244269

theorem typhoon_tree_difference (initial_trees : ℕ) (dead_trees : ℕ) : 
  initial_trees = 3 → dead_trees = 13 → dead_trees - initial_trees = 10 :=
by sorry

end NUMINAMATH_CALUDE_typhoon_tree_difference_l2442_244269


namespace NUMINAMATH_CALUDE_hiker_speed_day3_l2442_244257

/-- A hiker's three-day journey --/
structure HikerJourney where
  day1_distance : ℝ
  day1_speed : ℝ
  day2_hours_reduction : ℝ
  day2_speed_increase : ℝ
  day3_hours : ℝ
  total_distance : ℝ

/-- Theorem about the hiker's speed on the third day --/
theorem hiker_speed_day3 (journey : HikerJourney)
  (h1 : journey.day1_distance = 18)
  (h2 : journey.day1_speed = 3)
  (h3 : journey.day2_hours_reduction = 1)
  (h4 : journey.day2_speed_increase = 1)
  (h5 : journey.day3_hours = 3)
  (h6 : journey.total_distance = 53) :
  (journey.total_distance
    - journey.day1_distance
    - (journey.day1_distance / journey.day1_speed - journey.day2_hours_reduction)
      * (journey.day1_speed + journey.day2_speed_increase))
  / journey.day3_hours = 5 := by
  sorry


end NUMINAMATH_CALUDE_hiker_speed_day3_l2442_244257


namespace NUMINAMATH_CALUDE_unique_base_solution_l2442_244247

/-- Given a natural number b ≥ 2, convert a number in base b to its decimal representation -/
def toDecimal (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Given a natural number b ≥ 2, check if the equation 161_b + 134_b = 315_b holds -/
def checkEquation (b : ℕ) : Prop :=
  toDecimal 161 b + toDecimal 134 b = toDecimal 315 b

/-- The main theorem stating that 8 is the unique solution to the equation -/
theorem unique_base_solution :
  ∃! b : ℕ, b ≥ 2 ∧ checkEquation b ∧ b = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l2442_244247


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l2442_244221

theorem meal_cost_calculation (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  total_bill / (adults + children : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l2442_244221


namespace NUMINAMATH_CALUDE_linear_inequality_solution_l2442_244292

theorem linear_inequality_solution (a : ℝ) : 
  (|2 + 3 * a| = 1) ↔ (a = -1 ∨ a = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_l2442_244292


namespace NUMINAMATH_CALUDE_equation_has_real_root_l2442_244259

theorem equation_has_real_root (K : ℝ) (h : K ≠ 0) :
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l2442_244259


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_quadratic_roots_difference_l2442_244200

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*m*x + 3*m^2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: If m > 0 and the difference between roots is 2, then m = 1
theorem quadratic_roots_difference (m : ℝ) :
  m > 0 →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁ - x₂ = 2) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_quadratic_roots_difference_l2442_244200


namespace NUMINAMATH_CALUDE_riyas_speed_l2442_244229

/-- Proves that Riya's speed is 21 kmph given the problem conditions -/
theorem riyas_speed (riya_speed priya_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  priya_speed = 22 →
  time = 1 →
  distance = 43 →
  distance = (riya_speed + priya_speed) * time →
  riya_speed = 21 := by
  sorry

#check riyas_speed

end NUMINAMATH_CALUDE_riyas_speed_l2442_244229


namespace NUMINAMATH_CALUDE_determine_xy_condition_l2442_244248

/-- Given two integers m and n, this theorem states the conditions under which 
    it's always possible to determine xy given x^m + y^m and x^n + y^n. -/
theorem determine_xy_condition (m n : ℤ) :
  (∀ x y : ℝ, ∃! (xy : ℝ), ∀ x' y' : ℝ, 
    x'^m + y'^m = x^m + y^m ∧ x'^n + y'^n = x^n + y^n → x' * y' = xy) ↔
  (∃ k t : ℤ, m = 2*k + 1 ∧ n = 2*t*(2*k + 1) ∧ t > 0) :=
sorry

end NUMINAMATH_CALUDE_determine_xy_condition_l2442_244248


namespace NUMINAMATH_CALUDE_day_division_count_l2442_244242

-- Define the number of seconds in a day
def seconds_in_day : ℕ := 72000

-- Define a function to count the number of ways to divide the day
def count_divisions (total_seconds : ℕ) : ℕ :=
  -- The actual implementation is not provided, as per instructions
  sorry

-- Theorem statement
theorem day_division_count :
  count_divisions seconds_in_day = 60 := by sorry

end NUMINAMATH_CALUDE_day_division_count_l2442_244242


namespace NUMINAMATH_CALUDE_distance_between_B_and_D_l2442_244205

theorem distance_between_B_and_D 
  (a b c d : ℝ) 
  (h1 : |2*a - 3*c| = 1) 
  (h2 : |2*b - 3*c| = 1) 
  (h3 : 2/3 * |d - a| = 1) 
  (h4 : a ≠ b) : 
  |d - b| = 1/2 ∨ |d - b| = 5/2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_B_and_D_l2442_244205


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2442_244211

theorem inequality_system_solution_set :
  let S := { x : ℝ | x - 1 < 7 ∧ 3 * x + 1 ≥ -2 }
  S = { x : ℝ | -1 ≤ x ∧ x < 8 } :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2442_244211


namespace NUMINAMATH_CALUDE_jose_and_jane_time_l2442_244231

-- Define the time taken by Jose to complete the task alone
def jose_time : ℝ := 15

-- Define the total time when Jose does half and Jane does half
def half_half_time : ℝ := 15

-- Define the time taken by Jose and Jane together
def combined_time : ℝ := 7.5

-- Theorem statement
theorem jose_and_jane_time : 
  (jose_time : ℝ) = 15 ∧ 
  (half_half_time : ℝ) = 15 → 
  (combined_time : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_jose_and_jane_time_l2442_244231


namespace NUMINAMATH_CALUDE_taxi_charge_theorem_l2442_244212

/-- Calculates the total charge for a taxi trip given the initial fee, rate per increment, increment distance, and total distance. -/
def totalCharge (initialFee : ℚ) (ratePerIncrement : ℚ) (incrementDistance : ℚ) (totalDistance : ℚ) : ℚ :=
  initialFee + (totalDistance / incrementDistance).floor * ratePerIncrement

/-- Theorem stating that the total charge for a 3.6-mile trip with given fee structure is $3.60 -/
theorem taxi_charge_theorem :
  let initialFee : ℚ := 225/100
  let ratePerIncrement : ℚ := 15/100
  let incrementDistance : ℚ := 2/5
  let totalDistance : ℚ := 36/10
  totalCharge initialFee ratePerIncrement incrementDistance totalDistance = 360/100 := by
  sorry


end NUMINAMATH_CALUDE_taxi_charge_theorem_l2442_244212


namespace NUMINAMATH_CALUDE_unique_quadruple_solution_l2442_244273

theorem unique_quadruple_solution :
  ∃! (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d) * (a^4 + b^4 + c^4 + d^4) = 32 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_solution_l2442_244273


namespace NUMINAMATH_CALUDE_inequalities_not_always_true_l2442_244297

theorem inequalities_not_always_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : abs x < abs a) (hyb : abs y > abs b) :
  ∃ (x' y' a' b' : ℝ), 
    x' ≠ 0 ∧ y' ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    abs x' < abs a' ∧ abs y' > abs b' ∧
    ¬(abs (x' + y') < abs (a' + b')) ∧
    ¬(abs (x' - y') < abs (a' - b')) ∧
    ¬(abs (x' * y') < abs (a' * b')) ∧
    ¬(abs (x' / y') < abs (a' / b')) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_always_true_l2442_244297


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2442_244294

theorem solution_set_of_inequality (x : ℝ) :
  {x : ℝ | 4 * x^2 - 4 * x + 1 ≤ 0} = {1/2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2442_244294


namespace NUMINAMATH_CALUDE_equation_root_of_increase_implies_m_equals_two_l2442_244237

-- Define the equation
def equation (x m : ℝ) : Prop := (x - 1) / (x - 3) = m / (x - 3)

-- Define the root of increase
def root_of_increase (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem equation_root_of_increase_implies_m_equals_two :
  ∀ x m : ℝ, equation x m → root_of_increase x → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_of_increase_implies_m_equals_two_l2442_244237


namespace NUMINAMATH_CALUDE_rectangle_area_l2442_244282

theorem rectangle_area (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) :
  square_side ^ 2 = 1296 →
  circle_radius = square_side →
  rectangle_length = circle_radius / 6 →
  rectangle_breadth = 10 →
  rectangle_length * rectangle_breadth = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2442_244282


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_three_l2442_244236

/-- The polynomial x^3 - 7x^2 + 12x - 18 -/
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 12*x - 18

/-- The sum of the kth powers of the roots of p -/
def s (k : ℕ) : ℝ := sorry

/-- The recursive relationship for s_k -/
def recursive_relation (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 2 → s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)

theorem sum_of_coefficients_is_negative_three :
  ∀ a b c : ℝ,
  (∃ α β γ : ℝ, p α = 0 ∧ p β = 0 ∧ p γ = 0) →
  s 0 = 3 →
  s 1 = 7 →
  s 2 = 13 →
  recursive_relation a b c →
  a + b + c = -3 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_three_l2442_244236


namespace NUMINAMATH_CALUDE_x_value_l2442_244220

/-- The equation that defines x -/
def x_equation (x : ℝ) : Prop := x = Real.sqrt (2 + x)

/-- Theorem stating that the solution to the equation is 2 -/
theorem x_value : ∃ x : ℝ, x_equation x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_x_value_l2442_244220


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l2442_244279

theorem smallest_divisible_by_10_and_24 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 0 → m % 24 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l2442_244279


namespace NUMINAMATH_CALUDE_not_perfect_square_l2442_244202

theorem not_perfect_square (a b : ℕ+) : ¬ ∃ k : ℤ, (a : ℤ)^2 + Int.ceil ((4 * (a : ℤ)^2) / (b : ℤ)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2442_244202


namespace NUMINAMATH_CALUDE_parabola_point_x_coord_l2442_244296

/-- The x-coordinate of a point on a parabola given its distance from the focus -/
theorem parabola_point_x_coord (x y : ℝ) : 
  y^2 = 4*x →  -- Point P(x,y) lies on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 3^2 →  -- Distance from P to focus (1,0) is 3
  x = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coord_l2442_244296


namespace NUMINAMATH_CALUDE_bus_driver_worked_54_hours_l2442_244215

/-- Represents the bus driver's compensation structure and work details for a week --/
structure BusDriverWeek where
  regularRate : ℝ
  overtimeRateMultiplier : ℝ
  regularHoursLimit : ℕ
  bonusPerPassenger : ℝ
  totalCompensation : ℝ
  passengersTransported : ℕ

/-- Calculates the total hours worked by the bus driver --/
def totalHoursWorked (week : BusDriverWeek) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, the bus driver worked 54 hours --/
theorem bus_driver_worked_54_hours :
  let week : BusDriverWeek := {
    regularRate := 14,
    overtimeRateMultiplier := 1.75,
    regularHoursLimit := 40,
    bonusPerPassenger := 0.25,
    totalCompensation := 998,
    passengersTransported := 350
  }
  totalHoursWorked week = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_worked_54_hours_l2442_244215


namespace NUMINAMATH_CALUDE_staircase_arrangement_count_l2442_244251

/-- The number of ways to arrange 3 people on a 7-step staircase --/
def arrangement_count : ℕ := 336

/-- The number of steps on the staircase --/
def num_steps : ℕ := 7

/-- The maximum number of people that can stand on a single step --/
def max_per_step : ℕ := 2

/-- The number of people to be arranged on the staircase --/
def num_people : ℕ := 3

/-- Theorem stating that the number of arrangements is 336 --/
theorem staircase_arrangement_count :
  arrangement_count = 336 :=
by sorry

end NUMINAMATH_CALUDE_staircase_arrangement_count_l2442_244251


namespace NUMINAMATH_CALUDE_c_k_value_l2442_244241

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℝ) (n : ℕ) : ℝ :=
  1 + (n - 1 : ℝ) * d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℝ) (n : ℕ) : ℝ :=
  r ^ (n - 1)

/-- Sum of nth terms of arithmetic and geometric sequences -/
def c_seq (d r : ℝ) (n : ℕ) : ℝ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r : ℝ) (k : ℕ) :
  (∃ k : ℕ, c_seq d r (k - 1) = 150 ∧ c_seq d r (k + 1) = 900) →
  c_seq d r k = 314 := by
  sorry

end NUMINAMATH_CALUDE_c_k_value_l2442_244241


namespace NUMINAMATH_CALUDE_total_ground_beef_weight_l2442_244254

theorem total_ground_beef_weight (package_weight : ℕ) (butcher1_packages : ℕ) (butcher2_packages : ℕ) (butcher3_packages : ℕ) 
  (h1 : package_weight = 4)
  (h2 : butcher1_packages = 10)
  (h3 : butcher2_packages = 7)
  (h4 : butcher3_packages = 8) :
  package_weight * (butcher1_packages + butcher2_packages + butcher3_packages) = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_ground_beef_weight_l2442_244254


namespace NUMINAMATH_CALUDE_area_perimeter_product_l2442_244249

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℕ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

/-- Represents a square on the grid -/
structure Square where
  E : Point
  F : Point
  G : Point
  H : Point

/-- The specific square EFGH from the problem -/
def EFGH : Square :=
  { E := { x := 1, y := 5 },
    F := { x := 5, y := 6 },
    G := { x := 6, y := 2 },
    H := { x := 2, y := 1 } }

/-- Theorem stating the product of area and perimeter of EFGH -/
theorem area_perimeter_product (s : Square) (h : s = EFGH) :
  (↑(squaredDistance s.E s.F) : ℝ) * (4 * Real.sqrt (↑(squaredDistance s.E s.F))) = 68 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_area_perimeter_product_l2442_244249


namespace NUMINAMATH_CALUDE_probability_point_in_ellipsoid_l2442_244288

/-- The probability of a point in a rectangular prism satisfying an ellipsoid equation -/
theorem probability_point_in_ellipsoid : 
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  let probability := ellipsoid_volume / prism_volume
  probability = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_ellipsoid_l2442_244288


namespace NUMINAMATH_CALUDE_final_number_can_be_zero_l2442_244207

/-- Represents the operation of replacing two numbers with their absolute difference -/
def difference_operation (S : Finset ℕ) : Finset ℕ :=
  sorry

/-- The initial set of integers from 1 to 2013 -/
def initial_set : Finset ℕ :=
  Finset.range 2013

/-- Applies the difference operation n times to the given set -/
def apply_n_times (S : Finset ℕ) (n : ℕ) : Finset ℕ :=
  sorry

theorem final_number_can_be_zero :
  ∃ (result : Finset ℕ), apply_n_times initial_set 2012 = result ∧ 0 ∈ result :=
sorry

end NUMINAMATH_CALUDE_final_number_can_be_zero_l2442_244207


namespace NUMINAMATH_CALUDE_calm_snakes_not_blue_l2442_244274

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (isBlue : Snake → Prop)
variable (isCalm : Snake → Prop)
variable (canMultiply : Snake → Prop)
variable (canDivide : Snake → Prop)

-- State the theorem
theorem calm_snakes_not_blue 
  (h1 : ∀ s : Snake, isCalm s → canMultiply s)
  (h2 : ∀ s : Snake, isBlue s → ¬canDivide s)
  (h3 : ∀ s : Snake, ¬canDivide s → ¬canMultiply s) :
  ∀ s : Snake, isCalm s → ¬isBlue s :=
by
  sorry


end NUMINAMATH_CALUDE_calm_snakes_not_blue_l2442_244274


namespace NUMINAMATH_CALUDE_cyclic_iff_concurrent_l2442_244263

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if four points are cyclic -/
def are_cyclic (A B C D : Point) : Prop :=
  sorry

/-- Check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Get the line passing through two points -/
def line_through_points (A B : Point) : Line :=
  sorry

theorem cyclic_iff_concurrent (A B C D E F : Point) :
  are_cyclic A B C D → are_cyclic C D E F →
  (are_cyclic A B E F ↔ 
    are_concurrent 
      (line_through_points A B) 
      (line_through_points C D) 
      (line_through_points E F)) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_iff_concurrent_l2442_244263


namespace NUMINAMATH_CALUDE_area_triangle_DEF_area_triangle_DEF_is_six_l2442_244280

/-- Triangle DEF with vertices D, E, and F, where F lies on the line x + y = 6 -/
structure TriangleDEF where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_D : D = (2, 1)
  h_E : E = (1, 4)
  h_F : F.1 + F.2 = 6

/-- The area of triangle DEF is 6 -/
theorem area_triangle_DEF (t : TriangleDEF) : ℝ :=
  6

/-- The area of triangle DEF is indeed 6 -/
theorem area_triangle_DEF_is_six (t : TriangleDEF) :
  area_triangle_DEF t = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_DEF_area_triangle_DEF_is_six_l2442_244280


namespace NUMINAMATH_CALUDE_trapezoidal_fence_poles_l2442_244289

/-- Calculates the number of poles needed for a trapezoidal fence --/
theorem trapezoidal_fence_poles
  (parallel_side1 parallel_side2 non_parallel_side : ℕ)
  (parallel_pole_interval non_parallel_pole_interval : ℕ)
  (h1 : parallel_side1 = 60)
  (h2 : parallel_side2 = 80)
  (h3 : non_parallel_side = 50)
  (h4 : parallel_pole_interval = 5)
  (h5 : non_parallel_pole_interval = 7) :
  (parallel_side1 / parallel_pole_interval + 1) +
  (parallel_side2 / parallel_pole_interval + 1) +
  2 * (⌈(non_parallel_side : ℝ) / non_parallel_pole_interval⌉ + 1) - 4 = 44 := by
  sorry

#check trapezoidal_fence_poles

end NUMINAMATH_CALUDE_trapezoidal_fence_poles_l2442_244289


namespace NUMINAMATH_CALUDE_library_book_loan_l2442_244250

theorem library_book_loan (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 65 / 100)
  (h3 : final_books = 54)
  : ∃ (loaned_books : ℕ), loaned_books = 60 ∧ 
    (initial_books - final_books : ℚ) = (1 - return_rate) * loaned_books := by
  sorry

end NUMINAMATH_CALUDE_library_book_loan_l2442_244250


namespace NUMINAMATH_CALUDE_cookie_radius_l2442_244210

theorem cookie_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 6*x + 10*y + 12 = 0 ↔ (x - 3)^2 + (y + 5)^2 = r^2) →
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 6*x + 10*y + 12 = 0 ↔ (x - 3)^2 + (y + 5)^2 = r^2 ∧ r = Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l2442_244210


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2442_244204

/-- The ratio of the area of an inscribed circle to the area of a right triangle -/
theorem inscribed_circle_area_ratio (h a r : ℝ) (h_pos : h > 0) (a_pos : a > 0) (r_pos : r > 0) 
  (h_gt_a : h > a) :
  let b := Real.sqrt (h^2 - a^2)
  let s := (a + b + h) / 2
  let triangle_area := (1 / 2) * a * b
  let circle_area := π * r^2
  (r * s = triangle_area) →
  (circle_area / triangle_area = π * a * (h^2 - a^2) / (2 * (a + b + h))) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2442_244204


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l2442_244264

theorem unique_positive_integer_solution :
  ∃! (m n : ℕ+), 15 * m * n = 75 - 5 * m - 3 * n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l2442_244264


namespace NUMINAMATH_CALUDE_day_relationship_l2442_244244

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek : YearDay → DayOfWeek := sorry

/-- Theorem stating the relationship between the days in different years -/
theorem day_relationship (N : Int) :
  dayOfWeek { year := N, day := 275 } = DayOfWeek.Thursday →
  dayOfWeek { year := N + 1, day := 215 } = DayOfWeek.Thursday →
  dayOfWeek { year := N - 1, day := 150 } = DayOfWeek.Saturday :=
by sorry

end NUMINAMATH_CALUDE_day_relationship_l2442_244244


namespace NUMINAMATH_CALUDE_junk_mail_calculation_l2442_244230

theorem junk_mail_calculation (blocks : ℕ) (houses_per_block : ℕ) (mail_per_house : ℕ)
  (h1 : blocks = 16)
  (h2 : houses_per_block = 17)
  (h3 : mail_per_house = 4) :
  blocks * houses_per_block * mail_per_house = 1088 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_calculation_l2442_244230


namespace NUMINAMATH_CALUDE_total_time_circling_island_l2442_244227

def time_per_round : ℕ := 30
def saturday_rounds : ℕ := 11
def sunday_rounds : ℕ := 15

theorem total_time_circling_island : 
  time_per_round * (saturday_rounds + sunday_rounds) = 780 := by
  sorry

end NUMINAMATH_CALUDE_total_time_circling_island_l2442_244227


namespace NUMINAMATH_CALUDE_expression_evaluation_l2442_244219

theorem expression_evaluation :
  1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) =
  1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2442_244219


namespace NUMINAMATH_CALUDE_arrangements_five_singers_l2442_244238

/-- The number of singers --/
def n : ℕ := 5

/-- The number of different arrangements for n singers with constraints --/
def arrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1) + (n - 2) * (n - 2) * Nat.factorial (n - 2)

/-- Theorem: The number of arrangements for 5 singers with constraints is 78 --/
theorem arrangements_five_singers : arrangements n = 78 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_five_singers_l2442_244238


namespace NUMINAMATH_CALUDE_solution_value_l2442_244293

theorem solution_value (a b : ℝ) : 
  (2 * a + (-1) * b = 1) →
  (2 * b + (-1) * a = 7) →
  (a + b) * (a - b) = -16 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l2442_244293


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l2442_244214

theorem right_triangle_ratio (a d : ℝ) (h_d_pos : d > 0) (h_d_odd : ∃ k : ℤ, d = 2 * k + 1) :
  (a + 4 * d)^2 = a^2 + (a + 2 * d)^2 → a / d = 1 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l2442_244214


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2442_244216

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : 3 * Real.cos (2 * α) + Real.sin α = 1) : 
  Real.sin (Real.pi - α) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2442_244216


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2442_244201

-- Problem 1
theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - a*(a + 4*b) = 4*b^2 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -1) :
  ((2 / (m - 1) + 1) / ((2*m + 2) / (m^2 - 2*m + 1))) = (m - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2442_244201


namespace NUMINAMATH_CALUDE_jackson_investment_ratio_l2442_244243

-- Define the initial investment amount
def initial_investment : ℝ := 500

-- Define Brandon's final investment as a percentage of the initial
def brandon_final_percentage : ℝ := 0.2

-- Define the difference between Jackson's and Brandon's final investments
def investment_difference : ℝ := 1900

-- Theorem to prove
theorem jackson_investment_ratio :
  let brandon_final := initial_investment * brandon_final_percentage
  let jackson_final := brandon_final + investment_difference
  jackson_final / initial_investment = 4 := by
  sorry

end NUMINAMATH_CALUDE_jackson_investment_ratio_l2442_244243


namespace NUMINAMATH_CALUDE_cubic_identity_l2442_244266

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2442_244266


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2442_244286

theorem largest_integer_satisfying_inequality : 
  ∀ x : ℤ, x ≤ 3 ↔ (x : ℚ) / 4 + 7 / 6 < 8 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2442_244286


namespace NUMINAMATH_CALUDE_intersection_length_range_l2442_244275

def interval_length (a b : ℝ) := b - a

theorem intersection_length_range :
  ∀ (a b : ℝ),
  (∀ x ∈ {x | a ≤ x ∧ x ≤ a+1}, -1 ≤ x ∧ x ≤ 1) →
  (∀ x ∈ {x | b-3/2 ≤ x ∧ x ≤ b}, -1 ≤ x ∧ x ≤ 1) →
  ∃ (l : ℝ), l = interval_length (max a (b-3/2)) (min (a+1) b) ∧
  1/2 ≤ l ∧ l ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_range_l2442_244275


namespace NUMINAMATH_CALUDE_fraction_equality_l2442_244223

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) : 
  18 / 7 + (2 * q - p) / ((14/5) * q) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2442_244223


namespace NUMINAMATH_CALUDE_lines_are_parallel_l2442_244268

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem lines_are_parallel : 
  let line1 : Line := { a := 1, b := -1, c := 2 }
  let line2 : Line := { a := 1, b := -1, c := 1 }
  parallel line1 line2 := by sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l2442_244268


namespace NUMINAMATH_CALUDE_distance_ratio_forms_circle_l2442_244222

/-- Given points A(0,0) and B(1,0) on a plane, the set of all points M(x,y) such that 
    the distance from M to A is three times the distance from M to B forms a circle 
    with center (-1/8, 0) and radius 3/8. -/
theorem distance_ratio_forms_circle :
  ∀ (x y : ℝ),
    (Real.sqrt (x^2 + y^2) = 3 * Real.sqrt ((x-1)^2 + y^2)) →
    ((x + 1/8)^2 + y^2 = (3/8)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_forms_circle_l2442_244222


namespace NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l2442_244291

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 := by
  sorry

theorem max_y_value_achievable : ∃ x y : ℤ, x * y + 3 * x + 2 * y = -4 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l2442_244291


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l2442_244270

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed. -/
def upstreamSpeed (downstreamSpeed streamSpeed : ℝ) : ℝ :=
  downstreamSpeed - 2 * streamSpeed

/-- Theorem stating that given a downstream speed of 10 kmph and a stream speed of 1 kmph, 
    the upstream speed is 8 kmph. -/
theorem upstream_speed_calculation :
  upstreamSpeed 10 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_calculation_l2442_244270


namespace NUMINAMATH_CALUDE_quadratic_sum_l2442_244245

/-- A quadratic function f(x) = px² + qx + r with vertex (3, 4) passing through (1, 2) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := fun x ↦ p * x^2 + q * x + r

theorem quadratic_sum (p q r : ℝ) :
  (∀ x, QuadraticFunction p q r x = p * x^2 + q * x + r) →
  (∃ a, ∀ x, QuadraticFunction p q r x = a * (x - 3)^2 + 4) →
  QuadraticFunction p q r 1 = 2 →
  p + q + r = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2442_244245


namespace NUMINAMATH_CALUDE_sphere_radius_is_one_l2442_244256

/-- Represents a cone with a given base radius -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of three cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  sphere : Sphere
  sameHeight : cone1.height = cone2.height ∧ cone2.height = cone3.height
  baseRadii : cone1.baseRadius = 1 ∧ cone2.baseRadius = 2 ∧ cone3.baseRadius = 3
  touching : True  -- Cones are touching each other
  sphereTouchingCones : True  -- Sphere touches all cones
  sphereTouchingTable : True  -- Sphere touches the table
  centerEquidistant : True  -- Center of sphere is equidistant from all points of contact with cones

/-- The theorem stating that the radius of the sphere is 1 -/
theorem sphere_radius_is_one (problem : ConeSphereProblem) : problem.sphere.radius = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_one_l2442_244256


namespace NUMINAMATH_CALUDE_annas_car_rental_cost_l2442_244272

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, 
    number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (perMileRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Proves that Anna's car rental cost is $275 given the specified conditions. -/
theorem annas_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_annas_car_rental_cost_l2442_244272


namespace NUMINAMATH_CALUDE_evaluate_expression_l2442_244235

theorem evaluate_expression (b : ℝ) (h : b = 2) : (6*b^2 - 15*b + 7)*(3*b - 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2442_244235


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_1_range_when_a_ge_1_l2442_244252

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1: Solution set when a = 1
theorem solution_set_when_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = Set.Icc (-3) 2 := by sorry

-- Part 2: Range of f(x) when a ≥ 1
theorem range_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  Set.range (f a) = Set.Ici (a + 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_1_range_when_a_ge_1_l2442_244252


namespace NUMINAMATH_CALUDE_continued_fraction_value_l2442_244290

theorem continued_fraction_value : ∃ y : ℝ, y > 0 ∧ y = 3 + 9 / (2 + 9 / y) ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l2442_244290


namespace NUMINAMATH_CALUDE_expression_factorization_l2442_244260

theorem expression_factorization (x : ℝ) :
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2442_244260


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l2442_244217

theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_vertices : List (ℝ × ℝ × ℝ) := [(0, 0, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)]
  let cube_surface_area : ℝ := 6 * cube_side_length ^ 2
  let tetrahedron_side_length : ℝ := Real.sqrt ((2 - 0)^2 + (2 - 0)^2 + (0 - 0)^2)
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l2442_244217


namespace NUMINAMATH_CALUDE_oil_after_eight_hours_l2442_244287

/-- Represents the remaining oil in a car's fuel tank as a function of time -/
def remaining_oil (initial_oil : ℝ) (consumption_rate : ℝ) (time : ℝ) : ℝ :=
  initial_oil - consumption_rate * time

theorem oil_after_eight_hours 
  (initial_oil : ℝ) 
  (consumption_rate : ℝ) 
  (h1 : initial_oil = 50) 
  (h2 : consumption_rate = 5) :
  remaining_oil initial_oil consumption_rate 8 = 10 := by
  sorry

#check oil_after_eight_hours

end NUMINAMATH_CALUDE_oil_after_eight_hours_l2442_244287


namespace NUMINAMATH_CALUDE_calculator_price_proof_l2442_244213

theorem calculator_price_proof (total_calculators : ℕ) (total_sales : ℕ) 
  (first_type_count : ℕ) (first_type_price : ℕ) (second_type_count : ℕ) :
  total_calculators = 85 →
  total_sales = 3875 →
  first_type_count = 35 →
  first_type_price = 15 →
  second_type_count = total_calculators - first_type_count →
  (first_type_count * first_type_price + second_type_count * 67 = total_sales) :=
by
  sorry

#check calculator_price_proof

end NUMINAMATH_CALUDE_calculator_price_proof_l2442_244213


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l2442_244265

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ k : ℕ, k > 0 →
    (∃ m : ℤ, x^(2*k-1) + y^(2*k-1) = m * (x + y)) →
    (∃ n : ℤ, x^(2*k+1) + y^(2*k+1) = n * (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l2442_244265


namespace NUMINAMATH_CALUDE_simplify_monomial_l2442_244261

theorem simplify_monomial (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_monomial_l2442_244261


namespace NUMINAMATH_CALUDE_train_speed_l2442_244224

/-- Given a train of length 800 meters that crosses an electric pole in 20 seconds,
    prove that its speed is 144 km/h. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ)
    (h1 : train_length = 800)
    (h2 : crossing_time = 20)
    (h3 : speed_ms = train_length / crossing_time)
    (h4 : speed_kmh = speed_ms * 3.6) :
    speed_kmh = 144 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2442_244224


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_m_part_II_l2442_244240

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- Theorem for part (I)
theorem solution_set_part_I :
  {x : ℝ | f x ≥ 3 - |x - 2|} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} :=
sorry

-- Theorem for part (II)
theorem range_of_m_part_II :
  ∀ m : ℝ, (∃ x : ℝ, f x ≤ 2*m - |x + 4|) → m ≥ 7/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_m_part_II_l2442_244240


namespace NUMINAMATH_CALUDE_four_solutions_range_l2442_244209

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := abs (x - 2)

-- Define the proposition
theorem four_solutions_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    a * (f x₁)^2 - f x₁ + 1 = 0 ∧
    a * (f x₂)^2 - f x₂ + 1 = 0 ∧
    a * (f x₃)^2 - f x₃ + 1 = 0 ∧
    a * (f x₄)^2 - f x₄ + 1 = 0) →
  0 < a ∧ a < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_four_solutions_range_l2442_244209


namespace NUMINAMATH_CALUDE_unique_solution_for_B_l2442_244277

theorem unique_solution_for_B : ∃! B : ℕ, ∃ A : ℕ, 
  (A < 10 ∧ B < 10) ∧ (100 * A + 78 - (20 * B + B) = 364) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_B_l2442_244277


namespace NUMINAMATH_CALUDE_expression_evaluation_l2442_244208

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := 3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 3*y*z = 33 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2442_244208


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2442_244246

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 + a - 1 = 0) → 
  (b^3 - 2*b^2 + b - 1 = 0) → 
  (c^3 - 2*c^2 + c - 1 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = -5) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2442_244246


namespace NUMINAMATH_CALUDE_min_teachers_is_ten_l2442_244226

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  math : Nat
  physics : Nat
  chemistry : Nat
  biology : Nat
  computerScience : Nat

/-- Represents the school schedule constraints -/
structure SchoolConstraints where
  teacherCounts : TeacherCounts
  maxSubjectsPerTeacher : Nat
  periodsPerDay : Nat

/-- Calculates the total number of teaching slots required per day -/
def totalSlotsPerDay (c : SchoolConstraints) : Nat :=
  (c.teacherCounts.math + c.teacherCounts.physics + c.teacherCounts.chemistry +
   c.teacherCounts.biology + c.teacherCounts.computerScience) * c.periodsPerDay

/-- Calculates the number of slots a single teacher can fill per day -/
def slotsPerTeacher (c : SchoolConstraints) : Nat :=
  c.maxSubjectsPerTeacher * c.periodsPerDay

/-- Calculates the minimum number of teachers required -/
def minTeachersRequired (c : SchoolConstraints) : Nat :=
  (totalSlotsPerDay c + slotsPerTeacher c - 1) / slotsPerTeacher c

/-- The main theorem stating the minimum number of teachers required -/
theorem min_teachers_is_ten (c : SchoolConstraints) :
  c.teacherCounts = { math := 5, physics := 4, chemistry := 4, biology := 4, computerScience := 3 } →
  c.maxSubjectsPerTeacher = 2 →
  c.periodsPerDay = 6 →
  minTeachersRequired c = 10 := by
  sorry


end NUMINAMATH_CALUDE_min_teachers_is_ten_l2442_244226


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l2442_244262

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 30)
  (h2 : b + d = 5) : 
  a + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l2442_244262


namespace NUMINAMATH_CALUDE_swimming_pool_area_l2442_244203

/-- Theorem: Area of a rectangular swimming pool --/
theorem swimming_pool_area (w l : ℝ) (h1 : l = 3 * w + 10) (h2 : 2 * w + 2 * l = 320) :
  w * l = 4593.75 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_area_l2442_244203


namespace NUMINAMATH_CALUDE_range_of_f_l2442_244218

def f (x : ℝ) : ℝ := x^2 - 2*x + 9

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, f x = y) →
  y ∈ Set.Icc 8 12 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2442_244218


namespace NUMINAMATH_CALUDE_roberta_garage_sale_records_l2442_244206

/-- The number of records Roberta bought at the garage sale -/
def records_bought_at_garage_sale (initial_records : ℕ) (gifted_records : ℕ) (days_per_record : ℕ) (total_listening_days : ℕ) : ℕ :=
  (total_listening_days / days_per_record) - (initial_records + gifted_records)

/-- Theorem stating that Roberta bought 30 records at the garage sale -/
theorem roberta_garage_sale_records : 
  records_bought_at_garage_sale 8 12 2 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_roberta_garage_sale_records_l2442_244206


namespace NUMINAMATH_CALUDE_equal_wealth_after_transfer_l2442_244295

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equal_wealth_after_transfer (w : MerchantWealth) 
  (h : problem_conditions w) : 
  w.foma - 55 = w.ierema + 55 := by
  sorry

end NUMINAMATH_CALUDE_equal_wealth_after_transfer_l2442_244295


namespace NUMINAMATH_CALUDE_females_soccer_not_basketball_l2442_244255

/-- Represents the number of students in various categories -/
structure SchoolTeams where
  soccer_males : ℕ
  soccer_females : ℕ
  basketball_males : ℕ
  basketball_females : ℕ
  males_in_both : ℕ
  total_students : ℕ

/-- The theorem to be proved -/
theorem females_soccer_not_basketball (teams : SchoolTeams)
  (h1 : teams.soccer_males = 120)
  (h2 : teams.soccer_females = 60)
  (h3 : teams.basketball_males = 100)
  (h4 : teams.basketball_females = 80)
  (h5 : teams.males_in_both = 70)
  (h6 : teams.total_students = 260) :
  teams.soccer_females - (teams.soccer_females + teams.basketball_females - 
    (teams.total_students - (teams.soccer_males + teams.basketball_males - teams.males_in_both))) = 30 := by
  sorry


end NUMINAMATH_CALUDE_females_soccer_not_basketball_l2442_244255


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_mod_11_and_7_l2442_244284

theorem greatest_three_digit_number_mod_11_and_7 :
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    n % 11 = 10 ∧ 
    n % 7 = 4 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 11 = 10 ∧ m % 7 = 4 → m ≤ n) ∧
    n = 956 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_mod_11_and_7_l2442_244284


namespace NUMINAMATH_CALUDE_equal_consecutive_subgroup_exists_l2442_244281

/-- A person can be either of type A or type B -/
inductive PersonType
| A
| B

/-- A circular arrangement of people -/
def CircularArrangement := List PersonType

/-- Count the number of type A persons in a list -/
def countTypeA : List PersonType → Nat
| [] => 0
| (PersonType.A :: rest) => 1 + countTypeA rest
| (_ :: rest) => countTypeA rest

/-- Take n consecutive elements from a circular list, starting from index i -/
def takeCircular (n : Nat) (i : Nat) (l : List α) : List α :=
  (List.drop i l ++ l).take n

/-- Main theorem -/
theorem equal_consecutive_subgroup_exists (arrangement : CircularArrangement) 
    (h1 : arrangement.length = 8)
    (h2 : countTypeA arrangement = 4) :
    ∃ i, countTypeA (takeCircular 4 i arrangement) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_consecutive_subgroup_exists_l2442_244281


namespace NUMINAMATH_CALUDE_projectile_height_time_l2442_244298

theorem projectile_height_time (t : ℝ) : 
  (∃ t₁ t₂ : ℝ, t₁ < t₂ ∧ -4.9 * t₁^2 + 30 * t₁ = 35 ∧ -4.9 * t₂^2 + 30 * t₂ = 35) → 
  (∀ t' : ℝ, -4.9 * t'^2 + 30 * t' = 35 → t' ≥ 10/7) ∧
  -4.9 * (10/7)^2 + 30 * (10/7) = 35 :=
sorry

end NUMINAMATH_CALUDE_projectile_height_time_l2442_244298


namespace NUMINAMATH_CALUDE_smallest_number_l2442_244267

theorem smallest_number (s : Set ℚ) (hs : s = {-2, 0, 3, 5}) : 
  ∃ m ∈ s, ∀ x ∈ s, m ≤ x ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l2442_244267


namespace NUMINAMATH_CALUDE_cherry_pies_profit_independence_l2442_244239

/-- Proves that the number of cherry pies does not affect the profit in Benny's pie sale scenario -/
theorem cherry_pies_profit_independence (num_pumpkin : ℕ) (cost_pumpkin : ℚ) (cost_cherry : ℚ) (sell_price : ℚ) (target_profit : ℚ) :
  num_pumpkin = 10 →
  cost_pumpkin = 3 →
  cost_cherry = 5 →
  sell_price = 5 →
  target_profit = 20 →
  ∀ num_cherry : ℕ,
    sell_price * (num_pumpkin + num_cherry) - (num_pumpkin * cost_pumpkin + num_cherry * cost_cherry) = target_profit :=
by sorry


end NUMINAMATH_CALUDE_cherry_pies_profit_independence_l2442_244239


namespace NUMINAMATH_CALUDE_billy_initial_dandelions_l2442_244258

/-- The number of dandelions Billy picked initially -/
def billy_initial : ℕ := sorry

/-- The number of dandelions George picked initially -/
def george_initial : ℕ := sorry

/-- The number of additional dandelions each person picked -/
def additional_picks : ℕ := 10

/-- The average number of dandelions picked -/
def average_picks : ℕ := 34

theorem billy_initial_dandelions :
  billy_initial = 36 ∧
  george_initial = billy_initial / 3 ∧
  (billy_initial + george_initial + 2 * additional_picks) / 2 = average_picks :=
sorry

end NUMINAMATH_CALUDE_billy_initial_dandelions_l2442_244258


namespace NUMINAMATH_CALUDE_function_composition_l2442_244234

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_composition (x : ℝ) (h : x ≥ -1) :
  f (Real.sqrt x - 1) = x - 2 * Real.sqrt x →
  f x = x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2442_244234


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2442_244283

/-- Given that x and y are inversely proportional, and x + y = 30 and x - y = 10, 
    prove that y = 200/7 when x = 7. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
  (h1 : x * y = k)  -- x and y are inversely proportional
  (h2 : x + y = 30) -- sum condition
  (h3 : x - y = 10) -- difference condition
  : (7 : ℝ) * (200 / 7) = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2442_244283


namespace NUMINAMATH_CALUDE_triangle_properties_l2442_244228

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.a * t.c * Real.sin t.B = t.b^2 - (t.a - t.c)^2) :
  Real.sin t.B = 4/5 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → x^2 / (x^2 + y^2) ≥ 2/5) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 / (x^2 + y^2) = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2442_244228
