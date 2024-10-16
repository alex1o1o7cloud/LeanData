import Mathlib

namespace NUMINAMATH_CALUDE_max_flow_increase_l413_41374

/-- Represents a water purification system with two sections of pipes -/
structure WaterSystem :=
  (pipes_AB : ℕ)
  (pipes_BC : ℕ)
  (flow_increase : ℝ)

/-- The theorem stating the maximum flow rate increase -/
theorem max_flow_increase (system : WaterSystem) 
  (h1 : system.pipes_AB = 10)
  (h2 : system.pipes_BC = 10)
  (h3 : system.flow_increase = 40) : 
  ∃ (max_increase : ℝ), max_increase = 200 ∧ 
  ∀ (new_system : WaterSystem), 
    new_system.pipes_AB + new_system.pipes_BC = system.pipes_AB + system.pipes_BC →
    new_system.flow_increase ≤ max_increase :=
sorry

end NUMINAMATH_CALUDE_max_flow_increase_l413_41374


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l413_41393

/-- Proves that adding 1.8 liters of pure alcohol to a 6-liter solution
    that is 35% alcohol will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.35
  let target_concentration : ℝ := 0.50
  let added_alcohol : ℝ := 1.8

  let final_volume : ℝ := initial_volume + added_alcohol
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_alcohol : ℝ := initial_alcohol + added_alcohol

  (final_alcohol / final_volume) = target_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_solution_proof_l413_41393


namespace NUMINAMATH_CALUDE_quadratic_opens_downwards_iff_a_negative_l413_41394

/-- A quadratic function of the form y = ax² - 2 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2

/-- The property that the graph of a quadratic function opens downwards -/
def opens_downwards (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f ((x + y) / 2) > (f x + f y) / 2

theorem quadratic_opens_downwards_iff_a_negative (a : ℝ) :
  opens_downwards (quadratic_function a) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_opens_downwards_iff_a_negative_l413_41394


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l413_41367

theorem rectangle_area_problem (square_area : Real) (rectangle_breadth : Real) :
  square_area = 784 →
  rectangle_breadth = 5 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := circle_radius / 4
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 35 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l413_41367


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l413_41391

/-- The number of penalty kicks in a soccer challenge --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: In a soccer team with 26 players, including 4 goalies, 
    where each player kicks against each goalie once, 
    the total number of penalty kicks is 100. --/
theorem soccer_penalty_kicks : 
  penalty_kicks 26 4 = 100 := by
sorry

#eval penalty_kicks 26 4

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l413_41391


namespace NUMINAMATH_CALUDE_georgia_carnation_cost_l413_41319

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 0.5

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia wants to send carnations to -/
def number_of_teachers : ℕ := 5

/-- The number of friends Georgia wants to buy carnations for -/
def number_of_friends : ℕ := 14

/-- The total cost of Georgia's carnation purchases -/
def total_cost : ℚ := 
  (number_of_teachers * dozen_carnation_cost) + 
  dozen_carnation_cost + 
  (2 * single_carnation_cost)

/-- Theorem stating that the total cost of Georgia's carnation purchases is $25.00 -/
theorem georgia_carnation_cost : total_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_georgia_carnation_cost_l413_41319


namespace NUMINAMATH_CALUDE_sum_divisible_by_twelve_l413_41377

theorem sum_divisible_by_twelve (b : ℤ) : 
  ∃ k : ℤ, 6 * b * (b + 1) = 12 * k := by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_twelve_l413_41377


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l413_41385

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / ((x - 3)^2)) →
  A = 1/4 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l413_41385


namespace NUMINAMATH_CALUDE_simplified_expression_equals_negative_sqrt_three_l413_41356

theorem simplified_expression_equals_negative_sqrt_three :
  let a := 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180)
  let b := 3
  1 - (a - b) / (a + 2 * b) / ((a^2 - b^2) / (a^2 + 4 * a * b + 4 * b^2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_negative_sqrt_three_l413_41356


namespace NUMINAMATH_CALUDE_trajectory_is_parallel_plane_l413_41320

-- Define the type for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the set of points P satisfying y = 3
def TrajectorySet : Set Point3D :=
  {p : Point3D | p.y = 3}

-- Define a plane parallel to xOz plane
def ParallelPlane (h : ℝ) : Set Point3D :=
  {p : Point3D | p.y = h}

-- Theorem statement
theorem trajectory_is_parallel_plane :
  ∃ h : ℝ, TrajectorySet = ParallelPlane h := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parallel_plane_l413_41320


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l413_41340

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The probability of a four-digit palindrome -/
def prob_digit_palindrome : ℚ := 1 / 100

/-- The probability of a four-letter palindrome with at least one 'X' -/
def prob_letter_palindrome : ℚ := 1 / 8784

/-- The probability of both four-digit and four-letter palindromes occurring -/
def prob_both_palindromes : ℚ := prob_digit_palindrome * prob_letter_palindrome

/-- The probability of at least one palindrome in a license plate -/
def prob_at_least_one_palindrome : ℚ := 
  prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes

theorem license_plate_palindrome_probability :
  prob_at_least_one_palindrome = 8883 / 878400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l413_41340


namespace NUMINAMATH_CALUDE_log_inequality_l413_41396

theorem log_inequality (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  Real.log ((4 * x - 5) / |x - 2|) / Real.log (x^2) ≥ 1/2 ↔ -1 + Real.sqrt 6 ≤ x ∧ x ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l413_41396


namespace NUMINAMATH_CALUDE_cone_base_circumference_l413_41321

/-- 
Given a right circular cone with volume 24π cubic centimeters and height 6 cm,
prove that the circumference of its base is 4√3π cm.
-/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 24 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l413_41321


namespace NUMINAMATH_CALUDE_new_person_weight_is_102_l413_41343

/-- The weight of a new person joining a group, given the initial group size,
    average weight increase, and weight of the person being replaced. -/
def new_person_weight (initial_group_size : ℕ) (avg_weight_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + initial_group_size * avg_weight_increase

/-- Theorem stating that the weight of the new person is 102 kg -/
theorem new_person_weight_is_102 :
  new_person_weight 6 4.5 75 = 102 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_102_l413_41343


namespace NUMINAMATH_CALUDE_ellipse_k_range_l413_41322

/-- An ellipse represented by the equation x² + ky² = 2 with foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ (x y : ℝ), x^2 + k * y^2 = 2
  foci_on_y_axis : True  -- This is a placeholder for the foci condition

/-- The range of k for an ellipse with the given properties is (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : 0 < e.k ∧ e.k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l413_41322


namespace NUMINAMATH_CALUDE_smallest_clock_equivalent_hour_l413_41376

theorem smallest_clock_equivalent_hour : ∃ (n : ℕ), n > 10 ∧ n % 12 = (n^2) % 12 ∧ ∀ (m : ℕ), m > 10 ∧ m < n → m % 12 ≠ (m^2) % 12 :=
  sorry

end NUMINAMATH_CALUDE_smallest_clock_equivalent_hour_l413_41376


namespace NUMINAMATH_CALUDE_price_reduction_5_price_reduction_20_no_2200_profit_l413_41342

/-- Represents the supermarket's sales model -/
structure SupermarketSales where
  initial_profit : ℕ
  initial_sales : ℕ
  price_reduction : ℕ
  sales_increase_rate : ℕ

/-- Calculates the new sales volume after a price reduction -/
def new_sales_volume (s : SupermarketSales) : ℕ :=
  s.initial_sales + s.price_reduction * s.sales_increase_rate

/-- Calculates the new profit per item after a price reduction -/
def new_profit_per_item (s : SupermarketSales) : ℤ :=
  s.initial_profit - s.price_reduction

/-- Calculates the total daily profit after a price reduction -/
def total_daily_profit (s : SupermarketSales) : ℤ :=
  (new_profit_per_item s) * (new_sales_volume s)

/-- Theorem: A price reduction of $5 results in 40 items sold and $1800 daily profit -/
theorem price_reduction_5 (s : SupermarketSales) 
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2)
  (h4 : s.price_reduction = 5) :
  new_sales_volume s = 40 ∧ total_daily_profit s = 1800 := by sorry

/-- Theorem: A price reduction of $20 results in $2100 daily profit -/
theorem price_reduction_20 (s : SupermarketSales)
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2)
  (h4 : s.price_reduction = 20) :
  total_daily_profit s = 2100 := by sorry

/-- Theorem: There is no price reduction that results in $2200 daily profit -/
theorem no_2200_profit (s : SupermarketSales)
  (h1 : s.initial_profit = 50)
  (h2 : s.initial_sales = 30)
  (h3 : s.sales_increase_rate = 2) :
  ∀ (x : ℕ), total_daily_profit { s with price_reduction := x } ≠ 2200 := by sorry

end NUMINAMATH_CALUDE_price_reduction_5_price_reduction_20_no_2200_profit_l413_41342


namespace NUMINAMATH_CALUDE_greatest_of_three_integers_l413_41359

theorem greatest_of_three_integers (a b c : ℤ) : 
  a + b + c = 21 → 
  c = max a (max b c) →
  c = 8 →
  max a (max b c) = 8 := by
sorry

end NUMINAMATH_CALUDE_greatest_of_three_integers_l413_41359


namespace NUMINAMATH_CALUDE_sharon_journey_distance_l413_41300

/-- Represents the journey from Sharon's house to her mother's house -/
structure Journey where
  distance : ℝ
  normalTime : ℝ
  reducedTime : ℝ
  speedReduction : ℝ

/-- The specific journey with given conditions -/
def sharonJourney : Journey where
  distance := 140  -- to be proved
  normalTime := 240
  reducedTime := 330
  speedReduction := 15

theorem sharon_journey_distance :
  ∀ j : Journey,
  j.normalTime = 240 ∧
  j.reducedTime = 330 ∧
  j.speedReduction = 15 ∧
  (j.distance / j.normalTime - j.speedReduction / 60) * (j.reducedTime - j.normalTime / 2) = j.distance / 2 →
  j.distance = 140 := by
  sorry

#check sharon_journey_distance

end NUMINAMATH_CALUDE_sharon_journey_distance_l413_41300


namespace NUMINAMATH_CALUDE_john_completion_time_l413_41379

/-- The time it takes for John to complete the task alone -/
def john_time : ℝ := 20

/-- The time it takes for Jane to complete the task alone -/
def jane_time : ℝ := 10

/-- The total time they worked together -/
def total_time : ℝ := 10

/-- The time Jane worked before stopping -/
def jane_work_time : ℝ := 5

theorem john_completion_time :
  (jane_work_time * (1 / john_time + 1 / jane_time) + (total_time - jane_work_time) * (1 / john_time) = 1) →
  john_time = 20 := by
sorry

end NUMINAMATH_CALUDE_john_completion_time_l413_41379


namespace NUMINAMATH_CALUDE_field_trip_fraction_l413_41327

theorem field_trip_fraction (b : ℚ) (g : ℚ) : 
  g = 2 * b →  -- There are twice as many girls as boys
  (2 / 3 * g + 3 / 5 * b) ≠ 0 → -- Total students on trip is not zero
  (2 / 3 * g) / (2 / 3 * g + 3 / 5 * b) = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_fraction_l413_41327


namespace NUMINAMATH_CALUDE_original_candle_length_l413_41354

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (original_length : ℝ) : 
  current_length = 48 →
  factor = 1.33 →
  original_length = current_length * factor →
  original_length = 63.84 := by
sorry

end NUMINAMATH_CALUDE_original_candle_length_l413_41354


namespace NUMINAMATH_CALUDE_cube_values_l413_41363

def f (n : ℕ) : ℤ := n^3 - 18*n^2 + 115*n - 391

def is_cube (x : ℤ) : Prop := ∃ y : ℤ, y^3 = x

theorem cube_values :
  {n : ℕ | is_cube (f n)} = {7, 11, 12, 25} := by sorry

end NUMINAMATH_CALUDE_cube_values_l413_41363


namespace NUMINAMATH_CALUDE_laundry_drying_time_l413_41360

theorem laundry_drying_time 
  (num_loads : ℕ) 
  (wash_time_per_load : ℕ) 
  (total_laundry_time : ℕ) 
  (h1 : num_loads = 2) 
  (h2 : wash_time_per_load = 45) 
  (h3 : total_laundry_time = 165) : 
  total_laundry_time - (num_loads * wash_time_per_load) = 75 := by
sorry

end NUMINAMATH_CALUDE_laundry_drying_time_l413_41360


namespace NUMINAMATH_CALUDE_problem_statement_l413_41318

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem statement
theorem problem_statement :
  (∃ m : ℝ, m > 0 ∧
    (Set.Icc (-2 : ℝ) 2 = { x | f (x + 1/2) ≤ 2*m + 1 }) ∧
    m = 3/2) ∧
  (∀ x y : ℝ, f x ≤ 2^y + 4/2^y + |2*x + 3|) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l413_41318


namespace NUMINAMATH_CALUDE_y_change_when_x_increases_l413_41334

/-- Regression equation: y = 3 - 5x -/
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

/-- Theorem: When x increases by 1, y decreases by 5 -/
theorem y_change_when_x_increases (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 5 := by
  sorry

end NUMINAMATH_CALUDE_y_change_when_x_increases_l413_41334


namespace NUMINAMATH_CALUDE_x_squared_plus_5xy_plus_y_squared_l413_41378

theorem x_squared_plus_5xy_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 4) 
  (h2 : x - y = 5) : 
  x^2 + 5*x*y + y^2 = 53 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_5xy_plus_y_squared_l413_41378


namespace NUMINAMATH_CALUDE_coeff_x3_is_30_l413_41347

/-- The coefficient of x^3 in the expansion of (2x-1)(1/x + x)^6 -/
def coeff_x3 : ℤ := 30

/-- The expression (2x-1)(1/x + x)^6 -/
def expression (x : ℚ) : ℚ := (2*x - 1) * (1/x + x)^6

theorem coeff_x3_is_30 : coeff_x3 = 30 := by sorry

end NUMINAMATH_CALUDE_coeff_x3_is_30_l413_41347


namespace NUMINAMATH_CALUDE_total_crayons_l413_41345

theorem total_crayons (billy_crayons jane_crayons : Float) 
  (h1 : billy_crayons = 62.0) 
  (h2 : jane_crayons = 52.0) : 
  billy_crayons + jane_crayons = 114.0 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l413_41345


namespace NUMINAMATH_CALUDE_win_sector_area_l413_41305

/-- Given a circular spinner with radius 12 cm and probability of winning 1/4,
    the area of the WIN sector is 36π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (total_area : ℝ) (win_area : ℝ) :
  r = 12 →
  p = 1 / 4 →
  total_area = π * r^2 →
  win_area = p * total_area →
  win_area = 36 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l413_41305


namespace NUMINAMATH_CALUDE_polygon_sides_count_polygon_sides_count_proof_l413_41384

theorem polygon_sides_count : ℕ → Prop :=
  fun n =>
    (n - 2) * 180 = 2 * 360 →
    n = 6

-- The proof is omitted
theorem polygon_sides_count_proof : ∃ n : ℕ, polygon_sides_count n :=
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_polygon_sides_count_proof_l413_41384


namespace NUMINAMATH_CALUDE_ten_points_chords_l413_41329

/-- The number of chords that can be drawn from n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: There are 45 different chords that can be drawn from 10 points on a circle -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_points_chords_l413_41329


namespace NUMINAMATH_CALUDE_bent_strips_odd_l413_41373

/-- Represents a paper strip covering two unit squares on the cube's surface -/
structure Strip where
  isBent : Bool

/-- Represents a 9x9x9 cube covered with 2x1 paper strips -/
structure Cube where
  strips : List Strip

/-- The number of unit squares on the surface of a 9x9x9 cube -/
def surfaceSquares : Nat := 6 * 9 * 9

/-- Theorem: The number of bent strips covering a 9x9x9 cube is odd -/
theorem bent_strips_odd (cube : Cube) (h1 : cube.strips.length * 2 = surfaceSquares) : 
  Odd (cube.strips.filter Strip.isBent).length := by
  sorry


end NUMINAMATH_CALUDE_bent_strips_odd_l413_41373


namespace NUMINAMATH_CALUDE_probability_different_colors_is_two_thirds_l413_41383

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def balls_drawn : ℕ := 2

def total_ways : ℕ := Nat.choose total_balls balls_drawn
def different_color_ways : ℕ := red_balls * white_balls

def probability_different_colors : ℚ := different_color_ways / total_ways

theorem probability_different_colors_is_two_thirds :
  probability_different_colors = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_two_thirds_l413_41383


namespace NUMINAMATH_CALUDE_round_trip_time_l413_41341

/-- Calculates the time for a round trip given boat speed, current speed, and distance -/
theorem round_trip_time (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) :
  boat_speed = 18 ∧ current_speed = 4 ∧ distance = 85.56 →
  (distance / (boat_speed - current_speed) + distance / (boat_speed + current_speed)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_time_l413_41341


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_range_l413_41365

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The condition that f has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- The main theorem stating that if f has three distinct zeros, then -2 < a < 2 -/
theorem three_zeros_implies_a_range (a : ℝ) :
  has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_implies_a_range_l413_41365


namespace NUMINAMATH_CALUDE_present_ages_l413_41386

/-- Represents the ages of Rahul, Deepak, and Karan -/
structure Ages where
  rahul : ℕ
  deepak : ℕ
  karan : ℕ

/-- The present age ratio between Rahul, Deepak, and Karan -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.deepak = 3 * ages.rahul ∧ 5 * ages.deepak = 3 * ages.karan

/-- In 8 years, the sum of Rahul's and Deepak's ages will equal Karan's age -/
def future_age_sum (ages : Ages) : Prop :=
  ages.rahul + ages.deepak + 16 = ages.karan

/-- Rahul's age after 6 years will be 26 years -/
def rahul_future_age (ages : Ages) : Prop :=
  ages.rahul + 6 = 26

/-- The theorem to be proved -/
theorem present_ages (ages : Ages) 
  (h1 : age_ratio ages) 
  (h2 : future_age_sum ages) 
  (h3 : rahul_future_age ages) : 
  ages.deepak = 15 ∧ ages.karan = 51 := by
  sorry

end NUMINAMATH_CALUDE_present_ages_l413_41386


namespace NUMINAMATH_CALUDE_product_of_powers_l413_41398

theorem product_of_powers : 2^4 * 3^2 * 5^2 * 7 * 11 = 277200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l413_41398


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l413_41390

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52) 
    (Finset.range 200)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l413_41390


namespace NUMINAMATH_CALUDE_square_perimeter_is_96_l413_41389

/-- A square ABCD with side lengths expressed in terms of x -/
structure Square (x : ℝ) where
  AB : ℝ := x + 16
  BC : ℝ := 3 * x
  is_square : AB = BC

/-- The perimeter of the square ABCD is 96 -/
theorem square_perimeter_is_96 (x : ℝ) (ABCD : Square x) : 
  4 * ABCD.AB = 96 := by
  sorry

#check square_perimeter_is_96

end NUMINAMATH_CALUDE_square_perimeter_is_96_l413_41389


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l413_41338

/-- The meeting point of two people, given their positions and the fraction of the distance between them -/
def meetingPoint (x₁ y₁ x₂ y₂ t : ℝ) : ℝ × ℝ :=
  ((1 - t) * x₁ + t * x₂, (1 - t) * y₁ + t * y₂)

/-- Theorem stating that the meeting point one-third of the way from (2, 3) to (8, -5) is (4, 1/3) -/
theorem meeting_point_theorem :
  let mark_pos : ℝ × ℝ := (2, 3)
  let sandy_pos : ℝ × ℝ := (8, -5)
  let t : ℝ := 1/3
  meetingPoint mark_pos.1 mark_pos.2 sandy_pos.1 sandy_pos.2 t = (4, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l413_41338


namespace NUMINAMATH_CALUDE_regular_octagon_perimeter_regular_octagon_perimeter_three_l413_41353

/-- The perimeter of a regular octagon with side length 3 is 24 -/
theorem regular_octagon_perimeter : ℕ → ℕ
  | side_length =>
    8 * side_length

theorem regular_octagon_perimeter_three : regular_octagon_perimeter 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_perimeter_regular_octagon_perimeter_three_l413_41353


namespace NUMINAMATH_CALUDE_roadwork_problem_l413_41306

/-- Roadwork problem statement -/
theorem roadwork_problem (total_length pitch_day3 : ℝ) (h1 h2 h3 : ℕ) : 
  total_length = 16 ∧ 
  pitch_day3 = 6 ∧ 
  h1 = 2 ∧ 
  h2 = 5 ∧ 
  h3 = 3 → 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < total_length ∧ 
    (2 * x - 1) > 0 ∧ 
    (2 * x - 1) < total_length ∧
    3 * x - 1 = total_length - (pitch_day3 / h2) / h3 ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roadwork_problem_l413_41306


namespace NUMINAMATH_CALUDE_rain_probability_l413_41339

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l413_41339


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l413_41364

theorem circle_circumference_increase (d : Real) : 
  let increase_in_diameter : Real := 2 * Real.pi
  let original_circumference : Real := Real.pi * d
  let new_circumference : Real := Real.pi * (d + increase_in_diameter)
  let Q : Real := new_circumference - original_circumference
  Q = 2 * Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l413_41364


namespace NUMINAMATH_CALUDE_equality_of_M_and_N_l413_41388

theorem equality_of_M_and_N (a b : ℝ) (hab : a * b = 1) : 
  (1 / (1 + a) + 1 / (1 + b)) = (a / (1 + a) + b / (1 + b)) := by
  sorry

#check equality_of_M_and_N

end NUMINAMATH_CALUDE_equality_of_M_and_N_l413_41388


namespace NUMINAMATH_CALUDE_kim_cousins_count_l413_41352

theorem kim_cousins_count (gum_per_cousin : ℕ) (total_gum : ℕ) (h1 : gum_per_cousin = 5) (h2 : total_gum = 20) :
  total_gum / gum_per_cousin = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_cousins_count_l413_41352


namespace NUMINAMATH_CALUDE_interview_selection_theorem_l413_41369

structure InterviewSelection where
  total_people : ℕ
  group3_size : ℕ
  group4_size : ℕ
  group5_size : ℕ
  interview_slots : ℕ

def stratified_sampling (s : InterviewSelection) : ℕ × ℕ × ℕ :=
  let total := s.group3_size + s.group4_size + s.group5_size
  ( (s.group3_size * s.interview_slots) / total,
    (s.group4_size * s.interview_slots) / total,
    (s.group5_size * s.interview_slots) / total )

def probability_at_least_one (group_size : ℕ) (selected : ℕ) : ℚ :=
  1 - (Nat.choose (group_size - 2) selected / Nat.choose group_size selected : ℚ)

theorem interview_selection_theorem (s : InterviewSelection) 
  (h1 : s.total_people = 36)
  (h2 : s.group3_size = 18)
  (h3 : s.group4_size = 12)
  (h4 : s.group5_size = 6)
  (h5 : s.interview_slots = 12) :
  let (g3, g4, g5) := stratified_sampling s
  g3 = 6 ∧ g4 = 4 ∧ g5 = 2 ∧ 
  probability_at_least_one s.group5_size g5 = 3/5 :=
sorry

end NUMINAMATH_CALUDE_interview_selection_theorem_l413_41369


namespace NUMINAMATH_CALUDE_line_intersects_cubic_at_two_points_l413_41358

/-- The function representing the cubic curve y = x^3 -/
def cubic_curve (x : ℝ) : ℝ := x^3

/-- The function representing the line y = ax + 16 -/
def line (a x : ℝ) : ℝ := a * x + 16

/-- Predicate to check if the line intersects the curve at exactly two distinct points -/
def intersects_at_two_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    cubic_curve x₁ = line a x₁ ∧
    cubic_curve x₂ = line a x₂ ∧
    ∀ x : ℝ, cubic_curve x = line a x → x = x₁ ∨ x = x₂

theorem line_intersects_cubic_at_two_points (a : ℝ) :
  intersects_at_two_points a → a = 12 := by sorry

end NUMINAMATH_CALUDE_line_intersects_cubic_at_two_points_l413_41358


namespace NUMINAMATH_CALUDE_trig_inequality_l413_41330

theorem trig_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : 0 < γ ∧ γ < Real.pi / 2)
  (h4 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_inequality_l413_41330


namespace NUMINAMATH_CALUDE_possible_values_of_a_l413_41350

theorem possible_values_of_a (a b c d : ℕ) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 3010)
  (h3 : a^2 - b^2 + c^2 - d^2 = 3010) :
  ∃! (s : Finset ℕ), s.card = 751 ∧ ∀ x, x ∈ s ↔ 
    ∃ (b' c' d' : ℕ), 
      a = x ∧
      x > b' ∧ b' > c' ∧ c' > d' ∧
      x + b' + c' + d' = 3010 ∧
      x^2 - b'^2 + c'^2 - d'^2 = 3010 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l413_41350


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l413_41324

/-- Calculates the loss percentage for a watch sale given specific conditions. -/
def loss_percentage (cost_price selling_price increased_price : ℚ) : ℚ :=
  let gain_percentage : ℚ := 2 / 100
  let price_difference : ℚ := increased_price - selling_price
  let loss : ℚ := cost_price - selling_price
  (loss / cost_price) * 100

/-- Theorem stating that the loss percentage is 10% under given conditions. -/
theorem watch_loss_percentage : 
  let cost_price : ℚ := 1166.67
  let selling_price : ℚ := cost_price - 116.67
  let increased_price : ℚ := selling_price + 140
  loss_percentage cost_price selling_price increased_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l413_41324


namespace NUMINAMATH_CALUDE_sheep_value_is_16_l413_41381

/-- Represents the agreement between Kuba and the shepherd -/
structure Agreement where
  fullYearCoins : ℕ
  fullYearSheep : ℕ
  monthsWorked : ℕ
  coinsReceived : ℕ
  sheepReceived : ℕ

/-- Calculates the value of a sheep in gold coins based on the agreement -/
def sheepValue (a : Agreement) : ℕ :=
  let monthlyRate := a.fullYearCoins / 12
  let expectedCoins := monthlyRate * a.monthsWorked
  expectedCoins - a.coinsReceived

/-- The main theorem stating that the value of a sheep is 16 gold coins -/
theorem sheep_value_is_16 (a : Agreement) 
  (h1 : a.fullYearCoins = 20)
  (h2 : a.fullYearSheep = 1)
  (h3 : a.monthsWorked = 7)
  (h4 : a.coinsReceived = 5)
  (h5 : a.sheepReceived = 1) :
  sheepValue a = 16 := by
  sorry

#eval sheepValue { fullYearCoins := 20, fullYearSheep := 1, monthsWorked := 7, coinsReceived := 5, sheepReceived := 1 }

end NUMINAMATH_CALUDE_sheep_value_is_16_l413_41381


namespace NUMINAMATH_CALUDE_sum_product_nonpositive_l413_41308

theorem sum_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_nonpositive_l413_41308


namespace NUMINAMATH_CALUDE_alex_total_marbles_l413_41333

/-- The number of marbles each person has -/
structure MarbleCount where
  lorin_black : ℕ
  jimmy_yellow : ℕ
  alex_black : ℕ
  alex_yellow : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.lorin_black = 4 ∧
  m.jimmy_yellow = 22 ∧
  m.alex_black = 2 * m.lorin_black ∧
  m.alex_yellow = m.jimmy_yellow / 2

/-- The theorem stating that Alex has 19 marbles in total -/
theorem alex_total_marbles (m : MarbleCount) 
  (h : marble_problem m) : m.alex_black + m.alex_yellow = 19 := by
  sorry


end NUMINAMATH_CALUDE_alex_total_marbles_l413_41333


namespace NUMINAMATH_CALUDE_rectangular_field_area_l413_41351

/-- Calculates the area of a rectangular field given its perimeter and width. -/
theorem rectangular_field_area
  (perimeter : ℝ) (width : ℝ)
  (h_perimeter : perimeter = 30)
  (h_width : width = 5) :
  width * (perimeter / 2 - width) = 50 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l413_41351


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l413_41370

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_series : ℕ := 
  let terms := List.range 16 |>.map (fun i => 6 * i + 6)
  terms.map (fun n => last_two_digits ((factorial n) + 1)) |>.sum

theorem last_two_digits_of_sum : last_two_digits sum_series = 36 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_l413_41370


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l413_41303

theorem complete_square_quadratic (x : ℝ) : 
  2 * x^2 + 3 * x + 1 = 0 ↔ 2 * (x + 3/4)^2 - 1/8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l413_41303


namespace NUMINAMATH_CALUDE_monopoly_wins_ratio_l413_41399

/-- 
Proves that given the conditions of the Monopoly game wins, 
the ratio of Susan's wins to Betsy's wins is 3:1
-/
theorem monopoly_wins_ratio :
  ∀ (betsy helen susan : ℕ),
  betsy = 5 →
  helen = 2 * betsy →
  betsy + helen + susan = 30 →
  susan / betsy = 3 := by
sorry

end NUMINAMATH_CALUDE_monopoly_wins_ratio_l413_41399


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l413_41336

theorem complex_magnitude_equality (t : ℝ) (h1 : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 2 * Real.sqrt 17 → t = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l413_41336


namespace NUMINAMATH_CALUDE_first_plot_germination_rate_l413_41309

def seeds_first_plot : ℕ := 300
def seeds_second_plot : ℕ := 200
def germination_rate_second_plot : ℚ := 35 / 100
def overall_germination_rate : ℚ := 28999999999999996 / 100000000000000000

theorem first_plot_germination_rate :
  let total_seeds := seeds_first_plot + seeds_second_plot
  let germinated_second_plot := (seeds_second_plot : ℚ) * germination_rate_second_plot
  let total_germinated := (total_seeds : ℚ) * overall_germination_rate
  let germinated_first_plot := total_germinated - germinated_second_plot
  germinated_first_plot / seeds_first_plot = 1 / 4 := by
    sorry

end NUMINAMATH_CALUDE_first_plot_germination_rate_l413_41309


namespace NUMINAMATH_CALUDE_smallest_k_for_periodic_sum_l413_41395

/-- Represents a rational number with a periodic decimal representation -/
structure PeriodicDecimal where
  numerator : ℤ
  period : ℕ+

/-- Returns true if the given natural number is the minimal period of the decimal representation -/
def is_minimal_period (r : ℚ) (p : ℕ+) : Prop :=
  ∃ (m : ℤ), r = m / (10^p.val - 1) ∧ 
  ∀ (q : ℕ+), q < p → ¬∃ (n : ℤ), r = n / (10^q.val - 1)

theorem smallest_k_for_periodic_sum (a b : PeriodicDecimal) : 
  (is_minimal_period (a.numerator / (10^30 - 1)) 30) →
  (is_minimal_period (b.numerator / (10^30 - 1)) 30) →
  (is_minimal_period ((a.numerator - b.numerator) / (10^30 - 1)) 15) →
  (∀ k : ℕ, k < 6 → ¬is_minimal_period ((a.numerator + k * b.numerator) / (10^30 - 1)) 15) →
  is_minimal_period ((a.numerator + 6 * b.numerator) / (10^30 - 1)) 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_periodic_sum_l413_41395


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l413_41348

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l413_41348


namespace NUMINAMATH_CALUDE_divisor_log_sum_l413_41368

theorem divisor_log_sum (n : ℕ) : (n * (n + 1)^2) / 2 = 1080 ↔ n = 12 := by sorry

end NUMINAMATH_CALUDE_divisor_log_sum_l413_41368


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l413_41357

theorem integral_sqrt_minus_2x (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - (x - 1)^2)) →
  (∀ x, g x = 2 * x) →
  ∫ x in (0 : ℝ)..1, (f x - g x) = π / 4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l413_41357


namespace NUMINAMATH_CALUDE_ellipse_properties_l413_41380

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

-- Define the conditions
def is_valid_ellipse (e : Ellipse) : Prop :=
  ∃ c : ℝ, 
    e.a + c = Real.sqrt 2 + 1 ∧
    e.a = Real.sqrt 2 * c ∧
    e.a^2 = e.b^2 + c^2

-- Define the standard equation
def standard_equation (e : Ellipse) : Prop :=
  e.a^2 = 2 ∧ e.b^2 = 1

-- Define the line passing through the left focus
def line_through_focus (k : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, y t = k * (x t + 1)

-- Define the condition for the midpoint
def midpoint_on_line (x y : ℝ → ℝ) : Prop :=
  ∃ t₁ t₂, x ((t₁ + t₂)/2) + y ((t₁ + t₂)/2) = 0

-- The main theorem
theorem ellipse_properties (e : Ellipse) (h : is_valid_ellipse e) :
  standard_equation e ∧
  (∀ k x y, line_through_focus k x y → midpoint_on_line x y →
    (k = 0 ∨ k = 1/2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l413_41380


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l413_41304

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 255)
  (n_eq_8x : n = 8 * x)
  (n_eq_y_minus_11 : n = y - 11)
  (n_eq_z_plus_13 : n = z + 13) :
  x * y * z = 209805 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l413_41304


namespace NUMINAMATH_CALUDE_inverse_proposition_l413_41314

theorem inverse_proposition :
  (∀ a : ℝ, a > 0 → a > 1) →
  (∀ a : ℝ, a > 1 → a > 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l413_41314


namespace NUMINAMATH_CALUDE_lobster_poorly_chosen_l413_41302

/-- Represents a character in "Alice's Adventures in Wonderland" --/
inductive Character : Type
| Lobster : Character
| Gryphon : Character
| MockTurtle : Character

/-- Represents the role of a character in the story --/
inductive CharacterRole : Type
| ActiveCharacter : CharacterRole
| PoemCharacter : CharacterRole

/-- Determines if a character is poorly chosen for a problem --/
def isPoorlyChosen (c : Character) (role : CharacterRole) : Prop :=
  c = Character.Lobster ∧ role = CharacterRole.PoemCharacter

/-- The main theorem stating why the Lobster is poorly chosen --/
theorem lobster_poorly_chosen (c : Character) (role : CharacterRole) :
  isPoorlyChosen c role →
  ∃ (explanation : String),
    explanation = "The character only appears in a poem within the story and not as an active character" :=
by
  sorry

#check lobster_poorly_chosen

end NUMINAMATH_CALUDE_lobster_poorly_chosen_l413_41302


namespace NUMINAMATH_CALUDE_water_tank_full_time_l413_41392

/-- Represents the state of a water tank system with three pipes -/
structure WaterTankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the net water change after one cycle -/
def net_change_per_cycle (system : WaterTankSystem) : ℤ :=
  system.pipeA_rate + system.pipeB_rate - system.pipeC_rate

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : WaterTankSystem) : ℕ :=
  (system.capacity / (net_change_per_cycle system).natAbs) * 3

/-- Theorem stating that the given water tank system will be full after 48 minutes -/
theorem water_tank_full_time (system : WaterTankSystem) 
  (h1 : system.capacity = 800)
  (h2 : system.pipeA_rate = 40)
  (h3 : system.pipeB_rate = 30)
  (h4 : system.pipeC_rate = 20) : 
  time_to_fill system = 48 := by
  sorry

#eval time_to_fill { capacity := 800, pipeA_rate := 40, pipeB_rate := 30, pipeC_rate := 20 }

end NUMINAMATH_CALUDE_water_tank_full_time_l413_41392


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l413_41316

theorem smallest_number_in_sequence (x : ℝ) : 
  let second := 4 * x
  let third := 2 * x
  (x + second + third) / 3 = 77 →
  x = 33 := by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l413_41316


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l413_41313

theorem consecutive_integers_cube_sum (a b c d : ℕ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 9340) →
  (a^3 + b^3 + c^3 + d^3 = 457064) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l413_41313


namespace NUMINAMATH_CALUDE_triangle_area_sides_circumradius_l413_41328

/-- The area of a triangle in terms of its sides and circumradius -/
theorem triangle_area_sides_circumradius (a b c R S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = (a * b * c) / (4 * S))
  (h_area : S > 0) :
  S = (a * b * c) / (4 * R) := by
sorry


end NUMINAMATH_CALUDE_triangle_area_sides_circumradius_l413_41328


namespace NUMINAMATH_CALUDE_constant_expression_l413_41315

theorem constant_expression (x y k : ℝ) 
  (eq1 : x + 2*y = k + 2) 
  (eq2 : 2*x - 3*y = 3*k - 1) : 
  x + 9*y = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_l413_41315


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l413_41375

theorem sqrt_2_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a / b : ℚ)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l413_41375


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l413_41344

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (7 * a) % 72 = 1 ∧ (13 * b) % 72 = 1 →
  (3 * a + 9 * b) % 72 = 18 := by
sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l413_41344


namespace NUMINAMATH_CALUDE_gcd_pow_minus_one_l413_41317

theorem gcd_pow_minus_one (a n m : ℕ) (ha : a > 0) :
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd n m) - 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_pow_minus_one_l413_41317


namespace NUMINAMATH_CALUDE_jump_rope_time_ratio_l413_41323

/-- Given information about jump rope times for Cindy, Betsy, and Tina, 
    prove that the ratio of Tina's time to Betsy's time is 3. -/
theorem jump_rope_time_ratio :
  ∀ (cindy_time betsy_time tina_time : ℕ),
    cindy_time = 12 →
    betsy_time = cindy_time / 2 →
    tina_time = cindy_time + 6 →
    tina_time / betsy_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_time_ratio_l413_41323


namespace NUMINAMATH_CALUDE_jace_neighbor_payment_l413_41397

/-- Proves that Jace gave 0 cents to his neighbor -/
theorem jace_neighbor_payment (earned : ℕ) (debt : ℕ) (remaining : ℕ) : 
  earned = 1000 → debt = 358 → remaining = 642 → (earned - debt - remaining) * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_jace_neighbor_payment_l413_41397


namespace NUMINAMATH_CALUDE_exercise_books_quantity_l413_41312

/-- Given a ratio of items and the quantity of one item, calculate the quantity of another item in the ratio. -/
def calculate_quantity (ratio_a : ℕ) (ratio_b : ℕ) (quantity_a : ℕ) : ℕ :=
  (quantity_a * ratio_b) / ratio_a

/-- Prove that given 140 pencils and a ratio of 14 : 4 : 3 for pencils : pens : exercise books, 
    the number of exercise books is 30. -/
theorem exercise_books_quantity (pencils : ℕ) (ratio_pencils ratio_pens ratio_books : ℕ) 
    (h1 : pencils = 140)
    (h2 : ratio_pencils = 14)
    (h3 : ratio_pens = 4)
    (h4 : ratio_books = 3) :
  calculate_quantity ratio_pencils ratio_books pencils = 30 := by
  sorry

#eval calculate_quantity 14 3 140

end NUMINAMATH_CALUDE_exercise_books_quantity_l413_41312


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l413_41349

/-- Proves that the ratio of boat speed to stream speed is 3:1 given the time relation -/
theorem boat_stream_speed_ratio 
  (D : ℝ) -- Distance rowed
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (h_positive : D > 0 ∧ B > 0 ∧ S > 0) -- Positive distances and speeds
  (h_time_ratio : D / (B - S) = 2 * (D / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l413_41349


namespace NUMINAMATH_CALUDE_water_in_mixture_l413_41311

theorem water_in_mixture (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (b * x) / (a + b) = x * (b / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_water_in_mixture_l413_41311


namespace NUMINAMATH_CALUDE_johns_remaining_money_l413_41346

/-- Given John's initial money and his purchases, calculate the remaining amount --/
theorem johns_remaining_money (initial : ℕ) (roast : ℕ) (vegetables : ℕ) :
  initial = 100 ∧ roast = 17 ∧ vegetables = 11 →
  initial - (roast + vegetables) = 72 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l413_41346


namespace NUMINAMATH_CALUDE_jury_duty_ratio_l413_41326

theorem jury_duty_ratio (jury_selection : ℕ) (trial_duration : ℕ) (jury_deliberation : ℕ) (total_days : ℕ) :
  jury_selection = 2 →
  jury_deliberation = 6 →
  total_days = 19 →
  total_days = jury_selection + trial_duration + jury_deliberation →
  (trial_duration : ℚ) / (jury_selection : ℚ) = 11 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jury_duty_ratio_l413_41326


namespace NUMINAMATH_CALUDE_function_always_positive_l413_41371

theorem function_always_positive (k : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (k - 2) * x + 2 * |k| - 1 > 0) ↔ k > 5/4 := by
  sorry

end NUMINAMATH_CALUDE_function_always_positive_l413_41371


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l413_41307

theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l413_41307


namespace NUMINAMATH_CALUDE_jill_age_l413_41301

/-- Represents the ages of individuals in the problem -/
structure Ages where
  gina : ℕ
  helen : ℕ
  ian : ℕ
  jill : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.gina + 4 = ages.helen ∧
  ages.helen = ages.ian + 5 ∧
  ages.jill = ages.ian + 2 ∧
  ages.gina = 18

/-- The theorem stating Jill's age -/
theorem jill_age (ages : Ages) (h : problem_conditions ages) : ages.jill = 19 := by
  sorry

#check jill_age

end NUMINAMATH_CALUDE_jill_age_l413_41301


namespace NUMINAMATH_CALUDE_train_meeting_distance_l413_41310

/-- Proves that when two trains starting 200 miles apart and traveling towards each other
    at 20 miles per hour each meet, one train will have traveled 100 miles. -/
theorem train_meeting_distance (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) 
  (h1 : total_distance = 200)
  (h2 : speed_a = 20)
  (h3 : speed_b = 20) :
  speed_a * (total_distance / (speed_a + speed_b)) = 100 :=
by sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l413_41310


namespace NUMINAMATH_CALUDE_factor_81_minus_36x4_l413_41335

theorem factor_81_minus_36x4 (x : ℝ) : 
  81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_36x4_l413_41335


namespace NUMINAMATH_CALUDE_max_discount_l413_41366

/-- Given a product with a marked price and markup percentage, calculate the maximum discount --/
theorem max_discount (marked_price : ℝ) (markup_percent : ℝ) (min_markup_percent : ℝ) : 
  marked_price = 360 ∧ 
  markup_percent = 0.8 ∧ 
  min_markup_percent = 0.2 →
  marked_price - (marked_price / (1 + markup_percent) * (1 + min_markup_percent)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_discount_l413_41366


namespace NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l413_41362

-- Define the quadratic function
def f (x m : ℝ) : ℝ := x^2 - x + m

-- Define the condition for the vertex being on the x-axis
def vertex_on_x_axis (m : ℝ) : Prop :=
  let x₀ := 1/2  -- x-coordinate of the vertex
  f x₀ m = 0

-- Theorem statement
theorem quadratic_vertex_on_x_axis (m : ℝ) :
  vertex_on_x_axis m → m = 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_on_x_axis_l413_41362


namespace NUMINAMATH_CALUDE_expand_and_simplify_l413_41387

theorem expand_and_simplify (x : ℝ) : (x - 1) * (x + 3) - x * (x - 2) = 4 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l413_41387


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l413_41337

theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l413_41337


namespace NUMINAMATH_CALUDE_avery_build_time_l413_41325

theorem avery_build_time (tom_time : ℝ) (joint_work_time : ℝ) (tom_remaining_time : ℝ)
  (h1 : tom_time = 2.5)
  (h2 : joint_work_time = 1)
  (h3 : tom_remaining_time = 2/3) :
  ∃ avery_time : ℝ, 
    (1 / avery_time + 1 / tom_time) * joint_work_time + 
    (1 / tom_time) * tom_remaining_time = 1 ∧ 
    avery_time = 3 := by
sorry

end NUMINAMATH_CALUDE_avery_build_time_l413_41325


namespace NUMINAMATH_CALUDE_revenue_decrease_l413_41382

theorem revenue_decrease (tax_reduction : Real) (consumption_increase : Real) :
  tax_reduction = 0.24 →
  consumption_increase = 0.12 →
  let new_tax_rate := 1 - tax_reduction
  let new_consumption := 1 + consumption_increase
  let revenue_change := 1 - (new_tax_rate * new_consumption)
  revenue_change = 0.1488 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l413_41382


namespace NUMINAMATH_CALUDE_simplify_expression_l413_41355

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l413_41355


namespace NUMINAMATH_CALUDE_log_stack_sum_l413_41332

theorem log_stack_sum (a l n : ℕ) (h1 : a = 5) (h2 : l = 15) (h3 : n = 11) :
  n * (a + l) / 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l413_41332


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l413_41361

/-- Given a parallelogram with area 242 sq m and base 11 m, prove its altitude to base ratio is 2 -/
theorem parallelogram_altitude_base_ratio : 
  ∀ (area base altitude : ℝ), 
  area = 242 ∧ base = 11 ∧ area = base * altitude → 
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l413_41361


namespace NUMINAMATH_CALUDE_train_speed_l413_41372

/-- The speed of a train passing a platform -/
theorem train_speed (train_length platform_length : ℝ) (time : ℝ) 
  (h1 : train_length = 50)
  (h2 : platform_length = 100)
  (h3 : time = 10) :
  (train_length + platform_length) / time = 15 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l413_41372


namespace NUMINAMATH_CALUDE_complement_of_union_l413_41331

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union : 
  (A ∪ B)ᶜ = {2, 4} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_l413_41331
