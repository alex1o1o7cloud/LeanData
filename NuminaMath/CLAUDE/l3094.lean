import Mathlib

namespace NUMINAMATH_CALUDE_initial_marbles_equals_sum_l3094_309426

/-- The number of marbles Connie initially had -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles equals the sum of marbles given away and marbles left -/
theorem initial_marbles_equals_sum : initial_marbles = marbles_given + marbles_left := by sorry

end NUMINAMATH_CALUDE_initial_marbles_equals_sum_l3094_309426


namespace NUMINAMATH_CALUDE_jimmy_wins_l3094_309423

/-- Represents a fan with four blades -/
structure Fan :=
  (rotation_speed : ℝ)
  (blade_count : ℕ)

/-- Represents a bullet trajectory -/
structure Trajectory :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Checks if a trajectory intersects a blade at a given position and time -/
def intersects_blade (f : Fan) (t : Trajectory) (position : ℕ) (time : ℝ) : Prop :=
  sorry

/-- The main theorem stating that there exists a trajectory that intersects all blades -/
theorem jimmy_wins (f : Fan) : 
  f.rotation_speed = 50 ∧ f.blade_count = 4 → 
  ∃ t : Trajectory, ∀ p : ℕ, p < f.blade_count → 
    ∃ time : ℝ, intersects_blade f t p time :=
sorry

end NUMINAMATH_CALUDE_jimmy_wins_l3094_309423


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l3094_309479

theorem inequality_holds_for_all_reals (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l3094_309479


namespace NUMINAMATH_CALUDE_complex_square_of_one_plus_i_l3094_309496

theorem complex_square_of_one_plus_i :
  ∀ z : ℂ, (z.re = 1 ∧ z.im = 1) → z^2 = 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_of_one_plus_i_l3094_309496


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3094_309408

theorem geometric_sequence_ratio_sum (m : ℝ) (a₂ a₃ b₂ b₃ : ℝ) :
  m ≠ 0 →
  (∃ x : ℝ, x ≠ 1 ∧ a₂ = m * x ∧ a₃ = m * x^2) →
  (∃ y : ℝ, y ≠ 1 ∧ b₂ = m * y ∧ b₃ = m * y^2) →
  (∀ x y : ℝ, a₂ = m * x ∧ a₃ = m * x^2 ∧ b₂ = m * y ∧ b₃ = m * y^2 → x ≠ y) →
  a₃ - b₃ = 3 * (a₂ - b₂) →
  ∃ x y : ℝ, (a₂ = m * x ∧ a₃ = m * x^2 ∧ b₂ = m * y ∧ b₃ = m * y^2) ∧ x + y = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3094_309408


namespace NUMINAMATH_CALUDE_solution_difference_l3094_309437

theorem solution_difference (p q : ℝ) : 
  ((p - 5) * (p + 5) = 17 * p - 85) →
  ((q - 5) * (q + 5) = 17 * q - 85) →
  p ≠ q →
  p > q →
  p - q = 7 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3094_309437


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3094_309450

/-- The interval of segmentation for systematic sampling -/
def interval_of_segmentation (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The interval of segmentation for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  interval_of_segmentation 1200 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3094_309450


namespace NUMINAMATH_CALUDE_decimal_132_to_binary_l3094_309428

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

-- Theorem statement
theorem decimal_132_to_binary :
  decimalToBinary 132 = [true, false, false, false, false, true, false, false] := by
  sorry

#eval decimalToBinary 132

end NUMINAMATH_CALUDE_decimal_132_to_binary_l3094_309428


namespace NUMINAMATH_CALUDE_painter_completion_time_l3094_309409

-- Define the start time
def start_time : Nat := 9

-- Define the quarter completion time
def quarter_time : Nat := 12

-- Define the time taken for quarter completion
def quarter_duration : Nat := quarter_time - start_time

-- Define the total duration
def total_duration : Nat := 4 * quarter_duration

-- Define the completion time
def completion_time : Nat := start_time + total_duration

-- Theorem statement
theorem painter_completion_time :
  start_time = 9 →
  quarter_time = 12 →
  completion_time = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_painter_completion_time_l3094_309409


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3094_309404

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 5 * x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 2000 / 7 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 3 * y₂ + 5 * y₃ = 100 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 2000 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3094_309404


namespace NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l3094_309405

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c ∧ t.b * t.c = 1

-- Define the additional condition for part II
def satisfiesAdditionalCondition (t : Triangle) : Prop :=
  4 * Real.cos t.B * Real.cos t.C - 1 = 0

-- Theorem for part I
theorem area_of_triangle (t : Triangle) (h : satisfiesConditions t) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 4 := by
  sorry

-- Theorem for part II
theorem perimeter_of_triangle (t : Triangle) 
  (h1 : satisfiesConditions t) (h2 : satisfiesAdditionalCondition t) :
  t.a + t.b + t.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l3094_309405


namespace NUMINAMATH_CALUDE_salary_increase_after_five_years_l3094_309419

theorem salary_increase_after_five_years (annual_raise : ℝ) (num_years : ℕ) : 
  annual_raise = 0.12 → num_years = 5 → (1 + annual_raise) ^ num_years > 1.76 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_five_years_l3094_309419


namespace NUMINAMATH_CALUDE_angle_terminal_side_l3094_309461

theorem angle_terminal_side (θ : Real) (a : Real) : 
  (2 * Real.sin (π / 8) ^ 2 - 1, a) ∈ Set.range (λ t : Real × Real => (t.1 * Real.cos θ - t.2 * Real.sin θ, t.1 * Real.sin θ + t.2 * Real.cos θ)) ∧ 
  Real.sin θ = 2 * Real.sqrt 3 * Real.sin (13 * π / 12) * Real.cos (π / 12) →
  a = - Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l3094_309461


namespace NUMINAMATH_CALUDE_ludwig_daily_salary_l3094_309463

def weekly_salary : ℚ := 55
def full_days : ℕ := 4
def half_days : ℕ := 3

theorem ludwig_daily_salary : 
  ∃ (daily_salary : ℚ), 
    (daily_salary * full_days + daily_salary * half_days / 2 = weekly_salary) ∧
    daily_salary = 10 := by
sorry

end NUMINAMATH_CALUDE_ludwig_daily_salary_l3094_309463


namespace NUMINAMATH_CALUDE_number_puzzle_l3094_309441

theorem number_puzzle : ∃! x : ℝ, x - 18 = 3 * (86 - x) := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l3094_309441


namespace NUMINAMATH_CALUDE_kelly_travel_days_l3094_309476

/-- Kelly's vacation details -/
structure VacationSchedule where
  total_days : ℕ
  initial_travel : ℕ
  grandparents : ℕ
  brother : ℕ
  to_sister_travel : ℕ
  sister : ℕ
  final_travel : ℕ

/-- The number of days Kelly spent traveling between her Grandparents' house and her brother's house -/
def days_between_grandparents_and_brother (schedule : VacationSchedule) : ℕ :=
  schedule.total_days - (schedule.initial_travel + schedule.grandparents + 
    schedule.brother + schedule.to_sister_travel + schedule.sister + schedule.final_travel)

/-- Theorem stating that Kelly spent 1 day traveling between her Grandparents' and brother's houses -/
theorem kelly_travel_days (schedule : VacationSchedule) 
  (h1 : schedule.total_days = 21)  -- Three weeks
  (h2 : schedule.initial_travel = 1)
  (h3 : schedule.grandparents = 5)
  (h4 : schedule.brother = 5)
  (h5 : schedule.to_sister_travel = 2)
  (h6 : schedule.sister = 5)
  (h7 : schedule.final_travel = 2) :
  days_between_grandparents_and_brother schedule = 1 := by
  sorry

end NUMINAMATH_CALUDE_kelly_travel_days_l3094_309476


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3094_309494

theorem quadratic_one_root (m : ℝ) (h : m > 0) :
  (∃! x : ℝ, x^2 + 4*m*x + m = 0) ↔ m = 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3094_309494


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3094_309470

theorem intersection_of_sets : 
  let A : Set Int := {-1, 0, 1}
  let B : Set Int := {0, 1, 2, 3}
  A ∩ B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3094_309470


namespace NUMINAMATH_CALUDE_set_membership_implies_a_values_l3094_309442

def A (a : ℝ) : Set ℝ := {2, 1-a, a^2-a+2}

theorem set_membership_implies_a_values (a : ℝ) :
  4 ∈ A a → a = -3 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_a_values_l3094_309442


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3094_309440

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4*x} = {0, 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3094_309440


namespace NUMINAMATH_CALUDE_circle_problem_l3094_309474

theorem circle_problem (P : ℝ × ℝ) (S : ℝ × ℝ) (k : ℝ) :
  P = (5, 12) →
  S = (0, k) →
  (∃ (O : ℝ × ℝ), O = (0, 0) ∧
    ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧
      (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₁^2 ∧
      (S.1 - O.1)^2 + (S.2 - O.2)^2 = r₂^2 ∧
      r₁ - r₂ = 4) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_circle_problem_l3094_309474


namespace NUMINAMATH_CALUDE_fuel_station_cost_fuel_station_cost_example_l3094_309416

/-- Calculates the total cost of filling up vehicles at a fuel station -/
theorem fuel_station_cost (service_cost : ℝ) (fuel_cost : ℝ) (minivan_count : ℕ) (truck_count : ℕ) 
  (minivan_tank : ℝ) (truck_tank_increase : ℝ) : ℝ :=
  let truck_tank := minivan_tank * (1 + truck_tank_increase)
  let minivan_fuel_cost := minivan_count * minivan_tank * fuel_cost
  let truck_fuel_cost := truck_count * truck_tank * fuel_cost
  let total_service_cost := (minivan_count + truck_count) * service_cost
  minivan_fuel_cost + truck_fuel_cost + total_service_cost

/-- Proves that the total cost for filling up 3 mini-vans and 2 trucks is $347.20 -/
theorem fuel_station_cost_example : 
  fuel_station_cost 2.10 0.70 3 2 65 1.20 = 347.20 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_cost_fuel_station_cost_example_l3094_309416


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l3094_309499

/-- Given a circle centered at F(4,0) with radius 2, and points A(-4,0) and B on the circle,
    the point P is defined as the intersection of the perpendicular bisector of AB and line BF.
    This theorem states that the trajectory of P as B moves along the circle
    is a hyperbola with equation x^2 - y^2/15 = 1 (x ≠ 0). -/
theorem trajectory_of_point_P (A B P F : ℝ × ℝ) :
  A = (-4, 0) →
  F = (4, 0) →
  (B.1 - 4)^2 + B.2^2 = 4 →
  (∀ M : ℝ × ℝ, (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 →
                 (M.1 - P.1) * (B.2 - F.2) = (M.2 - P.2) * (B.1 - F.1)) →
  (P.1 - F.1) * (B.2 - F.2) = (P.2 - F.2) * (B.1 - F.1) →
  P.1 ≠ 0 →
  P.1^2 - P.2^2 / 15 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l3094_309499


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l3094_309444

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    x ≥ 13^6 * 5^7 ∧
    (∀ (x' : ℕ+) (a' b' c' d' : ℕ), 5 * x'^7 = 13 * y^11 → x' = a'^c' * b'^d' → x' ≥ x) ∧
    a + b + c + d = 31 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l3094_309444


namespace NUMINAMATH_CALUDE_consecutive_product_3024_l3094_309425

theorem consecutive_product_3024 :
  ∀ n : ℕ, n > 0 →
  (n * (n + 1) * (n + 2) * (n + 3) = 3024) ↔ n = 6 := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_3024_l3094_309425


namespace NUMINAMATH_CALUDE_min_value_of_cubic_function_l3094_309431

/-- Given a function f(x) = 2x^3 - 6x^2 + a, where a is a constant,
    prove that if the maximum value of f(x) on the interval [-2, 2] is 3,
    then the minimum value of f(x) on [-2, 2] is -37. -/
theorem min_value_of_cubic_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^3 - 6 * x^2 + a
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, f x = -37) ∧ (∀ x ∈ Set.Icc (-2) 2, f x ≥ -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_cubic_function_l3094_309431


namespace NUMINAMATH_CALUDE_S_is_infinite_l3094_309491

-- Define the set of points that satisfy the conditions
def S : Set (ℚ × ℚ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 5}

-- Theorem: The set S is infinite
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l3094_309491


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l3094_309452

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.exp x + x

def line (x : ℝ) : ℝ := 3 * x - 1

theorem min_distance_curve_line :
  ∃ (d : ℝ), d = (3 * Real.sqrt 10) / 10 ∧
  ∀ (x₁ x₂ : ℝ), 
    let y₁ := curve x₁
    let y₂ := line x₂
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l3094_309452


namespace NUMINAMATH_CALUDE_coin_probability_l3094_309472

theorem coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) : 
  (Nat.choose 6 3 : ℝ) * p^3 * (1-p)^3 = 1/20 → p = 1/400 := by
  sorry

end NUMINAMATH_CALUDE_coin_probability_l3094_309472


namespace NUMINAMATH_CALUDE_library_visitors_average_l3094_309443

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (sundays_in_month : ℕ) (h1 : sunday_visitors = 140) 
  (h2 : other_day_visitors = 80) (h3 : days_in_month = 30) (h4 : sundays_in_month = 4) :
  (sunday_visitors * sundays_in_month + other_day_visitors * (days_in_month - sundays_in_month)) / 
  days_in_month = 88 := by
  sorry

#check library_visitors_average

end NUMINAMATH_CALUDE_library_visitors_average_l3094_309443


namespace NUMINAMATH_CALUDE_good_number_exists_l3094_309417

/-- A function that checks if two numbers have the same digits (possibly in different order) --/
def sameDigits (a b : ℕ) : Prop := sorry

/-- A function that checks if a number is a four-digit number --/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem good_number_exists : ∃ n : ℕ, 
  isFourDigit n ∧ 
  n % 11 = 0 ∧ 
  sameDigits n (3 * n) ∧
  n = 2475 := by sorry

end NUMINAMATH_CALUDE_good_number_exists_l3094_309417


namespace NUMINAMATH_CALUDE_commute_days_theorem_l3094_309478

theorem commute_days_theorem (x : ℕ) 
  (morning_bus : ℕ) 
  (afternoon_train : ℕ) 
  (bike_commute : ℕ) 
  (h1 : morning_bus = 12) 
  (h2 : afternoon_train = 20) 
  (h3 : bike_commute = 10) 
  (h4 : x = morning_bus + afternoon_train - bike_commute) : x = 30 := by
  sorry

#check commute_days_theorem

end NUMINAMATH_CALUDE_commute_days_theorem_l3094_309478


namespace NUMINAMATH_CALUDE_ski_prices_l3094_309456

theorem ski_prices (x y : ℝ) 
  (eq1 : 2 * x + y = 340)
  (eq2 : 3 * x + 2 * y = 570) : 
  x = 110 ∧ y = 120 := by
  sorry

end NUMINAMATH_CALUDE_ski_prices_l3094_309456


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3094_309481

/-- Represents the train journey with given conditions -/
structure TrainJourney where
  speed : ℝ
  stop_interval : ℝ
  regular_stop_duration : ℝ
  fifth_stop_duration : ℝ
  total_travel_time : ℝ

/-- Calculates the total distance traveled by the train -/
def total_distance (journey : TrainJourney) : ℝ :=
  sorry

/-- Theorem stating the total distance traveled by the train -/
theorem train_distance_theorem (journey : TrainJourney) 
  (h1 : journey.speed = 60)
  (h2 : journey.stop_interval = 48)
  (h3 : journey.regular_stop_duration = 1/6)
  (h4 : journey.fifth_stop_duration = 1/2)
  (h5 : journey.total_travel_time = 58) :
  total_distance journey = 2870 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l3094_309481


namespace NUMINAMATH_CALUDE_equation_transformation_l3094_309449

theorem equation_transformation (x y : ℝ) (h : 2*x - 3*y + 6 = 0) : 
  6*x - 9*y + 6 = -12 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3094_309449


namespace NUMINAMATH_CALUDE_brendan_lawn_cutting_l3094_309485

/-- The number of yards Brendan could cut per day before buying the lawnmower -/
def initial_yards : ℝ := 8

/-- The increase in cutting capacity after buying the lawnmower -/
def capacity_increase : ℝ := 0.5

/-- The number of days Brendan worked with the new lawnmower -/
def days_worked : ℕ := 7

/-- The total number of yards cut with the new lawnmower -/
def total_yards_cut : ℕ := 84

theorem brendan_lawn_cutting :
  initial_yards * (1 + capacity_increase) * days_worked = total_yards_cut :=
by sorry

end NUMINAMATH_CALUDE_brendan_lawn_cutting_l3094_309485


namespace NUMINAMATH_CALUDE_central_square_side_length_l3094_309484

/-- Given a rectangular hallway and total flooring area, calculates the side length of a central square area --/
theorem central_square_side_length 
  (hallway_length : ℝ) 
  (hallway_width : ℝ) 
  (total_area : ℝ) 
  (h1 : hallway_length = 6)
  (h2 : hallway_width = 4)
  (h3 : total_area = 124) :
  let hallway_area := hallway_length * hallway_width
  let central_area := total_area - hallway_area
  let side_length := Real.sqrt central_area
  side_length = 10 := by sorry

end NUMINAMATH_CALUDE_central_square_side_length_l3094_309484


namespace NUMINAMATH_CALUDE_circle_point_x_value_l3094_309475

theorem circle_point_x_value (x : ℝ) :
  let center : ℝ × ℝ := ((21 - (-3)) / 2 + (-3), 0)
  let radius : ℝ := (21 - (-3)) / 2
  (x - center.1) ^ 2 + (12 - center.2) ^ 2 = radius ^ 2 →
  x = 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_x_value_l3094_309475


namespace NUMINAMATH_CALUDE_find_x_l3094_309422

theorem find_x : ∃ x : ℕ, 
  (∃ k : ℕ, x = 9 * k) ∧ 
  x^2 > 120 ∧ 
  x < 25 ∧ 
  x % 2 = 1 ∧
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_find_x_l3094_309422


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3094_309432

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3094_309432


namespace NUMINAMATH_CALUDE_unique_poly3_satisfying_conditions_l3094_309400

/-- A polynomial function of degree exactly 3 -/
structure Poly3 where
  f : ℝ → ℝ
  degree_3 : ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

/-- The conditions that the polynomial function must satisfy -/
def satisfies_conditions (p : Poly3) : Prop :=
  ∀ x : ℝ, p.f (x^2) = (p.f x)^2 ∧
            p.f (x^2) = p.f (p.f x) ∧
            p.f 1 = 1

/-- Theorem stating the uniqueness of the polynomial function -/
theorem unique_poly3_satisfying_conditions :
  ∃! p : Poly3, satisfies_conditions p :=
sorry

end NUMINAMATH_CALUDE_unique_poly3_satisfying_conditions_l3094_309400


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_C_R_B_l3094_309402

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}

-- Define the complement of B in ℝ
def C_R_B : Set ℝ := {x | ¬ (x ∈ B)}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the union of A and complement of B
theorem union_A_C_R_B : A ∪ C_R_B = {x : ℝ | -4 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_C_R_B_l3094_309402


namespace NUMINAMATH_CALUDE_specific_value_calculation_l3094_309465

theorem specific_value_calculation : ∀ (x : ℕ), x = 11 → x + 3 + 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_specific_value_calculation_l3094_309465


namespace NUMINAMATH_CALUDE_problem_statement_l3094_309498

open Real

theorem problem_statement :
  (∃ x : ℝ, x - 2 > log x) ∧
  ¬(∀ x : ℝ, exp x > 1) ∧
  ((∃ x : ℝ, x - 2 > log x) ∧ ¬(∀ x : ℝ, exp x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3094_309498


namespace NUMINAMATH_CALUDE_perimeter_710_implies_n_66_l3094_309418

/-- Represents the perimeter of the nth figure in the sequence -/
def perimeter (n : ℕ) : ℕ := 60 + (n - 1) * 10

/-- Theorem stating that if the perimeter of the nth figure is 710 cm, then n is 66 -/
theorem perimeter_710_implies_n_66 : ∃ n : ℕ, perimeter n = 710 ∧ n = 66 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_710_implies_n_66_l3094_309418


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3094_309414

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : 3 * Real.cos (2 * α) + Real.sin α = 1) : 
  Real.sin (Real.pi - α) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3094_309414


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3094_309457

def U : Set Nat := {1, 2, 3, 4, 5}
def P : Set Nat := {2, 4}
def Q : Set Nat := {1, 3, 4, 6}

theorem complement_intersection_theorem :
  (U \ P) ∩ Q = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3094_309457


namespace NUMINAMATH_CALUDE_xy_value_l3094_309488

theorem xy_value (x y : ℝ) (h : Real.sqrt (x - 3) + |y + 2| = 0) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3094_309488


namespace NUMINAMATH_CALUDE_function_coefficient_sum_l3094_309464

theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_function_coefficient_sum_l3094_309464


namespace NUMINAMATH_CALUDE_nonshaded_perimeter_is_64_l3094_309434

/-- A structure representing the geometric configuration described in the problem -/
structure GeometricConfig where
  outer_length : ℝ
  outer_width : ℝ
  inner_length : ℝ
  inner_width : ℝ
  extension : ℝ
  shaded_area : ℝ

/-- The perimeter of the non-shaded region given the geometric configuration -/
def nonshaded_perimeter (config : GeometricConfig) : ℝ :=
  2 * (config.outer_width + (config.outer_length + config.extension - config.inner_length))

/-- Theorem stating that given the specific geometric configuration, 
    the perimeter of the non-shaded region is 64 inches -/
theorem nonshaded_perimeter_is_64 (config : GeometricConfig) 
  (h1 : config.outer_length = 12)
  (h2 : config.outer_width = 10)
  (h3 : config.inner_length = 3)
  (h4 : config.inner_width = 4)
  (h5 : config.extension = 3)
  (h6 : config.shaded_area = 120) :
  nonshaded_perimeter config = 64 := by
  sorry

end NUMINAMATH_CALUDE_nonshaded_perimeter_is_64_l3094_309434


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3094_309455

theorem power_of_two_equality : ∃ x : ℕ, 8^12 + 8^12 + 8^12 = 2^x ∧ x = 38 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3094_309455


namespace NUMINAMATH_CALUDE_number_difference_l3094_309448

theorem number_difference (x y : ℤ) (h1 : x + y = 62) (h2 : y = 25) : |x - y| = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3094_309448


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3094_309489

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 46 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 100 / 11.5 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3094_309489


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3094_309420

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3094_309420


namespace NUMINAMATH_CALUDE_power_fraction_evaluation_l3094_309435

theorem power_fraction_evaluation :
  ((5^2014)^2 - (5^2012)^2) / ((5^2013)^2 - (5^2011)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_evaluation_l3094_309435


namespace NUMINAMATH_CALUDE_train_length_l3094_309473

/-- Given a train with speed 50 km/hr crossing a pole in 9 seconds, its length is 125 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 50 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = speed * (1000 / 3600) * time → -- length calculation
  length = 125 := by sorry

end NUMINAMATH_CALUDE_train_length_l3094_309473


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3094_309411

/-- Represents a quadratic equation ax² - 6x + c = 0 with exactly one solution -/
structure UniqueQuadratic where
  a : ℝ
  c : ℝ
  has_unique_solution : ∃! x, a * x^2 - 6 * x + c = 0

theorem unique_quadratic_solution (q : UniqueQuadratic)
  (sum_eq_12 : q.a + q.c = 12)
  (a_lt_c : q.a < q.c) :
  q.a = 6 - 3 * Real.sqrt 3 ∧ q.c = 6 + 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3094_309411


namespace NUMINAMATH_CALUDE_simplest_radical_among_options_l3094_309468

def is_simplest_radical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ 
  (∀ m : ℕ, m ^ 2 ≤ n → m ^ 2 = n ∨ m ^ 2 < n) ∧
  (∀ a b : ℕ, n = a * b → (a = 1 ∨ b = 1 ∨ ¬ ∃ k : ℕ, k ^ 2 = a))

theorem simplest_radical_among_options :
  is_simplest_radical (Real.sqrt 7) ∧
  ¬ is_simplest_radical (Real.sqrt 9) ∧
  ¬ is_simplest_radical (Real.sqrt 20) ∧
  ¬ is_simplest_radical (Real.sqrt (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simplest_radical_among_options_l3094_309468


namespace NUMINAMATH_CALUDE_equation_solution_l3094_309433

theorem equation_solution : ∃ x : ℚ, 300 * 2 + (12 + 4) * x / 8 = 602 :=
  by
    use 1
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3094_309433


namespace NUMINAMATH_CALUDE_domain_of_function_1_l3094_309413

theorem domain_of_function_1 (x : ℝ) : 
  Set.univ = {x : ℝ | ∃ y : ℝ, y = (2 * x^2 - 1) / (x^2 + 3)} :=
sorry

end NUMINAMATH_CALUDE_domain_of_function_1_l3094_309413


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3094_309459

theorem no_solution_absolute_value_equation :
  ¬∃ (x : ℝ), |2*x - 5| = 3*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3094_309459


namespace NUMINAMATH_CALUDE_bowen_purchase_ratio_l3094_309427

/-- Represents the purchase of pens and pencils -/
structure Purchase where
  pen_price : ℚ
  pencil_price : ℚ
  num_pens : ℕ
  total_spent : ℚ

/-- Calculates the ratio of pencils to pens for a given purchase -/
def pencil_to_pen_ratio (p : Purchase) : ℚ × ℚ :=
  let pencil_cost := p.total_spent - p.pen_price * p.num_pens
  let num_pencils := pencil_cost / p.pencil_price
  let gcd := Nat.gcd (Nat.floor num_pencils) p.num_pens
  ((num_pencils / gcd), (p.num_pens / gcd))

/-- Theorem stating that for the given purchase conditions, the ratio of pencils to pens is 7:5 -/
theorem bowen_purchase_ratio : 
  let p : Purchase := {
    pen_price := 15/100,
    pencil_price := 25/100,
    num_pens := 40,
    total_spent := 20
  }
  pencil_to_pen_ratio p = (7, 5) := by sorry

end NUMINAMATH_CALUDE_bowen_purchase_ratio_l3094_309427


namespace NUMINAMATH_CALUDE_yogurt_topping_combinations_l3094_309486

/-- The number of yogurt flavors --/
def yogurt_flavors : ℕ := 6

/-- The number of available toppings --/
def toppings : ℕ := 8

/-- The number of toppings to choose --/
def choose_toppings : ℕ := 2

/-- Theorem stating the number of unique combinations --/
theorem yogurt_topping_combinations : 
  yogurt_flavors * Nat.choose toppings choose_toppings = 168 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_topping_combinations_l3094_309486


namespace NUMINAMATH_CALUDE_tom_search_cost_l3094_309436

/-- Calculates the total cost for Tom's search service given the number of days -/
def total_cost (days : ℕ) : ℕ :=
  if days ≤ 5 then
    100 * days
  else
    500 + 60 * (days - 5)

/-- The problem statement -/
theorem tom_search_cost : total_cost 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_tom_search_cost_l3094_309436


namespace NUMINAMATH_CALUDE_ball_reaches_top_left_corner_l3094_309401

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  width : ℕ
  length : ℕ

/-- Represents the path of a ball on the billiard table -/
def ball_path (table : BilliardTable) : ℕ :=
  Nat.lcm table.width table.length

/-- Theorem: A ball launched at 45° from the bottom-left corner of a 26x1965 table
    will reach the top-left corner after traveling the LCM of 26 and 1965 in both directions -/
theorem ball_reaches_top_left_corner (table : BilliardTable) 
    (h1 : table.width = 26) (h2 : table.length = 1965) :
    ball_path table = 50990 ∧ 
    50990 % table.width = 0 ∧ 
    50990 % table.length = 0 := by
  sorry

#eval ball_path { width := 26, length := 1965 }

end NUMINAMATH_CALUDE_ball_reaches_top_left_corner_l3094_309401


namespace NUMINAMATH_CALUDE_parabola_focus_l3094_309412

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -1/8 * x^2

-- Define symmetry about y-axis
def symmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the focus of a parabola
def focus (f h : ℝ) : Prop :=
  ∀ x y, parabola x y → (x - 0)^2 + (y - h)^2 = (y + 2)^2

-- Theorem statement
theorem parabola_focus :
  symmetricAboutYAxis (λ x => -1/8 * x^2) →
  focus 0 (-2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3094_309412


namespace NUMINAMATH_CALUDE_max_value_of_min_expression_l3094_309421

theorem max_value_of_min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (min (-1/y) (y + 1/x)) ≤ Real.sqrt 2 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ min x (min (-1/y) (y + 1/x)) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_min_expression_l3094_309421


namespace NUMINAMATH_CALUDE_power_seven_mod_twelve_l3094_309445

theorem power_seven_mod_twelve : 7^150 % 12 = 1 := by sorry

end NUMINAMATH_CALUDE_power_seven_mod_twelve_l3094_309445


namespace NUMINAMATH_CALUDE_x_cubed_plus_y_cubed_le_two_l3094_309492

theorem x_cubed_plus_y_cubed_le_two (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_ineq : x^2 + y^3 ≥ x^3 + y^4) : 
  x^3 + y^3 ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_x_cubed_plus_y_cubed_le_two_l3094_309492


namespace NUMINAMATH_CALUDE_arcsin_arccos_pi_sixth_l3094_309471

theorem arcsin_arccos_pi_sixth : 
  Real.arcsin (1/2) = π/6 ∧ Real.arccos (Real.sqrt 3/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_arccos_pi_sixth_l3094_309471


namespace NUMINAMATH_CALUDE_complex_sum_real_imag_parts_l3094_309458

theorem complex_sum_real_imag_parts (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : 
  z.re + z.im = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_real_imag_parts_l3094_309458


namespace NUMINAMATH_CALUDE_reach_all_integers_l3094_309407

/-- Represents the allowed operations on positive integers -/
inductive Operation
  | append4 : Operation
  | append0 : Operation
  | divideBy2 : Operation

/-- Applies an operation to a positive integer -/
def applyOperation (n : ℕ+) (op : Operation) : ℕ+ :=
  match op with
  | Operation.append4 => ⟨10 * n.val + 4, by sorry⟩
  | Operation.append0 => ⟨10 * n.val, by sorry⟩
  | Operation.divideBy2 => if n.val % 2 = 0 then ⟨n.val / 2, by sorry⟩ else n

/-- Applies a sequence of operations to a positive integer -/
def applyOperations (n : ℕ+) (ops : List Operation) : ℕ+ :=
  ops.foldl applyOperation n

/-- Theorem stating that any positive integer can be reached from 4 using the allowed operations -/
theorem reach_all_integers (n : ℕ+) : 
  ∃ (ops : List Operation), applyOperations ⟨4, by norm_num⟩ ops = n := by
  sorry

end NUMINAMATH_CALUDE_reach_all_integers_l3094_309407


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3094_309466

theorem triangle_third_side_length (a b c : ℕ) (h1 : a = 5) (h2 : b = 2) (h3 : Odd c) : 
  (a + b > c ∧ a + c > b ∧ b + c > a) → c = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3094_309466


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3094_309483

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x + 1/2 ≥ 0) → 
  (k > 0 ∧ k ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3094_309483


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3094_309424

theorem unique_integer_solution (w x y z : ℤ) :
  w^2 + 11*x^2 - 8*y^2 - 12*y*z - 10*z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3094_309424


namespace NUMINAMATH_CALUDE_light_blocks_count_is_twenty_l3094_309430

/-- Represents a tower with light colored blocks -/
structure LightTower where
  central_column_height : ℕ
  outer_columns_count : ℕ
  outer_column_height : ℕ

/-- Calculates the total number of light colored blocks in the tower -/
def total_light_blocks (tower : LightTower) : ℕ :=
  tower.central_column_height + tower.outer_columns_count * tower.outer_column_height

/-- Theorem stating that the total number of light colored blocks in the specific tower is 20 -/
theorem light_blocks_count_is_twenty :
  ∃ (tower : LightTower),
    tower.central_column_height = 4 ∧
    tower.outer_columns_count = 8 ∧
    tower.outer_column_height = 2 ∧
    total_light_blocks tower = 20 := by
  sorry


end NUMINAMATH_CALUDE_light_blocks_count_is_twenty_l3094_309430


namespace NUMINAMATH_CALUDE_direct_proportion_increasing_iff_m_gt_two_l3094_309447

/-- A direct proportion function with coefficient (m - 2) -/
def direct_proportion (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x

/-- The function is increasing -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem direct_proportion_increasing_iff_m_gt_two (m : ℝ) :
  is_increasing (direct_proportion m) ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_increasing_iff_m_gt_two_l3094_309447


namespace NUMINAMATH_CALUDE_x₂_1994th_place_l3094_309438

-- Define the equation
def equation (x : ℝ) : Prop := x * Real.sqrt 8 + 1 / (x * Real.sqrt 8) = Real.sqrt 8

-- Define the two real solutions
axiom x₁ : ℝ
axiom x₂ : ℝ

-- Define that x₁ and x₂ satisfy the equation
axiom x₁_satisfies : equation x₁
axiom x₂_satisfies : equation x₂

-- Define the decimal place function (simplified for this problem)
def decimal_place (x : ℝ) (n : ℕ) : ℕ := sorry

-- Define that the 1994th decimal place of x₁ is 6
axiom x₁_1994th_place : decimal_place x₁ 1994 = 6

-- Theorem to prove
theorem x₂_1994th_place : decimal_place x₂ 1994 = 3 := by sorry

end NUMINAMATH_CALUDE_x₂_1994th_place_l3094_309438


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3094_309490

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023*a - 1 = 0) :
  a*(a+1)*(a-1) + 2023*a^2 + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3094_309490


namespace NUMINAMATH_CALUDE_train_length_l3094_309446

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 290 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 230 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3094_309446


namespace NUMINAMATH_CALUDE_rectangle_division_l3094_309403

theorem rectangle_division (original_width original_height : ℕ) 
  (piece1_width piece1_height : ℕ) (piece2_width piece2_height : ℕ) 
  (piece3_width piece3_height : ℕ) (piece4_width piece4_height : ℕ) :
  original_width = 15 ∧ original_height = 7 ∧
  piece1_width = 7 ∧ piece1_height = 7 ∧
  piece2_width = 8 ∧ piece2_height = 3 ∧
  piece3_width = 7 ∧ piece3_height = 4 ∧
  piece4_width = 8 ∧ piece4_height = 4 →
  original_width * original_height = 
    piece1_width * piece1_height + 
    piece2_width * piece2_height + 
    piece3_width * piece3_height + 
    piece4_width * piece4_height :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l3094_309403


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3094_309482

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from a pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  area pan.panDimensions / area pan.pieceDimensions

/-- Theorem stating that a 24x15 inch pan can be divided into exactly 60 pieces of 3x2 inch brownies -/
theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3094_309482


namespace NUMINAMATH_CALUDE_expression_lower_bound_l3094_309406

theorem expression_lower_bound (n : ℤ) (L : ℤ) :
  (∃! (S : Finset ℤ), S.card = 25 ∧ ∀ m ∈ S, L < 4*m + 7 ∧ 4*m + 7 < 100) →
  L = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l3094_309406


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l3094_309453

theorem complex_quadratic_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = Complex.I * 2 ∧ 
  z₂ = -2 - Complex.I * 2 ∧ 
  z₁^2 + 2*z₁ = -3 + Complex.I * 4 ∧
  z₂^2 + 2*z₂ = -3 + Complex.I * 4 := by
sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l3094_309453


namespace NUMINAMATH_CALUDE_exists_five_digit_number_with_digit_sum_31_divisible_by_31_l3094_309439

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_five_digit_number_with_digit_sum_31_divisible_by_31 :
  ∃ n : ℕ, is_five_digit n ∧ digit_sum n = 31 ∧ n % 31 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_five_digit_number_with_digit_sum_31_divisible_by_31_l3094_309439


namespace NUMINAMATH_CALUDE_power_function_m_equals_four_l3094_309429

/-- A function f is a power function if it has the form f(x) = ax^b where a and b are constants and a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ b

/-- Given f(x) = (m^2 - 3m - 3)x^(√m) is a power function, prove that m = 4 -/
theorem power_function_m_equals_four (m : ℝ) 
  (h : is_power_function (fun x ↦ (m^2 - 3*m - 3) * x^(Real.sqrt m))) : 
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_power_function_m_equals_four_l3094_309429


namespace NUMINAMATH_CALUDE_exists_perpendicular_angles_not_equal_or_180_l3094_309469

/-- Two angles in 3D space with perpendicular sides -/
structure PerpendicularAngles where
  α : Real
  β : Real
  perp_sides : Bool

/-- Predicate for angles being equal or summing to 180° -/
def equal_or_sum_180 (angles : PerpendicularAngles) : Prop :=
  angles.α = angles.β ∨ angles.α + angles.β = 180

/-- Theorem stating the existence of perpendicular angles that don't satisfy the condition -/
theorem exists_perpendicular_angles_not_equal_or_180 :
  ∃ (angles : PerpendicularAngles), angles.perp_sides ∧ ¬(equal_or_sum_180 angles) :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_angles_not_equal_or_180_l3094_309469


namespace NUMINAMATH_CALUDE_inequality_preservation_l3094_309497

theorem inequality_preservation (m n : ℝ) (h : m > n) : m / 4 > n / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3094_309497


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3094_309467

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- The problem statement -/
theorem arithmetic_sequence_remainder (a₁ aₙ d : ℕ) 
  (h₁ : a₁ = 3)
  (h₂ : aₙ = 273)
  (h₃ : d = 6)
  (h₄ : ∀ k, 1 ≤ k ∧ k ≤ (aₙ - a₁) / d + 1 → a₁ + (k - 1) * d = 6 * k - 3) :
  arithmetic_sum a₁ aₙ d % 8 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3094_309467


namespace NUMINAMATH_CALUDE_sine_of_sum_inverse_sine_and_tangent_l3094_309454

theorem sine_of_sum_inverse_sine_and_tangent :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_sum_inverse_sine_and_tangent_l3094_309454


namespace NUMINAMATH_CALUDE_number_of_books_a_l3094_309480

/-- Proves that the number of books (a) is 12, given the conditions -/
theorem number_of_books_a (total : ℕ) (diff : ℕ) : 
  (total = 20) → (diff = 4) → ∃ (a b : ℕ), (a + b = total) ∧ (a = b + diff) ∧ (a = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_books_a_l3094_309480


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l3094_309493

/-- Represents an arithmetic sequence with n+1 terms, first term y, and common difference 4 -/
def arithmetic_sequence (y : ℤ) (n : ℕ) : List ℤ :=
  List.range (n + 1) |>.map (fun i => y + 4 * i)

/-- The sum of cubes of all terms in the sequence -/
def sum_of_cubes (seq : List ℤ) : ℤ :=
  seq.map (fun x => x^3) |>.sum

theorem arithmetic_sequence_sum_of_cubes (y : ℤ) (n : ℕ) :
  n > 6 →
  sum_of_cubes (arithmetic_sequence y n) = -5832 →
  n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l3094_309493


namespace NUMINAMATH_CALUDE_min_focal_chord_length_is_2p_l3094_309477

/-- Represents a parabola defined by the equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The minimum length of focal chords for a given parabola -/
def min_focal_chord_length (par : Parabola) : ℝ := 2 * par.p

/-- Theorem stating that the minimum length of focal chords is 2p -/
theorem min_focal_chord_length_is_2p (par : Parabola) :
  min_focal_chord_length par = 2 * par.p := by sorry

end NUMINAMATH_CALUDE_min_focal_chord_length_is_2p_l3094_309477


namespace NUMINAMATH_CALUDE_two_digit_sum_to_four_digit_sum_l3094_309415

/-- Given two two-digit numbers that sum to 137, prove that the sum of the four-digit numbers
    formed by concatenating these digits in order and in reverse order is 13837. -/
theorem two_digit_sum_to_four_digit_sum
  (A B C D : ℕ)
  (h_AB_two_digit : A * 10 + B < 100)
  (h_CD_two_digit : C * 10 + D < 100)
  (h_sum : A * 10 + B + C * 10 + D = 137) :
  (A * 1000 + B * 100 + C * 10 + D) + (C * 1000 + D * 100 + A * 10 + B) = 13837 := by
  sorry


end NUMINAMATH_CALUDE_two_digit_sum_to_four_digit_sum_l3094_309415


namespace NUMINAMATH_CALUDE_min_value_expression_l3094_309462

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  ∃ (x₀ y₀ : ℝ), 2*x₀*y₀ - 2*x₀ - y₀ = 8 ∧ ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 1 → 2*x*y - 2*x - y ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3094_309462


namespace NUMINAMATH_CALUDE_p_investment_calculation_l3094_309451

def investment_ratio (p_investment q_investment : ℚ) : ℚ := p_investment / q_investment

theorem p_investment_calculation (q_investment : ℚ) (profit_ratio : ℚ) :
  q_investment = 30000 →
  profit_ratio = 3 / 5 →
  ∃ p_investment : ℚ, 
    investment_ratio p_investment q_investment = profit_ratio ∧
    p_investment = 18000 := by
  sorry

end NUMINAMATH_CALUDE_p_investment_calculation_l3094_309451


namespace NUMINAMATH_CALUDE_area_perimeter_ratio_inequality_l3094_309410

/-- A convex polygon -/
structure ConvexPolygon where
  area : ℝ
  perimeter : ℝ

/-- X is contained within Y -/
def isContainedIn (X Y : ConvexPolygon) : Prop := sorry

theorem area_perimeter_ratio_inequality {X Y : ConvexPolygon} 
  (h : isContainedIn X Y) :
  X.area / X.perimeter < 2 * Y.area / Y.perimeter := by
  sorry

end NUMINAMATH_CALUDE_area_perimeter_ratio_inequality_l3094_309410


namespace NUMINAMATH_CALUDE_inequality_solution_l3094_309487

theorem inequality_solution (x : ℝ) :
  (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ↔ 
  (x > -9/2 ∧ x < -2) ∨ (x > (1 - Real.sqrt 5) / 2 ∧ x < (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3094_309487


namespace NUMINAMATH_CALUDE_additional_houses_built_l3094_309495

/-- Proves the number of additional houses built between the first half of the year and October -/
theorem additional_houses_built
  (total_houses : ℕ)
  (first_half_fraction : ℚ)
  (remaining_houses : ℕ)
  (h1 : total_houses = 2000)
  (h2 : first_half_fraction = 3/5)
  (h3 : remaining_houses = 500) :
  (total_houses - remaining_houses) - (first_half_fraction * total_houses) = 300 := by
  sorry

end NUMINAMATH_CALUDE_additional_houses_built_l3094_309495


namespace NUMINAMATH_CALUDE_classroom_sum_problem_l3094_309460

theorem classroom_sum_problem (a b : ℤ) : 
  3 * a + 4 * b = 161 → (a = 17 ∨ b = 17) → (a = 31 ∨ b = 31) := by
  sorry

end NUMINAMATH_CALUDE_classroom_sum_problem_l3094_309460
