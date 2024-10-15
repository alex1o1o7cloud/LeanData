import Mathlib

namespace NUMINAMATH_CALUDE_real_part_of_sum_l3114_311489

theorem real_part_of_sum (z₁ z₂ : ℂ) (h₁ : z₁ = 4 + 19 * Complex.I) (h₂ : z₂ = 6 + 9 * Complex.I) :
  (z₁ + z₂).re = 10 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_sum_l3114_311489


namespace NUMINAMATH_CALUDE_curve_tangent_problem_l3114_311476

/-- The curve C is defined by the equation y = 2x³ + ax + a -/
def C (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x + a

/-- The derivative of C with respect to x -/
def C_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + a

theorem curve_tangent_problem (a : ℝ) :
  (C a (-1) = 0) →  -- C passes through point M(-1, 0)
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    C_derivative a t₁ + C_derivative a t₂ = 0 ∧  -- |MA| = |MB| condition
    4 * t₁^3 + 6 * t₁^2 = 0 ∧                   -- Tangent line condition for t₁
    4 * t₂^3 + 6 * t₂^2 = 0) →                  -- Tangent line condition for t₂
  a = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_problem_l3114_311476


namespace NUMINAMATH_CALUDE_cricket_player_innings_l3114_311420

/-- A cricket player's innings problem -/
theorem cricket_player_innings (current_average : ℚ) (next_innings_runs : ℕ) (average_increase : ℚ) :
  current_average = 25 →
  next_innings_runs = 121 →
  average_increase = 6 →
  (∃ n : ℕ, (n * current_average + next_innings_runs) / (n + 1) = current_average + average_increase ∧ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_cricket_player_innings_l3114_311420


namespace NUMINAMATH_CALUDE_jenny_sleep_hours_l3114_311470

theorem jenny_sleep_hours (minutes_per_hour : ℕ) (total_sleep_minutes : ℕ) 
  (h1 : minutes_per_hour = 60) 
  (h2 : total_sleep_minutes = 480) : 
  total_sleep_minutes / minutes_per_hour = 8 := by
sorry

end NUMINAMATH_CALUDE_jenny_sleep_hours_l3114_311470


namespace NUMINAMATH_CALUDE_f_2009_eq_zero_l3114_311493

/-- An even function on ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- An odd function on ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- Main theorem -/
theorem f_2009_eq_zero
  (f g : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_f_1 : f 1 = 0)
  (h_odd : OddFunction g)
  (h_g_def : ∀ x, g x = f (x - 1)) :
  f 2009 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_2009_eq_zero_l3114_311493


namespace NUMINAMATH_CALUDE_second_month_sale_correct_l3114_311494

/-- Calculates the sale in the second month given the sales figures for other months and the average sale -/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (average_sale : ℕ) : ℕ :=
  5 * average_sale - (first_month + third_month + fourth_month + fifth_month)

/-- Theorem stating that the calculated second month sale is correct -/
theorem second_month_sale_correct (first_month third_month fourth_month fifth_month average_sale : ℕ) :
  first_month = 5700 →
  third_month = 6855 →
  fourth_month = 3850 →
  fifth_month = 14045 →
  average_sale = 7800 →
  calculate_second_month_sale first_month third_month fourth_month fifth_month average_sale = 7550 :=
by
  sorry

#eval calculate_second_month_sale 5700 6855 3850 14045 7800

end NUMINAMATH_CALUDE_second_month_sale_correct_l3114_311494


namespace NUMINAMATH_CALUDE_max_dot_product_unit_vector_l3114_311412

theorem max_dot_product_unit_vector (a b : ℝ × ℝ) :
  (∀ (x y : ℝ), a = (x, y) → x^2 + y^2 = 1) →
  b = (Real.sqrt 3, -1) →
  (∃ (m : ℝ), m = (a.1 * b.1 + a.2 * b.2) ∧ 
    ∀ (x y : ℝ), x^2 + y^2 = 1 → (x * b.1 + y * b.2) ≤ m) →
  (∃ (max : ℝ), max = 2 ∧ 
    ∀ (x y : ℝ), x^2 + y^2 = 1 → (x * b.1 + y * b.2) ≤ max) :=
by
  sorry

end NUMINAMATH_CALUDE_max_dot_product_unit_vector_l3114_311412


namespace NUMINAMATH_CALUDE_max_solitar_result_l3114_311407

/-- The greatest prime divisor of a natural number -/
def greatestPrimeDivisor (n : ℕ) : ℕ := sorry

/-- The set of numbers from 1 to 16 -/
def initialSet : Finset ℕ := Finset.range 16

/-- The result of one step in the solitar game -/
def solitarStep (s : Finset ℕ) : Finset ℕ := sorry

/-- The final result of the solitar game -/
def solitarResult (s : Finset ℕ) : ℕ := sorry

/-- The maximum possible final number in the solitar game -/
theorem max_solitar_result : 
  ∃ (result : ℕ), solitarResult initialSet = result ∧ result ≤ 19 ∧ 
  ∀ (other : ℕ), solitarResult initialSet = other → other ≤ result :=
sorry

end NUMINAMATH_CALUDE_max_solitar_result_l3114_311407


namespace NUMINAMATH_CALUDE_removed_number_value_l3114_311464

theorem removed_number_value (S : ℝ) (X : ℝ) : 
  S / 50 = 56 →
  (S - X - 55) / 48 = 56.25 →
  X = 45 := by
sorry

end NUMINAMATH_CALUDE_removed_number_value_l3114_311464


namespace NUMINAMATH_CALUDE_min_period_cos_x_div_3_l3114_311402

/-- The minimum positive period of y = cos(x/3) is 6π -/
theorem min_period_cos_x_div_3 : ∃ (T : ℝ), T > 0 ∧ T = 6 * Real.pi ∧
  ∀ (x : ℝ), Real.cos (x / 3) = Real.cos ((x + T) / 3) ∧
  ∀ (T' : ℝ), 0 < T' ∧ T' < T → ∃ (x : ℝ), Real.cos (x / 3) ≠ Real.cos ((x + T') / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_period_cos_x_div_3_l3114_311402


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3114_311409

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 20 →
  l * w ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3114_311409


namespace NUMINAMATH_CALUDE_peanut_bags_needed_l3114_311437

-- Define the flight duration in hours
def flight_duration : ℕ := 2

-- Define the number of peanuts per bag
def peanuts_per_bag : ℕ := 30

-- Define the interval between eating peanuts in minutes
def eating_interval : ℕ := 1

-- Theorem statement
theorem peanut_bags_needed : 
  (flight_duration * 60) / peanuts_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_peanut_bags_needed_l3114_311437


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3114_311408

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - x + a = 0) ↔ a ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3114_311408


namespace NUMINAMATH_CALUDE_function_property_l3114_311418

theorem function_property (f : ℝ → ℝ) (k : ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + k * f x * y) :
  ∃ (a b : ℝ), (a = 0 ∧ b = 4) ∧ 
    (f 2 = a ∨ f 2 = b) ∧
    (∀ c : ℝ, f 2 = c → (c = a ∨ c = b)) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3114_311418


namespace NUMINAMATH_CALUDE_number_puzzle_l3114_311475

theorem number_puzzle (x : ℝ) : 3 * (2 * x^2 + 15) - 7 = 91 → x = Real.sqrt (53 / 6) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3114_311475


namespace NUMINAMATH_CALUDE_satisfying_polynomial_iff_polynomial_form_l3114_311453

/-- A polynomial that satisfies the given equation for all real x -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 6*x + 8) * P x = (x^2 + 2*x) * P (x - 2)

/-- The form of the polynomial that satisfies the equation -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, P x = c * x^2 * (x^2 - 4)

/-- Theorem stating the equivalence between satisfying the equation and having the specific form -/
theorem satisfying_polynomial_iff_polynomial_form :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ PolynomialForm P :=
sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_iff_polynomial_form_l3114_311453


namespace NUMINAMATH_CALUDE_intersection_complement_P_and_Q_l3114_311415

-- Define the set P
def P : Set ℝ := {y | ∃ x > 0, y = (1/2)^x}

-- Define the set Q
def Q : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- Define the complement of P in ℝ
def complement_P : Set ℝ := {y | y ≤ 0 ∨ y ≥ 1}

-- Theorem statement
theorem intersection_complement_P_and_Q :
  (complement_P ∩ Q) = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_P_and_Q_l3114_311415


namespace NUMINAMATH_CALUDE_profit_percentage_15_20_l3114_311477

/-- Represents the profit percentage when selling articles -/
def profit_percentage (sold : ℕ) (cost_equivalent : ℕ) : ℚ :=
  (cost_equivalent - sold) / sold

/-- Theorem: The profit percentage when selling 15 articles at the cost of 20 is 1/3 -/
theorem profit_percentage_15_20 : profit_percentage 15 20 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_15_20_l3114_311477


namespace NUMINAMATH_CALUDE_system_solution_l3114_311467

/-- Given a system of equations in x, y, and m, prove the relationship between x and y,
    and find the value of m when x + y = -10 -/
theorem system_solution (x y m : ℝ) 
  (eq1 : 3 * x + 5 * y = m + 2)
  (eq2 : 2 * x + 3 * y = m) :
  (y = 1 - x / 2) ∧ 
  (x + y = -10 → m = -8) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3114_311467


namespace NUMINAMATH_CALUDE_car_distance_theorem_l3114_311474

/-- Represents a car traveling between two points --/
structure Car where
  speed_forward : ℝ
  speed_backward : ℝ

/-- The problem setup --/
def problem_setup : Prop :=
  ∃ (distance : ℝ) (car_a car_b : Car),
    distance = 900 ∧
    car_a.speed_forward = 40 ∧
    car_a.speed_backward = 50 ∧
    car_b.speed_forward = 50 ∧
    car_b.speed_backward = 40

/-- The theorem to be proved --/
theorem car_distance_theorem (setup : problem_setup) :
  ∃ (total_distance : ℝ),
    total_distance = 1813900 ∧
    (∀ (distance : ℝ) (car_a car_b : Car),
      distance = 900 ∧
      car_a.speed_forward = 40 ∧
      car_a.speed_backward = 50 ∧
      car_b.speed_forward = 50 ∧
      car_b.speed_backward = 40 →
      total_distance = 
        (2016 / 2 - 1) * 2 * distance + 
        (car_a.speed_backward * distance) / (car_a.speed_backward + car_b.speed_backward)) :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l3114_311474


namespace NUMINAMATH_CALUDE_magic_8_ball_theorem_l3114_311422

def magic_8_ball_probability : ℚ := 242112 / 823543

theorem magic_8_ball_theorem (n : ℕ) (k : ℕ) (p : ℚ) 
  (h1 : n = 7) 
  (h2 : k = 3) 
  (h3 : p = 3 / 7) :
  Nat.choose n k * p^k * (1 - p)^(n - k) = magic_8_ball_probability := by
  sorry

#check magic_8_ball_theorem

end NUMINAMATH_CALUDE_magic_8_ball_theorem_l3114_311422


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l3114_311438

theorem smallest_perfect_square_divisible_by_5_and_7 : ∃ n : ℕ,
  n > 0 ∧
  (∃ m : ℕ, n = m ^ 2) ∧
  n % 5 = 0 ∧
  n % 7 = 0 ∧
  n = 1225 ∧
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l ^ 2) → k % 5 = 0 → k % 7 = 0 → k ≥ 1225) :=
by
  sorry

#check smallest_perfect_square_divisible_by_5_and_7

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l3114_311438


namespace NUMINAMATH_CALUDE_divisor_problem_l3114_311414

theorem divisor_problem (N : ℕ) (D : ℕ) : 
  N % 5 = 0 ∧ N / 5 = 5 ∧ N % D = 3 → D = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3114_311414


namespace NUMINAMATH_CALUDE_fliers_left_for_next_day_l3114_311498

def total_fliers : ℕ := 2500
def morning_fraction : ℚ := 1/5
def afternoon_fraction : ℚ := 1/4

theorem fliers_left_for_next_day :
  let morning_sent := total_fliers * morning_fraction
  let remaining_after_morning := total_fliers - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  total_fliers - morning_sent - afternoon_sent = 1500 := by sorry

end NUMINAMATH_CALUDE_fliers_left_for_next_day_l3114_311498


namespace NUMINAMATH_CALUDE_divisibility_by_1989_l3114_311497

theorem divisibility_by_1989 (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℤ, n^(n^(n^n)) - n^(n^n) = 1989 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_by_1989_l3114_311497


namespace NUMINAMATH_CALUDE_bicycle_parking_income_l3114_311465

/-- Represents the total income from bicycle parking --/
def total_income (x : ℝ) : ℝ := -0.3 * x + 1600

/-- Theorem stating the relationship between the number of ordinary bicycles parked and the total income --/
theorem bicycle_parking_income (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 2000) : 
  total_income x = 0.5 * x + 0.8 * (2000 - x) := by
  sorry

#check bicycle_parking_income

end NUMINAMATH_CALUDE_bicycle_parking_income_l3114_311465


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l3114_311496

/-- Represents a cylindrical fuel tank -/
structure FuelTank where
  capacity : ℝ
  initial_percentage : ℝ
  initial_volume : ℝ

/-- Theorem stating the capacity of the fuel tank -/
theorem fuel_tank_capacity (tank : FuelTank)
  (h1 : tank.initial_percentage = 0.25)
  (h2 : tank.initial_volume = 60)
  : tank.capacity = 240 := by
  sorry

#check fuel_tank_capacity

end NUMINAMATH_CALUDE_fuel_tank_capacity_l3114_311496


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3114_311466

/-- Represents the speeds and distances in a race between two runners A and B -/
structure RaceParameters where
  speedA : ℝ
  speedB : ℝ
  totalDistance : ℝ
  headStart : ℝ

/-- Theorem stating that if A and B finish at the same time in a race with given parameters,
    then A's speed is 4 times B's speed -/
theorem race_speed_ratio 
  (race : RaceParameters) 
  (h1 : race.totalDistance = 100)
  (h2 : race.headStart = 75)
  (h3 : race.totalDistance / race.speedA = (race.totalDistance - race.headStart) / race.speedB) :
  race.speedA = 4 * race.speedB := by
  sorry


end NUMINAMATH_CALUDE_race_speed_ratio_l3114_311466


namespace NUMINAMATH_CALUDE_platform_length_l3114_311455

/-- Given a train of length 600 meters that takes 54 seconds to cross a platform
    and 36 seconds to cross a signal pole, the length of the platform is 300 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 600 →
  time_platform = 54 →
  time_pole = 36 →
  ∃ platform_length : ℝ,
    platform_length = 300 ∧
    train_length / time_pole = (train_length + platform_length) / time_platform :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3114_311455


namespace NUMINAMATH_CALUDE_fuel_calculation_correct_l3114_311411

/-- Calculates the total fuel needed for a plane trip given the specified conditions -/
def total_fuel_needed (base_fuel_per_mile : ℕ) (fuel_increase_per_person : ℕ) 
  (fuel_increase_per_bag : ℕ) (passengers : ℕ) (crew : ℕ) (bags_per_person : ℕ) 
  (trip_distance : ℕ) : ℕ :=
  let total_people := passengers + crew
  let total_bags := total_people * bags_per_person
  let fuel_per_mile := base_fuel_per_mile + 
    total_people * fuel_increase_per_person + 
    total_bags * fuel_increase_per_bag
  fuel_per_mile * trip_distance

/-- Theorem stating that the total fuel needed for the given conditions is 106,000 gallons -/
theorem fuel_calculation_correct : 
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 := by
  sorry

end NUMINAMATH_CALUDE_fuel_calculation_correct_l3114_311411


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l3114_311425

/-- Represents the daily rainfall probabilities and amounts -/
structure DailyRainfall where
  no_rain_prob : Real
  light_rain_prob : Real
  heavy_rain_prob : Real
  light_rain_amount : Real
  heavy_rain_amount : Real

/-- Calculates the expected rainfall for a single day -/
def expected_daily_rainfall (d : DailyRainfall) : Real :=
  d.no_rain_prob * 0 + d.light_rain_prob * d.light_rain_amount + d.heavy_rain_prob * d.heavy_rain_amount

/-- Theorem: Expected total rainfall for a week -/
theorem expected_weekly_rainfall (d : DailyRainfall)
  (h1 : d.no_rain_prob = 0.2)
  (h2 : d.light_rain_prob = 0.3)
  (h3 : d.heavy_rain_prob = 0.5)
  (h4 : d.light_rain_amount = 2)
  (h5 : d.heavy_rain_amount = 8)
  (h6 : d.no_rain_prob + d.light_rain_prob + d.heavy_rain_prob = 1) :
  7 * (expected_daily_rainfall d) = 32.2 := by
  sorry

#eval 7 * (0.2 * 0 + 0.3 * 2 + 0.5 * 8)

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l3114_311425


namespace NUMINAMATH_CALUDE_movies_watched_count_l3114_311444

/-- The number of movies watched in the 'crazy silly school' series -/
def movies_watched : ℕ := 21

/-- The number of books read in the 'crazy silly school' series -/
def books_read : ℕ := 7

/-- Theorem stating that the number of movies watched is 21 -/
theorem movies_watched_count : 
  movies_watched = books_read + 14 := by sorry

end NUMINAMATH_CALUDE_movies_watched_count_l3114_311444


namespace NUMINAMATH_CALUDE_range_of_a_l3114_311426

-- Define the conditions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (x a : ℝ) :
  (∀ x, p x → (x < -3 ∨ x > 1)) →
  (∀ x, ¬(p x) ↔ (-3 ≤ x ∧ x ≤ 1)) →
  (∀ x, ¬(q x a) ↔ x ≤ a) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ q x a) →
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3114_311426


namespace NUMINAMATH_CALUDE_work_completion_time_l3114_311430

/-- The number of days B needs to complete the entire work alone -/
def B_total_days : ℝ := 14.999999999999996

/-- The number of days A works before leaving -/
def A_partial_days : ℝ := 5

/-- The number of days B needs to complete the remaining work after A leaves -/
def B_remaining_days : ℝ := 10

/-- The number of days A needs to complete the entire work alone -/
def A_total_days : ℝ := 15

theorem work_completion_time :
  B_total_days = 14.999999999999996 →
  A_partial_days = 5 →
  B_remaining_days = 10 →
  A_total_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3114_311430


namespace NUMINAMATH_CALUDE_harry_terry_calculation_l3114_311410

theorem harry_terry_calculation (x : ℤ) : 
  let H := 12 - (3 + 7) + x
  let T := 12 - 3 + 7 + x
  H - T + x = -14 + x := by
sorry

end NUMINAMATH_CALUDE_harry_terry_calculation_l3114_311410


namespace NUMINAMATH_CALUDE_max_quarters_in_box_l3114_311488

/-- Represents the number of coins of each type in the coin box -/
structure CoinBox where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the box -/
def total_coins (box : CoinBox) : ℕ :=
  box.nickels + box.dimes + box.quarters

/-- The total value of coins in cents -/
def total_value (box : CoinBox) : ℕ :=
  5 * box.nickels + 10 * box.dimes + 25 * box.quarters

/-- Theorem stating the maximum number of quarters possible -/
theorem max_quarters_in_box :
  ∃ (box : CoinBox),
    total_coins box = 120 ∧
    total_value box = 1000 ∧
    (∀ (other_box : CoinBox),
      total_coins other_box = 120 →
      total_value other_box = 1000 →
      other_box.quarters ≤ box.quarters) ∧
    box.quarters = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_quarters_in_box_l3114_311488


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3114_311417

theorem inequality_system_solution_set :
  ∀ a : ℝ, (2 * a - 3 < 0 ∧ 1 - a < 0) ↔ (1 < a ∧ a < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3114_311417


namespace NUMINAMATH_CALUDE_exists_card_with_1024_l3114_311428

/-- The number of cards for each natural number up to 1968 -/
def num_cards (n : ℕ) : ℕ := n

/-- The condition that each card has divisors of its number written on it -/
def has_divisors (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d ≤ 1968 → (num_cards d) ≥ 1

/-- The main theorem to prove -/
theorem exists_card_with_1024 (h : ∀ n ≤ 1968, has_divisors n) :
  (num_cards 1024) > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_card_with_1024_l3114_311428


namespace NUMINAMATH_CALUDE_adjacent_zero_point_functions_range_l3114_311491

def adjacent_zero_point_functions (f g : ℝ → ℝ) : Prop :=
  ∀ (α β : ℝ), f α = 0 → g β = 0 → |α - β| ≤ 1

def f (x : ℝ) : ℝ := x - 1

def g (a x : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_functions_range (a : ℝ) :
  adjacent_zero_point_functions f (g a) → a ∈ Set.Icc 2 (7/3) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_zero_point_functions_range_l3114_311491


namespace NUMINAMATH_CALUDE_problem_solution_l3114_311487

-- Define the function f
def f (m : ℕ) (x : ℝ) : ℝ := |x - m| + |x|

-- State the theorem
theorem problem_solution (m : ℕ) (α β : ℝ) 
  (h1 : m > 0)
  (h2 : ∃ x : ℝ, f m x < 2)
  (h3 : α > 1)
  (h4 : β > 1)
  (h5 : f m α + f m β = 6) :
  (m = 1) ∧ ((4 / α) + (1 / β) ≥ 9 / 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3114_311487


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3114_311492

/-- Given the following conditions for a cloth sale:
    - Total cloth length: 400 meters
    - Total selling price: Rs. 18,000
    - Loss per meter: Rs. 5
    Prove that the cost price for one meter of cloth is Rs. 50. -/
theorem cloth_cost_price 
  (total_length : ℝ) 
  (total_selling_price : ℝ) 
  (loss_per_meter : ℝ) 
  (h1 : total_length = 400)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  (total_selling_price / total_length) + loss_per_meter = 50 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3114_311492


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3114_311479

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two angles equal
  α = β →
  -- One of the equal angles is 50°
  α = 50 →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  max α (max β γ) = 80 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3114_311479


namespace NUMINAMATH_CALUDE_cosine_value_implies_expression_value_l3114_311495

theorem cosine_value_implies_expression_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.cos (π/2 + x) = 4/5) : 
  (Real.sin (2*x) - 2 * (Real.sin x)^2) / (1 + Real.tan x) = -168/25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_implies_expression_value_l3114_311495


namespace NUMINAMATH_CALUDE_three_eighths_percent_of_160_l3114_311449

theorem three_eighths_percent_of_160 : (3 / 8 / 100) * 160 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_percent_of_160_l3114_311449


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3114_311499

theorem rectangle_dimension_change (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let A := L * W
  let W' := 0.4 * W
  let A' := 1.36 * A
  ∃ L', L' = 3.4 * L ∧ A' = L' * W' :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3114_311499


namespace NUMINAMATH_CALUDE_problem_solution_l3114_311485

/-- The function f(x) defined in the problem -/
def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 4 * c * x^3 + 2 * (c^2 - 3) * x

theorem problem_solution :
  ∃! c : ℝ,
    (∀ x : ℝ, x < -1 → (f_derivative c x < 0)) ∧
    (∀ x : ℝ, -1 < x ∧ x < 0 → (f_derivative c x > 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3114_311485


namespace NUMINAMATH_CALUDE_power_equality_l3114_311469

theorem power_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3114_311469


namespace NUMINAMATH_CALUDE_equation_proof_l3114_311440

theorem equation_proof : (12 : ℕ)^3 * 6^4 / 432 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3114_311440


namespace NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l3114_311439

theorem complex_roots_equilateral_triangle (z₁ z₂ p q : ℂ) : 
  z₁^2 + p*z₁ + q = 0 →
  z₂^2 + p*z₂ + q = 0 →
  z₂ = Complex.exp (2*Real.pi*Complex.I/3) * z₁ →
  p^2 / q = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l3114_311439


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3114_311481

/-- Configuration of squares and rectangle forming a large square -/
structure SquareConfiguration where
  s : ℝ  -- Side length of small squares
  large_square_side : ℝ  -- Side length of the large square
  rectangle_length : ℝ  -- Length of the rectangle
  rectangle_width : ℝ  -- Width of the rectangle

/-- Properties of the square configuration -/
def valid_configuration (config : SquareConfiguration) : Prop :=
  config.s > 0 ∧
  config.large_square_side = 3 * config.s ∧
  config.rectangle_length = config.large_square_side ∧
  config.rectangle_width = config.s

/-- Theorem stating the ratio of rectangle's length to width -/
theorem rectangle_ratio (config : SquareConfiguration) 
  (h : valid_configuration config) : 
  config.rectangle_length / config.rectangle_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3114_311481


namespace NUMINAMATH_CALUDE_donut_selection_problem_l3114_311471

theorem donut_selection_problem :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l3114_311471


namespace NUMINAMATH_CALUDE_construct_equilateral_from_given_l3114_311431

-- Define the given triangle
structure GivenTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_angles : angle1 + angle2 + angle3 = 180
  angle_values : angle1 = 40 ∧ angle2 = 70 ∧ angle3 = 70

-- Define an equilateral triangle
def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

-- Theorem statement
theorem construct_equilateral_from_given (t : GivenTriangle) :
  ∃ (a b c : ℝ), is_equilateral a b c :=
sorry


end NUMINAMATH_CALUDE_construct_equilateral_from_given_l3114_311431


namespace NUMINAMATH_CALUDE_sum_of_fractions_to_decimal_l3114_311451

theorem sum_of_fractions_to_decimal : (5 : ℚ) / 16 + (1 : ℚ) / 4 = (5625 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_to_decimal_l3114_311451


namespace NUMINAMATH_CALUDE_constant_fifth_term_binomial_expansion_l3114_311456

theorem constant_fifth_term_binomial_expansion (a x : ℝ) (n : ℕ) :
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 4) * a^(n-4) * (-1)^4 * x^(n-8) = k) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_constant_fifth_term_binomial_expansion_l3114_311456


namespace NUMINAMATH_CALUDE_range_of_f_l3114_311457

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3114_311457


namespace NUMINAMATH_CALUDE_fraction_equality_l3114_311460

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 8)
  (h2 : c / b = 4)
  (h3 : c / d = 1 / 3) :
  d / a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3114_311460


namespace NUMINAMATH_CALUDE_combination_permutation_properties_l3114_311421

-- Define combination function
def C (n m : ℕ) : ℕ := 
  if m ≤ n then Nat.choose n m else 0

-- Define permutation function
def A (n m : ℕ) : ℕ := 
  if m ≤ n then Nat.factorial n / Nat.factorial (n - m) else 0

-- Theorem statement
theorem combination_permutation_properties (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (C n m = C n (n - m)) ∧
  (C (n + 1) m = C n (m - 1) + C n m) ∧
  (A n m = C n m * A m m) ∧
  (A (n + 1) (m + 1) ≠ (m + 1) * A n m) := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_properties_l3114_311421


namespace NUMINAMATH_CALUDE_series_sum_8_eq_43690_l3114_311480

def series_sum : ℕ → ℕ 
  | 0 => 2
  | n + 1 => 2 * (1 + 4 * series_sum n)

theorem series_sum_8_eq_43690 : series_sum 7 = 43690 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_8_eq_43690_l3114_311480


namespace NUMINAMATH_CALUDE_problem_solution_l3114_311404

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3114_311404


namespace NUMINAMATH_CALUDE_sum_plus_even_count_l3114_311445

def sum_of_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_plus_even_count : 
  let x := sum_of_range 50 60
  let y := count_even_in_range 50 60
  x + y = 611 := by sorry

end NUMINAMATH_CALUDE_sum_plus_even_count_l3114_311445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3114_311403

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- The sequence is increasing -/
def IncreasingSequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  IncreasingSequence b →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3114_311403


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3114_311447

open Real

theorem max_value_of_expression (t : ℝ) :
  (∃ (max : ℝ), ∀ (t : ℝ), (3^t - 4*t^2)*t / 9^t ≤ max) ∧
  (∃ (t_max : ℝ), (3^t_max - 4*t_max^2)*t_max / 9^t_max = sqrt 3 / 9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3114_311447


namespace NUMINAMATH_CALUDE_extreme_value_derivative_l3114_311468

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for f to have an extreme value at x₀
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

-- State the theorem
theorem extreme_value_derivative (x₀ : ℝ) :
  (has_extreme_value f x₀ → deriv f x₀ = 0) ∧
  ¬(deriv f x₀ = 0 → has_extreme_value f x₀) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_derivative_l3114_311468


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3114_311461

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 5}

theorem intersection_of_P_and_Q : P ∩ Q = {(4, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3114_311461


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3114_311413

theorem constant_term_expansion (a : ℝ) : 
  a > 0 → (∃ k : ℕ, k = (a^2 * 2^2 * 6 : ℝ) ∧ k = 96) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3114_311413


namespace NUMINAMATH_CALUDE_f_value_at_one_l3114_311427

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 50*x + c

theorem f_value_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -217 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l3114_311427


namespace NUMINAMATH_CALUDE_minimize_sqrt_difference_l3114_311419

theorem minimize_sqrt_difference (p : ℕ) (h_p : Nat.Prime p) (h_odd : Odd p) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    (∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0 ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

#check minimize_sqrt_difference

end NUMINAMATH_CALUDE_minimize_sqrt_difference_l3114_311419


namespace NUMINAMATH_CALUDE_polynomial_evaluation_part1_polynomial_evaluation_part2_l3114_311490

/-- Part 1: Polynomial evaluation given a condition -/
theorem polynomial_evaluation_part1 (x : ℝ) (h : x^2 - x = 3) :
  x^4 - 2*x^3 + 3*x^2 - 2*x + 2 = 17 := by
  sorry

/-- Part 2: Polynomial evaluation given a condition -/
theorem polynomial_evaluation_part2 (x y : ℝ) (h : x^2 + y^2 = 1) :
  2*x^4 + 3*x^2*y^2 + y^4 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_part1_polynomial_evaluation_part2_l3114_311490


namespace NUMINAMATH_CALUDE_collinear_vectors_n_equals_one_l3114_311429

def a (n : ℝ) : Fin 2 → ℝ := ![1, n]
def b (n : ℝ) : Fin 2 → ℝ := ![-1, n - 2]

def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem collinear_vectors_n_equals_one :
  ∀ n : ℝ, collinear (a n) (b n) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_n_equals_one_l3114_311429


namespace NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3114_311441

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℚ
  /-- The common difference of the sequence -/
  d : ℚ
  /-- The sum of the first 60 terms is 660 -/
  sum_first_60 : (60 : ℚ) / 2 * (2 * a + 59 * d) = 660
  /-- The sum of the next 60 terms (terms 61 to 120) is 3660 -/
  sum_next_60 : (60 : ℚ) / 2 * (2 * (a + 60 * d) + 59 * d) = 3660

/-- The first term of the arithmetic sequence with the given properties is -163/12 -/
theorem first_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) : seq.a = -163/12 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3114_311441


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l3114_311434

/-- The function f(x) = x^3 - 3x has a minimum value of -2. -/
theorem min_value_cubic_function :
  ∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, x^3 - 3*x ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l3114_311434


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3114_311446

theorem complex_equation_solution (a : ℝ) (z : ℂ) : 
  z * Complex.I = (a + 1 : ℂ) + 4 * Complex.I → Complex.abs z = 5 → a = 2 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3114_311446


namespace NUMINAMATH_CALUDE_max_b_value_l3114_311458

noncomputable section

variable (a b : ℝ)

def f (x : ℝ) := (3/2) * x^2 - 2*a*x

def g (x : ℝ) := a^2 * Real.log x + b

def common_point (x : ℝ) := f a x = g a b x

def common_tangent (x : ℝ) := deriv (f a) x = deriv (g a b) x

theorem max_b_value (h1 : a > 0) 
  (h2 : ∃ x > 0, common_point a b x ∧ common_tangent a b x) :
  ∃ b_max : ℝ, b_max = 1 / (2 * Real.exp 2) ∧ 
  (∀ b : ℝ, (∃ x > 0, common_point a b x ∧ common_tangent a b x) → b ≤ b_max) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l3114_311458


namespace NUMINAMATH_CALUDE_cabin_cost_l3114_311400

theorem cabin_cost (total_cost land_cost cabin_cost : ℕ) : 
  total_cost = 30000 →
  land_cost = 4 * cabin_cost →
  total_cost = land_cost + cabin_cost →
  cabin_cost = 6000 := by
  sorry

end NUMINAMATH_CALUDE_cabin_cost_l3114_311400


namespace NUMINAMATH_CALUDE_interval_condition_l3114_311405

theorem interval_condition (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) :=
by sorry

end NUMINAMATH_CALUDE_interval_condition_l3114_311405


namespace NUMINAMATH_CALUDE_certain_number_problem_l3114_311436

theorem certain_number_problem (x : ℝ) (h : 0.6 * x = 0.4 * 30 + 18) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3114_311436


namespace NUMINAMATH_CALUDE_abc_inequality_l3114_311483

theorem abc_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : c = -1) :
  a * b + a * c + b * c ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3114_311483


namespace NUMINAMATH_CALUDE_max_movies_watched_l3114_311472

def movie_duration : ℕ := 90
def tuesday_watch_time : ℕ := 270

theorem max_movies_watched (wednesday_multiplier : ℕ) (h : wednesday_multiplier = 2) :
  let tuesday_movies := tuesday_watch_time / movie_duration
  let wednesday_movies := wednesday_multiplier * tuesday_movies
  tuesday_movies + wednesday_movies = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_movies_watched_l3114_311472


namespace NUMINAMATH_CALUDE_trig_identity_equivalence_l3114_311443

theorem trig_identity_equivalence (x : ℝ) :
  (2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x)) ↔
  (∃ k : ℤ, x = (π / 16) * (4 * ↑k + 1)) :=
sorry

end NUMINAMATH_CALUDE_trig_identity_equivalence_l3114_311443


namespace NUMINAMATH_CALUDE_disprove_line_tangent_to_circle_l3114_311484

theorem disprove_line_tangent_to_circle :
  ∃ (a b : ℝ), a^2 + b^2 ≠ 0 ∧ a^2 + b^2 ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_disprove_line_tangent_to_circle_l3114_311484


namespace NUMINAMATH_CALUDE_parallel_lines_x_intercept_l3114_311401

/-- Given two lines that are parallel, prove that the x-intercept of one line is -1. -/
theorem parallel_lines_x_intercept (m : ℝ) :
  (∀ x y, y + m * (x + 1) = 0 ↔ m * y - (2 * m + 1) * x = 1) →
  (m ≠ 0 ∧ 2 * m + 1 ≠ 0) →
  ∃ x, x + m * (x + 1) = 0 ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_x_intercept_l3114_311401


namespace NUMINAMATH_CALUDE_tan_alpha_third_quadrant_l3114_311432

theorem tan_alpha_third_quadrant (α : Real) 
  (h1 : Real.sin (Real.pi + α) = 3/5)
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.tan α = 3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_third_quadrant_l3114_311432


namespace NUMINAMATH_CALUDE_f_max_values_l3114_311423

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

theorem f_max_values (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔ a < -4 ∨ a > 4 :=
sorry

end NUMINAMATH_CALUDE_f_max_values_l3114_311423


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3114_311424

def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 12*x^2 + 5*x - 20

theorem polynomial_remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x + 2) * q x + 98 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3114_311424


namespace NUMINAMATH_CALUDE_obtuse_angle_measure_l3114_311452

/-- An obtuse angle divided by a perpendicular line into two angles with a ratio of 6:1 measures 105°. -/
theorem obtuse_angle_measure (θ : ℝ) (h1 : 90 < θ) (h2 : θ < 180) : 
  ∃ (α β : ℝ), α + β = θ ∧ α / β = 6 ∧ θ = 105 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angle_measure_l3114_311452


namespace NUMINAMATH_CALUDE_dog_park_problem_l3114_311486

theorem dog_park_problem (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_eared_dogs : ℕ) :
  (2 * spotted_dogs = total_dogs) →  -- Half of the dogs have spots
  (5 * pointy_eared_dogs = total_dogs) →  -- 1/5 of the dogs have pointy ears
  (spotted_dogs = 15) →  -- 15 dogs have spots
  pointy_eared_dogs = 6 :=
by sorry

end NUMINAMATH_CALUDE_dog_park_problem_l3114_311486


namespace NUMINAMATH_CALUDE_johns_age_ratio_l3114_311442

/-- The ratio of John's age 5 years ago to his age in 8 years -/
def age_ratio (current_age : ℕ) : ℚ :=
  (current_age - 5 : ℚ) / (current_age + 8)

/-- Theorem stating that the ratio of John's age 5 years ago to his age in 8 years is 1:2 -/
theorem johns_age_ratio :
  age_ratio 18 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_ratio_l3114_311442


namespace NUMINAMATH_CALUDE_three_chapters_eight_pages_l3114_311450

/-- Calculates the total number of pages read given the number of chapters and pages per chapter -/
def pages_read (chapters : ℕ) (pages_per_chapter : ℕ) : ℕ :=
  chapters * pages_per_chapter

/-- Proves that reading 3 chapters of 8 pages each results in 24 pages read -/
theorem three_chapters_eight_pages :
  pages_read 3 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_chapters_eight_pages_l3114_311450


namespace NUMINAMATH_CALUDE_students_per_group_l3114_311454

theorem students_per_group (total_students : ℕ) (num_teachers : ℕ) 
  (h1 : total_students = 850) (h2 : num_teachers = 23) :
  (total_students / num_teachers : ℕ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l3114_311454


namespace NUMINAMATH_CALUDE_sum_of_roots_l3114_311416

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 4) = 12) (hb : b * (b - 4) = 12) (hab : a ≠ b) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3114_311416


namespace NUMINAMATH_CALUDE_house_locations_contradiction_l3114_311482

-- Define the directions
inductive Direction
  | North
  | South
  | East
  | West
  | Northeast
  | Northwest
  | Southeast
  | Southwest

-- Define a function to get the opposite direction
def oppositeDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East
  | Direction.Northeast => Direction.Southwest
  | Direction.Northwest => Direction.Southeast
  | Direction.Southeast => Direction.Northwest
  | Direction.Southwest => Direction.Northeast

-- Define the theorem
theorem house_locations_contradiction :
  ∀ (house1 house2 : Type) (direction1 direction2 : Direction),
    (direction1 = Direction.Southeast ∧ direction2 = Direction.Southwest) →
    (oppositeDirection direction1 ≠ direction2) :=
by sorry

end NUMINAMATH_CALUDE_house_locations_contradiction_l3114_311482


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3114_311478

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 1000 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3114_311478


namespace NUMINAMATH_CALUDE_initial_investment_l3114_311435

/-- Given an initial investment A at a simple annual interest rate r,
    prove that A = 5000 when the interest on A is $250 and
    the interest on $20,000 at the same rate is $1000. -/
theorem initial_investment (A r : ℝ) : 
  A > 0 →
  r > 0 →
  A * r / 100 = 250 →
  20000 * r / 100 = 1000 →
  A = 5000 := by
sorry

end NUMINAMATH_CALUDE_initial_investment_l3114_311435


namespace NUMINAMATH_CALUDE_triangle_properties_l3114_311448

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that under certain conditions, we can determine the values of A, b, and c. -/
theorem triangle_properties (t : Triangle) 
  (h1 : ∃ (k : ℝ), k * (Real.sqrt 3 * t.a) = t.c ∧ k * (1 + Real.cos t.A) = Real.sin t.C) 
  (h2 : 3 * t.b * t.c = 16 - t.a^2)
  (h3 : t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3114_311448


namespace NUMINAMATH_CALUDE_average_half_median_l3114_311433

theorem average_half_median (a b c : ℤ) : 
  a < b → b < c → a = 0 → (a + b + c) / 3 = b / 2 → c / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_half_median_l3114_311433


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3114_311463

theorem modulus_of_complex_number (z : ℂ) : z = 1 - 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3114_311463


namespace NUMINAMATH_CALUDE_cylinder_volume_l3114_311406

/-- The volume of a cylinder with radius 5 cm and height 8 cm is 628 cm³, given that π ≈ 3.14 -/
theorem cylinder_volume : 
  let r : ℝ := 5
  let h : ℝ := 8
  let π : ℝ := 3.14
  π * r^2 * h = 628 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3114_311406


namespace NUMINAMATH_CALUDE_autumn_grain_purchase_l3114_311473

/-- 
Theorem: If the total purchase of autumn grain nationwide exceeded 180 million tons, 
and x represents the amount of autumn grain purchased in China this year in billion tons, 
then x > 1.8.
-/
theorem autumn_grain_purchase (x : ℝ) 
  (h : x * 1000 > 180) : x > 1.8 := by
  sorry

end NUMINAMATH_CALUDE_autumn_grain_purchase_l3114_311473


namespace NUMINAMATH_CALUDE_club_officer_selection_l3114_311459

/-- Represents the number of ways to choose officers in a club -/
def chooseOfficers (totalMembers boyCount girlCount : ℕ) : ℕ :=
  totalMembers * (if boyCount = girlCount then boyCount else 0) * (boyCount - 1)

/-- Theorem stating the number of ways to choose officers in the given conditions -/
theorem club_officer_selection :
  let totalMembers := 24
  let boyCount := 12
  let girlCount := 12
  chooseOfficers totalMembers boyCount girlCount = 3168 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l3114_311459


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3114_311462

/-- 
Given three people a, b, and c, where:
- a is two years older than b
- The total age of a, b, and c is 27
- b is 10 years old

This theorem proves that the ratio of b's age to c's age is 2:1
-/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 27 →
  b = 10 →
  b = 2 * c :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3114_311462
