import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_correct_sum_l1689_168911

theorem incorrect_correct_sum : ∃ x : ℤ, 
  (x - 5 + 14 = 39) ∧ (39 + (5 * x + 14) = 203) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_correct_sum_l1689_168911


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l1689_168924

theorem isabella_currency_exchange :
  ∃ (d : ℕ), d > 0 ∧ (10 : ℚ) / 7 * d - 40 = d ∧ (d / 10 + d % 10 = 12) := by
  sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l1689_168924


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l1689_168991

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 2) (h_a4 : a 4 = 4) : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l1689_168991


namespace NUMINAMATH_CALUDE_dress_design_combinations_l1689_168965

theorem dress_design_combinations (num_colors num_patterns : ℕ) : 
  num_colors = 5 → num_patterns = 6 → num_colors * num_patterns = 30 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l1689_168965


namespace NUMINAMATH_CALUDE_three_digit_geometric_progression_exists_l1689_168902

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : Nat) : Prop :=
  b * b = a * c

/-- Converts a ThreeDigitNumber to its decimal representation -/
def to_decimal (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- The main theorem statement -/
theorem three_digit_geometric_progression_exists : ∃! (n : ThreeDigitNumber),
  (is_geometric_progression n.1 n.2.1 n.2.2) ∧
  (to_decimal (n.2.2, n.2.1, n.1) = to_decimal n - 594) ∧
  (10 * n.2.2 + n.2.1 = 10 * n.2.1 + n.2.2 - 18) ∧
  (to_decimal n = 842) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_geometric_progression_exists_l1689_168902


namespace NUMINAMATH_CALUDE_parallelogram_base_l1689_168959

theorem parallelogram_base (area height : ℝ) (h1 : area = 231) (h2 : height = 11) :
  area / height = 21 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1689_168959


namespace NUMINAMATH_CALUDE_counterexample_exists_l1689_168983

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem counterexample_exists : ∃ n : ℕ, 
  ¬ is_prime n ∧ ¬ is_prime (n - 3) ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1689_168983


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1689_168972

noncomputable def nondecreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem functional_equation_solution
  (f : ℝ → ℝ)
  (h_nondecreasing : nondecreasing_function f)
  (h_f_0 : f 0 = 0)
  (h_f_1 : f 1 = 1)
  (h_equation : ∀ a b, a < 1 ∧ 1 < b → f a + f b = f a * f b + f (a + b - a * b)) :
  ∃ c k, c > 0 ∧ k ≥ 0 ∧
    (∀ x, x > 1 → f x = c * (x - 1) ^ k) ∧
    (∀ x, x < 1 → f x = 1 - (1 - x) ^ k) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1689_168972


namespace NUMINAMATH_CALUDE_power_function_through_point_l1689_168939

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 = 2) : 
  f 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1689_168939


namespace NUMINAMATH_CALUDE_line_equation_proof_l1689_168945

/-- A line parameterized by t ∈ ℝ -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific line given in the problem -/
def given_line : ParametricLine where
  x := λ t => 3 * t + 6
  y := λ t => 5 * t - 7

/-- The equation of a line in slope-intercept form -/
structure LineEquation where
  slope : ℝ
  intercept : ℝ

theorem line_equation_proof (L : ParametricLine) 
    (h : L = given_line) : 
    ∃ (eq : LineEquation), 
      eq.slope = 5/3 ∧ 
      eq.intercept = -17 ∧
      ∀ t, L.y t = eq.slope * (L.x t) + eq.intercept :=
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1689_168945


namespace NUMINAMATH_CALUDE_smallest_n_complex_equality_l1689_168906

theorem smallest_n_complex_equality (n : ℕ) (a b c : ℝ) :
  (n > 0) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (∀ k : ℕ, k > 0 ∧ k < n → ¬ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y*I + z*I)^k = (x - y*I - z*I)^k) →
  ((a + b*I + c*I)^n = (a - b*I - c*I)^n) →
  ((b + c) / a = Real.sqrt (12 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_complex_equality_l1689_168906


namespace NUMINAMATH_CALUDE_fraction_comparison_l1689_168982

theorem fraction_comparison : -3/4 > -4/5 := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1689_168982


namespace NUMINAMATH_CALUDE_f_min_value_l1689_168943

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x)^2 / (cos x * sin x - (sin x)^2)

theorem f_min_value (x : ℝ) (h : 0 < x ∧ x < π/4) :
  f x ≥ 4 ∧ ∃ y, 0 < y ∧ y < π/4 ∧ f y = 4 :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l1689_168943


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1689_168995

theorem complex_magnitude_problem (a : ℝ) (z : ℂ) : 
  z = (a * Complex.I) / (4 - 3 * Complex.I) → 
  Complex.abs z = 5 → 
  a = 25 ∨ a = -25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1689_168995


namespace NUMINAMATH_CALUDE_ivan_walking_time_l1689_168954

/-- Represents the problem of determining how long Ivan Ivanovich walked. -/
theorem ivan_walking_time 
  (s : ℝ) -- Total distance from home to work
  (t : ℝ) -- Usual time taken by car
  (v : ℝ) -- Car's speed
  (u : ℝ) -- Ivan's walking speed
  (h1 : s = v * t) -- Total distance equals car speed times usual time
  (h2 : s = u * T + v * (t - T + 1/6)) -- Distance covered by walking and car
  (h3 : v * (1/12) = s - u * T) -- Car meets Ivan halfway through its usual journey
  (h4 : v > 0) -- Car speed is positive
  (h5 : u > 0) -- Walking speed is positive
  (h6 : v > u) -- Car is faster than walking
  : T = 55 := by
  sorry

#check ivan_walking_time

end NUMINAMATH_CALUDE_ivan_walking_time_l1689_168954


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_negative_condition_l1689_168908

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem tangent_line_at_one (h : ℝ → ℝ) :
  (∀ x > 0, h x = f (-2) x) →
  (∀ x, h x = -x + 1) →
  ∃ c, h c = f (-2) c ∧ (∀ x, h x - (f (-2) c) = -(x - c)) :=
sorry

theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, 0 < x → x < 1/a → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, 1/a < x → x < y → f a y < f a x) :=
sorry

theorem negative_condition (a : ℝ) :
  (∀ x > 0, f a x < 0) ↔ a > 1/exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_negative_condition_l1689_168908


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1689_168970

/-- Calculates the number of whole cubes that fit along a given dimension -/
def cubesAlongDimension (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the volume of a rectangular box -/
def boxVolume (length width height : ℕ) : ℕ :=
  length * width * height

/-- Calculates the volume of a cube -/
def cubeVolume (size : ℕ) : ℕ :=
  size * size * size

/-- Calculates the total volume occupied by cubes in the box -/
def occupiedVolume (boxLength boxWidth boxHeight cubeSize : ℕ) : ℕ :=
  let numCubesLength := cubesAlongDimension boxLength cubeSize
  let numCubesWidth := cubesAlongDimension boxWidth cubeSize
  let numCubesHeight := cubesAlongDimension boxHeight cubeSize
  let totalCubes := numCubesLength * numCubesWidth * numCubesHeight
  totalCubes * cubeVolume cubeSize

theorem cube_volume_ratio (boxLength boxWidth boxHeight cubeSize : ℕ) 
  (h1 : boxLength = 4)
  (h2 : boxWidth = 7)
  (h3 : boxHeight = 8)
  (h4 : cubeSize = 2) :
  (occupiedVolume boxLength boxWidth boxHeight cubeSize : ℚ) / 
  (boxVolume boxLength boxWidth boxHeight : ℚ) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1689_168970


namespace NUMINAMATH_CALUDE_ana_charging_time_proof_l1689_168992

def smartphone_full_charge : ℕ := 26
def tablet_full_charge : ℕ := 53

def ana_charging_time : ℕ :=
  tablet_full_charge + (smartphone_full_charge / 2)

theorem ana_charging_time_proof :
  ana_charging_time = 66 := by
  sorry

end NUMINAMATH_CALUDE_ana_charging_time_proof_l1689_168992


namespace NUMINAMATH_CALUDE_s_equals_2012_l1689_168900

/-- S(n, k) is the number of coefficients in the expansion of (x+1)^n that are not divisible by k -/
def S (n k : ℕ) : ℕ := sorry

/-- Theorem stating that S(2012^2011, 2011) equals 2012 -/
theorem s_equals_2012 : S (2012^2011) 2011 = 2012 := by sorry

end NUMINAMATH_CALUDE_s_equals_2012_l1689_168900


namespace NUMINAMATH_CALUDE_harriett_us_dollars_l1689_168960

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def half_dollar_value : ℚ := 0.50
def dollar_coin_value : ℚ := 1.00

def num_quarters : ℕ := 23
def num_dimes : ℕ := 15
def num_nickels : ℕ := 17
def num_pennies : ℕ := 29
def num_half_dollars : ℕ := 6
def num_dollar_coins : ℕ := 10

def total_us_dollars : ℚ := 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value + 
  num_pennies * penny_value + 
  num_half_dollars * half_dollar_value + 
  num_dollar_coins * dollar_coin_value

theorem harriett_us_dollars : total_us_dollars = 21.39 := by
  sorry

end NUMINAMATH_CALUDE_harriett_us_dollars_l1689_168960


namespace NUMINAMATH_CALUDE_total_fish_l1689_168933

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 9) : 
  lilly_fish + rosy_fish = 19 := by
sorry

end NUMINAMATH_CALUDE_total_fish_l1689_168933


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1689_168938

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^4 + a^2 * b^2 + b^4 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1689_168938


namespace NUMINAMATH_CALUDE_min_additional_marbles_for_lisa_l1689_168999

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem min_additional_marbles_for_lisa : min_additional_marbles 12 34 = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_marbles_for_lisa_l1689_168999


namespace NUMINAMATH_CALUDE_fraction_of_girls_at_dance_l1689_168956

theorem fraction_of_girls_at_dance (
  dalton_total : ℕ) (dalton_ratio_boys : ℕ) (dalton_ratio_girls : ℕ)
  (berkeley_total : ℕ) (berkeley_ratio_boys : ℕ) (berkeley_ratio_girls : ℕ)
  (kingston_total : ℕ) (kingston_ratio_boys : ℕ) (kingston_ratio_girls : ℕ)
  (h1 : dalton_total = 300)
  (h2 : dalton_ratio_boys = 3)
  (h3 : dalton_ratio_girls = 2)
  (h4 : berkeley_total = 210)
  (h5 : berkeley_ratio_boys = 3)
  (h6 : berkeley_ratio_girls = 4)
  (h7 : kingston_total = 240)
  (h8 : kingston_ratio_boys = 5)
  (h9 : kingston_ratio_girls = 7)
  : (dalton_total * dalton_ratio_girls / (dalton_ratio_boys + dalton_ratio_girls) +
     berkeley_total * berkeley_ratio_girls / (berkeley_ratio_boys + berkeley_ratio_girls) +
     kingston_total * kingston_ratio_girls / (kingston_ratio_boys + kingston_ratio_girls)) /
    (dalton_total + berkeley_total + kingston_total) = 38 / 75 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_girls_at_dance_l1689_168956


namespace NUMINAMATH_CALUDE_percentage_problem_l1689_168937

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.3 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1689_168937


namespace NUMINAMATH_CALUDE_stream_top_width_l1689_168917

/-- 
Theorem: Given a trapezoidal cross-section of a stream with specified dimensions,
prove that the width at the top of the stream is 10 meters.
-/
theorem stream_top_width 
  (area : ℝ) 
  (depth : ℝ) 
  (bottom_width : ℝ) 
  (h_area : area = 640) 
  (h_depth : depth = 80) 
  (h_bottom : bottom_width = 6) :
  let top_width := (2 * area / depth) - bottom_width
  top_width = 10 := by
sorry

end NUMINAMATH_CALUDE_stream_top_width_l1689_168917


namespace NUMINAMATH_CALUDE_sphere_to_wire_length_l1689_168964

-- Define constants
def sphere_radius : ℝ := 12
def wire_radius : ℝ := 0.8

-- Define the theorem
theorem sphere_to_wire_length :
  let sphere_volume := (4/3) * Real.pi * (sphere_radius ^ 3)
  let wire_volume := Real.pi * (wire_radius ^ 2) * wire_length
  let wire_length := sphere_volume / (Real.pi * (wire_radius ^ 2))
  wire_length = 3600 := by sorry

end NUMINAMATH_CALUDE_sphere_to_wire_length_l1689_168964


namespace NUMINAMATH_CALUDE_berry_pie_theorem_l1689_168955

/-- Represents the amount of berries picked by each person -/
structure BerryPicker where
  strawberries : ℕ
  blueberries : ℕ
  raspberries : ℕ

/-- Represents the requirements for each type of pie -/
structure PieRequirements where
  strawberry : ℕ
  blueberry : ℕ
  raspberry : ℕ

/-- Calculates the maximum number of complete pies that can be made -/
def max_pies (christine : BerryPicker) (rachel : BerryPicker) (req : PieRequirements) : ℕ × ℕ × ℕ :=
  let total_strawberries := christine.strawberries + rachel.strawberries
  let total_blueberries := christine.blueberries + rachel.blueberries
  let total_raspberries := christine.raspberries + rachel.raspberries
  (total_strawberries / req.strawberry,
   total_blueberries / req.blueberry,
   total_raspberries / req.raspberry)

theorem berry_pie_theorem (christine : BerryPicker) (rachel : BerryPicker) (req : PieRequirements) :
  christine.strawberries = 10 ∧
  christine.blueberries = 8 ∧
  christine.raspberries = 20 ∧
  rachel.strawberries = 2 * christine.strawberries ∧
  rachel.blueberries = 2 * christine.blueberries ∧
  rachel.raspberries = christine.raspberries / 2 ∧
  req.strawberry = 3 ∧
  req.blueberry = 2 ∧
  req.raspberry = 4 →
  max_pies christine rachel req = (10, 12, 7) := by
  sorry

end NUMINAMATH_CALUDE_berry_pie_theorem_l1689_168955


namespace NUMINAMATH_CALUDE_initial_average_production_l1689_168901

theorem initial_average_production (n : ℕ) (A : ℝ) (today_production : ℝ) (new_average : ℝ)
  (h1 : n = 5)
  (h2 : today_production = 90)
  (h3 : new_average = 65)
  (h4 : (n * A + today_production) / (n + 1) = new_average) :
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l1689_168901


namespace NUMINAMATH_CALUDE_playground_area_l1689_168923

/-- Given a rectangular landscape with specific dimensions and a playground, 
    prove that the playground area is 3200 square meters. -/
theorem playground_area (length breadth : ℝ) (playground_area : ℝ) : 
  breadth = 8 * length →
  breadth = 480 →
  playground_area = (1 / 9) * (length * breadth) →
  playground_area = 3200 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l1689_168923


namespace NUMINAMATH_CALUDE_first_train_length_l1689_168986

-- Define the given constants
def train1_speed : ℝ := 80 -- km/h
def train2_speed : ℝ := 65 -- km/h
def clearing_time : ℝ := 7.199424046076314 -- seconds
def train2_length : ℝ := 180 -- meters

-- Define the theorem
theorem first_train_length :
  let relative_speed : ℝ := (train1_speed + train2_speed) * 1000 / 3600 -- Convert km/h to m/s
  let total_distance : ℝ := relative_speed * clearing_time
  let train1_length : ℝ := total_distance - train2_length
  train1_length = 110 := by sorry

end NUMINAMATH_CALUDE_first_train_length_l1689_168986


namespace NUMINAMATH_CALUDE_max_a_is_pi_over_four_l1689_168963

/-- If f(x) = cos x - sin x is a decreasing function on the interval [-a, a], 
    then the maximum value of a is π/4 -/
theorem max_a_is_pi_over_four (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.cos x - Real.sin x) →
  (∀ x y, -a ≤ x ∧ x < y ∧ y ≤ a → f y < f x) →
  a ≤ π / 4 ∧ ∀ b, (∀ x y, -b ≤ x ∧ x < y ∧ y ≤ b → f y < f x) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_is_pi_over_four_l1689_168963


namespace NUMINAMATH_CALUDE_factor_calculation_l1689_168922

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 8 →
  factor * (2 * initial_number + 9) = 75 →
  factor = 3 := by
sorry

end NUMINAMATH_CALUDE_factor_calculation_l1689_168922


namespace NUMINAMATH_CALUDE_beijing_to_lanzhou_distance_l1689_168990

/-- The distance from Beijing to Lanzhou, given the distances from Beijing to Lhasa (via Lanzhou) and from Lanzhou to Lhasa. -/
theorem beijing_to_lanzhou_distance 
  (beijing_to_lhasa : ℕ) 
  (lanzhou_to_lhasa : ℕ) 
  (h1 : beijing_to_lhasa = 3985)
  (h2 : lanzhou_to_lhasa = 2054) :
  beijing_to_lhasa - lanzhou_to_lhasa = 1931 :=
by sorry

end NUMINAMATH_CALUDE_beijing_to_lanzhou_distance_l1689_168990


namespace NUMINAMATH_CALUDE_daily_rental_cost_is_30_l1689_168978

/-- Represents a car rental with a daily rate and a per-mile rate. -/
structure CarRental where
  dailyRate : ℝ
  perMileRate : ℝ

/-- Calculates the total cost of renting a car for one day and driving a given distance. -/
def totalCost (rental : CarRental) (distance : ℝ) : ℝ :=
  rental.dailyRate + rental.perMileRate * distance

/-- Theorem: Given the specified conditions, the daily rental cost is 30 dollars. -/
theorem daily_rental_cost_is_30 (rental : CarRental)
    (h1 : rental.perMileRate = 0.18)
    (h2 : totalCost rental 250.0 = 75) :
    rental.dailyRate = 30 := by
  sorry

end NUMINAMATH_CALUDE_daily_rental_cost_is_30_l1689_168978


namespace NUMINAMATH_CALUDE_mukesh_travel_distance_l1689_168910

theorem mukesh_travel_distance : ∀ x : ℝ,
  (x / 90 - x / 120 = 4 / 15) →
  x = 96 := by
  sorry

end NUMINAMATH_CALUDE_mukesh_travel_distance_l1689_168910


namespace NUMINAMATH_CALUDE_base_conversion_and_sum_l1689_168998

-- Define the value of 537 in base 8
def base_8_value : ℕ := 5 * 8^2 + 3 * 8^1 + 7 * 8^0

-- Define the value of 1C2E in base 16, where C = 12 and E = 14
def base_16_value : ℕ := 1 * 16^3 + 12 * 16^2 + 2 * 16^1 + 14 * 16^0

-- Theorem statement
theorem base_conversion_and_sum :
  base_8_value + base_16_value = 7565 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_and_sum_l1689_168998


namespace NUMINAMATH_CALUDE_sqrt_real_range_l1689_168920

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 3 + x) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l1689_168920


namespace NUMINAMATH_CALUDE_missing_files_l1689_168996

/-- Proves that the number of missing files is 15 --/
theorem missing_files (total_files : ℕ) (afternoon_files : ℕ) : 
  total_files = 60 → 
  afternoon_files = 15 → 
  total_files - (total_files / 2 + afternoon_files) = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_files_l1689_168996


namespace NUMINAMATH_CALUDE_keith_seashells_l1689_168934

/-- Proves the number of seashells Keith found given the problem conditions -/
theorem keith_seashells (mary_shells : ℕ) (total_shells : ℕ) (cracked_shells : ℕ) :
  mary_shells = 2 →
  total_shells = 7 →
  cracked_shells = 9 →
  total_shells - mary_shells = 5 :=
by sorry

end NUMINAMATH_CALUDE_keith_seashells_l1689_168934


namespace NUMINAMATH_CALUDE_second_supply_cost_l1689_168928

def first_supply_cost : ℕ := 13
def total_budget : ℕ := 56
def remaining_budget : ℕ := 19

theorem second_supply_cost :
  total_budget - remaining_budget - first_supply_cost = 24 :=
by sorry

end NUMINAMATH_CALUDE_second_supply_cost_l1689_168928


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l1689_168919

theorem boys_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_children : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 22)
  (h6 : total_girls = 38)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l1689_168919


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1689_168913

/-- Given a bag with red and yellow balls, calculate the probability of drawing a red ball -/
theorem probability_of_red_ball (num_red : ℕ) (num_yellow : ℕ) :
  num_red = 6 → num_yellow = 3 →
  (num_red : ℚ) / (num_red + num_yellow : ℚ) = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1689_168913


namespace NUMINAMATH_CALUDE_sequence_formula_l1689_168940

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ+, a n * a (n + 1) = n / (n + 2)) :
  ∀ n : ℕ+, a n = n / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1689_168940


namespace NUMINAMATH_CALUDE_birdwatching_sites_l1689_168984

theorem birdwatching_sites (x : ℕ) : 
  (7 * x + 5 * x + 80) / (2 * x + 10) = 7 → x + x = 10 := by
  sorry

end NUMINAMATH_CALUDE_birdwatching_sites_l1689_168984


namespace NUMINAMATH_CALUDE_least_satisfying_number_l1689_168905

def is_multiple_of_50 (n : ℕ) : Prop := ∃ k : ℕ, n = 50 * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let digits := Nat.digits 10 n
  List.foldl (·*·) 1 digits

def satisfies_condition (n : ℕ) : Prop :=
  is_multiple_of_50 n ∧ 
  is_multiple_of_50 (digit_product n) ∧ 
  digit_product n > 0

theorem least_satisfying_number :
  satisfies_condition 5550 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < 5550 → ¬(satisfies_condition m) :=
sorry

end NUMINAMATH_CALUDE_least_satisfying_number_l1689_168905


namespace NUMINAMATH_CALUDE_art_class_problem_l1689_168979

theorem art_class_problem (total_students : ℕ) (total_kits : ℕ) (total_artworks : ℕ) 
  (h1 : total_students = 10)
  (h2 : total_kits = 20)
  (h3 : total_artworks = 35)
  (h4 : 2 * total_kits = total_students) -- 1 kit for 2 students
  (h5 : total_students % 2 = 0) -- Ensures even number of students for equal halves
  : ∃ x : ℕ, 
    x * (total_students / 2) + 4 * (total_students / 2) = total_artworks ∧ 
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_art_class_problem_l1689_168979


namespace NUMINAMATH_CALUDE_polynomial_degree_l1689_168930

/-- The degree of the polynomial (3x^5 + 2x^4 - x + 5)(4x^11 - 2x^8 + 5x^5 - 9) - (x^2 - 3)^9 is 18 -/
theorem polynomial_degree : ℕ := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l1689_168930


namespace NUMINAMATH_CALUDE_parabola_equation_l1689_168961

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in general form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The greatest common divisor of the absolute values of all coefficients is 1 -/
def coefficientsAreCoprime (p : Parabola) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs p.a) (Int.natAbs p.b)) (Int.natAbs p.c)) (Int.natAbs p.d)) (Int.natAbs p.e)) (Int.natAbs p.f) = 1

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (focus : Point) (directrix : Line) : 
  focus.x = 4 ∧ focus.y = 2 ∧ 
  directrix.a = 2 ∧ directrix.b = 5 ∧ directrix.c = 20 →
  ∃ (p : Parabola), 
    p.a = 25 ∧ p.b = -20 ∧ p.c = 4 ∧ p.d = -152 ∧ p.e = 84 ∧ p.f = -180 ∧
    p.a > 0 ∧
    coefficientsAreCoprime p :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1689_168961


namespace NUMINAMATH_CALUDE_cosine_sum_eleven_l1689_168962

theorem cosine_sum_eleven : 
  Real.cos (π / 11) - Real.cos (2 * π / 11) + Real.cos (3 * π / 11) - 
  Real.cos (4 * π / 11) + Real.cos (5 * π / 11) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_eleven_l1689_168962


namespace NUMINAMATH_CALUDE_factorization_equality_l1689_168976

theorem factorization_equality (a x y : ℝ) :
  5 * a * x^2 - 5 * a * y^2 = 5 * a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1689_168976


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_diameter_l1689_168907

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The diameter of the circumscribed circle of a cyclic quadrilateral -/
def circumscribedCircleDiameter (q : CyclicQuadrilateral) : ℝ := sorry

/-- Theorem: The diameter of the circumscribed circle of a cyclic quadrilateral 
    with side lengths 25, 39, 52, and 60 is 65 -/
theorem cyclic_quadrilateral_diameter :
  ∀ (q : CyclicQuadrilateral), 
    q.a = 25 ∧ q.b = 39 ∧ q.c = 52 ∧ q.d = 60 →
    circumscribedCircleDiameter q = 65 := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_diameter_l1689_168907


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_difference_l1689_168916

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := n + 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 32 * (1/2)^(n-1)

-- Define the sum of the first n terms of a_n - b_n
def T (n : ℕ) : ℚ := n * (n + 3) / 2 + 2^(6-n) - 64

theorem arithmetic_geometric_sum_difference 
  (h1 : a 1 = 2) 
  (h2 : S 5 = 20) 
  (h3 : a 4 + b 4 = 9) :
  ∀ n : ℕ, T n = (S n) - (b 1 * (1 - (1/2)^n) / (1 - 1/2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_difference_l1689_168916


namespace NUMINAMATH_CALUDE_vector_OC_on_angle_bisector_l1689_168989

/-- Given points A and B, and a point C on the angle bisector of ∠AOB with |OC| = 2,
    prove that OC is equal to the specified vector. -/
theorem vector_OC_on_angle_bisector (A B C : ℝ × ℝ) : 
  A = (0, 1) →
  B = (-3, 4) →
  C.1^2 + C.2^2 = 4 →  -- |OC| = 2
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    C = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
    t * (A.1^2 + A.2^2) = (1 - t) * (B.1^2 + B.2^2)) →  -- C is on the angle bisector
  C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_OC_on_angle_bisector_l1689_168989


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1689_168981

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 7 ↔ (x : ℚ) / 4 + 3 / 7 < 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1689_168981


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1689_168953

theorem repeating_decimal_sum (x : ℚ) : x = 23 / 99 → (x.num + x.den = 122) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1689_168953


namespace NUMINAMATH_CALUDE_least_integer_satisfying_conditions_l1689_168985

theorem least_integer_satisfying_conditions : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_conditions_l1689_168985


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_range_l1689_168944

/-- A polynomial of the form x^4 + bx^3 + x^2 + bx + 1 -/
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + b*x^3 + x^2 + b*x + 1

/-- The polynomial has at least one real root -/
def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, polynomial b x = 0

/-- Theorem: The polynomial has at least one real root if and only if b is in [-3/4, 0) -/
theorem polynomial_real_root_iff_b_in_range :
  ∀ b : ℝ, has_real_root b ↔ -3/4 ≤ b ∧ b < 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_range_l1689_168944


namespace NUMINAMATH_CALUDE_expression_evaluation_l1689_168987

theorem expression_evaluation (a b c : ℚ) : 
  a = 6 → 
  b = 2 * a - 1 → 
  c = 2 * b - 30 → 
  a + 2 ≠ 0 → 
  b - 3 ≠ 0 → 
  c + 7 ≠ 0 → 
  (a + 3) / (a + 2) * (b + 5) / (b - 3) * (c + 10) / (c + 7) = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1689_168987


namespace NUMINAMATH_CALUDE_intersection_characterization_l1689_168950

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

theorem intersection_characterization :
  ∀ x : ℝ, x ∈ (M ∩ N) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_characterization_l1689_168950


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1689_168977

/-- Given two points A(a,3) and B(4,b) that are symmetric with respect to the y-axis,
    prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (a, 3) ∧ B = (4, b) ∧ 
    (A.1 = -B.1 ∧ A.2 = B.2)) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1689_168977


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l1689_168971

/-- Proves the number of computer literate female employees in an office -/
theorem computer_literate_female_employees
  (total_employees : ℕ)
  (female_percentage : ℚ)
  (male_computer_literate_percentage : ℚ)
  (total_computer_literate_percentage : ℚ)
  (h_total : total_employees = 1200)
  (h_female : female_percentage = 60 / 100)
  (h_male_cl : male_computer_literate_percentage = 50 / 100)
  (h_total_cl : total_computer_literate_percentage = 62 / 100) :
  ↑(total_employees : ℚ) * female_percentage * total_computer_literate_percentage -
  (↑(total_employees : ℚ) * (1 - female_percentage) * male_computer_literate_percentage) = 504 :=
sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l1689_168971


namespace NUMINAMATH_CALUDE_ab_value_l1689_168968

theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 + b^2 = 3) (h2 : a^4 + b^4 = 15/4) : a * b = Real.sqrt 42 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1689_168968


namespace NUMINAMATH_CALUDE_alpha_range_l1689_168947

theorem alpha_range (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos α - Real.sin α = Real.tan α) : 
  α ∈ Set.Ioo 0 (Real.pi / 6) := by
sorry

end NUMINAMATH_CALUDE_alpha_range_l1689_168947


namespace NUMINAMATH_CALUDE_f_monotone_increasing_on_neg_reals_l1689_168967

-- Define the function f(x) = -|x|
def f (x : ℝ) : ℝ := -abs x

-- State the theorem
theorem f_monotone_increasing_on_neg_reals :
  MonotoneOn f (Set.Iic 0) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_on_neg_reals_l1689_168967


namespace NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l1689_168904

/-- Given a set of boxes with pens and pencils, this theorem proves
    the number of boxes containing neither pens nor pencils. -/
theorem boxes_with_neither_pens_nor_pencils
  (total_boxes : ℕ)
  (pencil_boxes : ℕ)
  (pen_boxes : ℕ)
  (both_boxes : ℕ)
  (h1 : total_boxes = 10)
  (h2 : pencil_boxes = 6)
  (h3 : pen_boxes = 3)
  (h4 : both_boxes = 2)
  : total_boxes - (pencil_boxes + pen_boxes - both_boxes) = 3 := by
  sorry

#check boxes_with_neither_pens_nor_pencils

end NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l1689_168904


namespace NUMINAMATH_CALUDE_expected_twos_is_half_l1689_168952

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1/6

/-- The probability of not rolling a 2 on a standard die -/
def prob_not_two : ℚ := 1 - prob_two

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 2's when rolling three standard dice -/
def expected_twos : ℚ :=
  0 * (prob_not_two ^ num_dice) +
  1 * (num_dice * prob_two * prob_not_two ^ 2) +
  2 * (num_dice * prob_two ^ 2 * prob_not_two) +
  3 * (prob_two ^ num_dice)

theorem expected_twos_is_half : expected_twos = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_twos_is_half_l1689_168952


namespace NUMINAMATH_CALUDE_tickets_to_be_sold_l1689_168949

theorem tickets_to_be_sold (total : ℕ) (jude andrea sandra : ℕ) : 
  total = 100 → 
  andrea = 2 * jude → 
  sandra = jude / 2 + 4 → 
  jude = 16 → 
  total - (jude + andrea + sandra) = 40 := by
sorry

end NUMINAMATH_CALUDE_tickets_to_be_sold_l1689_168949


namespace NUMINAMATH_CALUDE_toms_age_difference_l1689_168958

theorem toms_age_difference (sister_age : ℕ) : 
  sister_age + 9 = 14 →
  2 * sister_age - 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_toms_age_difference_l1689_168958


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1689_168921

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) :
  (6 - x) * x ≤ 9 ∧ ∃ y, 0 < y ∧ y < 6 ∧ (6 - y) * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1689_168921


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_5_l1689_168926

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a}
def B : Set ℝ := {x | x^2 - 5*x < 0}

-- State the theorem
theorem intersection_equality_implies_a_geq_5 (a : ℝ) :
  A a ∩ B = B → a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_5_l1689_168926


namespace NUMINAMATH_CALUDE_square_area_is_16_l1689_168918

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the horizontal line
def horizontal_line : ℝ := 3

-- Theorem statement
theorem square_area_is_16 : ∃ (x₁ x₂ : ℝ),
  x₁ ≠ x₂ ∧
  parabola x₁ = horizontal_line ∧
  parabola x₂ = horizontal_line ∧
  (x₂ - x₁)^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_square_area_is_16_l1689_168918


namespace NUMINAMATH_CALUDE_art_students_count_l1689_168969

/-- Represents the number of students taking art in a high school -/
def students_taking_art (total students_taking_music students_taking_both students_taking_neither : ℕ) : ℕ :=
  total - students_taking_music - students_taking_neither + students_taking_both

/-- Theorem stating that 10 students are taking art given the conditions -/
theorem art_students_count :
  students_taking_art 500 30 10 470 = 10 := by
  sorry

end NUMINAMATH_CALUDE_art_students_count_l1689_168969


namespace NUMINAMATH_CALUDE_smallest_student_count_l1689_168931

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- Checks if the given grade counts satisfy the required ratios --/
def satisfies_ratios (gc : GradeCount) : Prop :=
  4 * gc.ninth = 3 * gc.tenth ∧ 6 * gc.tenth = 5 * gc.eleventh

/-- The total number of students across the three grades --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- Theorem stating that 59 is the smallest number of students satisfying the ratios --/
theorem smallest_student_count : 
  ∃ (gc : GradeCount), satisfies_ratios gc ∧ total_students gc = 59 ∧
  ∀ (gc' : GradeCount), satisfies_ratios gc' → total_students gc' ≥ 59 :=
sorry

end NUMINAMATH_CALUDE_smallest_student_count_l1689_168931


namespace NUMINAMATH_CALUDE_donut_selections_l1689_168909

theorem donut_selections (n k : ℕ) (hn : n = 5) (hk : k = 4) : 
  Nat.choose (n + k - 1) (k - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l1689_168909


namespace NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l1689_168980

/-- Two unit vectors have equal squared magnitudes -/
theorem unit_vectors_equal_squared_magnitude
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]
  (a b : n) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  ‖a‖^2 = ‖b‖^2 := by sorry

end NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l1689_168980


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1689_168942

theorem complex_magnitude_problem (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1689_168942


namespace NUMINAMATH_CALUDE_no_valid_bracelet_arrangement_l1689_168966

/-- The number of bracelets Elizabeth has -/
def n : ℕ := 100

/-- The number of bracelets Elizabeth wears each day -/
def k : ℕ := 3

/-- Represents a valid arrangement of bracelets -/
structure BraceletArrangement where
  days : ℕ
  worn : Fin days → Finset (Fin n)
  size_correct : ∀ d, (worn d).card = k
  all_pairs_once : ∀ i j, i < j → ∃! d, i ∈ worn d ∧ j ∈ worn d

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_bracelet_arrangement : ¬ ∃ arr : BraceletArrangement, True := by
  sorry

end NUMINAMATH_CALUDE_no_valid_bracelet_arrangement_l1689_168966


namespace NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l1689_168946

theorem not_p_or_not_q_is_true :
  ∀ (a b c : ℝ),
  let p := ∀ (a b c : ℝ), a > b → a + c > b + c
  let q := ∀ (a b c : ℝ), a > b ∧ b > 0 → a * c > b * c
  ¬p ∨ ¬q := by sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_is_true_l1689_168946


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1689_168927

theorem cube_volume_ratio :
  let cube1_edge : ℚ := 8
  let cube2_edge : ℚ := 16
  let volume_ratio := (cube1_edge ^ 3) / (cube2_edge ^ 3)
  volume_ratio = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1689_168927


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_equality_l1689_168925

theorem arithmetic_sequence_sum_equality (n : ℕ) (hn : n > 0) : 
  (n * (2 * 3 + (n - 1) * 4)) / 2 = (n * (2 * 23 + (n - 1) * 4)) / 2 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_equality_l1689_168925


namespace NUMINAMATH_CALUDE_scaled_tetrahedron_volume_ratio_l1689_168997

-- Define a regular tetrahedron
def RegularTetrahedron : Type := Unit

-- Define a function to scale down coordinates
def scaleDown (t : RegularTetrahedron) : RegularTetrahedron := sorry

-- Define a function to calculate the volume of a tetrahedron
def volume (t : RegularTetrahedron) : ℝ := sorry

-- Theorem statement
theorem scaled_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) : 
  volume (scaleDown t) / volume t = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_scaled_tetrahedron_volume_ratio_l1689_168997


namespace NUMINAMATH_CALUDE_min_balls_to_draw_theorem_l1689_168951

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to ensure at least 15 of one color -/
def minBallsToDraw (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_to_draw_theorem (counts : BallCounts) 
  (h1 : counts.red = 28)
  (h2 : counts.green = 20)
  (h3 : counts.yellow = 13)
  (h4 : counts.blue = 19)
  (h5 : counts.white = 11)
  (h6 : counts.black = 9)
  (h_total : counts.red + counts.green + counts.yellow + counts.blue + counts.white + counts.black = 100) :
  minBallsToDraw counts = 76 :=
sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_theorem_l1689_168951


namespace NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l1689_168914

/-- The function f(x) defined as x^2 + 3x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

/-- Theorem stating that the largest value of c such that -2 is in the range of f(x) = x^2 + 3x + c is 1/4 -/
theorem largest_c_for_negative_two_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = -2) → d ≤ c) ∧
  (∃ (x : ℝ), f (1/4) x = -2) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l1689_168914


namespace NUMINAMATH_CALUDE_lucy_grocery_shopping_l1689_168974

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The number of cans of soup Lucy bought -/
def soup : ℕ := 28

/-- The number of boxes of cereals Lucy bought -/
def cereals : ℕ := 5

/-- The number of packs of crackers Lucy bought -/
def crackers : ℕ := 45

/-- The total number of packs and boxes Lucy bought -/
def total_packs_and_boxes : ℕ := cookies + noodles + cereals + crackers

theorem lucy_grocery_shopping :
  total_packs_and_boxes = 78 := by sorry

end NUMINAMATH_CALUDE_lucy_grocery_shopping_l1689_168974


namespace NUMINAMATH_CALUDE_complex_subtraction_l1689_168993

theorem complex_subtraction (i : ℂ) (h : i^2 = -1) :
  (5 - 3*i) - (7 - 7*i) = -2 + 4*i :=
sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1689_168993


namespace NUMINAMATH_CALUDE_card_sorting_moves_l1689_168973

theorem card_sorting_moves (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) : ℕ := by
  let card_label (i : ℕ) : ℕ := (i + k - 1) % n + 1
  let min_moves := n - Nat.gcd n k
  sorry

#check card_sorting_moves

end NUMINAMATH_CALUDE_card_sorting_moves_l1689_168973


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1689_168936

/-- A square inscribed in a semicircle with radius 1 -/
structure InscribedSquare where
  /-- The center of the semicircle -/
  center : ℝ × ℝ
  /-- The vertices of the square -/
  vertices : Fin 4 → ℝ × ℝ
  /-- Two vertices are on the semicircle -/
  on_semicircle : ∃ (i j : Fin 4), i ≠ j ∧
    (vertices i).1^2 + (vertices i).2^2 = 1 ∧
    (vertices j).1^2 + (vertices j).2^2 = 1 ∧
    (vertices i).2 ≥ 0 ∧ (vertices j).2 ≥ 0
  /-- Two vertices are on the diameter -/
  on_diameter : ∃ (i j : Fin 4), i ≠ j ∧
    (vertices i).2 = 0 ∧ (vertices j).2 = 0 ∧
    abs ((vertices i).1 - (vertices j).1) = 2
  /-- The vertices form a square -/
  is_square : ∀ (i j : Fin 4), i ≠ j →
    (vertices i).1^2 + (vertices i).2^2 =
    (vertices j).1^2 + (vertices j).2^2

/-- The area of an inscribed square is 4/5 -/
theorem inscribed_square_area (s : InscribedSquare) :
  let side_length := abs ((s.vertices 0).1 - (s.vertices 1).1)
  side_length^2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1689_168936


namespace NUMINAMATH_CALUDE_ferris_wheel_small_seat_capacity_l1689_168912

/-- The number of small seats on the Ferris wheel -/
def small_seats : ℕ := 2

/-- The number of people each small seat can hold -/
def small_seat_capacity : ℕ := 14

/-- The total number of people who can ride on small seats -/
def total_small_seat_riders : ℕ := small_seats * small_seat_capacity

theorem ferris_wheel_small_seat_capacity : total_small_seat_riders = 28 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_small_seat_capacity_l1689_168912


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l1689_168932

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | |x - 2| ≥ 4 - |x - 4|} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 1) :
  ({x : ℝ | |f a (2*x + a) - 2*f a x| ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l1689_168932


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1689_168948

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = 3 ∧ m = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1689_168948


namespace NUMINAMATH_CALUDE_pens_problem_l1689_168957

theorem pens_problem (initial_pens : ℕ) (final_pens : ℕ) (sharon_pens : ℕ) 
  (h1 : initial_pens = 20)
  (h2 : final_pens = 65)
  (h3 : sharon_pens = 19) :
  ∃ (mike_pens : ℕ), 2 * (initial_pens + mike_pens) - sharon_pens = final_pens ∧ mike_pens = 22 := by
  sorry

end NUMINAMATH_CALUDE_pens_problem_l1689_168957


namespace NUMINAMATH_CALUDE_transaction_error_l1689_168941

theorem transaction_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 →
  (100 * y + x) - (100 * x + y) = 5616 →
  y = x + 56 := by
sorry

end NUMINAMATH_CALUDE_transaction_error_l1689_168941


namespace NUMINAMATH_CALUDE_expression_equality_l1689_168903

theorem expression_equality (x y z : ℝ) : 
  (2 * x - (3 * y - 4 * z)) - ((2 * x - 3 * y) - 5 * z) = 9 * z := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1689_168903


namespace NUMINAMATH_CALUDE_flowers_to_grandma_vs_mom_l1689_168915

theorem flowers_to_grandma_vs_mom (total : ℕ) (to_mom : ℕ) (in_vase : ℕ) :
  total = 52 →
  to_mom = 15 →
  in_vase = 16 →
  total - to_mom - in_vase - to_mom = 6 := by
  sorry

end NUMINAMATH_CALUDE_flowers_to_grandma_vs_mom_l1689_168915


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l1689_168975

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem max_sum_arithmetic_sequence 
  (a d : ℤ) 
  (h1 : a + 16 * d = 52) 
  (h2 : a + 29 * d = 13) :
  ∃ n : ℕ, 
    (arithmetic_sequence a d n > 0) ∧ 
    (arithmetic_sequence a d (n + 1) ≤ 0) ∧
    (∀ m : ℕ, m > n → arithmetic_sequence a d m ≤ 0) ∧
    (sum_arithmetic_sequence a d n = 1717) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l1689_168975


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1689_168935

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1689_168935


namespace NUMINAMATH_CALUDE_starting_player_winning_strategy_l1689_168929

/-- Represents the color of a disk -/
inductive DiskColor
| Red
| Blue

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the state of the chessboard -/
def BoardState (n : Nat) := Position → DiskColor

/-- Checks if a position is within the bounds of the board -/
def isValidPosition (n : Nat) (pos : Position) : Prop :=
  pos.row < n ∧ pos.col < n

/-- Represents a move in the game -/
structure Move :=
  (pos : Position)

/-- Applies a move to the board state -/
def applyMove (n : Nat) (state : BoardState n) (move : Move) : BoardState n :=
  sorry

/-- Checks if a player can make a move -/
def canMove (n : Nat) (state : BoardState n) : Prop :=
  ∃ (move : Move), isValidPosition n move.pos ∧ state move.pos = DiskColor.Blue

/-- Defines a winning strategy for the starting player -/
def hasWinningStrategy (n : Nat) (initialState : BoardState n) : Prop :=
  sorry

/-- The main theorem stating the winning condition for the starting player -/
theorem starting_player_winning_strategy (n : Nat) (initialState : BoardState n) :
  hasWinningStrategy n initialState ↔ 
  initialState ⟨n - 1, n - 1⟩ = DiskColor.Blue :=
sorry

end NUMINAMATH_CALUDE_starting_player_winning_strategy_l1689_168929


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1689_168988

-- Define the function f(x) = x³ - x² + x + 1
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1689_168988


namespace NUMINAMATH_CALUDE_james_investment_l1689_168994

theorem james_investment (initial_balance : ℝ) (weekly_investment : ℝ) (weeks : ℕ) (windfall_percentage : ℝ) : 
  initial_balance = 250000 ∧ 
  weekly_investment = 2000 ∧ 
  weeks = 52 ∧ 
  windfall_percentage = 0.5 →
  let final_balance := initial_balance + weekly_investment * weeks
  let windfall := windfall_percentage * final_balance
  final_balance + windfall = 531000 := by
sorry

end NUMINAMATH_CALUDE_james_investment_l1689_168994
