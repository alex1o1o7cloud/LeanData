import Mathlib

namespace NUMINAMATH_CALUDE_parabola_properties_l1982_198232

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - 4*x + 3)

-- Define the theorem
theorem parabola_properties (a : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry
  (∀ x : ℝ, parabola a (2 + x) = parabola a (2 - x)) ∧
  -- 2. When PQ = QA, C is at (0, 3)
  (∃ m : ℝ, m > 2 ∧ parabola a m = 3 ∧ 3 = m - 1 → parabola a 0 = 3) ∧
  -- 3. When PQ > QA, 3 < m < 4
  (∀ m : ℝ, m > 2 ∧ parabola a m = 3 ∧ 3 > m - 1 → 3 < m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1982_198232


namespace NUMINAMATH_CALUDE_spinning_tops_theorem_l1982_198202

/-- The number of spinning tops obtained from gift boxes --/
def spinning_tops_count (red_price yellow_price : ℕ) (red_tops yellow_tops : ℕ) 
  (total_spent total_boxes : ℕ) : ℕ :=
  let red_boxes := (total_spent - yellow_price * total_boxes) / (red_price - yellow_price)
  let yellow_boxes := total_boxes - red_boxes
  red_boxes * red_tops + yellow_boxes * yellow_tops

/-- Theorem stating the number of spinning tops obtained --/
theorem spinning_tops_theorem : 
  spinning_tops_count 5 9 3 5 600 72 = 336 := by
  sorry

#eval spinning_tops_count 5 9 3 5 600 72

end NUMINAMATH_CALUDE_spinning_tops_theorem_l1982_198202


namespace NUMINAMATH_CALUDE_parallel_planes_perpendicular_line_l1982_198248

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_perpendicular_line 
  (α β : Plane) (m : Line) 
  (h1 : parallel α β) 
  (h2 : perpendicular m α) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_perpendicular_line_l1982_198248


namespace NUMINAMATH_CALUDE_multiplication_associative_l1982_198200

theorem multiplication_associative (x y z : ℝ) : (x * y) * z = x * (y * z) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_associative_l1982_198200


namespace NUMINAMATH_CALUDE_omitted_number_proof_l1982_198201

/-- Sequence of even numbers starting from 2 -/
def evenSeq (n : ℕ) : ℕ := 2 * n

/-- Sum of even numbers from 2 to 2n -/
def evenSum (n : ℕ) : ℕ := n * (n + 1)

/-- The incorrect sum obtained -/
def incorrectSum : ℕ := 2014

/-- The omitted number -/
def omittedNumber : ℕ := 56

theorem omitted_number_proof :
  ∃ n : ℕ, evenSum n - incorrectSum = omittedNumber ∧
  evenSeq (n + 1) = omittedNumber :=
sorry

end NUMINAMATH_CALUDE_omitted_number_proof_l1982_198201


namespace NUMINAMATH_CALUDE_special_function_property_l1982_198244

/-- A continuous function f: ℝ → ℝ satisfying f(x) · f(f(x)) = 1 for all real x, and f(1000) = 999 -/
def special_function (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x : ℝ, f x * f (f x) = 1) ∧
  f 1000 = 999

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) : 
  f 500 = 1 / 500 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1982_198244


namespace NUMINAMATH_CALUDE_erroneous_product_theorem_l1982_198235

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem erroneous_product_theorem (a b : ℕ) (h1 : is_two_digit a) (h2 : reverse_digits a * b = 180) :
  a * b = 315 ∨ a * b = 810 := by
  sorry

end NUMINAMATH_CALUDE_erroneous_product_theorem_l1982_198235


namespace NUMINAMATH_CALUDE_monotonic_sequence_bound_l1982_198263

theorem monotonic_sequence_bound (b : ℝ) :
  (∀ n : ℕ, (n + 1)^2 + b*(n + 1) > n^2 + b*n) →
  b > -3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_sequence_bound_l1982_198263


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1982_198282

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  lg (x^2) > (lg x)^2 ∧ (lg x)^2 > lg (lg x) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1982_198282


namespace NUMINAMATH_CALUDE_sample_size_is_30_l1982_198212

/-- Represents the company's employee data and sampling information -/
structure CompanyData where
  total_employees : ℕ
  young_employees : ℕ
  sample_young : ℕ
  h_young_le_total : young_employees ≤ total_employees

/-- Calculates the sample size based on stratified sampling -/
def calculate_sample_size (data : CompanyData) : ℕ :=
  (data.sample_young * data.total_employees) / data.young_employees

/-- Proves that the sample size is 30 given the specific company data -/
theorem sample_size_is_30 (data : CompanyData) 
  (h_total : data.total_employees = 900)
  (h_young : data.young_employees = 450)
  (h_sample_young : data.sample_young = 15) :
  calculate_sample_size data = 30 := by
  sorry

#eval calculate_sample_size ⟨900, 450, 15, by norm_num⟩

end NUMINAMATH_CALUDE_sample_size_is_30_l1982_198212


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_l1982_198265

theorem x_gt_one_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧
  (∃ x : ℝ, x^2 > x ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_l1982_198265


namespace NUMINAMATH_CALUDE_values_equal_l1982_198272

/-- The value of the expression at point C -/
def value_at_C : ℝ := 5 * 5 + 6 * 8.73

/-- The value of the expression at point D -/
def value_at_D : ℝ := 105

/-- Theorem stating that the values at points C and D are equal -/
theorem values_equal : value_at_C = value_at_D := by
  sorry

#eval value_at_C
#eval value_at_D

end NUMINAMATH_CALUDE_values_equal_l1982_198272


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1982_198231

theorem trig_expression_simplification (θ : ℝ) :
  (Real.tan (2 * Real.pi - θ) * Real.sin (-2 * Real.pi - θ) * Real.cos (6 * Real.pi - θ)) /
  (Real.cos (θ - Real.pi) * Real.sin (5 * Real.pi + θ)) = Real.tan θ :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l1982_198231


namespace NUMINAMATH_CALUDE_cos_negative_750_degrees_l1982_198273

theorem cos_negative_750_degrees : Real.cos ((-750 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_750_degrees_l1982_198273


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l1982_198209

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first five terms of a sequence -/
def SumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5

/-- The sum of the squares of the first five terms of a sequence -/
def SumSquaresFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2

/-- The alternating sum of the first five terms of a sequence -/
def AlternatingSumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 - a 2 + a 3 - a 4 + a 5

theorem arithmetic_geometric_sequence_property (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  SumFirstFive a = 3 →
  SumSquaresFirstFive a = 12 →
  AlternatingSumFirstFive a = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l1982_198209


namespace NUMINAMATH_CALUDE_bus_distance_calculation_l1982_198245

/-- Represents a round trip journey with walking and bus ride components. -/
structure Journey where
  total_distance : ℕ
  walking_distance : ℕ
  bus_distance : ℕ

/-- 
Theorem: If a person travels a total of 50 blocks in a round trip, 
where they walk 5 blocks at the beginning and end of each leg of the trip, 
then the distance traveled by bus in one direction is 20 blocks.
-/
theorem bus_distance_calculation (j : Journey) 
  (h1 : j.total_distance = 50)
  (h2 : j.walking_distance = 5) : 
  j.bus_distance = 20 := by
  sorry

#check bus_distance_calculation

end NUMINAMATH_CALUDE_bus_distance_calculation_l1982_198245


namespace NUMINAMATH_CALUDE_javier_to_anna_fraction_l1982_198292

/-- Represents the number of stickers each person has -/
structure StickerCount where
  lee : ℕ
  anna : ℕ
  javier : ℕ

/-- Calculates the fraction of stickers Javier should give to Anna -/
def fraction_to_anna (initial : StickerCount) (final : StickerCount) : ℚ :=
  (final.anna - initial.anna : ℚ) / initial.javier

/-- Theorem stating that Javier should give 0 fraction of his stickers to Anna -/
theorem javier_to_anna_fraction (l : ℕ) : 
  let initial := StickerCount.mk l (3 * l) (12 * l)
  let final := StickerCount.mk (2 * l) (3 * l) (6 * l)
  fraction_to_anna initial final = 0 := by
  sorry

#check javier_to_anna_fraction

end NUMINAMATH_CALUDE_javier_to_anna_fraction_l1982_198292


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l1982_198281

theorem swimming_pool_volume :
  let shallow_width : ℝ := 9
  let shallow_length : ℝ := 12
  let shallow_depth : ℝ := 1
  let deep_width : ℝ := 15
  let deep_length : ℝ := 18
  let deep_depth : ℝ := 4
  let island_width : ℝ := 3
  let island_length : ℝ := 6
  let island_height : ℝ := 1
  let shallow_volume := shallow_width * shallow_length * shallow_depth
  let deep_volume := deep_width * deep_length * deep_depth
  let island_volume := island_width * island_length * island_height
  let total_volume := shallow_volume + deep_volume - island_volume
  total_volume = 1170 := by
sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l1982_198281


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l1982_198205

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

theorem tangent_slope_angle_at_zero :
  let slope := (deriv f) 0
  Real.arctan slope = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l1982_198205


namespace NUMINAMATH_CALUDE_smallest_four_digit_non_divisor_l1982_198222

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

theorem smallest_four_digit_non_divisor :
  (∀ m : ℕ, 1000 ≤ m → m < 1005 → (product_of_first_n m)^2 % (sum_of_first_n m) = 0) ∧
  (product_of_first_n 1005)^2 % (sum_of_first_n 1005) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_non_divisor_l1982_198222


namespace NUMINAMATH_CALUDE_sum_coefficients_without_x_cubed_l1982_198240

theorem sum_coefficients_without_x_cubed : 
  let n : ℕ := 5
  let all_coeff_sum : ℕ := 2^n
  let x_cubed_coeff : ℕ := n.choose 3
  all_coeff_sum - x_cubed_coeff = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_coefficients_without_x_cubed_l1982_198240


namespace NUMINAMATH_CALUDE_fibonacci_fourth_term_divisible_by_three_l1982_198225

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_fourth_term_divisible_by_three (k : ℕ) :
  3 ∣ fibonacci (4 * k) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_fourth_term_divisible_by_three_l1982_198225


namespace NUMINAMATH_CALUDE_intersection_A_B_l1982_198227

def A : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

theorem intersection_A_B : A ∩ B = {x | -Real.pi < x ∧ x < 0 ∨ Real.pi < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1982_198227


namespace NUMINAMATH_CALUDE_only_7_3_1_wins_for_second_player_l1982_198217

/-- Represents a wall configuration in the game --/
structure WallConfig :=
  (walls : List Nat)

/-- Calculates the nim-value of a single wall --/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values --/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a given configuration is a winning position for the second player --/
def isWinningForSecondPlayer (config : WallConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The main theorem stating that (7,3,1) is the only winning configuration for the second player --/
theorem only_7_3_1_wins_for_second_player :
  let configs := [
    WallConfig.mk [7, 1, 1],
    WallConfig.mk [7, 2, 1],
    WallConfig.mk [7, 2, 2],
    WallConfig.mk [7, 3, 1],
    WallConfig.mk [7, 3, 2]
  ]
  ∀ config ∈ configs, isWinningForSecondPlayer config ↔ config = WallConfig.mk [7, 3, 1] :=
  sorry

end NUMINAMATH_CALUDE_only_7_3_1_wins_for_second_player_l1982_198217


namespace NUMINAMATH_CALUDE_potato_fetch_time_l1982_198211

-- Define the constants
def football_fields : ℕ := 6
def yards_per_field : ℕ := 200
def feet_per_yard : ℕ := 3
def dog_speed : ℕ := 400  -- feet per minute

-- Define the theorem
theorem potato_fetch_time :
  let total_distance : ℕ := football_fields * yards_per_field * feet_per_yard
  let fetch_time : ℕ := total_distance / dog_speed
  fetch_time = 9 := by sorry

end NUMINAMATH_CALUDE_potato_fetch_time_l1982_198211


namespace NUMINAMATH_CALUDE_notebook_distribution_l1982_198242

theorem notebook_distribution (S : ℕ) 
  (h1 : S > 0)
  (h2 : S * (S / 8) = (S / 2) * 16) : 
  S * (S / 8) = 512 := by
sorry

end NUMINAMATH_CALUDE_notebook_distribution_l1982_198242


namespace NUMINAMATH_CALUDE_function_value_at_two_l1982_198213

/-- Given a function f(x) = x^5 + px^3 + qx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (p q : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + p*x^3 + q*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1982_198213


namespace NUMINAMATH_CALUDE_smallest_base_for_27_l1982_198297

theorem smallest_base_for_27 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(x^2 ≤ 27 ∧ 27 < x^3)) ∧
  (b^2 ≤ 27 ∧ 27 < b^3) := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_27_l1982_198297


namespace NUMINAMATH_CALUDE_odd_function_property_l1982_198275

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_property :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x - 1) →    -- f(x) = x - 1 for x > 0
  (∀ x < 0, f x * f (-x) ≤ 0) -- f(x)f(-x) ≤ 0 for x < 0
:= by sorry

end NUMINAMATH_CALUDE_odd_function_property_l1982_198275


namespace NUMINAMATH_CALUDE_distance_to_origin_l1982_198260

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) 
  (h2 : x = 2 + 2 * Real.sqrt 30) 
  (h3 : Real.sqrt ((x - 2)^2 + (y - 8)^2) = 13) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1982_198260


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l1982_198279

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x - 4| - |2*x - 6|

-- Define the domain
def domain : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }

-- State the theorem
theorem sum_of_max_and_min_is_two :
  ∃ (max min : ℝ), 
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max + min = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_two_l1982_198279


namespace NUMINAMATH_CALUDE_game_winner_l1982_198250

/-- Represents the possible moves in the game -/
inductive Move
| one
| two
| three

/-- Represents a player in the game -/
inductive Player
| first
| second

/-- Defines the game state -/
structure GameState :=
  (position : Nat)
  (currentPlayer : Player)

/-- Determines if a move is valid given the current position -/
def isValidMove (pos : Nat) (move : Move) : Bool :=
  match move with
  | Move.one => pos ≥ 1
  | Move.two => pos ≥ 2
  | Move.three => pos ≥ 3

/-- Applies a move to the current position -/
def applyMove (pos : Nat) (move : Move) : Nat :=
  match move with
  | Move.one => pos - 1
  | Move.two => pos - 2
  | Move.three => pos - 3

/-- Switches the current player -/
def switchPlayer (player : Player) : Player :=
  match player with
  | Player.first => Player.second
  | Player.second => Player.first

/-- Determines the winner given an initial position -/
def winningPlayer (initialPos : Nat) : Player :=
  if initialPos = 4 ∨ initialPos = 8 ∨ initialPos = 12 then
    Player.second
  else
    Player.first

/-- The main theorem to prove -/
theorem game_winner (initialPos : Nat) :
  initialPos ≤ 14 →
  winningPlayer initialPos = Player.second ↔ (initialPos = 4 ∨ initialPos = 8 ∨ initialPos = 12) :=
by sorry

end NUMINAMATH_CALUDE_game_winner_l1982_198250


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1982_198261

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  (x - a)^3 / ((a - b) * (a - c)) + (x - b)^3 / ((b - a) * (b - c)) + (x - c)^3 / ((c - a) * (c - b)) =
  a + b + c - 3 * x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1982_198261


namespace NUMINAMATH_CALUDE_evaluate_g_l1982_198299

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem evaluate_g : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l1982_198299


namespace NUMINAMATH_CALUDE_rectangle_length_l1982_198206

/-- Proves that a rectangle with area 6 m² and width 150 cm has length 400 cm -/
theorem rectangle_length (area : ℝ) (width_cm : ℝ) (length_cm : ℝ) : 
  area = 6 → 
  width_cm = 150 → 
  area = (width_cm / 100) * (length_cm / 100) → 
  length_cm = 400 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l1982_198206


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1982_198298

/-- Given a simple interest loan where:
    - The interest after 10 years is 1500
    - The principal amount is 1250
    Prove that the interest rate is 12% --/
theorem simple_interest_rate : 
  ∀ (rate : ℝ),
  (1250 * rate * 10 / 100 = 1500) →
  rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1982_198298


namespace NUMINAMATH_CALUDE_lily_trip_distance_l1982_198247

/-- Represents Lily's car and trip details -/
structure CarTrip where
  /-- Miles per gallon of the car -/
  mpg : ℝ
  /-- Capacity of the gas tank in gallons -/
  tank_capacity : ℝ
  /-- Initial distance driven in miles -/
  initial_distance : ℝ
  /-- First gas purchase in gallons -/
  first_gas_purchase : ℝ
  /-- Second gas purchase in gallons -/
  second_gas_purchase : ℝ
  /-- Fraction of tank full at arrival -/
  final_tank_fraction : ℝ

/-- Calculates the total distance driven given the car trip details -/
def total_distance (trip : CarTrip) : ℝ :=
  trip.initial_distance +
  trip.first_gas_purchase * trip.mpg +
  (trip.second_gas_purchase + trip.final_tank_fraction * trip.tank_capacity - trip.tank_capacity) * trip.mpg

/-- Theorem stating that Lily's total distance driven is 880 miles -/
theorem lily_trip_distance :
  let trip : CarTrip := {
    mpg := 40,
    tank_capacity := 12,
    initial_distance := 480,
    first_gas_purchase := 6,
    second_gas_purchase := 4,
    final_tank_fraction := 3/4
  }
  total_distance trip = 880 := by sorry

end NUMINAMATH_CALUDE_lily_trip_distance_l1982_198247


namespace NUMINAMATH_CALUDE_parabola_unique_intersection_l1982_198224

/-- A parabola defined by x = -4y^2 - 6y + 10 -/
def parabola (y : ℝ) : ℝ := -4 * y^2 - 6 * y + 10

/-- The condition for a vertical line x = m to intersect the parabola at exactly one point -/
def unique_intersection (m : ℝ) : Prop :=
  ∃! y, parabola y = m

theorem parabola_unique_intersection :
  ∀ m : ℝ, unique_intersection m → m = 49 / 4 := by sorry

end NUMINAMATH_CALUDE_parabola_unique_intersection_l1982_198224


namespace NUMINAMATH_CALUDE_log2_derivative_l1982_198256

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) := Real.log x

-- Define the logarithm with base 2
noncomputable def log2 (x : ℝ) := ln x / ln 2

-- State the theorem
theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * ln 2) :=
sorry

end NUMINAMATH_CALUDE_log2_derivative_l1982_198256


namespace NUMINAMATH_CALUDE_sequence_properties_l1982_198243

def sequence_a (n : ℕ) : ℝ := 3 * (2^n - 1)

def sequence_b (n : ℕ) : ℝ := sequence_a n + 3

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 3 * n

theorem sequence_properties :
  (∀ n : ℕ, sum_S (n + 1) = 2 * sequence_a (n + 1) - 3 * (n + 1)) ∧
  sequence_a 1 = 3 ∧
  sequence_a 2 = 9 ∧
  sequence_a 3 = 21 ∧
  (∀ n : ℕ, sequence_b (n + 1) = 2 * sequence_b n) ∧
  (∀ n : ℕ, sequence_a n = 3 * (2^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1982_198243


namespace NUMINAMATH_CALUDE_find_k_n_l1982_198218

theorem find_k_n : ∃ (k n : ℕ), k * n^2 - k * n - n^2 + n = 94 ∧ k = 48 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_k_n_l1982_198218


namespace NUMINAMATH_CALUDE_james_cattle_profit_l1982_198274

/-- Represents the profit calculation for James' cattle business --/
theorem james_cattle_profit :
  let num_cattle : ℕ := 100
  let total_buying_cost : ℚ := 40000
  let buying_cost_per_cattle : ℚ := total_buying_cost / num_cattle
  let feeding_cost_per_cattle : ℚ := buying_cost_per_cattle * 1.2
  let total_feeding_cost_per_month : ℚ := feeding_cost_per_cattle * num_cattle
  let months_held : ℕ := 6
  let total_feeding_cost : ℚ := total_feeding_cost_per_month * months_held
  let total_cost : ℚ := total_buying_cost + total_feeding_cost
  let weight_per_cattle : ℕ := 1000
  let june_price_per_pound : ℚ := 2.2
  let total_selling_price : ℚ := num_cattle * weight_per_cattle * june_price_per_pound
  let profit : ℚ := total_selling_price - total_cost
  profit = -108000 := by sorry

end NUMINAMATH_CALUDE_james_cattle_profit_l1982_198274


namespace NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l1982_198262

def total_men : ℕ := 500
def married_men : ℕ := 350
def men_with_tv : ℕ := 375
def men_with_radio : ℕ := 450
def men_with_car : ℕ := 325
def men_with_refrigerator : ℕ := 275
def men_with_ac : ℕ := 300

theorem max_men_with_all_items_and_married (men_with_all_items_and_married : ℕ) :
  men_with_all_items_and_married ≤ men_with_refrigerator :=
by sorry

end NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l1982_198262


namespace NUMINAMATH_CALUDE_rectangle_transformation_l1982_198255

-- Define the rectangle OPQR
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (2, 0)
def Q : ℝ × ℝ := (2, 2)
def R : ℝ × ℝ := (0, 2)

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x^2 - y^2, 2*x*y)

-- Define the set of transformed points
def transformedSet : Set (ℝ × ℝ) :=
  {(0, 0), (4, 0), (0, 8), (-4, 0)}

-- Theorem statement
theorem rectangle_transformation :
  {transform O, transform P, transform Q, transform R} = transformedSet := by
  sorry

end NUMINAMATH_CALUDE_rectangle_transformation_l1982_198255


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1982_198219

/-- The lateral surface area of a cone with base radius 3 and central angle of its lateral surface unfolded diagram 90° is 36π. -/
theorem cone_lateral_surface_area :
  let base_radius : ℝ := 3
  let central_angle : ℝ := 90
  let lateral_surface_area : ℝ := (1 / 2) * (2 * Real.pi * base_radius) * (4 * base_radius)
  lateral_surface_area = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1982_198219


namespace NUMINAMATH_CALUDE_breath_holding_improvement_l1982_198287

theorem breath_holding_improvement (initial_time : ℝ) : 
  initial_time = 10 → 
  (((initial_time * 2) * 2) * 1.5) = 60 := by
sorry

end NUMINAMATH_CALUDE_breath_holding_improvement_l1982_198287


namespace NUMINAMATH_CALUDE_smallest_difference_in_digit_sum_sequence_l1982_198296

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Predicate for numbers whose digit sum is divisible by 5 -/
def digitSumDivisibleBy5 (n : ℕ) : Prop := digitSum n % 5 = 0

theorem smallest_difference_in_digit_sum_sequence :
  ∃ (a b : ℕ), a < b ∧ 
    digitSumDivisibleBy5 a ∧ 
    digitSumDivisibleBy5 b ∧ 
    b - a = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_in_digit_sum_sequence_l1982_198296


namespace NUMINAMATH_CALUDE_fifth_score_calculation_l1982_198253

theorem fifth_score_calculation (s1 s2 s3 s4 : ℕ) (avg : ℚ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : avg = 75) :
  ∃ (s5 : ℕ), (s1 + s2 + s3 + s4 + s5) / 5 = avg ∧ s5 = 85 := by
  sorry

end NUMINAMATH_CALUDE_fifth_score_calculation_l1982_198253


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1982_198237

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The center of the ellipse is at the origin -/
  center_at_origin : Unit
  /-- The endpoints of the minor axis are at (0, ±1) -/
  minor_axis_endpoints : Unit
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The product of the eccentricity of this ellipse and that of the hyperbola y^2 - x^2 = 1 is 1 -/
  eccentricity_product : e * Real.sqrt 2 = 1

/-- The equation of the special ellipse -/
def ellipse_equation (E : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Theorem stating that the given ellipse has the equation x^2/2 + y^2 = 1 -/
theorem special_ellipse_equation (E : SpecialEllipse) :
  ∀ x y : ℝ, ellipse_equation E x y :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1982_198237


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1982_198294

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_with_complement : A ∩ (Set.univ \ B) = Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1982_198294


namespace NUMINAMATH_CALUDE_marks_lost_per_incorrect_sum_l1982_198252

/-- Given Sandy's quiz results, prove the number of marks lost per incorrect sum --/
theorem marks_lost_per_incorrect_sum :
  ∀ (marks_per_correct : ℕ) 
    (total_attempts : ℕ) 
    (total_marks : ℕ) 
    (correct_sums : ℕ) 
    (marks_lost_per_incorrect : ℕ),
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 60 →
  correct_sums = 24 →
  marks_lost_per_incorrect * (total_attempts - correct_sums) = 
    marks_per_correct * correct_sums - total_marks →
  marks_lost_per_incorrect = 2 :=
by sorry

end NUMINAMATH_CALUDE_marks_lost_per_incorrect_sum_l1982_198252


namespace NUMINAMATH_CALUDE_min_value_quadratic_equation_l1982_198230

theorem min_value_quadratic_equation (a b : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a*x + b - 3 = 0) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a'*x + b' - 3 = 0) → 
    a^2 + (b - 4)^2 ≤ a'^2 + (b' - 4)^2) →
  a^2 + (b - 4)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_equation_l1982_198230


namespace NUMINAMATH_CALUDE_derivative_one_implies_x_is_one_l1982_198269

open Real

theorem derivative_one_implies_x_is_one (f : ℝ → ℝ) (x₀ : ℝ) :
  (f = λ x => x * log x) →
  (deriv f x₀ = 1) →
  x₀ = 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_one_implies_x_is_one_l1982_198269


namespace NUMINAMATH_CALUDE_can_form_123_l1982_198276

-- Define a data type for arithmetic expressions
inductive Expr
  | num : Nat → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

-- Define a function to evaluate an expression
def eval : Expr → Int
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

-- Define a predicate to check if an expression uses all numbers exactly once
def usesAllNumbers (e : Expr) : Prop := sorry

-- Theorem stating that 123 can be formed
theorem can_form_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by
  sorry

end NUMINAMATH_CALUDE_can_form_123_l1982_198276


namespace NUMINAMATH_CALUDE_matts_assignment_problems_l1982_198204

/-- The number of minutes it takes Matt to solve one problem with a calculator -/
def time_with_calculator : ℕ := 2

/-- The number of minutes it takes Matt to solve one problem without a calculator -/
def time_without_calculator : ℕ := 5

/-- The total number of minutes saved by using a calculator -/
def time_saved : ℕ := 60

/-- The number of problems in Matt's assignment -/
def number_of_problems : ℕ := 20

theorem matts_assignment_problems :
  (time_without_calculator - time_with_calculator) * number_of_problems = time_saved :=
by sorry

end NUMINAMATH_CALUDE_matts_assignment_problems_l1982_198204


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l1982_198258

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 36 ways to distribute 4 distinct objects into 3 distinct boxes,
    where each box must contain at least one object. -/
theorem distribute_four_to_three : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_four_to_three_l1982_198258


namespace NUMINAMATH_CALUDE_water_added_to_container_l1982_198220

theorem water_added_to_container (capacity : ℝ) (initial_fullness : ℝ) (final_fullness : ℝ) : 
  capacity = 120 →
  initial_fullness = 0.3 →
  final_fullness = 0.75 →
  (final_fullness - initial_fullness) * capacity = 54 := by
sorry

end NUMINAMATH_CALUDE_water_added_to_container_l1982_198220


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l1982_198254

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 5 → x^4 + 1/x^4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l1982_198254


namespace NUMINAMATH_CALUDE_complement_of_M_l1982_198239

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x ∈ U | (x - 1) * (x - 4) = 0}

theorem complement_of_M (x : ℕ) : x ∈ (U \ M) ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l1982_198239


namespace NUMINAMATH_CALUDE_article_word_limit_l1982_198223

/-- Calculates the word limit for an article given specific font and page constraints. -/
theorem article_word_limit 
  (total_pages : ℕ) 
  (large_font_pages : ℕ) 
  (large_font_words_per_page : ℕ) 
  (small_font_words_per_page : ℕ) 
  (h1 : total_pages = 21)
  (h2 : large_font_pages = 4)
  (h3 : large_font_words_per_page = 1800)
  (h4 : small_font_words_per_page = 2400) :
  large_font_pages * large_font_words_per_page + 
  (total_pages - large_font_pages) * small_font_words_per_page = 48000 :=
by sorry

end NUMINAMATH_CALUDE_article_word_limit_l1982_198223


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1982_198226

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 2*y + z + 8 = 0) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (x' y' z' : ℝ), 2*x' + 2*y' + z' + 8 = 0 →
    (x' - 1)^2 + (y' + 2)^2 + (z' - 3)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1982_198226


namespace NUMINAMATH_CALUDE_largest_product_in_S_largest_product_is_attained_l1982_198251

def S : Set Int := {-8, -3, 0, 2, 4}

theorem largest_product_in_S (a b : Int) : 
  a ∈ S → b ∈ S → a * b ≤ 24 := by sorry

theorem largest_product_is_attained : 
  ∃ (a b : Int), a ∈ S ∧ b ∈ S ∧ a * b = 24 := by sorry

end NUMINAMATH_CALUDE_largest_product_in_S_largest_product_is_attained_l1982_198251


namespace NUMINAMATH_CALUDE_correct_operation_l1982_198288

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1982_198288


namespace NUMINAMATH_CALUDE_distribute_10_4_l1982_198241

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem distribute_10_4 : distribute 10 4 = 286 := by
  sorry

end NUMINAMATH_CALUDE_distribute_10_4_l1982_198241


namespace NUMINAMATH_CALUDE_stream_speed_ratio_l1982_198216

/-- Given a boat and a stream where:
  1. It takes twice as long to row against the stream as with it for the same distance.
  2. The speed of the boat in still water is three times the speed of the stream.
  This theorem proves that the speed of the stream is one-third of the speed of the boat in still water. -/
theorem stream_speed_ratio (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 : ℝ) / (B - S) = 2 * (1 / (B + S))) : S / B = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_ratio_l1982_198216


namespace NUMINAMATH_CALUDE_walkers_speed_l1982_198229

/-- Proves that a walker's speed is 5 mph given specific conditions involving a cyclist --/
theorem walkers_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (walker_catchup_time : ℝ) : ℝ :=
  let walker_speed : ℝ :=
    (cyclist_speed * cyclist_travel_time) / walker_catchup_time
  by
    sorry

#check walkers_speed 20 (5/60) (20/60) = 5

end NUMINAMATH_CALUDE_walkers_speed_l1982_198229


namespace NUMINAMATH_CALUDE_tan_sum_one_fortyfour_l1982_198203

theorem tan_sum_one_fortyfour : (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_one_fortyfour_l1982_198203


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l1982_198246

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 1| < 4} = Set.Ioo (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l1982_198246


namespace NUMINAMATH_CALUDE_divisors_half_of_n_l1982_198259

theorem divisors_half_of_n (n : ℕ) : 
  (n > 0) → (Finset.card (Nat.divisors n) = n / 2) → (n = 8 ∨ n = 12) := by
  sorry

end NUMINAMATH_CALUDE_divisors_half_of_n_l1982_198259


namespace NUMINAMATH_CALUDE_parking_garage_problem_l1982_198215

theorem parking_garage_problem (first_level : ℕ) (second_level : ℕ) (third_level : ℕ) (fourth_level : ℕ) 
  (h1 : first_level = 90)
  (h2 : second_level = first_level + 8)
  (h3 : third_level = second_level + 12)
  (h4 : fourth_level = third_level - 9)
  (h5 : first_level + second_level + third_level + fourth_level - 299 = 100) : 
  ∃ (cars_parked : ℕ), cars_parked = 100 := by
sorry

end NUMINAMATH_CALUDE_parking_garage_problem_l1982_198215


namespace NUMINAMATH_CALUDE_number_difference_l1982_198278

theorem number_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = 2 / 3) (h4 : a^3 + b^3 = 945) : b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1982_198278


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1982_198257

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The equation of a line in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of tangency between a line and a circle -/
def is_tangent (circle line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (ρ₀ θ₀ : ℝ), circle ρ₀ θ₀ ∧ line ρ₀ θ₀ ∧
    ∀ (ρ θ : ℝ), circle ρ θ ∧ line ρ θ → (ρ = ρ₀ ∧ θ = θ₀)

theorem line_tangent_to_circle :
  is_tangent circle_equation line_equation :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1982_198257


namespace NUMINAMATH_CALUDE_cube_of_sum_fractions_is_three_l1982_198214

theorem cube_of_sum_fractions_is_three (a b c : ℤ) 
  (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) : 
  ∃ n : ℤ, a * b * c = n^3 := by
sorry

end NUMINAMATH_CALUDE_cube_of_sum_fractions_is_three_l1982_198214


namespace NUMINAMATH_CALUDE_division_problem_l1982_198286

theorem division_problem (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 
  11 * y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1982_198286


namespace NUMINAMATH_CALUDE_M_remainder_mod_55_l1982_198289

def M : ℕ := sorry

theorem M_remainder_mod_55 : M % 55 = 44 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_55_l1982_198289


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l1982_198284

theorem cos_squared_alpha_minus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 1/3) : 
  Real.cos (α - Real.pi/4)^2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l1982_198284


namespace NUMINAMATH_CALUDE_cos_shift_symmetry_axis_l1982_198268

/-- The axis of symmetry for a cosine function shifted left by π/12 -/
theorem cos_shift_symmetry_axis (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * (x + π / 12))
  ∀ x : ℝ, f (k * π / 2 - π / 12 - x) = f (k * π / 2 - π / 12 + x) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_symmetry_axis_l1982_198268


namespace NUMINAMATH_CALUDE_mixed_tea_sale_price_l1982_198264

/-- Calculates the sale price of mixed tea to earn a specified profit -/
theorem mixed_tea_sale_price
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (profit_percentage : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : profit_percentage = 20)
  : ∃ (sale_price : ℝ), sale_price = 19.2 := by
  sorry

#check mixed_tea_sale_price

end NUMINAMATH_CALUDE_mixed_tea_sale_price_l1982_198264


namespace NUMINAMATH_CALUDE_product_of_digits_l1982_198290

def is_valid_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧
  a + b + c = 11 ∧
  (100 * a + 10 * b + c) % 5 = 0 ∧
  a = 2 * b

theorem product_of_digits (a b c : ℕ) :
  is_valid_number a b c → a * b * c = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_l1982_198290


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_six_l1982_198210

theorem no_solution_implies_a_equals_six (a : ℝ) : 
  (∀ x y : ℝ, (x + 2*y = 4 ∧ 3*x + a*y = 6) → False) → a = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_six_l1982_198210


namespace NUMINAMATH_CALUDE_square_field_area_specific_field_area_l1982_198270

/-- The area of a square field given diagonal travel time and speed -/
theorem square_field_area (travel_time : Real) (speed : Real) : Real :=
  let diagonal_length : Real := speed * (travel_time / 60)
  let side_length : Real := (diagonal_length * 1000) / Real.sqrt 2
  side_length * side_length

/-- Proof of the specific field area -/
theorem specific_field_area : 
  square_field_area 2 3 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_specific_field_area_l1982_198270


namespace NUMINAMATH_CALUDE_quadratic_trinomial_exists_l1982_198271

/-- A quadratic trinomial satisfying the given conditions -/
def f (a c : ℝ) (m : ℝ) : ℝ := a * m^2 - a * m + c

theorem quadratic_trinomial_exists :
  ∃ (a c : ℝ), a ≠ 0 ∧ f a c 4 = 13 :=
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_exists_l1982_198271


namespace NUMINAMATH_CALUDE_min_value_not_e_squared_minus_2m_l1982_198208

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.exp x - (m / 2) * x^2 - m * x

theorem min_value_not_e_squared_minus_2m (m : ℝ) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f m x < Real.exp 2 - 2 * m :=
sorry

end NUMINAMATH_CALUDE_min_value_not_e_squared_minus_2m_l1982_198208


namespace NUMINAMATH_CALUDE_product_of_base8_digits_8670_l1982_198236

/-- The base 8 representation of a natural number -/
def base8Representation (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 8670 is 0 -/
theorem product_of_base8_digits_8670 :
  listProduct (base8Representation 8670) = 0 :=
sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_8670_l1982_198236


namespace NUMINAMATH_CALUDE_parentheses_expression_l1982_198221

theorem parentheses_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ∃ z : ℝ, x * y * z = -x^3 * y^2 → z = -x^2 * y :=
sorry

end NUMINAMATH_CALUDE_parentheses_expression_l1982_198221


namespace NUMINAMATH_CALUDE_orthocenter_locus_is_circle_l1982_198283

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point lies outside a circle -/
def outsideCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

/-- Intersection points of a secant with a circle -/
def secantIntersection (c : Circle) (k : Point) (p q : Point) : Prop :=
  onCircle c p ∧ onCircle c q ∧ ∃ t : ℝ, p = (k.1 + t * (q.1 - k.1), k.2 + t * (q.2 - k.2))

/-- Orthocenter of a triangle -/
def orthocenter (a p q : Point) : Point := sorry

/-- Main theorem -/
theorem orthocenter_locus_is_circle 
  (c : Circle) (a k : Point) 
  (h_a : onCircle c a) 
  (h_k : outsideCircle c k) :
  ∃ c' : Circle, ∀ p q : Point, 
    secantIntersection c k p q → 
    onCircle c' (orthocenter a p q) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_locus_is_circle_l1982_198283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1982_198295

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence)
  (h1 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5)
  (h2 : sum_n seq 4 = 27) :
  seq.a 1 = 135 / 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1982_198295


namespace NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_4_l1982_198280

theorem gcd_n_cubed_plus_16_and_n_plus_4 (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 4^2) (n + 4) = Nat.gcd 48 (n + 4) := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_4_l1982_198280


namespace NUMINAMATH_CALUDE_smallest_bound_for_cubic_coefficient_smallest_k_is_four_l1982_198238

-- Define the set of polynomials M
def M : Set (ℝ → ℝ) :=
  {P | ∃ (a b c d : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + d ∧ 
                         ∀ x ∈ Set.Icc (-1 : ℝ) 1, |P x| ≤ 1}

-- State the theorem
theorem smallest_bound_for_cubic_coefficient :
  ∃ k, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) → |a| ≤ k) ∧
       (∀ k' < k, ∃ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| > k') :=
by
  -- The proof goes here
  sorry

-- State that the smallest k is 4
theorem smallest_k_is_four :
  ∃! k, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) → |a| ≤ k) ∧
       (∀ k' < k, ∃ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| > k') ∧
       k = 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_bound_for_cubic_coefficient_smallest_k_is_four_l1982_198238


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l1982_198228

theorem simplify_algebraic_expression (a : ℝ) : 2*a - 7*a + 4*a = -a := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l1982_198228


namespace NUMINAMATH_CALUDE_digit_swap_l1982_198285

theorem digit_swap (x : ℕ) (h : 9 < x ∧ x < 100) : 
  10 * (x % 10) + (x / 10) = 10 * (x % 10) + (x / 10) :=
by
  sorry

#check digit_swap

end NUMINAMATH_CALUDE_digit_swap_l1982_198285


namespace NUMINAMATH_CALUDE_geometric_sequence_relation_l1982_198277

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1,
    if a_m, a_n, a_p form a geometric sequence in that order,
    then 2n = m + p, where m, n, and p are natural numbers. -/
theorem geometric_sequence_relation (a : ℕ → ℝ) (q : ℝ) (m n p : ℕ) :
  (∀ k, a (k + 1) = q * a k) →  -- geometric sequence condition
  q ≠ 1 →                       -- common ratio ≠ 1
  (a n)^2 = a m * a p →         -- a_m, a_n, a_p form a geometric sequence
  2 * n = m + p :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_relation_l1982_198277


namespace NUMINAMATH_CALUDE_f_properties_l1982_198266

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| - |x - b|

-- Main theorem
theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  -- Part I: Solution set for f(x) > 2 when a = 1 and b = 2
  (∀ x, f x 1 2 > 2 ↔ x > 3/2) ∧
  -- Part II: If max(f) = 3, then min(1/a + 2/b) = (3 + 2√2)/3
  (∃ x, ∀ y, f y a b ≤ f x a b) ∧ (∀ y, f y a b ≤ 3) →
    ∀ a' b', a' > 0 → b' > 0 → 1/a' + 2/b' ≥ (3 + 2*Real.sqrt 2)/3 ∧
    ∃ a'' b'', a'' > 0 ∧ b'' > 0 ∧ 1/a'' + 2/b'' = (3 + 2*Real.sqrt 2)/3 :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1982_198266


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l1982_198267

/-- The number of ways to distribute volunteers to venues -/
def distribute_volunteers (num_volunteers : ℕ) (num_venues : ℕ) : ℕ :=
  if num_venues > num_volunteers then 0
  else if num_venues = 1 then 1
  else (num_volunteers - 1).choose (num_venues - 1) * num_venues.factorial

/-- Theorem: Distributing 4 volunteers to 3 venues yields 36 schemes -/
theorem distribute_four_to_three :
  distribute_volunteers 4 3 = 36 := by
  sorry

#eval distribute_volunteers 4 3

end NUMINAMATH_CALUDE_distribute_four_to_three_l1982_198267


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1982_198249

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1982_198249


namespace NUMINAMATH_CALUDE_odd_function_derivative_range_l1982_198293

open Real

theorem odd_function_derivative_range (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Ioo (-1) 1, deriv f x = 5 + cos x) →  -- f'(x) = 5 + cos(x) for x ∈ (-1, 1)
  (f (1 - t) + f (1 - t^2) < 0) →  -- given condition
  t ∈ Set.Ioo 1 (sqrt 2) :=  -- t ∈ (1, √2)
by sorry

end NUMINAMATH_CALUDE_odd_function_derivative_range_l1982_198293


namespace NUMINAMATH_CALUDE_polynomial_satisfies_conditions_l1982_198234

theorem polynomial_satisfies_conditions :
  ∃ (p : ℝ → ℝ), 
    (∀ x, p x = x^2 + 1) ∧ 
    (p 3 = 10) ∧ 
    (∀ x y, p x * p y = p x + p y + p (x * y) - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_satisfies_conditions_l1982_198234


namespace NUMINAMATH_CALUDE_perimeter_decrease_percentage_l1982_198291

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Perimeter decrease percentage for different length and width reductions --/
theorem perimeter_decrease_percentage
  (r : Rectangle)
  (h1 : perimeter { length := 0.9 * r.length, width := 0.8 * r.width } = 0.88 * perimeter r) :
  perimeter { length := 0.8 * r.length, width := 0.9 * r.width } = 0.82 * perimeter r := by
  sorry


end NUMINAMATH_CALUDE_perimeter_decrease_percentage_l1982_198291


namespace NUMINAMATH_CALUDE_girls_entered_l1982_198233

theorem girls_entered (initial_children final_children boys_left : ℕ) 
  (h1 : initial_children = 85)
  (h2 : boys_left = 31)
  (h3 : final_children = 78) :
  final_children - (initial_children - boys_left) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_entered_l1982_198233


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1982_198207

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1982_198207
