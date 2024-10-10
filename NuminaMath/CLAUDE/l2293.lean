import Mathlib

namespace rectangle_parallel_to_diagonals_l2293_229334

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a point on the side of a square
structure PointOnSquareSide where
  square : Square
  x : ℝ
  y : ℝ
  on_side : (x = 0 ∧ 0 ≤ y ∧ y ≤ square.side) ∨
            (y = 0 ∧ 0 ≤ x ∧ x ≤ square.side) ∨
            (x = square.side ∧ 0 ≤ y ∧ y ≤ square.side) ∨
            (y = square.side ∧ 0 ≤ x ∧ x ≤ square.side)

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_positive : width > 0
  height_positive : height > 0

-- Theorem statement
theorem rectangle_parallel_to_diagonals
  (s : Square) (p : PointOnSquareSide) (h : p.square = s) :
  ∃ (r : Rectangle), 
    -- One vertex of the rectangle is at point p
    (r.width = p.x ∧ r.height = p.y) ∨
    (r.width = s.side - p.x ∧ r.height = p.y) ∨
    (r.width = p.x ∧ r.height = s.side - p.y) ∨
    (r.width = s.side - p.x ∧ r.height = s.side - p.y) ∧
    -- Sides of the rectangle are parallel to the diagonals of the square
    (r.width / r.height = 1 ∨ r.width / r.height = -1) :=
sorry

end rectangle_parallel_to_diagonals_l2293_229334


namespace pen_price_problem_l2293_229337

theorem pen_price_problem (price : ℝ) (quantity : ℝ) : 
  (price * quantity = (price - 1) * (quantity + 100)) →
  (price * quantity = (price + 2) * (quantity - 100)) →
  price = 4 := by
sorry

end pen_price_problem_l2293_229337


namespace direct_square_variation_theorem_l2293_229389

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_theorem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →
  y 3 = 18 →
  y 6 = 72 := by
  sorry

end direct_square_variation_theorem_l2293_229389


namespace reciprocal_sum_pairs_l2293_229355

theorem reciprocal_sum_pairs : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ 1 / m + 1 / n = 1 / 5) ∧
    pairs.card = count ∧
    count = 3 := by sorry

end reciprocal_sum_pairs_l2293_229355


namespace division_remainder_problem_l2293_229339

theorem division_remainder_problem (a b : ℕ) 
  (h1 : a - b = 1390)
  (h2 : a = 1650)
  (h3 : a / b = 6) :
  a % b = 90 := by
sorry

end division_remainder_problem_l2293_229339


namespace minimum_m_in_range_l2293_229301

/-- Represents a sequence of five consecutive integers -/
structure FiveConsecutiveIntegers where
  m : ℕ  -- The middle integer
  h1 : m > 2  -- Ensures all integers are positive

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- Checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k * k

/-- The main theorem -/
theorem minimum_m_in_range (seq : FiveConsecutiveIntegers) :
  (isPerfectSquare (3 * seq.m)) →
  (isPerfectCube (5 * seq.m)) →
  (∃ min_m : ℕ, 
    (∀ m : ℕ, m < min_m → ¬(isPerfectSquare (3 * m) ∧ isPerfectCube (5 * m))) ∧
    600 < min_m ∧
    min_m ≤ 800) :=
sorry

end minimum_m_in_range_l2293_229301


namespace sqrt_3_times_sqrt_12_l2293_229382

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l2293_229382


namespace complex_value_calculation_l2293_229342

theorem complex_value_calculation : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end complex_value_calculation_l2293_229342


namespace square_plus_reciprocal_square_l2293_229328

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 10) : x^2 + 1/x^2 = 98 := by
  sorry

end square_plus_reciprocal_square_l2293_229328


namespace n_sided_polygon_interior_angles_l2293_229349

theorem n_sided_polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 720 → n = 6 := by sorry

end n_sided_polygon_interior_angles_l2293_229349


namespace no_infinite_power_arithmetic_progression_l2293_229384

/-- Represents a term in the sequence of the form a^b -/
def PowerTerm := Nat → Nat

/-- Represents an arithmetic progression -/
def ArithmeticProgression (f : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, f (n + 1) = f n + d

/-- A function that checks if a number is of the form a^b with a, b positive integers and b ≥ 2 -/
def IsPowerForm (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b ≥ 2 ∧ n = a^b

/-- The main theorem stating that no infinite non-constant arithmetic progression
    exists where each term is of the form a^b with a, b positive integers and b ≥ 2 -/
theorem no_infinite_power_arithmetic_progression :
  ¬∃ f : PowerTerm, ArithmeticProgression f ∧
    (∀ n, IsPowerForm (f n)) ∧
    (∃ d : ℕ, d > 0 ∧ ∀ n : ℕ, f (n + 1) = f n + d) :=
sorry

end no_infinite_power_arithmetic_progression_l2293_229384


namespace midpoint_trajectory_l2293_229369

/-- The trajectory of the midpoint of a line segment PQ, where P is fixed at (4, 0) and Q is on the circle x^2 + y^2 = 4 -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (qx qy : ℝ), qx^2 + qy^2 = 4 ∧ x = (4 + qx) / 2 ∧ y = qy / 2) → 
  (x - 2)^2 + y^2 = 1 :=
sorry

end midpoint_trajectory_l2293_229369


namespace officer_selection_count_l2293_229380

/-- The number of members in the club -/
def club_size : ℕ := 12

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 5

/-- The number of ways to select distinct officers from the club members -/
def officer_selection_ways : ℕ := club_size * (club_size - 1) * (club_size - 2) * (club_size - 3) * (club_size - 4)

/-- Theorem stating that the number of ways to select officers is 95,040 -/
theorem officer_selection_count :
  officer_selection_ways = 95040 :=
by sorry

end officer_selection_count_l2293_229380


namespace sphere_volume_with_diameter_10_l2293_229393

/-- The volume of a sphere with diameter 10 meters is 500/3 * π cubic meters. -/
theorem sphere_volume_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := (4 / 3) * π * radius^3
  volume = (500 / 3) * π :=
by sorry

end sphere_volume_with_diameter_10_l2293_229393


namespace animal_permutations_l2293_229359

/-- The number of animals excluding Rat and Snake -/
def n : ℕ := 4

/-- The total number of animals -/
def total_animals : ℕ := 6

/-- Theorem stating that the number of permutations of n distinct objects
    is equal to n factorial, where n is the number of animals excluding
    Rat and Snake -/
theorem animal_permutations :
  (Finset.range n).card.factorial = 24 :=
sorry

end animal_permutations_l2293_229359


namespace number_of_stoplights_l2293_229385

-- Define the number of stoplights
variable (n : ℕ)

-- Define the time for the first route with all green lights
def green_time : ℕ := 10

-- Define the additional time for each red light
def red_light_delay : ℕ := 3

-- Define the time for the second route
def second_route_time : ℕ := 14

-- Define the additional time when all lights are red compared to the second route
def all_red_additional_time : ℕ := 5

-- Theorem statement
theorem number_of_stoplights :
  (green_time + n * red_light_delay = second_route_time + all_red_additional_time) →
  n = 3 := by
  sorry

end number_of_stoplights_l2293_229385


namespace fractional_equation_solution_l2293_229320

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 :=
by sorry

end fractional_equation_solution_l2293_229320


namespace beetle_average_speed_l2293_229312

/-- Represents the terrain types --/
inductive Terrain
  | Flat
  | Sandy
  | Gravel

/-- Represents an insect (ant or beetle) --/
structure Insect where
  flatSpeed : ℝ  -- Speed on flat terrain in meters per minute
  sandySpeedFactor : ℝ  -- Factor to multiply flat speed for sandy terrain
  gravelSpeedFactor : ℝ  -- Factor to multiply flat speed for gravel terrain

/-- Calculates the distance traveled by an insect on a given terrain for a given time --/
def distanceTraveled (insect : Insect) (terrain : Terrain) (time : ℝ) : ℝ :=
  match terrain with
  | Terrain.Flat => insect.flatSpeed * time
  | Terrain.Sandy => insect.flatSpeed * insect.sandySpeedFactor * time
  | Terrain.Gravel => insect.flatSpeed * insect.gravelSpeedFactor * time

/-- The main theorem to prove --/
theorem beetle_average_speed :
  let ant : Insect := {
    flatSpeed := 50,  -- 600 meters / 12 minutes
    sandySpeedFactor := 0.9,  -- 10% decrease
    gravelSpeedFactor := 0.8  -- 20% decrease
  }
  let beetle : Insect := {
    flatSpeed := ant.flatSpeed * 0.85,  -- 15% less than ant
    sandySpeedFactor := 0.95,  -- 5% decrease
    gravelSpeedFactor := 0.75  -- 25% decrease
  }
  let totalDistance := 
    distanceTraveled beetle Terrain.Flat 4 +
    distanceTraveled beetle Terrain.Sandy 3 +
    distanceTraveled beetle Terrain.Gravel 5
  let totalTime := 12
  let averageSpeed := totalDistance / totalTime
  averageSpeed * (60 / 1000) = 2.2525 := by
  sorry

end beetle_average_speed_l2293_229312


namespace min_value_theorem_l2293_229304

theorem min_value_theorem (x y : ℝ) : (x^2*y - 1)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end min_value_theorem_l2293_229304


namespace existence_of_four_numbers_l2293_229325

theorem existence_of_four_numbers : ∃ (a b c d : ℕ+), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c) ∧ (3 ∣ d) ∧
  (d ∣ (a + b + c)) ∧ (c ∣ (a + b + d)) ∧ 
  (b ∣ (a + c + d)) ∧ (a ∣ (b + c + d)) :=
sorry

end existence_of_four_numbers_l2293_229325


namespace solution_set_f_less_g_plus_a_range_of_a_for_f_plus_g_greater_a_squared_l2293_229367

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 2| + a
def g (x : ℝ) : ℝ := |x + 4|

-- Theorem for part I
theorem solution_set_f_less_g_plus_a (a : ℝ) :
  {x : ℝ | f x a < g x + a} = {x : ℝ | x > -1} :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_plus_g_greater_a_squared :
  {a : ℝ | ∀ x, f x a + g x > a^2} = {a : ℝ | -2 < a ∧ a < 3} :=
sorry

end solution_set_f_less_g_plus_a_range_of_a_for_f_plus_g_greater_a_squared_l2293_229367


namespace geese_flock_count_l2293_229313

theorem geese_flock_count : ∃ x : ℕ, 
  (x + x + x / 2 + x / 4 + 1 = 100) ∧ (x = 36) := by
  sorry

end geese_flock_count_l2293_229313


namespace power_equality_l2293_229354

theorem power_equality : 5^29 * 4^15 = 2 * 10^29 := by
  sorry

end power_equality_l2293_229354


namespace arianna_position_l2293_229372

/-- The length of the race in meters -/
def race_length : ℝ := 1000

/-- The distance between Ethan and Arianna when Ethan finished, in meters -/
def distance_between : ℝ := 816

/-- Arianna's distance from the start line when Ethan finished -/
def arianna_distance : ℝ := race_length - distance_between

theorem arianna_position : arianna_distance = 184 := by
  sorry

end arianna_position_l2293_229372


namespace wall_ratio_l2293_229399

/-- Given a wall with specific dimensions, prove that the ratio of its length to its height is 7:1 -/
theorem wall_ratio (w h l : ℝ) : 
  w = 3 →                 -- width is 3 meters
  h = 6 * w →             -- height is 6 times the width
  w * h * l = 6804 →      -- volume is 6804 cubic meters
  l / h = 7 := by
sorry

end wall_ratio_l2293_229399


namespace max_page_number_proof_l2293_229336

def max_page_number (ones : ℕ) (twos : ℕ) : ℕ :=
  let digits : List ℕ := [0, 3, 4, 5, 6, 7, 8, 9]
  199

theorem max_page_number_proof (ones twos : ℕ) :
  ones = 25 → twos = 30 → max_page_number ones twos = 199 := by
  sorry

end max_page_number_proof_l2293_229336


namespace aluminum_foil_thickness_thickness_satisfies_density_l2293_229324

/-- The thickness of a rectangular piece of aluminum foil -/
noncomputable def thickness (d m l w : ℝ) : ℝ := m / (d * l * w)

/-- The volume of a rectangular piece of aluminum foil -/
noncomputable def volume (l w t : ℝ) : ℝ := l * w * t

/-- Theorem: The thickness of a rectangular piece of aluminum foil
    is equal to its mass divided by the product of its density, length, and width -/
theorem aluminum_foil_thickness (d m l w : ℝ) (hd : d > 0) (hl : l > 0) (hw : w > 0) :
  thickness d m l w = m / (d * l * w) :=
by sorry

/-- Theorem: The thickness formula satisfies the density definition -/
theorem thickness_satisfies_density (d m l w : ℝ) (hd : d > 0) (hl : l > 0) (hw : w > 0) :
  d = m / volume l w (thickness d m l w) :=
by sorry

end aluminum_foil_thickness_thickness_satisfies_density_l2293_229324


namespace no_integer_solution_l2293_229309

theorem no_integer_solution : ¬∃ (a b c : ℤ), a^2 + b^2 + 1 = 4*c := by
  sorry

end no_integer_solution_l2293_229309


namespace f_value_at_negative_five_pi_thirds_l2293_229358

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_value_at_negative_five_pi_thirds 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f π)
  (h_cos : ∀ x ∈ Set.Icc (-π/2) 0, f x = Real.cos x) :
  f (-5*π/3) = -1/2 := by
  sorry

end f_value_at_negative_five_pi_thirds_l2293_229358


namespace rectangle_area_rectangle_area_proof_l2293_229331

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 2500 10 = 200 := by
  sorry

end rectangle_area_rectangle_area_proof_l2293_229331


namespace union_set_problem_l2293_229338

theorem union_set_problem (A B : Set ℕ) (m : ℕ) :
  A = {1, 2, m} →
  B = {2, 4} →
  A ∪ B = {1, 2, 3, 4} →
  m = 3 := by
sorry

end union_set_problem_l2293_229338


namespace money_transfer_game_probability_l2293_229322

/-- Represents the state of the game as a triple of integers -/
def GameState := ℕ × ℕ × ℕ

/-- The initial state of the game -/
def initialState : GameState := (3, 3, 3)

/-- Represents a single transfer in the game -/
def Transfer := GameState → GameState

/-- The set of all possible transfers in the game -/
def allTransfers : Set Transfer := sorry

/-- The transition matrix for the Markov chain representing the game -/
def transitionMatrix : GameState → GameState → ℝ := sorry

/-- The steady state distribution of the Markov chain -/
def steadyStateDistribution : GameState → ℝ := sorry

theorem money_transfer_game_probability :
  steadyStateDistribution initialState = 8 / 13 := by sorry

end money_transfer_game_probability_l2293_229322


namespace sunghoon_scores_l2293_229364

theorem sunghoon_scores (korean math english : ℝ) 
  (h1 : korean / math = 1.2) 
  (h2 : math / english = 5/6) : 
  korean / english = 1 := by
  sorry

end sunghoon_scores_l2293_229364


namespace complex_number_location_l2293_229300

theorem complex_number_location :
  let z : ℂ := 2 / (1 - Complex.I)
  z = 1 + Complex.I ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_location_l2293_229300


namespace horner_method_f_2_l2293_229333

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_method_f_2 :
  f 2 = horner_eval [1, 0, 2, 3, 1, 1] 2 ∧ horner_eval [1, 0, 2, 3, 1, 1] 2 = 41 := by
  sorry

#eval f 2
#eval horner_eval [1, 0, 2, 3, 1, 1] 2

end horner_method_f_2_l2293_229333


namespace exactly_two_support_probability_l2293_229375

theorem exactly_two_support_probability (p : ℝ) (h : p = 0.6) :
  let q := 1 - p
  3 * p^2 * q = 0.432 := by sorry

end exactly_two_support_probability_l2293_229375


namespace greatest_prime_divisor_digit_sum_l2293_229319

def number : Nat := 18191

theorem greatest_prime_divisor_digit_sum (p : Nat) : 
  Nat.Prime p ∧ 
  p ∣ number ∧ 
  (∀ q : Nat, Nat.Prime q → q ∣ number → q ≤ p) →
  (p / 10 + p % 10) = 16 := by
sorry

end greatest_prime_divisor_digit_sum_l2293_229319


namespace swimmer_speed_in_still_water_verify_solution_l2293_229371

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given a SwimmerSpeed and a direction. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 10 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed)
  (h_downstream : effectiveSpeed s true * 3 = 45)
  (h_upstream : effectiveSpeed s false * 3 = 15) : 
  s.swimmer = 10 := by
  sorry

/-- Verifies that the solution satisfies the given conditions. -/
theorem verify_solution : 
  let s : SwimmerSpeed := ⟨10, 5⟩
  effectiveSpeed s true * 3 = 45 ∧ 
  effectiveSpeed s false * 3 = 15 := by
  sorry

end swimmer_speed_in_still_water_verify_solution_l2293_229371


namespace binomial_coefficient_20_19_l2293_229348

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l2293_229348


namespace linear_system_solution_l2293_229374

theorem linear_system_solution :
  ∀ x y : ℚ,
  (2 * x + y = 6) →
  (x + 2 * y = 5) →
  ((x + y) / 3 = 11 / 9) :=
by
  sorry

end linear_system_solution_l2293_229374


namespace pirate_game_l2293_229311

/-- Represents the number of coins each pirate has -/
structure PirateCoins where
  first : ℕ
  second : ℕ

/-- Simulates one round of the game where the first pirate loses half their coins -/
def firstLosesHalf (coins : PirateCoins) : PirateCoins :=
  { first := coins.first / 2,
    second := coins.second + coins.first / 2 }

/-- Simulates one round of the game where the second pirate loses half their coins -/
def secondLosesHalf (coins : PirateCoins) : PirateCoins :=
  { first := coins.first + coins.second / 2,
    second := coins.second / 2 }

/-- The main theorem to prove -/
theorem pirate_game (initial : ℕ) :
  (firstLosesHalf (secondLosesHalf (firstLosesHalf { first := initial, second := 0 })))
  = { first := 15, second := 33 } →
  initial = 24 := by
  sorry

end pirate_game_l2293_229311


namespace pipe_length_l2293_229307

theorem pipe_length : ∀ (shorter_piece longer_piece total_length : ℕ),
  shorter_piece = 28 →
  longer_piece = shorter_piece + 12 →
  total_length = shorter_piece + longer_piece →
  total_length = 68 :=
by
  sorry

end pipe_length_l2293_229307


namespace base12_remainder_theorem_l2293_229352

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (a b c d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

/-- The base-12 number 2563₁₂ --/
def base12Number : ℕ := base12ToDecimal 2 5 6 3

/-- The theorem stating that the remainder of 2563₁₂ divided by 17 is 1 --/
theorem base12_remainder_theorem : base12Number % 17 = 1 := by
  sorry

end base12_remainder_theorem_l2293_229352


namespace optimal_solution_satisfies_criteria_l2293_229306

/-- Represents the optimal solution for the medicine problem -/
def optimal_solution : ℕ × ℕ := (6, 3)

/-- Vitamin contents of the first medicine -/
def medicine1_contents : Fin 4 → ℕ
| 0 => 3  -- Vitamin A
| 1 => 1  -- Vitamin B
| 2 => 1  -- Vitamin C
| 3 => 0  -- Vitamin D

/-- Vitamin contents of the second medicine -/
def medicine2_contents : Fin 4 → ℕ
| 0 => 0  -- Vitamin A
| 1 => 1  -- Vitamin B
| 2 => 3  -- Vitamin C
| 3 => 2  -- Vitamin D

/-- Daily vitamin requirements -/
def daily_requirements : Fin 4 → ℕ
| 0 => 3  -- Vitamin A
| 1 => 9  -- Vitamin B
| 2 => 15 -- Vitamin C
| 3 => 2  -- Vitamin D

/-- Cost of medicines in fillér -/
def medicine_costs : Fin 2 → ℕ
| 0 => 20  -- Cost of medicine 1
| 1 => 60  -- Cost of medicine 2

/-- Theorem stating that the optimal solution satisfies all criteria -/
theorem optimal_solution_satisfies_criteria :
  let (x, y) := optimal_solution
  (x + y = 9) ∧ 
  (medicine_costs 0 * x + medicine_costs 1 * y = 300) ∧
  (x + 2 * y = 12) ∧
  (∀ i : Fin 4, medicine1_contents i * x + medicine2_contents i * y ≥ daily_requirements i) :=
by sorry

end optimal_solution_satisfies_criteria_l2293_229306


namespace extreme_value_conditions_l2293_229394

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extreme_value_conditions (a b : ℝ) :
  f a b 1 = 10 ∧ 
  (deriv (f a b)) 1 = 0 →
  a = 4 ∧ b = -11 := by sorry

end extreme_value_conditions_l2293_229394


namespace parabola_intersecting_line_slope_l2293_229302

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def point_M : ℝ × ℝ := (-1, 1)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line_through_point (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = k * (x - p.1)

-- Define the condition for a right angle
def is_right_angle (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

-- Main theorem
theorem parabola_intersecting_line_slope :
  ∀ (k : ℝ) (a b : ℝ × ℝ),
    (∀ x y, parabola x y ↔ (x, y) = a ∨ (x, y) = b) →
    (∀ x y, line_through_point focus k x y ↔ (x, y) = a ∨ (x, y) = b) →
    is_right_angle point_M a b →
    k = 2 := by sorry

end parabola_intersecting_line_slope_l2293_229302


namespace min_major_axis_length_l2293_229383

/-- The line l: x + y - 4 = 0 -/
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

/-- The ellipse x²/16 + y²/12 = 1 -/
def ellipse_e (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- A point M on line l -/
def point_on_line_l (M : ℝ × ℝ) : Prop := line_l M.1 M.2

/-- One focus of the ellipse e -/
def focus_of_ellipse_e : ℝ × ℝ := (-2, 0)

/-- An ellipse passing through M with one focus being a focus of ellipse e -/
def new_ellipse (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  point_on_line_l M ∧ F = focus_of_ellipse_e

/-- The length of the major axis of an ellipse -/
noncomputable def major_axis_length (M F : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the minimum length of the major axis -/
theorem min_major_axis_length :
  ∀ M : ℝ × ℝ, new_ellipse M focus_of_ellipse_e →
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 10 ∧
  ∀ F : ℝ × ℝ, new_ellipse M F →
  major_axis_length M F ≥ min_length :=
sorry

end min_major_axis_length_l2293_229383


namespace sum_of_reciprocals_squared_l2293_229344

theorem sum_of_reciprocals_squared (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a * b = 1) :
  1 / (1 + a^2) + 1 / (1 + b^2) = 1 := by
sorry

end sum_of_reciprocals_squared_l2293_229344


namespace barbed_wire_rate_problem_solution_l2293_229326

/-- Calculates the rate of drawing barbed wire per meter given the conditions of the problem. -/
theorem barbed_wire_rate (field_area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : ℝ :=
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  let rate_per_meter := total_cost / wire_length
  by
    -- Proof goes here
    sorry

/-- The rate of drawing barbed wire per meter for the given problem is 4.5 Rs/m. -/
theorem problem_solution : 
  barbed_wire_rate 3136 1 2 999 = 4.5 := by sorry

end barbed_wire_rate_problem_solution_l2293_229326


namespace mandatoryQuestions_eq_13_l2293_229395

/-- Represents a math competition with mandatory and optional questions -/
structure MathCompetition where
  totalQuestions : ℕ
  correctAnswers : ℕ
  totalScore : ℕ
  mandatoryCorrectPoints : ℕ
  mandatoryIncorrectPoints : ℕ
  optionalCorrectPoints : ℕ

/-- Calculates the number of mandatory questions in the competition -/
def mandatoryQuestions (comp : MathCompetition) : ℕ :=
  sorry

/-- Theorem stating that the number of mandatory questions is 13 -/
theorem mandatoryQuestions_eq_13 (comp : MathCompetition) 
  (h1 : comp.totalQuestions = 25)
  (h2 : comp.correctAnswers = 15)
  (h3 : comp.totalScore = 49)
  (h4 : comp.mandatoryCorrectPoints = 3)
  (h5 : comp.mandatoryIncorrectPoints = 2)
  (h6 : comp.optionalCorrectPoints = 5) :
  mandatoryQuestions comp = 13 := by
  sorry

end mandatoryQuestions_eq_13_l2293_229395


namespace max_volume_pyramid_l2293_229392

/-- A triangular prism with vertices A, B, C, A₁, B₁, C₁ -/
structure TriangularPrism where
  volume : ℝ
  AA₁ : ℝ
  BB₁ : ℝ
  CC₁ : ℝ

/-- Points M, N, K on edges AA₁, BB₁, CC₁ respectively -/
structure PrismPoints (prism : TriangularPrism) where
  M : ℝ
  N : ℝ
  K : ℝ
  h_M : M ≤ prism.AA₁
  h_N : N ≤ prism.BB₁
  h_K : K ≤ prism.CC₁

/-- Theorem stating the maximum volume of pyramid MNKP -/
theorem max_volume_pyramid (prism : TriangularPrism) (points : PrismPoints prism) :
  prism.volume = 35 →
  points.M / prism.AA₁ = 5 / 6 →
  points.N / prism.BB₁ = 6 / 7 →
  points.K / prism.CC₁ = 2 / 3 →
  (∃ (P : ℝ), (P ≥ 0 ∧ P ≤ prism.AA₁) ∨ (P ≥ 0 ∧ P ≤ prism.BB₁) ∨ (P ≥ 0 ∧ P ≤ prism.CC₁)) →
  ∃ (pyramid_volume : ℝ), pyramid_volume ≤ 10 ∧ 
    ∀ (other_volume : ℝ), other_volume ≤ pyramid_volume := by
  sorry

end max_volume_pyramid_l2293_229392


namespace trick_deck_cost_is_nine_l2293_229347

/-- The cost of a single trick deck, given that 8 decks cost 72 dollars -/
def trick_deck_cost : ℚ :=
  72 / 8

/-- Theorem stating that the cost of each trick deck is 9 dollars -/
theorem trick_deck_cost_is_nine : trick_deck_cost = 9 := by
  sorry

end trick_deck_cost_is_nine_l2293_229347


namespace expression_evaluation_l2293_229370

theorem expression_evaluation :
  let x : ℚ := -2
  let expr := (1 - 2 / (x + 1)) / ((x^2 - x) / (x^2 - 1))
  expr = 3/2 := by sorry

end expression_evaluation_l2293_229370


namespace incircle_radius_of_specific_triangle_l2293_229317

theorem incircle_radius_of_specific_triangle : 
  ∀ (a b c h : ℝ) (r : ℝ),
  a = 5 ∧ b = 12 ∧ c = 13 ∧ h = 10 →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem to ensure right-angled triangle
  r = (b * h / 2) / ((a + b + c) / 2) →
  r = 4 := by sorry

end incircle_radius_of_specific_triangle_l2293_229317


namespace smaller_to_larger_volume_ratio_l2293_229329

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  -- Add necessary fields if needed

/-- Represents the smaller octahedron formed by face centers -/
def smaller_octahedron (o : RegularOctahedron) : RegularOctahedron :=
  sorry

/-- Calculates the volume of an octahedron -/
def volume (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem stating the volume ratio of smaller to larger octahedron -/
theorem smaller_to_larger_volume_ratio (o : RegularOctahedron) :
  volume (smaller_octahedron o) / volume o = 1 / 64 := by
  sorry

end smaller_to_larger_volume_ratio_l2293_229329


namespace bear_color_theorem_l2293_229321

/-- Represents the Earth's surface --/
structure EarthSurface where
  latitude : ℝ
  longitude : ℝ

/-- Represents a bear --/
inductive Bear
| Polar
| Other

/-- Represents the hunter's position and orientation --/
structure HunterState where
  position : EarthSurface
  facing : EarthSurface

/-- Function to determine if a point is at the North Pole --/
def isNorthPole (p : EarthSurface) : Prop :=
  p.latitude = 90 -- Assuming 90 degrees latitude is the North Pole

/-- Function to move a point on the Earth's surface --/
def move (start : EarthSurface) (direction : String) (distance : ℝ) : EarthSurface :=
  sorry -- Implementation details omitted

/-- Function to determine the type of bear based on location --/
def bearType (location : EarthSurface) : Bear :=
  sorry -- Implementation details omitted

/-- The main theorem --/
theorem bear_color_theorem 
  (bear_position : EarthSurface)
  (initial_hunter_position : EarthSurface)
  (h1 : initial_hunter_position = move bear_position "south" 100)
  (h2 : let east_position := move initial_hunter_position "east" 100
        east_position.latitude = initial_hunter_position.latitude)
  (h3 : let final_hunter_state := HunterState.mk (move initial_hunter_position "east" 100) bear_position
        final_hunter_state.facing = bear_position)
  : bearType bear_position = Bear.Polar :=
sorry


end bear_color_theorem_l2293_229321


namespace alok_payment_l2293_229390

/-- Represents the order and prices of items in Alok's purchase --/
structure AlokOrder where
  chapati_quantity : ℕ
  rice_quantity : ℕ
  vegetable_quantity : ℕ
  icecream_quantity : ℕ
  chapati_price : ℕ
  rice_price : ℕ
  vegetable_price : ℕ

/-- Calculates the total cost of Alok's order --/
def total_cost (order : AlokOrder) : ℕ :=
  order.chapati_quantity * order.chapati_price +
  order.rice_quantity * order.rice_price +
  order.vegetable_quantity * order.vegetable_price

/-- Theorem stating that Alok's total payment is 811 --/
theorem alok_payment (order : AlokOrder)
  (h1 : order.chapati_quantity = 16)
  (h2 : order.rice_quantity = 5)
  (h3 : order.vegetable_quantity = 7)
  (h4 : order.icecream_quantity = 6)
  (h5 : order.chapati_price = 6)
  (h6 : order.rice_price = 45)
  (h7 : order.vegetable_price = 70) :
  total_cost order = 811 := by
  sorry

end alok_payment_l2293_229390


namespace shopping_discount_l2293_229332

theorem shopping_discount (shoe_price : ℝ) (dress_price : ℝ) 
  (shoe_discount : ℝ) (dress_discount : ℝ) (num_shoes : ℕ) :
  shoe_price = 50 →
  dress_price = 100 →
  shoe_discount = 0.4 →
  dress_discount = 0.2 →
  num_shoes = 2 →
  (num_shoes : ℝ) * shoe_price * (1 - shoe_discount) + 
    dress_price * (1 - dress_discount) = 140 := by
  sorry

end shopping_discount_l2293_229332


namespace ash_cloud_radius_l2293_229388

theorem ash_cloud_radius (height : ℝ) (diameter_ratio : ℝ) : 
  height = 300 → diameter_ratio = 18 → (diameter_ratio * height) / 2 = 2700 := by
  sorry

end ash_cloud_radius_l2293_229388


namespace absolute_value_inequality_solution_set_l2293_229343

theorem absolute_value_inequality (a b : ℝ) (ha : a ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧
  ∀ (m' : ℝ), (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ m' * |a|) → m' ≤ m :=
sorry

theorem solution_set (m : ℝ) (hm : m = 2) :
  {x : ℝ | |x - 1| + |x - 2| ≤ m} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 5/2} :=
sorry

end absolute_value_inequality_solution_set_l2293_229343


namespace equation_solution_l2293_229391

theorem equation_solution :
  ∃ y : ℚ, (y + 1/3 = 3/8 - 1/4) ∧ (y = -5/24) := by
  sorry

end equation_solution_l2293_229391


namespace distance_point_to_line_bounded_l2293_229303

/-- The distance from a point to a line in 2D space is bounded. -/
theorem distance_point_to_line_bounded (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let P : ℝ × ℝ := (-2, 2)
  let l := {(x, y) : ℝ × ℝ | a * (x - 1) + b * (y + 2) = 0}
  let d := Real.sqrt ((a * (-2 - 1) + b * (2 + 2))^2 / (a^2 + b^2))
  0 ≤ d ∧ d ≤ 5 :=
by sorry

end distance_point_to_line_bounded_l2293_229303


namespace circular_ring_area_l2293_229318

/-- The area of a circular ring enclosed between two concentric circles -/
theorem circular_ring_area (C₁ C₂ : ℝ) (h : C₁ > C₂) :
  let S := (C₁^2 - C₂^2) / (4 * Real.pi)
  ∃ (R₁ R₂ : ℝ), R₁ > R₂ ∧ 
    C₁ = 2 * Real.pi * R₁ ∧ 
    C₂ = 2 * Real.pi * R₂ ∧
    S = Real.pi * R₁^2 - Real.pi * R₂^2 :=
by sorry

end circular_ring_area_l2293_229318


namespace optimal_revenue_model_depends_on_factors_l2293_229314

/-- Represents the revenue model for a movie --/
inductive RevenueModel
  | Forever
  | Rental

/-- Represents various economic factors --/
structure EconomicFactors where
  immediateRevenue : ℝ
  longTermRevenuePotential : ℝ
  customerPriceSensitivity : ℝ
  administrativeCosts : ℝ
  piracyRisks : ℝ

/-- Calculates the overall economic value of a revenue model --/
def economicValue (model : RevenueModel) (factors : EconomicFactors) : ℝ :=
  sorry

/-- The theorem stating that the optimal revenue model depends on economic factors --/
theorem optimal_revenue_model_depends_on_factors
  (factors : EconomicFactors) :
  ∃ (model : RevenueModel),
    ∀ (other : RevenueModel),
      economicValue model factors ≥ economicValue other factors :=
  sorry

end optimal_revenue_model_depends_on_factors_l2293_229314


namespace chord_length_no_intersection_tangent_two_intersections_one_intersection_l2293_229379

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12*x

-- Define the line y = 2x - 6
def line1 (x y : ℝ) : Prop := y = 2*x - 6

-- Define the line y = kx + 1
def line2 (k x y : ℝ) : Prop := y = k*x + 1

-- Theorem for the chord length
theorem chord_length : ∃ (x1 y1 x2 y2 : ℝ),
  parabola x1 y1 ∧ parabola x2 y2 ∧ 
  line1 x1 y1 ∧ line1 x2 y2 ∧
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 15 := by sorry

-- Theorems for the positional relationships
theorem no_intersection (k : ℝ) : 
  k > 3 → ¬∃ (x y : ℝ), parabola x y ∧ line2 k x y := by sorry

theorem tangent : 
  ∃! (x y : ℝ), parabola x y ∧ line2 3 x y := by sorry

theorem two_intersections (k : ℝ) : 
  k < 3 ∧ k ≠ 0 → ∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ parabola x1 y1 ∧ parabola x2 y2 ∧ 
    line2 k x1 y1 ∧ line2 k x2 y2 := by sorry

theorem one_intersection : 
  ∃! (x y : ℝ), parabola x y ∧ line2 0 x y := by sorry

end chord_length_no_intersection_tangent_two_intersections_one_intersection_l2293_229379


namespace douglas_vote_percentage_l2293_229316

theorem douglas_vote_percentage (total_percentage : ℝ) (x_percentage : ℝ) (x_ratio : ℝ) (y_ratio : ℝ) :
  total_percentage = 0.54 →
  x_percentage = 0.62 →
  x_ratio = 3 →
  y_ratio = 2 →
  let total_ratio := x_ratio + y_ratio
  let y_percentage := (total_percentage * total_ratio - x_percentage * x_ratio) / y_ratio
  y_percentage = 0.42 := by
  sorry

end douglas_vote_percentage_l2293_229316


namespace polynomial_sum_property_l2293_229350

/-- Generate all words of length n using letters A and B -/
def generateWords (n : ℕ) : List String :=
  sorry

/-- Convert a word to a polynomial by replacing A with x and B with (1-x) -/
def wordToPolynomial (word : String) : ℝ → ℝ :=
  sorry

/-- Sum the first k polynomials -/
def sumPolynomials (n : ℕ) (k : ℕ) : ℝ → ℝ :=
  sorry

/-- A function is increasing on [0,1] -/
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y

/-- A function is constant on [0,1] -/
def isConstant (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x = f y

theorem polynomial_sum_property (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2^n) :
  let f := sumPolynomials n k
  isConstant f ∨ isIncreasing f :=
sorry

end polynomial_sum_property_l2293_229350


namespace express_y_in_terms_of_x_l2293_229368

theorem express_y_in_terms_of_x (x y : ℝ) :
  4 * x - y = 7 → y = 4 * x - 7 := by
  sorry

end express_y_in_terms_of_x_l2293_229368


namespace average_income_of_A_and_B_l2293_229398

/-- Given the average monthly incomes of different pairs of people and the income of one person,
    prove that the average monthly income of A and B is 5050. -/
theorem average_income_of_A_and_B (A B C : ℕ) : 
  A = 4000 →
  (B + C) / 2 = 6250 →
  (A + C) / 2 = 5200 →
  (A + B) / 2 = 5050 := by
  sorry


end average_income_of_A_and_B_l2293_229398


namespace first_group_size_correct_l2293_229361

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 63

/-- The number of days the first group takes to repair the road -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group takes to repair the road -/
def second_group_days : ℕ := 21

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- The theorem stating that the first group size is correct -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours =
  second_group_size * second_group_days * second_group_hours :=
by sorry

end first_group_size_correct_l2293_229361


namespace find_a_l2293_229335

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) < 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 2}

-- Theorem statement
theorem find_a : ∃ a : ℝ, (∀ x : ℝ, inequality a x ↔ x ∈ solution_set a) ∧ a = 1/2 := by
  sorry

end find_a_l2293_229335


namespace perpendicular_parallel_lines_l2293_229346

/-- Given a line l with inclination 45°, line l₁ passing through A(3,2) and B(a,-1) perpendicular to l,
    and line l₂: 2x+by+1=0 parallel to l₁, prove that a + b = 8 -/
theorem perpendicular_parallel_lines (a b : ℝ) : 
  (∃ (l l₁ l₂ : Set (ℝ × ℝ)),
    -- l has inclination 45°
    (∀ (x y : ℝ), (x, y) ∈ l ↔ y = x) ∧
    -- l₁ passes through A(3,2) and B(a,-1)
    ((3, 2) ∈ l₁ ∧ (a, -1) ∈ l₁) ∧
    -- l₁ is perpendicular to l
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l ∧ (x₂, y₂) ∈ l₁ → (x₂ - x₁) * (y₂ - y₁) = -1) ∧
    -- l₂: 2x+by+1=0
    (∀ (x y : ℝ), (x, y) ∈ l₂ ↔ 2*x + b*y + 1 = 0) ∧
    -- l₂ is parallel to l₁
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (x₂ - x₁) * (y₂ - y₁) = 0))
  → a + b = 8 := by
  sorry


end perpendicular_parallel_lines_l2293_229346


namespace ellipse_m_range_l2293_229378

/-- Represents an ellipse with equation x^2/m^2 + y^2/(2+m) = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2/m^2 + y^2/(2+m) = 1

/-- Condition for foci on x-axis -/
def foci_on_x_axis (m : ℝ) : Prop := m^2 > 2 + m

/-- The range of m for which the ellipse is valid and has foci on x-axis -/
def valid_m_range (m : ℝ) : Prop := (m > 2 ∨ (-2 < m ∧ m < -1))

theorem ellipse_m_range (m : ℝ) (e : Ellipse m) :
  foci_on_x_axis m → valid_m_range m :=
by sorry

end ellipse_m_range_l2293_229378


namespace winnie_balloon_distribution_l2293_229323

/-- The number of balloons Winnie keeps for herself when distributing balloons to friends -/
def balloons_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_balloon_distribution :
  let total_balloons : ℕ := 20 + 40 + 70 + 90
  let num_friends : ℕ := 9
  balloons_kept total_balloons num_friends = 4 := by
  sorry

end winnie_balloon_distribution_l2293_229323


namespace car_speed_l2293_229381

/-- Proves that if a car travels 1 km in 5 seconds more than it would take at 90 km/hour, then its speed is 80 km/hour. -/
theorem car_speed (v : ℝ) (h : v > 0) : 
  (3600 / v) = (3600 / 90) + 5 → v = 80 := by
sorry

end car_speed_l2293_229381


namespace no_solutions_to_equation_l2293_229386

theorem no_solutions_to_equation : 
  ¬∃ (x : ℝ), |x - 1| = |2*x - 4| + |x - 5| := by
sorry

end no_solutions_to_equation_l2293_229386


namespace unattainable_y_l2293_229330

theorem unattainable_y (x : ℝ) :
  (2 * x^2 + 3 * x + 4 ≠ 0) →
  ∃ y : ℝ, y = (1 - x) / (2 * x^2 + 3 * x + 4) ∧ y ≠ 0 :=
by sorry

end unattainable_y_l2293_229330


namespace trigonometric_equality_l2293_229360

theorem trigonometric_equality (α : ℝ) :
  (2 * Real.cos (π/6 - 2*α) - Real.sqrt 3 * Real.sin (5*π/2 - 2*α)) /
  (Real.cos (9*π/2 - 2*α) + 2 * Real.cos (π/6 + 2*α)) =
  Real.tan (2*α) / Real.sqrt 3 := by
  sorry

end trigonometric_equality_l2293_229360


namespace gcf_of_72_90_120_l2293_229366

theorem gcf_of_72_90_120 : Nat.gcd 72 (Nat.gcd 90 120) = 6 := by
  sorry

end gcf_of_72_90_120_l2293_229366


namespace right_triangle_median_to_hypotenuse_l2293_229365

theorem right_triangle_median_to_hypotenuse (DE DF : ℝ) :
  DE = 15 →
  DF = 20 →
  let EF := Real.sqrt (DE^2 + DF^2)
  let median := EF / 2
  median = 12.5 := by
sorry

end right_triangle_median_to_hypotenuse_l2293_229365


namespace inequality_solution_l2293_229363

theorem inequality_solution (x : ℝ) : 
  (4 ≤ x^2 - 3*x - 6 ∧ x^2 - 3*x - 6 ≤ 2*x + 8) ↔ 
  ((5 ≤ x ∧ x ≤ 7) ∨ x = -2) :=
by sorry

end inequality_solution_l2293_229363


namespace angle_cosine_equivalence_l2293_229376

theorem angle_cosine_equivalence (A B : Real) (hA : 0 < A ∧ A < Real.pi) (hB : 0 < B ∧ B < Real.pi) :
  A > B ↔ Real.cos A < Real.cos B := by
  sorry

end angle_cosine_equivalence_l2293_229376


namespace quadratic_inequality_l2293_229315

theorem quadratic_inequality (x : ℝ) :
  x^2 - 50*x + 575 ≤ 25 ↔ 25 - 5*Real.sqrt 3 ≤ x ∧ x ≤ 25 + 5*Real.sqrt 3 := by
  sorry

end quadratic_inequality_l2293_229315


namespace square_between_squares_l2293_229396

theorem square_between_squares (n k l m : ℕ) :
  m^2 < n ∧ n < (m+1)^2 ∧ n - k = m^2 ∧ n + l = (m+1)^2 →
  ∃ p : ℕ, n - k * l = p^2 := by
sorry

end square_between_squares_l2293_229396


namespace longest_path_old_town_l2293_229340

structure OldTown where
  intersections : Nat
  start_color : Bool
  end_color : Bool

def longest_path (town : OldTown) : Nat :=
  sorry

theorem longest_path_old_town (town : OldTown) :
  town.intersections = 36 →
  town.start_color = town.end_color →
  longest_path town = 34 := by
  sorry

end longest_path_old_town_l2293_229340


namespace nine_to_power_2023_div_3_l2293_229397

theorem nine_to_power_2023_div_3 (n : ℕ) : n = 9^2023 → n / 3 = 3^4045 :=
by
  sorry

end nine_to_power_2023_div_3_l2293_229397


namespace ab_power_2023_l2293_229377

theorem ab_power_2023 (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) : (a * b)^2023 = -1 := by
  sorry

end ab_power_2023_l2293_229377


namespace dentist_bill_calculation_dentist_cleaning_cost_l2293_229308

theorem dentist_bill_calculation (filling_cost : ℕ) (extraction_cost : ℕ) : ℕ :=
  let total_bill := 5 * filling_cost
  let cleaning_cost := total_bill - (2 * filling_cost + extraction_cost)
  cleaning_cost

theorem dentist_cleaning_cost : dentist_bill_calculation 120 290 = 70 := by
  sorry

end dentist_bill_calculation_dentist_cleaning_cost_l2293_229308


namespace share_ratio_l2293_229362

/-- Proves that the ratio of B's share to C's share is 3:2 given the problem conditions -/
theorem share_ratio (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 544 →
  a_share = 384 →
  b_share = 96 →
  c_share = 64 →
  amount = a_share + b_share + c_share →
  a_share = (2 : ℚ) / 3 * b_share →
  b_share / c_share = (3 : ℚ) / 2 := by
  sorry

end share_ratio_l2293_229362


namespace range_of_a_l2293_229357

theorem range_of_a (a b : ℝ) 
  (h1 : 0 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1/2 ≤ a ∧ a ≤ 5/2 := by
  sorry

end range_of_a_l2293_229357


namespace exists_number_with_properties_l2293_229341

/-- A function that counts the occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- A function that checks if a natural number contains only 7s and 5s -/
def containsOnly7sAnd5s (n : ℕ) : Prop := sorry

/-- Theorem stating the existence of a number with the required properties -/
theorem exists_number_with_properties : ∃ n : ℕ, 
  containsOnly7sAnd5s n ∧ 
  countDigit n 7 = countDigit n 5 ∧ 
  n % 7 = 0 ∧ 
  n % 5 = 0 := by sorry

end exists_number_with_properties_l2293_229341


namespace equidistant_line_slope_l2293_229353

-- Define the points P and Q
def P : ℝ × ℝ := (4, 6)
def Q : ℝ × ℝ := (6, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem equidistant_line_slope :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = m * x → 
      (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2) ∧
    m = 1/2 := by
  sorry

end equidistant_line_slope_l2293_229353


namespace book_gain_percent_l2293_229387

theorem book_gain_percent (MP : ℝ) (CP : ℝ) (SP : ℝ) : 
  CP = 0.64 * MP →
  SP = 0.84 * MP →
  (SP - CP) / CP * 100 = 31.25 :=
by sorry

end book_gain_percent_l2293_229387


namespace marias_water_bottles_l2293_229310

/-- Calculates the final number of water bottles Maria has -/
def final_bottle_count (initial : ℕ) (drunk : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Proves that Maria's final bottle count is correct -/
theorem marias_water_bottles (initial : ℕ) (drunk : ℕ) (bought : ℕ) 
  (h1 : initial ≥ drunk) : 
  final_bottle_count initial drunk bought = initial - drunk + bought :=
by
  sorry

#eval final_bottle_count 14 8 45

end marias_water_bottles_l2293_229310


namespace trivia_team_selection_l2293_229356

theorem trivia_team_selection (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 36)
  (h2 : num_groups = 3)
  (h3 : students_per_group = 9) :
  total_students - (num_groups * students_per_group) = 9 := by
  sorry

end trivia_team_selection_l2293_229356


namespace min_value_theorem_l2293_229305

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  ∃ (min_val : ℝ), min_val = 8 + 4 * Real.sqrt 3 ∧
  ∀ z, z = (x + 1) * (y + 1) / (x * y) → z ≥ min_val :=
sorry

end min_value_theorem_l2293_229305


namespace video_game_spending_ratio_l2293_229373

theorem video_game_spending_ratio (initial_amount : ℚ) (video_game_cost : ℚ) (remaining : ℚ) :
  initial_amount = 100 →
  remaining = initial_amount - video_game_cost - (1/5) * (initial_amount - video_game_cost) →
  remaining = 60 →
  video_game_cost / initial_amount = 1/3 := by
  sorry

end video_game_spending_ratio_l2293_229373


namespace m_range_l2293_229345

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 - m) * x + 1 < (2 - m) * y + 1

-- Define the theorem
theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m ∈ Set.Icc 1 2 ∧ m ≠ 2 :=
sorry

end m_range_l2293_229345


namespace math_olympiad_properties_l2293_229351

/-- Represents the math Olympiad team composition and assessment rules -/
structure MathOlympiadTeam where
  total_students : Nat
  grade_11_students : Nat
  grade_12_students : Nat
  grade_13_students : Nat
  selected_students : Nat
  prob_correct_easy : Rat
  prob_correct_hard : Rat
  points_easy : Nat
  points_hard : Nat
  excellent_threshold : Nat

/-- Calculates the probability of selecting exactly 2 students from Grade 11 -/
def prob_two_from_grade_11 (team : MathOlympiadTeam) : Rat :=
  sorry

/-- Calculates the mathematical expectation of Zhang's score -/
def expected_score (team : MathOlympiadTeam) : Rat :=
  sorry

/-- Calculates the probability of Zhang being an excellent student -/
def prob_excellent_student (team : MathOlympiadTeam) : Rat :=
  sorry

/-- The main theorem proving the three required properties -/
theorem math_olympiad_properties (team : MathOlympiadTeam)
  (h1 : team.total_students = 20)
  (h2 : team.grade_11_students = 8)
  (h3 : team.grade_12_students = 7)
  (h4 : team.grade_13_students = 5)
  (h5 : team.selected_students = 3)
  (h6 : team.prob_correct_easy = 2/3)
  (h7 : team.prob_correct_hard = 1/2)
  (h8 : team.points_easy = 1)
  (h9 : team.points_hard = 2)
  (h10 : team.excellent_threshold = 5) :
  prob_two_from_grade_11 team = 28/95 ∧
  expected_score team = 10/3 ∧
  prob_excellent_student team = 2/9 :=
by
  sorry

end math_olympiad_properties_l2293_229351


namespace dolphin_count_theorem_l2293_229327

/-- Given an initial number of dolphins in the ocean and a factor for additional dolphins joining,
    calculate the total number of dolphins after joining. -/
def total_dolphins (initial : ℕ) (joining_factor : ℕ) : ℕ :=
  initial + joining_factor * initial

/-- Theorem stating that with 65 initial dolphins and 3 times that number joining,
    the total number of dolphins is 260. -/
theorem dolphin_count_theorem :
  total_dolphins 65 3 = 260 := by
  sorry

end dolphin_count_theorem_l2293_229327
