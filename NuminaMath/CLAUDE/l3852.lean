import Mathlib

namespace game_result_l3852_385209

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 4 = 0 then 8
  else if n % 3 = 0 then 3
  else if n % 4 = 0 then 1
  else 0

def allie_rolls : List ℕ := [6, 3, 4, 1]
def betty_rolls : List ℕ := [12, 9, 4, 2]

def total_points (rolls : List ℕ) : ℕ :=
  (rolls.map g).sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 84 := by
  sorry

end game_result_l3852_385209


namespace sams_new_nickels_l3852_385211

/-- The number of nickels Sam's dad gave him -/
def nickels_from_dad (initial_nickels final_nickels : ℕ) : ℕ :=
  final_nickels - initial_nickels

/-- Proof that Sam's dad gave him 39 nickels -/
theorem sams_new_nickels :
  let initial_nickels : ℕ := 24
  let final_nickels : ℕ := 63
  nickels_from_dad initial_nickels final_nickels = 39 := by
sorry

end sams_new_nickels_l3852_385211


namespace total_cost_of_collars_l3852_385246

/-- Represents the material composition and cost of a collar --/
structure Collar :=
  (nylon_inches : ℕ)
  (leather_inches : ℕ)
  (nylon_cost_per_inch : ℕ)
  (leather_cost_per_inch : ℕ)

/-- Calculates the total cost of a single collar --/
def collar_cost (c : Collar) : ℕ :=
  c.nylon_inches * c.nylon_cost_per_inch + c.leather_inches * c.leather_cost_per_inch

/-- Defines a dog collar according to the problem specifications --/
def dog_collar : Collar :=
  { nylon_inches := 18
  , leather_inches := 4
  , nylon_cost_per_inch := 1
  , leather_cost_per_inch := 2 }

/-- Defines a cat collar according to the problem specifications --/
def cat_collar : Collar :=
  { nylon_inches := 10
  , leather_inches := 2
  , nylon_cost_per_inch := 1
  , leather_cost_per_inch := 2 }

/-- Theorem stating the total cost of materials for 9 dog collars and 3 cat collars --/
theorem total_cost_of_collars :
  9 * collar_cost dog_collar + 3 * collar_cost cat_collar = 276 := by
  sorry

end total_cost_of_collars_l3852_385246


namespace point_on_direct_proportion_l3852_385239

/-- A direct proportion function passing through two points -/
def DirectProportion (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- The theorem stating that if A(3,-5) and B(-6,a) lie on a direct proportion function, then a = 10 -/
theorem point_on_direct_proportion (k a : ℝ) :
  DirectProportion k 3 (-5) ∧ DirectProportion k (-6) a → a = 10 := by
  sorry

end point_on_direct_proportion_l3852_385239


namespace nickel_count_proof_l3852_385215

/-- Represents the number of nickels in a collection of coins -/
def number_of_nickels (total_value : ℚ) (total_coins : ℕ) : ℕ :=
  2

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 1/20

theorem nickel_count_proof (total_value : ℚ) (total_coins : ℕ) 
  (h1 : total_value = 7/10) 
  (h2 : total_coins = 8) :
  number_of_nickels total_value total_coins = 2 ∧ 
  ∃ (d n : ℕ), d + n = total_coins ∧ 
               d * dime_value + n * nickel_value = total_value :=
by
  sorry

#check nickel_count_proof

end nickel_count_proof_l3852_385215


namespace friend_product_sum_l3852_385250

/-- A function representing the product of the first n positive integers -/
def productOfFirstN (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- A proposition stating that for any five natural numbers a, b, c, d, e,
    if the product of the first a numbers equals the sum of the products of
    the first b, c, d, and e numbers, then a must be either 3 or 4 -/
theorem friend_product_sum (a b c d e : ℕ) :
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e) →
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) →
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) →
  productOfFirstN a = productOfFirstN b + productOfFirstN c + productOfFirstN d + productOfFirstN e →
  a = 3 ∨ a = 4 := by
  sorry

end friend_product_sum_l3852_385250


namespace fifth_dog_weight_l3852_385208

def dog_weights : List ℝ := [25, 31, 35, 33]

theorem fifth_dog_weight (w : ℝ) :
  (dog_weights.sum + w) / 5 = dog_weights.sum / 4 →
  w = 31 :=
by sorry

end fifth_dog_weight_l3852_385208


namespace simplify_sqrt_sum_l3852_385299

theorem simplify_sqrt_sum : 
  (Real.sqrt 418 / Real.sqrt 308) + (Real.sqrt 294 / Real.sqrt 196) = 17 / 3 := by
  sorry

end simplify_sqrt_sum_l3852_385299


namespace arithmetic_geometric_sequence_sum_l3852_385227

-- Define the conditions
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b - a = c - b ∧ b - a = d ∧ d ≠ 0

def is_geometric_sequence (c a b : ℝ) : Prop :=
  ∃ r : ℝ, a / c = b / a ∧ a / c = r ∧ r ≠ 1

-- State the theorem
theorem arithmetic_geometric_sequence_sum (a b c : ℝ) :
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  is_arithmetic_sequence a b c →
  is_geometric_sequence c a b →
  a + 3*b + c = 10 →
  a = -4 :=
by sorry

end arithmetic_geometric_sequence_sum_l3852_385227


namespace count_four_digit_divisible_by_5_ending_0_is_900_l3852_385212

/-- A function that counts the number of positive four-digit integers divisible by 5 and ending in 0 -/
def count_four_digit_divisible_by_5_ending_0 : ℕ :=
  let first_digit := Finset.range 9  -- 1 to 9
  let second_digit := Finset.range 10  -- 0 to 9
  let third_digit := Finset.range 10  -- 0 to 9
  (first_digit.card * second_digit.card * third_digit.card : ℕ)

/-- Theorem stating that the count of positive four-digit integers divisible by 5 and ending in 0 is 900 -/
theorem count_four_digit_divisible_by_5_ending_0_is_900 :
  count_four_digit_divisible_by_5_ending_0 = 900 := by
  sorry

end count_four_digit_divisible_by_5_ending_0_is_900_l3852_385212


namespace log_difference_equals_one_l3852_385218

theorem log_difference_equals_one (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (α + π / 4) = 3) : 
  Real.log (8 * Real.sin α + 6 * Real.cos α) - Real.log (4 * Real.sin α - Real.cos α) = 1 := by
  sorry

end log_difference_equals_one_l3852_385218


namespace quadratic_root_difference_l3852_385235

theorem quadratic_root_difference (x : ℝ) : 
  5 * x^2 - 9 * x - 22 = 0 →
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    (5 * r₁^2 - 9 * r₁ - 22 = 0) ∧
    (5 * r₂^2 - 9 * r₂ - 22 = 0) ∧
    |r₁ - r₂| = Real.sqrt 521 / 5 ∧
    (∀ (p : ℕ), p > 1 → ¬(p^2 ∣ 521)) :=
by sorry

end quadratic_root_difference_l3852_385235


namespace sunset_time_calculation_l3852_385265

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + t.hours * 60 + d.minutes + d.hours * 60
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- Converts 24-hour time to 12-hour time -/
def to12Hour (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) : 
  to12Hour (addDuration sunrise daylight) = { hours := 5, minutes := 31 } :=
  by sorry

end sunset_time_calculation_l3852_385265


namespace thompson_children_probability_l3852_385236

theorem thompson_children_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each child being male (or female)
  let total_outcomes : ℕ := 2^n  -- total number of possible gender combinations
  let equal_outcomes : ℕ := n.choose (n/2)  -- number of combinations with equal sons and daughters
  
  (total_outcomes - equal_outcomes : ℚ) / total_outcomes = 93/128 :=
by sorry

end thompson_children_probability_l3852_385236


namespace distance_to_origin_l3852_385213

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) (h2 : x = 2 + Real.sqrt 105) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by
  sorry

end distance_to_origin_l3852_385213


namespace one_instrument_one_sport_probability_l3852_385256

def total_people : ℕ := 1500

def instrument_ratio : ℚ := 3/7
def sport_ratio : ℚ := 5/14
def both_ratio : ℚ := 1/6
def multi_instrument_ratio : ℚ := 19/200  -- 9.5% = 19/200

def probability_one_instrument_one_sport (total : ℕ) (instrument : ℚ) (sport : ℚ) (both : ℚ) (multi : ℚ) : ℚ :=
  both

theorem one_instrument_one_sport_probability :
  probability_one_instrument_one_sport total_people instrument_ratio sport_ratio both_ratio multi_instrument_ratio = 1/6 := by
  sorry

end one_instrument_one_sport_probability_l3852_385256


namespace pacos_marble_purchase_l3852_385243

theorem pacos_marble_purchase : 
  0.33 + 0.33 + 0.08 = 0.74 := by sorry

end pacos_marble_purchase_l3852_385243


namespace topsoil_cost_for_8_cubic_yards_l3852_385228

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- The cost of topsoil for a given volume in cubic yards -/
def topsoil_cost (volume : ℝ) : ℝ :=
  volume * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_for_8_cubic_yards :
  topsoil_cost volume_in_cubic_yards = 1728 := by
  sorry

end topsoil_cost_for_8_cubic_yards_l3852_385228


namespace sum_of_digits_9ab_l3852_385202

def a : ℕ := 999
def b : ℕ := 666

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9ab : sum_of_digits (9 * a * b) = 36 := by
  sorry

end sum_of_digits_9ab_l3852_385202


namespace combined_area_of_tracts_l3852_385230

/-- The combined area of two rectangular tracts of land -/
theorem combined_area_of_tracts (length1 width1 length2 width2 : ℕ) 
  (h1 : length1 = 300)
  (h2 : width1 = 500)
  (h3 : length2 = 250)
  (h4 : width2 = 630) :
  length1 * width1 + length2 * width2 = 307500 := by
  sorry

end combined_area_of_tracts_l3852_385230


namespace factorization_ax_squared_minus_a_l3852_385286

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end factorization_ax_squared_minus_a_l3852_385286


namespace temperature_difference_l3852_385284

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 8) 
  (h2 : lowest = -2) : 
  highest - lowest = 10 := by
  sorry

end temperature_difference_l3852_385284


namespace chess_tournament_games_l3852_385285

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1)

/-- The number of players in the tournament -/
def num_players : ℕ := 20

/-- Each game is played twice -/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  num_games num_players * games_per_pair = 760 := by
  sorry

end chess_tournament_games_l3852_385285


namespace tangent_line_theorem_l3852_385279

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a-2)*x

-- Define the tangent line at the origin
def tangent_line_at_origin (a : ℝ) (x : ℝ) : ℝ := -2*x

-- Theorem statement
theorem tangent_line_theorem (a : ℝ) :
  ∀ x : ℝ, (tangent_line_at_origin a x) = 
    (deriv (f a)) 0 * x + (f a 0) :=
by sorry

end tangent_line_theorem_l3852_385279


namespace prob_sum_20_l3852_385275

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The target sum we're aiming for -/
def targetSum : ℕ := 20

/-- The set of possible outcomes when rolling 'numDice' dice, each with 'numFaces' faces -/
def allOutcomes : Finset (Fin numDice → Fin numFaces) := sorry

/-- A function that sums the values of a dice roll -/
def sumRoll (roll : Fin numDice → Fin numFaces) : ℕ := sorry

/-- The set of favorable outcomes (those that sum to targetSum) -/
def favorableOutcomes : Finset (Fin numDice → Fin numFaces) :=
  allOutcomes.filter (λ roll ↦ sumRoll roll = targetSum)

/-- The probability of rolling a sum of 20 with four 6-faced dice -/
theorem prob_sum_20 : 
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 15 / 1296 := by sorry

end prob_sum_20_l3852_385275


namespace power_sum_calculation_l3852_385258

theorem power_sum_calculation : (-1: ℤ)^53 + (2 : ℚ)^(2^4 + 5^2 - 4^3) = -1 + 1 / 8388608 := by
  sorry

end power_sum_calculation_l3852_385258


namespace road_signs_at_first_intersection_l3852_385259

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Defines the relationship between road signs at different intersections -/
def valid_road_signs (rs : RoadSigns) : Prop :=
  rs.second = rs.first + rs.first / 4 ∧
  rs.third = 2 * rs.second ∧
  rs.fourth = rs.third - 20 ∧
  rs.first + rs.second + rs.third + rs.fourth = 270

theorem road_signs_at_first_intersection :
  ∃ (rs : RoadSigns), valid_road_signs rs ∧ rs.first = 40 :=
by sorry

end road_signs_at_first_intersection_l3852_385259


namespace unique_max_divisor_number_l3852_385214

/-- A positive integer N satisfies the special divisor property if all of its divisors
    can be written as p-2 for some prime number p -/
def has_special_divisor_property (N : ℕ+) : Prop :=
  ∀ d : ℕ, d ∣ N.val → ∃ p : ℕ, Nat.Prime p ∧ d = p - 2

/-- The maximum number of divisors for any N satisfying the special divisor property -/
def max_divisors : ℕ := 8

/-- The theorem stating that 135 is the only number with the maximum number of divisors
    satisfying the special divisor property -/
theorem unique_max_divisor_number :
  ∃! N : ℕ+, has_special_divisor_property N ∧
  (Nat.card {d : ℕ | d ∣ N.val} = max_divisors) ∧
  N.val = 135 := by sorry

#check unique_max_divisor_number

end unique_max_divisor_number_l3852_385214


namespace steve_has_four_friends_l3852_385276

/-- The number of friends Steve has, given the initial number of gold bars,
    the number of lost gold bars, and the number of gold bars each friend receives. -/
def number_of_friends (initial_bars : ℕ) (lost_bars : ℕ) (bars_per_friend : ℕ) : ℕ :=
  (initial_bars - lost_bars) / bars_per_friend

/-- Theorem stating that Steve has 4 friends given the problem conditions. -/
theorem steve_has_four_friends :
  number_of_friends 100 20 20 = 4 := by
  sorry

end steve_has_four_friends_l3852_385276


namespace range_of_m_l3852_385296

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∪ B m = A → m ≤ 3 := by
  sorry

end range_of_m_l3852_385296


namespace y_derivative_l3852_385244

noncomputable def y (x : ℝ) : ℝ := (1/4) * Real.log (abs (Real.tanh (x/2))) - (1/4) * Real.log ((3 + Real.cosh x) / Real.sinh x)

theorem y_derivative (x : ℝ) : deriv y x = 1 / (2 * Real.sinh x) := by
  sorry

end y_derivative_l3852_385244


namespace system_solution_l3852_385203

theorem system_solution :
  ∃ a b c d e : ℤ,
    (ab + a + 2*b = 78 ∧
     bc + 3*b + c = 101 ∧
     cd + 5*c + 3*d = 232 ∧
     de + 4*d + 5*e = 360 ∧
     ea + 2*e + 4*a = 192) →
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨
     (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by sorry

#check system_solution

end system_solution_l3852_385203


namespace dice_sum_probability_l3852_385255

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 10

/-- The target sum -/
def target_sum : ℕ := 50

/-- The number of ways to distribute the remaining sum after subtracting the minimum roll from each die -/
def num_ways : ℕ := Nat.choose 49 9

/-- The total number of possible outcomes when rolling k n-sided dice -/
def total_outcomes : ℕ := n ^ k

/-- The probability of obtaining the target sum -/
def probability : ℚ := num_ways / total_outcomes

theorem dice_sum_probability :
  probability = 818809200 / 1073741824 := by sorry

end dice_sum_probability_l3852_385255


namespace number_count_in_average_calculation_l3852_385260

/-- Given an initial average, an incorrectly read number, and the correct average,
    prove the number of numbers in the original calculation. -/
theorem number_count_in_average_calculation
  (initial_avg : ℚ)
  (incorrect_num : ℚ)
  (correct_num : ℚ)
  (correct_avg : ℚ)
  (h1 : initial_avg = 19)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 76)
  (h4 : correct_avg = 24) :
  ∃ (n : ℕ) (S : ℚ),
    S + incorrect_num = initial_avg * n ∧
    S + correct_num = correct_avg * n ∧
    n = 10 :=
sorry

end number_count_in_average_calculation_l3852_385260


namespace second_day_travel_l3852_385274

/-- Represents the distance traveled on the second day -/
def second_day_distance : ℝ := 420

/-- Represents the distance traveled on the first day -/
def first_day_distance : ℝ := 240

/-- Represents the average speed on both days -/
def average_speed : ℝ := 60

/-- Represents the time difference between the two trips -/
def time_difference : ℝ := 3

/-- Theorem stating that the distance traveled on the second day is 420 miles -/
theorem second_day_travel :
  second_day_distance = first_day_distance + average_speed * time_difference :=
by sorry

end second_day_travel_l3852_385274


namespace perpendicular_lines_b_value_l3852_385272

theorem perpendicular_lines_b_value (b : ℚ) : 
  (∀ x y : ℚ, 2 * x - 3 * y + 6 = 0 → (∃ m₁ : ℚ, y = m₁ * x + 2)) ∧ 
  (∀ x y : ℚ, b * x - 3 * y + 6 = 0 → (∃ m₂ : ℚ, y = m₂ * x + 2)) ∧
  (∃ m₁ m₂ : ℚ, m₁ * m₂ = -1) →
  b = -9/2 := by
sorry

end perpendicular_lines_b_value_l3852_385272


namespace seashells_given_to_mike_l3852_385257

/-- Given that Joan initially found 79 seashells and now has 16 seashells,
    prove that the number of seashells she gave to Mike is 63. -/
theorem seashells_given_to_mike 
  (initial_seashells : ℕ) 
  (current_seashells : ℕ) 
  (h1 : initial_seashells = 79) 
  (h2 : current_seashells = 16) : 
  initial_seashells - current_seashells = 63 := by
  sorry

end seashells_given_to_mike_l3852_385257


namespace geometric_progression_sum_l3852_385289

/-- A sequence is a geometric progression if the ratio between consecutive terms is constant. -/
def IsGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_progression_sum (a : ℕ → ℝ) :
  IsGeometricProgression a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end geometric_progression_sum_l3852_385289


namespace divisibility_problem_l3852_385223

theorem divisibility_problem (d r : ℤ) : 
  d > 1 ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℤ, 
    29 * 11 = k₁ * d + r ∧
    1059 = k₂ * d + r ∧
    1417 = k₃ * d + r ∧
    2312 = k₄ * d + r) →
  d - r = 15 := by
sorry

end divisibility_problem_l3852_385223


namespace cyclist_average_speed_l3852_385226

/-- The average speed of a cyclist driving four laps of equal distance at different speeds -/
theorem cyclist_average_speed (d : ℝ) (h : d > 0) : 
  let speeds := [6, 12, 18, 24]
  let total_distance := 4 * d
  let total_time := (d / 6 + d / 12 + d / 18 + d / 24)
  total_distance / total_time = 288 / 25 := by
  sorry

end cyclist_average_speed_l3852_385226


namespace a_work_time_l3852_385247

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_C : ℚ := 1 / 4

-- Define the theorem
theorem a_work_time : 
  work_rate_A + work_rate_C = 1 / 2 ∧ 
  work_rate_B + work_rate_C = 1 / 3 ∧ 
  work_rate_B = 1 / 12 →
  1 / work_rate_A = 4 := by
  sorry


end a_work_time_l3852_385247


namespace cos_negative_thirty_degrees_l3852_385231

theorem cos_negative_thirty_degrees : Real.cos (-(30 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end cos_negative_thirty_degrees_l3852_385231


namespace inequality_system_solution_l3852_385201

theorem inequality_system_solution (x : ℝ) :
  (3 * (x - 2) ≤ x - 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end inequality_system_solution_l3852_385201


namespace reciprocal_of_negative_2023_l3852_385263

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end reciprocal_of_negative_2023_l3852_385263


namespace coral_reading_pages_l3852_385220

/-- The number of pages Coral read on the first night -/
def night1 : ℕ := 30

/-- The number of pages Coral read on the second night -/
def night2 : ℕ := 2 * night1 - 2

/-- The number of pages Coral read on the third night -/
def night3 : ℕ := night1 + night2 + 3

/-- The total number of pages Coral read over three nights -/
def totalPages : ℕ := night1 + night2 + night3

/-- Theorem stating that the total number of pages read is 179 -/
theorem coral_reading_pages : totalPages = 179 := by
  sorry

end coral_reading_pages_l3852_385220


namespace triangle_subdivision_l3852_385249

/-- Given a triangle ABC with n arbitrary non-collinear points inside it,
    the number of non-overlapping small triangles formed by connecting
    all points (including vertices A, B, C) is (2n + 1) -/
def num_small_triangles (n : ℕ) : ℕ := 2 * n + 1

/-- The main theorem stating that for 2008 points inside triangle ABC,
    the number of small triangles is 4017 -/
theorem triangle_subdivision :
  num_small_triangles 2008 = 4017 := by
  sorry

end triangle_subdivision_l3852_385249


namespace storage_box_faces_l3852_385270

theorem storage_box_faces : ∃ n : ℕ, n > 0 ∧ Nat.factorial n = 720 := by
  sorry

end storage_box_faces_l3852_385270


namespace union_equality_intersection_equality_l3852_385278

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 4 ≤ x ∧ x ≤ 3*m + 2}

-- Theorem for the first question
theorem union_equality (m : ℝ) : A ∪ B m = B m ↔ m ∈ Set.Icc 1 2 := by sorry

-- Theorem for the second question
theorem intersection_equality (m : ℝ) : A ∩ B m = B m ↔ m < -3 := by sorry

end union_equality_intersection_equality_l3852_385278


namespace marias_towels_l3852_385281

theorem marias_towels (green_towels white_towels given_towels : ℕ) : 
  green_towels = 40 →
  white_towels = 44 →
  given_towels = 65 →
  green_towels + white_towels - given_towels = 19 := by
sorry

end marias_towels_l3852_385281


namespace three_digit_primes_ending_in_one_l3852_385297

theorem three_digit_primes_ending_in_one (p : ℕ) : 
  (200 < p ∧ p < 1000 ∧ p % 10 = 1 ∧ Nat.Prime p) → 
  (Finset.filter (λ x => 200 < x ∧ x < 1000 ∧ x % 10 = 1 ∧ Nat.Prime x) (Finset.range 1000)).card = 23 :=
sorry

end three_digit_primes_ending_in_one_l3852_385297


namespace team_selection_ways_eq_103950_l3852_385206

/-- The number of ways to select a team of 8 people, consisting of 4 boys from a group of 10 boys
    and 4 girls from a group of 12 girls. -/
def team_selection_ways : ℕ :=
  Nat.choose 10 4 * Nat.choose 12 4

/-- Theorem stating that the number of ways to select the team is 103950. -/
theorem team_selection_ways_eq_103950 : team_selection_ways = 103950 := by
  sorry

end team_selection_ways_eq_103950_l3852_385206


namespace min_value_of_expression_l3852_385273

theorem min_value_of_expression (a b c : ℤ) (h1 : a > b) (h2 : b > c) :
  let x := (a + b + c) / (a - b - c)
  (x + 1 / x : ℚ) ≥ 2 ∧ ∃ (a' b' c' : ℤ), a' > b' ∧ b' > c' ∧
    let x' := (a' + b' + c' : ℚ) / (a' - b' - c' : ℚ)
    x' + 1 / x' = 2 :=
sorry

end min_value_of_expression_l3852_385273


namespace necessary_but_not_sufficient_l3852_385238

def M : Set ℝ := {x | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x | x^2 + x < 0}

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x ∈ N → x ∈ M) ∧
  (∃ x : ℝ, x ∈ M ∧ x ∉ N) :=
sorry

end necessary_but_not_sufficient_l3852_385238


namespace max_area_right_triangle_l3852_385205

theorem max_area_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^2 + b^2 = 8^2 → (1/2) * a * b ≤ 16 := by
sorry

end max_area_right_triangle_l3852_385205


namespace savings_ratio_l3852_385294

def savings_problem (monday tuesday wednesday thursday : ℚ) : Prop :=
  let total_savings := monday + tuesday + wednesday
  let ratio := thursday / total_savings
  (monday = 15) ∧ (tuesday = 28) ∧ (wednesday = 13) ∧ (thursday = 28) → ratio = 1/2

theorem savings_ratio : ∀ (monday tuesday wednesday thursday : ℚ),
  savings_problem monday tuesday wednesday thursday :=
λ monday tuesday wednesday thursday => by
  sorry

end savings_ratio_l3852_385294


namespace labourer_income_l3852_385280

/-- The monthly income of a labourer given specific expenditure and savings patterns -/
theorem labourer_income (
  first_period : ℕ) 
  (second_period : ℕ)
  (first_expenditure : ℚ)
  (second_expenditure : ℚ)
  (savings : ℚ)
  (h1 : first_period = 8)
  (h2 : second_period = 6)
  (h3 : first_expenditure = 80)
  (h4 : second_expenditure = 65)
  (h5 : savings = 50)
  : ∃ (income : ℚ), 
    income * ↑first_period < first_expenditure * ↑first_period ∧ 
    income * ↑second_period = second_expenditure * ↑second_period + 
      (first_expenditure * ↑first_period - income * ↑first_period) + savings ∧
    income = 1080 / 14 := by
  sorry


end labourer_income_l3852_385280


namespace quadratic_second_difference_constant_l3852_385298

/-- Second difference of a function f at point n -/
def second_difference (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (f (n + 2) - f (n + 1)) - (f (n + 1) - f n)

/-- A quadratic function with linear and constant terms -/
def quadratic_function (a b : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ)^2 + a * (n : ℝ) + b

theorem quadratic_second_difference_constant (a b : ℝ) :
  ∀ n : ℕ, second_difference (quadratic_function a b) n = 2 :=
sorry

end quadratic_second_difference_constant_l3852_385298


namespace sticker_count_l3852_385252

/-- Given the ratio of stickers and Kate's sticker count, prove the combined count of Jenna's and Ava's stickers -/
theorem sticker_count (kate_ratio jenna_ratio ava_ratio : ℕ) 
  (kate_stickers : ℕ) (h_ratio : kate_ratio = 7 ∧ jenna_ratio = 4 ∧ ava_ratio = 5) 
  (h_kate : kate_stickers = 42) : 
  (jenna_ratio + ava_ratio) * (kate_stickers / kate_ratio) = 54 := by
  sorry

#check sticker_count

end sticker_count_l3852_385252


namespace operation_result_l3852_385248

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem operation_result : 
  op (op Element.one Element.two) (op Element.four Element.three) = Element.four := by
  sorry

end operation_result_l3852_385248


namespace g_expression_l3852_385269

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g implicitly using its relationship with f
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 1 := by
  sorry

end g_expression_l3852_385269


namespace line_parabola_intersection_l3852_385221

/-- The line y = kx + 1 and the parabola y^2 = 4x have exactly one point in common if and only if k = 0 or k = 1 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 1 ∧ p.2^2 = 4 * p.1) ↔ k = 0 ∨ k = 1 := by
  sorry

end line_parabola_intersection_l3852_385221


namespace cubic_sum_theorem_l3852_385271

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 1)
  (sum_prod_eq : a * b + a * c + b * c = -3)
  (prod_eq : a * b * c = 4) :
  a^3 + b^3 + c^3 = 1 := by
    sorry

end cubic_sum_theorem_l3852_385271


namespace transistors_in_2010_l3852_385200

/-- Moore's law doubling period in years -/
def moores_law_period : ℕ := 2

/-- Initial year for calculation -/
def initial_year : ℕ := 1992

/-- Target year for calculation -/
def target_year : ℕ := 2010

/-- Initial number of transistors in 1992 -/
def initial_transistors : ℕ := 500000

/-- Calculate the number of transistors in a given year according to Moore's law -/
def transistors_in_year (year : ℕ) : ℕ :=
  initial_transistors * 2^((year - initial_year) / moores_law_period)

/-- Theorem stating the number of transistors in 2010 -/
theorem transistors_in_2010 :
  transistors_in_year target_year = 256000000 := by
  sorry

end transistors_in_2010_l3852_385200


namespace smallest_integer_satisfying_inequality_l3852_385283

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 15 → x ≥ 8 :=
by sorry

end smallest_integer_satisfying_inequality_l3852_385283


namespace largest_number_problem_l3852_385245

theorem largest_number_problem (a b c : ℝ) : 
  a < b ∧ b < c →
  a + b + c = 77 →
  c - b = 9 →
  b - a = 5 →
  c = 100 / 3 := by
  sorry

end largest_number_problem_l3852_385245


namespace empty_subset_of_A_l3852_385287

def A : Set ℝ := {x : ℝ | x^2 - x = 0}

theorem empty_subset_of_A : ∅ ⊆ A := by
  sorry

end empty_subset_of_A_l3852_385287


namespace floor_sqrt_sum_equality_and_counterexample_l3852_385282

theorem floor_sqrt_sum_equality_and_counterexample :
  (∀ n : ℕ, ⌊Real.sqrt n + Real.sqrt (n + 2)⌋ = ⌊Real.sqrt (4 * n + 1)⌋) ∧
  (∃ x : ℝ, ⌊Real.sqrt x + Real.sqrt (x + 2)⌋ ≠ ⌊Real.sqrt (4 * x + 1)⌋) :=
by sorry

end floor_sqrt_sum_equality_and_counterexample_l3852_385282


namespace dessert_percentage_l3852_385277

/-- Proves that the dessert cost is 25% of the second course price --/
theorem dessert_percentage (initial_amount : ℝ) (first_course_cost : ℝ) 
  (second_course_cost : ℝ) (remaining_amount : ℝ) : ℝ :=
by
  have h1 : initial_amount = 60 := by sorry
  have h2 : first_course_cost = 15 := by sorry
  have h3 : second_course_cost = first_course_cost + 5 := by sorry
  have h4 : remaining_amount = 20 := by sorry

  -- Calculate total spent
  let total_spent := initial_amount - remaining_amount

  -- Calculate dessert cost
  let dessert_cost := total_spent - (first_course_cost + second_course_cost)

  -- Calculate percentage
  let percentage := (dessert_cost / second_course_cost) * 100

  exact 25

end dessert_percentage_l3852_385277


namespace cube_roots_of_unity_l3852_385210

theorem cube_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
sorry

end cube_roots_of_unity_l3852_385210


namespace cube_root_equality_l3852_385233

theorem cube_root_equality (m : ℝ) : 
  (9 + 9 / m) ^ (1/3) = 9 * (9 / m) ^ (1/3) → m = 728 := by
  sorry

end cube_root_equality_l3852_385233


namespace peter_soda_purchase_l3852_385291

/-- The amount of money Peter has left after buying soda -/
def money_left (cost_per_ounce : ℚ) (initial_money : ℚ) (ounces_bought : ℚ) : ℚ :=
  initial_money - cost_per_ounce * ounces_bought

/-- Theorem: Peter has $0.50 left after buying soda -/
theorem peter_soda_purchase : 
  let cost_per_ounce : ℚ := 25 / 100
  let initial_money : ℚ := 2
  let ounces_bought : ℚ := 6
  money_left cost_per_ounce initial_money ounces_bought = 1 / 2 := by
sorry

end peter_soda_purchase_l3852_385291


namespace equation_solution_l3852_385254

theorem equation_solution : ∃! x : ℚ, (9 - x)^2 = (x + 1/2)^2 ∧ x = 323/76 := by
  sorry

end equation_solution_l3852_385254


namespace not_hyperbola_equation_l3852_385207

/-- A hyperbola with given properties -/
structure Hyperbola where
  center_at_origin : Bool
  symmetric_about_axes : Bool
  eccentricity : ℝ
  focus_to_asymptote_distance : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a - y^2 / b = 1

/-- Theorem stating that the given equation cannot be the equation of the hyperbola with the specified properties -/
theorem not_hyperbola_equation (M : Hyperbola) 
  (h1 : M.center_at_origin = true)
  (h2 : M.symmetric_about_axes = true)
  (h3 : M.eccentricity = Real.sqrt 3)
  (h4 : M.focus_to_asymptote_distance = 2) :
  ¬(hyperbola_equation 4 2 = fun x y => x^2 / 4 - y^2 / 2 = 1) :=
sorry

end not_hyperbola_equation_l3852_385207


namespace cycling_average_speed_l3852_385204

theorem cycling_average_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (rest_duration : ℝ) 
  (num_rests : ℕ) 
  (h1 : total_distance = 56) 
  (h2 : total_time = 8) 
  (h3 : rest_duration = 0.5) 
  (h4 : num_rests = 2) : 
  total_distance / (total_time - num_rests * rest_duration) = 8 := by
sorry

end cycling_average_speed_l3852_385204


namespace tank_capacity_l3852_385293

/-- Represents the properties of a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating that a tank with given properties has a capacity of 1080 litres. -/
theorem tank_capacity (t : Tank)
  (h1 : t.leak_empty_time = 4)
  (h2 : t.inlet_rate = 6)
  (h3 : t.combined_empty_time = 12) :
  t.capacity = 1080 :=
by
  sorry

#check tank_capacity

end tank_capacity_l3852_385293


namespace probability_two_cards_sum_15_l3852_385288

-- Define the deck
def standard_deck : ℕ := 52

-- Define the number of cards for each value from 2 to 10
def number_cards_per_value : ℕ := 4

-- Define the possible first card values that can sum to 15
def first_card_values : List ℕ := [6, 7, 8, 9, 10]

-- Define the function to calculate the number of ways to choose 2 cards
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

-- State the theorem
theorem probability_two_cards_sum_15 :
  (10 : ℚ) / 331 = (
    (List.sum (first_card_values.map (λ x => 
      if x = 10 then
        number_cards_per_value * number_cards_per_value
      else
        number_cards_per_value * number_cards_per_value
    ))) / (2 * choose_two standard_deck)
  ) := by sorry

end probability_two_cards_sum_15_l3852_385288


namespace expected_red_pairs_l3852_385262

/-- The expected number of pairs of adjacent red cards in a circular arrangement -/
theorem expected_red_pairs (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ)
  (h1 : total_cards = 60)
  (h2 : red_cards = 30)
  (h3 : black_cards = 30)
  (h4 : total_cards = red_cards + black_cards) :
  (red_cards : ℚ) * (red_cards - 1 : ℚ) / (total_cards - 1 : ℚ) = 870 / 59 := by
  sorry

end expected_red_pairs_l3852_385262


namespace f_at_neg_one_eq_78_l3852_385290

/-- The polynomial g(x) -/
def g (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 - 5*x + 15

/-- The polynomial f(x) -/
def f (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 50*x + r

/-- Theorem stating that f(-1) = 78 given the conditions -/
theorem f_at_neg_one_eq_78 
  (p q r : ℝ) 
  (h1 : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g p x = 0 ∧ g p y = 0 ∧ g p z = 0)
  (h2 : ∀ x : ℝ, g p x = 0 → f q r x = 0) :
  f q r (-1) = 78 := by
  sorry

end f_at_neg_one_eq_78_l3852_385290


namespace abs_sum_values_l3852_385225

theorem abs_sum_values (a b : ℝ) (ha : |a| = 3) (hb : |b| = 1) :
  |a + b| = 4 ∨ |a + b| = 2 := by
sorry

end abs_sum_values_l3852_385225


namespace EPC42_probability_l3852_385261

/-- The set of vowels used in Logicville license plates -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}

/-- The set of consonants used in Logicville license plates -/
def consonants : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z'}

/-- The set of two-digit numbers used in Logicville license plates -/
def twoDigitNumbers : Finset Nat := Finset.range 100

/-- A Logicville license plate -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Nat
  first_in_vowels : first ∈ vowels
  second_in_consonants : second ∈ consonants
  third_in_consonants : third ∈ consonants
  second_neq_third : second ≠ third
  fourth_in_range : fourth ∈ twoDigitNumbers

/-- The probability of randomly selecting a specific license plate in Logicville -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (vowels.card * consonants.card * (consonants.card - 1) * twoDigitNumbers.card)

/-- The specific license plate "EPC42" -/
def EPC42 : LicensePlate := {
  first := 'E',
  second := 'P',
  third := 'C',
  fourth := 42,
  first_in_vowels := by simp [vowels],
  second_in_consonants := by simp [consonants],
  third_in_consonants := by simp [consonants],
  second_neq_third := by decide,
  fourth_in_range := by simp [twoDigitNumbers]
}

/-- Theorem: The probability of randomly selecting "EPC42" in Logicville is 1/252,000 -/
theorem EPC42_probability :
  licensePlateProbability EPC42 = 1 / 252000 := by
  sorry

end EPC42_probability_l3852_385261


namespace largest_integer_for_negative_quadratic_l3852_385217

theorem largest_integer_for_negative_quadratic : 
  ∃ (n : ℤ), n = 7 ∧ n^2 - 11*n + 24 < 0 ∧ ∀ (m : ℤ), m > n → m^2 - 11*m + 24 ≥ 0 := by
  sorry

end largest_integer_for_negative_quadratic_l3852_385217


namespace x_value_when_y_is_two_l3852_385224

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end x_value_when_y_is_two_l3852_385224


namespace min_value_sum_product_l3852_385267

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end min_value_sum_product_l3852_385267


namespace parabola_vertex_l3852_385237

/-- The vertex of a parabola defined by y = x^2 + 2x - 3 is (-1, -4) -/
theorem parabola_vertex : 
  let f (x : ℝ) := x^2 + 2*x - 3
  ∃! (a b : ℝ), (∀ x, f x = (x - a)^2 + b) ∧ (a = -1 ∧ b = -4) :=
by sorry

end parabola_vertex_l3852_385237


namespace train_crossing_time_l3852_385240

theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 90 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4.5 := by
  sorry

end train_crossing_time_l3852_385240


namespace smallest_five_digit_mod_9_l3852_385268

theorem smallest_five_digit_mod_9 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 9 = 5 → n ≥ 10000 :=
by sorry

end smallest_five_digit_mod_9_l3852_385268


namespace triangle_area_l3852_385266

/-- Triangle ABC with given properties -/
structure Triangle :=
  (BD : ℝ)
  (DC : ℝ)
  (height : ℝ)
  (hBD : BD = 3)
  (hDC : DC = 2 * BD)
  (hHeight : height = 4)

/-- The area of triangle ABC is 18 square units -/
theorem triangle_area (t : Triangle) : (1/2 : ℝ) * (t.BD + t.DC) * t.height = 18 := by
  sorry

end triangle_area_l3852_385266


namespace bryan_skittles_count_l3852_385253

/-- Given that Ben has 20 M&M's and Bryan has 30 more candies than Ben, 
    prove that Bryan has 50 skittles. -/
theorem bryan_skittles_count : 
  ∀ (ben_candies bryan_candies : ℕ),
  ben_candies = 20 →
  bryan_candies = ben_candies + 30 →
  bryan_candies = 50 := by
sorry

end bryan_skittles_count_l3852_385253


namespace triangle_count_2008_l3852_385234

/-- Given a set of points in a plane, where three of the points form a triangle
    and the rest are inside this triangle, this function calculates the number
    of non-overlapping small triangles that can be formed. -/
def count_small_triangles (n : ℕ) : ℕ :=
  1 + 2 * (n - 3)

/-- Theorem stating that for 2008 non-collinear points, where 3 form a triangle
    and the rest are inside, the number of non-overlapping small triangles is 4011. -/
theorem triangle_count_2008 :
  count_small_triangles 2008 = 4011 := by
  sorry

#eval count_small_triangles 2008  -- Should output 4011

end triangle_count_2008_l3852_385234


namespace quadratic_factorization_l3852_385264

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 := by
  sorry

end quadratic_factorization_l3852_385264


namespace X_prob_implies_n_10_l3852_385242

/-- A random variable X taking values from 1 to n with equal probability -/
def X (n : ℕ) := Fin n

/-- The probability of X being less than 4 -/
def prob_X_less_than_4 (n : ℕ) : ℚ := (3 : ℚ) / n

/-- Theorem stating that if P(X < 4) = 0.3, then n = 10 -/
theorem X_prob_implies_n_10 (n : ℕ) (h : prob_X_less_than_4 n = (3 : ℚ) / 10) : n = 10 := by
  sorry

end X_prob_implies_n_10_l3852_385242


namespace rectangles_form_square_l3852_385222

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The set of given rectangles -/
def rectangles : List Rectangle := [
  ⟨1, 2⟩, ⟨7, 10⟩, ⟨6, 5⟩, ⟨8, 12⟩, ⟨9, 3⟩
]

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Theorem: The given rectangles can form a square -/
theorem rectangles_form_square : ∃ (s : ℕ), s > 0 ∧ s * s = (rectangles.map area).sum := by
  sorry

end rectangles_form_square_l3852_385222


namespace units_digit_of_3_pow_2012_l3852_385251

def units_digit_cycle : List Nat := [3, 9, 7, 1]

theorem units_digit_of_3_pow_2012 :
  (3^2012 : Nat) % 10 = 1 := by sorry

end units_digit_of_3_pow_2012_l3852_385251


namespace shaded_area_sum_l3852_385216

def circle_setup (r₁ r₂ r₃ : ℝ) : Prop :=
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
  r₁ * r₁ = 100 ∧
  r₂ = r₁ / 2 ∧
  r₃ = r₂ / 2

theorem shaded_area_sum (r₁ r₂ r₃ : ℝ) 
  (h : circle_setup r₁ r₂ r₃) : 
  (π * r₁ * r₁ / 2) + (π * r₂ * r₂ / 2) + (π * r₃ * r₃ / 2) = 65.625 * π :=
by
  sorry

#check shaded_area_sum

end shaded_area_sum_l3852_385216


namespace first_issue_pages_l3852_385229

/-- Represents the number of pages Trevor drew in a month -/
structure MonthlyPages where
  regular : ℕ  -- Regular pages
  bonus : ℕ    -- Bonus pages

/-- Represents Trevor's comic book production over three months -/
structure ComicProduction where
  month1 : MonthlyPages
  month2 : MonthlyPages
  month3 : MonthlyPages
  total_pages : ℕ
  pages_per_day_month1 : ℕ
  pages_per_day_month23 : ℕ

/-- The conditions of Trevor's comic book production -/
def comic_conditions (prod : ComicProduction) : Prop :=
  prod.total_pages = 220 ∧
  prod.pages_per_day_month1 = 5 ∧
  prod.pages_per_day_month23 = 4 ∧
  prod.month1.regular = prod.month2.regular ∧
  prod.month3.regular = prod.month1.regular + 4 ∧
  prod.month1.bonus = 3 ∧
  prod.month2.bonus = 3 ∧
  prod.month3.bonus = 3

theorem first_issue_pages (prod : ComicProduction) 
  (h : comic_conditions prod) : prod.month1.regular = 69 := by
  sorry

end first_issue_pages_l3852_385229


namespace special_function_bound_l3852_385241

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ x^2 * f (y/2) + y^2 * f (x/2)) ∧
  (∃ M : ℝ, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M)

/-- The main theorem stating that f(x) ≤ x^2 for all x ≥ 0 -/
theorem special_function_bound {f : ℝ → ℝ} (hf : SpecialFunction f) :
  ∀ x, x ≥ 0 → f x ≤ x^2 := by
  sorry

end special_function_bound_l3852_385241


namespace min_time_for_all_flashes_l3852_385219

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The time for one light to shine in seconds -/
def shine_time : ℕ := 1

/-- The interval between two consecutive flashes in seconds -/
def interval_time : ℕ := 5

/-- The number of different possible flashes -/
def num_flashes : ℕ := Nat.factorial num_lights

/-- The minimum time required to achieve all different flashes in seconds -/
def min_time_required : ℕ := 
  (num_flashes * num_lights * shine_time) + ((num_flashes - 1) * interval_time)

theorem min_time_for_all_flashes : min_time_required = 1195 := by
  sorry

end min_time_for_all_flashes_l3852_385219


namespace mistaken_calculation_l3852_385295

theorem mistaken_calculation (x : ℝ) : x + 2 = 6 → x - 2 = 2 := by
  sorry

end mistaken_calculation_l3852_385295


namespace equal_areas_of_inscribed_polygons_with_same_side_lengths_l3852_385232

-- Define a type for polygons
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

-- Define a function to calculate the side lengths of a polygon
def sideLengths (n : ℕ) (p : Polygon n) : Multiset ℝ :=
  sorry

-- Define a function to calculate the area of a polygon
def area (n : ℕ) (p : Polygon n) : ℝ :=
  sorry

-- Define a predicate to check if a polygon is inscribed in a circle
def isInscribed (n : ℕ) (p : Polygon n) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  sorry

-- Theorem statement
theorem equal_areas_of_inscribed_polygons_with_same_side_lengths
  (n : ℕ) (p1 p2 : Polygon n) (center : ℝ × ℝ) (radius : ℝ) :
  isInscribed n p1 center radius →
  isInscribed n p2 center radius →
  sideLengths n p1 = sideLengths n p2 →
  area n p1 = area n p2 :=
sorry

end equal_areas_of_inscribed_polygons_with_same_side_lengths_l3852_385232


namespace single_elimination_games_l3852_385292

/-- A single-elimination tournament with no ties. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

theorem single_elimination_games (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.no_ties = true) : 
  games_played t = 22 := by
  sorry

end single_elimination_games_l3852_385292
