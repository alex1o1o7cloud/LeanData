import Mathlib

namespace smallest_valid_sum_of_cubes_l1971_197127

def is_valid (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p ∣ n → p > 18

def is_sum_of_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a^3 + b^3

theorem smallest_valid_sum_of_cubes : 
  is_valid 1843 ∧ 
  is_sum_of_cubes 1843 ∧ 
  ∀ m : ℕ, m < 1843 → ¬(is_valid m ∧ is_sum_of_cubes m) :=
sorry

end smallest_valid_sum_of_cubes_l1971_197127


namespace base_10_to_base_7_conversion_l1971_197107

theorem base_10_to_base_7_conversion :
  ∃ (a b c d : ℕ),
    746 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 4 := by
  sorry

end base_10_to_base_7_conversion_l1971_197107


namespace smallest_cube_with_specific_digits_l1971_197140

/-- Returns the first n digits of a natural number -/
def firstNDigits (n : ℕ) (x : ℕ) : ℕ := sorry

/-- Returns the last n digits of a natural number -/
def lastNDigits (n : ℕ) (x : ℕ) : ℕ := sorry

/-- Checks if the first n digits of a natural number are all 1 -/
def firstNDigitsAreOne (n : ℕ) (x : ℕ) : Prop :=
  firstNDigits n x = 10^n - 1

/-- Checks if the last n digits of a natural number are all 1 -/
def lastNDigitsAreOne (n : ℕ) (x : ℕ) : Prop :=
  lastNDigits n x = 10^n - 1

theorem smallest_cube_with_specific_digits :
  ∀ x : ℕ, x ≥ 1038471 →
    (firstNDigitsAreOne 3 (x^3) ∧ lastNDigitsAreOne 4 (x^3)) →
    x = 1038471 := by sorry

end smallest_cube_with_specific_digits_l1971_197140


namespace fabric_sales_fraction_l1971_197122

theorem fabric_sales_fraction (total_sales stationery_sales : ℕ) 
  (h1 : total_sales = 36)
  (h2 : stationery_sales = 15)
  (h3 : ∃ jewelry_sales : ℕ, jewelry_sales = total_sales / 4)
  (h4 : ∃ fabric_sales : ℕ, fabric_sales + total_sales / 4 + stationery_sales = total_sales) :
  ∃ fabric_sales : ℕ, (fabric_sales : ℚ) / total_sales = 1 / 3 := by
  sorry

end fabric_sales_fraction_l1971_197122


namespace system_solution_l1971_197199

theorem system_solution (x y b : ℚ) : 
  (4 * x + 3 * y = b) →
  (3 * x + 4 * y = 3 * b) →
  (x = 3) →
  (b = -21 / 5) := by
sorry

end system_solution_l1971_197199


namespace power_four_times_four_equals_eight_l1971_197193

theorem power_four_times_four_equals_eight (a : ℝ) : a ^ 4 * a ^ 4 = a ^ 8 := by
  sorry

end power_four_times_four_equals_eight_l1971_197193


namespace lady_arrangements_proof_l1971_197194

def num_gentlemen : ℕ := 6
def num_ladies : ℕ := 3
def total_positions : ℕ := 9

def valid_arrangements : ℕ := 129600

theorem lady_arrangements_proof :
  (num_gentlemen + num_ladies = total_positions) →
  (valid_arrangements = num_gentlemen.factorial * (num_gentlemen + 1).choose num_ladies) :=
by sorry

end lady_arrangements_proof_l1971_197194


namespace fair_queue_size_l1971_197154

/-- Calculates the final number of people in a queue after changes -/
def final_queue_size (initial : ℕ) (left : ℕ) (joined : ℕ) : ℕ :=
  initial - left + joined

/-- Theorem: Given the specific scenario, the final queue size is 6 -/
theorem fair_queue_size : final_queue_size 9 6 3 = 6 := by
  sorry

end fair_queue_size_l1971_197154


namespace second_meeting_time_is_six_minutes_l1971_197184

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the pool and the swimming scenario --/
structure Pool where
  length : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting --/
def secondMeetingTime (pool : Pool) : ℝ :=
  sorry

/-- Theorem stating the conditions and the result to be proved --/
theorem second_meeting_time_is_six_minutes 
  (pool : Pool)
  (h1 : pool.length = 120)
  (h2 : pool.swimmer1.startPosition = 0)
  (h3 : pool.swimmer2.startPosition = 120)
  (h4 : pool.firstMeetingPosition = 40)
  (h5 : pool.firstMeetingTime = 2) :
  secondMeetingTime pool = 6 := by
  sorry

end second_meeting_time_is_six_minutes_l1971_197184


namespace line_through_point_l1971_197117

/-- Given a line ax + (a+1)y = a+2 that passes through the point (4, -8), prove that a = -2 -/
theorem line_through_point (a : ℝ) : 
  (∀ x y : ℝ, a * x + (a + 1) * y = a + 2 → x = 4 ∧ y = -8) → 
  a = -2 := by
sorry

end line_through_point_l1971_197117


namespace shortest_path_bound_l1971_197171

/-- Represents an equilateral tetrahedron -/
structure EquilateralTetrahedron where
  /-- The side length of the tetrahedron -/
  side_length : ℝ
  /-- Assertion that the side length is positive -/
  side_length_pos : side_length > 0

/-- Represents a point on the surface of an equilateral tetrahedron -/
structure SurfacePoint (T : EquilateralTetrahedron) where
  /-- Coordinates of the point on the surface -/
  coords : ℝ × ℝ × ℝ

/-- Calculates the shortest path between two points on the surface of an equilateral tetrahedron -/
def shortest_path (T : EquilateralTetrahedron) (p1 p2 : SurfacePoint T) : ℝ :=
  sorry

/-- Calculates the diameter of the circumscribed circle around a face of an equilateral tetrahedron -/
def face_circumcircle_diameter (T : EquilateralTetrahedron) : ℝ :=
  sorry

/-- Theorem: The shortest path between any two points on the surface of an equilateral tetrahedron
    is at most equal to the diameter of the circumscribed circle around a face of the tetrahedron -/
theorem shortest_path_bound (T : EquilateralTetrahedron) (p1 p2 : SurfacePoint T) :
  shortest_path T p1 p2 ≤ face_circumcircle_diameter T :=
  sorry

end shortest_path_bound_l1971_197171


namespace divided_value_problem_l1971_197132

theorem divided_value_problem (x : ℝ) : (6.5 / x) * 12 = 13 → x = 6 := by
  sorry

end divided_value_problem_l1971_197132


namespace negation_equivalence_l1971_197162

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l1971_197162


namespace retailer_loss_percentage_l1971_197158

-- Define the initial conditions
def initial_cost_price_A : ℝ := 800
def initial_retail_price_B : ℝ := 900
def initial_exchange_rate : ℝ := 1.1
def first_discount : ℝ := 0.1
def second_discount : ℝ := 0.15
def sales_tax : ℝ := 0.1
def final_exchange_rate : ℝ := 1.5

-- Define the theorem
theorem retailer_loss_percentage :
  let price_after_first_discount := initial_retail_price_B * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  let price_with_tax := price_after_second_discount * (1 + sales_tax)
  let final_price_A := price_with_tax / final_exchange_rate
  let loss := initial_cost_price_A - final_price_A
  let percentage_loss := loss / initial_cost_price_A * 100
  ∃ ε > 0, abs (percentage_loss - 36.89) < ε :=
by sorry

end retailer_loss_percentage_l1971_197158


namespace albatrocity_to_finchester_distance_l1971_197178

/-- The distance from Albatrocity to Finchester in miles -/
def distance : ℝ := 75

/-- The speed of the pigeon in still air in miles per hour -/
def pigeon_speed : ℝ := 40

/-- The wind speed from Albatrocity to Finchester in miles per hour -/
def wind_speed : ℝ := 10

/-- The time for a round trip without wind in hours -/
def no_wind_time : ℝ := 3.75

/-- The time for a round trip with wind in hours -/
def wind_time : ℝ := 4

theorem albatrocity_to_finchester_distance :
  (2 * distance / pigeon_speed = no_wind_time) ∧
  (distance / (pigeon_speed + wind_speed) + distance / (pigeon_speed - wind_speed) = wind_time) →
  distance = 75 := by sorry

end albatrocity_to_finchester_distance_l1971_197178


namespace cube_root_of_27_l1971_197187

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 := by
  sorry

end cube_root_of_27_l1971_197187


namespace base10_to_base13_172_l1971_197164

/-- Converts a number from base 10 to base 13 --/
def toBase13 (n : ℕ) : List ℕ := sorry

theorem base10_to_base13_172 :
  toBase13 172 = [1, 0, 3] := by sorry

end base10_to_base13_172_l1971_197164


namespace specific_numbers_in_range_range_closed_under_multiplication_l1971_197170

-- Define the polynomial p
def p (m n : ℤ) : ℤ := 2 * m^2 - 6 * m * n + 5 * n^2

-- Define the range of p
def range_p : Set ℤ := {k | ∃ m n : ℤ, p m n = k}

-- List of specific numbers from 1 to 100 that are in the range of p
def specific_numbers : List ℤ := [1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100]

-- Theorem 1: The specific numbers are in the range of p
theorem specific_numbers_in_range : ∀ k ∈ specific_numbers, k ∈ range_p := by sorry

-- Theorem 2: If h and k are in the range of p, then hk is also in the range of p
theorem range_closed_under_multiplication : 
  ∀ h k : ℤ, h ∈ range_p → k ∈ range_p → (h * k) ∈ range_p := by sorry

end specific_numbers_in_range_range_closed_under_multiplication_l1971_197170


namespace number_line_points_l1971_197110

theorem number_line_points (A B : ℝ) : 
  (|A - B| = 4 * Real.sqrt 2) → 
  (A = 3 * Real.sqrt 2) → 
  (B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2) := by
sorry

end number_line_points_l1971_197110


namespace intersection_slope_l1971_197144

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y + 4 = 0

/-- Theorem stating that the slope of the line formed by the intersection points of the two circles is -1 -/
theorem intersection_slope : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle1 x1 y1 ∧ circle1 x2 y2 ∧ 
    circle2 x1 y1 ∧ circle2 x2 y2 ∧ 
    x1 ≠ x2 ∧
    (y2 - y1) / (x2 - x1) = -1 := by
  sorry

end intersection_slope_l1971_197144


namespace toms_allowance_l1971_197181

theorem toms_allowance (allowance : ℝ) : 
  (allowance - allowance / 3 - (allowance - allowance / 3) / 4 = 6) → allowance = 12 := by
  sorry

end toms_allowance_l1971_197181


namespace base_85_congruence_l1971_197157

/-- Converts a base 85 number to base 10 -/
def base85ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 85^i) 0

/-- The base 85 representation of 746392847₈₅ -/
def num : List Nat := [7, 4, 6, 3, 9, 2, 8, 4, 7]

theorem base_85_congruence : 
  ∃ (b : ℕ), 0 ≤ b ∧ b ≤ 20 ∧ (base85ToBase10 num - b) % 17 = 0 → b = 16 := by
  sorry

end base_85_congruence_l1971_197157


namespace g_3_equals_25_l1971_197151

-- Define the function g
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^7 + q * x^3 + r * x + 7

-- State the theorem
theorem g_3_equals_25 (p q r : ℝ) :
  (g p q r (-3) = -11) →
  (∀ x, g p q r x + g p q r (-x) = 14) →
  g p q r 3 = 25 := by
sorry

end g_3_equals_25_l1971_197151


namespace max_value_of_function_l1971_197198

theorem max_value_of_function (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = x * (1 - x^2)) →
  (∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f x ≤ f x₀) →
  (∃ x₀ ∈ Set.Icc 0 1, f x₀ = 2 * Real.sqrt 3 / 9) :=
by sorry

end max_value_of_function_l1971_197198


namespace test_questions_missed_l1971_197119

theorem test_questions_missed (T : ℕ) (X Y : ℝ) : 
  T > 0 → 
  0 ≤ X ∧ X ≤ 100 →
  0 ≤ Y ∧ Y ≤ 100 →
  ∃ (M F : ℕ),
    M = 5 * F ∧
    M + F = 216 ∧
    M = T * (1 - X / 100) ∧
    F = T * (1 - Y / 100) →
  M = 180 := by
sorry

end test_questions_missed_l1971_197119


namespace x_y_values_l1971_197175

theorem x_y_values (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x + y = 30) : x = 6 ∧ y = 24 := by
  sorry

end x_y_values_l1971_197175


namespace coefficient_x4_in_expansion_l1971_197106

theorem coefficient_x4_in_expansion : 
  (Finset.range 9).sum (fun k => Nat.choose 8 k * 3^(8 - k) * if k = 4 then 1 else 0) = 5670 := by
  sorry

end coefficient_x4_in_expansion_l1971_197106


namespace sqrt_three_bounds_l1971_197161

theorem sqrt_three_bounds (n : ℕ+) : 
  (1 + 3 / (n + 1 : ℝ) < Real.sqrt 3) ∧ (Real.sqrt 3 < 1 + 3 / (n : ℝ)) → n = 4 := by
  sorry

end sqrt_three_bounds_l1971_197161


namespace divisibility_by_five_l1971_197149

theorem divisibility_by_five (a b : ℕ) (n : ℕ) : 
  (5 ∣ n^2 - 1) → (5 ∣ a ∨ 5 ∣ b) := by
  sorry

end divisibility_by_five_l1971_197149


namespace fraction_equality_l1971_197196

theorem fraction_equality : (2222 - 2121)^2 / 196 = 52 := by
  sorry

end fraction_equality_l1971_197196


namespace hcf_proof_l1971_197128

/-- Given two positive integers with specific HCF and LCM, prove that their HCF is 20 -/
theorem hcf_proof (a b : ℕ) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 396) (h3 : a = 36) :
  Nat.gcd a b = 20 := by
  sorry

end hcf_proof_l1971_197128


namespace negation_of_proposition_negation_of_specific_proposition_l1971_197191

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l1971_197191


namespace complex_arithmetic_equation_l1971_197167

theorem complex_arithmetic_equation : 
  (22 / 3 : ℚ) - ((12 / 5 + 5 / 3 * 4) / (17 / 10)) = 2 := by
  sorry

end complex_arithmetic_equation_l1971_197167


namespace shortest_distance_on_specific_cone_l1971_197125

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone --/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculate the shortest distance between two points on the surface of a cone --/
def shortestDistanceOnCone (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let c : Cone := { baseRadius := 500, height := 150 * Real.sqrt 7 }
  let p1 : ConePoint := { distanceFromVertex := 100 }
  let p2 : ConePoint := { distanceFromVertex := 300 * Real.sqrt 2 }
  shortestDistanceOnCone c p1 p2 = Real.sqrt (460000 + 60000 * Real.sqrt 2) := by
  sorry

end shortest_distance_on_specific_cone_l1971_197125


namespace store_earnings_l1971_197137

/-- Calculates the total earnings from selling shirts and jeans --/
def total_earnings (shirt_price : ℕ) (shirt_quantity : ℕ) (jeans_quantity : ℕ) : ℕ :=
  let jeans_price := 2 * shirt_price
  shirt_price * shirt_quantity + jeans_price * jeans_quantity

/-- Proves that the total earnings from selling 20 shirts at $10 each and 10 pairs of jeans at twice the price of a shirt is $400 --/
theorem store_earnings : total_earnings 10 20 10 = 400 := by
  sorry

end store_earnings_l1971_197137


namespace trailing_zeros_100_factorial_l1971_197126

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end trailing_zeros_100_factorial_l1971_197126


namespace unique_solution_l1971_197190

theorem unique_solution : ∃! (x y : ℕ+), 
  (x : ℝ)^(y : ℝ) + 3 = (y : ℝ)^(x : ℝ) ∧ 
  3 * (x : ℝ)^(y : ℝ) = (y : ℝ)^(x : ℝ) + 13 ∧
  x = 2 ∧ y = 3 := by
  sorry

end unique_solution_l1971_197190


namespace walking_speed_problem_l1971_197135

/-- Proves that the speed at which a person would have walked is 10 km/hr,
    given the conditions of the problem. -/
theorem walking_speed_problem (actual_distance : ℝ) (additional_distance : ℝ) 
  (actual_speed : ℝ) :
  actual_distance = 20 →
  additional_distance = 20 →
  actual_speed = 5 →
  ∃ (speed : ℝ),
    speed = actual_speed + 5 ∧
    actual_distance / actual_speed = (actual_distance + additional_distance) / speed ∧
    speed = 10 := by
  sorry

end walking_speed_problem_l1971_197135


namespace students_suggesting_both_l1971_197123

/-- Given the total number of students suggesting bacon and the number of students suggesting only bacon,
    prove that the number of students suggesting both mashed potatoes and bacon
    is equal to the difference between these two values. -/
theorem students_suggesting_both (total_bacon : ℕ) (only_bacon : ℕ)
    (h : total_bacon = 569 ∧ only_bacon = 351) :
    total_bacon - only_bacon = 218 := by
  sorry

end students_suggesting_both_l1971_197123


namespace aaron_final_card_count_l1971_197136

/-- Given that Aaron initially has 5 cards and finds 62 more cards,
    prove that Aaron ends up with 67 cards in total. -/
theorem aaron_final_card_count :
  let initial_cards : ℕ := 5
  let found_cards : ℕ := 62
  initial_cards + found_cards = 67 :=
by sorry

end aaron_final_card_count_l1971_197136


namespace tank_capacity_l1971_197103

/-- Represents the flow rate in kiloliters per minute -/
def flow_rate (volume : ℚ) (time : ℚ) : ℚ := volume / time

/-- Calculates the net flow rate into the tank -/
def net_flow_rate (fill_rate drain_rate1 drain_rate2 : ℚ) : ℚ :=
  fill_rate - (drain_rate1 + drain_rate2)

/-- Calculates the amount of water added to the tank -/
def water_added (net_rate : ℚ) (time : ℚ) : ℚ := net_rate * time

/-- Converts kiloliters to liters -/
def kiloliters_to_liters (kl : ℚ) : ℚ := kl * 1000

theorem tank_capacity :
  let fill_rate := flow_rate 1 2
  let drain_rate1 := flow_rate 1 4
  let drain_rate2 := flow_rate 1 6
  let net_rate := net_flow_rate fill_rate drain_rate1 drain_rate2
  let added_water := water_added net_rate 36
  let full_capacity := 2 * added_water
  kiloliters_to_liters full_capacity = 6000 := by
  sorry

end tank_capacity_l1971_197103


namespace max_flour_mass_difference_l1971_197113

/-- The mass of a bag of flour in kg -/
structure FlourBag where
  mass : ℝ
  mass_range : mass ∈ Set.Icc (25 - 0.2) (25 + 0.2)

/-- The maximum difference in mass between two bags of flour -/
def max_mass_difference (bag1 bag2 : FlourBag) : ℝ :=
  |bag1.mass - bag2.mass|

/-- Theorem stating the maximum possible difference in mass between two bags of flour -/
theorem max_flour_mass_difference :
  ∃ (bag1 bag2 : FlourBag), max_mass_difference bag1 bag2 = 0.4 ∧
  ∀ (bag3 bag4 : FlourBag), max_mass_difference bag3 bag4 ≤ 0.4 := by
sorry

end max_flour_mass_difference_l1971_197113


namespace min_absolute_difference_l1971_197179

/-- The minimum absolute difference between n and m, given f(m) = g(n) -/
theorem min_absolute_difference (f g : ℝ → ℝ) (m n : ℝ) : 
  (f = fun x ↦ Real.exp x + 2 * x) →
  (g = fun x ↦ 4 * x) →
  (f m = g n) →
  ∃ (min_diff : ℝ), 
    (∀ (m' n' : ℝ), f m' = g n' → |n' - m'| ≥ min_diff) ∧ 
    (min_diff = 1/2 - 1/2 * Real.log 2) := by
  sorry

end min_absolute_difference_l1971_197179


namespace exists_same_dimensions_l1971_197101

/-- Represents a rectangle with width and height as powers of two -/
structure Rectangle where
  width : Nat
  height : Nat
  width_pow_two : ∃ k : Nat, width = 2^k
  height_pow_two : ∃ k : Nat, height = 2^k

/-- Represents a tiling of a square -/
structure Tiling where
  n : Nat
  rectangles : List Rectangle
  at_least_two : rectangles.length ≥ 2
  covers_square : ∀ (x y : Nat), x < 2^n ∧ y < 2^n → 
    ∃ (r : Rectangle), r ∈ rectangles ∧ x < r.width ∧ y < r.height
  non_overlapping : ∀ (r1 r2 : Rectangle), r1 ∈ rectangles ∧ r2 ∈ rectangles ∧ r1 ≠ r2 →
    ∀ (x y : Nat), ¬(x < r1.width ∧ y < r1.height ∧ x < r2.width ∧ y < r2.height)

/-- Main theorem: There exist at least two rectangles with the same dimensions in any valid tiling -/
theorem exists_same_dimensions (t : Tiling) : 
  ∃ (r1 r2 : Rectangle), r1 ∈ t.rectangles ∧ r2 ∈ t.rectangles ∧ r1 ≠ r2 ∧ 
    r1.width = r2.width ∧ r1.height = r2.height :=
by sorry

end exists_same_dimensions_l1971_197101


namespace late_secondary_spermatocyte_homomorphic_l1971_197111

-- Define the stages of meiosis
inductive MeiosisStage
  | PrimaryMidFirst
  | PrimaryLateFirst
  | SecondaryMidSecond
  | SecondaryLateSecond

-- Define the types of sex chromosome pairs
inductive SexChromosomePair
  | Heteromorphic
  | Homomorphic

-- Define a function that determines the sex chromosome pair for each stage
def sexChromosomePairAtStage (stage : MeiosisStage) : SexChromosomePair :=
  match stage with
  | MeiosisStage.PrimaryMidFirst => SexChromosomePair.Heteromorphic
  | MeiosisStage.PrimaryLateFirst => SexChromosomePair.Heteromorphic
  | MeiosisStage.SecondaryMidSecond => SexChromosomePair.Heteromorphic
  | MeiosisStage.SecondaryLateSecond => SexChromosomePair.Homomorphic

-- State the theorem
theorem late_secondary_spermatocyte_homomorphic :
  ∀ (stage : MeiosisStage),
    sexChromosomePairAtStage stage = SexChromosomePair.Homomorphic
    ↔ stage = MeiosisStage.SecondaryLateSecond :=
by sorry

end late_secondary_spermatocyte_homomorphic_l1971_197111


namespace prism_volume_l1971_197147

/-- The volume of a right rectangular prism with face areas 100, 200, and 300 square units -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 100)
  (h2 : b * c = 200)
  (h3 : c * a = 300) : 
  a * b * c = 1000 * Real.sqrt 6 := by sorry

end prism_volume_l1971_197147


namespace quadratic_inequality_range_l1971_197169

-- Define the quadratic function
def f (a x : ℝ) := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- State the theorem
theorem quadratic_inequality_range :
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-2) 2 := by sorry

end quadratic_inequality_range_l1971_197169


namespace multiplication_difference_l1971_197160

theorem multiplication_difference : 
  let correct_number : ℕ := 134
  let correct_multiplier : ℕ := 43
  let incorrect_multiplier : ℕ := 34
  (correct_number * correct_multiplier) - (correct_number * incorrect_multiplier) = 1206 :=
by
  sorry

end multiplication_difference_l1971_197160


namespace linear_function_property_l1971_197124

theorem linear_function_property (x y : ℝ) : 
  let f : ℝ → ℝ := fun x => 3 * x
  f ((x + y) / 2) = (1 / 2) * (f x + f y) := by
  sorry

end linear_function_property_l1971_197124


namespace inverse_function_graph_point_l1971_197102

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse function of f
variable (h_inv : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f)

-- Given condition: f(3) = 0
variable (h_f_3 : f 3 = 0)

-- Theorem statement
theorem inverse_function_graph_point :
  (f_inv ((-1) + 1) = 3) ∧ (f_inv ∘ (fun x ↦ x + 1)) (-1) = 3 :=
sorry

end inverse_function_graph_point_l1971_197102


namespace y_intercept_of_line_l1971_197148

/-- The y-intercept of the line 4x + 7y = 28 is the point (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 4 ∧ x = 0 := by
  sorry

end y_intercept_of_line_l1971_197148


namespace total_lunch_combinations_l1971_197129

def meat_dishes : ℕ := 4
def vegetable_dishes : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def case1_combinations : ℕ := (choose meat_dishes 2) * (choose vegetable_dishes 2)
def case2_combinations : ℕ := (choose meat_dishes 1) * (choose vegetable_dishes 2)

theorem total_lunch_combinations : 
  case1_combinations + case2_combinations = 210 :=
by sorry

end total_lunch_combinations_l1971_197129


namespace x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three_l1971_197138

theorem x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three :
  (∀ x : ℝ, x < 0 → x ≠ 3) ∧
  (∃ x : ℝ, x ≠ 3 ∧ x ≥ 0) := by
  sorry

end x_less_than_zero_sufficient_not_necessary_for_x_not_equal_three_l1971_197138


namespace cube_root_of_five_cubed_times_two_to_sixth_l1971_197185

theorem cube_root_of_five_cubed_times_two_to_sixth (x : ℝ) : x^3 = 5^3 * 2^6 → x = 10 := by
  sorry

end cube_root_of_five_cubed_times_two_to_sixth_l1971_197185


namespace total_cost_is_eight_times_shorts_l1971_197109

/-- The cost of football equipment relative to shorts -/
def FootballEquipmentCost (x : ℝ) : Prop :=
  let shorts := x
  let tshirt := x
  let boots := 4 * x
  let shinguards := 2 * x
  (shorts + tshirt = 2 * x) ∧
  (shorts + boots = 5 * x) ∧
  (shorts + shinguards = 3 * x) ∧
  (shorts + tshirt + boots + shinguards = 8 * x)

/-- Theorem: The total cost of all items is 8 times the cost of shorts -/
theorem total_cost_is_eight_times_shorts (x : ℝ) (h : FootballEquipmentCost x) :
  ∃ (shorts tshirt boots shinguards : ℝ),
    shorts = x ∧
    shorts + tshirt + boots + shinguards = 8 * x :=
by
  sorry

end total_cost_is_eight_times_shorts_l1971_197109


namespace haley_money_received_l1971_197159

/-- Proves that Haley received 13 dollars from doing chores and her birthday -/
theorem haley_money_received (initial_amount : ℕ) (difference : ℕ) : 
  initial_amount = 2 → difference = 11 → initial_amount + difference = 13 :=
by
  sorry

end haley_money_received_l1971_197159


namespace scientific_notation_of_35_billion_l1971_197165

-- Define 35 billion
def thirty_five_billion : ℝ := 35000000000

-- Theorem statement
theorem scientific_notation_of_35_billion :
  thirty_five_billion = 3.5 * (10 : ℝ) ^ 10 := by
  sorry

end scientific_notation_of_35_billion_l1971_197165


namespace two_numbers_problem_l1971_197182

theorem two_numbers_problem (a b : ℝ) : 
  a + b = 90 ∧ 
  0.4 * a = 0.3 * b + 15 → 
  a = 60 ∧ b = 30 := by
sorry

end two_numbers_problem_l1971_197182


namespace volume_inscribed_sphere_l1971_197163

/-- The volume of a sphere inscribed in a cube -/
theorem volume_inscribed_sphere (cube_volume : ℝ) (sphere_volume : ℝ) : 
  cube_volume = 343 →
  sphere_volume = (343 * Real.pi) / 6 :=
by sorry

end volume_inscribed_sphere_l1971_197163


namespace incorrect_solution_set_proof_l1971_197173

def Equation := ℝ → Prop

def SolutionSet (eq : Equation) := {x : ℝ | eq x}

theorem incorrect_solution_set_proof (eq : Equation) (S : Set ℝ) :
  (∀ x, ¬(eq x) → x ∉ S) ∧ (∀ x ∈ S, eq x) → S = SolutionSet eq → False :=
sorry

end incorrect_solution_set_proof_l1971_197173


namespace gcd_15_70_l1971_197121

theorem gcd_15_70 : Nat.gcd 15 70 = 5 := by
  sorry

end gcd_15_70_l1971_197121


namespace spring_length_at_6kg_l1971_197168

/-- Represents the relationship between weight and spring length -/
def spring_length (initial_length : ℝ) (stretch_rate : ℝ) (weight : ℝ) : ℝ :=
  initial_length + stretch_rate * weight

/-- Theorem stating that a spring with initial length 8 cm and stretch rate 0.5 cm/kg 
    will have a length of 11 cm when a 6 kg weight is hung -/
theorem spring_length_at_6kg 
  (initial_length : ℝ) (stretch_rate : ℝ) (weight : ℝ)
  (h1 : initial_length = 8)
  (h2 : stretch_rate = 0.5)
  (h3 : weight = 6) :
  spring_length initial_length stretch_rate weight = 11 := by
  sorry

end spring_length_at_6kg_l1971_197168


namespace cricket_team_average_age_l1971_197174

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ),
    team_size = 11 →
    captain_age = 25 →
    wicket_keeper_age_diff = 3 →
    remaining_players_age_diff = 1 →
    ∃ (team_average_age : ℚ),
      team_average_age = 22 ∧
      team_average_age * team_size =
        captain_age + (captain_age + wicket_keeper_age_diff) +
        (team_size - 2) * (team_average_age - remaining_players_age_diff) :=
by
  sorry

end cricket_team_average_age_l1971_197174


namespace ellipse_a_plus_k_l1971_197115

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (-1, -1)
  f2 : ℝ × ℝ := (-1, -3)
  -- Point on the ellipse
  p : ℝ × ℝ := (4, -2)
  -- Constants in the ellipse equation
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- a and b are positive
  a_pos : a > 0
  b_pos : b > 0
  -- The point p satisfies the ellipse equation
  eq_satisfied : (((p.1 - h)^2 / a^2) + ((p.2 - k)^2 / b^2)) = 1

/-- Theorem stating that a + k = 3 for the given ellipse -/
theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 3 := by
  sorry

end ellipse_a_plus_k_l1971_197115


namespace minnie_horses_per_day_l1971_197186

theorem minnie_horses_per_day (mickey_weekly : ℕ) (days_per_week : ℕ) 
  (h1 : mickey_weekly = 98)
  (h2 : days_per_week = 7) :
  ∃ (minnie_daily : ℕ),
    (2 * minnie_daily - 6) * days_per_week = mickey_weekly ∧
    minnie_daily > days_per_week ∧
    minnie_daily - days_per_week = 3 := by
  sorry

end minnie_horses_per_day_l1971_197186


namespace child_play_time_l1971_197130

theorem child_play_time (num_children : ℕ) (children_per_game : ℕ) (total_time : ℕ) 
  (h1 : num_children = 7)
  (h2 : children_per_game = 2)
  (h3 : total_time = 140)
  (h4 : children_per_game ≤ num_children)
  (h5 : children_per_game > 0)
  (h6 : total_time > 0) :
  (children_per_game * total_time) / num_children = 40 := by
sorry

end child_play_time_l1971_197130


namespace unfactorable_quartic_l1971_197150

theorem unfactorable_quartic : ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end unfactorable_quartic_l1971_197150


namespace cyclist_round_trip_l1971_197118

/-- Cyclist's round trip problem -/
theorem cyclist_round_trip
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (second_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_speed : ℝ)
  (total_round_trip_time : ℝ)
  (h1 : total_distance = first_leg_distance + second_leg_distance)
  (h2 : first_leg_distance = 18)
  (h3 : second_leg_distance = 12)
  (h4 : first_leg_speed = 9)
  (h5 : second_leg_speed = 10)
  (h6 : total_round_trip_time = 7.2)
  : (2 * total_distance) / (total_round_trip_time - (first_leg_distance / first_leg_speed + second_leg_distance / second_leg_speed)) = 7.5 := by
  sorry


end cyclist_round_trip_l1971_197118


namespace tangent_line_equations_l1971_197131

/-- The equation of a tangent line to y = x^3 passing through (1, 1) -/
def IsTangentLine (m b : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    (x₀^3 = m * x₀ + b) ∧  -- The line touches the curve at some point (x₀, x₀^3)
    (1 = m * 1 + b) ∧      -- The line passes through (1, 1)
    (m = 3 * x₀^2)         -- The slope of the line equals the derivative of x^3 at x₀

theorem tangent_line_equations :
  ∀ m b : ℝ, IsTangentLine m b ↔ (m = 3 ∧ b = -2) ∨ (m = 3/4 ∧ b = 1/4) :=
sorry

end tangent_line_equations_l1971_197131


namespace polynomial_coefficient_equality_l1971_197100

theorem polynomial_coefficient_equality (k d m : ℚ) : 
  (∀ x : ℚ, (6 * x^3 - 4 * x^2 + 9/4) * (d * x^3 + k * x^2 + m) = 
   18 * x^6 - 17 * x^5 + 34 * x^4 - (36/4) * x^3 + (18/4) * x^2) → 
  k = -5/6 := by
  sorry

end polynomial_coefficient_equality_l1971_197100


namespace mother_bought_pencils_l1971_197112

def dozen : ℕ := 12

def initial_pencils : ℕ := 17

def total_pencils : ℕ := 2 * dozen

theorem mother_bought_pencils : total_pencils - initial_pencils = 7 := by
  sorry

end mother_bought_pencils_l1971_197112


namespace equilateral_triangle_roots_l1971_197166

theorem equilateral_triangle_roots (a b z₁ z₂ : ℂ) : 
  (z₁^2 + a*z₁ + b = 0) → 
  (z₂^2 + a*z₂ + b = 0) → 
  (∃ ω : ℂ, ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁) →
  a^2 / b = 3 := by
sorry

end equilateral_triangle_roots_l1971_197166


namespace factorial_fraction_equals_one_l1971_197195

theorem factorial_fraction_equals_one :
  (3 * Nat.factorial 5 + 15 * Nat.factorial 4) / Nat.factorial 6 = 1 := by
  sorry

end factorial_fraction_equals_one_l1971_197195


namespace city_d_sand_amount_l1971_197155

/-- The amount of sand received by each city and the total amount --/
structure SandDistribution where
  cityA : Rat
  cityB : Rat
  cityC : Rat
  total : Rat

/-- The amount of sand received by City D --/
def sandCityD (sd : SandDistribution) : Rat :=
  sd.total - (sd.cityA + sd.cityB + sd.cityC)

/-- Theorem stating that City D received 28 tons of sand --/
theorem city_d_sand_amount :
  let sd : SandDistribution := {
    cityA := 33/2,
    cityB := 26,
    cityC := 49/2,
    total := 95
  }
  sandCityD sd = 28 := by sorry

end city_d_sand_amount_l1971_197155


namespace fraction_to_decimal_l1971_197176

theorem fraction_to_decimal : (17 : ℚ) / 200 = (34 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l1971_197176


namespace walters_chores_l1971_197133

theorem walters_chores (normal_pay exceptional_pay total_days total_earnings : ℕ) 
  (h1 : normal_pay = 3)
  (h2 : exceptional_pay = 6)
  (h3 : total_days = 10)
  (h4 : total_earnings = 42) :
  ∃ (normal_days exceptional_days : ℕ),
    normal_days + exceptional_days = total_days ∧
    normal_days * normal_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 4 := by
  sorry

end walters_chores_l1971_197133


namespace kangaroo_exhibition_arrangements_l1971_197134

/-- The number of ways to arrange n uniquely tall kangaroos in a row -/
def kangaroo_arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n uniquely tall kangaroos in a row,
    with the two tallest at the ends -/
def kangaroo_arrangements_with_tallest_at_ends (n : ℕ) : ℕ :=
  2 * kangaroo_arrangements (n - 2)

theorem kangaroo_exhibition_arrangements :
  kangaroo_arrangements_with_tallest_at_ends 8 = 1440 := by
  sorry

end kangaroo_exhibition_arrangements_l1971_197134


namespace candy_chocolate_price_difference_l1971_197114

def candy_bar_original_price : ℝ := 6
def candy_bar_discount : ℝ := 0.25
def chocolate_original_price : ℝ := 3
def chocolate_discount : ℝ := 0.10

def discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

theorem candy_chocolate_price_difference :
  discounted_price candy_bar_original_price candy_bar_discount -
  discounted_price chocolate_original_price chocolate_discount = 1.80 := by
sorry

end candy_chocolate_price_difference_l1971_197114


namespace absolute_value_reciprocal_intersection_l1971_197142

/-- The equation |x + a| = 1/x has exactly two solutions if and only if a = -2 -/
theorem absolute_value_reciprocal_intersection (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |x₁ + a| = 1/x₁ ∧ |x₂ + a| = 1/x₂) ↔ a = -2 :=
by sorry

end absolute_value_reciprocal_intersection_l1971_197142


namespace xyz_product_l1971_197183

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := by
sorry

end xyz_product_l1971_197183


namespace accurate_to_thousands_l1971_197143

/-- Represents a large number in millions with one decimal place -/
structure LargeNumber where
  whole : ℕ
  decimal : ℕ
  inv_ten : decimal < 10

/-- Converts a LargeNumber to its full integer representation -/
def LargeNumber.toInt (n : LargeNumber) : ℕ := n.whole * 1000000 + n.decimal * 100000

/-- Represents the place value in a number system -/
inductive PlaceValue
  | Thousands
  | Hundreds
  | Tens
  | Ones
  | Tenths
  | Hundredths

/-- Determines the smallest accurately represented place value for a given LargeNumber -/
def smallestAccuratePlaceValue (n : LargeNumber) : PlaceValue := 
  if n.decimal % 10 = 0 then PlaceValue.Hundreds else PlaceValue.Thousands

theorem accurate_to_thousands (n : LargeNumber) 
  (h : n.whole = 42 ∧ n.decimal = 3) : 
  smallestAccuratePlaceValue n = PlaceValue.Thousands := by
  sorry

end accurate_to_thousands_l1971_197143


namespace triangle_angle_A_l1971_197188

/-- Given a triangle ABC with angle B = 45°, side c = 2√2, and side b = 4√3/3,
    prove that angle A is either 7π/12 or π/12 -/
theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) : 
  B = π/4 → c = 2 * Real.sqrt 2 → b = 4 * Real.sqrt 3 / 3 →
  A = 7*π/12 ∨ A = π/12 :=
by sorry

end triangle_angle_A_l1971_197188


namespace average_weight_b_c_l1971_197139

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 45 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- The average weight of a, b, and c is 45 kg
  (a + b) / 2 = 40 →       -- The average weight of a and b is 40 kg
  b = 35 →                 -- The weight of b is 35 kg
  (b + c) / 2 = 45 :=      -- The average weight of b and c is 45 kg
by sorry

end average_weight_b_c_l1971_197139


namespace zero_subset_M_l1971_197177

-- Define the set M
def M : Set ℝ := {x | x > -2}

-- State the theorem
theorem zero_subset_M : {0} ⊆ M := by
  sorry

end zero_subset_M_l1971_197177


namespace not_both_perfect_squares_l1971_197153

theorem not_both_perfect_squares (x y : ℕ+) (h1 : Nat.gcd x.val y.val = 1) 
  (h2 : ∃ k : ℕ, x.val + 3 * y.val^2 = k^2) : 
  ¬ ∃ z : ℕ, x.val^2 + 9 * y.val^4 = z^2 := by
  sorry

end not_both_perfect_squares_l1971_197153


namespace quadratic_equation_solution_l1971_197197

theorem quadratic_equation_solution :
  ∀ x : ℝ, x > 0 → (7 * x^2 - 8 * x - 6 = 0) → (x = 6/7 ∨ x = 1) :=
by sorry

end quadratic_equation_solution_l1971_197197


namespace age_ratio_sachin_rahul_l1971_197152

/-- Given that Sachin is 5 years old and 7 years younger than Rahul, 
    prove that the ratio of Sachin's age to Rahul's age is 5:12. -/
theorem age_ratio_sachin_rahul :
  let sachin_age : ℕ := 5
  let age_difference : ℕ := 7
  let rahul_age : ℕ := sachin_age + age_difference
  (sachin_age : ℚ) / (rahul_age : ℚ) = 5 / 12 := by
  sorry

end age_ratio_sachin_rahul_l1971_197152


namespace range_of_a_l1971_197146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) :=
sorry

end range_of_a_l1971_197146


namespace other_sales_is_fifteen_percent_l1971_197105

/-- The percentage of sales not attributed to books, magazines, or stationery -/
def other_sales_percentage (books magazines stationery : ℝ) : ℝ :=
  100 - (books + magazines + stationery)

/-- Theorem stating that the percentage of other sales is 15% -/
theorem other_sales_is_fifteen_percent :
  other_sales_percentage 45 30 10 = 15 := by
  sorry

#eval other_sales_percentage 45 30 10

end other_sales_is_fifteen_percent_l1971_197105


namespace sufficient_but_not_necessary_l1971_197180

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  ¬(∀ x : ℝ, x > 1 → x > 3) := by
sorry

end sufficient_but_not_necessary_l1971_197180


namespace regression_change_l1971_197172

/-- Represents a linear regression equation of the form ŷ = a + bx̂ -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the change in ŷ when x̂ increases by 1 unit -/
def change_in_y (eq : LinearRegression) : ℝ := -eq.b

/-- Theorem: For the regression equation ŷ = 2 - 3x̂, 
    when x̂ increases by 1 unit, ŷ decreases by 3 units -/
theorem regression_change : 
  let eq := LinearRegression.mk 2 (-3)
  change_in_y eq = -3 := by sorry

end regression_change_l1971_197172


namespace ribbon_length_difference_l1971_197189

/-- Proves that the difference in ribbon length between two wrapping methods
    for a box matches one side of the box. -/
theorem ribbon_length_difference (l w h bow : ℕ) 
  (hl : l = 22) (hw : w = 22) (hh : h = 11) (hbow : bow = 24) :
  (2 * l + 4 * w + 2 * h + bow) - (2 * l + 2 * w + 4 * h + bow) = l := by
  sorry

end ribbon_length_difference_l1971_197189


namespace fraction_sum_zero_l1971_197156

theorem fraction_sum_zero : 
  (1 / 12 : ℚ) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + 
  (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 := by
  sorry

end fraction_sum_zero_l1971_197156


namespace prob_fourth_six_after_three_ones_l1971_197145

/-- Represents a six-sided die --/
inductive Die
| Fair
| Biased

/-- Probability of rolling a specific number on a given die --/
def prob_roll (d : Die) (n : Nat) : ℚ :=
  match d, n with
  | Die.Fair, _ => 1/6
  | Die.Biased, 1 => 1/3
  | Die.Biased, 6 => 1/3
  | Die.Biased, _ => 1/15

/-- Probability of rolling three ones in a row on a given die --/
def prob_three_ones (d : Die) : ℚ :=
  (prob_roll d 1) ^ 3

/-- Prior probability of choosing each die --/
def prior_prob : ℚ := 1/2

/-- Theorem stating the probability of rolling a six on the fourth roll
    after observing three ones --/
theorem prob_fourth_six_after_three_ones :
  let posterior_fair := (prior_prob * prob_three_ones Die.Fair) /
    (prior_prob * prob_three_ones Die.Fair + prior_prob * prob_three_ones Die.Biased)
  let posterior_biased := (prior_prob * prob_three_ones Die.Biased) /
    (prior_prob * prob_three_ones Die.Fair + prior_prob * prob_three_ones Die.Biased)
  posterior_fair * (prob_roll Die.Fair 6) + posterior_biased * (prob_roll Die.Biased 6) = 17/54 := by
  sorry

/-- The sum of numerator and denominator in the final probability --/
def result : ℕ := 17 + 54

#eval result  -- Should output 71

end prob_fourth_six_after_three_ones_l1971_197145


namespace prob_heads_and_five_l1971_197192

/-- The probability of getting heads on a fair coin flip -/
def prob_heads : ℚ := 1 / 2

/-- The probability of rolling a 5 on a regular eight-sided die -/
def prob_five : ℚ := 1 / 8

/-- The events (coin flip and die roll) are independent -/
axiom events_independent : True

theorem prob_heads_and_five : prob_heads * prob_five = 1 / 16 := by
  sorry

end prob_heads_and_five_l1971_197192


namespace max_pogs_purchase_l1971_197120

theorem max_pogs_purchase (x y z : ℕ) : 
  x ≥ 1 → y ≥ 1 → z ≥ 1 →
  3 * x + 4 * y + 9 * z = 75 →
  z ≤ 7 :=
sorry

end max_pogs_purchase_l1971_197120


namespace multiply_5915581_7907_l1971_197141

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := by
  sorry

end multiply_5915581_7907_l1971_197141


namespace committee_selection_l1971_197108

theorem committee_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 5) : 
  Nat.choose n m = 252 := by
  sorry

end committee_selection_l1971_197108


namespace babysitting_earnings_l1971_197104

/-- Calculates the earnings for a given hourly rate and number of minutes worked. -/
def calculate_earnings (hourly_rate : ℚ) (minutes_worked : ℚ) : ℚ :=
  hourly_rate * minutes_worked / 60

/-- Proves that given an hourly rate of $12 and 50 minutes of work, the earnings are equal to $10. -/
theorem babysitting_earnings :
  calculate_earnings 12 50 = 10 := by
  sorry

end babysitting_earnings_l1971_197104


namespace am_gm_squared_max_value_on_interval_max_value_sqrt_function_l1971_197116

-- Statement 1
theorem am_gm_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a * b ≤ ((a + b) / 2) ^ 2 := by sorry

-- Statement 2
theorem max_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) :
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c := by sorry

theorem max_value_sqrt_function :
  ∃ c ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, x * Real.sqrt (4 - x^2) ≤ 2 := by sorry

end am_gm_squared_max_value_on_interval_max_value_sqrt_function_l1971_197116
