import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2170_217070

def U : Set ℝ := {x | Real.exp x > 1}
def A : Set ℝ := {x | x > 1}

theorem complement_of_A_in_U : Set.compl A ∩ U = Set.Ioo 0 1 ∪ Set.singleton 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2170_217070


namespace NUMINAMATH_CALUDE_marcus_initial_mileage_l2170_217025

/-- Represents the mileage and fuel efficiency of a car --/
structure Car where
  mpg : ℕ  -- Miles per gallon
  tankCapacity : ℕ  -- Gallons
  currentMileage : ℕ  -- Current mileage

/-- Calculates the initial mileage of a car before a road trip --/
def initialMileage (c : Car) (numFillUps : ℕ) : ℕ :=
  c.currentMileage - (c.mpg * c.tankCapacity * numFillUps)

/-- Theorem: Given the conditions of Marcus's road trip, his car's initial mileage was 1728 miles --/
theorem marcus_initial_mileage :
  let marcusCar : Car := { mpg := 30, tankCapacity := 20, currentMileage := 2928 }
  initialMileage marcusCar 2 = 1728 := by
  sorry

#eval initialMileage { mpg := 30, tankCapacity := 20, currentMileage := 2928 } 2

end NUMINAMATH_CALUDE_marcus_initial_mileage_l2170_217025


namespace NUMINAMATH_CALUDE_prob_king_then_ten_l2170_217017

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of 10s in a standard deck -/
def TensInDeck : ℕ := 4

/-- Probability of drawing a King first and then a 10 from a standard deck -/
theorem prob_king_then_ten (deck : ℕ) (kings : ℕ) (tens : ℕ) :
  deck = StandardDeck → kings = KingsInDeck → tens = TensInDeck →
  (kings : ℚ) / deck * tens / (deck - 1) = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_then_ten_l2170_217017


namespace NUMINAMATH_CALUDE_taxi_problem_l2170_217081

def taxi_distances : List Int := [9, -3, -5, 4, 8, 6, 3, -6, -4, 10]
def price_per_km : ℝ := 2.4

theorem taxi_problem (distances : List Int) (price : ℝ) 
  (h_distances : distances = taxi_distances) (h_price : price = price_per_km) :
  (distances.sum = 22) ∧ 
  ((distances.map Int.natAbs).sum * price = 139.2) := by
  sorry

end NUMINAMATH_CALUDE_taxi_problem_l2170_217081


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2170_217014

theorem smallest_integer_quadratic_inequality :
  ∃ (n : ℤ), (∀ (m : ℤ), m^2 - 9*m + 18 ≥ 0 → n ≤ m) ∧ (n^2 - 9*n + 18 ≥ 0) ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2170_217014


namespace NUMINAMATH_CALUDE_exists_sequence_mod_23_l2170_217052

/-- Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence with the desired property -/
theorem exists_sequence_mod_23 : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ 
  F 1 = 1 ∧ 
  (∀ n : ℕ, F (n + 2) = 3 * F (n + 1) - F n) ∧
  F 12 ≡ 0 [ZMOD 23] := by
  sorry


end NUMINAMATH_CALUDE_exists_sequence_mod_23_l2170_217052


namespace NUMINAMATH_CALUDE_farm_animal_count_l2170_217076

theorem farm_animal_count :
  ∀ (cows chickens ducks : ℕ),
    (4 * cows + 2 * chickens + 2 * ducks = 20 + 2 * (cows + chickens + ducks)) →
    (chickens + ducks = 2 * cows) →
    cows = 10 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_count_l2170_217076


namespace NUMINAMATH_CALUDE_rectangle_count_6x5_grid_l2170_217044

/-- Represents a grid of lines in a coordinate plane -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Represents a point in a 2D coordinate plane -/
structure Point :=
  (x : ℕ)
  (y : ℕ)

/-- Counts the number of ways to form a rectangle enclosing a given point -/
def count_rectangles (g : Grid) (p : Point) : ℕ :=
  sorry

/-- Theorem stating the number of ways to form a rectangle enclosing (3, 4) in a 6x5 grid -/
theorem rectangle_count_6x5_grid :
  let g : Grid := ⟨6, 5⟩
  let p : Point := ⟨3, 4⟩
  count_rectangles g p = 24 :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_6x5_grid_l2170_217044


namespace NUMINAMATH_CALUDE_jeremy_watermelon_weeks_l2170_217031

/-- The number of weeks watermelons will last for Jeremy -/
def watermelon_weeks (total : ℕ) (eaten_per_week : ℕ) (given_to_dad : ℕ) : ℕ :=
  total / (eaten_per_week + given_to_dad)

/-- Theorem: Given Jeremy's watermelon consumption pattern, the watermelons will last 6 weeks -/
theorem jeremy_watermelon_weeks :
  watermelon_weeks 30 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_watermelon_weeks_l2170_217031


namespace NUMINAMATH_CALUDE_product_in_base_10_l2170_217035

-- Define the binary number 11001₂
def binary_num : ℕ := 25

-- Define the ternary number 112₃
def ternary_num : ℕ := 14

-- Theorem to prove
theorem product_in_base_10 : binary_num * ternary_num = 350 := by
  sorry

end NUMINAMATH_CALUDE_product_in_base_10_l2170_217035


namespace NUMINAMATH_CALUDE_border_collie_grooming_time_l2170_217095

/-- Represents the time in minutes Karen takes to groom different dog breeds -/
structure GroomingTimes where
  rottweiler : ℕ
  borderCollie : ℕ
  chihuahua : ℕ

/-- Represents the number of dogs Karen grooms in a specific session -/
structure DogCounts where
  rottweilers : ℕ
  borderCollies : ℕ
  chihuahuas : ℕ

/-- Given Karen's grooming times and dog counts, calculates the total grooming time -/
def totalGroomingTime (times : GroomingTimes) (counts : DogCounts) : ℕ :=
  times.rottweiler * counts.rottweilers +
  times.borderCollie * counts.borderCollies +
  times.chihuahua * counts.chihuahuas

/-- Theorem stating that Karen takes 10 minutes to groom a border collie -/
theorem border_collie_grooming_time :
  ∀ (times : GroomingTimes) (counts : DogCounts),
    times.rottweiler = 20 →
    times.chihuahua = 45 →
    counts.rottweilers = 6 →
    counts.borderCollies = 9 →
    counts.chihuahuas = 1 →
    totalGroomingTime times counts = 255 →
    times.borderCollie = 10 := by
  sorry

end NUMINAMATH_CALUDE_border_collie_grooming_time_l2170_217095


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2170_217029

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) : (x + 3) * (x - 2) + x * (4 - x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2170_217029


namespace NUMINAMATH_CALUDE_person_a_parts_l2170_217064

/-- Represents the number of parts made by each person -/
structure PartProduction where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the production scenario described in the problem -/
def production_scenario (p : PartProduction) : Prop :=
  p.c = 20 ∧
  4 * p.b = 3 * p.c ∧
  10 * p.a = 3 * (p.a + p.b + p.c)

theorem person_a_parts :
  ∀ p : PartProduction, production_scenario p → p.a = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_person_a_parts_l2170_217064


namespace NUMINAMATH_CALUDE_f_zero_one_eq_neg_one_one_l2170_217030

/-- The type of points in the real plane -/
def RealPair := ℝ × ℝ

/-- The mapping f: A → B -/
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that f(0, 1) = (-1, 1) -/
theorem f_zero_one_eq_neg_one_one :
  f (0, 1) = (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_f_zero_one_eq_neg_one_one_l2170_217030


namespace NUMINAMATH_CALUDE_prize_distribution_l2170_217028

theorem prize_distribution (total_winners : ℕ) (min_award : ℚ) (max_award : ℚ) :
  total_winners = 15 →
  min_award = 15 →
  max_award = 285 →
  ∃ (total_prize : ℚ),
    (2 / 5 : ℚ) * total_prize = max_award * ((3 / 5 : ℚ) * total_winners) ∧
    total_prize = 6502.5 :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l2170_217028


namespace NUMINAMATH_CALUDE_unique_power_of_two_plus_one_l2170_217034

theorem unique_power_of_two_plus_one : 
  ∃! (n : ℕ), ∃ (A p : ℕ), p > 1 ∧ 2^n + 1 = A^p :=
by
  sorry

end NUMINAMATH_CALUDE_unique_power_of_two_plus_one_l2170_217034


namespace NUMINAMATH_CALUDE_correct_operation_l2170_217032

theorem correct_operation (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2170_217032


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2170_217085

/-- Represents the staff categories in the unit -/
inductive StaffCategory
  | Business
  | Management
  | Logistics

/-- Represents the staff distribution in the unit -/
structure StaffDistribution where
  total : ℕ
  business : ℕ
  management : ℕ
  logistics : ℕ
  sum_eq_total : business + management + logistics = total

/-- Represents the sample size and distribution -/
structure Sample where
  size : ℕ
  business : ℕ
  management : ℕ
  logistics : ℕ
  sum_eq_size : business + management + logistics = size

/-- Checks if a sample is proportionally correct for a given staff distribution -/
def is_proportional_sample (staff : StaffDistribution) (sample : Sample) : Prop :=
  staff.business * sample.size = sample.business * staff.total ∧
  staff.management * sample.size = sample.management * staff.total ∧
  staff.logistics * sample.size = sample.logistics * staff.total

/-- Theorem: The given sample is proportionally correct for the given staff distribution -/
theorem correct_stratified_sample :
  let staff : StaffDistribution := ⟨160, 112, 16, 32, rfl⟩
  let sample : Sample := ⟨20, 14, 2, 4, rfl⟩
  is_proportional_sample staff sample := by sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l2170_217085


namespace NUMINAMATH_CALUDE_expression_value_l2170_217084

theorem expression_value : 
  let x : ℝ := 4
  let y : ℝ := -3
  let z : ℝ := 5
  x^2 + y^2 - z^2 + 2*y*z = -30 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2170_217084


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2170_217096

/-- The height of a tree that a monkey can climb in 15 hours, 
    given that it hops 3 ft up and slips 2 ft back each hour except for the last hour. -/
def tree_height : ℕ :=
  let hop_distance : ℕ := 3
  let slip_distance : ℕ := 2
  let total_hours : ℕ := 15
  let net_progress_per_hour : ℕ := hop_distance - slip_distance
  let height_before_last_hour : ℕ := net_progress_per_hour * (total_hours - 1)
  height_before_last_hour + hop_distance

theorem monkey_climb_theorem : tree_height = 17 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2170_217096


namespace NUMINAMATH_CALUDE_string_average_length_l2170_217088

theorem string_average_length (s1 s2 s3 : ℝ) 
  (h1 : s1 = 2) (h2 : s2 = 3) (h3 : s3 = 7) : 
  (s1 + s2 + s3) / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_string_average_length_l2170_217088


namespace NUMINAMATH_CALUDE_composite_29n_plus_11_l2170_217098

theorem composite_29n_plus_11 (n : ℕ) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a ^ 2) 
  (h2 : ∃ b : ℕ, 10 * n + 1 = b ^ 2) : 
  ¬(Nat.Prime (29 * n + 11)) :=
sorry

end NUMINAMATH_CALUDE_composite_29n_plus_11_l2170_217098


namespace NUMINAMATH_CALUDE_total_notes_count_l2170_217080

/-- Proves that the total number of notes is 126 given the conditions -/
theorem total_notes_count (total_amount : ℕ) (note_50_count : ℕ) (note_50_value : ℕ) (note_500_value : ℕ) :
  total_amount = 10350 ∧
  note_50_count = 117 ∧
  note_50_value = 50 ∧
  note_500_value = 500 ∧
  total_amount = note_50_count * note_50_value + (total_amount - note_50_count * note_50_value) / note_500_value * note_500_value →
  note_50_count + (total_amount - note_50_count * note_50_value) / note_500_value = 126 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l2170_217080


namespace NUMINAMATH_CALUDE_exists_arithmetic_progression_of_5_primes_exists_arithmetic_progression_of_6_primes_l2170_217090

/-- An arithmetic progression of primes -/
def ArithmeticProgressionOfPrimes (n : ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ k : Fin n, Prime (a + k * d)

/-- There exists an arithmetic progression of 5 primes -/
theorem exists_arithmetic_progression_of_5_primes :
  ArithmeticProgressionOfPrimes 5 := by
  sorry

/-- There exists an arithmetic progression of 6 primes -/
theorem exists_arithmetic_progression_of_6_primes :
  ArithmeticProgressionOfPrimes 6 := by
  sorry

end NUMINAMATH_CALUDE_exists_arithmetic_progression_of_5_primes_exists_arithmetic_progression_of_6_primes_l2170_217090


namespace NUMINAMATH_CALUDE_oliver_water_usage_l2170_217040

/-- Calculates the weekly water usage for Oliver's baths given the specified conditions. -/
def weekly_water_usage (bucket_capacity : ℕ) (fill_count : ℕ) (remove_count : ℕ) (days_per_week : ℕ) : ℕ :=
  (fill_count * bucket_capacity - remove_count * bucket_capacity) * days_per_week

/-- Theorem stating that Oliver's weekly water usage is 9240 ounces under the given conditions. -/
theorem oliver_water_usage :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

#eval weekly_water_usage 120 14 3 7

end NUMINAMATH_CALUDE_oliver_water_usage_l2170_217040


namespace NUMINAMATH_CALUDE_unique_root_of_abs_equation_l2170_217067

/-- The equation x|x| - 3|x| - 4 = 0 has exactly one real root -/
theorem unique_root_of_abs_equation : ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_abs_equation_l2170_217067


namespace NUMINAMATH_CALUDE_average_of_multiples_of_seven_l2170_217063

def is_between (a b x : ℝ) : Prop := a < x ∧ x < b

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

theorem average_of_multiples_of_seven (numbers : List ℕ) : 
  (∀ n ∈ numbers, is_between 6 36 n ∧ divisible_by n 7) →
  numbers.length > 0 →
  (numbers.sum / numbers.length : ℝ) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_seven_l2170_217063


namespace NUMINAMATH_CALUDE_bridge_length_l2170_217037

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 156 →
  train_speed_kmh = 45 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 344 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2170_217037


namespace NUMINAMATH_CALUDE_log_sum_equality_l2170_217062

theorem log_sum_equality : 2 * Real.log 25 / Real.log 5 + 3 * Real.log 64 / Real.log 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2170_217062


namespace NUMINAMATH_CALUDE_f_is_h_function_l2170_217006

def is_h_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁)

def f (x : ℝ) : ℝ := x * |x|

theorem f_is_h_function : is_h_function f := by sorry

end NUMINAMATH_CALUDE_f_is_h_function_l2170_217006


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2170_217060

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (2, m)

theorem perpendicular_vectors_magnitude (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) →
  Real.sqrt ((b m).1^2 + (b m).2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2170_217060


namespace NUMINAMATH_CALUDE_wanda_eating_theorem_l2170_217091

/-- Pascal's triangle up to row n -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Check if a number is odd -/
def isOdd (n : ℕ) : Bool :=
  sorry

/-- Count odd numbers in Pascal's triangle up to row n -/
def countOddNumbers (triangle : List (List ℕ)) : ℕ :=
  sorry

/-- Check if a path in Pascal's triangle satisfies the no-sum condition -/
def validPath (path : List ℕ) : Bool :=
  sorry

/-- Main theorem -/
theorem wanda_eating_theorem :
  ∃ (path : List ℕ), 
    (path.length > 100000) ∧ 
    (∀ n ∈ path, n ∈ (PascalTriangle 2011).join) ∧
    (∀ n ∈ path, isOdd n) ∧
    validPath path :=
  sorry

end NUMINAMATH_CALUDE_wanda_eating_theorem_l2170_217091


namespace NUMINAMATH_CALUDE_turquoise_color_perception_l2170_217013

theorem turquoise_color_perception (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  more_blue = 90 →
  both = 40 →
  neither = 20 →
  ∃ (more_green : ℕ), more_green = 80 ∧ 
    more_green + more_blue - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_turquoise_color_perception_l2170_217013


namespace NUMINAMATH_CALUDE_metal_bar_weight_l2170_217038

/-- The weight of Harry's custom creation at the gym -/
def total_weight : ℕ := 25

/-- The weight of each blue weight -/
def blue_weight : ℕ := 2

/-- The weight of each green weight -/
def green_weight : ℕ := 3

/-- The number of blue weights Harry put on the bar -/
def num_blue_weights : ℕ := 4

/-- The number of green weights Harry put on the bar -/
def num_green_weights : ℕ := 5

/-- The weight of the metal bar -/
def bar_weight : ℕ := total_weight - (num_blue_weights * blue_weight + num_green_weights * green_weight)

theorem metal_bar_weight : bar_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_metal_bar_weight_l2170_217038


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l2170_217033

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 + 9*x - 22 = 0 ∧ (∀ y : ℝ, y^2 + 9*y - 22 = 0 → x ≤ y) → x = -11 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l2170_217033


namespace NUMINAMATH_CALUDE_count_9_in_1_to_1000_l2170_217054

/-- Count of digit 9 in a specific place value for numbers from 1 to 1000 -/
def count_digit_9_in_place (place : Nat) : Nat :=
  1000 / (10 ^ place)

/-- Total count of digit 9 in all integers from 1 to 1000 -/
def total_count_9 : Nat :=
  count_digit_9_in_place 0 + count_digit_9_in_place 1 + count_digit_9_in_place 2

theorem count_9_in_1_to_1000 :
  total_count_9 = 300 := by
  sorry

end NUMINAMATH_CALUDE_count_9_in_1_to_1000_l2170_217054


namespace NUMINAMATH_CALUDE_common_ratio_is_four_l2170_217051

/-- Geometric sequence with sum S_n of first n terms -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

theorem common_ratio_is_four 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : geometric_sequence a S)
  (h1 : 3 * S 3 = a 4 - 2)
  (h2 : 3 * S 2 = a 3 - 2) :
  a 2 / a 1 = 4 := by sorry

end NUMINAMATH_CALUDE_common_ratio_is_four_l2170_217051


namespace NUMINAMATH_CALUDE_stamp_cost_difference_l2170_217079

/-- The cost of a single rooster stamp -/
def rooster_stamp_cost : ℚ := 1.5

/-- The cost of a single daffodil stamp -/
def daffodil_stamp_cost : ℚ := 0.75

/-- The number of rooster stamps purchased -/
def rooster_stamp_count : ℕ := 2

/-- The number of daffodil stamps purchased -/
def daffodil_stamp_count : ℕ := 5

/-- The theorem stating the cost difference between daffodil and rooster stamps -/
theorem stamp_cost_difference : 
  (daffodil_stamp_count : ℚ) * daffodil_stamp_cost - 
  (rooster_stamp_count : ℚ) * rooster_stamp_cost = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_difference_l2170_217079


namespace NUMINAMATH_CALUDE_product_simplification_l2170_217075

theorem product_simplification : 
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l2170_217075


namespace NUMINAMATH_CALUDE_unique_number_l2170_217005

theorem unique_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n + 3) % 3 = 0 ∧ 
  (n + 4) % 4 = 0 ∧ 
  (n + 5) % 5 = 0 ∧ 
  n = 60 := by sorry

end NUMINAMATH_CALUDE_unique_number_l2170_217005


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l2170_217056

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def fairCoinProbability (n k : ℕ) : ℚ :=
  (binomial n k : ℚ) * (1 / 2) ^ n

theorem fair_coin_probability_difference :
  let p3 := fairCoinProbability 5 3
  let p4 := fairCoinProbability 5 4
  abs (p3 - p4) = 5 / 32 := by sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l2170_217056


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2170_217099

theorem sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ (∃ a : ℝ, a ≤ 1 ∧ 1 / a < 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2170_217099


namespace NUMINAMATH_CALUDE_fourth_number_in_expression_l2170_217026

theorem fourth_number_in_expression (x : ℝ) : 
  0.3 * 0.8 + 0.1 * x = 0.29 → x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_expression_l2170_217026


namespace NUMINAMATH_CALUDE_always_two_real_roots_root_geq_7_implies_k_leq_neg_5_l2170_217077

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : ℝ := x^2 + (k+1)*x + 3*k - 6

-- Theorem 1: The quadratic equation always has two real roots
theorem always_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ k = 0 ∧ quadratic_equation x₂ k = 0 :=
sorry

-- Theorem 2: If one root is not less than 7, then k ≤ -5
theorem root_geq_7_implies_k_leq_neg_5 (k : ℝ) :
  (∃ x : ℝ, quadratic_equation x k = 0 ∧ x ≥ 7) → k ≤ -5 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_root_geq_7_implies_k_leq_neg_5_l2170_217077


namespace NUMINAMATH_CALUDE_second_interest_rate_is_20_percent_l2170_217068

/-- Given a total amount, an amount at 10% interest, and a total profit,
    calculate the second interest rate. -/
def calculate_second_interest_rate (total_amount : ℕ) (amount_at_10_percent : ℕ) (total_profit : ℕ) : ℚ :=
  let amount_at_second_rate := total_amount - amount_at_10_percent
  let interest_from_first_part := (10 : ℚ) / 100 * amount_at_10_percent
  let interest_from_second_part := total_profit - interest_from_first_part
  (interest_from_second_part * 100) / amount_at_second_rate

/-- Theorem stating that under the given conditions, the second interest rate is 20%. -/
theorem second_interest_rate_is_20_percent :
  calculate_second_interest_rate 80000 70000 9000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_interest_rate_is_20_percent_l2170_217068


namespace NUMINAMATH_CALUDE_fraction_simplification_l2170_217089

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (7 * a + 7 * b) / (a^2 - b^2) = 7 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2170_217089


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2170_217000

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2170_217000


namespace NUMINAMATH_CALUDE_tan_product_simplification_l2170_217043

theorem tan_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l2170_217043


namespace NUMINAMATH_CALUDE_min_lamps_l2170_217042

theorem min_lamps (n p : ℕ) (h1 : p > 0) : 
  (∃ (p : ℕ), p > 0 ∧ 
    (p + 10*n - 30) - p = 100 ∧ 
    (∀ m : ℕ, m < n → ¬(∃ (q : ℕ), q > 0 ∧ (q + 10*m - 30) - q = 100))) → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_min_lamps_l2170_217042


namespace NUMINAMATH_CALUDE_fraction_sum_greater_than_sum_fraction_l2170_217097

theorem fraction_sum_greater_than_sum_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b > 1 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_greater_than_sum_fraction_l2170_217097


namespace NUMINAMATH_CALUDE_x_twenty_percent_greater_than_98_l2170_217021

theorem x_twenty_percent_greater_than_98 (x : ℝ) :
  x = 98 * (1 + 20 / 100) → x = 117.6 := by
  sorry

end NUMINAMATH_CALUDE_x_twenty_percent_greater_than_98_l2170_217021


namespace NUMINAMATH_CALUDE_fraction_simplification_l2170_217073

theorem fraction_simplification (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2170_217073


namespace NUMINAMATH_CALUDE_jake_weight_loss_l2170_217053

theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
  (h1 : total_weight = 212)
  (h2 : jake_weight = 152)
  (h3 : total_weight = jake_weight + sister_weight) :
  jake_weight - (2 * sister_weight) = 32 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l2170_217053


namespace NUMINAMATH_CALUDE_cube_roots_l2170_217055

theorem cube_roots : (39 : ℕ)^3 = 59319 ∧ (47 : ℕ)^3 = 103823 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_l2170_217055


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2170_217020

theorem diophantine_equation_solution :
  ∀ x y : ℕ+, x^4 = y^2 + 71 ↔ x = 6 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2170_217020


namespace NUMINAMATH_CALUDE_equation_roots_l2170_217011

theorem equation_roots : 
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2170_217011


namespace NUMINAMATH_CALUDE_puzzle_missing_pieces_l2170_217036

/-- Calculates the number of missing puzzle pieces. -/
def missing_pieces (total : ℕ) (border : ℕ) (trevor : ℕ) (joe_multiplier : ℕ) : ℕ :=
  total - (border + trevor + joe_multiplier * trevor)

/-- Proves that the number of missing puzzle pieces is 5. -/
theorem puzzle_missing_pieces :
  missing_pieces 500 75 105 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_missing_pieces_l2170_217036


namespace NUMINAMATH_CALUDE_inverse_composition_equals_negative_one_l2170_217047

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 5

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ := (y - 5) / 4

-- Theorem statement
theorem inverse_composition_equals_negative_one :
  f_inv (f_inv 9) = -1 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_negative_one_l2170_217047


namespace NUMINAMATH_CALUDE_gcf_of_180_270_450_l2170_217069

theorem gcf_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by sorry

end NUMINAMATH_CALUDE_gcf_of_180_270_450_l2170_217069


namespace NUMINAMATH_CALUDE_division_scaling_certain_number_proof_l2170_217086

theorem division_scaling (a b c : ℝ) (h : a / b = c) : (100 * a) / (100 * b) = c := by
  sorry

theorem certain_number_proof :
  29.94 / 1.45 = 17.7 → 2994 / 14.5 = 17.7 := by
  sorry

end NUMINAMATH_CALUDE_division_scaling_certain_number_proof_l2170_217086


namespace NUMINAMATH_CALUDE_video_game_cost_l2170_217024

def allowance_period1 : ℕ := 8
def allowance_rate1 : ℕ := 5
def allowance_period2 : ℕ := 6
def allowance_rate2 : ℕ := 6
def remaining_money : ℕ := 3

def total_savings : ℕ := allowance_period1 * allowance_rate1 + allowance_period2 * allowance_rate2

def money_after_clothes : ℕ := total_savings / 2

theorem video_game_cost : money_after_clothes - remaining_money = 35 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_l2170_217024


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l2170_217041

-- Define the steps of linear regression analysis
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | DrawScatterPlot

-- Define a sequence of steps
def StepSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : StepSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

-- Define a proposition that x and y are linearly related
def linearlyRelated (x y : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ t : ℝ, y t = a * x t + b

-- Theorem stating that given linear relationship, the correct sequence is as defined
theorem correct_regression_sequence (x y : ℝ → ℝ) :
  linearlyRelated x y →
  (∀ seq : StepSequence,
    seq = correctSequence ↔
    seq = [RegressionStep.CollectData,
           RegressionStep.DrawScatterPlot,
           RegressionStep.CalculateEquation,
           RegressionStep.InterpretEquation]) :=
by sorry


end NUMINAMATH_CALUDE_correct_regression_sequence_l2170_217041


namespace NUMINAMATH_CALUDE_digits_of_3_pow_24_times_7_pow_36_l2170_217001

theorem digits_of_3_pow_24_times_7_pow_36 : ∃ n : ℕ, 
  n > 0 ∧ n < 10^32 ∧ 10^31 ≤ 3^24 * 7^36 ∧ 3^24 * 7^36 < 10^32 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_3_pow_24_times_7_pow_36_l2170_217001


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_four_l2170_217057

-- Define the equation
def equation (x m : ℝ) : Prop := 2 / x = m / (2 * x + 1)

-- Theorem stating the condition for no solution
theorem no_solution_iff_m_eq_four :
  (∀ x : ℝ, ¬ equation x m) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_four_l2170_217057


namespace NUMINAMATH_CALUDE_ali_age_l2170_217023

/-- Given the ages of Ali, Yusaf, and Umar, prove Ali's age -/
theorem ali_age (ali yusaf umar : ℕ) 
  (h1 : ali = yusaf + 3)
  (h2 : umar = 2 * yusaf)
  (h3 : umar = 10) : 
  ali = 8 := by
  sorry

end NUMINAMATH_CALUDE_ali_age_l2170_217023


namespace NUMINAMATH_CALUDE_quadratic_function_range_difference_l2170_217004

-- Define the quadratic function
def f (x c : ℝ) : ℝ := -2 * x^2 + c

-- Define the theorem
theorem quadratic_function_range_difference (c m : ℝ) :
  (m + 2 ≤ 0) →
  (∃ (min : ℝ), ∀ (m' : ℝ), m' + 2 ≤ 0 → 
    (f (m' + 2) c - f m' c) ≥ min) ∧
  (¬∃ (max : ℝ), ∀ (m' : ℝ), m' + 2 ≤ 0 → 
    (f (m' + 2) c - f m' c) ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_difference_l2170_217004


namespace NUMINAMATH_CALUDE_range_of_z_l2170_217059

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_range_of_z_l2170_217059


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l2170_217010

theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 8) :
  let side := (4 * Real.sqrt 6) / 3
  let area := (1 / 2) * side * h
  area = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l2170_217010


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2170_217018

/-- Given a geometric sequence with first term √3 and second term 3√3, 
    the seventh term is 729√3 -/
theorem seventh_term_of_geometric_sequence 
  (a₁ : ℝ) 
  (a₂ : ℝ) 
  (h₁ : a₁ = Real.sqrt 3)
  (h₂ : a₂ = 3 * Real.sqrt 3) :
  (a₁ * (a₂ / a₁)^6 : ℝ) = 729 * Real.sqrt 3 := by
  sorry

#check seventh_term_of_geometric_sequence

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2170_217018


namespace NUMINAMATH_CALUDE_intersection_point_l2170_217058

/-- The linear function y = 2x + 4 -/
def f (x : ℝ) : ℝ := 2 * x + 4

/-- The y-axis is the vertical line with x-coordinate 0 -/
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- The graph of the linear function f -/
def graph_f : Set (ℝ × ℝ) := {p | p.2 = f p.1}

/-- The intersection point of the graph of f with the y-axis -/
def intersection : ℝ × ℝ := (0, f 0)

theorem intersection_point :
  intersection ∈ y_axis ∧ intersection ∈ graph_f ∧ intersection = (0, 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2170_217058


namespace NUMINAMATH_CALUDE_toms_profit_l2170_217087

def flour_needed : ℕ := 500
def flour_bag_size : ℕ := 50
def flour_bag_price : ℕ := 20
def salt_needed : ℕ := 10
def salt_price : ℚ := 1/5
def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 500

def total_cost : ℚ := 
  (flour_needed / flour_bag_size * flour_bag_price : ℚ) + 
  (salt_needed * salt_price) + 
  promotion_cost

def total_revenue : ℕ := ticket_price * tickets_sold

theorem toms_profit : 
  total_revenue - total_cost = 8798 := by sorry

end NUMINAMATH_CALUDE_toms_profit_l2170_217087


namespace NUMINAMATH_CALUDE_passing_marks_l2170_217015

/-- The number of marks for passing an exam, given conditions about failing and passing candidates. -/
theorem passing_marks (T : ℝ) (P : ℝ) : 
  (0.4 * T = P - 40) →  -- Condition 1
  (0.6 * T = P + 20) →  -- Condition 2
  P = 160 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_l2170_217015


namespace NUMINAMATH_CALUDE_melanie_plums_l2170_217007

/-- The number of plums picked by different people and in total -/
structure PlumPicking where
  dan : ℕ
  sally : ℕ
  total : ℕ

/-- The theorem stating how many plums Melanie picked -/
theorem melanie_plums (p : PlumPicking) (h1 : p.dan = 9) (h2 : p.sally = 3) (h3 : p.total = 16) :
  p.total - (p.dan + p.sally) = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_l2170_217007


namespace NUMINAMATH_CALUDE_profit_calculation_l2170_217071

theorem profit_calculation (cost_price : ℝ) (x : ℝ) : 
  (40 * cost_price = x * (cost_price * 1.25)) → x = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2170_217071


namespace NUMINAMATH_CALUDE_function_equation_solution_l2170_217009

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_equation_solution :
  (∀ x : ℝ, 2 * f (x - 1) - 3 * f (1 - x) = 5 * x) →
  (∀ x : ℝ, f x = x - 5) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2170_217009


namespace NUMINAMATH_CALUDE_optimal_well_position_l2170_217046

open Real

/-- Represents the positions of 6 houses along a road -/
structure HousePositions where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  x₅ : ℝ
  x₆ : ℝ
  h₁₂ : x₁ < x₂
  h₂₃ : x₂ < x₃
  h₃₄ : x₃ < x₄
  h₄₅ : x₄ < x₅
  h₅₆ : x₅ < x₆

/-- The sum of absolute distances from a point x to all house positions -/
def sumOfDistances (hp : HousePositions) (x : ℝ) : ℝ :=
  |x - hp.x₁| + |x - hp.x₂| + |x - hp.x₃| + |x - hp.x₄| + |x - hp.x₅| + |x - hp.x₆|

/-- The theorem stating that the optimal well position is the average of x₃ and x₄ -/
theorem optimal_well_position (hp : HousePositions) :
  ∃ (x : ℝ), ∀ (y : ℝ), sumOfDistances hp x ≤ sumOfDistances hp y ∧ x = (hp.x₃ + hp.x₄) / 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_well_position_l2170_217046


namespace NUMINAMATH_CALUDE_article_cost_price_l2170_217003

theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C ∧ 
  S - 1 = 1.045 * C → 
  C = 200 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l2170_217003


namespace NUMINAMATH_CALUDE_power_of_81_l2170_217082

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l2170_217082


namespace NUMINAMATH_CALUDE_same_type_square_roots_l2170_217045

theorem same_type_square_roots :
  ∃ (k₁ k₂ : ℝ) (x : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ Real.sqrt 12 = k₁ * x ∧ Real.sqrt (1/3) = k₂ * x :=
by sorry

end NUMINAMATH_CALUDE_same_type_square_roots_l2170_217045


namespace NUMINAMATH_CALUDE_first_sampling_immediate_l2170_217019

/-- Represents the stages of the yeast population experiment -/
inductive ExperimentStage
  | Inoculation
  | Sampling
  | Counting

/-- Represents the timing of the first sampling test -/
inductive SamplingTiming
  | Immediate
  | Delayed

/-- The correct procedure for the yeast population experiment -/
def correctYeastExperimentProcedure : ExperimentStage → SamplingTiming
  | ExperimentStage.Inoculation => SamplingTiming.Immediate
  | _ => SamplingTiming.Delayed

/-- Theorem stating that the first sampling test should be conducted immediately after inoculation -/
theorem first_sampling_immediate :
  correctYeastExperimentProcedure ExperimentStage.Inoculation = SamplingTiming.Immediate :=
by sorry

end NUMINAMATH_CALUDE_first_sampling_immediate_l2170_217019


namespace NUMINAMATH_CALUDE_symmetric_point_correct_specific_case_l2170_217074

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem symmetric_point_correct (p : ℝ × ℝ) : 
  symmetric_point p = (-p.1, -p.2) := by sorry

theorem specific_case : 
  symmetric_point (3, -1) = (-3, 1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_specific_case_l2170_217074


namespace NUMINAMATH_CALUDE_smallest_valid_assembly_is_four_l2170_217072

/-- Represents a modified cube with snaps and receptacles -/
structure ModifiedCube :=
  (snaps : Fin 2)
  (receptacles : Fin 4)

/-- Represents an assembly of modified cubes -/
structure CubeAssembly :=
  (cubes : List ModifiedCube)
  (all_snaps_covered : Bool)
  (only_receptacles_visible : Bool)

/-- Returns true if the assembly is valid according to the problem constraints -/
def is_valid_assembly (assembly : CubeAssembly) : Prop :=
  assembly.all_snaps_covered ∧ assembly.only_receptacles_visible

/-- The smallest number of cubes needed for a valid assembly -/
def smallest_valid_assembly : ℕ := 4

/-- Theorem stating that the smallest valid assembly consists of 4 cubes -/
theorem smallest_valid_assembly_is_four :
  ∀ (assembly : CubeAssembly),
    is_valid_assembly assembly →
    assembly.cubes.length ≥ smallest_valid_assembly :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_assembly_is_four_l2170_217072


namespace NUMINAMATH_CALUDE_car_distribution_l2170_217049

def total_production : ℕ := 5650000
def first_supplier : ℕ := 1000000
def second_supplier : ℕ := first_supplier + 500000
def third_supplier : ℕ := first_supplier + second_supplier

theorem car_distribution (fourth_supplier fifth_supplier : ℕ) : 
  fourth_supplier = fifth_supplier ∧
  first_supplier + second_supplier + third_supplier + fourth_supplier + fifth_supplier = total_production →
  fourth_supplier = 325000 := by
sorry

end NUMINAMATH_CALUDE_car_distribution_l2170_217049


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2170_217048

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2170_217048


namespace NUMINAMATH_CALUDE_proportion_third_term_l2170_217008

/-- Given a proportion 0.75 : 1.65 :: y : 11, prove that y = 5 -/
theorem proportion_third_term (y : ℝ) : 
  (0.75 : ℝ) / 1.65 = y / 11 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_term_l2170_217008


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2170_217039

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 + x - 2) (x + 2)
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2170_217039


namespace NUMINAMATH_CALUDE_ab_relation_to_a_over_b_l2170_217022

theorem ab_relation_to_a_over_b (a b : ℝ) (h : a * b ≠ 0) :
  ¬(∀ a b, a * b > 1 → a > 1 / b) ∧
  ¬(∀ a b, a > 1 / b → a * b > 1) := by
  sorry

end NUMINAMATH_CALUDE_ab_relation_to_a_over_b_l2170_217022


namespace NUMINAMATH_CALUDE_equation_solution_l2170_217012

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 15 / (x / 3) → x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2170_217012


namespace NUMINAMATH_CALUDE_product_seven_l2170_217066

theorem product_seven : ∃ (x y : ℤ), x * y = 7 :=
sorry

end NUMINAMATH_CALUDE_product_seven_l2170_217066


namespace NUMINAMATH_CALUDE_four_roots_implies_a_in_interval_l2170_217050

-- Define the polynomial
def P (x a : ℝ) : ℝ := x^4 + 8*x^3 + 18*x^2 + 8*x + a

-- Define the property of having four distinct real roots
def has_four_distinct_real_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (P x₁ a = 0 ∧ P x₂ a = 0 ∧ P x₃ a = 0 ∧ P x₄ a = 0)

-- Theorem statement
theorem four_roots_implies_a_in_interval :
  ∀ a : ℝ, has_four_distinct_real_roots a → a ∈ Set.Ioo (-8 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_four_roots_implies_a_in_interval_l2170_217050


namespace NUMINAMATH_CALUDE_is_integer_division_l2170_217027

theorem is_integer_division : ∃ k : ℤ, (19^92 - 91^29) / 90 = k := by
  sorry

end NUMINAMATH_CALUDE_is_integer_division_l2170_217027


namespace NUMINAMATH_CALUDE_greater_number_proof_l2170_217092

theorem greater_number_proof (a b : ℝ) (h_sum : a + b = 40) (h_diff : a - b = 2) (h_greater : a > b) : a = 21 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l2170_217092


namespace NUMINAMATH_CALUDE_salesman_pears_morning_sales_salesman_pears_morning_sales_proof_l2170_217002

/-- Proof that a salesman sold 120 kilograms of pears in the morning -/
theorem salesman_pears_morning_sales : ℝ → Prop :=
  fun morning_sales : ℝ =>
    let afternoon_sales := 240
    let total_sales := 360
    (afternoon_sales = 2 * morning_sales) ∧
    (total_sales = morning_sales + afternoon_sales) →
    morning_sales = 120

-- The proof is omitted
theorem salesman_pears_morning_sales_proof : salesman_pears_morning_sales 120 := by
  sorry

end NUMINAMATH_CALUDE_salesman_pears_morning_sales_salesman_pears_morning_sales_proof_l2170_217002


namespace NUMINAMATH_CALUDE_fourth_week_sugar_l2170_217061

def sugar_amount (week : ℕ) : ℚ :=
  24 / (2 ^ (week - 1))

theorem fourth_week_sugar : sugar_amount 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_sugar_l2170_217061


namespace NUMINAMATH_CALUDE_parabola_vertex_l2170_217083

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 8*y + 3*x + 7 = 0

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola_equation (x + t) y → t = 0

-- Theorem statement
theorem parabola_vertex :
  is_vertex 3 (-4) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2170_217083


namespace NUMINAMATH_CALUDE_smallest_value_of_quadratic_l2170_217016

theorem smallest_value_of_quadratic :
  (∀ x : ℝ, x^2 + 6*x + 9 ≥ 0) ∧ (∃ x : ℝ, x^2 + 6*x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_quadratic_l2170_217016


namespace NUMINAMATH_CALUDE_calories_per_candy_bar_l2170_217078

/-- Given that there are 15 calories in 5 candy bars, prove that there are 3 calories in one candy bar. -/
theorem calories_per_candy_bar :
  let total_calories : ℕ := 15
  let total_bars : ℕ := 5
  let calories_per_bar : ℚ := total_calories / total_bars
  calories_per_bar = 3 := by sorry

end NUMINAMATH_CALUDE_calories_per_candy_bar_l2170_217078


namespace NUMINAMATH_CALUDE_john_boxes_l2170_217065

theorem john_boxes (stan jules joseph : ℕ) (john : ℚ) : 
  stan = 100 →
  joseph = stan / 5 →
  jules = joseph + 5 →
  john = jules * (6/5) →
  john = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_john_boxes_l2170_217065


namespace NUMINAMATH_CALUDE_remainder_proof_l2170_217093

theorem remainder_proof : ∃ r : ℕ, r < 33 ∧ r < 8 ∧ 266 % 33 = r ∧ 266 % 8 = r :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2170_217093


namespace NUMINAMATH_CALUDE_firm_partners_count_l2170_217094

theorem firm_partners_count :
  ∀ (partners associates : ℕ),
  (partners : ℚ) / associates = 2 / 63 →
  partners / (associates + 50) = 1 / 34 →
  partners = 20 := by
sorry

end NUMINAMATH_CALUDE_firm_partners_count_l2170_217094
