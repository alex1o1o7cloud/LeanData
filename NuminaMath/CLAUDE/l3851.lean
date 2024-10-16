import Mathlib

namespace NUMINAMATH_CALUDE_drummer_trombone_difference_l3851_385198

/-- Represents the number of players for each instrument in the school band --/
structure BandComposition where
  flute : Nat
  trumpet : Nat
  trombone : Nat
  clarinet : Nat
  frenchHorn : Nat
  drummer : Nat

/-- Theorem stating the difference between drummers and trombone players --/
theorem drummer_trombone_difference (band : BandComposition) : 
  band.flute = 5 →
  band.trumpet = 3 * band.flute →
  band.trombone = band.trumpet - 8 →
  band.clarinet = 2 * band.flute →
  band.frenchHorn = band.trombone + 3 →
  band.flute + band.trumpet + band.trombone + band.clarinet + band.frenchHorn + band.drummer = 65 →
  band.drummer > band.trombone →
  band.drummer - band.trombone = 11 := by
sorry


end NUMINAMATH_CALUDE_drummer_trombone_difference_l3851_385198


namespace NUMINAMATH_CALUDE_series_calculation_l3851_385194

def series_sum (n : ℕ) : ℤ :=
  (n + 1) * 3

theorem series_calculation : series_sum 32 = 1584 := by
  sorry

#eval series_sum 32

end NUMINAMATH_CALUDE_series_calculation_l3851_385194


namespace NUMINAMATH_CALUDE_initial_books_count_l3851_385175

theorem initial_books_count (initial_books additional_books total_books : ℕ) 
  (h1 : additional_books = 23)
  (h2 : total_books = 77)
  (h3 : initial_books + additional_books = total_books) :
  initial_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l3851_385175


namespace NUMINAMATH_CALUDE_election_total_votes_l3851_385169

/-- Represents the number of votes for each candidate in the election. -/
structure ElectionResult where
  winner : ℕ
  opponent1 : ℕ
  opponent2 : ℕ
  opponent3 : ℕ
  fourth_place : ℕ

/-- Conditions of the election result. -/
def valid_election_result (e : ElectionResult) : Prop :=
  e.winner = e.opponent1 + 53 ∧
  e.winner = e.opponent2 + 79 ∧
  e.winner = e.opponent3 + 105 ∧
  e.fourth_place = 199

/-- Calculates the total votes in the election. -/
def total_votes (e : ElectionResult) : ℕ :=
  e.winner + e.opponent1 + e.opponent2 + e.opponent3 + e.fourth_place

/-- Theorem stating that the total votes in the election is 1598. -/
theorem election_total_votes :
  ∀ e : ElectionResult, valid_election_result e → total_votes e = 1598 :=
by sorry

end NUMINAMATH_CALUDE_election_total_votes_l3851_385169


namespace NUMINAMATH_CALUDE_min_cuts_for_daily_payment_min_cuts_for_all_lengths_l3851_385112

/-- Represents a chain of links -/
structure Chain where
  length : ℕ

/-- Represents a cut strategy for a chain -/
structure CutStrategy where
  cuts : ℕ

/-- Checks if a cut strategy is valid for daily payments -/
def is_valid_daily_payment_strategy (chain : Chain) (strategy : CutStrategy) : Prop :=
  ∀ day : ℕ, day ≤ chain.length → ∃ payment : ℕ, payment = day

/-- Checks if a cut strategy can produce any number of links up to the chain length -/
def can_produce_all_lengths (chain : Chain) (strategy : CutStrategy) : Prop :=
  ∀ n : ℕ, n ≤ chain.length → ∃ combination : List ℕ, combination.sum = n

/-- Theorem for the minimum cuts needed for daily payments -/
theorem min_cuts_for_daily_payment (chain : Chain) (strategy : CutStrategy) : 
  chain.length = 7 → 
  strategy.cuts = 1 → 
  is_valid_daily_payment_strategy chain strategy :=
sorry

/-- Theorem for the minimum cuts needed to produce all lengths -/
theorem min_cuts_for_all_lengths (chain : Chain) (strategy : CutStrategy) :
  chain.length = 2000 →
  strategy.cuts = 7 →
  can_produce_all_lengths chain strategy :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_daily_payment_min_cuts_for_all_lengths_l3851_385112


namespace NUMINAMATH_CALUDE_total_books_bought_l3851_385103

/-- Proves that the total number of books bought is 90 -/
theorem total_books_bought (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_count : ℕ) :
  total_price = 390 →
  math_book_price = 4 →
  history_book_price = 5 →
  math_books_count = 60 →
  math_books_count + (total_price - math_books_count * math_book_price) / history_book_price = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_bought_l3851_385103


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3851_385105

/-- Given that y varies inversely as the square of x, and y = 15 when x = 5,
    prove that y = 375/9 when x = 3. -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → y x = k / (x^2)) →  -- y varies inversely as the square of x
  y 5 = 15 →                           -- y = 15 when x = 5
  y 3 = 375 / 9 :=                     -- y = 375/9 when x = 3
by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l3851_385105


namespace NUMINAMATH_CALUDE_exact_arrival_speed_l3851_385191

theorem exact_arrival_speed (d : ℝ) (t : ℝ) (h1 : d = 30 * (t + 1/30)) (h2 : d = 50 * (t - 1/30)) :
  d / t = 37.5 := by sorry

end NUMINAMATH_CALUDE_exact_arrival_speed_l3851_385191


namespace NUMINAMATH_CALUDE_rachel_picked_four_apples_l3851_385165

/-- The number of apples Rachel picked from her tree -/
def apples_picked (initial_apples remaining_apples : ℕ) : ℕ :=
  initial_apples - remaining_apples

/-- Theorem: Rachel picked 4 apples -/
theorem rachel_picked_four_apples :
  apples_picked 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_picked_four_apples_l3851_385165


namespace NUMINAMATH_CALUDE_range_of_m_given_inequality_and_point_l3851_385157

/-- Given a planar region defined by an inequality and a point within that region,
    this theorem states the range of the parameter m. -/
theorem range_of_m_given_inequality_and_point (m : ℝ) : 
  (∀ x y : ℝ, x - (m^2 - 2*m + 4)*y + 6 > 0 → 
    (1 : ℝ) - (m^2 - 2*m + 4)*(1 : ℝ) + 6 > 0) → 
  m ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_given_inequality_and_point_l3851_385157


namespace NUMINAMATH_CALUDE_square_decomposition_l3851_385110

theorem square_decomposition (a b c k : ℕ) (n : ℕ) (h1 : c^2 = n * a^2 + n * b^2) 
  (h2 : (5*k)^2 = (4*k)^2 + (3*k)^2) (h3 : n = k^2) (h4 : n = 9) : c = 15 :=
sorry

end NUMINAMATH_CALUDE_square_decomposition_l3851_385110


namespace NUMINAMATH_CALUDE_g_of_2_eq_1_l3851_385152

/-- The function that keeps only the last k digits of a number --/
def lastKDigits (n : ℕ) (k : ℕ) : ℕ :=
  n % (10^k)

/-- The sequence of numbers on the board for a given k --/
def boardSequence (k : ℕ) : List ℕ :=
  sorry

/-- The function g(k) as defined in the problem --/
def g (k : ℕ) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem g_of_2_eq_1 : g 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_eq_1_l3851_385152


namespace NUMINAMATH_CALUDE_passengers_in_buses_l3851_385184

/-- Given that 456 passengers fit into 12 buses, 
    prove that 266 passengers fit into 7 buses. -/
theorem passengers_in_buses 
  (total_passengers : ℕ) 
  (total_buses : ℕ) 
  (target_buses : ℕ) 
  (h1 : total_passengers = 456) 
  (h2 : total_buses = 12) 
  (h3 : target_buses = 7) :
  (total_passengers / total_buses) * target_buses = 266 := by
  sorry

end NUMINAMATH_CALUDE_passengers_in_buses_l3851_385184


namespace NUMINAMATH_CALUDE_sound_propagation_all_directions_l3851_385171

/-- Represents the medium through which sound travels -/
inductive Medium
| Air
| Water
| Solid

/-- Represents a direction in 3D space -/
structure Direction where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents sound as a mechanical wave -/
structure Sound where
  medium : Medium
  frequency : ℝ
  amplitude : ℝ

/-- Represents the propagation of sound in a medium -/
def SoundPropagation (s : Sound) (d : Direction) : Prop :=
  match s.medium with
  | Medium.Air => true
  | Medium.Water => true
  | Medium.Solid => true

/-- Theorem stating that sound propagates in all directions in a classroom -/
theorem sound_propagation_all_directions 
  (s : Sound) 
  (h1 : s.medium = Medium.Air) 
  (h2 : ∀ (d : Direction), SoundPropagation s d) : 
  ∀ (d : Direction), SoundPropagation s d :=
sorry

end NUMINAMATH_CALUDE_sound_propagation_all_directions_l3851_385171


namespace NUMINAMATH_CALUDE_ten_thousand_one_divides_same_first_last_four_digits_l3851_385170

-- Define an 8-digit number type
def EightDigitNumber := { n : ℕ // 10000000 ≤ n ∧ n < 100000000 }

-- Define the property of having the same first and last four digits
def SameFirstLastFourDigits (n : EightDigitNumber) : Prop :=
  ∃ (a b c d : ℕ), 
    0 ≤ a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧
    n.val = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
            1000 * a + 100 * b + 10 * c + d

-- Theorem statement
theorem ten_thousand_one_divides_same_first_last_four_digits 
  (n : EightDigitNumber) (h : SameFirstLastFourDigits n) : 
  10001 ∣ n.val :=
sorry

end NUMINAMATH_CALUDE_ten_thousand_one_divides_same_first_last_four_digits_l3851_385170


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_or_cube_l3851_385174

theorem power_of_two_plus_one_square_or_cube (n : ℕ) :
  (∃ m : ℕ, 2^n + 1 = m^2) ∨ (∃ m : ℕ, 2^n + 1 = m^3) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_or_cube_l3851_385174


namespace NUMINAMATH_CALUDE_oranges_in_bin_l3851_385177

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + added = initial + added - thrown_away :=
by sorry

end NUMINAMATH_CALUDE_oranges_in_bin_l3851_385177


namespace NUMINAMATH_CALUDE_speed_conversion_l3851_385161

-- Define the conversion factor from m/s to km/h
def meters_per_second_to_kmph : ℝ := 3.6

-- Define the given speed in meters per second
def speed_ms : ℝ := 200.016

-- Define the speed in km/h that we want to prove
def speed_kmph : ℝ := 720.0576

-- Theorem statement
theorem speed_conversion :
  speed_ms * meters_per_second_to_kmph = speed_kmph :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3851_385161


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3851_385115

/-- The distance between the vertices of a hyperbola with equation (y^2 / 27) - (x^2 / 11) = 1 is 6√3. -/
theorem hyperbola_vertices_distance :
  let hyperbola := {p : ℝ × ℝ | (p.2^2 / 27) - (p.1^2 / 11) = 1}
  ∃ v₁ v₂ : ℝ × ℝ, v₁ ∈ hyperbola ∧ v₂ ∈ hyperbola ∧ 
    ∀ p ∈ hyperbola, dist p v₁ ≤ dist v₁ v₂ ∧ dist p v₂ ≤ dist v₁ v₂ ∧
    dist v₁ v₂ = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3851_385115


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l3851_385121

/-- Given a right pyramid with a square base, prove that the side length of the base is 6 meters 
    when the area of one lateral face is 120 square meters and the slant height is 40 meters. -/
theorem pyramid_base_side_length (lateral_face_area slant_height : ℝ) 
  (h1 : lateral_face_area = 120)
  (h2 : slant_height = 40) : 
  let base_side := lateral_face_area / (0.5 * slant_height)
  base_side = 6 := by sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l3851_385121


namespace NUMINAMATH_CALUDE_magician_hourly_rate_l3851_385140

/-- Proves that the hourly rate for a magician who works 3 hours per day for 2 weeks
    and receives a total payment of $2520 is $60 per hour. -/
theorem magician_hourly_rate :
  let hours_per_day : ℕ := 3
  let days : ℕ := 14
  let total_payment : ℕ := 2520
  let total_hours : ℕ := hours_per_day * days
  let hourly_rate : ℚ := total_payment / total_hours
  hourly_rate = 60 := by
  sorry

#check magician_hourly_rate

end NUMINAMATH_CALUDE_magician_hourly_rate_l3851_385140


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3851_385180

theorem smallest_n_for_inequality : 
  (∃ n : ℕ, ∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) ∧
  (∀ m : ℕ, m < 3 → ∃ x y z : ℝ, (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) :=
by sorry

#check smallest_n_for_inequality

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3851_385180


namespace NUMINAMATH_CALUDE_checker_moves_fibonacci_checker_moves_10_checker_moves_11_l3851_385159

def checkerMoves : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => checkerMoves (n + 1) + checkerMoves n

theorem checker_moves_fibonacci (n : ℕ) :
  checkerMoves n = checkerMoves (n - 1) + checkerMoves (n - 2) :=
by sorry

theorem checker_moves_10 : checkerMoves 10 = 89 :=
by sorry

theorem checker_moves_11 : checkerMoves 11 = 144 :=
by sorry

end NUMINAMATH_CALUDE_checker_moves_fibonacci_checker_moves_10_checker_moves_11_l3851_385159


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3851_385197

/-- Proves that the original bill before tip was $139 given the problem conditions -/
theorem dining_bill_calculation (people : ℕ) (tip_percentage : ℚ) (individual_payment : ℚ) :
  people = 5 ∧ 
  tip_percentage = 1/10 ∧ 
  individual_payment = 3058/100 →
  ∃ (original_bill : ℚ),
    original_bill * (1 + tip_percentage) = people * individual_payment ∧
    original_bill = 139 :=
by sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3851_385197


namespace NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l3851_385149

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 := by
  sorry

theorem max_y_value_achievable : ∃ x y : ℤ, x * y + 3 * x + 2 * y = -4 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_max_y_value_achievable_l3851_385149


namespace NUMINAMATH_CALUDE_grace_age_l3851_385113

/-- Represents the ages of the people in the problem -/
structure Ages where
  grace : ℕ
  faye : ℕ
  chad : ℕ
  eduardo : ℕ
  diana : ℕ

/-- Defines the age relationships between the people -/
def valid_ages (a : Ages) : Prop :=
  a.faye = a.grace + 6 ∧
  a.faye = a.chad + 2 ∧
  a.eduardo = a.chad + 3 ∧
  a.eduardo = a.diana + 4 ∧
  a.diana = 17

/-- Theorem stating that if the ages are valid, Grace's age is 14 -/
theorem grace_age (a : Ages) : valid_ages a → a.grace = 14 := by
  sorry

end NUMINAMATH_CALUDE_grace_age_l3851_385113


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3851_385154

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x > 1 ∧ y < -1 ∧ 
   x^2 + (a^2 + 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 + 1)*y + a - 2 = 0) →
  -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3851_385154


namespace NUMINAMATH_CALUDE_right_triangle_semicircle_segments_l3851_385178

theorem right_triangle_semicircle_segments 
  (a b : ℝ) 
  (ha : a = 75) 
  (hb : b = 100) : 
  ∃ (x y : ℝ), 
    x = 48 ∧ 
    y = 36 ∧ 
    x * (a^2 + b^2) = a * b^2 ∧ 
    y * (a^2 + b^2) = b * a^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_semicircle_segments_l3851_385178


namespace NUMINAMATH_CALUDE_original_number_of_people_l3851_385128

theorem original_number_of_people (x : ℕ) : 
  (x / 2 : ℚ) - (x / 2 : ℚ) / 3 = 12 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_people_l3851_385128


namespace NUMINAMATH_CALUDE_jasper_candy_count_jasper_candy_proof_l3851_385158

theorem jasper_candy_count : ℕ → Prop :=
  fun initial_candies =>
    let day1_remaining := initial_candies - (initial_candies / 4) - 3
    let day2_remaining := day1_remaining - (day1_remaining / 5) - 5
    let day3_remaining := day2_remaining - (day2_remaining / 6) - 2
    day3_remaining = 10 → initial_candies = 537

theorem jasper_candy_proof : jasper_candy_count 537 := by
  sorry

end NUMINAMATH_CALUDE_jasper_candy_count_jasper_candy_proof_l3851_385158


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3851_385190

theorem binomial_square_constant (a : ℚ) : 
  (∃ b c : ℚ, ∀ x, 9*x^2 + 27*x + a = (b*x + c)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3851_385190


namespace NUMINAMATH_CALUDE_lamplighter_monkey_distance_l3851_385114

/-- Represents the speed and time of a monkey's movement --/
structure MonkeyMovement where
  speed : ℝ
  time : ℝ

/-- Calculates the total distance traveled by a Lamplighter monkey --/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.time + swinging.speed * swinging.time

/-- Theorem stating the total distance traveled by the Lamplighter monkey --/
theorem lamplighter_monkey_distance :
  let running := MonkeyMovement.mk 15 5
  let swinging := MonkeyMovement.mk 10 10
  totalDistance running swinging = 175 := by sorry

end NUMINAMATH_CALUDE_lamplighter_monkey_distance_l3851_385114


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3851_385136

theorem quadratic_root_property (n : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ + n = 0) ∧ (x₂^2 - 3*x₂ + n = 0) ∧ (x₁ + x₂ - 2 = x₁ * x₂)) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3851_385136


namespace NUMINAMATH_CALUDE_angle_beta_proof_l3851_385100

theorem angle_beta_proof (α β : Real) (h1 : π / 2 < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19) (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_beta_proof_l3851_385100


namespace NUMINAMATH_CALUDE_linear_regression_center_point_l3851_385168

/-- Given a linear regression equation y = 0.2x - m with the center of sample points at (m, 1.6), prove that m = -2 -/
theorem linear_regression_center_point (m : ℝ) : 
  (∀ x y : ℝ, y = 0.2 * x - m) → -- Linear regression equation
  (m, 1.6) = (m, 0.2 * m - m) → -- Center of sample points
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_linear_regression_center_point_l3851_385168


namespace NUMINAMATH_CALUDE_inequality_range_l3851_385181

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 - Real.log x / Real.log a < 0) ↔ a ∈ Set.Ioo (1/16) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3851_385181


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l3851_385137

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Theorem for A ⊆ B
theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ (4/3 ≤ a ∧ a ≤ 2 ∧ a ≠ 0) :=
sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ (a ≤ 2/3 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l3851_385137


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_non_primes_l3851_385151

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem largest_of_five_consecutive_non_primes :
  ∀ a b c d e : ℕ,
    (10 ≤ a ∧ a < 40) →
    (b = a + 1) →
    (c = b + 1) →
    (d = c + 1) →
    (e = d + 1) →
    (¬ is_prime a) →
    (¬ is_prime b) →
    (¬ is_prime c) →
    (¬ is_prime d) →
    (¬ is_prime e) →
    e = 36 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_non_primes_l3851_385151


namespace NUMINAMATH_CALUDE_sum_coordinates_of_endpoint_l3851_385187

/-- Given a line segment CD with midpoint M(5,5) and endpoint C(7,3),
    the sum of the coordinates of the other endpoint D is 10. -/
theorem sum_coordinates_of_endpoint (C D M : ℝ × ℝ) : 
  M = (5, 5) →
  C = (7, 3) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 10 := by
  sorry

#check sum_coordinates_of_endpoint

end NUMINAMATH_CALUDE_sum_coordinates_of_endpoint_l3851_385187


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3851_385141

theorem complex_expression_equality : (2 - Complex.I)^2 - (1 + 3 * Complex.I) = 2 - 7 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3851_385141


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3851_385122

theorem modulus_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 * i / (i - 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3851_385122


namespace NUMINAMATH_CALUDE_sequence_proof_l3851_385118

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sequence_proof
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = -1)
  (h_b_diff : ∀ n : ℕ, n ≥ 2 → b n - b (n - 1) = a n)
  (h_b1 : b 1 = 1)
  (h_b3 : b 3 = 1) :
  (a 1 = -3) ∧
  (∀ n : ℕ, n ≥ 1 → b n = n^2 - 4*n + 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_proof_l3851_385118


namespace NUMINAMATH_CALUDE_emily_marbles_ratio_l3851_385142

-- Define the initial number of marbles Emily has
def initial_marbles : ℕ := 6

-- Define the number of marbles Megan gives to Emily
def megan_gives : ℕ := 2 * initial_marbles

-- Define Emily's new total before giving marbles back
def new_total : ℕ := initial_marbles + megan_gives

-- Define the number of marbles Emily has at the end
def final_marbles : ℕ := 8

-- Define the number of marbles Emily gives back to Megan
def marbles_given_back : ℕ := new_total - final_marbles

-- Define the ratio of marbles given back to new total
def ratio : Rat := marbles_given_back / new_total

theorem emily_marbles_ratio : ratio = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_emily_marbles_ratio_l3851_385142


namespace NUMINAMATH_CALUDE_expand_product_l3851_385182

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 5) = x^3 + 7*x^2 + 17*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3851_385182


namespace NUMINAMATH_CALUDE_angle_CAD_measure_l3851_385108

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a pentagon -/
structure Pentagon :=
  (B : Point)
  (C : Point)
  (D : Point)
  (E : Point)
  (G : Point)

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Checks if a pentagon is regular -/
def is_regular_pentagon (p : Pentagon) : Prop := sorry

/-- Calculates the angle between three points in degrees -/
def angle_deg (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem angle_CAD_measure 
  (t : Triangle) 
  (p : Pentagon) 
  (h1 : is_equilateral t)
  (h2 : is_regular_pentagon p)
  (h3 : t.B = p.B)
  (h4 : t.C = p.C) :
  angle_deg t.A p.D t.C = 24 := by sorry

end NUMINAMATH_CALUDE_angle_CAD_measure_l3851_385108


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3851_385196

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)

-- Define the constraint for x and y
def constraint (a b x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ a / x + b / y = 1

-- Main theorem
theorem quadratic_inequality_theorem :
  ∃ a b : ℝ,
    -- Part I: Values of a and b
    solution_set a b ∧ a = 1 ∧ b = 2 ∧
    -- Part II: Minimum value of 2x + y
    (∀ x y, constraint a b x y → 2 * x + y ≥ 8) ∧
    -- Part II: Range of k
    (∀ k, (∀ x y, constraint a b x y → 2 * x + y ≥ k^2 + k + 2) ↔ -3 ≤ k ∧ k ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3851_385196


namespace NUMINAMATH_CALUDE_bakery_storage_l3851_385123

theorem bakery_storage (sugar flour baking_soda : ℕ) : 
  (sugar : ℚ) / flour = 5 / 2 →
  (flour : ℚ) / baking_soda = 10 / 1 →
  (flour : ℚ) / (baking_soda + 60) = 8 / 1 →
  sugar = 6000 := by
sorry


end NUMINAMATH_CALUDE_bakery_storage_l3851_385123


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l3851_385130

/-- Represents the number of students in each grade -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Calculates the total number of students -/
def totalStudents (pop : GradePopulation) : ℕ :=
  pop.freshmen + pop.sophomores + pop.seniors

/-- Calculates the number of students to sample from each grade -/
def stratifiedSample (pop : GradePopulation) (sampleSize : ℕ) : GradePopulation :=
  let total := totalStudents pop
  let factor := sampleSize / total
  { freshmen := pop.freshmen * factor,
    sophomores := pop.sophomores * factor,
    seniors := pop.seniors * factor }

theorem stratified_sampling_seniors
  (pop : GradePopulation)
  (h1 : pop.freshmen = 520)
  (h2 : pop.sophomores = 500)
  (h3 : pop.seniors = 580)
  (h4 : totalStudents pop = 1600)
  (sampleSize : ℕ)
  (h5 : sampleSize = 80) :
  (stratifiedSample pop sampleSize).seniors = 29 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l3851_385130


namespace NUMINAMATH_CALUDE_place_value_ratio_l3851_385138

def number : ℚ := 86549.2047

theorem place_value_ratio : 
  let thousands_place_value : ℚ := 1000
  let tenths_place_value : ℚ := 0.1
  thousands_place_value / tenths_place_value = 10000 := by sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3851_385138


namespace NUMINAMATH_CALUDE_floor_of_5_7_l3851_385116

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l3851_385116


namespace NUMINAMATH_CALUDE_original_number_form_l3851_385192

theorem original_number_form (N : ℤ) : 
  (∃ m : ℤ, (N + 3) = 9 * m) → ∃ k : ℤ, N = 9 * k + 3 :=
by sorry

end NUMINAMATH_CALUDE_original_number_form_l3851_385192


namespace NUMINAMATH_CALUDE_min_product_of_three_l3851_385183

def S : Finset Int := {-9, -5, -3, 0, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * y * z ≤ a * b * c ∧ x * y * z = -432 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3851_385183


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3851_385139

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ -3 ↔ x - 5 > 3*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3851_385139


namespace NUMINAMATH_CALUDE_prove_length_l3851_385120

-- Define the points
variable (A O B A1 B1 : ℝ)

-- Define the conditions
axiom collinear : ∃ (t : ℝ), O = t • A + (1 - t) • B
axiom symmetric_A : A1 - O = O - A
axiom symmetric_B : B1 - O = O - B
axiom given_length : abs (A - B1) = 2

-- State the theorem
theorem prove_length : abs (A1 - B) = 2 := by sorry

end NUMINAMATH_CALUDE_prove_length_l3851_385120


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l3851_385109

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- Conversion of a natural number to its base 7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem largest_three_digit_square_base_7 :
  (M * M ≥ 7^2) ∧ 
  (M * M < 7^3) ∧ 
  (∀ n : ℕ, n > M → n * n ≥ 7^3) ∧
  (to_base_7 M = [2, 4]) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l3851_385109


namespace NUMINAMATH_CALUDE_perimeter_difference_l3851_385134

-- Define the perimeter of the first figure
def perimeter_figure1 : ℕ :=
  -- Outer rectangle perimeter
  2 * (5 + 2) +
  -- Middle vertical rectangle contribution
  2 * 3 +
  -- Inner vertical rectangle contribution
  2 * 2

-- Define the perimeter of the second figure
def perimeter_figure2 : ℕ :=
  -- Outer rectangle perimeter
  2 * (5 + 3) +
  -- Vertical lines contribution
  5 * 2

-- Theorem statement
theorem perimeter_difference : perimeter_figure2 - perimeter_figure1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l3851_385134


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3851_385156

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3851_385156


namespace NUMINAMATH_CALUDE_doritos_piles_l3851_385131

theorem doritos_piles (total_bags : ℕ) (doritos_fraction : ℚ) (bags_per_pile : ℕ) : 
  total_bags = 80 →
  doritos_fraction = 1/4 →
  bags_per_pile = 5 →
  (total_bags * doritos_fraction / bags_per_pile : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_doritos_piles_l3851_385131


namespace NUMINAMATH_CALUDE_division_inequality_quotient_invariance_l3851_385106

theorem division_inequality : 0.056 / 0.08 ≠ 0.56 / 0.08 := by
  -- The proof goes here
  sorry

-- Property of invariance of quotient
theorem quotient_invariance (a b c : ℝ) (hc : c ≠ 0) :
  a / b = (a * c) / (b * c) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_inequality_quotient_invariance_l3851_385106


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l3851_385173

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the non-coincidence relation for lines
variable (non_coincident_lines : Line → Line → Prop)

-- Define the non-coincidence relation for planes
variable (non_coincident_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_implies_parallel 
  (m n : Line) (α β : Plane)
  (h1 : non_coincident_lines m n)
  (h2 : non_coincident_planes α β)
  (h3 : perpendicular m α)
  (h4 : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l3851_385173


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3851_385179

theorem multiply_mixed_number : 9 * (7 + 2/5) = 66 + 3/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3851_385179


namespace NUMINAMATH_CALUDE_mets_fan_count_l3851_385125

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def fan_conditions (f : FanCounts) : Prop :=
  (f.yankees : ℚ) / f.mets = 3 / 2 ∧
  (f.mets : ℚ) / f.redsox = 4 / 5 ∧
  f.yankees + f.mets + f.redsox = 330

/-- The theorem to be proved -/
theorem mets_fan_count (f : FanCounts) :
  fan_conditions f → f.mets = 88 := by
  sorry

end NUMINAMATH_CALUDE_mets_fan_count_l3851_385125


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3851_385164

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - x < 0 → -1 < x ∧ x < 1) ∧
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ ¬(x^2 - x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3851_385164


namespace NUMINAMATH_CALUDE_fencing_probability_theorem_l3851_385167

/-- Represents the increase in winning probability for player A in a fencing match -/
def fencing_probability_increase (k l : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose (k + l) k) * (1 - p)^(l + 1) * p^k

/-- Theorem stating the increase in winning probability for player A in a fencing match -/
theorem fencing_probability_theorem (k l : ℕ) (p : ℝ) 
    (h1 : 0 ≤ k ∧ k ≤ 14) (h2 : 0 ≤ l ∧ l ≤ 14) (h3 : 0 ≤ p ∧ p ≤ 1) : 
  fencing_probability_increase k l p = 
    (Nat.choose (k + l) k) * (1 - p)^(l + 1) * p^k := by
  sorry

#check fencing_probability_theorem

end NUMINAMATH_CALUDE_fencing_probability_theorem_l3851_385167


namespace NUMINAMATH_CALUDE_theodore_tax_rate_l3851_385153

/-- Calculates the tax rate for Theodore's statue business --/
theorem theodore_tax_rate :
  let stone_statues : ℕ := 10
  let wooden_statues : ℕ := 20
  let stone_price : ℚ := 20
  let wooden_price : ℚ := 5
  let after_tax_earnings : ℚ := 270
  let before_tax_earnings := stone_statues * stone_price + wooden_statues * wooden_price
  let tax_rate := (before_tax_earnings - after_tax_earnings) / before_tax_earnings
  tax_rate = 1/10 := by sorry

end NUMINAMATH_CALUDE_theodore_tax_rate_l3851_385153


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3851_385102

theorem chinese_remainder_theorem_example :
  ∃ x : ℤ, x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 1 ∧ x % 60 = 11 := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3851_385102


namespace NUMINAMATH_CALUDE_joy_pencil_count_l3851_385107

/-- The number of pencils Colleen has -/
def colleen_pencils : ℕ := 50

/-- The cost of each pencil in dollars -/
def pencil_cost : ℕ := 4

/-- The difference in dollars between what Colleen and Joy paid -/
def payment_difference : ℕ := 80

/-- The number of pencils Joy has -/
def joy_pencils : ℕ := 30

theorem joy_pencil_count :
  colleen_pencils * pencil_cost = joy_pencils * pencil_cost + payment_difference :=
sorry

end NUMINAMATH_CALUDE_joy_pencil_count_l3851_385107


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_is_one_range_of_a_when_B_subset_complementA_l3851_385176

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x ≤ -1 ∨ x ≥ 2}

-- Theorem for part (1)
theorem intersection_A_B_when_a_is_one :
  A ∩ B 1 = {x | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem range_of_a_when_B_subset_complementA :
  ∀ a > 0, (B a ⊆ complementA) ↔ (0 < a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_is_one_range_of_a_when_B_subset_complementA_l3851_385176


namespace NUMINAMATH_CALUDE_power_equality_l3851_385188

theorem power_equality (n b : ℝ) (h1 : n = 2^(1/4)) (h2 : n^b = 8) : b = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3851_385188


namespace NUMINAMATH_CALUDE_max_point_range_l3851_385135

/-- Given a differentiable function f : ℝ → ℝ and a real number a, 
    if f'(x) = a(x-1)(x-a) for all x and f attains a maximum at x = a, 
    then 0 < a < 1 -/
theorem max_point_range (f : ℝ → ℝ) (a : ℝ) 
    (h1 : Differentiable ℝ f) 
    (h2 : ∀ x, deriv f x = a * (x - 1) * (x - a))
    (h3 : IsLocalMax f a) : 
    0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_max_point_range_l3851_385135


namespace NUMINAMATH_CALUDE_percentage_difference_l3851_385101

theorem percentage_difference (T : ℝ) (h1 : T > 0) : 
  let F := 0.70 * T
  let S := 0.90 * F
  (T - S) / T = 0.37 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3851_385101


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3851_385133

/-- Proves that for the line 2x - 3y - 6k = 0, if the sum of its x-intercept and y-intercept is 1, then k = 1 -/
theorem line_intercepts_sum (k : ℝ) : 
  (∃ x y : ℝ, 2*x - 3*y - 6*k = 0 ∧ 
   (2*(3*k) - 3*0 - 6*k = 0) ∧ 
   (2*0 - 3*(-2*k) - 6*k = 0) ∧ 
   3*k + (-2*k) = 1) → 
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3851_385133


namespace NUMINAMATH_CALUDE_first_player_wins_petya_wins_1000x2020_l3851_385193

/-- Represents the state of the rectangular grid game -/
structure GameState where
  m : ℕ+
  n : ℕ+

/-- Determines if a player has a winning strategy based on the game state -/
def has_winning_strategy (state : GameState) : Prop :=
  ∃ (a b : ℕ), state.m = 2^a * (2 * state.m.val + 1) ∧ state.n = 2^b * (2 * state.n.val + 1) ∧ a ≠ b

/-- Theorem stating the winning condition for the first player -/
theorem first_player_wins (initial_state : GameState) :
  has_winning_strategy initial_state ↔ 
  ∀ (strategy : GameState → GameState), 
    ∃ (counter_strategy : GameState → GameState),
      ∀ (game_length : ℕ),
        (game_length > 0 ∧ has_winning_strategy (counter_strategy (strategy initial_state))) ∨
        (game_length = 0 ∧ ¬ has_winning_strategy initial_state) :=
by sorry

/-- The specific case for the 1000 × 2020 grid -/
theorem petya_wins_1000x2020 :
  has_winning_strategy { m := 1000, n := 2020 } :=
by sorry

end NUMINAMATH_CALUDE_first_player_wins_petya_wins_1000x2020_l3851_385193


namespace NUMINAMATH_CALUDE_team_red_cards_l3851_385132

/-- Calculates the number of red cards a soccer team would collect given the total number of players,
    the number of players without cautions, and the yellow-to-red card ratio. -/
def red_cards (total_players : ℕ) (players_without_cautions : ℕ) (yellow_per_red : ℕ) : ℕ :=
  ((total_players - players_without_cautions) * 1) / yellow_per_red

/-- Proves that a team of 11 players with 5 players without cautions would collect 3 red cards,
    given that each red card corresponds to 2 yellow cards. -/
theorem team_red_cards :
  red_cards 11 5 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_red_cards_l3851_385132


namespace NUMINAMATH_CALUDE_edge_probability_is_one_l3851_385119

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, staying in bounds -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨min 3 (p.x + 1), p.y⟩
  | Direction.Down => ⟨max 0 (p.x - 1), p.y⟩
  | Direction.Left => ⟨p.x, max 0 (p.y - 1)⟩
  | Direction.Right => ⟨p.x, min 3 (p.y + 1)⟩

/-- The starting position (2,2) -/
def startPos : Position := ⟨2, 2⟩

/-- Theorem: The probability of reaching an edge cell within three hops from (2,2) is 1 -/
theorem edge_probability_is_one :
  ∀ (hops : List Direction),
    hops.length ≤ 3 →
    isEdge (hops.foldl hop startPos) = true :=
by sorry

end NUMINAMATH_CALUDE_edge_probability_is_one_l3851_385119


namespace NUMINAMATH_CALUDE_distance_on_segment_triangle_inequality_l3851_385166

/-- Custom distance function for points in 2D space -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

/-- Theorem: If C is on the line segment AB, then AC + CB = AB -/
theorem distance_on_segment (x₁ y₁ x₂ y₂ x y : ℝ) 
  (h_x : min x₁ x₂ ≤ x ∧ x ≤ max x₁ x₂) 
  (h_y : min y₁ y₂ ≤ y ∧ y ≤ max y₁ y₂) :
  distance x₁ y₁ x y + distance x y x₂ y₂ = distance x₁ y₁ x₂ y₂ := by sorry

/-- Theorem: For any triangle ABC, AC + CB > AB -/
theorem triangle_inequality (x₁ y₁ x₂ y₂ x y : ℝ) :
  distance x₁ y₁ x y + distance x y x₂ y₂ ≥ distance x₁ y₁ x₂ y₂ := by sorry

end NUMINAMATH_CALUDE_distance_on_segment_triangle_inequality_l3851_385166


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3851_385146

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3851_385146


namespace NUMINAMATH_CALUDE_permutation_five_three_l3851_385186

/-- The number of permutations of n objects taken r at a time -/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else (n - r + 1).factorial / (n - r).factorial

theorem permutation_five_three :
  permutation 5 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_permutation_five_three_l3851_385186


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_l3851_385144

theorem largest_of_five_consecutive_odd_integers (a b c d e : ℤ) : 
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7 ∧ e = 2*n + 9) →
  a + b + c + d + e = 255 →
  max a (max b (max c (max d e))) = 55 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_l3851_385144


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3851_385155

theorem arithmetic_calculation : 4 * 6 * 9 - 18 / 3 + 2^3 = 218 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3851_385155


namespace NUMINAMATH_CALUDE_continued_fraction_value_l3851_385148

theorem continued_fraction_value : ∃ y : ℝ, y > 0 ∧ y = 3 + 9 / (2 + 9 / y) ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l3851_385148


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l3851_385147

theorem arithmetic_progression_of_primes (p₁ p₂ p₃ d : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ →  -- The numbers are prime
  p₁ > 3 → p₂ > 3 → p₃ > 3 →        -- The numbers are greater than 3
  p₁ < p₂ ∧ p₂ < p₃ →               -- The numbers are in ascending order
  p₂ = p₁ + d →                     -- Definition of arithmetic progression
  p₃ = p₁ + 2*d →                   -- Definition of arithmetic progression
  ∃ k : ℕ, d = 6 * k                -- The common difference is divisible by 6
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l3851_385147


namespace NUMINAMATH_CALUDE_no_real_roots_l3851_385195

theorem no_real_roots : ∀ x : ℝ, 2 * x^2 - 4 * x + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3851_385195


namespace NUMINAMATH_CALUDE_table_capacity_l3851_385163

theorem table_capacity (invited : ℕ) (no_show : ℕ) (tables : ℕ) : 
  invited = 68 → no_show = 50 → tables = 6 → 
  (invited - no_show) / tables = 3 := by
  sorry

end NUMINAMATH_CALUDE_table_capacity_l3851_385163


namespace NUMINAMATH_CALUDE_prob_three_correct_is_one_twelfth_l3851_385127

def number_of_houses : ℕ := 5

def probability_three_correct_deliveries : ℚ :=
  (number_of_houses.choose 3 * 1) / number_of_houses.factorial

theorem prob_three_correct_is_one_twelfth :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_correct_is_one_twelfth_l3851_385127


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3851_385104

theorem min_value_of_expression (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3851_385104


namespace NUMINAMATH_CALUDE_inequality_proof_l3851_385126

theorem inequality_proof (a b c d e f : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c ≥ |d * x^2 + e * x + f|) : 
  4 * a * c - b^2 ≥ |4 * d * f - e^2| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3851_385126


namespace NUMINAMATH_CALUDE_sine_sum_greater_cosine_sum_increasing_geometric_sequence_l3851_385111

-- Define an acute-angled triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_to_pi : A + B + C = π

-- Define a geometric sequence
def GeometricSequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Statement for proposition ③
theorem sine_sum_greater_cosine_sum (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B + Real.sin t.C > Real.cos t.A + Real.cos t.B + Real.cos t.C :=
sorry

-- Statement for proposition ④
theorem increasing_geometric_sequence (a : ℕ → ℝ) :
  (GeometricSequence a ∧ (∃ q > 1, ∀ n : ℕ, a (n + 1) = q * a n)) →
  (∀ n : ℕ, a (n + 1) > a n) ∧
  ¬((∀ n : ℕ, a (n + 1) > a n) → (∃ q > 1, ∀ n : ℕ, a (n + 1) = q * a n)) :=
sorry

end NUMINAMATH_CALUDE_sine_sum_greater_cosine_sum_increasing_geometric_sequence_l3851_385111


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3851_385117

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3851_385117


namespace NUMINAMATH_CALUDE_problem_solution_l3851_385145

theorem problem_solution (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y) 
  (h4 : x + y = 3) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3851_385145


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3851_385162

/-- Represents a hyperbola in 2D space -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- Properties of a specific hyperbola -/
def hyperbola_properties (h : Hyperbola) : Prop :=
  ∃ (a b : ℝ),
    -- The center is at the origin
    h.equation 0 0 ∧
    -- The right focus is at (3,0)
    (∃ (x y : ℝ), h.equation x y ∧ x = 3 ∧ y = 0) ∧
    -- The eccentricity is 3/2
    (3 / a = 3 / 2) ∧
    -- The equation of the hyperbola
    (∀ (x y : ℝ), h.equation x y ↔ x^2 / a^2 - y^2 / b^2 = 1)

/-- Theorem: The hyperbola with given properties has the equation x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (h : Hyperbola) (hp : hyperbola_properties h) :
  ∀ (x y : ℝ), h.equation x y ↔ x^2 / 4 - y^2 / 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3851_385162


namespace NUMINAMATH_CALUDE_missing_exponent_proof_l3851_385185

theorem missing_exponent_proof :
  (9 ^ 5.6 * 9 ^ 10.3) / 9 ^ 2.56256 = 9 ^ 13.33744 := by
  sorry

end NUMINAMATH_CALUDE_missing_exponent_proof_l3851_385185


namespace NUMINAMATH_CALUDE_multiply_three_point_six_by_zero_point_twenty_five_l3851_385160

theorem multiply_three_point_six_by_zero_point_twenty_five :
  3.6 * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_point_six_by_zero_point_twenty_five_l3851_385160


namespace NUMINAMATH_CALUDE_fuel_station_total_cost_l3851_385189

/-- Calculates the total cost for filling up mini-vans and trucks at a fuel station -/
theorem fuel_station_total_cost
  (service_cost : ℝ)
  (fuel_cost_per_liter : ℝ)
  (num_minivans : ℕ)
  (num_trucks : ℕ)
  (minivan_tank_capacity : ℝ)
  (truck_tank_capacity_factor : ℝ)
  (h1 : service_cost = 2.20)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : num_minivans = 4)
  (h4 : num_trucks = 2)
  (h5 : minivan_tank_capacity = 65)
  (h6 : truck_tank_capacity_factor = 2.20) :
  let truck_tank_capacity := minivan_tank_capacity * truck_tank_capacity_factor
  let total_service_cost := service_cost * (num_minivans + num_trucks)
  let total_fuel_cost_minivans := num_minivans * minivan_tank_capacity * fuel_cost_per_liter
  let total_fuel_cost_trucks := num_trucks * truck_tank_capacity * fuel_cost_per_liter
  let total_cost := total_service_cost + total_fuel_cost_minivans + total_fuel_cost_trucks
  total_cost = 395.40 := by
  sorry

#eval 2.20 * (4 + 2) + 4 * 65 * 0.70 + 2 * (65 * 2.20) * 0.70

end NUMINAMATH_CALUDE_fuel_station_total_cost_l3851_385189


namespace NUMINAMATH_CALUDE_sqrt_sum_upper_bound_sqrt_sum_upper_bound_tight_l3851_385150

theorem sqrt_sum_upper_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) + 
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 4 / Real.sqrt 3 :=
by sorry

theorem sqrt_sum_upper_bound_tight :
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) + 
    Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) = 4 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_upper_bound_sqrt_sum_upper_bound_tight_l3851_385150


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l3851_385172

/-- Represents the molecular weight of a compound in g/mol -/
def molecular_weight : ℝ := 816

/-- Represents the number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- Theorem stating that the molecular weight remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  ∀ n : ℝ, n > 0 → molecular_weight = 816 := by
  sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l3851_385172


namespace NUMINAMATH_CALUDE_impossible_equal_distribution_l3851_385199

/-- Represents the state of coins on the hexagon vertices -/
def HexagonState := Fin 6 → ℕ

/-- The initial state of the hexagon -/
def initial_state : HexagonState := fun i => if i = 0 then 1 else 0

/-- Represents a valid move in the game -/
def valid_move (s1 s2 : HexagonState) : Prop :=
  ∃ (i j : Fin 6) (n : ℕ), 
    (j = i + 1 ∨ j = i - 1 ∨ (i = 5 ∧ j = 0) ∨ (i = 0 ∧ j = 5)) ∧
    s2 i + 6 * n = s1 i ∧
    s2 j = s1 j + 6 * n ∧
    ∀ k, k ≠ i ∧ k ≠ j → s2 k = s1 k

/-- A sequence of valid moves -/
def valid_sequence (s : ℕ → HexagonState) : Prop :=
  s 0 = initial_state ∧ ∀ n, valid_move (s n) (s (n + 1))

/-- The theorem to be proved -/
theorem impossible_equal_distribution :
  ¬∃ (s : ℕ → HexagonState) (n : ℕ), 
    valid_sequence s ∧ 
    (∀ (i j : Fin 6), s n i = s n j) :=
sorry

end NUMINAMATH_CALUDE_impossible_equal_distribution_l3851_385199


namespace NUMINAMATH_CALUDE_age_difference_l3851_385143

/-- Proves that the age difference between a man and his student is 26 years -/
theorem age_difference (student_age man_age : ℕ) : 
  student_age = 24 →
  man_age + 2 = 2 * (student_age + 2) →
  man_age - student_age = 26 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3851_385143


namespace NUMINAMATH_CALUDE_students_not_eating_lunch_l3851_385124

theorem students_not_eating_lunch (total_students : ℕ) 
  (cafeteria_students : ℕ) (h1 : total_students = 60) 
  (h2 : cafeteria_students = 10) : 
  total_students - (3 * cafeteria_students + cafeteria_students) = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_not_eating_lunch_l3851_385124


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3851_385129

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m + 2) * 0 - (m + 1) * 1 + m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3851_385129
