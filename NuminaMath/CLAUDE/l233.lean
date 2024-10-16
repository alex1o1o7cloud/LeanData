import Mathlib

namespace NUMINAMATH_CALUDE_square_area_14m_l233_23365

/-- The area of a square with side length 14 meters is 196 square meters. -/
theorem square_area_14m (side_length : ℝ) (h : side_length = 14) : 
  side_length * side_length = 196 := by
  sorry

end NUMINAMATH_CALUDE_square_area_14m_l233_23365


namespace NUMINAMATH_CALUDE_max_notebooks_buyable_l233_23306

def john_money : ℚ := 35.45
def notebook_cost : ℚ := 3.75

theorem max_notebooks_buyable :
  ⌊john_money / notebook_cost⌋ = 9 :=
sorry

end NUMINAMATH_CALUDE_max_notebooks_buyable_l233_23306


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l233_23350

theorem sum_of_odd_numbers (N : ℕ) : 
  1001 + 1003 + 1005 + 1007 + 1009 = 5050 - N → N = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l233_23350


namespace NUMINAMATH_CALUDE_partnership_investment_timing_l233_23360

theorem partnership_investment_timing 
  (x : ℝ) 
  (m : ℝ) 
  (total_gain : ℝ) 
  (a_share : ℝ) 
  (h1 : total_gain = 18600) 
  (h2 : a_share = 6200) 
  (h3 : a_share / total_gain = 1 / 3) 
  (h4 : x * 12 = (1 / 3) * (x * 12 + 2 * x * (12 - m) + 3 * x * 4)) : 
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_timing_l233_23360


namespace NUMINAMATH_CALUDE_bug_position_after_2012_jumps_l233_23300

/-- Represents the five points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is even -/
def Point.isEven : Point → Bool
  | .two => true
  | .four => true
  | _ => false

/-- Calculates the next point after a jump -/
def nextPoint (p : Point) : Point :=
  match p with
  | .one => .three
  | .two => .five
  | .three => .five
  | .four => .two
  | .five => .two

/-- Calculates the point after n jumps -/
def jumpNTimes (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpNTimes start n)

theorem bug_position_after_2012_jumps :
  jumpNTimes Point.five 2012 = Point.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2012_jumps_l233_23300


namespace NUMINAMATH_CALUDE_sector_area_l233_23356

/-- The area of a circular sector with central angle 120° and radius 4 is 16π/3 -/
theorem sector_area : 
  let central_angle : ℝ := 120
  let radius : ℝ := 4
  let sector_area : ℝ := (central_angle * π * radius^2) / 360
  sector_area = 16 * π / 3 := by sorry

end NUMINAMATH_CALUDE_sector_area_l233_23356


namespace NUMINAMATH_CALUDE_min_value_expression_l233_23371

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + z^2 = 8) : 
  (x + y) / z + (y + z) / x^2 + (z + x) / y^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l233_23371


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l233_23322

/-- Calculates the number of students selected from a class in stratified sampling -/
def stratified_sample (class_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (class_size * sample_size) / total_size

/-- Represents the stratified sampling scenario -/
structure StratifiedSampling where
  class1_size : ℕ
  class2_size : ℕ
  total_sample_size : ℕ

/-- Theorem stating the result of the stratified sampling problem -/
theorem stratified_sampling_result (s : StratifiedSampling) 
  (h1 : s.class1_size = 36)
  (h2 : s.class2_size = 42)
  (h3 : s.total_sample_size = 13) :
  stratified_sample s.class2_size (s.class1_size + s.class2_size) s.total_sample_size = 7 := by
  sorry

#eval stratified_sample 42 (36 + 42) 13

end NUMINAMATH_CALUDE_stratified_sampling_result_l233_23322


namespace NUMINAMATH_CALUDE_fraction_equality_l233_23348

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 4 * y) = 3) : 
  (2 * x + 4 * y) / (4 * x - 2 * y) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l233_23348


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l233_23346

theorem reciprocal_of_negative_2022 : (1 : ℚ) / (-2022 : ℚ) = -1 / 2022 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l233_23346


namespace NUMINAMATH_CALUDE_perpendicular_line_angle_of_inclination_l233_23361

theorem perpendicular_line_angle_of_inclination 
  (line_eq : ℝ → ℝ → Prop) 
  (h_line_eq : ∀ x y, line_eq x y ↔ x + Real.sqrt 3 * y + 2 = 0) :
  ∃ θ : ℝ, 
    0 ≤ θ ∧ 
    θ < π ∧ 
    (∀ x y, line_eq x y → 
      ∃ m : ℝ, m * Real.tan θ = -1 ∧ 
      ∀ x' y', y' - y = m * (x' - x)) ∧ 
    θ = π / 3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_angle_of_inclination_l233_23361


namespace NUMINAMATH_CALUDE_min_value_expression_l233_23394

theorem min_value_expression :
  (∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ -1.125) ∧
  (∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = -1.125) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l233_23394


namespace NUMINAMATH_CALUDE_find_number_l233_23307

theorem find_number (N : ℚ) : (4 / 5 * N) + 18 = N / (4 / 5) → N = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l233_23307


namespace NUMINAMATH_CALUDE_f_minus_one_equals_eight_l233_23304

def f (x : ℝ) (c : ℝ) := x^2 + c

theorem f_minus_one_equals_eight (c : ℝ) (h : f 1 c = 8) : f (-1) c = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_one_equals_eight_l233_23304


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l233_23353

theorem relationship_between_exponents 
  (m p t q : ℝ) 
  (n r s u : ℕ) 
  (h1 : (m^n)^2 = p^r)
  (h2 : p^r = t)
  (h3 : p^s = (m^u)^3)
  (h4 : (m^u)^3 = q)
  : 3 * u * r = 2 * n * s := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l233_23353


namespace NUMINAMATH_CALUDE_rational_includes_positive_and_negative_l233_23316

-- Define rational numbers
def RationalNumber : Type := ℚ

-- Define positive and negative rational numbers
def PositiveRational (q : ℚ) : Prop := q > 0
def NegativeRational (q : ℚ) : Prop := q < 0

-- State the theorem
theorem rational_includes_positive_and_negative :
  (∃ q : ℚ, PositiveRational q) ∧ (∃ q : ℚ, NegativeRational q) :=
sorry

end NUMINAMATH_CALUDE_rational_includes_positive_and_negative_l233_23316


namespace NUMINAMATH_CALUDE_m_range_l233_23342

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being decreasing on [-1,1]
def is_decreasing_on (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f x > f y

-- State the theorem
theorem m_range (f : ℝ → ℝ) (m : ℝ) 
  (h1 : is_decreasing_on f) 
  (h2 : f (m - 1) > f (2*m - 1)) : 
  0 < m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_m_range_l233_23342


namespace NUMINAMATH_CALUDE_quadratic_polynomial_prime_values_l233_23355

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial := ℤ → ℤ

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := sorry

/-- Predicate to check if a polynomial takes prime values at three consecutive integer points -/
def TakesPrimeValuesAtThreeConsecutivePoints (f : QuadraticPolynomial) : Prop :=
  ∃ n : ℤ, IsPrime (f (n - 1)) ∧ IsPrime (f n) ∧ IsPrime (f (n + 1))

/-- Predicate to check if a polynomial takes a prime value at least at one more integer point -/
def TakesPrimeValueAtOneMorePoint (f : QuadraticPolynomial) : Prop :=
  ∃ m : ℤ, (∀ n : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1) → IsPrime (f m)

/-- Theorem stating that if a quadratic polynomial with integer coefficients takes prime values
    at three consecutive integer points, then it takes a prime value at least at one more integer point -/
theorem quadratic_polynomial_prime_values (f : QuadraticPolynomial) :
  TakesPrimeValuesAtThreeConsecutivePoints f → TakesPrimeValueAtOneMorePoint f :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_prime_values_l233_23355


namespace NUMINAMATH_CALUDE_expected_worth_is_one_third_l233_23399

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the probability of a coin flip outcome -/
def probability : CoinFlip → ℚ
| CoinFlip.Heads => 2/3
| CoinFlip.Tails => 1/3

/-- Represents the monetary outcome of a coin flip -/
def monetaryOutcome : CoinFlip → ℤ
| CoinFlip.Heads => 5
| CoinFlip.Tails => -9

/-- The expected worth of a coin flip -/
def expectedWorth : ℚ :=
  (probability CoinFlip.Heads * monetaryOutcome CoinFlip.Heads) +
  (probability CoinFlip.Tails * monetaryOutcome CoinFlip.Tails)

theorem expected_worth_is_one_third :
  expectedWorth = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_is_one_third_l233_23399


namespace NUMINAMATH_CALUDE_average_parking_cost_for_9_hours_l233_23343

/-- Calculates the average cost per hour for parking given the following conditions:
  * Base cost for up to 2 hours
  * Additional cost per hour after 2 hours
  * Total number of hours parked
-/
def averageParkingCost (baseCost hourlyRate : ℚ) (totalHours : ℕ) : ℚ :=
  let totalCost := baseCost + hourlyRate * (totalHours - 2)
  totalCost / totalHours

/-- Theorem stating that the average parking cost for 9 hours is $3.03 -/
theorem average_parking_cost_for_9_hours :
  averageParkingCost 15 (7/4) 9 = 303/100 := by
  sorry

#eval averageParkingCost 15 (7/4) 9

end NUMINAMATH_CALUDE_average_parking_cost_for_9_hours_l233_23343


namespace NUMINAMATH_CALUDE_min_value_tangent_line_circle_l233_23351

theorem min_value_tangent_line_circle (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 ≥ 2) → 
  (∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, x + y + a' = 0 → (x - b')^2 + (y - 1)^2 ≥ 2) → 
    (∃ x y : ℝ, x + y + a' = 0 ∧ (x - b')^2 + (y - 1)^2 = 2) → 
    (3 - 2*b)^2 / (2*a) ≤ (3 - 2*b')^2 / (2*a')) → 
  (3 - 2*b)^2 / (2*a) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_tangent_line_circle_l233_23351


namespace NUMINAMATH_CALUDE_least_b_value_l233_23340

theorem least_b_value (a b : ℕ+) : 
  (∃ p : ℕ+, p.val.Prime ∧ p > 2 ∧ a = p^2) → -- a is the square of the next smallest prime after 2
  (Finset.card (Nat.divisors a) = 3) →        -- a has 3 factors
  (Finset.card (Nat.divisors b) = a) →        -- b has a factors
  (a ∣ b) →                                   -- b is divisible by a
  b ≥ 36 :=                                   -- the least possible value of b is 36
by sorry

end NUMINAMATH_CALUDE_least_b_value_l233_23340


namespace NUMINAMATH_CALUDE_game_probability_result_l233_23383

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℚ) : ℚ :=
  let chelsea_prob := (1 - alex_prob) / (1 + mel_chelsea_ratio)
  let mel_prob := chelsea_prob * mel_chelsea_ratio
  let specific_sequence_prob := alex_prob^4 * mel_prob^2 * chelsea_prob
  let arrangements := (Nat.factorial total_rounds) / 
                      ((Nat.factorial 4) * (Nat.factorial 2) * (Nat.factorial 1))
  arrangements * specific_sequence_prob

theorem game_probability_result : 
  game_probability 7 (1/2) 2 = 35/288 := by sorry

end NUMINAMATH_CALUDE_game_probability_result_l233_23383


namespace NUMINAMATH_CALUDE_horse_cost_problem_l233_23364

theorem horse_cost_problem (selling_price : ℕ) (cost : ℕ) : 
  selling_price = 56 →
  selling_price = cost + (cost * cost) / 100 →
  cost = 40 := by
sorry

end NUMINAMATH_CALUDE_horse_cost_problem_l233_23364


namespace NUMINAMATH_CALUDE_intersection_M_N_l233_23384

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l233_23384


namespace NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l233_23301

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initial_level : ℚ
  fill_rate : ℚ
  empty_rate : ℚ

/-- Calculates the time to empty or fill the tank when both pipes are open -/
def time_to_empty_or_fill (tank : WaterTank) : ℚ :=
  tank.initial_level / (tank.empty_rate - tank.fill_rate)

/-- Theorem stating that the tank will be emptied in 3 minutes under given conditions -/
theorem tank_emptied_in_three_minutes :
  let tank : WaterTank := {
    initial_level := 1/5,
    fill_rate := 1/10,
    empty_rate := 1/6
  }
  time_to_empty_or_fill tank = 3 := by
  sorry

#eval time_to_empty_or_fill {
  initial_level := 1/5,
  fill_rate := 1/10,
  empty_rate := 1/6
}

end NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l233_23301


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l233_23305

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l233_23305


namespace NUMINAMATH_CALUDE_tigger_climbing_speed_ratio_l233_23398

theorem tigger_climbing_speed_ratio :
  ∀ (T t : ℝ),
  T > 0 ∧ t > 0 →
  2 * T = t / 3 →
  T + t = 2 * T + t / 3 →
  T / t = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_tigger_climbing_speed_ratio_l233_23398


namespace NUMINAMATH_CALUDE_total_fruit_salads_l233_23362

/-- The total number of fruit salads in three restaurants -/
theorem total_fruit_salads (alaya angel betty : ℕ) : 
  alaya = 200 →
  angel = 2 * alaya →
  betty = 3 * angel →
  alaya + angel + betty = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fruit_salads_l233_23362


namespace NUMINAMATH_CALUDE_evaluate_expression_l233_23372

theorem evaluate_expression : 3 * Real.sqrt 32 + 2 * Real.sqrt 50 = 22 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l233_23372


namespace NUMINAMATH_CALUDE_line_generates_surface_l233_23320

-- Define the parabolas and the plane
def parabola1 (x y z : ℝ) : Prop := y^2 = 2*x ∧ z = 0
def parabola2 (x y z : ℝ) : Prop := 3*x = z^2 ∧ y = 0
def plane (y z : ℝ) : Prop := y = z

-- Define a line parallel to the plane y = z
def parallel_line (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ × ℝ), p ∈ L → q ∈ L → plane p.2.1 p.2.2 = plane q.2.1 q.2.2

-- Define the intersection of the line with the parabolas
def intersects_parabolas (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  (∃ p ∈ L, parabola1 p.1 p.2.1 p.2.2) ∧ (∃ q ∈ L, parabola2 q.1 q.2.1 q.2.2)

-- The main theorem
theorem line_generates_surface (L : Set (ℝ × ℝ × ℝ)) :
  parallel_line L → intersects_parabolas L →
  ∀ (x y z : ℝ), (x, y, z) ∈ L → x = (y - z) * (y/2 - z/3) :=
sorry

end NUMINAMATH_CALUDE_line_generates_surface_l233_23320


namespace NUMINAMATH_CALUDE_geometric_series_equality_l233_23397

/-- Defines the sum of the first n terms of the geometric series A_n -/
def A (n : ℕ) : ℚ := 704 * (1 - (1/2)^n) / (1 - 1/2)

/-- Defines the sum of the first n terms of the geometric series B_n -/
def B (n : ℕ) : ℚ := 1984 * (1 - (1/(-2))^n) / (1 + 1/2)

/-- Proves that the smallest positive integer n for which A_n = B_n is 5 -/
theorem geometric_series_equality :
  ∀ n : ℕ, n ≥ 1 → (A n = B n ↔ n = 5) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l233_23397


namespace NUMINAMATH_CALUDE_third_number_proof_l233_23317

theorem third_number_proof (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 922) : 
  (∃ (k l m : ℕ), a = 64 * k + 22 ∧ b = 64 * l + 22 ∧ c = 64 * m + 22) ∧ 
  (∀ x : ℕ, b < x ∧ x < c → ¬(∃ n : ℕ, x = 64 * n + 22)) := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l233_23317


namespace NUMINAMATH_CALUDE_water_velocity_proof_l233_23311

-- Define the relationship between force, height, and velocity
def force_relation (k : ℝ) (H : ℝ) (V : ℝ) : ℝ := k * H * V^3

-- Theorem statement
theorem water_velocity_proof :
  ∀ k : ℝ,
  -- Given conditions
  (force_relation k 1 5 = 100) →
  -- Prove that
  (force_relation k 8 10 = 6400) :=
by
  sorry

end NUMINAMATH_CALUDE_water_velocity_proof_l233_23311


namespace NUMINAMATH_CALUDE_total_amount_pens_pencils_l233_23331

/-- The total amount spent on pens and pencils -/
def total_amount (num_pens : ℕ) (num_pencils : ℕ) (price_pen : ℚ) (price_pencil : ℚ) : ℚ :=
  num_pens * price_pen + num_pencils * price_pencil

/-- Theorem stating the total amount spent on pens and pencils -/
theorem total_amount_pens_pencils :
  total_amount 30 75 12 2 = 510 := by
  sorry

#eval total_amount 30 75 12 2

end NUMINAMATH_CALUDE_total_amount_pens_pencils_l233_23331


namespace NUMINAMATH_CALUDE_factorial_1000_trailing_zeros_l233_23393

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: 1000! ends with 249 zeros -/
theorem factorial_1000_trailing_zeros :
  trailingZeros 1000 = 249 := by
  sorry

end NUMINAMATH_CALUDE_factorial_1000_trailing_zeros_l233_23393


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l233_23358

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) := ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_third_term : a 3 = 27)
  (h_ninth_term : a 9 = 3) :
  a 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l233_23358


namespace NUMINAMATH_CALUDE_square_side_lengths_l233_23321

theorem square_side_lengths (a b : ℕ) : 
  a > b → a ^ 2 - b ^ 2 = 2001 → a ∈ ({1001, 335, 55, 49} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_square_side_lengths_l233_23321


namespace NUMINAMATH_CALUDE_total_persimmons_in_boxes_l233_23390

/-- Given that each box contains 100 persimmons and there are 6 boxes,
    prove that the total number of persimmons is 600. -/
theorem total_persimmons_in_boxes : 
  let persimmons_per_box : ℕ := 100
  let number_of_boxes : ℕ := 6
  persimmons_per_box * number_of_boxes = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_persimmons_in_boxes_l233_23390


namespace NUMINAMATH_CALUDE_circle_square_intersection_l233_23330

theorem circle_square_intersection (r : ℝ) (s : ℝ) (x : ℝ) :
  r = 2 →
  s = 2 →
  (π * r^2 - (s^2 - (π * r^2 - 2 * r * x + x^2))) = 2 →
  x = π / 3 + Real.sqrt 3 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_square_intersection_l233_23330


namespace NUMINAMATH_CALUDE_tangent_range_l233_23318

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The equation of the tangent line passing through (a, f(a)) and (2, t) --/
def tangent_equation (a t : ℝ) : Prop :=
  t - (f a) = (f' a) * (2 - a)

/-- The condition for three distinct tangent lines --/
def three_tangents (t : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    tangent_equation a t ∧ tangent_equation b t ∧ tangent_equation c t

/-- Theorem: If a point (2, t) can be used to draw three tangent lines to y = f(x),
    then t is in the open interval (-6, 2) --/
theorem tangent_range :
  ∀ t : ℝ, three_tangents t → -6 < t ∧ t < 2 := by sorry

end NUMINAMATH_CALUDE_tangent_range_l233_23318


namespace NUMINAMATH_CALUDE_point_on_line_l233_23332

/-- Given a line y = mx + b where m is the slope and b is the y-intercept,
    if m + b = 3, then the point (1, 3) lies on this line. -/
theorem point_on_line (m b : ℝ) (h : m + b = 3) :
  let f : ℝ → ℝ := fun x ↦ m * x + b
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l233_23332


namespace NUMINAMATH_CALUDE_festival_attendance_l233_23375

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900)
  (h3 : ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧
    (3 * girls) / 4 + (boys / 3) = festival_attendees) :
  ∃ (girls : ℕ), (3 * girls) / 4 = 720 := by
sorry

end NUMINAMATH_CALUDE_festival_attendance_l233_23375


namespace NUMINAMATH_CALUDE_meeting_organization_count_l233_23395

/-- The number of ways to organize a leadership meeting -/
def organize_meeting (total_schools : ℕ) (members_per_school : ℕ) 
  (host_representatives : ℕ) (other_representatives : ℕ) : ℕ :=
  total_schools * (members_per_school.choose host_representatives) * 
  ((members_per_school.choose other_representatives) ^ (total_schools - 1))

/-- Theorem stating the number of ways to organize the meeting -/
theorem meeting_organization_count :
  organize_meeting 4 6 3 1 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_meeting_organization_count_l233_23395


namespace NUMINAMATH_CALUDE_trapezoid_parallel_sides_l233_23337

/-- Trapezoid properties and parallel sides calculation -/
theorem trapezoid_parallel_sides 
  (t : ℝ) 
  (m : ℝ) 
  (n : ℝ) 
  (E : ℝ) 
  (h_t : t = 204) 
  (h_m : m = 14) 
  (h_n : n = 2) 
  (h_E : E = 59 + 29/60 + 23/3600) : 
  ∃ (a c : ℝ), 
    a - c = m ∧ 
    t = (a + c) / 2 * (2 * t / (a + c)) ∧ 
    a = 24 ∧ 
    c = 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_parallel_sides_l233_23337


namespace NUMINAMATH_CALUDE_complement_of_union_l233_23352

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def M : Set ℕ := {1,3,5,7}
def N : Set ℕ := {5,6,7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l233_23352


namespace NUMINAMATH_CALUDE_largest_power_dividing_product_l233_23396

-- Define pow function
def pow (n : ℕ) : ℕ :=
  sorry

-- Define the product of pow(n) from 2 to 7000
def product : ℕ :=
  sorry

-- State the theorem
theorem largest_power_dividing_product :
  ∃ m : ℕ, (4620 ^ m : ℕ) ∣ product ∧
  ∀ k : ℕ, (4620 ^ k : ℕ) ∣ product → k ≤ m ∧
  m = 698 :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_product_l233_23396


namespace NUMINAMATH_CALUDE_quadratic_factorization_l233_23328

theorem quadratic_factorization (a b : ℤ) : 
  (∀ x : ℝ, 12 * x^2 - 38 * x - 40 = (4 * x + a) * (3 * x + b)) → 
  a - 3 * b = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l233_23328


namespace NUMINAMATH_CALUDE_smallest_value_l233_23347

theorem smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l233_23347


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l233_23329

theorem min_positive_temperatures (x y : ℕ) : 
  x * (x - 1) = 90 →
  y * (y - 1) + (10 - y) * (9 - y) = 48 →
  y ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l233_23329


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainders_l233_23391

theorem unique_divisor_with_remainders :
  ∃! N : ℕ,
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧
    5879 % N = 14 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainders_l233_23391


namespace NUMINAMATH_CALUDE_find_x_when_y_is_8_l233_23378

-- Define the relationship between x and y
def varies_directly (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k * Real.sqrt x

-- State the theorem
theorem find_x_when_y_is_8 :
  ∀ x₀ y₀ x y : ℝ,
  varies_directly x₀ y₀ →
  varies_directly x y →
  x₀ = 3 →
  y₀ = 2 →
  y = 8 →
  x = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_find_x_when_y_is_8_l233_23378


namespace NUMINAMATH_CALUDE_jack_total_miles_driven_l233_23367

/-- Calculates the total miles driven given the number of years and miles driven per four-month period -/
def total_miles_driven (years : ℕ) (miles_per_period : ℕ) : ℕ :=
  let months : ℕ := years * 12
  let periods : ℕ := months / 4
  periods * miles_per_period

/-- Proves that given 9 years of driving and 37,000 miles driven every four months, the total miles driven is 999,000 -/
theorem jack_total_miles_driven :
  total_miles_driven 9 37000 = 999000 := by
  sorry

#eval total_miles_driven 9 37000

end NUMINAMATH_CALUDE_jack_total_miles_driven_l233_23367


namespace NUMINAMATH_CALUDE_didi_fundraising_price_per_slice_l233_23308

/-- Proves that the price per slice is $1 given the conditions of Didi's fundraising event --/
theorem didi_fundraising_price_per_slice :
  ∀ (price_per_slice : ℚ),
    (10 : ℕ) * (8 : ℕ) * price_per_slice +  -- Revenue from slice sales
    (10 : ℕ) * (8 : ℕ) * (1/2 : ℚ) +        -- Donation from first business owner
    (10 : ℕ) * (8 : ℕ) * (1/4 : ℚ) = 140    -- Donation from second business owner
    → price_per_slice = 1 := by
  sorry

end NUMINAMATH_CALUDE_didi_fundraising_price_per_slice_l233_23308


namespace NUMINAMATH_CALUDE_power_simplification_l233_23334

theorem power_simplification : 
  ((12^15 / 12^7)^3 * 8^3) / 2^9 = 12^24 := by sorry

end NUMINAMATH_CALUDE_power_simplification_l233_23334


namespace NUMINAMATH_CALUDE_max_value_of_expression_l233_23370

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 ∧
    (a' * b') / (a' + b') + (a' * c') / (a' + c') + (b' * c') / (b' + c') = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l233_23370


namespace NUMINAMATH_CALUDE_intersection_point_sum_l233_23376

theorem intersection_point_sum (a b : ℚ) : 
  (∃ x y : ℚ, x = (1/4)*y + a ∧ y = (1/4)*x + b ∧ x = 1 ∧ y = 2) →
  a + b = 9/4 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l233_23376


namespace NUMINAMATH_CALUDE_distance_P_to_y_axis_l233_23315

/-- The distance from a point to the y-axis in a Cartesian coordinate system --/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- Point P in the Cartesian coordinate system --/
def P : ℝ × ℝ := (-3, 4)

/-- Theorem: The distance from P(-3,4) to the y-axis is 3 --/
theorem distance_P_to_y_axis :
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_y_axis_l233_23315


namespace NUMINAMATH_CALUDE_circle_symmetry_l233_23382

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 2*y = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 3/2)^2 = 5/4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
  original_circle x y →
  symmetry_line ((x + x') / 2) ((y + y') / 2) →
  symmetric_circle x' y' :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l233_23382


namespace NUMINAMATH_CALUDE_train_passing_tree_l233_23363

/-- Proves that a train of given length and speed takes the calculated time to pass a tree -/
theorem train_passing_tree (train_length : ℝ) (train_speed_km_hr : ℝ) (time : ℝ) :
  train_length = 175 →
  train_speed_km_hr = 63 →
  time = train_length / (train_speed_km_hr * (1000 / 3600)) →
  time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_tree_l233_23363


namespace NUMINAMATH_CALUDE_milk_cost_percentage_l233_23359

-- Define the costs and total amount
def sandwich_cost : ℝ := 4
def juice_cost : ℝ := 2 * sandwich_cost
def total_paid : ℝ := 21

-- Define the total cost of sandwich and juice
def sandwich_juice_total : ℝ := sandwich_cost + juice_cost

-- Define the cost of milk
def milk_cost : ℝ := total_paid - sandwich_juice_total

-- The theorem to prove
theorem milk_cost_percentage : 
  (milk_cost / sandwich_juice_total) * 100 = 75 := by sorry

end NUMINAMATH_CALUDE_milk_cost_percentage_l233_23359


namespace NUMINAMATH_CALUDE_symmetry_probability_l233_23314

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- The size of the square grid -/
def gridSize : Nat := 13

/-- The center point P -/
def centerPoint : GridPoint := ⟨7, 7⟩

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of points excluding the center point -/
def pointsExcludingCenter : Nat := totalPoints - 1

/-- Checks if a point is on a line of symmetry through the center -/
def isOnSymmetryLine (q : GridPoint) : Prop :=
  q.x = centerPoint.x ∨ 
  q.y = centerPoint.y ∨ 
  q.x - centerPoint.x = q.y - centerPoint.y ∨
  q.x - centerPoint.x = centerPoint.y - q.y

/-- The number of points on lines of symmetry (excluding the center) -/
def symmetricPoints : Nat := 48

/-- The theorem stating the probability of Q being on a line of symmetry -/
theorem symmetry_probability : 
  (symmetricPoints : ℚ) / pointsExcludingCenter = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_symmetry_probability_l233_23314


namespace NUMINAMATH_CALUDE_food_storage_temperature_l233_23366

-- Define the temperature range
def temp_center : ℝ := -2
def temp_range : ℝ := 3

-- Define the function to check if a temperature is within the range
def is_within_range (temp : ℝ) : Prop :=
  temp ≥ temp_center - temp_range ∧ temp ≤ temp_center + temp_range

-- State the theorem
theorem food_storage_temperature :
  is_within_range (-1) ∧
  ¬is_within_range 2 ∧
  ¬is_within_range (-6) ∧
  ¬is_within_range 4 :=
sorry

end NUMINAMATH_CALUDE_food_storage_temperature_l233_23366


namespace NUMINAMATH_CALUDE_curve_is_ellipse_l233_23333

open Real

/-- Given that θ is an internal angle of an oblique triangle and 
    F: x²sin²θcos²θ + y²sin²θ = cos²θ is the equation of a curve,
    prove that F represents an ellipse with foci on the x-axis and eccentricity sin θ. -/
theorem curve_is_ellipse (θ : ℝ) (h1 : 0 < θ ∧ θ < π) 
  (h2 : ∀ (x y : ℝ), x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2 → 
    ∃ (a b : ℝ), 0 < b ∧ b < a ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (a^2 - b^2) / a^2 = (sin θ)^2) : 
  ∃ (a b : ℝ), 0 < b ∧ b < a ∧ 
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ 
      x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2) ∧
    (a^2 - b^2) / a^2 = (sin θ)^2 := by
  sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_l233_23333


namespace NUMINAMATH_CALUDE_function_properties_l233_23319

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_properties (a : ℝ) (h : f a (π / 3) = 0) :
  (∃ T > 0, ∀ x, f a (x + T) = f a x ∧ ∀ S, 0 < S → S < T → ∃ y, f a (y + S) ≠ f a y) ∧
  (∀ y ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a y ∧ f a y ≤ 2) ∧
  (∃ y₁ ∈ Set.Icc (π / 2) (3 * π / 2), f a y₁ = -1) ∧
  (∃ y₂ ∈ Set.Icc (π / 2) (3 * π / 2), f a y₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l233_23319


namespace NUMINAMATH_CALUDE_power_product_reciprocals_l233_23344

theorem power_product_reciprocals (n : ℕ) : (1 / 4 : ℝ) ^ n * 4 ^ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_reciprocals_l233_23344


namespace NUMINAMATH_CALUDE_prime_power_cube_plus_one_l233_23341

def is_solution (x y z : ℕ+) : Prop :=
  z.val.Prime ∧ z^(x.val) = y^3 + 1

theorem prime_power_cube_plus_one :
  ∀ x y z : ℕ+, is_solution x y z ↔ (x, y, z) = (1, 1, 2) ∨ (x, y, z) = (2, 2, 3) :=
sorry

end NUMINAMATH_CALUDE_prime_power_cube_plus_one_l233_23341


namespace NUMINAMATH_CALUDE_necessary_unique_letters_l233_23303

def word : String := "necessary"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem necessary_unique_letters :
  (uniqueLetters word).card = 7 := by sorry

end NUMINAMATH_CALUDE_necessary_unique_letters_l233_23303


namespace NUMINAMATH_CALUDE_trihedral_dihedral_planar_equality_l233_23309

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- Represents a dihedral angle -/
def DihedralAngle : Type := Real

/-- 
Given a trihedral angle, there exists a planar angle equal to 
the dihedral angle opposite to one of its plane angles.
-/
theorem trihedral_dihedral_planar_equality 
  (t : TrihedralAngle) : 
  ∃ (planar_angle : Real) (dihedral : DihedralAngle), 
    planar_angle = dihedral := by
  sorry


end NUMINAMATH_CALUDE_trihedral_dihedral_planar_equality_l233_23309


namespace NUMINAMATH_CALUDE_childrens_tickets_l233_23385

theorem childrens_tickets (adult_price child_price total_tickets total_cost : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_cost ∧
    child_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_childrens_tickets_l233_23385


namespace NUMINAMATH_CALUDE_reflected_light_equation_l233_23338

/-- Given points A, B, and P in a plane, and a line l passing through P parallel to AB,
    prove that the equation of the reflected light line from B to A via l is 11x + 27y + 74 = 0 -/
theorem reflected_light_equation (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (8, -6) →
  B = (2, 2) →
  P = (2, -3) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 4*x + 3*y + 1 = 0) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y - P.2 = k * (x - P.1)) →
  (∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ l ∧ 
    ((y₀ - B.2) / (x₀ - B.1) = -(x₀ - A.1) / (y₀ - A.2))) →
  ∃ (x y : ℝ), 11*x + 27*y + 74 = 0 ↔ 
    (y - A.2) / (x - A.1) = (A.2 - y₀) / (A.1 - x₀) :=
by sorry

end NUMINAMATH_CALUDE_reflected_light_equation_l233_23338


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_l233_23336

theorem quadratic_integer_solutions (p q : ℝ) : 
  p + q = 1998 ∧ 
  (∃ a b : ℤ, ∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →
  (p = 1998 ∧ q = 0) ∨ (p = -2002 ∧ q = 4000) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_l233_23336


namespace NUMINAMATH_CALUDE_candy_mixture_price_prove_candy_mixture_price_l233_23349

/-- Given two types of candies with equal amounts, priced at 2 and 3 rubles per kilogram respectively,
    the price of their mixture is 2.4 rubles per kilogram. -/
theorem candy_mixture_price : ℝ → Prop :=
  fun (s : ℝ) ↦
    let candy1_weight := s / 2
    let candy2_weight := s / 3
    let total_weight := candy1_weight + candy2_weight
    let total_cost := 2 * candy1_weight + 3 * candy2_weight
    let mixture_price := total_cost / total_weight
    mixture_price = 2.4

/-- Proof of the candy mixture price theorem -/
theorem prove_candy_mixture_price : ∃ s : ℝ, candy_mixture_price s := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_prove_candy_mixture_price_l233_23349


namespace NUMINAMATH_CALUDE_gcd_372_684_l233_23386

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l233_23386


namespace NUMINAMATH_CALUDE_worker_schedule_theorem_l233_23388

/-- Represents a worker's daily schedule and pay --/
structure WorkerSchedule where
  baseHours : ℝ
  basePay : ℝ
  bonusPay : ℝ
  bonusHours : ℝ
  bonusHourlyRate : ℝ

/-- Theorem stating the conditions and conclusion about the worker's schedule --/
theorem worker_schedule_theorem (w : WorkerSchedule) 
  (h1 : w.basePay = 80)
  (h2 : w.bonusPay = 20)
  (h3 : w.bonusHours = 2)
  (h4 : w.bonusHourlyRate = 10)
  (h5 : w.bonusHourlyRate * (w.baseHours + w.bonusHours) = w.basePay + w.bonusPay) :
  w.baseHours = 8 := by
  sorry

#check worker_schedule_theorem

end NUMINAMATH_CALUDE_worker_schedule_theorem_l233_23388


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l233_23310

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

/-- Theorem: The 12th term of the arithmetic sequence with a₁ = 1/4 and d = 1/2 is 23/4 -/
theorem twelfth_term_of_sequence :
  arithmetic_sequence (1/4) (1/2) 12 = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l233_23310


namespace NUMINAMATH_CALUDE_representatives_selection_theorem_l233_23357

/-- The number of ways to select representatives from a group of students -/
def select_representatives (total_students : ℕ) (num_representatives : ℕ) (restricted_student : ℕ) : ℕ :=
  (total_students - 1) * (total_students - 1) * (total_students - 2)

/-- Theorem stating the number of ways to select 3 representatives from 5 students,
    with one student restricted from being the Mathematics representative -/
theorem representatives_selection_theorem :
  select_representatives 5 3 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_theorem_l233_23357


namespace NUMINAMATH_CALUDE_mr_gates_classes_l233_23369

/-- Proves that given the conditions in the problem, Mr. Gates has 4 classes --/
theorem mr_gates_classes : 
  ∀ (buns_per_package : ℕ) 
    (packages_bought : ℕ) 
    (students_per_class : ℕ) 
    (buns_per_student : ℕ),
  buns_per_package = 8 →
  packages_bought = 30 →
  students_per_class = 30 →
  buns_per_student = 2 →
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 :=
by sorry

end NUMINAMATH_CALUDE_mr_gates_classes_l233_23369


namespace NUMINAMATH_CALUDE_ellipse_and_line_equation_l233_23339

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 3

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ (a : ℝ) : Prop := 4 * a = 12 * Real.sqrt 2

-- Define points P and Q on the ellipse
def point_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the midpoint of PQ
def midpoint_PQ (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = 1

-- Define the theorem
theorem ellipse_and_line_equation 
  (a b c : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : focal_distance c) 
  (h₄ : perimeter_ABF₂ a) 
  (h₅ : point_on_ellipse x₁ y₁ a b) 
  (h₆ : point_on_ellipse x₂ y₂ a b) 
  (h₇ : x₁ ≠ x₂ ∨ y₁ ≠ y₂) 
  (h₈ : midpoint_PQ x₁ y₁ x₂ y₂) : 
  (ellipse_C 3 (Real.sqrt 2) = ellipse_C 3 3) ∧ 
  (∀ (x y : ℝ), y = -(x - 2) + 1 ↔ x + y = 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equation_l233_23339


namespace NUMINAMATH_CALUDE_mistaken_division_correction_l233_23379

theorem mistaken_division_correction (n : ℕ) : 
  (n / 7 = 12 ∧ n % 7 = 4) → n / 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_correction_l233_23379


namespace NUMINAMATH_CALUDE_similar_triangles_height_l233_23325

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 15 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l233_23325


namespace NUMINAMATH_CALUDE_zenobius_had_more_descendants_l233_23373

/-- Calculates the total number of descendants for King Pafnutius -/
def pafnutius_descendants : ℕ :=
  2 + 60 * 2 + 20 * 1

/-- Calculates the total number of descendants for King Zenobius -/
def zenobius_descendants : ℕ :=
  4 + 35 * 3 + 35 * 1

/-- Proves that King Zenobius had more descendants than King Pafnutius -/
theorem zenobius_had_more_descendants :
  zenobius_descendants > pafnutius_descendants :=
by sorry

end NUMINAMATH_CALUDE_zenobius_had_more_descendants_l233_23373


namespace NUMINAMATH_CALUDE_incorrect_conversion_l233_23387

def base4_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 4 + (n % 10)

def base2_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 2 + (n % 10)

theorem incorrect_conversion :
  base4_to_decimal 31 ≠ base2_to_decimal 62 :=
sorry

end NUMINAMATH_CALUDE_incorrect_conversion_l233_23387


namespace NUMINAMATH_CALUDE_count_equals_60_l233_23354

/-- A function that generates all 5-digit numbers composed of digits 1, 2, 3, 4, and 5 without repetition -/
def generate_numbers : List Nat := sorry

/-- A function that checks if a number is greater than 23145 and less than 43521 -/
def is_in_range (n : Nat) : Bool := 23145 < n && n < 43521

/-- The count of numbers in the specified range -/
def count_in_range : Nat :=
  (generate_numbers.filter is_in_range).length

theorem count_equals_60 : count_in_range = 60 := by sorry

end NUMINAMATH_CALUDE_count_equals_60_l233_23354


namespace NUMINAMATH_CALUDE_count_eight_digit_integers_l233_23389

/-- The number of different 8-digit positive integers -/
def eight_digit_integers : ℕ :=
  9 * (10 ^ 7)

/-- Theorem: The number of different 8-digit positive integers is 90,000,000 -/
theorem count_eight_digit_integers :
  eight_digit_integers = 90000000 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_integers_l233_23389


namespace NUMINAMATH_CALUDE_remaining_cookies_l233_23368

theorem remaining_cookies (white_initial : ℕ) (black_initial : ℕ) : 
  white_initial = 80 →
  black_initial = white_initial + 50 →
  (white_initial - (3 * white_initial / 4)) + (black_initial / 2) = 85 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cookies_l233_23368


namespace NUMINAMATH_CALUDE_dave_baseball_cards_l233_23312

/-- Calculates the number of pages required to organize baseball cards in a binder -/
def pages_required (cards_per_page new_cards old_cards : ℕ) : ℕ :=
  (new_cards + old_cards + cards_per_page - 1) / cards_per_page

/-- Proves that Dave will use 2 pages to organize his baseball cards -/
theorem dave_baseball_cards : pages_required 8 3 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_baseball_cards_l233_23312


namespace NUMINAMATH_CALUDE_proposition_p_or_q_exclusive_l233_23380

theorem proposition_p_or_q_exclusive (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0) ∨
  (∃ x : ℝ, x^2 + (a - 3)*x + 1 = 0) ∧
  ¬((∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0) ∧
    (∃ x : ℝ, x^2 + (a - 3)*x + 1 = 0)) ↔
  a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_or_q_exclusive_l233_23380


namespace NUMINAMATH_CALUDE_steel_bar_length_l233_23335

/-- Given three types of steel bars A, B, and C with lengths x, y, and z respectively,
    prove that the total length of 1 bar of type A, 2 bars of type B, and 3 bars of type C
    is x + 2y + 3z, given the conditions. -/
theorem steel_bar_length (x y z : ℝ) 
  (h1 : 2 * x + y + 3 * z = 23) 
  (h2 : x + 4 * y + 5 * z = 36) : 
  x + 2 * y + 3 * z = (7 * x + 14 * y + 21 * z) / 7 := by
  sorry

end NUMINAMATH_CALUDE_steel_bar_length_l233_23335


namespace NUMINAMATH_CALUDE_zero_in_M_l233_23323

def M : Set ℝ := {x | x^2 - 3 ≤ 0}

theorem zero_in_M : (0 : ℝ) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l233_23323


namespace NUMINAMATH_CALUDE_no_such_function_l233_23392

theorem no_such_function : ¬∃ f : ℤ → ℤ, ∀ m n : ℤ, f (m + f n) = f m - n := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l233_23392


namespace NUMINAMATH_CALUDE_sally_earnings_l233_23377

/-- Sally's earnings per house, given her total earnings and number of houses cleaned -/
def earnings_per_house (total_earnings : ℕ) (houses_cleaned : ℕ) : ℚ :=
  (total_earnings : ℚ) / houses_cleaned

/-- Conversion factor from dozens to units -/
def dozens_to_units : ℕ := 12

theorem sally_earnings :
  let total_dozens : ℕ := 200
  let houses_cleaned : ℕ := 96
  earnings_per_house (total_dozens * dozens_to_units) houses_cleaned = 25 := by
sorry

end NUMINAMATH_CALUDE_sally_earnings_l233_23377


namespace NUMINAMATH_CALUDE_integral_x_plus_x_squared_plus_sin_x_l233_23326

theorem integral_x_plus_x_squared_plus_sin_x : 
  ∫ x in (-1 : ℝ)..1, (x + x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_plus_x_squared_plus_sin_x_l233_23326


namespace NUMINAMATH_CALUDE_debby_spent_14_tickets_l233_23324

/-- The number of tickets Debby spent on a hat -/
def hat_tickets : ℕ := 2

/-- The number of tickets Debby spent on a stuffed animal -/
def stuffed_animal_tickets : ℕ := 10

/-- The number of tickets Debby spent on a yoyo -/
def yoyo_tickets : ℕ := 2

/-- The total number of tickets Debby spent -/
def total_tickets : ℕ := hat_tickets + stuffed_animal_tickets + yoyo_tickets

theorem debby_spent_14_tickets : total_tickets = 14 := by
  sorry

end NUMINAMATH_CALUDE_debby_spent_14_tickets_l233_23324


namespace NUMINAMATH_CALUDE_power_of_power_l233_23327

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l233_23327


namespace NUMINAMATH_CALUDE_irrational_sqrt_6_l233_23381

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Theorem statement
theorem irrational_sqrt_6 :
  IsIrrational (Real.sqrt 6) ∧ 
  IsRational 3.14 ∧
  IsRational (-1/3) ∧
  IsRational (22/7) :=
sorry

end NUMINAMATH_CALUDE_irrational_sqrt_6_l233_23381


namespace NUMINAMATH_CALUDE_cafe_bill_difference_l233_23302

theorem cafe_bill_difference (amy_tip beth_tip : ℝ) 
  (amy_percent beth_percent : ℝ) : 
  amy_tip = 4 ∧ 
  beth_tip = 5 ∧ 
  amy_percent = 0.08 ∧ 
  beth_percent = 0.10 ∧ 
  amy_tip = amy_percent * (amy_tip / amy_percent) ∧
  beth_tip = beth_percent * (beth_tip / beth_percent) →
  (amy_tip / amy_percent) - (beth_tip / beth_percent) = 0 := by
sorry

end NUMINAMATH_CALUDE_cafe_bill_difference_l233_23302


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l233_23313

/-- Given two functions f and g, prove that f(g(f(3))) = 79 -/
theorem composite_function_evaluation :
  let f : ℝ → ℝ := λ x ↦ 2 * x + 5
  let g : ℝ → ℝ := λ x ↦ 3 * x + 4
  f (g (f 3)) = 79 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l233_23313


namespace NUMINAMATH_CALUDE_triangle_inequality_l233_23374

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l233_23374


namespace NUMINAMATH_CALUDE_division_relation_l233_23345

theorem division_relation (a b c : ℚ) 
  (h1 : a / b = 2) 
  (h2 : b / c = 3/4) : 
  c / a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_division_relation_l233_23345
