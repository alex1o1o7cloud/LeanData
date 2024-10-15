import Mathlib

namespace NUMINAMATH_CALUDE_general_equation_l1335_133598

theorem general_equation (n : ℤ) : 
  (n / (n - 4)) + ((8 - n) / ((8 - n) - 4)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_general_equation_l1335_133598


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l1335_133596

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a repeating decimal of the form 0.ẋyȧ -/
def RepeatingDecimal (x y : Digit) : ℚ :=
  (x.val * 100 + y.val * 10 + 3) / 999

/-- The theorem to be proved -/
theorem repeating_decimal_fraction (x y : Digit) (a : ℤ) 
  (h1 : x ≠ y)
  (h2 : RepeatingDecimal x y = a / 27) :
  a = 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l1335_133596


namespace NUMINAMATH_CALUDE_gordons_lighter_bag_weight_l1335_133543

/-- 
Given:
- Trace has 5 shopping bags
- Gordon has 2 shopping bags
- Trace's 5 bags weigh the same as Gordon's 2 bags
- One of Gordon's bags weighs 7 pounds
- Each of Trace's bags weighs 2 pounds

Prove that Gordon's lighter bag weighs 3 pounds.
-/
theorem gordons_lighter_bag_weight :
  ∀ (trace_bags gordon_bags : ℕ) 
    (trace_bag_weight gordon_heavy_bag_weight : ℝ)
    (total_trace_weight total_gordon_weight : ℝ),
  trace_bags = 5 →
  gordon_bags = 2 →
  trace_bag_weight = 2 →
  gordon_heavy_bag_weight = 7 →
  total_trace_weight = trace_bags * trace_bag_weight →
  total_gordon_weight = gordon_heavy_bag_weight + (total_trace_weight - gordon_heavy_bag_weight) →
  total_trace_weight = total_gordon_weight →
  (total_trace_weight - gordon_heavy_bag_weight) = 3 :=
by sorry

end NUMINAMATH_CALUDE_gordons_lighter_bag_weight_l1335_133543


namespace NUMINAMATH_CALUDE_distance_ratio_l1335_133539

-- Define the total distance and traveled distance
def total_distance : ℝ := 234
def traveled_distance : ℝ := 156

-- Define the theorem
theorem distance_ratio :
  let remaining_distance := total_distance - traveled_distance
  (traveled_distance / remaining_distance) = 2 := by
sorry

end NUMINAMATH_CALUDE_distance_ratio_l1335_133539


namespace NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l1335_133589

theorem one_fourth_divided_by_one_eighth : (1 / 4 : ℚ) / (1 / 8 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_divided_by_one_eighth_l1335_133589


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l1335_133533

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 * (29/13 : ℝ) - 1, 5], ![4 + (29/13 : ℝ), 9]]
  ¬(IsUnit (Matrix.det A)) := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l1335_133533


namespace NUMINAMATH_CALUDE_divisibility_check_l1335_133537

theorem divisibility_check (n : ℕ) : 
  n = 1493826 → 
  n % 3 = 0 ∧ 
  ¬(n % 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_check_l1335_133537


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l1335_133530

/-- Calculates the interval for systematic sampling. -/
def systematicSamplingInterval (totalPopulation sampleSize : ℕ) : ℕ :=
  totalPopulation / sampleSize

/-- Proves that for a population of 72 and sample size of 8, the systematic sampling interval is 9. -/
theorem systematic_sampling_interval_72_8 :
  systematicSamplingInterval 72 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_72_8_l1335_133530


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1335_133511

theorem abs_sum_inequality (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 6| > a) → a < 5 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1335_133511


namespace NUMINAMATH_CALUDE_ships_met_equals_journey_duration_atlantic_crossing_meets_seven_ships_l1335_133574

/-- Represents a steamship journey between two ports -/
structure Journey where
  departureDays : ℕ  -- Number of days since the first ship departed
  travelDays : ℕ     -- Number of days the journey takes

/-- The number of ships met during a journey -/
def shipsMetDuringJourney (j : Journey) : ℕ :=
  j.travelDays

/-- Theorem: The number of ships met during a journey is equal to the journey's duration -/
theorem ships_met_equals_journey_duration (j : Journey) :
  shipsMetDuringJourney j = j.travelDays :=
by sorry

/-- The specific journey described in the problem -/
def atlanticCrossing : Journey :=
  { departureDays := 1,  -- A ship departs every day
    travelDays := 7 }    -- The journey takes 7 days

/-- Theorem: A ship crossing the Atlantic meets 7 other ships -/
theorem atlantic_crossing_meets_seven_ships :
  shipsMetDuringJourney atlanticCrossing = 7 :=
by sorry

end NUMINAMATH_CALUDE_ships_met_equals_journey_duration_atlantic_crossing_meets_seven_ships_l1335_133574


namespace NUMINAMATH_CALUDE_square_difference_sum_l1335_133526

theorem square_difference_sum : 
  27^2 - 25^2 + 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 392 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l1335_133526


namespace NUMINAMATH_CALUDE_coin_probability_l1335_133548

theorem coin_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- p is a probability
  (p * (1 - p)^4 = 1/32) →  -- probability of HTTT = 0.03125
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_coin_probability_l1335_133548


namespace NUMINAMATH_CALUDE_pokemon_cards_distribution_l1335_133528

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 56) (h2 : num_friends = 4) (h3 : num_friends > 0) :
  total_cards / num_friends = 14 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_distribution_l1335_133528


namespace NUMINAMATH_CALUDE_kennel_arrangement_count_l1335_133588

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_chickens : ℕ := 6
def num_dogs : ℕ := 4
def num_cats : ℕ := 5

def total_arrangements : ℕ := 2 * factorial num_chickens * factorial num_dogs * factorial num_cats

theorem kennel_arrangement_count :
  total_arrangements = 4147200 :=
by sorry

end NUMINAMATH_CALUDE_kennel_arrangement_count_l1335_133588


namespace NUMINAMATH_CALUDE_number_property_l1335_133561

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ),
    0 < k ∧ 
    1 ≤ a ∧ a ≤ 9 ∧
    n = 12 * a ∧
    n % 10 ≠ 0

theorem number_property :
  ∀ (N : ℕ),
    (N % 10 ≠ 0) →
    (∃ (N' : ℕ), 
      (∃ (k : ℕ) (m : ℕ) (n : ℕ),
        N = m + 10^k * (N' / 10^k % 10) + 10^(k+1) * n ∧
        N' = m + 10^(k+1) * n ∧
        m < 10^k) ∧
      N = 6 * N') →
    is_valid_number N :=
sorry

end NUMINAMATH_CALUDE_number_property_l1335_133561


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l1335_133569

theorem unique_prime_with_remainder : ∃! m : ℕ,
  Prime m ∧ 30 < m ∧ m < 50 ∧ m % 12 = 7 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l1335_133569


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1335_133578

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1335_133578


namespace NUMINAMATH_CALUDE_circle_equation_l1335_133503

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The equation of a line ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem circle_equation (C : Circle) (L : Line) (P1 P2 : Point) :
  L.a = 2 ∧ L.b = 1 ∧ L.c = -1 →  -- Line equation: 2x + y - 1 = 0
  C.h * L.a + C.k * L.b + L.c = 0 →  -- Center is on the line
  (0 - C.h)^2 + (0 - C.k)^2 = C.r^2 →  -- Circle passes through origin
  (P1.x - C.h)^2 + (P1.y - C.k)^2 = C.r^2 →  -- Circle passes through P1
  P1.x = -1 ∧ P1.y = -5 →  -- P1 coordinates
  C.h = 2 ∧ C.k = -3 ∧ C.r^2 = 13 →  -- Circle equation coefficients
  ∀ x y : ℝ, (x - C.h)^2 + (y - C.k)^2 = C.r^2 ↔ (x - 2)^2 + (y + 3)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1335_133503


namespace NUMINAMATH_CALUDE_gcd_of_175_100_75_base_conversion_l1335_133577

-- Part 1: GCD of 175, 100, and 75
theorem gcd_of_175_100_75 : Nat.gcd 175 (Nat.gcd 100 75) = 25 := by sorry

-- Part 2: Base conversion
def base_6_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

def decimal_to_base_8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem base_conversion :
  (base_6_to_decimal [5, 1, 0, 1] = 227) ∧
  (decimal_to_base_8 227 = [3, 4, 3]) := by sorry

end NUMINAMATH_CALUDE_gcd_of_175_100_75_base_conversion_l1335_133577


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1335_133599

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the length of the real axis
def real_axis_length : ℝ := 4

-- Define the distance from foci to asymptote
def foci_to_asymptote_distance : ℝ := 1

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    real_axis_length = 4 ∧ 
    foci_to_asymptote_distance = 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1335_133599


namespace NUMINAMATH_CALUDE_nth_equation_specific_case_l1335_133582

theorem nth_equation (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 - (2 * n - 1) / (n * n)) = (n - 1) / n :=
by sorry

theorem specific_case : Real.sqrt (1 - 199 / 10000) = 99 / 100 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_specific_case_l1335_133582


namespace NUMINAMATH_CALUDE_percentage_problem_l1335_133583

theorem percentage_problem (x y : ℝ) (P : ℝ) :
  (0.1 * x = P / 100 * y) →  -- 10% of x equals P% of y
  (x / y = 2) →              -- The ratio of x to y is 2
  P = 20 :=                  -- The percentage of y is 20%
by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1335_133583


namespace NUMINAMATH_CALUDE_delay_and_wait_l1335_133535

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

def addMinutes (t : Time) (m : Nat) : Time := sorry

theorem delay_and_wait (start : Time) (delay : Nat) (wait : Nat) : 
  start.hours = 3 ∧ start.minutes = 0 → 
  delay = 30 → 
  wait = 2500 → 
  (addMinutes (addMinutes start delay) wait).hours = 21 ∧ 
  (addMinutes (addMinutes start delay) wait).minutes = 10 := by
  sorry

end NUMINAMATH_CALUDE_delay_and_wait_l1335_133535


namespace NUMINAMATH_CALUDE_polynomial_equality_l1335_133538

theorem polynomial_equality : 11^5 - 5 * 11^4 + 10 * 11^3 - 10 * 11^2 + 5 * 11 - 1 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1335_133538


namespace NUMINAMATH_CALUDE_hundredth_digit_of_17_over_99_l1335_133502

/-- The 100th digit after the decimal point in the decimal representation of 17/99 is 7 -/
theorem hundredth_digit_of_17_over_99 : ∃ (d : ℕ), d = 7 ∧ 
  (∃ (a b : ℕ) (s : List ℕ), 
    (17 : ℚ) / 99 = (a : ℚ) + (b : ℚ) / 10 + (s.foldr (λ x acc => acc / 10 + (x : ℚ) / 10) 0) ∧
    s.length = 99 ∧
    d = s.reverse.head!) :=
sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_17_over_99_l1335_133502


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1335_133551

-- Define the polynomial g(x)
def g (c d x : ℝ) : ℝ := c * x^3 - 7 * x^2 + d * x - 4

-- State the theorem
theorem polynomial_remainder_theorem (c d : ℝ) :
  (g c d 2 = -4) ∧ (g c d (-1) = -22) → c = 19/3 ∧ d = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1335_133551


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1335_133525

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2552
def time_day1 : ℕ := 6
def time_day2 : ℕ := 4

theorem average_speed_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_time : ℕ := time_day1 + time_day2
  (total_distance : ℚ) / total_time = 22 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1335_133525


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1335_133506

theorem negation_of_proposition (P : Prop) :
  (¬ (∃ x : ℝ, 2 * x + 1 ≤ 0)) ↔ (∀ x : ℝ, 2 * x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1335_133506


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1335_133518

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) ↔ (∀ x : ℝ, |x - 2| + |x - 4| > 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l1335_133518


namespace NUMINAMATH_CALUDE_triangle_circle_relation_l1335_133547

/-- For a triangle with circumcircle radius R, excircle radius p, and distance d between their centers, d^2 = R^2 + 2Rp. -/
theorem triangle_circle_relation (R p d : ℝ) : d^2 = R^2 + 2*R*p :=
sorry

end NUMINAMATH_CALUDE_triangle_circle_relation_l1335_133547


namespace NUMINAMATH_CALUDE_subset_implies_zero_intersection_of_A_and_B_l1335_133524

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | 2 < x ∧ x < 6}

-- Theorem 1: If {x | ax = 1} is a subset of any set, then a = 0
theorem subset_implies_zero (a : ℝ) (h : ∀ S : Set ℝ, {x | a * x = 1} ⊆ S) : a = 0 := by
  sorry

-- Theorem 2: A ∩ B = {x | 2 < x ∧ x < 4}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_zero_intersection_of_A_and_B_l1335_133524


namespace NUMINAMATH_CALUDE_scooter_profit_percentage_l1335_133523

theorem scooter_profit_percentage 
  (C : ℝ)  -- Cost of the scooter
  (h1 : C * 0.1 = 500)  -- 10% of cost spent on repairs
  (h2 : C + 1100 = C * 1.22)  -- Sold for a profit of $1100, which is 22% more than cost
  : (1100 / C) * 100 = 22 :=
by sorry

end NUMINAMATH_CALUDE_scooter_profit_percentage_l1335_133523


namespace NUMINAMATH_CALUDE_student_A_consecutive_days_probability_l1335_133549

/-- The number of days for the volunteer activity -/
def total_days : ℕ := 5

/-- The total number of students participating -/
def total_students : ℕ := 4

/-- The number of days student A participates -/
def student_A_days : ℕ := 2

/-- The number of days each other student participates -/
def other_student_days : ℕ := 1

/-- The probability that student A participates for two consecutive days -/
def consecutive_days_probability : ℚ := 2 / 5

/-- Theorem stating that the probability of student A participating for two consecutive days is 2/5 -/
theorem student_A_consecutive_days_probability :
  consecutive_days_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_student_A_consecutive_days_probability_l1335_133549


namespace NUMINAMATH_CALUDE_operation_not_equal_33_l1335_133522

/-- Given single digit positive integers a and b, where x = 1/5 a and z = 1/5 b,
    prove that (10a + b) - (10x + z) ≠ 33 -/
theorem operation_not_equal_33 (a b : ℕ) (x z : ℕ) 
  (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10)
  (hx : x = a / 5) (hz : z = b / 5)
  (hx_pos : 0 < x) (hz_pos : 0 < z) : 
  (10 * a + b) - (10 * x + z) ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_operation_not_equal_33_l1335_133522


namespace NUMINAMATH_CALUDE_right_triangle_area_l1335_133581

/-- The area of a right triangle with legs 20 and 21 is 210 -/
theorem right_triangle_area : 
  let a : ℝ := 20
  let b : ℝ := 21
  let area : ℝ := (1/2) * a * b
  area = 210 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1335_133581


namespace NUMINAMATH_CALUDE_prime_factors_of_2_pow_8_minus_1_l1335_133520

theorem prime_factors_of_2_pow_8_minus_1 :
  ∃ (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 2^8 - 1 ∧
    p + q + r = 25 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_2_pow_8_minus_1_l1335_133520


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l1335_133552

open Real

theorem angle_sum_theorem (x : ℝ) :
  (0 ≤ x ∧ x ≤ 2 * π) →
  (sin x ^ 5 - cos x ^ 5 = 1 / cos x - 1 / sin x) →
  ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 2 * π) ∧
             (sin y ^ 5 - cos y ^ 5 = 1 / cos y - 1 / sin y) ∧
             x + y = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l1335_133552


namespace NUMINAMATH_CALUDE_word_exists_l1335_133513

/-- Represents a word in the Russian language -/
structure RussianWord where
  word : String

/-- Represents a festive dance event -/
structure FestiveDanceEvent where
  name : String

/-- Represents a sport -/
inductive Sport
  | FigureSkating
  | RhythmicGymnastics

/-- Represents the Russian pension system -/
structure RussianPensionSystem where
  startYear : Nat
  calculationMethod : String

/-- The word we're looking for satisfies all conditions -/
def satisfiesAllConditions (w : RussianWord) (f : FestiveDanceEvent) (s : Sport) (p : RussianPensionSystem) : Prop :=
  (w.word.toLower = f.name.toLower) ∧ 
  (match s with
    | Sport.FigureSkating => true
    | Sport.RhythmicGymnastics => true) ∧
  (p.startYear = 2015 ∧ p.calculationMethod = w.word)

theorem word_exists : 
  ∃ (w : RussianWord) (f : FestiveDanceEvent) (s : Sport) (p : RussianPensionSystem), 
    satisfiesAllConditions w f s p :=
sorry

end NUMINAMATH_CALUDE_word_exists_l1335_133513


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l1335_133594

theorem divisibility_in_sequence (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l1335_133594


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1335_133586

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ a : ℝ, a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ a : ℝ, a ≥ 0 ∧ a^4 + a^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1335_133586


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1335_133563

/-- A quadratic function f(x) = ax^2 + bx satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) (ha : a ≠ 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

theorem quadratic_function_properties 
  (a b : ℝ) (ha : a ≠ 0) 
  (h1 : ∀ x, QuadraticFunction a b ha (x - 1) = QuadraticFunction a b ha (3 - x))
  (h2 : ∃! x, QuadraticFunction a b ha x = 2 * x) :
  (∀ x, QuadraticFunction a b ha x = -x^2 + 2*x) ∧ 
  (∀ t, (t : ℝ) > 0 → 
    (∀ x, x ∈ Set.Icc 0 t → QuadraticFunction a b ha x ≤ 
      (if t > 1 then 1 else -t^2 + 2*t))) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1335_133563


namespace NUMINAMATH_CALUDE_surface_generates_solid_l1335_133545

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- A cone formed by rotating a right-angled triangle -/
structure Cone where
  base_radius : ℝ
  height : ℝ

/-- Rotation of a right-angled triangle around one of its right-angle sides -/
def rotate_triangle (t : RightTriangle) : Cone :=
  { base_radius := t.base, height := t.height }

/-- Theorem: Rotating a right-angled triangle generates a solid (cone) -/
theorem surface_generates_solid (t : RightTriangle) :
  ∃ (c : Cone), c = rotate_triangle t :=
sorry

end NUMINAMATH_CALUDE_surface_generates_solid_l1335_133545


namespace NUMINAMATH_CALUDE_problem_solution_l1335_133573

theorem problem_solution (a b c d : ℚ) :
  (2*a + 2 = 3*b + 3) ∧
  (3*b + 3 = 4*c + 4) ∧
  (4*c + 4 = 5*d + 5) ∧
  (5*d + 5 = 2*a + 3*b + 4*c + 5*d + 6) →
  2*a + 3*b + 4*c + 5*d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1335_133573


namespace NUMINAMATH_CALUDE_soda_bottle_duration_l1335_133576

/-- Calculates the number of days a bottle of soda will last -/
def soda_duration (bottle_volume : ℚ) (daily_consumption : ℚ) : ℚ :=
  (bottle_volume * 1000) / daily_consumption

theorem soda_bottle_duration :
  let bottle_volume : ℚ := 2
  let daily_consumption : ℚ := 500
  soda_duration bottle_volume daily_consumption = 4 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_duration_l1335_133576


namespace NUMINAMATH_CALUDE_A_power_101_l1335_133557

def A : Matrix (Fin 3) (Fin 3) ℕ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = A ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_A_power_101_l1335_133557


namespace NUMINAMATH_CALUDE_complex_magnitude_l1335_133562

theorem complex_magnitude (a : ℝ) :
  (∃ (b : ℝ), (a + I) / (2 - I) = b * I) →
  Complex.abs (1/2 + (a + I) / (2 - I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1335_133562


namespace NUMINAMATH_CALUDE_complement_of_63_degrees_l1335_133509

theorem complement_of_63_degrees :
  let angle : ℝ := 63
  let complement (x : ℝ) : ℝ := 90 - x
  complement angle = 27 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_63_degrees_l1335_133509


namespace NUMINAMATH_CALUDE_lowest_divisible_by_primes_10_to_50_l1335_133514

def primes_10_to_50 : List Nat := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def is_divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ p ∈ list, n % p = 0

theorem lowest_divisible_by_primes_10_to_50 :
  ∃ (n : Nat), n > 0 ∧
  is_divisible_by_all n primes_10_to_50 ∧
  ∀ (m : Nat), m > 0 ∧ is_divisible_by_all m primes_10_to_50 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_lowest_divisible_by_primes_10_to_50_l1335_133514


namespace NUMINAMATH_CALUDE_min_distance_is_14000_l1335_133559

/-- Represents the problem of transporting and planting poles along a road -/
structure PolePlantingProblem where
  numPoles : ℕ
  startDistance : ℕ
  poleSpacing : ℕ
  maxPolesPerTrip : ℕ

/-- Calculates the minimum total distance traveled for a given pole planting problem -/
def minTotalDistance (p : PolePlantingProblem) : ℕ :=
  sorry

/-- The specific pole planting problem instance -/
def specificProblem : PolePlantingProblem :=
  { numPoles := 20
  , startDistance := 500
  , poleSpacing := 50
  , maxPolesPerTrip := 3 }

/-- Theorem stating that the minimum total distance for the specific problem is 14000 meters -/
theorem min_distance_is_14000 :
  minTotalDistance specificProblem = 14000 :=
sorry

end NUMINAMATH_CALUDE_min_distance_is_14000_l1335_133559


namespace NUMINAMATH_CALUDE_solution_triples_l1335_133550

theorem solution_triples : ∀ (a b c : ℝ),
  (a^2 + a*b + c = 0 ∧ b^2 + b*c + a = 0 ∧ c^2 + c*a + b = 0) →
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_triples_l1335_133550


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l1335_133570

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box. -/
def maxBlocksFit (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  let boxVolume := box.length * box.width * box.height
  let blockVolume := block.length * block.width * block.height
  boxVolume / blockVolume

/-- Theorem stating that the maximum number of 3×1×1 blocks that can fit in a 4×3×2 box is 8. -/
theorem max_blocks_in_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BlockDimensions.mk 3 1 1
  maxBlocksFit box block = 8 := by
  sorry

#eval maxBlocksFit (BoxDimensions.mk 4 3 2) (BlockDimensions.mk 3 1 1)

end NUMINAMATH_CALUDE_max_blocks_in_box_l1335_133570


namespace NUMINAMATH_CALUDE_minimal_fraction_sum_l1335_133564

theorem minimal_fraction_sum (a b : ℕ+) (h : (45:ℚ)/110 < (a:ℚ)/(b:ℚ) ∧ (a:ℚ)/(b:ℚ) < (50:ℚ)/110) :
  (∃ (c d : ℕ+), (45:ℚ)/110 < (c:ℚ)/(d:ℚ) ∧ (c:ℚ)/(d:ℚ) < (50:ℚ)/110 ∧ c+d ≤ a+b) →
  (3:ℚ)/7 = (a:ℚ)/(b:ℚ) :=
sorry

end NUMINAMATH_CALUDE_minimal_fraction_sum_l1335_133564


namespace NUMINAMATH_CALUDE_cubic_function_property_l1335_133519

theorem cubic_function_property (p q r s : ℝ) :
  (∀ x : ℝ, p * x^3 + q * x^2 + r * x + s = x * (x - 1) * (x + 2) / 6) →
  5 * p - 3 * q + 2 * r - s = 5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1335_133519


namespace NUMINAMATH_CALUDE_f_even_iff_a_zero_f_min_value_when_x_geq_a_l1335_133536

/-- Definition of the function f(x) -/
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

/-- Theorem about the evenness of f(x) -/
theorem f_even_iff_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

/-- Theorem about the minimum value of f(x) when x ≥ a -/
theorem f_min_value_when_x_geq_a (a : ℝ) :
  (∀ x ≥ a, f a x ≥ (if a ≤ -1/2 then 3/4 - a else a^2 + 1)) ∧
  (∃ x ≥ a, f a x = (if a ≤ -1/2 then 3/4 - a else a^2 + 1)) := by sorry

end NUMINAMATH_CALUDE_f_even_iff_a_zero_f_min_value_when_x_geq_a_l1335_133536


namespace NUMINAMATH_CALUDE_point_on_line_l1335_133560

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (p1 p2 p3 : Point) 
  (h1 : p1 = ⟨6, 12⟩) 
  (h2 : p2 = ⟨0, -6⟩) 
  (h3 : p3 = ⟨3, 3⟩) : 
  collinear p1 p2 p3 := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l1335_133560


namespace NUMINAMATH_CALUDE_correct_operation_l1335_133568

theorem correct_operation (a b : ℝ) : 5 * a * b - 3 * a * b = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1335_133568


namespace NUMINAMATH_CALUDE_emily_flight_remaining_time_l1335_133508

/-- Given a flight duration and a series of activities, calculate the remaining time -/
def remaining_flight_time (flight_duration : ℕ) (tv_episodes : ℕ) (tv_episode_duration : ℕ) 
  (sleep_duration : ℕ) (movies : ℕ) (movie_duration : ℕ) : ℕ :=
  flight_duration - (tv_episodes * tv_episode_duration + sleep_duration + movies * movie_duration)

/-- Theorem: Given Emily's flight and activities, prove that 45 minutes remain -/
theorem emily_flight_remaining_time : 
  remaining_flight_time 600 3 25 270 2 105 = 45 := by
  sorry

end NUMINAMATH_CALUDE_emily_flight_remaining_time_l1335_133508


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1335_133592

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a^2 + b^2 = c^2) : 
  (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1335_133592


namespace NUMINAMATH_CALUDE_unique_three_digit_odd_l1335_133575

/-- A function that returns true if a number is a three-digit odd number -/
def isThreeDigitOdd (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 1

/-- A function that returns the sum of squares of digits of a number -/
def sumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a * a + b * b + c * c

/-- The main theorem stating that 803 is the only three-digit odd number
    satisfying the given condition -/
theorem unique_three_digit_odd : ∀ n : ℕ, 
  isThreeDigitOdd n ∧ (n / 11 : ℚ) = (sumOfSquaresOfDigits n : ℚ) → n = 803 :=
by
  sorry

#check unique_three_digit_odd

end NUMINAMATH_CALUDE_unique_three_digit_odd_l1335_133575


namespace NUMINAMATH_CALUDE_factorial_ratio_l1335_133571

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 10 = 132 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1335_133571


namespace NUMINAMATH_CALUDE_jezebel_roses_l1335_133593

/-- The number of red roses Jezebel needs to buy -/
def num_red_roses : ℕ := sorry

/-- The cost of one red rose in dollars -/
def cost_red_rose : ℚ := 3/2

/-- The number of sunflowers Jezebel needs to buy -/
def num_sunflowers : ℕ := 3

/-- The cost of one sunflower in dollars -/
def cost_sunflower : ℚ := 3

/-- The total cost of all flowers in dollars -/
def total_cost : ℚ := 45

theorem jezebel_roses :
  num_red_roses * cost_red_rose + num_sunflowers * cost_sunflower = total_cost ∧
  num_red_roses = 24 := by sorry

end NUMINAMATH_CALUDE_jezebel_roses_l1335_133593


namespace NUMINAMATH_CALUDE_non_square_seq_2003_l1335_133597

/-- The sequence of positive integers with perfect squares removed -/
def non_square_seq : ℕ → ℕ := sorry

/-- The 2003rd term of the sequence of positive integers with perfect squares removed is 2048 -/
theorem non_square_seq_2003 : non_square_seq 2003 = 2048 := by sorry

end NUMINAMATH_CALUDE_non_square_seq_2003_l1335_133597


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1335_133504

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 
    (2 * π * r = 18 * π) → 
    (π * r^2 = 81 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1335_133504


namespace NUMINAMATH_CALUDE_max_sum_of_coefficients_l1335_133584

/-- Given a temperature function T(t) = a * sin(t) + b * cos(t) where t ∈ (0, +∞),
    a and b are positive real numbers, and the maximum temperature difference is 10°C,
    prove that the maximum value of a + b is 5√2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ t, t > 0 → t < Real.pi → ∃ T, T = a * Real.sin t + b * Real.cos t) →
  (∃ t₁ t₂, t₁ > 0 ∧ t₂ > 0 ∧ t₁ < Real.pi ∧ t₂ < Real.pi ∧
    a * Real.sin t₁ + b * Real.cos t₁ - (a * Real.sin t₂ + b * Real.cos t₂) = 10) →
  a + b ≤ 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_coefficients_l1335_133584


namespace NUMINAMATH_CALUDE_tank_height_problem_l1335_133556

/-- Given two right circular cylinders A and B, where A has a circumference of 6 meters,
    B has a height of 6 meters and a circumference of 10 meters, and A's capacity is 60% of B's capacity,
    prove that the height of A is 10 meters. -/
theorem tank_height_problem (h_A : ℝ) : 
  let r_A : ℝ := 3 / Real.pi
  let r_B : ℝ := 5 / Real.pi
  let volume_A : ℝ := Real.pi * r_A^2 * h_A
  let volume_B : ℝ := Real.pi * r_B^2 * 6
  volume_A = 0.6 * volume_B → h_A = 10 := by
  sorry

#check tank_height_problem

end NUMINAMATH_CALUDE_tank_height_problem_l1335_133556


namespace NUMINAMATH_CALUDE_not_divisible_by_power_of_five_l1335_133501

theorem not_divisible_by_power_of_five (n : ℕ+) (k : ℕ+) 
  (h : k < 5^n.val - 5^(n.val - 1)) : 
  ¬ (5^n.val ∣ 2^k.val - 1) :=
sorry

end NUMINAMATH_CALUDE_not_divisible_by_power_of_five_l1335_133501


namespace NUMINAMATH_CALUDE_folded_paper_sum_l1335_133554

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a folded piece of graph paper -/
structure FoldedPaper where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  h1 : p1 = ⟨3, 3⟩
  h2 : p2 = ⟨7, 1⟩
  h3 : p3 = ⟨9, 4⟩

/-- The theorem to be proven -/
theorem folded_paper_sum (paper : FoldedPaper) : paper.p4.x + paper.p4.y = 28/3 := by
  sorry


end NUMINAMATH_CALUDE_folded_paper_sum_l1335_133554


namespace NUMINAMATH_CALUDE_largest_divisible_number_l1335_133534

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem largest_divisible_number : 
  (∀ m : ℕ, 5 ≤ m ∧ m ≤ 10 → is_divisible 2520 m) ∧ 
  ¬(∀ m : ℕ, 5 ≤ m ∧ m ≤ 11 → is_divisible 2520 m) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_number_l1335_133534


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l1335_133527

def f (m : ℕ) : ℕ :=
  if m % 2 = 0 ∧ m > 0 then
    (List.range (m / 2)).foldl (λ acc i => acc * (2 * i + 2)) 1
  else
    0

theorem greatest_prime_factor_f_28 :
  (Nat.factors (f 28)).maximum? = some 13 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l1335_133527


namespace NUMINAMATH_CALUDE_ratio_to_percent_l1335_133585

theorem ratio_to_percent (a b : ℕ) (h : a = 15 ∧ b = 25) : 
  (a : ℝ) / b * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l1335_133585


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1335_133540

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {0, 3, 4, 5}

theorem intersection_of_P_and_Q : P ∩ Q = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1335_133540


namespace NUMINAMATH_CALUDE_deepak_age_l1335_133531

/-- Given the ratio of ages and future ages of Rahul and Sandeep, prove Deepak's present age --/
theorem deepak_age (r d s : ℕ) : 
  r / d = 4 / 3 →  -- ratio of Rahul to Deepak's age
  d / s = 1 / 2 →  -- ratio of Deepak to Sandeep's age
  r + 6 = 42 →     -- Rahul's age after 6 years
  s + 9 = 57 →     -- Sandeep's age after 9 years
  d = 27 :=        -- Deepak's present age
by sorry

end NUMINAMATH_CALUDE_deepak_age_l1335_133531


namespace NUMINAMATH_CALUDE_check_amount_l1335_133565

theorem check_amount (total_parts : ℕ) (expensive_parts : ℕ) (cheap_price : ℕ) (expensive_price : ℕ) : 
  total_parts = 59 → 
  expensive_parts = 40 → 
  cheap_price = 20 → 
  expensive_price = 50 → 
  (total_parts - expensive_parts) * cheap_price + expensive_parts * expensive_price = 2380 := by
sorry

end NUMINAMATH_CALUDE_check_amount_l1335_133565


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1335_133591

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2) 
  (h_equation : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1335_133591


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1335_133587

theorem quadratic_roots_problem (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1335_133587


namespace NUMINAMATH_CALUDE_temperature_equation_initial_temperature_temperature_increase_l1335_133546

/-- Represents the temperature in °C at a given time t in minutes -/
def temperature (t : ℝ) : ℝ := 7 * t + 30

theorem temperature_equation (t : ℝ) (h : t < 10) :
  temperature t = 7 * t + 30 :=
by sorry

theorem initial_temperature :
  temperature 0 = 30 :=
by sorry

theorem temperature_increase (t₁ t₂ : ℝ) (h₁ : t₁ < 10) (h₂ : t₂ < 10) (h₃ : t₁ < t₂) :
  temperature t₂ - temperature t₁ = 7 * (t₂ - t₁) :=
by sorry

end NUMINAMATH_CALUDE_temperature_equation_initial_temperature_temperature_increase_l1335_133546


namespace NUMINAMATH_CALUDE_white_squares_20th_row_l1335_133507

/-- The total number of squares in the nth row of the modified "stair-step" figure -/
def totalSquares (n : ℕ) : ℕ := 3 * n

/-- The number of white squares in the nth row of the modified "stair-step" figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n) / 2

theorem white_squares_20th_row :
  whiteSquares 20 = 30 := by sorry

end NUMINAMATH_CALUDE_white_squares_20th_row_l1335_133507


namespace NUMINAMATH_CALUDE_number_division_problem_l1335_133505

theorem number_division_problem (x : ℝ) : (x / 2.5) / 3.1 + 3.1 = 8.9 → x = 44.95 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1335_133505


namespace NUMINAMATH_CALUDE_rectangular_room_shorter_side_l1335_133516

/-- Given a rectangular room with perimeter 50 feet and area 126 square feet,
    prove that the length of the shorter side is 9 feet. -/
theorem rectangular_room_shorter_side
  (perimeter : ℝ)
  (area : ℝ)
  (h_perimeter : perimeter = 50)
  (h_area : area = 126) :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length ≥ width ∧
    2 * (length + width) = perimeter ∧
    length * width = area ∧
    width = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_room_shorter_side_l1335_133516


namespace NUMINAMATH_CALUDE_product_sum_relation_l1335_133515

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 10 → b = 9 → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l1335_133515


namespace NUMINAMATH_CALUDE_find_Z_l1335_133500

theorem find_Z : ∃ Z : ℝ, (100 + 20 / Z) * Z = 9020 ∧ Z = 90 := by
  sorry

end NUMINAMATH_CALUDE_find_Z_l1335_133500


namespace NUMINAMATH_CALUDE_bamboo_sections_volume_l1335_133544

theorem bamboo_sections_volume (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →
  a 1 + a 2 + a 3 = 3.9 →
  a 6 + a 7 + a 8 + a 9 = 3 →
  a 4 + a 5 = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_sections_volume_l1335_133544


namespace NUMINAMATH_CALUDE_hockey_league_games_l1335_133595

theorem hockey_league_games (n : ℕ) (total_games : ℕ) 
  (hn : n = 17) (htotal : total_games = 1360) : 
  (total_games * 2) / (n * (n - 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1335_133595


namespace NUMINAMATH_CALUDE_floor_negative_fraction_l1335_133555

theorem floor_negative_fraction : ⌊(-19 : ℝ) / 3⌋ = -7 := by sorry

end NUMINAMATH_CALUDE_floor_negative_fraction_l1335_133555


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l1335_133567

/-- The height of a tree after n years, given its initial height and growth factor -/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * growth_factor ^ n

/-- Theorem: If a tree triples its height every year and reaches 81 feet after 4 years,
    then its height after 2 years is 9 feet -/
theorem tree_height_after_two_years
  (h : ∃ initial_height : ℝ, tree_height initial_height 3 4 = 81) :
  ∃ initial_height : ℝ, tree_height initial_height 3 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l1335_133567


namespace NUMINAMATH_CALUDE_sum_of_roots_l1335_133590

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1335_133590


namespace NUMINAMATH_CALUDE_additional_apples_needed_l1335_133553

def apples_needed (current_apples : ℕ) (people : ℕ) (apples_per_person : ℕ) : ℕ :=
  if people * apples_per_person ≤ current_apples then
    0
  else
    people * apples_per_person - current_apples

theorem additional_apples_needed :
  apples_needed 68 14 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_apples_needed_l1335_133553


namespace NUMINAMATH_CALUDE_sector_angle_measure_l1335_133579

/-- Given a sector with radius 2 cm and area 4 cm², 
    the radian measure of its central angle is 2. -/
theorem sector_angle_measure (r : ℝ) (S : ℝ) (θ : ℝ) : 
  r = 2 →  -- radius is 2 cm
  S = 4 →  -- area is 4 cm²
  S = 1/2 * r^2 * θ →  -- formula for sector area
  θ = 2 :=  -- central angle is 2 radians
by sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l1335_133579


namespace NUMINAMATH_CALUDE_total_miles_jogged_l1335_133572

/-- The number of miles a person jogs per day on weekdays -/
def miles_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 3

/-- Theorem: A person who jogs 5 miles per day on weekdays will run 75 miles over three weeks -/
theorem total_miles_jogged : 
  miles_per_day * weekdays_per_week * num_weeks = 75 := by sorry

end NUMINAMATH_CALUDE_total_miles_jogged_l1335_133572


namespace NUMINAMATH_CALUDE_ratio_equivalences_l1335_133558

theorem ratio_equivalences (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a * d = b * c) : 
  (a / b = c / d) ∧ (b / a = d / c) ∧ (a / c = b / d) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalences_l1335_133558


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l1335_133510

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Murtha's pebble collection over 20 days -/
theorem murtha_pebble_collection : sum_of_first_n 20 = 210 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l1335_133510


namespace NUMINAMATH_CALUDE_square_side_length_l1335_133566

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 144 → side * side = area → side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1335_133566


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1335_133512

def U : Set Nat := {0, 1, 3, 5, 6, 8}
def A : Set Nat := {1, 5, 8}
def B : Set Nat := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1335_133512


namespace NUMINAMATH_CALUDE_parallel_lines_coefficient_l1335_133541

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) → m1 = m2

/-- The slope of a line ax + by + c = 0 is -a/b when b ≠ 0 -/
axiom slope_of_line {a b c : ℝ} (h : b ≠ 0) :
  ∃ m : ℝ, m = -a/b ∧ ∀ x y : ℝ, a*x + b*y + c = 0 ↔ y = m*x + (-c/b)

theorem parallel_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, a*x + 2*y + 2 = 0 ↔ 3*x - y - 2 = 0) → a = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_coefficient_l1335_133541


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1335_133517

/-- Two lines ax + 2y = 3 and x + (a-1)y = 1 are parallel -/
def are_parallel (a : ℝ) : Prop :=
  a = 3 ∨ a = -1

/-- a = 2 is a sufficient condition for parallelism -/
def is_sufficient : Prop :=
  ∀ a : ℝ, a = 2 → are_parallel a

/-- a = 2 is a necessary condition for parallelism -/
def is_necessary : Prop :=
  ∀ a : ℝ, are_parallel a → a = 2

theorem not_sufficient_nor_necessary : ¬is_sufficient ∧ ¬is_necessary :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1335_133517


namespace NUMINAMATH_CALUDE_fahrenheit_diff_is_18_l1335_133532

-- Define the conversion function from Celsius to Fahrenheit
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperature difference in Celsius
def celsius_diff : ℝ := 10

-- Theorem statement
theorem fahrenheit_diff_is_18 :
  celsius_to_fahrenheit (C + celsius_diff) - celsius_to_fahrenheit C = 18 :=
by sorry

end NUMINAMATH_CALUDE_fahrenheit_diff_is_18_l1335_133532


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l1335_133521

theorem function_satisfies_conditions (m n : ℕ) :
  let f : ℕ → ℕ → ℕ := λ m n => m * n
  (m ≥ 1 ∧ n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ k : ℕ, f k 0 = 0 ∧ f 0 k = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l1335_133521


namespace NUMINAMATH_CALUDE_not_divisible_5n_minus_1_by_4n_minus_1_l1335_133580

theorem not_divisible_5n_minus_1_by_4n_minus_1 (n : ℕ) :
  ¬ (5^n - 1 ∣ 4^n - 1) := by sorry

end NUMINAMATH_CALUDE_not_divisible_5n_minus_1_by_4n_minus_1_l1335_133580


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1335_133529

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1335_133529


namespace NUMINAMATH_CALUDE_constant_b_proof_l1335_133542

theorem constant_b_proof (a b c : ℝ) : 
  (∀ x : ℝ, (3 * x^2 - 2 * x + 4) * (a * x^2 + b * x + c) = 
    6 * x^4 - 5 * x^3 + 11 * x^2 - 8 * x + 16) → 
  b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_constant_b_proof_l1335_133542
