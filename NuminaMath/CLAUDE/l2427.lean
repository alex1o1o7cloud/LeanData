import Mathlib

namespace NUMINAMATH_CALUDE_maria_journey_distance_l2427_242756

/-- A journey with two stops and a final leg -/
structure Journey where
  total_distance : ℝ
  first_stop : ℝ
  second_stop : ℝ
  final_leg : ℝ

/-- The conditions of Maria's journey -/
def maria_journey (j : Journey) : Prop :=
  j.first_stop = j.total_distance / 2 ∧
  j.second_stop = (j.total_distance - j.first_stop) / 4 ∧
  j.final_leg = 135 ∧
  j.total_distance = j.first_stop + j.second_stop + j.final_leg

/-- Theorem stating that Maria's journey has a total distance of 360 miles -/
theorem maria_journey_distance :
  ∃ j : Journey, maria_journey j ∧ j.total_distance = 360 :=
sorry

end NUMINAMATH_CALUDE_maria_journey_distance_l2427_242756


namespace NUMINAMATH_CALUDE_triangle_side_length_l2427_242782

theorem triangle_side_length (a b : ℝ) (A B : ℝ) : 
  a = 4 →
  A = π / 3 →  -- 60° in radians
  B = π / 4 →  -- 45° in radians
  b = (4 * Real.sqrt 6) / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2427_242782


namespace NUMINAMATH_CALUDE_triangle_inequality_l2427_242722

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (h1 : C ≥ π / 3) :
  let s := (a + b + c) / 2
  (a + b) * (1 / a + 1 / b + 1 / c) ≥ 4 + 1 / Real.sin (C / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2427_242722


namespace NUMINAMATH_CALUDE_probability_sum_nine_l2427_242749

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of ways to roll a sum of 9 with three dice -/
def favorableOutcomes : ℕ := 25

/-- The probability of rolling a sum of 9 with three standard six-faced dice -/
theorem probability_sum_nine :
  (favorableOutcomes : ℚ) / totalOutcomes = 25 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_sum_nine_l2427_242749


namespace NUMINAMATH_CALUDE_sort_three_integers_correct_l2427_242711

/-- Algorithm to sort three positive integers in descending order -/
def sort_three_integers (a b c : ℕ+) : ℕ+ × ℕ+ × ℕ+ :=
  let step2 := if a ≤ b then (b, a, c) else (a, b, c)
  let step3 := let (x, y, z) := step2
                if x ≤ z then (z, y, x) else (x, y, z)
  let step4 := let (x, y, z) := step3
                if y ≤ z then (x, z, y) else (x, y, z)
  step4

/-- Theorem stating that the sorting algorithm produces a descending order result -/
theorem sort_three_integers_correct (a b c : ℕ+) :
  let (x, y, z) := sort_three_integers a b c
  x ≥ y ∧ y ≥ z :=
by
  sorry

end NUMINAMATH_CALUDE_sort_three_integers_correct_l2427_242711


namespace NUMINAMATH_CALUDE_principal_calculation_l2427_242751

/-- Calculates the principal amount given two interest rates, time period, and interest difference --/
def calculate_principal (rate1 rate2 : ℚ) (time : ℕ) (interest_diff : ℚ) : ℚ :=
  interest_diff / (rate1 - rate2) * 100 / time

/-- Theorem stating that the calculated principal is approximately 7142.86 --/
theorem principal_calculation :
  let rate1 : ℚ := 22
  let rate2 : ℚ := 15
  let time : ℕ := 5
  let interest_diff : ℚ := 2500
  let principal := calculate_principal rate1 rate2 time interest_diff
  abs (principal - 7142.86) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2427_242751


namespace NUMINAMATH_CALUDE_minimum_coins_l2427_242758

def nickel : ℚ := 5 / 100
def dime : ℚ := 10 / 100
def quarter : ℚ := 25 / 100
def half_dollar : ℚ := 50 / 100

def total_amount : ℚ := 3

theorem minimum_coins (n d q h : ℕ) : 
  n ≥ 1 → d ≥ 1 → q ≥ 1 → h ≥ 1 →
  n * nickel + d * dime + q * quarter + h * half_dollar = total_amount →
  n + d + q + h ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_minimum_coins_l2427_242758


namespace NUMINAMATH_CALUDE_angela_puzzle_palace_spending_l2427_242781

/-- The amount of money Angela got to spend at Puzzle Palace -/
def total_amount : ℕ := sorry

/-- The amount of money Angela spent at Puzzle Palace -/
def amount_spent : ℕ := 78

/-- The amount of money Angela had left after shopping -/
def amount_left : ℕ := 12

/-- Theorem stating that the total amount Angela got to spend at Puzzle Palace is $90 -/
theorem angela_puzzle_palace_spending :
  total_amount = amount_spent + amount_left :=
sorry

end NUMINAMATH_CALUDE_angela_puzzle_palace_spending_l2427_242781


namespace NUMINAMATH_CALUDE_pats_picnic_dessert_l2427_242739

/-- Pat's picnic dessert problem -/
theorem pats_picnic_dessert (cookies : ℕ) (candy : ℕ) (family_size : ℕ) (dessert_per_person : ℕ) 
  (h1 : cookies = 42)
  (h2 : candy = 63)
  (h3 : family_size = 7)
  (h4 : dessert_per_person = 18) :
  family_size * dessert_per_person - (cookies + candy) = 21 := by
  sorry

end NUMINAMATH_CALUDE_pats_picnic_dessert_l2427_242739


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2427_242785

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2427_242785


namespace NUMINAMATH_CALUDE_existence_of_close_multiple_l2427_242715

theorem existence_of_close_multiple (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * a - m| ≤ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_close_multiple_l2427_242715


namespace NUMINAMATH_CALUDE_sampling_suitable_for_yangtze_fish_yangtze_fish_sampling_only_correct_option_l2427_242774

-- Define the types of survey methods
inductive SurveyMethod
| Census
| Sampling

-- Define the scenarios
inductive Scenario
| ShellKillingRadius
| StudentHeight
| CityAirQuality
| YangtzeRiverFish

-- Define a function that determines the suitability of a survey method for a given scenario
def isSuitable (method : SurveyMethod) (scenario : Scenario) : Prop :=
  match scenario with
  | Scenario.ShellKillingRadius => method = SurveyMethod.Sampling
  | Scenario.StudentHeight => method = SurveyMethod.Census
  | Scenario.CityAirQuality => method = SurveyMethod.Sampling
  | Scenario.YangtzeRiverFish => method = SurveyMethod.Sampling

-- Theorem stating that sampling is suitable for the Yangtze River fish scenario
theorem sampling_suitable_for_yangtze_fish :
  isSuitable SurveyMethod.Sampling Scenario.YangtzeRiverFish :=
by sorry

-- Theorem stating that sampling for Yangtze River fish is the only correct option among the given scenarios
theorem yangtze_fish_sampling_only_correct_option :
  ∀ (scenario : Scenario) (method : SurveyMethod),
    (scenario = Scenario.YangtzeRiverFish ∧ method = SurveyMethod.Sampling) ↔
    (isSuitable method scenario ∧
     ((scenario = Scenario.ShellKillingRadius ∧ method = SurveyMethod.Census) ∨
      (scenario = Scenario.StudentHeight ∧ method = SurveyMethod.Sampling) ∨
      (scenario = Scenario.CityAirQuality ∧ method = SurveyMethod.Census) ∨
      (scenario = Scenario.YangtzeRiverFish ∧ method = SurveyMethod.Sampling))) :=
by sorry

end NUMINAMATH_CALUDE_sampling_suitable_for_yangtze_fish_yangtze_fish_sampling_only_correct_option_l2427_242774


namespace NUMINAMATH_CALUDE_circle_inequality_l2427_242730

/-- Given three circles with centers P, Q, R and radii p, q, r respectively,
    where p > q > r, prove that p + q + r ≠ dist P Q + dist Q R -/
theorem circle_inequality (P Q R : EuclideanSpace ℝ (Fin 2))
    (p q r : ℝ) (hp : p > q) (hq : q > r) :
    p + q + r ≠ dist P Q + dist Q R := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l2427_242730


namespace NUMINAMATH_CALUDE_negative_885_degrees_conversion_l2427_242793

theorem negative_885_degrees_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    -885 * (π / 180) = 2 * k * π + α ∧
    0 ≤ α ∧ α ≤ 2 * π ∧
    k = -6 ∧ α = 13 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_negative_885_degrees_conversion_l2427_242793


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_sum_l2427_242773

/-- The sum of arithmetic sequence with 5 terms, starting from x and with common difference 3 -/
def sequence_sum (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9) + (x + 12)

/-- A natural number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem smallest_x_for_perfect_cube_sum : 
  (∀ x : ℕ, x > 0 ∧ x < 19 → ¬(is_perfect_cube (sequence_sum x))) ∧ 
  (is_perfect_cube (sequence_sum 19)) := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_sum_l2427_242773


namespace NUMINAMATH_CALUDE_process_result_l2427_242753

def process (x : ℕ) : ℕ := 3 * (2 * x + 9)

theorem process_result : process 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_process_result_l2427_242753


namespace NUMINAMATH_CALUDE_total_strings_is_40_l2427_242763

/-- The number of strings on all instruments in Francis' family -/
def total_strings : ℕ :=
  let ukulele_count : ℕ := 2
  let guitar_count : ℕ := 4
  let violin_count : ℕ := 2
  let strings_per_ukulele : ℕ := 4
  let strings_per_guitar : ℕ := 6
  let strings_per_violin : ℕ := 4
  ukulele_count * strings_per_ukulele +
  guitar_count * strings_per_guitar +
  violin_count * strings_per_violin

theorem total_strings_is_40 : total_strings = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_strings_is_40_l2427_242763


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_three_l2427_242704

theorem sum_of_fractions_equals_three (a b c x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_three_l2427_242704


namespace NUMINAMATH_CALUDE_locus_of_center_l2427_242712

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  x^2 + y^2 - (2*a^2 - 4)*x - 4*a^2*y + 5*a^4 - 4 = 0

-- Define the locus equation
def locus_equation (x y : ℝ) : Prop :=
  2*x - y + 4 = 0

-- Define the x-coordinate range
def x_range (x : ℝ) : Prop :=
  -2 ≤ x ∧ x < 0

-- Theorem statement
theorem locus_of_center :
  ∀ a x y : ℝ, circle_C a x y → 
  ∃ h k : ℝ, (locus_equation h k ∧ x_range h) ∧
  (∀ x' y' : ℝ, locus_equation x' y' ∧ x_range x' → 
   ∃ a' : ℝ, circle_C a' x' y') :=
sorry

end NUMINAMATH_CALUDE_locus_of_center_l2427_242712


namespace NUMINAMATH_CALUDE_abs_sum_inequality_range_l2427_242726

theorem abs_sum_inequality_range :
  {x : ℝ | |x + 1| + |x| < 2} = Set.Ioo (-3/2 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_range_l2427_242726


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2427_242791

/-- The function f(x) = a^(x-1) + 2 passes through the point (1, 3) for any a > 0 and a ≠ 1 -/
theorem function_passes_through_point (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2427_242791


namespace NUMINAMATH_CALUDE_salmon_migration_l2427_242789

theorem salmon_migration (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261) 
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_l2427_242789


namespace NUMINAMATH_CALUDE_sqrt_four_ninths_l2427_242705

theorem sqrt_four_ninths : Real.sqrt (4 / 9) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_ninths_l2427_242705


namespace NUMINAMATH_CALUDE_variance_of_literary_works_l2427_242747

def literary_works : List ℕ := [6, 9, 5, 8, 10, 4]

def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def variance (data : List ℕ) : ℚ :=
  let μ := mean data
  (data.map (fun x => ((x : ℚ) - μ) ^ 2)).sum / data.length

theorem variance_of_literary_works : variance literary_works = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_literary_works_l2427_242747


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2427_242777

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_odd : ∀ x, f a b c (-x) = -f a b c x)
  (h_f1 : f a b c 1 = b + c)
  (h_f2 : f a b c 2 = 4 * a + 2 * b + c) :
  (a = 2 ∧ b = -3 ∧ c = 0) ∧
  (∀ x, x > 0 → ∀ y, y > x → f a b c y < f a b c x) ∧
  (∃ m, m = 2 ∧ ∀ x, x > 0 → f a b c x ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2427_242777


namespace NUMINAMATH_CALUDE_counterexample_twelve_l2427_242752

theorem counterexample_twelve : ∃ n : ℕ, 
  ¬(Nat.Prime n) ∧ (n = 12) ∧ ¬(Nat.Prime (n - 1) ∧ Nat.Prime (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_twelve_l2427_242752


namespace NUMINAMATH_CALUDE_price_change_l2427_242736

theorem price_change (q r : ℝ) (original_price : ℝ) :
  (original_price * (1 + q / 100) * (1 - r / 100) = 1) →
  (original_price = 1 / ((1 + q / 100) * (1 - r / 100))) :=
by sorry

end NUMINAMATH_CALUDE_price_change_l2427_242736


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2427_242702

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- The sum of the numbers in a pair -/
def pairSum (p : IntPair) : ℕ :=
  p.first + p.second

/-- The number of pairs before the nth group -/
def pairsBeforeGroup (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 60th pair is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  nthPair 60 = IntPair.mk 5 7 :=
sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l2427_242702


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_sixteenth_l2427_242731

theorem product_of_sines_equals_one_sixteenth :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_sixteenth_l2427_242731


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_is_one_tenth_l2427_242740

/-- A line passing through (10, 0) intersecting y = x^2 -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (10, 0) -/
  line_eq : ∀ x y : ℝ, y = k * (x - 10)
  /-- The line intersects y = x^2 at two distinct points -/
  intersects_parabola : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = k * (x₁ - 10) ∧ x₂^2 = k * (x₂ - 10)

/-- The sum of reciprocals of intersection x-coordinates is 1/10 -/
theorem sum_of_reciprocals_is_one_tenth (L : IntersectingLine) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 = L.k * (x₁ - 10) ∧ 
    x₂^2 = L.k * (x₂ - 10) ∧
    1 / x₁ + 1 / x₂ = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_is_one_tenth_l2427_242740


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2427_242796

/-- Given an arithmetic sequence {aₙ}, prove that S₂₀₁₀ = 1005 under the given conditions -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * (a 1 + a n) / 2) → -- Definition of Sₙ
  (∃ O A B C : ℝ × ℝ, 
    B - O = a 1005 • (A - O) + a 1006 • (C - O) ∧ -- Vector equation
    ∃ t : ℝ, B = t • A + (1 - t) • C ∧ -- Collinearity condition
    t ≠ 0 ∧ t ≠ 1) → -- Line doesn't pass through O
  S 2010 = 1005 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2427_242796


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l2427_242760

/-- Represents the minimum bailing rate problem --/
theorem minimum_bailing_rate 
  (distance_to_shore : ℝ) 
  (leaking_rate : ℝ) 
  (max_water_tolerance : ℝ) 
  (boat_speed : ℝ) 
  (h1 : distance_to_shore = 2) 
  (h2 : leaking_rate = 8) 
  (h3 : max_water_tolerance = 50) 
  (h4 : boat_speed = 3) :
  ∃ (bailing_rate : ℝ), 
    bailing_rate ≥ 7 ∧ 
    bailing_rate < 8 ∧
    (leaking_rate - bailing_rate) * (distance_to_shore / boat_speed * 60) ≤ max_water_tolerance :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l2427_242760


namespace NUMINAMATH_CALUDE_min_value_theorem_l2427_242779

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 45) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 45 → 1/x + 4/y ≥ 1/5) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 45 ∧ 1/x + 4/y = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2427_242779


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2427_242772

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℚ), 
    F > 0 ∧ 
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = -1 ∧ B = -3 ∧ C = 1 ∧ D = 2/3 ∧ E = 165 ∧ F = 17 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2427_242772


namespace NUMINAMATH_CALUDE_marble_selection_probability_l2427_242719

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 4

/-- The probability of selecting exactly 1 red, 2 blue, and 1 green marble -/
def probability : ℚ := 3 / 14

theorem marble_selection_probability : 
  (Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 : ℚ) / 
  (Nat.choose total_marbles selected_marbles) = probability := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l2427_242719


namespace NUMINAMATH_CALUDE_daily_profit_at_35_unique_profit_600_no_profit_900_l2427_242738

/-- The daily profit function for a product -/
def P (x : ℝ) : ℝ := (x - 30) * (-2 * x + 140)

/-- The purchase price of the product -/
def purchase_price : ℝ := 30

/-- The lower bound of the selling price -/
def lower_bound : ℝ := 30

/-- The upper bound of the selling price -/
def upper_bound : ℝ := 55

theorem daily_profit_at_35 :
  P 35 = 350 := by sorry

theorem unique_profit_600 :
  ∃! x : ℝ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ P x = 600 ∧ x = 40 := by sorry

theorem no_profit_900 :
  ¬ ∃ x : ℝ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ P x = 900 := by sorry

end NUMINAMATH_CALUDE_daily_profit_at_35_unique_profit_600_no_profit_900_l2427_242738


namespace NUMINAMATH_CALUDE_divisibility_of_p_l2427_242770

theorem divisibility_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 40)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 110 < Nat.gcd s p ∧ Nat.gcd s p < 150) :
  11 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_l2427_242770


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l2427_242765

theorem complex_magnitude_fourth_power : 
  Complex.abs ((7/5 : ℂ) + (24/5 : ℂ) * Complex.I) ^ 4 = 625 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l2427_242765


namespace NUMINAMATH_CALUDE_number_subtraction_l2427_242721

theorem number_subtraction (x : ℤ) : x + 30 = 55 → x - 23 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_l2427_242721


namespace NUMINAMATH_CALUDE_propositions_proof_l2427_242784

theorem propositions_proof :
  (∀ (a b c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ a * c ≤ b * d) ∧
  (∃ (a b c : ℝ), b > a ∧ a > 0 ∧ c > 0 ∧ a / b ≤ (a + c) / (b + c)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_proof_l2427_242784


namespace NUMINAMATH_CALUDE_sqrt_8_same_type_as_sqrt_2_l2427_242788

-- Define what it means for two square roots to be of the same type
def same_type (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b

-- State the theorem
theorem sqrt_8_same_type_as_sqrt_2 :
  same_type (Real.sqrt 8) (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_same_type_as_sqrt_2_l2427_242788


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l2427_242708

theorem min_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, a * x + b * y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) →
  a + b ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l2427_242708


namespace NUMINAMATH_CALUDE_water_bucket_problem_l2427_242769

theorem water_bucket_problem (bucket3 bucket5 bucket6 : ℕ) 
  (h1 : bucket3 = 3)
  (h2 : bucket5 = 5)
  (h3 : bucket6 = 6) :
  bucket6 - (bucket5 - bucket3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l2427_242769


namespace NUMINAMATH_CALUDE_last_two_digits_factorial_sum_l2427_242761

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_factorial_sum :
  last_two_digits (sum_factorials 15) = last_two_digits (sum_factorials 9) :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_factorial_sum_l2427_242761


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l2427_242707

theorem min_draws_for_even_product (S : Finset ℕ) : 
  S = Finset.range 16 →
  (∃ n : ℕ, n ∈ S ∧ Even n) →
  (∀ T ⊆ S, T.card = 9 → ∃ m ∈ T, Even m) ∧
  (∃ U ⊆ S, U.card = 8 ∧ ∀ k ∈ U, ¬Even k) :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l2427_242707


namespace NUMINAMATH_CALUDE_triangle_problem_l2427_242720

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) → -- Angles are in (0, π)
  (a > 0) ∧ (b > 0) ∧ (c > 0) → -- Sides are positive
  (sin A / sin C = a / c) ∧ (sin B / sin C = b / c) → -- Law of sines
  (cos C + c / b * cos B = 2) → -- Given equation
  (C = π / 3) → -- Given angle C
  (c = 2 * Real.sqrt 3) → -- Given side c
  -- Conclusions to prove
  (sin A / sin B = 2) ∧ 
  (1 / 2 * a * b * sin C = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2427_242720


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2427_242743

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2427_242743


namespace NUMINAMATH_CALUDE_sport_water_amount_l2427_242768

/-- Represents the ratios in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard drink formulation -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport drink formulation -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- The amount of corn syrup in the sport formulation -/
def sport_corn_syrup : ℚ := 3

theorem sport_water_amount :
  let water_amount := sport_corn_syrup * sport_ratio.water / sport_ratio.corn_syrup
  water_amount = 45 := by sorry

end NUMINAMATH_CALUDE_sport_water_amount_l2427_242768


namespace NUMINAMATH_CALUDE_eugene_toothpick_boxes_l2427_242716

/-- Represents the number of toothpicks needed for Eugene's model house --/
def toothpicks_needed (total_cards : ℕ) (unused_cards : ℕ) (wall_toothpicks : ℕ) 
  (window_count : ℕ) (door_count : ℕ) (window_door_toothpicks : ℕ) (roof_toothpicks : ℕ) : ℕ :=
  let used_cards := total_cards - unused_cards
  let wall_total := used_cards * wall_toothpicks
  let window_door_total := used_cards * (window_count + door_count) * window_door_toothpicks
  wall_total + window_door_total + roof_toothpicks

/-- Theorem stating that Eugene used at least 7 boxes of toothpicks --/
theorem eugene_toothpick_boxes : 
  ∀ (box_capacity : ℕ),
  box_capacity = 750 →
  ∃ (n : ℕ), n ≥ 7 ∧ 
  n * box_capacity ≥ toothpicks_needed 52 23 64 3 2 12 1250 :=
by sorry

end NUMINAMATH_CALUDE_eugene_toothpick_boxes_l2427_242716


namespace NUMINAMATH_CALUDE_dog_catches_fox_l2427_242748

/-- The speed of the dog in meters per second -/
def dog_speed : ℝ := 2

/-- The time the dog runs in each unit of time, in seconds -/
def dog_time : ℝ := 2

/-- The speed of the fox in meters per second -/
def fox_speed : ℝ := 3

/-- The time the fox runs in each unit of time, in seconds -/
def fox_time : ℝ := 1

/-- The initial distance between the dog and the fox in meters -/
def initial_distance : ℝ := 30

/-- The total distance the dog runs before catching the fox -/
def total_distance : ℝ := 120

theorem dog_catches_fox : 
  let dog_distance_per_unit := dog_speed * dog_time
  let fox_distance_per_unit := fox_speed * fox_time
  let distance_gained_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let units_to_catch := initial_distance / distance_gained_per_unit
  dog_distance_per_unit * units_to_catch = total_distance := by
sorry

end NUMINAMATH_CALUDE_dog_catches_fox_l2427_242748


namespace NUMINAMATH_CALUDE_dime_difference_is_90_l2427_242754

/-- Represents the number of coins of each type in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  halfDollars : ℕ

/-- Checks if the given coin count satisfies the problem conditions -/
def isValidCoinCount (c : CoinCount) : Prop :=
  c.nickels + c.dimes + c.halfDollars = 120 ∧
  5 * c.nickels + 10 * c.dimes + 50 * c.halfDollars = 1050

/-- Theorem stating the difference between max and min number of dimes -/
theorem dime_difference_is_90 :
  ∃ (min_dimes max_dimes : ℕ),
    (∃ c : CoinCount, isValidCoinCount c ∧ c.dimes = min_dimes) ∧
    (∃ c : CoinCount, isValidCoinCount c ∧ c.dimes = max_dimes) ∧
    (∀ c : CoinCount, isValidCoinCount c → c.dimes ≥ min_dimes ∧ c.dimes ≤ max_dimes) ∧
    max_dimes - min_dimes = 90 :=
by sorry

end NUMINAMATH_CALUDE_dime_difference_is_90_l2427_242754


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2427_242795

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = 7) ∧ 
  (3 * (-1)^2 + m * (-1) = 7) → 
  (∃ x : ℝ, x ≠ -1 ∧ 3 * x^2 + m * x = 7 ∧ x = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2427_242795


namespace NUMINAMATH_CALUDE_present_value_is_490_l2427_242737

/-- Given a banker's discount and true discount, calculates the present value. -/
def present_value (bankers_discount : ℚ) (true_discount : ℚ) : ℚ :=
  true_discount^2 / (bankers_discount - true_discount)

/-- Theorem stating that for the given banker's discount and true discount, the present value is 490. -/
theorem present_value_is_490 :
  present_value 80 70 = 490 := by
  sorry

#eval present_value 80 70

end NUMINAMATH_CALUDE_present_value_is_490_l2427_242737


namespace NUMINAMATH_CALUDE_one_is_last_digit_to_appear_l2427_242755

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => (modifiedFibonacci n + modifiedFibonacci (n + 1)) % 10

def digitAppearsInSequence (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ modifiedFibonacci k % 10 = d

def allDigitsAppear (n : ℕ) : Prop :=
  ∀ d, d < 10 → digitAppearsInSequence d n

def isLastDigitToAppear (d : ℕ) : Prop :=
  ∃ n, allDigitsAppear n ∧
    ¬(allDigitsAppear (n - 1)) ∧
    ¬(digitAppearsInSequence d (n - 1))

theorem one_is_last_digit_to_appear :
  isLastDigitToAppear 1 := by sorry

end NUMINAMATH_CALUDE_one_is_last_digit_to_appear_l2427_242755


namespace NUMINAMATH_CALUDE_function_properties_l2427_242725

def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

theorem function_properties (b : ℝ) :
  f b 0 = f b 4 →
  (b = 4 ∧
   (∀ x, f b x = 0 ↔ x = 1 ∨ x = 3) ∧
   (∀ x, f b x < 0 ↔ 1 < x ∧ x < 3) ∧
   (∀ x ∈ Set.Icc 0 3, f b x ≥ -1) ∧
   (∃ x ∈ Set.Icc 0 3, f b x = -1) ∧
   (∀ x ∈ Set.Icc 0 3, f b x ≤ 3) ∧
   (∃ x ∈ Set.Icc 0 3, f b x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2427_242725


namespace NUMINAMATH_CALUDE_map_scale_conversion_l2427_242783

/-- Given a map scale where 15 cm represents 90 km, prove that 25 cm represents 150 km -/
theorem map_scale_conversion (scale_cm : ℝ) (scale_km : ℝ) (distance_cm : ℝ) :
  scale_cm = 15 ∧ scale_km = 90 ∧ distance_cm = 25 →
  (distance_cm / scale_cm) * scale_km = 150 := by
sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l2427_242783


namespace NUMINAMATH_CALUDE_line_segment_division_l2427_242728

/-- Given a line segment with endpoints A(3, 2) and B(12, 8) divided into three equal parts,
    prove that the coordinates of the division points are C(6, 4) and D(9, 6),
    and the length of the segment AB is √117. -/
theorem line_segment_division (A B C D : ℝ × ℝ) : 
  A = (3, 2) → 
  B = (12, 8) → 
  C = ((3 + 0.5 * 12) / 1.5, (2 + 0.5 * 8) / 1.5) → 
  D = ((3 + 2 * 12) / 3, (2 + 2 * 8) / 3) → 
  C = (6, 4) ∧ 
  D = (9, 6) ∧ 
  Real.sqrt ((12 - 3)^2 + (8 - 2)^2) = Real.sqrt 117 :=
sorry

end NUMINAMATH_CALUDE_line_segment_division_l2427_242728


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2427_242713

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 8 = 5 / (x - 8) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2427_242713


namespace NUMINAMATH_CALUDE_sector_central_angle_central_angle_is_two_l2427_242718

theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 1) :
  ∃ (r : ℝ), r > 0 ∧ area = 1/2 * r * arc_length ∧ arc_length = 2 * r := by
  sorry

theorem central_angle_is_two (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 1) :
  let r := (2 * area) / arc_length
  arc_length / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_central_angle_is_two_l2427_242718


namespace NUMINAMATH_CALUDE_penumbra_ring_area_l2427_242742

/-- Given the ratio of radii of umbra to penumbra and the radius of the umbra,
    calculate the area of the penumbra ring around the umbra. -/
theorem penumbra_ring_area (umbra_radius : ℝ) (ratio_umbra : ℝ) (ratio_penumbra : ℝ) : 
  umbra_radius = 40 →
  ratio_umbra = 2 →
  ratio_penumbra = 6 →
  (ratio_penumbra / ratio_umbra * umbra_radius)^2 * Real.pi - umbra_radius^2 * Real.pi = 12800 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_penumbra_ring_area_l2427_242742


namespace NUMINAMATH_CALUDE_carton_height_is_70_l2427_242735

/-- Calculates the height of a carton given its base dimensions, soap box dimensions, and maximum capacity. -/
def carton_height (carton_length carton_width : ℕ) (box_length box_width box_height : ℕ) (max_boxes : ℕ) : ℕ :=
  let boxes_per_layer := (carton_length / box_length) * (carton_width / box_width)
  let num_layers := max_boxes / boxes_per_layer
  num_layers * box_height

/-- Theorem stating that the height of the carton is 70 inches given the specified conditions. -/
theorem carton_height_is_70 :
  carton_height 25 42 7 6 10 150 = 70 := by
  sorry

end NUMINAMATH_CALUDE_carton_height_is_70_l2427_242735


namespace NUMINAMATH_CALUDE_weeks_to_work_is_ten_l2427_242703

/-- The number of weeks Isabelle must work to afford concert tickets for herself and her brothers -/
def weeks_to_work : ℕ :=
let isabelle_ticket_cost : ℕ := 20
let brother_ticket_cost : ℕ := 10
let number_of_brothers : ℕ := 2
let total_savings : ℕ := 10
let weekly_earnings : ℕ := 3
let total_ticket_cost : ℕ := isabelle_ticket_cost + brother_ticket_cost * number_of_brothers
let additional_money_needed : ℕ := total_ticket_cost - total_savings
(additional_money_needed + weekly_earnings - 1) / weekly_earnings

theorem weeks_to_work_is_ten : weeks_to_work = 10 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_work_is_ten_l2427_242703


namespace NUMINAMATH_CALUDE_number_of_pears_number_of_pears_is_correct_l2427_242724

/-- The number of pears in a basket, given the following conditions:
  * There are 5 baskets in total
  * There are 58 fruits in total
  * One basket contains 18 mangoes
  * One basket contains 12 pawpaws
  * Two baskets contain the same number of kiwi and lemon respectively
  * There are 9 lemons
-/
theorem number_of_pears : ℕ :=
  let total_baskets : ℕ := 5
  let total_fruits : ℕ := 58
  let mangoes : ℕ := 18
  let pawpaws : ℕ := 12
  let lemons : ℕ := 9
  let kiwis : ℕ := lemons
  10

#check number_of_pears

theorem number_of_pears_is_correct : number_of_pears = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pears_number_of_pears_is_correct_l2427_242724


namespace NUMINAMATH_CALUDE_point_in_region_b_range_l2427_242700

theorem point_in_region_b_range (b : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  (2 * P.1 + 3 * P.2 - b > 0) → (b < 4) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_b_range_l2427_242700


namespace NUMINAMATH_CALUDE_unique_pairs_count_l2427_242759

/-- Represents the colors of marbles Tom has --/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | Yellow
  | Orange

/-- Represents Tom's collection of marbles --/
def toms_marbles : List MarbleColor :=
  [MarbleColor.Red, MarbleColor.Green, MarbleColor.Blue,
   MarbleColor.Yellow, MarbleColor.Yellow,
   MarbleColor.Orange, MarbleColor.Orange]

/-- Counts the number of unique pairs of marbles --/
def count_unique_pairs (marbles : List MarbleColor) : Nat :=
  sorry

/-- Theorem stating that the number of unique pairs Tom can choose is 12 --/
theorem unique_pairs_count :
  count_unique_pairs toms_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_count_l2427_242759


namespace NUMINAMATH_CALUDE_stating_smallest_n_no_arithmetic_progression_l2427_242780

/-- 
A function that checks if there exists an arithmetic progression of 
1999 terms containing exactly n integers
-/
def exists_arithmetic_progression (n : ℕ) : Prop :=
  ∃ (a d : ℝ), ∃ (k : ℕ), 
    k * n + k - 1 ≥ 1999 ∧
    (k + 1) * n - (k + 1) + 1 ≤ 1999

/-- 
Theorem stating that 70 is the smallest positive integer n such that 
there does not exist an arithmetic progression of 1999 terms of real 
numbers containing exactly n integers
-/
theorem smallest_n_no_arithmetic_progression : 
  (∀ m < 70, exists_arithmetic_progression m) ∧ 
  ¬ exists_arithmetic_progression 70 :=
sorry

end NUMINAMATH_CALUDE_stating_smallest_n_no_arithmetic_progression_l2427_242780


namespace NUMINAMATH_CALUDE_scientific_notation_of_billion_yuan_l2427_242744

def billion : ℝ := 1000000000

theorem scientific_notation_of_billion_yuan :
  let amount : ℝ := 2.175 * billion
  ∃ (a n : ℝ), amount = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_billion_yuan_l2427_242744


namespace NUMINAMATH_CALUDE_virus_infected_computers_office_virus_scenario_l2427_242798

/-- Represents the state of computers in an office before and after a virus infection. -/
structure ComputerNetwork where
  total : ℕ             -- Total number of computers
  infected : ℕ          -- Number of infected computers
  initialConnections : ℕ -- Number of initial connections per computer
  finalConnections : ℕ  -- Number of final connections per uninfected computer
  disconnectedCables : ℕ -- Number of cables disconnected due to virus

/-- The theorem stating the number of infected computers given the network conditions -/
theorem virus_infected_computers (network : ComputerNetwork) : 
  network.initialConnections = 5 ∧ 
  network.finalConnections = 3 ∧ 
  network.disconnectedCables = 26 →
  network.infected = 8 := by
  sorry

/-- Main theorem proving the number of infected computers in the given scenario -/
theorem office_virus_scenario : ∃ (network : ComputerNetwork), 
  network.initialConnections = 5 ∧
  network.finalConnections = 3 ∧
  network.disconnectedCables = 26 ∧
  network.infected = 8 := by
  sorry

end NUMINAMATH_CALUDE_virus_infected_computers_office_virus_scenario_l2427_242798


namespace NUMINAMATH_CALUDE_function_inequality_range_l2427_242701

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality_range 
  (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_value : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = Set.Icc 0 4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l2427_242701


namespace NUMINAMATH_CALUDE_min_value_problem_1_l2427_242746

theorem min_value_problem_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / (1 + y) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_1_l2427_242746


namespace NUMINAMATH_CALUDE_power_difference_quotient_l2427_242787

theorem power_difference_quotient : 
  (2^12)^2 - (2^10)^2 = 4 * ((2^11)^2 - (2^9)^2) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_quotient_l2427_242787


namespace NUMINAMATH_CALUDE_special_function_value_l2427_242775

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y + 6 * x * y) ∧
  (f (-1) * f 1 ≥ 9)

/-- Theorem stating that for any function satisfying the special conditions,
    f(2/3) = 4/3 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) :
  f (2/3) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2427_242775


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l2427_242767

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection :
  let line : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y + 1 = 0
  let circle : ℝ → ℝ → Prop := λ x y ↦ (x - 1)^2 + (y - 1)^2 = 1
  ∃ (A B : ℝ × ℝ),
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle A.1 A.2 ∧ circle B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l2427_242767


namespace NUMINAMATH_CALUDE_maci_pen_cost_l2427_242733

/-- The cost of Maci's pens given the number and prices of blue and red pens. -/
def cost_of_pens (blue_pens : ℕ) (red_pens : ℕ) (blue_pen_cost : ℚ) : ℚ :=
  let red_pen_cost := 2 * blue_pen_cost
  blue_pens * blue_pen_cost + red_pens * red_pen_cost

/-- Theorem stating that Maci pays $4.00 for her pens. -/
theorem maci_pen_cost : cost_of_pens 10 15 (10 / 100) = 4 := by
  sorry

#eval cost_of_pens 10 15 (10 / 100)

end NUMINAMATH_CALUDE_maci_pen_cost_l2427_242733


namespace NUMINAMATH_CALUDE_f_t_ratio_is_power_of_two_l2427_242778

/-- Define f_t(n) as the number of odd C_k^t for 1 ≤ k ≤ n -/
def f_t (t n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem f_t_ratio_is_power_of_two (t : ℕ) (h : ℕ) :
  t > 0 → ∃ r : ℕ, ∀ n : ℕ, n = 2^h → (f_t t n : ℚ) / n = 1 / (2^r) := by
  sorry

end NUMINAMATH_CALUDE_f_t_ratio_is_power_of_two_l2427_242778


namespace NUMINAMATH_CALUDE_hen_count_l2427_242750

theorem hen_count (total_heads : ℕ) (total_feet : ℕ) 
  (h_heads : total_heads = 48)
  (h_feet : total_feet = 140) :
  ∃ (hens cows : ℕ),
    hens + cows = total_heads ∧
    2 * hens + 4 * cows = total_feet ∧
    hens = 26 := by sorry

end NUMINAMATH_CALUDE_hen_count_l2427_242750


namespace NUMINAMATH_CALUDE_sean_train_track_length_l2427_242717

theorem sean_train_track_length 
  (ruth_piece_length : ℕ) 
  (total_length : ℕ) 
  (ruth_pieces : ℕ) 
  (sean_pieces : ℕ) :
  ruth_piece_length = 18 →
  total_length = 72 →
  ruth_pieces * ruth_piece_length = total_length →
  sean_pieces = ruth_pieces →
  sean_pieces * (total_length / sean_pieces) = total_length →
  total_length / sean_pieces = 18 :=
by
  sorry

#check sean_train_track_length

end NUMINAMATH_CALUDE_sean_train_track_length_l2427_242717


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l2427_242792

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8RS,
    the probability of a randomly selected point on PQ being between R and S is 1/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) 
  (h_order : P ≤ R ∧ R ≤ S ∧ S ≤ Q)
  (h_PQ_PR : Q - P = 4 * (R - P))
  (h_PQ_RS : Q - P = 8 * (S - R)) :
  (S - R) / (Q - P) = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l2427_242792


namespace NUMINAMATH_CALUDE_binomial_20_19_l2427_242729

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l2427_242729


namespace NUMINAMATH_CALUDE_prob_fewer_heads_12_fair_coins_l2427_242709

/-- The number of coin flips -/
def n : ℕ := 12

/-- The probability of getting heads on a single fair coin flip -/
def p : ℚ := 1/2

/-- The probability of getting fewer heads than tails in n fair coin flips -/
def prob_fewer_heads (n : ℕ) (p : ℚ) : ℚ :=
  sorry

theorem prob_fewer_heads_12_fair_coins : 
  prob_fewer_heads n p = 793/2048 := by sorry

end NUMINAMATH_CALUDE_prob_fewer_heads_12_fair_coins_l2427_242709


namespace NUMINAMATH_CALUDE_expenditure_ratio_l2427_242734

-- Define the monthly incomes and savings
def income_b : ℚ := 7200
def income_ratio : ℚ := 5 / 6
def savings_a : ℚ := 1800
def savings_b : ℚ := 1600

-- Define the monthly incomes
def income_a : ℚ := income_ratio * income_b

-- Define the monthly expenditures
def expenditure_a : ℚ := income_a - savings_a
def expenditure_b : ℚ := income_b - savings_b

-- Theorem to prove
theorem expenditure_ratio :
  expenditure_a / expenditure_b = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l2427_242734


namespace NUMINAMATH_CALUDE_marble_arrangement_mod_1000_l2427_242794

/-- The number of blue marbles -/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that allows for a valid arrangement -/
def yellow_marbles : ℕ := 18

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of ways to arrange the marbles -/
def arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem marble_arrangement_mod_1000 :
  arrangements % 1000 = 700 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_mod_1000_l2427_242794


namespace NUMINAMATH_CALUDE_minimum_formulas_to_memorize_l2427_242771

theorem minimum_formulas_to_memorize (total_formulas : ℕ) (min_score_percent : ℚ) : 
  total_formulas = 300 ∧ min_score_percent = 90 / 100 →
  ∃ (min_formulas : ℕ), 
    (min_formulas : ℚ) / total_formulas ≥ min_score_percent ∧
    ∀ (x : ℕ), (x : ℚ) / total_formulas ≥ min_score_percent → x ≥ min_formulas ∧
    min_formulas = 270 :=
by sorry

end NUMINAMATH_CALUDE_minimum_formulas_to_memorize_l2427_242771


namespace NUMINAMATH_CALUDE_color_natural_numbers_l2427_242706

theorem color_natural_numbers :
  ∃ (f : ℕ → Fin 2009),
    (∀ c : Fin 2009, Set.Infinite {n : ℕ | f n = c}) ∧
    (∀ x y z : ℕ, f x ≠ f y → f y ≠ f z → f x ≠ f z → x * y ≠ z) :=
by sorry

end NUMINAMATH_CALUDE_color_natural_numbers_l2427_242706


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l2427_242790

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  2 • (a - b) - 3 • (a + b) = -a - 5 • b :=
by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l2427_242790


namespace NUMINAMATH_CALUDE_unique_solution_2011_l2427_242766

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem unique_solution_2011 :
  ∃! n : ℕ, n + sum_of_digits n = 2011 ∧ n = 1991 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_2011_l2427_242766


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2427_242723

def fill_time_p : ℝ := 12
def fill_time_q : ℝ := 15
def initial_time : ℝ := 6

theorem cistern_fill_time : 
  let rate_p := 1 / fill_time_p
  let rate_q := 1 / fill_time_q
  let initial_fill := (rate_p + rate_q) * initial_time
  let remaining_fill := 1 - initial_fill
  remaining_fill / rate_q = 1.5 := by
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2427_242723


namespace NUMINAMATH_CALUDE_effective_treatment_combination_l2427_242786

structure Treatment where
  name : String
  relieves : List String
  causes : List String

def aspirin : Treatment :=
  { name := "Aspirin"
  , relieves := ["headache", "rheumatic knee pain"]
  , causes := ["heart pain", "stomach pain"] }

def antibiotics : Treatment :=
  { name := "Antibiotics"
  , relieves := ["migraine", "heart pain"]
  , causes := ["stomach pain", "knee pain", "itching"] }

def warmCompress : Treatment :=
  { name := "Warm compress"
  , relieves := ["itching", "stomach pain"]
  , causes := [] }

def initialSymptom : String := "headache"

def isEffectiveCombination (treatments : List Treatment) : Prop :=
  (initialSymptom ∈ (treatments.bind (λ t => t.relieves))) ∧
  (∀ s, s ∈ (treatments.bind (λ t => t.causes)) →
    ∃ t ∈ treatments, s ∈ t.relieves)

theorem effective_treatment_combination :
  isEffectiveCombination [aspirin, antibiotics, warmCompress] :=
sorry

end NUMINAMATH_CALUDE_effective_treatment_combination_l2427_242786


namespace NUMINAMATH_CALUDE_larger_number_proof_l2427_242727

/-- Given two positive integers with specific h.c.f. and l.c.m., prove the larger number is 391 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 23)
  (lcm : Nat.lcm a b = 23 * 13 * 17) :
  max a b = 391 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2427_242727


namespace NUMINAMATH_CALUDE_rabbits_ate_three_watermelons_l2427_242757

/-- The number of watermelons eaten by rabbits -/
def watermelons_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem: Given that Sam initially grew 4 watermelons and now has 1 left,
    prove that rabbits ate 3 watermelons -/
theorem rabbits_ate_three_watermelons :
  watermelons_eaten 4 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_ate_three_watermelons_l2427_242757


namespace NUMINAMATH_CALUDE_like_terms_implies_value_l2427_242776

-- Define the condition for like terms
def are_like_terms (m n : ℕ) : Prop := m = 3 ∧ n = 2

-- State the theorem
theorem like_terms_implies_value (m n : ℕ) :
  are_like_terms m n → (-n : ℤ)^m = -8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_implies_value_l2427_242776


namespace NUMINAMATH_CALUDE_area_regular_dodecagon_formula_l2427_242714

/-- A regular dodecagon inscribed in a circle -/
structure RegularDodecagon (r : ℝ) where
  -- The radius of the circumscribed circle
  radius : ℝ
  radius_pos : radius > 0
  -- The dodecagon is regular and inscribed in the circle

/-- The area of a regular dodecagon -/
def area_regular_dodecagon (d : RegularDodecagon r) : ℝ := 3 * r^2

/-- Theorem: The area of a regular dodecagon inscribed in a circle with radius r is 3r² -/
theorem area_regular_dodecagon_formula (r : ℝ) (hr : r > 0) :
  ∀ (d : RegularDodecagon r), area_regular_dodecagon d = 3 * r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_area_regular_dodecagon_formula_l2427_242714


namespace NUMINAMATH_CALUDE_circle_center_correct_l2427_242797

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-6) 1 2 (-75)
  findCenter eq = CircleCenter.mk 3 (-1) := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2427_242797


namespace NUMINAMATH_CALUDE_perspective_difference_l2427_242762

def num_students : ℕ := 250
def num_teachers : ℕ := 6
def class_sizes : List ℕ := [100, 50, 50, 25, 15, 10]

def teacher_perspective (sizes : List ℕ) : ℚ :=
  (sizes.sum : ℚ) / num_teachers

def student_perspective (sizes : List ℕ) : ℚ :=
  (sizes.map (λ x => x * x)).sum / num_students

theorem perspective_difference :
  teacher_perspective class_sizes - student_perspective class_sizes = -22.13 := by
  sorry

end NUMINAMATH_CALUDE_perspective_difference_l2427_242762


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2427_242764

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and an asymptote y = 2x/3,
    the focal length is 2√13. -/
theorem hyperbola_focal_length (m : ℝ) :
  (∃ (x y : ℝ), x^2/9 - y^2/m = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), y = 2*x/3) →         -- Asymptote equation
  2 * Real.sqrt 13 = 2 * Real.sqrt ((9:ℝ) + m) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2427_242764


namespace NUMINAMATH_CALUDE_pyramid_volume_l2427_242799

/-- The volume of a pyramid with a rectangular base and equal edge lengths from apex to base corners -/
theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) :
  base_length = 5 →
  base_width = 7 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (base_diagonal / 2)^2)
  (1 / 3 : ℝ) * base_area * height = (35 * Real.sqrt 188) / 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2427_242799


namespace NUMINAMATH_CALUDE_reinforcement_size_l2427_242741

/-- Calculates the size of a reinforcement given initial garrison size, initial provision duration,
    time passed before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - time_passed)
  (provisions_left / remaining_duration) - initial_garrison

theorem reinforcement_size :
  let initial_garrison := 2000
  let initial_duration := 62
  let time_passed := 15
  let remaining_duration := 20
  calculate_reinforcement initial_garrison initial_duration time_passed remaining_duration = 2700 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l2427_242741


namespace NUMINAMATH_CALUDE_value_of_a_l2427_242710

def A (a : ℝ) : Set ℝ := {-1, 0, a}
def B (a : ℝ) : Set ℝ := {0, Real.sqrt a}

theorem value_of_a (a : ℝ) (h : B a ⊆ A a) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2427_242710


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l2427_242732

theorem floor_sqrt_eight_count : 
  (Finset.range 81 \ Finset.range 64).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l2427_242732


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2427_242745

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 12) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 12 / (7 + 2 * Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2427_242745
