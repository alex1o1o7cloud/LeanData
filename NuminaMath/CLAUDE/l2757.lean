import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2757_275708

theorem quadratic_root_difference (x : ℝ) : 
  5 * x^2 - 9 * x - 22 = 0 →
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    (5 * r₁^2 - 9 * r₁ - 22 = 0) ∧
    (5 * r₂^2 - 9 * r₂ - 22 = 0) ∧
    |r₁ - r₂| = Real.sqrt 521 / 5 ∧
    (∀ (p : ℕ), p > 1 → ¬(p^2 ∣ 521)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2757_275708


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2757_275748

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2757_275748


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2757_275716

theorem simplify_sqrt_sum : 
  (Real.sqrt 418 / Real.sqrt 308) + (Real.sqrt 294 / Real.sqrt 196) = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2757_275716


namespace NUMINAMATH_CALUDE_pacos_marble_purchase_l2757_275739

theorem pacos_marble_purchase : 
  0.33 + 0.33 + 0.08 = 0.74 := by sorry

end NUMINAMATH_CALUDE_pacos_marble_purchase_l2757_275739


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2757_275734

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 1)
  (sum_prod_eq : a * b + a * c + b * c = -3)
  (prod_eq : a * b * c = 4) :
  a^3 + b^3 + c^3 = 1 := by
    sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2757_275734


namespace NUMINAMATH_CALUDE_percentage_difference_l2757_275760

theorem percentage_difference (A B C : ℝ) 
  (hB_C : B = 0.63 * C) 
  (hB_A : B = 0.90 * A) : 
  A = 0.70 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2757_275760


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l2757_275749

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

end NUMINAMATH_CALUDE_sunset_time_calculation_l2757_275749


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2757_275789

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2757_275789


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l2757_275712

theorem smallest_five_digit_mod_9 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 9 = 5 → n ≥ 10000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l2757_275712


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_2012_l2757_275723

def units_digit_cycle : List Nat := [3, 9, 7, 1]

theorem units_digit_of_3_pow_2012 :
  (3^2012 : Nat) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_2012_l2757_275723


namespace NUMINAMATH_CALUDE_dice_sum_probability_l2757_275743

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

end NUMINAMATH_CALUDE_dice_sum_probability_l2757_275743


namespace NUMINAMATH_CALUDE_log_difference_equals_one_l2757_275710

theorem log_difference_equals_one (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (α + π / 4) = 3) : 
  Real.log (8 * Real.sin α + 6 * Real.cos α) - Real.log (4 * Real.sin α - Real.cos α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_one_l2757_275710


namespace NUMINAMATH_CALUDE_sticker_count_l2757_275701

/-- Given the ratio of stickers and Kate's sticker count, prove the combined count of Jenna's and Ava's stickers -/
theorem sticker_count (kate_ratio jenna_ratio ava_ratio : ℕ) 
  (kate_stickers : ℕ) (h_ratio : kate_ratio = 7 ∧ jenna_ratio = 4 ∧ ava_ratio = 5) 
  (h_kate : kate_stickers = 42) : 
  (jenna_ratio + ava_ratio) * (kate_stickers / kate_ratio) = 54 := by
  sorry

#check sticker_count

end NUMINAMATH_CALUDE_sticker_count_l2757_275701


namespace NUMINAMATH_CALUDE_number_added_to_multiples_of_three_l2757_275761

theorem number_added_to_multiples_of_three : ∃ x : ℕ, 
  x + (3 * 14 + 3 * 15 + 3 * 18) = 152 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_added_to_multiples_of_three_l2757_275761


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2757_275773

def polynomial (x : ℤ) : ℤ := x^4 + 2*x^3 - x^2 + 3*x - 30

def possible_roots : Set ℤ := {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30}

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = possible_roots := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2757_275773


namespace NUMINAMATH_CALUDE_male_wage_is_35_l2757_275798

/-- Represents the daily wage structure and worker composition of a building contractor -/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage of a male worker given the contractor's data -/
def male_wage (data : ContractorData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers + data.child_workers
  let total_wage := total_workers * data.average_wage
  let female_total := data.female_workers * data.female_wage
  let child_total := data.child_workers * data.child_wage
  (total_wage - female_total - child_total) / data.male_workers

/-- Theorem stating that for the given contractor data, the male wage is 35 -/
theorem male_wage_is_35 (data : ContractorData) 
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.child_workers = 5)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  male_wage data = 35 := by
  sorry

#eval male_wage { 
  male_workers := 20, 
  female_workers := 15, 
  child_workers := 5, 
  female_wage := 20, 
  child_wage := 8, 
  average_wage := 26 
}

end NUMINAMATH_CALUDE_male_wage_is_35_l2757_275798


namespace NUMINAMATH_CALUDE_EPC42_probability_l2757_275753

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

end NUMINAMATH_CALUDE_EPC42_probability_l2757_275753


namespace NUMINAMATH_CALUDE_fraction_equality_l2757_275775

theorem fraction_equality : (8 : ℚ) / (5 * 48) = 0.8 / (5 * 0.48) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2757_275775


namespace NUMINAMATH_CALUDE_combined_area_of_tracts_l2757_275720

/-- The combined area of two rectangular tracts of land -/
theorem combined_area_of_tracts (length1 width1 length2 width2 : ℕ) 
  (h1 : length1 = 300)
  (h2 : width1 = 500)
  (h3 : length2 = 250)
  (h4 : width2 = 630) :
  length1 * width1 + length2 * width2 = 307500 := by
  sorry

end NUMINAMATH_CALUDE_combined_area_of_tracts_l2757_275720


namespace NUMINAMATH_CALUDE_evaluate_expression_l2757_275796

theorem evaluate_expression : 2000^3 - 1999 * 2000^2 - 1999^2 * 2000 + 1999^3 = 3999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2757_275796


namespace NUMINAMATH_CALUDE_topsoil_cost_for_8_cubic_yards_l2757_275765

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

end NUMINAMATH_CALUDE_topsoil_cost_for_8_cubic_yards_l2757_275765


namespace NUMINAMATH_CALUDE_y_derivative_l2757_275783

noncomputable def y (x : ℝ) : ℝ := 
  -1 / (3 * (Real.sin x)^3) - 1 / (Real.sin x) + (1/2) * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

theorem y_derivative (x : ℝ) (hx : Real.cos x ≠ 0) (hsx : Real.sin x ≠ 0) : 
  deriv y x = 1 / (Real.cos x * (Real.sin x)^4) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2757_275783


namespace NUMINAMATH_CALUDE_tank_capacity_l2757_275732

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

end NUMINAMATH_CALUDE_tank_capacity_l2757_275732


namespace NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l2757_275766

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l2757_275766


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_equality_and_counterexample_l2757_275769

theorem floor_sqrt_sum_equality_and_counterexample :
  (∀ n : ℕ, ⌊Real.sqrt n + Real.sqrt (n + 2)⌋ = ⌊Real.sqrt (4 * n + 1)⌋) ∧
  (∃ x : ℝ, ⌊Real.sqrt x + Real.sqrt (x + 2)⌋ ≠ ⌊Real.sqrt (4 * x + 1)⌋) :=
by sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_equality_and_counterexample_l2757_275769


namespace NUMINAMATH_CALUDE_clock_angle_at_9_l2757_275730

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of degrees each hour represents on a clock face -/
def degrees_per_hour : ℕ := full_circle / clock_hours

/-- The position of the minute hand at 9:00 in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 9:00 in degrees -/
def hour_hand_position : ℕ := 9 * degrees_per_hour

/-- The smaller angle between the hour hand and minute hand at 9:00 -/
def smaller_angle : ℕ := min (hour_hand_position - minute_hand_position) (full_circle - (hour_hand_position - minute_hand_position))

theorem clock_angle_at_9 : smaller_angle = 90 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_9_l2757_275730


namespace NUMINAMATH_CALUDE_probability_two_cards_sum_15_l2757_275736

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

end NUMINAMATH_CALUDE_probability_two_cards_sum_15_l2757_275736


namespace NUMINAMATH_CALUDE_exactly_one_absent_probability_l2757_275771

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 1 / 15

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students chosen -/
def n : ℕ := 3

/-- The number of students that should be absent -/
def k : ℕ := 1

theorem exactly_one_absent_probability :
  (n.choose k : ℚ) * p_absent^k * p_present^(n - k) = 588 / 3375 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_absent_probability_l2757_275771


namespace NUMINAMATH_CALUDE_triangle_count_2008_l2757_275707

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

end NUMINAMATH_CALUDE_triangle_count_2008_l2757_275707


namespace NUMINAMATH_CALUDE_prob_sum_20_l2757_275746

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

end NUMINAMATH_CALUDE_prob_sum_20_l2757_275746


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2757_275735

theorem perpendicular_lines_b_value (b : ℚ) : 
  (∀ x y : ℚ, 2 * x - 3 * y + 6 = 0 → (∃ m₁ : ℚ, y = m₁ * x + 2)) ∧ 
  (∀ x y : ℚ, b * x - 3 * y + 6 = 0 → (∃ m₂ : ℚ, y = m₂ * x + 2)) ∧
  (∃ m₁ m₂ : ℚ, m₁ * m₂ = -1) →
  b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2757_275735


namespace NUMINAMATH_CALUDE_range_of_m_l2757_275758

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∪ B m = A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2757_275758


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l2757_275726

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a-2)*x

-- Define the tangent line at the origin
def tangent_line_at_origin (a : ℝ) (x : ℝ) : ℝ := -2*x

-- Theorem statement
theorem tangent_line_theorem (a : ℝ) :
  ∀ x : ℝ, (tangent_line_at_origin a x) = 
    (deriv (f a)) 0 * x + (f a 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l2757_275726


namespace NUMINAMATH_CALUDE_geometric_progression_sum_l2757_275737

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

end NUMINAMATH_CALUDE_geometric_progression_sum_l2757_275737


namespace NUMINAMATH_CALUDE_number_count_in_average_calculation_l2757_275752

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

end NUMINAMATH_CALUDE_number_count_in_average_calculation_l2757_275752


namespace NUMINAMATH_CALUDE_prob_S7_eq_3_l2757_275768

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of a single draw -/
def drawOutcome (c : BallColor) : Int :=
  match c with
  | BallColor.Red => -1
  | BallColor.White => 1

/-- The probability of drawing a red ball -/
def probRed : ℚ := 2/3

/-- The probability of drawing a white ball -/
def probWhite : ℚ := 1/3

/-- The number of draws -/
def n : ℕ := 7

/-- The sum we're interested in -/
def targetSum : Int := 3

/-- The probability of getting the target sum after n draws -/
def probTargetSum (n : ℕ) (targetSum : Int) : ℚ :=
  sorry

theorem prob_S7_eq_3 :
  probTargetSum n targetSum = 28 / 3^6 :=
sorry

end NUMINAMATH_CALUDE_prob_S7_eq_3_l2757_275768


namespace NUMINAMATH_CALUDE_starters_count_l2757_275790

/-- Represents a set of twins -/
structure TwinSet :=
  (twin1 : ℕ)
  (twin2 : ℕ)

/-- Represents a basketball team -/
structure BasketballTeam :=
  (total_players : ℕ)
  (twin_set1 : TwinSet)
  (twin_set2 : TwinSet)

/-- Calculates the number of ways to choose starters with twin restrictions -/
def choose_starters (team : BasketballTeam) (num_starters : ℕ) : ℕ :=
  sorry

/-- The specific basketball team in the problem -/
def problem_team : BasketballTeam :=
  { total_players := 18
  , twin_set1 := { twin1 := 1, twin2 := 2 }  -- Representing Ben & Jerry
  , twin_set2 := { twin1 := 3, twin2 := 4 }  -- Representing Tom & Tim
  }

theorem starters_count : choose_starters problem_team 5 = 1834 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l2757_275790


namespace NUMINAMATH_CALUDE_abs_sum_values_l2757_275706

theorem abs_sum_values (a b : ℝ) (ha : |a| = 3) (hb : |b| = 1) :
  |a + b| = 4 ∨ |a + b| = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_values_l2757_275706


namespace NUMINAMATH_CALUDE_steve_reading_time_l2757_275799

/-- Represents Steve's daily reading schedule in pages -/
def daily_reading : Fin 7 → ℕ
  | 0 => 100  -- Monday
  | 1 => 150  -- Tuesday
  | 2 => 100  -- Wednesday
  | 3 => 150  -- Thursday
  | 4 => 100  -- Friday
  | 5 => 50   -- Saturday
  | 6 => 0    -- Sunday

/-- The total number of pages in the book -/
def book_length : ℕ := 2100

/-- Calculate the total pages read in a week -/
def pages_per_week : ℕ := (List.range 7).map daily_reading |>.sum

/-- The number of weeks needed to read the book -/
def weeks_to_read : ℕ := (book_length + pages_per_week - 1) / pages_per_week

theorem steve_reading_time :
  weeks_to_read = 4 := by sorry

end NUMINAMATH_CALUDE_steve_reading_time_l2757_275799


namespace NUMINAMATH_CALUDE_dessert_percentage_l2757_275742

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

end NUMINAMATH_CALUDE_dessert_percentage_l2757_275742


namespace NUMINAMATH_CALUDE_thompson_children_probability_l2757_275702

theorem thompson_children_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each child being male (or female)
  let total_outcomes : ℕ := 2^n  -- total number of possible gender combinations
  let equal_outcomes : ℕ := n.choose (n/2)  -- number of combinations with equal sons and daughters
  
  (total_outcomes - equal_outcomes : ℚ) / total_outcomes = 93/128 :=
by sorry

end NUMINAMATH_CALUDE_thompson_children_probability_l2757_275702


namespace NUMINAMATH_CALUDE_f_monotone_implies_a_range_l2757_275725

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x - 1 else Real.log x / Real.log a

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_monotone_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → 3 < a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_implies_a_range_l2757_275725


namespace NUMINAMATH_CALUDE_inverse_trig_sum_l2757_275795

theorem inverse_trig_sum : 
  Real.arcsin (-1/2) + Real.arccos (-Real.sqrt 3/2) + Real.arctan (-Real.sqrt 3) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_l2757_275795


namespace NUMINAMATH_CALUDE_friend_product_sum_l2757_275722

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

end NUMINAMATH_CALUDE_friend_product_sum_l2757_275722


namespace NUMINAMATH_CALUDE_triangle_subdivision_l2757_275755

/-- Given a triangle ABC with n arbitrary non-collinear points inside it,
    the number of non-overlapping small triangles formed by connecting
    all points (including vertices A, B, C) is (2n + 1) -/
def num_small_triangles (n : ℕ) : ℕ := 2 * n + 1

/-- The main theorem stating that for 2008 points inside triangle ABC,
    the number of small triangles is 4017 -/
theorem triangle_subdivision :
  num_small_triangles 2008 = 4017 := by
  sorry

end NUMINAMATH_CALUDE_triangle_subdivision_l2757_275755


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l2757_275705

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l2757_275705


namespace NUMINAMATH_CALUDE_steve_has_four_friends_l2757_275764

/-- The number of friends Steve has, given the initial number of gold bars,
    the number of lost gold bars, and the number of gold bars each friend receives. -/
def number_of_friends (initial_bars : ℕ) (lost_bars : ℕ) (bars_per_friend : ℕ) : ℕ :=
  (initial_bars - lost_bars) / bars_per_friend

/-- Theorem stating that Steve has 4 friends given the problem conditions. -/
theorem steve_has_four_friends :
  number_of_friends 100 20 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_steve_has_four_friends_l2757_275764


namespace NUMINAMATH_CALUDE_X_prob_implies_n_10_l2757_275717

/-- A random variable X taking values from 1 to n with equal probability -/
def X (n : ℕ) := Fin n

/-- The probability of X being less than 4 -/
def prob_X_less_than_4 (n : ℕ) : ℚ := (3 : ℚ) / n

/-- Theorem stating that if P(X < 4) = 0.3, then n = 10 -/
theorem X_prob_implies_n_10 (n : ℕ) (h : prob_X_less_than_4 n = (3 : ℚ) / 10) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_X_prob_implies_n_10_l2757_275717


namespace NUMINAMATH_CALUDE_mistaken_calculation_l2757_275757

theorem mistaken_calculation (x : ℝ) : x + 2 = 6 → x - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l2757_275757


namespace NUMINAMATH_CALUDE_annas_earnings_is_96_l2757_275778

/-- Calculates Anna's earnings from selling cupcakes given the number of trays, cupcakes per tray, price per cupcake, and fraction sold. -/
def annas_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (fraction_sold : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * fraction_sold * price_per_cupcake

/-- Theorem stating that Anna's earnings are $96 given the specific conditions. -/
theorem annas_earnings_is_96 :
  annas_earnings 4 20 2 (3/5) = 96 := by
  sorry

end NUMINAMATH_CALUDE_annas_earnings_is_96_l2757_275778


namespace NUMINAMATH_CALUDE_soil_bags_needed_l2757_275776

/-- Calculates the number of soil bags needed for raised beds -/
theorem soil_bags_needed
  (num_beds : ℕ)
  (length width height : ℝ)
  (soil_per_bag : ℝ)
  (h_num_beds : num_beds = 2)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_soil_per_bag : soil_per_bag = 4) :
  ⌈(num_beds * length * width * height) / soil_per_bag⌉ = 16 :=
by sorry

end NUMINAMATH_CALUDE_soil_bags_needed_l2757_275776


namespace NUMINAMATH_CALUDE_expected_red_pairs_l2757_275700

/-- The expected number of pairs of adjacent red cards in a circular arrangement -/
theorem expected_red_pairs (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ)
  (h1 : total_cards = 60)
  (h2 : red_cards = 30)
  (h3 : black_cards = 30)
  (h4 : total_cards = red_cards + black_cards) :
  (red_cards : ℚ) * (red_cards - 1 : ℚ) / (total_cards - 1 : ℚ) = 870 / 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_l2757_275700


namespace NUMINAMATH_CALUDE_suzanna_textbook_pages_l2757_275787

/-- Calculate the total number of pages in Suzanna's textbooks --/
theorem suzanna_textbook_pages : 
  let history_pages : ℕ := 160
  let geography_pages : ℕ := history_pages + 70
  let math_pages : ℕ := (history_pages + geography_pages) / 2
  let science_pages : ℕ := 2 * history_pages
  history_pages + geography_pages + math_pages + science_pages = 905 :=
by sorry

end NUMINAMATH_CALUDE_suzanna_textbook_pages_l2757_275787


namespace NUMINAMATH_CALUDE_seashells_given_to_mike_l2757_275745

/-- Given that Joan initially found 79 seashells and now has 16 seashells,
    prove that the number of seashells she gave to Mike is 63. -/
theorem seashells_given_to_mike 
  (initial_seashells : ℕ) 
  (current_seashells : ℕ) 
  (h1 : initial_seashells = 79) 
  (h2 : current_seashells = 16) : 
  initial_seashells - current_seashells = 63 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_mike_l2757_275745


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2757_275711

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2757_275711


namespace NUMINAMATH_CALUDE_coral_reading_pages_l2757_275762

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

end NUMINAMATH_CALUDE_coral_reading_pages_l2757_275762


namespace NUMINAMATH_CALUDE_select_five_from_ten_l2757_275779

theorem select_five_from_ten : Nat.choose 10 5 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_ten_l2757_275779


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2757_275759

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (x₀ y₁ y₂ y₃ : ℝ) :
  a ≠ 0 →
  f a b c (-2) = 0 →
  x₀ > 1 →
  f a b c x₀ = 0 →
  (a + b + c) * (4 * a + 2 * b + c) < 0 →
  ∃ y, y < 0 ∧ f a b c 0 = y →
  f a b c (-1) = y₁ →
  f a b c (-Real.sqrt 2 / 2) = y₂ →
  f a b c 1 = y₃ →
  y₃ > y₁ ∧ y₁ > y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2757_275759


namespace NUMINAMATH_CALUDE_g_expression_l2757_275713

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g implicitly using its relationship with f
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l2757_275713


namespace NUMINAMATH_CALUDE_road_signs_at_first_intersection_l2757_275751

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

end NUMINAMATH_CALUDE_road_signs_at_first_intersection_l2757_275751


namespace NUMINAMATH_CALUDE_one_instrument_one_sport_probability_l2757_275744

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

end NUMINAMATH_CALUDE_one_instrument_one_sport_probability_l2757_275744


namespace NUMINAMATH_CALUDE_y_derivative_l2757_275729

noncomputable def y (x : ℝ) : ℝ := (1/4) * Real.log (abs (Real.tanh (x/2))) - (1/4) * Real.log ((3 + Real.cosh x) / Real.sinh x)

theorem y_derivative (x : ℝ) : deriv y x = 1 / (2 * Real.sinh x) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l2757_275729


namespace NUMINAMATH_CALUDE_quadratic_second_difference_constant_l2757_275715

/-- Second difference of a function f at point n -/
def second_difference (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (f (n + 2) - f (n + 1)) - (f (n + 1) - f n)

/-- A quadratic function with linear and constant terms -/
def quadratic_function (a b : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ)^2 + a * (n : ℝ) + b

theorem quadratic_second_difference_constant (a b : ℝ) :
  ∀ n : ℕ, second_difference (quadratic_function a b) n = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_second_difference_constant_l2757_275715


namespace NUMINAMATH_CALUDE_statement_a_is_false_statement_b_is_true_statement_c_is_true_statement_d_is_true_statement_e_is_true_l2757_275785

-- Statement A
theorem statement_a_is_false : ∃ (a b c : ℝ), a > b ∧ c < 0 ∧ a * c ≤ b * c :=
  sorry

-- Statement B
theorem statement_b_is_true : ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y → 
  (2 * x * y) / (x + y) < Real.sqrt (x * y) :=
  sorry

-- Statement C
theorem statement_c_is_true : ∀ (s : ℝ), s > 0 → 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s → 
  x * y ≤ (s / 2) * (s / 2) :=
  sorry

-- Statement D
theorem statement_d_is_true : ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y → 
  (x^2 + y^2) / 2 > ((x + y) / 2)^2 :=
  sorry

-- Statement E
theorem statement_e_is_true : ∀ (p : ℝ), p > 0 → 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p → 
  x + y ≥ 2 * Real.sqrt p :=
  sorry

end NUMINAMATH_CALUDE_statement_a_is_false_statement_b_is_true_statement_c_is_true_statement_d_is_true_statement_e_is_true_l2757_275785


namespace NUMINAMATH_CALUDE_f_at_neg_one_eq_78_l2757_275740

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

end NUMINAMATH_CALUDE_f_at_neg_one_eq_78_l2757_275740


namespace NUMINAMATH_CALUDE_labourer_income_l2757_275727

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


end NUMINAMATH_CALUDE_labourer_income_l2757_275727


namespace NUMINAMATH_CALUDE_three_digit_primes_ending_in_one_l2757_275714

theorem three_digit_primes_ending_in_one (p : ℕ) : 
  (200 < p ∧ p < 1000 ∧ p % 10 = 1 ∧ Nat.Prime p) → 
  (Finset.filter (λ x => 200 < x ∧ x < 1000 ∧ x % 10 = 1 ∧ Nat.Prime x) (Finset.range 1000)).card = 23 :=
sorry

end NUMINAMATH_CALUDE_three_digit_primes_ending_in_one_l2757_275714


namespace NUMINAMATH_CALUDE_product_mod_seven_l2757_275780

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2757_275780


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2757_275770

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 15 → x ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2757_275770


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2757_275763

/-- The line y = kx + 1 and the parabola y^2 = 4x have exactly one point in common if and only if k = 0 or k = 1 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 1 ∧ p.2^2 = 4 * p.1) ↔ k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2757_275763


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2757_275772

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 + 5*x > 6

-- Define the solution set
def solution_set : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2757_275772


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l2757_275774

/-- Given a quadratic function f(x) = ax^2 - 4x + c where a ≠ 0,
    with range [0, +∞) and f(1) ≤ 4, the maximum value of
    u = a/(c^2+4) + c/(a^2+4) is 7/4. -/
theorem quadratic_function_max_value (a c : ℝ) (h1 : a ≠ 0) :
  let f := fun x => a * x^2 - 4 * x + c
  (∀ y, y ∈ Set.range f → y ≥ 0) →
  (f 1 ≤ 4) →
  (∃ u : ℝ, u = a / (c^2 + 4) + c / (a^2 + 4) ∧
    u ≤ 7/4 ∧
    ∀ v, v = a / (c^2 + 4) + c / (a^2 + 4) → v ≤ u) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l2757_275774


namespace NUMINAMATH_CALUDE_peter_soda_purchase_l2757_275741

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

end NUMINAMATH_CALUDE_peter_soda_purchase_l2757_275741


namespace NUMINAMATH_CALUDE_empty_subset_of_A_l2757_275767

def A : Set ℝ := {x : ℝ | x^2 - x = 0}

theorem empty_subset_of_A : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_subset_of_A_l2757_275767


namespace NUMINAMATH_CALUDE_truck_filling_time_truck_filling_time_proof_l2757_275786

/-- The time taken to fill a truck with stone blocks given specific worker rates and capacity -/
theorem truck_filling_time : ℕ :=
  let truck_capacity : ℕ := 6000
  let stella_initial_rate : ℕ := 250
  let twinkle_initial_rate : ℕ := 200
  let stella_changed_rate : ℕ := 220
  let twinkle_changed_rate : ℕ := 230
  let additional_workers_count : ℕ := 6
  let additional_workers_initial_rate1 : ℕ := 300
  let additional_workers_initial_rate2 : ℕ := 180
  let additional_workers_changed_rate1 : ℕ := 280
  let additional_workers_changed_rate2 : ℕ := 190
  let initial_period : ℕ := 2
  let second_period : ℕ := 4
  let additional_workers_initial_period : ℕ := 1

  8

theorem truck_filling_time_proof : truck_filling_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_truck_filling_time_truck_filling_time_proof_l2757_275786


namespace NUMINAMATH_CALUDE_triangle_area_l2757_275750

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

end NUMINAMATH_CALUDE_triangle_area_l2757_275750


namespace NUMINAMATH_CALUDE_marias_towels_l2757_275728

theorem marias_towels (green_towels white_towels given_towels : ℕ) : 
  green_towels = 40 →
  white_towels = 44 →
  given_towels = 65 →
  green_towels + white_towels - given_towels = 19 := by
sorry

end NUMINAMATH_CALUDE_marias_towels_l2757_275728


namespace NUMINAMATH_CALUDE_experience_ratio_l2757_275788

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.roger + 8 = 50 ∧
  e.peter = 12 ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

theorem experience_ratio (e : Experience) 
  (h : satisfiesConditions e) : e.tom = e.robert :=
sorry

end NUMINAMATH_CALUDE_experience_ratio_l2757_275788


namespace NUMINAMATH_CALUDE_alloy_composition_l2757_275791

theorem alloy_composition (gold_weight copper_weight alloy_weight : ℝ) 
  (h1 : gold_weight = 19)
  (h2 : alloy_weight = 17)
  (h3 : (4 * gold_weight + copper_weight) / 5 = alloy_weight) : 
  copper_weight = 9 := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_l2757_275791


namespace NUMINAMATH_CALUDE_min_value_of_f_l2757_275718

/-- The quadratic function f(x) = 3(x+2)^2 - 5 -/
def f (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

/-- The minimum value of f(x) is -5 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -5 ∧ ∃ x₀ : ℝ, f x₀ = -5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2757_275718


namespace NUMINAMATH_CALUDE_tom_car_washing_earnings_l2757_275724

/-- Represents the amount of money Tom had last week in dollars -/
def initial_amount : ℕ := 74

/-- Represents the amount of money Tom has now in dollars -/
def current_amount : ℕ := 86

/-- Represents the amount of money Tom made washing cars in dollars -/
def money_made : ℕ := current_amount - initial_amount

theorem tom_car_washing_earnings :
  money_made = current_amount - initial_amount :=
by sorry

end NUMINAMATH_CALUDE_tom_car_washing_earnings_l2757_275724


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2757_275784

theorem min_value_of_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  let f := (x₁ + x₃) / (x₅ + 2*x₂ + 3*x₄) + (x₂ + x₄) / (x₁ + 2*x₃ + 3*x₅) + 
           (x₃ + x₅) / (x₂ + 2*x₄ + 3*x₁) + (x₄ + x₁) / (x₃ + 2*x₅ + 3*x₂) + 
           (x₅ + x₂) / (x₄ + 2*x₁ + 3*x₃)
  f ≥ 5/3 ∧ (f = 5/3 ↔ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2757_275784


namespace NUMINAMATH_CALUDE_cricket_solution_l2757_275777

def cricket_problem (initial_average : ℝ) (runs_10th_innings : ℕ) : Prop :=
  let total_runs_9_innings := 9 * initial_average
  let total_runs_10_innings := total_runs_9_innings + runs_10th_innings
  let new_average := total_runs_10_innings / 10
  (new_average = initial_average + 8) ∧ (new_average = 128)

theorem cricket_solution :
  ∀ initial_average : ℝ,
  ∃ runs_10th_innings : ℕ,
  cricket_problem initial_average runs_10th_innings ∧
  runs_10th_innings = 200 :=
by sorry

end NUMINAMATH_CALUDE_cricket_solution_l2757_275777


namespace NUMINAMATH_CALUDE_cube_root_equality_l2757_275704

theorem cube_root_equality (m : ℝ) : 
  (9 + 9 / m) ^ (1/3) = 9 * (9 / m) ^ (1/3) → m = 728 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equality_l2757_275704


namespace NUMINAMATH_CALUDE_card_difference_l2757_275797

theorem card_difference (heike anton ann : ℕ) : 
  anton = 3 * heike →
  ann = 6 * heike →
  ann = 60 →
  ann - anton = 30 := by
sorry

end NUMINAMATH_CALUDE_card_difference_l2757_275797


namespace NUMINAMATH_CALUDE_operation_result_l2757_275754

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

end NUMINAMATH_CALUDE_operation_result_l2757_275754


namespace NUMINAMATH_CALUDE_max_correct_is_38_l2757_275782

/-- Represents the scoring system and results of a math contest. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions for a given contest. -/
def max_correct_answers (contest : MathContest) : ℕ :=
  sorry

/-- Theorem stating that for the given contest parameters, the maximum number of correct answers is 38. -/
theorem max_correct_is_38 :
  let contest := MathContest.mk 60 5 0 (-2) 150
  max_correct_answers contest = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_is_38_l2757_275782


namespace NUMINAMATH_CALUDE_all_triangles_present_l2757_275781

/-- A permissible triangle with angles represented as integers -/
structure PermissibleTriangle (p : ℕ) :=
  (a b c : ℕ)
  (sum_eq_p : a + b + c = p)
  (all_pos : 0 < a ∧ 0 < b ∧ 0 < c)

/-- The set of all permissible triangles for a given prime p -/
def AllPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | true}

/-- A function representing the division process -/
def DivideTriangle (p : ℕ) (t : PermissibleTriangle p) : Option (PermissibleTriangle p × PermissibleTriangle p) :=
  sorry

/-- The set of triangles after the division process is complete -/
def FinalTriangleSet (p : ℕ) : Set (PermissibleTriangle p) :=
  sorry

/-- The main theorem -/
theorem all_triangles_present (p : ℕ) (hp : Prime p) :
  FinalTriangleSet p = AllPermissibleTriangles p :=
sorry

end NUMINAMATH_CALUDE_all_triangles_present_l2757_275781


namespace NUMINAMATH_CALUDE_equal_areas_of_inscribed_polygons_with_same_side_lengths_l2757_275703

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

end NUMINAMATH_CALUDE_equal_areas_of_inscribed_polygons_with_same_side_lengths_l2757_275703


namespace NUMINAMATH_CALUDE_power_sum_calculation_l2757_275738

theorem power_sum_calculation : (-1: ℤ)^53 + (2 : ℚ)^(2^4 + 5^2 - 4^3) = -1 + 1 / 8388608 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_calculation_l2757_275738


namespace NUMINAMATH_CALUDE_maddie_weekend_watching_time_maddie_weekend_watching_time_proof_l2757_275793

/-- Given Maddie's TV watching schedule, prove she watched 105 minutes over the weekend -/
theorem maddie_weekend_watching_time : ℕ → Prop :=
  λ weekend_minutes : ℕ =>
    let total_episodes : ℕ := 8
    let minutes_per_episode : ℕ := 44
    let monday_minutes : ℕ := 138
    let thursday_minutes : ℕ := 21
    let friday_episodes : ℕ := 2

    let total_minutes : ℕ := total_episodes * minutes_per_episode
    let weekday_minutes : ℕ := monday_minutes + thursday_minutes + (friday_episodes * minutes_per_episode)

    weekend_minutes = total_minutes - weekday_minutes ∧ weekend_minutes = 105

/-- Proof of the theorem -/
theorem maddie_weekend_watching_time_proof : maddie_weekend_watching_time 105 := by
  sorry

end NUMINAMATH_CALUDE_maddie_weekend_watching_time_maddie_weekend_watching_time_proof_l2757_275793


namespace NUMINAMATH_CALUDE_special_function_bound_l2757_275733

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ x^2 * f (y/2) + y^2 * f (x/2)) ∧
  (∃ M : ℝ, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M)

/-- The main theorem stating that f(x) ≤ x^2 for all x ≥ 0 -/
theorem special_function_bound {f : ℝ → ℝ} (hf : SpecialFunction f) :
  ∀ x, x ≥ 0 → f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_bound_l2757_275733


namespace NUMINAMATH_CALUDE_first_issue_pages_l2757_275719

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

end NUMINAMATH_CALUDE_first_issue_pages_l2757_275719


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2757_275747

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2757_275747


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l2757_275709

theorem largest_integer_for_negative_quadratic : 
  ∃ (n : ℤ), n = 7 ∧ n^2 - 11*n + 24 < 0 ∧ ∀ (m : ℤ), m > n → m^2 - 11*m + 24 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l2757_275709


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2757_275794

-- Define the circle and points
def Circle : Type := ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the radius of the circle
def radius (c : Circle) : ℝ := sorry

-- Define the center of the circle
def center (c : Circle) : Point := sorry

-- Define a point on the circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define a secant
def is_secant (p q r : Point) (c : Circle) : Prop := 
  ¬(on_circle p c) ∧ on_circle q c ∧ on_circle r c

-- Theorem statement
theorem circle_radius_problem (c : Circle) (p q r : Point) :
  distance p (center c) = 17 →
  is_secant p q r c →
  distance p q = 11 →
  distance q r = 8 →
  radius c = 4 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2757_275794


namespace NUMINAMATH_CALUDE_savings_ratio_l2757_275756

def savings_problem (monday tuesday wednesday thursday : ℚ) : Prop :=
  let total_savings := monday + tuesday + wednesday
  let ratio := thursday / total_savings
  (monday = 15) ∧ (tuesday = 28) ∧ (wednesday = 13) ∧ (thursday = 28) → ratio = 1/2

theorem savings_ratio : ∀ (monday tuesday wednesday thursday : ℚ),
  savings_problem monday tuesday wednesday thursday :=
λ monday tuesday wednesday thursday => by
  sorry

end NUMINAMATH_CALUDE_savings_ratio_l2757_275756


namespace NUMINAMATH_CALUDE_cos_negative_thirty_degrees_l2757_275721

theorem cos_negative_thirty_degrees : Real.cos (-(30 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_thirty_degrees_l2757_275721


namespace NUMINAMATH_CALUDE_allocation_methods_l2757_275731

/-- The number of warriors in the class -/
def total_warriors : ℕ := 6

/-- The number of tasks to be completed -/
def num_tasks : ℕ := 4

/-- The number of leadership positions (captain and vice-captain) -/
def leadership_positions : ℕ := 2

/-- The number of participating warriors -/
def participating_warriors : ℕ := 4

theorem allocation_methods :
  (leadership_positions.choose 1) *
  ((total_warriors - leadership_positions).choose (participating_warriors - 1)) *
  (participating_warriors.factorial) = 192 :=
sorry

end NUMINAMATH_CALUDE_allocation_methods_l2757_275731


namespace NUMINAMATH_CALUDE_frustum_cone_volume_l2757_275792

/-- Given a frustum of a cone with volume 78 and one base area 9 times the other,
    the volume of the cone that cuts this frustum is 81. -/
theorem frustum_cone_volume (r R : ℝ) (h1 : r > 0) (h2 : R > 0) : 
  (π * (R^2 + r^2 + R*r) * (R - r) / 3 = 78) →
  (π * R^2 = 9 * π * r^2) →
  (π * R^3 / 3 = 81) := by
  sorry

end NUMINAMATH_CALUDE_frustum_cone_volume_l2757_275792
