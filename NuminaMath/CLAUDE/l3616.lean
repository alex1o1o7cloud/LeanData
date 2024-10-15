import Mathlib

namespace NUMINAMATH_CALUDE_odd_prime_square_root_l3616_361667

theorem odd_prime_square_root (p k : ℕ) : 
  Prime p → 
  Odd p → 
  k > 0 → 
  ∃ n : ℕ, n > 0 ∧ n * n = k * k - p * k → 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l3616_361667


namespace NUMINAMATH_CALUDE_bridge_length_l3616_361648

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time_s : ℝ) : 
  train_length = 160 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3616_361648


namespace NUMINAMATH_CALUDE_smallest_multiple_eighty_is_solution_eighty_is_smallest_l3616_361639

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 540 * x % 800 = 0 → x ≥ 80 := by
  sorry

theorem eighty_is_solution : 540 * 80 % 800 = 0 := by
  sorry

theorem eighty_is_smallest : ∀ y : ℕ, y > 0 ∧ 540 * y % 800 = 0 → y ≥ 80 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_eighty_is_solution_eighty_is_smallest_l3616_361639


namespace NUMINAMATH_CALUDE_ratio_proof_l3616_361649

theorem ratio_proof (a b c d : ℝ) 
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_proof_l3616_361649


namespace NUMINAMATH_CALUDE_larger_part_of_90_l3616_361676

theorem larger_part_of_90 (x : ℝ) : 
  x + (90 - x) = 90 ∧ 
  0.4 * x = 0.3 * (90 - x) + 15 → 
  max x (90 - x) = 60 := by
sorry

end NUMINAMATH_CALUDE_larger_part_of_90_l3616_361676


namespace NUMINAMATH_CALUDE_sum_of_roots_x4_minus_4x3_minus_1_l3616_361641

theorem sum_of_roots_x4_minus_4x3_minus_1 : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x : ℝ, x^4 - 4*x^3 - 1 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    r₁ + r₂ + r₃ + r₄ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_x4_minus_4x3_minus_1_l3616_361641


namespace NUMINAMATH_CALUDE_northern_village_conscription_l3616_361634

/-- The number of people to be conscripted from the northern village -/
def northern_conscription (total_population : ℕ) (northern_population : ℕ) (total_conscription : ℕ) : ℕ :=
  (northern_population * total_conscription) / total_population

theorem northern_village_conscription :
  northern_conscription 22500 8100 300 = 108 := by
sorry

end NUMINAMATH_CALUDE_northern_village_conscription_l3616_361634


namespace NUMINAMATH_CALUDE_unique_number_with_equal_sums_l3616_361692

theorem unique_number_with_equal_sums : 
  ∃! n : ℕ, 
    (n ≥ 10000) ∧ 
    (n % 10000 = 9876) ∧ 
    (n / 10000 + 9876 = n / 1000 + 876) ∧
    (n = 9999876) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_equal_sums_l3616_361692


namespace NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_and_s_l3616_361684

theorem log_10_7_in_terms_of_r_and_s (r s : ℝ) 
  (hr : Real.log 2 / Real.log 5 = r) 
  (hs : Real.log 7 / Real.log 2 = s) : 
  Real.log 7 / Real.log 10 = s * r / (r + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_and_s_l3616_361684


namespace NUMINAMATH_CALUDE_rotation_of_point_A_l3616_361615

-- Define the rotation function
def rotate_clockwise_90 (x y : ℝ) : ℝ × ℝ := (y, -x)

-- Define the theorem
theorem rotation_of_point_A : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := rotate_clockwise_90 A.1 A.2
  B = (1, -2) := by sorry

end NUMINAMATH_CALUDE_rotation_of_point_A_l3616_361615


namespace NUMINAMATH_CALUDE_pizza_distribution_l3616_361671

/-- Given 12 coworkers sharing 3 pizzas equally, where each pizza is cut into 8 slices,
    prove that each coworker will receive 2 slices. -/
theorem pizza_distribution (coworkers : ℕ) (pizzas : ℕ) (slices_per_pizza : ℕ) 
    (h1 : coworkers = 12)
    (h2 : pizzas = 3)
    (h3 : slices_per_pizza = 8) :
    (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_distribution_l3616_361671


namespace NUMINAMATH_CALUDE_netGoalsForTimesMiddleSchool_l3616_361630

/-- Calculates the net goals for a single match -/
def netGoals (goalsFor goalsAgainst : ℤ) : ℤ := goalsFor - goalsAgainst

/-- Represents the scores of three soccer matches -/
structure ThreeMatches where
  match1 : (ℤ × ℤ)
  match2 : (ℤ × ℤ)
  match3 : (ℤ × ℤ)

/-- The specific scores for the Times Middle School soccer team -/
def timesMiddleSchoolScores : ThreeMatches := {
  match1 := (5, 3)
  match2 := (2, 6)
  match3 := (2, 2)
}

/-- Theorem stating that the net number of goals for the given scores is -2 -/
theorem netGoalsForTimesMiddleSchool :
  (netGoals timesMiddleSchoolScores.match1.1 timesMiddleSchoolScores.match1.2) +
  (netGoals timesMiddleSchoolScores.match2.1 timesMiddleSchoolScores.match2.2) +
  (netGoals timesMiddleSchoolScores.match3.1 timesMiddleSchoolScores.match3.2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_netGoalsForTimesMiddleSchool_l3616_361630


namespace NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l3616_361697

/-- Represents the fraction of orange juice in a mixture -/
def orange_juice_fraction (juice_volume : ℚ) (total_volume : ℚ) : ℚ :=
  juice_volume / total_volume

/-- The volume of the first pitcher in mL -/
def pitcher1_volume : ℚ := 500

/-- The volume of the second pitcher in mL -/
def pitcher2_volume : ℚ := 700

/-- The fraction of orange juice in the first pitcher -/
def pitcher1_juice_fraction : ℚ := 1/2

/-- The fraction of orange juice in the second pitcher -/
def pitcher2_juice_fraction : ℚ := 3/5

/-- Theorem stating that the fraction of orange juice in the final mixture is 67/120 -/
theorem orange_juice_mixture_fraction : 
  orange_juice_fraction 
    (pitcher1_volume * pitcher1_juice_fraction + pitcher2_volume * pitcher2_juice_fraction)
    (pitcher1_volume + pitcher2_volume) = 67/120 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l3616_361697


namespace NUMINAMATH_CALUDE_max_npm_value_l3616_361636

/-- Represents a two-digit number with equal even digits -/
structure EvenTwoDigit where
  digit : Nat
  h1 : digit % 2 = 0
  h2 : digit < 10

/-- Represents a three-digit number of the form NPM -/
structure ThreeDigitNPM where
  n : Nat
  p : Nat
  m : Nat
  h1 : n > 0
  h2 : n < 10
  h3 : p < 10
  h4 : m < 10

/-- The main theorem stating the maximum value of NPM -/
theorem max_npm_value (mm : EvenTwoDigit) (m : Nat) (npm : ThreeDigitNPM) 
    (h1 : m < 10)
    (h2 : m = mm.digit)
    (h3 : m = npm.m)
    (h4 : (mm.digit * 10 + mm.digit) * m = npm.n * 100 + npm.p * 10 + npm.m) :
  npm.n * 100 + npm.p * 10 + npm.m ≤ 396 := by
  sorry

end NUMINAMATH_CALUDE_max_npm_value_l3616_361636


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3616_361672

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, mx^2 - mx + 2 = 0 ∧ (∀ y : ℝ, my^2 - my + 2 = 0 → y = x)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3616_361672


namespace NUMINAMATH_CALUDE_soda_machine_leak_time_difference_l3616_361653

/-- 
Given a machine that normally fills a barrel of soda in 3 minutes, 
but takes 5 minutes when leaking, prove that it will take 2n minutes 
longer to fill n barrels when leaking, given that it takes 24 minutes 
longer for 12 barrels.
-/
theorem soda_machine_leak_time_difference (n : ℕ) : 
  (3 : ℝ) = normal_fill_time_per_barrel →
  (5 : ℝ) = leaking_fill_time_per_barrel →
  24 = 12 * (leaking_fill_time_per_barrel - normal_fill_time_per_barrel) →
  2 * n = n * (leaking_fill_time_per_barrel - normal_fill_time_per_barrel) :=
by sorry


end NUMINAMATH_CALUDE_soda_machine_leak_time_difference_l3616_361653


namespace NUMINAMATH_CALUDE_unit_price_ratio_of_quantity_and_price_difference_l3616_361693

/-- Given two products A and B, where A offers 30% more quantity and costs 15% less than B,
    this theorem proves that the ratio of unit prices (A to B) is 17/26. -/
theorem unit_price_ratio_of_quantity_and_price_difference 
  (quantity_A quantity_B : ℝ) 
  (price_A price_B : ℝ) 
  (h_quantity : quantity_A = 1.3 * quantity_B)
  (h_price : price_A = 0.85 * price_B)
  (h_positive_quantity : quantity_B > 0)
  (h_positive_price : price_B > 0) :
  (price_A / quantity_A) / (price_B / quantity_B) = 17 / 26 := by
  sorry

end NUMINAMATH_CALUDE_unit_price_ratio_of_quantity_and_price_difference_l3616_361693


namespace NUMINAMATH_CALUDE_base_conversion_1987_to_base5_l3616_361675

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Theorem: 1987 in base 10 is equal to 30422 in base 5 -/
theorem base_conversion_1987_to_base5 :
  1987 = fromBase5 [2, 2, 4, 0, 3] := by sorry

end NUMINAMATH_CALUDE_base_conversion_1987_to_base5_l3616_361675


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3616_361694

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (p₁ p₂ p₃ p₄ : Nat), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  (∀ n : Nat, n > 0 → (∃ (q₁ q₂ q₃ q₄ : Nat), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
    n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0) → n ≥ 210) ∧
  210 % p₁ = 0 ∧ 210 % p₂ = 0 ∧ 210 % p₃ = 0 ∧ 210 % p₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3616_361694


namespace NUMINAMATH_CALUDE_exam_scores_l3616_361627

theorem exam_scores (x y : ℝ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  (x + 2 = 10) ∧ ((x * y + 98 + 70) / (x + 2) = 88) :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_l3616_361627


namespace NUMINAMATH_CALUDE_complex_multiplication_l3616_361698

theorem complex_multiplication (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (1 + a * Complex.I) = Real.sqrt 5) :
  (1 + a * Complex.I) * (1 + Complex.I) = -1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3616_361698


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l3616_361633

theorem sum_of_roots_equals_one :
  ∀ x₁ x₂ : ℝ, (x₁ + 3) * (x₁ - 4) = 22 ∧ (x₂ + 3) * (x₂ - 4) = 22 → x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l3616_361633


namespace NUMINAMATH_CALUDE_unique_numbers_with_lcm_conditions_l3616_361624

theorem unique_numbers_with_lcm_conditions :
  ∃! (x y z : ℕ),
    x > y ∧ x > z ∧
    Nat.lcm x y = 200 ∧
    Nat.lcm y z = 300 ∧
    Nat.lcm x z = 120 ∧
    x = 40 ∧ y = 25 ∧ z = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_numbers_with_lcm_conditions_l3616_361624


namespace NUMINAMATH_CALUDE_december_sales_multiple_l3616_361691

theorem december_sales_multiple (A : ℝ) (M : ℝ) (h1 : M > 0) :
  M * A = 0.3125 * (11 * A + M * A) → M = 5 := by
sorry

end NUMINAMATH_CALUDE_december_sales_multiple_l3616_361691


namespace NUMINAMATH_CALUDE_rectangular_plot_fence_l3616_361640

theorem rectangular_plot_fence (short_side : ℝ) : 
  short_side > 0 →
  2 * short_side + 2 * (3 * short_side) = 640 →
  short_side = 80 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_fence_l3616_361640


namespace NUMINAMATH_CALUDE_max_non_managers_l3616_361665

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 → 
  (managers : ℚ) / non_managers > 7 / 32 → 
  non_managers ≤ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_max_non_managers_l3616_361665


namespace NUMINAMATH_CALUDE_race_people_count_l3616_361681

theorem race_people_count (num_cars : ℕ) (initial_people_per_car : ℕ) (people_gained_halfway : ℕ) :
  num_cars = 20 →
  initial_people_per_car = 3 →
  people_gained_halfway = 1 →
  num_cars * (initial_people_per_car + people_gained_halfway) = 80 := by
sorry

end NUMINAMATH_CALUDE_race_people_count_l3616_361681


namespace NUMINAMATH_CALUDE_count_less_equal_04_l3616_361621

def count_less_equal (threshold : ℚ) (numbers : List ℚ) : ℕ :=
  (numbers.filter (λ x => x ≤ threshold)).length

theorem count_less_equal_04 : count_less_equal (4/10) [8/10, 1/2, 3/10] = 1 := by
  sorry

end NUMINAMATH_CALUDE_count_less_equal_04_l3616_361621


namespace NUMINAMATH_CALUDE_swan_count_l3616_361607

/-- The number of swans in a lake that has "a pair plus two more" -/
def pair_plus_two (x : ℕ) : Prop := ∃ n : ℕ, x = 2 * n + 2

/-- The number of swans in a lake that has "three minus three" -/
def three_minus_three (x : ℕ) : Prop := ∃ m : ℕ, x = 3 * m - 3

/-- The total number of swans satisfies both conditions -/
theorem swan_count : ∃ x : ℕ, pair_plus_two x ∧ three_minus_three x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_swan_count_l3616_361607


namespace NUMINAMATH_CALUDE_factory_output_increase_l3616_361623

theorem factory_output_increase (P : ℝ) : 
  (1 + P / 100) * 1.30 * (1 - 30.07 / 100) = 1 → P = 10 := by
sorry

end NUMINAMATH_CALUDE_factory_output_increase_l3616_361623


namespace NUMINAMATH_CALUDE_average_of_numbers_l3616_361696

def numbers : List ℤ := [54, 55, 57, 58, 59, 62, 62, 63, 65, 65]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3616_361696


namespace NUMINAMATH_CALUDE_book_club_task_distribution_l3616_361673

theorem book_club_task_distribution (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_club_task_distribution_l3616_361673


namespace NUMINAMATH_CALUDE_standard_deviation_calculation_l3616_361664

/-- A normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ

/-- The value that is exactly k standard deviations away from the mean -/
def value_k_std_dev_from_mean (d : NormalDistribution) (k : ℝ) : ℝ :=
  d.μ - k * d.σ

theorem standard_deviation_calculation (d : NormalDistribution) 
  (h1 : d.μ = 16.2)
  (h2 : value_k_std_dev_from_mean d 2 = 11.6) : 
  d.σ = 2.3 := by
sorry

end NUMINAMATH_CALUDE_standard_deviation_calculation_l3616_361664


namespace NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l3616_361688

/-- Given real numbers a and b, if x^2 + ax + b and x^2 + bx + a each have two distinct real roots,
    and the product of their roots results in exactly three distinct real roots,
    then the sum of these three distinct roots is 0. -/
theorem sum_of_distinct_roots_is_zero (a b : ℝ) 
    (h1 : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = r1 ∨ x = r2)
    (h2 : ∃ s1 s2 : ℝ, s1 ≠ s2 ∧ ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = s1 ∨ x = s2)
    (h3 : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
          ∀ x : ℝ, (x = t1 ∨ x = t2 ∨ x = t3) ↔ (x^2 + a*x + b = 0 ∨ x^2 + b*x + a = 0)) :
    t1 + t2 + t3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l3616_361688


namespace NUMINAMATH_CALUDE_firefighter_remaining_money_l3616_361626

-- Define the firefighter's financial parameters
def hourly_rate : ℚ := 30
def weekly_hours : ℚ := 48
def food_expense : ℚ := 500
def tax_expense : ℚ := 1000
def weeks_per_month : ℚ := 4

-- Calculate weekly and monthly earnings
def weekly_earnings : ℚ := hourly_rate * weekly_hours
def monthly_earnings : ℚ := weekly_earnings * weeks_per_month

-- Calculate monthly rent
def monthly_rent : ℚ := monthly_earnings / 3

-- Calculate total monthly expenses
def total_monthly_expenses : ℚ := monthly_rent + food_expense + tax_expense

-- Calculate remaining money after expenses
def remaining_money : ℚ := monthly_earnings - total_monthly_expenses

-- Theorem to prove
theorem firefighter_remaining_money :
  remaining_money = 2340 := by
  sorry

end NUMINAMATH_CALUDE_firefighter_remaining_money_l3616_361626


namespace NUMINAMATH_CALUDE_feeding_sequences_count_l3616_361603

/-- Represents the number of distinct pairs of animals -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to feed animals given the conditions -/
def feeding_sequences : ℕ :=
  1 * num_pairs.factorial * (num_pairs - 1) * (num_pairs - 2) * (num_pairs - 3) * (num_pairs - 4) * (num_pairs - 5)

/-- Theorem stating that the number of feeding sequences is 17280 -/
theorem feeding_sequences_count : feeding_sequences = 17280 := by
  sorry

end NUMINAMATH_CALUDE_feeding_sequences_count_l3616_361603


namespace NUMINAMATH_CALUDE_prime_sum_problem_l3616_361647

theorem prime_sum_problem (p q r s : ℕ) : 
  Prime p → Prime q → Prime r → Prime s →
  p < q → q < r → r < s →
  p * q * r * s + 1 = 4^(p + q) →
  r + s = 274 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l3616_361647


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l3616_361619

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l3616_361619


namespace NUMINAMATH_CALUDE_max_value_fraction_l3616_361635

theorem max_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  (x^2 / (2 * y)) ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3616_361635


namespace NUMINAMATH_CALUDE_true_propositions_l3616_361656

-- Define the propositions
def p₁ : Prop := ∀ a b : ℝ, a < b → a^2 < b^2
def p₂ : Prop := ∀ x : ℝ, x > 0 → Real.sin x < x
def p₃ : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f x / f (-x) = -1) ↔ (∀ x : ℝ, f (-x) = -f x)
def p₄ : Prop := ∀ a : ℕ → ℝ, (∀ n : ℕ, a (n+1) = a n * (a 2 / a 1)) →
  ((a 1 > a 2 ∧ a 2 > a 3) ↔ (∀ n : ℕ, a (n+1) < a n))

-- Theorem stating which propositions are true
theorem true_propositions :
  ¬p₁ ∧ p₂ ∧ ¬p₃ ∧ p₄ :=
sorry

end NUMINAMATH_CALUDE_true_propositions_l3616_361656


namespace NUMINAMATH_CALUDE_unique_assignment_exists_l3616_361680

-- Define the types for our images
inductive Image : Type
| cat
| chicken
| crab
| bear
| goat

-- Define a function type that assigns digits to images
def ImageAssignment := Image → Nat

-- Define the conditions for the row and column sums
def satisfiesRowSums (assignment : ImageAssignment) : Prop :=
  assignment Image.cat + assignment Image.chicken + assignment Image.crab + assignment Image.bear + assignment Image.goat = 15 ∧
  assignment Image.goat + assignment Image.goat + assignment Image.crab + assignment Image.bear + assignment Image.bear = 16 ∧
  assignment Image.chicken + assignment Image.chicken + assignment Image.goat + assignment Image.cat + assignment Image.cat = 15 ∧
  assignment Image.crab + assignment Image.crab + assignment Image.crab + assignment Image.crab + assignment Image.crab = 10 ∧
  assignment Image.bear + assignment Image.bear + assignment Image.chicken + assignment Image.chicken + assignment Image.goat = 21

def satisfiesColumnSums (assignment : ImageAssignment) : Prop :=
  assignment Image.cat + assignment Image.goat + assignment Image.chicken + assignment Image.crab + assignment Image.bear = 15 ∧
  assignment Image.chicken + assignment Image.bear + assignment Image.goat + assignment Image.crab + assignment Image.bear = 13 ∧
  assignment Image.crab + assignment Image.crab + assignment Image.chicken + assignment Image.chicken + assignment Image.goat = 17 ∧
  assignment Image.bear + assignment Image.bear + assignment Image.goat + assignment Image.cat + assignment Image.chicken = 20 ∧
  assignment Image.goat + assignment Image.bear + assignment Image.cat + assignment Image.crab + assignment Image.crab = 11

-- Define the condition for different images having different digits
def differentImagesHaveDifferentDigits (assignment : ImageAssignment) : Prop :=
  assignment Image.cat ≠ assignment Image.chicken ∧
  assignment Image.cat ≠ assignment Image.crab ∧
  assignment Image.cat ≠ assignment Image.bear ∧
  assignment Image.cat ≠ assignment Image.goat ∧
  assignment Image.chicken ≠ assignment Image.crab ∧
  assignment Image.chicken ≠ assignment Image.bear ∧
  assignment Image.chicken ≠ assignment Image.goat ∧
  assignment Image.crab ≠ assignment Image.bear ∧
  assignment Image.crab ≠ assignment Image.goat ∧
  assignment Image.bear ≠ assignment Image.goat

-- The main theorem
theorem unique_assignment_exists : 
  ∃! assignment : ImageAssignment, 
    satisfiesRowSums assignment ∧ 
    satisfiesColumnSums assignment ∧ 
    differentImagesHaveDifferentDigits assignment ∧
    assignment Image.cat = 1 ∧
    assignment Image.chicken = 5 ∧
    assignment Image.crab = 2 ∧
    assignment Image.bear = 4 ∧
    assignment Image.goat = 3 :=
  sorry


end NUMINAMATH_CALUDE_unique_assignment_exists_l3616_361680


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3616_361604

/-- The value of a when a line is tangent to a circle --/
theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 →
  (∃ (x y : ℝ), x^2 + y^2 - a*x = 0 ∧ x - y - 1 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 - a*x = 0 → x - y - 1 ≠ 0 ∨ 
    (∃ (x' y' : ℝ), x' ≠ x ∧ y' ≠ y ∧ x'^2 + y'^2 - a*x' = 0 ∧ x' - y' - 1 = 0)) →
  a = 2*(Real.sqrt 2 - 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3616_361604


namespace NUMINAMATH_CALUDE_not_divisible_by_81_l3616_361651

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ (n^3 - 9*n + 27)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_81_l3616_361651


namespace NUMINAMATH_CALUDE_breanna_books_count_l3616_361628

theorem breanna_books_count (tony_total : ℕ) (dean_total : ℕ) (tony_dean_shared : ℕ) (all_shared : ℕ) (total_different : ℕ) :
  tony_total = 23 →
  dean_total = 12 →
  tony_dean_shared = 3 →
  all_shared = 1 →
  total_different = 47 →
  ∃ breanna_total : ℕ,
    tony_total - tony_dean_shared - all_shared +
    dean_total - tony_dean_shared - all_shared +
    breanna_total - all_shared = total_different ∧
    breanna_total = 20 :=
by sorry

end NUMINAMATH_CALUDE_breanna_books_count_l3616_361628


namespace NUMINAMATH_CALUDE_circle_chord_tangent_relation_l3616_361683

/-- Given a circle with radius r, a chord FG extending to meet the tangent at F at point H,
    and a point I on FH such that FI = GH, prove that v^2 = u^3 / (r + u),
    where u is the distance of I from the tangent through G
    and v is the distance of I from the line through chord FG. -/
theorem circle_chord_tangent_relation (r : ℝ) (u v : ℝ) 
  (h_positive : r > 0) 
  (h_u_positive : u > 0) 
  (h_v_positive : v > 0) 
  (h_v_eq_r : v = r) : 
  v^2 = u^3 / (r + u) := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_tangent_relation_l3616_361683


namespace NUMINAMATH_CALUDE_belle_rawhide_bones_l3616_361645

/-- The number of dog biscuits Belle eats every evening -/
def dog_biscuits : ℕ := 4

/-- The cost of one dog biscuit in dollars -/
def dog_biscuit_cost : ℚ := 1/4

/-- The cost of one rawhide bone in dollars -/
def rawhide_bone_cost : ℚ := 1

/-- The total cost of Belle's treats for a week in dollars -/
def weekly_treat_cost : ℚ := 21

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of rawhide bones Belle eats every evening -/
def rawhide_bones : ℕ := 2

theorem belle_rawhide_bones :
  (dog_biscuits : ℚ) * dog_biscuit_cost * (days_in_week : ℚ) +
  (rawhide_bones : ℚ) * rawhide_bone_cost * (days_in_week : ℚ) =
  weekly_treat_cost :=
sorry

end NUMINAMATH_CALUDE_belle_rawhide_bones_l3616_361645


namespace NUMINAMATH_CALUDE_inequality_solution_l3616_361605

theorem inequality_solution (x : ℝ) :
  (x^3 / (x + 2) ≥ 3 / (x - 2) + 1) ↔ (x < -2 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3616_361605


namespace NUMINAMATH_CALUDE_pentagon_sum_l3616_361613

/-- Pentagon with specific properties -/
structure Pentagon where
  u : ℤ
  v : ℤ
  h1 : 1 ≤ v
  h2 : v < u
  A : ℝ × ℝ := (u, v)
  B : ℝ × ℝ := (v, u)
  C : ℝ × ℝ := (-v, u)
  D : ℝ × ℝ := (-u, v)
  E : ℝ × ℝ := (-u, -v)
  h3 : (D.1 - E.1) * (A.1 - E.1) + (D.2 - E.2) * (A.2 - E.2) = 0  -- ∠DEA = 90°
  h4 : (u^2 : ℝ) + v^2 = 500  -- Area of pentagon ABCDE is 500

/-- Theorem stating the sum of u and v -/
theorem pentagon_sum (p : Pentagon) : p.u + p.v = 20 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_l3616_361613


namespace NUMINAMATH_CALUDE_quadratic_properties_l3616_361659

def f (x : ℝ) := x^2 + 6*x + 5

theorem quadratic_properties :
  (f 0 = 5) ∧
  (∃ v : ℝ × ℝ, v = (-3, -4) ∧ ∀ x : ℝ, f x ≥ f v.1) ∧
  (∀ x : ℝ, f (x + (-3)) = f ((-3) - x)) ∧
  (∀ p : ℝ, f p ≠ -p^2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3616_361659


namespace NUMINAMATH_CALUDE_min_value_problem_l3616_361668

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x^2) + (1 / y^2) + (1 / (x * y)) ≥ 3 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ (1 / a^2) + (1 / b^2) + (1 / (a * b)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3616_361668


namespace NUMINAMATH_CALUDE_square_sum_eq_double_product_implies_zero_l3616_361606

theorem square_sum_eq_double_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_double_product_implies_zero_l3616_361606


namespace NUMINAMATH_CALUDE_optimal_bus_rental_l3616_361679

/-- Represents the rental problem for buses -/
structure BusRental where
  cost_a : ℕ  -- Cost of renting one bus A
  cost_b : ℕ  -- Cost of renting one bus B
  capacity_a : ℕ  -- Capacity of bus A
  capacity_b : ℕ  -- Capacity of bus B
  total_people : ℕ  -- Total number of people to transport
  total_buses : ℕ  -- Total number of buses to rent

/-- Calculates the total cost for a given number of buses A and B -/
def total_cost (br : BusRental) (num_a : ℕ) (num_b : ℕ) : ℕ :=
  num_a * br.cost_a + num_b * br.cost_b

/-- Calculates the total capacity for a given number of buses A and B -/
def total_capacity (br : BusRental) (num_a : ℕ) (num_b : ℕ) : ℕ :=
  num_a * br.capacity_a + num_b * br.capacity_b

/-- Theorem stating that renting 2 buses A and 6 buses B minimizes the cost -/
theorem optimal_bus_rental (br : BusRental) 
  (h1 : br.cost_a + br.cost_b = 500)
  (h2 : 2 * br.cost_a + 3 * br.cost_b = 1300)
  (h3 : br.capacity_a = 15)
  (h4 : br.capacity_b = 25)
  (h5 : br.total_people = 180)
  (h6 : br.total_buses = 8) :
  ∀ (num_a num_b : ℕ), 
    num_a + num_b = br.total_buses →
    total_capacity br num_a num_b ≥ br.total_people →
    total_cost br 2 6 ≤ total_cost br num_a num_b :=
sorry

end NUMINAMATH_CALUDE_optimal_bus_rental_l3616_361679


namespace NUMINAMATH_CALUDE_luke_total_score_l3616_361608

def total_points (points_per_round : ℕ) (num_rounds : ℕ) : ℕ :=
  points_per_round * num_rounds

theorem luke_total_score :
  let points_per_round : ℕ := 42
  let num_rounds : ℕ := 2
  total_points points_per_round num_rounds = 84 := by sorry

end NUMINAMATH_CALUDE_luke_total_score_l3616_361608


namespace NUMINAMATH_CALUDE_function_has_period_two_l3616_361637

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_unit_shift (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

def matches_exp_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = Real.exp (Real.log 2 * x)

-- State the theorem
theorem function_has_period_two (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : has_unit_shift f) 
  (h3 : matches_exp_on_unit_interval f) : 
  ∀ x, f (x + 2) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_has_period_two_l3616_361637


namespace NUMINAMATH_CALUDE_a_in_open_interval_l3616_361638

/-- The set A defined as {x | |x-a| ≤ 1} -/
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

/-- The set B defined as {x | x^2-5x+4 ≥ 0} -/
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

/-- Theorem stating that if A intersect B is empty, then a is in the open interval (2, 3) -/
theorem a_in_open_interval (a : ℝ) (h : A a ∩ B = ∅) : a ∈ Set.Ioo 2 3 := by
  sorry

#check a_in_open_interval

end NUMINAMATH_CALUDE_a_in_open_interval_l3616_361638


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l3616_361618

theorem quadratic_form_k_value :
  ∃ (a h k : ℚ), ∀ x, x^2 - 5*x = a*(x - h)^2 + k ∧ k = -25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l3616_361618


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_over_one_plus_i_l3616_361643

theorem imaginary_part_of_z_over_one_plus_i :
  ∀ (z : ℂ), z = 1 - 2 * I →
  (z / (1 + I)).im = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_over_one_plus_i_l3616_361643


namespace NUMINAMATH_CALUDE_high_school_population_change_l3616_361652

/-- Represents the number of students in a high school --/
structure SchoolPopulation where
  boys : ℕ
  girls : ℕ

/-- Represents the ratio of boys to girls --/
structure Ratio where
  boys : ℕ
  girls : ℕ

def SchoolPopulation.ratio (pop : SchoolPopulation) : Ratio :=
  { boys := pop.boys, girls := pop.girls }

theorem high_school_population_change 
  (initial_ratio : Ratio)
  (final_ratio : Ratio)
  (boys_left : ℕ)
  (girls_left : ℕ)
  (h1 : initial_ratio.boys = 3 ∧ initial_ratio.girls = 4)
  (h2 : final_ratio.boys = 4 ∧ final_ratio.girls = 5)
  (h3 : boys_left = 10)
  (h4 : girls_left = 20)
  (h5 : girls_left = 2 * boys_left) :
  ∃ (initial_pop : SchoolPopulation),
    initial_pop.ratio = initial_ratio ∧
    initial_pop.boys = 90 ∧
    let final_pop : SchoolPopulation :=
      { boys := initial_pop.boys - boys_left,
        girls := initial_pop.girls - girls_left }
    final_pop.ratio = final_ratio :=
  sorry


end NUMINAMATH_CALUDE_high_school_population_change_l3616_361652


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3616_361622

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 3 ∧ Real.sqrt 3 < ↑b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3616_361622


namespace NUMINAMATH_CALUDE_max_triangle_area_l3616_361663

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define a point on the parabola (excluding origin)
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ P ≠ (0, 0)

-- Define the tangent line from a point on the parabola
def tangent_line (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  point_on_parabola P ∧ ∃ (m b : ℝ), ∀ x, l x = m * x + b

-- Define the intersection points of the tangent with the ellipse
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  l A.1 = A.2 ∧ l B.1 = B.2

-- Theorem statement
theorem max_triangle_area 
  (P : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  tangent_line P l → intersection_points A B l →
  ∃ (S : ℝ), S ≤ 8 * Real.sqrt 3 ∧ 
  (∃ (P' : ℝ × ℝ) (l' : ℝ → ℝ) (A' B' : ℝ × ℝ),
    tangent_line P' l' ∧ intersection_points A' B' l' ∧
    S = 8 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3616_361663


namespace NUMINAMATH_CALUDE_reptiles_per_swamp_l3616_361646

theorem reptiles_per_swamp (total_reptiles : ℕ) (num_swamps : ℕ) 
  (h1 : total_reptiles = 1424) (h2 : num_swamps = 4) :
  total_reptiles / num_swamps = 356 := by
  sorry

end NUMINAMATH_CALUDE_reptiles_per_swamp_l3616_361646


namespace NUMINAMATH_CALUDE_cylinder_to_cone_base_area_l3616_361662

theorem cylinder_to_cone_base_area (cylinder_radius : Real) (cylinder_height : Real)
  (cone_height : Real) (h1 : cylinder_radius = 1) (h2 : cylinder_height = 1)
  (h3 : cone_height = cylinder_height)
  (h4 : π * cylinder_radius^2 * cylinder_height = (1/3) * π * cone_base_radius^2 * cone_height) :
  π * cone_base_radius^2 = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_to_cone_base_area_l3616_361662


namespace NUMINAMATH_CALUDE_shirts_washed_l3616_361602

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 21 → unwashed = 1 →
  short_sleeve + long_sleeve - unwashed = 29 := by
sorry

end NUMINAMATH_CALUDE_shirts_washed_l3616_361602


namespace NUMINAMATH_CALUDE_shopkeeper_loss_l3616_361677

/-- Represents the overall loss amount given stock worth and selling conditions --/
def overall_loss (stock_worth : ℝ) : ℝ :=
  let profit_part := 0.2 * stock_worth * 1.2
  let loss_part := 0.8 * stock_worth * 0.9
  stock_worth - (profit_part + loss_part)

/-- Theorem stating the overall loss for the given problem --/
theorem shopkeeper_loss : 
  overall_loss 12499.99 = 500 :=
by
  sorry

#eval overall_loss 12499.99

end NUMINAMATH_CALUDE_shopkeeper_loss_l3616_361677


namespace NUMINAMATH_CALUDE_neighborhood_b_cookie_boxes_l3616_361632

/-- 
Proves that each home in Neighborhood B buys 5 boxes of cookies given the conditions of the problem.
-/
theorem neighborhood_b_cookie_boxes : 
  let neighborhood_a_homes : ℕ := 10
  let neighborhood_a_boxes_per_home : ℕ := 2
  let neighborhood_b_homes : ℕ := 5
  let price_per_box : ℕ := 2
  let better_neighborhood_revenue : ℕ := 50
  
  neighborhood_b_homes > 0 →
  (neighborhood_a_homes * neighborhood_a_boxes_per_home * price_per_box < better_neighborhood_revenue) →
  
  ∃ (boxes_per_home_b : ℕ),
    boxes_per_home_b * neighborhood_b_homes * price_per_box = better_neighborhood_revenue ∧
    boxes_per_home_b = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_neighborhood_b_cookie_boxes_l3616_361632


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3616_361611

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_second_term 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1/4) 
  (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) :
  a 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3616_361611


namespace NUMINAMATH_CALUDE_sum_x_y_equals_three_halves_l3616_361609

theorem sum_x_y_equals_three_halves (x y : ℝ) : 
  y = Real.sqrt (3 - 2*x) + Real.sqrt (2*x - 3) → x + y = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_three_halves_l3616_361609


namespace NUMINAMATH_CALUDE_count_pairs_eq_32_l3616_361669

/-- The number of pairs of positive integers (m,n) satisfying m^2 + n^2 < 50 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 < 50) (Finset.product (Finset.range 50) (Finset.range 50))).card

theorem count_pairs_eq_32 : count_pairs = 32 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_32_l3616_361669


namespace NUMINAMATH_CALUDE_regular_tetrahedron_sphere_ratio_l3616_361661

/-- A regular tetrahedron is a tetrahedron with four congruent equilateral triangles as faces -/
structure RegularTetrahedron where
  -- We don't need to define the structure explicitly for this problem

/-- The ratio of the radius of the circumscribed sphere to the inscribed sphere of a regular tetrahedron -/
def circumscribed_to_inscribed_ratio (t : RegularTetrahedron) : ℚ :=
  3 / 1

/-- Theorem: The ratio of the radius of the circumscribed sphere to the inscribed sphere of a regular tetrahedron is 3:1 -/
theorem regular_tetrahedron_sphere_ratio (t : RegularTetrahedron) :
  circumscribed_to_inscribed_ratio t = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_sphere_ratio_l3616_361661


namespace NUMINAMATH_CALUDE_smallest_possible_value_l3616_361631

theorem smallest_possible_value (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 2) →
  (Nat.lcm a b = x * (x + 2)) →
  (a = 24) →
  (∀ c : ℕ+, c < b → ¬(Nat.gcd 24 c = x + 2 ∧ Nat.lcm 24 c = x * (x + 2))) →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l3616_361631


namespace NUMINAMATH_CALUDE_column_addition_sum_l3616_361695

theorem column_addition_sum : ∀ (w x y z : ℕ),
  w ≤ 9 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 →  -- digits are between 0 and 9
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →  -- digits are distinct
  y + w = 10 →  -- rightmost column
  x + y + 1 = 10 →  -- middle column
  w + z + 1 = 11 →  -- leftmost column
  w + x + y + z = 20 :=
by sorry

end NUMINAMATH_CALUDE_column_addition_sum_l3616_361695


namespace NUMINAMATH_CALUDE_rhombus_closeness_range_l3616_361614

-- Define the closeness function
def closeness (α β : ℝ) : ℝ := 180 - |α - β|

-- Theorem statement
theorem rhombus_closeness_range :
  ∀ α β : ℝ, 0 < α ∧ α < 180 → 0 < β ∧ β < 180 →
  0 < closeness α β ∧ closeness α β ≤ 180 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_closeness_range_l3616_361614


namespace NUMINAMATH_CALUDE_square_sum_formula_l3616_361644

theorem square_sum_formula (x y z a b c d : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (h4 : (x + y + z)^2 = d) : 
  x^2 + y^2 + z^2 = d - 2*(a + b + c) := by
sorry

end NUMINAMATH_CALUDE_square_sum_formula_l3616_361644


namespace NUMINAMATH_CALUDE_eight_divided_by_one_eighth_l3616_361600

theorem eight_divided_by_one_eighth (x y : ℝ) : x = 8 ∧ y = 1/8 → x / y = 64 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_one_eighth_l3616_361600


namespace NUMINAMATH_CALUDE_john_total_calories_l3616_361642

/-- Calculates the total calories consumed by John given the following conditions:
  * John eats 15 potato chips with a total of 90 calories
  * He eats 10 cheezits, each with 2/5 more calories than a chip
  * He eats 8 pretzels, each with 25% fewer calories than a cheezit
-/
theorem john_total_calories : ℝ := by
  -- Define the number of each item eaten
  let num_chips : ℕ := 15
  let num_cheezits : ℕ := 10
  let num_pretzels : ℕ := 8

  -- Define the total calories from chips
  let total_chip_calories : ℝ := 90

  -- Define the calorie increase ratio for cheezits compared to chips
  let cheezit_increase_ratio : ℝ := 2 / 5

  -- Define the calorie decrease ratio for pretzels compared to cheezits
  let pretzel_decrease_ratio : ℝ := 1 / 4

  -- Calculate the total calories
  have h : ∃ (total_calories : ℝ), total_calories = 224.4 := by sorry

  exact h.choose

end NUMINAMATH_CALUDE_john_total_calories_l3616_361642


namespace NUMINAMATH_CALUDE_star_calculation_l3616_361601

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := x^3 - y

-- State the theorem
theorem star_calculation :
  star (3^(star 5 18)) (2^(star 2 9)) = 3^321 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3616_361601


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l3616_361678

/-- The number of people in the group -/
def groupSize : ℕ := 6

/-- The number of ways to choose a President and Vice-President when A is not President -/
def waysWithoutA : ℕ := groupSize * (groupSize - 1)

/-- The number of ways to choose a President and Vice-President when A is President -/
def waysWithA : ℕ := 1 * (groupSize - 2)

/-- The total number of ways to choose a President and Vice-President -/
def totalWays : ℕ := waysWithoutA + waysWithA

theorem president_vice_president_selection :
  totalWays = 34 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l3616_361678


namespace NUMINAMATH_CALUDE_wasted_fruits_and_vegetables_is_15_l3616_361650

/-- Calculates the amount of wasted fruits and vegetables in pounds -/
def wasted_fruits_and_vegetables (meat_pounds : ℕ) (meat_price : ℚ) 
  (bread_pounds : ℕ) (bread_price : ℚ) (janitor_hours : ℕ) (janitor_normal_wage : ℚ)
  (minimum_wage : ℚ) (work_hours : ℕ) (fruit_veg_price : ℚ) : ℚ :=
  let meat_cost := meat_pounds * meat_price
  let bread_cost := bread_pounds * bread_price
  let janitor_cost := janitor_hours * (janitor_normal_wage * 1.5)
  let total_earnings := work_hours * minimum_wage
  let remaining_cost := total_earnings - (meat_cost + bread_cost + janitor_cost)
  remaining_cost / fruit_veg_price

theorem wasted_fruits_and_vegetables_is_15 :
  wasted_fruits_and_vegetables 20 5 60 (3/2) 10 10 8 50 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_wasted_fruits_and_vegetables_is_15_l3616_361650


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3616_361699

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 4, x^2 + a*x - 2 < 0) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3616_361699


namespace NUMINAMATH_CALUDE_euclidean_division_remainder_l3616_361686

theorem euclidean_division_remainder 
  (P : Polynomial ℝ) 
  (D : Polynomial ℝ) 
  (h1 : P = X^100 - 2*X^51 + 1)
  (h2 : D = X^2 - 1) :
  ∃ (Q R : Polynomial ℝ), 
    P = D * Q + R ∧ 
    R.degree < D.degree ∧ 
    R = -2*X + 2 := by
sorry

end NUMINAMATH_CALUDE_euclidean_division_remainder_l3616_361686


namespace NUMINAMATH_CALUDE_f_negative_a_l3616_361682

theorem f_negative_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + Real.sin x + 1
  f a = 2 → f (-a) = -2 := by
sorry

end NUMINAMATH_CALUDE_f_negative_a_l3616_361682


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3616_361655

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_condition : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3616_361655


namespace NUMINAMATH_CALUDE_factor_x12_minus_729_l3616_361620

theorem factor_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) :=
by
  have h : 729 = 3^6 := by norm_num
  sorry

end NUMINAMATH_CALUDE_factor_x12_minus_729_l3616_361620


namespace NUMINAMATH_CALUDE_coffee_mixture_proof_l3616_361654

/-- Given a total mixture and a ratio of coffee to milk, calculate the amount of coffee needed. -/
def coffee_amount (total_mixture : ℕ) (coffee_ratio milk_ratio : ℕ) : ℕ :=
  (total_mixture * coffee_ratio) / (coffee_ratio + milk_ratio)

/-- Theorem stating that for a 4400g mixture with a 2:9 coffee to milk ratio, 800g of coffee is needed. -/
theorem coffee_mixture_proof :
  coffee_amount 4400 2 9 = 800 := by
  sorry

end NUMINAMATH_CALUDE_coffee_mixture_proof_l3616_361654


namespace NUMINAMATH_CALUDE_train_speed_l3616_361625

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 360) 
  (h2 : bridge_length = 140) (h3 : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3616_361625


namespace NUMINAMATH_CALUDE_abc_fraction_equality_l3616_361658

theorem abc_fraction_equality (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 1)
  (h2 : a ≠ 1 ∧ a ≠ -1)
  (h3 : b ≠ 1 ∧ b ≠ -1)
  (h4 : c ≠ 1 ∧ c ≠ -1) :
  a / (1 - a^2) + b / (1 - b^2) + c / (1 - c^2) = 
  4 * a * b * c / ((1 - a^2) * (1 - b^2) * (1 - c^2)) := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_equality_l3616_361658


namespace NUMINAMATH_CALUDE_pencil_price_l3616_361617

theorem pencil_price (price : ℝ) : 
  price = 5000 - 20 → price / 10000 = 0.5 := by sorry

end NUMINAMATH_CALUDE_pencil_price_l3616_361617


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3616_361674

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  -- Right triangle condition
  a^2 + b^2 = c^2 →
  -- Perimeter condition
  a + b + c = 32 →
  -- Area condition
  (1/2) * a * b = 20 →
  -- Conclusion: hypotenuse length
  c = 59/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3616_361674


namespace NUMINAMATH_CALUDE_zongzi_purchase_problem_l3616_361689

/-- Represents the cost and quantity information for zongzi purchases -/
structure ZongziPurchase where
  cost_A : ℝ  -- Cost per bag of brand A zongzi
  cost_B : ℝ  -- Cost per bag of brand B zongzi
  quantity_A : ℕ  -- Quantity of brand A zongzi
  quantity_B : ℕ  -- Quantity of brand B zongzi
  total_cost : ℝ  -- Total cost of the purchase

/-- Theorem representing the zongzi purchase problem -/
theorem zongzi_purchase_problem 
  (purchase1 : ZongziPurchase)
  (purchase2 : ZongziPurchase)
  (h1 : purchase1.quantity_A = 100 ∧ purchase1.quantity_B = 150 ∧ purchase1.total_cost = 7000)
  (h2 : purchase2.quantity_A = 180 ∧ purchase2.quantity_B = 120 ∧ purchase2.total_cost = 8100)
  (h3 : purchase1.cost_A = purchase2.cost_A ∧ purchase1.cost_B = purchase2.cost_B) :
  ∃ (optimal_purchase : ZongziPurchase),
    purchase1.cost_A = 25 ∧
    purchase1.cost_B = 30 ∧
    optimal_purchase.quantity_A = 200 ∧
    optimal_purchase.quantity_B = 100 ∧
    optimal_purchase.total_cost = 8000 ∧
    optimal_purchase.quantity_A + optimal_purchase.quantity_B = 300 ∧
    optimal_purchase.quantity_A ≤ 2 * optimal_purchase.quantity_B ∧
    ∀ (other_purchase : ZongziPurchase),
      other_purchase.quantity_A + other_purchase.quantity_B = 300 →
      other_purchase.quantity_A ≤ 2 * other_purchase.quantity_B →
      other_purchase.total_cost ≥ optimal_purchase.total_cost := by
  sorry


end NUMINAMATH_CALUDE_zongzi_purchase_problem_l3616_361689


namespace NUMINAMATH_CALUDE_trig_fraction_value_l3616_361687

theorem trig_fraction_value (θ : Real) (h : Real.tan θ = -2) :
  (7 * Real.sin θ - 3 * Real.cos θ) / (4 * Real.sin θ + 5 * Real.cos θ) = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l3616_361687


namespace NUMINAMATH_CALUDE_balloon_multiple_l3616_361666

def nancy_balloons : ℝ := 7.0
def mary_balloons : ℝ := 1.75

theorem balloon_multiple : nancy_balloons / mary_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_balloon_multiple_l3616_361666


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_relation_l3616_361657

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Distance from the center of the inscribed circle to the right-angle vertex -/
  l : ℝ
  /-- Distance from the center of the inscribed circle to one of the other vertices -/
  m : ℝ
  /-- Distance from the center of the inscribed circle to the remaining vertex -/
  n : ℝ
  /-- l, m, and n are positive -/
  l_pos : l > 0
  m_pos : m > 0
  n_pos : n > 0

/-- The theorem relating the distances from the center of the inscribed circle to the vertices -/
theorem inscribed_circle_distance_relation (t : RightTriangleWithInscribedCircle) :
  1 / t.l^2 = 1 / t.m^2 + 1 / t.n^2 + Real.sqrt 2 / (t.m * t.n) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_relation_l3616_361657


namespace NUMINAMATH_CALUDE_integral_exp_2x_l3616_361616

theorem integral_exp_2x : ∫ x in (0)..(1/2), Real.exp (2*x) = (1/2) * (Real.exp 1 - 1) := by sorry

end NUMINAMATH_CALUDE_integral_exp_2x_l3616_361616


namespace NUMINAMATH_CALUDE_max_books_with_23_dollars_l3616_361610

/-- Represents the available book purchasing options -/
inductive BookOption
  | Single
  | Set4
  | Set7

/-- Returns the cost of a given book option -/
def cost (option : BookOption) : ℕ :=
  match option with
  | BookOption.Single => 2
  | BookOption.Set4 => 7
  | BookOption.Set7 => 12

/-- Returns the number of books in a given book option -/
def books (option : BookOption) : ℕ :=
  match option with
  | BookOption.Single => 1
  | BookOption.Set4 => 4
  | BookOption.Set7 => 7

/-- Represents a combination of book purchases -/
structure Purchase where
  singles : ℕ
  sets4 : ℕ
  sets7 : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * cost BookOption.Single +
  p.sets4 * cost BookOption.Set4 +
  p.sets7 * cost BookOption.Set7

/-- Calculates the total number of books in a purchase -/
def totalBooks (p : Purchase) : ℕ :=
  p.singles * books BookOption.Single +
  p.sets4 * books BookOption.Set4 +
  p.sets7 * books BookOption.Set7

/-- Theorem: The maximum number of books that can be purchased with $23 is 13 -/
theorem max_books_with_23_dollars :
  ∃ (p : Purchase), totalCost p ≤ 23 ∧
  totalBooks p = 13 ∧
  ∀ (q : Purchase), totalCost q ≤ 23 → totalBooks q ≤ 13 := by
  sorry


end NUMINAMATH_CALUDE_max_books_with_23_dollars_l3616_361610


namespace NUMINAMATH_CALUDE_sin_390_degrees_l3616_361670

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l3616_361670


namespace NUMINAMATH_CALUDE_hyperbola_and_ellipse_condition_l3616_361660

/-- Represents a hyperbola equation -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (1 - m) + y^2 / (m + 2) = 1

/-- Represents an ellipse equation with foci on the x-axis -/
def is_ellipse_x_foci (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2 * m) + y^2 / (2 - m) = 1

/-- Main theorem -/
theorem hyperbola_and_ellipse_condition (m : ℝ) 
  (h1 : is_hyperbola m) (h2 : is_ellipse_x_foci m) : 
  1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_ellipse_condition_l3616_361660


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3616_361685

/-- A line passing through point A(2, 1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point A(2, 1) -/
  passes_through_A : m * 2 + b = 1
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b = b / m

/-- The equation of the line is either x - 2y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3616_361685


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3616_361612

theorem inequality_equivalence (x : ℝ) : (x - 3) / (x^2 + 4*x + 10) ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3616_361612


namespace NUMINAMATH_CALUDE_range_of_a_characterize_solution_set_l3616_361629

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > x^2 + a*x - 1 - a) → a ∈ Set.Icc 1 5 := by sorry

-- Part 2
-- We define a function that characterizes the solution set based on 'a'
noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -(a+1)/a ∨ x > 1}
  else if a = 0 then {x | x > 1}
  else if -1/2 < a ∧ a < 0 then {x | 1 < x ∧ x < -(a+1)/a}
  else if a = -1/2 then ∅
  else {x | -(a+1)/a < x ∧ x < 1}

theorem characterize_solution_set (a : ℝ) :
  {x : ℝ | f a x > 1} = solution_set a := by sorry

end NUMINAMATH_CALUDE_range_of_a_characterize_solution_set_l3616_361629


namespace NUMINAMATH_CALUDE_frieda_hop_probability_l3616_361690

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Checks if a position is a corner -/
def is_corner (p : Position) : Bool :=
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 3) ∨
  (p.x = 3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 3)

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, considering wrap-around rules -/
def apply_hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Calculates the probability of reaching a corner in at most n hops -/
def prob_reach_corner (start : Position) (n : Nat) : ℚ :=
  sorry  -- Proof implementation goes here

/-- The main theorem to prove -/
theorem frieda_hop_probability :
  prob_reach_corner ⟨0, 1⟩ 3 = 21 / 32 :=
by sorry  -- Proof goes here

end NUMINAMATH_CALUDE_frieda_hop_probability_l3616_361690
