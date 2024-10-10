import Mathlib

namespace unique_element_quadratic_set_l3010_301019

theorem unique_element_quadratic_set (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a = 0 ∨ a = 1) := by
  sorry

end unique_element_quadratic_set_l3010_301019


namespace annie_total_travel_l3010_301075

def blocks_to_bus_stop : ℕ := 5
def blocks_on_bus : ℕ := 7

def one_way_trip : ℕ := blocks_to_bus_stop + blocks_on_bus

theorem annie_total_travel : one_way_trip * 2 = 24 := by
  sorry

end annie_total_travel_l3010_301075


namespace solution_to_equation_l3010_301076

theorem solution_to_equation : 
  ∃! x : ℝ, (Real.sqrt x + 3 * Real.sqrt (x^3 + 7*x) + Real.sqrt (x + 7) = 50 - x^2) ∧ 
            (x = (29/12)^2) := by
  sorry

end solution_to_equation_l3010_301076


namespace opposite_of_negative_four_thirds_l3010_301086

theorem opposite_of_negative_four_thirds :
  -(-(4/3 : ℚ)) = 4/3 := by sorry

end opposite_of_negative_four_thirds_l3010_301086


namespace point_in_second_quadrant_exists_l3010_301053

theorem point_in_second_quadrant_exists : ∃ (x y : ℤ), 
  x < 0 ∧ 
  y > 0 ∧ 
  y ≤ x + 4 ∧ 
  x = -1 ∧ 
  y = 3 := by
sorry

end point_in_second_quadrant_exists_l3010_301053


namespace square_root_equation_solution_l3010_301012

theorem square_root_equation_solution : 
  ∃ x : ℝ, (56^2 + 56^2) / x^2 = 8 ∧ x = 28 := by
  sorry

end square_root_equation_solution_l3010_301012


namespace n_mod_9_eq_6_l3010_301059

def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

theorem n_mod_9_eq_6 : n % 9 = 6 := by
  sorry

end n_mod_9_eq_6_l3010_301059


namespace infinitely_many_square_repetitions_l3010_301085

/-- The number of digits in a natural number -/
def num_digits (a : ℕ) : ℕ := sorry

/-- The repetition of a natural number -/
def repetition (a : ℕ) : ℕ := a * (10^(num_digits a)) + a

/-- There exist infinitely many natural numbers whose repetition is a perfect square -/
theorem infinitely_many_square_repetitions :
  ∀ n : ℕ, ∃ a > n, ∃ k : ℕ, repetition a = k^2 := by sorry

end infinitely_many_square_repetitions_l3010_301085


namespace C_ℝP_subset_Q_l3010_301022

-- Define set P
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}

-- Define set Q
def Q : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the complement of P in ℝ
def C_ℝP : Set ℝ := {y | y ∉ P}

-- Theorem statement
theorem C_ℝP_subset_Q : C_ℝP ⊆ Q := by
  sorry

end C_ℝP_subset_Q_l3010_301022


namespace kitchen_module_cost_is_20000_l3010_301069

/-- Represents the cost of a modular home construction --/
structure ModularHomeCost where
  totalSize : Nat
  kitchenSize : Nat
  bathroomSize : Nat
  bathroomCost : Nat
  otherCost : Nat
  kitchenCount : Nat
  bathroomCount : Nat
  totalCost : Nat

/-- Calculates the cost of the kitchen module --/
def kitchenModuleCost (home : ModularHomeCost) : Nat :=
  let otherSize := home.totalSize - home.kitchenSize * home.kitchenCount - home.bathroomSize * home.bathroomCount
  let otherTotalCost := otherSize * home.otherCost
  let bathroomTotalCost := home.bathroomCost * home.bathroomCount
  home.totalCost - otherTotalCost - bathroomTotalCost

/-- Theorem: The kitchen module costs $20,000 --/
theorem kitchen_module_cost_is_20000 (home : ModularHomeCost) 
  (h1 : home.totalSize = 2000)
  (h2 : home.kitchenSize = 400)
  (h3 : home.bathroomSize = 150)
  (h4 : home.bathroomCost = 12000)
  (h5 : home.otherCost = 100)
  (h6 : home.kitchenCount = 1)
  (h7 : home.bathroomCount = 2)
  (h8 : home.totalCost = 174000) :
  kitchenModuleCost home = 20000 := by
  sorry

end kitchen_module_cost_is_20000_l3010_301069


namespace smallest_part_of_proportional_division_l3010_301048

/-- 
Given a quantity y divided into three parts proportional to 1, 3, and 5,
the smallest part is equal to y/9.
-/
theorem smallest_part_of_proportional_division (y : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    x₁ + x₂ + x₃ = y ∧ 
    x₂ = 3 * x₁ ∧ 
    x₃ = 5 * x₁ ∧ 
    x₁ = y / 9 ∧
    x₁ ≤ x₂ ∧ 
    x₁ ≤ x₃ := by
  sorry

end smallest_part_of_proportional_division_l3010_301048


namespace sum_of_remainders_l3010_301016

theorem sum_of_remainders (d e f : ℕ+) 
  (hd : d ≡ 19 [ZMOD 53])
  (he : e ≡ 33 [ZMOD 53])
  (hf : f ≡ 14 [ZMOD 53]) :
  (d + e + f : ℤ) ≡ 13 [ZMOD 53] := by
  sorry

end sum_of_remainders_l3010_301016


namespace first_group_machines_correct_l3010_301035

/-- The number of machines in the first group -/
def first_group_machines : ℕ := 5

/-- The production rate of the first group (units per machine-hour) -/
def first_group_rate : ℚ := 20 / (first_group_machines * 10)

/-- The production rate of the second group (units per machine-hour) -/
def second_group_rate : ℚ := 180 / (20 * 22.5)

/-- Theorem stating that the number of machines in the first group is correct -/
theorem first_group_machines_correct :
  first_group_rate = second_group_rate ∧
  first_group_machines * first_group_rate * 10 = 20 := by
  sorry

#check first_group_machines_correct

end first_group_machines_correct_l3010_301035


namespace xiaoming_mother_expenses_l3010_301062

/-- Represents a financial transaction with an amount in Yuan -/
structure Transaction where
  amount : Int

/-- Calculates the net result of a list of transactions -/
def netResult (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => acc + t.amount) 0

theorem xiaoming_mother_expenses : 
  let transactions : List Transaction := [
    { amount := 42 },   -- Transfer from Hong
    { amount := -30 },  -- Paying phone bill
    { amount := -51 }   -- Scan QR code for payment
  ]
  netResult transactions = -39 := by
  sorry

end xiaoming_mother_expenses_l3010_301062


namespace airport_distance_l3010_301081

theorem airport_distance (initial_speed initial_time final_speed : ℝ)
  (late_time early_time : ℝ) :
  initial_speed = 40 →
  initial_time = 1 →
  final_speed = 60 →
  late_time = 1.5 →
  early_time = 1 →
  ∃ (total_time total_distance : ℝ),
    total_distance = initial_speed * initial_time +
      final_speed * (total_time - initial_time - early_time) ∧
    total_time = (total_distance / initial_speed) - late_time ∧
    total_distance = 420 :=
by sorry

end airport_distance_l3010_301081


namespace expected_games_value_l3010_301066

/-- The expected number of games in a best-of-seven basketball match -/
def expected_games : ℚ :=
  let p : ℚ := 1 / 2  -- Probability of winning each game
  let prob4 : ℚ := 2 * p^4  -- Probability of ending in 4 games
  let prob5 : ℚ := 2 * 4 * p^4 * (1 - p)  -- Probability of ending in 5 games
  let prob6 : ℚ := 2 * 5 * p^3 * (1 - p)^2  -- Probability of ending in 6 games
  let prob7 : ℚ := 20 * p^3 * (1 - p)^3  -- Probability of ending in 7 games
  4 * prob4 + 5 * prob5 + 6 * prob6 + 7 * prob7

/-- Theorem: The expected number of games in a best-of-seven basketball match
    with equal win probabilities is 93/16 -/
theorem expected_games_value : expected_games = 93 / 16 := by
  sorry

end expected_games_value_l3010_301066


namespace prime_power_sum_perfect_square_l3010_301020

theorem prime_power_sum_perfect_square (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  (∃ n : ℕ, p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = r ∧ ∃ k : ℕ, Prime k ∧ k > 2 ∧ q = k) ∨
   (p = 3 ∧ ((q = 3 ∧ r = 2) ∨ (q = 2 ∧ r = 3)))) :=
by sorry

end prime_power_sum_perfect_square_l3010_301020


namespace grade_assignment_count_l3010_301097

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of available grades -/
def num_grades : ℕ := 4

/-- Theorem stating that the number of ways to assign grades is 4^15 -/
theorem grade_assignment_count :
  (num_grades : ℕ) ^ num_students = 1073741824 := by sorry

end grade_assignment_count_l3010_301097


namespace alternating_sum_equals_neg_151_l3010_301049

/-- The sum of the alternating sequence 1-2+3-4+...+100-101 -/
def alternating_sum : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20 + 21 - 22 + 23 - 24 + 25 - 26 + 27 - 28 + 29 - 30 + 31 - 32 + 33 - 34 + 35 - 36 + 37 - 38 + 39 - 40 + 41 - 42 + 43 - 44 + 45 - 46 + 47 - 48 + 49 - 50 + 51 - 52 + 53 - 54 + 55 - 56 + 57 - 58 + 59 - 60 + 61 - 62 + 63 - 64 + 65 - 66 + 67 - 68 + 69 - 70 + 71 - 72 + 73 - 74 + 75 - 76 + 77 - 78 + 79 - 80 + 81 - 82 + 83 - 84 + 85 - 86 + 87 - 88 + 89 - 90 + 91 - 92 + 93 - 94 + 95 - 96 + 97 - 98 + 99 - 100 + 101

theorem alternating_sum_equals_neg_151 : alternating_sum = -151 := by
  sorry

end alternating_sum_equals_neg_151_l3010_301049


namespace product_of_reciprocals_equals_one_l3010_301054

theorem product_of_reciprocals_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end product_of_reciprocals_equals_one_l3010_301054


namespace complex_subtraction_simplification_l3010_301006

theorem complex_subtraction_simplification :
  (-5 - 3*I : ℂ) - (2 + 6*I) = -7 - 9*I := by sorry

end complex_subtraction_simplification_l3010_301006


namespace teacher_arrangements_eq_144_l3010_301042

/-- The number of ways to arrange 6 teachers (3 math, 2 English, 1 Chinese) such that the 3 math teachers are not adjacent -/
def teacher_arrangements : ℕ :=
  Nat.factorial 3 * (Nat.factorial 3 * Nat.choose 4 3)

theorem teacher_arrangements_eq_144 : teacher_arrangements = 144 := by
  sorry

end teacher_arrangements_eq_144_l3010_301042


namespace total_fabric_needed_l3010_301070

/-- The number of shirts Jenson makes per day -/
def jenson_shirts_per_day : ℕ := 3

/-- The number of pants Kingsley makes per day -/
def kingsley_pants_per_day : ℕ := 5

/-- The number of yards of fabric used for one shirt -/
def fabric_per_shirt : ℕ := 2

/-- The number of yards of fabric used for one pair of pants -/
def fabric_per_pants : ℕ := 5

/-- The number of days to calculate fabric for -/
def days : ℕ := 3

/-- Theorem stating the total yards of fabric needed every 3 days -/
theorem total_fabric_needed : 
  jenson_shirts_per_day * fabric_per_shirt * days + 
  kingsley_pants_per_day * fabric_per_pants * days = 93 := by
  sorry

end total_fabric_needed_l3010_301070


namespace student_count_l3010_301032

/-- Proves the number of students in a class given certain height data -/
theorem student_count (initial_avg : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 175 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg = 173 →
  ∃ n : ℕ, n * actual_avg = n * initial_avg - (incorrect_height - correct_height) ∧ n = 20 := by
  sorry

end student_count_l3010_301032


namespace sum_with_radical_conjugate_l3010_301052

theorem sum_with_radical_conjugate :
  (12 - Real.sqrt 2023) + (12 + Real.sqrt 2023) = 24 := by
  sorry

end sum_with_radical_conjugate_l3010_301052


namespace dave_performance_weeks_l3010_301064

/-- Given that Dave breaks 2 guitar strings per night, performs 6 shows per week,
    and needs to replace 144 guitar strings in total, prove that he performs for 12 weeks. -/
theorem dave_performance_weeks 
  (strings_per_night : ℕ)
  (shows_per_week : ℕ)
  (total_strings : ℕ)
  (h1 : strings_per_night = 2)
  (h2 : shows_per_week = 6)
  (h3 : total_strings = 144) :
  total_strings / (strings_per_night * shows_per_week) = 12 := by
sorry

end dave_performance_weeks_l3010_301064


namespace tangent_line_at_point_A_l3010_301082

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem tangent_line_at_point_A :
  ∃ (m b : ℝ), 
    (f 0 = 16) ∧ 
    (∀ x : ℝ, m * x + b = f' 0 * x + f 0) ∧
    (m = 9 ∧ b = 22) :=
sorry

end tangent_line_at_point_A_l3010_301082


namespace conference_hall_seating_l3010_301058

theorem conference_hall_seating
  (chairs_per_row : ℕ)
  (initial_chairs : ℕ)
  (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : initial_chairs = 195)
  (h3 : expected_participants = 120)
  : ∃ (removed_chairs : ℕ),
    removed_chairs = 75 ∧
    (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧
    initial_chairs - removed_chairs ≥ expected_participants ∧
    initial_chairs - removed_chairs < expected_participants + chairs_per_row :=
by
  sorry

end conference_hall_seating_l3010_301058


namespace sin_difference_range_l3010_301055

theorem sin_difference_range (a : ℝ) : 
  (∃ x : ℝ, Real.sin (x + π/4) - Real.sin (2*x) = a) → 
  -2 ≤ a ∧ a ≤ 9/8 := by
sorry

end sin_difference_range_l3010_301055


namespace repeating_decimal_equals_fraction_l3010_301088

/-- The repeating decimal 0.overline{43} -/
def repeating_decimal : ℚ := 43 / 99

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = 43 / 99 := by sorry

end repeating_decimal_equals_fraction_l3010_301088


namespace bacteria_growth_30_minutes_l3010_301045

/-- The number of bacteria after a given number of 2-minute intervals, 
    given an initial population and a tripling growth rate every 2 minutes. -/
def bacteria_population (initial_population : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * (3 ^ intervals)

/-- Theorem stating that after 15 intervals (30 minutes), 
    an initial population of 30 bacteria will grow to 430467210. -/
theorem bacteria_growth_30_minutes :
  bacteria_population 30 15 = 430467210 := by
  sorry

end bacteria_growth_30_minutes_l3010_301045


namespace linear_function_increasing_l3010_301014

/-- Given a linear function y = 2x + 1 and two points on this function,
    if the x-coordinate of the first point is less than the x-coordinate of the second point,
    then the y-coordinate of the first point is less than the y-coordinate of the second point. -/
theorem linear_function_increasing (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = 2 * x₁ + 1 →
  y₂ = 2 * x₂ + 1 →
  x₁ < x₂ →
  y₁ < y₂ :=
by sorry

end linear_function_increasing_l3010_301014


namespace product_of_primes_l3010_301084

def largest_odd_one_digit_prime : ℕ := 7

def largest_two_digit_prime : ℕ := 97

def second_largest_two_digit_prime : ℕ := 89

theorem product_of_primes : 
  largest_odd_one_digit_prime * largest_two_digit_prime * second_largest_two_digit_prime = 60431 := by
  sorry

end product_of_primes_l3010_301084


namespace tylenol_dosage_l3010_301003

/-- Represents the dosage schedule and total amount of medication taken -/
structure DosageInfo where
  interval : ℕ  -- Time interval between doses in hours
  duration : ℕ  -- Total duration of medication in hours
  tablets_per_dose : ℕ  -- Number of tablets taken per dose
  total_grams : ℕ  -- Total amount of medication taken in grams

/-- Calculates the milligrams per tablet given dosage information -/
def milligrams_per_tablet (info : DosageInfo) : ℕ :=
  let total_milligrams := info.total_grams * 1000
  let num_doses := info.duration / info.interval
  let milligrams_per_dose := total_milligrams / num_doses
  milligrams_per_dose / info.tablets_per_dose

/-- Theorem stating that under the given conditions, each tablet contains 500 milligrams -/
theorem tylenol_dosage (info : DosageInfo) 
  (h1 : info.interval = 4)
  (h2 : info.duration = 12)
  (h3 : info.tablets_per_dose = 2)
  (h4 : info.total_grams = 3) :
  milligrams_per_tablet info = 500 := by
  sorry

end tylenol_dosage_l3010_301003


namespace midpoint_chain_l3010_301043

theorem midpoint_chain (A B C D E F G : ℝ) : 
  C = (A + B) / 2 →  -- C is midpoint of AB
  D = (A + C) / 2 →  -- D is midpoint of AC
  E = (A + D) / 2 →  -- E is midpoint of AD
  F = (A + E) / 2 →  -- F is midpoint of AE
  G = (A + F) / 2 →  -- G is midpoint of AF
  G - A = 2 →        -- AG = 2
  B - A = 64 :=      -- AB = 64
by sorry

end midpoint_chain_l3010_301043


namespace fraction_addition_simplification_l3010_301047

theorem fraction_addition_simplification :
  3 / 462 + 13 / 42 = 73 / 231 := by sorry

end fraction_addition_simplification_l3010_301047


namespace customers_who_tipped_l3010_301011

/-- The number of customers who left a tip at 'The Greasy Spoon' restaurant -/
theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) :
  initial_customers = 29 →
  additional_customers = 20 →
  non_tipping_customers = 34 →
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end customers_who_tipped_l3010_301011


namespace problem_statement_l3010_301009

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end problem_statement_l3010_301009


namespace xy_value_l3010_301010

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 := by
  sorry

end xy_value_l3010_301010


namespace extended_square_counts_l3010_301039

/-- Represents a square configuration with extended sides -/
structure ExtendedSquare where
  /-- Side length of the small square -/
  a : ℝ
  /-- Area of the shaded triangle -/
  S : ℝ
  /-- Condition that S is a quarter of the area of the small square -/
  h_S : S = a^2 / 4

/-- Count of triangles with area 2S in the extended square configuration -/
def count_triangles_2S (sq : ExtendedSquare) : ℕ := 20

/-- Count of squares with area 8S in the extended square configuration -/
def count_squares_8S (sq : ExtendedSquare) : ℕ := 1

/-- Main theorem stating the counts of specific triangles and squares -/
theorem extended_square_counts (sq : ExtendedSquare) :
  count_triangles_2S sq = 20 ∧ count_squares_8S sq = 1 := by
  sorry

end extended_square_counts_l3010_301039


namespace find_A_l3010_301027

theorem find_A : ∃ A : ℕ, A ≥ 1 ∧ A ≤ 9 ∧ (10 * A + 72) - 23 = 549 := by
  sorry

end find_A_l3010_301027


namespace dress_discount_calculation_l3010_301051

def shoe_discount_percent : ℚ := 40 / 100
def original_shoe_price : ℚ := 50
def number_of_shoes : ℕ := 2
def original_dress_price : ℚ := 100
def total_spent : ℚ := 140

theorem dress_discount_calculation :
  let discounted_shoe_price := original_shoe_price * (1 - shoe_discount_percent)
  let total_shoe_cost := discounted_shoe_price * number_of_shoes
  let dress_cost := total_spent - total_shoe_cost
  original_dress_price - dress_cost = 20 := by sorry

end dress_discount_calculation_l3010_301051


namespace candles_remaining_l3010_301029

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def chelsea_fraction : ℚ := 70/100

theorem candles_remaining (total : ℕ) (alyssa_frac chelsea_frac : ℚ) : 
  total - (alyssa_frac * total).floor - (chelsea_frac * (total - (alyssa_frac * total).floor)).floor = 6 :=
by sorry

#check candles_remaining total_candles alyssa_fraction chelsea_fraction

end candles_remaining_l3010_301029


namespace triangle_area_equality_l3010_301038

/-- Given a triangle MNH with points U on MN and C on NH, where:
  MU = s, UN = 6, NC = 20, CH = s, HM = 25,
  and the areas of triangle UNC and quadrilateral MUCH are equal,
  prove that s = 4. -/
theorem triangle_area_equality (s : ℝ) : 
  s > 0 ∧ 
  (1/2 : ℝ) * 6 * 20 = (1/2 : ℝ) * (s + 6) * (s + 20) - (1/2 : ℝ) * 6 * 20 → 
  s = 4 := by
  sorry


end triangle_area_equality_l3010_301038


namespace ratio_of_numbers_l3010_301028

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end ratio_of_numbers_l3010_301028


namespace hardey_fitness_center_ratio_l3010_301040

theorem hardey_fitness_center_ratio 
  (avg_female : ℝ) 
  (avg_male : ℝ) 
  (avg_child : ℝ) 
  (avg_overall : ℝ) 
  (h1 : avg_female = 35)
  (h2 : avg_male = 30)
  (h3 : avg_child = 10)
  (h4 : avg_overall = 25) :
  ∃ (f m c : ℝ), 
    f > 0 ∧ m > 0 ∧ c > 0 ∧
    (avg_female * f + avg_male * m + avg_child * c) / (f + m + c) = avg_overall ∧
    c / (f + m) = 2 / 3 := by
  sorry

end hardey_fitness_center_ratio_l3010_301040


namespace ball_probability_l3010_301031

theorem ball_probability (x : ℕ) : 
  (4 : ℝ) / (4 + x) = (2 : ℝ) / 5 → x = 6 := by
  sorry

end ball_probability_l3010_301031


namespace odd_monotonic_function_conditions_l3010_301068

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- State the theorem
theorem odd_monotonic_function_conditions (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) →  -- f is an odd function
  (∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → f a b c x ≤ f a b c y) →  -- f is monotonic on [1, +∞)
  (a = 0 ∧ c = 0 ∧ b ≤ 3) :=
by sorry

end odd_monotonic_function_conditions_l3010_301068


namespace imaginary_part_implies_a_value_l3010_301067

theorem imaginary_part_implies_a_value (a : ℝ) :
  (Complex.im ((1 - a * Complex.I) / (1 + Complex.I)) = -1) → a = 1 := by
  sorry

end imaginary_part_implies_a_value_l3010_301067


namespace power_calculation_l3010_301036

theorem power_calculation : (10 ^ 6 : ℕ) * (10 ^ 2 : ℕ) ^ 3 / (10 ^ 4 : ℕ) = 10 ^ 8 := by
  sorry

end power_calculation_l3010_301036


namespace work_problem_solution_l3010_301037

/-- Proves that given the conditions of the work problem, c worked for 4 days -/
theorem work_problem_solution :
  let a_days : ℕ := 16
  let b_days : ℕ := 9
  let c_wage : ℚ := 71.15384615384615
  let total_earning : ℚ := 1480
  let wage_ratio_a : ℚ := 3
  let wage_ratio_b : ℚ := 4
  let wage_ratio_c : ℚ := 5
  let a_wage : ℚ := (wage_ratio_a / wage_ratio_c) * c_wage
  let b_wage : ℚ := (wage_ratio_b / wage_ratio_c) * c_wage
  ∃ c_days : ℕ,
    c_days * c_wage + a_days * a_wage + b_days * b_wage = total_earning ∧
    c_days = 4 :=
by sorry

end work_problem_solution_l3010_301037


namespace set_A_properties_l3010_301083

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (2 ∈ A) ∧
  (-2 ∈ A) ∧
  (A = {-2, 2}) ∧
  (∅ ⊆ A) := by
sorry

end set_A_properties_l3010_301083


namespace pentagon_area_fraction_l3010_301013

/-- Represents a rectangle with length 3 times its width -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_eq_3width : length = 3 * width

/-- Represents a pentagon formed by folding the rectangle -/
structure Pentagon where
  original : Rectangle
  area : ℝ

/-- The theorem to be proved -/
theorem pentagon_area_fraction (r : Rectangle) (p : Pentagon) 
  (h : p.original = r) : 
  p.area = (13 / 18) * (r.width * r.length) := by
  sorry

end pentagon_area_fraction_l3010_301013


namespace eighteen_percent_of_700_is_126_l3010_301077

theorem eighteen_percent_of_700_is_126 : (18 / 100) * 700 = 126 := by
  sorry

end eighteen_percent_of_700_is_126_l3010_301077


namespace singleEliminationTournament_l3010_301044

/-- Calculates the number of games required in a single-elimination tournament. -/
def gamesRequired (numTeams : ℕ) : ℕ := numTeams - 1

/-- Theorem: In a single-elimination tournament with 23 teams, 22 games are required to determine the winner. -/
theorem singleEliminationTournament :
  gamesRequired 23 = 22 := by sorry

end singleEliminationTournament_l3010_301044


namespace annas_walking_challenge_l3010_301025

/-- Anna's walking challenge in March -/
theorem annas_walking_challenge 
  (total_days : ℕ) 
  (daily_target : ℝ) 
  (days_passed : ℕ) 
  (distance_walked : ℝ) 
  (h1 : total_days = 31) 
  (h2 : daily_target = 5) 
  (h3 : days_passed = 16) 
  (h4 : distance_walked = 95) : 
  (total_days * daily_target - distance_walked) / (total_days - days_passed) = 4 := by
  sorry

end annas_walking_challenge_l3010_301025


namespace tan_alpha_two_l3010_301002

theorem tan_alpha_two (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 * Real.cos α - Real.cos α ^ 2 - 1) = 1) := by
  sorry

end tan_alpha_two_l3010_301002


namespace cos_A_value_cos_2A_plus_pi_over_4_l3010_301026

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = C ∧ 2 * b = Real.sqrt 3 * a

-- Theorem 1: cos A = 1/3
theorem cos_A_value (A B C : ℝ) (a b c : ℝ) 
  (h : triangle A B C a b c) : Real.cos A = 1 / 3 := by
  sorry

-- Theorem 2: cos(2A + π/4) = -(8 + 7√2)/18
theorem cos_2A_plus_pi_over_4 (A B C : ℝ) (a b c : ℝ) 
  (h : triangle A B C a b c) : 
  Real.cos (2 * A + Real.pi / 4) = -(8 + 7 * Real.sqrt 2) / 18 := by
  sorry

end cos_A_value_cos_2A_plus_pi_over_4_l3010_301026


namespace art_club_theorem_l3010_301017

/-- Represents the number of artworks created by the art club over three school years. -/
def artworks_three_years (initial_students : ℕ) (artworks_per_student : ℕ) 
  (joining_students : ℕ) (leaving_students : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  let students_q1 := initial_students
  let students_q2_q3 := initial_students + joining_students
  let students_q4_q5 := students_q2_q3 - leaving_students
  let artworks_q1 := students_q1 * artworks_per_student
  let artworks_q2_q3 := students_q2_q3 * artworks_per_student
  let artworks_q4_q5 := students_q4_q5 * artworks_per_student
  let artworks_per_year := artworks_q1 + 2 * artworks_q2_q3 + 2 * artworks_q4_q5
  artworks_per_year * years

/-- Represents the number of artworks created in each quarter for the entire club. -/
def artworks_per_quarter (initial_students : ℕ) (artworks_per_student : ℕ) 
  (joining_students : ℕ) (leaving_students : ℕ) : List ℕ :=
  let students_q1 := initial_students
  let students_q2_q3 := initial_students + joining_students
  let students_q4_q5 := students_q2_q3 - leaving_students
  [students_q1 * artworks_per_student,
   students_q2_q3 * artworks_per_student,
   students_q2_q3 * artworks_per_student,
   students_q4_q5 * artworks_per_student,
   students_q4_q5 * artworks_per_student]

theorem art_club_theorem :
  artworks_three_years 30 3 4 6 5 3 = 1386 ∧
  artworks_per_quarter 30 3 4 6 = [90, 102, 102, 84, 84] := by
  sorry

end art_club_theorem_l3010_301017


namespace sum_of_fifth_powers_l3010_301073

theorem sum_of_fifth_powers (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 8) :
  a^5 + b^5 + c^5 = 98/6 := by
  sorry

end sum_of_fifth_powers_l3010_301073


namespace complement_intersection_theorem_l3010_301004

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 5, 6}
def B : Set Nat := {3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3} := by sorry

end complement_intersection_theorem_l3010_301004


namespace min_value_fraction_l3010_301093

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hab : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_fraction_l3010_301093


namespace prob_B_not_occur_expected_value_B_l3010_301030

-- Define the sample space for a single die roll
def Ω : Finset ℕ := Finset.range 6

-- Define events A and B
def A : Finset ℕ := {0, 1, 2}
def B : Finset ℕ := {0, 1, 3}

-- Number of rolls
def n : ℕ := 10

-- Number of times event A occurred
def k : ℕ := 6

-- Probability of event A
def p_A : ℚ := (A.card : ℚ) / Ω.card

-- Probability of event B given A
def p_B_given_A : ℚ := ((A ∩ B).card : ℚ) / A.card

-- Probability of event B given not A
def p_B_given_not_A : ℚ := ((B \ A).card : ℚ) / (Ω \ A).card

-- Theorem for part (a)
theorem prob_B_not_occur (h : k = 6) :
  (Finset.card Ω)^n * (A.card)^k * ((Ω \ A).card)^(n - k) * ((A \ B).card)^k * ((Ω \ (A ∪ B)).card)^(n - k) / 
  (Finset.card Ω)^n / (Finset.card Ω)^n * Nat.choose n k = 64 / 236486 := by sorry

-- Theorem for part (b)
theorem expected_value_B (h : k = 6) :
  k * p_B_given_A + (n - k) * p_B_given_not_A = 16 / 3 := by sorry

end prob_B_not_occur_expected_value_B_l3010_301030


namespace minimum_candies_to_remove_l3010_301034

/-- Represents the number of candies of each flavor in the bag -/
structure CandyBag where
  chocolate : Nat
  mint : Nat
  butterscotch : Nat

/-- The initial state of the candy bag -/
def initialBag : CandyBag := { chocolate := 4, mint := 6, butterscotch := 10 }

/-- The total number of candies in the bag -/
def totalCandies (bag : CandyBag) : Nat :=
  bag.chocolate + bag.mint + bag.butterscotch

/-- Predicate to check if at least two candies of each flavor have been eaten -/
def atLeastTwoEachFlavor (removed : Nat) (bag : CandyBag) : Prop :=
  removed ≥ bag.chocolate - 1 ∧ removed ≥ bag.mint - 1 ∧ removed ≥ bag.butterscotch - 1

theorem minimum_candies_to_remove (bag : CandyBag) :
  totalCandies bag = 20 →
  bag = initialBag →
  ∃ (n : Nat), n = 18 ∧ 
    (∀ (m : Nat), m < n → ¬(atLeastTwoEachFlavor m bag)) ∧
    (atLeastTwoEachFlavor n bag) := by
  sorry

end minimum_candies_to_remove_l3010_301034


namespace nested_sqrt_value_l3010_301000

theorem nested_sqrt_value :
  ∀ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end nested_sqrt_value_l3010_301000


namespace juan_number_problem_l3010_301092

theorem juan_number_problem (n : ℝ) : 
  (2 * ((n + 3)^2) - 2) / 3 = 14 ↔ (n = -3 + Real.sqrt 22 ∨ n = -3 - Real.sqrt 22) :=
by sorry

end juan_number_problem_l3010_301092


namespace congruence_solution_l3010_301087

theorem congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 13258 [MOD 16] := by
  sorry

end congruence_solution_l3010_301087


namespace exponent_value_l3010_301091

theorem exponent_value : ∃ exponent : ℝ,
  (1/5 : ℝ)^35 * (1/4 : ℝ)^exponent = 1 / (2 * (10 : ℝ)^35) ∧ exponent = 17.5 := by
  sorry

end exponent_value_l3010_301091


namespace soccer_team_average_goals_l3010_301071

theorem soccer_team_average_goals (pizzas : ℕ) (slices_per_pizza : ℕ) (games : ℕ)
  (h1 : pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : games = 8) :
  (pizzas * slices_per_pizza) / games = 9 := by
  sorry

end soccer_team_average_goals_l3010_301071


namespace marathon_checkpoints_l3010_301095

/-- Represents a circular marathon with checkpoints -/
structure Marathon where
  total_distance : ℕ
  checkpoint_spacing : ℕ
  distance_to_first : ℕ
  distance_from_last : ℕ

/-- Calculates the number of checkpoints in a marathon -/
def num_checkpoints (m : Marathon) : ℕ :=
  (m.total_distance - m.distance_to_first - m.distance_from_last) / m.checkpoint_spacing + 1

/-- Theorem stating that a marathon with given specifications has 5 checkpoints -/
theorem marathon_checkpoints :
  ∃ (m : Marathon),
    m.total_distance = 26 ∧
    m.checkpoint_spacing = 6 ∧
    m.distance_to_first = 1 ∧
    m.distance_from_last = 1 ∧
    num_checkpoints m = 5 :=
by
  sorry


end marathon_checkpoints_l3010_301095


namespace complex_arithmetic_calculation_l3010_301024

theorem complex_arithmetic_calculation :
  let B : ℂ := 5 - 2*I
  let N : ℂ := -3 + 2*I
  let T : ℂ := 2*I
  let Q : ℝ := 3
  B - N + T - 2 * (Q : ℂ) = 2 - 2*I :=
by sorry

end complex_arithmetic_calculation_l3010_301024


namespace a_perpendicular_b_l3010_301098

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def isPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Vector a in ℝ² -/
def a : ℝ × ℝ := (1, -2)

/-- Vector b in ℝ² -/
def b : ℝ × ℝ := (2, 1)

/-- Theorem stating that vectors a and b are perpendicular -/
theorem a_perpendicular_b : isPerpendicular a b := by
  sorry

end a_perpendicular_b_l3010_301098


namespace consecutive_squares_equality_l3010_301061

theorem consecutive_squares_equality : ∃ x : ℕ+, 
  (x : ℤ)^2 + (x + 1)^2 + (x + 2)^2 + (x + 3)^2 = (x + 4)^2 + (x + 5)^2 + (x + 6)^2 ∧ 
  (x : ℤ)^2 = 441 := by
  sorry

end consecutive_squares_equality_l3010_301061


namespace expression_evaluation_l3010_301057

theorem expression_evaluation :
  let x : ℤ := -1
  (x - 1)^2 - x * (x + 3) + 2 * (x + 2) * (x - 2) = 0 := by
  sorry

end expression_evaluation_l3010_301057


namespace field_length_proof_l3010_301018

theorem field_length_proof (l w : ℝ) (h1 : l = 2 * w) (h2 : (8 * 8) = (1 / 98) * (l * w)) : l = 112 := by
  sorry

end field_length_proof_l3010_301018


namespace more_girls_than_boys_l3010_301050

theorem more_girls_than_boys (total_students : ℕ) 
  (h_total : total_students = 42) 
  (h_ratio : ∃ (x : ℕ), 3 * x + 4 * x = total_students) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    3 * girls = 4 * boys ∧ 
    girls = boys + 6 := by
  sorry

end more_girls_than_boys_l3010_301050


namespace derivative_log2_l3010_301046

-- Define the base-2 logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) := by
  sorry

end derivative_log2_l3010_301046


namespace second_grade_selection_l3010_301096

/-- Represents a stratified sampling scenario in a school -/
structure SchoolSampling where
  first_grade : ℕ
  second_grade : ℕ
  total_selected : ℕ
  first_grade_selected : ℕ

/-- Calculates the number of students selected from the second grade -/
def second_grade_selected (s : SchoolSampling) : ℕ :=
  s.total_selected - s.first_grade_selected

/-- Theorem stating that in the given scenario, 18 students are selected from the second grade -/
theorem second_grade_selection (s : SchoolSampling) 
  (h1 : s.first_grade = 400)
  (h2 : s.second_grade = 360)
  (h3 : s.total_selected = 56)
  (h4 : s.first_grade_selected = 20) :
  second_grade_selected s = 18 := by
  sorry

end second_grade_selection_l3010_301096


namespace extreme_value_implies_a_eq_five_l3010_301056

/-- The function f(x) = x³ + ax² + 3x - 9 has an extreme value at x = -3 -/
def has_extreme_value_at_neg_three (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => x^3 + a*x^2 + 3*x - 9
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x - (-3)| < ε → f x ≤ f (-3) ∨ f x ≥ f (-3)

/-- If f(x) = x³ + ax² + 3x - 9 has an extreme value at x = -3, then a = 5 -/
theorem extreme_value_implies_a_eq_five :
  ∀ (a : ℝ), has_extreme_value_at_neg_three a → a = 5 := by
  sorry

end extreme_value_implies_a_eq_five_l3010_301056


namespace continuous_fraction_value_l3010_301079

theorem continuous_fraction_value : 
  ∃ (x : ℝ), x = 2 + 4 / (1 + 4/x) ∧ x = 4 := by
sorry

end continuous_fraction_value_l3010_301079


namespace tower_remainder_l3010_301033

/-- Represents the number of different towers that can be built with cubes up to size n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+1) => if n ≤ 9 then T n * (min n 4) else T n

/-- The main theorem stating the remainder when T(10) is divided by 1000 -/
theorem tower_remainder : T 10 % 1000 = 216 := by sorry

end tower_remainder_l3010_301033


namespace isosceles_triangle_area_l3010_301089

theorem isosceles_triangle_area (h : ℝ) (p : ℝ) :
  h = 8 →
  p = 32 →
  ∃ (base : ℝ) (leg : ℝ),
    leg + leg + base = p ∧
    h^2 + (base/2)^2 = leg^2 ∧
    (1/2) * base * h = 48 :=
by
  sorry

end isosceles_triangle_area_l3010_301089


namespace doubled_cost_percentage_doubled_cost_percentage_1600_l3010_301008

-- Define the cost function
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percentage (t : ℝ) (b : ℝ) (h : t > 0) (h2 : b > 0) :
  cost t (2 * b) = 16 * cost t b := by
  sorry

-- Corollary to express the result as a percentage
theorem doubled_cost_percentage_1600 (t : ℝ) (b : ℝ) (h : t > 0) (h2 : b > 0) :
  cost t (2 * b) / cost t b = 16 := by
  sorry

end doubled_cost_percentage_doubled_cost_percentage_1600_l3010_301008


namespace friends_total_points_l3010_301065

def total_points (darius_points marius_points matt_points : ℕ) : ℕ :=
  darius_points + marius_points + matt_points

theorem friends_total_points :
  ∀ (darius_points marius_points matt_points : ℕ),
    darius_points = 10 →
    marius_points = darius_points + 3 →
    matt_points = darius_points + 5 →
    total_points darius_points marius_points matt_points = 38 :=
by
  sorry

end friends_total_points_l3010_301065


namespace no_snow_probability_l3010_301015

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 1/2) (h2 : p2 = 2/3) (h3 : p3 = 3/4) (h4 : p4 = 4/5) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1/120 := by
  sorry

end no_snow_probability_l3010_301015


namespace alcohol_percentage_after_dilution_l3010_301041

/-- Calculates the percentage of alcohol in a mixture after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 11)
  (h2 : initial_alcohol_percentage = 42)
  (h3 : water_added = 3) :
  let initial_alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let final_volume := initial_volume + water_added
  let final_alcohol_percentage := (initial_alcohol_volume / final_volume) * 100
  final_alcohol_percentage = 33 := by
  sorry

end alcohol_percentage_after_dilution_l3010_301041


namespace cloth_gain_theorem_l3010_301072

/-- Represents the gain percentage as a rational number -/
def gainPercentage : ℚ := 200 / 3

/-- Represents the number of meters of cloth sold -/
def metersSold : ℕ := 25

/-- Calculates the number of meters of cloth's selling price gained -/
def metersGained (gainPercentage : ℚ) (metersSold : ℕ) : ℚ :=
  (gainPercentage / 100) * metersSold / (1 + gainPercentage / 100)

/-- Theorem stating that the number of meters of cloth's selling price gained is 10 -/
theorem cloth_gain_theorem :
  metersGained gainPercentage metersSold = 10 := by
  sorry

end cloth_gain_theorem_l3010_301072


namespace complex_number_problem_l3010_301099

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
variable (h1 : ∃ (r : ℝ), z + 2*I = r)
variable (h2 : ∃ (t : ℝ), z - 4 = t*I)

-- Define m as a real number
variable (m : ℝ)

-- Define the fourth quadrant condition
def in_fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

-- Theorem statement
theorem complex_number_problem :
  z = 4 - 2*I ∧
  (in_fourth_quadrant ((z + m*I)^2) ↔ -2 < m ∧ m < 2) :=
sorry

end complex_number_problem_l3010_301099


namespace scoops_per_carton_is_ten_l3010_301005

/-- Represents the number of scoops in each carton of ice cream -/
def scoops_per_carton : ℕ := sorry

/-- The total number of cartons -/
def total_cartons : ℕ := 3

/-- The number of scoops Ethan wants -/
def ethan_scoops : ℕ := 2

/-- The number of people who want 2 scoops of chocolate -/
def chocolate_lovers : ℕ := 3

/-- The number of scoops Olivia wants -/
def olivia_scoops : ℕ := 2

/-- The number of scoops Shannon wants (twice as much as Olivia) -/
def shannon_scoops : ℕ := 2 * olivia_scoops

/-- The number of scoops left after everyone has taken their scoops -/
def scoops_left : ℕ := 16

/-- The total number of scoops taken -/
def total_scoops_taken : ℕ := 
  ethan_scoops + (chocolate_lovers * 2) + olivia_scoops + shannon_scoops

/-- Theorem stating that the number of scoops per carton is 10 -/
theorem scoops_per_carton_is_ten : scoops_per_carton = 10 := by
  sorry

end scoops_per_carton_is_ten_l3010_301005


namespace expression_decrease_l3010_301078

theorem expression_decrease (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let original := x^2 * y^3 * z
  let new_x := 0.8 * x
  let new_y := 0.75 * y
  let new_z := 0.9 * z
  let new_expression := new_x^2 * new_y^3 * new_z
  new_expression / original = 0.2414 :=
by sorry

end expression_decrease_l3010_301078


namespace f_g_properties_l3010_301060

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as f'
def g : ℝ → ℝ := f'

-- State the conditions
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorem
theorem f_g_properties : g (-1/2) = 0 ∧ f (-1) = f 4 := by sorry

end f_g_properties_l3010_301060


namespace greatest_integer_a_l3010_301090

theorem greatest_integer_a : ∃ (a : ℤ), 
  (∀ (x : ℤ), (x - a) * (x - 7) + 3 ≠ 0) ∧ 
  (∃ (x : ℤ), (x - 11) * (x - 7) + 3 = 0) ∧
  (∀ (b : ℤ), b > 11 → ∀ (x : ℤ), (x - b) * (x - 7) + 3 ≠ 0) :=
by sorry

end greatest_integer_a_l3010_301090


namespace min_sum_squares_l3010_301063

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by sorry

end min_sum_squares_l3010_301063


namespace constant_term_expansion_l3010_301021

/-- The constant term in the expansion of (x-1)(x^2- 1/x)^6 is -15 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x ≠ 0, f x = (x - 1) * (x^2 - 1/x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) → c = -15 :=
sorry

end constant_term_expansion_l3010_301021


namespace triangle_side_lengths_l3010_301094

/-- Given a triangle with angle α, internal angle bisector length f, and external angle bisector length g,
    calculate the side lengths a, b, and c. -/
theorem triangle_side_lengths
  (α : Real) (f g : ℝ) (h_α : 0 < α ∧ α < π) (h_f : f > 0) (h_g : g > 0) :
  ∃ (a b c : ℝ),
    a = (f * g * Real.sqrt (f^2 + g^2) * Real.sin α) / (g^2 * (Real.cos (α/2))^2 - f^2 * (Real.sin (α/2))^2) ∧
    b = (f * g) / (g * Real.cos (α/2) + f * Real.sin (α/2)) ∧
    c = (f * g) / (g * Real.cos (α/2) - f * Real.sin (α/2)) ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end triangle_side_lengths_l3010_301094


namespace board_number_is_91_l3010_301080

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def does_not_contain_seven (n : ℕ) : Prop :=
  ¬ (∃ d, d ∈ n.digits 10 ∧ d = 7)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem board_number_is_91 
  (n : ℕ) 
  (x : ℕ) 
  (h_consecutive : ∀ i < n, is_two_digit (x / 10^i % 100))
  (h_descending : ∀ i < n - 1, x / 10^i % 100 > x / 10^(i+1) % 100)
  (h_last_digit : does_not_contain_seven (x % 100))
  (h_prime_factors : ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ x = p * q ∧ q = p + 4) :
  x = 91 :=
sorry

end board_number_is_91_l3010_301080


namespace candy_pencils_l3010_301074

/-- Proves that Candy has 9 pencils given the conditions in the problem -/
theorem candy_pencils :
  ∀ (calen_original caleb candy : ℕ),
  calen_original = caleb + 5 →
  caleb = 2 * candy - 3 →
  calen_original - 10 = 10 →
  candy = 9 := by
sorry

end candy_pencils_l3010_301074


namespace initial_number_of_girls_l3010_301001

theorem initial_number_of_girls (initial_boys : ℕ) (boys_dropout : ℕ) (girls_dropout : ℕ) (remaining_students : ℕ) : 
  initial_boys = 14 →
  boys_dropout = 4 →
  girls_dropout = 3 →
  remaining_students = 17 →
  initial_boys - boys_dropout + (initial_girls - girls_dropout) = remaining_students →
  initial_girls = 10 :=
by
  sorry

#check initial_number_of_girls

end initial_number_of_girls_l3010_301001


namespace lauren_subscription_rate_l3010_301023

/-- Represents Lauren's earnings from her social media channel -/
structure Earnings where
  commercialRate : ℚ  -- Rate per commercial view
  commercialViews : ℕ -- Number of commercial views
  subscriptions : ℕ   -- Number of subscriptions
  totalRevenue : ℚ    -- Total revenue
  subscriptionRate : ℚ -- Rate per subscription

/-- Theorem stating that Lauren's subscription rate is $1 -/
theorem lauren_subscription_rate 
  (e : Earnings) 
  (h1 : e.commercialRate = 1/2)      -- $0.50 per commercial view
  (h2 : e.commercialViews = 100)     -- 100 commercial views
  (h3 : e.subscriptions = 27)        -- 27 subscriptions
  (h4 : e.totalRevenue = 77)         -- Total revenue is $77
  : e.subscriptionRate = 1 := by
  sorry


end lauren_subscription_rate_l3010_301023


namespace eight_in_C_l3010_301007

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C := by
  sorry

end eight_in_C_l3010_301007
