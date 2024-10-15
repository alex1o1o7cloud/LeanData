import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_sum_l3048_304830

/-- Given a line with slope 5 passing through the point (2,4), prove that m + b = -1 --/
theorem line_equation_sum (m b : ℝ) : 
  m = 5 →                   -- The slope is 5
  4 = 5 * 2 + b →           -- The line passes through (2,4)
  m + b = -1 :=             -- Prove that m + b = -1
by sorry

end NUMINAMATH_CALUDE_line_equation_sum_l3048_304830


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3048_304834

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a^(x-1) + 4
  f 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3048_304834


namespace NUMINAMATH_CALUDE_seating_arrangements_l3048_304833

/-- The number of seats in the row -/
def num_seats : ℕ := 8

/-- The number of people to be seated -/
def num_people : ℕ := 3

/-- The number of empty seats required between people and at the ends -/
def num_partitions : ℕ := 4

/-- The number of ways to arrange the double empty seat -/
def double_seat_arrangements : ℕ := 4

/-- The number of ways to arrange the people -/
def people_arrangements : ℕ := 6

/-- The total number of seating arrangements -/
def total_arrangements : ℕ := double_seat_arrangements * people_arrangements

theorem seating_arrangements :
  total_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3048_304833


namespace NUMINAMATH_CALUDE_sum_g_formula_l3048_304887

/-- g(n) is the largest odd divisor of the positive integer n -/
def g (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of g(k) for k from 1 to 2^n -/
def sum_g (n : ℕ) : ℕ+ :=
  sorry

/-- Theorem: The sum of g(k) for k from 1 to 2^n equals (4^n + 5) / 3 -/
theorem sum_g_formula (n : ℕ) : 
  (sum_g n : ℚ) = (4^n + 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_formula_l3048_304887


namespace NUMINAMATH_CALUDE_theater_tickets_l3048_304856

theorem theater_tickets (orchestra_price balcony_price : ℕ) 
  (total_tickets total_cost : ℕ) : 
  orchestra_price = 12 →
  balcony_price = 8 →
  total_tickets = 380 →
  total_cost = 3320 →
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost ∧
    balcony_tickets - orchestra_tickets = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_tickets_l3048_304856


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3048_304846

/-- Represents the fishing schedule in a coastal village over three days -/
structure FishingSchedule where
  everyday : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterdayCount : Nat
  todayCount : Nat

/-- Calculates the number of people fishing tomorrow given a FishingSchedule -/
def tomorrowFishers (schedule : FishingSchedule) : Nat :=
  schedule.everyday +
  schedule.everyThreeDay +
  (schedule.everyOtherDay - (schedule.yesterdayCount - schedule.everyday))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow (schedule : FishingSchedule)
  (h1 : schedule.everyday = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterdayCount = 12)
  (h5 : schedule.todayCount = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { everyday := 7, everyOtherDay := 8, everyThreeDay := 3, yesterdayCount := 12, todayCount := 10 }

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3048_304846


namespace NUMINAMATH_CALUDE_integer_root_values_l3048_304843

def polynomial (x b : ℤ) : ℤ := x^4 + 4*x^3 + 2*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-34, -19, -10, -9, -3, 2, 4, 6, 8, 11} :=
sorry

end NUMINAMATH_CALUDE_integer_root_values_l3048_304843


namespace NUMINAMATH_CALUDE_sixth_roll_sum_l3048_304860

/-- Represents the sum of numbers on a single die over 6 rolls -/
def single_die_sum : ℕ := 21

/-- Represents the number of dice -/
def num_dice : ℕ := 6

/-- Represents the sums of the top faces for the first 5 rolls -/
def first_five_rolls : List ℕ := [21, 19, 20, 18, 25]

/-- Theorem: The sum of the top faces on the 6th roll is 23 -/
theorem sixth_roll_sum :
  (num_dice * single_die_sum) - (first_five_rolls.sum) = 23 := by
  sorry

end NUMINAMATH_CALUDE_sixth_roll_sum_l3048_304860


namespace NUMINAMATH_CALUDE_binomial_200_200_l3048_304803

theorem binomial_200_200 : Nat.choose 200 200 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_200_200_l3048_304803


namespace NUMINAMATH_CALUDE_max_leap_years_in_200_years_l3048_304855

/-- 
In a calendrical system where leap years occur every three years without exception,
the maximum number of leap years in a 200-year period is 66.
-/
theorem max_leap_years_in_200_years : 
  ∀ (leap_year_count : ℕ → ℕ),
  (∀ n : ℕ, leap_year_count (3 * n) = n) →
  leap_year_count 200 = 66 := by
sorry

end NUMINAMATH_CALUDE_max_leap_years_in_200_years_l3048_304855


namespace NUMINAMATH_CALUDE_product_digit_count_l3048_304849

def number1 : ℕ := 925743857234987123123
def number2 : ℕ := 10345678909876

theorem product_digit_count : (String.length (toString (number1 * number2))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_count_l3048_304849


namespace NUMINAMATH_CALUDE_power_of_three_equation_l3048_304870

theorem power_of_three_equation (m : ℤ) : 
  3^2001 - 2 * 3^2000 - 3^1999 + 5 * 3^1998 = m * 3^1998 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equation_l3048_304870


namespace NUMINAMATH_CALUDE_percentage_change_equivalence_l3048_304827

theorem percentage_change_equivalence (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) : 
  N * (1 + r/100) * (1 - s/100) < N ↔ r < 50*s / (100 - s) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_equivalence_l3048_304827


namespace NUMINAMATH_CALUDE_college_student_ticket_cost_l3048_304859

/-- Proves that the cost of a college student ticket is $4 given the specified conditions -/
theorem college_student_ticket_cost : 
  ∀ (total_visitors : ℕ) 
    (nyc_resident_ratio : ℚ) 
    (college_student_ratio : ℚ) 
    (total_revenue : ℚ),
  total_visitors = 200 →
  nyc_resident_ratio = 1/2 →
  college_student_ratio = 3/10 →
  total_revenue = 120 →
  (total_visitors : ℚ) * nyc_resident_ratio * college_student_ratio * 4 = total_revenue :=
by
  sorry


end NUMINAMATH_CALUDE_college_student_ticket_cost_l3048_304859


namespace NUMINAMATH_CALUDE_quadratic_equation_equal_roots_l3048_304807

theorem quadratic_equation_equal_roots (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1/4 = 0 ∧
   ∀ y : ℝ, (3 * a - 1) * y^2 - a * y + 1/4 = 0 → y = x) →
  a^2 - 2 * a + 2021 + 1/a = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_equal_roots_l3048_304807


namespace NUMINAMATH_CALUDE_inequality_condition_l3048_304876

theorem inequality_condition (x y : ℝ) : 
  (((x^3 + y^3) / 2)^(1/3) ≥ ((x^2 + y^2) / 2)^(1/2)) ↔ (x + y ≥ 0 ∨ x + y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l3048_304876


namespace NUMINAMATH_CALUDE_area_covered_five_strips_l3048_304889

/-- The area covered by overlapping rectangular strips -/
def area_covered (n : ℕ) (length width : ℝ) (intersection_width : ℝ) : ℝ :=
  n * length * width - (n.choose 2) * 2 * intersection_width^2

/-- Theorem stating the area covered by the specific configuration of strips -/
theorem area_covered_five_strips :
  area_covered 5 15 2 2 = 70 := by sorry

end NUMINAMATH_CALUDE_area_covered_five_strips_l3048_304889


namespace NUMINAMATH_CALUDE_least_integer_for_triangle_with_integer_area_l3048_304836

theorem least_integer_for_triangle_with_integer_area : 
  ∃ (a : ℕ), a > 14 ∧ 
  (∀ b : ℕ, b > 14 ∧ b < a → 
    ¬(∃ A : ℕ, A^2 = (3*b^2/4) * ((b^2/4) - 1))) ∧
  (∃ A : ℕ, A^2 = (3*a^2/4) * ((a^2/4) - 1)) ∧
  a = 52 := by
sorry

end NUMINAMATH_CALUDE_least_integer_for_triangle_with_integer_area_l3048_304836


namespace NUMINAMATH_CALUDE_puppy_cost_puppy_cost_proof_l3048_304847

/-- The cost of a puppy in a pet shop, given the following conditions:
  * There are 2 puppies and 4 kittens in the pet shop.
  * A kitten costs $15.
  * The total stock is worth $100. -/
theorem puppy_cost : ℕ :=
  let num_puppies : ℕ := 2
  let num_kittens : ℕ := 4
  let kitten_cost : ℕ := 15
  let total_stock_value : ℕ := 100
  20

/-- Proof that the cost of a puppy is $20. -/
theorem puppy_cost_proof : puppy_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_puppy_cost_proof_l3048_304847


namespace NUMINAMATH_CALUDE_even_coverings_for_odd_height_l3048_304844

/-- Represents a covering of the lateral surface of a rectangular parallelepiped -/
def Covering (a b c : ℕ) := Unit

/-- Count the number of valid coverings for a rectangular parallelepiped -/
def countCoverings (a b c : ℕ) : ℕ := sorry

/-- Theorem: The number of valid coverings is even when the height is odd -/
theorem even_coverings_for_odd_height (a b c : ℕ) (h : c % 2 = 1) :
  ∃ k : ℕ, countCoverings a b c = 2 * k := by sorry

end NUMINAMATH_CALUDE_even_coverings_for_odd_height_l3048_304844


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3048_304877

theorem arithmetic_calculation : -12 * 5 - (-8 * -4) + (-15 * -6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3048_304877


namespace NUMINAMATH_CALUDE_congruence_problem_l3048_304883

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 25 [ZMOD 42]) (h2 : b ≡ 63 [ZMOD 42]) :
  ∃ n : ℤ, 200 ≤ n ∧ n ≤ 241 ∧ a - b ≡ n [ZMOD 42] ∧ n = 214 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3048_304883


namespace NUMINAMATH_CALUDE_max_students_above_mean_l3048_304817

theorem max_students_above_mean (n : ℕ) (h : n = 107) :
  ∃ (scores : Fin n → ℝ), ∃ (count : ℕ),
    count = (Finset.univ.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n)).card ∧
    count ≤ n - 1 ∧
    ∀ (other_count : ℕ),
      (∃ (other_scores : Fin n → ℝ),
        other_count = (Finset.univ.filter (λ i => other_scores i > (Finset.sum Finset.univ other_scores) / n)).card) →
      other_count ≤ count :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_mean_l3048_304817


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l3048_304845

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 1.2

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l3048_304845


namespace NUMINAMATH_CALUDE_complex_number_properties_l3048_304865

theorem complex_number_properties (z : ℂ) (h : (z - 2*Complex.I)/z = 2 + Complex.I) :
  (∃ (x y : ℝ), z = x + y*Complex.I ∧ y = -1) ∧
  (∀ (z₁ : ℂ), Complex.abs (z₁ - z) = 1 → 
    Real.sqrt 2 - 1 ≤ Complex.abs z₁ ∧ Complex.abs z₁ ≤ Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3048_304865


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3048_304815

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3048_304815


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3048_304879

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
if the distance from a vertex to one of its asymptotes is b/2,
then its eccentricity is 2.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_distance : (b * a) / Real.sqrt (a^2 + b^2) = b / 2) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3048_304879


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3048_304863

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : (ℝ × ℝ)) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

theorem parallel_vectors_k_value
  (a b : ℝ × ℝ)  -- a and b are plane vectors
  (h_not_collinear : ¬ are_parallel a b)  -- a and b are non-collinear
  (m : ℝ × ℝ)
  (h_m : m = (a.1 - 2 * b.1, a.2 - 2 * b.2))  -- m = a - 2b
  (k : ℝ)
  (n : ℝ × ℝ)
  (h_n : n = (3 * a.1 + k * b.1, 3 * a.2 + k * b.2))  -- n = 3a + kb
  (h_parallel : are_parallel m n)  -- m is parallel to n
  : k = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3048_304863


namespace NUMINAMATH_CALUDE_equation_solution_l3048_304892

theorem equation_solution (x : ℝ) : (24 / 36 : ℝ) = Real.sqrt (x / 36) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3048_304892


namespace NUMINAMATH_CALUDE_cyclist_distance_l3048_304866

/-- Represents the distance traveled by a cyclist at a constant speed -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that if a cyclist travels 24 km in 40 minutes at a constant speed, 
    then they will travel 18 km in 30 minutes -/
theorem cyclist_distance 
  (speed : ℝ) 
  (h1 : speed > 0) 
  (h2 : distance_traveled speed (40 / 60) = 24) : 
  distance_traveled speed (30 / 60) = 18 := by
sorry

end NUMINAMATH_CALUDE_cyclist_distance_l3048_304866


namespace NUMINAMATH_CALUDE_diana_wins_prob_l3048_304873

/-- Represents the number of sides on Diana's die -/
def diana_sides : ℕ := 8

/-- Represents the number of sides on Apollo's die -/
def apollo_sides : ℕ := 6

/-- Calculates the probability of Diana rolling higher than Apollo -/
def prob_diana_higher : ℚ :=
  (diana_sides * (diana_sides - 1) - apollo_sides * (apollo_sides - 1)) / (2 * diana_sides * apollo_sides)

/-- Theorem stating that the probability of Diana rolling higher than Apollo is 9/16 -/
theorem diana_wins_prob : prob_diana_higher = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_diana_wins_prob_l3048_304873


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3048_304852

/-- A positive geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ) (q : ℝ) (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  GeometricSequence a q →
  (a m * a n).sqrt = 4 * a 1 →
  a 7 = a 6 + 2 * a 5 →
  (1 : ℝ) / m + 5 / n ≥ 7 / 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3048_304852


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_247_l3048_304894

theorem smallest_n_divisible_by_247 :
  ∀ n : ℕ, n > 0 ∧ n < 37 → ¬(247 ∣ n * (n + 1) * (n + 2)) ∧ (247 ∣ 37 * 38 * 39) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_247_l3048_304894


namespace NUMINAMATH_CALUDE_f_3_minus_f_4_l3048_304897

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_3_minus_f_4 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 5)
  (h_f_1 : f 1 = 1)
  (h_f_2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
sorry

end NUMINAMATH_CALUDE_f_3_minus_f_4_l3048_304897


namespace NUMINAMATH_CALUDE_fish_ratio_proof_l3048_304811

theorem fish_ratio_proof (ken_caught : ℕ) (ken_released : ℕ) (kendra_caught : ℕ) (total_brought_home : ℕ)
  (h1 : ken_released = 3)
  (h2 : kendra_caught = 30)
  (h3 : (ken_caught - ken_released) + kendra_caught = total_brought_home)
  (h4 : total_brought_home = 87) :
  (ken_caught : ℚ) / kendra_caught = 19 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fish_ratio_proof_l3048_304811


namespace NUMINAMATH_CALUDE_jasons_books_count_l3048_304885

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 45

/-- The number of shelves Jason needs -/
def shelves_needed : ℕ := 7

/-- The total number of books Jason has -/
def total_books : ℕ := books_per_shelf * shelves_needed

theorem jasons_books_count : total_books = 315 := by
  sorry

end NUMINAMATH_CALUDE_jasons_books_count_l3048_304885


namespace NUMINAMATH_CALUDE_boxed_flowers_cost_l3048_304872

theorem boxed_flowers_cost (first_batch_total : ℕ) (second_batch_total : ℕ) 
  (second_batch_multiplier : ℕ) (price_difference : ℕ) :
  first_batch_total = 2000 →
  second_batch_total = 4200 →
  second_batch_multiplier = 3 →
  price_difference = 6 →
  ∃ (x : ℕ), 
    x * first_batch_total = second_batch_multiplier * second_batch_total * (x - price_difference) ∧
    x = 20 :=
by sorry

end NUMINAMATH_CALUDE_boxed_flowers_cost_l3048_304872


namespace NUMINAMATH_CALUDE_james_diet_result_l3048_304869

/-- Represents James' food intake and exercise routine --/
structure JamesDiet where
  cheezitBags : ℕ
  cheezitOuncesPerBag : ℕ
  cheezitCaloriesPerOunce : ℕ
  chocolateBars : ℕ
  chocolateBarCalories : ℕ
  popcornCalories : ℕ
  runningMinutes : ℕ
  runningCaloriesPerMinute : ℕ
  swimmingMinutes : ℕ
  swimmingCaloriesPerMinute : ℕ
  cyclingMinutes : ℕ
  cyclingCaloriesPerMinute : ℕ
  caloriesPerPound : ℕ

/-- Calculates the total calories consumed --/
def totalCaloriesConsumed (d : JamesDiet) : ℕ :=
  d.cheezitBags * d.cheezitOuncesPerBag * d.cheezitCaloriesPerOunce +
  d.chocolateBars * d.chocolateBarCalories +
  d.popcornCalories

/-- Calculates the total calories burned --/
def totalCaloriesBurned (d : JamesDiet) : ℕ :=
  d.runningMinutes * d.runningCaloriesPerMinute +
  d.swimmingMinutes * d.swimmingCaloriesPerMinute +
  d.cyclingMinutes * d.cyclingCaloriesPerMinute

/-- Calculates the excess calories --/
def excessCalories (d : JamesDiet) : ℤ :=
  (totalCaloriesConsumed d : ℤ) - (totalCaloriesBurned d : ℤ)

/-- Calculates the potential weight gain in pounds --/
def potentialWeightGain (d : JamesDiet) : ℚ :=
  (excessCalories d : ℚ) / d.caloriesPerPound

/-- Theorem stating James' excess calorie consumption and potential weight gain --/
theorem james_diet_result (d : JamesDiet) 
  (h1 : d.cheezitBags = 3)
  (h2 : d.cheezitOuncesPerBag = 2)
  (h3 : d.cheezitCaloriesPerOunce = 150)
  (h4 : d.chocolateBars = 2)
  (h5 : d.chocolateBarCalories = 250)
  (h6 : d.popcornCalories = 500)
  (h7 : d.runningMinutes = 40)
  (h8 : d.runningCaloriesPerMinute = 12)
  (h9 : d.swimmingMinutes = 30)
  (h10 : d.swimmingCaloriesPerMinute = 15)
  (h11 : d.cyclingMinutes = 20)
  (h12 : d.cyclingCaloriesPerMinute = 10)
  (h13 : d.caloriesPerPound = 3500) :
  excessCalories d = 770 ∧ potentialWeightGain d = 11/50 := by
  sorry

end NUMINAMATH_CALUDE_james_diet_result_l3048_304869


namespace NUMINAMATH_CALUDE_smurf_score_difference_l3048_304818

/-- The number of Smurfs in the village -/
def total_smurfs : ℕ := 45

/-- The number of top and bottom Smurfs with known average scores -/
def known_scores_count : ℕ := 25

/-- The average score of the top 25 Smurfs -/
def top_average : ℚ := 93

/-- The average score of the bottom 25 Smurfs -/
def bottom_average : ℚ := 89

/-- The number of top and bottom Smurfs we're comparing -/
def comparison_count : ℕ := 20

/-- The theorem stating the difference between top and bottom scores -/
theorem smurf_score_difference :
  (top_average * known_scores_count - bottom_average * known_scores_count : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_smurf_score_difference_l3048_304818


namespace NUMINAMATH_CALUDE_product_div_3_probability_l3048_304868

/-- The probability of rolling a number not divisible by 3 on a standard 6-sided die -/
def prob_not_div_3 : ℚ := 2/3

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that the product of the numbers rolled on 'num_dice' standard 6-sided dice is divisible by 3 -/
def prob_product_div_3 : ℚ := 1 - prob_not_div_3 ^ num_dice

theorem product_div_3_probability :
  prob_product_div_3 = 211/243 :=
sorry

end NUMINAMATH_CALUDE_product_div_3_probability_l3048_304868


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3048_304810

/-- The radius of the inscribed circle of a triangle with side lengths 8, 10, and 12 is √7 -/
theorem inscribed_circle_radius (a b c : ℝ) (h_a : a = 8) (h_b : b = 10) (h_c : c = 12) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3048_304810


namespace NUMINAMATH_CALUDE_keyboard_warrior_estimate_l3048_304805

theorem keyboard_warrior_estimate (total_population : ℕ) (sample_size : ℕ) (favorable_count : ℕ) 
  (h1 : total_population = 9600)
  (h2 : sample_size = 50)
  (h3 : favorable_count = 15) :
  (total_population : ℚ) * (1 - favorable_count / sample_size) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_keyboard_warrior_estimate_l3048_304805


namespace NUMINAMATH_CALUDE_roller_coaster_line_length_l3048_304881

theorem roller_coaster_line_length 
  (num_cars : ℕ) 
  (people_per_car : ℕ) 
  (num_runs : ℕ) 
  (h1 : num_cars = 7)
  (h2 : people_per_car = 2)
  (h3 : num_runs = 6) :
  num_cars * people_per_car * num_runs = 84 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_line_length_l3048_304881


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3048_304802

theorem fraction_sum_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a + b) / c = a / c + b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3048_304802


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3048_304813

theorem min_value_quadratic_sum (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) : 
  (a + 1)^2 + 4*b^2 + 9*c^2 ≥ 144/49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3048_304813


namespace NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angles_has_20_sides_l3048_304841

/-- Theorem: A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_with_18_degree_exterior_angles_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angles_has_20_sides_l3048_304841


namespace NUMINAMATH_CALUDE_xiaolins_age_l3048_304839

/-- Represents a person's age as a two-digit number -/
structure TwoDigitAge where
  tens : Nat
  units : Nat
  h_tens : tens < 10
  h_units : units < 10

/-- Swaps the digits of a two-digit age -/
def swapDigits (age : TwoDigitAge) : TwoDigitAge :=
  { tens := age.units,
    units := age.tens,
    h_tens := age.h_units,
    h_units := age.h_tens }

/-- Calculates the numeric value of a two-digit age -/
def toNumber (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.units

theorem xiaolins_age :
  ∀ (grandpa : TwoDigitAge),
    let dad := swapDigits grandpa
    toNumber grandpa - toNumber dad = 5 * 9 →
    9 = 9 := by sorry

end NUMINAMATH_CALUDE_xiaolins_age_l3048_304839


namespace NUMINAMATH_CALUDE_xiaoming_age_is_10_l3048_304857

/-- Xiao Ming's age this year -/
def xiaoming_age : ℕ := sorry

/-- Father's age this year -/
def father_age : ℕ := 4 * xiaoming_age

/-- The sum of their ages 25 years later -/
def sum_ages_25_years_later : ℕ := (xiaoming_age + 25) + (father_age + 25)

theorem xiaoming_age_is_10 :
  xiaoming_age = 10 ∧ father_age = 4 * xiaoming_age ∧ sum_ages_25_years_later = 100 :=
sorry

end NUMINAMATH_CALUDE_xiaoming_age_is_10_l3048_304857


namespace NUMINAMATH_CALUDE_unique_I_value_l3048_304837

def addition_problem (E I G T W O : Nat) : Prop :=
  E ≠ I ∧ E ≠ G ∧ E ≠ T ∧ E ≠ W ∧ E ≠ O ∧
  I ≠ G ∧ I ≠ T ∧ I ≠ W ∧ I ≠ O ∧
  G ≠ T ∧ G ≠ W ∧ G ≠ O ∧
  T ≠ W ∧ T ≠ O ∧
  W ≠ O ∧
  E < 10 ∧ I < 10 ∧ G < 10 ∧ T < 10 ∧ W < 10 ∧ O < 10 ∧
  E = 4 ∧
  G % 2 = 1 ∧
  100 * T + 10 * W + O = 100 * E + 10 * I + G + 100 * E + 10 * I + G

theorem unique_I_value :
  ∀ E I G T W O : Nat,
    addition_problem E I G T W O →
    I = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_I_value_l3048_304837


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3048_304838

-- Define the inverse proportionality relationship
def inverse_proportional (x y : ℝ) := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Theorem statement
theorem inverse_proportion_problem (x y : ℝ → ℝ) :
  (∀ a b : ℝ, inverse_proportional (x a) (y a)) →
  x 2 = 4 →
  x (-3) = -8/3 ∧ x 6 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3048_304838


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_two_l3048_304809

theorem points_four_units_from_negative_two :
  ∀ x : ℝ, |x - (-2)| = 4 ↔ x = 2 ∨ x = -6 := by sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_two_l3048_304809


namespace NUMINAMATH_CALUDE_fraction_equality_l3048_304854

theorem fraction_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3048_304854


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3048_304858

theorem a_minus_b_value (a b : ℝ) : 
  (|a - 2| = 5) → (|b| = 9) → (a + b < 0) → (a - b = 16 ∨ a - b = 6) := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3048_304858


namespace NUMINAMATH_CALUDE_root_difference_equals_2000_l3048_304824

theorem root_difference_equals_2000 : ∃ (a b : ℝ), 
  ((1998 * a)^2 - 1997 * 1999 * a - 1 = 0 ∧ 
   ∀ x, (1998 * x)^2 - 1997 * 1999 * x - 1 = 0 → x ≤ a) ∧
  (b^2 + 1998 * b - 1999 = 0 ∧ 
   ∀ y, y^2 + 1998 * y - 1999 = 0 → b ≤ y) ∧
  a - b = 2000 := by
sorry

end NUMINAMATH_CALUDE_root_difference_equals_2000_l3048_304824


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3048_304840

theorem complex_equation_solution (z : ℂ) : z * Complex.I = Complex.I - 1 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3048_304840


namespace NUMINAMATH_CALUDE_hawks_score_l3048_304829

theorem hawks_score (total_points margin : ℕ) (h1 : total_points = 50) (h2 : margin = 18) :
  (total_points - margin) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l3048_304829


namespace NUMINAMATH_CALUDE_valid_numbers_l3048_304878

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) / 143 = 136

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {9949, 9859, 9769, 9679, 9589, 9499} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3048_304878


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3048_304891

theorem unique_solution_cube_equation :
  ∃! (x : ℕ+), (2 * x.val)^3 - x.val = 726 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3048_304891


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3048_304864

theorem polynomial_simplification (x : ℝ) :
  (15 * x^12 + 8 * x^10 + 11 * x^9) + (5 * x^12 + 3 * x^10 + x^9 + 6 * x^7 + 4 * x^4 + 7 * x^2 + 10) =
  20 * x^12 + 11 * x^10 + 12 * x^9 + 6 * x^7 + 4 * x^4 + 7 * x^2 + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3048_304864


namespace NUMINAMATH_CALUDE_floor_tiles_proof_l3048_304890

/-- Represents the number of tiles in a row given its position -/
def tiles_in_row (n : ℕ) : ℕ := 53 - 2 * (n - 1)

/-- Represents the total number of tiles in the first n rows -/
def total_tiles (n : ℕ) : ℕ := n * (tiles_in_row 1 + tiles_in_row n) / 2

/-- The number of rows in the floor -/
def num_rows : ℕ := 9

theorem floor_tiles_proof :
  (total_tiles num_rows = 405) ∧
  (∀ i : ℕ, i > 0 → i ≤ num_rows → tiles_in_row i > 0) :=
sorry

#eval num_rows

end NUMINAMATH_CALUDE_floor_tiles_proof_l3048_304890


namespace NUMINAMATH_CALUDE_equation_solution_l3048_304800

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  x^2 + 36 / x^2 = 13 ↔ x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3048_304800


namespace NUMINAMATH_CALUDE_middle_school_students_l3048_304808

theorem middle_school_students (band_percentage : ℝ) (band_students : ℕ) 
  (h1 : band_percentage = 0.20)
  (h2 : band_students = 168) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_middle_school_students_l3048_304808


namespace NUMINAMATH_CALUDE_board_transformation_impossibility_l3048_304874

/-- Represents a board state as a list of integers -/
def Board := List Int

/-- Performs one move on the board, pairing integers and replacing with their sum and difference -/
def move (b : Board) : Board :=
  sorry

/-- Checks if a board contains 1000 consecutive integers -/
def isConsecutive1000 (b : Board) : Prop :=
  sorry

/-- Calculates the sum of squares of all integers on the board -/
def sumOfSquares (b : Board) : Int :=
  sorry

theorem board_transformation_impossibility (initial : Board) :
  (sumOfSquares initial) % 8 = 0 →
  ∀ n : Nat, ¬(isConsecutive1000 (n.iterate move initial)) :=
sorry

end NUMINAMATH_CALUDE_board_transformation_impossibility_l3048_304874


namespace NUMINAMATH_CALUDE_saras_quarters_l3048_304875

/-- Given that Sara initially had 783 quarters and now has 1054 quarters,
    prove that the number of quarters Sara's dad gave her is 271. -/
theorem saras_quarters (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 783)
  (h2 : final_quarters = 1054) :
  final_quarters - initial_quarters = 271 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l3048_304875


namespace NUMINAMATH_CALUDE_maria_anna_age_sum_prove_maria_anna_age_sum_l3048_304828

theorem maria_anna_age_sum : ℕ → ℕ → Prop :=
  fun maria_age anna_age =>
    (maria_age = anna_age + 5) →
    (maria_age + 7 = 3 * (anna_age - 3)) →
    (maria_age + anna_age = 27)

#check maria_anna_age_sum

theorem prove_maria_anna_age_sum :
  ∃ (maria_age anna_age : ℕ), maria_anna_age_sum maria_age anna_age :=
by
  sorry

end NUMINAMATH_CALUDE_maria_anna_age_sum_prove_maria_anna_age_sum_l3048_304828


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3048_304812

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3048_304812


namespace NUMINAMATH_CALUDE_direct_proportion_k_value_l3048_304893

def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x

theorem direct_proportion_k_value :
  ∀ k : ℝ,
  (∀ x y : ℝ, y = (k - 1) * x + k^2 - 1) →
  (is_direct_proportion (λ x => (k - 1) * x + k^2 - 1)) →
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_k_value_l3048_304893


namespace NUMINAMATH_CALUDE_hundred_passengers_sixteen_stops_l3048_304871

/-- The number of ways passengers can disembark from a train -/
def ways_to_disembark (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: 100 passengers disembarking at 16 stops results in 16^100 possibilities -/
theorem hundred_passengers_sixteen_stops :
  ways_to_disembark 100 16 = 16^100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_passengers_sixteen_stops_l3048_304871


namespace NUMINAMATH_CALUDE_pure_imaginary_solution_real_sum_solution_l3048_304804

def is_pure_imaginary (z : ℂ) : Prop := ∃ a : ℝ, z = Complex.I * a

theorem pure_imaginary_solution (z : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : Complex.abs (z - 1) = Complex.abs (z - 1 + Complex.I)) : 
  z = Complex.I ∨ z = -Complex.I :=
sorry

theorem real_sum_solution (z : ℂ) 
  (h1 : ∃ r : ℝ, z + 10 / z = r) 
  (h2 : 1 ≤ (z + 10 / z).re ∧ (z + 10 / z).re ≤ 6) :
  z = 1 + 3 * Complex.I ∨ z = 3 + Complex.I ∨ z = 3 - Complex.I :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solution_real_sum_solution_l3048_304804


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3048_304832

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^2 - 22 * X + 63 = (X - 3) * q + 24 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3048_304832


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l3048_304861

theorem circle_tangent_to_x_axis (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*x + 2*b*y + b^2 = 0 → 
   ∃ p : ℝ × ℝ, p.1^2 + p.2^2 = 0 ∧ p.2 = 0) → 
  b = 2 ∨ b = -2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l3048_304861


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l3048_304842

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (base_diameter : ℝ) (h : base_diameter = 12 * Real.sqrt 3) :
  let cone_height : ℝ := base_diameter / 2
  let sphere_radius : ℝ := 3 * Real.sqrt 6 - 3 * Real.sqrt 3
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (4 / 3) * Real.pi * (3 * Real.sqrt 6 - 3 * Real.sqrt 3) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l3048_304842


namespace NUMINAMATH_CALUDE_zeros_in_5000_to_50_l3048_304851

theorem zeros_in_5000_to_50 : ∃ n : ℕ, (5000 ^ 50 : ℕ) = n * (10 ^ 150) ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_5000_to_50_l3048_304851


namespace NUMINAMATH_CALUDE_tangent_curve_intersection_l3048_304880

-- Define the curve C
def C (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

-- Define the point M
def M : ℝ × ℝ := (1, -4)

-- Define the tangent line l
def l (x : ℝ) : ℝ := -12 * (x - 1) - 4

-- Define a function to count common points
def count_common_points (f g : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem tangent_curve_intersection :
  count_common_points C l = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_curve_intersection_l3048_304880


namespace NUMINAMATH_CALUDE_gcd_72_168_l3048_304814

theorem gcd_72_168 : Nat.gcd 72 168 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_72_168_l3048_304814


namespace NUMINAMATH_CALUDE_valid_outfit_count_l3048_304806

/-- Represents the number of shirts, pants, and hats -/
def total_items : ℕ := 8

/-- Represents the number of colors for each item -/
def total_colors : ℕ := 8

/-- Represents the number of colors with matching sets -/
def matching_colors : ℕ := 6

/-- Calculates the total number of outfit combinations -/
def total_combinations : ℕ := total_items * total_items * total_items

/-- Calculates the number of restricted combinations for one pair of matching items -/
def restricted_per_pair : ℕ := matching_colors * total_items

/-- Calculates the total number of restricted combinations -/
def total_restricted : ℕ := 3 * restricted_per_pair

/-- Represents the number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - total_restricted

theorem valid_outfit_count : valid_outfits = 368 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l3048_304806


namespace NUMINAMATH_CALUDE_range_of_a_l3048_304884

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (1 ∉ A a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3048_304884


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3048_304848

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a + 1) * x + a * b > 0 ↔ x < -1 ∨ x > 4) →
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3048_304848


namespace NUMINAMATH_CALUDE_total_collection_l3048_304853

/-- A group of students collecting money -/
structure StudentGroup where
  members : ℕ
  contribution_per_member : ℕ
  total_paise : ℕ

/-- Conversion rate from paise to rupees -/
def paise_to_rupees (paise : ℕ) : ℚ :=
  (paise : ℚ) / 100

/-- Theorem stating the total amount collected by the group -/
theorem total_collection (group : StudentGroup) 
    (h1 : group.members = 54)
    (h2 : group.contribution_per_member = group.members)
    (h3 : group.total_paise = group.members * group.contribution_per_member) : 
    paise_to_rupees group.total_paise = 29.16 := by
  sorry

end NUMINAMATH_CALUDE_total_collection_l3048_304853


namespace NUMINAMATH_CALUDE_stratified_sampling_l3048_304862

theorem stratified_sampling (total_students : ℕ) (class1_students : ℕ) (class2_students : ℕ) 
  (sample_size : ℕ) (h1 : total_students = class1_students + class2_students) 
  (h2 : total_students = 96) (h3 : class1_students = 54) (h4 : class2_students = 42) 
  (h5 : sample_size = 16) :
  (class1_students * sample_size / total_students : ℚ) = 9 ∧
  (class2_students * sample_size / total_students : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3048_304862


namespace NUMINAMATH_CALUDE_probability_of_perfect_square_sum_l3048_304825

/-- The number of faces on a standard die -/
def standardDieFaces : ℕ := 6

/-- The set of possible sums when rolling two dice -/
def possibleSums : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

/-- The set of perfect squares within the possible sums -/
def perfectSquareSums : Set ℕ := {4, 9}

/-- The number of ways to get a sum of 4 -/
def waysToGetFour : ℕ := 3

/-- The number of ways to get a sum of 9 -/
def waysToGetNine : ℕ := 4

/-- The total number of favorable outcomes -/
def favorableOutcomes : ℕ := waysToGetFour + waysToGetNine

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := standardDieFaces * standardDieFaces

/-- Theorem: The probability of rolling two standard 6-sided dice and getting a sum that is a perfect square is 7/36 -/
theorem probability_of_perfect_square_sum :
  (favorableOutcomes : ℚ) / totalOutcomes = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_perfect_square_sum_l3048_304825


namespace NUMINAMATH_CALUDE_lasagna_cost_l3048_304867

def cheese_quantity : Real := 1.5
def meat_quantity : Real := 0.550
def pasta_quantity : Real := 0.280
def tomatoes_quantity : Real := 2.2

def cheese_price : Real := 6.30
def meat_price : Real := 8.55
def pasta_price : Real := 2.40
def tomatoes_price : Real := 1.79

def cheese_tax : Real := 0.07
def meat_tax : Real := 0.06
def pasta_tax : Real := 0.08
def tomatoes_tax : Real := 0.05

def total_cost (cq mq pq tq : Real) (cp mp pp tp : Real) (ct mt pt tt : Real) : Real :=
  (cq * cp * (1 + ct)) + (mq * mp * (1 + mt)) + (pq * pp * (1 + pt)) + (tq * tp * (1 + tt))

theorem lasagna_cost :
  total_cost cheese_quantity meat_quantity pasta_quantity tomatoes_quantity
              cheese_price meat_price pasta_price tomatoes_price
              cheese_tax meat_tax pasta_tax tomatoes_tax = 19.9568 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_cost_l3048_304867


namespace NUMINAMATH_CALUDE_min_distinct_values_l3048_304816

theorem min_distinct_values (total : ℕ) (mode_count : ℕ) (h1 : total = 2023) (h2 : mode_count = 15) :
  ∃ (distinct : ℕ), distinct = 145 ∧ 
  (∀ d : ℕ, d < 145 → 
    ¬∃ (l : List ℕ), l.length = total ∧ 
    (∃! x : ℕ, x ∈ l ∧ l.count x = mode_count) ∧
    l.toFinset.card = d) :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l3048_304816


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3048_304899

theorem cyclic_sum_inequality (a b c : ℝ) (n : ℕ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1) :
  (a + b + c^n) / (a^(2*n+3) + b^(2*n+3) + a*b) +
  (b + c + a^n) / (b^(2*n+3) + c^(2*n+3) + b*c) +
  (c + a + b^n) / (c^(2*n+3) + a^(2*n+3) + c*a) ≤
  a^(n+1) + b^(n+1) + c^(n+1) := by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3048_304899


namespace NUMINAMATH_CALUDE_computer_price_increase_l3048_304831

theorem computer_price_increase (x : ℝ) (h : 2 * x = 540) : 
  (351 - x) / x * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3048_304831


namespace NUMINAMATH_CALUDE_kid_ticket_price_l3048_304819

theorem kid_ticket_price 
  (total_sales : ℕ) 
  (adult_price : ℕ) 
  (num_adults : ℕ) 
  (total_people : ℕ) : 
  total_sales = 3864 ∧ 
  adult_price = 28 ∧ 
  num_adults = 51 ∧ 
  total_people = 254 → 
  (total_sales - num_adults * adult_price) / (total_people - num_adults) = 12 := by
  sorry

#eval (3864 - 51 * 28) / (254 - 51)

end NUMINAMATH_CALUDE_kid_ticket_price_l3048_304819


namespace NUMINAMATH_CALUDE_equation_transformation_l3048_304823

theorem equation_transformation (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) + 3 = 3 * x / (1 - x) → 1 + 3 * (x - 1) = -3 * x := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3048_304823


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3048_304895

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + a - 2024 = 0) → (b^2 + b - 2024 = 0) → (a^2 + 2*a + b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3048_304895


namespace NUMINAMATH_CALUDE_exists_four_digit_divisible_by_23_digit_sum_23_l3048_304820

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Proposition: There exists a four-digit number divisible by 23 with digit sum 23 -/
theorem exists_four_digit_divisible_by_23_digit_sum_23 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ digit_sum n = 23 ∧ n % 23 = 0 := by
sorry

end NUMINAMATH_CALUDE_exists_four_digit_divisible_by_23_digit_sum_23_l3048_304820


namespace NUMINAMATH_CALUDE_profit_maximized_at_six_l3048_304822

/-- Sales revenue function -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Total production cost function -/
def total_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit function -/
def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

/-- The production quantity that maximizes profit -/
def optimal_quantity : ℝ := 6

theorem profit_maximized_at_six :
  ∀ x > 0, profit x ≤ profit optimal_quantity :=
by sorry

end NUMINAMATH_CALUDE_profit_maximized_at_six_l3048_304822


namespace NUMINAMATH_CALUDE_average_equation_l3048_304898

theorem average_equation (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 4) + (4*x + 6) + (5*x + 3)) = 3*x + 5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l3048_304898


namespace NUMINAMATH_CALUDE_simplify_expression_l3048_304896

theorem simplify_expression (x : ℝ) : 2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 8) = 8*x^3 - 4*x^2 + 6*x - 32 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3048_304896


namespace NUMINAMATH_CALUDE_roger_trays_first_table_l3048_304886

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := 4

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The number of trays Roger picked up from the second table -/
def trays_from_second_table : ℕ := 2

/-- The number of trays Roger picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * num_trips - trays_from_second_table

theorem roger_trays_first_table :
  trays_from_first_table = 10 := by sorry

end NUMINAMATH_CALUDE_roger_trays_first_table_l3048_304886


namespace NUMINAMATH_CALUDE_algae_growth_l3048_304821

/-- Calculates the population of algae after a given time period. -/
def algaePopulation (initialPopulation : ℕ) (minutes : ℕ) : ℕ :=
  initialPopulation * 2^(minutes / 5)

/-- Theorem stating that the algae population grows from 50 to 6400 in 35 minutes. -/
theorem algae_growth :
  algaePopulation 50 35 = 6400 :=
by
  sorry

#eval algaePopulation 50 35

end NUMINAMATH_CALUDE_algae_growth_l3048_304821


namespace NUMINAMATH_CALUDE_complex_simplification_l3048_304850

/-- Proof of complex number simplification -/
theorem complex_simplification :
  let i : ℂ := Complex.I
  (3 + 5*i) / (-2 + 7*i) = 29/53 - (31/53)*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3048_304850


namespace NUMINAMATH_CALUDE_vector_operation_l3048_304801

/-- Given plane vectors a and b, prove that -2a - b equals (-3, -1) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (-2 : ℝ) • a - b = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3048_304801


namespace NUMINAMATH_CALUDE_consecutive_numbers_multiple_l3048_304882

/-- Given three consecutive numbers where the first is 4.2, 
    prove that the multiple of the first number that satisfies 
    the equation is 9. -/
theorem consecutive_numbers_multiple (m : ℝ) : 
  m * 4.2 = 2 * (4.2 + 4) + 2 * (4.2 + 2) + 9 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_multiple_l3048_304882


namespace NUMINAMATH_CALUDE_distance_between_points_l3048_304888

/-- The distance between the points (3, -2) and (10, 8) is √149 units. -/
theorem distance_between_points : Real.sqrt 149 = Real.sqrt ((10 - 3)^2 + (8 - (-2))^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3048_304888


namespace NUMINAMATH_CALUDE_sidney_thursday_jumping_jacks_l3048_304826

/-- The number of jumping jacks Sidney did on Thursday -/
def thursday_jumping_jacks : ℕ := by sorry

/-- The total number of jumping jacks Sidney did from Monday to Wednesday -/
def monday_to_wednesday : ℕ := 20 + 36 + 40

/-- The total number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

theorem sidney_thursday_jumping_jacks :
  thursday_jumping_jacks = 50 :=
by
  have sidney_total : ℕ := brooke_total / 3
  have h1 : sidney_total = monday_to_wednesday + thursday_jumping_jacks := by sorry
  sorry


end NUMINAMATH_CALUDE_sidney_thursday_jumping_jacks_l3048_304826


namespace NUMINAMATH_CALUDE_angle_C_measure_l3048_304835

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem angle_C_measure (abc : Triangle) (h : abc.A + abc.B = 80) : abc.C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l3048_304835
