import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l65_6567

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c

theorem sum_of_roots_zero (Q : ℝ → ℝ) 
  (h_quad : is_quadratic Q)
  (h_ineq : ∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) :
  ∃ r₁ r₂ : ℝ, (∀ x, Q x = 0 ↔ x = r₁ ∨ x = r₂) ∧ r₁ + r₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l65_6567


namespace NUMINAMATH_CALUDE_calculate_expression_l65_6524

theorem calculate_expression : (π - 2023)^0 + |-9| - 3^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l65_6524


namespace NUMINAMATH_CALUDE_hanks_total_reading_time_l65_6515

/-- Represents Hank's weekly reading schedule --/
structure ReadingSchedule where
  newspaper_days : Nat
  newspaper_time : Nat
  magazine_time : Nat
  novel_days : Nat
  novel_time : Nat
  novel_friday_time : Nat
  novel_saturday_multiplier : Nat
  novel_sunday_multiplier : Nat
  scientific_journal_time : Nat
  nonfiction_time : Nat

/-- Calculates the total reading time for a week given a reading schedule --/
def total_reading_time (schedule : ReadingSchedule) : Nat :=
  let newspaper_total := schedule.newspaper_days * schedule.newspaper_time
  let magazine_total := schedule.magazine_time
  let novel_weekday_total := (schedule.novel_days - 3) * schedule.novel_time
  let novel_friday_total := schedule.novel_friday_time
  let novel_saturday_total := schedule.novel_time * schedule.novel_saturday_multiplier
  let novel_sunday_total := schedule.novel_time * schedule.novel_sunday_multiplier
  let scientific_journal_total := schedule.scientific_journal_time
  let nonfiction_total := schedule.nonfiction_time
  newspaper_total + magazine_total + novel_weekday_total + novel_friday_total +
  novel_saturday_total + novel_sunday_total + scientific_journal_total + nonfiction_total

/-- Hank's actual reading schedule --/
def hanks_schedule : ReadingSchedule :=
  { newspaper_days := 5
  , newspaper_time := 30
  , magazine_time := 15
  , novel_days := 5
  , novel_time := 60
  , novel_friday_time := 90
  , novel_saturday_multiplier := 2
  , novel_sunday_multiplier := 3
  , scientific_journal_time := 45
  , nonfiction_time := 40
  }

/-- Theorem stating that Hank's total reading time in a week is 760 minutes --/
theorem hanks_total_reading_time :
  total_reading_time hanks_schedule = 760 := by
  sorry

end NUMINAMATH_CALUDE_hanks_total_reading_time_l65_6515


namespace NUMINAMATH_CALUDE_exists_strictly_convex_function_with_constraints_l65_6566

-- Define the function type
def StrictlyConvexFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    f x₁ + f x₂ < 2 * f ((x₁ + x₂) / 2)

-- Main theorem
theorem exists_strictly_convex_function_with_constraints : 
  ∃ f : ℝ → ℝ,
    (∀ x : ℝ, x > 0 → f x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 4 → ∃ x : ℝ, x > 0 ∧ f x = y) ∧
    StrictlyConvexFunction f :=
sorry

end NUMINAMATH_CALUDE_exists_strictly_convex_function_with_constraints_l65_6566


namespace NUMINAMATH_CALUDE_solve_y_equation_l65_6538

theorem solve_y_equation : ∃ y : ℚ, (3 * y) / 7 = 21 ∧ y = 49 := by
  sorry

end NUMINAMATH_CALUDE_solve_y_equation_l65_6538


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_four_satisfies_inequality_four_is_smallest_l65_6550

theorem smallest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 36 ≤ 0 → n ≥ 4 :=
by sorry

theorem four_satisfies_inequality :
  4^2 - 13*4 + 36 ≤ 0 :=
by sorry

theorem four_is_smallest :
  ∀ n : ℤ, n < 4 → n^2 - 13*n + 36 > 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_four_satisfies_inequality_four_is_smallest_l65_6550


namespace NUMINAMATH_CALUDE_circle_radius_in_square_with_semicircles_l65_6501

/-- Given a square with side length 108 and semicircles constructed inward on two adjacent sides,
    the radius of a circle touching one side and both semicircles is 27. -/
theorem circle_radius_in_square_with_semicircles (square_side : ℝ) 
  (h_side : square_side = 108) : ∃ (r : ℝ), r = 27 ∧ 
  r + (square_side / 2) = square_side - r := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_square_with_semicircles_l65_6501


namespace NUMINAMATH_CALUDE_taeyeon_height_l65_6534

theorem taeyeon_height (seonghee_height : ℝ) (taeyeon_ratio : ℝ) :
  seonghee_height = 134.5 →
  taeyeon_ratio = 1.06 →
  taeyeon_ratio * seonghee_height = 142.57 := by
  sorry

end NUMINAMATH_CALUDE_taeyeon_height_l65_6534


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l65_6537

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 15 + 24) / 3
  let mean2 := (18 + z) / 2
  mean1 = mean2 → z = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l65_6537


namespace NUMINAMATH_CALUDE_share_premium_percentage_l65_6517

/-- Calculates the premium percentage on shares given investment details -/
theorem share_premium_percentage
  (total_investment : ℝ)
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : total_investment = 14400)
  (h2 : face_value = 100)
  (h3 : dividend_rate = 0.07)
  (h4 : total_dividend = 840) :
  (total_investment / (total_dividend / (dividend_rate * face_value)) - face_value) / face_value * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_share_premium_percentage_l65_6517


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l65_6572

/-- The function f(x) = e^x - ln(x+m) is monotonically increasing on [0,1] iff m ≥ 1 -/
theorem monotone_increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, MonotoneOn (fun x => Real.exp x - Real.log (x + m)) (Set.Icc 0 1)) ↔ m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l65_6572


namespace NUMINAMATH_CALUDE_function_properties_l65_6599

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 8)*x

/-- The theorem stating the properties of the function and the results to be proved -/
theorem function_properties (a m : ℝ) :
  (∀ x, f a x ≤ 5 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∀ x, f a x ≥ m^2 - 4*m - 9) →
  a = 2 ∧ -1 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_function_properties_l65_6599


namespace NUMINAMATH_CALUDE_quadratic_root_sum_inverse_cubes_l65_6551

theorem quadratic_root_sum_inverse_cubes 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * r^2 + b * r + c = 0)
  (h3 : a * s^2 + b * s + c = 0)
  (h4 : r ≠ s)
  (h5 : a + b + c = 0) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3*a^2 + 3*a*b) / (a + b)^3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_inverse_cubes_l65_6551


namespace NUMINAMATH_CALUDE_probability_in_specific_rectangle_l65_6539

/-- A rectangle in 2D space --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The probability that a randomly selected point in the rectangle is closer to one point than another --/
def probability_closer_to_point (r : Rectangle) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem statement --/
theorem probability_in_specific_rectangle : 
  let r : Rectangle := { x1 := 0, y1 := 0, x2 := 3, y2 := 2 }
  probability_closer_to_point r (0, 0) (4, 2) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_specific_rectangle_l65_6539


namespace NUMINAMATH_CALUDE_billy_apple_ratio_l65_6525

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := 9

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := 4 * friday_apples

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := total_apples - (monday_apples + wednesday_apples + thursday_apples + friday_apples)

/-- The ratio of apples eaten on Tuesday to Monday -/
def tuesday_to_monday_ratio : ℚ := tuesday_apples / monday_apples

theorem billy_apple_ratio : tuesday_to_monday_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_apple_ratio_l65_6525


namespace NUMINAMATH_CALUDE_dave_trips_l65_6511

/-- The number of trips Dave needs to make to carry all trays -/
def number_of_trips (trays_per_trip : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ) : ℕ :=
  (trays_table1 + trays_table2 + trays_per_trip - 1) / trays_per_trip

theorem dave_trips :
  number_of_trips 9 17 55 = 8 :=
by sorry

end NUMINAMATH_CALUDE_dave_trips_l65_6511


namespace NUMINAMATH_CALUDE_problem_solution_l65_6582

theorem problem_solution (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l65_6582


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l65_6512

theorem geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/5)
  (h2 : S = 100)
  (h3 : S = a / (1 - r)) :
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l65_6512


namespace NUMINAMATH_CALUDE_special_function_at_50_l65_6504

/-- A function satisfying f(xy) = xf(y) for all real x and y, and f(1) = 10 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y) ∧ (f 1 = 10)

/-- Theorem: If f is a special function, then f(50) = 500 -/
theorem special_function_at_50 (f : ℝ → ℝ) (h : special_function f) : f 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_50_l65_6504


namespace NUMINAMATH_CALUDE_range_of_a_l65_6526

/-- A function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 15 - 2*a

/-- Predicate to check if there are exactly two positive integers in an open interval -/
def exactly_two_positive_integers (lower upper : ℝ) : Prop :=
  ∃ (n m : ℕ), n < m ∧ 
    (∀ (k : ℕ), lower < k ∧ k < upper ↔ k = n ∨ k = m)

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
    exactly_two_positive_integers x₁ x₂) →
  (31/10 < a ∧ a ≤ 19/6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l65_6526


namespace NUMINAMATH_CALUDE_repeating_decimal_47_equals_fraction_sum_of_numerator_and_denominator_l65_6546

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

theorem repeating_decimal_47_equals_fraction :
  RepeatingDecimal 4 7 = 47 / 99 :=
sorry

theorem sum_of_numerator_and_denominator :
  (47 : ℕ) + 99 = 146 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_equals_fraction_sum_of_numerator_and_denominator_l65_6546


namespace NUMINAMATH_CALUDE_average_bag_weight_l65_6544

def bag_weights : List ℕ := [25, 30, 31, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 48]

theorem average_bag_weight :
  (bag_weights.sum : ℚ) / bag_weights.length = 71/2 := by sorry

end NUMINAMATH_CALUDE_average_bag_weight_l65_6544


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l65_6540

theorem sqrt_sum_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l65_6540


namespace NUMINAMATH_CALUDE_inverse_variation_l65_6520

/-- Given that a and b vary inversely, prove that when a = 800 and b = 0.5, 
    then b = 0.125 when a = 3200 -/
theorem inverse_variation (a b : ℝ) (h : a * b = 800 * 0.5) :
  3200 * (1 / 8) = 800 * 0.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l65_6520


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l65_6542

/-- Represents the current age of the son -/
def sonAge : ℕ := 22

/-- Represents the age difference between the man and his son -/
def ageDifference : ℕ := 24

/-- Represents the number of years until the man's age is twice his son's age -/
def yearsUntilTwice : ℕ := 2

/-- Theorem stating that in 'yearsUntilTwice' years, the man's age will be twice his son's age -/
theorem mans_age_twice_sons_age :
  (sonAge + ageDifference + yearsUntilTwice) = 2 * (sonAge + yearsUntilTwice) := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l65_6542


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l65_6507

theorem max_sum_of_roots (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ 3 * Real.sqrt 3 ∧
  (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = 3 * Real.sqrt 3 ↔ 
    x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l65_6507


namespace NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l65_6583

/-- A function that checks if a number is a sum of consecutive integers -/
def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (start k : ℕ), k ≥ 2 ∧ n = (k * (2 * start + k + 1)) / 2

/-- A function that checks if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- Theorem stating that a positive integer is a sum of two or more consecutive
    positive integers if and only if it is not a power of 2 -/
theorem sum_of_consecutive_iff_not_power_of_two (n : ℕ) (h : n > 0) :
  is_sum_of_consecutive n ↔ ¬ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l65_6583


namespace NUMINAMATH_CALUDE_rowing_round_trip_time_l65_6581

/-- Calculates the total time for a round trip rowing journey given the rowing speed, current speed, and distance. -/
theorem rowing_round_trip_time 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : distance = 24) : 
  (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 5 := by
  sorry

#check rowing_round_trip_time

end NUMINAMATH_CALUDE_rowing_round_trip_time_l65_6581


namespace NUMINAMATH_CALUDE_pie_eating_contest_l65_6590

/-- The amount of pie Erik ate -/
def erik_pie : ℚ := 0.6666666666666666

/-- The amount of pie Frank ate -/
def frank_pie : ℚ := 0.3333333333333333

/-- The difference between Erik's and Frank's pie consumption -/
def pie_difference : ℚ := erik_pie - frank_pie

theorem pie_eating_contest :
  pie_difference = 0.3333333333333333 := by sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l65_6590


namespace NUMINAMATH_CALUDE_radio_cost_price_l65_6548

theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 1430 →
  loss_percentage = 20.555555555555554 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  cost_price = 1800 := by
sorry

end NUMINAMATH_CALUDE_radio_cost_price_l65_6548


namespace NUMINAMATH_CALUDE_equation_solution_l65_6553

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 78 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l65_6553


namespace NUMINAMATH_CALUDE_negative_expression_l65_6557

theorem negative_expression : 
  (|(-4)| > 0) ∧ (-(-4) > 0) ∧ ((-4)^2 > 0) ∧ (-4^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negative_expression_l65_6557


namespace NUMINAMATH_CALUDE_tangent_line_correct_l65_6516

-- Define the curve
def curve (x : ℝ) : ℝ := -x^2 + 6*x

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := -2*x + 6

-- Define the proposed tangent line
def tangent_line (x : ℝ) : ℝ := 6*x

theorem tangent_line_correct :
  -- The tangent line passes through the origin
  tangent_line 0 = 0 ∧
  -- The tangent line touches the curve at some point
  ∃ x : ℝ, curve x = tangent_line x ∧
  -- The slope of the tangent line equals the derivative of the curve at the point of tangency
  curve_derivative x = 6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_correct_l65_6516


namespace NUMINAMATH_CALUDE_distribute_five_books_three_people_l65_6506

/-- The number of ways to distribute books among people -/
def distribute_books (n_books : ℕ) (n_people : ℕ) (min_books : ℕ) (max_books : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to distribute 5 books among 3 people -/
theorem distribute_five_books_three_people : 
  distribute_books 5 3 1 2 = 90 := by sorry

end NUMINAMATH_CALUDE_distribute_five_books_three_people_l65_6506


namespace NUMINAMATH_CALUDE_equation_solution_l65_6564

theorem equation_solution : ∃! x : ℚ, (x^2 + 3*x + 5) / (x + 6) = x + 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l65_6564


namespace NUMINAMATH_CALUDE_sin_square_plus_sin_minus_one_range_l65_6569

theorem sin_square_plus_sin_minus_one_range :
  ∀ x : ℝ, -5/4 ≤ Real.sin x ^ 2 + Real.sin x - 1 ∧ Real.sin x ^ 2 + Real.sin x - 1 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_square_plus_sin_minus_one_range_l65_6569


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l65_6586

/-- Given a rectangle with initial dimensions 5 × 7 inches, if reducing one side by 2 inches
    results in an area of 21 square inches, then reducing the other side by 2 inches
    will result in an area of 25 square inches. -/
theorem rectangle_area_reduction (initial_width initial_length : ℝ)
  (h_initial_width : initial_width = 5)
  (h_initial_length : initial_length = 7)
  (h_reduced_area : (initial_width - 2) * initial_length = 21) :
  initial_width * (initial_length - 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l65_6586


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l65_6508

theorem complex_power_magnitude : 
  Complex.abs ((2/3 : ℂ) + (5/6 : ℂ) * Complex.I)^8 = 2825761/1679616 := by
sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l65_6508


namespace NUMINAMATH_CALUDE_polynomial_maximum_l65_6565

/-- The polynomial function we're analyzing -/
def f (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 12

/-- The maximum value of the polynomial -/
def max_value : ℝ := 15

/-- The x-value at which the maximum occurs -/
def max_point : ℝ := -1

theorem polynomial_maximum :
  (∀ x : ℝ, f x ≤ max_value) ∧ f max_point = max_value :=
sorry

end NUMINAMATH_CALUDE_polynomial_maximum_l65_6565


namespace NUMINAMATH_CALUDE_min_coins_for_distribution_l65_6535

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 95) :
  min_additional_coins num_friends initial_coins = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_for_distribution_l65_6535


namespace NUMINAMATH_CALUDE_unique_k_value_l65_6519

theorem unique_k_value : ∃! k : ℝ, ∀ x : ℝ, x * (2 * x + 3) < k ↔ -5/2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_value_l65_6519


namespace NUMINAMATH_CALUDE_emptyBoxes_l65_6523

/-- Represents the state of the two boxes with pebbles -/
structure BoxState :=
  (p : ℕ)
  (q : ℕ)

/-- Defines a single step operation on the boxes -/
inductive Step
  | Remove : Step
  | TripleP : Step
  | TripleQ : Step

/-- Applies a single step to a BoxState -/
def applyStep (state : BoxState) (step : Step) : BoxState :=
  match step with
  | Step.Remove => ⟨state.p - 1, state.q - 1⟩
  | Step.TripleP => ⟨state.p * 3, state.q⟩
  | Step.TripleQ => ⟨state.p, state.q * 3⟩

/-- Checks if a BoxState is empty (both boxes have 0 pebbles) -/
def isEmpty (state : BoxState) : Prop :=
  state.p = 0 ∧ state.q = 0

/-- Defines if it's possible to empty both boxes from a given initial state -/
def canEmpty (initial : BoxState) : Prop :=
  ∃ (steps : List Step), isEmpty (steps.foldl applyStep initial)

/-- The main theorem to be proved -/
theorem emptyBoxes (p q : ℕ) :
  canEmpty ⟨p, q⟩ ↔ p % 2 = q % 2 := by sorry


end NUMINAMATH_CALUDE_emptyBoxes_l65_6523


namespace NUMINAMATH_CALUDE_johnny_money_left_l65_6593

def johnny_savings (september october november : ℕ) : ℕ := september + october + november

theorem johnny_money_left (september october november spending : ℕ) 
  (h1 : september = 30)
  (h2 : october = 49)
  (h3 : november = 46)
  (h4 : spending = 58) :
  johnny_savings september october november - spending = 67 := by
  sorry

end NUMINAMATH_CALUDE_johnny_money_left_l65_6593


namespace NUMINAMATH_CALUDE_integer_representation_l65_6578

theorem integer_representation (n : ℤ) : ∃ (x y z : ℕ+), (n : ℤ) = x^2 + y^2 - z^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_representation_l65_6578


namespace NUMINAMATH_CALUDE_range_of_a_l65_6562

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 - (a+1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) ∧ 
  ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1)) ∧
  (∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → 
  ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l65_6562


namespace NUMINAMATH_CALUDE_jennifer_spending_l65_6579

theorem jennifer_spending (total : ℝ) (sandwich_fraction : ℝ) (museum_fraction : ℝ) (book_fraction : ℝ)
  (h_total : total = 150)
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum : museum_fraction = 1/6)
  (h_book : book_fraction = 1/2) :
  total - (sandwich_fraction * total + museum_fraction * total + book_fraction * total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_spending_l65_6579


namespace NUMINAMATH_CALUDE_fifteen_tomorrow_fishers_l65_6509

/-- Represents the fishing schedule in the coastal village -/
structure FishingSchedule where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow based on the given schedule -/
def tomorrowFishers (schedule : FishingSchedule) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_tomorrow_fishers (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_tomorrow_fishers_l65_6509


namespace NUMINAMATH_CALUDE_alBr3_weight_calculation_l65_6556

/-- Calculates the weight of AlBr3 given moles and isotope data -/
def weightAlBr3 (moles : ℝ) (alMass : ℝ) (br79Mass br81Mass : ℝ) (br79Abundance br81Abundance : ℝ) : ℝ :=
  let brAvgMass := br79Mass * br79Abundance + br81Mass * br81Abundance
  let molarMass := alMass + 3 * brAvgMass
  moles * molarMass

/-- The weight of 4 moles of AlBr3 is approximately 1067.2344 grams -/
theorem alBr3_weight_calculation :
  let moles : ℝ := 4
  let alMass : ℝ := 27
  let br79Mass : ℝ := 79
  let br81Mass : ℝ := 81
  let br79Abundance : ℝ := 0.5069
  let br81Abundance : ℝ := 0.4931
  ∃ ε > 0, |weightAlBr3 moles alMass br79Mass br81Mass br79Abundance br81Abundance - 1067.2344| < ε :=
by sorry

end NUMINAMATH_CALUDE_alBr3_weight_calculation_l65_6556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l65_6532

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 1 + a 2017 = 10 →
  a 1 * a 2017 = 16 →
  a 2 + a 1009 + a 2016 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l65_6532


namespace NUMINAMATH_CALUDE_unique_solution_system_l65_6596

theorem unique_solution_system : 
  ∃! (x y z : ℕ+), 
    (x : ℝ)^2 = 2 * ((y : ℝ) + (z : ℝ)) ∧ 
    (x : ℝ)^6 = (y : ℝ)^6 + (z : ℝ)^6 + 31 * ((y : ℝ)^2 + (z : ℝ)^2) ∧
    x = 2 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l65_6596


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l65_6549

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x^2 - 6*x + 8 ≠ 0) 
  (h2 : x^2 - 8*x + 15 ≠ 0) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  (x - 3) / (x^2 - 6*x + 8) :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l65_6549


namespace NUMINAMATH_CALUDE_rectangular_park_area_l65_6576

/-- A rectangular park with a perimeter of 80 feet and length three times its width has an area of 300 square feet. -/
theorem rectangular_park_area : ∀ l w : ℝ,
  l > 0 → w > 0 →  -- Ensure positive dimensions
  2 * (l + w) = 80 →  -- Perimeter condition
  l = 3 * w →  -- Length is three times the width
  l * w = 300 := by
sorry

end NUMINAMATH_CALUDE_rectangular_park_area_l65_6576


namespace NUMINAMATH_CALUDE_travel_distance_l65_6568

/-- Proves that given a person traveling equal distances at speeds of 5 km/hr, 10 km/hr, and 15 km/hr,
    and taking a total time of 11 minutes, the total distance traveled is 1.5 km. -/
theorem travel_distance (d : ℝ) : 
  d / 5 + d / 10 + d / 15 = 11 / 60 → 3 * d = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_l65_6568


namespace NUMINAMATH_CALUDE_remainder_sum_l65_6503

theorem remainder_sum (c d : ℤ) 
  (hc : c % 52 = 48) 
  (hd : d % 87 = 82) : 
  (c + d) % 29 = 22 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l65_6503


namespace NUMINAMATH_CALUDE_div_power_eq_reciprocal_pow_l65_6573

/-- Definition of division power for rational numbers -/
def div_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (div_power a (n - 1))

/-- Theorem: Division power is equivalent to reciprocal exponentiation -/
theorem div_power_eq_reciprocal_pow (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  div_power a n = (1 / a) ^ (n - 2) :=
sorry

end NUMINAMATH_CALUDE_div_power_eq_reciprocal_pow_l65_6573


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l65_6595

/-- Given even integers a and b where b = a + 2, and c = ab, √(a^2 + b^2 + c^2) is always irrational. -/
theorem sqrt_D_irrational (a b c : ℤ) : 
  Even a → Even b → b = a + 2 → c = a * b → 
  Irrational (Real.sqrt ((a^2 : ℝ) + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l65_6595


namespace NUMINAMATH_CALUDE_smallest_transformed_sum_l65_6574

/-- The number of faces on a standard die -/
def facesOnDie : ℕ := 6

/-- The target sum we want to achieve -/
def targetSum : ℕ := 1994

/-- The function to calculate the transformed sum given the number of dice -/
def transformedSum (n : ℕ) : ℕ := 7 * n - targetSum

/-- The theorem stating the smallest possible value of the transformed sum -/
theorem smallest_transformed_sum :
  ∃ (n : ℕ), 
    (n * facesOnDie ≥ targetSum) ∧ 
    (∀ m : ℕ, m * facesOnDie ≥ targetSum → n ≤ m) ∧
    (transformedSum n = 337) := by
  sorry

end NUMINAMATH_CALUDE_smallest_transformed_sum_l65_6574


namespace NUMINAMATH_CALUDE_reflection_of_line_over_x_axis_l65_6587

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Reflects a line over the x-axis --/
def reflect_over_x_axis (l : Line) : Line :=
  { slope := -l.slope, intercept := -l.intercept }

theorem reflection_of_line_over_x_axis :
  let original_line : Line := { slope := 2, intercept := 3 }
  let reflected_line : Line := reflect_over_x_axis original_line
  reflected_line = { slope := -2, intercept := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_line_over_x_axis_l65_6587


namespace NUMINAMATH_CALUDE_pencil_profit_calculation_pencil_profit_proof_l65_6518

theorem pencil_profit_calculation (purchase_quantity : ℕ) (purchase_price : ℚ) 
  (selling_price : ℚ) (desired_profit : ℚ) (sold_quantity : ℕ) : Prop :=
  purchase_quantity = 2000 →
  purchase_price = 15/100 →
  selling_price = 30/100 →
  desired_profit = 150 →
  sold_quantity = 1500 →
  (sold_quantity : ℚ) * selling_price - (purchase_quantity : ℚ) * purchase_price = desired_profit

/-- Proof that selling 1500 pencils at $0.30 each results in a profit of $150 
    when 2000 pencils were purchased at $0.15 each. -/
theorem pencil_profit_proof :
  pencil_profit_calculation 2000 (15/100) (30/100) 150 1500 := by
  sorry

end NUMINAMATH_CALUDE_pencil_profit_calculation_pencil_profit_proof_l65_6518


namespace NUMINAMATH_CALUDE_only_setD_cannot_form_triangle_l65_6545

/-- A set of three line segments that might form a triangle -/
structure TriangleSegments where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three line segments can form a triangle -/
def canFormTriangle (t : TriangleSegments) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The four sets of line segments given in the problem -/
def setA : TriangleSegments := ⟨3, 4, 5⟩
def setB : TriangleSegments := ⟨5, 10, 8⟩
def setC : TriangleSegments := ⟨5, 4.5, 8⟩
def setD : TriangleSegments := ⟨7, 7, 15⟩

/-- Theorem: Among the given sets, only set D cannot form a triangle -/
theorem only_setD_cannot_form_triangle :
  canFormTriangle setA ∧ 
  canFormTriangle setB ∧ 
  canFormTriangle setC ∧ 
  ¬canFormTriangle setD := by
  sorry

end NUMINAMATH_CALUDE_only_setD_cannot_form_triangle_l65_6545


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l65_6589

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l65_6589


namespace NUMINAMATH_CALUDE_average_headcount_rounded_l65_6547

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11300
def fall_headcount_05_06 : ℕ := 11400

def average_headcount : ℚ := (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06) / 3

theorem average_headcount_rounded : 
  round average_headcount = 11400 := by sorry

end NUMINAMATH_CALUDE_average_headcount_rounded_l65_6547


namespace NUMINAMATH_CALUDE_equation_and_inequality_system_l65_6552

theorem equation_and_inequality_system :
  -- Part 1: Equation
  (let equation := fun x : ℝ => 2 * x * (x - 2) = 1
   let solution1 := (2 + Real.sqrt 6) / 2
   let solution2 := (2 - Real.sqrt 6) / 2
   equation solution1 ∧ equation solution2) ∧
  -- Part 2: Inequality system
  (let inequality1 := fun x : ℝ => 2 * x + 3 > 1
   let inequality2 := fun x : ℝ => x - 2 ≤ (1 / 2) * (x + 2)
   ∀ x : ℝ, (inequality1 x ∧ inequality2 x) ↔ (-1 < x ∧ x ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequality_system_l65_6552


namespace NUMINAMATH_CALUDE_number_of_divisors_l65_6575

-- Define the number we're working with
def n : ℕ := 3465

-- Define the prime factorization of n
axiom prime_factorization : n = 3^2 * 5^1 * 7^2

-- Define the function to count positive divisors
def count_divisors (m : ℕ) : ℕ := sorry

-- Theorem stating the number of positive divisors of n
theorem number_of_divisors : count_divisors n = 18 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_l65_6575


namespace NUMINAMATH_CALUDE_area_of_triangle_DCE_l65_6563

/-- Given a rectangle BDEF with AB = 24 and EF = 15, and triangle BCE with area 60,
    prove that the area of triangle DCE is 30 -/
theorem area_of_triangle_DCE (AB EF : ℝ) (area_BCE : ℝ) :
  AB = 24 →
  EF = 15 →
  area_BCE = 60 →
  let BC := (2 * area_BCE) / EF
  let DC := EF - BC
  let DE := (2 * area_BCE) / BC
  (1/2) * DC * DE = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DCE_l65_6563


namespace NUMINAMATH_CALUDE_equation_holds_l65_6533

theorem equation_holds (n : ℕ+) : 
  (n^2)^2 + n^2 + 1 = (n^2 + n + 1) * ((n-1)^2 + (n-1) + 1) := by
  sorry

#check equation_holds

end NUMINAMATH_CALUDE_equation_holds_l65_6533


namespace NUMINAMATH_CALUDE_special_triangle_st_length_l65_6585

/-- Triangle with given side lengths and a line segment parallel to one side passing through the incenter --/
structure SpecialTriangle where
  -- Side lengths of the triangle
  pq : ℝ
  pr : ℝ
  qr : ℝ
  -- Points S and T on sides PQ and PR respectively
  s : ℝ  -- distance PS
  t : ℝ  -- distance PT
  -- Conditions
  pq_positive : pq > 0
  pr_positive : pr > 0
  qr_positive : qr > 0
  s_on_pq : 0 < s ∧ s < pq
  t_on_pr : 0 < t ∧ t < pr
  st_parallel_qr : True  -- We can't directly express this geometric condition
  st_contains_incenter : True  -- We can't directly express this geometric condition

/-- The theorem stating that in the special triangle, ST has a specific value --/
theorem special_triangle_st_length (tri : SpecialTriangle) 
    (h_pq : tri.pq = 26) 
    (h_pr : tri.pr = 28) 
    (h_qr : tri.qr = 30) : 
  (tri.s - 0) / tri.pq + (tri.t - 0) / tri.pr = 135 / 7 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_st_length_l65_6585


namespace NUMINAMATH_CALUDE_inverse_113_mod_114_l65_6522

theorem inverse_113_mod_114 : ∃ x : ℕ, x ∈ Finset.range 114 ∧ (113 * x) % 114 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_inverse_113_mod_114_l65_6522


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_one_and_four_l65_6531

theorem arithmetic_mean_of_one_and_four :
  (1 + 4) / 2 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_one_and_four_l65_6531


namespace NUMINAMATH_CALUDE_planning_committee_selections_l65_6560

def student_council_size : ℕ := 6

theorem planning_committee_selections :
  (Nat.choose student_council_size 3 = 20) →
  (Nat.choose student_council_size 3 = 20) := by
  sorry

end NUMINAMATH_CALUDE_planning_committee_selections_l65_6560


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l65_6529

/-- The repeating decimal 0.37̄246 expressed as a rational number -/
def repeating_decimal : ℚ := 37246 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = 371874 / 99900 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l65_6529


namespace NUMINAMATH_CALUDE_equation_solution_l65_6571

theorem equation_solution : ∃ x : ℚ, (4 * x + 5 * x = 350 - 10 * (x - 5)) ∧ (x = 400 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l65_6571


namespace NUMINAMATH_CALUDE_simplify_expression_l65_6528

theorem simplify_expression (x y : ℝ) :
  3 * (x + y)^2 - 7 * (x + y) + 8 * (x + y)^2 + 6 * (x + y) = 11 * (x + y)^2 - (x + y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l65_6528


namespace NUMINAMATH_CALUDE_marks_of_a_l65_6521

theorem marks_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e = d + 3 →
  (b + c + d + e) / 4 = 48 →
  a = 43 := by
sorry

end NUMINAMATH_CALUDE_marks_of_a_l65_6521


namespace NUMINAMATH_CALUDE_correct_calculation_l65_6513

theorem correct_calculation (a : ℝ) : 2 * a^4 * 3 * a^5 = 6 * a^9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l65_6513


namespace NUMINAMATH_CALUDE_fruits_remaining_proof_l65_6500

def initial_apples : ℕ := 7
def initial_oranges : ℕ := 8
def initial_mangoes : ℕ := 15

def apples_taken : ℕ := 2
def oranges_taken : ℕ := 2 * apples_taken
def mangoes_taken : ℕ := (2 * initial_mangoes) / 3

def remaining_fruits : ℕ :=
  (initial_apples - apples_taken) +
  (initial_oranges - oranges_taken) +
  (initial_mangoes - mangoes_taken)

theorem fruits_remaining_proof :
  remaining_fruits = 14 := by sorry

end NUMINAMATH_CALUDE_fruits_remaining_proof_l65_6500


namespace NUMINAMATH_CALUDE_unique_solution_for_t_l65_6510

/-- A non-zero digit is an integer between 1 and 9, inclusive. -/
def NonZeroDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The expression as a function of k and t -/
def expression (k t : NonZeroDigit) : ℤ :=
  808 + 10 * k.val + 80 * k.val + 8 - (1600 + 6 * t.val + 6)

theorem unique_solution_for_t :
  ∃! (t : NonZeroDigit), ∀ (k : NonZeroDigit),
    ∃ (n : ℤ), expression k t = n ∧ n % 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_t_l65_6510


namespace NUMINAMATH_CALUDE_max_gcd_sum_780_l65_6530

theorem max_gcd_sum_780 :
  ∃ (a b : ℕ+), a + b = 780 ∧ 
  ∀ (c d : ℕ+), c + d = 780 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 390 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_780_l65_6530


namespace NUMINAMATH_CALUDE_triangle_area_is_36_l65_6555

/-- The area of a triangle formed by three lines in a 2D plane -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

theorem triangle_area_is_36 :
  let line1 := fun (x : ℝ) ↦ 8
  let line2 := fun (x : ℝ) ↦ 2 + x
  let line3 := fun (x : ℝ) ↦ 2 - x
  triangleArea line1 line2 line3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_36_l65_6555


namespace NUMINAMATH_CALUDE_shoe_price_ratio_l65_6594

/-- Given a shoe with a marked price, a discount of 1/4 off, and a cost that is 2/3 of the actual selling price, 
    the ratio of the cost to the marked price is 1/2. -/
theorem shoe_price_ratio (marked_price : ℝ) (marked_price_pos : 0 < marked_price) : 
  let selling_price := (3/4) * marked_price
  let cost := (2/3) * selling_price
  cost / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_ratio_l65_6594


namespace NUMINAMATH_CALUDE_logarithm_relation_l65_6554

theorem logarithm_relation (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hdist : a ≠ b ∧ a ≠ x ∧ b ≠ x)
  (heq : 4 * (Real.log x / Real.log a)^3 + 5 * (Real.log x / Real.log b)^3 = 7 * (Real.log x)^3) :
  ∃ k, b = a^k ∧ k = (3/5)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_relation_l65_6554


namespace NUMINAMATH_CALUDE_subsidy_calculation_l65_6561

/-- Represents the "Home Appliances to the Countryside" initiative subsidy calculation -/
theorem subsidy_calculation (x : ℝ) : 
  (20 * x * 0.13 = 2340) ↔ 
  (∃ (subsidy_rate : ℝ) (num_phones : ℕ) (total_subsidy : ℝ),
    subsidy_rate = 0.13 ∧ 
    num_phones = 20 ∧ 
    total_subsidy = 2340 ∧
    num_phones * (x * subsidy_rate) = total_subsidy) :=
by sorry

end NUMINAMATH_CALUDE_subsidy_calculation_l65_6561


namespace NUMINAMATH_CALUDE_d_equals_square_cases_l65_6559

/-- Function to move the last digit of a number to the first position -/
def moveLastToFirst (n : ℕ) : ℕ := sorry

/-- Function to move the first digit of a number to the last position -/
def moveFirstToLast (n : ℕ) : ℕ := sorry

/-- The d function as described in the problem -/
def d (a : ℕ) : ℕ := 
  let b := moveLastToFirst a
  let c := b * b
  moveFirstToLast c

/-- Theorem stating the possible forms of a when d(a) = a^2 -/
theorem d_equals_square_cases (a : ℕ) (h : 0 < a) : 
  d a = a * a → (a = 2 ∨ a = 3 ∨ ∃ x y : ℕ, a = 20000 + 100 * x + 10 * y + 21) := by
  sorry

end NUMINAMATH_CALUDE_d_equals_square_cases_l65_6559


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l65_6591

/-- Given a point P with coordinates (2, -1) in the standard coordinate system
    and (b-1, a+3) in another coordinate system with the same origin,
    prove that a + b = -1 -/
theorem point_coordinate_sum (a b : ℝ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l65_6591


namespace NUMINAMATH_CALUDE_square_difference_l65_6577

theorem square_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 2) : x^2 - y^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l65_6577


namespace NUMINAMATH_CALUDE_mrs_crocker_chicken_l65_6536

def chicken_problem (lyndee_pieces : ℕ) (friend_pieces : ℕ) (num_friends : ℕ) : Prop :=
  lyndee_pieces = 1 ∧ friend_pieces = 2 ∧ num_friends = 5 →
  lyndee_pieces + friend_pieces * num_friends = 11

theorem mrs_crocker_chicken : chicken_problem 1 2 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_crocker_chicken_l65_6536


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l65_6543

/-- Calculates the corrected mean when one observation in a dataset is incorrect -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean + (correct_value - incorrect_value)) / n

/-- Theorem stating that the corrected mean is 36.02 given the problem conditions -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let incorrect_value : ℚ := 47
  let correct_value : ℚ := 48
  corrected_mean n initial_mean incorrect_value correct_value = 3602/100 := by
  sorry

#eval corrected_mean 50 36 47 48

end NUMINAMATH_CALUDE_corrected_mean_problem_l65_6543


namespace NUMINAMATH_CALUDE_solution_equality_implies_k_equals_one_l65_6592

theorem solution_equality_implies_k_equals_one :
  ∀ x k : ℝ,
  (2 * x - 1 = 3 * x - 2) →
  (4 - (k * x + 2) / 3 = 3 * k - (2 - 2 * x) / 4) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_equality_implies_k_equals_one_l65_6592


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l65_6570

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → x ∈ Set.Icc (-1) 2 → y ∈ Set.Icc (-1) 2 → f x < f y) →
  (∀ x, x ∈ Set.Iio 2 → f x ≠ 0) →
  f (a - 1) > f (1 - 3 * a) →
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_increasing_function_inequality_l65_6570


namespace NUMINAMATH_CALUDE_inequality_proof_l65_6598

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) :
  (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l65_6598


namespace NUMINAMATH_CALUDE_max_int_diff_l65_6502

theorem max_int_diff (x y : ℤ) (hx : 6 < x ∧ x < 10) (hy : 10 < y ∧ y < 17) :
  (∀ a b : ℤ, 6 < a ∧ a < 10 ∧ 10 < b ∧ b < 17 → y - x ≥ b - a) ∧ y - x = 7 :=
sorry

end NUMINAMATH_CALUDE_max_int_diff_l65_6502


namespace NUMINAMATH_CALUDE_lexie_picked_12_apples_l65_6597

/-- The number of apples Lexie and Tom picked together -/
def total_apples : ℕ := 36

/-- Lexie's apples -/
def lexie_apples : ℕ := 12

/-- Tom's apples -/
def tom_apples : ℕ := 2 * lexie_apples

/-- Theorem stating that Lexie picked 12 apples given the conditions -/
theorem lexie_picked_12_apples : 
  (tom_apples = 2 * lexie_apples) ∧ (lexie_apples + tom_apples = total_apples) → 
  lexie_apples = 12 := by
  sorry

#check lexie_picked_12_apples

end NUMINAMATH_CALUDE_lexie_picked_12_apples_l65_6597


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l65_6541

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l65_6541


namespace NUMINAMATH_CALUDE_determinant_value_trig_expression_value_l65_6584

-- Define the determinant function for 2x2 matrices
def det2 (a11 a12 a21 a22 : ℝ) : ℝ := a11 * a22 - a12 * a21

-- Problem 1
theorem determinant_value : 
  det2 (Real.cos (π/4)) 1 1 (Real.cos (π/3)) = (Real.sqrt 2 - 2) / 4 := by
  sorry

-- Problem 2
theorem trig_expression_value (a : ℝ) (h : Real.tan (π/4 + a) = -1/2) :
  (Real.sin (2*a) - 2 * (Real.cos a)^2) / (1 + Real.tan a) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_determinant_value_trig_expression_value_l65_6584


namespace NUMINAMATH_CALUDE_first_division_percentage_l65_6514

theorem first_division_percentage 
  (total_students : ℕ) 
  (second_division_percentage : ℚ)
  (just_passed_count : ℕ) :
  total_students = 300 →
  second_division_percentage = 54/100 →
  just_passed_count = 51 →
  ∃ (first_division_percentage : ℚ),
    first_division_percentage = 29/100 ∧
    first_division_percentage + second_division_percentage + (just_passed_count : ℚ) / total_students = 1 :=
by sorry

end NUMINAMATH_CALUDE_first_division_percentage_l65_6514


namespace NUMINAMATH_CALUDE_last_digit_product_divisible_by_three_l65_6527

theorem last_digit_product_divisible_by_three (n : ℕ) :
  let a := (2^n % 10)
  ∃ k : ℤ, a * (2^n - a) = 3 * k :=
sorry

end NUMINAMATH_CALUDE_last_digit_product_divisible_by_three_l65_6527


namespace NUMINAMATH_CALUDE_fourth_power_inequality_l65_6505

theorem fourth_power_inequality (a b c : ℝ) : a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_inequality_l65_6505


namespace NUMINAMATH_CALUDE_solution_set_and_range_l65_6558

def f (x : ℝ) : ℝ := |2 * x - 1| + 1

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m : ℝ, (∃ n : ℝ, f n ≤ m - f (-n)) ↔ 4 ≤ m) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l65_6558


namespace NUMINAMATH_CALUDE_first_movie_length_is_correct_l65_6580

/-- The length of the first movie in minutes -/
def first_movie_length : ℕ := 90

/-- The length of the second movie in minutes -/
def second_movie_length : ℕ := first_movie_length + 30

/-- The time spent making popcorn in minutes -/
def popcorn_time : ℕ := 10

/-- The time spent making fries in minutes -/
def fries_time : ℕ := 2 * popcorn_time

/-- The total time spent cooking and watching movies in minutes -/
def total_time : ℕ := 4 * 60

theorem first_movie_length_is_correct : 
  first_movie_length + second_movie_length + popcorn_time + fries_time = total_time := by
  sorry

end NUMINAMATH_CALUDE_first_movie_length_is_correct_l65_6580


namespace NUMINAMATH_CALUDE_johnny_october_savings_l65_6588

/-- Proves that Johnny saved $49 in October given his savings and spending information. -/
theorem johnny_october_savings :
  let september_savings : ℕ := 30
  let november_savings : ℕ := 46
  let video_game_cost : ℕ := 58
  let remaining_money : ℕ := 67
  let october_savings : ℕ := 49
  september_savings + october_savings + november_savings - video_game_cost = remaining_money :=
by
  sorry

#check johnny_october_savings

end NUMINAMATH_CALUDE_johnny_october_savings_l65_6588
