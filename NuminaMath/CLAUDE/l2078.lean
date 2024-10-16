import Mathlib

namespace NUMINAMATH_CALUDE_theater_seating_l2078_207829

/-- Represents the number of seats in a given row of the theater. -/
def seats (n : ℕ) : ℕ := 3 * n + 57

theorem theater_seating :
  (seats 6 = 75) ∧
  (seats 8 = 81) ∧
  (∀ n : ℕ, seats n = 3 * n + 57) ∧
  (seats 21 = 120) := by
  sorry

#check theater_seating

end NUMINAMATH_CALUDE_theater_seating_l2078_207829


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2078_207807

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The perpendicular distances from the point to each side
  dist_to_side1 : ℝ
  dist_to_side2 : ℝ
  dist_to_side3 : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  is_equilateral : side_length > 0
  point_inside : dist_to_side1 > 0 ∧ dist_to_side2 > 0 ∧ dist_to_side3 > 0

/-- The theorem to be proved -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint) 
  (h1 : triangle.dist_to_side1 = 2) 
  (h2 : triangle.dist_to_side2 = 3) 
  (h3 : triangle.dist_to_side3 = 4) : 
  triangle.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2078_207807


namespace NUMINAMATH_CALUDE_M_binary_op_result_l2078_207862

def M : Set ℕ := {2, 3}

def binary_op (A : Set ℕ) : Set ℕ := {x | ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem M_binary_op_result : binary_op M = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_M_binary_op_result_l2078_207862


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l2078_207852

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (n : ℕ) (initial : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - initial

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marbles_problem :
  min_additional_marbles 12 40 = 38 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l2078_207852


namespace NUMINAMATH_CALUDE_log_product_equality_l2078_207867

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log y^4) * (Real.log y^3 / Real.log x^3) *
  (Real.log x^4 / Real.log y^5) * (Real.log y^4 / Real.log x^2) *
  (Real.log x^3 / Real.log y^3) = (1/5) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l2078_207867


namespace NUMINAMATH_CALUDE_weights_division_condition_l2078_207824

/-- A function that checks if a set of weights from 1 to n grams can be divided into three equal mass piles -/
def canDivideWeights (n : ℕ) : Prop :=
  ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧
                         a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
                         (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)

/-- The theorem stating the condition for when weights can be divided into three equal mass piles -/
theorem weights_division_condition (n : ℕ) (h : n > 3) :
  canDivideWeights n ↔ n % 3 = 0 ∨ n % 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_weights_division_condition_l2078_207824


namespace NUMINAMATH_CALUDE_range_of_a_l2078_207865

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, |x - a| > x - 1) ↔ (a < 1 ∨ a > 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2078_207865


namespace NUMINAMATH_CALUDE_waiter_customers_l2078_207851

/-- Calculates the final number of customers for a waiter given the initial number,
    the number who left, and the number of new customers. -/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Proves that for the given scenario, the final number of customers is 41. -/
theorem waiter_customers : final_customers 19 14 36 = 41 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l2078_207851


namespace NUMINAMATH_CALUDE_prob_second_science_example_l2078_207811

/-- Represents a set of questions with science and humanities subjects -/
structure QuestionSet where
  total : Nat
  science : Nat
  humanities : Nat
  h_total : total = science + humanities

/-- Calculates the probability of drawing a science question on the second draw,
    given that the first drawn question was a science question -/
def prob_second_science (qs : QuestionSet) : Rat :=
  if qs.science > 0 then
    (qs.science - 1) / (qs.total - 1)
  else
    0

theorem prob_second_science_example :
  let qs : QuestionSet := ⟨5, 3, 2, rfl⟩
  prob_second_science qs = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_second_science_example_l2078_207811


namespace NUMINAMATH_CALUDE_arrangement_count_l2078_207897

-- Define the number of red and blue balls
def red_balls : ℕ := 8
def blue_balls : ℕ := 9
def total_balls : ℕ := red_balls + blue_balls

-- Define the number of jars
def num_jars : ℕ := 2

-- Define a function to calculate the number of distinguishable arrangements
def count_arrangements (red : ℕ) (blue : ℕ) (jars : ℕ) : ℕ :=
  sorry -- The actual implementation would go here

-- State the theorem
theorem arrangement_count :
  count_arrangements red_balls blue_balls num_jars = 7 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l2078_207897


namespace NUMINAMATH_CALUDE_train_length_calculation_l2078_207844

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 12 → 
  ∃ (length_m : ℝ), abs (length_m - 200.04) < 0.01 ∧ length_m = (speed_kmh * 1000 / 3600) * time_s :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2078_207844


namespace NUMINAMATH_CALUDE_max_primes_arithmetic_seq_diff_12_l2078_207821

theorem max_primes_arithmetic_seq_diff_12 :
  ∀ (seq : ℕ → ℕ) (n : ℕ),
    (∀ i < n, seq i.succ = seq i + 12) →
    (∀ i < n, Nat.Prime (seq i)) →
    n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_primes_arithmetic_seq_diff_12_l2078_207821


namespace NUMINAMATH_CALUDE_second_task_end_time_l2078_207890

-- Define the start and end times in minutes since midnight
def start_time : Nat := 8 * 60  -- 8:00 AM
def end_time : Nat := 12 * 60 + 20  -- 12:20 PM

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem to prove
theorem second_task_end_time :
  let total_duration : Nat := end_time - start_time
  let task_duration : Nat := total_duration / num_tasks
  let second_task_end : Nat := start_time + 2 * task_duration
  second_task_end = 10 * 60 + 10  -- 10:10 AM
  := by sorry

end NUMINAMATH_CALUDE_second_task_end_time_l2078_207890


namespace NUMINAMATH_CALUDE_even_periodic_function_l2078_207880

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

theorem even_periodic_function 
  (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : is_period_two f) 
  (h3 : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-1) 0, f x = 2 - x :=
sorry

end NUMINAMATH_CALUDE_even_periodic_function_l2078_207880


namespace NUMINAMATH_CALUDE_marcel_corn_count_l2078_207860

/-- The number of ears of corn Marcel bought -/
def marcel_corn : ℕ := sorry

/-- The number of ears of corn Dale bought -/
def dale_corn : ℕ := sorry

/-- The number of potatoes Dale bought -/
def dale_potatoes : ℕ := 8

/-- The number of potatoes Marcel bought -/
def marcel_potatoes : ℕ := 4

/-- The total number of vegetables bought -/
def total_vegetables : ℕ := 27

theorem marcel_corn_count :
  (dale_corn = marcel_corn / 2) →
  (dale_potatoes = 8) →
  (marcel_potatoes = 4) →
  (marcel_corn + dale_corn + dale_potatoes + marcel_potatoes = total_vegetables) →
  marcel_corn = 10 := by sorry

end NUMINAMATH_CALUDE_marcel_corn_count_l2078_207860


namespace NUMINAMATH_CALUDE_rounding_problem_l2078_207885

theorem rounding_problem (x : ℝ) (n : ℕ) : 
  x > 0 ∧ n = ⌈x⌉ ∧ n = (1.28 : ℝ) * x → 
  x ∈ ({25/32, 25/16, 75/32, 25/8} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_rounding_problem_l2078_207885


namespace NUMINAMATH_CALUDE_line_through_quadrants_l2078_207896

/-- A line y = kx + b passes through the second, third, and fourth quadrants if and only if
    k and b satisfy the conditions: k + b = -5 and kb = 6 -/
theorem line_through_quadrants (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b → 
    ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ 
  (k + b = -5 ∧ k * b = 6) := by
  sorry


end NUMINAMATH_CALUDE_line_through_quadrants_l2078_207896


namespace NUMINAMATH_CALUDE_max_areas_theorem_l2078_207868

/-- Represents the number of non-overlapping areas in a circular disk -/
def max_areas (n : ℕ+) : ℕ :=
  3 * n + 3

/-- 
Theorem: The maximum number of non-overlapping areas in a circular disk 
divided by 2n equally spaced radii and two optimally placed secant lines 
is 3n + 3, where n is a positive integer.
-/
theorem max_areas_theorem (n : ℕ+) : 
  max_areas n = 3 * n + 3 := by
  sorry

#check max_areas_theorem

end NUMINAMATH_CALUDE_max_areas_theorem_l2078_207868


namespace NUMINAMATH_CALUDE_bread_baking_time_l2078_207814

/-- The time it takes for one ball of dough to rise -/
def rise_time : ℕ := 3

/-- The time it takes to bake one ball of dough -/
def bake_time : ℕ := 2

/-- The number of balls of dough Ellen makes -/
def num_balls : ℕ := 4

/-- The total time taken to make and bake all balls of dough -/
def total_time : ℕ := rise_time + (num_balls - 1) * rise_time + num_balls * bake_time

theorem bread_baking_time :
  total_time = 14 :=
sorry

end NUMINAMATH_CALUDE_bread_baking_time_l2078_207814


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2078_207819

/-- Given a circle with polar equation ρ = 5cos(θ) - 5√3sin(θ), 
    its center coordinates in polar form are (5, 5π/3) -/
theorem circle_center_coordinates (θ : Real) (ρ : Real) :
  ρ = 5 * Real.cos θ - 5 * Real.sqrt 3 * Real.sin θ →
  ∃ (r : Real) (φ : Real),
    r = 5 ∧ φ = 5 * Real.pi / 3 ∧
    r * Real.cos φ = 5 / 2 ∧
    r * Real.sin φ = -5 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2078_207819


namespace NUMINAMATH_CALUDE_lee_savings_l2078_207875

theorem lee_savings (initial_savings : ℕ) (num_figures : ℕ) (price_per_figure : ℕ) (sneaker_cost : ℕ) : 
  initial_savings = 15 →
  num_figures = 10 →
  price_per_figure = 10 →
  sneaker_cost = 90 →
  initial_savings + num_figures * price_per_figure - sneaker_cost = 25 := by
sorry

end NUMINAMATH_CALUDE_lee_savings_l2078_207875


namespace NUMINAMATH_CALUDE_prime_power_sum_l2078_207884

theorem prime_power_sum (p a n : ℕ) : 
  Prime p → 
  a > 0 → 
  n > 0 → 
  2^p + 3^p = a^n → 
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2078_207884


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l2078_207804

/-- A quadratic function passing through the origin with a given derivative -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ,
    (∀ x, f x = a * x^2 + b * x) ∧
    (f 0 = 0) ∧
    (∀ x, deriv f x = 3 * x - 1/2)

/-- The theorem stating the unique form of the quadratic function -/
theorem quadratic_function_unique (f : ℝ → ℝ) (hf : quadratic_function f) :
  ∀ x, f x = 3/2 * x^2 - 1/2 * x :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l2078_207804


namespace NUMINAMATH_CALUDE_complex_sum_reciprocal_imag_part_l2078_207837

theorem complex_sum_reciprocal_imag_part :
  let z : ℂ := 2 + I
  (z + z⁻¹).im = 4/5 := by sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocal_imag_part_l2078_207837


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2078_207887

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  6 * x / ((x - 4) * (x - 2)^2) = 6 / (x - 4) + (-6) / (x - 2) + (-6) / (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2078_207887


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2078_207888

/-- Expresses the repeating decimal 3.464646... as a rational number -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 46 / 99) ∧ (x = 343 / 99) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2078_207888


namespace NUMINAMATH_CALUDE_daisy_solution_l2078_207894

def daisy_problem (day1 day2 day3 day4 total : ℕ) : Prop :=
  day1 = 45 ∧
  day2 = day1 + 20 ∧
  day3 = 2 * day2 - 10 ∧
  day1 + day2 + day3 + day4 = total ∧
  total = 350

theorem daisy_solution :
  ∃ day1 day2 day3 day4 total, daisy_problem day1 day2 day3 day4 total ∧ day4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_daisy_solution_l2078_207894


namespace NUMINAMATH_CALUDE_unique_pair_sum_28_l2078_207820

theorem unique_pair_sum_28 (a b : ℕ) : 
  a ≠ b → a > 11 → b > 11 → a + b = 28 → (Even a ∨ Even b) → 
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) := by
sorry

end NUMINAMATH_CALUDE_unique_pair_sum_28_l2078_207820


namespace NUMINAMATH_CALUDE_candy_distribution_l2078_207846

theorem candy_distribution (bags : ℝ) (total_candy : ℕ) (h1 : bags = 15.0) (h2 : total_candy = 75) :
  (total_candy : ℝ) / bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2078_207846


namespace NUMINAMATH_CALUDE_exists_sweet_number_with_sweet_square_and_cube_l2078_207802

/-- A function that checks if a number is sweet (contains only digits 0, 1, 2, 4, and 8) -/
def is_sweet (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [0, 1, 2, 4, 8]

/-- The theorem to be proved -/
theorem exists_sweet_number_with_sweet_square_and_cube :
  ∃ n : ℕ, 
    is_sweet n ∧ 
    is_sweet (n^2) ∧ 
    is_sweet (n^3) ∧ 
    (n.digits 10).sum = 2014 := by
  sorry

end NUMINAMATH_CALUDE_exists_sweet_number_with_sweet_square_and_cube_l2078_207802


namespace NUMINAMATH_CALUDE_unique_reverse_half_ceiling_l2078_207817

/-- Function that reverses the digits of an integer -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Ceiling function -/
def ceil (x : ℚ) : ℕ := sorry

theorem unique_reverse_half_ceiling :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 10000 ∧ reverse_digits n = ceil (n / 2) ∧ n = 7993 := by sorry

end NUMINAMATH_CALUDE_unique_reverse_half_ceiling_l2078_207817


namespace NUMINAMATH_CALUDE_fruit_display_total_l2078_207866

/-- Fruit display problem -/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 →
  oranges = 2 * bananas →
  apples = 2 * oranges →
  bananas + oranges + apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_display_total_l2078_207866


namespace NUMINAMATH_CALUDE_lcm_14_21_35_l2078_207805

theorem lcm_14_21_35 : Nat.lcm 14 (Nat.lcm 21 35) = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_14_21_35_l2078_207805


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l2078_207839

/-- Calculates the perimeter of the non-shaded region in a composite figure --/
theorem non_shaded_perimeter (outer_length outer_width side_length side_width shaded_area : ℝ) : 
  outer_length = 10 →
  outer_width = 8 →
  side_length = 2 →
  side_width = 4 →
  shaded_area = 78 →
  let total_area := outer_length * outer_width + side_length * side_width
  let non_shaded_area := total_area - shaded_area
  let non_shaded_width := side_length
  let non_shaded_length := non_shaded_area / non_shaded_width
  2 * (non_shaded_length + non_shaded_width) = 14 :=
by sorry


end NUMINAMATH_CALUDE_non_shaded_perimeter_l2078_207839


namespace NUMINAMATH_CALUDE_algorithm_description_not_unique_l2078_207834

/-- Definition of an algorithm -/
structure Algorithm where
  steps : List String
  solves_problem : Bool

/-- There can be different ways to describe an algorithm -/
theorem algorithm_description_not_unique : ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ a1.solves_problem = a2.solves_problem := by
  sorry

end NUMINAMATH_CALUDE_algorithm_description_not_unique_l2078_207834


namespace NUMINAMATH_CALUDE_yield_prediction_80kg_l2078_207816

/-- Predicts the rice yield based on the amount of fertilizer applied. -/
def predict_yield (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem stating that when 80 kg of fertilizer is applied, the predicted yield is 650 kg. -/
theorem yield_prediction_80kg : predict_yield 80 = 650 := by
  sorry

end NUMINAMATH_CALUDE_yield_prediction_80kg_l2078_207816


namespace NUMINAMATH_CALUDE_max_consecutive_working_days_l2078_207856

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Checks if a given date is a working day for the guard -/
def isWorkingDay (d : Date) : Prop :=
  d.dayOfWeek = DayOfWeek.Tuesday ∨ 
  d.dayOfWeek = DayOfWeek.Friday ∨ 
  d.day % 2 = 1

/-- Represents a sequence of consecutive dates -/
def ConsecutiveDates (n : Nat) := Fin n → Date

/-- Checks if all dates in a sequence are working days -/
def allWorkingDays (dates : ConsecutiveDates n) : Prop :=
  ∀ i, isWorkingDay (dates i)

/-- The main theorem: The maximum number of consecutive working days is 6 -/
theorem max_consecutive_working_days :
  (∃ (dates : ConsecutiveDates 6), allWorkingDays dates) ∧
  (∀ (dates : ConsecutiveDates 7), ¬ allWorkingDays dates) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_working_days_l2078_207856


namespace NUMINAMATH_CALUDE_race_results_l2078_207818

/-- Represents a runner in the race -/
structure Runner where
  pace : ℕ  -- pace in minutes per mile
  breakTime : ℕ  -- break time in minutes
  breakStart : ℕ  -- time at which the break starts in minutes

/-- Calculates the total time taken by a runner to complete the race -/
def totalTime (r : Runner) (raceDistance : ℕ) : ℕ :=
  let distanceBeforeBreak := r.breakStart / r.pace
  let distanceAfterBreak := raceDistance - distanceBeforeBreak
  r.breakStart + r.breakTime + distanceAfterBreak * r.pace

/-- The main theorem stating the total time for each runner -/
theorem race_results (raceDistance : ℕ) (runner1 runner2 runner3 : Runner) : 
  raceDistance = 15 ∧ 
  runner1.pace = 6 ∧ runner1.breakTime = 3 ∧ runner1.breakStart = 42 ∧
  runner2.pace = 7 ∧ runner2.breakTime = 5 ∧ runner2.breakStart = 49 ∧
  runner3.pace = 8 ∧ runner3.breakTime = 7 ∧ runner3.breakStart = 56 →
  totalTime runner1 raceDistance = 93 ∧
  totalTime runner2 raceDistance = 110 ∧
  totalTime runner3 raceDistance = 127 := by
  sorry

end NUMINAMATH_CALUDE_race_results_l2078_207818


namespace NUMINAMATH_CALUDE_roses_difference_l2078_207815

theorem roses_difference (initial_roses : ℕ) (thrown_away : ℕ) (final_roses : ℕ)
  (h1 : initial_roses = 21)
  (h2 : thrown_away = 34)
  (h3 : final_roses = 15) :
  thrown_away - (thrown_away + final_roses - initial_roses) = 6 := by
  sorry

end NUMINAMATH_CALUDE_roses_difference_l2078_207815


namespace NUMINAMATH_CALUDE_carnival_friends_l2078_207891

theorem carnival_friends (total_tickets : ℕ) (tickets_per_person : ℕ) (h1 : total_tickets = 234) (h2 : tickets_per_person = 39) :
  total_tickets / tickets_per_person = 6 := by
  sorry

end NUMINAMATH_CALUDE_carnival_friends_l2078_207891


namespace NUMINAMATH_CALUDE_factorization_validity_l2078_207808

theorem factorization_validity (x y : ℝ) : 5 * x^2 * y - 10 * x * y^2 = 5 * x * y * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_validity_l2078_207808


namespace NUMINAMATH_CALUDE_daves_initial_apps_l2078_207859

theorem daves_initial_apps (initial_files : ℕ) (apps_left : ℕ) (files_left : ℕ) : 
  initial_files = 24 →
  apps_left = 21 →
  files_left = 4 →
  apps_left = files_left + 17 →
  ∃ initial_apps : ℕ, initial_apps = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_daves_initial_apps_l2078_207859


namespace NUMINAMATH_CALUDE_range_of_m_solution_sets_l2078_207869

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m
theorem range_of_m :
  {m : ℝ | ∀ x, y m x < 0} = Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution sets
theorem solution_sets (m : ℝ) :
  {x : ℝ | y m x < (1 - m) * x - 1} =
    if m = 0 then
      {x : ℝ | x > 0}
    else if m > 0 then
      {x : ℝ | 0 < x ∧ x < 1 / m}
    else
      {x : ℝ | x < 1 / m ∨ x > 0} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_sets_l2078_207869


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2078_207838

theorem arithmetic_computation : (-9 * 5) - (-7 * -2) + (11 * -4) = -103 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2078_207838


namespace NUMINAMATH_CALUDE_value_of_x_when_y_is_two_l2078_207836

theorem value_of_x_when_y_is_two (x y : ℚ) : 
  y = 1 / (5 * x + 2) → y = 2 → x = -3 / 10 := by sorry

end NUMINAMATH_CALUDE_value_of_x_when_y_is_two_l2078_207836


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l2078_207857

/-- Defines the continued fraction [2; 2, ..., 2] with n occurrences of 2 -/
def continued_fraction (n : ℕ) : ℚ :=
  if n = 0 then 2
  else 2 + 1 / continued_fraction (n - 1)

/-- The main theorem stating the equality of the continued fraction and the algebraic expression -/
theorem continued_fraction_equality (n : ℕ) :
  continued_fraction n = (((1 + Real.sqrt 2) ^ (n + 1) - (1 - Real.sqrt 2) ^ (n + 1)) /
                          ((1 + Real.sqrt 2) ^ n - (1 - Real.sqrt 2) ^ n)) := by
  sorry


end NUMINAMATH_CALUDE_continued_fraction_equality_l2078_207857


namespace NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l2078_207870

/-- Given a total of 5x + 10 problems and x + 2 missed problems, 
    the percentage of correctly answered problems is 80%. -/
theorem lucky_lacy_correct_percentage (x : ℕ) : 
  let total := 5 * x + 10
  let missed := x + 2
  let correct := total - missed
  (correct : ℚ) / total * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l2078_207870


namespace NUMINAMATH_CALUDE_power_product_exponent_l2078_207843

theorem power_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_exponent_l2078_207843


namespace NUMINAMATH_CALUDE_freshman_class_size_l2078_207899

theorem freshman_class_size :
  ∃ n : ℕ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧
  ∀ m : ℕ, m < n → (m % 25 ≠ 24 ∨ m % 19 ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_freshman_class_size_l2078_207899


namespace NUMINAMATH_CALUDE_complex_power_195_deg_60_l2078_207892

theorem complex_power_195_deg_60 :
  (Complex.exp (195 * π / 180 * Complex.I)) ^ 60 = (1 / 2 : ℂ) - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_195_deg_60_l2078_207892


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2078_207889

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2078_207889


namespace NUMINAMATH_CALUDE_double_pieces_count_l2078_207842

/-- Represents the number of circles on top of a Lego piece -/
inductive PieceType
| Single
| Double
| Triple
| Quadruple

/-- The cost of a Lego piece in cents -/
def cost (p : PieceType) : ℕ :=
  match p with
  | .Single => 1
  | .Double => 2
  | .Triple => 3
  | .Quadruple => 4

/-- The total revenue in cents -/
def total_revenue : ℕ := 1000

/-- The number of single pieces sold -/
def single_count : ℕ := 100

/-- The number of triple pieces sold -/
def triple_count : ℕ := 50

/-- The number of quadruple pieces sold -/
def quadruple_count : ℕ := 165

theorem double_pieces_count :
  ∃ (double_count : ℕ),
    double_count * cost PieceType.Double =
      total_revenue
        - (single_count * cost PieceType.Single
          + triple_count * cost PieceType.Triple
          + quadruple_count * cost PieceType.Quadruple)
    ∧ double_count = 45 := by
  sorry


end NUMINAMATH_CALUDE_double_pieces_count_l2078_207842


namespace NUMINAMATH_CALUDE_nine_keys_required_l2078_207858

/-- Represents the warehouse setup and retrieval task -/
structure WarehouseSetup where
  total_cabinets : ℕ
  boxes_per_cabinet : ℕ
  phones_per_box : ℕ
  phones_to_retrieve : ℕ

/-- Calculates the minimum number of keys required for the given warehouse setup -/
def min_keys_required (setup : WarehouseSetup) : ℕ :=
  let boxes_needed := (setup.phones_to_retrieve + setup.phones_per_box - 1) / setup.phones_per_box
  let cabinets_needed := (boxes_needed + setup.boxes_per_cabinet - 1) / setup.boxes_per_cabinet
  boxes_needed + cabinets_needed + 1

/-- Theorem stating that for the given warehouse setup, 9 keys are required -/
theorem nine_keys_required : 
  let setup : WarehouseSetup := {
    total_cabinets := 8,
    boxes_per_cabinet := 4,
    phones_per_box := 10,
    phones_to_retrieve := 52
  }
  min_keys_required setup = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_keys_required_l2078_207858


namespace NUMINAMATH_CALUDE_problem_statement_l2078_207876

theorem problem_statement : 
  (∃ x : ℝ, x - 2 > 0) ∧ ¬(∀ x : ℝ, Real.sqrt x < x) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2078_207876


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l2078_207809

theorem sum_of_squares_theorem (a b m : ℝ) 
  (h1 : a^2 + a*b = 16 + m) 
  (h2 : b^2 + a*b = 9 - m) : 
  (a + b = 5) ∨ (a + b = -5) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l2078_207809


namespace NUMINAMATH_CALUDE_smallest_sum_abcd_l2078_207825

/-- Given positive integers A, B, C forming an arithmetic sequence,
    and integers B, C, D forming a geometric sequence,
    with C/B = 7/3, prove that the smallest possible value of A + B + C + D is 76. -/
theorem smallest_sum_abcd (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = r * B ∧ D = r * C) →  -- B, C, D form a geometric sequence
  C = (7 : ℚ) / 3 * B →  -- C/B = 7/3
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = B' - A') →
    (∃ r : ℚ, C' = r * B' ∧ D' = r * C') →
    C' = (7 : ℚ) / 3 * B' →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_abcd_l2078_207825


namespace NUMINAMATH_CALUDE_square_plot_area_l2078_207822

/-- Given a square plot with a fence around it, if the price per foot of the fence is 58 and
    the total cost is 2088, then the area of the square plot is 81 square feet. -/
theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  price_per_foot = 58 →
  total_cost = 2088 →
  total_cost = 4 * side_length * price_per_foot →
  side_length ^ 2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_square_plot_area_l2078_207822


namespace NUMINAMATH_CALUDE_eight_million_scientific_notation_l2078_207827

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- States that 8 million in scientific notation is 8 * 10^6 -/
theorem eight_million_scientific_notation :
  to_scientific_notation 8000000 = ScientificNotation.mk 8 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_eight_million_scientific_notation_l2078_207827


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2078_207883

open Set

/-- The universal set U -/
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

/-- Set A -/
def A : Set Nat := {3, 4, 5}

/-- Set B -/
def B : Set Nat := {1, 3, 6}

/-- Theorem stating that the intersection of A and the complement of B in U equals {4, 5} -/
theorem intersection_A_complement_B : A ∩ (U \ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2078_207883


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l2078_207871

-- Define the type for planes
variable (Plane : Type)

-- Define the relations for parallel and perpendicular planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (α β γ : Plane) 
  (h1 : parallel α β) 
  (h2 : perpendicular β γ) : 
  perpendicular α γ :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l2078_207871


namespace NUMINAMATH_CALUDE_toy_store_shelves_l2078_207800

def shelves_required (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem toy_store_shelves :
  shelves_required 4 10 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l2078_207800


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2078_207872

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the asymptote of a hyperbola -/
def onAsymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity_theorem (h : Hyperbola) (a1 a2 b : Point) :
  a1.x = -h.a ∧ a1.y = 0 ∧  -- A₁ is left vertex
  a2.x = h.a ∧ a2.y = 0 ∧   -- A₂ is right vertex
  onAsymptote h b ∧         -- B is on asymptote
  angle a1 b a2 = π/3 →     -- ∠A₁BA₂ = 60°
  eccentricity h = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2078_207872


namespace NUMINAMATH_CALUDE_ratio_difference_l2078_207828

/-- Given two positive numbers in a 7:11 ratio where the smaller is 28, 
    prove that the larger exceeds the smaller by 16. -/
theorem ratio_difference (small large : ℝ) : 
  small > 0 ∧ large > 0 ∧ 
  large / small = 11 / 7 ∧ 
  small = 28 → 
  large - small = 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l2078_207828


namespace NUMINAMATH_CALUDE_exponent_division_l2078_207864

theorem exponent_division (a : ℝ) (ha : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2078_207864


namespace NUMINAMATH_CALUDE_correct_propositions_l2078_207813

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (linePerpPlane : Line → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)
variable (planePerpPlane : Plane → Plane → Prop)
variable (planeParallelPlane : Plane → Plane → Prop)
variable (planeIntersectLine : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem correct_propositions :
  (∀ (n : Line) (α β : Plane),
    linePerpPlane n α → linePerpPlane m β → parallel m n → planeParallelPlane α β) ∧
  (∀ (n : Line),
    planePerpPlane α β → planeIntersectLine α β m → lineInPlane n β → perpendicular n m → linePerpPlane n α) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l2078_207813


namespace NUMINAMATH_CALUDE_monotonic_cubic_range_l2078_207835

/-- The function f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ -/
def is_monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = -x^3 + ax^2 - x - 1 is monotonic on ℝ, then a ∈ [-√3, √3] -/
theorem monotonic_cubic_range (a : ℝ) :
  is_monotonic (fun x => -x^3 + a*x^2 - x - 1) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_range_l2078_207835


namespace NUMINAMATH_CALUDE_women_not_french_approx_97_14_percent_l2078_207861

/-- Represents the composition of employees in a company -/
structure Company where
  total_employees : ℕ
  men_percentage : ℚ
  men_french_percentage : ℚ
  total_french_percentage : ℚ

/-- Calculates the percentage of women who do not speak French in the company -/
def women_not_french_percentage (c : Company) : ℚ :=
  let women_percentage := 1 - c.men_percentage
  let men_french := c.men_percentage * c.men_french_percentage
  let women_french := c.total_french_percentage - men_french
  let women_not_french := women_percentage - women_french
  women_not_french / women_percentage

/-- Theorem stating that for a company with the given percentages,
    the percentage of women who do not speak French is approximately 97.14% -/
theorem women_not_french_approx_97_14_percent 
  (c : Company) 
  (h1 : c.men_percentage = 65/100)
  (h2 : c.men_french_percentage = 60/100)
  (h3 : c.total_french_percentage = 40/100) :
  ∃ ε > 0, |women_not_french_percentage c - 9714/10000| < ε :=
sorry

end NUMINAMATH_CALUDE_women_not_french_approx_97_14_percent_l2078_207861


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_l2078_207841

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def IsMonotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The function f(x) = x³ + x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

theorem monotonic_cubic_function (m : ℝ) :
  IsMonotonic (f m) ↔ m ∈ Set.Ici (1/3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_l2078_207841


namespace NUMINAMATH_CALUDE_quartic_polynomial_theorem_l2078_207830

-- Define a quartic polynomial
def is_quartic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Define the condition that p(n) = 1/n^2 for n = 1, 2, 3, 4, 5
def satisfies_condition (p : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n^2 : ℝ)

theorem quartic_polynomial_theorem (p : ℝ → ℝ) 
  (h1 : is_quartic_polynomial p) 
  (h2 : satisfies_condition p) : 
  p 6 = -67/180 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_theorem_l2078_207830


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_20_l2078_207849

theorem least_product_of_distinct_primes_above_20 :
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 20 ∧ q > 20 ∧ 
    p ≠ q ∧
    p * q = 667 ∧
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 20 → s > 20 → r ≠ s → r * s ≥ 667 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_20_l2078_207849


namespace NUMINAMATH_CALUDE_cakes_served_yesterday_l2078_207893

theorem cakes_served_yesterday (lunch_today dinner_today total : ℕ) 
  (h1 : lunch_today = 5)
  (h2 : dinner_today = 6)
  (h3 : total = 14) :
  total - (lunch_today + dinner_today) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_yesterday_l2078_207893


namespace NUMINAMATH_CALUDE_apples_ratio_proof_l2078_207877

theorem apples_ratio_proof (jim_apples jane_apples jerry_apples : ℕ) 
  (h1 : jim_apples = 20)
  (h2 : jane_apples = 60)
  (h3 : jerry_apples = 40) :
  (jim_apples + jane_apples + jerry_apples) / 3 / jim_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_ratio_proof_l2078_207877


namespace NUMINAMATH_CALUDE_expression_evaluation_l2078_207801

theorem expression_evaluation : 
  let f (x : ℚ) := (x + 2) / (x - 2)
  let g (x : ℚ) := (f x + 2) / (f x - 2)
  g (1/3) = -31/37 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2078_207801


namespace NUMINAMATH_CALUDE_probability_one_science_one_humanities_l2078_207806

def total_courses : ℕ := 5
def science_courses : ℕ := 3
def humanities_courses : ℕ := 2
def courses_chosen : ℕ := 2

theorem probability_one_science_one_humanities :
  (Nat.choose science_courses 1 * Nat.choose humanities_courses 1) / Nat.choose total_courses courses_chosen = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_science_one_humanities_l2078_207806


namespace NUMINAMATH_CALUDE_art_piece_original_price_l2078_207898

/-- Proves that the original purchase price of an art piece is $4000 -/
theorem art_piece_original_price :
  ∀ (original_price future_price : ℝ),
  future_price = 3 * original_price →
  future_price - original_price = 8000 →
  original_price = 4000 := by
sorry

end NUMINAMATH_CALUDE_art_piece_original_price_l2078_207898


namespace NUMINAMATH_CALUDE_third_candidate_votes_l2078_207886

theorem third_candidate_votes (total_votes : ℕ) (john_votes : ℕ) (james_percentage : ℚ) : 
  total_votes = 1150 →
  john_votes = 150 →
  james_percentage = 70 / 100 →
  ∃ (third_votes : ℕ), 
    third_votes = total_votes - john_votes - (james_percentage * (total_votes - john_votes)).floor ∧
    third_votes = john_votes + 150 :=
by sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l2078_207886


namespace NUMINAMATH_CALUDE_cricket_team_size_l2078_207854

/-- A cricket team with the following properties:
  * The team is 25 years old
  * The wicket keeper is 3 years older than the team
  * The average age of the remaining players (excluding team and wicket keeper) is 1 year less than the average age of the whole team
  * The average age of the team is 22 years
-/
structure CricketTeam where
  n : ℕ  -- number of team members
  team_age : ℕ
  wicket_keeper_age : ℕ
  avg_age : ℕ
  team_age_eq : team_age = 25
  wicket_keeper_age_eq : wicket_keeper_age = team_age + 3
  avg_age_eq : avg_age = 22
  remaining_avg_age : (n * avg_age - team_age - wicket_keeper_age) / (n - 2) = avg_age - 1

/-- The number of members in the cricket team is 11 -/
theorem cricket_team_size (team : CricketTeam) : team.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2078_207854


namespace NUMINAMATH_CALUDE_simplify_fraction_l2078_207878

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  (2 / (a^2 - 1)) * (1 / (a - 1)) = 2 / ((a - 1)^2 * (a + 1)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2078_207878


namespace NUMINAMATH_CALUDE_stone_count_is_odd_l2078_207832

/-- Represents the assembly of stones along a road -/
structure StoneAssembly where
  stone_interval : ℝ  -- Distance between stones in meters
  total_distance : ℝ  -- Total distance covered in meters

/-- Theorem: The total number of stones is odd given the conditions -/
theorem stone_count_is_odd (assembly : StoneAssembly) 
  (h_interval : assembly.stone_interval = 10)
  (h_distance : assembly.total_distance = 4800) : 
  ∃ (n : ℕ), (2 * n + 1) * assembly.stone_interval = assembly.total_distance / 2 := by
  sorry

#check stone_count_is_odd

end NUMINAMATH_CALUDE_stone_count_is_odd_l2078_207832


namespace NUMINAMATH_CALUDE_prob_face_or_ace_two_draws_l2078_207855

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (special_cards : ℕ)

/-- Calculates the probability of drawing at least one special card in two draws with replacement -/
def prob_at_least_one_special (d : Deck) : ℚ :=
  1 - (1 - d.special_cards / d.total_cards) ^ 2

/-- Theorem: The probability of drawing at least one face card or ace in two draws with replacement from a standard 52-card deck is 88/169 -/
theorem prob_face_or_ace_two_draws :
  let standard_deck : Deck := ⟨52, 16⟩
  prob_at_least_one_special standard_deck = 88 / 169 := by
  sorry


end NUMINAMATH_CALUDE_prob_face_or_ace_two_draws_l2078_207855


namespace NUMINAMATH_CALUDE_combinatorial_number_identity_l2078_207879

theorem combinatorial_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  Nat.choose n r = (n / r) * Nat.choose (n - 1) (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_number_identity_l2078_207879


namespace NUMINAMATH_CALUDE_bill_drew_140_lines_l2078_207823

/-- The number of lines drawn for a given shape --/
def lines_for_shape (num_shapes : ℕ) (sides_per_shape : ℕ) : ℕ :=
  num_shapes * sides_per_shape

/-- The total number of lines drawn by Bill --/
def total_lines : ℕ :=
  let triangles := lines_for_shape 12 3
  let squares := lines_for_shape 8 4
  let pentagons := lines_for_shape 4 5
  let hexagons := lines_for_shape 6 6
  let octagons := lines_for_shape 2 8
  triangles + squares + pentagons + hexagons + octagons

theorem bill_drew_140_lines : total_lines = 140 := by
  sorry

end NUMINAMATH_CALUDE_bill_drew_140_lines_l2078_207823


namespace NUMINAMATH_CALUDE_rectangle_to_square_transformation_l2078_207882

theorem rectangle_to_square_transformation (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 54 → (3 * a) * (b / 2) = 9^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_transformation_l2078_207882


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2078_207810

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 4 3
  y = 2 * x^2 → y = shifted.a * (x - 4)^2 + shifted.b * (x - 4) + shifted.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2078_207810


namespace NUMINAMATH_CALUDE_same_grade_percentage_is_42_5_l2078_207826

/-- Represents the number of students in the class -/
def total_students : ℕ := 40

/-- Represents the number of students who received the same grade on both tests -/
def same_grade_students : ℕ := 17

/-- Calculates the percentage of students who received the same grade on both tests -/
def same_grade_percentage : ℚ :=
  (same_grade_students : ℚ) / (total_students : ℚ) * 100

/-- Proves that the percentage of students who received the same grade on both tests is 42.5% -/
theorem same_grade_percentage_is_42_5 :
  same_grade_percentage = 42.5 := by
  sorry


end NUMINAMATH_CALUDE_same_grade_percentage_is_42_5_l2078_207826


namespace NUMINAMATH_CALUDE_wolves_heads_count_l2078_207803

/-- Represents the count of normal wolves -/
def normal_wolves : ℕ := sorry

/-- Represents the count of mutant wolves -/
def mutant_wolves : ℕ := sorry

/-- The total number of heads for all creatures -/
def total_heads : ℕ := 21

/-- The total number of legs for all creatures -/
def total_legs : ℕ := 57

/-- The number of heads a person has -/
def person_heads : ℕ := 1

/-- The number of legs a person has -/
def person_legs : ℕ := 2

/-- The number of heads a normal wolf has -/
def normal_wolf_heads : ℕ := 1

/-- The number of legs a normal wolf has -/
def normal_wolf_legs : ℕ := 4

/-- The number of heads a mutant wolf has -/
def mutant_wolf_heads : ℕ := 2

/-- The number of legs a mutant wolf has -/
def mutant_wolf_legs : ℕ := 3

theorem wolves_heads_count :
  normal_wolves * normal_wolf_heads + mutant_wolves * mutant_wolf_heads = total_heads - person_heads ∧
  normal_wolves * normal_wolf_legs + mutant_wolves * mutant_wolf_legs = total_legs - person_legs := by
  sorry

end NUMINAMATH_CALUDE_wolves_heads_count_l2078_207803


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2078_207840

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 4*i) / (1 + i) = -1/2 - 7/2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2078_207840


namespace NUMINAMATH_CALUDE_max_volume_inscribed_prism_l2078_207812

/-- The maximum volume of a right square prism inscribed in a right circular cone -/
theorem max_volume_inscribed_prism (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = π → ∃ (V : ℝ), V = (64 * Real.sqrt 3) / 27 ∧ 
  ∀ (a h : ℝ), 0 < a → a < 2 * r → 0 < h → 
  h ≤ Real.sqrt (4 * r ^ 2 - a ^ 2) → a ^ 2 * h ≤ V := by
  sorry

end NUMINAMATH_CALUDE_max_volume_inscribed_prism_l2078_207812


namespace NUMINAMATH_CALUDE_states_fraction_1840_to_1849_l2078_207850

theorem states_fraction_1840_to_1849 (total_states : ℕ) (joined_1840_to_1849 : ℕ) :
  total_states = 33 →
  joined_1840_to_1849 = 6 →
  (joined_1840_to_1849 : ℚ) / total_states = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_1840_to_1849_l2078_207850


namespace NUMINAMATH_CALUDE_next_simultaneous_occurrence_l2078_207847

def factory_whistle_period : ℕ := 18
def train_bell_period : ℕ := 30
def foghorn_period : ℕ := 45

def start_time : ℕ := 360  -- 6:00 a.m. in minutes since midnight

theorem next_simultaneous_occurrence :
  ∃ (t : ℕ), t > start_time ∧
  t % factory_whistle_period = 0 ∧
  t % train_bell_period = 0 ∧
  t % foghorn_period = 0 ∧
  t - start_time = 90 :=
sorry

end NUMINAMATH_CALUDE_next_simultaneous_occurrence_l2078_207847


namespace NUMINAMATH_CALUDE_bear_laps_in_scenario_l2078_207853

/-- Represents the number of laps completed by the bear in one hour -/
def bear_laps (lake_perimeter : ℝ) (salmon1_speed : ℝ) (salmon2_speed : ℝ) (bear_speed : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of laps completed by the bear in the given scenario -/
theorem bear_laps_in_scenario : 
  bear_laps 1000 500 750 200 = 7 := by sorry

end NUMINAMATH_CALUDE_bear_laps_in_scenario_l2078_207853


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2078_207881

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 160) 
  (diagonal : a^2 + b^2 + c^2 = 25^2) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2078_207881


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_ratio_l2078_207833

theorem complex_pure_imaginary_ratio (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (h : ∃ (y : ℝ), (5 - 3 * Complex.I) * (m + n * Complex.I) = y * Complex.I) : 
  m / n = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_ratio_l2078_207833


namespace NUMINAMATH_CALUDE_max_x_value_l2078_207874

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_sum_eq : x*y + x*z + y*z = 12) :
  x ≤ (13 + Real.sqrt 160) / 6 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l2078_207874


namespace NUMINAMATH_CALUDE_trapezoid_fg_squared_l2078_207845

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  /-- Length of EF -/
  ef : ℝ
  /-- Length of EH -/
  eh : ℝ
  /-- FG is perpendicular to EF and GH -/
  fg_perpendicular : Bool
  /-- Diagonals EG and FH are perpendicular -/
  diagonals_perpendicular : Bool

/-- Theorem about the length of FG in a specific trapezoid -/
theorem trapezoid_fg_squared (t : Trapezoid) 
  (h1 : t.ef = 3)
  (h2 : t.eh = Real.sqrt 2001)
  (h3 : t.fg_perpendicular = true)
  (h4 : t.diagonals_perpendicular = true) :
  ∃ (fg : ℝ), fg^2 = (9 + 3 * Real.sqrt 7977) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_fg_squared_l2078_207845


namespace NUMINAMATH_CALUDE_trivia_contest_probability_l2078_207848

/-- The number of questions in the trivia contest -/
def num_questions : ℕ := 5

/-- The number of possible answers for each question -/
def num_answers : ℕ := 5

/-- The probability of guessing a single question correctly -/
def p_correct : ℚ := 1 / num_answers

/-- The probability of guessing a single question incorrectly -/
def p_incorrect : ℚ := 1 - p_correct

/-- The probability of guessing all questions incorrectly -/
def p_all_incorrect : ℚ := p_incorrect ^ num_questions

/-- The probability of guessing at least one question correctly -/
def p_at_least_one_correct : ℚ := 1 - p_all_incorrect

theorem trivia_contest_probability :
  p_at_least_one_correct = 2101 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_trivia_contest_probability_l2078_207848


namespace NUMINAMATH_CALUDE_remainder_sum_l2078_207873

theorem remainder_sum (n : ℤ) (h : n % 18 = 4) : (n % 2 + n % 9 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2078_207873


namespace NUMINAMATH_CALUDE_minimal_distance_point_l2078_207831

/-- Given points A and B in ℝ², prove that P(0, 3) on the y-axis minimizes |PA| + |PB| -/
theorem minimal_distance_point (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  let P : ℝ × ℝ := (0, 3)
  (∀ Q : ℝ × ℝ, Q.1 = 0 → dist A P + dist B P ≤ dist A Q + dist B Q) :=
by sorry


end NUMINAMATH_CALUDE_minimal_distance_point_l2078_207831


namespace NUMINAMATH_CALUDE_clock_hands_opposite_period_l2078_207895

/-- The number of times clock hands are in opposite directions in 12 hours -/
def opposite_directions_per_12_hours : ℕ := 11

/-- The number of hours on a clock -/
def hours_on_clock : ℕ := 12

/-- The number of minutes between opposite directions -/
def minutes_between_opposite : ℕ := 30

/-- The observed number of times the hands are in opposite directions -/
def observed_opposite_directions : ℕ := 22

/-- The period in which the hands show opposite directions 22 times -/
def period : ℕ := 24

theorem clock_hands_opposite_period :
  opposite_directions_per_12_hours * 2 = observed_opposite_directions →
  period = 24 := by sorry

end NUMINAMATH_CALUDE_clock_hands_opposite_period_l2078_207895


namespace NUMINAMATH_CALUDE_factorial_22_representation_l2078_207863

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def base_ten_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem factorial_22_representation (V R C : ℕ) :
  V < 10 ∧ R < 10 ∧ C < 10 →
  base_ten_representation (factorial 22) =
    [1, 1, 2, 4, 0, V, 4, 6, 1, 7, 4, R, C, 8, 8, 0, 0, 0, 0] →
  V + R + C = 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_22_representation_l2078_207863
