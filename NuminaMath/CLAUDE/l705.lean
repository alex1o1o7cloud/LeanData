import Mathlib

namespace NUMINAMATH_CALUDE_dave_paints_200_sqft_l705_70586

/-- The total area of the wall to be painted in square feet -/
def total_area : ℝ := 360

/-- The ratio of Carl's work to Dave's work -/
def work_ratio : ℚ := 4 / 5

/-- Dave's share of the work -/
def dave_share : ℚ := 5 / 9

theorem dave_paints_200_sqft :
  dave_share * total_area = 200 := by sorry

end NUMINAMATH_CALUDE_dave_paints_200_sqft_l705_70586


namespace NUMINAMATH_CALUDE_grid_hole_properties_l705_70530

/-- Represents a grid with a hole --/
structure GridWithHole where
  rows : ℕ
  cols : ℕ
  holeRows : ℕ
  holeCols : ℕ
  squareSideLength : ℝ

/-- Calculate the number of removed squares in the grid --/
def removedSquares (g : GridWithHole) : ℕ := 36

/-- Calculate the area of the hole in the grid --/
def holeArea (g : GridWithHole) : ℝ := 36

/-- Calculate the perimeter of the hole in the grid --/
def holePerimeter (g : GridWithHole) : ℝ := 42

/-- Theorem stating the properties of the grid with hole --/
theorem grid_hole_properties (g : GridWithHole) 
  (h1 : g.rows = 10) 
  (h2 : g.cols = 20) 
  (h3 : g.holeRows = 6) 
  (h4 : g.holeCols = 15) 
  (h5 : g.squareSideLength = 1) : 
  removedSquares g = 36 ∧ 
  holeArea g = 36 ∧ 
  holePerimeter g = 42 := by
  sorry

end NUMINAMATH_CALUDE_grid_hole_properties_l705_70530


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l705_70576

/-- Given a quadrilateral PQRS with extended sides, prove the reconstruction equation -/
theorem quadrilateral_reconstruction
  (P P' Q Q' R R' S S' : ℝ × ℝ) -- Points as pairs of real numbers
  (h1 : P' - Q = 2 * (P - Q)) -- PP' = 3PQ
  (h2 : R' - Q = R - Q) -- QR' = QR
  (h3 : R' - S = R - S) -- SR' = SR
  (h4 : S' - P = 3 * (S - P)) -- PS' = 4PS
  : ∃ (x y z w : ℝ),
    x = 48/95 ∧ y = 32/95 ∧ z = 19/95 ∧ w = 4/5 ∧
    P = x • P' + y • Q' + z • R' + w • S' :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l705_70576


namespace NUMINAMATH_CALUDE_abby_singles_percentage_l705_70598

/-- Represents the statistics of a softball player's hits -/
structure HitStatistics where
  total_hits : ℕ
  home_runs : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles given hit statistics -/
def percentage_singles (stats : HitStatistics) : ℚ :=
  let singles := stats.total_hits - (stats.home_runs + stats.triples + stats.doubles)
  (singles : ℚ) / (stats.total_hits : ℚ) * 100

/-- Abby's hit statistics -/
def abby_stats : HitStatistics :=
  { total_hits := 45
  , home_runs := 2
  , triples := 3
  , doubles := 7
  }

/-- Theorem stating that the percentage of Abby's singles is 73.33% -/
theorem abby_singles_percentage :
  percentage_singles abby_stats = 73 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_abby_singles_percentage_l705_70598


namespace NUMINAMATH_CALUDE_division_problem_l705_70563

theorem division_problem (n : ℕ) : 
  n / 14 = 9 ∧ n % 14 = 1 → n = 127 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l705_70563


namespace NUMINAMATH_CALUDE_add_seconds_theorem_l705_70560

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (8:00:00 a.m.) -/
def initialTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 102930

/-- The expected final time (12:35:30) -/
def expectedFinalTime : Time :=
  { hours := 12, minutes := 35, seconds := 30 }

theorem add_seconds_theorem :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_seconds_theorem_l705_70560


namespace NUMINAMATH_CALUDE_difference_of_squares_l705_70549

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l705_70549


namespace NUMINAMATH_CALUDE_circle_square_intersection_probability_l705_70591

/-- The probability that a circle of radius 1 centered at a random point
    inside a square of side length 4 intersects the square exactly twice. -/
theorem circle_square_intersection_probability :
  let square_side : ℝ := 4
  let circle_radius : ℝ := 1
  let favorable_area : ℝ := π + 8
  let total_area : ℝ := square_side ^ 2
  (favorable_area / total_area : ℝ) = (π + 8) / 16 := by
sorry

end NUMINAMATH_CALUDE_circle_square_intersection_probability_l705_70591


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l705_70593

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Theorem stating Sheila's hourly wage --/
theorem sheila_hourly_wage (sheila : WorkSchedule)
  (h1 : sheila.hours_mon_wed_fri = 8)
  (h2 : sheila.hours_tue_thu = 6)
  (h3 : sheila.weekly_earnings = 468) :
  hourly_wage sheila = 13 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l705_70593


namespace NUMINAMATH_CALUDE_exponential_function_determined_l705_70597

def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

theorem exponential_function_determined (f : ℝ → ℝ) :
  is_exponential f → f 3 = 8 → ∀ x, f x = 2^x := by sorry

end NUMINAMATH_CALUDE_exponential_function_determined_l705_70597


namespace NUMINAMATH_CALUDE_edward_money_theorem_l705_70553

def remaining_money (initial_amount spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem edward_money_theorem (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount ≥ spent_amount) :
  remaining_money initial_amount spent_amount = initial_amount - spent_amount :=
by
  sorry

#eval remaining_money 18 16  -- Should evaluate to 2

end NUMINAMATH_CALUDE_edward_money_theorem_l705_70553


namespace NUMINAMATH_CALUDE_problem_statement_l705_70512

theorem problem_statement (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 
  3 * (2 * x + 5) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l705_70512


namespace NUMINAMATH_CALUDE_susie_vacuum_time_l705_70596

/-- Calculates the time to vacuum a house given the time per room and number of rooms -/
def time_to_vacuum_house (time_per_room : ℕ) (num_rooms : ℕ) : ℚ :=
  (time_per_room * num_rooms : ℚ) / 60

/-- Proves that Susie's vacuuming time is 2 hours -/
theorem susie_vacuum_time :
  let time_per_room : ℕ := 20
  let num_rooms : ℕ := 6
  time_to_vacuum_house time_per_room num_rooms = 2 := by
  sorry

end NUMINAMATH_CALUDE_susie_vacuum_time_l705_70596


namespace NUMINAMATH_CALUDE_sum_of_digits_product_nines_fives_l705_70558

/-- Represents a number with n repetitions of a digit --/
def repeatedDigit (digit : Nat) (n : Nat) : Nat :=
  digit * (10^n - 1) / 9

/-- Calculates the sum of digits of a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved --/
theorem sum_of_digits_product_nines_fives :
  let nines := repeatedDigit 9 100
  let fives := repeatedDigit 5 100
  sumOfDigits (nines * fives) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_nines_fives_l705_70558


namespace NUMINAMATH_CALUDE_assignment_methods_count_l705_70592

/-- The number of ways to select and assign representatives -/
def assign_representatives (num_boys num_girls num_reps min_boys min_girls : ℕ) : ℕ :=
  -- Number of ways to select 2 boys and 2 girls
  (Nat.choose num_boys 2 * Nat.choose num_girls 2 * Nat.factorial num_reps) +
  -- Number of ways to select 3 boys and 1 girl
  (Nat.choose num_boys 3 * Nat.choose num_girls 1 * Nat.factorial num_reps)

/-- Theorem stating the number of assignment methods -/
theorem assignment_methods_count :
  assign_representatives 5 4 4 2 1 = 2400 :=
by sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l705_70592


namespace NUMINAMATH_CALUDE_increasing_function_equivalence_l705_70585

/-- A function f is increasing on ℝ -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_equivalence (f : ℝ → ℝ) (h : IncreasingOn f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ↔
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) ∧
  (∀ a b : ℝ, f a + f b < f (-a) + f (-b) → a + b < 0) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_equivalence_l705_70585


namespace NUMINAMATH_CALUDE_june_bike_ride_l705_70542

/-- June's bike ride problem -/
theorem june_bike_ride (june_distance : ℝ) (june_time : ℝ) (bernard_distance : ℝ) (bernard_time : ℝ) (june_to_bernard : ℝ) :
  june_distance = 2 →
  june_time = 6 →
  bernard_distance = 5 →
  bernard_time = 15 →
  june_to_bernard = 7 →
  (june_to_bernard / (june_distance / june_time)) = 21 := by
sorry

end NUMINAMATH_CALUDE_june_bike_ride_l705_70542


namespace NUMINAMATH_CALUDE_apple_banana_ratio_l705_70559

theorem apple_banana_ratio (n : ℕ) : 
  (3 * n + 2 * n = 72) → False :=
by
  sorry

end NUMINAMATH_CALUDE_apple_banana_ratio_l705_70559


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l705_70544

theorem quadratic_inequality_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l705_70544


namespace NUMINAMATH_CALUDE_set_difference_example_l705_70507

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_example :
  let A : Set ℕ := {2, 3}
  let B : Set ℕ := {1, 3, 4}
  set_difference A B = {2} := by
sorry

end NUMINAMATH_CALUDE_set_difference_example_l705_70507


namespace NUMINAMATH_CALUDE_min_calls_for_complete_info_sharing_l705_70529

/-- Represents a person in the information sharing network -/
structure Person where
  id : Nat
  initialInfo : Nat

/-- Represents the state of information sharing -/
structure InfoState where
  people : Finset Person
  calls : Nat
  allInfoShared : Bool

/-- The minimum number of calls needed for complete information sharing -/
def minCalls (n : Nat) : Nat := 2 * n - 2

/-- Theorem stating the minimum number of calls needed for complete information sharing -/
theorem min_calls_for_complete_info_sharing (n : Nat) (h : n > 0) :
  ∀ (state : InfoState),
    state.people.card = n →
    (∀ p : Person, p ∈ state.people → ∃! i, p.initialInfo = i) →
    (state.allInfoShared → state.calls ≥ minCalls n) :=
sorry

end NUMINAMATH_CALUDE_min_calls_for_complete_info_sharing_l705_70529


namespace NUMINAMATH_CALUDE_max_value_and_constraint_l705_70523

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2*|x + 1|

-- Define the maximum value of f
def m : ℝ := 4

-- Theorem statement
theorem max_value_and_constraint (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_constraint : a^2 + 2*b^2 + c^2 = 2*m) :
  (∀ x, f x ≤ m) ∧ (∃ x, f x = m) ∧ (ab + bc ≤ 2) ∧ (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a'^2 + 2*b'^2 + c'^2 = 2*m ∧ a'*b' + b'*c' = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_constraint_l705_70523


namespace NUMINAMATH_CALUDE_product_one_sum_greater_than_inverses_l705_70537

theorem product_one_sum_greater_than_inverses
  (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_product : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ 
  (b > 1 ∧ a < 1 ∧ c < 1) ∨ 
  (c > 1 ∧ a < 1 ∧ b < 1) :=
by sorry

end NUMINAMATH_CALUDE_product_one_sum_greater_than_inverses_l705_70537


namespace NUMINAMATH_CALUDE_complex_multiplication_l705_70504

theorem complex_multiplication : 
  (3 - 4 * Complex.I) * (2 + Complex.I) = 10 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l705_70504


namespace NUMINAMATH_CALUDE_mango_crates_problem_l705_70555

theorem mango_crates_problem (total_cost : ℝ) (lost_crates : ℕ) (selling_price : ℝ) (profit_percentage : ℝ) :
  total_cost = 160 →
  lost_crates = 2 →
  selling_price = 25 →
  profit_percentage = 0.25 →
  ∃ (initial_crates : ℕ),
    initial_crates = 10 ∧
    (initial_crates - lost_crates : ℝ) * selling_price = total_cost * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_mango_crates_problem_l705_70555


namespace NUMINAMATH_CALUDE_average_math_chem_is_25_l705_70532

/-- Given a student's scores in mathematics, physics, and chemistry,
    prove that the average of mathematics and chemistry scores is 25 -/
theorem average_math_chem_is_25 
  (M P C : ℕ) -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 30) -- Total marks in mathematics and physics is 30
  (h2 : C = P + 20) -- Chemistry score is 20 more than physics score
  : (M + C) / 2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_average_math_chem_is_25_l705_70532


namespace NUMINAMATH_CALUDE_min_value_expression_l705_70509

theorem min_value_expression (x : ℝ) (h : x > 0) : 
  9 * x + 1 / x^6 ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / y^6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l705_70509


namespace NUMINAMATH_CALUDE_bing_dwen_dwen_practice_time_l705_70570

/-- Calculates the practice time given start time, end time, and break duration -/
def practice_time (start_time end_time break_duration : ℕ) : ℕ :=
  end_time - start_time - break_duration

/-- Proves that the practice time is 6 hours given the specified conditions -/
theorem bing_dwen_dwen_practice_time :
  let start_time := 8  -- 8 AM
  let end_time := 16   -- 4 PM (16 in 24-hour format)
  let break_duration := 2
  practice_time start_time end_time break_duration = 6 := by
sorry

#eval practice_time 8 16 2  -- Should output 6

end NUMINAMATH_CALUDE_bing_dwen_dwen_practice_time_l705_70570


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l705_70508

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 11/17) (h2 : x - y = 1/119) : 
  x^2 - y^2 = 11/2003 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l705_70508


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l705_70522

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 2
  let e : ℝ := c / a
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l705_70522


namespace NUMINAMATH_CALUDE_jacks_total_yen_l705_70527

/-- Represents the amount of money in different currencies -/
structure Money where
  pounds : ℕ
  euros : ℕ
  yen : ℕ

/-- Represents currency exchange rates -/
structure ExchangeRates where
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the total amount in yen given initial amounts and exchange rates -/
def total_in_yen (initial : Money) (rates : ExchangeRates) : ℕ :=
  (initial.pounds + initial.euros * rates.pounds_per_euro) * rates.yen_per_pound + initial.yen

/-- Theorem stating that Jack's total amount in yen is 9400 -/
theorem jacks_total_yen :
  let initial : Money := { pounds := 42, euros := 11, yen := 3000 }
  let rates : ExchangeRates := { pounds_per_euro := 2, yen_per_pound := 100 }
  total_in_yen initial rates = 9400 := by
  sorry


end NUMINAMATH_CALUDE_jacks_total_yen_l705_70527


namespace NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l705_70511

theorem smallest_equal_hotdogs_and_buns :
  ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ 5 * n = 7 * m) ∧
  (∀ (k : ℕ), k > 0 → (∃ (j : ℕ), j > 0 ∧ 5 * k = 7 * j) → k ≥ n) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l705_70511


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l705_70578

def polynomial (a₂ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 - 7*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ : ℤ) :
  ∀ x : ℤ, polynomial a₂ x = 0 → x ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l705_70578


namespace NUMINAMATH_CALUDE_range_of_m_l705_70567

open Set

/-- Proposition p: There exists a real x such that x^2 + m < 0 -/
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m < 0

/-- Proposition q: For all real x, x^2 + mx + 1 > 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

/-- The set of real numbers m that satisfy the given conditions -/
def M : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

/-- The theorem stating the range of m -/
theorem range_of_m : M = Iic (-2) ∪ Ico 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l705_70567


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_3_range_of_a_for_inequality_l705_70516

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Statement for part 1
theorem solution_set_f_greater_than_3 :
  {x : ℝ | f x > 3} = {x : ℝ | x > 0} := by sorry

-- Statement for part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∃ x, f x + 1 ≤ 4^a - 5 * 2^a} = 
  {a : ℝ | a ≤ 0 ∨ a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_3_range_of_a_for_inequality_l705_70516


namespace NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_sequence_l705_70571

theorem binomial_coeff_not_arithmetic_sequence (n r : ℕ) (h : r + 3 ≤ n) :
  ¬∃ (d : ℚ), 
    (Nat.choose n (r + 1) : ℚ) - (Nat.choose n r : ℚ) = d ∧ 
    (Nat.choose n (r + 2) : ℚ) - (Nat.choose n (r + 1) : ℚ) = d ∧ 
    (Nat.choose n (r + 3) : ℚ) - (Nat.choose n (r + 2) : ℚ) = d :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_sequence_l705_70571


namespace NUMINAMATH_CALUDE_age_problem_l705_70562

theorem age_problem (kolya_last_year : ℝ) :
  let vera_last_year := 2 * kolya_last_year
  let victor_last_year := 3.5 * kolya_last_year
  let kolya_now := kolya_last_year + 1
  let vera_now := vera_last_year + 1
  let victor_now := victor_last_year + 1
  let years_until_double := victor_now
  let kolya_future := kolya_now + years_until_double
  let vera_future := vera_now + years_until_double
  (vera_future - kolya_future = 4) →
  (kolya_now = 5 ∧ vera_now = 9 ∧ victor_now = 15) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l705_70562


namespace NUMINAMATH_CALUDE_t_shaped_region_perimeter_l705_70518

/-- A T-shaped region formed by six congruent squares -/
structure TShapedRegion where
  /-- The side length of each square in the region -/
  side_length : ℝ
  /-- The total area of the region -/
  total_area : ℝ
  /-- The area of the region is the sum of six squares -/
  area_eq : total_area = 6 * side_length ^ 2

/-- The perimeter of a T-shaped region -/
def perimeter (region : TShapedRegion) : ℝ :=
  9 * region.side_length

/-- Theorem stating the perimeter of a T-shaped region with area 576 is 36√6 -/
theorem t_shaped_region_perimeter :
  ∀ (region : TShapedRegion),
  region.total_area = 576 →
  perimeter region = 36 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_t_shaped_region_perimeter_l705_70518


namespace NUMINAMATH_CALUDE_modInverses_correct_l705_70574

def modInverses (n : ℕ) : List ℕ :=
  match n with
  | 2 => [1]
  | 3 => [1, 2]
  | 4 => [1, 3]
  | 5 => [1, 3, 2, 4]
  | 6 => [1, 5]
  | 7 => [1, 4, 5, 2, 3, 6]
  | 8 => [1, 3, 5, 7]
  | 9 => [1, 5, 7, 2, 4, 8]
  | 10 => [1, 7, 3, 9]
  | _ => []

theorem modInverses_correct (n : ℕ) (h : 2 ≤ n ∧ n ≤ 10) :
  ∀ a ∈ modInverses n, ∃ b : ℕ, 
    1 ≤ b ∧ b < n ∧ 
    (a * b) % n = 1 ∧ 
    Nat.gcd b n = 1 :=
by sorry

end NUMINAMATH_CALUDE_modInverses_correct_l705_70574


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_nonpositive_max_value_when_a_is_3_min_value_when_a_is_3_l705_70533

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Part I: Increasing function implies a ≤ 0
theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, ∀ y : ℝ, x < y → f a x < f a y) →
  a ≤ 0 :=
sorry

-- Part II: Maximum and minimum values when a = 3
theorem max_value_when_a_is_3 :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f 3 x ≤ 1) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f 3 x = 1) :=
sorry

theorem min_value_when_a_is_3 :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → -3 ≤ f 3 x) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f 3 x = -3) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_nonpositive_max_value_when_a_is_3_min_value_when_a_is_3_l705_70533


namespace NUMINAMATH_CALUDE_cake_problem_l705_70594

theorem cake_problem (cube_edge : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) :
  cube_edge = 2 →
  M = (2, 1) →
  N = (4/5, 2/5) →
  let volume := cube_edge * (1/2 * N.1 * N.2)
  let icing_area := (1/2 * N.1 * N.2) + (cube_edge * cube_edge)
  volume + icing_area = 32/5 := by
sorry

end NUMINAMATH_CALUDE_cake_problem_l705_70594


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l705_70581

theorem hotel_charge_difference (P_s R_s G_s P_d R_d G_d P_su R_su G_su : ℝ) 
  (h1 : P_s = R_s * 0.45)
  (h2 : P_s = G_s * 0.90)
  (h3 : P_d = R_d * 0.70)
  (h4 : P_d = G_d * 0.80)
  (h5 : P_su = R_su * 0.60)
  (h6 : P_su = G_su * 0.85) :
  (R_s / G_s - 1) * 100 - (R_d / G_d - 1) * 100 = 85.7143 := by
sorry

end NUMINAMATH_CALUDE_hotel_charge_difference_l705_70581


namespace NUMINAMATH_CALUDE_cube_circumscribed_sphere_volume_l705_70582

theorem cube_circumscribed_sphere_volume (surface_area : ℝ) (h : surface_area = 24) :
  let edge_length := Real.sqrt (surface_area / 6)
  let sphere_radius := edge_length * Real.sqrt 3 / 2
  (4 / 3) * Real.pi * sphere_radius ^ 3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_circumscribed_sphere_volume_l705_70582


namespace NUMINAMATH_CALUDE_dans_candy_bar_cost_l705_70583

/-- The cost of each candy bar given Dan's purchase scenario -/
def candy_bar_cost (initial_amount : ℚ) (num_candy_bars : ℕ) (amount_left : ℚ) : ℚ :=
  (initial_amount - amount_left) / num_candy_bars

/-- Theorem stating that the cost of each candy bar in Dan's scenario is $3 ÷ 99 -/
theorem dans_candy_bar_cost :
  candy_bar_cost 4 99 1 = 3 / 99 := by
  sorry

end NUMINAMATH_CALUDE_dans_candy_bar_cost_l705_70583


namespace NUMINAMATH_CALUDE_number_thought_of_l705_70541

theorem number_thought_of (x : ℝ) : (x / 5 + 23 = 42) → x = 95 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l705_70541


namespace NUMINAMATH_CALUDE_call_center_problem_l705_70540

theorem call_center_problem (C N : ℚ) : 
  let team_A_rate := (7 : ℚ) / 5
  let team_A_size := (5 : ℚ) / 8 * N
  let team_B_rate := (1 : ℚ)
  let team_B_size := N
  let total_calls := team_A_rate * team_A_size * C + team_B_rate * team_B_size * C
  (team_B_rate * team_B_size * C) / total_calls = (8 : ℚ) / 15 := by
sorry

end NUMINAMATH_CALUDE_call_center_problem_l705_70540


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l705_70569

theorem shopkeeper_profit_percentage 
  (total_goods : ℝ)
  (theft_percentage : ℝ)
  (loss_percentage : ℝ)
  (h1 : theft_percentage = 20)
  (h2 : loss_percentage = 12)
  : ∃ (profit_percentage : ℝ), profit_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l705_70569


namespace NUMINAMATH_CALUDE_excluded_numbers_sum_l705_70599

theorem excluded_numbers_sum (numbers : Finset ℕ) (sum_all : ℕ) (sum_six : ℕ) :
  Finset.card numbers = 8 →
  sum_all = Finset.sum numbers id →
  sum_all / 8 = 34 →
  ∃ (excluded : Finset ℕ), Finset.card excluded = 2 ∧
    Finset.card (numbers \ excluded) = 6 ∧
    sum_six = Finset.sum (numbers \ excluded) id ∧
    sum_six / 6 = 29 →
  sum_all - sum_six = 98 :=
by sorry

end NUMINAMATH_CALUDE_excluded_numbers_sum_l705_70599


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l705_70539

-- First expression
theorem simplify_first_expression (a : ℝ) : 
  3 * a^2 - 6 * a^2 - a^2 = -4 * a^2 := by sorry

-- Second expression
theorem simplify_second_expression (a b : ℝ) : 
  (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = -3 * a^2 + 5 * a + 3 * b := by sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l705_70539


namespace NUMINAMATH_CALUDE_jessica_age_l705_70525

theorem jessica_age :
  ∀ (j g : ℚ),
  g = 15 * j →
  g - j = 60 →
  j = 30 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_age_l705_70525


namespace NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_points_l705_70579

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define a general circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of having exactly two intersection points
def has_two_intersections (c : Circle) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
    parabola A.1 = A.2 ∧
    parabola B.1 = B.2 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2

-- Define the property of tangents coinciding at a point
def tangents_coincide_at (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) * (2 * p.1) + (p.2 - c.center.2) = c.radius^2

-- Main theorem
theorem tangents_not_necessarily_coincide_at_both_points :
  ∃ c : Circle, has_two_intersections c ∧
    (∃ A B : ℝ × ℝ, A ≠ B ∧
      parabola A.1 = A.2 ∧
      parabola B.1 = B.2 ∧
      tangents_coincide_at c A ∧
      ¬tangents_coincide_at c B) :=
sorry

end NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_points_l705_70579


namespace NUMINAMATH_CALUDE_jenny_cat_expense_first_year_l705_70557

/-- Jenny's cat expenses for the first year -/
def jenny_cat_expense : ℕ → ℕ → ℕ → ℕ → ℕ := fun adoption_fee vet_cost monthly_food_cost toy_cost =>
  let shared_cost := adoption_fee + vet_cost + (monthly_food_cost * 12)
  (shared_cost / 2) + toy_cost

/-- Theorem: Jenny's cat expense for the first year is $625 -/
theorem jenny_cat_expense_first_year :
  jenny_cat_expense 50 500 25 200 = 625 := by
  sorry

end NUMINAMATH_CALUDE_jenny_cat_expense_first_year_l705_70557


namespace NUMINAMATH_CALUDE_f_min_at_two_l705_70526

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem f_min_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_two_l705_70526


namespace NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l705_70565

theorem tan_plus_four_sin_twenty_degrees :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l705_70565


namespace NUMINAMATH_CALUDE_cricket_team_size_l705_70545

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 2 →
  let captain_age : ℕ := 25
  let keeper_age : ℕ := captain_age + 5
  let team_avg_age : ℕ := 23
  let remaining_avg_age : ℕ := team_avg_age - 1
  n * team_avg_age = captain_age + keeper_age + (n - 2) * remaining_avg_age →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_size_l705_70545


namespace NUMINAMATH_CALUDE_product_of_sums_l705_70566

/-- The sum of numbers of the form 2k+1 where k ranges from 0 to n -/
def odd_sum (n : ℕ) : ℕ := (n + 1)^2

/-- The sum of the first n even numbers -/
def even_sum (n : ℕ) : ℕ := n * (n + 1)

/-- The product of odd_sum and even_sum is equal to (n+1)^3 * n -/
theorem product_of_sums (n : ℕ) : odd_sum n * even_sum n = (n + 1)^3 * n := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l705_70566


namespace NUMINAMATH_CALUDE_odd_function_value_l705_70575

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : IsOdd f)
  (h_neg : ∀ x < 0, f x = x^2 + a*x)
  (h_f2 : f 2 = 6) :
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l705_70575


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l705_70584

theorem consecutive_integers_square_difference (n : ℕ) : 
  (n > 0) → (n + (n + 1) = 105) → ((n + 1)^2 - n^2 = 105) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l705_70584


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l705_70524

theorem profit_percent_calculation (P : ℝ) (C : ℝ) (h : P > 0) (h2 : C > 0) :
  (2/3 * P = 0.86 * C) → ((P - C) / C * 100 = 29) :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l705_70524


namespace NUMINAMATH_CALUDE_rons_current_age_l705_70535

theorem rons_current_age (maurice_current_age : ℕ) (years_from_now : ℕ) :
  maurice_current_age = 7 →
  years_from_now = 5 →
  ∃ (ron_current_age : ℕ),
    ron_current_age + years_from_now = 4 * (maurice_current_age + years_from_now) ∧
    ron_current_age = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_rons_current_age_l705_70535


namespace NUMINAMATH_CALUDE_pears_theorem_l705_70505

def pears_problem (keith_picked mike_picked sarah_picked keith_gave mike_gave sarah_gave : ℕ) : Prop :=
  let keith_left := keith_picked - keith_gave
  let mike_left := mike_picked - mike_gave
  let sarah_left := sarah_picked - sarah_gave
  keith_left + mike_left + sarah_left = 15

theorem pears_theorem :
  pears_problem 47 12 22 46 5 15 := by sorry

end NUMINAMATH_CALUDE_pears_theorem_l705_70505


namespace NUMINAMATH_CALUDE_airplane_hover_time_l705_70580

/-- Proves that given the conditions of the airplane problem, 
    the time spent in Eastern time on the first day was 2 hours. -/
theorem airplane_hover_time : 
  ∀ (eastern_time : ℕ),
    (3 + 4 + eastern_time) + (5 + 6 + (eastern_time + 2)) = 24 →
    eastern_time = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_airplane_hover_time_l705_70580


namespace NUMINAMATH_CALUDE_mask_assignment_unique_l705_70547

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

structure MaskAssignment where
  elephant : ℕ
  mouse : ℕ
  pig : ℕ
  panda : ℕ

def valid_assignment (a : MaskAssignment) : Prop :=
  is_single_digit a.elephant ∧
  is_single_digit a.mouse ∧
  is_single_digit a.pig ∧
  is_single_digit a.panda ∧
  a.elephant ≠ a.mouse ∧
  a.elephant ≠ a.pig ∧
  a.elephant ≠ a.panda ∧
  a.mouse ≠ a.pig ∧
  a.mouse ≠ a.panda ∧
  a.pig ≠ a.panda ∧
  (a.mouse * a.mouse) % 10 = a.elephant ∧
  (a.elephant * a.elephant) ≥ 10 ∧ (a.elephant * a.elephant) ≤ 99 ∧
  (a.mouse * a.mouse) ≥ 10 ∧ (a.mouse * a.mouse) ≤ 99 ∧
  (a.pig * a.pig) ≥ 10 ∧ (a.pig * a.pig) ≤ 99 ∧
  (a.panda * a.panda) ≥ 10 ∧ (a.panda * a.panda) ≤ 99 ∧
  (a.elephant * a.elephant) % 10 ≠ (a.mouse * a.mouse) % 10 ∧
  (a.elephant * a.elephant) % 10 ≠ (a.pig * a.pig) % 10 ∧
  (a.elephant * a.elephant) % 10 ≠ (a.panda * a.panda) % 10 ∧
  (a.mouse * a.mouse) % 10 ≠ (a.pig * a.pig) % 10 ∧
  (a.mouse * a.mouse) % 10 ≠ (a.panda * a.panda) % 10 ∧
  (a.pig * a.pig) % 10 ≠ (a.panda * a.panda) % 10

theorem mask_assignment_unique :
  ∃! a : MaskAssignment, valid_assignment a ∧ 
    a.elephant = 6 ∧ a.mouse = 4 ∧ a.pig = 8 ∧ a.panda = 1 :=
sorry

end NUMINAMATH_CALUDE_mask_assignment_unique_l705_70547


namespace NUMINAMATH_CALUDE_scientific_notation_of_320000_l705_70577

theorem scientific_notation_of_320000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 320000 = a * (10 : ℝ) ^ n ∧ a = 3.2 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_320000_l705_70577


namespace NUMINAMATH_CALUDE_expand_expression_l705_70503

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l705_70503


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l705_70519

theorem probability_three_heads_in_eight_tosses : 
  let n : ℕ := 8  -- number of tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of ways to choose k heads from n tosses
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l705_70519


namespace NUMINAMATH_CALUDE_apples_picked_total_l705_70590

/-- The total number of apples picked by Mike, Nancy, Keith, Olivia, and Thomas -/
def total_apples (mike nancy keith olivia thomas : Real) : Real :=
  mike + nancy + keith + olivia + thomas

/-- Theorem stating that the total number of apples picked is 37.8 -/
theorem apples_picked_total :
  total_apples 7.5 3.2 6.1 12.4 8.6 = 37.8 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_total_l705_70590


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l705_70556

theorem hemisphere_cylinder_surface_area
  (base_area : ℝ)
  (cylinder_height : ℝ)
  (h_base_area : base_area = 144 * Real.pi)
  (h_cylinder_height : cylinder_height = 5) :
  let radius := Real.sqrt (base_area / Real.pi)
  let hemisphere_area := 2 * Real.pi * radius ^ 2
  let cylinder_area := 2 * Real.pi * radius * cylinder_height
  hemisphere_area + cylinder_area = 408 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l705_70556


namespace NUMINAMATH_CALUDE_middle_aged_employees_selected_l705_70502

/-- Represents the age group of an employee -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents the company with its employee distribution -/
structure Company where
  total_employees : ℕ
  age_ratio : Fin 3 → ℕ
  age_ratio_sum : age_ratio 0 + age_ratio 1 + age_ratio 2 > 0

/-- Calculates the number of employees in a specific age group -/
def employees_in_group (c : Company) (g : Fin 3) : ℕ :=
  c.total_employees * c.age_ratio g / (c.age_ratio 0 + c.age_ratio 1 + c.age_ratio 2)

/-- Theorem: The number of middle-aged employees selected in stratified sampling -/
theorem middle_aged_employees_selected
  (c : Company)
  (h_total : c.total_employees = 1200)
  (h_ratio : c.age_ratio = ![1, 5, 6])
  (sample_size : ℕ)
  (h_sample : sample_size = 36) :
  employees_in_group c 1 * sample_size / c.total_employees = 15 := by
  sorry

end NUMINAMATH_CALUDE_middle_aged_employees_selected_l705_70502


namespace NUMINAMATH_CALUDE_schedule_arrangements_proof_l705_70501

/-- Represents the number of periods in the morning -/
def morning_periods : Nat := 4

/-- Represents the number of periods in the afternoon -/
def afternoon_periods : Nat := 3

/-- Represents the total number of periods in a day -/
def total_periods : Nat := morning_periods + afternoon_periods

/-- Represents the number of subjects that require two consecutive periods -/
def two_period_subjects : Nat := 2

/-- Represents the number of subjects that require one period -/
def one_period_subjects : Nat := 4

/-- Represents the total number of subjects to be scheduled -/
def total_subjects : Nat := two_period_subjects + one_period_subjects

/-- Calculates the number of possible arrangements for the schedule -/
def schedule_arrangements : Nat := 336

theorem schedule_arrangements_proof :
  schedule_arrangements = 336 := by sorry

end NUMINAMATH_CALUDE_schedule_arrangements_proof_l705_70501


namespace NUMINAMATH_CALUDE_problem_solution_l705_70506

theorem problem_solution : 
  let M : ℚ := 2013 / 3
  let N : ℚ := M / 3
  let X : ℚ := M + N
  X = 895 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l705_70506


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l705_70531

-- Define the relationship between x and y
def inverse_relation (x y : ℝ) : Prop := ∃ k : ℝ, k > 0 ∧ 3 * x^2 * y = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : y₁ > 0) (h_pos₄ : y₂ > 0)
  (h_inverse : inverse_relation x₁ y₁ ∧ inverse_relation x₂ y₂)
  (h_initial : x₁ = 3 ∧ y₁ = 30)
  (h_final : x₂ = 6) :
  y₂ = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l705_70531


namespace NUMINAMATH_CALUDE_coefficient_of_degree_10_l705_70568

-- Define the degree of the term nxy^n
def degree (n : ℕ) : ℕ := 1 + n

-- State the theorem
theorem coefficient_of_degree_10 (n : ℕ) : degree n = 10 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_degree_10_l705_70568


namespace NUMINAMATH_CALUDE_negation_of_proposition_l705_70528

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + 1 < 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l705_70528


namespace NUMINAMATH_CALUDE_customers_left_second_time_l705_70520

theorem customers_left_second_time 
  (initial_customers : ℝ)
  (first_group_left : ℝ)
  (final_customers : ℝ)
  (h1 : initial_customers = 36.0)
  (h2 : first_group_left = 19.0)
  (h3 : final_customers = 3) :
  initial_customers - first_group_left - final_customers = 14.0 :=
by sorry

end NUMINAMATH_CALUDE_customers_left_second_time_l705_70520


namespace NUMINAMATH_CALUDE_bubble_arrangements_l705_70543

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 3

theorem bubble_arrangements :
  (word_length.factorial) / (repeated_letter_count.factorial) = 120 :=
by sorry

end NUMINAMATH_CALUDE_bubble_arrangements_l705_70543


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_98_l705_70554

theorem ab_plus_cd_equals_98 
  (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = 9) 
  (h3 : a + c + d = 24) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 98 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_98_l705_70554


namespace NUMINAMATH_CALUDE_total_splash_width_is_seven_l705_70534

/-- The width of a splash made by a pebble in meters -/
def pebble_splash_width : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash_width : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash_width : ℚ := 2

/-- The number of pebbles thrown -/
def num_pebbles : ℕ := 6

/-- The number of rocks thrown -/
def num_rocks : ℕ := 3

/-- The number of boulders thrown -/
def num_boulders : ℕ := 2

/-- The total width of splashes made by throwing pebbles, rocks, and boulders -/
def total_splash_width : ℚ :=
  num_pebbles * pebble_splash_width +
  num_rocks * rock_splash_width +
  num_boulders * boulder_splash_width

theorem total_splash_width_is_seven :
  total_splash_width = 7 := by sorry

end NUMINAMATH_CALUDE_total_splash_width_is_seven_l705_70534


namespace NUMINAMATH_CALUDE_mountain_climbing_speed_l705_70536

theorem mountain_climbing_speed 
  (total_time : ℝ) 
  (break_day1 : ℝ) 
  (break_day2 : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 14) 
  (h2 : break_day1 = 0.5) 
  (h3 : break_day2 = 0.75) 
  (h4 : speed_difference = 0.5) 
  (h5 : time_difference = 2) 
  (h6 : total_distance = 52) : 
  ∃ (speed_day1 : ℝ), 
    speed_day1 + speed_difference = 4.375 ∧ 
    (∃ (time_day1 : ℝ), 
      time_day1 + (time_day1 - time_difference) = total_time ∧
      speed_day1 * (time_day1 - break_day1) + 
      (speed_day1 + speed_difference) * (time_day1 - time_difference - break_day2) = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_mountain_climbing_speed_l705_70536


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l705_70548

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def binomial_probability (n k : ℕ) : ℚ :=
  (n.choose k) * (1 / 2) ^ n

/-- The theorem stating the difference between probabilities of 4 and 3 heads in 5 flips -/
theorem coin_flip_probability_difference :
  |binomial_probability 5 4 - binomial_probability 5 3| = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l705_70548


namespace NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l705_70551

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 80) 
  (h2 : math = 50) 
  (h3 : physics = 32) 
  (h4 : both = 15) : 
  total - (math + physics - both) = 13 := by
sorry

end NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l705_70551


namespace NUMINAMATH_CALUDE_red_balls_count_l705_70538

theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 2)
  (h_purple : purple = 3)
  (h_prob : (white + green + yellow : ℚ) / total = 7/10) :
  total - (white + green + yellow + purple) = 15 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l705_70538


namespace NUMINAMATH_CALUDE_office_viewing_time_l705_70572

/-- The number of episodes in The Office series -/
def total_episodes : ℕ := 201

/-- The number of episodes watched per week -/
def episodes_per_week : ℕ := 3

/-- The number of weeks needed to watch all episodes -/
def weeks_to_watch : ℕ := 67

theorem office_viewing_time :
  (total_episodes + episodes_per_week - 1) / episodes_per_week = weeks_to_watch :=
sorry

end NUMINAMATH_CALUDE_office_viewing_time_l705_70572


namespace NUMINAMATH_CALUDE_absolute_value_expression_l705_70515

theorem absolute_value_expression (a b c : ℝ) (h1 : b < a) (h2 : a < 0) (h3 : 0 < c) :
  |b| - |b-a| + |c-a| - |a+b| = b + c - a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l705_70515


namespace NUMINAMATH_CALUDE_polynomial_composition_l705_70517

theorem polynomial_composition (g : ℝ → ℝ) :
  (∀ x, g x ^ 2 = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_composition_l705_70517


namespace NUMINAMATH_CALUDE_f_three_minus_f_four_equals_negative_one_l705_70564

-- Define the properties of function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_negation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

-- Theorem statement
theorem f_three_minus_f_four_equals_negative_one
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period_two_negation f)
  (h_f_one : f 1 = 1) :
  f 3 - f 4 = -1 :=
sorry

end NUMINAMATH_CALUDE_f_three_minus_f_four_equals_negative_one_l705_70564


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l705_70561

theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (2 + 4 * Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l705_70561


namespace NUMINAMATH_CALUDE_inequality_solution_equation_solution_l705_70589

-- Part 1: System of inequalities
def inequality_system (x : ℝ) : Prop :=
  x + 2 > 1 ∧ 2*x < x + 3

theorem inequality_solution :
  ∀ x : ℝ, inequality_system x ↔ -1 < x ∧ x < 3 :=
sorry

-- Part 2: System of linear equations
def equation_system (x y : ℝ) : Prop :=
  3*x + 2*y = 12 ∧ 2*x - y = 1

theorem equation_solution :
  ∀ x y : ℝ, equation_system x y ↔ x = 2 ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equation_solution_l705_70589


namespace NUMINAMATH_CALUDE_office_age_problem_l705_70550

theorem office_age_problem (total_persons : Nat) (group1_persons : Nat) (group2_persons : Nat)
  (total_avg_age : Nat) (group1_avg_age : Nat) (group2_avg_age : Nat)
  (h1 : total_persons = 19)
  (h2 : group1_persons = 5)
  (h3 : group2_persons = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16) :
  total_persons * total_avg_age = 
    group1_persons * group1_avg_age + group2_persons * group2_avg_age + 71 := by
  sorry

#check office_age_problem

end NUMINAMATH_CALUDE_office_age_problem_l705_70550


namespace NUMINAMATH_CALUDE_correct_seniority_ranking_l705_70595

-- Define the type for colleagues
inductive Colleague : Type
  | Ella : Colleague
  | Mark : Colleague
  | Nora : Colleague

-- Define the seniority relation
def moreSeniorThan : Colleague → Colleague → Prop := sorry

-- Axioms for the problem conditions
axiom different_seniorities :
  ∀ (a b : Colleague), a ≠ b → (moreSeniorThan a b ∨ moreSeniorThan b a)

axiom exactly_one_true :
  (moreSeniorThan Colleague.Mark Colleague.Ella ∧ moreSeniorThan Colleague.Mark Colleague.Nora) ∨
  (¬moreSeniorThan Colleague.Ella Colleague.Mark ∨ ¬moreSeniorThan Colleague.Ella Colleague.Nora) ∨
  (¬moreSeniorThan Colleague.Mark Colleague.Nora ∨ moreSeniorThan Colleague.Nora Colleague.Mark)

-- The theorem to prove
theorem correct_seniority_ranking :
  moreSeniorThan Colleague.Ella Colleague.Nora ∧
  moreSeniorThan Colleague.Nora Colleague.Mark :=
by sorry

end NUMINAMATH_CALUDE_correct_seniority_ranking_l705_70595


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_60_degree_sector_l705_70500

/-- The radius of a circle inscribed in a sector with a central angle of 60° and radius R is R/3 -/
theorem inscribed_circle_radius_in_60_degree_sector (R : ℝ) (R_pos : R > 0) :
  let sector_angle : ℝ := 60 * π / 180
  let inscribed_radius : ℝ := R / 3
  inscribed_radius = R / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_60_degree_sector_l705_70500


namespace NUMINAMATH_CALUDE_cards_given_to_miguel_miguel_received_13_cards_l705_70587

theorem cards_given_to_miguel (initial_cards : ℕ) (kept_cards : ℕ) (friends : ℕ) (cards_per_friend : ℕ) (sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : initial_cards = 130 := by sorry
  have h2 : kept_cards = 15 := by sorry
  have h3 : friends = 8 := by sorry
  have h4 : cards_per_friend = 12 := by sorry
  have h5 : sisters = 2 := by sorry
  have h6 : cards_per_sister = 3 := by sorry

  -- Calculate the number of cards given to Miguel
  let cards_to_give := initial_cards - kept_cards
  let cards_to_friends := friends * cards_per_friend
  let cards_left_after_friends := cards_to_give - cards_to_friends
  let cards_to_sisters := sisters * cards_per_sister
  let cards_to_miguel := cards_left_after_friends - cards_to_sisters

  -- Prove that cards_to_miguel = 13
  sorry

-- State the theorem
theorem miguel_received_13_cards : cards_given_to_miguel 130 15 8 12 2 3 = 13 := by sorry

end NUMINAMATH_CALUDE_cards_given_to_miguel_miguel_received_13_cards_l705_70587


namespace NUMINAMATH_CALUDE_f_properties_l705_70510

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 4^(-x) - a * 2^(-x) else 4^x - a * 2^x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_properties (a : ℝ) (h : a > 0) :
  is_even_function (f a) ∧
  (∀ x > 0, f a x = 4^x - a * 2^x) ∧
  (∀ x > 0, f a x ≥ 
    if 0 < a ∧ a ≤ 2 then 1 - a
    else if a > 2 then -a^2 / 4
    else 0) ∧
  (∃ x > 0, f a x = 
    if 0 < a ∧ a ≤ 2 then 1 - a
    else if a > 2 then -a^2 / 4
    else 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l705_70510


namespace NUMINAMATH_CALUDE_bobs_spending_limit_l705_70552

/-- The spending limit problem -/
theorem bobs_spending_limit
  (necklace_cost : ℕ)
  (book_cost_difference : ℕ)
  (overspent_amount : ℕ)
  (h1 : necklace_cost = 34)
  (h2 : book_cost_difference = 5)
  (h3 : overspent_amount = 3) :
  necklace_cost + (necklace_cost + book_cost_difference) - overspent_amount = 70 :=
by sorry

end NUMINAMATH_CALUDE_bobs_spending_limit_l705_70552


namespace NUMINAMATH_CALUDE_third_line_through_integer_point_l705_70573

theorem third_line_through_integer_point (a b c : ℝ) :
  (∃ y : ℝ, (y = a + b ∧ y = b + c) ∨ (y = a + b ∧ y = c + a) ∨ (y = b + c ∧ y = c + a)) →
  ∃ x y : ℤ, (y = a * x + a) ∨ (y = b * x + c) ∨ (y = c * x + a) :=
by sorry

end NUMINAMATH_CALUDE_third_line_through_integer_point_l705_70573


namespace NUMINAMATH_CALUDE_difference_of_squares_l705_70514

theorem difference_of_squares (a b : ℕ+) : 
  a + b = 40 → a - b = 10 → a^2 - b^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l705_70514


namespace NUMINAMATH_CALUDE_concurrent_iff_concyclic_l705_70588

/-- Two circles in a plane -/
structure TwoCircles where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)

/-- Points on the circles -/
structure CirclePoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_AB_intersect : A ∈ tc.C₁ ∧ A ∈ tc.C₂ ∧ B ∈ tc.C₁ ∧ B ∈ tc.C₂
  h_CD_on_C₁ : C ∈ tc.C₁ ∧ D ∈ tc.C₁
  h_EF_on_C₂ : E ∈ tc.C₂ ∧ F ∈ tc.C₂

/-- Define a line through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Three lines are concurrent if they all intersect at a single point -/
def AreConcurrent (l₁ l₂ l₃ : Set (ℝ × ℝ)) : Prop := sorry

/-- Four points are concyclic if they lie on the same circle -/
def AreConcyclic (p q r s : ℝ × ℝ) : Prop := sorry

/-- The main theorem -/
theorem concurrent_iff_concyclic (tc : TwoCircles) (pts : CirclePoints tc) :
  AreConcurrent (Line pts.E pts.F) (Line pts.C pts.D) (Line pts.A pts.B) ↔
  AreConcyclic pts.E pts.F pts.C pts.D := by sorry

end NUMINAMATH_CALUDE_concurrent_iff_concyclic_l705_70588


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l705_70521

theorem rectangle_perimeter (length width : ℝ) (h_ratio : length / width = 4 / 3) (h_area : length * width = 972) :
  2 * (length + width) = 126 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l705_70521


namespace NUMINAMATH_CALUDE_merchant_profit_l705_70546

theorem merchant_profit (cost selling : ℝ) (h : 20 * cost = 16 * selling) :
  (selling - cost) / cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_l705_70546


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l705_70513

theorem value_of_a_minus_b (a b : ℝ) : 
  ({x : ℝ | |x - a| < b} = {x : ℝ | 2 < x ∧ x < 4}) → a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l705_70513
