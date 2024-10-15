import Mathlib

namespace NUMINAMATH_CALUDE_maze_navigation_ways_l2753_275397

/-- Converts a list of digits in base 6 to a number in base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The number of ways the dog can navigate through the maze in base 6 -/
def mazeWaysBase6 : List Nat := [4, 1, 2, 5]

/-- Theorem: The number of ways the dog can navigate through the maze
    is 1162 when converted from base 6 to base 10 -/
theorem maze_navigation_ways :
  base6ToBase10 mazeWaysBase6 = 1162 := by
  sorry

end NUMINAMATH_CALUDE_maze_navigation_ways_l2753_275397


namespace NUMINAMATH_CALUDE_sqrt_y_fourth_power_l2753_275395

theorem sqrt_y_fourth_power (y : ℝ) (h : (Real.sqrt y) ^ 4 = 256) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_y_fourth_power_l2753_275395


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_extrema_l2753_275321

theorem sum_of_reciprocal_extrema (x y : ℝ) : 
  (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → 
  let S := x^2 + y^2
  ∃ (S_max S_min : ℝ), 
    (∀ (x' y' : ℝ), (4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5) → x'^2 + y'^2 ≤ S_max) ∧
    (∀ (x' y' : ℝ), (4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5) → S_min ≤ x'^2 + y'^2) ∧
    (1 / S_max + 1 / S_min = 8 / 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_extrema_l2753_275321


namespace NUMINAMATH_CALUDE_square_side_length_l2753_275333

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s * Real.sqrt 2 = d ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2753_275333


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l2753_275316

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 36) :
  x^3 * y^4 ≤ 18^3 * 6^4 ∧ (x^3 * y^4 = 18^3 * 6^4 ↔ x = 18 ∧ y = 6) :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l2753_275316


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2753_275365

/-- Represents the size of each stratum in the population -/
structure StratumSize where
  under30 : ℕ
  between30and40 : ℕ
  over40 : ℕ

/-- Represents the sample size for each stratum -/
structure StratumSample where
  under30 : ℕ
  between30and40 : ℕ
  over40 : ℕ

/-- Calculates the stratified sample size for a given population and total sample size -/
def stratifiedSample (populationSize : ℕ) (sampleSize : ℕ) (strata : StratumSize) : StratumSample :=
  { under30 := sampleSize * strata.under30 / populationSize,
    between30and40 := sampleSize * strata.between30and40 / populationSize,
    over40 := sampleSize * strata.over40 / populationSize }

theorem stratified_sampling_theorem (populationSize : ℕ) (sampleSize : ℕ) (strata : StratumSize) :
  populationSize = 100 →
  sampleSize = 20 →
  strata.under30 = 20 →
  strata.between30and40 = 60 →
  strata.over40 = 20 →
  let sample := stratifiedSample populationSize sampleSize strata
  sample.under30 = 4 ∧ sample.between30and40 = 12 ∧ sample.over40 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2753_275365


namespace NUMINAMATH_CALUDE_smallest_y_squared_value_l2753_275390

/-- Represents an isosceles trapezoid EFGH with a tangent circle -/
structure IsoscelesTrapezoidWithTangentCircle where
  EF : ℝ
  GH : ℝ
  y : ℝ
  is_isosceles : EF > GH
  tangent_circle : Bool

/-- The smallest possible value of y^2 in the given configuration -/
def smallest_y_squared (t : IsoscelesTrapezoidWithTangentCircle) : ℝ := sorry

/-- Theorem stating the smallest possible value of y^2 -/
theorem smallest_y_squared_value 
  (t : IsoscelesTrapezoidWithTangentCircle) 
  (h1 : t.EF = 102) 
  (h2 : t.GH = 26) 
  (h3 : t.tangent_circle = true) : 
  smallest_y_squared t = 1938 := by sorry

end NUMINAMATH_CALUDE_smallest_y_squared_value_l2753_275390


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_81_l2753_275345

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_81_l2753_275345


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2753_275391

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x < 18 →
  x + 4*x > 18 →
  x + 18 > 4*x →
  4*x + 18 > x →
  (∀ y : ℕ, y > x → y + 4*y ≤ 18 ∨ y + 18 ≤ 4*y ∨ 4*y + 18 ≤ y) →
  x + 4*x + 18 = 38 :=
by
  sorry

#check triangle_max_perimeter

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2753_275391


namespace NUMINAMATH_CALUDE_stationery_sales_l2753_275329

theorem stationery_sales (total_sales : ℕ) (fabric_fraction : ℚ) (jewelry_fraction : ℚ)
  (h_total : total_sales = 36)
  (h_fabric : fabric_fraction = 1/3)
  (h_jewelry : jewelry_fraction = 1/4)
  (h_stationery : fabric_fraction + jewelry_fraction < 1) :
  total_sales - (total_sales * fabric_fraction).floor - (total_sales * jewelry_fraction).floor = 15 :=
by sorry

end NUMINAMATH_CALUDE_stationery_sales_l2753_275329


namespace NUMINAMATH_CALUDE_total_candies_l2753_275325

theorem total_candies (linda_candies chloe_candies : ℕ) 
  (h1 : linda_candies = 34) 
  (h2 : chloe_candies = 28) : 
  linda_candies + chloe_candies = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l2753_275325


namespace NUMINAMATH_CALUDE_tomato_weight_l2753_275378

/-- Calculates the weight of a tomato based on grocery shopping information. -/
theorem tomato_weight (meat_price meat_weight buns_price lettuce_price pickle_price pickle_discount tomato_price_per_pound paid change : ℝ) :
  meat_price = 3.5 →
  meat_weight = 2 →
  buns_price = 1.5 →
  lettuce_price = 1 →
  pickle_price = 2.5 →
  pickle_discount = 1 →
  tomato_price_per_pound = 2 →
  paid = 20 →
  change = 6 →
  (paid - change - (meat_price * meat_weight + buns_price + lettuce_price + (pickle_price - pickle_discount))) / tomato_price_per_pound = 1.5 := by
sorry

end NUMINAMATH_CALUDE_tomato_weight_l2753_275378


namespace NUMINAMATH_CALUDE_reciprocal_roots_condition_l2753_275392

/-- The quadratic equation 5x^2 + 7x + k = 0 has reciprocal roots if and only if k = 5 -/
theorem reciprocal_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 5 * x^2 + 7 * x + k = 0 ∧ 5 * y^2 + 7 * y + k = 0 ∧ x * y = 1) ↔ 
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_condition_l2753_275392


namespace NUMINAMATH_CALUDE_two_white_balls_possible_l2753_275311

/-- Represents the contents of the box -/
structure BoxContents where
  black : ℕ
  white : ℕ

/-- Represents a single replacement rule -/
inductive ReplacementRule
  | ThreeBlack
  | TwoBlackOneWhite
  | OneBlackTwoWhite
  | ThreeWhite

/-- Applies a single replacement rule to the box contents -/
def applyRule (contents : BoxContents) (rule : ReplacementRule) : BoxContents :=
  match rule with
  | ReplacementRule.ThreeBlack => 
      ⟨contents.black - 2, contents.white⟩
  | ReplacementRule.TwoBlackOneWhite => 
      ⟨contents.black - 1, contents.white⟩
  | ReplacementRule.OneBlackTwoWhite => 
      ⟨contents.black - 1, contents.white⟩
  | ReplacementRule.ThreeWhite => 
      ⟨contents.black + 1, contents.white - 2⟩

/-- Applies a sequence of replacement rules to the box contents -/
def applyRules (initial : BoxContents) (rules : List ReplacementRule) : BoxContents :=
  rules.foldl applyRule initial

theorem two_white_balls_possible : 
  ∃ (rules : List ReplacementRule), 
    (applyRules ⟨100, 100⟩ rules).white = 2 :=
  sorry


end NUMINAMATH_CALUDE_two_white_balls_possible_l2753_275311


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2753_275320

theorem opposite_of_negative_two :
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2753_275320


namespace NUMINAMATH_CALUDE_apples_per_box_is_correct_l2753_275354

/-- The number of apples packed in a box -/
def apples_per_box : ℕ := 40

/-- The number of boxes packed per day in the first week -/
def boxes_per_day : ℕ := 50

/-- The number of fewer apples packed per day in the second week -/
def fewer_apples_per_day : ℕ := 500

/-- The total number of apples packed in two weeks -/
def total_apples : ℕ := 24500

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem apples_per_box_is_correct :
  (boxes_per_day * days_per_week * apples_per_box) +
  ((boxes_per_day * apples_per_box - fewer_apples_per_day) * days_per_week) = total_apples :=
by sorry

end NUMINAMATH_CALUDE_apples_per_box_is_correct_l2753_275354


namespace NUMINAMATH_CALUDE_total_volume_of_prisms_l2753_275304

theorem total_volume_of_prisms (length width height : ℝ) (num_prisms : ℕ) 
  (h1 : length = 5)
  (h2 : width = 3)
  (h3 : height = 6)
  (h4 : num_prisms = 4) :
  length * width * height * num_prisms = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_prisms_l2753_275304


namespace NUMINAMATH_CALUDE_triangle_side_c_equals_two_l2753_275370

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_side_c_equals_two (ABC : Triangle) 
  (h1 : ABC.B = 2 * ABC.A)  -- B = 2A
  (h2 : ABC.a = 1)          -- a = 1
  (h3 : ABC.b = Real.sqrt 3)  -- b = √3
  : ABC.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_c_equals_two_l2753_275370


namespace NUMINAMATH_CALUDE_book_cost_proof_l2753_275337

theorem book_cost_proof (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : num_books = 9)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / num_books = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_book_cost_proof_l2753_275337


namespace NUMINAMATH_CALUDE_max_pages_for_15_dollars_l2753_275349

/-- The cost in cents to copy 4 pages -/
def cost_per_4_pages : ℕ := 7

/-- The number of pages that can be copied for 4 cents -/
def pages_per_4_cents : ℕ := 4

/-- The amount in dollars available for copying -/
def available_dollars : ℕ := 15

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculates the maximum number of whole pages that can be copied -/
def max_pages : ℕ := 
  (dollars_to_cents available_dollars * pages_per_4_cents) / cost_per_4_pages

theorem max_pages_for_15_dollars : max_pages = 857 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_15_dollars_l2753_275349


namespace NUMINAMATH_CALUDE_pigeonhole_birthday_birthday_problem_l2753_275344

theorem pigeonhole_birthday (n : ℕ) (m : ℕ) (h : n > m) :
  ∀ f : Fin n → Fin m, ∃ i j : Fin n, i ≠ j ∧ f i = f j := by
  sorry

theorem birthday_problem :
  ∀ f : Fin 367 → Fin 366, ∃ i j : Fin 367, i ≠ j ∧ f i = f j := by
  exact pigeonhole_birthday 367 366 (by norm_num)

end NUMINAMATH_CALUDE_pigeonhole_birthday_birthday_problem_l2753_275344


namespace NUMINAMATH_CALUDE_linear_function_k_value_l2753_275317

/-- Given a linear function y = kx + 2 passing through the point (-2, -1), prove that k = 3/2 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2) →  -- Linear function condition
  (-1 : ℝ) = k * (-2 : ℝ) + 2 →  -- Point (-2, -1) condition
  k = 3/2 := by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l2753_275317


namespace NUMINAMATH_CALUDE_power_calculation_l2753_275375

theorem power_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2753_275375


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l2753_275356

/-- A two-digit natural number -/
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- The first digit of a two-digit number -/
def firstDigit (n : ℕ) : ℕ := n / 10

/-- The second digit of a two-digit number -/
def secondDigit (n : ℕ) : ℕ := n % 10

/-- The condition given in the problem -/
def satisfiesCondition (n : ℕ) : Prop :=
  4 * (firstDigit n) + 2 * (secondDigit n) = n / 2

theorem two_digit_number_theorem (n : ℕ) :
  TwoDigitNumber n ∧ satisfiesCondition n → n = 32 ∨ n = 64 ∨ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l2753_275356


namespace NUMINAMATH_CALUDE_tan_negative_five_pi_thirds_equals_sqrt_three_l2753_275388

theorem tan_negative_five_pi_thirds_equals_sqrt_three :
  Real.tan (-5 * π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_five_pi_thirds_equals_sqrt_three_l2753_275388


namespace NUMINAMATH_CALUDE_female_workers_count_l2753_275364

/-- Represents the number of workers of each type and their wages --/
structure WorkforceData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the total daily wage for all workers --/
def total_daily_wage (data : WorkforceData) : ℕ :=
  data.male_workers * data.male_wage +
  data.female_workers * data.female_wage +
  data.child_workers * data.child_wage

/-- Calculates the total number of workers --/
def total_workers (data : WorkforceData) : ℕ :=
  data.male_workers + data.female_workers + data.child_workers

/-- Theorem stating that the number of female workers is 15 --/
theorem female_workers_count (data : WorkforceData)
  (h1 : data.male_workers = 20)
  (h2 : data.child_workers = 5)
  (h3 : data.male_wage = 25)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 21)
  (h7 : (total_daily_wage data) / (total_workers data) = data.average_wage) :
  data.female_workers = 15 :=
sorry

end NUMINAMATH_CALUDE_female_workers_count_l2753_275364


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l2753_275342

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (running_pace : ℕ) (resting_time : ℕ) : ℕ :=
  let running_time := race_distance * running_pace
  let total_time := running_time + resting_time
  total_time * heart_rate

/-- Theorem: The athlete's heart beats 29250 times during the race --/
theorem athlete_heartbeats : 
  total_heartbeats 150 30 6 15 = 29250 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l2753_275342


namespace NUMINAMATH_CALUDE_fish_pond_problem_l2753_275359

/-- Represents the number of fish in a pond. -/
def N : ℕ := sorry

/-- The number of fish initially tagged and released. -/
def tagged_fish : ℕ := 40

/-- The number of fish caught in the second catch. -/
def second_catch : ℕ := 40

/-- The number of tagged fish found in the second catch. -/
def tagged_in_second_catch : ℕ := 2

/-- The fraction of tagged fish in the second catch. -/
def fraction_tagged_in_catch : ℚ := tagged_in_second_catch / second_catch

/-- The fraction of tagged fish in the pond. -/
def fraction_tagged_in_pond : ℚ := tagged_fish / N

theorem fish_pond_problem :
  fraction_tagged_in_catch = fraction_tagged_in_pond →
  N = 800 :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_problem_l2753_275359


namespace NUMINAMATH_CALUDE_expression_value_l2753_275322

theorem expression_value (a : ℝ) (h : a^2 + 2*a - 1 = 0) : 
  ((a^2 - 1)/(a^2 - 2*a + 1) - 1/(1-a)) / (1/(a^2 - a)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2753_275322


namespace NUMINAMATH_CALUDE_fifth_day_is_tuesday_l2753_275377

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Returns the day of the week for a given number of days after a reference day -/
def dayAfter (startDay : DayOfWeek) (daysAfter : Int) : DayOfWeek :=
  sorry

theorem fifth_day_is_tuesday
  (month : List DayInMonth)
  (h : ∃ d ∈ month, d.day = 20 ∧ d.dayOfWeek = DayOfWeek.Wednesday) :
  ∃ d ∈ month, d.day = 5 ∧ d.dayOfWeek = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_fifth_day_is_tuesday_l2753_275377


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2753_275380

/-- The necessary and sufficient condition for a circle and a line to have common points -/
theorem circle_line_intersection (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y = k*x - 3) ↔ -Real.sqrt 8 ≤ k ∧ k ≤ Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2753_275380


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2753_275307

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2753_275307


namespace NUMINAMATH_CALUDE_worker_a_alone_time_l2753_275396

/-- Represents the efficiency of a worker -/
structure WorkerEfficiency where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a job to be completed -/
structure Job where
  total_work : ℝ
  total_work_pos : total_work > 0

theorem worker_a_alone_time 
  (job : Job) 
  (a b : WorkerEfficiency) 
  (h1 : a.rate = 2 * b.rate) 
  (h2 : job.total_work / (a.rate + b.rate) = 20) : 
  job.total_work / a.rate = 30 := by
  sorry

end NUMINAMATH_CALUDE_worker_a_alone_time_l2753_275396


namespace NUMINAMATH_CALUDE_sum_of_cyclic_relations_l2753_275305

theorem sum_of_cyclic_relations (p q r : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  q = p * (4 - p) →
  r = q * (4 - q) →
  p = r * (4 - r) →
  p + q + r = 6 ∨ p + q + r = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_relations_l2753_275305


namespace NUMINAMATH_CALUDE_simplify_expression_l2753_275371

theorem simplify_expression (x y : ℝ) : 4 * x^2 + 3 * y^2 - 2 * x^2 - 4 * y^2 = 2 * x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2753_275371


namespace NUMINAMATH_CALUDE_min_value_of_f_l2753_275302

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 ∧ ∀ x ∈ Set.Icc 0 3, f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2753_275302


namespace NUMINAMATH_CALUDE_perfect_power_sequence_exists_l2753_275330

theorem perfect_power_sequence_exists : ∃ a : ℕ+, ∀ k ∈ Set.Icc 2015 2558, 
  ∃ (b : ℕ+) (n : ℕ), n ≥ 2 ∧ (k : ℝ) * a.val = b.val ^ n :=
sorry

end NUMINAMATH_CALUDE_perfect_power_sequence_exists_l2753_275330


namespace NUMINAMATH_CALUDE_pedestrians_meeting_l2753_275343

/-- The problem of two pedestrians meeting --/
theorem pedestrians_meeting 
  (distance : ℝ) 
  (initial_meeting_time : ℝ) 
  (adjusted_meeting_time : ℝ) 
  (speed_multiplier_1 : ℝ) 
  (speed_multiplier_2 : ℝ) 
  (h1 : distance = 105) 
  (h2 : initial_meeting_time = 7.5) 
  (h3 : adjusted_meeting_time = 8 + 1/13) 
  (h4 : speed_multiplier_1 = 1.5) 
  (h5 : speed_multiplier_2 = 0.5) :
  ∃ (speed1 speed2 : ℝ), 
    speed1 = 8 ∧ 
    speed2 = 6 ∧ 
    initial_meeting_time * (speed1 + speed2) = distance ∧ 
    adjusted_meeting_time * (speed_multiplier_1 * speed1 + speed_multiplier_2 * speed2) = distance :=
by sorry


end NUMINAMATH_CALUDE_pedestrians_meeting_l2753_275343


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2753_275399

/-- Represents the number of students in Mrs. Smith's chemistry class -/
def total_students : ℕ := 42

/-- Represents the ratio of boys in the class -/
def boys_ratio : ℕ := 3

/-- Represents the ratio of girls in the class -/
def girls_ratio : ℕ := 4

/-- Calculates the number of boys in the class -/
def num_boys : ℕ := (total_students * boys_ratio) / (boys_ratio + girls_ratio)

/-- Calculates the number of girls in the class -/
def num_girls : ℕ := (total_students * girls_ratio) / (boys_ratio + girls_ratio)

/-- Proves that there are 6 more girls than boys in the class -/
theorem more_girls_than_boys : num_girls - num_boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2753_275399


namespace NUMINAMATH_CALUDE_fraction_non_negative_iff_positive_denominator_l2753_275352

theorem fraction_non_negative_iff_positive_denominator :
  ∀ x : ℝ, (2 / x ≥ 0) ↔ (x > 0) := by sorry

end NUMINAMATH_CALUDE_fraction_non_negative_iff_positive_denominator_l2753_275352


namespace NUMINAMATH_CALUDE_shaded_area_is_half_l2753_275347

/-- Represents a rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- Represents the shaded region after transformation -/
structure ShadedRegion where
  rectangle : Rectangle
  -- The rectangle is cut in two by a vertical cut joining the midpoints of its longer edges
  is_cut_in_half : Bool
  -- The right-hand half is given a quarter turn (90 degrees) about its center
  is_quarter_turned : Bool

/-- The area of the shaded region is half the area of the original rectangle -/
theorem shaded_area_is_half (r : Rectangle) (s : ShadedRegion) 
  (h1 : s.rectangle = r)
  (h2 : s.is_cut_in_half = true)
  (h3 : s.is_quarter_turned = true) :
  (s.rectangle.area / 2 : ℝ) = r.area / 2 :=
by sorry

#check shaded_area_is_half

end NUMINAMATH_CALUDE_shaded_area_is_half_l2753_275347


namespace NUMINAMATH_CALUDE_min_area_triangle_abc_l2753_275367

/-- The minimum area of a triangle ABC where A = (0, 0), B = (30, 16), and C has integer coordinates --/
theorem min_area_triangle_abc : 
  ∀ (p q : ℤ), 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 16)
  let C : ℝ × ℝ := (p, q)
  let area := (1/2 : ℝ) * |16 * p - 30 * q|
  1 ≤ area ∧ (∃ (p' q' : ℤ), (1/2 : ℝ) * |16 * p' - 30 * q'| = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_abc_l2753_275367


namespace NUMINAMATH_CALUDE_original_price_calculation_l2753_275376

theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600) 
  (h2 : profit_percentage = 20) : 
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 + profit_percentage / 100) ∧ 
    original_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2753_275376


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2753_275389

theorem binomial_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  p^2 ∣ (Nat.choose (2*p - 1) (p - 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2753_275389


namespace NUMINAMATH_CALUDE_hoseok_social_studies_score_l2753_275319

/-- Represents Hoseok's test scores -/
structure HoseokScores where
  average_three : ℝ  -- Average score of Korean, English, and Science
  average_four : ℝ   -- Average score after including Social studies
  social_studies : ℝ -- Score of Social studies test

/-- Theorem stating that given Hoseok's average scores, his Social studies score must be 93 -/
theorem hoseok_social_studies_score (scores : HoseokScores)
  (h1 : scores.average_three = 89)
  (h2 : scores.average_four = 90) :
  scores.social_studies = 93 := by
  sorry

#check hoseok_social_studies_score

end NUMINAMATH_CALUDE_hoseok_social_studies_score_l2753_275319


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l2753_275350

theorem power_zero_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l2753_275350


namespace NUMINAMATH_CALUDE_sum_triangles_eq_sixteen_l2753_275301

/-- The triangle operation -/
def triangle (a b c : ℕ) : ℕ := a * b - c

/-- The sum of two triangle operations -/
def sum_triangles (a1 b1 c1 a2 b2 c2 : ℕ) : ℕ :=
  triangle a1 b1 c1 + triangle a2 b2 c2

/-- Theorem: The sum of the triangle operations for the given sets of numbers equals 16 -/
theorem sum_triangles_eq_sixteen :
  sum_triangles 2 4 3 3 6 7 = 16 := by sorry

end NUMINAMATH_CALUDE_sum_triangles_eq_sixteen_l2753_275301


namespace NUMINAMATH_CALUDE_solve_ttakji_problem_l2753_275393

def ttakji_problem (initial_large : ℕ) (initial_small : ℕ) (final_total : ℕ) : Prop :=
  ∃ (lost_large : ℕ),
    initial_large ≥ lost_large ∧
    initial_small ≥ 3 * lost_large ∧
    initial_large + initial_small - lost_large - 3 * lost_large = final_total ∧
    lost_large = 4

theorem solve_ttakji_problem :
  ttakji_problem 12 34 30 := by sorry

end NUMINAMATH_CALUDE_solve_ttakji_problem_l2753_275393


namespace NUMINAMATH_CALUDE_digit_permutation_theorem_l2753_275318

/-- A k-digit number -/
def kDigitNumber (k : ℕ) := { n : ℕ // n < 10^k ∧ n ≥ 10^(k-1) }

/-- Inserting a k-digit number between two adjacent digits of another number -/
def insertNumber (n : ℕ) (k : ℕ) (a : kDigitNumber k) : ℕ := sorry

/-- Permutation of digits -/
def isPermutationOf (a b : ℕ) : Prop := sorry

theorem digit_permutation_theorem (k : ℕ) (p : ℕ) (A B : kDigitNumber k) :
  Prime p →
  p > 10^k →
  (∀ m : ℕ, m % p = 0 → (insertNumber m k A) % p = 0) →
  (∃ n : ℕ, (insertNumber n k A) % p = 0 ∧ (insertNumber (insertNumber n k A) k B) % p = 0) →
  isPermutationOf A.val B.val := by sorry

end NUMINAMATH_CALUDE_digit_permutation_theorem_l2753_275318


namespace NUMINAMATH_CALUDE_liangliang_speed_l2753_275358

/-- The walking speeds of Mingming and Liangliang -/
structure WalkingSpeeds where
  mingming : ℝ
  liangliang : ℝ

/-- The initial and final distances between Mingming and Liangliang -/
structure Distances where
  initial : ℝ
  final : ℝ

/-- The time elapsed between the initial and final measurements -/
def elapsedTime : ℝ := 20

/-- The theorem stating the possible walking speeds of Liangliang -/
theorem liangliang_speed 
  (speeds : WalkingSpeeds) 
  (distances : Distances) 
  (h1 : speeds.mingming = 80) 
  (h2 : distances.initial = 3000) 
  (h3 : distances.final = 2900) :
  speeds.liangliang = 85 ∨ speeds.liangliang = 75 :=
sorry

end NUMINAMATH_CALUDE_liangliang_speed_l2753_275358


namespace NUMINAMATH_CALUDE_yellow_surface_fraction_proof_l2753_275334

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  yellow_cubes : ℕ
  blue_cubes : ℕ

/-- Calculates the fraction of yellow surface area -/
def yellow_surface_fraction (cube : LargeCube) : ℚ :=
  sorry

theorem yellow_surface_fraction_proof (cube : LargeCube) :
  cube.edge_length = 4 →
  cube.small_cubes = 64 →
  cube.yellow_cubes = 15 →
  cube.blue_cubes = 49 →
  yellow_surface_fraction cube = 1/6 :=
sorry

end NUMINAMATH_CALUDE_yellow_surface_fraction_proof_l2753_275334


namespace NUMINAMATH_CALUDE_student_score_l2753_275348

theorem student_score (num_questions num_correct_answers points_per_question : ℕ) 
  (h1 : num_questions = 5)
  (h2 : num_correct_answers = 3)
  (h3 : points_per_question = 2) :
  num_correct_answers * points_per_question = 6 := by sorry

end NUMINAMATH_CALUDE_student_score_l2753_275348


namespace NUMINAMATH_CALUDE_probability_greater_than_three_l2753_275351

-- Define a standard die
def StandardDie : ℕ := 6

-- Define the favorable outcomes (numbers greater than 3)
def FavorableOutcomes : ℕ := 3

-- Theorem statement
theorem probability_greater_than_three (d : ℕ) (h : d = StandardDie) : 
  (FavorableOutcomes : ℚ) / d = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_greater_than_three_l2753_275351


namespace NUMINAMATH_CALUDE_proportion_inconsistency_l2753_275309

theorem proportion_inconsistency : ¬ ∃ (x : ℚ), (x / 2 = 2 / 6) ∧ (x = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_proportion_inconsistency_l2753_275309


namespace NUMINAMATH_CALUDE_max_value_and_inequality_l2753_275327

noncomputable def f (x : ℝ) := Real.log (x + 1)

noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_and_inequality :
  (∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x) ∧
  (g (3 : ℝ) = 2 * Real.log 2 - 7 / 4) ∧
  (∀ (x : ℝ), x > 0 → f x < (Real.exp x - 1) / (x^2)) := by sorry

end NUMINAMATH_CALUDE_max_value_and_inequality_l2753_275327


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l2753_275310

theorem scientific_notation_conversion :
  (4.6 : ℝ) * (10 ^ 8) = 460000000 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l2753_275310


namespace NUMINAMATH_CALUDE_x_minus_y_equals_one_l2753_275394

-- Define x and y based on the given conditions
def x : Int := 2 - 4 + 6
def y : Int := 1 - 3 + 5

-- State the theorem to be proved
theorem x_minus_y_equals_one : x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_one_l2753_275394


namespace NUMINAMATH_CALUDE_decimal_to_scientific_notation_l2753_275387

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem decimal_to_scientific_notation :
  scientific_notation 0.00000011 = (1.1, -7) :=
sorry

end NUMINAMATH_CALUDE_decimal_to_scientific_notation_l2753_275387


namespace NUMINAMATH_CALUDE_tile_arrangements_l2753_275335

/-- The number of distinguishable arrangements of tiles of different colors -/
def distinguishable_arrangements (blue red green : ℕ) : ℕ :=
  Nat.factorial (blue + red + green) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green)

/-- Theorem stating that the number of distinguishable arrangements
    of 3 blue tiles, 2 red tiles, and 4 green tiles is 1260 -/
theorem tile_arrangements :
  distinguishable_arrangements 3 2 4 = 1260 := by
  sorry

#eval distinguishable_arrangements 3 2 4

end NUMINAMATH_CALUDE_tile_arrangements_l2753_275335


namespace NUMINAMATH_CALUDE_intersection_count_l2753_275385

/-- The number of distinct intersection points between two algebraic curves -/
def num_intersections (f g : ℝ → ℝ → ℝ) : ℕ :=
  sorry

/-- First curve equation -/
def curve1 (x y : ℝ) : ℝ :=
  (x - y + 3) * (3 * x + y - 7)

/-- Second curve equation -/
def curve2 (x y : ℝ) : ℝ :=
  (x + y - 3) * (2 * x - 5 * y + 12)

theorem intersection_count :
  num_intersections curve1 curve2 = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_count_l2753_275385


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2753_275331

theorem smallest_integer_with_given_remainders :
  ∀ x : ℕ,
  (x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7) →
  (∀ y : ℕ, y > 0 ∧ y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y) →
  x = 167 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2753_275331


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_problem_solution_l2753_275341

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)
def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_sum (a₁ : ℤ) :
  ∃ d : ℤ, 
    (sum_arithmetic_sequence a₁ d 6 - 2 * sum_arithmetic_sequence a₁ d 3 = 18) → 
    (sum_arithmetic_sequence a₁ d 2017 = 2017) := by
  sorry

-- Main theorem
theorem problem_solution : 
  ∃ d : ℤ, 
    (sum_arithmetic_sequence (-2015) d 6 - 2 * sum_arithmetic_sequence (-2015) d 3 = 18) → 
    (sum_arithmetic_sequence (-2015) d 2017 = 2017) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_problem_solution_l2753_275341


namespace NUMINAMATH_CALUDE_food_bank_donations_boudin_del_monte_multiple_of_seven_l2753_275362

/-- Represents the total number of food items donated by five companies to a local food bank. -/
def total_donations (foster_farms : ℕ) : ℕ :=
  let american_summits := 2 * foster_farms
  let hormel := 3 * foster_farms
  let boudin_butchers := hormel / 3
  let del_monte := american_summits - 30
  foster_farms + american_summits + hormel + boudin_butchers + del_monte

/-- Theorem stating the total number of food items donated by the five companies. -/
theorem food_bank_donations : 
  total_donations 45 = 375 ∧ 
  (total_donations 45 - (45 + (2 * 45 - 30))) % 7 = 0 := by
  sorry

/-- Verification that the combined donations from Boudin Butchers and Del Monte Foods is a multiple of 7. -/
theorem boudin_del_monte_multiple_of_seven (foster_farms : ℕ) : 
  ((3 * foster_farms) / 3 + (2 * foster_farms - 30)) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_donations_boudin_del_monte_multiple_of_seven_l2753_275362


namespace NUMINAMATH_CALUDE_divided_triangle_area_l2753_275323

/-- Represents a triangle divided into six smaller triangles -/
structure DividedTriangle where
  /-- Areas of four known smaller triangles -/
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- The theorem stating that if a triangle is divided as described, with the given areas, its total area is 380 -/
theorem divided_triangle_area (t : DividedTriangle) 
  (h1 : t.area1 = 84) 
  (h2 : t.area2 = 70) 
  (h3 : t.area3 = 35) 
  (h4 : t.area4 = 65) : 
  ∃ (area5 area6 : ℝ), t.area1 + t.area2 + t.area3 + t.area4 + area5 + area6 = 380 := by
  sorry

end NUMINAMATH_CALUDE_divided_triangle_area_l2753_275323


namespace NUMINAMATH_CALUDE_calculation_proof_l2753_275382

theorem calculation_proof : (2.5 - 0.3) * 0.25 = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2753_275382


namespace NUMINAMATH_CALUDE_factorization_problems_l2753_275357

variable (m a b : ℝ)

theorem factorization_problems :
  (ma^2 - mb^2 = m*(a+b)*(a-b)) ∧
  ((a+b) - 2*a*(a+b) + a^2*(a+b) = (a+b)*(a-1)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l2753_275357


namespace NUMINAMATH_CALUDE_sum_in_D_l2753_275369

-- Define the sets A, B, C, and D
def A : Set Int := {x | ∃ k : Int, x = 4 * k}
def B : Set Int := {x | ∃ m : Int, x = 4 * m + 1}
def C : Set Int := {x | ∃ n : Int, x = 4 * n + 2}
def D : Set Int := {x | ∃ t : Int, x = 4 * t + 3}

-- State the theorem
theorem sum_in_D (a b : Int) (ha : a ∈ B) (hb : b ∈ C) : a + b ∈ D := by
  sorry

end NUMINAMATH_CALUDE_sum_in_D_l2753_275369


namespace NUMINAMATH_CALUDE_crackers_per_friend_l2753_275366

theorem crackers_per_friend (initial_crackers : ℕ) (friends : ℕ) (remaining_crackers : ℕ) :
  initial_crackers = 23 →
  friends = 2 →
  remaining_crackers = 11 →
  (initial_crackers - remaining_crackers) / friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l2753_275366


namespace NUMINAMATH_CALUDE_martha_juice_bottles_l2753_275338

theorem martha_juice_bottles (initial_pantry : ℕ) (bought : ℕ) (consumed : ℕ) (final_total : ℕ) 
  (h1 : initial_pantry = 4)
  (h2 : bought = 5)
  (h3 : consumed = 3)
  (h4 : final_total = 10) :
  ∃ (initial_fridge : ℕ), 
    initial_fridge + initial_pantry + bought - consumed = final_total ∧ 
    initial_fridge = 4 := by
  sorry

end NUMINAMATH_CALUDE_martha_juice_bottles_l2753_275338


namespace NUMINAMATH_CALUDE_square_eq_nine_solutions_l2753_275326

theorem square_eq_nine_solutions (x : ℝ) : x^2 = 9 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_square_eq_nine_solutions_l2753_275326


namespace NUMINAMATH_CALUDE_tan_value_for_given_sum_l2753_275300

theorem tan_value_for_given_sum (x : ℝ) 
  (h1 : Real.sin x + Real.cos x = 1/5)
  (h2 : 0 ≤ x ∧ x < π) : 
  Real.tan x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_sum_l2753_275300


namespace NUMINAMATH_CALUDE_metal_collection_contest_solution_l2753_275339

/-- Represents the metal collection contest between boys and girls -/
structure MetalContest where
  totalMetal : ℕ
  boyAverage : ℕ
  girlAverage : ℕ
  numBoys : ℕ
  numGirls : ℕ

/-- Checks if the given numbers satisfy the contest conditions -/
def isValidContest (contest : MetalContest) : Prop :=
  contest.boyAverage * contest.numBoys + contest.girlAverage * contest.numGirls = contest.totalMetal

/-- Checks if boys won the contest -/
def boysWon (contest : MetalContest) : Prop :=
  contest.boyAverage * contest.numBoys > contest.girlAverage * contest.numGirls

/-- Theorem stating the solution to the metal collection contest -/
theorem metal_collection_contest_solution :
  ∃ (contest : MetalContest),
    contest.totalMetal = 2831 ∧
    contest.boyAverage = 95 ∧
    contest.girlAverage = 74 ∧
    contest.numBoys = 15 ∧
    contest.numGirls = 19 ∧
    isValidContest contest ∧
    boysWon contest :=
  sorry

end NUMINAMATH_CALUDE_metal_collection_contest_solution_l2753_275339


namespace NUMINAMATH_CALUDE_father_twice_as_old_father_four_times_now_l2753_275384

/-- Represents the current age of the father -/
def father_age : ℕ := 40

/-- Represents the current age of the daughter -/
def daughter_age : ℕ := 10

/-- Represents the number of years until the father is twice as old as the daughter -/
def years_until_twice : ℕ := 20

/-- Theorem stating that after the specified number of years, the father will be twice as old as the daughter -/
theorem father_twice_as_old :
  father_age + years_until_twice = 2 * (daughter_age + years_until_twice) :=
sorry

/-- Theorem stating that the father is currently 4 times as old as the daughter -/
theorem father_four_times_now :
  father_age = 4 * daughter_age :=
sorry

end NUMINAMATH_CALUDE_father_twice_as_old_father_four_times_now_l2753_275384


namespace NUMINAMATH_CALUDE_log_product_equals_four_l2753_275328

theorem log_product_equals_four : Real.log 9 / Real.log 2 * (Real.log 4 / Real.log 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_l2753_275328


namespace NUMINAMATH_CALUDE_problem_solution_l2753_275346

theorem problem_solution : 
  ∀ M : ℚ, (5 + 7 + 9) / 3 = (2005 + 2007 + 2009) / M → M = 860 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2753_275346


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l2753_275383

theorem complex_ratio_theorem (x y : ℂ) 
  (h1 : (x^2 + y^2) / (x + y) = 4)
  (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
  (x^6 + y^6) / (x^5 + y^5) = 10 + 2 * Real.sqrt 17 ∨
  (x^6 + y^6) / (x^5 + y^5) = 10 - 2 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l2753_275383


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l2753_275361

open Real

theorem inequality_implies_upper_bound (a : ℝ) : 
  (∀ x > 0, 2 * x * log x ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l2753_275361


namespace NUMINAMATH_CALUDE_larger_number_from_sum_and_difference_l2753_275374

theorem larger_number_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (diff_eq : x - y = 6) :
  max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_from_sum_and_difference_l2753_275374


namespace NUMINAMATH_CALUDE_trapezoid_determines_unique_plane_l2753_275353

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A trapezoid in 3D space -/
structure Trapezoid where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D
  is_trapezoid : ∃ (a b : ℝ), a ≠ b ∧
    (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x) ∧
    (p3.x - p2.x) * (p1.y - p4.y) = (p3.y - p2.y) * (p1.x - p4.x)

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Definition of a point lying on a plane -/
def Point3D.on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Theorem: A trapezoid determines a unique plane -/
theorem trapezoid_determines_unique_plane (t : Trapezoid) :
  ∃! (plane : Plane), t.p1.on_plane plane ∧ t.p2.on_plane plane ∧
                      t.p3.on_plane plane ∧ t.p4.on_plane plane :=
sorry

end NUMINAMATH_CALUDE_trapezoid_determines_unique_plane_l2753_275353


namespace NUMINAMATH_CALUDE_pants_cut_amount_l2753_275312

def skirt_cut : ℝ := 0.75
def difference : ℝ := 0.25

theorem pants_cut_amount : ∃ (x : ℝ), x = skirt_cut - difference := by sorry

end NUMINAMATH_CALUDE_pants_cut_amount_l2753_275312


namespace NUMINAMATH_CALUDE_james_marbles_l2753_275373

theorem james_marbles (total_marbles : ℕ) (num_bags : ℕ) (marbles_per_bag : ℕ) :
  total_marbles = 28 →
  num_bags = 4 →
  marbles_per_bag * num_bags = total_marbles →
  total_marbles - marbles_per_bag = 21 :=
by sorry

end NUMINAMATH_CALUDE_james_marbles_l2753_275373


namespace NUMINAMATH_CALUDE_eel_fat_l2753_275324

/-- The amount of fat in ounces for each type of fish --/
structure FishFat where
  herring : ℝ
  eel : ℝ
  pike : ℝ

/-- The number of each type of fish cooked --/
def fish_count : ℝ := 40

/-- The total amount of fat served in ounces --/
def total_fat : ℝ := 3600

/-- Theorem stating the amount of fat in an eel --/
theorem eel_fat (f : FishFat) 
  (herring_fat : f.herring = 40)
  (pike_fat : f.pike = f.eel + 10)
  (total_fat_eq : fish_count * (f.herring + f.eel + f.pike) = total_fat) :
  f.eel = 20 := by
  sorry

end NUMINAMATH_CALUDE_eel_fat_l2753_275324


namespace NUMINAMATH_CALUDE_jackson_money_l2753_275306

theorem jackson_money (williams_money : ℝ) (h1 : williams_money > 0) 
  (h2 : williams_money + 5 * williams_money = 150) : 
  5 * williams_money = 125 := by
sorry

end NUMINAMATH_CALUDE_jackson_money_l2753_275306


namespace NUMINAMATH_CALUDE_prime_divisor_form_l2753_275355

theorem prime_divisor_form (a p : ℕ) (ha : a > 0) (hp : Nat.Prime p) 
  (hdiv : p ∣ a^3 - 3*a + 1) (hp_neq_3 : p ≠ 3) :
  ∃ k : ℤ, p = 9*k + 1 ∨ p = 9*k - 1 := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_form_l2753_275355


namespace NUMINAMATH_CALUDE_sequence_non_positive_l2753_275308

theorem sequence_non_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h_ineq : ∀ k : ℕ, k < n → a k.pred - 2 * a k + a k.succ ≥ 0) :
  ∀ k : ℕ, k ≤ n → a k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l2753_275308


namespace NUMINAMATH_CALUDE_eight_power_32_sum_equals_2_power_99_l2753_275313

theorem eight_power_32_sum_equals_2_power_99 :
  (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + 
  (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 = (2:ℕ)^99 :=
by sorry

end NUMINAMATH_CALUDE_eight_power_32_sum_equals_2_power_99_l2753_275313


namespace NUMINAMATH_CALUDE_club_female_count_l2753_275363

theorem club_female_count (total : ℕ) (difference : ℕ) (female : ℕ) : 
  total = 82 →
  difference = 6 →
  female = total / 2 + difference / 2 →
  female = 44 := by
sorry

end NUMINAMATH_CALUDE_club_female_count_l2753_275363


namespace NUMINAMATH_CALUDE_evaluate_expression_l2753_275360

theorem evaluate_expression : (1500^2 : ℚ) / (306^2 - 294^2) = 312.5 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2753_275360


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2753_275368

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2753_275368


namespace NUMINAMATH_CALUDE_typing_time_is_35_minutes_l2753_275314

/-- Represents the typing scenario with given conditions -/
structure TypingScenario where
  barbaraMaxSpeed : ℕ
  barbaraInjuryReduction : ℕ
  barbaraFatigueReduction : ℕ
  barbaraFatigueInterval : ℕ
  jimSpeed : ℕ
  jimTime : ℕ
  monicaSpeed : ℕ
  monicaTime : ℕ
  breakDuration : ℕ
  breakInterval : ℕ
  documentLength : ℕ

/-- Calculates the minimum time required to type the document -/
def minTypingTime (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that the minimum typing time for the given scenario is 35 minutes -/
theorem typing_time_is_35_minutes (scenario : TypingScenario) 
  (h1 : scenario.barbaraMaxSpeed = 212)
  (h2 : scenario.barbaraInjuryReduction = 40)
  (h3 : scenario.barbaraFatigueReduction = 5)
  (h4 : scenario.barbaraFatigueInterval = 15)
  (h5 : scenario.jimSpeed = 100)
  (h6 : scenario.jimTime = 20)
  (h7 : scenario.monicaSpeed = 150)
  (h8 : scenario.monicaTime = 10)
  (h9 : scenario.breakDuration = 5)
  (h10 : scenario.breakInterval = 25)
  (h11 : scenario.documentLength = 3440) :
  minTypingTime scenario = 35 :=
by sorry

end NUMINAMATH_CALUDE_typing_time_is_35_minutes_l2753_275314


namespace NUMINAMATH_CALUDE_prob_black_second_draw_l2753_275386

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the state of the box -/
structure Box :=
  (red : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a black ball -/
def prob_black (b : Box) : ℚ :=
  b.black / (b.red + b.black)

/-- Adds balls to the box based on the color drawn -/
def add_balls (b : Box) (c : Color) : Box :=
  match c with
  | Color.Red => Box.mk (b.red + 3) b.black
  | Color.Black => Box.mk b.red (b.black + 3)

/-- The main theorem to prove -/
theorem prob_black_second_draw (initial_box : Box) 
  (h1 : initial_box.red = 4)
  (h2 : initial_box.black = 5) : 
  (prob_black initial_box * prob_black (add_balls initial_box Color.Black) +
   (1 - prob_black initial_box) * prob_black (add_balls initial_box Color.Red)) = 5/9 :=
by sorry

end NUMINAMATH_CALUDE_prob_black_second_draw_l2753_275386


namespace NUMINAMATH_CALUDE_negation_of_forall_abs_sum_nonnegative_l2753_275303

theorem negation_of_forall_abs_sum_nonnegative :
  (¬ (∀ x : ℝ, x + |x| ≥ 0)) ↔ (∃ x : ℝ, x + |x| < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_abs_sum_nonnegative_l2753_275303


namespace NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l2753_275381

-- Define the lines
def line1 (a x y : ℝ) : Prop := (3*a + 2)*x - 3*y + 8 = 0
def line2 (a x y : ℝ) : Prop := 3*x + (a + 4)*y - 7 = 0

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := 
  (3*a + 2) * 3 + (-3) * (a + 4) = 0

-- Theorem statement
theorem perpendicular_lines_imply_a_equals_one :
  ∀ a : ℝ, perpendicular a → a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l2753_275381


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l2753_275336

theorem ceiling_floor_product_range (y : ℝ) : 
  y < -1 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l2753_275336


namespace NUMINAMATH_CALUDE_average_rate_of_change_specific_average_rate_of_change_l2753_275398

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = ((b + a) + 2) :=
sorry

theorem specific_average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_average_rate_of_change_specific_average_rate_of_change_l2753_275398


namespace NUMINAMATH_CALUDE_sin_90_degrees_l2753_275372

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l2753_275372


namespace NUMINAMATH_CALUDE_compare_cubic_and_mixed_terms_l2753_275379

theorem compare_cubic_and_mixed_terms {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_compare_cubic_and_mixed_terms_l2753_275379


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_l2753_275315

/-- Given an augmented matrix representing a system of linear equations with a known solution,
    prove that the difference between certain elements of the augmented matrix is 16. -/
theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (∃ (x y : ℝ), x = 3 ∧ y = 5 ∧ 
   2 * x + 3 * y = c₁ ∧
   y = c₂) →
  c₁ - c₂ = 16 := by
sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_l2753_275315


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l2753_275332

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n - 3 = 0) :
  ∀ n, a (n + 1) > a n :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l2753_275332


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l2753_275340

theorem rationalize_and_simplify :
  (Real.sqrt 8 + Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) =
  Real.sqrt 10 - Real.sqrt 6 + (Real.sqrt 15) / 2 - 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l2753_275340
