import Mathlib

namespace NUMINAMATH_CALUDE_quarters_fraction_is_three_fifths_l4023_402333

/-- The total number of quarters Ella has -/
def total_quarters : ℕ := 30

/-- The number of quarters representing states that joined between 1790 and 1809 -/
def quarters_1790_1809 : ℕ := 18

/-- The fraction of quarters representing states that joined between 1790 and 1809 -/
def fraction_1790_1809 : ℚ := quarters_1790_1809 / total_quarters

theorem quarters_fraction_is_three_fifths : 
  fraction_1790_1809 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_quarters_fraction_is_three_fifths_l4023_402333


namespace NUMINAMATH_CALUDE_toms_gas_expense_l4023_402393

/-- Proves that given the conditions of Tom's lawn mowing business,
    his monthly gas expense is $17. -/
theorem toms_gas_expense (lawns_mowed : ℕ) (price_per_lawn : ℕ) (extra_income : ℕ) (profit : ℕ) 
    (h1 : lawns_mowed = 3)
    (h2 : price_per_lawn = 12)
    (h3 : extra_income = 10)
    (h4 : profit = 29) :
  lawns_mowed * price_per_lawn + extra_income - profit = 17 := by
  sorry

end NUMINAMATH_CALUDE_toms_gas_expense_l4023_402393


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l4023_402308

/-- Given a group of children with known happiness status and gender distribution,
    calculate the number of children who are neither happy nor sad. -/
theorem children_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 18)
  (h5 : girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : total_children = boys + girls)
  : total_children - (happy_children + sad_children) = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l4023_402308


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4023_402367

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (2 * x - 1 > 5) ∧ (-x < -6)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 6}

-- Theorem stating that the solution set is correct
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4023_402367


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_one_l4023_402383

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b (y : ℝ) : ℝ × ℝ := (1, -2*y)

/-- Definition of parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Theorem: If a and b(y) are parallel, then y = 1 -/
theorem parallel_vectors_imply_y_equals_one :
  parallel a (b y) → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_one_l4023_402383


namespace NUMINAMATH_CALUDE_jane_donuts_problem_l4023_402390

theorem jane_donuts_problem :
  ∀ (d c : ℕ),
  d + c = 6 →
  90 * d + 60 * c = 450 →
  d = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_donuts_problem_l4023_402390


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_seven_fortieths_l4023_402371

theorem ten_thousandths_place_of_seven_fortieths (n : ℕ) : 
  (7 : ℚ) / 40 * 10000 - ((7 : ℚ) / 40 * 10000).floor = (0 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_seven_fortieths_l4023_402371


namespace NUMINAMATH_CALUDE_or_propositions_true_l4023_402311

-- Define the properties of square and rectangle diagonals
def square_diagonals_perpendicular : Prop := True
def rectangle_diagonals_bisect : Prop := True

-- Theorem statement
theorem or_propositions_true : 
  ((2 = 2) ∨ (2 > 2)) ∧ 
  (square_diagonals_perpendicular ∨ rectangle_diagonals_bisect) := by
  sorry

end NUMINAMATH_CALUDE_or_propositions_true_l4023_402311


namespace NUMINAMATH_CALUDE_condition_necessity_sufficiency_l4023_402350

theorem condition_necessity_sufficiency : 
  (∀ x : ℝ, (x + 1) * (x^2 + 2) > 0 → (x + 1) * (x + 2) > 0) ∧ 
  (∃ x : ℝ, (x + 1) * (x + 2) > 0 ∧ (x + 1) * (x^2 + 2) ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessity_sufficiency_l4023_402350


namespace NUMINAMATH_CALUDE_third_factor_proof_l4023_402373

theorem third_factor_proof (w : ℕ) (h1 : w = 168) (h2 : 2^5 ∣ (936 * w)) (h3 : 3^3 ∣ (936 * w)) :
  (936 * w) / (2^5 * 3^3) = 182 := by
  sorry

end NUMINAMATH_CALUDE_third_factor_proof_l4023_402373


namespace NUMINAMATH_CALUDE_downstream_distance_l4023_402323

-- Define the given parameters
def boat_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- Define the theorem
theorem downstream_distance :
  boat_speed + stream_speed * travel_time = 81 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_l4023_402323


namespace NUMINAMATH_CALUDE_daily_average_is_40_l4023_402380

/-- Represents the daily average of borrowed books -/
def daily_average : ℝ := 40

/-- Represents the total number of books borrowed in a week -/
def total_weekly_books : ℕ := 216

/-- Represents the borrowing rate on Friday as a multiplier of the daily average -/
def friday_rate : ℝ := 1.4

/-- Theorem stating that given the conditions, the daily average of borrowed books is 40 -/
theorem daily_average_is_40 :
  daily_average * 4 + daily_average * friday_rate = total_weekly_books :=
by sorry

end NUMINAMATH_CALUDE_daily_average_is_40_l4023_402380


namespace NUMINAMATH_CALUDE_negative_six_div_three_l4023_402338

theorem negative_six_div_three : (-6) / 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_six_div_three_l4023_402338


namespace NUMINAMATH_CALUDE_bond_interest_rate_l4023_402341

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := 0.04

/-- The initial investment amount in yuan -/
def initial_investment : ℝ := 1000

/-- The amount spent after the first maturity in yuan -/
def spent_amount : ℝ := 440

/-- The final amount received after the second maturity in yuan -/
def final_amount : ℝ := 624

/-- Theorem stating that the annual interest rate is 4% given the problem conditions -/
theorem bond_interest_rate :
  (initial_investment * (1 + annual_interest_rate) - spent_amount) * (1 + annual_interest_rate) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_bond_interest_rate_l4023_402341


namespace NUMINAMATH_CALUDE_tv_sale_increase_l4023_402335

theorem tv_sale_increase (original_price original_quantity : ℝ) 
  (h_price_reduction : ℝ) (h_net_effect : ℝ) :
  h_price_reduction = 0.2 →
  h_net_effect = 0.44000000000000014 →
  ∃ (new_quantity : ℝ),
    (1 - h_price_reduction) * original_price * new_quantity = 
      (1 + h_net_effect) * original_price * original_quantity ∧
    (new_quantity / original_quantity - 1) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_tv_sale_increase_l4023_402335


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l4023_402386

theorem right_triangle_trig_identity (A B C : Real) : 
  -- ABC is a right-angled triangle with right angle at C
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 
  A + B + C = π / 2 ∧ 
  C = π / 2 →
  -- The trigonometric identity
  Real.sin A * Real.sin B * Real.sin (A - B) + 
  Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + 
  Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l4023_402386


namespace NUMINAMATH_CALUDE_log_always_defined_range_log_sometimes_undefined_range_l4023_402336

-- Define the function f(m, x)
def f (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + m + 3

-- Theorem 1: Range of m for which the logarithm is always defined
theorem log_always_defined_range (m : ℝ) :
  (∀ x : ℝ, f m x > 0) ↔ m ∈ Set.Ici 0 ∩ Set.Iio 1 :=
sorry

-- Theorem 2: Range of m for which the logarithm is undefined for some x
theorem log_sometimes_undefined_range (m : ℝ) :
  (∃ x : ℝ, f m x ≤ 0) ↔ m ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_log_always_defined_range_log_sometimes_undefined_range_l4023_402336


namespace NUMINAMATH_CALUDE_range_of_sum_l4023_402353

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) :
  ∃ (z : ℝ), z = x + y ∧ -2 ≤ z ∧ z ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_sum_l4023_402353


namespace NUMINAMATH_CALUDE_largest_digit_sum_l4023_402327

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c z : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 12 →
  a + b + c ≤ 8 ∧ ∃ a' b' c' z', 
    is_digit a' ∧ is_digit b' ∧ is_digit c' ∧
    (100 * a' + 10 * b' + c' : ℚ) / 1000 = 1 / z' ∧
    0 < z' ∧ z' ≤ 12 ∧
    a' + b' + c' = 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l4023_402327


namespace NUMINAMATH_CALUDE_amanda_keeps_121_candy_bars_l4023_402347

/-- The number of candy bars Amanda keeps for herself after four days of transactions --/
def amanda_candy_bars : ℕ :=
  let initial := 7
  let day1_remaining := initial - (initial / 3)
  let day2_total := day1_remaining + 30
  let day2_remaining := day2_total - (day2_total / 4)
  let day3_gift := day2_remaining * 3
  let day3_remaining := day2_remaining + (day3_gift / 2)
  let day4_bought := 20
  let day4_remaining := day3_remaining + (day4_bought / 3)
  day4_remaining

/-- Theorem stating that Amanda keeps 121 candy bars for herself --/
theorem amanda_keeps_121_candy_bars : amanda_candy_bars = 121 := by
  sorry

end NUMINAMATH_CALUDE_amanda_keeps_121_candy_bars_l4023_402347


namespace NUMINAMATH_CALUDE_total_first_grade_muffins_l4023_402317

def mrs_brier_muffins : ℕ := 218
def mrs_macadams_muffins : ℕ := 320
def mrs_flannery_muffins : ℕ := 417
def mrs_smith_muffins : ℕ := 292
def mr_jackson_muffins : ℕ := 389

theorem total_first_grade_muffins :
  mrs_brier_muffins + mrs_macadams_muffins + mrs_flannery_muffins +
  mrs_smith_muffins + mr_jackson_muffins = 1636 := by
  sorry

end NUMINAMATH_CALUDE_total_first_grade_muffins_l4023_402317


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_24_58_l4023_402379

/-- Represents a high school with average test scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  total_avg : ℝ

/-- Calculates the combined average score for boys given two schools and the combined girls' average -/
def combined_boys_avg (lincoln : School) (madison : School) (combined_girls_avg : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the combined boys' average is approximately 24.58 -/
theorem combined_boys_avg_is_24_58 
  (lincoln : School)
  (madison : School)
  (combined_girls_avg : ℝ)
  (h1 : lincoln.boys_avg = 65)
  (h2 : lincoln.girls_avg = 70)
  (h3 : lincoln.total_avg = 68)
  (h4 : madison.boys_avg = 75)
  (h5 : madison.girls_avg = 85)
  (h6 : madison.total_avg = 78)
  (h7 : combined_girls_avg = 80) :
  ∃ ε > 0, |combined_boys_avg lincoln madison combined_girls_avg - 24.58| < ε :=
by sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_24_58_l4023_402379


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4023_402326

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x ∧ -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4023_402326


namespace NUMINAMATH_CALUDE_point_sum_coordinates_l4023_402399

/-- Given that (3, 8) is on the graph of y = g(x), prove that the sum of the coordinates
    of the point on the graph of 5y = 4g(2x) + 6 is 9.1 -/
theorem point_sum_coordinates (g : ℝ → ℝ) (h : g 3 = 8) :
  ∃ x y : ℝ, 5 * y = 4 * g (2 * x) + 6 ∧ x + y = 9.1 := by
  sorry

end NUMINAMATH_CALUDE_point_sum_coordinates_l4023_402399


namespace NUMINAMATH_CALUDE_equation_solution_l4023_402340

theorem equation_solution : 
  ∀ m n : ℕ, 19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4023_402340


namespace NUMINAMATH_CALUDE_range_of_expression_l4023_402320

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l4023_402320


namespace NUMINAMATH_CALUDE_rectangle_properties_l4023_402374

/-- The equation representing the roots of the rectangle's sides -/
def side_equation (m x : ℝ) : Prop := x^2 - m*x + m/2 - 1/4 = 0

/-- The condition for the rectangle to be a square -/
def is_square (m : ℝ) : Prop := ∃ x : ℝ, side_equation m x ∧ ∀ y : ℝ, side_equation m y → y = x

/-- The perimeter of the rectangle given one side length -/
def perimeter (ab bc : ℝ) : ℝ := 2 * (ab + bc)

theorem rectangle_properties :
  (∃ m : ℝ, is_square m ∧ ∃ x : ℝ, side_equation m x ∧ x = 1/2) ∧
  (∃ m : ℝ, side_equation m 2 ∧ ∃ bc : ℝ, side_equation m bc ∧ perimeter 2 bc = 5) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_properties_l4023_402374


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l4023_402368

/-- The function f(x) = 3x + ax^3 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x + a * x^3

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 + 3 * a * x^2

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  f_derivative a 1 = 6 → a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l4023_402368


namespace NUMINAMATH_CALUDE_log_equation_roots_range_l4023_402351

-- Define the logarithmic equation
def log_equation (x a : ℝ) : Prop :=
  Real.log (x - 1) + Real.log (3 - x) = Real.log (a - x)

-- Define the condition for two distinct real roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3 ∧
  log_equation x₁ a ∧ log_equation x₂ a

-- Theorem statement
theorem log_equation_roots_range :
  ∀ a : ℝ, has_two_distinct_roots a ↔ 3 < a ∧ a < 13/4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_roots_range_l4023_402351


namespace NUMINAMATH_CALUDE_cow_daily_water_consumption_l4023_402369

/-- The number of cows on Mr. Reyansh's farm -/
def num_cows : ℕ := 40

/-- The ratio of sheep to cows on Mr. Reyansh's farm -/
def sheep_to_cow_ratio : ℕ := 10

/-- The ratio of water consumption of a sheep to a cow -/
def sheep_to_cow_water_ratio : ℚ := 1/4

/-- Total water usage for all animals in a week (in liters) -/
def total_weekly_water : ℕ := 78400

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem cow_daily_water_consumption :
  ∃ (cow_daily_water : ℚ),
    cow_daily_water * (num_cows : ℚ) * days_in_week +
    cow_daily_water * sheep_to_cow_water_ratio * (num_cows * sheep_to_cow_ratio : ℚ) * days_in_week =
    total_weekly_water ∧
    cow_daily_water = 80 := by
  sorry

end NUMINAMATH_CALUDE_cow_daily_water_consumption_l4023_402369


namespace NUMINAMATH_CALUDE_missing_village_population_l4023_402303

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 1249]
def total_villages : ℕ := 7
def average_population : ℕ := 1000

theorem missing_village_population :
  (village_populations.sum + (total_villages * average_population - village_populations.sum)) / total_villages = average_population ∧
  total_villages * average_population - village_populations.sum = 980 :=
by sorry

end NUMINAMATH_CALUDE_missing_village_population_l4023_402303


namespace NUMINAMATH_CALUDE_trig_equation_solution_l4023_402377

theorem trig_equation_solution (x : Real) :
  0 < x ∧ x < 180 →
  Real.tan ((150 - x) * Real.pi / 180) = 
    (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) / 
    (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180)) →
  x = 100 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l4023_402377


namespace NUMINAMATH_CALUDE_stratified_sampling_third_grade_l4023_402339

/-- Represents the number of students to be sampled from each grade in a stratified sampling -/
structure StratifiedSample where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the stratified sample given the total sample size and the ratio of students in each grade -/
def calculateStratifiedSample (totalSample : ℕ) (ratio1 : ℕ) (ratio2 : ℕ) (ratio3 : ℕ) : StratifiedSample :=
  let totalRatio := ratio1 + ratio2 + ratio3
  { first := (ratio1 * totalSample) / totalRatio,
    second := (ratio2 * totalSample) / totalRatio,
    third := (ratio3 * totalSample) / totalRatio }

theorem stratified_sampling_third_grade 
  (totalSample : ℕ) (ratio1 ratio2 ratio3 : ℕ) 
  (h1 : totalSample = 50)
  (h2 : ratio1 = 3)
  (h3 : ratio2 = 3)
  (h4 : ratio3 = 4) :
  (calculateStratifiedSample totalSample ratio1 ratio2 ratio3).third = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_grade_l4023_402339


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_l4023_402304

/-- Proves that a rectangular field with one side of 15 m and an area of 120 m² has a diagonal of 17 m -/
theorem rectangular_field_diagonal (side : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side = 15 → area = 120 → diagonal = 17 → 
  area = side * (area / side) ∧ diagonal^2 = side^2 + (area / side)^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_l4023_402304


namespace NUMINAMATH_CALUDE_table_tennis_cost_calculation_l4023_402354

/-- Represents the cost calculation for table tennis equipment purchase options. -/
def TableTennisCost (x : ℕ) : Prop :=
  (x > 20) →
  let racketPrice : ℕ := 80
  let ballPrice : ℕ := 20
  let racketCount : ℕ := 20
  let option1Cost : ℕ := racketPrice * racketCount + ballPrice * (x - racketCount)
  let option2Cost : ℕ := ((racketPrice * racketCount + ballPrice * x) * 9) / 10
  (option1Cost = 20 * x + 1200) ∧ (option2Cost = 18 * x + 1440)

/-- Theorem stating the cost calculation for both options is correct for any valid x. -/
theorem table_tennis_cost_calculation (x : ℕ) : TableTennisCost x := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_cost_calculation_l4023_402354


namespace NUMINAMATH_CALUDE_bird_sale_problem_l4023_402358

theorem bird_sale_problem (x y : ℝ) :
  x > 0 ∧ y > 0 ∧             -- Both purchase prices are positive
  0.8 * x = 1.2 * y ∧         -- Both birds sold for the same price
  (0.8 * x - x) + (1.2 * y - y) = -10 -- Total loss is 10 units
  →
  x = 30 ∧ y = 20 ∧ 0.8 * x = 24 := by
sorry

end NUMINAMATH_CALUDE_bird_sale_problem_l4023_402358


namespace NUMINAMATH_CALUDE_rose_purchase_problem_l4023_402315

theorem rose_purchase_problem :
  ∃! (x y : ℤ), 
    (y = 1) ∧ 
    (x > 0) ∧
    (100 / x : ℚ) - (200 / (x + 10) : ℚ) = 80 / 12 ∧
    x = 5 ∧
    y = 1 := by
  sorry

end NUMINAMATH_CALUDE_rose_purchase_problem_l4023_402315


namespace NUMINAMATH_CALUDE_one_positive_root_l4023_402385

def f (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem one_positive_root :
  ∃! x : ℝ, x > 0 ∧ x < 1 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_positive_root_l4023_402385


namespace NUMINAMATH_CALUDE_money_ratio_is_two_to_one_l4023_402307

/-- The ratio of Peter's money to John's money -/
def money_ratio : ℚ :=
  let peter_money : ℕ := 320
  let quincy_money : ℕ := peter_money + 20
  let andrew_money : ℕ := quincy_money + (quincy_money * 15 / 100)
  let total_money : ℕ := 1200 + 11
  let john_money : ℕ := total_money - peter_money - quincy_money - andrew_money
  peter_money / john_money

theorem money_ratio_is_two_to_one : money_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_is_two_to_one_l4023_402307


namespace NUMINAMATH_CALUDE_exists_real_geq_3_is_particular_l4023_402330

-- Define what a particular proposition is
def is_particular_proposition (p : Prop) : Prop :=
  ∃ (x : Type), p = ∃ (y : x), true

-- State the theorem
theorem exists_real_geq_3_is_particular : 
  is_particular_proposition (∃ (x : ℝ), x ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_exists_real_geq_3_is_particular_l4023_402330


namespace NUMINAMATH_CALUDE_factor_expression_l4023_402352

theorem factor_expression (x : ℝ) : 5*x*(x+2) + 9*(x+2) = (x+2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4023_402352


namespace NUMINAMATH_CALUDE_eggs_needed_proof_l4023_402321

def recipe_eggs : ℕ := 2
def recipe_people : ℕ := 4
def target_people : ℕ := 8
def available_eggs : ℕ := 3

theorem eggs_needed_proof : 
  (target_people / recipe_people * recipe_eggs) - available_eggs = 1 := by
sorry

end NUMINAMATH_CALUDE_eggs_needed_proof_l4023_402321


namespace NUMINAMATH_CALUDE_average_math_chem_score_l4023_402370

theorem average_math_chem_score (math physics chem : ℕ) : 
  math + physics = 40 →
  chem = physics + 20 →
  (math + chem) / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_average_math_chem_score_l4023_402370


namespace NUMINAMATH_CALUDE_congruence_problem_l4023_402310

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 19 = 3 → (3 * x + 15) % 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4023_402310


namespace NUMINAMATH_CALUDE_solution_is_ten_l4023_402306

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem solution_is_ten :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_ten_l4023_402306


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l4023_402329

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 + 17*x - 60 = 0 ∧ ∀ y, y^2 + 17*y - 60 = 0 → x ≤ y →
  x = -20 :=
sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l4023_402329


namespace NUMINAMATH_CALUDE_convex_number_probability_l4023_402314

-- Define the set of digits
def Digits : Finset Nat := {1, 2, 3, 4}

-- Define a three-digit number
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  digit_in_range : hundreds ∈ Digits ∧ tens ∈ Digits ∧ units ∈ Digits

-- Define a convex number
def isConvex (n : ThreeDigitNumber) : Prop :=
  n.hundreds < n.tens ∧ n.tens > n.units

-- Define the set of all possible three-digit numbers
def allNumbers : Finset ThreeDigitNumber := sorry

-- Define the set of convex numbers
def convexNumbers : Finset ThreeDigitNumber := sorry

-- Theorem to prove
theorem convex_number_probability :
  (Finset.card convexNumbers : Rat) / (Finset.card allNumbers : Rat) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_convex_number_probability_l4023_402314


namespace NUMINAMATH_CALUDE_xy_value_l4023_402382

theorem xy_value (x y : ℝ) (h : |x - y + 1| + (y + 5)^2 = 0) : x * y = 30 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l4023_402382


namespace NUMINAMATH_CALUDE_quadratic_coefficient_is_one_l4023_402322

/-- The quadratic equation x^2 - 2x + 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x + 1 = 0

/-- The coefficient of the quadratic term in the equation x^2 - 2x + 1 = 0 -/
def quadratic_coefficient : ℝ := 1

theorem quadratic_coefficient_is_one : 
  quadratic_coefficient = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_is_one_l4023_402322


namespace NUMINAMATH_CALUDE_andreas_erasers_l4023_402394

theorem andreas_erasers (andrea_erasers : ℕ) : 
  (4 * andrea_erasers = andrea_erasers + 12) → andrea_erasers = 4 := by
  sorry

end NUMINAMATH_CALUDE_andreas_erasers_l4023_402394


namespace NUMINAMATH_CALUDE_additional_members_needed_club_membership_increase_l4023_402344

/-- The number of additional members needed for a club to reach its desired membership. -/
theorem additional_members_needed (current_members : ℕ) (desired_members : ℕ) : ℕ :=
  desired_members - current_members

/-- Proof that the club needs 15 additional members. -/
theorem club_membership_increase : additional_members_needed 10 25 = 15 := by
  -- The proof goes here
  sorry

#check club_membership_increase

end NUMINAMATH_CALUDE_additional_members_needed_club_membership_increase_l4023_402344


namespace NUMINAMATH_CALUDE_fraction_power_product_l4023_402388

theorem fraction_power_product : (8 / 9 : ℚ)^3 * (1 / 3 : ℚ)^3 = 512 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l4023_402388


namespace NUMINAMATH_CALUDE_square_root_equation_l4023_402363

theorem square_root_equation (n : ℝ) : 3 * Real.sqrt (8 + n) = 15 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l4023_402363


namespace NUMINAMATH_CALUDE_fencing_tournament_l4023_402305

theorem fencing_tournament (n : ℕ) : n > 0 → (
  let total_participants := 4*n
  let total_bouts := (total_participants * (total_participants - 1)) / 2
  let womens_wins := 2*n*(3*n)
  let mens_wins := 3*n*(n + 3*n - 1)
  womens_wins * 3 = mens_wins * 2 ∧ womens_wins + mens_wins = total_bouts
) → n = 4 := by sorry

end NUMINAMATH_CALUDE_fencing_tournament_l4023_402305


namespace NUMINAMATH_CALUDE_child_ticket_price_l4023_402365

theorem child_ticket_price 
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (adult_price : ℕ)
  (child_tickets : ℕ)
  (h1 : total_tickets = 130)
  (h2 : total_receipts = 840)
  (h3 : adult_price = 12)
  (h4 : child_tickets = 90) :
  (total_receipts - (total_tickets - child_tickets) * adult_price) / child_tickets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_price_l4023_402365


namespace NUMINAMATH_CALUDE_ab_equals_six_l4023_402342

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l4023_402342


namespace NUMINAMATH_CALUDE_division_problem_l4023_402357

theorem division_problem : (64 : ℝ) / 0.08 = 800 := by sorry

end NUMINAMATH_CALUDE_division_problem_l4023_402357


namespace NUMINAMATH_CALUDE_bobs_first_lap_time_l4023_402392

/-- Proves that the time for the first lap is 70 seconds given the conditions of Bob's run --/
theorem bobs_first_lap_time (track_length : ℝ) (num_laps : ℕ) (time_second_lap : ℝ) (time_third_lap : ℝ) (average_speed : ℝ) :
  track_length = 400 →
  num_laps = 3 →
  time_second_lap = 85 →
  time_third_lap = 85 →
  average_speed = 5 →
  (track_length * num_laps) / average_speed - (time_second_lap + time_third_lap) = 70 :=
by sorry

end NUMINAMATH_CALUDE_bobs_first_lap_time_l4023_402392


namespace NUMINAMATH_CALUDE_valid_triplets_eq_solution_set_l4023_402319

def is_valid_triplet (a b c : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ (a * b * c - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0

def valid_triplets : Set (ℕ × ℕ × ℕ) :=
  {t | is_valid_triplet t.1 t.2.1 t.2.2}

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(3, 5, 15), (3, 15, 5), (2, 4, 8), (2, 8, 4), (2, 2, 4), (2, 4, 2), (2, 2, 2)}

theorem valid_triplets_eq_solution_set : valid_triplets = solution_set := by
  sorry

end NUMINAMATH_CALUDE_valid_triplets_eq_solution_set_l4023_402319


namespace NUMINAMATH_CALUDE_max_value_expression_l4023_402332

theorem max_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/4) ≤ 10/3 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    a' + Real.sqrt (a' * b') + (a' * b' * c') ^ (1/4) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l4023_402332


namespace NUMINAMATH_CALUDE_broken_line_length_bound_l4023_402378

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a broken line on a chessboard -/
structure BrokenLine :=
  (board : Chessboard)
  (isClosed : Bool)
  (noSelfIntersections : Bool)
  (joinsAdjacentCells : Bool)
  (isSymmetricToDiagonal : Bool)

/-- Calculates the length of a broken line -/
def brokenLineLength (line : BrokenLine) : ℝ :=
  sorry

/-- Theorem: The length of a specific broken line on a 15x15 chessboard is at most 200 -/
theorem broken_line_length_bound (line : BrokenLine) :
  line.board.size = 15 →
  line.isClosed = true →
  line.noSelfIntersections = true →
  line.joinsAdjacentCells = true →
  line.isSymmetricToDiagonal = true →
  brokenLineLength line ≤ 200 :=
by sorry

end NUMINAMATH_CALUDE_broken_line_length_bound_l4023_402378


namespace NUMINAMATH_CALUDE_ashton_initial_boxes_l4023_402355

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 14

/-- The number of pencils Ashton gave to his brother -/
def pencils_given : ℕ := 6

/-- The number of pencils Ashton had left after giving some away -/
def pencils_left : ℕ := 22

/-- The number of boxes Ashton had initially -/
def initial_boxes : ℕ := 2

theorem ashton_initial_boxes :
  initial_boxes * pencils_per_box = pencils_left + pencils_given :=
sorry

end NUMINAMATH_CALUDE_ashton_initial_boxes_l4023_402355


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l4023_402397

theorem gcd_of_specific_numbers : 
  let m : ℕ := 555555555
  let n : ℕ := 1111111111
  Nat.gcd m n = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l4023_402397


namespace NUMINAMATH_CALUDE_is_quadratic_equation_f_l4023_402396

-- Define a quadratic equation in one variable
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation (x-1)(x+2)=1
def f (x : ℝ) : ℝ := (x - 1) * (x + 2) - 1

-- Theorem statement
theorem is_quadratic_equation_f : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_f_l4023_402396


namespace NUMINAMATH_CALUDE_solution_set_l4023_402301

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f(x+1) is an odd function
axiom f_odd : ∀ x : ℝ, f (x + 1) = -f (-x - 1)

-- For any unequal real numbers x₁, x₂: x₁f(x₁) + x₂f(x₂) > x₁f(x₂) + x₂f(x₁)
axiom f_inequality : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Theorem: The solution set of f(2-x) < 0 is (1,+∞)
theorem solution_set : {x : ℝ | f (2 - x) < 0} = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l4023_402301


namespace NUMINAMATH_CALUDE_cos_alpha_value_l4023_402362

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l4023_402362


namespace NUMINAMATH_CALUDE_cos_600_degrees_l4023_402359

theorem cos_600_degrees : Real.cos (600 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_cos_600_degrees_l4023_402359


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l4023_402346

theorem quadratic_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = -a ∧ x₁ * x₂ = b) ∧
               (2*x₁ + 2*x₂ = -b ∧ 4*x₁*x₂ = c)) →
  a / c = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l4023_402346


namespace NUMINAMATH_CALUDE_range_of_q_l4023_402325

-- Define the function q(x)
def q (x : ℝ) : ℝ := (x^2 + 2)^3

-- State the theorem
theorem range_of_q : 
  {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ q x = y} = {y : ℝ | y ≥ 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_q_l4023_402325


namespace NUMINAMATH_CALUDE_max_distance_between_sets_l4023_402345

theorem max_distance_between_sets : ∃ (a b : ℂ),
  (a^4 - 16 = 0) ∧
  (b^4 - 16*b^3 - 16*b + 256 = 0) ∧
  (∀ (x y : ℂ), (x^4 - 16 = 0) → (y^4 - 16*y^3 - 16*y + 256 = 0) →
    Complex.abs (x - y) ≤ Complex.abs (a - b)) ∧
  Complex.abs (a - b) = 2 * Real.sqrt 65 :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_sets_l4023_402345


namespace NUMINAMATH_CALUDE_stream_current_rate_l4023_402384

/-- Represents the problem of finding the stream's current rate given rowing conditions. -/
theorem stream_current_rate
  (distance : ℝ)
  (normal_time_diff : ℝ)
  (triple_speed_time_diff : ℝ)
  (h1 : distance = 18)
  (h2 : normal_time_diff = 4)
  (h3 : triple_speed_time_diff = 2)
  (h4 : ∀ (r w : ℝ),
    (distance / (r + w) + normal_time_diff = distance / (r - w)) →
    (distance / (3 * r + w) + triple_speed_time_diff = distance / (3 * r - w)) →
    w = 9 / 8) :
  ∃ (w : ℝ), w = 9 / 8 ∧ 
    (∃ (r : ℝ), 
      (distance / (r + w) + normal_time_diff = distance / (r - w)) ∧
      (distance / (3 * r + w) + triple_speed_time_diff = distance / (3 * r - w))) :=
by
  sorry

end NUMINAMATH_CALUDE_stream_current_rate_l4023_402384


namespace NUMINAMATH_CALUDE_rectangle_fold_ef_length_l4023_402313

-- Define the rectangle
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the fold
structure Fold :=
  (distanceFromB : ℝ)
  (distanceFromC : ℝ)

-- Define the theorem
theorem rectangle_fold_ef_length 
  (rect : Rectangle)
  (fold : Fold)
  (h1 : rect.AB = 4)
  (h2 : rect.BC = 8)
  (h3 : fold.distanceFromB = 3)
  (h4 : fold.distanceFromC = 5)
  (h5 : fold.distanceFromB + fold.distanceFromC = rect.BC) :
  let EF := Real.sqrt ((rect.AB ^ 2) + (fold.distanceFromB ^ 2))
  EF = 5 := by sorry

end NUMINAMATH_CALUDE_rectangle_fold_ef_length_l4023_402313


namespace NUMINAMATH_CALUDE_x_value_l4023_402381

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^3) (h2 : x/9 = 9*y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4023_402381


namespace NUMINAMATH_CALUDE_cos_225_degrees_l4023_402391

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l4023_402391


namespace NUMINAMATH_CALUDE_ratio_of_P_and_Q_l4023_402387

-- Define the equation as a function
def equation (P Q : ℤ) (x : ℝ) : Prop :=
  (P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - x + 15) / (x^3 + x^2 - 30*x)) ∧ 
  (x ≠ -6) ∧ (x ≠ 0) ∧ (x ≠ 5)

-- State the theorem
theorem ratio_of_P_and_Q (P Q : ℤ) :
  (∀ x : ℝ, equation P Q x) → Q / P = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_P_and_Q_l4023_402387


namespace NUMINAMATH_CALUDE_edward_money_proof_l4023_402349

/-- The amount of money Edward had before spending, given his expenses and remaining money. -/
def edward_initial_money (books_cost pens_cost remaining : ℕ) : ℕ :=
  books_cost + pens_cost + remaining

/-- Theorem stating that Edward's initial money was $41 given the problem conditions. -/
theorem edward_money_proof :
  edward_initial_money 6 16 19 = 41 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_proof_l4023_402349


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l4023_402309

theorem ellipse_slope_product (a b m n x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (x^2 / a^2 + y^2 / b^2 = 1) →
  (m^2 / a^2 + n^2 / b^2 = 1) →
  ((y - n) / (x - m)) * ((y + n) / (x + m)) = -b^2 / a^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_l4023_402309


namespace NUMINAMATH_CALUDE_not_right_triangle_l4023_402300

theorem not_right_triangle : ∃ (a b c : ℝ), 
  (a = Real.sqrt 3 ∧ b = 4 ∧ c = 5) ∧ 
  (a^2 + b^2 ≠ c^2) ∧
  (∀ (x y z : ℝ), 
    ((x = 1 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 3) ∨
     (x = 7 ∧ y = 24 ∧ z = 25) ∨
     (x = 5 ∧ y = 12 ∧ z = 13)) →
    (x^2 + y^2 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l4023_402300


namespace NUMINAMATH_CALUDE_more_cylindrical_sandcastles_l4023_402356

/-- Represents the sandbox and sandcastle properties -/
structure Sandbox :=
  (base_area : ℝ)
  (sand_height : ℝ)
  (bucket_height : ℝ)
  (cylinder_base_area : ℝ)
  (m : ℕ)  -- number of cylindrical sandcastles
  (n : ℕ)  -- number of conical sandcastles

/-- Theorem stating that Masha's cylindrical sandcastles are more numerous -/
theorem more_cylindrical_sandcastles (sb : Sandbox) 
  (h1 : sb.sand_height = 1)
  (h2 : sb.bucket_height = 2)
  (h3 : sb.base_area = sb.cylinder_base_area * (sb.m + sb.n))
  (h4 : sb.base_area * sb.sand_height = 
        sb.cylinder_base_area * sb.bucket_height * sb.m + 
        (1/3) * sb.cylinder_base_area * sb.bucket_height * sb.n) :
  sb.m > sb.n := by
  sorry

end NUMINAMATH_CALUDE_more_cylindrical_sandcastles_l4023_402356


namespace NUMINAMATH_CALUDE_food_cost_calculation_l4023_402376

def hospital_bill_breakdown (total : ℝ) (medication_percent : ℝ) (overnight_percent : ℝ) (ambulance : ℝ) : ℝ := 
  let medication := medication_percent * total
  let remaining_after_medication := total - medication
  let overnight := overnight_percent * remaining_after_medication
  let food := total - medication - overnight - ambulance
  food

theorem food_cost_calculation :
  hospital_bill_breakdown 5000 0.5 0.25 1700 = 175 := by
  sorry

end NUMINAMATH_CALUDE_food_cost_calculation_l4023_402376


namespace NUMINAMATH_CALUDE_min_style_A_purchase_correct_l4023_402302

/-- Represents the clothing store problem -/
structure ClothingStore where
  total_pieces : ℕ
  total_cost : ℕ
  unit_price_A : ℕ
  unit_price_B : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ
  other_store_purchase : ℕ
  other_store_min_profit : ℕ

/-- The minimum number of style A clothing pieces to be purchased by another store -/
def min_style_A_purchase (store : ClothingStore) : ℕ :=
  23

/-- Theorem stating that the minimum number of style A clothing pieces to be purchased
    by another store is correct given the conditions -/
theorem min_style_A_purchase_correct (store : ClothingStore)
  (h1 : store.total_pieces = 100)
  (h2 : store.total_cost = 11200)
  (h3 : store.unit_price_A = 120)
  (h4 : store.unit_price_B = 100)
  (h5 : store.selling_price_A = 200)
  (h6 : store.selling_price_B = 140)
  (h7 : store.other_store_purchase = 60)
  (h8 : store.other_store_min_profit = 3300) :
  ∀ m : ℕ, m ≥ min_style_A_purchase store →
    (store.selling_price_A - store.unit_price_A) * m +
    (store.selling_price_B - store.unit_price_B) * (store.other_store_purchase - m) ≥
    store.other_store_min_profit :=
  sorry

end NUMINAMATH_CALUDE_min_style_A_purchase_correct_l4023_402302


namespace NUMINAMATH_CALUDE_binomial_fraction_is_integer_l4023_402328

theorem binomial_fraction_is_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ m : ℤ, (n - 2*k - 1 : ℚ) / (k + 1 : ℚ) * (n.choose k) = m := by
  sorry

end NUMINAMATH_CALUDE_binomial_fraction_is_integer_l4023_402328


namespace NUMINAMATH_CALUDE_unique_solution_l4023_402366

def problem (a : ℕ) (x : ℕ) : Prop :=
  a > 0 ∧
  x > 0 ∧
  x < a ∧
  71 * x + 69 * (a - x) = 3480

theorem unique_solution :
  ∃! a x, problem a x ∧ a = 50 ∧ x = 15 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l4023_402366


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l4023_402360

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the downstream speed of a rower given upstream and still water speeds -/
def calculateDownstreamSpeed (upstream stillWater : ℝ) : ℝ :=
  2 * stillWater - upstream

/-- Theorem stating that given the upstream and still water speeds, 
    the calculated downstream speed is correct -/
theorem downstream_speed_calculation 
  (speed : RowerSpeed) 
  (h1 : speed.upstream = 25)
  (h2 : speed.stillWater = 32) :
  speed.downstream = calculateDownstreamSpeed speed.upstream speed.stillWater ∧ 
  speed.downstream = 39 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l4023_402360


namespace NUMINAMATH_CALUDE_range_of_a_l4023_402375

-- Define the propositions
def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 2 < 0

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, proposition_p a ∧ ¬proposition_q a ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4023_402375


namespace NUMINAMATH_CALUDE_existence_of_unsolvable_degree_l4023_402372

-- Define a polynomial equation of degree n
def PolynomialEquation (n : ℕ) := ℕ → ℝ → Prop

-- Define a solution expressed in terms of radicals
def RadicalSolution (n : ℕ) := ℕ → ℝ → Prop

-- Axiom: Quadratic equations have solutions in terms of radicals
axiom quadratic_solvable : ∀ (eq : PolynomialEquation 2), ∃ (sol : RadicalSolution 2), sol 2 = eq 2

-- Axiom: Cubic equations have solutions in terms of radicals
axiom cubic_solvable : ∀ (eq : PolynomialEquation 3), ∃ (sol : RadicalSolution 3), sol 3 = eq 3

-- Axiom: Quartic equations have solutions in terms of radicals
axiom quartic_solvable : ∀ (eq : PolynomialEquation 4), ∃ (sol : RadicalSolution 4), sol 4 = eq 4

-- Theorem: There exists a degree n such that not all polynomial equations of degree ≥ n are solvable by radicals
theorem existence_of_unsolvable_degree :
  ∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → ¬(∀ (eq : PolynomialEquation m), ∃ (sol : RadicalSolution m), sol m = eq m) :=
sorry

end NUMINAMATH_CALUDE_existence_of_unsolvable_degree_l4023_402372


namespace NUMINAMATH_CALUDE_rectangle_max_area_l4023_402389

/-- Given a rectangle with perimeter 30 inches and one side 3 inches longer than the other,
    the maximum possible area is 54 square inches. -/
theorem rectangle_max_area :
  ∀ x : ℝ,
  x > 0 →
  2 * (x + (x + 3)) = 30 →
  x * (x + 3) ≤ 54 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l4023_402389


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4023_402324

-- Problem 1
theorem problem_1 : (-1/3)⁻¹ - Real.sqrt 12 - (2 - Real.sqrt 3)^0 = -4 - 2 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 : (1 + 1/2) + (2^2 - 1)/2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4023_402324


namespace NUMINAMATH_CALUDE_mary_final_weight_l4023_402334

def weight_change (initial_weight : ℕ) : ℕ :=
  let first_loss := 12
  let second_gain := 2 * first_loss
  let third_loss := 3 * first_loss
  let final_gain := 6
  initial_weight - first_loss + second_gain - third_loss + final_gain

theorem mary_final_weight (initial_weight : ℕ) (h : initial_weight = 99) :
  weight_change initial_weight = 81 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_weight_l4023_402334


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4023_402316

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4023_402316


namespace NUMINAMATH_CALUDE_investment_sum_l4023_402395

/-- Proves that given a sum P invested at 18% p.a. for two years generates Rs. 600 more interest
    than if invested at 12% p.a. for the same period, then P = 5000. -/
theorem investment_sum (P : ℚ) : 
  (P * 18 * 2 / 100) - (P * 12 * 2 / 100) = 600 → P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l4023_402395


namespace NUMINAMATH_CALUDE_parallel_segment_length_l4023_402348

theorem parallel_segment_length (a b c d : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a = 300) (h3 : b = 320) (h4 : c = 400) :
  let s := (a + b + c) / 2
  let area_abc := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let area_dpd := area_abc / 4
  ∃ d : ℝ, d > 0 ∧ d^2 / a^2 = area_dpd / area_abc ∧ d = 150 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l4023_402348


namespace NUMINAMATH_CALUDE_union_complement_when_a_is_one_subset_iff_a_in_range_l4023_402398

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem 1
theorem union_complement_when_a_is_one :
  (Set.univ \ B) ∪ (A 1) = {x | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Theorem 2
theorem subset_iff_a_in_range :
  ∀ a : ℝ, A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_complement_when_a_is_one_subset_iff_a_in_range_l4023_402398


namespace NUMINAMATH_CALUDE_xy_max_and_x2_4y2_min_l4023_402364

theorem xy_max_and_x2_4y2_min (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + 2*y = 3) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x*y ≥ a*b) ∧
  x*y = 9/8 ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b = 3 → x^2 + 4*y^2 ≤ a^2 + 4*b^2) ∧
  x^2 + 4*y^2 = 9/2 :=
sorry

end NUMINAMATH_CALUDE_xy_max_and_x2_4y2_min_l4023_402364


namespace NUMINAMATH_CALUDE_quadratic_independent_of_x_squared_l4023_402361

/-- For a quadratic polynomial -3x^2 + mx^2 - x + 3, if its value is independent of the quadratic term of x, then m = 3 -/
theorem quadratic_independent_of_x_squared (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, -3*x^2 + m*x^2 - x + 3 = -x + k) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_independent_of_x_squared_l4023_402361


namespace NUMINAMATH_CALUDE_perpendicular_vectors_coefficient_l4023_402312

/-- Given two vectors in the plane, if one is perpendicular to a linear combination of both,
    then the coefficient in the linear combination is -5. -/
theorem perpendicular_vectors_coefficient (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, -1) →
  b = (6, -4) →
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) →
  t = -5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_coefficient_l4023_402312


namespace NUMINAMATH_CALUDE_cos_sin_sum_zero_implies_double_angle_sum_zero_l4023_402343

theorem cos_sin_sum_zero_implies_double_angle_sum_zero 
  (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_zero_implies_double_angle_sum_zero_l4023_402343


namespace NUMINAMATH_CALUDE_max_distinct_permutations_eight_points_l4023_402337

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for directed lines in a plane
def DirectedLine : Type := ℝ × ℝ × ℝ  -- ax + by + c = 0, with (a,b) ≠ (0,0)

-- Define a function to project a point onto a directed line
def project (p : Point) (l : DirectedLine) : ℝ := sorry

-- Define a function to get the permutation from projections
def getPermutation (points : List Point) (l : DirectedLine) : List ℕ := sorry

-- Define a function to count distinct permutations
def countDistinctPermutations (points : List Point) : ℕ := sorry

theorem max_distinct_permutations_eight_points :
  ∀ (points : List Point),
    points.length = 8 →
    points.Nodup →
    (∀ (l : DirectedLine), (getPermutation points l).Nodup) →
    countDistinctPermutations points ≤ 56 ∧
    ∃ (points' : List Point),
      points'.length = 8 ∧
      points'.Nodup ∧
      (∀ (l : DirectedLine), (getPermutation points' l).Nodup) ∧
      countDistinctPermutations points' = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_permutations_eight_points_l4023_402337


namespace NUMINAMATH_CALUDE_books_bought_proof_l4023_402318

/-- Calculates the number of books bought given a ratio and total items -/
def books_bought (book_ratio pen_ratio notebook_ratio total_items : ℕ) : ℕ :=
  let total_ratio := book_ratio + pen_ratio + notebook_ratio
  let sets := total_items / total_ratio
  book_ratio * sets

/-- Theorem: Given the specified ratio and total items, prove the number of books bought -/
theorem books_bought_proof :
  books_bought 7 3 2 600 = 350 := by
  sorry

end NUMINAMATH_CALUDE_books_bought_proof_l4023_402318


namespace NUMINAMATH_CALUDE_radian_to_degree_conversion_l4023_402331

theorem radian_to_degree_conversion (π : ℝ) (h : π * (180 / π) = 180) :
  (23 / 12) * π * (180 / π) = 345 :=
sorry

end NUMINAMATH_CALUDE_radian_to_degree_conversion_l4023_402331
