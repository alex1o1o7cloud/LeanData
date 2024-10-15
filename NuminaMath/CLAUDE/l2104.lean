import Mathlib

namespace NUMINAMATH_CALUDE_scheduling_methods_count_l2104_210484

/-- The number of days for scheduling --/
def num_days : ℕ := 7

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- Function to calculate the number of scheduling methods --/
def scheduling_methods : ℕ := 
  (num_days.choose num_volunteers) * (num_volunteers.factorial / 2)

/-- Theorem stating that the number of scheduling methods is 420 --/
theorem scheduling_methods_count : scheduling_methods = 420 := by
  sorry

end NUMINAMATH_CALUDE_scheduling_methods_count_l2104_210484


namespace NUMINAMATH_CALUDE_percentage_with_no_conditions_is_22_5_l2104_210408

/-- Represents the survey results of teachers' health conditions -/
structure SurveyResults where
  total : ℕ
  highBloodPressure : ℕ
  heartTrouble : ℕ
  diabetes : ℕ
  highBloodPressureAndHeartTrouble : ℕ
  highBloodPressureAndDiabetes : ℕ
  heartTroubleAndDiabetes : ℕ
  allThree : ℕ

/-- Calculates the percentage of teachers with none of the health conditions -/
def percentageWithNoConditions (results : SurveyResults) : ℚ :=
  let withConditions :=
    results.highBloodPressure +
    results.heartTrouble +
    results.diabetes -
    results.highBloodPressureAndHeartTrouble -
    results.highBloodPressureAndDiabetes -
    results.heartTroubleAndDiabetes +
    results.allThree
  let withoutConditions := results.total - withConditions
  (withoutConditions : ℚ) / results.total * 100

/-- The survey results from the problem -/
def surveyData : SurveyResults :=
  { total := 200
  , highBloodPressure := 90
  , heartTrouble := 60
  , diabetes := 30
  , highBloodPressureAndHeartTrouble := 25
  , highBloodPressureAndDiabetes := 15
  , heartTroubleAndDiabetes := 10
  , allThree := 5 }

theorem percentage_with_no_conditions_is_22_5 :
  percentageWithNoConditions surveyData = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_no_conditions_is_22_5_l2104_210408


namespace NUMINAMATH_CALUDE_smallest_b_value_l2104_210497

theorem smallest_b_value (a b : ℤ) (h1 : 9 < a ∧ a < 21) (h2 : b < 31) (h3 : (a : ℚ) / b = 2/3) :
  ∃ (n : ℤ), n = 14 ∧ n < b ∧ ∀ m, m < b → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2104_210497


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l2104_210423

theorem chocolate_bar_cost (total_bars : ℕ) (bars_sold : ℕ) (revenue : ℝ) : 
  total_bars = 7 → 
  bars_sold = total_bars - 4 → 
  revenue = 9 → 
  revenue / bars_sold = 3 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l2104_210423


namespace NUMINAMATH_CALUDE_cube_sum_equality_l2104_210446

theorem cube_sum_equality (x y z a b c : ℝ) 
  (hx : x^2 = a) (hy : y^2 = b) (hz : z^2 = c) :
  ∃ (s : ℝ), s = 1 ∨ s = -1 ∧ x^3 + y^3 + z^3 = s * (a^(3/2) + b^(3/2) + c^(3/2)) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l2104_210446


namespace NUMINAMATH_CALUDE_exam_mean_score_l2104_210456

theorem exam_mean_score (score_below mean standard_deviation : ℝ) 
  (h1 : score_below = mean - 2 * standard_deviation)
  (h2 : 98 = mean + 3 * standard_deviation)
  (h3 : score_below = 58) : mean = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2104_210456


namespace NUMINAMATH_CALUDE_complex_exponentiation_205_deg_72_l2104_210444

theorem complex_exponentiation_205_deg_72 :
  (Complex.exp (205 * π / 180 * Complex.I)) ^ 72 = -1/2 - Complex.I * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponentiation_205_deg_72_l2104_210444


namespace NUMINAMATH_CALUDE_evaluate_expression_l2104_210499

theorem evaluate_expression : 12.543 - 3.219 + 1.002 = 10.326 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2104_210499


namespace NUMINAMATH_CALUDE_initial_men_count_l2104_210489

/-- Represents the number of days the initial food supply lasts -/
def initial_days : ℕ := 22

/-- Represents the number of days that pass before additional men join -/
def days_before_addition : ℕ := 2

/-- Represents the number of additional men that join -/
def additional_men : ℕ := 3040

/-- Represents the number of days the food lasts after additional men join -/
def remaining_days : ℕ := 4

/-- Theorem stating that the initial number of men is 760 -/
theorem initial_men_count : ℕ := by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l2104_210489


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2104_210434

theorem arithmetic_calculations :
  (- 4 - (- 2) + (- 5) + 8 = 1) ∧
  (- 1^2023 + 16 / (-2)^2 * |-(1/4)| = 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2104_210434


namespace NUMINAMATH_CALUDE_difference_of_squares_l2104_210483

theorem difference_of_squares (m : ℝ) : m^2 - 144 = (m - 12) * (m + 12) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2104_210483


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_390_l2104_210433

theorem sin_n_equals_cos_390 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (390 * π / 180) →
  n = 60 ∨ n = 120 := by
sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_390_l2104_210433


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l2104_210457

theorem polynomial_identity_sum (d₁ d₂ d₃ e₁ e₂ e₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^6 + x^4 - x^2 + 1 = 
    (x^2 + d₁*x + e₁) * (x^2 + d₂*x + e₂) * (x^2 + d₃*x + e₃) * (x^2 + 1)) :
  d₁*e₁ + d₂*e₂ + d₃*e₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l2104_210457


namespace NUMINAMATH_CALUDE_steak_eaten_l2104_210429

theorem steak_eaten (original_weight : ℝ) (burned_fraction : ℝ) (eaten_fraction : ℝ) : 
  original_weight = 30 ∧ 
  burned_fraction = 0.5 ∧ 
  eaten_fraction = 0.8 → 
  original_weight * (1 - burned_fraction) * eaten_fraction = 12 := by
  sorry

end NUMINAMATH_CALUDE_steak_eaten_l2104_210429


namespace NUMINAMATH_CALUDE_fifteenth_term_is_198_l2104_210435

/-- A second-order arithmetic sequence is a sequence where the differences between consecutive terms form an arithmetic sequence. -/
def SecondOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d₁ d₂ : ℕ, ∀ n : ℕ, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + d₂

/-- The specific second-order arithmetic sequence from the problem. -/
def SpecificSequence (a : ℕ → ℕ) : Prop :=
  SecondOrderArithmeticSequence a ∧ a 1 = 2 ∧ a 2 = 3 ∧ a 3 = 6 ∧ a 4 = 11

theorem fifteenth_term_is_198 (a : ℕ → ℕ) (h : SpecificSequence a) : a 15 = 198 := by
  sorry

#check fifteenth_term_is_198

end NUMINAMATH_CALUDE_fifteenth_term_is_198_l2104_210435


namespace NUMINAMATH_CALUDE_rakesh_distance_l2104_210417

/-- Represents the walking problem with four people: Hiro, Rakesh, Sanjay, and Charu -/
structure WalkingProblem where
  hiro_distance : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The conditions of the walking problem -/
def walking_conditions (wp : WalkingProblem) : Prop :=
  wp.total_distance = 85 ∧
  wp.total_time = 20 ∧
  ∃ (rakesh_time sanjay_time charu_time : ℝ),
    rakesh_time = wp.total_time - (wp.total_time - 2) - sanjay_time - charu_time ∧
    charu_time = wp.total_time - (wp.total_time - 2) ∧
    wp.total_distance = wp.hiro_distance + (4 * wp.hiro_distance - 10) + (2 * wp.hiro_distance + 3) +
      ((4 * wp.hiro_distance - 10) + (2 * wp.hiro_distance + 3)) / 2

/-- The theorem stating Rakesh's walking distance -/
theorem rakesh_distance (wp : WalkingProblem) (h : walking_conditions wp) :
    4 * wp.hiro_distance - 10 = 28.2 := by
  sorry


end NUMINAMATH_CALUDE_rakesh_distance_l2104_210417


namespace NUMINAMATH_CALUDE_fraction_simplification_l2104_210463

theorem fraction_simplification :
  201920192019 / 191719171917 = 673 / 639 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2104_210463


namespace NUMINAMATH_CALUDE_sum_reciprocals_l2104_210491

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω^2) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l2104_210491


namespace NUMINAMATH_CALUDE_intersection_M_N_l2104_210439

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2104_210439


namespace NUMINAMATH_CALUDE_fraction_equality_l2104_210416

theorem fraction_equality : (8 : ℝ) / (5 * 42) = 0.8 / (2.1 * 10) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2104_210416


namespace NUMINAMATH_CALUDE_power_multiplication_l2104_210474

theorem power_multiplication (a : ℝ) : a^5 * a^2 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2104_210474


namespace NUMINAMATH_CALUDE_charity_raffle_winnings_l2104_210496

theorem charity_raffle_winnings (W : ℝ) : 
  W / 2 - 2 = 55 → W = 114 := by
  sorry

end NUMINAMATH_CALUDE_charity_raffle_winnings_l2104_210496


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l2104_210479

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_equation_solutions :
  ∀ x y n : ℕ, (factorial x + factorial y) / factorial n = 3^n →
    ((x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l2104_210479


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2104_210402

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | -3 < x < 2}, prove the following statements -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) :
  (a < 0) ∧ 
  (a + b + c > 0) ∧
  (∀ x, c*x^2 + b*x + a < 0 ↔ -1/3 < x ∧ x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2104_210402


namespace NUMINAMATH_CALUDE_division_sum_dividend_l2104_210470

theorem division_sum_dividend (quotient divisor remainder : ℕ) : 
  quotient = 40 → divisor = 72 → remainder = 64 → 
  (divisor * quotient) + remainder = 2944 := by
sorry

end NUMINAMATH_CALUDE_division_sum_dividend_l2104_210470


namespace NUMINAMATH_CALUDE_window_area_theorem_l2104_210414

/-- Represents a rectangular glass pane with length and width in inches. -/
structure GlassPane where
  length : ℕ
  width : ℕ

/-- Calculates the area of a single glass pane in square inches. -/
def pane_area (pane : GlassPane) : ℕ :=
  pane.length * pane.width

/-- Represents a window composed of multiple identical glass panes. -/
structure Window where
  pane : GlassPane
  num_panes : ℕ

/-- Calculates the total area of a window in square inches. -/
def window_area (w : Window) : ℕ :=
  pane_area w.pane * w.num_panes

/-- Theorem: The area of a window with 8 panes, each 12 inches by 8 inches, is 768 square inches. -/
theorem window_area_theorem : 
  ∀ (w : Window), w.pane.length = 12 → w.pane.width = 8 → w.num_panes = 8 → 
  window_area w = 768 := by
  sorry

end NUMINAMATH_CALUDE_window_area_theorem_l2104_210414


namespace NUMINAMATH_CALUDE_crayons_count_l2104_210480

/-- Given a group of children where each child has a certain number of crayons,
    calculate the total number of crayons. -/
def total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : ℕ :=
  crayons_per_child * num_children

/-- Theorem stating that with 6 crayons per child and 12 children, 
    the total number of crayons is 72. -/
theorem crayons_count : total_crayons 6 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l2104_210480


namespace NUMINAMATH_CALUDE_cookies_remaining_l2104_210421

-- Define the given conditions
def pieces_per_pack : ℕ := 3
def original_packs : ℕ := 226
def packs_given_away : ℕ := 3

-- Define the theorem
theorem cookies_remaining :
  (original_packs - packs_given_away) * pieces_per_pack = 669 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l2104_210421


namespace NUMINAMATH_CALUDE_november_rainfall_is_180_inches_l2104_210473

/-- Calculates the total rainfall in November given the conditions -/
def november_rainfall (days_in_november : ℕ) (first_half_days : ℕ) (first_half_daily_rainfall : ℝ) : ℝ :=
  let second_half_days := days_in_november - first_half_days
  let second_half_daily_rainfall := 2 * first_half_daily_rainfall
  let first_half_total := first_half_daily_rainfall * first_half_days
  let second_half_total := second_half_daily_rainfall * second_half_days
  first_half_total + second_half_total

/-- Theorem stating that the total rainfall in November is 180 inches -/
theorem november_rainfall_is_180_inches :
  november_rainfall 30 15 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_november_rainfall_is_180_inches_l2104_210473


namespace NUMINAMATH_CALUDE_rectangle_relationships_l2104_210464

-- Define the rectangle
def rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ 2*x + 2*y = 10

-- Define the area function
def area (x y : ℝ) : ℝ := x * y

-- Theorem statement
theorem rectangle_relationships (x y : ℝ) (h : rectangle x y) :
  ∃ (a b : ℝ), y = a*x + b ∧    -- Linear relationship between y and x
  ∃ (p q r : ℝ), area x y = p*x^2 + q*x + r :=  -- Quadratic relationship between S and x
by sorry

end NUMINAMATH_CALUDE_rectangle_relationships_l2104_210464


namespace NUMINAMATH_CALUDE_dads_strawberry_weight_l2104_210426

/-- 
Given:
- The total initial weight of strawberries collected by Marco and his dad
- The weight of strawberries lost by Marco's dad
- The current weight of Marco's strawberries

Prove that the weight of Marco's dad's strawberries is equal to the difference between 
the total weight after loss and Marco's current weight of strawberries.
-/
theorem dads_strawberry_weight 
  (total_initial_weight : ℕ) 
  (weight_lost : ℕ) 
  (marcos_weight : ℕ) : 
  total_initial_weight - weight_lost - marcos_weight = 
    total_initial_weight - (weight_lost + marcos_weight) := by
  sorry

#check dads_strawberry_weight

end NUMINAMATH_CALUDE_dads_strawberry_weight_l2104_210426


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l2104_210430

def total_trip_time : ℕ := 90
def wait_for_cart : ℕ := 3
def wait_for_employee : ℕ := 13
def wait_for_stocker : ℕ := 14
def wait_in_line : ℕ := 18

theorem shopping_time_calculation :
  total_trip_time - (wait_for_cart + wait_for_employee + wait_for_stocker + wait_in_line) = 42 := by
  sorry

end NUMINAMATH_CALUDE_shopping_time_calculation_l2104_210430


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l2104_210405

/-- Calculates the upstream distance traveled by a boat in one hour -/
def upstreamDistance (stillWaterSpeed : ℝ) (downstreamDistance : ℝ) : ℝ :=
  let streamSpeed := downstreamDistance - stillWaterSpeed
  stillWaterSpeed - streamSpeed

theorem boat_upstream_distance :
  upstreamDistance 5 8 = 2 := by
  sorry

#eval upstreamDistance 5 8

end NUMINAMATH_CALUDE_boat_upstream_distance_l2104_210405


namespace NUMINAMATH_CALUDE_relationship_abcd_l2104_210458

theorem relationship_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abcd_l2104_210458


namespace NUMINAMATH_CALUDE_classroom_tables_count_l2104_210413

/-- The number of tables in Miss Smith's classroom --/
def number_of_tables : ℕ :=
  let total_students : ℕ := 47
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_students : ℕ := 3 * bathroom_students
  let new_group_students : ℕ := 2 * 4
  let exchange_students : ℕ := 3 * 3
  let missing_students : ℕ := bathroom_students + canteen_students + new_group_students + exchange_students
  let present_students : ℕ := total_students - missing_students
  present_students / students_per_table

theorem classroom_tables_count : number_of_tables = 6 := by
  sorry

end NUMINAMATH_CALUDE_classroom_tables_count_l2104_210413


namespace NUMINAMATH_CALUDE_available_storage_space_l2104_210482

/-- Represents a two-story warehouse with boxes stored on the second floor -/
structure Warehouse :=
  (second_floor_space : ℝ)
  (first_floor_space : ℝ)
  (box_space : ℝ)

/-- The conditions of the warehouse problem -/
def warehouse_conditions (w : Warehouse) : Prop :=
  w.first_floor_space = 2 * w.second_floor_space ∧
  w.box_space = w.second_floor_space / 4 ∧
  w.box_space = 5000

/-- The theorem stating the available storage space in the warehouse -/
theorem available_storage_space (w : Warehouse) 
  (h : warehouse_conditions w) : 
  w.first_floor_space + w.second_floor_space - w.box_space = 55000 := by
  sorry

end NUMINAMATH_CALUDE_available_storage_space_l2104_210482


namespace NUMINAMATH_CALUDE_peanuts_per_visit_l2104_210492

def store_visits : ℕ := 3
def total_peanuts : ℕ := 21

theorem peanuts_per_visit : total_peanuts / store_visits = 7 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_per_visit_l2104_210492


namespace NUMINAMATH_CALUDE_juvy_garden_chives_l2104_210432

theorem juvy_garden_chives (total_rows : ℕ) (plants_per_row : ℕ) 
  (parsley_rows : ℕ) (rosemary_rows : ℕ) (mint_rows : ℕ) (thyme_rows : ℕ) :
  total_rows = 50 →
  plants_per_row = 15 →
  parsley_rows = 5 →
  rosemary_rows = 7 →
  mint_rows = 10 →
  thyme_rows = 12 →
  (total_rows - (parsley_rows + rosemary_rows + mint_rows + thyme_rows)) * plants_per_row = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_juvy_garden_chives_l2104_210432


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2104_210406

theorem largest_integer_with_remainder : 
  ∀ n : ℕ, n < 100 ∧ n % 7 = 2 → n ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2104_210406


namespace NUMINAMATH_CALUDE_bijection_existence_l2104_210481

theorem bijection_existence :
  (∃ f : ℕ+ × ℕ+ → ℕ+, Function.Bijective f ∧
    (f (1, 1) = 1) ∧
    (∀ i > 1, ∃ d > 1, ∀ j, d ∣ f (i, j)) ∧
    (∀ j > 1, ∃ d > 1, ∀ i, d ∣ f (i, j))) ∧
  (∃ g : ℕ+ × ℕ+ → {n : ℕ+ // n ≠ 1}, Function.Bijective g ∧
    (∀ i, ∃ d > 1, ∀ j, d ∣ (g (i, j)).val) ∧
    (∀ j, ∃ d > 1, ∀ i, d ∣ (g (i, j)).val)) :=
by sorry

end NUMINAMATH_CALUDE_bijection_existence_l2104_210481


namespace NUMINAMATH_CALUDE_no_integer_solution_for_1980_l2104_210427

theorem no_integer_solution_for_1980 : ∀ m n : ℤ, m^2 + n^2 ≠ 1980 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_1980_l2104_210427


namespace NUMINAMATH_CALUDE_tangent_ratio_theorem_l2104_210495

theorem tangent_ratio_theorem (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (Real.cos θ ^ 2 + Real.sin θ * Real.cos θ) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_theorem_l2104_210495


namespace NUMINAMATH_CALUDE_equation_solution_l2104_210459

theorem equation_solution : 
  ∀ x : ℝ, 4 * x^2 - (x^2 - 2*x + 1) = 0 ↔ x = 1/3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2104_210459


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l2104_210425

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_slope_angle_at_zero :
  let slope := deriv f 0
  Real.arctan slope = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l2104_210425


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2104_210486

theorem unique_solution_exists (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃! x : ℝ, a^x = Real.log x / Real.log (1/4) := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2104_210486


namespace NUMINAMATH_CALUDE_admin_personnel_count_l2104_210407

/-- Represents the total number of employees in the unit -/
def total_employees : ℕ := 280

/-- Represents the sample size -/
def sample_size : ℕ := 56

/-- Represents the number of ordinary staff sampled -/
def ordinary_staff_sampled : ℕ := 49

/-- Calculates the number of administrative personnel -/
def admin_personnel : ℕ := total_employees - (total_employees * ordinary_staff_sampled / sample_size)

/-- Theorem stating that the number of administrative personnel is 35 -/
theorem admin_personnel_count : admin_personnel = 35 := by
  sorry

end NUMINAMATH_CALUDE_admin_personnel_count_l2104_210407


namespace NUMINAMATH_CALUDE_prob_sum_five_l2104_210475

/-- The number of sides on each die -/
def dice_sides : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def roll_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range dice_sides) (Finset.range dice_sides)

/-- The set of outcomes that sum to 5 -/
def sum_five : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 + p.2 = 5) roll_outcomes

/-- The probability of rolling a sum of 5 with two fair dice -/
theorem prob_sum_five :
  (Finset.card sum_five : ℚ) / (Finset.card roll_outcomes : ℚ) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_five_l2104_210475


namespace NUMINAMATH_CALUDE_min_moves_to_win_l2104_210468

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  black : Nat

/-- Represents the state of the game -/
structure GameState where
  round : Box
  square : Box

/-- Checks if two boxes have identical contents -/
def boxesIdentical (b1 b2 : Box) : Bool :=
  b1.white = b2.white ∧ b1.black = b2.black

/-- Defines a single move in the game -/
inductive Move
  | discard : Box → Move
  | transfer : Box → Box → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is in a winning state -/
def isWinningState (state : GameState) : Bool :=
  boxesIdentical state.round state.square

/-- The initial state of the game -/
def initialState : GameState :=
  { round := { white := 3, black := 10 }
  , square := { white := 0, black := 8 } }

/-- Theorem: The minimum number of moves to reach a winning state is 17 -/
theorem min_moves_to_win :
  ∃ (moves : List Move),
    moves.length = 17 ∧
    isWinningState (moves.foldl applyMove initialState) ∧
    ∀ (shorter_moves : List Move),
      shorter_moves.length < 17 →
      ¬isWinningState (shorter_moves.foldl applyMove initialState) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_win_l2104_210468


namespace NUMINAMATH_CALUDE_gcd_2720_1530_l2104_210462

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2720_1530_l2104_210462


namespace NUMINAMATH_CALUDE_triangle_ratio_l2104_210449

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b * sin(2A) = 3a * sin(B) and c = 2b, then a/b = √2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l2104_210449


namespace NUMINAMATH_CALUDE_pasha_mistake_l2104_210465

theorem pasha_mistake : 
  ¬ ∃ (K R O S C T A P : ℕ),
    (K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ A < 10 ∧ P < 10) ∧
    (K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ A ∧ K ≠ P ∧
     R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ A ∧ R ≠ P ∧
     O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ A ∧ O ≠ P ∧
     S ≠ C ∧ S ≠ T ∧ S ≠ A ∧ S ≠ P ∧
     C ≠ T ∧ C ≠ A ∧ C ≠ P ∧
     T ≠ A ∧ T ≠ P ∧
     A ≠ P) ∧
    (K * 10000 + R * 1000 + O * 100 + S * 10 + S + 2011 = 
     C * 10000 + T * 1000 + A * 100 + P * 10 + T) :=
by sorry

end NUMINAMATH_CALUDE_pasha_mistake_l2104_210465


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2104_210428

/-- Calculates the initial amount of money given the spending pattern and final amount --/
theorem initial_money_calculation (final_amount : ℚ) : 
  final_amount = 500 →
  ∃ initial_amount : ℚ,
    initial_amount * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = final_amount ∧
    initial_amount = 1250 :=
by sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2104_210428


namespace NUMINAMATH_CALUDE_ratio_average_problem_l2104_210415

theorem ratio_average_problem (a b c : ℕ+) (h_ratio : a.val / 2 = b.val / 3 ∧ b.val / 3 = c.val / 4) (h_a : a = 28) :
  (a.val + b.val + c.val) / 3 = 42 := by
sorry

end NUMINAMATH_CALUDE_ratio_average_problem_l2104_210415


namespace NUMINAMATH_CALUDE_shaded_area_is_700_l2104_210466

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the large square -/
def squareVertices : List Point := [
  ⟨0, 0⟩, ⟨40, 0⟩, ⟨40, 40⟩, ⟨0, 40⟩
]

/-- The vertices of the shaded polygon -/
def shadedVertices : List Point := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨40, 30⟩, ⟨40, 40⟩, ⟨30, 40⟩, ⟨0, 10⟩
]

/-- The side length of the large square -/
def squareSideLength : ℝ := 40

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- Calculate the area of the shaded region -/
def shadedArea : ℝ :=
  squareSideLength ^ 2 -
  (triangleArea ⟨10, 0⟩ ⟨40, 0⟩ ⟨40, 30⟩ +
   triangleArea ⟨0, 10⟩ ⟨30, 40⟩ ⟨0, 40⟩)

/-- Theorem: The area of the shaded region is 700 square units -/
theorem shaded_area_is_700 : shadedArea = 700 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_700_l2104_210466


namespace NUMINAMATH_CALUDE_odd_function_value_l2104_210411

def f (x : ℝ) : ℝ := sorry

theorem odd_function_value (a : ℝ) : 
  (∀ x : ℝ, f x = -f (-x)) → 
  (∀ x : ℝ, x ≥ 0 → f x = 3^x - 2*x + a) → 
  f (-2) = -4 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l2104_210411


namespace NUMINAMATH_CALUDE_sum_of_odds_15_to_45_l2104_210490

theorem sum_of_odds_15_to_45 :
  let a₁ : ℕ := 15  -- first term
  let aₙ : ℕ := 45  -- last term
  let d : ℕ := 2    -- common difference
  let n : ℕ := (aₙ - a₁) / d + 1  -- number of terms
  (n : ℚ) * (a₁ + aₙ) / 2 = 480 := by sorry

end NUMINAMATH_CALUDE_sum_of_odds_15_to_45_l2104_210490


namespace NUMINAMATH_CALUDE_balki_cereal_boxes_l2104_210487

theorem balki_cereal_boxes : 
  let total_raisins : ℕ := 437
  let box1_raisins : ℕ := 72
  let box2_raisins : ℕ := 74
  let other_boxes_raisins : ℕ := 97
  let num_other_boxes : ℕ := (total_raisins - box1_raisins - box2_raisins) / other_boxes_raisins
  let total_boxes : ℕ := 2 + num_other_boxes
  total_boxes = 5 ∧ 
  box1_raisins + box2_raisins + num_other_boxes * other_boxes_raisins = total_raisins :=
by sorry

end NUMINAMATH_CALUDE_balki_cereal_boxes_l2104_210487


namespace NUMINAMATH_CALUDE_number_division_problem_l2104_210453

theorem number_division_problem : ∃ N : ℕ,
  (N / (555 + 445) = 2 * (555 - 445)) ∧
  (N % (555 + 445) = 25) ∧
  (N = 220025) := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2104_210453


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_sin_2beta_zero_l2104_210445

theorem sin_2alpha_plus_sin_2beta_zero (α β : ℝ) 
  (h : Real.sin α * Real.sin β + Real.cos α * Real.cos β = 0) : 
  Real.sin (2 * α) + Real.sin (2 * β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_sin_2beta_zero_l2104_210445


namespace NUMINAMATH_CALUDE_bag_original_price_l2104_210469

theorem bag_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 135 → discount_percentage = 10 → 
  sale_price = (1 - discount_percentage / 100) * 150 := by
  sorry

end NUMINAMATH_CALUDE_bag_original_price_l2104_210469


namespace NUMINAMATH_CALUDE_dave_trips_l2104_210471

/-- The number of trays Dave can carry at a time -/
def trays_per_trip : ℕ := 12

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 26

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 49

/-- The number of trays on the third table -/
def trays_table3 : ℕ := 65

/-- The number of trays on the fourth table -/
def trays_table4 : ℕ := 38

/-- The total number of trays Dave needs to pick up -/
def total_trays : ℕ := trays_table1 + trays_table2 + trays_table3 + trays_table4

/-- The minimum number of trips Dave needs to make -/
def min_trips : ℕ := (total_trays + trays_per_trip - 1) / trays_per_trip

theorem dave_trips : min_trips = 15 := by
  sorry

end NUMINAMATH_CALUDE_dave_trips_l2104_210471


namespace NUMINAMATH_CALUDE_value_of_d_l2104_210412

theorem value_of_d (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 4 * d) : d = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l2104_210412


namespace NUMINAMATH_CALUDE_power_multiplication_l2104_210404

theorem power_multiplication (x : ℝ) : x^5 * x^9 = x^14 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l2104_210404


namespace NUMINAMATH_CALUDE_not_perfect_square_l2104_210420

-- Define the number with 300 ones followed by zeros
def number_with_300_ones : ℕ → ℕ 
  | n => 10^n * (10^300 - 1) / 9

-- Theorem statement
theorem not_perfect_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), number_with_300_ones n = m^2 := by
  sorry


end NUMINAMATH_CALUDE_not_perfect_square_l2104_210420


namespace NUMINAMATH_CALUDE_symmetric_angle_660_l2104_210454

def is_symmetric_angle (θ : ℤ) : Prop :=
  ∃ k : ℤ, θ = -60 + 360 * k

theorem symmetric_angle_660 :
  is_symmetric_angle 660 ∧
  ¬ is_symmetric_angle (-660) ∧
  ¬ is_symmetric_angle 690 ∧
  ¬ is_symmetric_angle (-690) :=
sorry

end NUMINAMATH_CALUDE_symmetric_angle_660_l2104_210454


namespace NUMINAMATH_CALUDE_pencils_per_row_l2104_210431

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 35)
  (h2 : num_rows = 7)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2104_210431


namespace NUMINAMATH_CALUDE_expression_value_l2104_210493

theorem expression_value : (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2104_210493


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l2104_210418

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for part (1)
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 9/4} :=
sorry

-- Theorem for part (2)
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, m^2 - m < f x} = {m : ℝ | -1 < m ∧ m < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l2104_210418


namespace NUMINAMATH_CALUDE_max_cos_sum_in_triangle_l2104_210448

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < angleA ∧ 0 < angleB ∧ 0 < angleC
  h_sum_angles : angleA + angleB + angleC = π
  h_cosine_law : b^2 + c^2 - a^2 = b * c

-- Theorem statement
theorem max_cos_sum_in_triangle (t : Triangle) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ (x : ℝ), x = Real.cos t.angleB + Real.cos t.angleC → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_cos_sum_in_triangle_l2104_210448


namespace NUMINAMATH_CALUDE_stable_performance_comparison_l2104_210419

/-- Represents a shooter's performance statistics -/
structure ShooterStats where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines the concept of stability based on variance -/
def more_stable (a b : ShooterStats) : Prop :=
  a.variance < b.variance

theorem stable_performance_comparison 
  (A B : ShooterStats)
  (h_avg : A.average_score = B.average_score)
  (h_var_A : A.variance = 0.4)
  (h_var_B : B.variance = 3.2) :
  more_stable A B :=
sorry

end NUMINAMATH_CALUDE_stable_performance_comparison_l2104_210419


namespace NUMINAMATH_CALUDE_dexter_cards_count_l2104_210437

/-- The number of boxes filled with basketball cards -/
def basketball_boxes : ℕ := 15

/-- The number of cards in each basketball box -/
def basketball_cards_per_box : ℕ := 20

/-- The difference in the number of boxes between basketball and football cards -/
def box_difference : ℕ := 7

/-- The number of cards in each football box -/
def football_cards_per_box : ℕ := 25

/-- The total number of cards Dexter has -/
def total_cards : ℕ := basketball_boxes * basketball_cards_per_box + 
  (basketball_boxes - box_difference) * football_cards_per_box

theorem dexter_cards_count : total_cards = 500 := by
  sorry

end NUMINAMATH_CALUDE_dexter_cards_count_l2104_210437


namespace NUMINAMATH_CALUDE_combined_mpg_l2104_210498

/-- Combined miles per gallon calculation -/
theorem combined_mpg (alice_mpg bob_mpg alice_miles bob_miles : ℚ) 
  (h1 : alice_mpg = 30)
  (h2 : bob_mpg = 20)
  (h3 : alice_miles = 120)
  (h4 : bob_miles = 180) :
  (alice_miles + bob_miles) / (alice_miles / alice_mpg + bob_miles / bob_mpg) = 300 / 13 := by
  sorry

#eval (120 + 180) / (120 / 30 + 180 / 20) -- For verification

end NUMINAMATH_CALUDE_combined_mpg_l2104_210498


namespace NUMINAMATH_CALUDE_largest_m_for_negative_integral_solutions_l2104_210440

def is_negative_integer (x : ℝ) : Prop := x < 0 ∧ ∃ n : ℤ, x = n

theorem largest_m_for_negative_integral_solutions :
  ∃ (m : ℝ),
    m = 570 ∧
    (∀ m' : ℝ, m' > m →
      ¬∃ (x y : ℝ),
        10 * x^2 - m' * x + 560 = 0 ∧
        10 * y^2 - m' * y + 560 = 0 ∧
        x ≠ y ∧
        is_negative_integer x ∧
        is_negative_integer y) ∧
    (∃ (x y : ℝ),
      10 * x^2 - m * x + 560 = 0 ∧
      10 * y^2 - m * y + 560 = 0 ∧
      x ≠ y ∧
      is_negative_integer x ∧
      is_negative_integer y) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_negative_integral_solutions_l2104_210440


namespace NUMINAMATH_CALUDE_sixth_test_score_l2104_210477

def average_score : ℝ := 84
def num_tests : ℕ := 6
def known_scores : List ℝ := [83, 77, 92, 85, 89]

theorem sixth_test_score :
  let total_sum := average_score * num_tests
  let sum_of_known_scores := known_scores.sum
  total_sum - sum_of_known_scores = 78 := by
  sorry

end NUMINAMATH_CALUDE_sixth_test_score_l2104_210477


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2104_210467

theorem imaginary_part_of_z (z : ℂ) (h : (z * (1 + Complex.I) * Complex.I^3) / (1 - Complex.I) = 1 - Complex.I) : 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2104_210467


namespace NUMINAMATH_CALUDE_function_characterization_l2104_210478

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

-- State the theorem
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesProperty f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l2104_210478


namespace NUMINAMATH_CALUDE_min_value_theorem_l2104_210442

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + a*b + a*c + b*c = 4) :
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 →
  2*a + b + c ≤ 2*x + y + z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2104_210442


namespace NUMINAMATH_CALUDE_negation_equivalence_l2104_210461

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2104_210461


namespace NUMINAMATH_CALUDE_prism_volume_l2104_210436

/-- Given a right rectangular prism with face areas 24 cm², 32 cm², and 48 cm², 
    its volume is 192 cm³. -/
theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2104_210436


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2104_210400

theorem ceiling_floor_difference : 
  ⌈(14 : ℚ) / 5 * (-31 : ℚ) / 3⌉ - ⌊(14 : ℚ) / 5 * ⌊(-31 : ℚ) / 3⌋⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2104_210400


namespace NUMINAMATH_CALUDE_probability_same_color_eq_l2104_210422

def total_marbles : ℕ := 5 + 4 + 6 + 3 + 2

def black_marbles : ℕ := 5
def red_marbles : ℕ := 4
def green_marbles : ℕ := 6
def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 2

def probability_same_color : ℚ :=
  (black_marbles * (black_marbles - 1) * (black_marbles - 2) * (black_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_eq : probability_same_color = 129 / 31250 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_eq_l2104_210422


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l2104_210410

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 2 * x + y → a + b ≤ x + y ∧ a + b = 2 * Real.sqrt 2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l2104_210410


namespace NUMINAMATH_CALUDE_third_chapter_pages_l2104_210441

/-- A book with three chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

/-- The theorem stating the number of pages in the third chapter -/
theorem third_chapter_pages (b : Book) 
  (h1 : b.chapter1 = 35)
  (h2 : b.chapter2 = 18)
  (h3 : b.chapter2 = b.chapter3 + 15) :
  b.chapter3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_chapter_pages_l2104_210441


namespace NUMINAMATH_CALUDE_minimize_y_l2104_210485

variable (a b c : ℝ)

def y (x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y a b c x_min ≤ y a b c x ∧ x_min = (a + b + c) / 3 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l2104_210485


namespace NUMINAMATH_CALUDE_honey_jar_theorem_l2104_210403

def initial_honey : ℝ := 1.2499999999999998
def draw_percentage : ℝ := 0.20
def num_iterations : ℕ := 4

def honey_left (initial : ℝ) (draw : ℝ) (iterations : ℕ) : ℝ :=
  initial * (1 - draw) ^ iterations

theorem honey_jar_theorem :
  honey_left initial_honey draw_percentage num_iterations = 0.512 := by
  sorry

end NUMINAMATH_CALUDE_honey_jar_theorem_l2104_210403


namespace NUMINAMATH_CALUDE_prob_queen_of_diamonds_l2104_210476

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- Represents a specific card -/
structure Card :=
  (rank : String)
  (suit : String)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4 }

/-- Probability of drawing a specific card from a deck -/
def prob_draw_card (d : Deck) (c : Card) : ℚ :=
  1 / d.total_cards

/-- Queen of Diamonds card -/
def queen_of_diamonds : Card :=
  { rank := "Queen",
    suit := "Diamonds" }

/-- Theorem: Probability of drawing Queen of Diamonds from a standard deck is 1/52 -/
theorem prob_queen_of_diamonds :
  prob_draw_card standard_deck queen_of_diamonds = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_queen_of_diamonds_l2104_210476


namespace NUMINAMATH_CALUDE_speed_difference_l2104_210424

theorem speed_difference (distance : ℝ) (emma_time lucas_time : ℝ) 
  (h1 : distance = 8)
  (h2 : emma_time = 12 / 60)
  (h3 : lucas_time = 40 / 60) :
  (distance / emma_time) - (distance / lucas_time) = 28 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l2104_210424


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2104_210450

theorem smallest_lcm_with_gcd_5 :
  ∃ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    Nat.gcd a b = 5 ∧
    Nat.lcm a b = 203010 ∧
    (∀ (c d : ℕ), 1000 ≤ c ∧ c < 10000 ∧ 1000 ≤ d ∧ d < 10000 ∧ Nat.gcd c d = 5 → 
      Nat.lcm c d ≥ 203010) :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2104_210450


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2104_210409

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 21*x^2 + 130*x - 210 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 21*s^2 + 130*s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 275 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2104_210409


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2104_210452

theorem right_triangle_inequality (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a_nonneg : a ≥ 0) (h_b_nonneg : b ≥ 0) (h_c_pos : c > 0) : 
  c ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2104_210452


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l2104_210443

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg correct_num incorrect_num : ℚ) : 
  n = 10 ∧ 
  initial_avg = 16 ∧ 
  correct_avg = 18 ∧ 
  correct_num = 46 ∧ 
  n * initial_avg + (correct_num - incorrect_num) = n * correct_avg → 
  incorrect_num = 26 := by
sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l2104_210443


namespace NUMINAMATH_CALUDE_midpoint_theorem_l2104_210488

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -4), 
    prove that the other endpoint is (-1, 6) -/
theorem midpoint_theorem :
  let midpoint : ℝ × ℝ := (3, 1)
  let endpoint1 : ℝ × ℝ := (7, -4)
  let endpoint2 : ℝ × ℝ := (-1, 6)
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_theorem_l2104_210488


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_three_l2104_210472

def G : ℕ → ℚ
  | 0 => 1
  | 1 => 4/3
  | (n + 2) => 3 * G (n + 1) - 2 * G n

theorem sum_of_reciprocal_G_powers_of_three : ∑' n, 1 / G (3^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_three_l2104_210472


namespace NUMINAMATH_CALUDE_E_opposite_Z_l2104_210451

/-- Represents a face of the cube -/
inductive Face : Type
| A : Face
| B : Face
| C : Face
| D : Face
| E : Face
| Z : Face

/-- Represents the net of the cube before folding -/
structure CubeNet :=
(faces : List Face)
(can_fold_to_cube : Bool)

/-- Represents the folded cube -/
structure Cube :=
(net : CubeNet)
(opposite_faces : Face → Face)

/-- The theorem stating that E is opposite to Z in the folded cube -/
theorem E_opposite_Z (net : CubeNet) (cube : Cube) :
  net.can_fold_to_cube = true →
  cube.net = net →
  cube.opposite_faces Face.Z = Face.E :=
sorry

end NUMINAMATH_CALUDE_E_opposite_Z_l2104_210451


namespace NUMINAMATH_CALUDE_f_simplification_f_third_quadrant_f_specific_angle_l2104_210438

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

theorem f_specific_angle : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_third_quadrant_f_specific_angle_l2104_210438


namespace NUMINAMATH_CALUDE_daily_earnings_of_c_l2104_210455

theorem daily_earnings_of_c (a b c : ℕ) 
  (h1 : a + b + c = 600)
  (h2 : a + c = 400)
  (h3 : b + c = 300) :
  c = 100 := by
sorry

end NUMINAMATH_CALUDE_daily_earnings_of_c_l2104_210455


namespace NUMINAMATH_CALUDE_nine_power_equation_solution_l2104_210460

theorem nine_power_equation_solution :
  ∃! n : ℝ, (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (81 : ℝ)^4 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_power_equation_solution_l2104_210460


namespace NUMINAMATH_CALUDE_alice_savings_difference_l2104_210494

def type_a_sales : ℝ := 1800
def type_b_sales : ℝ := 800
def type_c_sales : ℝ := 500
def basic_salary : ℝ := 500
def type_a_commission_rate : ℝ := 0.04
def type_b_commission_rate : ℝ := 0.06
def type_c_commission_rate : ℝ := 0.10
def monthly_expenses : ℝ := 600
def saving_goal : ℝ := 450
def usual_saving_rate : ℝ := 0.15

def total_commission : ℝ := 
  type_a_sales * type_a_commission_rate + 
  type_b_sales * type_b_commission_rate + 
  type_c_sales * type_c_commission_rate

def total_earnings : ℝ := basic_salary + total_commission

def net_earnings : ℝ := total_earnings - monthly_expenses

def actual_savings : ℝ := net_earnings * usual_saving_rate

theorem alice_savings_difference : saving_goal - actual_savings = 439.50 := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_difference_l2104_210494


namespace NUMINAMATH_CALUDE_characterize_S_l2104_210447

open Set
open Real

-- Define the set of points satisfying the condition
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = sin p.1 / |sin p.1|}

-- Define the set of x-values where sin(x) = 0
def ZeroSin : Set ℝ :=
  {x : ℝ | ∃ n : ℤ, x = n * π}

-- Define the set of x-values where y should be 1
def X1 : Set ℝ :=
  {x : ℝ | ∃ n : ℤ, x ∈ Ioo (π * (2 * n - 1)) (2 * n * π) \ ZeroSin}

-- Define the set of x-values where y should be -1
def X2 : Set ℝ :=
  {x : ℝ | ∃ n : ℤ, x ∈ Ioo (2 * n * π) (π * (2 * n + 1)) \ ZeroSin}

-- The main theorem
theorem characterize_S : ∀ p ∈ S, 
  (p.1 ∈ X1 ∧ p.2 = 1) ∨ (p.1 ∈ X2 ∧ p.2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_characterize_S_l2104_210447


namespace NUMINAMATH_CALUDE_ratio_arithmetic_properties_l2104_210401

-- Define a ratio arithmetic sequence
def is_ratio_arithmetic_seq (p : ℕ → ℝ) (k : ℝ) :=
  ∀ n ≥ 2, p (n + 1) / p n - p n / p (n - 1) = k

-- Define a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

-- Define an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℝ) (d : ℝ) :=
  ∀ n, b (n + 1) = b n + d

-- Define the Fibonacci-like sequence
def fib_like (a : ℕ → ℝ) :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 2, a (n + 1) = a n + a (n - 1)

theorem ratio_arithmetic_properties :
  (∀ a q, q ≠ 0 → is_geometric_seq a q → is_ratio_arithmetic_seq a 0) ∧
  (∃ b d, is_arithmetic_seq b d ∧ ∃ k, is_ratio_arithmetic_seq b k) ∧
  (∃ a b q d, is_arithmetic_seq a d ∧ is_geometric_seq b q ∧
    ¬∃ k, is_ratio_arithmetic_seq (fun n ↦ a n * b n) k) ∧
  (∀ a, fib_like a → ¬∃ k, is_ratio_arithmetic_seq a k) :=
sorry

end NUMINAMATH_CALUDE_ratio_arithmetic_properties_l2104_210401
