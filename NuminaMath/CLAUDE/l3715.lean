import Mathlib

namespace NUMINAMATH_CALUDE_passengers_ratio_l3715_371557

/-- Proves that the ratio of first class to second class passengers is 1:50 given the problem conditions -/
theorem passengers_ratio (fare_ratio : ℚ) (total_amount : ℕ) (second_class_amount : ℕ) :
  fare_ratio = 3 / 1 →
  total_amount = 1325 →
  second_class_amount = 1250 →
  ∃ (x y : ℕ), x ≠ 0 ∧ y ≠ 0 ∧ (x : ℚ) / y = 1 / 50 ∧
    fare_ratio * x * (second_class_amount : ℚ) / y = (total_amount - second_class_amount : ℚ) := by
  sorry

#check passengers_ratio

end NUMINAMATH_CALUDE_passengers_ratio_l3715_371557


namespace NUMINAMATH_CALUDE_student_average_less_than_true_average_l3715_371562

theorem student_average_less_than_true_average 
  (w x y z : ℝ) (hw : w < x) (hx : x < y) (hy : y < z) :
  (w + x + (y + z) / 2) / 3 < (w + x + y + z) / 4 := by
sorry

end NUMINAMATH_CALUDE_student_average_less_than_true_average_l3715_371562


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l3715_371554

theorem police_emergency_number_prime_divisor (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = 1000 * k + 133) : 
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l3715_371554


namespace NUMINAMATH_CALUDE_area_of_triangle_def_l3715_371599

/-- Triangle DEF with vertices D, E, and F -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The line on which point F lies -/
def line_equation (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 = 9

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (t : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the area of triangle DEF is 10 -/
theorem area_of_triangle_def :
  ∀ (t : Triangle),
    t.D = (4, 0) →
    t.E = (0, 4) →
    line_equation t.F →
    triangle_area t = 10 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_def_l3715_371599


namespace NUMINAMATH_CALUDE_sudoku_unique_solution_l3715_371540

def Sudoku := Fin 4 → Fin 4 → Fin 4

def valid_sudoku (s : Sudoku) : Prop :=
  (∀ i j₁ j₂, j₁ ≠ j₂ → s i j₁ ≠ s i j₂) ∧  -- rows
  (∀ i₁ i₂ j, i₁ ≠ i₂ → s i₁ j ≠ s i₂ j) ∧  -- columns
  (∀ b₁ b₂ c₁ c₂, (b₁ ≠ c₁ ∨ b₂ ≠ c₂) →     -- 2x2 subgrids
    s (2*b₁) (2*b₂) ≠ s (2*b₁+c₁) (2*b₂+c₂))

def initial_constraints (s : Sudoku) : Prop :=
  s 0 0 = 0 ∧  -- 3 in top-left (0-indexed)
  s 3 0 = 0 ∧  -- 1 in bottom-left
  s 2 2 = 1 ∧  -- 2 in third row, third column
  s 1 3 = 0    -- 1 in second row, fourth column

theorem sudoku_unique_solution (s : Sudoku) :
  valid_sudoku s ∧ initial_constraints s → s 0 1 = 1 := by sorry

end NUMINAMATH_CALUDE_sudoku_unique_solution_l3715_371540


namespace NUMINAMATH_CALUDE_problem_l3715_371535

theorem problem (a b : ℝ) (h : a - |b| > 0) : a^2 - b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_l3715_371535


namespace NUMINAMATH_CALUDE_temperature_data_inconsistency_l3715_371543

theorem temperature_data_inconsistency 
  (x_bar : ℝ) 
  (m : ℝ) 
  (S_squared : ℝ) 
  (h_x_bar : x_bar = 0) 
  (h_m : m = 4) 
  (h_S_squared : S_squared = 15.917) : 
  ¬(|x_bar - m| ≤ Real.sqrt S_squared) := by
  sorry

end NUMINAMATH_CALUDE_temperature_data_inconsistency_l3715_371543


namespace NUMINAMATH_CALUDE_red_faced_cubes_l3715_371544

theorem red_faced_cubes (n : ℕ) (h : n = 4) : 
  (n ^ 3) - (8 + 12 * (n - 2) + (n - 2) ^ 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_red_faced_cubes_l3715_371544


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3715_371555

/-- The total number of dogwood trees after planting operations -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating that the total number of trees after planting is 16 -/
theorem dogwood_tree_count :
  total_trees 7 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3715_371555


namespace NUMINAMATH_CALUDE_map_age_conversion_l3715_371536

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem map_age_conversion :
  octal_to_decimal 7324 = 2004 := by
  sorry

end NUMINAMATH_CALUDE_map_age_conversion_l3715_371536


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l3715_371592

/-- Represents a contestant's score for a single day -/
structure DailyScore where
  scored : ℚ
  attempted : ℚ

/-- Represents a contestant's scores for the three-day contest -/
structure ContestScore where
  day1 : DailyScore
  day2 : DailyScore
  day3 : DailyScore

def Charlie : ContestScore :=
  { day1 := { scored := 200, attempted := 300 },
    day2 := { scored := 160, attempted := 200 },
    day3 := { scored := 90, attempted := 100 } }

def totalAttempted (score : ContestScore) : ℚ :=
  score.day1.attempted + score.day2.attempted + score.day3.attempted

def totalScored (score : ContestScore) : ℚ :=
  score.day1.scored + score.day2.scored + score.day3.scored

def successRatio (score : ContestScore) : ℚ :=
  totalScored score / totalAttempted score

def dailySuccessRatio (day : DailyScore) : ℚ :=
  day.scored / day.attempted

theorem delta_max_success_ratio :
  ∀ delta : ContestScore,
    totalAttempted delta = 600 →
    dailySuccessRatio delta.day1 < dailySuccessRatio Charlie.day1 →
    dailySuccessRatio delta.day2 < dailySuccessRatio Charlie.day2 →
    dailySuccessRatio delta.day3 < dailySuccessRatio Charlie.day3 →
    delta.day1.attempted ≠ Charlie.day1.attempted →
    delta.day2.attempted ≠ Charlie.day2.attempted →
    delta.day3.attempted ≠ Charlie.day3.attempted →
    successRatio delta ≤ 399 / 600 :=
by sorry

#check delta_max_success_ratio

end NUMINAMATH_CALUDE_delta_max_success_ratio_l3715_371592


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l3715_371587

/-- Given two points P₁(x₁, y₁) and P₂(x₂, y₂) on the line y = -3x + 4,
    if x₁ < x₂, then y₁ > y₂ -/
theorem linear_function_point_relation (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -3 * x₁ + 4 →
  y₂ = -3 * x₂ + 4 →
  x₁ < x₂ →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_point_relation_l3715_371587


namespace NUMINAMATH_CALUDE_f_value_at_100_l3715_371523

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_value_at_100 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x * f (x + 3) = 12)
  (h2 : f 1 = 4) :
  f 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_100_l3715_371523


namespace NUMINAMATH_CALUDE_probability_yellow_or_green_l3715_371559

def yellow_marbles : ℕ := 4
def green_marbles : ℕ := 3
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 1

def total_marbles : ℕ := yellow_marbles + green_marbles + red_marbles + blue_marbles
def favorable_marbles : ℕ := yellow_marbles + green_marbles

theorem probability_yellow_or_green : 
  (favorable_marbles : ℚ) / total_marbles = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_or_green_l3715_371559


namespace NUMINAMATH_CALUDE_eleven_sided_polygon_diagonals_l3715_371501

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon has an obtuse angle --/
def has_obtuse_angle (p : ConvexPolygon n) : Prop := sorry

theorem eleven_sided_polygon_diagonals :
  ∀ (p : ConvexPolygon 11), has_obtuse_angle p → num_diagonals 11 = 44 := by
  sorry

end NUMINAMATH_CALUDE_eleven_sided_polygon_diagonals_l3715_371501


namespace NUMINAMATH_CALUDE_value_of_x_l3715_371517

theorem value_of_x (z y x : ℚ) : z = 48 → y = z / 4 → x = y / 3 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3715_371517


namespace NUMINAMATH_CALUDE_non_binary_listeners_l3715_371584

/-- Represents the survey data from StreamNow -/
structure StreamNowSurvey where
  total_listeners : ℕ
  male_listeners : ℕ
  female_non_listeners : ℕ
  non_binary_non_listeners : ℕ
  total_non_listeners : ℕ

/-- Theorem stating the number of non-binary listeners based on the survey data -/
theorem non_binary_listeners (survey : StreamNowSurvey) 
  (h1 : survey.total_listeners = 250)
  (h2 : survey.male_listeners = 85)
  (h3 : survey.female_non_listeners = 95)
  (h4 : survey.non_binary_non_listeners = 45)
  (h5 : survey.total_non_listeners = 230) :
  survey.total_listeners - survey.male_listeners - survey.female_non_listeners = 70 :=
by sorry

end NUMINAMATH_CALUDE_non_binary_listeners_l3715_371584


namespace NUMINAMATH_CALUDE_circle_tangent_line_radius_l3715_371526

/-- Given a circle and a line that are tangent, prove that the radius of the circle is 4. -/
theorem circle_tangent_line_radius (r : ℝ) (h1 : r > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ 3*x - 4*y + 20 = 0) →
  (∀ x y : ℝ, x^2 + y^2 ≤ r^2 → 3*x - 4*y + 20 ≥ 0) →
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ 3*x - 4*y + 20 = 0) →
  r = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_line_radius_l3715_371526


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3715_371594

def polynomial (x : ℝ) : ℝ := -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = 48 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3715_371594


namespace NUMINAMATH_CALUDE_graph_shift_l3715_371551

-- Define the functions f and g
def f (x : ℝ) : ℝ := (x + 3)^2 - 1
def g (x : ℝ) : ℝ := (x - 2)^2 + 3

-- State the theorem
theorem graph_shift : ∀ x : ℝ, f x = g (x - 5) + 4 := by sorry

end NUMINAMATH_CALUDE_graph_shift_l3715_371551


namespace NUMINAMATH_CALUDE_toys_per_day_l3715_371539

def toys_per_week : ℕ := 5500
def work_days_per_week : ℕ := 4

theorem toys_per_day (equal_daily_production : True) : 
  toys_per_week / work_days_per_week = 1375 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l3715_371539


namespace NUMINAMATH_CALUDE_batsman_average_l3715_371577

/-- Proves that given a batsman's average of 45 runs in 25 matches and an overall average of 38.4375 in 32 matches, the average runs scored in the last 7 matches is 15. -/
theorem batsman_average (first_25_avg : ℝ) (total_32_avg : ℝ) (first_25_matches : ℕ) (total_matches : ℕ) :
  first_25_avg = 45 →
  total_32_avg = 38.4375 →
  first_25_matches = 25 →
  total_matches = 32 →
  let last_7_matches := total_matches - first_25_matches
  let total_runs := total_32_avg * total_matches
  let first_25_runs := first_25_avg * first_25_matches
  let last_7_runs := total_runs - first_25_runs
  last_7_runs / last_7_matches = 15 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l3715_371577


namespace NUMINAMATH_CALUDE_f_min_value_l3715_371518

/-- The function f(x) defined as (x+1)(x+2)(x+3)(x+4) + 35 -/
def f (x : ℝ) : ℝ := (x+1)*(x+2)*(x+3)*(x+4) + 35

/-- Theorem stating that the minimum value of f(x) is 34 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 34 ∧ ∃ x₀ : ℝ, f x₀ = 34 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l3715_371518


namespace NUMINAMATH_CALUDE_limit_fraction_three_n_l3715_371537

/-- The limit of (3^n - 1) / (3^(n+1) + 1) as n approaches infinity is 1/3 -/
theorem limit_fraction_three_n (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((3^n - 1) / (3^(n+1) + 1)) - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_three_n_l3715_371537


namespace NUMINAMATH_CALUDE_unstable_products_selection_l3715_371583

theorem unstable_products_selection (n : ℕ) (d : ℕ) (k : ℕ) (h1 : n = 10) (h2 : d = 2) (h3 : k = 3) :
  (Nat.choose (n - d) 1 * d * Nat.choose (d - 1) 1) = 32 :=
sorry

end NUMINAMATH_CALUDE_unstable_products_selection_l3715_371583


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l3715_371590

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the y-value when x = -3 -/
def y₁ : ℝ := f (-3)

/-- y₂ is the y-value when x = 4 -/
def y₂ : ℝ := f 4

/-- Theorem: For the linear function f(x) = 2x + 1, y₁ < y₂ -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l3715_371590


namespace NUMINAMATH_CALUDE_infinite_fraction_value_l3715_371532

theorem infinite_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
sorry

end NUMINAMATH_CALUDE_infinite_fraction_value_l3715_371532


namespace NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l3715_371579

/-- The number of ways to arrange 4 boys and 2 girls in a row such that the 2 girls are not adjacent --/
def non_adjacent_arrangements : ℕ := 480

/-- The number of boys --/
def num_boys : ℕ := 4

/-- The number of girls --/
def num_girls : ℕ := 2

/-- The number of spaces available for girls (including ends) --/
def num_spaces : ℕ := num_boys + 1

theorem non_adjacent_arrangement_count :
  non_adjacent_arrangements = num_boys.factorial * (num_spaces.choose num_girls) := by
  sorry

end NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l3715_371579


namespace NUMINAMATH_CALUDE_infinite_slips_with_same_number_l3715_371533

-- Define a type for slip numbers
def SlipNumber : Type := ℕ

-- Define the set of all slips
def AllSlips : Set SlipNumber := Set.univ

-- Define the property that any infinite subset has at least two slips with the same number
def HasDuplicatesInInfiniteSubsets (S : Set SlipNumber) : Prop :=
  ∀ (T : Set SlipNumber), T ⊆ S → T.Infinite → ∃ (n : SlipNumber), (∃ (s t : SlipNumber), s ∈ T ∧ t ∈ T ∧ s ≠ t ∧ n = s ∧ n = t)

-- State the theorem
theorem infinite_slips_with_same_number :
  AllSlips.Infinite →
  HasDuplicatesInInfiniteSubsets AllSlips →
  ∃ (n : SlipNumber), {s : SlipNumber | s ∈ AllSlips ∧ n = s}.Infinite :=
by sorry

end NUMINAMATH_CALUDE_infinite_slips_with_same_number_l3715_371533


namespace NUMINAMATH_CALUDE_table_loss_percentage_l3715_371598

theorem table_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h1 : 15 * cost_price = 20 * selling_price) 
  (discount_rate : ℝ) (h2 : discount_rate = 0.1)
  (tax_rate : ℝ) (h3 : tax_rate = 0.08) : 
  (cost_price * (1 - discount_rate) - selling_price * (1 + tax_rate)) / cost_price = 0.09 := by
sorry

end NUMINAMATH_CALUDE_table_loss_percentage_l3715_371598


namespace NUMINAMATH_CALUDE_johns_hats_cost_l3715_371524

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_per_week * cost_per_hat

theorem johns_hats_cost :
  total_cost = 700 :=
by sorry

end NUMINAMATH_CALUDE_johns_hats_cost_l3715_371524


namespace NUMINAMATH_CALUDE_min_sum_values_l3715_371567

theorem min_sum_values (a b x y : ℝ) : 
  a > 0 → b > 0 → x > 0 → y > 0 →
  a + b = 10 →
  a / x + b / y = 1 →
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → a / x' + b / y' = 1 → x' + y' ≥ 16) →
  x + y = 16 →
  ((a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1)) := by
sorry

end NUMINAMATH_CALUDE_min_sum_values_l3715_371567


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l3715_371574

theorem fixed_point_of_linear_function (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x - k + 2
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l3715_371574


namespace NUMINAMATH_CALUDE_product_squares_relation_l3715_371588

theorem product_squares_relation (a b : ℝ) (h : a * b = 2 * (a^2 + b^2)) :
  2 * a * b - (a^2 + b^2) = a * b := by
  sorry

end NUMINAMATH_CALUDE_product_squares_relation_l3715_371588


namespace NUMINAMATH_CALUDE_middle_number_problem_l3715_371538

theorem middle_number_problem (x y z : ℤ) : 
  x < y ∧ y < z →
  x + y = 18 →
  x + z = 25 →
  y + z = 27 →
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l3715_371538


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l3715_371552

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x = 1) 
  (hy : y^3 - 3*y^2 + 5*y = 5) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l3715_371552


namespace NUMINAMATH_CALUDE_sarah_max_correct_l3715_371578

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam results. -/
structure ExamResult where
  exam : Exam
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_score : ℤ

/-- Checks if the exam result is valid according to the exam rules. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct + result.incorrect + result.unanswered = result.exam.total_questions ∧
  result.correct * result.exam.correct_score + result.incorrect * result.exam.incorrect_score = result.total_score

/-- The specific exam Sarah took. -/
def sarah_exam : Exam :=
  { total_questions := 25
  , correct_score := 4
  , incorrect_score := -3 }

/-- Sarah's exam result. -/
def sarah_result (correct : ℕ) : ExamResult :=
  { exam := sarah_exam
  , correct := correct
  , incorrect := (4 * correct - 40) / 3
  , unanswered := 25 - correct - (4 * correct - 40) / 3
  , total_score := 40 }

theorem sarah_max_correct :
  ∀ c : ℕ, c > 13 → ¬(is_valid_result (sarah_result c)) ∧
  is_valid_result (sarah_result 13) :=
sorry

end NUMINAMATH_CALUDE_sarah_max_correct_l3715_371578


namespace NUMINAMATH_CALUDE_bag_probability_l3715_371534

/-- Given a bag of 5 balls where the probability of picking a red ball is 0.4,
    prove that the probability of picking exactly one red ball and one white ball
    when two balls are picked is 3/5 -/
theorem bag_probability (total_balls : ℕ) (prob_red : ℝ) :
  total_balls = 5 →
  prob_red = 0.4 →
  (2 : ℝ) * prob_red * (1 - prob_red) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_bag_probability_l3715_371534


namespace NUMINAMATH_CALUDE_integral_nonnegative_function_integral_positive_at_point_l3715_371530

open MeasureTheory
open Measure
open Set
open Interval

theorem integral_nonnegative_function
  {a b : ℝ} (hab : a ≤ b)
  {f : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
  (hfnonneg : ∀ x ∈ Icc a b, f x ≥ 0) :
  ∫ x in a..b, f x ≥ 0 :=
sorry

theorem integral_positive_at_point
  {a b : ℝ} (hab : a ≤ b)
  {f : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
  (hfnonneg : ∀ x ∈ Icc a b, f x ≥ 0)
  (x₀ : ℝ) (hx₀ : x₀ ∈ Icc a b) (hfx₀ : f x₀ > 0) :
  ∫ x in a..b, f x > 0 :=
sorry

end NUMINAMATH_CALUDE_integral_nonnegative_function_integral_positive_at_point_l3715_371530


namespace NUMINAMATH_CALUDE_e_general_term_l3715_371581

/-- A sequence is a DQ sequence if it can be expressed as the sum of an arithmetic sequence
and a geometric sequence, both with positive integer terms. -/
def is_dq_sequence (e : ℕ → ℕ) : Prop :=
  ∃ (a b : ℕ → ℕ) (d q : ℕ),
    (∀ n, a n = a 1 + (n - 1) * d) ∧
    (∀ n, b n = b 1 * q^(n - 1)) ∧
    (∀ n, e n = a n + b n) ∧
    (∀ n, a n > 0 ∧ b n > 0)

/-- The sequence e_n satisfies the given conditions -/
def e_satisfies_conditions (e : ℕ → ℕ) : Prop :=
  is_dq_sequence e ∧
  e 1 = 3 ∧ e 2 = 6 ∧ e 3 = 11 ∧ e 4 = 20 ∧ e 5 = 37

theorem e_general_term (e : ℕ → ℕ) (h : e_satisfies_conditions e) :
  ∀ n : ℕ, e n = n + 2^n :=
by sorry

end NUMINAMATH_CALUDE_e_general_term_l3715_371581


namespace NUMINAMATH_CALUDE_benny_lunch_payment_l3715_371515

/-- The cost of a lunch special -/
def lunch_special_cost : ℕ := 8

/-- The number of people having lunch -/
def number_of_people : ℕ := 3

/-- The total cost Benny will pay for lunch -/
def total_cost : ℕ := number_of_people * lunch_special_cost

theorem benny_lunch_payment : total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_benny_lunch_payment_l3715_371515


namespace NUMINAMATH_CALUDE_triangle_side_length_l3715_371569

/-- In a triangle ABC, given side lengths a and c, and angle A, prove that side length b has a specific value. -/
theorem triangle_side_length (a c b : ℝ) (A : ℝ) : 
  a = 3 → c = Real.sqrt 3 → A = π / 3 → 
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A → 
  b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3715_371569


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3715_371506

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0 ∧ (∀ y : ℝ, y^2 - y + m = 0 → y = x)) → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3715_371506


namespace NUMINAMATH_CALUDE_point_distance_sum_l3715_371585

theorem point_distance_sum (a : ℝ) : 
  (2 * a < 0) →  -- P is in the second quadrant (x-coordinate negative)
  (1 - 3 * a > 0) →  -- P is in the second quadrant (y-coordinate positive)
  (abs (2 * a) + abs (1 - 3 * a) = 6) →  -- Sum of distances to axes is 6
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_point_distance_sum_l3715_371585


namespace NUMINAMATH_CALUDE_houses_visited_per_day_l3715_371531

-- Define the parameters
def buyerPercentage : Real := 0.2
def cheapKnivesPrice : Real := 50
def expensiveKnivesPrice : Real := 150
def weeklyRevenue : Real := 5000
def workDaysPerWeek : Nat := 5

-- Define the theorem
theorem houses_visited_per_day :
  ∃ (housesPerDay : Nat),
    (housesPerDay : Real) * buyerPercentage * 
    ((cheapKnivesPrice + expensiveKnivesPrice) / 2) * 
    (workDaysPerWeek : Real) = weeklyRevenue ∧
    housesPerDay = 50 := by
  sorry

end NUMINAMATH_CALUDE_houses_visited_per_day_l3715_371531


namespace NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l3715_371558

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem eleventh_term_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_1 : a 1 = 100) 
  (h_10 : a 10 = 10) : 
  a 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_of_arithmetic_sequence_l3715_371558


namespace NUMINAMATH_CALUDE_three_digit_numbers_theorem_l3715_371529

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

def satisfies_condition (n : ℕ) : Prop :=
  let (a, b, c) := digits n
  a ≠ 0 ∧ (26 ∣ (a^2 + b^2 + c^2))

def valid_numbers : Finset ℕ :=
  {100, 110, 101, 320, 302, 230, 203, 510, 501, 150, 105}

theorem three_digit_numbers_theorem :
  ∀ n : ℕ, is_three_digit n → (satisfies_condition n ↔ n ∈ valid_numbers) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_theorem_l3715_371529


namespace NUMINAMATH_CALUDE_october_birthdays_percentage_l3715_371573

theorem october_birthdays_percentage (total : ℕ) (october_births : ℕ) : 
  total = 120 → october_births = 18 → (october_births : ℚ) / total * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_october_birthdays_percentage_l3715_371573


namespace NUMINAMATH_CALUDE_probability_open_path_correct_l3715_371548

/-- The probability of being able to go from the first to the last floor using only open doors -/
def probability_open_path (n : ℕ) : ℚ :=
  (2 ^ (n - 1 : ℕ)) / (Nat.choose (2 * (n - 1)) (n - 1))

/-- Theorem stating the probability of an open path in a building with n floors -/
theorem probability_open_path_correct (n : ℕ) (h : n > 1) :
  probability_open_path n = (2 ^ (n - 1 : ℕ)) / (Nat.choose (2 * (n - 1)) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_open_path_correct_l3715_371548


namespace NUMINAMATH_CALUDE_expression_equality_l3715_371565

theorem expression_equality : 
  (84 + 4 / 19 : ℚ) * (1375 / 1000 : ℚ) + (105 + 5 / 19 : ℚ) * (9 / 10 : ℚ) = 210 + 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3715_371565


namespace NUMINAMATH_CALUDE_binary_111011_equals_59_l3715_371542

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of the number we're converting -/
def binary_111011 : List Bool := [true, true, true, false, true, true]

/-- Theorem stating that the decimal representation of 111011(2) is 59 -/
theorem binary_111011_equals_59 : binary_to_decimal binary_111011 = 59 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011_equals_59_l3715_371542


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3715_371545

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^4 + x + y^2 = 3*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3715_371545


namespace NUMINAMATH_CALUDE_regression_analysis_relationship_l3715_371563

/-- Represents a statistical relationship between two variables -/
inductive StatisticalRelationship
| Correlation

/-- Represents a method of statistical analysis -/
inductive StatisticalAnalysisMethod
| RegressionAnalysis

/-- The relationship between variables in regression analysis -/
def relationship_in_regression_analysis : StatisticalRelationship := StatisticalRelationship.Correlation

theorem regression_analysis_relationship :
  relationship_in_regression_analysis = StatisticalRelationship.Correlation := by
  sorry

end NUMINAMATH_CALUDE_regression_analysis_relationship_l3715_371563


namespace NUMINAMATH_CALUDE_paint_for_solar_system_l3715_371541

/-- Amount of paint available for the solar system given the usage by Mary, Mike, and Lucy --/
theorem paint_for_solar_system 
  (total_paint : ℝ) 
  (mary_paint : ℝ) 
  (mike_extra_paint : ℝ) 
  (lucy_paint : ℝ) 
  (h1 : total_paint = 25) 
  (h2 : mary_paint = 3) 
  (h3 : mike_extra_paint = 2) 
  (h4 : lucy_paint = 4) : 
  total_paint - (mary_paint + (mary_paint + mike_extra_paint) + lucy_paint) = 13 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_solar_system_l3715_371541


namespace NUMINAMATH_CALUDE_unique_prime_f_l3715_371519

/-- The polynomial function f(n) = n^3 - 7n^2 + 18n - 10 -/
def f (n : ℕ) : ℤ := n^3 - 7*n^2 + 18*n - 10

/-- Theorem stating that there exists exactly one positive integer n such that f(n) is prime -/
theorem unique_prime_f : ∃! (n : ℕ+), Nat.Prime (Int.natAbs (f n)) := by sorry

end NUMINAMATH_CALUDE_unique_prime_f_l3715_371519


namespace NUMINAMATH_CALUDE_exactly_one_root_l3715_371550

-- Define the function f(x) = -x^3 - x
def f (x : ℝ) : ℝ := -x^3 - x

-- State the theorem
theorem exactly_one_root (m n : ℝ) (h_interval : m ≤ n) (h_product : f m * f n < 0) :
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_root_l3715_371550


namespace NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l3715_371500

theorem sum_of_powers_implies_sum_power (a b : ℝ) :
  a^2009 + b^2009 = 0 → (a + b)^2009 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l3715_371500


namespace NUMINAMATH_CALUDE_puddle_depth_calculation_l3715_371566

/-- Represents the rainfall rate in centimeters per hour -/
def rainfall_rate : ℝ := 10

/-- Represents the duration of rainfall in hours -/
def rainfall_duration : ℝ := 3

/-- Represents the base area of the puddle in square centimeters -/
def puddle_base_area : ℝ := 300

/-- Calculates the depth of the puddle given the rainfall rate and duration -/
def puddle_depth : ℝ := rainfall_rate * rainfall_duration

theorem puddle_depth_calculation :
  puddle_depth = 30 := by sorry

end NUMINAMATH_CALUDE_puddle_depth_calculation_l3715_371566


namespace NUMINAMATH_CALUDE_max_value_of_f_l3715_371504

def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≤ m := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3715_371504


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3715_371527

theorem arithmetic_sequence_proof :
  ∃ (a b : ℕ), 
    a = 1477 ∧ 
    b = 2089 ∧ 
    a ≤ 2000 ∧ 
    2000 ≤ b ∧ 
    ∃ (d : ℕ), a * (a + 1) - 2 = d ∧ b * (b + 1) - a * (a + 1) = d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3715_371527


namespace NUMINAMATH_CALUDE_coffee_percentage_contribution_l3715_371502

def pancake_price : ℚ := 4
def bacon_price : ℚ := 2
def egg_price : ℚ := 3/2
def coffee_price : ℚ := 1

def pancake_sold : ℕ := 60
def bacon_sold : ℕ := 90
def egg_sold : ℕ := 75
def coffee_sold : ℕ := 50

def total_sales : ℚ := 
  pancake_price * pancake_sold + 
  bacon_price * bacon_sold + 
  egg_price * egg_sold + 
  coffee_price * coffee_sold

def coffee_contribution : ℚ := coffee_price * coffee_sold / total_sales

theorem coffee_percentage_contribution : 
  coffee_contribution * 100 = 858/100 := by sorry

end NUMINAMATH_CALUDE_coffee_percentage_contribution_l3715_371502


namespace NUMINAMATH_CALUDE_parallelogram_area_36_18_l3715_371564

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 36 cm and height 18 cm is 648 square centimeters -/
theorem parallelogram_area_36_18 : parallelogram_area 36 18 = 648 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_36_18_l3715_371564


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l3715_371570

theorem pure_imaginary_complex (a : ℝ) : 
  (a - (17 : ℂ) / (4 - Complex.I)).im = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l3715_371570


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3715_371510

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (12 * q) * Real.sqrt (8 * q^3) * Real.sqrt (9 * q^5) = 6 * q^4 * Real.sqrt (6 * q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3715_371510


namespace NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l3715_371514

theorem cube_volume_from_face_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 2) :
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 2 = d ∧ s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l3715_371514


namespace NUMINAMATH_CALUDE_max_value_3m_plus_4n_l3715_371560

theorem max_value_3m_plus_4n (m n : ℕ) (even_nums : Finset ℕ) (odd_nums : Finset ℕ) : 
  m = 15 →
  even_nums.card = m →
  odd_nums.card = n →
  (∀ x ∈ even_nums, x % 2 = 0 ∧ x > 0) →
  (∀ x ∈ odd_nums, x % 2 = 1 ∧ x > 0) →
  (even_nums.sum id + odd_nums.sum id = 1987) →
  (3 * m + 4 * n ≤ 221) :=
by sorry

end NUMINAMATH_CALUDE_max_value_3m_plus_4n_l3715_371560


namespace NUMINAMATH_CALUDE_existence_of_m_l3715_371507

theorem existence_of_m : ∃ m : ℝ, m ≤ 3 ∧ ∀ x : ℝ, |x - 1| ≤ m → -2 ≤ x ∧ x ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l3715_371507


namespace NUMINAMATH_CALUDE_problem_solution_l3715_371528

theorem problem_solution (x y z : ℕ+) 
  (h1 : x^2 + y^2 + z^2 = 2*(y*z + 1)) 
  (h2 : x + y + z = 4032) : 
  x^2 * y + z = 4031 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3715_371528


namespace NUMINAMATH_CALUDE_third_group_frequency_l3715_371553

/-- Given a sample of data distributed into groups, calculate the frequency of the unspecified group --/
theorem third_group_frequency 
  (total : ℕ) 
  (num_groups : ℕ) 
  (group1 : ℕ) 
  (group2 : ℕ) 
  (group4 : ℕ) 
  (h1 : total = 40) 
  (h2 : num_groups = 4) 
  (h3 : group1 = 5) 
  (h4 : group2 = 12) 
  (h5 : group4 = 8) : 
  total - (group1 + group2 + group4) = 15 := by
  sorry

#check third_group_frequency

end NUMINAMATH_CALUDE_third_group_frequency_l3715_371553


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3715_371549

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of possible 6-digit numbers -/
def total_numbers : ℕ := 9 * 10^(num_digits - 1)

/-- The number of 6-digit numbers with no zeros -/
def numbers_without_zero : ℕ := 9^num_digits

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero : 
  total_numbers - numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l3715_371549


namespace NUMINAMATH_CALUDE_exactly_two_out_of_four_l3715_371556

def probability_of_success : ℚ := 4/5

def number_of_trials : ℕ := 4

def number_of_successes : ℕ := 2

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exactly_two_out_of_four :
  binomial_probability number_of_trials number_of_successes probability_of_success = 96/625 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_four_l3715_371556


namespace NUMINAMATH_CALUDE_chocolate_box_weight_l3715_371595

/-- The weight of a box of chocolate bars in kilograms -/
def box_weight (bar_weight : ℕ) (num_bars : ℕ) : ℚ :=
  (bar_weight * num_bars : ℚ) / 1000

/-- Theorem: The weight of a box containing 16 chocolate bars, each weighing 125 grams, is 2 kilograms -/
theorem chocolate_box_weight :
  box_weight 125 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_weight_l3715_371595


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_X_l3715_371513

/-- Proves that the percentage of ryegrass in seed mixture X is 40% -/
theorem ryegrass_percentage_in_mixture_X : ∀ (x : ℝ),
  -- Seed mixture X has x% ryegrass and 60% bluegrass
  x + 60 = 100 →
  -- A mixture of 86.67% X and 13.33% Y contains 38% ryegrass
  0.8667 * x + 0.1333 * 25 = 38 →
  -- The percentage of ryegrass in seed mixture X is 40%
  x = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_X_l3715_371513


namespace NUMINAMATH_CALUDE_vector_magnitude_l3715_371589

theorem vector_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2)
  (h3 : ‖a - b‖ = Real.sqrt 3) :
  ‖a + b‖ = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3715_371589


namespace NUMINAMATH_CALUDE_age_difference_is_54_l3715_371525

/-- Represents a person's age with tens and units digits -/
structure Age where
  tens : Nat
  units : Nat
  tens_nonzero : tens ≠ 0

/-- The problem statement -/
theorem age_difference_is_54 
  (jack : Age) 
  (bill : Age) 
  (h1 : jack.tens * 10 + jack.units + 10 = 3 * (bill.tens * 10 + bill.units + 10))
  (h2 : jack.tens = bill.units ∧ jack.units = bill.tens) :
  (jack.tens * 10 + jack.units) - (bill.tens * 10 + bill.units) = 54 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_54_l3715_371525


namespace NUMINAMATH_CALUDE_infinitely_many_factorizable_numbers_l3715_371547

theorem infinitely_many_factorizable_numbers :
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧
    ∃ a b : ℕ, 
      (n^3 + 4*n + 505 : ℤ) = (a * b : ℤ) ∧
      a > n.sqrt ∧
      b > n.sqrt :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_factorizable_numbers_l3715_371547


namespace NUMINAMATH_CALUDE_complex_square_roots_l3715_371521

theorem complex_square_roots : 
  ∀ z : ℂ, z^2 = -99 - 40*I ↔ z = 2 - 10*I ∨ z = -2 + 10*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_roots_l3715_371521


namespace NUMINAMATH_CALUDE_map_distance_calculation_map_distance_proof_l3715_371561

theorem map_distance_calculation (scale_map : Real) (scale_actual : Real) (actual_distance : Real) : Real :=
  let scale_factor := scale_actual / scale_map
  let map_distance := actual_distance / scale_factor
  map_distance

theorem map_distance_proof (h1 : map_distance_calculation 0.4 5.3 848 = 64) : 
  ∃ (d : Real), map_distance_calculation 0.4 5.3 848 = d ∧ d = 64 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_map_distance_proof_l3715_371561


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l3715_371575

-- Define the quadratic function
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (∀ x ∈ Set.Ioo 2 5, quadratic a b x > 0) ∧ quadratic a b 1 = 1 →
  a ∈ Set.Ioi (3 - 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ a ∈ Set.Icc (-2) (-1), quadratic a (-a-1) x > 0) ∧ quadratic 0 (-1) 1 = 1 →
  x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4) :=
sorry

-- Part 3
theorem min_value_ratio (a b : ℝ) :
  b > 0 ∧ (∀ x : ℝ, quadratic a b x ≥ 0) →
  (a + 2) / b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_min_value_ratio_l3715_371575


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3715_371582

/-- The remainder when the sum of an arithmetic sequence is divided by 8 -/
theorem arithmetic_sequence_sum_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) : 
  a₁ = 3 → d = 6 → aₙ = 309 → n * (a₁ + aₙ) % 16 = 8 → 
  (n * (a₁ + aₙ) / 2) % 8 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l3715_371582


namespace NUMINAMATH_CALUDE_new_person_age_l3715_371580

theorem new_person_age (initial_group_size : ℕ) (age_decrease : ℕ) (replaced_person_age : ℕ) :
  initial_group_size = 10 →
  age_decrease = 3 →
  replaced_person_age = 42 →
  ∃ (new_person_age : ℕ),
    new_person_age = initial_group_size * age_decrease + replaced_person_age - initial_group_size * age_decrease :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l3715_371580


namespace NUMINAMATH_CALUDE_ticket_multiple_calculation_l3715_371522

/-- The multiple of fair tickets compared to baseball game tickets -/
def ticket_multiple (fair_tickets baseball_tickets : ℕ) : ℚ :=
  (fair_tickets - 6 : ℚ) / baseball_tickets

theorem ticket_multiple_calculation (fair_tickets baseball_tickets : ℕ) 
  (h1 : fair_tickets = ticket_multiple fair_tickets baseball_tickets * baseball_tickets + 6)
  (h2 : fair_tickets = 25)
  (h3 : baseball_tickets = 56) :
  ticket_multiple fair_tickets baseball_tickets = 19 / 56 := by
  sorry

#eval ticket_multiple 25 56

end NUMINAMATH_CALUDE_ticket_multiple_calculation_l3715_371522


namespace NUMINAMATH_CALUDE_cookie_difference_l3715_371597

/-- The number of chocolate chip cookies Helen baked yesterday -/
def helen_choc_yesterday : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def helen_raisin_today : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def helen_choc_today : ℕ := 237

/-- The number of oatmeal cookies Helen baked this morning -/
def helen_oatmeal_today : ℕ := 107

/-- The number of chocolate chip cookies Giselle baked -/
def giselle_choc : ℕ := 156

/-- The number of raisin cookies Giselle baked -/
def giselle_raisin : ℕ := 89

/-- The number of chocolate chip cookies Timmy baked -/
def timmy_choc : ℕ := 135

/-- The number of oatmeal cookies Timmy baked -/
def timmy_oatmeal : ℕ := 246

theorem cookie_difference : 
  (helen_choc_yesterday + helen_choc_today + giselle_choc + timmy_choc) - 
  (helen_raisin_today + giselle_raisin) = 227 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l3715_371597


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3715_371505

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Symmetry about the origin -/
def symmetricAboutOrigin (c1 c2 : Circle) : Prop :=
  c2.center = (-c1.center.1, -c1.center.2) ∧ c2.radius = c1.radius

/-- The main theorem -/
theorem symmetric_circle_equation (c1 c2 : Circle) :
  c1.equation = λ x y => (x + 2)^2 + y^2 = 5 →
  symmetricAboutOrigin c1 c2 →
  c2.equation = λ x y => (x - 2)^2 + y^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3715_371505


namespace NUMINAMATH_CALUDE_percentage_problem_l3715_371568

theorem percentage_problem (P : ℝ) : P = 20 → (P / 100) * 680 = 0.4 * 140 + 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3715_371568


namespace NUMINAMATH_CALUDE_probability_prime_sum_three_dice_l3715_371509

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The set of possible prime sums when rolling three 6-sided dice -/
def primeSums : Set ℕ := {3, 5, 7, 11, 13, 17}

/-- The total number of possible outcomes when rolling three 6-sided dice -/
def totalOutcomes : ℕ := numSides ^ 3

/-- The number of ways to roll a prime sum with three 6-sided dice -/
def primeOutcomes : ℕ := 58

/-- The probability of rolling a prime sum with three 6-sided dice -/
theorem probability_prime_sum_three_dice :
  (primeOutcomes : ℚ) / totalOutcomes = 58 / 216 := by
  sorry


end NUMINAMATH_CALUDE_probability_prime_sum_three_dice_l3715_371509


namespace NUMINAMATH_CALUDE_polynomial_equality_l3715_371512

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) →
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3715_371512


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3715_371586

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a + b ≠ 3) → ¬(a ≠ 1 ∨ b ≠ 2)) ↔ (a = 1 ∧ b = 2 → a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3715_371586


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3715_371516

theorem two_numbers_difference (a b : ℝ) 
  (sum_condition : a + b = 9)
  (square_difference_condition : a^2 - b^2 = 45) :
  |a - b| = 5 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3715_371516


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_equals_result_l3715_371520

-- Define the universe set U
def U : Set ℤ := {x : ℤ | x^2 - x - 12 ≤ 0}

-- Define set A
def A : Set ℤ := {-2, -1, 3}

-- Define set B
def B : Set ℤ := {0, 1, 3, 4}

-- Define the result set
def result : Set ℤ := {0, 1, 4}

-- Theorem statement
theorem complement_A_intersect_B_equals_result :
  (U \ A) ∩ B = result := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_equals_result_l3715_371520


namespace NUMINAMATH_CALUDE_arithmetic_mean_inequality_and_minimum_t_l3715_371576

theorem arithmetic_mean_inequality_and_minimum_t :
  (∀ a b c : ℝ, (((a + b + c) / 3) ^ 2 ≤ (a ^ 2 + b ^ 2 + c ^ 2) / 3) ∧
    (((a + b + c) / 3) ^ 2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3 ↔ a = b ∧ b = c)) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (∀ t : ℝ, Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z) →
      t ≥ Real.sqrt 3) ∧
    ∃ t : ℝ, t = Real.sqrt 3 ∧
      Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_inequality_and_minimum_t_l3715_371576


namespace NUMINAMATH_CALUDE_no_consecutive_sum_32_l3715_371571

theorem no_consecutive_sum_32 : ¬∃ (n k : ℕ), n > 0 ∧ (n * (2 * k + n - 1)) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_sum_32_l3715_371571


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l3715_371503

/-- The total number of houses in Lincoln County after the housing boom -/
def total_houses (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem: The total number of houses in Lincoln County after the housing boom is 118558 -/
theorem lincoln_county_houses :
  total_houses 20817 97741 = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l3715_371503


namespace NUMINAMATH_CALUDE_max_point_inequality_l3715_371591

noncomputable section

variables (a : ℝ) (x₁ : ℝ)

def f (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (x : ℝ) : ℝ := f a x + (1/2) * x^2

theorem max_point_inequality (h1 : x₁ > 0) (h2 : IsLocalMax (g a) x₁) :
  (Real.log x₁) / x₁ + 1 / x₁^2 > a :=
sorry

end

end NUMINAMATH_CALUDE_max_point_inequality_l3715_371591


namespace NUMINAMATH_CALUDE_base6_243_equals_base10_99_l3715_371511

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem base6_243_equals_base10_99 :
  base6ToBase10 [3, 4, 2] = 99 := by
  sorry

end NUMINAMATH_CALUDE_base6_243_equals_base10_99_l3715_371511


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l3715_371593

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 3) ∧              -- congruent to 3 modulo 17
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 3 → m ≥ n) ∧  -- smallest such integer
  n = 10012 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l3715_371593


namespace NUMINAMATH_CALUDE_equation_equivalence_l3715_371508

theorem equation_equivalence (x y : ℝ) : 
  (5 * x + y = 1) ↔ (y = 1 - 5 * x) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3715_371508


namespace NUMINAMATH_CALUDE_sum_of_digits_cd_l3715_371596

/-- c is an integer made up of a sequence of 2023 sixes -/
def c : ℕ := (6 : ℕ) * ((10 ^ 2023 - 1) / 9)

/-- d is an integer made up of a sequence of 2023 ones -/
def d : ℕ := (10 ^ 2023 - 1) / 9

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits in cd is 12133 -/
theorem sum_of_digits_cd : sum_of_digits (c * d) = 12133 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_cd_l3715_371596


namespace NUMINAMATH_CALUDE_soda_preference_result_l3715_371572

/-- The number of people who prefer calling soft drinks "Soda" in a survey. -/
def soda_preference (total_surveyed : ℕ) (central_angle : ℕ) : ℕ :=
  (total_surveyed * central_angle) / 360

/-- Theorem stating that 330 people prefer calling soft drinks "Soda" in the given survey. -/
theorem soda_preference_result : soda_preference 600 198 = 330 := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_result_l3715_371572


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l3715_371546

/-- Proves that given the conditions of the coconut grove problem, 
    when x = 8, the yield Y of each (x - 4) tree is 180 nuts per year. -/
theorem coconut_grove_yield (x : ℕ) (Y : ℕ) : 
  x = 8 →
  ((x + 4) * 60 + x * 120 + (x - 4) * Y) / (3 * x) = 100 →
  Y = 180 := by
  sorry

#check coconut_grove_yield

end NUMINAMATH_CALUDE_coconut_grove_yield_l3715_371546
