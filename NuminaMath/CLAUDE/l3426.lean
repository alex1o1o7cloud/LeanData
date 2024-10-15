import Mathlib

namespace NUMINAMATH_CALUDE_triangle_problem_l3426_342628

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : 2 * abc.b * Real.cos abc.A = abc.c * Real.cos abc.A + abc.a * Real.cos abc.C)
  (h2 : abc.a = Real.sqrt 7)
  (h3 : abc.b + abc.c = 4) :
  abc.A = π / 3 ∧ abc.b * abc.c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3426_342628


namespace NUMINAMATH_CALUDE_lcm_problem_l3426_342690

theorem lcm_problem (m n : ℕ+) :
  m - n = 189 →
  Nat.lcm m n = 133866 →
  m = 22311 ∧ n = 22122 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3426_342690


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3426_342691

theorem root_sum_theorem (a b c : ℝ) : 
  (a * b * c = -22) → 
  (a + b + c = 20) → 
  (a * b + b * c + c * a = 0) → 
  (b * c / a^2 + a * c / b^2 + a * b / c^2 = 3) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3426_342691


namespace NUMINAMATH_CALUDE_cherry_tart_fraction_l3426_342616

theorem cherry_tart_fraction (total : ℝ) (blueberry : ℝ) (peach : ℝ) 
  (h1 : total = 0.91)
  (h2 : blueberry = 0.75)
  (h3 : peach = 0.08)
  (h4 : ∃ cherry : ℝ, cherry + blueberry + peach = total) :
  ∃ cherry : ℝ, cherry = 0.08 ∧ cherry + blueberry + peach = total := by
sorry

end NUMINAMATH_CALUDE_cherry_tart_fraction_l3426_342616


namespace NUMINAMATH_CALUDE_george_total_blocks_l3426_342613

/-- The number of boxes George has -/
def num_boxes : ℕ := 2

/-- The number of blocks in each box -/
def blocks_per_box : ℕ := 6

/-- Theorem stating the total number of blocks George has -/
theorem george_total_blocks : num_boxes * blocks_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_george_total_blocks_l3426_342613


namespace NUMINAMATH_CALUDE_log_equation_solution_l3426_342660

-- Define the logarithm function with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_equation_solution :
  ∃! x : ℝ, x > 1 ∧ log_one_third (x^2 + 3*x - 4) = log_one_third (2*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3426_342660


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3426_342630

theorem quadratic_one_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 30 * x + 12 = 0) :
  ∃ x, a * x^2 + 30 * x + 12 = 0 ∧ x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3426_342630


namespace NUMINAMATH_CALUDE_laborer_income_l3426_342640

theorem laborer_income (
  avg_expenditure_6months : ℝ)
  (fell_into_debt : Prop)
  (reduced_expenses_4months : ℝ)
  (debt_cleared_and_saved : ℝ) :
  avg_expenditure_6months = 85 →
  fell_into_debt →
  reduced_expenses_4months = 60 →
  debt_cleared_and_saved = 30 →
  ∃ (monthly_income : ℝ), monthly_income = 78 :=
by sorry

end NUMINAMATH_CALUDE_laborer_income_l3426_342640


namespace NUMINAMATH_CALUDE_bird_migration_difference_l3426_342676

/-- The number of bird families that flew to Asia is greater than the number
    of bird families that flew to Africa by 47. -/
theorem bird_migration_difference :
  let mountain_families : ℕ := 38
  let africa_families : ℕ := 47
  let asia_families : ℕ := 94
  asia_families - africa_families = 47 := by sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l3426_342676


namespace NUMINAMATH_CALUDE_adam_remaining_candy_l3426_342602

/-- The number of boxes of chocolate candy Adam initially bought -/
def initial_boxes : ℕ := 13

/-- The number of boxes Adam gave to his little brother -/
def given_boxes : ℕ := 7

/-- The number of pieces of candy in each box -/
def pieces_per_box : ℕ := 6

/-- Theorem: Adam still had 36 pieces of chocolate candy -/
theorem adam_remaining_candy : 
  (initial_boxes - given_boxes) * pieces_per_box = 36 := by
  sorry

end NUMINAMATH_CALUDE_adam_remaining_candy_l3426_342602


namespace NUMINAMATH_CALUDE_eleven_pow_2023_mod_50_l3426_342609

theorem eleven_pow_2023_mod_50 : 11^2023 % 50 = 31 := by
  sorry

end NUMINAMATH_CALUDE_eleven_pow_2023_mod_50_l3426_342609


namespace NUMINAMATH_CALUDE_exercise_time_is_9_25_hours_l3426_342646

/-- Represents the exercise schedule for a week -/
structure ExerciseSchedule where
  initial_jogging : ℕ
  jogging_increment : ℕ
  swimming_increment : ℕ
  wednesday_reduction : ℕ
  friday_kickboxing : ℕ
  kickboxing_multiplier : ℕ

/-- Calculates the total exercise time for the week -/
def total_exercise_time (schedule : ExerciseSchedule) : ℚ :=
  sorry

/-- Theorem stating that the total exercise time is 9.25 hours -/
theorem exercise_time_is_9_25_hours (schedule : ExerciseSchedule) 
  (h1 : schedule.initial_jogging = 30)
  (h2 : schedule.jogging_increment = 5)
  (h3 : schedule.swimming_increment = 10)
  (h4 : schedule.wednesday_reduction = 10)
  (h5 : schedule.friday_kickboxing = 20)
  (h6 : schedule.kickboxing_multiplier = 2) :
  total_exercise_time schedule = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_exercise_time_is_9_25_hours_l3426_342646


namespace NUMINAMATH_CALUDE_union_of_sets_l3426_342659

theorem union_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-1, 1}
  A ∪ B = {-1, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3426_342659


namespace NUMINAMATH_CALUDE_circle_circumference_bounds_l3426_342699

/-- The circumference of a circle with diameter 1 is between 3 and 4 -/
theorem circle_circumference_bounds :
  ∀ C : ℝ, C = π * 1 → 3 < C ∧ C < 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_bounds_l3426_342699


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3426_342693

/-- Calculates the percentage profit of a retailer given the wholesale price, retail price, and discount percentage. -/
theorem retailer_profit_percentage 
  (wholesale_price retail_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : wholesale_price = 90) 
  (h2 : retail_price = 120) 
  (h3 : discount_percentage = 0.1) :
  let selling_price := retail_price * (1 - discount_percentage)
  let profit := selling_price - wholesale_price
  let profit_percentage := (profit / wholesale_price) * 100
  profit_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3426_342693


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3426_342626

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, -2; 1, 1] →
  (B^3)⁻¹ = !![13, -22; 11, -9] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3426_342626


namespace NUMINAMATH_CALUDE_floor_length_calculation_l3426_342606

theorem floor_length_calculation (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 80 →
  length = 3 * Real.sqrt (80 / 3) := by
sorry

end NUMINAMATH_CALUDE_floor_length_calculation_l3426_342606


namespace NUMINAMATH_CALUDE_modular_inverse_three_mod_187_l3426_342651

theorem modular_inverse_three_mod_187 :
  ∃ x : ℕ, x < 187 ∧ (3 * x) % 187 = 1 :=
by
  use 125
  sorry

end NUMINAMATH_CALUDE_modular_inverse_three_mod_187_l3426_342651


namespace NUMINAMATH_CALUDE_function_properties_l3426_342648

/-- A function type that represents the relationship between x and y --/
def Function := ℝ → ℝ

/-- The given values in the table --/
structure TableValues where
  y_neg5 : ℝ
  y_neg2 : ℝ
  y_2 : ℝ
  y_5 : ℝ

/-- Proposition: If y is an inverse proportion function of x, then 2m + 5n = 0 --/
def inverse_proportion_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ k : ℝ, ∀ x : ℝ, f x * x = k) →
  2 * tv.y_neg2 + 5 * tv.y_5 = 0

/-- Proposition: If y is a linear function of x, then n - m = 7 --/
def linear_function_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) →
  tv.y_5 - tv.y_neg2 = 7

/-- Proposition: If y is a quadratic function of x and the graph opens downwards, 
    then m > n is not necessarily true --/
def quadratic_function_prop (f : Function) (tv : TableValues) : Prop :=
  (∃ a b c : ℝ, a < 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c) →
  ¬(tv.y_neg2 > tv.y_5)

/-- The main theorem that combines all three propositions --/
theorem function_properties (f : Function) (tv : TableValues) : 
  inverse_proportion_prop f tv ∧ 
  linear_function_prop f tv ∧ 
  quadratic_function_prop f tv := by sorry

end NUMINAMATH_CALUDE_function_properties_l3426_342648


namespace NUMINAMATH_CALUDE_square_pentagon_angle_sum_l3426_342682

/-- In a figure composed of a square and a regular pentagon, the sum of angles a° and b° is 324°. -/
theorem square_pentagon_angle_sum (a b : ℝ) : 
  -- The figure is composed of a square and a regular pentagon
  -- a and b are angles in degrees as shown in the diagram
  a + b = 324 := by sorry

end NUMINAMATH_CALUDE_square_pentagon_angle_sum_l3426_342682


namespace NUMINAMATH_CALUDE_third_task_end_time_l3426_342664

-- Define the start time of the first task
def start_time : Nat := 13 * 60  -- 1:00 PM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 14 * 60 + 40  -- 2:40 PM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 3

-- Theorem statement
theorem third_task_end_time :
  let task_duration := (end_second_task - start_time) / 2
  let end_third_task := end_second_task + task_duration
  end_third_task = 15 * 60 + 30  -- 3:30 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_third_task_end_time_l3426_342664


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3426_342675

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h : ArithmeticSequence a) (h2 : a 2 + a 12 = 32) :
  a 3 + a 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3426_342675


namespace NUMINAMATH_CALUDE_function_symmetry_l3426_342674

-- Define the function f
variable (f : ℝ → ℝ)
-- Define the constant a
variable (a : ℝ)

-- State the theorem
theorem function_symmetry 
  (h : ∀ x : ℝ, f (a - x) = -f (a + x)) : 
  ∀ x : ℝ, f (2 * a - x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3426_342674


namespace NUMINAMATH_CALUDE_product_correction_l3426_342679

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (p q : Nat) :
  p ≥ 10 ∧ p < 100 →  -- p is a two-digit number
  q > 0 →  -- q is positive
  reverseDigits p * q = 221 →
  p * q = 923 := by
sorry

end NUMINAMATH_CALUDE_product_correction_l3426_342679


namespace NUMINAMATH_CALUDE_french_students_count_l3426_342633

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 69)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 15) :
  ∃ french : ℕ, french = total - german + both - neither :=
by
  sorry

end NUMINAMATH_CALUDE_french_students_count_l3426_342633


namespace NUMINAMATH_CALUDE_lowest_price_option2_l3426_342642

def initial_amount : ℝ := 12000

def option1_price : ℝ := initial_amount * (1 - 0.15) * (1 - 0.10) * (1 - 0.05)

def option2_price : ℝ := initial_amount * (1 - 0.25) * (1 - 0.05)

def option3_price : ℝ := initial_amount * (1 - 0.20) - 500

theorem lowest_price_option2 :
  option2_price < option1_price ∧ option2_price < option3_price :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_option2_l3426_342642


namespace NUMINAMATH_CALUDE_books_movies_difference_l3426_342635

/-- The number of books in the "crazy silly school" series -/
def num_books : ℕ := 36

/-- The number of movies in the "crazy silly school" series -/
def num_movies : ℕ := 25

/-- The number of books read -/
def books_read : ℕ := 17

/-- The number of movies watched -/
def movies_watched : ℕ := 13

/-- Theorem stating the difference between the number of books and movies -/
theorem books_movies_difference : num_books - num_movies = 11 := by
  sorry

end NUMINAMATH_CALUDE_books_movies_difference_l3426_342635


namespace NUMINAMATH_CALUDE_min_value_expression_l3426_342681

theorem min_value_expression (x : ℝ) (h : x > 0) : 6 * x + 1 / x^6 ≥ 7 ∧ (6 * x + 1 / x^6 = 7 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3426_342681


namespace NUMINAMATH_CALUDE_total_insect_legs_l3426_342643

/-- The number of insects in the laboratory -/
def num_insects : ℕ := 6

/-- The number of legs each insect has -/
def legs_per_insect : ℕ := 6

/-- The total number of legs for all insects in the laboratory -/
def total_legs : ℕ := num_insects * legs_per_insect

/-- Theorem stating that the total number of legs is 36 -/
theorem total_insect_legs : total_legs = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_insect_legs_l3426_342643


namespace NUMINAMATH_CALUDE_sample_size_is_100_l3426_342639

/-- A structure representing a statistical sampling process -/
structure SamplingProcess where
  totalStudents : Nat
  selectedStudents : Nat

/-- Definition of sample size for a SamplingProcess -/
def sampleSize (sp : SamplingProcess) : Nat := sp.selectedStudents

/-- Theorem stating that for the given sampling process, the sample size is 100 -/
theorem sample_size_is_100 (sp : SamplingProcess) 
  (h1 : sp.totalStudents = 1000) 
  (h2 : sp.selectedStudents = 100) : 
  sampleSize sp = 100 := by
  sorry

#check sample_size_is_100

end NUMINAMATH_CALUDE_sample_size_is_100_l3426_342639


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_conditions_l3426_342624

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- The asymptotic lines of a hyperbola -/
def asymptotic_lines (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ y = -(3/4) * x

/-- The focus of a hyperbola -/
def focus (h : Hyperbola) : ℝ × ℝ := (0, 5)

/-- The main theorem -/
theorem hyperbola_equation_from_conditions (h : Hyperbola) :
  (∀ x y, asymptotic_lines h x y ↔ y = (3/4) * x ∨ y = -(3/4) * x) →
  focus h = (0, 5) →
  ∀ x y, hyperbola_equation h x y ↔ y^2 / 9 - x^2 / 16 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_conditions_l3426_342624


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3426_342637

/-- Profit percentage calculation for Company N --/
theorem profit_percentage_calculation (R : ℝ) (P : ℝ) :
  R > 0 ∧ P > 0 →  -- Assuming positive revenue and profit
  (0.8 * R) * 0.14 = 0.112 * R →  -- 1999 profit calculation
  0.112 * R = 1.1200000000000001 * P →  -- Profit comparison between years
  P / R * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_profit_percentage_calculation_l3426_342637


namespace NUMINAMATH_CALUDE_solution_set_min_value_l3426_342612

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2*x - 5

-- Statement for the solution set
theorem solution_set : 
  {x : ℝ | |f x| + |g x| ≤ 2} = {x : ℝ | 5/3 ≤ x ∧ x ≤ 3} := by sorry

-- Statement for the minimum value
theorem min_value : 
  ∀ x : ℝ, |f (2*x)| + |g x| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_l3426_342612


namespace NUMINAMATH_CALUDE_square_side_increase_l3426_342634

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.1025 → p = 5 := by
sorry

end NUMINAMATH_CALUDE_square_side_increase_l3426_342634


namespace NUMINAMATH_CALUDE_total_amount_paid_prove_total_amount_l3426_342666

/-- Calculate the total amount paid for grapes and mangoes -/
theorem total_amount_paid 
  (grape_quantity : ℕ) (grape_price : ℕ) 
  (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price

/-- Prove that the total amount paid for the given quantities and prices is 1135 -/
theorem prove_total_amount : 
  total_amount_paid 8 80 9 55 = 1135 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_prove_total_amount_l3426_342666


namespace NUMINAMATH_CALUDE_karen_tom_race_l3426_342698

/-- Karen's race against Tom -/
theorem karen_tom_race (karen_speed : ℝ) (karen_delay : ℝ) (lead_distance : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 4 / 60 →
  lead_distance = 4 →
  tom_distance = 24 →
  ∃ (tom_speed : ℝ), tom_speed = 45 ∧ 
    karen_speed * (tom_distance / karen_speed + lead_distance / karen_speed) = 
    tom_speed * (tom_distance / karen_speed + lead_distance / karen_speed + karen_delay) :=
by sorry

end NUMINAMATH_CALUDE_karen_tom_race_l3426_342698


namespace NUMINAMATH_CALUDE_correct_distribution_l3426_342641

/-- Represents the amount of coins each person receives -/
structure CoinDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  e : ℚ

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (dist : CoinDistribution) : Prop :=
  -- The total amount is 5 coins
  dist.a + dist.b + dist.c + dist.d + dist.e = 5 ∧
  -- The difference between each person is equal
  (dist.b - dist.a = dist.c - dist.b) ∧
  (dist.c - dist.b = dist.d - dist.c) ∧
  (dist.d - dist.c = dist.e - dist.d) ∧
  -- The total amount received by A and B equals that received by C, D, and E
  dist.a + dist.b = dist.c + dist.d + dist.e

/-- The theorem stating the correct distribution -/
theorem correct_distribution :
  ∃ (dist : CoinDistribution),
    isValidDistribution dist ∧
    dist.a = 2/3 ∧
    dist.b = 5/6 ∧
    dist.c = 1 ∧
    dist.d = 7/6 ∧
    dist.e = 4/3 :=
  sorry

end NUMINAMATH_CALUDE_correct_distribution_l3426_342641


namespace NUMINAMATH_CALUDE_binomial_square_value_l3426_342663

theorem binomial_square_value (a : ℚ) : 
  (∃ p q : ℚ, ∀ x, 9*x^2 + 27*x + a = (p*x + q)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_value_l3426_342663


namespace NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l3426_342629

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rectangle in a plane --/
structure Rectangle where
  corners : Fin 4 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment --/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The number of intersection points between a circle and a rectangle --/
def intersectionPointsCircleRectangle (c : Circle) (r : Rectangle) : ℕ :=
  (intersectionPointsCircleLine c (r.corners 0) (r.corners 1)) +
  (intersectionPointsCircleLine c (r.corners 1) (r.corners 2)) +
  (intersectionPointsCircleLine c (r.corners 2) (r.corners 3)) +
  (intersectionPointsCircleLine c (r.corners 3) (r.corners 0))

/-- Theorem: The maximum number of intersection points between a circle and a rectangle is 8 --/
theorem max_intersection_points_circle_rectangle :
  ∀ c : Circle, ∀ r : Rectangle, intersectionPointsCircleRectangle c r ≤ 8 ∧
  ∃ c : Circle, ∃ r : Rectangle, intersectionPointsCircleRectangle c r = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_circle_rectangle_l3426_342629


namespace NUMINAMATH_CALUDE_line_point_distance_l3426_342611

/-- Given a line x = 3y + 5 passing through points (m, n) and (m + d, n + p),
    where p = 0.6666666666666666, prove that d = 2 -/
theorem line_point_distance (m n d p : ℝ) : 
  p = 0.6666666666666666 →
  m = 3 * n + 5 →
  (m + d) = 3 * (n + p) + 5 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_line_point_distance_l3426_342611


namespace NUMINAMATH_CALUDE_median_is_twelve_l3426_342622

def group_sizes : List ℕ := [10, 10, 8]

def median (l : List ℕ) (x : ℕ) : ℚ :=
  sorry

theorem median_is_twelve (x : ℕ) : median (x :: group_sizes) x = 12 :=
  sorry

end NUMINAMATH_CALUDE_median_is_twelve_l3426_342622


namespace NUMINAMATH_CALUDE_expression_evaluation_l3426_342645

theorem expression_evaluation (m n : ℚ) (hm : m = -1/3) (hn : n = 1/2) :
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3426_342645


namespace NUMINAMATH_CALUDE_john_hard_hat_ratio_l3426_342667

/-- Proves that the ratio of green to pink hard hats John took away is 2:1 -/
theorem john_hard_hat_ratio :
  let initial_pink : ℕ := 26
  let initial_green : ℕ := 15
  let initial_yellow : ℕ := 24
  let carl_pink : ℕ := 4
  let john_pink : ℕ := 6
  let remaining_total : ℕ := 43
  let initial_total : ℕ := initial_pink + initial_green + initial_yellow
  let john_green : ℕ := initial_total - carl_pink - john_pink - remaining_total
  (john_green : ℚ) / (john_pink : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_john_hard_hat_ratio_l3426_342667


namespace NUMINAMATH_CALUDE_circles_have_another_common_tangent_l3426_342615

-- Define the basic geometric objects
structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given conditions
def Semicircle (k : Circle) (A B : Point) : Prop :=
  k.center = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) ∧
  k.radius = (((B.x - A.x)^2 + (B.y - A.y)^2)^(1/2)) / 2

def OnCircle (P : Point) (k : Circle) : Prop :=
  (P.x - k.center.x)^2 + (P.y - k.center.y)^2 = k.radius^2

def Perpendicular (C D : Point) (A B : Point) : Prop :=
  (B.x - A.x) * (C.x - D.x) + (B.y - A.y) * (C.y - D.y) = 0

def Incircle (k : Circle) (A B C : Point) : Prop :=
  -- Definition of incircle omitted for brevity
  sorry

def TouchesSegmentAndCircle (k : Circle) (C D : Point) (semicircle : Circle) : Prop :=
  -- Definition of touching segment and circle omitted for brevity
  sorry

def CommonTangent (k1 k2 k3 : Circle) (A B : Point) : Prop :=
  -- Definition of common tangent omitted for brevity
  sorry

-- Main theorem
theorem circles_have_another_common_tangent
  (k semicircle : Circle) (A B C D : Point) (k1 k2 k3 : Circle) :
  Semicircle semicircle A B →
  OnCircle C semicircle →
  C ≠ A ∧ C ≠ B →
  Perpendicular C D A B →
  Incircle k1 A B C →
  TouchesSegmentAndCircle k2 C D semicircle →
  TouchesSegmentAndCircle k3 C D semicircle →
  CommonTangent k1 k2 k3 A B →
  ∃ (E F : Point), E ≠ F ∧ CommonTangent k1 k2 k3 E F ∧ (E ≠ A ∨ F ≠ B) :=
by
  sorry

end NUMINAMATH_CALUDE_circles_have_another_common_tangent_l3426_342615


namespace NUMINAMATH_CALUDE_closest_to_product_l3426_342669

def product : ℝ := 0.001532 * 2134672

def options : List ℝ := [3100, 3150, 3200, 3500, 4000]

theorem closest_to_product : 
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |product - x| ≤ |product - y| ∧
  x = 3150 :=
sorry

end NUMINAMATH_CALUDE_closest_to_product_l3426_342669


namespace NUMINAMATH_CALUDE_min_abs_z_l3426_342677

open Complex

theorem min_abs_z (z : ℂ) (h : abs (z - 1) + abs (z - (3 + 2*I)) = 2 * Real.sqrt 2) :
  ∃ (w : ℂ), abs w ≤ abs z ∧ abs w = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l3426_342677


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3426_342614

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (x + 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  (a₁ + 3*a₃ + 5*a₅ + 7*a₇ + 9*a₉)^2 - (2*a₂ + 4*a₄ + 6*a₆ + 8*a₈)^2 = 3^12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3426_342614


namespace NUMINAMATH_CALUDE_mixture_alcohol_percentage_l3426_342662

/-- Represents the alcohol content and volume of a solution -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the amount of pure alcohol in a solution -/
def alcoholContent (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

theorem mixture_alcohol_percentage 
  (x : Solution) 
  (y : Solution) 
  (h1 : x.volume = 250)
  (h2 : x.alcoholPercentage = 0.1)
  (h3 : y.alcoholPercentage = 0.3)
  (h4 : y.volume = 750) :
  let mixedSolution : Solution := ⟨x.volume + y.volume, (alcoholContent x + alcoholContent y) / (x.volume + y.volume)⟩
  mixedSolution.alcoholPercentage = 0.25 := by
sorry

end NUMINAMATH_CALUDE_mixture_alcohol_percentage_l3426_342662


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3426_342617

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3426_342617


namespace NUMINAMATH_CALUDE_uncovered_area_is_64_l3426_342623

/-- Represents the dimensions of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the dimensions of a square carpet -/
structure Carpet :=
  (side : ℝ)

/-- Calculates the area of a rectangular floor -/
def floorArea (f : Floor) : ℝ :=
  f.length * f.width

/-- Calculates the area of a square carpet -/
def carpetArea (c : Carpet) : ℝ :=
  c.side * c.side

/-- Calculates the uncovered area when placing a carpet on a floor -/
def uncoveredArea (f : Floor) (c : Carpet) : ℝ :=
  floorArea f - carpetArea c

theorem uncovered_area_is_64 (f : Floor) (c : Carpet) 
    (h1 : f.length = 10)
    (h2 : f.width = 8)
    (h3 : c.side = 4) :
  uncoveredArea f c = 64 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_is_64_l3426_342623


namespace NUMINAMATH_CALUDE_delta_f_P0_approx_df_P0_l3426_342607

-- Define the function f
def f (x y : ℝ) : ℝ := x^2 * y

-- Define the point P0
def P0 : ℝ × ℝ := (5, 4)

-- Define Δx and Δy
def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

-- Theorem for Δf(P0)
theorem delta_f_P0_approx : 
  let (x0, y0) := P0
  abs (f (x0 + Δx) (y0 + Δy) - f x0 y0 + 1.162) < 0.001 := by sorry

-- Theorem for df(P0)
theorem df_P0 : 
  let (x0, y0) := P0
  (2 * x0 * y0) * Δx + x0^2 * Δy = -1 := by sorry

end NUMINAMATH_CALUDE_delta_f_P0_approx_df_P0_l3426_342607


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_palindrome_l3426_342668

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem four_digit_perfect_square_palindrome :
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n := by sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_palindrome_l3426_342668


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l3426_342697

/-- The probability of selecting 3 non-defective pencils from a box of 9 pencils with 2 defective pencils -/
theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 9
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 :=
by sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l3426_342697


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3426_342608

theorem min_distance_to_origin (x y : ℝ) : 
  3 * x + 4 * y = 24 → 
  x - 2 * y = 0 → 
  ∃ (min_dist : ℝ), 
    min_dist = Real.sqrt 28.8 ∧ 
    ∀ (x' y' : ℝ), 3 * x' + 4 * y' = 24 → x' - 2 * y' = 0 → 
      Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3426_342608


namespace NUMINAMATH_CALUDE_sequence_exists_l3426_342654

theorem sequence_exists : ∃ (seq : Fin 2000 → ℝ), 
  (∀ i : Fin 1998, seq i + seq (i + 1) + seq (i + 2) < 0) ∧ 
  (Finset.sum Finset.univ seq > 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_exists_l3426_342654


namespace NUMINAMATH_CALUDE_jack_marbles_remaining_l3426_342644

/-- Given Jack starts with 62 marbles and shares 33 marbles, prove that he ends up with 29 marbles. -/
theorem jack_marbles_remaining (initial_marbles : ℕ) (shared_marbles : ℕ) 
  (h1 : initial_marbles = 62)
  (h2 : shared_marbles = 33) :
  initial_marbles - shared_marbles = 29 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_remaining_l3426_342644


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3426_342621

theorem fraction_multiplication : (1/4 - 1/2 + 2/3) * (-12 : ℚ) = -8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3426_342621


namespace NUMINAMATH_CALUDE_school_population_l3426_342671

theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t := by sorry

end NUMINAMATH_CALUDE_school_population_l3426_342671


namespace NUMINAMATH_CALUDE_complex_cube_equation_l3426_342653

theorem complex_cube_equation :
  ∃! (z : ℂ), ∃ (x y c : ℤ), 
    x > 0 ∧ y > 0 ∧ 
    z = x + y * I ∧
    z^3 = -74 + c * I ∧
    z = 1 + 5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_equation_l3426_342653


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l3426_342672

/-- Given the probabilities of selection for three siblings, prove that the probability of all three being selected is 3/28 -/
theorem siblings_selection_probability
  (p_ram : ℚ) (p_ravi : ℚ) (p_rani : ℚ)
  (h_ram : p_ram = 5 / 7)
  (h_ravi : p_ravi = 1 / 5)
  (h_rani : p_rani = 3 / 4) :
  p_ram * p_ravi * p_rani = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l3426_342672


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3426_342686

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3426_342686


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l3426_342678

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l3426_342678


namespace NUMINAMATH_CALUDE_units_digit_17_2011_l3426_342638

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the property that powers of 17 have the same units digit as powers of 7
axiom units_digit_17_7 (n : ℕ) : unitsDigit (17^n) = unitsDigit (7^n)

-- Define the cycle of units digits for powers of 7
def sevenPowerCycle : List ℕ := [7, 9, 3, 1]

-- Theorem stating that the units digit of 17^2011 is 3
theorem units_digit_17_2011 : unitsDigit (17^2011) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_2011_l3426_342638


namespace NUMINAMATH_CALUDE_star_four_three_l3426_342657

-- Define the new operation
def star (a b : ℤ) : ℤ := a^2 - a*b + b^2

-- State the theorem
theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l3426_342657


namespace NUMINAMATH_CALUDE_correct_distribution_l3426_342650

/-- Represents the distribution of chestnuts among three girls -/
structure ChestnutDistribution where
  alya : ℕ
  valya : ℕ
  galya : ℕ

/-- Checks if the given distribution satisfies the problem conditions -/
def isValidDistribution (d : ChestnutDistribution) : Prop :=
  d.alya + d.valya + d.galya = 70 ∧
  4 * d.valya = 3 * d.alya ∧
  7 * d.alya = 6 * d.galya

/-- Theorem stating that the given distribution is correct -/
theorem correct_distribution :
  let d : ChestnutDistribution := ⟨24, 18, 28⟩
  isValidDistribution d := by
  sorry


end NUMINAMATH_CALUDE_correct_distribution_l3426_342650


namespace NUMINAMATH_CALUDE_probability_ascending_rolls_eq_five_fiftyfour_l3426_342647

/-- A standard die has faces labeled from 1 to 6 -/
def standardDie : Finset Nat := Finset.range 6

/-- The number of times the die is rolled -/
def numRolls : Nat := 3

/-- The probability of rolling three dice and getting three distinct numbers in ascending order -/
def probabilityAscendingRolls : Rat :=
  (Nat.choose 6 3 : Rat) / (6 ^ numRolls)

theorem probability_ascending_rolls_eq_five_fiftyfour : 
  probabilityAscendingRolls = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_probability_ascending_rolls_eq_five_fiftyfour_l3426_342647


namespace NUMINAMATH_CALUDE_baseball_team_groups_l3426_342604

theorem baseball_team_groups (new_players : ℕ) (returning_players : ℕ) (players_per_group : ℕ) :
  new_players = 48 →
  returning_players = 6 →
  players_per_group = 6 →
  (new_players + returning_players) / players_per_group = 9 :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l3426_342604


namespace NUMINAMATH_CALUDE_wolf_tail_growth_l3426_342619

theorem wolf_tail_growth (x y : ℕ) : 1 * 2^x * 3^y = 864 ↔ x = 5 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_wolf_tail_growth_l3426_342619


namespace NUMINAMATH_CALUDE_sum_greater_than_two_l3426_342658

theorem sum_greater_than_two (x y : ℝ) 
  (h1 : x^7 > y^6) 
  (h2 : y^7 > x^6) : 
  x + y > 2 := by
sorry

end NUMINAMATH_CALUDE_sum_greater_than_two_l3426_342658


namespace NUMINAMATH_CALUDE_function_existence_l3426_342695

theorem function_existence : ∃ (f : ℤ → ℤ), ∀ (k : ℕ) (m : ℤ), k ≤ 1996 → ∃ (x : ℤ), f x + k * x = m := by
  sorry

end NUMINAMATH_CALUDE_function_existence_l3426_342695


namespace NUMINAMATH_CALUDE_greatest_common_divisor_780_180_240_l3426_342655

theorem greatest_common_divisor_780_180_240 :
  (∃ (d : ℕ), d ∣ 780 ∧ d ∣ 180 ∧ d ∣ 240 ∧ d < 100 ∧
    ∀ (x : ℕ), x ∣ 780 ∧ x ∣ 180 ∧ x ∣ 240 ∧ x < 100 → x ≤ d) ∧
  (60 ∣ 780 ∧ 60 ∣ 180 ∧ 60 ∣ 240 ∧ 60 < 100) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_780_180_240_l3426_342655


namespace NUMINAMATH_CALUDE_dans_age_l3426_342652

theorem dans_age (dan_age ben_age : ℕ) : 
  ben_age = dan_age - 3 →
  ben_age + dan_age = 53 →
  dan_age = 28 := by
sorry

end NUMINAMATH_CALUDE_dans_age_l3426_342652


namespace NUMINAMATH_CALUDE_xyz_value_l3426_342661

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 8/3) :
  x * y * z = (17 + Real.sqrt 285) / 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3426_342661


namespace NUMINAMATH_CALUDE_spatial_relationships_l3426_342656

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem spatial_relationships 
  (m n : Line) (α β : Plane) : 
  (∀ (m n : Line) (β : Plane), 
    perpendicular m β → perpendicular n β → parallel_lines m n) ∧
  (∀ (m : Line) (α β : Plane), 
    perpendicular m α → perpendicular m β → parallel_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relationships_l3426_342656


namespace NUMINAMATH_CALUDE_initial_ribbon_tape_length_l3426_342670

/-- The initial length of ribbon tape Yujin had, in meters. -/
def initial_length : ℝ := 8.9

/-- The length of ribbon tape required for one ribbon, in meters. -/
def ribbon_length : ℝ := 0.84

/-- The number of ribbons made. -/
def num_ribbons : ℕ := 10

/-- The length of remaining ribbon tape, in meters. -/
def remaining_length : ℝ := 0.5

/-- Theorem stating that the initial length of ribbon tape equals 8.9 meters. -/
theorem initial_ribbon_tape_length :
  initial_length = ribbon_length * num_ribbons + remaining_length := by
  sorry

end NUMINAMATH_CALUDE_initial_ribbon_tape_length_l3426_342670


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l3426_342631

/-- The maximum number of sections a rectangle can be divided into by n line segments -/
def maxSections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => maxSections m + m + 1

/-- Theorem stating that 5 line segments can divide a rectangle into at most 16 sections -/
theorem max_sections_five_lines :
  maxSections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l3426_342631


namespace NUMINAMATH_CALUDE_birds_to_africa_l3426_342649

/-- The number of bird families that flew away to Africa -/
def families_to_africa : ℕ := 118 - 80

/-- The number of bird families that flew away to Asia -/
def families_to_asia : ℕ := 80

/-- The total number of bird families that flew away for the winter -/
def total_families_away : ℕ := 118

/-- The number of bird families living near the mountain (not used in the proof) -/
def families_near_mountain : ℕ := 18

theorem birds_to_africa :
  families_to_africa = 38 ∧
  families_to_africa + families_to_asia = total_families_away :=
sorry

end NUMINAMATH_CALUDE_birds_to_africa_l3426_342649


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3426_342689

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 729 -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3426_342689


namespace NUMINAMATH_CALUDE_karl_process_preserves_swapped_pairs_l3426_342627

/-- Represents a permutation of cards -/
def Permutation := List Nat

/-- Counts the number of swapped pairs (inversions) in a permutation -/
def countSwappedPairs (p : Permutation) : Nat :=
  sorry

/-- Karl's process of rearranging cards -/
def karlProcess (p : Permutation) : Permutation :=
  sorry

theorem karl_process_preserves_swapped_pairs (n : Nat) (initial : Permutation) :
  initial.length = n →
  initial.toFinset = Finset.range n →
  countSwappedPairs initial = countSwappedPairs (karlProcess initial) :=
sorry

end NUMINAMATH_CALUDE_karl_process_preserves_swapped_pairs_l3426_342627


namespace NUMINAMATH_CALUDE_andy_ate_six_cookies_six_is_max_cookies_for_andy_l3426_342620

/-- Represents the number of cookies eaten by Andy -/
def andys_cookies : ℕ := sorry

/-- The total number of cookies baked -/
def total_cookies : ℕ := 36

/-- Theorem stating that Andy ate 6 cookies, given the problem conditions -/
theorem andy_ate_six_cookies : andys_cookies = 6 := by
  have h1 : andys_cookies + 2 * andys_cookies + 3 * andys_cookies = total_cookies := sorry
  have h2 : andys_cookies ≤ 6 := sorry
  sorry

/-- Theorem proving that 6 is the maximum number of cookies Andy could have eaten -/
theorem six_is_max_cookies_for_andy :
  ∀ n : ℕ, n > 6 → n + 2 * n + 3 * n > total_cookies := by
  sorry

end NUMINAMATH_CALUDE_andy_ate_six_cookies_six_is_max_cookies_for_andy_l3426_342620


namespace NUMINAMATH_CALUDE_planted_field_fraction_l3426_342632

theorem planted_field_fraction (a b h x : ℝ) (ha : a = 5) (hb : b = 12) (hh : h = 3) :
  let c := (a^2 + b^2).sqrt
  let s := x^2
  let triangle_area := a * b / 2
  h = (2 * triangle_area) / c - (b * x) / c →
  (triangle_area - s) / triangle_area = 431 / 480 :=
by sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l3426_342632


namespace NUMINAMATH_CALUDE_arith_geom_seq_sum_30_l3426_342692

/-- An arithmetic-geometric sequence with its partial sums -/
structure ArithGeomSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums of the sequence
  is_arith_geom : ∀ n, (S (n + 10) - S n) / (S (n + 20) - S (n + 10)) = (S (n + 20) - S (n + 10)) / (S (n + 30) - S (n + 20))

/-- Theorem: For an arithmetic-geometric sequence, if S_10 = 10 and S_20 = 30, then S_30 = 70 -/
theorem arith_geom_seq_sum_30 (seq : ArithGeomSeq) (h1 : seq.S 10 = 10) (h2 : seq.S 20 = 30) : 
  seq.S 30 = 70 := by
  sorry

end NUMINAMATH_CALUDE_arith_geom_seq_sum_30_l3426_342692


namespace NUMINAMATH_CALUDE_frog_corner_probability_l3426_342680

/-- Represents a position on the 4x4 grid -/
inductive Position
| Corner
| Edge
| Middle

/-- Represents the state of the frog's movement -/
structure FrogState where
  position : Position
  hops : Nat

/-- Transition function for the frog's movement -/
def transition (state : FrogState) : FrogState :=
  sorry

/-- Probability of reaching a corner from a given state -/
def cornerProbability (state : FrogState) : Rat :=
  sorry

/-- The starting state of the frog -/
def initialState : FrogState :=
  { position := Position.Edge, hops := 0 }

/-- Main theorem: Probability of reaching a corner within 4 hops -/
theorem frog_corner_probability :
  cornerProbability { position := initialState.position, hops := 4 } = 35 / 64 := by
  sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l3426_342680


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3426_342673

theorem max_sum_of_factors (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  A + B + C ≤ 42 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3426_342673


namespace NUMINAMATH_CALUDE_integral_problems_l3426_342688

theorem integral_problems :
  (∃ k : ℝ, (∫ x in (0:ℝ)..2, (3*x^2 + k)) = 10 ∧ k = 1) ∧
  (∫ x in (-1:ℝ)..8, x^(1/3)) = 45/4 :=
by sorry

end NUMINAMATH_CALUDE_integral_problems_l3426_342688


namespace NUMINAMATH_CALUDE_min_value_theorem_l3426_342694

theorem min_value_theorem (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 36*c^2 = 4) :
  ∃ (m : ℝ), m = -2 * Real.sqrt 14 ∧ ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 36*z^2 = 4 → 3*x + 6*y + 12*z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3426_342694


namespace NUMINAMATH_CALUDE_square_point_distance_probability_l3426_342603

-- Define the square
def Square := {p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 2 ∧ (p.2 = 0 ∨ p.2 = 2)) ∨ (0 ≤ p.2 ∧ p.2 ≤ 2 ∧ (p.1 = 0 ∨ p.1 = 2))}

-- Define the probability function
noncomputable def probability : ℝ := sorry

-- Define the gcd function
def gcd (a b c : ℕ) : ℕ := sorry

-- State the theorem
theorem square_point_distance_probability :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    gcd a b c = 1 ∧
    probability = (a - b * Real.pi) / c ∧
    a = 28 ∧ b = 1 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_square_point_distance_probability_l3426_342603


namespace NUMINAMATH_CALUDE_language_school_solution_l3426_342687

/-- Represents the state of the language school at a given time --/
structure SchoolState where
  num_teachers : ℕ
  total_age : ℕ

/-- The language school problem --/
def language_school_problem (initial : SchoolState) 
  (new_teacher_age : ℕ) (left_teacher_age : ℕ) : Prop :=
  -- Initial state (2007)
  initial.num_teachers = 7 ∧
  -- State after new teacher joins (2010)
  (initial.total_age + 21 + new_teacher_age) / 8 = initial.total_age / 7 ∧
  -- State after one teacher leaves (2012)
  (initial.total_age + 37 + new_teacher_age - left_teacher_age) / 7 = initial.total_age / 7 ∧
  -- New teacher's age in 2010
  new_teacher_age = 25

theorem language_school_solution (initial : SchoolState) 
  (new_teacher_age : ℕ) (left_teacher_age : ℕ) 
  (h : language_school_problem initial new_teacher_age left_teacher_age) :
  left_teacher_age = 62 ∧ initial.total_age / 7 = 46 := by
  sorry

#check language_school_solution

end NUMINAMATH_CALUDE_language_school_solution_l3426_342687


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_choose_2_l3426_342610

theorem binomial_coefficient_n_choose_2 (n : ℕ) (h : n ≥ 2) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_choose_2_l3426_342610


namespace NUMINAMATH_CALUDE_inequality_proof_l3426_342684

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 ≥ 2*(a + b - 1) ∧ 
  (a > 0 ∧ b > 0 ∧ a + b = 3 → 1/a + 4/(b+1) ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3426_342684


namespace NUMINAMATH_CALUDE_orchid_to_rose_ratio_l3426_342600

/-- Proves that the ratio of orchids to roses in each centerpiece is 2:1 given the specified conditions. -/
theorem orchid_to_rose_ratio 
  (num_centerpieces : ℕ) 
  (roses_per_centerpiece : ℕ) 
  (lilies_per_centerpiece : ℕ) 
  (total_budget : ℕ) 
  (cost_per_flower : ℕ) 
  (h1 : num_centerpieces = 6)
  (h2 : roses_per_centerpiece = 8)
  (h3 : lilies_per_centerpiece = 6)
  (h4 : total_budget = 2700)
  (h5 : cost_per_flower = 15) : 
  ∃ (orchids_per_centerpiece : ℕ), 
    orchids_per_centerpiece = 2 * roses_per_centerpiece :=
by sorry

end NUMINAMATH_CALUDE_orchid_to_rose_ratio_l3426_342600


namespace NUMINAMATH_CALUDE_cat_average_weight_l3426_342665

theorem cat_average_weight : 
  let num_cats : ℕ := 4
  let weight_cat1 : ℝ := 12
  let weight_cat2 : ℝ := 12
  let weight_cat3 : ℝ := 14.7
  let weight_cat4 : ℝ := 9.3
  let total_weight : ℝ := weight_cat1 + weight_cat2 + weight_cat3 + weight_cat4
  let average_weight : ℝ := total_weight / num_cats
  average_weight = 12 := by
    sorry

end NUMINAMATH_CALUDE_cat_average_weight_l3426_342665


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3426_342696

theorem polynomial_evaluation :
  ∀ x : ℝ, 
    x > 0 → 
    x^2 - 3*x - 10 = 0 → 
    x^3 - 3*x^2 - 9*x + 5 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3426_342696


namespace NUMINAMATH_CALUDE_factorial_ratio_52_50_l3426_342605

theorem factorial_ratio_52_50 : Nat.factorial 52 / Nat.factorial 50 = 2652 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_52_50_l3426_342605


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3426_342618

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3426_342618


namespace NUMINAMATH_CALUDE_probability_problem_l3426_342685

structure JarContents where
  red : Nat
  white : Nat
  black : Nat

def jarA : JarContents := { red := 5, white := 2, black := 3 }
def jarB : JarContents := { red := 4, white := 3, black := 3 }

def totalBalls (jar : JarContents) : Nat :=
  jar.red + jar.white + jar.black

def P_A1 : Rat := jarA.red / totalBalls jarA
def P_A2 : Rat := jarA.white / totalBalls jarA
def P_A3 : Rat := jarA.black / totalBalls jarA

def P_B_given_A1 : Rat := (jarB.red + 1) / (totalBalls jarB + 1)
def P_B_given_A2 : Rat := jarB.red / (totalBalls jarB + 1)
def P_B_given_A3 : Rat := jarB.red / (totalBalls jarB + 1)

theorem probability_problem :
  (P_B_given_A1 = 5 / 11) ∧
  (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 9 / 22) ∧
  (P_A1 + P_A2 + P_A3 = 1) :=
by sorry

#check probability_problem

end NUMINAMATH_CALUDE_probability_problem_l3426_342685


namespace NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_smallest_integer_m_for_distinct_roots_l3426_342683

theorem smallest_m_for_distinct_roots (m : ℤ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) ↔ m ≥ 0 :=
by sorry

theorem smallest_integer_m_for_distinct_roots : 
  ∃ m₀ : ℤ, m₀ ≥ 0 ∧ ∀ m : ℤ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) → m ≥ m₀ :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_smallest_integer_m_for_distinct_roots_l3426_342683


namespace NUMINAMATH_CALUDE_sweater_markup_l3426_342625

theorem sweater_markup (wholesale : ℝ) (retail : ℝ) (h1 : retail > 0) (h2 : wholesale > 0) :
  (retail * (1 - 0.6) = wholesale * 1.2) →
  ((retail - wholesale) / wholesale * 100 = 200) := by
sorry

end NUMINAMATH_CALUDE_sweater_markup_l3426_342625


namespace NUMINAMATH_CALUDE_work_completion_l3426_342636

/-- Given that 36 men can complete a piece of work in 25 hours,
    prove that 10 men can complete the same work in 90 hours. -/
theorem work_completion (work : ℝ) : 
  work = 36 * 25 → work = 10 * 90 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3426_342636


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_6_14_22_30_l3426_342601

theorem smallest_perfect_square_divisible_by_6_14_22_30 :
  ∃ (n : ℕ), n > 0 ∧ n = 5336100 ∧ 
  (∃ (k : ℕ), n = k^2) ∧
  6 ∣ n ∧ 14 ∣ n ∧ 22 ∣ n ∧ 30 ∣ n ∧
  (∀ (m : ℕ), m > 0 → (∃ (j : ℕ), m = j^2) → 
    6 ∣ m → 14 ∣ m → 22 ∣ m → 30 ∣ m → m ≥ n) :=
by sorry


end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_6_14_22_30_l3426_342601
