import Mathlib

namespace NUMINAMATH_CALUDE_gym_membership_ratio_l110_11097

theorem gym_membership_ratio (f m : ℕ) (h1 : f > 0) (h2 : m > 0) : 
  (35 : ℝ) * f + 20 * m = 25 * (f + m) → f / m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_gym_membership_ratio_l110_11097


namespace NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l110_11026

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem custom_op_neg_two_neg_one :
  customOp (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l110_11026


namespace NUMINAMATH_CALUDE_partition_displacement_is_one_sixth_length_l110_11082

/-- Represents a cylindrical vessel with a movable partition -/
structure Vessel where
  length : ℝ
  initial_partition_position : ℝ
  final_partition_position : ℝ

/-- Calculates the displacement of the partition -/
def partition_displacement (v : Vessel) : ℝ :=
  v.initial_partition_position - v.final_partition_position

/-- Theorem stating the displacement of the partition -/
theorem partition_displacement_is_one_sixth_length (v : Vessel) 
  (h1 : v.length > 0)
  (h2 : v.initial_partition_position = 2 * v.length / 3)
  (h3 : v.final_partition_position = v.length / 2) :
  partition_displacement v = v.length / 6 := by
  sorry

#check partition_displacement_is_one_sixth_length

end NUMINAMATH_CALUDE_partition_displacement_is_one_sixth_length_l110_11082


namespace NUMINAMATH_CALUDE_iron_weight_is_11_16_l110_11078

/-- The weight of the piece of aluminum in pounds -/
def aluminum_weight : ℝ := 0.83

/-- The difference in weight between the piece of iron and the piece of aluminum in pounds -/
def weight_difference : ℝ := 10.33

/-- The weight of the piece of iron in pounds -/
def iron_weight : ℝ := aluminum_weight + weight_difference

/-- Theorem stating that the weight of the piece of iron is 11.16 pounds -/
theorem iron_weight_is_11_16 : iron_weight = 11.16 := by sorry

end NUMINAMATH_CALUDE_iron_weight_is_11_16_l110_11078


namespace NUMINAMATH_CALUDE_average_problem_l110_11037

theorem average_problem (x : ℝ) : 
  (2 + 4 + 1 + 3 + x) / 5 = 3 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l110_11037


namespace NUMINAMATH_CALUDE_unique_intersection_point_l110_11071

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies the equation 3x + 2y - 9 = 0 -/
def satisfiesLine1 (p : Point) : Prop :=
  3 * p.x + 2 * p.y - 9 = 0

/-- Checks if a point satisfies the equation 5x - 2y - 10 = 0 -/
def satisfiesLine2 (p : Point) : Prop :=
  5 * p.x - 2 * p.y - 10 = 0

/-- Checks if a point satisfies the equation x = 3 -/
def satisfiesLine3 (p : Point) : Prop :=
  p.x = 3

/-- Checks if a point satisfies the equation y = 1 -/
def satisfiesLine4 (p : Point) : Prop :=
  p.y = 1

/-- Checks if a point satisfies the equation x + y = 4 -/
def satisfiesLine5 (p : Point) : Prop :=
  p.x + p.y = 4

/-- Checks if a point satisfies all five line equations -/
def satisfiesAllLines (p : Point) : Prop :=
  satisfiesLine1 p ∧ satisfiesLine2 p ∧ satisfiesLine3 p ∧ satisfiesLine4 p ∧ satisfiesLine5 p

/-- Theorem stating that there is exactly one point satisfying all five line equations -/
theorem unique_intersection_point : ∃! p : Point, satisfiesAllLines p := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l110_11071


namespace NUMINAMATH_CALUDE_brady_record_chase_l110_11039

/-- The minimum average yards per game needed to beat the record -/
def min_avg_yards_per_game (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ) : ℚ :=
  (current_record + 1 - current_yards) / games_left

theorem brady_record_chase (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ)
  (h1 : current_record = 5999)
  (h2 : current_yards = 4200)
  (h3 : games_left = 6) :
  min_avg_yards_per_game current_record current_yards games_left = 300 := by
sorry

end NUMINAMATH_CALUDE_brady_record_chase_l110_11039


namespace NUMINAMATH_CALUDE_cubic_function_b_value_l110_11040

/-- A cubic function f(x) = x³ + bx² + cx + d passing through (-1, 0), (1, 0), and (0, 2) has b = -2 -/
theorem cubic_function_b_value (b c d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x + d
  (f (-1) = 0) → (f 1 = 0) → (f 0 = 2) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_b_value_l110_11040


namespace NUMINAMATH_CALUDE_dodecagon_rectangle_area_equality_l110_11089

/-- The area of a regular dodecagon inscribed in a circle of radius r -/
def area_inscribed_dodecagon (r : ℝ) : ℝ := 3 * r^2

/-- The area of a rectangle with sides r and 3r -/
def area_rectangle (r : ℝ) : ℝ := r * (3 * r)

theorem dodecagon_rectangle_area_equality (r : ℝ) :
  area_inscribed_dodecagon r = area_rectangle r :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_rectangle_area_equality_l110_11089


namespace NUMINAMATH_CALUDE_circle_area_tripled_l110_11010

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l110_11010


namespace NUMINAMATH_CALUDE_simplify_tan_product_l110_11047

-- Define the tangent function
noncomputable def tan (x : Real) : Real := Real.tan x

-- State the theorem
theorem simplify_tan_product : 
  (1 + tan (10 * Real.pi / 180)) * (1 + tan (35 * Real.pi / 180)) = 2 := by
  -- Assuming the angle addition formula for tangent
  have angle_addition_formula : ∀ a b, 
    tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry
  
  -- Assuming tan 45° = 1
  have tan_45_deg : tan (45 * Real.pi / 180) = 1 := by sorry

  sorry -- The proof goes here

end NUMINAMATH_CALUDE_simplify_tan_product_l110_11047


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l110_11057

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 7)
  (h_sum : a 3 + a 6 = 16)
  (h_an : ∃ n : ℕ, a n = 31) :
  ∃ n : ℕ, a n = 31 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l110_11057


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l110_11048

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l110_11048


namespace NUMINAMATH_CALUDE_monotonically_decreasing_condition_l110_11020

def f (x : ℝ) := x^2 - 2*x + 3

theorem monotonically_decreasing_condition (m : ℝ) :
  (∀ x y, x < y ∧ y < m → f x > f y) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_condition_l110_11020


namespace NUMINAMATH_CALUDE_spinner_direction_l110_11072

-- Define the directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation := 3 + 3/4
  let counterclockwise_rotation := 2 + 2/4
  rotate initial_direction clockwise_rotation counterclockwise_rotation = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_direction_l110_11072


namespace NUMINAMATH_CALUDE_eva_max_silver_tokens_l110_11050

/-- Represents the number of tokens Eva has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  silverOut : ℕ
  redOut : ℕ
  blueOut : ℕ

/-- The maximum number of silver tokens Eva can obtain -/
def maxSilverTokens (initial : TokenCount) (booth1 booth2 : Booth) : ℕ :=
  sorry

/-- Theorem stating that Eva can obtain at most 57 silver tokens -/
theorem eva_max_silver_tokens :
  let initial := TokenCount.mk 60 90 0
  let booth1 := Booth.mk 3 0 2 0 1
  let booth2 := Booth.mk 0 4 3 1 0
  maxSilverTokens initial booth1 booth2 = 57 :=
by sorry

end NUMINAMATH_CALUDE_eva_max_silver_tokens_l110_11050


namespace NUMINAMATH_CALUDE_second_chapter_pages_l110_11045

theorem second_chapter_pages (total_pages first_chapter third_chapter : ℕ) 
  (h1 : total_pages = 125)
  (h2 : first_chapter = 66)
  (h3 : third_chapter = 24) :
  total_pages - first_chapter - third_chapter = 59 :=
by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l110_11045


namespace NUMINAMATH_CALUDE_carrots_planted_per_hour_l110_11062

theorem carrots_planted_per_hour 
  (rows : ℕ) 
  (plants_per_row : ℕ) 
  (total_hours : ℕ) 
  (h1 : rows = 400) 
  (h2 : plants_per_row = 300) 
  (h3 : total_hours = 20) : 
  (rows * plants_per_row) / total_hours = 6000 := by
sorry

end NUMINAMATH_CALUDE_carrots_planted_per_hour_l110_11062


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l110_11007

theorem algebraic_expression_value (x y : ℝ) 
  (hx : x = 1 / (Real.sqrt 3 - 2))
  (hy : y = 1 / (Real.sqrt 3 + 2)) :
  (x^2 + x*y + y^2) / (13 * (x + y)) = -(Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l110_11007


namespace NUMINAMATH_CALUDE_envelope_addressing_problem_l110_11012

/-- A manufacturer's envelope addressing problem -/
theorem envelope_addressing_problem 
  (initial_machine : ℝ) 
  (first_added_machine : ℝ) 
  (combined_initial_and_first : ℝ) 
  (all_three_machines : ℝ) 
  (h1 : initial_machine = 600 / 10)
  (h2 : first_added_machine = 600 / 5)
  (h3 : combined_initial_and_first = 600 / 3)
  (h4 : all_three_machines = 600 / 1) :
  (600 / (all_three_machines - initial_machine - first_added_machine)) = 10 / 7 := by
  sorry

#check envelope_addressing_problem

end NUMINAMATH_CALUDE_envelope_addressing_problem_l110_11012


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l110_11052

open Real

theorem function_inequality_implies_a_range (a b : ℝ) :
  (∀ x ∈ Set.Ioo (Real.exp 1) ((Real.exp 1) ^ 2),
    ∀ b ≤ 0,
      a * log x - b * x^2 ≥ x) →
  a ≥ (Real.exp 1)^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l110_11052


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l110_11093

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (bars_left : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - bars_left) * price_per_bar

/-- Theorem stating that Olivia would make $9 from selling the chocolate bars -/
theorem olivia_chocolate_sales : 
  money_made 7 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l110_11093


namespace NUMINAMATH_CALUDE_not_right_triangle_2_3_4_l110_11076

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that the set {2, 3, 4} cannot form a right triangle -/
theorem not_right_triangle_2_3_4 : ¬ is_right_triangle 2 3 4 := by
  sorry

#check not_right_triangle_2_3_4

end NUMINAMATH_CALUDE_not_right_triangle_2_3_4_l110_11076


namespace NUMINAMATH_CALUDE_lcm_5_6_8_9_l110_11051

theorem lcm_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_8_9_l110_11051


namespace NUMINAMATH_CALUDE_banana_groups_l110_11034

theorem banana_groups (total_bananas : ℕ) (group_size : ℕ) (h1 : total_bananas = 180) (h2 : group_size = 18) :
  total_bananas / group_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_l110_11034


namespace NUMINAMATH_CALUDE_billys_weekend_activities_l110_11094

/-- Billy's weekend activities theorem -/
theorem billys_weekend_activities :
  -- Define the given conditions
  let free_time_per_day : ℕ := 8
  let weekend_days : ℕ := 2
  let pages_per_hour : ℕ := 60
  let pages_per_book : ℕ := 80
  let books_read : ℕ := 3

  -- Calculate total free time
  let total_free_time : ℕ := free_time_per_day * weekend_days

  -- Calculate total pages read
  let total_pages_read : ℕ := pages_per_book * books_read

  -- Calculate time spent reading
  let reading_time : ℕ := total_pages_read / pages_per_hour

  -- Calculate time spent playing video games
  let gaming_time : ℕ := total_free_time - reading_time

  -- Calculate percentage of time spent playing video games
  let gaming_percentage : ℚ := (gaming_time : ℚ) / (total_free_time : ℚ) * 100

  -- Prove that Billy spends 75% of his time playing video games
  gaming_percentage = 75 := by sorry

end NUMINAMATH_CALUDE_billys_weekend_activities_l110_11094


namespace NUMINAMATH_CALUDE_days_worked_by_a_l110_11055

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 16

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the number of days worked by person c -/
def days_c : ℕ := 4

/-- Represents the daily wage ratio of person a -/
def wage_ratio_a : ℚ := 3

/-- Represents the daily wage ratio of person b -/
def wage_ratio_b : ℚ := 4

/-- Represents the daily wage ratio of person c -/
def wage_ratio_c : ℚ := 5

/-- Represents the daily wage of person c -/
def wage_c : ℚ := 71.15384615384615

/-- Represents the total earnings of all three workers -/
def total_earnings : ℚ := 1480

/-- Theorem stating that given the conditions, the number of days worked by person a is 16 -/
theorem days_worked_by_a : 
  (days_a : ℚ) * (wage_ratio_a * wage_c / wage_ratio_c) + 
  (days_b : ℚ) * (wage_ratio_b * wage_c / wage_ratio_c) + 
  (days_c : ℚ) * wage_c = total_earnings :=
sorry

end NUMINAMATH_CALUDE_days_worked_by_a_l110_11055


namespace NUMINAMATH_CALUDE_select_students_result_l110_11024

/-- The number of ways to select 4 students from two classes, with 2 students from each class, 
    such that exactly 1 female student is among them. -/
def select_students (class_a_male class_a_female class_b_male class_b_female : ℕ) : ℕ :=
  Nat.choose class_a_male 1 * Nat.choose class_a_female 1 * Nat.choose class_b_male 2 +
  Nat.choose class_a_male 2 * Nat.choose class_b_male 1 * Nat.choose class_b_female 1

/-- Theorem stating that the number of ways to select 4 students from two classes, 
    with 2 students from each class, such that exactly 1 female student is among them, 
    is equal to 345, given the specific class compositions. -/
theorem select_students_result : select_students 5 3 6 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_select_students_result_l110_11024


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l110_11096

theorem quadratic_inequality_range (x : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc 2 4 ∧ a * x^2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Ioi (-1) ∪ Set.Iio (3/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l110_11096


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l110_11001

theorem cube_root_of_eight :
  (8 : ℝ) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l110_11001


namespace NUMINAMATH_CALUDE_tens_digit_of_6_to_19_l110_11098

-- Define a function to calculate the tens digit
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

-- State the theorem
theorem tens_digit_of_6_to_19 :
  tens_digit (6^19) = 1 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_to_19_l110_11098


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l110_11090

open Complex

theorem pure_imaginary_condition (θ : ℝ) :
  let Z : ℂ := 1 / (sin θ + cos θ * I) - (1 : ℂ) / 2
  (∃ y : ℝ, Z = y * I) →
  (∃ k : ℤ, θ = π / 6 + 2 * k * π ∨ θ = 5 * π / 6 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l110_11090


namespace NUMINAMATH_CALUDE_f_max_value_l110_11079

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + x + 1)

theorem f_max_value (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1/2 → ∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ Real.exp (-1/a)) ∧
  (a > 1/2 → ∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ Real.exp (-2) * (4*a - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l110_11079


namespace NUMINAMATH_CALUDE_circle_equation_with_center_and_tangent_line_l110_11021

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ = a * sin(θ - α) + b -/
structure PolarLine where
  a : ℝ
  α : ℝ
  b : ℝ

/-- Represents a circle in polar form ρ = R * sin(θ - β) -/
structure PolarCircle where
  R : ℝ
  β : ℝ

def is_tangent (c : PolarCircle) (l : PolarLine) : Prop :=
  sorry

theorem circle_equation_with_center_and_tangent_line 
  (P : PolarPoint) 
  (l : PolarLine) 
  (h1 : P.r = 2 ∧ P.θ = π/3) 
  (h2 : l.a = 1 ∧ l.α = π/3 ∧ l.b = 2) : 
  ∃ (c : PolarCircle), c.R = 4 ∧ c.β = -π/6 ∧ is_tangent c l :=
sorry

end NUMINAMATH_CALUDE_circle_equation_with_center_and_tangent_line_l110_11021


namespace NUMINAMATH_CALUDE_xy_value_l110_11041

theorem xy_value (x y : ℝ) (h : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) - 2) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l110_11041


namespace NUMINAMATH_CALUDE_expected_value_binomial_l110_11088

/-- The number of missile launches -/
def n : ℕ := 10

/-- The probability of an accident in a single launch -/
def p : ℝ := 0.01

/-- The random variable representing the number of accidents -/
def ξ : Nat → ℝ := sorry

theorem expected_value_binomial :
  Finset.sum (Finset.range (n + 1)) (fun k => k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)) = n * p :=
sorry

end NUMINAMATH_CALUDE_expected_value_binomial_l110_11088


namespace NUMINAMATH_CALUDE_leak_empty_time_l110_11043

/-- Represents the time it takes for a leak to empty a full tank, given the filling times with and without the leak. -/
theorem leak_empty_time (fill_time : ℝ) (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) : 
  fill_time > 0 ∧ fill_time_with_leak > fill_time →
  (1 / fill_time) - (1 / fill_time_with_leak) = 1 / leak_empty_time →
  fill_time = 6 →
  fill_time_with_leak = 9 →
  leak_empty_time = 18 := by
sorry

end NUMINAMATH_CALUDE_leak_empty_time_l110_11043


namespace NUMINAMATH_CALUDE_theater_admission_revenue_l110_11067

/-- Calculates the total amount collected from theater admission tickets. -/
theorem theater_admission_revenue
  (total_persons : ℕ)
  (num_children : ℕ)
  (adult_price : ℚ)
  (child_price : ℚ)
  (h1 : total_persons = 280)
  (h2 : num_children = 80)
  (h3 : adult_price = 60 / 100)
  (h4 : child_price = 25 / 100) :
  (total_persons - num_children) * adult_price + num_children * child_price = 140 / 100 := by
  sorry

end NUMINAMATH_CALUDE_theater_admission_revenue_l110_11067


namespace NUMINAMATH_CALUDE_smallest_divisor_is_number_itself_l110_11087

def form_number (a b : Nat) (digit : Nat) : Nat :=
  a * 1000 + digit * 100 + b

theorem smallest_divisor_is_number_itself :
  let complete_number := form_number 761 829 3
  complete_number % complete_number = 0 ∧
  ∀ d : Nat, d > 0 ∧ d < complete_number → complete_number % d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_is_number_itself_l110_11087


namespace NUMINAMATH_CALUDE_rectangle_length_equals_two_l110_11064

theorem rectangle_length_equals_two (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 4 →
  rect_width = 8 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_two_l110_11064


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l110_11044

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 →
  a * b + b * c + c * d + d * a ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l110_11044


namespace NUMINAMATH_CALUDE_tan_is_periodic_l110_11014

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define the property of being periodic
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- State the theorem
theorem tan_is_periodic : is_periodic tan π := by
  sorry

end NUMINAMATH_CALUDE_tan_is_periodic_l110_11014


namespace NUMINAMATH_CALUDE_fraction_equality_l110_11038

theorem fraction_equality (a b : ℝ) (h : a ≠ b) :
  (a^2 - b^2) / (a - b)^2 = (a + b) / (a - b) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l110_11038


namespace NUMINAMATH_CALUDE_walts_age_l110_11017

theorem walts_age (walt_age music_teacher_age : ℕ) : 
  music_teacher_age = 3 * walt_age →
  (music_teacher_age + 12) = 2 * (walt_age + 12) →
  walt_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_walts_age_l110_11017


namespace NUMINAMATH_CALUDE_product_not_in_set_l110_11022

def a (n : ℕ) : ℕ := n^2 + n + 1

theorem product_not_in_set : ∃ m k : ℕ, ¬∃ n : ℕ, a m * a k = a n := by
  sorry

end NUMINAMATH_CALUDE_product_not_in_set_l110_11022


namespace NUMINAMATH_CALUDE_james_writing_speed_l110_11095

/-- James writes some pages an hour. -/
def pages_per_hour : ℝ := sorry

/-- James writes 5 pages a day to 2 different people. -/
def pages_per_day : ℝ := 5 * 2

/-- James spends 7 hours a week writing. -/
def hours_per_week : ℝ := 7

/-- The number of days in a week. -/
def days_per_week : ℝ := 7

theorem james_writing_speed :
  pages_per_hour = 10 :=
sorry

end NUMINAMATH_CALUDE_james_writing_speed_l110_11095


namespace NUMINAMATH_CALUDE_gizmos_produced_l110_11005

/-- Represents the production scenario in a factory -/
structure ProductionScenario where
  a : ℝ  -- Time to produce a gadget
  b : ℝ  -- Time to produce a gizmo

/-- Checks if the production scenario satisfies the given conditions -/
def satisfies_conditions (s : ProductionScenario) : Prop :=
  s.a ≥ 0 ∧ s.b ≥ 0 ∧  -- Non-negative production times
  450 * s.a + 300 * s.b = 150 ∧  -- 150 workers in 1 hour
  360 * s.a + 450 * s.b = 180 ∧  -- 90 workers in 2 hours
  300 * s.a = 300  -- 75 workers produce 300 gadgets in 4 hours

/-- Theorem stating the number of gizmos produced by 75 workers in 4 hours -/
theorem gizmos_produced (s : ProductionScenario) 
  (h : satisfies_conditions s) : 
  75 * 4 / s.b = 150 := by
  sorry


end NUMINAMATH_CALUDE_gizmos_produced_l110_11005


namespace NUMINAMATH_CALUDE_total_money_is_250_l110_11028

/-- The amount of money James owns -/
def james_money : ℕ := 145

/-- The difference between James' and Ali's money -/
def difference : ℕ := 40

/-- The amount of money Ali owns -/
def ali_money : ℕ := james_money - difference

/-- The total amount of money owned by James and Ali -/
def total_money : ℕ := james_money + ali_money

theorem total_money_is_250 : total_money = 250 := by sorry

end NUMINAMATH_CALUDE_total_money_is_250_l110_11028


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l110_11015

theorem regular_polygon_sides (exterior_angle : ℝ) :
  exterior_angle = 40 →
  (∃ n : ℕ, n * exterior_angle = 360 ∧ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l110_11015


namespace NUMINAMATH_CALUDE_math_books_count_l110_11099

/-- Given a shelf of books with the following properties:
  * There are 100 books in total
  * 32 of them are history books
  * 25 of them are geography books
  * The rest are math books
  This theorem proves that there are 43 math books. -/
theorem math_books_count (total : ℕ) (history : ℕ) (geography : ℕ) (math : ℕ) 
  (h_total : total = 100)
  (h_history : history = 32)
  (h_geography : geography = 25)
  (h_sum : total = history + geography + math) :
  math = 43 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l110_11099


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l110_11013

theorem sum_of_coefficients_equals_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + 2*x)^7 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + 
                      a₄*(1-x)^4 + a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l110_11013


namespace NUMINAMATH_CALUDE_exists_solution_with_y_seven_l110_11016

theorem exists_solution_with_y_seven :
  ∃ (x y z t : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    x + y + z + t = 10 ∧
    y = 7 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_with_y_seven_l110_11016


namespace NUMINAMATH_CALUDE_circle_area_equals_circumference_squared_l110_11032

theorem circle_area_equals_circumference_squared : 
  ∀ (r : ℝ), r > 0 → 2 * (π * r^2 / 2) = (2 * π * r)^2 / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equals_circumference_squared_l110_11032


namespace NUMINAMATH_CALUDE_weighted_average_calculation_l110_11033

theorem weighted_average_calculation (math_score math_weight history_score history_weight third_weight target_average : ℚ)
  (h1 : math_score = 72 / 100)
  (h2 : math_weight = 50 / 100)
  (h3 : history_score = 84 / 100)
  (h4 : history_weight = 30 / 100)
  (h5 : third_weight = 20 / 100)
  (h6 : target_average = 75 / 100)
  (h7 : math_weight + history_weight + third_weight ≤ 1) :
  ∃ (third_score fourth_weight : ℚ),
    third_score = 69 / 100 ∧
    fourth_weight = 0 ∧
    math_weight + history_weight + third_weight + fourth_weight = 1 ∧
    math_score * math_weight + history_score * history_weight + third_score * third_weight = target_average :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_calculation_l110_11033


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l110_11027

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (3 : ℚ) / 99 + (4 : ℚ) / 9999 = (843 : ℚ) / 3333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l110_11027


namespace NUMINAMATH_CALUDE_min_k_theorem_l110_11061

/-- The set S of powers of 1996 -/
def S : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 1996^m}

/-- Definition of a valid sequence pair -/
def ValidSequencePair (k : ℕ) (a b : ℕ → ℕ) : Prop :=
  (∀ i ∈ Finset.range k, a i ∈ S ∧ b i ∈ S) ∧
  (∀ i ∈ Finset.range k, a i ≠ b i) ∧
  (∀ i ∈ Finset.range (k-1), a i ≤ a (i+1) ∧ b i ≤ b (i+1)) ∧
  (Finset.sum (Finset.range k) a = Finset.sum (Finset.range k) b)

/-- The theorem stating the minimum k -/
theorem min_k_theorem :
  (∃ k : ℕ, ∃ a b : ℕ → ℕ, ValidSequencePair k a b) ∧
  (∀ k < 1997, ¬∃ a b : ℕ → ℕ, ValidSequencePair k a b) ∧
  (∃ a b : ℕ → ℕ, ValidSequencePair 1997 a b) :=
sorry

end NUMINAMATH_CALUDE_min_k_theorem_l110_11061


namespace NUMINAMATH_CALUDE_train_length_proof_l110_11070

/-- Given a train with constant speed that crosses two platforms of different lengths,
    prove that the length of the train is 110 meters. -/
theorem train_length_proof (speed : ℝ) (length : ℝ) :
  speed > 0 →
  speed * 15 = length + 160 →
  speed * 20 = length + 250 →
  length = 110 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l110_11070


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l110_11029

open Set

theorem complement_intersection_problem (I A B : Set ℕ) : 
  I = {0, 1, 2, 3, 4} →
  A = {0, 2, 3} →
  B = {1, 3, 4} →
  (I \ A) ∩ B = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l110_11029


namespace NUMINAMATH_CALUDE_triangle_side_minimization_l110_11060

theorem triangle_side_minimization (t C : ℝ) (ht : t > 0) (hC : 0 < C ∧ C < π) :
  let min_c := 2 * Real.sqrt (t * Real.tan (C / 2))
  ∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/2 * a * b * Real.sin C = t) →
    (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
    c ≥ min_c ∧
    (c = min_c ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_minimization_l110_11060


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l110_11004

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 6 + a 8 + a 10 = 80) →
  (a 1 + a 13 = 40) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l110_11004


namespace NUMINAMATH_CALUDE_justin_flower_gathering_l110_11085

def minutes_per_flower : ℕ := 10
def gathering_hours : ℕ := 2
def lost_flowers : ℕ := 3
def classmates : ℕ := 30

def additional_minutes_needed : ℕ :=
  let gathered_flowers := gathering_hours * 60 / minutes_per_flower
  let remaining_flowers := classmates - (gathered_flowers - lost_flowers)
  remaining_flowers * minutes_per_flower

theorem justin_flower_gathering :
  additional_minutes_needed = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_flower_gathering_l110_11085


namespace NUMINAMATH_CALUDE_cone_base_radius_l110_11011

/-- A cone with surface area 3π and lateral surface that unfolds into a semicircle has a base radius of 1 -/
theorem cone_base_radius (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  l = 2 * r →  -- Lateral surface unfolds into a semicircle
  3 * π * r^2 = 3 * π →  -- Surface area is 3π
  r = 1 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l110_11011


namespace NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l110_11000

/-- Given a cylinder with height H and base radius R, and a pyramid inside the cylinder
    with its height coinciding with the cylinder's slant height, and its base being an
    isosceles triangle ABC inscribed in the cylinder's base with ∠A = 120°,
    the lateral surface area of the pyramid is (R/4) * (4H + √(3R² + 12H²)). -/
theorem pyramid_lateral_surface_area 
  (H R : ℝ) 
  (H_pos : H > 0) 
  (R_pos : R > 0) : 
  ∃ (pyramid_area : ℝ), 
    pyramid_area = (R / 4) * (4 * H + Real.sqrt (3 * R^2 + 12 * H^2)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l110_11000


namespace NUMINAMATH_CALUDE_clubs_distribution_l110_11066

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A gets the clubs"
def A_gets_clubs (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B gets the clubs"
def B_gets_clubs (d : Distribution) : Prop := d Person.B = Card.Clubs

-- State the theorem
theorem clubs_distribution :
  (∀ d : Distribution, ¬(A_gets_clubs d ∧ B_gets_clubs d)) ∧ 
  (∃ d : Distribution, ¬A_gets_clubs d ∧ ¬B_gets_clubs d) :=
sorry

end NUMINAMATH_CALUDE_clubs_distribution_l110_11066


namespace NUMINAMATH_CALUDE_arrangement_of_cards_l110_11030

def number_of_arrangements (total_cards : ℕ) (interchangeable_cards : ℕ) : ℕ :=
  (total_cards.factorial) / (interchangeable_cards.factorial)

theorem arrangement_of_cards : number_of_arrangements 15 13 = 210 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_of_cards_l110_11030


namespace NUMINAMATH_CALUDE_prob_product_div_by_3_l110_11002

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a number not divisible by 3 on one die -/
def prob_not_div_by_3 : ℚ := 2/3

/-- The probability that the product of the numbers rolled on 5 dice is divisible by 3 -/
theorem prob_product_div_by_3 : 
  (1 - prob_not_div_by_3 ^ num_dice) = 211/243 := by sorry

end NUMINAMATH_CALUDE_prob_product_div_by_3_l110_11002


namespace NUMINAMATH_CALUDE_prob_more_heads_ten_coins_l110_11035

/-- The number of coins being flipped -/
def n : ℕ := 10

/-- The probability of getting heads on a single coin flip -/
def p : ℚ := 1/2

/-- The probability of getting more heads than tails when flipping n fair coins -/
def prob_more_heads (n : ℕ) (p : ℚ) : ℚ :=
  1/2 * (1 - (n.choose (n/2)) / (2^n))

theorem prob_more_heads_ten_coins :
  prob_more_heads n p = 193/512 := by
  sorry

#eval prob_more_heads n p

end NUMINAMATH_CALUDE_prob_more_heads_ten_coins_l110_11035


namespace NUMINAMATH_CALUDE_find_x_given_exponential_equation_l110_11009

theorem find_x_given_exponential_equation : ∃ x : ℝ, (2 : ℝ)^(x - 4) = 4^2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_x_given_exponential_equation_l110_11009


namespace NUMINAMATH_CALUDE_largest_divisor_of_10000_l110_11086

theorem largest_divisor_of_10000 :
  ∀ n : ℕ, n ∣ 10000 ∧ ¬(n ∣ 9999) → n ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_10000_l110_11086


namespace NUMINAMATH_CALUDE_angle_value_l110_11042

theorem angle_value (a : ℝ) : 
  (180 - a = 3 * (90 - a)) → a = 45 := by sorry

end NUMINAMATH_CALUDE_angle_value_l110_11042


namespace NUMINAMATH_CALUDE_pet_food_price_l110_11063

/-- Given a manufacturer's suggested retail price and discount conditions, prove the price is $35 -/
theorem pet_food_price (M : ℝ) : 
  (M * (1 - 0.3) * (1 - 0.2) = 19.6) → M = 35 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_price_l110_11063


namespace NUMINAMATH_CALUDE_solve_system_l110_11068

theorem solve_system (x y z : ℚ) 
  (eq1 : x - y - z = 8)
  (eq2 : x + y + z = 20)
  (eq3 : x - y + 2*z = 16) :
  z = 8/3 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l110_11068


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l110_11031

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l110_11031


namespace NUMINAMATH_CALUDE_last_number_is_one_l110_11006

/-- A sequence of 1999 numbers with specific properties -/
def SpecialSequence : Type :=
  { a : Fin 1999 → ℤ // 
    a 0 = 1 ∧ 
    ∀ i : Fin 1997, a (i + 1) = a i + a (i + 2) }

/-- The last number in the SpecialSequence is 1 -/
theorem last_number_is_one (seq : SpecialSequence) : 
  seq.val (Fin.last 1998) = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_one_l110_11006


namespace NUMINAMATH_CALUDE_difference_of_squares_l110_11036

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l110_11036


namespace NUMINAMATH_CALUDE_green_beads_count_l110_11049

/-- The number of white beads in each necklace -/
def white_beads : ℕ := 6

/-- The number of orange beads in each necklace -/
def orange_beads : ℕ := 3

/-- The maximum number of necklaces that can be made -/
def max_necklaces : ℕ := 5

/-- The total number of beads available for each color -/
def total_beads : ℕ := 45

/-- The number of green beads in each necklace -/
def green_beads : ℕ := 9

theorem green_beads_count : 
  white_beads * max_necklaces ≤ total_beads ∧ 
  orange_beads * max_necklaces ≤ total_beads ∧ 
  green_beads * max_necklaces = total_beads := by
  sorry

end NUMINAMATH_CALUDE_green_beads_count_l110_11049


namespace NUMINAMATH_CALUDE_no_real_a_for_unique_solution_l110_11008

theorem no_real_a_for_unique_solution : ¬∃ a : ℝ, ∃! x : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_a_for_unique_solution_l110_11008


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l110_11046

theorem simplify_product_of_radicals (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (30 * x) = 30 * x * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l110_11046


namespace NUMINAMATH_CALUDE_set_operations_l110_11056

def I : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 5, 6, 7}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7}) ∧
  (A ∩ (I \ B) = {1, 2, 4}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l110_11056


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l110_11023

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  1043 = 23 * q + r ∧ 
  r < 23 ∧
  ∀ (q' r' : ℕ), 1043 = 23 * q' + r' ∧ r' < 23 → q' - r' ≤ q - r :=
by sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l110_11023


namespace NUMINAMATH_CALUDE_smallest_max_sum_l110_11019

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 4020) : 
  (∃ (a' b' c' d' e' f' : ℕ+), 
    a' + b' + c' + d' + e' + f' = 4020 ∧ 
    max (a' + b') (max (b' + c') (max (c' + d') (max (d' + e') (e' + f')))) = 805) ∧
  (∀ (a'' b'' c'' d'' e'' f'' : ℕ+),
    a'' + b'' + c'' + d'' + e'' + f'' = 4020 →
    max (a'' + b'') (max (b'' + c'') (max (c'' + d'') (max (d'' + e'') (e'' + f'')))) ≥ 805) :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l110_11019


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l110_11081

theorem necessary_but_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b c : ℝ, a > b → a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l110_11081


namespace NUMINAMATH_CALUDE_jogging_distance_l110_11003

/-- Alice's jogging speed in miles per minute -/
def alice_speed : ℚ := 1 / 12

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Total jogging time in minutes -/
def total_time : ℕ := 120

/-- The distance between Alice and Bob after jogging for the total time -/
def distance_apart : ℚ := alice_speed * total_time + bob_speed * total_time

theorem jogging_distance : distance_apart = 19 := by sorry

end NUMINAMATH_CALUDE_jogging_distance_l110_11003


namespace NUMINAMATH_CALUDE_fisherman_daily_earnings_l110_11074

/-- Calculates the daily earnings of a fisherman based on their catch and fish prices -/
theorem fisherman_daily_earnings (red_snapper_count : ℕ) (tuna_count : ℕ) (red_snapper_price : ℕ) (tuna_price : ℕ) : 
  red_snapper_count = 8 → 
  tuna_count = 14 → 
  red_snapper_price = 3 → 
  tuna_price = 2 → 
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 := by
sorry

end NUMINAMATH_CALUDE_fisherman_daily_earnings_l110_11074


namespace NUMINAMATH_CALUDE_meeting_percentage_theorem_l110_11059

def work_day_hours : ℝ := 10
def first_meeting_minutes : ℝ := 45
def second_meeting_multiplier : ℝ := 3

def total_meeting_time : ℝ := first_meeting_minutes + second_meeting_multiplier * first_meeting_minutes
def work_day_minutes : ℝ := work_day_hours * 60

theorem meeting_percentage_theorem :
  (total_meeting_time / work_day_minutes) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_meeting_percentage_theorem_l110_11059


namespace NUMINAMATH_CALUDE_second_month_sale_l110_11069

def sales_data : List ℕ := [8435, 8855, 9230, 8562, 6991]
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem second_month_sale :
  let total_sale := average_sale * num_months
  let known_sales_sum := sales_data.sum
  let second_month_sale := total_sale - known_sales_sum
  second_month_sale = 8927 := by sorry

end NUMINAMATH_CALUDE_second_month_sale_l110_11069


namespace NUMINAMATH_CALUDE_correct_substitution_l110_11054

/-- Given a system of equations { y = 1 - x, x - 2y = 4 }, 
    the correct substitution using the substitution method is x - 2 + 2x = 4 -/
theorem correct_substitution (x y : ℝ) : 
  (y = 1 - x ∧ x - 2*y = 4) → (x - 2 + 2*x = 4) :=
by sorry

end NUMINAMATH_CALUDE_correct_substitution_l110_11054


namespace NUMINAMATH_CALUDE_expression_greater_than_e_l110_11053

theorem expression_greater_than_e (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  Real.exp y - 8/x > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_expression_greater_than_e_l110_11053


namespace NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l110_11058

theorem consecutive_integers_product_336_sum_21 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l110_11058


namespace NUMINAMATH_CALUDE_proportionality_coefficient_l110_11018

/-- Given variables x y z : ℝ and a constant k : ℕ+, prove that k = 3 -/
theorem proportionality_coefficient (x y z : ℝ) (k : ℕ+) : 
  (z - y = k * x) →  -- The difference of z and y is proportional to x
  (x - z = k * y) →  -- The difference of x and z is proportional to y
  (∃ (x' y' z' : ℝ), z' = (5/3) * (x' - y')) →  -- A certain value of z is 5/3 times the difference of x and y
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_proportionality_coefficient_l110_11018


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1547_l110_11025

theorem smallest_prime_factor_of_1547 :
  Nat.minFac 1547 = 7 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1547_l110_11025


namespace NUMINAMATH_CALUDE_max_goals_scored_l110_11092

/-- Represents the number of goals scored by Marlon in a soccer game --/
def goals_scored (penalty_shots free_kicks : ℕ) : ℝ :=
  0.4 * penalty_shots + 0.5 * free_kicks

/-- Proves that the maximum number of goals Marlon could have scored is 20 --/
theorem max_goals_scored : 
  ∀ penalty_shots free_kicks : ℕ, 
  penalty_shots + free_kicks = 40 →
  goals_scored penalty_shots free_kicks ≤ 20 :=
by
  sorry

#check max_goals_scored

end NUMINAMATH_CALUDE_max_goals_scored_l110_11092


namespace NUMINAMATH_CALUDE_total_volume_of_four_cubes_l110_11084

theorem total_volume_of_four_cubes (edge_length : ℝ) (num_cubes : ℕ) : 
  edge_length = 5 → num_cubes = 4 → num_cubes * (edge_length ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_four_cubes_l110_11084


namespace NUMINAMATH_CALUDE_coloring_properties_l110_11077

/-- A coloring of natural numbers with N colors. -/
def Coloring (N : ℕ) := ℕ → Fin N

/-- Property that there are infinitely many numbers of each color. -/
def InfinitelyMany (c : Coloring N) : Prop :=
  ∀ (k : Fin N), ∀ (m : ℕ), ∃ (n : ℕ), n > m ∧ c n = k

/-- Property that the color of the half-sum of two different numbers of the same parity
    depends only on the colors of the summands. -/
def HalfSumProperty (c : Coloring N) : Prop :=
  ∀ (a b x y : ℕ), a ≠ b → x ≠ y → a % 2 = b % 2 → x % 2 = y % 2 →
    c a = c x → c b = c y → c ((a + b) / 2) = c ((x + y) / 2)

/-- Main theorem about the properties of the coloring. -/
theorem coloring_properties (N : ℕ) (c : Coloring N)
    (h1 : InfinitelyMany c) (h2 : HalfSumProperty c) :
  (∀ (a b : ℕ), a % 2 = b % 2 → c a = c b → c ((a + b) / 2) = c a) ∧
  (∃ (coloring : Coloring N), InfinitelyMany coloring ∧ HalfSumProperty coloring ↔ N % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coloring_properties_l110_11077


namespace NUMINAMATH_CALUDE_female_students_count_l110_11091

theorem female_students_count (x : ℕ) : 
  (8 * x < 200) → 
  (9 * x > 200) → 
  (11 * (x + 4) > 300) → 
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_female_students_count_l110_11091


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l110_11073

/-- Given two real numbers with sum S, prove that adding 5 to each and then tripling results in a sum of 3S + 30 -/
theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry


end NUMINAMATH_CALUDE_final_sum_after_operations_l110_11073


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_zero_l110_11065

/-- Two lines in the coordinate plane -/
structure Line where
  slope : ℝ
  intercept : ℝ
  is_x_intercept : Bool

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- Theorem: The sum of coordinates of the intersection point is 0 -/
theorem intersection_coordinate_sum_zero :
  let line_a : Line := ⟨-1, 2, true⟩
  let line_b : Line := ⟨5, -10, false⟩
  let (a, b) := intersection line_a line_b
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_coordinate_sum_zero_l110_11065


namespace NUMINAMATH_CALUDE_tantrix_impossibility_l110_11080

/-- Represents a tile in the Tantrix Solitaire game -/
structure Tile where
  blue_lines : Nat
  red_lines : Nat

/-- Represents the game board -/
structure Board where
  tiles : List Tile
  blue_loop : Bool
  no_gaps : Bool
  red_intersections : Nat

/-- Checks if a board configuration is valid -/
def is_valid_board (b : Board) : Prop :=
  b.tiles.length = 13 ∧ b.blue_loop ∧ b.no_gaps ∧ b.red_intersections = 3

/-- Theorem stating the impossibility of arranging 13 tiles to form a valid board -/
theorem tantrix_impossibility : ¬ ∃ (b : Board), is_valid_board b := by
  sorry

end NUMINAMATH_CALUDE_tantrix_impossibility_l110_11080


namespace NUMINAMATH_CALUDE_expression_simplification_l110_11083

theorem expression_simplification :
  (-8 : ℚ) * (18 / 14) * (49 / 27) + 4 / 3 = -52 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l110_11083


namespace NUMINAMATH_CALUDE_brick_surface_area_l110_11075

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 8 cm x 6 cm x 2 cm brick is 152 cm² -/
theorem brick_surface_area :
  surface_area 8 6 2 = 152 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l110_11075
