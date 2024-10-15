import Mathlib

namespace NUMINAMATH_CALUDE_point_coordinate_sum_l1275_127550

/-- Given points P and Q, where P is at the origin and Q is on the line y = 6,
    if the slope of PQ is 3/4, then the sum of Q's coordinates is 14. -/
theorem point_coordinate_sum (x : ℝ) : 
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (x, 6)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 3/4 → Q.1 + Q.2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l1275_127550


namespace NUMINAMATH_CALUDE_smallest_palindrome_base2_base4_l1275_127527

/-- Convert a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBinaryAux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBinaryAux (m / 2) ((m % 2) :: acc)
  toBinaryAux n []

/-- Convert a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBase4Aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBase4Aux (m / 4) ((m % 4) :: acc)
  toBase4Aux n []

/-- Check if a list is a palindrome -/
def isPalindrome (l : List ℕ) : Prop :=
  l = l.reverse

/-- The main theorem statement -/
theorem smallest_palindrome_base2_base4 :
  ∀ n : ℕ, n > 10 →
  (isPalindrome (toBinary n) ∧ isPalindrome (toBase4 n)) →
  n ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_palindrome_base2_base4_l1275_127527


namespace NUMINAMATH_CALUDE_correct_operation_l1275_127590

theorem correct_operation (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = -x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1275_127590


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1275_127523

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  a 3 = 4 →
  a 7 = 12 →
  a 11 = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1275_127523


namespace NUMINAMATH_CALUDE_smallest_possible_a_l1275_127561

theorem smallest_possible_a (a b c d x : ℤ) 
  (h1 : (a - 2*b) * x = 1)
  (h2 : (b - 3*c) * x = 1)
  (h3 : (c - 4*d) * x = 1)
  (h4 : x + 100 = d)
  (h5 : x > 0) :
  a ≥ 2433 ∧ ∃ (a₀ b₀ c₀ d₀ x₀ : ℤ), 
    a₀ = 2433 ∧
    (a₀ - 2*b₀) * x₀ = 1 ∧
    (b₀ - 3*c₀) * x₀ = 1 ∧
    (c₀ - 4*d₀) * x₀ = 1 ∧
    x₀ + 100 = d₀ ∧
    x₀ > 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l1275_127561


namespace NUMINAMATH_CALUDE_candy_distribution_l1275_127572

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 42 → num_bags = 2 → total_candy = num_bags * candy_per_bag → candy_per_bag = 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1275_127572


namespace NUMINAMATH_CALUDE_checkerboard_ratio_l1275_127580

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The sum of squares from 1 to n -/
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles on an n x n checkerboard -/
def num_rectangles (n : ℕ) : ℕ := (choose_2 (n + 1)) ^ 2

/-- The number of squares on an n x n checkerboard -/
def num_squares (n : ℕ) : ℕ := sum_squares n

theorem checkerboard_ratio :
  (num_squares 9 : ℚ) / (num_rectangles 9 : ℚ) = 19 / 135 := by sorry

end NUMINAMATH_CALUDE_checkerboard_ratio_l1275_127580


namespace NUMINAMATH_CALUDE_total_boys_in_class_l1275_127594

/-- Given a circular arrangement of students, if the 10th and 40th positions
    are opposite each other and only every other student is counted,
    then the total number of boys in the class is 30. -/
theorem total_boys_in_class (n : ℕ) 
  (circular_arrangement : n > 0)
  (opposite_positions : 40 - 10 = n / 2)
  (count_every_other : n % 2 = 0) : 
  n / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_boys_in_class_l1275_127594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1275_127531

/-- An arithmetic sequence with first term 3 and sum of second and third terms 12 has second term equal to 5 -/
theorem arithmetic_sequence_second_term (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 3 →                                -- first term is 3
  a 2 + a 3 = 12 →                         -- sum of second and third terms is 12
  a 2 = 5 :=                               -- second term is 5
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1275_127531


namespace NUMINAMATH_CALUDE_population_below_five_percent_in_five_years_population_below_five_percent_in_2011_l1275_127569

/-- Represents the yearly decrease rate of the sparrow population -/
def yearly_decrease_rate : ℝ := 0.5

/-- Represents the target percentage of the original population -/
def target_percentage : ℝ := 0.05

/-- Calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (yearly_decrease_rate ^ years)

/-- Theorem: It takes 5 years for the population to become less than 5% of the original -/
theorem population_below_five_percent_in_five_years (initial_population : ℝ) 
  (h : initial_population > 0) : 
  population_after_years initial_population 5 < target_percentage * initial_population ∧
  ∀ n : ℕ, n < 5 → population_after_years initial_population n ≥ target_percentage * initial_population :=
by sorry

/-- The year when the population becomes less than 5% of the original -/
def year_below_five_percent : ℕ := 2011

/-- Theorem: The population becomes less than 5% of the original in 2011 -/
theorem population_below_five_percent_in_2011 (initial_year : ℕ) (h : initial_year = 2006) :
  year_below_five_percent - initial_year = 5 :=
by sorry

end NUMINAMATH_CALUDE_population_below_five_percent_in_five_years_population_below_five_percent_in_2011_l1275_127569


namespace NUMINAMATH_CALUDE_circle_equation_l1275_127586

/-- Given a circle with center at (-2, 3) and tangent to the y-axis, 
    its equation is (x+2)^2+(y-3)^2=4 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (-2, 3)
  let tangent_to_y_axis : ℝ → Prop := λ r => r = 2
  tangent_to_y_axis (abs center.1) →
  (x + 2)^2 + (y - 3)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l1275_127586


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1275_127562

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 5*x - 14

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 7}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1275_127562


namespace NUMINAMATH_CALUDE_problem_statement_l1275_127508

theorem problem_statement (a b : ℝ) 
  (h1 : 0 < (1 : ℝ) / a) 
  (h2 : (1 : ℝ) / a < (1 : ℝ) / b) 
  (h3 : (1 : ℝ) / b < 1) 
  (h4 : Real.log a * Real.log b = 1) : 
  (2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ 
  a * b > Real.exp 2 ∧ 
  Real.exp (a - b) > a / b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1275_127508


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1275_127540

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 15*x + 36 = 0 → 
  ∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ x^2 - 15*x + 36 = (x - s₁) * (x - s₂) ∧ 
  (1 / s₁ + 1 / s₂ = 5 / 12) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1275_127540


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l1275_127592

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem: The fraction of upgraded sensors on the satellite is 1/4 -/
theorem upgraded_fraction_is_one_fourth (s : Satellite) 
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.total_upgraded / 8) :
  upgraded_fraction s = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l1275_127592


namespace NUMINAMATH_CALUDE_average_growth_rate_equation_l1275_127593

/-- Represents the average monthly growth rate as a real number between 0 and 1 -/
def average_growth_rate : ℝ := sorry

/-- The initial output value in January in billions of yuan -/
def initial_output : ℝ := 50

/-- The final output value in March in billions of yuan -/
def final_output : ℝ := 60

/-- The number of months between January and March -/
def months : ℕ := 2

theorem average_growth_rate_equation :
  initial_output * (1 + average_growth_rate) ^ months = final_output :=
sorry

end NUMINAMATH_CALUDE_average_growth_rate_equation_l1275_127593


namespace NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_174_main_result_l1275_127547

theorem least_number_with_remainder (n : ℕ) : 
  (n % 34 = 4 ∧ n % 5 = 4) → n ≥ 174 := by
  sorry

theorem least_number_is_174 : 
  174 % 34 = 4 ∧ 174 % 5 = 4 := by
  sorry

theorem main_result : 
  ∀ n : ℕ, (n % 34 = 4 ∧ n % 5 = 4) → n ≥ 174 ∧ (174 % 34 = 4 ∧ 174 % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_least_number_is_174_main_result_l1275_127547


namespace NUMINAMATH_CALUDE_sandwich_count_l1275_127571

/-- Represents the number of days in the workweek -/
def workweek_days : ℕ := 6

/-- Represents the cost of a donut in cents -/
def donut_cost : ℕ := 80

/-- Represents the cost of a sandwich in cents -/
def sandwich_cost : ℕ := 120

/-- Represents the condition that the total expenditure is an exact number of dollars -/
def is_exact_dollar_amount (sandwiches : ℕ) : Prop :=
  ∃ (dollars : ℕ), sandwich_cost * sandwiches + donut_cost * (workweek_days - sandwiches) = 100 * dollars

theorem sandwich_count : 
  ∃! (sandwiches : ℕ), sandwiches ≤ workweek_days ∧ is_exact_dollar_amount sandwiches ∧ sandwiches = 3 :=
sorry

end NUMINAMATH_CALUDE_sandwich_count_l1275_127571


namespace NUMINAMATH_CALUDE_picture_frame_perimeter_l1275_127589

theorem picture_frame_perimeter (width height : ℕ) (h1 : width = 6) (h2 : height = 9) :
  2 * width + 2 * height = 30 :=
by sorry

end NUMINAMATH_CALUDE_picture_frame_perimeter_l1275_127589


namespace NUMINAMATH_CALUDE_black_socks_bought_is_12_l1275_127510

/-- The number of pairs of black socks Dmitry bought -/
def black_socks_bought : ℕ := sorry

/-- The initial number of blue sock pairs -/
def initial_blue : ℕ := 14

/-- The initial number of black sock pairs -/
def initial_black : ℕ := 24

/-- The initial number of white sock pairs -/
def initial_white : ℕ := 10

/-- The total number of sock pairs after buying more black socks -/
def total_after : ℕ := initial_blue + initial_white + initial_black + black_socks_bought

/-- The number of black sock pairs after buying more -/
def black_after : ℕ := initial_black + black_socks_bought

theorem black_socks_bought_is_12 : 
  black_socks_bought = 12 ∧ 
  black_after = (3 : ℚ) / 5 * total_after := by sorry

end NUMINAMATH_CALUDE_black_socks_bought_is_12_l1275_127510


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l1275_127557

theorem isosceles_trapezoid_angles (a d : ℝ) : 
  -- The trapezoid is isosceles and angles form an arithmetic sequence
  a > 0 ∧ d > 0 ∧ 
  -- The sum of angles in a quadrilateral is 360°
  a + (a + d) + (a + 2*d) + 140 = 360 ∧ 
  -- The largest angle is 140°
  a + 3*d = 140 → 
  -- The smallest angle is 40°
  a = 40 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_angles_l1275_127557


namespace NUMINAMATH_CALUDE_painted_subcubes_count_l1275_127542

/-- Represents a cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (isPainted : ℕ → ℕ → ℕ → Bool)

/-- Counts subcubes with at least two painted faces -/
def countSubcubesWithTwoPaintedFaces (cube : PaintedCube) : ℕ :=
  sorry

/-- The main theorem -/
theorem painted_subcubes_count (cube : PaintedCube) 
  (h1 : cube.size = 4)
  (h2 : ∀ x y z, (x = 0 ∨ x = cube.size - 1 ∨ 
                  y = 0 ∨ y = cube.size - 1 ∨ 
                  z = 0 ∨ z = cube.size - 1) → 
                 cube.isPainted x y z = true) :
  countSubcubesWithTwoPaintedFaces cube = 32 :=
sorry

end NUMINAMATH_CALUDE_painted_subcubes_count_l1275_127542


namespace NUMINAMATH_CALUDE_rest_of_body_length_l1275_127513

theorem rest_of_body_length
  (total_height : ℝ)
  (leg_ratio : ℝ)
  (head_ratio : ℝ)
  (h1 : total_height = 60)
  (h2 : leg_ratio = 1 / 3)
  (h3 : head_ratio = 1 / 4)
  : total_height - (leg_ratio * total_height + head_ratio * total_height) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rest_of_body_length_l1275_127513


namespace NUMINAMATH_CALUDE_library_card_lineup_l1275_127524

theorem library_card_lineup : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_library_card_lineup_l1275_127524


namespace NUMINAMATH_CALUDE_project_duration_calculation_l1275_127579

/-- The number of weeks a project lasts based on breakfast expenses -/
def project_duration (people : ℕ) (days_per_week : ℕ) (meal_cost : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (people * days_per_week * meal_cost)

theorem project_duration_calculation :
  let people : ℕ := 4
  let days_per_week : ℕ := 5
  let meal_cost : ℚ := 4
  let total_spent : ℚ := 1280
  project_duration people days_per_week meal_cost total_spent = 16 := by
  sorry

end NUMINAMATH_CALUDE_project_duration_calculation_l1275_127579


namespace NUMINAMATH_CALUDE_equality_division_property_l1275_127503

theorem equality_division_property (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) :
  a / (c^2) = b / (c^2) := by sorry

end NUMINAMATH_CALUDE_equality_division_property_l1275_127503


namespace NUMINAMATH_CALUDE_certain_number_problem_l1275_127555

theorem certain_number_problem : ∃ x : ℝ, x * (5^4) = 70000 ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1275_127555


namespace NUMINAMATH_CALUDE_equation_solution_l1275_127541

theorem equation_solution (x y : ℝ) : 9 * x^2 - 25 * y^2 = 0 ↔ x = (5/3) * y ∨ x = -(5/3) * y := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1275_127541


namespace NUMINAMATH_CALUDE_gym_time_calculation_l1275_127598

/-- Calculates the total time spent at the gym per week -/
def gym_time_per_week (visits_per_week : ℕ) (weightlifting_time : ℝ) (warmup_cardio_ratio : ℝ) : ℝ :=
  visits_per_week * (weightlifting_time + warmup_cardio_ratio * weightlifting_time)

/-- Theorem: Given the specified gym routine, the total time spent at the gym per week is 4 hours -/
theorem gym_time_calculation :
  let visits_per_week : ℕ := 3
  let weightlifting_time : ℝ := 1
  let warmup_cardio_ratio : ℝ := 1/3
  gym_time_per_week visits_per_week weightlifting_time warmup_cardio_ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_gym_time_calculation_l1275_127598


namespace NUMINAMATH_CALUDE_v_4_value_l1275_127559

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecursiveSequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem v_4_value (v : ℕ → ℝ) (h_rec : RecursiveSequence v) 
    (h_v2 : v 2 = 7) (h_v5 : v 5 = 53) : v 4 = 22.6 := by
  sorry

end NUMINAMATH_CALUDE_v_4_value_l1275_127559


namespace NUMINAMATH_CALUDE_point_motion_l1275_127521

/-- Given two points A and B on a number line, prove properties about their motion and positions. -/
theorem point_motion (a b : ℝ) (h : |a + 20| + |b - 12| = 0) :
  -- 1. Initial positions
  (a = -20 ∧ b = 12) ∧ 
  -- 2. Time when A and B are equidistant from origin
  (∃ t : ℝ, t = 2 ∧ |a - 6*t| = |b - 2*t|) ∧
  -- 3. Times when A and B are 8 units apart
  (∃ t : ℝ, (t = 3 ∨ t = 5 ∨ t = 10) ∧ 
    |a - 6*t - (b - 2*t)| = 8) := by
  sorry

end NUMINAMATH_CALUDE_point_motion_l1275_127521


namespace NUMINAMATH_CALUDE_isosceles_when_neg_one_root_right_triangle_when_equal_roots_equilateral_roots_l1275_127570

/-- Triangle represented by side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Quadratic equation associated with a triangle -/
def triangleQuadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.b) * x^2 + 2 * t.c * x + (t.b - t.a)

theorem isosceles_when_neg_one_root (t : Triangle) :
  triangleQuadratic t (-1) = 0 → t.b = t.c := by sorry

theorem right_triangle_when_equal_roots (t : Triangle) :
  (2 * t.c)^2 = 4 * (t.a + t.b) * (t.b - t.a) → t.a^2 + t.c^2 = t.b^2 := by sorry

theorem equilateral_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∃ x : ℝ, triangleQuadratic t x = 0) →
  (triangleQuadratic t 0 = 0 ∧ triangleQuadratic t (-1) = 0) := by sorry

end NUMINAMATH_CALUDE_isosceles_when_neg_one_root_right_triangle_when_equal_roots_equilateral_roots_l1275_127570


namespace NUMINAMATH_CALUDE_correct_number_of_values_l1275_127549

theorem correct_number_of_values 
  (original_mean : ℝ) 
  (incorrect_value : ℝ) 
  (correct_value : ℝ) 
  (correct_mean : ℝ) 
  (h1 : original_mean = 190) 
  (h2 : incorrect_value = 130) 
  (h3 : correct_value = 165) 
  (h4 : correct_mean = 191.4) : 
  ∃ n : ℕ, n > 0 ∧ 
    n * original_mean + (correct_value - incorrect_value) = n * correct_mean ∧ 
    n = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_values_l1275_127549


namespace NUMINAMATH_CALUDE_not_p_or_q_is_false_l1275_127517

-- Define proposition p
def p : Prop := ∀ x : ℝ, (λ x : ℝ => x^3) (-x) = -((λ x : ℝ => x^3) x)

-- Define proposition q
def q : Prop := ∀ a b c : ℝ, b^2 = a*c → ∃ r : ℝ, (a = b/r ∧ b = c*r) ∨ (a = b*r ∧ b = c/r)

-- Theorem to prove
theorem not_p_or_q_is_false : ¬(¬p ∨ q) := by sorry

end NUMINAMATH_CALUDE_not_p_or_q_is_false_l1275_127517


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1275_127575

theorem exponent_multiplication (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1275_127575


namespace NUMINAMATH_CALUDE_games_in_division_is_sixty_l1275_127591

/-- Represents a baseball league with specified conditions -/
structure BaseballLeague where
  n : ℕ  -- Number of games against each team in the same division
  m : ℕ  -- Number of games against each team in the other division
  h1 : n > 2 * m
  h2 : m > 5
  h3 : 4 * n + 5 * m = 100

/-- The number of games a team plays within its own division -/
def gamesInDivision (league : BaseballLeague) : ℕ := 4 * league.n

theorem games_in_division_is_sixty (league : BaseballLeague) :
  gamesInDivision league = 60 := by
  sorry

#check games_in_division_is_sixty

end NUMINAMATH_CALUDE_games_in_division_is_sixty_l1275_127591


namespace NUMINAMATH_CALUDE_debby_soda_bottles_l1275_127564

/-- The number of soda bottles Debby drinks per day -/
def soda_per_day : ℕ := 9

/-- The number of days the soda bottles lasted -/
def days_lasted : ℕ := 40

/-- The total number of soda bottles Debby bought -/
def total_soda_bottles : ℕ := soda_per_day * days_lasted

/-- Theorem stating that the total number of soda bottles Debby bought is 360 -/
theorem debby_soda_bottles : total_soda_bottles = 360 := by
  sorry

end NUMINAMATH_CALUDE_debby_soda_bottles_l1275_127564


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_l1275_127501

theorem x_gt_one_sufficient_not_necessary :
  (∃ x : ℝ, x > 1 → (1 / x) < 1) ∧
  (∃ x : ℝ, (1 / x) < 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_l1275_127501


namespace NUMINAMATH_CALUDE_intersection_M_N_l1275_127520

-- Define the set M
def M : Set ℝ := {x | Real.log (1 - x) < 0}

-- Define the set N
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1275_127520


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_ratio_l1275_127567

/-- A hexahedron with equilateral triangle faces congruent to those of a regular octahedron -/
structure SpecialHexahedron where
  -- The faces are equilateral triangles
  faces_equilateral : Bool
  -- The faces are congruent to those of a regular octahedron
  faces_congruent_to_octahedron : Bool

/-- A regular octahedron -/
structure RegularOctahedron where

/-- The radius of the inscribed sphere in a polyhedron -/
def inscribed_sphere_radius (P : Type) : ℝ := sorry

/-- The theorem stating the ratio of inscribed sphere radii -/
theorem inscribed_sphere_radius_ratio 
  (h : SpecialHexahedron) 
  (o : RegularOctahedron) 
  (h_valid : h.faces_equilateral ∧ h.faces_congruent_to_octahedron) :
  inscribed_sphere_radius SpecialHexahedron / inscribed_sphere_radius RegularOctahedron = 2/3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_ratio_l1275_127567


namespace NUMINAMATH_CALUDE_motorcycle_toll_correct_l1275_127576

/-- Represents the weekly commute scenario for Geordie --/
structure CommuteScenario where
  workDaysPerWeek : ℕ
  carToll : ℚ
  mpg : ℚ
  commuteDistance : ℚ
  gasPrice : ℚ
  carTripsPerWeek : ℕ
  motorcycleTripsPerWeek : ℕ
  totalWeeklyCost : ℚ

/-- Calculates the motorcycle toll given a commute scenario --/
def calculateMotorcycleToll (scenario : CommuteScenario) : ℚ :=
  sorry

/-- Theorem stating that the calculated motorcycle toll is correct --/
theorem motorcycle_toll_correct (scenario : CommuteScenario) :
  scenario.workDaysPerWeek = 5 ∧
  scenario.carToll = 25/2 ∧
  scenario.mpg = 35 ∧
  scenario.commuteDistance = 14 ∧
  scenario.gasPrice = 15/4 ∧
  scenario.carTripsPerWeek = 3 ∧
  scenario.motorcycleTripsPerWeek = 2 ∧
  scenario.totalWeeklyCost = 118 →
  calculateMotorcycleToll scenario = 131/4 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_toll_correct_l1275_127576


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1275_127577

theorem diophantine_equation_solution (x y z : ℤ) :
  5 * x^3 + 11 * y^3 + 13 * z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1275_127577


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1275_127587

/-- A complex number z satisfying z ⋅ i = 3 + 4i corresponds to a point in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant (z : ℂ) : z * Complex.I = 3 + 4 * Complex.I → z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1275_127587


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l1275_127581

theorem ceiling_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l1275_127581


namespace NUMINAMATH_CALUDE_money_distribution_sum_l1275_127512

/-- Represents the share of money for each person --/
structure Share where
  amount : ℝ

/-- Represents the distribution of money among A, B, and C --/
structure Distribution where
  a : Share
  b : Share
  c : Share

/-- The conditions of the problem --/
def satisfiesConditions (d : Distribution) : Prop :=
  d.b.amount = 0.65 * d.a.amount ∧
  d.c.amount = 0.40 * d.a.amount ∧
  d.c.amount = 32

/-- The total sum of money --/
def totalSum (d : Distribution) : ℝ :=
  d.a.amount + d.b.amount + d.c.amount

/-- The theorem to prove --/
theorem money_distribution_sum :
  ∀ d : Distribution, satisfiesConditions d → totalSum d = 164 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_sum_l1275_127512


namespace NUMINAMATH_CALUDE_five_pow_minus_two_pow_div_by_three_l1275_127556

theorem five_pow_minus_two_pow_div_by_three (n : ℕ) :
  ∃ k : ℤ, 5^n - 2^n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_five_pow_minus_two_pow_div_by_three_l1275_127556


namespace NUMINAMATH_CALUDE_tom_tim_ratio_l1275_127506

/-- The typing speeds of Tim and Tom, and their relationship -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ
  total_normal : tim + tom = 15
  total_increased : tim + 1.6 * tom = 18

/-- The ratio of Tom's normal typing speed to Tim's is 1:2 -/
theorem tom_tim_ratio (s : TypingSpeed) : s.tom / s.tim = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_tim_ratio_l1275_127506


namespace NUMINAMATH_CALUDE_sin_2x_value_l1275_127500

theorem sin_2x_value (x : ℝ) (h : Real.tan (x + π/4) = 2) : Real.sin (2*x) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l1275_127500


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l1275_127532

/-- Given that M(4, 6) is the midpoint of CD and C has coordinates (10, 2),
    prove that the sum of the coordinates of point D is 8. -/
theorem sum_of_coordinates_of_D (C D M : ℝ × ℝ) : 
  C = (10, 2) →
  M = (4, 6) →
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l1275_127532


namespace NUMINAMATH_CALUDE_x_plus_y_squared_l1275_127588

theorem x_plus_y_squared (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : 
  (x + y)^2 = 135 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_squared_l1275_127588


namespace NUMINAMATH_CALUDE_alice_investment_ratio_l1275_127518

theorem alice_investment_ratio (initial_investment : ℝ) 
  (alice_final : ℝ) (bob_final : ℝ) :
  initial_investment = 2000 →
  bob_final = 6 * initial_investment →
  bob_final = alice_final + 8000 →
  alice_final / initial_investment = 2 :=
by sorry

end NUMINAMATH_CALUDE_alice_investment_ratio_l1275_127518


namespace NUMINAMATH_CALUDE_fourth_roll_prob_is_five_sixths_l1275_127534

-- Define the types of dice
inductive DieType
| Fair
| BiasedSix
| BiasedOne

-- Define the probability of rolling a six for each die type
def probSix (d : DieType) : ℚ :=
  match d with
  | DieType.Fair => 1/6
  | DieType.BiasedSix => 1/2
  | DieType.BiasedOne => 1/10

-- Define the probability of selecting each die
def probSelectDie (d : DieType) : ℚ := 1/3

-- Define the probability of rolling three sixes in a row for a given die
def probThreeSixes (d : DieType) : ℚ := (probSix d) ^ 3

-- Define the total probability of rolling three sixes
def totalProbThreeSixes : ℚ :=
  (probSelectDie DieType.Fair) * (probThreeSixes DieType.Fair) +
  (probSelectDie DieType.BiasedSix) * (probThreeSixes DieType.BiasedSix) +
  (probSelectDie DieType.BiasedOne) * (probThreeSixes DieType.BiasedOne)

-- Define the updated probability of having used each die type given three sixes were rolled
def updatedProbDie (d : DieType) : ℚ :=
  (probSelectDie d) * (probThreeSixes d) / totalProbThreeSixes

-- The main theorem
theorem fourth_roll_prob_is_five_sixths :
  (updatedProbDie DieType.Fair) * (probSix DieType.Fair) +
  (updatedProbDie DieType.BiasedSix) * (probSix DieType.BiasedSix) +
  (updatedProbDie DieType.BiasedOne) * (probSix DieType.BiasedOne) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_roll_prob_is_five_sixths_l1275_127534


namespace NUMINAMATH_CALUDE_dolphin_ratio_l1275_127599

theorem dolphin_ratio (initial_dolphins final_dolphins : ℕ) 
  (h1 : initial_dolphins = 65)
  (h2 : final_dolphins = 260) :
  (final_dolphins - initial_dolphins) / initial_dolphins = 3 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_ratio_l1275_127599


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1275_127516

def A : Set ℝ := {x | x^2 - 2*x = 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1275_127516


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l1275_127537

theorem simplify_radical_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l1275_127537


namespace NUMINAMATH_CALUDE_total_length_is_24_l1275_127533

/-- Represents a geometric figure with perpendicular adjacent sides -/
structure GeometricFigure where
  bottom : ℝ
  right : ℝ
  top_left : ℝ
  top_right : ℝ
  middle_horizontal : ℝ
  middle_vertical : ℝ
  left : ℝ

/-- Calculates the total length of visible segments in the transformed figure -/
def total_length_after_transform (fig : GeometricFigure) : ℝ :=
  fig.bottom + (fig.right - 2) + (fig.top_left - 3) + fig.left

/-- Theorem stating that the total length of segments in Figure 2 is 24 units -/
theorem total_length_is_24 (fig : GeometricFigure) 
  (h1 : fig.bottom = 5)
  (h2 : fig.right = 10)
  (h3 : fig.top_left = 4)
  (h4 : fig.top_right = 4)
  (h5 : fig.middle_horizontal = 3)
  (h6 : fig.middle_vertical = 3)
  (h7 : fig.left = 10) :
  total_length_after_transform fig = 24 := by
  sorry


end NUMINAMATH_CALUDE_total_length_is_24_l1275_127533


namespace NUMINAMATH_CALUDE_triangle_side_constraints_l1275_127505

theorem triangle_side_constraints (n : ℕ+) : 
  (2 * n + 10 < 3 * n + 5 ∧ 3 * n + 5 < n + 15) ∧
  (2 * n + 10 + (n + 15) > 3 * n + 5) ∧
  (2 * n + 10 + (3 * n + 5) > n + 15) ∧
  (n + 15 + (3 * n + 5) > 2 * n + 10) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_constraints_l1275_127505


namespace NUMINAMATH_CALUDE_race_distance_proof_l1275_127585

/-- The total distance of a race where:
  * A covers the distance in 20 seconds
  * B covers the distance in 25 seconds
  * A beats B by 14 meters
-/
def race_distance : ℝ := 56

/-- A's time to complete the race in seconds -/
def time_A : ℝ := 20

/-- B's time to complete the race in seconds -/
def time_B : ℝ := 25

/-- The distance by which A beats B in meters -/
def beat_distance : ℝ := 14

theorem race_distance_proof :
  race_distance = (time_B * beat_distance) / (time_B / time_A - 1) :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1275_127585


namespace NUMINAMATH_CALUDE_fudge_difference_l1275_127528

/-- Converts pounds to ounces -/
def pounds_to_ounces (pounds : ℚ) : ℚ := 16 * pounds

theorem fudge_difference (marina_fudge : ℚ) (lazlo_fudge : ℚ) : 
  marina_fudge = 4.5 →
  lazlo_fudge = pounds_to_ounces 4 - 6 →
  pounds_to_ounces marina_fudge - lazlo_fudge = 14 := by
  sorry

end NUMINAMATH_CALUDE_fudge_difference_l1275_127528


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1275_127535

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1275_127535


namespace NUMINAMATH_CALUDE_at_most_one_square_l1275_127536

theorem at_most_one_square (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n + 1) = (a n)^3 + 1999) :
  ∃! n : ℕ, ∃ k : ℤ, a n = k^2 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_square_l1275_127536


namespace NUMINAMATH_CALUDE_quadratic_circle_properties_l1275_127538

/-- A quadratic function that intersects both coordinate axes at three points -/
structure QuadraticFunction where
  b : ℝ
  intersects_axes : ∃ (x₁ x₂ y : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 + 2*x₁ + b = 0 ∧ 
    x₂^2 + 2*x₂ + b = 0 ∧ 
    b = y

/-- The circle passing through the three intersection points -/
def circle_equation (f : QuadraticFunction) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - (f.b + 1)*y + f.b = 0

theorem quadratic_circle_properties (f : QuadraticFunction) :
  (f.b < 1 ∧ f.b ≠ 0) ∧
  (∀ x y, circle_equation f x y ↔ x^2 + y^2 + 2*x - (f.b + 1)*y + f.b = 0) ∧
  circle_equation f (-2) 1 ∧
  circle_equation f 0 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_circle_properties_l1275_127538


namespace NUMINAMATH_CALUDE_beach_towel_loads_l1275_127545

/-- The number of laundry loads required for beach towels during a vacation. -/
def laundry_loads (num_families : ℕ) (people_per_family : ℕ) (days : ℕ) 
                  (towels_per_person_per_day : ℕ) (towels_per_load : ℕ) : ℕ :=
  (num_families * people_per_family * days * towels_per_person_per_day) / towels_per_load

theorem beach_towel_loads : 
  laundry_loads 7 6 10 2 10 = 84 := by sorry

end NUMINAMATH_CALUDE_beach_towel_loads_l1275_127545


namespace NUMINAMATH_CALUDE_movie_children_count_l1275_127553

/-- Calculates the maximum number of children that can be taken to the movies given the ticket costs and total budget. -/
def max_children (adult_ticket_cost child_ticket_cost total_budget : ℕ) : ℕ :=
  ((total_budget - adult_ticket_cost) / child_ticket_cost)

/-- Theorem stating that given the specific costs and budget, the maximum number of children is 9. -/
theorem movie_children_count :
  let adult_ticket_cost := 8
  let child_ticket_cost := 3
  let total_budget := 35
  max_children adult_ticket_cost child_ticket_cost total_budget = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_children_count_l1275_127553


namespace NUMINAMATH_CALUDE_complex_power_220_36_l1275_127544

theorem complex_power_220_36 : (Complex.exp (220 * π / 180 * I))^36 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_220_36_l1275_127544


namespace NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l1275_127583

/-- The number of triangles formed with a fixed vertex from 8 points on a circle -/
theorem triangles_with_fixed_vertex (n : ℕ) (h : n = 8) : 
  (Nat.choose (n - 1) 2) = 21 := by
  sorry

#check triangles_with_fixed_vertex

end NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l1275_127583


namespace NUMINAMATH_CALUDE_stella_annual_income_l1275_127519

/-- Calculates the annual income given monthly income and months of unpaid leave -/
def annual_income (monthly_income : ℕ) (unpaid_leave_months : ℕ) : ℕ :=
  monthly_income * (12 - unpaid_leave_months)

/-- Theorem: Given Stella's monthly income and unpaid leave, her annual income is 49190 dollars -/
theorem stella_annual_income :
  annual_income 4919 2 = 49190 := by
  sorry

end NUMINAMATH_CALUDE_stella_annual_income_l1275_127519


namespace NUMINAMATH_CALUDE_counterfeit_coin_location_l1275_127597

/-- Represents a coin that can be either genuine or counterfeit -/
inductive Coin
  | genuine
  | counterfeit

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | equal
  | notEqual

/-- Function to perform a weighing operation on two pairs of coins -/
def weighPairs (c1 c2 c3 c4 : Coin) : WeighingResult :=
  sorry

/-- Theorem stating that we can narrow down the location of the counterfeit coin -/
theorem counterfeit_coin_location
  (coins : Fin 6 → Coin)
  (h_one_counterfeit : ∃! i, coins i = Coin.counterfeit) :
  (weighPairs (coins 0) (coins 1) (coins 2) (coins 3) = WeighingResult.equal
    → (coins 4 = Coin.counterfeit ∨ coins 5 = Coin.counterfeit))
  ∧
  (weighPairs (coins 0) (coins 1) (coins 2) (coins 3) = WeighingResult.notEqual
    → (coins 0 = Coin.counterfeit ∨ coins 1 = Coin.counterfeit ∨
       coins 2 = Coin.counterfeit ∨ coins 3 = Coin.counterfeit)) :=
  sorry

end NUMINAMATH_CALUDE_counterfeit_coin_location_l1275_127597


namespace NUMINAMATH_CALUDE_problem_statement_l1275_127566

-- Define the base 10 logarithm
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the problem statement
theorem problem_statement (a b : ℝ) :
  a > 0 ∧ b > 0 ∧
  (∃ (m n : ℕ), 
    Real.sqrt (log a) = m ∧
    Real.sqrt (log b) = n ∧
    m + 2*n + m^2 + (n^2/2) = 150) ∧
  a^2 * b = 10^81 →
  a * b = 10^85 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1275_127566


namespace NUMINAMATH_CALUDE_abc_product_equals_k_absolute_value_l1275_127515

theorem abc_product_equals_k_absolute_value 
  (a b c k : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ k ≠ 0) 
  (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) : 
  |a * b * c| = |k| := by
  sorry

end NUMINAMATH_CALUDE_abc_product_equals_k_absolute_value_l1275_127515


namespace NUMINAMATH_CALUDE_smallest_solution_is_smaller_root_l1275_127558

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 + 9 * x - 92 = 0

-- Define the original equation
def original_eq (x : ℝ) : Prop := 3 * x^2 + 24 * x - 92 = x * (x + 15)

-- Theorem statement
theorem smallest_solution_is_smaller_root :
  ∃ (x : ℝ), quadratic_eq x ∧ 
  (∀ (y : ℝ), quadratic_eq y → x ≤ y) ∧
  (∀ (z : ℝ), original_eq z → x ≤ z) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_is_smaller_root_l1275_127558


namespace NUMINAMATH_CALUDE_score_79_implies_93_correct_l1275_127546

/-- Represents the grading system for a test -/
structure TestGrade where
  total_questions : ℕ
  correct_answers : ℕ
  score : ℤ

/-- Theorem stating that for a 100-question test with the given grading system,
    a score of 79 implies 93 correct answers -/
theorem score_79_implies_93_correct
  (test : TestGrade)
  (h1 : test.total_questions = 100)
  (h2 : test.score = test.correct_answers - 2 * (test.total_questions - test.correct_answers))
  (h3 : test.score = 79) :
  test.correct_answers = 93 := by
  sorry

end NUMINAMATH_CALUDE_score_79_implies_93_correct_l1275_127546


namespace NUMINAMATH_CALUDE_fourth_term_constant_implies_n_equals_5_l1275_127511

theorem fourth_term_constant_implies_n_equals_5 (n : ℕ) (x : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 3) * (-1/2)^3 * x^((n-5)/2) = k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_constant_implies_n_equals_5_l1275_127511


namespace NUMINAMATH_CALUDE_classic_rock_collections_l1275_127595

/-- The number of albums in either Andrew's or Bob's collection, but not both -/
def albums_not_shared (andrew_total : ℕ) (bob_not_andrew : ℕ) (shared : ℕ) : ℕ :=
  (andrew_total - shared) + bob_not_andrew

theorem classic_rock_collections :
  let andrew_total := 20
  let bob_not_andrew := 8
  let shared := 11
  albums_not_shared andrew_total bob_not_andrew shared = 17 := by sorry

end NUMINAMATH_CALUDE_classic_rock_collections_l1275_127595


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1275_127509

/-- The complex number z defined as 2/(1-i) - 2i^3 is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant :
  let z : ℂ := 2 / (1 - Complex.I) - 2 * Complex.I^3
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1275_127509


namespace NUMINAMATH_CALUDE_paper_number_sum_paper_number_sum_proof_l1275_127548

/-- Given n pieces of paper, each containing 3 different positive integers no greater than n,
    and any two pieces sharing exactly one common number, prove that the sum of all numbers
    written on these pieces of paper is equal to 3 * n(n+1)/2. -/
theorem paper_number_sum (n : ℕ) : ℕ :=
  let paper_count := n
  let max_number := n
  let numbers_per_paper := 3
  let shared_number_count := 1
  3 * (n * (n + 1) / 2)

-- The proof is omitted as per instructions
theorem paper_number_sum_proof (n : ℕ) :
  paper_number_sum n = 3 * (n * (n + 1) / 2) := by sorry

end NUMINAMATH_CALUDE_paper_number_sum_paper_number_sum_proof_l1275_127548


namespace NUMINAMATH_CALUDE_bob_walking_distance_l1275_127565

/-- Proves that Bob walked 16 miles when he met Yolanda given the problem conditions --/
theorem bob_walking_distance (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) 
  (head_start : ℝ) : 
  total_distance = 31 ∧ 
  yolanda_speed = 3 ∧ 
  bob_speed = 4 ∧ 
  head_start = 1 →
  ∃ t : ℝ, t > 0 ∧ yolanda_speed * (t + head_start) + bob_speed * t = total_distance ∧ 
  bob_speed * t = 16 :=
by sorry

end NUMINAMATH_CALUDE_bob_walking_distance_l1275_127565


namespace NUMINAMATH_CALUDE_parallel_lines_l1275_127568

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Two lines are coincident if and only if they have the same slope and y-intercept -/
def coincident (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

theorem parallel_lines (a : ℝ) : 
  (parallel (-a/2) (-3/(a-1)) ∧ ¬coincident (-a/2) (-1/2) (-3/(a-1)) (-1/(a-1))) → 
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_l1275_127568


namespace NUMINAMATH_CALUDE_wx_plus_yz_equals_99_l1275_127539

theorem wx_plus_yz_equals_99 
  (w x y z : ℝ) 
  (h1 : w + x + y = -2)
  (h2 : w + x + z = 4)
  (h3 : w + y + z = 19)
  (h4 : x + y + z = 12) :
  w * x + y * z = 99 := by
sorry

end NUMINAMATH_CALUDE_wx_plus_yz_equals_99_l1275_127539


namespace NUMINAMATH_CALUDE_sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one_l1275_127563

theorem sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one_l1275_127563


namespace NUMINAMATH_CALUDE_farm_size_l1275_127522

/-- Represents a farm with sunflowers and flax -/
structure Farm where
  flax : ℕ
  sunflowers : ℕ

/-- The total size of the farm in acres -/
def Farm.total_size (f : Farm) : ℕ := f.flax + f.sunflowers

/-- Theorem: Given the conditions, the farm's total size is 240 acres -/
theorem farm_size (f : Farm) 
  (h1 : f.sunflowers = f.flax + 80)  -- 80 more acres of sunflowers than flax
  (h2 : f.flax = 80)                 -- 80 acres of flax
  : f.total_size = 240 := by
  sorry

end NUMINAMATH_CALUDE_farm_size_l1275_127522


namespace NUMINAMATH_CALUDE_max_value_of_t_l1275_127530

theorem max_value_of_t (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (y / (x^2 + y^2)) ≤ 1 / Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ min x (y / (x^2 + y^2)) = 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_t_l1275_127530


namespace NUMINAMATH_CALUDE_percentage_calculation_l1275_127526

def total_population : ℕ := 40000
def part_population : ℕ := 36000

theorem percentage_calculation : 
  (part_population : ℚ) / (total_population : ℚ) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1275_127526


namespace NUMINAMATH_CALUDE_log_inequality_equivalence_l1275_127543

/-- A function that is even and monotonically increasing on [0,+∞) -/
def EvenMonoIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem log_inequality_equivalence (f : ℝ → ℝ) (h : EvenMonoIncreasing f) :
  (∀ x : ℝ, f 1 < f (Real.log x) ↔ (x > 10 ∨ 0 < x ∧ x < (1/10))) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_equivalence_l1275_127543


namespace NUMINAMATH_CALUDE_probability_red_before_green_l1275_127529

def red_chips : ℕ := 4
def green_chips : ℕ := 3
def total_chips : ℕ := red_chips + green_chips

def favorable_arrangements : ℕ := Nat.choose (total_chips - 1) green_chips
def total_arrangements : ℕ := Nat.choose total_chips green_chips

theorem probability_red_before_green :
  (favorable_arrangements : ℚ) / total_arrangements = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_before_green_l1275_127529


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l1275_127596

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The squared distance between two points in a 2D plane --/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between the intersection points of two specific circles --/
theorem intersection_distance_squared (c1 c2 : Circle) 
  (h1 : c1 = ⟨(3, -2), 5⟩) 
  (h2 : c2 = ⟨(3, 4), 3⟩) : 
  ∃ (p1 p2 : ℝ × ℝ), 
    squaredDistance p1 c1.center = c1.radius^2 ∧ 
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧ 
    squaredDistance p2 c2.center = c2.radius^2 ∧
    squaredDistance p1 p2 = 224/9 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l1275_127596


namespace NUMINAMATH_CALUDE_toilet_paper_supply_duration_l1275_127504

/-- Calculates the number of days a toilet paper supply will last for a family -/
def toilet_paper_duration (bill_usage : ℕ) (wife_usage : ℕ) (kid_usage : ℕ) (num_kids : ℕ) (num_rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  let total_squares := num_rolls * squares_per_roll
  let daily_usage := bill_usage + wife_usage + kid_usage * num_kids
  total_squares / daily_usage

theorem toilet_paper_supply_duration :
  toilet_paper_duration 15 32 30 2 1000 300 = 2803 := by
  sorry

end NUMINAMATH_CALUDE_toilet_paper_supply_duration_l1275_127504


namespace NUMINAMATH_CALUDE_remainder_theorem_l1275_127573

theorem remainder_theorem (n : ℤ) (h : ∃ (a : ℤ), n = 100 * a - 1) : 
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1275_127573


namespace NUMINAMATH_CALUDE_base_number_proof_l1275_127578

theorem base_number_proof (base : ℝ) : base ^ 7 = 3 ^ 14 → base = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l1275_127578


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1275_127552

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1275_127552


namespace NUMINAMATH_CALUDE_number_of_pupils_theorem_l1275_127502

/-- The number of pupils sent up for examination -/
def N : ℕ := 28

/-- The average marks of all pupils -/
def overall_average : ℚ := 39

/-- The average marks if 7 specific pupils were not sent up -/
def new_average : ℚ := 45

/-- The marks of the 7 specific pupils -/
def specific_pupils_marks : List ℕ := [25, 12, 15, 19, 31, 18, 27]

/-- The sum of marks of the 7 specific pupils -/
def sum_specific_marks : ℕ := specific_pupils_marks.sum

theorem number_of_pupils_theorem :
  (N * overall_average - sum_specific_marks) / (N - 7) = new_average :=
sorry

end NUMINAMATH_CALUDE_number_of_pupils_theorem_l1275_127502


namespace NUMINAMATH_CALUDE_power_function_through_point_and_value_l1275_127574

/-- A power function that passes through the point (2,8) -/
def f : ℝ → ℝ := fun x ↦ x^3

theorem power_function_through_point_and_value :
  f 2 = 8 ∧ f 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_and_value_l1275_127574


namespace NUMINAMATH_CALUDE_cara_seating_arrangement_l1275_127584

theorem cara_seating_arrangement (n : ℕ) (k : ℕ) : n = 7 ∧ k = 2 → Nat.choose n k = 21 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangement_l1275_127584


namespace NUMINAMATH_CALUDE_f_min_value_g_leq_f_range_l1275_127514

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (m : ℝ), m = 3 ∧ ∀ x, f x ≥ m := by sorry

-- Theorem for the range of a where g(a) ≤ f(x) for all x
theorem g_leq_f_range : ∀ a : ℝ, (∀ x : ℝ, g a ≤ f x) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end NUMINAMATH_CALUDE_f_min_value_g_leq_f_range_l1275_127514


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1275_127525

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 4 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2/5 * x ∨ y = -2/5 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(2/5)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1275_127525


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1275_127551

theorem longest_side_of_triangle (y : ℚ) : 
  6 + (y + 3) + (3 * y - 2) = 40 →
  max 6 (max (y + 3) (3 * y - 2)) = 91 / 4 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1275_127551


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1275_127507

def units_digit (n : ℕ) : ℕ := n % 10

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem quadratic_function_theorem (a b c : ℤ) (p : ℕ) :
  is_positive_even p →
  10 ≤ p →
  p ≤ 50 →
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  (a * p^2 + b * p + c : ℤ) = 0 →
  (a * p^4 + b * p^2 + c : ℤ) = (a * p^6 + b * p^3 + c : ℤ) →
  units_digit (p + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1275_127507


namespace NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l1275_127554

theorem temperature_difference_product (P : ℝ) : 
  (∃ (A B : ℝ), A = B + P ∧ 
   ∃ (A_t B_t : ℝ), A_t = B + P - 8 ∧ B_t = B + 2 ∧ 
   |A_t - B_t| = 4) →
  (P = 12 ∨ P = 4) :=
by sorry

theorem product_of_possible_P_values : 
  (∀ P : ℝ, (∃ (A B : ℝ), A = B + P ∧ 
   ∃ (A_t B_t : ℝ), A_t = B + P - 8 ∧ B_t = B + 2 ∧ 
   |A_t - B_t| = 4) →
  (P = 12 ∨ P = 4)) →
  12 * 4 = 48 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l1275_127554


namespace NUMINAMATH_CALUDE_worker_assessment_correct_l1275_127582

/-- Worker's skill assessment model -/
structure WorkerAssessment where
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- Probability of ending assessment with 10 products -/
def prob_end_10 (w : WorkerAssessment) : ℝ :=
  w.p^9 * (10 - 9 * w.p)

/-- Expected value of total products produced and debugged -/
def expected_total (w : WorkerAssessment) : ℝ :=
  20 - 10 * w.p - 10 * w.p^9 + 10 * w.p^10

/-- Main theorem: Correctness of worker assessment model -/
theorem worker_assessment_correct (w : WorkerAssessment) :
  (prob_end_10 w = w.p^9 * (10 - 9 * w.p)) ∧
  (expected_total w = 20 - 10 * w.p - 10 * w.p^9 + 10 * w.p^10) := by
  sorry

end NUMINAMATH_CALUDE_worker_assessment_correct_l1275_127582


namespace NUMINAMATH_CALUDE_certain_number_l1275_127560

theorem certain_number : ∃ x : ℕ, x - 2 - 2 = 5 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_l1275_127560
