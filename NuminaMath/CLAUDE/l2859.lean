import Mathlib

namespace genevieve_code_lines_l2859_285952

/-- Represents the number of lines of code per debugging session -/
def lines_per_debug : ℕ := 100

/-- Represents the number of errors found per debugging session -/
def errors_per_debug : ℕ := 3

/-- Represents the total number of errors fixed so far -/
def total_errors_fixed : ℕ := 129

/-- Calculates the number of lines of code written based on the given conditions -/
def lines_of_code : ℕ := (total_errors_fixed / errors_per_debug) * lines_per_debug

/-- Theorem stating that the number of lines of code written is 4300 -/
theorem genevieve_code_lines : lines_of_code = 4300 := by
  sorry

end genevieve_code_lines_l2859_285952


namespace laptop_price_l2859_285964

theorem laptop_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_payment = 200 → 
  upfront_percentage * total_price = upfront_payment → 
  total_price = 1000 :=
by sorry

end laptop_price_l2859_285964


namespace constant_d_value_l2859_285913

theorem constant_d_value (x y d : ℝ) 
  (h1 : x / (2 * y) = d / 2)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 25) :
  d = 3 := by
sorry

end constant_d_value_l2859_285913


namespace units_digit_factorial_50_l2859_285930

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem units_digit_factorial_50 :
  ∃ k : ℕ, factorial 50 = 10 * k :=
sorry

end units_digit_factorial_50_l2859_285930


namespace fatimas_number_probability_l2859_285928

def first_three_options : List Nat := [296, 299, 295]
def last_five_digits : List Nat := [0, 1, 6, 7, 8]

def total_possibilities : Nat :=
  (first_three_options.length) * (Nat.factorial last_five_digits.length)

theorem fatimas_number_probability :
  (1 : ℚ) / total_possibilities = (1 : ℚ) / 360 := by
  sorry

end fatimas_number_probability_l2859_285928


namespace right_triangle_cosine_l2859_285906

theorem right_triangle_cosine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 7) (h3 : c = 25) :
  b / c = 24 / 25 := by
  sorry

end right_triangle_cosine_l2859_285906


namespace football_team_members_l2859_285926

theorem football_team_members :
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 5 = 1 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  n = 251 := by
sorry

end football_team_members_l2859_285926


namespace draw_with_even_ball_l2859_285940

/-- The number of balls in the bin -/
def total_balls : ℕ := 15

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 4

/-- The number of odd-numbered balls -/
def odd_balls : ℕ := 8

/-- Calculate the number of ways to draw n balls from m balls in order -/
def ways_to_draw (m n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (m - i)) 1

/-- The main theorem: number of ways to draw 4 balls with at least one even-numbered ball -/
theorem draw_with_even_ball :
  ways_to_draw total_balls drawn_balls - ways_to_draw odd_balls drawn_balls = 31080 := by
  sorry

end draw_with_even_ball_l2859_285940


namespace wedding_decoration_cost_l2859_285957

/-- The cost of decorations for a wedding reception --/
def decorationCost (numTables : ℕ) (tableclothCost : ℕ) (placeSettingsPerTable : ℕ) 
  (placeSettingCost : ℕ) (rosesPerCenterpiece : ℕ) (roseCost : ℕ) 
  (liliesPerCenterpiece : ℕ) (lilyCost : ℕ) : ℕ :=
  numTables * tableclothCost + 
  numTables * placeSettingsPerTable * placeSettingCost +
  numTables * rosesPerCenterpiece * roseCost +
  numTables * liliesPerCenterpiece * lilyCost

/-- The total cost of decorations for Nathan's wedding reception is $3500 --/
theorem wedding_decoration_cost : 
  decorationCost 20 25 4 10 10 5 15 4 = 3500 := by
  sorry

end wedding_decoration_cost_l2859_285957


namespace rhombus_diagonal_length_l2859_285973

theorem rhombus_diagonal_length (area : ℝ) (ratio : ℚ) (shorter_diagonal : ℝ) : 
  area = 144 →
  ratio = 4/3 →
  shorter_diagonal = 6 * Real.sqrt 6 →
  area = (1/2) * shorter_diagonal * (ratio * shorter_diagonal) :=
by sorry

end rhombus_diagonal_length_l2859_285973


namespace even_sum_of_even_l2859_285988

theorem even_sum_of_even (a b : ℤ) : Even a ∧ Even b → Even (a + b) := by sorry

end even_sum_of_even_l2859_285988


namespace sum_of_15th_set_l2859_285997

/-- The first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- The last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end sum_of_15th_set_l2859_285997


namespace expected_girls_left_of_boys_l2859_285934

theorem expected_girls_left_of_boys (num_boys num_girls : ℕ) 
  (h1 : num_boys = 10) (h2 : num_girls = 7) :
  let total := num_boys + num_girls
  let expected_value := (num_girls : ℚ) / (total + 1 : ℚ)
  expected_value = 7 / 11 := by
  sorry

end expected_girls_left_of_boys_l2859_285934


namespace quadratic_one_root_l2859_285925

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 - 6*m*x + 2*m = 0) → m = 2/9 := by
  sorry

end quadratic_one_root_l2859_285925


namespace existence_of_solution_specific_solution_valid_l2859_285992

theorem existence_of_solution :
  ∃ (n m : ℝ), n ≠ 0 ∧ m ≠ 0 ∧ (n * 5^n)^n = m * 5^9 :=
by sorry

theorem specific_solution_valid :
  let n : ℝ := 3
  let m : ℝ := 27
  (n * 5^n)^n = m * 5^9 :=
by sorry

end existence_of_solution_specific_solution_valid_l2859_285992


namespace prize_location_questions_l2859_285963

/-- Represents the three doors in the game show. -/
inductive Door
| left
| center
| right

/-- Represents the host's response to a question. -/
inductive Response
| yes
| no

/-- The maximum number of lies the host can tell. -/
def max_lies : ℕ := 10

/-- The function that determines the minimum number of questions needed to locate the prize. -/
def min_questions_to_locate_prize (doors : List Door) (max_lies : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 32 questions are needed to locate the prize with certainty. -/
theorem prize_location_questions (doors : List Door) (h1 : doors.length = 3) :
  min_questions_to_locate_prize doors max_lies = 32 :=
sorry

end prize_location_questions_l2859_285963


namespace trigonometric_expression_l2859_285978

theorem trigonometric_expression (α : Real) (m : Real) 
  (h : Real.tan (5 * Real.pi + α) = m) : 
  (Real.sin (α - 3 * Real.pi) + Real.cos (-α)) / (Real.sin α - Real.cos (Real.pi + α)) = (m + 1) / (m - 1) := by
  sorry

end trigonometric_expression_l2859_285978


namespace completing_square_quadratic_l2859_285968

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by
  sorry

end completing_square_quadratic_l2859_285968


namespace five_numbers_product_1000_l2859_285902

theorem five_numbers_product_1000 :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                     c ≠ d ∧ c ≠ e ∧
                     d ≠ e ∧
                     a * b * c * d * e = 1000 :=
by sorry

end five_numbers_product_1000_l2859_285902


namespace gcd_problem_l2859_285949

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 3883) :
  Int.gcd (4 * b^2 + 35 * b + 56) (3 * b + 8) = 8 := by
  sorry

end gcd_problem_l2859_285949


namespace optimal_configuration_prevents_loosening_l2859_285939

/-- A rectangular prism box with trapezoid end faces on a cart -/
structure BoxOnCart where
  d : ℝ  -- Distance between parallel sides of trapezoid
  k : ℝ  -- Width of cart
  b : ℝ  -- Height of trapezoid at one end
  c : ℝ  -- Height of trapezoid at other end
  h_k_gt_d : k > d
  h_b_gt_c : b > c

/-- The optimal configuration of the box on the cart -/
def optimal_configuration (box : BoxOnCart) : Prop :=
  let DC₁ := box.c / (box.b - box.c) * (box.k - box.d)
  let AB₁ := box.b / (box.b + box.c) * (box.k - box.d)
  DC₁ > 0 ∧ AB₁ > 0 ∧ box.k ≤ 2 * box.b * box.d / (box.b - box.c)

/-- The theorem stating the optimal configuration prevents the rope from loosening -/
theorem optimal_configuration_prevents_loosening (box : BoxOnCart) :
  optimal_configuration box →
  ∃ (DC₁ AB₁ : ℝ),
    DC₁ = box.c / (box.b - box.c) * (box.k - box.d) ∧
    AB₁ = box.b / (box.b + box.c) * (box.k - box.d) ∧
    DC₁ > 0 ∧ AB₁ > 0 ∧
    box.k ≤ 2 * box.b * box.d / (box.b - box.c) :=
by sorry

end optimal_configuration_prevents_loosening_l2859_285939


namespace mikes_salary_increase_l2859_285989

theorem mikes_salary_increase (freds_salary_then : ℝ) (mikes_salary_now : ℝ) :
  freds_salary_then = 1000 →
  mikes_salary_now = 15400 →
  let mikes_salary_then := 10 * freds_salary_then
  (mikes_salary_now - mikes_salary_then) / mikes_salary_then * 100 = 54 := by
  sorry

end mikes_salary_increase_l2859_285989


namespace hyperbola_theorem_l2859_285969

def hyperbola (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

def asymptote1 (x y : ℝ) : Prop := y = 3 * x + 6
def asymptote2 (x y : ℝ) : Prop := y = -3 * x + 2

theorem hyperbola_theorem (a b h k : ℝ) :
  a > 0 → b > 0 →
  (∀ x y, asymptote1 x y ∨ asymptote2 x y → hyperbola a b h k x y) →
  hyperbola a b h k 1 5 →
  a + h = 6 * Real.sqrt 2 - 2/3 := by
  sorry

end hyperbola_theorem_l2859_285969


namespace log_sum_equality_l2859_285984

theorem log_sum_equality : 2 * Real.log 25 / Real.log 5 + 3 * Real.log 64 / Real.log 2 = 22 := by
  sorry

end log_sum_equality_l2859_285984


namespace solve_equations_and_sum_l2859_285945

/-- Given two equations involving x and y, prove the values of x, y, and their sum. -/
theorem solve_equations_and_sum :
  ∀ (x y : ℝ),
  (0.65 * x = 0.20 * 552.50) →
  (0.35 * y = 0.30 * 867.30) →
  (x = 170) ∧ (y = 743.40) ∧ (x + y = 913.40) := by
  sorry

end solve_equations_and_sum_l2859_285945


namespace bonus_calculation_l2859_285909

/-- Represents the wages of a worker for three months -/
structure Wages where
  october : ℝ
  november : ℝ
  december : ℝ

/-- Calculates the bonus based on the given wages -/
def calculate_bonus (w : Wages) : ℝ :=
  0.2 * (w.october + w.november + w.december)

theorem bonus_calculation (w : Wages) 
  (h1 : w.october / w.november = 3/2 / (4/3))
  (h2 : w.november / w.december = 2 / (8/3))
  (h3 : w.december = w.october + 450) :
  calculate_bonus w = 1494 := by
  sorry

#eval calculate_bonus { october := 2430, november := 2160, december := 2880 }

end bonus_calculation_l2859_285909


namespace average_of_multiples_of_seven_l2859_285985

def is_between (a b x : ℝ) : Prop := a < x ∧ x < b

def divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

theorem average_of_multiples_of_seven (numbers : List ℕ) : 
  (∀ n ∈ numbers, is_between 6 36 n ∧ divisible_by n 7) →
  numbers.length > 0 →
  (numbers.sum / numbers.length : ℝ) = 24.5 := by
  sorry

end average_of_multiples_of_seven_l2859_285985


namespace circus_dogs_count_l2859_285955

theorem circus_dogs_count :
  ∀ (total_dogs : ℕ) (paws_on_ground : ℕ),
    paws_on_ground = 36 →
    (total_dogs / 2 : ℕ) * 2 + (total_dogs / 2 : ℕ) * 4 = paws_on_ground →
    total_dogs = 12 :=
by sorry

end circus_dogs_count_l2859_285955


namespace crop_planting_arrangement_l2859_285916

theorem crop_planting_arrangement (n : ℕ) (h : n = 10) : 
  (Finset.sum (Finset.range (n - 6)) (λ i => n - i - 6)) = 12 := by
  sorry

end crop_planting_arrangement_l2859_285916


namespace gas_cost_per_gallon_l2859_285987

/-- Proves that the cost of gas per gallon is $4 given the specified conditions --/
theorem gas_cost_per_gallon 
  (pay_rate : ℝ) 
  (truck_efficiency : ℝ) 
  (profit : ℝ) 
  (trip_distance : ℝ) 
  (h1 : pay_rate = 0.50)
  (h2 : truck_efficiency = 20)
  (h3 : profit = 180)
  (h4 : trip_distance = 600) :
  (trip_distance * pay_rate - profit) / (trip_distance / truck_efficiency) = 4 := by
  sorry

end gas_cost_per_gallon_l2859_285987


namespace concentric_circles_ratio_l2859_285937

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (k : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = Real.sqrt 5 / 5 := by
sorry

end concentric_circles_ratio_l2859_285937


namespace min_sum_of_product_1716_l2859_285994

theorem min_sum_of_product_1716 (a b c : ℕ+) (h : a * b * c = 1716) :
  ∃ (x y z : ℕ+), x * y * z = 1716 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 31 := by
  sorry

end min_sum_of_product_1716_l2859_285994


namespace third_term_is_18_l2859_285953

def arithmetic_geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

theorem third_term_is_18 (a₁ q : ℝ) (h₁ : a₁ = 2) (h₂ : q = 3) :
  arithmetic_geometric_sequence a₁ q 3 = 18 := by
  sorry

end third_term_is_18_l2859_285953


namespace function_characterization_l2859_285996

theorem function_characterization (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 
    a * f (x / y) + a * f (x / z) - f x * f ((y + z) / 2) ≥ a^2) :
  ∀ (x : ℝ), x ≠ 0 → f x = a := by
sorry

end function_characterization_l2859_285996


namespace eighth_grade_students_l2859_285900

theorem eighth_grade_students (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 68 → 
  girls = 28 → 
  boys < 2 * girls → 
  boys = total - girls → 
  2 * girls - boys = 16 := by
sorry

end eighth_grade_students_l2859_285900


namespace intersection_ordinate_l2859_285929

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 1)^2 - 3

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point of the parabola and the y-axis -/
def intersection_point (x y : ℝ) : Prop := parabola x y ∧ y_axis x

theorem intersection_ordinate :
  ∃ x y : ℝ, intersection_point x y ∧ y = -5 := by sorry

end intersection_ordinate_l2859_285929


namespace grains_in_gray_parts_grains_in_gray_parts_specific_l2859_285933

/-- Given two circles with the same number of grains in their white parts,
    and their respective total grains, calculate the sum of grains in both gray parts. -/
theorem grains_in_gray_parts
  (white_grains : ℕ)
  (total_grains_1 total_grains_2 : ℕ)
  (h1 : total_grains_1 ≥ white_grains)
  (h2 : total_grains_2 ≥ white_grains) :
  (total_grains_1 - white_grains) + (total_grains_2 - white_grains) = 61 :=
by
  sorry

/-- Specific instance of the theorem with given values -/
theorem grains_in_gray_parts_specific :
  (87 - 68) + (110 - 68) = 61 :=
by
  sorry

end grains_in_gray_parts_grains_in_gray_parts_specific_l2859_285933


namespace cone_volume_l2859_285920

theorem cone_volume (s : Real) (c : Real) (h : s = 8) (k : c = 6 * Real.pi) :
  let r := c / (2 * Real.pi)
  let height := Real.sqrt (s^2 - r^2)
  (1/3) * Real.pi * r^2 * height = 3 * Real.sqrt 55 * Real.pi := by
sorry

end cone_volume_l2859_285920


namespace bus_stop_problem_l2859_285980

theorem bus_stop_problem (girls boys : ℕ) : 
  (girls - 15 = 5 * (boys - 45)) →
  (boys = 2 * (girls - 15)) →
  (girls = 40 ∧ boys = 50) :=
by sorry

end bus_stop_problem_l2859_285980


namespace apple_distribution_l2859_285943

theorem apple_distribution (jim jerry : ℕ) (h1 : jim = 20) (h2 : jerry = 40) : 
  ∃ jane : ℕ, 
    (2 * jim = (jim + jerry + jane) / 3) ∧ 
    (jane = 30) := by
sorry

end apple_distribution_l2859_285943


namespace range_of_a_l2859_285967

open Real

theorem range_of_a (m : ℝ) (hm : m > 0) :
  (∃ x : ℝ, x + a * (2*x + 2*m - 4*Real.exp 1*x) * (log (x + m) - log x) = 0) →
  a ∈ Set.Iic 0 ∪ Set.Ici (1 / (2 * Real.exp 1)) := by
  sorry

end range_of_a_l2859_285967


namespace common_difference_is_two_l2859_285919

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_two
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 10)
  (h_fourth : a 4 = 7) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end common_difference_is_two_l2859_285919


namespace train_passing_time_l2859_285918

/-- Proves that a train of given length and speed takes a specific time to pass a stationary object. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 180 →
  train_speed_kmh = 36 →
  passing_time = 18 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

end train_passing_time_l2859_285918


namespace coffee_pod_box_cost_l2859_285908

/-- Calculates the cost of a box of coffee pods given vacation details --/
theorem coffee_pod_box_cost
  (vacation_days : ℕ)
  (daily_pods : ℕ)
  (pods_per_box : ℕ)
  (total_spending : ℚ)
  (h1 : vacation_days = 40)
  (h2 : daily_pods = 3)
  (h3 : pods_per_box = 30)
  (h4 : total_spending = 32)
  : (total_spending / (vacation_days * daily_pods / pods_per_box : ℚ)) = 8 := by
  sorry

end coffee_pod_box_cost_l2859_285908


namespace problem_statement_l2859_285915

theorem problem_statement (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) :
  b < a := by
  sorry

end problem_statement_l2859_285915


namespace polynomial_simplification_l2859_285904

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 + 4*x + 6)*(x-2) - (x-2)*(x^2 + 5*x - 72) + (2*x - 7)*(x-2)*(x+4) = 
  4*x^3 - 8*x^2 + 50*x - 100 := by
sorry

end polynomial_simplification_l2859_285904


namespace pen_cost_l2859_285974

theorem pen_cost (x y : ℕ) (h1 : 5 * x + 4 * y = 345) (h2 : 3 * x + 6 * y = 285) : x = 52 := by
  sorry

end pen_cost_l2859_285974


namespace line_slope_intercept_sum_l2859_285938

/-- Given a line with slope -8 passing through (4, -3), prove that m + b = 21 in y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -8 → 
  -3 = m * 4 + b → 
  m + b = 21 := by
sorry

end line_slope_intercept_sum_l2859_285938


namespace cos_negative_135_degrees_l2859_285923

theorem cos_negative_135_degrees : Real.cos ((-135 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_negative_135_degrees_l2859_285923


namespace logarithm_expression_equals_zero_l2859_285917

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_zero :
  (lg 2)^2 + (lg 2) * (lg 5) + lg 5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  -- Assume the given condition
  have h : lg 2 + lg 5 = 1 := by sorry
  
  -- The proof would go here
  sorry

end logarithm_expression_equals_zero_l2859_285917


namespace triangle_perimeter_l2859_285947

/-- The perimeter of a triangle with vertices A(1,2), B(1,5), and C(4,5) on a Cartesian coordinate plane is 6 + 3√2. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 5)
  let C : ℝ × ℝ := (4, 5)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := distance A B + distance B C + distance C A
  perimeter = 6 + 3 * Real.sqrt 2 := by sorry

end triangle_perimeter_l2859_285947


namespace complement_M_N_l2859_285911

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem complement_M_N : M \ N = {-1, 0, 3} := by sorry

end complement_M_N_l2859_285911


namespace unique_solution_l2859_285998

theorem unique_solution (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a*b - 9) : a = 3 ∧ b = 3 := by
  sorry

end unique_solution_l2859_285998


namespace division_problem_l2859_285931

theorem division_problem : (70 / 4 + 90 / 4) / 4 = 10 := by
  sorry

end division_problem_l2859_285931


namespace inverse_composition_l2859_285927

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition
variable (h : ∀ x, f⁻¹ (g x) = 7 * x - 4)

-- State the theorem
theorem inverse_composition :
  g⁻¹ (f (-9)) = -5/7 := by
  sorry

end inverse_composition_l2859_285927


namespace sequence_limit_l2859_285942

def x (n : ℕ) : ℚ := (2 * n - 1) / (3 * n + 5)

theorem sequence_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - 2/3| < ε := by
  sorry

end sequence_limit_l2859_285942


namespace min_reciprocal_sum_l2859_285999

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 9/2 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1/x + 1/y + 1/z = 9/2 := by
  sorry

end min_reciprocal_sum_l2859_285999


namespace rowing_speed_l2859_285976

/-- The speed of a man rowing in still water, given his downstream speed and current speed. -/
theorem rowing_speed (downstream_speed current_speed : ℝ) 
  (h_downstream : downstream_speed = 18)
  (h_current : current_speed = 3) :
  downstream_speed - current_speed = 15 := by
  sorry

#check rowing_speed

end rowing_speed_l2859_285976


namespace tan_three_expression_zero_l2859_285905

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
  sorry

end tan_three_expression_zero_l2859_285905


namespace direct_proportion_is_straight_line_direct_proportion_passes_through_origin_l2859_285959

/-- A direct proportion function -/
def direct_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ k * x

/-- The graph of a function -/
def graph (f : ℝ → ℝ) : Set (ℝ × ℝ) := {p | p.2 = f p.1}

theorem direct_proportion_is_straight_line (k : ℝ) :
  ∃ (a b : ℝ), ∀ x y, (x, y) ∈ graph (direct_proportion k) ↔ a * x + b * y = 0 :=
sorry

theorem direct_proportion_passes_through_origin (k : ℝ) :
  (0, 0) ∈ graph (direct_proportion k) :=
sorry

end direct_proportion_is_straight_line_direct_proportion_passes_through_origin_l2859_285959


namespace package_size_l2859_285946

/-- The number of candies Shirley ate -/
def candies_eaten : ℕ := 10

/-- The number of candies Shirley has left -/
def candies_left : ℕ := 2

/-- The number of candies in one package -/
def candies_in_package : ℕ := candies_eaten + candies_left

theorem package_size : candies_in_package = 12 := by
  sorry

end package_size_l2859_285946


namespace arc_length_specific_sector_l2859_285956

/-- Arc length formula for a sector -/
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

/-- Theorem: The length of an arc in a sector with radius 2 and central angle π/3 is 2π/3 -/
theorem arc_length_specific_sector :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  arc_length r θ = 2 * π / 3 := by
  sorry

end arc_length_specific_sector_l2859_285956


namespace rotate180_unique_l2859_285971

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a rigid transformation (isometry) in 2D space -/
def RigidTransformation := Point2D → Point2D

/-- Clockwise rotation by 180° about the origin -/
def rotate180 : RigidTransformation :=
  fun p => Point2D.mk (-p.x) (-p.y)

/-- The given points -/
def C : Point2D := Point2D.mk 3 (-2)
def D : Point2D := Point2D.mk 4 (-5)
def C' : Point2D := Point2D.mk (-3) 2
def D' : Point2D := Point2D.mk (-4) 5

/-- Statement: rotate180 is the unique isometry that maps C to C' and D to D' -/
theorem rotate180_unique : 
  (rotate180 C = C') ∧ 
  (rotate180 D = D') ∧ 
  (∀ (f : RigidTransformation), (f C = C' ∧ f D = D') → f = rotate180) :=
sorry

end rotate180_unique_l2859_285971


namespace sum_of_specific_S_l2859_285995

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

theorem sum_of_specific_S : S 15 + S 28 + S 39 = 14 := by
  sorry

end sum_of_specific_S_l2859_285995


namespace chess_pawns_remaining_l2859_285961

theorem chess_pawns_remaining (initial_pawns : ℕ) 
  (sophia_lost : ℕ) (chloe_lost : ℕ) : 
  initial_pawns = 8 → 
  sophia_lost = 5 → 
  chloe_lost = 1 → 
  (initial_pawns - sophia_lost) + (initial_pawns - chloe_lost) = 10 := by
  sorry

end chess_pawns_remaining_l2859_285961


namespace smallest_x_is_correct_l2859_285935

/-- The smallest positive integer x such that 1980x is a perfect fourth power -/
def smallest_x : ℕ := 6006250

/-- Predicate to check if a number is a perfect fourth power -/
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^4

theorem smallest_x_is_correct :
  (∀ y : ℕ, y < smallest_x → ¬ is_fourth_power (1980 * y)) ∧
  is_fourth_power (1980 * smallest_x) :=
sorry

end smallest_x_is_correct_l2859_285935


namespace alcohol_concentration_in_second_vessel_l2859_285924

/-- 
Given two vessels with different capacities and alcohol concentrations, 
prove that when mixed and diluted to a certain concentration, 
the alcohol percentage in the second vessel can be determined.
-/
theorem alcohol_concentration_in_second_vessel 
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_capacity : ℝ)
  (final_mixture_capacity : ℝ)
  (final_mixture_percentage : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 30)
  (h3 : vessel2_capacity = 6)
  (h4 : total_capacity = vessel1_capacity + vessel2_capacity)
  (h5 : final_mixture_capacity = 10)
  (h6 : final_mixture_percentage = 30) :
  ∃ vessel2_alcohol_percentage : ℝ, 
    vessel2_alcohol_percentage = 30 ∧
    vessel1_capacity * (vessel1_alcohol_percentage / 100) + 
    vessel2_capacity * (vessel2_alcohol_percentage / 100) = 
    total_capacity * (final_mixture_percentage / 100) :=
by sorry

end alcohol_concentration_in_second_vessel_l2859_285924


namespace tangent_slope_circle_l2859_285962

/-- The slope of the line tangent to a circle at the point (8, 3) is -1, 
    given that the center of the circle is at (1, -4). -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (1, -4) → point = (8, 3) → 
  (point.1 - center.1) * (point.2 - center.2) = -1 := by
sorry

end tangent_slope_circle_l2859_285962


namespace excellent_chinese_or_math_excellent_all_subjects_l2859_285912

def excellent_chinese : Finset ℕ := sorry
def excellent_math : Finset ℕ := sorry
def excellent_english : Finset ℕ := sorry

axiom total_excellent : (excellent_chinese ∪ excellent_math ∪ excellent_english).card = 18
axiom chinese_count : excellent_chinese.card = 9
axiom math_count : excellent_math.card = 11
axiom english_count : excellent_english.card = 8
axiom chinese_math_count : (excellent_chinese ∩ excellent_math).card = 5
axiom math_english_count : (excellent_math ∩ excellent_english).card = 3
axiom chinese_english_count : (excellent_chinese ∩ excellent_english).card = 4

theorem excellent_chinese_or_math : 
  (excellent_chinese ∪ excellent_math).card = 15 := by sorry

theorem excellent_all_subjects : 
  (excellent_chinese ∩ excellent_math ∩ excellent_english).card = 2 := by sorry

end excellent_chinese_or_math_excellent_all_subjects_l2859_285912


namespace line_intercepts_l2859_285993

/-- Given a line with equation x/4 - y/3 = 1, prove that its x-intercept is 4 and y-intercept is -3 -/
theorem line_intercepts (x y : ℝ) :
  x/4 - y/3 = 1 → (x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = -3) := by
  sorry

end line_intercepts_l2859_285993


namespace extreme_point_range_l2859_285922

theorem extreme_point_range (m : ℝ) : 
  (∃! x₀ : ℝ, x₀ > 0 ∧ 1/2 ≤ x₀ ∧ x₀ ≤ 3 ∧
    (∀ x : ℝ, x > 0 → (x₀ + 1/x₀ + m = 0 ∧
      ∀ y : ℝ, y > 0 → y ≠ x₀ → y + 1/y + m ≠ 0))) →
  -10/3 ≤ m ∧ m < -5/2 :=
sorry

end extreme_point_range_l2859_285922


namespace solve_for_y_l2859_285975

theorem solve_for_y (x y : ℝ) 
  (h1 : x = 151) 
  (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3423000) : 
  y = 3423000 / 3375001 := by
sorry

end solve_for_y_l2859_285975


namespace sqrt_fifth_root_of_five_sixth_power_l2859_285960

theorem sqrt_fifth_root_of_five_sixth_power :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 5 ^ (15 / 2) := by sorry

end sqrt_fifth_root_of_five_sixth_power_l2859_285960


namespace fibonacci_polynomial_property_l2859_285990

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_polynomial_property (n : ℕ) (P : ℕ → ℕ) :
  (∀ k ∈ Finset.range (n + 1), P (k + n + 2) = fibonacci (k + n + 2)) →
  P (2 * n + 3) = fibonacci (2 * n + 3) - 1 := by
  sorry

end fibonacci_polynomial_property_l2859_285990


namespace dave_tickets_l2859_285951

def tickets_problem (initial_tickets spent_tickets later_tickets : ℕ) : Prop :=
  initial_tickets - spent_tickets + later_tickets = 16

theorem dave_tickets : tickets_problem 11 5 10 := by
  sorry

end dave_tickets_l2859_285951


namespace same_type_square_roots_l2859_285948

theorem same_type_square_roots :
  ∃ (k₁ k₂ : ℝ) (x : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ Real.sqrt 12 = k₁ * x ∧ Real.sqrt (1/3) = k₂ * x :=
by sorry

end same_type_square_roots_l2859_285948


namespace bull_count_l2859_285986

theorem bull_count (total_cattle : ℕ) (cow_ratio bull_ratio : ℕ) 
  (h1 : total_cattle = 555)
  (h2 : cow_ratio = 10)
  (h3 : bull_ratio = 27) : 
  (bull_ratio : ℚ) / (cow_ratio + bull_ratio : ℚ) * total_cattle = 405 :=
by sorry

end bull_count_l2859_285986


namespace initial_egg_count_l2859_285950

theorem initial_egg_count (total : ℕ) (taken : ℕ) (left : ℕ) : 
  taken = 5 → left = 42 → total = taken + left → total = 47 := by
  sorry

end initial_egg_count_l2859_285950


namespace train_passengers_proof_l2859_285932

/-- Calculates the number of passengers on each return trip given the total number of round trips, 
    passengers on each one-way trip, and total passengers transported. -/
def return_trip_passengers (round_trips : ℕ) (one_way_passengers : ℕ) (total_passengers : ℕ) : ℕ :=
  (total_passengers - round_trips * one_way_passengers) / round_trips

/-- Proves that given the specified conditions, the number of passengers on each return trip is 60. -/
theorem train_passengers_proof :
  let round_trips : ℕ := 4
  let one_way_passengers : ℕ := 100
  let total_passengers : ℕ := 640
  return_trip_passengers round_trips one_way_passengers total_passengers = 60 := by
  sorry

#eval return_trip_passengers 4 100 640

end train_passengers_proof_l2859_285932


namespace squares_four_greater_than_prime_l2859_285954

theorem squares_four_greater_than_prime :
  ∃! n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n^2 = p + 4 :=
sorry

end squares_four_greater_than_prime_l2859_285954


namespace part1_part2_l2859_285941

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - 1

-- Part 1
theorem part1 (m : ℝ) : 
  (∀ x, f m x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 := by sorry

-- Part 2
theorem part2 (t : ℝ) :
  (∀ x, f 2 x + f 2 (x + 5) ≥ t - 2) → t ≤ 5 := by sorry

end part1_part2_l2859_285941


namespace range_of_m_l2859_285901

-- Define the conditions
def p (x : ℝ) : Prop := (x - 2) * (x - 6) ≤ 32
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m ≤ 3) :=
by sorry

end range_of_m_l2859_285901


namespace max_negative_coefficients_no_real_roots_l2859_285972

def polynomial_coefficients (p : ℝ → ℝ) : List ℤ :=
  sorry

def has_no_real_roots (p : ℝ → ℝ) : Prop :=
  sorry

def count_negative_coefficients (coeffs : List ℤ) : ℕ :=
  sorry

theorem max_negative_coefficients_no_real_roots 
  (p : ℝ → ℝ) 
  (h1 : ∃ (coeffs : List ℤ), polynomial_coefficients p = coeffs ∧ coeffs.length = 2011)
  (h2 : has_no_real_roots p) :
  count_negative_coefficients (polynomial_coefficients p) ≤ 1005 :=
sorry

end max_negative_coefficients_no_real_roots_l2859_285972


namespace two_number_problem_l2859_285910

theorem two_number_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : 4 * y = 6 * x) (h5 : x + y = 36) : y = 21.6 := by
  sorry

end two_number_problem_l2859_285910


namespace candidate_a_votes_l2859_285979

def total_votes : ℕ := 560000
def invalid_vote_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 85 / 100

theorem candidate_a_votes : 
  ⌊(1 - invalid_vote_percentage) * candidate_a_percentage * total_votes⌋ = 404600 := by
  sorry

end candidate_a_votes_l2859_285979


namespace sum_reciprocals_lower_bound_l2859_285981

theorem sum_reciprocals_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 1 / y ≥ 4 := by
  sorry

end sum_reciprocals_lower_bound_l2859_285981


namespace sum_of_last_two_digits_of_9_pow_2002_l2859_285958

theorem sum_of_last_two_digits_of_9_pow_2002 : ∃ (n : ℕ), 9^2002 = 100 * n + 81 :=
sorry

end sum_of_last_two_digits_of_9_pow_2002_l2859_285958


namespace nonagon_diagonals_l2859_285903

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l2859_285903


namespace zachary_needs_money_l2859_285982

/-- The amount of additional money Zachary needs to buy football equipment -/
def additional_money_needed (football_cost shorts_cost shoes_cost socks_cost bottle_cost : ℝ)
  (shorts_count socks_count : ℕ) (current_money : ℝ) : ℝ :=
  let total_cost := football_cost + shorts_count * shorts_cost + shoes_cost +
                    socks_count * socks_cost + bottle_cost
  total_cost - current_money

/-- Theorem stating the additional money Zachary needs -/
theorem zachary_needs_money :
  additional_money_needed 3.756 2.498 11.856 1.329 7.834 2 4 24.042 = 9.716 := by
  sorry

end zachary_needs_money_l2859_285982


namespace qin_jiushao_v_1_l2859_285991

def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

def nested_f (x : ℝ) : ℝ := ((3 * x + 0) * x + 2) * x + 1 * x + 4

def v_1 (x : ℝ) : ℝ := 3 * x + 0

theorem qin_jiushao_v_1 : v_1 10 = 30 := by sorry

end qin_jiushao_v_1_l2859_285991


namespace cookie_banana_price_ratio_l2859_285977

theorem cookie_banana_price_ratio :
  ∀ (cookie_price banana_price : ℝ),
  cookie_price > 0 →
  banana_price > 0 →
  6 * cookie_price + 5 * banana_price > 0 →
  3 * (6 * cookie_price + 5 * banana_price) = 3 * cookie_price + 27 * banana_price →
  cookie_price / banana_price = 4 / 5 := by
sorry

end cookie_banana_price_ratio_l2859_285977


namespace gerald_chores_per_month_l2859_285965

/-- Represents the number of chores Gerald needs to do per month to save for baseball supplies. -/
def chores_per_month (monthly_expense : ℕ) (season_length : ℕ) (chore_price : ℕ) : ℕ :=
  let total_expense := monthly_expense * season_length
  let off_season_months := 12 - season_length
  let monthly_savings_needed := total_expense / off_season_months
  monthly_savings_needed / chore_price

/-- Theorem stating that Gerald needs to average 5 chores per month to save for his baseball supplies. -/
theorem gerald_chores_per_month :
  chores_per_month 100 4 10 = 5 := by
  sorry

#eval chores_per_month 100 4 10

end gerald_chores_per_month_l2859_285965


namespace apple_theorem_l2859_285921

/-- Represents the number of apples each person has -/
structure Apples where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions of the apple distribution problem -/
def apple_distribution (a : Apples) : Prop :=
  a.A + a.B + a.C < 100 ∧
  a.A - a.A / 6 - a.A / 4 = a.B + a.A / 6 ∧
  a.B + a.A / 6 = a.C + a.A / 4

theorem apple_theorem (a : Apples) (h : apple_distribution a) :
  a.A ≤ 48 ∧ a.B = a.C + 4 := by
  sorry

#check apple_theorem

end apple_theorem_l2859_285921


namespace six_planes_max_parts_l2859_285970

/-- The maximum number of parts that n planes can divide space into -/
def max_parts (n : ℕ) : ℕ := (n^3 + 5*n + 6) / 6

/-- Theorem: 6 planes can divide space into at most 42 parts -/
theorem six_planes_max_parts : max_parts 6 = 42 := by
  sorry

end six_planes_max_parts_l2859_285970


namespace fourth_week_sugar_l2859_285983

def sugar_amount (week : ℕ) : ℚ :=
  24 / (2 ^ (week - 1))

theorem fourth_week_sugar : sugar_amount 4 = 3 := by
  sorry

end fourth_week_sugar_l2859_285983


namespace no_intersection_points_l2859_285936

/-- Parabola 1 defined by y = 2x^2 + 3x - 4 -/
def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

/-- Parabola 2 defined by y = 3x^2 + 12 -/
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 12

/-- Theorem stating that the two parabolas have no real intersection points -/
theorem no_intersection_points : ∀ x : ℝ, parabola1 x ≠ parabola2 x := by
  sorry

end no_intersection_points_l2859_285936


namespace fraction_sum_simplification_l2859_285914

theorem fraction_sum_simplification : 1 / 462 + 23 / 42 = 127 / 231 := by
  sorry

end fraction_sum_simplification_l2859_285914


namespace actual_average_height_l2859_285944

/-- The number of boys in the class -/
def num_boys : ℕ := 50

/-- The initially calculated average height in cm -/
def initial_avg : ℝ := 175

/-- The incorrectly recorded heights of three boys in cm -/
def incorrect_heights : List ℝ := [155, 185, 170]

/-- The actual heights of the three boys in cm -/
def actual_heights : List ℝ := [145, 195, 160]

/-- The actual average height of the boys in the class -/
def actual_avg : ℝ := 174.8

theorem actual_average_height :
  let total_incorrect := num_boys * initial_avg
  let height_difference := (List.sum incorrect_heights) - (List.sum actual_heights)
  let total_correct := total_incorrect - height_difference
  (total_correct / num_boys) = actual_avg :=
sorry

end actual_average_height_l2859_285944


namespace find_m_l2859_285907

theorem find_m : ∃ m : ℕ, 62519 * m = 624877405 ∧ m = 9995 := by
  sorry

end find_m_l2859_285907


namespace negation_of_proposition_is_true_l2859_285966

theorem negation_of_proposition_is_true : 
  ∃ a : ℝ, (a > 2 ∧ a^2 ≥ 4) := by sorry

end negation_of_proposition_is_true_l2859_285966
