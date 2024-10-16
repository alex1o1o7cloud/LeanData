import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l443_44327

theorem equation_solution :
  ∀ a b c : ℤ,
  (∀ x : ℝ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔
  ((a = 2 ∧ b = -3 ∧ c = -4) ∨ (a = 8 ∧ b = -6 ∧ c = -7)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l443_44327


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_extremum_implies_monotonicity_l443_44343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / (x + 1)

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x - a) / ((x + 1)^2)

theorem tangent_slope_implies_a (a : ℝ) : 
  f_derivative a 1 = 1/2 → a = 1 := by sorry

theorem extremum_implies_monotonicity : 
  f_derivative 3 1 = 0 → 
  (∀ x < -3, f_derivative 3 x > 0) ∧ 
  (∀ x ∈ Set.Ioo (-3) (-1), f_derivative 3 x < 0) ∧
  (∀ x ∈ Set.Ioo (-1) 1, f_derivative 3 x < 0) ∧
  (∀ x > 1, f_derivative 3 x > 0) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_extremum_implies_monotonicity_l443_44343


namespace NUMINAMATH_CALUDE_shoe_probability_l443_44355

theorem shoe_probability (total_pairs : ℕ) (black_pairs brown_pairs gray_pairs : ℕ)
  (h1 : total_pairs = black_pairs + brown_pairs + gray_pairs)
  (h2 : total_pairs = 15)
  (h3 : black_pairs = 8)
  (h4 : brown_pairs = 4)
  (h5 : gray_pairs = 3) :
  let total_shoes := 2 * total_pairs
  let prob_black := (2 * black_pairs / total_shoes) * (black_pairs / (total_shoes - 1))
  let prob_brown := (2 * brown_pairs / total_shoes) * (brown_pairs / (total_shoes - 1))
  let prob_gray := (2 * gray_pairs / total_shoes) * (gray_pairs / (total_shoes - 1))
  prob_black + prob_brown + prob_gray = 89 / 435 :=
by sorry

end NUMINAMATH_CALUDE_shoe_probability_l443_44355


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l443_44377

/-- The function f(x) = 2bx - 3b + 1 has a zero point in (-1, 1) iff b ∈ (1/5, 1) -/
theorem zero_point_in_interval (b : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, 2 * b * x - 3 * b + 1 = 0) ↔ b ∈ Set.Ioo (1/5 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_zero_point_in_interval_l443_44377


namespace NUMINAMATH_CALUDE_problem_statement_l443_44318

theorem problem_statement (a b c d : ℝ) : 
  (Real.sqrt (a + b + c + d) + Real.sqrt (a^2 - 2*a + 3 - b) - Real.sqrt (b - c^2 + 4*c - 8) = 3) →
  (a - b + c - d = -7) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l443_44318


namespace NUMINAMATH_CALUDE_robert_basic_salary_l443_44391

/-- Represents Robert's financial situation --/
structure RobertFinances where
  basic_salary : ℝ
  total_sales : ℝ
  monthly_expenses : ℝ

/-- Calculates Robert's total earnings --/
def total_earnings (r : RobertFinances) : ℝ :=
  r.basic_salary + 0.1 * r.total_sales

/-- Theorem stating Robert's basic salary --/
theorem robert_basic_salary :
  ∃ (r : RobertFinances),
    r.total_sales = 23600 ∧
    r.monthly_expenses = 2888 ∧
    0.8 * (total_earnings r) = r.monthly_expenses ∧
    r.basic_salary = 1250 := by
  sorry


end NUMINAMATH_CALUDE_robert_basic_salary_l443_44391


namespace NUMINAMATH_CALUDE_sqrt_problem_proportional_function_l443_44368

-- Problem 1
theorem sqrt_problem : Real.sqrt 18 - Real.sqrt 24 / Real.sqrt 3 = Real.sqrt 2 := by sorry

-- Problem 2
theorem proportional_function (f : ℝ → ℝ) (h1 : ∀ x y, f (x + y) = f x + f y) (h2 : f 1 = 2) :
  ∀ x, f x = 2 * x := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_proportional_function_l443_44368


namespace NUMINAMATH_CALUDE_remainder_1632_times_2024_div_400_l443_44349

theorem remainder_1632_times_2024_div_400 : (1632 * 2024) % 400 = 368 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1632_times_2024_div_400_l443_44349


namespace NUMINAMATH_CALUDE_treasure_chest_rubies_l443_44346

/-- Given a treasure chest with gems, calculate the number of rubies. -/
theorem treasure_chest_rubies (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_rubies_l443_44346


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l443_44367

theorem absolute_value_inequality_solution :
  {y : ℝ | 3 ≤ |y - 4| ∧ |y - 4| ≤ 7} = {y : ℝ | (7 ≤ y ∧ y ≤ 11) ∨ (-3 ≤ y ∧ y ≤ 1)} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l443_44367


namespace NUMINAMATH_CALUDE_magic_balls_theorem_l443_44306

theorem magic_balls_theorem :
  ∃ (n : ℕ), 5 + 4 * n = 2005 :=
by sorry

end NUMINAMATH_CALUDE_magic_balls_theorem_l443_44306


namespace NUMINAMATH_CALUDE_triangular_grid_theorem_l443_44390

/-- Represents an infinite triangular grid with black unit equilateral triangles -/
structure TriangularGrid where
  black_triangles : ℕ

/-- Represents an equilateral triangle whose sides align with grid lines -/
structure AlignedTriangle where
  -- Add necessary fields

/-- Checks if there's exactly one black triangle outside the given aligned triangle -/
def has_one_outside (grid : TriangularGrid) (triangle : AlignedTriangle) : Prop :=
  sorry

/-- The main theorem statement -/
theorem triangular_grid_theorem (N : ℕ) :
  N > 0 →
  (∃ (grid : TriangularGrid) (triangle : AlignedTriangle),
    grid.black_triangles = N ∧ has_one_outside grid triangle) ↔
  N = 1 ∨ N = 2 ∨ N = 3 :=
sorry

end NUMINAMATH_CALUDE_triangular_grid_theorem_l443_44390


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l443_44331

theorem least_x_for_even_prime_fraction (x p : ℕ) : 
  x > 0 → 
  Prime p → 
  Prime (x / (12 * p)) → 
  Even (x / (12 * p)) → 
  (∀ y : ℕ, y > 0 → Prime p → Prime (y / (12 * p)) → Even (y / (12 * p)) → x ≤ y) → 
  x = 48 :=
sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l443_44331


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l443_44328

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents a way to cut the plywood --/
structure CutPattern where
  piece : Rectangle
  num_pieces : ℕ

theorem plywood_cut_perimeter_difference :
  ∀ (cuts : List CutPattern),
    (∀ c ∈ cuts, c.num_pieces = 6 ∧ c.piece.length * c.piece.width * 6 = 54) →
    (∃ c ∈ cuts, perimeter c.piece = 20) →
    (∀ c ∈ cuts, perimeter c.piece ≥ 15) →
    (∃ c ∈ cuts, perimeter c.piece = 15) →
    20 - 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l443_44328


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l443_44365

theorem greatest_integer_solution : 
  ∃ (x : ℤ), (8 - 6*x > 26) ∧ (∀ (y : ℤ), y > x → 8 - 6*y ≤ 26) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l443_44365


namespace NUMINAMATH_CALUDE_water_container_problem_l443_44324

/-- Given a container with capacity 120 liters, if adding 48 liters makes it 3/4 full,
    then the initial percentage of water in the container was 35%. -/
theorem water_container_problem :
  let capacity : ℝ := 120
  let added_water : ℝ := 48
  let final_fraction : ℝ := 3/4
  let initial_percentage : ℝ := (final_fraction * capacity - added_water) / capacity * 100
  initial_percentage = 35 := by sorry

end NUMINAMATH_CALUDE_water_container_problem_l443_44324


namespace NUMINAMATH_CALUDE_father_son_work_time_work_completed_in_three_days_l443_44388

/-- Given a task that takes 6 days for either a man or his son to complete alone,
    prove that they can complete it together in 3 days. -/
theorem father_son_work_time : ℝ → Prop :=
  fun total_work =>
    let man_rate := total_work / 6
    let son_rate := total_work / 6
    let combined_rate := man_rate + son_rate
    (total_work / combined_rate) = 3

/-- The main theorem stating that the work will be completed in 3 days -/
theorem work_completed_in_three_days (total_work : ℝ) (h : total_work > 0) :
  father_son_work_time total_work := by
  sorry

#check work_completed_in_three_days

end NUMINAMATH_CALUDE_father_son_work_time_work_completed_in_three_days_l443_44388


namespace NUMINAMATH_CALUDE_aruns_weight_lower_limit_l443_44305

theorem aruns_weight_lower_limit 
  (lower_bound : ℝ) 
  (upper_bound : ℝ) 
  (h1 : lower_bound > 65)
  (h2 : upper_bound ≤ 68)
  (h3 : (lower_bound + upper_bound) / 2 = 67) :
  lower_bound = 66 := by
sorry

end NUMINAMATH_CALUDE_aruns_weight_lower_limit_l443_44305


namespace NUMINAMATH_CALUDE_television_price_proof_l443_44387

theorem television_price_proof (discount_rate : ℝ) (final_price : ℝ) (num_tvs : ℕ) :
  discount_rate = 0.25 →
  final_price = 975 →
  num_tvs = 2 →
  ∃ (original_price : ℝ),
    original_price = 650 ∧
    final_price = (1 - discount_rate) * (num_tvs * original_price) :=
by sorry

end NUMINAMATH_CALUDE_television_price_proof_l443_44387


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_probability_nine_heads_in_twelve_flips_proof_l443_44362

/-- The probability of getting exactly 9 heads in 12 flips of a fair coin -/
theorem probability_nine_heads_in_twelve_flips : ℚ :=
  55 / 1024

/-- Proof that the probability of getting exactly 9 heads in 12 flips of a fair coin is 55/1024 -/
theorem probability_nine_heads_in_twelve_flips_proof :
  probability_nine_heads_in_twelve_flips = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_probability_nine_heads_in_twelve_flips_proof_l443_44362


namespace NUMINAMATH_CALUDE_digit_2007_in_2003_digit_number_l443_44315

/-- The sequence of digits formed by concatenating positive integers -/
def digit_sequence : ℕ → ℕ := sorry

/-- The function G(n) that calculates the number of digits preceding 10^n in the sequence -/
def G (n : ℕ) : ℕ := sorry

/-- The function f(n) that returns the number of digits in the number where the 10^n-th digit occurs -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 10^2007-th digit occurs in a 2003-digit number -/
theorem digit_2007_in_2003_digit_number : f 2007 = 2003 := by sorry

end NUMINAMATH_CALUDE_digit_2007_in_2003_digit_number_l443_44315


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l443_44351

/-- Given a quadratic equation (a-5)x^2 - 4x - 1 = 0 with real roots, prove that a ≥ 1 and a ≠ 5 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) → 
  (a ≥ 1 ∧ a ≠ 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l443_44351


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_point_l443_44348

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  passes_through : center.1 - 1 ^ 2 + center.2 ^ 2 = (center.1 + 1) ^ 2
  tangent_to_line : center.1 + 1 = ((center.1 - 1) ^ 2 + center.2 ^ 2).sqrt

-- Define the trajectory C
def trajectory (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define a point on the trajectory
structure PointOnTrajectory where
  point : ℝ × ℝ
  on_trajectory : trajectory point.1 point.2
  not_origin : point ≠ (0, 0)

-- Theorem statement
theorem trajectory_and_fixed_point 
  (M : MovingCircle) 
  (A B : PointOnTrajectory) 
  (h : A.point.1 * B.point.1 + A.point.2 * B.point.2 = 0) :
  ∃ (t : ℝ), 
    t * A.point.2 + (1 - t) * B.point.2 = 0 ∧ 
    t * A.point.1 + (1 - t) * B.point.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_point_l443_44348


namespace NUMINAMATH_CALUDE_sequence_with_geometric_differences_formula_l443_44381

def sequence_with_geometric_differences (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = (1/3)^(n-1)

theorem sequence_with_geometric_differences_formula (a : ℕ → ℝ) :
  sequence_with_geometric_differences a →
  ∀ n : ℕ, n ≥ 1 → a n = 3/2 * (1 - (1/3)^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_with_geometric_differences_formula_l443_44381


namespace NUMINAMATH_CALUDE_equation_is_linear_with_one_var_l443_44325

/-- A linear equation with one variable is an equation of the form ax + b = c, where a ≠ 0 and x is the variable. --/
def is_linear_equation_one_var (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, eq x ↔ a * x + b = c)

/-- The equation 4a - 1 = 8 --/
def equation (a : ℝ) : Prop := 4 * a - 1 = 8

theorem equation_is_linear_with_one_var : is_linear_equation_one_var equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_with_one_var_l443_44325


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l443_44375

theorem solution_set_abs_inequality (x : ℝ) :
  (|2*x + 3| < 1) ↔ (-2 < x ∧ x < -1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l443_44375


namespace NUMINAMATH_CALUDE_three_squares_balance_l443_44321

/-- A balance system with three symbols: triangle, square, and circle. -/
structure BalanceSystem where
  triangle : ℚ
  square : ℚ
  circle : ℚ

/-- The balance rules for the system. -/
def balance_rules (s : BalanceSystem) : Prop :=
  5 * s.triangle + 2 * s.square = 21 * s.circle ∧
  2 * s.triangle = s.square + 3 * s.circle

/-- The theorem to prove. -/
theorem three_squares_balance (s : BalanceSystem) :
  balance_rules s → 3 * s.square = 9 * s.circle :=
by
  sorry

end NUMINAMATH_CALUDE_three_squares_balance_l443_44321


namespace NUMINAMATH_CALUDE_tourist_walking_speed_l443_44308

/-- Represents the problem of calculating tourist walking speed -/
def TouristWalkingSpeedProblem (scheduled_arrival : ℝ) (actual_arrival : ℝ) (early_arrival : ℝ) (bus_speed : ℝ) : Prop :=
  ∃ (walking_speed : ℝ),
    walking_speed > 0 ∧
    scheduled_arrival > actual_arrival ∧
    early_arrival > 0 ∧
    bus_speed > 0 ∧
    let time_diff := scheduled_arrival - actual_arrival
    let encounter_time := time_diff - early_arrival
    let bus_travel_time := early_arrival / 2
    let distance := bus_speed * bus_travel_time
    walking_speed = distance / encounter_time ∧
    walking_speed = 5

/-- The main theorem stating the solution to the tourist walking speed problem -/
theorem tourist_walking_speed :
  TouristWalkingSpeedProblem 5 3.25 0.25 60 :=
by
  sorry


end NUMINAMATH_CALUDE_tourist_walking_speed_l443_44308


namespace NUMINAMATH_CALUDE_jerrie_carrie_difference_l443_44382

/-- The number of sit-ups Barney can perform in one minute -/
def barney_rate : ℕ := 45

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_rate : ℕ := 2 * barney_rate

/-- The number of minutes Barney performs sit-ups -/
def barney_time : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_time : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_time : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 510

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_rate : ℕ := (total_situps - (barney_rate * barney_time + carrie_rate * carrie_time)) / jerrie_time

theorem jerrie_carrie_difference :
  jerrie_rate - carrie_rate = 5 :=
sorry

end NUMINAMATH_CALUDE_jerrie_carrie_difference_l443_44382


namespace NUMINAMATH_CALUDE_chess_board_configurations_l443_44336

/-- Represents a chess board configuration -/
def ChessBoard := Fin 5 → Fin 5

/-- The number of ways to arrange n distinct items -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to place pawns on the board -/
def num_pawn_placements : ℕ := factorial 5

/-- The number of ways to assign distinct pawns to positions -/
def num_pawn_assignments : ℕ := factorial 5

/-- The total number of valid configurations -/
def total_configurations : ℕ := num_pawn_placements * num_pawn_assignments

/-- Theorem stating the total number of valid configurations -/
theorem chess_board_configurations :
  total_configurations = 14400 := by sorry

end NUMINAMATH_CALUDE_chess_board_configurations_l443_44336


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l443_44392

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 3 / Real.sqrt 6) * (Real.sqrt 4 / Real.sqrt 8) * (Real.sqrt 5 / Real.sqrt 9) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l443_44392


namespace NUMINAMATH_CALUDE_answer_choices_per_mc_question_l443_44357

/-- The number of ways to answer 3 true-false questions where all answers cannot be the same -/
def true_false_combinations : ℕ := 6

/-- The total number of ways to write the answer key -/
def total_combinations : ℕ := 96

/-- The number of multiple-choice questions -/
def num_mc_questions : ℕ := 2

theorem answer_choices_per_mc_question :
  ∃ n : ℕ, n > 0 ∧ true_false_combinations * n^num_mc_questions = total_combinations :=
sorry

end NUMINAMATH_CALUDE_answer_choices_per_mc_question_l443_44357


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_forms_triangle_circle_passes_through_M_and_N_with_center_on_y_axis_l443_44363

-- Define the points
def P : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (-2, 3)
def N : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 5

-- Theorem for the line
theorem line_passes_through_P_and_forms_triangle :
  line_equation P.1 P.2 ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (1/2 : ℝ) * a * b = 1/2) :=
sorry

-- Theorem for the circle
theorem circle_passes_through_M_and_N_with_center_on_y_axis :
  circle_equation M.1 M.2 ∧
  circle_equation N.1 N.2 ∧
  (∃ y : ℝ, circle_equation 0 y) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_forms_triangle_circle_passes_through_M_and_N_with_center_on_y_axis_l443_44363


namespace NUMINAMATH_CALUDE_square_field_side_length_l443_44354

theorem square_field_side_length (area : ℝ) (side : ℝ) : 
  area = 400 → side ^ 2 = area → side = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l443_44354


namespace NUMINAMATH_CALUDE_divisibility_theorem_l443_44301

theorem divisibility_theorem (N : ℕ) (h : N > 1) :
  ∃ k : ℤ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) := by sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l443_44301


namespace NUMINAMATH_CALUDE_customers_without_tip_l443_44378

theorem customers_without_tip (total_customers : ℕ) (total_tips : ℕ) (tip_per_customer : ℕ) :
  total_customers = 7 →
  total_tips = 6 →
  tip_per_customer = 3 →
  total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_customers_without_tip_l443_44378


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l443_44323

/-- The standard equation of a hyperbola with foci on the y-axis -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 - x^2 / b^2 = 1

/-- Theorem: The standard equation of a hyperbola with foci on the y-axis,
    semi-minor axis length of 4, and semi-focal distance of 6 -/
theorem hyperbola_standard_equation :
  let b : ℝ := 4  -- semi-minor axis length
  let c : ℝ := 6  -- semi-focal distance
  let a : ℝ := (c^2 - b^2).sqrt  -- semi-major axis length
  ∀ x y : ℝ, hyperbola_equation a b x y ↔ y^2 / 20 - x^2 / 16 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l443_44323


namespace NUMINAMATH_CALUDE_solution_set_inequality_l443_44356

theorem solution_set_inequality (x : ℝ) :
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l443_44356


namespace NUMINAMATH_CALUDE_remainder_11_power_2023_mod_19_l443_44332

theorem remainder_11_power_2023_mod_19 : 11^2023 % 19 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_2023_mod_19_l443_44332


namespace NUMINAMATH_CALUDE_students_not_excelling_l443_44334

theorem students_not_excelling (total : ℕ) (basketball : ℕ) (soccer : ℕ) (both : ℕ) : 
  total = 40 → basketball = 12 → soccer = 18 → both = 6 → 
  total - (basketball + soccer - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_not_excelling_l443_44334


namespace NUMINAMATH_CALUDE_three_in_A_l443_44344

def A : Set ℝ := {x | x ≤ Real.sqrt 13}

theorem three_in_A : 3 ∈ A := by sorry

end NUMINAMATH_CALUDE_three_in_A_l443_44344


namespace NUMINAMATH_CALUDE_polynomial_remainder_l443_44359

/-- The polynomial p(x) = x^3 - 2x^2 + x + 1 -/
def p (x : ℝ) : ℝ := x^3 - 2*x^2 + x + 1

/-- The remainder when p(x) is divided by (x-4) -/
def remainder : ℝ := p 4

theorem polynomial_remainder : remainder = 37 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l443_44359


namespace NUMINAMATH_CALUDE_parabola_shift_correct_l443_44326

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola := { a := 2, b := 0, c := 0 }

/-- Shift a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_correct :
  let shifted := shift_parabola original_parabola 1 5
  shifted.a = 2 ∧ shifted.b = -4 ∧ shifted.c = -5 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_correct_l443_44326


namespace NUMINAMATH_CALUDE_unique_solution_l443_44316

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_solution :
  ∃! x : ℕ, digit_product x = x^2 - 10*x - 22 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l443_44316


namespace NUMINAMATH_CALUDE_factorial_product_simplification_l443_44366

theorem factorial_product_simplification (a : ℝ) :
  (1 * 1) * (2 * 1 * a) * (3 * 2 * 1 * a^3) * (4 * 3 * 2 * 1 * a^6) * (5 * 4 * 3 * 2 * 1 * a^10) = 34560 * a^20 := by
  sorry

end NUMINAMATH_CALUDE_factorial_product_simplification_l443_44366


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l443_44304

theorem equilateral_triangle_perimeter (s : ℝ) (h_positive : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l443_44304


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l443_44396

/-- Calculates the total charge for a taxi trip with given conditions -/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_two_fifths_mile : ℚ) 
  (trip_distance : ℚ) (non_peak_discount : ℚ) (standard_car_discount : ℚ) : ℚ :=
  let base_charge := initial_fee + (trip_distance / (2/5)) * charge_per_two_fifths_mile
  let discount := base_charge * (non_peak_discount + standard_car_discount)
  base_charge - discount

/-- The total charge for the taxi trip is $4.95 -/
theorem taxi_charge_calculation :
  let initial_fee : ℚ := 235/100
  let charge_per_two_fifths_mile : ℚ := 35/100
  let trip_distance : ℚ := 36/10
  let non_peak_discount : ℚ := 7/100
  let standard_car_discount : ℚ := 3/100
  calculate_taxi_charge initial_fee charge_per_two_fifths_mile trip_distance 
    non_peak_discount standard_car_discount = 495/100 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l443_44396


namespace NUMINAMATH_CALUDE_polynomial_remainder_l443_44317

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 12*x^2 + 18*x - 22

theorem polynomial_remainder (x : ℝ) : 
  ∃ q : ℝ → ℝ, f x = (x - 4) * q x + 114 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l443_44317


namespace NUMINAMATH_CALUDE_octal_addition_l443_44350

/-- Addition of octal numbers -/
def octal_add (a b c : ℕ) : ℕ :=
  (a * 8^2 + (a / 8) * 8 + (a % 8)) +
  (b * 8^2 + (b / 8) * 8 + (b % 8)) +
  (c * 8^2 + (c / 8) * 8 + (c % 8))

/-- Conversion from decimal to octal -/
def to_octal (n : ℕ) : ℕ :=
  (n / 8^2) * 100 + ((n / 8) % 8) * 10 + (n % 8)

theorem octal_addition :
  to_octal (octal_add 176 725 63) = 1066 := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_l443_44350


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l443_44389

/-- Theorem about a triangle ABC with specific side lengths and angle properties -/
theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  -- Triangle inequality and positive side lengths
  a + b > c ∧ b + c > a ∧ c + a > b →
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- A, B, C form angles of a triangle
  A + B + C = π →
  -- Side lengths correspond to opposite angles
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Conclusions
  A = π / 6 ∧
  (1 / 2) * a * b * Real.sin C = (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l443_44389


namespace NUMINAMATH_CALUDE_greater_number_in_ratio_l443_44370

theorem greater_number_in_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a / b = 3 / 4 → a + b = 21 → max a b = 12 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_in_ratio_l443_44370


namespace NUMINAMATH_CALUDE_parallelogram_area_l443_44337

/-- The area of a parallelogram with vertices at (1, 1), (7, 1), (4, 9), and (10, 9) is 48 square units. -/
theorem parallelogram_area : ℝ := by
  -- Define the vertices
  let v1 : ℝ × ℝ := (1, 1)
  let v2 : ℝ × ℝ := (7, 1)
  let v3 : ℝ × ℝ := (4, 9)
  let v4 : ℝ × ℝ := (10, 9)

  -- Define the parallelogram
  let parallelogram := [v1, v2, v3, v4]

  -- Calculate the area
  let area := 48

  -- Prove that the area of the parallelogram is 48 square units
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l443_44337


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l443_44395

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l443_44395


namespace NUMINAMATH_CALUDE_mame_probability_theorem_l443_44310

/-- Represents a piece of paper with 8 possible surfaces (4 on each side) -/
structure Paper :=
  (surfaces : Fin 8)

/-- The probability of a specific surface being on top -/
def probability_on_top (paper : Paper) : ℚ := 1 / 8

/-- The surface with "MAME" written on it -/
def mame_surface : Fin 8 := 0

theorem mame_probability_theorem :
  probability_on_top { surfaces := mame_surface } = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_mame_probability_theorem_l443_44310


namespace NUMINAMATH_CALUDE_square_roots_theorem_l443_44386

theorem square_roots_theorem (a : ℝ) :
  (3 - a) ^ 2 = (2 * a + 1) ^ 2 → (3 - a) ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l443_44386


namespace NUMINAMATH_CALUDE_circular_road_circumference_sum_l443_44372

theorem circular_road_circumference_sum (R : ℝ) (h1 : R > 0) : 
  let r := R / 3
  let road_width := R - r
  road_width = 7 →
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circular_road_circumference_sum_l443_44372


namespace NUMINAMATH_CALUDE_matrix_equality_l443_44352

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = !![5, 1; -2, 4]) :
  B * A = !![10, 2; -4, 8] := by
sorry

end NUMINAMATH_CALUDE_matrix_equality_l443_44352


namespace NUMINAMATH_CALUDE_a_range_l443_44374

theorem a_range (P : ∀ x > 0, x + 4 / x ≥ a) 
                (q : ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0) : 
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_a_range_l443_44374


namespace NUMINAMATH_CALUDE_video_game_earnings_l443_44300

def total_games : ℕ := 16
def non_working_games : ℕ := 8
def price_per_game : ℕ := 7

theorem video_game_earnings : 
  (total_games - non_working_games) * price_per_game = 56 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l443_44300


namespace NUMINAMATH_CALUDE_electronics_store_cost_l443_44399

/-- Given the cost of 5 MP3 players and 8 headphones is $840, and the cost of one set of headphones
is $30, prove that the cost of 3 MP3 players and 4 headphones is $480. -/
theorem electronics_store_cost (mp3_cost headphones_cost : ℕ) : 
  5 * mp3_cost + 8 * headphones_cost = 840 →
  headphones_cost = 30 →
  3 * mp3_cost + 4 * headphones_cost = 480 := by
  sorry

end NUMINAMATH_CALUDE_electronics_store_cost_l443_44399


namespace NUMINAMATH_CALUDE_no_more_permutations_than_value_l443_44397

theorem no_more_permutations_than_value (b n : ℕ) : b > 1 → n > 1 → 
  let r := (Nat.log b n).succ
  let digits := Nat.digits b n
  (List.permutations digits).length ≤ n := by
  sorry

end NUMINAMATH_CALUDE_no_more_permutations_than_value_l443_44397


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l443_44360

/-- Calculates the gain percentage of a shopkeeper using a false weight --/
theorem shopkeeper_gain_percentage 
  (true_weight : ℝ) 
  (false_weight : ℝ) 
  (h1 : true_weight = 1000) 
  (h2 : false_weight = 980) : 
  (true_weight - false_weight) / true_weight * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l443_44360


namespace NUMINAMATH_CALUDE_infinitely_many_primes_congruent_one_mod_power_of_two_l443_44345

theorem infinitely_many_primes_congruent_one_mod_power_of_two (r : ℕ) (hr : r ≥ 1) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p ≡ 1 [MOD 2^r]} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_congruent_one_mod_power_of_two_l443_44345


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l443_44364

/-- Given points A, B, C, and the conditions that A' and B' lie on y = x,
    prove that the length of A'B' is 3√2/28 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 10) →
  C = (3, 6) →
  (∃ t : ℝ, A' = (t, t)) →
  (∃ s : ℝ, B' = (s, s)) →
  (∃ k : ℝ, A'.1 = k * (C.1 - A.1) + A.1 ∧ A'.2 = k * (C.2 - A.2) + A.2) →
  (∃ m : ℝ, B'.1 = m * (C.1 - B.1) + B.1 ∧ B'.2 = m * (C.2 - B.2) + B.2) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 3 * Real.sqrt 2 / 28 := by
sorry

end NUMINAMATH_CALUDE_length_of_AB_prime_l443_44364


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l443_44369

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l443_44369


namespace NUMINAMATH_CALUDE_longest_side_is_72_l443_44358

/-- A rectangle with specific properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 2880

/-- The longest side of a SpecialRectangle is 72 --/
theorem longest_side_is_72 (rect : SpecialRectangle) : 
  max rect.length rect.width = 72 := by
  sorry

#check longest_side_is_72

end NUMINAMATH_CALUDE_longest_side_is_72_l443_44358


namespace NUMINAMATH_CALUDE_fraction_simplification_l443_44394

theorem fraction_simplification : (20 : ℚ) / 19 * 15 / 28 * 76 / 45 = 95 / 21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l443_44394


namespace NUMINAMATH_CALUDE_tenth_odd_multiple_of_5_l443_44302

/-- The nth odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Predicate for a number being odd and a multiple of 5 -/
def isOddMultipleOf5 (k : ℕ) : Prop := k % 2 = 1 ∧ k % 5 = 0

theorem tenth_odd_multiple_of_5 : 
  (∃ (k : ℕ), k > 0 ∧ isOddMultipleOf5 k ∧ 
    (∃ (count : ℕ), count = 10 ∧ 
      (∀ (j : ℕ), j > 0 ∧ j < k → isOddMultipleOf5 j → 
        (∃ (m : ℕ), m < count ∧ nthOddMultipleOf5 m = j)))) → 
  nthOddMultipleOf5 10 = 95 :=
sorry

end NUMINAMATH_CALUDE_tenth_odd_multiple_of_5_l443_44302


namespace NUMINAMATH_CALUDE_system_solution_l443_44309

theorem system_solution (x y : ℝ) : 
  (x + y = 1 ∧ 2*x + y = 5) → (x = 4 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l443_44309


namespace NUMINAMATH_CALUDE_triangle_problem_l443_44303

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C)
  (h_a : a = Real.sqrt 5)
  (h_b : b = 3)
  (h_sin_C : Real.sin C = 2 * Real.sin A) : 
  c = 2 * Real.sqrt 5 ∧ Real.sin (2 * A - Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l443_44303


namespace NUMINAMATH_CALUDE_min_red_chips_l443_44339

theorem min_red_chips (w b r : ℕ) : 
  b ≥ (3 * w) / 4 →
  b ≤ r / 4 →
  w + b ≥ 75 →
  r ≥ 132 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ (3 * w') / 4 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 75) → r' ≥ 132 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l443_44339


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l443_44329

theorem smallest_fraction_greater_than_three_fourths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 3 / 4 →
    (73 : ℚ) / 97 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l443_44329


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_choose_branch_A_l443_44353

/-- Represents the grades of products --/
inductive Grade
| A
| B
| C
| D

/-- Represents the branches of the factory --/
inductive Branch
| A
| B

/-- Processing fee for each grade --/
def processingFee (g : Grade) : Int :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Processing cost for each branch --/
def processingCost (b : Branch) : Int :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Frequency distribution for each branch --/
def frequency (b : Branch) (g : Grade) : Int :=
  match b, g with
  | Branch.A, Grade.A => 40
  | Branch.A, Grade.B => 20
  | Branch.A, Grade.C => 20
  | Branch.A, Grade.D => 20
  | Branch.B, Grade.A => 28
  | Branch.B, Grade.B => 17
  | Branch.B, Grade.C => 34
  | Branch.B, Grade.D => 21

/-- Calculate average profit for a branch --/
def averageProfit (b : Branch) : Int :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem: Branch A has higher average profit than Branch B --/
theorem branch_A_more_profitable :
  averageProfit Branch.A > averageProfit Branch.B :=
by sorry

/-- Corollary: Factory should choose Branch A --/
theorem choose_branch_A :
  ∀ b : Branch, b ≠ Branch.A → averageProfit Branch.A > averageProfit b :=
by sorry

end NUMINAMATH_CALUDE_branch_A_more_profitable_choose_branch_A_l443_44353


namespace NUMINAMATH_CALUDE_count_rearranged_even_numbers_l443_44330

/-- The number of different even numbers that can be formed by rearranging the digits of 124669 -/
def rearrangedEvenNumbers : ℕ := 240

/-- The original number -/
def originalNumber : ℕ := 124669

/-- Theorem stating that the number of different even numbers formed by rearranging the digits of 124669 is 240 -/
theorem count_rearranged_even_numbers :
  rearrangedEvenNumbers = 240 ∧ originalNumber ≠ rearrangedEvenNumbers :=
by sorry

end NUMINAMATH_CALUDE_count_rearranged_even_numbers_l443_44330


namespace NUMINAMATH_CALUDE_total_length_QP_PL_l443_44333

-- Define the triangle XYZ
def X : ℝ × ℝ := (1, 4)
def Y : ℝ × ℝ := (0, 0)
def Z : ℝ × ℝ := (3, 0)

-- Define the altitudes
def XK : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = X.1 ∧ p.2 ≤ X.2}
def YL : Set (ℝ × ℝ) := {p : ℝ × ℝ | (Z.1 - X.1) * (p.1 - X.1) = (Z.2 - X.2) * (p.2 - X.2)}

-- Define the angle bisectors
def ZD : Set (ℝ × ℝ) := {p : ℝ × ℝ | (X.1 - Z.1) * (p.2 - Z.2) = (X.2 - Z.2) * (p.1 - Z.1)}
def XE : Set (ℝ × ℝ) := {p : ℝ × ℝ | (Y.1 - X.1) * (p.2 - X.2) = (Y.2 - X.2) * (p.1 - X.1)}

-- Define Q and P
def Q : ℝ × ℝ := (1, 1)
noncomputable def P : ℝ × ℝ := (0.5, 3)

-- Theorem statement
theorem total_length_QP_PL : 
  let qp_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let pl_length := Real.sqrt ((P.1 - (3/4))^2 + (P.2 - 3)^2)
  qp_length + pl_length = 1.5 := by sorry

end NUMINAMATH_CALUDE_total_length_QP_PL_l443_44333


namespace NUMINAMATH_CALUDE_average_income_proof_l443_44342

def income_days : Nat := 5

def daily_incomes : List ℝ := [400, 250, 650, 400, 500]

theorem average_income_proof :
  (daily_incomes.sum / income_days : ℝ) = 440 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l443_44342


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_l443_44335

def num_flips : ℕ := 8
def min_heads : ℕ := 6

-- Probability of getting at least min_heads in num_flips flips of a fair coin
def prob_at_least_heads : ℚ :=
  (Finset.sum (Finset.range (num_flips - min_heads + 1))
    (λ i => Nat.choose num_flips (num_flips - i))) / 2^num_flips

theorem prob_at_least_six_heads_in_eight_flips :
  prob_at_least_heads = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_l443_44335


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l443_44322

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  eval : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The theorem statement -/
theorem quadratic_coefficient (f : QuadraticFunction) 
  (h1 : f.eval 1 = 4)
  (h2 : f.eval (-2) = 3)
  (h3 : f.eval (-1) = 2)
  (h4 : ∀ x : ℝ, f.eval x ≤ f.eval (-1)) :
  f.a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l443_44322


namespace NUMINAMATH_CALUDE_regina_farm_correct_l443_44385

/-- Represents the farm animals and their selling prices -/
structure Farm where
  cows : ℕ
  pigs : ℕ
  cow_price : ℕ
  pig_price : ℕ

/-- Regina's farm satisfying the given conditions -/
def regina_farm : Farm where
  cows := 20  -- We'll prove this is correct
  pigs := 80  -- Four times the number of cows
  cow_price := 800
  pig_price := 400

/-- The total sale value of all animals on the farm -/
def total_sale_value (f : Farm) : ℕ :=
  f.cows * f.cow_price + f.pigs * f.pig_price

theorem regina_farm_correct :
  regina_farm.pigs = 4 * regina_farm.cows ∧
  total_sale_value regina_farm = 48000 := by
  sorry

#eval regina_farm.cows  -- Should output 20

end NUMINAMATH_CALUDE_regina_farm_correct_l443_44385


namespace NUMINAMATH_CALUDE_jason_seashells_l443_44341

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
  initial_seashells - given_seashells = 36 := by
  sorry

#check jason_seashells

end NUMINAMATH_CALUDE_jason_seashells_l443_44341


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l443_44384

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 3}
def B : Set ℝ := {x | x < -2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -2 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l443_44384


namespace NUMINAMATH_CALUDE_five_sqrt_two_gt_three_sqrt_three_l443_44307

theorem five_sqrt_two_gt_three_sqrt_three : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_five_sqrt_two_gt_three_sqrt_three_l443_44307


namespace NUMINAMATH_CALUDE_inequality_proof_l443_44398

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π / 2) : 
  π / 2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l443_44398


namespace NUMINAMATH_CALUDE_sum_of_two_angles_in_plane_l443_44371

/-- 
Given three angles meeting at a point in a plane, where one angle is 130°, 
prove that the sum of the other two angles is 230°.
-/
theorem sum_of_two_angles_in_plane (x y : ℝ) : 
  x + y + 130 = 360 → x + y = 230 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_angles_in_plane_l443_44371


namespace NUMINAMATH_CALUDE_event_guests_l443_44376

theorem event_guests (men : ℕ) (women : ℕ) (children : ℕ) : 
  men = 40 →
  women = men / 2 →
  children + 10 = 30 →
  men + women + children = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_event_guests_l443_44376


namespace NUMINAMATH_CALUDE_power_steering_count_l443_44373

theorem power_steering_count (total : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 65)
  (h2 : power_windows = 25)
  (h3 : both = 17)
  (h4 : neither = 12) :
  total - neither - (power_windows - both) = 45 :=
by sorry

end NUMINAMATH_CALUDE_power_steering_count_l443_44373


namespace NUMINAMATH_CALUDE_stick_swap_theorem_l443_44361

/-- Represents a set of three sticks --/
structure StickSet where
  stick1 : Real
  stick2 : Real
  stick3 : Real
  sum_is_one : stick1 + stick2 + stick3 = 1
  all_positive : stick1 > 0 ∧ stick2 > 0 ∧ stick3 > 0

/-- Checks if a triangle can be formed from a set of sticks --/
def can_form_triangle (s : StickSet) : Prop :=
  s.stick1 + s.stick2 > s.stick3 ∧
  s.stick1 + s.stick3 > s.stick2 ∧
  s.stick2 + s.stick3 > s.stick1

theorem stick_swap_theorem (vintik_initial shpuntik_initial vintik_final shpuntik_final : StickSet) :
  can_form_triangle vintik_initial →
  can_form_triangle shpuntik_initial →
  ¬can_form_triangle vintik_final →
  (∃ (x y : Real), 
    vintik_final.stick1 = vintik_initial.stick1 ∧
    vintik_final.stick2 = vintik_initial.stick2 ∧
    vintik_final.stick3 = y ∧
    shpuntik_final.stick1 = shpuntik_initial.stick1 ∧
    shpuntik_final.stick2 = shpuntik_initial.stick2 ∧
    shpuntik_final.stick3 = x ∧
    x + y = vintik_initial.stick3 + shpuntik_initial.stick3) →
  can_form_triangle shpuntik_final := by
  sorry

end NUMINAMATH_CALUDE_stick_swap_theorem_l443_44361


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l443_44311

/-- The mass of a man causing a boat to sink by a certain depth --/
def mass_of_man (boat_length boat_breadth sinking_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * water_density

/-- Theorem stating the mass of the man in the given problem --/
theorem man_mass_on_boat : 
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let sinking_depth : ℝ := 0.018  -- 1.8 cm converted to meters
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth sinking_depth water_density = 108 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_on_boat_l443_44311


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l443_44347

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = (1 : ℝ) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l443_44347


namespace NUMINAMATH_CALUDE_y_axis_symmetry_of_P_l443_44393

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The y-axis symmetry operation on a point -/
def yAxisSymmetry (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that the y-axis symmetry of P(0, -2, 3) is (0, -2, -3) -/
theorem y_axis_symmetry_of_P :
  let P : Point3D := { x := 0, y := -2, z := 3 }
  yAxisSymmetry P = { x := 0, y := -2, z := -3 } := by
  sorry

end NUMINAMATH_CALUDE_y_axis_symmetry_of_P_l443_44393


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l443_44320

theorem inverse_sum_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l443_44320


namespace NUMINAMATH_CALUDE_last_card_is_diamond_six_l443_44319

/-- Represents a playing card --/
inductive Card
| Joker : Bool → Card  -- True for Big Joker, False for Little Joker
| Number : Nat → Suit → Card
| Face : Face → Suit → Card

/-- Represents the suit of a card --/
inductive Suit
| Spades | Hearts | Diamonds | Clubs

/-- Represents face cards --/
inductive Face
| Jack | Queen | King

/-- Represents a deck of cards --/
def Deck := List Card

/-- Creates a standard deck of 54 cards in the specified order --/
def standardDeck : Deck := sorry

/-- Combines two decks --/
def combinedDeck (d1 d2 : Deck) : Deck := sorry

/-- Applies the discard-and-place rule to a deck --/
def applyRule (d : Deck) : Card := sorry

/-- Theorem: The last remaining card is the Diamond 6 --/
theorem last_card_is_diamond_six :
  let d1 := standardDeck
  let d2 := standardDeck
  let combined := combinedDeck d1 d2
  applyRule combined = Card.Number 6 Suit.Diamonds := by sorry

end NUMINAMATH_CALUDE_last_card_is_diamond_six_l443_44319


namespace NUMINAMATH_CALUDE_percentage_female_on_duty_l443_44379

def total_on_duty : ℕ := 200
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 1000

theorem percentage_female_on_duty :
  (female_ratio_on_duty * total_on_duty) / total_female_officers * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_female_on_duty_l443_44379


namespace NUMINAMATH_CALUDE_min_omega_l443_44314

theorem min_omega (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin (ω * x)) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), f x = -2) →
  ω ≥ 3/2 ∧ ∀ ω' ≥ 3/2, ∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) = -2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_l443_44314


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l443_44313

theorem max_value_sqrt_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  (Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c))) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l443_44313


namespace NUMINAMATH_CALUDE_year_spans_weeks_l443_44340

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of days in a common year -/
def daysInCommonYear : ℕ := 365

/-- Represents the number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- Represents the minimum number of days a week must have in a year to be counted -/
def minDaysInWeekForYear : ℕ := 6

/-- Definition of how many weeks a year can span -/
def weeksInYear : Set ℕ := {53, 54}

/-- Theorem stating the number of weeks a year can span -/
theorem year_spans_weeks : 
  ∀ (year : ℕ), 
    (year = daysInCommonYear ∨ year = daysInLeapYear) → 
    ∃ (weeks : ℕ), weeks ∈ weeksInYear ∧ 
      (weeks - 1) * daysInWeek + minDaysInWeekForYear ≤ year ∧
      year < (weeks + 1) * daysInWeek :=
sorry

end NUMINAMATH_CALUDE_year_spans_weeks_l443_44340


namespace NUMINAMATH_CALUDE_batting_bowling_average_change_l443_44383

/-- Represents a batsman's performance in a cricket inning -/
structure InningPerformance where
  runs : ℕ
  boundaries : ℕ
  sixes : ℕ
  strike_rate : ℝ
  wickets : ℕ

/-- Calculates the new batting average after an inning -/
def new_batting_average (initial_average : ℝ) (performance : InningPerformance) : ℝ :=
  initial_average + 5

/-- Calculates the new bowling average after an inning -/
def new_bowling_average (initial_average : ℝ) (performance : InningPerformance) : ℝ :=
  initial_average - 3

theorem batting_bowling_average_change 
  (A B : ℝ) 
  (performance : InningPerformance) 
  (h1 : performance.runs = 100) 
  (h2 : performance.boundaries = 12) 
  (h3 : performance.sixes = 2) 
  (h4 : performance.strike_rate = 130) 
  (h5 : performance.wickets = 1) :
  new_batting_average A performance = A + 5 ∧ 
  new_bowling_average B performance = B - 3 := by
  sorry


end NUMINAMATH_CALUDE_batting_bowling_average_change_l443_44383


namespace NUMINAMATH_CALUDE_smallest_number_l443_44338

theorem smallest_number (a b c d : ℝ) (h1 : a = 2) (h2 : b = -2.5) (h3 : c = 0) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l443_44338


namespace NUMINAMATH_CALUDE_last_digit_largest_power_of_3_dividing_27_factorial_l443_44312

/-- The largest power of 3 that divides n! -/
def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  sorry

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem last_digit_largest_power_of_3_dividing_27_factorial :
  lastDigit (3^(largestPowerOf3DividingFactorial 27)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_largest_power_of_3_dividing_27_factorial_l443_44312


namespace NUMINAMATH_CALUDE_simplify_expression_l443_44380

theorem simplify_expression : 
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l443_44380
