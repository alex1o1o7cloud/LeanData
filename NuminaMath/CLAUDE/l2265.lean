import Mathlib

namespace least_product_of_primes_above_30_l2265_226595

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p > 30 ∧ q > 30 ∧
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ r s : ℕ, is_prime r → is_prime s → r > 30 → s > 30 → r ≠ s → p * q ≤ r * s :=
sorry

end least_product_of_primes_above_30_l2265_226595


namespace gideon_marbles_fraction_l2265_226568

/-- The fraction of marbles Gideon gave to his sister -/
def fraction_given : ℚ := 3/4

theorem gideon_marbles_fraction :
  ∀ (f : ℚ),
  (100 : ℚ) = 100 →  -- Gideon has 100 marbles
  (45 : ℚ) = 45 →    -- Gideon is currently 45 years old
  2 * ((1 - f) * 100) = (45 + 5 : ℚ) →  -- After giving fraction f and doubling, he gets his age in 5 years
  f = fraction_given :=
by sorry

end gideon_marbles_fraction_l2265_226568


namespace power_product_equals_one_l2265_226553

theorem power_product_equals_one : (0.25 ^ 2023) * (4 ^ 2023) = 1 := by
  sorry

end power_product_equals_one_l2265_226553


namespace completing_square_transformation_l2265_226574

theorem completing_square_transformation (x : ℝ) : 
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end completing_square_transformation_l2265_226574


namespace accounting_course_count_l2265_226584

/-- Represents the number of employees who took an accounting course -/
def accounting_course : ℕ := sorry

/-- Represents the number of employees who took a finance course -/
def finance_course : ℕ := 14

/-- Represents the number of employees who took a marketing course -/
def marketing_course : ℕ := 15

/-- Represents the number of employees who took exactly two courses -/
def two_courses : ℕ := 10

/-- Represents the number of employees who took all three courses -/
def all_courses : ℕ := 1

/-- Represents the number of employees who took none of the courses -/
def no_courses : ℕ := 11

/-- The total number of employees -/
def total_employees : ℕ := 50

theorem accounting_course_count : accounting_course = 19 := by
  sorry

end accounting_course_count_l2265_226584


namespace spilled_bag_candies_l2265_226559

theorem spilled_bag_candies (bags : ℕ) (average : ℕ) (known_bags : List ℕ) : 
  bags = 8 → 
  average = 22 → 
  known_bags = [12, 14, 18, 22, 24, 26, 29] → 
  (List.sum known_bags + (bags - known_bags.length) * average - List.sum known_bags) = 31 := by
  sorry

end spilled_bag_candies_l2265_226559


namespace worker_completion_times_l2265_226545

def job_completion_time (worker1_time worker2_time : ℝ) : Prop :=
  (1 / worker1_time + 1 / worker2_time = 1 / 8) ∧
  (worker1_time = worker2_time - 12)

theorem worker_completion_times :
  ∃ (worker1_time worker2_time : ℝ),
    job_completion_time worker1_time worker2_time ∧
    worker1_time = 24 ∧
    worker2_time = 12 := by
  sorry

end worker_completion_times_l2265_226545


namespace complex_fraction_simplification_l2265_226591

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I :=
by sorry

end complex_fraction_simplification_l2265_226591


namespace floor_equation_solution_l2265_226550

theorem floor_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 90) : x = 10 := by
  sorry

end floor_equation_solution_l2265_226550


namespace subtraction_of_decimals_l2265_226505

/-- Subtraction of two specific decimal numbers -/
theorem subtraction_of_decimals : (678.90 : ℝ) - (123.45 : ℝ) = 555.55 := by
  sorry

end subtraction_of_decimals_l2265_226505


namespace probability_not_red_blue_purple_l2265_226519

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_not_red_blue_purple :
  let favorable_outcomes := total_balls - (red_balls + blue_balls + purple_balls)
  (favorable_outcomes : ℚ) / total_balls = 35 / 48 := by
  sorry

end probability_not_red_blue_purple_l2265_226519


namespace ellipse_and_line_properties_l2265_226514

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  eq : ℝ → ℝ → Prop := λ x y => x^2 / a^2 + y^2 / b^2 = 1
  vertex : ℝ × ℝ := (0, -1)
  focus_distance : ℝ := 3

/-- The line that intersects the ellipse -/
structure IntersectingLine where
  k : ℝ
  m : ℝ
  h_k : k ≠ 0
  eq : ℝ → ℝ → Prop := λ x y => y = k * x + m

/-- Main theorem about the ellipse and intersecting line -/
theorem ellipse_and_line_properties (e : Ellipse) (l : IntersectingLine) :
  (e.eq = λ x y => x^2 / 3 + y^2 = 1) ∧
  (∀ M N : ℝ × ℝ, e.eq M.1 M.2 → e.eq N.1 N.2 → l.eq M.1 M.2 → l.eq N.1 N.2 → M ≠ N →
    (dist M e.vertex = dist N e.vertex) → (1/2 < l.m ∧ l.m < 2)) := by
  sorry


end ellipse_and_line_properties_l2265_226514


namespace exponent_multiplication_l2265_226522

theorem exponent_multiplication (a : ℝ) : a * a^2 * a^3 = a^6 := by sorry

end exponent_multiplication_l2265_226522


namespace log_sum_equals_three_main_theorem_l2265_226503

theorem log_sum_equals_three : Real.log 8 + 3 * Real.log 5 = 3 * Real.log 10 := by
  sorry

theorem main_theorem : Real.log 8 + 3 * Real.log 5 = 3 := by
  sorry

end log_sum_equals_three_main_theorem_l2265_226503


namespace triangle_side_length_l2265_226577

theorem triangle_side_length (AB BC AC : ℝ) : 
  AB = 6 → BC = 4 → 2 < AC ∧ AC < 10 → AC = 5 → True :=
by
  sorry

end triangle_side_length_l2265_226577


namespace oranges_in_bin_l2265_226572

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 40 → thrown_away = 37 → final = 10 → final - (initial - thrown_away) = 7 := by
  sorry

end oranges_in_bin_l2265_226572


namespace parabola_directrix_l2265_226524

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 6 * x + 2

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -11/12

/-- Theorem: The directrix of the given parabola is y = -11/12 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    ∃ f : ℝ × ℝ, (f.2 - p.2)^2 = 4 * 3 * ((p.1 - f.1)^2 + (p.2 - d)^2)) :=
by sorry

end parabola_directrix_l2265_226524


namespace fraction_equality_l2265_226549

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 4 * y) = 3) : 
  (2 * x + 4 * y) / (4 * x - 2 * y) = 9 / 13 := by
  sorry

end fraction_equality_l2265_226549


namespace factor_tree_problem_l2265_226590

theorem factor_tree_problem (X Y Z F G : ℕ) :
  X = Y * Z ∧
  Y = 5 * F ∧
  Z = 7 * G ∧
  F = 5 * 3 ∧
  G = 7 * 3 →
  X = 11025 :=
by sorry

end factor_tree_problem_l2265_226590


namespace parabola_directrix_l2265_226555

/-- Represents a parabola with equation x² = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => y = -1

theorem parabola_directrix (p : Parabola) : 
  directrix p = fun y => y = -1 := by sorry

end parabola_directrix_l2265_226555


namespace ellipse_property_l2265_226585

-- Define the basic concepts
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a moving point
def MovingPoint := ℝ → Point

-- Define the concept of an ellipse
def is_ellipse (trajectory : MovingPoint) : Prop := sorry

-- Define the concept of constant sum of distances
def constant_sum_distances (trajectory : MovingPoint) (f1 f2 : Point) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, distance (trajectory t) f1 + distance (trajectory t) f2 = k

-- State the theorem
theorem ellipse_property :
  (∀ trajectory : MovingPoint, ∀ f1 f2 : Point,
    is_ellipse trajectory → constant_sum_distances trajectory f1 f2) ∧
  (∃ trajectory : MovingPoint, ∃ f1 f2 : Point,
    constant_sum_distances trajectory f1 f2 ∧ ¬is_ellipse trajectory) :=
sorry

end ellipse_property_l2265_226585


namespace max_decreasing_votes_is_five_l2265_226592

/-- A movie rating system with integer ratings from 0 to 10 -/
structure MovieRating where
  ratings : List ℕ
  valid_ratings : ∀ r ∈ ratings, r ≤ 10

/-- Calculate the current rating as the average of all ratings -/
def current_rating (mr : MovieRating) : ℚ :=
  (mr.ratings.sum : ℚ) / mr.ratings.length

/-- The maximum number of consecutive votes that can decrease the rating by 1 each time -/
def max_consecutive_decreasing_votes (mr : MovieRating) : ℕ :=
  sorry

/-- Theorem: The maximum number of consecutive votes that can decrease 
    an integer rating by 1 each time is 5 -/
theorem max_decreasing_votes_is_five (mr : MovieRating) 
  (h : ∃ n : ℕ, current_rating mr = n) :
  max_consecutive_decreasing_votes mr = 5 :=
sorry

end max_decreasing_votes_is_five_l2265_226592


namespace no_real_solutions_l2265_226576

theorem no_real_solutions : ¬∃ x : ℝ, x + 36 / (x - 3) = -9 := by
  sorry

end no_real_solutions_l2265_226576


namespace exists_m_divides_polynomial_l2265_226525

theorem exists_m_divides_polynomial (p : ℕ) (h_prime : Nat.Prime p) (h_cong : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ p ∣ (m^3 + m^2 - 2*m - 1) := by
  sorry

end exists_m_divides_polynomial_l2265_226525


namespace number_equality_l2265_226534

theorem number_equality (x : ℝ) : (0.4 * x = 0.25 * 80) → x = 50 := by
  sorry

end number_equality_l2265_226534


namespace imaginary_power_sum_l2265_226594

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_power_sum : i^23 + i^45 = 0 := by
  sorry

end imaginary_power_sum_l2265_226594


namespace product_of_fractions_l2265_226526

theorem product_of_fractions :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end product_of_fractions_l2265_226526


namespace power_division_equality_l2265_226551

theorem power_division_equality : (2 ^ 24) / (8 ^ 3) = 32768 := by
  sorry

end power_division_equality_l2265_226551


namespace phillips_money_l2265_226560

/-- The amount of money Phillip's mother gave him -/
def total_money : ℕ := sorry

/-- The amount Phillip spent on oranges -/
def oranges_cost : ℕ := 14

/-- The amount Phillip spent on apples -/
def apples_cost : ℕ := 25

/-- The amount Phillip spent on candy -/
def candy_cost : ℕ := 6

/-- The amount Phillip has left -/
def money_left : ℕ := 50

/-- Theorem stating that the total money given by Phillip's mother
    is equal to the sum of his expenses plus the amount left -/
theorem phillips_money :
  total_money = oranges_cost + apples_cost + candy_cost + money_left :=
sorry

end phillips_money_l2265_226560


namespace remainder_twelve_thousand_one_hundred_eleven_div_three_l2265_226530

theorem remainder_twelve_thousand_one_hundred_eleven_div_three : 
  12111 % 3 = 0 := by
sorry

end remainder_twelve_thousand_one_hundred_eleven_div_three_l2265_226530


namespace polynomial_equality_constant_l2265_226589

theorem polynomial_equality_constant (s : ℚ) : 
  (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 
    15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) → 
  s = -95/9 := by
sorry

end polynomial_equality_constant_l2265_226589


namespace max_circumference_error_l2265_226523

def actual_radius : ℝ := 15
def max_error_rate : ℝ := 0.25

theorem max_circumference_error :
  let min_measured_radius := actual_radius * (1 - max_error_rate)
  let max_measured_radius := actual_radius * (1 + max_error_rate)
  let actual_circumference := 2 * Real.pi * actual_radius
  let min_computed_circumference := 2 * Real.pi * min_measured_radius
  let max_computed_circumference := 2 * Real.pi * max_measured_radius
  let min_error := (actual_circumference - min_computed_circumference) / actual_circumference
  let max_error := (max_computed_circumference - actual_circumference) / actual_circumference
  max min_error max_error = max_error_rate :=
by sorry

end max_circumference_error_l2265_226523


namespace cubic_function_sign_properties_l2265_226538

/-- Given a cubic function with three real roots, prove specific sign properties -/
theorem cubic_function_sign_properties 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = x^3 - 6*x^2 + 9*x - a*b*c)
  (h2 : a < b ∧ b < c)
  (h3 : f a = 0 ∧ f b = 0 ∧ f c = 0) :
  f 0 * f 1 < 0 ∧ f 0 * f 3 > 0 := by
sorry

end cubic_function_sign_properties_l2265_226538


namespace mikes_remaining_nickels_l2265_226508

/-- Given Mike's initial number of nickels and the number borrowed by his dad,
    proves that the number of nickels Mike has now is the difference between
    the initial number and the borrowed number. -/
theorem mikes_remaining_nickels
  (initial_nickels : ℕ)
  (borrowed_nickels : ℕ)
  (h1 : initial_nickels = 87)
  (h2 : borrowed_nickels = 75)
  : initial_nickels - borrowed_nickels = 12 := by
  sorry

end mikes_remaining_nickels_l2265_226508


namespace triangle_property_l2265_226571

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((Real.sin A + Real.sin B) * (a - b) = c * (Real.sin C - Real.sin B)) →
  (D.1 * (B / (B + C)) + D.2 * (C / (B + C)) = 2) →
  (A / 2 = Real.arctan ((D.2 - D.1) / 2)) →
  (A = π / 3 ∧ 4 * Real.sqrt 3 / 3 ≤ (1 / 2) * a * b * Real.sin C) :=
by sorry

end triangle_property_l2265_226571


namespace barbara_has_winning_strategy_l2265_226562

/-- A game played on a matrix where two players alternately fill entries --/
structure MatrixGame where
  n : ℕ
  entries : Fin n → Fin n → ℝ

/-- A strategy for the second player in the matrix game --/
def SecondPlayerStrategy (n : ℕ) := 
  (Fin n → Fin n → ℝ) → Fin n → Fin n → ℝ

/-- The determinant of a matrix is zero if two of its rows are identical --/
axiom det_zero_if_identical_rows {n : ℕ} (M : Fin n → Fin n → ℝ) :
  (∃ i j, i ≠ j ∧ (∀ k, M i k = M j k)) → Matrix.det M = 0

/-- The second player can always make two rows identical --/
axiom second_player_can_make_identical_rows (n : ℕ) :
  ∃ (strategy : SecondPlayerStrategy n),
    ∀ (game : MatrixGame),
    game.n = n →
    ∃ i j, i ≠ j ∧ (∀ k, game.entries i k = game.entries j k)

theorem barbara_has_winning_strategy :
  ∃ (strategy : SecondPlayerStrategy 2008),
    ∀ (game : MatrixGame),
    game.n = 2008 →
    Matrix.det game.entries = 0 := by
  sorry

end barbara_has_winning_strategy_l2265_226562


namespace ellipse_a_value_l2265_226583

-- Define the ellipse equation
def ellipse_equation (a x y : ℝ) : Prop := x^2 / a^2 + y^2 / 2 = 1

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem ellipse_a_value :
  ∃ (a : ℝ), 
    (∀ (x y : ℝ), ellipse_equation a x y → 
      ∃ (c : ℝ), c = 2 ∧ a^2 = 2 + c^2) ∧ 
    (∀ (x y : ℝ), parabola_equation x y → 
      ∃ (f : ℝ × ℝ), f = parabola_focus) →
  a = Real.sqrt 6 ∨ a = -Real.sqrt 6 :=
sorry

end ellipse_a_value_l2265_226583


namespace ball_box_arrangements_l2265_226511

/-- The number of different arrangements of 4 balls in 4 boxes -/
def arrangements (n : ℕ) : ℕ := sorry

/-- The number of arrangements where exactly one box contains 2 balls -/
def one_box_two_balls : ℕ := arrangements 1

/-- The number of arrangements where exactly two boxes are left empty -/
def two_boxes_empty : ℕ := arrangements 2

theorem ball_box_arrangements :
  (one_box_two_balls = 144) ∧ (two_boxes_empty = 84) := by sorry

end ball_box_arrangements_l2265_226511


namespace earth_sun_distance_in_scientific_notation_l2265_226542

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3 * (10 ^ 8)

/-- The time it takes for sunlight to reach Earth in seconds -/
def time_to_earth : ℝ := 5 * (10 ^ 2)

/-- The distance between Earth and Sun in meters -/
def earth_sun_distance : ℝ := speed_of_light * time_to_earth

theorem earth_sun_distance_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), a ≥ 1 ∧ a < 10 ∧ earth_sun_distance = a * (10 ^ n) ∧ a = 1.5 ∧ n = 11 :=
sorry

end earth_sun_distance_in_scientific_notation_l2265_226542


namespace soap_box_length_l2265_226500

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Theorem: Given the carton and soap box dimensions, if 360 soap boxes fit exactly in the carton,
    then the length of a soap box is 7 inches -/
theorem soap_box_length
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton.length = 30 ∧ carton.width = 42 ∧ carton.height = 60)
  (h2 : soap.width = 6 ∧ soap.height = 5)
  (h3 : boxVolume carton = 360 * boxVolume soap) :
  soap.length = 7 := by
  sorry

end soap_box_length_l2265_226500


namespace quadratic_equal_roots_equation_C_has_equal_roots_l2265_226581

theorem quadratic_equal_roots (a b c : ℝ) (h : a ≠ 0) :
  (b^2 - 4*a*c = 0) ↔ ∃! x, a*x^2 + b*x + c = 0 :=
sorry

theorem equation_C_has_equal_roots :
  ∃! x, x^2 + 12*x + 36 = 0 :=
sorry

end quadratic_equal_roots_equation_C_has_equal_roots_l2265_226581


namespace product_of_numbers_l2265_226598

theorem product_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 20) 
  (sum_squares_condition : x^2 + y^2 = 200) : 
  x * y = 100 := by
sorry

end product_of_numbers_l2265_226598


namespace point_of_tangency_parabolas_l2265_226569

/-- The point of tangency of two parabolas -/
theorem point_of_tangency_parabolas :
  let f (x : ℝ) := x^2 + 8*x + 15
  let g (y : ℝ) := y^2 + 16*y + 63
  let point : ℝ × ℝ := (-7/2, -15/2)
  (f (point.1) = point.2 ∧ g (point.2) = point.1) ∧
  ∀ x y : ℝ, (f x = y ∧ g y = x) → (x, y) = point :=
by sorry


end point_of_tangency_parabolas_l2265_226569


namespace john_chess_probability_l2265_226515

theorem john_chess_probability (p_win : ℚ) (h : p_win = 2 / 5) : 1 - p_win = 3 / 5 := by
  sorry

end john_chess_probability_l2265_226515


namespace function_equality_l2265_226596

theorem function_equality (f g : ℕ+ → ℕ+) 
  (h1 : ∀ n : ℕ+, f (g n) = f n + 1) 
  (h2 : ∀ n : ℕ+, g (f n) = g n + 1) : 
  ∀ n : ℕ+, f n = g n := by
sorry

end function_equality_l2265_226596


namespace river_problem_solution_l2265_226528

/-- Represents the problem of a boat traveling along a river -/
structure RiverProblem where
  total_distance : ℝ
  total_time : ℝ
  upstream_distance : ℝ
  downstream_distance : ℝ
  hTotalDistance : total_distance = 10
  hTotalTime : total_time = 5
  hEqualTime : upstream_distance / downstream_distance = 2 / 3

/-- Solution to the river problem -/
structure RiverSolution where
  current_speed : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Theorem stating the solution to the river problem -/
theorem river_problem_solution (p : RiverProblem) : 
  ∃ (s : RiverSolution), 
    s.current_speed = 5 / 12 ∧ 
    s.upstream_time = 3 ∧ 
    s.downstream_time = 2 := by
  sorry

end river_problem_solution_l2265_226528


namespace negation_of_universal_proposition_l2265_226546

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1) ↔ (∃ x : ℝ, x ≤ 1) := by sorry

end negation_of_universal_proposition_l2265_226546


namespace inequality_proof_l2265_226564

theorem inequality_proof (x : ℝ) (hx : x > 0) : 1 + x^2018 ≥ (2*x)^2017 / (1 + x)^2016 := by
  sorry

end inequality_proof_l2265_226564


namespace cube_root_of_1331_l2265_226504

theorem cube_root_of_1331 (y : ℝ) (h1 : y > 0) (h2 : y^3 = 1331) : y = 11 := by
  sorry

end cube_root_of_1331_l2265_226504


namespace complex_multiplication_l2265_226552

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (1 - i) * i = 1 + i := by
  sorry

end complex_multiplication_l2265_226552


namespace proposition_q_undetermined_l2265_226517

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) ∧ ¬(q ∧ ¬q) :=
by sorry

end proposition_q_undetermined_l2265_226517


namespace smallest_integer_in_set_l2265_226597

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → 
  n ≥ 0 :=
by
  sorry

end smallest_integer_in_set_l2265_226597


namespace barycentric_coords_exist_and_unique_l2265_226588

-- Define a triangle in 2D space
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

-- Define barycentric coordinates
structure BarycentricCoords where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- State the theorem
theorem barycentric_coords_exist_and_unique (t : Triangle) (X : Point) :
  ∃! (b : BarycentricCoords),
    b.m₁ + b.m₂ + b.m₃ = 1 ∧
    X = (b.m₁ * t.A₁.1 + b.m₂ * t.A₂.1 + b.m₃ * t.A₃.1,
         b.m₁ * t.A₁.2 + b.m₂ * t.A₂.2 + b.m₃ * t.A₃.2) :=
  sorry

end barycentric_coords_exist_and_unique_l2265_226588


namespace complex_equation_solution_l2265_226566

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : 
  a + b * i = -1 + 2 * i :=
sorry

end complex_equation_solution_l2265_226566


namespace class_average_weight_l2265_226537

theorem class_average_weight (students_A students_B : ℕ) (avg_weight_A avg_weight_B : ℝ) :
  students_A = 36 →
  students_B = 24 →
  avg_weight_A = 30 →
  avg_weight_B = 30 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 30 :=
by
  sorry

end class_average_weight_l2265_226537


namespace compound_interest_problem_l2265_226509

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem compound_interest_problem :
  let principal : ℝ := 3600
  let rate : ℝ := 0.05
  let time : ℕ := 2
  let final_amount : ℝ := 3969
  compound_interest principal rate time = final_amount := by
  sorry

end compound_interest_problem_l2265_226509


namespace gcd_sequence_limit_l2265_226586

theorem gcd_sequence_limit (n : ℕ) : 
  ∃ N : ℕ, ∀ m : ℕ, m ≥ N → 
    Nat.gcd (100 + 2 * m^2) (100 + 2 * (m + 1)^2) = 1 := by
  sorry

#check gcd_sequence_limit

end gcd_sequence_limit_l2265_226586


namespace vector_magnitude_l2265_226531

def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (-1, 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_magnitude (t : ℝ) :
  parallel (a t) b →
  ‖(a t - b)‖ = 3 * Real.sqrt 5 :=
by sorry

end vector_magnitude_l2265_226531


namespace inequality_system_subset_circle_l2265_226561

theorem inequality_system_subset_circle (m : ℝ) :
  m > 0 →
  (∀ x y : ℝ, x - 2*y + 5 ≥ 0 ∧ 3 - x ≥ 0 ∧ x + y ≥ 0 → x^2 + y^2 ≤ m^2) →
  m ≥ 3 * Real.sqrt 2 :=
by sorry

end inequality_system_subset_circle_l2265_226561


namespace painted_cubes_count_l2265_226556

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  red_faces : Nat
  blue_faces : Nat

/-- Calculates the number of painted unit cubes in a PaintedCube -/
def num_painted_cubes (cube : PaintedCube) : Nat :=
  cube.size ^ 3 - (cube.size - 2) ^ 3

/-- Theorem: In a 5x5x5 cube with 2 red faces and 4 blue faces, 101 unit cubes are painted -/
theorem painted_cubes_count (cube : PaintedCube) 
  (h_size : cube.size = 5)
  (h_red : cube.red_faces = 2)
  (h_blue : cube.blue_faces = 4) :
  num_painted_cubes cube = 101 := by
  sorry

#check painted_cubes_count

end painted_cubes_count_l2265_226556


namespace binomial_coefficient_two_l2265_226529

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2265_226529


namespace oilseed_germination_theorem_l2265_226580

/-- The average germination rate of oilseeds -/
def average_germination_rate : ℝ := 0.96

/-- The total number of oilseeds -/
def total_oilseeds : ℕ := 2000

/-- The number of oilseeds that cannot germinate -/
def non_germinating_oilseeds : ℕ := 80

/-- Theorem stating that given the average germination rate,
    approximately 80 out of 2000 oilseeds cannot germinate -/
theorem oilseed_germination_theorem :
  ⌊(1 - average_germination_rate) * total_oilseeds⌋ = non_germinating_oilseeds :=
sorry

end oilseed_germination_theorem_l2265_226580


namespace tangent_line_equations_l2265_226502

/-- The function f(x) = x³ + 2 -/
def f (x : ℝ) : ℝ := x^3 + 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_equations (x : ℝ) :
  /- Part 1: Tangent line equation at x = 1 -/
  (∀ y : ℝ, (y - f 1) = f' 1 * (x - 1) ↔ 3 * x - y = 0) ∧
  /- Part 2: Tangent line equation passing through (0, 4) -/
  (∃ t : ℝ, t^3 + 2 = f t ∧
            4 - (t^3 + 2) = f' t * (0 - t) ∧
            (∀ y : ℝ, (y - f t) = f' t * (x - t) ↔ 3 * x - y + 4 = 0)) :=
by sorry

end tangent_line_equations_l2265_226502


namespace inequality_equivalence_l2265_226532

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 6 ∪ {6} := by sorry

end inequality_equivalence_l2265_226532


namespace sues_family_travel_l2265_226547

/-- Given a constant speed and travel time, calculates the distance traveled -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Sue's family traveled 300 miles to the campground -/
theorem sues_family_travel : distance_traveled 60 5 = 300 := by
  sorry

end sues_family_travel_l2265_226547


namespace triangle_shortest_side_l2265_226573

theorem triangle_shortest_side 
  (a b c : ℕ) 
  (h : ℕ) 
  (area : ℕ) 
  (ha : a = 24) 
  (hperim : a + b + c = 55) 
  (harea : area = a * h / 2) 
  (hherons : area^2 = ((a + b + c) / 2) * (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) : 
  min b c = 14 := by
sorry

end triangle_shortest_side_l2265_226573


namespace sum_cos_dihedral_angles_eq_one_l2265_226578

/-- A trihedral angle is a three-dimensional figure formed by three planes intersecting at a point. -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  plane_angles : Fin 3 → ℝ
  /-- The sum of the plane angles is 180° (π radians) -/
  sum_plane_angles : (plane_angles 0) + (plane_angles 1) + (plane_angles 2) = π

/-- The dihedral angles of a trihedral angle -/
def dihedral_angles (t : TrihedralAngle) : Fin 3 → ℝ := sorry

/-- Theorem: For a trihedral angle with plane angles summing to 180°, 
    the sum of the cosines of its dihedral angles is equal to 1 -/
theorem sum_cos_dihedral_angles_eq_one (t : TrihedralAngle) : 
  (Real.cos (dihedral_angles t 0)) + (Real.cos (dihedral_angles t 1)) + (Real.cos (dihedral_angles t 2)) = 1 := by
  sorry

end sum_cos_dihedral_angles_eq_one_l2265_226578


namespace remainder_of_large_number_l2265_226599

theorem remainder_of_large_number (n : ℕ) (d : ℕ) (h : n = 123456789012 ∧ d = 126) :
  n % d = 18 := by
  sorry

end remainder_of_large_number_l2265_226599


namespace perfect_cube_units_digits_l2265_226510

theorem perfect_cube_units_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n : ℕ, n ∈ S ↔ ∃ m : ℕ, n = m^3 % 10) ∧ 
    Finset.card S = 10 :=
sorry

end perfect_cube_units_digits_l2265_226510


namespace quadratic_inequality_solution_set_l2265_226507

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + 2*x - 3 > 0 ↔ x < -3 ∨ x > 1 := by sorry

end quadratic_inequality_solution_set_l2265_226507


namespace circle_radius_circle_radius_proof_l2265_226540

/-- The radius of a circle with center (2, -3) passing through (5, -7) is 5 -/
theorem circle_radius : ℝ → Prop :=
  fun r : ℝ =>
    let center : ℝ × ℝ := (2, -3)
    let point : ℝ × ℝ := (5, -7)
    (center.1 - point.1)^2 + (center.2 - point.2)^2 = r^2 → r = 5

/-- Proof of the theorem -/
theorem circle_radius_proof : circle_radius 5 := by
  sorry

end circle_radius_circle_radius_proof_l2265_226540


namespace smaller_fraction_problem_l2265_226557

theorem smaller_fraction_problem (x y : ℚ) 
  (sum_cond : x + y = 7/8)
  (prod_cond : x * y = 1/12) :
  min x y = (7 - Real.sqrt 17) / 16 := by
  sorry

end smaller_fraction_problem_l2265_226557


namespace factorization_equality_l2265_226501

theorem factorization_equality (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) := by
  sorry

end factorization_equality_l2265_226501


namespace neighbor_birth_year_l2265_226520

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year ≤ 9999

def first_two_digits (year : ℕ) : ℕ :=
  year / 100

def last_two_digits (year : ℕ) : ℕ :=
  year % 100

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def diff_of_digits (n : ℕ) : ℕ :=
  (n / 10) - (n % 10)

theorem neighbor_birth_year :
  ∀ year : ℕ, is_valid_year year →
    (sum_of_digits (first_two_digits year) = diff_of_digits (last_two_digits year)) →
    year = 1890 :=
by sorry

end neighbor_birth_year_l2265_226520


namespace mistaken_subtraction_l2265_226516

theorem mistaken_subtraction (x : ℤ) : x - 59 = 43 → x - 46 = 56 := by
  sorry

end mistaken_subtraction_l2265_226516


namespace cinema_seating_arrangement_l2265_226558

def number_of_arrangements (n : ℕ) (must_together : ℕ) (must_not_together : ℕ) : ℕ :=
  (must_together.factorial * (n - must_together + 1).factorial) -
  (must_together.factorial * must_not_together.factorial * (n - must_together - must_not_together + 2).factorial)

theorem cinema_seating_arrangement :
  number_of_arrangements 6 2 2 = 144 := by
  sorry

end cinema_seating_arrangement_l2265_226558


namespace parallel_lines_parameter_sum_l2265_226575

/-- Given two parallel lines with a specific distance between them, prove that the sum of their parameters is either 3 or -3. -/
theorem parallel_lines_parameter_sum (n m : ℝ) : 
  (∀ x y : ℝ, 2 * x + y + n = 0 ↔ 4 * x + m * y - 4 = 0) →  -- parallelism condition
  (∃ d : ℝ, d = (3 / 5) * Real.sqrt 5 ∧ 
    d = |n + 2| / Real.sqrt 5) →  -- distance condition
  m = 2 →  -- parallelism implies m = 2
  (m + n = 3 ∨ m + n = -3) :=
by sorry

end parallel_lines_parameter_sum_l2265_226575


namespace placards_per_person_l2265_226582

def total_placards : ℕ := 5682
def people_entered : ℕ := 2841

theorem placards_per_person :
  total_placards / people_entered = 2 := by
  sorry

end placards_per_person_l2265_226582


namespace gcd_18_30_l2265_226535

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l2265_226535


namespace only_13_remains_prime_l2265_226536

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

def remains_prime_when_reversed (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem only_13_remains_prime : 
  (remains_prime_when_reversed 13) ∧ 
  (¬remains_prime_when_reversed 29) ∧ 
  (¬remains_prime_when_reversed 53) ∧ 
  (¬remains_prime_when_reversed 23) ∧ 
  (¬remains_prime_when_reversed 41) :=
sorry

end only_13_remains_prime_l2265_226536


namespace bianca_coloring_books_l2265_226541

/-- Represents the number of coloring books Bianca gave away -/
def books_given_away : ℕ := 6

/-- Represents Bianca's initial number of coloring books -/
def initial_books : ℕ := 45

/-- Represents the number of coloring books Bianca bought -/
def books_bought : ℕ := 20

/-- Represents Bianca's final number of coloring books -/
def final_books : ℕ := 59

theorem bianca_coloring_books : 
  initial_books - books_given_away + books_bought = final_books :=
by sorry

end bianca_coloring_books_l2265_226541


namespace twelveSidedFigureArea_l2265_226539

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices --/
structure Polygon where
  vertices : List Point

/-- The area of a polygon --/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- Our specific 12-sided figure --/
def twelveSidedFigure : Polygon := {
  vertices := [
    { x := 2, y := 1 }, { x := 3, y := 2 }, { x := 3, y := 3 }, { x := 5, y := 3 },
    { x := 6, y := 4 }, { x := 5, y := 5 }, { x := 4, y := 5 }, { x := 3, y := 6 },
    { x := 2, y := 5 }, { x := 2, y := 4 }, { x := 1, y := 3 }, { x := 2, y := 2 }
  ]
}

theorem twelveSidedFigureArea : area twelveSidedFigure = 12 := by sorry

end twelveSidedFigureArea_l2265_226539


namespace riku_sticker_count_l2265_226533

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := 85

/-- The ratio of Riku's stickers to Kristoff's stickers -/
def riku_to_kristoff_ratio : ℕ := 25

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := kristoff_stickers * riku_to_kristoff_ratio

theorem riku_sticker_count : riku_stickers = 2125 := by
  sorry

end riku_sticker_count_l2265_226533


namespace triangle_midpoint_line_sum_l2265_226521

/-- Given a triangle ABC with vertices A(0,6), B(0,0), C(10,0), and D the midpoint of AB,
    the sum of the slope and y-intercept of line CD is 27/10 -/
theorem triangle_midpoint_line_sum (A B C D : ℝ × ℝ) : 
  A = (0, 6) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2
  m + b = 27 / 10 := by sorry

end triangle_midpoint_line_sum_l2265_226521


namespace towels_used_is_285_l2265_226548

/-- Calculates the total number of towels used in a gym over 4 hours -/
def totalTowelsUsed (firstHourGuests : ℕ) : ℕ :=
  let secondHourGuests := firstHourGuests + (firstHourGuests * 20 / 100)
  let thirdHourGuests := secondHourGuests + (secondHourGuests * 25 / 100)
  let fourthHourGuests := thirdHourGuests + (thirdHourGuests / 3)
  firstHourGuests + secondHourGuests + thirdHourGuests + fourthHourGuests

/-- Theorem stating that the total number of towels used is 285 -/
theorem towels_used_is_285 :
  totalTowelsUsed 50 = 285 := by
  sorry

#eval totalTowelsUsed 50

end towels_used_is_285_l2265_226548


namespace cubic_equation_solution_l2265_226544

theorem cubic_equation_solution (x y z : ℕ) : 
  x^3 + 4*y^3 = 16*z^3 + 4*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end cubic_equation_solution_l2265_226544


namespace intersection_points_on_circle_l2265_226565

/-- The parabolas y = (x - 1)^2 and x - 3 = (y + 2)^2 intersect at four points that lie on a circle with radius squared equal to 1/2 -/
theorem intersection_points_on_circle : ∃ (c : ℝ × ℝ) (r : ℝ),
  (∀ (p : ℝ × ℝ), (p.2 = (p.1 - 1)^2 ∧ p.1 - 3 = (p.2 + 2)^2) →
    ((p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2)) ∧
  r^2 = (1 : ℝ) / 2 :=
sorry

end intersection_points_on_circle_l2265_226565


namespace water_fraction_after_four_replacements_l2265_226563

/-- The fraction of water remaining in a radiator after multiple replacements with antifreeze -/
def water_fraction (total_capacity : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  (1 - replacement_volume / total_capacity) ^ num_replacements

/-- The fraction of water remaining in a 20-quart radiator after 4 replacements of 5 quarts each -/
theorem water_fraction_after_four_replacements :
  water_fraction 20 5 4 = 81 / 256 := by
  sorry

#eval water_fraction 20 5 4

end water_fraction_after_four_replacements_l2265_226563


namespace line_properties_l2265_226512

/-- Represents a line in the form ax + 3y + 1 = 0 -/
structure Line where
  a : ℝ

/-- Checks if the intercepts of the line on the coordinate axes are equal -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.a = 3

/-- Checks if the line l is parallel to the line x + (a-2)y + a = 0 -/
def is_parallel_to_given_line (l : Line) : Prop :=
  l.a * (l.a - 2) - 3 = 0 ∧ l.a^2 - 1 ≠ 0

theorem line_properties (l : Line) :
  (has_equal_intercepts l ↔ l.a = 3) ∧
  (is_parallel_to_given_line l ↔ l.a = 3) := by sorry

end line_properties_l2265_226512


namespace wyatts_money_l2265_226543

theorem wyatts_money (bread_quantity : ℕ) (juice_quantity : ℕ) 
  (bread_price : ℕ) (juice_price : ℕ) (money_left : ℕ) :
  bread_quantity = 5 →
  juice_quantity = 4 →
  bread_price = 5 →
  juice_price = 2 →
  money_left = 41 →
  bread_quantity * bread_price + juice_quantity * juice_price + money_left = 74 :=
by sorry

end wyatts_money_l2265_226543


namespace greg_sisters_count_l2265_226593

def number_of_sisters (total_bars : ℕ) (days_in_week : ℕ) (traded_bars : ℕ) (bars_per_sister : ℕ) : ℕ :=
  (total_bars - days_in_week - traded_bars) / bars_per_sister

theorem greg_sisters_count :
  let total_bars : ℕ := 20
  let days_in_week : ℕ := 7
  let traded_bars : ℕ := 3
  let bars_per_sister : ℕ := 5
  number_of_sisters total_bars days_in_week traded_bars bars_per_sister = 2 := by
  sorry

end greg_sisters_count_l2265_226593


namespace multiply_by_9999_l2265_226506

theorem multiply_by_9999 : ∃ x : ℕ, x * 9999 = 4690640889 ∧ x = 469131 := by sorry

end multiply_by_9999_l2265_226506


namespace tan_435_degrees_l2265_226567

theorem tan_435_degrees : Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_435_degrees_l2265_226567


namespace cube_sum_magnitude_l2265_226587

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by sorry

end cube_sum_magnitude_l2265_226587


namespace terez_cows_l2265_226570

theorem terez_cows (total : ℕ) (females : ℕ) (pregnant : ℕ) : 
  2 * females = total → 
  2 * pregnant = females → 
  pregnant = 11 → 
  total = 44 := by
sorry

end terez_cows_l2265_226570


namespace batsman_average_l2265_226579

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  inningScore : ℝ
  averageIncrease : ℝ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem: Given the conditions, the batsman's new average is 55 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.inningScore = 95)
  (h2 : b.averageIncrease = 2.5)
  : newAverage b = 55 := by
  sorry

#eval newAverage { initialAverage := 52.5, inningScore := 95, averageIncrease := 2.5 }

end batsman_average_l2265_226579


namespace imaginary_part_of_z_l2265_226527

theorem imaginary_part_of_z (z : ℂ) (h : (z - 2*Complex.I)*Complex.I = 1 + Complex.I) : 
  Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l2265_226527


namespace clock_angle_at_2pm_l2265_226554

/-- The number of hours on a standard clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a complete rotation -/
def full_rotation : ℕ := 360

/-- The number of degrees the hour hand moves per hour -/
def hour_hand_degrees_per_hour : ℚ := full_rotation / clock_hours

/-- The position of the hour hand at 2:00 -/
def hour_hand_position_at_2 : ℚ := 2 * hour_hand_degrees_per_hour

/-- The position of the minute hand at 2:00 -/
def minute_hand_position_at_2 : ℚ := 0

/-- The smaller angle between the hour hand and minute hand at 2:00 -/
def smaller_angle_at_2 : ℚ := hour_hand_position_at_2 - minute_hand_position_at_2

theorem clock_angle_at_2pm :
  smaller_angle_at_2 = 60 := by sorry

end clock_angle_at_2pm_l2265_226554


namespace min_value_of_expression_existence_of_minimum_l2265_226518

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  (a / (4 - a)) + (1 / (a - 1)) ≥ 2 :=
sorry

theorem existence_of_minimum (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  ∃ a, (a / (4 - a)) + (1 / (a - 1)) = 2 :=
sorry

end min_value_of_expression_existence_of_minimum_l2265_226518


namespace smallest_sum_of_ten_numbers_l2265_226513

theorem smallest_sum_of_ten_numbers (S : Finset ℕ) : 
  S.card = 10 ∧ 
  (∀ T ⊆ S, T.card = 5 → Even (T.prod id)) ∧
  Odd (S.sum id) →
  65 ≤ S.sum id :=
sorry

end smallest_sum_of_ten_numbers_l2265_226513
