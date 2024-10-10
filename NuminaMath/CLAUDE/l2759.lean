import Mathlib

namespace solve_for_y_l2759_275916

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 2*x = y - 4) (h2 : x = -6) : y = 28 := by
  sorry

end solve_for_y_l2759_275916


namespace evaluate_expression_l2759_275921

theorem evaluate_expression (y : ℝ) (h : y = -3) : 
  (5 + y * (5 + y) - 5^2) / (y - 5 + y^2) = -26 := by
  sorry

end evaluate_expression_l2759_275921


namespace largest_of_five_consecutive_odds_l2759_275991

theorem largest_of_five_consecutive_odds (n : ℤ) : 
  (n % 2 = 1) → 
  (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) → 
  (n + 8 = 19) := by
  sorry

end largest_of_five_consecutive_odds_l2759_275991


namespace min_120_degree_turns_l2759_275959

/-- A triangular graph representing a city --/
structure TriangularCity where
  /-- The number of triangular blocks in the city --/
  blocks : Nat
  /-- The number of intersections (squares) in the city --/
  intersections : Nat
  /-- The path taken by the tourist --/
  tourist_path : List Nat
  /-- Ensures the number of blocks is 16 --/
  blocks_count : blocks = 16
  /-- Ensures the number of intersections is 15 --/
  intersections_count : intersections = 15
  /-- Ensures the tourist visits each intersection exactly once --/
  path_visits_all_once : tourist_path.length = intersections ∧ tourist_path.Nodup

/-- The number of 120° turns in a given path --/
def count_120_degree_turns (path : List Nat) : Nat :=
  sorry

/-- Theorem stating that a tourist in a triangular city must make at least 4 turns of 120° --/
theorem min_120_degree_turns (city : TriangularCity) :
  count_120_degree_turns city.tourist_path ≥ 4 := by
  sorry

end min_120_degree_turns_l2759_275959


namespace trig_simplification_l2759_275928

theorem trig_simplification (α : ℝ) :
  Real.cos (π / 3 + α) + Real.sin (π / 6 + α) = Real.cos α := by sorry

end trig_simplification_l2759_275928


namespace equality_condition_l2759_275912

theorem equality_condition (a b c d : ℝ) :
  a + b * c * d = (a + b) * (a + c) * (a + d) ↔ a^2 + a * (b + c + d) + b * c + b * d + c * d = 1 :=
sorry

end equality_condition_l2759_275912


namespace f_symmetry_l2759_275966

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + Real.pi^2 * x^2) - Real.pi * x) + Real.pi

theorem f_symmetry (m : ℝ) : f m = 3 → f (-m) = 2 * Real.pi - 3 := by
  sorry

end f_symmetry_l2759_275966


namespace power_two_mod_four_l2759_275917

theorem power_two_mod_four : 2^300 % 4 = 0 := by
  sorry

end power_two_mod_four_l2759_275917


namespace discounted_good_price_l2759_275954

/-- The price of a good after applying successive discounts -/
def discounted_price (initial_price : ℝ) : ℝ :=
  initial_price * 0.75 * 0.85 * 0.90 * 0.93

theorem discounted_good_price (P : ℝ) :
  discounted_price P = 6600 → P = 11118.75 := by
  sorry

end discounted_good_price_l2759_275954


namespace unique_solution_l2759_275942

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := log (2 * x + 1) + log x = 1

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x > 0 ∧ 2 * x + 1 > 0 ∧ equation x ∧ x = 2 :=
by sorry

end unique_solution_l2759_275942


namespace exponent_division_l2759_275950

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by
  sorry

end exponent_division_l2759_275950


namespace line_L_equation_l2759_275956

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0
def l₂ (x y : ℝ) : Prop := 5 * x + 2 * y + 1 = 0
def l₃ (x y : ℝ) : Prop := 3 * x - 5 * y + 6 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point : ℝ × ℝ := (-1, 2)

-- Define line L
def L (x y : ℝ) : Prop := 5 * x + 3 * y - 1 = 0

-- Theorem statement
theorem line_L_equation :
  (∀ x y : ℝ, L x y ↔ 
    (x = intersection_point.1 ∧ y = intersection_point.2 ∨
    ∃ t : ℝ, x = intersection_point.1 + t ∧ y = intersection_point.2 - (5/3) * t)) ∧
  (∀ x y : ℝ, L x y → l₃ x y → 
    (x - intersection_point.1) * 3 + (y - intersection_point.2) * (-5) = 0) :=
by sorry


end line_L_equation_l2759_275956


namespace probability_two_first_grade_pens_l2759_275996

/-- The probability of selecting 2 first-grade pens from a box of 6 pens, where 3 are first-grade -/
theorem probability_two_first_grade_pens (total_pens : ℕ) (first_grade_pens : ℕ) 
  (h1 : total_pens = 6) (h2 : first_grade_pens = 3) : 
  (Nat.choose first_grade_pens 2 : ℚ) / (Nat.choose total_pens 2) = 1/5 := by
  sorry

end probability_two_first_grade_pens_l2759_275996


namespace inequality_solution_implies_a_less_than_three_l2759_275952

theorem inequality_solution_implies_a_less_than_three (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 := by
  sorry

end inequality_solution_implies_a_less_than_three_l2759_275952


namespace decimal_to_binary_27_l2759_275986

theorem decimal_to_binary_27 : 
  (27 : ℕ).digits 2 = [1, 1, 0, 1, 1] := by sorry

end decimal_to_binary_27_l2759_275986


namespace triangle_inequality_ratio_123_l2759_275973

theorem triangle_inequality_ratio_123 :
  ∀ (x : ℝ), x > 0 → ¬(x + 2*x > 3*x) :=
by
  sorry

end triangle_inequality_ratio_123_l2759_275973


namespace car_dealership_problem_l2759_275919

theorem car_dealership_problem (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_cars : ℕ) (total_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 15 / 100)
  (h3 : new_cars = 80)
  (h4 : total_silver_percent = 25 / 100) :
  (new_cars - (total_silver_percent * (initial_cars + new_cars) - initial_silver_percent * initial_cars)) / new_cars = 70 / 100 := by
  sorry

end car_dealership_problem_l2759_275919


namespace dispersion_measures_l2759_275922

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define the concept of a statistic as a function from a sample to a real number
def Statistic := Sample → Real

-- Define the concept of measuring dispersion
def MeasuresDispersion (s : Statistic) : Prop := sorry

-- Define standard deviation
def StandardDeviation : Statistic := sorry

-- Define median
def Median : Statistic := sorry

-- Define range
def Range : Statistic := sorry

-- Define mean
def Mean : Statistic := sorry

-- Theorem stating that only standard deviation and range measure dispersion
theorem dispersion_measures (sample : Sample) :
  MeasuresDispersion StandardDeviation ∧
  MeasuresDispersion Range ∧
  ¬MeasuresDispersion Median ∧
  ¬MeasuresDispersion Mean :=
sorry

end dispersion_measures_l2759_275922


namespace negative_three_star_five_l2759_275932

-- Define the operation *
def star (a b : ℚ) : ℚ := (a - 2*b) / (2*a - b)

-- Theorem statement
theorem negative_three_star_five :
  star (-3) 5 = 13/11 := by sorry

end negative_three_star_five_l2759_275932


namespace equal_size_meetings_l2759_275939

/-- Given n sets representing daily meetings, prove that all sets have the same size. -/
theorem equal_size_meetings (n : ℕ) (A : Fin n → Finset (Fin n)) 
  (h_n : n ≥ 3)
  (h_size : ∀ i, (A i).card ≥ 3)
  (h_cover : ∀ i j, i < j → ∃! k, i ∈ A k ∧ j ∈ A k) :
  ∃ k, ∀ i, (A i).card = k :=
sorry

end equal_size_meetings_l2759_275939


namespace percentage_of_x_l2759_275961

theorem percentage_of_x (x y : ℝ) (h1 : x / y = 4) (h2 : y ≠ 0) : (2 * x - y) / x = 175 / 100 := by
  sorry

end percentage_of_x_l2759_275961


namespace two_left_movements_l2759_275970

-- Define the direction type
inductive Direction
| Left
| Right

-- Define a function to convert direction to sign
def directionToSign (d : Direction) : Int :=
  match d with
  | Direction.Left => -1
  | Direction.Right => 1

-- Define a single movement
def singleMovement (distance : ℝ) (direction : Direction) : ℝ :=
  (directionToSign direction : ℝ) * distance

-- Define the problem statement
theorem two_left_movements (distance : ℝ) :
  distance = 3 →
  (singleMovement distance Direction.Left + singleMovement distance Direction.Left) = -6 :=
by sorry

end two_left_movements_l2759_275970


namespace expression_simplification_l2759_275918

theorem expression_simplification (x : ℝ) (h : x^2 + 2*x - 6 = 0) :
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3*x) / (x^2 - 6*x + 9)) = -1/2 := by
  sorry

end expression_simplification_l2759_275918


namespace circle_and_line_theorem_l2759_275923

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define the line y = -x + 2
def line_intercept (x y : ℝ) : Prop := y = -x + 2

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 3/2

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem circle_and_line_theorem :
  -- Circle C passes through (1, √3)
  circle_C 1 (Real.sqrt 3) →
  -- The center of C is on the line y = x
  ∃ a : ℝ, line_y_eq_x a a ∧ ∀ x y : ℝ, circle_C x y ↔ (x - a)^2 + (y - a)^2 = 4 →
  -- The chord intercepted by y = -x + 2 has length 2√2
  ∃ x1 y1 x2 y2 : ℝ, 
    line_intercept x1 y1 ∧ line_intercept x2 y2 ∧ 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = 8 →
  -- Line l passes through (3/2, 0) and intersects C at P and Q
  ∃ m xP yP xQ yQ : ℝ,
    line_l m (3/2) 0 ∧ 
    circle_C xP yP ∧ circle_C xQ yQ ∧
    line_l m xP yP ∧ line_l m xQ yQ ∧
    -- OP · OQ = -2
    dot_product xP yP xQ yQ = -2 →
  -- Conclusion 1: Equation of circle C
  (∀ x y : ℝ, circle_C x y ↔ x^2 + y^2 = 4) ∧
  -- Conclusion 2: Equation of line l
  (m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2) ∧
  (∀ x y : ℝ, line_l m x y ↔ 2*x + m*y - 3 = 0) := by
sorry

end circle_and_line_theorem_l2759_275923


namespace min_bodyguards_tournament_l2759_275983

/-- A tournament where each bodyguard is defeated by at least three others -/
def BodyguardTournament (n : ℕ) := 
  ∃ (defeats : Fin n → Fin n → Prop),
    (∀ i j k : Fin n, i ≠ j → ∃ l : Fin n, defeats l i ∧ defeats l j) ∧
    (∀ i : Fin n, ∃ j k l : Fin n, j ≠ i ∧ k ≠ i ∧ l ≠ i ∧ defeats j i ∧ defeats k i ∧ defeats l i)

/-- The minimum number of bodyguards in a tournament satisfying the conditions is 7 -/
theorem min_bodyguards_tournament : 
  (∃ n : ℕ, BodyguardTournament n) ∧ 
  (∀ m : ℕ, m < 7 → ¬BodyguardTournament m) ∧
  BodyguardTournament 7 :=
sorry

end min_bodyguards_tournament_l2759_275983


namespace bridge_length_problem_l2759_275947

/-- The length of a bridge crossed by a man walking at a given speed in a given time -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A man walking at 10 km/hr crosses a bridge in 18 minutes. The bridge length is 3 km. -/
theorem bridge_length_problem :
  let walking_speed : ℝ := 10  -- km/hr
  let crossing_time : ℝ := 18 / 60  -- 18 minutes converted to hours
  bridge_length walking_speed crossing_time = 3 := by
sorry


end bridge_length_problem_l2759_275947


namespace min_value_problem_l2759_275926

theorem min_value_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (heq : 2*m + n = 1) :
  (1/m) + (2/n) ≥ 8 ∧ ((1/m) + (2/n) = 8 ↔ n = 2*m ∧ n = 1/2) :=
sorry

end min_value_problem_l2759_275926


namespace price_reduction_percentage_l2759_275938

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : ∃ x : ℝ, initial_price * (1 - x)^2 = final_price ∧ 0 < x ∧ x < 1) :
  ∃ x : ℝ, initial_price * (1 - x)^2 = final_price ∧ x = 0.2 :=
sorry

end price_reduction_percentage_l2759_275938


namespace magpie_call_not_correlation_l2759_275901

-- Define a type for statements
inductive Statement
| HeavySnow : Statement
| GreatTeachers : Statement
| Smoking : Statement
| MagpieCall : Statement

-- Define a predicate for correlation
def IsCorrelation (s : Statement) : Prop :=
  match s with
  | Statement.HeavySnow => True
  | Statement.GreatTeachers => True
  | Statement.Smoking => True
  | Statement.MagpieCall => False

-- Theorem statement
theorem magpie_call_not_correlation :
  ∀ s : Statement, 
    (s = Statement.HeavySnow ∨ s = Statement.GreatTeachers ∨ s = Statement.Smoking → IsCorrelation s) ∧
    (s = Statement.MagpieCall → ¬IsCorrelation s) :=
by sorry

end magpie_call_not_correlation_l2759_275901


namespace card_statements_l2759_275958

/-- Represents the number of true statements on the card -/
def TrueStatements : Nat → Prop
  | 0 => True
  | 1 => False
  | 2 => False
  | 3 => False
  | 4 => False
  | 5 => False
  | _ => False

/-- The five statements on the card -/
def Statement : Nat → Prop
  | 1 => TrueStatements 1
  | 2 => TrueStatements 2
  | 3 => TrueStatements 3
  | 4 => TrueStatements 4
  | 5 => TrueStatements 5
  | _ => False

/-- Theorem stating that the number of true statements is 0 -/
theorem card_statements :
  (∀ n : Nat, Statement n ↔ TrueStatements n) →
  TrueStatements 0 := by
  sorry

end card_statements_l2759_275958


namespace triangle_midpoint_sum_l2759_275993

theorem triangle_midpoint_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end triangle_midpoint_sum_l2759_275993


namespace arithmetic_calculation_l2759_275924

theorem arithmetic_calculation : 
  let a := 65 * ((13/3 + 7/2) / (11/5 - 5/3))
  ∃ (n : ℕ) (m : ℚ), 0 ≤ m ∧ m < 1 ∧ a = n + m ∧ n = 954 ∧ m = 33/48 := by
  sorry

end arithmetic_calculation_l2759_275924


namespace license_plate_ratio_l2759_275957

/-- The number of possible letters in a license plate -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate -/
def old_letters : ℕ := 2

/-- The number of digits in an old license plate -/
def old_digits : ℕ := 3

/-- The number of letters in a new license plate -/
def new_letters : ℕ := 3

/-- The number of digits in a new license plate -/
def new_digits : ℕ := 4

/-- The ratio of new license plates to old license plates -/
theorem license_plate_ratio :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) = 260 := by
  sorry

end license_plate_ratio_l2759_275957


namespace twelfth_term_of_sequence_l2759_275994

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 12 = 25/6 := by
  sorry

end twelfth_term_of_sequence_l2759_275994


namespace second_point_x_coordinate_l2759_275931

/-- Given two points (m, n) and (m + 2, n + 1) on the line x = 2y + 3,
    the x-coordinate of the second point is m + 2. -/
theorem second_point_x_coordinate (m n : ℝ) : 
  (m = 2 * n + 3) → -- First point (m, n) lies on the line
  (m + 2 = 2 * (n + 1) + 3) → -- Second point (m + 2, n + 1) lies on the line
  (m + 2 = m + 2) -- The x-coordinate of the second point is m + 2
:= by sorry

end second_point_x_coordinate_l2759_275931


namespace line_plane_parallelism_condition_l2759_275971

-- Define the concepts of line and plane
variable (m : Line) (α : Plane)

-- Define what it means for a line to be parallel to a plane
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define what it means for a line to be parallel to countless lines in a plane
def line_parallel_to_countless_lines_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_parallelism_condition :
  (line_parallel_to_countless_lines_in_plane m α → line_parallel_to_plane m α) ∧
  ¬(line_parallel_to_plane m α → line_parallel_to_countless_lines_in_plane m α) := by sorry

end line_plane_parallelism_condition_l2759_275971


namespace simple_interest_problem_l2759_275977

/-- 
Given a sum P put at simple interest for 7 years, if increasing the interest rate 
by 2% results in $140 more interest, then P = $1000.
-/
theorem simple_interest_problem (P : ℚ) (R : ℚ) : 
  (P * (R + 2) * 7 / 100 = P * R * 7 / 100 + 140) → P = 1000 := by
  sorry

end simple_interest_problem_l2759_275977


namespace profit_difference_maddox_profit_exceeds_theo_by_15_l2759_275964

/-- Calculates the profit difference between two sellers of Polaroid cameras. -/
theorem profit_difference (num_cameras : ℕ) (cost_per_camera : ℕ) 
  (maddox_selling_price : ℕ) (theo_selling_price : ℕ) : ℕ :=
  let maddox_profit := num_cameras * maddox_selling_price - num_cameras * cost_per_camera
  let theo_profit := num_cameras * theo_selling_price - num_cameras * cost_per_camera
  maddox_profit - theo_profit

/-- Proves that Maddox made $15 more profit than Theo. -/
theorem maddox_profit_exceeds_theo_by_15 : 
  profit_difference 3 20 28 23 = 15 := by
  sorry

end profit_difference_maddox_profit_exceeds_theo_by_15_l2759_275964


namespace variables_positively_correlated_l2759_275911

/-- Represents a simple linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Defines positive correlation between variables in a linear regression model -/
def positively_correlated (model : LinearRegression) : Prop :=
  model.slope > 0

/-- The specific linear regression model given in the problem -/
def given_model : LinearRegression :=
  { slope := 0.5, intercept := 2 }

/-- Theorem stating that the variables in the given model are positively correlated -/
theorem variables_positively_correlated : 
  positively_correlated given_model := by sorry

end variables_positively_correlated_l2759_275911


namespace fraction_equality_l2759_275995

theorem fraction_equality (a b : ℝ) : - (a / (b - a)) = a / (a - b) := by sorry

end fraction_equality_l2759_275995


namespace semicircle_radius_theorem_l2759_275925

/-- Theorem: Given a rectangle with length 48 cm and width 24 cm, and a semicircle
    attached to one side of the rectangle (with the diameter equal to the length
    of the rectangle), if the perimeter of the combined shape is 144 cm, then the
    radius of the semicircle is 48 / (π + 2) cm. -/
theorem semicircle_radius_theorem (rectangle_length : ℝ) (rectangle_width : ℝ) 
    (combined_perimeter : ℝ) (semicircle_radius : ℝ) :
  rectangle_length = 48 →
  rectangle_width = 24 →
  combined_perimeter = 144 →
  combined_perimeter = 2 * rectangle_width + rectangle_length + π * semicircle_radius →
  semicircle_radius = 48 / (π + 2) :=
by sorry

end semicircle_radius_theorem_l2759_275925


namespace circle_op_difference_l2759_275936

/-- The custom operation ⊙ for three natural numbers -/
def circle_op (a b c : ℕ) : ℕ :=
  (a * b) * 100 + (b * c)

/-- Theorem stating the result of the calculation -/
theorem circle_op_difference : circle_op 5 7 4 - circle_op 7 4 5 = 708 := by
  sorry

end circle_op_difference_l2759_275936


namespace union_equal_M_l2759_275914

def M : Set Char := {'a', 'b', 'c', 'd', 'e'}
def N : Set Char := {'b', 'd', 'e'}

theorem union_equal_M : M ∪ N = M := by sorry

end union_equal_M_l2759_275914


namespace boat_speed_in_still_water_l2759_275920

/-- The speed of a boat in still water, given downstream and upstream speeds and current speed. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : current_speed = 17)
  (h2 : downstream_speed = 77)
  (h3 : upstream_speed = 43) :
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 60 ∧ 
    still_water_speed + current_speed = downstream_speed ∧ 
    still_water_speed - current_speed = upstream_speed :=
by sorry

end boat_speed_in_still_water_l2759_275920


namespace fraction_to_decimal_l2759_275981

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l2759_275981


namespace average_of_first_12_even_numbers_l2759_275903

def first_12_even_numbers : List ℤ :=
  [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

theorem average_of_first_12_even_numbers :
  (List.sum first_12_even_numbers) / (List.length first_12_even_numbers) = -1 := by
sorry

end average_of_first_12_even_numbers_l2759_275903


namespace total_sum_calculation_l2759_275915

theorem total_sum_calculation (maggie_share : ℚ) (total_sum : ℚ) : 
  maggie_share = 7500 → 
  maggie_share = (1/8 : ℚ) * total_sum → 
  total_sum = 60000 := by sorry

end total_sum_calculation_l2759_275915


namespace solve_equation_l2759_275930

theorem solve_equation (y : ℝ) (h : (2 * y) / 3 = 12) : y = 18 := by
  sorry

end solve_equation_l2759_275930


namespace ideal_function_iff_l2759_275945

def IdealFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂)

theorem ideal_function_iff (f : ℝ → ℝ) :
  IdealFunction f ↔
  ((∀ x, f x + f (-x) = 0) ∧
   (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)) :=
by sorry

end ideal_function_iff_l2759_275945


namespace train_speed_l2759_275940

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 800 →
  crossing_time = 47.99616030717543 →
  man_speed_kmh = 5 →
  ∃ (train_speed : ℝ), abs (train_speed - 64.9848) < 0.0001 :=
by sorry

end train_speed_l2759_275940


namespace width_of_sum_l2759_275955

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields here
  
/-- The width of a convex curve in a given direction -/
def width (K : ConvexCurve) (direction : ℝ × ℝ) : ℝ :=
  sorry

/-- The sum of two convex curves -/
def curve_sum (K₁ K₂ : ConvexCurve) : ConvexCurve :=
  sorry

/-- Theorem: The width of the sum of two convex curves is the sum of their individual widths -/
theorem width_of_sum (K₁ K₂ : ConvexCurve) (direction : ℝ × ℝ) :
  width (curve_sum K₁ K₂) direction = width K₁ direction + width K₂ direction :=
sorry

end width_of_sum_l2759_275955


namespace fourth_grade_students_l2759_275967

theorem fourth_grade_students (initial_students leaving_students new_students : ℕ) :
  initial_students = 35 →
  leaving_students = 10 →
  new_students = 10 →
  initial_students - leaving_students + new_students = 35 := by
sorry

end fourth_grade_students_l2759_275967


namespace trig_problem_l2759_275906

theorem trig_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 4 / 5) : 
  (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 20 ∧ 
  Real.tan (α - 5 * Real.pi / 4) = 1 / 7 := by sorry

end trig_problem_l2759_275906


namespace triangle_angle_proof_l2759_275927

theorem triangle_angle_proof 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (a - b + c) = a * c)
  (h2 : Real.sin A * Real.sin C = (Real.sqrt 3 - 1) / 4) : 
  B = 2 * π / 3 ∧ (C = π / 12 ∨ C = π / 4) := by
  sorry


end triangle_angle_proof_l2759_275927


namespace sim_tetrahedron_volume_l2759_275909

/-- A tetrahedron with similar but not all equal triangular faces -/
structure SimTetrahedron where
  /-- The faces are similar triangles -/
  similar_faces : Bool
  /-- Not all faces are equal -/
  not_all_equal : Bool
  /-- Any two faces share at least one pair of equal edges, not counting the common edge -/
  shared_equal_edges : Bool
  /-- Two edges in one face have lengths 3 and 5 -/
  edge_lengths : (ℝ × ℝ)

/-- The volume of a SimTetrahedron is either (55 * √6) / 18 or (11 * √10) / 10 -/
theorem sim_tetrahedron_volume (t : SimTetrahedron) : 
  t.similar_faces ∧ t.not_all_equal ∧ t.shared_equal_edges ∧ t.edge_lengths = (3, 5) →
  (∃ v : ℝ, v = (55 * Real.sqrt 6) / 18 ∨ v = (11 * Real.sqrt 10) / 10) :=
by sorry

end sim_tetrahedron_volume_l2759_275909


namespace library_books_total_l2759_275988

theorem library_books_total (initial_books additional_books : ℕ) : 
  initial_books = 54 → additional_books = 23 → initial_books + additional_books = 77 := by
  sorry

end library_books_total_l2759_275988


namespace additional_beads_needed_bella_needs_twelve_more_beads_l2759_275972

/-- Given the number of friends, beads per bracelet, and beads Bella has,
    calculate the number of additional beads needed. -/
theorem additional_beads_needed 
  (num_friends : ℕ) 
  (beads_per_bracelet : ℕ) 
  (beads_bella_has : ℕ) : ℕ :=
  (num_friends * beads_per_bracelet) - beads_bella_has

/-- Prove that Bella needs 12 more beads to make bracelets for her friends. -/
theorem bella_needs_twelve_more_beads : 
  additional_beads_needed 6 8 36 = 12 := by
  sorry

end additional_beads_needed_bella_needs_twelve_more_beads_l2759_275972


namespace evaluate_expression_l2759_275941

theorem evaluate_expression (a : ℚ) (h : a = 4/3) : (6*a^2 - 11*a + 2)*(3*a - 4) = 0 := by
  sorry

end evaluate_expression_l2759_275941


namespace injective_function_property_l2759_275974

theorem injective_function_property {A : Type*} (f : A → A) (h : Function.Injective f) :
  ∀ (x₁ x₂ : A), x₁ ≠ x₂ → f x₁ ≠ f x₂ := by
  sorry

end injective_function_property_l2759_275974


namespace rectangle_max_area_l2759_275953

/-- Given a rectangle with sides a and b and perimeter p, 
    the area is maximized when the rectangle is a square. -/
theorem rectangle_max_area (a b p : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_perimeter : p = 2 * (a + b)) :
  ∃ (max_area : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 2 * (x + y) = p → x * y ≤ max_area ∧ 
  (x * y = max_area ↔ x = y) :=
sorry

end rectangle_max_area_l2759_275953


namespace triangle_side_length_l2759_275978

noncomputable section

/-- Given a triangle ABC with angles A, B, C and opposite sides a, b, c respectively,
    if A = 30°, B = 45°, and a = √2, then b = 2 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → B = π/4 → a = Real.sqrt 2 → 
  (a / Real.sin A = b / Real.sin B) → 
  b = 2 := by sorry

end triangle_side_length_l2759_275978


namespace candle_weight_theorem_l2759_275913

/-- The weight of beeswax used in each candle, in ounces. -/
def beeswax_weight : ℕ := 8

/-- The weight of coconut oil used in each candle, in ounces. -/
def coconut_oil_weight : ℕ := 1

/-- The number of candles Ethan makes. -/
def num_candles : ℕ := 10 - 3

/-- The total weight of one candle, in ounces. -/
def candle_weight : ℕ := beeswax_weight + coconut_oil_weight

/-- The combined weight of all candles, in ounces. -/
def total_weight : ℕ := num_candles * candle_weight

theorem candle_weight_theorem : total_weight = 63 := by
  sorry

end candle_weight_theorem_l2759_275913


namespace parabola_directrix_l2759_275998

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -25 / 12

/-- Theorem stating that the given directrix equation is correct for the parabola -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → ∃ (d : ℝ), directrix_equation d ∧ 
  (d = y - 1 / (4 * 3) - (y - 1 / (4 * 3) - (-2 - 1 / (4 * 3)))) :=
sorry

end parabola_directrix_l2759_275998


namespace sum_properties_l2759_275992

theorem sum_properties (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 
  (∃ x y : ℤ, a + b = 2*x + 1 ∧ a + b = 2*y) ∧ 
  (∃ z : ℤ, a + b = 6*z + 3) ∧ 
  (∃ w : ℤ, a + b = 9*w + 3) ∧ 
  (∃ v : ℤ, a + b = 9*v) :=
by sorry

end sum_properties_l2759_275992


namespace square_sum_equals_48_l2759_275963

theorem square_sum_equals_48 (x y : ℝ) (h1 : x - 2*y = 4) (h2 : x*y = 8) : x^2 + 4*y^2 = 48 := by
  sorry

end square_sum_equals_48_l2759_275963


namespace portias_school_students_l2759_275951

theorem portias_school_students (portia_students lara_students : ℕ) : 
  portia_students = 2 * lara_students →
  portia_students + lara_students = 3000 →
  portia_students = 2000 := by
  sorry

end portias_school_students_l2759_275951


namespace decimal_equals_base5_l2759_275943

-- Define a function to convert a list of digits in base 5 to decimal
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 5 * acc) 0

-- Define the decimal number
def decimalNumber : Nat := 111

-- Define the base-5 representation as a list of digits
def base5Representation : List Nat := [4, 2, 1]

-- Theorem stating that the decimal number is equal to its base-5 representation
theorem decimal_equals_base5 : decimalNumber = base5ToDecimal base5Representation := by
  sorry

end decimal_equals_base5_l2759_275943


namespace women_in_salem_l2759_275979

def leesburg_population : ℕ := 58940
def salem_population_multiplier : ℕ := 15
def people_moved_out : ℕ := 130000

def salem_original_population : ℕ := leesburg_population * salem_population_multiplier
def salem_current_population : ℕ := salem_original_population - people_moved_out

theorem women_in_salem : 
  (salem_current_population / 2 : ℕ) = 377050 := by sorry

end women_in_salem_l2759_275979


namespace min_value_of_f_fourth_composition_l2759_275944

/-- The function f(x) = x^2 + 6x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 7

/-- The statement that the minimum value of f(f(f(f(x)))) over all real x is 23 -/
theorem min_value_of_f_fourth_composition :
  ∀ x : ℝ, f (f (f (f x))) ≥ 23 ∧ ∃ y : ℝ, f (f (f (f y))) = 23 :=
sorry

end min_value_of_f_fourth_composition_l2759_275944


namespace small_circle_radius_l2759_275975

/-- Given a large circle with radius 6 meters containing five congruent smaller circles
    arranged such that the diameter of the large circle equals the sum of the diameters
    of three smaller circles, the radius of each smaller circle is 2 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 6 → 2 * R = 3 * (2 * r) → r = 2 :=
by sorry

end small_circle_radius_l2759_275975


namespace hyperbola_eccentricity_l2759_275905

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the angle between its two asymptotes is 60°, then its eccentricity e
    is either 2 or 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let asymptote_angle := Real.pi / 3
  let eccentricity := Real.sqrt (1 + b^2 / a^2)
  asymptote_angle = Real.arctan (b / a) * 2 →
  eccentricity = 2 ∨ eccentricity = 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_eccentricity_l2759_275905


namespace trigonometric_ratio_proof_l2759_275989

theorem trigonometric_ratio_proof (α : Real) 
  (h : ∃ (x y : Real), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ x = Real.cos α ∧ y = Real.sin α) : 
  (Real.cos (2*α)) / (1 + Real.sin (2*α)) = -1/7 := by
  sorry

end trigonometric_ratio_proof_l2759_275989


namespace problem_solution_l2759_275976

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := Real.log x + a
noncomputable def h (x : ℝ) := x * f x

theorem problem_solution :
  (∃ (x_min : ℝ), ∀ (x : ℝ), h x ≥ h x_min ∧ h x_min = -1 / Real.exp 1) ∧
  (∀ (a : ℝ), (∃! (p : ℝ), f p = g a p) →
    (∃ (p : ℝ), f p = g a p ∧
      (deriv f p : ℝ) = (deriv (g a) p : ℝ) ∧
      2 < a ∧ a < 5/2)) :=
by sorry

end problem_solution_l2759_275976


namespace probability_of_one_out_of_four_l2759_275900

theorem probability_of_one_out_of_four (S : Finset α) (h : S.card = 4) :
  ∀ a ∈ S, (1 : ℝ) / S.card = 1/4 := by
  sorry

end probability_of_one_out_of_four_l2759_275900


namespace cos_pi_plus_2alpha_l2759_275934

theorem cos_pi_plus_2alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 3) : 
  Real.cos (π + 2 * α) = 7 / 9 := by
  sorry

end cos_pi_plus_2alpha_l2759_275934


namespace subset_intersection_iff_range_l2759_275908

def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

theorem subset_intersection_iff_range (a : ℝ) :
  A a ⊆ (A a ∩ B) ↔ 1 ≤ a ∧ a ≤ 9 := by sorry

end subset_intersection_iff_range_l2759_275908


namespace prime_cube_difference_l2759_275997

theorem prime_cube_difference (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 11 * p = q^3 - r^3 → 
  p = 199 ∧ q = 13 ∧ r = 2 := by
  sorry

end prime_cube_difference_l2759_275997


namespace cuboid_third_edge_length_l2759_275969

/-- Given a cuboid with two edges of 4 cm and 5 cm, and a surface area of 148 cm², 
    the length of the third edge is 6 cm. -/
theorem cuboid_third_edge_length : 
  ∀ (x : ℝ), 
    (2 * (4 * 5 + 4 * x + 5 * x) = 148) → 
    x = 6 := by
  sorry

end cuboid_third_edge_length_l2759_275969


namespace pearl_cutting_theorem_l2759_275984

/-- Represents a string of pearls -/
structure PearlString where
  color : Bool  -- true for black, false for white
  length : Nat
  length_pos : length > 0

/-- The state of the pearl-cutting process -/
structure PearlState where
  strings : List PearlString
  step : Nat

/-- The rules for cutting pearls -/
def cut_pearls (k : Nat) (state : PearlState) : PearlState :=
  sorry

/-- Predicate to check if a white pearl is isolated -/
def has_isolated_white_pearl (state : PearlState) : Prop :=
  sorry

/-- Predicate to check if there's a string of at least two black pearls -/
def has_two_or_more_black_pearls (state : PearlState) : Prop :=
  sorry

/-- The main theorem -/
theorem pearl_cutting_theorem (k b w : Nat) (h1 : k > 0) (h2 : b > w) (h3 : w > 1) :
  ∀ (final_state : PearlState),
    (∃ (initial_state : PearlState),
      initial_state.strings = [PearlString.mk true b sorry, PearlString.mk false w sorry] ∧
      final_state = cut_pearls k initial_state) →
    has_isolated_white_pearl final_state →
    has_two_or_more_black_pearls final_state :=
  sorry

end pearl_cutting_theorem_l2759_275984


namespace locus_of_T_l2759_275935

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse that is not A or B
def M (x y : ℝ) : Prop := ellipse x y ∧ (x, y) ≠ A ∧ (x, y) ≠ B

-- Define point N as the intersection of MP and the ellipse
def N (x y : ℝ) : Prop := 
  M x y → ∃ t : ℝ, ellipse (x + t * (1 - x)) (y + t * (-y)) ∧ t ≠ 0

-- Define point T as the intersection of AM and BN
def T (x y : ℝ) : Prop :=
  ∃ (xm ym : ℝ), M xm ym ∧
  ∃ (xn yn : ℝ), N xn yn ∧
  (y / (x + 2) = ym / (xm + 2)) ∧
  (y / (x - 2) = yn / (xn - 2))

-- Theorem statement
theorem locus_of_T : ∀ x y : ℝ, T x y → y ≠ 0 → x = 4 := by sorry

end locus_of_T_l2759_275935


namespace family_average_age_l2759_275960

theorem family_average_age
  (num_members : ℕ)
  (youngest_age : ℕ)
  (birth_average_age : ℚ)
  (h1 : num_members = 5)
  (h2 : youngest_age = 10)
  (h3 : birth_average_age = 12.5) :
  (birth_average_age * (num_members - 1) + youngest_age * num_members) / num_members = 20 :=
by sorry

end family_average_age_l2759_275960


namespace prob_even_sum_two_balls_l2759_275999

def num_balls : ℕ := 20

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem prob_even_sum_two_balls :
  let total_outcomes := num_balls * (num_balls - 1)
  let favorable_outcomes := (num_balls / 2) * (num_balls / 2 - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 19 := by sorry

end prob_even_sum_two_balls_l2759_275999


namespace max_display_sum_l2759_275949

def hour_sum (h : Nat) : Nat :=
  if h < 10 then h
  else if h < 20 then (h / 10) + (h % 10)
  else 2 + (h % 10)

def minute_sum (m : Nat) : Nat :=
  (m / 10) + (m % 10)

def display_sum (h m : Nat) : Nat :=
  hour_sum h + minute_sum m

theorem max_display_sum :
  (∀ h m, h < 24 → m < 60 → display_sum h m ≤ 24) ∧
  (∃ h m, h < 24 ∧ m < 60 ∧ display_sum h m = 24) :=
sorry

end max_display_sum_l2759_275949


namespace system_solution_unique_l2759_275985

theorem system_solution_unique (x y : ℚ) : 
  (2 * x - 3 * y = 1) ∧ ((2 + x) / 3 = (y + 1) / 4) ↔ (x = -3 ∧ y = -7/3) :=
by sorry

end system_solution_unique_l2759_275985


namespace latest_departure_time_correct_l2759_275962

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : Nat :=
  (t1.hours - t2.hours) * 60 + (t1.minutes - t2.minutes)

/-- The flight departure time -/
def flightTime : Time := { hours := 20, minutes := 0, valid := by simp }

/-- The recommended check-in time in minutes -/
def checkInTime : Nat := 120

/-- The time needed to drive to the airport in minutes -/
def driveTime : Nat := 45

/-- The time needed to park and reach the terminal in minutes -/
def parkAndWalkTime : Nat := 15

/-- The latest time they can leave their house -/
def latestDepartureTime : Time := { hours := 17, minutes := 0, valid := by simp }

theorem latest_departure_time_correct :
  timeDiffMinutes flightTime latestDepartureTime = checkInTime + driveTime + parkAndWalkTime :=
sorry

end latest_departure_time_correct_l2759_275962


namespace arithmetic_calculation_l2759_275948

theorem arithmetic_calculation : 6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 := by
  sorry

end arithmetic_calculation_l2759_275948


namespace negative_fractions_comparison_l2759_275933

theorem negative_fractions_comparison : -1/3 < -1/4 := by
  sorry

end negative_fractions_comparison_l2759_275933


namespace investment_amount_l2759_275980

/-- Calculates the investment amount given the dividend received and share details --/
def calculate_investment (share_value : ℕ) (premium_percentage : ℕ) (dividend_percentage : ℕ) (dividend_received : ℕ) : ℕ :=
  let premium_factor := 1 + premium_percentage / 100
  let share_price := share_value * premium_factor
  let dividend_per_share := share_value * dividend_percentage / 100
  let num_shares := dividend_received / dividend_per_share
  num_shares * share_price

/-- Proves that the investment amount is 14375 given the problem conditions --/
theorem investment_amount : calculate_investment 100 25 5 576 = 14375 := by
  sorry

#eval calculate_investment 100 25 5 576

end investment_amount_l2759_275980


namespace compound_molecular_weight_l2759_275904

/-- Calculates the molecular weight of a compound given the atomic weights and number of atoms of each element. -/
def molecular_weight (al_weight o_weight h_weight : ℝ) (al_count o_count h_count : ℕ) : ℝ :=
  al_weight * al_count + o_weight * o_count + h_weight * h_count

/-- Theorem stating that the molecular weight of a compound with 1 Aluminium, 3 Oxygen, and 3 Hydrogen atoms is 78.001 g/mol. -/
theorem compound_molecular_weight :
  let al_weight : ℝ := 26.98
  let o_weight : ℝ := 15.999
  let h_weight : ℝ := 1.008
  let al_count : ℕ := 1
  let o_count : ℕ := 3
  let h_count : ℕ := 3
  molecular_weight al_weight o_weight h_weight al_count o_count h_count = 78.001 := by
  sorry

end compound_molecular_weight_l2759_275904


namespace selection_ways_eq_756_l2759_275937

/-- The number of ways to select 5 people from a group of 12, 
    where at most 2 out of 3 specific people can be selected -/
def selection_ways : ℕ :=
  Nat.choose 9 5 + 
  (Nat.choose 3 1 * Nat.choose 9 4) + 
  (Nat.choose 3 2 * Nat.choose 9 3)

/-- Theorem stating that the number of selection ways is 756 -/
theorem selection_ways_eq_756 : selection_ways = 756 := by
  sorry

end selection_ways_eq_756_l2759_275937


namespace divisibility_probability_l2759_275929

def is_divisible (r k : ℤ) : Prop := ∃ m : ℤ, r = k * m

def count_divisible_pairs : ℕ := 30

def total_pairs : ℕ := 88

theorem divisibility_probability :
  (count_divisible_pairs : ℚ) / (total_pairs : ℚ) = 15 / 44 := by sorry

end divisibility_probability_l2759_275929


namespace math_city_intersections_l2759_275987

/-- Represents a street in Math City -/
structure Street where
  curved : Bool

/-- Represents Math City -/
structure MathCity where
  streets : Finset Street
  no_parallel : True  -- Assumption that no streets are parallel
  curved_count : Nat
  curved_additional_intersections : Nat

/-- Calculates the maximum number of intersections in Math City -/
def max_intersections (city : MathCity) : Nat :=
  let basic_intersections := city.streets.card.choose 2
  let additional_intersections := city.curved_count * city.curved_additional_intersections
  basic_intersections + additional_intersections

/-- Theorem stating the maximum number of intersections in the given scenario -/
theorem math_city_intersections :
  ∀ (city : MathCity),
    city.streets.card = 10 →
    city.curved_count = 2 →
    city.curved_additional_intersections = 3 →
    max_intersections city = 51 := by
  sorry

end math_city_intersections_l2759_275987


namespace sum_of_function_values_positive_l2759_275968

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem sum_of_function_values_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_monotone : is_monotone_increasing f)
  (h_odd : is_odd_function f)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a3_positive : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end sum_of_function_values_positive_l2759_275968


namespace min_colors_theorem_l2759_275990

theorem min_colors_theorem : ∃ (f : Fin 2013 → Fin 3), 
  (∀ i j : Fin 2013, f i = f j → ¬(((i.val + 1) * (j.val + 1)) % 2014 = 0)) ∧
  (∀ n : ℕ, n < 3 → ¬∃ (g : Fin 2013 → Fin n), 
    ∀ i j : Fin 2013, g i = g j → ¬(((i.val + 1) * (j.val + 1)) % 2014 = 0)) :=
sorry

end min_colors_theorem_l2759_275990


namespace regular_polygon_diagonals_l2759_275902

theorem regular_polygon_diagonals (n : ℕ) (h1 : n > 2) (h2 : (n - 2) * 180 / n = 120) :
  n - 3 = 3 := by sorry

end regular_polygon_diagonals_l2759_275902


namespace concert_attendance_l2759_275910

/-- The number of students from School A who went to the concert -/
def school_a_students : ℕ := 15 * 30

/-- The number of students from School B who went to the concert -/
def school_b_students : ℕ := 18 * 7 + 5 * 6

/-- The number of students from School C who went to the concert -/
def school_c_students : ℕ := 13 * 33 + 10 * 4

/-- The total number of students who went to the concert -/
def total_students : ℕ := school_a_students + school_b_students + school_c_students

theorem concert_attendance : total_students = 1075 := by
  sorry

end concert_attendance_l2759_275910


namespace imaginary_part_of_fraction_l2759_275946

open Complex

theorem imaginary_part_of_fraction (i : ℂ) (h : i * i = -1) :
  (((1 : ℂ) + i) / ((1 : ℂ) - i)).im = 1 := by sorry

end imaginary_part_of_fraction_l2759_275946


namespace train_distance_l2759_275965

/-- Given a train traveling at a certain speed for a certain time, 
    calculate the distance covered. -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
    (h1 : speed = 150) 
    (h2 : time = 8) 
    (h3 : distance = speed * time) : 
  distance = 1200 := by
  sorry

end train_distance_l2759_275965


namespace contaminated_constant_l2759_275982

theorem contaminated_constant (x : ℝ) (h : 2 * (x - 3) - 2 = x + 1) (h_sol : x = 9) : 2 * (9 - 3) - (9 + 1) = 2 := by
  sorry

end contaminated_constant_l2759_275982


namespace four_digit_number_proof_l2759_275907

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (n = 10 * 23) ∧
  (a + b + c + d = 26) ∧
  ((b * d) / 10 % 10 = a + c) ∧
  (∃ m : ℕ, b * d - c^2 = 2^m)

theorem four_digit_number_proof :
  is_valid_number 1979 :=
by sorry

end four_digit_number_proof_l2759_275907
