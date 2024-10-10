import Mathlib

namespace largest_n_for_equation_l1173_117367

theorem largest_n_for_equation : 
  (∃ (x y z : ℕ+), 6^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 9) ∧ 
  (∀ (n : ℕ+), n > 6 → ¬∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 9) :=
by sorry

#check largest_n_for_equation

end largest_n_for_equation_l1173_117367


namespace inequality_system_solution_l1173_117322

/-- Definition of a double subtraction point -/
def is_double_subtraction_point (k b x y : ℝ) : Prop :=
  k ≠ 0 ∧ y = k * x ∧ y = b

/-- The main theorem -/
theorem inequality_system_solution 
  (k : ℝ) 
  (h_k : k ≠ 0)
  (a : ℝ)
  (h_double_sub : is_double_subtraction_point k (a - 2) 3 (3 * k)) :
  {y : ℝ | 2 * (y + 1) < 5 * y - 7 ∧ (y + a) / 2 < 5} = {y : ℝ | 3 < y ∧ y < 8} :=
sorry

end inequality_system_solution_l1173_117322


namespace original_candle_length_l1173_117318

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (original_length : ℝ) : 
  current_length = 48 →
  factor = 1.33 →
  original_length = current_length * factor →
  original_length = 63.84 := by
sorry

end original_candle_length_l1173_117318


namespace no_solution_exists_l1173_117343

theorem no_solution_exists : ¬∃ x : ℝ, x > 0 ∧ x * Real.sqrt (9 - x) + Real.sqrt (9 * x - x^3) ≥ 10 := by
  sorry

end no_solution_exists_l1173_117343


namespace product_change_theorem_l1173_117390

theorem product_change_theorem (k : ℝ) (x y z : ℝ) (h1 : x * y * z = k) :
  ∃ (p q : ℝ),
    1.805 * (1 - p / 100) * (1 + q / 100) = 1 ∧
    Real.log p - Real.cos q = 0 ∧
    x * 1.805 * y * (1 - p / 100) * z * (1 + q / 100) = k := by
  sorry

end product_change_theorem_l1173_117390


namespace eight_digit_numbers_a_eight_digit_numbers_b_eight_digit_numbers_b_start_with_1_l1173_117397

-- Define the set of digits for part a
def digits_a : Finset ℕ := {0, 1, 2}

-- Define the multiset of digits for part b
def digits_b : Multiset ℕ := {0, 0, 0, 1, 2, 2, 2, 2}

-- Define the number of digits in the numbers we're forming
def num_digits : ℕ := 8

-- Theorem for part a
theorem eight_digit_numbers_a : 
  (Finset.card digits_a ^ num_digits) - (Finset.card digits_a ^ (num_digits - 1)) = 4374 :=
sorry

-- Theorem for part b (total valid numbers)
theorem eight_digit_numbers_b : 
  (Multiset.card digits_b).factorial / ((Multiset.count 0 digits_b).factorial * (Multiset.count 2 digits_b).factorial) - 
  ((Multiset.card digits_b - 1).factorial / ((Multiset.count 0 digits_b - 1).factorial * (Multiset.count 2 digits_b).factorial)) = 175 :=
sorry

-- Theorem for part b (numbers starting with 1)
theorem eight_digit_numbers_b_start_with_1 : 
  (Multiset.card digits_b - 1).factorial / ((Multiset.count 0 digits_b).factorial * (Multiset.count 2 digits_b).factorial) = 35 :=
sorry

end eight_digit_numbers_a_eight_digit_numbers_b_eight_digit_numbers_b_start_with_1_l1173_117397


namespace test_score_calculation_l1173_117307

/-- The average test score for a portion of the class -/
def average_score (portion : ℝ) (score : ℝ) : ℝ := portion * score

/-- The overall class average -/
def class_average (score1 : ℝ) (score2 : ℝ) (score3 : ℝ) : ℝ :=
  average_score 0.45 0.95 + average_score 0.50 score2 + average_score 0.05 0.60

theorem test_score_calculation (score2 : ℝ) :
  class_average 0.95 score2 0.60 = 0.8475 → score2 = 0.78 := by
  sorry

end test_score_calculation_l1173_117307


namespace regular_octagon_perimeter_regular_octagon_perimeter_three_l1173_117317

/-- The perimeter of a regular octagon with side length 3 is 24 -/
theorem regular_octagon_perimeter : ℕ → ℕ
  | side_length =>
    8 * side_length

theorem regular_octagon_perimeter_three : regular_octagon_perimeter 3 = 24 := by
  sorry

end regular_octagon_perimeter_regular_octagon_perimeter_three_l1173_117317


namespace total_snowfall_calculation_l1173_117349

theorem total_snowfall_calculation (monday tuesday wednesday : Real) 
  (h1 : monday = 0.327)
  (h2 : tuesday = 0.216)
  (h3 : wednesday = 0.184) :
  monday + tuesday + wednesday = 0.727 := by
  sorry

end total_snowfall_calculation_l1173_117349


namespace rectangular_plot_length_l1173_117319

/-- Proves that the length of a rectangular plot is 63 meters given the specified conditions -/
theorem rectangular_plot_length : 
  ∀ (breadth length : ℝ),
  length = breadth + 26 →
  2 * (length + breadth) * 26.5 = 5300 →
  length = 63 := by
sorry

end rectangular_plot_length_l1173_117319


namespace unique_line_divides_triangle_l1173_117362

/-- A triangle in a 2D plane --/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- A line in the form y = mx --/
structure Line where
  m : ℝ

/-- Checks if a line divides a triangle into two equal areas --/
def dividesEqualArea (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The specific triangle in the problem --/
def specificTriangle : Triangle :=
  { v1 := (0, 0),
    v2 := (4, 4),
    v3 := (12, 0) }

theorem unique_line_divides_triangle :
  ∃! m : ℝ, dividesEqualArea specificTriangle { m := m } ∧ m = 1/4 := by
  sorry

end unique_line_divides_triangle_l1173_117362


namespace distance_driven_l1173_117330

/-- Represents the efficiency of a car in kilometers per gallon -/
def car_efficiency : ℝ := 10

/-- Represents the amount of gas available in gallons -/
def gas_available : ℝ := 10

/-- Theorem stating the distance that can be driven given the car's efficiency and available gas -/
theorem distance_driven : car_efficiency * gas_available = 100 := by sorry

end distance_driven_l1173_117330


namespace sum_of_three_squares_l1173_117347

theorem sum_of_three_squares (a k : ℕ) :
  ¬ ∃ x y z : ℤ, (4^a * (8*k + 7) : ℤ) = x^2 + y^2 + z^2 := by
  sorry

end sum_of_three_squares_l1173_117347


namespace cubes_to_add_l1173_117301

theorem cubes_to_add (small_cube_side : ℕ) (large_cube_side : ℕ) (add_cube_side : ℕ) : 
  small_cube_side = 8 →
  large_cube_side = 12 →
  add_cube_side = 2 →
  (large_cube_side^3 - small_cube_side^3) / add_cube_side^3 = 152 := by
  sorry

end cubes_to_add_l1173_117301


namespace frustum_small_cone_altitude_l1173_117379

-- Define the frustum
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

-- Define the theorem
theorem frustum_small_cone_altitude 
  (f : Frustum) 
  (h1 : f.altitude = 30)
  (h2 : f.lower_base_area = 324 * Real.pi)
  (h3 : f.upper_base_area = 36 * Real.pi) : 
  ∃ (small_cone_altitude : ℝ), small_cone_altitude = 15 := by
  sorry

end frustum_small_cone_altitude_l1173_117379


namespace min_overtakes_is_five_l1173_117325

/-- Represents the girls in the race -/
inductive Girl
| Fiona
| Gertrude
| Hannah
| India
| Janice

/-- Represents the order of girls in the race -/
def RaceOrder := List Girl

/-- The initial order of the girls in the race -/
def initial_order : RaceOrder :=
  [Girl.Fiona, Girl.Gertrude, Girl.Hannah, Girl.India, Girl.Janice]

/-- The final order of the girls in the race -/
def final_order : RaceOrder :=
  [Girl.India, Girl.Gertrude, Girl.Fiona, Girl.Janice, Girl.Hannah]

/-- Calculates the minimum number of overtakes required to transform the initial order to the final order -/
def min_overtakes (initial : RaceOrder) (final : RaceOrder) : Nat :=
  sorry

/-- Theorem stating that the minimum number of overtakes is 5 -/
theorem min_overtakes_is_five :
  min_overtakes initial_order final_order = 5 := by
  sorry

end min_overtakes_is_five_l1173_117325


namespace hyperbola_eccentricity_l1173_117368

/-- The eccentricity of a hyperbola given its equation and asymptote angle -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1 →
  (angle_between_asymptotes : ℝ) → angle_between_asymptotes = π / 3 →
  ∃ (e : ℝ), (e = 2*Real.sqrt 3/3 ∨ e = 2) ∧ 
  e^2 * a^2 = a^2 + b^2 := by
  sorry

end hyperbola_eccentricity_l1173_117368


namespace task_assignments_and_arrangements_l1173_117369

def num_volunteers : ℕ := 5
def num_tasks : ℕ := 4

theorem task_assignments_and_arrangements :
  let assign_all_tasks := (num_volunteers.choose 2) * num_tasks.factorial
  let assign_one_task_to_two := (num_volunteers.choose 2) * (num_tasks - 1).factorial
  let photo_arrangement := (num_volunteers.factorial / (num_volunteers - 2).factorial) * 2
  (assign_all_tasks = 240) ∧
  (assign_one_task_to_two = 60) ∧
  (photo_arrangement = 40) := by sorry

end task_assignments_and_arrangements_l1173_117369


namespace point_on_y_axis_l1173_117300

/-- A point M with coordinates (t-3, 5-t) is on the y-axis if and only if its coordinates are (0, 2) -/
theorem point_on_y_axis (t : ℝ) :
  (t - 3 = 0 ∧ (t - 3, 5 - t) = (0, 2)) ↔ (t - 3, 5 - t).1 = 0 := by
  sorry

end point_on_y_axis_l1173_117300


namespace expression_simplification_l1173_117316

theorem expression_simplification (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) = (3 * x - 2) / (x + 2) := by
  sorry

end expression_simplification_l1173_117316


namespace smallest_number_divisible_by_all_l1173_117309

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 6) % 12 = 0 ∧
  (n - 6) % 16 = 0 ∧
  (n - 6) % 18 = 0 ∧
  (n - 6) % 21 = 0 ∧
  (n - 6) % 28 = 0 ∧
  (n - 6) % 35 = 0 ∧
  (n - 6) % 39 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 65526 ∧
  ∀ m : ℕ, m < 65526 → ¬is_divisible_by_all m :=
by sorry

end smallest_number_divisible_by_all_l1173_117309


namespace fraction_equality_l1173_117327

theorem fraction_equality : 
  (4 + 2/3 + 3 + 1/3) - (2 + 1/2 - 1/2) = 4 + 2/3 - (2 + 1/2) + 1/2 + 3 + 1/3 := by
  sorry

end fraction_equality_l1173_117327


namespace greatest_integer_satisfying_conditions_l1173_117302

theorem greatest_integer_satisfying_conditions : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k m : ℕ), n = 9 * k - 2 ∧ n = 11 * m - 4) ∧
  (∀ (n' : ℕ), n' < 150 → 
    (∃ (k' m' : ℕ), n' = 9 * k' - 2 ∧ n' = 11 * m' - 4) → 
    n' ≤ n) ∧
  n = 139 := by
sorry

end greatest_integer_satisfying_conditions_l1173_117302


namespace probability_two_changing_yao_l1173_117376

theorem probability_two_changing_yao (n : Nat) (p : Real) (k : Nat) : 
  n = 6 → p = 1/4 → k = 2 →
  Nat.choose n k * p^k * (1-p)^(n-k) = 1215/4096 := by
sorry

end probability_two_changing_yao_l1173_117376


namespace honeycomb_thickness_scientific_notation_l1173_117332

theorem honeycomb_thickness_scientific_notation :
  0.000073 = 7.3 * 10^(-5) := by
  sorry

end honeycomb_thickness_scientific_notation_l1173_117332


namespace largest_integer_in_range_l1173_117326

theorem largest_integer_in_range : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/5 ∧ (x : ℚ)/5 < 2/3 ∧ 
  ∀ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/5 ∧ (y : ℚ)/5 < 2/3 → y ≤ x :=
by
  -- The proof goes here
  sorry

end largest_integer_in_range_l1173_117326


namespace cube_root_negative_three_l1173_117381

theorem cube_root_negative_three (x : ℝ) : x^(1/3) = (-3)^(1/3) → x = -3 := by
  sorry

end cube_root_negative_three_l1173_117381


namespace sum_of_coordinates_after_reflection_l1173_117303

/-- Given a point A with coordinates (3, y) and its reflection B over the x-axis,
    prove that the sum of all coordinates of A and B is 6. -/
theorem sum_of_coordinates_after_reflection (y : ℝ) : 
  let A : ℝ × ℝ := (3, y)
  let B : ℝ × ℝ := (3, -y)  -- reflection of A over x-axis
  (A.1 + A.2 + B.1 + B.2) = 6 := by
sorry

end sum_of_coordinates_after_reflection_l1173_117303


namespace ratio_problem_l1173_117352

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3) 
  (h2 : c / b = 2) 
  (h3 : d / c = 4) : 
  (a + c) / (b + d) = 7 / 27 := by
  sorry

end ratio_problem_l1173_117352


namespace battery_current_at_12_ohms_l1173_117374

/-- A battery with voltage 48V and a relationship between current and resistance --/
structure Battery where
  voltage : ℝ
  current : ℝ → ℝ
  resistance : ℝ
  h_voltage : voltage = 48
  h_current : ∀ r, current r = voltage / r

/-- The theorem states that for a battery with 48V and the given current-resistance relationship,
    when the resistance is 12Ω, the current is 4A --/
theorem battery_current_at_12_ohms (b : Battery) (h : b.resistance = 12) :
  b.current b.resistance = 4 := by
  sorry

end battery_current_at_12_ohms_l1173_117374


namespace optimal_well_placement_l1173_117393

/-- Three houses positioned along a straight road -/
structure Village where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between adjacent houses is 50 meters -/
def house_distance : ℝ := 50

/-- A village with houses positioned at the correct intervals -/
def village : Village :=
  { A := 0,
    B := house_distance,
    C := 2 * house_distance }

/-- The sum of distances from a point to all houses -/
def total_distance (x : ℝ) : ℝ :=
  |x - village.A| + |x - village.B| + |x - village.C|

/-- The well position that minimizes the total distance -/
def optimal_well_position : ℝ := village.B

theorem optimal_well_placement :
  ∀ x : ℝ, total_distance optimal_well_position ≤ total_distance x :=
sorry

end optimal_well_placement_l1173_117393


namespace incorrect_equation_property_l1173_117357

theorem incorrect_equation_property : ¬ (∀ a b : ℝ, a * b = a → b = 1) := by
  sorry

end incorrect_equation_property_l1173_117357


namespace bird_nest_problem_l1173_117360

theorem bird_nest_problem (birds : ℕ) (difference : ℕ) (nests : ℕ) : 
  birds = 6 → difference = 3 → birds - nests = difference → nests = 3 := by
  sorry

end bird_nest_problem_l1173_117360


namespace intersection_in_second_quadrant_l1173_117361

theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k) →
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k ∧ x < 0 ∧ y > 0) →
  0 < k ∧ k < 1/2 :=
by sorry

end intersection_in_second_quadrant_l1173_117361


namespace total_holiday_savings_l1173_117382

/-- The total money saved for holiday spending by Victory and Sam -/
theorem total_holiday_savings (sam_savings : ℕ) (victory_savings : ℕ) : 
  sam_savings = 1200 → 
  victory_savings = sam_savings - 200 →
  sam_savings + victory_savings = 2200 := by
sorry

end total_holiday_savings_l1173_117382


namespace remainder_of_1235678901_mod_101_l1173_117341

theorem remainder_of_1235678901_mod_101 : 1235678901 % 101 = 1 := by
  sorry

end remainder_of_1235678901_mod_101_l1173_117341


namespace inequality_chain_l1173_117370

theorem inequality_chain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / b + 1 / c ≥ 2 / (a + b) + 2 / (b + c) + 2 / (c + a) ∧
  2 / (a + b) + 2 / (b + c) + 2 / (c + a) ≥ 9 / (a + b + c) := by
  sorry

end inequality_chain_l1173_117370


namespace fish_fillet_problem_l1173_117366

theorem fish_fillet_problem (total : ℕ) (team1 : ℕ) (team2 : ℕ) 
  (h1 : total = 500) 
  (h2 : team1 = 189) 
  (h3 : team2 = 131) : 
  total - (team1 + team2) = 180 := by
sorry

end fish_fillet_problem_l1173_117366


namespace min_value_theorem_l1173_117334

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4*x + 3*y = 1) :
  ∀ z : ℝ, z = (1 / (2*x - y)) + (2 / (x + 2*y)) → z ≥ 9 :=
by sorry

end min_value_theorem_l1173_117334


namespace solution_set_f_geq_3_range_of_x_satisfying_inequality_l1173_117331

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the range of x satisfying the inequality
theorem range_of_x_satisfying_inequality :
  {x : ℝ | ∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ a * f x} = 
  {x : ℝ | 1/2 ≤ x ∧ x ≤ 5/2} := by sorry

end solution_set_f_geq_3_range_of_x_satisfying_inequality_l1173_117331


namespace binary_remainder_by_four_l1173_117375

/-- The binary number 111001101101₂ -/
def binary_number : Nat := 3693

/-- Theorem: The remainder when 111001101101₂ is divided by 4 is 1 -/
theorem binary_remainder_by_four :
  binary_number % 4 = 1 := by
  sorry

end binary_remainder_by_four_l1173_117375


namespace simplify_and_evaluate_l1173_117395

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (1 - 1 / (a + 1)) * ((a^2 + 2*a + 1) / a) = Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l1173_117395


namespace digit_sum_problem_l1173_117313

theorem digit_sum_problem (P Q R S : ℕ) : 
  P < 10 → Q < 10 → R < 10 → S < 10 →
  P * 100 + 45 + Q * 10 + R + S = 654 →
  P + Q + R + S = 15 := by
  sorry

end digit_sum_problem_l1173_117313


namespace next_feb29_sunday_l1173_117339

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 ≠ 0 || year % 400 == 0)

/-- Advances the day of the week by the given number of days -/
def advanceDayOfWeek (day : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (day, days % 7) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday  -- Default case, should not occur

/-- Calculates the day of the week for February 29 in the given year, starting from 2004 -/
def feb29DayOfWeek (year : Nat) : DayOfWeek :=
  let daysAdvanced := (year - 2004) / 4 * 2  -- Each leap year advances by 2 days
  advanceDayOfWeek DayOfWeek.Sunday daysAdvanced

/-- Theorem: The next year after 2004 when February 29 falls on a Sunday is 2032 -/
theorem next_feb29_sunday : 
  (∀ y : Nat, 2004 < y → y < 2032 → feb29DayOfWeek y ≠ DayOfWeek.Sunday) ∧ 
  feb29DayOfWeek 2032 = DayOfWeek.Sunday :=
sorry

end next_feb29_sunday_l1173_117339


namespace arithmetic_sequence_property_l1173_117342

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def f (a : ℕ → ℝ) : ℝ := 1  -- Definition of f, which always returns 1

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 5 / a 3 = 5 / 9) : 
  f a = 1 := by sorry

end arithmetic_sequence_property_l1173_117342


namespace brokerage_percentage_calculation_l1173_117336

/-- The brokerage percentage calculation problem -/
theorem brokerage_percentage_calculation
  (cash_realized : ℝ)
  (total_amount : ℝ)
  (h1 : cash_realized = 106.25)
  (h2 : total_amount = 106) :
  let brokerage_amount := cash_realized - total_amount
  let brokerage_percentage := (brokerage_amount / total_amount) * 100
  ∃ ε > 0, abs (brokerage_percentage - 0.236) < ε :=
by sorry

end brokerage_percentage_calculation_l1173_117336


namespace stating_standard_polar_coord_example_l1173_117314

/-- 
Given a point in polar coordinates (r, θ) where r can be negative,
this function returns the equivalent standard polar coordinate representation
where r > 0 and 0 ≤ θ < 2π.
-/
def standardPolarCoord (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  sorry

/-- 
Theorem stating that the standard polar coordinate representation
of the point (-3, 5π/6) is (3, 11π/6).
-/
theorem standard_polar_coord_example : 
  standardPolarCoord (-3) (5 * Real.pi / 6) = (3, 11 * Real.pi / 6) := by
  sorry

end stating_standard_polar_coord_example_l1173_117314


namespace distance_between_intersection_points_l1173_117371

/-- The curve C in rectangular coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 = 4*y

/-- The line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := y = x + 1

/-- The intersection points of curve C and line l -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve_C p.1 p.2 ∧ line_l p.1 p.2}

theorem distance_between_intersection_points :
  ∃ (p q : ℝ × ℝ), p ∈ intersection_points ∧ q ∈ intersection_points ∧ p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 8 :=
sorry

end distance_between_intersection_points_l1173_117371


namespace product_equals_square_minus_one_l1173_117333

theorem product_equals_square_minus_one (r : ℕ) (hr : r > 5) :
  let a := r^3 + r^2 + r
  (a) * (a + 1) * (a + 2) * (a + 3) = (r^6 + 2*r^5 + 3*r^4 + 5*r^3 + 4*r^2 + 3*r + 1)^2 - 1 :=
by sorry

end product_equals_square_minus_one_l1173_117333


namespace zero_in_interval_l1173_117399

-- Define the function f(x) = x³ - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- State the theorem
theorem zero_in_interval :
  (f 1.5 < 0) → (f 2 > 0) → ∃ x, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 := by
  sorry

-- Note: Set.Ioo represents an open interval (1.5, 2)

end zero_in_interval_l1173_117399


namespace problem_solution_l1173_117305

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + 2*x + 2*y = 7) :
  x^2*y + x*y^2 = 245/121 := by
sorry

end problem_solution_l1173_117305


namespace binomial_16_13_l1173_117351

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by sorry

end binomial_16_13_l1173_117351


namespace swimming_frequency_difference_l1173_117396

def camden_total : ℕ := 16
def susannah_total : ℕ := 24
def weeks_in_month : ℕ := 4

theorem swimming_frequency_difference :
  (susannah_total / weeks_in_month) - (camden_total / weeks_in_month) = 2 :=
by sorry

end swimming_frequency_difference_l1173_117396


namespace base8_addition_l1173_117340

/-- Addition in base 8 -/
def base8_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def to_base8 (n : ℕ) : ℕ := sorry

/-- Conversion from base 8 to base 10 -/
def from_base8 (n : ℕ) : ℕ := sorry

theorem base8_addition : base8_add (from_base8 12) (from_base8 157) = from_base8 171 := by sorry

end base8_addition_l1173_117340


namespace photo_collection_inconsistency_l1173_117380

/-- Represents the number of photos each person has --/
structure PhotoCollection where
  tom : ℕ
  tim : ℕ
  paul : ℕ
  jane : ℕ

/-- The problem statement --/
theorem photo_collection_inconsistency 
  (photos : PhotoCollection) 
  (total_photos : photos.tom + photos.tim + photos.paul + photos.jane = 200)
  (paul_more_than_tim : photos.paul = photos.tim + 10)
  (tim_less_than_total : photos.tim = 200 - 100) :
  False :=
by
  sorry


end photo_collection_inconsistency_l1173_117380


namespace tan_two_implies_fraction_l1173_117308

theorem tan_two_implies_fraction (θ : Real) (h : Real.tan θ = 2) :
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1/3 := by
  sorry

end tan_two_implies_fraction_l1173_117308


namespace houses_around_square_l1173_117320

/-- The number of houses around the square. -/
def n : ℕ := 32

/-- Maria's starting position relative to João's. -/
def m_start : ℕ := 8

/-- Proposition that the given conditions imply there are 32 houses around the square. -/
theorem houses_around_square :
  (∀ k : ℕ, (k + 5 - m_start) % n = (k + 12) % n) ∧
  (∀ k : ℕ, (k + 30 - m_start) % n = (k + 5) % n) →
  n = 32 :=
by sorry

end houses_around_square_l1173_117320


namespace sum_of_three_integers_2015_l1173_117312

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem sum_of_three_integers_2015 :
  ∃ (a b c : ℕ),
    a + b + c = 2015 ∧
    is_prime a ∧
    b % 3 = 0 ∧
    400 < c ∧ c < 500 ∧
    ¬(c % 3 = 0) :=
by sorry

end sum_of_three_integers_2015_l1173_117312


namespace chef_nuts_weight_l1173_117344

/-- The total weight of nuts bought by a chef -/
def total_weight (almonds pecans walnuts cashews pistachios : ℝ) : ℝ :=
  almonds + pecans + walnuts + cashews + pistachios

/-- Theorem stating that the total weight of nuts is 1.50 kg -/
theorem chef_nuts_weight :
  let almonds : ℝ := 0.14
  let pecans : ℝ := 0.38
  let walnuts : ℝ := 0.22
  let cashews : ℝ := 0.47
  let pistachios : ℝ := 0.29
  total_weight almonds pecans walnuts cashews pistachios = 1.50 := by
  sorry

end chef_nuts_weight_l1173_117344


namespace fraction_inequality_condition_l1173_117364

theorem fraction_inequality_condition (x : ℝ) : 
  (2 * x + 1) / (1 - x) ≥ 0 ↔ -1/2 ≤ x ∧ x < 1 :=
by sorry

end fraction_inequality_condition_l1173_117364


namespace race_track_width_l1173_117337

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 440 →
  outer_radius = 84.02817496043394 →
  ∃ width : ℝ, abs (width - 14.02056077700854) < 1e-10 ∧
    width = outer_radius - inner_circumference / (2 * Real.pi) :=
by sorry

end race_track_width_l1173_117337


namespace simplify_expression_l1173_117372

theorem simplify_expression :
  1 / ((3 / (Real.sqrt 2 + 2)) + (4 / (Real.sqrt 5 - 2))) =
  1 / (11 + 4 * Real.sqrt 5 - (3 * Real.sqrt 2) / 2) := by
  sorry

end simplify_expression_l1173_117372


namespace remainder_divisibility_l1173_117350

theorem remainder_divisibility (n : ℕ) (h : n % 10 = 7) : n % 5 = 2 := by
  sorry

end remainder_divisibility_l1173_117350


namespace quadratic_equation_m_value_l1173_117335

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m - 3) * x^(m^2 - 7) - 4*x - 8 = a*x^2 + b*x + c) →
  (m - 3 ≠ 0) →
  m = -3 :=
by sorry

end quadratic_equation_m_value_l1173_117335


namespace sum_of_coefficients_equals_two_l1173_117310

theorem sum_of_coefficients_equals_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10 + a₁₁*(x - 1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 := by
sorry

end sum_of_coefficients_equals_two_l1173_117310


namespace bank_investment_problem_l1173_117387

theorem bank_investment_problem (total_investment interest_rate1 interest_rate2 total_interest : ℝ)
  (h1 : total_investment = 5000)
  (h2 : interest_rate1 = 0.04)
  (h3 : interest_rate2 = 0.065)
  (h4 : total_interest = 282.5)
  (h5 : ∃ x y : ℝ, x + y = total_investment ∧ interest_rate1 * x + interest_rate2 * y = total_interest) :
  ∃ x : ℝ, x = 1700 ∧ interest_rate1 * x + interest_rate2 * (total_investment - x) = total_interest :=
by
  sorry

end bank_investment_problem_l1173_117387


namespace arithmetic_sequence_term_count_l1173_117328

/-- 
Given an arithmetic sequence with:
  - First term: 156
  - Last term: 36
  - Common difference: -6
This theorem proves that the number of terms in the sequence is 21.
-/
theorem arithmetic_sequence_term_count : 
  let a₁ : ℤ := 156  -- First term
  let aₙ : ℤ := 36   -- Last term
  let d : ℤ := -6    -- Common difference
  ∃ n : ℕ, n > 0 ∧ aₙ = a₁ + (n - 1) * d ∧ n = 21
  := by sorry

end arithmetic_sequence_term_count_l1173_117328


namespace total_gulbis_count_l1173_117311

/-- The number of dureums of gulbis -/
def num_dureums : ℕ := 156

/-- The number of gulbis in one dureum -/
def gulbis_per_dureum : ℕ := 20

/-- The total number of gulbis -/
def total_gulbis : ℕ := num_dureums * gulbis_per_dureum

theorem total_gulbis_count : total_gulbis = 3120 := by
  sorry

end total_gulbis_count_l1173_117311


namespace least_nickels_l1173_117385

theorem least_nickels (n : ℕ) : 
  (n > 0) → 
  (n % 7 = 2) → 
  (n % 4 = 3) → 
  (∀ m : ℕ, m > 0 → m % 7 = 2 → m % 4 = 3 → n ≤ m) → 
  n = 23 := by
sorry

end least_nickels_l1173_117385


namespace f_intersects_all_lines_l1173_117329

/-- A function that intersects every line in the coordinate plane at least once -/
def f (x : ℝ) : ℝ := x^3

/-- Proposition: The function f intersects every line in the coordinate plane at least once -/
theorem f_intersects_all_lines :
  ∀ (k b : ℝ), ∃ (x : ℝ), f x = k * x + b :=
sorry

end f_intersects_all_lines_l1173_117329


namespace quadrilateral_congruence_l1173_117306

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The median line of a quadrilateral -/
def median_line (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Two quadrilaterals are equal if their corresponding sides and median lines are equal -/
theorem quadrilateral_congruence (q1 q2 : Quadrilateral) :
  (q1.A = q2.A ∧ q1.B = q2.B ∧ q1.C = q2.C ∧ q1.D = q2.D) →
  median_line q1 = median_line q2 →
  q1 = q2 :=
sorry

end quadrilateral_congruence_l1173_117306


namespace m_range_l1173_117358

theorem m_range : ∀ m : ℝ, m = 5 * Real.sqrt (1/5) - Real.sqrt 45 → -5 < m ∧ m < -4 := by
  sorry

end m_range_l1173_117358


namespace initial_books_correct_l1173_117388

/-- Calculates the initial number of books in Mary's mystery book library --/
def initial_books : ℕ :=
  let books_received := 12 -- 1 book per month for 12 months
  let books_bought := 5 + 2 -- 5 from bookstore, 2 from yard sales
  let books_gifted := 1 + 4 -- 1 from daughter, 4 from mother
  let books_removed := 12 + 3 -- 12 donated, 3 sold
  let final_books := 81

  final_books - (books_received + books_bought + books_gifted) + books_removed

theorem initial_books_correct :
  initial_books = 72 :=
by
  sorry

end initial_books_correct_l1173_117388


namespace offspring_trisomy_is_heritable_variation_l1173_117315

-- Define the genotype structure
structure Genotype where
  allele1 : Char
  allele2 : Char
  allele3 : Char
  allele4 : Char

-- Define the chromosome structure
structure Chromosome where
  gene1 : Char
  gene2 : Char

-- Define the diploid tomato
def diploidTomato : Genotype := { allele1 := 'A', allele2 := 'a', allele3 := 'B', allele4 := 'b' }

-- Define the offspring with trisomy
def offspringTrisomy : Genotype := { allele1 := 'A', allele2 := 'a', allele3 := 'B', allele4 := 'b' }

-- Define the property of genes being on different homologous chromosomes
def genesOnDifferentChromosomes (g : Genotype) : Prop :=
  ∃ (c1 c2 : Chromosome), (c1.gene1 = g.allele1 ∧ c1.gene2 = g.allele3) ∧
                          (c2.gene1 = g.allele2 ∧ c2.gene2 = g.allele4)

-- Define heritable variation
def heritableVariation (parent offspring : Genotype) : Prop :=
  parent ≠ offspring ∧ ∃ (gene : Char), (gene ∈ [parent.allele1, parent.allele2, parent.allele3, parent.allele4]) ∧
                                        (gene ∈ [offspring.allele1, offspring.allele2, offspring.allele3, offspring.allele4])

-- Theorem statement
theorem offspring_trisomy_is_heritable_variation :
  genesOnDifferentChromosomes diploidTomato →
  heritableVariation diploidTomato offspringTrisomy :=
by sorry

end offspring_trisomy_is_heritable_variation_l1173_117315


namespace pta_fundraiser_l1173_117389

theorem pta_fundraiser (initial_amount : ℝ) (school_supplies_fraction : ℝ) (food_fraction : ℝ) : 
  initial_amount = 400 →
  school_supplies_fraction = 1/4 →
  food_fraction = 1/2 →
  initial_amount * (1 - school_supplies_fraction) * (1 - food_fraction) = 150 := by
sorry

end pta_fundraiser_l1173_117389


namespace min_sum_squares_roots_l1173_117378

theorem min_sum_squares_roots (m : ℝ) (α β : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 2 - m^2 = 0 ↔ x = α ∨ x = β) → 
  ∃ (k : ℝ), ∀ m : ℝ, α^2 + β^2 ≥ k ∧ ∃ m : ℝ, α^2 + β^2 = k :=
by sorry

end min_sum_squares_roots_l1173_117378


namespace bob_bought_four_candies_l1173_117304

/-- The number of candies bought by each person -/
structure CandyPurchase where
  emily : ℕ
  jennifer : ℕ
  bob : ℕ

/-- The conditions of the candy purchase scenario -/
def candy_scenario (p : CandyPurchase) : Prop :=
  p.emily = 6 ∧
  p.jennifer = 2 * p.emily ∧
  p.jennifer = 3 * p.bob

/-- Theorem stating that Bob bought 4 candies -/
theorem bob_bought_four_candies :
  ∀ p : CandyPurchase, candy_scenario p → p.bob = 4 := by
  sorry

end bob_bought_four_candies_l1173_117304


namespace can_capacity_proof_l1173_117321

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 8

theorem can_capacity_proof (initial : CanContents) (final : CanContents) :
  -- Initial ratio of milk to water is 1:5
  initial.milk / initial.water = 1 / 5 →
  -- Final contents after adding 2 liters of milk
  final.milk = initial.milk + 2 ∧
  final.water = initial.water →
  -- New ratio of milk to water is 3:5
  final.milk / final.water = 3 / 5 →
  -- The can is full after adding 2 liters of milk
  final.milk + final.water = canCapacity :=
by sorry


end can_capacity_proof_l1173_117321


namespace max_value_of_sum_of_squares_max_value_achieved_l1173_117363

theorem max_value_of_sum_of_squares (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ 2 := by
  sorry

-- The maximum value is indeed achieved
theorem max_value_achieved : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y ≥ x^3 + y^2 ∧ x^2 + y^2 = 2 := by
  sorry

end max_value_of_sum_of_squares_max_value_achieved_l1173_117363


namespace angle_problem_l1173_117323

theorem angle_problem (angle1 angle2 angle3 angle4 angle5 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle5 →
  angle3 + angle4 = 180 →
  angle4 = 35 := by
sorry

end angle_problem_l1173_117323


namespace forty_percent_of_two_l1173_117346

theorem forty_percent_of_two : (40 / 100) * 2 = 0.8 := by
  sorry

end forty_percent_of_two_l1173_117346


namespace factorial_squared_greater_than_power_l1173_117391

theorem factorial_squared_greater_than_power (n : ℕ) (h : n > 2) : (n.factorial ^ 2 : ℝ) > n ^ n := by
  sorry

end factorial_squared_greater_than_power_l1173_117391


namespace sphere_surface_area_ratio_l1173_117324

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 3) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 := by
  sorry

end sphere_surface_area_ratio_l1173_117324


namespace frustum_central_angle_l1173_117356

/-- Represents a frustum of a cone -/
structure Frustum where
  lateral_area : ℝ
  total_area : ℝ

/-- 
Given a frustum of a cone with lateral surface area 10π and total surface area 19π,
the central angle of the lateral surface when laid flat is 324°.
-/
theorem frustum_central_angle (f : Frustum) 
  (h1 : f.lateral_area = 10 * Real.pi)
  (h2 : f.total_area = 19 * Real.pi) : 
  ∃ (angle : ℝ), angle = 324 ∧ 
  (angle / 360) * Real.pi * ((6 * 360) / angle)^2 = f.lateral_area := by
  sorry


end frustum_central_angle_l1173_117356


namespace area_of_PQRS_l1173_117348

/-- Reflect a point (x, y) in the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflect a point (x, y) in the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Reflect a point (x, y) in the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Calculate the area of a quadrilateral given its four vertices -/
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ := sorry

theorem area_of_PQRS : 
  let P : ℝ × ℝ := (-1, 4)
  let Q := reflect_y P
  let R := reflect_y_eq_x Q
  let S := reflect_x R
  quadrilateral_area P Q R S = 8 := by sorry

end area_of_PQRS_l1173_117348


namespace irrational_sqrt_three_others_rational_l1173_117394

theorem irrational_sqrt_three_others_rational :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 3 = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-1 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (1/2 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = a / b) :=
by sorry

end irrational_sqrt_three_others_rational_l1173_117394


namespace expression_simplification_and_evaluation_l1173_117384

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  1 - (x / (x + 1)) / (x / (x^2 - 1)) = 3 - Real.sqrt 3 :=
by sorry

end expression_simplification_and_evaluation_l1173_117384


namespace fraction_of_number_l1173_117353

theorem fraction_of_number (original : ℕ) (target : ℚ) : 
  original = 5040 → target = 756.0000000000001 → 
  (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * original = target := by
  sorry

end fraction_of_number_l1173_117353


namespace study_days_needed_l1173_117345

/-- Represents the study requirements for a subject --/
structure SubjectRequirements where
  chapters : ℕ
  worksheets : ℕ
  chapterTime : ℚ
  worksheetTime : ℚ

/-- Calculates the total study time for a subject --/
def totalStudyTime (req : SubjectRequirements) : ℚ :=
  req.chapters * req.chapterTime + req.worksheets * req.worksheetTime

/-- Represents the break schedule --/
structure BreakSchedule where
  firstThreeHours : ℚ
  nextThreeHours : ℚ
  lastHour : ℚ
  snackBreaks : ℚ
  lunchBreak : ℚ

/-- Calculates the total break time per day --/
def totalBreakTime (schedule : BreakSchedule) : ℚ :=
  3 * schedule.firstThreeHours + 3 * schedule.nextThreeHours + schedule.lastHour +
  2 * schedule.snackBreaks + schedule.lunchBreak

theorem study_days_needed :
  let math := SubjectRequirements.mk 4 7 (5/2) (3/2)
  let physics := SubjectRequirements.mk 5 9 3 2
  let chemistry := SubjectRequirements.mk 6 8 (7/2) (7/4)
  let breakSchedule := BreakSchedule.mk (1/6) (1/4) (1/3) (1/3) (3/4)
  let totalStudyHours := totalStudyTime math + totalStudyTime physics + totalStudyTime chemistry
  let effectiveStudyHoursPerDay := 7 - totalBreakTime breakSchedule
  ⌈totalStudyHours / effectiveStudyHoursPerDay⌉ = 23 := by
  sorry

end study_days_needed_l1173_117345


namespace gym_cost_comparison_l1173_117365

/-- Represents the cost of gym sessions under two different schemes -/
def gym_cost (x : ℕ) : ℝ × ℝ :=
  let y₁ := 12 * x + 40  -- Scheme 1: 40% discount + membership
  let y₂ := 16 * x       -- Scheme 2: 20% discount, no membership
  (y₁, y₂)

/-- Theorem stating which scheme is cheaper based on the number of sessions -/
theorem gym_cost_comparison (x : ℕ) (h : 5 ≤ x ∧ x ≤ 20) :
  let (y₁, y₂) := gym_cost x
  (x < 10 → y₂ < y₁) ∧
  (x = 10 → y₁ = y₂) ∧
  (10 < x → y₁ < y₂) :=
by sorry

end gym_cost_comparison_l1173_117365


namespace line_parallel_plane_necessary_not_sufficient_l1173_117355

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (planeParallelPlane : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_necessary_not_sufficient
  (α β : Plane) (l : Line) (h : subset l α) :
  (lineParallelPlane l β → planeParallelPlane α β) ∧
  ¬(planeParallelPlane α β → lineParallelPlane l β) :=
sorry

end line_parallel_plane_necessary_not_sufficient_l1173_117355


namespace order_of_6_undefined_l1173_117377

def f (x : ℤ) : ℤ := x^2 % 13

def f_iter (n : ℕ) (x : ℤ) : ℤ := 
  match n with
  | 0 => x
  | n+1 => f (f_iter n x)

theorem order_of_6_undefined : ¬ ∃ m : ℕ, m > 0 ∧ f_iter m 6 = 6 := by
  sorry

end order_of_6_undefined_l1173_117377


namespace optimal_sampling_for_populations_l1173_117386

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a population with its characteristics -/
structure Population where
  total : ℕ
  subgroups : List ℕ
  has_distinct_subgroups : Bool

/-- Determines the optimal sampling method for a given population -/
def optimal_sampling_method (pop : Population) : SamplingMethod :=
  if pop.has_distinct_subgroups then
    SamplingMethod.Stratified
  else
    SamplingMethod.Random

/-- The main theorem stating the optimal sampling methods for given populations -/
theorem optimal_sampling_for_populations 
  (pop1 : Population) 
  (pop2 : Population) 
  (h1 : pop1.has_distinct_subgroups = true) 
  (h2 : pop2.has_distinct_subgroups = false) :
  (optimal_sampling_method pop1 = SamplingMethod.Stratified) ∧
  (optimal_sampling_method pop2 = SamplingMethod.Random) := by
  sorry

#check optimal_sampling_for_populations

end optimal_sampling_for_populations_l1173_117386


namespace sum_of_squares_of_roots_l1173_117398

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) → 
  (10 * x₂^2 + 15 * x₂ - 20 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
  sorry

end sum_of_squares_of_roots_l1173_117398


namespace function_value_at_three_l1173_117338

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(3) = 11 -/
theorem function_value_at_three (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : f 1 = 5)
    (h2 : ∀ x, f x = a * x + b * x + 2) : 
  f 3 = 11 := by
sorry

end function_value_at_three_l1173_117338


namespace probability_same_color_is_two_fifths_l1173_117383

/-- Represents the number of white balls in the box -/
def num_white_balls : ℕ := 3

/-- Represents the number of black balls in the box -/
def num_black_balls : ℕ := 2

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- Calculates the number of ways to choose 2 balls from the total number of balls -/
def total_ways_to_choose : ℕ := total_balls.choose 2

/-- Calculates the number of ways to choose 2 white balls -/
def ways_to_choose_white : ℕ := num_white_balls.choose 2

/-- Calculates the number of ways to choose 2 black balls -/
def ways_to_choose_black : ℕ := num_black_balls.choose 2

/-- Calculates the total number of ways to choose 2 balls of the same color -/
def same_color_ways : ℕ := ways_to_choose_white + ways_to_choose_black

/-- The probability of drawing two balls of the same color -/
def probability_same_color : ℚ := same_color_ways / total_ways_to_choose

theorem probability_same_color_is_two_fifths :
  probability_same_color = 2 / 5 := by sorry

end probability_same_color_is_two_fifths_l1173_117383


namespace circles_externally_tangent_l1173_117354

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define what it means for two circles to be externally tangent
def externally_tangent (C₁ C₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), (C₁ x' y' ∧ C₂ x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_externally_tangent : externally_tangent C₁ C₂ :=
sorry

end circles_externally_tangent_l1173_117354


namespace prob_three_unused_correct_expected_hits_nine_targets_correct_l1173_117359

/-- Rocket artillery system model -/
structure RocketSystem where
  total_rockets : ℕ
  hit_probability : ℝ

/-- Probability of exactly three unused rockets after firing at five targets -/
def prob_three_unused (system : RocketSystem) : ℝ :=
  10 * system.hit_probability^3 * (1 - system.hit_probability)^2

/-- Expected number of targets hit when firing at nine targets -/
def expected_hits_nine_targets (system : RocketSystem) : ℝ :=
  10 * system.hit_probability - system.hit_probability^10

/-- Theorem: Probability of exactly three unused rockets after firing at five targets -/
theorem prob_three_unused_correct (system : RocketSystem) :
  prob_three_unused system = 10 * system.hit_probability^3 * (1 - system.hit_probability)^2 := by
  sorry

/-- Theorem: Expected number of targets hit when firing at nine targets -/
theorem expected_hits_nine_targets_correct (system : RocketSystem) :
  expected_hits_nine_targets system = 10 * system.hit_probability - system.hit_probability^10 := by
  sorry

end prob_three_unused_correct_expected_hits_nine_targets_correct_l1173_117359


namespace special_function_result_l1173_117373

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_result (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 3 - f 4) / f 2 = -7/4 := by
  sorry

end special_function_result_l1173_117373


namespace parallel_lines_k_value_l1173_117392

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end parallel_lines_k_value_l1173_117392
