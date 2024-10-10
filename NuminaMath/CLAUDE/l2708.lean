import Mathlib

namespace percentage_problem_l2708_270882

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.3 * x = 120 := by
  sorry

end percentage_problem_l2708_270882


namespace circle_equation_solution_l2708_270801

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3 ∧ x = 40/3 ∧ y = 41/3 := by
  sorry

end circle_equation_solution_l2708_270801


namespace expected_faces_six_rolls_l2708_270870

/-- The number of sides on a fair die -/
def n : ℕ := 6

/-- The number of times the die is rolled -/
def k : ℕ := 6

/-- The probability that a specific face does not appear in a single roll -/
def p : ℚ := (n - 1) / n

/-- The expected number of different faces that appear when rolling a fair n-sided die k times -/
def expected_different_faces : ℚ := n * (1 - p^k)

/-- Theorem stating that the expected number of different faces that appear when 
    rolling a fair 6-sided die 6 times is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_faces_six_rolls : 
  expected_different_faces = (n^k - (n-1)^k) / n^(k-1) :=
sorry

end expected_faces_six_rolls_l2708_270870


namespace smallest_positive_integer_modulo_l2708_270859

theorem smallest_positive_integer_modulo (y : ℕ) : y = 14 ↔ 
  (y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ ∀ z : ℕ, z > 0 → (z + 3050) % 15 = 1234 % 15 → y ≤ z) :=
by sorry

end smallest_positive_integer_modulo_l2708_270859


namespace pizza_area_increase_l2708_270842

/-- Theorem: Percent increase in pizza area
    If the radius of a large pizza is 60% larger than that of a medium pizza,
    then the percent increase in area between a medium and a large pizza is 156%. -/
theorem pizza_area_increase (r : ℝ) (h : r > 0) : 
  let large_radius := 1.6 * r
  let medium_area := π * r^2
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area * 100 = 156 :=
by
  sorry

#check pizza_area_increase

end pizza_area_increase_l2708_270842


namespace sin_2023pi_over_6_l2708_270819

theorem sin_2023pi_over_6 : Real.sin (2023 * Real.pi / 6) = -(1 / 2) := by
  sorry

end sin_2023pi_over_6_l2708_270819


namespace brownie_cutting_l2708_270823

theorem brownie_cutting (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) : 
  pan_length * pan_width - (pan_length * pan_width / (piece_length * piece_width)) * (piece_length * piece_width) = 0 :=
by sorry

end brownie_cutting_l2708_270823


namespace kennedy_gas_consumption_l2708_270864

-- Define the problem parameters
def miles_per_gallon : ℝ := 19
def distance_to_school : ℝ := 15
def distance_to_softball : ℝ := 6
def distance_to_restaurant : ℝ := 2
def distance_to_friend : ℝ := 4
def distance_to_home : ℝ := 11

-- Define the theorem
theorem kennedy_gas_consumption :
  let total_distance := distance_to_school + distance_to_softball + distance_to_restaurant + distance_to_friend + distance_to_home
  total_distance / miles_per_gallon = 2 := by
  sorry

end kennedy_gas_consumption_l2708_270864


namespace equation_solutions_l2708_270899

/-- The set of real solutions to the equation ∛(3 - x) + √(x - 2) = 1 -/
def solution_set : Set ℝ := {2, 3, 11}

/-- The equation ∛(3 - x) + √(x - 2) = 1 -/
def equation (x : ℝ) : Prop := Real.rpow (3 - x) (1/3) + Real.sqrt (x - 2) = 1

theorem equation_solutions :
  ∀ x : ℝ, x ∈ solution_set ↔ equation x ∧ x ≥ 2 := by sorry

end equation_solutions_l2708_270899


namespace general_position_lines_regions_l2708_270847

/-- 
A configuration of lines in general position.
-/
structure GeneralPositionLines where
  n : ℕ
  no_parallel : True  -- Represents the condition that no two lines are parallel
  no_concurrent : True -- Represents the condition that no three lines are concurrent

/-- 
The number of regions created by n lines in general position.
-/
def num_regions (lines : GeneralPositionLines) : ℕ :=
  1 + (lines.n * (lines.n + 1)) / 2

/-- 
Theorem: n lines in general position divide a plane into 1 + (1/2) * n * (n + 1) regions.
-/
theorem general_position_lines_regions (lines : GeneralPositionLines) :
  num_regions lines = 1 + (lines.n * (lines.n + 1)) / 2 := by
  sorry

end general_position_lines_regions_l2708_270847


namespace new_year_after_10_years_l2708_270894

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year in the 21st century -/
structure Year21stCentury where
  year : Nat
  is_21st_century : 2001 ≤ year ∧ year ≤ 2100

/-- Function to determine if a year is a leap year -/
def isLeapYear (y : Year21stCentury) : Bool :=
  y.year % 4 = 0 && (y.year % 100 ≠ 0 || y.year % 400 = 0)

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem stating that New Year's Day 10 years after a Friday is a Thursday -/
theorem new_year_after_10_years 
  (start_year : Year21stCentury)
  (h1 : DayOfWeek.Friday = advanceDays DayOfWeek.Friday 0)  -- New Year's Day is Friday in start_year
  (h2 : ∀ d : DayOfWeek, (advanceDays d (5 * 365 + 2)) = d) -- All days occur equally often in 5 years
  : DayOfWeek.Thursday = advanceDays DayOfWeek.Friday (10 * 365 + 3) :=
by sorry


end new_year_after_10_years_l2708_270894


namespace arithmetic_calculations_l2708_270861

theorem arithmetic_calculations :
  (3.21 - 1.05 - 1.95 = 0.21) ∧
  (15 - (2.95 + 8.37) = 3.68) ∧
  (14.6 * 2 - 0.6 * 2 = 28) ∧
  (0.25 * 1.25 * 32 = 10) := by
  sorry

end arithmetic_calculations_l2708_270861


namespace no_real_solutions_l2708_270857

theorem no_real_solutions : 
  ∀ x : ℝ, (2 * x^2 - 3 * x + 5)^2 + 1 ≠ 1 := by
sorry

end no_real_solutions_l2708_270857


namespace equal_chord_lengths_l2708_270824

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the condition D^2 ≠ E^2 > 4F
def condition (D E F : ℝ) : Prop :=
  D^2 ≠ E^2 ∧ E^2 > 4*F

-- Theorem statement
theorem equal_chord_lengths (D E F : ℝ) 
  (h : condition D E F) : 
  ∃ (chord_x chord_y : ℝ), 
    (∀ (x y : ℝ), circle_equation x y D E F → 
      (x = chord_x/2 ∨ x = -chord_x/2) ∨ (y = chord_y/2 ∨ y = -chord_y/2)) ∧
    chord_x = chord_y :=
sorry

end equal_chord_lengths_l2708_270824


namespace f_unique_zero_g_max_increasing_param_l2708_270855

noncomputable section

def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

def g (a : ℝ) (x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem f_unique_zero :
  ∃! x : ℝ, x ≥ 1 ∧ f x = 0 :=
sorry

theorem g_max_increasing_param :
  (∃ (a : ℤ), ∀ x ≥ 1, Monotone (g a)) ∧
  (∀ a : ℤ, a > 6 → ∃ x ≥ 1, ¬Monotone (g a)) :=
sorry

end f_unique_zero_g_max_increasing_param_l2708_270855


namespace average_speed_calculation_l2708_270804

/-- Proves that given the average speed from y to x and the average speed for the whole journey,
    we can determine the average speed from x to y. -/
theorem average_speed_calculation (speed_y_to_x : ℝ) (speed_round_trip : ℝ) (speed_x_to_y : ℝ) :
  speed_y_to_x = 36 →
  speed_round_trip = 39.6 →
  speed_x_to_y = 44 :=
by sorry

end average_speed_calculation_l2708_270804


namespace product_of_polynomials_l2708_270874

theorem product_of_polynomials (g h : ℚ) : 
  (∀ x, (9*x^2 - 5*x + g) * (4*x^2 + h*x - 12) = 36*x^4 - 41*x^3 + 7*x^2 + 13*x - 72) →
  g + h = -11/6 := by sorry

end product_of_polynomials_l2708_270874


namespace integral_condition_implies_b_value_l2708_270858

open MeasureTheory Measure Set Real
open intervalIntegral

theorem integral_condition_implies_b_value (b : ℝ) :
  (∫ x in (-1)..0, (2 * x + b)) = 2 →
  b = 3 := by
  sorry

end integral_condition_implies_b_value_l2708_270858


namespace goods_train_speed_l2708_270821

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) 
  (h1 : man_train_speed = 20) 
  (h2 : passing_time = 9) 
  (h3 : goods_train_length = 0.28) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 92 := by
  sorry

end goods_train_speed_l2708_270821


namespace trisha_total_distance_l2708_270810

/-- The total distance Trisha walked during her vacation in New York City -/
def total_distance (hotel_to_postcard postcard_to_tshirt tshirt_to_hotel : ℝ) : ℝ :=
  hotel_to_postcard + postcard_to_tshirt + tshirt_to_hotel

/-- Theorem stating that Trisha's total walking distance is 0.89 miles -/
theorem trisha_total_distance :
  total_distance 0.11 0.11 0.67 = 0.89 := by sorry

end trisha_total_distance_l2708_270810


namespace basketballs_in_boxes_l2708_270845

theorem basketballs_in_boxes 
  (total_basketballs : ℕ) 
  (basketballs_per_bag : ℕ) 
  (bags_per_box : ℕ) 
  (h1 : total_basketballs = 720) 
  (h2 : basketballs_per_bag = 8) 
  (h3 : bags_per_box = 6) : 
  (total_basketballs / (basketballs_per_bag * bags_per_box)) = 15 := by
  sorry

end basketballs_in_boxes_l2708_270845


namespace days_worked_together_is_two_l2708_270890

-- Define the efficiencies and time ratios
def efficiency_ratio_A_C : ℚ := 5 / 3
def time_ratio_B_C : ℚ := 2 / 3

-- Define the difference in days between A and C
def days_difference_A_C : ℕ := 6

-- Define the time A took to finish the remaining work
def remaining_work_days_A : ℕ := 6

-- Function to calculate the number of days B and C worked together
def days_worked_together (efficiency_ratio_A_C : ℚ) (time_ratio_B_C : ℚ) 
                         (days_difference_A_C : ℕ) (remaining_work_days_A : ℕ) : ℚ := 
  sorry

-- Theorem statement
theorem days_worked_together_is_two :
  days_worked_together efficiency_ratio_A_C time_ratio_B_C days_difference_A_C remaining_work_days_A = 2 := by
  sorry

end days_worked_together_is_two_l2708_270890


namespace putnam_inequality_l2708_270829

theorem putnam_inequality (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2 < 0 :=
by sorry

end putnam_inequality_l2708_270829


namespace square_plus_reciprocal_square_l2708_270848

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 7) : a^2 + 1/a^2 = 47 := by
  sorry

end square_plus_reciprocal_square_l2708_270848


namespace sum_of_squares_equality_l2708_270860

theorem sum_of_squares_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 + b^2 + c^2 = a*b + b*c + c*a) :
  (a^2*b^2)/((a^2+b*c)*(b^2+a*c)) + (a^2*c^2)/((a^2+b*c)*(c^2+a*b)) + (b^2*c^2)/((b^2+a*c)*(c^2+a*b)) = 1 := by
  sorry

end sum_of_squares_equality_l2708_270860


namespace inequality_condition_l2708_270826

theorem inequality_condition (a b : ℝ) : 
  (a * |a + b| < |a| * (a + b)) ↔ (a < 0 ∧ b > -a) := by sorry

end inequality_condition_l2708_270826


namespace lassis_production_l2708_270867

/-- Given a ratio of lassis to fruit units, calculate the number of lassis that can be made from a given number of fruit units -/
def calculate_lassis (ratio_lassis ratio_fruits fruits : ℕ) : ℕ :=
  (ratio_lassis * fruits) / ratio_fruits

/-- Proof that 25 fruit units produce 75 lassis given the initial ratio -/
theorem lassis_production : calculate_lassis 15 5 25 = 75 := by
  sorry

end lassis_production_l2708_270867


namespace arithmetic_sequence_squares_l2708_270818

theorem arithmetic_sequence_squares (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_arithmetic : ∃ d : ℝ, b / (c + a) - a / (b + c) = d ∧ c / (a + b) - b / (c + a) = d) :
  ∃ d' : ℝ, b^2 - a^2 = d' ∧ c^2 - b^2 = d' :=
sorry

end arithmetic_sequence_squares_l2708_270818


namespace expression_simplification_l2708_270877

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 1 = 0) :
  ((a - 2) / (a^2 + 2*a) - (a - 1) / (a^2 + 4*a + 4)) / ((a - 4) / (a + 2)) = 1/3 :=
by sorry

end expression_simplification_l2708_270877


namespace geometric_sequence_property_l2708_270891

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 :=
sorry

end geometric_sequence_property_l2708_270891


namespace tablecloth_diameter_is_ten_l2708_270827

/-- The diameter of a circular tablecloth with a given radius --/
def tablecloth_diameter (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The diameter of a circular tablecloth with a radius of 5 feet is 10 feet --/
theorem tablecloth_diameter_is_ten :
  tablecloth_diameter 5 = 10 := by
  sorry

end tablecloth_diameter_is_ten_l2708_270827


namespace abs_sum_inequality_l2708_270886

theorem abs_sum_inequality (x b : ℝ) (hb : b > 0) :
  (|x - 2| + |x + 3| < b) ↔ (b > 5) := by
  sorry

end abs_sum_inequality_l2708_270886


namespace tangent_line_cubic_curve_l2708_270849

theorem tangent_line_cubic_curve (m : ℝ) : 
  (∃ x y : ℝ, y = 12 * x + m ∧ y = x^3 - 2 ∧ 12 = 3 * x^2) → 
  (m = -18 ∨ m = 14) := by
sorry

end tangent_line_cubic_curve_l2708_270849


namespace abs_negative_eight_l2708_270893

theorem abs_negative_eight : |(-8 : ℤ)| = 8 := by
  sorry

end abs_negative_eight_l2708_270893


namespace factorization_m_squared_minus_3m_l2708_270846

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m-3) := by
  sorry

end factorization_m_squared_minus_3m_l2708_270846


namespace rectangle_perimeter_equal_area_l2708_270836

theorem rectangle_perimeter_equal_area (x y : ℕ) : 
  x > 0 ∧ y > 0 → 2 * x + 2 * y = x * y → (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) := by
  sorry

#check rectangle_perimeter_equal_area

end rectangle_perimeter_equal_area_l2708_270836


namespace collinear_probability_in_5x5_grid_l2708_270888

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 dots out of 25 -/
def totalWays : ℕ := Nat.choose totalDots chosenDots

/-- The number of horizontal lines in the grid -/
def horizontalLines : ℕ := gridSize

/-- The number of vertical lines in the grid -/
def verticalLines : ℕ := gridSize

/-- The number of major diagonals in the grid -/
def majorDiagonals : ℕ := 2

/-- The total number of collinear sets of 4 dots -/
def collinearSets : ℕ := horizontalLines + verticalLines + majorDiagonals

/-- The probability of selecting four collinear dots -/
def collinearProbability : ℚ := collinearSets / totalWays

theorem collinear_probability_in_5x5_grid :
  collinearProbability = 6 / 6325 := by sorry

end collinear_probability_in_5x5_grid_l2708_270888


namespace shoe_box_problem_l2708_270813

theorem shoe_box_problem (num_pairs : ℕ) (prob : ℚ) (total_shoes : ℕ) : 
  num_pairs = 12 → 
  prob = 1 / 23 → 
  prob = num_pairs / (total_shoes.choose 2) → 
  total_shoes = 24 := by
sorry

end shoe_box_problem_l2708_270813


namespace division_problem_l2708_270833

theorem division_problem : (96 / 6) / 2 = 8 := by
  sorry

end division_problem_l2708_270833


namespace between_a_and_b_l2708_270898

theorem between_a_and_b (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a < b) :
  a < (|3*a + 2*b| / 5) ∧ (|3*a + 2*b| / 5) < b := by
  sorry

end between_a_and_b_l2708_270898


namespace factor_expression_l2708_270853

theorem factor_expression (x y : ℝ) : 286 * x^2 * y + 143 * x = 143 * x * (2 * x * y + 1) := by
  sorry

end factor_expression_l2708_270853


namespace ellipse_standard_equation_l2708_270808

/-- An ellipse with given major axis and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
inductive StandardEquation where
  | x_axis : StandardEquation
  | y_axis : StandardEquation

/-- Theorem: For an ellipse with major axis 8 and eccentricity 3/4, 
    its standard equation is either (x²/16) + (y²/7) = 1 or (x²/7) + (y²/16) = 1 -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.major_axis = 8) 
  (h2 : e.eccentricity = 3/4) :
  ∃ (eq : StandardEquation), 
    (eq = StandardEquation.x_axis → ∀ (x y : ℝ), x^2/16 + y^2/7 = 1) ∧ 
    (eq = StandardEquation.y_axis → ∀ (x y : ℝ), x^2/7 + y^2/16 = 1) :=
by sorry

end ellipse_standard_equation_l2708_270808


namespace fraction_sum_theorem_l2708_270837

theorem fraction_sum_theorem (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 4) 
  (h2 : a/x + b/y + c/z = 3) 
  (h3 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 + 6*(x*y*z)/(a*b*c) = 16 := by
  sorry

end fraction_sum_theorem_l2708_270837


namespace b2_a2_minus_a1_value_l2708_270814

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  a₂ - 4 = 4 - a₁ ∧ 1 - a₂ = a₂ - 4

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  4 / b₁ = b₂ / 4 ∧ b₂ / 4 = 1 / b₂ ∧ 1 / b₂ = b₃ / 1

theorem b2_a2_minus_a1_value (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  arithmetic_sequence a₁ a₂ → geometric_sequence b₁ b₂ b₃ →
  (b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6) :=
by sorry

end b2_a2_minus_a1_value_l2708_270814


namespace race_time_difference_l2708_270812

/-- Proves that given the speeds of A and B are in the ratio 3:4, and A takes 2 hours to reach the destination, A takes 30 minutes more than B to reach the destination. -/
theorem race_time_difference (speed_a speed_b : ℝ) (time_a : ℝ) : 
  speed_a / speed_b = 3 / 4 →
  time_a = 2 →
  (time_a - (speed_a * time_a / speed_b)) * 60 = 30 := by
  sorry

end race_time_difference_l2708_270812


namespace total_interest_calculation_l2708_270830

/-- Calculate the total interest after 10 years given the following conditions:
    1. The simple interest on the initial principal for 10 years is 400.
    2. The principal is trebled after 5 years. -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 400) 
  (h2 : P > 0) 
  (h3 : R > 0) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1000 := by
  sorry

#check total_interest_calculation

end total_interest_calculation_l2708_270830


namespace two_digit_number_interchange_l2708_270838

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 4 → 
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end two_digit_number_interchange_l2708_270838


namespace quadratic_roots_inequality_l2708_270896

/-- Given quadratic polynomials f(x) = x² + bx + c and g(x) = x² + px + q with roots m₁, m₂ and k₁, k₂ respectively,
    prove that f(k₁) + f(k₂) + g(m₁) + g(m₂) ≥ 0. -/
theorem quadratic_roots_inequality (b c p q m₁ m₂ k₁ k₂ : ℝ) :
  let f := fun x => x^2 + b*x + c
  let g := fun x => x^2 + p*x + q
  (f m₁ = 0) ∧ (f m₂ = 0) ∧ (g k₁ = 0) ∧ (g k₂ = 0) →
  f k₁ + f k₂ + g m₁ + g m₂ ≥ 0 := by
  sorry

end quadratic_roots_inequality_l2708_270896


namespace square_difference_equals_product_l2708_270835

theorem square_difference_equals_product (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 2/15) : 
  x^2 - y^2 = 16/225 := by
sorry

end square_difference_equals_product_l2708_270835


namespace cube_packing_percentage_l2708_270805

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculates the number of cubes that fit along a given dimension -/
def cubesFitAlongDimension (boxDim : ℕ) (cubeSide : ℕ) : ℕ :=
  boxDim / cubeSide

/-- Calculates the total number of cubes that fit in the box -/
def totalCubesFit (box : BoxDimensions) (cube : CubeDimensions) : ℕ :=
  (cubesFitAlongDimension box.length cube.side) *
  (cubesFitAlongDimension box.width cube.side) *
  (cubesFitAlongDimension box.height cube.side)

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : CubeDimensions) : ℕ :=
  cube.side * cube.side * cube.side

/-- Calculates the percentage of box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cube : CubeDimensions) : ℚ :=
  let totalCubes := totalCubesFit box cube
  let volumeOccupied := totalCubes * (cubeVolume cube)
  (volumeOccupied : ℚ) / (boxVolume box : ℚ) * 100

/-- Theorem stating that the percentage of volume occupied by 3-inch cubes
    in a 9x8x12 inch box is 75% -/
theorem cube_packing_percentage :
  let box := BoxDimensions.mk 9 8 12
  let cube := CubeDimensions.mk 3
  percentageOccupied box cube = 75 := by
  sorry

end cube_packing_percentage_l2708_270805


namespace working_partner_receives_8160_l2708_270809

/-- Calculates the money received by the working partner in a business partnership --/
def money_received_by_working_partner (a_investment : ℕ) (b_investment : ℕ) (management_fee_percent : ℕ) (total_profit : ℕ) : ℕ :=
  let management_fee := (management_fee_percent * total_profit) / 100
  let remaining_profit := total_profit - management_fee
  let total_investment := a_investment + b_investment
  let a_share := (a_investment * remaining_profit) / total_investment
  management_fee + a_share

/-- Theorem stating that under given conditions, the working partner receives 8160 rs --/
theorem working_partner_receives_8160 :
  money_received_by_working_partner 5000 1000 10 9600 = 8160 := by
  sorry

#eval money_received_by_working_partner 5000 1000 10 9600

end working_partner_receives_8160_l2708_270809


namespace cube_root_of_three_cubed_l2708_270856

theorem cube_root_of_three_cubed (b : ℝ) : b^3 = 3 → b = 3^(1/3) :=
by
  sorry

end cube_root_of_three_cubed_l2708_270856


namespace election_votes_l2708_270868

theorem election_votes (votes1 votes3 : ℕ) (winning_percentage : ℚ) 
  (h1 : votes1 = 1136)
  (h2 : votes3 = 11628)
  (h3 : winning_percentage = 55371428571428574 / 100000000000000000)
  (h4 : votes3 > votes1)
  (h5 : ↑votes3 = winning_percentage * ↑(votes1 + votes3 + votes2)) :
  ∃ votes2 : ℕ, votes2 = 8236 := by sorry

end election_votes_l2708_270868


namespace rearrange_segments_l2708_270866

theorem rearrange_segments (a b : ℕ) : 
  ∃ (f g : Fin 1961 → Fin 1961), 
    ∀ i : Fin 1961, ∃ k : ℕ, 
      (a + f i) + (b + g i) = k + i.val ∧ 
      k + 1960 = (a + f ⟨1960, by norm_num⟩) + (b + g ⟨1960, by norm_num⟩) := by
  sorry

end rearrange_segments_l2708_270866


namespace train_speed_problem_l2708_270892

/-- Given two trains starting from the same station, traveling along parallel tracks in the same direction,
    with one train traveling at 31 mph, and the distance between them after 8 hours being 160 miles,
    prove that the speed of the first train is 51 mph. -/
theorem train_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed for the first train
  (v - 31) * 8 = 160 → 
  v = 51 := by
  sorry

end train_speed_problem_l2708_270892


namespace price_quantity_change_cost_difference_l2708_270873

theorem price_quantity_change (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  P * Q * 1.1 * 0.9 = P * Q * 0.99 := by
sorry

theorem cost_difference (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  P * Q * 1.1 * 0.9 - P * Q = P * Q * (-0.01) := by
sorry

end price_quantity_change_cost_difference_l2708_270873


namespace qin_jiushao_count_for_specific_polynomial_l2708_270839

/-- The "Qin Jiushao" algorithm for polynomial evaluation -/
def qin_jiushao_eval (coeffs : List ℝ) (x : ℝ) : ℝ := sorry

/-- Counts the number of multiplications and additions in the "Qin Jiushao" algorithm -/
def qin_jiushao_count (coeffs : List ℝ) : (ℕ × ℕ) := sorry

theorem qin_jiushao_count_for_specific_polynomial :
  let coeffs := [5, 4, 3, 2, 1, 1]
  qin_jiushao_count coeffs = (5, 5) := by sorry

end qin_jiushao_count_for_specific_polynomial_l2708_270839


namespace conic_section_focal_distance_l2708_270872

theorem conic_section_focal_distance (a : ℝ) (h1 : a ≠ 0) :
  (∀ x y : ℝ, x^2 + a * y^2 + a^2 = 0 → 
    ∃ c : ℝ, c = 2 ∧ c^2 = a^2 - a) →
  a = (1 - Real.sqrt 17) / 2 := by
sorry

end conic_section_focal_distance_l2708_270872


namespace chuck_puppy_shot_cost_l2708_270865

/-- The total cost of shots for puppies --/
def total_shot_cost (num_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) : ℕ :=
  num_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot

/-- Theorem stating the total cost of shots for Chuck's puppies --/
theorem chuck_puppy_shot_cost :
  total_shot_cost 3 4 2 5 = 120 := by
  sorry

end chuck_puppy_shot_cost_l2708_270865


namespace circle_equation_coefficients_l2708_270817

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the coefficients of the general circle equation -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a given equation represents a specific circle -/
def represents_circle (eq : CircleEquation) (circle : Circle) : Prop :=
  ∀ (x y : ℝ), 
    x^2 + y^2 + 2*eq.a*x - eq.b*y + eq.c = 0 ↔ 
    (x - circle.center.1)^2 + (y - circle.center.2)^2 = circle.radius^2

/-- The main theorem to prove -/
theorem circle_equation_coefficients 
  (circle : Circle) 
  (h_center : circle.center = (2, 3)) 
  (h_radius : circle.radius = 3) :
  ∃ (eq : CircleEquation),
    represents_circle eq circle ∧ 
    eq.a = -2 ∧ 
    eq.b = 6 ∧ 
    eq.c = 4 := by
  sorry

end circle_equation_coefficients_l2708_270817


namespace sara_oranges_l2708_270851

/-- Given that Joan picked 37 oranges, 47 oranges were picked in total, 
    and Alyssa picked 30 pears, prove that Sara picked 10 oranges. -/
theorem sara_oranges (joan_oranges : ℕ) (total_oranges : ℕ) (alyssa_pears : ℕ) 
    (h1 : joan_oranges = 37)
    (h2 : total_oranges = 47)
    (h3 : alyssa_pears = 30) : 
  total_oranges - joan_oranges = 10 := by
  sorry

end sara_oranges_l2708_270851


namespace pascal_triangle_specific_element_l2708_270869

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in the row of Pascal's triangle -/
def row_elements : ℕ := 56

/-- The row number (0-indexed) in Pascal's triangle -/
def row_number : ℕ := row_elements - 1

/-- The position (0-indexed) of the number we're looking for in the row -/
def position : ℕ := 23

theorem pascal_triangle_specific_element : 
  binomial row_number position = 29248649430 := by sorry

end pascal_triangle_specific_element_l2708_270869


namespace tangent_circles_distance_l2708_270815

/-- The distance between the centers of two tangent circles with radii 1 and 7 is either 6 or 8. -/
theorem tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 1 → r₂ = 7 → (d = |r₁ - r₂| ∨ d = r₁ + r₂) → d = 6 ∨ d = 8 := by
  sorry

end tangent_circles_distance_l2708_270815


namespace remove_parentheses_l2708_270834

theorem remove_parentheses (x y z : ℝ) : -(x - (y - z)) = -x + y - z := by
  sorry

end remove_parentheses_l2708_270834


namespace square_root_problem_l2708_270802

theorem square_root_problem (a b : ℝ) 
  (h1 : 3^2 = a + 7)
  (h2 : 2^3 = 2*b + 2) :
  ∃ (x : ℝ), x^2 = 3*a + b ∧ (x = 3 ∨ x = -3) := by
  sorry

end square_root_problem_l2708_270802


namespace arithmetic_mean_after_removal_l2708_270820

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 50 →
  x ∈ S →
  y ∈ S →
  x = 45 →
  y = 55 →
  (S.sum id) / S.card = 38 →
  ((S.sum id - (x + y)) / (S.card - 2) : ℝ) = 37.5 := by
  sorry

end arithmetic_mean_after_removal_l2708_270820


namespace least_addition_for_divisibility_least_addition_to_3198_for_divisibility_by_8_l2708_270800

theorem least_addition_for_divisibility (n : Nat) (d : Nat) : ∃ (x : Nat), x < d ∧ (n + x) % d = 0 :=
by
  -- The proof would go here
  sorry

theorem least_addition_to_3198_for_divisibility_by_8 :
  ∃ (x : Nat), x < 8 ∧ (3198 + x) % 8 = 0 ∧ ∀ (y : Nat), y < x → (3198 + y) % 8 ≠ 0 :=
by
  -- The proof would go here
  sorry

end least_addition_for_divisibility_least_addition_to_3198_for_divisibility_by_8_l2708_270800


namespace hyperbola_focal_length_l2708_270840

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let focal_length := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ (f₁ f₂ : ℝ × ℝ), 
      f₁.1 = focal_length ∧ f₁.2 = 0 ∧
      f₂.1 = -focal_length ∧ f₂.2 = 0 ∧
      ∀ p : ℝ × ℝ, p.1^2/a^2 - p.2^2/b^2 = 1 → 
        Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
        Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 2*a :=
by
  sorry


end hyperbola_focal_length_l2708_270840


namespace derivative_f_l2708_270880

noncomputable def f (x : ℝ) : ℝ := (Real.sinh x) / (2 * (Real.cosh x)^2) + (1/2) * Real.arctan (Real.sinh x)

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (Real.cosh x)^3 := by sorry

end derivative_f_l2708_270880


namespace car_speed_is_60_l2708_270885

/-- Represents the scenario of two friends traveling to a hunting base -/
structure HuntingTrip where
  walker_distance : ℝ  -- Distance of walker from base
  car_distance : ℝ     -- Distance of car owner from base
  total_time : ℝ       -- Total time to reach the base
  early_start : ℝ      -- Time walker would start earlier in alternative scenario
  early_meet : ℝ       -- Distance from walker's home where they'd meet in alternative scenario

/-- Calculates the speed of the car given the hunting trip scenario -/
def calculate_car_speed (trip : HuntingTrip) : ℝ :=
  60  -- Placeholder for the actual calculation

/-- Theorem stating that the car speed is 60 km/h given the specific scenario -/
theorem car_speed_is_60 (trip : HuntingTrip) 
  (h1 : trip.walker_distance = 46)
  (h2 : trip.car_distance = 30)
  (h3 : trip.total_time = 1)
  (h4 : trip.early_start = 8/3)
  (h5 : trip.early_meet = 11) :
  calculate_car_speed trip = 60 := by
  sorry

#eval calculate_car_speed { 
  walker_distance := 46, 
  car_distance := 30, 
  total_time := 1, 
  early_start := 8/3, 
  early_meet := 11 
}

end car_speed_is_60_l2708_270885


namespace geometric_sequence_middle_term_l2708_270883

theorem geometric_sequence_middle_term (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 15 * r = b ∧ b * r = 1) → b = Real.sqrt 15 := by
  sorry

end geometric_sequence_middle_term_l2708_270883


namespace quadratic_real_roots_condition_l2708_270871

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2*x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) := by
  sorry

end quadratic_real_roots_condition_l2708_270871


namespace f_monotone_increasing_F_lower_bound_l2708_270863

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 5 * a^2

def F (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

-- Theorem 1: f is monotonically increasing when a ≤ 0
theorem f_monotone_increasing (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 :=
sorry

-- Theorem 2: F has a lower bound
theorem F_lower_bound (a : ℝ) (x : ℝ) :
  F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
sorry

end f_monotone_increasing_F_lower_bound_l2708_270863


namespace parabola_zeros_difference_l2708_270897

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x-coordinate on the quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents the zeros of a quadratic function -/
structure QuadraticZeros where
  m : ℝ
  n : ℝ
  h_order : m > n

theorem parabola_zeros_difference (f : QuadraticFunction) (zeros : QuadraticZeros) :
  f.eval 1 = -3 →
  f.eval 3 = 9 →
  f.eval zeros.m = 0 →
  f.eval zeros.n = 0 →
  zeros.m - zeros.n = 2 := by
  sorry


end parabola_zeros_difference_l2708_270897


namespace smallest_k_divisible_by_nine_l2708_270822

theorem smallest_k_divisible_by_nine (k : ℕ) : k = 2024 ↔ 
  k > 2019 ∧ 
  (∀ m : ℕ, m > 2019 ∧ m < k → ¬(9 ∣ (m * (m + 1) / 2))) ∧ 
  (9 ∣ (k * (k + 1) / 2)) :=
sorry

end smallest_k_divisible_by_nine_l2708_270822


namespace triangle_angle_measure_l2708_270881

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 88 →
  E = 4 * F + 20 →
  D + E + F = 180 →
  F = 14.4 := by
sorry

end triangle_angle_measure_l2708_270881


namespace initial_fish_count_l2708_270816

def days_in_three_weeks : ℕ := 21

def koi_added_per_day : ℕ := 2
def goldfish_added_per_day : ℕ := 5

def final_koi_count : ℕ := 227
def final_goldfish_count : ℕ := 200

def total_koi_added : ℕ := days_in_three_weeks * koi_added_per_day
def total_goldfish_added : ℕ := days_in_three_weeks * goldfish_added_per_day

def initial_koi_count : ℕ := final_koi_count - total_koi_added
def initial_goldfish_count : ℕ := final_goldfish_count - total_goldfish_added

theorem initial_fish_count :
  initial_koi_count + initial_goldfish_count = 280 := by
  sorry

end initial_fish_count_l2708_270816


namespace monthly_salary_proof_l2708_270832

/-- Proves that a person's monthly salary is 1000 Rs, given the conditions -/
theorem monthly_salary_proof (salary : ℝ) : salary = 1000 :=
  let initial_savings_rate : ℝ := 0.25
  let initial_expense_rate : ℝ := 1 - initial_savings_rate
  let expense_increase_rate : ℝ := 0.10
  let new_savings_amount : ℝ := 175

  have h1 : initial_savings_rate * salary = 
            salary - initial_expense_rate * salary := by sorry

  have h2 : new_savings_amount = 
            salary - (initial_expense_rate * salary * (1 + expense_increase_rate)) := by sorry

  sorry

end monthly_salary_proof_l2708_270832


namespace wreath_problem_l2708_270844

/-- Represents the number of flowers in a wreath -/
structure Wreath where
  dandelions : ℕ
  cornflowers : ℕ
  daisies : ℕ

/-- The problem statement -/
theorem wreath_problem (masha katya : Wreath) : 
  (masha.dandelions + masha.cornflowers + masha.daisies + 
   katya.dandelions + katya.cornflowers + katya.daisies = 70) →
  (masha.dandelions = (5 * (masha.dandelions + masha.cornflowers + masha.daisies)) / 9) →
  (katya.daisies = (7 * (katya.dandelions + katya.cornflowers + katya.daisies)) / 17) →
  (masha.dandelions = katya.dandelions) →
  (masha.daisies = katya.daisies) →
  (masha.cornflowers = 2 ∧ katya.cornflowers = 0) :=
by sorry

end wreath_problem_l2708_270844


namespace triangle_inequality_theorem_l2708_270895

/-- A predicate that determines if three positive real numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating the triangle inequality for forming a triangle -/
theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c ↔ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_theorem_l2708_270895


namespace part_I_part_II_l2708_270854

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := { x | 0 < 2*x + a ∧ 2*x + a ≤ 3 }
def B : Set ℝ := { x | -1/2 < x ∧ x < 2 }

-- Part I
theorem part_I : 
  (Set.univ \ B) ∪ (A 1) = { x | x ≤ 1 ∨ x ≥ 2 } := by sorry

-- Part II
theorem part_II : 
  ∀ a : ℝ, (A a) ∩ B = A a ↔ -1 < a ∧ a ≤ 1 := by sorry

end part_I_part_II_l2708_270854


namespace find_n_l2708_270825

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 := by sorry

end find_n_l2708_270825


namespace sequence_expression_l2708_270807

theorem sequence_expression (a : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 + Real.log n :=
by sorry

end sequence_expression_l2708_270807


namespace geese_migration_rate_ratio_l2708_270841

/-- Given a population of geese where 50% are male and 20% of migrating geese are male,
    the ratio of migration rates between male and female geese is 1:4. -/
theorem geese_migration_rate_ratio :
  ∀ (total_geese male_geese migrating_geese male_migrating : ℕ),
  male_geese = total_geese / 2 →
  male_migrating = migrating_geese / 5 →
  (male_migrating : ℚ) / male_geese = (migrating_geese - male_migrating : ℚ) / (total_geese - male_geese) / 4 :=
by sorry

end geese_migration_rate_ratio_l2708_270841


namespace prob_b_greater_a_l2708_270862

-- Define the sets for a and b
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

-- Define the event space
def Ω : Finset (ℕ × ℕ) := A.product B

-- Define the favorable event (b > a)
def E : Finset (ℕ × ℕ) := Ω.filter (fun p => p.2 > p.1)

-- Theorem statement
theorem prob_b_greater_a :
  (E.card : ℚ) / Ω.card = 1 / 5 := by sorry

end prob_b_greater_a_l2708_270862


namespace conference_handshakes_l2708_270850

theorem conference_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 := by
  sorry

end conference_handshakes_l2708_270850


namespace gcd_from_lcm_and_ratio_l2708_270889

theorem gcd_from_lcm_and_ratio (X Y : ℕ) (h_lcm : Nat.lcm X Y = 180) (h_ratio : 5 * X = 2 * Y) : 
  Nat.gcd X Y = 18 := by
  sorry

end gcd_from_lcm_and_ratio_l2708_270889


namespace min_reciprocal_sum_l2708_270875

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end min_reciprocal_sum_l2708_270875


namespace cookie_difference_theorem_l2708_270887

def combined_difference (a b c : ℕ) : ℕ :=
  (a.max b - a.min b) + (a.max c - a.min c) + (b.max c - b.min c)

theorem cookie_difference_theorem :
  combined_difference 129 140 167 = 76 := by
  sorry

end cookie_difference_theorem_l2708_270887


namespace hyperbola_condition_l2708_270843

theorem hyperbola_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 - k) + y^2 / (k - 1) = 1) ∧ 
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (k - 1) = 1 → 
    ∃ a b : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∨ (y^2 / a^2 - x^2 / b^2 = 1)) →
  k < 1 ∨ k > 2 :=
by sorry

end hyperbola_condition_l2708_270843


namespace naza_market_averages_l2708_270831

/-- Represents an electronic shop with TV sets and models -/
structure Shop where
  name : Char
  tv_sets : ℕ
  tv_models : ℕ

/-- The list of shops in the Naza market -/
def naza_shops : List Shop := [
  ⟨'A', 20, 3⟩,
  ⟨'B', 30, 4⟩,
  ⟨'C', 60, 5⟩,
  ⟨'D', 80, 6⟩,
  ⟨'E', 50, 2⟩,
  ⟨'F', 40, 4⟩,
  ⟨'G', 70, 3⟩
]

/-- The total number of shops -/
def total_shops : ℕ := naza_shops.length

/-- Calculates the average of a list of natural numbers -/
def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

/-- Theorem stating the average number of TV sets and models in Naza market shops -/
theorem naza_market_averages :
  average (naza_shops.map Shop.tv_sets) = 50 ∧
  average (naza_shops.map Shop.tv_models) = 27 / 7 := by
  sorry

end naza_market_averages_l2708_270831


namespace nancy_home_economics_marks_l2708_270876

/-- Represents the marks obtained in different subjects -/
structure Marks where
  american_literature : ℕ
  history : ℕ
  physical_education : ℕ
  art : ℕ
  home_economics : ℕ

/-- Calculates the average marks -/
def average (m : Marks) : ℚ :=
  (m.american_literature + m.history + m.physical_education + m.art + m.home_economics) / 5

theorem nancy_home_economics_marks :
  ∀ m : Marks,
    m.american_literature = 66 →
    m.history = 75 →
    m.physical_education = 68 →
    m.art = 89 →
    average m = 70 →
    m.home_economics = 52 := by
  sorry

end nancy_home_economics_marks_l2708_270876


namespace expanded_parallelepiped_volume_l2708_270878

/-- The volume of a set of points inside or within one unit of a rectangular parallelepiped -/
def volume_expanded_parallelepiped (a b c : ℝ) : ℝ :=
  (a + 2) * (b + 2) * (c + 2) - (a * b * c)

/-- Represents the condition that two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem expanded_parallelepiped_volume 
  (m n p : ℕ) 
  (h_positive : m > 0 ∧ n > 0 ∧ p > 0) 
  (h_coprime : coprime n p) 
  (h_volume : volume_expanded_parallelepiped 2 3 4 = (m + n * Real.pi) / p) :
  m + n + p = 262 := by
  sorry

end expanded_parallelepiped_volume_l2708_270878


namespace one_meeting_before_completion_l2708_270828

/-- Represents the number of meetings between two runners on a circular track. -/
def number_of_meetings (circumference : ℝ) (speed1 speed2 : ℝ) : ℕ :=
  sorry

/-- Theorem stating that under given conditions, the runners meet once before completing a lap. -/
theorem one_meeting_before_completion :
  let circumference : ℝ := 300
  let speed1 : ℝ := 7
  let speed2 : ℝ := 3
  number_of_meetings circumference speed1 speed2 = 1 := by
  sorry

end one_meeting_before_completion_l2708_270828


namespace workers_total_earning_l2708_270884

/-- Calculates the total earning of three workers given their daily wages and work days -/
def total_earning (daily_wage_a daily_wage_b daily_wage_c : ℚ) 
  (days_a days_b days_c : ℕ) : ℚ :=
  daily_wage_a * days_a + daily_wage_b * days_b + daily_wage_c * days_c

/-- The total earning of three workers with given conditions -/
theorem workers_total_earning : 
  ∃ (daily_wage_a daily_wage_b daily_wage_c : ℚ),
    -- Daily wages ratio is 3:4:5
    daily_wage_a / daily_wage_b = 3 / 4 ∧
    daily_wage_b / daily_wage_c = 4 / 5 ∧
    -- Daily wage of c is Rs. 115
    daily_wage_c = 115 ∧
    -- Total earning calculation
    total_earning daily_wage_a daily_wage_b daily_wage_c 6 9 4 = 1702 := by
  sorry

end workers_total_earning_l2708_270884


namespace triangle_equality_condition_l2708_270806

/-- In a triangle ABC, the sum of squares of its sides is equal to 4√3 times its area 
    if and only if the triangle is equilateral. -/
theorem triangle_equality_condition (a b c : ℝ) (Δ : ℝ) :
  (a > 0) → (b > 0) → (c > 0) → (Δ > 0) →
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * Δ) ↔ (a = b ∧ b = c) :=
sorry

end triangle_equality_condition_l2708_270806


namespace octagon_theorem_l2708_270879

def is_permutation (l : List ℕ) : Prop :=
  l.length = 8 ∧ l.toFinset = Finset.range 8

def cyclic_shift (l : List ℕ) (k : ℕ) : List ℕ :=
  (l.drop k ++ l.take k).take 8

def product_sum (l1 l2 : List ℕ) : ℕ :=
  List.sum (List.zipWith (· * ·) l1 l2)

theorem octagon_theorem (l1 l2 : List ℕ) (h1 : is_permutation l1) (h2 : is_permutation l2) :
  ∃ k, product_sum l1 (cyclic_shift l2 k) ≥ 162 := by
  sorry

end octagon_theorem_l2708_270879


namespace greatest_common_factor_48_180_240_l2708_270852

theorem greatest_common_factor_48_180_240 : Nat.gcd 48 (Nat.gcd 180 240) = 12 := by
  sorry

end greatest_common_factor_48_180_240_l2708_270852


namespace min_cards_36_4suits_l2708_270803

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- The minimum number of cards to draw to guarantee all suits are represented -/
def min_cards_to_draw (d : Deck) : ℕ :=
  (d.num_suits - 1) * d.cards_per_suit + 1

/-- Theorem stating the minimum number of cards to draw for a 36-card deck with 4 suits -/
theorem min_cards_36_4suits :
  ∃ (d : Deck), d.total_cards = 36 ∧ d.num_suits = 4 ∧ min_cards_to_draw d = 28 :=
sorry

end min_cards_36_4suits_l2708_270803


namespace second_quadrant_trig_identity_l2708_270811

theorem second_quadrant_trig_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π) : 
  (Real.sin α / Real.cos α) * Real.sqrt (1 / Real.sin α^2 - 1) = -1 := by
  sorry

end second_quadrant_trig_identity_l2708_270811
